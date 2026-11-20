# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM80 (Ampere) forward pass kernel — fully self-contained (no base-class inheritance).

from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils_basic
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import BaseDSL
from quack import copy_utils, layout_utils

from ..common import utils
from ..common.block_info import BlockInfo
from ..common.cute_dsl_utils import assume_tensor_aligned
from ..common.kernel_config import KernelConfig
from ..common.mask import AttentionMask
from ..common.named_barrier import NamedBarrierFwd
from ..common.pack_gqa import PackGQA
from ..common.seqlen_info import SeqlenInfoQK
from ..common.softmax import Softmax
from ..common.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from . import ampere_helpers as sm80_utils


class FlashAttentionForwardSm80:
    def __init__(
        self,
        config: KernelConfig,
        dtype,
        num_stages=1,
        num_threads=128,
        Q_in_regs=False,
        is_split_kv=False,
        has_beam_sparse=False,
    ):
        # Copy config values to self for DSL compatibility
        self.dtype = dtype
        self.head_dim_padded = config.head_dim_padded  # was tile_hdim
        self.head_dim_v_padded = config.head_dim_v_padded  # was tile_hdimv
        self.check_hdim_oob = config.check_hdim_oob
        self.check_hdim_v_oob = config.check_hdim_v_oob
        self.same_hdim_kv = config.same_hdim_kv
        self.qhead_per_kvhead = config.qhead_per_kvhead
        self.pack_gqa = config.pack_gqa
        self.tile_m = config.tile_m
        self.tile_n = config.tile_n
        # SM80-specific
        self.num_stages = num_stages
        self.num_threads = num_threads
        self.Q_in_regs = Q_in_regs
        self.is_split_kv = is_split_kv
        self.has_beam_sparse = has_beam_sparse
        self.is_causal = False
        self.is_local = False
        self.score_mod = None
        self.mask_mod = None
        self.qk_acc_dtype = Float32
        self.vec_size = 2
        self.arch = BaseDSL._get_dsl().get_arch_enum()

    # ///////////////////////////////////////////////////////////////////////////
    # Static methods
    # ///////////////////////////////////////////////////////////////////////////

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        Q_in_regs=False,
        smem_arch="sm_80",
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # Check if block size setting is out of shared memory capacity
        # Shared memory usage: Q tile + (K tile + V tile) where K and V use the same tile size
        smem_usage_Q = tile_m * head_dim * 2
        smem_usage_K = tile_n * head_dim * num_stages * 2
        smem_usage_V = tile_n * head_dim_v * num_stages * 2
        smem_usage_QV = (
            (smem_usage_Q + smem_usage_V)
            if not Q_in_regs
            else max(smem_usage_Q, smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_K
        smem_capacity = utils_basic.get_smem_capacity_in_bytes(smem_arch)
        if smem_usage > smem_capacity:
            return False
        # Check if twice the block size is divisible by the number of threads
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

    # ///////////////////////////////////////////////////////////////////////////
    # SM80-specific tile / MMA helpers
    # ///////////////////////////////////////////////////////////////////////////

    def _get_smem_layout_atom(self):
        sQ_layout_atom = sm80_utils.get_smem_layout_atom(
            self.dtype, self.head_dim_padded
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = sm80_utils.get_smem_layout_atom(
            self.dtype, self.head_dim_v_padded
        )
        if const_expr(self.o_dtype == self.dtype):
            sO_layout_atom = sV_layout_atom
        else:
            sO_layout_atom = sm80_utils.get_smem_layout_atom(
                self.o_dtype, self.head_dim_v_padded
            )
        sP_layout_atom = None
        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )

    def _get_tiled_mma(self):
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        # sO reuses sQ buffer; when o_dtype is fp32, sQ must be large enough.
        # Exception: split-KV writes fp32 O directly to GMEM (no SMEM staging),
        # so sQ doesn't need to accommodate fp32 sO.
        if const_expr(self.is_split_kv):
            sQ_alloc = cute.cosize(self.sQ_layout)
        else:
            sO_cosize_in_q_elems = (
                cute.cosize(self.sO_layout) * self.o_dtype.width // self.dtype.width
            )
            sQ_alloc = max(cute.cosize(self.sQ_layout), sO_cosize_in_q_elems)
        sQ_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, sQ_alloc], 1024]
        sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024
            ]
            for layout in (self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(sQ_alloc, cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQV], 1024
        ]

        @cute.struct
        class SharedStorageQKV:
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        @cute.struct
        class SharedStorageSharedQV:
            sQ: sQV_struct
            sK: sK_struct

        return (
            SharedStorageQKV
            if const_expr(not self.Q_in_regs)
            else SharedStorageSharedQV
        )

    # ///////////////////////////////////////////////////////////////////////////
    # Merged from Base: _setup_attributes
    # ///////////////////////////////////////////////////////////////////////////

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        ) = self._get_smem_layout_atom()
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self.tile_m, self.head_dim_padded),
            (0, 1),
        )
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom,
            (self.tile_n, self.head_dim_padded, self.num_stages),
            (0, 1, 2),
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            (self.tile_n, self.head_dim_v_padded, self.num_stages),
            (0, 1, 2),
        )
        self.sO_layout = cute.tile_to_shape(
            sO_layout_atom,
            (self.tile_m, self.head_dim_v_padded),
            (0, 1),
        )
        if const_expr(sP_layout_atom is not None):
            self.sP_layout = cute.tile_to_shape(
                sP_layout_atom,
                (self.tile_m, self.tile_n),
                (0, 1),
            )
        else:
            self.sP_layout = None

        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store (uses o_dtype for fp32 split-KV)
        async_copy_elems_O = universal_copy_bits // self.o_dtype.width
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.o_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQ_layout and tK_layout: thread layout for QK load
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert (
            self.num_Q_load_threads % tQK_shape_dim_1 == 0
        ), "num_threads must be divisible by tQK_shape_dim_1"
        assert (
            self.num_producer_threads % tQK_shape_dim_1 == 0
        ), "num_threads must be divisible by tQK_shape_dim_1"
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        tK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we load Q
        assert self.tile_m % tQ_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tV_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        # tO_layout: thread layout for O store (uses o_dtype width for fp32 split-KV)
        tO_shape_dim_1 = sO_layout_atom.outer.shape[1] // async_copy_elems_O
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we store O
        assert self.tile_m % tO_layout.shape[0] == 0

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = cute.make_layout((1, async_copy_elems_O))

        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(
            atom_async_copy, tQ_layout, vQKV_layout
        )
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(
            atom_async_copy, tK_layout, vQKV_layout
        )
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(
            atom_async_copy, tV_layout, vQKV_layout
        )
        # gmem_tiled_copy_O: tiled copy for O store
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

    # ///////////////////////////////////////////////////////////////////////////
    # Merged from Base: epilogue
    # ///////////////////////////////////////////////////////////////////////////

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32 = Int32(0),
    ):
        cO = cute.make_identity_tensor((self.tile_m, self.head_dim_v_padded))
        thr_mma = tiled_mma.get_slice(tidx)
        pack_gqa = PackGQA(
            self.tile_m,
            self.head_dim_v_padded,
            self.check_hdim_v_oob,
            self.qhead_per_kvhead,
        )

        # Write LSE from rmem -> gmem
        if const_expr(mLSE is not None):
            # Split-KV: mLSE after transpose is (W, Hq, B, splits) for 4D
            if const_expr(len(mLSE.shape) == 4):
                mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
            else:
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded_layout = cute.append(
                    gLSE.layout,
                    cute.make_layout((self.head_dim_v_padded,), stride=(0,)),
                )
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                taccOgLSE = layout_utils.reshape_acc_to_mn(
                    thr_mma.partition_C(gLSE_expanded)
                )
                assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
                taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
                t0accOcO = layout_utils.reshape_acc_to_mn(
                    thr_mma.get_slice(0).partition_C(cO)
                )
                # Only the thread corresponding to column 0 writes out the lse to gmem
                if taccOcO[0][1] == 0:
                    for m in cutlass.range(
                        cute.size(taccOgLSE.shape[1]), unroll_full=True
                    ):
                        if (
                            t0accOcO[m, 0][0]
                            < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                        ):
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa.store_LSE(
                    mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q
                )

        ragged = self.use_tma_O and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
        # Split-KV: mO after transpose is (W, D, Hq, B, splits) for 5D
        if const_expr(len(mO.shape) == 5):
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                None, None, head_idx, split_idx
            ]
        else:
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[
                None, None, head_idx
            ]

        if const_expr(self.is_split_kv):
            # Split-KV: write fp32 O directly from registers to GMEM (no SMEM staging).
            # This avoids the fp32 sO buffer that would exceed SMEM on SM120 (99KB).
            # Follows FA Hopper C++ epilogue_fwd.hpp direct GMEM path.
            acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
            thr_cO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
            num_rows = cute.size(acc_O_mn, mode=[0])
            num_cols = cute.size(acc_O_mn, mode=[1])
            gO = cute.local_tile(
                mO_cur, (self.tile_m, self.head_dim_v_padded), (m_block, 0)
            )
            thr_gO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gO))
            for r in cutlass.range(num_rows, unroll_full=True):
                if thr_cO[r, 0][0] < seqlen.seqlen_q - m_block * self.tile_m:
                    for c in cutlass.range(num_cols, unroll_full=True):
                        if thr_cO[r, c][1] < mO.shape[1]:
                            thr_gO[r, c] = acc_O_mn[r, c]
        else:
            # Non-split: stage through SMEM for wider vectorization
            rO = cute.make_fragment_like(acc_O, self.o_dtype)
            rO.store(acc_O.load().to(self.o_dtype))
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads,
            )
            smem_copy_atom_O = utils.get_smem_store_atom(
                self.arch.major * 10 + self.arch.minor, self.o_dtype
            )
            smem_thr_copy_O = cute.make_tiled_copy_C(
                smem_copy_atom_O, tiled_mma
            ).get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

            if const_expr(self.use_tma_O):
                cute.arch.fence_view_async_shared()
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
                )
                gO = cute.local_tile(
                    mO_cur, (self.tile_m, self.head_dim_v_padded), (m_block, 0)
                )
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
                )
                warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
                if warp_idx == 4:
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierFwd.Epilogue),
                        number_of_threads=self.num_epilogue_threads
                        + cute.arch.WARP_SIZE,
                    )
                    store_O()
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            else:
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_epilogue_threads,
                )
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                tOsO = gmem_thr_copy_O.partition_S(sO)
                tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
                cute.autovec_copy(tOsO, tOrO)
                if const_expr(not self.pack_gqa):
                    gO = cute.local_tile(
                        mO_cur, (self.tile_m, self.head_dim_v_padded), (m_block, 0)
                    )
                    tOgO = gmem_thr_copy_O.partition_D(gO)
                    tOcO = gmem_thr_copy_O.partition_S(cO)
                    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                    tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                    for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                        if (
                            t0OcO[0, rest_m, 0][0]
                            < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]
                        ):
                            cute.copy(
                                gmem_tiled_copy_O,
                                tOrO[None, rest_m, None],
                                tOgO[None, rest_m, None],
                                pred=tOpO[None, rest_m, None]
                                if const_expr(self.check_hdim_v_oob)
                                else None,
                            )
                else:
                    pack_gqa.store_O(
                        mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q
                    )

    # ///////////////////////////////////////////////////////////////////////////
    # Merged from Base: advance_pipeline
    # ///////////////////////////////////////////////////////////////////////////

    @cute.jit
    def advance_pipeline(self, pipeline_index):
        return pipeline_index + 1 if pipeline_index < self.num_stages - 1 else 0

    # ///////////////////////////////////////////////////////////////////////////
    # Merged from Base: load_Q, load_K, load_V
    # ///////////////////////////////////////////////////////////////////////////

    @cute.jit
    def load_Q(
        self,
        gmem_thr_copy: cute.TiledCopy,
        gQ: cute.Tensor,
        sQ: cute.Tensor,
        block: Int32,
        seqlen: Int32,
        headdim: Int32,
    ):
        tQsQ, tQgQ = gmem_thr_copy.partition_D(sQ), gmem_thr_copy.partition_S(gQ)
        cQ = cute.make_identity_tensor((self.tile_m, self.head_dim_padded))
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=headdim)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            # Instead of using tQcQ, we using t0QcQ and subtract the offset from the limit
            # (seqlen - block * kBlockM). This is because the entries of t0QcQ are known at compile time.
            if t0QcQ[0, m, 0][0] < seqlen - block * self.tile_m - tQcQ[0][0]:
                cute.copy(
                    gmem_thr_copy,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None]
                    if const_expr(self.check_hdim_oob)
                    else None,
                )
            # We don't need to clear the sQ smem tiles since we'll only write out the valid outputs

    @cute.jit
    def load_K(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tKcK: cute.Tensor,
        t0KcK: cute.Tensor,
        tKpK: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        # Do we need to check if we overshoot kBlockN when we load K?
        is_even_n_smem_k = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_k):
            # Instead of using tKcK, we using t0KcK and subtract the offset from the limit
            # (seqlen - block * kBlockN). This is because the entries of t0KcK are known at compile time.
            if const_expr(is_even_n_smem_k):
                seqlen_limit = seqlen - block * self.tile_n
            else:
                if const_expr(not need_predicates):
                    seqlen_limit = self.tile_n
                else:
                    seqlen_limit = cutlass.min(
                        seqlen - block * self.tile_n, self.tile_n
                    )
            seqlen_limit -= tKcK[0][0]
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if t0KcK[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy,
                        tKgK[None, n, None, block],
                        tKsK[
                            None,
                            n,
                            None,
                            smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                        ],
                        pred=tKpK[None, n, None]
                        if const_expr(self.check_hdim_oob)
                        else None,
                    )
                # We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
        else:
            cute.copy(
                gmem_tiled_copy,
                tKgK[None, None, None, block],
                tKsK[
                    None,
                    None,
                    None,
                    smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                ],
                pred=tKpK if const_expr(self.check_hdim_oob) else None,
            )

    @cute.jit
    def load_V(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        tVcV: cute.Tensor,
        t0VcV: cute.Tensor,
        tVpV: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        # Do we need to check if we overshoot kBlockN when we load V?
        is_even_n_smem_v = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_v):
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
                if (
                    is_even_n_smem_v
                    or n < cute.size(tVsV.shape[1]) - 1
                    or tVcV[0, n, 0][0] < self.tile_n
                ):
                    predicate = (
                        tVpV[None, n, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None
                    )
                    if const_expr(need_predicates):
                        seqlen_limit = seqlen - block * self.tile_n - tVcV[0][0]
                        predicate_n = t0VcV[0, n, 0][0] < seqlen_limit
                        predicate = cute.make_fragment_like(tVpV[None, 0, None])
                        for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                            for i in cutlass.range_constexpr(
                                cute.size(predicate.shape[0])
                            ):
                                predicate[i, k] = (
                                    tVpV[i, n, k]
                                    if const_expr(self.check_hdim_v_oob)
                                    else True
                                ) and predicate_n
                    cute.copy(
                        gmem_tiled_copy,
                        tVgV[None, n, None, block],
                        tVsV[
                            None,
                            n,
                            None,
                            smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                        ],
                        pred=predicate,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tVgV[None, None, None, block],
                tVsV[
                    None,
                    None,
                    None,
                    smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                ],
                pred=tVpV if const_expr(self.check_hdim_v_oob) else None,
            )

    # ///////////////////////////////////////////////////////////////////////////
    # __call__ — kernel launch
    # ///////////////////////////////////////////////////////////////////////////

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        # Beam sparse fusion parameters (only used when has_beam_sparse=True)
        mQ_beam: Optional[cute.Tensor] = None,  # unused on SM80 (sQ is affine)
        mK_beam: Optional[cute.Tensor] = None,
        mV_beam: Optional[cute.Tensor] = None,
        mTopkIdxs: Optional[cute.Tensor] = None,
        beam_width: Int32 = Int32(0),
        decode_nums: Int32 = Int32(0),
        num_splits: Int32 = Int32(1),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        KernelConfig.check_type(
            *(
                t.element_type if t is not None else None
                for t in (
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                )
            )
        )
        self.o_dtype = mO.element_type  # Float32 for split-KV, else same as self.dtype
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        # self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None
        self.use_tma_O = False  # SM80 kernel always uses non-TMA O store
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # Layout permutation: 4D non-varlen vs 3D varlen
        QO_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mQ = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=QO_layout_transpose)
        )
        # Split-KV with num_splits>1: mO is 5D (splits,B,W,Hq,D) → (W,D,Hq,B,splits)
        if const_expr(len(mO.shape) == 5):
            mO = cute.make_tensor(
                mO.iterator,
                cute.select(
                    mO.layout,
                    mode=[2, 4, 3, 1, 0]
                    if const_expr(mCuSeqlensQ is None)
                    else [1, 3, 2, 0],
                ),
            )
        else:
            mO = cute.make_tensor(
                mO.iterator, cute.select(mO.layout, mode=QO_layout_transpose)
            )
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        # Transpose beam K/V for beam sparse phase
        if const_expr(mK_beam is not None):
            mK_beam = cute.make_tensor(
                mK_beam.iterator, cute.select(mK_beam.layout, mode=KV_layout_transpose)
            )
            mV_beam = cute.make_tensor(
                mV_beam.iterator, cute.select(mV_beam.layout, mode=KV_layout_transpose)
            )
        if const_expr(mLSE is not None):
            # Split-KV: mLSE is 4D (splits,B,Hq,W) → (W,Hq,B,splits)
            if const_expr(len(mLSE.shape) == 4):
                mLSE = cute.make_tensor(
                    mLSE.iterator,
                    cute.select(
                        mLSE.layout,
                        mode=[3, 2, 1, 0]
                        if const_expr(mCuSeqlensQ is None)
                        else [2, 1, 0],
                    ),
                )
            else:
                LSE_layout_transpose = (
                    [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
                )
                mLSE = cute.make_tensor(
                    mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
                )
        # TileScheduler for varlen, simple grid for non-varlen
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        num_batch = (
            mCuSeqlensQ.shape[0] - 1
            if const_expr(mCuSeqlensQ is not None)
            else mQ.shape[3]
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mQ.shape[0], self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=num_batch,
            num_splits=num_splits,
            seqlen_k=0,
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale
        )

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            softmax_scale_log2,
            softmax_scale,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            mQ_beam,
            mK_beam,
            mV_beam,
            mTopkIdxs,
            beam_width,
            decode_nums,
            num_splits,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    # ///////////////////////////////////////////////////////////////////////////
    # kernel — device code
    # ///////////////////////////////////////////////////////////////////////////

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        mQ_beam: Optional[cute.Tensor],  # unused on SM80
        mK_beam: Optional[cute.Tensor],
        mV_beam: Optional[cute.Tensor],
        mTopkIdxs: Optional[cute.Tensor],
        beam_width: Int32,
        decode_nums: Int32,
        num_splits: Int32,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, num_head, batch_size, split_idx = work_tile.tile_idx

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            is_split_kv=self.is_split_kv,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_size,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(
            seqlen, m_block, split_idx, num_splits
        )
        # For varlen, wasted grid tiles (where batch_idx >= num_batch) will have
        # seqlen_q=seqlen_k=0 and n_block_max=0.  Clamp to 0 so we don't use a
        # negative block index for K/V loads; the load/store predicates already
        # guard all memory accesses when seqlen is 0.
        n_block = cutlass.max(n_block_max - 1, 0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkQ_shape = (self.tile_m, self.head_dim_padded)
        blkK_shape = (self.tile_n, self.head_dim_padded)
        blkV_shape = (self.tile_n, self.head_dim_v_padded)
        num_head_kv = num_head // self.qhead_per_kvhead
        if const_expr(not seqlen.has_cu_seqlens_q):
            mQ_cur = mQ[None, None, num_head, batch_size]
        else:
            mQ_cur = cute.domain_offset((seqlen.offset_q, 0), mQ[None, None, num_head])
        if const_expr(not seqlen.has_cu_seqlens_k):
            mK_cur = mK[None, None, num_head_kv, batch_size]
            mV_cur = mV[None, None, num_head_kv, batch_size]
        else:
            mK_cur = cute.domain_offset(
                (seqlen.offset_k, 0), mK[None, None, num_head_kv]
            )
            mV_cur = cute.domain_offset(
                (seqlen.offset_k, 0), mV[None, None, num_head_kv]
            )
        gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
        gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
        gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout
            )
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)

        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C(
            (self.tile_m, self.head_dim_v_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.dtype,
        )
        smem_thr_copy_Q = utils.make_tiled_copy_A(
            smem_copy_atom_QK, tiled_mma_qk
        ).get_slice(tidx)
        smem_thr_copy_K = utils.make_tiled_copy_B(
            smem_copy_atom_QK, tiled_mma_qk
        ).get_slice(tidx)
        smem_thr_copy_V = utils.make_tiled_copy_B(
            smem_copy_atom_V, tiled_mma_pv
        ).get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for KV
        cK = cute.make_identity_tensor((self.tile_n, self.head_dim_padded))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
        if const_expr(self.head_dim_padded == self.head_dim_v_padded):
            tVcV = tKcK
            t0VcV = t0KcK
        else:
            cV = cute.make_identity_tensor((self.tile_n, self.head_dim_v_padded))
            tVcV = gmem_thr_copy_V.partition_S(cV)
            t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
        # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
        # use "if" on the mn dimension.
        # This is to reduce register pressure and gets 2-3% performance gain.
        tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
        if const_expr(self.same_hdim_kv):
            tVpV = tKpK
        else:
            tVpV = utils.predicate_k(tVcV, limit=mV.shape[1])

        # shape: (atom_v_m * rest_m)
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()

        # group parameters for compute_one_n_block
        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk,
            thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            tSsQ=tSsQ,
            tSsK=tSsK,
            tOsVt=tOsVt,
        )
        load_K = partial(
            self.load_K,
            gmem_tiled_copy_K,
            tKgK,
            tKsK,
            tKcK,
            t0KcK,
            tKpK,
            seqlen=seqlen.seqlen_k,
        )
        load_V = partial(
            self.load_V,
            gmem_tiled_copy_V,
            tVgV,
            tVsV,
            tVcV,
            t0VcV,
            tVpV,
            seqlen=seqlen.seqlen_k,
        )

        compute_one_n_block = partial(
            self.compute_one_n_block,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            load_K=load_K,
            load_V=load_V,
            batch_idx=batch_size,
            head_idx=num_head,
            m_block=m_block,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        self.load_Q(
            gmem_thr_copy_Q,
            gQ,
            sQ,
            m_block,
            seqlen=seqlen.seqlen_q,
            headdim=mQ.shape[1],
        )
        cute.arch.cp_async_commit_group()

        def preprocess_Q():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
            if const_expr(self.Q_in_regs):
                cute.arch.barrier()
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

        # If Q_in_regs, we load Q, then load 1 stage of K, then (optionally) rotate Q and
        # read from smem_q to registers, then load V.
        # If !Q_in_regs, we load Q, load all stages of K & V, then (optionally) rotate Q.
        if const_expr(self.Q_in_regs):
            load_K(n_block, smem_pipe_write=0, need_predicates=True)
            cute.arch.cp_async_commit_group()
            preprocess_Q()
            cute.arch.barrier()  # Make sure all threads have read smem_q before loading V

        for stage in cutlass.range_constexpr(self.num_stages):
            if const_expr(not self.Q_in_regs or stage > 0):
                if stage == 0 or n_block - stage >= 0:
                    load_K(
                        n_block - stage,
                        smem_pipe_write=stage,
                        need_predicates=stage == 0,
                    )
                cute.arch.cp_async_commit_group()
            if const_expr(stage < self.num_stages - 1):
                if stage == 0 or n_block - stage >= 0:
                    load_V(
                        n_block - stage,
                        smem_pipe_write=stage,
                        need_predicates=stage == 0,
                    )
                cute.arch.cp_async_commit_group()
        if const_expr(not self.Q_in_regs):
            preprocess_Q()

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop
        # ///////////////////////////////////////////////////////////////////////////////
        # Start processing of the first n-block.
        # For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of tile_n.
        # We also need masking on S if it's causal, for the last several blocks.
        mask = AttentionMask(
            self.tile_m,
            self.tile_n,
            seqlen,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        mask_fn = partial(
            mask.apply_mask,
            batch_idx=batch_size,
            head_idx=num_head,
            m_block=m_block,
            thr_mma=thr_mma_qk,
        )

        # Guard: skip mainloop + beam phase for empty splits
        if n_block_max > n_block_min:
            # First iteration with seqlen masking (starts at n_block_max - 1)
            smem_pipe_read = Int32(0)
            smem_pipe_write = Int32(self.num_stages - 1)
            compute_one_n_block(
                n_block,
                smem_pipe_read,
                smem_pipe_write,
                is_first_n_block=True,
                seqlen=seqlen,
                mask_fn=partial(mask_fn, mask_seqlen=True),
            )
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)
            # Remaining iterations: from n_block-1 down to n_block_min (inclusive)
            num_remaining = n_block - n_block_min
            for n_tile in cutlass.range(num_remaining, unroll=1):
                compute_one_n_block(
                    n_block - n_tile - 1,
                    smem_pipe_read,
                    smem_pipe_write,
                    seqlen=seqlen,
                    is_first_n_block=False,
                    mask_fn=partial(mask_fn, mask_seqlen=False),
                )
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)
            # =====================================================================
            # Phase 2: Beam Sparse (CUDA core FMA)
            # =====================================================================
            if const_expr(self.has_beam_sparse):
                if split_idx == num_splits - 1:
                    self._beam_sparse_phase(
                        acc_O,
                        softmax,
                        tiled_mma_pv,
                        tidx,
                        sQ,
                        mK_beam,
                        mV_beam,
                        mTopkIdxs,
                        beam_width,
                        decode_nums,
                        batch_size,
                        m_block,
                        num_head,
                        softmax_scale_log2,
                    )

            # normalize acc_O by row_sum and calculate the lse
            row_scale = softmax.finalize()
            softmax.rescale_O(acc_O, row_scale)

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue — skip for empty splits
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
            # reuse sQ's data iterator (o_dtype may be fp32 for split-KV)
            sO = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, dtype=self.o_dtype), sO_layout
            )
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                None,
                tiled_mma_pv,
                tidx,
                m_block,
                num_head,
                batch_size,
                split_idx,
            )

    # ///////////////////////////////////////////////////////////////////////////
    # Beam sparse phase (fused into context attention)
    # ///////////////////////////////////////////////////////////////////////////

    @cute.jit
    def _beam_sparse_phase(
        self,
        acc_O: cute.Tensor,
        softmax,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        sQ: cute.Tensor,
        mK_beam: cute.Tensor,
        mV_beam: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        beam_width: Int32,
        decode_nums: Int32,
        batch_idx: Int32,
        m_block: Int32,
        head_idx: Int32,
        softmax_scale_log2: Float32,
    ):
        """Fuse beam sparse KV into context attention's softmax state (SM80).

        Uses CUDA core FMA. Q read from sQ (affine layout, direct indexing).
        Same algorithm as SM90 _beam_sparse_phase.
        """
        import operator as _operator

        # MMA TV coordinate mapping
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        thr_mma = tiled_mma.get_slice(tidx)
        cO = cute.make_identity_tensor((self.tile_m, self.head_dim_v_padded))
        thr_cO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))

        num_rows = cute.size(acc_O_mn, mode=[0])
        num_cols = cute.size(acc_O_mn, mode=[1])

        row_max = softmax.row_max
        row_sum = softmax.row_sum

        kv_head_idx = head_idx // self.qhead_per_kvhead

        # 4 threads share each row; only t0==0 adds to row_sum
        t0 = tidx % 4
        is_row_sum_writer = t0 == 0

        # K/V beam: (s_k, d, h_k, b) after transpose
        gK_beam = mK_beam[None, None, kv_head_idx, batch_idx]
        gV_beam = mV_beam[None, None, kv_head_idx, batch_idx]

        for dn in cutlass.range(decode_nums):
            for r in cutlass.range(num_rows, unroll_full=True):
                row_in_tile = thr_cO[r, 0][0]
                beam_idx = m_block * self.tile_m + row_in_tile

                active = beam_idx < beam_width
                kv_idx = Int32(0)
                partial_score = Float32(0.0)
                if active:
                    kv_idx = mTopkIdxs[batch_idx, kv_head_idx, dn, beam_idx]

                    # QK: 4-thread parallel, Q from sQ (affine, direct access)
                    gK_row = gK_beam[kv_idx, None]
                    chunk = self.head_dim_padded // 4
                    k_start = t0 * chunk
                    for k in cutlass.range(chunk, unroll_full=True):
                        partial_score = partial_score + Float32(
                            sQ[row_in_tile, k_start + k]
                        ) * Float32(gK_row[k_start + k])

                # Keep warp-level reduction in uniform control flow. The last beam
                # fragment can be partial when beam_width is not a multiple of 8.
                score_raw = utils.warp_reduce(partial_score, _operator.add, width=4)

                if active:
                    # Online softmax (unscaled row_max)
                    m_prev = row_max[r]
                    if score_raw > m_prev:
                        row_max[r] = score_raw
                    m_new = row_max[r]
                    o_scale = cute.math.exp2(
                        (m_prev - m_new) * softmax_scale_log2, fastmath=True
                    )
                    p_val = cute.math.exp2(
                        (score_raw - m_new) * softmax_scale_log2, fastmath=True
                    )
                    row_sum[r] = row_sum[r] * o_scale
                    if is_row_sum_writer:
                        row_sum[r] = row_sum[r] + p_val

                    # PV accumulate
                    gV_row = gV_beam[kv_idx, None]
                    for c in cutlass.range(num_cols, unroll_full=True):
                        col = thr_cO[r, c][1]
                        acc_O_mn[r, c] = acc_O_mn[r, c] * o_scale + p_val * Float32(
                            gV_row[col]
                        )

    # ///////////////////////////////////////////////////////////////////////////
    # compute_one_n_block
    # ///////////////////////////////////////////////////////////////////////////

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: Int32,
        smem_pipe_write: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        seqlen: SeqlenInfoQK,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus
        subsequent blocks.
        """

        def sync():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 2)
            cute.arch.barrier()

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C(
            (self.tile_m, self.tile_n)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        # wait for smem tile QK before mma calculation for S
        sync()

        # need predicates for the first tile
        def load_V_next():
            if self.num_stages == 1 or n_block - self.num_stages + 1 >= 0:
                load_V(
                    n_block - self.num_stages + 1,
                    smem_pipe_write,
                    need_predicates=is_first_n_block and self.num_stages == 1,
                )
            cute.arch.cp_async_commit_group()

        load_V_next()
        sm80_utils.gemm(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[
                None,
                None,
                None,
                smem_pipe_read if const_expr(self.num_stages > 1) else 0,
            ],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
            # hook_fn=load_V_next,
            A_in_regs=self.Q_in_regs,
        )
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        def load_K_next():
            if n_block - self.num_stages >= 0:
                load_K(
                    n_block - self.num_stages, smem_pipe_write, need_predicates=False
                )
            cute.arch.cp_async_commit_group()

        # wait for smem tile V for O
        if const_expr(self.num_stages == 1):
            sync()
            load_K_next()
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        if const_expr(self.num_stages > 1):
            sync()
            load_K_next()
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            mma_params.tOrVt,
            smem_copy_params.tOsVt[
                None,
                None,
                None,
                smem_pipe_read if const_expr(self.num_stages > 1) else 0,
            ],
            smem_copy_params.smem_thr_copy_V,
            # hook_fn=load_K_next,
        )
        # if const_expr(self.num_stages > 1):
        #     load_K_next()
