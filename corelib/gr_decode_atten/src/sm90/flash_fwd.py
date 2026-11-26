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
# SM90 (Hopper) forward pass for flash attention, extracted from flash_fwd.py.

import operator
from functools import partial
from types import SimpleNamespace
from typing import Callable, Literal, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, const_expr, pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum
from quack import copy_utils, layout_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase

from ..common import pipeline as pipeline_custom
from ..common import utils
from ..common.block_info import BlockInfo
from ..common.cute_dsl_utils import assume_tensor_aligned
from ..common.kernel_config import KernelConfig
from ..common.mask import AttentionMask
from ..common.named_barrier import NamedBarrierFwd
from ..common.pack_gqa import PackGQA, make_packgqa_tiled_tma_atom, pack_gqa_layout
from ..common.paged_kv import PagedKVManager
from ..common.seqlen_info import SeqlenInfoQK
from ..common.softmax import Softmax
from ..common.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FlashAttentionForwardSm90:
    def __init__(
        self,
        config: KernelConfig,
        dtype,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        is_split_kv: bool = False,
        has_beam_sparse: bool = False,
    ):
        # Copy config to self for DSL compat
        self.dtype = dtype
        self.head_dim_padded = config.head_dim_padded
        self.head_dim_v_padded = config.head_dim_v_padded
        self.check_hdim_oob = config.check_hdim_oob
        self.check_hdim_v_oob = config.check_hdim_v_oob
        self.same_hdim_kv = config.same_hdim_kv
        self.qhead_per_kvhead = config.qhead_per_kvhead
        self.pack_gqa = config.pack_gqa
        self.tile_m = config.tile_m
        self.tile_n = config.tile_n
        # SM90-specific
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs
        self.is_causal = False
        self.is_local = False
        self.score_mod = None
        self.mask_mod = None
        self.qk_acc_dtype = Float32
        self.vec_size = 2
        self.Q_in_regs = False
        self.num_stages = 2  # SM90 default
        self.buffer_align_bytes = 1024
        self.is_split_kv = is_split_kv
        self.has_beam_sparse = has_beam_sparse
        self.use_tma_KV = not paged_kv_non_tma
        self.cluster_shape_mn = (1, 1)
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert (
            self.arch >= Arch.sm_90 and self.arch <= Arch.sm_90a
        ), "Only SM 9.x is supported"
        assert not paged_kv_non_tma or not (
            self.check_hdim_oob or self.check_hdim_v_oob
        ), "Paged KV does not support irregular head dim"

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
        # store acc_O (o_dtype: fp32 for split-KV, else bf16)
        rO = cute.make_fragment_like(acc_O, self.o_dtype)
        rO.store(acc_O.load().to(self.o_dtype))
        # Make sure all threads have finished reading V
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        smem_copy_atom_O = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.o_dtype
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(
            tidx
        )
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.head_dim_v_padded))
        pack_gqa = PackGQA(
            self.tile_m,
            self.head_dim_v_padded,
            self.check_hdim_v_oob,
            self.qhead_per_kvhead,
        )

        # Write LSE from rmem -> gmem
        if const_expr(mLSE is not None):
            if const_expr(len(mLSE.shape) == 4):
                # 5D split-KV: mLSE after transpose is (W, Hq, B, splits)
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
                thr_mma = tiled_mma.get_slice(tidx)
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
        if const_expr(len(mO.shape) == 5):
            # 5D split-KV: mO after transpose is (W, D, Hq, B, splits)
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                None, None, head_idx, split_idx
            ]
        else:
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[
                None, None, head_idx
            ]
        # sync to make sure all smem stores are done
        if const_expr(self.use_tma_O):
            # ensure smem writes are visible to TMA
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
                    number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
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
            # load acc O from smem to rmem for wider vectorization
            cute.autovec_copy(tOsO, tOrO)
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(
                    mO_cur, (self.tile_m, self.head_dim_v_padded), (m_block, 0)
                )
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                # copy acc O from rmem to gmem
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

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim_padded
            ),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim_v_padded
            ),
            self.dtype,
        )
        if const_expr(self.o_dtype == self.dtype):
            sO_layout_atom = sV_layout_atom
        else:
            sO_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.o_dtype, self.head_dim_v_padded
                ),
                self.o_dtype,
            )
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype, self.tile_n
                ),
                self.dtype,
            )
        else:
            sP_layout_atom = None
        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(
                self.tile_m // 64,
                1,
                1,
            ),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.head_dim_v_padded),
            a_source=warpgroup.OperandSource.RMEM
            if self.mma_pv_is_rs
            else warpgroup.OperandSource.SMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        # sO reuses sQ buffer; when o_dtype is fp32, sQ must be large enough
        sO_cosize_in_q_elems = (
            cute.cosize(self.sO_layout) * self.o_dtype.width // self.dtype.width
        )
        sQ_alloc = max(cute.cosize(self.sQ_layout), sO_cosize_in_q_elems)
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, sQ_alloc], self.buffer_align_bytes
        ]
        sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)],
                self.buffer_align_bytes,
            ]
            for layout in (self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(sQ_alloc, cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQV], 1024
        ]
        cosize_sP = (
            cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        )
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        return (
            SharedStorageQKV
            if const_expr(not self.Q_in_regs)
            else SharedStorageSharedQV
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        # Beam sparse fusion parameters (only used when has_beam_sparse=True)
        mQ_beam: Optional[
            cute.Tensor
        ] = None,  # (b, s_q, h, d) same as mQ, raw (non-transposed)
        mK_beam: Optional[cute.Tensor] = None,  # (b, seqlen_beam, h_kv, d)
        mV_beam: Optional[cute.Tensor] = None,  # (b, seqlen_beam, h_kv, d_v)
        mTopkIdxs: Optional[cute.Tensor] = None,  # (b, h_kv, decode_nums, beam_width)
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
        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mQ = layout_utils.select(mQ, QO_layout_transpose)
        # Split-KV with num_splits>1: mO is 5D (splits,B,W,Hq,D) → (W,D,Hq,B,splits)
        if const_expr(len(mO.shape) == 5):
            mO = layout_utils.select(
                mO, [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
        else:
            mO = layout_utils.select(mO, QO_layout_transpose)
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        # Transpose beam Q/K/V using cute.select (supports element-wise indexing)
        if const_expr(mQ_beam is not None):
            mQ_beam = cute.make_tensor(
                mQ_beam.iterator, cute.select(mQ_beam.layout, mode=QO_layout_transpose)
            )
            mK_beam = cute.make_tensor(
                mK_beam.iterator, cute.select(mK_beam.layout, mode=KV_layout_transpose)
            )
            mV_beam = cute.make_tensor(
                mV_beam.iterator, cute.select(mV_beam.layout, mode=KV_layout_transpose)
            )
        # LSE transpose: 4D split-KV (splits,B,Hq,W) vs 3D non-split (B,Hq,W)
        if const_expr(mLSE is not None):
            if const_expr(len(mLSE.shape) == 4):
                mLSE = layout_utils.select(
                    mLSE, [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
                )
            else:
                mLSE = layout_utils.select(
                    mLSE, [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
                )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {
            1: (256, 56),
            2: (240, 24),
            3: (160, 32),
        }[self.num_wg_mma]
        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.head_dim_padded <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        # Disable TMA for O when split-KV (TMA can't handle 5D output or split indexing)
        self.use_tma_O = self.use_tma_Q and not self.is_split_kv
        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(
            self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)
        ):
            self.num_mma_regs, self.num_producer_regs = 224, 40
        self.rescale_O_before_gemm = (
            self.head_dim_v_padded > 128 and self.intra_wg_overlap
        )
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(
                mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage
            )
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.head_dim_padded), None),
                (mK, (self.tile_n, self.head_dim_padded), self.num_stages),
                (mV, (self.tile_n, self.head_dim_v_padded), self.num_stages),
                (mO, (self.tile_m, self.head_dim_v_padded), None),
            ]
        ]
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )

        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        make_tiled_tma_atom_fn = (
            partial(
                make_packgqa_tiled_tma_atom,
                qhead_per_kvhead=self.qhead_per_kvhead,
                head_idx=2,
            )
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.head_dim_padded),  # No mcast
            )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.head_dim_padded),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.head_dim_v_padded),
                1,  # No mcast for now
            )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.head_dim_v_padded),  # No mcast
            )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=False,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale
        )
        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
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
            tile_sched_params,
            TileScheduler,
            SharedStorage,
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
            stream=stream,
            min_blocks_per_mp=1,
        )

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
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
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
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        mQ_beam: Optional[cute.Tensor],
        mK_beam: Optional[cute.Tensor],
        mV_beam: Optional[cute.Tensor],
        mTopkIdxs: Optional[cute.Tensor],
        beam_width: Int32,
        decode_nums: Int32,
        num_splits: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier / pipeline init
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        if const_expr(self.use_tma_KV):
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        # reuse sQ's data iterator (o_dtype may be fp32 for split-KV)
        sO = storage.sQ.get_tensor(
            sO_layout.outer, swizzle=sO_layout.inner, dtype=self.o_dtype
        )

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            is_split_kv=self.is_split_kv,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0]
            if const_expr(not self.pack_gqa)
            else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Cluster wait before starting
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_Q,
                mPageTable,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                num_splits,
            )

        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                mQ_beam,
                mK_beam,
                mV_beam,
                mTopkIdxs,
                beam_width,
                decode_nums,
                num_splits,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: Optional[cute.Tensor],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        num_splits: Int32 = Int32(1),
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()

        # TMA: only warp 0 loads. cp_async: all warps load.
        # When not use_tma_Q, all 128 producer threads participate in Q loading.
        is_load_warp = warp_idx_in_wg == 0 or const_expr(
            not self.use_tma_KV or not self.use_tma_Q
        )
        # KV loading restricted to warp 0 for TMA, all warps for non-TMA KV
        is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)

        if is_load_warp:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, split_idx_p = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead
                    if const_expr(not self.pack_gqa)
                    else head_idx
                )

                load_Q = None
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(
                        mQ_cur, (self.tile_m, self.head_dim_padded), (m_block, 0)
                    )
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )

                paged_kv_manager = None
                tma_load_K_fn = None
                tma_load_V_fn = None
                if const_expr(self.use_tma_KV):
                    # === TMA path (non-paged and paged with page_size == n_block_size) ===
                    if const_expr(mPageTable is not None):
                        # Paged TMA: keep page dimension indexable
                        mK_cur = mK[None, None, head_idx_kv, None]
                        mV_cur = mV[None, None, head_idx_kv, None]
                        gK = cute.local_tile(
                            mK_cur, (self.tile_n, self.head_dim_padded), (0, 0, None)
                        )
                        gV = cute.local_tile(
                            mV_cur, (self.tile_n, self.head_dim_v_padded), (0, 0, None)
                        )
                    else:
                        # Non-paged TMA
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        gK = cute.local_tile(
                            mK_cur, (self.tile_n, self.head_dim_padded), (None, 0)
                        )
                        gV = cute.local_tile(
                            mV_cur, (self.tile_n, self.head_dim_v_padded), (None, 0)
                        )
                    # TODO: mcast
                    tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gK, sK
                    )
                    tma_load_K_fn = copy_utils.tma_producer_copy_fn(
                        tma_load_K_fn, pipeline_k
                    )
                    tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V, 0, cute.make_layout(1), gV, sV
                    )
                    tma_load_V_fn = copy_utils.tma_producer_copy_fn(
                        tma_load_V_fn, pipeline_v
                    )
                else:
                    # === cp_async path (paged KV with page_size != n_block_size) ===
                    paged_kv_manager = PagedKVManager.create(
                        mPageTable,
                        mK,
                        mV,
                        FastDivmodDivisor(mK.shape[0]),
                        batch_idx,
                        head_idx_kv,
                        tidx,
                        seqlen.seqlen_k,
                        0,  # leftpad_k
                        self.tile_n,
                        self.head_dim_padded,
                        self.head_dim_v_padded,
                        self.num_threads_per_warp_group,
                        mK.element_type,
                        arch=self.arch.major * 10 + self.arch.minor,
                    )

                load_K = partial(
                    self.load_KV,
                    tma_load_K_fn,
                    paged_kv_manager,
                    sK,
                    pipeline_kv=pipeline_k,
                    K_or_V="K",
                )
                load_V = partial(
                    self.load_KV,
                    tma_load_V_fn,
                    paged_kv_manager,
                    sV,
                    pipeline_kv=pipeline_v,
                    K_or_V="V",
                )

                pack_gqa = None
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(
                        self.tile_m,
                        self.head_dim_padded,
                        self.check_hdim_oob,
                        self.qhead_per_kvhead,
                    )

                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx_p, num_splits
                )
                # Clamp n_block to 0 when n_block_max == 0 (can happen when
                # seqlen_k < tile_n). TMA handles n_block=-1
                # gracefully (fills zeros), but cp.async would crash on
                # out-of-bounds page table access.
                n_block = (
                    n_block_max - 1
                    if const_expr(self.use_tma_KV)
                    else cutlass.max(n_block_max - 1, 0)
                )
                page_idx = (
                    mPageTable[batch_idx, n_block]
                    if const_expr(mPageTable is not None and self.use_tma_KV)
                    else None
                )

                # First iteration: load K on pipeline_k, Q on pipeline_q
                if is_kv_load_warp:
                    pipeline_k.producer_acquire(kv_producer_state)
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(n_block)
                    load_K(
                        block=n_block,
                        producer_state=kv_producer_state,
                        page_idx=page_idx,
                    )
                if const_expr(self.use_tma_Q):
                    if warp_idx_in_wg == 0:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                        q_producer_phase ^= 1
                else:
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    pack_gqa.load_Q(
                        mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                    )
                    cute.arch.cp_async_commit_group()
                    pipeline_q.producer_commit_w_index(0)
                    q_producer_phase ^= 1

                if is_kv_load_warp:
                    if const_expr(not self.intra_wg_overlap or not self.use_tma_KV):
                        pipeline_v.producer_acquire(kv_producer_state)
                        load_V(
                            block=n_block,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )
                        kv_producer_state.advance()
                        for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - i - 1
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(
                                    mPageTable is not None and self.use_tma_KV
                                )
                                else None
                            )
                            if const_expr(not self.use_tma_KV):
                                paged_kv_manager.load_page_table(n_block)
                            pipeline_k.producer_acquire(kv_producer_state)
                            load_K(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            kv_producer_state.advance()
                    else:
                        for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                            n_block_prev = n_block_max - i - 1
                            n_block = n_block_prev - 1
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(mPageTable is not None)
                                else None
                            )
                            page_idx_prev = (
                                mPageTable[batch_idx, n_block_prev]
                                if const_expr(mPageTable is not None)
                                else None
                            )
                            kv_producer_state_prev = kv_producer_state.clone()
                            kv_producer_state.advance()
                            pipeline_k.producer_acquire(kv_producer_state)
                            load_K(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            pipeline_v.producer_acquire(kv_producer_state_prev)
                            load_V(
                                block=n_block_prev,
                                producer_state=kv_producer_state_prev,
                                page_idx=page_idx_prev,
                            )
                        n_block = n_block_min
                        page_idx = (
                            mPageTable[batch_idx, n_block]
                            if const_expr(mPageTable is not None)
                            else None
                        )
                        pipeline_v.producer_acquire(kv_producer_state)
                        load_V(
                            block=n_block,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )
                        kv_producer_state.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            # Producer tail is only useful for cluster to avoid early exit of blocks.
            # We only need producer_tail on V since that's the last that's loaded, we don't
            # need it for Q (no cluster) and K.
            if is_kv_load_warp:
                pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def load_KV(
        self,
        tma_load_fn: Optional[Callable],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
    ):
        if const_expr(self.use_tma_KV):
            src_idx = block if const_expr(page_idx is None) else page_idx
            tma_load_fn(src_idx=src_idx, producer_state=producer_state)
        else:
            paged_kv_manager.load_KV(
                block, sX[None, None, producer_state.index], K_or_V
            )
            cute.arch.cp_async_commit_group()
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        mQ_beam: Optional[cute.Tensor] = None,  # (s_q, d, h, b) for beam phase
        mK_beam: Optional[cute.Tensor] = None,
        mV_beam: Optional[cute.Tensor] = None,
        mTopkIdxs: Optional[cute.Tensor] = None,
        beam_width: Int32 = Int32(0),
        decode_nums: Int32 = Int32(0),
        num_splits: Int32 = Int32(1),
    ):
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.head_dim_padded), sQ, sK
        )
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init,
            tiled_mma_qk,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrK,
        )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.head_dim_v_padded, self.tile_n), sP, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_P = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype
        )
        smem_thr_copy_P = cute.make_tiled_copy_C(
            smem_copy_atom_P, tiled_mma_qk
        ).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )

        # For RescaleOBeforeGemm: persistent scores_scale across iterations
        scores_scale = None
        if const_expr(self.rescale_O_before_gemm):
            scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap
            if const_expr(self.intra_wg_overlap)
            else self.mma_one_n_block,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
            scores_scale=scores_scale,
        )

        process_first_half_block = partial(
            self.first_half_block_overlap,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        process_last_half_block = partial(
            self.last_half_block_overlap,
            pipeline_v=pipeline_v,
            mma_pv_fn=mma_pv_fn,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:

            # shape: (atom_v_m * rest_m)
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
            )
            mma_one_n_block = partial(
                mma_one_n_block_all, seqlen=seqlen, softmax=softmax
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            n_block = cutlass.max(n_block_max - 1, 0)
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)

            if n_block_max > n_block_min:
                # We need masking on S for the very last block when K and V has length not multiple of tile_n.
                O_should_accumulate = False

                # ==========================================
                # MAINLOOP
                # ==========================================
                # First iteration with seqlen masking
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_first_half_block(
                        n_block=n_block,
                        seqlen=seqlen,
                        kv_consumer_state=kv_consumer_state,
                        mask_fn=mask_fn,
                        is_first_block=True,
                    )
                else:
                    self.warp_scheduler_barrier_sync()
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                        is_first_n_block=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                    )
                    O_should_accumulate = True
                # Remaining iterations: from n_block-1 down to n_block_min
                num_remaining = n_block - n_block_min
                for n_tile in cutlass.range(num_remaining, unroll=1):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block - 1 - n_tile,
                        seqlen=seqlen,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                    )
                    O_should_accumulate = True
                # Last "half" iteration
                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_last_half_block(
                        kv_consumer_state=kv_consumer_state,
                        zero_init=not O_should_accumulate,
                    )
                    O_should_accumulate = True
                else:
                    self.warp_scheduler_barrier_arrive()

                # ==========================================
                # Phase 2: Beam Sparse (CUDA core FMA)
                # ==========================================
                if const_expr(self.has_beam_sparse):
                    if split_idx == num_splits - 1:
                        self._beam_sparse_phase(
                            acc_O,
                            softmax,
                            thr_mma_qk,
                            tidx,
                            sQ,
                            mK_beam,
                            mV_beam,
                            mTopkIdxs,
                            beam_width,
                            decode_nums,
                            batch_idx,
                            m_block,
                            head_idx,
                            softmax_scale_log2,
                        )

                # normalize acc_O by row_sum and calculate the lse
                row_scale = softmax.finalize(sink_val=None)
                softmax.rescale_O(acc_O, row_scale)

            # Release Q pipeline AFTER beam phase (beam phase reads Q from sQ)
            pipeline_q.consumer_release_w_index(0)
            q_consumer_phase ^= 1

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue — skip for empty splits (split-KV with no KV blocks)
            # ///////////////////////////////////////////////////////////////////////////////
            if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                self.epilogue(
                    acc_O,
                    softmax.row_sum,
                    mO,
                    mLSE,
                    sO,
                    seqlen,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    tiled_mma_pv,
                    tidx,
                    m_block,
                    head_idx,
                    batch_idx,
                    split_idx,
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        is_first_block: bool = False,
    ):
        """Processes the first half block when using intra-warpgroup-overlap"""

        pipeline_k.consumer_wait(
            kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state)
        )
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        # Apply mask; mask_seqlen always True for first block
        # Caveat: if full block further right than mask block, seqlen masking is redundant;
        # however, masking is being applied anyway, so essentially no perf hit
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)

        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))

        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            # Fence and barrier to make smem store visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()

        # For RescaleOBeforeGemm: initialize acc_O
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        """Processes the final PV GEMM when using intra-warpgroup-overlap"""

        # For RescaleOBeforeGemm: rescale O before the final PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)

        pipeline_v.consumer_wait(
            kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state)
        )
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def _beam_sparse_phase(
        self,
        acc_O: cute.Tensor,
        softmax: Softmax,
        thr_mma: cute.ThrMma,
        tidx: Int32,
        sQ: cute.Tensor,  # swizzled SMEM Q (swizzle on pointer, affine layout)
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
        """Phase 2: Fuse beam sparse KV into context attention's softmax state.

        Uses CUDA core FMA (not HGMMA) to compute per-row QK dot products,
        update online softmax, and accumulate PV into acc_O.
        Reads Q from global memory (swizzled sQ doesn't support element-wise indexing).
        """
        # MMA TV coordinate mapping
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        cO = cute.make_identity_tensor((self.tile_m, self.head_dim_v_padded))
        thr_cO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))

        num_rows = cute.size(acc_O_mn, mode=[0])
        num_cols = cute.size(acc_O_mn, mode=[1])

        # Softmax state for per-row access
        row_max = softmax.row_max
        row_sum = softmax.row_sum

        kv_head_idx = head_idx // self.qhead_per_kvhead

        # In HGMMA TV layout, 4 threads (t0=0..3) share each row.
        # finalize() does warp_reduce(width=4) to sum row_sum across them.
        # Beam phase score is a per-row scalar (not partitioned by N columns),
        # so only 1 of the 4 threads should add to row_sum.
        t0 = tidx % 4
        is_row_sum_writer = t0 == 0

        # Q from swizzled SMEM (swizzle on pointer, affine layout).
        # sQ shape: (tile_m, head_dim) — already loaded by TMA in context phase.

        # K/V beam: (s_k, d, h_k, b) after transpose
        gK_beam = mK_beam[None, None, kv_head_idx, batch_idx]  # (s_beam, d)
        gV_beam = mV_beam[None, None, kv_head_idx, batch_idx]  # (s_beam, d_v)

        # For each decode_nums entry
        for dn in cutlass.range(decode_nums):
            for r in cutlass.range(num_rows, unroll_full=True):
                row_in_tile = thr_cO[r, 0][0]
                beam_idx = m_block * self.tile_m + row_in_tile

                active = beam_idx < beam_width
                kv_idx = Int32(0)
                partial_score = Float32(0.0)
                if active:
                    kv_idx = mTopkIdxs[batch_idx, kv_head_idx, dn, beam_idx]

                    # QK dot product — 4-thread parallel (t0=0..3 each do head_dim/4 FMA)
                    # Q from swizzled SMEM, K from global memory (L2 hit)
                    gK_row = gK_beam[kv_idx, None]  # (d,)
                    chunk = self.head_dim_padded // 4
                    k_start = t0 * chunk
                    for k in cutlass.range(chunk, unroll_full=True):
                        partial_score = partial_score + Float32(
                            sQ[row_in_tile, k_start + k]
                        ) * Float32(gK_row[k_start + k])

                # Keep warp-level reduction in uniform control flow. The last beam
                # fragment can be partial when beam_width is not a multiple of 8.
                score_raw = utils.warp_reduce(partial_score, operator.add, width=4)

                if active:
                    # Online softmax update (row_max stores UNSCALED max,
                    # same convention as FA softmax.online_softmax)
                    m_prev = row_max[r]
                    if score_raw > m_prev:
                        row_max[r] = score_raw
                    m_new = row_max[r]
                    # Scale by softmax_scale_log2 for exp2 computation
                    o_scale = cute.math.exp2(
                        (m_prev - m_new) * softmax_scale_log2, fastmath=True
                    )
                    p_val = cute.math.exp2(
                        (score_raw - m_new) * softmax_scale_log2, fastmath=True
                    )
                    # row_sum: only 1 of 4 threads adds p_val (warp_reduce in finalize)
                    row_sum[r] = row_sum[r] * o_scale
                    if is_row_sum_writer:
                        row_sum[r] = row_sum[r] + p_val

                    # Rescale acc_O and accumulate PV
                    gV_row = gV_beam[kv_idx, None]  # (d_v,)
                    for c in cutlass.range(num_cols, unroll_full=True):
                        col = thr_cO[r, c][1]
                        acc_O_mn[r, c] = acc_O_mn[r, c] * o_scale + p_val * Float32(
                            gV_row[col]
                        )

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,  # not used
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle masking
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        # RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        pipeline_v.consumer_wait(
            smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v)
        )
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        # handle masking
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))

        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP_cur.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )
