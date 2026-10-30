# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Beam attention and combine fused into one launch, bit-identical to ``interface.beam_decode_attn``.

The arithmetic per KV tile is ``FlashAttentionForwardDecode``'s sparse mainloop and the epilogue is
``FlashAttentionForwardCombine`` for ``num_splits=2``, in the same operation order.
"""

import math
import operator

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import interface
import torch
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from src.common import utils
from src.common.cute_dsl_utils import assume_tensor_aligned, to_cute_tensor

from gr_inference.gr_kernels.attention.fused_beam_inputs import (
    check_fused_inputs,
    supports_vector_rows,
)

_COMPILED: dict = {}


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Un-contractable a * b: the public beam attention rescales o before its PV FMAs;
    letting LLVM fuse the rescale changes the bits."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ld_global_s32(addr: cutlass.Int64, *, loc=None, ip=None) -> Int32:
    """Volatile int32 load: neither LLVM nor ptxas sinks it into the branch using it,
    so all top-k indices issue together."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.s32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class BeamSparseCombineVec:
    def __init__(
        self,
        dtype,
        head_dim,
        qhead_per_kvhead,
        max_decode,
        num_threads=128,
        rows_in_flight=8,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.vec_size = max(128 // dtype.width, head_dim // 32)
        assert (
            self.vec_size * dtype.width == 128
        ), "one 128-bit access per thread per row"
        self.bdx = head_dim // self.vec_size
        self.bdy = qhead_per_kvhead
        self.bdz = max(1, num_threads // (self.bdx * self.bdy))
        # FlashAttentionForwardDecode's sparse tile: bdy rows per iteration (tile_size_per_bdx 8 if bdy == 1).
        self.tile_size = self.bdy * (8 if self.bdy == 1 else 1)
        self.max_decode = max_decode
        self.max_iters = -(-max_decode // self.tile_size)
        # K/V rows are gathered a stage at a time (at least rows_in_flight rows, whole tiles) into one of two
        # register buffers, so registers stay fixed at any decode depth.
        self.tiles_per_stage = -(-rows_in_flight // self.tile_size)
        self.stage_rows = self.tiles_per_stage * self.tile_size
        self.num_stages = -(-self.max_iters // self.tiles_per_stage)
        self.buf_rows = min(self.stage_rows, self.max_iters * self.tile_size)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b*W, 1, h, d)
        mK: cute.Tensor,  # (b, s_k, h_k, d)
        mV: cute.Tensor,  # (b, s_k, h_k, d)
        mOCtx: cute.Tensor,  # (b*W, 1, h, d) fp32, context attention output
        mLSECtx: cute.Tensor,  # (b, h, W) fp32, context attention LSE
        mO: cute.Tensor,  # (b*W, 1, h, d)
        mLSE: cute.Tensor,  # (b, h, W) fp32
        softmax_scale: Float32,
        mTopkIdxs: cute.Tensor,  # (b, h_k, dn, W)
        beam_width: Int32,
        mDecodeNums: cute.Tensor,  # (1,) int32
        stream: cuda.CUstream = None,
    ):
        if const_expr(
            not (mQ.element_type == mK.element_type == mV.element_type == self.dtype)
        ):
            raise TypeError("Q/K/V must match the kernel dtype")
        if const_expr(
            mOCtx.element_type != Float32
            or mLSECtx.element_type != Float32
            or mLSE.element_type != Float32
        ):
            raise TypeError("context partials and LSE must be Float32")
        self.o_dtype = mO.element_type
        softmax_scale_log2 = softmax_scale * Float32(1.44269504088896340736)
        mQ, mK, mV, mO, mOCtx = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mO, mOCtx)
        ]
        transpose = [1, 3, 2, 0]  # (b, s, h, d) -> (s, d, h, b)
        mQ, mK, mV, mO, mOCtx = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=transpose))
            for t in (mQ, mK, mV, mO, mOCtx)
        ]
        self.kernel(
            mQ,
            mK,
            mV,
            mOCtx,
            mLSECtx,
            mO,
            mLSE,
            softmax_scale_log2,
            mTopkIdxs,
            beam_width,
            mDecodeNums,
        ).launch(
            grid=(cute.ceil_div(mQ.shape[3], self.bdz), mK.shape[2], 1),
            block=(self.bdx, self.bdy, self.bdz),
            stream=stream,
        )

    @cute.jit
    def _load_vec(self, src: cute.Tensor, offset: Int32, dtype, dst: cute.Tensor):
        ptr = cute.make_ptr(
            dtype,
            (src.iterator + offset).toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        cute.autovec_copy(
            cute.make_tensor(ptr, cute.make_layout((cute.size(dst),))), dst
        )

    @cute.jit
    def _load_stage_indices(self, gTopk: cute.Tensor, stage, kv_idx):
        kv_idx.fill(0)
        if const_expr(isinstance(stage, int)):
            for j in cutlass.range_constexpr(
                min(self.buf_rows, self.max_decode - stage * self.stage_rows)
            ):
                kv_idx[j] = ld_global_s32(
                    utils.elem_pointer(gTopk, stage * self.stage_rows + j).toint()
                )
        else:
            for j in cutlass.range_constexpr(self.buf_rows):
                if stage * self.stage_rows + j < self.max_decode:
                    kv_idx[j] = ld_global_s32(
                        utils.elem_pointer(gTopk, stage * self.stage_rows + j).toint()
                    )

    @cute.jit
    def _gather_stage(
        self,
        gK: cute.Tensor,
        gV: cute.Tensor,
        kv_idx,
        stage,
        valid_rows: Int32,
        tx: Int32,
        k_buf,
        v_buf,
    ):
        # Rows past decode_nums stay zero, as in the public beam attention's smem fill.
        k_buf.fill(0.0)
        v_buf.fill(0.0)
        rows = (
            min(
                self.buf_rows, self.max_iters * self.tile_size - stage * self.stage_rows
            )
            if const_expr(isinstance(stage, int))
            else self.buf_rows
        )
        for j in cutlass.range_constexpr(rows):
            if stage * self.stage_rows + j < valid_rows:
                row_off = cute.crd2idx((kv_idx[j], tx * self.vec_size), gK.layout)
                self._load_vec(gK, row_off, self.dtype, k_buf[None, j])
                self._load_vec(gV, row_off, self.dtype, v_buf[None, j])

    @cute.jit
    def _attend_stage(
        self,
        stage,
        k_buf,
        v_buf,
        q_reg,
        s_reg,
        o_reg,
        softmax_state,
        seqlen_k: Int32,
        num_iters: Int32,
        softmax_scale_log2: Float32,
    ):
        vec = self.vec_size
        tiles = (
            min(self.tiles_per_stage, self.max_iters - stage * self.tiles_per_stage)
            if const_expr(isinstance(stage, int))
            else self.tiles_per_stage
        )
        for t in cutlass.range_constexpr(tiles):
            tile_idx = stage * self.tiles_per_stage + t
            if tile_idx < num_iters:
                kv_base = tile_idx * self.tile_size
                m_prev = softmax_state[0]
                st_m = m_prev
                for j in cutlass.range_constexpr(self.tile_size):
                    score = Float32(0.0)
                    for i in cutlass.range_constexpr(vec):
                        score = score + q_reg[i] * Float32(
                            k_buf[i, t * self.tile_size + j]
                        )
                    score = utils.warp_reduce(score, operator.add, width=self.bdx)
                    score = score * softmax_scale_log2
                    if kv_base + j >= seqlen_k:
                        score = -Float32.inf
                    if score > st_m:
                        st_m = score
                    s_reg[j] = score

                o_scale = cute.math.exp2(m_prev - st_m, fastmath=True)
                st_d = softmax_state[1] * o_scale
                for j in cutlass.range_constexpr(self.tile_size):
                    s_reg[j] = cute.math.exp2(s_reg[j] - st_m, fastmath=True)
                    st_d = st_d + s_reg[j]
                for i in cutlass.range_constexpr(vec):
                    o_reg[i] = mul_rn(o_reg[i], o_scale)
                for j in cutlass.range_constexpr(self.tile_size):
                    for i in cutlass.range_constexpr(vec):
                        o_reg[i] = o_reg[i] + s_reg[j] * Float32(
                            v_buf[i, t * self.tile_size + j]
                        )
                softmax_state[0] = st_m
                softmax_state[1] = st_d

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mOCtx: cute.Tensor,
        mLSECtx: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        softmax_scale_log2: Float32,
        mTopkIdxs: cute.Tensor,
        beam_width: Int32,
        mDecodeNums: cute.Tensor,
    ):
        tx, ty, tz = cute.arch.thread_idx()
        flat_batch_idx = cute.arch.block_idx()[0] * self.bdz + tz
        kv_head_idx = cute.arch.block_idx()[1]
        qo_head_idx = kv_head_idx * self.bdy + ty
        if flat_batch_idx < mQ.shape[3]:
            batch_idx = flat_batch_idx // beam_width
            beam_idx = flat_batch_idx % beam_width
            seqlen_k = mDecodeNums[0]
            vec = self.vec_size

            gTopk = mTopkIdxs[batch_idx, kv_head_idx, None, beam_idx]
            kv_idx = [cute.make_rmem_tensor((self.buf_rows,), Int32) for _ in range(2)]
            self._load_stage_indices(gTopk, 0, kv_idx[0])

            gQ = mQ[0, None, qo_head_idx, flat_batch_idx]
            q_raw = cute.make_rmem_tensor((vec,), self.dtype)
            self._load_vec(gQ, tx * vec, self.dtype, q_raw)
            gOCtx = mOCtx[0, None, qo_head_idx, flat_batch_idx]
            o_ctx = cute.make_rmem_tensor((vec,), Float32)
            self._load_vec(gOCtx, tx * vec, Float32, o_ctx)
            lse_ctx = mLSECtx[batch_idx, qo_head_idx, beam_idx]

            gK = mK[None, None, kv_head_idx, batch_idx]
            gV = mV[None, None, kv_head_idx, batch_idx]
            k_bufs = [
                cute.make_rmem_tensor((vec, self.buf_rows), self.dtype)
                for _ in range(2)
            ]
            v_bufs = [
                cute.make_rmem_tensor((vec, self.buf_rows), self.dtype)
                for _ in range(2)
            ]
            valid_rows = cutlass.min(seqlen_k, self.max_iters * self.tile_size)
            self._gather_stage(
                gK, gV, kv_idx[0], 0, valid_rows, tx, k_bufs[0], v_bufs[0]
            )

            q_reg = cute.make_rmem_tensor((vec,), Float32)
            for i in cutlass.range_constexpr(vec):
                q_reg[i] = Float32(q_raw[i])

            num_iters = cutlass.min(
                cute.ceil_div(seqlen_k, self.tile_size), self.max_iters
            )
            # running max (log2 units) and denominator
            softmax_state = cute.make_rmem_tensor((2,), Float32)
            softmax_state[0] = Float32(-5e4)
            softmax_state[1] = Float32(0.0)
            o_reg = cute.make_rmem_tensor((vec,), Float32)
            o_reg.fill(Float32(0.0))
            s_reg = cute.make_rmem_tensor((self.tile_size,), Float32)
            # Each stage's loads are in flight while the previous stage computes; stages go in
            # order, so the arithmetic order matches the public kernel's tile loop. Past the first
            # stage, a runtime loop over pairs of stages keeps registers at two stages for any
            # decode depth.
            if const_expr(self.num_stages == 1):
                self._attend_stage(
                    0,
                    k_bufs[0],
                    v_bufs[0],
                    q_reg,
                    s_reg,
                    o_reg,
                    softmax_state,
                    seqlen_k,
                    num_iters,
                    softmax_scale_log2,
                )
            else:
                for stage in cutlass.range(
                    0, cute.ceil_div(valid_rows, self.stage_rows), 2, unroll=1
                ):
                    self._load_stage_indices(gTopk, stage + 1, kv_idx[1])
                    self._gather_stage(
                        gK,
                        gV,
                        kv_idx[1],
                        stage + 1,
                        valid_rows,
                        tx,
                        k_bufs[1],
                        v_bufs[1],
                    )
                    self._attend_stage(
                        stage,
                        k_bufs[0],
                        v_bufs[0],
                        q_reg,
                        s_reg,
                        o_reg,
                        softmax_state,
                        seqlen_k,
                        num_iters,
                        softmax_scale_log2,
                    )
                    self._load_stage_indices(gTopk, stage + 2, kv_idx[0])
                    self._gather_stage(
                        gK,
                        gV,
                        kv_idx[0],
                        stage + 2,
                        valid_rows,
                        tx,
                        k_bufs[0],
                        v_bufs[0],
                    )
                    self._attend_stage(
                        stage + 1,
                        k_bufs[1],
                        v_bufs[1],
                        q_reg,
                        s_reg,
                        o_reg,
                        softmax_state,
                        seqlen_k,
                        num_iters,
                        softmax_scale_log2,
                    )
            st_m = softmax_state[0]
            st_d = softmax_state[1]

            st_d_is_zero_or_nan = (st_d == Float32(0.0)) or (st_d != st_d)
            inv_d = cute.arch.rcp_approx(
                st_d if not st_d_is_zero_or_nan else Float32(1.0)
            )
            for i in cutlass.range_constexpr(vec):
                o_reg[i] = o_reg[i] * inv_d
            LN2 = math.log(2.0)
            lse_beam = (
                (st_m + cute.math.log2(st_d, fastmath=True)) * LN2
                if not st_d_is_zero_or_nan
                else -Float32.inf
            )

            # Same order and arithmetic as the public combine kernel.
            lse_max = cute.arch.fmax(lse_ctx, lse_beam)
            max_valid_split = Int32(-1)
            if lse_ctx != -Float32.inf:
                max_valid_split = Int32(0)
            if lse_beam != -Float32.inf:
                max_valid_split = Int32(1)
            lse_max_cur = 0.0 if lse_max == -Float32.inf else lse_max
            LOG2_E = math.log2(math.e)
            scale_ctx = cute.math.exp2(
                lse_ctx * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True
            )
            scale_beam = cute.math.exp2(
                lse_beam * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True
            )
            lse_sum_cur = Float32(scale_ctx + scale_beam)
            lse_out = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
            inv_sum = (
                0.0
                if (lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur)
                else 1.0 / lse_sum_cur
            )
            scale_ctx = scale_ctx * inv_sum
            scale_beam = scale_beam * inv_sum

            o_acc = cute.make_rmem_tensor((vec,), Float32)
            o_acc.fill(0.0)
            if max_valid_split >= 0 and scale_ctx > 0.0:
                o_acc.store(o_acc.load() + scale_ctx * o_ctx.load())
            if max_valid_split >= 1 and scale_beam > 0.0:
                o_acc.store(o_acc.load() + scale_beam * o_reg.load())

            r_o = cute.make_rmem_tensor((vec,), self.o_dtype)
            r_o.store(o_acc.load().to(self.o_dtype))
            gO = mO[0, None, qo_head_idx, flat_batch_idx]
            o_ptr = cute.make_ptr(
                self.o_dtype,
                (gO.iterator + tx * vec).toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            cute.autovec_copy(r_o, cute.make_tensor(o_ptr, cute.make_layout((vec,))))
            if tx == 0:
                mLSE[batch_idx, qo_head_idx, beam_idx] = lse_out


def beam_decode_fused(
    q,
    k_context,
    v_context,
    k_beam,
    v_beam,
    topk_indices,
    decode_nums,
    softmax_scale=None,
    return_lse=False,
    out=None,
    backend="dsl",
    seqused_k=None,
    cu_seqlens_k=None,
):
    interface._validate_inputs(
        q,
        k_context,
        v_context,
        k_beam,
        v_beam,
        topk_indices,
        decode_nums,
        jagged_k_context=cu_seqlens_k is not None,
    )
    check_fused_inputs(q, topk_indices, decode_nums, seqused_k, cu_seqlens_k, backend)
    if not (
        (backend == "3kernel" or interface._get_compute_capability() in (10, 11))
        and topk_indices.dtype == torch.int32
        and supports_vector_rows(q, k_beam, v_beam)
        and _context_num_splits(q, k_context, seqused_k, cu_seqlens_k) == 1
    ):
        return interface.beam_decode_attn(
            q,
            k_context,
            v_context,
            k_beam,
            v_beam,
            topk_indices,
            decode_nums,
            softmax_scale=softmax_scale,
            return_lse=return_lse,
            out=out,
            backend=backend,
            seqused_k=seqused_k,
            cu_seqlens_k=cu_seqlens_k,
        )
    B, _, W, Hq, D = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    o_ctx = torch.empty(B, W, Hq, D, device=q.device, dtype=torch.float32)
    lse_ctx = torch.empty(B, Hq, W, device=q.device, dtype=torch.float32)
    o_out = torch.empty(B, W, Hq, D, device=q.device, dtype=q.dtype)
    lse_out = torch.empty(B, Hq, W, device=q.device, dtype=torch.float32)
    interface._context_attention(
        q,
        k_context,
        v_context,
        softmax_scale,
        out=o_ctx,
        lse=lse_ctx,
        seqused_k=seqused_k,
        cu_seqlens_k=cu_seqlens_k,
    )
    # A fill kernel, not a host copy, so it is safe under CUDA graph capture.
    decode_nums_dev = torch.full((1,), decode_nums, device=q.device, dtype=torch.int32)
    _launch_fused_beam_combine(
        q,
        k_beam,
        v_beam,
        topk_indices,
        softmax_scale,
        o_ctx,
        lse_ctx,
        o_out,
        lse_out,
        decode_nums_dev,
    )
    o_out = o_out.unsqueeze(1)
    if out is not None:
        assert out.shape == q.shape and out.dtype == q.dtype
        out.copy_(o_out)
        o_out = out
    return o_out, (lse_out.transpose(-1, -2).unsqueeze(1) if return_lse else None)


def _context_num_splits(q, k_context, seqused_k, cu_seqlens_k):
    """Must match the K1 split count of public ``BeamDecodeAttn.forward``'s 3-kernel path."""
    if seqused_k is not None or cu_seqlens_k is not None:
        return 1
    B, _, W, Hq, D = q.shape
    Sk = k_context.shape[1]
    tile = 128 if D <= 128 else 64
    num_m_blocks = math.ceil(W / tile)
    num_n_blocks = math.ceil(Sk / tile)
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    return interface.num_splits_heuristic(
        B * Hq * num_m_blocks, num_sms, num_n_blocks, num_m_blocks, Sk * D * 2 * 2
    )


def _launch_fused_beam_combine(
    q,
    k_beam,
    v_beam,
    topk_indices,
    softmax_scale,
    o_ctx,
    lse_ctx,
    o_out,
    lse_out,
    decode_nums,
):
    B, _, W, Hq, D = q.shape
    Hkv = k_beam.shape[2]
    qhead_per_kv = Hq // Hkv
    q_flat = q.reshape(B * W, 1, Hq, D)
    o_ctx_flat = o_ctx.reshape(B * W, 1, Hq, D)
    o_flat = o_out.reshape(B * W, 1, Hq, D)
    topk_kv = topk_indices[:, 0, ::qhead_per_kv, :, :]
    cutlass_dtype = interface.TORCH_TO_CUTLASS[q.dtype]
    key = (
        "fused_beam_combine",
        torch.cuda.get_device_capability(q.device),
        D,
        qhead_per_kv,
        cutlass_dtype,
        interface.TORCH_TO_CUTLASS[o_out.dtype],
        int(topk_kv.shape[2]),
    )
    if key not in _COMPILED:
        kernel = BeamSparseCombineVec(
            cutlass_dtype,
            head_dim=D,
            qhead_per_kvhead=qhead_per_kv,
            max_decode=int(topk_kv.shape[2]),
        )
        _COMPILED[key] = cute.compile(
            kernel,
            to_cute_tensor(q_flat),
            to_cute_tensor(k_beam),
            to_cute_tensor(v_beam),
            to_cute_tensor(o_ctx_flat),
            to_cute_tensor(lse_ctx, assumed_align=4),
            to_cute_tensor(o_flat),
            to_cute_tensor(lse_out, assumed_align=4),
            softmax_scale,
            to_cute_tensor(topk_kv),
            W,
            to_cute_tensor(decode_nums),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _COMPILED[key](
        q_flat.detach(),
        k_beam.detach(),
        v_beam.detach(),
        o_ctx_flat,
        lse_ctx,
        o_flat,
        lse_out,
        softmax_scale,
        topk_kv,
        W,
        decode_nums,
    )
