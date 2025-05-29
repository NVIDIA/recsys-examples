# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
try:
    from triton.language.extra.libdevice import fast_dividef
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import fast_dividef
    except ImportError:
        from triton.language.math import fast_dividef


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


@triton.jit
def _hstu_attn_fwd_one_block(  # noqa: C901
    start_n,
    seq_len,
    offs_m,
    offs_n,
    mask_m,
    mask_n,
    q,
    K_block_ptr,
    V_block_ptr,
    n_targets,
    alpha,
    MAX_SEQ_LEN,
    contextual_seq_len,
    MAX_ATTN_LEN: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
    # k = tl.load(K_block_ptr)
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(
            offs_m > 0,
            offs_m,
            0,
        )
        offs_n = offs_n - contextual_seq_len + 1
        offs_n = tl.where(
            offs_n > 0,
            offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        offs_n = tl.where(
            offs_n < max_ids,
            offs_n,
            max_ids,
        )
    offs_n_minus_m = offs_n[None, :] - offs_m[:, None]
    if MAX_ATTN_LEN > 0:
        if CAUSAL:
            invalid_mask = invalid_mask or (
                offs_n_minus_m < 0 and offs_n_minus_m >= -MAX_ATTN_LEN
            )
    else:
        if CAUSAL:
            invalid_mask = invalid_mask or offs_n_minus_m < 0
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask or (
            offs_m[:, None] == 0 and offs_n[None, :] < max_ids
        )
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)
    silu = tl.where(invalid_mask, silu, 0)
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    # v = tl.load(V_block_ptr)
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)

@triton.jit
def _paged_hstu_attn_fwd_mask_one_block_no_boundary_check(  # noqa: C901
    start_n,                    # only for compiler hint
    seq_len,
    offs_m,
    offs_n,
    q,
    K_block_ptr,
    V_block_ptr,
    alpha,
    MAX_SEQ_LEN,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl.load(K_block_ptr)
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)

    invalid_mask = offs_n[None, :] <= offs_m[:, None]
    silu = tl.where(invalid_mask, silu, 0)
    
    v = tl.load(V_block_ptr)
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)

@triton.jit
def _paged_hstu_attn_fwd_mask_one_block(  # noqa: C901
    start_n,                    # only for compiler hint
    seq_len,
    offs_m,
    offs_n,
    q,
    K_block_ptr,
    V_block_ptr,
    alpha,
    MAX_SEQ_LEN,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)

    invalid_mask = offs_n[None, :] <= offs_m[:, None]
    silu = tl.where(invalid_mask, silu, 0)
    
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)

@triton.jit
def _paged_hstu_attn_fwd_one_block_no_boundary_check(  # noqa: C901
    start_n,
    q,
    K_block_ptr,
    V_block_ptr,
    alpha,
    MAX_SEQ_LEN,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl.load(K_block_ptr)
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha

    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)
    v = tl.load(V_block_ptr)
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)

@triton.jit
def _paged_hstu_attn_fwd_one_block(  # noqa: C901
    start_n,
    q,
    K_block_ptr,
    V_block_ptr,
    alpha,
    MAX_SEQ_LEN,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha

    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.jit
def _paged_hstu_attn_fwd_compute(  # noqa C901
    Q_seq,
    K_seq,
    V_seq,
    KV_cache,
    kv_cache_indices,
    kv_cache_indptr,
    kv_cache_tail_lens,
    seq_offsets,
    num_candidates,
    history_lens,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    stride_kv_page,
    stride_k_v,
    alpha,
    MAX_SEQ_LEN,
    off_z,
    off_h,
    pid,
    H,
    CACHE_PAGE_SIZE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)

    stride_per_token = H * BLOCK_DIM

    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    candidate_len = tl.load(num_candidates + off_z).to(tl.int64)
    candidate_start = seq_end - candidate_len
    
    history_len = tl.load(history_lens + off_z)

    start_m_delta = pid * BLOCK_M
    start_m = start_m_delta + history_len

    if start_m_delta < candidate_len:
        Q_block_ptr = tl.make_block_ptr(
            base=Q_seq + off_h * stride_qh + candidate_start * stride_qm,
            shape=(candidate_len, BLOCK_DIM),
            strides=(stride_qm, 1),
            offsets=(start_m_delta, 0),
            block_shape=(BLOCK_M, BLOCK_DIM),
            order=(1, 0),
        )

        if start_m_delta + BLOCK_M > candidate_len:
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        else:
            q = tl.load(Q_block_ptr)
        acc = tl.zeros([BLOCK_M, BLOCK_DIM], dtype=tl.float32)
        
        # pyre-ignore[61]
        # history loop
        hist_cache_page_id_start = tl.load(kv_cache_indptr + off_z).to(tl.int64)
        hist_cache_page_id_end = tl.load(kv_cache_indptr + off_z + 1)
        hist_cache_page_id_len = hist_cache_page_id_end - hist_cache_page_id_start
        hist_last_page_size = tl.load(kv_cache_tail_lens + off_z).to(tl.int32)

        history_len_boundary = history_len - history_len % BLOCK_N
        for start_n in range(0, history_len_boundary, BLOCK_N):
            seq_page_idx = start_n // CACHE_PAGE_SIZE
            page_token_offset = start_n % CACHE_PAGE_SIZE
            token_len = CACHE_PAGE_SIZE if seq_page_idx < (hist_cache_page_id_len - 1) else hist_last_page_size
            page_id = tl.load(kv_cache_indices + hist_cache_page_id_start + seq_page_idx).to(tl.int64)
            
            K_history_base_ptr = tl.make_block_ptr(
                base=KV_cache + off_h * BLOCK_DIM + page_token_offset * stride_per_token + page_id * stride_kv_page,
                shape=(BLOCK_DIM, token_len),
                strides=(1, stride_per_token),
                offsets=(0, 0),
                block_shape=(BLOCK_DIM, BLOCK_N),
                order=(0, 1),
            )
            V_history_base_ptr = tl.make_block_ptr(
                base=KV_cache + off_h * BLOCK_DIM + page_token_offset * stride_per_token + page_id * stride_kv_page + stride_k_v,
                shape=(token_len, BLOCK_DIM),
                strides=(stride_per_token, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_DIM),
                order=(1, 0),
            )
            acc += _paged_hstu_attn_fwd_one_block_no_boundary_check(
                start_n=start_n,
                q=q,
                K_block_ptr=K_history_base_ptr,
                V_block_ptr=V_history_base_ptr,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_N=BLOCK_N,
            )
        if history_len_boundary < history_len:
            seq_page_idx = history_len_boundary // CACHE_PAGE_SIZE
            page_token_offset = history_len_boundary % CACHE_PAGE_SIZE
            token_len = CACHE_PAGE_SIZE if seq_page_idx < (hist_cache_page_id_len - 1) else hist_last_page_size
            page_id = tl.load(kv_cache_indices + hist_cache_page_id_start + seq_page_idx).to(tl.int64)
            
            K_history_base_ptr = tl.make_block_ptr(
                base=KV_cache + off_h * BLOCK_DIM + page_token_offset * stride_per_token + page_id * stride_kv_page,
                shape=(BLOCK_DIM, token_len),
                strides=(1, stride_per_token),
                offsets=(0, 0),
                block_shape=(BLOCK_DIM, BLOCK_N),
                order=(0, 1),
            )
            V_history_base_ptr = tl.make_block_ptr(
                base=KV_cache + off_h * BLOCK_DIM + page_token_offset * stride_per_token + page_id * stride_kv_page + stride_k_v,
                shape=(token_len, BLOCK_DIM),
                strides=(stride_per_token, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_DIM),
                order=(1, 0),
            )
            acc += _paged_hstu_attn_fwd_one_block(
                start_n=history_len_boundary,
                q=q,
                K_block_ptr=K_history_base_ptr,
                V_block_ptr=V_history_base_ptr,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_N=BLOCK_N,
            )

        # candidate loop
        K_block_ptr = tl.make_block_ptr(
            base=K_seq + off_h * stride_kh + candidate_start * stride_kn,
            shape=(BLOCK_DIM, candidate_len),
            strides=(1, stride_kn),
            offsets=(0, start_m_delta),
            block_shape=(BLOCK_DIM, BLOCK_N),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_seq + off_h * stride_vh + candidate_start * stride_vn,
            shape=(candidate_len, BLOCK_DIM),
            strides=(stride_vn, 1),
            offsets=(start_m_delta, 0),
            block_shape=(BLOCK_N, BLOCK_DIM),
            order=(1, 0),
        )

        offs_m = start_m_delta + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        for start_delta in tl.range(start_m_delta, (start_m_delta + BLOCK_M), BLOCK_N):
            cur_offs_n = offs_n + start_delta
            acc += _hstu_attn_fwd_one_block(
                start_n=start_delta,
                seq_len=candidate_len,
                offs_m=offs_m,       # offs within candidate
                offs_n=cur_offs_n,   # offs within candidate
                mask_m=None,         # un-used
                mask_n=None,         # un-used
                q=q,
                K_block_ptr=K_block_ptr,
                V_block_ptr=V_block_ptr,
                n_targets=candidate_len,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                MAX_ATTN_LEN=0,
                contextual_seq_len=0,
                CAUSAL=True,
                HAS_MULTIPLE_TARGETS=True,
                HAS_CONTEXTUAL_SEQ_LEN=False,
                IS_DELTA_Q=True,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
        offs_v_d = tl.arange(0, BLOCK_DIM)
        off_o = Out + candidate_start * stride_om + off_h * stride_oh
        out_ptrs = off_o + offs_m_delta[:, None] * stride_om + offs_v_d[None, :]
        tl.store(out_ptrs, acc, mask=(offs_m_delta < candidate_len)[:, None])

@triton.jit
def _history_paged_hstu_attn_fwd_compute(  # noqa C901
    Q_seq,
    KV_cache,
    kv_cache_indices,
    kv_cache_indptr,
    kv_cache_tail_lens,
    seq_offsets,
    num_candidates,
    history_lens,
    Out,
    stride_qm,
    stride_qh,
    stride_om,
    stride_oh,
    stride_kv_page,
    stride_k_v,
    alpha,
    MAX_SEQ_LEN,
    off_z,
    off_h,
    pid,
    H,
    CACHE_PAGE_SIZE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)

    stride_per_token = H * BLOCK_DIM

    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1).to(tl.int64)
    candidate_len = tl.load(num_candidates + off_z).to(tl.int64)
    delta_hist_len = (seq_end - seq_start - candidate_len).to(tl.int32)

    history_len = tl.load(history_lens + off_z).to(tl.int64)

    start_m_delta = pid * BLOCK_M
    start_m = start_m_delta + history_len - delta_hist_len

    if start_m_delta < delta_hist_len:
        Q_block_ptr = tl.make_block_ptr(
            base=Q_seq + off_h * stride_qh + seq_start * stride_qm,
            shape=(delta_hist_len, BLOCK_DIM),
            strides=(stride_qm, 1),
            offsets=(start_m_delta, 0),
            block_shape=(BLOCK_M, BLOCK_DIM),
            order=(1, 0),
        )

        if start_m_delta + BLOCK_M > delta_hist_len:
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        else:
            q = tl.load(Q_block_ptr)
        acc = tl.zeros([BLOCK_M, BLOCK_DIM], dtype=tl.float32)

        # pyre-ignore[61]
        # history loop
        hist_cache_page_id_start = tl.load(kv_cache_indptr + off_z).to(tl.int64)
        hist_cache_page_id_end = tl.load(kv_cache_indptr + off_z + 1).to(tl.int64)
        hist_cache_page_id_len = hist_cache_page_id_end - hist_cache_page_id_start
        hist_last_page_size = tl.load(kv_cache_tail_lens + off_z).to(tl.int32)

        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        history_len_boundary = history_len - history_len % BLOCK_N
        for start_n in range(0, history_len_boundary, BLOCK_N):
            seq_page_idx = start_n // CACHE_PAGE_SIZE
            page_token_offset = start_n % CACHE_PAGE_SIZE
            token_len = CACHE_PAGE_SIZE if seq_page_idx < (hist_cache_page_id_len - 1) else hist_last_page_size
            page_id = tl.load(kv_cache_indices + hist_cache_page_id_start + seq_page_idx).to(tl.int64)
            
            K_base_ptr = tl.make_block_ptr(
                base=KV_cache + off_h * BLOCK_DIM + page_token_offset * stride_per_token + page_id * stride_kv_page,
                shape=(BLOCK_DIM, token_len),
                strides=(1, stride_per_token),
                offsets=(0, 0),
                block_shape=(BLOCK_DIM, BLOCK_N),
                order=(0, 1),
            )
            V_base_ptr = tl.make_block_ptr(
                base=KV_cache + off_h * BLOCK_DIM + page_token_offset * stride_per_token + page_id * stride_kv_page + stride_k_v,
                shape=(token_len, BLOCK_DIM),
                strides=(stride_per_token, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_DIM),
                order=(1, 0),
            )

            cur_offs_n = offs_n + start_n
            acc += _paged_hstu_attn_fwd_mask_one_block_no_boundary_check(
                start_n=start_n,
                seq_len=history_len,
                offs_m=offs_m,
                offs_n=cur_offs_n,
                q=q,
                K_block_ptr=K_base_ptr,
                V_block_ptr=V_base_ptr,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_N=BLOCK_N,
            )
        
        if history_len_boundary < history_len:
            seq_page_idx = history_len_boundary // CACHE_PAGE_SIZE
            page_token_offset = history_len_boundary % CACHE_PAGE_SIZE
            token_len = CACHE_PAGE_SIZE if seq_page_idx < (hist_cache_page_id_len - 1) else hist_last_page_size
            page_id = tl.load(kv_cache_indices + hist_cache_page_id_start + seq_page_idx).to(tl.int64)
            
            K_base_ptr = tl.make_block_ptr(
                base=KV_cache + off_h * BLOCK_DIM + page_token_offset * stride_per_token + page_id * stride_kv_page,
                shape=(BLOCK_DIM, token_len),
                strides=(1, stride_per_token),
                offsets=(0, 0),
                block_shape=(BLOCK_DIM, BLOCK_N),
                order=(0, 1),
            )
            V_base_ptr = tl.make_block_ptr(
                base=KV_cache + off_h * BLOCK_DIM + page_token_offset * stride_per_token + page_id * stride_kv_page + stride_k_v,
                shape=(token_len, BLOCK_DIM),
                strides=(stride_per_token, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_DIM),
                order=(1, 0),
            )

            cur_offs_n = offs_n + history_len_boundary
            acc += _paged_hstu_attn_fwd_mask_one_block(
                start_n=history_len_boundary,
                seq_len=history_len,
                offs_m=offs_m,
                offs_n=cur_offs_n,
                q=q,
                K_block_ptr=K_base_ptr,
                V_block_ptr=V_base_ptr,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_N=BLOCK_N,
            )

        offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
        offs_v_d = tl.arange(0, BLOCK_DIM)
        off_o = Out + seq_start * stride_om + off_h * stride_oh
        out_ptrs = off_o + offs_m_delta[:, None] * stride_om + offs_v_d[None, :]
        tl.store(out_ptrs, acc, mask=(offs_m_delta < delta_hist_len)[:, None])

@triton.autotune(
    configs=[triton.Config({}, num_stages=2, num_warps=4)],
    key=[],
)
@triton.jit
def _paged_hstu_attn_fwd(  # noqa C901
    Q_seq,
    K_seq,
    V_seq,
    KV_cache,
    kv_cache_indices,
    kv_cache_indptr,
    kv_cache_tail_lens,
    seq_offsets,
    num_candidates,
    history_lens,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    stride_kv_page,
    stride_k_v,
    alpha,
    MAX_SEQ_LEN,
    H,
    CACHE_PAGE_SIZE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    pid = tl.program_id(0)
    _paged_hstu_attn_fwd_compute(
        Q_seq=Q_seq,
        K_seq=K_seq,
        V_seq=V_seq,
        KV_cache=KV_cache,
        kv_cache_indices=kv_cache_indices,
        kv_cache_indptr=kv_cache_indptr,
        kv_cache_tail_lens=kv_cache_tail_lens,
        seq_offsets=seq_offsets,
        num_candidates=num_candidates,
        history_lens=history_lens,
        Out=Out,
        stride_qm=stride_qm,
        stride_qh=stride_qh,
        stride_kn=stride_kn,
        stride_kh=stride_kh,
        stride_vn=stride_vn,
        stride_vh=stride_vh,
        stride_om=stride_om,
        stride_oh=stride_oh,
        stride_kv_page=stride_kv_page,
        stride_k_v=stride_k_v,
        alpha=alpha,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        off_z=off_z,
        off_h=off_h,
        pid=pid,
        H=H,
        CACHE_PAGE_SIZE = CACHE_PAGE_SIZE,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

@triton.autotune(
    configs=[triton.Config({}, num_stages=2, num_warps=4)],
    key=[],
)
@triton.jit
def _history_paged_hstu_attn_fwd(  # noqa C901
    Q_seq,
    KV_cache,
    kv_cache_indices,
    kv_cache_indptr,
    kv_cache_tail_lens,
    seq_offsets,
    num_candidates,
    history_lens,
    Out,
    stride_qm,
    stride_qh,
    stride_om,
    stride_oh,
    stride_kv_page,
    stride_k_v,
    alpha,
    MAX_SEQ_LEN,
    H: tl.constexpr,
    CACHE_PAGE_SIZE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    pid = tl.program_id(0)
    _history_paged_hstu_attn_fwd_compute(
        Q_seq=Q_seq,
        KV_cache=KV_cache,
        kv_cache_indices=kv_cache_indices,
        kv_cache_indptr=kv_cache_indptr,
        kv_cache_tail_lens=kv_cache_tail_lens,
        seq_offsets=seq_offsets,
        num_candidates=num_candidates,
        history_lens=history_lens,
        Out=Out,
        stride_qm=stride_qm,
        stride_qh=stride_qh,
        stride_om=stride_om,
        stride_oh=stride_oh,
        stride_kv_page=stride_kv_page,
        stride_k_v=stride_k_v,
        alpha=alpha,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        off_z=off_z,
        off_h=off_h,
        pid=pid,
        H = H,
        CACHE_PAGE_SIZE=CACHE_PAGE_SIZE,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


def native_triton_candidate_hstu_mha(
    N: int,
    alpha: float,
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    output: torch.Tensor,
    max_num_candidate: int,
    kv_cache: torch.Tensor,
    kv_cache_indices: torch.Tensor,
    kv_cache_indptr: torch.Tensor,
    kv_cache_tail_lens: torch.Tensor,
    kv_cache_page_size: int,
    seq_offsets: torch.Tensor,
    num_candidates: torch.Tensor,
    history_lens: torch.Tensor,
) -> None:
    Z = seq_offsets.size(0) - 1
    L, H, Dim = q_seq.shape
    max_seq_len = N
    block_m, block_n = (64, 32)

    grid = ( triton.cdiv(max_num_candidate, block_m), Z * H )
    _paged_hstu_attn_fwd[grid](
        Q_seq=q_seq,
        K_seq=k_seq,
        V_seq=v_seq,
        KV_cache=kv_cache,
        kv_cache_indices=kv_cache_indices,
        kv_cache_indptr=kv_cache_indptr,
        kv_cache_tail_lens=kv_cache_tail_lens,
        seq_offsets=seq_offsets,
        num_candidates=num_candidates,
        history_lens=history_lens,
        Out=output,
        stride_qm=q_seq.stride(0),
        stride_qh=q_seq.stride(1),
        stride_kn=k_seq.stride(0),
        stride_kh=k_seq.stride(1),
        stride_vn=v_seq.stride(0),
        stride_vh=v_seq.stride(1),
        stride_om=output.stride(0),
        stride_oh=output.stride(1),
        stride_kv_page=kv_cache.stride(0),
        stride_k_v=kv_cache.stride(1),
        alpha=alpha,
        MAX_SEQ_LEN=N,
        H=H,
        CACHE_PAGE_SIZE=kv_cache_page_size,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_DIM=Dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return None

def native_triton_history_hstu_mha(
    N: int,
    alpha: float,
    q_seq: torch.Tensor,
    output: torch.Tensor,
    max_delta_history_length: int,
    kv_cache: torch.Tensor,
    kv_cache_indices: torch.Tensor,
    kv_cache_indptr: torch.Tensor,
    kv_cache_tail_lens: torch.Tensor,
    kv_cache_page_size: int,
    seq_offsets: torch.Tensor,
    num_candidates: torch.Tensor,
    history_lens: torch.Tensor,
) -> None:
    Z = seq_offsets.size(0) - 1
    L, H, Dim = q_seq.shape
    max_seq_len = N
    block_m, block_n = (64, 32)

    grid = (triton.cdiv(max_delta_history_length, block_m), Z * H)
    _history_paged_hstu_attn_fwd[grid](
        Q_seq=q_seq,
        KV_cache=kv_cache,
        kv_cache_indices=kv_cache_indices,
        kv_cache_indptr=kv_cache_indptr,
        kv_cache_tail_lens=kv_cache_tail_lens,
        seq_offsets=seq_offsets,
        num_candidates=num_candidates,
        history_lens=history_lens,
        Out=output,
        stride_qm=q_seq.stride(0),
        stride_qh=q_seq.stride(1),
        stride_om=output.stride(0),
        stride_oh=output.stride(1),
        stride_kv_page=kv_cache.stride(0),
        stride_k_v=kv_cache.stride(1),
        alpha=alpha,
        MAX_SEQ_LEN=N,
        H=H,
        CACHE_PAGE_SIZE=kv_cache_page_size,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_DIM=Dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return None

def triton_candidate_hstu_mha(
    N: int,
    alpha: float,
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    output: torch.Tensor,
    max_num_candidate: int,
    kv_cache: torch.Tensor,
    kv_cache_indices: torch.Tensor,
    kv_cache_indptr: torch.Tensor,
    kv_cache_tail_lens: torch.Tensor,
    kv_cache_page_size: int,
    seq_offsets: torch.Tensor,
    num_candidates: torch.Tensor,
    history_lens: torch.Tensor,
    triton_cc: bool = False,   # not used
) -> None:
    q_seq = switch_to_contiguous_if_needed(q_seq)
    k_seq = switch_to_contiguous_if_needed(k_seq)
    v_seq = switch_to_contiguous_if_needed(v_seq)
    output = switch_to_contiguous_if_needed(output)

    return native_triton_candidate_hstu_mha(
        N=N,
        alpha=alpha,
        q_seq=q_seq,
        k_seq=k_seq,
        v_seq=v_seq,
        output=output,
        max_num_candidate=max_num_candidate,
        kv_cache=kv_cache,
        kv_cache_indices=kv_cache_indices,
        kv_cache_indptr=kv_cache_indptr,
        kv_cache_tail_lens=kv_cache_tail_lens,
        kv_cache_page_size=kv_cache_page_size,
        seq_offsets=seq_offsets,
        num_candidates=num_candidates,
        history_lens=history_lens,
    )

def triton_history_hstu_mha(
    N: int,
    alpha: float,
    q_seq: torch.Tensor,
    output: torch.Tensor,
    max_delta_history_length: int,
    kv_cache: torch.Tensor,
    kv_cache_indices: torch.Tensor,
    kv_cache_indptr: torch.Tensor,
    kv_cache_tail_lens: torch.Tensor,
    kv_cache_page_size: int,
    seq_offsets: torch.Tensor,
    num_candidates: torch.Tensor,
    history_lens: torch.Tensor,
    triton_cc: bool = False,   # not used
) -> None:
    q_seq = switch_to_contiguous_if_needed(q_seq)
    output = switch_to_contiguous_if_needed(output)

    return native_triton_history_hstu_mha(
        N=N,
        alpha=alpha,
        q_seq=q_seq,
        output=output,
        max_delta_history_length=max_delta_history_length,
        kv_cache=kv_cache,
        kv_cache_indices=kv_cache_indices,
        kv_cache_indptr=kv_cache_indptr,
        kv_cache_tail_lens=kv_cache_tail_lens,
        kv_cache_page_size=kv_cache_page_size,
        seq_offsets=seq_offsets,
        num_candidates=num_candidates,
        history_lens=history_lens,
    )