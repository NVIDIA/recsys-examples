# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
HSTU Context-Parallel attention wrapper (Slice 3 — multi-GPU forward).

Public entry point: `hstu_attn_varlen_cp_func`.

What this module provides for v0
================================
- A drop-in callable users can swap in for `hstu_attn_varlen_func`.
  Signature mirrors the installed kernel exactly plus four CP arguments.
- Hard guards rejecting v0+ modes with `ValueError` (per SPEC §2 / plan T3.1).
- `cp_size == 1` short-circuit: direct delegation to `hstu_attn_varlen_func`,
  no autograd wrap, no comm. Guards still run uniformly per plan T3.1
  (cost is a few Python conditionals — well within plan §Global rule 3
  cp=1 perf budget).
- DualChunkSwap dispatch helper `get_batch_on_this_cp_rank_for_hstu`
  (pure permutation; T3.2) plus testing-only `gather_global_from_cp_rank`.
- Multi-GPU forward path is implemented (T3.3): single CUDA stream,
  sequential ring P2P via `dist.batch_isend_irecv`, plain-sum reduction
  in fp32 across the (rank, step) classification grid (diagonal /
  lower-triangle / upper-triangle).
- Backward (T4.2): explicit reverse-direction ring. dQ stays local; dK/dV
  partials ride the reverse ring back to their owning rank with
  copy-on-first-receive / add-after semantics.

What this module does NOT do (v0 / SPEC §2)
===========================================
- Sliding-causal, `rab`, heterogeneous mask (`num_contexts`,
  `num_targets`, `target_group_size > 1`), FP8, KV-cache, Ulysses,
  comm/compute overlap (Slice 5), training-loop integration (Slice 6).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from .hstu_attn_interface import hstu_attn_varlen_func

__all__ = [
    "hstu_attn_varlen_cp_func",
    "get_batch_on_this_cp_rank_for_hstu",
    "gather_global_from_cp_rank",
    "GuardError",
]


# ----------------------------------------------------------------------------
# Errors. We reuse `ValueError` for guard rejections (matches SPEC §7) but
# expose a typed alias so tests / callers can `except GuardError`.
# ----------------------------------------------------------------------------
class GuardError(ValueError):
    """Raised when an input doesn't satisfy v0 contract (SPEC §2)."""


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
_SUPPORTED_HEAD_DIMS = (32, 64, 128, 256)
_SUPPORTED_WINDOW_SIZE = (-1, 0)  # pure causal only
_SPEC_REF = "see SPEC §2 (out-of-scope) and plan T3.1 (hard-guard list)"


# ----------------------------------------------------------------------------
# DualChunkSwap dispatch helper (T3.2).
#
# Maps a global packed batch onto the local shard owned by `(cp_rank, cp_size)`.
# Pure permutation — no `torch.distributed` calls.
#
# Per sequence of length L (must be divisible by 2*cp_size), each chunk is
# size c = L / (2*cp_size). Rank r owns chunks {r, 2*cp_size-1-r}; local
# layout per sample is [chunk_r, chunk_(2cp-1-r)].
# ----------------------------------------------------------------------------
def get_batch_on_this_cp_rank_for_hstu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_global: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]
]:
    """Gather rank `cp_rank`'s DualChunkSwap chunks from a global batch.

    Args:
        q, k, v: global packed tensors, shape (total_tokens, num_heads, head_dim).
        cu_seqlens_global: int32 (B+1,) global cu_seqlens.
        cp_size: number of CP ranks.
        cp_rank: this rank's id, in `[0, cp_size)`.

    Returns:
        q_local, k_local, v_local: per-rank shards in DualChunkSwap order.
        cu_seqlens_local: int32 (B+1,), each sample of length 2 * (per-sample chunk size).
        local_to_global: int64 (sum_b 2*c_b,), maps local row → global row index
          (used by `gather_global_from_cp_rank` and by the multi-GPU output
          scatter logic).
        chunk_sizes: list of per-sample chunk sizes c_b.

    Raises:
        GuardError: if `cp_size < 1`, `cp_rank` out of range, or any per-sample
          seqlen not divisible by `2 * cp_size`.
    """
    if cp_size < 1:
        raise GuardError(f"cp_size must be ≥ 1; got {cp_size}")
    if not 0 <= cp_rank < cp_size:
        raise GuardError(f"cp_rank must be in [0, {cp_size}); got {cp_rank}")
    if cp_size == 1:
        # Degenerate: every rank holds the entire global batch. Return as-is.
        idx = torch.arange(q.shape[0], device=q.device, dtype=torch.long)
        seqlens = (cu_seqlens_global[1:] - cu_seqlens_global[:-1]).tolist()
        return q, k, v, cu_seqlens_global, idx, list(seqlens)

    chunks_per_seq = 2 * cp_size
    own = (cp_rank, chunks_per_seq - 1 - cp_rank)
    device = q.device

    seqlens_global = (cu_seqlens_global[1:] - cu_seqlens_global[:-1]).tolist()
    cu_global_list = cu_seqlens_global.tolist()

    rows: list[torch.Tensor] = []
    local_lens: list[int] = []
    chunk_sizes: list[int] = []
    for b, L in enumerate(seqlens_global):
        if L % chunks_per_seq != 0:
            raise GuardError(
                f"sample {b} seqlen {L} is not divisible by 2*cp_size={chunks_per_seq}"
            )
        c_b = L // chunks_per_seq
        chunk_sizes.append(c_b)
        base = cu_global_list[b]
        for chunk_id in own:
            rows.append(
                torch.arange(
                    base + chunk_id * c_b, base + (chunk_id + 1) * c_b, device=device
                )
            )
        local_lens.append(2 * c_b)

    local_to_global = torch.cat(rows)
    q_local = q[local_to_global].contiguous()
    k_local = k[local_to_global].contiguous()
    v_local = v[local_to_global].contiguous()
    cu_local = (
        torch.tensor([0] + local_lens, dtype=torch.int32, device=device).cumsum(0).int()
    )
    return q_local, k_local, v_local, cu_local, local_to_global, chunk_sizes


def gather_global_from_cp_rank(
    local: torch.Tensor,
    local_to_global: torch.Tensor,
    *,
    global_total_tokens: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse of `get_batch_on_this_cp_rank_for_hstu` for testing.

    Scatters a single rank's local tensor back into a globally-shaped buffer.
    In production, this gather runs across CP ranks (e.g. an all-gather);
    here it is local-only and used by single-process tests.
    """
    if out is None:
        shape = (global_total_tokens, *local.shape[1:])
        out = torch.zeros(shape, dtype=local.dtype, device=local.device)
    out[local_to_global] += local
    return out


# ----------------------------------------------------------------------------
# Hard guards (T3.1). Returns silently if all ok; raises `GuardError` otherwise.
# ----------------------------------------------------------------------------
def _enforce_v0_contract(
    *,
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size: tuple[int, int],
    rab: Optional[torch.Tensor],
    has_drab: bool,
    kv_cache: Optional[torch.Tensor],
    page_offsets: Optional[torch.Tensor],
    page_ids: Optional[torch.Tensor],
    last_page_lens: Optional[torch.Tensor],
    func: Optional[torch.Tensor],
    quant_mode: Optional[int],
    cp_size: int,
) -> None:
    # 1-2. Heterogeneous mask
    if num_contexts is not None:
        raise GuardError(f"num_contexts is not supported in v0 ({_SPEC_REF})")
    if num_targets is not None:
        raise GuardError(f"num_targets is not supported in v0 ({_SPEC_REF})")
    # 3. target_group_size != 1 (v0 supports only the default size 1)
    if target_group_size != 1:
        raise GuardError(
            f"target_group_size != 1 is not supported in v0 (got {target_group_size}; {_SPEC_REF})"
        )
    # 4. window_size != (-1, 0)
    ws = tuple(window_size)
    if ws != _SUPPORTED_WINDOW_SIZE:
        raise GuardError(
            f"window_size={ws} not supported in v0; only causal (-1, 0) ({_SPEC_REF})"
        )
    # 5. rab / has_drab
    if rab is not None:
        raise GuardError(f"rab is not supported in v0 ({_SPEC_REF})")
    if has_drab:
        raise GuardError(f"has_drab=True is not supported in v0 ({_SPEC_REF})")
    # 6-9. KV cache + paging
    if kv_cache is not None:
        raise GuardError(f"kv_cache is not supported in v0 ({_SPEC_REF})")
    if page_offsets is not None:
        raise GuardError(f"page_offsets is not supported in v0 ({_SPEC_REF})")
    if page_ids is not None:
        raise GuardError(f"page_ids is not supported in v0 ({_SPEC_REF})")
    if last_page_lens is not None:
        raise GuardError(f"last_page_lens is not supported in v0 ({_SPEC_REF})")
    # 10. func (post-attention hook)
    if func is not None:
        raise GuardError(f"func hook is not supported in v0 ({_SPEC_REF})")
    # 11. quant_mode (only `-1` (== off) is allowed; both `None` and any other
    #     int are rejected so users can't accidentally bypass quantisation
    #     guards by leaving the kwarg unset on a build that defaults to None).
    if quant_mode is None or quant_mode != -1:
        raise GuardError(
            f"quant_mode={quant_mode!r} not supported in v0; only -1 ({_SPEC_REF})"
        )
    # 12. seqused_q/k (the kernel takes them but v0 wrapper doesn't pass through)
    if seqused_q is not None:
        raise GuardError(f"seqused_q is not supported in v0 ({_SPEC_REF})")
    if seqused_k is not None:
        raise GuardError(f"seqused_k is not supported in v0 ({_SPEC_REF})")
    # 13. head_dim
    if q.dim() != 3:
        raise GuardError(
            f"q must be 3-D (total_tokens, num_heads, head_dim); got {q.dim()}-D"
        )
    head_dim = q.shape[-1]
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise GuardError(
            f"head_dim={head_dim} not in supported set {_SUPPORTED_HEAD_DIMS} ({_SPEC_REF})"
        )
    # Self-attention contract: cu_seqlens_q must equal cu_seqlens_k for any
    # cp_size (HSTU is self-attention; the in-tree kernel ignores cu_seqlens_k
    # for true varlen self-attn but the wrapper enforces the contract so a
    # mismatched call is caught early).
    if not torch.equal(cu_seqlens_q, cu_seqlens_k):
        raise GuardError(
            "cu_seqlens_q must equal cu_seqlens_k (HSTU is self-attention only in v0)"
        )
    # DualChunkSwap divisibility — only meaningful when chunking actually
    # happens (cp_size > 1).
    if cp_size > 1:
        chunks_per_seq = 2 * cp_size
        seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        for b, L in enumerate(seqlens):
            if L % chunks_per_seq != 0:
                raise GuardError(
                    f"sample {b}: seqlen {L} not divisible by 2*cp_size={chunks_per_seq} "
                    f"(DualChunkSwap requirement; pre-pad in caller)"
                )


# ----------------------------------------------------------------------------
# Per-tile slice helpers (varlen-aware). These mirror the validated PoC at
# `examples/hstu/cp/poc_simrank_sim.py`. Pure Python/torch — no fused CUDA
# (per SPEC §2 v0 contract; CUDA fusion is a Slice 5 follow-up if profiling
# shows it matters).
# ----------------------------------------------------------------------------
def _chunk_sizes_from_cu(cu_local: torch.Tensor) -> list[int]:
    """Each sample's chunk size c_b given the local layout (2 chunks per sample,
    total 2*c_b)."""
    seqlens = (cu_local[1:] - cu_local[:-1]).tolist()
    return [s // 2 for s in seqlens]


def _zero_second_half_per_sample(
    t: torch.Tensor, cu_local: torch.Tensor, chunk_sizes: list[int]
) -> torch.Tensor:
    """Zero the second-half (chunk_(2cp-1-src)) slot of each sample's local layout."""
    out = t.clone()
    cu = cu_local.tolist()
    for b, c_b in enumerate(chunk_sizes):
        start = cu[b] + c_b
        end = cu[b + 1]
        out[start:end] = 0
    return out


def _select_second_half_per_sample(
    t: torch.Tensor, cu_local: torch.Tensor, chunk_sizes: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the second-half slot per sample. Returns (concat tensor, cu_seqlens_half)."""
    cu = cu_local.tolist()
    parts: list[torch.Tensor] = []
    half_lens: list[int] = []
    for b, c_b in enumerate(chunk_sizes):
        start = cu[b] + c_b
        end = cu[b + 1]
        parts.append(t[start:end])
        half_lens.append(c_b)
    out = torch.cat(parts, dim=0).contiguous()
    cu_half = (
        torch.tensor([0] + half_lens, dtype=torch.int32, device=t.device)
        .cumsum(0)
        .int()
    )
    return out, cu_half


def _scatter_second_half_per_sample(
    out_local: torch.Tensor,
    partial_half: torch.Tensor,
    cu_local: torch.Tensor,
    chunk_sizes: list[int],
) -> None:
    """In-place add `partial_half` (B*c_b rows concatenated) into `out_local`'s
    per-sample second-half slots."""
    cu = cu_local.tolist()
    cum_half = 0
    for b, c_b in enumerate(chunk_sizes):
        start = cu[b] + c_b
        end = cu[b + 1]
        out_local[start:end] += partial_half[cum_half : cum_half + c_b]
        cum_half += c_b


# ----------------------------------------------------------------------------
# Per-tile kernel calls. All three flavours pass the GLOBAL `scaling_seqlen`
# so partial outputs across ring steps share the same normaliser (plain-sum
# remains correct).
# ----------------------------------------------------------------------------
def _diag_call(
    q_loc, k_loc, v_loc, cu_loc, local_max, scaling_seqlen, alpha
) -> torch.Tensor:
    return hstu_attn_varlen_func(
        q=q_loc,
        k=k_loc,
        v=v_loc,
        cu_seqlens_q=cu_loc,
        cu_seqlens_k=cu_loc,
        max_seqlen_q=local_max,
        max_seqlen_k=local_max,
        scaling_seqlen=scaling_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=alpha,
    )


def _lower_call(
    q_loc, k_pad, v_pad, cu_loc, local_max, scaling_seqlen, alpha
) -> torch.Tensor:
    return hstu_attn_varlen_func(
        q=q_loc,
        k=k_pad,
        v=v_pad,
        cu_seqlens_q=cu_loc,
        cu_seqlens_k=cu_loc,
        max_seqlen_q=local_max,
        max_seqlen_k=local_max,
        scaling_seqlen=scaling_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, -1),
        alpha=alpha,
    )


def _upper_call(
    q_half,
    k_full,
    v_full,
    cu_q_half,
    cu_full,
    half_max,
    local_max,
    scaling_seqlen,
    alpha,
) -> torch.Tensor:
    return hstu_attn_varlen_func(
        q=q_half,
        k=k_full,
        v=v_full,
        cu_seqlens_q=cu_q_half,
        cu_seqlens_k=cu_full,
        max_seqlen_q=half_max,
        max_seqlen_k=local_max,
        scaling_seqlen=scaling_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, -1),
        alpha=alpha,
    )


# ----------------------------------------------------------------------------
# Ring P2P helper (sequential single-stream — Slice 5 adds two-stream overlap).
# ----------------------------------------------------------------------------
def _ring_send_recv_kv(
    cur_k: torch.Tensor,
    cur_v: torch.Tensor,
    recv_k: torch.Tensor,
    recv_v: torch.Tensor,
    *,
    cp_group: dist.ProcessGroup,
    cp_global_ranks: list[int],
    cp_rank: int,
    cp_size: int,
    direction: str = "forward",
) -> list[dist.Work]:
    """Issue P2P send + recv for one ring step.

    `direction="forward"`: send to `(rank+1)`, recv from `(rank-1)`.
    `direction="backward"`: send to `(rank-1)`, recv from `(rank+1)`. Used by
    T4.2 (multi-GPU backward) to send dKV partials home along the reverse
    ring. Note that for backward, the tensors typically named `cur_k/cur_v`
    actually carry dK/dV gradients — the helper is direction-agnostic.

    Uses `batch_isend_irecv` to avoid the deadlock pattern of naive isend/irecv
    pairs. Returns the list of `Work` handles; caller must call `.wait()`
    before consuming `recv_k`/`recv_v`.
    """
    if direction == "forward":
        dst = cp_global_ranks[(cp_rank + 1) % cp_size]
        src = cp_global_ranks[(cp_rank - 1) % cp_size]
    elif direction == "backward":
        dst = cp_global_ranks[(cp_rank - 1) % cp_size]
        src = cp_global_ranks[(cp_rank + 1) % cp_size]
    else:
        raise ValueError(
            f"direction must be 'forward' or 'backward'; got {direction!r}"
        )
    ops = [
        dist.P2POp(dist.isend, cur_k, dst, group=cp_group),
        dist.P2POp(dist.isend, cur_v, dst, group=cp_group),
        dist.P2POp(dist.irecv, recv_k, src, group=cp_group),
        dist.P2POp(dist.irecv, recv_v, src, group=cp_group),
    ]
    return dist.batch_isend_irecv(ops)


# ----------------------------------------------------------------------------
# T3.3: multi-GPU forward. Single CUDA stream, sequential ring P2P.
# ----------------------------------------------------------------------------
def _multi_gpu_forward(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    *,
    max_seqlen_q_global: int,
    scaling_seqlen: int,
    alpha: float,
    cp_group: dist.ProcessGroup,
    cp_global_ranks: list[int],
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Run the (rank, step) classification grid as a real multi-GPU ring.

    `q_local, k_local, v_local, cu_seqlens_local` are this rank's DualChunkSwap
    shard (already produced by `get_batch_on_this_cp_rank_for_hstu` upstream).
    `max_seqlen_q_global` is the unsharded global max; we compute `local_max`
    internally. `scaling_seqlen` is the global `1/N` divisor (must NOT change
    across ring steps — that's why every per-tile call passes the same value).

    Reduction is in fp32 (per SPEC §2). The returned tensor is cast back to
    `q_local.dtype` on exit.
    """
    local_max = (
        max_seqlen_q_global // cp_size
    )  # 2 chunks per sample → local len = global / cp_size
    half_max = local_max // 2  # one chunk per sample
    chunk_sizes = _chunk_sizes_from_cu(cu_seqlens_local)

    # Ping-pong KV buffers. Recv buffer must be same shape as send (DualChunkSwap
    # gives every rank identical local total tokens, even under varlen).
    cur_k = k_local
    cur_v = v_local
    recv_k = torch.empty_like(k_local)
    recv_v = torch.empty_like(v_local)

    # Output accumulator in fp32 for numerical stability across cp_size adds.
    out_local = torch.zeros_like(q_local, dtype=torch.float32)

    for step in range(cp_size):
        # 1. Issue next-step KV exchange (skip on last step).
        reqs: list[dist.Work] = []
        if step < cp_size - 1:
            reqs = _ring_send_recv_kv(
                cur_k,
                cur_v,
                recv_k,
                recv_v,
                cp_group=cp_group,
                cp_global_ranks=cp_global_ranks,
                cp_rank=cp_rank,
                cp_size=cp_size,
            )

        # 2. Compute on the current KV (still owned).
        if step == 0:
            partial = _diag_call(
                q_local,
                cur_k,
                cur_v,
                cu_seqlens_local,
                local_max,
                scaling_seqlen,
                alpha,
            )
            out_local += partial.float()
        elif step <= cp_rank:
            # Lower-tri: zero peer's second-half (chunk_(2cp-1-src)) so
            # K_len == Q_len; SiLU(α Q · 0) · 0 contributes 0.
            k_pad = _zero_second_half_per_sample(cur_k, cu_seqlens_local, chunk_sizes)
            v_pad = _zero_second_half_per_sample(cur_v, cu_seqlens_local, chunk_sizes)
            partial = _lower_call(
                q_local,
                k_pad,
                v_pad,
                cu_seqlens_local,
                local_max,
                scaling_seqlen,
                alpha,
            )
            out_local += partial.float()
        else:
            # Upper-tri: Q's second-half (chunk_(2cp-1-rank)) × peer's full K.
            q_half, cu_q_half = _select_second_half_per_sample(
                q_local, cu_seqlens_local, chunk_sizes
            )
            partial_half = _upper_call(
                q_half,
                cur_k,
                cur_v,
                cu_q_half,
                cu_seqlens_local,
                half_max,
                local_max,
                scaling_seqlen,
                alpha,
            )
            _scatter_second_half_per_sample(
                out_local, partial_half.float(), cu_seqlens_local, chunk_sizes
            )

        # 3. Wait for next-step KV to arrive before overwriting `cur_*`.
        for r in reqs:
            r.wait()

        # 4. Swap buffers for the next iteration.
        if step < cp_size - 1:
            cur_k, recv_k = recv_k, cur_k
            cur_v, recv_v = recv_v, cur_v

    return out_local.to(q_local.dtype)


# ----------------------------------------------------------------------------
# T4.2: multi-GPU backward. Reverse-direction ring; dQ stays local; dK/dV
# partials ride the reverse ring back to their owning rank with copy-on-first /
# add-after semantics.
# ----------------------------------------------------------------------------
def _per_tile_partial_grads(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    max_q: int,
    max_k: int,
    scaling_seqlen: int,
    alpha: float,
    window_size: tuple[int, int],
    dout_partial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a per-tile forward with autograd on the LOCAL (detached + cloned)
    inputs, then call torch.autograd.grad to extract partial dQ, dK, dV.

    The wrapping of `hstu_attn_varlen_func` already has its own
    `autograd.Function`, so this is just a thin re-execution that propagates
    `dout_partial` back through it.
    """
    q_in = q_input.detach().clone().requires_grad_(True)
    k_in = k_input.detach().clone().requires_grad_(True)
    v_in = v_input.detach().clone().requires_grad_(True)
    out = hstu_attn_varlen_func(
        q=q_in,
        k=k_in,
        v=v_in,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        scaling_seqlen=scaling_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=window_size,
        alpha=alpha,
    )
    dq, dk, dv = torch.autograd.grad(out, (q_in, k_in, v_in), dout_partial)
    return dq.detach(), dk.detach(), dv.detach()


def _multi_gpu_backward(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    dout_local: torch.Tensor,
    *,
    max_seqlen_q_global: int,
    scaling_seqlen: int,
    alpha: float,
    cp_group: dist.ProcessGroup,
    cp_global_ranks: list[int],
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reverse-direction-ring backward for HSTU CP forward.

    Algorithm:
      1. Initialise dq_local = 0, dk_local = 0, dv_local = 0.
      2. For each forward step `i` in 0..cp_size-1, redo the per-tile forward
         with autograd-enabled inputs and use `torch.autograd.grad` to extract
         (dq_partial, dk_partial, dv_partial).
         dq_partial accumulates locally; dk_partial / dv_partial are gradients
         for the KV that was held at this step (peer rank src=(rank-i)%cp_size).
      3. dKV partials ride the **reverse** ring back to their owners:
         - At backward iteration 0, the rank holds its OWN dKV (step==0
           diagonal tile). Add to local dk/dv directly.
         - For iteration j>=1, the rank sends the dKV computed at forward
           step j (which belongs to peer src=(rank-j)%cp) to that peer
           via reverse ring (dst = (rank - j) % cp_size, equivalently
           dst = src). Receives from rank `(rank + j) % cp_size` the dKV
           that they computed for OUR K/V at their forward step j.
         - The received dKV adds to local dk_local, dv_local.

    Returned dtypes match the forward inputs (cast back from the fp32
    accumulators).
    """
    2 * cp_size
    local_max = max_seqlen_q_global // cp_size
    half_max = local_max // 2
    chunk_sizes = _chunk_sizes_from_cu(cu_seqlens_local)

    # We need the SAME KV stream the forward saw at each step. Re-run the
    # forward ring locally (read-only) to reconstruct kv_at_step[i]. Cheap
    # because comm dominates and we already paid that cost in forward.
    cur_k = k_local
    cur_v = v_local
    recv_k = torch.empty_like(k_local)
    recv_v = torch.empty_like(v_local)

    # fp32 accumulators (per SPEC §2 "Reduction in fp32").
    dq_acc = torch.zeros_like(q_local, dtype=torch.float32)
    dk_acc = torch.zeros_like(k_local, dtype=torch.float32)
    dv_acc = torch.zeros_like(v_local, dtype=torch.float32)

    # We collect (step, dk_partial, dv_partial) so that after the forward-pass
    # backward computation, we send each dKV back to its rightful owner via
    # the reverse ring. dq is purely local — accumulated inline.
    # dk/dv at forward step 0 belong to rank itself; add directly.
    dkv_to_send: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    for step in range(cp_size):
        # Issue next-step KV exchange (forward direction) so cur_k/v matches
        # what was used in forward.
        reqs: list[dist.Work] = []
        if step < cp_size - 1:
            reqs = _ring_send_recv_kv(
                cur_k,
                cur_v,
                recv_k,
                recv_v,
                cp_group=cp_group,
                cp_global_ranks=cp_global_ranks,
                cp_rank=cp_rank,
                cp_size=cp_size,
                direction="forward",
            )

        # Compute per-tile partial grads.
        if step == 0:
            # Diagonal: full Q × full local KV, causal.
            dq_p, dk_p, dv_p = _per_tile_partial_grads(
                q_local,
                cur_k,
                cur_v,
                cu_seqlens_local,
                cu_seqlens_local,
                max_q=local_max,
                max_k=local_max,
                scaling_seqlen=scaling_seqlen,
                alpha=alpha,
                window_size=(-1, 0),
                dout_partial=dout_local,
            )
            dq_acc += dq_p.float()
            # Diagonal dKV is ours; add to local accumulator immediately.
            dk_acc += dk_p.float()
            dv_acc += dv_p.float()
        elif step <= cp_rank:
            # Lower-tri: full Q × peer K first-half (zero-padded), no causal.
            k_pad = _zero_second_half_per_sample(cur_k, cu_seqlens_local, chunk_sizes)
            v_pad = _zero_second_half_per_sample(cur_v, cu_seqlens_local, chunk_sizes)
            dq_p, dk_p, dv_p = _per_tile_partial_grads(
                q_local,
                k_pad,
                v_pad,
                cu_seqlens_local,
                cu_seqlens_local,
                max_q=local_max,
                max_k=local_max,
                scaling_seqlen=scaling_seqlen,
                alpha=alpha,
                window_size=(-1, -1),
                dout_partial=dout_local,
            )
            dq_acc += dq_p.float()
            # The padded second-half slots received zero contribution in
            # forward, so dK/dV at those positions is exactly 0. Zero them
            # again for safety before sending back.
            dk_p_for_peer = _zero_second_half_per_sample(
                dk_p, cu_seqlens_local, chunk_sizes
            )
            dv_p_for_peer = _zero_second_half_per_sample(
                dv_p, cu_seqlens_local, chunk_sizes
            )
            dkv_to_send.append((step, dk_p_for_peer, dv_p_for_peer))
        else:
            # Upper-tri: Q's second-half × full peer K, no mask.
            q_half, cu_q_half = _select_second_half_per_sample(
                q_local, cu_seqlens_local, chunk_sizes
            )
            dout_half, _ = _select_second_half_per_sample(
                dout_local, cu_seqlens_local, chunk_sizes
            )
            dq_half_p, dk_p, dv_p = _per_tile_partial_grads(
                q_half,
                cur_k,
                cur_v,
                cu_q_half,
                cu_seqlens_local,
                max_q=half_max,
                max_k=local_max,
                scaling_seqlen=scaling_seqlen,
                alpha=alpha,
                window_size=(-1, -1),
                dout_partial=dout_half,
            )
            # Scatter dq_half_p into rank's second-half slots.
            _scatter_second_half_per_sample(
                dq_acc, dq_half_p.float(), cu_seqlens_local, chunk_sizes
            )
            # dk_p, dv_p are full local-shape (matching cur_k/v). Send back.
            dkv_to_send.append((step, dk_p, dv_p))

        # Wait for next-step KV (forward ring) to arrive before swap.
        for r in reqs:
            r.wait()
        if step < cp_size - 1:
            cur_k, recv_k = recv_k, cur_k
            cur_v, recv_v = recv_v, cur_v

    # 3. Reverse-ring exchange of dKV partials. For each (step, dk_p, dv_p)
    # in `dkv_to_send`, dk_p / dv_p belong to peer src = (rank - step) % cp.
    # We send to dst=src and receive from peer that computed grads for our
    # K/V at THEIR step (which is our same step index).
    #
    # Why simultaneous: at step `i`, every rank computed grads-for-its-peer
    # at the same `i`. So each rank's `i`-step partial → dst=(rank-i)%cp;
    # each rank's `i`-step incoming → from src=(rank+i)%cp.
    recv_dk = torch.empty_like(k_local)
    recv_dv = torch.empty_like(v_local)
    for step, dk_p, dv_p in dkv_to_send:
        send_dst = cp_global_ranks[(cp_rank - step) % cp_size]
        recv_src = cp_global_ranks[(cp_rank + step) % cp_size]
        ops = [
            dist.P2POp(dist.isend, dk_p.contiguous(), send_dst, group=cp_group),
            dist.P2POp(dist.isend, dv_p.contiguous(), send_dst, group=cp_group),
            dist.P2POp(dist.irecv, recv_dk, recv_src, group=cp_group),
            dist.P2POp(dist.irecv, recv_dv, recv_src, group=cp_group),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()
        dk_acc += recv_dk.float()
        dv_acc += recv_dv.float()

    return (
        dq_acc.to(q_local.dtype),
        dk_acc.to(k_local.dtype),
        dv_acc.to(v_local.dtype),
    )


# ----------------------------------------------------------------------------
# Multi-GPU autograd Function. Forward (T3.3) and backward (T4.2) implemented.
# ----------------------------------------------------------------------------
class _HSTUVarlenCPFunc(torch.autograd.Function):
    """Multi-GPU forward+backward driver."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scaling_seqlen,
        alpha,
        cp_group,
        cp_global_ranks,
        cp_stream,
        cp_comm_type,
    ):
        # `cp_stream` is reserved (Slice 5 will use it for two-stream overlap);
        # v0 is single-stream so we ignore it here.
        del cp_stream  # unused in v0
        if cp_comm_type != "p2p":
            raise GuardError(
                f"cp_comm_type={cp_comm_type!r} not supported in v0; only 'p2p'"
            )
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)

        out = _multi_gpu_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            max_seqlen_q_global=max_seqlen_q,
            scaling_seqlen=scaling_seqlen,
            alpha=alpha,
            cp_group=cp_group,
            cp_global_ranks=cp_global_ranks,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )

        # Save for backward (T4.2).
        ctx.save_for_backward(q, k, v, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.scaling_seqlen = scaling_seqlen
        ctx.alpha = alpha
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        return out

    @staticmethod
    def backward(ctx, dout):  # type: ignore[override]
        q, k, v, cu_seqlens_q, _cu_seqlens_k = ctx.saved_tensors
        return (
            *_multi_gpu_backward(
                q,
                k,
                v,
                cu_seqlens_q,
                dout,
                max_seqlen_q_global=ctx.max_seqlen_q,
                scaling_seqlen=ctx.scaling_seqlen,
                alpha=ctx.alpha,
                cp_group=ctx.cp_group,
                cp_global_ranks=ctx.cp_global_ranks,
                cp_rank=ctx.cp_rank,
                cp_size=ctx.cp_size,
            ),
            # No gradients for non-Tensor / metadata args. Forward took:
            #   q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            #   scaling_seqlen, alpha, cp_group, cp_global_ranks, cp_stream,
            #   cp_comm_type
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # scaling_seqlen
            None,  # alpha
            None,  # cp_group
            None,  # cp_global_ranks
            None,  # cp_stream
            None,  # cp_comm_type
        )


# ----------------------------------------------------------------------------
# Public entry point.
#
# Signature mirrors the installed `hstu_attn_varlen_func` exactly (per
# `examples/hstu/test/cp/conftest.py::CANONICAL_HSTU_PARAMS`) plus four CP
# arguments. Body order:
#   1. Determine cp_size from cp_group.
#   2. Run the 13-item hard-guard battery uniformly (cp=1 included).
#   3. cp_size == 1 ⇒ direct delegation to `hstu_attn_varlen_func`.
#   4. cp_size > 1 ⇒ dispatch via `_HSTUVarlenCPFunc.apply` (T3.3 forward
#      + T4.2 backward).
# ----------------------------------------------------------------------------
def hstu_attn_varlen_cp_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    scaling_seqlen: int,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int = 1,
    window_size: tuple[int, int] = (-1, -1),
    alpha: float = 1.0,
    rab: Optional[torch.Tensor] = None,
    has_drab: bool = False,
    kv_cache: Optional[torch.Tensor] = None,
    page_offsets: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    last_page_lens: Optional[torch.Tensor] = None,
    func: Optional[torch.Tensor] = None,
    quant_mode: Optional[int] = -1,
    *,
    cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    cp_global_ranks: Optional[list[int]] = None,
    cp_stream: Optional[torch.cuda.Stream] = None,
    cp_comm_type: str = "p2p",
) -> torch.Tensor:
    """HSTU varlen attention with optional context parallelism.

    See SPEC §1-§2 for v0 scope. When `cp_group is None` or the group has size 1,
    the call short-circuits to the production single-GPU `hstu_attn_varlen_func`.
    Otherwise the CP path runs (plan T3.3 forward / T4.2 backward).
    """
    # 1. Determine cp_size up front.
    if cp_group is None:
        cp_size = 1
    else:
        cp_size = dist.get_world_size(cp_group)

    # 2. Hard guards. Applied UNIFORMLY at both cp=1 and cp>1 paths so the
    #    contract is the same regardless of CP topology. The cost is a few
    #    Python conditionals (well under any kernel-side overhead).
    _enforce_v0_contract(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=target_group_size,
        window_size=window_size,
        rab=rab,
        has_drab=has_drab,
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
        func=func,
        quant_mode=quant_mode,  # leave None as None so guard fires
        cp_size=cp_size,
    )
    if max_seqlen_q != max_seqlen_k:
        raise GuardError(
            f"v0 supports self-attention only; got max_seqlen_q={max_seqlen_q} "
            f"!= max_seqlen_k={max_seqlen_k}"
        )

    # 3. cp_size == 1 short-circuit. After guards have rejected non-v0 modes,
    #    the call is just the bare in-tree kernel with the v0-only kwargs.
    if cp_size == 1:
        return hstu_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=window_size,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
        )

    # 4. Multi-GPU CP path. cp_global_ranks defaults to the absolute world-rank
    #    IDs of `cp_group` (correct for both the default world group and any
    #    sub-group). NCCL P2P needs absolute ranks, so we resolve them now.
    if cp_global_ranks is None:
        cp_global_ranks = dist.get_process_group_ranks(cp_group)
    if (
        not isinstance(cp_global_ranks, (list, tuple))
        or len(cp_global_ranks) != cp_size
    ):
        raise GuardError(
            f"cp_global_ranks must be a list of length cp_size={cp_size}; "
            f"got {cp_global_ranks!r}"
        )
    if cp_comm_type != "p2p":
        raise GuardError(
            f"cp_comm_type={cp_comm_type!r} not supported in v0; only 'p2p' (SPEC §2)"
        )

    return _HSTUVarlenCPFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scaling_seqlen,
        alpha,
        cp_group,
        cp_global_ranks,
        cp_stream,
        cp_comm_type,
    )
