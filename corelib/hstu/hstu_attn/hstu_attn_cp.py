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
  no autograd wrap, no guard cycle. Required to keep cp=1 passthrough
  perf within plan §Global rule 3 thresholds.
- DualChunkSwap dispatch helper `get_batch_on_this_cp_rank_for_hstu`
  (pure permutation; T3.2) plus testing-only `gather_global_from_cp_rank`.
- Multi-GPU forward path is a stub `_HSTUVarlenCPFunc.forward` that
  currently raises `NotImplementedError("v0 forward arrives in T3.3");
  T3.3 will fill it in. Backward is `NotImplementedError("v0 backward
  arrives in T4.2")`.

What this module does NOT do (v0 / SPEC §2)
===========================================
- Sliding-causal, `rab`, heterogeneous mask (`num_contexts`,
  `num_targets`, `target_group_size > 1`), FP8, KV-cache, Ulysses,
  comm/compute overlap (Slice 5), training-loop integration (Slice 6).
"""

from __future__ import annotations

from typing import Optional

import torch

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
    quant_mode: int,
    cp_size: int,
) -> None:
    # 1-2. Heterogeneous mask
    if num_contexts is not None:
        raise GuardError(f"num_contexts is not supported in v0 ({_SPEC_REF})")
    if num_targets is not None:
        raise GuardError(f"num_targets is not supported in v0 ({_SPEC_REF})")
    # 3. target_group_size > 1
    if target_group_size != 1:
        raise GuardError(
            f"target_group_size > 1 is not supported in v0 (got {target_group_size}; {_SPEC_REF})"
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
    # 11. quant_mode
    if quant_mode != -1:
        raise GuardError(
            f"quant_mode={quant_mode} not supported in v0; only -1 ({_SPEC_REF})"
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
    # Divisibility (DualChunkSwap requirement)
    if cp_size > 1:
        chunks_per_seq = 2 * cp_size
        seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        for b, L in enumerate(seqlens):
            if L % chunks_per_seq != 0:
                raise GuardError(
                    f"sample {b}: seqlen {L} not divisible by 2*cp_size={chunks_per_seq} "
                    f"(DualChunkSwap requirement; pre-pad in caller)"
                )
        # cu_seqlens_q must equal cu_seqlens_k (HSTU is self-attention).
        if not torch.equal(cu_seqlens_q, cu_seqlens_k):
            raise GuardError(
                "cu_seqlens_q must equal cu_seqlens_k (self-attention only in v0)"
            )


# ----------------------------------------------------------------------------
# Multi-GPU autograd Function. Stubbed for T3.1; T3.3 / T4.2 fill in.
# ----------------------------------------------------------------------------
class _HSTUVarlenCPFunc(torch.autograd.Function):
    """Multi-GPU forward+backward driver. Forward is implemented in T3.3,
    backward in T4.2. T3.1 only commits the autograd-Function shell so the
    public API has its full shape from day 1."""

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
        raise NotImplementedError(
            "v0 multi-GPU forward arrives in plan T3.3 (Slice 3). "
            "Currently only `cp_size==1` (passthrough) and the dispatch helper "
            "`get_batch_on_this_cp_rank_for_hstu` are implemented."
        )

    @staticmethod
    def backward(ctx, *grads):  # type: ignore[override]
        raise NotImplementedError(
            "v0 multi-GPU backward arrives in plan T4.2 (Slice 4). "
            "Forward is the prerequisite (T3.3)."
        )


# ----------------------------------------------------------------------------
# Public entry point.
#
# Signature mirrors the installed `hstu_attn_varlen_func` exactly (per
# `examples/hstu/test/cp/conftest.py::CANONICAL_HSTU_PARAMS`) plus four CP
# arguments. The body is the exact T3.1 deliverable (plan §T3.1):
#   1. cp_group is None or cp_size == 1 → direct passthrough (no wrap).
#   2. Hard guards (13 items).
#   3. Multi-GPU dispatch via `_HSTUVarlenCPFunc.apply` (currently NotImpl).
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
    Otherwise the CP path runs (plan T3.3+ / T4.2+).
    """
    # 1. Determine cp_size BEFORE any other logic so we can short-circuit early.
    if cp_group is None:
        cp_size = 1
    else:
        cp_size = torch.distributed.get_world_size(cp_group)

    # cp_size == 1 short-circuit: direct passthrough, no wrap, no guard cycle,
    # no autograd extra layer. Keeps cp=1 perf within plan §Global rule 3.
    if cp_size == 1:
        return hstu_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            scaling_seqlen,
            num_contexts,
            num_targets,
            target_group_size=target_group_size,
            window_size=window_size,
            alpha=alpha,
            rab=rab,
            has_drab=has_drab,
            kv_cache=kv_cache,
            page_offsets=page_offsets,
            page_ids=page_ids,
            last_page_lens=last_page_lens,
            func=func,
            quant_mode=quant_mode if quant_mode is not None else -1,
        )

    # 2. Hard guards. Only on the CP path (cp_size > 1) — passthrough relies on
    #    the underlying kernel's own validation.
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
        quant_mode=quant_mode if quant_mode is not None else -1,
        cp_size=cp_size,
    )

    # 3. Multi-GPU CP path. Currently raises NotImplementedError until T3.3.
    if cp_global_ranks is None:
        # Default to identity rank-list — T3.3 will use it for absolute NCCL P2P ranks.
        cp_global_ranks = list(range(cp_size))
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
