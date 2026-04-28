# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
HSTU CP PoC — single-rank simulation of cp_size=2 (DualChunkSwap).

What this validates (numerically):
  Splitting causal HSTU attention across `cp_size=2` logical ranks using
  TransformerEngine's DualChunkSwap chunking + 3-region tile classification
  + plain-sum reduction reproduces the single-call HSTU output to bf16
  tolerance.

Why no LSE merge:
  HSTU = SiLU(αQK^T) · V / N — no softmax, no log-sum-exp. The K-axis sum
  decomposes linearly across partitions, so partial outputs add.

Tile recipe (cp_size=2, only 1 non-diagonal step):

  rank 0 owns chunks {0, 3}, rank 1 owns chunks {1, 2}.

  step 0 (own KV) — diagonal, full causal sub-attention on local concat
    [chunk_r, chunk_(3-r)]. Call kernel with window_size=(-1, 0).

  step 1, rank 0 — i=1 > rank=0 → upper-triangle.
    half-Q (chunk_3) × full-K_peer (chunks 1+2). window_size=(-1, -1).
    All Q rows are globally later than all K cols → full attention.

  step 1, rank 1 — i=1 == rank=1 → lower-triangle.
    full-Q (chunks 1+2) × half-K_peer (chunk 0). window_size=(-1, -1).
    Wrapper guard refuses Q_len > K_len, so we zero-pad K_peer's second
    half to make K_len == Q_len. Zeroed K columns contribute SiLU(0)·V=0.

`scaling_seqlen` is the **global** padded max_seqlen for every kernel call —
otherwise the four partial outputs are normalized inconsistently.

Run:  python examples/hstu/cp/poc_dualrank_sim.py
"""

from __future__ import annotations

import torch

# Run from a checkout where corelib/hstu is on PYTHONPATH (or installed as `hstu`).
from hstu import hstu_attn_varlen_func

CP_SIZE = 2
CHUNKS_PER_SEQ = 2 * CP_SIZE  # 4


def make_batch(
    seqlens: list[int],
    num_heads: int,
    head_dim: int,
    dtype=torch.bfloat16,
    device="cuda",
    seed: int = 0,
):
    """Varlen jagged batch. Each per-sample length must be divisible by CHUNKS_PER_SEQ."""
    for L in seqlens:
        assert (
            L % CHUNKS_PER_SEQ == 0
        ), f"seqlen {L} must be divisible by {CHUNKS_PER_SEQ}"
    g = torch.Generator(device=device).manual_seed(seed)
    total = sum(seqlens)
    q = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    k = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    v = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    cu_seqlens = (
        torch.tensor([0] + seqlens, dtype=torch.int32, device=device).cumsum(0).int()
    )
    return q, k, v, cu_seqlens


def _call_kernel(q, k, v, cu_q, cu_k, max_q, max_k, scaling_seqlen, alpha, window_size):
    """Wrapper that matches the *installed* hstu_attn_varlen_func signature
    (which includes seqused_q/k and required scaling_seqlen / num_contexts / num_targets).
    Passes everything via kwargs to be position-agnostic.
    """
    return hstu_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        scaling_seqlen=scaling_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=window_size,
        alpha=alpha,
    )


def baseline_hstu(q, k, v, cu_seqlens, max_seqlen: int, alpha: float):
    """Single-call HSTU on the full global batch, causal."""
    return _call_kernel(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
        scaling_seqlen=max_seqlen,
        alpha=alpha,
        window_size=(-1, 0),
    )


def build_local_shard(q, k, v, cu_seqlens_global, rank: int):
    """
    Gather rank's chunks {rank, 2*cp_size-1-rank} from each sample (varlen).

    Returns:
        q_loc, k_loc, v_loc: (sum_b 2c_b, H, D) — per-sample [chunk_r, chunk_(3-r)] concatenated.
        cu_loc: (B+1,) cumulative offsets in local layout, each sample of length 2*c_b.
        local_to_global: (sum_b 2c_b,) — maps local row index → original global row index.
        chunk_sizes: (B,) per-sample chunk size c_b.
    """
    own = (rank, CHUNKS_PER_SEQ - 1 - rank)  # (r, 3-r)
    device = q.device
    seqlens = (cu_seqlens_global[1:] - cu_seqlens_global[:-1]).tolist()
    cu_global = cu_seqlens_global.tolist()

    rows = []
    local_lens = []
    chunk_sizes = []
    for b, L in enumerate(seqlens):
        c_b = L // CHUNKS_PER_SEQ
        chunk_sizes.append(c_b)
        base = cu_global[b]
        for chunk_id in own:
            rows.append(
                torch.arange(
                    base + chunk_id * c_b, base + (chunk_id + 1) * c_b, device=device
                )
            )
        local_lens.append(2 * c_b)

    local_to_global = torch.cat(rows)
    q_loc = q[local_to_global].contiguous()
    k_loc = k[local_to_global].contiguous()
    v_loc = v[local_to_global].contiguous()

    cu_loc = (
        torch.tensor([0] + local_lens, dtype=torch.int32, device=device).cumsum(0).int()
    )
    return q_loc, k_loc, v_loc, cu_loc, local_to_global, chunk_sizes


def run_diagonal_tile(
    q_loc, k_loc, v_loc, cu_loc, two_c: int, scaling_seqlen: int, alpha: float
):
    """Step 0: own KV, full causal sub-attention on local layout."""
    return _call_kernel(
        q_loc,
        k_loc,
        v_loc,
        cu_loc,
        cu_loc,
        max_q=two_c,
        max_k=two_c,
        scaling_seqlen=scaling_seqlen,
        alpha=alpha,
        window_size=(-1, 0),
    )


def run_upper_triangle_tile(
    q_loc_second_half,
    k_peer,
    v_peer,
    cu_q_half,
    cu_peer,
    c: int,
    two_c: int,
    scaling_seqlen: int,
    alpha: float,
):
    """Step 1, rank 0: half-Q × full-K_peer, full attention.

    Q_view is chunk_(3-r) per sample (length c each).
    K_view, V_view are full peer local (length 2c each).
    Q_len < K_len ✓ wrapper accepts.
    """
    return _call_kernel(
        q_loc_second_half,
        k_peer,
        v_peer,
        cu_q_half,
        cu_peer,
        max_q=c,
        max_k=two_c,
        scaling_seqlen=scaling_seqlen,
        alpha=alpha,
        window_size=(-1, -1),
    )


def run_lower_triangle_tile(
    q_loc,
    k_peer_first_half_padded,
    v_peer_first_half_padded,
    cu_loc,
    two_c: int,
    scaling_seqlen: int,
    alpha: float,
):
    """Step 1, rank 1: full-Q × half-K_peer (padded with zeros), full attention.

    K_peer's second half is zeroed so K_len == Q_len = 2c. SiLU(α Q · 0) · 0 = 0,
    so the zeroed half contributes nothing.
    """
    return _call_kernel(
        q_loc,
        k_peer_first_half_padded,
        v_peer_first_half_padded,
        cu_loc,
        cu_loc,
        max_q=two_c,
        max_k=two_c,
        scaling_seqlen=scaling_seqlen,
        alpha=alpha,
        window_size=(-1, -1),
    )


def zero_second_half_per_sample(t: torch.Tensor, cu_loc, chunk_sizes):
    """Zero out the second half of each sample's local layout (varlen-aware)."""
    out = t.clone()
    cu = cu_loc.tolist()
    for b, c_b in enumerate(chunk_sizes):
        start = cu[b] + c_b  # second-half begin
        end = cu[b + 1]  # second-half end (= start + c_b)
        out[start:end] = 0
    return out


def select_second_half_per_sample(t: torch.Tensor, cu_loc, chunk_sizes):
    """Take the second half of each sample's local layout (varlen-aware).
    Returns concatenated tensor of length sum(c_b) and the cu_seqlens for that view.
    """
    cu = cu_loc.tolist()
    parts = []
    half_lens = []
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


def cp_simulate(q, k, v, cu_seqlens_global, max_seqlen: int, alpha: float):
    """
    Run the full cp_size=2 simulation on a single GPU. Varlen-aware.
    `max_seqlen` is the global max seqlen, used as scaling_seqlen for every kernel call.
    Returns the global-token-ordered output, shape == q.shape.
    """
    out_global = torch.zeros_like(q, dtype=torch.float32)

    # Pre-build both ranks' shards.
    shards = {}
    for r in range(CP_SIZE):
        q_loc, k_loc, v_loc, cu_loc, l2g, chunk_sizes = build_local_shard(
            q, k, v, cu_seqlens_global, r
        )
        shards[r] = dict(
            q=q_loc, k=k_loc, v=v_loc, cu=cu_loc, l2g=l2g, chunk_sizes=chunk_sizes
        )

    # Per-rank max local seqlen (= max over samples of 2*c_b)
    local_max = (
        max_seqlen // CP_SIZE
    )  # since seqlens divisible by CHUNKS_PER_SEQ=2*CP_SIZE
    half_max = max_seqlen // CHUNKS_PER_SEQ

    for rank in range(CP_SIZE):
        s = shards[rank]
        peer_rank = (rank - 1) % CP_SIZE
        peer = shards[peer_rank]

        # Diagonal tile: own KV, full causal sub-attention.
        out_diag = run_diagonal_tile(
            s["q"],
            s["k"],
            s["v"],
            s["cu"],
            two_c=local_max,
            scaling_seqlen=max_seqlen,
            alpha=alpha,
        )
        out_local = out_diag.float()

        # Off-diagonal tile, classified by (rank, step=1).
        if 1 > rank:
            # Upper-triangle: half-Q × full peer KV.
            q_half, cu_q_half = select_second_half_per_sample(
                s["q"], s["cu"], s["chunk_sizes"]
            )
            out_off_half = run_upper_triangle_tile(
                q_half,
                peer["k"],
                peer["v"],
                cu_q_half,
                peer["cu"],
                c=half_max,
                two_c=local_max,
                scaling_seqlen=max_seqlen,
                alpha=alpha,
            )
            # Scatter half-output back into the second-half slots of out_local.
            cu = s["cu"].tolist()
            cum_half = 0
            for b, c_b in enumerate(s["chunk_sizes"]):
                start = cu[b] + c_b
                end = cu[b + 1]
                out_local[start:end] += out_off_half[cum_half : cum_half + c_b].float()
                cum_half += c_b
        else:
            # Lower-triangle: full-Q × half-K_peer (zero-padded second half).
            k_peer_padded = zero_second_half_per_sample(
                peer["k"], peer["cu"], peer["chunk_sizes"]
            )
            v_peer_padded = zero_second_half_per_sample(
                peer["v"], peer["cu"], peer["chunk_sizes"]
            )
            out_off = run_lower_triangle_tile(
                s["q"],
                k_peer_padded,
                v_peer_padded,
                s["cu"],
                two_c=local_max,
                scaling_seqlen=max_seqlen,
                alpha=alpha,
            )
            out_local += out_off.float()

        # Scatter out_local back into global token order.
        out_global[s["l2g"]] += out_local

    return out_global.to(q.dtype)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    batch_size = 4
    chunk_size = 16  # chunk size per chunk
    seqlen = chunk_size * CHUNKS_PER_SEQ  # = 64, divisible by 4
    num_heads = 2
    head_dim = 32
    alpha = 1.0 / head_dim**0.5

    print(
        f"=== HSTU CP PoC: cp_size={CP_SIZE}, batch_size={batch_size}, "
        f"seqlen={seqlen}, num_heads={num_heads}, head_dim={head_dim} ==="
    )

    q, k, v, cu_seqlens = make_batch(
        [seqlen] * batch_size, num_heads, head_dim, device=device, seed=0
    )

    out_baseline = baseline_hstu(q, k, v, cu_seqlens, seqlen, alpha)
    out_cp = cp_simulate(q, k, v, cu_seqlens, seqlen, alpha)

    diff = (out_baseline.float() - out_cp.float()).abs()
    print(f"baseline shape: {tuple(out_baseline.shape)}, dtype={out_baseline.dtype}")
    print(f"cp_sim   shape: {tuple(out_cp.shape)},     dtype={out_cp.dtype}")
    print(f"max |diff|:        {diff.max().item():.3e}")
    print(f"mean |diff|:       {diff.mean().item():.3e}")
    print(f"max |baseline|:    {out_baseline.float().abs().max().item():.3e}")
    print(
        f"max rel diff:      {(diff / (out_baseline.float().abs() + 1e-9)).max().item():.3e}"
    )

    # bf16 tolerance — HSTU has known precision noise. Tighten on fp32 input later if needed.
    torch.testing.assert_close(out_cp, out_baseline, rtol=2e-2, atol=2e-2)
    print("PASS")


if __name__ == "__main__":
    main()
