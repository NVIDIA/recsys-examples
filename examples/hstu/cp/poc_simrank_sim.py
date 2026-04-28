# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
HSTU CP PoC — single-rank simulation of arbitrary `cp_size`-way DualChunkSwap.

This is the Phase-1 (Slice 2) generalisation of the cp_size=2-only PoC
(`poc_dualrank_sim.py`, kept as a thin alias for back-compat). Replaces
the hardcoded special case with the full `(cp_size × cp_size)` (rank, step)
classification grid:

  diagonal     (step == 0)        — full local Q × full local K, causal
  lower-tri    (step ≤ rank, > 0) — full local Q × peer's K first-half (zero-padded)
  upper-tri    (step > rank)      — Q's second-half × peer's full K, no mask

Reduction is plain sum across steps (no LSE — see SPEC §1 / design §3).

Used as the **numerical oracle** for production CP tests in Slices 3-4.
Does not import or depend on `torch.distributed` — pure single-GPU sim.

Run:
    python examples/hstu/cp/poc_simrank_sim.py --cp-size 2
    python examples/hstu/cp/poc_simrank_sim.py --cp-size 4 --varlen
    python examples/hstu/cp/poc_simrank_sim.py --matrix
"""

from __future__ import annotations

import argparse
from typing import Sequence

import torch

# Run from a checkout where `hstu` is importable.
from hstu import hstu_attn_varlen_func


# =============================================================================
# Input helpers
# =============================================================================
def make_batch(
    seqlens: Sequence[int],
    num_heads: int,
    head_dim: int,
    *,
    cp_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Varlen jagged batch. Each per-sample length must be divisible by 2*cp_size."""
    chunks_per_seq = 2 * cp_size
    for L in seqlens:
        if L % chunks_per_seq != 0:
            raise ValueError(
                f"seqlen {L} must be divisible by 2*cp_size={chunks_per_seq}"
            )
    g = torch.Generator(device=device).manual_seed(seed)
    total = sum(seqlens)
    q = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    k = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    v = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    cu_seqlens = (
        torch.tensor([0] + list(seqlens), dtype=torch.int32, device=device)
        .cumsum(0)
        .int()
    )
    return q, k, v, cu_seqlens


# =============================================================================
# CUTLASS kernel call (kwargs-only to match the installed signature; see
# `examples/hstu/test/cp/conftest.py::CANONICAL_HSTU_PARAMS`).
# =============================================================================
def _call_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    max_q: int,
    max_k: int,
    scaling_seqlen: int,
    alpha: float,
    window_size: tuple[int, int],
) -> torch.Tensor:
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


def baseline_hstu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    alpha: float,
) -> torch.Tensor:
    """Single global call, pure causal — the oracle every CP test compares against."""
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


# =============================================================================
# DualChunkSwap shard builder.
#
# For a sequence of length L (divisible by chunks_per_seq=2*cp_size), each chunk
# is L/chunks_per_seq tokens. Rank r owns chunks {r, chunks_per_seq-1-r}. Local
# layout per sample: [chunk_r, chunk_(2cp-1-r)] concatenated.
# =============================================================================
def build_local_shard(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_global: torch.Tensor,
    rank: int,
    cp_size: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]
]:
    """Returns (q_loc, k_loc, v_loc, cu_loc, local_to_global, chunk_sizes)."""
    chunks_per_seq = 2 * cp_size
    own = (rank, chunks_per_seq - 1 - rank)
    device = q.device
    seqlens = (cu_seqlens_global[1:] - cu_seqlens_global[:-1]).tolist()
    cu_global = cu_seqlens_global.tolist()

    rows: list[torch.Tensor] = []
    local_lens: list[int] = []
    chunk_sizes: list[int] = []
    for b, L in enumerate(seqlens):
        c_b = L // chunks_per_seq
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


# =============================================================================
# Per-sample first/second half ops (varlen-aware).
# =============================================================================
def zero_second_half_per_sample(
    t: torch.Tensor, cu_loc: torch.Tensor, chunk_sizes: list[int]
) -> torch.Tensor:
    """Zero out the second-half slots of each sample's local layout."""
    out = t.clone()
    cu = cu_loc.tolist()
    for b, c_b in enumerate(chunk_sizes):
        start = cu[b] + c_b
        end = cu[b + 1]
        out[start:end] = 0
    return out


def select_second_half_per_sample(
    t: torch.Tensor, cu_loc: torch.Tensor, chunk_sizes: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take only the second-half slot per sample. Returns (concat tensor, cu_seqlens)."""
    cu = cu_loc.tolist()
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


# =============================================================================
# Per-tile kernel calls.
# =============================================================================
def _diagonal(q_loc, k_loc, v_loc, cu_loc, local_max, scaling_seqlen, alpha):
    """step == 0: full causal sub-attention on own concat [chunk_r, chunk_(2cp-1-r)]."""
    return _call_kernel(
        q_loc,
        k_loc,
        v_loc,
        cu_loc,
        cu_loc,
        max_q=local_max,
        max_k=local_max,
        scaling_seqlen=scaling_seqlen,
        alpha=alpha,
        window_size=(-1, 0),
    )


def _lower_triangle(q_loc, k_peer, v_peer, cu_loc, local_max, scaling_seqlen, alpha):
    """step ≤ rank: full Q × peer K first-half. Peer second-half is zeroed
    upstream so K_len == Q_len; SiLU(α Q · 0) · 0 = 0 ⇒ zeroed columns
    contribute nothing.
    """
    return _call_kernel(
        q_loc,
        k_peer,
        v_peer,
        cu_loc,
        cu_loc,
        max_q=local_max,
        max_k=local_max,
        scaling_seqlen=scaling_seqlen,
        alpha=alpha,
        window_size=(-1, -1),
    )


def _upper_triangle(
    q_half,
    k_peer,
    v_peer,
    cu_q_half,
    cu_peer,
    half_max,
    local_max,
    scaling_seqlen,
    alpha,
):
    """step > rank: Q's second-half × peer's full K, no mask."""
    return _call_kernel(
        q_half,
        k_peer,
        v_peer,
        cu_q_half,
        cu_peer,
        max_q=half_max,
        max_k=local_max,
        scaling_seqlen=scaling_seqlen,
        alpha=alpha,
        window_size=(-1, -1),
    )


# =============================================================================
# Top-level CP simulator (single GPU, no comm).
# =============================================================================
def cp_simulate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_global: torch.Tensor,
    max_seqlen: int,
    alpha: float,
    cp_size: int,
) -> torch.Tensor:
    """Reproduce single-call HSTU output via DualChunkSwap+plain-sum across cp_size logical ranks."""
    chunks_per_seq = 2 * cp_size
    local_max = max_seqlen // cp_size  # per-rank ceiling (each rank holds 2 chunks)
    half_max = max_seqlen // chunks_per_seq

    # Pre-build all rank shards. Cross-referenced for off-diagonal tiles.
    shards: list[dict] = []
    for r in range(cp_size):
        q_loc, k_loc, v_loc, cu_loc, l2g, chunk_sizes = build_local_shard(
            q, k, v, cu_seqlens_global, r, cp_size
        )
        shards.append(
            dict(q=q_loc, k=k_loc, v=v_loc, cu=cu_loc, l2g=l2g, chunk_sizes=chunk_sizes)
        )

    # Accumulate in fp32 (per SPEC §2 "Reduction in fp32").
    out_global = torch.zeros_like(q, dtype=torch.float32)

    for rank in range(cp_size):
        s = shards[rank]
        out_local = torch.zeros_like(s["q"], dtype=torch.float32)

        for step in range(cp_size):
            src = (rank - step) % cp_size  # who holds the KV at this ring step
            peer = shards[src]

            if step == 0:
                # Diagonal — full causal on own layout.
                partial = _diagonal(
                    s["q"], s["k"], s["v"], s["cu"], local_max, max_seqlen, alpha
                )
                out_local += partial.float()
            elif step <= rank:
                # Lower-triangle — full Q × peer K first-half (zero-padded).
                k_padded = zero_second_half_per_sample(
                    peer["k"], peer["cu"], peer["chunk_sizes"]
                )
                v_padded = zero_second_half_per_sample(
                    peer["v"], peer["cu"], peer["chunk_sizes"]
                )
                partial = _lower_triangle(
                    s["q"], k_padded, v_padded, s["cu"], local_max, max_seqlen, alpha
                )
                out_local += partial.float()
            else:
                # Upper-triangle — Q's second-half × peer's full K, no mask.
                q_half, cu_q_half = select_second_half_per_sample(
                    s["q"], s["cu"], s["chunk_sizes"]
                )
                partial_half = _upper_triangle(
                    q_half,
                    peer["k"],
                    peer["v"],
                    cu_q_half,
                    peer["cu"],
                    half_max,
                    local_max,
                    max_seqlen,
                    alpha,
                )
                # Scatter half-output back into out_local's second-half slots
                # (per-sample, varlen-aware).
                cu = s["cu"].tolist()
                cum_half = 0
                for b, c_b in enumerate(s["chunk_sizes"]):
                    start = cu[b] + c_b
                    end = cu[b + 1]
                    out_local[start:end] += partial_half[
                        cum_half : cum_half + c_b
                    ].float()
                    cum_half += c_b

        # Scatter the rank's local output back into global token order.
        out_global[s["l2g"]] += out_local

    return out_global.to(q.dtype)


# =============================================================================
# Driver / matrix sweep
# =============================================================================
DEFAULT_MATRIX: list[dict] = [
    # SPEC §3 Slice 2 matrix (causal-only)
    dict(id="cp2_eq_64", cp_size=2, seqlens=[64, 64, 64, 64], num_heads=2, head_dim=32),
    dict(
        id="cp2_varlen", cp_size=2, seqlens=[16, 32, 48, 64], num_heads=2, head_dim=32
    ),
    dict(
        id="cp4_padded_heavy",
        cp_size=4,
        seqlens=[8, 8, 8, 256],
        num_heads=2,
        head_dim=32,
    ),
    dict(
        id="cp4_eq", cp_size=4, seqlens=[128, 256, 384, 512], num_heads=4, head_dim=64
    ),
    dict(
        id="cp8_padded_heavy",
        cp_size=8,
        seqlens=[16] * 7 + [1024],
        num_heads=4,
        head_dim=128,
    ),
    dict(
        id="cp8_eq",
        cp_size=8,
        seqlens=[512, 1024, 1024, 2048],
        num_heads=4,
        head_dim=128,
    ),
]


def run_one(
    entry: dict, *, device: torch.device, seed: int = 0, bwd: bool = False
) -> dict:
    """Run a single matrix entry. Returns metrics + pass/fail for the report.

    With `bwd=True`, also runs the autograd backward pass through both the
    baseline and the CP simulator; compares q.grad / k.grad / v.grad. Since
    `cp_simulate` is built from differentiable ops (index_select + kernel
    calls + scatter-add), single-GPU autograd traces through it and gives
    the same gradients that the production multi-GPU reverse-ring backward
    must produce. T4.1 PoC oracle.
    """
    cp_size = entry["cp_size"]
    seqlens = entry["seqlens"]
    head_dim = entry["head_dim"]
    alpha = 1.0 / head_dim**0.5
    max_seqlen = max(seqlens)

    q, k, v, cu = make_batch(
        seqlens, entry["num_heads"], head_dim, cp_size=cp_size, device=device, seed=seed
    )

    if not bwd:
        # Forward-only path (Slice 2 oracle).
        out_baseline = baseline_hstu(q, k, v, cu, max_seqlen, alpha)
        out_cp = cp_simulate(q, k, v, cu, max_seqlen, alpha, cp_size)
        diff = (out_baseline.float() - out_cp.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        base_max = out_baseline.float().abs().max().item()
        return dict(
            id=entry["id"],
            cp_size=cp_size,
            seqlens=seqlens,
            head_dim=head_dim,
            mode="fwd",
            max_abs_fwd=max_abs,
            mean_abs_fwd=mean_abs,
            base_max=base_max,
            passed=bool(max_abs <= 2e-2 + 2e-2 * base_max),
            finite_baseline=bool(torch.isfinite(out_baseline).all().item()),
            finite_cp=bool(torch.isfinite(out_cp).all().item()),
        )

    # bwd path: forward + backward, compare grads.
    # Independent input copies so requires_grad is set cleanly.
    q_b = q.detach().clone().requires_grad_(True)
    k_b = k.detach().clone().requires_grad_(True)
    v_b = v.detach().clone().requires_grad_(True)
    out_baseline = baseline_hstu(q_b, k_b, v_b, cu, max_seqlen, alpha)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    dout = torch.randn(
        out_baseline.shape, generator=g, dtype=out_baseline.dtype, device=device
    )
    out_baseline.backward(dout)
    dq_baseline = q_b.grad.detach()
    dk_baseline = k_b.grad.detach()
    dv_baseline = v_b.grad.detach()

    q_c = q.detach().clone().requires_grad_(True)
    k_c = k.detach().clone().requires_grad_(True)
    v_c = v.detach().clone().requires_grad_(True)
    out_cp = cp_simulate(q_c, k_c, v_c, cu, max_seqlen, alpha, cp_size)
    out_cp.backward(dout)
    dq_cp = q_c.grad.detach()
    dk_cp = k_c.grad.detach()
    dv_cp = v_c.grad.detach()

    diff_fwd = (out_baseline.float() - out_cp.float()).abs()
    diff_dq = (dq_baseline.float() - dq_cp.float()).abs()
    diff_dk = (dk_baseline.float() - dk_cp.float()).abs()
    diff_dv = (dv_baseline.float() - dv_cp.float()).abs()
    max_abs_fwd = diff_fwd.max().item()
    max_abs_dq = diff_dq.max().item()
    max_abs_dk = diff_dk.max().item()
    max_abs_dv = diff_dv.max().item()
    base_max = out_baseline.float().abs().max().item()

    # bf16 tolerance per SPEC §3 (looser for gradients per existing
    # `assert_hstu_close` convention: multiplier 5 vs 2).
    tol_fwd = 2e-2 + 2e-2 * base_max
    tol_grad = 5e-2 + 5e-2 * max(
        dq_baseline.float().abs().max().item(),
        dk_baseline.float().abs().max().item(),
        dv_baseline.float().abs().max().item(),
    )
    passed = (
        max_abs_fwd <= tol_fwd
        and max_abs_dq <= tol_grad
        and max_abs_dk <= tol_grad
        and max_abs_dv <= tol_grad
    )

    return dict(
        id=entry["id"],
        cp_size=cp_size,
        seqlens=seqlens,
        head_dim=head_dim,
        mode="bwd",
        max_abs_fwd=max_abs_fwd,
        max_abs_dq=max_abs_dq,
        max_abs_dk=max_abs_dk,
        max_abs_dv=max_abs_dv,
        base_max=base_max,
        passed=passed,
        finite_baseline=all(
            torch.isfinite(t).all().item()
            for t in (out_baseline, dq_baseline, dk_baseline, dv_baseline)
        ),
        finite_cp=all(
            torch.isfinite(t).all().item() for t in (out_cp, dq_cp, dk_cp, dv_cp)
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument(
        "--seqlens",
        type=str,
        default=None,
        help="Comma-separated per-sample seqlens, e.g. '64,64,64,64' or '16,32,48,64'.",
    )
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--matrix", action="store_true", help="Run the SPEC §3 Slice 2 matrix."
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="Use a default varlen seqlens distribution at the chosen cp-size.",
    )
    parser.add_argument(
        "--bwd",
        action="store_true",
        help="Run forward + backward (autograd through cp_simulate) and "
        "compare gradients (T4.1 PoC oracle for T4.2 multi-GPU backward).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    print("=== HSTU CP single-rank simulator ===")
    if args.matrix:
        print(f"running matrix ({len(DEFAULT_MATRIX)} entries; bwd={args.bwd})")
        results = [
            run_one(e, device=device, seed=args.seed, bwd=args.bwd)
            for e in DEFAULT_MATRIX
        ]
    else:
        cp_size = args.cp_size
        if args.seqlens is not None:
            seqlens = [int(s) for s in args.seqlens.split(",")]
        elif args.varlen:
            chunk_unit = 2 * cp_size
            seqlens = [chunk_unit * 1, chunk_unit * 2, chunk_unit * 3, chunk_unit * 4]
        else:
            chunk_unit = 2 * cp_size
            seqlens = [chunk_unit * 8] * 4
        entry = dict(
            id=f"manual_cp{cp_size}",
            cp_size=cp_size,
            seqlens=seqlens,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
        )
        results = [run_one(entry, device=device, seed=args.seed, bwd=args.bwd)]

    # Print results table — wider when bwd is on.
    print()
    if args.bwd:
        fmt = "{:<22} {:<3} {:<6} {:<22} {:>10} {:>10} {:>10} {:>10} {:<6}"
        print(
            fmt.format(
                "id",
                "cp",
                "head_d",
                "seqlens",
                "|fwd|",
                "|dq|",
                "|dk|",
                "|dv|",
                "verdict",
            )
        )
        print("-" * 116)
    else:
        fmt = "{:<22} {:<3} {:<6} {:<28} {:>11} {:>11} {:<6}"
        print(
            fmt.format(
                "id", "cp", "head_d", "seqlens", "max|diff|", "base_max", "verdict"
            )
        )
        print("-" * 100)
    n_pass = 0
    for r in results:
        verdict = (
            "PASS"
            if r["passed"] and r["finite_baseline"] and r["finite_cp"]
            else "FAIL"
        )
        if verdict == "PASS":
            n_pass += 1
        seqlens_repr = str(r["seqlens"])[:26]
        if args.bwd:
            print(
                fmt.format(
                    r["id"],
                    r["cp_size"],
                    r["head_dim"],
                    seqlens_repr,
                    f"{r['max_abs_fwd']:.2e}",
                    f"{r['max_abs_dq']:.2e}",
                    f"{r['max_abs_dk']:.2e}",
                    f"{r['max_abs_dv']:.2e}",
                    verdict,
                )
            )
        else:
            print(
                fmt.format(
                    r["id"],
                    r["cp_size"],
                    r["head_dim"],
                    seqlens_repr,
                    f"{r['max_abs_fwd']:.3e}",
                    f"{r['base_max']:.3e}",
                    verdict,
                )
            )
    print()
    print(f"=== summary: {n_pass}/{len(results)} PASS ===")
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
