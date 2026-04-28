# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Phase-0 reference benchmark harness (plan T0.2).

Times the production single-GPU `hstu_attn_varlen_func` over a fixed shape
grid and emits machine-readable JSON. Every PR after Phase 0 runs this
script and uses `compare.py` to assert no perf regression.

The committed numbers in `tasks/bench_baseline.json` are the canonical
baseline. Future PRs do NOT silently update them — see plan §Risk table.

Run:
    python examples/hstu/cp/bench/baseline.py --output tasks/bench_baseline.json

Stable-numbers checklist (re-running the same commit twice):
    - 100 warmup iters, 50 timed iters per shape,
    - cuda.synchronize before/after each timed window,
    - median + p95 reported,
    - same fixed seed → same input across runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

# Use the installed kernel — runtime authority (Global rule 6 in plan).
from hstu import hstu_attn_varlen_func

# ----------------------------------------------------------------------------
# Shape grid: small / medium / larger, spanning typical HSTU training shapes.
# Matches the plan T0.2 acceptance: ≥ 6 entries, range from h=2 d=32 small
# up to h=8 d=128 medium.
# ----------------------------------------------------------------------------
SHAPE_GRID: list[dict] = [
    dict(label="h2_d32_b4_s64", batch=4, seqlen=64, num_heads=2, head_dim=32),
    dict(label="h2_d64_b4_s256", batch=4, seqlen=256, num_heads=2, head_dim=64),
    dict(label="h4_d64_b8_s512", batch=8, seqlen=512, num_heads=4, head_dim=64),
    dict(label="h4_d128_b8_s1024", batch=8, seqlen=1024, num_heads=4, head_dim=128),
    dict(label="h8_d128_b8_s4096", batch=8, seqlen=4096, num_heads=8, head_dim=128),
    dict(label="h8_d128_b8_s8192", batch=8, seqlen=8192, num_heads=8, head_dim=128),
]


def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _device_label() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_properties(0).name


def _build_equal_len_batch(
    batch: int,
    seqlen: int,
    num_heads: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    total = batch * seqlen
    q = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    k = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    v = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    cu = torch.arange(0, total + 1, seqlen, dtype=torch.int32, device=device)
    return q, k, v, cu


def _time_one_shape(
    shape: dict,
    *,
    warmup: int,
    iters: int,
) -> dict:
    """Time a single shape in fwd-only. Returns metric dict for this entry."""
    q, k, v, cu = _build_equal_len_batch(
        shape["batch"], shape["seqlen"], shape["num_heads"], shape["head_dim"]
    )
    alpha = 1.0 / (shape["head_dim"] ** 0.5)
    max_seqlen = shape["seqlen"]
    total_tokens = shape["batch"] * shape["seqlen"]

    def _step() -> torch.Tensor:
        return hstu_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            scaling_seqlen=max_seqlen,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=(-1, 0),
            alpha=alpha,
        )

    # Warmup.
    for _ in range(warmup):
        _step()
    torch.cuda.synchronize()

    # Timed.
    samples_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    samples_ms.sort()
    median_ms = statistics.median(samples_ms)
    p95_ms = samples_ms[min(int(0.95 * iters), iters - 1)]
    tokens_per_s = (
        total_tokens / (median_ms / 1000.0) if median_ms > 0 else float("inf")
    )
    return dict(
        label=shape["label"],
        median_ms=median_ms,
        p95_ms=p95_ms,
        tokens_per_s=tokens_per_s,
        total_tokens=total_tokens,
        warmup=warmup,
        iters=iters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-GPU HSTU baseline benchmark")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--shape-grid",
        choices=["default"],
        default="default",
        help="Which shape grid to use (only 'default' for now).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.synchronize()

    results: list[dict] = []
    for shape in SHAPE_GRID:
        print(f"== {shape['label']}")
        res = _time_one_shape(shape, warmup=args.warmup, iters=args.iters)
        print(
            f"   median={res['median_ms']:.3f}ms  p95={res['p95_ms']:.3f}ms  "
            f"throughput={res['tokens_per_s']:.3e} tokens/s"
        )
        results.append(res)

    payload = dict(
        commit=_git_commit_sha(),
        device=_device_label(),
        warmup=args.warmup,
        iters=args.iters,
        shapes=results,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {args.output} ({len(results)} shapes)")


if __name__ == "__main__":
    main()
