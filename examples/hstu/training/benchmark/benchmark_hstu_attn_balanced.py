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

"""
Benchmark HSTUAttention forward & backward with Karmarkar-Karp load-balanced
batch shuffling in a multi-GPU distributed environment.

Compares per-rank attention performance between:
  - **Unbalanced**: each rank uses its original random seqlens
  - **Balanced**: seqlens are redistributed across ranks via KK partitioning

Workflow:
  1. Each rank generates random seqlens via ``RandomDistribution`` (from gin).
  2. Compute attention workloads → KK partition → balanced seqlens per rank.
  3. Generate q, k, v from **original** seqlens → benchmark (unbalanced).
  4. Generate q, k, v from **balanced** seqlens → benchmark (balanced).
  5. Gather per-rank timing from all ranks → print comparison on rank 0.

``NetworkArgs`` and ``AttnBalancedBenchmarkArgs`` are read from a gin-config
file.  CLI arguments can override benchmark parameters.

Usage (multi-GPU):
    cd examples/hstu
    torchrun --nproc_per_node=8 training/benchmark/benchmark_hstu_attn_balanced.py \\
        --gin-config-file training/configs/balanced_attn_benchmark.gin

Usage (single-GPU, for smoke testing):
    python training/benchmark/benchmark_hstu_attn_balanced.py \\
        --gin-config-file training/configs/balanced_attn_benchmark.gin
"""

import argparse
import os
import random
import statistics
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import commons.utils.initialize as init
import gin
import numpy as np
import nvtx
import torch
import torch.distributed as dist
import utils.gin_config_args as _gin_args  # noqa: F401 — registers gin configurables
from commons.datasets.hstu_batch import RandomDistribution
from commons.distributed.batch_shuffler import BaseTaskBalancedBatchShuffler
from commons.perf_model.task_estimator import HSTUAttentionTask
from commons.utils.perf import _compute_attn_fwd_flops, get_current_device_spec
from configs.hstu_config import KernelBackend
from modules.hstu_attention import create_hstu_attention
from utils.gin_config_args import NetworkArgs

# ---------------------------------------------------------------------------
# Gin-configurable benchmark args
# ---------------------------------------------------------------------------


@gin.configurable
@dataclass
class AttnBalancedBenchmarkArgs:
    """Gin-configurable arguments for the balanced-attention benchmark.

    .. note::
        **Benchmark only** — used exclusively by
        ``benchmark_hstu_attn_balanced.py``.

    Attributes:
        batch_size: Per-rank batch size.
        max_seqlen: Maximum sequence length (also used as the upper bound
            of *seqlen_dist* when its ``high`` is not set).
        seqlen_dist: Distribution for generating random sequence lengths.
            If None, defaults to uniform [1, max_seqlen).
        warmup_iters: Number of warmup iterations.
        bench_iters: Number of benchmark iterations (median is reported).
        seed: Base random seed (combined with rank for cross-rank diversity).
    """

    batch_size: int = 32
    max_seqlen: int = 2048
    seqlen_dist: Optional[RandomDistribution] = None
    warmup_iters: int = 10
    bench_iters: int = 50
    seed: int = 1234

    def __post_init__(self):
        if self.seqlen_dist is not None and self.seqlen_dist.high is None:
            self.seqlen_dist.high = self.max_seqlen


# ---------------------------------------------------------------------------
# Lightweight shuffler (operates on raw seqlens, not BaseBatch)
# ---------------------------------------------------------------------------


class _SimpleAttnShuffler(BaseTaskBalancedBatchShuffler):
    """Compute HSTU attention workloads from seqlens and run KK partitioning.

    This is a thin wrapper that reuses
    :class:`~commons.perf_model.task_estimator.HSTUAttentionTask` for workload
    estimation and :meth:`compute_partition_indices` for KK redistribution,
    without requiring a full ``BaseBatch``.
    """

    def __init__(self, num_heads: int, head_dim: int) -> None:
        self._task = HSTUAttentionTask()
        self._num_heads = num_heads
        self._head_dim = head_dim

    def get_workloads(self, seqlens, *args, **kwargs):  # type: ignore[override]
        return self._task.get_workloads(seqlens, self._num_heads, self._head_dim)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _generate_data(
    seqlens: torch.Tensor,
    num_heads: int,
    attn_dim: int,
    linear_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Create random q, k, v, offsets, grad_output from *seqlens*."""
    seqlens_i64 = seqlens.to(torch.int64)
    T = int(seqlens_i64.sum().item())
    offsets = torch.zeros(len(seqlens_i64) + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(seqlens_i64, dim=0)
    max_seqlen = int(seqlens_i64.max().item())

    tq = torch.randn(T, num_heads * attn_dim, dtype=dtype, device=device)
    tk = torch.randn(T, num_heads * attn_dim, dtype=dtype, device=device)
    tv = torch.randn(T, num_heads * linear_dim, dtype=dtype, device=device)
    grad = torch.randn(T, num_heads * linear_dim, dtype=dtype, device=device)
    return tq, tk, tv, offsets, max_seqlen, grad


# ---------------------------------------------------------------------------
# Core benchmark routine
# ---------------------------------------------------------------------------


def _benchmark(
    attn_module: torch.nn.Module,
    tq: torch.Tensor,
    tk: torch.Tensor,
    tv: torch.Tensor,
    offsets: torch.Tensor,
    max_seqlen: int,
    scaling_seqlen: int,
    grad: torch.Tensor,
    fwd_flops: float,
    warmup_iters: int,
    bench_iters: int,
) -> dict:
    """Run fwd / bwd / e2e benchmark; return timing (ms) and TFLOPS."""
    bwd_flops = fwd_flops * 2.5

    # ── warmup ──────────────────────────────────────────────────────────
    with nvtx.annotate("warmup"):
        for _ in range(warmup_iters):
            tq.requires_grad_(True)
            tk.requires_grad_(True)
            tv.requires_grad_(True)
            out = attn_module(tq, tk, tv, offsets, max_seqlen, scaling_seqlen)
            out.backward(grad)
            tq, tk, tv = tq.detach(), tk.detach(), tv.detach()
        torch.cuda.synchronize()

    # ── pre-allocate CUDA events ─────────────────────────────────────────
    _new_evt = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    ev_fwd_s = [_new_evt() for _ in range(bench_iters)]
    ev_fwd_e = [_new_evt() for _ in range(bench_iters)]
    ev_bwd_e = [_new_evt() for _ in range(bench_iters)]
    ev_ge_s = [_new_evt() for _ in range(bench_iters)]
    ev_ge_e = [_new_evt() for _ in range(bench_iters)]

    # ── fwd / bwd / e2e / global_e2e in a single loop ───────────────────
    # barrier aligns all ranks so that ev_ge_s captures a common start;
    # per-rank events (ev_fwd_s/ev_fwd_e/ev_bwd_e) are unaffected because
    # the barrier has already completed on the stream before they record.
    for i in range(bench_iters):
        nvtx.push_range(f"iter/{i}")
        tq.requires_grad_(True)
        tk.requires_grad_(True)
        tv.requires_grad_(True)
        dist.barrier()  # align ranks
        ev_ge_s[i].record()  # ← global start
        ev_fwd_s[i].record()  # ← fwd start
        out = attn_module(tq, tk, tv, offsets, max_seqlen, scaling_seqlen)
        ev_fwd_e[i].record()  # ← fwd end / bwd start
        out.backward(grad)
        ev_bwd_e[i].record()  # ← bwd end
        ev_ge_e[i].record()  # ← global end
        tq, tk, tv = tq.detach(), tk.detach(), tv.detach()
        nvtx.pop_range()
    torch.cuda.synchronize()
    fwd_ms = statistics.median(
        [ev_fwd_s[i].elapsed_time(ev_fwd_e[i]) for i in range(bench_iters)]
    )
    bwd_ms = statistics.median(
        [ev_fwd_e[i].elapsed_time(ev_bwd_e[i]) for i in range(bench_iters)]
    )
    e2e_ms = statistics.median(
        [ev_fwd_s[i].elapsed_time(ev_bwd_e[i]) for i in range(bench_iters)]
    )
    global_e2e_ms = statistics.median(
        [ev_ge_s[i].elapsed_time(ev_ge_e[i]) for i in range(bench_iters)]
    )

    fwd_tflops = fwd_flops / (fwd_ms * 1e-3) / 1e12 if fwd_ms > 0 else 0.0
    bwd_tflops = bwd_flops / (bwd_ms * 1e-3) / 1e12 if bwd_ms > 0 else 0.0
    e2e_tflops = (fwd_flops + bwd_flops) / (e2e_ms * 1e-3) / 1e12 if e2e_ms > 0 else 0.0

    return {
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "e2e_ms": e2e_ms,
        "global_e2e_ms": global_e2e_ms,
        "fwd_tflops": fwd_tflops,
        "bwd_tflops": bwd_tflops,
        "e2e_tflops": e2e_tflops,
        "fwd_flops": fwd_flops,
        "bwd_flops": bwd_flops,
    }


# ---------------------------------------------------------------------------
# Distributed gather helpers
# ---------------------------------------------------------------------------


def _gather_float_metrics(local_metrics: dict, world_size: int) -> List[dict]:
    """Gather per-rank metrics to all ranks via ``all_gather``."""
    keys = [
        "fwd_ms",
        "bwd_ms",
        "e2e_ms",
        "global_e2e_ms",
        "fwd_tflops",
        "bwd_tflops",
        "e2e_tflops",
        "fwd_flops",
        "bwd_flops",
    ]
    local_t = torch.tensor(
        [local_metrics[k] for k in keys], dtype=torch.float64, device="cuda"
    )
    gathered = [torch.zeros_like(local_t) for _ in range(world_size)]
    dist.all_gather(gathered, local_t)
    return [
        {k: gathered[r][i].item() for i, k in enumerate(keys)}
        for r in range(world_size)
    ]


def _gather_seqlen_stats(
    seqlens: torch.Tensor, workloads: torch.Tensor, world_size: int
) -> List[Dict[str, float]]:
    """Gather seqlen and workload statistics from all ranks."""
    stats = torch.tensor(
        [
            float(seqlens.sum().item()),
            float(seqlens.min().item()),
            float(seqlens.max().item()),
            float(len(seqlens)),
            float(workloads.sum().item()),
        ],
        dtype=torch.float64,
        device="cuda",
    )
    gathered = [torch.zeros_like(stats) for _ in range(world_size)]
    dist.all_gather(gathered, stats)
    result = []
    for g in gathered:
        cnt = g[3].item()
        result.append(
            {
                "tokens": int(g[0].item()),
                "min": int(g[1].item()),
                "max": int(g[2].item()),
                "count": int(cnt),
                "mean": g[0].item() / cnt if cnt > 0 else 0.0,
                "workload": g[4].item(),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Pretty-printing helpers (rank 0 only)
# ---------------------------------------------------------------------------


def _print_header(
    device_name: str,
    peak_tflops: float,
    dtype_key: str,
    kernel_backend: str,
    num_heads: int,
    dim_per_head: int,
    is_causal: bool,
    world_size: int,
    batch_size: int,
    max_seqlen: int,
    seqlen_dist: Optional[RandomDistribution],
    warmup_iters: int,
    bench_iters: int,
) -> None:
    sep = "=" * 120
    print(sep)
    print("HSTU Attention Balanced Benchmark (Karmarkar-Karp load balancing)")
    print(sep)
    print(f"  Device          : {device_name}")
    print(f"  Peak TFLOPS     : {peak_tflops:.1f} ({dtype_key})")
    print(f"  Kernel backend  : {kernel_backend}")
    print(f"  num_heads       : {num_heads}")
    print(f"  dim_per_head    : {dim_per_head}")
    print(f"  is_causal       : {is_causal}")
    print(f"  world_size      : {world_size}")
    print(f"  batch_size/rank : {batch_size}")
    print(f"  max_seqlen      : {max_seqlen}")
    if seqlen_dist is not None:
        dt = seqlen_dist.dist_type
        parts = [f"type={dt.value if hasattr(dt, 'value') else dt}"]
        parts.append(f"low={seqlen_dist.low}")
        if seqlen_dist.high is not None:
            parts.append(f"high={seqlen_dist.high}")
        if seqlen_dist.mean is not None:
            parts.append(f"mean={seqlen_dist.mean}")
        if seqlen_dist.std is not None:
            parts.append(f"std={seqlen_dist.std}")
        if seqlen_dist.alpha is not None:
            parts.append(f"alpha={seqlen_dist.alpha}")
        print(f"  seqlen_dist     : {', '.join(parts)}")
    else:
        print(f"  seqlen_dist     : uniform [1, {max_seqlen})")
    print(f"  warmup/bench    : {warmup_iters}/{bench_iters} iters")
    print(sep)


def _print_results_table(
    label: str,
    all_metrics: List[dict],
    all_stats: List[Dict[str, float]],
    peak_tflops: float,
) -> None:
    """Print per-rank results table with two-level header."""
    # Column widths for each group
    #   Input:  Rank(6) + Tokens(10) + Workload(12) + MinSL(7) + MaxSL(7) + MeanSL(8) + spaces
    #   fwd/bwd/e2e: ms(9) + TF(8) + MFU(8) + spaces = 27 each
    input_w = 56
    group_w = 27
    hdr1 = (
        f"  {'Input':^{input_w}} │ "
        f"{'fwd':^{group_w}} │ "
        f"{'bwd':^{group_w}} │ "
        f"{'e2e':^{group_w}}"
    )
    hdr2 = (
        f"  {'Rank':>6} {'Tokens':>10} {'Workload':>12} {'MinSL':>7} {'MaxSL':>7} {'MeanSL':>8} │ "
        f"{'ms':>9} {'TFLOPS':>8} {'MFU':>8} │ "
        f"{'ms':>9} {'TFLOPS':>8} {'MFU':>8} │ "
        f"{'ms':>9} {'TFLOPS':>8} {'MFU':>8}"
    )
    print(f"\n  ── {label} ──")
    print(hdr1)
    print(hdr2)
    print(f"  {'-' * (len(hdr2) - 2)}")

    for r, (m, st) in enumerate(zip(all_metrics, all_stats)):
        fwd_mfu = m["fwd_tflops"] / peak_tflops * 100
        bwd_mfu = m["bwd_tflops"] / peak_tflops * 100
        e2e_mfu = m["e2e_tflops"] / peak_tflops * 100
        print(
            f"  {r:>6} {st['tokens']:>10} {st['workload']:>12.3e} {st['min']:>7} {st['max']:>7} "
            f"{st['mean']:>8.1f} │ "
            f"{m['fwd_ms']:>9.3f} {m['fwd_tflops']:>8.1f} {fwd_mfu:>7.1f}% │ "
            f"{m['bwd_ms']:>9.3f} {m['bwd_tflops']:>8.1f} {bwd_mfu:>7.1f}% │ "
            f"{m['e2e_ms']:>9.3f} {m['e2e_tflops']:>8.1f} {e2e_mfu:>7.1f}%"
        )

    # Summary
    total_tokens = sum(st["tokens"] for st in all_stats)
    all_workloads = [st["workload"] for st in all_stats]
    max_wl = max(all_workloads)
    min_wl = min(all_workloads)
    wl_imb = (max_wl - min_wl) / max_wl * 100 if max_wl > 0 else 0.0
    max_fwd = max(m["fwd_ms"] for m in all_metrics)
    min_fwd = min(m["fwd_ms"] for m in all_metrics)
    max_bwd = max(m["bwd_ms"] for m in all_metrics)
    min_bwd = min(m["bwd_ms"] for m in all_metrics)
    max_e2e = max(m["e2e_ms"] for m in all_metrics)
    min_e2e = min(m["e2e_ms"] for m in all_metrics)
    fwd_imb = (max_fwd - min_fwd) / max_fwd * 100 if max_fwd > 0 else 0.0
    bwd_imb = (max_bwd - min_bwd) / max_bwd * 100 if max_bwd > 0 else 0.0
    e2e_imb = (max_e2e - min_e2e) / max_e2e * 100 if max_e2e > 0 else 0.0
    print(
        f"\n  Total tokens: {total_tokens:,}   "
        f"workload imbalance: {wl_imb:.1f}% (max={max_wl:.3e} min={min_wl:.3e})"
    )
    print(
        f"  fwd imbalance: {fwd_imb:.1f}% (max={max_fwd:.3f}ms min={min_fwd:.3f}ms)   "
        f"bwd imbalance: {bwd_imb:.1f}% (max={max_bwd:.3f}ms min={min_bwd:.3f}ms)   "
        f"e2e imbalance: {e2e_imb:.1f}% (max={max_e2e:.3f}ms min={min_e2e:.3f}ms)"
    )

    # Global TFLOPS / MFU  (barrier-synced: all ranks start fwd together)
    W = len(all_metrics)
    global_e2e_max_ms = max(m["global_e2e_ms"] for m in all_metrics)
    total_fwd_flops = sum(m["fwd_flops"] for m in all_metrics)
    total_bwd_flops = sum(m["bwd_flops"] for m in all_metrics)
    total_flops = total_fwd_flops + total_bwd_flops
    global_tflops = (
        total_flops / (global_e2e_max_ms * 1e-3) / 1e12
        if global_e2e_max_ms > 0
        else 0.0
    )
    global_mfu = global_tflops / (peak_tflops * W) * 100 if peak_tflops > 0 else 0.0
    print(
        f"  Global e2e (barrier-synced): max_time={global_e2e_max_ms:.3f}ms  "
        f"total_TFLOPS={global_tflops:.1f}  MFU={global_mfu:.1f}%  "
        f"(peak={peak_tflops:.1f} × {W} ranks = {peak_tflops * W:.1f})"
    )


def _print_comparison(
    unbal_metrics: List[dict],
    bal_metrics: List[dict],
    peak_tflops: float,
) -> None:
    """Print side-by-side comparison of global TFLOPS / MFU."""
    w = len(unbal_metrics)
    total_peak = peak_tflops * w

    # Global e2e time (barrier-synced, max across ranks)
    u_global = max(m["global_e2e_ms"] for m in unbal_metrics)
    b_global = max(m["global_e2e_ms"] for m in bal_metrics)

    # Total FLOPs across all ranks
    u_total_flops = sum(m["fwd_flops"] + m["bwd_flops"] for m in unbal_metrics)
    b_total_flops = sum(m["fwd_flops"] + m["bwd_flops"] for m in bal_metrics)

    u_global_tflops = u_total_flops / (u_global * 1e-3) / 1e12 if u_global > 0 else 0.0
    b_global_tflops = b_total_flops / (b_global * 1e-3) / 1e12 if b_global > 0 else 0.0
    u_global_mfu = u_global_tflops / total_peak * 100 if total_peak > 0 else 0.0
    b_global_mfu = b_global_tflops / total_peak * 100 if total_peak > 0 else 0.0

    tflops_speedup = (
        b_global_tflops / u_global_tflops if u_global_tflops > 0 else float("inf")
    )

    print(f"\n  ── Comparison (global, barrier-synced across {w} ranks) ──")
    print(f"  {'':>20} {'Unbalanced':>14} {'Balanced':>14} {'Speedup':>10}")
    print(f"  {'-' * 60}")
    print(
        f"  {'global TFLOPS':>20} "
        f"{u_global_tflops:>14.1f} {b_global_tflops:>14.1f} {tflops_speedup:>9.2f}x"
    )
    print(f"  {'global MFU':>20} " f"{u_global_mfu:>13.1f}% {b_global_mfu:>13.1f}%")
    print(f"\n  peak = {peak_tflops:.1f} × {w} ranks = {total_peak:.1f} TFLOPS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark HSTUAttention fwd/bwd with KK load-balanced batch "
            "shuffling.  Compares unbalanced vs balanced performance."
        ),
    )
    parser.add_argument(
        "--gin-config-file",
        type=str,
        required=True,
        help="Path to gin config file (NetworkArgs + AttnBalancedBenchmarkArgs).",
    )
    parser.add_argument(
        "--gin-bindings",
        nargs="*",
        default=[],
        help="Additional gin parameter bindings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override AttnBalancedBenchmarkArgs.batch_size.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=None,
        help="Override AttnBalancedBenchmarkArgs.warmup_iters.",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=None,
        help="Override AttnBalancedBenchmarkArgs.bench_iters.",
    )
    args = parser.parse_args()
    # ── Distributed / single-rank init ─────────────────────────────────
    if "LOCAL_RANK" in os.environ:
        init.initialize_distributed()
    else:
        init.initialize_single_rank()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ── Parse gin config ───────────────────────────────────────────────
    gin.parse_config_file(args.gin_config_file)
    if args.gin_bindings:
        gin.parse_config(args.gin_bindings)

    net = NetworkArgs()
    bench = AttnBalancedBenchmarkArgs()

    # CLI overrides
    if args.batch_size is not None:
        bench.batch_size = args.batch_size
    if args.warmup_iters is not None:
        bench.warmup_iters = args.warmup_iters
    if args.bench_iters is not None:
        bench.bench_iters = args.bench_iters

    # ── Extract attention parameters ───────────────────────────────────
    num_heads = net.num_attention_heads
    dim_per_head = net.kv_channels
    is_causal = net.is_causal
    kernel_backend_str = net.kernel_backend
    dtype_str = net.dtype_str
    scaling_seqlen = net.scaling_seqlen

    kernel_backend = KernelBackend[kernel_backend_str.upper()]
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    # ── Create attention module ────────────────────────────────────────
    attn = create_hstu_attention(
        kernel_backend=kernel_backend,
        num_heads=num_heads,
        attention_dim=dim_per_head,
        linear_dim=dim_per_head,
        is_causal=is_causal,
    )
    attn = attn.to(dtype).cuda()

    # ── Device info ────────────────────────────────────────────────────
    device_spec = get_current_device_spec()
    dtype_key = "bf16" if dtype == torch.bfloat16 else "fp16"
    peak_tflops = device_spec.peak_tflops.get(
        dtype_key, device_spec.peak_tflops.get("fp16", 312.0)
    )
    # ── Print header (rank 0) ─────────────────────────────────────────
    if rank == 0:
        _print_header(
            device_spec.device_name,
            peak_tflops,
            dtype_key,
            kernel_backend_str,
            num_heads,
            dim_per_head,
            is_causal,
            world_size,
            bench.batch_size,
            bench.max_seqlen,
            bench.seqlen_dist,
            bench.warmup_iters,
            bench.bench_iters,
        )
    # ── Per-rank random seed ──────────────────────────────────────────
    rank_seed = bench.seed + rank * 137
    torch.manual_seed(rank_seed)
    np.random.seed(rank_seed)
    random.seed(rank_seed)

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    batch_size = bench.batch_size

    # ==================================================================
    # Phase 1: Generate seqlens per rank
    # ==================================================================
    if bench.seqlen_dist is not None:
        seqlens = bench.seqlen_dist.sample(batch_size, device)
        seqlens = seqlens.clamp(min=1)  # ensure at least length 1
    else:
        seqlens = torch.randint(1, bench.max_seqlen, (batch_size,), device=device)

    # ==================================================================
    # Phase 2: Compute balanced seqlens via KK
    # ==================================================================
    shuffler = _SimpleAttnShuffler(num_heads, dim_per_head)
    workloads = shuffler.get_workloads(seqlens)
    # workloads may be int-like tensor — ensure float for gather_along_first_dim
    if isinstance(workloads, torch.Tensor):
        workloads = workloads.float()
    else:
        workloads = torch.tensor(workloads, dtype=torch.float32, device=device)

    indices = shuffler.compute_partition_indices(
        workloads, batch_size, dist.group.WORLD
    )
    balanced_seqlens = BaseTaskBalancedBatchShuffler.shuffle_tensor_by_global_indices(
        seqlens, indices, dist.group.WORLD
    )

    # ==================================================================
    # Phase 3: Generate data for both cases
    # ==================================================================
    tq, tk, tv, offsets, max_sl, grad = _generate_data(
        seqlens, num_heads, dim_per_head, dim_per_head, dtype, device
    )
    fwd_flops = _compute_attn_fwd_flops(
        offsets, num_heads, dim_per_head, dim_per_head, is_causal, None, None
    )

    bal_tq, bal_tk, bal_tv, bal_offsets, bal_max_sl, bal_grad = _generate_data(
        balanced_seqlens, num_heads, dim_per_head, dim_per_head, dtype, device
    )
    bal_fwd_flops = _compute_attn_fwd_flops(
        bal_offsets, num_heads, dim_per_head, dim_per_head, is_causal, None, None
    )

    # ==================================================================
    # Phase 4: Benchmark unbalanced
    # ==================================================================
    if rank == 0:
        print("\nBenchmarking unbalanced ...")
    dist.barrier()
    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range("unbalanced"):
        unbal_result = _benchmark(
            attn,
            tq,
            tk,
            tv,
            offsets,
            max_sl,
            scaling_seqlen,
            grad,
            fwd_flops,
            bench.warmup_iters,
            bench.bench_iters,
        )

    # Free unbalanced data to save memory
    del tq, tk, tv, offsets, grad
    torch.cuda.empty_cache()

    # ==================================================================
    # Phase 5: Benchmark balanced
    # ==================================================================
    if rank == 0:
        print("Benchmarking balanced ...")
    dist.barrier()
    with torch.cuda.nvtx.range("balanced"):
        bal_result = _benchmark(
            attn,
            bal_tq,
            bal_tk,
            bal_tv,
            bal_offsets,
            bal_max_sl,
            scaling_seqlen,
            bal_grad,
            bal_fwd_flops,
            bench.warmup_iters,
            bench.bench_iters,
        )
    torch.cuda.cudart().cudaProfilerStop()

    # ==================================================================
    # Phase 6: Gather results from all ranks
    # ==================================================================
    all_unbal = _gather_float_metrics(unbal_result, world_size)
    all_bal = _gather_float_metrics(bal_result, world_size)
    # Compute balanced workloads for stats
    bal_workloads = shuffler.get_workloads(balanced_seqlens)
    if isinstance(bal_workloads, torch.Tensor):
        bal_workloads = bal_workloads.float()
    else:
        bal_workloads = torch.tensor(bal_workloads, dtype=torch.float32, device=device)
    unbal_stats = _gather_seqlen_stats(seqlens, workloads, world_size)
    bal_stats = _gather_seqlen_stats(balanced_seqlens, bal_workloads, world_size)

    # ==================================================================
    # Phase 7: Print comparison (rank 0)
    # ==================================================================
    if rank == 0:
        _print_results_table("Unbalanced", all_unbal, unbal_stats, peak_tflops)
        _print_results_table("Balanced (KK)", all_bal, bal_stats, peak_tflops)
        _print_comparison(all_unbal, all_bal, peak_tflops)
        print()


if __name__ == "__main__":
    main()
