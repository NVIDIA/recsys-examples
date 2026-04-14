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
Benchmark HSTUAttention forward & backward kernels across a grid of
(batch_size, seqlen) configurations to measure TFLOPS and MFU.

Only **non-jagged** (padded / uniform) inputs are considered:
all sequences in a batch share the same sequence length.

Results are printed as 2-D tables in the terminal and saved as heatmap
images to ``--output-dir``.

Usage (run from examples/hstu/):
    python training/benchmark/scripts/benchmark_hstu_attn_mfu.py \\
        --gin-config-file training/configs/benchmark_ranking.gin \\
        --batch-sizes 1,2,4,8,16,32,64,128 \\
        --seqlens 64,128,256,512,1024,2048,4096 \\
        --warmup-iters 10 --bench-iters 50

NetworkArgs (num_heads, kv_channels, kernel_backend, is_causal, dtype_str)
are read from the gin-config file.
"""

import argparse
import json
import os
import statistics
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

import gin
import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no display needed
import commons.utils.initialize as init
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import all gin-configurable classes so gin.parse_config_file succeeds
# even when the config file contains bindings for unrelated classes.
import utils.gin_config_args as _gin_args  # noqa: F401
from commons.utils.perf import _compute_attn_fwd_flops, get_current_device_spec
from configs.hstu_config import KernelBackend
from modules.hstu_attention import create_hstu_attention
from utils.gin_config_args import NetworkArgs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uniform_offsets(
    batch_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Create offsets for non-jagged (uniform) sequences."""
    return torch.arange(0, batch_size + 1, dtype=torch.int64, device=device) * seqlen


def _benchmark_one(
    attn_module: torch.nn.Module,
    batch_size: int,
    seqlen: int,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    is_causal: bool,
    dtype: torch.dtype,
    warmup_iters: int = 10,
    bench_iters: int = 50,
) -> dict:
    """Run forward + backward benchmark for a single (batch_size, seqlen) config.

    Returns a dict with timing (ms) and TFLOPS for both forward and backward.
    """
    device = torch.cuda.current_device()
    T = batch_size * seqlen

    offsets = _make_uniform_offsets(batch_size, seqlen, torch.device(device))

    # Pre-allocate tensors (re-used across iterations to avoid alloc noise)
    tq = torch.randn(T, num_heads * attention_dim, dtype=dtype, device=device)
    tk = torch.randn(T, num_heads * attention_dim, dtype=dtype, device=device)
    tv = torch.randn(T, num_heads * linear_dim, dtype=dtype, device=device)
    grad_output = torch.randn(T, num_heads * linear_dim, dtype=dtype, device=device)

    # Compute FLOPs (attention only)
    fwd_flops = _compute_attn_fwd_flops(
        offsets,
        num_heads,
        attention_dim,
        linear_dim,
        is_causal,
        num_candidates=None,
        num_contextuals=None,
    )
    bwd_flops = fwd_flops * 2.5  # backward ≈ 2.5× forward for attention

    # ----- warmup -----
    for _ in range(warmup_iters):
        tq.requires_grad_(True)
        tk.requires_grad_(True)
        tv.requires_grad_(True)
        out = attn_module(tq, tk, tv, offsets, seqlen, seqlen)
        out.backward(grad_output)
        tq = tq.detach()
        tk = tk.detach()
        tv = tv.detach()
    torch.cuda.synchronize()

    # ----- benchmark forward -----
    fwd_events = []
    for _ in range(bench_iters):
        tq.requires_grad_(True)
        tk.requires_grad_(True)
        tv.requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = attn_module(tq, tk, tv, offsets, seqlen, seqlen)
        end.record()
        fwd_events.append((start, end))
        # must do backward so that next iteration can call requires_grad_ again
        out.backward(grad_output)
        tq = tq.detach()
        tk = tk.detach()
        tv = tv.detach()
    torch.cuda.synchronize()
    fwd_times = [s.elapsed_time(e) for s, e in fwd_events]
    fwd_median_ms = statistics.median(fwd_times)

    # ----- benchmark backward -----
    bwd_events = []
    for _ in range(bench_iters):
        tq.requires_grad_(True)
        tk.requires_grad_(True)
        tv.requires_grad_(True)
        out = attn_module(tq, tk, tv, offsets, seqlen, seqlen)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out.backward(grad_output)
        end.record()
        bwd_events.append((start, end))
        tq = tq.detach()
        tk = tk.detach()
        tv = tv.detach()
    torch.cuda.synchronize()
    bwd_times = [s.elapsed_time(e) for s, e in bwd_events]
    bwd_median_ms = statistics.median(bwd_times)

    # ----- benchmark e2e (fwd + bwd) -----
    e2e_events = []
    for _ in range(bench_iters):
        tq.requires_grad_(True)
        tk.requires_grad_(True)
        tv.requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = attn_module(tq, tk, tv, offsets, seqlen, seqlen)
        out.backward(grad_output)
        end.record()
        e2e_events.append((start, end))
        tq = tq.detach()
        tk = tk.detach()
        tv = tv.detach()
    torch.cuda.synchronize()
    e2e_times = [s.elapsed_time(e) for s, e in e2e_events]
    e2e_median_ms = statistics.median(e2e_times)

    fwd_tflops = fwd_flops / (fwd_median_ms * 1e-3) / 1e12 if fwd_median_ms > 0 else 0.0
    bwd_tflops = bwd_flops / (bwd_median_ms * 1e-3) / 1e12 if bwd_median_ms > 0 else 0.0
    e2e_tflops = (
        (fwd_flops + bwd_flops) / (e2e_median_ms * 1e-3) / 1e12
        if e2e_median_ms > 0
        else 0.0
    )

    return {
        "fwd_ms": fwd_median_ms,
        "bwd_ms": bwd_median_ms,
        "e2e_ms": e2e_median_ms,
        "fwd_tflops": fwd_tflops,
        "bwd_tflops": bwd_tflops,
        "e2e_tflops": e2e_tflops,
        "fwd_flops": fwd_flops,
        "bwd_flops": bwd_flops,
        "tokens": T,
    }


# ---------------------------------------------------------------------------
# Terminal 2-D table printing
# ---------------------------------------------------------------------------


def _print_2d_tables(
    results: dict,
    batch_sizes: list,
    seqlens: list,
    peak_tflops: float,
) -> None:
    """Print 2-D tables (batch_size × seqlen) for TFLOPS and MFU in the terminal."""

    metrics = [
        ("fwd_tflops", "Forward TFLOPS"),
        ("bwd_tflops", "Backward TFLOPS"),
        ("e2e_tflops", "End-to-End TFLOPS"),
        ("fwd_mfu", "Forward MFU (%)"),
        ("bwd_mfu", "Backward MFU (%)"),
        ("e2e_mfu", "End-to-End MFU (%)"),
    ]

    for key, title in metrics:
        print(f"\n  ── {title} ──")
        # Header row: BS \ SeqLen ...
        sl_labels = [f"{sl:>8}" for sl in seqlens]
        print(f"  {'BS \\ SeqLen':>12}" + "".join(sl_labels))
        print("  " + "-" * (12 + 8 * len(seqlens)))

        for bs in batch_sizes:
            row = f"  {bs:>12}"
            for sl in seqlens:
                if (bs, sl) in results:
                    v = results[(bs, sl)][key]
                    row += f"{v:>8.1f}"
                else:
                    row += f"{'OOM':>8}"
            print(row)


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------


def _draw_heatmap(
    ax: plt.Axes,
    tflops_mat: np.ndarray,
    mfu_mat: np.ndarray,
    batch_sizes: list,
    seqlens: list,
    title: str,
) -> None:
    """Draw a single heatmap with TFLOPS as colour and MFU in parentheses."""
    n_bs, n_sl = tflops_mat.shape
    masked = np.ma.masked_invalid(tflops_mat)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="lightgrey")

    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="upper")
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("TFLOPS", fontsize=21)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xticks(range(n_sl))
    ax.set_xticklabels([str(s) for s in seqlens], fontsize=18)
    ax.set_yticks(range(n_bs))
    ax.set_yticklabels([str(b) for b in batch_sizes], fontsize=18)
    ax.set_xlabel("Sequence Length", fontsize=21)
    ax.set_ylabel("Batch Size", fontsize=21)

    vmin, vmax = np.nanmin(tflops_mat), np.nanmax(tflops_mat)
    mid = (vmin + vmax) / 2 if vmax > vmin else vmax

    for i in range(n_bs):
        for j in range(n_sl):
            if np.isnan(tflops_mat[i, j]):
                ax.text(
                    j,
                    i,
                    "OOM",
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="grey",
                    fontstyle="italic",
                )
            else:
                tv = tflops_mat[i, j]
                mv = mfu_mat[i, j]
                t_fmt = f"{tv:.0f}" if tv >= 10 else f"{tv:.1f}"
                m_fmt = f"({mv:.1f}%)"
                text_color = "white" if tv > mid else "black"
                mfu_color = "#66FF66" if tv > mid else "#006600"
                ax.text(
                    j,
                    i - 0.12,
                    t_fmt,
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=text_color,
                )
                ax.text(
                    j,
                    i + 0.22,
                    m_fmt,
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color=mfu_color,
                )

    ax.set_title(title, fontsize=21)


def _plot_heatmaps(
    results: dict,
    batch_sizes: list,
    seqlens: list,
    peak_tflops: float,
    device_name: str,
    kernel_backend_str: str,
    num_heads: int,
    dim_per_head: int,
    output_dir: str,
) -> None:
    """Generate a single combined heatmap image with Forward / Backward / E2E.

    Each cell shows TFLOPS (bold) with MFU in parentheses below it.
    """

    phases = [
        ("fwd_tflops", "fwd_mfu", "Forward"),
        ("bwd_tflops", "bwd_mfu", "Backward"),
        ("e2e_tflops", "e2e_mfu", "End-to-End"),
    ]

    os.makedirs(output_dir, exist_ok=True)

    n_bs, n_sl = len(batch_sizes), len(seqlens)

    matrices: dict = {}
    for tflops_key, mfu_key, label in phases:
        tflops_mat = np.full((n_bs, n_sl), np.nan)
        mfu_mat = np.full((n_bs, n_sl), np.nan)
        for i, bs in enumerate(batch_sizes):
            for j, sl in enumerate(seqlens):
                if (bs, sl) in results:
                    tflops_mat[i, j] = results[(bs, sl)][tflops_key]
                    mfu_mat[i, j] = results[(bs, sl)][mfu_key]
        matrices[label] = {"tflops": tflops_mat, "mfu": mfu_mat}

    hw_info = (
        f"{device_name}  |  {kernel_backend_str}  |  "
        f"H={num_heads} D={dim_per_head}  |  peak {peak_tflops:.0f} TFLOPS"
    )

    cell_w = max(10, n_sl * 1.4)
    cell_h = max(5, n_bs * 0.8)
    fig, axes = plt.subplots(3, 1, figsize=(cell_w, cell_h * 3 + 4))

    for idx, (ax, (_, _, phase_label)) in enumerate(zip(axes, phases)):
        _draw_heatmap(
            ax,
            matrices[phase_label]["tflops"],
            matrices[phase_label]["mfu"],
            batch_sizes,
            seqlens,
            title=phase_label,
        )
        if idx < len(phases) - 1:
            ax.set_xlabel("")

    fig.suptitle(
        f"HSTU Attention TFLOPS  (MFU%)\n{hw_info}",
        fontsize=22,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    fname = "hstu_attn_mfu.png"
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HSTUAttention fwd/bwd across (batch_size, seqlen) grid and output TFLOPS/MFU heatmaps."
    )
    parser.add_argument(
        "--gin-config-file",
        type=str,
        required=True,
        help="Path to gin config file (NetworkArgs are read from here).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help="Comma-separated list of batch sizes to sweep.",
    )
    parser.add_argument(
        "--seqlens",
        type=str,
        default="64,128,256,512,1024,2048,4096",
        help="Comma-separated list of sequence lengths to sweep.",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="Warmup iterations per config."
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=50,
        help="Benchmark iterations per config (median is reported).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/benchmark/figs",
        help="Directory to save heatmap images (default: training/benchmark/figs).",
    )
    args = parser.parse_args()

    # ---- Init (single-rank, no TP) ----
    init.initialize_single_rank()

    # ---- Parse gin config ----
    gin.parse_config_file(args.gin_config_file)
    net = NetworkArgs()

    num_heads = net.num_attention_heads
    dim_per_head = net.kv_channels  # attention_dim == linear_dim == kv_channels
    hidden_size = net.hidden_size
    is_causal = net.is_causal
    kernel_backend_str = net.kernel_backend
    dtype_str = net.dtype_str
    net.scaling_seqlen

    kernel_backend = KernelBackend[kernel_backend_str.upper()]
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    # ---- Create attention module ----
    attn = create_hstu_attention(
        kernel_backend=kernel_backend,
        num_heads=num_heads,
        attention_dim=dim_per_head,
        linear_dim=dim_per_head,
        is_causal=is_causal,
    )
    attn = attn.to(dtype).cuda()
    attn.eval()  # deterministic (no dropout)

    # ---- Device info ----
    device_spec = get_current_device_spec()
    dtype_key = "bf16" if dtype == torch.bfloat16 else "fp16"
    peak_tflops = device_spec.peak_tflops.get(
        dtype_key, device_spec.peak_tflops.get("fp16", 312.0)
    )

    # ---- Print header ----
    sep = "=" * 120
    print(sep)
    print("HSTU Attention MFU Benchmark  (non-jagged / uniform seqlen)")
    print(sep)
    print(f"  Device          : {device_spec.device_name}")
    print(f"  Peak {dtype_key} TFLOPS : {peak_tflops:.1f}")
    print(f"  Kernel backend  : {kernel_backend_str}")
    print(f"  num_heads       : {num_heads}")
    print(f"  dim_per_head    : {dim_per_head}")
    print(f"  hidden_size     : {hidden_size}")
    print(f"  is_causal       : {is_causal}")
    print(f"  dtype           : {dtype_str}")
    print(f"  warmup/bench    : {args.warmup_iters}/{args.bench_iters} iters")
    print(sep)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seqlens = [int(x) for x in args.seqlens.split(",")]

    # ---- Sweep grid ----
    col_hdr = (
        f"{'BS':>6} {'SeqLen':>8} {'Tokens':>10} | "
        f"{'fwd_ms':>9} {'fwd_TFLOPS':>12} {'fwd_MFU':>9} | "
        f"{'bwd_ms':>9} {'bwd_TFLOPS':>12} {'bwd_MFU':>9} | "
        f"{'e2e_ms':>9} {'e2e_TFLOPS':>12} {'e2e_MFU':>9}"
    )
    print(col_hdr)
    print("-" * len(col_hdr))

    results: dict = {}
    best_fwd_mfu = 0.0
    best_bwd_mfu = 0.0
    best_e2e_mfu = 0.0
    best_fwd_cfg = None
    best_bwd_cfg = None
    best_e2e_cfg = None

    for bs in batch_sizes:
        for sl in seqlens:
            tokens = bs * sl
            try:
                r = _benchmark_one(
                    attn,
                    bs,
                    sl,
                    num_heads,
                    dim_per_head,
                    dim_per_head,
                    is_causal,
                    dtype,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                )
                fwd_mfu = r["fwd_tflops"] / peak_tflops * 100.0
                bwd_mfu = r["bwd_tflops"] / peak_tflops * 100.0
                e2e_mfu = r["e2e_tflops"] / peak_tflops * 100.0

                results[(bs, sl)] = {
                    **r,
                    "fwd_mfu": fwd_mfu,
                    "bwd_mfu": bwd_mfu,
                    "e2e_mfu": e2e_mfu,
                }

                print(
                    f"{bs:>6} {sl:>8} {tokens:>10} | "
                    f"{r['fwd_ms']:>9.3f} {r['fwd_tflops']:>12.2f} {fwd_mfu:>8.1f}% | "
                    f"{r['bwd_ms']:>9.3f} {r['bwd_tflops']:>12.2f} {bwd_mfu:>8.1f}% | "
                    f"{r['e2e_ms']:>9.3f} {r['e2e_tflops']:>12.2f} {e2e_mfu:>8.1f}%"
                )

                if fwd_mfu > best_fwd_mfu:
                    best_fwd_mfu = fwd_mfu
                    best_fwd_cfg = (bs, sl)
                if bwd_mfu > best_bwd_mfu:
                    best_bwd_mfu = bwd_mfu
                    best_bwd_cfg = (bs, sl)
                if e2e_mfu > best_e2e_mfu:
                    best_e2e_mfu = e2e_mfu
                    best_e2e_cfg = (bs, sl)

            except torch.cuda.OutOfMemoryError:
                print(
                    f"{bs:>6} {sl:>8} {tokens:>10} | "
                    f"{'OOM':>9} {'---':>12} {'---':>9} | "
                    f"{'OOM':>9} {'---':>12} {'---':>9} | "
                    f"{'OOM':>9} {'---':>12} {'---':>9}"
                )
                torch.cuda.empty_cache()
                break  # larger seqlens at this BS will also OOM

    # ---- Summary: best configs ----
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    if best_fwd_cfg:
        print(
            f"  Best fwd MFU : {best_fwd_mfu:>6.1f}%  at BS={best_fwd_cfg[0]}, SeqLen={best_fwd_cfg[1]}"
        )
    if best_bwd_cfg:
        print(
            f"  Best bwd MFU : {best_bwd_mfu:>6.1f}%  at BS={best_bwd_cfg[0]}, SeqLen={best_bwd_cfg[1]}"
        )
    if best_e2e_cfg:
        print(
            f"  Best e2e MFU : {best_e2e_mfu:>6.1f}%  at BS={best_e2e_cfg[0]}, SeqLen={best_e2e_cfg[1]}"
        )

    # ---- Print 2-D tables in terminal ----
    _print_2d_tables(results, batch_sizes, seqlens, peak_tflops)

    # ---- Save raw results as JSON for later re-plotting ----
    os.makedirs(args.output_dir, exist_ok=True)
    json_data = {
        "device_name": device_spec.device_name,
        "peak_tflops": peak_tflops,
        "kernel_backend": kernel_backend_str,
        "num_heads": num_heads,
        "dim_per_head": dim_per_head,
        "dtype": dtype_str,
        "batch_sizes": batch_sizes,
        "seqlens": seqlens,
        "warmup_iters": args.warmup_iters,
        "bench_iters": args.bench_iters,
        "results": {f"{bs},{sl}": v for (bs, sl), v in results.items()},
    }
    json_path = os.path.join(args.output_dir, "hstu_attn_mfu_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved results: {json_path}")

    # ---- Plot and save heatmaps ----
    print("\nGenerating heatmaps ...")
    _plot_heatmaps(
        results=results,
        batch_sizes=batch_sizes,
        seqlens=seqlens,
        peak_tflops=peak_tflops,
        device_name=device_spec.device_name,
        kernel_backend_str=kernel_backend_str,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        output_dir=args.output_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
