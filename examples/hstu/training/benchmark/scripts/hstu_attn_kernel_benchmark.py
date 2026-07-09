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

Results are printed as 2-D tables in the terminal and saved as combined
TFLOPS/MFU/time heatmap images to ``--output-dir``.

Usage (run from examples/hstu/):
    # Direct invocation
    python training/benchmark/scripts/hstu_attn_kernel_benchmark.py \\
        --gin-config-file training/configs/benchmark_ranking.gin \\
        --batch-sizes 1,2,4,8,16,32,64,128 \\
        --seqlens 128,256,512,1024,2048,4096,8192,16384 \\
        --phase fwd,bwd,e2e \\
        --warmup-iters 10 --bench-iters 50 \\
        --timing-mode per-iter \\
        --profiler-start-iter 0 --profiler-stop-iter -1 \\
        --cuda-graph

    # Via launch wrapper (sensible defaults)
    bash training/benchmark/scripts/run_hstu_attn_kernel_benchmark.sh

Each selected phase and sweep configuration emits one cudaProfilerStart/Stop
window. For multiple windows, set nsys ``--capture-range-end`` to
``repeat-shutdown:N`` with N at least as large as the number of windows.

NetworkArgs (num_heads, kv_channels, kernel_backend, is_causal, dtype_str)
are read from the gin-config file.
"""

import argparse
import contextlib
import inspect
import json
import os
import warnings
from typing import Callable

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
from commons.utils.nvtx_op import nvtx_hooks_enabled
from commons.utils.perf import _compute_attn_fwd_flops, get_current_device_spec
from configs.hstu_config import KernelBackend
from modules.hstu_attention import create_hstu_attention
from utils.gin_config_args import NetworkArgs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHASE_LABELS = {
    "fwd": "Forward",
    "bwd": "Backward",
    "e2e": "End-to-End",
}
ALL_PHASES = tuple(PHASE_LABELS)
TIMING_MODES = ("per-iter", "aggregate")
TIME_PERCENTILES = (1, 10, 20, 50, 100)
PRINTED_TIME_PERCENTILES = (1, 20, 50, 100)
PERFORMANCE_PERCENTILE = 10


def _parse_phases(value: str) -> tuple[str, ...]:
    """Parse a comma-separated, de-duplicated list of benchmark phases."""
    phases = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",")))
    invalid = [phase for phase in phases if phase not in PHASE_LABELS]
    if not phases or invalid:
        choices = ",".join(ALL_PHASES)
        detail = f": {','.join(invalid)}" if invalid else ""
        raise argparse.ArgumentTypeError(
            f"phases must be selected from {choices}{detail}"
        )
    return phases


def _make_uniform_offsets(
    batch_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Create offsets for non-jagged (uniform) sequences."""
    return torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seqlen


def _get_unwrapped_attention_forward(
    attn_module: torch.nn.Module,
) -> Callable[..., torch.Tensor]:
    """Bypass benchmark-external module and NVTX autograd hooks."""
    raw_forward = inspect.unwrap(type(attn_module).forward)
    return raw_forward.__get__(attn_module, type(attn_module))


def _make_cuda_events(
    bench_iters: int,
    timing_mode: str,
) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]:
    """Allocate and eagerly initialize CUDA events before a timed loop."""
    if timing_mode not in TIMING_MODES:
        raise ValueError(f"Unsupported timing mode: {timing_mode}")

    event_pair_count = bench_iters if timing_mode == "per-iter" else 1
    events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(event_pair_count)
    ]
    # torch.cuda.Event creates its underlying cudaEvent_t lazily on first use.
    # Record every event once so cudaEventCreateWithFlags stays outside profiling.
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()
    return events


def _record_timing_start(
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    iter_idx: int,
    timing_mode: str,
) -> None:
    if timing_mode == "per-iter":
        events[iter_idx][0].record()
    elif iter_idx == 0:
        events[0][0].record()


def _record_timing_end(
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    iter_idx: int,
    bench_iters: int,
    timing_mode: str,
) -> None:
    if timing_mode == "per-iter":
        events[iter_idx][1].record()
    elif iter_idx == bench_iters - 1:
        events[0][1].record()


def _summarize_cuda_time(
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]],
    bench_iters: int,
    timing_mode: str,
) -> dict[str, float]:
    if timing_mode == "aggregate":
        start, end = events[0]
        elapsed_times = [start.elapsed_time(end) / bench_iters]
    else:
        elapsed_times = [start.elapsed_time(end) for start, end in events]

    percentile_values = np.percentile(elapsed_times, TIME_PERCENTILES)
    return {
        f"p{percentile}": float(value)
        for percentile, value in zip(TIME_PERCENTILES, percentile_values)
    }


def _profiler_start_if_needed(iter_idx: int, profiler_start_iter: int) -> None:
    if iter_idx == profiler_start_iter:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()


def _profiler_stop_if_needed(iter_idx: int, profiler_stop_iter: int) -> None:
    if iter_idx == profiler_stop_iter:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


def _time_cuda_graph(
    graph: torch.cuda.CUDAGraph,
    bench_iters: int,
    timing_mode: str,
    profiler_start_iter: int,
    profiler_stop_iter: int,
) -> dict[str, float]:
    """Return per-replay CUDA event time percentiles in milliseconds."""
    events = _make_cuda_events(bench_iters, timing_mode)
    for iter_idx in range(bench_iters):
        _profiler_start_if_needed(iter_idx, profiler_start_iter)
        _record_timing_start(events, iter_idx, timing_mode)
        graph.replay()
        _record_timing_end(events, iter_idx, bench_iters, timing_mode)
        _profiler_stop_if_needed(iter_idx, profiler_stop_iter)

    torch.cuda.synchronize()
    return _summarize_cuda_time(events, bench_iters, timing_mode)


@contextlib.contextmanager
def _cutlass_current_stream_for_capture():
    """Route CUTLASS DSL launches to the active CUDA graph capture stream."""
    try:
        import cutlass.torch as cutlass_torch
    except ImportError:
        yield
        return

    default_stream = cutlass_torch.default_stream
    cutlass_torch.default_stream = cutlass_torch.current_stream
    try:
        yield
    finally:
        cutlass_torch.default_stream = default_stream


def _capture_and_time_cuda_graph(
    phase: str,
    attn_fn: Callable[..., torch.Tensor],
    tq: torch.Tensor,
    tk: torch.Tensor,
    tv: torch.Tensor,
    offsets: torch.Tensor,
    seqlen: int,
    grad_output: torch.Tensor,
    bench_iters: int,
    timing_mode: str,
    profiler_start_iter: int,
    profiler_stop_iter: int,
) -> dict[str, float]:
    """Capture one benchmark phase and return per-replay time percentiles."""
    requires_grad = phase != "fwd"
    static_tq = tq.detach().requires_grad_(requires_grad)
    static_tk = tk.detach().requires_grad_(requires_grad)
    static_tv = tv.detach().requires_grad_(requires_grad)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    with _cutlass_current_stream_for_capture():
        if phase == "bwd":
            with torch.cuda.stream(capture_stream):
                static_out = attn_fn(
                    static_tq, static_tk, static_tv, offsets, seqlen, seqlen
                )
            capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            if phase == "fwd":
                static_out = attn_fn(
                    static_tq, static_tk, static_tv, offsets, seqlen, seqlen
                )
            elif phase == "bwd":
                static_grads = torch.autograd.grad(
                    static_out,
                    (static_tq, static_tk, static_tv),
                    grad_outputs=grad_output,
                    retain_graph=True,
                )
            elif phase == "e2e":
                static_out = attn_fn(
                    static_tq, static_tk, static_tv, offsets, seqlen, seqlen
                )
                static_grads = torch.autograd.grad(
                    static_out,
                    (static_tq, static_tk, static_tv),
                    grad_outputs=grad_output,
                )
            else:
                raise ValueError(f"Unsupported CUDA graph benchmark phase: {phase}")

    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    # Exclude first-launch graph setup and upload costs from benchmark timing.
    graph.replay()
    torch.cuda.synchronize()

    phase_time = _time_cuda_graph(
        graph,
        bench_iters,
        timing_mode,
        profiler_start_iter,
        profiler_stop_iter,
    )
    if phase != "fwd":
        del static_grads
    return phase_time


def _benchmark_one(
    attn_fn: Callable[..., torch.Tensor],
    batch_size: int,
    seqlen: int,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    is_causal: bool,
    dtype: torch.dtype,
    warmup_iters: int = 10,
    bench_iters: int = 50,
    timing_mode: str = "per-iter",
    use_cuda_graph: bool = False,
    phases: tuple[str, ...] = ALL_PHASES,
    profiler_start_iter: int = 0,
    profiler_stop_iter: int = -1,
) -> dict:
    """Run forward + backward benchmark for a single (batch_size, seqlen) config.

    Returns a dict with timing (ms) and TFLOPS for both forward and backward.
    """
    if profiler_stop_iter == -1:
        profiler_stop_iter = bench_iters - 1

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
    warmup_backward = "bwd" in phases or "e2e" in phases
    for _ in range(warmup_iters):
        tq.requires_grad_(warmup_backward)
        tk.requires_grad_(warmup_backward)
        tv.requires_grad_(warmup_backward)
        out = attn_fn(tq, tk, tv, offsets, seqlen, seqlen)
        if warmup_backward:
            torch.autograd.grad(
                out,
                (tq, tk, tv),
                grad_outputs=grad_output,
            )
            tq = tq.detach()
            tk = tk.detach()
            tv = tv.detach()
    torch.cuda.synchronize()

    phase_time_percentiles: dict[str, dict[str, float]] = {}
    if use_cuda_graph:
        for phase in phases:
            phase_time_percentiles[phase] = _capture_and_time_cuda_graph(
                phase,
                attn_fn,
                tq,
                tk,
                tv,
                offsets,
                seqlen,
                grad_output,
                bench_iters,
                timing_mode,
                profiler_start_iter,
                profiler_stop_iter,
            )
    else:
        if "fwd" in phases:
            fwd_events = _make_cuda_events(bench_iters, timing_mode)
            tq = tq.detach()
            tk = tk.detach()
            tv = tv.detach()
            for iter_idx in range(bench_iters):
                _profiler_start_if_needed(iter_idx, profiler_start_iter)
                _record_timing_start(fwd_events, iter_idx, timing_mode)
                out = attn_fn(tq, tk, tv, offsets, seqlen, seqlen)
                _record_timing_end(fwd_events, iter_idx, bench_iters, timing_mode)
                _profiler_stop_if_needed(iter_idx, profiler_stop_iter)
            torch.cuda.synchronize()
            phase_time_percentiles["fwd"] = _summarize_cuda_time(
                fwd_events, bench_iters, timing_mode
            )

        if "bwd" in phases:
            bwd_events = _make_cuda_events(bench_iters, timing_mode)
            if timing_mode == "aggregate":
                tq.requires_grad_(True)
                tk.requires_grad_(True)
                tv.requires_grad_(True)
                out = attn_fn(tq, tk, tv, offsets, seqlen, seqlen)

            for iter_idx in range(bench_iters):
                if timing_mode == "per-iter":
                    tq.requires_grad_(True)
                    tk.requires_grad_(True)
                    tv.requires_grad_(True)
                    out = attn_fn(tq, tk, tv, offsets, seqlen, seqlen)
                _profiler_start_if_needed(iter_idx, profiler_start_iter)
                _record_timing_start(bwd_events, iter_idx, timing_mode)
                torch.autograd.grad(
                    out,
                    (tq, tk, tv),
                    grad_outputs=grad_output,
                    retain_graph=timing_mode == "aggregate",
                )
                _record_timing_end(bwd_events, iter_idx, bench_iters, timing_mode)
                _profiler_stop_if_needed(iter_idx, profiler_stop_iter)
                if timing_mode == "per-iter":
                    tq = tq.detach()
                    tk = tk.detach()
                    tv = tv.detach()
            torch.cuda.synchronize()
            phase_time_percentiles["bwd"] = _summarize_cuda_time(
                bwd_events, bench_iters, timing_mode
            )
            if timing_mode == "aggregate":
                del out
                tq = tq.detach()
                tk = tk.detach()
                tv = tv.detach()

        if "e2e" in phases:
            e2e_events = _make_cuda_events(bench_iters, timing_mode)
            for iter_idx in range(bench_iters):
                tq.requires_grad_(True)
                tk.requires_grad_(True)
                tv.requires_grad_(True)
                _profiler_start_if_needed(iter_idx, profiler_start_iter)
                _record_timing_start(e2e_events, iter_idx, timing_mode)
                out = attn_fn(tq, tk, tv, offsets, seqlen, seqlen)
                torch.autograd.grad(
                    out,
                    (tq, tk, tv),
                    grad_outputs=grad_output,
                )
                _record_timing_end(e2e_events, iter_idx, bench_iters, timing_mode)
                _profiler_stop_if_needed(iter_idx, profiler_stop_iter)
                tq = tq.detach()
                tk = tk.detach()
                tv = tv.detach()
            torch.cuda.synchronize()
            phase_time_percentiles["e2e"] = _summarize_cuda_time(
                e2e_events, bench_iters, timing_mode
            )

    phase_flops = {
        "fwd": fwd_flops,
        "bwd": bwd_flops,
        "e2e": fwd_flops + bwd_flops,
    }
    result = {
        "fwd_flops": fwd_flops,
        "bwd_flops": bwd_flops,
        "tokens": T,
    }
    for phase in phases:
        time_percentiles = phase_time_percentiles[phase]
        elapsed_ms = time_percentiles[f"p{PERFORMANCE_PERCENTILE}"]
        result[f"{phase}_ms"] = elapsed_ms
        result[f"{phase}_time_percentiles_ms"] = time_percentiles
        result[f"{phase}_tflops"] = (
            phase_flops[phase] / (elapsed_ms * 1e-3) / 1e12 if elapsed_ms > 0 else 0.0
        )
    return result


# ---------------------------------------------------------------------------
# Terminal 2-D table printing
# ---------------------------------------------------------------------------


def _print_2d_tables(
    results: dict,
    batch_sizes: list,
    seqlens: list,
    phases: tuple[str, ...],
) -> None:
    """Print 2-D tables (batch_size × seqlen) for TFLOPS and MFU in the terminal."""

    metrics = [
        (f"{phase}_tflops", f"{PHASE_LABELS[phase]} TFLOPS") for phase in phases
    ] + [(f"{phase}_mfu", f"{PHASE_LABELS[phase]} MFU (%)") for phase in phases]

    for key, title in metrics:
        print(f"\n  ── {title} ──")
        # Header row: BS \ SeqLen ...
        sl_labels = [f"{sl:>8}" for sl in seqlens]
        axis_label = "BS \\ SeqLen"
        print(f"  {axis_label:>12}" + "".join(sl_labels))
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
    time_percentile_mats: dict[str, np.ndarray],
    batch_sizes: list,
    seqlens: list,
    title: str,
    show_x_label: bool,
) -> None:
    """Draw a TFLOPS heatmap annotated with MFU and time percentiles."""
    n_bs, n_sl = tflops_mat.shape
    masked = np.ma.masked_invalid(tflops_mat)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#d9d9d9")

    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="upper")
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("TFLOPS", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(n_sl))
    ax.set_xticklabels([str(s) for s in seqlens], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_bs))
    ax.set_yticklabels([str(b) for b in batch_sizes], fontsize=9)
    ax.set_xlabel("Sequence length" if show_x_label else "", fontsize=11)
    ax.set_ylabel(f"{title}\nBatch size", fontsize=11)

    finite_tflops = tflops_mat[np.isfinite(tflops_mat)]
    midpoint = np.nanmedian(finite_tflops) if finite_tflops.size else 0.0

    for i in range(n_bs):
        for j in range(n_sl):
            if not np.isfinite(tflops_mat[i, j]):
                ax.text(
                    j,
                    i,
                    "OOM",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#333333",
                )
            else:
                tv = tflops_mat[i, j]
                mv = mfu_mat[i, j]
                percentile_values = {
                    key: values[i, j] for key, values in time_percentile_mats.items()
                }
                elapsed_ms = percentile_values[f"p{PERFORMANCE_PERCENTILE}"]
                lines = [f"{tv:.0f} TF"]
                if np.isfinite(mv):
                    lines.append(f"{mv:.1f}% | P10 {elapsed_ms:.3g} ms")
                if np.isfinite(elapsed_ms):
                    lines.extend(
                        [
                            "P1/20 "
                            f"{percentile_values['p1']:.3g}/"
                            f"{percentile_values['p20']:.3g}",
                            "P50/100 "
                            f"{percentile_values['p50']:.3g}/"
                            f"{percentile_values['p100']:.3g} ms",
                        ]
                    )
                ax.text(
                    j,
                    i,
                    "\n".join(lines),
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if tv < midpoint else "#111111",
                )

    ax.tick_params(length=2.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


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
    phases: tuple[str, ...],
) -> None:
    """Generate a combined heatmap image for the selected benchmark phases.

    Each cell shows TFLOPS, MFU, and P1/P10/P20/P50/P100 elapsed times.
    TFLOPS and MFU are calculated from P10; TFLOPS controls the colour.
    """

    phase_specs = [
        (f"{phase}_tflops", f"{phase}_mfu", PHASE_LABELS[phase]) for phase in phases
    ]

    os.makedirs(output_dir, exist_ok=True)

    n_bs, n_sl = len(batch_sizes), len(seqlens)

    matrices: dict = {}
    for phase, (tflops_key, mfu_key, label) in zip(phases, phase_specs):
        tflops_mat = np.full((n_bs, n_sl), np.nan)
        mfu_mat = np.full((n_bs, n_sl), np.nan)
        time_percentile_mats = {
            f"p{percentile}": np.full((n_bs, n_sl), np.nan)
            for percentile in TIME_PERCENTILES
        }
        for i, bs in enumerate(batch_sizes):
            for j, sl in enumerate(seqlens):
                if (bs, sl) in results:
                    tflops_mat[i, j] = results[(bs, sl)][tflops_key]
                    mfu_mat[i, j] = results[(bs, sl)][mfu_key]
                    for key in time_percentile_mats:
                        time_percentile_mats[key][i, j] = results[(bs, sl)][
                            f"{phase}_time_percentiles_ms"
                        ][key]
        matrices[label] = {
            "tflops": tflops_mat,
            "mfu": mfu_mat,
            "time_percentiles": time_percentile_mats,
        }

    hw_info = (
        f"{device_name}  |  {kernel_backend_str}  |  "
        f"H={num_heads} D={dim_per_head}  |  peak {peak_tflops:.0f} TFLOPS"
    )

    figure_width = max(11, n_sl * 1.55)
    phase_height = max(4.0, n_bs * 0.85)
    fig, axes = plt.subplots(
        len(phase_specs),
        1,
        figsize=(figure_width, phase_height * len(phase_specs) + 1.5),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, (ax, (_, _, phase_label)) in enumerate(zip(axes[:, 0], phase_specs)):
        _draw_heatmap(
            ax,
            matrices[phase_label]["tflops"],
            matrices[phase_label]["mfu"],
            matrices[phase_label]["time_percentiles"],
            batch_sizes,
            seqlens,
            title=phase_label,
            show_x_label=idx == len(phase_specs) - 1,
        )

    fig.suptitle(
        "HSTU Attention TFLOPS / MFU / Time Percentiles "
        f"(performance=P{PERFORMANCE_PERCENTILE})\n{hw_info}",
        fontsize=16,
    )

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
        description=(
            "Benchmark HSTUAttention fwd/bwd across a (batch_size, seqlen) "
            "grid and output TFLOPS/MFU/time heatmaps."
        )
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
        default="128,256,512,1024,2048,4096,8192,16384",
        help="Comma-separated list of sequence lengths to sweep.",
    )
    parser.add_argument(
        "--phase",
        type=_parse_phases,
        default=ALL_PHASES,
        help="Comma-separated phases to run: fwd,bwd,e2e (default: all).",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="Warmup iterations per config."
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=50,
        help="Benchmark iterations per config.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=TIMING_MODES,
        default="per-iter",
        help=(
            "CUDA event timing method: per-iter uses one event pair per iteration "
            "and uses P10 for TFLOPS/MFU; aggregate uses one pair around the full "
            "loop and reports total elapsed time divided by the iteration count."
        ),
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture each benchmark phase in a CUDA graph.",
    )
    parser.add_argument(
        "--profiler-start-iter",
        type=int,
        default=0,
        help="First profiled iteration, inclusive (default: 0).",
    )
    parser.add_argument(
        "--profiler-stop-iter",
        type=int,
        default=-1,
        help="Last profiled iteration, inclusive; -1 selects the last iteration.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/benchmark/figs",
        help="Directory to save heatmap images (default: training/benchmark/figs).",
    )
    args = parser.parse_args()

    if args.bench_iters <= 0:
        parser.error("--bench-iters must be greater than zero")
    profiler_stop_iter = (
        args.bench_iters - 1
        if args.profiler_stop_iter == -1
        else args.profiler_stop_iter
    )
    if not 0 <= args.profiler_start_iter < args.bench_iters:
        parser.error("--profiler-start-iter must select a benchmark iteration")
    if not args.profiler_start_iter <= profiler_stop_iter < args.bench_iters:
        parser.error(
            "--profiler-stop-iter must be -1 or an iteration at/after profiler start"
        )

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
    attn_fn = _get_unwrapped_attention_forward(attn)

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
    print(f"  timing_mode     : {args.timing_mode}")
    print(f"  performance time: P{PERFORMANCE_PERCENTILE}")
    print("  attention_call  : unwrapped forward")
    print(f"  nvtx_hooks      : {'enabled' if nvtx_hooks_enabled() else 'disabled'}")
    print(f"  phases          : {','.join(args.phase)}")
    print(f"  cuda_graph      : {'enabled' if args.cuda_graph else 'disabled'}")
    print(
        "  profiler iters  : "
        f"{args.profiler_start_iter}..{profiler_stop_iter} inclusive"
    )
    print(sep)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seqlens = [int(x) for x in args.seqlens.split(",")]

    # ---- Sweep grid ----
    col_hdr = f"{'BS':>6} {'SeqLen':>8} {'Tokens':>10}"
    for phase in args.phase:
        col_hdr += (
            f" | {phase + '_P10_ms':>11} "
            f"{phase + '_TFLOPS':>12} {phase + '_MFU':>9}"
        )
    print(col_hdr)
    print("-" * len(col_hdr))

    results: dict = {}
    best_mfu = {phase: 0.0 for phase in args.phase}
    best_cfg = {phase: None for phase in args.phase}

    for bs in batch_sizes:
        for sl in seqlens:
            tokens = bs * sl
            try:
                r = _benchmark_one(
                    attn_fn,
                    bs,
                    sl,
                    num_heads,
                    dim_per_head,
                    dim_per_head,
                    is_causal,
                    dtype,
                    warmup_iters=args.warmup_iters,
                    bench_iters=args.bench_iters,
                    timing_mode=args.timing_mode,
                    use_cuda_graph=args.cuda_graph,
                    phases=args.phase,
                    profiler_start_iter=args.profiler_start_iter,
                    profiler_stop_iter=profiler_stop_iter,
                )
                for phase in args.phase:
                    r[f"{phase}_mfu"] = r[f"{phase}_tflops"] / peak_tflops * 100.0
                results[(bs, sl)] = r

                row = f"{bs:>6} {sl:>8} {tokens:>10}"
                for phase in args.phase:
                    row += (
                        f" | {r[f'{phase}_ms']:>11.3f} "
                        f"{r[f'{phase}_tflops']:>12.2f} "
                        f"{r[f'{phase}_mfu']:>8.1f}%"
                    )
                    if r[f"{phase}_mfu"] > best_mfu[phase]:
                        best_mfu[phase] = r[f"{phase}_mfu"]
                        best_cfg[phase] = (bs, sl)
                print(row)
                printed_percentiles = []
                for phase in args.phase:
                    phase_percentiles = r[f"{phase}_time_percentiles_ms"]
                    values = " ".join(
                        f"P{percentile}=" f"{phase_percentiles[f'p{percentile}']:.3f}"
                        for percentile in PRINTED_TIME_PERCENTILES
                    )
                    printed_percentiles.append(f"{phase}: {values} ms")
                print(" " * 28 + "percentiles | " + " | ".join(printed_percentiles))

            except torch.cuda.OutOfMemoryError:
                row = f"{bs:>6} {sl:>8} {tokens:>10}"
                row += "".join(
                    f" | {'OOM':>11} {'---':>12} {'---':>9}" for _ in args.phase
                )
                print(row)
                torch.cuda.empty_cache()
                break  # larger seqlens at this BS will also OOM

    # ---- Summary: best configs ----
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    for phase in args.phase:
        if best_cfg[phase]:
            print(
                f"  Best {phase} MFU : {best_mfu[phase]:>6.1f}%  "
                f"at BS={best_cfg[phase][0]}, SeqLen={best_cfg[phase][1]}"
            )

    # ---- Print 2-D tables in terminal ----
    _print_2d_tables(results, batch_sizes, seqlens, args.phase)

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
        "timing_mode": args.timing_mode,
        "performance_percentile": PERFORMANCE_PERCENTILE,
        "time_percentiles": list(TIME_PERCENTILES),
        "attention_call": "unwrapped_forward",
        "nvtx_hooks_enabled": nvtx_hooks_enabled(),
        "phases": list(args.phase),
        "cuda_graph": args.cuda_graph,
        "profiler_start_iter": args.profiler_start_iter,
        "profiler_stop_iter": args.profiler_stop_iter,
        "profiler_stop_iter_effective": profiler_stop_iter,
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
        phases=args.phase,
    )
    print("Done.")


if __name__ == "__main__":
    main()
