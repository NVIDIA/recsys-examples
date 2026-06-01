#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Parse fused HSTU layer benchmark logs under a results batch and emit two
figures:

    1. ``layer_tflops.png``     — progressive TFLOPS grouped bars
       (fwd / bwd / e2e per exp) + speedup (×) line vs baseline.
    2. ``layer_time_breakdown.png`` — stacked fwd+bwd time per exp with
       the true e2e-step time annotated on top.

Matches the log format produced by hstu_layer_benchmark.py:
    [<LAYER>] [train_fwd] tokens N;time (median): X ms;achieved flops: Y TFLOPS
    [<LAYER>] [bwd]       tokens N;time (median): X ms;achieved flops: Y TFLOPS
    [<LAYER>] [e2e]       tokens N;time (median): X ms;achieved flops: Y TFLOPS

Usage:
    python analyze_layer_results.py training/benchmark/results/<batch_ts>
    python analyze_layer_results.py <batch_dir> --output-dir /tmp/plots
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


LINE_RE = re.compile(
    r"\[(?P<layer>\w+)\]\s+\[(?P<stage>train_fwd|bwd|e2e)\]\s+"
    r"tokens\s+(?P<tokens>\d+);"
    r"time \(median\):\s+(?P<time_ms>[\d.]+)\s+ms;"
    r"achieved flops:\s+(?P<tflops>[\d.]+)\s+TFLOPS"
    r"(?:;MFU:\s+(?P<mfu>[\d.]+)%)?"  # MFU optional (older logs may lack it)
)


def parse_log(path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    """Return {stage: {time_ms, tflops, tokens, layer}} or None if no match."""
    results: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            stage = m.group("stage")
            mfu_raw = m.group("mfu")
            results[stage] = {
                "time_ms": float(m.group("time_ms")),
                "tflops": float(m.group("tflops")),
                "tokens": int(m.group("tokens")),
                "layer": m.group("layer"),
                "mfu": float(mfu_raw) if mfu_raw is not None else None,
            }
    return results if results else None


def collect(batch_dir: Path) -> List[Tuple[str, Dict[str, Dict[str, Any]]]]:
    """Collect (exp_name, parsed_log) for every exp subdir, sorted by name."""
    out: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []
    for exp_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
        logs = sorted(exp_dir.glob("*.log"))
        if not logs:
            continue
        parsed = parse_log(logs[0])
        if parsed is None:
            print(
                f"  [skip] {exp_dir.name}: no layer benchmark lines in {logs[0].name}"
            )
            continue
        out.append((exp_dir.name, parsed))
    return out


def plot_tflops(
    data: List[Tuple[str, Dict[str, Dict[str, Any]]]],
    output_path: Path,
    title: str,
) -> None:
    exp_names = [n for n, _ in data]
    stages = ["train_fwd", "bwd", "e2e"]
    stage_labels = {
        "train_fwd": "Forward (train)",
        "bwd": "Backward",
        "e2e": "Fwd+Bwd (e2e)",
    }
    stage_colors = {"train_fwd": "#4C72B0", "bwd": "#DD8452", "e2e": "#55A868"}

    n_exp = len(exp_names)
    n_stage = len(stages)
    x_base = range(n_exp)
    bar_w = 0.25

    fig, ax_tf = plt.subplots(figsize=(max(8, n_exp * 1.8), 6.5))
    ax_su = ax_tf.twinx()

    # --- Bars: TFLOPS per stage ---
    for i, stage in enumerate(stages):
        values = [data[j][1].get(stage, {}).get("tflops", 0.0) for j in range(n_exp)]
        offsets = [x + (i - (n_stage - 1) / 2) * bar_w for x in x_base]
        bars = ax_tf.bar(
            offsets,
            values,
            bar_w,
            label=stage_labels[stage],
            color=stage_colors[stage],
            edgecolor="black",
            linewidth=0.4,
        )
        for bar, v in zip(bars, values):
            if v > 0:
                ax_tf.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax_tf.set_xticks(list(x_base))
    ax_tf.set_xticklabels(exp_names, rotation=20, ha="right")
    ax_tf.set_ylabel("TFLOPS (higher is better)", fontsize=11)
    ax_tf.set_xlabel("Experiment", fontsize=11)
    ax_tf.grid(axis="y", alpha=0.3)
    ax_tf.set_title(title, fontsize=13, fontweight="bold")

    # --- Line: e2e MFU% (preferred) + e2e speedup × vs first exp ---
    e2e_tflops = [data[j][1].get("e2e", {}).get("tflops", 0.0) for j in range(n_exp)]
    e2e_mfus = [data[j][1].get("e2e", {}).get("mfu") for j in range(n_exp)]
    baseline = e2e_tflops[0] if e2e_tflops and e2e_tflops[0] > 0 else 1.0
    speedups = [v / baseline if baseline > 0 else 0.0 for v in e2e_tflops]

    has_mfu = any(m is not None for m in e2e_mfus)
    if has_mfu:
        mfu_display = [m if m is not None else 0.0 for m in e2e_mfus]
        ax_su.plot(
            list(x_base),
            mfu_display,
            "o-",
            color="#C44E52",
            markersize=8,
            linewidth=2,
            label="e2e MFU %",
        )
        for x, mv, sp in zip(x_base, mfu_display, speedups):
            ax_su.annotate(
                f"{mv:.1f}%\n({sp:.2f}×)",
                (x, mv),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="#C44E52",
                fontweight="bold",
                linespacing=1.2,
            )
        ax_su.set_ylabel("e2e MFU (%)", fontsize=11, color="#C44E52")
        ax_su.set_ylim(0, max(mfu_display) * 1.6 if mfu_display else 1)
    else:
        ax_su.plot(
            list(x_base),
            speedups,
            "s--",
            color="#C44E52",
            markersize=8,
            linewidth=2,
            label="e2e speedup ×",
        )
        for x, sp in zip(x_base, speedups):
            ax_su.annotate(
                f"{sp:.2f}×",
                (x, sp),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="#C44E52",
                fontweight="bold",
            )
        ax_su.set_ylabel("e2e Speedup (×) vs first exp", fontsize=11, color="#C44E52")
        ax_su.set_ylim(0, max(speedups) * 1.4 if speedups else 1)
    ax_su.tick_params(axis="y", labelcolor="#C44E52")

    # Combine legends
    h1, l1 = ax_tf.get_legend_handles_labels()
    h2, l2 = ax_su.get_legend_handles_labels()
    ax_tf.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def plot_time_breakdown(
    data: List[Tuple[str, Dict[str, Dict[str, Any]]]],
    output_path: Path,
    title: str,
) -> None:
    exp_names = [n for n, _ in data]
    fwd_ms = [
        data[j][1].get("train_fwd", {}).get("time_ms", 0.0) for j in range(len(data))
    ]
    bwd_ms = [data[j][1].get("bwd", {}).get("time_ms", 0.0) for j in range(len(data))]
    e2e_ms = [data[j][1].get("e2e", {}).get("time_ms", 0.0) for j in range(len(data))]

    fig, ax = plt.subplots(figsize=(max(8, len(data) * 1.6), 6.0))
    x = range(len(data))

    ax.bar(x, fwd_ms, color="#4C72B0", edgecolor="black", linewidth=0.4, label="fwd")
    ax.bar(
        x,
        bwd_ms,
        bottom=fwd_ms,
        color="#DD8452",
        edgecolor="black",
        linewidth=0.4,
        label="bwd",
    )

    # Annotate stacked bar + true e2e (single-region measurement)
    for i in x:
        total = fwd_ms[i] + bwd_ms[i]
        ax.text(
            i,
            total + max(e2e_ms) * 0.03,
            f"fwd+bwd={total:.2f} ms\n  e2e (1-region)={e2e_ms[i]:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=8,
            linespacing=1.3,
        )
        # Mark the true e2e with a red tick so the gap between sum-of-medians
        # and the single-region measurement is visible.
        ax.plot(
            [i - 0.35, i + 0.35], [e2e_ms[i], e2e_ms[i]], color="#C44E52", linewidth=2
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(exp_names, rotation=20, ha="right")
    ax.set_ylabel("Time per step (ms, lower is better)", fontsize=11)
    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Leave headroom for annotation + red e2e mark.
    ymax = max(fwd_ms[i] + bwd_ms[i] for i in range(len(data))) * 1.35
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def print_table(data: List[Tuple[str, Dict[str, Dict[str, Any]]]]) -> None:
    header = (
        f"{'exp':<20} {'layer':<8} "
        f"{'fwd ms':>8} {'bwd ms':>8} {'e2e ms':>8} "
        f"{'fwd TF':>8} {'bwd TF':>8} {'e2e TF':>8} "
        f"{'fwd MFU':>8} {'bwd MFU':>8} {'e2e MFU':>8} "
        f"{'e2e×':>6}"
    )
    print(header)
    print("-" * len(header))
    baseline_e2e = data[0][1].get("e2e", {}).get("tflops", 0.0) if data else 0.0
    for name, parsed in data:
        fwd = parsed.get("train_fwd", {})
        bwd = parsed.get("bwd", {})
        e2e = parsed.get("e2e", {})
        layer = (fwd or bwd or e2e).get("layer", "?")
        speedup = e2e.get("tflops", 0.0) / baseline_e2e if baseline_e2e > 0 else 0.0
        fwd_mfu = fwd.get("mfu")
        bwd_mfu = bwd.get("mfu")
        e2e_mfu = e2e.get("mfu")
        print(
            f"{name:<20} {layer:<8} "
            f"{fwd.get('time_ms', 0):>8.3f} {bwd.get('time_ms', 0):>8.3f} {e2e.get('time_ms', 0):>8.3f} "
            f"{fwd.get('tflops', 0):>8.1f} {bwd.get('tflops', 0):>8.1f} {e2e.get('tflops', 0):>8.1f} "
            f"{(f'{fwd_mfu:.1f}%' if fwd_mfu is not None else '-'):>8} "
            f"{(f'{bwd_mfu:.1f}%' if bwd_mfu is not None else '-'):>8} "
            f"{(f'{e2e_mfu:.1f}%' if e2e_mfu is not None else '-'):>8} "
            f"{speedup:>5.2f}×"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_dir", help="Results batch dir (results/<batch_timestamp>/)")
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Where to save figures (default: same as batch_dir).",
    )
    ap.add_argument(
        "--title",
        default="HSTU Layer Benchmark",
        help="Title prefix for the figures.",
    )
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        print(f"Error: not a directory: {batch_dir}")
        return 2

    output_dir = Path(args.output_dir).resolve() if args.output_dir else batch_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {batch_dir}")
    data = collect(batch_dir)
    if not data:
        print("No layer benchmark logs found.")
        return 1

    print_table(data)
    print()

    plot_tflops(
        data, output_dir / "layer_tflops.png", f"{args.title} — TFLOPS & speedup"
    )
    plot_time_breakdown(
        data,
        output_dir / "layer_time_breakdown.png",
        f"{args.title} — Step-time breakdown",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
