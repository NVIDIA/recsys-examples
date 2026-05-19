# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Plot benchmark_results.json produced by benchmark_batched_dynamicemb_tables.sh.
#
# Reads a list of per-config result dicts, groups by (optimizer, mode),
# and renders a 2-row x 4-col bar chart comparing DynamicEmb vs TorchRec
# for forward / backward / train / eval latencies (ms).
#
# Usage:
#   python plot_benchmark_results.py
#   python plot_benchmark_results.py --results benchmark_results.json --out bench.png
#   python plot_benchmark_results.py --log              # log-scale y-axis
#   python plot_benchmark_results.py --no-values        # hide bar labels

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# Canonical ordering for the four storage modes the suite parametrizes over.
MODE_ORDER = ["gpu", "caching", "no_caching", "no_hbm"]
MODE_LABEL = {
    "gpu": "GPU\n(full HBM)",
    "caching": "Caching\n(10% LRU)",
    "no_caching": "NoCaching\n(10% HBM)",
    "no_hbm": "NoHBM\n(UVM)",
}

METRICS = ["forward", "backward", "train", "eval"]


def parse_mode(label: str) -> str:
    """Pick the suite mode out of the dataclass label.

    Order matters: ``no_caching`` and ``no_hbm`` are checked before the
    shorter substring ``gpu`` / ``caching`` so they don't get misclassified.
    """
    for m in ("no_caching", "no_hbm", "caching", "gpu"):
        if f"_{m}_" in label:
            return m
    return "unknown"


def filter_timing_entries(results: List[dict]) -> List[dict]:
    """Skip profile-only entries (ncu_gen / ncu_run / nsys / torch_profile)."""
    return [r for r in results if "dyn_train_ms" in r]


def collect_suite(results: List[dict], optimizer: str, mode: str):
    """Return (dyn_values, trc_values) over METRICS for one (optimizer, mode) suite.

    Returns ``(None, None)`` when the suite isn't present in the results.
    """
    for r in results:
        if (
            r["optimizer_type"] == optimizer
            and parse_mode(r["label"]) == mode
        ):
            dyn_v = [r[f"dyn_{m}_ms"] for m in METRICS]
            trc_v = [r[f"trc_{m}_ms"] for m in METRICS]
            return dyn_v, trc_v
    return None, None


def _label_bars(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


DYN_FWD_COLOR = "#c2e07a"  # light NVIDIA green
DYN_BWD_COLOR = "#76b900"  # NVIDIA green
TRC_FWD_COLOR = "#9dc3e6"  # light blue
TRC_BWD_COLOR = "#1f77b4"  # matplotlib blue


def _annotate_total(ax, x, total, fmt="{:.2f}"):
    ax.annotate(
        fmt.format(total),
        xy=(x, total),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
    )


def make_figure(results: List[dict], log: bool, show_values: bool) -> plt.Figure:
    """Per-suite layout: rows = optimizer, cols = storage mode.

    Each subplot shows two grouped bars: a stacked train bar (forward at
    the bottom, backward on top -- their sum equals the measured train
    latency) and a plain eval bar.  Within each group DynamicEmb and
    TorchRec sit side by side so a single panel summarizes the entire
    workload.
    """
    optimizers = sorted({r["optimizer_type"] for r in results})

    # Each subplot auto-scales its y-axis: the four modes span ~0.5 ms (GPU)
    # to ~40 ms (NoHBM) so a shared row axis would squash the GPU panel into
    # a sliver.  Independent axes keep every panel readable.
    fig, axes = plt.subplots(
        len(optimizers), len(MODE_ORDER),
        figsize=(4.0 * len(MODE_ORDER), 3.5 * len(optimizers)),
        squeeze=False,
    )

    width = 0.30
    x_train, x_eval = 0, 1
    first_handles = None  # captured from first non-empty subplot for fig-level legend
    for row, opt in enumerate(optimizers):
        for col, mode in enumerate(MODE_ORDER):
            ax = axes[row, col]
            dyn_v, trc_v = collect_suite(results, opt, mode)
            if dyn_v is None:
                ax.set_visible(False)
                continue
            dyn_fwd, dyn_bwd, dyn_train, dyn_eval = dyn_v
            trc_fwd, trc_bwd, trc_train, trc_eval = trc_v

            # Train: stacked forward (bottom) + backward (top) per backend.
            ax.bar(x_train - width / 2, dyn_fwd, width,
                   color=DYN_FWD_COLOR, label="DynamicEmb · fwd / eval")
            ax.bar(x_train - width / 2, dyn_bwd, width, bottom=dyn_fwd,
                   color=DYN_BWD_COLOR, label="DynamicEmb · bwd")
            ax.bar(x_train + width / 2, trc_fwd, width,
                   color=TRC_FWD_COLOR, label="TorchRec · fwd / eval")
            ax.bar(x_train + width / 2, trc_bwd, width, bottom=trc_fwd,
                   color=TRC_BWD_COLOR, label="TorchRec · bwd")

            # Eval is forward-only on each backend so it shares the fwd shade.
            ax.bar(x_eval - width / 2, dyn_eval, width, color=DYN_FWD_COLOR)
            ax.bar(x_eval + width / 2, trc_eval, width, color=TRC_FWD_COLOR)

            if show_values:
                # Annotate train totals (= top of stack) and eval values.
                _annotate_total(ax, x_train - width / 2, dyn_train)
                _annotate_total(ax, x_train + width / 2, trc_train)
                _annotate_total(ax, x_eval - width / 2, dyn_eval)
                _annotate_total(ax, x_eval + width / 2, trc_eval)

            ax.set_xticks([x_train, x_eval])
            ax.set_xticklabels(["train\n(fwd + bwd)", "eval"], fontsize=9)
            ax.set_ylabel("ms")
            ax.set_title(
                f"{opt} · {MODE_LABEL.get(mode, mode).replace(chr(10), ' ')}",
                fontsize=10,
            )
            ax.grid(axis="y", alpha=0.3)
            if log:
                ax.set_yscale("log")
            if first_handles is None:
                first_handles = ax.get_legend_handles_labels()

    fig.suptitle(
        "BatchedDynamicEmbeddingTablesV2 vs TorchRec TBE  (lower is better)",
        fontsize=12,
    )
    if first_handles is not None:
        handles, labels = first_handles
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=len(labels),
            frameon=False,
            fontsize=9,
        )
    # Reserve top strip for the suptitle + legend so they don't overlap axes.
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def make_speedup_figure(results: List[dict]) -> plt.Figure:
    """TorchRec / DynamicEmb latency ratio per (optimizer, mode, metric).

    Layout mirrors :func:`make_figure`: rows = optimizer, cols = storage
    mode, one panel per (optimizer, mode) suite.  Inside each panel four
    bars show the ratio for forward / backward / train / eval.  A dashed
    horizontal line at y=1.0 marks parity; bars above the line mean
    DynamicEmb is faster, below means TorchRec is faster.  Each panel
    auto-scales independently so 10× wins don't squash 0.3× cells.
    """
    optimizers = sorted({r["optimizer_type"] for r in results})

    fig, axes = plt.subplots(
        len(optimizers), len(MODE_ORDER),
        figsize=(4.0 * len(MODE_ORDER), 3.5 * len(optimizers)),
        squeeze=False,
    )

    width = 0.55
    metric_colors = [DYN_FWD_COLOR, DYN_BWD_COLOR, TRC_FWD_COLOR, TRC_BWD_COLOR]
    x = np.arange(len(METRICS))

    for row, opt in enumerate(optimizers):
        for col, mode in enumerate(MODE_ORDER):
            ax = axes[row, col]
            dyn_v, trc_v = collect_suite(results, opt, mode)
            if dyn_v is None:
                ax.set_visible(False)
                continue
            speedups = [
                trc_v[i] / dyn_v[i] if dyn_v[i] > 0 else float("nan")
                for i in range(len(METRICS))
            ]
            bars = ax.bar(
                x, speedups, width,
                color=metric_colors, edgecolor="#333", linewidth=0.4,
            )
            for b, s in zip(bars, speedups):
                if np.isnan(s):
                    continue
                ax.annotate(
                    f"{s:.2f}×",
                    xy=(b.get_x() + b.get_width() / 2, s),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                )

            ax.axhline(1.0, color="black", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(METRICS, fontsize=9)
            ax.set_ylabel("trc / dyn")
            ax.set_title(
                f"{opt} · {MODE_LABEL.get(mode, mode).replace(chr(10), ' ')}",
                fontsize=10,
            )
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "DynamicEmb Speedup over TorchRec TBE  "
        "(>1 = DynamicEmb faster, <1 = TorchRec faster)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", default=os.path.join(here, "benchmark_results.json"),
        help="path to benchmark_results.json",
    )
    parser.add_argument(
        "--out", default=os.path.join(here, "benchmark_bdet_plot.png"),
        help="output image path for the main figure "
             "(PNG / PDF / SVG by extension)",
    )
    parser.add_argument(
        "--speedup-out",
        default=os.path.join(here, "benchmark_bdet_speedup_plot.png"),
        help="output image path for the --speedup figure",
    )
    parser.add_argument("--log", action="store_true",
                        help="log-scale y-axis (useful when one mode dominates)")
    parser.add_argument("--no-values", action="store_true",
                        help="hide numeric labels on bars")
    parser.add_argument(
        "--speedup", action="store_true",
        help="also save a second figure with the trc/dyn ratio per metric, "
             "written to --speedup-out",
    )
    args = parser.parse_args()

    with open(args.results) as f:
        results = filter_timing_entries(json.load(f))
    if not results:
        raise SystemExit(f"No timing entries found in {args.results}")

    fig = make_figure(results, log=args.log, show_values=not args.no_values)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved -> {args.out}")

    if args.speedup:
        sp_fig = make_speedup_figure(results)
        sp_fig.savefig(args.speedup_out, dpi=130, bbox_inches="tight")
        print(f"Saved -> {args.speedup_out}")


if __name__ == "__main__":
    main()
