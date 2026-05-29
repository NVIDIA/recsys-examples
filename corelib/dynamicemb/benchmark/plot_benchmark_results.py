# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Plot benchmark_results.json produced by benchmark_batched_dynamicemb_tables.sh.
#
# Reads a list of per-config result dicts, groups by (optimizer, mode), and
# renders a one-axes-per-optimizer figure: the GPU / Caching (one entry per
# cache_footprint_ratio) / NoCaching / NoHBM panels sit side-by-side on a
# shared y-axis, each panel showing forward / backward / train / eval bars
# for DynamicEmb vs TorchRec.
#
# Usage:
#   python plot_benchmark_results.py
#   python plot_benchmark_results.py --results benchmark_results.json --out-dir plots/
#   python plot_benchmark_results.py --log              # log-scale y-axis
#   python plot_benchmark_results.py --no-values        # hide bar labels

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt

# Canonical ordering for the four storage modes the suite parametrizes over.
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


def build_panels(results: List[dict]):
    """Column ordering for figure panels: (mode, cache_footprint_ratio).

    ``cache_footprint_ratio`` is None for non-caching modes; for caching it
    expands one column per ratio found in the data, sorted ascending.  This
    keeps cfr=0.8 and cfr=1.0 in the same figure rather than splitting them
    into separate PNGs.
    """
    ratios = sorted(
        {
            r.get("cache_footprint_ratio")
            for r in results
            if r.get("cache_footprint_ratio") is not None
        }
    )
    panels = [("gpu", None)]
    if ratios:
        panels.extend([("caching", r) for r in ratios])
    else:
        panels.append(("caching", None))
    panels.append(("no_caching", None))
    panels.append(("no_hbm", None))
    return panels


def panel_label(mode: str, cfr) -> str:
    """Title text for a panel, gpu_ratio-free."""
    if mode == "gpu":
        return "GPU\n(full HBM)"
    if mode == "caching":
        return f"Caching\ncfr={cfr}" if cfr is not None else "Caching"
    if mode == "no_caching":
        return "NoCaching"
    if mode == "no_hbm":
        return "NoHBM\n(UVM)"
    return mode


def collect_panel(results: List[dict], optimizer: str, mode: str, cfr):
    """Return (dyn_values, trc_values) over METRICS for one panel.

    A panel is uniquely identified by ``(optimizer, mode, cache_footprint_ratio)``.
    Returns ``(None, None)`` when no matching result is present.
    """
    for r in results:
        if r.get("optimizer_type") != optimizer:
            continue
        if parse_mode(r["label"]) != mode:
            continue
        if r.get("cache_footprint_ratio") != cfr:
            continue
        dyn_v = [r[f"dyn_{m}_ms"] for m in METRICS]
        trc_v = [r[f"trc_{m}_ms"] for m in METRICS]
        return dyn_v, trc_v
    return None, None


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


def make_figure(
    results: List[dict],
    panels: List[tuple],
    log: bool,
    show_values: bool,
    subtitle: str = "",
) -> plt.Figure:
    """One axes per optimizer with all ``panels`` drawn side by side.

    ``panels`` is a list of ``(mode, cache_footprint_ratio)`` tuples produced
    by :func:`build_panels` -- caching is fanned out into one entry per
    ratio so cfr=0.8 / cfr=1.0 sit side by side.

    The figure has ``n_optimizers`` rows × 1 col; within each row, every
    panel contributes a (stacked train, eval) bar group, all sharing the
    same y-axis.  Note: modes span ~0.5 ms (GPU) to ~40 ms (NoHBM) so a
    shared y-axis squashes the GPU bars -- pass ``log=True`` if that
    matters.  Panel headers live on a secondary x-axis at the top of each
    axes, dotted vertical lines separate the panel regions.
    """
    optimizers = sorted({r["optimizer_type"] for r in results})

    fig, axes = plt.subplots(
        len(optimizers),
        1,
        figsize=(max(2.6 * len(panels), 10.0), 4.2 * len(optimizers)),
        squeeze=False,
    )
    axes = [row[0] for row in axes]

    width = 0.30
    panel_inner_width = 2.0  # group positions: x=0 (train) and x=1 (eval)
    panel_gap = 0.9  # blank space between panels
    panel_step = panel_inner_width + panel_gap

    first_handles = None  # captured from first non-empty group for fig-level legend
    for row, opt in enumerate(optimizers):
        ax = axes[row]
        for col, (mode, cfr) in enumerate(panels):
            dyn_v, trc_v = collect_panel(results, opt, mode, cfr)
            if dyn_v is None:
                continue
            dyn_fwd, dyn_bwd, dyn_train, dyn_eval = dyn_v
            trc_fwd, trc_bwd, trc_train, trc_eval = trc_v

            x_train = col * panel_step
            x_eval = col * panel_step + 1

            # Train: stacked forward (bottom) + backward (top) per backend.
            ax.bar(
                x_train - width / 2,
                dyn_fwd,
                width,
                color=DYN_FWD_COLOR,
                label="DynamicEmb · fwd / eval",
            )
            ax.bar(
                x_train - width / 2,
                dyn_bwd,
                width,
                bottom=dyn_fwd,
                color=DYN_BWD_COLOR,
                label="DynamicEmb · bwd",
            )
            ax.bar(
                x_train + width / 2,
                trc_fwd,
                width,
                color=TRC_FWD_COLOR,
                label="TorchRec · fwd / eval",
            )
            ax.bar(
                x_train + width / 2,
                trc_bwd,
                width,
                bottom=trc_fwd,
                color=TRC_BWD_COLOR,
                label="TorchRec · bwd",
            )

            # Eval is forward-only on each backend so it shares the fwd shade.
            ax.bar(x_eval - width / 2, dyn_eval, width, color=DYN_FWD_COLOR)
            ax.bar(x_eval + width / 2, trc_eval, width, color=TRC_FWD_COLOR)

            if show_values:
                # Annotate train totals (= top of stack) and eval values.
                _annotate_total(ax, x_train - width / 2, dyn_train)
                _annotate_total(ax, x_train + width / 2, trc_train)
                _annotate_total(ax, x_eval - width / 2, dyn_eval)
                _annotate_total(ax, x_eval + width / 2, trc_eval)

            if first_handles is None:
                first_handles = ax.get_legend_handles_labels()

        # Dotted separators between panel regions
        for i in range(1, len(panels)):
            sep_x = i * panel_step - panel_gap / 2
            ax.axvline(sep_x, color="#aaa", linestyle=":", linewidth=0.8, alpha=0.7)

        # Inner x-ticks: train / eval label under every panel.
        xticks: list = []
        xticklabels: list = []
        for i in range(len(panels)):
            xticks.extend([i * panel_step, i * panel_step + 1])
            xticklabels.extend(["train\n(fwd+bwd)", "eval"])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.set_xlim(-panel_gap / 2, len(panels) * panel_step - panel_gap / 2)
        ax.set_ylabel("ms")
        ax.grid(axis="y", alpha=0.3)
        if log:
            ax.set_yscale("log")

        # Secondary x-axis on top: panel header per panel region.
        sax = ax.secondary_xaxis("top")
        sax.set_xticks([i * panel_step + 0.5 for i in range(len(panels))])
        sax.set_xticklabels(
            [panel_label(m, c).replace("\n", " ") for m, c in panels],
            fontsize=10,
            fontweight="bold",
        )
        sax.tick_params(axis="x", length=0)  # no tick marks, just labels

        # Optimizer name above the panel-header strip.
        ax.set_title(opt, fontsize=12, fontweight="bold", pad=24)

    fig.suptitle(
        "BatchedDynamicEmbeddingTablesV2 vs TorchRec TBE  (lower is better)",
        fontsize=12,
    )
    # n_lines: vertical-space budget for the subtitle strip (0 = no subtitle,
    # 1 = hardware only, 2 = hardware + workload).  Each line takes ~2.5% of
    # the figure height; legend and tight_layout shift down accordingly.
    n_lines = subtitle.count("\n") + 1 if subtitle else 0
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=10, color="#444")
    if first_handles is not None:
        handles, labels = first_handles
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95 - 0.025 * n_lines),
            ncol=len(labels),
            frameon=False,
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.93 - 0.025 * n_lines))
    return fig


def _meta_str(results: List[dict]):
    """Two-line header strip.

    Line 1: hardware / model dims (GPU, D, batch).
    Line 2: input data (distribution, hotness, pool).

    ``cache_footprint_ratio`` is intentionally NOT included here: with the
    single-figure layout the caching column shows one panel per ratio, so
    the ratio is already visible in the panel titles.

    Each field is skipped when its source key is absent from the result dict
    so legacy JSONs degrade gracefully.
    """
    if not results:
        return ""
    r0 = results[0]

    hw = []
    if r0.get("gpu_name"):
        hw.append(str(r0["gpu_name"]))
    if r0.get("embedding_dim") is not None:
        hw.append(f"D={r0['embedding_dim']}")
    if r0.get("batch_size") is not None and r0.get("num_tables") is not None:
        hw.append(f"batch={r0['batch_size'] * r0['num_tables']:,}")

    wk = []
    dist = r0.get("feature_distribution")
    alpha = r0.get("alpha")
    if dist:
        if alpha is not None and dist in ("pow-law", "zipf"):
            wk.append(f"{dist}(α={alpha})")
        else:
            wk.append(str(dist))
    if r0.get("max_hotness") is not None:
        wk.append(f"hotness={r0['max_hotness']}")
    if r0.get("pooling_mode"):
        wk.append(f"pool={r0['pooling_mode']}")

    lines = []
    if hw:
        lines.append("  ·  ".join(hw))
    if wk:
        lines.append("  ·  ".join(wk))
    return "\n".join(lines)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=os.path.join(here, "benchmark_results.json"),
        help="path to benchmark_results.json",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(here, "plots"),
        help="directory to write generated PNGs into (created if missing)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="log-scale y-axis (useful when one mode dominates)",
    )
    parser.add_argument(
        "--no-values", action="store_true", help="hide numeric labels on bars"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results) as f:
        results = filter_timing_entries(json.load(f))
    if not results:
        raise SystemExit(f"No timing entries found in {args.results}")

    # All cache_footprint_ratio variants share a single figure now: the
    # caching column is fanned out into one panel per ratio, so cfr=0.8 and
    # cfr=1.0 sit side-by-side instead of going to separate PNGs.
    panels = build_panels(results)
    subtitle = _meta_str(results)

    def _save(fig, name):
        path = os.path.join(args.out_dir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"Saved -> {path}")

    fig = make_figure(
        results, panels, log=args.log, show_values=not args.no_values, subtitle=subtitle
    )
    _save(fig, "benchmark_bdet_plot.png")


if __name__ == "__main__":
    main()
