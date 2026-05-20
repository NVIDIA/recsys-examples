#!/usr/bin/env python3
"""1×3 panel: bs=8 scenario totals for len 1024 / 2048 / 4096 (perf_counter data)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_offload_scenario_total import (  # noqa: E402
    discover_summary_files,
    load_scenario_totals,
    parse_int_list,
    plot_scenario_on_axes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 1×3 bs=8 len panel.")
    parser.add_argument("--summary-root", type=str, default="profiler_python_result/summarization")
    parser.add_argument("--origin-root", type=str, default="profiler_python_result/origin_data")
    parser.add_argument(
        "--output",
        type=str,
        default="profiler_python_result/plot/microbench2_bs8_len_panel.png",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lens", type=str, default="1024,2048,4096")
    parser.add_argument(
        "--offload-batch-counts",
        type=str,
        default="50,100,150,200,250,300",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    import matplotlib.pyplot as plt

    args = parse_args()
    summary_root = (_SCRIPT_DIR / args.summary_root).resolve()
    origin_root = (_SCRIPT_DIR / args.origin_root).resolve()
    output_path = (_SCRIPT_DIR / args.output).resolve()
    lens = parse_int_list(args.lens)
    batch_counts = parse_int_list(args.offload_batch_counts)

    by_len = {
        ln: path
        for bs, ln, path in discover_summary_files(summary_root)
        if bs == args.batch_size and ln in lens
    }
    missing = [ln for ln in lens if ln not in by_len]
    if missing:
        raise SystemExit(f"Missing summary CSV for len={missing} under {summary_root}")

    # Extra width per panel: thin-segment callouts sit to the right of each bar group.
    fig, axes = plt.subplots(1, len(lens), figsize=(6.0 * len(lens), 5.8), sharey=False)
    if len(lens) == 1:
        axes = [axes]

    for ax, ln in zip(axes, lens):
        summary_path = by_len[ln]
        origin_path = origin_root / summary_path.name
        scenario_data = load_scenario_totals(
            summary_path,
            origin_path if origin_path.is_file() else None,
            batch_counts,
        )
        present = [c for c in batch_counts if c in scenario_data]
        plot_scenario_on_axes(
            ax,
            args.batch_size,
            ln,
            scenario_data,
            present,
            min_segment_label_ms=0.0,
            min_frac_inner_label=0.04,
            show_legend=(ln == lens[0]),
        )

    fig.suptitle(
        "Micro-bench 2 — scenario total (perf_counter, exclusive stack)",
        fontsize=12,
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        "Bars = put_async + launch_shell + try_wait (poll) + client.wait; "
        "line = flexkv_offload_effective_total_ms per scenario.",
        ha="center",
        fontsize=8,
        color="#555",
    )
    fig.subplots_adjust(wspace=0.32)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {output_path}")


if __name__ == "__main__":
    main()
