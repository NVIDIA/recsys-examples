#!/usr/bin/env python3
"""
Plot per-iteration FlexKV offload timings from origin_data CSVs.

For each (request_batch_size, len_per_seq) run file, emit one figure with a 2x2 grid.
Each panel:
  - Line: inlier points only (outliers excluded from the line)
  - Red "x" markers: outliers (not connected)
  - Broken y-axis when outliers exist (main range shows trend, top range shows spikes)
  - Independent y-axis scale per panel
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import exclusive_total_ms_from_row  # noqa: E402

FILENAME_RE = re.compile(r"^offload_flexkv_bs(\d+)_len(\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot iteration curves for offload stress tests.")
    parser.add_argument("--origin-root", type=str, default="origin_data")
    parser.add_argument("--output-root", type=str, default="plot")
    parser.add_argument(
        "--metric",
        type=str,
        default="flexkv_offload_effective_total_ms",
        help=(
            "CSV column to plot. Use flexkv_offload_effective_total_ms (default) to avoid "
            "double-counting finish_task + client.wait; raw flexkv_offload_total_ms sums all hooks."
        ),
    )
    parser.add_argument(
        "--offload-batch-counts",
        type=str,
        default="50,100,150,200",
    )
    parser.add_argument(
        "--outlier-iqr",
        type=float,
        default=1.5,
        help="Upper fence = Q3 + iqr_mult * IQR (0 disables outlier split)",
    )
    parser.add_argument(
        "--break-height-ratio",
        type=float,
        default=0.22,
        help="Top axes height ratio when y-axis break is used",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def discover_origin_files(origin_root: Path) -> List[Tuple[int, int, Path]]:
    files: List[Tuple[int, int, Path]] = []
    for path in sorted(origin_root.glob("offload_flexkv_bs*_len*.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        files.append((int(match.group(1)), int(match.group(2)), path))
    return files


def _row_metric_value(row: dict, metric: str) -> float:
    if metric == "flexkv_offload_effective_total_ms":
        return exclusive_total_ms_from_row(row)
    return float(row[metric])


def load_scenarios_metric(csv_path: Path, metric: str) -> Dict[int, List[Tuple[int, float]]]:
    grouped: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise KeyError(f"empty csv: {csv_path}")
        if metric not in reader.fieldnames and metric != "flexkv_offload_effective_total_ms":
            raise KeyError(f"metric '{metric}' not in {csv_path}")
        for row in reader:
            grouped[int(row["offload_batch_count"])].append(
                (int(row["iteration"]), _row_metric_value(row, metric))
            )
    for scenario in grouped:
        grouped[scenario].sort(key=lambda x: x[0])
    return grouped


def split_inliers_outliers(
    xs: Sequence[int],
    ys: Sequence[float],
    iqr_mult: float,
) -> Tuple[List[int], List[float], List[int], List[float]]:
    if iqr_mult <= 0 or len(ys) < 4:
        return list(xs), list(ys), [], []

    arr = np.asarray(ys, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return list(xs), list(ys), [], []

    upper = q3 + iqr_mult * iqr
    inlier_x, inlier_y, outlier_x, outlier_y = [], [], [], []
    for x, y in zip(xs, ys):
        if y > upper:
            outlier_x.append(x)
            outlier_y.append(y)
        else:
            inlier_x.append(x)
            inlier_y.append(y)
    return inlier_x, inlier_y, outlier_x, outlier_y


def mark_yaxis_break(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, length=0)
    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="0.35", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs["transform"] = ax_bottom.transAxes
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def style_yaxis(ax: plt.Axes, nbins: int = 6) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=4))
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)


def plot_series_on_axes(
    ax_main: plt.Axes,
    ax_top: Optional[plt.Axes],
    inlier_x: List[int],
    inlier_y: List[float],
    outlier_x: List[int],
    outlier_y: List[float],
    metric: str,
) -> None:
    if inlier_x:
        ax_main.plot(
            inlier_x,
            inlier_y,
            color="C0",
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label="inlier (line)",
            zorder=2,
        )
        y_main_max = max(inlier_y)
        pad = max(y_main_max * 0.12, 1.0)
        ax_main.set_ylim(0, y_main_max + pad)
    else:
        ax_main.set_ylim(0, 1)

    if inlier_y and len(inlier_y) >= 3:
        ax_main.axhline(
            sum(inlier_y[:3]) / 3,
            color="C1",
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
        )
        ax_main.axhline(
            sum(inlier_y[-3:]) / 3,
            color="C2",
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
        )

    style_yaxis(ax_main, nbins=6)

    if ax_top is not None and outlier_x:
        ax_top.scatter(
            outlier_x,
            outlier_y,
            c="C3",
            marker="x",
            s=55,
            linewidths=1.8,
            label="outlier",
            zorder=5,
        )
        o_min = min(outlier_y)
        o_max = max(outlier_y)
        span = max(o_max - o_min, o_max * 0.05, 1.0)
        ax_top.set_ylim(o_min - span * 0.15, o_max + span * 0.15)
        style_yaxis(ax_top, nbins=3)
        ax_top.grid(True, alpha=0.25)


def plot_panel(
    fig: plt.Figure,
    outer_spec: gridspec.SubplotSpec,
    series: Optional[List[Tuple[int, float]]],
    count: int,
    metric: str,
    iqr_mult: float,
    break_height_ratio: float,
) -> None:
    if not series:
        ax = fig.add_subplot(outer_spec)
        ax.set_title(f"offload_batch_count={count} (missing)")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    in_x, in_y, out_x, out_y = split_inliers_outliers(xs, ys, iqr_mult)

    title = f"offload_batch_count={count} (n={len(xs)}"
    if out_x:
        title += f", {len(out_x)} outlier"
    title += ")"

    if out_x and in_y:
        ratio = max(min(break_height_ratio, 0.45), 0.12)
        inner = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer_spec,
            height_ratios=[ratio, 1.0 - ratio],
            hspace=0.08,
        )
        ax_top = fig.add_subplot(inner[0])
        ax_main = fig.add_subplot(inner[1], sharex=ax_top)
        mark_yaxis_break(ax_top, ax_main)
        plot_series_on_axes(ax_main, ax_top, in_x, in_y, out_x, out_y, metric)
        ax_main.set_xlabel("iteration")
        ax_main.set_ylabel(f"{metric} (ms)")
        ax_top.set_ylabel("outlier\n(ms)", fontsize=8)
    else:
        ax_main = fig.add_subplot(outer_spec)
        plot_series_on_axes(ax_main, None, in_x, in_y, out_x, out_y, metric)
        ax_main.set_xlabel("iteration")
        ax_main.set_ylabel(f"{metric} (ms)")

    ax_main.set_title(title)
    ax_main.grid(True, alpha=0.3)


def plot_one_config(
    batch_size: int,
    len_per_seq: int,
    csv_path: Path,
    offload_counts: List[int],
    metric: str,
    output_path: Path,
    iqr_mult: float,
    break_height_ratio: float,
    dpi: int,
) -> None:
    data = load_scenarios_metric(csv_path, metric)

    fig = plt.figure(figsize=(14, 10))
    subtitle = (
        "effective total = launch + wait(poll) + wait(done), no finish_task/client.wait overlap"
        if metric == "flexkv_offload_effective_total_ms"
        else "raw metric (may double-count nested hooks)"
    )
    fig.suptitle(
        f"FlexKV offload iteration — bs={batch_size}, len={len_per_seq}, metric={metric}\n"
        f"{subtitle}; line excludes outliers; red x = outlier",
        fontsize=11,
    )
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, count in enumerate(offload_counts[:4]):
        row, col = positions[idx]
        plot_panel(
            fig=fig,
            outer_spec=outer[row, col],
            series=data.get(count),
            count=count,
            metric=metric,
            iqr_mult=iqr_mult,
            break_height_ratio=break_height_ratio,
        )

    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def main() -> None:
    args = parse_args()
    origin_root = Path(args.origin_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    offload_counts = parse_int_list(args.offload_batch_counts)
    if len(offload_counts) != 4:
        offload_counts = (offload_counts + [0, 0, 0, 0])[:4]

    files = discover_origin_files(origin_root)
    if not files:
        raise FileNotFoundError(f"No offload_flexkv_bs*_len*.csv under {origin_root}")

    print(f"[INFO] origin_root={origin_root}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] metric={args.metric}, outlier_iqr={args.outlier_iqr}")

    for batch_size, len_per_seq, csv_path in files:
        out_name = f"iteration_bs{batch_size}_len{len_per_seq}_{args.metric}.png"
        plot_one_config(
            batch_size=batch_size,
            len_per_seq=len_per_seq,
            csv_path=csv_path,
            offload_counts=offload_counts,
            metric=args.metric,
            output_path=output_root / out_name,
            iqr_mult=args.outlier_iqr,
            break_height_ratio=args.break_height_ratio,
            dpi=args.dpi,
        )

    print(f"[DONE] wrote figures to {output_root}")


if __name__ == "__main__":
    main()
