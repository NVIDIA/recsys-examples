#!/usr/bin/env python3
"""
Plot scenario-total FlexKV offload time vs offload_batch_count.

Stacked bars use exclusive breakdown (no finish_task + client.wait double count).
Line = sum of per-iteration flexkv_offload_effective_total_ms.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import (  # noqa: E402
    EXCLUSIVE_STACK,
    exclusive_breakdown_ms,
    exclusive_total_ms,
    scenario_exclusive_sums,
)

FILENAME_RE = re.compile(r"^offload_flexkv_bs(\d+)_len(\d+)\.csv$")

STACK_COLORS = [
    "#4C78A8",
    "#72B7B2",
    "#F58518",
    "#E45756",
    "#54A24B",
]

LABEL_ZORDER = 25
# Screen-space callout angle for thin stack segments (launch down, try_wait up).
CALLOUT_ANGLE_DEG = 60.0
CALLOUT_DX_PT = 40.0

TOTAL_COL = "sum_flexkv_offload_effective_total_ms"
LEGACY_RAW_TOTAL = "sum_flexkv_offload_total_ms"
LEGACY_AVG_TOTAL = "avg_flexkv_offload_total_ms"
LEGACY_NUM_COL = "num_iterations_averaged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scenario-total offload breakdown vs offload_batch_count."
    )
    parser.add_argument("--summary-root", type=str, default="summarization")
    parser.add_argument("--origin-root", type=str, default="origin_data")
    parser.add_argument("--output-root", type=str, default="plot")
    parser.add_argument(
        "--offload-batch-counts",
        type=str,
        default="50,100,150,200,250,300",
    )
    parser.add_argument(
        "--min-segment-label-ms",
        type=float,
        default=0.0,
        help="Minimum segment height (ms) to place label inside bar; thinner → arrow",
    )
    parser.add_argument(
        "--min-frac-inner-label",
        type=float,
        default=0.04,
        help="If segment height / stack height < this, use arrow callout",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def discover_summary_files(summary_root: Path) -> List[Tuple[int, int, Path]]:
    files: List[Tuple[int, int, Path]] = []
    for path in sorted(summary_root.glob("offload_flexkv_bs*_len*.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        files.append((int(match.group(1)), int(match.group(2)), path))
    return files


def _accumulate_raw_hook_sums(rows: List[dict]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for row in rows:
        for key, val in row.items():
            if not key.endswith("_ms"):
                continue
            if key in (
                "flexkv_offload_total_ms",
                "flexkv_offload_effective_total_ms",
                "launch_shell_ms",
                "offload_wait_shell_ms",
                "len_per_seq",
            ):
                continue
            totals[key] = totals.get(key, 0.0) + float(val)
    return totals


def load_from_origin(origin_path: Path, batch_counts: List[int]) -> Dict[int, Dict[str, float]]:
    grouped: Dict[int, List[dict]] = {c: [] for c in batch_counts}
    with origin_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty csv: {origin_path}")
        for row in reader:
            obc = int(row["offload_batch_count"])
            if obc in grouped:
                grouped[obc].append(row)

    out: Dict[int, Dict[str, float]] = {}
    for obc, rows in grouped.items():
        if not rows:
            continue
        raw = _accumulate_raw_hook_sums(rows)
        excl = scenario_exclusive_sums(raw)
        if "flexkv_offload_effective_total_ms" in rows[0]:
            effective = sum(float(r["flexkv_offload_effective_total_ms"]) for r in rows)
        else:
            effective = sum(exclusive_total_ms(r) for r in rows)
        out[obc] = {
            "offload_batch_count": float(obc),
            "num_offloads_measured": float(len(rows)),
            TOTAL_COL: effective,
            **{f"sum_{k}_ms": v for k, v in excl.items()},
        }
    return out


def _raw_totals_from_summary_row(row: dict, n: float) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    for key, val in row.items():
        if key.startswith("sum_") and key.endswith("_ms"):
            if key.startswith("sum_put_") or key.startswith("sum_launch_shell"):
                continue
            if key in (TOTAL_COL, LEGACY_RAW_TOTAL):
                continue
            raw[key] = float(val)
        elif key.startswith("avg_") and key.endswith("_ms"):
            short = key.replace("avg_", "").replace("_ms", "")
            raw[f"sum_{short}_ms"] = float(val) * n
    return raw


def load_summary_row(row: dict) -> Dict[str, float]:
    obc = int(row["offload_batch_count"])
    n = float(row.get("num_offloads_measured", row.get(LEGACY_NUM_COL, 0)))
    raw = _raw_totals_from_summary_row(row, n)
    excl = scenario_exclusive_sums(raw)
    out: Dict[str, float] = {
        "offload_batch_count": float(obc),
        "num_offloads_measured": n,
        TOTAL_COL: float(row[TOTAL_COL]) if TOTAL_COL in row else sum(excl.values()),
    }
    for key, val in excl.items():
        out[f"sum_{key}_ms"] = val
    return out


def load_scenario_totals(
    summary_path: Path,
    origin_path: Optional[Path],
    batch_counts: List[int],
) -> Dict[int, Dict[str, float]]:
    if summary_path.is_file():
        with summary_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and (
                TOTAL_COL in reader.fieldnames
                or "sum_offload_kvcache_launch_ms" in reader.fieldnames
                or LEGACY_AVG_TOTAL in reader.fieldnames
            ):
                data = {
                    int(row["offload_batch_count"]): load_summary_row(row)
                    for row in reader
                }
                return {c: data[c] for c in batch_counts if c in data}

    if origin_path is not None and origin_path.is_file():
        return load_from_origin(origin_path, batch_counts)

    raise FileNotFoundError(
        f"Need {summary_path} or origin file {origin_path}"
    )


def format_ms(v: float) -> str:
    """Adaptive precision: small poll-phase totals keep 2–4 decimal places."""
    if v <= 0.0:
        return "0"
    if v >= 10000:
        return f"{v / 1000:.2f}s"
    if v >= 1000:
        return f"{v:.1f}"
    if v >= 100:
        return f"{v:.1f}"
    if v >= 10:
        return f"{v:.2f}"
    if v >= 1:
        return f"{v:.2f}"
    if v >= 0.1:
        return f"{v:.3f}"
    return f"{v:.4f}"


class _SegmentLabel(TypedDict):
    xi: float
    y_center: float
    h: float
    stack_total: float
    text: str
    color: str
    seg_index: int


def _draw_bar_segment_labels(
    ax,
    pending: Sequence[_SegmentLabel],
    bar_width: float,
    min_inner_ms: float,
    min_frac_inner: float,
) -> None:
    """Draw labels after all bar artists so callouts are not covered by later stacks."""
    has_external = False

    for item in pending:
        xi = item["xi"]
        y_center = item["y_center"]
        h = item["h"]
        stack_total = item["stack_total"]
        text = item["text"]
        color = item["color"]
        seg_index = item["seg_index"]

        inside = h >= min_inner_ms and (
            stack_total <= 0 or h / stack_total >= min_frac_inner
        )
        if inside:
            ax.text(
                xi,
                y_center,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if h > 400 else "black",
                fontweight="bold",
                zorder=LABEL_ZORDER,
                clip_on=False,
            )
            continue

        has_external = True
        bar_right = xi + bar_width / 2.0
        # ~60° callouts in points (stable visual angle across subplots / y scales).
        fan_sign = (-1.0, -1.0, 1.0, 1.0)[seg_index % 4]
        dx_pt = CALLOUT_DX_PT + 6.0 * seg_index
        dy_pt = dx_pt * math.tan(math.radians(CALLOUT_ANGLE_DEG)) * fan_sign
        ax.annotate(
            text,
            xy=(bar_right, y_center),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            fontsize=7,
            color=color,
            fontweight="bold",
            ha="left",
            va="center",
            annotation_clip=False,
            zorder=LABEL_ZORDER,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=color,
                linewidth=0.7,
                alpha=0.95,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=0.9,
                shrinkA=2,
                shrinkB=2,
                connectionstyle="arc3,rad=0.0",
            ),
        )

    if has_external and pending:
        x_right = max(p["xi"] for p in pending) + bar_width / 2.0 + 0.35
        left, right = ax.get_xlim()
        ax.set_xlim(left, max(right, x_right))
        y_min, y_max = ax.get_ylim()
        pad = (y_max - y_min) * 0.08 if y_max > y_min else 200.0
        ax.set_ylim(y_min - pad * 0.35, y_max + pad * 0.35)


def plot_scenario_on_axes(
    ax,
    batch_size: int,
    len_per_seq: int,
    scenario_data: Dict[int, Dict[str, float]],
    batch_counts: List[int],
    min_segment_label_ms: float,
    min_frac_inner_label: float,
    *,
    show_legend: bool = True,
    title_suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(batch_counts))
    width = 0.62

    stacks: List[np.ndarray] = []
    for key, _ in EXCLUSIVE_STACK:
        col = f"sum_{key}_ms"
        stacks.append(
            np.array([scenario_data[c].get(col, 0.0) for c in batch_counts], dtype=float)
        )
    totals = np.array([scenario_data[c][TOTAL_COL] for c in batch_counts], dtype=float)
    stack_heights = np.zeros(len(batch_counts))
    for arr in stacks:
        stack_heights += arr
    bottom = np.zeros(len(batch_counts))
    bars = []
    pending_labels: List[_SegmentLabel] = []
    for i, ((key, label), color) in enumerate(zip(EXCLUSIVE_STACK, STACK_COLORS)):
        heights = stacks[i]
        bar = ax.bar(
            x,
            heights,
            width,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        bars.append(bar)
        for xi, btm, h, st_total in zip(x, bottom, heights, stack_heights):
            if h <= 0.0:
                continue
            pending_labels.append(
                _SegmentLabel(
                    xi=float(xi),
                    y_center=float(btm + h / 2.0),
                    h=float(h),
                    stack_total=float(st_total),
                    text=format_ms(h),
                    color=color,
                    seg_index=i,
                )
            )
        bottom = bottom + heights

    ax.set_xlim(-0.55, len(batch_counts) - 0.45 + width / 2.0)
    _draw_bar_segment_labels(
        ax,
        pending_labels,
        bar_width=width,
        min_inner_ms=min_segment_label_ms,
        min_frac_inner=min_frac_inner_label,
    )

    line = ax.plot(
        x,
        totals,
        color="#1a1a1a",
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="total (effective)",
        zorder=8,
    )
    for xi, tv in zip(x, totals):
        if tv <= 0.0:
            continue
        ax.annotate(
            format_ms(tv),
            (xi, tv),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#1a1a1a",
            zorder=LABEL_ZORDER,
            annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, lw=0),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in batch_counts])
    ax.set_xlabel("offload_batch_count")
    if show_legend:
        ax.set_ylabel("Scenario total time (ms)")
    title = f"bs={batch_size}, len={len_per_seq}"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))

    handles, labels = [], []
    for i, (bar, (_, lbl)) in enumerate(zip(bars, EXCLUSIVE_STACK)):
        if float(stacks[i].max()) <= 0.0:
            continue
        handles.append(bar[0])
        labels.append(lbl)
    handles.extend(line)
    labels.append("total (effective)")
    if handles and show_legend:
        ax.legend(handles, labels, loc="upper left", fontsize=7, framealpha=0.92)


def plot_one(
    batch_size: int,
    len_per_seq: int,
    scenario_data: Dict[int, Dict[str, float]],
    batch_counts: List[int],
    output_path: Path,
    min_segment_label_ms: float,
    min_frac_inner_label: float,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.8))
    plot_scenario_on_axes(
        ax,
        batch_size,
        len_per_seq,
        scenario_data,
        batch_counts,
        min_segment_label_ms,
        min_frac_inner_label,
        show_legend=True,
        title_suffix="try_wait=poll; wait=client.wait",
    )
    fig.text(
        0.5,
        0.01,
        "try_wait & client.wait are sequential (no wall-time overlap). "
        "Host wait timer wraps try_wait (nested accounting only).",
        ha="center",
        fontsize=7.5,
        color="#555",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {output_path}")


def main() -> None:
    import matplotlib  # noqa: F401

    args = parse_args()
    summary_root = Path(args.summary_root).expanduser().resolve()
    origin_root = Path(args.origin_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    batch_counts = parse_int_list(args.offload_batch_counts)

    files = discover_summary_files(summary_root)
    if not files:
        raise SystemExit(f"No summary files under {summary_root}")

    for batch_size, len_per_seq, summary_path in files:
        origin_path = origin_root / summary_path.name
        scenario_data = load_scenario_totals(
            summary_path, origin_path if origin_path.is_file() else None, batch_counts
        )
        missing = [c for c in batch_counts if c not in scenario_data]
        if missing:
            print(
                f"[WARN] bs={batch_size} len={len_per_seq}: missing batch_counts {missing}"
            )
        out_path = output_root / (
            f"scenario_total_bs{batch_size}_len{len_per_seq}.png"
        )
        plot_one(
            batch_size=batch_size,
            len_per_seq=len_per_seq,
            scenario_data=scenario_data,
            batch_counts=[c for c in batch_counts if c in scenario_data],
            output_path=out_path,
            min_segment_label_ms=args.min_segment_label_ms,
            min_frac_inner_label=args.min_frac_inner_label,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
