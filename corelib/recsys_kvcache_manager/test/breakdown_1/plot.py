#!/usr/bin/env python3

import argparse
import glob
import math
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Wedge


def parse_int_list(spec: str, arg_name: str) -> List[int]:
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"{arg_name} cannot be empty")
    if any(v <= 0 for v in values):
        raise ValueError(f"{arg_name} values must be positive integers")
    return values


def find_nvtx_csv(csv_root: str, mode_dir: str, case_tag: str) -> str:
    mode_path = os.path.join(csv_root, mode_dir)
    pattern = os.path.join(mode_path, f"{case_tag}*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        available = sorted(glob.glob(os.path.join(mode_path, "*.csv")))
        available_short = [os.path.basename(p) for p in available[:10]]
        raise FileNotFoundError(
            "NVTX CSV not found. "
            f"pattern={pattern}, mode_dir={mode_dir}, case_tag={case_tag}, "
            f"available_examples={available_short}"
        )

    def priority(path: str) -> Tuple[int, str]:
        name = os.path.basename(path)
        if "nvtxsum" in name:
            return (0, name)
        if "nvtxppsum" in name:
            return (1, name)
        if "nvtx_pushpop_sum" in name:
            return (2, name)
        if "nvtx" in name:
            return (3, name)
        return (4, name)

    matches = sorted(matches, key=priority)
    return matches[0]


def detect_label_and_time_columns(df: pd.DataFrame) -> Tuple[str, str, float]:
    label_candidates = ["Range", "Name", "NVTX Range"]
    time_candidates = [
        ("Total Time (ns)", 1e-6),
        ("Total Time (us)", 1e-3),
        ("Total Time (ms)", 1.0),
        ("Total Time", 1.0),
    ]

    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"Cannot find NVTX label column in {list(df.columns)}")

    time_col = None
    scale_to_ms = None
    for c, scale in time_candidates:
        if c in df.columns:
            time_col = c
            scale_to_ms = scale
            break
    if time_col is None:
        raise ValueError(f"Cannot find NVTX time column in {list(df.columns)}")
    return label_col, time_col, scale_to_ms


def default_prefixes_for_step_flow() -> List[str]:
    return ["step1.", "step2.", "step3."]


def matches_any_prefix(label: str, prefixes: List[str]) -> bool:
    if not prefixes:
        return True
    parts = label.split("::")
    for prefix in prefixes:
        if label.startswith(prefix):
            return True
        for part in parts[1:]:
            if part.startswith(prefix):
                return True
    return False


def load_mode_breakdown(
    csv_root: str,
    mode_dir: str,
    x_values: List[int],
    case_pattern: str,
    include_prefixes: List[str],
    *,
    group_by_step_op: bool = False,
) -> pd.DataFrame:
    records: Dict[int, Dict[str, float]] = {}
    for x in x_values:
        case_tag = case_pattern.format(value=x)
        csv_path = find_nvtx_csv(csv_root, mode_dir, case_tag)
        df = pd.read_csv(csv_path, comment="#")
        label_col, time_col, scale_to_ms = detect_label_and_time_columns(df)
        df = df[[label_col, time_col]].copy()
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0)
        df["time_ms"] = df[time_col] * scale_to_ms
        df["label_norm"] = df[label_col].astype(str).str.strip().str.lstrip(":")
        if include_prefixes:
            mask = df["label_norm"].apply(
                lambda lbl: matches_any_prefix(lbl, include_prefixes)
            )
            df = df[mask]
        if group_by_step_op:
            df = df[~df["label_norm"].str.contains("::", regex=False)]
            df["group_label"] = df["label_norm"].apply(
                lambda s: re.match(r"(step\d+\.[^.:\s]+)", s).group(1)
                if re.match(r"(step\d+\.[^.:\s]+)", s)
                else s
            )
        else:
            df["group_label"] = df["label_norm"]
        grouped = df.groupby("group_label")["time_ms"].sum().to_dict()
        records[x] = grouped

    breakdown_df = pd.DataFrame.from_dict(records, orient="index").fillna(0.0)
    breakdown_df.index.name = "x_value"
    return breakdown_df.sort_index()


def _timeline_sort_key(label: str) -> Tuple[int, int, int, str, str]:
    s = label.lstrip(":")
    top, sep, suffix = s.partition("::")

    gpu_wall_match = re.match(r"gpu_wall_clock\.step(\d+)_(.+)$", top)
    if gpu_wall_match:
        step_num = int(gpu_wall_match.group(1))
        op = gpu_wall_match.group(2)
        op_order = {
            "input": 0,
            "lookup": 1,
            "allocate": 2,
            "offload_launch": 3,
            "offload_wait": 4,
            "evict_gpu": 5,
            "onboard_launch": 6,
            "onboard_wait": 7,
            "verify_get": 8,
            "post_lookup": 9,
        }
        return (1, step_num, op_order.get(op, 99), op, suffix)

    step_match = re.match(r"step(\d+)\.([^.:\s]+)", top)
    if step_match:
        step_num = int(step_match.group(1))
        op = step_match.group(2)
        op_order = {
            "input": 0,
            "lookup": 1,
            "allocate": 2,
            "offload_launch": 3,
            "offload_wait": 4,
            "evict_gpu": 5,
            "onboard_launch": 6,
            "onboard_wait": 7,
            "verify_get": 8,
            "post_lookup": 9,
        }
        nested_rank = 1 if sep else 0
        return (1, step_num, op_order.get(op, 99), op, f"{nested_rank}:{suffix}")

    if "try_wait" in s or "offload_wait" in s:
        return (1, 1, 4, "offload_wait", s)
    if "evict" in s:
        return (1, 2, 5, "evict_gpu", s)
    if "onboard_launch" in s or "launch_cpp" in s:
        return (1, 3, 6, "onboard_launch", s)
    if "onboard_wait" in s:
        return (1, 3, 7, "onboard_wait", s)

    return (99, 99, 99, top, suffix)


def add_value_labels_with_arrows(
    ax: plt.Axes,
    x_positions: List[float],
    heights_by_component: List[List[float]],
    totals: List[float],
    *,
    bar_width: float = 0.65,
    component_colors: Optional[List] = None,
) -> None:
    max_total = max(totals) if totals else 0.0
    if max_total <= 0:
        return

    ax.figure.canvas.draw()
    font_size = 8
    bar_half_width = bar_width / 2.0
    x_offset_pts = 8
    y_clearance_pts = 11
    placed_external_px: List[Tuple[float, float]] = []

    def overlaps_px(px: float, py: float) -> bool:
        for prev_x, prev_y in placed_external_px:
            if abs(px - prev_x) < 28 and abs(py - prev_y) < y_clearance_pts:
                return True
        return False

    bottoms = [0.0 for _ in x_positions]
    for comp_idx, comp_vals in enumerate(heights_by_component):
        seg_color = "black"
        if component_colors is not None and comp_idx < len(component_colors):
            seg_color = component_colors[comp_idx]
        for i, v in enumerate(comp_vals):
            if v <= 0:
                bottoms[i] += v
                continue
            x = x_positions[i]
            y_center = bottoms[i] + v / 2.0
            label = f"{v:.5f}"
            yb0 = ax.transData.transform((x, bottoms[i]))[1]
            yb1 = ax.transData.transform((x, bottoms[i] + v))[1]
            seg_h_px = abs(yb1 - yb0)
            can_fit_inside = seg_h_px > font_size + 2
            if can_fit_inside:
                ax.text(
                    x,
                    y_center,
                    label,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="black",
                    clip_on=True,
                    zorder=6,
                )
            else:
                anchor_px = ax.transData.transform((x + bar_half_width, y_center))
                text_px = (anchor_px[0] + x_offset_pts, anchor_px[1])
                for level in range(0, 24):
                    direction = 1 if level % 2 == 0 else -1
                    magnitude = level // 2 + 1
                    candidate = (
                        text_px[0],
                        anchor_px[1] + direction * magnitude * y_clearance_pts,
                    )
                    if not overlaps_px(candidate[0], candidate[1]):
                        text_px = candidate
                        break

                ax.annotate(
                    label,
                    xy=(x + bar_half_width, y_center),
                    xytext=(x_offset_pts, text_px[1] - anchor_px[1]),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=font_size,
                    color=seg_color,
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=0.75,
                        shrinkA=1,
                        shrinkB=1,
                        color=seg_color,
                    ),
                    clip_on=False,
                    zorder=7,
                )
                placed_external_px.append(text_px)
            bottoms[i] += v


def plot_stacked_breakdown(
    df: pd.DataFrame,
    title: str,
    x_label: str,
    output_png: str,
    *,
    stack_order: str = "timeline",
) -> None:
    if df.empty:
        raise ValueError("Breakdown dataframe is empty, nothing to plot.")

    if stack_order == "input":
        ordered_components = list(df.columns)
    elif stack_order == "size":
        ordered_components = list(df.sum(axis=0).sort_values(ascending=False).index)
    else:
        ordered_components = sorted(df.columns, key=_timeline_sort_key)

    plot_df = df[ordered_components].copy()
    x_values = list(plot_df.index)
    x_positions = list(range(len(x_values)))
    bottoms = [0.0 for _ in x_values]
    heights_by_component: List[List[float]] = []
    bar_width = 0.65
    components = list(plot_df.columns)

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap("tab20")
    component_colors: List = []
    for i, comp in enumerate(components):
        vals = plot_df[comp].tolist()
        heights_by_component.append(vals)
        bar_color = cmap(i % 20)
        component_colors.append(bar_color)
        ax.bar(
            x_positions,
            vals,
            bottom=bottoms,
            label=comp,
            color=bar_color,
            width=bar_width,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    add_value_labels_with_arrows(
        ax,
        x_positions=x_positions,
        heights_by_component=heights_by_component,
        totals=bottoms,
        bar_width=bar_width,
        component_colors=component_colors,
    )

    ax.set_xticks(x_positions, [str(s) for s in x_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Total Time (ms)")
    ax.set_title(title)
    ax.set_xlim(-0.55, len(x_positions) - 0.45)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(right=0.72)
    fig.tight_layout()
    out_dir = os.path.dirname(output_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sum_columns(df: pd.DataFrame, pred: Callable[[str], bool]) -> pd.Series:
    cols = [c for c in df.columns if pred(c)]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].sum(axis=1)


def sum_metric(
    detail_df: pd.DataFrame,
    step_op: Optional[str],
    token: str,
    *,
    leaf_only: bool = False,
) -> pd.Series:
    bare = token.lstrip(":")

    def include_column(col: str) -> bool:
        if leaf_only:
            if col != bare and not col.endswith(f"::{bare}"):
                return False
        elif bare not in col and f"::{bare}" not in col:
            return False
        if step_op is None:
            return True
        if col.startswith(f"{step_op}::"):
            return True
        if col == bare:
            return step_op.startswith("step1.") or step_op.startswith("step3.")
        if col.endswith(f"::{bare}"):
            top_scope = col.split("::", 1)[0]
            if re.match(r"step\d+\.", top_scope):
                return False
            return step_op.startswith("step1.") or step_op.startswith("step3.")
        return False

    return sum_columns(detail_df, include_column)


def drop_zero_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [c for c in df.columns if float(df[c].sum()) > 1e-9]
    return df[keep].copy()


L2_CALL_TIMELINE: List[Tuple[str, Optional[str], str]] = [
    ("step1.input", None, "step1.input"),
    ("step1.gpu.lookup_py", "step1.lookup", "::gpu.lookup_py"),
    ("step1.flexkv.build_index_meta", "step1.lookup", "::flexkv.build_index_meta"),
    (
        "step1.flexkv.adapter.to_get_match_requests",
        "step1.lookup",
        "::flexkv.adapter.to_get_match_requests",
    ),
    ("step1.flexkv.client.get_match", "step1.lookup", "::flexkv.client.get_match"),
    (
        "step1.recsys.merge_lookup_results",
        "step1.lookup",
        "::recsys.merge_lookup_results",
    ),
    ("step1.gpu.allocate_py", "step1.allocate", "::gpu.allocate_py"),
    (
        "step1.gpu.acquire_offload_pages_py",
        "step1.offload_launch",
        "::gpu.acquire_offload_pages_py",
    ),
    (
        "step1.flexkv._build_slot_mappings",
        "step1.offload_launch",
        "::flexkv._build_slot_mappings",
    ),
    (
        "step1.flexkv.client.put_async",
        "step1.offload_launch",
        "::flexkv.client.put_async",
    ),
    ("step1.flexkv.client.try_wait", "step1.offload_wait", "::flexkv.client.try_wait"),
    ("step1.flexkv.finish_task", "step1.offload_wait", "::flexkv.finish_task"),
    (
        "step1.gpu.release_offload_pages_py",
        "step1.offload_wait",
        "::gpu.release_offload_pages_py",
    ),
    ("step2.gpu.evict_py", "step2.evict_gpu", "::gpu.evict_py"),
    ("step3.input", None, "step3.input"),
    ("step3.gpu.lookup_py", "step3.lookup", "::gpu.lookup_py"),
    ("step3.flexkv.build_index_meta", "step3.lookup", "::flexkv.build_index_meta"),
    (
        "step3.flexkv.adapter.to_get_match_requests",
        "step3.lookup",
        "::flexkv.adapter.to_get_match_requests",
    ),
    ("step3.flexkv.client.get_match", "step3.lookup", "::flexkv.client.get_match"),
    (
        "step3.recsys.merge_lookup_results",
        "step3.lookup",
        "::recsys.merge_lookup_results",
    ),
    ("step3.gpu.allocate_py", "step3.allocate", "::gpu.allocate_py"),
    (
        "step3.flexkv._build_slot_mappings",
        "step3.onboard_launch",
        "::flexkv._build_slot_mappings",
    ),
    ("step3.flexkv.client.launch", "step3.onboard_launch", "::flexkv.client.launch"),
    ("step3.flexkv.client.wait", "step3.onboard_wait", "::flexkv.client.wait"),
]

LEAF_TOKENS = {
    "flexkv.finish_task",
    "flexkv.client.try_wait",
    "flexkv.client.wait",
    "flexkv.client.get_match",
    "flexkv.client.put_async",
    "flexkv.client.launch",
}


def build_l2_call_timeline_df(detail_step_df: pd.DataFrame, index) -> pd.DataFrame:
    data = {}
    for col, step_op, token in L2_CALL_TIMELINE:
        bare = token.lstrip(":")
        leaf_only = bare in LEAF_TOKENS or bare.endswith("_py")
        data[col] = sum_metric(detail_step_df, step_op, token, leaf_only=leaf_only)
    return drop_zero_columns(pd.DataFrame(data, index=index))


def reorder_l2_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    preferred = [name for name, _, _ in L2_CALL_TIMELINE if name in df.columns]
    tail = [c for c in df.columns if c not in preferred]
    return df[preferred + tail].copy()


STEP_OP_TIMELINE = [
    "step1.lookup",
    "step1.allocate",
    "step1.offload_launch",
    "step1.offload_wait",
    "step2.evict_gpu",
    "step3.lookup",
    "step3.allocate",
    "step3.onboard_launch",
    "step3.onboard_wait",
]


def reorder_step_op_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    front = [c for c in STEP_OP_TIMELINE if c in df.columns]
    tail = [c for c in df.columns if c not in front]
    return df[front + tail].copy()


def drop_step2_step_ops(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in df.columns if not str(c).startswith("step2.")]
    return df[cols].copy()


def warn_l2_exceeds_l1(step_op_df: pd.DataFrame, l2_df: pd.DataFrame) -> None:
    if step_op_df.empty or l2_df.empty:
        return

    l2_by_op: Dict[str, pd.Series] = {}
    for col, step_op, _ in L2_CALL_TIMELINE:
        if step_op is None or col not in l2_df.columns:
            continue
        if step_op not in l2_by_op:
            l2_by_op[step_op] = pd.Series(0.0, index=l2_df.index)
        l2_by_op[step_op] = l2_by_op[step_op] + l2_df[col]

    for step_op, l2_sum in l2_by_op.items():
        if step_op not in step_op_df.columns:
            continue
        l1_total = step_op_df[step_op]
        exceeded = l2_sum > (l1_total + 1e-6)
        if exceeded.any():
            xs = [str(x) for x in step_op_df.index[exceeded]]
            print(
                f"[WARN] L2 sum exceeds L1 total for {step_op} at x={','.join(xs)}. "
                "Check NVTX nesting or selected L2 call list."
            )


L1_DONUT_EXCLUDE = {"step1.lookup", "step1.allocate", "step1.input", "step3.input"}
L1_DONUT_ORDER = [
    "step1.offload_launch",
    "step1.offload_wait",
    "step3.lookup",
    "step3.allocate",
    "step3.onboard_launch",
    "step3.onboard_wait",
]
DONUT_TITLE = "L1/L2_latency_breakdown percentage"
OUTER_RING_R = 1.0
OUTER_RING_WIDTH = 0.36
INNER_RING_R = 0.62
INNER_RING_WIDTH = 0.34
WEDGE_EDGE_LW_OUTER = 1.2
WEDGE_EDGE_LW_INNER = 1.0


def _polar_point(theta_deg: float, radius: float) -> Tuple[float, float]:
    theta = math.radians(theta_deg)
    return radius * math.cos(theta), radius * math.sin(theta)


def l2_donut_exclude_columns() -> set:
    return {
        col
        for col, step_op, _ in L2_CALL_TIMELINE
        if step_op in ("step1.lookup", "step1.allocate")
        or (step_op is not None and step_op.startswith("step2."))
        or col.startswith("step2.")
    }


def _is_step2_label(label: str) -> bool:
    return bool(re.match(r"step2[.\s]", label))


def op_legend_label(name: str) -> str:
    return re.sub(r"^step\d+\.", "", name)


def l2_callout_label(name: str) -> str:
    label = re.sub(r"^step\d+\.", "", name)
    return label.replace("_py", "")


def pct_label(value: float) -> str:
    return f"{value:.2f}%" if value < 0.1 else f"{value:.1f}%"


def build_nested_donut_groups(
    l1_row: pd.Series,
    l2_row: pd.Series,
) -> Tuple[List[Dict], float]:
    l2_skip = l2_donut_exclude_columns()
    groups: List[Dict] = []
    for l1_key in L1_DONUT_ORDER:
        if l1_key in L1_DONUT_EXCLUDE or _is_step2_label(l1_key):
            continue
        if l1_key not in l1_row or float(l1_row[l1_key]) <= 1e-9:
            continue
        children: List[Tuple[str, float]] = []
        for col, step_op, _ in L2_CALL_TIMELINE:
            if col in l2_skip or col not in l2_row:
                continue
            if _is_step2_label(col) or (
                step_op is not None and _is_step2_label(step_op)
            ):
                continue
            parent = step_op if step_op is not None else col
            if parent != l1_key:
                continue
            val = float(l2_row[col])
            if val > 1e-9:
                children.append((col, val))
        groups.append(
            {
                "l1_key": l1_key,
                "l1_val": float(l1_row[l1_key]),
                "children": children,
            }
        )
    total = sum(g["l1_val"] for g in groups)
    if total <= 0:
        raise ValueError("Donut breakdown has no data after exclusions.")
    return groups, total


def plot_nested_donut_breakdown(
    l1_row: pd.Series,
    l2_row: pd.Series,
    *,
    batch_size: int,
    seq_len: int,
    output_png: str,
) -> None:
    groups, total = build_nested_donut_groups(l1_row, l2_row)
    ensure_dir(os.path.dirname(output_png))

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    cmap = plt.get_cmap("tab10")
    legend_handles: List[Patch] = []
    legend_labels: List[str] = []
    outer_callouts: List[Dict] = []

    cursor = 90.0

    for gi, group in enumerate(groups):
        color = cmap(gi % 10)
        l1_key = group["l1_key"]
        l1_val = group["l1_val"]
        span = (l1_val / total) * 360.0
        l1_pct = (l1_val / total) * 100.0

        theta2 = cursor
        theta1 = cursor - span
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                INNER_RING_R,
                theta1,
                theta2,
                width=INNER_RING_WIDTH,
                facecolor=color,
                edgecolor="white",
                linewidth=WEDGE_EDGE_LW_INNER,
                zorder=2,
            )
        )

        legend_handles.append(
            Patch(facecolor=color, edgecolor="white", label=op_legend_label(l1_key))
        )
        legend_labels.append(op_legend_label(l1_key))

        mid = (theta1 + theta2) / 2.0
        if l1_pct >= 0.4 and span >= 3.0:
            text_xy = _polar_point(mid, INNER_RING_R - INNER_RING_WIDTH / 2.0)
            ax.text(
                text_xy[0],
                text_xy[1],
                f"{l1_pct:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    fc="white",
                    ec="none",
                    alpha=0.65,
                ),
                zorder=7,
            )

        children = group["children"]
        if children:
            child_total = sum(v for _, v in children)
            outer_cursor = theta2
            for child_name, child_val in children:
                child_span = (
                    (child_val / child_total) * span if child_total > 0 else 0.0
                )
                c_theta2 = outer_cursor
                c_theta1 = outer_cursor - child_span
                child_pct = (child_val / total) * 100.0
                ax.add_patch(
                    Wedge(
                        (0.0, 0.0),
                        OUTER_RING_R,
                        c_theta1,
                        c_theta2,
                        width=OUTER_RING_WIDTH,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=WEDGE_EDGE_LW_OUTER,
                        alpha=0.92,
                        zorder=3,
                    )
                )
                c_mid = (c_theta1 + c_theta2) / 2.0
                anchor = _polar_point(c_mid, OUTER_RING_R - OUTER_RING_WIDTH / 2.0)
                label_base = _polar_point(c_mid, OUTER_RING_R + 0.30)
                label_x = 1.18 if label_base[0] >= 0 else -1.08
                label_y = label_base[1]
                if (
                    child_name.startswith("step1.")
                    and group["l1_key"] == "step1.offload_launch"
                ):
                    label_x = 0.88
                    label_y *= 0.76
                elif child_name == "step1.flexkv.client.try_wait":
                    label_x = 1.18
                    label_y -= 0.02
                elif child_name == "step1.gpu.release_offload_pages_py":
                    label_x = -0.36
                    label_y = -1.12
                elif child_name == "step1.flexkv.finish_task":
                    label_x = 0.08
                    label_y = -1.16
                outer_callouts.append(
                    {
                        "anchor": anchor,
                        "x": label_x,
                        "y": label_y,
                        "label": f"{l2_callout_label(child_name)} {pct_label(child_pct)}",
                        "color": color,
                        "side": 1 if label_base[0] >= 0 else -1,
                    }
                )
                outer_cursor = c_theta1

        cursor = theta1

    min_gap = 0.075
    for side in (-1, 1):
        side_callouts = sorted(
            [c for c in outer_callouts if c["side"] == side],
            key=lambda c: c["y"],
        )
        prev_y = None
        for callout in side_callouts:
            if prev_y is not None and callout["y"] - prev_y < min_gap:
                callout["y"] = prev_y + min_gap
            prev_y = callout["y"]

    for callout in outer_callouts:
        ax.annotate(
            callout["label"],
            xy=callout["anchor"],
            xytext=(callout["x"], callout["y"]),
            ha="left" if callout["side"] > 0 else "right",
            va="center",
            fontsize=5.6,
            arrowprops=dict(arrowstyle="-", color=callout["color"], lw=0.7),
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.78),
            zorder=7,
        )

    plot_radius = OUTER_RING_R + 0.35
    ax.set(aspect="equal")
    ax.set_xlim(-plot_radius, plot_radius)
    ax.set_ylim(-plot_radius, plot_radius)
    ax.axis("off")
    ax.set_title(
        f"{DONUT_TITLE}\nbs={batch_size}, seq_len={seq_len}",
        fontsize=11,
        pad=4,
    )
    ax.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        borderaxespad=0.2,
        handletextpad=0.35,
        labelspacing=0.3,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.0, right=0.82, top=0.90, bottom=0.0)
    fig.savefig(output_png, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved donut plot: {output_png}")


def save_donut_views(
    l1_df: pd.DataFrame,
    l2_df: pd.DataFrame,
    plot_dir: str,
    batch_size: int,
) -> None:
    for seq_len in l1_df.index:
        if seq_len not in l2_df.index:
            continue
        png_path = os.path.join(
            plot_dir,
            f"L1L2_latency_breakdown_latency_bs{batch_size}_len{seq_len}.png",
        )
        plot_nested_donut_breakdown(
            l1_df.loc[seq_len],
            l2_df.loc[seq_len],
            batch_size=batch_size,
            seq_len=int(seq_len),
            output_png=png_path,
        )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_view(
    df: pd.DataFrame,
    png_path: str,
    csv_path: str,
    x_label: str,
    *,
    stack_order: str = "timeline",
) -> None:
    ensure_dir(os.path.dirname(png_path))
    ensure_dir(os.path.dirname(csv_path))
    df.to_csv(csv_path)
    title = os.path.splitext(os.path.basename(png_path))[0]
    if title.startswith("L1_breakdown_"):
        round_note = (
            "step1: offload round (100% GPU miss) | "
            "step3: onboard round (100% GPU hit)"
        )
        title = f"{title}\n{round_note}"
    elif title.startswith("L2_breakdown_"):
        round_note = (
            "step1: offload round (100% GPU miss) | "
            "step2: evict | "
            "step3: onboard round (100% GPU hit)"
        )
        title = f"{title}\n{round_note}"
    plot_stacked_breakdown(
        df=df,
        title=title,
        x_label=x_label,
        output_png=png_path,
        stack_order=stack_order,
    )
    print(f"Saved plot: {png_path}")
    print(f"Saved table: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate L1 (step-op) and L2 (function timeline) plots from NVTX CSV."
    )
    parser.add_argument("--csv-root", required=True)
    parser.add_argument("--mode-dir", default="flexkv_profile_fine")
    parser.add_argument("--seq-lens", default="1024,2048,4096")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    seq_lens = parse_int_list(args.seq_lens, "--seq-lens")
    case_pattern = f"len{{value}}_bs{args.batch_size}"
    plot_dir = args.output_root
    csv_dir = os.path.join(args.output_root, "csv_summarization")
    x_label = "Sequence Length"
    bs = args.batch_size
    step_prefixes = default_prefixes_for_step_flow()

    # L1: step-op breakdown
    step_op_df = load_mode_breakdown(
        csv_root=args.csv_root,
        mode_dir=args.mode_dir,
        x_values=seq_lens,
        case_pattern=case_pattern,
        include_prefixes=step_prefixes,
        group_by_step_op=True,
    )
    l1_step_op_df = reorder_step_op_columns(drop_step2_step_ops(step_op_df))
    save_view(
        l1_step_op_df,
        os.path.join(plot_dir, f"L1_breakdown_bs{bs}.png"),
        os.path.join(csv_dir, f"L1_breakdown_bs{bs}.csv"),
        x_label,
    )

    # L2: function-level timeline
    detail_step_df = load_mode_breakdown(
        csv_root=args.csv_root,
        mode_dir=args.mode_dir,
        x_values=seq_lens,
        case_pattern=case_pattern,
        include_prefixes=step_prefixes + ["recsys.", "step1.input", "step3.input"],
        group_by_step_op=False,
    )
    l2_df = build_l2_call_timeline_df(detail_step_df, step_op_df.index)
    warn_l2_exceeds_l1(step_op_df, l2_df)
    l2_df = reorder_l2_columns(l2_df)
    save_view(
        l2_df,
        os.path.join(plot_dir, f"L2_breakdown_bs{bs}.png"),
        os.path.join(csv_dir, f"L2_breakdown_bs{bs}.csv"),
        x_label,
        stack_order="input",
    )

    save_donut_views(
        reorder_step_op_columns(step_op_df),
        l2_df,
        plot_dir,
        bs,
    )

    print("[DONE] L1/L2 bar plots and nested donut plots saved.")


if __name__ == "__main__":
    main()
