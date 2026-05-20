import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


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

    # Compatible with multiple nsys report names:
    # nvtxsum / nvtxppsum / nvtx_pushpop_sum / other nvtx*.csv variants.
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


def default_prefixes_for_mode(mode: str) -> List[str]:
    if mode == "pipeline_coarse":
        return ["pipeline."]
    if mode == "flexkv_coarse":
        return ["flexkv."]
    if mode == "flexkv_fine":
        return ["flexkv.adapter.", "flexkv.client.", "flexkv._"]
    return [""]


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
    group_by_step: bool = False,
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
        # nsys CSV often prefixes range names with ":" (e.g. ":step1.lookup").
        # Normalize once so prefix filtering/grouping works for both forms.
        df["label_norm"] = (
            df[label_col].astype(str).str.strip().str.lstrip(":")
        )
        if include_prefixes:
            mask = df["label_norm"].apply(lambda x: matches_any_prefix(x, include_prefixes))
            df = df[mask]
        if group_by_step:
            # For step-flow wall-clock view, keep only top-level step labels.
            # Exclude nested internal labels, e.g. step1.lookup::flexkv.lookup_kvcache.
            df = df[~df["label_norm"].str.contains("::", regex=False)]
            df["group_label"] = (
                df["label_norm"]
                .apply(
                    lambda s: re.match(r"(step\d+)\.", s).group(1)
                    if re.match(r"(step\d+)\.", s)
                    else s
                )
            )
        elif group_by_step_op:
            # Keep only top-level step operations, e.g. step1.lookup.
            # Exclude nested internal labels, e.g. step1.lookup::flexkv.lookup_kvcache.
            df = df[~df["label_norm"].str.contains("::", regex=False)]
            df["group_label"] = (
                df["label_norm"]
                .apply(
                    lambda s: re.match(r"(step\d+\.[^.:\s]+)", s).group(1)
                    if re.match(r"(step\d+\.[^.:\s]+)", s)
                    else s
                )
            )
        else:
            df["group_label"] = df["label_norm"]
        grouped = df.groupby("group_label")["time_ms"].sum().to_dict()
        records[x] = grouped

    breakdown_df = pd.DataFrame.from_dict(records, orient="index").fillna(0.0)
    breakdown_df.index.name = "x_value"
    return breakdown_df.sort_index()


def aggregate_step_op_avg7(df: pd.DataFrame) -> pd.DataFrame:
    def col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(0.0, index=df.index)

    out = pd.DataFrame(index=df.index)
    out["lookup_avg(step1,step3)"] = (col("step1.lookup") + col("step3.lookup")) / 2.0
    out["allocate_avg(step1,step3)"] = (col("step1.allocate") + col("step3.allocate")) / 2.0
    out["step1.offload_launch"] = col("step1.offload_launch")
    out["step1.offload_wait"] = col("step1.offload_wait")
    out["step2.evict_gpu"] = col("step2.evict_gpu")
    out["step3.onboard_launch"] = col("step3.onboard_launch")
    out["step3.onboard_wait"] = col("step3.onboard_wait")
    out.index.name = df.index.name
    return out


def step_op_group_key(column: str) -> str:
    """Group stacked segments by top-level step op (e.g. step1.lookup)."""
    normalized = column.lstrip(":")
    match = re.match(r"(step\d+\.[^.:\s]+)", normalized)
    if match:
        return match.group(1)
    if "::" in normalized:
        return normalized.split("::", 1)[0]
    return normalized


def column_short_top_label(column: str) -> str:
    """Legend / short name: last :: segment (e.g. gpu.lookup_cpp)."""
    parts = column.split("::")
    return parts[-1] if parts else column


def build_group_top_legend_map(
    ordered_components: List[str],
) -> Tuple[Dict[str, str], List[str]]:
    """
    For each step-op group, only the topmost stack segment gets a legend entry.
    Returns col->short label for top segments, and legend labels in stack order.
    """
    group_top_col: Dict[str, str] = {}
    for col in ordered_components:
        group_top_col[step_op_group_key(col)] = col

    col_to_label: Dict[str, str] = {}
    legend_labels: List[str] = []
    seen: set[str] = set()
    for col in ordered_components:
        group_key = step_op_group_key(col)
        if group_top_col.get(group_key) != col:
            continue
        short = column_short_top_label(col)
        col_to_label[col] = short
        if short not in seen:
            legend_labels.append(short)
            seen.add(short)
    return col_to_label, legend_labels


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
    # External labels use offset points so they stay beside their bar, not on the next one.
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


def _timeline_sort_key(label: str) -> Tuple[int, int, int, str, str]:
    s = label.lstrip(":")
    top, sep, suffix = s.partition("::")

    init_order = {
        "init.create_kvcache_manager": 0,
        "init.prepare_inputs": 1,
        "init.shutdown": 2,
    }
    if top.startswith("init."):
        return (0, init_order.get(top, 99), 0, top, suffix)

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
        # top-level op first, then nested internal labels.
        nested_rank = 1 if sep else 0
        return (1, step_num, op_order.get(op, 99), op, f"{nested_rank}:{suffix}")

    # Legacy/non-step labels still get timeline-biased placement where possible.
    if "to_get_match_requests" in s or "flexkv.client.get_match" in s:
        return (1, 1, 1, "lookup", s)
    if "build_slot_mappings" in s:
        return (1, 1, 2, "allocate", s)
    if "put_async" in s:
        return (1, 1, 3, "offload_launch", s)
    if "gpu.put_py" in s or "gpu.put" in s:
        return (1, 1, 2, "allocate", s)
    if "append_kvcache" in s:
        return (1, 1, 2, "allocate", s)
    if "try_wait" in s or "offload_wait" in s:
        return (1, 1, 4, "offload_wait", s)
    if "evict" in s:
        return (1, 2, 5, "evict_gpu", s)
    if "onboard_launch" in s or "launch_cpp" in s:
        return (1, 3, 6, "onboard_launch", s)
    if "onboard_wait" in s:
        return (1, 3, 7, "onboard_wait", s)
    if "residual" in s:
        return (1, 9, 99, "residual", s)

    # Keep the rest deterministic and after init/step labels.
    return (99, 99, 99, top, suffix)


def plot_stacked_breakdown(
    df: pd.DataFrame,
    title: str,
    x_label: str,
    output_png: str,
    topk: Optional[int] = None,
    stack_order: str = "timeline",
    legend_mode: str = "full",
) -> None:
    if df.empty:
        raise ValueError("Breakdown dataframe is empty, nothing to plot.")

    if stack_order == "input":
        ordered_components = list(df.columns)
    elif stack_order == "size":
        ordered_components = list(df.sum(axis=0).sort_values(ascending=False).index)
    elif stack_order == "name":
        ordered_components = sorted(df.columns)
    else:
        ordered_components = sorted(df.columns, key=_timeline_sort_key)

    if topk is None or topk <= 0 or topk >= len(ordered_components):
        keep_components = ordered_components
        other_components = []
    else:
        keep_components = ordered_components[:topk]
        other_components = [c for c in ordered_components if c not in keep_components]

    plot_df = df[keep_components].copy()
    if other_components:
        plot_df["others"] = df[other_components].sum(axis=1)

    x_values = list(plot_df.index)
    x_positions = list(range(len(x_values)))
    bottoms = [0.0 for _ in x_values]
    heights_by_component: List[List[float]] = []
    bar_width = 0.65

    components = list(plot_df.columns)
    if legend_mode == "group_top_short":
        col_legend_map, _ = build_group_top_legend_map(components)
    else:
        col_legend_map = {c: c for c in components}

    fig_w = 14 if legend_mode != "group_top_short" else 15
    fig, ax = plt.subplots(figsize=(fig_w, 8))
    cmap = plt.get_cmap("tab20")
    component_colors: List = []
    for i, comp in enumerate(components):
        vals = plot_df[comp].tolist()
        heights_by_component.append(vals)
        bar_color = cmap(i % 20)
        component_colors.append(bar_color)
        legend_label = col_legend_map.get(comp)
        ax.bar(
            x_positions,
            vals,
            bottom=bottoms,
            label=legend_label if legend_label else "_nolegend_",
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

    if legend_mode == "group_top_short":
        handles, labels = [], []
        for container, comp in zip(ax.containers, components):
            if comp not in col_legend_map:
                continue
            short = col_legend_map[comp]
            if short in labels:
                continue
            handles.append(container)
            labels.append(short)
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(right=0.80)
    else:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(right=0.72)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stacked breakdown across sequence lengths."
    )
    parser.add_argument("--csv-root", required=True, help="CSV root directory")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pipeline_coarse", "flexkv_coarse", "flexkv_fine"],
        help="Profiling mode",
    )
    parser.add_argument(
        "--mode-dir",
        default="",
        help="Optional subdirectory name under csv-root (default: same as --mode)",
    )
    parser.add_argument("--seq-lens", default="256,512,1024,2048")
    parser.add_argument(
        "--batch-sizes",
        default="",
        help="Comma-separated batch sizes, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--x-kind",
        choices=["seq_len", "batch_size"],
        default="seq_len",
        help="X-axis dimension and case selection dimension",
    )
    parser.add_argument(
        "--case-pattern",
        default="",
        help=(
            "Case tag pattern used to match CSV filename (supports {value}), "
            "e.g. seq{value}, bs{value}, len1024_bs{value}"
        ),
    )
    parser.add_argument(
        "--view",
        choices=["mode_breakdown", "step_flow", "step_op_flow", "step_op_avg7"],
        default="mode_breakdown",
        help=(
            "mode_breakdown: plot selected NVTX labels directly; "
            "step_flow: aggregate labels into step1/step2/step3; "
            "step_op_flow: aggregate into top-level step ops (step1.lookup etc.); "
            "step_op_avg7: average step1/step3 lookup+allocate into 7 ops"
        ),
    )
    parser.add_argument("--topk", type=int, default=8, help="Top-K ranges to keep")
    parser.add_argument(
        "--stack-order",
        choices=["timeline", "input", "size", "name"],
        default="timeline",
        help=(
            "Stacking order for components. "
            "timeline keeps operation order, input keeps dataframe order, size sorts by total time."
        ),
    )
    parser.add_argument(
        "--include-prefixes",
        default="",
        help="Comma-separated NVTX range prefixes to include; empty uses mode defaults",
    )
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    if args.x_kind == "seq_len":
        x_values = parse_int_list(args.seq_lens, "--seq-lens")
        default_case_pattern = "seq{value}"
        x_label = "Sequence Length"
    else:
        x_values = parse_int_list(args.batch_sizes, "--batch-sizes")
        default_case_pattern = "bs{value}"
        x_label = "Batch Size"

    mode_dir = args.mode_dir.strip() if args.mode_dir.strip() else args.mode
    case_pattern = args.case_pattern.strip() if args.case_pattern.strip() else default_case_pattern
    if "{value}" not in case_pattern:
        raise ValueError("--case-pattern must contain '{value}' placeholder")

    if args.include_prefixes.strip():
        prefixes = [p.strip() for p in args.include_prefixes.split(",") if p.strip()]
    elif args.view in ("step_flow", "step_op_flow", "step_op_avg7"):
        prefixes = default_prefixes_for_step_flow()
    else:
        prefixes = default_prefixes_for_mode(args.mode)

    df = load_mode_breakdown(
        csv_root=args.csv_root,
        mode_dir=mode_dir,
        x_values=x_values,
        case_pattern=case_pattern,
        include_prefixes=prefixes,
        group_by_step=(args.view == "step_flow"),
        group_by_step_op=(args.view in ("step_op_flow", "step_op_avg7")),
    )
    if args.view == "step_op_avg7":
        df = aggregate_step_op_avg7(df)
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df.to_csv(args.output_csv)

    if args.view == "step_flow":
        title = f"{args.mode} Step Flow Breakdown"
    elif args.view == "step_op_flow":
        title = f"{args.mode} Step-Op Breakdown"
    elif args.view == "step_op_avg7":
        title = f"{args.mode} Step-Op Avg7 Breakdown"
    else:
        title = f"{args.mode} Time Breakdown"

    effective_topk = args.topk
    if args.view == "step_op_flow":
        effective_topk = max(args.topk, 9)
    if args.view == "step_op_avg7":
        effective_topk = max(args.topk, 7)
    plot_stacked_breakdown(
        df=df,
        title=title,
        x_label=x_label,
        output_png=args.output_png,
        topk=effective_topk,
        stack_order=args.stack_order,
    )
    print(f"Saved plot: {args.output_png}")
    if args.output_csv:
        print(f"Saved table: {args.output_csv}")


if __name__ == "__main__":
    main()
