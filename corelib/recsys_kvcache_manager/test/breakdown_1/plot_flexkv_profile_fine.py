#!/usr/bin/env python3
"""Plot L1/L2/L3 (+ init) from nsys NVTX csv.

L4/L5 (raw cpu/gpu py and cpp/kernel stacks) are omitted: they largely duplicate L3.
Py/cpp per-path drill-down is available in the raw nsys CSV if needed.
"""

import argparse
import os

import pandas as pd

from plot_nsys_breakdown import (
    _timeline_sort_key,
    default_prefixes_for_step_flow,
    load_mode_breakdown,
    parse_int_list,
    plot_stacked_breakdown,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sum_columns(df: pd.DataFrame, pred) -> pd.Series:
    cols = [c for c in df.columns if pred(c)]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].sum(axis=1)


def sum_metric(
    detail_df: pd.DataFrame,
    step_op: str | None,
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
        if col == bare or col.endswith(f"::{bare}"):
            return step_op.startswith("step1.") or step_op.startswith("step3.")
        return False

    return sum_columns(detail_df, include_column)


def drop_zero_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [c for c in df.columns if float(df[c].sum()) > 1e-9]
    return df[keep].copy()


L3_CALL_TIMELINE: list[tuple[str, str | None, str]] = [
    ("step1.input", None, "step1.input"),
    ("step1.gpu.lookup_py", "step1.lookup", "::gpu.lookup_py"),
    ("step1.flexkv.build_index_meta", "step1.lookup", "::flexkv.build_index_meta"),
    (
        "step1.flexkv.adapter.to_get_match_requests",
        "step1.lookup",
        "::flexkv.adapter.to_get_match_requests",
    ),
    ("step1.flexkv.client.get_match", "step1.lookup", "::flexkv.client.get_match"),
    ("step1.recsys.merge_lookup_results", "step1.lookup", "::recsys.merge_lookup_results"),
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
    ("step1.flexkv.client.put_async", "step1.offload_launch", "::flexkv.client.put_async"),
    ("step1.flexkv.client.launch", "step1.offload_launch", "::flexkv.client.launch"),
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
    ("step3.recsys.merge_lookup_results", "step3.lookup", "::recsys.merge_lookup_results"),
    ("step3.gpu.allocate_py", "step3.allocate", "::gpu.allocate_py"),
    (
        "step3.flexkv._build_slot_mappings",
        "step3.onboard_launch",
        "::flexkv._build_slot_mappings",
    ),
    ("step3.flexkv.client.put_async", "step3.onboard_launch", "::flexkv.client.put_async"),
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


def build_l3_call_timeline_df(detail_step_df: pd.DataFrame, index) -> pd.DataFrame:
    data = {}
    for col, step_op, token in L3_CALL_TIMELINE:
        bare = token.lstrip(":")
        leaf_only = bare in LEAF_TOKENS
        data[col] = sum_metric(detail_step_df, step_op, token, leaf_only=leaf_only)
    return drop_zero_columns(pd.DataFrame(data, index=index))


def reorder_l3_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    preferred = [name for name, _, _ in L3_CALL_TIMELINE if name in df.columns]
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


def reorder_columns_timeline(df):
    if df.empty:
        return df
    ordered = sorted(df.columns, key=_timeline_sort_key)
    return df[ordered].copy()


def reorder_step_op_columns(df):
    if df.empty:
        return df
    front = [c for c in STEP_OP_TIMELINE if c in df.columns]
    tail = [c for c in df.columns if c not in front]
    return df[front + tail].copy()


def save_view(
    df,
    png_path: str,
    csv_path: str,
    x_label: str,
    stack_order: str = "timeline",
    legend_mode: str = "full",
) -> None:
    ensure_dir(os.path.dirname(png_path))
    ensure_dir(os.path.dirname(csv_path))
    df.to_csv(csv_path)
    title = os.path.splitext(os.path.basename(png_path))[0]
    round_note = (
        "step1: offload round (100% GPU miss) | "
        "step2: evict | "
        "step3: onboard round (100% GPU hit)"
    )
    if title.startswith("step_flow_") or title.startswith("step_op_flow_") or title.startswith(
        "L3_breakdown_"
    ):
        title = f"{title}\n{round_note}"
    plot_stacked_breakdown(
        df=df,
        title=title,
        x_label=x_label,
        output_png=png_path,
        stack_order=stack_order,
        legend_mode=legend_mode,
    )
    print(f"Saved plot: {png_path}")
    print(f"Saved table: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate plot set for FlexKV fine profiling "
            "(step flow + step op + L3 + init)."
        )
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
    plot_dir = os.path.join(args.output_root, "plot")
    csv_dir = os.path.join(args.output_root, "csv_summarization")
    x_label = "Sequence Length"
    bs = args.batch_size

    # L1
    step_flow_df = load_mode_breakdown(
        csv_root=args.csv_root,
        mode_dir=args.mode_dir,
        x_values=seq_lens,
        case_pattern=case_pattern,
        include_prefixes=default_prefixes_for_step_flow(),
        group_by_step=True,
        group_by_step_op=False,
    )
    save_view(
        reorder_columns_timeline(step_flow_df),
        os.path.join(plot_dir, f"step_flow_bs{bs}.png"),
        os.path.join(csv_dir, f"step_flow_bs{bs}.csv"),
        x_label,
    )

    # L2
    step_op_df = load_mode_breakdown(
        csv_root=args.csv_root,
        mode_dir=args.mode_dir,
        x_values=seq_lens,
        case_pattern=case_pattern,
        include_prefixes=default_prefixes_for_step_flow(),
        group_by_step=False,
        group_by_step_op=True,
    )
    save_view(
        reorder_step_op_columns(step_op_df),
        os.path.join(plot_dir, f"step_op_flow_bs{bs}.png"),
        os.path.join(csv_dir, f"step_op_flow_bs{bs}.csv"),
        x_label,
    )

    # L3 (function timeline)
    detail_step_df = load_mode_breakdown(
        csv_root=args.csv_root,
        mode_dir=args.mode_dir,
        x_values=seq_lens,
        case_pattern=case_pattern,
        include_prefixes=default_prefixes_for_step_flow()
        + ["recsys.", "step1.input", "step3.input"],
        group_by_step=False,
        group_by_step_op=False,
    )
    l3_df = build_l3_call_timeline_df(detail_step_df, step_op_df.index)
    save_view(
        reorder_l3_columns(l3_df),
        os.path.join(plot_dir, f"L3_breakdown_bs{bs}.png"),
        os.path.join(csv_dir, f"L3_breakdown_bs{bs}.csv"),
        x_label,
        stack_order="input",
    )

    # init (separate scale)
    init_df = load_mode_breakdown(
        csv_root=args.csv_root,
        mode_dir=args.mode_dir,
        x_values=seq_lens,
        case_pattern=case_pattern,
        include_prefixes=["init."],
        group_by_step=False,
        group_by_step_op=False,
    )
    save_view(
        reorder_columns_timeline(init_df),
        os.path.join(plot_dir, f"init_breakdown_bs{bs}.png"),
        os.path.join(csv_dir, f"init_breakdown_bs{bs}.csv"),
        x_label,
    )

    print("[DONE] L1/L2/L3/init plots (no L4/L5 — see L3_breakdown for function timeline).")


if __name__ == "__main__":
    main()
