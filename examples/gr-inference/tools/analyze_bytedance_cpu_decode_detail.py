"""Deep-dive CPU overhead and decode non-attention breakdown for Bytedance ecom."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from tool_utils import bootstrap_repo_paths, write_json

bootstrap_repo_paths(__file__, include_tools=True)
import analyze_nsys_gr_sglang as nsys  # noqa: E402

NS_PER_MS = 1_000_000.0


def analyze_trace(path: Path, *, framework: str) -> dict[str, Any]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        tables = _tables(conn)
        strings = _strings(conn)
        kernels = _rows(
            conn,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            strings,
            ("demangledName", "shortName", "mangledName", "name"),
            tables=tables,
        )
        runtime = _rows(
            conn,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            strings,
            ("nameId", "name"),
            tables=tables,
        )
        nvtx = _rows(
            conn,
            "NVTX_EVENTS",
            strings,
            ("text", "name", "message", "textId", "nameId", "messageId"),
            tables=tables,
        )
        memcpy = _rows(
            conn,
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            strings,
            ("copyKind",),
            tables=tables,
        )
        memset = _rows(
            conn,
            "CUPTI_ACTIVITY_KIND_MEMSET",
            strings,
            ("memKind",),
            tables=tables,
        )

    base_windows = nsys._stage_windows(nvtx, kernels)
    windows = (
        _sglang_corrected_windows(nvtx, kernels, base_windows)
        if framework == "sglang"
        else base_windows
    )
    stages = {
        stage: _stage_summary(
            stage_windows,
            kernels=kernels,
            runtime=runtime,
            memcpy=memcpy,
            memset=memset,
        )
        for stage, stage_windows in windows.items()
        if stage in {"overall", "prefill", "decode"}
    }
    decode_windows = windows.get("decode") or []
    decode_kernel_categories = _kernel_categories(kernels, decode_windows)
    decode_top_non_attention = _top_kernel_names(
        kernels,
        decode_windows,
        exclude_category="attention",
        limit=16,
    )
    result = {
        "path": str(path),
        "framework": framework,
        "stage_windows": _serialize_windows(windows),
        "stage_summaries": stages,
        "decode_kernel_categories_ms": decode_kernel_categories,
        "decode_top_non_attention_kernels": decode_top_non_attention,
        "top_runtime_calls_overall": _top_runtime_calls(
            runtime, windows.get("overall") or []
        ),
        "nvtx_ranges": _top_nvtx(nvtx, windows.get("overall") or []),
    }
    if framework == "sglang":
        result["original_nvtx_stage_windows"] = _serialize_windows(base_windows)
        result["sglang_decode_async_tail"] = _sglang_decode_tail_summary(
            base_windows,
            windows,
            kernels=kernels,
        )
    return result


def build_report(
    *,
    gr_fixed: Path,
    gr_dynamic: Path,
    sglang: Path,
) -> dict[str, Any]:
    traces = {
        "gr_fixed_256": analyze_trace(gr_fixed, framework="gr"),
        "gr_dynamic_64_128_256": analyze_trace(gr_dynamic, framework="gr"),
        "sglang_fixed_256": analyze_trace(sglang, framework="sglang"),
    }
    return {
        "traces": traces,
        "cpu_overhead_rows": _cpu_overhead_rows(traces),
        "decode_category_rows": _decode_category_rows(traces),
        "decode_top_diff_gr_fixed_vs_sglang_rows": _decode_top_diff_rows(
            traces["gr_fixed_256"],
            traces["sglang_fixed_256"],
        ),
        "notes": [
            "SGLang stage NVTX ranges wrap host forward launch. CUDA graph kernels can continue after the NVTX range ends, so the corrected SGLang decode window runs from the first sglang.decode NVTX start to the active CUDA window end.",
            "GPU idle/host gap uses stage window minus the union of kernel, memcpy, and memset activity. Runtime gap is separately computed from gaps between CUDA runtime API calls over 50us.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Bytedance ecom CPU / Decode Deep Dive",
        "",
        "This report corrects SGLang decode attribution for asynchronous CUDA graph tails and separates GPU activity from host/runtime gaps.",
        "",
        "## CPU Overhead Evidence",
        "",
        *_cpu_table_lines(report["cpu_overhead_rows"]),
        "",
        "## Decode Non-Attention Kernel Categories",
        "",
        *_decode_category_lines(report["decode_category_rows"]),
        "",
        "## GR Fixed vs Corrected SGLang Decode Non-Attention Diff",
        "",
        *_diff_lines(report["decode_top_diff_gr_fixed_vs_sglang_rows"]),
        "",
        "## SGLang Decode Async Tail",
        "",
    ]
    tail = report["traces"]["sglang_fixed_256"].get("sglang_decode_async_tail") or {}
    lines.extend(
        [
            f"- Original NVTX decode window: {_fmt(tail.get('original_decode_window_ms'))} ms",
            f"- Corrected decode window: {_fmt(tail.get('corrected_decode_window_ms'))} ms",
            f"- Kernel time after the original decode NVTX window: {_fmt(tail.get('tail_kernel_sum_ms'))} ms",
            f"- Tail attention kernels: {_fmt(tail.get('tail_attention_kernel_ms'))} ms",
            f"- Tail non-attention kernels: {_fmt(tail.get('tail_non_attention_kernel_ms'))} ms",
            "",
            "Top tail kernels:",
        ]
    )
    for row in tail.get("tail_top_kernels", [])[:10]:
        lines.append(
            f"- `{row['name']}`: {_fmt(row['total_ms'])} ms, count={row['count']}"
        )
    lines.extend(["", "## Notes", ""])
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _strings(conn: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in _tables(conn):
        return {}
    return {
        int(row["id"]): str(row["value"])
        for row in conn.execute("SELECT id, value FROM StringIds")
    }


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _rows(
    conn: sqlite3.Connection,
    table: str,
    strings: dict[int, str],
    name_columns: tuple[str, ...],
    *,
    tables: set[str],
) -> list[dict[str, Any]]:
    if table not in tables:
        return []
    columns = _columns(conn, table)
    rows = []
    for row in conn.execute(f"SELECT * FROM {table}"):
        start = row["start"] if "start" in columns else None
        end = row["end"] if "end" in columns else None
        if start is None or end is None or end <= start:
            continue
        rows.append(
            {
                "start": int(start),
                "end": int(end),
                "name": _resolve_name(row, columns, strings, name_columns),
            }
        )
    return rows


def _resolve_name(
    row: sqlite3.Row,
    columns: set[str],
    strings: dict[int, str],
    name_columns: tuple[str, ...],
) -> str:
    for column in name_columns:
        if column not in columns:
            continue
        value = row[column]
        if value is None:
            continue
        if isinstance(value, int):
            return strings.get(value, str(value))
        return str(value)
    return "<unknown>"


def _sglang_corrected_windows(
    nvtx: list[dict[str, Any]],
    kernels: list[dict[str, Any]],
    base_windows: dict[str, list[tuple[int, int]]],
) -> dict[str, list[tuple[int, int]]]:
    active_start, active_end = nsys._event_window(kernels)
    decode_starts = sorted(
        event["start"]
        for event in nvtx
        if event["name"] == "sglang.decode" and event.get("start") is not None
    )
    prefill_starts = sorted(
        event["start"]
        for event in nvtx
        if event["name"] == "sglang.prefill" and event.get("start") is not None
    )
    if active_start is None or active_end is None or not decode_starts:
        return base_windows
    first_decode = decode_starts[0]
    prefill_start = prefill_starts[0] if prefill_starts else active_start
    return {
        "overall": [(active_start, active_end)],
        "prefill": [(prefill_start, first_decode)],
        "decode": [(first_decode, active_end)],
    }


def _stage_summary(
    windows: list[tuple[int, int]],
    *,
    kernels: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
    memcpy: list[dict[str, Any]],
    memset: list[dict[str, Any]],
) -> dict[str, Any]:
    gpu_activity = kernels + memcpy + memset
    window_ms = _windows_ms(windows)
    gpu_union_ms = _union_ms(gpu_activity, windows)
    runtime_gap_ms, runtime_gap_count = _runtime_gap_ms(runtime, windows)
    categories = _kernel_categories(kernels, windows)
    attention_ms = categories.get("attention", 0.0)
    kernel_sum_ms = _sum_overlap_ms(kernels, windows)
    return {
        "window_ms": window_ms,
        "kernel_sum_ms": kernel_sum_ms,
        "kernel_union_ms": _union_ms(kernels, windows),
        "memcpy_sum_ms": _sum_overlap_ms(memcpy, windows),
        "memset_sum_ms": _sum_overlap_ms(memset, windows),
        "gpu_activity_union_ms": gpu_union_ms,
        "gpu_idle_or_host_gap_ms": max(0.0, window_ms - gpu_union_ms),
        "runtime_api_sum_ms": _sum_overlap_ms(runtime, windows),
        "runtime_api_count": _count_overlapping(runtime, windows),
        "runtime_gap_gt_50us_ms": runtime_gap_ms,
        "runtime_gap_gt_50us_count": runtime_gap_count,
        "cuda_graph_launches": _count_graph_launches(runtime, windows),
        "kernel_launches": _count_overlapping(kernels, windows),
        "attention_kernel_ms": attention_ms,
        "non_attention_kernel_ms": max(0.0, kernel_sum_ms - attention_ms),
        "kernel_categories_ms": categories,
    }


def _windows_ms(windows: list[tuple[int, int]]) -> float:
    return sum(end - start for start, end in nsys._merge_windows(windows)) / NS_PER_MS


def _event_overlap_ns(event: dict[str, Any], windows: list[tuple[int, int]]) -> int:
    total = 0
    for window_start, window_end in nsys._merge_windows(windows):
        overlap_start = max(event["start"], window_start)
        overlap_end = min(event["end"], window_end)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def _sum_overlap_ms(
    events: Iterable[dict[str, Any]], windows: list[tuple[int, int]]
) -> float:
    return sum(_event_overlap_ns(event, windows) for event in events) / NS_PER_MS


def _count_overlapping(
    events: Iterable[dict[str, Any]], windows: list[tuple[int, int]]
) -> int:
    return sum(1 for event in events if _event_overlap_ns(event, windows) > 0)


def _union_ms(
    events: Iterable[dict[str, Any]], windows: list[tuple[int, int]]
) -> float:
    intervals = []
    for event in events:
        for window_start, window_end in nsys._merge_windows(windows):
            overlap_start = max(event["start"], window_start)
            overlap_end = min(event["end"], window_end)
            if overlap_end > overlap_start:
                intervals.append((overlap_start, overlap_end))
    if not intervals:
        return 0.0
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged) / NS_PER_MS


def _runtime_gap_ms(
    runtime: list[dict[str, Any]],
    windows: list[tuple[int, int]],
    *,
    threshold_ns: int = 50_000,
) -> tuple[float, int]:
    clipped = []
    for event in runtime:
        for window_start, window_end in nsys._merge_windows(windows):
            start = max(event["start"], window_start)
            end = min(event["end"], window_end)
            if end > start:
                clipped.append((start, end))
    clipped.sort()
    previous_end = None
    total = 0
    count = 0
    for start, end in clipped:
        if previous_end is not None and start > previous_end:
            gap = start - previous_end
            if gap >= threshold_ns:
                total += gap
                count += 1
        previous_end = max(previous_end or end, end)
    return total / NS_PER_MS, count


def _count_graph_launches(
    runtime: list[dict[str, Any]],
    windows: list[tuple[int, int]],
) -> int:
    return sum(
        1
        for event in runtime
        if _event_overlap_ns(event, windows) > 0
        and "graph" in event["name"].lower()
        and "launch" in event["name"].lower()
    )


def _kernel_category(name: str) -> str:
    lower = name.lower()
    for category, needles in nsys.KERNEL_CATEGORIES:
        if any(needle in lower for needle in needles):
            return category
    return "other"


def _kernel_categories(
    kernels: list[dict[str, Any]],
    windows: list[tuple[int, int]],
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for kernel in kernels:
        duration_ms = _event_overlap_ns(kernel, windows) / NS_PER_MS
        if duration_ms <= 0.0:
            continue
        totals[_kernel_category(kernel["name"])] += duration_ms
    return dict(sorted(totals.items()))


def _normalized_kernel_name(name: str) -> str:
    checks = (
        ("direct_copy_kernel", "aten direct_copy elementwise"),
        ("mbtopk::computeBlockDigitCounts", "aten mbtopk computeBlockDigitCounts"),
        ("mbtopk::gatherTopK", "aten mbtopk gatherTopK"),
        ("mbtopk::radixFindKthValues", "aten mbtopk radixFindKthValues"),
        ("computeDigitCumSum", "aten mbtopk computeDigitCumSum"),
        ("computeBlockwiseWithinKCounts", "aten mbtopk blockwiseWithinKCounts"),
        ("cunn_SoftMaxForward", "aten LogSoftMaxForward"),
        ("silu_and_mul_packed_vec_kernel", "GR silu_and_mul_packed_vec"),
        ("act_and_mul_kernel", "SGLang act_and_mul_kernel"),
        ("FusedAddRMSNormKernel", "flashinfer fused_add_rmsnorm"),
        ("fused_qknorm", "fused_qknorm_warp"),
        ("fused_rope_kernel", "fused_rope_kernel"),
        ("store_kvcache", "store_kvcache"),
        ("vectorized_elementwise_kernel", "aten vectorized elementwise"),
        ("reduce_kernel", "aten reduce_kernel"),
        ("index_elementwise_kernel", "aten index_elementwise_kernel"),
    )
    for needle, label in checks:
        if needle in name:
            return label
    if "flash" in name.lower() and "attn" in name.lower():
        return "flash attention kernel"
    if name.startswith("nvjet_sm90"):
        return name
    return name[:120]


def _top_kernel_names(
    kernels: list[dict[str, Any]],
    windows: list[tuple[int, int]],
    *,
    exclude_category: str | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for kernel in kernels:
        if exclude_category and _kernel_category(kernel["name"]) == exclude_category:
            continue
        duration_ms = _event_overlap_ns(kernel, windows) / NS_PER_MS
        if duration_ms <= 0.0:
            continue
        name = _normalized_kernel_name(kernel["name"])
        totals[name] += duration_ms
        counts[name] += 1
    rows = [
        {"name": name, "total_ms": total, "count": counts[name]}
        for name, total in totals.items()
    ]
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)[:limit]


def _top_runtime_calls(
    runtime: list[dict[str, Any]],
    windows: list[tuple[int, int]],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for event in runtime:
        duration_ms = _event_overlap_ns(event, windows) / NS_PER_MS
        if duration_ms <= 0.0:
            continue
        totals[event["name"]] += duration_ms
        counts[event["name"]] += 1
    rows = [
        {"name": name, "total_ms": total, "count": counts[name]}
        for name, total in totals.items()
    ]
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)[:limit]


def _top_nvtx(
    nvtx: list[dict[str, Any]],
    windows: list[tuple[int, int]],
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for event in nvtx:
        duration_ms = _event_overlap_ns(event, windows) / NS_PER_MS
        if duration_ms <= 0.0:
            continue
        totals[event["name"]] += duration_ms
        counts[event["name"]] += 1
    rows = [
        {"name": name, "total_ms": total, "count": counts[name]}
        for name, total in totals.items()
    ]
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)[:limit]


def _sglang_decode_tail_summary(
    original_windows: dict[str, list[tuple[int, int]]],
    corrected_windows: dict[str, list[tuple[int, int]]],
    *,
    kernels: list[dict[str, Any]],
) -> dict[str, Any]:
    original_decode = nsys._merge_windows(original_windows.get("decode") or [])
    corrected_decode = nsys._merge_windows(corrected_windows.get("decode") or [])
    if not original_decode or not corrected_decode:
        return {}
    original_end = max(end for _, end in original_decode)
    corrected_end = max(end for _, end in corrected_decode)
    if corrected_end <= original_end:
        tail_windows: list[tuple[int, int]] = []
    else:
        tail_windows = [(original_end, corrected_end)]
    categories = _kernel_categories(kernels, tail_windows)
    kernel_sum = _sum_overlap_ms(kernels, tail_windows)
    attention = categories.get("attention", 0.0)
    return {
        "original_decode_window_ms": _windows_ms(original_decode),
        "corrected_decode_window_ms": _windows_ms(corrected_decode),
        "tail_window_ms": _windows_ms(tail_windows),
        "tail_kernel_sum_ms": kernel_sum,
        "tail_attention_kernel_ms": attention,
        "tail_non_attention_kernel_ms": max(0.0, kernel_sum - attention),
        "tail_kernel_categories_ms": categories,
        "tail_top_kernels": _top_kernel_names(kernels, tail_windows, limit=12),
    }


def _serialize_windows(
    windows: dict[str, list[tuple[int, int]]]
) -> dict[str, list[list[int]]]:
    return {
        key: [[start, end] for start, end in value]
        for key, value in sorted(windows.items())
    }


def _cpu_overhead_rows(traces: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stage in ("prefill", "decode", "overall"):
        for key, label in (
            ("gr_fixed_256", "GR fixed 256"),
            ("gr_dynamic_64_128_256", "GR dynamic 64/128/256"),
            ("sglang_fixed_256", "SGLang fixed 256"),
        ):
            summary = traces[key]["stage_summaries"].get(stage) or {}
            rows.append(
                {
                    "stage": stage,
                    "case": label,
                    **summary,
                }
            )
    return rows


def _decode_category_rows(traces: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    categories = sorted(
        {
            category
            for trace in traces.values()
            for category in trace.get("decode_kernel_categories_ms", {})
        }
    )
    rows = []
    for category in categories:
        rows.append(
            {
                "category": category,
                "gr_fixed_256_ms": traces["gr_fixed_256"][
                    "decode_kernel_categories_ms"
                ].get(category, 0.0),
                "gr_dynamic_64_128_256_ms": traces["gr_dynamic_64_128_256"][
                    "decode_kernel_categories_ms"
                ].get(category, 0.0),
                "sglang_fixed_256_ms": traces["sglang_fixed_256"][
                    "decode_kernel_categories_ms"
                ].get(category, 0.0),
            }
        )
    return sorted(rows, key=lambda row: row["gr_fixed_256_ms"], reverse=True)


def _decode_top_diff_rows(
    gr_trace: dict[str, Any],
    sglang_trace: dict[str, Any],
) -> list[dict[str, Any]]:
    gr = {row["name"]: row for row in gr_trace["decode_top_non_attention_kernels"]}
    sg = {row["name"]: row for row in sglang_trace["decode_top_non_attention_kernels"]}
    names = set(gr) | set(sg)
    rows = []
    for name in names:
        gr_ms = (gr.get(name) or {}).get("total_ms", 0.0)
        sg_ms = (sg.get(name) or {}).get("total_ms", 0.0)
        rows.append(
            {
                "name": name,
                "gr_fixed_ms": gr_ms,
                "sglang_corrected_ms": sg_ms,
                "delta_ms": gr_ms - sg_ms,
                "gr_count": (gr.get(name) or {}).get("count", 0),
                "sglang_count": (sg.get(name) or {}).get("count", 0),
            }
        )
    return sorted(rows, key=lambda row: row["delta_ms"], reverse=True)[:18]


def _cpu_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| stage | case | window ms | GPU activity union ms | idle/host gap ms | runtime API ms | runtime calls | runtime gaps >50us ms | graph launches | kernel launches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {case} | {window} | {gpu} | {idle} | {runtime} | {runtime_count} | {gap} | {graphs} | {kernels} |".format(
                stage=row["stage"],
                case=row["case"],
                window=_fmt(row.get("window_ms")),
                gpu=_fmt(row.get("gpu_activity_union_ms")),
                idle=_fmt(row.get("gpu_idle_or_host_gap_ms")),
                runtime=_fmt(row.get("runtime_api_sum_ms")),
                runtime_count=row.get("runtime_api_count", ""),
                gap=_fmt(row.get("runtime_gap_gt_50us_ms")),
                graphs=row.get("cuda_graph_launches", ""),
                kernels=row.get("kernel_launches", ""),
            )
        )
    return lines


def _decode_category_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| category | GR fixed ms | GR dynamic ms | SGLang corrected ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{category}` | {fixed} | {dynamic} | {sglang} |".format(
                category=row["category"],
                fixed=_fmt(row["gr_fixed_256_ms"]),
                dynamic=_fmt(row["gr_dynamic_64_128_256_ms"]),
                sglang=_fmt(row["sglang_fixed_256_ms"]),
            )
        )
    return lines


def _diff_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| kernel bucket | GR fixed ms | SGLang corrected ms | delta ms | GR count | SGLang count |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{name}` | {gr} | {sg} | {delta} | {gr_count} | {sg_count} |".format(
                name=row["name"],
                gr=_fmt(row["gr_fixed_ms"]),
                sg=_fmt(row["sglang_corrected_ms"]),
                delta=_fmt(row["delta_ms"]),
                gr_count=row["gr_count"],
                sg_count=row["sglang_count"],
            )
        )
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr-fixed-sqlite", required=True)
    parser.add_argument("--gr-dynamic-sqlite", required=True)
    parser.add_argument("--sglang-sqlite", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(
        gr_fixed=Path(args.gr_fixed_sqlite),
        gr_dynamic=Path(args.gr_dynamic_sqlite),
        sglang=Path(args.sglang_sqlite),
    )
    write_json(args.output_json, report)
    write_markdown(report, Path(args.output_markdown))
    print(f"wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
