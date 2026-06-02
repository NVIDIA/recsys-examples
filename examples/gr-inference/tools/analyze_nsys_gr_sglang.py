"""Summarize GR vs SGLang nsys sqlite exports for short-context beam runs."""

from __future__ import annotations

import argparse
import ast
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from tool_utils import json_dumps
from tool_utils import read_json as _load_json
from tool_utils import write_json

NS_PER_MS = 1_000_000.0

RUNTIME_TABLE = "CUPTI_ACTIVITY_KIND_RUNTIME"
KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
NVTX_TABLE = "NVTX_EVENTS"
STRINGS_TABLE = "StringIds"

KERNEL_CATEGORIES = (
    (
        "topK / beam selection",
        ("topk", "top_k", "sort", "select", "gather", "scatter", "take_along", "beam"),
    ),
    (
        "logits / log_softmax",
        ("logsoftmax", "log_softmax", "lm_head", "logits"),
    ),
    (
        "QKV / qk_norm_rope",
        ("qk_norm", "rope", "rotary", "qkv", "query_key_value"),
    ),
    (
        "attention",
        ("attention", "attn", "flashattn", "flash_attn", "sdpa", "fmha", "decode_att"),
    ),
    (
        "GEMM / linear",
        ("nvjet", "gemm", "matmul", "cublas", "ampere_fp16_s16816"),
    ),
    (
        "MLP activation",
        ("mlp", "silu", "swiglu", "act_and_mul", "gate_up", "down_proj"),
    ),
    (
        "RMSNorm / layernorm",
        ("rmsnorm", "layernorm"),
    ),
    (
        "memory copy / fill",
        ("direct_copy", "copy_kernel", "memcpy", "memset"),
    ),
)

NVTX_STAGE_PATTERNS = {
    "prefill": ("prefill",),
    "decode_microbatch": ("decode_microbatch_total", "decode_microbatch"),
    "topK / beam selection": ("topk_indices", "beam_selection"),
    "model.forward_decode_step": ("model.forward_decode_step",),
}

STAGES = ("overall", "prefill", "decode")


def analyze(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        tables = _tables(conn)
        strings = _strings(conn) if STRINGS_TABLE in tables else {}
        kernels = _kernel_rows(conn, strings) if KERNEL_TABLE in tables else []
        runtime = _runtime_rows(conn, strings) if RUNTIME_TABLE in tables else []
        nvtx = _nvtx_rows(conn, strings) if NVTX_TABLE in tables else []

    kernel_total_ms = _sum_duration_ms(kernels)
    runtime_total_ms = _sum_duration_ms(runtime)
    cuda_window_start, cuda_window_end = _event_window(kernels, runtime)
    active_window_start, active_window_end = _event_window(kernels)
    nvtx_window_start, nvtx_window_end = _event_window(nvtx)
    cuda_window_ms = _duration_ms(cuda_window_start, cuda_window_end)
    kernel_gap_ms = (
        max(0.0, cuda_window_ms - kernel_total_ms)
        if cuda_window_ms is not None
        else None
    )
    stage_windows = _stage_windows(nvtx, kernels)
    return {
        "path": str(path),
        "window_ms": cuda_window_ms,
        "cuda_window_ms": cuda_window_ms,
        "active_cuda_window_ms": _duration_ms(active_window_start, active_window_end),
        "nvtx_window_ms": _duration_ms(nvtx_window_start, nvtx_window_end),
        "kernel_total_ms": kernel_total_ms,
        "cuda_window_minus_kernel_total_ms": kernel_gap_ms,
        "runtime_total_ms": runtime_total_ms,
        "active_runtime_total_ms": _sum_overlap_ms(
            runtime, stage_windows.get("overall") or []
        ),
        "kernel_launch_count": len(kernels),
        "cuda_graph_launches": _count_graph_launches(runtime),
        "cpu_gap_ms_over_50us": _gap_ms(runtime, threshold_ns=50_000),
        "kernel_categories_ms": _kernel_categories(kernels),
        "stage_windows": _serialize_windows(stage_windows),
        "stage_breakdown": {
            stage: _stage_breakdown(kernels, runtime, stage_windows.get(stage) or [])
            for stage in STAGES
        },
        "top_kernel_names": _top_names(kernels, limit=12),
        "cuda_runtime_calls": _top_names(runtime, limit=12),
        "nvtx_stages_ms": _nvtx_stages(nvtx),
        "nvtx_module_buckets_ms": _nvtx_module_buckets(nvtx),
        "top_nvtx_names": _top_names(nvtx, limit=20),
        "decode_step_ms": _decode_step_ms(nvtx),
        "available_tables": sorted(tables),
    }


def build_comparison(
    gr: dict[str, Any],
    sglang: dict[str, Any],
    *,
    gr_benchmark: dict[str, Any] | None = None,
    sglang_benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overall_rows = _overall_rows(gr, sglang)
    stage_rows = {
        "prefill": _stage_rows(gr, sglang, "prefill"),
        "decode": _stage_rows(gr, sglang, "decode"),
    }
    legacy_rows = [
        _row(metric, gr_value, sglang_value, notes)
        for metric, gr_value, sglang_value, notes in (
            (
                "CUDA capture window",
                gr.get("cuda_window_ms"),
                sglang.get("cuda_window_ms"),
                "raw kernel/runtime event window; can include CUDA Graph precapture",
            ),
            (
                "Kernel total",
                gr.get("kernel_total_ms"),
                sglang.get("kernel_total_ms"),
                "sum of all captured CUDA kernel durations",
            ),
            (
                "Non-attention kernel total",
                _non_attention_kernel_total(gr),
                _non_attention_kernel_total(sglang),
                "legacy mixed bucket; prefer the stage tables above",
            ),
            (
                "CUDA window - kernel total",
                gr.get("cuda_window_minus_kernel_total_ms"),
                sglang.get("cuda_window_minus_kernel_total_ms"),
                "rough host/runtime/scheduling gap; kernels may overlap",
            ),
            (
                "CUDA runtime API total",
                gr.get("runtime_total_ms"),
                sglang.get("runtime_total_ms"),
                "raw sum of captured CUDA runtime API durations",
            ),
            (
                "NVTX span",
                gr.get("nvtx_window_ms"),
                sglang.get("nvtx_window_ms"),
                "diagnostic only; may include long-lived ranges",
            ),
            (
                "prefill NVTX total",
                _stage(gr, "prefill"),
                _stage(sglang, "prefill"),
                "legacy broad NVTX-name sum; can double count nested ranges",
            ),
        )
    ]
    decode_gr = gr.get("decode_step_ms") or []
    decode_sg = sglang.get("decode_step_ms") or []
    for idx in range(max(3, len(decode_gr), len(decode_sg))):
        legacy_rows.append(
            _row(
                f"decode step {idx + 1}",
                _list_get(decode_gr, idx),
                _list_get(decode_sg, idx),
                "GR has initial token selection in prefill path",
            )
        )
    legacy_rows.extend(
        _row(metric, gr.get(key), sglang.get(key), notes)
        for metric, key, notes in (
            (
                "CUDA graph launches",
                "cuda_graph_launches",
                "raw CUDA runtime API calls",
            ),
            ("kernel launch count", "kernel_launch_count", "captured kernel rows"),
            (
                "CPU gaps",
                "cpu_gap_ms_over_50us",
                "raw gaps between CUDA runtime calls >50us",
            ),
        )
    )
    for category, _ in KERNEL_CATEGORIES:
        legacy_rows.append(
            _row(
                category,
                _category_value(gr, category),
                _category_value(sglang, category),
                "legacy NVTX module bucket if present, else heuristic kernel-name bucket",
            )
        )
    for bucket in (
        "qkv_proj",
        "qk_norm_rope",
        "attention",
        "gate_up_proj",
        "silu_mul",
        "down_proj",
        "logits",
        "rmsnorm",
    ):
        gr_value = (gr.get("nvtx_module_buckets_ms") or {}).get(bucket)
        sglang_value = (sglang.get("nvtx_module_buckets_ms") or {}).get(bucket)
        if gr_value is not None or sglang_value is not None:
            legacy_rows.append(
                _row(
                    f"NVTX {bucket}",
                    gr_value,
                    sglang_value,
                    "fine-grained module/stage NVTX bucket",
                )
            )
    return {
        "gr": gr,
        "sglang": sglang,
        "gr_benchmark": gr_benchmark,
        "sglang_benchmark": sglang_benchmark,
        "comparison_rows": overall_rows,
        "overall_rows": overall_rows,
        "stage_rows": stage_rows,
        "legacy_comparison_rows": legacy_rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# GR vs SGLang nsys Breakdown",
        "",
        "Primary rows use active CUDA/stage windows and pure kernel-name buckets. This avoids mixing CUDA Graph visibility differences with SGLang module NVTX buckets.",
        "",
        "## Overall",
        "",
        *_comparison_table_lines(report["overall_rows"]),
        "",
        "## Prefill Stage",
        "",
        *_comparison_table_lines(report["stage_rows"]["prefill"]),
        "",
        "## Decode Stage",
        "",
        *_comparison_table_lines(report["stage_rows"]["decode"]),
        "",
        "## Legacy Flat Details",
        "",
        *_comparison_table_lines(report["legacy_comparison_rows"]),
    ]
    lines.extend(["", "## Overall Kernel Buckets", "", *_kernel_bucket_lines(report)])
    for label, data in (("GR", report["gr"]), ("SGLang", report["sglang"])):
        for title, key in (
            ("Kernels", "top_kernel_names"),
            ("CUDA Runtime Calls", "cuda_runtime_calls"),
            ("NVTX Ranges", "top_nvtx_names"),
        ):
            lines.extend(
                [
                    "",
                    f"## Top {label} {title}",
                    "",
                    *_top_name_lines(data.get(key) or []),
                ]
            )
    lines.extend(["", "## NVTX Bucket Totals", "", *_nvtx_bucket_lines(report), ""])
    warnings = _diagnostic_warnings(report)
    if warnings:
        lines.extend(
            ["## Diagnostic Notes", "", *[f"- {warning}" for warning in warnings], ""]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _strings(conn: sqlite3.Connection) -> dict[int, str]:
    columns = _columns(conn, STRINGS_TABLE)
    id_col = "id" if "id" in columns else "_id_"
    value_col = "value" if "value" in columns else "string"
    return {
        int(row[id_col]): str(row[value_col])
        for row in conn.execute(f"SELECT {id_col}, {value_col} FROM {STRINGS_TABLE}")
    }


def _kernel_rows(
    conn: sqlite3.Connection, strings: dict[int, str]
) -> list[dict[str, Any]]:
    return _event_rows(
        conn,
        KERNEL_TABLE,
        strings,
        ("demangledName", "shortName", "mangledName", "name"),
    )


def _runtime_rows(
    conn: sqlite3.Connection, strings: dict[int, str]
) -> list[dict[str, Any]]:
    return _event_rows(conn, RUNTIME_TABLE, strings, ("nameId", "name"))


def _nvtx_rows(
    conn: sqlite3.Connection, strings: dict[int, str]
) -> list[dict[str, Any]]:
    return _event_rows(
        conn,
        NVTX_TABLE,
        strings,
        ("text", "name", "message", "textId", "nameId", "messageId"),
    )


def _event_rows(
    conn: sqlite3.Connection,
    table: str,
    strings: dict[int, str],
    name_cols: tuple[str, ...],
) -> list[dict[str, Any]]:
    columns = _columns(conn, table)
    return [
        _event(row, columns=columns, strings=strings, name_cols=name_cols)
        for row in conn.execute(f"SELECT * FROM {table}")
    ]


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _event(
    row: sqlite3.Row,
    *,
    columns: set[str],
    strings: dict[int, str],
    name_cols: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "start": _number(row["start"]) if "start" in columns else None,
        "end": _number(row["end"]) if "end" in columns else None,
        "name": _resolve_name(
            row, columns=columns, strings=strings, name_cols=name_cols
        ),
    }


def _resolve_name(
    row: sqlite3.Row,
    *,
    columns: set[str],
    strings: dict[int, str],
    name_cols: tuple[str, ...],
) -> str:
    for column in name_cols:
        if column not in columns:
            continue
        value = row[column]
        if value is None:
            continue
        if isinstance(value, int) and value in strings:
            return strings[value]
        return str(value)
    return "<unknown>"


def _number(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _duration_ms(start: int | None, end: int | None) -> float | None:
    if start is None or end is None or end < start:
        return None
    return (end - start) / NS_PER_MS


def _event_duration_ms(event: dict[str, Any]) -> float:
    return _duration_ms(event.get("start"), event.get("end")) or 0.0


def _sum_duration_ms(events: Iterable[dict[str, Any]]) -> float:
    return sum(_event_duration_ms(event) for event in events)


def _event_window(*groups: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    starts = [
        event["start"]
        for group in groups
        for event in group
        if event.get("start") is not None
    ]
    ends = [
        event["end"]
        for group in groups
        for event in group
        if event.get("end") is not None
    ]
    return (min(starts), max(ends)) if starts and ends else (None, None)


def _stage_windows(
    nvtx: list[dict[str, Any]],
    kernels: list[dict[str, Any]],
) -> dict[str, list[tuple[int, int]]]:
    kernel_start, kernel_end = _event_window(kernels)
    windows: dict[str, list[tuple[int, int]]] = {}
    if kernel_start is not None and kernel_end is not None:
        windows["overall"] = [(kernel_start, kernel_end)]

    gr_prefill = _windows_with_exact_name(nvtx, {"continuous.prefill"})
    gr_decode = _windows_with_exact_name(nvtx, {"continuous.decode_microbatch_total"})
    if not gr_prefill:
        gr_prefill = _windows_with_exact_name(nvtx, {"model.forward_prefill"})
    if not gr_decode:
        gr_decode = _windows_with_exact_name(nvtx, {"model.forward_decode_step"})
    if gr_prefill or gr_decode:
        windows.update(_paired_stage_windows(gr_prefill, gr_decode))
        return windows

    sglang_prefill = _windows_with_exact_name(nvtx, {"sglang.prefill"})
    sglang_decode = _windows_with_exact_name(nvtx, {"sglang.decode"})
    if sglang_prefill or sglang_decode:
        windows.update(_paired_stage_windows(sglang_prefill, sglang_decode))
        return windows

    if kernel_start is None or kernel_end is None:
        return windows
    windows.update(_sglang_stage_windows(nvtx, (kernel_start, kernel_end)))
    return windows


def _windows_with_exact_name(
    events: list[dict[str, Any]],
    names: set[str],
) -> list[tuple[int, int]]:
    windows = []
    for event in events:
        if event.get("name") not in names:
            continue
        start = event.get("start")
        end = event.get("end")
        if start is not None and end is not None and end > start:
            windows.append((start, end))
    return windows


def _sglang_stage_windows(
    nvtx: list[dict[str, Any]],
    kernel_window: tuple[int, int],
) -> dict[str, list[tuple[int, int]]]:
    model_events = []
    for event in nvtx:
        parsed = _parse_sglang_model_event(event)
        if parsed is not None:
            model_events.append(parsed)
    if not model_events:
        return {}
    model_events.sort(key=lambda event: event["start"])
    dims = [event["tokens"] for event in model_events if event["tokens"] is not None]
    if not dims:
        return {}
    prefill_tokens = max(dims)
    decode_starts = [
        event["start"]
        for event in model_events
        if event["tokens"] is not None and event["tokens"] < prefill_tokens
    ]
    start, end = kernel_window
    if not decode_starts:
        return {"prefill": [(start, end)]}
    decode_start = min(decode_starts)
    return {
        "prefill": [(start, decode_start)],
        "decode": [(decode_start, end)],
    }


def _parse_sglang_model_event(event: dict[str, Any]) -> dict[str, int | None] | None:
    name = event.get("name")
    if not isinstance(name, str) or not name.startswith("{"):
        return None
    try:
        payload = ast.literal_eval(name)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("Module") != "model.model":
        return None
    start = event.get("start")
    end = event.get("end")
    if start is None or end is None or end <= start:
        return None
    tokens = _first_input_dim(payload.get("Inputs"))
    return {"start": start, "end": end, "tokens": tokens}


def _first_input_dim(inputs: Any) -> int | None:
    if not isinstance(inputs, list) or not inputs:
        return None
    first = inputs[0]
    if not isinstance(first, list) or not first:
        return None
    value = first[0]
    return int(value) if isinstance(value, int) else None


def _bounds_window(windows: list[tuple[int, int]]) -> tuple[int, int]:
    return min(start for start, _ in windows), max(end for _, end in windows)


def _paired_stage_windows(
    prefill_windows: list[tuple[int, int]],
    decode_windows: list[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]]:
    windows: dict[str, list[tuple[int, int]]] = {}
    if prefill_windows and decode_windows:
        prefill_start = min(start for start, _ in prefill_windows)
        decode_start = min(start for start, _ in decode_windows)
        prefill_end = max(decode_start, max(end for _, end in prefill_windows))
        windows["prefill"] = [(prefill_start, prefill_end)]
        windows["decode"] = [(decode_start, max(end for _, end in decode_windows))]
        return windows
    if prefill_windows:
        windows["prefill"] = [_bounds_window(prefill_windows)]
    if decode_windows:
        windows["decode"] = [_bounds_window(decode_windows)]
    return windows


def _serialize_windows(
    windows: dict[str, list[tuple[int, int]]]
) -> dict[str, list[list[int]]]:
    return {
        stage: [[start, end] for start, end in values]
        for stage, values in windows.items()
    }


def _stage_breakdown(
    kernels: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
    windows: list[tuple[int, int]],
) -> dict[str, Any]:
    if not windows:
        return {
            "window_ms": None,
            "cuda_runtime_api_ms": None,
            "cuda_graph_runtime_api_ms": None,
            "cpu_gap_ms_over_50us": None,
            "cpu_overhead_ms": None,
            "kernel_total_ms": None,
            "attention_kernel_ms": None,
            "other_kernel_ms": None,
            "kernel_launch_count": None,
            "cuda_graph_launches": None,
            "kernel_categories_ms": {},
        }
    kernel_categories = _kernel_categories(kernels, windows=windows)
    kernel_total_ms = _sum_overlap_ms(kernels, windows)
    attention_ms = kernel_categories.get("attention", 0.0)
    window_ms = _windows_duration_ms(windows)
    clipped_runtime = _clip_events_to_windows(runtime, windows)
    return {
        "window_ms": window_ms,
        "cuda_runtime_api_ms": _sum_duration_ms(clipped_runtime),
        "cuda_graph_runtime_api_ms": _sum_graph_runtime_api_ms(clipped_runtime),
        "cpu_gap_ms_over_50us": _gap_ms(clipped_runtime, threshold_ns=50_000),
        "cpu_overhead_ms": (
            max(0.0, window_ms - kernel_total_ms) if window_ms is not None else None
        ),
        "kernel_total_ms": kernel_total_ms,
        "attention_kernel_ms": attention_ms,
        "other_kernel_ms": max(0.0, kernel_total_ms - attention_ms),
        "kernel_launch_count": _count_overlapping(kernels, windows),
        "cuda_graph_launches": _count_graph_launches(clipped_runtime),
        "kernel_categories_ms": kernel_categories,
    }


def _windows_duration_ms(windows: list[tuple[int, int]]) -> float | None:
    if not windows:
        return None
    return (
        sum(max(0, end - start) for start, end in _merge_windows(windows)) / NS_PER_MS
    )


def _sum_overlap_ms(
    events: Iterable[dict[str, Any]],
    windows: list[tuple[int, int]],
) -> float:
    return sum(_event_overlap_ms(event, windows) for event in events)


def _sum_graph_runtime_api_ms(events: Iterable[dict[str, Any]]) -> float:
    return sum(
        _event_duration_ms(event)
        for event in events
        if "graph" in str(event.get("name", "")).lower()
    )


def _event_overlap_ms(event: dict[str, Any], windows: list[tuple[int, int]]) -> float:
    start = event.get("start")
    end = event.get("end")
    if start is None or end is None or end <= start or not windows:
        return 0.0
    total = 0
    for window_start, window_end in _merge_windows(windows):
        overlap_start = max(start, window_start)
        overlap_end = min(end, window_end)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total / NS_PER_MS


def _count_overlapping(
    events: Iterable[dict[str, Any]],
    windows: list[tuple[int, int]],
) -> int:
    return sum(1 for event in events if _event_overlap_ms(event, windows) > 0.0)


def _clip_events_to_windows(
    events: Iterable[dict[str, Any]],
    windows: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    clipped = []
    merged = _merge_windows(windows)
    for event in events:
        start = event.get("start")
        end = event.get("end")
        if start is None or end is None or end <= start:
            continue
        for window_start, window_end in merged:
            overlap_start = max(start, window_start)
            overlap_end = min(end, window_end)
            if overlap_end > overlap_start:
                clipped.append(
                    {
                        "start": overlap_start,
                        "end": overlap_end,
                        "name": event.get("name", "<unknown>"),
                    }
                )
    return clipped


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in windows if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _count_graph_launches(events: list[dict[str, Any]]) -> int:
    return sum(
        1
        for event in events
        if "graph" in event["name"].lower() and "launch" in event["name"].lower()
    )


def _gap_ms(events: list[dict[str, Any]], *, threshold_ns: int) -> float:
    ordered = sorted(
        (
            event
            for event in events
            if event.get("start") is not None and event.get("end") is not None
        ),
        key=lambda event: event["start"],
    )
    total = 0
    previous_end = None
    for event in ordered:
        if previous_end is not None and event["start"] > previous_end:
            gap = event["start"] - previous_end
            if gap >= threshold_ns:
                total += gap
        previous_end = max(previous_end or event["end"], event["end"])
    return total / NS_PER_MS


def _kernel_categories(
    kernels: list[dict[str, Any]],
    *,
    windows: list[tuple[int, int]] | None = None,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for kernel in kernels:
        lower = kernel["name"].lower()
        duration = (
            _event_overlap_ms(kernel, windows)
            if windows is not None
            else _event_duration_ms(kernel)
        )
        if duration <= 0.0:
            continue
        matched = False
        for category, needles in KERNEL_CATEGORIES:
            if any(needle in lower for needle in needles):
                totals[category] += duration
                matched = True
                break
        if not matched:
            totals["other"] += duration
    return dict(sorted(totals.items()))


def _top_names(events: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for event in events:
        name = event["name"]
        totals[name] += _event_duration_ms(event)
        counts[name] += 1
    rows = [
        {"name": name, "total_ms": total, "count": counts[name]}
        for name, total in totals.items()
    ]
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)[:limit]


def _nvtx_stages(events: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for event in events:
        lower = event["name"].lower()
        for stage, needles in NVTX_STAGE_PATTERNS.items():
            if any(needle in lower for needle in needles):
                totals[stage] += _event_duration_ms(event)
    return dict(sorted(totals.items()))


def _nvtx_module_buckets(events: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, float] = defaultdict(float)
    for event in events:
        name = event["name"]
        lower = name.lower()
        duration = _event_duration_ms(event)
        if "qkv_proj" in lower:
            buckets["qkv_proj"] += duration
            buckets["QKV / qk_norm_rope"] += duration
        elif "q_norm" in lower or "k_norm" in lower or "rotary_emb" in lower:
            buckets["qk_norm_rope"] += duration
            buckets["QKV / qk_norm_rope"] += duration
        elif "self_attn" in lower or "radixattention" in lower or ".attn" in lower:
            buckets["attention"] += duration
        elif "gate_up_proj" in lower:
            buckets["gate_up_proj"] += duration
            buckets["MLP"] += duration
        elif "siluandmul" in lower or "act_fn" in lower:
            buckets["silu_mul"] += duration
            buckets["MLP"] += duration
        elif "down_proj" in lower:
            buckets["down_proj"] += duration
            buckets["MLP"] += duration
        elif "mlp" in lower:
            buckets["MLP"] += duration
        elif "logits_processor" in lower or "lm_head" in lower:
            buckets["logits"] += duration
        elif (
            "layernorm" in lower
            or "rmsnorm" in lower
            or "input_norm" in lower
            or "post_norm" in lower
            or "final_norm" in lower
        ):
            buckets["rmsnorm"] += duration
    return dict(sorted(buckets.items()))


def _decode_step_ms(events: list[dict[str, Any]]) -> list[float]:
    rows = [
        _event_duration_ms(event)
        for event in events
        if "decode_microbatch" in event["name"].lower()
    ]
    return rows


def _stage(report: dict[str, Any], name: str) -> float | None:
    return (report.get("nvtx_stages_ms") or {}).get(name)


def _list_get(values: list[float], index: int) -> float | None:
    return values[index] if index < len(values) else None


def _row(metric: str, gr: Any, sglang: Any, notes: str) -> dict[str, Any]:
    return {"metric": metric, "gr": gr, "sglang": sglang, "notes": notes}


def _overall_rows(
    gr: dict[str, Any],
    sglang: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        *_breakdown_rows(
            gr,
            sglang,
            (
                (
                    "Active CUDA window",
                    "overall",
                    "window_ms",
                    "first-to-last captured kernel; avoids CUDA Graph precapture runtime",
                ),
                (
                    "CUDA runtime API total",
                    "overall",
                    "cuda_runtime_api_ms",
                    "runtime API duration clipped to the active CUDA window",
                ),
                (
                    "CUDA graph API total",
                    "overall",
                    "cuda_graph_runtime_api_ms",
                    "CUDA graph runtime API duration clipped to the active CUDA window",
                ),
                (
                    "CPU runtime gaps >50us",
                    "overall",
                    "cpu_gap_ms_over_50us",
                    "gaps between CUDA runtime calls inside the active window",
                ),
                (
                    "Kernel total",
                    "overall",
                    "kernel_total_ms",
                    "sum of captured CUDA kernel durations",
                ),
                (
                    "Decode attention kernels",
                    "decode",
                    "attention_kernel_ms",
                    "attention bucket inside the decode stage",
                ),
            ),
        ),
        _row(
            "Other kernels excluding decode attention",
            _other_excluding_decode_attention(gr),
            _other_excluding_decode_attention(sglang),
            "overall kernel total minus decode-stage attention kernels",
        ),
        *_breakdown_rows(
            gr,
            sglang,
            (
                (
                    "CUDA graph launches",
                    "overall",
                    "cuda_graph_launches",
                    "cudaGraphLaunch runtime calls in the active window",
                ),
                (
                    "Kernel launches",
                    "overall",
                    "kernel_launch_count",
                    "kernel rows overlapping the active window",
                ),
            ),
        ),
    ]


def _stage_rows(
    gr: dict[str, Any],
    sglang: dict[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    stage_label = "prefill" if stage == "prefill" else "decode"
    return _breakdown_rows(
        gr,
        sglang,
        (
            (
                "Stage total",
                stage,
                "window_ms",
                f"{stage_label} stage window from NVTX boundary; includes host gaps inside the stage",
            ),
            (
                "Attention kernels",
                stage,
                "attention_kernel_ms",
                "prefill attention for prefill; decode attention for decode",
            ),
            (
                "Non-attention kernels",
                stage,
                "other_kernel_ms",
                "stage kernel total minus stage attention kernels",
            ),
            (
                "CPU overhead",
                stage,
                "cpu_overhead_ms",
                "stage total minus stage kernel total; rough host/runtime/bubble component",
            ),
        ),
    )


def _breakdown_rows(
    gr: dict[str, Any],
    sglang: dict[str, Any],
    specs: tuple[tuple[str, str, str, str], ...],
) -> list[dict[str, Any]]:
    return [
        _row(
            metric,
            _breakdown_value(gr, stage, key),
            _breakdown_value(sglang, stage, key),
            notes,
        )
        for metric, stage, key, notes in specs
    ]


def _breakdown_value(report: dict[str, Any], stage: str, key: str) -> Any:
    return ((report.get("stage_breakdown") or {}).get(stage) or {}).get(key)


def _other_excluding_decode_attention(report: dict[str, Any]) -> float | None:
    kernel_total = _breakdown_value(report, "overall", "kernel_total_ms")
    if kernel_total is None:
        return None
    decode_attention = _breakdown_value(report, "decode", "attention_kernel_ms") or 0.0
    return max(0.0, float(kernel_total) - float(decode_attention))


def _category_value(report: dict[str, Any], category: str) -> float | None:
    return (report.get("nvtx_module_buckets_ms") or {}).get(category) or (
        report.get("kernel_categories_ms") or {}
    ).get(category)


def _non_attention_kernel_total(report: dict[str, Any]) -> float | None:
    kernel_total = report.get("kernel_total_ms")
    if kernel_total is None:
        return None
    attention = _category_value(report, "attention") or 0.0
    return max(0.0, float(kernel_total) - float(attention))


def _nvtx_bucket_lines(report: dict[str, Any]) -> list[str]:
    gr = report["gr"].get("nvtx_module_buckets_ms") or {}
    sglang = report["sglang"].get("nvtx_module_buckets_ms") or {}
    return _bucket_lines(gr, sglang, empty="No NVTX bucket rows found.")


def _comparison_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Metric / stage | GR | SGLang | Notes |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {gr} | {sglang} | {notes} |".format(
                metric=row["metric"],
                gr=_format_value(row["gr"]),
                sglang=_format_value(row["sglang"]),
                notes=row["notes"],
            )
        )
    return lines


def _kernel_bucket_lines(report: dict[str, Any]) -> list[str]:
    gr = _breakdown_value(report["gr"], "overall", "kernel_categories_ms") or {}
    sglang = _breakdown_value(report["sglang"], "overall", "kernel_categories_ms") or {}
    return _bucket_lines(gr, sglang, empty="No kernel bucket rows found.")


def _bucket_lines(
    gr: dict[str, Any], sglang: dict[str, Any], *, empty: str
) -> list[str]:
    keys = sorted(set(gr) | set(sglang))
    if not keys:
        return [empty]
    lines = [
        "| bucket | GR ms | SGLang ms |",
        "| --- | ---: | ---: |",
    ]
    for key in keys:
        lines.append(
            f"| `{key}` | {_format_value(gr.get(key))} | {_format_value(sglang.get(key))} |"
        )
    return lines


def _diagnostic_warnings(report: dict[str, Any]) -> list[str]:
    warnings = []
    if (report.get("sglang_benchmark") or {}).get("profile_modules"):
        warnings.append(
            "SGLang benchmark JSON has profile_modules=true. The wrapper enables layerwise NVTX by disabling SGLang piecewise CUDA Graph, so this report is for attribution and not a fair CUDA Graph/runtime comparison."
        )
    if not (report["sglang"].get("nvtx_module_buckets_ms") or {}):
        warnings.append(
            "SGLang module NVTX buckets are empty. For per-layer/per-op SGLang breakdown, rerun with SGLANG_PROFILE_MODULES=1; use the non-profile run for fair latency."
        )
    if not (report["gr"].get("nvtx_stages_ms") or {}):
        warnings.append(
            "GR NVTX stages are empty. Rerun with GR_PROFILE_CONTINUOUS_DECODE=1 for GR prefill/decode stage ranges."
        )
    for label in ("gr", "sglang"):
        data = report[label]
        decode = (data.get("stage_breakdown") or {}).get("decode") or {}
        if (decode.get("cuda_graph_launches") or 0) > 0 and (
            (decode.get("kernel_launch_count") or 0) == 0
            or (decode.get("attention_kernel_ms") or 0.0) == 0.0
        ):
            warnings.append(
                f"{label} decode has CUDA graph launches but no expanded decode attention kernels. Rerun nsys with --cuda-graph-trace=node:nvtx-precapture when comparing decode attention and other decode kernels."
            )
    return warnings


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _top_name_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No kernel rows found."]
    return [
        f"- `{row['name']}`: {_format_value(row['total_ms'])} ms, count={row['count']}"
        for row in rows
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr-sqlite", required=True)
    parser.add_argument("--sglang-sqlite", required=True)
    parser.add_argument("--gr-json")
    parser.add_argument("--sglang-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_comparison(
        analyze(Path(args.gr_sqlite)),
        analyze(Path(args.sglang_sqlite)),
        gr_benchmark=_load_json(Path(args.gr_json)) if args.gr_json else None,
        sglang_benchmark=_load_json(Path(args.sglang_json))
        if args.sglang_json
        else None,
    )
    write_json(args.output_json, report)
    write_markdown(report, Path(args.output_markdown))
    print(json_dumps(report["comparison_rows"]))


if __name__ == "__main__":
    main()
