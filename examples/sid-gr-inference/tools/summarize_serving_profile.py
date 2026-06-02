# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize GR continuous serving profile JSON snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tool_utils import read_json

DEFAULT_PROFILE_KEYS = (
    "model_forward_prefill_ms",
    "prefill_layer_total_ms",
    "prefill_attention_ms",
    "prefill_qkv_ms",
    "prefill_qkv_proj_ms",
    "prefill_qk_norm_rope_ms",
    "prefill_post_attention_ms",
    "prefill_mlp_ms",
    "prefill_embed_tokens_ms",
    "prefill_final_norm_ms",
    "prefill_lm_head_ms",
    "continuous_decode_batch_build_ms",
    "continuous_topk_indices_ms",
    "continuous_beam_selection_ms",
    "continuous_beam_kv_scatter_ms",
    "model_forward_decode_step_ms",
    "decode_attention_ms",
    "qkv_ms",
    "qk_norm_rope_ms",
    "post_attention_mlp_ms",
    "beam_kv_write_ms",
)


def summarize_profile(
    summary: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None = None,
    keys: tuple[str, ...] = DEFAULT_PROFILE_KEYS,
) -> dict[str, Any]:
    aggregate = _aggregate(summary)
    baseline_aggregate = _aggregate(baseline or {}) if baseline is not None else {}
    wall_ms = _number(summary.get("wall_ms"))
    decode_ms = _number(summary.get("scheduler_metrics", {}).get("decode_ms"))
    rows = []
    for key in keys:
        value = _number(aggregate.get(key))
        baseline_value = _number(baseline_aggregate.get(key))
        row: dict[str, Any] = {
            "name": key,
            "ms": value,
            "pct_of_decode": _pct(value, decode_ms),
            "pct_of_wall": _pct(value, wall_ms),
        }
        if baseline is not None:
            row["baseline_ms"] = baseline_value
            row["delta_ms"] = (
                None
                if value is None or baseline_value is None
                else value - baseline_value
            )
            row["improvement_pct"] = _improvement_pct(value, baseline_value)
        rows.append(row)

    return {
        "wall_ms": wall_ms,
        "decode_ms": decode_ms,
        "prefill_ms": _number(summary.get("scheduler_metrics", {}).get("prefill_ms")),
        "responses": summary.get("responses"),
        "kernel_backend_selection": summary.get("kernel_backend_selection", {}),
        "flashinfer_call_counts": summary.get("flashinfer_call_counts", {}),
        "gr_trtllm_call_counts": summary.get("gr_trtllm_call_counts", {}),
        "top_buckets": sorted(
            rows,
            key=lambda row: row["ms"] if row["ms"] is not None else -1.0,
            reverse=True,
        ),
        "cache_metrics": _cache_metrics(summary),
    }


def _aggregate(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    aggregate = summary.get("decode_profile_aggregate")
    return aggregate if isinstance(aggregate, Mapping) else {}


def _cache_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    status = summary.get("scheduler_status")
    if not isinstance(status, Mapping):
        return {}
    metrics: dict[str, Any] = {}
    for key in (
        "decode_inputs_cache",
        "topk_indices_cache",
    ):
        value = status.get(key)
        if isinstance(value, Mapping):
            metrics[key] = dict(value)
    return metrics


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: float | None, total: float | None) -> float | None:
    if value is None or total is None or total <= 0:
        return None
    return 100.0 * value / total


def _improvement_pct(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    return 100.0 * (baseline - value) / baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a serving profile JSON summary")
    parser.add_argument(
        "--baseline", help="Optional baseline profile JSON for comparison"
    )
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = read_json(Path(args.path))
    baseline = read_json(Path(args.baseline)) if args.baseline else None
    report = summarize_profile(summary, baseline=baseline)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
