# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare SID-GR Inference and SGLang beam benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tool_utils import optional_float, optional_int, read_json


def compare(gr: Mapping[str, Any], sglang: Mapping[str, Any]) -> dict[str, Any]:
    gr_outputs = _gr_outputs(gr)
    sglang_outputs = _sglang_outputs(sglang)
    gr_decode_steps = _gr_decode_steps(gr)
    sglang_decode_steps = optional_int(sglang.get("decode_steps"))
    gr_output_token_budget = _gr_output_token_budget(gr_decode_steps)
    correctness = [
        _compare_request(
            match["gr_request_id"],
            gr_outputs[match["gr_request_id"]],
            match["sglang_request_id"],
            sglang_outputs[match["sglang_request_id"]],
            match_strategy=match["strategy"],
        )
        for match in _matched_requests(gr_outputs, sglang_outputs)
    ]
    return {
        "workload": {
            "model_dir": gr.get("model_dir") or sglang.get("model_dir"),
            "context_len": gr.get("context_len") or sglang.get("context_len"),
            "decode_steps": gr_decode_steps
            if gr_decode_steps == sglang_decode_steps
            else None,
            "gr_decode_steps": gr_decode_steps,
            "sglang_decode_steps": sglang_decode_steps,
            "gr_output_token_budget": gr_output_token_budget,
            "sglang_output_token_budget": sglang_decode_steps,
            "output_token_budget_match": (
                gr_output_token_budget is not None
                and sglang_decode_steps is not None
                and gr_output_token_budget == sglang_decode_steps
            ),
            "beam_width": gr.get("beam_width") or sglang.get("beam_width"),
            "gr_serving_mode": gr.get("serving_mode"),
            "sglang_arrival_mode": sglang.get("arrival_mode"),
            "gr_arrival_stagger_ticks": gr.get("arrival_stagger_ticks"),
            "sglang_arrival_stagger_ms": sglang.get("arrival_stagger_ms"),
            "sglang_arrival_burst_size": sglang.get("arrival_burst_size"),
            "matched_requests": len(correctness),
            "match_strategy": correctness[0]["match_strategy"] if correctness else None,
        },
        "performance": {
            "gr": _gr_performance(gr),
            "sglang": _sglang_performance(sglang),
        },
        "correctness": {
            "requests": correctness,
            "top1_exact_match_rate": _mean(
                row["top1_exact_match"] for row in correctness
            ),
            "topk_set_overlap_mean": _mean(
                row["topk_set_overlap"] for row in correctness
            ),
            "ordered_prefix_match_mean": _mean(
                row["ordered_prefix_match"] for row in correctness
            ),
            "token_length_match_rate": _mean(
                row["token_length_match"] for row in correctness
            ),
        },
        "caveats": (
            "GR max_decode_steps counts decode iterations after the initial prefill beam selection; "
            "fixed-length GR outputs normally contain GR decode_steps + 1 token ids.",
            "SGLang score semantics may differ from GR cumulative logprob/logit score.",
            "For fixed-length outputs, SGLang sequence_score appears to be length-normalized; "
            "the report also includes SGLang score scaled by token length for diagnostics.",
            "If SGLang output token IDs are reconstructed from text, token comparison is diagnostic.",
            "Stop handling, length penalty, tokenizer round-trip, and floating point tie-breaks can change ordering.",
        ),
    }


def _gr_outputs(summary: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records = {}
    for output in summary.get("outputs", ()) or ():
        request_id = str(output.get("workload_id") or output.get("request_id"))
        beam_results = output.get("beam_results", ()) or ()
        beams = []
        for rank, result in enumerate(beam_results):
            if not isinstance(result, Mapping):
                continue
            output_ids = result.get("output_ids")
            if not isinstance(output_ids, (list, tuple)):
                continue
            meta_info = result.get("meta_info", {}) or {}
            beams.append(
                {
                    "rank": rank,
                    "token_ids": tuple(int(token) for token in output_ids),
                    "score": optional_float(meta_info.get("sequence_score")),
                    "text": result.get("text"),
                    "raw": result,
                }
            )
        beam_details = output.get("beam_details", ()) or ()
        for detail in beam_details:
            if beams:
                break
            beams.append(
                {
                    "rank": int(detail.get("rank", len(beams))),
                    "token_ids": tuple(
                        int(token) for token in detail.get("token_ids", ()) or ()
                    ),
                    "score": optional_float(
                        detail.get("cumulative_score", detail.get("score"))
                    ),
                    "raw": detail,
                }
            )
        if not beams:
            beams.append(
                {
                    "rank": 0,
                    "token_ids": tuple(
                        int(token) for token in output.get("token_ids", ()) or ()
                    ),
                    "score": None,
                    "raw": output,
                }
            )
        records[request_id] = beams
    return records


def _sglang_outputs(summary: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    records = {}
    runs = summary.get("runs", ()) or ()
    if not runs:
        return records
    outputs = runs[-1].get("outputs", ()) or ()
    for output in outputs:
        request_id = str(output.get("workload_id") or output.get("request_id"))
        beams = []
        for beam in output.get("beams", ()) or ():
            beams.append(
                {
                    "rank": int(beam.get("rank", len(beams))),
                    "token_ids": tuple(
                        int(token) for token in beam.get("token_ids", ()) or ()
                    ),
                    "score": optional_float(beam.get("score")),
                    "text": beam.get("text"),
                    "raw": beam,
                }
            )
        records[request_id] = beams
    return records


def _gr_decode_steps(summary: Mapping[str, Any]) -> int | None:
    decode_steps = optional_int(summary.get("decode_steps"))
    if decode_steps is not None:
        return decode_steps
    for output in summary.get("outputs", ()) or ():
        metadata = output.get("metadata", {}) or {}
        decode_steps = optional_int(metadata.get("decode_steps"))
        if decode_steps is not None:
            return decode_steps
    engine_status = summary.get("engine_status", {}) or {}
    return optional_int(engine_status.get("max_decode_steps"))


def _gr_output_token_budget(decode_steps: int | None) -> int | None:
    if decode_steps is None:
        return None
    return decode_steps + 1


def _matched_requests(
    gr_outputs: Mapping[str, list[dict[str, Any]]],
    sglang_outputs: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, str]]:
    shared_ids = sorted(set(gr_outputs) & set(sglang_outputs))
    if shared_ids:
        return [
            {
                "gr_request_id": request_id,
                "sglang_request_id": request_id,
                "strategy": "request_id",
            }
            for request_id in shared_ids
        ]
    if len(gr_outputs) == len(sglang_outputs):
        return [
            {
                "gr_request_id": gr_request_id,
                "sglang_request_id": sglang_request_id,
                "strategy": "request_order_fallback",
            }
            for gr_request_id, sglang_request_id in zip(gr_outputs, sglang_outputs)
        ]
    return []


def _compare_request(
    gr_request_id: str,
    gr_beams: list[dict[str, Any]],
    sglang_request_id: str,
    sglang_beams: list[dict[str, Any]],
    *,
    match_strategy: str,
) -> dict[str, Any]:
    gr_tokens = [beam["token_ids"] for beam in gr_beams]
    sglang_tokens = [beam["token_ids"] for beam in sglang_beams]
    gr_rank_by_tokens = {tokens: rank for rank, tokens in enumerate(gr_tokens)}
    sglang_rank_by_tokens = {tokens: rank for rank, tokens in enumerate(sglang_tokens)}
    gr_set = set(gr_tokens)
    sglang_set = set(sglang_tokens)
    overlap = gr_set & sglang_set
    topk_denom = max(len(sglang_set), 1)
    ordered_matches = sum(
        1 for left, right in zip(gr_tokens, sglang_tokens) if left == right
    )
    common_prefix_count = _common_prefix_count(gr_tokens, sglang_tokens)
    gr_token_len = len(gr_tokens[0]) if gr_tokens else 0
    sglang_token_len = len(sglang_tokens[0]) if sglang_tokens else 0
    rank_movements = []
    for tokens in overlap:
        gr_rank = gr_rank_by_tokens[tokens]
        sglang_rank = sglang_rank_by_tokens[tokens]
        gr_score = gr_beams[gr_rank].get("score")
        sglang_score = sglang_beams[sglang_rank].get("score")
        rank_movements.append(
            {
                "token_ids": tokens,
                "gr_rank": gr_rank,
                "sglang_rank": sglang_rank,
                "rank_delta": gr_rank - sglang_rank,
                "gr_score": gr_score,
                "sglang_score": sglang_score,
                "score_delta": _score_delta(gr_score, sglang_score),
                "score_delta_sglang_scaled_by_token_len": (
                    _score_delta(gr_score, sglang_score * gr_token_len)
                    if sglang_score is not None and gr_token_len
                    else None
                ),
            }
        )
    rank_movements.sort(key=lambda row: row["sglang_rank"])
    scaled_score_deltas = [
        row["score_delta_sglang_scaled_by_token_len"]
        for row in rank_movements
        if row["score_delta_sglang_scaled_by_token_len"] is not None
    ]
    score_correlation = _pearson_correlation(
        [row["gr_score"] for row in rank_movements],
        [
            row["sglang_score"] * gr_token_len
            if row["sglang_score"] is not None and gr_token_len
            else None
            for row in rank_movements
        ],
    )
    gr_top1_score = gr_beams[0].get("score") if gr_beams else None
    sglang_top1_score = sglang_beams[0].get("score") if sglang_beams else None
    gr_top1 = gr_tokens[0] if gr_tokens else ()
    sglang_top1 = sglang_tokens[0] if sglang_tokens else ()
    gr_top1_rank_in_sglang = sglang_rank_by_tokens.get(gr_top1)
    sglang_top1_rank_in_gr = gr_rank_by_tokens.get(sglang_top1)
    gr_top1_sglang_score = (
        sglang_beams[gr_top1_rank_in_sglang].get("score")
        if gr_top1_rank_in_sglang is not None
        else None
    )
    sglang_top1_gr_score = (
        gr_beams[sglang_top1_rank_in_gr].get("score")
        if sglang_top1_rank_in_gr is not None
        else None
    )
    return {
        "gr_request_id": gr_request_id,
        "sglang_request_id": sglang_request_id,
        "match_strategy": match_strategy,
        "top1_exact_match": bool(
            gr_tokens and sglang_tokens and gr_tokens[0] == sglang_tokens[0]
        ),
        "topk_set_overlap": len(overlap) / topk_denom,
        "ordered_prefix_match": ordered_matches
        / max(min(len(gr_tokens), len(sglang_tokens)), 1),
        "gr_beam_count": len(gr_tokens),
        "sglang_beam_count": len(sglang_tokens),
        "gr_token_len": gr_token_len,
        "sglang_token_len": sglang_token_len,
        "token_length_match": (
            bool(gr_tokens and sglang_tokens) and gr_token_len == sglang_token_len
        ),
        "same_position_count": ordered_matches,
        "common_prefix_count": common_prefix_count,
        "rank_delta_summary": _rank_delta_summary(rank_movements),
        "score_delta_sglang_scaled_by_token_len_summary": _numeric_summary(
            scaled_score_deltas
        ),
        "score_correlation_sglang_scaled_by_token_len": score_correlation,
        "rank_movements_sample": rank_movements[:8],
        "gr_top1": gr_top1,
        "sglang_top1": sglang_top1,
        "gr_top1_score": gr_top1_score,
        "sglang_top1_score": sglang_top1_score,
        "top1_score_delta": _score_delta(gr_top1_score, sglang_top1_score),
        "gr_top1_rank_in_sglang": gr_top1_rank_in_sglang,
        "sglang_top1_rank_in_gr": sglang_top1_rank_in_gr,
        "gr_top1_sglang_score": gr_top1_sglang_score,
        "sglang_top1_gr_score": sglang_top1_gr_score,
        "gr_top1_score_delta_sglang_scaled_by_token_len": (
            _score_delta(gr_top1_score, gr_top1_sglang_score * gr_token_len)
            if gr_top1_sglang_score is not None and gr_token_len
            else None
        ),
        "sglang_top1_score_delta_sglang_scaled_by_token_len": (
            _score_delta(sglang_top1_gr_score, sglang_top1_score * gr_token_len)
            if sglang_top1_gr_score is not None
            and sglang_top1_score is not None
            and gr_token_len
            else None
        ),
    }


def _common_prefix_count(left: list[Any], right: list[Any]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def _rank_delta_summary(rank_movements: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [int(row["rank_delta"]) for row in rank_movements]
    if not deltas:
        return {
            "overlap_count": 0,
            "same_rank_count": 0,
            "within_1_count": 0,
            "within_5_count": 0,
            "within_10_count": 0,
            "mean": None,
            "median": None,
            "max_abs": None,
        }
    sorted_deltas = sorted(deltas)
    abs_deltas = [abs(delta) for delta in deltas]
    return {
        "overlap_count": len(deltas),
        "same_rank_count": sum(1 for delta in deltas if delta == 0),
        "within_1_count": sum(1 for delta in deltas if abs(delta) <= 1),
        "within_5_count": sum(1 for delta in deltas if abs(delta) <= 5),
        "within_10_count": sum(1 for delta in deltas if abs(delta) <= 10),
        "mean": sum(deltas) / len(deltas),
        "median": _median(sorted_deltas),
        "max_abs": max(abs_deltas),
    }


def _median(sorted_values: list[int] | list[float]) -> float | None:
    if not sorted_values:
        return None
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[middle])
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "mean": sum(sorted_values) / len(sorted_values),
        "median": _median(sorted_values),
        "p05": _percentile(sorted_values, 0.05),
        "p95": _percentile(sorted_values, 0.95),
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }


def _percentile(sorted_values: list[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    index = int(quantile * (len(sorted_values) - 1))
    return sorted_values[index]


def _pearson_correlation(left: list[Any], right: list[Any]) -> float | None:
    pairs = [
        (float(left_value), float(right_value))
        for left_value, right_value in zip(left, right)
        if left_value is not None and right_value is not None
    ]
    if len(pairs) < 2:
        return None
    left_mean = sum(left_value for left_value, _ in pairs) / len(pairs)
    right_mean = sum(right_value for _, right_value in pairs) / len(pairs)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    )
    left_variance = sum((left_value - left_mean) ** 2 for left_value, _ in pairs)
    right_variance = sum((right_value - right_mean) ** 2 for _, right_value in pairs)
    if left_variance == 0 or right_variance == 0:
        return None
    return covariance / ((left_variance * right_variance) ** 0.5)


def _gr_performance(summary: Mapping[str, Any]) -> dict[str, Any]:
    requests = int(summary.get("responses", 0) or 0)
    wall_ms = optional_float(summary.get("wall_ms_median", summary.get("wall_ms")))
    decode_ms = optional_float(summary.get("decode_ms_median"))
    prefill_ms = optional_float(summary.get("prefill_ms_median"))
    return {
        "wall_ms_median": wall_ms,
        "qps": _qps(requests, wall_ms),
        "decode_ms_median": decode_ms,
        "prefill_ms_median": prefill_ms,
        "decode_profile_aggregate": summary.get("decode_profile_aggregate", {}),
    }


def _sglang_performance(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "wall_ms_median": summary.get("wall_ms_median"),
        "qps_median": summary.get("qps_median"),
        "generated_tokens_per_s_median": summary.get("generated_tokens_per_s_median"),
        "beam_candidates_per_s_median": summary.get("beam_candidates_per_s_median"),
        "request_latency_ms_p50_median": summary.get("request_latency_ms_p50_median"),
        "request_latency_ms_p95_median": summary.get("request_latency_ms_p95_median"),
    }


def _qps(requests: int, wall_ms: float | None) -> float | None:
    if wall_ms is None or wall_ms <= 0:
        return None
    return requests / (wall_ms / 1000.0)


def _score_delta(left: Any, right: Any) -> float | None:
    left_float = optional_float(left)
    right_float = optional_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _mean(values) -> float | None:
    rows = [float(value) for value in values]
    if not rows:
        return None
    return sum(rows) / len(rows)


def render_markdown(report: Mapping[str, Any]) -> str:
    perf = report["performance"]
    correctness = report["correctness"]
    workload = report["workload"]
    return "\n".join(
        [
            "# SID-GR Inference vs SGLang Beam Search 对比",
            "",
            "## Workload",
            "",
            f"- model: `{workload.get('model_dir')}`",
            f"- context_len: `{workload.get('context_len')}`",
            f"- GR decode_steps: `{workload.get('gr_decode_steps')}`",
            f"- SGLang decode_steps: `{workload.get('sglang_decode_steps')}`",
            f"- GR output_token_budget: `{workload.get('gr_output_token_budget')}`",
            f"- SGLang output_token_budget: `{workload.get('sglang_output_token_budget')}`",
            f"- output_token_budget_match: `{workload.get('output_token_budget_match')}`",
            f"- beam_width: `{workload.get('beam_width')}`",
            f"- GR serving_mode: `{workload.get('gr_serving_mode')}`",
            f"- SGLang arrival_mode: `{workload.get('sglang_arrival_mode')}`",
            f"- GR arrival_stagger_ticks: `{workload.get('gr_arrival_stagger_ticks')}`",
            f"- SGLang arrival_stagger_ms: `{workload.get('sglang_arrival_stagger_ms')}`",
            f"- SGLang arrival_burst_size: `{workload.get('sglang_arrival_burst_size')}`",
            f"- matched_requests: `{workload.get('matched_requests')}`",
            f"- match_strategy: `{workload.get('match_strategy')}`",
            "",
            "## Performance",
            "",
            f"- GR wall_ms_median: `{perf['gr'].get('wall_ms_median')}`",
            f"- GR qps: `{perf['gr'].get('qps')}`",
            f"- SGLang wall_ms_median: `{perf['sglang'].get('wall_ms_median')}`",
            f"- SGLang qps_median: `{perf['sglang'].get('qps_median')}`",
            f"- SGLang request_latency_ms_p50_median: `{perf['sglang'].get('request_latency_ms_p50_median')}`",
            f"- SGLang request_latency_ms_p95_median: `{perf['sglang'].get('request_latency_ms_p95_median')}`",
            "",
            "## Correctness Against SGLang",
            "",
            f"- Top1 exact match rate: `{correctness.get('top1_exact_match_rate')}`",
            f"- TopK set overlap mean: `{correctness.get('topk_set_overlap_mean')}`",
            f"- Ordered prefix match mean: `{correctness.get('ordered_prefix_match_mean')}`",
            f"- Token length match rate: `{correctness.get('token_length_match_rate')}`",
            *[
                (
                    "- "
                    f"{row.get('gr_request_id')} vs {row.get('sglang_request_id')}: "
                    f"top1={row.get('top1_exact_match')}, "
                    f"topk_overlap={row.get('topk_set_overlap')}, "
                    f"gr_token_len={row.get('gr_token_len')}, "
                    f"sglang_token_len={row.get('sglang_token_len')}, "
                    f"top1_score_delta={row.get('top1_score_delta')}, "
                    f"gr_top1_rank_in_sglang={row.get('gr_top1_rank_in_sglang')}, "
                    f"sglang_top1_rank_in_gr={row.get('sglang_top1_rank_in_gr')}, "
                    f"same_position_count={row.get('same_position_count')}, "
                    f"common_prefix_count={row.get('common_prefix_count')}"
                )
                for row in correctness.get("requests", ())
            ],
            "",
            "## Rank Movement Summary",
            "",
            *[
                (
                    "- "
                    f"{row.get('gr_request_id')}: "
                    f"{row.get('rank_delta_summary')}"
                )
                for row in correctness.get("requests", ())
            ],
            "",
            "## Top1 Cross Scores",
            "",
            *[
                (
                    "- "
                    f"{row.get('gr_request_id')}: "
                    f"GR top1={row.get('gr_top1')} score={row.get('gr_top1_score')} "
                    f"SGLang-score={row.get('gr_top1_sglang_score')} "
                    f"scaled_delta={row.get('gr_top1_score_delta_sglang_scaled_by_token_len')}; "
                    f"SGLang top1={row.get('sglang_top1')} score={row.get('sglang_top1_score')} "
                    f"GR-score={row.get('sglang_top1_gr_score')} "
                    f"scaled_delta={row.get('sglang_top1_score_delta_sglang_scaled_by_token_len')}"
                )
                for row in correctness.get("requests", ())
            ],
            "",
            "## Scaled Score Delta Summary",
            "",
            *[
                (
                    "- "
                    f"{row.get('gr_request_id')}: "
                    f"GR - SGLang*token_len="
                    f"{row.get('score_delta_sglang_scaled_by_token_len_summary')}, "
                    f"corr={row.get('score_correlation_sglang_scaled_by_token_len')}"
                )
                for row in correctness.get("requests", ())
            ],
            "",
            "## Caveats",
            "",
            *[f"- {item}" for item in report.get("caveats", ())],
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr-json", required=True)
    parser.add_argument("--sglang-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gr = read_json(Path(args.gr_json))
    sglang = read_json(Path(args.sglang_json))
    report = compare(gr, sglang)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.output_markdown).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
