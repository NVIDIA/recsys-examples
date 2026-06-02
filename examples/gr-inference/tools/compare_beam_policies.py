"""Compare fixed and dynamic beam policies on the same inputs.

This is a quality/latency probe, not a production benchmark. It keeps the input
tokens fixed across policies and reports final-beam overlap plus score deltas so
dynamic beam changes are visible instead of treated as a pure speed optimization.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tool_utils import bootstrap_repo_paths, json_dumps, jsonable, write_json

bootstrap_repo_paths(__file__, include_tools=True)

from gr_inference import (  # noqa: E402
    GRContinuousBatchingPolicy,
    GRContinuousScheduler,
    GRContinuousServingExecutor,
    GRDecodeAttention,
    GRDecodeEngine,
    GRDenseBeamKVPool,
    GRServingConfig,
    GRServingEngine,
    GRServingRequest,
    ScheduledBeamPolicy,
    ScoreMarginBeamPolicy,
)
from run_qwen3_real_weight_serving import (  # noqa: E402
    choose_dtype,
    load_model,
    make_decode_backend,
)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    make_policy: Callable[[], Any | None]


def parse_schedule(value: str) -> dict[int, int]:
    schedule: dict[int, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        step, width = part.split(":", 1)
        schedule[int(step.strip())] = int(width.strip())
    if 0 not in schedule:
        raise ValueError("schedule must include step 0")
    return schedule


def make_policy_specs(args) -> list[PolicySpec]:
    specs = [PolicySpec("fixed", lambda: None)]
    for schedule_text in args.schedules:
        schedule = parse_schedule(schedule_text)
        specs.append(
            PolicySpec(
                f"scheduled:{schedule_text}",
                lambda schedule=schedule: ScheduledBeamPolicy(dict(schedule)),
            )
        )
    for min_width in args.score_margin_min_widths:
        for margin in args.score_margins:
            specs.append(
                PolicySpec(
                    f"score_margin:{margin},min_width:{min_width}",
                    lambda margin=margin, min_width=min_width: ScoreMarginBeamPolicy(
                        max_beam_width=args.beam_width,
                        score_margin=margin,
                        min_beam_width=min_width,
                    ),
                )
            )
    return specs


def make_engine(args, torch, config, device: str, model) -> GRServingEngine:
    decode_backend_args = SimpleNamespace(
        decode_backend=args.decode_backend,
        batched_decode=True,
    )
    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=make_decode_backend(decode_backend_args, device)),
        fixed_beam_width=args.beam_width,
    )
    return GRServingEngine(
        model=model,
        decode_engine=decode_engine,
        config=GRServingConfig(
            max_decode_steps=args.decode_steps,
            max_beam_width=args.beam_width,
            enable_batched_decode=True,
            return_beam_details=True,
            beam_score_mode=args.beam_score_mode,
        ),
    )


def make_beam_kv_pool(args, torch, config, device: str) -> GRDenseBeamKVPool:
    dtype = choose_dtype(torch, device)
    shape = (
        config.num_layers,
        1,
        args.decode_steps,
        args.beam_width,
        config.num_kv_heads,
        config.head_dim,
    )
    return GRDenseBeamKVPool(
        key=torch.empty(shape, device=device, dtype=dtype),
        value=torch.empty(shape, device=device, dtype=dtype),
    )


def run_policy(
    *,
    args,
    torch,
    config,
    device: str,
    engine: GRServingEngine,
    input_ids,
    spec: PolicySpec,
    case_idx: int,
) -> dict[str, Any]:
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=1,
            max_decode_batch_size=1,
            max_running_requests=1,
        )
    )
    executor = GRContinuousServingExecutor(
        engine=engine,
        scheduler=scheduler,
        synchronize=torch.cuda.synchronize if device == "cuda" else None,
        beam_kv_pool=make_beam_kv_pool(args, torch, config, device),
    )
    request = GRServingRequest(
        request_id=f"case-{case_idx}-{spec.name}",
        input_ids=input_ids.clone(),
        max_decode_steps=args.decode_steps,
        beam_width=args.beam_width,
        beam_width_policy=spec.make_policy(),
    )
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    executor.submit(request)
    responses = executor.run_until_empty()
    if device == "cuda":
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000.0
    if len(responses) != 1:
        raise RuntimeError(f"expected one response, got {len(responses)}")
    response = responses[0]
    return {
        "policy": spec.name,
        "wall_ms": wall_ms,
        "items": [list(item) for item in _final_items(response)],
        "token_ids": list(response.token_ids),
        "scores": list(response.scores),
        "metadata": jsonable(response.metadata),
        "scheduler_metrics": jsonable(executor.metrics()),
    }


def compare_to_fixed(fixed: dict[str, Any], other: dict[str, Any], *, compare_top_k: int) -> dict[str, Any]:
    fixed_items = [tuple(item) for item in fixed["items"]]
    other_items = [tuple(item) for item in other["items"]]
    fixed_scores = _final_scores(fixed)
    other_scores = _final_scores(other)
    k = min(compare_top_k, len(fixed_items), len(other_items))
    fixed_top = fixed_items[:k]
    other_top = other_items[:k]
    overlap_items = set(fixed_top).intersection(other_top)
    overlap = len(overlap_items)
    score_deltas = [
        other_scores[idx] - fixed_scores[idx]
        for idx in range(min(k, len(fixed_scores), len(other_scores)))
    ]
    fixed_score_by_item = {
        item: fixed_scores[idx]
        for idx, item in enumerate(fixed_items[:k])
        if idx < len(fixed_scores)
    }
    other_score_by_item = {
        item: other_scores[idx]
        for idx, item in enumerate(other_items[:k])
        if idx < len(other_scores)
    }
    matched_score_deltas = [
        other_score_by_item[item] - fixed_score_by_item[item]
        for item in overlap_items
        if item in fixed_score_by_item and item in other_score_by_item
    ]
    fixed_wall = float(fixed["wall_ms"])
    other_wall = float(other["wall_ms"])
    fixed_decode_ms = _metadata_float(fixed, "decode_ms")
    other_decode_ms = _metadata_float(other, "decode_ms")
    top10_k = min(10, len(fixed_items), len(other_items))
    return {
        "policy": other["policy"],
        "compared_top_k": k,
        "top1_match": bool(fixed_items and other_items and fixed_items[0] == other_items[0]),
        "topk_overlap_count": overlap,
        "topk_overlap_ratio": overlap / k if k else None,
        "top10_changed": bool(
            top10_k and set(fixed_items[:top10_k]) != set(other_items[:top10_k])
        ),
        "fixed_top10_missing_count": (
            top10_k - len(set(fixed_items[:top10_k]).intersection(other_items[:top10_k]))
            if top10_k
            else None
        ),
        "mean_rank_score_delta": sum(score_deltas) / len(score_deltas) if score_deltas else None,
        "max_abs_rank_score_delta": max((abs(delta) for delta in score_deltas), default=None),
        "mean_matched_item_score_delta": (
            sum(matched_score_deltas) / len(matched_score_deltas)
            if matched_score_deltas
            else None
        ),
        "max_abs_matched_item_score_delta": max(
            (abs(delta) for delta in matched_score_deltas),
            default=None,
        ),
        "fixed_wall_ms": fixed_wall,
        "policy_wall_ms": other_wall,
        "latency_improvement_pct": (
            (fixed_wall - other_wall) / fixed_wall * 100.0 if fixed_wall else None
        ),
        "fixed_decode_ms": fixed_decode_ms,
        "policy_decode_ms": other_decode_ms,
        "decode_latency_improvement_pct": (
            (fixed_decode_ms - other_decode_ms) / fixed_decode_ms * 100.0
            if fixed_decode_ms
            else None
        ),
        "fixed_final_width": len(fixed_items),
        "policy_final_width": len(other_items),
    }


def _final_items(response) -> list[tuple[int, ...]]:
    beam_details = response.metadata.get("beam_details")
    if beam_details:
        return [tuple(int(token) for token in detail["token_ids"]) for detail in beam_details]
    return [(int(token),) for token in response.token_ids]


def _final_scores(result: dict[str, Any]) -> list[float]:
    beam_details = result["metadata"].get("beam_details")
    if beam_details:
        return [float(detail["cumulative_score"]) for detail in beam_details]
    return [float(score) for score in result["scores"]]


def _metadata_float(result: dict[str, Any], key: str) -> float | None:
    value = result["metadata"].get(key)
    if value is None:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--context-len", type=int, default=4700)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument(
        "--schedule",
        action="append",
        dest="schedules",
        help=(
            "Step:width schedule. Can be repeated, for example "
            "--schedule 0:256,1:192,2:128 --schedule 0:256,1:192,2:192."
        ),
    )
    parser.add_argument("--score-margins", type=float, nargs="*", default=[1.0, 2.0])
    parser.add_argument("--score-margin-min-width", type=int, default=8)
    parser.add_argument(
        "--score-margin-min-widths",
        type=int,
        nargs="*",
        help="Optional list of min widths to sweep for score-margin policies.",
    )
    parser.add_argument("--warmup-cases", type=int, default=1)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compare-top-k", type=int, default=10)
    parser.add_argument("--decode-backend", choices=["fake", "real"], default="real")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--beam-score-mode", choices=["raw_logits", "logprob"], default="logprob")
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    parser.add_argument("--min-top1-match-rate", type=float, default=1.0)
    parser.add_argument("--min-topk-overlap-ratio", type=float, default=0.9)
    parser.add_argument("--max-top10-changed-rate", type=float, default=0.25)
    parser.add_argument("--min-decode-latency-improvement-pct", type=float, default=0.0)
    parser.add_argument(
        "--fail-on-quality-gate",
        action="store_true",
        help="Exit non-zero when no dynamic policy passes the configured quality gate.",
    )
    args = parser.parse_args()
    if args.warmup_cases < 0:
        raise ValueError("--warmup-cases must be non-negative")
    if args.cases <= 0:
        raise ValueError("--cases must be positive")
    args.schedules = args.schedules or ["0:256,1:192,2:192"]
    args.score_margin_min_widths = (
        args.score_margin_min_widths
        if args.score_margin_min_widths
        else [args.score_margin_min_width]
    )
    for min_width in args.score_margin_min_widths:
        if min_width <= 0 or min_width > args.beam_width:
            raise ValueError(
                "--score-margin-min-widths values must be in (0, beam_width]"
            )

    import torch

    model_args = SimpleNamespace(
        model_dir=args.model_dir,
        context_len=args.context_len,
        decode_steps=args.decode_steps,
        beam_width=args.beam_width,
        device=args.device,
    )
    model, config, device = load_model(model_args, torch)
    engine = make_engine(args, torch, config, device, model)
    policy_specs = make_policy_specs(args)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    for warmup_idx in range(args.warmup_cases):
        input_ids = make_input_ids(args, torch, config, device, generator)
        for spec in policy_specs:
            run_policy(
                args=args,
                torch=torch,
                config=config,
                device=device,
                engine=engine,
                input_ids=input_ids,
                spec=spec,
                case_idx=-(warmup_idx + 1),
            )

    cases: list[dict[str, Any]] = []
    for case_idx in range(args.cases):
        input_ids = make_input_ids(args, torch, config, device, generator)
        policy_results = [
            run_policy(
                args=args,
                torch=torch,
                config=config,
                device=device,
                engine=engine,
                input_ids=input_ids,
                spec=spec,
                case_idx=case_idx,
            )
            for spec in policy_specs
        ]
        fixed = policy_results[0]
        comparisons = [
            compare_to_fixed(fixed, result, compare_top_k=args.compare_top_k)
            for result in policy_results[1:]
        ]
        cases.append(
            {
                "case_idx": case_idx,
                "policies": policy_results,
                "comparisons_to_fixed": comparisons,
            }
        )

    aggregate = aggregate_comparisons(cases)
    quality_gate = evaluate_quality_gate(aggregate, args)
    ranked_policies = rank_policies(aggregate, quality_gate)
    summary = {
        "context_len": args.context_len,
        "decode_steps": args.decode_steps,
        "beam_width": args.beam_width,
        "beam_score_mode": args.beam_score_mode,
        "compare_top_k": args.compare_top_k,
        "schedules": args.schedules,
        "score_margins": args.score_margins,
        "score_margin_min_widths": args.score_margin_min_widths,
        "cases": cases,
        "aggregate": aggregate,
        "quality_gate": quality_gate,
        "ranked_policies": ranked_policies,
    }
    print(
        json_dumps(
            {
                "aggregate": aggregate,
                "quality_gate": quality_gate,
                "ranked_policies": ranked_policies,
            },
        )
    )
    if args.output_json:
        write_json(args.output_json, summary, trailing_newline=False)
    if args.output_markdown:
        output_path = Path(args.output_markdown)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown_report(summary), encoding="utf-8")
    if args.fail_on_quality_gate and not any(
        policy_gate["passes"] for policy_gate in quality_gate["policies"].values()
    ):
        raise SystemExit(2)


def make_input_ids(args, torch, config, device: str, generator):
    return torch.randint(
        0,
        config.vocab_size,
        (1, args.context_len),
        device=device,
        generator=generator,
    )


def aggregate_comparisons(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        for comparison in case["comparisons_to_fixed"]:
            by_policy.setdefault(comparison["policy"], []).append(comparison)
    aggregate: dict[str, Any] = {}
    for policy, rows in by_policy.items():
        aggregate[policy] = {
            "cases": len(rows),
            "top1_match_rate": _mean(row["top1_match"] for row in rows),
            "topk_overlap_ratio_mean": _mean(row["topk_overlap_ratio"] for row in rows),
            "top10_changed_rate": _mean(row["top10_changed"] for row in rows),
            "latency_improvement_pct_mean": _mean(row["latency_improvement_pct"] for row in rows),
            "decode_latency_improvement_pct_mean": _mean(
                row["decode_latency_improvement_pct"] for row in rows
            ),
            "mean_rank_score_delta_mean": _mean(row["mean_rank_score_delta"] for row in rows),
            "max_abs_rank_score_delta_max": max(
                (
                    row["max_abs_rank_score_delta"]
                    for row in rows
                    if row["max_abs_rank_score_delta"] is not None
                ),
                default=None,
            ),
            "mean_matched_item_score_delta_mean": _mean(
                row["mean_matched_item_score_delta"] for row in rows
            ),
            "max_abs_matched_item_score_delta_max": max(
                (
                    row["max_abs_matched_item_score_delta"]
                    for row in rows
                    if row["max_abs_matched_item_score_delta"] is not None
                ),
                default=None,
            ),
            "policy_final_width_mean": _mean(row["policy_final_width"] for row in rows),
        }
    return aggregate


def evaluate_quality_gate(aggregate: dict[str, Any], args) -> dict[str, Any]:
    thresholds = {
        "min_top1_match_rate": args.min_top1_match_rate,
        "min_topk_overlap_ratio": args.min_topk_overlap_ratio,
        "max_top10_changed_rate": args.max_top10_changed_rate,
        "min_decode_latency_improvement_pct": args.min_decode_latency_improvement_pct,
    }
    policies: dict[str, Any] = {}
    for policy, metrics in aggregate.items():
        failures: list[str] = []
        if _lt(metrics.get("top1_match_rate"), args.min_top1_match_rate):
            failures.append("top1_match_rate")
        if _lt(metrics.get("topk_overlap_ratio_mean"), args.min_topk_overlap_ratio):
            failures.append("topk_overlap_ratio_mean")
        if _gt(metrics.get("top10_changed_rate"), args.max_top10_changed_rate):
            failures.append("top10_changed_rate")
        if _lt(
            metrics.get("decode_latency_improvement_pct_mean"),
            args.min_decode_latency_improvement_pct,
        ):
            failures.append("decode_latency_improvement_pct_mean")
        policies[policy] = {
            "passes": not failures,
            "label": _quality_label(metrics, failures),
            "failures": failures,
        }
    return {
        "thresholds": thresholds,
        "policies": policies,
        "passing_policies": [
            policy for policy, gate in policies.items() if gate["passes"]
        ],
    }


def rank_policies(
    aggregate: dict[str, Any],
    quality_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for policy, metrics in aggregate.items():
        gate = quality_gate["policies"][policy]
        rows.append(
            {
                "policy": policy,
                "passes": gate["passes"],
                "label": gate["label"],
                "top1_match_rate": metrics.get("top1_match_rate"),
                "topk_overlap_ratio_mean": metrics.get("topk_overlap_ratio_mean"),
                "top10_changed_rate": metrics.get("top10_changed_rate"),
                "decode_latency_improvement_pct_mean": metrics.get(
                    "decode_latency_improvement_pct_mean"
                ),
                "policy_final_width_mean": metrics.get("policy_final_width_mean"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            bool(row["passes"]),
            _float_or_default(row["top1_match_rate"], -1.0),
            _float_or_default(row["topk_overlap_ratio_mean"], -1.0),
            -_float_or_default(row["top10_changed_rate"], 1.0),
            _float_or_default(row["decode_latency_improvement_pct_mean"], -1.0),
        ),
        reverse=True,
    )


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Dynamic Beam Quality Report",
        "",
        "## Workload",
        "",
        f"- context_len: `{summary['context_len']}`",
        f"- decode_steps: `{summary['decode_steps']}`",
        f"- beam_width: `{summary['beam_width']}`",
        f"- beam_score_mode: `{summary['beam_score_mode']}`",
        f"- compare_top_k: `{summary['compare_top_k']}`",
        f"- schedules: `{summary['schedules']}`",
        f"- score_margins: `{summary['score_margins']}`",
        f"- score_margin_min_widths: `{summary['score_margin_min_widths']}`",
        "",
        "## Aggregate",
        "",
        "| policy | gate | top1 | topK overlap | top10 changed | decode speedup | final width |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    quality_gate = summary["quality_gate"]["policies"]
    for policy, metrics in summary["aggregate"].items():
        gate = quality_gate[policy]
        lines.append(
            "| {policy} | {label} | {top1} | {overlap} | {top10} | {speedup} | {width} |".format(
                policy=policy,
                label=gate["label"],
                top1=_fmt_rate(metrics.get("top1_match_rate")),
                overlap=_fmt_rate(metrics.get("topk_overlap_ratio_mean")),
                top10=_fmt_rate(metrics.get("top10_changed_rate")),
                speedup=_fmt_pct(metrics.get("decode_latency_improvement_pct_mean")),
                width=_fmt_num(metrics.get("policy_final_width_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Ranked Policies",
            "",
            "| rank | policy | gate | top1 | topK overlap | top10 changed | decode speedup | final width |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(summary["ranked_policies"], start=1):
        lines.append(
            "| {rank} | {policy} | {label} | {top1} | {overlap} | {top10} | {speedup} | {width} |".format(
                rank=rank,
                policy=row["policy"],
                label=row["label"],
                top1=_fmt_rate(row["top1_match_rate"]),
                overlap=_fmt_rate(row["topk_overlap_ratio_mean"]),
                top10=_fmt_rate(row["top10_changed_rate"]),
                speedup=_fmt_pct(row["decode_latency_improvement_pct_mean"]),
                width=_fmt_num(row["policy_final_width_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Quality Gate",
            "",
            "Thresholds are advisory unless `--fail-on-quality-gate` is enabled.",
            "",
        ]
    )
    for policy, gate in quality_gate.items():
        failure_text = ", ".join(gate["failures"]) if gate["failures"] else "none"
        lines.append(
            f"- `{policy}`: `{gate['label']}`, passes=`{gate['passes']}`, failures=`{failure_text}`"
        )
    lines.append("")
    return "\n".join(lines)


def _quality_label(metrics: dict[str, Any], failures: list[str]) -> str:
    if not failures:
        return "candidate"
    if failures == ["top10_changed_rate"]:
        return "top1_preserved_top10_changed"
    if "top1_match_rate" in failures:
        return "top1_regression"
    if "decode_latency_improvement_pct_mean" in failures:
        return "no_latency_win"
    return "quality_risk"


def _lt(value: Any, threshold: float) -> bool:
    return value is None or float(value) < threshold


def _gt(value: Any, threshold: float) -> bool:
    return value is None or float(value) > threshold


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def _fmt_num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _mean(values) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


if __name__ == "__main__":
    main()
