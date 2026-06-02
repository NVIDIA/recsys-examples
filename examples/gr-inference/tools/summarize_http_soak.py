"""Summarize and validate GR HTTP soak JSON snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tool_utils import read_json


def summarize_soak(
    summary: Mapping[str, Any],
    *,
    expected_requests: int | None = None,
    expected_cancelled: int | None = None,
) -> dict[str, Any]:
    requested = int(summary.get("requested", 0) or 0)
    accepted = int(summary.get("accepted", 0) or 0)
    completed = int(summary.get("completed", 0) or 0)
    pending = tuple(summary.get("pending", ()) or ())
    outcomes = _mapping(summary.get("outcomes"))
    metrics = _mapping(summary.get("metrics"))
    benchmark_metrics = _mapping(summary.get("benchmark_metrics"))
    response_diagnostics = _mapping(summary.get("response_diagnostics"))
    latency_ms = _mapping(summary.get("latency_ms"))

    failures = []
    target_requests = expected_requests if expected_requests is not None else requested
    if requested != target_requests:
        failures.append(f"requested={requested} expected={target_requests}")
    if accepted != requested:
        failures.append(f"accepted={accepted} requested={requested}")
    if completed != accepted:
        failures.append(f"completed={completed} accepted={accepted}")
    if pending:
        failures.append(f"pending={len(pending)}")
    if int(summary.get("overload_errors", 0) or 0):
        failures.append(f"overload_errors={summary.get('overload_errors')}")
    if int(outcomes.get("failed", 0) or 0):
        failures.append(f"failed={outcomes.get('failed')}")
    if int(outcomes.get("timed_out", 0) or 0):
        failures.append(f"timed_out={outcomes.get('timed_out')}")
    if (
        expected_cancelled is not None
        and int(outcomes.get("cancelled", 0) or 0) != expected_cancelled
    ):
        failures.append(
            f"cancelled={outcomes.get('cancelled')} expected={expected_cancelled}"
        )
    if int(metrics.get("kv_health_kv_allocator_leak_detected", 0) or 0):
        failures.append("kv_allocator_leak_detected=1")
    if int(metrics.get("beam_kv_pool_health_beam_kv_pool_leak_detected", 0) or 0):
        failures.append("beam_kv_pool_leak_detected=1")
    if int(metrics.get("worker_errors", 0) or 0):
        failures.append(f"worker_errors={metrics.get('worker_errors')}")

    return {
        "schema_version": summary.get("schema_version"),
        "request_prefix": _mapping(summary.get("config")).get("request_prefix"),
        "requested": requested,
        "accepted": accepted,
        "completed": completed,
        "pending": len(pending),
        "outcomes": dict(outcomes),
        "response_diagnostics": {
            "stop_reason_counts": dict(
                _mapping(response_diagnostics.get("stop_reason_counts"))
            ),
            "error_type_counts": dict(
                _mapping(response_diagnostics.get("error_type_counts"))
            ),
            "failed_samples": tuple(
                response_diagnostics.get("failed_samples", ()) or ()
            ),
        },
        "latency_ms": dict(latency_ms),
        "benchmark_metrics": dict(benchmark_metrics),
        "slowest_requests_ms": tuple(summary.get("slowest_requests_ms", ()) or ()),
        "overload_errors": int(summary.get("overload_errors", 0) or 0),
        "decode_ms_total": metrics.get("decode_ms"),
        "prefill_ms_total": metrics.get("prefill_ms"),
        "beam_pool_max_used": metrics.get("beam_kv_pool_max_used"),
        "kv_leak": metrics.get("kv_health_kv_allocator_leak_detected"),
        "beam_pool_leak": metrics.get("beam_kv_pool_health_beam_kv_pool_leak_detected"),
        "worker_errors": metrics.get("worker_errors"),
        "passed": not failures,
        "failures": tuple(failures),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a gr_http_soak_v1 JSON summary")
    parser.add_argument("--expected-requests", type=int)
    parser.add_argument("--expected-cancelled", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = read_json(Path(args.path))
    report = summarize_soak(
        summary,
        expected_requests=args.expected_requests,
        expected_cancelled=args.expected_cancelled,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if args.fail_on_error and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
