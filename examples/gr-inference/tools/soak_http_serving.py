"""Soak/load probe for a running GR HTTP serving endpoint."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class HTTPResult:
    status: int
    body: dict[str, Any]


@dataclass(frozen=True)
class SoakConfig:
    requests: int = 16
    submit_batch_size: int = 1
    max_inflight_requests: int | None = None
    input_len: int = 16
    decode_steps: int = 1
    beam_width: int = 128
    timeout_ticks: int | None = None
    cancel_every: int = 0
    poll_interval_s: float = 0.01
    max_polls: int = 200
    manual_tick: bool = False
    ready_sample_interval: int = 0
    progress_interval: int = 0
    request_prefix: str = "soak"
    drain_at_end: bool = False
    shutdown_at_end: bool = False
    output_details: bool = False


class UrllibHTTPClient:
    def __init__(self, base_url: str, *, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> HTTPResult:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                return HTTPResult(
                    status=int(response.status),
                    body=_decode_response_body(response.read()),
                )
        except HTTPError as exc:
            return HTTPResult(
                status=int(exc.code),
                body=_decode_response_body(exc.read()),
            )
        except URLError as exc:
            return HTTPResult(
                status=0,
                body={"error": {"code": "connection_error", "message": str(exc)}},
            )


def run_soak(client: Any, config: SoakConfig) -> dict[str, Any]:
    started = time.perf_counter()
    created_at_s = time.time()
    response_status_counts: dict[str, int] = {}
    accepted_ids: list[str] = []
    pending_ids: set[str] = set()
    responses: dict[str, dict[str, Any]] = {}
    submitted_at_s: dict[str, float] = {}
    completed_at_s: dict[str, float] = {}
    errors: list[dict[str, Any]] = []
    ready_samples: list[dict[str, Any]] = []

    health = _request(client, "GET", "/health", response_status_counts)
    initial_ready = _request(client, "GET", "/ready", response_status_counts)
    serving_config = _request(client, "GET", "/config", response_status_counts)
    build = _request(client, "GET", "/build", response_status_counts)
    ready_samples.append({"label": "initial", "body": initial_ready.body})

    next_request = 0
    while next_request < config.requests:
        if config.max_inflight_requests is not None:
            available_slots = config.max_inflight_requests - len(pending_ids)
            if available_slots <= 0:
                if config.manual_tick:
                    _request(client, "POST", "/tick", response_status_counts, {})
                _poll_pending_once(
                    client,
                    pending_ids,
                    responses,
                    response_status_counts,
                    submitted_at_s,
                    completed_at_s,
                )
                if pending_ids:
                    time.sleep(config.poll_interval_s)
                continue
            batch_size = min(
                config.submit_batch_size,
                config.requests - next_request,
                available_slots,
            )
        else:
            batch_size = min(config.submit_batch_size, config.requests - next_request)
        payloads = tuple(
            _request_payload(config, idx)
            for idx in range(next_request, next_request + batch_size)
        )
        submit_started = time.perf_counter()
        submit = _submit_payloads(client, payloads, response_status_counts)
        if submit.status == 202:
            request_ids = _accepted_request_ids(submit.body)
            accepted_ids.extend(request_ids)
            pending_ids.update(request_ids)
            for request_id in request_ids:
                submitted_at_s[request_id] = submit_started
            _cancel_configured_requests(
                client,
                config,
                request_ids=request_ids,
                response_status_counts=response_status_counts,
                pending_ids=pending_ids,
                responses=responses,
            )
        else:
            errors.append(
                {"stage": "submit", "status": submit.status, "body": submit.body}
            )
            if submit.status == 429:
                ready = _request(client, "GET", "/ready", response_status_counts)
                ready_samples.append(
                    {"label": f"overload_at_{next_request}", "body": ready.body}
                )

        next_request += batch_size
        if config.manual_tick:
            _request(client, "POST", "/tick", response_status_counts, {})
        if (
            config.ready_sample_interval
            and next_request % config.ready_sample_interval == 0
        ):
            ready = _request(client, "GET", "/ready", response_status_counts)
            ready_samples.append({"label": f"after_{next_request}", "body": ready.body})
        if config.progress_interval and next_request % config.progress_interval == 0:
            _print_progress(
                label=f"submitted_{next_request}",
                accepted=len(accepted_ids),
                completed=len(responses),
                pending=len(pending_ids),
                started=started,
            )
        _poll_pending_once(
            client,
            pending_ids,
            responses,
            response_status_counts,
            submitted_at_s,
            completed_at_s,
        )

    poll_rounds = 0
    while pending_ids and poll_rounds < config.max_polls:
        if config.manual_tick:
            _request(client, "POST", "/tick", response_status_counts, {})
        _poll_pending_once(
            client,
            pending_ids,
            responses,
            response_status_counts,
            submitted_at_s,
            completed_at_s,
        )
        if pending_ids:
            time.sleep(config.poll_interval_s)
        poll_rounds += 1
        if config.progress_interval and poll_rounds % config.progress_interval == 0:
            _print_progress(
                label=f"poll_round_{poll_rounds}",
                accepted=len(accepted_ids),
                completed=len(responses),
                pending=len(pending_ids),
                started=started,
            )

    final_ready = _request(client, "GET", "/ready", response_status_counts)
    status = _request(client, "GET", "/status", response_status_counts)
    metrics = _request(client, "GET", "/metrics", response_status_counts)
    kv_events = _request(client, "GET", "/kv/events", response_status_counts)
    ready_samples.append({"label": "final", "body": final_ready.body})

    lifecycle_result = None
    if config.drain_at_end:
        lifecycle_result = _request(
            client, "POST", "/drain", response_status_counts, {}
        )
    if config.shutdown_at_end:
        lifecycle_result = _request(
            client,
            "POST",
            "/shutdown",
            response_status_counts,
            {"max_ticks": 0, "timeout_unfinished": True},
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    outcome_counts = _response_outcome_counts(responses)
    response_diagnostics = _response_diagnostics(responses)
    latency_by_request_ms = _latency_by_request_ms(
        submitted_at_s, completed_at_s, responses
    )
    latency_summary = _latency_summary_ms(latency_by_request_ms)
    benchmark_metrics = _benchmark_metrics(
        config=config,
        elapsed_ms=elapsed_ms,
        succeeded_requests=int(outcome_counts.get("succeeded", 0) or 0),
        latency_by_request_ms=latency_by_request_ms,
    )
    slowest_requests = _slowest_requests_ms(latency_by_request_ms, limit=8)
    overload_errors = sum(1 for error in errors if error["status"] == 429)
    result: dict[str, Any] = {
        "schema_version": "gr_http_soak_v1",
        "created_at_s": created_at_s,
        "config": _config_payload(config),
        "requested": config.requests,
        "accepted": len(accepted_ids),
        "completed": len(responses),
        "pending": sorted(pending_ids),
        "errors": errors,
        "overload_errors": overload_errors,
        "outcomes": outcome_counts,
        "response_diagnostics": response_diagnostics,
        "poll_rounds": poll_rounds,
        "elapsed_ms": elapsed_ms,
        "latency_ms": latency_summary,
        "benchmark_metrics": benchmark_metrics,
        "slowest_requests_ms": slowest_requests,
        "response_status_counts": response_status_counts,
        "health": health.body,
        "initial_ready": initial_ready.body,
        "final_ready": final_ready.body,
        "ready_samples": ready_samples,
        "build": build.body,
        "serving_config": serving_config.body,
        "status": status.body,
        "metrics": metrics.body,
        "kv_events": kv_events.body,
        "lifecycle_result": lifecycle_result.body
        if lifecycle_result is not None
        else None,
    }
    if config.output_details:
        result["latency_details_ms"] = [
            {"request_id": request_id, "latency_ms": latency_ms}
            for request_id, latency_ms in sorted(latency_by_request_ms.items())
        ]
    return result


def _request(
    client: Any,
    method: str,
    path: str,
    counts: dict[str, int],
    payload: Mapping[str, Any] | None = None,
) -> HTTPResult:
    result = client.request(method, path, payload)
    counts[str(result.status)] = counts.get(str(result.status), 0) + 1
    return result


def _submit_payloads(
    client: Any,
    payloads: tuple[Mapping[str, Any], ...],
    counts: dict[str, int],
) -> HTTPResult:
    if len(payloads) == 1:
        return _request(client, "POST", "/submit", counts, payloads[0])
    return _request(
        client, "POST", "/submit_many", counts, {"requests": list(payloads)}
    )


def _request_payload(config: SoakConfig, idx: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": f"{config.request_prefix}-{idx}",
        "input_ids": list(range(1, config.input_len + 1)),
        "max_decode_steps": config.decode_steps,
        "beam_width": config.beam_width,
        "metadata": {"source": "soak_http_serving"},
    }
    if config.timeout_ticks is not None:
        payload["timeout_ticks"] = config.timeout_ticks
    return payload


def _print_progress(
    *,
    label: str,
    accepted: int,
    completed: int,
    pending: int,
    started: float,
) -> None:
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        "soak_progress "
        f"label={label} accepted={accepted} completed={completed} "
        f"pending={pending} elapsed_ms={elapsed_ms:.1f}",
        flush=True,
    )


def _accepted_request_ids(body: Mapping[str, Any]) -> tuple[str, ...]:
    if "request_id" in body:
        return (str(body["request_id"]),)
    return tuple(str(request_id) for request_id in body.get("request_ids", ()))


def _cancel_configured_requests(
    client: Any,
    config: SoakConfig,
    *,
    request_ids: tuple[str, ...],
    response_status_counts: dict[str, int],
    pending_ids: set[str],
    responses: dict[str, dict[str, Any]],
) -> None:
    if config.cancel_every <= 0:
        return
    for request_id in request_ids:
        suffix = int(request_id.rsplit("-", 1)[-1])
        if (suffix + 1) % config.cancel_every != 0:
            continue
        result = _request(
            client,
            "POST",
            "/cancel",
            response_status_counts,
            {"request_id": request_id, "reason": "soak_cancelled"},
        )
        if result.status == 200:
            responses[request_id] = result.body
            pending_ids.discard(request_id)


def _poll_pending_once(
    client: Any,
    pending_ids: set[str],
    responses: dict[str, dict[str, Any]],
    counts: dict[str, int],
    submitted_at_s: Mapping[str, float] | None = None,
    completed_at_s: dict[str, float] | None = None,
) -> None:
    for request_id in tuple(sorted(pending_ids)):
        result = _request(client, "GET", f"/poll/{request_id}", counts)
        if result.status != 200:
            continue
        if result.body.get("ready"):
            response = result.body.get("response")
            if isinstance(response, dict):
                responses[request_id] = response
                if completed_at_s is not None and request_id not in completed_at_s:
                    completed_at_s[request_id] = time.perf_counter()
            pending_ids.discard(request_id)


def _latency_by_request_ms(
    submitted_at_s: Mapping[str, float],
    completed_at_s: Mapping[str, float],
    responses: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    return {
        request_id: (completed_at_s[request_id] - submitted_at_s[request_id]) * 1000.0
        for request_id in responses
        if request_id in submitted_at_s and request_id in completed_at_s
    }


def _latency_summary_ms(
    latency_by_request_ms: Mapping[str, float],
) -> dict[str, Any]:
    latencies = list(latency_by_request_ms.values())
    latencies.sort()
    if not latencies:
        return {"count": 0}
    return {
        "count": len(latencies),
        "mean": sum(latencies) / len(latencies),
        "median": _percentile(latencies, 0.50),
        "p90": _percentile(latencies, 0.90),
        "p99": _percentile(latencies, 0.99),
        "min": latencies[0],
        "max": latencies[-1],
    }


def _slowest_requests_ms(
    latency_by_request_ms: Mapping[str, float],
    *,
    limit: int,
) -> tuple[dict[str, Any], ...]:
    slowest = sorted(
        latency_by_request_ms.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return tuple(
        {"request_id": request_id, "latency_ms": latency_ms}
        for request_id, latency_ms in slowest
    )


def _benchmark_metrics(
    *,
    config: SoakConfig,
    elapsed_ms: float,
    succeeded_requests: int,
    latency_by_request_ms: Mapping[str, float],
) -> dict[str, Any]:
    duration_s = elapsed_ms / 1000.0
    total_input_tokens = succeeded_requests * config.input_len
    # GR produces one initial beam-token plus `decode_steps` follow-up tokens,
    # while the matching SGLang command sets max_new_tokens to that effective length.
    effective_output_len = config.decode_steps + 1
    total_output_tokens = succeeded_requests * config.beam_width * effective_output_len
    total_tokens = total_input_tokens + total_output_tokens
    total_request_time_s = sum(latency_by_request_ms.values()) / 1000.0
    if duration_s <= 0:
        return {
            "duration_s": duration_s,
            "completed": succeeded_requests,
            "effective_output_len": effective_output_len,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_throughput": 0.0,
            "request_throughput": 0.0,
            "input_throughput": 0.0,
            "output_throughput": 0.0,
            "concurrency": 0.0,
        }
    return {
        "duration_s": duration_s,
        "completed": succeeded_requests,
        "effective_output_len": effective_output_len,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_throughput": total_tokens / duration_s,
        "request_throughput": succeeded_requests / duration_s,
        "input_throughput": total_input_tokens / duration_s,
        "output_throughput": total_output_tokens / duration_s,
        "concurrency": total_request_time_s / duration_s,
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _response_outcome_counts(
    responses: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    counts = {
        "succeeded": 0,
        "cancelled": 0,
        "failed": 0,
        "timed_out": 0,
    }
    for response in responses.values():
        metadata = response.get("metadata", {})
        if not isinstance(metadata, Mapping):
            counts["succeeded"] += 1
            continue
        if metadata.get("cancelled"):
            counts["cancelled"] += 1
        elif metadata.get("stop_reason") == "request_timeout":
            counts["timed_out"] += 1
            counts["failed"] += 1
        elif metadata.get("failed"):
            counts["failed"] += 1
        else:
            counts["succeeded"] += 1
    return counts


def _response_diagnostics(
    responses: Mapping[str, Mapping[str, Any]],
    *,
    failed_sample_limit: int = 8,
) -> dict[str, Any]:
    stop_reason_counts: dict[str, int] = {}
    error_type_counts: dict[str, int] = {}
    failed_samples: list[dict[str, Any]] = []
    for request_id, response in responses.items():
        metadata = response.get("metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        stop_reason = metadata.get("stop_reason")
        if stop_reason is not None:
            _increment_count(stop_reason_counts, str(stop_reason))
        error_type = metadata.get("error_type")
        if error_type is not None:
            _increment_count(error_type_counts, str(error_type))
        if metadata.get("failed") and len(failed_samples) < failed_sample_limit:
            failed_samples.append(
                {
                    "request_id": str(request_id),
                    "stop_reason": stop_reason,
                    "error_type": error_type,
                    "error_message": _truncate(metadata.get("error_message")),
                }
            )
    return {
        "stop_reason_counts": stop_reason_counts,
        "error_type_counts": error_type_counts,
        "failed_samples": tuple(failed_samples),
    }


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _truncate(value: Any, *, limit: int = 512) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _config_payload(config: SoakConfig) -> dict[str, Any]:
    return {
        "requests": config.requests,
        "submit_batch_size": config.submit_batch_size,
        "max_inflight_requests": config.max_inflight_requests,
        "input_len": config.input_len,
        "decode_steps": config.decode_steps,
        "beam_width": config.beam_width,
        "timeout_ticks": config.timeout_ticks,
        "cancel_every": config.cancel_every,
        "poll_interval_s": config.poll_interval_s,
        "max_polls": config.max_polls,
        "manual_tick": config.manual_tick,
        "ready_sample_interval": config.ready_sample_interval,
        "progress_interval": config.progress_interval,
        "request_prefix": config.request_prefix,
        "drain_at_end": config.drain_at_end,
        "shutdown_at_end": config.shutdown_at_end,
        "output_details": config.output_details,
    }


def _decode_response_body(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw.decode("utf-8"))
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--submit-batch-size", type=int, default=1)
    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        help="Limit accepted-but-not-completed requests, matching max-concurrency style clients.",
    )
    parser.add_argument("--input-len", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--timeout-ticks", type=int)
    parser.add_argument("--cancel-every", type=int, default=0)
    parser.add_argument("--poll-interval-s", type=float, default=0.01)
    parser.add_argument("--max-polls", type=int, default=200)
    parser.add_argument("--manual-tick", action="store_true")
    parser.add_argument("--ready-sample-interval", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--request-prefix", default="soak")
    parser.add_argument("--drain-at-end", action="store_true")
    parser.add_argument("--shutdown-at-end", action="store_true")
    parser.add_argument("--output-details", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--output-json")
    return parser


def _config_from_args(args: argparse.Namespace) -> SoakConfig:
    if args.requests <= 0:
        raise ValueError("--requests must be positive")
    if args.submit_batch_size <= 0:
        raise ValueError("--submit-batch-size must be positive")
    max_inflight_requests = getattr(args, "max_inflight_requests", None)
    if max_inflight_requests is not None and max_inflight_requests <= 0:
        raise ValueError("--max-inflight-requests must be positive")
    if args.input_len <= 0:
        raise ValueError("--input-len must be positive")
    if args.decode_steps <= 0:
        raise ValueError("--decode-steps must be positive")
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be positive")
    return SoakConfig(
        requests=args.requests,
        submit_batch_size=args.submit_batch_size,
        max_inflight_requests=max_inflight_requests,
        input_len=args.input_len,
        decode_steps=args.decode_steps,
        beam_width=args.beam_width,
        timeout_ticks=args.timeout_ticks,
        cancel_every=args.cancel_every,
        poll_interval_s=args.poll_interval_s,
        max_polls=args.max_polls,
        manual_tick=args.manual_tick,
        ready_sample_interval=args.ready_sample_interval,
        progress_interval=args.progress_interval,
        request_prefix=args.request_prefix,
        drain_at_end=args.drain_at_end,
        shutdown_at_end=args.shutdown_at_end,
        output_details=getattr(args, "output_details", False),
    )


def main() -> None:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    summary = run_soak(
        UrllibHTTPClient(args.base_url, timeout_s=args.timeout_s), config
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
