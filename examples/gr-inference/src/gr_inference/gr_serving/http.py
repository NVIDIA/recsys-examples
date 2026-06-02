"""Minimal HTTP adapter for the in-process serving facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
import json
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
import uuid

from gr_inference.gr_runtime import logits_processors_from_specs
from gr_inference.gr_scheduler import ScheduledBeamPolicy, ScoreMarginBeamPolicy
from gr_inference.gr_serving.api import GRInProcessServingFacade
from gr_inference.gr_serving.beam_metadata import normalized_beam_results_from_metadata
from gr_inference.gr_serving.payload import (
    optional_int as _optional_int,
    payload_list as _payload_list,
    required_field as _required_field,
    required_int as _required_int,
    required_str as _required_str,
)
from gr_inference.gr_serving.request import GRServingRequest, GRServingResponse


RequestFactory = Callable[[Mapping[str, Any]], GRServingRequest]
LogSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class GRHTTPResponse:
    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GRHTTPValidationPolicy:
    """Request limits for the minimal HTTP adapter."""

    max_request_bytes: int | None = 1 << 20
    max_context_len: int | None = None
    max_decode_steps: int | None = None
    max_beam_width: int | None = None
    max_submit_many: int | None = 32
    max_waiting_requests: int | None = None
    max_timeout_ticks: int | None = None
    allow_manual_tick: bool = True
    allow_catalog_reload: bool = True

    def validate(self) -> None:
        for name, value in (
            ("max_request_bytes", self.max_request_bytes),
            ("max_context_len", self.max_context_len),
            ("max_decode_steps", self.max_decode_steps),
            ("max_beam_width", self.max_beam_width),
            ("max_submit_many", self.max_submit_many),
            ("max_waiting_requests", self.max_waiting_requests),
            ("max_timeout_ticks", self.max_timeout_ticks),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")


class GRHTTPAdapterError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        code: str = "http_error",
        retryable: bool = False,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.retryable = retryable
        self.headers = dict(headers or {})


@dataclass
class GRHTTPServingAdapter:
    """Transport adapter over ``GRInProcessServingFacade``.

    The adapter is intentionally framework-free. Tests and future HTTP/gRPC
    wrappers can exercise the same route logic without binding sockets.
    """

    facade: GRInProcessServingFacade
    request_factory: RequestFactory | None = None
    validation_policy: GRHTTPValidationPolicy = field(default_factory=GRHTTPValidationPolicy)
    api_key: str | None = None
    api_key_header: str = "X-GR-API-Key"
    build_info: Mapping[str, Any] = field(default_factory=lambda: _default_build_info())
    enable_request_logging: bool = False
    log_requests_level: str = "summary"
    request_log_sink: LogSink | None = None

    def __post_init__(self) -> None:
        self.validation_policy.validate()

    def handle(
        self,
        method: str,
        path: str,
        body: bytes | str | Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> GRHTTPResponse:
        started = time.perf_counter()
        method = method.upper()
        route = _route_path(path)
        response: GRHTTPResponse
        try:
            self._validate_auth(method, route, headers or {})
            self._validate_body_size(body)
            payload = _json_payload(body)
            response = self._dispatch(method, route, payload)
        except GRHTTPAdapterError as exc:
            response = _error_response(
                exc.status,
                exc.message,
                code=exc.code,
                retryable=exc.retryable,
                headers=exc.headers,
            )
        except ValueError as exc:
            response = _error_response(400, str(exc), code="validation_error")
        except KeyError as exc:
            response = _error_response(404, str(exc), code="not_found")
        except RuntimeError as exc:
            response = _runtime_error_response(exc)
        self._emit_request_log(method, path, route, response, started)
        return response

    def serve(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> ThreadingHTTPServer:
        handler_cls = make_http_handler(self)
        server = ThreadingHTTPServer((host, port), handler_cls)
        return server

    def _dispatch(
        self,
        method: str,
        route: tuple[str, ...],
        payload: Mapping[str, Any],
    ) -> GRHTTPResponse:
        if method == "GET" and route == ("health",):
            return _ok({"ok": True})
        if method == "GET" and route == ("ready",):
            return _ok(self._readiness_payload())
        if method == "GET" and route == ("config",):
            return _ok(self._config_payload())
        if method == "GET" and route == ("build",):
            return _ok({"build": dict(self.build_info)})
        if method == "GET" and route == ("v1", "models"):
            model_id = (
                self.build_info.get("model_name")
                or self.build_info.get("model_dir")
                or "gr-inference"
            )
            return _ok(
                {
                    "object": "list",
                    "data": (
                        {
                            "id": model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "gr-inference",
                        },
                    ),
                }
            )
        if method == "GET" and route == ("status",):
            return _ok(self.facade.status())
        if method == "GET" and route == ("metrics",):
            return _ok(self.facade.metrics())
        if method == "GET" and route == ("metrics", "prometheus"):
            return _text(
                _prometheus_metrics_text(self.facade.metrics()),
                content_type="text/plain; version=0.0.4; charset=utf-8",
            )
        if method == "GET" and route == ("requests",):
            return _ok({"requests": self.facade.request_statuses()})
        if method == "GET" and len(route) == 2 and route[0] == "requests":
            return _ok({"request": self.facade.request_status(route[1])})
        if method == "GET" and route == ("catalog", "status"):
            return _ok({"catalog": self.facade.catalog_status()})
        if method == "GET" and route == ("kv", "events"):
            return _ok(_kv_events_payload(self.facade.status()))
        if method == "POST" and route == ("submit",):
            self._validate_admission(1)
            request = self._make_request(payload)
            self._validate_request(request)
            request_id = self.facade.submit(request)
            return _ok({"request_id": request_id}, status=202)
        if method == "POST" and route == ("submit_many",):
            rows = _payload_list(payload, "requests")
            if (
                self.validation_policy.max_submit_many is not None
                and len(rows) > self.validation_policy.max_submit_many
            ):
                raise GRHTTPAdapterError(
                    413,
                    f"requests exceeds max_submit_many={self.validation_policy.max_submit_many}",
                    code="too_many_requests",
                )
            self._validate_admission(len(rows))
            requests = tuple(self._make_request(row) for row in rows)
            for request in requests:
                self._validate_request(request)
            return _ok({"request_ids": self.facade.submit_many(requests)}, status=202)
        if method == "POST" and route == ("generate",):
            return self._handle_sglang_generate(payload)
        if method == "POST" and route == ("tick",):
            if not self.validation_policy.allow_manual_tick:
                raise GRHTTPAdapterError(
                    403,
                    "manual tick is disabled",
                    code="route_disabled",
                )
            return _ok(_result_payload(self.facade.tick()))
        if method == "POST" and route == ("run_until_idle",):
            if not self.validation_policy.allow_manual_tick:
                raise GRHTTPAdapterError(
                    403,
                    "manual run_until_idle is disabled",
                    code="route_disabled",
                )
            responses = self.facade.run_until_idle(
                max_ticks=_optional_int(payload, "max_ticks"),
                timeout_unfinished=bool(payload.get("timeout_unfinished", False)),
            )
            return _ok({"responses": tuple(_response_payload(response) for response in responses)})
        if method == "GET" and len(route) == 2 and route[0] == "poll":
            response = self.facade.poll(route[1])
            return _ok(
                {
                    "ready": response is not None,
                    "response": _response_payload(response) if response is not None else None,
                }
            )
        if method == "GET" and len(route) == 2 and route[0] == "result":
            return _ok(_response_payload(self.facade.require_result(route[1])))
        if method == "POST" and route == ("cancel",):
            request_id = _required_str(payload, "request_id")
            reason = str(payload.get("reason", "cancelled"))
            return _ok(_response_payload(self.facade.cancel(request_id, reason=reason)))
        if method == "POST" and route == ("drain",):
            return _ok({"lifecycle": self.facade.drain()})
        if method == "POST" and route == ("shutdown",):
            shutdown_kwargs: dict[str, Any] = {
                "max_ticks": _optional_int(payload, "max_ticks"),
                "timeout_unfinished": bool(payload.get("timeout_unfinished", False)),
            }
            if "stop_timeout_s" in payload and hasattr(self.facade, "stop"):
                shutdown_kwargs["stop_timeout_s"] = float(payload["stop_timeout_s"])
            return _ok(_shutdown_payload(self.facade.shutdown(**shutdown_kwargs)))
        if method == "POST" and route == ("catalog", "reload"):
            if not self.validation_policy.allow_catalog_reload:
                raise GRHTTPAdapterError(
                    403,
                    "catalog reload is disabled",
                    code="route_disabled",
                )
            version = self.facade.reload_item_catalog_jsonl(
                _required_str(payload, "path"),
                vocab_size=_required_int(payload, "vocab_size"),
                eos_token_id=_optional_int(payload, "eos_token_id"),
                allow_eos_for_terminal=bool(payload.get("allow_eos_for_terminal", True)),
                item_id_field=str(payload.get("item_id_field", "item_id")),
                token_ids_field=str(payload.get("token_ids_field", "token_ids")),
                metadata_field=payload.get("metadata_field", "metadata"),
                allow_duplicate_item_ids=bool(payload.get("allow_duplicate_item_ids", False)),
                allow_duplicate_token_paths=bool(
                    payload.get("allow_duplicate_token_paths", False)
                ),
            )
            return _ok({"version": version, "catalog": self.facade.catalog_status()})
        if method == "POST" and route == ("catalog", "rollback"):
            if not self.validation_policy.allow_catalog_reload:
                raise GRHTTPAdapterError(
                    403,
                    "catalog rollback is disabled",
                    code="route_disabled",
                )
            version = self.facade.rollback_item_catalog()
            return _ok({"version": version, "catalog": self.facade.catalog_status()})
        return _error_response(404, f"unknown route: {method} /{'/'.join(route)}")

    def _emit_request_log(
        self,
        method: str,
        path: str,
        route: tuple[str, ...],
        response: GRHTTPResponse,
        started: float,
    ) -> None:
        if not self.enable_request_logging:
            return
        payload: dict[str, Any] = {
            "event": "gr_http_request",
            "method": method,
            "path": path,
            "route": "/" + "/".join(route),
            "status": response.status,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        if isinstance(response.body, Mapping):
            if "request_id" in response.body:
                payload["request_id"] = response.body["request_id"]
            error = response.body.get("error")
            if isinstance(error, Mapping):
                payload["error_code"] = error.get("code")
        _emit_structured_log(payload, self.request_log_sink)

    def _validate_auth(
        self,
        method: str,
        route: tuple[str, ...],
        headers: Mapping[str, str],
    ) -> None:
        if self.api_key is None:
            return
        if method == "GET" and route in {("health",), ("ready",)}:
            return
        provided = _auth_token_from_headers(headers, api_key_header=self.api_key_header)
        if provided is None:
            raise GRHTTPAdapterError(
                401,
                "missing API key",
                code="unauthorized",
                retryable=False,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if provided != self.api_key:
            raise GRHTTPAdapterError(
                403,
                "invalid API key",
                code="forbidden",
                retryable=False,
            )

    def _handle_sglang_generate(self, payload: Mapping[str, Any]) -> GRHTTPResponse:
        self._validate_admission(1)
        request_payload = _sglang_generate_payload_to_gr_payload(payload)
        request = self._make_request(request_payload)
        self._validate_request(request)
        request_id = self.facade.submit(request)
        response = self._wait_for_response(request_id)
        return _ok(_sglang_generate_response(response, request))

    def _wait_for_response(self, request_id: str) -> GRServingResponse:
        started = time.perf_counter()
        while True:
            response = self.facade.poll(request_id)
            if response is not None:
                return response
            if self.validation_policy.allow_manual_tick and not _worker_is_running(self.facade.status()):
                self.facade.tick()
            if time.perf_counter() - started > 300.0:
                raise GRHTTPAdapterError(
                    504,
                    f"generate timed out waiting for request_id={request_id}",
                    code="generate_timeout",
                    retryable=True,
                )
            time.sleep(0.001)

    def _make_request(self, payload: Mapping[str, Any]) -> GRServingRequest:
        factory = self.request_factory or default_request_factory
        return factory(payload)

    def _validate_body_size(
        self,
        body: bytes | str | Mapping[str, Any] | None,
    ) -> None:
        limit = self.validation_policy.max_request_bytes
        if limit is None or body is None or isinstance(body, Mapping):
            return
        size = len(body if isinstance(body, bytes) else body.encode("utf-8"))
        if size > limit:
            raise GRHTTPAdapterError(
                413,
                f"request body exceeds {limit} bytes",
                code="payload_too_large",
            )

    def _validate_admission(self, new_requests: int) -> None:
        max_waiting = self.validation_policy.max_waiting_requests
        if max_waiting is None:
            return
        status = self.facade.status()
        waiting = int(status.get("waiting_prefill", status.get("waiting", 0)) or 0)
        if waiting + new_requests > max_waiting:
            raise GRHTTPAdapterError(
                429,
                f"waiting queue capacity exceeded: {waiting}+{new_requests}>{max_waiting}",
                code="overloaded",
                retryable=True,
            )

    def _validate_request(self, request: GRServingRequest) -> None:
        policy = self.validation_policy
        if policy.max_decode_steps is not None and request.max_decode_steps > policy.max_decode_steps:
            raise GRHTTPAdapterError(
                400,
                f"max_decode_steps exceeds limit {policy.max_decode_steps}",
                code="validation_error",
            )
        if policy.max_beam_width is not None and request.beam_width > policy.max_beam_width:
            raise GRHTTPAdapterError(
                400,
                f"beam_width exceeds limit {policy.max_beam_width}",
                code="validation_error",
            )
        context_len = _input_context_len(request.input_ids)
        if (
            context_len is not None
            and policy.max_context_len is not None
            and context_len > policy.max_context_len
        ):
            raise GRHTTPAdapterError(
                400,
                f"input context length exceeds limit {policy.max_context_len}",
                code="validation_error",
            )
        timeout_ticks = request.metadata.get("timeout_ticks")
        if timeout_ticks is not None and policy.max_timeout_ticks is not None:
            if int(timeout_ticks) > policy.max_timeout_ticks:
                raise GRHTTPAdapterError(
                    400,
                    f"timeout_ticks exceeds limit {policy.max_timeout_ticks}",
                    code="validation_error",
                )

    def _readiness_payload(self) -> dict[str, Any]:
        status = self.facade.status()
        worker = status.get("worker")
        lifecycle = status.get("lifecycle", {})
        admission = self._admission_payload(status)
        reasons: list[str] = []
        if isinstance(lifecycle, Mapping) and lifecycle.get("draining", False):
            reasons.append("draining")
        if admission["queue_full"]:
            reasons.append("waiting_queue_full")
        if isinstance(worker, Mapping):
            if not worker.get("running", False):
                reasons.append("worker_not_running")
            if worker.get("last_error"):
                reasons.append("worker_last_error")
        elif not self.validation_policy.allow_manual_tick:
            reasons.append("no_worker_and_manual_tick_disabled")
        return {
            "ready": not reasons,
            "reasons": reasons,
            "lifecycle": lifecycle,
            "worker": worker,
            "admission": admission,
            "scheduler": {
                "waiting_prefill": status.get("waiting_prefill", 0),
                "decoding": status.get("decoding", 0),
                "finished": status.get("finished", 0),
                "failed_requests": status.get("failed_requests", 0),
            },
        }

    def _config_payload(self) -> dict[str, Any]:
        status = self.facade.status()
        return {
            "validation_policy": _validation_policy_payload(self.validation_policy),
            "auth": _auth_config_payload(self),
            "logging": _logging_config_payload(self),
            "build": dict(self.build_info),
            "readiness_policy": _readiness_policy_payload(self.validation_policy),
            "lifecycle": status.get("lifecycle"),
            "scheduler_policy": status.get("policy"),
            "worker": status.get("worker"),
            "item_constraints_configured": status.get("item_constraints") is not None,
            "routes": _route_manifest(self.validation_policy),
        }

    def _admission_payload(self, status: Mapping[str, Any]) -> dict[str, Any]:
        waiting = int(status.get("waiting_prefill", status.get("waiting", 0)) or 0)
        max_waiting = self.validation_policy.max_waiting_requests
        available_slots = None if max_waiting is None else max(0, max_waiting - waiting)
        return {
            "waiting_prefill": waiting,
            "max_waiting_requests": max_waiting,
            "available_waiting_slots": available_slots,
            "queue_full": max_waiting is not None and waiting >= max_waiting,
        }


def default_request_factory(payload: Mapping[str, Any]) -> GRServingRequest:
    beam_width = _required_int(payload, "beam_width")
    metadata = dict(payload.get("metadata", {}))
    if "timeout_ticks" in payload:
        metadata["timeout_ticks"] = _optional_int(payload, "timeout_ticks")
    return GRServingRequest(
        request_id=_required_str(payload, "request_id"),
        input_ids=_required_field(payload, "input_ids"),
        max_decode_steps=_required_int(payload, "max_decode_steps"),
        beam_width=beam_width,
        metadata=metadata,
        beam_width_policy=beam_width_policy_from_payload(
            payload,
            max_beam_width=beam_width,
        ),
        stop_token_ids=tuple(int(token) for token in payload.get("stop_token_ids", ())),
        logits_processors=logits_processors_from_specs(payload.get("logits_processors")),
    )


def _sglang_generate_payload_to_gr_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "input_ids" not in payload:
        if "text" in payload:
            raise ValueError(
                "GR /generate currently requires input_ids. "
                "Run sglang.bench_serving with --tokenize-prompt when targeting GR."
            )
        raise ValueError("missing required field: input_ids")
    sampling_params = payload.get("sampling_params", {})
    if sampling_params is None:
        sampling_params = {}
    if not isinstance(sampling_params, Mapping):
        raise ValueError("sampling_params must be a JSON object")
    max_new_tokens = int(sampling_params.get("max_new_tokens", payload.get("max_new_tokens", 1)))
    beam_width = int(sampling_params.get("n", payload.get("n", payload.get("beam_width", 1))))
    gr_decode_steps = max(1, max_new_tokens - 1)
    request_id = str(payload.get("request_id") or payload.get("rid") or f"generate-{uuid.uuid4()}")
    return {
        "request_id": request_id,
        "input_ids": payload["input_ids"],
        "max_decode_steps": gr_decode_steps,
        "beam_width": beam_width,
        "stop_token_ids": tuple(int(token) for token in sampling_params.get("stop_token_ids", ())),
        "logits_processors": payload.get("logits_processors", ()),
        "metadata": {
            "source": "sglang_generate",
            "requested_max_new_tokens": max_new_tokens,
            "ignore_eos": bool(sampling_params.get("ignore_eos", False)),
            "stream": bool(payload.get("stream", True)),
        },
    }


def _sglang_generate_response(
    response: GRServingResponse,
    request: GRServingRequest,
) -> dict[str, Any]:
    requested_max_new_tokens = int(
        response.metadata.get("requested_max_new_tokens", request.max_decode_steps + 1)
    )
    completion_tokens = int(request.beam_width) * requested_max_new_tokens
    beam_results = normalized_beam_results_from_metadata(
        response.metadata,
        max_new_tokens=requested_max_new_tokens,
    )
    text_token_ids = _top_beam_token_ids(
        response,
        requested_max_new_tokens,
        beam_results=beam_results,
    )
    meta_info = {
        "completion_tokens": completion_tokens,
        "prompt_tokens": _input_context_len(request.input_ids),
        "finish_reason": response.metadata.get("stop_reason", "max_decode_steps"),
        "request_id": response.request_id,
        "token_ids": text_token_ids,
        "beam_width": request.beam_width,
    }
    if beam_results:
        meta_info["beam_results"] = beam_results
    return {
        "text": " ".join(str(token) for token in text_token_ids),
        "output_ids": text_token_ids,
        "meta_info": meta_info,
    }


def _top_beam_token_ids(
    response: GRServingResponse,
    max_new_tokens: int,
    *,
    beam_results: tuple[dict[str, Any], ...] | None = None,
) -> tuple[int, ...]:
    beam_results = beam_results or ()
    if beam_results:
        output_ids = beam_results[0].get("output_ids")
        if isinstance(output_ids, (list, tuple)):
            return tuple(int(token) for token in output_ids[:max_new_tokens])
    beam_details = response.metadata.get("beam_details")
    if isinstance(beam_details, Mapping):
        beams = beam_details.get("beams")
        if isinstance(beams, list) and beams:
            first = beams[0]
            if isinstance(first, Mapping) and isinstance(first.get("token_ids"), list):
                return tuple(int(token) for token in first["token_ids"][:max_new_tokens])
    if isinstance(beam_details, (list, tuple)) and beam_details:
        first = beam_details[0]
        if isinstance(first, Mapping) and isinstance(first.get("token_ids"), (list, tuple)):
            return tuple(int(token) for token in first["token_ids"][:max_new_tokens])
    if response.token_ids:
        return (int(response.token_ids[0]),) * max_new_tokens
    return ()


def _worker_is_running(status: Mapping[str, Any]) -> bool:
    worker = status.get("worker")
    return isinstance(worker, Mapping) and bool(worker.get("running", False))


def beam_width_policy_from_payload(
    payload: Mapping[str, Any],
    *,
    max_beam_width: int,
) -> Any | None:
    spec = payload.get("beam_width_policy")
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise ValueError("beam_width_policy must be a JSON object")
    policy_type = spec.get("type")
    if policy_type in {"score_margin", "score_margin_shrink"}:
        if "score_margin" not in spec:
            raise ValueError("score_margin beam policy requires score_margin")
        return ScoreMarginBeamPolicy(
            max_beam_width=max_beam_width,
            score_margin=float(spec["score_margin"]),
            min_beam_width=int(spec.get("min_beam_width", 1)),
            monotonic_shrink=bool(spec.get("monotonic_shrink", True)),
        )
    if policy_type in {"scheduled", "schedule"}:
        schedule = _beam_width_schedule(spec.get("schedule"), max_beam_width=max_beam_width)
        return ScheduledBeamPolicy(schedule)
    raise ValueError(f"unsupported beam_width_policy type: {policy_type!r}")


def _beam_width_schedule(value: Any, *, max_beam_width: int) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError("scheduled beam policy requires schedule object")
    schedule: dict[int, int] = {}
    for raw_step, raw_width in value.items():
        if isinstance(raw_step, bool) or isinstance(raw_width, bool):
            raise ValueError("schedule steps and widths must be integers")
        step = int(raw_step)
        width = int(raw_width)
        if step < 0:
            raise ValueError("schedule steps must be non-negative")
        if width <= 0 or width > max_beam_width:
            raise ValueError(
                f"scheduled beam width must be in (0, {max_beam_width}], got {width}"
            )
        schedule[step] = width
    return schedule


def make_http_handler(adapter: GRHTTPServingAdapter) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._handle()

        def do_POST(self) -> None:  # noqa: N802
            self._handle()

        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else None
            response = adapter.handle(self.command, self.path, body, headers=self.headers)
            if isinstance(response.body, str):
                encoded = response.body.encode("utf-8")
            else:
                encoded = json.dumps(_jsonable(response.body)).encode("utf-8")
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(encoded)))
            for name, value in response.headers.items():
                if name.lower() in {"content-type", "content-length"}:
                    continue
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def _route_path(path: str) -> tuple[str, ...]:
    parsed = urlparse(path)
    return tuple(part for part in parsed.path.split("/") if part)


def _json_payload(body: bytes | str | Mapping[str, Any] | None) -> Mapping[str, Any]:
    if body is None or body == b"" or body == "":
        return {}
    if isinstance(body, Mapping):
        return body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, Mapping):
        raise ValueError("JSON body must be an object")
    return parsed


def _emit_structured_log(payload: Mapping[str, Any], sink: LogSink | None) -> None:
    if sink is not None:
        sink(payload)
        return
    print(json.dumps(_jsonable(payload), sort_keys=True), file=sys.stderr)


def _auth_token_from_headers(
    headers: Mapping[str, str],
    *,
    api_key_header: str,
) -> str | None:
    bearer = _header_value(headers, "Authorization")
    if bearer is not None:
        prefix = "Bearer "
        if bearer.startswith(prefix):
            return bearer[len(prefix) :]
        return bearer
    return _header_value(headers, api_key_header)


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def _ok(body: dict[str, Any], *, status: int = 200) -> GRHTTPResponse:
    return GRHTTPResponse(status=status, body=_jsonable(body))


def _text(body: str, *, status: int = 200, content_type: str = "text/plain") -> GRHTTPResponse:
    return GRHTTPResponse(status=status, body=body, headers={"Content-Type": content_type})


def _error_response(
    status: int,
    message: str,
    *,
    code: str = "http_error",
    retryable: bool = False,
    headers: Mapping[str, str] | None = None,
) -> GRHTTPResponse:
    return _structured_error_response(
        status,
        message,
        code=code,
        retryable=retryable,
        headers=headers,
    )


def _structured_error_response(
    status: int,
    message: str,
    *,
    code: str,
    retryable: bool,
    headers: Mapping[str, str] | None = None,
) -> GRHTTPResponse:
    return GRHTTPResponse(
        status=status,
        body={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "status": status,
            }
        },
        headers=dict(headers or {}),
    )


def _runtime_error_response(exc: RuntimeError) -> GRHTTPResponse:
    message = str(exc)
    if "draining" in message:
        return _structured_error_response(
            409,
            message,
            code="service_draining",
            retryable=True,
        )
    if "duplicate request_id" in message:
        return _structured_error_response(
            409,
            message,
            code="duplicate_request_id",
            retryable=False,
        )
    return _structured_error_response(
        409,
        message,
        code="conflict",
        retryable=False,
    )


def _validation_policy_payload(policy: GRHTTPValidationPolicy) -> dict[str, Any]:
    return {
        "max_request_bytes": policy.max_request_bytes,
        "max_context_len": policy.max_context_len,
        "max_decode_steps": policy.max_decode_steps,
        "max_beam_width": policy.max_beam_width,
        "max_submit_many": policy.max_submit_many,
        "max_waiting_requests": policy.max_waiting_requests,
        "max_timeout_ticks": policy.max_timeout_ticks,
        "allow_manual_tick": policy.allow_manual_tick,
        "allow_catalog_reload": policy.allow_catalog_reload,
    }


def _auth_config_payload(adapter: GRHTTPServingAdapter) -> dict[str, Any]:
    return {
        "enabled": adapter.api_key is not None,
        "api_key_header": adapter.api_key_header if adapter.api_key is not None else None,
        "accepted_schemes": ("Bearer", adapter.api_key_header) if adapter.api_key is not None else (),
        "exempt_routes": ("GET /health", "GET /ready") if adapter.api_key is not None else (),
    }


def _logging_config_payload(adapter: GRHTTPServingAdapter) -> dict[str, Any]:
    return {
        "request_logging_enabled": adapter.enable_request_logging,
        "log_requests_level": adapter.log_requests_level,
    }


def _default_build_info() -> dict[str, Any]:
    try:
        version = importlib_metadata.version("gr-inference")
    except importlib_metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "framework": "gr-inference",
        "version": version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _readiness_policy_payload(policy: GRHTTPValidationPolicy) -> dict[str, Any]:
    return {
        "draining_blocks_ready": True,
        "queue_full_blocks_ready": policy.max_waiting_requests is not None,
        "worker_error_blocks_ready": True,
        "worker_required_when_manual_tick_disabled": not policy.allow_manual_tick,
    }


def _prometheus_metrics_text(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# HELP gr_serving_info GR serving Prometheus exporter info.",
        "# TYPE gr_serving_info gauge",
        "gr_serving_info 1",
    ]
    for raw_name, value in sorted(metrics.items()):
        if isinstance(value, bool):
            number = int(value)
        elif isinstance(value, int | float):
            number = value
        else:
            continue
        name = "gr_serving_" + _prometheus_metric_name(str(raw_name))
        lines.extend(
            (
                f"# HELP {name} GR serving metric {raw_name}.",
                f"# TYPE {name} gauge",
                f"{name} {number}",
            )
        )
    return "\n".join(lines) + "\n"


def _prometheus_metric_name(name: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in name.lower())
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    if not sanitized:
        return "metric"
    if sanitized[0].isdigit():
        return "metric_" + sanitized
    return sanitized


def _route_manifest(policy: GRHTTPValidationPolicy) -> dict[str, tuple[str, ...]]:
    routes: dict[str, tuple[str, ...]] = {
        "liveness": ("GET /health",),
        "readiness": ("GET /ready",),
        "configuration": ("GET /config",),
        "build": ("GET /build",),
        "openai_compat": ("GET /v1/models",),
        "status": (
            "GET /status",
            "GET /metrics",
            "GET /metrics/prometheus",
            "GET /kv/events",
            "GET /requests",
            "GET /requests/{request_id}",
        ),
        "requests": (
            "POST /generate",
            "POST /submit",
            "POST /submit_many",
            "GET /poll/{request_id}",
            "GET /result/{request_id}",
            "POST /cancel",
        ),
        "lifecycle": ("POST /drain", "POST /shutdown"),
    }
    if policy.allow_manual_tick:
        routes["manual_control"] = ("POST /tick", "POST /run_until_idle")
    if policy.allow_catalog_reload:
        routes["catalog"] = (
            "GET /catalog/status",
            "POST /catalog/reload",
            "POST /catalog/rollback",
        )
    else:
        routes["catalog"] = ("GET /catalog/status",)
    return routes


def _response_payload(response: GRServingResponse) -> dict[str, Any]:
    return {
        "request_id": response.request_id,
        "token_ids": response.token_ids,
        "scores": response.scores,
        "metadata": response.metadata,
    }


def _kv_events_payload(status: Mapping[str, Any]) -> dict[str, Any]:
    events = status.get("kv_events", ())
    return {
        "events": tuple(events) if isinstance(events, (list, tuple)) else (),
        "count": len(events) if isinstance(events, (list, tuple)) else 0,
    }


def _shutdown_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    responses = payload.get("responses", ())
    payload["responses"] = tuple(_response_payload(response) for response in responses)
    return payload


def _result_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "metadata"):
        return result.metadata()
    return {"result": result}


def _input_context_len(input_ids: Any) -> int | None:
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    if isinstance(input_ids, (list, tuple)):
        if not input_ids:
            return 0
        first = input_ids[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(input_ids)
    return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)
