import json

import pytest
from gr_inference.gr_kv import BeamPath
from gr_inference.gr_runtime import (
    TokenTrie,
    TrieItemMaskProvider,
    TrieItemMaskProviderStore,
)
from gr_inference.gr_serving import (
    GRContinuousBatchingPolicy,
    GRContinuousScheduler,
    GRHTTPServingAdapter,
    GRHTTPValidationPolicy,
    GRInProcessServingFacade,
    GRServingRequest,
    GRServingResponse,
    GRServingWorker,
    default_request_factory,
)
from gr_inference.gr_serving.http import _sglang_generate_response
from tools.serve_qwen3_gr_http import make_torch_request_factory


def make_adapter() -> GRHTTPServingAdapter:
    return GRHTTPServingAdapter(
        GRInProcessServingFacade(
            GRContinuousScheduler(
                policy=GRContinuousBatchingPolicy(
                    max_prefill_batch_size=2,
                    max_decode_batch_size=2,
                )
            )
        )
    )


def simple_adapter(**kwargs) -> GRHTTPServingAdapter:
    return GRHTTPServingAdapter(
        GRInProcessServingFacade(GRContinuousScheduler()),
        **kwargs,
    )


def policy_adapter(**policy_kwargs) -> GRHTTPServingAdapter:
    return simple_adapter(
        validation_policy=GRHTTPValidationPolicy(**policy_kwargs),
    )


def request_payload(request_id: str = "req-0") -> dict:
    return {
        "request_id": request_id,
        "input_ids": [[1, 2, 3]],
        "max_decode_steps": 1,
        "beam_width": 2,
        "metadata": {"source": "test"},
    }


def test_http_adapter_submit_run_and_poll() -> None:
    adapter = make_adapter()

    submit = adapter.handle("POST", "/submit", request_payload())
    status = adapter.handle("GET", "/status")
    run = adapter.handle("POST", "/run_until_idle", {})
    poll = adapter.handle("GET", "/poll/req-0")

    assert submit.status == 202
    assert submit.body["request_id"] == "req-0"
    assert status.body["waiting_prefill"] == 1
    assert run.body["responses"][0]["request_id"] == "req-0"
    assert poll.body["ready"] is True
    assert poll.body["response"]["request_id"] == "req-0"


def test_http_adapter_generate_sglang_native_input_ids() -> None:
    adapter = make_adapter()

    response = adapter.handle(
        "POST",
        "/generate",
        {
            "input_ids": [1, 2, 3],
            "sampling_params": {
                "max_new_tokens": 3,
                "n": 2,
                "ignore_eos": True,
            },
            "stream": False,
        },
    )

    assert response.status == 200
    assert "text" in response.body
    assert response.body["meta_info"]["completion_tokens"] == 6
    assert response.body["meta_info"]["prompt_tokens"] == 3
    assert response.body["meta_info"]["beam_width"] == 2


def test_http_adapter_generate_returns_beam_results_when_available() -> None:
    request = GRServingRequest(
        request_id="beam-results",
        input_ids=[1, 2, 3],
        max_decode_steps=2,
        beam_width=2,
        metadata={"requested_max_new_tokens": 3},
    )
    response = GRServingResponse(
        request_id="beam-results",
        token_ids=(6, 7),
        scores=(1.0, 0.5),
        metadata={
            "requested_max_new_tokens": 3,
            "stop_reason": "max_decode_steps",
            "beam_results": (
                {
                    "output_ids": (4, 5, 6),
                    "text": "4 5 6",
                    "meta_info": {"sequence_score": 1.0},
                },
                {
                    "output_ids": (7, 8, 9),
                    "text": "7 8 9",
                    "meta_info": {"sequence_score": 0.5},
                },
            ),
        },
    )

    payload = _sglang_generate_response(response, request)

    assert payload["output_ids"] == (4, 5, 6)
    assert payload["meta_info"]["token_ids"] == (4, 5, 6)
    assert payload["meta_info"]["beam_results"][1]["output_ids"] == (7, 8, 9)


def test_http_adapter_generate_builds_beam_results_from_beam_path_metadata() -> None:
    beam_path = BeamPath(max_decode_steps=2, max_beam_width=2)
    beam_path.append(parent_beams=(0, 0), token_ids=(4, 7), scores=(1.0, 0.5))
    beam_path.append(parent_beams=(0, 1), token_ids=(5, 8), scores=(1.5, 0.75))
    request = GRServingRequest(
        request_id="beam-path",
        input_ids=[1, 2, 3],
        max_decode_steps=1,
        beam_width=2,
        metadata={"requested_max_new_tokens": 2},
    )
    response = GRServingResponse(
        request_id="beam-path",
        token_ids=(4, 7),
        scores=(1.5, 0.75),
        metadata={
            "requested_max_new_tokens": 2,
            "stop_reason": "max_decode_steps",
            "active_beam_width": 2,
            "_beam_path": beam_path,
        },
    )

    payload = _sglang_generate_response(response, request)

    assert payload["output_ids"] == (4, 5)
    assert payload["meta_info"]["beam_results"][0]["output_ids"] == (4, 5)
    assert payload["meta_info"]["beam_results"][1]["output_ids"] == (7, 8)


def test_http_adapter_generate_requires_tokenized_input() -> None:
    adapter = make_adapter()

    response = adapter.handle(
        "POST",
        "/generate",
        {
            "text": "hello",
            "sampling_params": {"max_new_tokens": 3, "n": 2},
        },
    )

    assert response.status == 400
    assert response.body["error"]["code"] == "validation_error"
    assert "--tokenize-prompt" in response.body["error"]["message"]


def test_default_request_factory_parses_builtin_logits_processors() -> None:
    request = default_request_factory(
        {
            **request_payload("processor"),
            "logits_processors": [
                {"type": "token_suppress", "token_ids": [9], "phases": ["prefill"]},
                {"type": "token_bias", "token_bias": {"10": 1.25}},
            ],
        }
    )

    assert len(request.logits_processors) == 2
    assert request.logits_processors[0].metadata()["type"] == "token_suppress"
    assert request.logits_processors[1].metadata()["token_bias"] == {10: 1.25}


def test_torch_request_factory_suppresses_tokens_when_ignore_eos() -> None:
    torch = pytest.importorskip("torch")

    factory = make_torch_request_factory(
        torch,
        device="cpu",
        default_decode_steps=2,
        default_beam_width=4,
        suppress_token_ids_on_ignore_eos=(9, 10),
    )

    request = factory(
        {
            "request_id": "ignore-eos",
            "input_ids": [1, 2, 3],
            "max_decode_steps": 2,
            "beam_width": 4,
            "metadata": {"ignore_eos": True},
        }
    )

    assert len(request.logits_processors) == 1
    assert request.logits_processors[0].metadata()["token_ids"] == (9, 10)


def test_http_adapter_rejects_duplicate_submit_with_conflict() -> None:
    adapter = make_adapter()

    first = adapter.handle("POST", "/submit", request_payload("dup"))
    duplicate = adapter.handle("POST", "/submit", request_payload("dup"))

    assert first.status == 202
    assert duplicate.status == 409
    assert duplicate.body["error"]["code"] == "duplicate_request_id"
    assert adapter.handle("GET", "/status").body["waiting_prefill"] == 1


def test_http_adapter_submit_many_rejects_duplicates_atomically() -> None:
    adapter = make_adapter()

    duplicate_batch = adapter.handle(
        "POST",
        "/submit_many",
        {"requests": [request_payload("dup"), request_payload("dup")]},
    )
    existing = adapter.handle("POST", "/submit", request_payload("existing"))
    partial_conflict = adapter.handle(
        "POST",
        "/submit_many",
        {"requests": [request_payload("new"), request_payload("existing")]},
    )
    status = adapter.handle("GET", "/status")
    requests = adapter.handle("GET", "/requests")

    assert duplicate_batch.status == 409
    assert duplicate_batch.body["error"]["code"] == "duplicate_request_id"
    assert existing.status == 202
    assert partial_conflict.status == 409
    assert partial_conflict.body["error"]["code"] == "duplicate_request_id"
    assert status.body["waiting_prefill"] == 1
    assert [request["request_id"] for request in requests.body["requests"]] == [
        "existing"
    ]


def test_http_adapter_reports_readiness_and_config() -> None:
    adapter = make_adapter()

    health = adapter.handle("GET", "/health")
    ready = adapter.handle("GET", "/ready")
    config = adapter.handle("GET", "/config")
    build = adapter.handle("GET", "/build")
    models = adapter.handle("GET", "/v1/models")

    assert health.body == {"ok": True}
    assert ready.body["ready"] is True
    assert ready.body["reasons"] == []
    assert ready.body["admission"]["queue_full"] is False
    assert config.body["validation_policy"]["allow_manual_tick"] is True
    assert config.body["build"]["framework"] == "gr-inference"
    assert build.body["build"]["framework"] == "gr-inference"
    assert models.body["object"] == "list"
    assert models.body["data"][0]["object"] == "model"
    assert "python" in build.body["build"]
    assert config.body["readiness_policy"]["draining_blocks_ready"] is True
    assert "GET /build" in config.body["routes"]["build"]
    assert "GET /v1/models" in config.body["routes"]["openai_compat"]
    assert "GET /ready" in config.body["routes"]["readiness"]
    assert "POST /submit" in config.body["routes"]["requests"]


def test_http_adapter_api_key_auth_protects_non_probe_routes() -> None:
    adapter = simple_adapter(api_key="secret")

    health = adapter.handle("GET", "/health")
    ready = adapter.handle("GET", "/ready")
    unauthorized = adapter.handle("GET", "/metrics")
    forbidden = adapter.handle(
        "GET", "/metrics", headers={"Authorization": "Bearer wrong"}
    )
    authorized = adapter.handle(
        "GET", "/metrics", headers={"Authorization": "Bearer secret"}
    )
    build_authorized = adapter.handle(
        "GET", "/build", headers={"Authorization": "Bearer secret"}
    )
    header_authorized = adapter.handle(
        "GET", "/config", headers={"X-GR-API-Key": "secret"}
    )

    assert health.status == 200
    assert ready.status == 200
    assert unauthorized.status == 401
    assert unauthorized.body["error"]["code"] == "unauthorized"
    assert unauthorized.headers["WWW-Authenticate"] == "Bearer"
    assert forbidden.status == 403
    assert forbidden.body["error"]["code"] == "forbidden"
    assert authorized.status == 200
    assert build_authorized.status == 200
    assert header_authorized.body["auth"]["enabled"] is True
    assert header_authorized.body["auth"]["exempt_routes"] == [
        "GET /health",
        "GET /ready",
    ]


def test_http_adapter_emits_structured_request_logs() -> None:
    logs = []
    adapter = simple_adapter(
        enable_request_logging=True,
        request_log_sink=logs.append,
    )

    response = adapter.handle("POST", "/submit", request_payload("logged"))
    adapter.handle("GET", "/result/missing")
    config = adapter.handle("GET", "/config")

    assert response.status == 202
    assert len(logs) == 3
    assert logs[0]["event"] == "gr_http_request"
    assert logs[0]["method"] == "POST"
    assert logs[0]["path"] == "/submit"
    assert logs[0]["status"] == 202
    assert logs[0]["request_id"] == "logged"
    assert logs[1]["error_code"] == "not_found"
    assert config.body["logging"]["request_logging_enabled"] is True


def test_http_adapter_exposes_kv_events() -> None:
    adapter = make_adapter()
    adapter.handle("POST", "/submit", request_payload("kv-event"))
    adapter.handle("POST", "/run_until_idle", {})

    events = adapter.handle("GET", "/kv/events")
    event_types = [event["type"] for event in events.body["events"]]

    assert events.status == 200
    assert event_types == ["allocate", "release"]
    assert events.body["events"][0]["request_id"] == "kv-event"


def test_serving_worker_emits_decode_interval_logs() -> None:
    logs = []
    worker = GRServingWorker(
        GRInProcessServingFacade(GRContinuousScheduler()),
        decode_log_interval=1,
        log_sink=logs.append,
    )
    worker.submit(
        GRServingRequest(
            request_id="worker-log",
            input_ids=[1, 2, 3],
            max_decode_steps=1,
            beam_width=2,
        )
    )

    tick = worker.tick()

    assert tick.tick == 1
    assert logs == [
        {
            "event": "gr_worker_decode_tick",
            "worker_ticks": 1,
            "tick": 1,
            "prefill_batch_size": 1,
            "decode_batches": 1,
            "finished_requests": 1,
        }
    ]


def test_http_adapter_readiness_requires_worker_when_manual_tick_disabled() -> None:
    adapter = policy_adapter(allow_manual_tick=False)

    ready = adapter.handle("GET", "/ready")
    config = adapter.handle("GET", "/config")

    assert ready.body["ready"] is False
    assert ready.body["reasons"] == ["no_worker_and_manual_tick_disabled"]
    assert "manual_control" not in config.body["routes"]


def test_http_adapter_readiness_reports_full_waiting_queue() -> None:
    adapter = policy_adapter(max_waiting_requests=1)
    adapter.handle("POST", "/submit", request_payload("queued"))

    ready = adapter.handle("GET", "/ready")
    config = adapter.handle("GET", "/config")

    assert ready.body["ready"] is False
    assert ready.body["reasons"] == ["waiting_queue_full"]
    assert ready.body["admission"] == {
        "waiting_prefill": 1,
        "max_waiting_requests": 1,
        "available_waiting_slots": 0,
        "queue_full": True,
    }
    assert config.body["readiness_policy"]["queue_full_blocks_ready"] is True


def test_http_adapter_readiness_reports_worker_error() -> None:
    worker = GRServingWorker(GRInProcessServingFacade(GRContinuousScheduler()))
    worker._last_error = "RuntimeError: boom"
    adapter = GRHTTPServingAdapter(worker)

    ready = adapter.handle("GET", "/ready")

    assert ready.body["ready"] is False
    assert ready.body["reasons"] == ["worker_not_running", "worker_last_error"]
    assert ready.body["worker"]["last_error"] == "RuntimeError: boom"


def test_http_adapter_drain_rejects_new_requests_and_marks_not_ready() -> None:
    adapter = make_adapter()

    drain = adapter.handle("POST", "/drain", {})
    ready = adapter.handle("GET", "/ready")
    status = adapter.handle("GET", "/status")
    submit = adapter.handle("POST", "/submit", request_payload("after-drain"))

    assert drain.body["lifecycle"]["draining"] is True
    assert ready.body["ready"] is False
    assert ready.body["reasons"] == ["draining"]
    assert status.body["lifecycle"]["accepting_requests"] is False
    assert submit.status == 409
    assert submit.body["error"]["code"] == "service_draining"
    assert submit.body["error"]["retryable"] is True


def test_http_adapter_shutdown_drains_and_times_out_unfinished_requests() -> None:
    adapter = make_adapter()
    adapter.handle(
        "POST",
        "/submit",
        {**request_payload("shutdown-req"), "max_decode_steps": 2},
    )

    shutdown = adapter.handle(
        "POST",
        "/shutdown",
        {"max_ticks": 1, "timeout_unfinished": True},
    )
    submit = adapter.handle("POST", "/submit", request_payload("after-shutdown"))

    assert shutdown.body["lifecycle"]["draining"] is True
    assert shutdown.body["lifecycle"]["idle"] is True
    assert shutdown.body["responses"][0]["request_id"] == "shutdown-req"
    assert shutdown.body["responses"][0]["metadata"]["stop_reason"] == "timeout"
    assert submit.status == 409


def test_http_adapter_reports_request_status_lifecycle() -> None:
    adapter = make_adapter()

    submit = adapter.handle(
        "POST",
        "/submit",
        {**request_payload("req-status"), "max_decode_steps": 2},
    )
    waiting = adapter.handle("GET", "/requests/req-status")
    listing = adapter.handle("GET", "/requests")
    tick = adapter.handle("POST", "/tick")
    decoding = adapter.handle("GET", "/requests/req-status")
    adapter.handle("POST", "/run_until_idle", {})
    finished = adapter.handle("GET", "/requests/req-status")
    missing = adapter.handle("GET", "/requests/missing")

    assert submit.status == 202
    assert waiting.body["request"]["stage"] == "waiting_prefill"
    assert waiting.body["request"]["ready"] is False
    assert listing.body["requests"][0]["request_id"] == "req-status"
    assert tick.body["decode_batches"][0]["request_ids"] == ["req-status"]
    assert decoding.body["request"]["stage"] == "decoding"
    assert decoding.body["request"]["current_decode_step"] == 1
    assert finished.body["request"]["stage"] == "finished"
    assert finished.body["request"]["ready"] is True
    assert finished.body["request"]["stop_reason"] == "max_decode_steps"
    assert missing.status == 404


def test_http_adapter_submit_many_tick_metrics_and_cancel() -> None:
    adapter = make_adapter()

    submit = adapter.handle(
        "POST",
        "/submit_many",
        {
            "requests": [
                request_payload("req-0"),
                {**request_payload("req-1"), "max_decode_steps": 2},
            ]
        },
    )
    tick = adapter.handle("POST", "/tick")
    cancel = adapter.handle(
        "POST",
        "/cancel",
        json.dumps({"request_id": "req-1", "reason": "client_cancelled"}),
    )
    metrics = adapter.handle("GET", "/metrics")

    assert submit.body["request_ids"] == ["req-0", "req-1"]
    assert tick.body["prefill_request_ids"] == ["req-0", "req-1"]
    assert cancel.body["metadata"]["cancelled"] is True
    assert metrics.body["cancelled_requests"] == 1


def test_http_adapter_catalog_status_and_reload(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    invalid = tmp_path / "invalid.jsonl"
    first.write_text(
        json.dumps({"item_id": "item-a", "token_ids": [1, 10]}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"item_id": "item-b", "token_ids": [2, 11]}),
        encoding="utf-8",
    )
    invalid.write_text(
        json.dumps({"item_id": "bad", "token_ids": [99]}),
        encoding="utf-8",
    )
    store = TrieItemMaskProviderStore.from_jsonl(first, vocab_size=32)
    adapter = GRHTTPServingAdapter(
        GRInProcessServingFacade(
            GRContinuousScheduler(),
            item_mask_provider_store=store,
        )
    )

    before = adapter.handle("GET", "/catalog/status")
    reload_response = adapter.handle(
        "POST",
        "/catalog/reload",
        {"path": str(second), "vocab_size": 32},
    )
    after = adapter.handle("GET", "/catalog/status")

    assert before.body["catalog"]["version"] == 1
    assert reload_response.body["version"] == 2
    assert after.body["catalog"]["version"] == 2
    assert after.body["catalog"]["previous_version"] == 1
    assert after.body["catalog"]["last_reload"]["operation"] == "reload_jsonl"
    assert after.body["catalog"]["last_reload"]["status"] == "succeeded"
    assert reload_response.body["catalog"]["source"] == str(second)
    assert reload_response.body["catalog"]["version"] == 2
    assert reload_response.body["catalog"]["last_reload"]["status"] == "succeeded"
    store_after_reload = TrieItemMaskProviderStore.from_jsonl(second, vocab_size=32)
    assert store_after_reload.snapshot().resolve_item_ids((2, 11)) == ("item-b",)
    assert store.snapshot().resolve_item_ids((2, 11)) == ("item-b",)

    failed_reload = adapter.handle(
        "POST",
        "/catalog/reload",
        {"path": str(invalid), "vocab_size": 32},
    )
    assert failed_reload.status == 400
    assert store.snapshot().resolve_item_ids((2, 11)) == ("item-b",)
    assert (
        adapter.handle("GET", "/catalog/status").body["catalog"]["last_reload"][
            "status"
        ]
        == "failed"
    )

    rollback = adapter.handle("POST", "/catalog/rollback", {})
    rolled_back = adapter.handle("GET", "/catalog/status")
    assert rollback.body["version"] == 3
    assert rolled_back.body["catalog"]["version"] == 3
    assert rolled_back.body["catalog"]["last_reload"]["operation"] == "rollback"
    assert store.snapshot().resolve_item_ids((1, 10)) == ("item-a",)


def test_http_adapter_injects_facade_item_constraints() -> None:
    provider = TrieItemMaskProvider(
        TokenTrie.from_items([("item-a", (1, 10))]),
        vocab_size=32,
    )
    scheduler = GRContinuousScheduler()
    adapter = GRHTTPServingAdapter(
        GRInProcessServingFacade(
            scheduler,
            item_mask_provider_store=TrieItemMaskProviderStore(provider),
        )
    )

    response = adapter.handle("POST", "/submit", request_payload())

    assert response.status == 202
    assert scheduler.states["req-0"].request.item_mask_provider is provider


def test_http_adapter_returns_json_errors() -> None:
    adapter = make_adapter()

    bad_submit = adapter.handle("POST", "/submit", {"request_id": "req-0"})
    not_found = adapter.handle("GET", "/missing")
    not_ready = adapter.handle("GET", "/result/req-missing")

    assert bad_submit.status == 400
    assert bad_submit.body["error"]["code"] == "validation_error"
    assert "missing required field" in bad_submit.body["error"]["message"]
    assert not_found.status == 404
    assert not_found.body["error"]["code"] == "http_error"
    assert not_ready.status == 404
    assert not_ready.body["error"]["code"] == "not_found"


def test_http_adapter_rejects_overloaded_waiting_queue() -> None:
    adapter = policy_adapter(max_waiting_requests=1)

    first = adapter.handle("POST", "/submit", request_payload("queued-0"))
    overloaded = adapter.handle("POST", "/submit", request_payload("queued-1"))

    assert first.status == 202
    assert overloaded.status == 429
    assert overloaded.body["error"]["code"] == "overloaded"
    assert overloaded.body["error"]["retryable"] is True


def test_http_adapter_accepts_top_level_timeout_ticks() -> None:
    adapter = make_adapter()

    submit = adapter.handle(
        "POST", "/submit", {**request_payload(), "timeout_ticks": 1}
    )
    request_status = adapter.handle("GET", "/requests/req-0")

    assert submit.status == 202
    assert (
        adapter.facade._scheduler().states["req-0"].request.metadata["timeout_ticks"]
        == 1
    )
    assert request_status.body["request"]["submitted_tick"] == 0
    assert request_status.body["request"]["age_ticks"] == 0
    assert request_status.body["request"]["timeout_ticks"] == 1
    assert request_status.body["request"]["deadline_tick"] == 1


def test_http_adapter_rejects_timeout_ticks_over_policy_limit() -> None:
    adapter = policy_adapter(max_timeout_ticks=2)

    response = adapter.handle(
        "POST", "/submit", {**request_payload(), "timeout_ticks": 3}
    )
    config = adapter.handle("GET", "/config")

    assert response.status == 400
    assert response.body["error"]["code"] == "validation_error"
    assert "timeout_ticks" in response.body["error"]["message"]
    assert config.body["validation_policy"]["max_timeout_ticks"] == 2


def test_http_adapter_metrics_include_request_outcomes() -> None:
    adapter = make_adapter()

    adapter.handle(
        "POST",
        "/submit",
        {**request_payload(), "max_decode_steps": 3, "timeout_ticks": 1},
    )
    adapter.handle("POST", "/tick", {})
    adapter.handle("POST", "/tick", {})
    metrics = adapter.handle("GET", "/metrics")
    request_status = adapter.handle("GET", "/requests/req-0")

    assert metrics.body["active_requests"] == 0
    assert metrics.body["finished_requests"] == 1
    assert metrics.body["failed_requests"] == 1
    assert metrics.body["timed_out_requests"] == 1
    assert metrics.body["succeeded_requests"] == 0
    assert request_status.body["request"]["ready"] is True
    assert request_status.body["request"]["stop_reason"] == "request_timeout"
    assert request_status.body["request"]["elapsed_ticks"] == 2


def test_http_adapter_exports_prometheus_metrics() -> None:
    adapter = make_adapter()
    adapter.handle("POST", "/submit", request_payload("prom"))
    adapter.handle("POST", "/run_until_idle", {})

    response = adapter.handle("GET", "/metrics/prometheus")
    config = adapter.handle("GET", "/config")

    assert response.status == 200
    assert response.headers["Content-Type"].startswith("text/plain")
    assert "gr_serving_submitted_requests 1" in response.body
    assert "gr_serving_succeeded_requests 1" in response.body
    assert "# TYPE gr_serving_finished_requests gauge" in response.body
    assert "GET /metrics/prometheus" in config.body["routes"]["status"]


def test_http_adapter_reports_retained_and_evicted_results() -> None:
    adapter = GRHTTPServingAdapter(
        GRInProcessServingFacade(
            GRContinuousScheduler(
                policy=GRContinuousBatchingPolicy(
                    max_prefill_batch_size=2,
                    max_decode_batch_size=2,
                    max_finished_requests=1,
                )
            )
        )
    )

    adapter.handle(
        "POST",
        "/submit_many",
        {"requests": [request_payload("req-0"), request_payload("req-1")]},
    )
    adapter.handle("POST", "/run_until_idle", {})
    metrics = adapter.handle("GET", "/metrics")
    status = adapter.handle("GET", "/status")
    evicted_status = adapter.handle("GET", "/requests/req-0")
    evicted_result = adapter.handle("GET", "/result/req-0")

    assert metrics.body["succeeded_requests"] == 2
    assert metrics.body["retained_finished_requests"] == 1
    assert metrics.body["evicted_finished_requests"] == 1
    assert status.body["policy"]["max_finished_requests"] == 1
    assert evicted_status.body["request"]["ready"] is True
    assert evicted_status.body["request"]["result_available"] is False
    assert evicted_result.status == 404


def test_http_adapter_reports_worker_status() -> None:
    adapter = GRHTTPServingAdapter(
        GRServingWorker(
            GRInProcessServingFacade(GRContinuousScheduler()),
        )
    )

    status = adapter.handle("GET", "/status")
    metrics = adapter.handle("GET", "/metrics")

    assert status.body["worker"]["running"] is False
    assert metrics.body["worker_running"] == 0


def test_http_adapter_rejects_requests_over_policy_limits() -> None:
    adapter = policy_adapter(
        max_context_len=2,
        max_decode_steps=1,
        max_beam_width=2,
    )

    long_context = adapter.handle("POST", "/submit", request_payload())
    large_beam = adapter.handle(
        "POST",
        "/submit",
        {**request_payload("req-beam"), "input_ids": [1], "beam_width": 3},
    )

    assert long_context.status == 400
    assert "context length" in long_context.body["error"]["message"]
    assert large_beam.status == 400
    assert "beam_width" in large_beam.body["error"]["message"]


def test_http_adapter_rejects_large_body_and_disabled_control_routes(tmp_path) -> None:
    adapter = policy_adapter(
        max_request_bytes=8,
        allow_manual_tick=False,
        allow_catalog_reload=False,
    )

    too_large = adapter.handle("POST", "/submit", '{"request_id":"req"}')
    tick = adapter.handle("POST", "/tick")
    reload_response = adapter.handle(
        "POST",
        "/catalog/reload",
        {"path": str(tmp_path / "catalog.jsonl"), "vocab_size": 32},
    )

    assert too_large.status == 413
    assert too_large.body["error"]["code"] == "payload_too_large"
    assert tick.status == 403
    assert tick.body["error"]["code"] == "route_disabled"
    assert reload_response.status == 403
    assert reload_response.body["error"]["code"] == "route_disabled"


def test_http_default_request_factory_builds_score_margin_policy() -> None:
    adapter = simple_adapter()

    response = adapter.handle(
        "POST",
        "/submit",
        {
            **request_payload("req-dynamic"),
            "beam_width_policy": {
                "type": "score_margin",
                "score_margin": 0.25,
                "min_beam_width": 1,
            },
        },
    )

    request = adapter.facade.executor.states["req-dynamic"].request
    assert response.status == 202
    assert request.beam_width_policy.width_for_step(0) == request.beam_width


def test_http_default_request_factory_builds_scheduled_policy() -> None:
    adapter = simple_adapter()

    response = adapter.handle(
        "POST",
        "/submit",
        {
            **request_payload("req-scheduled"),
            "beam_width": 4,
            "beam_width_policy": {
                "type": "scheduled",
                "schedule": {
                    "0": 4,
                    "1": 2,
                    "3": 1,
                },
            },
        },
    )

    request = adapter.facade.executor.states["req-scheduled"].request
    assert response.status == 202
    assert request.beam_width_policy.width_for_step(0) == 4
    assert request.beam_width_policy.width_for_step(1) == 2
    assert request.beam_width_policy.width_for_step(2) == 2
    assert request.beam_width_policy.width_for_step(3) == 1


def test_http_default_request_factory_rejects_scheduled_width_over_request_width() -> (
    None
):
    adapter = simple_adapter()

    response = adapter.handle(
        "POST",
        "/submit",
        {
            **request_payload("req-bad-scheduled"),
            "beam_width": 2,
            "beam_width_policy": {
                "type": "scheduled",
                "schedule": {"0": 4},
            },
        },
    )

    assert response.status == 400
    assert "scheduled beam width" in response.body["error"]["message"]
