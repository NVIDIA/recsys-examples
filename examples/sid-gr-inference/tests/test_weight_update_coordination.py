# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordination endpoints for the slime weight-sync flow
(pause_generation -> flush_cache -> update -> continue_generation +
get_weight_version): executor-level pause/continue/flush, the full HTTP slime
flow, route gating, and flush_cache returning 400 with in-flight requests.
"""

from __future__ import annotations

import json

import pytest

from gr_inference.gr_serving import GRServingRequest

import _weight_update_helpers as h

torch = pytest.importorskip("torch")


def _submit_in_flight(executor, cfg, request_id: str = "r1") -> None:
    executor.submit(
        GRServingRequest(
            request_id=request_id,
            input_ids=torch.randint(0, cfg.vocab_size, (1, 4)),
            max_decode_steps=1,
            beam_width=1,
        )
    )


def test_executor_pause_continue_flush_version() -> None:
    model, cfg = h.model()
    executor = h.executor(model, cfg)
    _submit_in_flight(executor, cfg)

    paused = executor.pause_generation(mode="abort")
    assert paused["paused"] is True
    assert paused["num_aborted_requests"] >= 1
    assert executor.is_paused is True
    assert len(executor.scheduler.decoding) == 0
    assert executor.status()["paused"] is True

    cont = executor.continue_generation()
    assert cont["paused"] is False
    assert executor.is_paused is False

    flush = executor.flush_cache()
    assert flush["success"] is True

    serialized = h.serialize_bucket(h.hf_named_tensors(model, cfg))
    executor.update_weights_from_tensor(
        [serialized], load_format="flattened_bucket", weight_version="rl-9"
    )
    assert executor.get_weight_version()["weight_version"] == "rl-9"


def test_http_slime_disk_coordination_flow(tmp_path) -> None:
    """Mirror slime actor_group._reload_rollout_weights_from_disk over HTTP."""
    model, cfg = h.model()
    adapter = h.adapter(model, cfg)
    checkpoint = h.write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
    executor = adapter.facade.facade.executor  # worker.facade.executor

    _submit_in_flight(executor, cfg)

    pause = adapter.handle("POST", "/pause_generation", json.dumps({"mode": "abort"}).encode())
    assert pause.status == 200 and pause.body["paused"] is True
    assert pause.body["num_aborted_requests"] >= 1

    flush = adapter.handle("POST", "/flush_cache", json.dumps({}).encode())
    assert flush.status == 200 and flush.body["success"] is True

    update = adapter.handle(
        "POST",
        "/update_weights_from_disk",
        json.dumps({"model_path": checkpoint, "weight_version": "rl-1"}).encode(),
    )
    assert update.status == 200 and update.body["success"] is True

    cont = adapter.handle("POST", "/continue_generation", json.dumps({}).encode())
    assert cont.status == 200 and cont.body["paused"] is False

    version = adapter.handle("GET", "/get_weight_version").body
    assert version["weight_version"] == "rl-1"


def test_http_coordination_endpoints_gated_and_validated(tmp_path) -> None:
    model, cfg = h.model()
    adapter = h.adapter(model, cfg, allow_weight_update=False)
    # pause/continue/get_weight_version are gated by allow_weight_update.
    assert adapter.handle("POST", "/pause_generation", b"{}").status == 403
    assert adapter.handle("POST", "/continue_generation", b"{}").status == 403
    assert adapter.handle("GET", "/get_weight_version").status == 403
    # flush_cache stays open (SGLang parity).
    assert adapter.handle("POST", "/flush_cache", b"{}").status == 200

    on = h.adapter(model, cfg, allow_weight_update=True)
    assert on.handle("POST", "/pause_generation", json.dumps({"mode": "bogus"}).encode()).status == 400
    routes = on.handle("GET", "/config").body["routes"]
    assert "POST /pause_generation" in routes["weights"]
    assert "GET /flush_cache" in routes["cache"]


def test_flush_cache_returns_400_with_in_flight() -> None:
    """SGLang semantics: flush returns non-200 with running/waiting requests."""
    model, cfg = h.model()
    adapter = h.adapter(model, cfg)
    executor = adapter.facade.facade.executor
    _submit_in_flight(executor, cfg)
    resp = adapter.handle("POST", "/flush_cache", json.dumps({}).encode())
    assert resp.status == 400
    assert resp.body["success"] is False

    # After pause(abort) clears in-flight, flush succeeds (200) -- the slime flow.
    adapter.handle("POST", "/pause_generation", json.dumps({"mode": "abort"}).encode())
    resp2 = adapter.handle("POST", "/flush_cache", json.dumps({}).encode())
    assert resp2.status == 200


def test_http_pause_rejects_new_inference_until_continue(tmp_path) -> None:
    """While paused for a weight update, new inference requests get 503
    (retryable) instead of silently queueing behind the swap, and /ready
    surfaces 'paused' so callers can drain traffic until continue_generation.

    Exercises the full path through GRServingWorker.is_paused -> facade ->
    executor, driven by the real /pause_generation endpoint.
    """
    model, cfg = h.model()
    adapter = h.adapter(model, cfg)

    adapter.handle("POST", "/pause_generation", json.dumps({"mode": "abort"}).encode())

    generate = adapter.handle("POST", "/generate", b"{}")
    assert generate.status == 503
    assert generate.body["error"]["code"] == "paused"
    assert generate.body["error"]["retryable"] is True

    submit = adapter.handle("POST", "/submit", b"{}")
    assert submit.status == 503
    assert submit.body["error"]["code"] == "paused"

    ready = adapter.handle("GET", "/ready")
    assert ready.body["ready"] is False
    assert "paused" in ready.body["reasons"]

    # continue_generation re-enables inference admission.
    adapter.handle("POST", "/continue_generation", json.dumps({}).encode())
    ready_after = adapter.handle("GET", "/ready")
    assert "paused" not in ready_after.body["reasons"]
