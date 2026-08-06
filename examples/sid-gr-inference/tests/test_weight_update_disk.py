# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""update_weights_from_disk: executor-level swap/version/cache/abort/guards,
config-incompatibility rejection, the no-engine guard, and the HTTP routes
(200 success, 400 missing path, 403 when allow_weight_update=False), plus the
get_weights_by_name diagnostic.
"""

from __future__ import annotations

import json

import pytest

from gr_inference.gr_serving import (
    GRContinuousScheduler,
    GRContinuousServingExecutor,
    GRInProcessServingFacade,
    GRServingRequest,
)

import _weight_update_helpers as h

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# Executor layer
# --------------------------------------------------------------------------- #


def test_update_weights_from_disk_swaps_and_versions(tmp_path) -> None:
    model, cfg = h.model()
    executor = h.executor(model, cfg)
    checkpoint = h.write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
    lm_before = model.lm_head.weight.data.clone()
    executor.prefill_cache_hits = 4
    executor.prefill_cache_misses = 2

    result = executor.update_weights_from_disk(
        checkpoint, weight_version="v1", token_step=7
    )

    assert result["success"] is True
    assert result["tensors_loaded"] > 0
    assert result["flushed_cache"] is True
    assert not torch.allclose(model.lm_head.weight.data, lm_before, atol=1e-5)
    assert executor.weight_version == "v1"
    assert executor.token_step == 7
    assert executor.weight_update_count == 1
    # Stale prefill-cache counters were reset.
    assert executor.prefill_cache_hits == 0
    assert executor.prefill_cache_misses == 0
    weights_status = executor.status()["weights"]
    assert weights_status["weight_version"] == "v1"


def test_update_weights_from_disk_rejects_incompatible_config(tmp_path) -> None:
    model, cfg = h.model()
    executor = h.executor(model, cfg)
    bad_config = h.hf_config(cfg)
    bad_config["hidden_size"] = 999
    checkpoint = h.write_checkpoint(
        model, cfg, tmp_path / "bad", override_config=bad_config
    )
    sentinel = model.embed_tokens.weight.data.clone()
    with pytest.raises(ValueError, match="structurally incompatible"):
        executor.update_weights_from_disk(checkpoint)
    # Failed update must not mutate weights.
    assert torch.allclose(model.embed_tokens.weight.data, sentinel)


def test_update_weights_from_disk_aborts_in_flight(tmp_path) -> None:
    model, cfg = h.model()
    executor = h.executor(model, cfg)
    checkpoint = h.write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
    executor.submit(
        GRServingRequest(
            request_id="r1",
            input_ids=torch.randint(0, cfg.vocab_size, (1, 4)),
            max_decode_steps=1,
            beam_width=1,
        )
    )
    result = executor.update_weights_from_disk(checkpoint, abort_all_requests=True)
    assert result["num_aborted_requests"] >= 1
    assert len(executor.scheduler.decoding) == 0


def test_executor_requires_model_for_weight_update(tmp_path) -> None:
    model, cfg = h.model()
    checkpoint = h.write_checkpoint(model, cfg, tmp_path / "ckpt")

    class _NoModel:
        pass

    executor = GRContinuousServingExecutor(
        engine=_NoModel(), scheduler=GRContinuousScheduler()
    )
    with pytest.raises(RuntimeError, match="serving engine"):
        executor.update_weights_from_disk(checkpoint)


def test_get_weights_by_name_returns_truncated_sample() -> None:
    model, cfg = h.model()
    executor = h.executor(model, cfg)
    sample = executor.get_weights_by_name("lm_head.weight", truncate_size=3)
    # SGLang shape: {"parameter": [first truncate_size rows along dim0]}.
    assert "parameter" in sample
    assert len(sample["parameter"]) == 3  # 3 rows
    assert len(sample["parameter"][0]) == cfg.hidden_size
    # Matches the model's actual lm_head rows.
    expected = model.lm_head.weight.detach()[:3].tolist()
    assert sample["parameter"] == expected


def test_facade_requires_engine_for_weight_update() -> None:
    facade = GRInProcessServingFacade(GRContinuousScheduler())
    with pytest.raises(RuntimeError, match="serving engine"):
        facade.update_weights_from_disk("/nonexistent")
    with pytest.raises(RuntimeError, match="serving engine"):
        facade.update_weights_from_tensor({"a": torch.zeros(1)})


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def test_http_update_weights_from_disk(tmp_path) -> None:
    model, cfg = h.model()
    adapter = h.adapter(model, cfg)
    checkpoint = h.write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
    lm_before = model.lm_head.weight.data.clone()

    body = json.dumps(
        {"model_path": checkpoint, "weight_version": "v1", "token_step": 9}
    ).encode("utf-8")
    resp = adapter.handle("POST", "/update_weights_from_disk", body)
    assert resp.status == 200
    assert resp.body["success"] is True
    assert resp.body["tensors_loaded"] > 0
    assert not torch.allclose(model.lm_head.weight.data, lm_before, atol=1e-5)

    status = adapter.handle("GET", "/status").body
    assert status["weights"]["weight_version"] == "v1"
    assert status["weights"]["token_step"] == 9

    routes = adapter.handle("GET", "/config").body["routes"]
    assert "POST /update_weights_from_disk" in routes["weights"]


def test_http_get_weights_by_name(tmp_path) -> None:
    model, cfg = h.model()
    adapter = h.adapter(model, cfg)
    resp = adapter.handle(
        "GET", "/get_weights_by_name?name=lm_head.weight&truncate_size=2"
    )
    assert resp.status == 200
    assert "parameter" in resp.body
    assert len(resp.body["parameter"]) == 2  # 2 rows along dim0
    assert len(resp.body["parameter"][0]) == cfg.hidden_size

    post = adapter.handle(
        "POST",
        "/get_weights_by_name",
        json.dumps({"name": "norm.weight", "truncate_size": 1}).encode("utf-8"),
    )
    assert post.status == 200
    # norm.weight is 1-D [hidden_size]; one "row" is a single scalar.
    assert len(post.body["parameter"]) == 1


def test_http_weight_routes_respect_allow_flag(tmp_path) -> None:
    model, cfg = h.model()
    adapter = h.adapter(model, cfg, allow_weight_update=False)
    checkpoint = h.write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
    body = json.dumps({"model_path": checkpoint}).encode("utf-8")
    assert adapter.handle("POST", "/update_weights_from_disk", body).status == 403
    assert (
        adapter.handle("GET", "/get_weights_by_name?name=lm_head.weight").status
        == 403
    )


def test_http_update_weights_from_disk_requires_model_path(tmp_path) -> None:
    model, cfg = h.model()
    adapter = h.adapter(model, cfg)
    resp = adapter.handle(
        "POST", "/update_weights_from_disk", json.dumps({}).encode("utf-8")
    )
    assert resp.status == 400
