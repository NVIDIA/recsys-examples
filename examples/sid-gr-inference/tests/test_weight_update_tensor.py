# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""update_weights_from_tensor (the colocate path): facade/worker delegation,
the HTTP route, slime's chunked + empty-bucket partial semantics, the in-flight
guard (no old/new mixing), the pickle allowlist (RCE block), malformed-payload
400s, and the real CUDA-IPC cross-process round-trip.
"""

from __future__ import annotations

import base64
import json
import pickle

import pytest

from gr_inference.gr_serving import (
    GRInProcessServingFacade,
    GRServingRequest,
    GRServingWorker,
)

import _weight_update_helpers as h

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# Facade + Worker delegation
# --------------------------------------------------------------------------- #


def test_facade_and_worker_delegate_weight_update() -> None:
    model, cfg = h.model()
    executor = h.executor(model, cfg)
    facade = GRInProcessServingFacade(executor)
    worker = GRServingWorker(facade, autostart=False)

    payload = [h.serialize_bucket(h.hf_named_tensors(model, cfg))]
    r = facade.update_weights_from_tensor(
        payload, load_format="flattened_bucket", weight_version="f1", token_step=3
    )
    assert r["success"] is True
    assert facade.status()["weights"]["weight_version"] == "f1"

    payload2 = [h.serialize_bucket(h.hf_named_tensors(model, cfg))]
    r2 = worker.update_weights_from_tensor(
        payload2, load_format="flattened_bucket", weight_version="w1", token_step=4
    )
    assert r2["success"] is True
    assert facade.status()["weights"]["weight_version"] == "w1"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def test_http_update_weights_from_tensor() -> None:
    model, cfg = h.model()
    adapter = h.adapter(model, cfg)
    lm_before = model.lm_head.weight.data.clone()

    serialized = h.serialize_bucket(h.hf_named_tensors(model, cfg, perturb=0.5))
    body = json.dumps(
        {
            "serialized_named_tensors": [serialized],
            "load_format": "flattened_bucket",
            "weight_version": "t1",
            "token_step": 5,
        }
    ).encode("utf-8")
    resp = adapter.handle("POST", "/update_weights_from_tensor", body)
    assert resp.status == 200
    assert resp.body["success"] is True
    assert resp.body["tensors_loaded"] > 0
    assert not torch.allclose(model.lm_head.weight.data, lm_before, atol=1e-5)
    assert adapter.facade.facade.executor.weight_version == "t1"

    # missing payload -> 400; disabled -> 403
    assert (
        adapter.handle("POST", "/update_weights_from_tensor", json.dumps({}).encode())
        .status
        == 400
    )
    off = h.adapter(model, cfg, allow_weight_update=False)
    assert off.handle("POST", "/update_weights_from_tensor", body).status == 403

    routes = adapter.handle("GET", "/config").body["routes"]
    assert "POST /update_weights_from_tensor" in routes["weights"]


# --------------------------------------------------------------------------- #
# Colocate chunked / partial semantics
# (slime POSTs the model in multiple chunks + empty alignment buckets)
# --------------------------------------------------------------------------- #


def test_tensor_path_chunked_applies_full_model() -> None:
    """Slime POSTs in multiple chunks; each chunk must apply partially."""
    model, cfg = h.model()
    executor = h.executor(model, cfg)
    orig = {n: t.clone() for n, t in model.named_parameters()}

    all_named = h.hf_named_tensors(model, cfg, perturb=1.0)
    mid = len(all_named) // 2
    for chunk in (all_named[:mid], all_named[mid:]):
        serialized = h.serialize_bucket(chunk)
        result = executor.update_weights_from_tensor(
            [serialized], load_format="flattened_bucket", flush_cache=False
        )
        assert result["success"] is True

    # Both chunks together updated every parameter by +1.0.
    for name, tensor in model.named_parameters():
        assert torch.allclose(tensor, orig[name] + 1.0, atol=1e-5), name
    assert executor.weight_update_count == 2


def test_tensor_path_empty_bucket_is_noop() -> None:
    """Empty alignment buckets (slime _empty_flattened_tensor_data) must not error."""
    from gr_inference.gr_serving.weight_ipc import MultiprocessingSerializer

    model, cfg = h.model()
    executor = h.executor(model, cfg)
    embed_before = model.embed_tokens.weight.data.clone()

    serialized = MultiprocessingSerializer.serialize(
        {"flattened_tensor": torch.empty(0, dtype=torch.uint8), "metadata": []},
        output_str=True,
    )
    result = executor.update_weights_from_tensor(
        [serialized], load_format="flattened_bucket", flush_cache=False
    )
    assert result["success"] is True
    assert result["tensors_loaded"] == 0
    assert torch.allclose(model.embed_tokens.weight.data, embed_before)


def test_tensor_path_partial_chunk_is_atomic_on_bad_shape() -> None:
    """A bad-shape tensor fails the chunk's validation before any tensor is copied."""
    model, cfg = h.model()
    executor = h.executor(model, cfg)
    named = h.hf_named_tensors(model, cfg)
    bad = list(named)
    bad[0] = (bad[0][0], torch.zeros(1, 1))  # wrong shape for its name
    serialized = h.serialize_bucket(bad)

    sentinel = model.get_parameter_by_name("layers.0.ops.down_proj.weight").clone()
    with pytest.raises(ValueError):
        executor.update_weights_from_tensor(
            [serialized], load_format="flattened_bucket", flush_cache=False
        )
    # validate-then-copy: the chunk's other (valid) tensors were not written.
    assert torch.allclose(
        model.get_parameter_by_name("layers.0.ops.down_proj.weight"), sentinel
    )


def test_update_rejects_in_flight_without_abort() -> None:
    """Without pause/abort, updating with in-flight requests is refused."""
    model, cfg = h.model()
    executor = h.executor(model, cfg)
    executor.submit(
        GRServingRequest(
            request_id="r1",
            input_ids=torch.randint(0, cfg.vocab_size, (1, 4)),
            max_decode_steps=1,
            beam_width=1,
        )
    )

    # Guard fires before any disk IO, so a bogus path is fine.
    with pytest.raises(RuntimeError, match="in-flight"):
        executor.update_weights_from_disk("/nonexistent")
    serialized = h.serialize_bucket(h.hf_named_tensors(model, cfg))
    with pytest.raises(RuntimeError, match="in-flight"):
        executor.update_weights_from_tensor(
            [serialized], load_format="flattened_bucket"
        )

    # abort_all_requests=True bypasses the guard and clears in-flight first.
    result = executor.update_weights_from_tensor(
        [serialized], load_format="flattened_bucket", abort_all_requests=True
    )
    assert result["success"] is True
    assert result["num_aborted_requests"] >= 1


# --------------------------------------------------------------------------- #
# Security / error-handling on the tensor endpoint
# --------------------------------------------------------------------------- #

_RCE_HIT: list[bool] = []


def _rce_canary(*args, **kwargs):  # noqa: ANN001 - pickle probe target
    """If the allowlist is bypassed, unpickling would call this."""
    _RCE_HIT.append(True)
    return None


def test_tensor_endpoint_blocks_pickle_rce() -> None:
    """A crafted pickle must be blocked by the allowlist -> 400, never executed."""
    model, cfg = h.model()
    adapter = h.adapter(model, cfg)
    _RCE_HIT.clear()
    # A pickle whose unpickling would call _rce_canary(...) -- the allowlist must
    # reject its module before the callable is ever invoked.
    payload = {
        "serialized_named_tensors": [
            base64.b64encode(pickle.dumps((_rce_canary, ("x",)))).decode()
        ]
    }
    resp = adapter.handle(
        "POST", "/update_weights_from_tensor", json.dumps(payload).encode()
    )
    assert resp.status == 400
    assert _RCE_HIT == [], "allowlist bypassed: pickle RCE probe was executed"


def test_tensor_endpoint_malformed_payload_is_400() -> None:
    """Malformed payloads return 400 (not a connection reset)."""
    model, cfg = h.model()
    adapter = h.adapter(model, cfg)
    bad_base64 = adapter.handle(
        "POST",
        "/update_weights_from_tensor",
        json.dumps({"serialized_named_tensors": ["!!!not-base64!!!"]}).encode(),
    )
    assert bad_base64.status == 400
    not_pickle = adapter.handle(
        "POST",
        "/update_weights_from_tensor",
        json.dumps(
            {"serialized_named_tensors": [base64.b64encode(b"not a pickle").decode()]}
        ).encode(),
    )
    assert not_pickle.status == 400


# --------------------------------------------------------------------------- #
# Real CUDA IPC cross-process round-trip
# (exercises reduce_tensor's CUDA branch + device-UUID remap; CPU tests cannot)
# --------------------------------------------------------------------------- #


def _cuda_ipc_producer(conn):
    """Subprocess: serialize a CUDA tensor to an IPC handle, stay alive until told."""
    import torch
    from gr_inference.gr_serving.weight_ipc import (
        FlattenedTensorBucket,
        MultiprocessingSerializer,
        monkey_patch_torch_reductions,
    )

    monkey_patch_torch_reductions()
    torch.manual_seed(42)
    tensor = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    bucket = FlattenedTensorBucket(named_tensors=[("model.norm.weight", tensor)])
    serialized = MultiprocessingSerializer.serialize(
        {"flattened_tensor": bucket.flattened_tensor, "metadata": bucket.metadata},
        output_str=True,
    )
    conn.send(serialized)
    conn.recv()  # block so the process (and its IPC memory) stays alive


def test_cuda_ipc_roundtrip_cross_process() -> None:
    """Producer serializes a CUDA tensor as an IPC handle; this process reconstructs
    it via CUDA IPC and verifies equality. This is the only test that exercises
    reduce_tensor's CUDA branch and the device-UUID remap (CPU-only tests serialize
    bytes inline and never hit it)."""
    import multiprocessing as mp

    from gr_inference.gr_serving.weight_ipc import (
        monkey_patch_torch_reductions,
        reconstruct_named_tensors,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the IPC round-trip test")
    monkey_patch_torch_reductions()
    torch.manual_seed(42)
    expected = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)

    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_cuda_ipc_producer, args=(child,))
    proc.start()
    try:
        serialized = parent.recv()
        out = reconstruct_named_tensors([serialized], load_format="flattened_bucket")
        parent.send("done")  # release the producer
        assert out[0][0] == "model.norm.weight"
        assert torch.equal(out[0][1], expected)
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join()
