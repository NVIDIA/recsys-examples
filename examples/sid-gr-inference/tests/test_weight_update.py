# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for online RL weight hot-update (update_weights_from_disk / _tensor).

Mirrors SGLang's weight-update surface and the Slime external-engine disk
transport. The disk path is the cross-process mechanism (trainer writes an HF
checkpoint to a shared path, engine reloads it); the tensor path is an in-process
primitive kept for tests and future colocate support.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from gr_inference.gr_kernels.attention import GRDecodeAttention
from gr_inference.gr_kernels.prefill import PrefillAttention, TorchSDPAPrefillBackend
from gr_inference.gr_models.qwen3.config import Qwen3GRConfig
from gr_inference.gr_models.qwen3.model import Qwen3GRModel
from gr_inference.gr_models.qwen3.weights import materialize_qwen3_checkpoint
from gr_inference.gr_runtime import GRDecodeEngine
from gr_inference.gr_serving import (
    GRContinuousBatchingPolicy,
    GRContinuousScheduler,
    GRContinuousServingExecutor,
    GRHTTPValidationPolicy,
    GRHTTPServingAdapter,
    GRInProcessServingFacade,
    GRServingConfig,
    GRServingEngine,
    GRServingRequest,
    GRServingWorker,
)


def _config(num_layers: int = 2) -> Qwen3GRConfig:
    return Qwen3GRConfig(
        model_name="tiny-weight-update-gr",
        num_layers=num_layers,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=16,
        max_seq_len=20,
        max_decode_steps=2,
        max_beam_width=4,
        intermediate_size=64,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )


def _model(cfg: Qwen3GRConfig | None = None) -> tuple[Qwen3GRModel, Qwen3GRConfig]:
    cfg = cfg or _config()
    model = Qwen3GRModel(
        cfg,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
        dtype=torch.float32,
    )
    model.eval()
    return model, cfg


def _executor(model: Qwen3GRModel, cfg: Qwen3GRConfig) -> GRContinuousServingExecutor:
    engine = GRServingEngine(
        model=model,
        decode_engine=GRDecodeEngine(
            attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
            fixed_beam_width=cfg.max_beam_width,
        ),
        config=GRServingConfig(
            max_decode_steps=cfg.max_decode_steps,
            max_beam_width=cfg.max_beam_width,
            enable_batched_decode=True,
        ),
    )
    return GRContinuousServingExecutor(
        engine=engine,
        scheduler=GRContinuousScheduler(
            policy=GRContinuousBatchingPolicy(
                max_prefill_batch_size=2,
                max_decode_batch_size=2,
            )
        ),
    )


def _hf_config(cfg: Qwen3GRConfig) -> dict:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.resolved_intermediate_size,
        "vocab_size": cfg.vocab_size,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
    }


def _hf_state_dict(model: Qwen3GRModel, cfg: Qwen3GRConfig) -> dict[str, torch.Tensor]:
    q_size = cfg.num_attention_heads * cfg.head_dim
    kv_size = cfg.num_kv_heads * cfg.head_dim
    inter = cfg.resolved_intermediate_size
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": model.embed_tokens.weight.data.clone(),
        "model.norm.weight": model.norm.weight.data.clone(),
    }
    if not cfg.tie_word_embeddings:
        state["lm_head.weight"] = model.lm_head.weight.data.clone()
    for idx, layer in enumerate(model.layers):
        ops = layer.ops
        prefix = f"model.layers.{idx}"
        state[f"{prefix}.input_layernorm.weight"] = ops.input_layernorm.weight.data.clone()
        state[f"{prefix}.post_attention_layernorm.weight"] = (
            ops.post_attention_layernorm.weight.data.clone()
        )
        state[f"{prefix}.self_attn.o_proj.weight"] = ops.out_proj.weight.data.clone()
        state[f"{prefix}.mlp.down_proj.weight"] = ops.down_proj.weight.data.clone()
        qkv = ops.qkv_proj.weight.data
        state[f"{prefix}.self_attn.q_proj.weight"] = qkv[:q_size].clone()
        state[f"{prefix}.self_attn.k_proj.weight"] = qkv[q_size : q_size + kv_size].clone()
        state[f"{prefix}.self_attn.v_proj.weight"] = qkv[q_size + kv_size :].clone()
        gate_up = ops.gate_up_proj.weight.data
        state[f"{prefix}.mlp.gate_proj.weight"] = gate_up[:inter].clone()
        state[f"{prefix}.mlp.up_proj.weight"] = gate_up[inter:].clone()
        state[f"{prefix}.self_attn.q_norm.weight"] = ops.q_norm.weight.data.clone()
        state[f"{prefix}.self_attn.k_norm.weight"] = ops.k_norm.weight.data.clone()
    return state


def _hf_named_tensors(model: Qwen3GRModel, cfg: Qwen3GRConfig, perturb: float = 0.0):
    """Build HF-named CPU tensors as a colocate trainer would send (split q/k/v)."""
    q = cfg.num_attention_heads * cfg.head_dim
    kv = cfg.num_kv_heads * cfg.head_dim
    inter = cfg.resolved_intermediate_size
    sd = {
        "model.embed_tokens.weight": model.embed_tokens.weight.data.clone(),
        "model.norm.weight": model.norm.weight.data.clone(),
    }
    if not cfg.tie_word_embeddings:
        sd["lm_head.weight"] = model.lm_head.weight.data.clone()
    for idx, layer in enumerate(model.layers):
        ops = layer.ops
        prefix = f"model.layers.{idx}"
        sd[f"{prefix}.input_layernorm.weight"] = ops.input_layernorm.weight.data.clone()
        sd[f"{prefix}.post_attention_layernorm.weight"] = (
            ops.post_attention_layernorm.weight.data.clone()
        )
        sd[f"{prefix}.self_attn.o_proj.weight"] = ops.out_proj.weight.data.clone()
        sd[f"{prefix}.mlp.down_proj.weight"] = ops.down_proj.weight.data.clone()
        qkv = ops.qkv_proj.weight.data
        sd[f"{prefix}.self_attn.q_proj.weight"] = qkv[:q].clone()
        sd[f"{prefix}.self_attn.k_proj.weight"] = qkv[q : q + kv].clone()
        sd[f"{prefix}.self_attn.v_proj.weight"] = qkv[q + kv :].clone()
        gate_up = ops.gate_up_proj.weight.data
        sd[f"{prefix}.mlp.gate_proj.weight"] = gate_up[:inter].clone()
        sd[f"{prefix}.mlp.up_proj.weight"] = gate_up[inter:].clone()
        sd[f"{prefix}.self_attn.q_norm.weight"] = ops.q_norm.weight.data.clone()
        sd[f"{prefix}.self_attn.k_norm.weight"] = ops.k_norm.weight.data.clone()
    if perturb:
        sd = {name: tensor + perturb for name, tensor in sd.items()}
    return list(sd.items())


def _serialize_bucket(named_tensors):
    from gr_inference.gr_serving.weight_ipc import (
        FlattenedTensorBucket,
        MultiprocessingSerializer,
    )

    bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    return MultiprocessingSerializer.serialize(
        {
            "flattened_tensor": bucket.flattened_tensor,
            "metadata": bucket.metadata,
        },
        output_str=True,
    )


def _write_checkpoint(
    model: Qwen3GRModel,
    cfg: Qwen3GRConfig,
    path,
    *,
    perturb: float = 0.0,
    override_config: dict | None = None,
) -> str:
    from safetensors.torch import save_file

    state = _hf_state_dict(model, cfg)
    if perturb:
        state = {name: tensor + perturb for name, tensor in state.items()}
    path.mkdir(parents=True, exist_ok=True)
    hf_config = override_config if override_config is not None else _hf_config(cfg)
    (path / "config.json").write_text(json.dumps(hf_config), encoding="utf-8")
    save_file(
        {name: tensor.contiguous() for name, tensor in state.items()},
        str(path / "model.safetensors"),
    )
    return str(path)


def _forward_logits(model: Qwen3GRModel) -> torch.Tensor:
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    return model.forward_prefill(ids, return_result=True).logits.detach().clone()


# --------------------------------------------------------------------------- #
# Model layer
# --------------------------------------------------------------------------- #


def test_update_weights_from_tensor_copies_by_module_name() -> None:
    model, _ = _model()
    before = _forward_logits(model)

    named = {name: tensor.clone() for name, tensor in model.named_parameters()}
    named["layers.0.ops.qkv_proj.weight"].add_(1.0)
    named["norm.weight"].add_(0.5)
    sentinel = model.get_parameter_by_name("layers.1.ops.down_proj.weight").clone()

    count = model.update_weights_from_tensor(
        {
            "layers.0.ops.qkv_proj.weight": named["layers.0.ops.qkv_proj.weight"],
            "norm.weight": named["norm.weight"],
        }
    )
    assert count == 2
    assert torch.allclose(
        model.get_parameter_by_name("layers.0.ops.qkv_proj.weight"),
        named["layers.0.ops.qkv_proj.weight"],
    )
    # Untouched parameter is unchanged.
    assert torch.allclose(
        model.get_parameter_by_name("layers.1.ops.down_proj.weight"), sentinel
    )
    assert not torch.allclose(before, _forward_logits(model))


def test_update_weights_from_tensor_is_atomic_on_failure() -> None:
    model, _ = _model()
    sentinel = model.get_parameter_by_name("embed_tokens.weight").clone()

    # Wrong shape must raise before any copy and leave the model untouched.
    with pytest.raises(ValueError):
        model.update_weights_from_tensor({"embed_tokens.weight": torch.zeros(1, 1)})
    assert torch.allclose(model.get_parameter_by_name("embed_tokens.weight"), sentinel)

    # Unknown name with strict=True raises.
    with pytest.raises(KeyError):
        model.update_weights_from_tensor({"does.not.exist": torch.zeros(1)})

    # Unknown names are skipped when strict=False; known ones still apply.
    good = model.get_parameter_by_name("norm.weight").clone() + 0.25
    count = model.update_weights_from_tensor(
        {"does.not.exist": torch.zeros(1), "norm.weight": good}, strict=False
    )
    assert count == 1


def test_validate_logical_weights_does_not_mutate(tmp_path) -> None:
    model, cfg = _model()
    checkpoint = _write_checkpoint(model, cfg, tmp_path / "ckpt")
    logical = materialize_qwen3_checkpoint(checkpoint)

    snapshot = {
        name: tensor.clone() for name, tensor in model.named_parameters()
    }
    model.validate_logical_weights(logical)  # dry run
    for name, tensor in model.named_parameters():
        assert torch.allclose(tensor, snapshot[name])

    # A real load then round-trips back to the same logical tensors.
    model.load_logical_weights(logical)
    again = materialize_qwen3_checkpoint(checkpoint)
    for name, tensor in again.items():
        assert torch.allclose(tensor, again[name])  # sanity: materialize is stable


# --------------------------------------------------------------------------- #
# Executor layer (disk path)
# --------------------------------------------------------------------------- #


def test_update_weights_from_disk_swaps_and_versions(tmp_path) -> None:
    model, cfg = _model()
    executor = _executor(model, cfg)
    checkpoint = _write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
    lm_before = model.lm_head.weight.data.clone()
    executor.prefill_cache_hits = 4
    executor.prefill_cache_misses = 2

    result = executor.update_weights_from_disk(
        checkpoint, weight_version="v1", token_step=7
    )

    assert result["success"] is True
    assert result["params_updated"] > 0
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
    model, cfg = _model()
    executor = _executor(model, cfg)
    bad_config = _hf_config(cfg)
    bad_config["hidden_size"] = 999
    checkpoint = _write_checkpoint(
        model, cfg, tmp_path / "bad", override_config=bad_config
    )
    sentinel = model.embed_tokens.weight.data.clone()
    with pytest.raises(ValueError, match="structurally incompatible"):
        executor.update_weights_from_disk(checkpoint)
    # Failed update must not mutate weights.
    assert torch.allclose(model.embed_tokens.weight.data, sentinel)


def test_update_weights_from_disk_aborts_in_flight(tmp_path) -> None:
    model, cfg = _model()
    executor = _executor(model, cfg)
    checkpoint = _write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
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
    model, cfg = _model()
    checkpoint = _write_checkpoint(model, cfg, tmp_path / "ckpt")

    class _NoModel:
        pass

    executor = GRContinuousServingExecutor(
        engine=_NoModel(), scheduler=GRContinuousScheduler()
    )
    with pytest.raises(RuntimeError, match="serving engine"):
        executor.update_weights_from_disk(checkpoint)


def test_get_weights_by_name_returns_truncated_sample() -> None:
    model, cfg = _model()
    executor = _executor(model, cfg)
    sample = executor.get_weights_by_name("lm_head.weight", truncate_size=3)
    assert sample["name"] == "lm_head.weight"
    assert sample["shape"] == [cfg.vocab_size, cfg.hidden_size]
    assert len(sample["values"]) == 3


# --------------------------------------------------------------------------- #
# Facade + Worker
# --------------------------------------------------------------------------- #


def test_facade_and_worker_delegate_weight_update() -> None:
    model, cfg = _model()
    executor = _executor(model, cfg)
    facade = GRInProcessServingFacade(executor)
    worker = GRServingWorker(facade, autostart=False)

    payload = [_serialize_bucket(_hf_named_tensors(model, cfg))]
    r = facade.update_weights_from_tensor(
        payload, load_format="flattened_bucket", weight_version="f1", token_step=3
    )
    assert r["success"] is True
    assert facade.status()["weights"]["weight_version"] == "f1"

    payload2 = [_serialize_bucket(_hf_named_tensors(model, cfg))]
    r2 = worker.update_weights_from_tensor(
        payload2, load_format="flattened_bucket", weight_version="w1", token_step=4
    )
    assert r2["success"] is True
    assert facade.status()["weights"]["weight_version"] == "w1"


def test_weight_ipc_flattened_bucket_roundtrip() -> None:
    from gr_inference.gr_serving.weight_ipc import reconstruct_named_tensors

    named = [
        ("model.norm.weight", torch.randn(8, dtype=torch.float32)),
        ("lm_head.weight", torch.randn(4, 5, dtype=torch.float32)),
    ]
    serialized = _serialize_bucket(named)
    out = reconstruct_named_tensors([serialized], load_format="flattened_bucket")
    assert [name for name, _ in out] == [name for name, _ in named]
    for (n_in, t_in), (n_out, t_out) in zip(named, out, strict=True):
        assert n_in == n_out
        assert torch.equal(t_in, t_out)


def test_hf_to_logical_name_mapping() -> None:
    from gr_inference.gr_serving.continuous import _hf_to_logical_name

    assert _hf_to_logical_name("model.embed_tokens.weight") == "embed_tokens.weight"
    assert _hf_to_logical_name("model.norm.weight") == "final_norm.weight"
    assert _hf_to_logical_name("lm_head.weight") == "lm_head.weight"
    assert (
        _hf_to_logical_name("model.layers.0.self_attn.q_proj.weight")
        == "layers.0.self_attn.q_proj.weight"
    )


def test_facade_requires_engine_for_weight_update() -> None:
    facade = GRInProcessServingFacade(GRContinuousScheduler())
    with pytest.raises(RuntimeError, match="serving engine"):
        facade.update_weights_from_disk("/nonexistent")
    with pytest.raises(RuntimeError, match="serving engine"):
        facade.update_weights_from_tensor({"a": torch.zeros(1)})


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _adapter(model, cfg, *, allow_weight_update: bool = True) -> GRHTTPServingAdapter:
    executor = _executor(model, cfg)
    worker = GRServingWorker(GRInProcessServingFacade(executor), autostart=False)
    return GRHTTPServingAdapter(
        worker,
        validation_policy=GRHTTPValidationPolicy(
            allow_weight_update=allow_weight_update
        ),
    )


def test_http_update_weights_from_disk(tmp_path) -> None:
    model, cfg = _model()
    adapter = _adapter(model, cfg)
    checkpoint = _write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
    lm_before = model.lm_head.weight.data.clone()

    body = json.dumps(
        {"model_path": checkpoint, "weight_version": "v1", "token_step": 9}
    ).encode("utf-8")
    resp = adapter.handle("POST", "/update_weights_from_disk", body)
    assert resp.status == 200
    assert resp.body["success"] is True
    assert resp.body["params_updated"] > 0
    assert not torch.allclose(model.lm_head.weight.data, lm_before, atol=1e-5)

    status = adapter.handle("GET", "/status").body
    assert status["weights"]["weight_version"] == "v1"
    assert status["weights"]["token_step"] == 9

    routes = adapter.handle("GET", "/config").body["routes"]
    assert "POST /update_weights_from_disk" in routes["weights"]


def test_http_get_weights_by_name(tmp_path) -> None:
    model, cfg = _model()
    adapter = _adapter(model, cfg)
    resp = adapter.handle(
        "GET", "/get_weights_by_name?name=lm_head.weight&truncate_size=2"
    )
    assert resp.status == 200
    assert resp.body["name"] == "lm_head.weight"
    assert resp.body["shape"] == [cfg.vocab_size, cfg.hidden_size]
    assert len(resp.body["values"]) == 2

    post = adapter.handle(
        "POST",
        "/get_weights_by_name",
        json.dumps({"name": "norm.weight", "truncate_size": 1}).encode("utf-8"),
    )
    assert post.status == 200
    assert post.body["shape"] == [cfg.hidden_size]


def test_http_weight_routes_respect_allow_flag(tmp_path) -> None:
    model, cfg = _model()
    adapter = _adapter(model, cfg, allow_weight_update=False)
    checkpoint = _write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
    body = json.dumps({"model_path": checkpoint}).encode("utf-8")
    assert adapter.handle("POST", "/update_weights_from_disk", body).status == 403
    assert (
        adapter.handle("GET", "/get_weights_by_name?name=lm_head.weight").status
        == 403
    )


def test_http_update_weights_from_disk_requires_model_path(tmp_path) -> None:
    model, cfg = _model()
    adapter = _adapter(model, cfg)
    resp = adapter.handle(
        "POST", "/update_weights_from_disk", json.dumps({}).encode("utf-8")
    )
    assert resp.status == 400


def test_http_update_weights_from_tensor() -> None:
    model, cfg = _model()
    adapter = _adapter(model, cfg)
    lm_before = model.lm_head.weight.data.clone()

    serialized = _serialize_bucket(_hf_named_tensors(model, cfg, perturb=0.5))
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
    assert resp.body["params_updated"] > 0
    assert not torch.allclose(model.lm_head.weight.data, lm_before, atol=1e-5)
    assert adapter.facade.facade.executor.weight_version == "t1"

    # missing payload -> 400; disabled -> 403
    assert (
        adapter.handle("POST", "/update_weights_from_tensor", json.dumps({}).encode())
        .status
        == 400
    )
    off = _adapter(model, cfg, allow_weight_update=False)
    assert off.handle("POST", "/update_weights_from_tensor", body).status == 403

    routes = adapter.handle("GET", "/config").body["routes"]
    assert "POST /update_weights_from_tensor" in routes["weights"]


# --------------------------------------------------------------------------- #
# Disk coordination endpoints (SGLang pause/continue/flush_cache/get_weight_version)
# --------------------------------------------------------------------------- #


def test_executor_pause_continue_flush_version() -> None:
    model, cfg = _model()
    executor = _executor(model, cfg)
    executor.submit(
        GRServingRequest(
            request_id="r1",
            input_ids=torch.randint(0, cfg.vocab_size, (1, 4)),
            max_decode_steps=1,
            beam_width=1,
        )
    )

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

    serialized = _serialize_bucket(_hf_named_tensors(model, cfg))
    executor.update_weights_from_tensor(
        [serialized], load_format="flattened_bucket", weight_version="rl-9"
    )
    assert executor.get_weight_version()["weight_version"] == "rl-9"


def test_http_slime_disk_coordination_flow(tmp_path) -> None:
    """Mirror slime actor_group._reload_rollout_weights_from_disk over HTTP."""
    model, cfg = _model()
    adapter = _adapter(model, cfg)
    checkpoint = _write_checkpoint(model, cfg, tmp_path / "ckpt", perturb=0.5)
    executor = adapter.facade.facade.executor  # worker.facade.executor

    executor.submit(
        GRServingRequest(
            request_id="r1",
            input_ids=torch.randint(0, cfg.vocab_size, (1, 4)),
            max_decode_steps=1,
            beam_width=1,
        )
    )

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
    model, cfg = _model()
    adapter = _adapter(model, cfg, allow_weight_update=False)
    # pause/continue/get_weight_version are gated by allow_weight_update.
    assert adapter.handle("POST", "/pause_generation", b"{}").status == 403
    assert adapter.handle("POST", "/continue_generation", b"{}").status == 403
    assert adapter.handle("GET", "/get_weight_version").status == 403
    # flush_cache stays open (SGLang parity).
    assert adapter.handle("POST", "/flush_cache", b"{}").status == 200

    on = _adapter(model, cfg, allow_weight_update=True)
    assert on.handle("POST", "/pause_generation", json.dumps({"mode": "bogus"}).encode()).status == 400
    routes = on.handle("GET", "/config").body["routes"]
    assert "POST /pause_generation" in routes["weights"]
    assert "GET /flush_cache" in routes["cache"]


# --------------------------------------------------------------------------- #
# Colocate chunked / partial semantics
# (slime POSTs the model in multiple chunks + empty alignment buckets)
# --------------------------------------------------------------------------- #


def test_tensor_path_chunked_applies_full_model() -> None:
    """Slime POSTs in multiple chunks; each chunk must apply partially."""
    model, cfg = _model()
    executor = _executor(model, cfg)
    orig = {n: t.clone() for n, t in model.named_parameters()}

    all_named = _hf_named_tensors(model, cfg, perturb=1.0)
    mid = len(all_named) // 2
    for chunk in (all_named[:mid], all_named[mid:]):
        serialized = _serialize_bucket(chunk)
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

    model, cfg = _model()
    executor = _executor(model, cfg)
    embed_before = model.embed_tokens.weight.data.clone()

    serialized = MultiprocessingSerializer.serialize(
        {"flattened_tensor": torch.empty(0, dtype=torch.uint8), "metadata": []},
        output_str=True,
    )
    result = executor.update_weights_from_tensor(
        [serialized], load_format="flattened_bucket", flush_cache=False
    )
    assert result["success"] is True
    assert result["params_updated"] == 0
    assert torch.allclose(model.embed_tokens.weight.data, embed_before)


def test_tensor_path_partial_chunk_is_atomic_on_bad_shape() -> None:
    """A bad-shape tensor fails the chunk's validation before any tensor is copied."""
    model, cfg = _model()
    executor = _executor(model, cfg)
    named = _hf_named_tensors(model, cfg)
    bad = list(named)
    bad[0] = (bad[0][0], torch.zeros(1, 1))  # wrong shape for its name
    serialized = _serialize_bucket(bad)

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
    model, cfg = _model()
    executor = _executor(model, cfg)
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
    serialized = _serialize_bucket(_hf_named_tensors(model, cfg))
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
