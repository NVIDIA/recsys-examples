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


def test_facade_and_worker_delegate_weight_update(tmp_path) -> None:
    model, cfg = _model()
    executor = _executor(model, cfg)
    facade = GRInProcessServingFacade(executor)
    worker = GRServingWorker(facade, autostart=False)

    r = facade.update_weights_from_tensor(
        {"norm.weight": model.norm.weight.data + 0.1},
        weight_version="f1",
        token_step=3,
    )
    assert r["success"] is True
    assert facade.status()["weights"]["weight_version"] == "f1"

    r2 = worker.update_weights_from_tensor(
        {"norm.weight": model.norm.weight.data + 0.2},
        weight_version="w1",
        token_step=4,
    )
    assert r2["success"] is True
    assert facade.status()["weights"]["weight_version"] == "w1"


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
