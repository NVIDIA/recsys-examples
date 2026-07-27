# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures/factories for the weight-hot-update tests.

This module is imported only after ``pytest.importorskip("torch")`` in each test
file, so it can ``import torch`` unconditionally. It is NOT collected as tests
(underscore prefix, no ``test_`` name).

Builds a tiny CPU Qwen3 model + continuous-serving stack, HF-named tensor sets
(as a slime trainer would push them, with split q/k/v), flattened-bucket
serialized payloads, and on-disk HF checkpoints (real safetensors) for the
disk path.
"""

from __future__ import annotations

import json

import torch

from gr_inference.gr_kernels.attention import GRDecodeAttention
from gr_inference.gr_kernels.prefill import PrefillAttention, TorchSDPAPrefillBackend
from gr_inference.gr_models.qwen3.config import Qwen3GRConfig
from gr_inference.gr_models.qwen3.model import Qwen3GRModel
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
    GRServingWorker,
)


def config(num_layers: int = 2) -> Qwen3GRConfig:
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


def model(cfg: Qwen3GRConfig | None = None) -> tuple[Qwen3GRModel, Qwen3GRConfig]:
    cfg = cfg or config()
    mdl = Qwen3GRModel(
        cfg,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
        dtype=torch.float32,
    )
    mdl.eval()
    return mdl, cfg


def executor(mdl: Qwen3GRModel, cfg: Qwen3GRConfig) -> GRContinuousServingExecutor:
    engine = GRServingEngine(
        model=mdl,
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


def adapter(mdl, cfg, *, allow_weight_update: bool = True) -> GRHTTPServingAdapter:
    """In-process HTTP adapter over a worker+facade+executor (no socket bound)."""
    worker = GRServingWorker(GRInProcessServingFacade(executor(mdl, cfg)), autostart=False)
    return GRHTTPServingAdapter(
        worker,
        validation_policy=GRHTTPValidationPolicy(allow_weight_update=allow_weight_update),
    )


def hf_config(cfg: Qwen3GRConfig) -> dict:
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


def hf_state_dict(mdl: Qwen3GRModel, cfg: Qwen3GRConfig) -> dict[str, torch.Tensor]:
    """The model's weights as HF-named tensors (split q/k/v, gate/up)."""
    q_size = cfg.num_attention_heads * cfg.head_dim
    kv_size = cfg.num_kv_heads * cfg.head_dim
    inter = cfg.resolved_intermediate_size
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": mdl.embed_tokens.weight.data.clone(),
        "model.norm.weight": mdl.norm.weight.data.clone(),
    }
    if not cfg.tie_word_embeddings:
        state["lm_head.weight"] = mdl.lm_head.weight.data.clone()
    for idx, layer in enumerate(mdl.layers):
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


def hf_named_tensors(mdl: Qwen3GRModel, cfg: Qwen3GRConfig, perturb: float = 0.0):
    """HF-named CPU tensors as a colocate trainer would send (split q/k/v)."""
    sd = hf_state_dict(mdl, cfg)
    if perturb:
        sd = {name: tensor + perturb for name, tensor in sd.items()}
    return list(sd.items())


def serialize_bucket(named_tensors) -> str:
    """Flatten + ForkingPickler-serialize a bucket to a base64 str (IPC payload)."""
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


def write_checkpoint(
    mdl: Qwen3GRModel,
    cfg: Qwen3GRConfig,
    path,
    *,
    perturb: float = 0.0,
    override_config: dict | None = None,
) -> str:
    from safetensors.torch import save_file

    state = hf_state_dict(mdl, cfg)
    if perturb:
        state = {name: tensor + perturb for name, tensor in state.items()}
    path.mkdir(parents=True, exist_ok=True)
    hf_cfg = override_config if override_config is not None else hf_config(cfg)
    (path / "config.json").write_text(json.dumps(hf_cfg), encoding="utf-8")
    save_file(
        {name: tensor.contiguous() for name, tensor in state.items()},
        str(path / "model.safetensors"),
    )
    return str(path)


def forward_logits(mdl: Qwen3GRModel) -> torch.Tensor:
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    return mdl.forward_prefill(ids, return_result=True).logits.detach().clone()
