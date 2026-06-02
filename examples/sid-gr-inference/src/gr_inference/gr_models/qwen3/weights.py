# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-family HuggingFace weight naming adapter."""

from __future__ import annotations

from dataclasses import dataclass

from gr_inference.gr_models.loader import (
    CheckpointLoadPlan,
    CheckpointManifest,
    HFCheckpointLoader,
    TensorLoadRequest,
)
from gr_inference.gr_models.qwen3.config import Qwen3GRConfig


@dataclass(frozen=True)
class QwenLayerWeightNames:
    """HF tensor names needed by one dense Qwen decoder layer."""

    input_layernorm: str
    q_proj: str
    k_proj: str
    v_proj: str
    o_proj: str
    post_attention_layernorm: str
    gate_proj: str
    up_proj: str
    down_proj: str
    q_norm: str | None = None
    k_norm: str | None = None

    def required(self) -> tuple[str, ...]:
        return (
            self.input_layernorm,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.post_attention_layernorm,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        )

    def optional(self) -> tuple[str, ...]:
        return tuple(name for name in (self.q_norm, self.k_norm) if name is not None)


@dataclass(frozen=True)
class QwenModelWeightNames:
    """HF tensor names for a Qwen-family causal LM checkpoint."""

    embed_tokens: str
    final_norm: str
    lm_head: str | None
    layers: tuple[QwenLayerWeightNames, ...]

    def required(self) -> tuple[str, ...]:
        names = [self.embed_tokens, self.final_norm]
        if self.lm_head is not None:
            names.append(self.lm_head)
        for layer in self.layers:
            names.extend(layer.required())
        return tuple(names)

    def optional(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.lm_head is None:
            names.append("lm_head.weight")
        for layer in self.layers:
            names.extend(layer.optional())
        return tuple(names)


class Qwen3HFAdapter:
    """Adapter for dense Qwen2/Qwen3-style HuggingFace checkpoints.

    TRT-LLM uses the same family names and maps Qwen3 q/k layernorm as
    ``q_norm`` / ``k_norm``. Keeping the mapping here makes later Qwen-family
    variants an adapter concern instead of a runtime concern.
    """

    def __init__(self, config: Qwen3GRConfig) -> None:
        self.config = config

    @classmethod
    def from_manifest(cls, manifest: CheckpointManifest) -> "Qwen3HFAdapter":
        return cls(Qwen3GRConfig.from_hf_config(manifest.config))

    def weight_names(self) -> QwenModelWeightNames:
        layers = tuple(
            self.layer_weight_names(layer_idx)
            for layer_idx in range(self.config.num_layers)
        )
        return QwenModelWeightNames(
            embed_tokens="model.embed_tokens.weight",
            final_norm="model.norm.weight",
            lm_head=None if self.config.tie_word_embeddings else "lm_head.weight",
            layers=layers,
        )

    def layer_weight_names(self, layer_idx: int) -> QwenLayerWeightNames:
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise ValueError(
                f"layer_idx={layer_idx} outside [0, {self.config.num_layers})"
            )
        prefix = f"model.layers.{layer_idx}"
        return QwenLayerWeightNames(
            input_layernorm=f"{prefix}.input_layernorm.weight",
            q_proj=f"{prefix}.self_attn.q_proj.weight",
            k_proj=f"{prefix}.self_attn.k_proj.weight",
            v_proj=f"{prefix}.self_attn.v_proj.weight",
            o_proj=f"{prefix}.self_attn.o_proj.weight",
            post_attention_layernorm=f"{prefix}.post_attention_layernorm.weight",
            gate_proj=f"{prefix}.mlp.gate_proj.weight",
            up_proj=f"{prefix}.mlp.up_proj.weight",
            down_proj=f"{prefix}.mlp.down_proj.weight",
            q_norm=f"{prefix}.self_attn.q_norm.weight",
            k_norm=f"{prefix}.self_attn.k_norm.weight",
        )

    def validate_manifest(self, manifest: CheckpointManifest) -> None:
        self.load_plan().validate(manifest)

    def load_plan(
        self,
        *,
        pack_qkv: bool = True,
        pack_gate_up: bool = True,
    ) -> CheckpointLoadPlan:
        """Return the GR runtime load plan for Qwen-family dense checkpoints."""

        names = self.weight_names()
        requests: list[TensorLoadRequest] = [
            TensorLoadRequest("embed_tokens.weight", (names.embed_tokens,)),
            TensorLoadRequest("final_norm.weight", (names.final_norm,)),
        ]
        if names.lm_head is not None:
            requests.append(TensorLoadRequest("lm_head.weight", (names.lm_head,)))

        for layer_idx, layer in enumerate(names.layers):
            prefix = f"layers.{layer_idx}"
            requests.extend(
                [
                    TensorLoadRequest(
                        f"{prefix}.input_layernorm.weight",
                        (layer.input_layernorm,),
                    ),
                    TensorLoadRequest(
                        f"{prefix}.self_attn.o_proj.weight",
                        (layer.o_proj,),
                    ),
                    TensorLoadRequest(
                        f"{prefix}.post_attention_layernorm.weight",
                        (layer.post_attention_layernorm,),
                    ),
                    TensorLoadRequest(
                        f"{prefix}.mlp.down_proj.weight",
                        (layer.down_proj,),
                    ),
                ]
            )

            if pack_qkv:
                requests.append(
                    TensorLoadRequest(
                        f"{prefix}.self_attn.qkv_proj.weight",
                        (layer.q_proj, layer.k_proj, layer.v_proj),
                        transform="concat",
                        dim=0,
                    )
                )
            else:
                requests.extend(
                    [
                        TensorLoadRequest(
                            f"{prefix}.self_attn.q_proj.weight", (layer.q_proj,)
                        ),
                        TensorLoadRequest(
                            f"{prefix}.self_attn.k_proj.weight", (layer.k_proj,)
                        ),
                        TensorLoadRequest(
                            f"{prefix}.self_attn.v_proj.weight", (layer.v_proj,)
                        ),
                    ]
                )

            if pack_gate_up:
                requests.append(
                    TensorLoadRequest(
                        f"{prefix}.mlp.gate_up_proj.weight",
                        (layer.gate_proj, layer.up_proj),
                        transform="concat",
                        dim=0,
                    )
                )
            else:
                requests.extend(
                    [
                        TensorLoadRequest(
                            f"{prefix}.mlp.gate_proj.weight", (layer.gate_proj,)
                        ),
                        TensorLoadRequest(
                            f"{prefix}.mlp.up_proj.weight", (layer.up_proj,)
                        ),
                    ]
                )

            if layer.q_norm is not None:
                requests.append(
                    TensorLoadRequest(
                        f"{prefix}.self_attn.q_norm.weight",
                        (layer.q_norm,),
                        required=False,
                    )
                )
            if layer.k_norm is not None:
                requests.append(
                    TensorLoadRequest(
                        f"{prefix}.self_attn.k_norm.weight",
                        (layer.k_norm,),
                        required=False,
                    )
                )

        return CheckpointLoadPlan(tuple(requests))


def materialize_qwen3_checkpoint(
    model_dir: str, *, pack_qkv: bool = True, pack_gate_up: bool = True
):
    """Load and materialize a Qwen3 HF checkpoint into logical GR tensors.

    This is the high-level bridge from HF checkpoint layout to the logical names
    consumed by ``Qwen3GRModel.load_logical_weights``.
    """

    loader = HFCheckpointLoader(model_dir)
    manifest = loader.manifest()
    adapter = Qwen3HFAdapter.from_manifest(manifest)
    plan = adapter.load_plan(pack_qkv=pack_qkv, pack_gate_up=pack_gate_up)
    plan.validate(manifest)
    return plan.materialize(lambda name: loader.load_tensor(manifest, name))
