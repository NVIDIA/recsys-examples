# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known Qwen3 dense-model variants used by GR tooling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Qwen3VariantSpec:
    canonical_name: str
    aliases: tuple[str, ...]
    default_model_dir: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    tie_word_embeddings: bool

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_size(self) -> int:
        return self.q_size + 2 * self.kv_size

    @property
    def gate_up_size(self) -> int:
        return 2 * self.intermediate_size

    def to_gr_config(self, **overrides: Any):
        from gr_inference.gr_models.qwen3.config import Qwen3GRConfig

        values = {
            "model_name": self.canonical_name,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
        values.update(overrides)
        return Qwen3GRConfig(**values)

    def matches_config(self, config: Any) -> bool:
        return (
            _int_config_value(config, "num_layers", "num_hidden_layers")
            == self.num_layers
            and _int_config_value(config, "hidden_size") == self.hidden_size
            and _int_config_value(config, "intermediate_size") == self.intermediate_size
            and _int_config_value(config, "num_attention_heads")
            == self.num_attention_heads
            and _int_config_value(config, "num_kv_heads", "num_key_value_heads")
            == self.num_kv_heads
            and _int_config_value(config, "head_dim") == self.head_dim
            and _int_config_value(config, "vocab_size") == self.vocab_size
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "default_model_dir": self.default_model_dir,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "q_size": self.q_size,
            "kv_size": self.kv_size,
            "qkv_size": self.qkv_size,
            "gate_up_size": self.gate_up_size,
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


QWEN3_0_6B = Qwen3VariantSpec(
    canonical_name="qwen3-0.6b",
    aliases=("qwen3-0.6b", "qwen3-0_6b", "qwen/qwen3-0.6b", "0.6b"),
    default_model_dir="models/Qwen3-0.6B",
    num_layers=28,
    hidden_size=1024,
    intermediate_size=3072,
    num_attention_heads=16,
    num_kv_heads=8,
    head_dim=128,
    vocab_size=151936,
    tie_word_embeddings=True,
)

QWEN3_1_7B = Qwen3VariantSpec(
    canonical_name="qwen3-1.7b",
    aliases=("qwen3-1.7b", "qwen3-1_7b", "qwen/qwen3-1.7b", "1.7b"),
    default_model_dir="models/Qwen3-1.7B",
    num_layers=28,
    hidden_size=2048,
    intermediate_size=6144,
    num_attention_heads=16,
    num_kv_heads=8,
    head_dim=128,
    vocab_size=151936,
    tie_word_embeddings=True,
)

KNOWN_QWEN3_VARIANTS: tuple[Qwen3VariantSpec, ...] = (QWEN3_0_6B, QWEN3_1_7B)
DEFAULT_QWEN3_VARIANT = QWEN3_1_7B.canonical_name
DEFAULT_QWEN3_MODEL_ID = "Qwen/Qwen3-1.7B"


def get_qwen3_variant(name: str | None) -> Qwen3VariantSpec:
    normalized = _normalize_variant_name(name or DEFAULT_QWEN3_VARIANT)
    for spec in KNOWN_QWEN3_VARIANTS:
        if normalized == _normalize_variant_name(spec.canonical_name):
            return spec
        if normalized in {_normalize_variant_name(alias) for alias in spec.aliases}:
            return spec
    known = ", ".join(spec.canonical_name for spec in KNOWN_QWEN3_VARIANTS)
    raise ValueError(f"unknown Qwen3 variant {name!r}; known variants: {known}")


def identify_qwen3_variant(config: Any) -> Qwen3VariantSpec | None:
    for spec in KNOWN_QWEN3_VARIANTS:
        if spec.matches_config(config):
            return spec
    return None


def resolve_qwen3_model_dir(
    model_dir: str | None = None,
    *,
    variant: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    if model_dir:
        return model_dir
    env = os.environ if env is None else env
    spec = get_qwen3_variant(
        variant or env.get("GR_QWEN3_MODEL_VARIANT") or env.get("QWEN3_MODEL_VARIANT")
    )
    for key in (*_variant_model_dir_env_names(spec), "GR_QWEN3_MODEL_DIR"):
        value = env.get(key)
        if value:
            return value
    return spec.default_model_dir


def _variant_model_dir_env_names(spec: Qwen3VariantSpec) -> tuple[str, str]:
    key = spec.canonical_name.upper().replace("-", "_").replace(".", "_")
    return f"GR_{key}_MODEL_DIR", f"{key}_MODEL_DIR"


def _normalize_variant_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _config_get(config: Any, *names: str) -> Any:
    if isinstance(config, Mapping):
        for name in names:
            if name in config:
                return config[name]
        return None
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return None


def _int_config_value(config: Any, *names: str) -> int | None:
    value = _config_get(config, *names)
    return None if value is None else int(value)
