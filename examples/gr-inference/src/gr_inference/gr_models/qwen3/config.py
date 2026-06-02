"""Qwen3 GR model metadata used by the MVP runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3GRConfig:
    """Qwen3-family GR shape metadata.

    The defaults preserve the original 0.6B development target. Real checkpoint
    paths should construct this from HuggingFace ``config.json`` so larger Qwen3
    variants such as 1.7B use their own hidden and MLP sizes.
    """

    model_name: str = "Qwen3-0.6B-GR"
    num_layers: int = 28
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    max_context_len: int = 4700
    max_seq_len: int = 4900
    max_decode_steps: int = 3
    max_beam_width: int = 128
    intermediate_size: int | None = None
    vocab_size: int | None = None
    tie_word_embeddings: bool = False
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    dtype: str = "bf16"

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
    def resolved_intermediate_size(self) -> int:
        return self.intermediate_size or self.hidden_size * 4

    @property
    def gate_up_size(self) -> int:
        return 2 * self.resolved_intermediate_size

    @property
    def q_heads_per_kv_head(self) -> int:
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        return self.num_attention_heads // self.num_kv_heads

    def validate_decode_shape(self, decode_steps: int, beam_width: int) -> None:
        if decode_steps <= 0:
            raise ValueError("decode_steps must be positive")
        if decode_steps > self.max_decode_steps:
            raise ValueError(
                f"decode_steps={decode_steps} exceeds max_decode_steps={self.max_decode_steps}"
            )
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if beam_width > self.max_beam_width:
            raise ValueError(
                f"beam_width={beam_width} exceeds max_beam_width={self.max_beam_width}"
            )

    @classmethod
    def from_hf_config(
        cls,
        hf_config: dict,
        *,
        max_context_len: int = 4700,
        max_seq_len: int = 4900,
        max_decode_steps: int = 3,
        max_beam_width: int = 128,
        dtype: str = "bf16",
    ) -> "Qwen3GRConfig":
        return cls(
            model_name=str(hf_config.get("_name_or_path", "Qwen3-GR")),
            num_layers=int(hf_config["num_hidden_layers"]),
            hidden_size=int(hf_config["hidden_size"]),
            num_attention_heads=int(hf_config["num_attention_heads"]),
            num_kv_heads=int(
                hf_config.get("num_key_value_heads", hf_config["num_attention_heads"])
            ),
            head_dim=int(
                hf_config.get(
                    "head_dim",
                    hf_config["hidden_size"] // hf_config["num_attention_heads"],
                )
            ),
            max_context_len=max_context_len,
            max_seq_len=max_seq_len,
            max_decode_steps=max_decode_steps,
            max_beam_width=max_beam_width,
            intermediate_size=(
                int(hf_config["intermediate_size"])
                if "intermediate_size" in hf_config
                else None
            ),
            vocab_size=int(hf_config["vocab_size"]) if "vocab_size" in hf_config else None,
            tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
            rms_norm_eps=float(hf_config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(hf_config.get("rope_theta", 1_000_000.0)),
            dtype=dtype,
        )
