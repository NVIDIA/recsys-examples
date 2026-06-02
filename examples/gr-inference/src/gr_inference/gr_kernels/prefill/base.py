"""Prefill attention backend abstraction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gr_inference.gr_kv.layouts import Shape, shape_of


class MissingPrefillBackend(RuntimeError):
    """Raised when prefill attention is launched without a backend."""


@dataclass(frozen=True)
class PrefillAttentionInputs:
    """Layer-local dense causal prefill attention inputs.

    Shapes:
      q: [B, S, Hq, D]
      k: [B, S, Hkv, D]
      v: [B, S, Hkv, D]
    """

    q: Any
    k: Any
    v: Any
    layer_idx: int
    causal: bool = True

    @property
    def q_shape(self) -> Shape:
        return shape_of(self.q)

    @property
    def k_shape(self) -> Shape:
        return shape_of(self.k)

    @property
    def v_shape(self) -> Shape:
        return shape_of(self.v)


@dataclass(frozen=True)
class PrefillAttentionOutput:
    """Output from a prefill backend."""

    hidden: Any

    @property
    def hidden_shape(self) -> Shape:
        return shape_of(self.hidden)


PrefillBackend = Callable[[PrefillAttentionInputs], PrefillAttentionOutput | Any]


class PrefillAttention:
    """Shape validator and dispatch boundary for prefill attention.

    This keeps the framework independent from FlashAttention, TRT-LLM, vLLM, or
    any other implementation while still making the required ABI explicit.
    """

    def __init__(self, backend: PrefillBackend | None = None) -> None:
        self._backend = backend

    @property
    def has_backend(self) -> bool:
        return self._backend is not None

    def validate(self, inputs: PrefillAttentionInputs) -> None:
        q_shape = inputs.q_shape
        k_shape = inputs.k_shape
        v_shape = inputs.v_shape
        if len(q_shape) != 4:
            raise ValueError(f"q expects [B, S, Hq, D], got {q_shape}")
        if len(k_shape) != 4:
            raise ValueError(f"k expects [B, S, Hkv, D], got {k_shape}")
        if v_shape != k_shape:
            raise ValueError(f"k/v shapes differ: {k_shape} vs {v_shape}")

        batch, seq_len, num_q_heads, head_dim = q_shape
        k_batch, k_seq_len, num_kv_heads, k_head_dim = k_shape
        if (batch, seq_len, head_dim) != (k_batch, k_seq_len, k_head_dim):
            raise ValueError(
                "q and k must agree on batch, sequence length, and head_dim: "
                f"q={q_shape}, k={k_shape}"
            )
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"Hq={num_q_heads} must be divisible by Hkv={num_kv_heads}"
            )
        if inputs.layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")

    def __call__(self, inputs: PrefillAttentionInputs) -> PrefillAttentionOutput:
        self.validate(inputs)
        if self._backend is None:
            raise MissingPrefillBackend(
                "Prefill attention backend is not installed. Inject a backend "
                "such as TorchSDPAPrefillBackend, FlashAttention, or TRT-LLM."
            )
        output = self._backend(inputs)
        if not isinstance(output, PrefillAttentionOutput):
            output = PrefillAttentionOutput(hidden=output)
        if output.hidden_shape != inputs.q_shape:
            raise ValueError(
                f"prefill output shape must match q shape {inputs.q_shape}, "
                f"got {output.hidden_shape}"
            )
        return output
