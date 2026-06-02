"""Prefill runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gr_inference.gr_kernels.prefill import PrefillAttention, PrefillAttentionInputs
from gr_inference.gr_kv.context_kv import ContextKV
from gr_inference.gr_kv.layouts import Shape, shape_of


@dataclass(frozen=True)
class ContextKVWrite:
    """Metadata for writing one layer's prefill K/V into ContextKV."""

    layer_idx: int
    expected_layer_shape: Shape


class ContextKVWriter:
    """Validate and optionally write layer-local K/V into framework ContextKV."""

    def __init__(self, context_kv: ContextKV) -> None:
        self.context_kv = context_kv

    def validate_layer(self, layer_idx: int, k: Any, v: Any) -> ContextKVWrite:
        if layer_idx < 0 or layer_idx >= self.context_kv.num_layers:
            raise ValueError(
                f"layer_idx={layer_idx} outside [0, {self.context_kv.num_layers})"
            )
        expected = self.context_kv.expected_layer_shape()
        if shape_of(k) != expected:
            raise ValueError(f"k layer shape must be {expected}, got {shape_of(k)}")
        if shape_of(v) != expected:
            raise ValueError(f"v layer shape must be {expected}, got {shape_of(v)}")
        return ContextKVWrite(layer_idx=layer_idx, expected_layer_shape=expected)

    def write_layer(self, layer_idx: int, k: Any, v: Any) -> ContextKVWrite:
        write = self.validate_layer(layer_idx, k, v)
        key = self.context_kv.key
        value = self.context_kv.value
        if hasattr(key, "__setitem__"):
            key[layer_idx] = k
        if hasattr(value, "__setitem__"):
            value[layer_idx] = v
        return write


@dataclass
class GRPrefillRunner:
    """Layer-local prefill runner.

    Model code owns Q/K/V projection and RoPE. This runner owns the backend call
    and the ContextKV write contract.
    """

    attention: PrefillAttention

    def run_layer(
        self,
        *,
        q: Any,
        k: Any,
        v: Any,
        context_kv: ContextKV,
        layer_idx: int,
        causal: bool = True,
        write_context_kv: bool = True,
    ) -> Any:
        writer = ContextKVWriter(context_kv)
        if write_context_kv:
            writer.write_layer(layer_idx, k, v)
        else:
            writer.validate_layer(layer_idx, k, v)

        output = self.attention(
            PrefillAttentionInputs(
                q=q,
                k=k,
                v=v,
                layer_idx=layer_idx,
                causal=causal,
            )
        )
        return output.hidden
