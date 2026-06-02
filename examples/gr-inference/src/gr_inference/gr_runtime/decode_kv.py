"""Decode-step BeamKV write helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gr_inference.gr_kv import BeamKV
from gr_inference.gr_kv.layouts import Shape, shape_of


@dataclass(frozen=True)
class BeamKVWrite:
    """Metadata for writing one layer's decode-step K/V into BeamKV."""

    layer_idx: int
    step: int
    active_beam_width: int
    expected_layer_step_shape: Shape
    flat_offset: int


class BeamKVWriter:
    """Validate and write layer-local decode K/V into BeamKV."""

    def __init__(self, beam_kv: BeamKV) -> None:
        self.beam_kv = beam_kv

    def validate_layer_step(
        self,
        *,
        layer_idx: int,
        step: int,
        active_beam_width: int,
        k: Any,
        v: Any,
    ) -> BeamKVWrite:
        if layer_idx < 0 or layer_idx >= self.beam_kv.num_layers:
            raise ValueError(
                f"layer_idx={layer_idx} outside [0, {self.beam_kv.num_layers})"
            )
        step_write = self.beam_kv.validate_step(step, active_beam_width)
        expected = (
            self.beam_kv.batch_size,
            active_beam_width,
            self.beam_kv.num_kv_heads,
            self.beam_kv.head_dim,
        )
        if shape_of(k) != expected:
            raise ValueError(f"k step shape must be {expected}, got {shape_of(k)}")
        if shape_of(v) != expected:
            raise ValueError(f"v step shape must be {expected}, got {shape_of(v)}")
        return BeamKVWrite(
            layer_idx=layer_idx,
            step=step,
            active_beam_width=active_beam_width,
            expected_layer_step_shape=expected,
            flat_offset=step_write.flat_offset,
        )

    def write_layer_step(
        self,
        *,
        layer_idx: int,
        step: int,
        active_beam_width: int,
        k: Any,
        v: Any,
    ) -> BeamKVWrite:
        write = self.validate_layer_step(
            layer_idx=layer_idx,
            step=step,
            active_beam_width=active_beam_width,
            k=k,
            v=v,
        )
        key = self.beam_kv.key
        value = self.beam_kv.value
        if _try_cuda_beam_kv_write(
            key,
            value,
            k,
            v,
            layer_idx=layer_idx,
            step=step,
            active_beam_width=active_beam_width,
        ):
            return write
        if hasattr(key, "__setitem__"):
            key[layer_idx, :, step, :active_beam_width] = k
        if hasattr(value, "__setitem__"):
            value[layer_idx, :, step, :active_beam_width] = v
        return write


def _try_cuda_beam_kv_write(
    key: Any,
    value: Any,
    k: Any,
    v: Any,
    *,
    layer_idx: int,
    step: int,
    active_beam_width: int,
) -> bool:
    import os

    if os.environ.get("GR_INFERENCE_GR_TRTLLM_BEAM_KV_WRITE_JIT", "0") != "1":
        return False
    try:
        from gr_inference_trtllm_kernels.qwen3 import write_beam_kv_step
    except Exception:
        return False
    return write_beam_kv_step(
        key,
        value,
        k,
        v,
        layer_idx=layer_idx,
        step=step,
        active_beam_width=active_beam_width,
    )
