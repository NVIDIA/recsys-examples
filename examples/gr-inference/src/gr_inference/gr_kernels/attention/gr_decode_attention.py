"""Wrapper contract for the existing GR decode attention kernel."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gr_inference.gr_kv.beam_kv import BeamKV
from gr_inference.gr_kv.beam_path import BeamPath
from gr_inference.gr_kv.context_kv import ContextKV
from gr_inference.gr_kv.layouts import Shape, shape_of


class MissingKernelBackend(RuntimeError):
    """Raised when runtime tries to launch before a kernel backend is installed."""


@dataclass(frozen=True)
class GRDecodeAttentionInputs:
    """Inputs passed from runtime to the decode attention backend."""

    q: Any
    context_kv: ContextKV
    beam_kv: BeamKV
    beam_path: BeamPath
    layer_idx: int
    step: int
    active_beam_width: int
    topk_indices: Any | None = None
    decode_nums: int | None = None
    return_lse: bool = False
    backend_name: str = "dsl"

    @property
    def q_shape(self) -> Shape:
        return shape_of(self.q)


KernelBackend = Callable[[GRDecodeAttentionInputs], Any]


class GRDecodeAttention:
    """Validates and dispatches to the existing decode attention kernel.

    The real CUDA/Triton/extension binding should be injected as ``backend``.
    Tests use a Python callable with the same high-level contract.
    """

    def __init__(self, backend: KernelBackend | None = None) -> None:
        self._backend = backend

    @property
    def has_backend(self) -> bool:
        return self._backend is not None

    def validate(self, inputs: GRDecodeAttentionInputs) -> None:
        q_shape = inputs.q_shape
        if len(q_shape) != 4:
            raise ValueError(f"Q expects [B, W_t, Hq, D], got {q_shape}")

        batch, width, _num_q_heads, head_dim = q_shape
        if batch != inputs.context_kv.batch_size:
            raise ValueError(
                f"Q batch={batch} differs from ContextKV batch={inputs.context_kv.batch_size}"
            )
        if batch != inputs.beam_kv.batch_size:
            raise ValueError(
                f"Q batch={batch} differs from BeamKV batch={inputs.beam_kv.batch_size}"
            )
        if width != inputs.active_beam_width:
            raise ValueError(
                f"Q beam width={width} differs from active_beam_width={inputs.active_beam_width}"
            )
        if head_dim != inputs.context_kv.head_dim:
            raise ValueError(
                f"Q head_dim={head_dim} differs from ContextKV head_dim={inputs.context_kv.head_dim}"
            )
        if head_dim != inputs.beam_kv.head_dim:
            raise ValueError(
                f"Q head_dim={head_dim} differs from BeamKV head_dim={inputs.beam_kv.head_dim}"
            )
        if inputs.context_kv.num_layers != inputs.beam_kv.num_layers:
            raise ValueError("ContextKV and BeamKV must have the same num_layers")
        if inputs.context_kv.num_kv_heads != inputs.beam_kv.num_kv_heads:
            raise ValueError("ContextKV and BeamKV must have the same num_kv_heads")
        if inputs.layer_idx < 0 or inputs.layer_idx >= inputs.context_kv.num_layers:
            raise ValueError(
                f"layer_idx={inputs.layer_idx} outside [0, {inputs.context_kv.num_layers})"
            )

        inputs.beam_kv.validate_step(inputs.step, inputs.active_beam_width)
        if inputs.beam_path.steps_done > inputs.step + 1:
            raise ValueError(
                "BeamPath contains future steps relative to current attention step"
            )
        decode_nums = inputs.step if inputs.decode_nums is None else inputs.decode_nums
        if decode_nums < 0:
            raise ValueError("decode_nums must be non-negative")
        if decode_nums > inputs.beam_kv.max_decode_steps:
            raise ValueError(
                f"decode_nums={decode_nums} exceeds "
                f"max_decode_steps={inputs.beam_kv.max_decode_steps}"
            )
        if inputs.topk_indices is not None:
            topk_shape = shape_of(inputs.topk_indices)
            if len(topk_shape) != 5:
                raise ValueError(
                    "topk_indices expects [B, Sq, Hq, max_decode_nums, W], "
                    f"got {topk_shape}"
                )
            if topk_shape[0] != batch or topk_shape[2] != _num_q_heads:
                raise ValueError(
                    "topk_indices batch/head dimensions must match Q: "
                    f"topk={topk_shape}, q={q_shape}"
                )
            if topk_shape[3] < decode_nums:
                raise ValueError(
                    f"topk max_decode_nums={topk_shape[3]} is smaller than "
                    f"decode_nums={decode_nums}"
                )
            if topk_shape[4] < inputs.active_beam_width:
                raise ValueError(
                    f"topk beam width={topk_shape[4]} is smaller than "
                    f"active_beam_width={inputs.active_beam_width}"
                )

    def __call__(self, inputs: GRDecodeAttentionInputs) -> Any:
        self.validate(inputs)
        if self._backend is None:
            raise MissingKernelBackend(
                "GR decode attention backend is not installed. Inject the existing "
                "kernel binding into GRDecodeAttention(backend=...)."
            )
        return self._backend(inputs)
