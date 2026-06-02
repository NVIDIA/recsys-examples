"""Short decode BeamKV metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gr_inference.gr_kv.layouts import Shape, shape_of


@dataclass(frozen=True)
class BeamStepWrite:
    """Metadata returned when a decode step writes into BeamKV."""

    step: int
    active_beam_width: int
    expected_step_shape: Shape
    flat_offset: int


@dataclass(frozen=True)
class BeamKV:
    """Short-lived decode KV with step-major beam layout.

    Expected layout:
      key/value: [layers, batch, max_decode_steps, max_beam_width, kv_heads, head_dim]
    """

    key: Any
    value: Any

    def __post_init__(self) -> None:
        if self.key_shape != self.value_shape:
            raise ValueError(
                f"BeamKV key/value shapes differ: {self.key_shape} vs {self.value_shape}"
            )
        if len(self.key_shape) != 6:
            raise ValueError(
                "BeamKV expects [layers, batch, max_decode_steps, max_beam_width, "
                f"kv_heads, head_dim], got {self.key_shape}"
            )

    @property
    def key_shape(self) -> Shape:
        return shape_of(self.key)

    @property
    def value_shape(self) -> Shape:
        return shape_of(self.value)

    @property
    def num_layers(self) -> int:
        return self.key_shape[0]

    @property
    def batch_size(self) -> int:
        return self.key_shape[1]

    @property
    def max_decode_steps(self) -> int:
        return self.key_shape[2]

    @property
    def max_beam_width(self) -> int:
        return self.key_shape[3]

    @property
    def num_kv_heads(self) -> int:
        return self.key_shape[4]

    @property
    def head_dim(self) -> int:
        return self.key_shape[5]

    def flattened_beam_shape(self) -> Shape:
        """Kernel-friendly [B, S_dec_max * W_max, Hkv, D] view shape."""

        return (
            self.batch_size,
            self.max_decode_steps * self.max_beam_width,
            self.num_kv_heads,
            self.head_dim,
        )

    def validate_step(self, step: int, active_beam_width: int) -> BeamStepWrite:
        if step < 0 or step >= self.max_decode_steps:
            raise ValueError(
                f"step={step} outside [0, {self.max_decode_steps})"
            )
        if active_beam_width <= 0:
            raise ValueError("active_beam_width must be positive")
        if active_beam_width > self.max_beam_width:
            raise ValueError(
                f"active_beam_width={active_beam_width} exceeds "
                f"max_beam_width={self.max_beam_width}"
            )
        return BeamStepWrite(
            step=step,
            active_beam_width=active_beam_width,
            expected_step_shape=(
                self.num_layers,
                self.batch_size,
                active_beam_width,
                self.num_kv_heads,
                self.head_dim,
            ),
            flat_offset=step * self.max_beam_width,
        )
