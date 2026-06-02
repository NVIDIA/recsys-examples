# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared long-context KV metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gr_inference.gr_kv.layouts import Shape, shape_of


@dataclass(frozen=True)
class ContextKV:
    """Shared prompt KV for all beams in a request.

    Expected layout:
      key/value: [layers, batch, context_len, kv_heads, head_dim]
    """

    key: Any
    value: Any

    def __post_init__(self) -> None:
        if self.key_shape != self.value_shape:
            raise ValueError(
                f"ContextKV key/value shapes differ: {self.key_shape} vs {self.value_shape}"
            )
        if len(self.key_shape) != 5:
            raise ValueError(
                "ContextKV expects [layers, batch, context_len, kv_heads, head_dim], "
                f"got {self.key_shape}"
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
    def context_len(self) -> int:
        return self.key_shape[2]

    @property
    def num_kv_heads(self) -> int:
        return self.key_shape[3]

    @property
    def head_dim(self) -> int:
        return self.key_shape[4]

    def expected_layer_shape(self) -> Shape:
        return (self.batch_size, self.context_len, self.num_kv_heads, self.head_dim)

    def slice_batch(self, index: int) -> "ContextKV":
        """Return a one-request view for batch index."""

        if index < 0 or index >= self.batch_size:
            raise IndexError("batch index out of range")
        return ContextKV(
            self.key[:, index : index + 1],
            self.value[:, index : index + 1],
        )
