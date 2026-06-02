# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request runtime state."""

from __future__ import annotations

from dataclasses import dataclass

from gr_inference.gr_kv.beam_kv import BeamKV
from gr_inference.gr_kv.beam_path import BeamPath
from gr_inference.gr_kv.context_kv import ContextKV


@dataclass
class GRRequestState:
    """Runtime-owned state for one GR request."""

    request_id: str
    context_kv: ContextKV
    beam_kv: BeamKV
    beam_path: BeamPath

    @property
    def max_decode_steps(self) -> int:
        return self.beam_kv.max_decode_steps

    @property
    def max_beam_width(self) -> int:
        return self.beam_kv.max_beam_width

    def validate(self) -> None:
        if self.context_kv.num_layers != self.beam_kv.num_layers:
            raise ValueError("ContextKV and BeamKV must have matching layer count")
        if self.context_kv.batch_size != self.beam_kv.batch_size:
            raise ValueError("ContextKV and BeamKV must have matching batch size")
        if self.context_kv.num_kv_heads != self.beam_kv.num_kv_heads:
            raise ValueError("ContextKV and BeamKV must have matching KV heads")
        if self.context_kv.head_dim != self.beam_kv.head_dim:
            raise ValueError("ContextKV and BeamKV must have matching head_dim")
        if self.beam_path.max_decode_steps not in {
            self.beam_kv.max_decode_steps,
            self.beam_kv.max_decode_steps + 1,
        }:
            raise ValueError(
                "BeamPath max_decode_steps must match BeamKV capacity or include "
                "one extra initial-beam entry"
            )
        if self.beam_path.max_beam_width != self.beam_kv.max_beam_width:
            raise ValueError("BeamPath and BeamKV must have matching max_beam_width")
