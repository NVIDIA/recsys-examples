# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode batch planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gr_inference.gr_runtime import GRGenerationState


@dataclass(frozen=True)
class GRDecodeBatch:
    """Requests that can share a decode step launch in the future."""

    step: int
    beam_width: int
    generations: tuple[GRGenerationState, ...]

    @property
    def size(self) -> int:
        return len(self.generations)

    def metadata(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "beam_width": self.beam_width,
            "size": self.size,
            "request_ids": [generation.request_id for generation in self.generations],
        }


class GRDecodeBatchPlanner:
    """Plan decode batches by step and active beam width."""

    def plan(
        self,
        generations: tuple[GRGenerationState, ...],
        *,
        step: int,
    ) -> tuple[GRDecodeBatch, ...]:
        groups: dict[int, list[GRGenerationState]] = {}
        for generation in generations:
            width = _active_width(generation)
            groups.setdefault(width, []).append(generation)
        return tuple(
            GRDecodeBatch(
                step=step,
                beam_width=width,
                generations=tuple(group),
            )
            for width, group in sorted(groups.items())
        )


def _active_width(generation: GRGenerationState) -> int:
    if generation.beam_path.entries:
        return generation.beam_path.active_beam_width
    return generation.fixed_beam_width
