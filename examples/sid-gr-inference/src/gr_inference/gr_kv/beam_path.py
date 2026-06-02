# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Beam parent path tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class BeamPathEntry:
    """One decode step of beam ancestry."""

    parent_beams: tuple[int, ...]
    token_ids: tuple[int, ...]
    scores: tuple[float, ...]

    @property
    def width(self) -> int:
        return len(self.parent_beams)


@dataclass
class BeamPath:
    """Compact logical history for beams stored in step-major BeamKV."""

    max_decode_steps: int
    max_beam_width: int
    entries: list[BeamPathEntry] = field(default_factory=list)

    @property
    def steps_done(self) -> int:
        return len(self.entries)

    @property
    def active_beam_width(self) -> int:
        if not self.entries:
            return 1
        return self.entries[-1].width

    def append(
        self,
        parent_beams: Sequence[int],
        token_ids: Sequence[int],
        scores: Sequence[float],
    ) -> BeamPathEntry:
        if self.steps_done >= self.max_decode_steps:
            raise ValueError("BeamPath already reached max_decode_steps")

        parents = tuple(int(beam) for beam in parent_beams)
        tokens = tuple(int(token) for token in token_ids)
        score_values = tuple(float(score) for score in scores)

        if not parents:
            raise ValueError("BeamPath step must contain at least one beam")
        if len(parents) > self.max_beam_width:
            raise ValueError(
                f"beam width {len(parents)} exceeds max_beam_width={self.max_beam_width}"
            )
        if len(tokens) != len(parents) or len(score_values) != len(parents):
            raise ValueError(
                "parent_beams, token_ids, and scores must have equal length"
            )

        previous_width = self.active_beam_width
        for parent in parents:
            if parent < 0 or parent >= previous_width:
                raise ValueError(
                    f"parent beam {parent} outside previous width {previous_width}"
                )

        entry = BeamPathEntry(parents, tokens, score_values)
        self.entries.append(entry)
        return entry

    def token_trace(self, beam: int, step: int | None = None) -> tuple[int, ...]:
        """Return token ids from root to the selected beam at a decode step."""

        if step is None:
            step = self.steps_done - 1
        if step < 0 or step >= self.steps_done:
            raise ValueError(f"step={step} outside completed steps={self.steps_done}")
        if beam < 0 or beam >= self.entries[step].width:
            raise ValueError(f"beam={beam} outside width={self.entries[step].width}")

        tokens: list[int] = []
        current_beam = beam
        for current_step in range(step, -1, -1):
            entry = self.entries[current_step]
            tokens.append(entry.token_ids[current_beam])
            current_beam = entry.parent_beams[current_beam]
        tokens.reverse()
        return tuple(tokens)

    def score(self, beam: int, step: int | None = None) -> float:
        if step is None:
            step = self.steps_done - 1
        if step < 0 or step >= self.steps_done:
            raise ValueError(f"step={step} outside completed steps={self.steps_done}")
        return self.entries[step].scores[beam]
