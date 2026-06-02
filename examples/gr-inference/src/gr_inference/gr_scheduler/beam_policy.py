"""Beam-width policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence


class BeamWidthPolicy(Protocol):
    """Return active beam width for a beam state depth."""

    def width_for_step(self, step: int) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class FixedBeamPolicy:
    """MVP policy where W_t is always W_max."""

    beam_width: int

    def __post_init__(self) -> None:
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")

    def width_for_step(self, step: int) -> int:
        if step < 0:
            raise ValueError("step must be non-negative")
        return self.beam_width


@dataclass(frozen=True)
class ScheduledBeamPolicy:
    """Step-indexed dynamic beam width policy.

    ``schedule`` maps beam depth to width. Depth 0 is initial beams from prefill
    logits; depth N is the beam width after N decode steps. Missing depths reuse
    the latest previous scheduled width.
    """

    schedule: dict[int, int]

    def __post_init__(self) -> None:
        if not self.schedule:
            raise ValueError("schedule must not be empty")
        if 0 not in self.schedule:
            raise ValueError("schedule must include step 0")
        for step, width in self.schedule.items():
            if step < 0:
                raise ValueError("schedule steps must be non-negative")
            if width <= 0:
                raise ValueError("beam widths must be positive")

    def width_for_step(self, step: int) -> int:
        if step < 0:
            raise ValueError("step must be non-negative")
        candidates = [
            scheduled_step for scheduled_step in self.schedule if scheduled_step <= step
        ]
        return self.schedule[max(candidates)]


@dataclass
class ScoreMarginBeamPolicy:
    """Shrink active beam width when scores are concentrated near the top beam.

    The policy starts at ``max_beam_width``. After each selection, the decode loop
    may call ``observe_scores(step, scores)``. The width for the next beam depth is
    the number of beams whose score is within ``score_margin`` of the best score,
    clamped to ``[min_beam_width, max_beam_width]``. By default width only shrinks.
    """

    max_beam_width: int
    score_margin: float
    min_beam_width: int = 1
    monotonic_shrink: bool = True
    widths: dict[int, int] | None = None

    def __post_init__(self) -> None:
        if self.max_beam_width <= 0:
            raise ValueError("max_beam_width must be positive")
        if self.min_beam_width <= 0:
            raise ValueError("min_beam_width must be positive")
        if self.min_beam_width > self.max_beam_width:
            raise ValueError("min_beam_width must not exceed max_beam_width")
        if self.score_margin < 0:
            raise ValueError("score_margin must be non-negative")
        if self.widths is None:
            self.widths = {0: self.max_beam_width}
        elif 0 not in self.widths:
            self.widths[0] = self.max_beam_width

    def width_for_step(self, step: int) -> int:
        if step < 0:
            raise ValueError("step must be non-negative")
        assert self.widths is not None
        candidates = [known_step for known_step in self.widths if known_step <= step]
        if not candidates:
            return self.max_beam_width
        return self.widths[max(candidates)]

    def observe_scores(self, step: int, scores: Sequence[float]) -> int:
        if step < 0:
            raise ValueError("step must be non-negative")
        if not scores:
            raise ValueError("scores must not be empty")
        finite_scores = [
            float(score) for score in scores if math.isfinite(float(score))
        ]
        if not finite_scores:
            next_width = self.min_beam_width
        else:
            best = max(finite_scores)
            next_width = sum(
                1 for score in finite_scores if best - score <= self.score_margin
            )
        next_width = max(self.min_beam_width, min(self.max_beam_width, next_width))
        if self.monotonic_shrink:
            next_width = min(self.width_for_step(step), next_width)
        assert self.widths is not None
        self.widths[step + 1] = next_width
        return next_width
