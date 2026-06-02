"""Serving engine configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BeamScoreMode = Literal["raw_logits", "logprob"]


@dataclass(frozen=True)
class GRServingConfig:
    """Configuration for the synchronous serving engine MVP."""

    max_decode_steps: int
    max_beam_width: int
    kernel_profile_path: str | None = None
    enable_batched_decode: bool = False
    return_beam_details: bool = False
    beam_score_mode: BeamScoreMode = "logprob"

    def validate(self) -> None:
        if self.max_decode_steps <= 0:
            raise ValueError("max_decode_steps must be positive")
        if self.max_beam_width <= 0:
            raise ValueError("max_beam_width must be positive")
        if self.beam_score_mode not in {"raw_logits", "logprob"}:
            raise ValueError("beam_score_mode must be 'raw_logits' or 'logprob'")
