"""Serving request and response contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gr_inference.gr_runtime.logits_processor import validate_logits_processors


@dataclass(frozen=True)
class GRServingRequest:
    """One GR inference request.

    MVP scope is one already-tokenized request. Tokenization and network API
    layers can wrap this contract later.
    """

    request_id: str
    input_ids: Any
    max_decode_steps: int
    beam_width: int
    metadata: dict[str, Any] = field(default_factory=dict)
    item_mask_provider: Any | None = None
    beam_width_policy: Any | None = None
    stop_token_ids: tuple[int, ...] = ()
    logits_processors: tuple[Any, ...] = ()

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.max_decode_steps <= 0:
            raise ValueError("max_decode_steps must be positive")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.input_ids is None:
            raise ValueError("input_ids must be provided")
        if any(token < 0 for token in self.stop_token_ids):
            raise ValueError("stop_token_ids must be non-negative")
        validate_logits_processors(tuple(self.logits_processors))
        timeout_ticks = self.metadata.get("timeout_ticks")
        if timeout_ticks is not None:
            if isinstance(timeout_ticks, bool):
                raise ValueError("timeout_ticks must be an integer")
            if int(timeout_ticks) < 0:
                raise ValueError("timeout_ticks must be non-negative")
        _validate_beam_width_policy(self.beam_width_policy, self.beam_width)


@dataclass(frozen=True)
class GRServingResponse:
    """Serving response for one GR inference request."""

    request_id: str
    token_ids: tuple[int, ...]
    scores: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_beam_width_policy(policy: Any | None, request_beam_width: int) -> None:
    if policy is None:
        return
    width_for_step = getattr(policy, "width_for_step", None)
    if width_for_step is None:
        raise ValueError("beam_width_policy must define width_for_step(step)")
    initial_width = int(width_for_step(0))
    _validate_policy_width(initial_width, request_beam_width)

    max_beam_width = getattr(policy, "max_beam_width", None)
    if max_beam_width is not None and int(max_beam_width) > request_beam_width:
        raise ValueError("beam_width_policy max_beam_width exceeds request beam_width")
    for width in getattr(policy, "schedule", {}).values():
        _validate_policy_width(int(width), request_beam_width)
    for width in getattr(policy, "widths", {}).values():
        _validate_policy_width(int(width), request_beam_width)


def _validate_policy_width(width: int, request_beam_width: int) -> None:
    if width <= 0 or width > request_beam_width:
        raise ValueError(
            "beam_width_policy produced width outside request beam_width: "
            f"width={width}, request_beam_width={request_beam_width}"
        )

