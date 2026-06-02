"""Prefill-to-decode generation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gr_inference.gr_kv import BeamKV, BeamPath, ContextKV
from gr_inference.gr_runtime.beam_search import (
    BeamSelection,
    select_initial_topk,
    select_next_topk,
)
from gr_inference.gr_runtime.request import GRRequestState


@dataclass(frozen=True)
class PrefillResult:
    """Structured result produced by model prefill."""

    logits: Any
    context_kv: ContextKV
    hidden_states: Any | None = None

    @property
    def batch_size(self) -> int:
        return self.context_kv.batch_size

    @property
    def context_len(self) -> int:
        return self.context_kv.context_len


@dataclass
class GRGenerationState:
    """Runtime state that bridges prefill output into fixed-beam decode."""

    request_id: str
    prefill: PrefillResult
    beam_kv: BeamKV
    beam_path: BeamPath
    fixed_beam_width: int
    beam_score_mode: str = "raw_logits"

    @classmethod
    def from_prefill(
        cls,
        *,
        request_id: str,
        prefill: PrefillResult,
        max_decode_steps: int,
        max_beam_width: int,
        fixed_beam_width: int | None = None,
        beam_score_mode: str = "raw_logits",
        beam_kv: BeamKV | None = None,
    ) -> "GRGenerationState":
        if fixed_beam_width is None:
            fixed_beam_width = max_beam_width
        if fixed_beam_width <= 0 or fixed_beam_width > max_beam_width:
            raise ValueError("fixed_beam_width must be in (0, max_beam_width]")
        if beam_score_mode not in {"raw_logits", "logprob"}:
            raise ValueError("beam_score_mode must be 'raw_logits' or 'logprob'")
        if beam_kv is None:
            beam_kv = allocate_beam_kv_like_context(
                prefill.context_kv,
                max_decode_steps=max_decode_steps,
                max_beam_width=max_beam_width,
            )
        elif (
            beam_kv.max_decode_steps < max_decode_steps
            or beam_kv.max_beam_width < max_beam_width
        ):
            raise ValueError("provided beam_kv is smaller than requested decode capacity")
        beam_path = BeamPath(
            # BeamPath stores the initial prefill topK plus every decode step.
            max_decode_steps=max_decode_steps + 1,
            max_beam_width=max_beam_width,
        )
        return cls(
            request_id=request_id,
            prefill=prefill,
            beam_kv=beam_kv,
            beam_path=beam_path,
            fixed_beam_width=fixed_beam_width,
            beam_score_mode=beam_score_mode,
        )

    def request_state(self) -> GRRequestState:
        state = GRRequestState(
            request_id=self.request_id,
            context_kv=self.prefill.context_kv,
            beam_kv=self.beam_kv,
            beam_path=self.beam_path,
        )
        state.validate()
        return state

    def initialize_beams(
        self,
        *,
        item_mask: Any | None = None,
        logits: Any | None = None,
    ) -> BeamSelection:
        """Select initial beams from prefill logits and append BeamPath step 0."""

        if self.beam_path.steps_done != 0:
            raise ValueError("initial beams have already been selected")
        return self.initialize_beams_with_width(
            self.fixed_beam_width,
            item_mask=item_mask,
            logits=logits,
        )

    def initialize_beams_with_width(
        self,
        beam_width: int,
        *,
        item_mask: Any | None = None,
        logits: Any | None = None,
    ) -> BeamSelection:
        """Select initial beams with an explicit active width."""

        if self.beam_path.steps_done != 0:
            raise ValueError("initial beams have already been selected")
        if beam_width <= 0 or beam_width > self.beam_kv.max_beam_width:
            raise ValueError("beam_width must be in (0, max_beam_width]")
        selection = select_initial_topk(
            self.prefill.logits if logits is None else logits,
            beam_width=beam_width,
            item_mask=item_mask,
            score_mode=self.beam_score_mode,
        )
        self.beam_path.append(
            parent_beams=selection.parent_beams,
            token_ids=selection.token_ids,
            scores=selection.scores,
        )
        return selection

    def initialize_beams_from_selection(self, selection: BeamSelection) -> BeamSelection:
        """Append a precomputed initial selection to the beam path."""

        if self.beam_path.steps_done != 0:
            raise ValueError("initial beams have already been selected")
        if selection.width <= 0 or selection.width > self.beam_kv.max_beam_width:
            raise ValueError("selection width must be in (0, max_beam_width]")
        self.beam_path.append(
            parent_beams=selection.parent_beams,
            token_ids=selection.token_ids,
            scores=selection.scores,
        )
        return selection

    def update_beams_from_logits(
        self,
        logits: Any,
        *,
        item_mask: Any | None = None,
    ) -> BeamSelection:
        """Select next beams from decode logits and append a BeamPath step."""

        if self.beam_path.steps_done == 0:
            raise ValueError("initialize_beams must be called before update")
        previous_scores = self.beam_path.entries[-1].scores
        return self.update_beams_from_logits_with_width(
            logits,
            self.fixed_beam_width,
            item_mask=item_mask,
        )

    def update_beams_from_logits_with_width(
        self,
        logits: Any,
        beam_width: int,
        *,
        item_mask: Any | None = None,
    ) -> BeamSelection:
        """Select next beams using an explicit active width."""

        if self.beam_path.steps_done == 0:
            raise ValueError("initialize_beams must be called before update")
        if beam_width <= 0 or beam_width > self.beam_kv.max_beam_width:
            raise ValueError("beam_width must be in (0, max_beam_width]")
        previous_scores = self.beam_path.entries[-1].scores
        selection = select_next_topk(
            logits,
            previous_scores=previous_scores,
            beam_width=beam_width,
            item_mask=item_mask,
            score_mode=self.beam_score_mode,
        )
        self.beam_path.append(
            parent_beams=selection.parent_beams,
            token_ids=selection.token_ids,
            scores=selection.scores,
        )
        return selection


def allocate_beam_kv_like_context(
    context_kv: ContextKV,
    *,
    max_decode_steps: int,
    max_beam_width: int,
) -> BeamKV:
    """Allocate BeamKV with the same framework/device properties as ContextKV."""

    if max_decode_steps <= 0:
        raise ValueError("max_decode_steps must be positive")
    if max_beam_width <= 0:
        raise ValueError("max_beam_width must be positive")

    shape = (
        context_kv.num_layers,
        context_kv.batch_size,
        max_decode_steps,
        max_beam_width,
        context_kv.num_kv_heads,
        context_kv.head_dim,
    )
    key = _empty_like_kv(context_kv.key, shape)
    value = _empty_like_kv(context_kv.value, shape)
    return BeamKV(key, value)


def _empty_like_kv(reference: Any, shape: tuple[int, ...]) -> Any:
    if hasattr(reference, "new_empty"):
        return reference.new_empty(shape)
    if hasattr(reference, "with_shape"):
        return reference.with_shape(shape)
    raise TypeError(f"cannot allocate BeamKV from reference type {type(reference)!r}")
