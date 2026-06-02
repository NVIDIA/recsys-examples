"""Fixed-beam decode loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gr_inference.gr_runtime.beam_search import item_mask_limited_beam_width
from gr_inference.gr_runtime.engine import GRDecodeEngine
from gr_inference.gr_runtime.generation import GRGenerationState
from gr_inference.gr_runtime.logits_processor import (
    LogitsProcessorContext,
    apply_logits_processors,
)
from gr_inference.gr_scheduler import BeamWidthPolicy, FixedBeamPolicy


@dataclass(frozen=True)
class DecodeLoopStep:
    step: int
    logits: Any
    token_ids: tuple[int, ...]
    parent_beams: tuple[int, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True)
class DecodeLoopResult:
    generation: GRGenerationState
    steps: tuple[DecodeLoopStep, ...]
    stop_reason: str

    @property
    def final_token_ids(self) -> tuple[int, ...]:
        if not self.steps:
            if not self.generation.beam_path.entries:
                return ()
            return self.generation.beam_path.entries[-1].token_ids
        return self.steps[-1].token_ids


@dataclass
class FixedBeamDecodeLoop:
    """Run a fixed-width decode loop for a Qwen-like model.

    The model must implement ``forward_decode_step(beam_token_ids, generation,
    decode_engine, step=...)``.
    """

    model: Any
    decode_engine: GRDecodeEngine
    item_masks: dict[int, Any] = field(default_factory=dict)
    item_mask_provider: Any | None = None
    beam_width_policy: BeamWidthPolicy | None = None
    stop_token_ids: tuple[int, ...] = ()
    logits_processors: tuple[Any, ...] = ()

    def run(
        self,
        generation: GRGenerationState,
        *,
        max_steps: int,
        initial_item_mask: Any | None = None,
    ) -> DecodeLoopResult:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_steps > generation.beam_kv.max_decode_steps:
            raise ValueError("max_steps exceeds BeamKV capacity")

        import torch

        if initial_item_mask is None and self.item_mask_provider is not None:
            initial_item_mask = self.item_mask_provider.initial_mask(
                generation.prefill.logits
            )
        policy = self.beam_width_policy or FixedBeamPolicy(generation.fixed_beam_width)
        initial_width = policy.width_for_step(0)
        initial_logits = apply_logits_processors(
            generation.prefill.logits,
            tuple(self.logits_processors),
            LogitsProcessorContext(
                request_id=generation.request_id,
                phase="prefill",
                step=0,
                beam_width=initial_width,
                beam_path=generation.beam_path,
            ),
        )

        if generation.beam_path.steps_done == 0:
            initial_width = item_mask_limited_beam_width(
                initial_width,
                initial_item_mask,
            )
            selection = generation.initialize_beams_with_width(
                initial_width,
                item_mask=initial_item_mask,
                logits=initial_logits,
            )
        else:
            selection = generation.beam_path.entries[-1]
        _observe_policy_scores(policy, step=0, scores=selection.scores)

        device = _infer_logits_device(generation.prefill.logits)
        beam_token_ids = torch.tensor(
            [selection.token_ids],
            dtype=torch.long,
            device=device,
        )
        steps: list[DecodeLoopStep] = []
        stop_reason = _selection_stop_reason(
            generation,
            token_ids=selection.token_ids,
            stop_token_ids=self.stop_token_ids,
            item_mask_provider=self.item_mask_provider,
        )

        for step in range(max_steps):
            if stop_reason is not None:
                break
            active_width = len(selection.token_ids)
            logits = self.model.forward_decode_step(
                beam_token_ids,
                generation,
                self.decode_engine,
                step=step,
                active_beam_width=active_width,
            )
            logits = apply_logits_processors(
                logits,
                tuple(self.logits_processors),
                LogitsProcessorContext(
                    request_id=generation.request_id,
                    phase="decode",
                    step=step,
                    beam_width=active_width,
                    beam_path=generation.beam_path,
                ),
            )
            item_mask = self.item_masks.get(step)
            if item_mask is None and self.item_mask_provider is not None:
                item_mask = self.item_mask_provider.step_mask(generation, logits)
            next_width = policy.width_for_step(step + 1)
            next_width = item_mask_limited_beam_width(next_width, item_mask)
            selection = generation.update_beams_from_logits_with_width(
                logits,
                next_width,
                item_mask=item_mask,
            )
            steps.append(
                DecodeLoopStep(
                    step=step,
                    logits=logits,
                    token_ids=selection.token_ids,
                    parent_beams=selection.parent_beams,
                    scores=selection.scores,
                )
            )
            _observe_policy_scores(policy, step=step + 1, scores=selection.scores)
            stop_reason = _selection_stop_reason(
                generation,
                token_ids=selection.token_ids,
                stop_token_ids=self.stop_token_ids,
                item_mask_provider=self.item_mask_provider,
            )
            if stop_reason is not None:
                break
            beam_token_ids = torch.tensor(
                [selection.token_ids],
                dtype=torch.long,
                device=device,
            )

        return DecodeLoopResult(
            generation=generation,
            steps=tuple(steps),
            stop_reason=stop_reason or "max_decode_steps",
        )


def _infer_logits_device(logits: Any) -> Any:
    return getattr(logits, "device", None)


def _observe_policy_scores(
    policy: BeamWidthPolicy, *, step: int, scores: tuple[float, ...]
) -> None:
    observer = getattr(policy, "observe_scores", None)
    if observer is not None:
        observer(step, scores)


def _selection_all_stop(
    token_ids: tuple[int, ...], stop_token_ids: tuple[int, ...]
) -> bool:
    if not stop_token_ids:
        return False
    stop_tokens = set(stop_token_ids)
    return bool(token_ids) and all(token in stop_tokens for token in token_ids)


def _selection_stop_reason(
    generation: GRGenerationState,
    *,
    token_ids: tuple[int, ...],
    stop_token_ids: tuple[int, ...],
    item_mask_provider: Any | None,
) -> str | None:
    if _selection_all_stop(token_ids, stop_token_ids):
        return "stop_token"
    if _selection_all_item_complete(generation, item_mask_provider=item_mask_provider):
        return "item_complete"
    return None


def _selection_all_item_complete(
    generation: GRGenerationState,
    *,
    item_mask_provider: Any | None,
) -> bool:
    is_complete = getattr(item_mask_provider, "is_complete", None)
    if is_complete is None or generation.beam_path.steps_done == 0:
        return False
    width = len(generation.beam_path.entries[-1].token_ids)
    return width > 0 and all(
        bool(is_complete(generation.beam_path.token_trace(beam)))
        for beam in range(width)
    )
