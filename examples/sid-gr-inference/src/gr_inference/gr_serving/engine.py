# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronous serving engine skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from gr_inference.gr_kv import BatchedBeamPath
from gr_inference.gr_runtime import GRDecodeEngine, GRGenerationState
from gr_inference.gr_runtime.batched_beam_search import (
    batched_item_mask_limited_beam_width,
    select_initial_topk_batched,
    select_next_topk_batched,
)
from gr_inference.gr_runtime.batched_decode_inputs import make_batched_beam_token_ids
from gr_inference.gr_runtime.batched_topk_indices import make_batched_topk_indices
from gr_inference.gr_runtime.generation import PrefillResult
from gr_inference.gr_runtime.logits_processor import (
    LogitsProcessorContext,
    apply_logits_processors,
    logits_processors_metadata,
)
from gr_inference.gr_serving.batch import GRRequestBatch
from gr_inference.gr_serving.beam_metadata import (
    attach_item_results as _attach_item_results,
)
from gr_inference.gr_serving.beam_metadata import batched_beam_details as _beam_details
from gr_inference.gr_serving.beam_metadata import (
    batched_selection_stop_reason as _batched_selection_stop_reason,
)
from gr_inference.gr_serving.beam_metadata import beam_score_type as _beam_score_type
from gr_inference.gr_serving.beam_metadata import (
    beam_width_policy_metadata as _beam_width_policy_metadata,
)
from gr_inference.gr_serving.beam_metadata import (
    request_stop_token_ids as _request_stop_token_ids,
)
from gr_inference.gr_serving.beam_metadata import (
    selected_decode_token_logprobs as _selected_decode_token_logprobs,
)
from gr_inference.gr_serving.beam_metadata import (
    selected_initial_token_logprobs_batched as _selected_initial_token_logprobs,
)
from gr_inference.gr_serving.config import GRServingConfig
from gr_inference.gr_serving.decode_batch import GRDecodeBatchPlanner
from gr_inference.gr_serving.metrics import ServingMetrics
from gr_inference.gr_serving.request import GRServingRequest, GRServingResponse


@dataclass(frozen=True)
class _BatchedDecodeAttempt:
    responses: tuple[GRServingResponse, ...] | None = None
    fallback_reason: str | None = None


@dataclass
class GRServingEngine:
    """Minimal single-request serving engine.

    This is intentionally synchronous. It establishes the request lifecycle
    boundary before adding API servers, schedulers, batching, and async workers.
    """

    model: Any
    decode_engine: GRDecodeEngine
    config: GRServingConfig

    def __post_init__(self) -> None:
        self.config.validate()

    def status(self) -> dict[str, Any]:
        return {
            "max_decode_steps": self.config.max_decode_steps,
            "max_beam_width": self.config.max_beam_width,
            "kernel_profile_path": self.config.kernel_profile_path,
            "enable_batched_decode": self.config.enable_batched_decode,
            "return_beam_details": self.config.return_beam_details,
            "beam_score_mode": self.config.beam_score_mode,
            "model": getattr(getattr(self.model, "config", None), "model_name", None),
        }

    def warmup(self, request: GRServingRequest) -> GRServingResponse:
        response = self.generate(request)
        response.metadata["warmup"] = True
        return response

    def generate_batch(self, batch: GRRequestBatch) -> tuple[GRServingResponse, ...]:
        """Generate responses for a scheduler batch.

        MVP implementation runs requests sequentially. The method exists so a
        future implementation can swap in a true tensor-batched path without
        changing scheduler ownership.
        """

        if batch.compatible_for_tensor_batch():
            batched = self._generate_batch_with_batched_prefill(batch)
            if batched is not None:
                return batched
        return tuple(self.generate(request) for request in batch.requests)

    def _generate_batch_with_batched_prefill(
        self,
        batch: GRRequestBatch,
    ) -> tuple[GRServingResponse, ...] | None:
        try:
            import torch
        except ImportError:
            return None
        if not all(hasattr(request.input_ids, "shape") for request in batch.requests):
            return None

        metrics = ServingMetrics()
        with metrics.section("batch_prefill_ms"):
            input_ids = torch.cat(
                [request.input_ids for request in batch.requests], dim=0
            )
            prefill = self.model.forward_prefill(input_ids, return_result=True)

        responses: list[GRServingResponse] = []
        generations: list[GRGenerationState] = []
        request_prefills: list[tuple[GRServingRequest, PrefillResult]] = []
        for index, request in enumerate(batch.requests):
            request.validate()
            per_request_prefill = PrefillResult(
                logits=prefill.logits[index : index + 1],
                context_kv=prefill.context_kv.slice_batch(index),
                hidden_states=(
                    prefill.hidden_states[index : index + 1]
                    if prefill.hidden_states is not None
                    else None
                ),
            )
            request_prefills.append((request, per_request_prefill))
            generations.append(
                GRGenerationState.from_prefill(
                    request_id=request.request_id,
                    prefill=per_request_prefill,
                    max_decode_steps=request.max_decode_steps,
                    max_beam_width=self.config.max_beam_width,
                    fixed_beam_width=request.beam_width,
                    beam_score_mode=self.config.beam_score_mode,
                )
            )
        decode_batch_metadata = _decode_batch_metadata(
            tuple(generations),
            max_decode_steps=batch.max_decode_steps or 0,
        )
        batched_decode_fallback_reason: str | None = None
        if self.config.enable_batched_decode:
            batched_decode = self._try_generate_batched_decode(
                batch,
                prefill,
                decode_batch_metadata=decode_batch_metadata,
                batch_prefill_metadata=metrics.to_metadata(),
            )
            if batched_decode.responses is not None:
                return batched_decode.responses
            batched_decode_fallback_reason = batched_decode.fallback_reason
        for request, per_request_prefill in request_prefills:
            metadata: dict[str, Any] = {
                "batched_prefill": True,
                "batch_size": batch.size,
                "decode_batch_plan": decode_batch_metadata,
                **metrics.to_metadata(),
            }
            if self.config.enable_batched_decode:
                metadata.update(
                    {
                        "batched_decode": False,
                        "batched_decode_fallback_reason": (
                            batched_decode_fallback_reason
                            or "batched_decode_unavailable"
                        ),
                    }
                )
            response = self._generate_from_prefill(
                request,
                per_request_prefill,
                metadata=metadata,
            )
            response.metadata.setdefault(
                "prefill_ms",
                response.metadata.get("batch_prefill_ms", 0.0),
            )
            response.metadata.setdefault(
                "total_ms",
                response.metadata.get("batch_prefill_ms", 0.0)
                + response.metadata.get("decode_ms", 0.0),
            )
            responses.append(response)
        return tuple(responses)

    def _try_generate_batched_decode(
        self,
        batch: GRRequestBatch,
        prefill,
        *,
        decode_batch_metadata: list[dict[str, Any]],
        batch_prefill_metadata: dict[str, float],
    ) -> _BatchedDecodeAttempt:
        try:
            prefill_logits = _batched_apply_logits_processors(
                batch.requests,
                prefill.logits,
                phase="prefill",
                step=0,
                beam_width=batch.beam_width or 0,
            )
            initial_item_mask = _batched_initial_item_mask(
                batch.requests, prefill_logits
            )
            initial_beam_width = batched_item_mask_limited_beam_width(
                batch.beam_width or 0,
                initial_item_mask,
            )
            selection = select_initial_topk_batched(
                prefill_logits,
                beam_width=initial_beam_width,
                item_mask=initial_item_mask,
                score_mode=self.config.beam_score_mode,
            )
            max_decode_steps = batch.max_decode_steps or 0
            batched_beam_path = BatchedBeamPath.create(
                batch_size=batch.size,
                max_decode_steps=max_decode_steps + 1,
                max_beam_width=self.config.max_beam_width,
            )
            batched_beam_path.append(selection)
            stop_reason = _batched_selection_stop_reason(
                batch.requests,
                batched_beam_path,
                selection.token_ids,
            )
            token_logprob_steps = (
                [_selected_initial_token_logprobs(prefill_logits, selection)]
                if self.config.return_beam_details
                else None
            )
            generation = GRGenerationState.from_prefill(
                request_id="batched-decode",
                prefill=prefill,
                max_decode_steps=max_decode_steps,
                max_beam_width=self.config.max_beam_width,
                fixed_beam_width=batch.beam_width,
                beam_score_mode=self.config.beam_score_mode,
            )
            metrics = ServingMetrics()
            step_metadata: list[dict[str, Any]] = []
            with metrics.section("batched_decode_ms"):
                for step in range(max_decode_steps):
                    if stop_reason is not None:
                        break
                    decode_inputs = make_batched_beam_token_ids(
                        selection,
                        device=getattr(prefill.logits, "device", None),
                    )
                    decode_nums = step + 1
                    topk_indices = make_batched_topk_indices(
                        batched_beam_path,
                        num_q_heads=self.model.config.num_attention_heads,
                        decode_nums=decode_nums,
                        beam_width=decode_inputs.beam_width,
                        device=getattr(prefill.logits, "device", None),
                    )
                    logits = self.model.forward_decode_step(
                        decode_inputs.beam_token_ids,
                        generation,
                        self.decode_engine,
                        step=step,
                        active_beam_width=decode_inputs.beam_width,
                        topk_indices=topk_indices,
                        decode_nums=decode_nums,
                    )
                    logits = _batched_apply_logits_processors(
                        batch.requests,
                        logits,
                        phase="decode",
                        step=step,
                        beam_width=decode_inputs.beam_width,
                        beam_paths=batched_beam_path.paths,
                    )
                    item_mask = _batched_step_item_mask(
                        batch.requests,
                        batched_beam_path,
                        logits,
                    )
                    next_beam_width = batched_item_mask_limited_beam_width(
                        decode_inputs.beam_width,
                        item_mask,
                    )
                    selection = select_next_topk_batched(
                        logits,
                        previous_scores=selection.scores,
                        beam_width=next_beam_width,
                        item_mask=item_mask,
                        score_mode=self.config.beam_score_mode,
                    )
                    if token_logprob_steps is not None:
                        token_logprob_steps.append(
                            _selected_decode_token_logprobs(logits, selection)
                        )
                    batched_beam_path.append(selection)
                    step_info = {
                        "step": step,
                        "batch_size": decode_inputs.batch_size,
                        "beam_width": decode_inputs.beam_width,
                        "topk_indices_shape": tuple(topk_indices.shape),
                    }
                    stop_reason = _batched_selection_stop_reason(
                        batch.requests,
                        batched_beam_path,
                        selection.token_ids,
                    )
                    if stop_reason is not None:
                        step_info["early_stop"] = True
                        step_info["stop_reason"] = stop_reason
                        step_metadata.append(step_info)
                        break
                    step_metadata.append(step_info)
        except Exception as exc:
            return _BatchedDecodeAttempt(
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

        responses: list[GRServingResponse] = []
        decode_metadata = metrics.to_metadata()
        for index, request in enumerate(batch.requests):
            metadata = {
                "beam_width": len(selection.token_ids[index]),
                "requested_beam_width": request.beam_width,
                "decode_steps": request.max_decode_steps,
                "batched_prefill": True,
                "batched_decode": True,
                "batched_decode_steps": len(step_metadata),
                "batched_decode_step_plan": step_metadata,
                "batched_beam_path_steps": batched_beam_path.steps_done,
                "stop_reason": stop_reason or "max_decode_steps",
                "batch_size": batch.size,
                "decode_batch_plan": decode_batch_metadata,
                **batch_prefill_metadata,
                **decode_metadata,
                "prefill_ms": batch_prefill_metadata.get("batch_prefill_ms", 0.0),
                "decode_ms": decode_metadata.get("batched_decode_ms", 0.0),
                "total_ms": batch_prefill_metadata.get("batch_prefill_ms", 0.0)
                + decode_metadata.get("batched_decode_ms", 0.0),
            }
            _attach_logits_processors_metadata(metadata, request)
            if self.config.return_beam_details:
                metadata["beam_details"] = _beam_details(
                    batched_beam_path,
                    batch_idx=index,
                    beam_width=len(selection.token_ids[index]),
                    token_logprob_steps=token_logprob_steps,
                    score_type=_beam_score_type(self.config.beam_score_mode),
                )
            _attach_item_results(
                metadata,
                request=request,
                beam_path=batched_beam_path.paths[index],
                beam_width=len(selection.token_ids[index]),
            )
            responses.append(
                GRServingResponse(
                    request_id=request.request_id,
                    token_ids=selection.token_ids[index],
                    scores=selection.scores[index],
                    metadata=metadata,
                )
            )
        return _BatchedDecodeAttempt(responses=tuple(responses))

    def generate(self, request: GRServingRequest) -> GRServingResponse:
        metrics = ServingMetrics()
        request.validate()
        if request.max_decode_steps > self.config.max_decode_steps:
            raise ValueError("request max_decode_steps exceeds serving capacity")
        if request.beam_width > self.config.max_beam_width:
            raise ValueError("request beam_width exceeds serving capacity")

        with metrics.section("total_ms"):
            with metrics.section("prefill_ms"):
                prefill = self.model.forward_prefill(
                    request.input_ids, return_result=True
                )
            response = self._generate_from_prefill(
                request,
                prefill,
                metadata=metrics.to_metadata(),
                outer_metrics=metrics,
            )
        response.metadata.setdefault(
            "total_ms", metrics.to_metadata().get("total_ms", 0.0)
        )
        return response

    def _generate_from_prefill(
        self,
        request: GRServingRequest,
        prefill,
        *,
        metadata: dict[str, Any] | None = None,
        outer_metrics: ServingMetrics | None = None,
    ) -> GRServingResponse:
        metrics = outer_metrics or ServingMetrics()
        if request.max_decode_steps > self.config.max_decode_steps:
            raise ValueError("request max_decode_steps exceeds serving capacity")
        if request.beam_width > self.config.max_beam_width:
            raise ValueError("request beam_width exceeds serving capacity")
        generation = GRGenerationState.from_prefill(
            request_id=request.request_id,
            prefill=prefill,
            max_decode_steps=request.max_decode_steps,
            max_beam_width=self.config.max_beam_width,
            fixed_beam_width=request.beam_width,
            beam_score_mode=self.config.beam_score_mode,
        )
        with metrics.section("decode_ms"):
            result = self.model.generate_fixed_beam(
                generation,
                self.decode_engine,
                max_steps=request.max_decode_steps,
                item_mask_provider=request.item_mask_provider,
                beam_width_policy=request.beam_width_policy,
                stop_token_ids=_request_stop_token_ids(request),
                logits_processors=tuple(request.logits_processors),
            )
        scores = (
            generation.beam_path.entries[-1].scores
            if generation.beam_path.entries
            else ()
        )
        response_metadata = {
            "beam_width": len(result.final_token_ids),
            "requested_beam_width": request.beam_width,
            "decode_steps": len(result.steps),
            "stop_reason": result.stop_reason,
            **metrics.to_metadata(),
            **(metadata or {}),
        }
        _attach_logits_processors_metadata(response_metadata, request)
        if request.beam_width_policy is not None:
            response_metadata["beam_width_policy"] = _beam_width_policy_metadata(
                request.beam_width_policy
            )
        _attach_item_results(
            response_metadata,
            request=request,
            beam_path=generation.beam_path,
            beam_width=len(result.final_token_ids),
        )
        return GRServingResponse(
            request_id=request.request_id,
            token_ids=result.final_token_ids,
            scores=scores,
            metadata=response_metadata,
        )


def _decode_batch_metadata(
    generations: tuple[GRGenerationState, ...],
    *,
    max_decode_steps: int,
) -> list[dict[str, Any]]:
    planner = GRDecodeBatchPlanner()
    return [
        decode_batch.metadata()
        for step in range(max_decode_steps)
        for decode_batch in planner.plan(generations, step=step)
    ]


def _batched_apply_logits_processors(
    requests: tuple[GRServingRequest, ...],
    logits: Any,
    *,
    phase: str,
    step: int,
    beam_width: int,
    beam_paths: tuple[Any, ...] | None = None,
) -> Any:
    if not any(request.logits_processors for request in requests):
        return logits

    import torch

    rows = []
    for batch_idx, request in enumerate(requests):
        row = logits[batch_idx : batch_idx + 1]
        beam_path = None if beam_paths is None else beam_paths[batch_idx]
        processed = apply_logits_processors(
            row,
            tuple(request.logits_processors),
            LogitsProcessorContext(
                request_id=request.request_id,
                phase=phase,  # type: ignore[arg-type]
                step=step,
                beam_width=beam_width,
                beam_path=beam_path,
                metadata=request.metadata,
            ),
        )
        rows.append(processed)
    return torch.cat(rows, dim=0)


def _batched_initial_item_mask(
    requests: tuple[GRServingRequest, ...],
    logits: Any,
):
    if not any(request.item_mask_provider is not None for request in requests):
        return None

    import torch

    scores = logits[:, -1, :] if logits.dim() == 3 else logits
    vocab_size = scores.shape[-1]
    rows = []
    for batch_idx, request in enumerate(requests):
        if request.item_mask_provider is None:
            rows.append(torch.ones(vocab_size, dtype=torch.bool, device=scores.device))
            continue
        mask = request.item_mask_provider.initial_mask(
            scores[batch_idx : batch_idx + 1]
        )
        if mask.dim() == 2 and mask.shape[0] == 1:
            mask = mask[0]
        if tuple(mask.shape) != (vocab_size,):
            raise ValueError(
                f"initial item mask must be shaped {(vocab_size,)}, got {tuple(mask.shape)}"
            )
        rows.append(mask.bool())
    return torch.stack(rows, dim=0)


def _batched_step_item_mask(
    requests: tuple[GRServingRequest, ...],
    batched_beam_path: BatchedBeamPath,
    logits: Any,
):
    if not any(request.item_mask_provider is not None for request in requests):
        return None

    import torch

    batch_size, beam_width, vocab_size = logits.shape
    rows = []
    for batch_idx, request in enumerate(requests):
        if request.item_mask_provider is None:
            rows.append(
                torch.ones(
                    (beam_width, vocab_size),
                    dtype=torch.bool,
                    device=logits.device,
                )
            )
            continue
        generation_like = SimpleNamespace(beam_path=batched_beam_path.paths[batch_idx])
        mask = request.item_mask_provider.step_mask(
            generation_like,
            logits[batch_idx : batch_idx + 1],
        )
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if tuple(mask.shape) != (beam_width, vocab_size):
            raise ValueError(
                f"step item mask must be shaped {(beam_width, vocab_size)}, got {tuple(mask.shape)}"
            )
        rows.append(mask.bool())
    if len(rows) != batch_size:
        raise ValueError("request count must match decode logits batch size")
    return torch.stack(rows, dim=0)


def _attach_logits_processors_metadata(
    metadata: dict[str, Any],
    request: GRServingRequest,
) -> None:
    if request.logits_processors:
        metadata["logits_processors"] = logits_processors_metadata(
            tuple(request.logits_processors)
        )
