"""Continuous batching scheduler core.

This module models the serving state machine separately from the synchronous
``SyncGRScheduler``. It is intentionally lightweight: the first implementation
plans prefill admission and decode-step batches, then advances request state one
decode step per tick. The model execution path can later replace the logical
step advancement without changing scheduler ownership.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from gr_inference.gr_kv import BatchedBeamPath, BeamKV, ContextKV
from gr_inference.gr_runtime import BeamSelection, GRGenerationState
from gr_inference.gr_runtime.batched_beam_search import (
    BatchedBeamSelection,
    batched_item_mask_limited_beam_width,
    select_initial_topk_batched,
    select_next_topk_batched,
)
from gr_inference.gr_runtime.batched_decode_inputs import make_batched_beam_token_ids
from gr_inference.gr_runtime.batched_topk_indices import (
    make_batched_topk_indices,
    make_compacted_batched_topk_indices,
)
from gr_inference.gr_runtime.beam_kv_compaction import (
    compact_batched_beam_kv_history,
    needs_batched_beam_kv_history_compaction,
)
from gr_inference.gr_runtime.generation import (
    PrefillResult,
    allocate_beam_kv_like_context,
)
from gr_inference.gr_runtime.logits_processor import (
    LogitsProcessorContext,
    apply_logits_processors,
    logits_processors_metadata,
)
from gr_inference.gr_serving.beam_metadata import (
    attach_item_results as _attach_item_results,
)
from gr_inference.gr_serving.beam_metadata import (
    beam_details as _continuous_beam_details,
)
from gr_inference.gr_serving.beam_metadata import (
    beam_results as _continuous_beam_results,
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
    selected_initial_token_logprobs as _selected_initial_token_logprobs,
)
from gr_inference.gr_serving.cuda_graph_utils import is_cuda_tensor as _is_cuda_tensor
from gr_inference.gr_serving.decode_cuda_graph import GRDecodeCudaGraphRunner
from gr_inference.gr_serving.memory import (
    GRBeamKVPoolLease,
    GRContextKVPoolLease,
    GRDenseBeamKVPool,
    GRDenseContextKVPool,
    GRKVLease,
    GRKVLeaseAllocator,
)
from gr_inference.gr_serving.prefill_cuda_graph import GRPrefillCudaGraphRunner
from gr_inference.gr_serving.prefix_cache import GRPrefixCacheMatch, GRPromptPrefixCache
from gr_inference.gr_serving.queue import GRRequestQueue
from gr_inference.gr_serving.request import GRServingRequest, GRServingResponse

ContinuousRequestStage = Literal["waiting_prefill", "decoding", "finished"]


@dataclass(frozen=True)
class GRMemoryBudget:
    """Simple request admission budget for continuous batching."""

    max_running_requests: int | None = None
    max_context_tokens: int | None = None
    max_beam_slots: int | None = None

    def validate(self) -> None:
        self.allocator()

    def usage(self, states: tuple["GRContinuousRequestState", ...]) -> dict[str, int]:
        allocator = self.allocator()
        for state in states:
            if state.stage == "decoding":
                allocator.allocate(
                    request_id=state.request_id,
                    context_tokens=_request_context_tokens(state.request),
                    beam_slots=_request_beam_slots(state.request),
                )
        return allocator.usage()

    def can_admit(
        self,
        active_states: tuple["GRContinuousRequestState", ...],
        request: GRServingRequest,
    ) -> bool:
        self.validate()
        allocator = self.allocator()
        for state in active_states:
            if state.stage == "decoding":
                allocator.allocate(
                    request_id=state.request_id,
                    context_tokens=_request_context_tokens(state.request),
                    beam_slots=_request_beam_slots(state.request),
                )
        return allocator.can_allocate(
            request_id=request.request_id,
            context_tokens=_request_context_tokens(request),
            beam_slots=_request_beam_slots(request),
        )

    def allocator(self) -> GRKVLeaseAllocator:
        return GRKVLeaseAllocator(
            max_running_requests=self.max_running_requests,
            max_context_tokens=self.max_context_tokens,
            max_beam_slots=self.max_beam_slots,
        )


@dataclass(frozen=True)
class GRContinuousBatchingPolicy:
    """Policy knobs for the continuous batching core."""

    max_prefill_batch_size: int = 1
    max_decode_batch_size: int = 1
    max_running_requests: int | None = None
    max_context_tokens: int | None = None
    max_beam_slots: int | None = None
    max_finished_requests: int | None = None

    def __post_init__(self) -> None:
        if self.max_prefill_batch_size <= 0:
            raise ValueError("max_prefill_batch_size must be positive")
        if self.max_decode_batch_size <= 0:
            raise ValueError("max_decode_batch_size must be positive")
        if self.max_finished_requests is not None and self.max_finished_requests <= 0:
            raise ValueError("max_finished_requests must be positive when set")
        self.memory_budget.validate()

    @property
    def memory_budget(self) -> GRMemoryBudget:
        return GRMemoryBudget(
            max_running_requests=self.max_running_requests,
            max_context_tokens=self.max_context_tokens,
            max_beam_slots=self.max_beam_slots,
        )


@dataclass
class GRContinuousRequestState:
    """Scheduler-owned state for one request."""

    request: GRServingRequest
    stage: ContinuousRequestStage = "waiting_prefill"
    current_decode_step: int = 0
    submitted_tick: int = 0
    admitted_tick: int | None = None
    finished_tick: int | None = None
    stop_reason: str | None = None
    generation: GRGenerationState | None = None
    token_logprob_steps: list[tuple[float, ...]] | None = None
    kv_lease: GRKVLease | None = None
    beam_kv_pool_lease: GRBeamKVPoolLease | None = None
    context_kv_pool_lease: GRContextKVPoolLease | None = None
    active_beam_width: int | None = None
    decode_selection_token_ids: Any | None = None
    decode_selection_scores: Any | None = None
    decode_parent_history: list[Any] = field(default_factory=list)
    pending_decode_token_ids: list[Any] = field(default_factory=list)
    pending_decode_scores: list[Any] = field(default_factory=list)
    pending_decode_parent_beams: list[Any] = field(default_factory=list)

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def beam_width(self) -> int:
        return self.active_beam_width or self.request.beam_width

    @property
    def remaining_decode_steps(self) -> int:
        return max(0, self.request.max_decode_steps - self.current_decode_step)


@dataclass(frozen=True)
class GRContinuousDecodeBatch:
    """Decode-step group planned by the continuous scheduler."""

    step: int
    beam_width: int
    next_beam_width: int
    context_len: int
    request_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.request_ids)

    def metadata(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "beam_width": self.beam_width,
            "active_beam_width": self.beam_width,
            "next_beam_width": self.next_beam_width,
            "context_len": self.context_len,
            "size": self.size,
            "request_ids": list(self.request_ids),
            "group_key": {
                "step": self.step,
                "active_beam_width": self.beam_width,
                "next_beam_width": self.next_beam_width,
                "context_len": self.context_len,
            },
        }


@dataclass(frozen=True)
class GRContinuousTickResult:
    """A single scheduler tick summary."""

    tick: int
    prefill_request_ids: tuple[str, ...]
    decode_batches: tuple[GRContinuousDecodeBatch, ...]
    finished_request_ids: tuple[str, ...]
    memory_usage: dict[str, int]

    def metadata(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "prefill_request_ids": list(self.prefill_request_ids),
            "decode_batches": [batch.metadata() for batch in self.decode_batches],
            "finished_request_ids": list(self.finished_request_ids),
            "memory_usage": self.memory_usage,
        }


@dataclass(frozen=True)
class _DecodeCudaGraphInputs:
    beam_token_ids: Any
    generation: GRGenerationState
    topk_indices: Any | None
    actual_batch_size: int
    graph_batch_size: int
    use_cuda_graph: bool = True


@dataclass
class GRContinuousScheduler:
    """Synchronous continuous batching state machine.

    Each tick admits a prefill microbatch under the memory budget, plans decode
    microbatches by ``(decode_step, beam_width)``, and advances those requests by
    one logical decode step.
    """

    policy: GRContinuousBatchingPolicy = field(
        default_factory=GRContinuousBatchingPolicy
    )
    waiting_prefill: GRRequestQueue = field(default_factory=GRRequestQueue)
    decoding: dict[str, GRContinuousRequestState] = field(default_factory=dict)
    finished: dict[str, GRServingResponse] = field(default_factory=dict)
    states: dict[str, GRContinuousRequestState] = field(default_factory=dict)
    kv_allocator: GRKVLeaseAllocator | None = None
    tick_count: int = 0
    submitted_requests: int = 0
    succeeded_requests: int = 0
    cancelled_requests: int = 0
    failed_requests: int = 0
    timed_out_requests: int = 0
    evicted_finished_requests: int = 0
    admitted_prefill_batches: int = 0
    planned_decode_batches: int = 0
    tick_history: list[dict[str, Any]] = field(default_factory=list)
    kv_events: list[dict[str, Any]] = field(default_factory=list)
    max_kv_events: int = 128

    def __post_init__(self) -> None:
        if self.kv_allocator is None:
            self.kv_allocator = self.policy.memory_budget.allocator()

    def submit(self, request: GRServingRequest) -> None:
        request.validate()
        if request.request_id in self.states:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        state = GRContinuousRequestState(
            request=request, submitted_tick=self.tick_count
        )
        self.states[request.request_id] = state
        self.waiting_prefill.submit(request)
        self.submitted_requests += 1

    def cancel(
        self, request_id: str, *, reason: str = "cancelled"
    ) -> GRServingResponse:
        if not reason:
            raise ValueError("cancel reason must be non-empty")
        if request_id in self.finished:
            return self.finished[request_id]
        state = self.states.get(request_id)
        if state is None:
            raise KeyError(f"unknown request_id: {request_id}")

        if state.stage == "waiting_prefill":
            self.waiting_prefill.remove(request_id)
        elif state.stage == "decoding":
            self.decoding.pop(request_id, None)
            self._release_kv_lease(state)

        state.stage = "finished"
        state.finished_tick = self.tick_count
        state.stop_reason = reason
        response = GRServingResponse(
            request_id=request_id,
            token_ids=(),
            scores=(),
            metadata={
                "continuous_batching": True,
                "cancelled": True,
                "decode_steps": state.current_decode_step,
                "stop_reason": reason,
                "admitted_tick": state.admitted_tick,
                "finished_tick": state.finished_tick,
            },
        )
        self._store_finished_response(response)
        self.cancelled_requests += 1
        return response

    def fail(
        self,
        request_id: str,
        *,
        reason: str = "failed",
        error: BaseException | str | None = None,
    ) -> GRServingResponse:
        if not reason:
            raise ValueError("failure reason must be non-empty")
        if request_id in self.finished:
            return self.finished[request_id]
        state = self.states.get(request_id)
        if state is None:
            raise KeyError(f"unknown request_id: {request_id}")
        return self._fail_state(state, reason=reason, error=error)

    def fail_unfinished(
        self,
        *,
        reason: str = "timeout",
        error: BaseException | str | None = None,
    ) -> tuple[str, ...]:
        failed: list[str] = []
        for request_id in self.waiting_prefill.request_ids():
            response = self.fail(request_id, reason=reason, error=error)
            failed.append(response.request_id)
        for request_id in tuple(self.decoding):
            response = self.fail(request_id, reason=reason, error=error)
            failed.append(response.request_id)
        return tuple(failed)

    def tick(
        self,
        *,
        prefill_executor: Any | None = None,
        decode_executor: Any | None = None,
    ) -> GRContinuousTickResult:
        self.tick_count += 1
        timed_out_ids = self._fail_timed_out_requests()
        prefill_ids = self._admit_prefill_batch()
        if prefill_executor is not None:
            try:
                prefill_executor(prefill_ids)
            except Exception as exc:
                failed_ids = self._fail_request_ids(
                    prefill_ids,
                    reason="prefill_failed",
                    error=exc,
                )
                result = GRContinuousTickResult(
                    tick=self.tick_count,
                    prefill_request_ids=prefill_ids,
                    decode_batches=(),
                    finished_request_ids=(*timed_out_ids, *failed_ids),
                    memory_usage=self._memory_usage(),
                )
                self.tick_history.append(result.metadata())
                return result
        decode_batches = self._plan_decode_batches()
        try:
            finished_ids = (
                decode_executor(decode_batches)
                if decode_executor is not None
                else self._advance_decode_batches(decode_batches)
            )
        except Exception as exc:
            finished_ids = self._fail_request_ids(
                tuple(
                    request_id
                    for decode_batch in decode_batches
                    for request_id in decode_batch.request_ids
                ),
                reason="decode_failed",
                error=exc,
            )
        result = GRContinuousTickResult(
            tick=self.tick_count,
            prefill_request_ids=prefill_ids,
            decode_batches=decode_batches,
            finished_request_ids=(*timed_out_ids, *finished_ids),
            memory_usage=self._memory_usage(),
        )
        self.tick_history.append(result.metadata())
        return result

    def run_until_empty(
        self,
        *,
        max_ticks: int | None = None,
        timeout_unfinished: bool = False,
    ) -> tuple[GRServingResponse, ...]:
        ticks = 0
        while len(self.waiting_prefill) or self.decoding:
            if max_ticks is not None and ticks >= max_ticks:
                if timeout_unfinished:
                    self.fail_unfinished(reason="timeout")
                break
            self.tick()
            ticks += 1
        return tuple(self.finished.values())

    def status(self) -> dict[str, Any]:
        kv_allocator_status = _require_allocator(self.kv_allocator).status()
        return {
            "waiting_prefill": len(self.waiting_prefill),
            "decoding": len(self.decoding),
            "finished": len(self.finished),
            "retained_finished_requests": len(self.finished),
            "submitted_requests": self.submitted_requests,
            "succeeded_requests": self.succeeded_requests,
            "cancelled_requests": self.cancelled_requests,
            "failed_requests": self.failed_requests,
            "timed_out_requests": self.timed_out_requests,
            "evicted_finished_requests": self.evicted_finished_requests,
            "admitted_prefill_batches": self.admitted_prefill_batches,
            "planned_decode_batches": self.planned_decode_batches,
            "ticks": self.tick_count,
            "policy": {
                "max_prefill_batch_size": self.policy.max_prefill_batch_size,
                "max_decode_batch_size": self.policy.max_decode_batch_size,
                "max_running_requests": self.policy.max_running_requests,
                "max_context_tokens": self.policy.max_context_tokens,
                "max_beam_slots": self.policy.max_beam_slots,
                "max_finished_requests": self.policy.max_finished_requests,
                "decode_batch_grouping": (
                    "step,active_beam_width,next_beam_width,context_len"
                ),
            },
            "memory_usage": self._memory_usage(),
            "kv_allocator": kv_allocator_status,
            "kv_health": self._kv_health_status(kv_allocator_status),
            "kv_events": tuple(self.kv_events),
        }

    def metrics(self) -> dict[str, float | int]:
        metrics = {
            "waiting_prefill_requests": len(self.waiting_prefill),
            "decoding_requests": len(self.decoding),
            "active_requests": len(self.waiting_prefill) + len(self.decoding),
            "finished_requests": len(self.finished),
            "retained_finished_requests": len(self.finished),
            "submitted_requests": self.submitted_requests,
            "succeeded_requests": self.succeeded_requests,
            "cancelled_requests": self.cancelled_requests,
            "failed_requests": self.failed_requests,
            "timed_out_requests": self.timed_out_requests,
            "evicted_finished_requests": self.evicted_finished_requests,
            "admitted_prefill_batches": self.admitted_prefill_batches,
            "planned_decode_batches": self.planned_decode_batches,
            "ticks": self.tick_count,
            "kv_events_recorded": len(self.kv_events),
            "avg_prefill_batch_size": self._avg_prefill_batch_size(),
            "avg_decode_batch_size": self._avg_decode_batch_size(),
            **{f"kv_{name}": value for name, value in self._memory_usage().items()},
        }
        metrics.update(
            _numeric_status_metrics(
                "kv_allocator",
                _require_allocator(self.kv_allocator).status(),
            )
        )
        metrics.update(_numeric_status_metrics("kv_health", self._kv_health_status()))
        return metrics

    def _admit_prefill_batch(self) -> tuple[str, ...]:
        admitted: list[str] = []
        deferred: list[GRServingRequest] = []
        while len(admitted) < self.policy.max_prefill_batch_size:
            request = self.waiting_prefill.pop()
            if request is None:
                break
            if not self._can_allocate_kv(request):
                if self._memory_usage()["running_requests"] == 0 and not admitted:
                    raise ValueError(
                        f"request {request.request_id} exceeds continuous batching memory budget"
                    )
                deferred.append(request)
                break
            state = self.states[request.request_id]
            state.kv_lease = self._allocate_kv_lease(request)
            state.active_beam_width = _policy_width_for_step(
                request,
                step=0,
                fallback=request.beam_width,
            )
            state.stage = "decoding"
            state.admitted_tick = self.tick_count
            self.decoding[request.request_id] = state
            admitted.append(request.request_id)

        for request in reversed(deferred):
            self.waiting_prefill.push_front(request)
        if admitted:
            self.admitted_prefill_batches += 1
        return tuple(admitted)

    def _plan_decode_batches(self) -> tuple[GRContinuousDecodeBatch, ...]:
        groups: dict[tuple[int, int, int, int], list[str]] = defaultdict(list)
        for state in self.decoding.values():
            groups[
                (
                    state.current_decode_step,
                    state.beam_width,
                    _policy_width_for_step(
                        state.request,
                        step=state.current_decode_step + 1,
                        fallback=state.beam_width,
                    ),
                    _request_context_tokens(state.request),
                )
            ].append(state.request_id)

        batches: list[GRContinuousDecodeBatch] = []
        for (step, beam_width, next_beam_width, context_len), request_ids in sorted(
            groups.items()
        ):
            for chunk in _chunks(tuple(request_ids), self.policy.max_decode_batch_size):
                batches.append(
                    GRContinuousDecodeBatch(
                        step=step,
                        beam_width=beam_width,
                        next_beam_width=next_beam_width,
                        context_len=context_len,
                        request_ids=chunk,
                    )
                )
        self.planned_decode_batches += len(batches)
        return tuple(batches)

    def _advance_decode_batches(
        self,
        decode_batches: tuple[GRContinuousDecodeBatch, ...],
    ) -> tuple[str, ...]:
        finished: list[str] = []
        for decode_batch in decode_batches:
            for request_id in decode_batch.request_ids:
                state = self.decoding.get(request_id)
                if state is None:
                    continue
                state.current_decode_step += 1
                if state.current_decode_step >= state.request.max_decode_steps:
                    state.stage = "finished"
                    state.finished_tick = self.tick_count
                    state.stop_reason = "max_decode_steps"
                    self.decoding.pop(request_id, None)
                    self._release_kv_lease(state)
                    self._store_finished_response(
                        GRServingResponse(
                            request_id=request_id,
                            token_ids=(),
                            scores=(),
                            metadata={
                                **state.request.metadata,
                                "continuous_batching": True,
                                "decode_steps": state.current_decode_step,
                                "stop_reason": state.stop_reason,
                                "submitted_tick": state.submitted_tick,
                                "admitted_tick": state.admitted_tick,
                                "finished_tick": state.finished_tick,
                            },
                        )
                    )
                    self.succeeded_requests += 1
                    finished.append(request_id)
                else:
                    state.active_beam_width = decode_batch.next_beam_width
        return tuple(finished)

    def _fail_request_ids(
        self,
        request_ids: tuple[str, ...],
        *,
        reason: str,
        error: BaseException | str | None = None,
    ) -> tuple[str, ...]:
        failed: list[str] = []
        for request_id in request_ids:
            state = self.states.get(request_id)
            if state is None or state.stage == "finished":
                continue
            response = self._fail_state(state, reason=reason, error=error)
            failed.append(response.request_id)
        return tuple(failed)

    def _fail_state(
        self,
        state: GRContinuousRequestState,
        *,
        reason: str,
        error: BaseException | str | None = None,
    ) -> GRServingResponse:
        if state.stage == "waiting_prefill":
            self.waiting_prefill.remove(state.request_id)
        elif state.stage == "decoding":
            self.decoding.pop(state.request_id, None)
            self._release_kv_lease(state)

        state.stage = "finished"
        state.finished_tick = self.tick_count
        state.stop_reason = reason
        metadata: dict[str, Any] = {
            **state.request.metadata,
            "continuous_batching": True,
            "failed": True,
            "decode_steps": state.current_decode_step,
            "stop_reason": reason,
            "submitted_tick": state.submitted_tick,
            "admitted_tick": state.admitted_tick,
            "finished_tick": state.finished_tick,
        }
        timeout_ticks = _request_timeout_ticks(state.request)
        if timeout_ticks is not None:
            metadata["timeout_ticks"] = timeout_ticks
        if error is not None:
            metadata["error_type"] = (
                type(error).__name__ if not isinstance(error, str) else "Error"
            )
            metadata["error_message"] = str(error)
        response = GRServingResponse(
            request_id=state.request_id,
            token_ids=(),
            scores=(),
            metadata=metadata,
        )
        self._store_finished_response(response)
        self.failed_requests += 1
        if reason in {"request_timeout", "timeout"}:
            self.timed_out_requests += 1
        return response

    def _store_finished_response(self, response: GRServingResponse) -> None:
        self.finished[response.request_id] = response
        self._enforce_finished_retention()

    def _enforce_finished_retention(self) -> None:
        max_finished = self.policy.max_finished_requests
        if max_finished is None:
            return
        while len(self.finished) > max_finished:
            self.finished.pop(next(iter(self.finished)))
            self.evicted_finished_requests += 1

    def _fail_timed_out_requests(self) -> tuple[str, ...]:
        timed_out: list[str] = []
        for state in tuple(self.states.values()):
            if state.stage == "finished":
                continue
            timeout_ticks = _request_timeout_ticks(state.request)
            if timeout_ticks is None:
                continue
            if self.tick_count - state.submitted_tick > timeout_ticks:
                response = self._fail_state(
                    state,
                    reason="request_timeout",
                    error=f"request exceeded timeout_ticks={timeout_ticks}",
                )
                timed_out.append(response.request_id)
        return tuple(timed_out)

    def _memory_usage(self) -> dict[str, int]:
        return _require_allocator(self.kv_allocator).usage()

    def _kv_health_status(
        self, allocator_status: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if allocator_status is None:
            allocator_status = _require_allocator(self.kv_allocator).status()
        decoding_request_ids = set(self.decoding)
        lease_request_ids = set(allocator_status.get("lease_request_ids", ()))
        orphaned = sorted(lease_request_ids - decoding_request_ids)
        missing = sorted(decoding_request_ids - lease_request_ids)
        return {
            "kv_allocator_leak_detected": bool(orphaned or missing),
            "kv_allocator_orphaned_leases": orphaned,
            "kv_allocator_missing_leases": missing,
            "kv_allocator_orphaned_lease_count": len(orphaned),
            "kv_allocator_missing_lease_count": len(missing),
        }

    def _can_allocate_kv(self, request: GRServingRequest) -> bool:
        return _require_allocator(self.kv_allocator).can_allocate(
            request_id=request.request_id,
            context_tokens=_request_context_tokens(request),
            beam_slots=_request_beam_slots(request),
        )

    def _allocate_kv_lease(self, request: GRServingRequest) -> GRKVLease:
        lease = _require_allocator(self.kv_allocator).allocate(
            request_id=request.request_id,
            context_tokens=_request_context_tokens(request),
            beam_slots=_request_beam_slots(request),
        )
        self._record_kv_event(
            "allocate",
            request_id=request.request_id,
            context_tokens=lease.context_tokens,
            beam_slots=lease.beam_slots,
        )
        return lease

    def _release_kv_lease(self, state: GRContinuousRequestState) -> None:
        if state.kv_lease is None:
            return
        lease = _require_allocator(self.kv_allocator).release(state.request_id)
        if lease is not None:
            self._record_kv_event(
                "release",
                request_id=state.request_id,
                context_tokens=lease.context_tokens,
                beam_slots=lease.beam_slots,
            )
        state.kv_lease = None

    def _record_kv_event(self, event_type: str, **payload: Any) -> None:
        self.kv_events.append(
            {
                "type": event_type,
                "tick": self.tick_count,
                **payload,
            }
        )
        if self.max_kv_events > 0 and len(self.kv_events) > self.max_kv_events:
            del self.kv_events[: len(self.kv_events) - self.max_kv_events]

    def _avg_prefill_batch_size(self) -> float:
        sizes = [len(tick["prefill_request_ids"]) for tick in self.tick_history]
        non_empty = [size for size in sizes if size]
        if not non_empty:
            return 0.0
        return sum(non_empty) / len(non_empty)

    def _avg_decode_batch_size(self) -> float:
        sizes = [
            batch["size"]
            for tick in self.tick_history
            for batch in tick["decode_batches"]
        ]
        if not sizes:
            return 0.0
        return sum(sizes) / len(sizes)


@dataclass
class GRContinuousServingExecutor:
    """Execute continuous batching ticks through a real serving engine."""

    engine: Any
    scheduler: GRContinuousScheduler = field(default_factory=GRContinuousScheduler)
    synchronize: Any | None = None
    sync_timing: bool = True
    timing_recorder: Any | None = None
    beam_kv_pool: GRDenseBeamKVPool | None = None
    context_kv_pool: GRDenseContextKVPool | None = None
    prefill_cache_enabled: bool = False
    max_prefill_cache_entries: int | None = None
    max_prefill_cache_tokens: int | None = None
    prefill_cache_page_size: int | None = None
    min_prefill_cache_prefix_tokens: int | None = None
    max_prefill_cache_decode_extend_tokens: int | None = None
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    prefill_cache: GRPromptPrefixCache = field(default_factory=GRPromptPrefixCache)
    batched_prefill_cache: dict[tuple[str, ...], PrefillResult] = field(
        default_factory=dict
    )
    decode_inputs_cache: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    topk_indices_cache: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    prefill_cache_hits: int = 0
    prefill_cache_exact_hits: int = 0
    prefill_cache_prefix_hits: int = 0
    prefill_cache_misses: int = 0
    prefill_cache_prefix_tokens: int = 0
    prefill_cache_extend_tokens: int = 0
    prefill_cache_skips: int = 0
    decode_inputs_cache_hits: int = 0
    decode_inputs_cache_misses: int = 0
    topk_indices_cache_hits: int = 0
    topk_indices_cache_misses: int = 0
    decode_cuda_graph_batch_buckets: tuple[int, ...] | None = None
    decode_cuda_graph_padding_requests: int = 0
    decode_cuda_graph_padding_applied: int = 0
    decode_cuda_graph_padding_slots: int = 0
    decode_cuda_graph_padding_skips: int = 0
    decode_cuda_graph_padding_skip_reasons: dict[str, int] = field(default_factory=dict)
    decode_cuda_graph_padding_cache: dict[tuple[Any, ...], Any] = field(
        default_factory=dict
    )
    decode_cuda_graph_padding_buffer_hits: int = 0
    decode_cuda_graph_padding_buffer_misses: int = 0
    decode_cuda_graph_dynamic_skips: int = 0
    decode_cuda_graph_dynamic_skip_reasons: dict[str, int] = field(default_factory=dict)
    decode_cuda_graph_runner: GRDecodeCudaGraphRunner | None = field(
        default=None,
        init=False,
    )
    prefill_cuda_graph_runner: GRPrefillCudaGraphRunner | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.max_prefill_cache_entries is None:
            self.max_prefill_cache_entries = _env_int(
                "GR_INFERENCE_PREFILL_CACHE_MAX_ENTRIES",
                default=16,
            )
        if self.max_prefill_cache_tokens is None:
            max_tokens = _env_int(
                "GR_INFERENCE_PREFILL_CACHE_MAX_TOKENS",
                default=0,
            )
            self.max_prefill_cache_tokens = max_tokens or None
        if self.prefill_cache_page_size is None:
            self.prefill_cache_page_size = _env_int(
                "GR_INFERENCE_PREFILL_CACHE_PAGE_SIZE",
                default=1,
            )
        if self.min_prefill_cache_prefix_tokens is None:
            self.min_prefill_cache_prefix_tokens = _env_int(
                "GR_INFERENCE_PREFILL_CACHE_MIN_PREFIX_TOKENS",
                default=32,
            )
        if self.max_prefill_cache_decode_extend_tokens is None:
            self.max_prefill_cache_decode_extend_tokens = _env_int(
                "GR_INFERENCE_PREFILL_CACHE_MAX_DECODE_EXTEND_TOKENS",
                default=0,
            )
        if self.decode_cuda_graph_batch_buckets is None:
            self.decode_cuda_graph_batch_buckets = _env_int_tuple(
                "GR_INFERENCE_DECODE_CUDA_GRAPH_BATCH_BUCKETS",
                default=(1, 2, 4, 8),
            )
        shared_caches = getattr(self.engine, "_continuous_decode_template_caches", None)
        if shared_caches is None:
            shared_caches = {
                "prefill_cache": GRPromptPrefixCache(
                    max_entries=self.max_prefill_cache_entries,
                    max_tokens=self.max_prefill_cache_tokens,
                    page_size=self.prefill_cache_page_size,
                ),
                "decode_inputs_cache": {},
                "topk_indices_cache": {},
            }
            try:
                setattr(
                    self.engine, "_continuous_decode_template_caches", shared_caches
                )
            except AttributeError:
                # Some unit tests use a plain object() as a lightweight engine.
                # Keep those executor-local while sharing caches on real engines.
                pass
        shared_caches.setdefault(
            "prefill_cache",
            GRPromptPrefixCache(
                max_entries=self.max_prefill_cache_entries,
                max_tokens=self.max_prefill_cache_tokens,
                page_size=self.prefill_cache_page_size,
            ),
        )
        self.decode_inputs_cache = shared_caches["decode_inputs_cache"]
        self.topk_indices_cache = shared_caches["topk_indices_cache"]
        shared_prefill_cache = shared_caches["prefill_cache"]
        if not isinstance(shared_prefill_cache, GRPromptPrefixCache):
            shared_prefill_cache = GRPromptPrefixCache(
                max_entries=self.max_prefill_cache_entries,
                max_tokens=self.max_prefill_cache_tokens,
                page_size=self.prefill_cache_page_size,
            )
            shared_caches["prefill_cache"] = shared_prefill_cache
        shared_prefill_cache.configure(
            max_entries=self.max_prefill_cache_entries,
            max_tokens=self.max_prefill_cache_tokens,
            page_size=self.prefill_cache_page_size,
        )
        self.prefill_cache = shared_prefill_cache
        if (
            not _env_flag("GR_INFERENCE_DISABLE_PREFILL_CUDA_GRAPH")
            and hasattr(self.engine, "model")
            and callable(getattr(self.engine.model, "forward_prefill", None))
        ):
            self.prefill_cuda_graph_runner = getattr(
                self.engine,
                "_prefill_cuda_graph_runner",
                None,
            )
            if self.prefill_cuda_graph_runner is None:
                self.prefill_cuda_graph_runner = GRPrefillCudaGraphRunner(
                    self.engine.model,
                )
                setattr(
                    self.engine,
                    "_prefill_cuda_graph_runner",
                    self.prefill_cuda_graph_runner,
                )
        if (
            not _env_flag("GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH")
            and hasattr(self.engine, "model")
            and hasattr(self.engine, "decode_engine")
        ):
            self.decode_cuda_graph_runner = getattr(
                self.engine,
                "_decode_cuda_graph_runner",
                None,
            )
            if self.decode_cuda_graph_runner is None:
                self.decode_cuda_graph_runner = GRDecodeCudaGraphRunner(
                    self.engine.model,
                    self.engine.decode_engine,
                )
                setattr(
                    self.engine,
                    "_decode_cuda_graph_runner",
                    self.decode_cuda_graph_runner,
                )

    def submit(self, request: GRServingRequest) -> None:
        self.scheduler.submit(request)

    def cancel(
        self, request_id: str, *, reason: str = "cancelled"
    ) -> GRServingResponse:
        response = self.scheduler.cancel(request_id, reason=reason)
        self._attach_execution_metadata((request_id,))
        return response

    def fail(
        self,
        request_id: str,
        *,
        reason: str = "failed",
        error: BaseException | str | None = None,
    ) -> GRServingResponse:
        response = self.scheduler.fail(request_id, reason=reason, error=error)
        self._attach_execution_metadata((request_id,))
        return response

    def tick(self) -> GRContinuousTickResult:
        result = self.scheduler.tick(
            prefill_executor=self._run_prefill,
            decode_executor=self._run_decode_batches,
        )
        self._attach_execution_metadata(result.finished_request_ids)
        return result

    def run_until_empty(
        self,
        *,
        max_ticks: int | None = None,
        timeout_unfinished: bool = False,
    ) -> tuple[GRServingResponse, ...]:
        ticks = 0
        while len(self.scheduler.waiting_prefill) or self.scheduler.decoding:
            if max_ticks is not None and ticks >= max_ticks:
                if timeout_unfinished:
                    failed_ids = self.scheduler.fail_unfinished(reason="timeout")
                    self._attach_execution_metadata(failed_ids)
                break
            self.tick()
            ticks += 1
        return tuple(self.scheduler.finished.values())

    def status(self) -> dict[str, Any]:
        status = self.scheduler.status()
        status.update(self._execution_status())
        status["prefill_cache"] = self._prefill_cache_status()
        status["topk_indices_cache"] = _cache_status(
            self.topk_indices_cache,
            hits=self.topk_indices_cache_hits,
            misses=self.topk_indices_cache_misses,
        )
        status["decode_inputs_cache"] = _cache_status(
            self.decode_inputs_cache,
            hits=self.decode_inputs_cache_hits,
            misses=self.decode_inputs_cache_misses,
        )
        if self.beam_kv_pool is not None:
            beam_kv_pool_status = self.beam_kv_pool.status()
            status["beam_kv_pool"] = beam_kv_pool_status
            status["beam_kv_pool_health"] = self._beam_kv_pool_health_status(
                beam_kv_pool_status
            )
        if self.context_kv_pool is not None:
            context_kv_pool_status = self.context_kv_pool.status()
            status["context_kv_pool"] = context_kv_pool_status
        if self.prefill_cuda_graph_runner is not None:
            status["prefill_cuda_graph"] = self.prefill_cuda_graph_runner.status()
        if self.decode_cuda_graph_runner is not None:
            status["decode_cuda_graph"] = self._decode_cuda_graph_status()
        return status

    def metrics(self) -> dict[str, float | int]:
        metrics = dict(self.scheduler.metrics())
        metrics.update(self._execution_metrics())
        metrics.update(
            _numeric_status_metrics("prefill_cache", self._prefill_cache_status())
        )
        metrics.update(
            _numeric_status_metrics(
                "decode_inputs_cache",
                _cache_status(
                    self.decode_inputs_cache,
                    hits=self.decode_inputs_cache_hits,
                    misses=self.decode_inputs_cache_misses,
                ),
            )
        )
        metrics.update(
            _numeric_status_metrics(
                "topk_indices_cache",
                _cache_status(
                    self.topk_indices_cache,
                    hits=self.topk_indices_cache_hits,
                    misses=self.topk_indices_cache_misses,
                ),
            )
        )
        if self.beam_kv_pool is not None:
            metrics.update(self.beam_kv_pool.usage())
            metrics.update(
                _numeric_status_metrics(
                    "beam_kv_pool",
                    self.beam_kv_pool.status(),
                )
            )
            metrics.update(
                _numeric_status_metrics(
                    "beam_kv_pool_health",
                    self._beam_kv_pool_health_status(),
                )
            )
        if self.context_kv_pool is not None:
            metrics.update(self.context_kv_pool.usage())
            metrics.update(
                _numeric_status_metrics(
                    "context_kv_pool",
                    self.context_kv_pool.status(),
                )
            )
        if self.prefill_cuda_graph_runner is not None:
            metrics.update(
                _numeric_status_metrics(
                    "prefill_cuda_graph",
                    self.prefill_cuda_graph_runner.status(),
                )
            )
        if self.decode_cuda_graph_runner is not None:
            metrics.update(
                _numeric_status_metrics(
                    "decode_cuda_graph",
                    self._decode_cuda_graph_status(),
                )
            )
        return metrics

    def _execution_status(self) -> dict[str, Any]:
        return {
            "continuous_execution": "model_step",
            "sync_timing": self.sync_timing,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "total_ms": self.prefill_ms + self.decode_ms,
        }

    def _execution_metrics(self) -> dict[str, float | int]:
        return {
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "total_ms": self.prefill_ms + self.decode_ms,
            "sync_timing_enabled": int(self.sync_timing),
        }

    def _prefill_cache_status(self) -> dict[str, Any]:
        return {
            "enabled": self.prefill_cache_enabled,
            "entries": len(self.prefill_cache),
            "hits": self.prefill_cache_hits,
            "exact_hits": self.prefill_cache_exact_hits,
            "prefix_hits": self.prefill_cache_prefix_hits,
            "misses": self.prefill_cache_misses,
            "skips": self.prefill_cache_skips,
            "prefix_tokens": self.prefill_cache_prefix_tokens,
            "extend_tokens": self.prefill_cache_extend_tokens,
            "max_entries": self.max_prefill_cache_entries,
            "max_tokens": self.max_prefill_cache_tokens,
            "page_size": self.prefill_cache_page_size,
            "min_prefix_tokens": self.min_prefill_cache_prefix_tokens,
            "max_decode_extend_tokens": self.max_prefill_cache_decode_extend_tokens,
            "extend_mode": self._prefill_cache_extend_mode(),
            **self.prefill_cache.status(),
        }

    def _decode_cuda_graph_status(self) -> dict[str, Any]:
        return {
            **self.decode_cuda_graph_runner.status(),
            "decode_cuda_graph_batch_buckets": self.decode_cuda_graph_batch_buckets,
            "decode_cuda_graph_padding_requests": self.decode_cuda_graph_padding_requests,
            "decode_cuda_graph_padding_applied": self.decode_cuda_graph_padding_applied,
            "decode_cuda_graph_padding_slots": self.decode_cuda_graph_padding_slots,
            "decode_cuda_graph_padding_skips": self.decode_cuda_graph_padding_skips,
            "decode_cuda_graph_padding_buffer_entries": len(
                self.decode_cuda_graph_padding_cache
            ),
            "decode_cuda_graph_padding_buffer_hits": self.decode_cuda_graph_padding_buffer_hits,
            "decode_cuda_graph_padding_buffer_misses": self.decode_cuda_graph_padding_buffer_misses,
            "decode_cuda_graph_dynamic_skips": self.decode_cuda_graph_dynamic_skips,
            **{
                f"decode_cuda_graph_padding_skip_{reason}": count
                for reason, count in sorted(
                    self.decode_cuda_graph_padding_skip_reasons.items()
                )
            },
            **{
                f"decode_cuda_graph_dynamic_skip_{reason}": count
                for reason, count in sorted(
                    self.decode_cuda_graph_dynamic_skip_reasons.items()
                )
            },
        }

    def _run_prefill(self, request_ids: tuple[str, ...]) -> None:
        if not request_ids:
            return

        with _torch_no_grad_context():
            requests_by_shape: dict[Any, list[GRServingRequest]] = defaultdict(list)
            for request_id in request_ids:
                request = self.scheduler.states[request_id].request
                requests_by_shape[getattr(request.input_ids, "shape", None)].append(
                    request
                )
            for requests in requests_by_shape.values():
                start = self._start_timer()
                with self._profile_section("continuous.prefill"):
                    self._run_prefill_requests(requests)
                self.prefill_ms += self._elapsed_ms(start)

    def _run_prefill_requests(self, requests: list[GRServingRequest]) -> None:
        if not self.prefill_cache_enabled:
            self._run_uncached_prefill_requests(requests)
            return

        misses: list[GRServingRequest] = []
        for request in requests:
            cached_prefill = self._prefill_from_cache(request)
            if cached_prefill is None:
                self.prefill_cache_misses += 1
                misses.append(request)
                continue
            self._store_single_prefill_result(request, cached_prefill)

        if not misses:
            return

        self._run_uncached_prefill_requests(misses, update_cache=True)

    def _run_uncached_prefill_requests(
        self,
        requests: list[GRServingRequest],
        *,
        update_cache: bool = False,
    ) -> None:
        import torch

        input_ids = torch.cat([request.input_ids for request in requests], dim=0)
        input_ids = input_ids.to(
            device=self._model_device(input_ids.device),
            non_blocking=True,
        )
        context_kv = self._context_kv_for_prefill(requests)
        prefill = self._forward_prefill(
            input_ids,
            context_kv=context_kv,
            last_token_logits_only=True,
        )
        self._store_prefill_results(tuple(requests), prefill, update_cache=update_cache)

    def _forward_prefill(
        self,
        input_ids: Any,
        *,
        context_kv: ContextKV | None,
        last_token_logits_only: bool,
    ) -> PrefillResult:
        if (
            self.prefill_cuda_graph_runner is not None
            and _profile_allows_prefill_cuda_graph(self.timing_recorder)
        ):
            prefill = self.prefill_cuda_graph_runner.forward_prefill(
                input_ids,
                context_kv=context_kv,
                last_token_logits_only=last_token_logits_only,
            )
            if prefill is not None:
                return prefill
        return self.engine.model.forward_prefill(
            input_ids,
            context_kv=context_kv,
            return_result=True,
            timing_recorder=self.timing_recorder,
            last_token_logits_only=last_token_logits_only,
        )

    def _store_prefill_results(
        self,
        requests: tuple[GRServingRequest, ...],
        prefill: PrefillResult,
        *,
        update_cache: bool = False,
    ) -> None:
        batched_initial_selection = self._select_initial_prefill_beams_batched(
            requests,
            prefill,
        )
        for index, request in enumerate(requests):
            per_request_prefill = PrefillResult(
                logits=prefill.logits[index : index + 1],
                context_kv=prefill.context_kv.slice_batch(index),
                hidden_states=(
                    prefill.hidden_states[index : index + 1]
                    if prefill.hidden_states is not None
                    else None
                ),
            )
            if update_cache and self.prefill_cache_enabled:
                self._store_prefill_cache_entry(
                    request.input_ids,
                    per_request_prefill,
                    extra_key=_prefill_cache_extra_key(request),
                )
            initial_selection = (
                BeamSelection(
                    token_ids=batched_initial_selection.token_ids[index],
                    scores=batched_initial_selection.scores[index],
                    parent_beams=batched_initial_selection.parent_beams[index],
                )
                if batched_initial_selection is not None
                else None
            )
            self._store_single_prefill_result(
                request,
                per_request_prefill,
                initial_selection=initial_selection,
            )

    def _select_initial_prefill_beams_batched(
        self,
        requests: tuple[GRServingRequest, ...],
        prefill: PrefillResult,
    ) -> BatchedBeamSelection | None:
        if self.engine.config.return_beam_details:
            return None
        if len(requests) <= 1:
            return None
        logits = prefill.logits
        if not hasattr(logits, "dim") or getattr(logits, "shape", None) is None:
            return None
        if int(logits.shape[0]) != len(requests):
            return None
        beam_width = requests[0].beam_width
        for request in requests:
            if (
                request.beam_width != beam_width
                or request.item_mask_provider is not None
                or request.beam_width_policy is not None
                or request.logits_processors
            ):
                return None
        return select_initial_topk_batched(
            logits,
            beam_width=beam_width,
            score_mode=self.engine.config.beam_score_mode,
        )

    def _store_single_prefill_result(
        self,
        request: GRServingRequest,
        per_request_prefill: PrefillResult,
        *,
        initial_selection: BeamSelection | None = None,
    ) -> None:
        state = self.scheduler.states[request.request_id]
        generation = GRGenerationState.from_prefill(
            request_id=request.request_id,
            prefill=per_request_prefill,
            max_decode_steps=request.max_decode_steps,
            max_beam_width=self.engine.config.max_beam_width,
            fixed_beam_width=request.beam_width,
            beam_score_mode=self.engine.config.beam_score_mode,
            beam_kv=self._allocate_beam_kv_from_pool(state),
        )
        if initial_selection is None:
            initial_mask = (
                request.item_mask_provider.initial_mask(per_request_prefill.logits)
                if request.item_mask_provider is not None
                else None
            )
            initial_width = _policy_width_for_step(
                request,
                step=0,
                fallback=request.beam_width,
            )
            initial_width = batched_item_mask_limited_beam_width(
                initial_width,
                initial_mask,
            )
            initial_logits = apply_logits_processors(
                per_request_prefill.logits,
                tuple(request.logits_processors),
                LogitsProcessorContext(
                    request_id=request.request_id,
                    phase="prefill",
                    step=0,
                    beam_width=initial_width,
                    beam_path=generation.beam_path,
                    metadata=request.metadata,
                ),
            )
            generation.initialize_beams_with_width(
                initial_width,
                item_mask=initial_mask,
                logits=initial_logits,
            )
        else:
            initial_width = initial_selection.width
            generation.initialize_beams_from_selection(initial_selection)
        state.active_beam_width = initial_width
        state.generation = generation
        _initialize_decode_tensor_state(
            state,
            device=getattr(per_request_prefill.logits, "device", None),
        )
        _observe_request_policy_scores(
            request,
            step=0,
            scores=generation.beam_path.entries[-1].scores,
        )
        if self.engine.config.return_beam_details:
            state.token_logprob_steps = [
                _selected_initial_token_logprobs(
                    initial_logits,
                    generation.beam_path.entries[-1].token_ids,
                )
            ]

    def _store_prefill_cache_entry(
        self,
        input_ids: Any,
        prefill: PrefillResult,
        *,
        extra_key: Any = None,
    ) -> None:
        max_entries = self.max_prefill_cache_entries
        if max_entries == 0:
            return
        self.prefill_cache.configure(
            max_entries=max_entries,
            max_tokens=self.max_prefill_cache_tokens,
            page_size=self.prefill_cache_page_size or 1,
        )
        self.prefill_cache.insert(
            input_ids,
            _clone_prefill_for_cache(prefill),
            extra_key=extra_key,
        )

    def _prefill_from_cache(
        self,
        request: GRServingRequest,
    ) -> PrefillResult | None:
        match = self.prefill_cache.match(
            request.input_ids,
            extra_key=_prefill_cache_extra_key(request),
        )
        if match is None:
            return None

        context_len = _request_context_tokens(request)
        suffix_len = context_len - match.prefix_len
        if match.exact and suffix_len == 0:
            self.prefill_cache_hits += 1
            self.prefill_cache_exact_hits += 1
            self.prefill_cache_prefix_tokens += context_len
            return self._materialize_cached_prefill(request, match.prefill)

        if suffix_len <= 0:
            self.prefill_cache_skips += 1
            return None
        if not self._can_extend_cached_prefix(match, suffix_len=suffix_len):
            self.prefill_cache_skips += 1
            return None

        self.prefill_cache_hits += 1
        self.prefill_cache_prefix_hits += 1
        self.prefill_cache_prefix_tokens += match.prefix_len
        self.prefill_cache_extend_tokens += suffix_len
        return self._extend_cached_prefix_prefill(request, match, suffix_len=suffix_len)

    def _can_extend_cached_prefix(
        self,
        match: GRPrefixCacheMatch,
        *,
        suffix_len: int,
    ) -> bool:
        min_prefix = int(self.min_prefill_cache_prefix_tokens or 0)
        max_extend = self.max_prefill_cache_decode_extend_tokens
        if match.prefix_len < min_prefix:
            return False
        if max_extend is not None and suffix_len > max_extend:
            return False
        if match.prefill.context_len < match.prefix_len:
            return False
        if callable(getattr(self.engine.model, "forward_prefill_extend", None)):
            return True
        if not _env_flag("GR_INFERENCE_PREFILL_CACHE_ENABLE_DECODE_EXTEND"):
            return False
        return (
            hasattr(self.engine, "decode_engine")
            and hasattr(self.engine, "model")
            and hasattr(self.engine.model, "forward_decode_step")
        )

    def _prefill_cache_extend_mode(self) -> str:
        model = getattr(self.engine, "model", None)
        if callable(getattr(model, "forward_prefill_extend", None)):
            return "prefill_extend"
        if _env_flag("GR_INFERENCE_PREFILL_CACHE_ENABLE_DECODE_EXTEND"):
            return "decode_extend"
        return "disabled"

    def _materialize_cached_prefill(
        self,
        request: GRServingRequest,
        cached: PrefillResult,
    ) -> PrefillResult:
        if self.context_kv_pool is None:
            return cached
        if not self.context_kv_pool.can_allocate(
            request.request_id,
            context_len=cached.context_len,
        ):
            return cached
        lease = self.context_kv_pool.allocate(
            request.request_id,
            context_len=cached.context_len,
        )
        _copy_tensor(lease.context_kv.key, cached.context_kv.key)
        _copy_tensor(lease.context_kv.value, cached.context_kv.value)
        self.scheduler.states[request.request_id].context_kv_pool_lease = lease
        return PrefillResult(
            logits=cached.logits,
            context_kv=lease.context_kv,
            hidden_states=cached.hidden_states,
        )

    def _extend_cached_prefix_prefill(
        self,
        request: GRServingRequest,
        match: GRPrefixCacheMatch,
        *,
        suffix_len: int,
    ) -> PrefillResult:
        forward_prefill_extend = getattr(
            self.engine.model, "forward_prefill_extend", None
        )
        if callable(forward_prefill_extend):
            return self._prefill_extend_cached_prefix_prefill(
                request,
                match,
                suffix_len=suffix_len,
                forward_prefill_extend=forward_prefill_extend,
            )
        return self._decode_extend_cached_prefix_prefill(
            request,
            match,
            suffix_len=suffix_len,
        )

    def _prefill_extend_cached_prefix_prefill(
        self,
        request: GRServingRequest,
        match: GRPrefixCacheMatch,
        *,
        suffix_len: int,
        forward_prefill_extend: Any,
    ) -> PrefillResult:
        input_ids = request.input_ids.to(
            device=self._model_device(request.input_ids.device),
            non_blocking=True,
        )
        prefix_len = int(match.prefix_len)
        context_len = int(input_ids.shape[-1])
        suffix_ids = input_ids[:, prefix_len:]
        if int(suffix_ids.shape[-1]) != suffix_len:
            raise ValueError(
                "prefix cache suffix length changed during materialization"
            )

        cached_prefix = _slice_context_kv(match.prefill.context_kv, prefix_len)
        full_context_kv = self._allocate_context_kv_for_prefill_cache_extend(
            request,
            context_len=context_len,
            reference=cached_prefix,
            device=input_ids.device,
        )
        prefill = forward_prefill_extend(
            suffix_ids,
            cached_prefix,
            context_kv=full_context_kv,
            return_result=True,
            timing_recorder=self.timing_recorder,
            last_token_logits_only=True,
        )
        self._store_prefill_cache_entry(
            request.input_ids,
            prefill,
            extra_key=_prefill_cache_extra_key(request),
        )
        return prefill

    def _decode_extend_cached_prefix_prefill(
        self,
        request: GRServingRequest,
        match: GRPrefixCacheMatch,
        *,
        suffix_len: int,
    ) -> PrefillResult:
        pass

        input_ids = request.input_ids.to(
            device=self._model_device(request.input_ids.device),
            non_blocking=True,
        )
        prefix_len = int(match.prefix_len)
        context_len = int(input_ids.shape[-1])
        suffix_ids = input_ids[:, prefix_len:]
        if int(suffix_ids.shape[-1]) != suffix_len:
            raise ValueError(
                "prefix cache suffix length changed during materialization"
            )

        cached_prefix = _slice_context_kv(match.prefill.context_kv, prefix_len)
        full_context_kv = self._allocate_context_kv_for_prefill_cache_extend(
            request,
            context_len=context_len,
            reference=cached_prefix,
            device=input_ids.device,
        )
        _copy_context_prefix(full_context_kv, cached_prefix, prefix_len=prefix_len)

        prefix_prefill = PrefillResult(
            logits=match.prefill.logits,
            context_kv=cached_prefix,
            hidden_states=None,
        )
        beam_kv = allocate_beam_kv_like_context(
            cached_prefix,
            max_decode_steps=suffix_len,
            max_beam_width=1,
        )
        generation = GRGenerationState.from_prefill(
            request_id=f"{request.request_id}:prefix-cache-extend",
            prefill=prefix_prefill,
            max_decode_steps=suffix_len,
            max_beam_width=1,
            fixed_beam_width=1,
            beam_score_mode=self.engine.config.beam_score_mode,
            beam_kv=beam_kv,
        )
        logits = None
        device = getattr(suffix_ids, "device", None)
        for step in range(suffix_len):
            beam_token_ids = suffix_ids[:, step : step + 1]
            topk_indices = make_compacted_batched_topk_indices(
                batch_size=1,
                num_q_heads=self.engine.model.config.num_attention_heads,
                decode_nums=step + 1,
                beam_width=1,
                device=device,
            )
            logits = self.engine.model.forward_decode_step(
                beam_token_ids,
                generation,
                self.engine.decode_engine,
                step=step,
                active_beam_width=1,
                topk_indices=topk_indices,
                decode_nums=step + 1,
                timing_recorder=self.timing_recorder,
            )

        if logits is None:
            raise RuntimeError("prefix cache extend did not run any suffix tokens")
        _copy_suffix_beam_kv_to_context(
            full_context_kv,
            generation.beam_kv,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
        )
        prefill = PrefillResult(
            logits=logits,
            context_kv=full_context_kv,
            hidden_states=None,
        )
        self._store_prefill_cache_entry(
            request.input_ids,
            prefill,
            extra_key=_prefill_cache_extra_key(request),
        )
        return prefill

    def _allocate_context_kv_for_prefill_cache_extend(
        self,
        request: GRServingRequest,
        *,
        context_len: int,
        reference: ContextKV,
        device: Any | None,
    ) -> ContextKV:
        if self.context_kv_pool is not None and self.context_kv_pool.can_allocate(
            request.request_id,
            context_len=context_len,
        ):
            lease = self.context_kv_pool.allocate(
                request.request_id,
                context_len=context_len,
            )
            self.scheduler.states[request.request_id].context_kv_pool_lease = lease
            return lease.context_kv

        allocate_context_kv = getattr(self.engine.model, "allocate_context_kv", None)
        if callable(allocate_context_kv):
            return allocate_context_kv(
                batch_size=1,
                context_len=context_len,
                device=device,
                dtype=getattr(reference.key, "dtype", None),
            )
        return ContextKV(
            _empty_context_like(reference.key, context_len=context_len),
            _empty_context_like(reference.value, context_len=context_len),
        )

    def _run_decode_batches(
        self,
        decode_batches: tuple[GRContinuousDecodeBatch, ...],
    ) -> tuple[str, ...]:
        finished: list[str] = []
        for decode_batch in decode_batches:
            states = tuple(
                self.scheduler.decoding[request_id]
                for request_id in decode_batch.request_ids
                if request_id in self.scheduler.decoding
            )
            if not states:
                continue
            finished.extend(self._run_decode_batch(decode_batch, states))
        return tuple(finished)

    def _run_decode_batch(
        self,
        decode_batch: GRContinuousDecodeBatch,
        states: tuple[GRContinuousRequestState, ...],
    ) -> tuple[str, ...]:
        if self._can_run_decode_tensor_selection(decode_batch, states):
            return self._run_decode_batch_tensor_selection(decode_batch, states)

        for state in states:
            _flush_pending_decode_tensor_selections(state)

        start = self._start_timer()
        with _torch_no_grad_context(), self._profile_section(
            "continuous.decode_microbatch_total"
        ):
            with self._profile_section("continuous.decode_batch_build"):
                generations = tuple(_require_generation(state) for state in states)
                selection = _current_batched_selection(generations)
                batched_beam_path = BatchedBeamPath(
                    tuple(generation.beam_path for generation in generations)
                )
                decode_inputs = self._decode_inputs_for_selection(
                    selection,
                    request_ids=decode_batch.request_ids,
                    device=getattr(generations[0].prefill.logits, "device", None),
                )
                decode_nums = decode_batch.step + 1
                needs_history_compaction = needs_batched_beam_kv_history_compaction(
                    batched_beam_path,
                    decode_nums=decode_nums,
                    active_beam_width=decode_batch.beam_width,
                )
                batched_generation = _make_batched_generation(
                    request_id="continuous-decode:"
                    + ",".join(decode_batch.request_ids),
                    generations=generations,
                    beam_score_mode=self.engine.config.beam_score_mode,
                    prefill=self._batched_prefill_for_states(states, generations),
                    beam_kv=self._batched_beam_kv_for_states(states),
                    batched_beam_path=batched_beam_path,
                    decode_nums=decode_nums,
                    active_beam_width=decode_batch.beam_width,
                    needs_history_compaction=needs_history_compaction,
                )
            with self._profile_section("continuous.topk_indices"):
                topk_indices = self._topk_indices_for_decode_batch(
                    batched_beam_path,
                    request_ids=decode_batch.request_ids,
                    num_q_heads=self.engine.model.config.num_attention_heads,
                    decode_nums=decode_nums,
                    beam_width=decode_batch.beam_width,
                    device=getattr(generations[0].prefill.logits, "device", None),
                    needs_history_compaction=needs_history_compaction,
                )
            with self._profile_section("model.forward_decode_step"):
                logits = self._forward_decode_step(
                    decode_inputs.beam_token_ids,
                    batched_generation,
                    states=states,
                    step=decode_batch.step,
                    active_beam_width=decode_batch.beam_width,
                    topk_indices=topk_indices,
                    decode_nums=decode_nums,
                )
                logits = _continuous_apply_logits_processors(
                    states,
                    logits,
                    step=decode_batch.step,
                    beam_width=decode_batch.beam_width,
                    batched_beam_path=batched_beam_path,
                )
            with self._profile_section("continuous.beam_selection"):
                item_mask = _continuous_step_item_mask(states, logits)
                next_beam_width = batched_item_mask_limited_beam_width(
                    decode_batch.next_beam_width,
                    item_mask,
                )
                next_selection = select_next_topk_batched(
                    logits,
                    previous_scores=selection.scores,
                    beam_width=next_beam_width,
                    item_mask=item_mask,
                    score_mode=self.engine.config.beam_score_mode,
                )
                token_logprob_rows = (
                    _selected_decode_token_logprobs(logits, next_selection)
                    if self.engine.config.return_beam_details
                    else None
                )
            with self._profile_section("continuous.beam_kv_scatter"):
                _scatter_batched_beam_kv(
                    batched_generation.beam_kv,
                    generations,
                    step=decode_batch.step,
                    active_beam_width=decode_batch.beam_width,
                )
            batched_beam_path.append(next_selection)
        self.decode_ms += self._elapsed_ms(start)

        finished: list[str] = []
        for batch_idx, state in enumerate(states):
            if token_logprob_rows is not None and state.token_logprob_steps is not None:
                state.token_logprob_steps.append(token_logprob_rows[batch_idx])
            state.current_decode_step += 1
            state.active_beam_width = next_selection.beam_width
            _observe_request_policy_scores(
                state.request,
                step=state.current_decode_step,
                scores=next_selection.scores[batch_idx],
            )
            stop_reason = _continuous_stop_reason(
                state,
                next_selection.token_ids[batch_idx],
            )
            if stop_reason is not None:
                self._finish_state(
                    state,
                    token_ids=next_selection.token_ids[batch_idx],
                    scores=next_selection.scores[batch_idx],
                    stop_reason=stop_reason,
                    decode_batch=decode_batch,
                )
                finished.append(state.request_id)
        return tuple(finished)

    def _can_run_decode_tensor_selection(
        self,
        decode_batch: GRContinuousDecodeBatch,
        states: tuple[GRContinuousRequestState, ...],
    ) -> bool:
        if _env_flag("GR_INFERENCE_DISABLE_DECODE_TENSOR_SELECTION"):
            return False
        if self.engine.config.return_beam_details:
            return False
        if decode_batch.beam_width != decode_batch.next_beam_width:
            return False
        decode_nums = decode_batch.step + 1
        for state in states:
            request = state.request
            if (
                request.item_mask_provider is not None
                or request.beam_width_policy is not None
                or request.logits_processors
                or _request_stop_token_ids(request)
            ):
                return False
            if (
                state.decode_selection_token_ids is None
                or state.decode_selection_scores is None
                or len(state.decode_parent_history) < decode_nums
            ):
                return False
            if not _is_cuda_tensor(
                state.decode_selection_token_ids
            ) or not _is_cuda_tensor(state.decode_selection_scores):
                return False
        return True

    def _run_decode_batch_tensor_selection(
        self,
        decode_batch: GRContinuousDecodeBatch,
        states: tuple[GRContinuousRequestState, ...],
    ) -> tuple[str, ...]:
        start = self._start_timer()
        finalized_rows: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {}
        with _torch_no_grad_context(), self._profile_section(
            "continuous.decode_microbatch_total"
        ):
            with self._profile_section("continuous.decode_batch_build"):
                generations = tuple(_require_generation(state) for state in states)
                beam_token_ids = _stack_state_tensor_rows(
                    states,
                    "decode_selection_token_ids",
                )
                previous_scores_tensor = _stack_state_tensor_rows(
                    states,
                    "decode_selection_scores",
                )
                decode_nums = decode_batch.step + 1
                batched_generation = _make_batched_generation(
                    request_id="continuous-decode:"
                    + ",".join(decode_batch.request_ids),
                    generations=generations,
                    beam_score_mode=self.engine.config.beam_score_mode,
                    prefill=self._batched_prefill_for_states(states, generations),
                    beam_kv=self._batched_beam_kv_for_states(states),
                    needs_history_compaction=False,
                )
            with self._profile_section("continuous.topk_indices"):
                topk_indices = _make_tensor_parent_history_topk_indices(
                    states,
                    num_q_heads=self.engine.model.config.num_attention_heads,
                    decode_nums=decode_nums,
                    beam_width=decode_batch.beam_width,
                    device=getattr(beam_token_ids, "device", None),
                )
            with self._profile_section("model.forward_decode_step"):
                logits = self._forward_decode_step(
                    beam_token_ids,
                    batched_generation,
                    states=states,
                    step=decode_batch.step,
                    active_beam_width=decode_batch.beam_width,
                    topk_indices=topk_indices,
                    decode_nums=decode_nums,
                )
            with self._profile_section("continuous.beam_selection"):
                next_selection = select_next_topk_batched(
                    logits,
                    previous_scores_tensor=previous_scores_tensor,
                    beam_width=decode_batch.next_beam_width,
                    score_mode=self.engine.config.beam_score_mode,
                    materialize=False,
                    validate_finite=False,
                )
                _update_decode_tensor_state(states, next_selection)
            with self._profile_section("continuous.beam_kv_scatter"):
                _scatter_batched_beam_kv(
                    batched_generation.beam_kv,
                    generations,
                    step=decode_batch.step,
                    active_beam_width=decode_batch.beam_width,
                )
            with self._profile_section("continuous.decode_finalize"):
                finishing_states = tuple(
                    state
                    for state in states
                    if state.current_decode_step + 1 >= state.request.max_decode_steps
                )
                finalized_rows.update(
                    _flush_pending_decode_tensor_selections_batch(finishing_states)
                )
        self.decode_ms += self._elapsed_ms(start)

        finished: list[str] = []
        for state in states:
            state.current_decode_step += 1
            state.active_beam_width = next_selection.beam_width
            if state.current_decode_step >= state.request.max_decode_steps:
                finalized = finalized_rows.get(state.request_id)
                if finalized is None:
                    finalized = _flush_pending_decode_tensor_selections(state)
                token_ids, scores = finalized
                self._finish_state(
                    state,
                    token_ids=token_ids,
                    scores=scores,
                    stop_reason="max_decode_steps",
                    decode_batch=decode_batch,
                )
                finished.append(state.request_id)
        return tuple(finished)

    def _model_device(self, fallback: Any) -> Any:
        parameters = getattr(self.engine.model, "parameters", None)
        if callable(parameters):
            try:
                return next(parameters()).device
            except StopIteration:
                pass
        return fallback

    def _forward_decode_step(
        self,
        beam_token_ids: Any,
        batched_generation: GRGenerationState,
        *,
        states: tuple[GRContinuousRequestState, ...],
        step: int,
        active_beam_width: int,
        topk_indices: Any | None,
        decode_nums: int,
    ) -> Any:
        if (
            self.decode_cuda_graph_runner is not None
            and _profile_allows_decode_cuda_graph(self.timing_recorder)
        ):
            graph_inputs = self._decode_cuda_graph_inputs(
                beam_token_ids,
                batched_generation,
                topk_indices=topk_indices,
                states=states,
            )
            if graph_inputs.use_cuda_graph:
                logits = self.decode_cuda_graph_runner.forward_decode_step(
                    graph_inputs.beam_token_ids,
                    graph_inputs.generation,
                    step=step,
                    active_beam_width=active_beam_width,
                    topk_indices=graph_inputs.topk_indices,
                    decode_nums=decode_nums,
                )
                if logits is not None:
                    return _slice_tensor_batch(logits, graph_inputs.actual_batch_size)
        return self.engine.model.forward_decode_step(
            beam_token_ids,
            batched_generation,
            self.engine.decode_engine,
            step=step,
            active_beam_width=active_beam_width,
            topk_indices=topk_indices,
            decode_nums=decode_nums,
            timing_recorder=self.timing_recorder,
        )

    def _decode_cuda_graph_inputs(
        self,
        beam_token_ids: Any,
        batched_generation: GRGenerationState,
        *,
        topk_indices: Any | None,
        states: tuple[GRContinuousRequestState, ...],
    ) -> _DecodeCudaGraphInputs:
        actual_batch_size = int(getattr(beam_token_ids, "shape", (len(states),))[0])
        bucket_size = _decode_cuda_graph_bucket_for(
            actual_batch_size,
            self.decode_cuda_graph_batch_buckets or (),
        )
        unpadded = _DecodeCudaGraphInputs(
            beam_token_ids=beam_token_ids,
            generation=batched_generation,
            topk_indices=topk_indices,
            actual_batch_size=actual_batch_size,
            graph_batch_size=actual_batch_size,
        )
        if bucket_size <= actual_batch_size:
            if not self._decode_cuda_graph_has_stable_pool_window(states):
                self._record_decode_cuda_graph_dynamic_skip("pool_window")
                return _DecodeCudaGraphInputs(
                    beam_token_ids=beam_token_ids,
                    generation=batched_generation,
                    topk_indices=topk_indices,
                    actual_batch_size=actual_batch_size,
                    graph_batch_size=actual_batch_size,
                    use_cuda_graph=False,
                )
            return unpadded

        self.decode_cuda_graph_padding_requests += 1
        padded_context_kv = self._context_pool_view_for_states(
            states,
            batch_size=bucket_size,
        )
        if padded_context_kv is None:
            self._record_decode_cuda_graph_padding_skip("context_pool")
            self._record_decode_cuda_graph_dynamic_skip("padding_context_pool")
            return _DecodeCudaGraphInputs(
                beam_token_ids=beam_token_ids,
                generation=batched_generation,
                topk_indices=topk_indices,
                actual_batch_size=actual_batch_size,
                graph_batch_size=actual_batch_size,
                use_cuda_graph=False,
            )
        padded_beam_kv = self._beam_pool_view_for_states(
            states,
            batch_size=bucket_size,
        )
        if padded_beam_kv is None:
            self._record_decode_cuda_graph_padding_skip("beam_pool")
            self._record_decode_cuda_graph_dynamic_skip("padding_beam_pool")
            return _DecodeCudaGraphInputs(
                beam_token_ids=beam_token_ids,
                generation=batched_generation,
                topk_indices=topk_indices,
                actual_batch_size=actual_batch_size,
                graph_batch_size=actual_batch_size,
                use_cuda_graph=False,
            )

        padded_generation = GRGenerationState(
            request_id=f"{batched_generation.request_id}:graph-pad{bucket_size}",
            prefill=PrefillResult(
                logits=batched_generation.prefill.logits,
                context_kv=padded_context_kv,
                hidden_states=None,
            ),
            beam_kv=padded_beam_kv,
            beam_path=batched_generation.beam_path,
            fixed_beam_width=batched_generation.fixed_beam_width,
            beam_score_mode=batched_generation.beam_score_mode,
        )
        self.decode_cuda_graph_padding_applied += 1
        self.decode_cuda_graph_padding_slots += bucket_size - actual_batch_size
        return _DecodeCudaGraphInputs(
            beam_token_ids=self._padded_tensor_batch(
                beam_token_ids,
                bucket_size,
                name="beam_token_ids",
            ),
            generation=padded_generation,
            topk_indices=(
                self._padded_tensor_batch(
                    topk_indices,
                    bucket_size,
                    name="topk_indices",
                )
                if topk_indices is not None
                else None
            ),
            actual_batch_size=actual_batch_size,
            graph_batch_size=bucket_size,
        )

    def _decode_cuda_graph_has_stable_pool_window(
        self,
        states: tuple[GRContinuousRequestState, ...],
    ) -> bool:
        return (
            self._context_pool_view_for_states(states) is not None
            and self._beam_pool_view_for_states(states) is not None
        )

    def _context_pool_view_for_states(
        self,
        states: tuple[GRContinuousRequestState, ...],
        *,
        batch_size: int | None = None,
    ) -> ContextKV | None:
        if self.context_kv_pool is None:
            return None
        leases = _state_pool_leases(states, "context_kv_pool_lease")
        if leases is None:
            return None
        context_lens = {int(lease.context_len) for lease in leases}
        if len(context_lens) != 1:
            return None
        window = _pool_slot_window(
            leases,
            batch_size=batch_size,
            capacity=self.context_kv_pool.capacity,
            leased_slots={
                int(lease.slot) for lease in self.context_kv_pool.leases.values()
            },
        )
        if window is None:
            return None
        first, width = window
        context_len = next(iter(context_lens))
        return ContextKV(
            self.context_kv_pool.key[:, first : first + width, :context_len],
            self.context_kv_pool.value[:, first : first + width, :context_len],
        )

    def _beam_pool_view_for_states(
        self,
        states: tuple[GRContinuousRequestState, ...],
        *,
        batch_size: int | None = None,
    ) -> BeamKV | None:
        if self.beam_kv_pool is None:
            return None
        leases = _state_pool_leases(states, "beam_kv_pool_lease")
        if leases is None:
            return None
        window = _pool_slot_window(
            leases,
            batch_size=batch_size,
            capacity=self.beam_kv_pool.capacity,
            leased_slots={
                int(lease.slot) for lease in self.beam_kv_pool.leases.values()
            },
        )
        if window is None:
            return None
        first, width = window
        return BeamKV(
            self.beam_kv_pool.key[:, first : first + width],
            self.beam_kv_pool.value[:, first : first + width],
        )

    def _record_decode_cuda_graph_padding_skip(self, reason: str) -> None:
        self.decode_cuda_graph_padding_skips += 1
        self.decode_cuda_graph_padding_skip_reasons[reason] = (
            self.decode_cuda_graph_padding_skip_reasons.get(reason, 0) + 1
        )

    def _record_decode_cuda_graph_dynamic_skip(self, reason: str) -> None:
        self.decode_cuda_graph_dynamic_skips += 1
        self.decode_cuda_graph_dynamic_skip_reasons[reason] = (
            self.decode_cuda_graph_dynamic_skip_reasons.get(reason, 0) + 1
        )

    def _padded_tensor_batch(
        self,
        tensor: Any,
        batch_size: int,
        *,
        name: str,
    ) -> Any:
        actual = int(tensor.shape[0])
        if batch_size <= actual:
            return tensor

        cache_key = (
            name,
            str(getattr(tensor, "device", "")),
            getattr(tensor, "dtype", None),
            tuple(int(dim) for dim in tensor.shape[1:]),
            int(batch_size),
        )
        padded = self.decode_cuda_graph_padding_cache.get(cache_key)
        if padded is None:
            padded = tensor.new_zeros((batch_size, *tuple(tensor.shape[1:])))
            self.decode_cuda_graph_padding_cache[cache_key] = padded
            self.decode_cuda_graph_padding_buffer_misses += 1
        else:
            self.decode_cuda_graph_padding_buffer_hits += 1
        padded[:actual].copy_(tensor)
        padded[actual:].zero_()
        return padded

    def _batched_prefill_for_states(
        self,
        states: tuple[GRContinuousRequestState, ...],
        generations: tuple[GRGenerationState, ...],
    ) -> PrefillResult:
        request_ids = tuple(state.request_id for state in states)
        cached = self.batched_prefill_cache.get(request_ids)
        if cached is not None:
            return cached
        context_kv = self._batched_context_kv_for_states(states)
        if context_kv is None:
            prefill = _make_batched_prefill(generations)
        else:
            prefill = PrefillResult(
                logits=_cat_tensors(
                    tuple(generation.prefill.logits for generation in generations),
                    dim=0,
                ),
                context_kv=context_kv,
                hidden_states=None,
            )
        self.batched_prefill_cache[request_ids] = prefill
        return prefill

    def _batched_context_kv_for_states(
        self,
        states: tuple[GRContinuousRequestState, ...],
    ) -> ContextKV | None:
        return self._context_pool_view_for_states(states)

    def _batched_beam_kv_for_states(
        self,
        states: tuple[GRContinuousRequestState, ...],
    ) -> BeamKV | None:
        return self._beam_pool_view_for_states(states)

    def _finish_state(
        self,
        state: GRContinuousRequestState,
        *,
        token_ids: tuple[int, ...],
        scores: tuple[float, ...],
        stop_reason: str,
        decode_batch: GRContinuousDecodeBatch,
    ) -> None:
        state.stage = "finished"
        state.finished_tick = self.scheduler.tick_count
        state.stop_reason = stop_reason
        self.scheduler.decoding.pop(state.request_id, None)
        self.scheduler._release_kv_lease(state)
        response = GRServingResponse(
            request_id=state.request_id,
            token_ids=token_ids,
            scores=scores,
            metadata={
                **state.request.metadata,
                "continuous_batching": True,
                "continuous_execution": "model_step",
                "requested_beam_width": state.request.beam_width,
                "active_beam_width": state.beam_width,
                "decode_steps": state.current_decode_step,
                "prefill_ms": self.prefill_ms,
                "decode_ms": self.decode_ms,
                "total_ms": self.prefill_ms + self.decode_ms,
                "sync_timing": self.sync_timing,
                "stop_reason": stop_reason,
                "submitted_tick": state.submitted_tick,
                "admitted_tick": state.admitted_tick,
                "finished_tick": state.finished_tick,
                "decode_batch": decode_batch.metadata(),
            },
        )
        if state.request.logits_processors:
            response.metadata["logits_processors"] = logits_processors_metadata(
                tuple(state.request.logits_processors)
            )
        _attach_item_results(
            response.metadata,
            request=state.request,
            beam_path=state.generation.beam_path
            if state.generation is not None
            else None,
            beam_width=state.beam_width,
        )
        if state.generation is not None:
            response.metadata["_beam_path"] = state.generation.beam_path
            response.metadata["beam_results"] = _continuous_beam_results(
                state.generation.beam_path,
                beam_width=state.beam_width,
                stop_reason=stop_reason,
            )
        if self.engine.config.return_beam_details and state.generation is not None:
            response.metadata["beam_details"] = _continuous_beam_details(
                state.generation.beam_path,
                beam_width=state.beam_width,
                token_logprob_steps=state.token_logprob_steps,
                score_type=_beam_score_type(self.engine.config.beam_score_mode),
            )
        if state.request.beam_width_policy is not None:
            response.metadata["beam_width_policy"] = _beam_width_policy_metadata(
                state.request.beam_width_policy
            )
        self.scheduler._store_finished_response(response)
        self.scheduler.succeeded_requests += 1

    def _start_timer(self) -> float:
        if self.sync_timing:
            self._synchronize()
        return time.perf_counter()

    def _elapsed_ms(self, start: float) -> float:
        if self.sync_timing:
            self._synchronize()
        return (time.perf_counter() - start) * 1000.0

    def _synchronize(self) -> None:
        if self.synchronize is not None:
            self.synchronize()

    def _profile_section(self, name: str):
        if self.timing_recorder is None:
            return nullcontext()
        return self.timing_recorder.section(name)

    def _attach_execution_metadata(self, request_ids: tuple[str, ...]) -> None:
        for request_id in request_ids:
            response = self.scheduler.finished.get(request_id)
            if response is not None:
                response.metadata.setdefault("continuous_execution", "model_step")
                response.metadata.setdefault("prefill_ms", self.prefill_ms)
                response.metadata.setdefault("decode_ms", self.decode_ms)
                response.metadata.setdefault(
                    "total_ms", self.prefill_ms + self.decode_ms
                )
                response.metadata.setdefault("sync_timing", self.sync_timing)
            self._release_beam_kv_pool_lease(request_id)
            self._release_context_kv_pool_lease(request_id)
            self._clear_finished_runtime_state(request_id)
        self._drop_batched_prefill_cache_for(request_ids)

    def _clear_finished_runtime_state(self, request_id: str) -> None:
        state = self.scheduler.states.get(request_id)
        if state is None or state.stage != "finished":
            return
        state.generation = None
        state.token_logprob_steps = None
        state.decode_selection_token_ids = None
        state.decode_selection_scores = None
        state.decode_parent_history.clear()
        state.pending_decode_token_ids.clear()
        state.pending_decode_scores.clear()
        state.pending_decode_parent_beams.clear()
        state.beam_kv_pool_lease = None
        object.__setattr__(state.request, "input_ids", None)

    def _context_kv_for_prefill(
        self,
        requests: list[GRServingRequest],
    ) -> ContextKV | None:
        if self.context_kv_pool is None:
            return None
        context_lens = {_request_context_tokens(request) for request in requests}
        if len(context_lens) != 1:
            return None
        context_len = next(iter(context_lens))
        request_ids = tuple(request.request_id for request in requests)
        leases = self.context_kv_pool.allocate_batch(
            request_ids,
            context_len=context_len,
        )
        if leases is None:
            return None
        for lease in leases:
            self.scheduler.states[lease.request_id].context_kv_pool_lease = lease
        first_slot = leases[0].slot
        last_slot = leases[-1].slot
        if last_slot - first_slot + 1 != len(leases):
            return None
        return ContextKV(
            self.context_kv_pool.key[:, first_slot : last_slot + 1, :context_len],
            self.context_kv_pool.value[:, first_slot : last_slot + 1, :context_len],
        )

    def _release_context_kv_pool_lease(self, request_id: str) -> None:
        state = self.scheduler.states.get(request_id)
        if state is None or state.context_kv_pool_lease is None:
            return
        if self.context_kv_pool is not None:
            self.context_kv_pool.release(request_id)
        state.context_kv_pool_lease = None
        object.__setattr__(state.request, "input_ids", None)

    def _drop_batched_prefill_cache_for(self, request_ids: tuple[str, ...]) -> None:
        if not request_ids:
            return
        finished = set(request_ids)
        for cache_key in tuple(self.batched_prefill_cache):
            if finished.intersection(cache_key):
                self.batched_prefill_cache.pop(cache_key, None)
        for cache_key in tuple(self.decode_inputs_cache):
            if _request_scoped_cache_key_matches_finished(cache_key, finished):
                self.decode_inputs_cache.pop(cache_key, None)
        for cache_key in tuple(self.topk_indices_cache):
            if _request_scoped_cache_key_matches_finished(cache_key, finished):
                self.topk_indices_cache.pop(cache_key, None)

    def _decode_inputs_for_selection(
        self,
        selection: BatchedBeamSelection,
        *,
        request_ids: tuple[str, ...],
        device: Any | None,
    ) -> Any:
        cache_key = _decode_inputs_cache_key(
            selection,
            request_ids=request_ids,
            device=device,
        )
        cached = self.decode_inputs_cache.get(cache_key)
        if cached is not None:
            self.decode_inputs_cache_hits += 1
            return cached
        self.decode_inputs_cache_misses += 1
        decode_inputs = make_batched_beam_token_ids(selection, device=device)
        self.decode_inputs_cache[cache_key] = decode_inputs
        return decode_inputs

    def _topk_indices_for_decode_batch(
        self,
        batched_beam_path: BatchedBeamPath,
        *,
        request_ids: tuple[str, ...],
        num_q_heads: int,
        decode_nums: int,
        beam_width: int,
        device: Any | None,
        needs_history_compaction: bool,
    ) -> Any:
        cache_key = _topk_indices_cache_key(
            batched_beam_path,
            request_ids=request_ids,
            num_q_heads=num_q_heads,
            decode_nums=decode_nums,
            beam_width=beam_width,
            device=device,
            needs_history_compaction=needs_history_compaction,
        )
        cached = self.topk_indices_cache.get(cache_key)
        if cached is not None:
            self.topk_indices_cache_hits += 1
            return cached
        self.topk_indices_cache_misses += 1
        if needs_history_compaction:
            topk_indices = make_compacted_batched_topk_indices(
                batch_size=batched_beam_path.batch_size,
                num_q_heads=num_q_heads,
                decode_nums=decode_nums,
                beam_width=beam_width,
                device=device,
            )
        else:
            topk_indices = make_batched_topk_indices(
                batched_beam_path,
                num_q_heads=num_q_heads,
                decode_nums=decode_nums,
                beam_width=beam_width,
                device=device,
            )
        self.topk_indices_cache[cache_key] = topk_indices
        return topk_indices

    def _allocate_beam_kv_from_pool(
        self,
        state: GRContinuousRequestState,
    ) -> BeamKV | None:
        if self.beam_kv_pool is None:
            return None
        state.beam_kv_pool_lease = self.beam_kv_pool.allocate(state.request_id)
        return state.beam_kv_pool_lease.beam_kv

    def _release_beam_kv_pool_lease(self, request_id: str) -> None:
        state = self.scheduler.states.get(request_id)
        if state is None or state.beam_kv_pool_lease is None:
            return
        if self.beam_kv_pool is not None:
            self.beam_kv_pool.release(request_id)
        state.beam_kv_pool_lease = None

    def _beam_kv_pool_health_status(
        self,
        pool_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.beam_kv_pool is None:
            return {
                "beam_kv_pool_leak_detected": False,
                "beam_kv_pool_orphaned_leases": (),
                "beam_kv_pool_missing_leases": (),
                "beam_kv_pool_orphaned_lease_count": 0,
                "beam_kv_pool_missing_lease_count": 0,
            }
        if pool_status is None:
            pool_status = self.beam_kv_pool.status()
        active_pool_request_ids = {
            request_id
            for request_id, state in self.scheduler.decoding.items()
            if state.beam_kv_pool_lease is not None
        }
        lease_request_ids = set(pool_status.get("lease_request_ids", ()))
        orphaned = sorted(lease_request_ids - active_pool_request_ids)
        missing = sorted(active_pool_request_ids - lease_request_ids)
        return {
            "beam_kv_pool_leak_detected": bool(orphaned or missing),
            "beam_kv_pool_orphaned_leases": orphaned,
            "beam_kv_pool_missing_leases": missing,
            "beam_kv_pool_orphaned_lease_count": len(orphaned),
            "beam_kv_pool_missing_lease_count": len(missing),
        }


def _request_context_tokens(request: GRServingRequest) -> int:
    shape = getattr(request.input_ids, "shape", None)
    if shape is None:
        return 1
    if len(shape) == 0:
        return 1
    return int(shape[-1])


def _request_beam_slots(request: GRServingRequest) -> int:
    return int(request.max_decode_steps) * int(request.beam_width)


def _numeric_status_metrics(
    prefix: str, status: Mapping[str, Any]
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for name, value in status.items():
        metric_name = name if name.startswith(f"{prefix}_") else f"{prefix}_{name}"
        if isinstance(value, bool):
            metrics[metric_name] = int(value)
        elif isinstance(value, int | float):
            metrics[metric_name] = value
    return metrics


def _cache_status(
    cache: Mapping[Any, Any], *, hits: int, misses: int
) -> dict[str, int]:
    return {"entries": len(cache), "hits": hits, "misses": misses}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _env_int_tuple(name: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    values: list[int] = []
    for part in raw.replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{name} must contain integers, got {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{name} entries must be positive")
        values.append(value)
    return tuple(sorted(set(values)))


def _decode_cuda_graph_bucket_for(
    actual_batch_size: int,
    buckets: tuple[int, ...],
) -> int:
    if actual_batch_size <= 0:
        return actual_batch_size
    for bucket in sorted(buckets):
        if bucket >= actual_batch_size:
            return bucket
    return actual_batch_size


def _slice_tensor_batch(tensor: Any, batch_size: int) -> Any:
    if int(tensor.shape[0]) == batch_size:
        return tensor
    return tensor[:batch_size]


def _slots_are_contiguous_prefix(slots: tuple[int, ...]) -> bool:
    if not slots:
        return False
    first = slots[0]
    return slots == tuple(range(first, first + len(slots)))


def _state_pool_leases(
    states: tuple[GRContinuousRequestState, ...],
    attribute: str,
) -> tuple[Any, ...] | None:
    if not states:
        return None
    leases = tuple(getattr(state, attribute) for state in states)
    return None if any(lease is None for lease in leases) else leases


def _pool_slot_window(
    leases: tuple[Any, ...],
    *,
    batch_size: int | None = None,
    capacity: int,
    leased_slots: set[int],
) -> tuple[int, int] | None:
    slots = tuple(int(lease.slot) for lease in leases)
    if not _slots_are_contiguous_prefix(slots):
        return None
    first = slots[0]
    width = len(slots) if batch_size is None else batch_size
    if batch_size is not None and not _pool_slots_available_for_padding(
        first=first,
        actual_slots=set(slots),
        bucket_size=batch_size,
        capacity=capacity,
        leased_slots=leased_slots,
    ):
        return None
    return first, width


def _pool_slots_available_for_padding(
    *,
    first: int,
    actual_slots: set[int],
    bucket_size: int,
    capacity: int,
    leased_slots: set[int],
) -> bool:
    if first < 0 or first + bucket_size > capacity:
        return False
    bucket_slots = set(range(first, first + bucket_size))
    return not (bucket_slots - actual_slots).intersection(leased_slots)


def _clone_prefill_for_cache(prefill: PrefillResult) -> PrefillResult:
    return PrefillResult(
        logits=_clone_for_cache(prefill.logits),
        context_kv=ContextKV(
            _clone_for_cache(prefill.context_kv.key),
            _clone_for_cache(prefill.context_kv.value),
        ),
        hidden_states=(
            _clone_for_cache(prefill.hidden_states)
            if prefill.hidden_states is not None
            else None
        ),
    )


def _clone_for_cache(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if hasattr(value, "clone"):
        return value.clone()
    return value


def _prefill_cache_extra_key(request: GRServingRequest) -> Any:
    metadata = request.metadata or {}
    for key in ("prefill_cache_namespace", "cache_salt", "lora_id"):
        if key in metadata:
            return metadata[key]
    return None


def _copy_tensor(destination: Any, source: Any) -> None:
    if hasattr(destination, "copy_"):
        destination.copy_(source)
        return
    destination[...] = source


def _profile_allows_decode_cuda_graph(timing_recorder: Any | None) -> bool:
    if timing_recorder is None:
        return True
    return not bool(getattr(timing_recorder, "sync_timing", True))


def _profile_allows_prefill_cuda_graph(timing_recorder: Any | None) -> bool:
    return _profile_allows_decode_cuda_graph(timing_recorder)


def _slice_context_kv(context_kv: ContextKV, context_len: int) -> ContextKV:
    if context_len <= 0 or context_len > context_kv.context_len:
        raise ValueError("context_len must be in (0, cached context_len]")
    return ContextKV(
        context_kv.key[:, :, :context_len],
        context_kv.value[:, :, :context_len],
    )


def _copy_context_prefix(
    destination: ContextKV,
    source: ContextKV,
    *,
    prefix_len: int,
) -> None:
    if prefix_len <= 0:
        return
    _copy_tensor(destination.key[:, :, :prefix_len], source.key[:, :, :prefix_len])
    _copy_tensor(destination.value[:, :, :prefix_len], source.value[:, :, :prefix_len])


def _copy_suffix_beam_kv_to_context(
    destination: ContextKV,
    beam_kv: BeamKV,
    *,
    prefix_len: int,
    suffix_len: int,
) -> None:
    if suffix_len <= 0:
        return
    suffix_key = beam_kv.key[:, :, :suffix_len, 0]
    suffix_value = beam_kv.value[:, :, :suffix_len, 0]
    _copy_tensor(
        destination.key[:, :, prefix_len : prefix_len + suffix_len], suffix_key
    )
    _copy_tensor(
        destination.value[:, :, prefix_len : prefix_len + suffix_len],
        suffix_value,
    )


def _empty_context_like(reference: Any, *, context_len: int) -> Any:
    shape = tuple(int(dim) for dim in getattr(reference, "shape", ()))
    if len(shape) != 5:
        raise ValueError(f"context reference expects rank 5, got {shape}")
    new_shape = (shape[0], 1, context_len, shape[3], shape[4])
    if hasattr(reference, "new_empty"):
        return reference.new_empty(new_shape)
    if hasattr(reference, "with_shape"):
        return reference.with_shape(new_shape)
    raise TypeError(
        f"cannot allocate ContextKV from reference type {type(reference)!r}"
    )


def _topk_indices_cache_key(
    batched_beam_path: BatchedBeamPath,
    *,
    request_ids: tuple[str, ...],
    num_q_heads: int,
    decode_nums: int,
    beam_width: int,
    device: Any | None,
    needs_history_compaction: bool,
) -> tuple[Any, ...]:
    common = (
        num_q_heads,
        decode_nums,
        beam_width,
        _device_cache_key(device),
    )
    request_scope = ("request_scoped", tuple(request_ids))
    if needs_history_compaction:
        return ("compacted", batched_beam_path.batch_size, *common)
    if _is_identity_batched_beam_path(
        batched_beam_path,
        decode_nums=decode_nums,
        beam_width=beam_width,
    ):
        return (*request_scope, "identity", batched_beam_path.batch_size, *common)
    return (
        *request_scope,
        "path",
        _batched_beam_path_signature(batched_beam_path, decode_nums=decode_nums),
        *common,
    )


def _decode_inputs_cache_key(
    selection: BatchedBeamSelection,
    *,
    request_ids: tuple[str, ...],
    device: Any | None,
) -> tuple[Any, ...]:
    return (
        "request_scoped",
        tuple(request_ids),
        "decode_inputs",
        selection.token_ids,
        _device_cache_key(device),
    )


def _request_scoped_cache_key_matches_finished(
    cache_key: tuple[Any, ...],
    finished: set[str],
) -> bool:
    return bool(
        cache_key
        and cache_key[0] in {"request_scoped"}
        and finished.intersection(cache_key[1])
    )


def _batched_beam_path_signature(
    batched_beam_path: BatchedBeamPath,
    *,
    decode_nums: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(entry.parent_beams for entry in beam_path.entries[:decode_nums])
        for beam_path in batched_beam_path.paths
    )


def _is_identity_batched_beam_path(
    batched_beam_path: BatchedBeamPath,
    *,
    decode_nums: int,
    beam_width: int,
) -> bool:
    if decode_nums <= 1:
        return True
    if batched_beam_path.steps_done < decode_nums:
        return False
    for beam_path in batched_beam_path.paths:
        for step in range(decode_nums):
            entry = beam_path.entries[step]
            if entry.width < beam_width:
                return False
            expected = tuple(range(beam_width))
            if tuple(entry.parent_beams[:beam_width]) != expected and step > 0:
                return False
    return True


def _device_cache_key(device: Any | None) -> tuple[str, int | None]:
    if device is None:
        return ("cpu", None)
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    device_index = getattr(device, "index", None)
    if device_index is None and isinstance(device, str) and ":" in device:
        device_index = int(device.split(":", 1)[1])
    return (str(device_type), device_index)


def _torch_no_grad_context():
    try:
        import torch
    except ImportError:  # pragma: no cover - torch-free unit environments
        return nullcontext()
    return torch.no_grad()


def _policy_width_for_step(
    request: GRServingRequest,
    *,
    step: int,
    fallback: int,
) -> int:
    policy = request.beam_width_policy
    width = fallback if policy is None else int(policy.width_for_step(step))
    if width <= 0 or width > request.beam_width:
        raise ValueError(
            "beam_width_policy produced width outside request beam_width: "
            f"step={step}, width={width}, request_beam_width={request.beam_width}"
        )
    return width


def _observe_request_policy_scores(
    request: GRServingRequest,
    *,
    step: int,
    scores: tuple[float, ...],
) -> None:
    observer = getattr(request.beam_width_policy, "observe_scores", None)
    if observer is not None:
        observer(step, scores)


def _require_allocator(allocator: GRKVLeaseAllocator | None) -> GRKVLeaseAllocator:
    if allocator is None:
        raise RuntimeError("continuous scheduler KV allocator is not initialized")
    return allocator


def _chunks(values: tuple[str, ...], chunk_size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(
        values[index : index + chunk_size]
        for index in range(0, len(values), chunk_size)
    )


def _require_generation(state: GRContinuousRequestState) -> GRGenerationState:
    if state.generation is None:
        raise ValueError(f"request {state.request_id} has not run prefill")
    return state.generation


def _current_batched_selection(
    generations: tuple[GRGenerationState, ...],
) -> BatchedBeamSelection:
    token_ids = []
    scores = []
    parent_beams = []
    for generation in generations:
        if not generation.beam_path.entries:
            raise ValueError(
                f"request {generation.request_id} has no initialized beams"
            )
        entry = generation.beam_path.entries[-1]
        token_ids.append(entry.token_ids)
        scores.append(entry.scores)
        parent_beams.append(entry.parent_beams)
    return BatchedBeamSelection(
        token_ids=tuple(token_ids),
        scores=tuple(scores),
        parent_beams=tuple(parent_beams),
    )


def _initialize_decode_tensor_state(
    state: GRContinuousRequestState,
    *,
    device: Any | None,
) -> None:
    generation = _require_generation(state)
    if not generation.beam_path.entries:
        return

    import torch

    entry = generation.beam_path.entries[-1]
    state.decode_selection_token_ids = torch.tensor(
        entry.token_ids,
        dtype=torch.long,
        device=device,
    )
    state.decode_selection_scores = torch.tensor(
        entry.scores,
        dtype=torch.float32,
        device=device,
    )
    state.decode_parent_history = [
        torch.tensor(
            entry.parent_beams,
            dtype=torch.long,
            device=device,
        )
    ]
    state.pending_decode_token_ids.clear()
    state.pending_decode_scores.clear()
    state.pending_decode_parent_beams.clear()


def _stack_state_tensor_rows(
    states: tuple[GRContinuousRequestState, ...],
    attribute: str,
) -> Any:
    import torch

    rows = tuple(getattr(state, attribute) for state in states)
    if any(row is None for row in rows):
        raise ValueError(f"state tensor attribute {attribute} is missing")
    if len(rows) == 1:
        return rows[0].reshape(1, -1)
    return torch.stack(tuple(row.reshape(-1) for row in rows), dim=0)


def _make_tensor_parent_history_topk_indices(
    states: tuple[GRContinuousRequestState, ...],
    *,
    num_q_heads: int,
    decode_nums: int,
    beam_width: int,
    device: Any | None,
) -> Any:
    if num_q_heads <= 0:
        raise ValueError("num_q_heads must be positive")
    if decode_nums <= 0:
        raise ValueError("decode_nums must be positive")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if not states:
        raise ValueError("states must be non-empty")

    import torch

    if decode_nums == 1:
        pattern = torch.arange(beam_width, dtype=torch.int32, device=device).view(
            1,
            1,
            1,
            1,
            beam_width,
        )
        return pattern.expand(
            len(states),
            1,
            num_q_heads,
            1,
            beam_width,
        ).contiguous()

    parent_history = []
    for step in range(decode_nums):
        rows = []
        for state in states:
            if len(state.decode_parent_history) <= step:
                raise ValueError("decode parent history is shorter than decode_nums")
            rows.append(state.decode_parent_history[step].reshape(-1))
        parent_history.append(
            torch.stack(tuple(rows), dim=0).to(device=device, dtype=torch.long)
        )

    current = torch.arange(beam_width, dtype=torch.long, device=device).view(
        1, beam_width
    )
    current = current.expand(len(states), beam_width)
    ancestry = [current for _ in range(decode_nums)]
    for step in range(decode_nums - 1, -1, -1):
        ancestry[step] = current
        current = parent_history[step].gather(1, current)

    step_offsets = (
        torch.arange(decode_nums, dtype=torch.int32, device=device).view(
            1, decode_nums, 1
        )
        * beam_width
    )
    pattern = torch.stack(
        tuple(values.to(dtype=torch.int32) for values in ancestry),
        dim=1,
    )
    pattern = pattern + step_offsets
    return (
        pattern.view(len(states), 1, 1, decode_nums, beam_width)
        .expand(
            len(states),
            1,
            num_q_heads,
            decode_nums,
            beam_width,
        )
        .contiguous()
    )


def _update_decode_tensor_state(
    states: tuple[GRContinuousRequestState, ...],
    selection: BatchedBeamSelection,
) -> None:
    if (
        selection.token_ids_tensor is None
        or selection.scores_tensor is None
        or selection.parent_beams_tensor is None
    ):
        raise ValueError("decode tensor state update requires tensor-backed selection")
    for batch_idx, state in enumerate(states):
        token_ids = selection.token_ids_tensor[batch_idx].detach()
        scores = selection.scores_tensor[batch_idx].detach()
        parent_beams = selection.parent_beams_tensor[batch_idx].detach()
        state.decode_selection_token_ids = token_ids
        state.decode_selection_scores = scores
        state.decode_parent_history.append(parent_beams)
        state.pending_decode_token_ids.append(token_ids)
        state.pending_decode_scores.append(scores)
        state.pending_decode_parent_beams.append(parent_beams)


def _flush_pending_decode_tensor_selections_batch(
    states: tuple[GRContinuousRequestState, ...],
) -> dict[str, tuple[tuple[int, ...], tuple[float, ...]]]:
    if not states:
        return {}
    pending_steps = len(states[0].pending_decode_token_ids)
    if (
        pending_steps == 0
        or any(len(state.pending_decode_token_ids) != pending_steps for state in states)
        or any(len(state.pending_decode_scores) != pending_steps for state in states)
        or any(
            len(state.pending_decode_parent_beams) != pending_steps for state in states
        )
    ):
        return {
            state.request_id: _flush_pending_decode_tensor_selections(state)
            for state in states
        }

    import torch

    try:
        parent_rows = (
            _stack_pending_decode_field(
                torch,
                states,
                "pending_decode_parent_beams",
                pending_steps=pending_steps,
            )
            .detach()
            .cpu()
            .tolist()
        )
        token_rows = (
            _stack_pending_decode_field(
                torch,
                states,
                "pending_decode_token_ids",
                pending_steps=pending_steps,
            )
            .detach()
            .cpu()
            .tolist()
        )
        score_rows = (
            _stack_pending_decode_field(
                torch,
                states,
                "pending_decode_scores",
                pending_steps=pending_steps,
            )
            .detach()
            .cpu()
            .tolist()
        )
    except (RuntimeError, ValueError):
        return {
            state.request_id: _flush_pending_decode_tensor_selections(state)
            for state in states
        }

    finalized: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {}
    for batch_idx, state in enumerate(states):
        generation = state.generation
        if generation is None:
            finalized[state.request_id] = ((), ())
            continue
        last_token_ids = generation.beam_path.entries[-1].token_ids
        last_scores = generation.beam_path.entries[-1].scores
        for step in range(pending_steps):
            parent_beams = tuple(int(parent) for parent in parent_rows[step][batch_idx])
            token_ids = tuple(int(token) for token in token_rows[step][batch_idx])
            scores = tuple(float(score) for score in score_rows[step][batch_idx])
            generation.beam_path.append(
                parent_beams=parent_beams,
                token_ids=token_ids,
                scores=scores,
            )
            last_token_ids = token_ids
            last_scores = scores
        state.pending_decode_parent_beams.clear()
        state.pending_decode_token_ids.clear()
        state.pending_decode_scores.clear()
        finalized[state.request_id] = (last_token_ids, last_scores)
    return finalized


def _stack_pending_decode_field(
    torch,
    states: tuple[GRContinuousRequestState, ...],
    attribute: str,
    *,
    pending_steps: int,
) -> Any:
    step_rows = []
    for step in range(pending_steps):
        rows = [getattr(state, attribute)[step].reshape(-1) for state in states]
        step_rows.append(torch.stack(tuple(rows), dim=0))
    return torch.stack(tuple(step_rows), dim=0)


def _flush_pending_decode_tensor_selections(
    state: GRContinuousRequestState,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    generation = state.generation
    if generation is None:
        return (), ()
    last_token_ids = generation.beam_path.entries[-1].token_ids
    last_scores = generation.beam_path.entries[-1].scores
    pending = zip(
        state.pending_decode_parent_beams,
        state.pending_decode_token_ids,
        state.pending_decode_scores,
    )
    for parent_beams_tensor, token_ids_tensor, scores_tensor in pending:
        parent_beams = tuple(
            int(parent) for parent in parent_beams_tensor.detach().cpu().tolist()
        )
        token_ids = tuple(
            int(token) for token in token_ids_tensor.detach().cpu().tolist()
        )
        scores = tuple(float(score) for score in scores_tensor.detach().cpu().tolist())
        generation.beam_path.append(
            parent_beams=parent_beams,
            token_ids=token_ids,
            scores=scores,
        )
        last_token_ids = token_ids
        last_scores = scores
    state.pending_decode_parent_beams.clear()
    state.pending_decode_token_ids.clear()
    state.pending_decode_scores.clear()
    return last_token_ids, last_scores


def _make_batched_generation(
    *,
    request_id: str,
    generations: tuple[GRGenerationState, ...],
    beam_score_mode: str,
    prefill: PrefillResult | None = None,
    beam_kv: BeamKV | None = None,
    batched_beam_path: BatchedBeamPath | None = None,
    decode_nums: int | None = None,
    active_beam_width: int | None = None,
    needs_history_compaction: bool | None = None,
) -> GRGenerationState:
    if not generations:
        raise ValueError("generations must be non-empty")
    first = generations[0]
    if prefill is None:
        prefill = _make_batched_prefill(generations)
    if beam_kv is None:
        beam_kv = _make_batched_beam_kv(generations)
    if needs_history_compaction is None:
        needs_history_compaction = (
            batched_beam_path is not None
            and decode_nums is not None
            and active_beam_width is not None
            and needs_batched_beam_kv_history_compaction(
                batched_beam_path,
                decode_nums=decode_nums,
                active_beam_width=active_beam_width,
            )
        )
    if needs_history_compaction:
        if (
            batched_beam_path is None
            or decode_nums is None
            or active_beam_width is None
        ):
            raise ValueError("BeamKV compaction requires beam path and decode shape")
        beam_kv = compact_batched_beam_kv_history(
            beam_kv,
            batched_beam_path,
            decode_nums=decode_nums,
            active_beam_width=active_beam_width,
        )
    return GRGenerationState(
        request_id=request_id,
        prefill=prefill,
        beam_kv=beam_kv,
        beam_path=first.beam_path,
        fixed_beam_width=first.fixed_beam_width,
        beam_score_mode=beam_score_mode,
    )


def _make_batched_prefill(
    generations: tuple[GRGenerationState, ...],
) -> PrefillResult:
    context_kv = _make_batched_context_kv(generations)
    return PrefillResult(
        logits=_cat_tensors(
            tuple(generation.prefill.logits for generation in generations), dim=0
        ),
        context_kv=context_kv,
        hidden_states=None,
    )


def _make_batched_context_kv(generations: tuple[GRGenerationState, ...]) -> ContextKV:
    if not generations:
        raise ValueError("generations must be non-empty")
    contiguous = _contiguous_pool_view(
        tuple(generation.prefill.context_kv.key for generation in generations)
    )
    if contiguous is not None:
        key = contiguous
        value = _contiguous_pool_view(
            tuple(generation.prefill.context_kv.value for generation in generations)
        )
        if value is not None:
            return ContextKV(key, value)
    return ContextKV(
        _cat_tensors(
            tuple(generation.prefill.context_kv.key for generation in generations),
            dim=1,
        ),
        _cat_tensors(
            tuple(generation.prefill.context_kv.value for generation in generations),
            dim=1,
        ),
    )


def _make_batched_beam_kv(generations: tuple[GRGenerationState, ...]) -> BeamKV:
    if not generations:
        raise ValueError("generations must be non-empty")
    contiguous = _contiguous_pool_view(
        tuple(generation.beam_kv.key for generation in generations)
    )
    if contiguous is not None:
        key = contiguous
        value = _contiguous_pool_view(
            tuple(generation.beam_kv.value for generation in generations)
        )
        if value is not None:
            return BeamKV(key, value)
    return BeamKV(
        _cat_tensors(
            tuple(generation.beam_kv.key for generation in generations), dim=1
        ),
        _cat_tensors(
            tuple(generation.beam_kv.value for generation in generations), dim=1
        ),
    )


def _contiguous_pool_view(tensors: tuple[Any, ...]) -> Any | None:
    if not tensors or len(tensors) == 1:
        return tensors[0] if tensors else None
    first = tensors[0]
    if not all(
        hasattr(tensor, "storage_offset") and hasattr(tensor, "stride")
        for tensor in tensors
    ):
        return None
    storage_ptr = _storage_data_ptr(first)
    if storage_ptr is None or any(
        _storage_data_ptr(tensor) != storage_ptr for tensor in tensors
    ):
        return None
    try:
        starts = [int(tensor.storage_offset()) for tensor in tensors]
        shapes = [tuple(tensor.shape) for tensor in tensors]
        if len({shape for shape in shapes}) != 1:
            return None
        shape = shapes[0]
        strides = [tuple(tensor.stride()) for tensor in tensors]
        if len({stride for stride in strides}) != 1:
            return None
        stride = strides[0]
        slot_stride = int(first.stride(1))
        if any(shape[1] != 1 for shape in shapes):
            return None
        order = sorted(range(len(starts)), key=lambda idx: starts[idx])
        sorted_starts = [starts[idx] for idx in order]
        if order != list(range(len(starts))):
            return None
        for left, right in zip(sorted_starts, sorted_starts[1:]):
            if right - left != slot_stride:
                return None
        base = _root_base_tensor(first) or first
        return base.as_strided(
            (shape[0], len(tensors), *shape[2:]),
            stride,
            storage_offset=sorted_starts[0],
        )
    except Exception:
        return None


def _storage_data_ptr(tensor: Any) -> int | None:
    storage = getattr(tensor, "untyped_storage", None)
    if callable(storage):
        return int(storage().data_ptr())
    storage = getattr(tensor, "storage", None)
    if callable(storage):
        return int(storage().data_ptr())
    return None


def _tensor_view_signature(tensor: Any) -> tuple[Any, ...] | None:
    data_ptr = _storage_data_ptr(tensor)
    storage_offset = getattr(tensor, "storage_offset", None)
    stride = getattr(tensor, "stride", None)
    if data_ptr is None or not callable(storage_offset) or not callable(stride):
        return None
    return (
        data_ptr,
        int(storage_offset()),
        tuple(int(dim) for dim in getattr(tensor, "shape", ())),
        tuple(int(value) for value in stride()),
    )


def _same_tensor_view(left: Any, right: Any) -> bool:
    left_signature = _tensor_view_signature(left)
    return left_signature is not None and left_signature == _tensor_view_signature(
        right
    )


def _root_base_tensor(tensor: Any) -> Any | None:
    base = getattr(tensor, "_base", None)
    if base is None:
        return None
    while getattr(base, "_base", None) is not None:
        base = base._base
    return base


def _cat_tensors(tensors: tuple[Any, ...], *, dim: int) -> Any:
    if not tensors:
        raise ValueError("tensors must be non-empty")
    if len(tensors) == 1:
        return tensors[0]
    import torch

    return torch.cat(tensors, dim=dim)


def _scatter_batched_beam_kv(
    batched_beam_kv: BeamKV,
    generations: tuple[GRGenerationState, ...],
    *,
    step: int,
    active_beam_width: int,
) -> None:
    for batch_idx, generation in enumerate(generations):
        key_destination = generation.beam_kv.key[:, 0:1, step, :active_beam_width]
        key_source = batched_beam_kv.key[
            :,
            batch_idx : batch_idx + 1,
            step,
            :active_beam_width,
        ]
        if not _same_tensor_view(key_destination, key_source):
            _copy_tensor(key_destination, key_source)

        value_destination = generation.beam_kv.value[:, 0:1, step, :active_beam_width]
        value_source = batched_beam_kv.value[
            :,
            batch_idx : batch_idx + 1,
            step,
            :active_beam_width,
        ]
        if not _same_tensor_view(value_destination, value_source):
            _copy_tensor(value_destination, value_source)


def _continuous_step_item_mask(
    states: tuple[GRContinuousRequestState, ...],
    logits: Any,
) -> Any | None:
    if not any(state.request.item_mask_provider is not None for state in states):
        return None

    import torch

    _batch_size, beam_width, vocab_size = logits.shape
    masks = []
    for batch_idx, state in enumerate(states):
        provider = state.request.item_mask_provider
        if provider is None:
            masks.append(
                torch.ones(
                    (beam_width, vocab_size), dtype=torch.bool, device=logits.device
                )
            )
            continue
        mask = provider.step_mask(
            _require_generation(state), logits[batch_idx : batch_idx + 1]
        )
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask[0]
        masks.append(mask.bool())
    return torch.stack(masks, dim=0)


def _continuous_apply_logits_processors(
    states: tuple[GRContinuousRequestState, ...],
    logits: Any,
    *,
    step: int,
    beam_width: int,
    batched_beam_path: BatchedBeamPath,
) -> Any:
    if not any(state.request.logits_processors for state in states):
        return logits

    import torch

    rows = []
    for batch_idx, state in enumerate(states):
        rows.append(
            apply_logits_processors(
                logits[batch_idx : batch_idx + 1],
                tuple(state.request.logits_processors),
                LogitsProcessorContext(
                    request_id=state.request_id,
                    phase="decode",
                    step=step,
                    beam_width=beam_width,
                    beam_path=batched_beam_path.paths[batch_idx],
                    metadata=state.request.metadata,
                ),
            )
        )
    return torch.cat(rows, dim=0)


def _continuous_stop_reason(
    state: GRContinuousRequestState,
    token_ids: tuple[int, ...],
) -> str | None:
    stop_token_ids = _request_stop_token_ids(state.request)
    if stop_token_ids:
        stop_tokens = set(stop_token_ids)
        if token_ids and all(token in stop_tokens for token in token_ids):
            return "stop_token"
    if state.current_decode_step >= state.request.max_decode_steps:
        return "max_decode_steps"
    return None


def _request_timeout_ticks(request: GRServingRequest) -> int | None:
    value = request.metadata.get("timeout_ticks")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("timeout_ticks must be an integer")
    timeout_ticks = int(value)
    if timeout_ticks < 0:
        raise ValueError("timeout_ticks must be non-negative")
    return timeout_ticks
