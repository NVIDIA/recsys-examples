# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process serving control facade.

This layer is intentionally transport-free. HTTP/gRPC adapters can wrap it
without learning scheduler internals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from gr_inference.gr_serving.request import GRServingRequest, GRServingResponse


@dataclass
class GRInProcessServingFacade:
    """Small API surface for driving a continuous serving executor."""

    executor: Any
    item_mask_provider_store: Any | None = None
    accepting_requests: bool = True

    def submit(self, request: GRServingRequest) -> str:
        if not self.accepting_requests:
            raise RuntimeError("serving facade is draining; rejecting new request")
        prepared = self._prepare_request(request)
        prepared.validate()
        self._ensure_request_ids_available((prepared.request_id,))
        self.executor.submit(prepared)
        return prepared.request_id

    def submit_many(self, requests: tuple[GRServingRequest, ...]) -> tuple[str, ...]:
        if not self.accepting_requests:
            raise RuntimeError("serving facade is draining; rejecting new requests")
        prepared = tuple(self._prepare_request(request) for request in requests)
        for request in prepared:
            request.validate()
        request_ids = tuple(request.request_id for request in prepared)
        _ensure_unique_request_ids(request_ids)
        self._ensure_request_ids_available(request_ids)
        for request in prepared:
            self.executor.submit(request)
        return request_ids

    def tick(self) -> Any:
        return self.executor.tick()

    def run_until_idle(
        self,
        *,
        max_ticks: int | None = None,
        timeout_unfinished: bool = False,
    ) -> tuple[GRServingResponse, ...]:
        return self.executor.run_until_empty(
            max_ticks=max_ticks,
            timeout_unfinished=timeout_unfinished,
        )

    def poll(self, request_id: str) -> GRServingResponse | None:
        return self._finished().get(request_id)

    def require_result(self, request_id: str) -> GRServingResponse:
        response = self.poll(request_id)
        if response is None:
            raise KeyError(f"request {request_id} has not finished")
        return response

    def request_status(self, request_id: str) -> dict[str, Any]:
        scheduler = self._scheduler()
        state = scheduler.states.get(request_id)
        if state is None:
            raise KeyError(f"unknown request_id: {request_id}")
        return _request_status_payload(
            state,
            result_available=request_id in scheduler.finished,
            current_tick=scheduler.tick_count,
        )

    def request_statuses(self) -> tuple[dict[str, Any], ...]:
        scheduler = self._scheduler()
        return tuple(
            _request_status_payload(
                state,
                result_available=request_id in scheduler.finished,
                current_tick=scheduler.tick_count,
            )
            for request_id, state in scheduler.states.items()
        )

    def cancel(
        self, request_id: str, *, reason: str = "cancelled"
    ) -> GRServingResponse:
        return self.executor.cancel(request_id, reason=reason)

    def drain(self) -> dict[str, Any]:
        self.accepting_requests = False
        return self.lifecycle_status()

    def shutdown(
        self,
        *,
        max_ticks: int | None = None,
        timeout_unfinished: bool = False,
    ) -> dict[str, Any]:
        self.drain()
        responses = self.run_until_idle(
            max_ticks=max_ticks,
            timeout_unfinished=timeout_unfinished,
        )
        return {
            "lifecycle": self.lifecycle_status(),
            "responses": responses,
            "status": self.status(),
        }

    def status(self) -> dict[str, Any]:
        status = dict(self.executor.status())
        status["lifecycle"] = self._lifecycle_status_from_status(status)
        if self.item_mask_provider_store is not None:
            status["item_constraints"] = self.item_mask_provider_store.status()
        return status

    def metrics(self) -> dict[str, float | int]:
        return self.executor.metrics()

    def catalog_status(self) -> dict[str, Any] | None:
        if self.item_mask_provider_store is None:
            return None
        return self.item_mask_provider_store.status()

    def reload_item_catalog_jsonl(
        self,
        path: str | Path,
        *,
        vocab_size: int,
        eos_token_id: int | None = None,
        allow_eos_for_terminal: bool = True,
        item_id_field: str = "item_id",
        token_ids_field: str = "token_ids",
        metadata_field: str | None = "metadata",
        allow_duplicate_item_ids: bool = False,
        allow_duplicate_token_paths: bool = False,
    ) -> int:
        if self.item_mask_provider_store is None:
            raise RuntimeError("item_mask_provider_store is not configured")
        return self.item_mask_provider_store.reload_jsonl(
            path,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            allow_eos_for_terminal=allow_eos_for_terminal,
            item_id_field=item_id_field,
            token_ids_field=token_ids_field,
            metadata_field=metadata_field,
            allow_duplicate_item_ids=allow_duplicate_item_ids,
            allow_duplicate_token_paths=allow_duplicate_token_paths,
        )

    def rollback_item_catalog(self) -> int:
        if self.item_mask_provider_store is None:
            raise RuntimeError("item_mask_provider_store is not configured")
        rollback = getattr(self.item_mask_provider_store, "rollback", None)
        if rollback is None:
            raise RuntimeError("item_mask_provider_store does not support rollback")
        return int(rollback())

    def lifecycle_status(self) -> dict[str, Any]:
        return self._lifecycle_status_from_status(dict(self.executor.status()))

    def _lifecycle_status_from_status(self, status: dict[str, Any]) -> dict[str, Any]:
        in_flight = _status_work_count(status)
        return {
            "accepting_requests": self.accepting_requests,
            "draining": not self.accepting_requests,
            "idle": in_flight == 0,
            "in_flight_requests": in_flight,
        }

    def _finished(self) -> dict[str, GRServingResponse]:
        return self._scheduler().finished

    def _scheduler(self) -> Any:
        return getattr(self.executor, "scheduler", self.executor)

    def _prepare_request(self, request: GRServingRequest) -> GRServingRequest:
        if (
            request.item_mask_provider is not None
            or self.item_mask_provider_store is None
        ):
            return request
        return replace(
            request,
            item_mask_provider=self.item_mask_provider_store.snapshot(),
        )

    def _ensure_request_ids_available(self, request_ids: tuple[str, ...]) -> None:
        states = getattr(self._scheduler(), "states", {})
        conflicts = [request_id for request_id in request_ids if request_id in states]
        if conflicts:
            raise RuntimeError(f"duplicate request_id: {conflicts[0]}")


def _request_status_payload(
    state: Any,
    *,
    result_available: bool,
    current_tick: int,
) -> dict[str, Any]:
    request = state.request
    timeout_ticks = request.metadata.get("timeout_ticks")
    finished_tick = state.finished_tick
    elapsed_until_tick = finished_tick if finished_tick is not None else current_tick
    return {
        "request_id": state.request_id,
        "stage": state.stage,
        "ready": state.stage == "finished",
        "result_available": result_available,
        "current_decode_step": state.current_decode_step,
        "remaining_decode_steps": state.remaining_decode_steps,
        "max_decode_steps": request.max_decode_steps,
        "beam_width": request.beam_width,
        "active_beam_width": state.beam_width,
        "submitted_tick": state.submitted_tick,
        "admitted_tick": state.admitted_tick,
        "finished_tick": state.finished_tick,
        "age_ticks": max(0, current_tick - state.submitted_tick),
        "elapsed_ticks": max(0, elapsed_until_tick - state.submitted_tick),
        "timeout_ticks": timeout_ticks,
        "deadline_tick": (
            state.submitted_tick + int(timeout_ticks)
            if timeout_ticks is not None
            else None
        ),
        "stop_reason": state.stop_reason,
        "has_item_constraints": request.item_mask_provider is not None,
        "beam_width_policy": _beam_width_policy_payload(request.beam_width_policy),
    }


def _status_work_count(status: dict[str, Any]) -> int:
    total = 0
    for key in ("waiting_prefill", "decoding", "waiting", "running"):
        value = status.get(key, 0)
        if isinstance(value, int):
            total += value
    return total


def _ensure_unique_request_ids(request_ids: tuple[str, ...]) -> None:
    seen: set[str] = set()
    for request_id in request_ids:
        if request_id in seen:
            raise RuntimeError(f"duplicate request_id in batch: {request_id}")
        seen.add(request_id)


def _beam_width_policy_payload(policy: Any | None) -> dict[str, Any] | None:
    if policy is None:
        return None
    return {
        "type": type(policy).__name__,
        "widths": dict(getattr(policy, "widths", {}) or {}),
        "schedule": dict(getattr(policy, "schedule", {}) or {}),
    }
