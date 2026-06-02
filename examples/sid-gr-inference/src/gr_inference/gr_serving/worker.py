# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background worker for driving continuous serving ticks."""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any, Callable, Mapping

from gr_inference.gr_serving.api import GRInProcessServingFacade
from gr_inference.gr_serving.request import GRServingRequest, GRServingResponse


@dataclass
class GRServingWorker:
    """Threaded facade wrapper that continuously advances scheduler ticks."""

    facade: GRInProcessServingFacade
    tick_interval_s: float = 0.001
    idle_sleep_s: float = 0.005
    error_sleep_s: float = 0.05
    autostart: bool = False
    decode_log_interval: int = 0
    log_sink: Callable[[Mapping[str, Any]], None] | None = None
    _lock: RLock = field(default_factory=RLock, init=False)
    _pending_lock: RLock = field(default_factory=RLock, init=False)
    _pending_submissions: deque[GRServingRequest] = field(
        default_factory=deque, init=False
    )
    _stop_event: Event = field(default_factory=Event, init=False)
    _thread: Thread | None = field(default=None, init=False)
    _started_at_s: float | None = field(default=None, init=False)
    _last_error: str | None = field(default=None, init=False)
    _worker_ticks: int = field(default=0, init=False)
    _worker_errors: int = field(default=0, init=False)
    _worker_submitted_requests: int = field(default=0, init=False)
    _worker_batch_fill_waits: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.tick_interval_s < 0:
            raise ValueError("tick_interval_s must be non-negative")
        if self.idle_sleep_s < 0:
            raise ValueError("idle_sleep_s must be non-negative")
        if self.error_sleep_s < 0:
            raise ValueError("error_sleep_s must be non-negative")
        if self.decode_log_interval < 0:
            raise ValueError("decode_log_interval must be non-negative")
        if self.autostart:
            self.start()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._started_at_s = time.time()
            self._thread = Thread(
                target=self._run,
                name="gr-serving-worker",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float | None = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def submit(self, request: GRServingRequest) -> str:
        request.validate()
        with self._pending_lock:
            self._pending_submissions.append(request)
            self._worker_submitted_requests += 1
        return request.request_id

    def submit_many(self, requests: tuple[GRServingRequest, ...]) -> tuple[str, ...]:
        request_ids = tuple(request.request_id for request in requests)
        for request in requests:
            request.validate()
        with self._pending_lock:
            self._pending_submissions.extend(requests)
            self._worker_submitted_requests += len(requests)
        return request_ids

    def tick(self) -> Any:
        with self._lock:
            return self._tick_unlocked()

    def run_until_idle(
        self,
        *,
        max_ticks: int | None = None,
        timeout_unfinished: bool = False,
    ) -> tuple[GRServingResponse, ...]:
        with self._lock:
            self._drain_pending_submissions_unlocked()
            return self.facade.run_until_idle(
                max_ticks=max_ticks,
                timeout_unfinished=timeout_unfinished,
            )

    def poll(self, request_id: str) -> GRServingResponse | None:
        with self._lock:
            return self.facade.poll(request_id)

    def require_result(self, request_id: str) -> GRServingResponse:
        with self._lock:
            return self.facade.require_result(request_id)

    def request_status(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            return self.facade.request_status(request_id)

    def request_statuses(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return self.facade.request_statuses()

    def cancel(
        self, request_id: str, *, reason: str = "cancelled"
    ) -> GRServingResponse:
        with self._lock:
            return self.facade.cancel(request_id, reason=reason)

    def drain(self) -> dict[str, Any]:
        with self._lock:
            return self.facade.drain()

    def shutdown(
        self,
        *,
        max_ticks: int | None = None,
        timeout_unfinished: bool = False,
        stop_timeout_s: float | None = 2.0,
    ) -> dict[str, Any]:
        with self._lock:
            result = self.facade.shutdown(
                max_ticks=max_ticks,
                timeout_unfinished=timeout_unfinished,
            )
        self.stop(timeout=stop_timeout_s)
        result["worker"] = self.worker_status()
        return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self.facade.status())
            pending = self._pending_count()
            status["pending_submissions"] = pending
            status["waiting_prefill"] = (
                int(status.get("waiting_prefill", 0) or 0) + pending
            )
            status["worker"] = self.worker_status()
            return status

    def metrics(self) -> dict[str, float | int]:
        with self._lock:
            metrics = dict(self.facade.metrics())
            metrics.update(
                {
                    "worker_running": int(self.running),
                    "worker_ticks": self._worker_ticks,
                    "worker_errors": self._worker_errors,
                    "worker_pending_submissions": self._pending_count(),
                    "worker_submitted_requests": self._worker_submitted_requests,
                    "worker_batch_fill_waits": self._worker_batch_fill_waits,
                }
            )
            return metrics

    def catalog_status(self) -> dict[str, Any] | None:
        with self._lock:
            return self.facade.catalog_status()

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
        with self._lock:
            return self.facade.reload_item_catalog_jsonl(
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

    def worker_status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "ticks": self._worker_ticks,
            "errors": self._worker_errors,
            "last_error": self._last_error,
            "started_at_s": self._started_at_s,
            "pending_submissions": self._pending_count(),
            "submitted_requests": self._worker_submitted_requests,
            "batch_fill_waits": self._worker_batch_fill_waits,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._should_wait_for_batch_fill():
                    self._worker_batch_fill_waits += 1
                    self._stop_event.wait(self.tick_interval_s)
                with self._lock:
                    has_work = self._has_work_unlocked()
                    if has_work:
                        self._tick_unlocked()
                self._stop_event.wait(
                    self.tick_interval_s if has_work else self.idle_sleep_s
                )
            except Exception as exc:  # pragma: no cover - defensive worker guard
                self._worker_errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._stop_event.wait(self.error_sleep_s)

    def _tick_unlocked(self) -> Any:
        self._drain_pending_submissions_unlocked()
        result = self.facade.tick()
        self._worker_ticks += 1
        self._last_error = None
        self._emit_decode_interval_log(result)
        return result

    def _emit_decode_interval_log(self, result: Any) -> None:
        if self.decode_log_interval <= 0:
            return
        if self._worker_ticks % self.decode_log_interval != 0:
            return
        metadata = result.metadata() if hasattr(result, "metadata") else {}
        payload = {
            "event": "gr_worker_decode_tick",
            "worker_ticks": self._worker_ticks,
            "tick": metadata.get("tick"),
            "prefill_batch_size": len(metadata.get("prefill_request_ids", ())),
            "decode_batches": len(metadata.get("decode_batches", ())),
            "finished_requests": len(metadata.get("finished_request_ids", ())),
        }
        _emit_worker_log(payload, self.log_sink)

    def _has_work_unlocked(self) -> bool:
        status = self.facade.status()
        return bool(self._pending_count() or _status_has_work(status))

    def _pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending_submissions)

    def _drain_pending_submissions_unlocked(self) -> None:
        with self._pending_lock:
            if not self._pending_submissions:
                return
            requests = tuple(self._pending_submissions)
            self._pending_submissions.clear()
        self.facade.submit_many(requests)

    def _should_wait_for_batch_fill(self) -> bool:
        pending = self._pending_count()
        if pending <= 0:
            return False
        target = self._prefill_batch_target()
        if pending >= target:
            return False
        with self._lock:
            status = self.facade.status()
        return not _status_has_work(status)

    def _prefill_batch_target(self) -> int:
        with self._lock:
            status = self.facade.status()
        policy = status.get("policy")
        if isinstance(policy, Mapping):
            return max(1, int(policy.get("max_prefill_batch_size") or 1))
        return 1


def _emit_worker_log(
    payload: Mapping[str, Any], sink: Callable[[Mapping[str, Any]], None] | None
) -> None:
    if sink is not None:
        sink(payload)
        return
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)


def _status_has_work(status: Mapping[str, Any]) -> bool:
    return bool(
        status.get("waiting_prefill", 0)
        or status.get("decoding", 0)
        or status.get("waiting", 0)
        or status.get("running", 0)
    )
