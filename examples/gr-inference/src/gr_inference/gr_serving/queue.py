"""Synchronous serving queue skeleton."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any

from gr_inference.gr_serving.batch import FIFOBatchAssembler, SchedulerPolicy
from gr_inference.gr_serving.engine import GRServingEngine
from gr_inference.gr_serving.request import GRServingRequest, GRServingResponse


@dataclass
class GRRequestQueue:
    """FIFO queue for serving requests."""

    _items: deque[GRServingRequest] = field(default_factory=deque)

    def submit(self, request: GRServingRequest) -> None:
        request.validate()
        self._items.append(request)

    def pop(self) -> GRServingRequest | None:
        if not self._items:
            return None
        return self._items.popleft()

    def push_front(self, request: GRServingRequest) -> None:
        request.validate()
        self._items.appendleft(request)

    def remove(self, request_id: str) -> GRServingRequest | None:
        for index, request in enumerate(self._items):
            if request.request_id == request_id:
                del self._items[index]
                return request
        return None

    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class SyncGRScheduler:
    """Run queued requests synchronously through one serving engine."""

    engine: GRServingEngine
    policy: SchedulerPolicy = field(default_factory=SchedulerPolicy)
    waiting: GRRequestQueue = field(default_factory=GRRequestQueue)
    running: dict[str, GRServingRequest] = field(default_factory=dict)
    finished: dict[str, GRServingResponse] = field(default_factory=dict)
    submitted_requests: int = 0
    processed_requests: int = 0
    total_scheduler_ms: float = 0.0
    assembled_batches: int = 0
    batch_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def queue(self) -> GRRequestQueue:
        """Backward-compatible alias for the waiting queue."""

        return self.waiting

    def submit(self, request: GRServingRequest) -> None:
        self.waiting.submit(request)
        self.submitted_requests += 1

    def run_until_empty(self) -> tuple[GRServingResponse, ...]:
        start = time.perf_counter()
        responses: list[GRServingResponse] = []
        assembler = FIFOBatchAssembler(self.policy)
        while True:
            batch = assembler.assemble(self.waiting)
            if batch is None:
                break
            self.assembled_batches += 1
            self.batch_history.append(batch.metadata())
            for request in batch.requests:
                self.running[request.request_id] = request
            try:
                batch_responses = self.engine.generate_batch(batch)
                responses.extend(batch_responses)
                for response in batch_responses:
                    self.finished[response.request_id] = response
                    self.processed_requests += 1
            finally:
                for request in batch.requests:
                    self.running.pop(request.request_id, None)
        self.total_scheduler_ms += (time.perf_counter() - start) * 1000
        return tuple(responses)

    def status(self) -> dict[str, Any]:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished": len(self.finished),
            "submitted_requests": self.submitted_requests,
            "processed_requests": self.processed_requests,
            "assembled_batches": self.assembled_batches,
            "max_batch_size": self.policy.max_batch_size,
            "avg_batch_size": self._avg_batch_size(),
        }

    def metrics(self) -> dict[str, float | int]:
        return {
            "queue_size": len(self.waiting),
            "waiting_requests": len(self.waiting),
            "running_requests": len(self.running),
            "finished_requests": len(self.finished),
            "submitted_requests": self.submitted_requests,
            "processed_requests": self.processed_requests,
            "assembled_batches": self.assembled_batches,
            "max_batch_size": self.policy.max_batch_size,
            "avg_batch_size": self._avg_batch_size(),
            "total_scheduler_ms": self.total_scheduler_ms,
        }

    def _avg_batch_size(self) -> float:
        if not self.batch_history:
            return 0.0
        return sum(batch["size"] for batch in self.batch_history) / len(self.batch_history)

