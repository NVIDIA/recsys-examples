"""Serving batch assembly policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gr_inference.gr_serving.request import GRServingRequest


@dataclass(frozen=True)
class GRRequestBatch:
    """A batch of requests selected by the scheduler."""

    requests: tuple[GRServingRequest, ...]

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def max_decode_steps(self) -> int | None:
        if not self.requests:
            return None
        return self.requests[0].max_decode_steps

    @property
    def beam_width(self) -> int | None:
        if not self.requests:
            return None
        return self.requests[0].beam_width

    def input_shapes(self) -> tuple[Any, ...]:
        return tuple(getattr(request.input_ids, "shape", None) for request in self.requests)

    def compatible_for_tensor_batch(self) -> bool:
        if not self.requests:
            return False
        first = self.requests[0]
        first_shape = getattr(first.input_ids, "shape", None)
        return all(
            request.max_decode_steps == first.max_decode_steps
            and request.beam_width == first.beam_width
            and request.beam_width_policy is None
            and getattr(request.input_ids, "shape", None) == first_shape
            for request in self.requests
        ) and first.beam_width_policy is None

    def metadata(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_decode_steps": self.max_decode_steps,
            "beam_width": self.beam_width,
            "input_shapes": [str(shape) for shape in self.input_shapes()],
            "tensor_batch_compatible": self.compatible_for_tensor_batch(),
        }


@dataclass(frozen=True)
class SchedulerPolicy:
    """MVP scheduler policy for FIFO batch assembly."""

    max_batch_size: int = 1
    max_wait_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")


class FIFOBatchAssembler:
    """Assemble batches from a FIFO queue."""

    def __init__(self, policy: SchedulerPolicy) -> None:
        self.policy = policy

    def assemble(self, queue) -> GRRequestBatch | None:
        requests: list[GRServingRequest] = []
        for _ in range(self.policy.max_batch_size):
            request = queue.pop()
            if request is None:
                break
            requests.append(request)
        if not requests:
            return None
        return GRRequestBatch(tuple(requests))

