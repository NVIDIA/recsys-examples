"""Serving-level KV lease accounting.

The first allocator is deliberately metadata-only. It tracks request ownership
and budgets before a paged/block GPU allocator replaces the backing storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gr_inference.gr_kv import BeamKV, ContextKV


@dataclass(frozen=True)
class GRKVMemoryEstimate:
    """GR-specific KV/workspace memory estimate in bytes."""

    batch_size: int
    num_layers: int
    context_len: int
    max_decode_steps: int
    max_beam_width: int
    num_kv_heads: int
    head_dim: int
    bytes_per_element: int = 2
    vocab_size: int | None = None
    active_beam_width: int | None = None
    context_kv_bytes: int = 0
    beam_kv_bytes: int = 0
    dense_per_beam_context_kv_bytes: int = 0
    logits_workspace_bytes: int = 0
    topk_scores_workspace_bytes: int = 0

    @property
    def total_kv_bytes(self) -> int:
        return self.context_kv_bytes + self.beam_kv_bytes

    @property
    def total_with_workspace_bytes(self) -> int:
        return (
            self.total_kv_bytes
            + self.logits_workspace_bytes
            + self.topk_scores_workspace_bytes
        )

    @property
    def context_kv_sharing_savings_bytes(self) -> int:
        return max(0, self.dense_per_beam_context_kv_bytes - self.context_kv_bytes)

    def metadata(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_layers": self.num_layers,
            "context_len": self.context_len,
            "max_decode_steps": self.max_decode_steps,
            "max_beam_width": self.max_beam_width,
            "active_beam_width": self.active_beam_width,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "bytes_per_element": self.bytes_per_element,
            "vocab_size": self.vocab_size,
            "context_kv_bytes": self.context_kv_bytes,
            "beam_kv_bytes": self.beam_kv_bytes,
            "total_kv_bytes": self.total_kv_bytes,
            "logits_workspace_bytes": self.logits_workspace_bytes,
            "topk_scores_workspace_bytes": self.topk_scores_workspace_bytes,
            "total_with_workspace_bytes": self.total_with_workspace_bytes,
            "dense_per_beam_context_kv_bytes": self.dense_per_beam_context_kv_bytes,
            "context_kv_sharing_savings_bytes": self.context_kv_sharing_savings_bytes,
            "context_kv_sharing_savings_ratio": _ratio(
                self.context_kv_sharing_savings_bytes,
                self.dense_per_beam_context_kv_bytes,
            ),
        }


def estimate_gr_kv_memory(
    *,
    batch_size: int,
    num_layers: int,
    context_len: int,
    max_decode_steps: int,
    max_beam_width: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int = 2,
    vocab_size: int | None = None,
    active_beam_width: int | None = None,
) -> GRKVMemoryEstimate:
    """Estimate shared ContextKV + short BeamKV footprint for GR serving."""

    for name, value in (
        ("batch_size", batch_size),
        ("num_layers", num_layers),
        ("context_len", context_len),
        ("max_decode_steps", max_decode_steps),
        ("max_beam_width", max_beam_width),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("bytes_per_element", bytes_per_element),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if vocab_size is not None and int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive when provided")
    if active_beam_width is None:
        active_beam_width = max_beam_width
    if active_beam_width <= 0 or active_beam_width > max_beam_width:
        raise ValueError("active_beam_width must be in (0, max_beam_width]")

    kv_unit = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
    context_kv_bytes = batch_size * context_len * kv_unit
    beam_kv_bytes = batch_size * max_decode_steps * max_beam_width * kv_unit
    dense_per_beam_context_kv_bytes = (
        batch_size * context_len * max_beam_width * kv_unit
    )
    logits_workspace_bytes = 0
    topk_scores_workspace_bytes = 0
    if vocab_size is not None:
        # Logits/topK are FP32-like ranking workspaces regardless of KV dtype.
        logits_workspace_bytes = batch_size * active_beam_width * vocab_size * 4
        topk_scores_workspace_bytes = batch_size * active_beam_width * 4

    return GRKVMemoryEstimate(
        batch_size=batch_size,
        num_layers=num_layers,
        context_len=context_len,
        max_decode_steps=max_decode_steps,
        max_beam_width=max_beam_width,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_element=bytes_per_element,
        vocab_size=vocab_size,
        active_beam_width=active_beam_width,
        context_kv_bytes=context_kv_bytes,
        beam_kv_bytes=beam_kv_bytes,
        dense_per_beam_context_kv_bytes=dense_per_beam_context_kv_bytes,
        logits_workspace_bytes=logits_workspace_bytes,
        topk_scores_workspace_bytes=topk_scores_workspace_bytes,
    )


def estimate_gr_kv_memory_from_model_config(
    config: Any,
    *,
    batch_size: int,
    context_len: int,
    max_decode_steps: int,
    max_beam_width: int,
    bytes_per_element: int = 2,
    vocab_size: int | None = None,
    active_beam_width: int | None = None,
) -> GRKVMemoryEstimate:
    """Estimate memory from a Qwen-like config object."""

    return estimate_gr_kv_memory(
        batch_size=batch_size,
        num_layers=int(config.num_layers),
        context_len=context_len,
        max_decode_steps=max_decode_steps,
        max_beam_width=max_beam_width,
        num_kv_heads=int(config.num_kv_heads),
        head_dim=int(config.head_dim),
        bytes_per_element=bytes_per_element,
        vocab_size=vocab_size or getattr(config, "vocab_size", None),
        active_beam_width=active_beam_width,
    )


@dataclass(frozen=True)
class GRKVLease:
    """Accounting lease for one request's ContextKV and BeamKV footprint."""

    request_id: str
    context_tokens: int
    beam_slots: int
    context_pages: tuple[int, ...] = ()
    beam_pages: tuple[int, ...] = ()
    context_page_size: int | None = None
    beam_page_size: int | None = None

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "request_id": self.request_id,
            "context_tokens": self.context_tokens,
            "beam_slots": self.beam_slots,
        }
        if self.context_pages:
            metadata.update(
                {
                    "context_pages": self.context_pages,
                    "context_page_size": self.context_page_size,
                    "context_capacity_tokens": len(self.context_pages)
                    * int(self.context_page_size or 0),
                }
            )
        if self.beam_pages:
            metadata.update(
                {
                    "beam_pages": self.beam_pages,
                    "beam_page_size": self.beam_page_size,
                    "beam_capacity_slots": len(self.beam_pages)
                    * int(self.beam_page_size or 0),
                }
            )
        return metadata


@dataclass(frozen=True)
class GRBeamKVPoolLease:
    """A dense BeamKV view leased from a service-level pool."""

    request_id: str
    slot: int
    beam_kv: BeamKV

    def metadata(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "slot": self.slot,
            "beam_kv_shape": self.beam_kv.key_shape,
        }


@dataclass(frozen=True)
class GRContextKVPoolLease:
    """A dense ContextKV view leased from a service-level pool."""

    request_id: str
    slot: int
    context_len: int
    context_kv: ContextKV

    def metadata(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "slot": self.slot,
            "context_len": self.context_len,
            "context_kv_shape": self.context_kv.key_shape,
        }


@dataclass
class GRKVLeaseAllocator:
    """Track service-level KV ownership under simple capacity limits."""

    max_running_requests: int | None = None
    max_context_tokens: int | None = None
    max_beam_slots: int | None = None
    leases: dict[str, GRKVLease] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_positive_optional_limits(
            (
                ("max_running_requests", self.max_running_requests),
                ("max_context_tokens", self.max_context_tokens),
                ("max_beam_slots", self.max_beam_slots),
            )
        )

    def can_allocate(
        self,
        *,
        request_id: str,
        context_tokens: int,
        beam_slots: int,
    ) -> bool:
        self.validate()
        if request_id in self.leases:
            return True
        usage = self.usage()
        if (
            self.max_running_requests is not None
            and usage["running_requests"] + 1 > self.max_running_requests
        ):
            return False
        if (
            self.max_context_tokens is not None
            and usage["context_tokens"] + int(context_tokens) > self.max_context_tokens
        ):
            return False
        if (
            self.max_beam_slots is not None
            and usage["beam_slots"] + int(beam_slots) > self.max_beam_slots
        ):
            return False
        return True

    def allocate(
        self,
        *,
        request_id: str,
        context_tokens: int,
        beam_slots: int,
    ) -> GRKVLease:
        if request_id in self.leases:
            raise ValueError(f"request already has a KV lease: {request_id}")
        if not self.can_allocate(
            request_id=request_id,
            context_tokens=context_tokens,
            beam_slots=beam_slots,
        ):
            raise MemoryError(f"insufficient KV capacity for request {request_id}")
        lease = GRKVLease(
            request_id=request_id,
            context_tokens=int(context_tokens),
            beam_slots=int(beam_slots),
        )
        self.leases[request_id] = lease
        return lease

    def release(self, request_id: str) -> GRKVLease | None:
        return self.leases.pop(request_id, None)

    def usage(self) -> dict[str, int]:
        return {
            "running_requests": len(self.leases),
            "context_tokens": sum(
                lease.context_tokens for lease in self.leases.values()
            ),
            "beam_slots": sum(lease.beam_slots for lease in self.leases.values()),
        }

    def status(self) -> dict[str, Any]:
        usage = self.usage()
        return {
            **usage,
            "max_running_requests": self.max_running_requests,
            "max_context_tokens": self.max_context_tokens,
            "max_beam_slots": self.max_beam_slots,
            "available_running_requests": _remaining_capacity(
                self.max_running_requests,
                usage["running_requests"],
            ),
            "available_context_tokens": _remaining_capacity(
                self.max_context_tokens,
                usage["context_tokens"],
            ),
            "available_beam_slots": _remaining_capacity(
                self.max_beam_slots,
                usage["beam_slots"],
            ),
            "running_request_utilization": _utilization(
                usage["running_requests"],
                self.max_running_requests,
            ),
            "context_token_utilization": _utilization(
                usage["context_tokens"],
                self.max_context_tokens,
            ),
            "beam_slot_utilization": _utilization(
                usage["beam_slots"],
                self.max_beam_slots,
            ),
            "lease_request_ids": sorted(self.leases),
        }


@dataclass
class GRPagedKVLeaseAllocator(GRKVLeaseAllocator):
    """Metadata-only paged KV allocator.

    It assigns page ids and tracks reuse. Tensor backing storage can later be
    attached to these page ids without changing scheduler lifecycle ownership.
    """

    context_page_size: int = 1
    beam_page_size: int = 1
    max_context_pages: int | None = None
    max_beam_pages: int | None = None
    free_context_pages: list[int] = field(default_factory=list)
    free_beam_pages: list[int] = field(default_factory=list)
    max_used_context_pages: int = 0
    max_used_beam_pages: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
            ("context_page_size", self.context_page_size),
            ("beam_page_size", self.beam_page_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("max_context_pages", self.max_context_pages),
            ("max_beam_pages", self.max_beam_pages),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.max_context_pages is not None and not self.free_context_pages:
            self.free_context_pages = list(range(self.max_context_pages))
        if self.max_beam_pages is not None and not self.free_beam_pages:
            self.free_beam_pages = list(range(self.max_beam_pages))

    def can_allocate(
        self,
        *,
        request_id: str,
        context_tokens: int,
        beam_slots: int,
    ) -> bool:
        if not super().can_allocate(
            request_id=request_id,
            context_tokens=context_tokens,
            beam_slots=beam_slots,
        ):
            return False
        if request_id in self.leases:
            return True
        return self._context_pages_needed(context_tokens) <= len(
            self.free_context_pages
        ) and self._beam_pages_needed(beam_slots) <= len(self.free_beam_pages)

    def allocate(
        self,
        *,
        request_id: str,
        context_tokens: int,
        beam_slots: int,
    ) -> GRKVLease:
        if request_id in self.leases:
            raise ValueError(f"request already has a KV lease: {request_id}")
        if not self.can_allocate(
            request_id=request_id,
            context_tokens=context_tokens,
            beam_slots=beam_slots,
        ):
            raise MemoryError(f"insufficient KV capacity for request {request_id}")
        context_pages = self._take_pages(
            self.free_context_pages,
            self._context_pages_needed(context_tokens),
        )
        beam_pages = self._take_pages(
            self.free_beam_pages,
            self._beam_pages_needed(beam_slots),
        )
        lease = GRKVLease(
            request_id=request_id,
            context_tokens=int(context_tokens),
            beam_slots=int(beam_slots),
            context_pages=context_pages,
            beam_pages=beam_pages,
            context_page_size=self.context_page_size,
            beam_page_size=self.beam_page_size,
        )
        self.leases[request_id] = lease
        usage = self.usage()
        self.max_used_context_pages = max(
            self.max_used_context_pages, usage["context_pages"]
        )
        self.max_used_beam_pages = max(self.max_used_beam_pages, usage["beam_pages"])
        return lease

    def release(self, request_id: str) -> GRKVLease | None:
        lease = self.leases.pop(request_id, None)
        if lease is None:
            return None
        self.free_context_pages.extend(lease.context_pages)
        self.free_context_pages.sort()
        self.free_beam_pages.extend(lease.beam_pages)
        self.free_beam_pages.sort()
        return lease

    def usage(self) -> dict[str, int]:
        usage = super().usage()
        usage.update(
            {
                "context_pages": sum(
                    len(lease.context_pages) for lease in self.leases.values()
                ),
                "beam_pages": sum(
                    len(lease.beam_pages) for lease in self.leases.values()
                ),
            }
        )
        return usage

    def status(self) -> dict[str, Any]:
        usage = self.usage()
        return {
            **super().status(),
            "context_page_size": self.context_page_size,
            "beam_page_size": self.beam_page_size,
            "max_context_pages": self.max_context_pages,
            "max_beam_pages": self.max_beam_pages,
            "max_used_context_pages": self.max_used_context_pages,
            "max_used_beam_pages": self.max_used_beam_pages,
            "free_context_pages": len(self.free_context_pages),
            "free_beam_pages": len(self.free_beam_pages),
            "available_context_pages": len(self.free_context_pages),
            "available_beam_pages": len(self.free_beam_pages),
            "context_page_capacity_tokens": usage["context_pages"]
            * self.context_page_size,
            "beam_page_capacity_slots": usage["beam_pages"] * self.beam_page_size,
            "context_internal_fragmentation_tokens": max(
                0,
                usage["context_pages"] * self.context_page_size
                - usage["context_tokens"],
            ),
            "beam_internal_fragmentation_slots": max(
                0,
                usage["beam_pages"] * self.beam_page_size - usage["beam_slots"],
            ),
            "context_free_page_runs": _free_runs(self.free_context_pages),
            "beam_free_page_runs": _free_runs(self.free_beam_pages),
            "largest_free_context_page_run": _largest_free_run(self.free_context_pages),
            "largest_free_beam_page_run": _largest_free_run(self.free_beam_pages),
            "context_page_utilization": _utilization(
                usage["context_pages"],
                self.max_context_pages,
            ),
            "beam_page_utilization": _utilization(
                usage["beam_pages"],
                self.max_beam_pages,
            ),
        }

    def _context_pages_needed(self, context_tokens: int) -> int:
        return _ceil_div(int(context_tokens), self.context_page_size)

    def _beam_pages_needed(self, beam_slots: int) -> int:
        return _ceil_div(int(beam_slots), self.beam_page_size)

    @staticmethod
    def _take_pages(free_pages: list[int], count: int) -> tuple[int, ...]:
        pages = tuple(free_pages[:count])
        del free_pages[:count]
        return pages


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0
    return (value + divisor - 1) // divisor


def validate_positive_optional_limits(
    limits: tuple[tuple[str, int | None], ...],
) -> None:
    for name, value in limits:
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when set")


def _remaining_capacity(capacity: int | None, used: int) -> int | None:
    if capacity is None:
        return None
    return max(0, int(capacity) - int(used))


def _utilization(used: int, capacity: int | None) -> float | None:
    if capacity is None or capacity <= 0:
        return None
    return int(used) / int(capacity)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return int(numerator) / int(denominator)


def _free_runs(values: list[int]) -> tuple[tuple[int, int], ...]:
    if not values:
        return ()
    sorted_values = sorted(values)
    runs: list[tuple[int, int]] = []
    start = previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))
    return tuple(runs)


def _largest_free_run(values: list[int]) -> int:
    runs = _free_runs(values)
    if not runs:
        return 0
    return max(end - start + 1 for start, end in runs)


def _first_free_run(values: list[int], length: int) -> int | None:
    for start, end in _free_runs(values):
        if end - start + 1 >= length:
            return start
    return None


def _release_slot_lease(
    request_id: str,
    *,
    leases: dict[str, Any],
    free_slots: list[int],
) -> Any | None:
    lease = leases.pop(request_id, None)
    if lease is None:
        return None
    free_slots.append(lease.slot)
    free_slots.sort()
    return lease


def _release_dense_pool_slot(pool: Any, request_id: str) -> Any | None:
    lease = _release_slot_lease(
        request_id,
        leases=pool.leases,
        free_slots=pool.free_slots,
    )
    if lease is not None:
        pool.release_count += 1
    return lease


@dataclass
class GRDenseBeamKVPool:
    """Dense BeamKV pool with one contiguous batch slot per request.

    This is intentionally not a paged attention layout. Each request gets a
    dense `[L, 1, S_dec, W, Hkv, D]` view, so the existing BeamKV writer and GR
    decode attention adapter keep their current hot-path contract.
    """

    key: Any
    value: Any
    free_slots: list[int] = field(default_factory=list)
    leases: dict[str, GRBeamKVPoolLease] = field(default_factory=dict)
    max_used_slots: int = 0
    allocation_count: int = 0
    release_count: int = 0

    def __post_init__(self) -> None:
        shape = _shape_of(self.key)
        if shape != _shape_of(self.value):
            raise ValueError(
                f"BeamKV pool key/value shapes differ: {shape} vs {_shape_of(self.value)}"
            )
        if len(shape) != 6:
            raise ValueError(
                "BeamKV pool expects [layers, slots, max_decode_steps, max_beam_width, "
                f"kv_heads, head_dim], got {shape}"
            )
        if not self.free_slots:
            self.free_slots = list(range(shape[1]))

    @classmethod
    def like_context(
        cls,
        context_kv: ContextKV,
        *,
        max_requests: int,
        max_decode_steps: int,
        max_beam_width: int,
    ) -> "GRDenseBeamKVPool":
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if max_decode_steps <= 0:
            raise ValueError("max_decode_steps must be positive")
        if max_beam_width <= 0:
            raise ValueError("max_beam_width must be positive")
        shape = (
            context_kv.num_layers,
            max_requests,
            max_decode_steps,
            max_beam_width,
            context_kv.num_kv_heads,
            context_kv.head_dim,
        )
        key = _empty_like(context_kv.key, shape)
        value = _empty_like(context_kv.value, shape)
        return cls(key=key, value=value)

    @property
    def capacity(self) -> int:
        return _shape_of(self.key)[1]

    def can_allocate(self, request_id: str) -> bool:
        return request_id in self.leases or bool(self.free_slots)

    def allocate(self, request_id: str) -> GRBeamKVPoolLease:
        if request_id in self.leases:
            raise ValueError(f"request already has a BeamKV pool lease: {request_id}")
        if not self.free_slots:
            raise MemoryError(
                f"no BeamKV pool slots available for request {request_id}"
            )
        slot = self.free_slots.pop(0)
        lease = GRBeamKVPoolLease(
            request_id=request_id,
            slot=slot,
            beam_kv=BeamKV(
                self.key[:, slot : slot + 1],
                self.value[:, slot : slot + 1],
            ),
        )
        self.leases[request_id] = lease
        self.allocation_count += 1
        self.max_used_slots = max(self.max_used_slots, len(self.leases))
        return lease

    def release(self, request_id: str) -> GRBeamKVPoolLease | None:
        return _release_dense_pool_slot(self, request_id)

    def usage(self) -> dict[str, int]:
        return {
            "beam_kv_pool_capacity": self.capacity,
            "beam_kv_pool_used": len(self.leases),
            "beam_kv_pool_free": len(self.free_slots),
        }

    def status(self) -> dict[str, Any]:
        usage = self.usage()
        return {
            **usage,
            "beam_kv_pool_slot_shape": _shape_of(self.key),
            "beam_kv_pool_slot_allocation_policy": "lowest_free_slot",
            "beam_kv_pool_release_policy": "immediate_reuse",
            "beam_kv_pool_allocation_count": self.allocation_count,
            "beam_kv_pool_release_count": self.release_count,
            "beam_kv_pool_max_used": self.max_used_slots,
            "beam_kv_pool_utilization": _utilization(
                usage["beam_kv_pool_used"],
                usage["beam_kv_pool_capacity"],
            ),
            "beam_kv_pool_high_watermark_utilization": _utilization(
                self.max_used_slots,
                usage["beam_kv_pool_capacity"],
            ),
            "free_slot_runs": _free_runs(self.free_slots),
            "lease_request_ids": sorted(self.leases),
        }


@dataclass
class GRDenseContextKVPool:
    """Dense ContextKV pool with one contiguous batch slot per request."""

    key: Any
    value: Any
    free_slots: list[int] = field(default_factory=list)
    leases: dict[str, GRContextKVPoolLease] = field(default_factory=dict)
    max_used_slots: int = 0
    allocation_count: int = 0
    release_count: int = 0

    def __post_init__(self) -> None:
        shape = _shape_of(self.key)
        if shape != _shape_of(self.value):
            raise ValueError(
                f"ContextKV pool key/value shapes differ: {shape} vs {_shape_of(self.value)}"
            )
        if len(shape) != 5:
            raise ValueError(
                "ContextKV pool expects [layers, slots, max_context_len, kv_heads, head_dim], "
                f"got {shape}"
            )
        if not self.free_slots:
            self.free_slots = list(range(shape[1]))

    @property
    def capacity(self) -> int:
        return _shape_of(self.key)[1]

    @property
    def max_context_len(self) -> int:
        return _shape_of(self.key)[2]

    def can_allocate(self, request_id: str, *, context_len: int) -> bool:
        return request_id in self.leases or (
            bool(self.free_slots) and context_len <= self.max_context_len
        )

    def allocate(self, request_id: str, *, context_len: int) -> GRContextKVPoolLease:
        if request_id in self.leases:
            raise ValueError(
                f"request already has a ContextKV pool lease: {request_id}"
            )
        if context_len <= 0 or context_len > self.max_context_len:
            raise ValueError("context_len must be in (0, max_context_len]")
        if not self.free_slots:
            raise MemoryError(
                f"no ContextKV pool slots available for request {request_id}"
            )
        slot = self.free_slots.pop(0)
        lease = GRContextKVPoolLease(
            request_id=request_id,
            slot=slot,
            context_len=context_len,
            context_kv=ContextKV(
                self.key[:, slot : slot + 1, :context_len],
                self.value[:, slot : slot + 1, :context_len],
            ),
        )
        self.leases[request_id] = lease
        self.allocation_count += 1
        self.max_used_slots = max(self.max_used_slots, len(self.leases))
        return lease

    def allocate_batch(
        self,
        request_ids: tuple[str, ...],
        *,
        context_len: int,
    ) -> tuple[GRContextKVPoolLease, ...] | None:
        if not request_ids:
            return ()
        if any(request_id in self.leases for request_id in request_ids):
            raise ValueError("request already has a ContextKV pool lease")
        if context_len <= 0 or context_len > self.max_context_len:
            raise ValueError("context_len must be in (0, max_context_len]")
        count = len(request_ids)
        run_start = _first_free_run(self.free_slots, count)
        if run_start is None:
            return None
        slots = tuple(range(run_start, run_start + count))
        for slot in slots:
            self.free_slots.remove(slot)
        leases = tuple(
            GRContextKVPoolLease(
                request_id=request_id,
                slot=slot,
                context_len=context_len,
                context_kv=ContextKV(
                    self.key[:, slot : slot + 1, :context_len],
                    self.value[:, slot : slot + 1, :context_len],
                ),
            )
            for request_id, slot in zip(request_ids, slots)
        )
        for lease in leases:
            self.leases[lease.request_id] = lease
        self.allocation_count += count
        self.max_used_slots = max(self.max_used_slots, len(self.leases))
        return leases

    def release(self, request_id: str) -> GRContextKVPoolLease | None:
        return _release_dense_pool_slot(self, request_id)

    def usage(self) -> dict[str, int]:
        return {
            "context_kv_pool_capacity": self.capacity,
            "context_kv_pool_used": len(self.leases),
            "context_kv_pool_free": len(self.free_slots),
        }

    def status(self) -> dict[str, Any]:
        usage = self.usage()
        return {
            **usage,
            "context_kv_pool_slot_shape": _shape_of(self.key),
            "context_kv_pool_allocation_count": self.allocation_count,
            "context_kv_pool_release_count": self.release_count,
            "context_kv_pool_max_used": self.max_used_slots,
            "context_kv_pool_utilization": _utilization(
                usage["context_kv_pool_used"],
                usage["context_kv_pool_capacity"],
            ),
            "context_kv_pool_high_watermark_utilization": _utilization(
                self.max_used_slots,
                usage["context_kv_pool_capacity"],
            ),
            "lease_request_ids": sorted(self.leases),
        }


def _empty_like(reference: Any, shape: tuple[int, ...]) -> Any:
    if hasattr(reference, "new_empty"):
        return reference.new_empty(shape)
    if hasattr(reference, "with_shape"):
        return reference.with_shape(shape)
    raise TypeError(
        f"cannot allocate BeamKV pool from reference type {type(reference)!r}"
    )


def _shape_of(tensor: Any) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise TypeError(f"object has no shape: {type(tensor)!r}")
    return tuple(int(dim) for dim in shape)
