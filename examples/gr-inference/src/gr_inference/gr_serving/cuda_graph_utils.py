"""Shared CUDA graph runner utilities."""

from __future__ import annotations

import os
from typing import Any


class CudaGraphCacheMixin:
    """Common miss accounting and LRU storage for graph runners."""

    max_entries: int
    _graphs: Any
    evictions: int
    miss_count: int
    miss_reasons: dict[str, int]
    fallback_eager_count: int

    def _record_miss(self, reason: str, *, fallback: bool = False) -> None:
        self.miss_count += 1
        self.miss_reasons[reason] = self.miss_reasons.get(reason, 0) + 1
        if fallback:
            self.fallback_eager_count += 1

    def _record_fallback(self, reason: str) -> None:
        self.miss_reasons[reason] = self.miss_reasons.get(reason, 0) + 1
        self.fallback_eager_count += 1

    def _store_graph(self, key: tuple[Any, ...], entry: Any) -> None:
        if self.max_entries <= 0:
            if self._graphs:
                self.evictions += len(self._graphs)
                self._graphs.clear()
            return
        self._graphs[key] = entry
        self._graphs.move_to_end(key)
        while len(self._graphs) > self.max_entries:
            self._graphs.popitem(last=False)
            self.evictions += 1


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def is_cuda_tensor(value: Any) -> bool:
    return bool(getattr(value, "is_cuda", False))


def tensor_nbytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel()) * int(value.element_size())
    return 0


def tensor_data_ptr(value: Any) -> int | None:
    data_ptr = getattr(value, "data_ptr", None)
    if data_ptr is None:
        return None
    return int(data_ptr())


def tensor_view_key(value: Any) -> tuple[int | None, int]:
    storage_offset = getattr(value, "storage_offset", None)
    offset = int(storage_offset()) if callable(storage_offset) else 0
    return tensor_data_ptr(value), offset
