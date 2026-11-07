"""Pure-Python tool-result cache (TTL + LRU) with the same lookup/complete
contract as the C++ ToolCache of the original code base - used when no C++ extension
is built (e.g. inside a stock slime/verl/AReaL container).

lookup(key, now, ticket) -> (result, value):
    Hit    - value is the cached bytes
    Joined - someone is already computing this key; wait for complete()
    Miss   - caller must compute and then call complete(key, value, now, ok)
Thread-safe (one lock), so it can be shared across event loops/threads.
"""
from __future__ import annotations

import enum
import threading
from collections import OrderedDict
from typing import Optional, Tuple


class ToolCacheResult(enum.IntEnum):
    Miss = 0
    Hit = 1
    Joined = 2


class ToolCache:
    def __init__(self, capacity: int = 1 << 20, ttl_s: float = 7 * 86400):
        self.capacity = int(capacity)
        self.ttl_s = float(ttl_s)
        self._d: "OrderedDict[int, Tuple[bytes, float]]" = OrderedDict()
        self._inflight: dict = {}
        self._lock = threading.Lock()
        self.hits = self.joins = self.misses = 0

    def lookup(self, key: int, now: float, ticket: int = 0) -> Tuple[ToolCacheResult, Optional[bytes]]:
        with self._lock:
            e = self._d.get(key)
            if e is not None:
                if now - e[1] <= self.ttl_s:
                    self._d.move_to_end(key)
                    self.hits += 1
                    return ToolCacheResult.Hit, e[0]
                del self._d[key]
            if key in self._inflight:
                self.joins += 1
                return ToolCacheResult.Joined, None
            self._inflight[key] = ticket
            self.misses += 1
            return ToolCacheResult.Miss, None

    def complete(self, key: int, value: bytes, now: float, ok: bool = True) -> None:
        with self._lock:
            self._inflight.pop(key, None)
            if not ok:
                return
            self._d[key] = (bytes(value), now)
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)
