"""Framework-agnostic helpers used by all adapters (pure Python; the original code base could also use a C++
extension for the tool cache and partitioner - ai_search_plugin ships only the pure-Python versions)."""
import asyncio
import hashlib
import time
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

solvers = None  # no C++ extension in ai_search_plugin


def query_key(q: str) -> int:
    """Stable 64-bit key of a normalized search query."""
    norm = " ".join(q.lower().split())
    return int.from_bytes(hashlib.blake2b(norm.encode(), digest_size=8).digest(), "little") & ((1 << 63) - 1)


def _tool_cache_impl(capacity, ttl_s):
    """C++ ToolCache if the extension is built, else the pure-Python one."""
    if solvers is not None and hasattr(solvers, "ToolCache"):
        return solvers.ToolCache(capacity, ttl_s), solvers.ToolCacheResult
    from ..toolcache_py import ToolCache, ToolCacheResult

    return ToolCache(capacity, ttl_s), ToolCacheResult


class CachedTool:
    """asyncio wrapper around a tool cache: TTL cache + single-flight.

    Wrap any async tool function `fn(query) -> str`. Identical concurrent
    queries share one call; finished results are served from cache.
    Uses the pure-Python cache (`ai_search_plugin.toolcache_py`). Safe to share between event loops / threads
    (joiners on another loop are woken with call_soon_threadsafe).
    `key_fn` maps a query to the cache key (default: case/space-normalized).
    """

    def __init__(self, fn: Callable[[str], Awaitable[str]], capacity: int = 1 << 20, ttl_s: float = 7 * 86400,
                 key_fn: Optional[Callable[[str], int]] = None, no_cache: Optional[Callable[[str], bool]] = None):
        import threading

        # no_cache(out) -> True: a returned value that must not be cached (e.g. a tool's final error document,
        # HTTP 200 + error body); joiners of that call still get it, the next identical query calls the tool again.
        self.no_cache = no_cache
        self.errors_not_cached = 0

        self.fn = fn
        self.cache, self._R = _tool_cache_impl(capacity, ttl_s)
        self.backend = "cpp" if solvers is not None and hasattr(solvers, "ToolCache") else "python"
        self.key_fn = key_fn or query_key
        self._waiters: Dict[int, List] = {}
        self._next = 0
        self._lock = threading.Lock()

    async def __call__(self, query: str) -> str:
        k = self.key_fn(query)
        with self._lock:
            self._next += 1
            res, val = self.cache.lookup(k, time.monotonic(), self._next)
            if res == self._R.Joined:
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                self._waiters.setdefault(k, []).append((loop, fut))
        if res == self._R.Hit:
            return val.decode() if isinstance(val, (bytes, bytearray)) else val
        if res == self._R.Joined:
            return await fut
        err = None
        try:
            out = await self.fn(query)
            ok = True
        except Exception as e:  # do not cache failures; propagate to joiners
            out, ok, err = "", False, e
        store = ok and not (self.no_cache is not None and self.no_cache(out))
        if ok and not store:
            self.errors_not_cached += 1
        with self._lock:
            self.cache.complete(k, out.encode(), time.monotonic(), store)
            waiters = self._waiters.pop(k, [])
        for loop, f in waiters:
            if ok:
                loop.call_soon_threadsafe(_set_result, f, out)
            else:
                loop.call_soon_threadsafe(_set_exc, f, err)
        if not ok:
            raise err
        return out

    def stats(self):
        return {"hits": self.cache.hits, "joins": self.cache.joins, "misses": self.cache.misses,
                "backend": self.backend, "errors_not_cached": self.errors_not_cached}


def _set_result(f, v):
    if not f.done():
        f.set_result(v)


def _set_exc(f, e):
    if not f.done():
        f.set_exception(e)


def balanced_partitions(seqlens: Sequence[int], k: int, equal_size: bool = True, quad_coef: float = 0.0) -> List[List[int]]:
    """Cost-balanced k-way partition (cost = L + quad_coef * L^2).

    Same contract as verl/slime `get_seqlen_balanced_partitions(seqlen_list,
    k_partitions, equal_size)`: returns k lists of indices."""
    from ..partition_py import balanced_partitions as _bp
    return [sorted(p) for p in _bp(list(map(int, seqlens)), k, equal_size, quad_coef)]
