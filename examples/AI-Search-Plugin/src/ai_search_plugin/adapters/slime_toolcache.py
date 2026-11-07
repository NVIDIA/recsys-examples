"""Tool-result cache + request merging for slime custom generate functions.

Two ways to use it:

1. No code change in the agent (recommended):
       --custom-generate-function-path ai_search_plugin.adapters.slime_toolcache.generate
       env ASRT_TOOLCACHE_GENERATE=my_agent.generate                (the real generate fn; required)
           ASRT_TOOLCACHE_TARGETS=my_agent_tools.search,my_agent.search
                                    (module.attr of every name the agent calls the tool through)
           ASRT_TOOLCACHE_TTL_S=604800  ASRT_TOOLCACHE_CAPACITY=1048576   (optional)
   On first call this replaces each target with one shared ``CachedTool``
   (identical concurrent queries -> one real call; repeats -> served from
   cache; failures are not cached) and then delegates to the real generate.
   Cache stats are printed every ``ASRT_TOOLCACHE_LOG_EVERY`` calls (default
   2000) and written to ``$ASRT_LOG_DIR/toolcache.jsonl``.

2. In your own generate function:
       from ai_search_plugin.adapters.slime_toolcache import cached
       search = cached(search)            # async fn(query, **kw) -> str

Works without the C++ extension (pure-Python cache), without slime installed
(import is lazy) and across event loops/threads.
Only the first positional argument (the query) is the cache key; calls with
extra keyword arguments (e.g. a non-default topk) bypass the cache unless
``ASRT_TOOLCACHE_KEY_KWARGS=1`` (then kwargs are part of the key).
"""
from __future__ import annotations

import functools
import importlib
import json
import os
import time

from .common import CachedTool, query_key

_state = {"tool": None, "patched": [], "gen": None, "calls": 0}


def _no_cache():
    """ final tool error documents are not cached (ASRT_TOOLCACHE_NOCACHE_RE; '' disables)."""
    import re

    pat = os.environ.get("ASRT_TOOLCACHE_NOCACHE_RE", r"tool_error|[Ss]earch failed|service failed")
    if not pat:
        return None
    rx = re.compile(pat)
    return lambda out: isinstance(out, str) and bool(rx.search(out))


def cached(fn, capacity: int = None, ttl_s: float = None):
    """Wrap ``async fn(query, *a, **kw)``; returns an async callable with ``.stats()``."""
    capacity = int(capacity or os.environ.get("ASRT_TOOLCACHE_CAPACITY", 1 << 20))
    ttl_s = float(ttl_s or os.environ.get("ASRT_TOOLCACHE_TTL_S", 7 * 86400))
    key_kwargs = os.environ.get("ASRT_TOOLCACHE_KEY_KWARGS", "0") == "1"
    inner = {}

    def tool_for(extra_key):
        t = inner.get(extra_key)
        if t is None:
            def call(q, _ek=extra_key):
                a, kw = _ek_args.get(_ek, ((), {}))
                return fn(q, *a, **kw)
            t = inner[extra_key] = CachedTool(call, capacity=capacity, ttl_s=ttl_s, no_cache=_no_cache())
        return t

    _ek_args = {}
    every = int(os.environ.get("ASRT_TOOLCACHE_LOG_EVERY", 2000))

    @functools.wraps(fn)
    async def wrapper(query, *a, **kw):
        if (a or kw) and not key_kwargs:
            return await fn(query, *a, **kw)
        ek = json.dumps([list(a), kw], sort_keys=True, default=str) if (a or kw) else ""
        _ek_args.setdefault(ek, (a, kw))
        out = await tool_for(ek)(query)
        _state["calls"] += 1
        if every and _state["calls"] % every == 0:
            _report(wrapper)
        return out

    def stats():
        agg = {"hits": 0, "joins": 0, "misses": 0, "errors_not_cached": 0}
        for t in inner.values():
            s = t.stats()
            for k in agg:
                agg[k] += s[k]
            agg["backend"] = s["backend"]
        n = agg["hits"] + agg["joins"] + agg["misses"]
        agg["real_call_ratio"] = round(agg["misses"] / n, 4) if n else None
        return agg

    wrapper.stats = stats
    wrapper.__wrapped_tool__ = fn
    return wrapper


def _report(w):
    st = w.stats()
    if os.environ.get("ASRT_TOOL_HEDGE"):
        from .tool_hedge import stats as _hs
        st["hedge"] = _hs()
    print(f"[asrt] toolcache {st}", flush=True)
    d = os.environ.get("ASRT_LOG_DIR")
    if d:
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "toolcache.jsonl"), "a") as f:
                f.write(json.dumps({"t": time.time(), "pid": os.getpid(), **st}) + "\n")
        except OSError:
            pass


def _resolve(path):
    mod, _, attr = path.rpartition(".")
    return importlib.import_module(mod), attr


def install(targets=None) -> list:
    """Replace each ``module.attr`` target with one shared cached wrapper."""
    targets = targets if targets is not None else [
        t for t in os.environ.get("ASRT_TOOLCACHE_TARGETS", "").split(",") if t.strip()]
    patched = []
    for t in targets:
        try:
            mod, attr = _resolve(t.strip())
        except Exception:
            continue
        cur = getattr(mod, attr, None)
        if cur is None:
            continue
        if getattr(cur, "__wrapped_tool__", None) is not None:
            patched.append(t)
            continue
        if _state["tool"] is None:
            if os.environ.get("ASRT_TOOL_HEDGE"):  # hedged tool calls under the cache (ai_search_plugin.adapters.tool_hedge)
                from .tool_hedge import allowed, describe, hedged
                if allowed(t):  # read-only targets only (ASRT_TOOL_HEDGE_TARGETS)
                    cur = hedged(cur)
                    print(f"[asrt] tool_hedge: {describe([x.strip() for x in targets if allowed(x.strip())])}", flush=True)
                else:
                    print(f"[asrt] tool_hedge: {t} not in ASRT_TOOL_HEDGE_TARGETS, not hedged", flush=True)
            _state["tool"] = cached(cur)
        setattr(mod, attr, _state["tool"])
        patched.append(t)
    _state["patched"] = patched
    return patched


async def generate(args, sample, sampling_params, *a, **kw):
    """Drop-in ``--custom-generate-function-path`` entry (see module doc)."""
    if _state["gen"] is None:
        target = os.environ.get("ASRT_TOOLCACHE_GENERATE", "").strip()
        if not target:
            raise RuntimeError("ai_search_plugin toolcache: set ASRT_TOOLCACHE_GENERATE=<module.attr of your real "
                               "slime generate function> (and ASRT_TOOLCACHE_TARGETS=<module.attr of the tool functions>)")
        mod, attr = _resolve(target)
        install()
        _state["gen"] = getattr(mod, attr)
        print(f"[asrt] toolcache: generate={target} patched={_state['patched']}", flush=True)
    return await _state["gen"](args, sample, sampling_params, *a, **kw)


def stats():
    return _state["tool"].stats() if _state["tool"] is not None else {}


__all__ = ["cached", "install", "generate", "stats", "query_key"]
