"""Hedged tool calls: a duplicate request when a tool call is slower than its recent tail. 

Why: in tool-heavy workloads the trajectory, not the GPU, is the bottleneck.  Example (a flaky search service: 5% of
tool attempts hang for the 60 s service timeout, 5% fail fast, periodic short outages): trajectories spent ~70% of their
wall time in tool calls, and ~75% of that tool time was the ~1 hung attempt per trajectory.  Under a staleness ceiling
the number of trajectories in flight is bounded (one batch ahead), so the generation rate is
(trajectories in flight) / (trajectory latency): every hung call directly stretches the trainer's data wait.

What: ``hedged(fn)`` wraps ``async fn(query, *a, **kw) -> str``.  The call starts normally; if it has not returned after
the hedge delay, an identical second call starts (and a third at 2x the delay, ``ASRT_TOOL_HEDGE_MAX``); the first
successful answer wins and the others are cancelled.  A result that looks like a tool error (``ASRT_TOOL_HEDGE_ERR``
regex, default matches a "tool_error" document) only wins if every attempt failed.  Delay
(``ASRT_TOOL_HEDGE``): ``auto`` (default) = factor x the ``ASRT_TOOL_HEDGE_Q`` (0.9) quantile of the last ``window``
times-to-answer (floor ``ASRT_TOOL_HEDGE_MIN``=2 s, 10 s until 64 samples), or a fixed number of seconds; ``0`` = off.
Not the p95: with 5% hung attempts the p95 IS the hang (example: p50 0.8 s, p90 5.8 s, p95 61 s).  Offline replay of
that tool log: hedge at 5-8 s -> mean call 5.27 -> 2.3-2.6 s for 9-12% extra tool requests.
The answer is the tool's own answer for the same query (no content change); cost = extra tool requests (~p of calls
slower than the delay, reported in the stats).  Standard tail-at-scale technique; works for any async tool function.
In slime: set ``ASRT_TOOL_HEDGE`` together with ai_search_plugin.adapters.slime_toolcache (the hedge sits under the cache, so
joiners of an in-flight query are released by whichever attempt answers first); only targets listed in
``ASRT_TOOL_HEDGE_TARGETS`` are hedged (no default: list your read-only search functions).  Measured cost in one
workload: +15% tool requests, same tool content, unchanged train/rollout logp diff -> turn it down/off for a
rate-limited or paid external API: ASRT_TOOL_HEDGE=0, a larger fixed delay, or ASRT_TOOL_HEDGE_MAX=1).
"""
from __future__ import annotations

import asyncio
import collections
import functools
import json
import os
import re
import time

_STATS = {"calls": 0, "hedged": 0, "hedge_won": 0, "extra_requests": 0, "errors_masked": 0}


def _cfg():
    v = os.environ.get("ASRT_TOOL_HEDGE", "auto").strip().lower()
    if v in ("0", "off", "false", "no", ""):
        return None
    return v


class _Delay:
    def __init__(self, spec, window=512, factor=1.0, floor=2.0, initial=10.0):
        self.fixed = None if spec == "auto" else float(spec)
        self.lat = collections.deque(maxlen=window)
        self.factor = float(os.environ.get("ASRT_TOOL_HEDGE_FACTOR", factor))
        self.floor = float(os.environ.get("ASRT_TOOL_HEDGE_MIN", floor))
        self.initial = initial

    def add(self, s):
        self.lat.append(s)

    def __call__(self):
        if self.fixed is not None:
            return self.fixed
        if len(self.lat) < 64:
            return self.initial
        xs = sorted(self.lat)
        q = float(os.environ.get("ASRT_TOOL_HEDGE_Q", 0.9))
        return max(self.floor, self.factor * xs[int(q * (len(xs) - 1))])


# Only read-only, idempotent tools may be hedged (a duplicate request must not change anything).  Code execution, writes,
# anything with side effects: never.  ASRT_TOOL_HEDGE_TARGETS = comma list of module.attr names allowed (default: the
# nothing - the user lists the read-only search/retrieve functions of their agent).
_SAFE_DEFAULT = ""


def describe(targets=None) -> str:
    """ one-line startup description: on/off, hedged targets, delay rule, extra-request cap."""
    spec = _cfg()
    if spec is None:
        return f"off (ASRT_TOOL_HEDGE={os.environ.get('ASRT_TOOL_HEDGE', '')!r})"
    if spec == "auto":
        d = (f"auto(q{os.environ.get('ASRT_TOOL_HEDGE_Q', '0.9')} x{os.environ.get('ASRT_TOOL_HEDGE_FACTOR', '1.0')}"
                          f" of last 512, floor {os.environ.get('ASRT_TOOL_HEDGE_MIN', '2.0')} s, 10 s until 64 samples)")
    else:
        d = f"{float(spec):g} s"
    tg = ",".join(targets) if targets is not None else os.environ.get("ASRT_TOOL_HEDGE_TARGETS", _SAFE_DEFAULT)
    return f"on targets={tg} delay={d} max_extra={os.environ.get('ASRT_TOOL_HEDGE_MAX', '2')}"


def allowed(target: str) -> bool:
    ok = {t.strip() for t in os.environ.get("ASRT_TOOL_HEDGE_TARGETS", _SAFE_DEFAULT).split(",") if t.strip()}
    return target.strip() in ok


def hedged(fn, spec=None, max_extra=None):
    spec = spec if spec is not None else _cfg()
    if spec is None:
        return fn
    max_extra = int(max_extra if max_extra is not None else os.environ.get("ASRT_TOOL_HEDGE_MAX", 2))
    err_re = re.compile(os.environ.get("ASRT_TOOL_HEDGE_ERR", r"tool_error"))
    delay = _Delay(spec)

    def is_err(out):
        return isinstance(out, str) and bool(err_re.search(out[:400]))

    @functools.wraps(fn)
    async def wrapper(query, *a, **kw):
        _STATS["calls"] += 1
        t0 = time.monotonic()
        d = delay()
        tasks = [asyncio.ensure_future(fn(query, *a, **kw))]
        pending = set(tasks)
        last_err = None
        last_exc = None
        try:
            while True:
                n_more = (len(tasks) - 1) < max_extra
                timeout = (d * len(tasks) - (time.monotonic() - t0)) if n_more else None
                done, pending = await asyncio.wait(pending, timeout=None if timeout is None else max(0.0, timeout),
                                                   return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    exc = t.exception()
                    if exc is not None:
                        last_exc = exc
                        continue
                    out = t.result()
                    if is_err(out):
                        last_err = out
                        continue
                    if t is not tasks[0]:
                        _STATS["hedge_won"] += 1
                    if last_err is not None:
                        _STATS["errors_masked"] += 1
                    delay.add(time.monotonic() - t0)  # time to answer (hedged or not)
                    return out
                if not pending and not n_more:
                    break
                if not done or not pending:  # hedge delay passed (or every attempt so far failed): one more attempt
                    if (len(tasks) - 1) < max_extra:
                        if len(tasks) == 1:
                            _STATS["hedged"] += 1
                        _STATS["extra_requests"] += 1
                        nt = asyncio.ensure_future(fn(query, *a, **kw))
                        tasks.append(nt)
                        pending.add(nt)
                    elif not pending:
                        break
            if last_err is not None:
                return last_err
            raise last_exc if last_exc is not None else RuntimeError("tool_hedge: no attempt finished")
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    wrapper.__hedged__ = True
    return wrapper


def stats():
    s = dict(_STATS)
    s["extra_ratio"] = round(s["extra_requests"] / s["calls"], 4) if s["calls"] else None
    return s


def log_stats():
    st = stats()
    print(f"[asrt] tool_hedge {st}", flush=True)
    d = os.environ.get("ASRT_LOG_DIR")
    if d:
        try:
            with open(os.path.join(d, "toolhedge.jsonl"), "a") as f:
                f.write(json.dumps({"t": time.time(), "pid": os.getpid(), **st}) + "\n")
        except OSError:
            pass
