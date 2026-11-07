"""Size slime's fully-async in-flight pool from the batch and the engines, not from N. 

slime's ``AsyncRolloutWorker`` keeps ``sglang_server_concurrency * engines / n_samples_per_prompt``
groups in flight (x ``--fully-async-group-prefetch-factor``) and stops topping up once that many
completed groups are queued.  The per-sample semaphore is ``sglang_server_concurrency * engines``
and a sample holds it for its whole multi-turn trajectory (tool calls included).

With a large group (GRPO n=16) the pool collapses: e.g. single node, 2 engines, concurrency 64,
batch 32 groups x 16 admits 8 groups at a time = 1/4 batch.  When train(i) ends only ~15 groups
are ready, the other ~17 are generated serially (wait ~180 s/step, ~33 running requests per engine
vs verl's ~100 with the same max-running 128).

Fix (generic, only acts when the pool is smaller than one batch, so layouts already sized to one
batch - e.g. 68 x 4 engines / 4 = 68 groups = batch - are untouched):
  * per-engine sample concurrency -> min(max_running_requests, batch samples / engines)
    (never lowered);
  * group prefetch factor -> enough groups for one full batch in flight (+ the queue gate that
    goes with it, i.e. at most ~1 batch generated ahead; staleness rises to ~1, like verl's
    0.7 at the same knobs);
  * re-create slime's HTTP client with the new connection budget (it was built in the
    RolloutManager's __init__ from the old value; still idle on the first rollout call).
Staleness ceiling (ASRT_MAX_STALENESS=S, slime_admit): the in-flight target is clipped to the
group budget the ceiling allows - at most max(1, S-1) batches outstanding under the overlap driver (ASRT_PREFETCH has
``overlap``: batch i+1 is handed out during the sync after train(i)), S batches otherwise (same formula slime_admit's
cap uses) - and the factor is rounded DOWN so groups in flight never exceed it (no cap: rounded up, unchanged).  A pool
that is already at/above the budget is left alone (slime_admit's gate bounds it).  Example (1 engine, conc 128,
n 4, batch 68): no cap 32 -> 96 groups (1.4 batches; staleness up to 5 with abort syncs); S=2 + overlap: 32 -> 64.
Must run before the first ``generate`` call (the worker / GenerateState singleton read the args
there).  Env: ASRT_POOL=auto (default) | 0 off; ASRT_POOL_BATCHES=<float> groups in flight in
batches (default 1.0); ASRT_POOL_SEM=<int> per-engine sample concurrency override.
Log: one ``[asrt] pool ...`` line + ``$ASRT_LOG_DIR/pool.jsonl``.
"""
from __future__ import annotations

import json
import math
import os

_done = False


def _engines(args) -> int:
    try:
        from slime.utils.http_utils import get_rollout_num_engines

        return int(get_rollout_num_engines(args))
    except Exception:
        g = int(getattr(args, "rollout_num_gpus", 0) or 0)
        return max(1, g // int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)) if g else 0


def max_staleness() -> int | None:
    """ASRT_MAX_STALENESS, parsed like ai_search_plugin.adapters.slime_admit.max_staleness (not imported: importing slime_admit
    installs its patches)."""
    v = os.environ.get("ASRT_MAX_STALENESS", "").strip().lower()
    if v in ("", "off", "none", "0", "no"):
        return None
    return max(1, int(v))


def _fill_on() -> bool:
    """Default ON in ticket mode (keeps every engine at max_running_requests; measured 10-20% faster steps on
    multi-turn workloads, no change when the pool is already full).  ASRT_POOL_FILL=off restores the max(1, S-1)-batch pool."""
    return os.environ.get("ASRT_POOL_FILL", "engines").strip().lower() not in ("off", "0", "no", "false", "")


def _ticket_mode(S) -> bool:
    return S is not None and os.environ.get("ASRT_STALE_MODE", "ticket").strip().lower() == "ticket"


def cap_batches(S: int | None, overlap: bool) -> float | None:
    """Batches the staleness ceiling lets be outstanding.  Count cap (slime_admit install_worker_cap): (ahead + 1) x
    batch = max(1, S-1) under the overlap driver.  Ticket mode (default): slime_admit admits a group only into
    a ticket t with t - rel <= S, so tickets m..m+S may be live = S+1 batches - the ticket gate, not the pool, bounds
    staleness; clipping the pool to 1 batch (the count-cap rule) left one benchmark at 48 generating requests/engine vs
    verl's 71."""
    if S is None:
        return None
    if _ticket_mode(S) and _fill_on():
        return float(S + 1)
    return float(max(1, S - 1)) if overlap else float(S)


def plan(conc: int, engines: int, n: int, rbs: int, max_running, factor: int = 1,
         batches: float = 1.0, sem_override: int = 0, cap: float | None = None, fill_engines: bool = False) -> dict:
    """Pure: returns {conc, factor, changed, groups_before, groups_after[, cap_groups]}.  ``cap`` = batches the
    staleness ceiling allows outstanding (None = no ceiling)."""
    n = max(1, n)
    before = max(1, conc * engines // n) * factor
    out = {"conc": conc, "factor": factor, "changed": False, "groups_before": before, "groups_after": before}
    want = int(math.ceil(batches * rbs))
    if fill_engines and max_running and engines > 0:
        # ASRT_POOL_FILL=engines: enough groups to keep every engine at max_running_requests (a sample holds its
        # slot through tool calls too, so this is an upper bound on generating requests; SGLang queues the excess)
        want = max(want, int(math.ceil(int(max_running) * engines / n)))
    limit = None
    if cap is not None and rbs > 0:
        limit = max(1, int(math.floor(cap * rbs)))
        out["cap_groups"] = limit
        want = min(want, limit)
    if engines <= 0 or rbs <= 0 or before >= want:
        return out
    new_conc = conc
    if sem_override > 0:
        new_conc = max(conc, sem_override)
    elif max_running:
        per = int(math.ceil(want * n / engines)) if fill_engines else int(math.ceil(rbs * n / engines))
        new_conc = max(conc, min(int(max_running), per))
    if limit is not None and new_conc * engines // n > limit:   # raised concurrency alone would exceed the budget
        new_conc = max(conc, limit * n // engines)
    groups = max(1, new_conc * engines // n)
    new_factor = max(factor, int(math.ceil(want / groups)))
    if limit is not None:   # round down to the budget: never more groups in flight than the ceiling allows
        new_factor = max(factor, min(new_factor, limit // groups))
    out.update(conc=new_conc, factor=new_factor, changed=(new_conc, new_factor) != (conc, factor),
               groups_after=groups * new_factor)
    return out


def _overlap_driver() -> bool:
    return "overlap" in {x.strip() for x in os.environ.get("ASRT_PREFETCH", "").split(",")}


def autosize(args) -> dict | None:
    global _done
    if _done:
        return None
    _done = True
    mode = os.environ.get("ASRT_POOL", "auto").strip().lower()
    if mode in ("0", "off", "false", "no"):
        return None
    E = _engines(args)
    p = plan(int(getattr(args, "sglang_server_concurrency", 0) or 0), E,
             int(getattr(args, "n_samples_per_prompt", 1) or 1), int(getattr(args, "rollout_batch_size", 0) or 0),
             getattr(args, "sglang_max_running_requests", None),
             int(getattr(args, "fully_async_group_prefetch_factor", 1) or 1),
             float(os.environ.get("ASRT_POOL_BATCHES", "1.0") or 1.0), int(os.environ.get("ASRT_POOL_SEM", "0") or 0),
             cap=cap_batches(max_staleness(), _overlap_driver()),
             fill_engines=_fill_on() and _ticket_mode(max_staleness()))
    p["engines"], p["max_staleness"] = E, max_staleness()
    if p["changed"]:
        old = args.sglang_server_concurrency
        args.sglang_server_concurrency = p["conc"]
        args.fully_async_group_prefetch_factor = p["factor"]
        if p["conc"] > old:
            try:
                from slime.utils import http_utils as H

                if getattr(H, "_http_client", None) is not None:
                    H._http_client = None
                    H.init_http_client(args)
                p["http_reinit"] = True
            except Exception as e:  # keep the old client (only caps connections)
                p["http_reinit"] = repr(e)
    print(f"[asrt] pool {'resized' if p['changed'] else 'unchanged'}: {p}", flush=True)
    d = os.environ.get("ASRT_LOG_DIR")
    if d:
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "pool.jsonl"), "a") as f:
                f.write(json.dumps(p) + "\n")
        except Exception:
            pass
    return p
