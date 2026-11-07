"""Rollout client sharding + per-engine admission for slime's fully-async rollout.

Why (observed with ~10k short multi-turn trajectories per step on 4 SGLang engines):
  * slime runs every trajectory of the step in ONE asyncio loop inside the RolloutManager: the agent loop, the
    tool calls, every HTTP generate call (JSON in/out incl. logprobs and R3 routed experts) and the bookkeeping.
    All requests go to the single sglang router, which forwards them to the engines.  The client kept ~8200
    generate calls in flight, a ~39-token call took p50 48 s, but each engine only ran 17-74 requests with an
    empty queue: the requests were stuck in the client / HTTP / router path, not on the GPU.  verl splits the
    trajectories over 8 agent-loop worker processes that talk to the engines directly (~153 in flight per engine,
    2.4x our decode tok/s) and generates the same batch in about half the time.

What this does (monkey patches, no slime source change, off unless ASRT_RS is set):
  * ``shard``  K worker processes (Ray actors, default 2 per engine, placed on the engine's node); shard k owns
               engine k mod E and sends its generate calls straight to that engine's HTTP server (the router is
               bypassed: each shard gets a copy of args whose sglang_router_ip/port point at its engine, so any
               custom generate function that builds its URL from the router address works unchanged).  The
               RolloutManager keeps slime's AsyncRolloutWorker (admission in prompt order, group pool, completed
               queue, pause/resume, ABORTED-group requeue, staleness accounting): only the per-group call
               ``generate_and_rm_group`` is replaced by "send the group to the least-loaded shard and await the
               finished group".  The shard runs slime's own generate_and_rm_group (custom generate function,
               tool calls, reward) on its event loop, so the samples carry exactly the fields slime produces
               (tokens, logprobs, loss masks, rollout_routed_experts, rewards, metadata, weight versions).
               A whole group goes to one shard -> one engine (its prompt prefix stays in that engine's cache).
  * ``admission`` each shard caps its in-flight generate calls to its engine at ceil(INFLIGHT / shards per
               engine); further calls wait in the shard (not in sockets / router / the engine's tokenizer).
               INFLIGHT = ASRT_RS_INFLIGHT (per engine; default auto = --sglang-max-running-requests, else
               --sglang-server-concurrency).
  * ``auto``   (default mode) on only when the trajectory pool is large: worker pool samples
               (groups x n_samples_per_prompt) >= ASRT_RS_MIN (default 256: the
               group-affinity value shows already at a few hundred multi-turn trajectories - in one multi-turn
               benchmark slime's router lost the whole prefix on ~4-5% of later turns (+77% prefill), rshard 0.0%, and
               the step time went 189 -> 124 s) and the rollout uses slime's
               non-distributed fully-async worker; otherwise slime runs unchanged.  Fallbacks keep stock
               behaviour: shards not ready in time -> off; a shard process dies -> its groups run in the manager
               (stock path through the router) from then on.
  * ``profile`` every ASRT_RS_LOG_S (10) s one line in $ASRT_LOG_DIR/rshard.jsonl: client in-flight / waiting
               per engine, generate-call latency p50/p99 (slot wait and HTTP separately), output tokens/s per
               engine, and the engines' own #running / #queue from their /metrics.

Turn on (chains with slime_admit / slime_fill / slime_prefetch / slime_r3pack / slime_toolcache):
    --rollout-function-path ai_search_plugin.adapters.slime_rshard.generate_rollout_fully_async
    ASRT_RS=auto,profile  (or on,profile / off)  ASRT_RS_INNER=<the rollout function used before>
    ASRT_RS_SHARDS (total, default 2 x engines, max ASRT_RS_MAX_SHARDS=32)  ASRT_RS_INFLIGHT (per engine)
ASRT_RS unset/empty = slime unchanged.
"""
from __future__ import annotations

import asyncio
import collections
import contextvars
import copy
import heapq
import itertools
import json
import math
import os
import threading
import time

DEFAULT_INNER = "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
_ENV_PREFIXES = ("ASRT_", "AISEARCH_", "SLIME_", "HF_", "TOKENIZERS_", "TRANSFORMERS_")
_ENV_KEYS = ("PYTHONPATH", "HOME", "TMPDIR")


def opts() -> set:
    return {s.strip() for s in os.environ.get("ASRT_RS", "").split(",") if s.strip()}


def _mode() -> str:
    o = opts()
    if not o or "off" in o or "0" in o:
        return "off"
    if "measure" in o:  # stock slime + client/engine counters in rshard.jsonl (the A/B baseline arm)
        return "measure"
    if "on" in o or "1" in o or "shard" in o:
        return "on"
    return "auto"


def _log(rec: dict):
    d = os.environ.get("ASRT_LOG_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "rshard.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _pct(xs, q):
    try:
        s = sorted(list(xs))
    except RuntimeError:  # mutated by the event loop thread while copying
        return None
    if not s:
        return None
    return round(s[min(len(s) - 1, int(q * len(s)))], 4)


# ============================================================================================ client probe
_PRIO = contextvars.ContextVar("asrt_rs_prio", default=0)  # admission order of the group a request belongs to


class PrioSemaphore:
    """Counting semaphore whose waiters are served lowest priority value first (= oldest group first), FIFO
    within a priority.  Lets the groups the trainer needs next finish first when requests queue in the shard."""

    def __init__(self, n: int):
        self.free = int(n)
        self.heap = []
        self.seq = itertools.count()

    def waiting(self) -> int:
        return len(self.heap)

    async def acquire(self, prio=0):
        if self.free > 0 and not self.heap:
            self.free -= 1
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self.heap, (prio, next(self.seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():  # granted, but the waiter was cancelled: pass the slot on
                self.release()
            raise

    def release(self):
        while self.heap:
            fut = heapq.heappop(self.heap)[2]
            if not fut.done():
                fut.set_result(None)
                return
        self.free += 1


class ClientProbe:
    """Wraps slime.utils.http_utils._post (module global, used by every ``post``) for /generate calls: optional
    in-flight cap (asyncio semaphore), in-flight / waiting counts, slot-wait and HTTP latency, output tokens."""

    def __init__(self, cap=None, url=None):
        self.cap, self.url = cap, url
        self.sem = None
        self.inflight = self.waiting = self.calls = self.out_tok = self.errors = 0
        self.lat_http = collections.deque(maxlen=8192)
        self.lat_wait = collections.deque(maxlen=8192)
        self.win = {"calls": 0, "tok": 0, "t": time.time()}

    def install(self, hu):
        orig = hu._post
        if getattr(orig, "_asrt_rs", False):
            return
        me = self

        async def _post(client, url, payload, *a, **kw):
            if not str(url).endswith("/generate"):
                return await orig(client, url, payload, *a, **kw)
            if me.cap and me.sem is None:
                me.sem = PrioSemaphore(me.cap)
            t0 = time.monotonic()
            me.waiting += 1
            if me.sem is not None:
                try:
                    await me.sem.acquire(_PRIO.get())
                except BaseException:
                    me.waiting -= 1
                    raise
            me.waiting -= 1
            me.inflight += 1
            t1 = time.monotonic()
            try:
                out = await orig(client, url, payload, *a, **kw)
            except BaseException:
                me.errors += 1
                raise
            finally:
                me.inflight -= 1
                if me.sem is not None:
                    me.sem.release()
            t2 = time.monotonic()
            me.calls += 1
            me.win["calls"] += 1
            me.lat_wait.append(t1 - t0)
            me.lat_http.append(t2 - t1)
            try:
                mi = out.get("meta_info", {}) if isinstance(out, dict) else {}
                n = mi.get("completion_tokens")
                if n is None:
                    n = len(mi.get("output_token_logprobs") or ())
                me.out_tok += int(n)
                me.win["tok"] += int(n)
            except Exception:
                pass
            return out

        _post._asrt_rs = True
        hu._post = _post

    def stats(self) -> dict:
        now = time.time()
        dt = max(1e-6, now - self.win["t"])
        rec = {"url": self.url, "inflight": self.inflight, "waiting": self.waiting, "calls": self.calls,
               "errors": self.errors, "out_tok": self.out_tok,
               "calls_s": round(self.win["calls"] / dt, 2), "tok_s": round(self.win["tok"] / dt, 1),
               "http_p50": _pct(self.lat_http, 0.5), "http_p99": _pct(self.lat_http, 0.99),
               "wait_p50": _pct(self.lat_wait, 0.5), "wait_p99": _pct(self.lat_wait, 0.99)}
        self.win = {"calls": 0, "tok": 0, "t": now}
        self.lat_http.clear()
        self.lat_wait.clear()
        return rec


# ============================================================================================ shard (worker process)
class RShard:
    """One rollout client shard: runs slime's generate_and_rm_group for the groups it is given, on its own event
    loop, against ONE engine (its args point the "router" address at the engine)."""

    def __init__(self, args, idx: int, url: str, cap: int, conc: int, env: dict):
        os.environ.update(env or {})
        self.idx, self.url, self.cap, self.conc = idx, url, max(1, int(cap)), max(1, int(conc))
        host, port = _host_port(url)
        a = copy.copy(args)
        a.sglang_router_ip, a.sglang_router_port = (f"[{host}]" if ":" in host else host), port  # slime wraps IPv6
        a.use_distributed_post = False
        a.use_wandb = a.use_tensorboard = False
        self.args = a
        self.groups = 0
        self.probe = ClientProbe(self.cap, url)
        from slime.utils import http_utils as hu

        hu.init_http_client(a, concurrency=self.cap + 64)
        self.probe.install(hu)

    async def ready(self):
        import socket

        from slime.rollout.sglang_rollout import GenerateState

        GenerateState(self.args, concurrency=self.conc)  # singleton: tokenizer + generation semaphore of this shard
        return {"idx": self.idx, "pid": os.getpid(), "host": socket.gethostname(), "url": self.url, "cap": self.cap}

    async def run_group(self, group, sampling_params, evaluation=False, prio=0):
        from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group

        GenerateState(self.args, concurrency=self.conc)
        self.groups += 1
        _PRIO.set(prio)  # inherited by the per-sample tasks generate_and_rm_group creates
        return await generate_and_rm_group(self.args, group, sampling_params, evaluation=evaluation)

    async def stats(self):
        return {"shard": self.idx, "groups": self.groups, **self.probe.stats()}


def _host_port(url: str):
    from urllib.parse import urlparse

    u = urlparse(url if "//" in url else f"http://{url}")
    return u.hostname, int(u.port or 80)


# ============================================================================================ manager side
def _engine_urls(args) -> list:
    """Live engine URLs behind slime's router (GET /workers), in a stable order."""
    import requests

    r = requests.get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers", timeout=30)
    r.raise_for_status()
    urls = sorted({w["url"] for w in r.json().get("workers", []) if w.get("url")})
    return urls


def _scrape(url: str, timeout: float = 2.0) -> dict:
    """Engine #running / #queue / gen throughput from SGLang's Prometheus /metrics (max over TP ranks)."""
    import requests

    want = {"sglang:num_running_reqs": "running", "sglang:num_queue_reqs": "queue",
            "sglang:gen_throughput": "gen_tok_s", "sglang:token_usage": "kv_usage"}
    out = {}
    try:
        txt = requests.get(f"{url}/metrics", timeout=timeout).text
    except Exception as e:  # noqa: BLE001
        return {"err": repr(e)[:80]}
    for line in txt.splitlines():
        if not line.startswith("sglang:"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        k = want.get(name)
        if k is None:
            continue
        try:
            v = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        out[k] = max(out.get(k, v), v)
    return out


class ShardPool:
    def __init__(self, args, worker):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        self.args = args
        urls = _engine_urls(args)
        if not urls:
            raise RuntimeError("router lists no engines")
        E = len(urls)
        n = max(1, int(getattr(args, "n_samples_per_prompt", 1)))
        pool_samples = int(worker.concurrency) * n
        per = max(1, int(os.environ.get("ASRT_RS_PER_ENGINE", "2")))
        K = int(os.environ.get("ASRT_RS_SHARDS", "0") or 0) or per * E
        K = max(E, min(K, int(os.environ.get("ASRT_RS_MAX_SHARDS", "32"))))
        K = E * max(1, K // E)  # same number of shards per engine
        per = K // E
        inf = os.environ.get("ASRT_RS_INFLIGHT", "auto").strip().lower()
        if inf in ("", "auto"):
            mr = getattr(args, "sglang_max_running_requests", None)
            cap_e = int(mr) if mr else int(getattr(args, "sglang_server_concurrency", 512))
        else:
            cap_e = int(inf)
        cap_s = max(1, math.ceil(cap_e / per))
        conc_s = max(64, math.ceil(pool_samples / K) * 4)  # trajectories per shard: never the binding limit
        env = {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIXES) or k in _ENV_KEYS}
        nodes = {nd["NodeManagerAddress"]: nd["NodeID"] for nd in ray.nodes() if nd.get("Alive")}
        Actor = ray.remote(RShard)
        self.shards, self.urls, self.engine_of = [], urls, []

        def make(k, url):
            host = _host_port(url)[0]
            opt = dict(num_cpus=float(os.environ.get("ASRT_RS_CPUS", "0.01")), max_concurrency=1 << 20,
                       max_restarts=0, max_task_retries=0, runtime_env={"env_vars": env})
            if host in nodes:
                opt["scheduling_strategy"] = NodeAffinitySchedulingStrategy(nodes[host], soft=True)
            return Actor.options(**opt).remote(args, k, url, cap_s, conc_s, env)

        for k in range(K):
            e = k % E
            self.shards.append(make(k, urls[e]))
            self.engine_of.append(e)
        t = float(os.environ.get("ASRT_RS_READY_S", "300"))
        info = ray.get([s.ready.remote() for s in self.shards], timeout=t)
        self.load = [0] * K
        self.dead = set()
        self.local_groups = 0
        self.closing = False
        self.seq = 0
        self.prio = os.environ.get("ASRT_RS_PRIO", "1") != "0"  # oldest group's requests first in the shard queue
        self.cfg = {"engines": E, "shards": K, "per_engine": per, "inflight_per_engine": cap_e,
                    "inflight_per_shard": cap_s, "conc_per_shard": conc_s, "pool_samples": pool_samples,
                    "urls": urls, "shard_hosts": sorted({i["host"] for i in info})}
        self._stop = False
        if "profile" in opts():
            threading.Thread(target=self._monitor, name="asrt-rshard-mon", daemon=True).start()

    def close(self):
        """Worker shutdown: stop the shards so in-flight groups return at once (slime drains for 30 s otherwise)."""
        import ray

        self.closing = self._stop = True
        for a in self.shards:
            try:
                ray.kill(a, no_restart=True)
            except Exception:
                pass

    def pick(self):
        best, bl = None, None
        for k, l in enumerate(self.load):
            if k in self.dead:
                continue
            if bl is None or l < bl:
                best, bl = k, l
        return best

    def _monitor(self):
        import ray

        def client():
            live = [k for k in range(len(self.shards)) if k not in self.dead]
            return ray.get([self.shards[k].stats.remote() for k in live], timeout=30)

        def extra():
            return {"local_groups": self.local_groups, "dead": sorted(self.dead), "load_samples": list(self.load)}

        _monitor_loop(client, self.urls, extra, lambda: self._stop)


def _monitor_loop(client, urls, extra=None, stop=lambda: False):
    """Every ASRT_RS_LOG_S s: client-side stats (list of probe records, keyed by engine url or 'router') and the
    engines' own /metrics -> rshard.jsonl."""
    every = float(os.environ.get("ASRT_RS_LOG_S", "10"))
    while not stop():
        time.sleep(every)
        rec = {"t": time.time(), **(extra() if extra else {})}
        try:
            st = client()
        except Exception as e:  # noqa: BLE001
            st = []
            rec["stats_err"] = repr(e)[:200]
        eng = []
        for e, url in enumerate(urls):
            ss = [x for x in st if x.get("url") == url]
            eng.append({"engine": e, "client_inflight": sum(x["inflight"] for x in ss),
                        "client_waiting": sum(x["waiting"] for x in ss),
                        "calls_s": round(sum(x["calls_s"] for x in ss), 2),
                        "client_tok_s": round(sum(x["tok_s"] for x in ss), 1), **_scrape(url)})
        rec["engines"] = eng
        rec["client"] = st
        _log(rec)


_S = {"pool": None, "decided": False, "why": None, "orig_group": None}
_patched = False


def _n_samples(group) -> int:
    return sum(0 if isinstance(s, list) else 1 for s in group)


async def _sharded_group(args, group, sampling_params, evaluation=False):
    """Replacement for slime's generate_and_rm_group inside the fully-async worker loop."""
    pool = _S["pool"]
    orig = _S["orig_group"]
    if pool is None or evaluation:
        return await orig(args, group, sampling_params=sampling_params, evaluation=evaluation)
    k = pool.pick()
    if k is None:
        pool.local_groups += 1
        return await orig(args, group, sampling_params=sampling_params, evaluation=evaluation)
    n = _n_samples(group)
    pool.load[k] += n
    pool.seq += 1
    prio = pool.seq if pool.prio else 0
    try:
        out = await pool.shards[k].run_group.remote(group, sampling_params, evaluation, prio)
    except Exception as e:  # noqa: BLE001
        from ray.exceptions import RayActorError, RayTaskError

        _fill_done(n)
        if isinstance(e, RayTaskError):  # the generation itself failed: same as stock (task crashed)
            raise e.as_instanceof_cause()
        if not isinstance(e, RayActorError):
            raise
        if pool.closing:  # slime is shutting the worker down: hand the untouched input group back (requeued)
            return group
        pool.dead.add(k)  # shard process gone: its groups run in the manager (stock path) from now on
        _log({"t": time.time(), "event": "shard_dead", "shard": k, "err": repr(e)[:300]})
        pool.local_groups += 1
        _fill_done(-n)  # the stock path below counts its own samples (slime_fill wraps generate_and_rm)
        return await orig(args, group, sampling_params=sampling_params, evaluation=evaluation)
    finally:
        pool.load[k] -= n
    _fill_done(n)
    return out


def _fill_done(n: int):
    """slime_fill counts finished samples in generate_and_rm, which runs in the shard for sharded groups."""
    try:
        from . import slime_fill

        c = slime_fill._ctr
        if c is not None and "fill" in slime_fill.opts():
            with c.lock:
                c.s_done += n
    except Exception:
        pass


def _decide(worker):
    _S["decided"] = True
    m = _mode()
    args = worker.args
    n = max(1, int(getattr(args, "n_samples_per_prompt", 1)))
    pool_samples = int(worker.concurrency) * n
    why = None
    if m == "off":
        why = "ASRT_RS off"
    elif m == "measure":
        why = "measure only"
    elif m == "auto" and pool_samples < int(os.environ.get("ASRT_RS_MIN", "256")):
        why = f"auto: trajectory pool {pool_samples} < ASRT_RS_MIN {os.environ.get('ASRT_RS_MIN', '256')}"
    elif getattr(args, "rollout_external", False) or not getattr(args, "sglang_router_ip", None):
        why = "no slime router (external rollout)"
    elif getattr(args, "sglang_config", None):
        why = "multi-model --sglang-config"
    if why is None:
        t0 = time.time()
        try:
            _S["pool"] = ShardPool(args, worker)
        except Exception as e:  # noqa: BLE001
            why = f"shard start failed: {e!r}"[:400]
            _S["pool"] = None
        else:
            cfg = _S["pool"].cfg
            print(f"[asrt] slime_rshard ON: {cfg['shards']} shards ({cfg['per_engine']}/engine) x {cfg['engines']} "
                  f"engines, in-flight cap {cfg['inflight_per_engine']}/engine ({cfg['inflight_per_shard']}/shard), "
                  f"trajectory pool {pool_samples}, start {time.time() - t0:.1f}s, hosts {cfg['shard_hosts']}",
                  flush=True)
            _log({"t": time.time(), "event": "on", "start_s": round(time.time() - t0, 2), **cfg})
    if why is not None:
        _S["why"] = why
        print(f"[asrt] slime_rshard off: {why}", flush=True)
        _log({"t": time.time(), "event": "off", "why": why, "pool_samples": pool_samples})
        if "profile" in opts() or m == "measure":  # stock path: same counters, measured in the manager
            try:
                from slime.utils import http_utils as hu

                probe = ClientProbe(None, "router")
                probe.install(hu)
                urls = _engine_urls(args)

                threading.Thread(target=_monitor_loop, args=(lambda: [probe.stats()], urls), name="asrt-rshard-mon",
                                 daemon=True).start()
            except Exception as e:  # noqa: BLE001
                print(f"[asrt] slime_rshard measure failed: {e!r}", flush=True)


def install(args=None) -> bool:
    """Patch slime's fully-async worker: generate_and_rm_group (looked up in the module at call time) and
    AsyncRolloutWorker.start (decide + start the shards before the first group is admitted)."""
    global _patched
    if _patched or _mode() == "off":
        return _patched
    from slime.rollout import fully_async_rollout as far

    _S["orig_group"] = far.generate_and_rm_group
    far.generate_and_rm_group = _sharded_group
    W = far.AsyncRolloutWorker
    orig_start = W.start

    def start(self):
        if not _S["decided"]:
            _decide(self)
        return orig_start(self)

    orig_close = W.close

    def close(self):
        pool = _S["pool"]
        if pool is not None:
            pool.close()
        return orig_close(self)

    W.start, W.close = start, close
    _patched = True
    return True


_inner_fn = None


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """``--rollout-function-path`` entry: installs the patches, then runs ASRT_RS_INNER."""
    global _inner_fn
    if _mode() != "off":
        install(args)
    if _inner_fn is None:
        from slime.utils.misc import load_function

        _inner_fn = load_function(os.environ.get("ASRT_RS_INNER") or DEFAULT_INNER)
    return _inner_fn(args, rollout_id, data_buffer, evaluation=evaluation)


# The RolloutManager imports the rollout function in its __init__ (load_function): install before the worker exists.
if _mode() != "off":
    try:
        install(None)
    except Exception as _e:  # never break the framework import
        print(f"[asrt] slime_rshard: patch at import failed: {_e!r}", flush=True)
