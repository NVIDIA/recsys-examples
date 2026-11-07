"""CPU test of ai_search_plugin.adapters.verl_rollout / verl_route (features route, keepkv,
tokcache, profile) against fake verl modules with the same names/signatures as
verl main of 2026-10.
Run: python3 tests/scripts/verl_rollout_cpu.py"""
import asyncio
import functools
import inspect
import os
import random
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
LOG = tempfile.mkdtemp()
os.environ["ASRT_LOG_DIR"] = LOG


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


# ---------------------------------------------------------------- fake verl
for p in ["verl", "verl.workers", "verl.workers.rollout", "verl.workers.rollout.sglang_rollout", "verl.utils",
          "verl.experimental", "verl.experimental.agent_loop", "verl.checkpoint_engine"]:
    _mod(p).__path__ = []


def auto_await(func):
    @functools.wraps(func)
    def wrapper(*a, **k):
        coro = func(*a, **k)
        if not inspect.iscoroutine(coro):
            return coro
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return coro
    return wrapper


_mod("verl.utils.ray_utils", auto_await=auto_await)


class GlobalRequestLoadBalancer:  # verl router.py (sticky + least in-flight)
    def __init__(self, servers, max_cache_size=10000, full_determinism=False):
        self._servers = dict(servers)
        self._inflight_requests = {s: 0 for s in servers}
        self._request_id_to_server = {}

    def acquire_server(self, request_id):
        if request_id in self._request_id_to_server:
            s = self._request_id_to_server[request_id]
            self._inflight_requests[s] += 1
            return s, self._servers[s]
        m = min(self._inflight_requests.values())
        s = random.choice([k for k, v in self._inflight_requests.items() if v == m])
        self._request_id_to_server[request_id] = s
        self._inflight_requests[s] += 1
        return s, self._servers[s]

    def release_server(self, server_id, request_id=None):
        self._inflight_requests[server_id] -= 1

    def require_acquire_fields(self):
        return []

    def require_release_fields(self):
        return []

    def get_status(self):
        return {"servers": dict(self._inflight_requests)}


CREATED = {}


def get_router_handle(servers, router_config_path=None, full_determinism=False, load_balancer_cls=None):
    cls = load_balancer_cls or GlobalRequestLoadBalancer
    lb = cls(servers=servers, max_cache_size=10000)
    CREATED["lb"] = lb
    return lb


_mod("verl.workers.rollout.router", GlobalRequestLoadBalancer=GlobalRequestLoadBalancer,
     get_router_handle=get_router_handle)


class LLMServerClient:
    def __init__(self, lb):
        self._load_balancer = lb
        self._lb_require_acquire_fields = None

    async def _acquire_server(self, request_id, **extra):
        if self._lb_require_acquire_fields is None:
            self._lb_require_acquire_fields = list(self._load_balancer.require_acquire_fields())
        fields = {n: extra[n] for n in self._lb_require_acquire_fields if n in extra}
        return self._load_balancer.acquire_server(request_id=request_id, **fields)

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
        sid, _ = await self._acquire_server(request_id, prompt_ids=prompt_ids, sampling_params=sampling_params)
        return sid


_mod("verl.workers.rollout.llm_server", LLMServerClient=LLMServerClient)

EVENTS = []


class SGLangHttpServer:
    async def release_kv_cache(self):
        EVENTS.append("release")

    async def resume_kv_cache(self):
        EVENTS.append("resume+flush")


_mod("verl.workers.rollout.sglang_rollout.async_sglang_server", SGLangHttpServer=SGLangHttpServer)


class Req:
    def __init__(self):
        self.flush_cache = True


class AsyncHttpServerAdapter:
    async def flush_cache(self):
        EVENTS.append("flush")
        return {"ok": True}

    async def update_weights_from_tensor(self, req):
        EVENTS.append(f"bucket flush={req.flush_cache}")


_mod("verl.workers.rollout.sglang_rollout.http_server_engine", AsyncHttpServerAdapter=AsyncHttpServerAdapter)


class ServerAdapter:
    def __init__(self):
        self._engine = AsyncHttpServerAdapter()

    async def update_weights(self, weights, global_steps=None, **kw):
        for _ in range(2):
            await self._engine.update_weights_from_tensor(Req())
        await self._engine.flush_cache()


_mod("verl.workers.rollout.sglang_rollout.sglang_rollout", ServerAdapter=ServerAdapter)

TOK_CALLS = []


class AgentLoopBase:
    async def ct_build_initial_tokens(self, messages, tools=None, images=None, videos=None, audios=None,
                                      mm_inputs_out=None):
        TOK_CALLS.append(1)
        await asyncio.sleep(0.01)
        return [len(m["content"]) for m in messages]


_mod("verl.experimental.agent_loop.agent_loop", AgentLoopBase=AgentLoopBase)


class CheckpointEngineManager:
    @auto_await
    async def abort_replicas(self):
        await asyncio.sleep(0.01)

    @auto_await
    async def release_kv_cache_replicas(self):
        await asyncio.sleep(0.02)

    def build_process_group(self, rollout):
        import time
        time.sleep(0.01)

    @auto_await
    async def resume_kv_cache_replicas(self):
        await asyncio.sleep(0.01)

    @auto_await
    async def resume_generation_replicas(self):
        await asyncio.sleep(0.01)

    @auto_await
    async def update_weights(self, global_steps=None):
        await self.abort_replicas()
        await self.release_kv_cache_replicas()
        self.build_process_group(None)
        await asyncio.sleep(0.03)  # transfer
        await self.resume_kv_cache_replicas()
        await self.resume_generation_replicas()
        return {"ok": 1}


_mod("verl.checkpoint_engine.base", CheckpointEngineManager=CheckpointEngineManager)

# ---------------------------------------------------------------- tests
from ai_search_plugin.adapters import verl as V  # noqa: E402
from ai_search_plugin.adapters import verl_rollout as R  # noqa: E402


def test_pick():
    loads = {"a": 3, "b": 1, "c": 1}
    # 35k-token prompt, affinity on a: penalty 8.5 req for b/c -> stay on a
    assert R.affinity_pick(loads, "a", 35000, 4096) == "a"
    # short prompt: least loaded wins
    assert R.affinity_pick(loads, "a", 300, 4096) in ("b", "c")
    # heavy imbalance spills
    assert R.affinity_pick({"a": 20, "b": 1}, "a", 35000, 4096) == "b"
    print("pick ok")


def test_route():
    V.install("route")
    servers = {f"s{i}": object() for i in range(4)}
    import verl.workers.rollout.router as router
    lb = router.get_router_handle(servers)
    assert type(lb).__name__ == "AsrtAffinityBalancer", type(lb)
    cl = LLMServerClient(lb)
    rnd = random.Random(0)
    prompts = [[rnd.randrange(1000) for _ in range(30000)] for _ in range(16)]

    async def run():
        out = {}
        tasks = []
        for g, p in enumerate(prompts):
            for i in range(4):
                tasks.append((g, cl.generate(f"r{g}-{i}", prompt_ids=p, sampling_params={})))
        for g, t in tasks:
            out.setdefault(g, set()).add(await t)
        return out
    res = asyncio.run(run())
    same = sum(1 for v in res.values() if len(v) == 1)
    loads = lb._inflight_requests
    print("route: groups on one server", same, "/16; loads", loads, lb._asrt.st)
    assert same >= 14, res
    assert max(loads.values()) - min(loads.values()) <= 12
    # sticky multi-turn
    s1 = asyncio.run(cl.generate("r0-0", prompt_ids=prompts[5], sampling_params={}))
    assert s1 in res[0]
    # baseline verl balancer for comparison
    base = GlobalRequestLoadBalancer(servers)
    cb = LLMServerClient(base)

    async def run_b():
        out = {}
        for g, p in enumerate(prompts):
            for i in range(4):
                out.setdefault(g, set()).add(await cb.generate(f"q{g}-{i}", prompt_ids=p, sampling_params={}))
        return out
    rb = asyncio.run(run_b())
    print("plain verl: groups on one server", sum(1 for v in rb.values() if len(v) == 1), "/16")


def test_keepkv():
    V.install("keepkv")
    EVENTS.clear()

    async def run():
        s = SGLangHttpServer()
        await s.release_kv_cache()
        await ServerAdapter().update_weights(None, global_steps=1)
        await s.resume_kv_cache()
        await AsyncHttpServerAdapter().flush_cache()  # outside a weight update: still flushes
    asyncio.run(run())
    print("keepkv events", EVENTS, R.stats()["keepkv"])
    assert EVENTS == ["bucket flush=False", "bucket flush=False", "flush"], EVENTS


def test_tokcache_profile():
    V.install("tokcache,profile")
    TOK_CALLS.clear()

    async def run():
        a = AgentLoopBase()
        msgs = [{"role": "user", "content": "x" * 100}]
        outs = await asyncio.gather(*[a.ct_build_initial_tokens(msgs) for _ in range(4)])
        outs.append(await a.ct_build_initial_tokens(msgs))
        outs.append(await a.ct_build_initial_tokens([{"role": "user", "content": "y"}]))
        return outs
    outs = asyncio.run(run())
    assert outs[0] == outs[4] == [100] and outs[5] == [1], outs
    assert len(TOK_CALLS) == 2, TOK_CALLS
    outs[0].append(5)
    assert outs[1] == [100]  # copies, not shared lists
    m = CheckpointEngineManager()
    assert m.update_weights(3) == {"ok": 1}
    import glob
    import json
    recs = [json.loads(l) for f in glob.glob(f"{LOG}/verl-sync-*.jsonl") for l in open(f)]
    print("tokcache", R.stats()["tokcache"], "sync record", recs[-1])
    r = recs[-1]
    assert r["step"] == 3 and r["release_kv_cache_replicas"] >= 0.015 and r["transfer_finalize_s"] >= 0.02, r


if __name__ == "__main__":
    test_pick()
    test_route()
    test_keepkv()
    test_tokcache_profile()
    print("ALL OK")
