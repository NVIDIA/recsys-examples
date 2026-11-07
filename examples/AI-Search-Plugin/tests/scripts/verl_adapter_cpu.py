"""CPU test of ai_search_plugin.adapters.verl against fake verl modules with the same names and signatures as verl
main (2026-10). Inside a verl container (real verl importable: VERL_REAL=1) the balance part runs against the real
verl.utils.seqlen_balancing.  (Expert placement has its own tests: cpu/test_verl_placement_cpu.py.)
Run: python3 tests/scripts/verl_adapter_cpu.py"""
import asyncio
import json
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
T = types.SimpleNamespace(E=8)  # toy MoE size used by the fake engine below

REAL = os.environ.get("VERL_REAL") == "1"


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def fake_verl():
    """Register a fake 'verl' package whose submodules are created lazily on
    import (so the adapter's post-import hooks fire like with real verl)."""
    import importlib.abc
    import importlib.machinery

    defs = {}

    def kk(seqlen_list, k_partitions, equal_size):  # round-robin: deliberately poor
        order = sorted(range(len(seqlen_list)), key=lambda i: -seqlen_list[i])
        parts = [[] for _ in range(k_partitions)]
        for j, i in enumerate(order):
            parts[j % k_partitions].append(i)
        return [sorted(p) for p in parts]

    def calculate_workload(x):
        return 24576 * x + x ** 2
    defs["verl.utils.seqlen_balancing"] = dict(get_seqlen_balanced_partitions=kk, calculate_workload=calculate_workload)
    defs["verl.trainer.ppo.ray_trainer"] = dict(get_seqlen_balanced_partitions=kk, calculate_workload=calculate_workload)

    class ToolResponse:
        def __init__(self, text=None, image=None, video=None):
            self.text, self.image, self.video = text, image, video
    defs["verl.tools.schemas"] = dict(ToolResponse=ToolResponse)

    calls = []

    class ToolAgentLoop:
        async def _call_tool(self, tool_call, tools_kwargs, agent_data):
            calls.append(tool_call.name)
            await asyncio.sleep(0.01)
            q = json.loads(tool_call.arguments).get("query", "")
            return ToolResponse(text=f"docs:{q.lower()}"), 0.0, {"n": 1}
    defs["verl.experimental.agent_loop.tool_agent_loop"] = dict(ToolAgentLoop=ToolAgentLoop)

    class RouterReplay:
        router_instances = []

        def __init__(self):
            self.target, self.recorded = None, None
            RouterReplay.router_instances.append(self)

        def set_target_indices(self, topk_indices, replay_mask=None):
            self.target = topk_indices

        def record_indices(self, topk_indices):
            self.recorded = topk_indices
    defs["verl.utils.megatron.router_replay_patch"] = dict(RouterReplay=RouterReplay)

    class BaseEngine:
        def train_batch(self, data, loss_function):
            self.trained = getattr(self, "trained", 0) + 1
            return {"metrics": {}}

    class MegatronEngine(BaseEngine):
        def __init__(self, model, opt, r3=True):
            self.module, self.optimizer = [model], opt
            self.tf_config = types.SimpleNamespace(num_layers=2, num_moe_experts=T.E)
            self.engine_config = types.SimpleNamespace(
                forward_only=False, router_replay=types.SimpleNamespace(mode="R3" if r3 else "disabled"))

        def initialize(self):
            pass

        def get_per_tensor_param(self, base_sync_done=False, **kw):
            m = self.module[0]
            out = []
            for li, layer in enumerate(m.layers):
                out.append((f"model.layers.{li}.mlp.gate.weight", layer.mlp.router.weight.detach().clone()))
                for s, ex in enumerate(layer.mlp.experts.local_experts):
                    out.append((f"model.layers.{li}.mlp.experts.{s}.up_proj.weight", ex.fc1.detach().clone()))
            return iter(out), None

        def save_checkpoint(self, *a, **k):
            self.saved = {n: p.detach().clone() for n, p in self.module[0].named_parameters()}
    defs["verl.workers.engine.megatron.transformer_impl"] = dict(MegatronEngine=MegatronEngine)

    class TrainingWorker:
        def __init__(self, engine):
            self.engine = engine

        def train_mini_batch(self, data):
            return self.engine.train_batch(data, None)
    defs["verl.workers.engine_workers"] = dict(TrainingWorker=TrainingWorker)

    class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name == "verl" or name.startswith("verl."):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__path__ = []
            module.__dict__.update(defs.get(module.__name__, {}))
    sys.meta_path.append(Finder())
    return calls, RouterReplay


def test_balance():
    from ai_search_plugin.adapters import verl as V
    V.install_balanced_partitions(mode="best")
    import importlib
    sb = importlib.import_module("verl.utils.seqlen_balancing")
    assert getattr(sb.get_seqlen_balanced_partitions, "__asrt_patched__", False), "partitioner not patched"
    if not REAL:
        rt = importlib.import_module("verl.trainer.ppo.ray_trainer")
        assert rt.get_seqlen_balanced_partitions is sb.get_seqlen_balanced_partitions, "importer not patched"
    g = torch.Generator().manual_seed(0)
    lens = (torch.randint(200, 30000, (64,), generator=g)).tolist()
    w = [24576 * x + x * x for x in lens]
    for eq in (True, False):
        parts = sb.get_seqlen_balanced_partitions(w, 8, eq)
        assert sorted(i for p in parts for i in p) == list(range(64)) and all(parts)
        if eq:
            assert all(len(p) == 8 for p in parts)
    print("balance OK", V.stats())


def test_toolcache(calls):
    from ai_search_plugin.adapters import verl as V
    os.environ["ASRT_VERL_TOOLCACHE_TOOLS"] = "search"
    V.install_tool_cache()
    import importlib
    tal = importlib.import_module("verl.experimental.agent_loop.tool_agent_loop")
    loop = tal.ToolAgentLoop()

    def tc(name, q):
        return types.SimpleNamespace(name=name, arguments=json.dumps({"query": q}))

    async def main():
        rs = await asyncio.gather(*[loop._call_tool(tc("search", "Paris  France"), {}, None) for _ in range(8)],
                                  loop._call_tool(tc("search", "paris france"), {}, None),
                                  loop._call_tool(tc("search", "berlin"), {}, None),
                                  loop._call_tool(tc("code", "x"), {}, None))
        again = await loop._call_tool(tc("search", "PARIS FRANCE"), {}, None)
        return rs, again
    rs, again = asyncio.run(main())
    assert rs[0][0].text == "docs:paris  france" and rs[8][0].text == rs[0][0].text and again[0].text == rs[0][0].text
    assert rs[0][2] == {"n": 1}
    assert calls.count("search") == 2 and calls.count("code") == 1, calls
    st = V.stats()["toolcache"]
    assert st["misses"] == 2 and st["hits"] == 1 and st["joins"] == 8, st
    print("toolcache OK", st)



def test_full_expand():
    from ai_search_plugin.adapters import verl as V
    os.environ.pop("ASRT_VERL_DROP", None)
    assert V.expand("full") == V.FULL
    assert V.expand("profile,full,route") == V.FULL
    os.environ["ASRT_VERL_DROP"] = "route,sglhooks"
    assert V.expand("full") == [f for f in V.FULL if f not in ("route", "sglhooks")]
    assert V.expand("off") == [] and V.expand("") == []
    os.environ.pop("ASRT_VERL_DROP")
    print("full expand OK", V.FULL)


def test_rlogits_guard():
    """response_logits only where answers are a small part of the rows (else verl's own forward)."""
    import types
    import ai_search_plugin
    rl = types.ModuleType("ai_search_plugin.response_logits"); rl.__path__ = []
    ad = types.ModuleType("ai_search_plugin.response_logits.adapters"); ad.__path__ = []
    rv = types.ModuleType("ai_search_plugin.response_logits.adapters.verl")
    rl.adapters, ad.verl = ad, rv
    calls = []

    def _wrap_forward_fn(fn):
        def w(*a, **k):
            calls.append("compact")
            return fn(*a, **k)
        return w
    rv._wrap_forward_fn = _wrap_forward_fn
    names = ("ai_search_plugin.response_logits", "ai_search_plugin.response_logits.adapters",
             "ai_search_plugin.response_logits.adapters.verl")
    saved = {k: sys.modules.get(k) for k in names}
    saved_attr = getattr(ai_search_plugin, "response_logits", None)
    sys.modules.update(dict(zip(names, (rl, ad, rv))))
    ai_search_plugin.response_logits = rl
    try:
        from ai_search_plugin.adapters import verl as V
        assert V._guard_response_logits()

        def fwd(model, input_ids, logits_processor=None, logits_processor_args=None):
            calls.append("stock")

        def nt(lens):
            return torch.nested.nested_tensor_from_jagged(torch.ones(sum(lens)), offsets=torch.tensor([0] + list(torch.tensor(lens).cumsum(0))))
        g = rv._wrap_forward_fn(fwd)
        g(None, nt([100, 100]), logits_processor=1, logits_processor_args={"loss_mask": nt([10, 20])})   # 15% -> compact
        g(None, nt([100, 100]), logits_processor=1, logits_processor_args={"loss_mask": nt([90, 95])})   # 92% -> stock
        assert calls == ["compact", "stock", "stock"] and rv._wrap_forward_fn(fwd) is g, calls
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if saved_attr is None:
            ai_search_plugin.__dict__.pop("response_logits", None)
        else:
            ai_search_plugin.response_logits = saved_attr
    print("rlogits guard OK")


def main():
    test_full_expand()
    test_rlogits_guard()
    calls, RR = (None, None)
    if not REAL:
        calls, RR = fake_verl()
    test_balance()
    if not REAL:
        test_toolcache(calls)
    print("verl adapter cpu test OK")


if __name__ == "__main__":
    main()
