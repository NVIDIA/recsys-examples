# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""verl glue of expert placement (adapters.verl_placement, feature ``ASRT_VERL=place``) against fake verl modules
with the names and signatures of verl's Megatron engine: R3 routes of the mini-batch drive a re-placement, replayed
ids are mapped logical -> physical and recorded ids physical -> logical, the HF-named export comes out in logical
order (with its self-check), checkpoints are saved in logical order, balanced batches are skipped with probe
back-off, and a dense model gets no manager."""
import importlib
import importlib.abc
import importlib.machinery
import io
import sys
import types
from contextlib import redirect_stdout

import pytest

torch = pytest.importorskip("torch")
import placement_toy as T  # noqa: E402


@pytest.fixture
def env(monkeypatch):
    for k in ("ASRT_PLACE_RANDOM", "ASRT_PLACE_LOG", "ASRT_PLACE_EVERY", "ASRT_PLACE_VERIFY", "ASRT_PLACE_VERIFY_FWD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ASRT_PLACE_STRIDE", "1")
    monkeypatch.setenv("ASRT_PLACE_THRESH", "1.01")
    monkeypatch.setenv("ASRT_PLACE_SELFCHECK", "raise")
    monkeypatch.setenv("ASRT_PLACE_MIN_IMB", "1.15")
    cleanup = T.fake_env(monkeypatch)
    finder = _fake_verl()
    sys.meta_path.append(finder)
    yield monkeypatch
    sys.meta_path.remove(finder)
    cleanup()


def _fake_verl():
    """A fake 'verl' package whose submodules are created on import (so the post-import hooks fire like with
    real verl)."""
    defs = {}

    class RouterReplay:
        def __init__(self):
            self.target, self.recorded = None, None

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
    return Finder()


def _setup(r3=True):
    from ai_search_plugin.adapters import verl_placement as VP
    VP.install()
    ti = importlib.import_module("verl.workers.engine.megatron.transformer_impl")
    ew = importlib.import_module("verl.workers.engine_workers")
    rrm = importlib.import_module("verl.utils.megatron.router_replay_patch")
    m, opt, x = T.trained_model(n=256)
    for layer in m.layers:  # verl attaches a RouterReplay to each TopKRouter
        layer.mlp.router.router_replay = rrm.RouterReplay()
        layer.mlp.router.layer_number = layer.layer_number
    eng = ti.MegatronEngine(m, opt, r3=r3)
    with redirect_stdout(io.StringIO()):
        eng.initialize()
    mgr = eng._asrt_mgr
    assert mgr is not None and mgr._asrt_r3 == r3
    mgr.ep, mgr.n_local = 4, 4  # pretend EP4 (the toy model runs EP world 1)
    return VP, m, x, eng, mgr, ew.TrainingWorker(eng)


def test_r3_replace_replay_export_save(env):
    VP, m, x, eng, mgr, w = _setup()
    y0 = m(x).detach()
    snap = {n: p.detach().clone() for n, p in m.named_parameters()}
    # R3 routes of this batch: hot logical experts 0..3, padded batch [bs, seq, L, topk]
    routes = torch.cat([torch.randint(0, 4, (4, 50, 2, T.K)), torch.randint(0, T.E, (4, 20, 2, T.K))], 1)
    routes[:, -5:] = 0  # padding rows
    with redirect_stdout(io.StringIO()):
        w.train_mini_batch({"routed_experts": routes})
    assert eng.trained == 1 and not mgr.is_identity(), "hot experts should have moved"
    assert torch.allclose(m(x).detach(), y0, atol=1e-3)
    # replay: logical -> physical; record: physical -> logical
    rr0 = m.layers[0].mlp.router.router_replay
    ids = routes[0, :10, 0]
    rr0.set_target_indices(ids)
    assert torch.equal(rr0.target, mgr.slot_of[0][ids])
    rr0.record_indices(mgr.slot_of[0][ids])
    assert torch.equal(rr0.recorded, ids)
    # export: logical names + router rows in logical order (HF names)
    gen, _ = eng.get_per_tensor_param()
    with redirect_stdout(io.StringIO()):
        out = dict(gen)
    assert torch.equal(out["model.layers.0.mlp.gate.weight"], snap["layers.0.mlp.router.weight"])
    for e in range(T.E):
        assert torch.equal(out[f"model.layers.0.mlp.experts.{e}.up_proj.weight"],
                           snap[f"layers.0.mlp.experts.local_experts.{e}.fc1"]), e
    # checkpoint in logical order, placement re-applied after
    before = mgr.slot_of.clone()
    eng.save_checkpoint("x")
    for n, p in eng.saved.items():
        assert torch.equal(p, snap[n]), n
    assert torch.equal(mgr.slot_of, before)
    assert torch.allclose(m(x).detach(), y0, atol=1e-3)


def test_auto_off_probe_backoff(env):
    env.setenv("ASRT_PLACE_PROBE_EVERY", "3")
    VP, m, x, eng, mgr, w = _setup()
    uniform = torch.randint(1, T.E, (4, 4000, 2, T.K))
    with redirect_stdout(io.StringIO()):
        for _ in range(4):
            w.train_mini_batch({"routed_experts": uniform})
    # step 0 checks and skips, steps 1-2 back off (no counting, no collective), step 3 probes again
    assert len(mgr.history) == 2 and all(h.get("skipped") for h in mgr.history), mgr.history
    assert mgr.is_identity() and eng.trained == 4


def test_export_selfcheck(env):
    VP, m, x, eng, mgr, w = _setup()
    g = torch.Generator().manual_seed(3)
    with redirect_stdout(io.StringIO()):
        mgr.maybe_rebalance(force_perms={lid: torch.randperm(T.E, generator=g).tolist() for lid in range(2)})
        full = [(f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight", torch.zeros(1))
                for l in (0, 1) for e in range(T.E)]
        out = list(VP._checked_export(mgr, iter(full)))
    assert len(out) == len(full)
    logical = {int(n.split(".experts.")[1].split(".")[0]) for n, _ in out if ".layers.0." in n}
    assert logical == set(range(T.E))
    with pytest.raises(RuntimeError, match="export covered MoE layers"):
        with redirect_stdout(io.StringIO()):
            list(VP._checked_export(mgr, iter(full[: T.E])))   # layer 1 never exported under a matching name


def test_dense_model_no_manager(env):
    from ai_search_plugin.adapters import verl_placement as VP
    m = T.Dense()
    eng = types.SimpleNamespace(module=[m], optimizer=None, tf_config=types.SimpleNamespace(num_layers=2))
    with redirect_stdout(io.StringIO()):
        assert VP.build_manager(eng) is None


def test_verl_feature_place_installs(env):
    from ai_search_plugin.adapters import verl as V
    env.setattr(V, "_INSTALLED", set())
    V._install_one("place")
    assert "place" in V._INSTALLED
    from ai_search_plugin.adapters import verl_placement as VP
    assert VP._S["installed"]
    for mod in ("verl.utils.megatron.router_replay_patch", "verl.workers.engine.megatron.transformer_impl",
                "verl.workers.engine_workers"):   # fire (and so consume) the post-import hooks it registered
        importlib.import_module(mod)
    ti = sys.modules["verl.workers.engine.megatron.transformer_impl"]
    assert getattr(ti.MegatronEngine.train_batch, "__asrt_patched__", False)
