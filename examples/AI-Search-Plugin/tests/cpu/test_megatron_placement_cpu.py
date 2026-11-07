# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert placement manager (adapters.megatron_placement) on a toy MoE model, single process, no Megatron:
renaming keeps the layer function identical, optimizer state follows the weights, cumulative placement maps
(R3 ids, weight export, restore) are consistent, the planner balances a hot cluster, and the auto-off path skips
balanced batches.  Plus: the placement modules import without torch."""
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
import placement_toy as T  # noqa: E402

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("ASRT_PLACE_MIN_IMB", raising=False)
    cleanup = T.fake_env(monkeypatch)
    yield
    cleanup()


def _mgr(m, opt):
    from ai_search_plugin.adapters.megatron_placement import ExpertPlacementManager
    mgr = ExpertPlacementManager([m], opt, num_layers=2, num_experts=T.E, ep_group=None)
    mgr.ep, mgr.n_local = 4, 4   # pretend EP4 for the planner; moves are local (world 1)
    return mgr


def test_rename_equivalence_and_maps(env):
    m, opt, x = T.trained_model(steps=2)
    y0 = m(x).detach()
    snap = {n: p.detach().clone() for n, p in m.named_parameters()}
    st0 = {n: opt.state[p]["exp_avg"].clone() for n, p in m.named_parameters()}
    mgr = _mgr(m, opt)
    g = torch.Generator().manual_seed(1)
    mgr.maybe_rebalance(force_perms={lid: torch.randperm(T.E, generator=g).tolist() for lid in range(2)})
    assert torch.allclose(m(x).detach(), y0, atol=1e-3)
    # optimizer state followed: expert at its new slot has the old moments of its logical expert
    for lid in range(2):
        for e in range(T.E):
            s = int(mgr.slot_of[lid][e])
            p = m.layers[lid].mlp.experts.local_experts[s].fc1
            assert torch.equal(opt.state[p]["exp_avg"], st0[f"layers.{lid}.mlp.experts.local_experts.{e}.fc1"])
        r = m.layers[lid].mlp.router.weight     # router Adam rows permuted too
        assert torch.equal(opt.state[r]["exp_avg"][mgr.slot_of[lid]], st0[f"layers.{lid}.mlp.router.weight"])
    # R3 ids: logical -> physical
    ids = torch.tensor([[0, 3], [5, 7]])
    assert torch.equal(mgr.map_routed_ids(0, ids), mgr.slot_of[0][ids])
    # export: names / router rows back to logical order
    _, w = mgr.export_name_and_tensor("module.decoder.layers.0.mlp.router.weight",
                                      m.layers[0].mlp.router.weight.detach())
    assert torch.equal(w, snap["layers.0.mlp.router.weight"])
    s = int(mgr.slot_of[1][5])
    n2, _ = mgr.export_name_and_tensor(f"module.decoder.layers.1.mlp.experts.linear_fc1.weight{s}", None)
    assert n2.endswith("weight5"), n2
    # a second permutation composes; then restoring identity is bit exact
    mgr.maybe_rebalance(force_perms={0: torch.randperm(T.E, generator=g).tolist()})
    assert torch.allclose(m(x).detach(), y0, atol=1e-3)
    mgr.restore_identity()
    assert mgr.is_identity()
    for n, p in m.named_parameters():
        assert torch.equal(p.detach(), snap[n]), n


def test_planner_balances_hot_cluster(env):
    m, opt, x = T.trained_model()
    y0 = m(x).detach()
    mgr = _mgr(m, opt)
    hot = torch.zeros(2, T.E, dtype=torch.float64)
    hot[:, :4] = 100
    hot[:, 4:] = 10
    mgr.counts += hot
    info = mgr.maybe_rebalance()
    assert info["imbalance_before"] > 1.5 and info["imbalance_after_pred"] < 1.2, info
    assert info["experts_moved"] > 0 and not mgr.is_identity()
    assert torch.allclose(m(x).detach(), y0, atol=1e-3)
    assert float(mgr.counts.abs().sum()) == 0.0          # counts cleared after a step


def test_auto_off_when_balanced(env, monkeypatch):
    m, opt, _ = T.trained_model()
    mgr = _mgr(m, opt)
    mgr.counts += 10.0                                    # perfectly balanced
    info = mgr.maybe_rebalance()
    assert info.get("skipped") and info["experts_moved"] == 0 and mgr.is_identity()
    # mildly unbalanced, below the default 1.15: still off
    mgr.counts[:, :4] = 12.0
    mgr.counts[:, 4:] = 10.0
    assert mgr.maybe_rebalance().get("skipped")
    # same batch with the threshold lowered: moves
    monkeypatch.setenv("ASRT_PLACE_MIN_IMB", "1.01")
    mgr.counts[:, :4] = 12.0
    mgr.counts[:, 4:] = 10.0
    info = mgr.maybe_rebalance()
    assert not info.get("skipped") and info["experts_moved"] > 0


def test_r3_counting(env):
    m, opt, _ = T.trained_model()
    mgr = _mgr(m, opt)
    routes = torch.tensor([[[0, 1], [2, 3]], [[0, 5], [2, 7]]])   # [tokens=2, layers=2, topk=2]
    mgr.add_logical_ids(routes)
    assert mgr.counts[0].tolist()[:6] == [2, 1, 0, 0, 0, 1]
    assert mgr.counts[1].tolist()[:8] == [0, 0, 2, 1, 0, 0, 0, 1]


def test_place_verify_flags_corruption(env, monkeypatch):
    monkeypatch.setenv("ASRT_PLACE_VERIFY", "1")
    from ai_search_plugin.adapters import place_verify as V
    logs = []
    monkeypatch.setattr(V, "_log", lambda rec: logs.append(rec))
    m, opt, x = T.trained_model(lr=1e-6)

    def step():
        m(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    mgr = _mgr(m, opt)
    g = torch.Generator().manual_seed(1)
    pre = V.before_step(mgr, 0)
    mgr.maybe_rebalance(force_perms={lid: torch.randperm(T.E, generator=g).tolist() for lid in range(2)})
    V.after_move(mgr, 0, pre)
    mv = logs[-1]
    assert mv["phase"] == "post-move" and mv["worst"] < 1e-9, mv
    step()
    V.before_step(mgr, 1)
    assert logs[-1]["worst"] < 1e-3, logs[-1]
    # corruption: swap the weights of slots 0 and 1 in layer 0 without telling the manager
    e = m.layers[0].mlp.experts.local_experts
    with torch.no_grad():
        a = e[0].fc1.detach().clone()
        e[0].fc1.copy_(e[1].fc1)
        e[1].fc1.copy_(a)
    V.before_step(mgr, 2)
    assert logs[-1]["drift"]["w"] > 0.1, logs[-1]


def test_modules_import_without_torch():
    """No torch / numpy / megatron / slime / verl at import time (the registry imports every module)."""
    code = textwrap.dedent("""
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("torch", "numpy", "megatron", "slime", "verl"):
                    raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Block())
        import ai_search_plugin.placement_py as P
        import ai_search_plugin.adapters.megatron_placement
        import ai_search_plugin.adapters.slime_placement
        import ai_search_plugin.adapters.verl_placement
        import ai_search_plugin.adapters.place_verify
        assert P.rebalance_swaps([9, 8, 1, 1], [0, 0, 1, 1], 2)[1] == 2
        print("ok")
    """)
    envv = dict(os.environ, PYTHONPATH=SRC)
    r = subprocess.run([sys.executable, "-c", code], env=envv, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]
