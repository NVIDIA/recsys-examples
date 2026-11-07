# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R3 EP plan dispatcher patch on CPU torch, with a stand-in for Megatron's alltoall dispatcher whose ``preprocess``
does Megatron's arithmetic (all-gather of counts over a simulated tp x ep group).  Checks: planned attributes equal the
original's on every rank / micro-batch / layer, a planned call never runs the original (no collective, no host sync),
verify steps, the device guard (wrong micro-batch -> fallback / strict raise), symmetric build skip, the id mapper,
and the slime glue (layer registry, replay cursor, clear_all step hook)."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai_search_plugin.adapters import megatron_r3plan as mp  # noqa: E402
from ai_search_plugin.adapters import slime_r3plan as sp  # noqa: E402

EP, TP, NLE, K = 2, 2, 3, 2
E, G = EP * NLE, EP * TP
M, L = 3, 2


class FakeGroup:
    def __init__(self, rank):
        self._rank = rank

    def size(self):
        return G

    def rank(self):
        return self._rank


class World:
    """routes[rank][m][li] = ids [T, K]; the 'current' (m, li) lets the stand-in original gather its peers' counts."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.routes = {r: [[torch.from_numpy(np.stack([rng.choice(E, K, replace=False) for _ in range(int(T))]))
                            for _ in range(L)] for T in rng.integers(4, 20, size=M)] for r in range(G)}
        self.cur = None
        self.meta_override = None

    def counts(self, r, m, li):
        return torch.bincount(self.routes[r][m][li].reshape(-1), minlength=E)

    def gather(self, group, t):
        if t.ndim == 1 and t.numel() == 4:                       # build meta
            rows = [t.clone() for _ in range(G)]
            if self.meta_override is not None:
                rows[1] = torch.tensor(self.meta_override, dtype=t.dtype)
            return torch.stack(rows)
        if t.ndim == 3:                                           # plan counts [M, L, E]
            return torch.stack([torch.stack([torch.stack([self.counts(r, m, li) for li in range(L)])
                                             for m in range(M)]).to(t.dtype) for r in range(G)])
        m, li = self.cur                                          # original preprocess: one layer call's counts
        return torch.stack([self.counts(r, m, li) for r in range(G)]).to(t.dtype)


class FakeDispatcher:
    """Megatron MoEAlltoAllTokenDispatcher.preprocess arithmetic (dropless, ep*tp > 1)."""
    calls = 0

    def __init__(self, world, rank, fusion=True):
        self.world = world
        self.config = SimpleNamespace(moe_router_topk=K, moe_permute_fusion=fusion, moe_expert_capacity_factor=None,
                                      moe_router_padding_for_quantization=False, cuda_graph_impl="none")
        self.drop_and_pad = False
        self.cuda_dtoh_point = "before_permutation_1"
        self.cuda_sync_point = "no_sync"
        self.num_experts, self.num_local_experts = E, NLE
        self.ep_size, self.tp_size = EP, TP
        e_idx, self.tp_rank = divmod(rank, TP)
        self.local_expert_indices = list(range(e_idx * NLE, (e_idx + 1) * NLE))
        self.tp_ep_group = FakeGroup(rank)
        self.num_global_tokens_per_local_expert = None

    def preprocess(self, routing_map):
        FakeDispatcher.calls += 1
        local = routing_map.sum(dim=0).long()
        self.num_out_tokens = routing_map.size(0) * K
        self.input_splits = local.reshape(EP, NLE).sum(axis=1)
        glob = self.world.gather(self.tp_ep_group, local).reshape(EP, TP, E).transpose(0, 1)
        ngtle = glob[:, :, self.local_expert_indices[0]:self.local_expert_indices[-1] + 1].contiguous()
        per_rank = ngtle.sum(axis=2)
        self.output_splits = per_rank[self.tp_rank]
        self.output_splits_tp = per_rank.sum(axis=1)
        self.cuda_sync_point = "before_ep_alltoall"
        if NLE > 1:
            self.num_global_tokens_per_local_expert = ngtle.view(-1, NLE)
        return ngtle.sum(dim=(0, 1))


class Source:
    def __init__(self, world, rank, disps):
        self.world, self.rank, self.disps = world, rank, disps
        self.on = True
        self.shift = 0          # wrong micro-batch mapping when != 0
        self.empty = False

    def active(self):
        return self.on

    def lists(self):
        if self.empty:
            return []
        return [(d, [self.world.routes[self.rank][m][li] for m in range(M)]) for li, d in enumerate(self.disps)]

    def index(self, li):
        return (self.world.cur[0] + self.shift) % M


def _rmap(ids):
    return torch.zeros(ids.shape[0], E, dtype=torch.bool).scatter_(1, ids.long(), True)


ATTRS = ("input_splits", "output_splits", "output_splits_tp", "num_out_tokens", "num_global_tokens_per_local_expert")


def _host(v):
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().astype(np.int64).reshape(-1)
    return np.asarray(v, dtype=np.int64).reshape(-1)


@pytest.fixture
def env(monkeypatch):
    mp._reset_for_tests()
    monkeypatch.setattr(FakeDispatcher, "preprocess", FakeDispatcher.preprocess)   # restored after the test
    for k in ("ASRT_R3PLAN_VERIFY_STEPS", "ASRT_R3PLAN_STRICT", "ASRT_LOG_DIR"):
        monkeypatch.delenv(k, raising=False)
    orig = FakeDispatcher.preprocess
    assert mp.patch_class(FakeDispatcher)
    yield orig
    mp._reset_for_tests()
    mp._S["installed"] = False


def _setup(rank, fusion=True, seed=0):
    w = World(seed)
    disps = [FakeDispatcher(w, rank, fusion) for _ in range(L)]
    src = Source(w, rank, disps)
    mp.set_source(src)
    mp._set_gather(w.gather)
    return w, disps, src


def _step(w, disps, order=None):
    out = []
    for m in (order or range(M)):
        for li, d in enumerate(disps):
            w.cur = (m, li)
            tok = d.preprocess(_rmap(w.routes[d.tp_ep_group.rank()][m][li]))
            out.append((m, li, _host(tok), {a: _host(getattr(d, a)) for a in ATTRS if getattr(d, a) is not None},
                        d.cuda_sync_point))
    return out


@pytest.mark.parametrize("rank", range(G))
@pytest.mark.parametrize("fusion", [True, False])
def test_planned_equals_original_and_skips_it(env, rank, fusion):
    orig = env
    w, disps, src = _setup(rank, fusion)
    ref_disps = [FakeDispatcher(w, rank, fusion) for _ in range(L)]
    ref = []
    for m in range(M):
        for li, d in enumerate(ref_disps):
            w.cur = (m, li)
            tok = orig(d, _rmap(w.routes[rank][m][li]))
            ref.append((_host(tok), {a: _host(getattr(d, a)) for a in ATTRS}))
    # step 1: verify step (original runs on every call, compared exactly)
    FakeDispatcher.calls = 0
    _step(w, disps)
    assert FakeDispatcher.calls == M * L and mp.STATS["verified_calls"] == M * L and mp.STATS["verify_fail"] == 0
    s = mp.check_step()
    assert not s["disabled"] and s["planned_steps"] == 1
    # step 2: planned, never calls the original
    FakeDispatcher.calls = 0
    got = _step(w, disps)
    assert FakeDispatcher.calls == 0 and mp.STATS["planned_calls"] == M * L
    for (m, li, tok, attrs, sync), (rtok, rattrs) in zip(got, ref):
        assert np.array_equal(tok, rtok), (m, li)
        for a in ATTRS:
            assert np.array_equal(attrs[a], rattrs[a]), (a, m, li)
        assert sync == "no_sync"
    s = mp.check_step()
    assert s["mismatch"] == 0 and not s["disabled"] and s["builds"] == 2
    if fusion:   # Megatron keeps num_global_tokens_per_local_expert as a device tensor under permute fusion
        assert torch.is_tensor(disps[0].num_global_tokens_per_local_expert)


def test_inactive_and_ineligible_fall_back(env):
    w, disps, src = _setup(0)
    src.on = False
    FakeDispatcher.calls = 0
    _step(w, disps)
    assert FakeDispatcher.calls == M * L and mp.STATS["fallback_calls"] == M * L and mp.STATS["builds"] == 0
    src.on = True
    for d in disps:
        d.config.moe_expert_capacity_factor = 1.0
    _step(w, disps)
    assert mp.STATS["fallback_calls"] == 2 * M * L and mp.STATS["builds"] == 0


def test_guard_wrong_microbatch_falls_back(env, monkeypatch):
    monkeypatch.setenv("ASRT_R3PLAN_VERIFY_STEPS", "0")
    w, disps, src = _setup(1)
    for li in range(L):                              # equal T across micro-batches, so only the device check
        T = min(w.routes[1][m][li].shape[0] for m in range(M))   # can see the wrong micro-batch
        for m in range(M):
            w.routes[1][m][li] = w.routes[1][m][li][:T]
    src.shift = 1                                    # plan rows of the wrong micro-batch
    FakeDispatcher.calls = 0
    _step(w, disps)
    assert FakeDispatcher.calls == 0 and mp.STATS["planned_calls"] == M * L
    s = mp.check_step()
    assert s["mismatch"] > 0 and s["disabled"] and s["step_record"]["action"] == "fallback"
    FakeDispatcher.calls = 0
    _step(w, disps)
    assert FakeDispatcher.calls == M * L          # original path from now on


def test_guard_strict_raises(env, monkeypatch):
    monkeypatch.setenv("ASRT_R3PLAN_VERIFY_STEPS", "0")
    monkeypatch.setenv("ASRT_R3PLAN_STRICT", "1")
    w, disps, src = _setup(2)
    for li in range(L):
        T = min(w.routes[2][m][li].shape[0] for m in range(M))
        for m in range(M):
            w.routes[2][m][li] = w.routes[2][m][li][:T]
    src.shift = 1
    _step(w, disps)
    with pytest.raises(RuntimeError, match="STRICT"):
        mp.check_step()


def test_verify_step_catches_wrong_mapping_with_original_values(env):
    orig = env
    w, disps, src = _setup(3)
    for li in range(L):
        T = min(w.routes[3][m][li].shape[0] for m in range(M))
        for m in range(M):
            w.routes[3][m][li] = w.routes[3][m][li][:T]
    src.shift = 1
    got = _step(w, disps)
    assert mp.STATS["verify_fail"] > 0
    ref_d = FakeDispatcher(w, 3)
    for m, li, tok, attrs, _ in got:                 # verify calls kept the original values
        w.cur = (m, li)
        assert np.array_equal(tok, _host(orig(ref_d, _rmap(w.routes[3][m][li]))))
    s = mp.check_step()
    assert s["disabled"] and "original values were used" in s["disabled_why"]


def test_build_skipped_symmetrically(env):
    w, disps, src = _setup(0)
    w.meta_override = [0, 0, 0, E]                  # another rank has no replay data
    FakeDispatcher.calls = 0
    _step(w, disps)
    assert mp.STATS["build_skipped"] == 1 and FakeDispatcher.calls == M * L and mp.STATS["planned_calls"] == 0
    mp.check_step()
    w.meta_override = [1, M + 1, L, E]              # ranks disagree on the micro-batch count
    _step(w, disps)
    assert mp.STATS["build_skipped"] == 2 and "disagree" in mp.snapshot()["why"]


def test_token_count_mismatch_raises(env, monkeypatch):
    monkeypatch.setenv("ASRT_R3PLAN_VERIFY_STEPS", "0")
    w, disps, src = _setup(0)
    w.cur = (0, 0)
    mp._build(disps[0])
    with pytest.raises(RuntimeError, match="tokens"):
        disps[0].preprocess(_rmap(w.routes[0][0][0][:-1]))


def test_mapper_applied(env, monkeypatch):
    monkeypatch.setenv("ASRT_R3PLAN_VERIFY_STEPS", "0")
    w, disps, src = _setup(0)
    perm = torch.tensor([1, 0, 2, 3, 4, 5])
    mp.set_route_mapper(lambda li, t: perm[t.long()])
    w.cur = (0, 0)
    p = mp._build(disps[0])
    assert np.array_equal(p.local_host[0, 0], np.bincount(perm[w.routes[0][0][0].long()].reshape(-1), minlength=E))


# ------------------------------------------------------------------------------------------------- slime glue
def test_slime_source_cursor_and_lists(monkeypatch):
    class RR:
        def __init__(self, lst):
            self.top_indices_list, self.forward_index, self.backward_index = lst, 0, 0

    d = object()
    a = [torch.zeros(3, K, dtype=torch.int32)] * M
    layer = SimpleNamespace(router=SimpleNamespace(routing_replay=RR(a)), token_dispatcher=d,
                            config=SimpleNamespace(moe_token_dispatcher_type="alltoall"))
    src = sp.SlimeReplaySource([layer])
    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "fallthrough")
    assert not src.active()
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")
    assert src.active()
    ent = src.lists()
    assert len(ent) == 1 and ent[0][0] is d and len(ent[0][1]) == M
    layer.router.routing_replay.forward_index = 2
    assert src.index(0) == 1
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_backward")
    layer.router.routing_replay.backward_index = 3
    assert src.index(0) == 2
    import sys
    mgr = SimpleNamespace(map_routed_ids=lambda lid, t: t + 1 + lid)
    monkeypatch.setitem(sys.modules, "ai_search_plugin.adapters.slime_placement", SimpleNamespace(_STATE={"mgr": mgr}))
    layer.router.routing_replay._asrt_layer = 2
    assert int(src.lists()[0][1][0][0, 0]) == 3        # placement remap applied like its pop hook
    lazy = SimpleNamespace(materialize_for_routing_replay=lambda stage: None)
    layer.router.routing_replay.top_indices_list = [lazy] * M
    assert src.lists() == []                         # lazy disk refs: no plan
    layer.router.routing_replay = None
    assert src.lists() == []
    layer.config.moe_token_dispatcher_type = "flex"
    assert sp.SlimeReplaySource([layer]).lists() == []


def test_slime_hooks(monkeypatch):
    mp._reset_for_tests()

    class Layer:
        def __init__(self, x):
            self.x = x

    sp._LAYERS.clear()
    assert sp.register_layer_class(Layer) and sp.register_layer_class(Layer)
    a, b = Layer(1), Layer(2)
    assert sp._LAYERS == [a, b]
    sp._LAYERS.clear()

    seen = []

    class RR:
        @staticmethod
        def clear_all():
            seen.append("orig")

    assert sp.patch_routing_replay(RR) and sp.patch_routing_replay(RR)
    RR.clear_all()
    assert seen == ["orig"] and mp.STATS["steps"] == 1

    assert sp.megatron_init(SimpleNamespace(use_rollout_routing_replay=False))["r3plan"].startswith("SKIP")
    assert "flex" in sp.megatron_init(SimpleNamespace(use_rollout_routing_replay=True,
                                                      moe_token_dispatcher_type="flex"))["r3plan"]
    monkeypatch.setenv("ASRT_R3PLAN", "0")
    assert sp.megatron_init(SimpleNamespace(use_rollout_routing_replay=True))["r3plan"].startswith("SKIP")
    mp._reset_for_tests()


def test_layout_reported():
    mp._reset_for_tests()
    assert mp.set_layout(4, 3) == "validated"
    assert mp.snapshot()["layout"] == "PP4/VPP3 (validated)"
    assert mp.set_layout(8, None) == "guarded"
    mp._reset_for_tests()
