# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""slime glue of expert placement (adapters.slime_placement) against fake slime modules:
R3 counts drive a re-placement before the replay is recorded, replayed ids come out physical, weight export comes
out logical (also through a weight-sync module that a model init hook re-wrapped, before or after install, with and
without slime's conversion-core function), save_model sees logical order, the self-checks catch a bypassed export
and an unbound replay buffer, a dense model turns placement off, and balanced batches are skipped (auto-off)."""
import io
import sys
import types
from contextlib import redirect_stdout

import pytest

torch = pytest.importorskip("torch")
import placement_toy as T  # noqa: E402


@pytest.fixture
def env(monkeypatch):
    for k in ("ASRT_PLACE", "ASRT_PLACE_RANDOM", "ASRT_PLACE_LOG", "ASRT_PLACE_EVERY", "ASRT_PLACE_PROBE_EVERY",
              "ASRT_PLACE_MIN_IMB"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ASRT_PLACE_STRIDE", "1")
    monkeypatch.setenv("ASRT_PLACE_THRESH", "1.01")
    monkeypatch.setenv("ASRT_PLACE_SELFCHECK", "raise")
    cleanup = T.fake_env(monkeypatch)
    yield monkeypatch
    cleanup()


def fake_slime(monkeypatch, core_hook=True):
    names = ["slime", "slime.backends", "slime.backends.megatron_utils", "slime.utils",
             "slime.backends.megatron_utils.actor", "slime.backends.megatron_utils.megatron_to_hf",
             "slime.backends.megatron_utils.update_weight",
             "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
             "slime.utils.routing_replay", "slime.utils.routed_experts"]
    mods = {n: types.ModuleType(n) for n in names}
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    mods["slime.backends.megatron_utils"].actor = mods["slime.backends.megatron_utils.actor"]
    mods["slime.backends.megatron_utils"].megatron_to_hf = mods["slime.backends.megatron_utils.megatron_to_hf"]
    mods["slime.utils"].routing_replay = mods["slime.utils.routing_replay"]
    mods["slime.utils.routed_experts"].materialize_routed_experts = lambda v, **k: torch.as_tensor(v)
    M2H = mods["slime.backends.megatron_utils.megatron_to_hf"]
    UWD = mods["slime.backends.megatron_utils.update_weight.update_weight_from_distributed"]

    class RoutingReplay:
        all_routing_replays = []

        def __init__(self):
            self.lst, self.i = [], 0
            RoutingReplay.all_routing_replays.append(self)

        def record(self, t):
            self.lst.append(t)

        def pop_forward(self):
            t = self.lst[self.i]
            self.i += 1
            return t

        def pop_backward(self):
            return self.lst[-1]
    mods["slime.utils.routing_replay"].RoutingReplay = RoutingReplay

    exported = []
    M2H.exported = exported
    if core_hook:
        # like slime: convert_to_hf looks the core up as a module global at call time
        exec("def _convert_to_hf_core(args, model_name, name, param):\n"
             "    exported.append((name, param))\n"
             "    return [(name, param)]\n"
             "def convert_to_hf(args, model_name, name, param, *a, **k):\n"
             "    return _convert_to_hf_core(args, model_name, name, param)\n", M2H.__dict__)
    else:  # a slime without the core function: convert_to_hf does the work itself
        def convert_to_hf(args, model_name, name, param, *a, **k):
            exported.append((name, param))
            return [(name, param)]
        M2H.convert_to_hf = convert_to_hf
    UWD.convert_to_hf = M2H.convert_to_hf     # `from ..megatron_to_hf import convert_to_hf`

    class MegatronTrainRayActor:
        def __init__(self, model, opt):
            self.model, self.optimizer = [model], opt
            self.args = types.SimpleNamespace(num_layers=2, num_experts=T.E, use_rollout_routing_replay=True)
            self.saved = None

        def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
            for v in rollout_data["rollout_routed_experts"]:
                for rp, lid in zip(RoutingReplay.all_routing_replays, [0, 1]):
                    rp.record(torch.as_tensor(v)[:, lid])
            del rollout_data["rollout_routed_experts"]

        def train_actor(self, rollout_id, rollout_data, external_data=None):
            pass

        def save_model(self, *a, **k):
            self.saved = {n: p.detach().clone() for n, p in self.model[0].named_parameters()}

        def update_weights(self):
            # slime's weight sync: experts and routers through the *weight-sync module's* convert_to_hf
            for lid in range(2):
                moe = self.model[0].layers[lid].mlp
                moe = getattr(moe, "module", moe)
                UWD.convert_to_hf(self.args, "moe", f"module.decoder.layers.{lid}.mlp.router.weight",
                                  moe.router.weight.detach())
                for s in range(T.E):
                    UWD.convert_to_hf(self.args, "moe",
                                      f"module.decoder.layers.{lid}.mlp.experts.linear_fc1.weight{s}",
                                      moe.experts.local_experts[s].fc1.detach())
    mods["slime.backends.megatron_utils.actor"].MegatronTrainRayActor = MegatronTrainRayActor
    return MegatronTrainRayActor, RoutingReplay, M2H, UWD


def model_init_hook_wraps(UWD):
    """What a model-specific init hook may do: give the weight-sync module its own convert_to_hf wrapper."""
    inner = UWD.convert_to_hf

    def convert_to_hf(args, model_name, name, param, *a, **k):
        return inner(args, model_name, name, param.contiguous(), *a, **k)
    UWD.convert_to_hf = convert_to_hf


def _glue():
    import ai_search_plugin.adapters.slime_placement as SP
    return SP


def test_glue_r3_replay_export_save(env):
    Actor, RR, M2H, _ = fake_slime(env)
    SP = _glue()
    SP.megatron_init(None)
    m, opt, x = T.trained_model(n=256)
    y0 = m(x).detach()
    snap = {n: p.detach().clone() for n, p in m.named_parameters()}
    actor = Actor(m, opt)
    RR(), RR()  # one replay per MoE layer, as slime creates them (not reachable from the router here)
    routes = [torch.randint(0, 4, (50, 2, T.K)), torch.randint(0, T.E, (20, 2, T.K))]
    mgr = SP._ensure_manager(actor)
    mgr.ep, mgr.n_local = 4, 4
    actor.fill_routing_replay(None, None, {"rollout_routed_experts": list(routes)})
    assert not mgr.is_identity(), "hot experts should have triggered a move"
    assert mgr.history[-1]["imbalance_before"] > mgr.history[-1]["imbalance_after_pred"]
    assert torch.allclose(m(x).detach(), y0, atol=1e-3)
    # replay returns physical ids
    got = RR.all_routing_replays[0].pop_forward()
    assert torch.equal(got, mgr.slot_of[0][routes[0][:, 0]]), "R3 ids must be mapped to slots"
    # export: logical names + router rows in logical order
    M2H.convert_to_hf(None, "x", "module.module.decoder.layers.0.mlp.router.weight",
                      m.layers[0].mlp.router.weight.detach())
    assert torch.equal(M2H.exported[-1][1], snap["layers.0.mlp.router.weight"])
    s = int(mgr.slot_of[0][2])
    M2H.convert_to_hf(None, "x", f"module.module.decoder.layers.0.mlp.experts.linear_fc1.weight{s}", torch.zeros(1))
    assert M2H.exported[-1][0].endswith("weight2"), M2H.exported[-1][0]
    # save_model sees logical order, placement is re-applied afterwards
    before = mgr.slot_of.clone()
    actor.save_model(0)
    for n, p in actor.saved.items():
        assert torch.equal(p, snap[n]), n
    assert torch.equal(mgr.slot_of, before)
    assert torch.allclose(m(x).detach(), y0, atol=1e-3)


def scenario(monkeypatch, hook_when, wrap_mlp=False, core_hook=True, extra_replay=False, bypass=False):
    """Returns (export logical?, self-check error or None)."""
    Actor, RR, M2H, UWD = fake_slime(monkeypatch, core_hook)
    SP = _glue()
    if hook_when == "before":
        model_init_hook_wraps(UWD)
    SP.megatron_init(None)
    if hook_when == "after":
        model_init_hook_wraps(UWD)
    if bypass:  # an export path that never reaches the conversion core / the mapping
        def raw(args, model_name, name, param, *a, **k):
            M2H.exported.append((name, param))
            return [(name, param)]
        UWD.convert_to_hf = raw
    m, opt, _ = T.trained_model(wrap_mlp=wrap_mlp)
    actor = Actor(m, opt)
    mgr = SP._ensure_manager(actor)
    mgr.ep, mgr.n_local = 4, 4
    for lid, moe in mgr.layers:  # slime: router.routing_replay = RoutingReplay()
        moe.router.routing_replay = RR()
    if extra_replay:
        RR()
    try:
        actor.update_weights()  # initial sync (identity): the self-check runs before any move too
        routes = [torch.randint(0, 4, (60, 2, T.K)), torch.randint(0, T.E, (20, 2, T.K))]
        actor.fill_routing_replay(None, None, {"rollout_routed_experts": list(routes)})
        assert not mgr.is_identity()
        M2H.exported.clear()
        actor.update_weights()
        snap = {lid: mgr.physical_to_logical(lid).clone() for lid in range(2)}
        moved = mgr.moved_local_layers()
        exported = list(M2H.exported)
        actor.train_actor(0, {})     # end of the rollout: the glue's step counter advances
        actor.fill_routing_replay(None, None, {"rollout_routed_experts": [torch.randint(0, T.E, (8, 2, T.K))]})
    except RuntimeError as e:
        return None, str(e), SP
    assert moved
    ok = True
    for lid in moved:
        moe = m.layers[lid].mlp
        moe = getattr(moe, "module", moe)
        for name, t in exported:
            if f"layers.{lid}.mlp.experts" in name:
                s = int(name.rsplit("weight", 1)[1])
                # the tensor came from physical slot p (logical id snap[lid][p]); the exported name must carry it
                p = [q for q in range(T.E) if torch.equal(moe.experts.local_experts[q].fc1.detach(), t)][0]
                if s != int(snap[lid][p]):
                    ok = False
    return ok, None, SP


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("core", [True, False])
def test_wrapped_weight_sync_exports_logical(env, when, core):
    ok, err, _ = scenario(env, when, core_hook=core)
    assert err is None and ok is True


def test_wrapped_moe_block(env):
    ok, err, SP = scenario(env, "before", wrap_mlp=True)
    mgr = SP.get_manager()
    assert mgr.local_layer_ids() == [0, 1] and err is None and ok is True
    assert mgr.export_name_and_tensor("decoder.layers.0.mlp.module.experts.linear_fc1.weight3",
                                      torch.zeros(1))[0].endswith("weight%d" % int(mgr.physical_to_logical(0)[3]))
    assert mgr.export_name_and_tensor("decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
                                      torch.zeros(1))[0].endswith("shared_experts.linear_fc1.weight")


def test_selfcheck_bypassed_export(env):
    _, err, _ = scenario(env, "before", bypass=True)
    assert err and "never passed through the placement hook" in err


def test_selfcheck_unbound_replay(env):
    _, err, _ = scenario(env, "before", extra_replay=True)
    assert err and "R3 replay layers" in err


def test_auto_off_probe_backoff(env):
    env.setenv("ASRT_PLACE_PROBE_EVERY", "3")
    Actor, RR, _, _ = fake_slime(env)
    SP = _glue()
    SP.megatron_init(None)
    m, opt, _ = T.trained_model()
    actor = Actor(m, opt)
    mgr = SP._ensure_manager(actor)
    mgr.ep, mgr.n_local = 4, 4
    for _, moe in mgr.layers:
        moe.router.routing_replay = RR()
    uniform = torch.arange(T.E).repeat(64).view(-1, 1, 1).expand(-1, 2, T.K).contiguous()   # perfectly balanced
    for i in range(4):
        actor.fill_routing_replay(None, None, {"rollout_routed_experts": [uniform]})
        actor.update_weights()
        actor.train_actor(i, {})
    # step 0 checks and skips, steps 1-2 back off (no counting, no collective), step 3 probes again
    assert len(mgr.history) == 2 and all(h.get("skipped") for h in mgr.history), mgr.history
    assert mgr.is_identity()


def test_off_switch(env):
    env.setenv("ASRT_PLACE", "0")
    Actor, _, _, _ = fake_slime(env)
    SP = _glue()
    assert "off" in SP.megatron_init(None)
    assert not SP._STATE["installed"]
    assert "make_fill" not in Actor.fill_routing_replay.__qualname__


def test_dense_model_turns_off(env):
    Actor, _, M2H, _ = fake_slime(env)
    SP = _glue()
    SP.megatron_init(None)
    m = T.Dense()
    actor = Actor(m, torch.optim.Adam(m.parameters()))
    actor.args = types.SimpleNamespace(num_layers=2, num_experts=None, use_rollout_routing_replay=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert SP._ensure_manager(actor) is None
        actor.fill_routing_replay(None, None, {"rollout_routed_experts": []})
        actor.train_actor(0, {})
        actor.args.use_rollout_routing_replay = False
        actor.train_actor(1, {})
        out = M2H.convert_to_hf(None, "x", "module.decoder.layers.0.mlp.weight", torch.zeros(2))
        actor.save_model(0)
    assert "[asrt] place: no MoE layers -> off" in buf.getvalue(), buf.getvalue()
    assert out[0][0].endswith("layers.0.mlp.weight") and SP.get_manager() is None


def test_dense_model_with_num_experts_set(env):
    """num_experts set but the model has no MoE block (e.g. wrong args): also off, not an error."""
    Actor, _, _, _ = fake_slime(env)
    SP = _glue()
    SP.megatron_init(None)
    m = T.Dense()
    actor = Actor(m, torch.optim.Adam(m.parameters()))
    actor.args = types.SimpleNamespace(num_layers=2, num_experts=16, use_rollout_routing_replay=True)
    with redirect_stdout(io.StringIO()):
        assert SP._ensure_manager(actor) is None
