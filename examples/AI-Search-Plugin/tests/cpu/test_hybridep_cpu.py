# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""hybridep_overlap without torch: switches, chunk/slot arithmetic, Megatron arguments, fallback rules, version /
dependency detection and the slime init hook's on / auto / off behaviour (fake Megatron + DeepEP modules)."""
import types

import pytest

from _hybridep_fakes import (OldHybridEPBufferAPI, cuda_env, hide_megatron, install_fake_megatron, moe_args)
from ai_search_plugin.adapters import megatron_hybridep as mh
from ai_search_plugin.adapters import slime_hybridep as sh


@pytest.fixture(autouse=True)
def _clean_patch():
    mh.uninstall()
    yield
    mh.uninstall()


# ------------------------------------------------------------------------------------------------ switches
def test_parse_spec_defaults_and_aliases():
    s = mh.parse_spec({})
    assert (s.mode, s.chunks, s.max_tokens, s.sms, s.allow_untested) == ("auto", 2, 32768, None, False)
    assert mh.parse_spec({"ASRT_HYBRIDEP": "1"}).mode == "on"
    assert mh.parse_spec({"ASRT_HYBRIDEP": "0"}).mode == "off"
    s = mh.parse_spec({"ASRT_HYBRIDEP_CHUNKS": "3", "ASRT_HYBRIDEP_MAX_TOKENS": "4096", "ASRT_HYBRIDEP_SMS": "16",
                       "ASRT_HYBRIDEP_ALLOW_UNTESTED": "1"})
    assert (s.chunks, s.max_tokens, s.sms, s.allow_untested) == (3, 4096, 16, True)


@pytest.mark.parametrize("env", [{"ASRT_HYBRIDEP": "maybe"}, {"ASRT_HYBRIDEP_CHUNKS": "0"},
                                 {"ASRT_HYBRIDEP_MAX_TOKENS": "1000"}, {"ASRT_HYBRIDEP_MAX_TOKENS": "0"},
                                 {"ASRT_HYBRIDEP_SMS": "0"}])
def test_parse_spec_rejects(env):
    with pytest.raises(ValueError):
        mh.parse_spec(env)


# ------------------------------------------------------------------------------------- chunk / slot arithmetic
@pytest.mark.parametrize("T,n", [(0, 2), (1, 2), (7, 3), (26624, 2), (26625, 2), (13, 4), (5, 8)])
def test_split_sizes(T, n):
    s = mh.split_sizes(T, n)
    assert len(s) == n and sum(s) == T and max(s) - min(s) <= 1 and s == sorted(s, reverse=True)


@pytest.mark.parametrize("cap", [64, 128, 4096, 32768, 65536 + 64])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_chunk_slots_bound_every_chunk(cap, n):
    slots = mh.chunk_slots(cap, n)
    assert slots % mh.ALIGN == 0
    assert slots <= cap                      # never more than the buffer was sized for
    for T in {0, 1, cap // 2, cap - 1, cap}:  # every micro-batch the buffer accepts fits chunk by chunk
        assert max(mh.split_sizes(T, n)) <= slots


def test_chunk_slots_values():
    assert mh.chunk_slots(32768, 2) == 16384
    assert mh.chunk_slots(32768, 3) == 10944   # ceil(32768/3)=10923 -> next multiple of 64


# ------------------------------------------------------------------------------------------ Megatron arguments
def test_megatron_args():
    assert mh.megatron_args() == ["--moe-token-dispatcher-type", "flex", "--moe-flex-dispatcher-backend", "hybridep"]
    assert mh.megatron_args(24)[-2:] == ["--moe-flex-dispatcher-num-sms", "24"]


def test_config_overrides_filter_to_available_fields():
    full = mh.config_overrides(16)
    assert full["moe_token_dispatcher_type"] == "flex" and full["moe_flex_dispatcher_backend"] == "hybridep"
    assert full["moe_hybridep_pad_uneven_dispatch_inputs"] is True and full["moe_permute_fusion_into_hybridep"] is False
    assert full["moe_flex_dispatcher_num_sms"] == 16 and full["moe_hybridep_routing_map_mode"] == "bool"
    old = mh.config_overrides(None, available_fields={"moe_hybridep_pad_uneven_dispatch_inputs"})
    assert set(old) == {"moe_token_dispatcher_type", "moe_flex_dispatcher_backend",
                        "moe_hybridep_pad_uneven_dispatch_inputs"}


def test_plan_fragment():
    f = sh.plan_fragment({"hybridep_sms": 20})
    assert f["env"] == {"ASRT_HYBRIDEP": "auto", "ASRT_HYBRIDEP_CHUNKS": "2", "ASRT_HYBRIDEP_MAX_TOKENS": "32768",
                        "ASRT_HYBRIDEP_SMS": "20"}
    assert f["args"] == mh.megatron_args(20)
    assert f["chain"] == ["ai_search_plugin.adapters.slime_hybridep.megatron_init"]
    assert f["notes"]
    with pytest.raises(ValueError):
        sh.plan_fragment({"hybridep_max_tokens": 100})
    with pytest.raises(ValueError):
        sh.plan_fragment({"hybridep_mode": "off"})


# --------------------------------------------------------------------------------------------- fallback rules
def _layer(**cfg):
    c = dict(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="hybridep", moe_expert_capacity_factor=None)
    c.update(cfg)
    return types.SimpleNamespace(config=types.SimpleNamespace(**c), training=True, shared_expert_overlap=False,
                                 token_dispatcher=types.SimpleNamespace(tp_size=1))


def test_fallback_reason_plain_training_path():
    assert mh.fallback_reason(_layer()) is None
    lay = _layer()
    lay.fwd_execution_map = ("route", "expert_compute", "postprocess")
    assert mh.fallback_reason(lay) is None


@pytest.mark.parametrize("change", [
    dict(cfg=dict(moe_flex_dispatcher_backend="deepep")), dict(cfg=dict(moe_token_dispatcher_type="alltoall")),
    dict(cfg=dict(moe_permute_fusion_into_hybridep=True)), dict(cfg=dict(moe_expert_capacity_factor=1.0)),
    dict(cfg=dict(moe_pad_expert_input_to_capacity=True)), dict(cfg=dict(moe_latent_size=512)),
    dict(cfg=dict(fp8="hybrid")), dict(cfg=dict(fp4="nvfp4")), dict(cfg=dict(moe_hybridep_routing_map_mode="indices")),
    dict(cfg=dict(overlap_dispatch_backward_with_experts_wgrad=True)),
    dict(cfg=dict(moe_expert_rank_capacity_factor=1.5)),
    dict(layer=dict(training=False)), dict(layer=dict(shared_expert_overlap=True)),
    dict(layer=dict(_inference_token_dispatcher=object())), dict(layer=dict(fwd_execution_map=("route",))),
    dict(layer=dict(token_dispatcher=types.SimpleNamespace(tp_size=2))),
    dict(call=dict(intermediate_tensors=(1, 2))), dict(call=dict(extra_kwargs={"x": 1})),
])
def test_fallback_reason_cases(change):
    lay = _layer(**change.get("cfg", {}))
    for k, v in change.get("layer", {}).items():
        setattr(lay, k, v)
    assert mh.fallback_reason(lay, **change.get("call", {})) is not None


# --------------------------------------------------------------------------------- dependency / version checks
def test_check_environment_ok(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch)
    r = mh.check_environment(env=cuda_env(tmp_path))
    assert r == {"version": (0, 19, 0), "hybridep": [], "overlap": []}


def test_check_environment_no_megatron(monkeypatch, tmp_path):
    hide_megatron(monkeypatch)
    r = mh.check_environment(env=cuda_env(tmp_path))
    assert r["version"] is None and r["hybridep"]


def test_check_environment_no_hybridep_backend(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch, have_hybridep=False, buffer_cls=None)
    r = mh.check_environment(env=cuda_env(tmp_path))
    assert any("HAVE_HYBRIDEP" in p for p in r["hybridep"]) and any("HybridEPBuffer" in p for p in r["hybridep"])


def test_check_environment_untested_version(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch, version=(0, 21, 0))
    r = mh.check_environment(env=cuda_env(tmp_path))
    assert not r["hybridep"] and len(r["overlap"]) == 1 and "0.21.0" in r["overlap"][0]
    assert mh.check_environment(allow_untested=True, env=cuda_env(tmp_path))["overlap"] == []


def test_check_environment_api_and_fields(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch, buffer_cls=OldHybridEPBufferAPI,
                          fields=mh.DISPATCHER_FIELDS + ("moe_flex_dispatcher_num_sms",))
    r = mh.check_environment(env=cuda_env(tmp_path))
    assert not r["hybridep"]
    assert any("num_of_tokens_per_rank" in p for p in r["overlap"])
    assert any("moe_hybridep_num_blocks_permute" in p for p in r["overlap"])


def test_check_environment_no_nvcc(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch)
    r = mh.check_environment(env={"CUDA_HOME": str(tmp_path / "nowhere")})
    assert any("nvcc" in p for p in r["hybridep"])


def test_presize_buffer(monkeypatch):
    ns = install_fake_megatron(monkeypatch)
    assert mh.presize_buffer(4096) and mh.presize_buffer(4096)          # idempotent: wrapped once
    ns.fused_a2a.init_hybrid_ep_buffer("g", 2560, 100, 64, 24, 24)
    ns.fused_a2a.init_hybrid_ep_buffer("g", 2560, 9000, 64)
    assert [c[1] for c in ns.fused_a2a.init_calls] == [4096, 9000]
    assert ns.fused_a2a.init_calls[0][3:] == (24, 24)


# ------------------------------------------------------------------------------------- slime init hook decisions
def test_init_off(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch)
    args = moe_args()
    r = sh.megatron_init(args, environ=cuda_env(tmp_path, ASRT_HYBRIDEP="off"))
    assert r["why"] == "ASRT_HYBRIDEP=off" and args.moe_token_dispatcher_type == "flex" and not mh.snapshot()["installed"]


def test_init_dense_model_is_noop(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch)
    args = moe_args(num_experts=None)
    r = sh.megatron_init(args, environ=cuda_env(tmp_path))
    assert not r["hybridep"] and not mh.snapshot()["installed"]


def test_init_auto_without_hybridep_reverts_to_alltoall(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch, have_hybridep=False)
    args = moe_args()
    r = sh.megatron_init(args, environ=cuda_env(tmp_path))
    assert not r["hybridep"] and not r["overlap"] and "HAVE_HYBRIDEP" in r["why"]
    assert args.moe_token_dispatcher_type == "alltoall" and not mh.snapshot()["installed"]


def test_init_auto_without_megatron(monkeypatch, tmp_path):
    hide_megatron(monkeypatch)
    args = moe_args()
    r = sh.megatron_init(args, environ=cuda_env(tmp_path))
    assert not r["hybridep"] and args.moe_token_dispatcher_type == "alltoall"


def test_init_on_without_hybridep_raises(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch, have_hybridep=False)
    with pytest.raises(RuntimeError, match="HybridEP not usable"):
        sh.megatron_init(moe_args(), environ=cuda_env(tmp_path, ASRT_HYBRIDEP="on"))


def test_init_auto_untested_megatron_keeps_dispatcher_only(monkeypatch, tmp_path):
    ns = install_fake_megatron(monkeypatch, version=(0, 21, 0))
    orig = ns.MoELayer.forward
    args = moe_args()
    r = sh.megatron_init(args, environ=cuda_env(tmp_path))
    assert r["hybridep"] and not r["overlap"] and "0.21.0" in r["why"]
    assert args.moe_hybridep_pad_uneven_dispatch_inputs is True and args.moe_hybridep_routing_map_mode == "bool"
    assert r["presized"] and ns.MoELayer.forward is orig
    with pytest.raises(RuntimeError, match="overlap patch not applicable"):
        sh.megatron_init(moe_args(), environ=cuda_env(tmp_path, ASRT_HYBRIDEP="on"))


def test_init_expert_tp_disables_overlap(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch)
    r = sh.megatron_init(moe_args(expert_tensor_parallel_size=2), environ=cuda_env(tmp_path))
    assert r["hybridep"] and not r["overlap"] and "expert tensor parallel" in r["why"]


def test_init_one_chunk_is_dispatcher_only_even_when_on(monkeypatch, tmp_path):
    install_fake_megatron(monkeypatch)
    r = sh.megatron_init(moe_args(), environ=cuda_env(tmp_path, ASRT_HYBRIDEP="on", ASRT_HYBRIDEP_CHUNKS=1))
    assert r["hybridep"] and not r["overlap"] and not mh.snapshot()["installed"]


def test_init_installs_patch(monkeypatch, tmp_path):
    ns = install_fake_megatron(monkeypatch)
    orig = ns.MoELayer.forward
    env = cuda_env(tmp_path, ASRT_HYBRIDEP_SMS=24)
    args = moe_args(moe_permute_fusion_into_hybridep=True)
    r = sh.megatron_init(args, environ=env)
    assert r["hybridep"] and r["overlap"] and r["install"]["num_chunks"] == 2
    assert args.moe_flex_dispatcher_num_sms == 24 and args.moe_permute_fusion_into_hybridep is False
    assert ns.MoELayer.forward is not orig and getattr(ns.MoELayer.forward, "_asrt_hybridep", False)
    assert env["HYBRID_EP_CACHE_DIR"] == str(tmp_path)
    # the patched forward still takes Megatron's path when the layer is not training
    layer = ns.MoELayer()
    layer.config = types.SimpleNamespace(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="hybridep",
                                         moe_expert_capacity_factor=None)
    layer.training, layer.shared_expert_overlap = False, False
    assert layer.forward("h") == ("orig", None) and layer.orig_calls == 1
    assert mh.snapshot()["fallback_why"] == {"not training": 1}
    # second init with the same switches is a no-op; different chunk count is refused
    assert sh.megatron_init(moe_args(), environ=env)["overlap"]
    with pytest.raises(RuntimeError, match="already installed"):
        mh.install(3)
    mh.uninstall()
    assert ns.MoELayer.forward is orig


def test_install_rejects_bad_params():
    with pytest.raises(ValueError):
        mh.install(1)
    with pytest.raises(ValueError):
        mh.install(2, max_tokens=100)
