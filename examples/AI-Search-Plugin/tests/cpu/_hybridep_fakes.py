# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-in Megatron / DeepEP modules for the hybridep_overlap CPU tests (no GPU, no Megatron, no DeepEP)."""
import dataclasses
import sys
import types

from ai_search_plugin.adapters import megatron_hybridep as mh

ALL_FIELDS = mh.DISPATCHER_FIELDS + mh.OVERLAP_FIELDS + ("moe_hybridep_routing_map_mode",)
MODULES = ("megatron", "megatron.core", "megatron.core.package_info", "megatron.core.transformer",
           "megatron.core.transformer.moe", mh._FUSED_A2A, mh._MOE_LAYER, mh._TCONFIG, "megatron.core.tensor_parallel",
           "deep_ep")


class FakeHybridEPBufferAPI:
    """Signature-only HybridEPBuffer (the validated parameter names)."""

    def dispatch_with_permute(self, hidden, routing_map=None, probs=None, scaling_factor=None,
                              num_of_experts_per_rank=None, pad_multiple=None, num_permuted_tokens=None,
                              non_blocking=False, num_of_tokens_per_rank=None, handle=None):
        raise NotImplementedError

    def combine_with_unpermute(self, hidden, probs=None, handle=None):
        raise NotImplementedError


class OldHybridEPBufferAPI:
    def dispatch_with_permute(self, hidden, routing_map=None, probs=None, handle=None):
        raise NotImplementedError

    def combine_with_unpermute(self, hidden, probs=None, handle=None):
        raise NotImplementedError


class FakeMoELayerBase:
    def forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
        self.orig_calls = getattr(self, "orig_calls", 0) + 1
        return "orig", None

    def route(self, hidden_states, padding_mask=None):
        raise NotImplementedError

    def shared_experts_compute(self, hidden_states):
        return None

    def postprocess(self, output, shared):
        return output


def hide_megatron(monkeypatch):
    """Make every Megatron / DeepEP import fail."""
    for name in MODULES:
        monkeypatch.setitem(sys.modules, name, None)


def install_fake_megatron(monkeypatch, version=(0, 19, 0), have_hybridep=True, fields=ALL_FIELDS,
                          buffer_cls=FakeHybridEPBufferAPI, moe_layer_cls=None, buffer=None):
    """Register fake modules in sys.modules; returns a namespace with the pieces."""
    ns = types.SimpleNamespace()

    def mod(name):
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    for name in ("megatron", "megatron.core", "megatron.core.transformer", "megatron.core.transformer.moe"):
        mod(name)
    pi = mod("megatron.core.package_info")
    pi.MAJOR, pi.MINOR, pi.PATCH = version

    fa = mod(mh._FUSED_A2A)
    fa.HAVE_HYBRIDEP = have_hybridep
    fa._hybrid_ep_buffer = buffer
    fa.init_calls = []

    def init_hybrid_ep_buffer(group, hidden_dim, num_tokens, num_local_experts, *rest):
        fa.init_calls.append((hidden_dim, num_tokens, num_local_experts) + tuple(rest))

    fa.init_hybrid_ep_buffer = init_hybrid_ep_buffer
    ns.fused_a2a = fa

    tc = mod(mh._TCONFIG)
    tc.TransformerConfig = dataclasses.make_dataclass("TransformerConfig", [(f, object, None) for f in fields])

    de = mod("deep_ep")
    if buffer_cls is not None:
        de.HybridEPBuffer = buffer_cls

    ml = mod(mh._MOE_LAYER)
    ml.MoELayer = moe_layer_cls or type("MoELayer", (FakeMoELayerBase,), {})
    ns.MoELayer = ml.MoELayer

    tp = mod("megatron.core.tensor_parallel")
    tp.calls = []

    def checkpoint(fn, distribute, *args):
        tp.calls.append(distribute)
        return fn(*args)

    tp.checkpoint = checkpoint
    ns.tensor_parallel = tp
    return ns


def cuda_env(tmp_path, **extra):
    """An env mapping whose CUDA_HOME has bin/nvcc."""
    b = tmp_path / "cuda" / "bin"
    b.mkdir(parents=True, exist_ok=True)
    (b / "nvcc").write_text("")
    env = {"CUDA_HOME": str(tmp_path / "cuda"), "TMPDIR": str(tmp_path)}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def moe_args(**kw):
    a = dict(num_experts=8, expert_tensor_parallel_size=1, moe_token_dispatcher_type="flex",
             moe_flex_dispatcher_backend="hybridep", moe_expert_capacity_factor=None, fp8=None)
    a.update(kw)
    return types.SimpleNamespace(**a)
