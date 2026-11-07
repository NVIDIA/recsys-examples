# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the expert-placement CPU tests (not a test module).

* ``fake_env(monkeypatch)``: fake ``megatron.core.parallel_state`` (EP world 1), torch.cuda device -> "cpu", and a
  sys.modules snapshot so fake slime / verl / megatron modules and freshly imported placement modules are dropped
  again after the test.
* a toy MoE model with the Megatron-Core structure the placement code walks (``layers.<i>.mlp.{router, experts}``,
  ``experts.num_local_experts``, SequentialMLP-style ``experts.local_experts``).
"""
import sys
import types

import torch

E, H, F, K = 16, 8, 6, 2

_PLACEMENT_MODS = ("ai_search_plugin.adapters.slime_placement", "ai_search_plugin.adapters.verl_placement",
                   "ai_search_plugin.adapters.megatron_placement", "ai_search_plugin.adapters.place_verify")


def fake_env(monkeypatch):
    before = set(sys.modules)
    ps = types.ModuleType("megatron.core.parallel_state")
    ps.get_expert_model_parallel_world_size = lambda: 1
    ps.get_expert_model_parallel_group = lambda: None
    meg = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    core.parallel_state = ps
    meg.core = core
    monkeypatch.setitem(sys.modules, "megatron", meg)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", ps)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    import ai_search_plugin.adapters as pkg
    for m in _PLACEMENT_MODS:   # fresh module state (_STATE / _S / _V) for every test
        monkeypatch.delitem(sys.modules, m, raising=False)
        monkeypatch.delattr(pkg, m.rsplit(".", 1)[1], raising=False)   # `from pkg import mod` reads the attribute

    def cleanup():
        for k in list(sys.modules):
            if k not in before and (k.split(".")[0] in ("slime", "verl", "megatron") or k in _PLACEMENT_MODS):
                sys.modules.pop(k, None)
    return cleanup


class Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(E, H))

    def forward(self, x):
        logits = x @ self.weight.t()
        probs, idx = torch.topk(torch.softmax(logits, -1), K, dim=-1)
        rm = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, idx, True)
        return probs, rm


class Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Parameter(torch.randn(F, H))
        self.fc2 = torch.nn.Parameter(torch.randn(H, F))

    def forward(self, x):
        return torch.relu(x @ self.fc1.t()) @ self.fc2.t()


class Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_local_experts = E
        self.local_experts = torch.nn.ModuleList([Expert() for _ in range(E)])


class MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = Router()
        self.experts = Experts()

    def forward(self, x):
        probs, rm = self.router(x)
        out = torch.zeros_like(x)
        for s in range(E):
            sel = rm[:, s]
            if sel.any():
                w = torch.softmax(x[sel] @ self.router.weight.t(), -1)[:, s:s + 1]
                out[sel] += w * self.experts.local_experts[s](x[sel])
        return out


class Layer(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layer_number = n
        self.mlp = MoE()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(1), Layer(2)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer.mlp(x)
        return x


class Wrapped(torch.nn.Module):
    """A model-specific wrapper around the MoE block (layers.N.mlp.module.*)."""

    def __init__(self, inner):
        super().__init__()
        self.module = inner

    def forward(self, x):
        return self.module(x)


class Dense(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        for i, layer in enumerate(self.layers):
            layer.layer_number = i + 1
            layer.mlp = torch.nn.Linear(8, 8)


def trained_model(lr=1e-2, wrap_mlp=False, steps=1, n=64):
    """Toy model + Adam with optimizer state already created, plus the input batch."""
    torch.manual_seed(0)
    m = Model()
    if wrap_mlp:
        for layer in m.layers:
            layer.mlp = Wrapped(layer.mlp)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    x = torch.randn(n, H)
    for _ in range(steps):
        m(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    return m, opt, x
