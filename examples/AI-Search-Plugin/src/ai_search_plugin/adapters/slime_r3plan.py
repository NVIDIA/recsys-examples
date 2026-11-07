# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""slime glue for the R3 EP plan (:mod:`ai_search_plugin.adapters.megatron_r3plan`).

slime's rollout routing replay (``--use-rollout-routing-replay``) puts a ``RoutingReplay`` object on every MoE router
and fills it with all micro-batches' routes before the passes of a step (``fill_routing_replay``).  This module

  * records every Megatron ``MoELayer`` in construction order (= layer order on this rank, identical on all ranks of
    the tp x ep group) - so it must be installed before the model is built: slime's ``--custom-megatron-init-path``
    runs before the model provider;
  * gives the dispatcher patch its source: replay lists per layer from ``router.routing_replay.top_indices_list`` and
    the micro-batch of the current call from the replay cursor of the stage slime set (``ROUTING_REPLAY_STAGE``:
    ``replay_forward`` -> ``forward_index - 1``, ``replay_backward`` (activation recompute) -> ``backward_index - 1``;
    ``fallthrough`` / ``record`` (reference model, R3 off) -> not planned).  Exact under any pipeline schedule,
    because it reads the same counter the router just used;
  * closes each step from ``RoutingReplay.clear_all`` (slime calls it on every training rank right after
    ``train``): the guard check (collective), stats, and the plan is dropped so the next step rebuilds it.

With the expert_placement component active, the replay ids are remapped exactly as its pop-time hook does.

Falls back to Megatron's original path (no plan) when: R3 is off, the dispatcher is not ``alltoall``, a layer has no
replay data, or the replay lists are lazy disk references (``materialize_for_routing_replay``; not counted ahead).

Enable: chain ``ai_search_plugin.adapters.slime_r3plan.megatron_init`` in ``ASRT_CHAIN_MEGATRON_INIT`` (or use it as
``--custom-megatron-init-path``).  Being in the chain is the switch; ``ASRT_R3PLAN=0`` turns it off without editing
the chain.  Guard knobs: ``ASRT_R3PLAN_VERIFY_STEPS`` (default 1), ``ASRT_R3PLAN_STRICT`` (default 0).
Prints stats on rank 0 for steps 1, 2 and every ``ASRT_R3PLAN_LOG_EVERY`` (10).
"""
from __future__ import annotations

import os

from .. import r3plan as core
from . import megatron_r3plan as mp

_LAYERS: list = []
_state = {"installed": False, "calls": 0}


class SlimeReplaySource:
    """Source for megatron_r3plan from slime's RoutingReplay objects (one per router)."""

    def __init__(self, layers=None):
        self.layers = _LAYERS if layers is None else layers
        self._rr = None

    @staticmethod
    def stage():
        return os.environ.get("ROUTING_REPLAY_STAGE")

    def active(self) -> bool:
        return os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1" and self.stage() in core.REPLAY_STAGES

    def _replays(self):
        return [getattr(getattr(layer, "router", None), "routing_replay", None) for layer in self.layers]

    def lists(self):
        out = []
        rrs = []
        for layer, rr in zip(self.layers, self._replays()):
            cfg = getattr(layer, "config", None)
            if getattr(cfg, "moe_token_dispatcher_type", "alltoall") != "alltoall":
                return []
            if rr is None or not rr.top_indices_list or any(
                    hasattr(t, "materialize_for_routing_replay") for t in rr.top_indices_list):
                return []
            ids = list(rr.top_indices_list)
            remap = _placement_remap(rr)
            if remap is not None:     # expert placement maps logical -> physical ids at pop time: do the same
                ids = [remap(t) for t in ids]
            out.append((layer.token_dispatcher, ids))
            rrs.append(rr)
        self._rr = rrs
        return out

    def index(self, layer_pos):
        rrs = self._rr
        if rrs is None or layer_pos is None or layer_pos >= len(rrs):
            return None
        rr = rrs[layer_pos]
        return core.replay_cursor(self.stage(), rr.forward_index, rr.backward_index)


def _placement_remap(rr):
    """The pop-time id remap of the expert_placement component (ai_search_plugin.adapters.slime_placement), if it is
    active: it re-places experts in fill_routing_replay (before the step's first MoE call, where the plan is built)
    and maps every popped replay tensor with map_routed_ids(layer, ids)."""
    import sys

    sp_mod = sys.modules.get("ai_search_plugin.adapters.slime_placement")
    state = getattr(sp_mod, "_STATE", None) if sp_mod is not None else None
    mgr = state.get("mgr") if isinstance(state, dict) else None
    lid = getattr(rr, "_asrt_layer", None)
    if mgr is None or lid is None:
        return None
    import torch

    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    return lambda t: mgr.map_routed_ids(lid, t.to(dev, non_blocking=True))


def _register_layers() -> bool:
    try:
        from megatron.core.transformer.moe.moe_layer import MoELayer
    except Exception:  # pragma: no cover - needs Megatron
        return False
    return register_layer_class(MoELayer)


def register_layer_class(cls) -> bool:
    if getattr(cls.__init__, "__asrt_r3plan__", False):
        return True
    orig = cls.__init__

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        _LAYERS.append(self)

    __init__.__asrt_r3plan__ = True
    __init__.__wrapped__ = orig
    cls.__init__ = __init__
    return True


def register_models(models) -> int:
    """Alternative to the constructor hook when the model already exists: record its MoE layers in module order."""
    from megatron.core.transformer.moe.moe_layer import MoELayer

    _LAYERS.clear()
    for model in models if isinstance(models, (list, tuple)) else [models]:
        _LAYERS.extend(m for m in model.modules() if isinstance(m, MoELayer))
    return len(_LAYERS)


def _on_clear_all():
    """Step end (every training rank, same point): guard check + stats, then drop the plan."""
    _state["calls"] += 1
    s = mp.check_step()
    every = int(os.environ.get("ASRT_R3PLAN_LOG_EVERY", "10") or 0)
    n = _state["calls"]
    if n <= 2 or (every and n % every == 0) or s.get("step_record", {}).get("action"):
        try:
            import torch.distributed as dist

            rank0 = not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            rank0 = True
        if rank0:
            keep = ("steps", "builds", "build_skipped", "planned_calls", "fallback_calls", "verified_calls",
                    "verify_fail", "mismatch", "build_ms_host", "disabled", "disabled_why", "layout", "why")
            print(f"[asrt] r3plan step {n}: {({k: s.get(k) for k in keep})}", flush=True)
    return s


def patch_routing_replay(RR) -> bool:
    if getattr(RR.clear_all, "__asrt_r3plan__", False):
        return True
    orig = RR.clear_all

    def clear_all():
        try:
            _on_clear_all()
        finally:
            orig()

    clear_all.__asrt_r3plan__ = True
    clear_all.__wrapped__ = orig
    RR.clear_all = staticmethod(clear_all)
    return True


def install(args=None) -> dict:
    """Install the dispatcher patch, the layer registry and the step hook.  Call before the model is built."""
    if _state["installed"]:
        return {"installed": True}
    if not mp.install():
        return {"installed": False, "why": "Megatron alltoall dispatcher not importable"}
    _register_layers()
    try:
        from slime.utils.routing_replay import RoutingReplay

        patch_routing_replay(RoutingReplay)
    except ImportError:
        return {"installed": False, "why": "slime.utils.routing_replay not importable"}
    mp.set_source(SlimeReplaySource())
    if args is not None:
        mp.set_layout(getattr(args, "pipeline_model_parallel_size", 1),
                      getattr(args, "virtual_pipeline_model_parallel_size", None))
    _state["installed"] = True
    return {"installed": True, "layout": mp.snapshot()["layout"]}


def megatron_init(args=None, *rest, **kwargs):
    """slime ``--custom-megatron-init-path`` target (or an entry of ASRT_CHAIN_MEGATRON_INIT).  Acts only with R3
    (``--use-rollout-routing-replay``) and the ``alltoall`` dispatcher; otherwise reports SKIP and changes nothing."""
    if os.environ.get("ASRT_R3PLAN", "1") in ("0", "off"):
        return {"r3plan": "SKIP: ASRT_R3PLAN=0"}
    if args is not None and not getattr(args, "use_rollout_routing_replay", False):
        return {"r3plan": "SKIP: R3 off (--use-rollout-routing-replay not set)"}
    disp = getattr(args, "moe_token_dispatcher_type", "alltoall") if args is not None else "alltoall"
    if disp != "alltoall":
        return {"r3plan": f"SKIP: dispatcher {disp} (only alltoall is planned)"}
    r = install(args)
    return {"r3plan": r}
