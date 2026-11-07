# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""slime glue for expert re-placement (``expert_placement``), no slime source changes.

Enable as a training-actor init hook, either directly

    --custom-megatron-init-path ai_search_plugin.adapters.slime_placement.megatron_init

or chained behind the plugin's own init hook (``ASRT_CHAIN_MEGATRON_INIT``, which ``frameworks.slime`` writes).
Knobs via env (read in every actor):
    ASRT_PLACE=1            turn on (default 1 when the init hook is set; 0 = off)
    ASRT_PLACE_EVERY=1      re-plan every N training steps
    ASRT_PLACE_THRESH=1.05  only move a layer if its EP-rank imbalance >= this
    ASRT_PLACE_STRIDE=4     count every k-th token of the R3 routes (cheap estimate)
    ASRT_PLACE_LOG=path     per-step jsonl (written by global rank 0)
    ASRT_PLACE_RANDOM=1     test only: random permutation of all experts every step
    ASRT_PLACE_MIN_IMB=1.15 auto-off: skip planning when the batch's EP-rank imbalance is below this
    ASRT_PLACE_PROBE_EVERY=4 after a skipped (balanced) check, re-check only every N steps (no counting, no collective)
    ASRT_PLACE_SELFCHECK=raise  self-check failures raise (warn: print only; off)

What it patches (all inside the Megatron actor process):
  1. MegatronTrainRayActor.fill_routing_replay  (R3 on, `--use-rollout-routing-replay`):
     counts the *rollout* routed experts of the batch about to be trained
     (logical ids from SGLang), plans, re-places experts (router rows + expert
     weights + fp32 main params + Adam moments, DistributedOptimizer shards
     included), then lets slime record the replay.  The routes of this very
     batch drive the placement, so the balance is exact for this step.
  2. RoutingReplay.pop_forward / pop_backward: logical -> physical slot ids. Each replay buffer is bound to
     its layer through the router that owns it (router.routing_replay), not by list position; the first
     fill checks bound layers == re-placed layers.
  3. MegatronTrainRayActor.train_actor (R3 off): re-place from the router's own
     counts of the previous step.
  4. megatron_to_hf._convert_to_hf_core (weight sync to SGLang and HF export; a
     module global looked up at call time, so every convert_to_hf object -
     however a model init hook wrapped or re-imported it - goes through it):
     names carry logical expert ids again and router rows are put back in
     logical order, so the rollout engine never sees the placement.
     Self-check at the first fill of every rollout from step 1 on (covers the sync(s)
     since the previous rollout, before their routes are replayed): every MoE layer of
     a converting rank went through the mapping, and somebody converted MoE tensors at
     all; else raise.
  5. MegatronTrainRayActor.save_model: save in logical order (restore, save,
     re-apply) so checkpoints stay portable.
  6. slime restores the "actor" CPU weight backup (by name) inside every
     train_actor(), so the backup is refreshed right after a move.

A dense model (no MoE layers) turns the component off for good ("[asrt] place: no MoE layers -> off").
"""
import json
import os
import sys
import threading
import time

_STATE = {"mgr": None, "installed": False}


def _env(name, default):
    return type(default)(os.environ.get(name, default))


def get_manager():
    return _STATE["mgr"]


def _ensure_manager(actor):
    """The placement manager, or None on a dense model (no MoE layers: placement turns itself off for good)."""
    if _STATE.get("off"):
        return None
    if _STATE["mgr"] is None:
        from .megatron_placement import ExpertPlacementManager, _find_moe_layers
        args = actor.args
        if not getattr(args, "num_experts", None) or not _find_moe_layers(actor.model):
            _STATE["off"] = True
            print("[asrt] place: no MoE layers -> off", flush=True)
            return None
        _STATE["mgr"] = ExpertPlacementManager(actor.model, actor.optimizer, num_layers=args.num_layers,
                                               num_experts=args.num_experts,
                                               threshold=_env("ASRT_PLACE_THRESH", 1.05))
        if not getattr(args, "use_rollout_routing_replay", False):
            _STATE["mgr"].install_router_counters()
    return _STATE["mgr"]


def _log(rec):
    path = os.environ.get("ASRT_PLACE_LOG")
    if not path:
        return
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def _count_rollout_routes(mgr, values, stride):
    import torch
    from slime.utils.routed_experts import materialize_routed_experts

    from ..r3pack import is_lazy_ref, sampled_rows
    t0 = time.perf_counter()
    for v in values:
        if v is None:
            continue
        if is_lazy_ref(v):  # lazy route reference: count a few row chunks, scaled; never load the whole tensor
            a, scale = sampled_rows(v)
            c0 = mgr.counts.clone()
            mgr.add_logical_ids(torch.as_tensor(a).to(mgr.counts.device, non_blocking=True))
            mgr.counts.add_((mgr.counts - c0) * (scale / max(stride, 1) - 1.0))
            continue
        t = materialize_routed_experts(v) if not isinstance(v, torch.Tensor) else v
        t = torch.as_tensor(t)
        if stride > 1:
            t = t[::stride]
        mgr.add_logical_ids(t.to(mgr.counts.device, non_blocking=True))
    return time.perf_counter() - t0


def _after_move(actor, info):
    if info.get("layers", 0) and hasattr(actor, "weights_backuper"):
        # train_actor() always calls _switch_model("actor"), which restores the CPU
        # backup by name -> it would undo the move. Refresh the backup right away.
        tags = set(getattr(actor.weights_backuper, "backup_tags", []))
        if "actor" in tags:
            actor.weights_backuper.backup("actor")


def _live_classes(cls, method):
    """The module class plus Ray's per-actor wrapper subclass (which carries its own
    copies of every method, so patching only the module class is bypassed)."""
    import gc
    out = [cls]
    for o in gc.get_objects():
        try:
            if type(o) is type and o is not cls and type.__getattribute__(o, "__name__") == cls.__name__ \
                    and method in type.__getattribute__(o, "__dict__"):
                out.append(o)
        except Exception:
            continue
    return out


def _patch_method(cls, name, make):
    """make(orig) -> new; applied to every live class defining `name`."""
    for C in _live_classes(cls, name):
        orig = C.__dict__.get(name) or getattr(C, name)
        setattr(C, name, make(orig))


_TLS = threading.local()


def _selfcheck_mode():
    return os.environ.get("ASRT_PLACE_SELFCHECK", "raise")   # raise | warn | off


def _fail(msg):
    msg = "[asrt] placement self-check FAILED: " + msg
    if _selfcheck_mode() == "raise":
        raise RuntimeError(msg)
    print(msg, flush=True)


def _map_export(name, param):
    """Megatron-named tensor on its way to HF conversion -> logical order (no-op while unmoved); records the
    layer for the self-check. Re-entrancy guard: an outer wrapper and the core hook never both map."""
    mgr = _STATE["mgr"]
    if mgr is None or getattr(_TLS, "busy", False):
        return name, param
    _STATE["export_names"] = _STATE.get("export_names", 0) + 1
    return mgr.export_name_and_tensor(name, param)


def _install_export_hook(M2H):
    """Map at slime's conversion *core* (`megatron_to_hf._convert_to_hf_core`), a module global that every
    convert_to_hf - the original function object, whoever imported or wrapped it (e.g. a model init hook that
    gives update_weight_from_distributed its own convert_to_hf wrapper) - looks up at call time. Matching the
    convert_to_hf objects by identity would miss such wrappers: SGLang would then get experts in physical
    order and the R3 ids it returns would be mapped a second time (log-prob drift).
    Fallback for a slime without the core function: wrap every module's *current* convert_to_hf."""
    core = getattr(M2H, "_convert_to_hf_core", None)
    if core is not None and not getattr(core, "_asrt_place", False):
        def _convert_to_hf_core(args, model_name, name, param, *a, **k):
            name2, param2 = _map_export(name, param)
            _TLS.busy = True
            try:
                return core(args, model_name, name2, param2, *a, **k)
            finally:
                _TLS.busy = False
        _convert_to_hf_core._asrt_place = True
        M2H._convert_to_hf_core = _convert_to_hf_core
        _STATE["hook"] = "core"
        return
    _STATE["hook"] = "convert_to_hf"
    _rescan_convert_wrappers()


def _wrap_convert(fn):
    def convert_to_hf(args, model_name, name, param, *a, **k):
        if getattr(_TLS, "busy", False):
            return fn(args, model_name, name, param, *a, **k)
        name2, param2 = _map_export(name, param)
        _TLS.busy = True
        try:
            return fn(args, model_name, name2, param2, *a, **k)
        finally:
            _TLS.busy = False
    convert_to_hf._asrt_place = True
    return convert_to_hf


def _rescan_convert_wrappers():
    """Only for the fallback hook: wrap the convert_to_hf each loaded slime/plugin module holds *now*."""
    if _STATE.get("hook") != "convert_to_hf":
        return
    for mn, m in list(sys.modules.items()):
        f = getattr(m, "convert_to_hf", None) if m is not None else None
        if callable(f) and not getattr(f, "_asrt_place", False) and \
                (mn.startswith("slime") or "convert_to_hf" in getattr(m, "__dict__", {})):
            try:
                setattr(m, "convert_to_hf", _wrap_convert(f))
            except Exception:
                pass


def _bind_replays(mgr, RoutingReplay):
    """Bind each R3 replay buffer to its layer through the router that owns it (slime registers
    `router.routing_replay`), not through list order; check the bound set equals the re-placed set."""
    by_obj = {}
    for lid, moe in mgr.layers:
        rp = getattr(moe.router, "routing_replay", None)
        if rp is not None:
            by_obj[id(rp)] = (rp, lid)
    allrp = list(getattr(RoutingReplay, "all_routing_replays", []))
    if by_obj:
        bound = []
        for rp, lid in by_obj.values():
            rp._asrt_layer = lid
            bound.append(lid)
        stray = [i for i, rp in enumerate(allrp) if id(rp) not in by_obj and getattr(rp, "_asrt_layer", None) is None]
    else:  # replay object not reachable from the router: slime's own order (= ascending global layer id)
        lids = mgr.local_layer_ids()
        if len(allrp) != len(lids):
            _fail(f"R3 replay buffers ({len(allrp)}) != re-placed MoE layers ({len(lids)}: {lids})")
        for rp, lid in zip(allrp, lids):
            rp._asrt_layer = lid
        bound, stray = lids[: len(allrp)], []
    if not _STATE.get("bind_checked"):
        _STATE["bind_checked"] = True
        bad = mgr.check_replay_binding(bound)
        if bad or stray:
            _fail(f"R3 replay layers {sorted(bound)} != re-placed layers {mgr.local_layer_ids()} "
                  f"(mismatch {bad}, unbound replay buffers #{stray})")
        _log({"selfcheck": "r3_bind", "layers": len(bound), "ok": not (bad or stray)})


def _external_exporter():
    """Name of a registered model-specific weight exporter that does its own logical-order mapping, or None.
    Exporters opt in by setting `_asrt_external_export = "<name>"` on slime's UpdateWeightFromDistributed
    (and map expert rows themselves through ``get_manager()``)."""
    m = sys.modules.get("slime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    C = getattr(m, "UpdateWeightFromDistributed", None) if m is not None else None
    if C is None:
        return None
    return getattr(C, "_asrt_external_export", None) or None


def _self_check_export(mgr):
    """After a weight sync: the conversion of every MoE layer of this rank went through our mapping.
    Only the PP-source ranks convert, so: (a) a rank that converted anything must have seen all its MoE
    layers; (b) across the world somebody must have converted MoE tensors at all (else the export path
    bypasses the hook entirely)."""
    if _selfcheck_mode() == "off":
        return
    import torch
    import torch.distributed as dist
    saw = len(mgr.export_seen) > 0
    missing = mgr.check_export(saw_any=saw)
    if dist.is_initialized():
        t = torch.tensor([1.0 if saw else 0.0, float(len(missing))], device=mgr.counts.device)
        dist.all_reduce(t)
        n_saw, n_missing = int(t[0]), int(t[1])
    else:
        n_saw, n_missing = int(saw), len(missing)
    first = not _STATE.get("export_checked")
    _STATE["export_checked"] = True
    if first or n_missing or n_saw == 0:
        _log({"selfcheck": "export", "hook": _STATE.get("hook"), "ranks_converting": n_saw,
              "missing_layers_total": n_missing, "moved": len(mgr.moved_local_layers())})
    if n_saw == 0 and _external_exporter():
        # a model-specific exporter replaced slime's conversion and maps expert rows itself via get_manager()
        if first:
            _log({"selfcheck": "export", "delegated_to": _external_exporter()})
        return
    if n_saw == 0:
        _fail("weight export never passed through the placement hook (hook=%s): the rollout engine would get "
              "experts in physical order after a re-placement" % _STATE.get("hook"))
    elif missing:
        _fail(f"this rank exported weights but MoE layers {missing} never went through the placement mapping "
              f"(seen {sorted(mgr.export_seen)})")
    elif n_missing:
        _fail(f"{n_missing} MoE layer(s) on other ranks missed the placement mapping")


def install():
    if _STATE["installed"]:
        return
    import torch
    from slime.backends.megatron_utils import actor as A
    from slime.backends.megatron_utils import megatron_to_hf as M2H
    from slime.utils import routing_replay as RR

    every = _env("ASRT_PLACE_EVERY", 1)
    stride = _env("ASRT_PLACE_STRIDE", 4)
    cls = A.MegatronTrainRayActor
    step = {"n": 0}

    def _fill_and_bind(self, orig_fill, mgr, data_iterator, num_microbatches, rollout_data):
        out = orig_fill(self, data_iterator, num_microbatches, rollout_data)
        _bind_replays(mgr, RR.RoutingReplay)
        return out

    def make_fill(orig_fill):
        def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
            mgr = _ensure_manager(self)
            if mgr is None:
                return orig_fill(self, data_iterator, num_microbatches, rollout_data)
            if not _STATE.get("bind_checked"):
                _bind_replays(mgr, RR.RoutingReplay)   # self-check before the first move
            if step.get("acted", False):
                # fill_routing_replay can run more than once per rollout (log-prob passes, several train calls)
                # and not the same number of times on every rank: re-place at most once per rollout, at the first
                # call, so every rank takes part in the same collectives
                return _fill_and_bind(self, orig_fill, mgr, data_iterator, num_microbatches, rollout_data)
            step["acted"] = True
            if step["n"] > 0:
                # export self-check of the weight sync(s) since the last rollout (that sync produced the routes
                # about to be replayed). Done here, not in update_weights: slime calls update_weights as a Ray
                # remote method, which does not reliably go through a class patch applied after the actor started.
                _self_check_export(mgr)
                mgr.export_seen.clear()
            rec = {"step": step["n"]}
            # probe / back-off: after a balanced check (placement skipped), do no counting and no collective for
            # the next ASRT_PLACE_PROBE_EVERY-1 steps (all ranks see the same reduced imbalance -> same decision)
            if step.get("skip_left", 0) > 0:
                step["skip_left"] -= 1
                rec["probe_skip"] = step["skip_left"]
            elif step["n"] % every == 0:
                rec["count_s"] = _count_rollout_routes(mgr, rollout_data.get("rollout_routed_experts", []), stride)
                force = None
                if os.environ.get("ASRT_PLACE_RANDOM") == "1":  # stress test: random permutation of every layer
                    g = torch.Generator().manual_seed(4242 + step["n"])
                    force = {lid: torch.randperm(mgr.E, generator=g).tolist() for lid in range(mgr.L)}
                info = mgr.maybe_rebalance(force)
                _after_move(self, info)
                rec.update(info)
                if info.get("skipped") and force is None:
                    step["skip_left"] = max(0, _env("ASRT_PLACE_PROBE_EVERY", 4) - 1)
            out = _fill_and_bind(self, orig_fill, mgr, data_iterator, num_microbatches, rollout_data)
            _log(rec)
            return out
        return fill_routing_replay

    _patch_method(cls, "fill_routing_replay", make_fill)

    def _wrap_pop(fn):
        def pop(self):
            t = fn(self)
            lid = getattr(self, "_asrt_layer", None)
            mgr = _STATE["mgr"]
            if lid is None or mgr is None or t is None:
                return t
            return mgr.map_routed_ids(lid, t)
        return pop

    RR.RoutingReplay.pop_forward = _wrap_pop(RR.RoutingReplay.pop_forward)
    RR.RoutingReplay.pop_backward = _wrap_pop(RR.RoutingReplay.pop_backward)

    def make_train_actor(orig_train_actor):
        def train_actor(self, rollout_id, rollout_data, external_data=None):
            # one step per rollout on every rank (fill_routing_replay may run a different number of times per rank)
            step["acted"] = False
            if not self.args.use_rollout_routing_replay:
                mgr = _ensure_manager(self)
                if mgr is not None and step["n"] > 0 and step["n"] % every == 0:
                    info = mgr.maybe_rebalance()
                    _after_move(self, info)
                    _log({"step": step["n"], **info})
            try:
                return orig_train_actor(self, rollout_id, rollout_data, external_data=external_data)
            finally:
                step["n"] += 1
        return train_actor

    _patch_method(cls, "train_actor", make_train_actor)

    _install_export_hook(M2H)

    def make_update(orig_update):
        def update_weights(self, *a, **k):
            mgr = _STATE["mgr"]
            skip = mgr is None or getattr(self.args, "debug_train_only", False) or \
                getattr(self.args, "debug_rollout_only", False)
            if not skip:
                _rescan_convert_wrappers()   # modules wrapped by model init hooks after install()
            return orig_update(self, *a, **k)
        return update_weights

    if hasattr(cls, "update_weights"):
        _patch_method(cls, "update_weights", make_update)

    def make_save(orig_save):
        def save_model(self, *a, **k):
            mgr = _STATE["mgr"]
            if mgr is None or mgr.is_identity():
                return orig_save(self, *a, **k)
            saved = mgr.slot_of.clone()
            mgr.restore_identity()
            try:
                return orig_save(self, *a, **k)
            finally:
                perms = {lid: saved[lid].tolist() for lid in range(mgr.L)
                         if not torch.equal(saved[lid], torch.arange(mgr.E, device=saved.device))}
                mgr.apply(perms)  # from identity, perm == old slot_of
        return save_model

    _patch_method(cls, "save_model", make_save)
    _STATE["installed"] = True


def megatron_init(args=None, *rest, **kwargs):
    """Entry point for `--custom-megatron-init-path` (or ASRT_CHAIN_MEGATRON_INIT)."""
    if os.environ.get("ASRT_PLACE", "1") == "0":
        return "expert_placement: off (ASRT_PLACE=0)"
    install()
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            from slime.backends.megatron_utils import actor as A
            n = len(_live_classes(A.MegatronTrainRayActor, "fill_routing_replay"))
            print(f"[asrt] slime_placement installed on {n} class object(s)", flush=True)
    except Exception:
        pass
    return "expert_placement: installed"
