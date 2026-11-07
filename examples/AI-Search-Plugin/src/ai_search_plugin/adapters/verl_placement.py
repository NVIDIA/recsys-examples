# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""verl (Megatron engine) glue for expert re-placement (``expert_placement``).

Enabled by `ASRT_VERL=place` (see ai_search_plugin.adapters.verl) or `install()`.
Patches, in the actor worker processes (post-import, no verl source change):

  1. MegatronEngine.initialize: for a trainable engine (not forward_only, i.e.
     not the ref model) with MoE layers, build an ExpertPlacementManager over
     engine.module / engine.optimizer (DistributedOptimizer shards handled) and
     bind every TopKRouter's RouterReplay object to (manager, layer).
  2. TrainingWorker.train_mini_batch (= one `update_actor` call): remembers the
     batch's R3 `routed_experts` (logical ids from the rollout engine).
     MegatronEngine.train_batch (first mini-batch only, inside train_mode so
     params/optimizer are on GPU even with offload): counts those routes
     (R3), or uses the router's own counts of previous steps (no R3), then
     `maybe_rebalance()` (collective over the actor world).
  3. RouterReplay.set_target_indices (R3/R2 replay): logical -> physical slot;
     RouterReplay.record_indices (R2 record): physical -> logical, so every id
     that leaves the model is logical and stays valid across re-placements.
  4. MegatronEngine.get_per_tensor_param (weight sync to the rollout engine and
     HF export through mbridge / Megatron-Bridge): HF names
     `layers.{L}.mlp.experts.{slot}.*` get the logical id back and router
     (`mlp.gate.weight`, `e_score_correction_bias`) rows and stacked expert
     tensors ([E, ...]) are put back in logical order -> the rollout engine
     never sees the placement.
  5. MegatronEngine.save_checkpoint: restore logical order, save, re-apply.

verl has no "restore weights by name from a CPU backup" step (param offload
moves the same tensors to CPU and back, in place), so the slime backup issue
does not arise. Not handled: the delta_sharded checkpoint engine
(get_per_tensor_param_shard exports Megatron-named local shards) - a warning
is printed and placement is frozen (no further moves) when it is used.

Env: ASRT_PLACE_EVERY (default 1), ASRT_PLACE_THRESH (1.05),
ASRT_PLACE_STRIDE (count every k-th R3 token, default 4), ASRT_PLACE_LOG
(jsonl, rank 0), ASRT_PLACE_RANDOM=1 (test: random permutation of all
experts at every re-placement), ASRT_PLACE_MIN_IMB (1.15: auto-off, a batch
whose EP-rank imbalance is below it moves nothing), ASRT_PLACE_PROBE_EVERY (4:
after such a balanced check, no counting / no collective for N-1 steps),
ASRT_PLACE_SELFCHECK (raise | warn | off), ASRT_PLACE_VERIFY* (diagnostics, see place_verify).
A dense model (no MoE layers) gets no manager (placement off).
"""
import json
import os
import re
import time

from ._hooks import patch_method, when_imported

_S = {"installed": False, "step": 0, "frozen": False}


def _env(name, default):
    return type(default)(os.environ.get(name, default))


def _rank0():
    import torch.distributed as dist
    return not dist.is_initialized() or dist.get_rank() == 0


def _log(rec):
    path = os.environ.get("ASRT_PLACE_LOG")
    if path and _rank0():
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    if _rank0():
        print(f"[asrt] verl place {rec}", flush=True)


def _iter_routers(model_chunks):
    for chunk in model_chunks:
        for name, m in chunk.named_modules():
            if "mtp" in name:  # MTP layers restart layer numbering at 1 (alias decoder layers); never re-placed
                continue
            if type(m).__name__.endswith("TopKRouter") or hasattr(m, "router_replay"):
                ln = getattr(m, "layer_number", None)
                if ln is not None and hasattr(m, "weight"):
                    yield ln - 1, m


def _fail(msg):
    msg = "[asrt] verl placement self-check FAILED: " + msg
    if os.environ.get("ASRT_PLACE_SELFCHECK", "raise") == "raise":
        raise RuntimeError(msg)
    print(msg, flush=True)


def _bind(engine, mgr):
    """Bind every RouterReplay to its layer through the re-placed MoE layer's own router (structure, not
    list order or a separately-walked module tree); routers that are not reached that way but carry a
    layer_number are bound by it. Returns the bound layer ids."""
    bound = {}
    for lid, moe in mgr.layers:
        rr = getattr(moe.router, "router_replay", None)
        if rr is not None:
            rr._asrt = (mgr, lid)
            bound[lid] = rr
    for lid, router in _iter_routers(engine.module):
        rr = getattr(router, "router_replay", None)
        if rr is not None and lid not in bound and getattr(rr, "_asrt", None) is None:
            rr._asrt = (mgr, lid)
            bound[lid] = rr
    return sorted(bound)


def build_manager(engine):
    """The placement manager, or None on a dense model (no MoE layers found: placement off)."""
    from .megatron_placement import ExpertPlacementManager, _find_moe_layers
    tf = engine.tf_config
    if not _find_moe_layers(engine.module):
        if _rank0():
            print("[asrt] place: no MoE layers -> off", flush=True)
        return None
    mgr = ExpertPlacementManager(engine.module, engine.optimizer, num_layers=tf.num_layers,
                                 num_experts=getattr(tf, "num_moe_experts", None),
                                 threshold=_env("ASRT_PLACE_THRESH", 1.05))
    mgr._asrt_r3 = getattr(engine.engine_config.router_replay, "mode", "disabled") == "R3"
    if not mgr._asrt_r3:
        mgr.install_router_counters()
    bound = _bind(engine, mgr)
    nb = len(bound)
    if mgr._asrt_r3 and os.environ.get("ASRT_PLACE_SELFCHECK", "raise") != "off":
        bad = mgr.check_replay_binding(bound)
        if bad:
            _fail(f"R3 replay bound to layers {bound} but re-placed layers are {mgr.local_layer_ids()} "
                  f"(mismatch {bad})")
    if _rank0():
        print(f"[asrt] verl placement manager: layers={len(mgr.layers)} E={mgr.E} ep={mgr.ep} "
              f"r3={mgr._asrt_r3} bound_replay={nb}", flush=True)
    return mgr


def _count_routes(mgr, routed, stride):
    import torch
    if routed is None:
        return 0
    t = routed.values() if getattr(routed, "is_nested", False) else routed
    t = torch.as_tensor(t)
    if t.dim() < 3:
        return 0
    t = t.reshape(-1, t.shape[-2], t.shape[-1])
    t = t[(t != 0).any(dim=(1, 2))]  # drop padding rows (all zeros)
    if stride > 1:
        t = t[::stride]
    mgr.add_logical_ids(t.to(mgr.counts.device, non_blocking=True))
    return int(t.shape[0])


def _hf_map(mgr, name, tensor):
    """HF-named export -> logical order."""
    if "mtp" in name:
        return name, tensor
    m = re.search(r"layers\.(\d+)\.mlp\.(?:\w+\.)*?experts\.(\d+)\.", name)
    if m:
        lid, slot = int(m.group(1)), int(m.group(2))
        mgr.export_seen.add(lid)
        if lid < mgr.L and mgr._ident[lid]:
            return name, tensor
        if lid < mgr.L:
            logical = int(mgr.physical_to_logical(lid)[slot])
            return name[: m.start(2)] + str(logical) + name[m.end(2):], tensor
        return name, tensor
    m = re.search(r"layers\.(\d+)\.mlp\.(gate\.weight|gate\.e_score_correction_bias|router\.weight|"
                  r"experts\.[A-Za-z_]+(?:\.weight)?)$", name)
    if m:
        mgr.export_seen.add(int(m.group(1)))
    if m and tensor is not None and int(m.group(1)) < mgr.L:
        lid = int(m.group(1))
        if tensor.dim() >= 1 and tensor.shape[0] == mgr.E:
            if not mgr._ident[lid]:
                tensor = tensor.index_select(0, mgr.slot_of[lid].to(tensor.device))
                _S["export_remapped"] = _S.get("export_remapped", 0) + 1
            from . import place_verify
            place_verify.on_export(mgr, name, tensor, lid, "gate" if "gate.weight" in name or "router" in name
                                   else m.group(2))
        elif not mgr._ident[lid]:
            # an expert / router tensor that is not [E, ...] (e.g. an EP-local chunk) cannot be put back in
            # logical order here -> the rollout engine would get experts in physical order
            _S.setdefault("export_bad", []).append((name, tuple(tensor.shape)))
    return name, tensor


def _checked_export(mgr, gen):
    """Map every exported HF tensor to logical order and, when the export is exhausted, check that every
    MoE layer of this rank went through the mapping (also on the first export, before any move). A rank that
    exported no MoE tensor at all is not checked (it may not hold the full model)."""
    mgr.export_seen.clear()
    n = 0
    for name, t in gen:
        n += 1
        yield _hf_map(mgr, name, t)
    if os.environ.get("ASRT_PLACE_SELFCHECK", "raise") == "off" or not mgr.export_seen:
        return
    first = not _S.get("export_checked")
    _S["export_checked"] = True
    missing = mgr.check_export()
    if first or missing:
        _log({"selfcheck": "export", "tensors": n, "moe_layers_seen": len(mgr.export_seen), "missing": missing})
    if missing:
        _fail(f"export covered MoE layers {sorted(mgr.export_seen)} but not {missing} (HF names not matched -> "
              f"those experts would reach the rollout engine in physical order)")


def install():
    if _S["installed"]:
        return
    _S["installed"] = True

    def on_rr(mod):
        RR = mod.RouterReplay

        def make_set(orig):
            def set_target_indices(self, topk_indices, replay_mask=None):
                b = getattr(self, "_asrt", None)
                if b is not None and topk_indices is not None:
                    topk_indices = b[0].map_routed_ids(b[1], topk_indices)  # no-op while the layer is unmoved
                return orig(self, topk_indices, replay_mask)
            return set_target_indices

        def make_rec(orig):
            def record_indices(self, topk_indices):
                b = getattr(self, "_asrt", None)
                if b is not None and topk_indices is not None and not b[0]._ident[b[1]]:
                    inv = b[0].physical_to_logical(b[1]).to(topk_indices.device)
                    topk_indices = inv[topk_indices.long()].to(topk_indices.dtype)
                return orig(self, topk_indices)
            return record_indices

        patch_method(RR, "set_target_indices", make_set)
        patch_method(RR, "record_indices", make_rec)

    def on_engine(mod):
        E = mod.MegatronEngine

        def make_init(orig):
            def initialize(self, *a, **k):
                out = orig(self, *a, **k)
                self._asrt_mgr = None
                if not self.engine_config.forward_only and getattr(self, "optimizer", None) is not None \
                        and (getattr(self.tf_config, "num_moe_experts", None) or 0) > 0:
                    self._asrt_mgr = build_manager(self)
                return out
            return initialize

        def make_train(orig):
            def train_batch(self, data, loss_function):
                mgr = getattr(self, "_asrt_mgr", None)
                pend = getattr(self, "_asrt_pending", None)
                if mgr is not None and pend is not None and not _S["frozen"]:
                    self._asrt_pending = None
                    step = _S["step"]
                    _S["step"] += 1
                    from . import place_verify
                    if _S.get("export_remapped") or _S.get("export_bad"):
                        # what the last weight sync(s) did: tensors put back in logical order / not remappable
                        _log({"step": step, "export_remapped": _S.pop("export_remapped", 0),
                              "export_bad": _S.pop("export_bad", [])[:8]})
                    pre = place_verify.before_step(mgr, step)
                    every = _env("ASRT_PLACE_EVERY", 1)
                    # probe / back-off (same as slime_placement): after a balanced check (imbalance <
                    # ASRT_PLACE_MIN_IMB, nothing moved) skip counting and the collective for the next
                    # ASRT_PLACE_PROBE_EVERY-1 steps; every rank sees the same all-reduced imbalance -> same decision
                    if _S.get("skip_left", 0) > 0:
                        _S["skip_left"] -= 1
                        if not mgr._asrt_r3:
                            mgr.counts.zero_()  # router counters keep accumulating; start the next probe fresh
                        _log({"step": step, "probe_skip": _S["skip_left"]})
                    elif step % every == 0 and (mgr._asrt_r3 or step > 0):
                        rec = {"step": step}
                        t0 = time.perf_counter()
                        if mgr._asrt_r3:
                            rec["tokens_counted"] = _count_routes(mgr, pend.get("routed"), _env("ASRT_PLACE_STRIDE", 4))
                        rec["count_s"] = time.perf_counter() - t0
                        force = None
                        if os.environ.get("ASRT_PLACE_RANDOM") == "1":
                            import torch
                            g = torch.Generator().manual_seed(4242 + step)
                            force = {lid: torch.randperm(mgr.E, generator=g).tolist() for lid in range(mgr.L)}
                        rec.update(mgr.maybe_rebalance(force))
                        if rec.get("layers"):
                            place_verify.after_move(mgr, step, pre)
                        if rec.get("skipped") and force is None:
                            _S["skip_left"] = max(0, _env("ASRT_PLACE_PROBE_EVERY", 4) - 1)
                        _log(rec)
                return orig(self, data, loss_function)
            return train_batch

        def make_export(orig):
            def get_per_tensor_param(self, *a, **k):
                gen, peft = orig(self, *a, **k)
                mgr = getattr(self, "_asrt_mgr", None)
                if mgr is None:
                    return gen, peft
                return _checked_export(mgr, gen), peft
            return get_per_tensor_param

        def make_shard(orig):
            def get_per_tensor_param_shard(self, *a, **k):
                if getattr(self, "_asrt_mgr", None) is not None and not _S["frozen"]:
                    _S["frozen"] = True
                    if not self._asrt_mgr.is_identity():
                        self._asrt_mgr.restore_identity()
                    print("[asrt] verl placement: delta_sharded export not supported -> placement frozen "
                          "at identity", flush=True)
                return orig(self, *a, **k)
            return get_per_tensor_param_shard

        def make_save(orig):
            def save_checkpoint(self, *a, **k):
                mgr = getattr(self, "_asrt_mgr", None)
                if mgr is None or mgr.is_identity():
                    return orig(self, *a, **k)
                import torch
                saved = mgr.slot_of.clone()
                mgr.restore_identity()
                try:
                    return orig(self, *a, **k)
                finally:
                    ar = torch.arange(mgr.E, device=saved.device)
                    mgr.apply({lid: saved[lid].tolist() for lid in range(mgr.L) if not torch.equal(saved[lid], ar)})
            return save_checkpoint

        def make_infer(orig):
            def infer_batch(self, data, loss_function=None):
                mgr = getattr(self, "_asrt_mgr", None)
                if mgr is None or not os.environ.get("ASRT_PLACE_VERIFY_FWD"):
                    return orig(self, data, loss_function=loss_function)
                from . import place_verify
                return place_verify.forward_ab(self, mgr, orig, data, loss_function)
            return infer_batch

        if hasattr(E, "infer_batch"):
            patch_method(E, "infer_batch", make_infer)
        patch_method(E, "initialize", make_init)
        patch_method(E, "train_batch", make_train)
        patch_method(E, "get_per_tensor_param", make_export)
        if hasattr(E, "get_per_tensor_param_shard"):
            patch_method(E, "get_per_tensor_param_shard", make_shard)
        patch_method(E, "save_checkpoint", make_save)

    def on_workers(mod):
        W = mod.TrainingWorker

        def make_tmb(orig):
            def train_mini_batch(self, data):
                eng = getattr(self, "engine", None)
                if getattr(eng, "_asrt_mgr", None) is not None:
                    routed = None
                    try:
                        routed = data.get("routed_experts", None)
                    except Exception:
                        routed = None
                    eng._asrt_pending = {"routed": routed}
                return orig(self, data)
            return train_mini_batch

        patch_method(W, "train_mini_batch", make_tmb)

    when_imported("verl.utils.megatron.router_replay_patch", on_rr)
    when_imported("verl.workers.engine.megatron.transformer_impl", on_engine)
    when_imported("verl.workers.engine_workers", on_workers)
