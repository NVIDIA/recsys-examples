# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert placement for Megatron-Core MoE layers by *renaming*.

Idea: logical expert e lives in physical slot s = slot_of[e]; EP rank r owns
slots [r*n_local, (r+1)*n_local). Re-placing experts = permuting slots:
  * router: permute the rows of the gating weight (and bias / expert_bias),
    so the logit for an expert is produced at its new slot;
  * experts: move each expert's weights (and its optimizer state: fp32 main
    params + Adam moments, also when they are sharded by Megatron's
    DistributedOptimizer) to the EP rank / local index of its new slot.
Softmax/sigmoid top-k routing is permutation-equivariant, so the layer
computes exactly the same function; the token dispatcher is untouched (it
still maps slot -> rank as slot // n_local). Works with Megatron-Core TopKRouter
+ TEGroupedMLP (per-expert `weight{i}` params or single grouped `weight`) or
SequentialMLP. Not valid with group-limited routing (moe_router_num_groups),
because groups are defined on logical ids.

Components (each usable alone):
  * permute_moe_layer(layer, perm, ep_group, extra=...)    - low level
  * MegatronOptimizerState(optimizer)                      - full/sharded optimizer state access
  * ExpertPlacementManager(model_chunks, optimizer, ...)   - the pluggable piece for a training loop:
        counts (router hook or R3 logical ids) -> planner (pure Python, ai_search_plugin.placement_py)
        -> rename, plus logical<->physical maps for routing replay and weight export.

Auto-off: ``maybe_rebalance`` skips planning and moving when the batch's mean EP-rank imbalance is below
``ASRT_PLACE_MIN_IMB`` (default 1.15); the framework glue then backs off for a few steps.

perm convention: perm[old_slot] = new_slot (a permutation of range(E)).

torch / megatron are imported lazily (inside the functions), so this module imports on a machine without them.
"""
from __future__ import annotations

import functools
import os
import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def _no_grad(fn):
    """``torch.no_grad()`` as a decorator, with torch imported at call time."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        import torch
        with torch.no_grad():
            return fn(*a, **k)
    return wrapper


def _dist():
    import torch.distributed as dist
    return dist


# ----------------------------------------------------------------------------- perms

def placement_to_perm(cur_rank_of: Sequence[int], new_rank_of: Sequence[int], n_local: int) -> List[int]:
    """Turn planner output (expert -> EP rank, before/after; equal experts per
    rank) into a slot permutation that keeps experts already on the right rank
    in place where possible. Here 'expert' == current slot."""
    E = len(cur_rank_of)
    ranks = E // n_local
    perm = [-1] * E
    free = {r: list(range(r * n_local, (r + 1) * n_local)) for r in range(ranks)}
    stay = [c for c in range(E) if new_rank_of[c] == c // n_local]
    for c in stay:
        perm[c] = c
        free[c // n_local].remove(c)
    for c in range(E):
        if perm[c] < 0:
            perm[c] = free[new_rank_of[c]].pop(0)
    assert sorted(perm) == list(range(E))
    return perm


def _inverse(perm: Sequence[int]) -> List[int]:
    inv = [0] * len(perm)
    for c, p in enumerate(perm):
        inv[p] = c
    return inv


# ----------------------------------------------------------------------------- router

@_no_grad
def permute_router(router, perm: Sequence[int], opt_state=None) -> None:
    import torch
    idx = torch.tensor(_inverse(perm), device=router.weight.device)  # new[p] = old[inv[p]]
    params = [router.weight]
    bias = getattr(router, "bias", None)
    if isinstance(bias, torch.Tensor) and bias.numel() >= len(perm):
        params.append(bias)
    for p in params:
        if opt_state is not None:
            full = opt_state.full(p)
            for k in full:
                full[k] = full[k].index_select(0, idx)
            opt_state.write_back(p, full)
        p.data.copy_(p.data.index_select(0, idx))
    for name in ("expert_bias", "local_tokens_per_expert"):
        t = getattr(router, name, None)
        if isinstance(t, torch.Tensor) and t.numel() == len(perm):
            t.copy_(t.index_select(0, idx.to(t.device)))


# ----------------------------------------------------------------------------- experts

def _expert_tensors(experts) -> Dict[str, list]:
    """name -> list of per-local-expert tensors (views) for every expert param."""
    import torch
    out: Dict[str, list] = {}
    n = experts.num_local_experts
    if hasattr(experts, "local_experts"):  # SequentialMLP
        for j, mlp in enumerate(experts.local_experts):
            for name, p in mlp.named_parameters():
                out.setdefault(name, [None] * n)[j] = p
        return out
    for lname in ("linear_fc1", "linear_fc2"):
        lin = getattr(experts, lname)
        for kind in ("weight", "bias"):
            single = getattr(lin, kind, None)
            if isinstance(single, torch.Tensor) and single.dim() >= 1 and single.shape[0] == n and \
                    getattr(lin, f"{kind}0", None) is None:
                out[f"{lname}.{kind}"] = [single[j] for j in range(n)]
            else:
                ts = [getattr(lin, f"{kind}{j}", None) for j in range(n)]
                if all(isinstance(t, torch.Tensor) for t in ts):
                    out[f"{lname}.{kind}"] = ts
    return out


def _expert_params(experts) -> Dict[str, Tuple[list, bool]]:
    """name -> (list of parameter objects, grouped?) where grouped means one
    parameter holds all local experts along dim 0."""
    import torch
    out = {}
    n = experts.num_local_experts
    if hasattr(experts, "local_experts"):
        for j, mlp in enumerate(experts.local_experts):
            for name, p in mlp.named_parameters():
                out.setdefault(name, ([None] * n, False))[0][j] = p
        return out
    for lname in ("linear_fc1", "linear_fc2"):
        lin = getattr(experts, lname)
        for kind in ("weight", "bias"):
            single = getattr(lin, kind, None)
            if isinstance(single, torch.nn.Parameter) and single.dim() >= 1 and single.shape[0] == n and \
                    getattr(lin, f"{kind}0", None) is None:
                out[f"{lname}.{kind}"] = ([single], True)
            else:
                ts = [getattr(lin, f"{kind}{j}", None) for j in range(n)]
                if all(isinstance(t, torch.nn.Parameter) for t in ts):
                    out[f"{lname}.{kind}"] = (ts, False)
    return out


def _moved_local(perm: Sequence[int], n_local: int, me: int) -> List[int]:
    return [j for j in range(n_local) if perm[me * n_local + j] != me * n_local + j]


@_no_grad
def move_local_expert_tensors(local, perm: Sequence[int], n_local: int, ep_group=None) -> int:
    """In-place: after the call, local[j] holds the tensor of the expert whose
    new slot is (my_rank * n_local + j). `local` is a list (all local experts)
    or a dict {local_idx: tensor} that must contain at least every local index
    whose slot changes. Only changed slots are sent (one all_to_all over the
    EP group). Returns bytes sent by this rank."""
    import torch
    dist = _dist()
    world = dist.get_world_size(ep_group) if dist.is_initialized() else 1
    me = dist.get_rank(ep_group) if dist.is_initialized() else 0
    inv = _inverse(perm)
    moved = _moved_local(perm, n_local, me)
    get = local.__getitem__
    if isinstance(local, dict):
        any_t = next(iter(local.values())) if local else None
    else:
        any_t = local[0]
    if any_t is None:
        if world > 1:  # still have to take part in the collective
            raise ValueError("need at least one tensor to infer shape")
        return 0
    shape, dtype, dev = any_t.shape, any_t.dtype, any_t.device
    send_lists = [[] for _ in range(world)]
    for j in moved:
        c = me * n_local + j
        send_lists[perm[c] // n_local].append((perm[c], j))
    by_src = [[] for _ in range(world)]
    for j in moved:
        g = me * n_local + j
        by_src[inv[g] // n_local].append(g)
    if world == 1:
        new = {j: get(inv[me * n_local + j] - me * n_local).clone() for j in moved}
        for j, t in new.items():
            get(j).copy_(t)
        return 0
    send, sent = [], 0
    for d in range(world):
        items = sorted(send_lists[d])
        if items:
            send.append(torch.stack([get(j) for _, j in items]))
            sent += send[-1].numel() * send[-1].element_size()
        else:
            send.append(torch.empty((0, *shape), dtype=dtype, device=dev))
    recv = [torch.empty((len(by_src[s]), *shape), dtype=dtype, device=dev) for s in range(world)]
    dist.all_to_all(recv, send, group=ep_group)
    for s in range(world):  # from source s, tensors arrive ordered by new slot
        for k, g in enumerate(sorted(by_src[s])):
            get(g - me * n_local).copy_(recv[s][k])
    return sent


# ----------------------------------------------------------------------------- optimizer state

class MegatronOptimizerState:
    """Uniform access to the (fp32 main param + moments) state of a model
    param, for plain torch optimizers, Megatron Float16 optimizers and
    Megatron DistributedOptimizer (state sharded over its DP group; `full`
    rebuilds the whole tensor with one all_reduce per key, `write_back`
    keeps only this rank's shard). Keys always come in the same order on all
    ranks so the collectives line up."""

    def __init__(self, optimizer):
        self.opts = list(getattr(optimizer, "chained_optimizers", None) or [optimizer])
        self._owner: Dict[int, object] = {}
        for o in self.opts:
            for p in self._params_of(o):
                self._owner[id(p)] = o

    @staticmethod
    def _hybrid(o):
        """Megatron's CPU-offload HybridDeviceOptimizer behind a DistributedOptimizer (--optimizer-cpu-offload),
        or None.  Its real state lives in the CPU/GPU sub-optimizers; hdo.state is only a mirror rebuilt after
        every step, and with the precision-aware optimizer it has no 'master_param' unless param_update_in_fp32
        (Megatron's own getter then raises KeyError 'master_param')."""
        inner = getattr(o, "optimizer", None)
        return inner if type(inner).__name__ == "HybridDeviceOptimizer" else None

    def _hybrid_local(self, o, p):
        """(tensors, range) for p's shard under HybridDeviceOptimizer: the sub-optimizer's own fp32/CPU inner
        param is the main param, its moments are the optimizer state.  Tensors are the live objects (written in
        place by write_back), on whatever device they live (CPU when offloaded)."""
        import torch
        hdo = self._hybrid(o)
        gi, go = o.model_param_group_index_map[p]
        sharded = hdo.param_groups[gi]["params"][go]
        inner = hdo.param_to_inner_param.get(sharded, sharded)
        st = {}
        for sub in hdo.sub_optimizers:
            if inner in sub.state:
                st = {k: v for k, v in sub.state[inner].items() if isinstance(v, torch.Tensor) and k != "master_param"}
                break
        st["param"] = inner
        r = o._get_model_param_range_map(p)["param"]
        return st, (r.start, r.end)

    @staticmethod
    def _is_distopt(o) -> bool:
        return hasattr(o, "_get_model_param_range_map") and hasattr(o, "model_param_gbuf_map")

    def _params_of(self, o) -> Iterable:
        if self._is_distopt(o):
            for buf in getattr(o, "buffers", []):
                yield from buf.param_index_map.keys()
        elif hasattr(o, "float16_groups"):
            for g in o.float16_groups:
                yield from g
            for g in getattr(o, "fp32_from_fp32_groups", []):
                yield from g
        else:
            inner = getattr(o, "optimizer", o)
            for g in inner.param_groups:
                yield from g["params"]

    def keys(self, o) -> List[str]:
        if self._is_distopt(o):
            return ["param"] + list(o.optimizer_state_keys)
        return ["param", "exp_avg", "exp_avg_sq"]

    def _local(self, o, p) -> Tuple[Optional[dict], Optional[Tuple[int, int]]]:
        """(tensors, (start, end) in flattened p) for the part of p this rank holds."""
        import torch
        if self._is_distopt(o):
            if p not in o.model_param_group_index_map:
                return None, None
            if self._hybrid(o) is not None:
                return self._hybrid_local(o, p)
            r = o._get_model_param_range_map(p)["param"]
            return o._get_main_param_and_optimizer_states(p), (r.start, r.end)
        if hasattr(o, "float16_groups"):
            for g16, g32 in zip(o.float16_groups, o.fp32_from_float16_groups):
                for a, b in zip(g16, g32):
                    if a is p:
                        st = dict(o.optimizer.state.get(b, {}))
                        st = {k: v for k, v in st.items() if isinstance(v, torch.Tensor)}
                        st["param"] = b
                        return st, (0, p.numel())
            inner = o.optimizer
            st = {k: v for k, v in inner.state.get(p, {}).items() if isinstance(v, torch.Tensor)}
            return st, (0, p.numel())  # fp32 param: no separate main copy
        inner = getattr(o, "optimizer", o)
        st = {k: v for k, v in inner.state.get(p, {}).items() if isinstance(v, torch.Tensor)}
        return st, (0, p.numel())

    def full(self, p) -> Dict[str, object]:
        import torch
        dist = _dist()
        o = self._owner.get(id(p))
        if o is None:
            return {}
        tensors, rng = self._local(o, p)
        group = o.data_parallel_group if self._is_distopt(o) else None
        dp = dist.get_world_size(group) if (group is not None and dist.is_initialized()) else 1
        out = {}
        for k in self.keys(o):
            if k == "param" and not self._is_distopt(o) and not hasattr(o, "float16_groups"):
                continue  # plain optimizer: the model param is the main param
            t = None if tensors is None else tensors.get(k)
            if dp == 1:
                if t is None or t.numel() != p.numel():  # lazily-created state: move zeros
                    out[k] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                else:
                    # offloaded state lives on CPU
                    out[k] = t.detach().to(device=p.device, dtype=torch.float32).reshape(p.shape).clone()
                continue
            buf = torch.zeros(p.numel(), dtype=torch.float32, device=p.device)
            if t is not None and rng is not None:
                buf[rng[0]:rng[1]].copy_(t.detach().reshape(-1).float())
            dist.all_reduce(buf, group=group)
            out[k] = buf.view(p.shape)
        return out

    def write_back(self, p, full: Dict[str, object]) -> None:
        import torch
        o = self._owner.get(id(p))
        if o is None or not full:
            return
        tensors, rng = self._local(o, p)
        if tensors is None:
            return
        if self._is_distopt(o) and self._hybrid(o) is not None:
            # write the live sub-optimizer tensors in place (hdo.state is only a mirror, Megatron's setter
            # would write the mirror and be overwritten at the next step)
            for k, v in tensors.items():
                if k in full and isinstance(v, torch.Tensor):
                    v.data.copy_(full[k].reshape(-1)[rng[0]:rng[1]].to(device=v.device, dtype=v.dtype).view_as(v))
            return
        if self._is_distopt(o):
            new = {}
            for k, v in tensors.items():
                if k in full and isinstance(v, torch.Tensor):
                    new[k] = full[k].reshape(-1)[rng[0]:rng[1]].to(v.dtype).view_as(v)
            if new:
                for k in tensors:  # keep any key we did not rebuild
                    new.setdefault(k, tensors[k])
                o._set_main_param_and_optimizer_states(p, new)
            return
        for k, v in tensors.items():
            if k in full and isinstance(v, torch.Tensor):
                v.copy_(full[k].reshape(v.shape).to(v.dtype))


def adam_state_extra(optimizer):
    """`extra` callback for a plain torch Adam/AdamW: moves exp_avg/exp_avg_sq
    (and fp32 master copies if the optimizer keeps them per param)."""
    def extra(name: str, params: list) -> List[list]:
        import torch
        out = []
        for key in ("exp_avg", "exp_avg_sq"):
            st = [optimizer.state.get(p, {}).get(key) for p in params]
            if all(isinstance(s, torch.Tensor) for s in st):
                out.append(st)
        return out
    return extra


@_no_grad
def permute_moe_layer(moe_layer, perm: Sequence[int], ep_group=None,
                      extra: Optional[Callable[[str, list], List[list]]] = None,
                      opt_state: Optional[MegatronOptimizerState] = None) -> int:
    """Apply slot permutation to one MoELayer. Optimizer state travels with
    the weights either through `opt_state` (Megatron optimizers incl.
    DistributedOptimizer) or the older `extra(name, params)` callback.
    Only experts whose slot changes are touched. Returns bytes sent."""
    dist = _dist()
    experts = moe_layer.experts
    n_local = experts.num_local_experts
    if ep_group is None:
        ep_group = getattr(experts, "ep_group", None)
    me = dist.get_rank(ep_group) if dist.is_initialized() else 0
    permute_router(moe_layer.router, perm, opt_state)
    moved = _moved_local(perm, n_local, me)
    sent = 0
    if opt_state is None:
        for name, ts in _expert_tensors(experts).items():
            for g in [ts] + (extra(name, ts) if extra else []):
                sent += move_local_expert_tensors(g, perm, n_local, ep_group)
        return sent
    for name, (params, grouped) in _expert_params(experts).items():
        # weights (bf16 model params, full on every DP rank)
        views = [params[0][j] for j in range(n_local)] if grouped else params
        sent += move_local_expert_tensors(views, perm, n_local, ep_group)
        # optimizer state: rebuild full tensors only for moved experts
        if grouped:
            full = opt_state.full(params[0])
            for k, t in full.items():
                sent += move_local_expert_tensors([t[j] for j in range(n_local)], perm, n_local, ep_group)
            opt_state.write_back(params[0], full)
        else:
            fulls = {j: opt_state.full(params[j]) for j in moved}
            keys = sorted(next(iter(fulls.values())).keys()) if fulls else []
            # all ranks must run the same number of all_to_alls: agree on the key list
            keys = _agree_keys(keys, ep_group, params[0].device)
            for k in keys:
                sent += _move_sparse({j: fulls[j][k] for j in moved}, params[0], perm, n_local, ep_group)
            for j in moved:
                opt_state.write_back(params[j], fulls[j])
    return sent


_KEY_ORDER = ["param", "exp_avg", "exp_avg_sq", "master_param"]


def _agree_keys(keys: List[str], group, device=None) -> List[str]:
    import torch
    dist = _dist()
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return keys
    # the device of the expert params (a CUDA tensor for NCCL, CPU for gloo)
    dev = device if device is not None else torch.cuda.current_device()
    mask = torch.tensor([1 if k in keys else 0 for k in _KEY_ORDER], device=dev)
    dist.all_reduce(mask, op=dist.ReduceOp.MAX, group=group)
    return [k for k, m in zip(_KEY_ORDER, mask.tolist()) if m]


def _move_sparse(local: dict, like, perm, n_local, group) -> int:
    import torch
    dist = _dist()
    if not local:  # nothing moves here, but the collective still needs a shape
        world = dist.get_world_size(group) if dist.is_initialized() else 1
        if world == 1:
            return 0
        send = [torch.empty((0, *like.shape), dtype=torch.float32, device=like.device) for _ in range(world)]
        recv = [torch.empty((0, *like.shape), dtype=torch.float32, device=like.device) for _ in range(world)]
        dist.all_to_all(recv, send, group=group)
        return 0
    return move_local_expert_tensors(local, perm, n_local, group)


# ----------------------------------------------------------------------------- manager

def _is_moe(mod) -> bool:
    import torch
    return isinstance(mod, torch.nn.Module) and isinstance(getattr(mod, "router", None), torch.nn.Module) \
        and hasattr(getattr(mod, "experts", None), "num_local_experts")


def _find_moe_layers(model_chunks) -> list:
    """(global 0-based layer index, MoELayer) for every decoder MoE layer on this rank.

    Matched by structure and name, never by the object a caller happens to hold: every module that owns a
    `router` and an `experts` (with num_local_experts) counts, wherever it sits under `...layers.<i>`
    (plain `.mlp`, or wrapped one level deeper such as `.mlp.module` / `.mlp.inner` by a model-specific
    init hook). The layer id comes from the enclosing decoder layer's `layer_number` (global, PP/VPP-aware),
    else from the `layers.<i>` index in the name. DDP / Float16Module / VPP chunk lists are transparent
    (named_modules walks through them). MTP layers are skipped (their numbering restarts)."""
    import torch
    out = {}
    if isinstance(model_chunks, torch.nn.Module):
        model_chunks = [model_chunks]
    for chunk in model_chunks:
        mods = dict(chunk.named_modules())
        for name, mod in mods.items():
            if "mtp" in name or not _is_moe(mod):
                continue
            m = None
            for m in re.finditer(r"(?:^|\.)layers\.(\d+)(?=\.|$)", name):
                pass
            if m is None:
                continue
            parent = mods.get(name[: m.end()].lstrip("."))
            ln = getattr(parent, "layer_number", None)
            lid = (ln if isinstance(ln, int) else int(m.group(1)) + 1) - 1
            prev = out.get(lid)
            if prev is not None and prev is not mod and prev.experts is not mod.experts:
                raise RuntimeError(f"[asrt] placement: two different MoE modules claim layer {lid} ({name})")
            out.setdefault(lid, mod)
    return sorted(out.items(), key=lambda x: x[0])


# HF / Megatron export names: the MoE block may sit one wrapper deeper than `.mlp` (e.g. `.mlp.module.experts`)
_RX_EXPERT = re.compile(r"layers\.(\d+)\.mlp\.(?:\w+\.)*?experts\.(.+)\.(weight|bias)(\d+)$")
_RX_ROUTER = re.compile(r"layers\.(\d+)\.mlp\.(?:\w+\.)*?router\.(weight|bias|expert_bias)$")
_RX_MOE_ANY = re.compile(r"layers\.(\d+)\.mlp\.(?:\w+\.)*?(?:experts|router)\.")


def _min_imb() -> float:
    return float(os.environ.get("ASRT_PLACE_MIN_IMB", "1.15"))


class ExpertPlacementManager:
    """Drop-in expert re-placement for a Megatron training loop.

        mgr = ExpertPlacementManager(model, optimizer, num_layers=cfg.num_layers)
        mgr.install_router_counters()         # or feed R3 ids via add_logical_ids()
        ... each step, before forward/backward:
        mgr.maybe_rebalance()                 # collective over the whole world

    State kept: slot_of[layer][logical] (cumulative placement). Helpers map
    R3 logical ids to physical slots and export weights back in logical order.
    """

    def __init__(self, model_chunks, optimizer, num_layers: int, num_experts: Optional[int] = None,
                 ep_group=None, threshold: float = 1.03, max_swaps: int = 256, min_gain_s: float = 0.0,
                 count_group=None):
        import torch
        from megatron.core import parallel_state as ps
        self.model = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
        self.opt_state = MegatronOptimizerState(optimizer) if optimizer is not None else None
        self.layers = _find_moe_layers(self.model)
        if not self.layers:
            raise ValueError("no MoE layers found")
        ex = self.layers[0][1].experts
        self.E = num_experts or ex.num_local_experts * ps.get_expert_model_parallel_world_size()
        self.n_local = ex.num_local_experts
        self.ep = self.E // self.n_local
        self.ep_group = ep_group or ps.get_expert_model_parallel_group()
        self.L = num_layers
        self.threshold, self.max_swaps = threshold, max_swaps
        self.count_group = count_group  # None = world
        dev = torch.cuda.current_device()
        self.counts = torch.zeros(self.L, self.E, dtype=torch.float64, device=dev)  # logical
        self.slot_of = torch.arange(self.E, device=dev).repeat(self.L, 1)          # logical -> slot
        self._ident = [True] * self.L     # per layer: still in logical order -> R3 ids need no mapping
        self._slot_cache = {}             # (layer, device) -> slot table on that device (no per-call D2H/H2D)
        self._hooks = []
        self.history: List[dict] = []
        self.export_seen = set()          # layer ids whose MoE tensors went through export_name_and_tensor

    # --- counting -----------------------------------------------------------
    def install_router_counters(self):
        """Count tokens per expert from the router's routing_map (physical slots)."""
        import torch
        for lid, mod in self.layers:
            def hook(m, inp, out, lid=lid):
                rm = out[1] if isinstance(out, tuple) and len(out) > 1 else None
                if rm is not None and torch.is_grad_enabled():  # count training forwards only
                    phys = rm.sum(0).double()
                    self.counts[lid] += phys.index_select(0, self.slot_of[lid])  # slot -> logical
            self._hooks.append(mod.router.register_forward_hook(hook))

    def add_logical_ids(self, routed) -> None:
        """R3 / rollout routing: `routed` [tokens, layers, topk] of logical ids
        (np array or tensor). Counts all layers (the world all_reduce later
        makes every rank see the global counts)."""
        import torch
        t = torch.as_tensor(routed, device=self.counts.device)
        L = min(t.shape[1], self.L)
        # one indexed add for all layers: ids + layer * E into the flattened [L, E] counts
        off = (torch.arange(L, device=t.device, dtype=torch.long) * self.E).view(1, L, 1)
        flat = (t[:, :L, :].long() + off).reshape(-1)
        self.counts.view(-1).index_add_(0, flat, torch.ones(flat.numel(), dtype=self.counts.dtype, device=t.device))

    def add_logical_counts(self, layer: int, counts) -> None:
        import torch
        self.counts[layer] += torch.as_tensor(counts, dtype=torch.float64, device=self.counts.device)

    # --- maps ---------------------------------------------------------------
    def logical_to_physical(self, layer: int):
        return self.slot_of[layer]

    def physical_to_logical(self, layer: int):
        import torch
        inv = torch.empty_like(self.slot_of[layer])
        inv[self.slot_of[layer]] = torch.arange(self.E, device=inv.device)
        return inv

    def is_identity(self) -> bool:
        import torch
        return bool((self.slot_of == torch.arange(self.E, device=self.slot_of.device)).all())

    # --- planning + moving ---------------------------------------------------
    def plan(self, counts, force_perms: Optional[Dict[int, List[int]]] = None):
        """Returns {layer: perm over slots}. Runs on rank 0 with the all-reduced counts; the result is broadcast."""
        from ..placement_py import imbalance, rebalance_swaps
        if force_perms is not None:
            return force_perms
        perms = {}
        c = counts.detach().cpu().tolist()
        for lid in range(self.L):
            if sum(c[lid]) <= 0:
                continue
            slot_of = self.slot_of[lid].cpu().tolist()
            rank_of = [s // self.n_local for s in slot_of]
            if imbalance(c[lid], rank_of, self.ep) < self.threshold:
                continue
            new_rank_of, moved = rebalance_swaps(c[lid], rank_of, self.ep, self.max_swaps)
            if moved == 0:
                continue
            # planner works on logical experts; convert to slots
            cur_rank_of_slot = [s // self.n_local for s in range(self.E)]
            new_rank_of_slot = [0] * self.E
            for e, s in enumerate(slot_of):
                new_rank_of_slot[s] = new_rank_of[e]
            perms[lid] = placement_to_perm(cur_rank_of_slot, new_rank_of_slot, self.n_local)
        return perms

    def _sync_params(self):
        for chunk in self.model:
            if hasattr(chunk, "start_param_sync") and getattr(getattr(chunk, "ddp_config", None),
                                                              "overlap_param_gather", False):
                chunk.start_param_sync(force_sync=True)

    @_no_grad
    def apply(self, perms: Dict[int, List[int]]) -> dict:
        import torch
        t0 = time.perf_counter()
        self._sync_params()
        sent = 0
        local = dict(self.layers)
        for lid in sorted(perms):
            perm = perms[lid]
            if lid in local:
                sent += permute_moe_layer(local[lid], perm, self.ep_group, opt_state=self.opt_state)
            p = torch.tensor(perm, device=self.slot_of.device)
            self.slot_of[lid] = p[self.slot_of[lid]]
            self._ident[lid] = bool((self.slot_of[lid] == torch.arange(self.E, device=self.slot_of.device)).all())
        self._slot_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {"layers": len(perms), "bytes_sent": sent, "apply_s": time.perf_counter() - t0}

    def _broadcast_perms(self, perms: Dict[int, List[int]]) -> Dict[int, List[int]]:
        import torch
        dist = _dist()
        if not dist.is_initialized():
            return perms
        t = torch.full((self.L, self.E), -1, dtype=torch.long, device=self.counts.device)
        if dist.get_rank() == 0:
            for lid, p in perms.items():
                t[lid] = torch.tensor(p, device=t.device)
        dist.broadcast(t, src=0)
        return {lid: t[lid].tolist() for lid in range(self.L) if t[lid, 0] >= 0}

    def maybe_rebalance(self, force_perms: Optional[Dict[int, List[int]]] = None) -> dict:
        """Collective (all ranks): reduce counts, plan, move. Clears counts."""
        import torch
        dist = _dist()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        counts = self.counts.clone()
        if dist.is_initialized():
            dist.all_reduce(counts, group=self.count_group)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        # time spent here is mostly waiting for the slowest rank to reach this first collective of the step
        # (e.g. its data fetch), not placement work - logged separately so it is not mistaken for planning
        t_wait = time.perf_counter() - t0
        self.last_counts = counts   # global logical counts of this step
        t0 = time.perf_counter()
        rank_of = (self.slot_of // self.n_local)
        imb_before = self._imbalance(counts, rank_of)
        # auto-off: nothing worth fixing -> skip planning (rank-0 planning costs time while all ranks wait)
        min_imb = _min_imb()
        skip = force_perms is None and imb_before < min_imb
        if dist.is_initialized() and dist.get_world_size() > 1:
            # one decision for everybody: rank 0's (a rank that decided differently would enter / skip the
            # broadcast + all_to_all of the move alone and hang the step)
            d = torch.tensor([1.0 if skip else 0.0, imb_before], dtype=torch.float64, device=self.counts.device)
            dist.broadcast(d, src=0)
            skip, imb_before = bool(d[0].item() > 0.5), float(d[1].item())
        if skip:
            self.counts.zero_()
            info = {"layers": 0, "bytes_sent": 0, "apply_s": 0.0, "plan_s": time.perf_counter() - t0,
                    "wait_peers_s": t_wait, "imbalance_before": imb_before, "imbalance_after_pred": imb_before,
                    "experts_moved": 0, "skipped": f"imbalance {imb_before:.3f} < {min_imb}"}
            self.history.append(info)
            return info
        perms = self.plan(counts, force_perms) if (not dist.is_initialized() or dist.get_rank() == 0) else {}
        perms = self._broadcast_perms(perms)
        t_plan = time.perf_counter() - t0
        info = self.apply(perms) if perms else {"layers": 0, "bytes_sent": 0, "apply_s": 0.0}
        info.update(plan_s=t_plan, wait_peers_s=t_wait, imbalance_before=imb_before,
                    imbalance_after_pred=self._imbalance(counts, self.slot_of // self.n_local),
                    experts_moved=int(sum(sum(1 for c, p in enumerate(v) if c != p) for v in perms.values())))
        self.counts.zero_()
        self.history.append(info)
        return info

    def _imbalance(self, counts, rank_of) -> float:
        """Mean over layers (with any load) of max/mean EP-rank load."""
        import torch
        loads = torch.zeros(self.L, self.ep, dtype=torch.float64, device=counts.device)
        loads.scatter_add_(1, rank_of.long().to(counts.device), counts)
        tot = loads.sum(1)
        ok = tot > 0
        if not bool(ok.any()):
            return 1.0
        return float((loads[ok].max(1).values / loads[ok].mean(1)).mean())

    # --- integration helpers -----------------------------------------------------
    def map_routed_ids(self, layer: int, ids):
        """R3: logical expert ids (from the rollout engine) -> physical slots.
        Called for every layer of every micro-batch (forward, backward, recompute), so it must be cheap:
        layers still in logical order return the ids untouched, and the slot table is cached per device
        (moving it on every call would be a blocking GPU<->CPU copy that stalls the training stream)."""
        if self._ident[layer]:
            return ids
        key = (layer, ids.device)
        tab = self._slot_cache.get(key)
        if tab is None:
            tab = self._slot_cache[key] = self.slot_of[layer].to(ids.device)
        return tab[ids.long()].to(ids.dtype)

    def export_name_and_tensor(self, name: str, tensor):
        """Weight export (e.g. slime update_weights / HF conversion): global
        Megatron names carry physical slot ids; return logical names and
        router rows in logical order. Name formats handled:
        '...layers.{L}.mlp[.<wrapper>].experts.<rest>.weight{slot}' and '...layers.{L}.mlp[.<wrapper>].router.weight'.
        Every MoE name that passes here is recorded in `export_seen` (self-check: see check_export)."""
        if "mtp" in name:
            return name, tensor
        m = _RX_EXPERT.search(name)
        if m:
            lid, slot = int(m.group(1)), int(m.group(4))
            self.export_seen.add(lid)
            if self._ident[lid] if lid < self.L else True:
                return name, tensor
            logical = int(self.physical_to_logical(lid)[slot])
            return name[: m.start(4)] + str(logical), tensor
        m = _RX_ROUTER.search(name)
        if m:
            lid = int(m.group(1))
            self.export_seen.add(lid)
            if lid < self.L and not self._ident[lid] and tensor.dim() >= 1 and tensor.shape[0] == self.E:
                return name, tensor.index_select(0, self.slot_of[lid].to(tensor.device))
            return name, tensor
        m = _RX_MOE_ANY.search(name)
        if m:
            self.export_seen.add(int(m.group(1)))
        return name, tensor

    # --- self-check: re-placed layers == layers the export path sees == layers the R3 replay path binds ----
    def local_layer_ids(self) -> List[int]:
        return [lid for lid, _ in self.layers]

    def moved_local_layers(self) -> List[int]:
        return [lid for lid, _ in self.layers if not self._ident[lid]]

    def check_export(self, where: str = "export", saw_any: Optional[bool] = None) -> List[int]:
        """After a full export on a rank that converted weights: every local MoE layer must have gone through
        export_name_and_tensor. Returns the missing layer ids (empty = ok); callers raise."""
        if saw_any is False:
            return []
        return sorted(set(self.local_layer_ids()) - self.export_seen)

    def check_replay_binding(self, bound_layers) -> List[int]:
        """R3: the set of layers whose replay buffer is bound to this manager must equal the re-placed set."""
        return sorted(set(self.local_layer_ids()) ^ set(bound_layers))

    def restore_identity(self) -> dict:
        """Move everything back to logical order (e.g. before saving a checkpoint)."""
        perms = {}
        for lid in range(self.L):
            inv = self.physical_to_logical(lid).tolist()  # slot -> logical == the perm back
            if any(i != s for s, i in enumerate(inv)):
                perms[lid] = inv
        return self.apply(perms) if perms else {"layers": 0}
