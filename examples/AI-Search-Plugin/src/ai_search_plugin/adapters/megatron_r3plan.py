# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R3 EP plan for Megatron's alltoall MoE token dispatcher (monkeypatch of ``MoEAlltoAllTokenDispatcher.preprocess``).

What it saves.  Per MoE layer call, Megatron's dropless alltoall path all-gathers the per-expert token counts over the
tp x ep group and waits on the host for the all-to-all splits before the EP all-to-all.  With rollout routing replay
(R3) all routes of the step are known before the first forward, so once per step this module counts tokens per
(micro-batch, layer, expert) from the replayed ids, does ONE all-gather of the counts and ONE host copy, and then every
planned layer call sets ``input_splits``, ``output_splits``, ``output_splits_tp``, ``num_out_tokens``,
``num_global_tokens_per_local_expert`` and the returned tokens-per-local-expert from the plan, with no collective and
no host sync (``cuda_sync_point = "no_sync"``).  The values are integers derived from the same routing, so everything
downstream (permute, all-to-all, experts, combine) is unchanged.  The plan is reused by every pass of the step
(log-prob forwards, training forward, activation recompute in backward).

Who drives it.  A *source* (``set_source``) says whether the current layer call replays routes (``active()``), gives
the replayed ids of every MoE layer on this rank (``lists()``) and the micro-batch of the current call
(``index(layer_pos)``).  The slime source is in :mod:`ai_search_plugin.adapters.slime_r3plan`; ``check_step()`` must be
called once per step on every rank (the slime glue calls it from ``RoutingReplay.clear_all``).

Runtime guard (always on).
  * Fallback before use, symmetric across the tp x ep group: ineligible configuration (token dropping / capacity
    factor, router padding for quantization, CUDA graphs), no replay in this call, no plan (any rank without replay
    data, or ranks disagreeing on micro-batches / layers / experts - checked by a small all-gather at build time),
    layer or micro-batch not in the plan.  These calls run Megatron's original ``preprocess``.
  * Verify steps (``ASRT_R3PLAN_VERIFY_STEPS``, default 1): the first planned steps run the ORIGINAL preprocess on
    every call (with its all-gather and host sync), compare every attribute exactly with the plan, and use the
    original values.  A layout whose micro-batch mapping is wrong is caught here before any plan value is used.
  * Every planned call adds ``routing_map.sum(0) != planned counts`` to a device counter (no host sync).
  * ``check_step()`` reduces (MAX over all ranks) the device counter and the verify failures.  Any mismatch: warning,
    the plan is switched off for the rest of the run on all ranks (Megatron's original path from then on).  The step
    that showed a device mismatch already dispatched with the plan, so the warning marks it as suspect;
    ``ASRT_R3PLAN_STRICT=1`` raises instead.
  * A token-count difference between the routing map and the plan for this rank cannot fall back alone (the other
    ranks skip the all-gather, the group would hang): it raises.

Layouts.  Checked in the source implementation with 0 mismatches: PP1, PP2/VPP2 (smoke model), PP4/VPP3 (with
interleaved schedule and block recompute).  Other layouts are expected to work (the micro-batch comes from the replay
cursor, not from call order) but rely on the guard above; ``layout`` in the stats says which case applies.

Not covered: the flex dispatcher (DeepEP / HybridEP; the source has a separate variant), token dropping, CUDA-graph
captured preprocess, ``moe_router_padding_for_quantization``.  All of them fall back automatically.

Env: ``ASRT_R3PLAN_VERIFY_STEPS`` (1), ``ASRT_R3PLAN_STRICT`` (0), ``ASRT_LOG_DIR`` (writes r3plan.jsonl).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, List, Optional

import numpy as np

from .. import r3plan as core

log = logging.getLogger("ai_search_plugin.megatron_r3plan")

STATS = {"steps": 0, "builds": 0, "build_skipped": 0, "planned_calls": 0, "fallback_calls": 0, "verified_calls": 0,
         "verify_fail": 0, "mismatch": 0, "build_ms_host": []}
_S = {"installed": False, "orig": None, "source": None, "plan": None, "need": True, "built": False,
      "dev": {}, "verify_fail_step": 0, "guard": None, "gather": None, "mapper": None, "why": "", "layout": None}

_ATTRS = ("input_splits", "output_splits", "output_splits_tp", "num_out_tokens", "num_global_tokens_per_local_expert")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def guard() -> core.StepGuard:
    if _S["guard"] is None:
        _S["guard"] = core.StepGuard(verify_steps=_env_int("ASRT_R3PLAN_VERIFY_STEPS", 1),
                                     strict=os.environ.get("ASRT_R3PLAN_STRICT", "0") == "1")
    return _S["guard"]


# ------------------------------------------------------------------------------------------------- public hooks
def set_source(source) -> None:
    """source: object with ``active() -> bool`` (this call replays routes; same answer on all ranks),
    ``lists() -> [(dispatcher, [ids per micro-batch])]`` in layer order (or [] when unavailable on this rank) and
    ``index(layer_pos) -> micro-batch or None``."""
    _S["source"] = source


def set_route_mapper(fn: Optional[Callable]) -> None:
    """fn(layer_pos, ids_tensor) -> ids as the router will see them, for components that remap expert ids at replay
    time (e.g. expert re-placement).  Call ``invalidate()`` collectively whenever the mapping changes."""
    _S["mapper"] = fn


def invalidate() -> None:
    """Drop the plan; the next replayed MoE call rebuilds it.  Call on every rank at the same point (it is collective
    at the next build)."""
    _S["plan"] = None
    _S["need"] = True


def set_layout(pp: int, vpp: Optional[int]) -> str:
    _S["layout"] = (int(pp or 1), int(vpp or 1), core.layout_status(pp, vpp))
    return _S["layout"][2]


def _set_gather(fn: Optional[Callable]) -> None:
    """Test hook: fn(group, tensor[...]) -> tensor[G, ...]."""
    _S["gather"] = fn


def _gather(group, t):
    fn = _S["gather"]
    if fn is not None:
        return fn(group, t)
    import torch
    import torch.distributed as dist

    G = group.size() if group is not None else 1
    if G == 1 or not dist.is_initialized():
        return t.unsqueeze(0)
    out = torch.empty((G,) + tuple(t.shape), dtype=t.dtype, device=t.device)
    dist.all_gather_into_tensor(out, t.contiguous(), group=group)
    return out


# ------------------------------------------------------------------------------------------------- plan
class Plan:
    """One step's plan for this rank.  Built collectively over the tp x ep group (two all-gathers)."""

    def __init__(self, entries, disp, device):
        import torch

        self.keys = {id(d): li for li, (d, _) in enumerate(entries)}
        lists = [lst for _, lst in entries]
        self.L, self.M = len(lists), len(lists[0])
        self.E = int(disp.num_experts)
        self.nle = int(disp.num_local_experts)
        self.lo = int(disp.local_expert_indices[0])
        self.T = [[int(lists[li][m].shape[0]) for li in range(self.L)] for m in range(self.M)]
        local = torch.empty((self.M, self.L, self.E), dtype=torch.int32, device=device)
        for m in range(self.M):
            keys = []
            for li, lst in enumerate(lists):   # key = layer * E + expert; holes (< 0) go to the spare last bin
                t = lst[m].to(device, torch.int64, non_blocking=True).reshape(-1)
                keys.append(torch.where(t >= 0, t + li * self.E, torch.full_like(t, self.L * self.E)))
            key = torch.cat(keys)
            cnt = torch.zeros(self.L * self.E + 1, dtype=torch.int32, device=device)  # scatter_add: bincount syncs
            cnt.scatter_add_(0, key.reshape(-1), torch.ones(key.numel(), dtype=torch.int32, device=device))
            local[m] = cnt[:self.L * self.E].view(self.L, self.E)
        glob = _gather(disp.tp_ep_group, local)                                   # [G, M, L, E]: the plan's one
        self.G = int(glob.shape[0])                                               # count collective of the step
        self.local_dev = local
        self.glob_local_dev = glob[..., self.lo:self.lo + self.nle].to(torch.int64).contiguous()  # [G, M, L, nle]
        self.local_host = local.to(torch.int64).cpu().numpy()                     # host copies: the step's only
        self.glob_local_host = self.glob_local_dev.cpu().numpy()                  # host sync of the plan

    def splits(self, disp, m, li) -> core.Splits:
        return core.splits_from_counts(self.local_host[m, li], self.glob_local_host[:, m, li], ep=disp.ep_size,
                                       tp=disp.tp_size, tp_rank=disp.tp_rank, num_local_experts=self.nle)


def _device(disp):
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _entries():
    src = _S["source"]
    try:
        entries = list(src.lists() or []) if src is not None else []
    except Exception as e:  # never break training: no plan this step
        _S["why"] = f"source failed: {e!r}"[:300]
        return []
    mapper = _S["mapper"]
    if entries and mapper is not None:
        entries = [(d, [mapper(li, t) for t in lst]) for li, (d, lst) in enumerate(entries)]
    return entries


def _build(disp):
    """Collective over disp.tp_ep_group: every rank of the group reaches its first replayed MoE call together."""
    import torch

    t0 = time.perf_counter()
    _S["need"] = False
    _S["plan"] = None
    _S["why"] = ""
    device = _device(disp)
    entries = _entries()
    ok = bool(entries) and all(len(lst) == len(entries[0][1]) and len(lst) > 0 for _, lst in entries)
    ok = ok and all(core.ineligible_reason(d) is None for d, _ in entries)
    M = len(entries[0][1]) if ok else 0
    meta = torch.tensor([int(ok), M, len(entries) if ok else 0, int(disp.num_experts)], dtype=torch.int64,
                        device=device)
    allm = _gather(disp.tp_ep_group, meta).cpu().numpy()                     # [G, 4]
    if not bool((allm[:, 0] == 1).all()) or not bool((allm[:, 1:] == allm[0, 1:]).all()):
        STATS["build_skipped"] += 1
        _S["why"] = _S["why"] or ("a rank has no replay data" if not bool((allm[:, 0] == 1).all())
                                  else "ranks disagree on micro-batches / layers / experts")
        return None
    with torch.cuda.nvtx.range("r3plan_build") if torch.cuda.is_available() else _null():
        plan = Plan(entries, disp, device)
    _S["plan"] = plan
    _S["built"] = True
    STATS["builds"] += 1
    STATS["build_ms_host"] = (STATS["build_ms_host"] + [round((time.perf_counter() - t0) * 1e3, 2)])[-4:]
    return plan


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _counter(device):
    import torch

    k = str(device)
    if k not in _S["dev"]:
        _S["dev"][k] = torch.zeros((), dtype=torch.int64, device=device)
    return _S["dev"][k]


def _host(v):
    import torch

    if torch.is_tensor(v):
        return v.detach().cpu().numpy().astype(np.int64).reshape(-1)
    if isinstance(v, np.ndarray):
        return v.astype(np.int64).reshape(-1)
    return np.asarray([int(v)], dtype=np.int64)


def _planned_values(disp, plan, m, li, routing_map):
    """(tokens, {attr: value}) of the planned call, Megatron types: numpy splits, torch int64 tokens."""
    import torch

    sp = plan.splits(disp, m, li)
    vals = {"input_splits": sp.input_splits, "output_splits": sp.output_splits,
            "output_splits_tp": sp.output_splits_tp,
            "num_out_tokens": int(routing_map.size(0)) * int(disp.config.moe_router_topk)}
    if plan.nle > 1:
        if disp.config.moe_permute_fusion:   # Megatron keeps this one on the device under permute fusion
            ep, tp, nle = disp.ep_size, disp.tp_size, plan.nle
            vals["num_global_tokens_per_local_expert"] = (
                plan.glob_local_dev[:, m, li].reshape(ep, tp, nle).transpose(0, 1).reshape(-1, nle))
        else:
            vals["num_global_tokens_per_local_expert"] = torch.from_numpy(sp.num_global_tokens_per_local_expert)
    return torch.from_numpy(sp.tokens_per_local_expert), vals


def _verify_call(disp, routing_map, orig, plan, m, li):
    import torch

    tok = orig(disp, routing_map)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ptok, vals = _planned_values(disp, plan, m, li, routing_map)
    bad = [] if np.array_equal(_host(tok), _host(ptok)) else ["tokens_per_expert"]
    for a, v in vals.items():
        cur = getattr(disp, a, None)
        if cur is None or not np.array_equal(_host(cur), _host(v)):
            bad.append(a)
    STATS["verified_calls"] += 1
    if bad:
        STATS["verify_fail"] += 1
        _S["verify_fail_step"] += 1
        _S["why"] = f"verify: layer {li} micro-batch {m}: {bad}"
    return tok   # original values stay in place: a verify step is exact either way


def preprocess_planned(disp, routing_map, orig):
    """Replacement body of ``MoEAlltoAllTokenDispatcher.preprocess``."""
    src = _S["source"]
    g = guard()
    if g.disabled or src is None or core.ineligible_reason(disp) is not None or not src.active():
        return _fallback(disp, routing_map, orig)
    if _S["need"]:
        _build(disp)
    plan = _S["plan"]
    li = plan.keys.get(id(disp)) if plan is not None else None
    m = src.index(li) if li is not None else None
    if plan is None or li is None or m is None or not 0 <= m < plan.M:
        return _fallback(disp, routing_map, orig)
    if int(routing_map.shape[0]) != plan.T[m][li]:
        raise RuntimeError(f"r3 EP plan: layer {li} micro-batch {m} has {int(routing_map.shape[0])} tokens, the plan "
                           f"{plan.T[m][li]} (replay data changed after the plan was built?); a single rank cannot "
                           "fall back without desynchronizing the tp x ep group")
    if g.verifying():
        return _verify_call(disp, routing_map, orig, plan, m, li)
    import torch

    cnt = _counter(routing_map.device)
    cnt += (routing_map.sum(dim=0).to(torch.int32) != plan.local_dev[m, li].to(routing_map.device)).any().to(
        torch.int64)
    tok, vals = _planned_values(disp, plan, m, li, routing_map)
    for a, v in vals.items():
        setattr(disp, a, v)
    disp.cuda_sync_point = "no_sync"   # every count is already on the host
    STATS["planned_calls"] += 1
    return tok


def _fallback(disp, routing_map, orig):
    STATS["fallback_calls"] += 1
    disp.cuda_sync_point = "no_sync"   # recomputed by the original (a planned call left it at no_sync)
    return orig(disp, routing_map)


def install() -> bool:
    """Patch ``MoEAlltoAllTokenDispatcher.preprocess`` (idempotent).  Inactive until a source is set."""
    if _S["installed"]:
        return True
    try:
        from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher as D
    except Exception as e:  # pragma: no cover - needs Megatron
        log.warning("r3 EP plan: Megatron alltoall dispatcher not importable (%s)", e)
        return False
    return patch_class(D)


def patch_class(D) -> bool:
    if getattr(D.preprocess, "__asrt_r3plan__", False):
        _S["installed"] = True
        return True
    orig = D.preprocess
    _S["orig"] = orig

    def preprocess(self, routing_map):
        return preprocess_planned(self, routing_map, orig)

    preprocess.__asrt_r3plan__ = True
    preprocess.__wrapped__ = orig
    preprocess.__doc__ = orig.__doc__
    D.preprocess = preprocess
    _S["installed"] = True
    return True


def _reduce_max(vals: List[int]) -> List[int]:
    try:
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return vals
        dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        t = torch.tensor(vals, dtype=torch.int64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return [int(x) for x in t.tolist()]
    except Exception as e:  # pragma: no cover
        log.warning("r3 EP plan: guard reduce failed (%r); using local counts", e)
        return vals


def check_step() -> dict:
    """Close the step on EVERY rank (collective when torch.distributed is up): reduce the guard counters, decide
    fallback / raise, drop the plan (the next step rebuilds it).  Returns the stats."""
    local = 0
    for c in _S["dev"].values():
        local += int(c.item())
        c.zero_()
    vf = _S["verify_fail_step"]
    built = _S["built"]
    mismatch, vf, built_any = _reduce_max([local, vf, int(built)])
    planned, fallback = STATS["planned_calls"], STATS["fallback_calls"]
    prev = _S.setdefault("prev_calls", (0, 0))
    _S["prev_calls"] = (planned, fallback)
    STATS["mismatch"] += mismatch
    STATS["steps"] += 1
    _S["verify_fail_step"] = 0
    _S["built"] = False
    _S["plan"] = None
    _S["need"] = True
    try:
        rec = guard().end_step(mismatch=mismatch, verify_fail=vf, planned_calls=planned - prev[0],
                               fallback_calls=fallback - prev[1], built=bool(built_any))
    finally:
        out = snapshot()
        _log_rec(out)
    out["step_record"] = rec
    return out


def snapshot() -> dict:
    g = guard()
    s = dict(STATS)
    s.update(disabled=g.disabled, disabled_why=g.disabled_why, verify_steps=g.verify_steps,
             planned_steps=g.planned_steps, why=_S["why"],
             layout=None if _S["layout"] is None else "PP{}/VPP{} ({})".format(*_S["layout"]))
    return s


def _log_rec(rec):
    d = os.environ.get("ASRT_LOG_DIR")
    if not d:
        return
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "r3plan.jsonl"), "a") as f:
            f.write(json.dumps(dict(rec, t=time.time()), default=str) + "\n")
    except OSError:
        pass


def _reset_for_tests():
    STATS.update(steps=0, builds=0, build_skipped=0, planned_calls=0, fallback_calls=0, verified_calls=0,
                 verify_fail=0, mismatch=0, build_ms_host=[])
    _S.update(source=None, plan=None, need=True, built=False, dev={}, verify_fail_step=0, guard=None, gather=None,
              mapper=None, why="", layout=None, prev_calls=(0, 0))
