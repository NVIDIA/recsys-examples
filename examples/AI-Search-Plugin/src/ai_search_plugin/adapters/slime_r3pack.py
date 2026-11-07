"""R3-aware micro-batch packing for slime (manager side).

slime's ``build_dp_schedule`` packs each step's samples into micro-batches by length only
(first-fit in rollout order) and deals them to DP ranks. All DP ranks' m-th micro-batches
share one EP all-to-all, so a micro-step whose samples happen to hit the same experts makes
the MoE layer wait for one EP rank, even after batch-level expert placement.
With ``--use-rollout-routing-replay`` the routes are known here, so we re-pack every step:
``ai_search_plugin.r3pack.pack_microsteps`` assigns samples to micro-steps so each micro-step's expert
histogram tracks the step average, then FFD splits a micro-step into one micro-batch per DP
rank (each <= max_tokens_per_gpu * cp). Same number of micro-batches per rank as slime, same
samples per step; any step that does not fit falls back to slime's own schedule.

Use (no slime change):
    --rollout-function-path ai_search_plugin.adapters.slime_r3pack.generate_rollout_fully_async
  or, together with the prefetch component, ASRT_R3PACK=1 (slime_prefetch's rollout function
  installs this too).
Env: ASRT_R3PACK_STRIDE (default 4: histogram every 4th token), ASRT_R3PACK_LAYERS (default all),
     ASRT_NUM_EXPERTS, ASRT_LOG_DIR (writes r3pack.jsonl).
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from ..r3pack import is_lazy_ref, pack_microbatches, pack_microbatches_fast, route_hist, sample_hist

log = logging.getLogger("ai_search_plugin.slime_r3pack")
_state = {"installed": False, "routes": None}


def _routes_np(v):
    from slime.utils.routed_experts import materialize_routed_experts

    t = materialize_routed_experts(v)
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


def _hist(v, num_experts, layers, stride):
    """straw refs: estimate from a few row chunks (never load the whole route tensor in the manager)."""
    if is_lazy_ref(v):
        return route_hist(v, num_experts, layers, stride)
    return sample_hist(_routes_np(v), num_experts, layers, stride)


def repack(args, tpc, total_lengths, rollout_indices, global_batch_size, result, routes, num_experts):
    """Return a new (partitions, micro_batch_indices, num_microbatches, gbs) or None to keep slime's."""
    partitions_o, mbi_o, nmb, gbs = result
    dp, cp = tpc["dp_size"], tpc["cp_size"]
    cap = args.max_tokens_per_gpu * cp
    stride = int(os.environ.get("ASRT_R3PACK_STRIDE", "4"))
    lay = os.environ.get("ASRT_R3PACK_LAYERS")
    layers = [int(x) for x in lay.split(",")] if lay else None
    groups_of = {}
    for pos, rid in enumerate(rollout_indices):
        groups_of.setdefault(rid, []).append(pos)
    rids = list(groups_of)
    partitions = [[] for _ in range(dp)]
    mbi = [[] for _ in range(dp)]
    stats = []
    new_nmb = []
    step_off = [0] * dp
    for s, n_mb in enumerate(nmb):
        idx = [p for rid in rids[s * global_batch_size:(s + 1) * global_batch_size] for p in groups_of[rid]]
        lens = [int(total_lengths[i]) for i in idx]
        if any(routes[i] is None for i in idx):
            _state["why"] = f"step {s}: missing routes"
            return None
        H = np.stack([_hist(routes[i], num_experts, layers, stride) for i in idx]).astype(np.float32)
        plan = None
        inc = tpc.get("microbatch_group_size_per_vp_stage", 1) if tpc.get("vpp_size", 1) > 1 else 1
        extra = int(os.environ.get("ASRT_R3PACK_EXTRA_MB", "2"))
        algo = os.environ.get("ASRT_R3PACK_ALGO", "fast")  # fast: round-based assignment (O(S/G) numpy rounds)
        greedy_max = int(os.environ.get("ASRT_R3PACK_GREEDY_MAX", "4096"))  # per-sample greedy only up to this size
        for n_try in [n_mb + j * inc for j in range(extra + 1)]:  # slime packs tight; allow a few extra micro-steps
            if algo != "greedy":
                plan = pack_microbatches_fast(H, lens, n_try, dp, cap)
            if plan is None and (algo == "greedy" or len(idx) <= greedy_max):
                plan = pack_microbatches(H, lens, n_try, dp, cap)
            if plan is not None:
                break
        if plan is None:
            _state["why"] = (f"step {s}: does not fit in {n_mb}+{extra} micro-steps x {dp} (cap {cap}, "
                             f"tokens {sum(lens)}, max sample {max(lens)})")
            return None
        new_nmb.append(len(plan))
        for bins in plan:
            for r in range(dp):
                start = len(partitions[r])
                partitions[r].extend(idx[i] for i in bins[r])
                mbi[r].append(list(range(start, start + len(bins[r]))))
        P = H.sum(0) / max(H.sum(), 1)
        def _dev(groups):
            d = [float(np.abs(H[g].sum(0) / max(H[g].sum(), 1) - P).sum()) for g in groups if g]
            return float(np.mean(d)) if d else 0.0
        groups = [[i for b in bins for i in b] for bins in plan]
        # slime's own micro-steps for this step (local positions), for the gain check
        pos = {gi: k for k, gi in enumerate(idx)}
        orig = []
        for m in range(n_mb):
            g = []
            for r in range(dp):
                mb = mbi_o[r][step_off[r] + m] if mbi_o is not None else []
                g += [pos[partitions_o[r][j]] for j in mb if partitions_o[r][j] in pos]
            orig.append(g)
        for r in range(dp):
            step_off[r] += n_mb
        stats.append({"step": s, "n_mb": n_mb, "n_mb_packed": len(plan), "samples": len(idx),
                      "hist_l1_dev_mean": _dev(groups), "hist_l1_dev_orig": _dev(orig) if mbi_o is not None else None})
    return (partitions, mbi, new_nmb, gbs), stats


def _log_rec(rec):
    d = os.environ.get("ASRT_LOG_DIR")
    if d:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "r3pack.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


def _routes_of(data):
    """Route values for the batch: a dict (object-store transport) or, with --rollout-data-transport straw, a
    disk payload reference -> read only the route field (lazy straw TensorRefs, no tensor bytes)."""
    if isinstance(data, dict):
        return data.get("rollout_routed_experts")
    if hasattr(data, "load"):
        try:
            sel = data.load(selection={"rollout_routed_experts": None})
            return sel.get("rollout_routed_experts") if isinstance(sel, dict) else None
        except Exception as e:  # never break slime's own schedule
            _log_rec({"t": time.time(), "applied": False, "why": f"straw route refs unreadable: {e!r}"[:300]})
    return None


def install() -> bool:
    if _state["installed"]:
        return True
    try:
        import slime.data.batch_builder as bb
    except Exception as e:  # pragma: no cover
        log.warning("asrt r3pack: slime batch_builder not importable (%s)", e)
        return False
    orig_build = bb.build_dp_schedule
    orig_split = bb.BatchBuilder.split_by_dp

    def split_by_dp(self, data, *a, **k):
        _state["routes"] = _routes_of(data)
        try:
            return orig_split(self, data, *a, **k)
        finally:
            _state["routes"] = None

    def build_dp_schedule(args, tpc, total_lengths, *, global_batch_size, rollout_indices):
        res = orig_build(args, tpc, total_lengths, global_batch_size=global_batch_size, rollout_indices=rollout_indices)
        routes = _state["routes"]
        if not routes or not args.use_dynamic_batch_size or len(routes) != len(total_lengths):
            _log_rec({"t": time.time(), "applied": False, "why": "no routes / static batch / length mismatch",
                      "n_routes": len(routes) if routes else 0, "n": len(total_lengths)})
            return res
        t0 = time.perf_counter()
        if _state.get("skip", 0) > 0:  # auto-off backoff: packing did not help recently
            _state["skip"] -= 1
            _log_rec({"t": time.time(), "applied": False, "why": "auto-off backoff", "skip_left": _state["skip"]})
            return res
        ne = int(os.environ.get("ASRT_NUM_EXPERTS", getattr(args, "num_experts", None) or 0)) or None
        try:
            out = repack(args, tpc, total_lengths, rollout_indices, global_batch_size, res, routes, ne or 512)
        except Exception as e:
            log.warning("asrt r3pack failed, using slime schedule: %r", e)
            _state["why"] = repr(e)[:500]
            out = None
        rec = {"t": time.time(), "s": time.perf_counter() - t0, "applied": out is not None}
        if out is None:
            rec["why"] = _state.pop("why", "unknown")
        else:
            # auto-off: keep slime's schedule unless packing cuts the micro-step expert-histogram deviation
            # by >= ASRT_R3PACK_MIN_GAIN (relative, default 0.2); then skip the next ASRT_R3PACK_BACKOFF steps.
            gain_min = float(os.environ.get("ASRT_R3PACK_MIN_GAIN", "0.2"))
            o = [st["hist_l1_dev_orig"] for st in out[1] if st.get("hist_l1_dev_orig")]
            n = [st["hist_l1_dev_mean"] for st in out[1] if st.get("hist_l1_dev_orig")]
            gain = (1 - sum(n) / sum(o)) if o and sum(o) > 0 else None
            rec["gain"] = gain
            if gain is not None and gain < gain_min:
                rec["applied"] = False
                rec["why"] = f"auto-off: deviation gain {gain:.2f} < {gain_min}"
                _state["skip"] = int(os.environ.get("ASRT_R3PACK_BACKOFF", "4"))
                out = None
        if out is not None:
            res, rec["steps"] = out
        _log_rec(rec)
        return res

    bb.build_dp_schedule = build_dp_schedule
    bb.BatchBuilder.split_by_dp = split_by_dp
    _state["installed"] = True
    return True


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Drop-in for ``--rollout-function-path`` (installs the packer in the rollout manager)."""
    from . import slime_pool

    slime_pool.autosize(args)               # in-flight pool >= one batch (GRPO n=16: 1/4 batch otherwise)
    install()
    from . import slime_r3check
    slime_r3check.install()
    from slime.rollout.fully_async_rollout import generate_rollout_fully_async as orig

    return orig(args, rollout_id, data_buffer, evaluation=evaluation)
