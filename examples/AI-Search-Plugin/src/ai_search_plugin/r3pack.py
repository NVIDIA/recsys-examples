"""R3-aware micro-batch packing (RoutePack-style, placement-agnostic).

With routing replay (R3) the expert ids of every training token are known before
the step. In an EP group, all DP ranks' m-th micro-batches go through the same
all-to-all ("micro-step" m), so the MoE layer waits for the EP rank that received
the most tokens *in that micro-step*. Batch-level expert placement balances the
whole step, but individual micro-steps still drift (content-correlated routing).

This module re-assigns samples to micro-steps so that every micro-step's expert
histogram is close to the step's average expert density (tokens * P). Then any
placement that balances the step also balances each micro-step. It needs no
model change and works with or without expert re-placement.

pack_microsteps(hists, lengths, n_groups, cap) -> list of sample-index lists
  hists   [S, F] float  per-sample expert counts (F = layers*experts, may be strided)
  lengths [S]    int    tokens per sample
  n_groups              number of micro-steps
  cap                   max tokens per micro-step (e.g. dp * max_tokens_per_gpu * cp)
Greedy: longest first; each sample goes to the feasible group whose deviation from
target (G_g - P*tok_g) is most anti-aligned with the sample's own deviation
(h_s - P*len_s), i.e. the group whose L2 error it reduces most.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def sample_hist(routes, num_experts: int, layers: Optional[Sequence[int]] = None, stride: int = 1) -> np.ndarray:
    """routes [tokens, layers, topk] (np/torch, logical ids) -> flat [L*E] float32 histogram."""
    r = np.asarray(routes)
    if stride > 1:
        r = r[::stride]
    if layers is not None:
        r = r[:, list(layers)]
    L = r.shape[1]
    off = (np.arange(L, dtype=np.int64) * num_experts)[None, :, None]
    idx = (r.astype(np.int64) + off).ravel()
    return np.bincount(idx, minlength=L * num_experts).astype(np.float32) * stride


def pack_microsteps(hists: np.ndarray, lengths: Sequence[int], n_groups: int, cap: float) -> List[List[int]]:
    H = np.asarray(hists, dtype=np.float64)
    T = np.asarray(lengths, dtype=np.float64)
    S = len(T)
    if n_groups <= 1 or S <= n_groups:
        return [list(range(S))] if n_groups <= 1 else [[i] for i in range(S)] + [[] for _ in range(n_groups - S)]
    P = H.sum(0) / max(T.sum(), 1.0)                 # expert density per token
    D = np.zeros((n_groups, H.shape[1]))             # G_g - P*tok_g
    tok = np.zeros(n_groups)
    groups: List[List[int]] = [[] for _ in range(n_groups)]
    for s in np.argsort(-T, kind="stable"):
        d = H[s] - P * T[s]
        ok = np.where(tok + T[s] <= cap)[0]
        if len(ok) == 0:
            g = int(np.argmin(tok))
        else:
            score = D[ok] @ d
            # tie-break toward emptier groups so token counts stay even
            score = score + 1e-9 * tok[ok]
            g = int(ok[np.argmin(score)])
        groups[g].append(int(s)); D[g] += d; tok[g] += T[s]
    return groups


def split_group(members: Sequence[int], lengths: Sequence[int], n_bins: int, cap: float) -> Optional[List[List[int]]]:
    """First-fit-decreasing of a micro-step's samples into n_bins micro-batches of <= cap tokens.
    Returns None if it does not fit (caller falls back). Empty bins are allowed only if
    there are fewer samples than bins."""
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    load = [0.0] * n_bins
    for s in sorted(members, key=lambda i: -lengths[i]):
        cand = [b for b in range(n_bins) if load[b] + lengths[s] <= cap]
        if not cand:
            if all(load[b] > 0 for b in range(n_bins)):
                return None
            cand = [b for b in range(n_bins) if load[b] == 0]
        b = min(cand, key=lambda b: load[b])
        bins[b].append(s); load[b] += lengths[s]
    if len(members) >= n_bins and any(not b for b in bins):
        return None
    return bins


def microstep_imbalance(hists: np.ndarray, groups: List[List[int]], rank_of: np.ndarray, num_experts: int) -> float:
    """Token-weighted mean over (group, layer) of max/mean EP-rank load. rank_of [L, E] or [E]."""
    H = np.asarray(hists)
    L = H.shape[1] // num_experts
    ro = np.broadcast_to(np.asarray(rank_of), (L, num_experts))
    ep = int(ro.max()) + 1
    num = den = 0.0
    for g in groups:
        if not g:
            continue
        c = H[g].sum(0).reshape(L, num_experts)
        for l in range(L):
            load = np.bincount(ro[l], weights=c[l], minlength=ep)
            w = load.sum()
            if w > 0:
                num += w * load.max() / load.mean(); den += w
    return num / max(den, 1.0)


def pack_microbatches(hists: np.ndarray, lengths: Sequence[int], n_groups: int, n_bins: int,
                      cap: float) -> Optional[List[List[List[int]]]]:
    """Like pack_microsteps, but places every sample directly into one of n_bins micro-batches
    (<= cap tokens each) of its micro-step, so the per-micro-batch cap always holds.
    Returns plan[g][b] = sample indices, or None if the samples do not fit / a bin stays empty."""
    H = np.asarray(hists, dtype=np.float64)
    T = np.asarray(lengths, dtype=np.float64)
    P = H.sum(0) / max(T.sum(), 1.0)
    D = np.zeros((n_groups, H.shape[1]))
    load = np.zeros((n_groups, n_bins))
    plan: List[List[List[int]]] = [[[] for _ in range(n_bins)] for _ in range(n_groups)]
    order = np.argsort(-T, kind="stable")
    need_full = len(T) >= n_groups * n_bins
    for k, s in enumerate(order):
        room = (load + T[s] <= cap) | ((load == 0) & (T[s] > cap))  # oversize sample: alone in its own micro-batch (as slime does)
        if need_full and (len(T) - k) <= int((load == 0).sum()):
            room &= (load == 0)  # just enough samples left to give every empty micro-batch one
        ok = np.where(room.any(1))[0]
        if len(ok) == 0:
            return None
        d = H[s] - P * T[s]
        # prefer groups that reduce expert-histogram error; tiny token term keeps groups even
        score = D[ok] @ d + 1e-6 * load[ok].sum(1) * float(np.abs(d).sum())
        g = int(ok[np.argmin(score)])
        cand = np.where(room[g])[0]
        b = int(cand[np.argmin(load[g, cand])])
        plan[g][b].append(int(s)); load[g, b] += T[s]; D[g] += d
    if len(T) >= n_groups * n_bins and any(not b for grp in plan for b in grp):
        return None
    return plan


def _assign(cost: np.ndarray) -> np.ndarray:
    """Row -> column (one row per column at most; rows <= cols): min-cost assignment."""
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(cost)
        out = np.full(cost.shape[0], -1, dtype=np.int64)
        out[r] = c
        return out
    except Exception:  # no scipy: greedy on the sorted cost matrix
        out = np.full(cost.shape[0], -1, dtype=np.int64)
        used = set()
        for flat in np.argsort(cost, axis=None, kind="stable"):
            i, j = divmod(int(flat), cost.shape[1])
            if out[i] < 0 and j not in used:
                out[i] = j
                used.add(j)
                if len(used) == cost.shape[0]:
                    break
        return out


def pack_microbatches_fast(hists: np.ndarray, lengths: Sequence[int], n_groups: int, n_bins: int,
                           cap: float) -> Optional[List[List[List[int]]]]:
    """Same contract and objective as ``pack_microbatches`` but dealt in rounds: the next n_groups samples
    (longest first) get one micro-step each, chosen by a min-cost assignment on the score matrix
    (sample deviation . group deviation) computed with one matmul per round.  O(S/G) numpy rounds instead of S
    Python iterations; also keeps the micro-steps' token counts even (one sample per micro-step per round).
    Inside the micro-step the sample goes to the least-loaded micro-batch (the one with the most room)."""
    H = np.asarray(hists, dtype=np.float32)
    T = np.asarray(lengths, dtype=np.float64)
    S = len(T)
    P = (H.sum(0, dtype=np.float64) / max(T.sum(), 1.0)).astype(np.float32)
    D = np.zeros((n_groups, H.shape[1]), dtype=np.float32)
    load = np.zeros((n_groups, n_bins))
    plan: List[List[List[int]]] = [[[] for _ in range(n_bins)] for _ in range(n_groups)]
    order = np.argsort(-T, kind="stable")
    BIG = 1e30
    for r0 in range(0, S, n_groups):
        rows = order[r0:r0 + n_groups]
        Tr = T[rows]
        d = H[rows] - P[None, :] * Tr[:, None].astype(np.float32)                 # [c, F]
        minl = load.min(1)                                                        # most room per micro-step
        feas = (minl[None, :] + Tr[:, None] <= cap) | ((minl[None, :] == 0) & (Tr[:, None] > cap))
        if not feas.any(1).all():
            return None
        cost = (d @ D.T).astype(np.float64)                                       # [c, G]
        cost += 1e-6 * load.sum(1)[None, :] * np.abs(d).sum(1, dtype=np.float64)[:, None]
        cost[~feas] = BIG
        g_of = _assign(cost)
        bad = (g_of < 0) | ~feas[np.arange(len(rows)), np.maximum(g_of, 0)]
        for i in np.where(bad)[0]:                                                # rare: forbidden cell picked
            s = rows[i]
            taken = set(int(x) for x in g_of[~bad])
            ml = load.min(1)
            ok = [g for g in np.where((ml + T[s] <= cap) | ((ml == 0) & (T[s] > cap)))[0] if g not in taken]
            if not ok:
                return None
            g_of[i] = ok[int(np.argmin(cost[i, ok]))]
            bad[i] = False
        b_of = np.argmin(load[g_of], axis=1)                                      # micro-steps distinct this round
        load[g_of, b_of] += Tr
        D[g_of] += d
        for s, g, b in zip(rows.tolist(), g_of.tolist(), b_of.tolist()):
            plan[g][b].append(s)
        if (load[g_of, b_of] > cap).any():
            over = load[g_of, b_of] > cap
            if not np.all(load[g_of, b_of][over] == Tr[over]):                    # only an oversize sample alone
                return None
    if S >= n_groups * n_bins and any(not b for grp in plan for b in grp):
        return None
    return plan


def is_lazy_ref(v) -> bool:
    """A lazily loaded route reference (slime --rollout-data-transport straw: straw.tensor.TensorRef), which
    supports len() and contiguous row-slice reads.  Loading it whole would undo the streaming handoff."""
    import numpy as _np
    try:
        import torch as _torch
        if isinstance(v, _torch.Tensor):
            return False
    except Exception:  # pragma: no cover
        pass
    return not isinstance(v, (_np.ndarray, list, tuple)) and hasattr(v, "load") and hasattr(v, "__getitem__") \
        and hasattr(v, "__len__")


def sampled_rows(v, chunks: int = 4, rows: int = 512):
    """(array of a few evenly spaced contiguous row chunks, scale = total_rows / rows_read) for a lazy ref."""
    n = len(v)
    if n <= chunks * rows:
        a = v[0:n]
        return (a.numpy() if hasattr(a, "numpy") else np.asarray(a)), 1.0
    starts = [int(i * (n - rows) / (chunks - 1)) for i in range(chunks)]
    parts = [v[st:st + rows] for st in starts]
    parts = [p.numpy() if hasattr(p, "numpy") else np.asarray(p) for p in parts]
    return np.concatenate(parts), n / (chunks * rows)


def route_hist(v, num_experts: int, layers=None, stride: int = 1) -> np.ndarray:
    """Expert histogram of one sample's routes; lazy refs are estimated from a few row chunks (no full load)."""
    if is_lazy_ref(v):
        a, scale = sampled_rows(v)
        return sample_hist(a, num_experts, layers, 1) * scale
    a = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    return sample_hist(a, num_experts, layers, stride)
