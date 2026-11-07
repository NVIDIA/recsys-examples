# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python expert placement planner (no C++ extension, no numpy).

Input: per-expert token loads and the EP rank each expert currently lives on (equal experts per rank).
Algorithm: repeatedly swap one expert of the hottest EP rank with one expert of another rank so that
max(rank load) drops the most; stop when no swap helps or after ``max_swaps``.  Experts per rank stay fixed, so the
result can always be applied as a slot permutation (see ``adapters.megatron_placement.placement_to_perm``).
"""
from __future__ import annotations

import bisect
from typing import List, Sequence, Tuple


def rank_loads(loads: Sequence[float], rank_of: Sequence[int], ranks: int) -> List[float]:
    """Sum of expert loads per EP rank."""
    out = [0.0] * ranks
    for x, r in zip(loads, rank_of):
        out[int(r)] += float(x)
    return out


def imbalance(loads: Sequence[float], rank_of: Sequence[int], ranks: int) -> float:
    """max(rank load) / mean(rank load); 1.0 when there is no load at all."""
    r = rank_loads(loads, rank_of, ranks)
    s = sum(r)
    return max(r) / (s / ranks) if s > 0 else 1.0


def rebalance_swaps(loads: Sequence[float], rank_of: Sequence[int], ranks: int,
                    max_swaps: int = 256) -> Tuple[List[int], int]:
    """Greedy pairwise swaps that lower the hottest rank's load.

    Returns ``(new_rank_of, moved)``: the new EP rank of every expert and how many experts changed rank."""
    loads = [float(x) for x in loads]
    p = [int(r) for r in rank_of]
    for _ in range(max_swaps):
        rl = rank_loads(loads, p, ranks)
        hi = max(range(ranks), key=rl.__getitem__)
        best_nm, be, bf = rl[hi], -1, -1
        by_rank: List[List[Tuple[float, int]]] = [[] for _ in range(ranks)]
        for e, r in enumerate(p):
            by_rank[r].append((loads[e], e))
        for v in by_rank:
            v.sort()
        for r in range(ranks):
            if r == hi:
                continue
            gap = rl[hi] - rl[r]
            if gap <= 0:
                continue
            other = by_rank[r]
            keys = [x[0] for x in other]
            for le, e in by_rank[hi]:
                # best partner: load closest to le - gap/2 (moves half the gap)
                k = bisect.bisect_left(keys, le - gap / 2)
                for j in (k - 1, k):
                    if 0 <= j < len(other):
                        d = le - other[j][0]
                        if 0 < d < gap:
                            nm = max(rl[hi] - d, rl[r] + d)
                            if nm < best_nm - 1e-12:
                                best_nm, be, bf = nm, e, other[j][1]
        if be < 0:
            break
        p[be], p[bf] = p[bf], p[be]
    moved = sum(1 for a, b in zip(p, rank_of) if a != int(b))
    return p, moved
