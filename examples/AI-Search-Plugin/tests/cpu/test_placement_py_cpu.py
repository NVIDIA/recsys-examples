# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python placement planner (ai_search_plugin.placement_py): no torch, no numpy, no C++ extension."""
import random

from ai_search_plugin.placement_py import imbalance, rank_loads, rebalance_swaps
from ai_search_plugin.adapters.megatron_placement import placement_to_perm


def test_imbalance_basics():
    assert imbalance([0, 0, 0, 0], [0, 0, 1, 1], 2) == 1.0          # no load at all
    assert imbalance([1, 1, 1, 1], [0, 0, 1, 1], 2) == 1.0
    assert abs(imbalance([3, 3, 1, 1], [0, 0, 1, 1], 2) - 1.5) < 1e-12
    assert rank_loads([3, 3, 1, 1], [0, 1, 0, 1], 2) == [4.0, 4.0]


def test_rebalance_hot_cluster():
    # 16 experts on 4 EP ranks, the 4 hot ones all on rank 0
    loads = [100.0] * 4 + [10.0] * 12
    cur = [e // 4 for e in range(16)]
    assert imbalance(loads, cur, 4) > 2.0
    new, moved = rebalance_swaps(loads, cur, 4)
    assert moved > 0
    assert sorted(new) == sorted(cur)                         # experts per rank unchanged
    assert imbalance(loads, new, 4) < 1.01                    # one hot expert per rank
    # every hot expert on its own rank
    assert sorted(new[:4]) == [0, 1, 2, 3]


def test_rebalance_never_worse_and_balanced_noop():
    rng = random.Random(0)
    for _ in range(50):
        ranks = rng.choice([2, 4, 8])
        n_local = rng.choice([2, 4, 8])
        E = ranks * n_local
        loads = [rng.random() ** 3 * 100 for _ in range(E)]
        cur = list(range(ranks)) * n_local
        rng.shuffle(cur)
        new, moved = rebalance_swaps(loads, cur, ranks)
        assert sorted(new) == sorted(cur)
        assert max(rank_loads(loads, new, ranks)) <= max(rank_loads(loads, cur, ranks)) + 1e-9
        assert moved == sum(1 for a, b in zip(new, cur) if a != b)
    # perfectly balanced: nothing to do
    new, moved = rebalance_swaps([5.0] * 8, [0, 1, 2, 3] * 2, 4)
    assert moved == 0 and new == [0, 1, 2, 3] * 2


def _reference_numpy(loads, rank_of, ranks, max_swaps=256):
    """The original numpy implementation of the planner, kept here to check the pure-Python port step for step."""
    import bisect

    import numpy as np

    def rl_of(p):
        out = np.zeros(ranks)
        np.add.at(out, np.asarray(p), np.asarray(loads, dtype=float))
        return out
    loads = [float(x) for x in loads]
    p = list(rank_of)
    for _ in range(max_swaps):
        rl = rl_of(p)
        hi = int(rl.argmax())
        best_nm, be, bf = rl[hi], -1, -1
        by_rank = [[] for _ in range(ranks)]
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
    return p, sum(1 for a, b in zip(p, rank_of) if a != b)


def test_same_result_as_numpy_reference():
    import pytest
    pytest.importorskip("numpy")
    rng = random.Random(1)
    for _ in range(40):
        ranks, n_local = rng.choice([(4, 4), (8, 4), (8, 16)])
        E = ranks * n_local
        loads = [float(rng.randint(0, 500)) for _ in range(E)]
        cur = [e // n_local for e in range(E)]
        assert rebalance_swaps(loads, cur, ranks) == _reference_numpy(loads, cur, ranks)


def test_placement_to_perm():
    n_local = 4
    cur = [s // n_local for s in range(16)]
    loads = [100.0] * 4 + [10.0] * 12
    new, _ = rebalance_swaps(loads, cur, 4)
    perm = placement_to_perm(cur, new, n_local)
    assert sorted(perm) == list(range(16))
    for s in range(16):
        assert perm[s] // n_local == new[s]          # slot lands on the planned rank
        if new[s] == cur[s]:
            assert perm[s] == s                      # unmoved experts keep their slot
