"""Pure-Python k-way cost-balanced partition (same algorithm and contract as
the C++ `solvers.balanced_partitions`), for training containers that do not
have a C++ extension built (stock verl / slime images).

cost(L) = L + quad_coef * L^2.
  equal_size=True : LPT with a cardinality cap of ceil(n/k) (exactly n/k each
                    when k divides n, as verl/slime require).
  equal_size=False: LPT on cost, then a few pairwise move/swap refinements.
Every partition is non-empty when n >= k. Returns k sorted index lists.
"""
import heapq
from typing import List, Sequence


def _cost(x: float, quad: float) -> float:
    return float(x) + quad * float(x) * float(x)


def balanced_partitions(values: Sequence[float], k: int, equal_size: bool = True,
                        quad_coef: float = 0.0, refine_rounds: int = 2) -> List[List[int]]:
    n = len(values)
    assert k >= 1 and n >= k, f"need n >= k (n={n}, k={k})"
    if equal_size:
        assert n % k == 0, f"equal_size needs k | n (n={n}, k={k})"
    c = [_cost(v, quad_coef) for v in values]
    order = sorted(range(n), key=lambda i: -c[i])
    cap = (n + k - 1) // k if equal_size else n
    parts: List[List[int]] = [[] for _ in range(k)]
    load = [0.0] * k
    if not equal_size:
        # heap LPT; the k largest go one per partition so none is empty
        h = [(0.0, r) for r in range(k)]
        for i in order:
            l, r = heapq.heappop(h)
            parts[r].append(i)
            load[r] = l + c[i]
            heapq.heappush(h, (load[r], r))
    else:
        for i in order:
            best = -1
            for r in range(k):
                if len(parts[r]) < cap and (best < 0 or load[r] < load[best]):
                    best = r
            parts[best].append(i)
            load[best] += c[i]
    # refinement: swap one item between the heaviest partition and another if it lowers the max
    for _ in range(refine_rounds * k):
        hi = max(range(k), key=load.__getitem__)
        improved = False
        for lo in sorted(range(k), key=load.__getitem__):
            if lo == hi:
                continue
            gap = load[hi] - load[lo]
            best = None
            for a in parts[hi]:
                if not equal_size and len(parts[hi]) > 1 and 0 < c[a] < gap:
                    d = c[a]
                    if best is None or abs(gap - 2 * d) < abs(gap - 2 * best[2]):
                        best = (a, None, d)
                for b in parts[lo]:
                    d = c[a] - c[b]
                    if 0 < d < gap and (best is None or abs(gap - 2 * d) < abs(gap - 2 * best[2])):
                        best = (a, b, d)
            if best is not None:
                a, b, d = best
                parts[hi].remove(a)
                parts[lo].append(a)
                if b is not None:
                    parts[lo].remove(b)
                    parts[hi].append(b)
                load[hi] -= d
                load[lo] += d
                improved = True
                break
        if not improved:
            break
    return [sorted(p) for p in parts]


def max_load(values: Sequence[float], parts: Sequence[Sequence[int]], quad_coef: float = 0.0) -> float:
    return max(sum(_cost(values[i], quad_coef) for i in p) for p in parts)
