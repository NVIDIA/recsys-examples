import numpy as np
from ai_search_plugin.r3pack import microstep_imbalance, pack_microsteps, sample_hist, split_group


def _data(S=240, E=64, L=2, K=4, seed=0):
    rng = np.random.default_rng(seed)
    topics = rng.dirichlet(np.ones(E) * 0.3, size=6)       # content-correlated routing
    routes, lens = [], []
    for s in range(S):
        n = int(rng.integers(300, 3000)); p = topics[s // 40 % 6]
        routes.append(rng.choice(E, size=(n, L, K), p=p)); lens.append(n)
    return routes, np.array(lens), E


def test_pack_reduces_microstep_imbalance_and_keeps_all_samples():
    routes, lens, E = _data()
    H = np.stack([sample_hist(r, E) for r in routes])
    n = 6; cap = lens.sum() / n * 1.05
    order = [list(range(i * 40, (i + 1) * 40)) for i in range(n)]   # rollout order = one topic per micro-step
    groups = pack_microsteps(H, lens, n, cap)
    assert sorted(i for g in groups for i in g) == list(range(len(lens)))
    assert max(lens[g].sum() for g in groups) <= cap + lens.max()
    rank_of = np.arange(E) // (E // 8)
    a, b = microstep_imbalance(H, order, rank_of, E), microstep_imbalance(H, groups, rank_of, E)
    assert b < a - 0.05, (a, b)


def test_split_group():
    lens = [100, 90, 80, 50, 40, 30, 20, 10]
    bins = split_group(range(8), lens, 3, 150)
    assert bins is not None and sorted(i for b in bins for i in b) == list(range(8))
    assert all(sum(lens[i] for i in b) <= 150 for b in bins)
    assert split_group(range(8), lens, 2, 150) is None
