"""repack() keeps slime's schedule invariants (same #mbs per rank, every sample once, token cap)."""
import sys
import types
from types import SimpleNamespace

import numpy as np

# minimal stub for slime.utils.routed_experts when slime is not installed
if "slime" not in sys.modules:
    try:
        import slime.utils.routed_experts  # noqa: F401
    except Exception:
        m = types.ModuleType("slime"); u = types.ModuleType("slime.utils"); r = types.ModuleType("slime.utils.routed_experts")
        r.materialize_routed_experts = lambda v: v
        sys.modules.update({"slime": m, "slime.utils": u, "slime.utils.routed_experts": r})

from ai_search_plugin.adapters.slime_r3pack import repack  # noqa: E402


def test_repack_invariants():
    rng = np.random.default_rng(0); E = 64; dp = 2; cp = 2
    n_roll, n_per = 32, 4
    lens, routes, rids = [], [], []
    for rid in range(n_roll):
        p = rng.dirichlet(np.ones(E) * 0.3)
        for _ in range(n_per):
            n = int(rng.integers(500, 4000)); lens.append(n); rids.append(rid)
            routes.append(rng.choice(E, size=(n, 3, 4), p=p))
    args = SimpleNamespace(max_tokens_per_gpu=8192, use_dynamic_batch_size=True)
    tpc = {"dp_size": dp, "cp_size": cp, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}
    gbs = 16; nmb = [6, 6]                     # 2 steps, 6 mbs per rank each (12 mbs of <=16k tokens)
    out = repack(args, tpc, lens, rids, gbs, (None, None, nmb, [gbs, gbs]), routes, E)
    assert out is not None
    (parts, mbi, nmb2, _), stats = out
    assert all(a <= b <= a + 2 for a, b in zip(nmb, nmb2)) and all(len(m) == sum(nmb2) for m in mbi)
    flat = sorted(i for p in parts for i in p)
    assert flat == list(range(len(lens)))
    for r in range(dp):
        assert sum(mbi[r], []) == list(range(len(parts[r])))
        for mb in mbi[r]:
            assert mb and sum(lens[parts[r][j]] for j in mb) <= args.max_tokens_per_gpu * cp
    # samples of one step stay in that step
    for r in range(dp):
        first = [parts[r][j] for mb in mbi[r][:nmb2[0]] for j in mb]
        assert all(rids[i] < gbs for i in first)
    assert len(stats) == 2


def test_oversize_sample_gets_own_microbatch():
    from ai_search_plugin.r3pack import pack_microbatches
    rng = np.random.default_rng(1); E = 16
    lens = [30000, 5000, 6000, 7000, 4000, 8000, 3000, 9000]
    H = np.stack([np.bincount(rng.integers(0, E, n), minlength=E).astype(float) for n in lens])
    plan = pack_microbatches(H, lens, 3, 2, 24576)
    assert plan is not None
    bins = [b for g in plan for b in g]
    assert [0] in bins and sorted(i for b in bins for i in b) == list(range(8))
    assert all(sum(lens[i] for i in b) <= 24576 for b in bins if b != [0])
