# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""prefix_tree host-side logic (numpy only): layout / index maps, grouping, slime packer, verl driver partition."""
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from ai_search_plugin.prefix_tree import plan as P  # noqa: E402
from ai_search_plugin.prefix_tree.adapters import slime_pack, verl_driver  # noqa: E402


def test_package_import_is_light():
    import ai_search_plugin.prefix_tree as pt
    assert pt.NAME == "prefix_tree"
    assert pt.enabled({"ASRT_PREFIX_TREE": "1"}) and not pt.enabled({})
    assert pt.install_verl() == [] or pt.enabled()      # off unless the env switch is set


def test_layout_index_maps_self_test():
    assert P.self_test(seed=0, trials=10) == {"passed": True}
    assert P.self_test(seed=1, trials=10) == {"passed": True}


def _toy_layout(cp=1, tp=1):
    a = 2 * cp
    return P.TreeLayout([(4 * a, [2 * a, 3 * a]), (0, [5 * a]), (2 * a, [a, a, 2 * a])], cp, tp, pad_multiple=4,
                        halo=2 * a)


@pytest.mark.parametrize("cp,tp", [(1, 1), (2, 2)])
def test_positions_and_mask_semantics(cp, tp):
    lay = _toy_layout(cp, tp)
    pos = P.tree_position_ids(lay)
    m = P.tree_attention_mask(lay)
    for g in lay.groups:
        T = g["T"]
        for b0, b in g["branches"]:
            assert list(pos[b0:b0 + b]) == list(range(T, T + b))           # branches continue from the trunk
            # the 'full' virtual document of this branch = trunk + branch, i.e. exactly what the mask lets it see
            for r in range(b):
                seen = np.flatnonzero(m[b0 + r])
                want = list(range(g["trunk"], g["trunk"] + T)) if T else []
                want += list(range(b0, b0 + r + 1))
                assert list(seen) == want
        if T:
            assert list(pos[g["trunk"]:g["trunk"] + T]) == list(range(T))
    # no row sees another group or a sibling branch
    assert m.sum() == sum(np.tril(np.ones((L, L))).sum() for _, L, *_ in lay.segments) + sum(
        g["T"] * b for g in lay.groups for _, b in g["branches"])


def test_virtual_full_documents_are_the_original_samples():
    lay = _toy_layout(2, 1)
    v = lay.virtual["full"]
    pos = P.tree_position_ids(lay)
    d = 0
    for g in lay.groups:
        for b0, b in g["branches"]:
            a = v["cu"][d]
            n = g["T"] + b
            assert v["cu"][d + 1] - a == n
            assert list(pos[v["v_src"][a:a + n]]) == list(range(n))     # positions restart per document like flat
            d += 1
    # canonical copies: one per tree row
    assert np.bincount(v["v_src"][v["canon"]], minlength=lay.total)[:lay.real_total].tolist() == [1] * lay.real_total


def test_group_by_prompt_slime_rule():
    p = np.arange(40)
    a = np.concatenate([p, [100, 101, 102]])
    b = np.concatenate([p, [100, 101, 7, 8]])          # shares 2 response tokens with a
    c = np.concatenate([p[:30], [5, 5, 5]])            # other prompt
    d = np.concatenate([p, [9]])
    groups = P.group_by_prompt([a, b, c, d], [40, 40, 30, 40], min_trunk=16)
    assert groups == [(40, [0, 1, 3]), (0, [2])]       # lcp(a, d) = 40; d has 41 tokens -> T <= 40 anyway
    g = P.group_by_prompt([a, b], [40, 40], min_trunk=16)
    assert g == [(42, [0, 1])]                          # trunk extends into the shared response tokens
    assert P.group_by_prompt([a, b], [40, 40], min_trunk=100) == [(0, [0]), (0, [1])]


def test_group_consecutive_and_units_verl_rule():
    rng = np.random.default_rng(0)
    prompt = rng.integers(0, 1000, 64)
    sib = [np.concatenate([prompt, rng.integers(0, 1000, int(n))]) for n in (10, 30, 5)]
    other = np.concatenate([prompt[:3], rng.integers(0, 1000, 60)])
    seqs = sib + [other]
    g = P.group_consecutive(seqs, align=4)
    assert g == [(60, [0, 1, 2]), (0, [3])]            # (lcp 64 - 1) // 4 * 4
    units = P.sibling_units([seqs[3], seqs[0], seqs[2], seqs[1]], align=4, budget=1 << 30)
    assert sorted(u[1] for u in units) == [[0], [1, 2, 3]]
    assert [u[0] for u in units if len(u[1]) == 3] == [60 + 14 + 9 + 34]
    small = P.sibling_units(seqs[:3], align=4, budget=110)       # split, trunk repeated
    assert [u[1] for u in small] == [[0, 1], [2]] and small[0][0] == 60 + 14 + 34 and small[1][0] == 60 + 9
    assert P.lcp(prompt, prompt) == 64 and P.lcp(prompt, other) == 3


def test_slime_pack_schedule_keeps_siblings_together():
    rng = np.random.default_rng(1)
    tokens, resp, rids = [], [], []
    for rid in range(8):                                 # 8 prompts x 4 siblings
        prompt = rng.integers(1, 1000, int(rng.integers(200, 400)))
        for _ in range(4):
            r = rng.integers(1, 1000, int(rng.integers(20, 120)))
            tokens.append(np.concatenate([prompt, r]))
            resp.append(len(r))
            rids.append(rid)
    args = SimpleNamespace(use_dynamic_batch_size=True, max_tokens_per_gpu=1200, balance_data=False)
    out, stats = slime_pack.tree_schedule(args, {"dp_size": 2, "cp_size": 1}, [len(t) for t in tokens], 4, rids,
                                          {"tokens": tokens, "response_lengths": resp})
    assert out is not None, stats
    partitions, mbi, nmb, gbs = out
    assert len(nmb) == 2 and gbs == [4, 4]
    assert sorted(i for p in partitions for i in p) == list(range(len(tokens)))
    for r in range(2):
        assert len(mbi[r]) == sum(nmb)                   # same number of micro-batches on every rank
    for r in range(2):
        for mb in mbi[r]:
            members = [partitions[r][j] for j in mb]
            by_rid = {}
            for i in members:
                by_rid.setdefault(rids[i], []).append(i)
            tree = sum(len(tokens[ix[0]]) - resp[ix[0]] - 1 + sum(resp[i] + 1 for i in ix) if len(ix) > 1
                       else len(tokens[ix[0]]) for ix in by_rid.values())
            assert tree <= args.max_tokens_per_gpu
    assert all(s["sib_units"] > 0 and s["tree_tokens"] < s["flat_tokens"] for s in stats)
    bad = slime_pack.tree_schedule(SimpleNamespace(use_dynamic_batch_size=False), {"dp_size": 2, "cp_size": 1},
                                   [], 4, rids, {})
    assert bad[0] is None


def test_verl_driver_group_permutation():
    keys = [f"u{g}_0_{i}" for g in range(4) for i in range(3)]
    tags = [{"seq_len": 100 + 10 * g + i, "prompt_len": 80} for g in range(4) for i in range(3)]

    def partitioner(work, k_partitions, equal_size):
        order = sorted(range(len(work)), key=lambda x: -work[x])
        parts = [[] for _ in range(k_partitions)]
        for j, x in enumerate(order):
            parts[j % k_partitions].append(x)
        return parts

    perm, parts, tree, flat = verl_driver.group_permutation(keys, tags, 2, partitioner)
    assert sorted(perm) == list(range(12))
    for p in parts:                                      # whole groups per rank, equal sample count
        assert len(p) == 6 and {verl_driver.group_key(keys[i]) for i in p}.__len__() == 2
    assert tree == sum(80 + sum(tags[g * 3 + i]["seq_len"] - 80 for i in range(3)) for g in range(4))
    assert flat == sum(t["seq_len"] for t in tags)
    assert verl_driver.group_permutation(keys[:9], tags[:9], 2, partitioner) is None   # 3 groups over 2 ranks
    assert verl_driver.group_key("abc") == "abc" and verl_driver.group_key("a_b_c_d") == "a_b"
