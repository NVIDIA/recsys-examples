# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R3 EP plan core (numpy only): counts, dispatcher attributes vs Megatron's arithmetic and vs a token-flow
simulation, replay cursor, eligibility, step guard."""
from types import SimpleNamespace

import numpy as np
import pytest

from ai_search_plugin import r3plan as core


def _routes(rng, T, k, E):
    return np.stack([rng.choice(E, size=k, replace=False) for _ in range(T)])


def test_count_routes_and_plan():
    rng = np.random.default_rng(0)
    E, k = 16, 3
    ids = _routes(rng, 50, k, E)
    ids[3, 1] = -1                                  # a hole: no route in that slot
    c = core.count_routes(ids, E)
    ref = np.zeros(E, np.int64)
    for t in ids.reshape(-1):
        if t >= 0:
            ref[t] += 1
    assert np.array_equal(c, ref) and c.sum() == 50 * k - 1
    lists = [[_routes(rng, T, k, E) for T in (7, 9)] for _ in range(4)]   # 4 layers x 2 micro-batches
    P = core.count_plan(lists, E)
    assert P.shape == (2, 4, E)
    for li in range(4):
        for m in range(2):
            assert np.array_equal(P[m, li], core.count_routes(lists[li][m], E))
    with pytest.raises(ValueError):
        core.count_routes(np.array([[E]]), E)
    with pytest.raises(ValueError):
        core.count_plan([[ids], [ids, ids]], E)


def _megatron_reference(all_counts, tp_ep_rank, ep, tp, local_expert_indices):
    """Literal numpy transcription of MoEAlltoAllTokenDispatcher.preprocess (dropless, ep*tp > 1)."""
    E = all_counts.shape[1]
    num_local_tokens_per_expert = all_counts[tp_ep_rank]
    nle = len(local_expert_indices)
    input_splits = num_local_tokens_per_expert.reshape(ep, nle).sum(axis=1)
    num_global_tokens_per_expert = all_counts.reshape(ep, tp, E).transpose(1, 0, 2)   # gather over tp_ep group
    ngtle = num_global_tokens_per_expert[:, :, local_expert_indices[0]:local_expert_indices[-1] + 1]
    per_rank = ngtle.sum(axis=2)
    tp_rank = tp_ep_rank % tp
    return dict(input_splits=input_splits, output_splits=per_rank[tp_rank], output_splits_tp=per_rank.sum(axis=1),
                tokens=ngtle.sum(axis=(0, 1)), ngtple=ngtle.reshape(-1, nle))


@pytest.mark.parametrize("ep,tp,nle", [(4, 1, 2), (2, 2, 3), (2, 4, 1), (1, 2, 4), (8, 1, 1)])
def test_splits_match_megatron_and_token_flow(ep, tp, nle):
    rng = np.random.default_rng(ep * 100 + tp * 10 + nle)
    E, k, G = ep * nle, min(2, ep * nle), ep * tp
    routes = {r: _routes(rng, int(rng.integers(5, 40)), k, E) for r in range(G)}    # uneven tokens per rank
    all_counts = np.stack([core.count_routes(routes[r], E) for r in range(G)])     # [G, E] = the gathered plan row
    for r in range(G):
        e_idx, t_idx = divmod(r, tp)                    # Megatron tp_ep group: ep-major, tp-minor
        lo = e_idx * nle
        sp = core.splits_from_counts(all_counts[r], core.local_expert_columns(all_counts, lo, nle), ep=ep, tp=tp,
                                     tp_rank=t_idx, num_local_experts=nle)
        ref = _megatron_reference(all_counts, r, ep, tp, list(range(lo, lo + nle)))
        assert np.array_equal(sp.input_splits, ref["input_splits"])
        assert np.array_equal(sp.output_splits, ref["output_splits"])
        assert np.array_equal(sp.output_splits_tp, ref["output_splits_tp"])
        assert np.array_equal(sp.tokens_per_local_expert, ref["tokens"])
        assert np.array_equal(sp.num_global_tokens_per_local_expert, ref["ngtple"])
        # first principles: count the actual token copies that move
        mine = set(range(lo, lo + nle))
        sent = [int((routes[r] // nle == d).sum()) for d in range(ep)]   # EP rank d owns experts d*nle..
        assert list(sp.input_splits) == sent
        recv_ep = [int(np.isin(routes[d * tp + t_idx], list(mine)).sum()) for d in range(ep)]
        assert list(sp.output_splits) == recv_ep                                     # EP all-to-all, own TP rank
        recv_tp = [sum(int(np.isin(routes[d * tp + t], list(mine)).sum()) for d in range(ep)) for t in range(tp)]
        assert list(sp.output_splits_tp) == recv_tp                                  # after the TP all-gather
        per_exp = [sum(int((routes[q] == x).sum()) for q in range(G)) for x in range(lo, lo + nle)]
        assert list(sp.tokens_per_local_expert) == per_exp
        assert sp.input_splits.sum() == routes[r].size


def test_splits_shape_checks():
    with pytest.raises(ValueError):
        core.splits_from_counts(np.zeros(6), np.zeros((4, 2)), ep=4, tp=1, tp_rank=0, num_local_experts=2)
    with pytest.raises(ValueError):
        core.splits_from_counts(np.zeros(8), np.zeros((3, 2)), ep=4, tp=1, tp_rank=0, num_local_experts=2)


def test_replay_cursor():
    assert core.replay_cursor("replay_forward", 3, 0) == 2
    assert core.replay_cursor("replay_backward", 5, 1) == 0
    assert core.replay_cursor("replay_forward", 0, 0) is None       # nothing popped yet
    for st in ("fallthrough", "record", None, ""):
        assert core.replay_cursor(st, 3, 3) is None                 # routes computed, not replayed


def _cfg(**kw):
    base = dict(moe_expert_capacity_factor=None, moe_router_padding_for_quantization=False, cuda_graph_impl="none")
    base.update(kw)
    return SimpleNamespace(**base)


def test_ineligible_reason():
    ok = SimpleNamespace(config=_cfg(), drop_and_pad=False, cuda_dtoh_point="before_permutation_1")
    assert core.ineligible_reason(ok) is None
    assert core.ineligible_reason(SimpleNamespace(config=_cfg(), drop_and_pad=True)) == "drop_and_pad"
    assert "capacity" in core.ineligible_reason(SimpleNamespace(config=_cfg(moe_expert_capacity_factor=1.0)))
    assert "padding" in core.ineligible_reason(SimpleNamespace(config=_cfg(moe_router_padding_for_quantization=True)))
    assert "cuda graphs" in core.ineligible_reason(SimpleNamespace(config=_cfg(cuda_graph_impl="local")))
    assert core.ineligible_reason(SimpleNamespace(config=_cfg(), cuda_dtoh_point="before_ep_alltoall")) is not None


def test_layout_status():
    assert core.layout_status(1, None) == "validated"
    assert core.layout_status(2, 2) == "validated"
    assert core.layout_status(4, 3) == "validated"
    assert core.layout_status(2, 1) == "guarded"
    assert core.layout_status(8, 2) == "guarded"


def test_guard_verify_then_plan_then_fallback():
    g = core.StepGuard(verify_steps=1)
    assert g.verifying()
    r = g.end_step(mismatch=0, verify_fail=0, planned_calls=0, fallback_calls=0, built=False)   # no plan: no verify
    assert g.verifying() and not r["verify"]
    r = g.end_step(mismatch=0, verify_fail=0, planned_calls=0, fallback_calls=0, built=True)
    assert r["verify"] and not g.verifying() and not g.disabled
    r = g.end_step(mismatch=0, verify_fail=0, planned_calls=12, fallback_calls=0, built=True)
    assert not g.disabled and "action" not in r
    r = g.end_step(mismatch=3, verify_fail=0, planned_calls=12, fallback_calls=0, built=True)
    assert g.disabled and r["action"] == "fallback" and "step 3" in g.disabled_why and "suspect" in g.disabled_why
    r = g.end_step(mismatch=0, verify_fail=0, planned_calls=0, fallback_calls=12, built=False)
    assert g.disabled and not g.verifying() and "action" not in r       # stays off, warns once


def test_guard_verify_failure_and_strict():
    g = core.StepGuard(verify_steps=2)
    r = g.end_step(mismatch=0, verify_fail=5, planned_calls=0, fallback_calls=0, built=True)
    assert g.disabled and r["action"] == "fallback" and "original values were used" in g.disabled_why
    s = core.StepGuard(verify_steps=0, strict=True)
    assert not s.verifying()
    with pytest.raises(RuntimeError, match="STRICT"):
        s.end_step(mismatch=1, verify_fail=0, planned_calls=4, fallback_calls=0, built=True)
    assert s.disabled and s.history[-1]["action"] == "raise"
