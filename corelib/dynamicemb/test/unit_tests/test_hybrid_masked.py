# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``lookup_masked`` / ``load_masked`` and what is built on them: the public ``find``
and ``HybridStorage.insert`` / ``load``.

The masked API looks every key up and reads nothing back; a key that must not take
part is the reserved key (``sf.mask_keys``). The reference is the code it replaced,
kept below as ``reference_*``: it compacted the HBM tier's misses (a host count),
looked those up in the host tier and merged the tiers with boolean-mask indexing.
Two storages are built the same way, one is driven through the code under test and
the other through the reference, and they must end up equal.

LRU timestamps differ between two storages built at different times, so scores are
compared exactly only for LFU (counts); for LRU the order of the timestamps is
checked instead.
"""

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy
from dynamicemb import key_value_table as kvt
from dynamicemb import sync_free as sf
from dynamicemb.dynamicemb_config import DynamicEmbEvictStrategy, DynamicEmbTableOptions
from dynamicemb.key_value_table import (
    DynamicEmbStorage,
    HybridStorage,
    MaskedFind,
    _find_keys,
    _insert_and_evict_keys,
    _insert_key_values,
    load_from_flat,
    score_source,
    store_to_flat,
)
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer
from dynamicemb.types import CopyMode
from dynamicemb_extensions import select_insert_failed_values

DIM = 8
BUCKET = 128
# A key no table accepts: what a masked lane is made of (``sf.mask_keys``).
EMPTY = -1


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(params=[False, True], ids=["default", "determinism"])
def deterministic(request, monkeypatch):
    """The HBM tier of a HybridStorage inserts with the sorted kernels in DEMB_DETERMINISM_MODE
    (``_insert_and_evict_keys(..., async_count=True)``, the non-overflow evicting variant).
    """
    if request.param:
        monkeypatch.setenv("DEMB_DETERMINISM_MODE", "1")
    return request.param


# ---------------------------------------------------------------------------
# Storages
# ---------------------------------------------------------------------------


def _options(lfu, capacity, host, dim=DIM):
    return DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        dim=dim,
        max_capacity=capacity,
        init_capacity=capacity,
        bucket_capacity=BUCKET,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=0 if host else 1024**3,
        score_strategy=(
            DynamicEmbScoreStrategy.LFU if lfu else DynamicEmbScoreStrategy.TIMESTAMP
        ),
        evict_strategy=(
            DynamicEmbEvictStrategy.LFU if lfu else DynamicEmbEvictStrategy.LRU
        ),
    )


def _optimizer():
    return SGDDynamicEmbeddingOptimizer(OptimizerArgs())


def _hybrid(lfu, hbm_capacity=BUCKET):
    """An HBM tier of one bucket (it fills up and spills) over a roomy host tier."""
    return HybridStorage(
        [_options(lfu, hbm_capacity, host=False)],
        [_options(lfu, 8 * BUCKET, host=True)],
        _optimizer(),
    )


def _single(lfu):
    return DynamicEmbStorage(
        [_options(lfu, 8 * BUCKET, host=False)], _optimizer(), use_ref_counter=False
    )


def _values(keys):
    """The row of a key: its own number in every column (exact in fp32)."""
    return keys.to(torch.float32).unsqueeze(1).repeat(1, DIM)


def _fill(state, keys, lfu_score):
    """Put ``keys`` into one tier directly, with the LFU count ``lfu_score``."""
    n = keys.numel()
    tids = torch.zeros(n, dtype=torch.int64, device=keys.device)
    scores = (
        None
        if lfu_score is None
        else torch.full((n,), lfu_score, dtype=torch.int64, device=keys.device)
    )
    _insert_key_values(state, keys, tids, _values(keys), scores)


def _hbm_and_host_keys(device):
    return (
        torch.arange(1, 41, dtype=torch.int64, device=device),
        torch.arange(101, 141, dtype=torch.int64, device=device),
    )


def _absent_keys(device):
    return torch.arange(1001, 1021, dtype=torch.int64, device=device)


def _populate(storage, lfu, device):
    """Keys 1..40 in the HBM tier (count 3), 101..140 in the host tier (count 5)."""
    hbm_keys, host_keys = _hbm_and_host_keys(device)
    if isinstance(storage, HybridStorage):
        _fill(storage._hbm, hbm_keys, 3 if lfu else None)
        _fill(storage._host, host_keys, 5 if lfu else None)
    else:
        _fill(storage._state, torch.cat([hbm_keys, host_keys]), 3 if lfu else None)


def _twins(lfu, device):
    a, b = _hybrid(lfu), _hybrid(lfu)
    _populate(a, lfu, device)
    _populate(b, lfu, device)
    return a, b


def _batch(device, absent=True):
    """Every key the storage holds, and keys it does not, in a fixed shuffled order."""
    hbm_keys, host_keys = _hbm_and_host_keys(device)
    parts = [hbm_keys, host_keys] + ([_absent_keys(device)] if absent else [])
    keys = torch.cat(parts)
    order = torch.randperm(keys.numel(), generator=torch.Generator().manual_seed(7))
    keys = keys[order.to(device)]
    return keys, torch.zeros_like(keys)


def _frequencies(keys):
    return torch.arange(keys.numel(), device=keys.device) % 4 + 1


def _tiers(storage):
    return (
        [storage._hbm, storage._host]
        if isinstance(storage, HybridStorage)
        else [storage._state]
    )


def _tier_scores(storage, keys):
    """(founds, scores) of ``keys`` in each tier, by const lookups (no score moves)."""
    tids = torch.zeros_like(keys)
    out = []
    for state in _tiers(storage):
        r = _find_keys(state, keys, tids, const_lookup=True, compact=False)
        out.append((r[5], r[6]))
    return out


def _contents(storage, keys):
    """(found in some tier, value row, score) of every key; no key may be in two tiers."""
    tids = torch.zeros_like(keys)
    found = torch.zeros(keys.numel(), dtype=torch.bool, device=keys.device)
    scores = torch.zeros(keys.numel(), dtype=torch.int64, device=keys.device)
    values = torch.zeros(keys.numel(), DIM, dtype=torch.float32, device=keys.device)
    for state in _tiers(storage):
        r = _find_keys(state, keys, tids, const_lookup=True, compact=False)
        assert not bool((found & r[5]).any()), "a key is in two tiers"
        values = load_from_flat(state, r[7], tids, CopyMode.VALUE, output=values)
        scores = torch.where(r[5], r[6], scores)
        found = found | r[5]
    return found, values, scores


def _assert_same_contents(a, b, keys, lfu):
    found_a, values_a, scores_a = _contents(a, keys)
    found_b, values_b, scores_b = _contents(b, keys)
    assert torch.equal(found_a, found_b)
    assert torch.equal(values_a[found_a], values_b[found_b])
    if lfu:
        assert torch.equal(scores_a[found_a], scores_b[found_b])


# ---------------------------------------------------------------------------
# The reference: what HybridStorage.find / insert did before the masked API
# ---------------------------------------------------------------------------


def reference_find(storage, keys, table_ids, copy_mode, lfu=None):
    (
        n_hbm,
        keys_hbm,
        idx_hbm,
        tids_hbm,
        scores_hbm_in,
        founds_hbm,
        scores_hbm,
        rows_hbm,
    ) = _find_keys(storage._hbm, keys, table_ids, score_source(storage._hbm, lfu))
    values = load_from_flat(storage._hbm, rows_hbm, table_ids, copy_mode=copy_mode)
    if n_hbm == 0:
        return (
            0,
            keys_hbm,
            idx_hbm,
            tids_hbm,
            scores_hbm_in,
            founds_hbm,
            scores_hbm,
            values,
        )
    (
        n_both,
        keys_both,
        idx_both,
        tids_both,
        scores_both,
        founds_host,
        scores_host,
        rows_host,
    ) = _find_keys(storage._host, keys_hbm, tids_hbm, scores_hbm_in)
    host_values = load_from_flat(
        storage._host, rows_host, tids_hbm, copy_mode=copy_mode
    )
    if founds_host.any():
        values[idx_hbm[founds_host]] = host_values[founds_host]
    founds = founds_hbm.clone()
    founds[idx_hbm[founds_host]] = True
    scores = scores_hbm.clone()
    scores[idx_hbm[founds_host]] = scores_host[founds_host]
    return (
        n_both,
        keys_both,
        idx_hbm[idx_both],
        tids_both,
        scores_both,
        founds,
        scores,
        values,
    )


def reference_refresh(storage, keys, table_ids, values):
    """``insert(..., preserve_existing=True)``: store into rows that exist."""
    (
        n_hbm,
        keys_hbm,
        idx_hbm,
        tids_hbm,
        scores_hbm_in,
        founds_hbm,
        _,
        rows_hbm,
    ) = _find_keys(storage._hbm, keys, table_ids)
    if founds_hbm.any():
        store_to_flat(
            storage._hbm,
            rows_hbm[founds_hbm],
            table_ids[founds_hbm],
            values[founds_hbm],
        )
    if n_hbm > 0:
        (_, _, _, _, _, founds_host, _, rows_host) = _find_keys(
            storage._host, keys_hbm, tids_hbm, scores_hbm_in
        )
        if founds_host.any():
            store_to_flat(
                storage._host,
                rows_host[founds_host],
                tids_hbm[founds_host],
                values[idx_hbm[founds_host]],
            )


def reference_insert(storage, keys, table_ids, values, scores=None):
    """``insert``: into the HBM tier, what it evicts or refuses to the host tier."""
    (
        indices,
        num_evicted,
        evicted_keys,
        evicted_tids,
        evicted_idx,
        evicted_scores,
    ) = _insert_and_evict_keys(storage._hbm, keys, table_ids, scores)
    evicted_values = load_from_flat(
        storage._hbm, evicted_idx, evicted_tids, copy_mode=CopyMode.VALUE
    )
    select_insert_failed_values(evicted_idx, values, evicted_values)
    store_to_flat(storage._hbm, indices, table_ids, values)
    if num_evicted != 0:
        _insert_key_values(
            storage._host, evicted_keys, evicted_tids, evicted_values, evicted_scores
        )


def reference_single_find(storage, keys, table_ids, copy_mode, lfu=None):
    """``DynamicEmbStorage.find`` as it was: the compaction came with the lookup."""
    state = storage._state
    (n, mkeys, midx, mtids, mscores, founds, scores, rows) = _find_keys(
        state, keys, table_ids, score_source(state, lfu)
    )
    return (
        n,
        mkeys,
        midx,
        mtids,
        mscores,
        founds,
        scores,
        load_from_flat(state, rows, table_ids, copy_mode=copy_mode),
    )


def _assert_same_find(new, old, lfu):
    (
        n_new,
        keys_new,
        idx_new,
        tids_new,
        scores_new,
        founds_new,
        out_new,
        vals_new,
    ) = new
    (
        n_old,
        keys_old,
        idx_old,
        tids_old,
        scores_old,
        founds_old,
        out_old,
        vals_old,
    ) = old
    assert isinstance(n_new, int) and n_new == n_old
    assert torch.equal(keys_new, keys_old)
    assert torch.equal(idx_new, idx_old)
    assert torch.equal(tids_new, tids_old)
    assert (scores_new is None) == (scores_old is None)
    if scores_new is not None:
        assert torch.equal(scores_new.long(), scores_old.long())
    assert torch.equal(founds_new, founds_old)
    assert torch.equal(vals_new[founds_new], vals_old[founds_old])
    if lfu:
        assert torch.equal(out_new[founds_new], out_old[founds_old])


# ---------------------------------------------------------------------------
# find == the reference, tier scores included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("absent", [True, False])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("lfu", [False, True])
def test_find_equals_reference(device, lfu, training, absent):
    """Keys in the HBM tier, in the host tier and in neither (or only the first two)."""
    a, b = _twins(lfu, device)
    a.training = b.training = training
    keys, tids = _batch(device, absent)
    freq = _frequencies(keys) if lfu else None

    new = a.find(keys, tids, CopyMode.VALUE, freq)
    old = reference_find(b, keys, tids, CopyMode.VALUE, freq)

    _assert_same_find(new, old, lfu)
    everything = torch.cat([*_hbm_and_host_keys(device), _absent_keys(device)])
    _assert_same_contents(a, b, everything, lfu)


@pytest.mark.parametrize("copy_mode", [CopyMode.EMBEDDING, CopyMode.VALUE])
def test_find_embedding_mode_equals_reference(device, copy_mode):
    a, b = _twins(True, device)
    keys, tids = _batch(device)
    new = a.find(keys, tids, copy_mode, _frequencies(keys))
    old = reference_find(b, keys, tids, copy_mode, _frequencies(keys))
    _assert_same_find(new, old, True)


@pytest.mark.parametrize("training", [False, True])
def test_lru_scores_move_for_the_looked_up_keys_only(device, training):
    """A find bumps the timestamp of the keys it looked up, in whichever tier holds
    them, and of no other key; outside training it bumps nothing."""
    a, b = _twins(False, device)
    a.training = b.training = training
    hbm_keys, host_keys = _hbm_and_host_keys(device)
    all_keys = torch.cat([hbm_keys, host_keys])
    asked = all_keys[all_keys % 2 == 0]
    tids = torch.zeros_like(asked)
    prior = [_tier_scores(s, all_keys) for s in (a, b)]

    a.find(asked, tids, CopyMode.VALUE)
    reference_find(b, asked, tids, CopyMode.VALUE)

    was_asked = all_keys % 2 == 0
    for storage, before in zip((a, b), prior):
        for (found0, s0), (found1, s1) in zip(before, _tier_scores(storage, all_keys)):
            assert torch.equal(found0, found1)
            moved, kept = found1 & was_asked, found1 & ~was_asked
            assert torch.equal(s0[kept], s1[kept])
            if training:
                assert bool((s1[moved] > s0[moved]).all())
            else:
                assert torch.equal(s0[moved], s1[moved])


@pytest.mark.parametrize("kind", ["hybrid", "single"])
@pytest.mark.parametrize("lfu", [False, True])
def test_find_result_contract(device, kind, lfu):
    """The 8-tuple callers rely on, checked against what the keys are known to be."""
    storage = _hybrid(lfu) if kind == "hybrid" else _single(lfu)
    _populate(storage, lfu, device)
    storage.training = True
    keys, tids = _batch(device)
    freq = _frequencies(keys) if lfu else None

    (
        n_missing,
        missing_keys,
        missing_idx,
        missing_tids,
        missing_scores,
        founds,
        _,
        values,
    ) = storage.find(keys, tids, CopyMode.VALUE, freq)

    absent = keys >= 1001
    positions = torch.nonzero(absent).flatten()
    assert isinstance(n_missing, int) and n_missing == positions.numel()
    assert torch.equal(founds, ~absent)
    assert torch.equal(missing_idx, positions)
    assert torch.equal(missing_keys, keys[positions])
    assert torch.equal(missing_tids, tids[positions])
    if lfu:
        assert torch.equal(missing_scores.long(), freq[positions])
    else:
        assert missing_scores is None
    assert values.shape == (keys.numel(), DIM)
    assert torch.equal(values[founds], _values(keys)[founds])


@pytest.mark.parametrize("lfu", [False, True])
def test_single_storage_find_equals_reference(device, lfu):
    a, b = _single(lfu), _single(lfu)
    _populate(a, lfu, device)
    _populate(b, lfu, device)
    a.training = b.training = True
    keys, tids = _batch(device)
    freq = _frequencies(keys) if lfu else None
    new = a.find(keys, tids, CopyMode.VALUE, freq)
    old = reference_single_find(b, keys, tids, CopyMode.VALUE, freq)
    _assert_same_find(new, old, lfu)
    everything = torch.cat([*_hbm_and_host_keys(device), _absent_keys(device)])
    _assert_same_contents(a, b, everything, lfu)


# ---------------------------------------------------------------------------
# Masked lanes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lfu", [False, True])
def test_masked_lanes_are_not_looked_up(device, lfu):
    """A masked lane is not found, owns no row in either tier, moves no score and is
    not loaded; the lanes around it are looked up and loaded as ever."""
    storage = _hybrid(lfu)
    _populate(storage, lfu, device)
    hbm_keys, host_keys = _hbm_and_host_keys(device)
    keys = torch.cat([hbm_keys, host_keys])
    tids = torch.zeros_like(keys)
    keep = keys % 3 != 0
    prior = _tier_scores(storage, keys)

    storage.training = True
    found = storage.lookup_masked(
        sf.mask_keys(keys, keep), tids, _frequencies(keys) if lfu else None
    )

    assert isinstance(found, MaskedFind)
    assert torch.equal(found.founds, keep)
    rows_hbm, rows_host = found.rows
    in_hbm = keys < 100
    assert torch.equal(rows_hbm >= 0, keep & in_hbm)
    assert torch.equal(rows_host >= 0, keep & ~in_hbm)
    for (found0, s0), (found1, s1) in zip(prior, _tier_scores(storage, keys)):
        assert torch.equal(found0, found1)
        masked, moved = found1 & ~keep, found1 & keep
        assert torch.equal(s0[masked], s1[masked])
        assert bool((s1[moved] > s0[moved]).all())

    untouched = -7.0
    out = torch.full((keys.numel(), DIM), untouched, device=device)
    returned = storage.load_masked(found, tids, CopyMode.VALUE, output=out)
    assert returned is out
    assert bool((out[~keep] == untouched).all())
    assert torch.equal(out[keep], _values(keys)[keep])


@pytest.mark.parametrize("lfu", [False, True])
def test_a_key_the_hbm_tier_has_is_not_asked_of_the_host_tier(device, lfu):
    """The host tier is looked up for the HBM tier's misses only. A key both tiers
    hold is not a state the storage reaches, but it shows which tier a lookup
    touched: its score moves in the HBM tier and stays in the host tier."""
    storage = _hybrid(lfu)
    hbm_keys, _ = _hbm_and_host_keys(device)
    tids = torch.zeros_like(hbm_keys)
    _fill(storage._hbm, hbm_keys, 3 if lfu else None)
    _fill(storage._host, hbm_keys, 5 if lfu else None)
    hbm_before, host_before = [s for _, s in _tier_scores(storage, hbm_keys)]

    storage.training = True
    found = storage.lookup_masked(hbm_keys, tids, None)
    hbm_after, host_after = [s for _, s in _tier_scores(storage, hbm_keys)]

    assert bool(found.founds.all())
    assert bool((found.rows[1] == -1).all())
    assert bool((hbm_after > hbm_before).all())
    assert torch.equal(host_after, host_before)


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("lfu", [False, True])
def test_refresh_equals_reference(device, lfu, training):
    """The backward's refresh: new values into the rows that exist, in either tier,
    and nothing for a key neither tier has; tier scores follow ``training``."""
    a, b = _twins(lfu, device)
    a.training = b.training = training
    keys, tids = _batch(device)
    fresh = _values(keys) * 2 + 0.5
    _, _, before = _contents(a, keys)

    a.insert(keys, tids, fresh, preserve_existing=True)
    reference_refresh(b, keys, tids, fresh)

    everything = torch.cat([*_hbm_and_host_keys(device), _absent_keys(device)])
    found, values, after = _contents(a, keys)
    assert torch.equal(found, keys < 1001)
    assert torch.equal(values[found], fresh[found])
    _assert_same_contents(a, b, everything, lfu)
    # the refresh's lookups score like any other: they bump what they find while
    # training (a count of 1 for LFU, a newer timestamp for LRU) and nothing otherwise
    if not training:
        assert torch.equal(before[found], after[found])
    elif lfu:
        assert torch.equal(after[found], before[found] + 1)
    else:
        assert bool((after[found] > before[found]).all())


def _spill_batches(device, lfu):
    """Three batches into an HBM tier of 128 slots: the first fits, the second evicts
    the lowest-scored of the first, the third scores below everything: each of its keys
    still evicts the lowest-scored resident (a low score never refuses a key, a newcomer is
    compared with the largest score there is), and no key of the call is evicted by the
    call, which is what keeps every value on its key."""
    out = []
    for first, last, score in [(1, 101, None), (101, 161, None), (1001, 1011, 1)]:
        keys = torch.arange(first, last, dtype=torch.int64, device=device)
        scores = (
            None
            if not lfu
            else (keys + 9 if score is None else torch.full_like(keys, score))
        )
        out.append((keys, scores))
    return out


@pytest.mark.parametrize("lfu", [False, True])
def test_insert_spills_what_the_hbm_tier_cannot_keep(device, lfu, deterministic):
    """What the HBM tier evicts lands in the host tier with its value and score; nothing
    is lost, and the result equals the reference's. (What it refuses is
    ``test_a_key_the_hbm_tier_refuses_reaches_the_host_tier_with_its_own_row``.)"""
    a, b = _hybrid(lfu), _hybrid(lfu)
    batches = _spill_batches(device, lfu)
    for keys, scores in batches:
        tids = torch.zeros_like(keys)
        a.insert(keys, tids, _values(keys), scores)
        reference_insert(b, keys, tids, _values(keys), scores)

    everything = torch.cat([keys for keys, _ in batches])
    for storage in (a, b):
        found, values, _ = _contents(storage, everything)
        assert bool(found.all()), "a key was lost between the tiers"
        assert torch.equal(values, _values(everything))
    assert int(a._host.key_index_map.size()) > 0, "nothing spilled: the test is idle"
    if lfu:
        # the refused batch's score on the host tier is the kernel's business: compare
        # the keys whose score the tiers carry over
        carried = torch.cat([keys for keys, _ in batches[:2]])
        _assert_same_contents(a, b, carried, lfu)


@pytest.mark.parametrize("variant", ["plain", "interleaved"])
@pytest.mark.parametrize("lfu", [False, True])
def test_a_key_the_hbm_tier_refuses_reaches_the_host_tier_with_its_own_row(
    device, lfu, variant, deterministic
):
    """One call brings more new keys than the HBM tier has slots (one bucket of 128; the
    tier keeps no reference counters, so this is the way it refuses a key). The keys it has
    no slot for come back from the insert as evicted entries of index ``-(input position +
    1)``, and ``HybridStorage.insert`` hands each to the host tier with its OWN row
    (``_select_refused_values``), also with masked lanes among the keys, whose positions
    count. Nothing is lost, no key is in both tiers, the HBM tier is full. In the
    determinism mode the keys of a bucket are inserted in key order, so the tier keeps the
    smallest keys of the call."""
    storage = _hybrid(lfu)
    keys = torch.arange(1, 201, dtype=torch.int64, device=device)
    n = keys.numel()
    real = (
        torch.ones(n, dtype=torch.bool, device=device)
        if variant == "plain"
        else torch.arange(n, device=device) % 4 != 3
    )
    nan_rows = torch.full((n, DIM), float("nan"), device=device)
    values = torch.where(real.unsqueeze(1), _values(keys), nan_rows)
    storage.insert(
        sf.mask_keys(keys, real),
        torch.zeros_like(keys),
        values,
        (keys + 9) if lfu else None,
    )

    found, rows, _ = _contents(storage, keys)
    assert torch.equal(found, real), "a key was lost between the tiers"
    assert torch.equal(rows[found], _values(keys)[found]), "a key got another key's row"
    (in_hbm, _), (in_host, _) = _tier_scores(storage, keys)
    assert int(in_hbm.sum()) == BUCKET, "the HBM tier is full"
    assert (
        int(in_host.sum()) == int(real.sum()) - BUCKET
    ), "the rest is in the host tier"
    if deterministic:
        admitted = real & (torch.cumsum(real.to(torch.int64), 0) <= BUCKET)
        assert torch.equal(in_hbm, admitted), "the smallest keys of the call are kept"


@pytest.mark.parametrize("variant", ["tail", "interleaved"])
@pytest.mark.parametrize("lfu", [False, True])
def test_insert_with_reserved_lanes_equals_the_exact_insert(
    device, lfu, variant, deterministic
):
    """Arrays with lanes that must not take part (a padded tail of the reserved key
    and table id -1, or masked keys among real ones) insert exactly the real keys."""
    a, b = _hybrid(lfu), _hybrid(lfu)
    keys = torch.arange(1, 161, dtype=torch.int64, device=device)
    n = keys.numel()
    scores = (keys + 9) if lfu else None
    nan_rows = torch.full((n, DIM), float("nan"), device=device)
    if variant == "tail":
        pad = 37
        real = torch.ones(n, dtype=torch.bool, device=device)
        padded_keys = torch.cat([keys, torch.full((pad,), EMPTY, device=device)])
        padded_tids = torch.full((n + pad,), -1, dtype=torch.int64, device=device)
        padded_tids[:n] = 0
        padded_values = torch.cat([_values(keys), nan_rows[:pad]])
        padded_scores = (
            torch.cat([scores, torch.zeros(pad, dtype=torch.int64, device=device)])
            if lfu
            else None
        )
    else:
        real = torch.arange(n, device=device) % 4 != 3
        padded_keys = sf.mask_keys(keys, real)
        padded_tids = torch.zeros_like(keys)
        padded_values = torch.where(real.unsqueeze(1), _values(keys), nan_rows)
        padded_scores = scores

    a.insert(padded_keys, padded_tids, padded_values, padded_scores)
    kept = keys[real]
    reference_insert(
        b,
        kept,
        torch.zeros_like(kept),
        _values(kept),
        scores[real] if lfu else None,
    )

    for storage in (a, b):
        found, values, _ = _contents(storage, keys)
        assert torch.equal(found, real)
        assert torch.equal(values[found], _values(keys)[found])
    _assert_same_contents(a, b, keys, lfu=False)


# ---------------------------------------------------------------------------
# What a refused key gets, and what the inert tail of the evicted arrays does not
# ---------------------------------------------------------------------------


def _evicted_arrays(device, size, real):
    """``size`` evicted entries: ``real`` evictions (rows of the table: nothing to
    copy for them) and an inert tail (row -1, table id -1)."""
    indices = torch.full((size,), -1, dtype=torch.int64, device=device)
    table_ids = torch.full((size,), -1, dtype=torch.int64, device=device)
    indices[:real] = torch.arange(real, dtype=torch.int64, device=device) + 100
    table_ids[:real] = 0
    return indices, table_ids


@pytest.mark.parametrize("narrow", [False, True])
def test_select_keeps_the_inert_tail_out_of_the_kernel(device, narrow):
    """The kernel takes the tail's row -1 for the refused insert of input row 0, and ONE
    row stride, ``values.size(1)``, for the rows it reads and the rows it writes. Run
    over the tail it would copy row 0 over every tail row, and over rows of real
    evictions when ``values`` is narrower than ``evicted_values``."""
    size, real, wide = 64, 12, 2 * DIM
    width = DIM if narrow else wide
    evicted_values = torch.arange(size * wide, dtype=torch.float32, device=device).view(
        size, wide
    )
    values = -1 - torch.arange(size * width, dtype=torch.float32, device=device).view(
        size, width
    )
    indices, table_ids = _evicted_arrays(device, size, real)
    before = evicted_values.clone()

    kvt._select_refused_values(indices, table_ids, values, evicted_values)

    assert torch.equal(evicted_values, before)


def test_select_copies_the_row_of_a_refused_key(device):
    """A refused insert is an evicted entry of index ``-(i + 1)`` and the table id of its
    key: it gets row ``i`` of the input, row 0 (index -1) included, and nothing else is
    written."""
    size, real = 32, 8
    values = (
        torch.arange(size * DIM, dtype=torch.float32, device=device).view(size, DIM)
        + 1000
    )
    evicted_values = torch.zeros(size, DIM, device=device)
    indices, table_ids = _evicted_arrays(device, size, real)
    # entry 2 is the refused insert of input row 5, entry 3 that of input row 0
    indices[2] = -6
    indices[3] = -1

    kvt._select_refused_values(indices, table_ids, values, evicted_values)

    assert torch.equal(evicted_values[2], values[5])
    assert torch.equal(evicted_values[3], values[0])
    others = torch.ones(size, dtype=torch.bool, device=device)
    others[2] = False
    others[3] = False
    assert bool((evicted_values[others] == 0).all())


def _two_table_hybrid(hbm_capacity, widest):
    """Two tables of dims ``DIM`` and ``widest`` fused into one table state per tier.
    The state's rows are as wide as the wider table's, so a row of the narrower table
    (what its checkpoint holds) is narrower than a row of the state."""
    return HybridStorage(
        [_options(False, hbm_capacity, False, dim=d) for d in (DIM, widest)],
        [_options(False, 8 * BUCKET, True, dim=d) for d in (DIM, widest)],
        _optimizer(),
    )


def _narrow_table_rows(keys, widest):
    """Rows of the narrower table in the state's layout: the key in every embedding
    column, the rest of the (wider) row unused."""
    rows = torch.zeros(keys.numel(), widest, device=keys.device)
    rows[:, :DIM] = keys.to(torch.float32).unsqueeze(1)
    return rows


@pytest.mark.parametrize("widest", [DIM, 2 * DIM])
def test_load_keeps_the_evicted_rows_of_a_narrower_table(device, tmp_path, widest):
    """``load`` into an HBM tier that has to evict: the evicted keys reach the host tier
    with their own rows. The evicted arrays are as long as the batch; the keys that
    took a free slot instead of evicting make their tail, and a checkpoint's rows are
    those of the narrower table, narrower than the state's."""
    loaded = torch.arange(1, 65, dtype=torch.int64, device=device)
    residents = torch.arange(1001, 1097, dtype=torch.int64, device=device)
    everything = torch.cat([loaded, residents])
    tids = torch.zeros_like(everything)

    source = _two_table_hybrid(8 * BUCKET, widest)
    _insert_key_values(
        source._hbm, loaded, tids[: loaded.numel()], _narrow_table_rows(loaded, widest)
    )
    names = ["meta.json", "keys.bin", "embeddings.bin", "scores.bin", "optimizer.bin"]
    paths = [str(tmp_path / name) for name in names]
    source.dump(0, *paths)

    # Table 0 has ONE bucket of BUCKET slots in the HBM tier. 96 residents and 64 keys
    # to load: 32 of the keys take the free slots and 32 evict a resident, so every key
    # finds a slot (none is refused) and 32 residents move to the host tier. The
    # checkpoint's scores are left out: the keys loaded get the tier's own LRU clock.
    target = _two_table_hybrid(BUCKET, widest)
    _insert_key_values(
        target._hbm,
        residents,
        tids[loaded.numel() :],
        _narrow_table_rows(residents, widest),
    )
    target.load(0, paths[0], paths[1], paths[2], None, paths[4])

    values = torch.zeros(everything.numel(), widest, device=device)
    held = []
    for state in (target._hbm, target._host):
        r = _find_keys(state, everything, tids, const_lookup=True, compact=False)
        values = load_from_flat(state, r[7], tids, CopyMode.EMBEDDING, output=values)
        held.append(r[5])
    in_hbm, in_host = held
    assert torch.equal(
        in_hbm ^ in_host, torch.ones_like(in_hbm)
    ), "a key is lost or in both tiers"
    assert bool(in_host.any()), "nothing was evicted: the test is idle"
    assert torch.equal(values[:, :DIM], _narrow_table_rows(everything, widest)[:, :DIM])


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lfu", [False, True])
def test_empty_batch(device, lfu):
    storage = _hybrid(lfu)
    _populate(storage, lfu, device)
    keys = torch.empty(0, dtype=torch.int64, device=device)
    tids = torch.empty(0, dtype=torch.int64, device=device)
    freq = keys.clone() if lfu else None

    found = storage.lookup_masked(keys, tids, freq)
    assert found.founds.numel() == 0 and found.scores.numel() == 0
    assert storage.load_masked(found, tids, CopyMode.VALUE).shape == (0, DIM)

    n_missing, missing_keys, *_, founds, scores, values = storage.find(
        keys, tids, CopyMode.VALUE, freq
    )
    assert n_missing == 0 and missing_keys.numel() == 0
    assert founds.numel() == 0 and scores.numel() == 0 and values.shape == (0, DIM)

    empty_values = torch.empty(0, DIM, device=device)
    storage.insert(keys, tids, empty_values, freq)
    storage.insert(keys, tids, empty_values, preserve_existing=True)
    everything = torch.cat([*_hbm_and_host_keys(device)])
    assert bool(_contents(storage, everything)[0].all())


# ---------------------------------------------------------------------------
# What the masked API does not do: read anything back
# ---------------------------------------------------------------------------


def _count_compactions(monkeypatch):
    calls = []
    real = kvt.flagged_compact

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(kvt, "flagged_compact", counting)
    return calls


@pytest.mark.parametrize("kind", ["hybrid", "single"])
def test_only_find_compacts(device, monkeypatch, kind):
    """``find`` is the host-int contract and costs ONE compaction (it was two for
    HybridStorage, one for DynamicEmbStorage); the masked API costs none."""
    storage = _hybrid(True) if kind == "hybrid" else _single(True)
    _populate(storage, True, device)
    storage.training = True
    keys, tids = _batch(device)
    freq = _frequencies(keys)
    values = _values(keys)
    calls = _count_compactions(monkeypatch)

    found = storage.lookup_masked(keys, tids, freq)
    storage.load_masked(found, tids, CopyMode.VALUE)
    if kind == "hybrid":
        storage.insert(keys, tids, values, preserve_existing=True)
        storage.insert(keys, tids, values, freq)
    assert calls == []

    storage.find(keys, tids, CopyMode.VALUE, freq)
    assert len(calls) == 1


def test_masked_api_has_no_python_visible_sync(device):
    """``torch.cuda.set_sync_debug_mode`` raises on ``.item()``, ``.cpu()``,
    ``nonzero`` and a tensor in an ``if``. (It does not see a sync made inside the
    extension: the census of ``no_host_sync`` does.)"""
    storage = _hybrid(True)
    _populate(storage, True, device)
    storage.training = True
    keys, tids = _batch(device)
    freq = _frequencies(keys)
    values = _values(keys)
    spill = torch.arange(2001, 2201, dtype=torch.int64, device=device)
    spill_tids = torch.zeros_like(spill)
    spill_values = _values(spill)
    spill_scores = spill + 9
    # warm up: first use of each kernel may load a module
    storage.load_masked(storage.lookup_masked(keys, tids, freq), tids, CopyMode.VALUE)
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        found = storage.lookup_masked(keys, tids, freq)
        storage.load_masked(found, tids, CopyMode.VALUE)
        storage.insert(keys, tids, values, preserve_existing=True)
        storage.insert(spill, spill_tids, spill_values, spill_scores)
    finally:
        torch.cuda.set_sync_debug_mode("default")
