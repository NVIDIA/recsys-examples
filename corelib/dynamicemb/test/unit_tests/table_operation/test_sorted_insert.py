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

"""The sorted inserts of DEMB_DETERMINISM_MODE against the waves they replace.

``table_insert_sorted`` / ``table_insert_and_evict_sorted`` sort the keys of a batch by
(home bucket, key) and let one thread insert the keys of each bucket, smallest first
(src/table_operation/sorted_insert.cuh). A slot a key takes stays locked until the run of
its bucket is done, as in the ordinary insert kernels: no key of a call is evicted by a later
key of the call, and a bucket that gets more new keys than it has free or evictable slots
refuses the last of them. The waves (``wave_oracle.py``, the code that the library used to
run, with the slots a wave took pinned until the last wave) did the same per bucket with one
launch per rank of a key in its bucket. Two fresh tables with the same construction (the
oracle's one has reference counters, which it pins with) are driven with the same batches,
one by the waves and one by the sorted insert, and must stay equal:

* the key table: keys and digests, the scores of every policy but GLOBAL_TIMER (their values
  come from the device timer and differ between two runs, their order inside a bucket does
  not), and ``bucket_sizes``;
* the indices of the batch (a lookup after all inserts);
* the evictions as a multiset of (key, table, index, score): the order of the records is a
  race in the waves and sorted by (bucket, key) here, and a refused key (index ``-(i + 1)``
  of its input position) is compared as "negative" only, since the waves report the number
  of its lane in the wave;
* the overflow bucket of a cache, whenever the call has at most one spill per table.

A call that spills more than once per table cannot be compared with the waves: they unlock
the overflow slots between two waves, so a later spill may evict an earlier one, which the
sorted insert never does (a slot a spill takes stays locked until every spill of the call is
placed). ``test_multi_spill_keeps_the_keys_of_the_call`` states what the new pass guarantees
instead. The batches of a ``cache`` comparison keep every bucket to at most its capacity of
new keys (``at_most_per_bucket``) and pin nothing, so that no key is refused and nothing
spills; the spills have their own tests.

The tests drive the library through ``new_insert`` / ``new_insert_and_evict``, which call
``LinearBucketTable._deterministic_insert*``: the Python glue (the indices of the final lookup,
the evicted arrays, the host form of the count) is covered with the kernels.
"""

import os
import sys

import dynamicemb_extensions as ext
import pytest
import torch
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import wave_oracle as oracle  # noqa: E402

pytestmark = pytest.mark.skipif(
    not getattr(ext, "sorted_insert", False),
    reason="this dynamicemb_extensions has no sorted insert: rebuild it "
    "(python setup.py build_ext --inplace)",
)

KEY_TYPES = [torch.int64, torch.uint64]
POLICIES = [
    ScorePolicy.ASSIGN,
    ScorePolicy.ACCUMULATE,
    ScorePolicy.CONST,
    ScorePolicy.FLAG,
    ScorePolicy.GLOBAL_TIMER,
]
# Policies that read their score from the input; the others read none and are given None.
SCORED = (ScorePolicy.ASSIGN, ScorePolicy.ACCUMULATE)
# The scores of a timestamp policy come from the device timer: they are not compared.
TIMER = ScorePolicy.GLOBAL_TIMER
OPS = ["insert", "insert_and_evict", "cache"]
GARBAGE = 0xDEAD0000  # the score of an inert lane: never read for a key that takes part

# capacity per table, slots per bucket, keys per batch
GEOMETRIES = {
    "dense": ([64, 64], 16, 600),  # 75 keys per bucket: eviction all the time
    "wide": ([128 * 256], 128, 4096),
    "zero_first": ([0, 64], 16, 300),
    "zero_mid": ([64, 0, 64], 16, 300),
    "zero_last": ([64, 0], 16, 300),
    "single_bucket": ([16], 16, 100),
}
EMPTY = -1
WIDE_MULTIPLIER = -7046029254386353131  # 0x9E3779B97F4A7C15 as an int64


# ---------------------------------------------------------------------------
# tables and drivers
# ---------------------------------------------------------------------------
def i64(t):
    return t.view(torch.int64) if t.dtype == torch.uint64 else t


def u64(values, device="cuda"):
    """A uint64 CUDA tensor of the non-negative ``values`` (a list, or an int64 tensor):
    torch has no uint64 arange/tensor/cat/index on every build, so those go through int64.
    """
    return torch.as_tensor(values, dtype=torch.int64, device=device).view(torch.uint64)


def take(t, index):
    """``t[index]`` for a tensor that may be uint64, which has no advanced indexing."""
    return i64(t)[index].view(t.dtype)


def make_table(key_type, policy, capacity, bucket, overflow=False, ref_counter=True):
    return get_scored_table(
        capacity=capacity,
        bucket_capacity=bucket,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score", policy=policy)],
        enable_overflow=overflow,
        use_ref_counter=ref_counter,
    )


def make_pair(key_type, policy, capacity, bucket, overflow=False, ref_counter=True):
    """``(oracle table, table under test)``, equal tables except for ``ref_counter``, which
    only the second one follows: the waves pin the slots they took with the counters, which
    are all zero outside the oracle's own pins, so the oracle decides as a table without them
    does."""
    return (
        make_table(key_type, policy, capacity, bucket, overflow, True),
        make_table(key_type, policy, capacity, bucket, overflow, ref_counter),
    )


def new_insert(
    table, keys, tids, score_value, policy, insert_results=None, score_out=None
):
    """The sorted insert, through the library (``LinearBucketTable._deterministic_insert``)."""
    return table._deterministic_insert(
        keys, tids, score_value, policy, insert_results, score_out
    )


def new_insert_and_evict(
    table, keys, tids, score_value, policy, insert_results=None, score_out=None
):
    """The sorted insert that reports its evictions (``_deterministic_insert_and_evict`` and
    ``..._with_overflow``): ``(indices, num_evicted [1] on the device, evicted keys, indices,
    scores, table ids)``, the evicted arrays as long as ``keys``."""
    insert = (
        table._deterministic_insert_and_evict_with_overflow
        if table.enable_overflow_
        else table._deterministic_insert_and_evict
    )
    return insert(
        keys,
        tids,
        score_value,
        policy,
        async_count=True,
        insert_results=insert_results,
        score_out=score_out,
    )


def run_old(table, op, keys, tids, score_value, policy):
    """The waves: ``(indices, records)``, records = (count, keys, indices, scores, table ids)
    sliced to the count."""
    if op == "insert":
        return oracle.insert(table, keys, tids, score_value, policy), None
    fn = (
        oracle.insert_and_evict_with_overflow
        if table.enable_overflow_
        else oracle.insert_and_evict
    )
    indices, count, ev_keys, ev_indices, ev_scores, ev_tids = fn(
        table, keys, tids, score_value, policy
    )
    return indices, (count, ev_keys, ev_indices, ev_scores, ev_tids)


def run_new(
    table, op, keys, tids, score_value, policy, insert_results=None, score_out=None
):
    """The sorted insert: the same shape as ``run_old``, plus the tails of the evicted
    arrays are checked to be inert before they are cut."""
    if op == "insert":
        indices = new_insert(
            table, keys, tids, score_value, policy, insert_results, score_out
        )
        return indices, None
    (
        indices,
        num_evicted,
        ev_keys,
        ev_indices,
        ev_scores,
        ev_tids,
    ) = new_insert_and_evict(
        table, keys, tids, score_value, policy, insert_results, score_out
    )
    count = int(num_evicted.item())
    n = keys.numel()
    assert ev_keys.numel() == ev_indices.numel() == ev_scores.numel() == n
    assert ev_tids.numel() == n and 0 <= count <= n
    assert bool((i64(ev_keys)[count:] == EMPTY).all()), "the key tail is inert"
    assert bool((ev_indices[count:] == -1).all()), "the index tail is inert"
    assert bool((ev_tids[count:] == -1).all()), "the table id tail is inert"
    assert bool((ev_scores.view(torch.int64)[count:] == 0).all()), "the score tail is 0"
    return indices, (
        count,
        ev_keys[:count],
        ev_indices[:count],
        ev_scores[:count],
        ev_tids[:count],
    )


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------
def make_batch(gen, n, num_tables, key_type, policy, pad=0.1, zeros=False):
    """``n`` lanes of distinct (key, table) pairs: the same key in several tables, small keys and
    wide ones (negative as int64 and with the top bit set as uint64, none reserved), and about
    ``pad`` of the lanes the padding of an N-aligned batch: the reserved key, table id -1 and a
    garbage score. Returns CUDA tensors (keys, table ids, scores). ASSIGN and ACCUMULATE read
    their score from the input; for the other policies the score is None, or zeros when
    ``zeros`` (an input that is given and not used)."""
    per_table = max(1, n // num_tables)
    base = torch.randperm(4 * per_table * num_tables, generator=gen)[:per_table] + 1
    keys = base.repeat(num_tables)
    tids = torch.arange(num_tables, dtype=torch.int64).repeat_interleave(per_table)
    wide = keys * WIDE_MULTIPLIER
    wide = torch.where((wide >= -4) & (wide <= -1), wide + 8, wide)
    keys = torch.where(torch.rand(keys.numel(), generator=gen) < 0.5, wide, keys)
    if policy in SCORED:
        high = 9 if policy == ScorePolicy.ASSIGN else 2  # ties on purpose
        scores = torch.randint(1, high, (keys.numel(),), generator=gen)
    else:
        scores = torch.zeros(keys.numel(), dtype=torch.int64) if zeros else None
    perm = torch.randperm(keys.numel(), generator=gen)
    keys, tids = keys[perm], tids[perm]
    scores = None if scores is None else scores[perm]
    padded = torch.rand(keys.numel(), generator=gen) < pad
    keys = torch.where(padded, torch.full_like(keys, EMPTY), keys)
    tids = torch.where(padded, torch.full_like(tids, -1), tids)
    if scores is not None and policy in SCORED:
        scores = torch.where(padded, torch.full_like(scores, GARBAGE), scores)
    return (
        keys.cuda().view(key_type),
        tids.cuda(),
        None if scores is None else u64(scores),
    )


def at_most_per_bucket(table, keys, tids, limit):
    """``keys`` / ``tids`` with every key past the first ``limit`` of its bucket (in key order)
    made a padding lane (the reserved key, table id -1): no bucket gets more than ``limit`` new
    keys. With ``limit`` the capacity of a bucket and no pinned slot a call cannot refuse a
    key: a bucket always has a slot that is empty or holds a resident, and the keys of the call
    lock one each. Done with the sort of the sorted insert, on the device."""
    _, inverse, segments = ext.bucketize_keys_segmented(
        keys,
        tids,
        table.table_bucket_offsets_,
        table.num_buckets_,
        table.bucket_capacity_,
    )
    rank = torch.arange(keys.numel(), device=keys.device) - torch.searchsorted(
        segments, segments
    )
    drop = torch.zeros(keys.numel(), dtype=torch.bool, device=keys.device)
    drop[inverse] = rank >= limit
    return (
        torch.where(drop, torch.full_like(i64(keys), EMPTY), i64(keys)).view(
            keys.dtype
        ),
        torch.where(drop, torch.full_like(tids, -1), tids),
    )


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------
def assert_tables_equal(a, b, policy, overflow=False, what=""):
    """Keys, digests and bucket sizes; the scores unless GLOBAL_TIMER; the overflow bucket
    when asked."""
    assert torch.equal(i64(a.keys_), i64(b.keys_)), "keys differ " + what
    assert torch.equal(a.digests_, b.digests_), "digests differ " + what
    if policy != TIMER:
        assert torch.equal(i64(a.scores_list[0]), i64(b.scores_list[0])), (
            "scores differ " + what
        )
    assert torch.equal(a.bucket_sizes, b.bucket_sizes), "bucket sizes differ " + what
    if overflow:
        ka, kb = a.overflow_table_storage_, b.overflow_table_storage_
        if policy == TIMER:
            # the keys and digests of the overflow bucket: everything but the score words
            # (stored after the 8 + 1 bytes per slot of one bucket's keys and digests)
            def cut(t, table):
                per_bucket = table.overflow_bucket_capacity_ * 9
                full = table.overflow_bucket_capacity_ * 17
                return torch.cat(
                    [
                        t[i * full : i * full + per_bucket]
                        for i in range(table.overflow_num_buckets_)
                    ]
                )

            ka, kb = cut(ka, a), cut(kb, b)
        assert torch.equal(ka, kb), "the overflow bucket differs " + what
        assert torch.equal(a.overflow_bucket_sizes, b.overflow_bucket_sizes), (
            "overflow bucket sizes differ " + what
        )


def records(rec, policy):
    """The evictions as a sorted list of (key, table id, index, score): a refused key's index
    is only "negative" (the waves count lanes, the sorted insert positions), and the scores of
    GLOBAL_TIMER are dropped."""
    count, ev_keys, ev_indices, ev_scores, ev_tids = rec
    return sorted(
        (
            key,
            tid,
            index if index >= 0 else -1,
            0 if policy == TIMER else score,
        )
        for key, index, score, tid in zip(
            i64(ev_keys).tolist(),
            ev_indices.tolist(),
            i64(ev_scores).tolist(),
            ev_tids.tolist(),
        )
    )


def assert_insert_outputs(table, keys, tids, insert_results, score_out, rec):
    """What the sorted insert reports per input position, checked against itself: every position
    has a result; exactly the padding and the keys of tables without buckets are Illegal; the
    evictions are the Evict and Busy results (the final ones: a spill that the overflow bucket
    took is an Insert or Assign); and a key that is still in the table reports the score it
    holds."""
    assert not bool(
        (insert_results == InsertResult.INIT.value).any()
    ), "an insert result was never written"
    reserved = i64(keys) == EMPTY
    offsets = table.table_bucket_offsets_
    buckets = offsets[1:] - offsets[:-1]
    no_buckets = buckets[tids.clamp(min=0)] == 0
    inert = reserved | no_buckets
    illegal = insert_results == InsertResult.ILLEGAL.value
    assert torch.equal(illegal, inert), "exactly the keys that take no part are Illegal"
    if rec is not None:
        evicted = (insert_results == InsertResult.EVICT.value) | (
            insert_results == InsertResult.BUSY.value
        )
        assert rec[0] == int(
            evicted.sum()
        ), "the evictions are the Evict and Busy results"
    score_arg = ScoreArg(name="score", value=None, policy=ScorePolicy.CONST)
    lookup = table.lookup_with_overflow if table.enable_overflow_ else table.lookup
    stored, founds, _ = lookup(keys, tids, score_arg)
    assert bool(
        (stored[founds] == score_out[founds]).all()
    ), "a key that is in the table reports the score it holds"


def erase_some(tables, gen, keys, tids, fraction=0.25):
    """Erase a random part of the keys of the previous batch from every table (the
    reserved lanes among them are no-ops)."""
    pick = (torch.rand(keys.numel(), generator=gen) < fraction).cuda()
    sel_keys = i64(keys)[pick].view(keys.dtype)
    sel_tids = tids[pick]
    for table in tables:
        table.erase(sel_keys, sel_tids)


def pin_some(tables, gen, keys, tids, fraction=0.2):
    """Pin a random part of the residents of the previous batch (reference counters), on every
    table; returns what to unpin. The tables are equal, so one set of slots serves all.
    """
    pick = (torch.rand(keys.numel(), generator=gen) < fraction).cuda()
    sel_keys = i64(keys)[pick].view(keys.dtype)
    sel_tids = tids[pick]
    slots = tables[0].slots_of(sel_keys, sel_tids)
    for table in tables:
        table.increment_counter(slots, sel_tids)
    return slots, sel_tids


def unpin(tables, pinned):
    slots, tids = pinned
    for table in tables:
        table.decrement_counter(slots, tids)


# ---------------------------------------------------------------------------
# equivalence with the waves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
@pytest.mark.parametrize("op", OPS)
def test_equals_the_waves(key_type, policy, geometry, op):
    capacity, bucket, n = GEOMETRIES[geometry]
    old, new = make_pair(key_type, policy, capacity, bucket, overflow=op == "cache")
    num_tables = len(capacity)
    seed = 100 * POLICIES.index(policy) + 10 * list(GEOMETRIES).index(geometry)
    gen = torch.Generator().manual_seed(seed + OPS.index(op))
    previous, pinned = None, None
    for step in range(5):
        keys, tids, scores = make_batch(
            gen, n, num_tables, key_type, policy, zeros=step % 2 == 1
        )
        if op == "cache":
            # no refusal, so nothing spills: the overflow bucket is compared as it is
            keys, tids = at_most_per_bucket(new, keys, tids, bucket)
        if previous is not None:
            # erase a quarter of the last batch (reclaimed slots), move the pins
            erase_some((old, new), gen, *previous)
            if pinned is not None:
                unpin((old, new), pinned)
            if op == "insert_and_evict":
                pinned = pin_some((old, new), gen, *previous)
        idx_old, rec_old = run_old(old, op, keys, tids, scores, policy)
        insert_results = torch.full(
            (keys.numel(),), InsertResult.INIT.value, dtype=torch.uint8, device="cuda"
        )
        score_out = torch.full((keys.numel(),), -777, dtype=torch.int64, device="cuda")
        idx_new, rec_new = run_new(
            new, op, keys, tids, scores, policy, insert_results, score_out
        )
        what = "(batch %d)" % step
        assert_tables_equal(old, new, policy, overflow=op == "cache", what=what)
        assert torch.equal(idx_old, idx_new), "indices differ " + what
        if op != "insert":
            assert records(rec_old, policy) == records(rec_new, policy), (
                "evictions differ " + what
            )
        assert_insert_outputs(new, keys, tids, insert_results, score_out, rec_new)
        previous = (keys, tids)


@pytest.mark.parametrize("policy", [ScorePolicy.ASSIGN, TIMER])
@pytest.mark.parametrize("op", ["insert", "insert_and_evict"])
@pytest.mark.parametrize("buckets", [8192, 256], ids=["fits", "evicts"])
def test_equals_the_waves_on_a_large_batch(policy, op, buckets):
    """Two batches of 65,536 keys into 8,192 buckets of 128 slots (runs of a few dozen keys, no
    eviction: it fits) or into 256 buckets (about 250 keys per bucket: evictions, and a few
    hundred waves for the oracle)."""
    old, new = make_pair(torch.int64, policy, [128 * buckets], 128)
    gen = torch.Generator().manual_seed(5)
    for step in range(2):
        keys, tids, scores = make_batch(gen, 65536, 1, torch.int64, policy, pad=0.05)
        idx_old, rec_old = run_old(old, op, keys, tids, scores, policy)
        idx_new, rec_new = run_new(new, op, keys, tids, scores, policy)
        assert_tables_equal(old, new, policy, what="(batch %d)" % step)
        assert torch.equal(idx_old, idx_new)
        if op != "insert":
            assert records(rec_old, policy) == records(rec_new, policy)


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("policy", [ScorePolicy.ASSIGN, ScorePolicy.ACCUMULATE])
@pytest.mark.parametrize("op", ["insert", "insert_and_evict"])
def test_a_table_without_a_ref_counter(key_type, policy, op):
    """The storage under a cache keeps no reference counters (the kernels get a null
    counter): the same equivalence, and nothing can be pinned."""
    old, new = make_pair(key_type, policy, [64, 64], 16, ref_counter=False)
    assert new._ref_counter is None
    gen = torch.Generator().manual_seed(77)
    previous = None
    for step in range(4):
        keys, tids, scores = make_batch(gen, 600, 2, key_type, policy)
        if previous is not None:
            erase_some((old, new), gen, *previous)
        idx_old, rec_old = run_old(old, op, keys, tids, scores, policy)
        idx_new, rec_new = run_new(new, op, keys, tids, scores, policy)
        assert_tables_equal(old, new, policy, what="(batch %d)" % step)
        assert torch.equal(idx_old, idx_new)
        if op != "insert":
            assert records(rec_old, policy) == records(rec_new, policy)
        previous = (keys, tids)


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("policy", [ScorePolicy.CONST, ScorePolicy.FLAG, TIMER])
@pytest.mark.parametrize("op", OPS)
def test_no_score_input(key_type, policy, op):
    """CONST, FLAG and GLOBAL_TIMER read no score: the input may be None (or zeros)."""
    old, new = make_pair(key_type, policy, [64, 64], 16, overflow=op == "cache")
    gen = torch.Generator().manual_seed(9)
    keys, tids, _ = make_batch(gen, 600, 2, key_type, policy, pad=0.2)
    if op == "cache":
        keys, tids = at_most_per_bucket(new, keys, tids, 16)  # nothing spills
    idx_old, rec_old = run_old(old, op, keys, tids, None, policy)
    insert_results = torch.empty(keys.numel(), dtype=torch.uint8, device="cuda")
    idx_new, rec_new = run_new(new, op, keys, tids, None, policy, insert_results)
    assert_tables_equal(old, new, policy, overflow=op == "cache")
    assert torch.equal(idx_old, idx_new)
    padded = i64(keys) == EMPTY
    assert bool(padded.any())
    assert bool((insert_results[padded] == InsertResult.ILLEGAL.value).all())
    assert bool((idx_new[padded] == -1).all())


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("policy", [ScorePolicy.ASSIGN, ScorePolicy.ACCUMULATE])
def test_garbage_scores_in_the_padding_stay_out_of_the_table(key_type, policy):
    """ASSIGN and ACCUMULATE read their input at every lane, the padding's included, whatever
    is there (here 0xDEAD0000): it stays out of the table, which is the one the same batch
    without the padding builds, and the padded lanes report the score they were given.
    """
    exact = make_table(key_type, policy, [64, 64], 16)
    padded = make_table(key_type, policy, [64, 64], 16)
    gen = torch.Generator().manual_seed(3)
    keys, tids, scores = make_batch(gen, 600, 2, key_type, policy, pad=0.2)
    real = i64(keys) != EMPTY
    new_insert(exact, take(keys, real), tids[real], take(scores, real), policy)
    score_out = torch.full((keys.numel(),), -1, dtype=torch.int64, device="cuda")
    new_insert(padded, keys, tids, scores, policy, score_out=score_out)
    assert_tables_equal(exact, padded, policy)
    assert bool((score_out[~real] == GARBAGE).all())


# ---------------------------------------------------------------------------
# refusals and the overflow tier
# ---------------------------------------------------------------------------
def pinned_pair(key_type, overflow, tables=1, bucket=16):
    """Two equal tables of one bucket per table, full and with every slot pinned, so that a
    new key is refused (or spills into the overflow bucket of a cache)."""
    old, new = make_pair(
        key_type, ScorePolicy.ASSIGN, [bucket] * tables, bucket, overflow=overflow
    )
    keys = (
        torch.arange(1, bucket + 1, dtype=torch.int64, device="cuda")
        .repeat(tables)
        .view(key_type)
    )
    tids = torch.arange(tables, dtype=torch.int64, device="cuda").repeat_interleave(
        bucket
    )
    scores = u64(torch.arange(1, bucket + 1, device="cuda").repeat(tables))
    for table in (old, new):
        new_insert(table, keys, tids, scores, ScorePolicy.ASSIGN)
        table.increment_counter(table.slots_of(keys, tids), tids)
    return old, new, keys, tids


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_refused_keys_report_their_position(key_type):
    """A bucket whose slots are all pinned refuses its newcomers: each comes back as an
    evicted entry of index -(input position + 1), the waves' lane numbers aside."""
    old, new, resident, _ = pinned_pair(key_type, overflow=False)
    newcomers = (torch.arange(101, 109, dtype=torch.int64, device="cuda")).view(
        key_type
    )
    # positions 8..15 are the newcomers; 0..7 are residents (they are assigned)
    keys = torch.cat([i64(resident[:8]), i64(newcomers)]).view(key_type)
    tids = torch.zeros(16, dtype=torch.int64, device="cuda")
    scores = u64(torch.arange(1000, 1016, device="cuda"))
    idx_old, rec_old = run_old(
        old, "insert_and_evict", keys, tids, scores, ScorePolicy.ASSIGN
    )
    insert_results = torch.empty(16, dtype=torch.uint8, device="cuda")
    idx_new, rec_new = run_new(
        new, "insert_and_evict", keys, tids, scores, ScorePolicy.ASSIGN, insert_results
    )
    assert torch.equal(idx_old, idx_new)
    assert records(rec_old, ScorePolicy.ASSIGN) == records(rec_new, ScorePolicy.ASSIGN)
    assert_tables_equal(old, new, ScorePolicy.ASSIGN)
    _, ev_keys, ev_indices, ev_scores, ev_tids = rec_new
    assert sorted(ev_indices.tolist()) == list(range(-16, -8))
    assert sorted(i64(ev_keys).tolist()) == list(range(101, 109))
    assert sorted(i64(ev_scores).tolist()) == list(
        range(1008, 1016)
    )  # their own scores
    assert (insert_results[8:] == InsertResult.BUSY.value).all()
    assert (insert_results[:8] == InsertResult.ASSIGN.value).all()
    # the evicted entries come in sorted (bucket, key) order: ascending keys here
    assert i64(ev_keys).tolist() == sorted(i64(ev_keys).tolist())


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_one_spill_per_table_equals_the_waves(key_type):
    """Two tables of one fully pinned bucket each; every call brings one new key per table,
    so each table spills once per call and the overflow bucket is comparable with the
    waves': the same slot, the same victim, the same records, for 40 calls (the overflow
    bucket of 48 slots fills up and then evicts)."""
    old, new, _, _ = pinned_pair(key_type, overflow=True, tables=2)
    gen = torch.Generator().manual_seed(21)
    fresh = (torch.randperm(10_000, generator=gen)[:80] + 1000).tolist()
    for step in range(40):
        pair = fresh[2 * step : 2 * step + 2]
        keys = torch.tensor(pair, dtype=torch.int64, device="cuda").view(key_type)
        tids = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        scores = u64([50 + step, 90 + step])
        idx_old, rec_old = run_old(old, "cache", keys, tids, scores, ScorePolicy.ASSIGN)
        idx_new, rec_new = run_new(new, "cache", keys, tids, scores, ScorePolicy.ASSIGN)
        what = "(call %d)" % step
        assert_tables_equal(old, new, ScorePolicy.ASSIGN, overflow=True, what=what)
        assert torch.equal(idx_old, idx_new), "indices differ " + what
        assert records(rec_old, ScorePolicy.ASSIGN) == records(
            rec_new, ScorePolicy.ASSIGN
        ), ("evictions differ " + what)
        assert bool((idx_new >= 16).all()), "both keys live in the overflow buckets"


def spill_sequence(key_type, calls, spills_per_table=30, seed=4):
    """Two tables of one bucket each, full and pinned, then ``calls`` calls that bring
    ``spills_per_table`` new keys per table: all of them spill into the overflow bucket of
    their table (48 slots). Returns the table and, per call, (keys, table ids, indices,
    evictions, insert results, the overflow bucket sizes right after that call): the table
    itself is only seen after the last call."""
    table = make_table(key_type, ScorePolicy.ASSIGN, [16, 16], 16, overflow=True)
    keys = (
        torch.arange(1, 17, dtype=torch.int64, device="cuda").repeat(2).view(key_type)
    )
    tids = torch.arange(2, dtype=torch.int64, device="cuda").repeat_interleave(16)
    scores = u64(torch.arange(1, 17, device="cuda").repeat(2))
    new_insert(table, keys, tids, scores, ScorePolicy.ASSIGN)
    table.increment_counter(table.slots_of(keys, tids), tids)
    gen = torch.Generator().manual_seed(seed)
    universe = torch.randperm(100_000, generator=gen) + 1000
    out = []
    for call in range(calls):
        mine = universe[call * 2 * spills_per_table : (call + 1) * 2 * spills_per_table]
        k = mine.cuda().view(torch.int64)
        t = torch.arange(2, dtype=torch.int64, device="cuda").repeat_interleave(
            spills_per_table
        )
        s = u64(torch.arange(1, 2 * spills_per_table + 1, device="cuda"))
        batch_keys = k.view(key_type)
        results = torch.empty(batch_keys.numel(), dtype=torch.uint8, device="cuda")
        idx, rec = run_new(
            table, "cache", batch_keys, t, s, ScorePolicy.ASSIGN, results
        )
        out.append(
            (batch_keys, t, idx, rec, results, table.overflow_bucket_sizes.clone())
        )
    return table, out


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_multi_spill_keeps_the_keys_of_the_call(key_type):
    """A call with many spills per table (30 into an overflow bucket of 48 slots). The waves
    could not be compared here (see the module docstring); what the pass guarantees is that a
    key the call put into the overflow bucket stays there: a later spill of the same call
    evicts residents of earlier calls, never the call's own spills."""
    table, calls = spill_sequence(key_type, 2)
    keys1, tids1, idx1, rec1, results1, sizes1 = calls[0]
    keys2, tids2, idx2, rec2, results2, sizes2 = calls[1]
    # the first call: 30 spills per table into an empty overflow bucket: all inserted
    assert rec1[0] == 0, "nothing is evicted while the overflow bucket has room"
    assert bool((results1 == InsertResult.INSERT.value).all())
    assert bool((idx1 >= 16).all())
    for t in range(2):
        mine = idx1[tids1 == t].tolist()
        assert len(set(mine)) == 30, "every spill has a slot of its own"
    # (the sizes right after this call: ``table`` has seen the second one as well)
    assert sizes1.tolist() == [30, 30]
    # the second call: 30 more per table; the residents of the first call are not pinned
    # and the scan evicts the first of them it meets before it finds an empty slot
    count2, ev_keys, ev_indices, _, ev_tids = rec2
    first = set(i64(keys1).tolist())
    second = set(i64(keys2).tolist())
    victims = i64(ev_keys).tolist()
    assert count2 == int((results2 == InsertResult.EVICT.value).sum())
    assert not (results2 == InsertResult.BUSY.value).any(), "48 slots, 30 spills: room"
    assert set(victims) <= first, "only the residents of the first call are evicted"
    assert (
        not set(victims) & second
    ), "a spill of this call is never evicted by this call"
    assert len(set(victims)) == len(victims)
    assert bool(
        (idx2 >= 16).all()
    ), "every spill of the second call is in the overflow bucket"
    for t in range(2):
        placed = idx2[tids2 == t].tolist()
        assert len(set(placed)) == 30
    # the sizes add up: every Insert result took a free slot
    inserted = [
        int(((results2 == InsertResult.INSERT.value) & (tids2 == t)).sum())
        for t in range(2)
    ]
    assert sizes2.tolist() == [30 + inserted[0], 30 + inserted[1]]
    assert table.overflow_bucket_sizes.tolist() == sizes2.tolist()  # the last call
    # the victims of the first call that were not evicted are still found
    _, founds, _ = table.lookup_with_overflow(
        keys1, tids1, ScoreArg(name="score", value=None, policy=ScorePolicy.CONST)
    )
    assert int(founds.sum()) == 60 - count2


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_multi_spill_is_the_same_on_every_run(key_type):
    """The overflow bucket of a call with many spills (and the order of its records) is the
    same on every run, also while another stream keeps the device busy."""
    reference = None
    for run in range(4):
        side = torch.cuda.Stream()
        if run % 2 == 1:
            with torch.cuda.stream(side):
                torch.cuda._sleep(2_000_000)
        table, calls = spill_sequence(key_type, 3, spills_per_table=25, seed=8)
        torch.cuda.synchronize()
        state = (
            table.overflow_table_storage_.clone(),
            table.overflow_bucket_sizes.clone(),
            [(rec[0], rec[1].clone(), rec[2].clone()) for _, _, _, rec, _, _ in calls],
        )
        if reference is None:
            reference = state
            continue
        assert torch.equal(state[0], reference[0]), "the overflow bucket differs"
        assert torch.equal(state[1], reference[1])
        for (c, k, i), (c0, k0, i0) in zip(state[2], reference[2]):
            assert (
                c == c0 and torch.equal(i64(k), i64(k0)) and torch.equal(i, i0)
            ), "the records (or their order) differ"


# ---------------------------------------------------------------------------
# a key of the call is never evicted by the call
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("policy", [ScorePolicy.ASSIGN, ScorePolicy.ACCUMULATE])
def test_a_newcomer_is_not_evicted_by_the_next_one(key_type, policy):
    """One bucket of 128 slots, full, and ten new keys that each score below every resident (the
    batches of ``test_hybrid_masked.py``: 160 keys scoring 10..169 go in, the 32 lowest are
    evicted, then ten keys scoring 1 arrive). The first newcomer takes the slot of the lowest
    resident, key 33, with score 1, which makes it the lowest of the bucket for the second one:
    if its slot were released between the two keys the second would evict it, the third the
    second, and the records would name one slot ten times. The cache and HybridStorage read the
    rows of the evicted slots before they store the rows of the new keys, so nine of the ten
    values would be written back under the wrong key. The slots of the call stay locked: each
    newcomer evicts a resident of its own, the lowest ones, in the order of the keys."""
    table = make_table(key_type, policy, [128], 128)

    def zeros(n):
        return torch.zeros(n, dtype=torch.int64, device="cuda")

    for first, last in ((1, 101), (101, 161)):
        keys = torch.arange(first, last, dtype=torch.int64, device="cuda")
        new_insert(
            table, keys.view(key_type), zeros(last - first), u64(keys + 9), policy
        )
    newcomers = torch.arange(1001, 1011, dtype=torch.int64, device="cuda")
    results = torch.empty(10, dtype=torch.uint8, device="cuda")
    indices, rec = run_new(
        table,
        "insert_and_evict",
        newcomers.view(key_type),
        zeros(10),
        u64(torch.ones(10, dtype=torch.int64, device="cuda")),
        policy,
        results,
    )
    count, ev_keys, ev_indices, _, _ = rec
    # the residents are 33..160, scoring 42..169: the ten lowest are keys 33..42
    assert count == 10
    assert i64(ev_keys).tolist() == list(range(33, 43)), "a resident each, lowest first"
    assert len(set(ev_indices.tolist())) == 10, "ten evictions, ten slots"
    assert bool((results == InsertResult.EVICT.value).all())
    # all ten are in the table, in the slots of their victims
    assert sorted(indices.tolist()) == sorted(ev_indices.tolist())
    score = ScoreArg(name="score", value=None, policy=ScorePolicy.CONST)
    _, founds, _ = table.lookup(newcomers.view(key_type), zeros(10), score)
    assert bool(founds.all())


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("op", ["insert", "insert_and_evict"])
def test_a_bucket_refuses_the_new_keys_it_has_no_slot_for(key_type, op):
    """One bucket of 16 slots and 40 new keys in one call: the 16 smallest take the slots, the
    other 24 are refused (BUSY, no index). The waves evicted the earlier newcomers of the call
    and kept the largest keys; a key of the call is never evicted by the call. The evictions
    report each refused key as an entry of its own key and score and index
    ``-(input position + 1)``."""
    table = make_table(key_type, ScorePolicy.ASSIGN, [16], 16)
    gen = torch.Generator().manual_seed(5)
    keys_cpu = torch.randperm(10_000, generator=gen)[:40] + 1
    keys = keys_cpu.cuda().view(key_type)
    tids = torch.zeros(40, dtype=torch.int64, device="cuda")
    scores_cpu = torch.randint(1, 1000, (40,), generator=gen)
    results = torch.empty(40, dtype=torch.uint8, device="cuda")
    indices, rec = run_new(
        table, op, keys, tids, u64(scores_cpu), ScorePolicy.ASSIGN, results
    )
    values = keys_cpu.tolist()
    smallest = set(sorted(values)[:16])
    live = i64(table.keys_).flatten()
    assert sorted(live[live > 0].tolist()) == sorted(smallest)
    expected = [
        InsertResult.INSERT.value if k in smallest else InsertResult.BUSY.value
        for k in values
    ]
    assert results.tolist() == expected
    assert (indices >= 0).tolist() == [k in smallest for k in values]
    if op == "insert":
        return
    count, ev_keys, ev_indices, ev_scores, _ = rec
    refused = {i: k for i, k in enumerate(values) if k not in smallest}
    assert count == 24
    assert dict(zip(ev_indices.tolist(), i64(ev_keys).tolist())) == {
        -(i + 1): k for i, k in refused.items()
    }
    assert dict(zip(i64(ev_keys).tolist(), i64(ev_scores).tolist())) == {
        k: int(scores_cpu[i]) for i, k in refused.items()
    }
    assert i64(ev_keys).tolist() == sorted(refused.values()), "in (bucket, key) order"


# ---------------------------------------------------------------------------
# determinism of the key table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("policy", [ScorePolicy.ASSIGN, TIMER])
@pytest.mark.parametrize("op", OPS)
def test_run_to_run_and_batch_order(key_type, policy, op):
    """Five fresh tables take the same six batches, once while another stream keeps the device
    busy: the tables and the evicted arrays, in order, are the same on every run. The same
    batch in a random order builds the same table, and the same evictions apart from the
    index of a refused key."""
    capacity, bucket, n = GEOMETRIES["dense"]

    def run(permute, noisy):
        table = make_table(key_type, policy, capacity, bucket, overflow=op == "cache")
        gen = torch.Generator().manual_seed(31)
        perm_gen = torch.Generator().manual_seed(
            999
        )  # the batches themselves do not change
        evictions = []
        for step in range(6):
            keys, tids, scores = make_batch(gen, n, 2, key_type, policy)
            if permute:
                perm = torch.randperm(keys.numel(), generator=perm_gen).cuda()
                keys, tids = take(keys, perm), tids[perm]
                scores = None if scores is None else take(scores, perm)
            if noisy:
                with torch.cuda.stream(torch.cuda.Stream()):
                    torch.cuda._sleep(500_000)
            indices, rec = run_new(table, op, keys, tids, scores, policy)
            evictions.append(rec)
        torch.cuda.synchronize()
        return table, evictions

    base_table, base_ev = run(False, False)
    for noisy in (False, True, True, False):
        table, ev = run(False, noisy)
        assert_tables_equal(base_table, table, policy, overflow=op == "cache")
        if op != "insert":
            for rec, rec0 in zip(ev, base_ev):
                assert rec[0] == rec0[0]
                assert torch.equal(i64(rec[1]), i64(rec0[1])), "evicted keys, in order"
                assert torch.equal(rec[2], rec0[2]), "evicted indices, in order"
                assert torch.equal(rec[4], rec0[4])
                if policy != TIMER:
                    assert torch.equal(i64(rec[3]), i64(rec0[3]))
    # another order of the batches' lanes: a different input position for every key, so
    # compare the multiset of evictions (a refused key's index encodes its position)
    table, ev = run(True, False)
    assert_tables_equal(base_table, table, policy, overflow=op == "cache")
    if op != "insert":
        for rec, rec0 in zip(ev, base_ev):
            assert records(rec, policy) == records(rec0, policy)


# ---------------------------------------------------------------------------
# GLOBAL_TIMER inside one bucket
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("seed", range(10))
def test_timer_ties_do_not_break_the_lru_order(key_type, seed):
    """One bucket of 128 slots: 128 distinct keys in one call, 128 others in a second. The device
    timer ticks more slowly than one insert takes, so two inserts of one thread can read the
    same value, and two equal timestamps would make the lowest slot win a tie in a later call
    instead of the oldest key. The sorted kernel keeps the timestamps of a run strictly
    increasing (score = max(timer, previous + 1)): the timestamps of the first call grow with
    the key, and the second call, whose keys each evict the least recently used resident, evicts
    the keys of the first in that order. (How far a timestamp can run ahead of the timer, the
    length of its run in nanoseconds, follows from that line and is below what a timer read
    after the call can resolve.)"""
    table = make_table(key_type, TIMER, [128], 128)
    gen = torch.Generator().manual_seed(seed)
    universe = torch.randperm(100_000, generator=gen) + 1
    first, second = universe[:128], universe[128:256]
    tids = torch.zeros(128, dtype=torch.int64, device="cuda")
    new_insert(table, first.cuda().view(key_type), tids, None, TIMER)
    flat = i64(table.keys_).flatten()
    live = flat > 0
    assert sorted(flat[live].tolist()) == sorted(first.tolist())
    by_key = i64(table.scores_list[0]).flatten()[live][torch.argsort(flat[live])]
    assert bool(
        (by_key[1:] > by_key[:-1]).all()
    ), "stored timestamps are strictly ordered"
    _, rec = run_new(
        table, "insert_and_evict", second.cuda().view(key_type), tids, None, TIMER
    )
    count, ev_keys = rec[0], rec[1]
    assert count == 128
    assert i64(ev_keys).tolist() == sorted(first.tolist()), "the oldest key goes first"


# ---------------------------------------------------------------------------
# lanes that take no part
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("op", ["insert", "insert_and_evict"])
def test_lanes_that_take_no_part_change_nothing(key_type, op):
    """Reserved keys of all four bit patterns (with a table id that is garbage: it is not read)
    and keys of a table without buckets are Illegal and leave the table as the batch without them
    builds it."""
    policy = ScorePolicy.ASSIGN
    exact = make_table(key_type, policy, [64, 0, 64], 16)
    mixed = make_table(key_type, policy, [64, 0, 64], 16)
    gen = torch.Generator().manual_seed(12)
    keys, tids, scores = make_batch(gen, 300, 3, key_type, policy, pad=0.0)
    on_empty_table = tids == 1
    real = ~on_empty_table
    run_new(exact, op, take(keys, real), tids[real], take(scores, real), policy)
    # add the four reserved patterns, with table ids that are -1, valid and garbage
    reserved = torch.tensor([-1, -2, -3, -4] * 3, dtype=torch.int64, device="cuda")
    garbage_tids = torch.tensor([-1, 0, 2, 12345] * 3, dtype=torch.int64, device="cuda")
    n = keys.numel()
    all_keys = torch.cat([i64(keys), reserved]).view(key_type)
    all_tids = torch.cat([tids, garbage_tids])
    all_scores = u64(
        torch.cat(
            [i64(scores), torch.full((12,), GARBAGE, dtype=torch.int64, device="cuda")]
        )
    )
    insert_results = torch.empty(all_keys.numel(), dtype=torch.uint8, device="cuda")
    indices, _ = run_new(
        mixed, op, all_keys, all_tids, all_scores, policy, insert_results
    )
    assert_tables_equal(exact, mixed, policy)
    illegal = InsertResult.ILLEGAL.value
    assert bool((insert_results[n:] == illegal).all())
    assert bool((insert_results[:n][on_empty_table] == illegal).all())
    assert bool((indices[n:] == -1).all())
    assert bool((indices[:n][on_empty_table] == -1).all())
