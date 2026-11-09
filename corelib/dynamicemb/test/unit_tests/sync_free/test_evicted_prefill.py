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

"""The evicted arrays of an insert are pre-filled with -1.

``table_insert_and_evict`` and ``table_insert_collect_evicted`` return their
``evicted_*`` arrays sized at the batch upper bound (one victim per input key);
the kernels write only the first ``num_evicted`` entries. The rest is -1 in
every array -- all ones for the keys, as an int64 key and as a uint64 one -- the
marker every consumer reads as "no eviction here" (EmptyKey is skipped by the
table kernels, index -1 by the load / store kernels, table id -1 by the
padded-buffer kernels), so that code taking all the entries needs no count.

The tests call the C++ entry points through the table (``insert(...,
collect_evicted=True)`` for the collecting insert) or directly for
``table_insert_and_evict``, whose Python wrapper slices the arrays to the count
and so hides the tail.
"""

import pytest
import torch

dynamicemb_extensions = pytest.importorskip("dynamicemb_extensions")

from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy, table_insert_and_evict

KEY_DTYPES = [torch.int64, torch.uint64]
POLICIES = ["assign", "lru_lfu"]
CAPACITY = 128  # one bucket per table: a full table evicts
OLD_PER_TABLE = 100
NEW_PER_TABLE = 58
EVICTED_PER_TABLE = OLD_PER_TABLE + NEW_PER_TABLE - CAPACITY  # 30
NUM_TABLES = 2


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def test_extension_announces_the_marked_evicted_arrays():
    """Callers that fall back without the marked tails ask for this attribute
    (``getattr(dynamicemb_extensions, NAME, False)``)."""
    assert hasattr(dynamicemb_extensions, "evicted_tail_inert")
    assert dynamicemb_extensions.evicted_tail_inert is True


def _score_name(policy):
    return "score1" if policy == "assign" else "frequency"


def _make_table(policy, key_dtype, capacities=None, enable_overflow=False):
    if policy == "assign":
        spec = ScoreSpec(name="score1", policy=ScorePolicy.ASSIGN)
    else:
        spec = ScoreSpec(
            name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True
        )
    return get_scored_table(
        capacity=capacities or [CAPACITY] * NUM_TABLES,
        bucket_capacity=CAPACITY,
        key_type=key_dtype,
        score_specs=[spec],
        enable_overflow=enable_overflow,
    )


def _score_arg(policy, scores):
    """``scores``: the score each key is given -- under ``assign``, the score
    itself, under ``lru_lfu`` every key is accessed once (the access is what
    ranks it, newer is higher)."""
    if policy == "assign":
        return ScoreArg(name="score1", value=scores, policy=ScorePolicy.ASSIGN)
    return ScoreArg(
        name="frequency", value=torch.ones_like(scores), policy=ScorePolicy.LRU_LFU
    )


def _insert_results(table, n):
    return torch.empty(n, dtype=table.result_type, device=_device()).fill_(
        InsertResult.INIT.value
    )


def _two_tables_of_keys(first_table0, first_table1, count):
    keys = torch.cat(
        [
            torch.arange(first_table0, first_table0 + count),
            torch.arange(first_table1, first_table1 + count),
        ]
    ).to(_device())
    table_ids = torch.cat(
        [
            torch.zeros(count, dtype=torch.int64),
            torch.ones(count, dtype=torch.int64),
        ]
    ).to(_device())
    return keys, table_ids


def _fill_table(table, policy, key_dtype):
    """100 keys per table, score == key. Returns what is stored:
    ``(keys, table_ids, slots)``."""
    keys, table_ids = _two_tables_of_keys(1, 1001, OLD_PER_TABLE)
    keys = keys.to(key_dtype)
    table.insert(
        keys,
        table_ids,
        _score_arg(policy, keys.to(torch.uint64)),
        _insert_results(table, keys.numel()),
    )
    torch.cuda.synchronize()
    _, founds, slots = table.lookup(
        keys,
        table_ids,
        ScoreArg(name=_score_name(policy), policy=ScorePolicy.CONST),
    )
    assert bool(founds.all())
    return keys, table_ids, slots


def _new_batch(policy, key_dtype):
    """58 new keys per table, scored far above everything stored, so under
    ``assign`` every one of them wins its slot and the lowest old scores go."""
    keys, table_ids = _two_tables_of_keys(10_000, 20_000, NEW_PER_TABLE)
    keys = keys.to(key_dtype)
    scores = torch.full(
        (keys.numel(),), 10_000_000, dtype=torch.uint64, device=_device()
    )
    return keys, table_ids, _score_arg(policy, scores)


def _insert_and_evict(table, keys, table_ids, score, insert_results):
    """``table_insert_and_evict`` with the arrays unsliced, as the table's own
    ``insert_and_evict`` would call it."""
    score_value, policy = table._parse_score(score)
    extra = {}
    if table.enable_overflow_:
        extra = dict(
            ovf_storage=table.overflow_table_storage_,
            ovf_bucket_capacity=table.overflow_bucket_capacity_,
            ovf_bucket_sizes=table.overflow_bucket_sizes,
            ovf_counter=table._ovf_counter,
            ovf_output_offsets=table.overflow_output_offsets_,
        )
    return table_insert_and_evict(
        table.table_storage_,
        table.table_bucket_offsets_,
        table.bucket_capacity_,
        table.bucket_sizes,
        keys,
        table_ids,
        score_value,
        policy,
        table._ref_counter,
        insert_results=insert_results,
        num_scores=table.num_scores_,
        score_fn_key=table.score_fn_key_,
        **extra,
    )


def _assert_marked_from(count, *arrays, num_total, key_dtype=None):
    """Every array has ``num_total`` entries and holds -1 from entry ``count``
    on; the first of them is the key array when ``key_dtype`` is given."""
    for position, array in enumerate(arrays):
        assert array.shape == (num_total,)
        if position == 0 and key_dtype is not None:
            assert array.dtype == key_dtype
            array = array.view(torch.int64)  # all ones, whatever the key dtype
        else:
            assert array.dtype == torch.int64
        assert bool((array[count:] == -1).all()), f"array {position}: tail not -1"


@pytest.mark.parametrize("enable_overflow", [False, True], ids=["plain", "overflow"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
def test_insert_and_evict_marks_the_tail(key_dtype, policy, enable_overflow):
    table = _make_table(policy, key_dtype, enable_overflow=enable_overflow)
    old_keys, old_ids, old_slots = _fill_table(table, policy, key_dtype)
    keys, table_ids, score = _new_batch(policy, key_dtype)
    results = _insert_results(table, keys.numel())

    indices, num_evicted, ek, ei, es, et = _insert_and_evict(
        table, keys, table_ids, score, results
    )
    torch.cuda.synchronize()

    count = int(num_evicted)
    num_total = keys.numel()
    assert 0 < count < num_total
    # What the kernels report: an eviction, or a key the table refused (Busy).
    reported = (results == InsertResult.EVICT.value) | (
        results == InsertResult.BUSY.value
    )
    assert count == int(reported.sum())
    _assert_marked_from(count, ek, ei, es, et, num_total=num_total, key_dtype=key_dtype)

    # The head is what it was: a victim is a stored key with the slot it sat in
    # and its table; a refused key (negative index) is an input one.
    slot_of = dict(
        zip(
            zip(old_keys.view(torch.int64).tolist(), old_ids.tolist()),
            old_slots.tolist(),
        )
    )
    new_set = set(zip(keys.view(torch.int64).tolist(), table_ids.tolist()))
    head = list(
        zip(
            ek[:count].view(torch.int64).tolist(),
            et[:count].tolist(),
            ei[:count].tolist(),
            es[:count].tolist(),
        )
    )
    for key, table_id, slot, score_word in head:
        if slot >= 0:
            assert (key, table_id) in slot_of, "a victim must be a stored key"
            assert slot_of[(key, table_id)] == slot
            if policy == "assign":
                assert score_word == key  # the score a stored key was given
        else:
            assert (key, table_id) in new_set
    if policy == "assign":
        # Exactly the lowest scores of each table, and every new key got a slot.
        assert count == NUM_TABLES * EVICTED_PER_TABLE
        assert {(k, t) for k, t, _, _ in head} == {
            (key, 0) for key in range(1, 1 + EVICTED_PER_TABLE)
        } | {(key, 1) for key in range(1001, 1001 + EVICTED_PER_TABLE)}
        assert bool((indices >= 0).all())


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
def test_insert_and_evict_without_eviction_is_all_marked(key_dtype, policy):
    table = _make_table(policy, key_dtype, capacities=[4096] * NUM_TABLES)
    keys, table_ids, score = _new_batch(policy, key_dtype)
    results = _insert_results(table, keys.numel())
    _, num_evicted, ek, ei, es, et = _insert_and_evict(
        table, keys, table_ids, score, results
    )
    torch.cuda.synchronize()
    assert int(num_evicted) == 0
    _assert_marked_from(0, ek, ei, es, et, num_total=keys.numel(), key_dtype=key_dtype)


@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
def test_evicted_arrays_are_independent_tensors(key_dtype):
    """The four arrays are views of one block: writing one must leave the
    others alone."""
    table = _make_table("assign", key_dtype)
    _fill_table(table, "assign", key_dtype)
    keys, table_ids, score = _new_batch("assign", key_dtype)
    _, _, ek, ei, es, et = _insert_and_evict(
        table, keys, table_ids, score, _insert_results(table, keys.numel())
    )
    torch.cuda.synchronize()
    before = [array.clone() for array in (ek, ei, es, et)]
    for position, array in enumerate((ek, ei, es, et)):
        array.view(torch.int64).fill_(position + 1000)
        for other, snapshot in enumerate(before):
            if other == position:
                continue
            current = (ek, ei, es, et)[other]
            assert torch.equal(current.view(torch.int64), snapshot.view(torch.int64))
        array.view(torch.int64).copy_(before[position].view(torch.int64))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
def test_insert_collect_evicted_marks_the_tail(key_dtype, policy):
    table = _make_table(policy, key_dtype)
    old_keys, old_ids, _ = _fill_table(table, policy, key_dtype)
    keys, table_ids, score = _new_batch(policy, key_dtype)
    results = _insert_results(table, keys.numel())

    _, num_evicted, ek, et = table.insert(
        keys, table_ids, score, results, collect_evicted=True
    )
    torch.cuda.synchronize()

    count = int(num_evicted)
    num_total = keys.numel()
    assert 0 < count < num_total
    # The collecting insert reports an eviction of a stored key only.
    assert count == int((results == InsertResult.EVICT.value).sum())
    _assert_marked_from(count, ek, et, num_total=num_total, key_dtype=key_dtype)

    stored = set(zip(old_keys.view(torch.int64).tolist(), old_ids.tolist()))
    head = list(zip(ek[:count].view(torch.int64).tolist(), et[:count].tolist()))
    assert set(head) <= stored and len(set(head)) == count
    if policy == "assign":
        assert set(head) == {(key, 0) for key in range(1, 1 + EVICTED_PER_TABLE)} | {
            (key, 1) for key in range(1001, 1001 + EVICTED_PER_TABLE)
        }


@pytest.mark.parametrize("key_dtype", KEY_DTYPES)
def test_insert_collect_evicted_without_eviction_is_all_marked(key_dtype):
    table = _make_table("assign", key_dtype, capacities=[4096] * NUM_TABLES)
    keys, table_ids, score = _new_batch("assign", key_dtype)
    _, num_evicted, ek, et = table.insert(
        keys,
        table_ids,
        score,
        _insert_results(table, keys.numel()),
        collect_evicted=True,
    )
    torch.cuda.synchronize()
    assert int(num_evicted) == 0
    _assert_marked_from(0, ek, et, num_total=keys.numel(), key_dtype=key_dtype)
