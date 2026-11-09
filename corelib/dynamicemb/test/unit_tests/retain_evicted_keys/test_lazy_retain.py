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

"""Retained evicted keys are appended without reading how many there are.

``_append_evicted`` with a count on the device keeps the arrays whole (the entries past
the count made inert), copies the count to a pinned ring and folds the entry into the
retain chunks once the count has landed; a pop folds whatever is still pending.
``_append_evicted_masked`` takes arrays whose members are scattered (the reserved
all-ones key and table id ``-1`` mark every other entry) and derives the count itself.

What is checked here is that a pop returns exactly the keys the old, synchronous
bookkeeping would have: the first ``n`` entries of the arrays of every insert, read on
the host, de-duplicated per table.
"""

import random

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy, EvictedItemMode
from dynamicemb import sync_free as sf
from dynamicemb.dynamicemb_config import DynamicEmbEvictStrategy, DynamicEmbTableOptions
from dynamicemb.key_value_table import (
    DynamicEmbStorage,
    _append_evicted,
    _append_evicted_masked,
    _pop_state_evicted_keys,
    _resolve_pending_evicted,
    _wait_for_stream,
    get_insert_score_arg,
)
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer

# The compound (TIMESTAMP, LFU) strategy of the other retain tests: a full bucket
# evicts, and the evictions are collected.
LRU_LFU_TS_FIRST = (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)
DIM = 8
BUCKET = 128  # one bucket per table: a table that holds 128 keys evicts on the next


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


def _storage(num_tables=2, retain=True, key_type=torch.int64):
    """A last-tier storage of ``num_tables`` tables of one bucket each."""
    device_id = torch.cuda.current_device()
    options = [
        DynamicEmbTableOptions(
            index_type=key_type,
            embedding_dtype=torch.float32,
            device_id=device_id,
            dim=DIM,
            max_capacity=BUCKET,
            init_capacity=BUCKET,
            bucket_capacity=BUCKET,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=LRU_LFU_TS_FIRST,
            evict_strategy=DynamicEmbEvictStrategy.LFU,
            evicted_item_mode=(
                EvictedItemMode.RETAIN_KEY if retain else EvictedItemMode.DISCARD
            ),
        )
        for _ in range(num_tables)
    ]
    return DynamicEmbStorage(options, SGDDynamicEmbeddingOptimizer(OptimizerArgs()))


def _upload(values, device):
    """An int64 tensor on ``device`` copied on the current stream without waiting:
    a copy from pageable memory would wait for everything queued on the stream, and
    the tests below keep the stream busy on purpose."""
    host = torch.tensor(values, dtype=torch.int64).pin_memory()
    return host.to(device, non_blocking=True)


def _as_keys(values, device, key_type=torch.int64):
    """Keys of ``key_type`` from int64 values; ``-1`` is the reserved all-ones key."""
    t = _upload(values, device)
    return t if key_type == torch.int64 else t.view(key_type)


def _masked_arrays(members, n, device, rng=None, key_type=torch.int64):
    """N long arrays with the ``members`` ((table_id, key) pairs) scattered over
    them and the sentinels (reserved key, table id -1) everywhere else."""
    rng = rng or random.Random(0)
    keys, tids = [-1] * n, [-1] * n
    for pos, (tid, key) in zip(rng.sample(range(n), len(members)), members):
        keys[pos], tids[pos] = key, tid
    return _as_keys(keys, device, key_type), _upload(tids, device)


def _chunk_keys(state, table_id):
    """Keys of the folded chunks of ``table_id`` as sorted int64 values (no pop)."""
    out = []
    for keys, tids in zip(state.evicted_key_chunks, state.evicted_tid_chunks):
        out += keys.view(torch.int64)[tids == table_id].tolist()
    return sorted(out)


@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64])
def test_masked_append_retains_the_scattered_members(device, key_type):
    """The members of a masked array are anywhere in it: the keys that are retained
    are the members, not the first ``count`` entries (a masked array appended with a
    count and cut ``[:n]`` would retain sentinels and miss members)."""
    state = _storage(key_type=key_type)._state
    members = [(0, 7), (1, 9), (0, 11), (1, 12), (0, 13)]
    keys, tids = _masked_arrays(members, 16, device, key_type=key_type)
    # make sure the members are NOT a prefix
    assert tids[: len(members)].ge(0).sum().item() < len(members)

    _append_evicted_masked(state, keys, tids)
    _resolve_pending_evicted(state, block=True)

    assert not state.evicted_pending
    assert _chunk_keys(state, 0) == [7, 11, 13]
    assert _chunk_keys(state, 1) == [9, 12]
    assert all(c.dtype == key_type for c in state.evicted_key_chunks)
    # a folded chunk holds members only, in the order they had in the arrays
    assert all(bool((t >= 0).all()) for t in state.evicted_tid_chunks)
    assert state.evicted_key_chunks[0].numel() == len(members)


def test_masked_append_of_nothing_retains_nothing(device):
    state = _storage()._state
    keys, tids = _masked_arrays([], 8, device)
    _append_evicted_masked(state, keys, tids)
    _resolve_pending_evicted(state, block=True)
    assert not state.evicted_pending
    assert state.evicted_key_chunks == [] and state.evicted_tid_chunks == []
    # arrays of no entries are not even kept
    _append_evicted_masked(state, keys[:0], tids[:0])
    assert not state.evicted_pending


def test_a_pop_drops_nothing_and_leaves_no_sentinel(device):
    state = _storage()._state
    _append_evicted_masked(state, *_masked_arrays([(0, 5), (1, 6)], 8, device))
    _append_evicted_masked(state, *_masked_arrays([(1, 7), (1, 6)], 8, device))
    assert _pop_state_evicted_keys(state, 0).tolist() == [5]
    assert not state.evicted_pending
    # table 1's keys survive table 0's pop, each once; no inert entry is left over
    assert all(bool((t >= 0).all()) for t in state.evicted_tid_chunks)
    assert _pop_state_evicted_keys(state, 1).tolist() == [6, 7]
    assert state.evicted_key_chunks == [] and state.evicted_tid_chunks == []
    assert _pop_state_evicted_keys(state, 0).numel() == 0


def test_a_chunk_with_sentinel_rows_loses_them_on_pop(device):
    """A sentinel row (table id -1) belongs to no table: the remainder of a pop must
    not keep it for ever."""
    state = _storage()._state
    state.evicted_key_chunks.append(_as_keys([1, -1, 2, 3], device))
    state.evicted_tid_chunks.append(_upload([0, -1, 1, 1], device))
    assert _pop_state_evicted_keys(state, 0).tolist() == [1]
    assert [k.tolist() for k in state.evicted_key_chunks] == [[2, 3]]
    assert _pop_state_evicted_keys(state, 1).tolist() == [2, 3]
    assert state.evicted_key_chunks == [] and state.evicted_tid_chunks == []


@pytest.mark.parametrize("count_kind", ["int", "cpu_tensor", "cuda_tensor"])
def test_a_prefix_count_keeps_the_first_entries(device, monkeypatch, count_kind):
    """Arrays the way a kernel leaves them: the first ``count`` entries are real, the
    rest is whatever was in the buffer (here: valid-looking keys and table ids).
    With the count on the host it is a slice; with the count on the device the tail is
    made inert by ``sf.inert_collected``, which the extension may already have done:
    the fallback is forced so that the garbage below really is garbage."""
    monkeypatch.setattr(sf, "EVICT_TAIL_INERT", False)
    state = _storage()._state
    keys = _as_keys([1, 2, 3, 900, 901], device)
    tids = _upload([0, 0, 1, 0, 1], device)
    count = {
        "int": 3,
        "cpu_tensor": torch.tensor([3]),
        "cuda_tensor": torch.tensor([3], device=device),
    }[count_kind]

    _append_evicted(state, keys, tids, count)

    assert _pop_state_evicted_keys(state, 0).tolist() == [1, 2]
    assert _pop_state_evicted_keys(state, 1).tolist() == [3]


def test_a_count_of_zero_retains_nothing(device, monkeypatch):
    monkeypatch.setattr(sf, "EVICT_TAIL_INERT", False)
    state = _storage()._state
    keys = _as_keys([1, 2, 3], device)
    tids = torch.zeros(3, dtype=torch.int64, device=device)
    _append_evicted(state, keys, tids, torch.zeros(1, dtype=torch.int64, device=device))
    assert _pop_state_evicted_keys(state, 0).numel() == 0


def test_pending_entries_stay_few_when_the_counts_land(device):
    """One append per step and the count of a step lands before the next: an entry is
    folded by the append after it, so at most the last one or two are pending."""
    state = _storage()._state
    rng = random.Random(1)
    expected = set()
    for step in range(30):
        members = [(0, 100 * step + i) for i in range(rng.randrange(0, 4))]
        _append_evicted_masked(state, *_masked_arrays(members, 8, device, rng))
        expected |= {key for _, key in members}
        torch.cuda.synchronize()
        assert len(state.evicted_pending) <= 2
        # nothing waits for the device on this path: a landed count is just a read
        # of the pinned ring, so the ring is never exhausted either
        assert state.retain_ring.free_slots() >= 6
    assert set(_pop_state_evicted_keys(state, 0).tolist()) == expected


def test_a_full_ring_degrades_to_a_blocking_read(device):
    """With every ring slot in flight the append neither blocks nor loses the entry;
    the pop reads what it needs."""
    state = _storage()._state
    state.retain_ring = sf.DeferredRead((1,), torch.int64, slots=1)
    expected = set()
    batches = []
    for step in range(4):
        members = [(0, 10 * step + 1), (1, 10 * step + 2), (0, 10 * step + 3)]
        batches.append(_masked_arrays(members, 8, device))
        expected |= {key for _, key in members}
    torch.cuda._sleep(5 * 10**8)  # no count can land while the entries are appended
    for keys, tids in batches:
        _append_evicted_masked(state, keys, tids)
    assert len(state.evicted_pending) == 4
    assert [e[3] is None for e in state.evicted_pending] == [False, True, True, True]

    popped = set(_pop_state_evicted_keys(state, 0).tolist()) | set(
        _pop_state_evicted_keys(state, 1).tolist()
    )
    assert popped == expected
    assert not state.evicted_pending
    assert state.retain_ring.free_slots() == 1, "every ticket is released"


def test_entries_without_a_ticket_are_retried_when_a_slot_frees(device):
    state = _storage()._state
    state.retain_ring = sf.DeferredRead((1,), torch.int64, slots=1)
    first = _masked_arrays([(0, 1)], 4, device)
    second = _masked_arrays([(0, 2)], 4, device)
    torch.cuda.synchronize()
    torch.cuda._sleep(5 * 10**8)
    _append_evicted_masked(state, *first)
    _append_evicted_masked(state, *second)
    assert [e[3] is None for e in state.evicted_pending] == [False, True]
    torch.cuda.synchronize()
    # the first count has landed: folding it frees the slot, and the entry without
    # a ticket is given it by the same call (on the stream that made the entry)
    _resolve_pending_evicted(state, block=False)
    assert _chunk_keys(state, 0) in ([1], [1, 2])
    assert all(e[3] is not None for e in state.evicted_pending)
    torch.cuda.synchronize()
    _resolve_pending_evicted(state, block=False)
    assert not state.evicted_pending
    assert _chunk_keys(state, 0) == [1, 2]


@pytest.mark.parametrize("slots", [8, 1])
def test_a_pop_on_another_stream_than_the_appends(device, slots):
    """The arrays and their counts are made on a side stream that is still busy when
    the default stream pops. With a free ring the pop waits for the event of the
    count; with a full one it reads the count after the producing stream."""
    state = _storage()._state
    state.retain_ring = sf.DeferredRead((1,), torch.int64, slots=slots)
    side = torch.cuda.Stream()
    expected = set()
    reads = sf.READ_NOW_CALLS
    with torch.cuda.stream(side):
        batches = []
        for step in range(3):
            members = [(0, 10 * step + 1), (0, 10 * step + 2)]
            batches.append(_masked_arrays(members, 8, device))
            expected |= {key for _, key in members}
        torch.cuda._sleep(5 * 10**8)
        for keys, tids in batches:
            _append_evicted_masked(state, keys, tids)
    # the default stream does not wait for `side`: the pop has to order itself
    popped = _pop_state_evicted_keys(state, 0)
    assert set(popped.tolist()) == expected
    assert popped.numel() == len(expected)
    if slots == 1:
        assert sf.READ_NOW_CALLS > reads, "two of the entries had no ticket"
    # the chunks were made on the default stream out of arrays of `side`
    torch.cuda.synchronize()


@pytest.mark.parametrize("producer_is_default", [True, False])
def test_waiting_for_a_stream_orders_the_current_stream_after_it(
    device, producer_is_default
):
    """The handle of the default stream is 0, which ``torch.cuda.ExternalStream`` takes
    for "no stream given" and answers with an idle stream of the pool: a wait built
    from that handle alone would order nothing."""
    default, side = torch.cuda.default_stream(), torch.cuda.Stream()
    producer, consumer = (default, side) if producer_is_default else (side, default)
    flag = torch.zeros(1, dtype=torch.int64, device=device)
    torch.cuda.synchronize()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(5 * 10**8)
        flag.fill_(7)  # behind the sleep
    with torch.cuda.stream(consumer):
        _wait_for_stream(producer.cuda_stream)
        seen = flag.clone()
    torch.cuda.synchronize()
    assert seen.item() == 7


def test_a_pop_on_a_side_stream_reads_the_counts_the_default_stream_made(device):
    """The entries without a ticket are read by the pop itself, after the stream that
    made them: here the default stream (its handle is 0) and the pop is on a side
    stream. The counts of the last two entries are made behind a long kernel that
    the wait for the first entry's ticket does not cover."""
    state = _storage()._state
    state.retain_ring = sf.DeferredRead((1,), torch.int64, slots=1)
    entries = [
        _masked_arrays([(0, 1), (0, 2)], 8, device),
        _masked_arrays([(0, 11), (0, 12), (0, 13)], 8, device),
        _masked_arrays([(0, 21)], 8, device),
    ]
    torch.cuda._sleep(5 * 10**8)  # no count can land while the entries are appended
    _append_evicted_masked(state, *entries[0])  # takes the only slot
    torch.cuda._sleep(5 * 10**8)  # the counts of the others are made behind this
    for keys, tids in entries[1:]:
        _append_evicted_masked(state, keys, tids)
    assert [e[3] is None for e in state.evicted_pending] == [False, True, True]

    with torch.cuda.stream(torch.cuda.Stream()):
        popped = _pop_state_evicted_keys(state, 0)
    torch.cuda.synchronize()
    assert popped.tolist() == [1, 2, 11, 12, 13, 21]
    assert not state.evicted_pending


def test_a_ready_entry_is_folded_by_another_stream_without_waiting(device):
    """A non-blocking fold on a stream other than the producer's: the count has
    landed, so the entry is folded there; the arrays stay valid for it."""
    state = _storage()._state
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        _append_evicted_masked(state, *_masked_arrays([(0, 21), (1, 22)], 8, device))
    side.synchronize()
    _resolve_pending_evicted(state, block=False)
    assert not state.evicted_pending
    torch.cuda.synchronize()
    assert _chunk_keys(state, 0) == [21] and _chunk_keys(state, 1) == [22]


@pytest.mark.parametrize("fallback", [False, True])
def test_retained_keys_of_real_inserts_equal_the_host_slices(
    device, monkeypatch, fallback
):
    """Inserts into a full last tier, with and without evictions and with their counts
    left on the device until the end: what a pop returns is what the old code would
    have kept, the first ``n`` entries of each insert's arrays, read on the host."""
    if fallback:
        # the Python tail fill instead of the extension's, on the same arrays
        monkeypatch.setattr(sf, "EVICT_TAIL_INERT", False)
    storage = _storage(num_tables=2)
    state = storage._state
    rng = random.Random(7)
    kept = []  # (evicted_keys, evicted_table_ids, num_evicted) of every insert
    present = {0: [], 1: []}
    next_key = {0: 1, 1: 1}

    def insert(per_table_new, per_table_again):
        keys, tids = [], []
        for t in (0, 1):
            fresh = list(range(next_key[t], next_key[t] + per_table_new[t]))
            next_key[t] += per_table_new[t]
            again = rng.sample(present[t], min(per_table_again[t], len(present[t])))
            keys += fresh + again
            tids += [t] * (len(fresh) + len(again))
            present[t] += fresh
        keys_t = _as_keys(keys, device)
        tids_t = _upload(tids, device)
        n = keys_t.numel()
        score = get_insert_score_arg(
            state, n, device, torch.ones(n, dtype=torch.uint64, device=device)
        )
        _, num_evicted, ev_keys, ev_tids = state.key_index_map.insert(
            keys_t, tids_t, score, collect_evicted=True
        )
        kept.append((ev_keys, ev_tids, num_evicted))
        _append_evicted(state, ev_keys, ev_tids, num_evicted)

    insert((BUCKET, BUCKET), (0, 0))  # fills both tables: nothing is evicted
    for _ in range(10):
        insert(
            (rng.choice([1, 5, 20]), rng.choice([1, 5, 20])),
            (rng.choice([0, 0, 10]), rng.choice([0, 0, 10])),
        )
    assert len(state.evicted_pending) >= 1, "the counts were not read back"

    expected = {0: set(), 1: set()}
    for ev_keys, ev_tids, num_evicted in kept:  # the old, synchronous reads
        n = int(num_evicted)
        for key, tid in zip(ev_keys[:n].tolist(), ev_tids[:n].tolist()):
            expected[tid].add(key)
    assert expected[0] or expected[1], "the test must evict something"

    for t in (0, 1):
        popped = _pop_state_evicted_keys(state, t)
        assert popped.tolist() == sorted(expected[t])
    assert not state.evicted_pending
    assert state.retain_ring.free_slots() == 8
