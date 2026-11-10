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

"""A pop of retained keys is one pass over the retained chunks of every table, and waits
twice: one read of how many keys each table gets, one copy of exactly those keys.

What is checked:

* ``_drain_chunks``, the pass, against a plain-Python reference on CPU tensors and on the
  GPU: int64 and uint64 keys (a uint64 key above ``2**63`` sorts after the smaller
  ones), ``[n, 2]`` slot-index chunks, sentinel rows (table id ``-1``), tables that are
  not asked for staying behind in the order they came;
* the storage's ``pop_many`` against the per-table ``pop_evicted_keys`` it replaces, with
  pending evictions of the lazy append in the mix (made on a side stream too);
* the waits of a pop with 2 and with 8 tables (two each, not per table), and the bytes it
  copies (the keys it returns, not the keys it leaves).

Everything the measured calls need is built before them: a CUDA tensor made from a host
list is a blocking copy.
"""

import importlib
import random

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)

EMPTY = -1
BUCKET = 128
DIM = 8
SLEEP_CYCLES = 5 * 10**8


def _need(name):
    """The module, or a skip: a tree without the extension has nothing to measure."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip("%s is not importable here: %s" % (name, e))


def _int64_view(t):
    t = t.detach().cpu()
    return t.view(torch.int64) if t.dtype == torch.uint64 else t


def _keys_of(values, dtype, device):
    """A key tensor of ``dtype`` holding the given int64 bit patterns."""
    return torch.tensor(values, dtype=torch.int64, device=device).view(dtype)


def _unsigned(value, dtype):
    """The number a key's int64 bit pattern stands for, in ``dtype``'s order."""
    return value & ((1 << 64) - 1) if dtype == torch.uint64 else value


def _reference(rows, tids, wanted, dtype):
    """What a pop is specified to return, in plain Python: per wanted table the distinct
    keys in ascending order of the key type, and the rows of the tables that were not
    asked for, in the order they came, sentinels dropped."""
    taken = {
        t: sorted({_unsigned(k, dtype) for k, tt in zip(rows, tids) if tt == t})
        for t in wanted
    }
    rest = [(k, tt) for k, tt in zip(rows, tids) if tt >= 0 and tt not in wanted]
    return taken, rest


def _as_numbers(t, dtype):
    return [_unsigned(v, dtype) for v in _int64_view(t).tolist()]


DEVICES = ["cpu", "cuda"]


@pytest.fixture(params=DEVICES)
def where(request, gpu):
    return (
        torch.device(request.param, torch.cuda.current_device())
        if request.param == "cuda"
        else torch.device("cpu")
    )


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("wanted", [[0], [1, 3], [0, 1, 2, 3], []])
def test_the_pass_returns_the_distinct_keys_of_the_wanted_tables_in_order(
    where, dtype, seed, wanted
):
    kvt = _need("dynamicemb.key_value_table")
    rng = random.Random(seed)
    num_tables = 4
    # keys from a small pool with duplicates, and some above 2**63 as unsigned numbers
    pool = [rng.randrange(-(1 << 63), 1 << 63) for _ in range(12)] + [5, 6, 7, -2, -3]
    chunks_k, chunks_t = [], []
    rows, tids = [], []
    for _ in range(rng.randrange(1, 5)):
        n = rng.randrange(1, 30)
        k = [rng.choice(pool) for _ in range(n)]
        t = [rng.choice([-1, 0, 1, 2, 3]) for _ in range(n)]
        # a sentinel row has the reserved key as well, as a masked append leaves it
        k = [EMPTY if tt == -1 else kk for kk, tt in zip(k, t)]
        chunks_k.append(_keys_of(k, dtype, where))
        chunks_t.append(torch.tensor(t, dtype=torch.int64, device=where))
        rows += k
        tids += t

    taken, counts, rest_keys, rest_tids = kvt._drain_chunks(
        chunks_k, chunks_t, wanted, num_tables
    )

    expected, expected_rest = _reference(rows, tids, set(wanted), dtype)
    assert taken.dtype == dtype and taken.dim() == 1
    assert counts == [len(expected.get(t, [])) for t in range(num_tables)]
    values = _as_numbers(taken, dtype)
    start = 0
    for t in range(num_tables):
        assert values[start : start + counts[t]] == expected.get(t, []), t
        start += counts[t]
    assert start == taken.numel()
    assert rest_keys.dtype == dtype
    got_rest = list(zip(_int64_view(rest_keys).tolist(), rest_tids.cpu().tolist()))
    assert (
        got_rest == expected_rest
    ), "the other tables' rows stay, in the order they came"


@pytest.mark.parametrize("wanted", [[0], [2], [0, 1, 2]])
def test_the_pass_over_slot_index_chunks_matches_unique_by_row(where, wanted):
    """The (slot, row) pairs of a slot index are ``[n, 2]`` chunks: the distinct pairs per
    table in lexicographic order, as ``torch.unique(dim=0)`` gives them."""
    kvt = _need("dynamicemb.key_value_table")
    rng = random.Random(4)
    n = 60
    slots = [rng.randrange(0, 12) for _ in range(n)]
    rows_of_slot = {s: 100 + s for s in range(12)}  # a slot owns one row
    pairs = torch.tensor([[s, rows_of_slot[s]] for s in slots], dtype=torch.int64)
    tids = torch.tensor(
        [rng.choice([-1, 0, 1, 2]) for _ in range(n)], dtype=torch.int64
    )
    taken, counts, rest, rest_tids = kvt._drain_chunks(
        [pairs[:25].to(where), pairs[25:].to(where)],
        [tids[:25].to(where), tids[25:].to(where)],
        wanted,
        3,
    )
    start = 0
    for t in range(3):
        mine = taken[start : start + counts[t]].cpu()
        start += counts[t]
        if t in wanted:
            reference = (
                torch.unique(pairs[tids == t], dim=0)
                if bool((tids == t).any())
                else pairs[:0]
            )
            assert torch.equal(mine, reference), t
        else:
            assert mine.numel() == 0
    stay = (tids >= 0) & torch.tensor([int(t) not in wanted for t in tids.tolist()])
    assert torch.equal(rest.cpu(), pairs[stay]) and torch.equal(
        rest_tids.cpu(), tids[stay]
    )


def test_the_pass_of_nothing(where):
    kvt = _need("dynamicemb.key_value_table")
    keys = _keys_of([EMPTY, EMPTY], torch.int64, where)
    tids = torch.tensor([-1, -1], dtype=torch.int64, device=where)
    taken, counts, rest_keys, rest_tids = kvt._drain_chunks([keys], [tids], [0, 1], 2)
    assert taken.numel() == 0 and counts == [0, 0]
    assert rest_keys.numel() == 0 and rest_tids.numel() == 0


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
def test_a_host_union_is_sorted_and_distinct_in_the_key_type_order(dtype):
    kvt = _need("dynamicemb.key_value_table")
    values = [9, -4, 3, 9, 1 << 62, -(1 << 63), 3, -4, 0]
    keys = torch.tensor(values, dtype=torch.int64).view(dtype)
    out = kvt._host_sorted_unique(keys)
    assert out.dtype == dtype
    assert _as_numbers(out, dtype) == sorted({_unsigned(v, dtype) for v in values})
    assert kvt._host_sorted_unique(keys[:1]).numel() == 1
    assert kvt._host_sorted_unique(keys[:0]).numel() == 0


# ---------------------------------------------------------------------------
# the storage
# ---------------------------------------------------------------------------
def _storage(num_tables=2, retain=True, key_type=torch.int64):
    dynamicemb = _need("dynamicemb")
    config = _need("dynamicemb.dynamicemb_config")
    optimizer = _need("dynamicemb.optimizer")
    kvt = _need("dynamicemb.key_value_table")
    strategy = (
        dynamicemb.DynamicEmbScoreStrategy.TIMESTAMP,
        dynamicemb.DynamicEmbScoreStrategy.LFU,
    )
    options = [
        config.DynamicEmbTableOptions(
            index_type=key_type,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=DIM,
            max_capacity=BUCKET,
            init_capacity=BUCKET,
            bucket_capacity=BUCKET,
            safe_check_mode=dynamicemb.DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=strategy,
            evict_strategy=config.DynamicEmbEvictStrategy.LFU,
            evicted_item_mode=(
                dynamicemb.EvictedItemMode.RETAIN_KEY
                if retain
                else dynamicemb.EvictedItemMode.DISCARD
            ),
        )
        for _ in range(num_tables)
    ]
    return kvt.DynamicEmbStorage(
        options, optimizer.SGDDynamicEmbeddingOptimizer(optimizer.OptimizerArgs())
    )


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _masked(members, n, device, rng, key_type=torch.int64):
    """N long arrays with the ``members`` ((table, key) pairs) scattered over them and the
    sentinels (reserved key, table id -1) everywhere else, on the device."""
    keys, tids = [EMPTY] * n, [-1] * n
    for pos, (t, k) in zip(rng.sample(range(n), len(members)), members):
        keys[pos], tids[pos] = k, t
    return _keys_of(keys, key_type, device), torch.tensor(
        tids, dtype=torch.int64, device=device
    )


@pytest.mark.parametrize("key_type", [torch.int64, torch.uint64])
def test_pop_many_and_a_pop_per_table_return_the_keys_that_were_retained(gpu, key_type):
    kvt = _need("dynamicemb.key_value_table")
    device = _device()
    one, other = _storage(3, key_type=key_type), _storage(3, key_type=key_type)
    expected = {0: set(), 1: set(), 2: set()}
    for storage in (one, other):
        # 6 entries: some fold as their count lands, some stay pending
        rng = random.Random(3)
        for step in range(6):
            members = [
                (rng.randrange(0, 3), 100 * step + 10 + rng.randrange(0, 5))
                for _ in range(rng.randrange(0, 6))
            ]
            if step == 2:
                # keys above 2**63 as unsigned numbers: after the others, not before
                members += [(0, -5), (1, -(1 << 62)), (0, 7)]
            arrays = _masked(members, 16, device, rng, key_type)
            kvt._append_evicted_masked(storage._state, *arrays)
            if storage is one:
                for t, k in members:
                    expected[t].add(_unsigned(k, key_type))
    torch.cuda.synchronize()

    got = other.pop_many("pop_evicted_keys", [0, 1, 2])
    assert sorted(got) == [0, 1, 2]
    for t in range(3):
        assert got[t].device.type == "cpu" and got[t].dtype == key_type
        assert _as_numbers(got[t], key_type) == sorted(expected[t]), t
        per_table = one.pop_evicted_keys(t)
        assert per_table.dtype == key_type
        assert _as_numbers(per_table, key_type) == sorted(expected[t]), t
    assert not other._state.evicted_pending
    assert other._state.evicted_key_chunks == []
    assert other._state.evicted_tid_chunks == []
    assert other._state.retain_ring.free_slots() == 8, "every ticket is given back"
    again = other.pop_many("pop_evicted_keys", [0, 1, 2])
    assert all(v.numel() == 0 and v.dtype == key_type for v in again.values())


def test_pop_many_of_some_tables_leaves_the_others(gpu):
    kvt = _need("dynamicemb.key_value_table")
    device = _device()
    rng = random.Random(5)
    storage = _storage(3)
    members = [(0, 5), (1, 6), (2, 7), (0, 5), (1, 8), (2, 9)]
    kvt._append_evicted_masked(storage._state, *_masked(members[:3], 8, device, rng))
    kvt._append_evicted_masked(storage._state, *_masked(members[3:], 8, device, rng))

    first = storage.pop_many("pop_evicted_keys", [0, 2])
    assert sorted(first) == [0, 2]
    assert first[0].tolist() == [5] and first[2].tolist() == [7, 9]
    # table 1's keys stayed, once each at least, and no sentinel was left behind
    assert all(bool((t >= 0).all()) for t in storage._state.evicted_tid_chunks)
    assert storage.pop_many("pop_evicted_keys", [1])[1].tolist() == [6, 8]
    assert storage.pop_many("pop_evicted_keys", []) == {}


def test_pop_many_with_nothing_retained_is_empty_and_does_not_wait(
    gpu, count_syncs, instruments_ok
):
    storage = _storage(2)
    count = count_syncs(lambda i: storage.pop_many("pop_evicted_keys", [0, 1]))
    assert count.blocking == 0, count.describe()
    got = storage.pop_many("pop_erased_keys", [0, 1])
    assert got[0].numel() == 0 and got[0].dtype == torch.int64


def test_pending_entries_of_another_stream_are_popped_in_order(gpu):
    """The entries were made behind a long kernel on a side stream, and the pop is on the
    default stream: it orders itself after them, and needs none of their counts."""
    kvt = _need("dynamicemb.key_value_table")
    device = _device()
    rng = random.Random(7)
    storage = _storage(2)
    side = torch.cuda.Stream()
    expected = {0: set(), 1: set()}
    arrays = []
    with torch.cuda.stream(side):
        for step in range(3):
            members = [(step % 2, 10 * step + 1), (1 - step % 2, 10 * step + 2)]
            arrays.append(_masked(members, 8, device, rng))
            for t, k in members:
                expected[t].add(k)
        torch.cuda._sleep(SLEEP_CYCLES)
        for keys, tids in arrays:
            kvt._append_evicted_masked(storage._state, keys, tids)
    got = storage.pop_many("pop_evicted_keys", [0, 1])
    torch.cuda.synchronize()
    assert {t: set(v.tolist()) for t, v in got.items()} == expected


def test_a_full_ring_costs_the_pop_nothing_more(gpu, count_syncs, instruments_ok):
    """Entries that never got a ring slot used to need a read of their own."""
    sf = _need("dynamicemb.sync_free")
    kvt = _need("dynamicemb.key_value_table")
    device = _device()
    rng = random.Random(9)
    storage = _storage(2)
    storage._state.retain_ring = sf.DeferredRead((1,), torch.int64, slots=1)
    arrays = [
        _masked([(0, 10 * s + 1), (1, 10 * s + 2)], 8, device, rng) for s in range(5)
    ]
    torch.cuda.synchronize()

    def fill_and_pop(i):
        for keys, tids in arrays:
            kvt._append_evicted_masked(storage._state, keys, tids)
        got = storage.pop_many("pop_evicted_keys", [0, 1])
        assert got[0].numel() == 5 and got[1].numel() == 5

    count = count_syncs(fill_and_pop)
    assert count.blocking <= 2.0, count.describe()


# ---------------------------------------------------------------------------
# the waits and the bytes of a pop
# ---------------------------------------------------------------------------
def _module(num_tables, retain=True):
    dynamicemb = _need("dynamicemb")
    tables = _need("dynamicemb.batched_dynamicemb_tables")
    strategy = (
        dynamicemb.DynamicEmbScoreStrategy.TIMESTAMP,
        dynamicemb.DynamicEmbScoreStrategy.LFU,
    )
    options = [
        dynamicemb.DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=DIM,
            max_capacity=256,
            bucket_capacity=BUCKET,
            safe_check_mode=dynamicemb.DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=strategy,
            caching=False,
            evicted_item_mode=(
                dynamicemb.EvictedItemMode.RETAIN_KEY
                if retain
                else dynamicemb.EvictedItemMode.DISCARD
            ),
        )
        for _ in range(num_tables)
    ]
    return tables.BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        output_dtype=torch.float32,
        table_names=["t_%d" % i for i in range(num_tables)],
        feature_table_map=list(range(num_tables)),
        pooling_mode=dynamicemb.DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
    )


@pytest.mark.parametrize("num_tables", [2, 8])
def test_a_module_pop_waits_twice_however_many_tables_there_are(
    gpu, count_syncs, instruments_ok, num_tables
):
    kvt = _need("dynamicemb.key_value_table")
    device = _device()
    rng = random.Random(11)
    module = _module(num_tables)
    state = module._storage._state
    arrays = [
        _masked(
            [(t, 1000 * s + 10 * t + 1) for t in range(num_tables)],
            4 * num_tables,
            device,
            rng,
        )
        for s in range(6)
    ]
    torch.cuda.synchronize()

    def fill_and_pop(i):
        for keys, tids in arrays:
            kvt._append_evicted_masked(state, keys, tids)
        got = module.pop_evicted_keys()
        assert sorted(got) == sorted("t_%d" % t for t in range(num_tables))
        assert all(v.numel() == 6 and v.device.type == "cpu" for v in got.values())

    count = count_syncs(fill_and_pop)
    assert count.blocking <= 2.0, count.describe()


def test_a_pop_copies_what_it_returns_not_what_it_leaves(gpu, monkeypatch):
    sf = _need("dynamicemb.sync_free")
    kvt = _need("dynamicemb.key_value_table")
    device = _device()
    rng = random.Random(13)
    storage = _storage(2)
    # table 1 holds far more keys than table 0
    kvt._append_evicted_masked(
        storage._state,
        *_masked(
            [(0, k) for k in range(5)] + [(1, 1000 + k) for k in range(60)],
            80,
            device,
            rng,
        ),
    )
    pulled = []
    real = sf.pull_to_host

    def spy(*tensors):
        pulled.append(sum(t.numel() for t in tensors))
        return real(*tensors)

    monkeypatch.setattr(sf, "pull_to_host", spy)
    got = storage.pop_many("pop_evicted_keys", [0])
    assert got[0].numel() == 5
    assert pulled == [5], "only the keys that were asked for cross to the host"
    assert storage.pop_many("pop_evicted_keys", [1])[1].numel() == 60
    assert pulled == [5, 60]
