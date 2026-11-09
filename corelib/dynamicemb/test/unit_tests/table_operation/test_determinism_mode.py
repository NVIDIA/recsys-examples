# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""The public inserts of DEMB_DETERMINISM_MODE.

The mode sorts a batch by (bucket, key) and inserts the keys of a bucket one after the other
in a single kernel (``table_insert_sorted`` / ``table_insert_and_evict_sorted``; the kernels
are checked against the waves they replaced in ``test_sorted_insert.py``). This file covers
what the public methods add on top, with the variable set:

* a batch padded with reserved keys (table id -1) on the lanes that hold no key: those lanes
  take no part, so the table, the indices of the real keys and the evictions are those of the
  same batch without the padding, in ONE extension call per public call;
* the overflow bucket of a cache is deterministic too: the host and the device form of the
  count build the same table;
* ``insert_results`` and ``score_out`` are filled (the admission counter reads its scores
  from them), a refused key is reported with ``-(input position + 1)``;
* the rehash and the checkpoint load go through the same kernels;
* LRU_LFU and the collection of evicted keys (``evicted_item_mode=RETAIN_KEY``) are refused,
  and a dynamicemb_extensions built without the sorted inserts fails with a message that says
  to rebuild it, instead of falling back to a path that reads.
"""

import os
import sys

import dynamicemb_extensions
import pytest
import torch
from dynamicemb import sync_free as sf
from dynamicemb.embedding_admission import KVCounter, MultiTableKVCounter
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import InsertResult, ScorePolicy

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BUCKET = 16
BUCKETS_PER_TABLE = 8
NUM_TABLES = 2
KEY_TYPES = [torch.int64, torch.uint64]


@pytest.fixture(autouse=True)
def determinism_mode(monkeypatch):
    monkeypatch.setenv("DEMB_DETERMINISM_MODE", "1")


class _Counted:
    """Wraps an extension function the table module calls, counting the calls."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def _count_calls(monkeypatch, name):
    counted = _Counted(getattr(dynamicemb_extensions, name))
    monkeypatch.setattr(dynamicemb_extensions, name, counted)
    return counted


def _table(key_type, overflow=False, **kwargs):
    return get_scored_table(
        capacity=[BUCKET * BUCKETS_PER_TABLE] * NUM_TABLES,
        bucket_capacity=BUCKET,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score", policy=ScorePolicy.ASSIGN)],
        enable_overflow=overflow,
        **kwargs,
    )


def _i64(keys):
    return keys.view(torch.int64) if keys.dtype == torch.uint64 else keys


class _Batch:
    """The same real keys as exact arrays and padded with reserved ones.

    400 real keys go into 16 buckets of 16 slots, so a bucket gets more keys than it has slots:
    it admits its smallest keys and refuses the rest (a key of a call is never evicted by the
    call), whatever the order of the batch."""

    def __init__(self, key_type, num_real, num_pad, seed=0):
        gen = torch.Generator().manual_seed(seed)
        real = torch.randperm(100_000, generator=gen)[:num_real] + 1
        real_tids = torch.randint(NUM_TABLES, (num_real,), generator=gen)
        size = num_real + num_pad
        # The real keys sit at random positions, the padding lanes in between.
        pos = torch.randperm(size, generator=gen)[:num_real]
        keys = torch.full((size,), -1, dtype=torch.int64)
        tids = torch.full((size,), -1, dtype=torch.int64)
        scores = torch.zeros(size, dtype=torch.int64)
        keys[pos] = real
        tids[pos] = real_tids
        scores[pos] = real
        self.pos = pos.cuda()
        self.pad_mask = keys.cuda() == -1

        self.exact_keys = real.cuda().to(key_type)
        self.exact_tids = real_tids.cuda()
        self.exact_score = self._score(real.cuda())
        self.padded_keys = keys.cuda().to(key_type)
        self.padded_tids = tids.cuda()
        self.padded_score = self._score(scores.cuda())

    @staticmethod
    def _score(values):
        return ScoreArg(
            name="score", value=values.to(torch.uint64), policy=ScorePolicy.ASSIGN
        )


def _triples(keys, indices, tids):
    return sorted(zip(_i64(keys).tolist(), indices.tolist(), tids.tolist()))


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("op", ["insert", "insert_and_evict"])
@pytest.mark.parametrize("async_count", [False, True], ids=["host", "async"])
def test_reserved_lanes_change_nothing(key_type, op, async_count, monkeypatch):
    if op == "insert" and async_count:
        pytest.skip("insert has no eviction count")
    batch = _Batch(key_type, num_real=400, num_pad=60)
    sorted_calls = _count_calls(
        monkeypatch,
        "table_insert_sorted" if op == "insert" else "table_insert_and_evict_sorted",
    )

    exact = _table(key_type)
    result_exact = getattr(exact, op)(
        batch.exact_keys, batch.exact_tids, batch.exact_score
    )
    assert sorted_calls.calls == 1, "one extension call per public call"
    sorted_calls.calls = 0

    padded = _table(key_type)
    kwargs = {"async_count": async_count} if op == "insert_and_evict" else {}
    result_padded = getattr(padded, op)(
        batch.padded_keys, batch.padded_tids, batch.padded_score, **kwargs
    )
    # The padding costs nothing: still one call, with no wave of its own for the reserved
    # lanes (there are no waves).
    assert sorted_calls.calls == 1
    # The table is the one the exact batch built, slot for slot.
    assert torch.equal(exact.table_storage_, padded.table_storage_)
    assert torch.equal(exact.bucket_sizes, padded.bucket_sizes)

    indices_exact = result_exact if op == "insert" else result_exact[0]
    indices_padded = result_padded if op == "insert" else result_padded[0]
    assert torch.equal(indices_padded[batch.pos], indices_exact)
    assert bool((indices_padded[batch.pad_mask] == -1).all())
    if op == "insert":
        return

    # The evictions are those of the exact batch, and nothing else.
    (_, count_exact, keys_exact, idx_exact, _, tids_exact) = result_exact
    (_, num_evicted, keys_padded, idx_padded, _, tids_padded) = result_padded
    assert count_exact > 0, "the batch must overflow the buckets"
    if async_count:
        assert isinstance(num_evicted, torch.Tensor) and num_evicted.is_cuda
        assert tuple(num_evicted.shape) == (1,)
        count_padded = int(num_evicted.item())
        size = batch.padded_keys.numel()
        assert keys_padded.numel() == idx_padded.numel() == tids_padded.numel() == size
        assert bool((_i64(keys_padded)[count_padded:] == -1).all())
        assert bool((idx_padded[count_padded:] == -1).all())
        assert bool((tids_padded[count_padded:] == -1).all())
    else:
        assert isinstance(num_evicted, int)
        count_padded = num_evicted
        assert keys_padded.numel() == count_padded
    assert count_padded == count_exact
    live = slice(0, count_padded)
    assert bool((_i64(keys_padded)[live] != -1).all())
    # A refused key (a bucket with more new keys than slots refuses the last of them) is reported
    # with -(input position + 1): the position of exact key i in the padded batch is pos[i].
    refused = idx_exact < 0
    idx_expected = torch.where(
        refused, -(batch.pos[(-idx_exact - 1).clamp(min=0)] + 1), idx_exact
    )
    assert bool(refused.any()), "the batch must refuse keys"
    assert _triples(keys_padded[live], idx_padded[live], tids_padded[live]) == (
        _triples(keys_exact, idx_expected, tids_exact)
    )
    # Sorted by (bucket, key) and the same on every run: the order is part of the result,
    # not just the set.
    again = _table(key_type)
    result_again = getattr(again, op)(
        batch.padded_keys, batch.padded_tids, batch.padded_score, **kwargs
    )
    assert torch.equal(_i64(result_again[2])[live], _i64(keys_padded)[live])
    assert torch.equal(result_again[3][live], idx_padded[live])


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_cache_variant_async_count(key_type):
    """The overflow variant, which a cache uses: tails inert, no reserved key among the
    evictions, and the same table, overflow bucket and evicted arrays whichever form of the
    count is asked for (the overflow bucket is deterministic too)."""
    batch = _Batch(key_type, num_real=400, num_pad=60)
    size = batch.padded_keys.numel()
    real = set(_i64(batch.exact_keys).tolist())
    tables, results = [], []
    for async_count in (False, True):
        table = _table(key_type, overflow=True)
        result = table.insert_and_evict_with_counter_and_overflow(
            batch.padded_keys,
            batch.padded_tids,
            batch.padded_score,
            async_count=async_count,
        )
        indices, num_evicted, ev_keys, ev_indices, _, ev_tids = result
        assert bool((indices[batch.pad_mask] == -1).all())
        if async_count:
            assert tuple(num_evicted.shape) == (1,) and num_evicted.is_cuda
            count = int(num_evicted.item())
            assert ev_keys.numel() == ev_indices.numel() == ev_tids.numel() == size
            assert bool((_i64(ev_keys)[count:] == -1).all())
            assert bool((ev_indices[count:] == -1).all())
            assert bool((ev_tids[count:] == -1).all())
        else:
            count = num_evicted
            assert isinstance(count, int)
            assert ev_keys.numel() == count
        assert 0 < count <= len(real)
        assert set(_i64(ev_keys)[:count].tolist()) <= real
        tables.append(table)
        results.append((count, _i64(ev_keys)[:count], ev_indices[:count]))
    host, device = tables
    assert torch.equal(host.table_storage_, device.table_storage_)
    assert torch.equal(host.overflow_table_storage_, device.overflow_table_storage_)
    assert results[0][0] == results[1][0]
    assert torch.equal(results[0][1], results[1][1])
    assert torch.equal(results[0][2], results[1][2])


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("op", ["insert", "insert_and_evict"])
@pytest.mark.parametrize("async_count", [False, True], ids=["host", "async"])
def test_all_lanes_reserved(key_type, op, async_count):
    if op == "insert" and async_count:
        pytest.skip("insert has no eviction count")
    size = 64
    keys = torch.full((size,), -1, dtype=torch.int64, device="cuda").to(key_type)
    tids = torch.full((size,), -1, dtype=torch.int64, device="cuda")
    zeros = torch.zeros(size, dtype=torch.uint64, device="cuda")
    score = ScoreArg(name="score", value=zeros, policy=ScorePolicy.ASSIGN)

    table = _table(key_type)
    kwargs = {"async_count": async_count} if op == "insert_and_evict" else {}
    result = getattr(table, op)(keys, tids, score, **kwargs)

    assert torch.equal(table.table_storage_, _table(key_type).table_storage_)
    assert bool((table.bucket_sizes == 0).all())
    indices = result if op == "insert" else result[0]
    assert bool((indices == -1).all())
    if op == "insert_and_evict":
        num_evicted = result[1]
        assert (int(num_evicted.item()) if async_count else num_evicted) == 0


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_insert_results_and_scores_are_reported(key_type):
    """The insert fills ``insert_results`` and ``score_out`` for every input position (the
    admission counter reads its scores from them), padding lanes included."""
    table = _table(key_type)
    keys = torch.tensor([5, 6, -1, 7, 8], dtype=torch.int64, device="cuda").to(key_type)
    tids = torch.tensor([0, 1, -1, 0, 1], dtype=torch.int64, device="cuda")
    values = torch.tensor([50, 60, 99, 70, 80], dtype=torch.int64, device="cuda")
    score = ScoreArg(
        name="score", value=values.to(torch.uint64), policy=ScorePolicy.ASSIGN
    )
    results = torch.empty(5, dtype=table.result_type, device="cuda")
    score_out = torch.empty(5, dtype=torch.int64, device="cuda")
    table.insert(keys, tids, score, results, score_out=score_out)
    assert results.tolist() == [
        InsertResult.INSERT.value,
        InsertResult.INSERT.value,
        InsertResult.ILLEGAL.value,
        InsertResult.INSERT.value,
        InsertResult.INSERT.value,
    ]
    assert score_out.tolist() == [50, 60, 99, 70, 80]
    table.insert(keys, tids, score, results, score_out=score_out)
    assert results.tolist() == [
        InsertResult.ASSIGN.value,
        InsertResult.ASSIGN.value,
        InsertResult.ILLEGAL.value,
        InsertResult.ASSIGN.value,
        InsertResult.ASSIGN.value,
    ]


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_the_admission_counter_counts(key_type):
    """MultiTableKVCounter.add returns the count of a key after this call, read from the
    ``score_out`` of the insert: a key seen once reads 1, seen twice reads 2. In the mode it
    used to read memory nobody wrote."""
    counter = MultiTableKVCounter(
        [KVCounter(capacity=1024, bucket_capacity=128, key_type=key_type)],
        torch.device("cuda"),
    )
    keys = torch.arange(1, 65, dtype=torch.int64, device="cuda").to(key_type)
    tids = torch.zeros(64, dtype=torch.int64, device="cuda")
    ones = torch.ones(64, dtype=torch.int64, device="cuda")
    assert bool((counter.add(keys, tids, ones) == 1).all())
    assert bool((counter.add(keys, tids, ones) == 2).all())
    assert bool((counter.add(keys[:32], tids[:32], ones[:32]) == 3).all())


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_a_rehash_insert_carries_the_rows(key_type):
    """``insert_carrying_rows`` (the rehash) goes through the sorted insert: the table and the
    rows it hands out are those of the waves plus ``set_rows``."""
    import wave_oracle as oracle

    def table():
        return get_scored_table(
            capacity=[BUCKET * BUCKETS_PER_TABLE],
            bucket_capacity=BUCKET,
            key_type=key_type,
            score_specs=[ScoreSpec(name="score", policy=ScorePolicy.ASSIGN)],
            use_rows=True,
        )

    gen = torch.Generator().manual_seed(4)
    keys = (torch.randperm(10_000, generator=gen)[:40] + 1).cuda().to(key_type)
    scores = torch.randint(1, 100, (40,), generator=gen).cuda().to(torch.uint64)
    rows = torch.randperm(1000, generator=gen)[:40].cuda()
    tids = torch.zeros(40, dtype=torch.int64, device="cuda")
    carried, waves = table(), table()
    carried.insert_carrying_rows(
        keys,
        0,
        ScoreArg(name="score", value=scores, policy=ScorePolicy.ASSIGN),
        rows,
    )
    slots = oracle.insert(waves, keys, tids, scores, ScorePolicy.ASSIGN)
    waves.set_rows(slots, 0, rows)
    assert torch.equal(_i64(carried.keys_), _i64(waves.keys_))
    assert torch.equal(carried.digests_, waves.digests_)
    assert torch.equal(_i64(carried.scores_list[0]), _i64(waves.scores_list[0]))
    assert torch.equal(carried.rows_, waves.rows_)
    assert torch.equal(carried.bucket_sizes, waves.bucket_sizes)


def test_a_checkpoint_load_is_deterministic(tmp_path, monkeypatch):
    """``load`` inserts the file in chunks through ``insert``: two loads of one file build one
    table, and every key of the file is in it."""
    monkeypatch.delenv("DEMB_DETERMINISM_MODE")
    source = _table(torch.int64)
    keys = torch.arange(1, 129, dtype=torch.int64, device="cuda")
    tids = torch.zeros(128, dtype=torch.int64, device="cuda")
    scores = ScoreArg(
        name="score", value=(keys * 3).to(torch.uint64), policy=ScorePolicy.ASSIGN
    )
    source.insert(keys, tids, scores)
    key_file, score_file = str(tmp_path / "keys.bin"), str(tmp_path / "scores.bin")
    source.dump(key_file, {"score": score_file}, table_id=0)

    monkeypatch.setenv("DEMB_DETERMINISM_MODE", "1")
    first, second = _table(torch.int64), _table(torch.int64)
    for table in (first, second):
        table.load(key_file, {"score": score_file}, table_id=0)
    assert torch.equal(first.table_storage_, second.table_storage_)
    found = ScoreArg(name="score", value=None, policy=ScorePolicy.CONST)
    _, founds, _ = first.lookup(keys, tids, found)
    assert bool(founds.all())


def test_lru_lfu_is_refused():
    """LRU_LFU keeps two score words and is evicted by a cubin: the sorted inserts take one word.
    The table refuses it before it reaches the extension, and the extension refuses it too (the
    Python check is an ``assert``, which ``python -O`` removes)."""
    table = get_scored_table(
        capacity=[256],
        bucket_capacity=128,
        key_type=torch.int64,
        score_specs=[ScoreSpec(name="freq", policy=ScorePolicy.LRU_LFU)],
    )
    keys = torch.arange(1, 9, dtype=torch.int64, device="cuda")
    tids = torch.zeros(8, dtype=torch.int64, device="cuda")
    freq = torch.ones(8, dtype=torch.uint64, device="cuda")
    score = ScoreArg(name="freq", value=freq, policy=ScorePolicy.LRU_LFU)
    with pytest.raises(AssertionError, match="auxiliary score columns"):
        table.insert(keys, tids, score)
    with pytest.raises(RuntimeError, match="LRU_LFU"):
        dynamicemb_extensions.table_insert_sorted(
            table.table_storage_,
            table.table_bucket_offsets_,
            table.bucket_capacity_,
            table.bucket_sizes,
            keys,
            tids,
            freq,
            ScorePolicy.LRU_LFU,
            table._ref_counter,
            num_scores=table.num_scores_,
        )


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_collecting_the_evicted_keys_is_refused(key_type):
    """``collect_evicted`` (the last tier's RETAIN_KEY) has no sorted form: a clear error, not a
    silent fall back to the insert that collects them in the order of an atomic."""
    batch = _Batch(key_type, num_real=100, num_pad=10)
    with pytest.raises(RuntimeError, match="RETAIN_KEY"):
        _table(key_type).insert(
            batch.padded_keys,
            batch.padded_tids,
            batch.padded_score,
            collect_evicted=True,
        )


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_an_extension_without_the_sorted_inserts_says_to_rebuild(key_type, monkeypatch):
    """No fallback to a path that reads: a build without ``sorted_insert`` raises at the first
    insert in the mode, naming the build command. Without the variable nothing changes.
    """
    batch = _Batch(key_type, num_real=100, num_pad=10)
    monkeypatch.setattr(sf, "SORTED_INSERT", False)
    insert = lambda t: t.insert(  # noqa: E731
        batch.padded_keys, batch.padded_tids, batch.padded_score
    )
    evict = lambda t: t.insert_and_evict(  # noqa: E731
        batch.padded_keys, batch.padded_tids, batch.padded_score
    )
    cache = lambda t: t.insert_and_evict_with_counter_and_overflow(  # noqa: E731
        batch.padded_keys, batch.padded_tids, batch.padded_score
    )
    for call, overflow in ((insert, False), (evict, False), (cache, True)):
        with pytest.raises(RuntimeError, match="build_ext"):
            call(_table(key_type, overflow=overflow))
    monkeypatch.delenv("DEMB_DETERMINISM_MODE")
    for call, overflow in ((insert, False), (evict, False), (cache, True)):
        call(_table(key_type, overflow=overflow))
