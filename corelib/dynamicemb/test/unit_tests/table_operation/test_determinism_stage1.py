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

"""DEMB_DETERMINISM_MODE inserts of a batch padded with reserved keys.

The deterministic insert sorts a batch by bucket and inserts it one wave at a time
(at most one key per bucket in a wave). A batch padded to a fixed length carries the
reserved all-ones key, with table id -1, on the lanes that hold no key. Those lanes
must take no part: not in the waves (each of them would hash to the same bucket and
need a wave of its own), not in the table, and not among the evictions. So the table
state, the indices of the real keys, the evictions and the number of waves must be
those of the same batch without the padding.
"""

import pytest
import torch
from dynamicemb import scored_hashtable as sht
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import ScorePolicy

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
    counted = _Counted(getattr(sht, name))
    monkeypatch.setattr(sht, name, counted)
    return counted


def _count_reads(monkeypatch):
    """Counts the host reads Python itself issues: ``Tensor.item``/``cpu``/``tolist``/
    ``numpy`` and ``torch.nonzero``. The read inside the extension's ``bucketize_keys``
    is C++ and cannot be seen from here."""
    reads = {}

    def count(owner, name):
        original = getattr(owner, name)
        reads[name] = 0

        def counted(*args, **kwargs):
            reads[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)

    count(torch, "nonzero")
    for name in ("item", "cpu", "tolist", "numpy"):
        count(torch.Tensor, name)
    return reads


def _table(key_type, overflow=False):
    return get_scored_table(
        capacity=[BUCKET * BUCKETS_PER_TABLE] * NUM_TABLES,
        bucket_capacity=BUCKET,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score", policy=ScorePolicy.ASSIGN)],
        enable_overflow=overflow,
    )


def _i64(keys):
    return keys.view(torch.int64) if keys.dtype == torch.uint64 else keys


class _Batch:
    """The same real keys as exact arrays and padded with reserved ones.

    A real key scores its own value, so within a bucket a later wave (a larger key)
    always outscores the residents and evicts the lowest one: the outcome does not
    depend on which keys share a wave, nor on how the kernels race."""

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
    wave_calls = _count_calls(
        monkeypatch, "table_insert" if op == "insert" else "table_insert_and_evict"
    )

    exact = _table(key_type)
    result_exact = getattr(exact, op)(
        batch.exact_keys, batch.exact_tids, batch.exact_score
    )
    waves_exact = wave_calls.calls
    wave_calls.calls = 0

    padded = _table(key_type)
    kwargs = {"async_count": async_count} if op == "insert_and_evict" else {}
    result_padded = getattr(padded, op)(
        batch.padded_keys, batch.padded_tids, batch.padded_score, **kwargs
    )
    waves_padded = wave_calls.calls

    # The padding costs no wave: several waves ran, and as many as without it.
    assert waves_exact > 1
    assert waves_padded == waves_exact
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
    assert _triples(keys_padded[live], idx_padded[live], tids_padded[live]) == (
        _triples(keys_exact, idx_exact, tids_exact)
    )


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_cache_variant_async_count(key_type):
    """The overflow variant, which a cache uses: tails inert, no reserved key among
    the evictions. (Its overflow bucket is shared across buckets, so a table state
    is not compared.)"""
    batch = _Batch(key_type, num_real=400, num_pad=60)
    size = batch.padded_keys.numel()
    real = set(_i64(batch.exact_keys).tolist())
    counts = []
    for async_count in (False, True):
        table = _table(key_type, overflow=True)
        (
            indices,
            num_evicted,
            ev_keys,
            ev_indices,
            _,
            ev_tids,
        ) = table.insert_and_evict_with_counter_and_overflow(
            batch.padded_keys,
            batch.padded_tids,
            batch.padded_score,
            async_count=async_count,
        )
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
        counts.append(count)
    # Both modes run the same waves, so they evict as much.
    assert counts[0] == counts[1]


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
def test_bucketize_and_pad_drops_reserved_keys(key_type):
    batch = _Batch(key_type, num_real=300, num_pad=40)
    table = _table(key_type)
    keys_t, tids_t, scores_t = table._bucketize_and_pad(
        batch.padded_keys, batch.padded_tids, batch.padded_score.value
    )
    keys_e, tids_e, scores_e = table._bucketize_and_pad(
        batch.exact_keys, batch.exact_tids, batch.exact_score.value
    )
    # Exactly the waves of the batch without the padding, and only real keys in them.
    assert keys_t.dtype == keys_e.dtype == key_type
    assert torch.equal(_i64(keys_t), _i64(keys_e))
    assert torch.equal(tids_t, tids_e)
    assert torch.equal(scores_t, scores_e)
    real = _i64(keys_t) != -1
    assert int(real.sum()) == batch.exact_keys.numel()
    assert sorted(_i64(keys_t)[real].tolist()) == sorted(
        _i64(batch.exact_keys).tolist()
    )
    # Each real key travels with its own table id and score (its value).
    tid_of = dict(zip(_i64(batch.exact_keys).tolist(), batch.exact_tids.tolist()))
    for key, tid, score in zip(
        _i64(keys_t)[real].tolist(), tids_t[real].tolist(), scores_t[real].tolist()
    ):
        assert tid == tid_of[key]
        assert score == key
    # Padding lanes: reserved key, table id 0, score 0.
    assert bool((tids_t[~real] == 0).all())
    assert bool((scores_t[~real] == 0).all())


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize(
    "op", ["insert", "insert_and_evict", "insert_and_evict_with_counter_and_overflow"]
)
@pytest.mark.parametrize("async_count", [False, True], ids=["host", "async"])
def test_reads_do_not_grow_with_the_waves(key_type, op, async_count, monkeypatch):
    """However many waves a batch takes, a call reads back twice from Python: the
    ``nonzero`` of the reserved-key pre-filter and the wave count (``lengths.max()``);
    with the host count of the evictions, once more. Together with the read inside the
    extension's ``bucketize_keys`` that is the 3 reads per call the mode is allowed,
    and none of them is per wave."""
    if op == "insert" and async_count:
        pytest.skip("insert has no eviction count")
    batch = _Batch(key_type, num_real=400, num_pad=60)
    table = _table(key_type, overflow=op.endswith("overflow"))
    kwargs = {"async_count": async_count} if op != "insert" else {}
    waves = _count_calls(
        monkeypatch, "table_insert" if op == "insert" else "table_insert_and_evict"
    )

    with monkeypatch.context() as patches:
        reads = _count_reads(patches)
        getattr(table, op)(
            batch.padded_keys, batch.padded_tids, batch.padded_score, **kwargs
        )
    torch.cuda.synchronize()

    assert waves.calls > 1, "the batch must take several waves"
    host_count = op != "insert" and not async_count
    assert reads == {
        "nonzero": 1,
        "item": 1 + int(host_count),
        "cpu": 0,
        "tolist": 0,
        "numpy": 0,
    }
