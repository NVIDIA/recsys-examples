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

"""``insert_and_evict(..., async_count=True)`` leaves the eviction count on the device.

The kernel counts its evictions into an int64[1] device tensor and writes only that
many entries of its four evicted_* arrays. With ``async_count`` the table hands the
counter back unread, the arrays keep the length of the batch, and everything past the
count is inert (key -1, index -1, table id -1), so a caller can use them as they are.
Without it, the default, the result is what it always was: a host int and arrays
sliced to it.

Every table here is one bucket of 16 slots, so all keys compete for the same slots
and the outcome of a batch is known: a newcomer scoring above every resident evicts
the lowest-scoring one (the existing eviction tests rely on the same), and a bucket
whose slots are all pinned refuses every newcomer.
"""

import pytest
import torch
from dynamicemb import sync_free as sf
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import ScorePolicy

BUCKET = 16
KEY_TYPES = [torch.int64, torch.uint64]
# "storage" and "storage_rows" go through insert_and_evict (a table that stores rows
# resolves the evicted indices to rows), "cache" through the overflow variant.
KINDS = ["storage", "storage_rows", "cache"]


@pytest.fixture(params=[False, True], ids=["default", "determinism"])
def deterministic(request, monkeypatch):
    """The same expectations hold in DEMB_DETERMINISM_MODE: one bucket, ties by slot, the
    outcome of a batch is known."""
    if request.param:
        monkeypatch.setenv("DEMB_DETERMINISM_MODE", "1")
    return request.param


def _table(kind, key_type):
    return get_scored_table(
        capacity=[BUCKET],
        bucket_capacity=BUCKET,
        key_type=key_type,
        score_specs=[ScoreSpec(name="score", policy=ScorePolicy.ASSIGN)],
        use_rows=kind == "storage_rows",
        enable_overflow=kind == "cache",
    )


def _insert_and_evict(table, kind, *args, **kwargs):
    if kind == "cache":
        return table.insert_and_evict_with_counter_and_overflow(*args, **kwargs)
    return table.insert_and_evict(*args, **kwargs)


def _keys(values, key_type):
    return torch.tensor(list(values), dtype=torch.int64, device="cuda").to(key_type)


def _i64(keys):
    return keys.view(torch.int64) if keys.dtype == torch.uint64 else keys


def _args(keys, scores):
    tids = torch.zeros(keys.numel(), dtype=torch.int64, device=keys.device)
    scores = torch.tensor(list(scores), dtype=torch.int64, device=keys.device)
    score = ScoreArg(
        name="score", value=scores.to(torch.uint64), policy=ScorePolicy.ASSIGN
    )
    return tids, score


def _const_lookup(table, keys, tids):
    zeros = torch.zeros(keys.numel(), dtype=torch.uint64, device=keys.device)
    score = ScoreArg(name="score", value=zeros, policy=ScorePolicy.CONST)
    _, founds, indices = table.lookup(keys, tids, score)
    return founds, indices


def _live(num_evicted, async_count):
    """The host count of a result, whichever form it came in."""
    if async_count:
        assert isinstance(num_evicted, torch.Tensor)
        assert num_evicted.is_cuda and num_evicted.dtype == torch.int64
        assert tuple(num_evicted.shape) == (1,)
        return int(num_evicted.item())
    assert isinstance(num_evicted, int)
    return num_evicted


def _check_tail(batch_size, count, ev_keys, ev_indices, ev_tids):
    """The arrays are batch-long and inert past the count."""
    assert ev_keys.numel() == ev_indices.numel() == ev_tids.numel() == batch_size
    assert bool((_i64(ev_keys)[count:] == -1).all())
    assert bool((ev_indices[count:] == -1).all())
    assert bool((ev_tids[count:] == -1).all())


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("python_tails", [False, True], ids=["as_built", "py_tails"])
@pytest.mark.parametrize("async_count", [False, True], ids=["host", "async"])
def test_evictions(
    key_type, kind, python_tails, async_count, deterministic, monkeypatch
):
    if python_tails:
        # Make the tails inert in Python even when the extension already does.
        monkeypatch.setattr(sf, "EVICT_TAIL_INERT", False)
    table = _table(kind, key_type)
    old = _keys(range(1, BUCKET + 1), key_type)
    tids, score = _args(old, range(1, BUCKET + 1))
    table.insert(old, tids, score)  # exactly full
    founds, old_index = _const_lookup(table, old, tids)
    assert bool(founds.all())
    index_of = dict(zip(_i64(old).tolist(), old_index.tolist()))

    # Four newcomers outscoring every resident evict the four lowest-scoring ones;
    # four residents are assigned again and are found, not evicted.
    newcomers = _keys(range(101, 105), key_type)
    kept = old[8:12]
    # Concatenated as int64: no existing test cats uint64 tensors, and not every torch
    # build has a kernel for it.
    batch = torch.cat([_i64(newcomers), _i64(kept)]).to(key_type)
    tids_b, score_b = _args(batch, list(range(1000, 1004)) + list(range(2000, 2004)))
    (indices, num_evicted, ev_keys, ev_indices, _, ev_tids) = _insert_and_evict(
        table, kind, batch, tids_b, score_b, async_count=async_count
    )

    count = _live(num_evicted, async_count)
    assert count == newcomers.numel()
    if async_count:
        _check_tail(batch.numel(), count, ev_keys, ev_indices, ev_tids)
    else:
        assert ev_keys.numel() == ev_indices.numel() == ev_tids.numel() == count
    victims = _i64(ev_keys)[:count].tolist()
    assert sorted(victims) == [1, 2, 3, 4]
    # Each victim comes with the index it was found at, and the table it lives in.
    assert ev_indices[:count].tolist() == [index_of[k] for k in victims]
    assert bool((ev_tids[:count] == 0).all())
    # The newcomers took exactly the victims' indices, the others kept theirs.
    assert sorted(indices[:4].tolist()) == sorted(ev_indices[:count].tolist())
    assert indices[4:].tolist() == [index_of[k] for k in _i64(kept).tolist()]

    founds, _ = _const_lookup(table, old, tids)
    assert founds.tolist() == [False] * 4 + [True] * 12
    founds, _ = _const_lookup(table, newcomers, tids[:4])
    assert bool(founds.all())


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("kind", ["storage", "storage_rows"])
@pytest.mark.parametrize("python_tails", [False, True], ids=["as_built", "py_tails"])
@pytest.mark.parametrize("async_count", [False, True], ids=["host", "async"])
def test_refused_inserts_keep_their_negative_index(
    key_type, kind, python_tails, async_count, deterministic, monkeypatch
):
    """A key the table refuses comes back as an "evicted" entry whose index is
    -(position + 1), also with the count on the device and for a table of rows."""
    if python_tails:
        monkeypatch.setattr(sf, "EVICT_TAIL_INERT", False)
    table = _table(kind, key_type)
    old = _keys(range(1, BUCKET + 1), key_type)
    tids, score = _args(old, range(1, BUCKET + 1))
    table.insert(old, tids, score)
    # Pin every slot: nothing can be evicted, so every newcomer is refused.
    table.increment_counter(table.slots_of(old, tids), tids)

    newcomers = _keys(range(101, 109), key_type)
    # Positions 8..15 are the newcomers (concatenated as int64, see test_evictions).
    batch = torch.cat([_i64(old[:8]), _i64(newcomers)]).to(key_type)
    tids_b, score_b = _args(batch, range(1000, 1000 + batch.numel()))
    (indices, num_evicted, ev_keys, ev_indices, _, ev_tids) = table.insert_and_evict(
        batch, tids_b, score_b, async_count=async_count
    )

    count = _live(num_evicted, async_count)
    assert count == newcomers.numel()
    if async_count:
        _check_tail(batch.numel(), count, ev_keys, ev_indices, ev_tids)
    assert sorted(ev_indices[:count].tolist()) == list(range(-16, -8))
    assert sorted(_i64(ev_keys)[:count].tolist()) == list(range(101, 109))
    assert bool((ev_tids[:count] == 0).all())
    assert bool((indices[:8] >= 0).all())
    assert bool((indices[8:] == -1).all())


@pytest.mark.parametrize("kind", KINDS)
def test_empty_batch(kind):
    table = _table(kind, torch.int64)
    keys = torch.empty(0, dtype=torch.int64, device="cuda")
    tids, score = _args(keys, [])
    (indices, num_evicted, ev_keys, ev_indices, ev_scores, ev_tids) = _insert_and_evict(
        table, kind, keys, tids, score, async_count=True
    )
    assert _live(num_evicted, True) == 0
    for t in (indices, ev_keys, ev_indices, ev_scores, ev_tids):
        assert t.numel() == 0


@pytest.mark.parametrize("op", ["insert", "insert_and_evict"])
def test_lru_lfu_without_score_out_tolerates_reserved_keys(op):
    """A 2-word table gets a scratch score_out of its own when the caller gives none,
    so the LruLfu cubin never writes its scores through a null pointer, which a
    reserved key (the padding of an N-aligned batch) would otherwise make it do in
    an extension built without the null guard. Because of that scratch this test
    passes with or without the guard: a test of the guard itself has to call the
    extension's table_insert with score_output=None."""
    table = get_scored_table(
        capacity=[256],
        bucket_capacity=128,
        key_type=torch.int64,
        score_specs=[
            ScoreSpec(name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True)
        ],
    )
    values = [1001, -1, 1002, -2, -3, -4, 1003]
    keys = torch.tensor(values, dtype=torch.int64, device="cuda")
    tids = torch.zeros(keys.numel(), dtype=torch.int64, device="cuda")
    freq = torch.ones(keys.numel(), dtype=torch.uint64, device="cuda")
    score = ScoreArg(name="frequency", value=freq, policy=ScorePolicy.LRU_LFU)
    if op == "insert":
        indices = table.insert(keys, tids, score)
    else:
        indices, num_evicted, *_ = table.insert_and_evict(
            keys, tids, score, async_count=True
        )
        assert _live(num_evicted, True) == 0
    torch.cuda.synchronize()
    assert indices[[1, 3, 4, 5]].tolist() == [-1] * 4
    assert bool((indices[[0, 2, 6]] >= 0).all())
