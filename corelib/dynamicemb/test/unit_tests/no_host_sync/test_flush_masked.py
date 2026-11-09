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

"""A flush of a cache over a storage waits for nothing.

``flush_cache`` exports every batch of the cache at full length, with the entries past the
count made inert (the reserved key, table id and slot ``-1``), and writes it back with the
masked write-back: the host never learns how many entries a batch holds. The reference is
the flush it replaced, which read the count of every batch and cut the arrays to it (kept
below as ``exact_flush``, built from the exact export that is still in the tree).

What is checked: the storage after the two flushes is the same (values of the keys it
holds, the keys it retains as evicted, the keys that reach a tier below), with the cache's
overflow region occupied; a flush of a cache and of an empty one waits for the device zero
times, with a storage that retains evicted keys not more than once in ``retain_ring``
batches; the values a flush loads per batch stay inside ``FLUSH_BATCH_BYTES``.
"""

import importlib

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)

CACHE_BUCKET = 1024  # the cache is one bucket per table, as the module builds it
STORAGE_CAPACITY = 4096


def _need(name):
    """The module, or a skip: a tree without the extension has nothing to measure."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip("%s is not importable here: %s" % (name, e))


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _options(dim, capacity, bucket, evict, strategy, retain):
    dynamicemb = _need("dynamicemb")
    config = _need("dynamicemb.dynamicemb_config")
    return config.DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        dim=dim,
        max_capacity=capacity,
        init_capacity=capacity,
        bucket_capacity=bucket,
        safe_check_mode=dynamicemb.DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=1024**3,
        score_strategy=strategy,
        evict_strategy=evict,
        evicted_item_mode=(
            dynamicemb.EvictedItemMode.RETAIN_KEY
            if retain
            else dynamicemb.EvictedItemMode.DISCARD
        ),
    )


def _cache_value(keys, dim, device):
    """What the cache holds for each of ``keys`` (an int64 tensor): ``key + column / 16``."""
    base = keys.to(device=device, dtype=torch.float32).unsqueeze(1)
    return base + torch.arange(dim, dtype=torch.float32, device=device) / 16


def _storage_value(keys, dim, device):
    return -_cache_value(keys, dim, device)


def _pair(retain=False, dim=8):
    """A cache over a storage, as the module builds them, with known contents.

    Table 0: the cache holds keys 1..1024, each pinned (a prefetched batch holds it), so
    its one bucket is full of keys that cannot be evicted, and keys 2001..2050 go to the
    overflow. Table 1: 300 keys that are not pinned. The storage holds the cache's keys that
    satisfy a rule (about 70% and 2/3), with another value, and some keys the cache does not
    have. Returns ``(cache, storage, cache_keys)``.
    """
    dynamicemb = _need("dynamicemb")
    config = _need("dynamicemb.dynamicemb_config")
    optimizer = _need("dynamicemb.optimizer")
    kvt = _need("dynamicemb.key_value_table")
    device = _device()
    sgd = optimizer.SGDDynamicEmbeddingOptimizer(optimizer.OptimizerArgs())
    cache = kvt.DynamicEmbCache(
        [
            _options(
                dim,
                CACHE_BUCKET,
                CACHE_BUCKET,
                config.DynamicEmbEvictStrategy.LRU,
                dynamicemb.DynamicEmbScoreStrategy.TIMESTAMP,
                False,
            )
            for _ in range(2)
        ],
        sgd,
    )
    storage = kvt.DynamicEmbStorage(
        [
            _options(
                dim,
                STORAGE_CAPACITY,
                128,
                config.DynamicEmbEvictStrategy.LRU,
                dynamicemb.DynamicEmbScoreStrategy.TIMESTAMP,
                retain,
            )
            for _ in range(2)
        ],
        sgd,
        use_ref_counter=False,
    )
    cache_keys = {
        0: list(range(1, 1025)) + list(range(2001, 2051)),
        1: list(range(5000, 5300)),
    }
    in_storage = {
        0: [k for k in cache_keys[0] if k % 10 < 7 or k > 2000 and k % 2 == 0]
        + [9000, 9001, 9002],
        1: [k for k in cache_keys[1] if k % 3 != 0] + [9100, 9101],
    }
    for table in (0, 1):
        keys = torch.tensor(in_storage[table], dtype=torch.int64, device=device)
        tids = torch.full_like(keys, table)
        storage.insert(keys, tids, _storage_value(keys, dim, device))
    for table, pin in ((0, True), (1, False)):
        # the pinned keys first, so that the others find the bucket full
        groups = [cache_keys[table][:1024], cache_keys[table][1024:]]
        for group in groups:
            if not group:
                continue
            keys = torch.tensor(group, dtype=torch.int64, device=device)
            tids = torch.full_like(keys, table)
            indices = cache.insert_and_evict(keys, tids)[0]
            kvt.store_to_flat(
                cache._state, indices, tids, _cache_value(keys, dim, device)
            )
            if pin and group is groups[0]:
                cache.increment_counter(indices, tids)
    torch.cuda.synchronize()
    return cache, storage, cache_keys


def exact_flush(cache, storage, external_storage=None):
    """The flush before it stopped counting: the exact export, one count read per batch."""
    kvt = _need("dynamicemb.key_value_table")
    types = _need("dynamicemb.types")
    state = cache._state
    for table in range(state.num_tables):
        for keys, _, indices in state.key_index_map._batched_export_keys_scores(
            [state.score_policy.name],
            state.device,
            batch_size=state.threads_in_wave,
            return_index=True,
            table_id=table,
        ):
            tid = torch.full(
                (keys.numel(),), table, dtype=torch.int64, device=keys.device
            )
            values = kvt.load_from_flat(
                state, indices, tid, copy_mode=types.CopyMode.VALUE
            )
            kvt._write_back_evicted(storage, external_storage, keys, tid, values)


def _contents(storage, table, dim):
    """``{key: values}`` of a table of the storage, read the slow way."""
    out = {}
    for keys, embeddings, _, _ in storage.export_keys_values(_device(), 4096, table):
        for key, row in zip(keys.cpu().tolist(), embeddings.cpu()):
            out[key] = row.clone()
    return out


def _same_contents(one, other, tables=(0, 1), dim=8):
    for table in tables:
        a, b = _contents(one, table, dim), _contents(other, table, dim)
        assert sorted(a) == sorted(b), table
        for key in a:
            assert torch.equal(a[key], b[key]), (table, key)


@pytest.mark.parametrize("retain", [False, True])
def test_the_scenario_reaches_the_overflow(gpu, retain):
    """A guard for the tests below: the overflow region has entries, so the flush of it
    (slots past the table's capacity) is what they compare."""
    cache, _, _ = _pair(retain)
    table = cache._state.key_index_map
    seen = []
    for _, keys, _, slots in table._export_batches_masked(0, 1 << 20):
        seen += [s for s in slots.cpu().tolist() if s >= 0]
    assert max(seen) >= table.per_table_capacity_[0]
    assert len(seen) == 1024 + 50


@pytest.mark.parametrize("retain", [False, True])
def test_the_flush_leaves_the_storage_as_the_exact_flush_does(gpu, retain):
    kvt = _need("dynamicemb.key_value_table")
    cache, storage, cache_keys = _pair(retain)
    ref_cache, ref_storage, _ = _pair(retain)
    before = _contents(storage, 0, 8)

    kvt.flush_cache(cache, storage)
    exact_flush(ref_cache, ref_storage)
    torch.cuda.synchronize()

    _same_contents(storage, ref_storage)
    after = _contents(storage, 0, 8)
    # the cache's value reached the storage for a key that both hold, and a key that only
    # the storage holds kept its own
    both = next(k for k in cache_keys[0] if k % 10 < 7)
    assert torch.equal(
        after[both], _cache_value(torch.tensor([both]), 8, _device())[0].cpu()
    )
    assert not torch.equal(before[both], after[both])
    assert torch.equal(after[9000], before[9000])
    # the cache's overflow entries that the storage holds were written back too
    overflow_both = next(k for k in cache_keys[0] if k > 2000 and k % 2 == 0)
    assert torch.equal(
        after[overflow_both],
        _cache_value(torch.tensor([overflow_both]), 8, _device())[0].cpu(),
    )
    # a key that only the cache holds was not inserted
    only_cache = next(k for k in cache_keys[0] if k % 10 >= 7 and k < 1025)
    assert only_cache not in after
    if retain:
        for table in (0, 1):
            got = storage.pop_evicted_keys(table).cpu().tolist()
            expected = ref_storage.pop_evicted_keys(table).cpu().tolist()
            assert got == expected and len(got) > 0, table
    else:
        assert storage.pop_evicted_keys(0).numel() == 0


def test_a_tier_below_gets_the_keys_the_storage_no_longer_has(gpu):
    """With an external storage the exact-length path is the one that runs, as before."""
    kvt = _need("dynamicemb.key_value_table")

    class Tier:
        def __init__(self):
            self.rows = {}

        def insert(self, keys, table_ids, values, scores=None, preserve_existing=False):
            for key, table, row in zip(keys.tolist(), table_ids.tolist(), values.cpu()):
                self.rows[(table, key)] = row.clone()

    cache, storage, _ = _pair()
    ref_cache, ref_storage, _ = _pair()
    tier, ref_tier = Tier(), Tier()
    kvt.flush_cache(cache, storage, tier)
    exact_flush(ref_cache, ref_storage, ref_tier)
    torch.cuda.synchronize()
    assert sorted(tier.rows) == sorted(ref_tier.rows) and tier.rows
    assert all(torch.equal(tier.rows[k], ref_tier.rows[k]) for k in tier.rows)
    _same_contents(storage, ref_storage)


def test_a_flush_of_an_empty_cache_writes_nothing(gpu):
    kvt = _need("dynamicemb.key_value_table")
    _, storage, _ = _pair()
    empty_cache, _, _ = _pair()
    empty_cache.reset()
    before = [_contents(storage, t, 8) for t in (0, 1)]
    kvt.flush_cache(empty_cache, storage)
    torch.cuda.synchronize()
    after = [_contents(storage, t, 8) for t in (0, 1)]
    for a, b in zip(before, after):
        assert sorted(a) == sorted(b)
        assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture
def small_batches(monkeypatch):
    """128 slots a batch: a table of the cache is 8 batches of its main region and 24 of its
    overflow, 64 for the two."""
    kvt = _need("dynamicemb.key_value_table")
    monkeypatch.setattr(kvt, "FLUSH_BATCH_BYTES", 128 * 8 * 4)
    return 64


def test_a_flush_waits_for_nothing(gpu, count_syncs, small_batches):
    kvt = _need("dynamicemb.key_value_table")
    cache, storage, _ = _pair()
    count = count_syncs(lambda i: kvt.flush_cache(cache, storage))
    assert count.blocking == 0, count.describe()


def test_a_flush_of_an_empty_cache_waits_for_nothing(gpu, count_syncs, small_batches):
    kvt = _need("dynamicemb.key_value_table")
    cache, storage, _ = _pair()
    cache.reset()
    count = count_syncs(lambda i: kvt.flush_cache(cache, storage))
    assert count.blocking == 0, count.describe()


def test_a_storage_that_retains_evictions_costs_a_wait_per_ring_of_batches(
    gpu, count_syncs, small_batches
):
    """Its entries keep their arrays until their counts land, so the flush waits for the
    newest count once in ``retain_ring.slots`` batches: an explicit call may, and the
    entries do not pile up behind a host that is far ahead."""
    kvt = _need("dynamicemb.key_value_table")
    cache, storage, _ = _pair(retain=True)
    slots = storage._state.retain_ring.slots
    count = count_syncs(lambda i: kvt.flush_cache(cache, storage))
    assert count.blocking <= small_batches // slots + 1, count.describe()
    # and what is left pending is bounded by that, not by the number of batches
    assert len(storage._state.evicted_pending) <= 3 * slots


def test_the_batch_of_a_flush_stays_inside_its_byte_budget(gpu, monkeypatch):
    kvt = _need("dynamicemb.key_value_table")
    dim = 256  # 1 KiB a row
    cache, storage, _ = _pair(dim=dim)
    state = cache._state
    row_bytes = state.value_dim * state.emb_dtype.itemsize
    budget = 1 << 20
    monkeypatch.setattr(kvt, "FLUSH_BATCH_BYTES", budget)
    rows = kvt._flush_batch_rows(state)
    assert (
        rows * row_bytes <= budget and rows % 32 == 0 and rows <= state.threads_in_wave
    )
    assert rows >= budget // row_bytes - 32, "no more than a tile short of the budget"
    # a narrow row does not make a batch longer than a wave of threads
    monkeypatch.setattr(kvt, "FLUSH_BATCH_BYTES", 1 << 40)
    assert kvt._flush_batch_rows(state) == state.threads_in_wave

    monkeypatch.setattr(kvt, "FLUSH_BATCH_BYTES", budget)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    kvt.flush_cache(cache, storage)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    assert peak <= 4 * budget, "the flush held %d bytes of transients" % peak
