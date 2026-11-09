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

"""``eval_lookup`` without host counts.

The eval initializer is a constant, so a key found nowhere takes it; a key the cache
has takes the cache's row, one only the storage has the storage's. The masked eval
looks every key up and writes each lane at most once; it is checked against what the
keys are known to be, and against the exact-length functions it replaced
(``legacy_oracle.legacy_eval_lookup_*``), for a storage, a hybrid storage and a cache in
front of a storage, over a batch with duplicates and keys found nowhere, and over the same
batch padded with lanes that must not take part (the reserved key, table id -1).
A storage that has not declared the padded contract is served through its own ``find``.
"""
import os
import sys

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy
from dynamicemb import key_value_table as kvt
from dynamicemb import sync_free as sf
from dynamicemb.dynamicemb_config import (
    DynamicEmbEvictStrategy,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.key_value_table import (
    DynamicEmbCache,
    DynamicEmbStorage,
    HybridStorage,
    _find_keys,
    _insert_key_values,
    eval_lookup,
    store_to_flat,
)
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legacy_oracle  # noqa: E402

DIM = 8
BUCKET = 128
CACHE_BUCKET = 1024
CONSTANT = 0.25
# Storage holds 1..40 (row value = the key), the cache holds 21..60 (row value = the
# key + 0.5): 21..40 are in both, and the cache's row wins.
STORAGE_KEYS = (1, 40)
CACHE_KEYS = (21, 60)
# duplicates, keys of each kind, keys found nowhere
BATCH = [5, 5, 25, 45, 1001, 1001, 30, 55, 7, 1002, 25, 60, 61, 20, 1]

CONFIGS = ["storage", "hybrid", "cache_storage", "cache_host_storage"]


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


def _options(capacity, bucket, host):
    return DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        dim=DIM,
        max_capacity=capacity,
        init_capacity=capacity,
        bucket_capacity=bucket,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=0 if host else 1024**3,
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        evict_strategy=DynamicEmbEvictStrategy.LRU,
    )


def _optimizer():
    return SGDDynamicEmbeddingOptimizer(OptimizerArgs())


def _initializer(device):
    return MultiTableInitializer.create(
        [
            DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT, value=CONSTANT
            )
        ],
        device,
    )


def _keys(first, last, device):
    return torch.arange(first, last + 1, dtype=torch.int64, device=device)


def _rows(keys, offset=0.0):
    return (keys.to(torch.float32) + offset).unsqueeze(1).repeat(1, DIM)


def _put_cache(cache, keys, rows):
    tids = torch.zeros_like(keys)
    slots = cache.insert_and_evict(keys, tids)[0]
    store_to_flat(cache._state, slots, tids, rows)


def _build(config, device):
    """(storage, cache) of a configuration, filled as described at the top."""
    keys = _keys(*STORAGE_KEYS, device)
    tids = torch.zeros_like(keys)
    cache = None
    if config == "hybrid":
        storage = HybridStorage(
            [_options(BUCKET, BUCKET, host=False)],
            [_options(8 * BUCKET, BUCKET, host=True)],
            _optimizer(),
        )
        # 1..20 in the HBM tier, 21..40 in the host tier
        _insert_key_values(storage._hbm, keys[:20], tids[:20], _rows(keys[:20]))
        _insert_key_values(storage._host, keys[20:], tids[20:], _rows(keys[20:]))
        return storage, cache
    storage = DynamicEmbStorage(
        [_options(8 * BUCKET, BUCKET, host=config == "cache_host_storage")],
        _optimizer(),
        use_ref_counter=False,
    )
    storage.insert(keys, tids, _rows(keys))
    if config != "storage":
        cache = DynamicEmbCache(
            [_options(2 * CACHE_BUCKET, CACHE_BUCKET, host=False)], _optimizer()
        )
        cached = _keys(*CACHE_KEYS, device)
        _put_cache(cache, cached, _rows(cached, 0.5))
    return storage, cache


def _expected(keys, with_cache, device):
    def value(k):
        if with_cache and CACHE_KEYS[0] <= k <= CACHE_KEYS[1]:
            return k + 0.5
        return float(k) if STORAGE_KEYS[0] <= k <= STORAGE_KEYS[1] else CONSTANT

    column = torch.tensor([value(k) for k in keys], dtype=torch.float32, device=device)
    return column.unsqueeze(1).repeat(1, DIM)


def _batch(device):
    keys = torch.tensor(BATCH, dtype=torch.int64, device=device)
    return keys, torch.zeros_like(keys)


def _legacy(storage, cache, keys, tids, initializer):
    if cache is None:
        return legacy_oracle.legacy_eval_lookup_storage(
            storage, keys, tids, initializer
        )
    return legacy_oracle.legacy_eval_lookup_cached(
        cache, storage, keys, tids, initializer
    )


def _scores(state, keys):
    """The score of every key the state holds, 0 for the others (const lookup)."""
    r = _find_keys(
        state, keys, torch.zeros_like(keys), const_lookup=True, compact=False
    )
    return torch.where(r[5], r[6], torch.zeros_like(r[6]))


@pytest.mark.parametrize("config", CONFIGS)
def test_eval_lookup_equals_what_the_keys_are_and_the_exact_function(device, config):
    storage, cache = _build(config, device)
    initializer = _initializer(device)
    keys, tids = _batch(device)

    out = eval_lookup(storage, keys, tids, initializer, cache=cache)
    old = _legacy(storage, cache, keys, tids, initializer)

    expected = _expected(BATCH, cache is not None, device)
    assert out.shape == (len(BATCH), DIM)
    assert torch.equal(out, expected)
    assert torch.equal(old, expected)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("config", CONFIGS)
def test_padded_batch(device, monkeypatch, config, fallback):
    """The lanes of a padded batch that are real come out as without padding; the
    padding is neither looked up nor written (its rows hold garbage). ``fallback``
    runs ``sf.initialize_masked`` without the initializer's skip-negative-index
    capability (the scratch-and-``where`` form)."""
    if fallback:
        monkeypatch.setattr(sf, "INIT_SKIPS_NEGATIVE", False)
    storage, cache = _build(config, device)
    initializer = _initializer(device)
    keys, tids = _batch(device)
    n, pad = keys.numel(), 13
    padded_keys = torch.cat([keys, torch.full((pad,), -1, device=device)])
    padded_tids = torch.cat([tids, torch.full((pad,), -1, device=device)])
    valid = torch.cat(
        [torch.ones(n, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)]
    ).to(device)

    out = eval_lookup(
        storage, padded_keys, padded_tids, initializer, cache=cache, valid=valid
    )

    assert out.shape == (n + pad, DIM)
    assert torch.equal(out[:n], _expected(BATCH, cache is not None, device))


@pytest.mark.parametrize("config", ["storage", "cache_storage"])
def test_eval_moves_no_score(device, config):
    """Eval is a const lookup: the timestamps of the storage's keys (and the cache's)
    do not move, whatever was looked up."""
    storage, cache = _build(config, device)
    keys = _keys(*STORAGE_KEYS, device)
    batch, tids = _batch(device)
    before = _scores(storage._state, keys)
    before_cache = None if cache is None else _scores(cache._state, keys)

    eval_lookup(storage, batch, tids, _initializer(device), cache=cache)

    assert torch.equal(before, _scores(storage._state, keys))
    if cache is not None:
        assert torch.equal(before_cache, _scores(cache._state, keys))


def test_empty_batch(device):
    storage, cache = _build("cache_storage", device)
    keys = torch.empty(0, dtype=torch.int64, device=device)
    out = eval_lookup(
        storage, keys, torch.zeros_like(keys), _initializer(device), cache
    )
    assert out.shape == (0, DIM)


def _count_compactions(monkeypatch):
    calls = []
    real = kvt.flagged_compact

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(kvt, "flagged_compact", counting)
    return calls


@pytest.mark.parametrize("config", CONFIGS)
def test_masked_eval_reads_nothing_back(device, monkeypatch, config):
    """No compaction, and nothing PyTorch reports as a sync (``set_sync_debug_mode``
    does not see one made inside the extension: the census of ``no_host_sync`` does)."""
    storage, cache = _build(config, device)
    initializer = _initializer(device)
    keys, tids = _batch(device)
    eval_lookup(storage, keys, tids, initializer, cache=cache)  # warm up
    torch.cuda.synchronize()
    calls = _count_compactions(monkeypatch)

    torch.cuda.set_sync_debug_mode("error")
    try:
        eval_lookup(storage, keys, tids, initializer, cache=cache)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert calls == []


class _CountingStorage(DynamicEmbStorage):
    """A storage whose ``find`` a subclass owns: eval must keep calling it."""

    finds = 0

    def find(self, *args, **kwargs):
        type(self).finds += 1
        return super().find(*args, **kwargs)


@pytest.mark.parametrize("switch", ["env", "subclass"])
def test_the_environment_selects_nothing_and_a_subclass_is_served_through_its_find(
    device, monkeypatch, switch
):
    """No environment variable selects a path: the old kill switch has no effect and the in-tree
    ``find`` is never called by eval. A storage that is not exactly an in-tree one (a subclass
    that overrides ``find`` without declaring the padded contract itself) is served through that
    ``find``, once, with the live keys; the answer is the same either way."""
    keys, tids = _batch(device)
    initializer = _initializer(device)
    _CountingStorage.finds = 0
    if switch == "env":
        monkeypatch.setenv("DYNAMICEMB_SYNC_FREE", "0")
        storage, cache = _build("cache_storage", device)
        spy_calls = []
        real_find = DynamicEmbStorage.find

        def spy(self, *args, **kwargs):
            spy_calls.append(1)
            return real_find(self, *args, **kwargs)

        monkeypatch.setattr(DynamicEmbStorage, "find", spy)
        out = eval_lookup(storage, keys, tids, initializer, cache=cache)
        assert (
            spy_calls == []
        ), "the variable is not read: the padded path never calls find"
    else:
        storage = _CountingStorage(
            [_options(8 * BUCKET, BUCKET, host=False)],
            _optimizer(),
            use_ref_counter=False,
        )
        stored = _keys(*STORAGE_KEYS, device)
        storage.insert(stored, torch.zeros_like(stored), _rows(stored))
        cache = None
        out = eval_lookup(storage, keys, tids, initializer)
        assert _CountingStorage.finds == 1
    assert torch.equal(out, _expected(BATCH, cache is not None, device))


class _DeclaringStorage(_CountingStorage):
    """The same subclass, that declares the padded contract itself: it is called natively."""

    accepts_empty_keys = True


def test_a_subclass_that_declares_the_contract_in_its_own_body_is_called_natively(
    device,
):
    keys, tids = _batch(device)
    _CountingStorage.finds = 0
    storage = _DeclaringStorage(
        [_options(8 * BUCKET, BUCKET, host=False)], _optimizer(), use_ref_counter=False
    )
    stored = _keys(*STORAGE_KEYS, device)
    storage.insert(stored, torch.zeros_like(stored), _rows(stored))

    out = eval_lookup(storage, keys, tids, _initializer(device))

    assert _CountingStorage.finds == 0, "its lookup_masked is used, not find"
    assert torch.equal(out, _expected(BATCH, False, device))
