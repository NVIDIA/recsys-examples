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

"""The cache's eviction guard, as the backing's insert sees it.

The backing chooses which keys to evict by its own score. A key the cache is using
-- its node holds a reference -- must not be taken, whatever its score. These tests
fill a one-bucket table so every insert competes for the same slots, cache some of
its keys, and check which keys survive an insert that has to evict.

Both eviction paths are covered: the single-score scan built into the extension
and the LruLfu one in the separately built cubin.
"""

import pytest
import torch
from dynamicemb.scored_hashtable import ScoreArg, ScoreSpec, get_scored_table
from dynamicemb_extensions import EmbeddingCacheCore, InsertResult, ScorePolicy

BUCKET = 128


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


def _table(kind):
    if kind == "lru_lfu":
        spec = ScoreSpec(name="frequency", policy=ScorePolicy.LRU_LFU, is_reduction=True)
    else:
        spec = ScoreSpec(name="timestamp", policy=ScorePolicy.GLOBAL_TIMER)
    return get_scored_table(
        capacity=[BUCKET], bucket_capacity=BUCKET, key_type=torch.int64, score_specs=[spec]
    )


def _score_arg(kind, n, device):
    if kind == "lru_lfu":
        return ScoreArg(
            name="frequency",
            value=torch.ones(n, dtype=torch.uint64, device=device),
            policy=ScorePolicy.LRU_LFU,
        )
    return ScoreArg(name="timestamp", value=None, policy=ScorePolicy.GLOBAL_TIMER)


def _insert(table, kind, keys, device, guard=None, collect=False):
    n = keys.numel()
    tids = torch.zeros(n, dtype=torch.int64, device=device)
    results = torch.empty(n, dtype=table.result_type, device=device).fill_(
        InsertResult.INIT.value
    )
    return table.insert(
        keys,
        tids,
        _score_arg(kind, n, device),
        results,
        collect_evicted=collect,
        guard=guard,
    )


def _present(table, keys, device):
    tids = torch.zeros(keys.numel(), dtype=torch.int64, device=device)
    _, founds, _ = table.lookup(
        keys,
        tids,
        ScoreArg(name=table.score_names_[0], value=None, policy=ScorePolicy.CONST),
    )
    return founds


def _cache_keys(core, keys, device, hold):
    """Bind `keys` in the cache's only table and publish them. The first `hold` of
    them keep the reference a step in flight would; the rest are released, idle."""
    n = keys.numel()
    nodes = core.lookup_pin(keys, [0, n])
    _, _, new_nodes = core.allocate(
        keys, torch.arange(n, dtype=torch.int64, device=device), [0, n], nodes
    )
    assert bool((new_nodes >= 0).all())
    core.publish(new_nodes, [0, n])
    if n > hold:
        idle = new_nodes[hold:].contiguous()
        core.release(idle, [0, idle.numel()])
    torch.cuda.synchronize()


@pytest.mark.parametrize("kind", ["timestamp", "lru_lfu"])
@pytest.mark.parametrize("collect", [False, True])
def test_referenced_keys_are_not_evicted(device, kind, collect):
    table = _table(kind)
    core = EmbeddingCacheCore([4 * BUCKET])

    stored = torch.arange(1, 101, dtype=torch.int64, device=device)  # 100 of 128 slots
    _insert(table, kind, stored, device)

    # 30 of them are held by a step in flight; 30 more are cached but idle.
    cached = torch.arange(1, 61, dtype=torch.int64, device=device)
    _cache_keys(core, cached, device, hold=30)
    held = torch.arange(1, 31, dtype=torch.int64, device=device)

    # 100 new keys into a bucket with 28 free slots: 72 evictions.
    new = torch.arange(1000, 1100, dtype=torch.int64, device=device)
    out = _insert(table, kind, new, device, guard=core.guard(), collect=collect)

    assert bool(_present(table, held, device).all()), "a referenced key was evicted"
    if collect:
        _, num_evicted, evicted_keys, _, _ = out
        evicted = evicted_keys[: int(num_evicted.item())]
        assert not bool(torch.isin(evicted, held).any())
        assert num_evicted.item() > 0, "the insert should have had to evict"


@pytest.mark.parametrize("kind", ["timestamp", "lru_lfu"])
def test_without_the_guard_the_same_keys_go(device, kind):
    """The control: the keys the guard protects are ones the score alone would evict,
    so the test above is not passing by accident."""
    table = _table(kind)
    stored = torch.arange(1, 101, dtype=torch.int64, device=device)
    _insert(table, kind, stored, device)
    held = torch.arange(1, 31, dtype=torch.int64, device=device)

    new = torch.arange(1000, 1100, dtype=torch.int64, device=device)
    _insert(table, kind, new, device)  # no guard

    assert not bool(_present(table, held, device).all()), (
        "with no guard none of the keys was evicted; the scenario does not exercise "
        "the eviction the guard is for"
    )


@pytest.mark.parametrize("kind", ["timestamp", "lru_lfu"])
def test_a_bucket_of_referenced_keys_refuses_the_insert(device, kind):
    table = _table(kind)
    core = EmbeddingCacheCore([4 * BUCKET])

    stored = torch.arange(1, BUCKET + 1, dtype=torch.int64, device=device)  # full
    _insert(table, kind, stored, device)
    _cache_keys(core, stored, device, hold=BUCKET)  # every one referenced

    extra = torch.tensor([5000, 5001], dtype=torch.int64, device=device)
    indices = _insert(table, kind, extra, device, guard=core.guard())

    assert bool((indices < 0).all()), "nothing could be evicted, so nothing was inserted"
    assert bool(_present(table, stored, device).all()), "a referenced key was evicted"

    # Give the references back and the same insert goes through.
    nodes = core.find_nodes(stored, torch.zeros_like(stored))
    core.release(nodes.contiguous(), [0, nodes.numel()])
    torch.cuda.synchronize()
    indices = _insert(table, kind, extra, device, guard=core.guard())
    assert bool((indices >= 0).all())


@pytest.mark.parametrize("kind", ["timestamp", "lru_lfu"])
def test_a_key_that_is_not_cached_is_evicted_as_before(device, kind):
    table = _table(kind)
    core = EmbeddingCacheCore([4 * BUCKET])  # empty: nothing to protect

    stored = torch.arange(1, BUCKET + 1, dtype=torch.int64, device=device)
    _insert(table, kind, stored, device)
    new = torch.arange(2000, 2000 + 20, dtype=torch.int64, device=device)
    indices = _insert(table, kind, new, device, guard=core.guard())
    assert bool((indices >= 0).all()), "an empty cache must not hold up an eviction"
