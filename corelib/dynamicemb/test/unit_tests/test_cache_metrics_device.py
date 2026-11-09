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

"""The cache's metrics are counted on the device and read when somebody asks.

``DynamicEmbCache.lookup`` and ``insert_and_evict`` fill a device buffer without
waiting for anything, with the count of evictions either read on the host (the default)
or left on the device (``async_count=True``); ``cache_metrics`` is where the host waits
for it, once, and returns a CPU snapshot. The numbers mean what they always did:
``[0]`` unique keys of the last lookup, ``[1]`` how many were found, ``[2]`` keys of the
last insert, ``[3]`` how many it evicted. A reserved key (the inert padding of a batch)
is not a key.
"""

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy
from dynamicemb.dynamicemb_config import DynamicEmbEvictStrategy, DynamicEmbTableOptions
from dynamicemb.key_value_table import DynamicEmbCache
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer

CAPACITY = 128  # one bucket: a batch of more keys evicts


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


def _cache():
    """A cache as the module builds one: it ranks by timestamp only."""
    options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=8,
            max_capacity=CAPACITY,
            init_capacity=CAPACITY,
            bucket_capacity=CAPACITY,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            evict_strategy=DynamicEmbEvictStrategy.LRU,
        )
    ]
    cache = DynamicEmbCache(options, SGDDynamicEmbeddingOptimizer(OptimizerArgs()))
    cache.set_record_cache_metrics(True)
    return cache


def _keys(first, count, device, pad=0):
    """``count`` unique keys of table 0 from ``first`` and ``pad`` inert entries."""
    keys = torch.arange(first, first + count, dtype=torch.int64, device=device)
    tids = torch.zeros_like(keys)
    if pad:
        keys = torch.cat(
            [keys, torch.full((pad,), -1, dtype=torch.int64, device=device)]
        )
        tids = torch.cat(
            [tids, torch.full((pad,), -1, dtype=torch.int64, device=device)]
        )
    return keys, tids


def test_metrics_exist_only_while_recording(device):
    cache = _cache()
    metrics = cache.cache_metrics
    assert metrics.device.type == "cpu" and metrics.dtype == torch.int64
    assert metrics.tolist() == [0] * 10
    cache.set_record_cache_metrics(False)
    assert cache.cache_metrics is None
    keys, tids = _keys(1, 8, device)
    cache.lookup(keys, tids)  # not recorded
    cache.set_record_cache_metrics(True)
    assert cache.cache_metrics.tolist() == [0] * 10


@pytest.mark.parametrize("async_count", [False, True])
@pytest.mark.parametrize("pad", [0, 16])
def test_the_device_metrics_are_the_counts_of_the_calls(device, async_count, pad):
    cache = _cache()
    keys, tids = _keys(1, 64, device, pad)

    cache.lookup(keys, tids)
    assert cache.cache_metrics[:2].tolist() == [64, 0]  # nothing is cached yet

    result = cache.insert_and_evict(keys, tids, async_count=async_count)
    evicted = int(result[1])
    assert evicted == 0 and cache.cache_metrics[2:4].tolist() == [64, evicted]

    cache.lookup(keys, tids)
    assert cache.cache_metrics[:2].tolist() == [64, 64]  # all of them, no padding

    # more keys than the cache holds: the evictions are counted as the call counts them
    keys, tids = _keys(1000, 200, device, pad)
    cache.lookup(keys, tids)
    assert cache.cache_metrics[:2].tolist() == [200, 0]
    result = cache.insert_and_evict(keys, tids, async_count=async_count)
    evicted = int(result[1])
    assert evicted > 0
    assert cache.cache_metrics[2:4].tolist() == [200, evicted]
    assert cache.cache_metrics[4:].tolist() == [0] * 6


def test_a_returned_snapshot_does_not_change(device):
    cache = _cache()
    cache.lookup(*_keys(1, 8, device))
    first = cache.cache_metrics
    cache.lookup(*_keys(1, 16, device))
    second = cache.cache_metrics
    assert first[0].item() == 8 and second[0].item() == 16


def test_a_reader_on_another_stream_than_the_lookup_gets_its_numbers(device):
    """The lookup runs on a side stream that is still busy when the default stream
    reads the metrics: the read is ordered after the lookup, not before it."""
    cache = _cache()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        keys, tids = _keys(1, 32, device)
        torch.cuda._sleep(5 * 10**8)
        cache.lookup(keys, tids)
        cache.insert_and_evict(keys, tids, async_count=True)
    metrics = cache.cache_metrics
    assert metrics[:4].tolist() == [32, 0, 32, 0]
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "lookup_on, insert_on, read_on",
    [
        ("default", "default", "c"),  # the default stream's handle is 0
        ("a", "b", "default"),  # a stream of its own for each of the two writers
        ("default", "b", "c"),
        ("a", "default", "c"),
    ],
)
def test_a_reader_waits_for_the_stream_of_each_writer(
    device, lookup_on, insert_on, read_on
):
    """Every stream is busy when its work starts, so a reader that is not ordered
    after both the lookup and the insert reads a buffer they have not filled yet. The
    two work on different keys: the numbers do not depend on which of them runs first.
    """
    cache = _cache()
    streams = {
        "default": torch.cuda.default_stream(),
        "a": torch.cuda.Stream(),
        "b": torch.cuda.Stream(),
        "c": torch.cuda.Stream(),
    }
    lookup_keys = _keys(1, 32, device)
    insert_keys = _keys(1000, 16, device)
    torch.cuda.synchronize()
    with torch.cuda.stream(streams[lookup_on]):
        torch.cuda._sleep(5 * 10**8)
        cache.lookup(*lookup_keys)
    with torch.cuda.stream(streams[insert_on]):
        torch.cuda._sleep(5 * 10**8)
        cache.insert_and_evict(*insert_keys, async_count=True)
    with torch.cuda.stream(streams[read_on]):
        metrics = cache.cache_metrics
    assert metrics[:4].tolist() == [32, 0, 16, 0]
    torch.cuda.synchronize()
