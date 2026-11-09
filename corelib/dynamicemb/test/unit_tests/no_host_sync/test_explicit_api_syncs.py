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

"""The explicit APIs of DynamicEmb return host data, so they wait for the device. What
they must not do is wait several times for one piece of data.

Each test counts the blocking events of one call with the probe's census
(``count_syncs``: ``cudaStreamSynchronize``, ``cudaEventSynchronize``, synchronous
copies, pageable device-to-host copies) and holds it to the floor of that API. This
file has the two cheap ones, the cache's metrics getter and the result of an erase;
the others have a file of their own, next to it:

* ``test_pop_one_pass.py``      pop_evicted_keys / pop_erased_keys
* ``test_flush_masked.py``      flush
* ``test_dump_pipelined.py``    dump
* ``test_load_staged.py``       load
* ``test_replay_two_phase.py``  replay_increment, erase_keys

Whatever else a test needs it builds before the measured calls: a CUDA tensor made from a
host list is itself a blocking copy.
"""

import importlib

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)


def _need(name):
    """The module, or a skip: a tree without the extension has nothing to measure."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip("%s is not importable here: %s" % (name, e))


SLEEP_CYCLES = 5 * 10**8


def _device():
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def eraser():
    """A worker thread that is shut down after the test, whether it passed or not."""
    async_erase = _need("dynamicemb.async_erase")
    worker = async_erase.AsyncEraser()
    yield worker
    worker.shutdown()


def test_a_handles_result_is_one_wait(gpu, count_syncs, eraser):
    """The counts ride the erase stream into pinned memory ahead of the event that
    ``wait`` waits for: ``result()`` waits for the device once, not once for the erase
    and once more for a copy of the counts."""
    device = _device()
    stream = torch.cuda.Stream()
    add = torch.tensor([3, 2], dtype=torch.int64, device=device)

    def one_erase_and_its_result(i):
        handle = eraser.submit(lambda counts: counts.add_(add), stream)
        assert handle.result() == (3, 2)

    count = count_syncs(one_erase_and_its_result)
    assert count.blocking <= 1.0, count.describe()


def test_a_group_of_handles_waits_once_per_handle(gpu, count_syncs, eraser):
    async_erase = _need("dynamicemb.async_erase")
    device = _device()
    stream = torch.cuda.Stream()
    add = torch.tensor([3, 2], dtype=torch.int64, device=device)

    def three_erases_and_their_sum(i):
        group = async_erase.EraseHandleGroup(
            [eraser.submit(lambda counts: counts.add_(add), stream) for _ in range(3)]
        )
        assert group.result() == (9, 6)

    count = count_syncs(three_erases_and_their_sum)
    assert count.blocking <= 3.0, count.describe()


def test_the_counts_are_what_the_kernels_made_not_what_the_launch_left(gpu, eraser):
    """The copy of the counts is ordered behind the work on the erase stream: with a
    long kernel in front of the update, the host must still read the updated numbers."""
    device = _device()
    stream = torch.cuda.Stream()
    add = torch.tensor([7, 5], dtype=torch.int64, device=device)

    def slow_work(counts):
        torch.cuda._sleep(SLEEP_CYCLES)
        counts.add_(add)

    handle = eraser.submit(slow_work, stream)
    assert handle.result() == (7, 5)
    assert handle.result() == (7, 5), "again: from what the handle kept"


def _cache(device):
    """A cache as the module builds one, recording its metrics."""
    dynamicemb = _need("dynamicemb")
    config = _need("dynamicemb.dynamicemb_config")
    optimizer = _need("dynamicemb.optimizer")
    kvt = _need("dynamicemb.key_value_table")
    options = [
        config.DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=8,
            max_capacity=1024,
            init_capacity=1024,
            bucket_capacity=128,
            safe_check_mode=dynamicemb.DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=dynamicemb.DynamicEmbScoreStrategy.TIMESTAMP,
            evict_strategy=config.DynamicEmbEvictStrategy.LRU,
        )
    ]
    cache = kvt.DynamicEmbCache(
        options, optimizer.SGDDynamicEmbeddingOptimizer(optimizer.OptimizerArgs())
    )
    cache.set_record_cache_metrics(True)
    return cache


def test_reading_the_cache_metrics_is_one_wait(gpu, count_syncs):
    """The lookups only write device counters; the getter is where the host waits, once,
    for the whole batch of them."""
    device = _device()
    cache = _cache(device)
    keys = torch.arange(1, 65, dtype=torch.int64, device=device)
    tids = torch.zeros_like(keys)
    cache.lookup(keys, tids)
    cache.insert_and_evict(keys, tids, async_count=True)
    torch.cuda.synchronize()

    def read(i):
        metrics = cache.cache_metrics
        assert metrics is not None and metrics.numel() == 10

    count = count_syncs(read)
    assert count.blocking <= 1.0, count.describe()
