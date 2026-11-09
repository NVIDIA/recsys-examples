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

"""The deterministic insert reads nothing from the device.

The sort of ``bucketize_keys_segmented`` keeps no count and no data-dependent shape, and the
sorted inserts (``table_insert_sorted``, ``table_insert_and_evict_sorted``, with or without the
overflow bucket of a cache) leave the count of their evictions on the device, so a call
enqueues its kernels and returns. The census is the probe's
(``host_sync_probe.measure_callable``: the CUDA profiler, which sees raw C++ syncs, the
PyTorch sync-debug warnings and interposed ``synchronize`` calls), taken through the
``count_syncs`` fixture of this directory. Every zero is read next to a positive control
that must not be zero (the compact ``bucketize_keys``, which reads the number of its
active buckets; the waves, the code the sorted inserts replaced, which read three times a
call), so a dead profiler cannot pass these tests: ``instruments_ok`` is requested as well.
"""

import os
import sys

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)
try:
    import dynamicemb_extensions as ext
except ImportError as e:
    pytest.skip(
        "the dynamicemb extension is not built here: %s" % (e,), allow_module_level=True
    )

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_OPERATION = os.path.join(os.path.dirname(HERE), "table_operation")
if TABLE_OPERATION not in sys.path:
    sys.path.insert(0, TABLE_OPERATION)

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")
KEY_TYPES = [torch.int64, torch.uint64]
CAPACITY = 16
BUCKETS_PER_TABLE = 4
NUM_TABLES = 2


def padded_batch(key_type, n=4096, seed=0):
    """A batch of ``n`` keys of two tables with 10% reserved lanes (table id -1) among them,
    and the offsets of the two tables' buckets."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    keys = torch.randint(
        1, 10**7, (n,), dtype=torch.int64, device="cuda", generator=gen
    )
    tids = torch.randint(
        NUM_TABLES, (n,), dtype=torch.int64, device="cuda", generator=gen
    )
    pad = torch.rand(n, device="cuda", generator=gen) < 0.1
    keys = torch.where(pad, -1, keys).view(key_type)
    tids = torch.where(pad, -1, tids)
    offsets = (
        torch.arange(NUM_TABLES + 1, dtype=torch.int64, device="cuda")
        * BUCKETS_PER_TABLE
    )
    return keys, tids, offsets


@cuda_only
@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_bucketize_keys_segmented_reads_nothing(count_syncs, instruments_ok, key_type):
    keys, tids, offsets = padded_batch(key_type)
    num_buckets = NUM_TABLES * BUCKETS_PER_TABLE

    def step(i):
        ext.bucketize_keys_segmented(keys, tids, offsets, num_buckets, CAPACITY)

    result = count_syncs(step, iters=4, warmup=3)
    assert result.blocking == 0, result.describe()


@cuda_only
def test_the_compact_bucketize_keys_blocks_as_a_control(count_syncs, instruments_ok):
    keys, tids, offsets = padded_batch(torch.int64)
    # bucketize_keys reads every table id, the padding's -1 included: give it real keys only.
    live = keys != -1
    keys, tids = keys[live], tids[live]
    num_buckets = NUM_TABLES * BUCKETS_PER_TABLE

    def step(i):
        ext.bucketize_keys(keys, tids, offsets, num_buckets, CAPACITY)

    result = count_syncs(step, iters=4, warmup=3)
    assert result.blocking >= 1, result.describe()


# ---------------------------------------------------------------------------
# the sorted inserts
# ---------------------------------------------------------------------------
OPS = ["insert", "insert_and_evict", "cache"]


def sorted_insert_step(op, key_type, policy):
    """A table, a padded batch of many keys per bucket, and the call of ``op`` that enqueues the
    sorted insert (and the lookup that gives the indices) and reads nothing."""
    from test_sorted_insert import (
        make_batch,
        make_table,
        new_insert,
        new_insert_and_evict,
    )

    table = make_table(key_type, policy, [64, 64], CAPACITY, overflow=op == "cache")
    gen = torch.Generator().manual_seed(4)
    keys, tids, scores = make_batch(gen, 600, NUM_TABLES, key_type, policy, pad=0.1)
    if op == "insert":

        def step(i):
            new_insert(table, keys, tids, scores, policy)

    else:

        def step(i):
            new_insert_and_evict(table, keys, tids, scores, policy)

    return table, step, (keys, tids, scores)


@cuda_only
@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("op", OPS)
def test_the_sorted_insert_reads_nothing(count_syncs, instruments_ok, key_type, op):
    if not getattr(ext, "sorted_insert", False):
        pytest.skip("this extension has no sorted insert: rebuild it")
    policy = ext.ScorePolicy.ASSIGN
    _, step, _ = sorted_insert_step(op, key_type, policy)
    result = count_syncs(step, iters=4, warmup=3)
    assert result.blocking == 0, result.describe()


@cuda_only
@pytest.mark.parametrize("op", OPS)
def test_the_waves_block_as_a_control(count_syncs, instruments_ok, op):
    """The code the sorted inserts replaced reads the device at least twice a call (the pre-filter
    of the reserved keys, the wave count, the read inside bucketize_keys, and the host count of
    the evictions)."""
    if not getattr(ext, "sorted_insert", False):
        pytest.skip("this extension has no sorted insert: rebuild it")
    import wave_oracle as oracle

    policy = ext.ScorePolicy.ASSIGN
    table, _, (keys, tids, scores) = sorted_insert_step(op, torch.int64, policy)
    if op == "insert":

        def step(i):
            oracle.insert(table, keys, tids, scores, policy)

    elif op == "insert_and_evict":

        def step(i):
            oracle.insert_and_evict(table, keys, tids, scores, policy)

    else:

        def step(i):
            oracle.insert_and_evict_with_overflow(table, keys, tids, scores, policy)

    result = count_syncs(step, iters=3, warmup=2)
    assert result.blocking >= 2, result.describe()


@cuda_only
@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("op", OPS)
def test_the_sorted_insert_survives_no_sync(no_sync, key_type, op):
    """PyTorch's sync-debug mode, with the guards of this directory, turns any sync the call
    issues through ATen (``.item()``, ``nonzero``, a pageable copy) into an error."""
    if not getattr(ext, "sorted_insert", False):
        pytest.skip("this extension has no sorted insert: rebuild it")
    _, step, _ = sorted_insert_step(op, key_type, ext.ScorePolicy.ASSIGN)
    step(-1)  # builds nothing lazily inside the guarded block
    torch.cuda.synchronize()
    with no_sync():
        step(0)
    torch.cuda.synchronize()
