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

The sort of ``bucketize_keys_segmented`` keeps no count and no data-dependent shape, so a
call enqueues its kernels and returns. The census is the probe's
(``host_sync_probe.measure_callable``: the CUDA profiler, which sees raw C++ syncs, the
PyTorch sync-debug warnings and interposed ``synchronize`` calls), taken through the
``count_syncs`` fixture of this directory. Every zero is read next to a positive control
that must not be zero (the compact ``bucketize_keys``, which reads the number of its
active buckets), so a dead profiler cannot pass these tests: ``instruments_ok`` is
requested as well.
"""

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
