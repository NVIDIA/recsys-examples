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

"""``bucketize_keys_segmented``: the sort of the deterministic insert.

It sorts keys by (home bucket, key) like ``bucketize_keys`` does, with the same radix
sort, but it hands back the home bucket of every sorted position instead of compacting
the keys into the active buckets, so it reads nothing back. A key that takes no part in
an insert (a reserved key, a key of a table that does not exist or has no buckets) gets
the segment ``num_buckets`` and sorts after all the others; its table id is never read
when the key is reserved.

The tests compare the op with a host reference of the arithmetic of the table kernels:
the home bucket is ``bkt_begin[t] + ((fmix64(key) & INT64_MAX) % (n_buckets[t] * cap)) //
cap``, and the order is (segment, key) with the key as a signed number for int64 keys and
an unsigned one for uint64 keys; equal (segment, key) pairs keep their input order, since
the radix sort is stable.
"""

import random

import pytest
import torch
from dynamicemb_extensions import bucketize_keys, bucketize_keys_segmented

KEY_TYPES = [torch.int64, torch.uint64]
MASK64 = (1 << 64) - 1
INT64_MAX = (1 << 63) - 1
RESERVED = (-1, -2, -3, -4)  # the int64 values of the four reserved bit patterns

# Buckets per table. Zero-capacity tables first, in the middle and last, and none at all.
GEOMETRIES = [
    pytest.param([3], id="1table"),
    pytest.param([1, 2], id="2tables"),
    pytest.param([2, 0, 3], id="zero-middle"),
    pytest.param([0, 3], id="zero-first"),
    pytest.param([3, 0], id="zero-last"),
    pytest.param([1, 2, 3, 4], id="4tables"),
    pytest.param([0, 0], id="no-buckets"),
]


def fmix64(x: int) -> int:
    """The host twin of ``murmur3_fmix64`` (src/murmur_hash.cuh)."""
    x &= MASK64
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & MASK64
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & MASK64
    x ^= x >> 33
    return x


class Geometry:
    """Tables of ``buckets[t]`` buckets of ``capacity`` slots each."""

    def __init__(self, buckets, capacity):
        self.buckets = list(buckets)
        self.capacity = capacity
        self.offsets = [0]
        for b in self.buckets:
            self.offsets.append(self.offsets[-1] + b)
        self.num_buckets = self.offsets[-1]

    def offsets_tensor(self, device):
        return torch.tensor(self.offsets, dtype=torch.int64, device=device)

    def segment(self, key: int, table_id: int) -> int:
        """The segment ``key`` (an int64 value) of table ``table_id`` sorts into."""
        if key in RESERVED:
            return self.num_buckets
        if table_id < 0 or table_id >= len(self.buckets) or self.buckets[table_id] == 0:
            return self.num_buckets
        table_cap = self.buckets[table_id] * self.capacity
        home = (fmix64(key) & INT64_MAX) % table_cap
        return self.offsets[table_id] + home // self.capacity


def as_keys(values, key_type, device):
    """A key tensor of ``key_type`` holding the int64 bit patterns ``values``."""
    return torch.tensor(values, dtype=torch.int64, device=device).view(key_type)


def signed(keys):
    """The keys as Python ints, uint64 ones as the int64 value of their bit pattern."""
    return (keys.view(torch.int64) if keys.dtype == torch.uint64 else keys).tolist()


def order_key(value: int, key_type) -> int:
    return value if key_type == torch.int64 else value & MASK64


def make_batch(rng, n, num_tables, duplicates):
    """``n`` (key, table id) pairs: real keys (small and huge, negative ones that are not
    reserved), the padding of an N-aligned batch (a reserved key whose table id is anything,
    -1 or garbage), and keys of a table that does not exist."""
    keys, tids = [], []
    for _ in range(n):
        r = rng.random()
        if r < 0.10:
            keys.append(rng.choice([-1, -1, -1, -2, -3, -4]))
            tids.append(rng.choice([-1, -1, 0, 7, 1_000_000]))
        elif r < 0.13:
            keys.append(rng.randrange(1, 1000))
            tids.append(rng.choice([-1, num_tables, num_tables + 2]))
        else:
            if duplicates:
                keys.append(rng.randrange(1, 40))
            else:
                kind = rng.random()
                if kind < 0.7:
                    keys.append(rng.randrange(0, 10**7))
                elif kind < 0.85:
                    keys.append(rng.randrange(2**62, 2**63 - 1))
                else:
                    keys.append(-rng.randrange(5, 2**62))
            tids.append(rng.randrange(num_tables))
    return keys, tids


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("bucket_capacity", [16, 128])
@pytest.mark.parametrize("batch_size", [1, 33, 1000, 5000])
@pytest.mark.parametrize("buckets", GEOMETRIES)
@pytest.mark.parametrize("duplicates", [False, True], ids=["unique", "duplicates"])
def test_segments_and_order_match_the_host_reference(
    device, key_type, bucket_capacity, batch_size, buckets, duplicates
):
    rng = random.Random(1000 * batch_size + bucket_capacity + len(buckets))
    geometry = Geometry(buckets, bucket_capacity)
    values, tids = make_batch(rng, batch_size, len(buckets), duplicates)
    segment = [geometry.segment(v, t) for v, t in zip(values, tids)]
    order = sorted(
        range(batch_size),
        key=lambda i: (segment[i], order_key(values[i], key_type), i),
    )

    sorted_keys, inverse, segments = bucketize_keys_segmented(
        as_keys(values, key_type, device),
        torch.tensor(tids, dtype=torch.int64, device=device),
        geometry.offsets_tensor(device),
        geometry.num_buckets,
        bucket_capacity,
    )

    assert sorted_keys.dtype == key_type
    assert inverse.dtype == segments.dtype == torch.int64
    assert sorted_keys.numel() == inverse.numel() == segments.numel() == batch_size
    assert inverse.tolist() == order
    assert signed(sorted_keys) == [values[i] for i in order]
    assert segments.tolist() == [segment[i] for i in order]


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("buckets", GEOMETRIES)
def test_the_keys_that_take_part_sort_as_in_bucketize_keys(device, key_type, buckets):
    """What the sorted insert applies to a bucket is what the waves applied: the keys of a
    bucket in the order of the compact ``bucketize_keys``. That op is given only the keys
    that take part: it reads the table id of every key it sorts."""
    rng = random.Random(7 + len(buckets))
    capacity, n = 16, 3000
    geometry = Geometry(buckets, capacity)
    values, tids = make_batch(rng, n, len(buckets), duplicates=False)
    segment = [geometry.segment(v, t) for v, t in zip(values, tids)]
    live = [i for i in range(n) if segment[i] != geometry.num_buckets]
    if not live:
        pytest.skip("no key takes part in this geometry")

    offsets = geometry.offsets_tensor(device)
    sorted_keys, inverse, segments = bucketize_keys_segmented(
        as_keys(values, key_type, device),
        torch.tensor(tids, dtype=torch.int64, device=device),
        offsets,
        geometry.num_buckets,
        capacity,
    )
    old_keys, old_offsets, old_inverse = bucketize_keys(
        as_keys([values[i] for i in live], key_type, device),
        torch.tensor([tids[i] for i in live], dtype=torch.int64, device=device),
        offsets,
        geometry.num_buckets,
        capacity,
    )

    k = len(live)
    assert signed(sorted_keys[:k]) == signed(old_keys)
    assert inverse[:k].tolist() == [live[j] for j in old_inverse.tolist()]
    runs, previous = [], None
    for s in segments[:k].tolist():
        if s != previous:
            runs.append(0)
            previous = s
        runs[-1] += 1
    assert old_offsets[0].item() == 0
    assert torch.diff(old_offsets).tolist() == runs
    assert bool((segments[k:] == geometry.num_buckets).all())


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_the_batch_order_does_not_change_the_sort(device, key_type):
    """The same keys in another order give the same sorted keys and segments, and an
    inverse that leads back into the permuted batch."""
    rng = random.Random(3)
    geometry = Geometry([2, 3], 16)
    n = 2000
    values = rng.sample(range(1, 10**6), n)
    tids = [rng.randrange(2) for _ in range(n)]
    perm = list(range(n))
    rng.shuffle(perm)

    def run(vs, ts):
        return bucketize_keys_segmented(
            as_keys(vs, key_type, device),
            torch.tensor(ts, dtype=torch.int64, device=device),
            geometry.offsets_tensor(device),
            geometry.num_buckets,
            16,
        )

    keys_a, _, segments_a = run(values, tids)
    perm_values = [values[p] for p in perm]
    keys_b, inverse_b, segments_b = run(perm_values, [tids[p] for p in perm])
    assert signed(keys_a) == signed(keys_b)
    assert segments_a.tolist() == segments_b.tolist()
    assert [perm_values[j] for j in inverse_b.tolist()] == signed(keys_b)


@pytest.mark.parametrize("key_type", KEY_TYPES)
@pytest.mark.parametrize("buckets", [[3], [0, 0]], ids=["table", "no-buckets"])
def test_a_batch_of_reserved_keys_is_all_tail(device, key_type, buckets):
    """The padding of a batch that has no real key: every position is inert, in input
    order, and no table id is read (they are all -1)."""
    geometry = Geometry(buckets, 16)
    n = 100
    sorted_keys, inverse, segments = bucketize_keys_segmented(
        as_keys([-1] * n, key_type, device),
        torch.full((n,), -1, dtype=torch.int64, device=device),
        geometry.offsets_tensor(device),
        geometry.num_buckets,
        16,
    )
    assert inverse.tolist() == list(range(n))
    assert signed(sorted_keys) == [-1] * n
    assert segments.tolist() == [geometry.num_buckets] * n


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_an_empty_batch(device, key_type):
    geometry = Geometry([2, 3], 16)
    sorted_keys, inverse, segments = bucketize_keys_segmented(
        as_keys([], key_type, device),
        torch.empty(0, dtype=torch.int64, device=device),
        geometry.offsets_tensor(device),
        geometry.num_buckets,
        16,
    )
    assert sorted_keys.dtype == key_type
    assert sorted_keys.numel() == inverse.numel() == segments.numel() == 0


@pytest.mark.parametrize("key_type", KEY_TYPES)
def test_the_compact_bucketize_skips_the_keys_of_a_last_table_without_buckets(
    device, key_type
):
    """``bucketize_keys`` sends a key of a table without buckets to the first bucket of the
    next table, which for the last table is one past the end. It used to write its
    ``buckets_end`` / ``mask`` entries there, out of bounds; now those keys sort last and
    no offset covers them. 64 buckets, so that the int64 array of ``buckets_end`` fills its
    512-byte block of the caching allocator and the old write landed in the next block (with
    fewer it went into the padding of its own and nothing showed); a sure check is a memory
    checker with the allocator off:
    ``PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck pytest -q -x -k
    last_table_without_buckets test_bucketize_segmented.py``."""
    geometry = Geometry([64, 0], 16)
    n = 200
    rng = random.Random(11)
    values = rng.sample(range(1, 10**6), n)
    tids = [rng.randrange(2) for _ in range(n)]
    live = sum(1 for t in tids if t == 0)
    assert 0 < live < n
    bkt_keys, offsets, inverse = bucketize_keys(
        as_keys(values, key_type, device),
        torch.tensor(tids, dtype=torch.int64, device=device),
        geometry.offsets_tensor(device),
        geometry.num_buckets,
        16,
    )
    assert sorted(inverse.tolist()) == list(range(n))
    assert signed(bkt_keys) == [values[i] for i in inverse.tolist()]
    assert offsets[-1].item() == live
