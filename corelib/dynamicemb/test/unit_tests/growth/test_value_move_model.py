# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Why a growth to a whole multiple can move the values of a table one chunk at a time.

The host tier of a hybrid storage keeps each value at the slot of its key, so a growth
copies every value to the slot its key gets. ``_ValueMove`` (``key_value_table.py``) loads
and stores a chunk before it reads the next one when the new size is a whole multiple of the
old one, which is only safe if a store never lands on a row that a later chunk has yet to
read. This is that argument, run on a pure-Python model of the table, with no GPU and no
torch:

* the model buckets a key the way ``insert_body`` does (``kernels.cuh``: ``local_idx =
  hash % table_slots``, ``bucket = local_idx / bucket_capacity``) and probes inside the
  bucket from ``hash % bucket_capacity``;
* a growth by a whole multiple sends a key of old bucket ``b`` to bucket ``b + j *
  old_buckets`` (``j >= 0``): its own bucket or one past the end of the old table, whatever
  the hash, so the stores of a chunk never clobber an unread row, for every chunking;
* a size that is not a multiple does clobber (the model says so: the same loop run on 1.5
  times the size loses values), which is why the host decides on the sizes
  (``chunks_are_independent``) and everything else loads before it stores.

It cannot show that the device hash is the one modelled here: the device asserts the
invariant on every lane of every chunk (``lands_in_the_bucket_of_origin_or_a_new_one``).
"""

import importlib.util
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # corelib/dynamicemb
PLAN = os.path.join(ROOT, "dynamicemb", "rehash_plan.py")


def load_plan():
    spec = importlib.util.spec_from_file_location("rehash_plan_under_test", PLAN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan = load_plan()

MASK64 = (1 << 64) - 1


def hash_of(key: int) -> int:
    """``Table::hash``: murmur3's 64-bit finalizer, masked to a non-negative int64."""
    k = key & MASK64
    k ^= k >> 33
    k = (k * 0xFF51AFD7ED558CCD) & MASK64
    k ^= k >> 33
    k = (k * 0xC4CEB9FE1A85EC53) & MASK64
    k ^= k >> 33
    return k & ((1 << 63) - 1)


class Table:
    """The key column of one table: ``slots`` slots in buckets of ``bucket`` slots."""

    def __init__(self, slots: int, bucket: int):
        assert slots % bucket == 0
        self.slots, self.bucket = slots, bucket
        self.keys = [None] * slots

    def insert(self, key: int) -> int:
        """The slot ``key`` takes (the model never overfills a bucket)."""
        h = hash_of(key)
        bucket = (h % self.slots) // self.bucket
        for i in range(self.bucket):
            slot = bucket * self.bucket + (h % self.bucket + i) % self.bucket
            if self.keys[slot] is None:
                self.keys[slot] = key
                return slot
        raise OverflowError("a bucket is full")


def populate(slots: int, bucket: int, load: float, rng: random.Random):
    table = Table(slots, bucket)
    rows = [None] * slots
    for key in rng.sample(range(1, 1 << 40), int(load * slots)):
        slot = table.insert(key)
        rows[slot] = ("value", key)
    return table, rows


def grow(old, rows, new_slots, chunk_buckets, chunk_by_chunk):
    """What ``rehash_table_from`` and ``_ValueMove`` do: per chunk of old buckets insert
    the keys, load their values, and either store at once or after the last chunk.
    Returns the rows of the grown table, how many stores landed on a row that was not
    read yet, and whether every lane kept the invariant the device asserts."""
    new = Table(new_slots, old.bucket)
    rows = list(rows) + [None] * (new_slots - old.slots)  # the buffer grew in place
    old_buckets = old.slots // old.bucket
    unread = {slot for slot, key in enumerate(old.keys) if key is not None}
    staged, clobbered, invariant = [], 0, True
    for first in range(0, old_buckets, chunk_buckets):
        lo = first * old.bucket
        hi = min(first + chunk_buckets, old_buckets) * old.bucket
        lanes = [(s, new.insert(old.keys[s])) for s in range(lo, hi) if old.keys[s]]
        loaded = [(dst, rows[src]) for src, dst in lanes]  # the load kernel
        unread -= {src for src, _ in lanes}
        invariant &= all(
            plan.lands_in_the_bucket_of_origin_or_a_new_one(
                src, dst, old.bucket, old_buckets
            )
            for src, dst in lanes
        )
        if chunk_by_chunk:
            for dst, value in loaded:  # the store kernel
                clobbered += dst in unread
                rows[dst] = value
        else:
            staged.extend(loaded)
    for dst, value in staged:
        rows[dst] = value
    return new, rows, clobbered, invariant


def wrong_values(new, rows):
    return sum(
        1 for slot, key in enumerate(new.keys) if key and rows[slot] != ("value", key)
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("factor", [2, 3, 4, 8])
@pytest.mark.parametrize("chunk_buckets", [1, 2, 5, 1000])
def test_a_whole_multiple_can_be_moved_one_chunk_at_a_time(seed, factor, chunk_buckets):
    rng = random.Random(1000 * factor + seed)
    bucket, old_slots = 128, 128 * 24
    old, rows = populate(old_slots, bucket, 0.4, rng)
    assert plan.chunks_are_independent(old_slots, factor * old_slots)

    new, moved, clobbered, invariant = grow(
        old, rows, factor * old_slots, chunk_buckets, chunk_by_chunk=True
    )

    assert invariant, "a key left its bucket for one that is neither its own nor new"
    assert clobbered == 0, "a store landed on a row that had not been read"
    assert wrong_values(new, moved) == 0
    assert sum(k is not None for k in new.keys) == sum(k is not None for k in old.keys)


@pytest.mark.parametrize("chunk_buckets", [1, 3, 1000])
def test_the_same_loop_on_a_size_that_is_not_a_multiple_loses_values(chunk_buckets):
    """The model is sensitive: where the host says no, storing as it goes is wrong."""
    old_slots, new_slots = 128 * 24, 128 * 36  # 1.5 times
    assert not plan.chunks_are_independent(old_slots, new_slots)
    bad = 0
    for seed in range(10):
        old, rows = populate(old_slots, 128, 0.4, random.Random(seed))
        new, moved, clobbered, invariant = grow(
            old, rows, new_slots, chunk_buckets, chunk_by_chunk=True
        )
        bad += clobbered + (0 if invariant else 1)
    assert bad > 0


@pytest.mark.parametrize("new_slots", [128 * 36, 128 * 25, 128 * 47])
@pytest.mark.parametrize("chunk_buckets", [1, 7, 1000])
def test_every_load_before_any_store_is_right_for_any_size(new_slots, chunk_buckets):
    old, rows = populate(128 * 24, 128, 0.4, random.Random(3))
    new, moved, clobbered, _ = grow(
        old, rows, new_slots, chunk_buckets, chunk_by_chunk=False
    )
    assert clobbered == 0 and wrong_values(new, moved) == 0


def test_the_rule_of_the_host():
    independent = plan.chunks_are_independent
    assert independent(1024, 2048) and independent(1024, 3072) and independent(128, 128)
    assert not independent(1024, 1536) and not independent(1024, 1152)
    assert not independent(0, 0) and not independent(0, 128)  # a table without slots
