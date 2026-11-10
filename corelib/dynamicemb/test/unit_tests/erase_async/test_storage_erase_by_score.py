# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""``DynamicEmbStorage.erase_by_score``: the scan of a table, a chunk at a time.

The chunk rule is a pure function and is tested as one. The walk over the chunks is
tested on a storage, with a guard around each chunk standing in for what the
background worker will put there.
"""

import pytest
import torch
from dynamicemb import DynamicEmbCheckMode, DynamicEmbScoreStrategy, EvictedItemMode
from dynamicemb.dynamicemb_config import DynamicEmbEvictStrategy, DynamicEmbTableOptions
from dynamicemb.key_value_table import (
    ERASE_CHUNK_SLOTS,
    ERASE_MAX_CHUNKS_IN_FLIGHT,
    DynamicEmbStorage,
    _expand_tables_impl,
    next_erase_chunk_end,
)
from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer
from dynamicemb.scored_hashtable import ScoreArg
from dynamicemb_extensions import ScorePolicy
from helpers import BUCKET, erased_and_spared, new_counts


def walk(table_slots, bucket_capacity, target):
    """The ends of the chunks of a table, from its first slot to its last."""
    ends, position = [], 0
    while position < table_slots:
        position = next_erase_chunk_end(position, table_slots, bucket_capacity, target)
        ends.append(position)
    return ends


# ---------------------------------------------------------------------------
# The chunk rule: whole buckets, clamped to the table, inside the table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table_slots, bucket_capacity, target, expected",
    [
        # A table smaller than a chunk is one chunk. (The default is ERASE_CHUNK_SLOTS.)
        (BUCKET * 8, BUCKET, ERASE_CHUNK_SLOTS, [BUCKET * 8]),
        # A chunk of whole buckets.
        (BUCKET * 8, BUCKET, 2 * BUCKET, [256, 512, 768, 1024]),
        # A target that is not a whole number of buckets is rounded down to one:
        # 1000 // 128 = 7 buckets, 896 slots, then the short last chunk.
        (BUCKET * 8, BUCKET, 1000, [896, 1024]),
        # A target below a bucket is a bucket, never zero.
        (BUCKET * 8, BUCKET, 64, [BUCKET * n for n in range(1, 9)]),
        # The last chunk of a table is the short one; bigger buckets, same rule.
        (1024 * 5, 1024, 2048, [2048, 4096, 5120]),
        # Big enough to need several default chunks.
        (1 << 25, BUCKET, ERASE_CHUNK_SLOTS, [ERASE_CHUNK_SLOTS * k for k in range(1, 9)]),
    ],
)
def test_the_chunks_of_a_table(table_slots, bucket_capacity, target, expected):
    assert walk(table_slots, bucket_capacity, target) == expected


@pytest.mark.parametrize("bucket_capacity", [16, 1024])
@pytest.mark.parametrize("target", [1, 5000, 1 << 24])
@pytest.mark.parametrize("num_buckets", [1, 7, 4097])
def test_the_chunks_tile_the_table(bucket_capacity, target, num_buckets):
    """Whatever the sizes: the chunks follow one another from the first slot to the
    last, each is whole buckets, none is longer than the target allows (a bucket at
    least), and none reaches past the table."""
    table_slots = num_buckets * bucket_capacity
    ends = walk(table_slots, bucket_capacity, target)
    starts = [0] + ends[:-1]
    longest = max(1, target // bucket_capacity) * bucket_capacity
    assert ends[-1] == table_slots
    for start, end in zip(starts, ends):
        assert start < end <= table_slots
        assert start % bucket_capacity == 0 and end % bucket_capacity == 0
        assert end - start <= longest


@pytest.mark.parametrize(
    "args, match",
    [
        ((0, BUCKET, 0), "positive"),  # no bucket
        ((0, BUCKET, BUCKET, 0), "positive"),  # no target
        ((0, BUCKET + 1, BUCKET), "whole number of buckets"),
        ((0, 0, BUCKET), "whole number of buckets"),
        ((5, BUCKET * 4, BUCKET), "start of a bucket"),  # not on a bucket
        ((BUCKET * 4, BUCKET * 4, BUCKET), "start of a bucket"),  # past the last slot
        ((-BUCKET, BUCKET * 4, BUCKET), "start of a bucket"),
    ],
)
def test_the_chunk_rule_refuses_what_is_not_a_table_or_a_position(args, match):
    with pytest.raises(ValueError, match=match):
        next_erase_chunk_end(*args)


# ---------------------------------------------------------------------------
# The walk, on a storage
# ---------------------------------------------------------------------------


def make_storage(capacities):
    """A storage of one table per capacity, whose scores the test sets itself."""
    device_id = torch.cuda.current_device()
    options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device_id,
            dim=8,
            max_capacity=capacity * 16,
            init_capacity=capacity,
            bucket_capacity=BUCKET,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
            evict_strategy=DynamicEmbEvictStrategy.CUSTOMIZED,
        )
        for capacity in capacities
    ]
    return DynamicEmbStorage(options, SGDDynamicEmbeddingOptimizer(OptimizerArgs()))


def score_name(storage):
    return storage._state.score_policy.name


def fill(storage, keys, scores, table_id, device):
    keys = torch.as_tensor(keys, dtype=torch.int64, device=device)
    scores = torch.as_tensor(scores, dtype=torch.uint64, device=device)
    storage.key_index_map.insert(
        keys,
        torch.full_like(keys, table_id),
        ScoreArg(name=score_name(storage), value=scores, policy=ScorePolicy.ASSIGN),
    )


def found(storage, keys, table_id, device):
    """Which of ``keys`` the table holds -- in the key map it has *now*."""
    keys = torch.as_tensor(keys, dtype=torch.int64, device=device)
    arg = ScoreArg(name=score_name(storage), value=None, policy=ScorePolicy.CONST)
    _, founds, _ = storage.key_index_map.lookup(keys, torch.full_like(keys, table_id), arg)
    return founds


class RecordingGuard:
    """What a chunk runs inside: made anew for each (the storage calls it), entered
    and left around the chunk. Counts the entries; ``on_enter`` is called on each,
    with the number of the entry, before the chunk reads anything."""

    def __init__(self, on_enter=None):
        self.entered = 0
        self.on_enter = on_enter

    def __call__(self):
        return self

    def __enter__(self):
        self.entered += 1
        if self.on_enter is not None:
            self.on_enter(self.entered)

    def __exit__(self, *exc):
        return False


KEYS = list(range(1, 601))
SCORES = [(k * 7) % 100 for k in KEYS]  # 0..99
THRESHOLD = 40
BELOW = [k for k, s in zip(KEYS, SCORES) if s < THRESHOLD]
KEPT = [k for k, s in zip(KEYS, SCORES) if s >= THRESHOLD]


@pytest.mark.parametrize("table_ids", [[0, 1], [1]])
@pytest.mark.parametrize("chunk_slots", [BUCKET, 2 * BUCKET, 1000, ERASE_CHUNK_SLOTS])
@pytest.mark.parametrize("max_in_flight", [1, 2, 8])
def test_scanning_in_chunks_erases_the_keys_below_the_threshold(
    device, chunk_slots, max_in_flight, table_ids
):
    """Tables of different sizes, so the same chunk size cuts each differently; one
    bucket at a time, a ragged target, and the default; with one, two or eight chunks
    allowed in flight -- how the scan is cut and paced does not change what it erases. A
    table that is not named is not touched, and each chunk of the ones that are runs
    inside a guard of its own -- one per chunk and no more: an empty one to find that the
    table was done would take the lock for nothing."""
    capacities = [BUCKET * 16, BUCKET * 8]
    storage = make_storage(capacities)
    for table_id in (0, 1):
        fill(storage, KEYS, SCORES, table_id, device)

    guard = RecordingGuard()
    counts = new_counts(device)
    storage.erase_by_score(
        THRESHOLD,
        table_ids,
        counts,
        chunk_slots=chunk_slots,
        chunk_guard=guard,
        max_in_flight=max_in_flight,
    )

    assert erased_and_spared(counts) == (len(table_ids) * len(BELOW), 0)
    for table_id in (0, 1):
        size = int(storage.key_index_map.size(table_id))
        if table_id in table_ids:
            assert not bool(found(storage, BELOW, table_id, device).any())
            assert bool(found(storage, KEPT, table_id, device).all())
            assert size == len(KEPT)
        else:
            assert bool(found(storage, KEYS, table_id, device).all()), "it was touched"
            assert size == len(KEYS)
    chunks = sum(len(walk(capacities[t], BUCKET, chunk_slots)) for t in table_ids)
    assert guard.entered == chunks


def test_a_key_map_replaced_between_two_chunks_restarts_that_table(device):
    """An expansion between chunks replaces the key map: the old one is gone, and the
    slots the scan had passed mean nothing in the new one. The table is scanned again
    from its first bucket of the new map, and nothing below the threshold is left."""
    storage = make_storage([BUCKET * 16, BUCKET * 8])
    state = storage._state
    fill(storage, KEYS, SCORES, 0, device)
    new_slots = BUCKET * 32

    def expand_before_the_third_chunk(entry):
        if entry == 3:
            _expand_tables_impl(state, [True, False], [new_slots, -1])

    guard = RecordingGuard(expand_before_the_third_chunk)
    counts = new_counts(device)
    storage.erase_by_score(
        THRESHOLD, [0], counts, chunk_slots=2 * BUCKET, chunk_guard=guard
    )

    assert state.key_map_generation == 1
    assert storage.key_index_map.main_capacity(0) == new_slots
    # Two chunks of the old map, then all of the new one.
    assert guard.entered == 2 + new_slots // (2 * BUCKET)
    # Each key is erased once, whether before the expansion or after it.
    assert erased_and_spared(counts) == (len(BELOW), 0)
    assert not bool(found(storage, BELOW, 0, device).any())
    assert bool(found(storage, KEPT, 0, device).all())
    assert int(storage.key_index_map.size(0)) == len(KEPT)


def test_a_table_that_is_not_there_is_refused_before_anything_is_scanned(device):
    storage = make_storage([BUCKET * 16, BUCKET * 8])
    fill(storage, KEYS, SCORES, 0, device)
    counts = new_counts(device)
    for table_ids in ([0, 2], [-1], [0, 1, 7]):
        with pytest.raises(ValueError, match="is not one of the 2 tables"):
            storage.erase_by_score(THRESHOLD, table_ids, counts)
    assert erased_and_spared(counts) == (0, 0)
    assert int(storage.key_index_map.size(0)) == len(KEYS), "table 0 was scanned first"


def test_a_table_the_scan_removed_keys_from_is_marked_untracked_and_no_other(device):
    """The removals are not recorded, so a delta dumped from a table that lost some cannot
    be replayed exactly: the table is marked, as ``erase_keys`` without ``RETAIN_KEY``
    marks one. A table the scan went over and found nothing to remove in is not, nor one
    it was not asked to scan, whatever it holds. The mark is sticky."""
    storage = make_storage([BUCKET * 16, BUCKET * 8, BUCKET * 8])
    fill(storage, KEYS, SCORES, 0, device)  # some below the threshold
    fill(storage, KEPT, [THRESHOLD + 5] * len(KEPT), 1, device)  # none below
    fill(storage, KEYS, SCORES, 2, device)  # some below, but not scanned
    assert [storage.erase_untracked(t) for t in (0, 1, 2)] == [False] * 3

    counts = new_counts(device)
    storage.erase_by_score(THRESHOLD, [0, 1], counts)
    assert erased_and_spared(counts) == (len(BELOW), 0), "each table's count is added"
    assert [storage.erase_untracked(t) for t in (0, 1, 2)] == [True, False, False]

    storage.erase_by_score(THRESHOLD, [0], new_counts(device))  # nothing left to remove
    assert storage.erase_untracked(0)


@pytest.mark.parametrize("chunk_slots", [BUCKET, 3 * BUCKET, ERASE_CHUNK_SLOTS])
def test_a_retained_scan_reports_the_keys_it_erased_and_does_not_mark_the_table(
    device, chunk_slots
):
    storage = make_storage([BUCKET * 16, BUCKET * 8])
    fill(storage, KEYS, SCORES, 0, device)
    fill(storage, KEYS, SCORES, 1, device)  # not scanned

    counts = new_counts(device)
    storage.erase_by_score(
        THRESHOLD,
        [0],
        counts,
        chunk_slots=chunk_slots,
        mode=EvictedItemMode.RETAIN_KEY,
    )

    assert erased_and_spared(counts) == (len(BELOW), 0)
    erased = storage.pop_erased_keys(0).tolist()
    assert sorted(erased) == BELOW
    slot_index = storage.pop_erased_slot_index(0)
    assert slot_index.shape == (len(BELOW), 2)
    assert bool((slot_index >= 0).all())
    assert storage.pop_erased_keys(1).numel() == 0
    assert [storage.erase_untracked(t) for t in (0, 1)] == [False, False]
    assert not bool(found(storage, BELOW, 0, device).any())
    assert bool(found(storage, KEPT, 0, device).all())


def test_a_key_map_replaced_during_a_retained_scan_reports_no_key_twice(device):
    storage = make_storage([BUCKET * 16, BUCKET * 8])
    state = storage._state
    fill(storage, KEYS, SCORES, 0, device)

    def expand_before_the_third_chunk(entry):
        if entry == 3:
            _expand_tables_impl(state, [True, False], [BUCKET * 32, -1])

    storage.erase_by_score(
        THRESHOLD,
        [0],
        new_counts(device),
        chunk_slots=2 * BUCKET,
        chunk_guard=RecordingGuard(expand_before_the_third_chunk),
        mode=EvictedItemMode.RETAIN_KEY,
    )

    assert state.key_map_generation == 1
    # What was recorded, before the pop de-duplicates it.
    recorded = torch.cat(state.erased_key_chunks).tolist()
    assert sorted(recorded) == BELOW
    assert not storage.erase_untracked(0)


class FailingAfterChunk(RecordingGuard):
    """Raises on leaving chunk ``fail_at``: its kernel is launched, its record not kept."""

    def __init__(self, fail_at):
        super().__init__()
        self.fail_at = fail_at

    def __exit__(self, *exc):
        if self.entered == self.fail_at:
            raise RuntimeError("the scan failed")
        return False


def test_a_retained_scan_that_fails_after_a_chunk_ran_marks_the_table(device):
    storage = make_storage([BUCKET * 16])
    fill(storage, KEYS, SCORES, 0, device)

    with pytest.raises(RuntimeError, match="the scan failed"):
        storage.erase_by_score(
            THRESHOLD,
            [0],
            new_counts(device),
            chunk_slots=4 * BUCKET,
            chunk_guard=FailingAfterChunk(2),
            mode=EvictedItemMode.RETAIN_KEY,
        )

    torch.cuda.synchronize()
    lost = len(KEYS) - int(storage.key_index_map.size(0))
    assert lost > storage.pop_erased_keys(0).numel(), "the failed chunk erased keys"
    assert storage.erase_untracked(0)


class SlowChunks:
    """The guard of a scan whose chunks take a while on the device, counting how many
    of the chunks before the one being entered are still running there."""

    def __init__(self):
        self.events, self.running_at_entry = [], []

    def __call__(self):
        return self

    def __enter__(self):
        self.running_at_entry.append(sum(1 for e in self.events if not e.query()))
        torch.cuda._sleep(2_000_000)  # about a millisecond, before this chunk's kernel

    def __exit__(self, *exc):
        event = torch.cuda.Event()
        event.record()
        self.events.append(event)
        return False


@pytest.mark.parametrize("max_in_flight", [1, 2, 4])
def test_the_scan_never_has_more_chunks_in_flight_than_it_is_allowed(
    device, max_in_flight
):
    """Launching a chunk takes microseconds and running it does not, so a scan left to
    itself queues the table's every chunk at once -- and whoever takes the table next
    waits for all of them on the device. Told to keep ``max_in_flight`` in flight,
    the thread waits for the chunk that many before it to finish, outside the guard."""
    storage = make_storage([BUCKET * 16, BUCKET * 8])
    guard = SlowChunks()
    storage.erase_by_score(
        THRESHOLD,
        [0],
        new_counts(device),
        chunk_slots=BUCKET,
        chunk_guard=guard,
        max_in_flight=max_in_flight,
    )
    torch.cuda.synchronize()
    assert len(guard.running_at_entry) == 16
    # Before this chunk's launch, the one ``max_in_flight`` back had finished.
    assert max(guard.running_at_entry) <= max_in_flight - 1, guard.running_at_entry


def test_without_the_limit_the_same_scan_queues_the_whole_table(device):
    """The control: the test above would pass for a scan that is slow for some other
    reason; this one shows the chunks do pile up when nothing holds them back."""
    storage = make_storage([BUCKET * 16, BUCKET * 8])
    guard = SlowChunks()
    storage.erase_by_score(
        THRESHOLD,
        [0],
        new_counts(device),
        chunk_slots=BUCKET,
        chunk_guard=guard,
        max_in_flight=1000,
    )
    torch.cuda.synchronize()
    assert max(guard.running_at_entry) >= 5, guard.running_at_entry


def test_the_limit_cannot_be_less_than_one_chunk(device):
    assert ERASE_MAX_CHUNKS_IN_FLIGHT >= 1
    storage = make_storage([BUCKET * 16, BUCKET * 8])
    with pytest.raises(ValueError, match="at least 1"):
        storage.erase_by_score(THRESHOLD, [0], new_counts(device), max_in_flight=0)
