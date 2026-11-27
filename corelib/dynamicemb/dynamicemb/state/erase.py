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

"""Erasing keys of a state: by score in chunks, or a given list of keys."""

import collections
import contextlib
from typing import Callable, ContextManager, Optional, Sequence

import torch
from dynamicemb.dynamicemb_config import EvictedItemMode
from dynamicemb.state.evicted_items import _append_erased
from dynamicemb.state.table_state import DynamicEmbTableState
from torch import Tensor

# The slots of a table one chunk of an erase by score scans, at most. A chunk is also
# the length of the kernel the training's own kernels can end up waiting behind, so
# the size is a trade between how much the scan disturbs the training and how long it
# takes. Measured on an H100, a table of 2**29 slots at half load, training steps of
# 0.7 ms, scans one after the other and two chunks in flight: 2**20 or 2**22 slots
# slow a step by about 10%, 2**24 by 110%, 2**26 by 8 times; in the same 300 steps
# 13-15 scans finish with 2**22, 115 with 2**24. A chunk of 2**22 slots is about 30
# microseconds of scan, so a smaller one is mostly the cost of launching it (2**18:
# 46 ms for the table, against 3.7 ms for 2**22). It is a parameter of the call.
ERASE_CHUNK_SLOTS = 1 << 22


# How many chunks of an erase by score may be launched and not yet finished on the
# device. Launching is some thirty times faster than a chunk runs, so without a limit
# the erase queues the whole table at once, and the prefetch that takes the table next
# -- whose stream waits for the erase's last event -- waits for all of it. With it, that
# wait is at most this many chunks (about a millisecond for two). Two keep the device
# busy: one chunk running, the next already queued.
ERASE_MAX_CHUNKS_IN_FLIGHT = 2


def next_erase_chunk_end(
    position: int,
    table_slots: int,
    bucket_capacity: int,
    target_slots: int = ERASE_CHUNK_SLOTS,
) -> int:
    """Where the chunk of an erase by score that starts at ``position`` ends.

    A chunk is a whole number of buckets and lies inside one table:

    * the size is counted in buckets, ``target_slots // bucket_capacity`` of them,
      so a bucket is never split;
    * never more than the table has, and never less than one bucket: a table smaller
      than a chunk is scanned as one, and a target below a bucket is a bucket;
    * it stops at the table's last slot (the last chunk is the short one), so it
      never reaches into the next table.

    ``position`` and the returned end are slots counted from the start of the table.
    """
    if bucket_capacity <= 0 or target_slots <= 0:
        raise ValueError("bucket_capacity and target_slots must be positive")
    if table_slots <= 0 or table_slots % bucket_capacity != 0:
        raise ValueError(
            f"a table is a whole number of buckets, got {table_slots} slots of "
            f"{bucket_capacity}-slot buckets"
        )
    if not 0 <= position < table_slots or position % bucket_capacity != 0:
        raise ValueError(
            f"position {position} is not the start of a bucket inside the table's "
            f"{table_slots} slots"
        )
    table_buckets = table_slots // bucket_capacity
    chunk_buckets = max(1, min(table_buckets, target_slots // bucket_capacity))
    return min(position + chunk_buckets * bucket_capacity, table_slots)


def _erase_state_by_score(
    state: DynamicEmbTableState,
    threshold: int,
    table_ids: Sequence[int],
    counts: Tensor,
    chunk_slots: int = ERASE_CHUNK_SLOTS,
    chunk_guard: Optional[Callable[[], ContextManager]] = None,
    max_in_flight: int = ERASE_MAX_CHUNKS_IN_FLIGHT,
    score_word: Optional[int] = None,
) -> None:
    """Erase the keys of ``table_ids`` whose score is below ``threshold``, a chunk
    of slots at a time (:func:`next_erase_chunk_end`). ``score_word`` is the word
    compared, as in :meth:`LinearBucketTable.erase_by_score`.

    No more than ``max_in_flight`` chunks are launched and unfinished at once: before
    launching one, the thread waits -- outside the guard, so holding nothing and not
    the GIL -- for the chunk ``max_in_flight`` before it to finish on the device. What a
    caller that takes the table after the guard has to wait for on the device is then
    bounded by those chunks, not by the rest of the scan.

    Each chunk runs inside ``chunk_guard()`` -- a context manager made anew for
    every chunk, which does nothing by default. That is where a caller that has to
    keep the table to itself for a chunk (and only a chunk) takes and drops its
    lock. Nothing about the table is planned ahead: the key map, the table's size
    and the chunk are all read inside the guard, because between two chunks
    another thread may have expanded the table, replacing the key map and freeing
    the old one.

    A replaced key map also moves every key to another slot, so where the scan had
    got to means nothing in the new one: when ``state.key_map_generation`` is not
    the one this table's scan started with, the table is scanned again from its first
    bucket. That is safe -- erasing by score is idempotent -- and a key an expansion
    moved past the scan is left for the next erase.

    ``counts`` is the device tensor of :meth:`LinearBucketTable.erase_by_score`,
    added to by every chunk.

    The keys it removes are not recorded (no ``RETAIN_KEY``), and a table that stores a
    row per key leaves each erased slot reclaimed with its row, which only a recorded
    erase lets an incremental dump report. So a table it removed any key from is marked,
    as :func:`_erase_state_keys` marks one (``erase_untracked_tables``): a delta dumped
    from it cannot be replayed exactly, and ``replay_increment`` refuses it. A table
    from which it removed nothing is not marked -- the mark is sticky. To know, each
    table is scanned into a count of its own, which the worker reads when the table is
    done (a wait for its own stream only) before adding it to ``counts``.
    """
    for table_id in table_ids:
        if not 0 <= table_id < state.num_tables:
            raise ValueError(
                f"table_id {table_id} is not one of the {state.num_tables} tables"
            )
    if max_in_flight < 1:
        raise ValueError(f"max_in_flight must be at least 1, got {max_in_flight}")
    guard = contextlib.nullcontext if chunk_guard is None else chunk_guard
    in_flight: "collections.deque[torch.cuda.Event]" = collections.deque()
    for table_id in table_ids:
        table_counts = torch.zeros(2, dtype=counts.dtype, device=counts.device)
        # Marked also when the scan fails half way: whatever it removed before is gone.
        try:
            position = 0
            generation: Optional[int] = None
            finished = False
            while not finished:
                if len(in_flight) >= max_in_flight:
                    in_flight.popleft().synchronize()
                with guard():
                    key_map = state.key_index_map
                    if generation != state.key_map_generation:
                        # The first chunk, or the key map was replaced since the last.
                        generation = state.key_map_generation
                        position = 0
                    table_slots = key_map.main_capacity(table_id)
                    end = next_erase_chunk_end(
                        position, table_slots, key_map.bucket_capacity_, chunk_slots
                    )
                    key_map.erase_by_score(
                        table_id,
                        threshold,
                        table_counts,
                        score_word=score_word,
                        begin=position,
                        end=end,
                    )
                    position = end
                    finished = position >= table_slots
                chunk_launched = torch.cuda.Event()
                chunk_launched.record()
                in_flight.append(chunk_launched)
        finally:
            counts.add_(table_counts)
            if state.key_index_map.use_rows_ and int(table_counts[0]) > 0:
                state.erase_untracked_tables.add(table_id)


def _erase_state_keys(
    state: DynamicEmbTableState,
    table_id: int,
    keys: torch.Tensor,
    mode: EvictedItemMode = EvictedItemMode.DISCARD,
    track_untracked: bool = True,
) -> int:
    """Erase keys from one table state. Returns how many were actually present.

    The erase kernel reports the slot it cleared, or ``-1`` for a key that was
    not there, so the count is the real number removed rather than the number
    asked for -- a delta can legitimately name keys this replica never held.

    *mode* is this call's, not the table's: retaining an erase costs only a copy
    of keys already in hand, so there is nothing to configure up front and each
    caller can decide whether its removals are worth reporting.

    A table that stores rows leaves each erased slot reclaimed with its row, which
    only a retained erase reports (see ``erase_untracked``). ``track_untracked``
    is off for a replica applying a delta's removals, which are not its own.
    """
    n = keys.numel()
    if n == 0:
        return 0
    device = state.device
    keys = keys.to(device=device, dtype=state.key_index_map.key_type)
    tids = torch.full((n,), table_id, dtype=torch.int64, device=device)
    indices = state.key_index_map.erase(keys, tids, return_indices=True)
    removed = indices >= 0
    if EvictedItemMode.RETAIN_KEY in mode:
        # Only the keys that were really there: a caller may name keys this
        # table never held, and reporting those as removed would make a replica
        # replay removals that never happened.
        slot_index = None
        if state.key_index_map.use_rows_:
            # The slot stays reclaimed with its row; record both so a replica can
            # recreate it, laid out the way a dumped slot_index is.
            slots = indices[removed]
            rows = state.key_index_map._rows_of(slots, tids[removed], False)
            slot_index = torch.stack([slots, rows], dim=1)
        _append_erased(state, keys[removed], table_id, slot_index)
    num_removed = int(removed.sum().item())
    if (
        track_untracked
        and num_removed > 0
        and state.key_index_map.use_rows_
        and EvictedItemMode.RETAIN_KEY not in mode
    ):
        state.erase_untracked_tables.add(table_id)
    return num_removed
