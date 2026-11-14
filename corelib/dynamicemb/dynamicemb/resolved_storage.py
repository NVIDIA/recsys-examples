# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live resolved rows for stores whose interface is find/insert (including PS).

Only outstanding batches are retained. Their rows act as a small write-through
working set: an optimizer update is published to overlapping outstanding batches
before the next lookup reads them. TableAccessGate orders these copies and the
backing-store operations across input_dist and compute streams. Backends keep
their existing Storage interface; no remote row-addressing API is assumed.
"""

from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import torch
from dynamicemb_extensions import copy_matching_storage_rows

if TYPE_CHECKING:
    from .batched_dynamicemb_function import PrefetchState


def matching_rows(source_keys, source_tids, target_keys, target_tids, table_num):
    """Positions of matching (table, key) pairs, without narrowing 64-bit keys."""
    source, target = [], []
    source_keys = source_keys.to(torch.int64)
    target_keys = target_keys.to(torch.int64)
    # Signed sorting is also a bijection for uint64 keys; both sides use the
    # same ordering, including keys whose high bit is set.
    for tid in range(table_num):
        si = (source_tids == tid).nonzero().flatten()
        ti = (target_tids == tid).nonzero().flatten()
        if si.numel() == 0 or ti.numel() == 0:
            continue
        keys, order = source_keys[si].sort()
        targets = target_keys[ti].contiguous()
        positions = torch.searchsorted(keys, targets)
        bounded = positions.clamp(max=keys.numel() - 1)
        hit = (positions < keys.numel()) & (keys[bounded] == targets)
        source.append(si[order[bounded[hit]]])
        target.append(ti[hit])
    empty = source_tids.new_empty(0, dtype=torch.int64)
    return (torch.cat(source), torch.cat(target)) if source else (empty, empty)


class StorageRowIndex:
    """Immutable per-batch key index for publishing live optimizer rows.

    Unique keys are contiguous by table. Keep that boundary and sort each
    segment once, rather than compacting matches (and reading their sizes back
    to the CPU) for every pair of overlapping batches.
    """

    def __init__(self, batch):
        self.keys = batch.unique_keys.to(torch.int64)
        self.sorted_keys = self.sorted_rows = None
        self.persisted = None
        if batch.persisted_unique_indices.numel() != self.keys.numel():
            self.persisted = torch.zeros_like(self.keys, dtype=torch.bool)
            self.persisted[batch.persisted_unique_indices] = True

    def prepare_source(self, sizes):
        if self.sorted_keys is not None:
            return
        keys, rows = [], []
        start = 0
        for size in sizes:
            sorted_keys, order = self.keys[start : start + size].sort()
            keys.append(sorted_keys)
            rows.append(order + start if start else order)
            start += size
        self.sorted_keys = torch.cat(keys) if len(keys) > 1 else keys[0]
        self.sorted_rows = torch.cat(rows) if len(rows) > 1 else rows[0]

    def record_stream(self, stream):
        for tensor in (self.keys, self.sorted_keys, self.sorted_rows, self.persisted):
            if tensor is not None:
                tensor.record_stream(stream)


def _copy_matching_rows(source, target, stream):
    # The registry/gate owns access to these caches; they never own another
    # batch. Keys, table boundaries and admission decisions stay immutable for
    # the batch's lifetime, including across retained backwards.
    for batch in (source, target):
        if batch.storage_row_index is None:
            batch.storage_row_index = StorageRowIndex(batch)
    source_index, target_index = source.storage_row_index, target.storage_row_index
    source_index.prepare_source(source.unique_sizes)
    source_index.record_stream(stream)
    target_index.record_stream(stream)
    # Full signed 64-bit keys, per-table ranges and optional persisted masks:
    # no key packing, dynamic output size or device-to-host count transfer.
    copy_matching_storage_rows(
        source_index.sorted_keys,
        source_index.sorted_rows,
        source.unique_table_offsets,
        source.unique_values,
        target_index.keys,
        target.unique_table_ids,
        target.unique_values,
        source_index.persisted,
        target_index.persisted,
    )


class ActiveStorageRows:
    def __init__(self):
        # The distributed input and autograd context own a batch. The registry
        # must not keep it alive when its graph is discarded without backward.
        self._states: WeakValueDictionary[int, "PrefetchState"] = WeakValueDictionary()

    @property
    def states(self):
        # Hold live batches while find/publish enqueues its CUDA copies, even
        # if their input/graph owners are concurrently discarded.
        return list(self._states.values())

    def overlay(self, storage):
        return _StorageWithActiveRows(storage, self)

    def add(self, batch):
        self._states[id(batch)] = batch

    def release(self, batch):
        self._states.pop(id(batch), None)

    def publish(self, batch):
        stream = torch.cuda.current_stream(batch.unique_keys.device.index)
        batch.record_stream(stream)
        if batch.persisted_unique_indices.numel() == 0:
            return
        for other in self.states:
            if other is batch or other.persisted_unique_indices.numel() == 0:
                continue
            other.record_stream(stream)
            _copy_matching_rows(batch, other, stream)


class _StorageWithActiveRows:
    def __init__(self, storage, active):
        self._storage = storage
        self._active = active

    def __getattr__(self, name):
        return getattr(self._storage, name)

    def find(self, keys, table_ids, **kwargs):
        result = self._storage.find(keys, table_ids, **kwargs)
        # Updates are written through under TableAccessGate before publication.
        # A backing-store hit is therefore current; only evicted/missing keys
        # need restoration from an outstanding snapshot.
        if result[0] == 0:
            return result
        states = self._active.states
        if not states:
            return result
        (
            n,
            missing_keys,
            missing_pos,
            missing_tids,
            missing_scores,
            found,
            scores,
            values,
        ) = result
        live = torch.zeros_like(found)
        stream = torch.cuda.current_stream()
        for batch in states:
            batch.record_stream(stream)
            persisted = batch.persisted_unique_indices
            si, ti = matching_rows(
                batch.unique_keys.to(torch.int64)[persisted],
                batch.unique_table_ids[persisted],
                keys,
                table_ids,
                batch.table_num,
            )
            values[ti] = batch.unique_values[persisted[si]]
            live[ti] = True
        # A finite backend may have evicted an in-flight key. Restore its live
        # value and current access score instead of initializing a different row.
        restore = live[missing_pos]
        restored_pos = missing_pos[restore]
        score_rows = (
            missing_scores
            if missing_scores is not None
            and missing_scores.numel() == missing_pos.numel()
            else None
        )
        if restored_pos.numel():
            self._storage.insert(
                keys[restored_pos],
                table_ids[restored_pos],
                values[restored_pos],
                score_rows[restore] if score_rows is not None else None,
            )
        keep = ~restore
        missing_pos = missing_pos[keep]
        return (
            missing_pos.numel(),
            missing_keys[keep],
            missing_pos,
            missing_tids[keep],
            score_rows[keep] if score_rows is not None else missing_scores,
            found | live,
            scores,
            values,
        )
