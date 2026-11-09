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

"""The waves: the deterministic insert as it was before the sorted kernels. TEST CODE ONLY.

``DEMB_DETERMINISM_MODE`` used to insert a batch one wave at a time: the keys were sorted by
(bucket, key) and wave ``k`` was the ``k``-th smallest key of every bucket, one launch of the
ordinary insert kernel per wave, with a host loop over the waves and reads of the wave count.
The library replaced that by ``table_insert_sorted`` / ``table_insert_and_evict_sorted``, which
give every bucket the same sequence of operations from one thread (see
src/table_operation/sorted_insert.cuh). This module keeps the old code, as module-level
functions of a table, as the ORACLE the tests compare the new inserts with: it blocks on the
device on purpose (``nonzero``, ``.item()``) and must never be imported by the package.

The functions are the code of ``LinearBucketTable._bucketize_and_pad``,
``_deterministic_insert``, ``_deterministic_insert_and_evict`` and
``_deterministic_insert_and_evict_with_overflow`` of the tree before the change
(355a1f61), verbatim, with these mechanical edits only: ``self`` is ``table``,
``self.bucketize_keys`` is the extension call it made, and the pre-filter of
``bucketize_and_pad`` also drops the keys of a table without buckets (see there).
"""

from typing import Optional, Tuple, Union

import torch
from dynamicemb import sync_free as sf
from dynamicemb_extensions import (
    ScorePolicy,
    bucketize_keys,
    table_insert,
    table_insert_and_evict,
    table_lookup,
)


def bucketize_and_pad(
    table,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    score_value: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Bucketize keys and pad into waves of non-conflicting inserts.

    Returns ``(keys_t, tids_t, scores_t)`` each shaped
    ``[max_bucket_len, num_buckets]``.  Each row ("wave") contains at most
    one key per bucket so that sequential per-wave inserts are
    deterministic.  Padding uses -1 for keys and 0 for tids/scores.

    Reserved all-ones keys (the padding of an N-aligned batch) are dropped
    first and take no part in any wave; callers look the keys up over their
    full arrays afterwards to get N-aligned indices.
    """
    # bucketize_keys reads table_ids[i] of every key, reserved or not (the
    # padding's table id is -1), and all reserved keys hash to one bucket, so
    # each of them would take a wave of its own. Drop them first. The read
    # this costs is made once, by indexing with the positions found, not by
    # a boolean mask per tensor; uint64 has no index_cuda, hence the views.
    keys_i64 = sf.as_i64(keys)
    # Not in the original: a key of a table without buckets is an Illegal no-op in a
    # wave, and bucketize_keys used to write its bookkeeping out of bounds for such a
    # key of a last table. Drop those keys with the reserved ones; the real keys and
    # their order do not change.
    buckets = table.table_bucket_offsets_[1:] - table.table_bucket_offsets_[:-1]
    takes_part = (keys_i64 != sf.EMPTY_KEY) & (buckets[table_ids] > 0)
    live = torch.nonzero(takes_part, as_tuple=True)[0]
    keys = keys_i64[live].view(keys.dtype)
    table_ids = table_ids[live]
    if score_value is not None:
        score_value = score_value.view(torch.int64)[live].view(torch.uint64)
    if keys.numel() == 0:
        return (
            torch.empty((0, 0), dtype=keys.dtype, device=keys.device),
            torch.empty((0, 0), dtype=torch.int64, device=keys.device),
            None,
        )

    bkt_keys, offsets, inverse = bucketize_keys(
        keys,
        table_ids,
        table.table_bucket_offsets_,
        table.num_buckets_,
        table.bucket_capacity_,
    )
    bkt_tids = table_ids[inverse]
    # uint64 doesn't support index_cuda; reinterpret as int64 for gather/scatter
    if score_value is not None:
        bkt_scores = score_value.view(torch.int64)[inverse].view(torch.uint64)
    else:
        bkt_scores = None

    num_buckets = offsets.numel() - 1
    lengths = offsets[1:] - offsets[:-1]
    max_len = int(lengths.max().item()) if num_buckets > 0 else 0

    if max_len == 0:
        empty_k = torch.empty((0, num_buckets), dtype=keys.dtype, device=keys.device)
        empty_t = torch.empty((0, num_buckets), dtype=torch.int64, device=keys.device)
        return empty_k, empty_t, None

    device = keys.device
    # offsets[-1] is the number of keys (the last segment ends at the batch
    # size), so the output size is known and neither call has to read it back.
    bucket_idx = torch.repeat_interleave(
        torch.arange(num_buckets, device=device),
        lengths,
        output_size=bkt_keys.numel(),
    )
    starts = torch.repeat_interleave(
        offsets[:-1], lengths, output_size=bkt_keys.numel()
    )
    positions = torch.arange(bkt_keys.numel(), device=device) - starts

    # The pad is built as int64 and viewed back: the scatter has no uint64 kernel.
    pad_keys = torch.full((num_buckets, max_len), -1, dtype=torch.int64, device=device)
    pad_tids = torch.zeros((num_buckets, max_len), dtype=torch.int64, device=device)
    pad_keys[bucket_idx, positions] = bkt_keys.view(torch.int64)
    pad_tids[bucket_idx, positions] = bkt_tids

    pad_scores = None
    if bkt_scores is not None:
        pad_scores = torch.zeros(
            (num_buckets, max_len), dtype=torch.int64, device=device
        )
        pad_scores[bucket_idx, positions] = bkt_scores.view(torch.int64)

    keys_t = pad_keys.transpose(0, 1).contiguous().view(keys.dtype)
    tids_t = pad_tids.transpose(0, 1).contiguous()
    scores_t = (
        pad_scores.transpose(0, 1).contiguous() if pad_scores is not None else None
    )
    return keys_t, tids_t, scores_t


def insert(
    table,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    score_value: Optional[torch.Tensor],
    policy: ScorePolicy,
) -> torch.Tensor:
    """Deterministic insert: bucketize, insert one wave at a time, then
    lookup to get final indices."""
    if keys.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=keys.device)

    keys_t, tids_t, scores_t = bucketize_and_pad(table, keys, table_ids, score_value)
    for i in range(keys_t.size(0)):
        # The rows are passed as padded: the pad key is reserved, so its lanes
        # take no part in the insert (Illegal), and there is nothing to read
        # back to find out which lanes are real.
        table_insert(
            table.table_storage_,
            table.table_bucket_offsets_,
            table.bucket_capacity_,
            table.bucket_sizes,
            keys_t[i],
            tids_t[i],
            scores_t[i].to(torch.uint64) if scores_t is not None else None,
            policy,
            table._ref_counter,
        )

    _, _, indices = table_lookup(
        table.table_storage_,
        table.table_bucket_offsets_,
        table.bucket_capacity_,
        keys,
        table_ids,
        None,
        ScorePolicy.CONST,
    )
    return indices


def insert_and_evict(
    table,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    score_value: Optional[torch.Tensor],
    policy: ScorePolicy,
    *,
    async_count: bool = False,
) -> Tuple[
    torch.Tensor,
    Union[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Deterministic insert_and_evict: bucketize, insert-and-evict one wave
    at a time, accumulate evicted data on the device, then lookup for final
    indices.
    Returns ``(indices, num_evicted, evicted_keys, evicted_indices,
    evicted_scores, evicted_table_ids)``: ``num_evicted`` is a host int and
    the evicted_* tensors are sliced to it, unless ``async_count``, which
    returns the unread int64[1] device count and evicted_* tensors of the
    length of ``keys`` whose tails past it are inert (key ``-1``, index and
    table id ``-1``, score 0)."""
    n = keys.numel()
    device = keys.device
    if n == 0:
        e = torch.empty(0, dtype=torch.int64, device=device)
        num_evicted = (
            torch.zeros(1, dtype=torch.int64, device=device) if async_count else 0
        )
        return (
            e,
            num_evicted,
            torch.empty_like(keys[:0]),
            e.clone(),
            e.clone(),
            e.clone(),
        )

    keys_t, tids_t, scores_t = bucketize_and_pad(table, keys, table_ids, score_value)

    # A wave has one lane per bucket and each lane evicts at most one key, so
    # a wave reports at most `width` evictions, in its first lanes. They are
    # appended on the device: the lanes past the count hold garbage and are
    # parked in [n, n + width), past anything returned, so that no wave has to
    # read its count back. The buffers are int64 (uint64 has no index_cuda);
    # keys and scores are viewed back when returned.
    width = keys_t.size(1)
    size = n + width
    evicted_keys_buf = torch.full((size,), -1, dtype=torch.int64, device=device)
    evicted_indices_buf = torch.full((size,), -1, dtype=torch.int64, device=device)
    evicted_scores_buf = torch.zeros(size, dtype=torch.int64, device=device)
    evicted_tids_buf = torch.full((size,), -1, dtype=torch.int64, device=device)
    num_evicted_accum = torch.zeros(1, dtype=torch.int64, device=device)
    lane = torch.arange(width, device=device)
    for i in range(keys_t.size(0)):
        (
            _wave_idx,
            num_ev,
            ev_keys,
            ev_indices,
            ev_scores,
            ev_tids,
        ) = table_insert_and_evict(
            table.table_storage_,
            table.table_bucket_offsets_,
            table.bucket_capacity_,
            table.bucket_sizes,
            keys_t[i],
            tids_t[i],
            scores_t[i].to(torch.uint64) if scores_t is not None else None,
            policy,
            table._ref_counter,
        )
        dst = torch.where(lane < num_ev, num_evicted_accum + lane, n + lane)
        evicted_keys_buf[dst] = ev_keys.view(torch.int64)
        evicted_indices_buf[dst] = ev_indices
        evicted_scores_buf[dst] = ev_scores
        evicted_tids_buf[dst] = ev_tids
        num_evicted_accum += num_ev

    _, _, indices = table_lookup(
        table.table_storage_,
        table.table_bucket_offsets_,
        table.bucket_capacity_,
        keys,
        table_ids,
        None,
        ScorePolicy.CONST,
    )
    evicted_keys_buf = evicted_keys_buf.view(keys.dtype)
    evicted_scores_buf = evicted_scores_buf.view(torch.uint64)
    if async_count:
        return (
            indices,
            num_evicted_accum,
            evicted_keys_buf[:n],
            evicted_indices_buf[:n],
            evicted_scores_buf[:n],
            evicted_tids_buf[:n],
        )
    h_num_evicted = num_evicted_accum.item()
    return (
        indices,
        h_num_evicted,
        evicted_keys_buf[:h_num_evicted],
        evicted_indices_buf[:h_num_evicted],
        evicted_scores_buf[:h_num_evicted],
        evicted_tids_buf[:h_num_evicted],
    )


def insert_and_evict_with_overflow(
    table,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    score_value: Optional[torch.Tensor],
    policy: ScorePolicy,
    *,
    async_count: bool = False,
) -> Tuple[
    torch.Tensor,
    Union[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Like ``_deterministic_insert_and_evict`` but passes overflow
    parameters to ``table_insert_and_evict`` and uses
    ``lookup_with_overflow``-style lookup for final indices; ``async_count``
    means the same."""
    n = keys.numel()
    device = keys.device
    if n == 0:
        e = torch.empty(0, dtype=torch.int64, device=device)
        num_evicted = (
            torch.zeros(1, dtype=torch.int64, device=device) if async_count else 0
        )
        return (
            e,
            num_evicted,
            torch.empty_like(keys[:0]),
            e.clone(),
            e.clone(),
            e.clone(),
        )

    keys_t, tids_t, scores_t = bucketize_and_pad(table, keys, table_ids, score_value)

    # See _deterministic_insert_and_evict: evictions are appended on the
    # device, the lanes without one are parked in [n, n + width).
    width = keys_t.size(1)
    size = n + width
    evicted_keys_buf = torch.full((size,), -1, dtype=torch.int64, device=device)
    evicted_indices_buf = torch.full((size,), -1, dtype=torch.int64, device=device)
    evicted_scores_buf = torch.zeros(size, dtype=torch.int64, device=device)
    evicted_tids_buf = torch.full((size,), -1, dtype=torch.int64, device=device)
    num_evicted_accum = torch.zeros(1, dtype=torch.int64, device=device)
    lane = torch.arange(width, device=device)
    for i in range(keys_t.size(0)):
        (
            _wave_idx,
            num_ev,
            ev_keys,
            ev_indices,
            ev_scores,
            ev_tids,
        ) = table_insert_and_evict(
            table.table_storage_,
            table.table_bucket_offsets_,
            table.bucket_capacity_,
            table.bucket_sizes,
            keys_t[i],
            tids_t[i],
            scores_t[i].to(torch.uint64) if scores_t is not None else None,
            policy,
            table._ref_counter,
            ovf_storage=table.overflow_table_storage_,
            ovf_bucket_capacity=table.overflow_bucket_capacity_,
            ovf_bucket_sizes=table.overflow_bucket_sizes,
            ovf_counter=table._ovf_counter,
            ovf_output_offsets=table.overflow_output_offsets_,
        )
        dst = torch.where(lane < num_ev, num_evicted_accum + lane, n + lane)
        evicted_keys_buf[dst] = ev_keys.view(torch.int64)
        evicted_indices_buf[dst] = ev_indices
        evicted_scores_buf[dst] = ev_scores
        evicted_tids_buf[dst] = ev_tids
        num_evicted_accum += num_ev

    _, _, indices = table_lookup(
        table.table_storage_,
        table.table_bucket_offsets_,
        table.bucket_capacity_,
        keys,
        table_ids,
        None,
        ScorePolicy.CONST,
        ovf_storage=table.overflow_table_storage_,
        ovf_bucket_capacity=table.overflow_bucket_capacity_,
        ovf_output_offsets=table.overflow_output_offsets_,
    )
    evicted_keys_buf = evicted_keys_buf.view(keys.dtype)
    evicted_scores_buf = evicted_scores_buf.view(torch.uint64)
    if async_count:
        return (
            indices,
            num_evicted_accum,
            evicted_keys_buf[:n],
            evicted_indices_buf[:n],
            evicted_scores_buf[:n],
            evicted_tids_buf[:n],
        )
    h_num_evicted = num_evicted_accum.item()
    return (
        indices,
        h_num_evicted,
        evicted_keys_buf[:h_num_evicted],
        evicted_indices_buf[:h_num_evicted],
        evicted_scores_buf[:h_num_evicted],
        evicted_tids_buf[:h_num_evicted],
    )
