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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, ContextManager, Optional, Tuple

import torch
from dynamicemb.async_erase import table_section
from dynamicemb.dynamicemb_config import EvictedItemMode
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.key_value_table import (
    Cache,
    DelayedCopy,
    DynamicEmbCache,
    DynamicEmbStorage,
    DynamicEmbTableState,
    Storage,
    _append_evicted,
    _find_keys,
    _insert_and_evict_keys,
    _lookup_keys,
    _write_back_evicted,
    eval_lookup,
    get_insert_score_arg,
    load_from_flat,
    score_source,
    store_to_flat,
)
from dynamicemb.lookup_layout import EmbeddingLayout
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import CopyMode, MultiTableAdmitter
from dynamicemb_extensions import (
    EvictStrategy,
    counted_gather,
    counted_scatter,
    counted_scatter_fill,
    expand_table_ids_cuda,
    flagged_compact,
    gather_embedding,
    gather_embedding_pooled,
    get_table_range,
    reduce_grads,
    segmented_unique_cuda,
    select_insert_failed_values,
)


def _copy_cuda_tensor_to_pinned_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda:
        return tensor.detach().cpu()
    cpu = torch.empty(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=torch.device("cpu"),
        pin_memory=True,
    )
    cpu.copy_(tensor.detach(), non_blocking=True)
    torch.cuda.current_stream(tensor.device).synchronize()
    return cpu


def _scalar_item(tensor: torch.Tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.numel() != 1:
        raise ValueError(f"expected scalar tensor, got shape {tuple(tensor.shape)}")
    return _copy_cuda_tensor_to_pinned_cpu(tensor.reshape(1))[0].item()


def _bool_item(tensor: torch.Tensor) -> bool:
    return bool(_scalar_item(tensor))


def segmented_unique(
    keys: torch.Tensor,
    segment_range: torch.Tensor,
    evict_strategy: Optional[EvictStrategy] = None,
    frequency_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Perform segmented unique operation on keys with segment_range.

    This function deduplicates keys within each table segment, using the
    GPU-accelerated segmented_unique_cuda kernel.

    Args:
        keys: Input key tensor (int64 or uint64)
        segment_range: Table boundary offsets where segment_range[i] is the
                       start index for table i (int64)
        evict_strategy: Optional eviction strategy (for LFU mode)
        frequency_counts: Optional input frequency counts per key

    Returns:
        Tuple of (unique_keys, reverse_indices, unique_keys_table_range,
                  output_scores).
        ``unique_keys`` and ``output_scores`` keep the length of ``keys``: only their
        first ``unique_keys_table_range[-1]`` entries are live (see
        :class:`dynamicemb.types.Storage`). ``unique_keys_table_range`` is on the
        device and has one entry more than there are tables: the unique keys of table
        ``i`` are those from ``unique_keys_table_range[i]`` to
        ``unique_keys_table_range[i + 1]``, and the last entry is the number of unique
        keys. Nothing is read back from the device.
    """
    with torch.cuda.nvtx.range("op:segmented_unique"):
        num_keys = keys.size(0)
        num_tables = segment_range.size(0) - 1
        device = keys.device

        if num_keys == 0:
            empty_keys = torch.empty(0, dtype=keys.dtype, device=device)
            empty_reverse_indices = torch.empty(0, dtype=torch.int64, device=device)
            d_table_range = torch.zeros(
                num_tables + 1, dtype=torch.int64, device=device
            )
            return empty_keys, empty_reverse_indices, d_table_range, None

        is_lfu_enabled = (
            evict_strategy == EvictStrategy.KLfu if evict_strategy else False
        )
        need_frequency_output = is_lfu_enabled or frequency_counts is not None

        input_frequencies = None
        if frequency_counts is not None:
            input_frequencies = frequency_counts
        elif need_frequency_output:
            input_frequencies = torch.empty(0, dtype=torch.int64, device=device)

        (
            _num_uniques,
            unique_keys,
            reverse_indices,
            table_offsets,
            freq_counters,
        ) = segmented_unique_cuda(keys, segment_range, num_tables, input_frequencies)

        output_scores = freq_counters if need_frequency_output else None
        return unique_keys, reverse_indices, table_offsets, output_scores


def largest_score_per_unique_key(
    scores: torch.Tensor, reverse_indices: torch.Tensor, num_unique: int
) -> torch.Tensor:
    """The score of each unique key: the largest of those its occurrences had.

    ``reverse_indices[i]`` is the unique key that input key ``i`` was merged
    into; a key that was not repeated keeps its own score.
    """
    unique_scores = torch.zeros(num_unique, dtype=torch.int64, device=scores.device)
    return unique_scores.scatter_reduce_(
        0, reverse_indices, scores, reduce="amax", include_self=False
    )


class StorageMode(Enum):
    DEFAULT = auto()
    CACHE = auto()
    HBM_DIRECT = auto()


@dataclass
class PrefetchState:
    unique_keys: torch.Tensor
    reverse_indices: torch.Tensor
    unique_table_ids: torch.Tensor
    lfu_accumulated_frequency: torch.Tensor
    storage_mode: StorageMode
    slot_indices: Optional[torch.Tensor]
    # How many leading entries of the per-key arrays of this state are live, as a
    # one-element int64 device tensor: the arrays keep the length of the input and what
    # lies past ``num_unique`` is undefined (see dynamicemb.types.Storage).
    num_unique: torch.Tensor
    # Value-buffer row per key -- what forward and backward both address values
    # with. It differs from ``slot_indices`` for a table that stores a row per
    # key (a storage): rows are handed out densely while slots come from the
    # key's hash, and the key map of a table that never evicts is deliberately
    # larger than its value buffer, so indexing values by slot would read past the
    # end of it. A cache points this at the slot tensor itself rather than leaving it unset, so a
    # consumer never has to ask which of the two it is holding. Anything indexed
    # by hash slot (the ref counter) uses ``slot_indices``.
    value_row_indices: Optional[torch.Tensor] = None
    non_admitted_positions: Optional[torch.Tensor] = None
    # One int64 per unique key, for tables that take a score per key.
    unique_scores: Optional[torch.Tensor] = None
    # Keys pinned: an upper bound until settled (see
    # ``BatchedDynamicEmbeddingTablesV2._settle_outstanding_keys``).
    num_prefetched_keys: int = 0
    outstanding_keys_ref: Optional[torch.Tensor] = None
    # The real number of unique keys on its way to the host, asked for by a batch that is
    # counted by its bound when the sum is past half of the cache: the bound is replaced
    # by it once it has landed.
    unique_count: Optional[DelayedCopy] = None
    # The keys the admitter refused: their positions among the unique keys and how many.
    non_admitted_count: Optional[torch.Tensor] = None
    # The key_map_generation (rehash count) of the table state the slots were found in: a later
    # prefetch may expand it before this batch's backward runs.
    key_map_generation: int = 0


def refresh_slots_after_rehash(state: DynamicEmbTableState, batch) -> None:
    """Bring a batch's slots up to date if a rehash moved its keys.

    ``batch`` is anything that holds what a prefetch found: a ``PrefetchState``,
    or the ``ctx`` of ``DynamicEmbeddingFunction`` that took it over (``unique_keys``,
    ``unique_table_ids``, ``slot_indices``, ``key_map_generation`` and ``num_unique``,
    the number of live entries of the arrays). If ``state`` has rehashed since (a
    later batch's prefetch may have, before this one's backward or its no-grad
    forward), the keys are looked up again, read only (their ref counts kept them in
    the table), and ``batch.slot_indices`` and ``batch.key_map_generation`` are
    replaced. A key the prefetch did not admit stays without a slot. Value rows do
    not move: a storage keeps each key's row through a rehash, and a cache does not
    rehash.
    """
    if batch.key_map_generation == state.key_map_generation:
        return
    # Only a table that stores a row per key keeps its value rows through a
    # rehash (``use_rows_``). Without it the row is the slot and the value moves
    # with the key, so ``value_row_indices`` would go stale too: not handled here.
    assert state.key_index_map.use_rows_, (
        "a rehash since the prefetch is only handled for a table that stores a "
        "value row per key (DynamicEmbStorage); a cache does not rehash"
    )
    slots = _lookup_keys(
        state,
        batch.unique_keys,
        batch.unique_table_ids,
        batch.num_unique,
        const_lookup=True,
        return_slots=True,
    )[-1]
    batch.slot_indices = torch.where(batch.slot_indices >= 0, slots, batch.slot_indices)
    batch.key_map_generation = state.key_map_generation


def _is_hbm_storage(storage: Storage) -> bool:
    """True if values live in GPU device memory (HBM). Use buffer type, not
    tensor.is_cuda: host memory registered to CUDA address space can report is_cuda True.
    """
    return (
        isinstance(storage, DynamicEmbStorage)
        and storage._state.tables[0].is_device_buffer()
    )


def _apply_admission(
    missing_keys: torch.Tensor,
    missing_indices: torch.Tensor,
    missing_table_ids: torch.Tensor,
    missing_scores: Optional[torch.Tensor],
    values: torch.Tensor,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    freq_for_admission: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    num_missing: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply admission filtering for missing keys.

    If an admission strategy is active, runs ``non_admitted_initializer`` over
    the rows it rejects when the strategy asked for one, then compacts
    keys/scores/table_ids/indices to only the admitted subset.

    Returns ``(num_to_insert, keys_to_insert, scores_to_insert,
    table_ids_to_insert, positions_in_unique, num_to_init, indices_to_init)``
    where ``indices_to_init`` are the positions in values that the caller should
    initialize with its embeddings initializer, ``num_to_init`` of them. Every
    missing row is covered exactly once: the non-admitted ones drop out of
    indices_to_init exactly when this function has already run the strategy's
    own initializer over them.
    """
    with torch.cuda.nvtx.range("_apply_admission"):
        if admitter is None:
            return (
                num_missing,
                missing_keys,
                missing_scores,
                missing_table_ids,
                missing_indices,
                num_missing,
                missing_indices,
            )
        admit_mask = admitter.admit(
            missing_keys, missing_table_ids, freq_for_admission, num_valid=num_missing
        )
        non_admitted_initializer = admitter.non_admitted_initializer
        if non_admitted_initializer is not None:
            num_refused, _, (non_admitted_indices,) = flagged_compact(
                ~admit_mask, [missing_indices], num_missing
            )
            with torch.cuda.nvtx.range("op:initializer"):
                non_admitted_initializer(
                    values[:, :max_emb_dim],
                    # Alongside ``values``, which is why these are the whole
                    # unique vectors and not the missing slice; the rows to
                    # write come last.
                    unique_keys,
                    unique_table_ids,
                    non_admitted_indices,
                    num_valid=num_refused,
                )
        with torch.cuda.nvtx.range("op:compact"):
            (
                num_admitted,
                _,
                (
                    keys_to_insert,
                    positions_in_unique,
                    table_ids_to_insert,
                    scores_to_insert,
                ),
            ) = flagged_compact(
                admit_mask,
                [missing_keys, missing_indices, missing_table_ids, missing_scores],
                num_missing,
            )
        # Whoever owns the non-admitted rows was settled at construction. When
        # the strategy asked for its own initializer the table's only has to
        # cover the admitted rows; otherwise it has to cover every miss, or the
        # non-admitted rows reach the forward holding whatever
        # ``storage.find`` left in its ``torch.empty`` value buffer.
        if non_admitted_initializer is not None:
            num_to_init, indices_to_init = num_admitted, positions_in_unique
        else:
            num_to_init, indices_to_init = num_missing, missing_indices
        return (
            num_admitted,
            keys_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
            num_to_init,
            indices_to_init,
        )


def _lookup_next_tier(
    next_tier: Optional[Storage],
    miss_keys: torch.Tensor,
    miss_tids: torch.Tensor,
    in_storage: torch.Tensor,
    values: torch.Tensor,
    num_miss: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Ask ``next_tier`` for the misses storage lacks; fill their ``values``.

    Returns the mask (over the misses) of those it had, or None when there is no
    tier below storage.
    """
    if next_tier is None:
        return None
    in_next = torch.zeros_like(in_storage)
    with torch.cuda.nvtx.range("op:compact"):
        num_asked, asked, (asked_keys, asked_tids) = flagged_compact(
            ~in_storage, [miss_keys, miss_tids], num_miss
        )
    with torch.cuda.nvtx.range("op:next_tier_find"):
        result = next_tier.find(
            asked_keys, asked_tids, copy_mode=CopyMode.VALUE, num_valid=num_asked
        )
    found, found_values = result[5], result[7]
    # ``asked[j]`` is the miss that the j-th asked key is; one the tier lacks has no
    # row to write.
    hit = torch.where(found, asked, -1)
    counted_scatter_fill(in_next, hit, True, num_asked)
    counted_scatter(values, hit, found_values, num_asked)
    return in_next


def _insert_into_storage(
    storage: DynamicEmbStorage,
    next_tier: Optional[Storage],
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    values: torch.Tensor,
    scores: Optional[torch.Tensor],
    num_valid: torch.Tensor,
) -> None:
    """Insert new keys into ``storage``; what it evicts for room moves to
    ``next_tier``."""
    if next_tier is None:
        storage.insert(keys, table_ids, values, scores, num_valid=num_valid)
        return
    # NO_EVICTION is deliberately not handled below: its storage never evicts
    # (it expands instead), so there is nothing to spill. The tables module
    # rejects NO_EVICTION together with an ``external_storage``.
    st = storage._state
    (
        slots,
        num_evicted,
        evicted_keys,
        evicted_table_ids,
        evicted_slots,
        _,
    ) = _insert_and_evict_keys(st, keys, table_ids, scores, num_valid)
    evicted_values = load_from_flat(
        st,
        evicted_slots,
        evicted_table_ids,
        copy_mode=CopyMode.VALUE,
        num_valid=num_evicted,
    )
    # A key the storage refused (e.g. its bucket is full and the score does
    # not beat any stored one) comes back as an "evicted" row with a negative
    # slot: load_from_flat left its value unwritten, so take the key's own
    # input value instead, as HybridStorage.insert does.
    select_insert_failed_values(
        evicted_slots, values, evicted_values, num_valid=num_evicted
    )
    store_to_flat(st, slots, table_ids, values, num_valid=num_valid)
    next_tier.insert(
        evicted_keys, evicted_table_ids, evicted_values, num_valid=num_evicted
    )


def _prefetch_cache_path(
    cache: DynamicEmbCache,
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    num_unique: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    initializer: MultiTableInitializer,
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    external_storage: Optional[Storage] = None,
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Cache prefetch with counter protection and overflow buffer.

    ``storage`` is the tier that owns every key's score; the cache above it
    ranks by timestamp only, and holds the freshest value of a key until it is
    written back. When ``external_storage`` is given (a
    score-less store below ``storage``, e.g. a PS) it holds the keys ``storage`` has
    evicted: keys missing from ``storage`` are fetched from it, and what
    ``storage`` evicts is spilled to it.

    Only admitted keys are inserted into the cache.  Non-admitted keys get
    slot_indices = -1 and their positions (in the unique_keys array) are
    returned so the forward pass can lazily initialize their embeddings.

    Index spaces: ``unique_*`` arrays are indexed by position in the unique batch;
    everything after the cache lookup is over the *misses* only, indexed by miss
    number, and ``miss_pos`` maps a miss number back to its position in the
    unique batch. The misses are compacted into arrays of the capacity of the
    input with a count of their own, ``num_miss``, and every later step takes the
    count of the arrays it is given. Nothing is skipped on a count: a step whose
    keys all hit runs the miss path for zero keys.

    Returns ``(slot_indices, non_admitted_positions, num_non_admitted)``.
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path"):
        device = unique_keys.device
        state = cache._state
        capacity = unique_keys.numel()
        if capacity == 0:
            return torch.empty(0, dtype=torch.int64, device=device), None, None

        # 1. Lookup with overflow fallback; pin what is found. A key that is not in the
        # cache has slot -1, which the counter kernel skips. ``slot_indices`` is the
        # array the lookup allocated; the later steps write the slots of the misses
        # into it.
        with torch.cuda.nvtx.range("op:cache_lookup"):
            _, founds, slot_indices = cache.lookup(
                unique_keys, unique_table_ids, num_valid=num_unique
            )
        cache.increment_counter(slot_indices, unique_table_ids, num_valid=num_unique)

        # 2. Storage is the only place scores live: a key-only lookup of *every*
        # key -- hits included, and even when nothing missed -- updates its score
        # there without moving a value. The cache only ranks by timestamp.
        # ``score_input`` is what the storage's score is made from, chosen once
        # here and carried with the keys from now on; ``accumulated_frequency``
        # is only for admission below.
        score_input = score_source(storage._state, accumulated_frequency, key_scores)
        with torch.cuda.nvtx.range("op:storage_find"):
            _, _, st_founds, st_rows, _ = _lookup_keys(
                storage._state,
                unique_keys,
                unique_table_ids,
                num_unique,
                score_input,
            )

        # 3. The misses, with everything later steps need about them.
        with torch.cuda.nvtx.range("op:compact"):
            (
                num_miss,
                miss_pos,
                (miss_keys, miss_tids, miss_freq, miss_score_input, miss_rows),
            ) = flagged_compact(
                ~founds,
                [
                    unique_keys,
                    unique_table_ids,
                    accumulated_frequency,
                    score_input,
                    st_rows,
                ],
                num_unique,
            )

        # 4. Where each miss lives: in storage (its row is loaded), in the tier below
        # it (its value is fetched), or nowhere (new). ``values`` holds the first two's
        # rows; new ones are filled in at 6.
        in_storage = counted_gather(st_founds, miss_pos, num_miss)
        values = load_from_flat(
            storage._state,
            miss_rows,
            miss_tids,
            copy_mode=CopyMode.VALUE,
            num_valid=num_miss,
        )
        in_next = _lookup_next_tier(
            external_storage, miss_keys, miss_tids, in_storage, values, num_miss
        )
        held = in_storage if in_next is None else in_storage | in_next

        # 5. Admit the new ones (a key that storage or the tier below it has was
        # admitted before). ``admitted`` starts as the new ones; the admitter's answer
        # for them is written into it.
        admitted = ~held
        non_admitted_positions: Optional[torch.Tensor] = None
        num_refused: Optional[torch.Tensor] = None
        if admitter is not None:
            with torch.cuda.nvtx.range("op:compact"):
                (
                    num_new,
                    new_rows,
                    (new_keys, new_tids, new_freq, new_pos),
                ) = flagged_compact(
                    admitted, [miss_keys, miss_tids, miss_freq, miss_pos], num_miss
                )
            admit_mask = admitter.admit(new_keys, new_tids, new_freq, num_valid=num_new)
            counted_scatter(admitted, new_rows, admit_mask, num_new)
            with torch.cuda.nvtx.range("op:compact"):
                num_refused, _, (refused_pos,) = flagged_compact(
                    ~admit_mask, [new_pos], num_new
                )
            non_admitted_positions = refused_pos

        to_cache = held | admitted
        with torch.cuda.nvtx.range("op:compact"):
            (
                num_cache,
                into_cache,
                (cache_keys, cache_tids, cache_pos),
            ) = flagged_compact(to_cache, [miss_keys, miss_tids, miss_pos], num_miss)

        # 6. Initialize the admitted-new rows (in place) and put what storage lacks
        # into it -- with its score -- before the cache sees it.
        with torch.cuda.nvtx.range("op:compact"):
            num_adm, adm_rows, _ = flagged_compact(admitted, [], num_miss)
        with torch.cuda.nvtx.range("op:initializer"):
            initializer(
                values[:, :max_emb_dim],
                miss_keys,
                miss_tids,
                adm_rows,
                num_valid=num_adm,
            )
        if max_val_dim != max_emb_dim:
            state.optimizer.reset_optimizer_states(
                values[:, max_emb_dim:],
                indices=adm_rows,
                table_ids=miss_tids,
                table_emb_dims=state.table_emb_dims,
                num_valid=num_adm,
            )
        with torch.cuda.nvtx.range("op:compact"):
            (
                num_st,
                into_storage,
                (st_keys, st_tids, st_score_input),
            ) = flagged_compact(
                admitted if in_next is None else in_next | admitted,
                [miss_keys, miss_tids, miss_score_input],
                num_miss,
            )
        with torch.cuda.nvtx.range("op:storage_insert"):
            _insert_into_storage(
                storage,
                external_storage,
                st_keys,
                st_tids,
                counted_gather(values, into_storage, num_st),
                st_score_input,
                num_st,
            )

        # 7. Bring the keys into the cache, ranked by timestamp only
        with torch.cuda.nvtx.range("op:cache_insert_evict"):
            (
                cache_slots,
                num_evicted,
                evicted_keys,
                evicted_slots,
                _,
                evicted_tids,
            ) = cache.insert_and_evict(
                cache_keys, cache_tids, None, num_valid=num_cache
            )

        # 8. What it evicted: read before the slots are overwritten, then write back
        with torch.cuda.nvtx.range("op:load_from_flat"):
            evicted_values = load_from_flat(
                state,
                evicted_slots,
                evicted_tids,
                copy_mode=CopyMode.VALUE,
                num_valid=num_evicted,
            )
        with torch.cuda.nvtx.range("op:store_to_flat"):
            store_to_flat(
                state,
                cache_slots,
                cache_tids,
                counted_gather(values, into_cache, num_cache),
                num_valid=num_cache,
            )
        _write_back_evicted(
            storage,
            evicted_keys,
            evicted_tids,
            evicted_values,
            num_evicted,
            next_tier=external_storage,
        )

        # 9. Counter & slot mapping
        cache.increment_counter(cache_slots, cache_tids, num_valid=num_cache)
        counted_scatter(slot_indices, cache_pos, cache_slots, num_cache)
        return slot_indices, non_admitted_positions, num_refused


def _prefetch_hbm_direct_path(
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    num_unique: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    initializer: MultiTableInitializer,
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """HBM-direct prefetch with counter protection.

    Only admitted keys are inserted.  Non-admitted keys get slot_indices = -1
    and their positions are returned for lazy initialization at forward time.

    Returns ``(slot_indices, non_admitted_positions, num_non_admitted,
    value_row_indices)``. The two index spaces are kept apart because they only
    coincide by accident: ``slot_indices`` addresses the hash table (and the ref
    counter beside it), ``value_row_indices`` addresses the flat value buffer. A
    storage stores a row per key, so they differ: rows are handed out densely
    while slots come from the key's hash, and the key map of a table that never
    evicts is sized ``1 / load factor`` times the value buffer, so a slot can
    point past the end of it. A table whose row is its slot returns the slot
    tensor itself as the rows, so the caller always gets something it can address
    values with.
    """
    with torch.cuda.nvtx.range("_prefetch_hbm_direct_path"):
        state = storage._state
        device = unique_keys.device
        capacity = unique_keys.numel()
        if capacity == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, None, None, empty

        with torch.cuda.nvtx.range("op:storage_find"):
            (
                num_missing,
                missing_keys,
                missing_indices,
                missing_table_ids,
                missing_scores,
                _,
                _,
                rows,
                slot_indices,
            ) = _find_keys(
                state,
                unique_keys,
                unique_table_ids,
                num_unique,
                score_source(state, accumulated_frequency, key_scores),
                return_slots=True,
            )

        # The ref counter is keyed by slot, the values by row: a storage stores a row
        # per key, so these are different tensors and both are kept up to date. A key
        # that was not found has slot -1, which the counter kernel skips.
        storage.increment_counter(slot_indices, unique_table_ids, num_valid=num_unique)

        non_admitted_positions: Optional[torch.Tensor] = None
        num_refused: Optional[torch.Tensor] = None
        if admitter is not None:
            freq_for_admission = (
                counted_gather(accumulated_frequency, missing_indices, num_missing)
                if accumulated_frequency is not None
                else None
            )
            admit_mask = admitter.admit(
                missing_keys,
                missing_table_ids,
                freq_for_admission,
                num_valid=num_missing,
            )
            with torch.cuda.nvtx.range("op:compact"):
                (
                    num_admitted,
                    _,
                    (
                        admitted_keys,
                        admitted_tids,
                        admitted_scores,
                        admitted_unique_positions,
                    ),
                ) = flagged_compact(
                    admit_mask,
                    [missing_keys, missing_table_ids, missing_scores, missing_indices],
                    num_missing,
                )
                num_refused, _, (non_admitted_positions,) = flagged_compact(
                    ~admit_mask, [missing_indices], num_missing
                )
        else:
            num_admitted = num_missing
            admitted_keys = missing_keys
            admitted_tids = missing_table_ids
            admitted_scores = missing_scores
            admitted_unique_positions = missing_indices

        # Initialize and insert the admitted keys
        init_values = torch.empty(
            capacity, max_val_dim, dtype=state.emb_dtype, device=device
        )
        with torch.cuda.nvtx.range("op:initializer"):
            initializer(
                init_values[:, :max_emb_dim],
                admitted_keys,
                admitted_tids,
                None,
                num_valid=num_admitted,
            )
        if max_val_dim != max_emb_dim:
            state.optimizer.reset_optimizer_states(
                init_values[:, max_emb_dim:],
                table_ids=admitted_tids,
                table_emb_dims=state.table_emb_dims,
                num_valid=num_admitted,
            )
        score_arg = get_insert_score_arg(state, capacity, device, admitted_scores)
        with torch.cuda.nvtx.range("op:storage_insert"):
            # HBM-direct is a last tier: this insert may evict, and the victims are
            # dropped (retained when configured, which reads their count: opt-in).
            # This is the forward-path analogue of the _insert_key_values collection.
            if EvictedItemMode.RETAIN_KEY in state.evicted_item_mode:
                (
                    new_indices,
                    num_evicted,
                    evicted_keys,
                    evicted_table_ids,
                    new_slots,
                ) = state.key_index_map.insert(
                    admitted_keys,
                    admitted_tids,
                    score_arg,
                    collect_evicted=True,
                    return_slots=True,
                    num_valid=num_admitted,
                )
                _append_evicted(state, evicted_keys, evicted_table_ids, num_evicted)
            else:
                new_indices, new_slots = state.key_index_map.insert(
                    admitted_keys,
                    admitted_tids,
                    score_arg,
                    return_slots=True,
                    num_valid=num_admitted,
                )
        # The insert returns rows, and the slots as well. Keep both: the counter is
        # indexed by slot, the value buffer by row.
        new_rows = new_indices
        with torch.cuda.nvtx.range("op:store_to_flat"):
            store_to_flat(
                state, new_rows, admitted_tids, init_values, num_valid=num_admitted
            )
        storage.increment_counter(new_slots, admitted_tids, num_valid=num_admitted)

        counted_scatter(
            slot_indices, admitted_unique_positions, new_slots, num_admitted
        )
        counted_scatter(rows, admitted_unique_positions, new_rows, num_admitted)
        return slot_indices, non_admitted_positions, num_refused, rows


def dynamicemb_prefetch(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    cache: Optional[Cache],
    storage: Storage,
    feature_offsets: torch.Tensor,
    initializer: MultiTableInitializer,
    forward_stream: Optional[torch.cuda.Stream] = None,
    evict_strategy=None,
    frequency_counters: Optional[torch.Tensor] = None,
    admitter: Optional[MultiTableAdmitter] = None,
    outstanding_keys_ref: Optional[torch.Tensor] = None,
    external_storage: Optional[Storage] = None,
    scores: Optional[torch.Tensor] = None,
    table_gate: Optional[Callable[[], ContextManager]] = None,
    outstanding_settle: Optional[Callable[[torch.Tensor, int], int]] = None,
) -> PrefetchState:
    """Unified prefetch for all storage types (cache, HBM-direct, generic).

    Returns a PrefetchState containing unique embeddings + optimizer states
    and the metadata needed by forward/backward.

    The host reads the number of unique keys only for a table that can still grow,
    which needs it to decide whether to expand.

    ``scores`` are one int64 per key, for tables that take a score per key. They
    follow the keys through the dedup on a channel of their own, beside the
    frequency counters: where a key repeats in the batch, the largest of its
    scores is the one the table gets.

    ``table_gate`` is what an erase running beside the training takes too (see
    ``async_erase.TableAccessGate``). Prefetch holds it while it reads or changes the
    tables -- from the expansion through the find and the counter increment -- so
    that a key it found is pinned before an erase can look at it.

    ``outstanding_settle(num_unique, bound)`` is asked when the keys that prefetched
    batches pin in the cache add up, by the host's bounds, to more than the cache holds
    (see ``BatchedDynamicEmbeddingTablesV2._settle_outstanding_keys``): it replaces the
    bounds of the batches in flight by their real counts where these have landed on the
    host already, and if that does not settle it reads the real count of this batch,
    ``num_unique``, from the device and returns it (else it returns the bound), or
    raises if the cache really is over-subscribed. It runs before the cache is touched,
    and is required whenever ``outstanding_keys_ref`` is given. A batch that is counted
    by its bound when the sum is past half of the cache has its real count copied to the
    host without waiting (``PrefetchState.unique_count``), for a later check.
    """
    with torch.cuda.nvtx.range("dynamicemb_prefetch"):
        table_num = feature_offsets.numel() - 1
        assert table_num != 0
        max_emb_dim = storage.max_embedding_dim()
        max_val_dim = storage.max_value_dim()
        caching = cache is not None

        evict_strat = EvictStrategy(evict_strategy.value) if evict_strategy else None
        # todo: double check
        frequency_counts_int64 = None
        if frequency_counters is not None:
            frequency_counts_int64 = frequency_counters.long()

        if hasattr(storage, "collect_table_sizes"):
            storage.collect_table_sizes(non_blocking=True)

        # The number of unique keys stays on the device. The host reads it (a read that
        # waits for the device) only to decide whether a table has to grow.
        may_grow = (
            hasattr(storage, "expand_if_need")
            and getattr(storage, "can_expand", lambda: True)()
        )

        indices_table_range = get_table_range(offsets, feature_offsets)
        (
            unique_keys,
            reverse_indices,
            unique_indices_table_range,
            lfu_accumulated_frequency,
        ) = segmented_unique(
            indices,
            indices_table_range,
            evict_strat,
            frequency_counts_int64,
        )
        if may_grow:
            # The one read of the step: it waits for the device.
            range_cpu = _copy_cuda_tensor_to_pinned_cpu(unique_indices_table_range)
            unique_per_table = range_cpu[1:] - range_cpu[:-1]
        unique_scores = (
            None
            if scores is None
            else largest_score_per_unique_key(
                scores, reverse_indices, unique_keys.numel()
            )
        )

        # What this prefetch holds pinned: the number of input keys, which is never
        # less than the number of unique ones; the real count replaces it when the
        # cache looks over-subscribed (see
        # ``BatchedDynamicEmbeddingTablesV2._settle_outstanding_keys``).
        num_prefetched_keys = unique_keys.numel()

        # The live part of the arrays.
        num_unique = unique_indices_table_range[table_num:]  # a view

        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range, unique_keys.numel()
        )

        slot_indices = None
        non_admitted_positions = None
        non_admitted_count = None
        value_row_indices = None
        unique_count = None
        key_map_generation = 0
        if caching:
            storage_mode = StorageMode.CACHE
        elif _is_hbm_storage(storage):
            storage_mode = StorageMode.HBM_DIRECT
        else:
            storage_mode = StorageMode.DEFAULT

        with table_section(table_gate):
            if may_grow:
                storage.expand_if_need(unique_per_table)

            if storage_mode == StorageMode.CACHE:
                if outstanding_keys_ref is not None:
                    outstanding_keys_ref += num_prefetched_keys
                    outstanding = _scalar_item(outstanding_keys_ref)
                    bound = num_prefetched_keys
                    if outstanding > cache._state.capacity:
                        num_prefetched_keys = outstanding_settle(num_unique, bound)
                    if (
                        num_prefetched_keys == bound
                        and outstanding > cache._state.capacity // 2
                    ):
                        # The bound stands and the sum is past half of the cache: the
                        # real count goes to the host without waiting, for a later check
                        # to use if it has landed by then.
                        unique_count = DelayedCopy(num_unique)
                (
                    slot_indices,
                    non_admitted_positions,
                    non_admitted_count,
                ) = _prefetch_cache_path(
                    cache,
                    storage,
                    unique_keys,
                    unique_table_ids,
                    num_unique,
                    max_emb_dim,
                    max_val_dim,
                    initializer,
                    lfu_accumulated_frequency,
                    admitter,
                    external_storage,
                    key_scores=unique_scores,
                )
                # The cache addresses its own value buffer by cache slot: a cache is
                # never NO_EVICTION (it enables overflow, which NO_EVICTION rejects
                # at construction), so the two index spaces coincide here.
                value_row_indices = slot_indices
            elif storage_mode == StorageMode.HBM_DIRECT:
                (
                    slot_indices,
                    non_admitted_positions,
                    non_admitted_count,
                    value_row_indices,
                ) = _prefetch_hbm_direct_path(
                    storage,
                    unique_keys,
                    unique_table_ids,
                    num_unique,
                    max_emb_dim,
                    max_val_dim,
                    initializer,
                    lfu_accumulated_frequency,
                    admitter,
                    key_scores=unique_scores,
                )
                key_map_generation = storage._state.key_map_generation

        return PrefetchState(
            unique_keys=unique_keys,
            reverse_indices=reverse_indices,
            unique_table_ids=unique_table_ids,
            lfu_accumulated_frequency=lfu_accumulated_frequency,
            slot_indices=slot_indices,
            storage_mode=storage_mode,
            non_admitted_positions=non_admitted_positions,
            unique_scores=unique_scores,
            num_prefetched_keys=num_prefetched_keys,
            outstanding_keys_ref=outstanding_keys_ref,
            unique_count=unique_count,
            value_row_indices=value_row_indices,
            key_map_generation=key_map_generation,
            num_unique=num_unique,
            non_admitted_count=non_admitted_count,
        )


def dynamicemb_eval_forward(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    cache: Optional[Cache],
    storage: Storage,
    feature_offsets: torch.Tensor,
    output_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    layout: EmbeddingLayout,
    evict_strategy=None,
    frequency_counters: Optional[torch.Tensor] = None,
    pooling_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eval-only forward for all storage configurations (no autograd)."""
    with torch.cuda.nvtx.range("dynamicemb_eval_forward"):
        table_num = feature_offsets.numel() - 1
        assert table_num != 0
        emb_dtype = storage.embedding_dtype()

        is_pooling = layout.is_pooling

        evict_strat = EvictStrategy(evict_strategy.value) if evict_strategy else None

        indices_table_range = get_table_range(offsets, feature_offsets)

        if not is_pooling:
            table_ids = expand_table_ids_cuda(
                indices_table_range,
                indices.numel(),
            )
            frequency_counts_int64 = (
                frequency_counters.long() if frequency_counters is not None else None
            )
            output_embs = eval_lookup(
                storage,
                indices,
                table_ids,
                initializer,
                cache=cache,
            )
            if output_dtype != emb_dtype:
                output_embs = output_embs.to(output_dtype)
            return output_embs

        frequency_counts_int64 = (
            frequency_counters.long() if frequency_counters is not None else None
        )

        (
            unique_indices,
            reverse_indices,
            unique_indices_table_range,
            lfu_accumulated_frequency,
        ) = segmented_unique(
            indices,
            indices_table_range,
            evict_strat,
            frequency_counts_int64,
        )

        num_unique = unique_indices_table_range[table_num:]
        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range, unique_indices.numel()
        )

        unique_embs = eval_lookup(
            storage,
            unique_indices,
            unique_table_ids,
            initializer,
            cache=cache,
            num_valid=num_unique,
        )

        output_embs = torch.empty(
            *layout.pooled_shape(),
            dtype=output_dtype,
            device=indices.device,
        )
        gather_embedding_pooled(
            unique_embs,
            output_embs,
            reverse_indices,
            offsets,
            layout.pooling_mode,
            layout.total_D,
            layout.batch_size,
            layout.D_offsets,
            layout.max_D,
            pooling_weights,
            layout.feature_dims_vec4,
        )
        return output_embs


def _generic_forward_path(
    storage: Storage,
    optimizer: BaseDynamicEmbeddingOptimizer,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    num_unique: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standalone forward for generic (DEFAULT) storage mode.

    Does storage.find + admission + init + insert
    in a single pass with no PrefetchState.

    Returns ``(unique_values, persisted_unique_indices, num_persisted)``. The
    indices are an int64 vector of positions among the unique keys of the keys that
    are stored in ``storage`` after this forward (lookup hit or admitted insert),
    ``num_persisted`` of them. Used in backward to ``storage.insert(...,
    preserve_existing=True)`` only for those rows, without compacting via a boolean
    mask again.
    """
    with torch.cuda.nvtx.range("_generic_forward_path"):
        device = unique_keys.device
        if unique_keys.numel() == 0:
            return (
                torch.empty(0, max_val_dim, dtype=emb_dtype, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
                num_unique,
            )
        with torch.cuda.nvtx.range("op:storage_find"):
            (
                num_missing,
                missing_keys,
                missing_indices,
                missing_table_ids,
                missing_scores,
                founds,
                _,
                unique_values,
            ) = storage.find(
                unique_keys,
                unique_table_ids,
                copy_mode=CopyMode.VALUE,
                lfu_accumulated_frequency=accumulated_frequency,
                key_scores=key_scores,
                num_valid=num_unique,
            )

        # A copy: the storage can be a class of the user's, whose ``founds`` may be a
        # tensor it keeps.
        key_persisted = founds.clone()
        freq_for_admission = (
            counted_gather(accumulated_frequency, missing_indices, num_missing)
            if admitter is not None and accumulated_frequency is not None
            else None
        )
        (
            num_to_insert,
            keys_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
            num_to_init,
            indices_to_init,
        ) = _apply_admission(
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            unique_values,
            unique_keys,
            unique_table_ids,
            max_emb_dim,
            freq_for_admission,
            admitter,
            num_missing,
        )
        with torch.cuda.nvtx.range("op:initializer"):
            initializer(
                unique_values[:, :max_emb_dim],
                unique_keys,
                unique_table_ids,
                indices_to_init,
                num_valid=num_to_init,
            )
        if max_val_dim != max_emb_dim:
            optimizer.reset_optimizer_states(
                unique_values[:, max_emb_dim:],
                indices=missing_indices,
                table_ids=unique_table_ids,
                table_emb_dims=storage.embedding_dims(on_device=True),
                num_valid=num_missing,
            )
        with torch.cuda.nvtx.range("op:storage_insert"):
            storage.insert(
                keys_to_insert,
                table_ids_to_insert,
                counted_gather(unique_values, positions_in_unique, num_to_insert),
                scores_to_insert,
                num_valid=num_to_insert,
            )
        counted_scatter_fill(key_persisted, positions_in_unique, True, num_to_insert)
        with torch.cuda.nvtx.range("op:compact"):
            num_persisted, persisted_unique_indices, _ = flagged_compact(
                key_persisted, [], num_unique
            )
        return unique_values, persisted_unique_indices, num_persisted


def release_slot_pins(
    cache: Optional[Cache],
    storage: Storage,
    storage_mode: StorageMode,
    slot_indices: torch.Tensor,
    unique_table_ids: torch.Tensor,
    *,
    num_valid: torch.Tensor,
) -> None:
    """Undo the ref-counter pins that prefetch took on ``slot_indices``.

    The counter sits beside the hash table of whichever of ``cache`` / ``storage``
    served the batch, and is indexed by hash slot (not by value row). Called by
    backward, or by a forward that has no backward (no autograd graph).
    """
    counter_owner = cache if storage_mode == StorageMode.CACHE else storage
    counter_owner.decrement_counter(slot_indices, unique_table_ids, num_valid=num_valid)


class DynamicEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        prefetch_state: PrefetchState,
        offsets: torch.Tensor,
        cache: Optional[Cache],
        storage: Storage,
        output_dtype: torch.dtype,
        initializer: MultiTableInitializer,
        optimizer: BaseDynamicEmbeddingOptimizer,
        layout: EmbeddingLayout,
        admitter=None,
        pooling_weights: Optional[torch.Tensor] = None,
        *args,
    ):
        with torch.cuda.nvtx.range("DynamicEmbeddingFunction.forward"):
            max_emb_dim = storage.max_embedding_dim()
            max_val_dim = storage.max_value_dim()
            emb_dtype = storage.embedding_dtype()

            # pooling_weights is validated by the sole caller,
            # BatchedDynamicEmbeddingTablesV2.forward (pooling mode, dtype,
            # numel, and exclusivity with frequency_counters), and again by
            # TORCH_CHECK in gather_embedding_pooled / reduce_grads.
            is_pooling = layout.is_pooling

            use_counter = prefetch_state.slot_indices is not None
            num_unique = prefetch_state.num_unique
            num_persisted = None
            if use_counter:
                state = (
                    cache._state
                    if prefetch_state.storage_mode == StorageMode.CACHE
                    else storage._state
                )
                with torch.cuda.nvtx.range("op:load_from_flat"):
                    unique_embs = load_from_flat(
                        state,
                        # Value buffer: address by row (the slot, for a cache).
                        prefetch_state.value_row_indices,
                        prefetch_state.unique_table_ids,
                        copy_mode=CopyMode.EMBEDDING,
                        num_valid=num_unique,
                    )
                if prefetch_state.non_admitted_positions is not None:
                    na = prefetch_state.non_admitted_positions
                    non_admitted_initializer = (
                        admitter.non_admitted_initializer
                        if admitter is not None
                        else None
                    )
                    with torch.cuda.nvtx.range("op:initializer"):
                        # Whichever initializer owns rejected rows was settled
                        # at construction, so this is one call either way.
                        (non_admitted_initializer or initializer)(
                            unique_embs[:, :max_emb_dim],
                            prefetch_state.unique_keys,
                            prefetch_state.unique_table_ids,
                            na,
                            num_valid=prefetch_state.non_admitted_count,
                        )
                unique_values = None
                persisted_unique_indices = None
            else:
                assert prefetch_state.storage_mode == StorageMode.DEFAULT
                (
                    unique_values,
                    persisted_unique_indices,
                    num_persisted,
                ) = _generic_forward_path(
                    storage,
                    optimizer,
                    prefetch_state.unique_keys,
                    prefetch_state.unique_table_ids,
                    num_unique,
                    max_emb_dim,
                    max_val_dim,
                    emb_dtype,
                    initializer,
                    prefetch_state.lfu_accumulated_frequency,
                    admitter,
                    key_scores=prefetch_state.unique_scores,
                )
                # find() was asked for CopyMode.VALUE, so a row here is
                # embedding ++ optimizer state; keep only the embedding.
                unique_embs = unique_values[:, :max_emb_dim]

            device = prefetch_state.unique_keys.device
            if is_pooling:
                output_embs = torch.empty(
                    *layout.pooled_shape(),
                    dtype=output_dtype,
                    device=device,
                )
                with torch.cuda.nvtx.range("op:gather_embedding"):
                    gather_embedding_pooled(
                        unique_embs,
                        output_embs,
                        prefetch_state.reverse_indices,
                        offsets,
                        layout.pooling_mode,
                        layout.total_D,
                        layout.batch_size,
                        layout.D_offsets,
                        layout.max_D,
                        pooling_weights,
                        layout.feature_dims_vec4,
                    )
            else:
                output_embs = torch.empty(
                    *layout.sequence_shape(),
                    dtype=output_dtype,
                    device=device,
                )
                with torch.cuda.nvtx.range("op:gather_embedding"):
                    gather_embedding(
                        unique_embs, output_embs, prefetch_state.reverse_indices
                    )

            ctx.unique_keys = prefetch_state.unique_keys
            ctx.reverse_indices = prefetch_state.reverse_indices
            ctx.unique_table_ids = prefetch_state.unique_table_ids
            ctx.cache = cache
            ctx.storage = storage
            ctx.slot_indices = prefetch_state.slot_indices
            # Same split as the forward: the fused optimizer writes into the
            # value buffer (rows), the ref counter is indexed by hash slot.
            ctx.update_value_rows = prefetch_state.value_row_indices
            ctx.key_map_generation = prefetch_state.key_map_generation
            ctx.storage_mode = prefetch_state.storage_mode
            ctx.optimizer = optimizer
            ctx.layout = layout
            ctx.offsets = offsets
            ctx.max_emb_dim = max_emb_dim
            ctx.max_val_dim = max_val_dim
            ctx.emb_dtype = emb_dtype
            ctx.pooling_weights = pooling_weights
            ctx.use_counter = use_counter
            ctx.unique_values = unique_values
            ctx.persisted_unique_indices = persisted_unique_indices
            ctx.num_unique = num_unique
            ctx.num_persisted = num_persisted
            ctx.outstanding_keys_ref = prefetch_state.outstanding_keys_ref
            ctx.num_prefetched_keys = prefetch_state.num_prefetched_keys

            # Recover outstanding count at end of forward so it is decremented
            # even when backward is not run, avoiding overflow.
            if prefetch_state.outstanding_keys_ref is not None:
                prefetch_state.outstanding_keys_ref -= (
                    prefetch_state.num_prefetched_keys
                )

            return output_embs

    @staticmethod
    def backward(ctx, grads):
        with torch.cuda.nvtx.range("DynamicEmbeddingFunction.backward"):
            cache = ctx.cache
            storage = ctx.storage
            optimizer = ctx.optimizer
            grads = grads.contiguous()

            if optimizer.need_gradient_clipping():
                optimizer.clip_gradient(grads)

            layout = ctx.layout
            num_unique = ctx.num_unique
            if layout.is_pooling:
                with torch.cuda.nvtx.range("op:reduce_grads"):
                    unique_grads = reduce_grads(
                        ctx.reverse_indices,
                        grads,
                        ctx.unique_keys.numel(),
                        layout.batch_size,
                        ctx.max_emb_dim,
                        ctx.offsets,
                        layout.D_offsets,
                        layout.pooling_mode,
                        layout.total_D,
                        ctx.pooling_weights,
                        layout.feature_dims_vec4,
                    )
            else:
                with torch.cuda.nvtx.range("op:reduce_grads"):
                    unique_grads = reduce_grads(
                        ctx.reverse_indices,
                        grads,
                        ctx.unique_keys.numel(),
                        layout.batch_size,
                        ctx.max_emb_dim,
                    )

            optimizer.step()

            unique_table_ids = ctx.unique_table_ids

            with torch.cuda.nvtx.range("DynamicEmbeddingFunction.update"):
                if ctx.use_counter:
                    state = (
                        cache._state
                        if ctx.storage_mode == StorageMode.CACHE
                        else storage._state
                    )
                    # A later batch's prefetch may have rehashed the table
                    # since this one's prefetch (prefetch pipelines, several
                    # forwards before their backwards).
                    refresh_slots_after_rehash(state, ctx)
                    with torch.cuda.nvtx.range("op:optimizer_update_fused"):
                        optimizer.fused_update_for_flat_table(
                            unique_grads.to(ctx.emb_dtype),
                            # Writes into the value buffer -> rows.
                            ctx.update_value_rows,
                            state.table_ptrs_dev,
                            unique_table_ids,
                            state.table_value_dims,
                            state.table_emb_dims,
                            state.max_emb_dim,
                            state.all_dims_vec4,
                            state.emb_dtype,
                            num_valid=num_unique,
                        )

                    release_slot_pins(
                        cache,
                        storage,
                        ctx.storage_mode,
                        ctx.slot_indices,
                        ctx.unique_table_ids,
                        num_valid=num_unique,
                    )

                if not ctx.use_counter and ctx.unique_values is not None:
                    # Per-table embedding dims (on the CUDA device) let the
                    # padded-buffer kernel place each row's optimizer state at the
                    # max_emb_dim offset using the row's own width. Provided
                    # uniformly across storage types via embedding_dims().
                    table_emb_dims = storage.embedding_dims(on_device=True)
                    # all_dims_vec4 must reflect per-table dims, not just
                    # max_emb_dim: a vec4 store on a row whose edim % 4 != 0 would
                    # run past its m region into v. storage.all_dims_vec4() is the
                    # per-table-correct flag used by load/store too.
                    all_dims_vec4 = storage.all_dims_vec4()
                    with torch.cuda.nvtx.range("op:optimizer_update_padded"):
                        optimizer.update_for_padded_buffer(
                            unique_grads,
                            ctx.unique_values,
                            unique_table_ids,
                            table_emb_dims,
                            ctx.max_emb_dim,
                            ctx.max_val_dim,
                            all_dims_vec4,
                            num_valid=num_unique,
                        )
                    pui = ctx.persisted_unique_indices
                    assert pui is not None
                    num_p = ctx.num_persisted
                    keys_c = counted_gather(ctx.unique_keys, pui, num_p)
                    tids_c = counted_gather(unique_table_ids, pui, num_p)
                    vals_c = counted_gather(ctx.unique_values, pui, num_p)
                    # preserve_existing=True → CONST score policy on insert:
                    # refresh embedding/optimizer buffers only; do not overwrite
                    # table scores (LFU / timestamp / etc.). For HybridStorage,
                    # the HBM tier uses this on backward; existing scores stay.
                    with torch.cuda.nvtx.range("op:storage_insert"):
                        storage.insert(
                            keys_c,
                            tids_c,
                            vals_c,
                            preserve_existing=True,
                            num_valid=num_p,
                        )

            return (None,) * 11
