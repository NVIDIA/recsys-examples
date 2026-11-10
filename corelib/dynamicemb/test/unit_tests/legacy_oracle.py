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

"""The host-sized lookup bodies of the sync-removal PR (commit 355a1f61), kept as a TEST-ONLY oracle.

The package no longer has them: a lookup group whose participants are written against exact-length
tensors (an external storage that implements ``find`` / ``insert``, a user admitter) is served by
the padded bodies through adapters that read the live-row count of each call once. These are the
functions the padded bodies replaced, copied verbatim and renamed ``legacy_*`` (they read the
number of missing keys from the device, compact with ``flagged_compact`` and index with boolean
masks: every host sync of the old design is in here, on purpose). Tests drive them on exact-length
arrays and compare what the padded bodies do with the same inputs.

Two edits to the copy, and nothing else:

* ``legacy_write_back_evicted`` with a next tier honors ``slots`` like the padded write-back: an
  entry whose slot is negative is a cache insert that was refused, a row of ``values`` that was
  never loaded, and the old code wrote it over the storage row of a key it had just inserted
  (uninitialized memory).
* every function carries the prefix ``legacy_``.

No production code imports this module and there is no switch that reaches it.
"""
from typing import Callable, Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import EvictedItemMode
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.key_value_table import (
    Cache,
    DynamicEmbCache,
    DynamicEmbStorage,
    Storage,
    _append_evicted,
    _find_keys,
    _insert_and_evict_keys,
    _write_back_evicted_masked,
    get_insert_score_arg,
    load_from_flat,
    score_source,
    store_to_flat,
)
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import CopyMode, MultiTableAdmitter
from dynamicemb_extensions import (
    EvictStrategy,
    flagged_compact,
    select_insert_failed_values,
)


def legacy_apply_admission(
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
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Apply admission filtering for missing keys.

    If an admission strategy is active, runs ``non_admitted_initializer`` over
    the rows it rejects when the strategy asked for one, then filters
    keys/scores/table_ids/indices to only the admitted subset.

    Returns (keys_to_insert, scores_to_insert, table_ids_to_insert,
             positions_in_unique, indices_to_init) where indices_to_init are
    the positions in values that the caller should initialize with its
    embeddings initializer. Every missing row is covered exactly once: the
    non-admitted ones drop out of indices_to_init exactly when this function
    has already run the strategy's own initializer over them.
    """
    with torch.cuda.nvtx.range("legacy_apply_admission"):
        if admitter is None or missing_keys.numel() == 0:
            return (
                missing_keys,
                missing_scores,
                missing_table_ids,
                missing_indices,
                missing_indices,
            )

        admit_mask = admitter.admit(missing_keys, missing_table_ids, freq_for_admission)
        non_admitted_initializer = admitter.non_admitted_initializer

        # Only a strategy with its own initializer needs the rejected rows (a host
        # fact); the compaction that finds them is a host read.
        if non_admitted_initializer is not None:
            non_admitted_mask = ~admit_mask
            with torch.cuda.nvtx.range("op:flagged_compact"):
                _, _, (non_admitted_indices,) = flagged_compact(
                    non_admitted_mask, [missing_indices]
                )
            if non_admitted_indices.numel() > 0:
                with torch.cuda.nvtx.range("op:initializer"):
                    non_admitted_initializer(
                        values[:, :max_emb_dim],
                        # Alongside ``values``, which is why these are the whole
                        # unique vectors and not the missing slice; the rows to
                        # write come last.
                        unique_keys,
                        unique_table_ids,
                        non_admitted_indices,
                    )

        with torch.cuda.nvtx.range("op:flagged_compact"):
            (
                _,
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
            )
        # Whoever owns the non-admitted rows was settled at construction. When
        # the strategy asked for its own initializer the table's only has to
        # cover the admitted rows; otherwise it has to cover every miss, or the
        # non-admitted rows reach the forward holding whatever
        # ``storage.find`` left in its ``torch.empty`` value buffer.
        indices_to_init = (
            positions_in_unique
            if non_admitted_initializer is not None
            else missing_indices
        )

        return (
            keys_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
            indices_to_init,
        )


def legacy_load_storage_values(
    storage: DynamicEmbStorage,
    slots: torch.Tensor,
    miss_tids: torch.Tensor,
) -> torch.Tensor:
    """Values of the misses, from the rows a key-only storage lookup found.

    A miss storage lacks has a -1 row, which the load skips: its row is left
    unwritten.
    """
    st = storage._state
    rows = slots
    return load_from_flat(st, rows, miss_tids, copy_mode=CopyMode.VALUE)


def legacy_lookup_next_tier(
    next_tier: Optional[Storage],
    miss_keys: torch.Tensor,
    miss_tids: torch.Tensor,
    in_storage: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Ask ``next_tier`` for the misses storage lacks; fill their ``values``.

    Returns the mask (over the misses) of those it had.
    """
    in_next = torch.zeros_like(in_storage)
    if next_tier is None:
        return in_next
    with torch.cuda.nvtx.range("op:flagged_compact"):
        num_asked, asked, (asked_keys, asked_tids) = flagged_compact(
            ~in_storage, [miss_keys, miss_tids]
        )
    if num_asked == 0:
        return in_next
    with torch.cuda.nvtx.range("op:next_tier_find"):
        result = next_tier.find(asked_keys, asked_tids, copy_mode=CopyMode.VALUE)
    found, found_values = result[5], result[7]
    # ``asked`` lists real, distinct positions, so both writes are plain scatters.
    # A boolean-mask index (``asked[found]``) or a Python scalar put through a CUDA
    # index would each wait on the host. ``found`` may come back from a host-side
    # store: it has to sit beside ``asked`` for the scatter.
    found = found.to(in_next.device)
    in_next[asked] = found
    values[asked] = torch.where(found.unsqueeze(1), found_values, values[asked])
    return in_next


def legacy_initialize_rows(
    state,
    initializer: MultiTableInitializer,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
) -> torch.Tensor:
    """Fresh ``[len(keys), max_val_dim]`` rows: initialized embedding, reset optimizer state."""
    n = keys.numel()
    rows = torch.empty(n, max_val_dim, dtype=emb_dtype, device=keys.device)
    with torch.cuda.nvtx.range("op:initializer"):
        initializer(
            rows[:, :max_emb_dim],
            keys,
            table_ids,
            torch.arange(n, dtype=torch.int64, device=keys.device),
        )
    if max_val_dim != max_emb_dim:
        state.optimizer.reset_optimizer_states(
            rows[:, max_emb_dim:], emb_dims=state.table_emb_dims[table_ids]
        )
    return rows


def legacy_insert_into_storage(
    storage: DynamicEmbStorage,
    next_tier: Optional[Storage],
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    values: torch.Tensor,
    scores: Optional[torch.Tensor],
) -> None:
    """Insert new keys into ``storage``; what it evicts for room moves to ``next_tier``."""
    if next_tier is None:
        storage.insert(keys, table_ids, values, scores)
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
    ) = _insert_and_evict_keys(st, keys, table_ids, scores)
    if num_evicted > 0:
        evicted_values = load_from_flat(
            st, evicted_slots, evicted_table_ids, copy_mode=CopyMode.VALUE
        )
        # A key the storage refused (e.g. its bucket is full and the score does
        # not beat any stored one) comes back as an "evicted" row with a negative
        # slot: load_from_flat left its value unwritten, so take the key's own
        # input value instead, as HybridStorage.insert does.
        select_insert_failed_values(evicted_slots, values, evicted_values)
    store_to_flat(st, slots, table_ids, values)
    if num_evicted > 0:
        next_tier.insert(evicted_keys, evicted_table_ids, evicted_values)


def legacy_prefetch_cache_path(
    cache: DynamicEmbCache,
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    external_storage: Optional[Storage] = None,
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Cache prefetch with counter protection and overflow buffer.

    The host-count path: it reads from the device how many keys each step narrows
    to and works on arrays of exactly that size. It serves an external storage,
    user-defined classes and DYNAMICEMB_SYNC_FREE=0.

    ``storage`` is the tier that owns every key's score; the cache above it
    ranks by timestamp only, and holds the freshest value of a key until it is
    written back. When ``external_storage`` is given (a
    score-less store below ``storage``, e.g. a PS) it holds the keys ``storage`` has
    evicted: keys missing from ``storage`` are fetched from it, and what
    ``storage`` evicts is spilled to it.

    Only admitted keys are inserted into the cache.  Non-admitted keys get
    slot_indices = -1 and their positions (in the unique_keys array) are
    returned so the forward pass can lazily initialize their embeddings.

    Index spaces: ``unique_*`` arrays are indexed by position in the unique
    batch; everything after the cache lookup is over the *misses* only, indexed
    by miss number, and ``miss_pos`` maps a miss number back to its position
    in the unique batch.

    Returns (slot_indices, non_admitted_positions).
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path"):
        device = unique_keys.device
        state = cache._state

        if unique_keys.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, None

        # 1. Lookup with overflow fallback; pin what is found
        with torch.cuda.nvtx.range("op:cache_lookup"):
            _, founds, cache_indices = cache.lookup(unique_keys, unique_table_ids)
        slot_indices = cache_indices.clone()
        # Pin every row at once: the counter kernel returns on a negative slot
        # (a miss) before it reads the table id, and the slots of the hits differ.
        cache.increment_counter(cache_indices, unique_table_ids)

        # 2. Storage is the only place scores live: a key-only lookup of *every*
        # key -- hits included, and even when nothing missed -- updates its score
        # there without moving a value. The cache only ranks by timestamp.
        # ``score_input`` is what the storage's score is made from, chosen once
        # here and carried with the keys from now on; ``accumulated_frequency``
        # is only for admission below.
        score_input = score_source(storage._state, accumulated_frequency, key_scores)
        with torch.cuda.nvtx.range("op:storage_find"):
            # Only the lookup is wanted here, not the compaction of its misses.
            (_, _, _, _, _, st_founds, _, st_slots) = _find_keys(
                storage._state,
                unique_keys,
                unique_table_ids,
                score_input=score_input,
                compact=False,
            )

        # 3. The misses, with everything later steps need about them
        with torch.cuda.nvtx.range("op:flagged_compact"):
            (
                num_miss,
                miss_pos,
                (
                    miss_keys,
                    miss_tids,
                    miss_freq,
                    miss_score_input,
                    miss_slots,
                ),
            ) = flagged_compact(
                ~founds,
                [
                    unique_keys,
                    unique_table_ids,
                    accumulated_frequency,
                    score_input,
                    st_slots,
                ],
            )
        if num_miss == 0:
            return slot_indices, None

        # 4. Where each miss lives: storage, the tier below it, or nowhere (new).
        # ``values`` holds the first two's rows; new ones are filled in at 6.
        in_storage = st_founds[miss_pos]
        values = legacy_load_storage_values(storage, miss_slots, miss_tids)
        in_next = legacy_lookup_next_tier(
            external_storage, miss_keys, miss_tids, in_storage, values
        )

        # 5. Admit the new ones (a key from the tier below was admitted before)
        is_new = ~(in_storage | in_next)
        admitted = is_new.clone()
        non_admitted_positions: Optional[torch.Tensor] = None
        if admitter is not None:
            with torch.cuda.nvtx.range("op:flagged_compact"):
                (
                    num_new,
                    new_rows,
                    (new_keys, new_tids, new_freq, new_pos),
                ) = flagged_compact(is_new, [miss_keys, miss_tids, miss_freq, miss_pos])
            if num_new > 0:
                admit_mask = admitter.admit(new_keys, new_tids, new_freq)
                admitted[new_rows] = admit_mask
                with torch.cuda.nvtx.range("op:flagged_compact"):
                    num_refused, _, (refused_pos,) = flagged_compact(
                        ~admit_mask, [new_pos]
                    )
                if num_refused > 0:
                    # A refused key was a miss, so its slot is already -1, and it
                    # stays out of ``to_cache`` below.
                    non_admitted_positions = refused_pos

        to_cache = in_storage | in_next | admitted
        with torch.cuda.nvtx.range("op:flagged_compact"):
            (
                num_cache,
                into_cache,
                (cache_keys, cache_tids, cache_pos),
            ) = flagged_compact(to_cache, [miss_keys, miss_tids, miss_pos])
        if num_cache == 0:
            return slot_indices, non_admitted_positions

        # 6. Initialize the admitted-new rows, and put everything storage lacks
        # into it -- with its score -- before the cache sees it.
        with torch.cuda.nvtx.range("op:flagged_compact"):
            num_adm, adm_rows, (adm_keys, adm_tids) = flagged_compact(
                admitted, [miss_keys, miss_tids]
            )
        if num_adm > 0:
            values[adm_rows] = legacy_initialize_rows(
                state,
                initializer,
                adm_keys,
                adm_tids,
                max_emb_dim,
                max_val_dim,
                emb_dtype,
            )
        with torch.cuda.nvtx.range("op:flagged_compact"):
            num_st, into_storage, (st_keys, st_tids, st_score_input) = flagged_compact(
                to_cache & ~in_storage, [miss_keys, miss_tids, miss_score_input]
            )
        if num_st > 0:
            with torch.cuda.nvtx.range("op:storage_insert"):
                legacy_insert_into_storage(
                    storage,
                    external_storage,
                    st_keys,
                    st_tids,
                    legacy_select(values, into_storage, num_st == values.size(0)),
                    st_score_input,
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
            ) = cache.insert_and_evict(cache_keys, cache_tids, None)

        # 8. What it evicted: read before the slots are overwritten, then write back
        with torch.cuda.nvtx.range("op:load_from_flat"):
            evicted_values = load_from_flat(
                state, evicted_slots, evicted_tids, copy_mode=CopyMode.VALUE
            )
        with torch.cuda.nvtx.range("op:store_to_flat"):
            store_to_flat(
                state,
                cache_slots,
                cache_tids,
                legacy_select(values, into_cache, num_cache == values.size(0)),
            )
        if num_evicted > 0:
            legacy_write_back_evicted(
                storage,
                external_storage,
                evicted_keys,
                evicted_tids,
                evicted_values,
                slots=evicted_slots,
            )

        # 9. Counter & slot mapping
        cache.increment_counter(cache_slots, cache_tids)
        slot_indices[cache_pos] = cache_slots
        return slot_indices, non_admitted_positions


def legacy_prefetch_hbm_direct_path(
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """HBM-direct prefetch with counter protection.

    Only admitted keys are inserted.  Non-admitted keys get slot_indices = -1
    and their positions are returned for lazy initialization at forward time.

    Returns ``(slot_indices, non_admitted_positions, value_row_indices)``. The
    two index spaces are kept apart because they only coincide by accident:
    ``slot_indices`` addresses the hash table (and the ref counter beside it),
    ``value_row_indices`` addresses the flat value buffer. A storage stores a
    row per key, so they differ: rows are handed out densely while slots come
    from the key's hash, and the key map of a table that never evicts is sized
    ``1 / load factor`` times the value buffer, so a slot can point past the end
    of it. A table whose row is its slot returns the slot tensor itself as the
    rows, so the caller always gets something it can address values with.

    This is the host-sized form: it reads the number of missing keys and works
    on arrays compacted to it, over exact-length unique arrays. It runs for what
    the mask form does not take (a user admitter, a batch too large for the
    mask form's temporaries, the kill switch) and is the oracle the mask form is
    tested against.
    """
    with torch.cuda.nvtx.range("_prefetch_hbm_direct_path"):
        state = storage._state
        device = unique_keys.device
        h_num_total = unique_keys.numel()

        if h_num_total == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, None, empty

        with torch.cuda.nvtx.range("op:storage_find"):
            (
                h_num_missing,
                missing_keys,
                missing_indices,
                missing_table_ids,
                missing_scores,
                _,
                _,
                indices,
                slot_indices,
            ) = _find_keys(
                state,
                unique_keys,
                unique_table_ids,
                score_input=score_source(state, accumulated_frequency, key_scores),
                return_slots=True,
            )

        # ``indices`` are rows for a table that stores them, hash slots otherwise,
        # and ``slot_indices`` the hash slots either way (the same tensor when the
        # row is the slot). The ref counter is keyed by slot. A key that was not
        # found has slot -1, which the counter skips before it reads the table id,
        # so every key goes in as it is.
        storage.increment_counter(slot_indices, unique_table_ids)

        # Values are addressed by row, the counter by slot: a storage stores a row
        # per key, so these are different tensors and both are kept up to date.
        rows = indices

        if h_num_missing == 0:
            return slot_indices, None, rows

        # Determine admission for missing keys
        non_admitted_positions: Optional[torch.Tensor] = None

        if admitter is not None:
            freq_for_admission = (
                accumulated_frequency[missing_indices]
                if accumulated_frequency is not None
                else None
            )
            admit_mask = admitter.admit(
                missing_keys, missing_table_ids, freq_for_admission
            )

            admitted_keys = missing_keys[admit_mask]
            admitted_tids = missing_table_ids[admit_mask]
            admitted_scores = (
                missing_scores[admit_mask] if missing_scores is not None else None
            )
            admitted_unique_positions = missing_indices[admit_mask]

            non_admit = ~admit_mask
            if non_admit.any():
                non_admitted_positions = missing_indices[non_admit]
        else:
            admitted_keys = missing_keys
            admitted_tids = missing_table_ids
            admitted_scores = missing_scores
            admitted_unique_positions = missing_indices

        # Initialize and insert admitted keys
        if admitted_keys.numel() > 0:
            n_admitted = admitted_keys.numel()
            init_values = torch.empty(
                n_admitted, max_val_dim, dtype=state.emb_dtype, device=device
            )
            init_idx = torch.arange(n_admitted, dtype=torch.int64, device=device)
            with torch.cuda.nvtx.range("op:initializer"):
                initializer(
                    init_values[:, :max_emb_dim],
                    admitted_keys,
                    admitted_tids,
                    init_idx,
                )

            if max_val_dim != max_emb_dim:
                state.optimizer.reset_optimizer_states(
                    init_values[:, max_emb_dim:],
                    emb_dims=state.table_emb_dims[admitted_tids],
                )

            score_arg = get_insert_score_arg(state, n_admitted, device, admitted_scores)
            with torch.cuda.nvtx.range("op:storage_insert"):
                # HBM-direct is a last tier: this insert may evict, and the
                # victims are dropped here (unlike the cache path, which spills
                # to storage). Retain them when configured -- this is the
                # forward-path analogue of the _insert_key_values collection.
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
                    )
                    _append_evicted(state, evicted_keys, evicted_table_ids, num_evicted)
                else:
                    new_indices, new_slots = state.key_index_map.insert(
                        admitted_keys,
                        admitted_tids,
                        score_arg,
                        return_slots=True,
                    )
            # The insert returns rows, and the slots as well. Keep both: the
            # counter is indexed by slot, the value buffer by row.
            new_rows = new_indices
            with torch.cuda.nvtx.range("op:store_to_flat"):
                store_to_flat(state, new_rows, admitted_tids, init_values)

            # A key the insert refused has slot -1, which the counter skips.
            storage.increment_counter(new_slots, admitted_tids)

            slot_indices[admitted_unique_positions] = new_slots
            rows[admitted_unique_positions] = new_rows

        # A non-admitted key was a miss, so its slot and row are already -1, and
        # it is not among ``admitted_unique_positions``.
        return slot_indices, non_admitted_positions, rows


def legacy_generic_forward_path(
    storage: Storage,
    optimizer: BaseDynamicEmbeddingOptimizer,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standalone forward for generic (DEFAULT) storage mode.

    Does storage.find + admission + init + insert
    in a single pass with no PrefetchState.

    Works on exact-length arrays and reads host counts, so it also serves what
    the mask form (``_generic_forward_path_masked``) cannot: an external or
    user-defined storage or admitter, a batch over the mask form's memory
    limit, and ``DYNAMICEMB_SYNC_FREE=0``.

    Returns (unique_values, persisted_unique_indices). The latter is an int64
    vector of row indices into the unique-key tensors for keys that are stored
    in ``storage`` after this forward (lookup hit or admitted insert). Used in
    backward to ``storage.insert(..., preserve_existing=True)`` only for those
    rows, without compacting via a boolean mask again.
    """
    with torch.cuda.nvtx.range("_generic_forward_path"):
        device = unique_keys.device
        n = unique_keys.numel()

        with torch.cuda.nvtx.range("op:storage_find"):
            (
                h_num_missing,
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
                # Only when there are some: an external store written against
                # the interface before per-key scores has no such parameter.
                **({} if key_scores is None else {"key_scores": key_scores}),
            )

        if h_num_missing == 0:
            if n == 0:
                persisted_unique_indices = torch.empty(
                    0, dtype=torch.int64, device=device
                )
            else:
                persisted_unique_indices = torch.arange(
                    n, device=device, dtype=torch.int64
                )
            return unique_values, persisted_unique_indices

        freq_for_admission = (
            accumulated_frequency[missing_indices]
            if admitter is not None and accumulated_frequency is not None
            else None
        )
        (
            keys_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
            indices_to_init,
        ) = legacy_apply_admission(
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
        )

        if indices_to_init.numel() > 0:
            with torch.cuda.nvtx.range("op:initializer"):
                initializer(
                    unique_values[:, :max_emb_dim],
                    unique_keys,
                    unique_table_ids,
                    indices_to_init,
                )

        if max_val_dim != max_emb_dim:
            optimizer.reset_optimizer_states(
                unique_values[:, max_emb_dim:],
                indices=missing_indices,
                emb_dims=storage.embedding_dims(on_device=True)[missing_table_ids],
            )

        values_to_insert = unique_values[positions_in_unique]

        with torch.cuda.nvtx.range("op:storage_insert"):
            storage.insert(
                keys_to_insert,
                table_ids_to_insert,
                values_to_insert,
                scores_to_insert,
            )
        if admitter is None:
            # Every missing key was inserted above, so every key is persisted
            # and there is nothing to compact.
            return unique_values, torch.arange(n, device=device, dtype=torch.int64)
        key_persisted = founds.clone()
        # index_fill_, not ``key_persisted[positions_in_unique] = True``: a
        # Python scalar put through a CUDA index tensor is a blocking H2D copy.
        key_persisted.index_fill_(0, positions_in_unique, True)
        with torch.cuda.nvtx.range("op:flagged_compact"):
            _, persisted_unique_indices, _ = flagged_compact(
                key_persisted, [unique_keys]
            )
        return unique_values, persisted_unique_indices


def legacy_select(
    values: torch.Tensor, rows: torch.Tensor, all_rows: bool
) -> torch.Tensor:
    """``values[rows]`` -- ``flagged_compact`` only takes 1-D inputs.

    ``all_rows`` says the caller's mask kept every row; ``flagged_compact`` emits
    its index in ascending order, so ``rows`` is then the identity and the gather
    is skipped. The row count alone cannot say that: an index of the same length
    may still permute the rows.
    """
    return values if all_rows else values[rows]


def legacy_write_back_evicted(
    storage: DynamicEmbStorage,
    next_tier: Optional[Storage],
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    values: torch.Tensor,
    slots: Optional[torch.Tensor] = None,
) -> None:
    """Return what the cache evicted to the tier below it.

    A key still in storage keeps its score, so only the value moves. One storage
    has evicted since goes to ``next_tier``; with none it is dropped, kept as an
    evicted key when storage retains them.

    Without a ``next_tier`` this is :func:`_write_back_evicted_masked`: the arrays
    may be of any length, padded with inert entries, and nothing is counted or
    compacted, so the host does not wait for the device. ``slots`` are the
    evicted indices the insert reported (see there).

    With a ``next_tier`` (an external store, which takes host-sized tensors and is
    written by code that may run on the host) the keys are compacted into the
    ones storage has and the ones it has evicted since, each with a count read.
    The arrays are then exact, and ``slots`` is not consulted.
    """
    if keys.numel() == 0:
        return
    if next_tier is None:
        return _write_back_evicted_masked(storage, keys, table_ids, values, slots)
    if slots is not None:
        # The one change to the copy (F4): an entry whose slot is negative is an insert the
        # cache refused, a row of ``values`` that was never loaded. It is neither written back
        # nor passed on (the old code ignored ``slots`` here and could write uninitialized
        # memory over the storage row of a key it had just inserted).
        live = slots >= 0
        keys, table_ids, values = keys[live], table_ids[live], values[live]
        if keys.numel() == 0:
            return
    st = storage._state
    in_storage = _find_keys(st, keys, table_ids, const_lookup=True, compact=False)[5]
    with torch.cuda.nvtx.range("op:flagged_compact"):
        num_present, present, (present_keys, present_tids) = flagged_compact(
            in_storage, [keys, table_ids]
        )
    with torch.cuda.nvtx.range("op:storage_insert"):
        if num_present > 0:
            storage.insert(
                present_keys,
                present_tids,
                legacy_select(values, present, num_present == keys.numel()),
                preserve_existing=True,
            )
        if num_present == keys.numel():
            return
        if num_present == 0:
            gone_keys, gone_tids, gone_values = keys, table_ids, values
        else:
            with torch.cuda.nvtx.range("op:flagged_compact"):
                _, gone, (gone_keys, gone_tids) = flagged_compact(
                    ~in_storage, [keys, table_ids]
                )
            gone_values = values[gone]
        next_tier.insert(gone_keys, gone_tids, gone_values)


def legacy_eval_lookup_storage(
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
) -> torch.Tensor:
    """Exact-length eval lookup through ``Storage.find`` (any ``Storage``)."""
    (
        h_num_missing,
        _,
        missing_indices,
        _,
        _,
        _,
        _,
        embs,
    ) = storage.find(
        keys,
        table_ids,
        copy_mode=CopyMode.EMBEDDING,
    )

    if h_num_missing > 0:
        initializer(embs, keys, table_ids, missing_indices)

    return embs


def legacy_eval_lookup_cached(
    cache: Cache,
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
) -> torch.Tensor:
    """Exact-length eval lookup through ``Storage.find`` (any ``Storage``)."""
    _, founds, cache_indices = cache.lookup(keys, table_ids)

    embs = load_from_flat(
        cache._state, cache_indices, table_ids, copy_mode=CopyMode.EMBEDDING
    )

    missing_mask = ~founds
    h_num_miss, miss_compact_idx, (missing_keys, missing_table_ids) = flagged_compact(
        missing_mask, [keys, table_ids]
    )

    if h_num_miss == 0:
        return embs

    (
        h_num_missing_in_storage,
        _,
        missing_indices_in_storage,
        _,
        _,
        _,
        _,
        storage_embs,
    ) = storage.find(
        missing_keys,
        missing_table_ids,
        copy_mode=CopyMode.EMBEDDING,
    )

    if h_num_missing_in_storage > 0:
        initializer(
            storage_embs, missing_keys, missing_table_ids, missing_indices_in_storage
        )

    embs[miss_compact_idx, :] = storage_embs

    return embs
