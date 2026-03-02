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
from typing import List, Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import DynamicEmbPoolingMode
from dynamicemb.initializer import BaseDynamicEmbInitializer
from dynamicemb.key_value_table import (
    Cache,
    DynamicEmbCache,
    DynamicEmbStorage,
    Storage,
    _find_keys,
    _prepare_insert_score_arg,
    eval_lookup,
    load_from_flat,
    store_to_flat,
)
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import AdmissionStrategy, CopyMode, Counter
from dynamicemb_extensions import (
    EvictStrategy,
    expand_table_ids_cuda,
    flagged_compact,
    gather_embedding,
    gather_embedding_pooled,
    get_table_range,
    reduce_grads,
    segmented_unique_cuda,
)


def segmented_unique(
    keys: torch.Tensor,
    segment_range: torch.Tensor,
    evict_strategy: Optional[EvictStrategy] = None,
    frequency_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                  output_scores)
    """
    with torch.cuda.nvtx.range("segmented_unique"):
        num_keys = keys.size(0)
        num_tables = segment_range.size(0) - 1
        device = keys.device

        if num_keys == 0:
            empty_keys = torch.empty(0, dtype=keys.dtype, device=device)
            empty_reverse_indices = torch.empty(0, dtype=torch.int64, device=device)
            d_table_range = torch.zeros(
                num_tables + 1, dtype=torch.int64, device=device
            )
            return (
                empty_keys,
                empty_reverse_indices,
                d_table_range,
                None,
            )

        is_lfu_enabled = (
            evict_strategy == EvictStrategy.KLfu if evict_strategy else False
        )
        need_frequency_output = is_lfu_enabled or frequency_counts is not None

        table_ids = expand_table_ids_cuda(
            segment_range,
            None,
            num_tables,
            1,
            num_keys,
        )

        input_frequencies = None
        if frequency_counts is not None:
            input_frequencies = frequency_counts
        elif need_frequency_output:
            input_frequencies = torch.empty(0, dtype=torch.int64, device=device)

        (
            num_uniques,
            unique_keys,
            reverse_indices,
            table_offsets,
            freq_counters,
        ) = segmented_unique_cuda(keys, table_ids, num_tables, input_frequencies)

        total_unique = num_uniques.item()
        unique_keys_out = unique_keys[:total_unique]

        output_scores = None
        if need_frequency_output and total_unique > 0:
            output_scores = freq_counters[:total_unique]

        return (
            unique_keys_out,
            reverse_indices,
            table_offsets,
            output_scores,
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
    table_num: int
    emb_dim: int
    value_dim: int
    emb_dtype: torch.dtype
    storage_mode: StorageMode
    slot_indices: Optional[torch.Tensor]
    update_slot_indices: Optional[torch.Tensor] = None
    non_admitted_positions: Optional[torch.Tensor] = None
    num_prefetched_keys: int = 0
    outstanding_keys_ref: Optional[torch.Tensor] = None


def _is_hbm_storage(storage: Storage) -> bool:
    return isinstance(storage, DynamicEmbStorage) and storage._state.tables[0].is_cuda


def _apply_admission(
    missing_keys: torch.Tensor,
    missing_indices: torch.Tensor,
    missing_table_ids: torch.Tensor,
    missing_scores: Optional[torch.Tensor],
    values: torch.Tensor,
    emb_dim: int,
    freq_for_admission: Optional[torch.Tensor],
    admit_strategy: Optional[AdmissionStrategy],
    admission_counter: Optional[Counter],
    device: torch.device,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Apply admission filtering for missing keys.

    If an admission strategy is active, also handles non-admitted embedding
    initialization via admit_strategy.initialize_non_admitted_embeddings,
    then filters keys/scores/table_ids/indices to only the admitted subset.

    Returns (keys_to_insert, scores_to_insert, table_ids_to_insert,
             positions_in_unique, indices_to_init) where indices_to_init are
    the positions in values that the caller should initialize with its
    embeddings initializer.
    """
    with torch.cuda.nvtx.range("_apply_admission"):
        if admit_strategy is None or missing_keys.numel() == 0:
            return (
                missing_keys,
                missing_scores,
                missing_table_ids,
                missing_indices,
                missing_indices,
            )

        if freq_for_admission is not None:
            counters_for_admission = freq_for_admission
        else:
            counters_for_admission = torch.ones(
                missing_keys.shape[0],
                dtype=torch.int64,
                device=device,
            )
        freq_for_missing_keys = admission_counter.add(
            missing_keys,
            missing_table_ids,
            counters_for_admission,
        )
        admit_mask = admit_strategy.admit(missing_keys, freq_for_missing_keys)

        non_admitted_mask = ~admit_mask
        _, _, (non_admitted_indices,) = flagged_compact(
            non_admitted_mask, [missing_indices]
        )
        initialized_non_admitted = False
        if non_admitted_indices.numel() > 0:
            initialized_non_admitted = (
                admit_strategy.initialize_non_admitted_embeddings(
                    values[:, :emb_dim],
                    non_admitted_indices,
                )
            )

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
        indices_to_init = (
            missing_indices if initialized_non_admitted else positions_in_unique
        )
        admission_counter.erase(keys_to_insert, table_ids_to_insert)

        return (
            keys_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
            indices_to_init,
        )


def _prefetch_cache_path(
    cache: DynamicEmbCache,
    storage: Storage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    emb_dim: int,
    val_dim: int,
    emb_dtype: torch.dtype,
    initializer: BaseDynamicEmbInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admit_strategy: Optional[AdmissionStrategy],
    admission_counter: Optional[Counter],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Cache prefetch with counter protection and overflow buffer.

    Only admitted keys are inserted into the cache.  Non-admitted keys get
    slot_indices = -1 and their positions (in the unique_keys array) are
    returned so the forward pass can lazily initialize their embeddings.

    Returns (slot_indices, update_slot_indices, non_admitted_positions).
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path"):
        device = unique_keys.device
        state = cache._state
        h_num_total = unique_keys.numel()

        if h_num_total == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty.clone(), None

        is_lfu_enabled = evict_strategy == EvictStrategy.KLfu

        # 1. Lookup with overflow fallback
        _, founds, cache_indices = cache.lookup(
            unique_keys,
            unique_table_ids,
            input_scores=accumulated_frequency if is_lfu_enabled else None,
        )

        slot_indices = cache_indices.clone()

        # 2. Increment counter for found keys
        found_slots = cache_indices[founds]
        if found_slots.numel() > 0:
            cache.increment_counter(found_slots)

        missing_mask = ~founds
        h_num_miss, miss_compact_idx, (miss_keys, miss_tids) = flagged_compact(
            missing_mask, [unique_keys, unique_table_ids]
        )

        if h_num_miss == 0:
            return slot_indices, slot_indices.clone(), None

        miss_scores_lfu = (
            accumulated_frequency[miss_compact_idx]
            if accumulated_frequency is not None
            else None
        )

        # 3. Storage lookup for cache-miss keys
        (
            h_num_new,
            new_keys,
            new_indices_in_miss,
            new_table_ids,
            _,
            storage_founds,
            _,
            storage_values,
        ) = storage.find(
            miss_keys,
            miss_tids,
            copy_mode=CopyMode.VALUE,
            input_scores=miss_scores_lfu,
        )

        # 4. Determine which new keys (not in storage) are admitted
        new_in_miss = ~storage_founds
        non_admitted_positions: Optional[torch.Tensor] = None
        keys_to_insert_mask = storage_founds.clone()

        if new_in_miss.any() and admit_strategy is not None:
            new_miss_indices = torch.where(new_in_miss)[0]
            new_keys_sub = miss_keys[new_miss_indices]
            new_tids_sub = miss_tids[new_miss_indices]

            freq_for_admission = (
                miss_scores_lfu[new_miss_indices]
                if miss_scores_lfu is not None
                else None
            )
            counters = (
                freq_for_admission
                if freq_for_admission is not None
                else torch.ones(new_keys_sub.shape[0], dtype=torch.int64, device=device)
            )
            freq = admission_counter.add(new_keys_sub, new_tids_sub, counters)
            admit_mask = admit_strategy.admit(new_keys_sub, freq)

            if admit_mask.any():
                admission_counter.erase(
                    new_keys_sub[admit_mask], new_tids_sub[admit_mask]
                )
                keys_to_insert_mask[new_miss_indices[admit_mask]] = True

            non_admit = ~admit_mask
            if non_admit.any():
                non_admitted_miss_pos = new_miss_indices[non_admit]
                non_admitted_positions = miss_compact_idx[non_admitted_miss_pos]
        elif new_in_miss.any():
            keys_to_insert_mask = torch.ones(
                h_num_miss, dtype=torch.bool, device=device
            )

        # 5. Insert only storage-found + admitted-new keys into cache
        if not keys_to_insert_mask.any():
            slot_indices[miss_compact_idx] = -1
            update_slot_indices = slot_indices.clone()
            return slot_indices, update_slot_indices, non_admitted_positions

        _, insert_to_miss, (insert_keys, insert_tids) = flagged_compact(
            keys_to_insert_mask, [miss_keys, miss_tids]
        )
        insert_scores = (
            miss_scores_lfu[insert_to_miss] if miss_scores_lfu is not None else None
        )

        (
            cache_insert_indices,
            num_evicted,
            evicted_keys,
            evicted_indices,
            evicted_scores,
            evicted_table_ids,
        ) = cache.insert_and_evict(insert_keys, insert_tids, insert_scores)

        # 6. Load evicted values before overwriting their slots
        evicted_values = load_from_flat(
            state, evicted_indices, evicted_table_ids, copy_mode=CopyMode.VALUE
        )

        # 7. Store storage-found values to their cache slots
        is_sf_in_insert = storage_founds[insert_to_miss]
        if is_sf_in_insert.any():
            store_to_flat(
                state,
                cache_insert_indices[is_sf_in_insert],
                insert_tids[is_sf_in_insert],
                storage_values[insert_to_miss[is_sf_in_insert]],
            )

        # 8. Initialize admitted-new keys in their cache slots
        is_new_in_insert = ~is_sf_in_insert
        if is_new_in_insert.any():
            n_new_admitted = is_new_in_insert.sum().item()
            init_vals = torch.empty(
                n_new_admitted, val_dim, dtype=emb_dtype, device=device
            )
            init_indices = torch.arange(
                n_new_admitted, dtype=torch.int64, device=device
            )
            new_admitted_keys = insert_keys[is_new_in_insert]
            initializer(init_vals[:, :emb_dim], init_indices, new_admitted_keys)

            if val_dim != emb_dim:
                init_vals[:, emb_dim:] = storage.init_optimizer_state()

            store_to_flat(
                state,
                cache_insert_indices[is_new_in_insert],
                insert_tids[is_new_in_insert],
                init_vals,
            )

        # 9. Write back evicted to storage
        if num_evicted > 0:
            storage.insert(
                evicted_keys, evicted_table_ids, evicted_values, evicted_scores
            )

        # 10. Counter & slot mapping
        cache.increment_counter(cache_insert_indices)
        slot_indices[miss_compact_idx[insert_to_miss]] = cache_insert_indices

        # 11. Non-admitted keys get slot_indices = -1
        if non_admitted_positions is not None:
            slot_indices[non_admitted_positions] = -1

        update_slot_indices = slot_indices.clone()
        if non_admitted_positions is not None:
            update_slot_indices[non_admitted_positions] = -1

        return slot_indices, update_slot_indices, non_admitted_positions


def _prefetch_hbm_direct_path(
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    emb_dim: int,
    val_dim: int,
    initializer: BaseDynamicEmbInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admit_strategy: Optional[AdmissionStrategy],
    admission_counter: Optional[Counter],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """HBM-direct prefetch with counter protection.

    Only admitted keys are inserted.  Non-admitted keys get slot_indices = -1
    and their positions are returned for lazy initialization at forward time.

    Returns (slot_indices, update_slot_indices, non_admitted_positions).
    """
    with torch.cuda.nvtx.range("_prefetch_hbm_direct_path"):
        state = storage._state
        device = unique_keys.device
        h_num_total = unique_keys.numel()

        if h_num_total == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty.clone(), None

        is_lfu_enabled = evict_strategy == EvictStrategy.KLfu

        (
            h_num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            _,
            _,
            indices,
        ) = _find_keys(
            state,
            unique_keys,
            unique_table_ids,
            input_scores=accumulated_frequency if is_lfu_enabled else None,
        )

        found_mask = indices >= 0
        found_slots = indices[found_mask]
        if found_slots.numel() > 0:
            storage.increment_counter(found_slots)

        if h_num_missing == 0:
            return indices, indices.clone(), None

        # Determine admission for missing keys
        non_admitted_positions: Optional[torch.Tensor] = None

        if admit_strategy is not None:
            freq_for_admission = (
                accumulated_frequency[missing_indices]
                if accumulated_frequency is not None
                else None
            )
            counters = (
                freq_for_admission
                if freq_for_admission is not None
                else torch.ones(missing_keys.shape[0], dtype=torch.int64, device=device)
            )
            freq = admission_counter.add(missing_keys, missing_table_ids, counters)
            admit_mask = admit_strategy.admit(missing_keys, freq)

            if admit_mask.any():
                admission_counter.erase(
                    missing_keys[admit_mask], missing_table_ids[admit_mask]
                )

            non_admit = ~admit_mask
            if non_admit.any():
                non_admitted_positions = missing_indices[non_admit]

            admitted_keys = missing_keys[admit_mask]
            admitted_tids = missing_table_ids[admit_mask]
            admitted_scores = (
                missing_scores[admit_mask] if missing_scores is not None else None
            )
            admitted_unique_positions = missing_indices[admit_mask]
        else:
            admitted_keys = missing_keys
            admitted_tids = missing_table_ids
            admitted_scores = missing_scores
            admitted_unique_positions = missing_indices

        # Initialize and insert admitted keys
        if admitted_keys.numel() > 0:
            n_admitted = admitted_keys.numel()
            init_values = torch.empty(
                n_admitted, val_dim, dtype=state.emb_dtype, device=device
            )
            init_idx = torch.arange(n_admitted, dtype=torch.int64, device=device)
            initializer(init_values[:, :emb_dim], init_idx, admitted_keys)

            if val_dim != emb_dim:
                init_values[:, emb_dim:] = state.initial_optim_state

            score_arg = _prepare_insert_score_arg(
                state, admitted_scores, n_admitted, device
            )
            new_indices = state.key_index_map.insert(
                admitted_keys,
                admitted_tids,
                score_arg,
            )
            store_to_flat(state, new_indices, admitted_tids, init_values)

            inserted_mask = new_indices >= 0
            if inserted_mask.any():
                storage.increment_counter(new_indices[inserted_mask])

            indices[admitted_unique_positions] = new_indices

        # Non-admitted keys: ensure slot_indices = -1
        if non_admitted_positions is not None:
            indices[non_admitted_positions] = -1

        update_slot_indices = indices.clone()
        still_missing = indices < 0
        if still_missing.any():
            update_slot_indices[still_missing] = -1

        return indices, update_slot_indices, non_admitted_positions


def dynamicemb_prefetch(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    cache: Optional[Cache],
    storage: Storage,
    feature_offsets: torch.Tensor,
    initializers: List[BaseDynamicEmbInitializer],
    forward_stream: Optional[torch.cuda.Stream] = None,
    evict_strategy=None,
    frequency_counters: Optional[torch.Tensor] = None,
    admit_strategy: Optional[AdmissionStrategy] = None,
    admission_counter: Optional[Counter] = None,
    outstanding_keys_ref: Optional[torch.Tensor] = None,
) -> PrefetchState:
    """Unified prefetch for all storage types (cache, HBM-direct, generic).

    Returns a PrefetchState containing unique embeddings + optimizer states
    and the metadata needed by forward/backward.
    """
    with torch.cuda.nvtx.range("dynamicemb_prefetch"):
        table_num = feature_offsets.numel() - 1
        assert table_num != 0
        emb_dtype = storage.embedding_dtype()
        emb_dim = storage.max_embedding_dim()
        val_dim = storage.max_value_dim()
        caching = cache is not None

        evict_strat = EvictStrategy(evict_strategy.value) if evict_strategy else None

        frequency_counts_int64 = None
        if frequency_counters is not None:
            frequency_counts_int64 = frequency_counters.long()

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

        num_prefetched_keys = unique_keys.numel()

        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range,
            None,
            table_num,
            1,
            unique_keys.numel(),
        )

        slot_indices = None
        update_slot_indices = None
        non_admitted_positions = None
        if caching:
            storage_mode = StorageMode.CACHE
        elif _is_hbm_storage(storage):
            storage_mode = StorageMode.HBM_DIRECT
        else:
            storage_mode = StorageMode.DEFAULT

        if storage_mode == StorageMode.CACHE:
            if outstanding_keys_ref is not None:
                outstanding_keys_ref += num_prefetched_keys
                cache_capacity = cache._state.capacity
                if outstanding_keys_ref.item() > cache_capacity:
                    raise RuntimeError(
                        f"Outstanding prefetched keys "
                        f"({outstanding_keys_ref.item()}) "
                        f"exceed cache capacity ({cache_capacity}). "
                        "Reduce prefetch pipeline depth or increase "
                        "cache size."
                    )
            (
                slot_indices,
                update_slot_indices,
                non_admitted_positions,
            ) = _prefetch_cache_path(
                cache,
                storage,
                unique_keys,
                unique_table_ids,
                emb_dim,
                val_dim,
                emb_dtype,
                initializers[0],
                evict_strat,
                lfu_accumulated_frequency,
                admit_strategy,
                admission_counter,
            )
        elif storage_mode == StorageMode.HBM_DIRECT:
            (
                slot_indices,
                update_slot_indices,
                non_admitted_positions,
            ) = _prefetch_hbm_direct_path(
                storage,
                unique_keys,
                unique_table_ids,
                emb_dim,
                val_dim,
                initializers[0],
                evict_strat,
                lfu_accumulated_frequency,
                admit_strategy,
                admission_counter,
            )

        return PrefetchState(
            unique_keys=unique_keys,
            reverse_indices=reverse_indices,
            unique_table_ids=unique_table_ids,
            lfu_accumulated_frequency=lfu_accumulated_frequency,
            table_num=table_num,
            emb_dim=emb_dim,
            value_dim=val_dim,
            emb_dtype=emb_dtype,
            slot_indices=slot_indices,
            storage_mode=storage_mode,
            update_slot_indices=update_slot_indices,
            non_admitted_positions=non_admitted_positions,
            num_prefetched_keys=num_prefetched_keys,
            outstanding_keys_ref=outstanding_keys_ref,
        )


def dynamicemb_eval_forward(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    cache: Optional[Cache],
    storage: Storage,
    feature_offsets: torch.Tensor,
    output_dtype: torch.dtype,
    initializers: List[BaseDynamicEmbInitializer],
    evict_strategy=None,
    frequency_counters: Optional[torch.Tensor] = None,
    pooling_mode: DynamicEmbPoolingMode = DynamicEmbPoolingMode.NONE,
    total_D: int = 0,
    batch_size: int = 0,
    dims: Optional[List[int]] = None,
    max_D: int = 0,
    D_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eval-only forward for all storage configurations (no autograd)."""
    with torch.cuda.nvtx.range("dynamicemb_eval_forward"):
        table_num = feature_offsets.numel() - 1
        assert table_num != 0
        emb_dtype = storage.embedding_dtype()
        storage.max_embedding_dim()

        is_pooling = pooling_mode != DynamicEmbPoolingMode.NONE

        evict_strat = EvictStrategy(evict_strategy.value) if evict_strategy else None

        indices_table_range = get_table_range(offsets, feature_offsets)

        if not is_pooling:
            table_ids = expand_table_ids_cuda(
                indices_table_range,
                None,
                table_num,
                1,
                indices.numel(),
            )
            frequency_counts_int64 = (
                frequency_counters.long() if frequency_counters is not None else None
            )
            output_embs = eval_lookup(
                storage,
                indices,
                table_ids,
                initializers[0],
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

        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range,
            None,
            table_num,
            1,
            unique_indices.numel(),
        )

        unique_embs = eval_lookup(
            storage,
            unique_indices,
            unique_table_ids,
            initializers[0],
            cache=cache,
        )

        combiner = 0 if pooling_mode == DynamicEmbPoolingMode.SUM else 1
        output_embs = torch.empty(
            batch_size,
            total_D,
            dtype=output_dtype,
            device=indices.device,
        )
        gather_embedding_pooled(
            unique_embs,
            output_embs,
            reverse_indices,
            offsets,
            combiner,
            total_D,
            batch_size,
            D_offsets,
            max_D,
        )
        return output_embs


def _generic_forward_path(
    storage: Storage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    emb_dim: int,
    val_dim: int,
    emb_dtype: torch.dtype,
    initializer: BaseDynamicEmbInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admit_strategy: Optional[AdmissionStrategy],
    admission_counter: Optional[Counter],
) -> torch.Tensor:
    """Standalone forward for generic (DEFAULT) storage mode.

    Does storage.find + admission + init + insert
    in a single pass with no PrefetchState.
    """
    with torch.cuda.nvtx.range("_generic_forward_path"):
        device = unique_keys.device
        unique_keys.numel()

        is_lfu_enabled = evict_strategy == EvictStrategy.KLfu
        (
            h_num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            _,
            _,
            unique_values,
        ) = storage.find(
            unique_keys,
            unique_table_ids,
            copy_mode=CopyMode.VALUE,
            input_scores=accumulated_frequency if is_lfu_enabled else None,
        )

        if h_num_missing == 0:
            return unique_values

        freq_for_admission = (
            accumulated_frequency[missing_indices]
            if admit_strategy is not None and accumulated_frequency is not None
            else None
        )
        (
            keys_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
            indices_to_init,
        ) = _apply_admission(
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            unique_values,
            emb_dim,
            freq_for_admission,
            admit_strategy,
            admission_counter,
            device,
        )

        if indices_to_init.numel() > 0:
            initializer(unique_values[:, :emb_dim], indices_to_init, unique_keys)

        if val_dim != emb_dim:
            unique_values[missing_indices, emb_dim:] = storage.init_optimizer_state()

        values_to_insert = unique_values[positions_in_unique]

        storage.insert(
            keys_to_insert,
            table_ids_to_insert,
            values_to_insert,
            scores_to_insert,
        )
        return unique_values


class DynamicEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        prefetch_state: PrefetchState,
        offsets: torch.Tensor,
        cache: Optional[Cache],
        storage: Storage,
        output_dtype: torch.dtype,
        initializers: List[BaseDynamicEmbInitializer],
        optimizer: BaseDynamicEmbeddingOptimizer,
        admit_strategy=None,
        evict_strategy=None,
        admission_counter: Optional[Counter] = None,
        pooling_mode: DynamicEmbPoolingMode = DynamicEmbPoolingMode.NONE,
        total_D: int = 0,
        batch_size: int = 0,
        dims: Optional[List[int]] = None,
        max_D: int = 0,
        D_offsets: Optional[torch.Tensor] = None,
        *args,
    ):
        with torch.cuda.nvtx.range("DynamicEmbeddingFunction.forward"):
            emb_dim = storage.max_embedding_dim()
            val_dim = storage.max_value_dim()
            emb_dtype = storage.embedding_dtype()

            is_pooling = pooling_mode != DynamicEmbPoolingMode.NONE
            mixed_D = is_pooling and dims is not None and max_D > min(dims)
            out_dim = max_D if mixed_D else emb_dim

            use_counter = prefetch_state.update_slot_indices is not None
            if use_counter:
                state = (
                    cache._state
                    if prefetch_state.storage_mode == StorageMode.CACHE
                    else storage._state
                )
                unique_embs = load_from_flat(
                    state,
                    prefetch_state.slot_indices,
                    prefetch_state.unique_table_ids,
                    copy_mode=CopyMode.EMBEDDING,
                )
                if out_dim != emb_dim:
                    unique_embs = unique_embs[:, :out_dim]

                if prefetch_state.non_admitted_positions is not None:
                    na = prefetch_state.non_admitted_positions
                    initializers[0](
                        unique_embs[:, :emb_dim],
                        na,
                        prefetch_state.unique_keys,
                    )
                unique_values = None
            else:
                assert prefetch_state.storage_mode == StorageMode.DEFAULT
                evict_strat = (
                    EvictStrategy(evict_strategy.value) if evict_strategy else None
                )
                unique_values = _generic_forward_path(
                    storage,
                    prefetch_state.unique_keys,
                    prefetch_state.unique_table_ids,
                    emb_dim,
                    val_dim,
                    emb_dtype,
                    initializers[0],
                    evict_strat,
                    prefetch_state.lfu_accumulated_frequency,
                    admit_strategy,
                    admission_counter,
                )
                unique_embs = unique_values[:, :out_dim]

            device = prefetch_state.unique_keys.device
            if is_pooling:
                combiner = 0 if pooling_mode == DynamicEmbPoolingMode.SUM else 1
                output_embs = torch.empty(
                    batch_size,
                    total_D,
                    dtype=output_dtype,
                    device=device,
                )
                gather_embedding_pooled(
                    unique_embs,
                    output_embs,
                    prefetch_state.reverse_indices,
                    offsets,
                    combiner,
                    total_D,
                    batch_size,
                    D_offsets,
                    max_D,
                )
            else:
                combiner = -1
                output_embs = torch.empty(
                    prefetch_state.reverse_indices.shape[0],
                    emb_dim,
                    dtype=output_dtype,
                    device=device,
                )
                gather_embedding(
                    unique_embs, output_embs, prefetch_state.reverse_indices
                )

            ctx.unique_keys = prefetch_state.unique_keys
            ctx.reverse_indices = prefetch_state.reverse_indices
            ctx.unique_table_ids = prefetch_state.unique_table_ids
            ctx.cache = cache
            ctx.storage = storage
            ctx.slot_indices = prefetch_state.slot_indices
            ctx.update_slot_indices = prefetch_state.update_slot_indices
            ctx.storage_mode = prefetch_state.storage_mode
            ctx.optimizer = optimizer
            ctx.pooling_mode = pooling_mode
            ctx.combiner = combiner
            ctx.offsets = offsets
            ctx.batch_size = batch_size
            ctx.total_D = total_D
            ctx.emb_dim = emb_dim
            ctx.value_dim = val_dim
            ctx.emb_dtype = emb_dtype
            ctx.mixed_D = mixed_D
            ctx.dims = dims
            ctx.max_D = max_D
            ctx.D_offsets = D_offsets
            ctx.num_features = (
                (offsets.shape[0] - 1) // batch_size if batch_size > 0 else 0
            )
            ctx.use_counter = use_counter
            ctx.unique_values = unique_values
            ctx.outstanding_keys_ref = prefetch_state.outstanding_keys_ref
            ctx.num_prefetched_keys = prefetch_state.num_prefetched_keys

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

            is_pooling = ctx.pooling_mode != DynamicEmbPoolingMode.NONE
            if is_pooling:
                out_dim = ctx.max_D if ctx.mixed_D else ctx.emb_dim
                unique_grads = reduce_grads(
                    ctx.reverse_indices,
                    grads,
                    ctx.unique_keys.numel(),
                    ctx.batch_size,
                    out_dim,
                    ctx.offsets,
                    ctx.D_offsets,
                    ctx.combiner,
                    ctx.total_D,
                )
            else:
                unique_grads = reduce_grads(
                    ctx.reverse_indices,
                    grads,
                    ctx.unique_keys.numel(),
                    ctx.batch_size,
                    ctx.emb_dim,
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
                    optimizer.fused_update_for_flat_table(
                        unique_grads.to(ctx.emb_dtype),
                        ctx.update_slot_indices,
                        state.table_ptrs,
                        unique_table_ids,
                        state.table_value_dims,
                        state.table_emb_dims,
                        state.max_emb_dim,
                        state.all_dims_vec4,
                        state.emb_dtype,
                    )
                    counter_owner = (
                        cache if ctx.storage_mode == StorageMode.CACHE else storage
                    )
                    counter_owner.decrement_counter(ctx.slot_indices)

                if not ctx.use_counter and ctx.unique_values is not None:
                    optimizer.update_for_padded_buffer(
                        unique_grads,
                        ctx.unique_values,
                        ctx.emb_dim,
                        ctx.value_dim,
                    )
                    storage.insert(ctx.unique_keys, unique_table_ids, ctx.unique_values)

            if ctx.outstanding_keys_ref is not None:
                ctx.outstanding_keys_ref -= ctx.num_prefetched_keys

            return (None,) * 17
