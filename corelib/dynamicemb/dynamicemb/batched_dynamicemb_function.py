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
    select_insert_failed_values,
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
    unique_values: torch.Tensor
    reverse_indices: torch.Tensor
    unique_indices_table_range: torch.Tensor
    unique_table_ids: torch.Tensor
    table_num: int
    emb_dim: int
    value_dim: int
    emb_dtype: torch.dtype
    slot_indices: Optional[torch.Tensor]
    storage_mode: StorageMode
    cache_miss_info: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


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


def _apply_admission_and_init(
    missing_keys: torch.Tensor,
    missing_indices: torch.Tensor,
    missing_table_ids: torch.Tensor,
    missing_scores: Optional[torch.Tensor],
    unique_values: torch.Tensor,
    emb_dim: int,
    val_dim: int,
    all_keys: torch.Tensor,
    freq_for_admission: Optional[torch.Tensor],
    admit_strategy: Optional[AdmissionStrategy],
    admission_counter: Optional[Counter],
    initializer: BaseDynamicEmbInitializer,
    initial_optim_state: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor
]:
    """Apply admission filtering, initialize embeddings, and prepare insert-ready data.

    Initializes unique_values for missing keys in-place (embeddings via the
    initializer, optimizer state via initial_optim_state), then filters by
    admission mask.

    Returns (keys_to_insert, values_to_insert, scores_to_insert,
             table_ids_to_insert, positions_in_unique).
    """
    with torch.cuda.nvtx.range("_apply_admission_and_init"):
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
            initializer(unique_values[:, :emb_dim], indices_to_init, all_keys)

        if val_dim != emb_dim:
            unique_values[missing_indices, emb_dim:] = initial_optim_state

        values_to_insert = unique_values[positions_in_unique]

        return (
            keys_to_insert,
            values_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
        )


def _prefetch_cache_path(
    cache: Cache,
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
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """Cache prefetch path.

    Returns (slot_indices, unique_values, cache_miss_info) for all unique keys.

    slot_indices[i] >= 0  ->  key is in cache at that slot
    slot_indices[i] < 0   ->  key is NOT in cache (insert failed); its value
                              was written to storage by the eviction handler.
    cache_miss_info is (miss_idx, miss_keys, miss_tids) for keys with negative
    slot_indices, or None when there are no misses.
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path"):
        device = unique_keys.device
        is_lfu_enabled = evict_strategy == EvictStrategy.KLfu

        (
            h_num_miss,
            miss_keys,
            miss_indices,
            miss_table_ids,
            miss_scores,
            _,
            _,
            cache_indices,
        ) = cache.find(
            unique_keys,
            unique_table_ids,
            input_scores=accumulated_frequency if is_lfu_enabled else None,
        )

        unique_values = load_from_flat(
            cache._state, cache_indices, unique_table_ids, copy_mode=CopyMode.VALUE
        )

        if h_num_miss == 0:
            return cache_indices, unique_values, None

        (
            h_num_new,
            new_keys,
            new_indices,
            new_table_ids,
            new_scores,
            founds,
            miss_scores,
            storage_values,
        ) = storage.find(
            miss_keys,
            miss_table_ids,
            copy_mode=CopyMode.VALUE,
            input_scores=miss_scores,
        )
        freq_for_admission = (
            accumulated_frequency[miss_indices][new_indices]
            if accumulated_frequency is not None
            else None
        )
        (
            _,
            _,
            _,
            admitted_positions,
            indices_to_init,
        ) = _apply_admission(
            new_keys,
            new_indices,
            new_table_ids,
            None,
            storage_values,
            emb_dim,
            freq_for_admission,
            admit_strategy,
            admission_counter,
            device,
        )

        if indices_to_init.numel() > 0:
            initializer(storage_values[:, :emb_dim], indices_to_init, miss_keys)

        if val_dim != emb_dim and h_num_new > 0:
            if admitted_positions.numel() > 0:
                storage_values[
                    admitted_positions, emb_dim:
                ] = storage.init_optimizer_state()

        unique_values[miss_indices] = storage_values

        keys_to_cache = miss_keys
        tids_to_cache = miss_table_ids
        vals_to_cache = storage_values
        scores_to_cache = miss_scores

        if admit_strategy is not None:
            founds[admitted_positions] = True
            (
                _,
                cache_idx,
                (keys_to_cache, tids_to_cache, miss_remap, scores_to_cache),
            ) = flagged_compact(
                founds, [miss_keys, miss_table_ids, miss_indices, miss_scores]
            )
            vals_to_cache = storage_values[cache_idx]
        else:
            miss_remap = miss_indices

        if keys_to_cache.numel() == 0:
            cache_miss_mask = cache_indices < 0
            _, miss_idx, (miss_keys, miss_tids) = flagged_compact(
                cache_miss_mask, [unique_keys, unique_table_ids]
            )
            return cache_indices, unique_values, (miss_idx, miss_keys, miss_tids)

        (
            insert_indices,
            num_evicted,
            evicted_keys,
            evicted_table_ids,
            evicted_indices,
            evicted_scores,
        ) = cache.insert_and_evict(keys_to_cache, tids_to_cache, scores_to_cache)

        evicted_values = load_from_flat(
            cache._state, evicted_indices, evicted_table_ids, copy_mode=CopyMode.VALUE
        )
        select_insert_failed_values(evicted_indices, vals_to_cache, evicted_values)
        store_to_flat(cache._state, insert_indices, tids_to_cache, vals_to_cache)

        if num_evicted != 0:
            storage.insert(
                evicted_keys,
                evicted_table_ids,
                evicted_values,
                evicted_scores,
            )

        cache_indices[miss_remap] = insert_indices
        slot_indices = cache_indices

        cache_miss_mask = slot_indices < 0
        _, miss_idx, (miss_keys, miss_tids) = flagged_compact(
            cache_miss_mask, [unique_keys, unique_table_ids]
        )

        return slot_indices, unique_values, (miss_idx, miss_keys, miss_tids)


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HBM-direct prefetch path. Returns (slot_indices, unique_values) for all unique keys."""
    with torch.cuda.nvtx.range("_prefetch_hbm_direct_path"):
        state = storage._state
        device = unique_keys.device
        h_num_total = unique_keys.numel()
        emb_dtype = state.emb_dtype

        if h_num_total == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, val_dim, dtype=emb_dtype, device=device),
            )

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

        unique_values = load_from_flat(
            state, indices, unique_table_ids, copy_mode=CopyMode.VALUE
        )

        freq_for_admission = (
            accumulated_frequency[missing_indices]
            if admit_strategy is not None and accumulated_frequency is not None
            else None
        )

        if h_num_missing == 0:
            return indices, unique_values
        (
            keys_to_insert,
            values_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
        ) = _apply_admission_and_init(
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            unique_values,
            emb_dim,
            val_dim,
            unique_keys,
            freq_for_admission,
            admit_strategy,
            admission_counter,
            initializer,
            state.initial_optim_state,
            device,
        )

        if keys_to_insert.numel() > 0:
            score_arg = _prepare_insert_score_arg(
                state, scores_to_insert, keys_to_insert.numel(), device
            )
            new_indices = state.key_index_map.insert(
                keys_to_insert,
                table_ids_to_insert,
                score_arg,
            )
            store_to_flat(state, new_indices, table_ids_to_insert, values_to_insert)
            indices[positions_in_unique] = new_indices

        return indices, unique_values


def _prefetch_generic_path(
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
    """Generic storage prefetch path. Returns unique_values."""
    with torch.cuda.nvtx.range("_prefetch_generic_path"):
        device = unique_keys.device
        h_num_total = unique_keys.numel()
        if h_num_total == 0:
            return torch.empty(0, val_dim, dtype=emb_dtype, device=device)

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
            values_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            _,
        ) = _apply_admission_and_init(
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            unique_values,
            emb_dim,
            val_dim,
            unique_keys,
            freq_for_admission,
            admit_strategy,
            admission_counter,
            initializer,
            storage.init_optimizer_state(),
            device,
        )

        storage.insert(
            keys_to_insert,
            table_ids_to_insert,
            values_to_insert,
            scores_to_insert,
        )

        return unique_values


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

        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range,
            None,
            table_num,
            1,
            unique_keys.numel(),
        )

        slot_indices = None
        cache_miss_info = None
        if caching:
            storage_mode = StorageMode.CACHE
        elif _is_hbm_storage(storage):
            storage_mode = StorageMode.HBM_DIRECT
        else:
            storage_mode = StorageMode.DEFAULT

        if storage_mode == StorageMode.CACHE:
            slot_indices, unique_values, cache_miss_info = _prefetch_cache_path(
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
            slot_indices, unique_values = _prefetch_hbm_direct_path(
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
        else:
            unique_values = _prefetch_generic_path(
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

        return PrefetchState(
            unique_keys=unique_keys,
            unique_values=unique_values,
            reverse_indices=reverse_indices,
            unique_indices_table_range=unique_indices_table_range,
            unique_table_ids=unique_table_ids,
            table_num=table_num,
            emb_dim=emb_dim,
            value_dim=val_dim,
            emb_dtype=emb_dtype,
            slot_indices=slot_indices,
            storage_mode=storage_mode,
            cache_miss_info=cache_miss_info,
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
        cache is not None

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

        dims is not None and max_D > min(dims)

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


class DynamicEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        cache: Optional[Cache],
        storage: Storage,
        feature_offsets: torch.Tensor,
        output_dtype: torch.dtype,
        initializers: List[BaseDynamicEmbInitializer],
        optimizer: BaseDynamicEmbeddingOptimizer,
        admit_strategy=None,
        evict_strategy=None,
        frequency_counters: Optional[torch.Tensor] = None,
        admission_counter: Optional[Counter] = None,
        pooling_mode: DynamicEmbPoolingMode = DynamicEmbPoolingMode.NONE,
        total_D: int = 0,
        batch_size: int = 0,
        dims: Optional[List[int]] = None,
        max_D: int = 0,
        D_offsets: Optional[torch.Tensor] = None,
        prefetch_state: Optional[PrefetchState] = None,
        *args,
    ):
        with torch.cuda.nvtx.range("DynamicEmbeddingFunction.forward"):
            emb_dim = storage.max_embedding_dim()
            emb_dtype = storage.embedding_dtype()

            is_pooling = pooling_mode != DynamicEmbPoolingMode.NONE
            mixed_D = is_pooling and dims is not None and max_D > min(dims)

            if prefetch_state is None:
                prefetch_state = dynamicemb_prefetch(
                    indices,
                    offsets,
                    cache,
                    storage,
                    feature_offsets,
                    initializers,
                    None,
                    evict_strategy,
                    frequency_counters,
                    admit_strategy,
                    admission_counter,
                )

            out_dim = max_D if mixed_D else emb_dim
            unique_embs = prefetch_state.unique_values[:, :out_dim]

            if is_pooling:
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
                    indices.shape[0],
                    emb_dim,
                    dtype=output_dtype,
                    device=indices.device,
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
            ctx.storage_mode = prefetch_state.storage_mode
            ctx.optimizer = optimizer
            ctx.pooling_mode = pooling_mode
            ctx.combiner = combiner
            ctx.offsets = offsets
            ctx.batch_size = batch_size
            ctx.total_D = total_D
            ctx.emb_dim = emb_dim
            ctx.value_dim = prefetch_state.value_dim
            ctx.emb_dtype = emb_dtype
            ctx.mixed_D = mixed_D
            ctx.dims = dims
            ctx.max_D = max_D
            ctx.D_offsets = D_offsets
            ctx.num_features = (
                (offsets.shape[0] - 1) // batch_size if batch_size > 0 else 0
            )

            ctx.cache_miss_info = prefetch_state.cache_miss_info
            if prefetch_state.storage_mode == StorageMode.HBM_DIRECT:
                ctx.unique_values = None
            elif prefetch_state.storage_mode == StorageMode.CACHE:
                has_misses = (
                    prefetch_state.cache_miss_info is not None
                    and prefetch_state.cache_miss_info[0].numel() > 0
                )
                ctx.unique_values = prefetch_state.unique_values if has_misses else None
            else:
                ctx.unique_values = prefetch_state.unique_values

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
                if ctx.slot_indices is not None:
                    state = (
                        cache._state
                        if ctx.storage_mode == StorageMode.CACHE
                        else storage._state
                    )
                    optimizer.fused_update_for_flat_table(
                        unique_grads.to(ctx.emb_dtype),
                        ctx.slot_indices,
                        state.table_ptrs,
                        unique_table_ids,
                        state.table_value_dims,
                        state.table_emb_dims,
                        state.max_emb_dim,
                        state.all_dims_vec4,
                        state.emb_dtype,
                    )

                # TODO: if we can avoid insert fail in cache, we can remove the cache-miss insert
                if ctx.unique_values is not None:
                    if ctx.cache_miss_info is not None:
                        miss_idx, keys_to_insert, tids_to_insert = ctx.cache_miss_info
                        grads_to_update = unique_grads[miss_idx]
                        values_to_update = ctx.unique_values[miss_idx]
                    else:
                        grads_to_update = unique_grads
                        values_to_update = ctx.unique_values
                        keys_to_insert = ctx.unique_keys
                        tids_to_insert = unique_table_ids
                    optimizer.update_for_padded_buffer(
                        grads_to_update,
                        values_to_update,
                        ctx.emb_dim,
                        ctx.value_dim,
                    )
                    storage.insert(keys_to_insert, tids_to_insert, values_to_update)

            return (None,) * 20
