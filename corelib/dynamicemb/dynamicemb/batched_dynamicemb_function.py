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
    eval_lookup,
    get_insert_score_arg,
    load_from_flat,
    store_to_flat,
)
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import (
    AdmissionStrategy,
    CacheExchangeRequest,
    CopyMode,
    Counter,
)
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                  output_scores, unique_size_per_table).
        unique_size_per_table is a CPU tensor of shape (num_tables,) with
        per-table unique counts; total unique count is its sum or
        unique_keys_table_range[-1].
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
            unique_size_per_table = torch.zeros(
                num_tables, dtype=torch.int64, device="cpu"
            )
            return (
                empty_keys,
                empty_reverse_indices,
                d_table_range,
                None,
                unique_size_per_table,
            )

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

        table_offsets_cpu = _copy_cuda_tensor_to_pinned_cpu(table_offsets)
        total_unique = table_offsets_cpu[num_tables].item()
        unique_size_per_table = (
            table_offsets_cpu[1 : num_tables + 1] - table_offsets_cpu[0:num_tables]
        )

        unique_keys_out = unique_keys[:total_unique]

        output_scores = None
        if need_frequency_output and total_unique > 0:
            output_scores = freq_counters[:total_unique]

        return (
            unique_keys_out,
            reverse_indices,
            table_offsets,
            output_scores,
            unique_size_per_table,
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
    non_admitted_mask: Optional[torch.Tensor] = None
    direct_storage_rows: Optional[torch.Tensor] = None
    direct_storage_slots: Optional[torch.Tensor] = None
    direct_storage_table_ptrs: Optional[torch.Tensor] = None
    num_prefetched_keys: int = 0
    outstanding_keys_ref: Optional[torch.Tensor] = None


def _is_hbm_storage(storage: Storage) -> bool:
    """True if values live in GPU device memory (HBM). Use buffer type, not
    tensor.is_cuda: host memory registered to CUDA address space can report is_cuda True.
    """
    return (
        isinstance(storage, DynamicEmbStorage)
        and storage._state.tables[0].is_device_buffer()
    )


def _storage_find_scores_are_logical_row_indices(storage: Storage) -> bool:
    """True when ``storage.find``'s score column is NO_EVICTION logical row index, not LFU/timestamp.

    Those values must not be copied into LFU accumulation buffers or reused as cache insert scores.
    """
    if isinstance(storage, DynamicEmbStorage):
        return storage._state.no_eviction_next_index_dev is not None
    return False


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
        missing_founds = torch.zeros(
            missing_keys.shape[0], dtype=torch.bool, device=device
        )
        freq_for_missing_keys = admission_counter.add(
            missing_keys,
            missing_table_ids,
            counters_for_admission,
            missing_founds,
        )
        admit_mask = admit_strategy.admit(
            missing_keys, freq_for_missing_keys, missing_founds
        )

        non_admitted_mask = ~admit_mask
        non_admitted_in_values = torch.zeros(
            values.shape[0], dtype=torch.bool, device=device
        )
        non_admitted_in_values[missing_indices] = non_admitted_mask
        # ``missing_keys`` is already known non-empty here, so a numel check is
        # always true. Do not replace it with ``mask.any()``: converting that
        # CUDA scalar to a Python bool would synchronize the hot path. The
        # initializer consumes the sparse mask and filters on device.
        initialized_non_admitted = (
            admit_strategy.initialize_non_admitted_embeddings(
                values[:, :emb_dim],
                non_admitted_in_values,
            )
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
        # A strategy-specific initializer has already populated rejected rows,
        # so the table initializer must only touch admitted rows. Without one,
        # initialize every missing row so rejected keys still have valid
        # forward values.
        indices_to_init = (
            positions_in_unique if initialized_non_admitted else missing_indices
        )
        admission_counter.erase(missing_keys, missing_table_ids, admit_mask)

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
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fused sparse cache/storage prefetch.

    Every input attempts to receive a protected cache slot. Backing storage
    then exchanges found rows and displaced rows directly with the cache
    buffers. If a failed backing insertion rolls back cache placement, a
    storage hit remains pinned and is consumed through the aligned direct-row
    tensors. Exhausting both the main cache bucket and overflow stash is a
    fatal cache-capacity invariant violation.
    Globally new keys are admitted in the original sparse layout; rejected
    provisional slots are reclaimed and represented by ``-1`` for forward.

    Returns cache rows, update rows, the sparse rejected mask, direct storage
    rows/slots, and direct storage value-buffer pointers.
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path"):
        device = unique_keys.device
        state = cache._state
        h_num_total = unique_keys.numel()

        if h_num_total == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return (
                empty,
                empty.clone(),
                None,
                empty.clone(),
                empty.clone(),
                state.table_ptrs_dev,
            )

        # 1. Find existing cache entries or provision protected slots for all
        # misses. Eviction metadata stays device-resident and input-aligned.
        with torch.cuda.nvtx.range("op:cache_find_or_insert"):
            cache_result = cache.find_or_insert(
                unique_keys,
                unique_table_ids,
                scores=accumulated_frequency,
            )
        slot_indices = cache_result.indices

        # 2. Backing storage performs sparse metadata lookup/insertion and a
        # single direct cache<->storage value exchange.
        with torch.cuda.nvtx.range("op:storage_exchange"):
            exchange_result = storage.exchange(
                CacheExchangeRequest(
                    cache_state=state,
                    keys=unique_keys,
                    table_ids=unique_table_ids,
                    scores=accumulated_frequency,
                    cache_founds=cache_result.founds,
                    cache_indices=slot_indices,
                    evicted_keys=cache_result.evicted_keys,
                    evicted_indices=cache_result.evicted_indices,
                    evicted_scores=cache_result.evicted_scores,
                    evicted_table_ids=cache_result.evicted_table_ids,
                    evicted_mask=cache_result.evicted_mask,
                )
            )

        founds = exchange_result.founds

        # LFU cache misses inherit the backing table's accumulated score.
        # NO_EVICTION backing scores are logical rows and must never become
        # cache eviction scores.
        if accumulated_frequency is not None and not (
            isinstance(storage, DynamicEmbStorage)
            and _storage_find_scores_are_logical_row_indices(storage)
        ):
            with torch.cuda.nvtx.range("op:cache_update_scores"):
                cache.update_scores(
                    unique_keys,
                    slot_indices,
                    unique_table_ids,
                    exchange_result.storage_scores,
                    exchange_result.storage_founds,
                )

        # 3. Admission remains in the original input layout. Found positions
        # are skipped inside the counter/strategy kernels.
        if admit_strategy is not None:
            assert admission_counter is not None
            counters = (
                accumulated_frequency
                if accumulated_frequency is not None
                else torch.ones(h_num_total, dtype=torch.int64, device=device)
            )
            with torch.cuda.nvtx.range("op:admission_counter"):
                frequencies = admission_counter.add(
                    unique_keys, unique_table_ids, counters, founds
                )
            admitted_mask = admit_strategy.admit(
                unique_keys, frequencies, founds
            )
            cache_init_mask = admitted_mask & (slot_indices >= 0)
            admission_counter.erase(
                unique_keys,
                unique_table_ids,
                cache_init_mask,
            )
            non_admitted_mask = (~founds) & (~admitted_mask)
        else:
            admitted_mask = ~founds
            cache_init_mask = admitted_mask
            non_admitted_mask = None

        # 4. Initialize newly admitted rows in-place. Rejected oversubscribed
        # entries become tombstones and their batch slots become -1; forward
        # initializes those output rows from the retained boolean mask.
        with torch.cuda.nvtx.range("op:initializer_flat"):
            initializer.initialize_flat(
                state.table_ptrs_dev,
                slot_indices,
                unique_table_ids,
                state.table_value_dims,
                state.table_emb_dims,
                cache_init_mask,
                unique_keys,
                state.emb_dtype,
                state.initial_optim_state,
            )

        if non_admitted_mask is not None:
            # Intentionally launch with the full sparse mask. A host-side
            # ``any()`` check would serialize prefetch; the reclaim kernel
            # filters false positions on device.
            with torch.cuda.nvtx.range("op:cache_reclaim"):
                cache.reclaim(
                    unique_keys,
                    unique_table_ids,
                    slot_indices,
                    non_admitted_mask,
                )

        # The optimizer and forward consume the same protected slots. Both
        # native users skip -1 entries, so no metadata clone is required.
        return (
            slot_indices,
            slot_indices,
            non_admitted_mask,
            exchange_result.direct_storage_rows,
            exchange_result.direct_storage_slots,
            exchange_result.direct_storage_table_ptrs,
        )


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
    and a full-length mask is returned for lazy initialization at forward time.

    Returns (slot_indices, update_slot_indices, non_admitted_mask).
    """
    with torch.cuda.nvtx.range("_prefetch_hbm_direct_path"):
        state = storage._state
        device = unique_keys.device
        h_num_total = unique_keys.numel()

        if h_num_total == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty.clone(), None

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
            ) = _find_keys(
                state,
                unique_keys,
                unique_table_ids,
                lfu_accumulated_frequency=accumulated_frequency,
            )

        found_mask = indices >= 0
        found_slots = indices[found_mask]
        if found_slots.numel() > 0:
            storage.increment_counter(found_slots, unique_table_ids[found_mask])

        if h_num_missing == 0:
            return indices, indices.clone(), None

        # Determine admission for missing keys
        non_admitted_mask: Optional[torch.Tensor] = None

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
            missing_founds = torch.zeros(
                missing_keys.shape[0], dtype=torch.bool, device=device
            )
            freq = admission_counter.add(
                missing_keys, missing_table_ids, counters, missing_founds
            )
            admit_mask = admit_strategy.admit(
                missing_keys, freq, missing_founds
            )

            admitted_keys = missing_keys[admit_mask]
            admitted_tids = missing_table_ids[admit_mask]
            admitted_scores = (
                missing_scores[admit_mask] if missing_scores is not None else None
            )
            admitted_unique_positions = missing_indices[admit_mask]

            admission_counter.erase(missing_keys, missing_table_ids, admit_mask)

            non_admit = ~admit_mask
            non_admitted_mask = torch.zeros(
                h_num_total, dtype=torch.bool, device=device
            )
            non_admitted_mask[missing_indices] = non_admit
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
            with torch.cuda.nvtx.range("op:initializer"):
                initializer(init_values[:, :emb_dim], init_idx, admitted_keys)

            if val_dim != emb_dim:
                init_values[:, emb_dim:] = state.initial_optim_state

            score_arg = get_insert_score_arg(
                state, n_admitted, device, admitted_scores, table_ids=admitted_tids
            )
            with torch.cuda.nvtx.range("op:storage_insert"):
                new_indices = state.key_index_map.insert(
                    admitted_keys,
                    admitted_tids,
                    score_arg,
                )
            new_indices = (
                score_arg.value.to(torch.int64)
                if state.no_eviction_next_index is not None
                else new_indices
            )
            with torch.cuda.nvtx.range("op:store_to_flat"):
                store_to_flat(state, new_indices, admitted_tids, init_values)

            inserted_mask = new_indices >= 0
            if inserted_mask.any():
                storage.increment_counter(
                    new_indices[inserted_mask],
                    admitted_tids[inserted_mask],
                )

            indices[admitted_unique_positions] = new_indices

        # Non-admitted keys: ensure slot_indices = -1
        if non_admitted_mask is not None:
            indices[non_admitted_mask] = -1

        update_slot_indices = indices.clone()
        still_missing = indices < 0
        if still_missing.any():
            update_slot_indices[still_missing] = -1

        return indices, update_slot_indices, non_admitted_mask


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
        # todo: double check
        frequency_counts_int64 = None
        if frequency_counters is not None:
            frequency_counts_int64 = frequency_counters.long()

        if hasattr(storage, "collect_table_sizes"):
            storage.collect_table_sizes(non_blocking=True)

        indices_table_range = get_table_range(offsets, feature_offsets)
        (
            unique_keys,
            reverse_indices,
            unique_indices_table_range,
            lfu_accumulated_frequency,
            unique_size_per_table,
        ) = segmented_unique(
            indices,
            indices_table_range,
            evict_strat,
            frequency_counts_int64,
        )

        num_prefetched_keys = unique_keys.numel()

        if hasattr(storage, "expand_if_need"):
            storage.expand_if_need(unique_size_per_table)

        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range,
            unique_keys.numel(),
        )

        slot_indices = None
        update_slot_indices = None
        non_admitted_mask = None
        direct_storage_rows = None
        direct_storage_slots = None
        direct_storage_table_ptrs = None
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
                outstanding_keys = _scalar_item(outstanding_keys_ref)
                if outstanding_keys > cache_capacity:
                    outstanding_keys_ref -= num_prefetched_keys
                    raise RuntimeError(
                        f"Outstanding prefetched keys "
                        f"({outstanding_keys}) "
                        f"exceed cache capacity ({cache_capacity}). "
                        "Reduce prefetch pipeline depth or increase "
                        "cache size."
                    )
            (
                slot_indices,
                update_slot_indices,
                non_admitted_mask,
                direct_storage_rows,
                direct_storage_slots,
                direct_storage_table_ptrs,
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
            # Keep the CPU-side pipeline reservation conservative until
            # forward releases this batch. Reading the sparse rejection count
            # here would synchronize the prefetch stream with the host; the
            # reclaimed cache slots themselves are immediately reusable by
            # the native find-or-insert path.
        elif storage_mode == StorageMode.HBM_DIRECT:
            (
                slot_indices,
                update_slot_indices,
                non_admitted_mask,
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
            non_admitted_mask=non_admitted_mask,
            direct_storage_rows=direct_storage_rows,
            direct_storage_slots=direct_storage_slots,
            direct_storage_table_ptrs=direct_storage_table_ptrs,
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

        is_pooling = pooling_mode != DynamicEmbPoolingMode.NONE

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
            unique_size_per_table,
        ) = segmented_unique(
            indices,
            indices_table_range,
            evict_strat,
            frequency_counts_int64,
        )

        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range,
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standalone forward for generic (DEFAULT) storage mode.

    Does storage.find + admission + init + insert
    in a single pass with no PrefetchState.

    Returns (unique_values, persisted_unique_indices). The latter is an int64
    vector of row indices into the unique-key tensors for keys that are stored
    in ``storage`` after this forward (lookup hit or admitted insert). Used in
    backward to ``storage.insert(..., preserve_existing=True)`` only for those
    rows, without compacting via a boolean mask again.
    """
    with torch.cuda.nvtx.range("_generic_forward_path"):
        device = unique_keys.device
        unique_keys.numel()

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
            )

        key_persisted = founds.clone()

        if h_num_missing == 0:
            n = unique_keys.numel()
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
            with torch.cuda.nvtx.range("op:initializer"):
                initializer(unique_values[:, :emb_dim], indices_to_init, unique_keys)

        if val_dim != emb_dim:
            unique_values[missing_indices, emb_dim:] = storage.init_optimizer_state()

        values_to_insert = unique_values[positions_in_unique]

        with torch.cuda.nvtx.range("op:storage_insert"):
            storage.insert(
                keys_to_insert,
                table_ids_to_insert,
                values_to_insert,
                scores_to_insert,
            )
        key_persisted[positions_in_unique] = True
        with torch.cuda.nvtx.range("op:flagged_compact"):
            _, persisted_unique_indices, _ = flagged_compact(
                key_persisted, [unique_keys]
            )
        return unique_values, persisted_unique_indices


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
                if prefetch_state.storage_mode == StorageMode.CACHE:
                    direct_rows = prefetch_state.direct_storage_rows
                    direct_table_ptrs = prefetch_state.direct_storage_table_ptrs
                    assert direct_rows is not None
                    assert direct_table_ptrs is not None
                    unique_embs = torch.empty(
                        prefetch_state.unique_keys.numel(),
                        emb_dim,
                        dtype=emb_dtype,
                        device=prefetch_state.unique_keys.device,
                    )
                    preinitialized_mask = None
                    if (
                        prefetch_state.non_admitted_mask is not None
                        and admit_strategy is not None
                    ):
                        # Populate strategy-specific transient rows first. The
                        # mixed kernel consumes the same sparse mask and skips
                        # default generation only when no cache/storage source
                        # exists, avoiding duplicate work and CURAND advances.
                        with torch.cuda.nvtx.range("op:initializer"):
                            if admit_strategy.initialize_non_admitted_embeddings(
                                unique_embs[:, :emb_dim],
                                prefetch_state.non_admitted_mask,
                            ):
                                preinitialized_mask = (
                                    prefetch_state.non_admitted_mask
                                )
                    with torch.cuda.nvtx.range("op:load_or_initialize_mixed"):
                        initializers[0].load_or_initialize_mixed(
                            unique_embs,
                            state.table_ptrs_dev,
                            prefetch_state.slot_indices,
                            direct_table_ptrs,
                            direct_rows,
                            prefetch_state.unique_table_ids,
                            state.table_value_dims,
                            state.table_emb_dims,
                            state.all_dims_vec4,
                            prefetch_state.unique_keys,
                            preinitialized_mask,
                        )
                else:
                    with torch.cuda.nvtx.range("op:load_from_flat"):
                        unique_embs = load_from_flat(
                            state,
                            prefetch_state.slot_indices,
                            prefetch_state.unique_table_ids,
                            copy_mode=CopyMode.EMBEDDING,
                        )
                if out_dim != emb_dim:
                    unique_embs = unique_embs[:, :out_dim]

                if prefetch_state.non_admitted_mask is not None:
                    na = prefetch_state.non_admitted_mask
                    # The mask may contain no true elements. Keeping the
                    # decision on device avoids a host synchronization; both
                    # initializer implementations filter the sparse mask.
                    if prefetch_state.storage_mode != StorageMode.CACHE:
                        with torch.cuda.nvtx.range("op:initializer"):
                            initialized_non_admitted = False
                            if admit_strategy is not None:
                                initialized_non_admitted = (
                                    admit_strategy.initialize_non_admitted_embeddings(
                                        unique_embs[:, :emb_dim],
                                        na,
                                    )
                                )
                            if not initialized_non_admitted:
                                initializers[0](
                                    unique_embs[:, :emb_dim],
                                    na,
                                    prefetch_state.unique_keys,
                                )
                unique_values = None
                persisted_unique_indices = None
            else:
                assert prefetch_state.storage_mode == StorageMode.DEFAULT
                evict_strat = (
                    EvictStrategy(evict_strategy.value) if evict_strategy else None
                )
                unique_values, persisted_unique_indices = _generic_forward_path(
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
                with torch.cuda.nvtx.range("op:gather_embedding"):
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
            ctx.update_slot_indices = prefetch_state.update_slot_indices
            ctx.direct_storage_rows = prefetch_state.direct_storage_rows
            ctx.direct_storage_slots = prefetch_state.direct_storage_slots
            ctx.direct_storage_table_ptrs = (
                prefetch_state.direct_storage_table_ptrs
            )
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
            ctx.persisted_unique_indices = persisted_unique_indices
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

            is_pooling = ctx.pooling_mode != DynamicEmbPoolingMode.NONE
            if is_pooling:
                out_dim = ctx.max_D if ctx.mixed_D else ctx.emb_dim
                with torch.cuda.nvtx.range("op:reduce_grads"):
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
                with torch.cuda.nvtx.range("op:reduce_grads"):
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
                    with torch.cuda.nvtx.range("op:optimizer_update_fused"):
                        optimizer.fused_update_for_flat_table(
                            unique_grads.to(ctx.emb_dtype),
                            ctx.update_slot_indices,
                            state.table_ptrs_dev,
                            unique_table_ids,
                            state.table_value_dims,
                            state.table_emb_dims,
                            state.max_emb_dim,
                            state.all_dims_vec4,
                            state.emb_dtype,
                            fallback_indices=(
                                ctx.direct_storage_rows
                                if ctx.storage_mode == StorageMode.CACHE
                                else None
                            ),
                            fallback_table_ptrs=(
                                ctx.direct_storage_table_ptrs
                                if ctx.storage_mode == StorageMode.CACHE
                                else None
                            ),
                        )

                    counter_owner = (
                        cache if ctx.storage_mode == StorageMode.CACHE else storage
                    )

                    if ctx.storage_mode == StorageMode.CACHE:
                        storage.release_cache_exchange_refs(
                            cache,
                            ctx.update_slot_indices,
                            ctx.direct_storage_slots,
                            ctx.unique_table_ids,
                        )
                    else:
                        counter_owner.decrement_counter(
                            ctx.update_slot_indices, ctx.unique_table_ids
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
                            ctx.emb_dim,
                            ctx.value_dim,
                            all_dims_vec4,
                        )
                    pui = ctx.persisted_unique_indices
                    assert pui is not None
                    if pui.numel() > 0:
                        keys_c = ctx.unique_keys[pui]
                        tids_c = unique_table_ids[pui]
                        vals_c = ctx.unique_values[pui]
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
                            )

            return (None,) * 17
