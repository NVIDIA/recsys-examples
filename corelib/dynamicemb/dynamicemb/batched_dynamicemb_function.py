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

from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import List, Optional, Tuple

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
    _flat_row_indices_for_value_load,
    _insert_and_evict_keys,
    _write_back_rows,
    take_values_of_keys_evicted_in_the_same_insert,
    eval_lookup,
    get_insert_score_arg,
    load_from_flat,
    store_to_flat,
)
from dynamicemb.lookup_layout import EmbeddingLayout
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import CopyMode, MultiTableAdmitter
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
    emb_dtype: torch.dtype
    storage_mode: StorageMode
    slot_indices: Optional[torch.Tensor]
    # Value-buffer row per key -- what forward and backward both address values
    # with. It differs from ``slot_indices`` only under NO_EVICTION, whose rows
    # come from a per-table auto-increment counter unrelated to where the key
    # hashes, and whose key map is deliberately larger than the value buffer, so
    # indexing values by slot reads past the end of it. Every other strategy
    # points this at the slot tensor itself rather than leaving it unset, so a
    # consumer never has to ask which of the two it is holding. Anything indexed
    # by hash slot (the ref counter) uses ``slot_indices``.
    value_row_indices: Optional[torch.Tensor] = None
    non_admitted_positions: Optional[torch.Tensor] = None
    num_prefetched_keys: int = 0
    outstanding_keys_ref: Optional[torch.Tensor] = None
    # Cache mode: where each table starts in the unique arrays, and the node each
    # key holds (int32, -1 for none), which is what the reference to give back in
    # the backward is indexed by. ``unique_size_per_table`` is the per-table count
    # the outstanding-key tally is kept in.
    unique_offsets: Optional[List[int]] = None
    cache_nodes: Optional[torch.Tensor] = None
    unique_size_per_table: Optional[torch.Tensor] = None


def _record_stream_of_prefetch_state(state: PrefetchState) -> None:
    """Tell the allocator the current stream reads what the prefetch allocated.

    The prefetch may run on its own stream, while forward and backward (which
    runs on the stream of its forward) read these tensors on another. Without the
    record, freeing the context returns their memory to the prefetch stream's pool,
    and the next prefetch can overwrite it under a kernel still reading it.
    ``outstanding_keys_ref`` belongs to the module and is never freed.
    """
    stream = torch.cuda.current_stream()
    for f in fields(state):
        t = getattr(state, f.name)
        if isinstance(t, torch.Tensor) and t.is_cuda and f.name != "outstanding_keys_ref":
            t.record_stream(stream)


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
    with torch.cuda.nvtx.range("_apply_admission"):
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

        non_admitted_mask = ~admit_mask
        with torch.cuda.nvtx.range("op:flagged_compact"):
            _, _, (non_admitted_indices,) = flagged_compact(
                non_admitted_mask, [missing_indices]
            )
        if non_admitted_initializer is not None and non_admitted_indices.numel() > 0:
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


def _prefetch_cache_path(
    cache: DynamicEmbCache,
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    unique_offsets: List[int],
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Cache prefetch where the storage is the authority on every key.

    The cache holds no score and decides neither admission nor which keys are kept;
    the storage does, so every key of the step goes through it, cached or not: its
    score is updated, and a key new to the system is admitted and inserted. Only
    then does the cache place anything. The order is what keeps it correct:

    1. pin the keys already cached, so the storage cannot evict them
    2. find every key in the storage, updating its score
    3. decide admission for the keys the storage does not have
    4. give a node to every key that needs one -- those the storage has but the
       cache does not, and those just admitted -- so they are protected too
    5. move rows: write the victims back to the storage rows they already have,
       load the rows with a history, initialise the new ones
    6. insert the admitted keys into the storage, which asks the cache before it
       evicts anything, and drop from the cache whatever the storage evicted

    See docs/lightweight_cache_design.md, 6.5.
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path"):
        device = unique_keys.device
        state = cache._state
        sstate = storage._state
        h_num_total = unique_keys.numel()
        is_no_eviction = sstate.no_eviction_next_index is not None

        if h_num_total == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                None,
                torch.empty(0, dtype=torch.int32, device=device),
            )

        # 1. Pin what is cached.
        with torch.cuda.nvtx.range("op:cache_lookup_pin"):
            nodes = cache.lookup_pin(unique_keys, unique_offsets)

        # 2. Every key, cached or not, is an access: the storage updates its score.
        with torch.cuda.nvtx.range("op:storage_find"):
            (
                h_num_missing,
                missing_keys,
                missing_positions,
                missing_table_ids,
                missing_scores,
                founds,
                score_out,
                slots,
            ) = _find_keys(
                sstate,
                unique_keys,
                unique_table_ids,
                lfu_accumulated_frequency=accumulated_frequency,
            )
        # Where each key's row is in the storage, for the keys it has.
        rows = _flat_row_indices_for_value_load(sstate, founds, score_out, slots)

        # A key the storage does not have may be in the tier below it. It is not
        # new to the system: it is not put to the admitter, and the row it comes
        # in with is the one the lower tier holds.
        lower = storage.lower
        from_lower: Optional[torch.Tensor] = None
        lower_values: Optional[torch.Tensor] = None
        if lower is not None and h_num_missing > 0:
            out = lower.find(
                missing_keys, missing_table_ids, CopyMode.VALUE, missing_scores
            )
            from_lower, lower_values = out[5], out[7]

        # 3. Admission, for the keys that are new to the system.
        non_admitted_positions: Optional[torch.Tensor] = None
        admitted_keys = missing_keys
        admitted_tids = missing_table_ids
        admitted_scores = missing_scores
        admitted_positions = missing_positions
        if h_num_missing > 0 and admitter is not None:
            freq_for_admission = (
                accumulated_frequency[missing_positions]
                if accumulated_frequency is not None
                else None
            )
            if from_lower is None:
                admit_mask = admitter.admit(
                    missing_keys, missing_table_ids, freq_for_admission
                )
            else:
                new_to_system = ~from_lower
                admit_mask = from_lower.clone()
                admit_mask[new_to_system] = admitter.admit(
                    missing_keys[new_to_system],
                    missing_table_ids[new_to_system],
                    freq_for_admission[new_to_system]
                    if freq_for_admission is not None
                    else None,
                )
            admitted_keys = missing_keys[admit_mask]
            admitted_tids = missing_table_ids[admit_mask]
            admitted_scores = (
                missing_scores[admit_mask] if missing_scores is not None else None
            )
            admitted_positions = missing_positions[admit_mask]
            non_admit = ~admit_mask
            if _bool_item(non_admit.any()):
                non_admitted_positions = missing_positions[non_admit]

        # 4. A node for every key that needs one and does not have it.
        need_node = (nodes < 0) & founds
        if admitted_positions.numel() > 0:
            need_node[admitted_positions] = True
        if not _bool_item(need_node.any()):
            return nodes.to(torch.int64), non_admitted_positions, nodes

        with torch.cuda.nvtx.range("op:flagged_compact"):
            _, alloc_positions, (alloc_keys, alloc_tids) = flagged_compact(
                need_node, [unique_keys, unique_table_ids]
            )
        counts = torch.bincount(alloc_tids, minlength=state.num_tables).tolist()
        alloc_offsets = [0]
        for n in counts:
            alloc_offsets.append(alloc_offsets[-1] + int(n))

        with torch.cuda.nvtx.range("op:cache_allocate"):
            victim_keys, has_victim, new_nodes = cache.allocate(
                alloc_keys, alloc_positions, alloc_offsets, nodes
            )
        if _bool_item((new_nodes < 0).any()):
            raise RuntimeError(
                "The cache has no row to give: every row of some table is held by "
                "a step in flight. Reduce the prefetch pipeline depth or increase "
                "the cache size."
            )
        new_rows = new_nodes.to(torch.int64)

        # 5a. The victims go back to the storage rows they already have. They are
        # written before anything is inserted, so those rows cannot have been
        # given to another key.
        victim_nodes = new_rows[has_victim]
        victim_keys = victim_keys[has_victim]
        victim_tids = alloc_tids[has_victim]
        if victim_keys.numel() > 0:
            with torch.cuda.nvtx.range("op:load_from_flat"):
                victim_values = load_from_flat(
                    state, victim_nodes, victim_tids, copy_mode=CopyMode.VALUE
                )
            (
                _h,
                _mk,
                _mp,
                _mt,
                _ms,
                v_founds,
                v_score,
                v_slots,
            ) = _find_keys(sstate, victim_keys, victim_tids, const_lookup=True)
            v_rows = _flat_row_indices_for_value_load(
                sstate, v_founds, v_score, v_slots
            )
            if _bool_item(v_founds.any()):
                with torch.cuda.nvtx.range("op:store_to_flat"):
                    store_to_flat(
                        sstate,
                        v_rows[v_founds],
                        victim_tids[v_founds],
                        victim_values[v_founds],
                    )
            if not _bool_item(v_founds.all()):
                # A cached key the storage no longer holds: the cache copy is the
                # only one, so it goes back by insert.
                gone = ~v_founds
                _write_back_rows(
                    storage, victim_keys[gone], victim_tids[gone], victim_values[gone]
                )

        # 5b. Rows with a history come in from the storage.
        alloc_in_storage = founds[alloc_positions]
        if _bool_item(alloc_in_storage.any()):
            old_positions = alloc_positions[alloc_in_storage]
            old_tids = alloc_tids[alloc_in_storage]
            with torch.cuda.nvtx.range("op:load_from_flat"):
                old_values = load_from_flat(
                    sstate, rows[old_positions], old_tids, copy_mode=CopyMode.VALUE
                )
            with torch.cuda.nvtx.range("op:store_to_flat"):
                store_to_flat(state, new_rows[alloc_in_storage], old_tids, old_values)

        # 5c. New keys are initialised in their cache rows.
        alloc_is_new = ~alloc_in_storage
        if _bool_item(alloc_is_new.any()):
            n_new = int(_scalar_item(alloc_is_new.sum()))
            init_vals = torch.empty(n_new, max_val_dim, dtype=emb_dtype, device=device)
            init_indices = torch.arange(n_new, dtype=torch.int64, device=device)
            new_keys = alloc_keys[alloc_is_new]
            new_tids = alloc_tids[alloc_is_new]
            with torch.cuda.nvtx.range("op:initializer"):
                initializer(init_vals[:, :max_emb_dim], new_keys, new_tids, init_indices)
            if max_val_dim != max_emb_dim:
                state.optimizer.reset_optimizer_states(
                    init_vals[:, max_emb_dim:],
                    emb_dims=state.table_emb_dims[new_tids],
                )
            if from_lower is not None and _bool_item(from_lower.any()):
                # What the lower tier holds replaces the initial values.
                lower_row = torch.full((h_num_total,), -1, dtype=torch.int64, device=device)
                lower_row[missing_positions[from_lower]] = torch.arange(
                    int(_scalar_item(from_lower.sum())), dtype=torch.int64, device=device
                )
                row_of_new = lower_row[alloc_positions[alloc_is_new]]
                has_row = row_of_new >= 0
                init_vals[has_row] = lower_values[from_lower][row_of_new[has_row]]
            with torch.cuda.nvtx.range("op:store_to_flat"):
                store_to_flat(state, new_rows[alloc_is_new], new_tids, init_vals)

        # 6. The admitted keys enter the storage. Their rows hold nothing there yet
        # -- the cache copy is the newest -- and the storage asks the cache before
        # it evicts, so nothing a step is using can go.
        if admitted_keys.numel() > 0:
            score_arg = get_insert_score_arg(
                sstate,
                admitted_keys.numel(),
                device,
                admitted_scores,
                table_ids=admitted_tids,
            )
            with torch.cuda.nvtx.range("op:storage_insert"):
                (
                    new_slots,
                    num_evicted,
                    evicted_keys,
                    evicted_tids,
                    evicted_slots,
                ) = sstate.key_index_map.insert(
                    admitted_keys,
                    admitted_tids,
                    score_arg,
                    collect_evicted=True,
                    guard=cache.eviction_guard,
                )
            if _bool_item((new_slots < 0).any()):
                raise RuntimeError(
                    "The backing store could not make room for a new key: every key "
                    "it could evict is in use by a step in flight. Increase its "
                    "capacity or reduce the prefetch pipeline depth."
                )
            # What the storage keeps about its own evictions, as before.
            if EvictedItemMode.RETAIN_KEY in sstate.evicted_item_mode:
                _append_evicted(sstate, evicted_keys, evicted_tids, num_evicted)

            # Those keys have left the storage, so they leave the cache too.
            n_ev = int(num_evicted.item())
            if n_ev > 0:
                ev_keys = evicted_keys[:n_ev]
                ev_tids = evicted_tids[:n_ev]
                ev_nodes = cache.find_nodes(ev_keys, ev_tids)
                if lower is not None:
                    # They go down a tier with their values. Their storage rows
                    # are not yet written over -- the new keys' rows are filled
                    # later -- but a cached one is newer than its storage row.
                    ev_values = load_from_flat(
                        sstate,
                        evicted_slots[:n_ev],
                        ev_tids,
                        copy_mode=CopyMode.VALUE,
                    )
                    cached = ev_nodes >= 0
                    if _bool_item(cached.any()):
                        ev_values[cached] = load_from_flat(
                            state,
                            ev_nodes[cached].to(torch.int64),
                            ev_tids[cached],
                            copy_mode=CopyMode.VALUE,
                        )
                    lower.insert(ev_keys, ev_tids, ev_values)
                cache.invalidate(ev_nodes, ev_tids)

        # 7. The rows are written: they are hits from now on.
        cache.publish(new_nodes, alloc_offsets)

        return nodes.to(torch.int64), non_admitted_positions, nodes


def _prefetch_hbm_direct_path(
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """HBM-direct prefetch with counter protection.

    Only admitted keys are inserted.  Non-admitted keys get slot_indices = -1
    and their positions are returned for lazy initialization at forward time.

    Returns ``(slot_indices, non_admitted_positions, value_row_indices)``. The
    two index spaces are kept apart because they only coincide by accident:
    ``slot_indices`` addresses the hash table (and the ref counter beside it),
    ``value_row_indices`` addresses the flat value buffer. For NO_EVICTION they
    differ -- rows are handed out by a per-table auto-increment counter while
    slots come from the key's hash, and the key map is sized
    ``1 / max_load_factor`` times the value buffer, so a slot can point past the
    end of it. Every other strategy returns the slot tensor itself as the rows,
    so the caller always gets something it can address values with.
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
                score_out,
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

        # NO_EVICTION stores each key's value row IN its score word, so a lookup
        # hit yields the row in ``score_out`` -- ``indices`` stays a hash slot and
        # must not be used to address the value buffer. Every other strategy
        # addresses values by slot, so ``rows`` is ``indices`` itself and the
        # writes below reach it for free.
        is_no_eviction = state.no_eviction_next_index is not None
        rows = (
            _flat_row_indices_for_value_load(state, found_mask, score_out, indices)
            if is_no_eviction
            else indices
        )

        if h_num_missing == 0:
            return indices, None, rows

        # A key the table does not have may be in the tier below it. Such a key
        # is not new: it is not put to the admitter, and the row it comes in
        # with is the one the lower tier holds.
        lower = storage.lower
        from_lower: Optional[torch.Tensor] = None
        lower_values: Optional[torch.Tensor] = None
        if lower is not None:
            out = lower.find(
                missing_keys, missing_table_ids, CopyMode.VALUE, missing_scores
            )
            from_lower, lower_values = out[5], out[7]

        # Determine admission for missing keys
        non_admitted_positions: Optional[torch.Tensor] = None

        if admitter is not None:
            freq_for_admission = (
                accumulated_frequency[missing_indices]
                if accumulated_frequency is not None
                else None
            )
            if from_lower is None:
                admit_mask = admitter.admit(
                    missing_keys, missing_table_ids, freq_for_admission
                )
            else:
                new_to_system = ~from_lower
                admit_mask = from_lower.clone()
                admit_mask[new_to_system] = admitter.admit(
                    missing_keys[new_to_system],
                    missing_table_ids[new_to_system],
                    freq_for_admission[new_to_system]
                    if freq_for_admission is not None
                    else None,
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
            admit_mask = None
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

            if from_lower is not None:
                # What the lower tier holds replaces the initial values.
                if admit_mask is None:
                    in_lower, lower_rows = from_lower, lower_values
                else:
                    in_lower, lower_rows = from_lower[admit_mask], lower_values[admit_mask]
                init_values[in_lower] = lower_rows[in_lower]

            if lower is not None and not is_no_eviction:
                # What the insert evicts goes down a tier with its value, which
                # is read before the new rows are written over it.
                (
                    new_indices,
                    num_evicted,
                    evicted_keys,
                    evicted_table_ids,
                    evicted_indices,
                    evicted_scores,
                ) = _insert_and_evict_keys(
                    state, admitted_keys, admitted_tids, admitted_scores
                )
                if num_evicted != 0:
                    evicted_values = load_from_flat(
                        state,
                        evicted_indices,
                        evicted_table_ids,
                        copy_mode=CopyMode.VALUE,
                    )
                    select_insert_failed_values(
                        evicted_indices, init_values, evicted_values
                    )
                    take_values_of_keys_evicted_in_the_same_insert(
                        evicted_keys,
                        evicted_table_ids,
                        evicted_values,
                        admitted_keys,
                        admitted_tids,
                        init_values,
                    )
                    lower.insert(
                        evicted_keys, evicted_table_ids, evicted_values, evicted_scores
                    )
                new_rows = new_indices
            else:
                score_arg = get_insert_score_arg(
                    state, n_admitted, device, admitted_scores, table_ids=admitted_tids
                )
                with torch.cuda.nvtx.range("op:storage_insert"):
                    # HBM-direct is a last tier: this insert may evict, and the
                    # victims are dropped here (unlike the cache path, which
                    # spills to storage). Retain them when configured -- this is
                    # the forward-path analogue of the _insert_key_values
                    # collection.
                    if EvictedItemMode.RETAIN_KEY in state.evicted_item_mode:
                        (
                            new_indices,
                            num_evicted,
                            evicted_keys,
                            evicted_table_ids,
                            _evicted_slots,
                        ) = state.key_index_map.insert(
                            admitted_keys,
                            admitted_tids,
                            score_arg,
                            collect_evicted=True,
                        )
                        _append_evicted(
                            state, evicted_keys, evicted_table_ids, num_evicted
                        )
                    else:
                        new_indices = state.key_index_map.insert(
                            admitted_keys,
                            admitted_tids,
                            score_arg,
                        )
                # The insert returns the hash slot; NO_EVICTION's value row is
                # the score it just assigned. Keep both: the counter is indexed
                # by slot, the value buffer by row.
                new_rows = (
                    score_arg.value.to(torch.int64) if is_no_eviction else new_indices
                )
            with torch.cuda.nvtx.range("op:store_to_flat"):
                store_to_flat(state, new_rows, admitted_tids, init_values)

            inserted_mask = new_indices >= 0
            if inserted_mask.any():
                storage.increment_counter(
                    new_indices[inserted_mask],
                    admitted_tids[inserted_mask],
                )

            indices[admitted_unique_positions] = new_indices
            # Skipped while ``rows`` aliases ``indices``: the write above already
            # covered it. Testing identity rather than ``is_no_eviction`` keeps
            # this correct even if ``rows`` ever stops being an alias -- it would
            # simply be filled explicitly with the same values.
            if rows is not indices:
                rows[admitted_unique_positions] = new_rows

        # Non-admitted keys: ensure slot_indices = -1
        if non_admitted_positions is not None:
            indices[non_admitted_positions] = -1
            if rows is not indices:
                rows[non_admitted_positions] = -1

        return indices, non_admitted_positions, rows


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
) -> PrefetchState:
    """Unified prefetch for all storage types (cache, HBM-direct, generic).

    Returns a PrefetchState containing unique embeddings + optimizer states
    and the metadata needed by forward/backward.
    """
    with torch.cuda.nvtx.range("dynamicemb_prefetch"):
        table_num = feature_offsets.numel() - 1
        assert table_num != 0
        emb_dtype = storage.embedding_dtype()
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
        non_admitted_positions = None
        value_row_indices = None
        if caching:
            storage_mode = StorageMode.CACHE
        elif _is_hbm_storage(storage):
            storage_mode = StorageMode.HBM_DIRECT
        else:
            storage_mode = StorageMode.DEFAULT

        unique_offsets: Optional[List[int]] = None
        cache_nodes: Optional[torch.Tensor] = None
        if storage_mode == StorageMode.CACHE:
            # Where each table starts in the unique arrays: the cache is driven
            # table by table, and the per-table counts are already on the host.
            unique_offsets = [0]
            for n in unique_size_per_table.tolist():
                unique_offsets.append(unique_offsets[-1] + int(n))

            if outstanding_keys_ref is not None:
                # A table's rows are all taken once the keys of the steps in flight
                # outnumber them; that is a capacity error, and clearer here than
                # a failed allocate.
                outstanding_keys_ref += unique_size_per_table
                for t in range(table_num):
                    cache_capacity = cache.capacity(t)
                    outstanding_keys = int(outstanding_keys_ref[t])
                    if outstanding_keys > cache_capacity:
                        raise RuntimeError(
                            f"Outstanding prefetched keys of table {t} "
                            f"({outstanding_keys}) "
                            f"exceed its cache capacity ({cache_capacity}). "
                            "Reduce prefetch pipeline depth or increase "
                            "cache size."
                        )
            (
                slot_indices,
                non_admitted_positions,
                cache_nodes,
            ) = _prefetch_cache_path(
                cache,
                storage,
                unique_keys,
                unique_table_ids,
                unique_offsets,
                max_emb_dim,
                max_val_dim,
                emb_dtype,
                initializer,
                evict_strat,
                lfu_accumulated_frequency,
                admitter,
            )
            # The cache addresses its own value buffer by node, which is the row.
            # There is no separate hash slot to tell apart from it: the cache keeps
            # no ref counter beside a hash table, so the two index spaces coincide.
            value_row_indices = slot_indices
        elif storage_mode == StorageMode.HBM_DIRECT:
            (
                slot_indices,
                non_admitted_positions,
                value_row_indices,
            ) = _prefetch_hbm_direct_path(
                storage,
                unique_keys,
                unique_table_ids,
                max_emb_dim,
                max_val_dim,
                initializer,
                evict_strat,
                lfu_accumulated_frequency,
                admitter,
            )

        return PrefetchState(
            unique_keys=unique_keys,
            reverse_indices=reverse_indices,
            unique_table_ids=unique_table_ids,
            lfu_accumulated_frequency=lfu_accumulated_frequency,
            table_num=table_num,
            emb_dtype=emb_dtype,
            slot_indices=slot_indices,
            storage_mode=storage_mode,
            non_admitted_positions=non_admitted_positions,
            num_prefetched_keys=num_prefetched_keys,
            outstanding_keys_ref=outstanding_keys_ref,
            value_row_indices=value_row_indices,
            unique_offsets=unique_offsets,
            cache_nodes=cache_nodes,
            unique_size_per_table=unique_size_per_table,
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
            initializer,
            cache=cache,
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
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
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
            if admitter is not None and accumulated_frequency is not None
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
        initializer: MultiTableInitializer,
        optimizer: BaseDynamicEmbeddingOptimizer,
        layout: EmbeddingLayout,
        admitter=None,
        evict_strategy=None,
        pooling_weights: Optional[torch.Tensor] = None,
        *args,
    ):
        with torch.cuda.nvtx.range("DynamicEmbeddingFunction.forward"):
            _record_stream_of_prefetch_state(prefetch_state)
            max_emb_dim = storage.max_embedding_dim()
            max_val_dim = storage.max_value_dim()
            emb_dtype = storage.embedding_dtype()

            # pooling_weights is validated by the sole caller,
            # BatchedDynamicEmbeddingTablesV2.forward (pooling mode, dtype,
            # numel, and exclusivity with frequency_counters), and again by
            # TORCH_CHECK in gather_embedding_pooled / reduce_grads.
            is_pooling = layout.is_pooling

            use_counter = prefetch_state.slot_indices is not None
            if use_counter:
                state = (
                    cache._state
                    if prefetch_state.storage_mode == StorageMode.CACHE
                    else storage._state
                )
                with torch.cuda.nvtx.range("op:load_from_flat"):
                    unique_embs = load_from_flat(
                        state,
                        # Value buffer: address by row, which is the slot for
                        # every strategy except NO_EVICTION.
                        prefetch_state.value_row_indices,
                        prefetch_state.unique_table_ids,
                        copy_mode=CopyMode.EMBEDDING,
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
                    optimizer,
                    prefetch_state.unique_keys,
                    prefetch_state.unique_table_ids,
                    max_emb_dim,
                    max_val_dim,
                    emb_dtype,
                    initializer,
                    evict_strat,
                    prefetch_state.lfu_accumulated_frequency,
                    admitter,
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
            ctx.outstanding_keys_ref = prefetch_state.outstanding_keys_ref
            ctx.num_prefetched_keys = prefetch_state.num_prefetched_keys
            ctx.unique_offsets = prefetch_state.unique_offsets
            ctx.cache_nodes = prefetch_state.cache_nodes

            # Recover outstanding count at end of forward so it is decremented
            # even when backward is not run, avoiding overflow.
            if prefetch_state.outstanding_keys_ref is not None:
                prefetch_state.outstanding_keys_ref -= (
                    prefetch_state.unique_size_per_table
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
                        )

                    if ctx.storage_mode == StorageMode.CACHE:
                        # Give back the reference the prefetch took on each node.
                        cache.release(ctx.cache_nodes, ctx.unique_offsets)
                    else:
                        # Ref counter sits beside the hash table -> slots.
                        storage.decrement_counter(
                            ctx.slot_indices, ctx.unique_table_ids
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

            return (None,) * 18
