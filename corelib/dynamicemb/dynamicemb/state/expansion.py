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

"""Growing the key map and the value buffers of a state when a table fills up.

A state is expanded before an insert when needed. Used in: the HBM-direct prefetch,
the cache write-back, the generic forward path and ``HybridStorage.load``.
"""

import math
from typing import List, Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import align_to_table_size
from dynamicemb.scored_hashtable import ScoreArg, ScorePolicy, get_scored_table
from dynamicemb.state.table_state import (
    NO_EVICTION_LOAD_FACTOR,
    DynamicEmbTableState,
    collect_row_counts_to_device,
    collect_table_sizes_for_state,
    collect_table_sizes_to_device,
)


def _expand_tables_impl(
    state: DynamicEmbTableState,
    tables_to_expand: List[bool],
    target_capacities: Optional[List[int]] = None,
) -> None:
    """Expand key_index_map and table for the given tables only.

    tables_to_expand: bool list, True for tables to expand.
    target_capacities: optional list of target key_index_map capacities per table.
        When provided and tables_to_expand[i] is True, new capacity for table i is
        target_capacities[i]; otherwise expanding tables get 2x current capacity.

    For tables that need expand: (1) create new key_index_map with new per_table_capacity
    for those tables; (2) extend existing ExtendableBuffer for those tables; (3) for each
    such table, for each export batch (key, score, src_index): insert that batch into new
    key_index_map, load values at src_index, collect (dst_index, values), then store each
    batch's values to dst_index (no key concatenation; load-then-store per batch avoids
    src/dst overlap). For tables that do not expand, key_index_map for that table is
    copied directly from the old key_index_map (copy_table_from). Updates key_index_map,
    capacity (expansion does not change the key set). Mutates state.

    A table that never evicts keeps its value buffer out of this: it is sized by the
    rows the table has handed out, not by the key map (see
    :func:`_grow_values_only`).

    Batches prefetched before the rehash whose backward has not run yet keep
    their keys' ref counts, which move with the keys, but their slots go
    stale: they look them up again (``state.key_map_generation`` counts the
    rehashes)."""
    base_opt = state.options_list[0]
    device = state.device
    enable_overflow = getattr(state.key_index_map, "enable_overflow_", False)
    key_caps = state.key_index_map.per_table_capacity_
    old_global_caps = list(
        getattr(state.key_index_map, "global_capacities_", [0] * state.num_tables)
    )
    new_caps = []
    for i in range(state.num_tables):
        # A table with a global capacity keeps it as it grows (passed on below):
        # its keys go to a rank by the whole table's bucket map, which does not
        # change with the table's size, and the table folds that map into its
        # buckets. More buckets split the old ones, so every key stays on its
        # rank and the re-insert below only moves it within it.
        if tables_to_expand[i]:
            if (
                target_capacities is not None
                and i < len(target_capacities)
                and target_capacities[i] >= 0
            ):
                new_caps.append(target_capacities[i])
            else:
                new_caps.append(2 * key_caps[i])
        else:
            new_caps.append(key_caps[i])
    new_key_index_map = get_scored_table(
        capacity=new_caps,
        bucket_capacity=base_opt.bucket_capacity,
        key_type=base_opt.index_type,
        score_specs=[state.score_policy],
        device=device,
        enable_overflow=enable_overflow,
        # Preserve the eviction routing (default Lex or custom score_function)
        # across rehash.
        score_fn_key=state.key_index_map.score_fn_key_,
        # And whether a key keeps a value row of its own: the expansion below
        # carries rows when it does, and moves values by slot when it does not.
        use_rows=state.key_index_map.use_rows_,
        global_capacities=[gc or None for gc in old_global_caps],
        # And whether it counts references at all.
        use_ref_counter=state.key_index_map.use_ref_counter_,
    )
    for i in range(state.num_tables):
        if tables_to_expand[i]:
            if state.no_eviction:
                continue  # its value buffer follows the rows, see the docstring
            if (
                target_capacities is not None
                and i < len(target_capacities)
                and target_capacities[i] >= 0
            ):
                # Hash buckets can round the requested target up. Grow values
                # by the actual key-map capacity change, including that padding.
                add_rows = new_key_index_map.per_table_capacity_[i] - key_caps[i]
                if add_rows > 0:
                    state.buffers.grow_to(i, state.buffers.num_rows(i) + add_rows)
            else:
                state.buffers.grow_to(i, 2 * state.buffers.num_rows(i))

    old_key_index_map = state.key_index_map
    for table_id in range(state.num_tables):
        if not tables_to_expand[table_id]:
            new_key_index_map.copy_table_from(old_key_index_map, table_id)
            continue
        # The ref counters are flat over the tables: the counter of a table's
        # slot is at its first bucket times the bucket capacity, plus the slot.
        src_base, dst_base = [
            int(key_map.table_bucket_offsets_cpu_[table_id].item())
            * key_map.bucket_capacity_
            for key_map in (old_key_index_map, new_key_index_map)
        ]
        dst_values_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for (
            keys,
            named_scores,
            src_rows,
            src_slots,
        ) in old_key_index_map._batched_export_keys_scores(
            [state.score_policy.name],
            device,
            table_id,
            thresholds=None,
            batch_size=65536,
            return_index=True,
            return_slots=True,
        ):
            if keys.numel() == 0:
                continue
            assert src_slots is not None, "return_slots=True requires slots"
            scores_batch = named_scores[state.score_policy.name].to(torch.uint64)

            score_arg = ScoreArg(
                name=state.score_policy.name,
                value=scores_batch,
                policy=ScorePolicy.ASSIGN,
            )
            if new_key_index_map.use_rows_:
                # The key keeps the row its value sits in, so the value stays put.
                dst_slots = new_key_index_map.insert_carrying_rows(
                    keys, table_id, score_arg, src_rows
                )
            else:
                tid_tensor = torch.full(
                    (keys.numel(),), table_id, dtype=torch.int64, device=device
                )
                dst_slots = new_key_index_map.insert(keys, tid_tensor, score_arg)
            # Single-value re-insert above only restores the leading score word.
            # For multi-word layouts (LruLfu: timestamp + frequency) copy the
            # whole score block so the frequency (and exact timestamp) survive
            # rehash; otherwise LFU eviction would see all frequencies reset to 0.
            if new_key_index_map.num_scores_ > 1:
                new_key_index_map.copy_score_blocks_from(
                    old_key_index_map, table_id, src_slots, dst_slots
                )
            if new_key_index_map._ref_counter is not None:
                new_key_index_map._ref_counter[
                    dst_base + dst_slots
                ] = old_key_index_map._ref_counter[src_base + src_slots]

            if not new_key_index_map.use_rows_:
                # Value row == hash slot: the value has to follow its key.
                values_batch = state.buffers.load(table_id, src_slots)
                dst_values_list.append((dst_slots, table_id, values_batch))

        if new_key_index_map.use_rows_:
            # Reclaimed (erased) slots still own their rows; keep them too.
            new_key_index_map.carry_rows_from(old_key_index_map, table_id)

        for dst_indices, table_id, values_batch in dst_values_list:
            state.buffers.store(table_id, dst_indices, values_batch)

    state.key_index_map = new_key_index_map
    state.key_map_generation += 1
    state.capacity = new_key_index_map.capacity()
    if new_key_index_map.use_rows_:
        new_key_index_map.set_row_limits(
            [state.buffers.num_rows(i) for i in range(state.num_tables)]
        )


def _value_rows_target(current_rows: int, need_rows: int, key_capacity: int) -> int:
    """Value rows a table that never evicts should have: its current ones, unless
    ``need_rows`` do not fit, then at least double them -- but never more than the
    key map has slots, since a slot owns one row and no more can ever be handed out."""
    if need_rows <= current_rows:
        return current_rows
    return max(current_rows, min(key_capacity, max(2 * current_rows, need_rows)))


def _grow_values_only(state: DynamicEmbTableState, need_rows: List[int]) -> None:
    """Grow the value buffer of a table that never evicts, leaving its key map.

    The key map is sized by the keys the table holds; the value buffer by the rows
    it has handed out, which can be more (an erased key's slot keeps its row until
    another key takes the slot) but never more than the key map's slots.
    """
    grown = False
    for i in range(state.num_tables):
        rows = state.buffers.num_rows(i)
        target = _value_rows_target(
            rows, need_rows[i], state.key_index_map.per_table_capacity_[i]
        )
        if target > rows:
            state.buffers.grow_to(i, target)
            grown = True
    if grown:
        state.key_index_map.set_row_limits(
            [state.buffers.num_rows(i) for i in range(state.num_tables)]
        )


def get_expand_info(
    state: DynamicEmbTableState,
    table_sizes: torch.Tensor,
    unique_per_table: torch.Tensor,
) -> Tuple[List[bool], List[int]]:
    """Return per-table expand flags and target capacities.

    Args:
        state: Table state (for capacity and options).
        table_sizes: Current size per table (length num_tables), CPU tensor: the keys held
            (see :func:`collect_table_sizes_to_device`). A table that never evicts grows its
            value buffer separately, by the rows handed out (:func:`_grow_values_only`).
        unique_per_table: Number of new unique keys per table to add (length num_tables), CPU tensor.

    Returns:
        (results, target_capacities): results[i] True if table i needs expansion;
        target_capacities[i] is the desired new key_index_map capacity for table i (or -1 if not expanding).

    Expansion uses ``max_load_factor`` for both NO_EVICTION and other strategies:
    expand when ``(current_size + n_new) / KIM_cap > max_load_factor`` (and ``max_lf > 0``),
    to at least ``(current_size + n_new) / max_load_factor`` so the inserted keys fit at
    ``max_load_factor``.
    NO_EVICTION does **not** apply ``max_capacity`` as an upper bound; other strategies still cap
    the target with ``max_capacity`` when it is set.
    """
    assert (
        unique_per_table.device.type == "cpu"
    ), "unique_per_table must be a CPU tensor"

    is_no_eviction = state.no_eviction
    results = [False] * state.num_tables
    target_capacities = [-1] * state.num_tables

    for table_id in range(state.num_tables):
        current_size = table_sizes[table_id].item()
        n_new = unique_per_table[table_id].item()
        if n_new == 0:
            continue
        cap = state.key_index_map.per_table_capacity_[table_id]
        new_total = current_size + n_new
        opt = state.options_list[table_id]
        max_lf = NO_EVICTION_LOAD_FACTOR if is_no_eviction else opt.max_load_factor
        if max_lf <= 0:
            continue
        if is_no_eviction:
            # Same load-factor rule as non–NO_EVICTION, but do not cap by max_capacity.
            if new_total / cap > max_lf:
                results[table_id] = True
                target_capacities[table_id] = max(
                    cap * 2,
                    align_to_table_size(
                        math.ceil(new_total / max_lf), opt.bucket_capacity
                    ),
                )
        else:
            max_cap = opt.max_capacity
            if max_cap is not None and cap >= max_cap:
                continue
            if new_total / cap > max_lf:
                results[table_id] = True
                target = min(
                    max_cap,
                    max(
                        cap * 2,
                        align_to_table_size(
                            math.ceil(new_total / max_lf), opt.bucket_capacity
                        ),
                    ),
                )
                # A table with a global capacity folds the whole table's bucket map
                # into the buckets it has (g % num_buckets). Where num_buckets does not
                # divide the shard's P buckets, a bucket holds floor(P / num_buckets) or
                # ceil(P / num_buckets) of them, a run of P mod num_buckets consecutive
                # buckets the extra one: just under P that is one against two, and at
                # the usual load factor the doubled buckets are full and evict. So it
                # does not stop between P / 2 and P: it grows to P, where the fold is
                # the identity.
                global_caps = getattr(state.key_index_map, "global_capacities_", None)
                if global_caps and global_caps[table_id] and 2 * target > max_cap:
                    target = max_cap
                target_capacities[table_id] = target
    return results, target_capacities


def _expand_if_need_no_eviction(
    state: DynamicEmbTableState, unique_size_per_table: torch.Tensor
) -> None:
    """Expansion for a table that never evicts.

    Two things grow, each by what it is made to hold:

    * the key map, when the keys it holds (plus this batch) pass the load factor;
    * the value buffer, when the rows it has handed out (plus this batch) pass
      the rows it has -- up to the key map's slots, since a slot owns one row.

    So erasing keys lets new ones come in without growing the key map, and a table
    that churns its keys settles at one row per slot instead of growing forever.
    Like the other strategies this works from estimates (the last confirmed
    numbers plus every batch's unique keys, which can only overcount) and confirms
    them against the device, one sync, only when an estimate says something has to
    grow.
    """
    if state.estimated_row_counts is None:
        state.estimated_row_counts = torch.zeros(
            state.num_tables, dtype=torch.int64, pin_memory=True
        )
    keys_est, rows_est = state.estimated_table_sizes, state.estimated_row_counts

    def rows_short(rows: torch.Tensor) -> bool:
        # A buffer that already has a row for every slot of the key map has nothing
        # more to grow to (see _value_rows_target), so the rows it is asked for are no
        # pressure: counting them would make every batch of a table that has settled
        # at one row per slot ask the device for numbers that change nothing. The key
        # map's own growth is checked apart, by get_expand_info, and when it grows the
        # buffer is short of slots again.
        need = rows + unique_size_per_table
        return any(
            int(need[i]) > state.buffers.num_rows(i)
            and state.buffers.num_rows(i) < state.key_index_map.per_table_capacity_[i]
            for i in range(state.num_tables)
        )

    est_results, _ = get_expand_info(state, keys_est, unique_size_per_table)
    if any(est_results) or rows_short(rows_est):
        keys = collect_table_sizes_to_device(state).cpu()
        rows = collect_row_counts_to_device(state).cpu()
        keys_est.copy_(keys)
        rows_est.copy_(rows)
        results, targets = get_expand_info(state, keys, unique_size_per_table)
        need_rows = (rows + unique_size_per_table).tolist()
        if any(results):
            _expand_tables_impl(state, results, targets)
        # The value buffers follow the rows, whether or not the key map grew.
        _grow_values_only(state, need_rows)
    keys_est.add_(unique_size_per_table)
    rows_est.add_(unique_size_per_table)


def expand_if_need_impl(
    state: DynamicEmbTableState,
    unique_size_per_table: torch.Tensor,
) -> None:
    """Accumulate per-table unique counts, optionally collect size and expand.

    unique_size_per_table is the CPU tensor from segmented_unique (shape
    (num_tables,) with per-table unique counts).
    When a second opinion on table sizes is needed, calls
    :func:`collect_table_sizes_for_state` with ``non_blocking=False``.
    """
    assert (
        unique_size_per_table.device.type == "cpu"
    ), "unique_size_per_table must be a CPU tensor"
    if state.estimated_table_sizes is None:
        state.estimated_table_sizes = torch.zeros(
            state.num_tables, dtype=torch.int64, pin_memory=True
        )
    if state.no_eviction:
        _expand_if_need_no_eviction(state, unique_size_per_table)
        return
    estimated_results, target_capacities = get_expand_info(
        state, state.estimated_table_sizes, unique_size_per_table
    )
    if any(estimated_results):
        if state.collect_table_sizes_flag:
            _expand_tables_impl(state, estimated_results, target_capacities)
            state.estimated_table_sizes.add_(unique_size_per_table)
            state.collect_table_sizes_flag = False
            return

        state.collect_table_sizes_flag = True
        collect_table_sizes_for_state(state, non_blocking=False)
        expand_results, target_capacities = get_expand_info(
            state, state.estimated_table_sizes, unique_size_per_table
        )
        if any(expand_results):
            _expand_tables_impl(state, expand_results, target_capacities)
            state.collect_table_sizes_flag = False
        state.estimated_table_sizes.add_(unique_size_per_table)
        return

    if state.collect_table_sizes_flag:
        state.collect_table_sizes_flag = False
    state.estimated_table_sizes.add_(unique_size_per_table)
    return
