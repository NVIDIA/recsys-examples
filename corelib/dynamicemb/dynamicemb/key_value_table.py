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

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.dynamicemb_config import (
    DEBUG_EMB_INITIALIZER_MOD,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    align_to_table_size,
)
from dynamicemb.extendable_tensor import (
    DeviceExtendableBuffer,
    ExtendableBuffer,
    HostExtendableBuffer,
)
from dynamicemb.optimizer import (
    BaseDynamicEmbeddingOptimizer,
    pad_optimizer_states_from_checkpoint,
    truncate_optimizer_states_for_checkpoint,
)
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb.types import (
    EMBEDDING_TYPE,
    KEY_TYPE,
    OPT_STATE_TYPE,
    SCORE_TYPE,
    Cache,
    CopyMode,
    DynamicEmbInitializerMode,
    Storage,
    torch_dtype_to_np_dtype,
)
from dynamicemb_extensions import EvictStrategy, flagged_compact
from dynamicemb_extensions import load_from_flat_table_contiguous as _load_contiguous
from dynamicemb_extensions import load_from_flat_table_emb as _load_emb
from dynamicemb_extensions import load_from_flat_table_value as _load_value
from dynamicemb_extensions import (
    no_eviction_assign_scores as _no_eviction_assign_scores,
)
from dynamicemb_extensions import segmented_sum_cuda, select_insert_failed_values
from dynamicemb_extensions import store_to_flat_table_contiguous as _store_contiguous
from dynamicemb_extensions import store_to_flat_table_value as _store_value
from torch import Tensor, nn  # usort:skip

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _all_gather_dumped_keys_values(
    keys: Tensor,
    values: Tensor,
    pg: dist.ProcessGroup,
) -> Tuple[Tensor, Tensor]:
    """Gather (keys, values) from all ranks into concatenated CPU tensors.

    keys: (N,) int64 on device; values: (N, D) on device.
    Returns (out_keys_cpu, out_values_cpu) with all ranks' data in rank order.
    """
    device = keys.device
    world_size = dist.get_world_size(group=pg)
    n = keys.numel()
    d_count = torch.tensor([n], dtype=torch.long, device=device)
    gathered_counts = [torch.empty_like(d_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, d_count, group=pg)
    max_n = max(c.item() for c in gathered_counts)
    emb_dim = values.shape[1]
    dtype_val = values.dtype
    keys_pad = torch.zeros(max_n, dtype=torch.int64, device=device)
    values_pad = torch.zeros(max_n, emb_dim, dtype=dtype_val, device=device)
    if n > 0:
        keys_pad[:n] = keys
        values_pad[:n, :] = values
    gathered_keys = [torch.empty_like(keys_pad) for _ in range(world_size)]
    gathered_values = [torch.empty_like(values_pad) for _ in range(world_size)]
    dist.all_gather(gathered_keys, keys_pad, group=pg)
    dist.all_gather(gathered_values, values_pad, group=pg)
    out_keys = torch.cat(
        [gathered_keys[i][: gathered_counts[i].item()] for i in range(world_size)],
        dim=0,
    ).cpu()
    out_values = torch.cat(
        [gathered_values[i][: gathered_counts[i].item()] for i in range(world_size)],
        dim=0,
    ).cpu()
    return out_keys, out_values


# ---------------------------------------------------------------------------
# Utility helpers (continued)
# ---------------------------------------------------------------------------


def save_to_json(data: Dict[str, Any], file_path: str) -> None:
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        raise RuntimeError(f"Error saving data to JSON file: {e}")


def load_from_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading data from JSON file: {e}")


def get_score_policy(score_strategy):
    if score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
        return ScoreSpec(name="timestamp", policy=ScorePolicy.GLOBAL_TIMER)
    elif score_strategy == DynamicEmbScoreStrategy.STEP:
        return ScoreSpec(name="step", policy=ScorePolicy.ASSIGN)
    elif score_strategy == DynamicEmbScoreStrategy.CUSTOMIZED:
        return ScoreSpec(name="customized", policy=ScorePolicy.ASSIGN)
    elif score_strategy == DynamicEmbScoreStrategy.LFU:
        return ScoreSpec(name="frequency", policy=ScorePolicy.ACCUMULATE)
    elif score_strategy == DynamicEmbScoreStrategy.NO_EVICTION:
        return ScoreSpec(name="index", policy=ScorePolicy.ASSIGN)
    else:
        raise RuntimeError("Not supported score strategy.")


def get_uvm_tensor(dim, dtype, device, is_managed=False):
    return torch.zeros(
        dim,
        out=torch.ops.fbgemm.new_unified_tensor(
            torch.zeros(1, device=device, dtype=dtype),
            [dim],
            is_host_mapped=(not is_managed),
        ),
    )


def get_table_ptrs(state: "DynamicEmbTableState") -> torch.Tensor:
    """Return current data pointers of table buffers from state.tables (ExtendableBuffer).
    Uses buffer.tensor() so pointers stay valid after ExtendableBuffer.extend()."""
    return torch.tensor(
        [b.tensor().data_ptr() for b in state.tables],
        dtype=torch.int64,
        device=state.device,
    )


# ---------------------------------------------------------------------------
# DynamicEmbTableState – shared state dataclass
# ---------------------------------------------------------------------------


@dataclass
class DynamicEmbTableState:
    options_list: List[DynamicEmbTableOptions]
    num_tables: int
    device: torch.device
    score_policy: ScoreSpec
    evict_strategy: EvictStrategy
    key_index_map: Any
    capacity: int
    tables: List[ExtendableBuffer]
    table_emb_dims: torch.Tensor
    table_value_dims: torch.Tensor
    table_emb_dims_cpu: List[int]
    table_value_dims_cpu: List[int]
    max_emb_dim: int
    emb_dim: int
    value_dim: int
    emb_dtype: torch.dtype
    all_dims_vec4: bool
    optimizer: BaseDynamicEmbeddingOptimizer
    initial_optim_state: float
    threads_in_wave: int
    score: Optional[int] = None
    training: bool = False
    # Overflow region fields (per-table, only set when overflow is enabled)
    overflow_caps: Optional[List[int]] = None
    # NO_EVICTION: per-table auto-increment index used as insert score (internal only).
    # no_eviction_next_index: CPU pinned tensor (num_tables,); no_eviction_next_index_dev: same on state.device.
    no_eviction_next_index: Optional[torch.Tensor] = None
    no_eviction_next_index_dev: Optional[torch.Tensor] = None
    # Estimated per-table size (last_collected + accumulated unique since collection);
    # CPU tensor of shape (num_tables,), used to avoid key_index_map.size() when not needed.
    estimated_table_sizes: Optional[torch.Tensor] = None
    collect_table_sizes_flag: bool = False


def create_table_state(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    enable_overflow: bool = False,
) -> DynamicEmbTableState:
    if not options:
        raise ValueError("options must be non-empty")

    base_opt = options[0]
    if (
        base_opt.score_strategy == DynamicEmbScoreStrategy.NO_EVICTION
        and enable_overflow
    ):
        raise ValueError(
            "enable_overflow is not supported when score_strategy is NO_EVICTION"
        )
    num_tables = len(options)

    device_idx = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_idx}")
    score_policy = get_score_policy(base_opt.score_strategy)
    evict_strategy = base_opt.evict_strategy.value

    # NO_EVICTION: key_index_map uses max_load_factor=0.5 to avoid eviction; table uses init_capacity.
    bucket_capacity = base_opt.bucket_capacity
    if base_opt.score_strategy == DynamicEmbScoreStrategy.NO_EVICTION:
        no_eviction_max_lf = 0.5
        capacities = []
        for opt in options:
            if opt.init_capacity is not None:
                cap = math.ceil(opt.init_capacity / no_eviction_max_lf)
                aligned = (
                    (cap + bucket_capacity - 1) // bucket_capacity
                ) * bucket_capacity
                capacities.append(aligned)
            else:
                capacities.append(opt.init_capacity)
    else:
        capacities = [opt.init_capacity for opt in options]

    key_index_map = get_scored_table(
        capacity=capacities,
        bucket_capacity=base_opt.bucket_capacity,
        key_type=base_opt.index_type,
        score_specs=[score_policy],
        device=device,
        enable_overflow=enable_overflow,
    )
    capacity = key_index_map.capacity()

    dims = [opt.dim for opt in options]
    max_emb_dim = max(dims)
    emb_dtype = base_opt.embedding_dtype
    emb_dim = max(dims)

    optim_state_dims = [optimizer.get_state_dim(d) for d in dims]
    value_dims = [d + s for d, s in zip(dims, optim_state_dims)]
    value_dim = max(value_dims)
    all_dims_vec4 = all((d % 4) == 0 for d in dims) and all(
        (v % 4) == 0 for v in value_dims
    )

    table_emb_dims = torch.tensor(dims, dtype=torch.int64, device=device)
    table_value_dims = torch.tensor(value_dims, dtype=torch.int64, device=device)

    key_index_map_caps = key_index_map.per_table_capacity_

    # Table (embedding) capacity may differ from key_index_map in NO_EVICTION:
    # key_index_map is larger (by max_load_factor); table uses init_capacity per table.
    if base_opt.score_strategy == DynamicEmbScoreStrategy.NO_EVICTION:
        table_caps = [
            (
                opt.init_capacity
                if opt.init_capacity is not None
                else key_index_map_caps[i]
            )
            for i, opt in enumerate(options)
        ]
    else:
        table_caps = list(key_index_map_caps)

    overflow_caps_list: Optional[List[int]] = None

    if enable_overflow:
        ovf_cap = key_index_map.overflow_bucket_capacity_
        overflow_caps_list = [ovf_cap] * num_tables

    tables: List[ExtendableBuffer] = []
    for i, (cap, vd) in enumerate(zip(table_caps, value_dims)):
        total_cap = cap
        if enable_overflow:
            total_cap += overflow_caps_list[i]
        shape = (total_cap, vd)
        if base_opt.local_hbm_for_values == 0:
            tables.append(HostExtendableBuffer(shape, emb_dtype, device))
        else:
            tables.append(DeviceExtendableBuffer(shape, emb_dtype, device))

    props = torch.cuda.get_device_properties(device_idx)
    threads_in_wave = (
        props.multi_processor_count * props.max_threads_per_multi_processor
    )

    return DynamicEmbTableState(
        options_list=options,
        num_tables=num_tables,
        device=device,
        score_policy=score_policy,
        evict_strategy=evict_strategy,
        key_index_map=key_index_map,
        capacity=capacity,
        tables=tables,
        table_emb_dims=table_emb_dims,
        table_value_dims=table_value_dims,
        table_emb_dims_cpu=dims,
        table_value_dims_cpu=value_dims,
        max_emb_dim=max_emb_dim,
        emb_dim=emb_dim,
        value_dim=value_dim,
        emb_dtype=emb_dtype,
        all_dims_vec4=all_dims_vec4,
        optimizer=optimizer,
        initial_optim_state=optimizer.get_initial_optim_states(),
        threads_in_wave=threads_in_wave,
        score=None,
        training=False,
        overflow_caps=overflow_caps_list,
        no_eviction_next_index=(
            torch.zeros(num_tables, dtype=torch.int64, pin_memory=True)
            if base_opt.score_strategy == DynamicEmbScoreStrategy.NO_EVICTION
            else None
        ),
        no_eviction_next_index_dev=(
            torch.zeros(num_tables, dtype=torch.int64, device=device)
            if base_opt.score_strategy == DynamicEmbScoreStrategy.NO_EVICTION
            else None
        ),
        estimated_table_sizes=torch.zeros(
            num_tables, dtype=torch.int64, pin_memory=True
        ),
    )


def collect_table_sizes_to_device(state: DynamicEmbTableState) -> torch.Tensor:
    """Collect per-table sizes (main table only, no overflow) into a tensor on state.device.

    Uses an async CUDA kernel when key_index_map exposes table_bucket_offsets_
    and bucket_sizes; otherwise falls back to a sync Python loop. No GPU-CPU
    synchronization when the kernel path is used.

    Returns:
        Tensor of shape (num_tables,) dtype torch.int64 on state.device.
    """
    km = state.key_index_map
    return segmented_sum_cuda(km.bucket_sizes, km.table_bucket_offsets_)


def collect_table_sizes_for_state(
    state: DynamicEmbTableState, non_blocking: bool = True
) -> None:
    """Copy device table sizes into ``state.estimated_table_sizes`` when flag is set."""
    if state.no_eviction_next_index is not None:
        state.no_eviction_next_index.copy_(
            state.no_eviction_next_index_dev, non_blocking=non_blocking
        )
    if not state.collect_table_sizes_flag:
        return
    table_sizes = collect_table_sizes_to_device(state)
    state.estimated_table_sizes.copy_(table_sizes, non_blocking=non_blocking)


# ---------------------------------------------------------------------------
# Storage expansion (expand before insert when needed)
# Used in: prefetch HBM direct, cache write-back, generic forward, HybridStorage.load
# ---------------------------------------------------------------------------


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
    capacity; state.no_eviction_next_index is not changed (expansion
    does not change the key set). Mutates state."""
    base_opt = state.options_list[0]
    device = state.device
    enable_overflow = getattr(state.key_index_map, "enable_overflow_", False)
    key_caps = state.key_index_map.per_table_capacity_
    new_caps = []
    for i in range(state.num_tables):
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
    )
    for i in range(state.num_tables):
        if tables_to_expand[i]:
            if (
                target_capacities is not None
                and i < len(target_capacities)
                and target_capacities[i] >= 0
            ):
                # Target is always new key_index_map capacity; grow value buffer by ΔKIM
                # (NO_EVICTION may start with value rows < KIM cap, same formula as non–NO_EVICTION).
                add_rows = target_capacities[i] - key_caps[i]
                vd = state.table_value_dims_cpu[i]
                if add_rows > 0:
                    state.tables[i].extend((add_rows, vd))
            else:
                state.tables[i].extend(state.tables[i].shape)

    old_key_index_map = state.key_index_map
    for table_id in range(state.num_tables):
        if not tables_to_expand[table_id]:
            new_key_index_map.copy_table_from(old_key_index_map, table_id)
            continue
        dst_values_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for (
            keys,
            named_scores,
            indices,
        ) in old_key_index_map._batched_export_keys_scores(
            [state.score_policy.name],
            device,
            table_id,
            thresholds=None,
            batch_size=65536,
            return_index=True,
        ):
            if keys.numel() == 0:
                continue
            assert indices is not None, "return_index=True requires indices"
            scores_batch = named_scores[state.score_policy.name].to(torch.uint64)
            src_indices = indices

            score_arg = ScoreArg(
                name=state.score_policy.name,
                value=scores_batch,
                policy=ScorePolicy.ASSIGN,
            )
            tid_tensor = torch.full(
                (keys.numel(),), table_id, dtype=torch.int64, device=device
            )
            dst_indices = new_key_index_map.insert(keys, tid_tensor, score_arg)
            new_key_index_map._ref_counter[dst_indices].copy_(
                old_key_index_map._ref_counter[src_indices]
            )

            if state.no_eviction_next_index is None:
                values_batch = load_from_flat_single_table(state, src_indices, table_id)
                dst_values_list.append((dst_indices, table_id, values_batch))

        for dst_indices, table_id, values_batch in dst_values_list:
            store_to_flat_single_table(state, dst_indices, table_id, values_batch)

    state.key_index_map = new_key_index_map
    state.capacity = new_key_index_map.capacity()


def get_expand_info(
    state: DynamicEmbTableState,
    table_sizes: torch.Tensor,
    unique_per_table: torch.Tensor,
) -> Tuple[List[bool], List[int]]:
    """Return per-table expand flags and target capacities.

    Args:
        state: Table state (for capacity and options).
        table_sizes: Current size per table (length num_tables), CPU tensor.
        unique_per_table: Number of new unique keys per table to add (length num_tables), CPU tensor.

    Returns:
        (results, target_capacities): results[i] True if table i needs expansion;
        target_capacities[i] is the desired new key_index_map capacity for table i (or -1 if not expanding).

    Expansion uses ``max_load_factor`` for both NO_EVICTION and other strategies:
    expand when ``(current_size + n_new) / KIM_cap > max_load_factor`` (and ``max_lf > 0``).
    NO_EVICTION does **not** apply ``max_capacity`` as an upper bound; other strategies still cap
    the target with ``max_capacity`` when it is set.
    """
    assert (
        unique_per_table.device.type == "cpu"
    ), "unique_per_table must be a CPU tensor"

    from dynamicemb.dynamicemb_config import DynamicEmbScoreStrategy

    is_no_eviction = (
        state.options_list[0].score_strategy == DynamicEmbScoreStrategy.NO_EVICTION
    )
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
        max_lf = opt.max_load_factor
        if max_lf <= 0:
            continue
        if is_no_eviction:
            # Same load-factor rule as non–NO_EVICTION, but do not cap by max_capacity.
            if new_total / cap > max_lf:
                results[table_id] = True
                target_capacities[table_id] = max(
                    cap * 2,
                    align_to_table_size(new_total, opt.bucket_capacity),
                )
        else:
            max_cap = opt.max_capacity
            if max_cap is not None and cap >= max_cap:
                continue
            if new_total / cap > max_lf:
                results[table_id] = True
                target_capacities[table_id] = min(
                    max_cap,
                    max(cap * 2, align_to_table_size(new_total, opt.bucket_capacity)),
                )
    return results, target_capacities


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


# ---------------------------------------------------------------------------
# Free functions operating on DynamicEmbTableState
# ---------------------------------------------------------------------------


def _flat_row_indices_for_value_load(
    state: DynamicEmbTableState,
    founds: torch.Tensor,
    score_out: torch.Tensor,
    kim_slot_indices: torch.Tensor,
) -> torch.Tensor:
    """Row indices for :func:`load_from_flat` after a KIM lookup.

    For NO_EVICTION, inserts use ``store_to_flat(..., score_arg.value, ...)`` where
    ``score_arg.value`` is the per-key logical row index; lookup ``indices`` are still
    hash-table slot positions and must not be used as flat-buffer rows.
    """
    missing = torch.logical_not(founds)
    if bool(missing.any()):
        assert bool(torch.all(kim_slot_indices[missing] == -1)), (
            "KIM lookup: missing keys must have slot index -1"
        )

    if state.no_eviction_next_index_dev is None:
        return kim_slot_indices
    return torch.where(
        founds,
        score_out.to(device=kim_slot_indices.device, dtype=torch.int64),
        kim_slot_indices,
    )


def _flat_row_indices_from_slots_and_scores(
    state: DynamicEmbTableState,
    slot_indices: torch.Tensor,
    stored_scores: torch.Tensor,
) -> torch.Tensor:
    """Flat-buffer row indices for load/store after export or incremental_dump.

    NO_EVICTION tables use stored score as logical flat row; otherwise slot index
    is the flat row (e.g. TIMESTAMP / LFU).
    """
    if state.no_eviction_next_index is not None:
        return stored_scores.to(device=state.device, dtype=torch.int64)
    return slot_indices.to(device=state.device, dtype=torch.int64)


def load_from_flat(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_ids: torch.Tensor,
    copy_mode: CopyMode,
) -> torch.Tensor:
    N = indices.numel()
    if copy_mode == CopyMode.EMBEDDING:
        max_dim = state.emb_dim
        _load = _load_emb
    else:
        max_dim = state.value_dim
        _load = _load_value
    output = torch.empty(N, max_dim, dtype=state.emb_dtype, device=state.device)
    if N > 0:
        _load(
            get_table_ptrs(state),
            indices,
            table_ids,
            output,
            state.table_value_dims,
            state.table_emb_dims,
            state.emb_dim,
            state.all_dims_vec4,
        )
    return output


def store_to_flat(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_ids: torch.Tensor,
    values: torch.Tensor,
) -> None:
    if values.dim() == 1:
        values = values.unsqueeze(1)
    _store_value(
        get_table_ptrs(state),
        indices,
        table_ids,
        values.to(state.emb_dtype),
        state.table_value_dims,
        state.table_emb_dims,
        state.emb_dim,
        state.all_dims_vec4,
    )


def load_from_flat_single_table(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_id: int,
) -> torch.Tensor:
    """Load full values for a single table. Returns compact [N, value_dim_t]."""
    N = indices.numel()
    vdim = state.table_value_dims_cpu[table_id]
    output = torch.empty(N, vdim, dtype=state.emb_dtype, device=state.device)
    if N > 0:
        _load_contiguous(
            get_table_ptrs(state),
            indices,
            table_id,
            output,
            state.table_value_dims,
            state.table_emb_dims,
            state.emb_dim,
            state.all_dims_vec4,
        )
    return output


def store_to_flat_single_table(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_id: int,
    values: torch.Tensor,
) -> None:
    """Store full values for a single table. Expects compact [N, value_dim_t]."""
    N = indices.numel()
    if N == 0:
        return
    if values.dim() == 1:
        values = values.unsqueeze(1)
    _store_contiguous(
        get_table_ptrs(state),
        indices,
        table_id,
        values.to(state.emb_dtype),
        state.table_value_dims,
        state.table_emb_dims,
        state.emb_dim,
        state.all_dims_vec4,
    )


def get_find_score_arg(
    state: DynamicEmbTableState,
    num_keys: int,
    device: torch.device,
    lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    *,
    const_lookup: bool = False,
) -> ScoreArg:
    """Build a ScoreArg for find/lookup operations.

    When ``state.training`` is False (eval), returns CONST policy so that
    existing scores in the hash table are never modified.

    When *const_lookup* is True (e.g. flush-time verification), always use the
    same read-only CONST policy regardless of ``state.training``.

    When ``state.training`` is True:
      - LFU: ACCUMULATE with provided or default (ones) frequency.
      - LRU (GLOBAL_TIMER): no explicit value needed.
      - CUSTOMIZED / STEP: ASSIGN with ``state.score``.
    """
    # NO_EVICTION: stored scores are logical flat-buffer row indices. A training-time
    # ASSIGN lookup would run ScorePolicy::update and clobber those slots; use CONST.
    if state.no_eviction_next_index_dev is not None:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    if const_lookup or not state.training:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    # LFU: use provided frequency for ACCUMULATE; must have length num_keys for lookup.
    if state.evict_strategy == EvictStrategy.KLfu:
        if (
            lfu_accumulated_frequency is not None
            and lfu_accumulated_frequency.numel() == num_keys
        ):
            scores = lfu_accumulated_frequency.contiguous()
        else:
            # Fallback: each key counts as 1 so score accumulates by 1 per lookup
            scores = torch.ones(num_keys, device=device, dtype=torch.long)
    elif state.evict_strategy == EvictStrategy.KCustomized:
        scores = torch.empty(num_keys, device=device, dtype=torch.long)
        scores.fill_(state.score)
    else:
        scores = None

    return ScoreArg(
        name=state.score_policy.name,
        value=scores,
        policy=state.score_policy.policy,
    )


def _get_no_eviction_insert_scores(
    state: DynamicEmbTableState,
    table_ids: torch.Tensor,
) -> torch.Tensor:
    """For NO_EVICTION: assign scores via GPU atomicAdd on no_eviction_next_index_dev.

    Returns a GPU tensor of scores; for each table_id, values are in
    [no_eviction_next_index_dev[table_id], no_eviction_next_index_dev[table_id] + count).
    Mutates state.no_eviction_next_index_dev only (no sync to CPU).
    """
    assert state.no_eviction_next_index_dev is not None
    return _no_eviction_assign_scores(state.no_eviction_next_index_dev, table_ids)


def get_insert_score_arg(
    state: DynamicEmbTableState,
    num_keys: int,
    device: torch.device,
    scores: Optional[torch.Tensor] = None,
    preserve_existing: bool = False,
    table_ids: Optional[torch.Tensor] = None,
) -> ScoreArg:
    """Build a ScoreArg for insert operations (new keys).

    *preserve_existing* should be True when re-inserting keys that already
    exist in the table (e.g. backward embedding update) so that their scores
    are not overwritten.  This fixes the bug where backward re-inserts
    incorrectly assigned new scores.

    When *preserve_existing* is False (the common case for genuinely new keys):
      - LRU: GLOBAL_TIMER (no explicit value).
      - LFU / CUSTOMIZED / STEP: ASSIGN with provided *scores* or
        ``state.score`` as default.
      - ACCUMULATE policy is converted to ASSIGN for inserts.

    *table_ids* is required when score_strategy is NO_EVICTION and
    preserve_existing is False (used for per-table atomic score assignment).
    """
    if preserve_existing:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    if state.no_eviction_next_index_dev is not None:
        assert table_ids is not None
        scores = _get_no_eviction_insert_scores(state, table_ids)

    is_lru = state.evict_strategy == EvictStrategy.KLru
    if not is_lru and scores is None:
        scores = torch.empty(num_keys, device=device, dtype=torch.uint64)
        scores.fill_(state.score)

    policy = state.score_policy.policy
    if policy == ScorePolicy.ACCUMULATE:
        policy = ScorePolicy.ASSIGN
    if is_lru and scores is not None:
        policy = ScorePolicy.ASSIGN

    return ScoreArg(name=state.score_policy.name, value=scores, policy=policy)


def _find_keys(
    state: DynamicEmbTableState,
    unique_keys: torch.Tensor,
    table_ids: torch.Tensor,
    lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    *,
    const_lookup: bool = False,
) -> Tuple[
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Key-only find: lookup in hash table, return missing info + slot indices."""
    if unique_keys.dtype != state.key_index_map.key_type:
        unique_keys = unique_keys.to(state.key_index_map.key_type)

    batch = unique_keys.size(0)
    device = unique_keys.device

    score_arg = get_find_score_arg(
        state, batch, device, lfu_accumulated_frequency, const_lookup=const_lookup
    )

    if batch == 0:
        return (
            0,
            torch.empty_like(unique_keys),
            torch.empty(batch, dtype=torch.long, device=device),
            torch.empty_like(table_ids),
            torch.empty(batch, dtype=torch.uint64, device=device)
            if score_arg.value is not None
            else None,
            torch.empty(batch, dtype=torch.bool, device=device),
            torch.empty(batch, dtype=torch.int64, device=device),
            torch.empty(batch, dtype=torch.int64, device=device),
        )

    km = state.key_index_map
    if getattr(km, "enable_overflow_", False):
        score_out, founds, indices = km.lookup_with_overflow(
            unique_keys, table_ids, score_arg
        )
    else:
        score_out, founds, indices = km.lookup(unique_keys, table_ids, score_arg)

    missing = torch.logical_not(founds)
    (
        h_num_missing,
        missing_indices,
        (missing_keys, missing_table_ids, missing_scores),
    ) = flagged_compact(
        missing,
        [unique_keys, table_ids, score_arg.value],
    )

    return (
        h_num_missing,
        missing_keys,
        missing_indices,
        missing_table_ids,
        missing_scores,
        founds,
        score_out,
        indices,
    )


def _insert_key_values(
    state: DynamicEmbTableState,
    unique_keys: torch.Tensor,
    table_ids: torch.Tensor,
    unique_values: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    preserve_existing: bool = False,
) -> None:
    score_arg = get_insert_score_arg(
        state,
        unique_keys.numel(),
        unique_keys.device,
        scores,
        preserve_existing,
        table_ids=table_ids,
    )
    n = unique_keys.numel()
    score_out_flat: Optional[torch.Tensor] = None
    if state.no_eviction_next_index is not None:
        score_out_flat = torch.empty(n, dtype=torch.int64, device=unique_keys.device)
    indices = state.key_index_map.insert(
        unique_keys, table_ids, score_arg, score_out=score_out_flat
    )
    if state.no_eviction_next_index is not None:        
        
        flat_indices = (
            score_arg.value
            if score_arg.value is not None
            else score_out_flat
        )
    else:
        flat_indices = indices
    store_to_flat(state, flat_indices, table_ids, unique_values)


def _insert_and_evict_keys(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    preserve_existing: bool = False,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Key-only insert_and_evict. Returns (indices, num_evicted, evicted_keys,
    evicted_table_ids, evicted_indices, evicted_scores).
    Caller is responsible for loading evicted values and storing new values.

    *preserve_existing* is forwarded to :func:`get_insert_score_arg` (e.g. backward
    re-insert should use True so existing slot scores are not overwritten).
    """
    score_arg = get_insert_score_arg(
        state,
        keys.numel(),
        keys.device,
        scores,
        preserve_existing,
        table_ids=table_ids,
    )
    (
        indices,
        num_evicted,
        evicted_keys,
        evicted_indices,
        evicted_scores,
        evicted_table_ids,
    ) = state.key_index_map.insert_and_evict(keys, table_ids, score_arg)

    return (
        indices if state.no_eviction_next_index is None else score_arg.value,
        num_evicted,
        evicted_keys,
        evicted_table_ids,
        evicted_indices,
        evicted_scores,
    )


def export_keys_values_iter(
    state: DynamicEmbTableState,
    device: torch.device,
    batch_size: int = 65536,
    table_id: int = 0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
    """Export keys, embeddings, opt_states, scores for a logical table.

    NO_EVICTION tables load flat values by stored score (logical row index), not by
    hash slot ``indices`` from export.
    """
    emb_dim_t = state.table_emb_dims_cpu[table_id]
    vdim = state.table_value_dims_cpu[table_id]
    optim_state_dim = vdim - emb_dim_t

    for (
        keys,
        named_scores,
        indices,
    ) in state.key_index_map._batched_export_keys_scores(
        [state.score_policy.name],
        state.device,
        batch_size=batch_size,
        return_index=True,
        table_id=table_id,
    ):
        scores = named_scores[state.score_policy.name]
        flat_rows = _flat_row_indices_from_slots_and_scores(state, indices, scores)
        values = load_from_flat_single_table(state, flat_rows, table_id)
        embeddings = values[:, :emb_dim_t].to(dtype=EMBEDDING_TYPE).contiguous()
        if optim_state_dim != 0:
            opt_states = (
                values[:, -optim_state_dim:].to(dtype=OPT_STATE_TYPE).contiguous()
            ).to(device)
        else:
            opt_states = None
        yield (
            keys.to(device),
            embeddings.to(device),
            opt_states,
            scores.to(SCORE_TYPE).to(device),
        )


def _dump_table(
    state: DynamicEmbTableState,
    table_id: int,
    meta_json_file_path: str,
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: str,
    opt_file_path: str,
    include_optim: bool,
    include_meta: bool,
    timestamp: int,
    current_score: Optional[int] = None,
    append: bool = False,
) -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    if not append and include_meta:
        meta_data = {}
        meta_data.update(state.optimizer.get_opt_args())
        meta_data["evict_strategy"] = str(state.evict_strategy)

        if current_score is not None:
            meta_data["step_score"] = current_score

        save_to_json(meta_data, meta_json_file_path)

    mode = "ab" if append else "wb"
    fkey = open(emb_key_path, mode)
    fembedding = open(embedding_file_path, mode)
    fscore = open(score_file_path, mode)
    fopt_states = open(opt_file_path, mode) if include_optim else None

    for keys, embeddings, opt_states_batch, scores in export_keys_values_iter(
        state, device=device, table_id=table_id
    ):
        fkey.write(keys.cpu().numpy().tobytes())
        fembedding.write(embeddings.cpu().numpy().tobytes())
        if state.evict_strategy == EvictStrategy.KLru:
            scores = timestamp - scores
        fscore.write(scores.cpu().numpy().tobytes())
        if fopt_states and opt_states_batch is not None:
            to_write = truncate_optimizer_states_for_checkpoint(
                state.optimizer,
                state.table_emb_dims_cpu[table_id],
                opt_states_batch,
            )
            fopt_states.write(to_write.cpu().numpy().tobytes())

    fkey.close()
    fembedding.close()
    if fscore:
        fscore.close()
    if fopt_states:
        fopt_states.close()


def _iter_batches_from_files(
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str],
    opt_file_path: Optional[str],
    dim: int,
    optstate_dim: int,
    device: torch.device,
    batch_size: int = 65536,
) -> Iterator[Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]]:
    """Yield (keys, embeddings, scores, opt_states) batches from checkpoint files.

    Handles file I/O, numpy deserialization, and distributed world_size filtering.
    Pass *score_file_path* / *opt_file_path* as ``None`` to skip those files.
    """
    fkey = open(emb_key_path, "rb")
    fembedding = open(embedding_file_path, "rb")
    fscore = (
        open(score_file_path, "rb")
        if score_file_path and os.path.exists(score_file_path)
        else None
    )
    fopt = open(opt_file_path, "rb") if opt_file_path else None
    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    try:
        for start in range(0, num_keys, batch_size):
            n = min(num_keys - start, batch_size)

            keys_bytes = fkey.read(KEY_TYPE.itemsize * n)
            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=torch_dtype_to_np_dtype[KEY_TYPE]),
                dtype=KEY_TYPE,
                device=device,
            )

            emb_bytes = fembedding.read(EMBEDDING_TYPE.itemsize * dim * n)
            embeddings = torch.tensor(
                np.frombuffer(emb_bytes, dtype=torch_dtype_to_np_dtype[EMBEDDING_TYPE]),
                dtype=EMBEDDING_TYPE,
                device=device,
            ).view(-1, dim)

            scores = None
            if fscore:
                score_bytes = fscore.read(SCORE_TYPE.itemsize * n)
                scores = torch.tensor(
                    np.frombuffer(
                        score_bytes, dtype=torch_dtype_to_np_dtype[SCORE_TYPE]
                    ),
                    dtype=SCORE_TYPE,
                    device=device,
                )

            opt_states = None
            if fopt:
                opt_bytes = fopt.read(OPT_STATE_TYPE.itemsize * optstate_dim * n)
                opt_states = torch.tensor(
                    np.frombuffer(
                        opt_bytes, dtype=torch_dtype_to_np_dtype[OPT_STATE_TYPE]
                    ),
                    dtype=OPT_STATE_TYPE,
                    device=device,
                ).view(-1, optstate_dim)

            if world_size > 1:
                masks = keys % world_size == rank
                keys = keys[masks]
                embeddings = embeddings[masks]
                if scores is not None:
                    scores = scores[masks]
                if opt_states is not None:
                    opt_states = opt_states[masks]

            yield keys, embeddings, scores, opt_states
    finally:
        fkey.close()
        fembedding.close()
        if fscore:
            fscore.close()
        if fopt:
            fopt.close()


@dataclass
class _LoadParams:
    meta_data: Dict[str, Any]
    dim: int
    runtime_optstate_dim: int
    file_optstate_dim: int
    include_optim: bool
    num_keys: int


def _validate_load_meta(
    state: DynamicEmbTableState,
    table_id: int,
    meta_json_file_path: str,
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str],
    opt_file_path: Optional[str],
    include_optim: bool,
) -> _LoadParams:
    """Shared validation for checkpoint loading.

    Reads meta JSON, validates opt_type / evict_strategy, resolves
    include_optim, and checks file-size consistency.
    """
    meta_data = load_from_json(meta_json_file_path)
    opt_type = meta_data.get("opt_type", None)
    if opt_type and state.optimizer.get_opt_args().get("opt_type", None) != opt_type:
        include_optim = False
        print(
            f"Optimizer type mismatch: {opt_type} != {state.optimizer.get_opt_args().get('opt_type')}. Will not load optimizer states."
        )

    evict_strategy = meta_data.get("evict_strategy", None)
    if evict_strategy and str(state.evict_strategy) != evict_strategy:
        raise ValueError(
            f"Evict strategy mismatch: {evict_strategy} != {state.evict_strategy}"
        )

    if score_file_path is None:
        print(
            f"Score file {score_file_path} does not exist. Will not load score states."
        )

    if not opt_file_path or not os.path.exists(opt_file_path):
        include_optim = False
        print(
            f"Optimizer file {opt_file_path} does not exist. Will not load optimizer states."
        )

    dim = state.table_emb_dims_cpu[table_id]
    runtime_optstate_dim = state.optimizer.get_state_dim(dim)

    if runtime_optstate_dim == 0:
        include_optim = False

    if include_optim:
        state.optimizer.set_opt_args(meta_data)

    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize
    num_embeddings = (
        os.path.getsize(embedding_file_path) // EMBEDDING_TYPE.itemsize // dim
    )
    if num_keys != num_embeddings:
        raise ValueError(
            f"The number of keys in {emb_key_path} does not match with number of embeddings in {embedding_file_path}."
        )
    if score_file_path and os.path.exists(score_file_path):
        num_scores = os.path.getsize(score_file_path) // SCORE_TYPE.itemsize
        if num_keys != num_scores:
            raise ValueError(
                f"The number of keys in {emb_key_path} does not match with number of scores in {score_file_path}."
            )

    file_optstate_dim = 0
    if include_optim and opt_file_path and os.path.exists(opt_file_path):
        file_bytes = os.path.getsize(opt_file_path)
        if num_keys == 0:
            if file_bytes != 0:
                raise ValueError(
                    f"Optimizer state file {opt_file_path} is non-empty but key file has no keys."
                )
        else:
            row_block = num_keys * OPT_STATE_TYPE.itemsize
            if file_bytes % row_block != 0:
                raise ValueError(
                    f"Optimizer state file {opt_file_path} size {file_bytes} is not divisible by "
                    f"{row_block} (num_keys={num_keys}, dtype itemsize={OPT_STATE_TYPE.itemsize})."
                )
            file_optstate_dim = file_bytes // row_block
            ckpt_dim = state.optimizer.get_ckpt_state_dim(dim)
            if file_optstate_dim != ckpt_dim:
                raise ValueError(
                    f"Optimizer state width in checkpoint is {file_optstate_dim}; expected "
                    f"{ckpt_dim}."
                )

    return _LoadParams(
        meta_data=meta_data,
        dim=dim,
        runtime_optstate_dim=runtime_optstate_dim,
        file_optstate_dim=file_optstate_dim,
        include_optim=include_optim,
        num_keys=num_keys,
    )


def _load_key_values(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    embeddings: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    opt_states: Optional[torch.Tensor] = None,
    table_id: int = 0,
) -> None:
    dim = embeddings.size(1)
    emb_dim_cfg = state.table_emb_dims_cpu[table_id]
    runtime_optstate_dim = state.optimizer.get_state_dim(emb_dim_cfg)
    if not keys.is_cuda:
        raise RuntimeError("Keys must be on GPU")
    if not embeddings.is_cuda:
        raise RuntimeError("Embeddings must be on GPU")
    if scores is not None and not scores.is_cuda:
        raise RuntimeError("Scores must be on GPU")
    if opt_states is not None and not opt_states.is_cuda:
        raise RuntimeError("Opt states must be on GPU")

    if opt_states is None and runtime_optstate_dim > 0:
        opt_states = (
            torch.ones(
                keys.numel(),
                runtime_optstate_dim,
                dtype=state.emb_dtype,
                device=embeddings.device,
            )
            * state.initial_optim_state
        )
    elif opt_states is not None and runtime_optstate_dim > 0:
        opt_states = pad_optimizer_states_from_checkpoint(
            state.optimizer,
            emb_dim_cfg,
            opt_states,
            state.initial_optim_state,
            state.emb_dtype,
            embeddings.device,
        )

    values = (
        torch.cat(
            [embeddings.view(-1, dim), opt_states.view(-1, runtime_optstate_dim)],
            dim=-1,
        )
        if opt_states is not None
        else embeddings
    )

    policy = ScorePolicy.ASSIGN
    tid_tensor = torch.full(
        (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
    )

    if state.no_eviction_next_index is not None:
        scores = _get_no_eviction_insert_scores(state, tid_tensor)
    elif scores is None:
        assert (
            state.evict_strategy == EvictStrategy.KLru
        ), "scores is None for KLru evict strategy is allowed but will be deprecated in future."
        policy = ScorePolicy.GLOBAL_TIMER
    else:
        scores = scores.to(SCORE_TYPE)

    score_arg_insert = ScoreArg(
        name=state.score_policy.name,
        value=scores,
        policy=policy,
    )

    score_out_flat: Optional[torch.Tensor] = None
    if state.no_eviction_next_index is not None:
        score_out_flat = torch.empty(
            keys.numel(), dtype=torch.int64, device=keys.device
        )
    indices = state.key_index_map.insert(
        keys, tid_tensor, score_arg_insert, score_out=score_out_flat
    )
    if state.no_eviction_next_index is not None:
        indices = score_arg_insert.value
    store_to_flat_single_table(state, indices, table_id, values)


# ---------------------------------------------------------------------------
# DynamicEmbCache – Cache interface (key-only find / insert_and_evict)
# ---------------------------------------------------------------------------


class DynamicEmbCache(Cache):
    def __init__(
        self,
        options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        self._state = create_table_state(options, optimizer, enable_overflow=True)
        self._cache_metrics = torch.zeros(10, dtype=torch.long, device="cpu")
        self._record_cache_metrics = False

    # -- Cache interface --

    def increment_counter(
        self,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> None:
        """Increment ref-counter at given per-table slot indices. table_ids must be provided and aligned with slot_indices."""
        self._state.key_index_map.increment_counter(slot_indices, table_ids)

    def decrement_counter(
        self,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> None:
        """Decrement ref-counter at given per-table slot indices. table_ids must be provided and aligned with slot_indices."""
        self._state.key_index_map.decrement_counter(slot_indices, table_ids)

    def lookup(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lookup with overflow fallback. Returns (score_out, founds, indices)."""
        state = self._state
        score_arg = get_find_score_arg(
            state, unique_keys.size(0), unique_keys.device, lfu_accumulated_frequency
        )
        result = state.key_index_map.lookup_with_overflow(
            unique_keys, table_ids, score_arg
        )
        if self._record_cache_metrics:
            self._cache_metrics[0] = unique_keys.size(0)
            founds = result[1]
            self._cache_metrics[1] = founds.sum().item()
        return result

    def insert_and_evict(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Insert with counter-aware eviction and overflow fallback."""
        state = self._state
        score_arg = get_insert_score_arg(
            state, keys.numel(), keys.device, scores, table_ids=table_ids
        )
        result = state.key_index_map.insert_and_evict_with_counter_and_overflow(
            keys, table_ids, score_arg
        )
        if self._record_cache_metrics:
            self._cache_metrics[2] = keys.numel()
            self._cache_metrics[3] = result[1]  # num_evicted
        return result

    def reset(self) -> None:
        self._state.key_index_map.reset()

    @property
    def cache_metrics(self) -> Optional[torch.Tensor]:
        return self._cache_metrics if self._record_cache_metrics else None

    def set_record_cache_metrics(self, record: bool) -> None:
        self._record_cache_metrics = record

    # -- Score management --

    def set_score(self, score: int) -> None:
        self._state.score = score

    @property
    def training(self) -> bool:
        return self._state.training

    @training.setter
    def training(self, value: bool) -> None:
        self._state.training = value

    # -- Convenience accessors --

    @property
    def num_tables(self) -> int:
        return self._state.num_tables

    def embedding_dtype(self) -> torch.dtype:
        return self._state.emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._state.table_emb_dims_cpu[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._state.table_value_dims_cpu[table_id]

    def max_embedding_dim(self) -> int:
        return self._state.emb_dim

    def max_value_dim(self) -> int:
        return self._state.value_dim

    def init_optimizer_state(self) -> float:
        return self._state.initial_optim_state

    def evict_strategy(self) -> EvictStrategy:
        return self._state.evict_strategy

    def size(self) -> int:
        return self._state.key_index_map.size()

    @property
    def key_index_map(self):
        return self._state.key_index_map


# ---------------------------------------------------------------------------
# DynamicEmbStorage – Storage interface (find with values, insert, dump, load)
# ---------------------------------------------------------------------------

class DynamicEmbStorage(Storage):
    def __init__(
        self,
        options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        self._state = create_table_state(
            options,
            optimizer,
        )

    @property
    def key_index_map(self):
        return self._state.key_index_map

    def fill_tables(
        self,
        load_factor: float = 0.95,
        tolerance: float = 1e-5,
    ) -> None:
        """
        Insert random keys into ``key_index_map`` until each logical table reaches
        ``min(capacity, int(load_factor * capacity))`` entries.

        Default ``load_factor`` is ``0.95``; any value above ``0.95`` is clamped
        to ``0.95``.

        After each successful insert batch (and before filling a table), the
        per-table load factor ``size(table_id) / capacity(table_id)`` is compared
        to ``load_factor``. When ``abs(actual - load_factor) <= tolerance`` (default
        ``1e-5``), that table is treated as done even if the integer entry target
        is not reached. Pass ``tolerance=0`` to disable this early exit.

        Keys are uniform in ``[0, torch.iinfo(torch.int64).max - 1]`` (i.e.
        ``[0, 2**63 - 2]``). ``torch.randint`` cannot use ``high = 2**63`` (the
        bound overflows in the C++ API), so ``2**63 - 1`` is never sampled.
        Only the hash map is updated;
        value buffers are not written. Call :meth:`set_score` first if inserts
        require a valid score (e.g. non-LRU strategies).
        """
        if load_factor < 0.0:
            raise ValueError(f"load_factor must be non-negative, got {load_factor}")
        load_factor = min(load_factor, 0.95)
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")

        state = self._state
        km = state.key_index_map
        device = state.device
        key_dtype = km.key_type
        batch_cap = 262144
        max_stale_tries = 1000

        def _as_int(sz: Any) -> int:
            if isinstance(sz, torch.Tensor):
                return int(sz.item())
            return int(sz)

        for table_id in range(state.num_tables):
            cap = km.capacity(table_id)
            if cap == 0:
                continue
            target = min(cap, int(load_factor * cap))
            remaining = max(0, target - _as_int(km.size(table_id)))
            stale_tries = 0

            def _lf_within_tol() -> bool:
                if tolerance <= 0.0:
                    return False
                actual = float(_as_int(km.size(table_id))) / float(cap)
                return abs(actual - load_factor) <= tolerance

            if remaining == 0:
                continue

            while remaining > 0:
                batch_size = min(batch_cap, max(4096, remaining * 4))
                n_gen = min(batch_size, remaining * 8 + 1024)
                # [low, high) with high = iinfo.max (2**63-1) -> values in [0, 2**63-2].
                # high = 2**63 overflows in the randint binding; no bit-packing needed.
                keys = torch.randint(
                    0,
                    torch.iinfo(torch.int64).max,
                    (n_gen,),
                    device=device,
                    dtype=torch.int64,
                ).to(key_dtype)
                keys, uniq_counts = torch.unique(keys, return_counts=True)
                lfu_freq = uniq_counts.to(dtype=torch.long)
                if keys.numel() == 0:
                    stale_tries += 1
                    if stale_tries > max_stale_tries:
                        raise RuntimeError(f"fill_tables: stalled on table {table_id}")
                    continue

                table_ids = torch.full(
                    (keys.numel(),),
                    table_id,
                    device=device,
                    dtype=torch.int64,
                )
                score_find = get_find_score_arg(
                    state,
                    keys.numel(),
                    device,
                    lfu_accumulated_frequency=lfu_freq,
                )
                _, founds, _ = km.lookup(keys, table_ids, score_find)
                missing = torch.logical_not(founds)
                new_keys = keys[missing]
                new_lfu_freq = lfu_freq[missing]
                if new_keys.numel() == 0:
                    stale_tries += 1
                    if stale_tries > max_stale_tries:
                        raise RuntimeError(f"fill_tables: stalled on table {table_id}")
                    continue

                stale_tries = 0
                take = min(int(new_keys.numel()), remaining)
                insert_keys = new_keys[:take]
                insert_lfu = new_lfu_freq[:take]
                insert_tids = torch.full(
                    (take,),
                    table_id,
                    device=device,
                    dtype=torch.int64,
                )
                insert_scores = (
                    insert_lfu.to(dtype=torch.uint64)
                    if state.evict_strategy == EvictStrategy.KLfu
                    else None
                )
                score_ins = get_insert_score_arg(
                    state, take, device, scores=insert_scores
                )
                km.insert(insert_keys, insert_tids, score_ins)
                remaining -= take
                if _lf_within_tol():
                    break

    def expand_if_need(self, unique_size_per_table: torch.Tensor) -> None:
        """Accumulate per-table unique counts, optionally collect size and expand."""
        expand_if_need_impl(self._state, unique_size_per_table)

    def collect_table_sizes(self, non_blocking: bool = True) -> None:
        """Collect per-table sizes from key_index_map into estimated_table_sizes (async copy)."""
        collect_table_sizes_for_state(self._state, non_blocking=non_blocking)

    # -- Storage interface --

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        result = _find_keys(
            self._state, unique_keys, table_ids, lfu_accumulated_frequency
        )
        (
            h_num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            founds,
            score_out,
            indices,
        ) = result
        flat_rows = _flat_row_indices_for_value_load(
            self._state, founds, score_out, indices
        )
        values = load_from_flat(self._state, flat_rows, table_ids, copy_mode=copy_mode)

        return (
            h_num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            founds,
            score_out,
            values,
        )

    def increment_counter(
        self,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> None:
        """Increment ref-counter at given per-table slot indices. table_ids must be provided and aligned with slot_indices."""
        self._state.key_index_map.increment_counter(slot_indices, table_ids)

    def decrement_counter(
        self,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> None:
        """Decrement ref-counter at given per-table slot indices. table_ids must be provided and aligned with slot_indices."""
        self._state.key_index_map.decrement_counter(slot_indices, table_ids)

    def insert(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
    ) -> None:
        _insert_key_values(
            self._state,
            unique_keys,
            table_ids,
            unique_values,
            scores,
            preserve_existing,
        )

    def dump(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: str,
        opt_file_path: str,
        include_optim: bool = True,
        include_meta: bool = True,
        current_score: Optional[int] = None,
        timestamp: int = 0,
    ) -> None:
        _dump_table(
            self._state,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim,
            include_meta,
            timestamp=timestamp,
            current_score=current_score,
        )

    def load(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = True,
        timestamp: int = 0,
    ) -> Optional[int]:
        params = _validate_load_meta(
            self._state,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim,
        )

        self._state.collect_table_sizes_flag = True
        collect_table_sizes_for_state(self._state, non_blocking=False)
        unique_size_per_table = torch.zeros(
            self._state.num_tables, dtype=torch.int64, device=torch.device("cpu")
        )
        unique_size_per_table[table_id] = max(
            0,
            params.num_keys
            + self._state.estimated_table_sizes[table_id].item()
            - self._state.tables[table_id].tensor().size(0),
        )
        self.expand_if_need(unique_size_per_table)

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        for keys, embeddings, scores, opt_states in _iter_batches_from_files(
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path if params.include_optim else None,
            params.dim,
            params.file_optstate_dim,
            device,
        ):
            if scores is not None and self._state.evict_strategy == EvictStrategy.KLru:
                scores = torch.clamp(timestamp - scores, min=0)
            _load_key_values(
                self._state, keys, embeddings, scores, opt_states, table_id=table_id
            )
        return params.meta_data.get("step_score", None)

    def incremental_dump(
        self,
        table_id: int,
        threshold: int,
        pg: Optional[dist.ProcessGroup],
    ) -> Tuple[Tensor, Tensor]:
        """Dump keys and embeddings for one table (score >= threshold). Multi-rank: all_gather so result is concatenated from all ranks."""
        state = self._state
        states_to_dump = [state]
        do_multi_rank_gather = (
            pg is not None
            and dist.is_initialized()
            and dist.get_world_size(group=pg) > 1
        )
        all_keys: List[Tensor] = []
        all_values: List[Tensor] = []
        for s in states_to_dump:
            keys, named_scores, indices = s.key_index_map.incremental_dump(
                {s.score_policy.name: threshold},
                pg=pg,
                return_index=True,
                table_id=table_id,
            )
            emb_dim = s.table_emb_dims_cpu[table_id]
            scores_batch = named_scores[s.score_policy.name]
            flat_rows = _flat_row_indices_from_slots_and_scores(s, indices, scores_batch)
            values = load_from_flat_single_table(s, flat_rows, table_id)
            value = values[:, :emb_dim].to(dtype=s.emb_dtype)
            key = keys.to(s.device) if keys.device.type != "cuda" else keys
            if not do_multi_rank_gather:
                value = value.cpu()
                key = key.cpu() if key.is_cuda else key
            all_keys.append(key)
            all_values.append(value)
        device_for_gather = state.device
        emb_dim_t = state.table_emb_dims_cpu[table_id]
        if all_keys:
            keys_cat = torch.cat(all_keys)
            values_cat = torch.cat(all_values, dim=0)
        else:
            if do_multi_rank_gather:
                keys_cat = torch.empty(0, dtype=torch.int64, device=device_for_gather)
                values_cat = torch.empty(
                    0, emb_dim_t, dtype=state.emb_dtype, device=device_for_gather
                )
            else:
                keys_cat = torch.empty(0, dtype=torch.int64, device="cpu")
                values_cat = torch.empty(0, emb_dim_t, dtype=state.emb_dtype)
        if do_multi_rank_gather:
            keys_cat = keys_cat.to(device_for_gather)
            values_cat = values_cat.to(device_for_gather)
            keys_cat, values_cat = _all_gather_dumped_keys_values(
                keys_cat, values_cat, pg
            )
        elif keys_cat.device.type == "cuda":
            keys_cat = keys_cat.cpu()
            values_cat = values_cat.cpu()
        return keys_cat, values_cat

    # -- Export --

    def export_keys_values(
        self,
        device: torch.device,
        batch_size: int = 65536,
        table_id: int = 0,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
    ]:
        yield from export_keys_values_iter(self._state, device, batch_size, table_id)

    # -- Property accessors --

    def embedding_dtype(self) -> torch.dtype:
        return self._state.emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._state.table_emb_dims_cpu[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._state.table_value_dims_cpu[table_id]

    def max_embedding_dim(self) -> int:
        return self._state.emb_dim

    def max_value_dim(self) -> int:
        return self._state.value_dim

    def init_optimizer_state(self) -> float:
        return self._state.initial_optim_state

    # -- Score management --

    def set_score(self, score: int) -> None:
        self._state.score = score

    @property
    def training(self) -> bool:
        return self._state.training

    @training.setter
    def training(self, value: bool) -> None:
        self._state.training = value

    def evict_strategy(self) -> EvictStrategy:
        return self._state.evict_strategy

    @property
    def num_tables(self) -> int:
        return self._state.num_tables

    def size(self) -> int:
        return self._state.key_index_map.size()


# ---------------------------------------------------------------------------
# HybridStorage – two-tier storage using two DynamicEmbTableState instances
# ---------------------------------------------------------------------------


class HybridStorage(Storage):
    """Two-tier storage: HBM (GPU) table + host table, disjoint partitions."""

    def __init__(
        self,
        hbm_options: List[DynamicEmbTableOptions],
        host_options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        self._hbm = create_table_state(hbm_options, optimizer)
        self._host = create_table_state(host_options, optimizer)
        self.optimizer = optimizer

    @property
    def tables(self) -> List[DynamicEmbTableState]:
        return [self._hbm, self._host]

    # -- Score management --

    @property
    def training(self) -> bool:
        return self._hbm.training

    @training.setter
    def training(self, value: bool) -> None:
        self._hbm.training = value
        self._host.training = value

    def set_score(self, score: int) -> None:
        self._hbm.score = score
        self._host.score = score

    def evict_strategy(self) -> EvictStrategy:
        return self._hbm.evict_strategy

    # -- Storage property accessors --

    def embedding_dtype(self) -> torch.dtype:
        return self._hbm.emb_dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._hbm.table_emb_dims_cpu[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._hbm.table_value_dims_cpu[table_id]

    def max_embedding_dim(self) -> int:
        return self._hbm.emb_dim

    def max_value_dim(self) -> int:
        return self._hbm.value_dim

    def init_optimizer_state(self) -> float:
        return self._hbm.initial_optim_state

    @property
    def num_tables(self) -> int:
        return self._hbm.num_tables

    def expand_if_need(self, unique_size_per_table: torch.Tensor) -> None:
        """Accumulate per-table unique counts (on host), optionally collect size and expand."""
        expand_if_need_impl(self._host, unique_size_per_table)

    def collect_table_sizes(self, non_blocking: bool = True) -> None:
        """Collect per-table sizes from key_index_map into estimated_table_sizes (host tier only, async copy)."""
        collect_table_sizes_for_state(self._host, non_blocking=non_blocking)

    # -- Two-tier find (with values) --

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        result_hbm = _find_keys(
            self._hbm, unique_keys, table_ids, lfu_accumulated_frequency
        )
        (
            h_num_missing_hbm,
            missing_keys_hbm,
            missing_indices_hbm,
            missing_table_ids_hbm,
            missing_scores_hbm,
            founds_hbm,
            scores_hbm,
            indices_hbm,
        ) = result_hbm

        flat_hbm = _flat_row_indices_for_value_load(
            self._hbm, founds_hbm, scores_hbm, indices_hbm
        )
        values = load_from_flat(self._hbm, flat_hbm, table_ids, copy_mode=copy_mode)

        if h_num_missing_hbm == 0:
            return (
                0,
                missing_keys_hbm,
                missing_indices_hbm,
                missing_table_ids_hbm,
                missing_scores_hbm,
                founds_hbm,
                scores_hbm,
                values,
            )

        result_host = _find_keys(
            self._host,
            missing_keys_hbm,
            missing_table_ids_hbm,
            missing_scores_hbm,
        )
        (
            h_num_missing_both,
            missing_keys_both,
            missing_indices_both,
            missing_table_ids_both,
            missing_scores_both,
            founds_host,
            scores_host,
            indices_host,
        ) = result_host

        flat_host = _flat_row_indices_for_value_load(
            self._host, founds_host, scores_host, indices_host
        )
        host_vals = load_from_flat(
            self._host, flat_host, missing_table_ids_hbm, copy_mode=copy_mode
        )

        host_found_mask = founds_host
        if host_found_mask.any():
            values[missing_indices_hbm[host_found_mask]] = host_vals[host_found_mask]

        founds_combined = founds_hbm.clone()
        founds_combined[missing_indices_hbm[host_found_mask]] = True

        # Merge host-tier scores into output: for keys found in _host, return scores_host
        # so caller sees correct score (scores_hbm has no valid value for those positions).
        output_scores = scores_hbm.clone()
        output_scores[missing_indices_hbm[host_found_mask]] = scores_host[
            host_found_mask
        ]

        global_missing_indices = missing_indices_hbm[missing_indices_both]
        global_missing_scores = missing_scores_both

        if (
            self._hbm.evict_strategy == EvictStrategy.KLfu
            and self._host.evict_strategy == EvictStrategy.KLfu
            and lfu_accumulated_frequency is not None
        ):
            if global_missing_indices.numel() == 0:
                assert (
                    global_missing_scores is None or global_missing_scores.numel() == 0
                )
            else:
                assert global_missing_scores is not None
                assert global_missing_scores.numel() == global_missing_indices.numel()
                expected = lfu_accumulated_frequency[global_missing_indices]
                assert torch.equal(
                    global_missing_scores.long(),
                    expected.long(),
                )

        return (
            h_num_missing_both,
            missing_keys_both,
            global_missing_indices,
            missing_table_ids_both,
            global_missing_scores,
            founds_combined,
            output_scores,
            values,
        )

    # -- Insert: HBM first, evictions to host --

    def insert(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        unique_values: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        preserve_existing: bool = False,
    ) -> None:
        """Insert or update rows.

        * ``preserve_existing=False`` (default): insert into the HBM tier; evicted
          keys are written to the host tier (existing behavior).

        * ``preserve_existing=True`` (e.g. autograd backward refresh): **no insert**.
          Performs the same two-tier lookup as :meth:`find`, and only
          ``store_to_flat`` updates rows that already exist in HBM or host. Keys
          absent from both tiers are ignored.
        """
        if preserve_existing:
            if unique_keys.numel() == 0:
                return
            (
                h_num_missing_hbm,
                missing_keys_hbm,
                missing_indices_hbm,
                missing_table_ids_hbm,
                missing_scores_hbm,
                founds_hbm,
                scores_hbm,
                indices_hbm,
            ) = _find_keys(self._hbm, unique_keys, table_ids)
            if founds_hbm.any():
                flat_hbm = _flat_row_indices_for_value_load(
                    self._hbm, founds_hbm, scores_hbm, indices_hbm
                )
                store_to_flat(
                    self._hbm,
                    flat_hbm[founds_hbm],
                    table_ids[founds_hbm],
                    unique_values[founds_hbm],
                )
            if h_num_missing_hbm > 0:
                (
                    _h_num_missing_both,
                    _missing_keys_both,
                    _missing_indices_both,
                    _missing_table_ids_both,
                    _missing_scores_both,
                    founds_host,
                    scores_host,
                    indices_host,
                ) = _find_keys(
                    self._host,
                    missing_keys_hbm,
                    missing_table_ids_hbm,
                    missing_scores_hbm,
                )
                if founds_host.any():
                    orig_rows = missing_indices_hbm[founds_host]
                    flat_host = _flat_row_indices_for_value_load(
                        self._host, founds_host, scores_host, indices_host
                    )
                    store_to_flat(
                        self._host,
                        flat_host[founds_host],
                        missing_table_ids_hbm[founds_host],
                        unique_values[orig_rows],
                    )
            return

        (
            indices,
            num_evicted,
            evicted_keys,
            evicted_table_ids,
            evicted_indices,
            evicted_scores,
        ) = _insert_and_evict_keys(
            self._hbm,
            unique_keys,
            table_ids,
            scores,
            preserve_existing=False,
        )

        evicted_values = load_from_flat(
            self._hbm, evicted_indices, evicted_table_ids, copy_mode=CopyMode.VALUE
        )
        select_insert_failed_values(evicted_indices, unique_values, evicted_values)
        store_to_flat(self._hbm, indices, table_ids, unique_values)

        if num_evicted != 0:
            _insert_key_values(
                self._host,
                evicted_keys,
                evicted_table_ids,
                evicted_values,
                evicted_scores,
            )

    def incremental_dump(
        self,
        table_id: int,
        threshold: int,
        pg: Optional[dist.ProcessGroup],
    ) -> Tuple[Tensor, Tensor]:
        """Dump keys and embeddings for one table (score >= threshold). Multi-rank: all_gather so result is concatenated from all ranks."""
        states_to_dump = self.tables
        do_multi_rank_gather = (
            pg is not None
            and dist.is_initialized()
            and dist.get_world_size(group=pg) > 1
        )
        all_keys = []
        all_values = []
        for s in states_to_dump:
            keys, named_scores, indices = s.key_index_map.incremental_dump(
                {s.score_policy.name: threshold},
                pg=pg,
                return_index=True,
                table_id=table_id,
            )
            emb_dim = s.table_emb_dims_cpu[table_id]
            scores_batch = named_scores[s.score_policy.name]
            flat_rows = _flat_row_indices_from_slots_and_scores(s, indices, scores_batch)
            values = load_from_flat_single_table(s, flat_rows, table_id)
            value = values[:, :emb_dim].to(dtype=s.emb_dtype)
            key = keys.to(s.device) if keys.device.type != "cuda" else keys
            if not do_multi_rank_gather:
                value = value.cpu()
                key = key.cpu() if key.is_cuda else key
            all_keys.append(key)
            all_values.append(value)
        device_for_gather = states_to_dump[0].device
        emb_dim_t = states_to_dump[0].table_emb_dims_cpu[table_id]
        if all_keys:
            keys_cat = torch.cat(all_keys)
            values_cat = torch.cat(all_values, dim=0)
        else:
            if do_multi_rank_gather:
                keys_cat = torch.empty(0, dtype=torch.int64, device=device_for_gather)
                values_cat = torch.empty(
                    0, emb_dim_t, dtype=self.embedding_dtype(), device=device_for_gather
                )
            else:
                keys_cat = torch.empty(0, dtype=torch.int64, device="cpu")
                values_cat = torch.empty(0, emb_dim_t, dtype=self.embedding_dtype())
        if do_multi_rank_gather:
            keys_cat = keys_cat.to(device_for_gather)
            values_cat = values_cat.to(device_for_gather)
            keys_cat, values_cat = _all_gather_dumped_keys_values(
                keys_cat, values_cat, pg
            )
        elif keys_cat.device.type == "cuda":
            keys_cat = keys_cat.cpu()
            values_cat = values_cat.cpu()
        return keys_cat, values_cat

    # -- Dump: write host first, then append HBM --

    def dump(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: str,
        opt_file_path: str,
        include_optim: bool = True,
        include_meta: bool = True,
        current_score: Optional[int] = None,
        timestamp: int = 0,
    ) -> None:
        _dump_table(
            self._host,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim=include_optim,
            include_meta=include_meta,
            timestamp=timestamp,
            current_score=current_score,
        )

        _dump_table(
            self._hbm,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim=include_optim,
            include_meta=False,
            timestamp=timestamp,
            append=True,
        )

    # -- Load: route through HBM, evictions to host --

    def load(
        self,
        table_id: int,
        meta_json_file_path: str,
        emb_key_path: str,
        embedding_file_path: str,
        score_file_path: Optional[str],
        opt_file_path: Optional[str],
        include_optim: bool = True,
        timestamp: int = 0,
    ) -> Optional[int]:
        params = _validate_load_meta(
            self._hbm,
            table_id,
            meta_json_file_path,
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path,
            include_optim,
        )
        self._hbm.collect_table_sizes_flag = True
        collect_table_sizes_for_state(self._hbm, non_blocking=False)

        self._host.collect_table_sizes_flag = True
        collect_table_sizes_for_state(self._host, non_blocking=False)

        unique_size_per_table = torch.zeros(
            self._host.num_tables, dtype=torch.int64, device=torch.device("cpu")
        )
        unique_size_per_table[table_id] = max(
            0,
            params.num_keys
            + self._hbm.estimated_table_sizes[table_id].item()
            - self._hbm.tables[table_id].tensor().size(0),
        )

        self.expand_if_need(unique_size_per_table)

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        for keys, embeddings, file_scores, opt_states in _iter_batches_from_files(
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path if params.include_optim else None,
            params.dim,
            params.file_optstate_dim,
            device,
        ):
            if keys.numel() == 0:
                continue

            if opt_states is None and params.runtime_optstate_dim > 0:
                opt_states = (
                    torch.ones(
                        keys.numel(),
                        params.runtime_optstate_dim,
                        dtype=self._hbm.emb_dtype,
                        device=device,
                    )
                    * self._hbm.initial_optim_state
                )
            elif opt_states is not None and params.runtime_optstate_dim > 0:
                opt_states = pad_optimizer_states_from_checkpoint(
                    self._hbm.optimizer,
                    self._hbm.table_emb_dims_cpu[table_id],
                    opt_states,
                    self._hbm.initial_optim_state,
                    self._hbm.emb_dtype,
                    device,
                )

            vtype = self._hbm.emb_dtype
            values = (
                torch.cat(
                    [embeddings.to(vtype), opt_states.to(vtype)],
                    dim=-1,
                )
                if opt_states is not None
                else embeddings.to(vtype)
            )

            tids = torch.full(
                (keys.numel(),), table_id, dtype=torch.int64, device=device
            )

            (
                ins_indices,
                num_evicted,
                evicted_keys,
                evicted_table_ids,
                evicted_indices,
                evicted_scores,
            ) = _insert_and_evict_keys(self._hbm, keys, tids, file_scores)

            evicted_values = load_from_flat(
                self._hbm,
                evicted_indices,
                evicted_table_ids,
                copy_mode=CopyMode.VALUE,
            )
            select_insert_failed_values(evicted_indices, values, evicted_values)
            store_to_flat_single_table(self._hbm, ins_indices, table_id, values)

            if num_evicted != 0:
                _insert_key_values(
                    self._host,
                    evicted_keys,
                    evicted_table_ids,
                    evicted_values,
                    evicted_scores,
                )

        return params.meta_data.get("step_score", None)

    # -- Export: yield from both tiers --

    def export_keys_values(
        self,
        device: torch.device,
        batch_size: int = 65536,
        table_id: int = 0,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
    ]:
        """Yield (keys, embeddings, opt_states, scores) batches from HBM then host.

        After streaming the HBM tier, any key that still exists in the host map for
        the same ``table_id`` is **erased from the host tier** so HBM remains
        authoritative and the two tiers do not duplicate rows. Missing keys on host
        are left unchanged. This **mutates** host storage (same as a logical cleanup
        before host export).
        """
        hbm_key_parts: List[torch.Tensor] = []
        for batch in export_keys_values_iter(
            self._hbm, device, batch_size, table_id
        ):
            keys, embeddings, opt_states, scores = batch
            hbm_key_parts.append(keys.detach())
            yield (keys, embeddings, opt_states, scores)

        if hbm_key_parts:
            hbm_unique = torch.unique(torch.cat(hbm_key_parts, dim=0))
        else:
            hbm_unique = torch.empty(0, dtype=torch.int64, device=device)

        if hbm_unique.numel() > 0:
            host_dev = self._host.device
            keys_erase = hbm_unique.to(host_dev)
            erase_chunk = min(batch_size, 65536)
            for s in range(0, keys_erase.numel(), erase_chunk):
                chunk = keys_erase[s : s + erase_chunk]
                tids = torch.full(
                    (chunk.numel(),),
                    table_id,
                    dtype=torch.int64,
                    device=host_dev,
                )
                self._host.key_index_map.erase(chunk, tids)

        yield from export_keys_values_iter(
            self._host, device, batch_size, table_id
        )


# ---------------------------------------------------------------------------
# Higher-level free functions
# ---------------------------------------------------------------------------


def _expand_storage_for_cache_flush_if_needed(
    storage: DynamicEmbStorage, cache: DynamicEmbCache
) -> None:
    """Grow backing storage before ``flush_cache`` so inserts need not evict cold keys.

    **Existing expansion** (:func:`expand_if_need_impl` / :func:`get_expand_info`):
    runs in prefetch from :func:`dynamicemb.batched_dynamicemb_function.dynamicemb_prefetch`.
    It expands when ``(estimated_backing_keys + batch_uniques) / physical_cap > max_load_factor``.
    With ``caching=True``, most keys stay in the GPU cache, so backing occupancy stays low
    and expansion often stops while physical cap is still far below ``max_capacity``.

    **Flush** then copies every cache entry into backing via :meth:`Storage.insert` without
    calling ``expand_if_need``, so backing hits its current cap and evicts (e.g. TIMESTAMP).

    This helper uses a conservative upper bound on distinct keys after merge,
    ``min(max_capacity, storage_size(table) + cache_size(table))`` per logical table
    (disjoint worst case), and reuses :func:`_expand_tables_impl`. Target key-map capacity
    is the max of ``align_to_table_size(2 * merge_ub, bucket_capacity)``, ``2 * key_cap``,
    and for NO_EVICTION also ``2 * value_row_count``. The ``2 * merge_ub`` alignment
    already covers ``merge_ub`` and ``align_to_table_size(merge_ub, ...)`` in the max.
    Capped by ``max_capacity``. Afterward it refreshes ``estimated_table_sizes``.
    """
    st = storage._state
    ca = cache._state
    assert st.num_tables == ca.num_tables

    is_no_eviction = (
        st.options_list[0].score_strategy == DynamicEmbScoreStrategy.NO_EVICTION
    )
    tables_to_expand: List[bool] = [False] * st.num_tables
    target_capacities: List[int] = [-1] * st.num_tables

    for table_id in range(st.num_tables):
        opt = st.options_list[table_id]
        s_sz = st.key_index_map.size(table_id)
        c_sz = ca.key_index_map.size(table_id)
        merge_ub = s_sz + c_sz
        max_cap_opt = opt.max_capacity
        if max_cap_opt is not None:
            merge_ub = min(merge_ub, int(max_cap_opt))

        key_cap = int(st.key_index_map.per_table_capacity_[table_id])
        bc = opt.bucket_capacity
        merge_ub_2x = align_to_table_size(2 * merge_ub, bc)
        # ``merge_ub_2x >= merge_ub`` (aligned 2× bound), so this also implies room for ``merge_ub``.
        if key_cap >= merge_ub_2x:
            continue

        if is_no_eviction:
            cur_v = int(st.tables[table_id].tensor().size(0))
            target = max(merge_ub_2x, key_cap * 2, cur_v * 2)
        else:
            if max_cap_opt is not None and key_cap >= int(max_cap_opt):
                continue
            max_lf = opt.max_load_factor
            if max_lf <= 0:
                continue
            target = max(merge_ub_2x, key_cap * 2)

        if max_cap_opt is not None:
            target = min(int(max_cap_opt), target)

        if is_no_eviction and target <= key_cap and target <= cur_v:
            continue

        tables_to_expand[table_id] = True
        target_capacities[table_id] = int(target)

    if not any(tables_to_expand):
        return

    _expand_tables_impl(st, tables_to_expand, target_capacities)
    st.collect_table_sizes_flag = True
    collect_table_sizes_for_state(st, non_blocking=False)


def flush_cache(cache: DynamicEmbCache, storage: Storage) -> None:
    if isinstance(storage, DynamicEmbStorage):
        _expand_storage_for_cache_flush_if_needed(storage, cache)

    state = cache._state
    batch_size = state.threads_in_wave
    state.value_dim

    for t in range(state.num_tables):
        for (
            keys,
            named_scores,
            indices,
        ) in state.key_index_map._batched_export_keys_scores(
            [state.score_policy.name],
            state.device,
            batch_size=batch_size,
            return_index=True,
            table_id=t,
        ):
            scores = named_scores[state.score_policy.name]
            tid = torch.full((keys.numel(),), t, dtype=torch.int64, device=keys.device)
            values = load_from_flat(state, indices, tid, copy_mode=CopyMode.VALUE)
            if isinstance(storage, DynamicEmbStorage) and (
                storage._state.no_eviction_next_index is not None
            ):
                st_b = storage._state
                (
                    _hm_fc,
                    _mk_fc,
                    _mi_fc,
                    _mt_fc,
                    _ms_fc,
                    in_backing,
                    _so_fc,
                    _ix_fc,
                ) = _find_keys(st_b, keys, tid)
                if bool(in_backing.all()):
                    storage.insert(keys, tid, values, preserve_existing=True)
                elif bool((~in_backing).all()):
                    storage.insert(keys, tid, values, scores, preserve_existing=False)
                else:
                    ex_b = in_backing
                    nw_b = ~in_backing
                    if ex_b.any():
                        storage.insert(
                            keys[ex_b],
                            tid[ex_b],
                            values[ex_b],
                            preserve_existing=True,
                        )
                    if nw_b.any():
                        storage.insert(
                            keys[nw_b],
                            tid[nw_b],
                            values[nw_b],
                            scores[nw_b],
                            preserve_existing=False,
                        )
            else:
                storage.insert(keys, tid, values, scores)


# ---------------------------------------------------------------------------
# eval_lookup – unified eval path for storage-only and cache+storage
# ---------------------------------------------------------------------------


def _eval_lookup_storage(
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
) -> torch.Tensor:
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
        initializer(embs, missing_indices, keys)

    return embs


def _eval_lookup_cached(
    cache: Cache,
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
) -> torch.Tensor:
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
        initializer(storage_embs, missing_indices_in_storage, missing_keys)

    embs[miss_compact_idx, :] = storage_embs

    return embs


def eval_lookup(
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
    cache: Optional[Cache] = None,
) -> torch.Tensor:
    """Eval-only lookup (no insertion, no admission, no backward).

    When *cache* is ``None``, looks up directly from *storage*.
    When *cache* is provided, looks up from cache first, then falls back to
    *storage* for cache misses.  Only embedding columns are copied
    (``CopyMode.EMBEDDING``); optimizer states are never touched.

    Returns the embedding tensor of shape ``[len(keys), emb_dim]``.
    """
    assert keys.dim() == 1
    if keys.numel() == 0:
        return torch.empty(
            0,
            storage.max_embedding_dim(),
            dtype=storage.embedding_dtype(),
            device=keys.device,
        )

    if cache is None:
        return _eval_lookup_storage(storage, keys, table_ids, initializer)

    return _eval_lookup_cached(
        cache,
        storage,
        keys,
        table_ids,
        initializer,
    )
