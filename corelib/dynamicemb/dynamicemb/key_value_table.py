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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.dynamicemb_config import (
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EvictedItemMode,
    ReplayContent,
    ScoreStrategy,
    align_to_table_size,
    get_physical_score_order,
    score_dump_permutation,
    score_load_permutation,
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
    ReplayStats,
    Storage,
    torch_dtype_to_np_dtype,
)
from dynamicemb_extensions import EvictStrategy, device_timestamp, flagged_compact
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


def _all_gather_dumped_columns(
    columns: List[Tensor],
    pg: dist.ProcessGroup,
) -> List[Tensor]:
    """Gather a set of row-aligned dump columns from all ranks into CPU tensors.

    Every tensor in *columns* must live on the same device and share ``size(0)``
    (the per-rank dumped-key count); trailing dimensions may differ (e.g. keys
    ``(N,)``, values ``(N, D)``, scores ``(N, W)``). Returns the same columns with
    all ranks' rows concatenated in rank order, on host and still row-aligned.

    NCCL has no variable-length gather, so this is one size all_gather followed by
    a padded all_gather per column.
    """
    assert columns, "_all_gather_dumped_columns needs at least one column"
    device = columns[0].device
    world_size = dist.get_world_size(group=pg)
    n = columns[0].size(0)
    d_count = torch.tensor([n], dtype=torch.long, device=device)
    gathered_counts = [torch.empty_like(d_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, d_count, group=pg)
    counts = [int(c.item()) for c in gathered_counts]
    max_n = max(counts)

    out: List[Tensor] = []
    for col in columns:
        padded = torch.zeros(
            (max_n,) + tuple(col.shape[1:]), dtype=col.dtype, device=device
        )
        if n > 0:
            padded[:n] = col
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded, group=pg)
        out.append(
            torch.cat(
                [gathered[i][: counts[i]] for i in range(world_size)], dim=0
            ).cpu()
        )
    return out


def _timestamp_score_columns(score_strategy: ScoreStrategy) -> List[int]:
    """Logical score-column indices holding a device timestamp.

    ``%globaltimer`` is per-device and resets across boots, so a raw timestamp is
    meaningless anywhere but the rank that produced it. These columns are dumped
    as an *age* (``cur_ts - score``) -- the same convention the file checkpoint
    uses for single-column LRU tables. Every other column (LFU frequency, STEP,
    CUSTOMIZED, NO_EVICTION row) is carried verbatim.
    """
    if isinstance(score_strategy, tuple):
        return [
            i
            for i, s in enumerate(score_strategy)
            if s == DynamicEmbScoreStrategy.TIMESTAMP
        ]
    return [0] if score_strategy == DynamicEmbScoreStrategy.TIMESTAMP else []


def _scores_to_age(
    score_strategy: ScoreStrategy, scores: Tensor, cur_ts: int
) -> Tensor:
    """Convert timestamp columns to ages, for dumping.

    *scores* is a ``[N, W]`` block in logical (configured) column order. When
    there is nothing to convert (no timestamp column, or an empty block) the
    input is returned unchanged rather than copied -- callers must not mutate the
    result in place. Otherwise a clone is returned.
    """
    ts_cols = _timestamp_score_columns(score_strategy)
    if not ts_cols or scores.numel() == 0:
        return scores
    out = scores.clone()
    for c in ts_cols:
        out[:, c] = torch.clamp(cur_ts - scores[:, c], min=0)
    return out


def _age_to_scores(
    score_strategy: ScoreStrategy, ages: Tensor, target_ts: int
) -> Tensor:
    """Inverse of :func:`_scores_to_age`, for replay.

    Same aliasing note: returns *ages* itself when there is nothing to convert.
    """
    ts_cols = _timestamp_score_columns(score_strategy)
    if not ts_cols or ages.numel() == 0:
        return ages
    out = ages.clone()
    for c in ts_cols:
        out[:, c] = torch.clamp(target_ts - ages[:, c], min=0)
    return out


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
    """Build the physical :class:`ScoreSpec` for a (validated) score strategy.

    *score_strategy* is either a single :class:`DynamicEmbScoreStrategy` or a
    supported compound tuple (currently only ``{TIMESTAMP, LFU}`` in either order).
    A compound strategy maps to the ``LruLfu`` policy whose physical layout is
    fixed at ``(timestamp, frequency)`` regardless of the user's tuple order; the
    user's ordering only affects checkpoint column order (see
    :func:`score_dump_permutation`).
    """
    if isinstance(score_strategy, tuple):
        # Only the {TIMESTAMP, LFU} compound is supported (validated at config
        # construction). Two adjacent AoS score words per key -- word 0 =
        # last-access timestamp (selects keys for a time-based incremental dump),
        # word 1 = frequency (drives LFU eviction). num_scores is derived from the
        # policy. The spec name maps to word 0 (score_index 0), i.e. the timestamp
        # column incremental_dump thresholds on -- hence "lru_lfu".
        if frozenset(score_strategy) == frozenset(
            {DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU}
        ):
            return ScoreSpec(
                name="lru_lfu",
                policy=ScorePolicy.LRU_LFU,
                dtype=torch.uint64,
                is_reduction=True,
            )
        raise NotImplementedError(
            f"Unsupported compound score_strategy {score_strategy}."
        )
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


def get_table_ptrs(
    tables: List[ExtendableBuffer], device: torch.device
) -> torch.Tensor:
    """Build a device tensor of current value-buffer base pointers for *tables*.

    Uses ``buffer.tensor().data_ptr()`` so results stay valid until the next
    ``ExtendableBuffer.extend()``; after extend, rebuild into ``table_ptrs_dev``
    (or another buffer) via this function.
    """
    return torch.tensor(
        [b.tensor().data_ptr() for b in tables],
        dtype=torch.int64,
        device=device,
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
    # Per-table value buffer base pointers on ``device``; refreshed on init and expand.
    table_ptrs_dev: torch.Tensor
    table_emb_dims: torch.Tensor
    # Persistent host (CPU) copy of table_emb_dims as a tensor (the ``_cpu`` field
    # below is the Python int list); long-lived so callers get a stable host tensor.
    table_emb_dims_host: torch.Tensor
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
    # Name of the score column incremental_dump thresholds on. Equals
    # score_policy.name (for LruLfu this is the leading, timestamp, word).
    incremental_score_name: Optional[str] = None
    # Retained-key buffers (last tier only). A key leaves a table two ways, and
    # the two are kept apart because consumers want different things from them:
    # an eviction is reproduced by whoever overwrites its slot, while an erase
    # leaves the slot to nobody and has to be replayed as a removal.
    #
    #   evicted_*_chunks -- victims of an insert that had to make room
    #   erased_*_chunks  -- keys an explicit erase removed
    #
    # Both accumulate as (key, table_id) GPU-tensor chunks, one chunk per
    # operation that removed anything, and are only concatenated + de-duplicated
    # on pop (no compaction while appending). Which buffers are fed is
    # ``evicted_item_mode``'s call.
    evicted_item_mode: EvictedItemMode = EvictedItemMode.DISCARD
    evicted_key_chunks: List[torch.Tensor] = field(default_factory=list)
    evicted_tid_chunks: List[torch.Tensor] = field(default_factory=list)
    erased_key_chunks: List[torch.Tensor] = field(default_factory=list)
    erased_tid_chunks: List[torch.Tensor] = field(default_factory=list)


def create_table_state(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    enable_overflow: bool = False,
    evicted_item_mode: EvictedItemMode = EvictedItemMode.DISCARD,
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

    # incremental_dump thresholds on the reduction score column (word 0 -- the
    # timestamp -- for the compound LruLfu policy).
    incremental_score_name = score_policy.name

    # LruLfu custom eviction: numba-compile + nvJitLink-link the user
    # score_function and route its inserts to that cubin via score_fn_key. Key 0
    # (no score_function) selects the default freq->ts Lex evictor (no numba).
    # The score_function is indexed in logical (tuple) order, so the compiler is
    # given score_strategy to remap logical->physical.
    score_fn_key = 0
    if (
        score_policy.policy == ScorePolicy.LRU_LFU
        and base_opt.score_function is not None
    ):
        from dynamicemb.jit.score_jit import register_score_function

        cc_major, cc_minor = torch.cuda.get_device_capability(device_idx)
        score_fn_key = register_score_function(
            base_opt.score_function, base_opt.score_strategy, cc_major, cc_minor
        )

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
        score_fn_key=score_fn_key,
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
    table_emb_dims_host = torch.tensor(dims, dtype=torch.int64)
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
        incremental_score_name=incremental_score_name,
        evict_strategy=evict_strategy,
        key_index_map=key_index_map,
        capacity=capacity,
        tables=tables,
        table_ptrs_dev=get_table_ptrs(tables, device),
        table_emb_dims=table_emb_dims,
        table_emb_dims_host=table_emb_dims_host,
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
        evicted_item_mode=evicted_item_mode,
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
        # Preserve the eviction routing (default Lex or custom score_function)
        # across rehash.
        score_fn_key=state.key_index_map.score_fn_key_,
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
    state.table_ptrs_dev.copy_(get_table_ptrs(state.tables, device))

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
            # Single-value re-insert above only restores the leading score word.
            # For multi-word layouts (LruLfu: timestamp + frequency) copy the
            # whole score block so the frequency (and exact timestamp) survive
            # rehash; otherwise LFU eviction would see all frequencies reset to 0.
            if new_key_index_map.num_scores_ > 1:
                new_key_index_map.copy_score_blocks_from(
                    old_key_index_map, table_id, src_indices, dst_indices
                )
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
        assert bool(
            torch.all(kim_slot_indices[missing] == -1)
        ), "KIM lookup: missing keys must have slot index -1"

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


def _encode_slot_index(
    state: DynamicEmbTableState,
    slot_indices: torch.Tensor,
    flat_rows: torch.Tensor,
    tier: Optional[int] = None,
) -> torch.Tensor:
    """int64 ``slot_index`` for a dumped table (for precise replay_increment).

    Single-tier storage (``tier is None``):
      - normal policies: the key_index_map slot IS the value flat row, so the
        single slot value locates both key and value -> return the slot as-is.
      - NO_EVICTION: key slot and value row are independent (row == score,
        auto-increment), so pack both into one int64: high 32 bits = key slot,
        low 32 bits = value row (tables are small, < 2^32; asserted).

    HybridStorage (``tier in {0, 1}``, called once per tier, HBM=0 / host=1):
      layout ``bit 63 = tier | bits 0..62 = key_slot``. HybridStorage does NOT
      support NO_EVICTION (rejected at construction), so every tier is a normal
      policy with key_slot == value_row -- one value locates both. Only the tier
      bit is added so replay can tell the two tiers' independent slot spaces apart.
    """
    slot_indices = slot_indices.to(device=state.device, dtype=torch.int64)
    if tier is not None:  # HybridStorage: bit 63 = tier, bits 0..62 = key_slot
        if slot_indices.numel() > 0:
            assert int(slot_indices.max()) < (
                1 << 63
            ), "HybridStorage key_slot exceeds 63 bits"
        if int(tier):
            return slot_indices | (torch.ones_like(slot_indices) << 63)
        return slot_indices
    if state.no_eviction_next_index is None:
        return slot_indices  # key slot == value row
    flat_rows = flat_rows.to(device=state.device, dtype=torch.int64)
    if slot_indices.numel() > 0:
        assert int(slot_indices.max()) < (1 << 32) and int(flat_rows.max()) < (
            1 << 32
        ), "NO_EVICTION slot/row exceeds 32 bits; cannot pack into int64 slot_index"
    return (slot_indices << 32) | flat_rows


def _fresh_score_block(
    state: DynamicEmbTableState,
    table_id: int,
    value_rows: torch.Tensor,
    timestamp: int,
    insert_score: int,
) -> torch.Tensor:
    """``[N, num_scores]`` physical score block for keys written by replay.

    A delta carries embeddings, not scores, so a replayed key is scored as if it
    had just been inserted here: recency words get *timestamp*, every other word
    gets *insert_score*. The replica therefore ranks its restored keys by when it
    received them rather than by how the source ranked them -- acceptable because
    the score only orders future evictions, and a replica that evicts on its own
    schedule still holds correct embeddings.

    *insert_score* is passed in rather than read off ``state.score``: that field
    is populated by the forward pass, and replay is the one write path that can
    run before a target has ever done a forward. The caller owns the table's
    score bookkeeping and always has a value.

    The exception is NO_EVICTION, whose score word is not a score at all but the
    value row. Reproducing the source's rows is the entire point of the replay,
    so that word is written verbatim from *value_rows*.
    """
    device = state.device
    if state.no_eviction_next_index is not None:
        return value_rows.to(device=device, dtype=SCORE_TYPE).view(-1, 1)
    n = value_rows.numel()
    num_scores = state.key_index_map.num_scores_
    block = torch.empty((n, num_scores), device=device, dtype=SCORE_TYPE)
    physical = get_physical_score_order(state.options_list[table_id].score_strategy)
    for word, strategy in enumerate(physical):
        block[:, word] = (
            timestamp if strategy == DynamicEmbScoreStrategy.TIMESTAMP else insert_score
        )
    return block


def _dump_score_block(
    state: DynamicEmbTableState,
    table_id: int,
    slot_indices: torch.Tensor,
    primary_scores: torch.Tensor,
    cur_ts: int,
) -> torch.Tensor:
    """``[N, num_scores]`` per-key score block for a dumped table.

    Columns are in the user's configured (logical) order -- the same order the
    file checkpoint uses -- and timestamp columns are converted to an age
    relative to *cur_ts* so they survive the trip to another device or host.
    Single-score tables reuse the score column the dump already thresholded on;
    multi-word layouts gather every word from the slots.
    """
    device = state.device
    num_scores = state.key_index_map.num_scores_
    score_strategy = state.options_list[table_id].score_strategy
    if num_scores > 1:
        block = state.key_index_map.gather_score_blocks(
            table_id, slot_indices.to(device=device, dtype=torch.int64)
        )
        perm = score_dump_permutation(score_strategy)
        if perm != list(range(block.size(1))):
            block = block[:, perm].contiguous()
    else:
        block = primary_scores.to(device=device, dtype=SCORE_TYPE).view(-1, 1)
    return _scores_to_age(score_strategy, block, cur_ts)


def _replay_score_block(
    state: DynamicEmbTableState,
    table_id: int,
    scores: torch.Tensor,
    target_ts: int,
) -> torch.Tensor:
    """Inverse of :func:`_dump_score_block`: logical ages -> physical scores.

    Takes the ``[N, num_scores]`` logical block carried in the delta, restores
    timestamp columns against this device's *target_ts*, and permutes back into
    the physical device layout.

    Both steps index by *column*, so a block that is not exactly as wide as this
    table's score layout would silently transform or permute the wrong words --
    e.g. writing a frequency into the timestamp column. The width is validated
    here rather than trusted.
    """
    device = state.device
    score_strategy = state.options_list[table_id].score_strategy
    num_scores = state.key_index_map.num_scores_
    block = scores.to(device=device, dtype=SCORE_TYPE)
    if block.dim() == 1:
        block = block.view(-1, 1)
    if block.dim() != 2 or block.size(1) != num_scores:
        raise ValueError(
            f"replay_increment: table {table_id} has {num_scores} score word(s) "
            f"per key, but the delta carries a score block of shape "
            f"{tuple(scores.shape)}."
        )
    block = _age_to_scores(score_strategy, block, target_ts)
    perm = score_load_permutation(score_strategy)
    if perm != list(range(num_scores)):
        block = block[:, perm]
    return block.contiguous()


def _split_value_row(
    state: DynamicEmbTableState, table_id: int, values: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split a loaded ``[N, value_dim]`` row into (embedding, optimizer state).

    ``load_from_flat_single_table`` returns the table's own compact row, so the
    optimizer state is the trailing ``value_dim - emb_dim`` columns. That runtime
    width is not all payload: rowwise Adagrad reserves a fixed 16 bytes per row
    in the fused FBGEMM layout but only ever fills one accumulator scalar. The
    dumped block is therefore narrowed to the width the file checkpoint uses --
    the same :func:`truncate_optimizer_states_for_checkpoint` ``_dump_table``
    applies -- so a delta and a checkpoint describe a row identically, and the
    padding is not paid for on every dump. Replay expands it back with
    :func:`pad_optimizer_states_from_checkpoint`.

    Tables whose optimizer keeps no per-row state (e.g. plain SGD) get ``None``
    rather than a zero-width tensor, matching :func:`export_keys_values_iter`.
    """
    emb_dim = state.table_emb_dims_cpu[table_id]
    optim_state_dim = state.table_value_dims_cpu[table_id] - emb_dim
    emb = values[:, :emb_dim].to(dtype=state.emb_dtype)
    if optim_state_dim <= 0:
        return emb, None
    opt = values[:, -optim_state_dim:]
    return (
        emb,
        truncate_optimizer_states_for_checkpoint(
            state.optimizer, emb_dim, opt
        ).contiguous(),
    )


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
            state.table_ptrs_dev,
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
        state.table_ptrs_dev,
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
            state.table_ptrs_dev,
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
        state.table_ptrs_dev,
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


def _append_evicted(
    state: DynamicEmbTableState,
    evicted_keys: torch.Tensor,
    evicted_table_ids: torch.Tensor,
    num_evicted: torch.Tensor,
) -> None:
    """Append this insert's evicted (key, table_id) to the state's retain chunks.

    ``evicted_keys`` / ``evicted_table_ids`` are sized at the batch upper bound;
    only the first ``num_evicted`` entries are valid, so we slice + clone (freeing
    the large per-batch buffers and keeping only the compact evicted rows). No
    de-duplication here -- that happens on pop. One D2H sync per evicting insert
    to read the count, acceptable since retain is opt-in.
    """
    n = int(num_evicted.item())
    if n <= 0:
        return
    state.evicted_key_chunks.append(evicted_keys[:n].clone())
    state.evicted_tid_chunks.append(evicted_table_ids[:n].clone())


def _append_erased(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    table_id: int,
) -> None:
    """Append an explicit erase's keys to the state's erased-key buffer.

    Unlike :func:`_append_evicted` there is no count to read back: the caller has
    already masked the keys down to the ones the erase actually removed, so this
    costs no extra device sync.
    """
    if keys.numel() == 0:
        return
    state.erased_key_chunks.append(keys.clone())
    state.erased_tid_chunks.append(
        torch.full((keys.numel(),), table_id, dtype=torch.int64, device=keys.device)
    )


def _pop_chunked_keys(
    key_chunks: List[torch.Tensor],
    tid_chunks: List[torch.Tensor],
    table_id: int,
    state: DynamicEmbTableState,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Drain one table's rows out of a (key, table_id) chunk buffer.

    Concatenates the chunks, selects this table's rows, de-duplicates, and
    returns the rebuilt chunk lists holding only the OTHER tables' rows -- so
    this table's keys are cleared while the rest stay for their own pop.
    """
    if not key_chunks:
        # Match the non-empty path's dtype: chunks carry key_index_map.key_type
        # (== the table's index_type, which may be int32/uint32), so a hardcoded
        # int64 here would make callers that concat/compare across empty and
        # non-empty pops hit a dtype mismatch.
        empty = torch.empty(0, dtype=state.key_index_map.key_type, device=state.device)
        return empty, [], []
    keys = torch.cat(key_chunks)
    tids = torch.cat(tid_chunks)
    mask = tids == table_id
    out = torch.unique(keys[mask])
    keep = ~mask
    if bool(keep.any()):
        return out, [keys[keep]], [tids[keep]]
    return out, [], []


def _pop_state_evicted_keys(state: DynamicEmbTableState, table_id: int) -> torch.Tensor:
    """Pop (return + clear) the unique keys ``table_id`` had evicted."""
    out, state.evicted_key_chunks, state.evicted_tid_chunks = _pop_chunked_keys(
        state.evicted_key_chunks, state.evicted_tid_chunks, table_id, state
    )
    return out


def _pop_state_erased_keys(state: DynamicEmbTableState, table_id: int) -> torch.Tensor:
    """Pop (return + clear) the unique keys an explicit erase removed."""
    out, state.erased_key_chunks, state.erased_tid_chunks = _pop_chunked_keys(
        state.erased_key_chunks, state.erased_tid_chunks, table_id, state
    )
    return out


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
    if EvictedItemMode.RETAIN_KEY in state.evicted_item_mode:
        (
            indices,
            num_evicted,
            evicted_keys,
            evicted_table_ids,
        ) = state.key_index_map.insert(
            unique_keys,
            table_ids,
            score_arg,
            score_out=score_out_flat,
            collect_evicted=True,
        )
        _append_evicted(state, evicted_keys, evicted_table_ids, num_evicted)
    else:
        indices = state.key_index_map.insert(
            unique_keys, table_ids, score_arg, score_out=score_out_flat
        )
    if state.no_eviction_next_index is not None:
        flat_indices = (
            score_arg.value if score_arg.value is not None else score_out_flat
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
        # Multi-word score layouts (e.g. LruLfu: timestamp + frequency) must
        # persist ALL words, not just the exported column, so the checkpoint can
        # restore them exactly. Gather the full [N, num_scores] block at the
        # exported slots; single-score tables keep the [N] shape unchanged.
        if state.key_index_map.num_scores_ > 1:
            out_scores = state.key_index_map.gather_score_blocks(table_id, indices)
            # Reorder the physical device layout into the user's configured
            # (logical) score-column order so both the checkpoint and callers of
            # export_keys_values see scores in the order they configured. Identity
            # when the logical order matches the physical layout.
            dump_perm = score_dump_permutation(
                state.options_list[table_id].score_strategy
            )
            if dump_perm != list(range(out_scores.size(1))):
                out_scores = out_scores[:, dump_perm].contiguous()
        else:
            out_scores = scores.to(SCORE_TYPE)
        yield (
            keys.to(device),
            embeddings.to(device),
            opt_states,
            out_scores.to(device),
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
        meta_data["dist_type"] = state.options_list[table_id].dist_type

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
    num_scores: int = 1,
) -> Iterator[Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]]:
    """Yield (keys, embeddings, scores, opt_states) batches from checkpoint files.

    ``num_scores`` is the number of score words per key in the score file. When
    > 1 (e.g. LruLfu's timestamp+frequency) the yielded ``scores`` is
    [n, num_scores]; otherwise it is [n].

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
                score_bytes = fscore.read(SCORE_TYPE.itemsize * n * num_scores)
                scores = torch.tensor(
                    np.frombuffer(
                        score_bytes, dtype=torch_dtype_to_np_dtype[SCORE_TYPE]
                    ),
                    dtype=SCORE_TYPE,
                    device=device,
                )
                if num_scores > 1:
                    scores = scores.view(-1, num_scores)

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
    ckpt_dist_type = meta_data.get("dist_type", "roundrobin")
    runtime_dist_type = state.options_list[table_id].dist_type
    if runtime_dist_type != ckpt_dist_type:
        raise ValueError(
            "Input dist_type mismatch: checkpoint was dumped with "
            f"{ckpt_dist_type!r}, but runtime table is configured with "
            f"{runtime_dist_type!r}. Please load with a matching dist_type."
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
        # The score file holds num_scores words per key (e.g. LruLfu = 2:
        # timestamp + frequency), so compare against num_keys * num_scores.
        per_key = state.key_index_map.num_scores_
        num_score_words = os.path.getsize(score_file_path) // SCORE_TYPE.itemsize
        if num_score_words != num_keys * per_key:
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

    # Multi-word score layouts (LruLfu: timestamp + frequency) cannot be restored
    # by a single-value insert (which only writes word 0). Place the keys without
    # touching scores (CONST), then scatter the full [N, num_scores] block.
    num_scores = state.key_index_map.num_scores_
    if num_scores > 1:
        # Validate the score-block shape before scatter_score_blocks writes it
        # into device memory: a malformed checkpoint (e.g. a 1-D score tensor)
        # would otherwise produce out-of-bounds / wrong writes. Use ValueError
        # (not assert) so the check survives `python -O`.
        if (
            scores is None
            or scores.dim() != 2
            or scores.size(1) != num_scores
            or scores.size(0) != keys.numel()
        ):
            raise ValueError(
                f"multi-word load expects [{keys.numel()}, {num_scores}] scores, "
                f"got {None if scores is None else tuple(scores.shape)}"
            )
        tid_tensor = torch.full(
            (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
        )
        place_arg = ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )
        indices = state.key_index_map.insert(keys, tid_tensor, place_arg)
        state.key_index_map.scatter_score_blocks(
            table_id, indices, scores.to(SCORE_TYPE).contiguous()
        )
        store_to_flat_single_table(state, indices, table_id, values)
        return

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
# Replay – write an incremental_dump delta back into a table
# ---------------------------------------------------------------------------


def _slice(t: Optional[torch.Tensor], start: int, end: int) -> Optional[torch.Tensor]:
    """Slice an optional column, keeping ``None`` as ``None``."""
    return None if t is None else t[start:end]


def _replay_write_values(
    state: DynamicEmbTableState,
    table_id: int,
    rows: torch.Tensor,
    embeddings: torch.Tensor,
    optimizer_states: Optional[torch.Tensor],
    keep_optimizer: torch.Tensor,
    content: ReplayContent,
) -> None:
    """Write a replayed row's value columns, per :class:`ReplayContent`.

    A value row is an embedding followed by the optimizer state, and the two are
    written together or not at all -- ``store_to_flat_single_table`` copies from
    the row base, so there is no way to write the tail without the head. That
    shapes the three cases:

    - **Both requested.** The delta carries the whole row; write it as it was
      dumped. Nothing is inferred.
    - **Embedding only** (the default for a serving replica). Rows that already
      belonged to this same key keep their optimizer state, because writing just
      the embedding columns leaves the tail untouched. Every other row is new to
      this key, so its tail still holds the previous occupant's moments and has
      to be reset to ``initial_optim_state``.
    - **Optimizer state only.** Every row must already belong to this key -- the
      caller has checked -- so the current embedding is read back and rewritten
      unchanged ahead of the new tail.
    """
    if rows.numel() == 0:
        return
    emb_dim_cfg = state.table_emb_dims_cpu[table_id]
    optstate_dim = state.optimizer.get_state_dim(emb_dim_cfg)
    embeddings = embeddings.to(dtype=state.emb_dtype)
    want_emb = ReplayContent.EMBEDDING in content
    want_opt = ReplayContent.OPTIMIZER_STATE in content and optstate_dim > 0

    if not want_emb:
        # Keep what is there: the caller guarantees every row already holds this
        # key, so its embedding is the one the replica should go on serving.
        embeddings = load_from_flat_single_table(state, rows, table_id)[:, :emb_dim_cfg]

    if optstate_dim == 0:
        store_to_flat_single_table(state, rows, table_id, embeddings)
        return

    if want_opt:
        if optimizer_states is None:
            raise ValueError(
                f"replay_increment: table {table_id} keeps optimizer state per "
                "row, but the delta carries none. Drop "
                "ReplayContent.OPTIMIZER_STATE, or replay a delta dumped from a "
                "table with the same optimizer."
            )
        ckpt_dim = state.optimizer.get_ckpt_state_dim(emb_dim_cfg)
        if optimizer_states.dim() != 2 or optimizer_states.size(1) != ckpt_dim:
            raise ValueError(
                f"replay_increment: table {table_id} dumps {ckpt_dim} "
                f"optimizer-state column(s) per row, but the delta carries a "
                f"block of shape {tuple(optimizer_states.shape)}."
            )
        # Back to the runtime width the fused value row expects.
        opt = pad_optimizer_states_from_checkpoint(
            state.optimizer,
            emb_dim_cfg,
            optimizer_states.to(device=embeddings.device),
            state.initial_optim_state,
            state.emb_dtype,
            embeddings.device,
        )
        store_to_flat_single_table(
            state, rows, table_id, torch.cat([embeddings, opt], dim=-1)
        )
        return

    keep = keep_optimizer
    if bool(keep.any()):
        store_to_flat_single_table(state, rows[keep], table_id, embeddings[keep])
    fresh = torch.logical_not(keep)
    if bool(fresh.any()):
        fresh_emb = embeddings[fresh]
        opt_states = torch.full(
            (fresh_emb.size(0), optstate_dim),
            state.initial_optim_state,
            dtype=state.emb_dtype,
            device=fresh_emb.device,
        )
        store_to_flat_single_table(
            state,
            rows[fresh],
            table_id,
            torch.cat([fresh_emb, opt_states], dim=-1),
        )


def _replay_at_slots(
    state: DynamicEmbTableState,
    table_id: int,
    keys: torch.Tensor,
    embeddings: torch.Tensor,
    optimizer_states: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    slot_index: torch.Tensor,
    timestamp: int,
    insert_score: int,
    content: ReplayContent,
) -> int:
    """Write keys at the exact slots they held in the source table.

    Every key must land: the caller has already established that this table's
    layout matches the source's, so a slot that does not fall inside its key's
    home bucket means the two have diverged anyway and the replay would silently
    lose that key. Raises rather than writing a partial result.

    Returns the number of keys written (always all of them, or it raised).
    """
    n = keys.numel()
    device = state.device
    if n == 0:
        return 0
    tids = torch.full((n,), table_id, dtype=torch.int64, device=device)

    if state.no_eviction_next_index is not None:
        # NO_EVICTION packs key slot (high 32) and value row (low 32); every
        # other strategy uses one index for both.
        key_slots = (slot_index >> 32) & 0xFFFFFFFF
        value_rows = slot_index & 0xFFFFFFFF
    else:
        key_slots = slot_index
        value_rows = slot_index

    # The score words are written with the key, so they are chosen here: either
    # restored from the delta or minted as if the key had just been inserted.
    if ReplayContent.SCORE in content and scores is not None:
        score_block = _replay_score_block(state, table_id, scores, timestamp)
    else:
        score_block = _fresh_score_block(
            state, table_id, value_rows, timestamp, insert_score
        )
    status, same_key = state.key_index_map.scatter_keys_at_slots(
        keys, tids, key_slots, score_block
    )
    placed = status >= 0
    num_unplaced = int(n - placed.sum().item())
    if num_unplaced:
        example = int(keys[torch.logical_not(placed)][0].item())
        raise RuntimeError(
            f"replay_increment: {num_unplaced} of {n} keys could not be written "
            f"at their source slot on table {table_id} (e.g. key {example}). "
            "Either the slot falls outside its key's home bucket -- where no "
            "lookup would ever probe it, meaning the target's hash layout "
            "differs from the source's despite matching metadata -- or the slot "
            "was held by another writer, meaning something ran against this "
            "table concurrently with the replay. Replaying past either would "
            "silently drop those keys."
        )

    if ReplayContent.EMBEDDING not in content:
        # Without an embedding to write, a row that did not already hold this key
        # would keep the previous occupant's vector and serve it under the new
        # key. Refuse rather than corrupt.
        stale = torch.logical_not(same_key)
        num_stale = int(stale.sum().item())
        if num_stale:
            example = int(keys[stale][0].item())
            raise ValueError(
                f"replay_increment: {num_stale} of {n} keys on table {table_id} "
                f"do not already occupy their target row (e.g. key {example}), "
                "so there is no embedding to keep. ReplayContent.EMBEDDING may "
                "only be omitted for a replica already aligned with its source."
            )
    _replay_write_values(
        state, table_id, value_rows, embeddings, optimizer_states, same_key, content
    )
    if state.no_eviction_next_index is not None:
        # Keep the auto-increment counter ahead of every restored row so a later
        # native insert cannot hand out a row that is already in use.
        next_row = int(value_rows.max().item()) + 1
        cur = int(state.no_eviction_next_index[table_id].item())
        if next_row > cur:
            state.no_eviction_next_index[table_id] = next_row
            state.no_eviction_next_index_dev[table_id] = next_row
    return n


def _replay_state_increment(
    state: DynamicEmbTableState,
    table_id: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    optimizer_states: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    slot_index: torch.Tensor,
    timestamp: int,
    insert_score: int,
    content: ReplayContent,
) -> ReplayStats:
    """Replay one delta batch into a single table state (one storage tier)."""
    stats = ReplayStats()
    n = keys.numel()
    if n == 0:
        return stats
    device = state.device
    keys = keys.to(device=device, dtype=state.key_index_map.key_type)
    embeddings = values.to(device=device, dtype=state.emb_dtype)
    slots = slot_index.to(device=device, dtype=torch.int64)
    stats.upserted = _replay_at_slots(
        state,
        table_id,
        keys,
        embeddings,
        optimizer_states,
        scores,
        slots,
        timestamp,
        insert_score,
        content,
    )
    return stats


def _erase_state_keys(
    state: DynamicEmbTableState,
    table_id: int,
    keys: torch.Tensor,
    mode: EvictedItemMode = EvictedItemMode.DISCARD,
) -> int:
    """Erase keys from one table state. Returns how many were actually present.

    The erase kernel reports the slot it cleared, or ``-1`` for a key that was
    not there, so the count is the real number removed rather than the number
    asked for -- a delta can legitimately name keys this replica never held.

    *mode* is this call's, not the table's: retaining an erase costs only a copy
    of keys already in hand, so there is nothing to configure up front and each
    caller can decide whether its removals are worth reporting.
    """
    n = keys.numel()
    if n == 0:
        return 0
    device = state.device
    keys = keys.to(device=device, dtype=state.key_index_map.key_type)
    tids = torch.full((n,), table_id, dtype=torch.int64, device=device)
    indices = state.key_index_map.erase(keys, tids, return_indices=True)
    removed = indices >= 0
    if EvictedItemMode.RETAIN_KEY in mode:
        # Only the keys that were really there: a caller may name keys this
        # table never held, and reporting those as removed would make a replica
        # replay removals that never happened.
        _append_erased(state, keys[removed], table_id)
    return int(removed.sum().item())


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
            # DynamicEmbStorage is always a last tier (single tier or the backing
            # store under a cache), so it honors evicted_item_mode.
            evicted_item_mode=options[0].evicted_item_mode,
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

    def pop_evicted_keys(self, table_id: int) -> torch.Tensor:
        """Return + clear this rank's unique evicted keys retained for ``table_id``.

        DynamicEmbStorage is always a last tier, so retained keys live on its
        state. Returns an empty tensor when the mode does not retain evictions or
        nothing was evicted for the table since the last pop."""
        return _pop_state_evicted_keys(self._state, table_id)

    def pop_erased_keys(self, table_id: int) -> torch.Tensor:
        """Return + clear the keys an explicit erase removed from ``table_id``.

        Separate from :meth:`pop_evicted_keys` because the two mean different
        things downstream: an eviction is reproduced by whoever takes the slot,
        an erase leaves the slot to nobody and has to be replayed as a removal."""
        return _pop_state_erased_keys(self._state, table_id)

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
        num_scores = self._state.key_index_map.num_scores_
        # The checkpoint stores multi-word scores in the user's configured
        # (logical) order; reorder each batch back into the physical device layout
        # before scatter. Identity when logical order matches the physical layout.
        load_perm = (
            score_load_permutation(self._state.options_list[table_id].score_strategy)
            if num_scores > 1
            else None
        )
        for keys, embeddings, scores, opt_states in _iter_batches_from_files(
            emb_key_path,
            embedding_file_path,
            score_file_path,
            opt_file_path if params.include_optim else None,
            params.dim,
            params.file_optstate_dim,
            device,
            num_scores=num_scores,
        ):
            if (
                scores is not None
                and load_perm is not None
                and load_perm != list(range(scores.size(1)))
            ):
                scores = scores[:, load_perm].contiguous()
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
        timestamp: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
        """Dump one table's matched rows (score >= threshold).

        Multi-rank: all_gather so the result is concatenated from all ranks.

        Returns ``(keys, values, slot_index, optimizer_states, scores)``, all
        column-aligned. ``values`` is embeddings only and ``optimizer_states``
        the rest of the same row (``None`` when the optimizer keeps no per-row
        state), so concatenating the two reproduces the stored row.
        ``slot_index`` is the packed key-slot/value-row used by replay (see
        :func:`_encode_slot_index`). ``scores`` is the ``[N, num_scores]`` block
        in logical column order with timestamp columns held as an age relative
        to *timestamp*, which defaults to this device's clock."""
        timestamp = device_timestamp() if timestamp is None else timestamp
        state = self._state
        if state.options_list[table_id].dist_type == "continuous":
            raise NotImplementedError(
                "incremental_dump with slot_index does not support dist_type "
                "'continuous' (replay cannot reconstruct the owning rank from a key)."
            )
        states_to_dump = [state]
        do_multi_rank_gather = (
            pg is not None
            and dist.is_initialized()
            and dist.get_world_size(group=pg) > 1
        )
        all_keys: List[Tensor] = []
        all_values: List[Tensor] = []
        all_slots: List[Tensor] = []
        all_opts: List[Optional[Tensor]] = []
        all_scores: List[Tensor] = []
        for s in states_to_dump:
            keys, named_scores, indices = s.key_index_map.incremental_dump(
                {s.incremental_score_name: threshold},
                pg=pg,
                return_index=True,
                table_id=table_id,
            )
            scores_batch = named_scores[s.incremental_score_name]
            flat_rows = _flat_row_indices_from_slots_and_scores(
                s, indices, scores_batch
            )
            values = load_from_flat_single_table(s, flat_rows, table_id)
            value, opt = _split_value_row(s, table_id, values)
            key = keys.to(s.device) if keys.device.type != "cuda" else keys
            slot = _encode_slot_index(s, indices, flat_rows)
            score_block = _dump_score_block(
                s, table_id, indices, scores_batch, timestamp
            )
            if not do_multi_rank_gather:
                value = value.cpu()
                key = key.cpu() if key.is_cuda else key
                slot = slot.cpu()
                score_block = score_block.cpu()
                opt = opt.cpu() if opt is not None else None
            all_keys.append(key)
            all_values.append(value)
            all_slots.append(slot)
            all_opts.append(opt)
            all_scores.append(score_block)
        device_for_gather = state.device
        emb_dim_t = state.table_emb_dims_cpu[table_id]
        # Checkpoint width, matching what _split_value_row emits: an empty
        # batch still has to agree on column count with the non-empty ranks it
        # is all_gathered against.
        opt_dim_t = (
            state.optimizer.get_ckpt_state_dim(emb_dim_t)
            if state.table_value_dims_cpu[table_id] > emb_dim_t
            else 0
        )
        num_scores = state.key_index_map.num_scores_
        has_opt = bool(all_opts) and all_opts[0] is not None
        if all_keys:
            keys_cat = torch.cat(all_keys)
            values_cat = torch.cat(all_values, dim=0)
            slots_cat = torch.cat(all_slots)
            scores_cat = torch.cat(all_scores, dim=0)
            opts_cat = torch.cat(all_opts, dim=0) if has_opt else None
        else:
            dev = device_for_gather if do_multi_rank_gather else "cpu"
            keys_cat = torch.empty(0, dtype=torch.int64, device=dev)
            values_cat = torch.empty(0, emb_dim_t, dtype=state.emb_dtype, device=dev)
            slots_cat = torch.empty(0, dtype=torch.int64, device=dev)
            scores_cat = torch.empty(0, num_scores, dtype=SCORE_TYPE, device=dev)
            opts_cat = (
                torch.empty(0, opt_dim_t, dtype=state.emb_dtype, device=dev)
                if opt_dim_t > 0
                else None
            )
        if do_multi_rank_gather:
            # Every rank runs the same table config, so ``opts_cat is None``
            # agrees across ranks and the column list below is the same length
            # everywhere -- an asymmetric all_gather would hang.
            columns = [keys_cat, values_cat, slots_cat, scores_cat]
            if opts_cat is not None:
                columns.append(opts_cat)
            gathered = _all_gather_dumped_columns(
                [c.to(device_for_gather) for c in columns], pg
            )
            keys_cat, values_cat, slots_cat, scores_cat = gathered[:4]
            opts_cat = gathered[4] if opts_cat is not None else None
        elif keys_cat.device.type == "cuda":
            keys_cat = keys_cat.cpu()
            values_cat = values_cat.cpu()
            slots_cat = slots_cat.cpu()
            scores_cat = scores_cat.cpu()
            opts_cat = opts_cat.cpu() if opts_cat is not None else None
        return keys_cat, values_cat, slots_cat, opts_cat, scores_cat

    # -- Replay --

    def replay_increment(
        self,
        table_id: int,
        keys: Tensor,
        values: Tensor,
        optimizer_states: Optional[Tensor],
        scores: Optional[Tensor],
        slot_index: Tensor,
        insert_score: int,
        content: ReplayContent = ReplayContent.ALL,
        timestamp: Optional[int] = None,
    ) -> ReplayStats:
        """Write an ``incremental_dump`` delta back into one table.

        Every key is written at the exact slot and value row it held in the
        source; a key that cannot be placed there raises. The table is never
        grown: an expansion rehashes, which would move every key's home bucket
        and invalidate the source slots.

        Args:
            table_id: logical table within this storage.
            keys / values / optimizer_states / scores: the delta's columns for
                this table, row aligned. The last two may be ``None`` when
                *content* does not ask for them.
            slot_index: the source's packed key-slot/value-row.
            insert_score: the score a restored key gets in every non-recency
                score word when *content* omits ``SCORE``.
            content: which parts of each row to write; see
                :class:`ReplayContent`.
            timestamp: reference clock. Rebases the delta's timestamp ages when
                ``SCORE`` is requested, and stamps restored keys as
                freshly-inserted when it is not. Defaults to this device's clock.


        Writing is batched at ``threads_in_wave`` keys -- one full pass over the
        device, since the scatter kernel runs one thread per key -- which both
        keeps the machine busy and bounds the temporary buffers a replay needs
        (embeddings + optimizer/score blocks + masks) independently of how large
        the delta is. Not a knob: it is a property of the GPU, not of the call,
        and ``flush_cache`` sizes its own batches the same way.

        Returns:
            :class:`ReplayStats` with ``upserted`` set; ``erased`` / ``skipped``
            are the caller's to fill.

        Raises:
            RuntimeError: a key could not be placed at its source slot.
        """
        stats = ReplayStats()
        n = keys.numel()
        if n == 0:
            return stats
        timestamp = device_timestamp() if timestamp is None else timestamp
        batch_size = self._state.threads_in_wave
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            stats.merge(
                _replay_state_increment(
                    self._state,
                    table_id,
                    keys[start:end],
                    values[start:end],
                    _slice(optimizer_states, start, end),
                    _slice(scores, start, end),
                    slot_index[start:end],
                    timestamp,
                    insert_score,
                    content,
                )
            )
        return stats

    def erase_keys(
        self,
        table_id: int,
        keys: Tensor,
        mode: EvictedItemMode = EvictedItemMode.DISCARD,
    ) -> int:
        """Erase keys from the table (used to replay a delta's removals).

        Returns how many of them were actually present. *mode* decides whether
        the keys actually removed are recorded for :meth:`pop_erased_keys`.
        """
        return _erase_state_keys(self._state, table_id, keys, mode)

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

    def embedding_dims(self, on_device: bool = False) -> torch.Tensor:
        return (
            self._state.table_emb_dims if on_device else self._state.table_emb_dims_host
        )

    def all_dims_vec4(self) -> bool:
        return self._state.all_dims_vec4

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
        # Only the host tier is a last tier here: the HBM tier spills its
        # evictions into the host tier (insert_and_evict), so it never evicts for
        # real and has nothing to retain. Erases are not covered by this setting
        # at all -- ``erase_keys`` runs against both tiers and passes its own
        # mode down, so a key erased out of HBM is recorded just the same.
        self._hbm = create_table_state(hbm_options, optimizer)
        self._host = create_table_state(
            host_options,
            optimizer,
            evicted_item_mode=host_options[0].evicted_item_mode,
        )
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

    def embedding_dims(self, on_device: bool = False) -> torch.Tensor:
        # find() builds the value buffer from the HBM tier (load_from_flat(self._hbm)),
        # and max_*_dim above also report the HBM tier, so its dims are authoritative.
        return self._hbm.table_emb_dims if on_device else self._hbm.table_emb_dims_host

    def all_dims_vec4(self) -> bool:
        # HBM tier produces the value buffer in find(); its alignment governs
        # whether the vec4 padded-buffer kernels are safe for all rows.
        return self._hbm.all_dims_vec4

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

    def pop_evicted_keys(self, table_id: int) -> torch.Tensor:
        """Return + clear this rank's unique evicted keys retained for ``table_id``.

        Only the host tier is a last tier here (the HBM tier spills to host), so
        retained keys live on the host state. Empty tensor when retain is off or
        nothing was evicted for the table since the last pop."""
        return _pop_state_evicted_keys(self._host, table_id)

    def pop_erased_keys(self, table_id: int) -> torch.Tensor:
        """Return + clear the keys an explicit erase removed from ``table_id``.

        Drained from **both** tiers, unlike :meth:`pop_evicted_keys`: an erase
        removes a key from whichever tier holds it, and the tiers hold disjoint
        key sets, so concatenating them is a union rather than double counting.
        See :meth:`DynamicEmbStorage.pop_erased_keys` for why erases and
        evictions are kept in separate buffers."""
        drained = [_pop_state_erased_keys(s, table_id) for s in self.tables]
        parts = [p for p in drained if p.numel() > 0]
        if not parts:
            return torch.empty(
                0, dtype=self._host.key_index_map.key_type, device=self._host.device
            )
        return torch.unique(torch.cat(parts))

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
        timestamp: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
        """Dump one table's matched rows (score >= threshold), both tiers.

        Same columns as :meth:`DynamicEmbStorage.incremental_dump`; the two
        tiers' results are concatenated, and ``slot_index`` bit 63 tags which
        tier each key came from.

        Raises:
            NotImplementedError: the table is sharded with
                ``dist_type="continuous"`` (replay cannot reconstruct a key's
                owning rank), or the two tiers disagree on ``num_scores`` (their
                score blocks are concatenated, so they must share a width).
        """
        timestamp = device_timestamp() if timestamp is None else timestamp
        states_to_dump = self.tables
        if states_to_dump[0].options_list[table_id].dist_type == "continuous":
            raise NotImplementedError(
                "incremental_dump with slot_index does not support dist_type "
                "'continuous' (replay cannot reconstruct the owning rank from a key)."
            )
        num_scores = states_to_dump[0].key_index_map.num_scores_
        if any(s.key_index_map.num_scores_ != num_scores for s in states_to_dump):
            # The tiers' score blocks are concatenated into one column-aligned
            # result, so they must agree on width (mirrors the dump/load guard).
            raise NotImplementedError(
                "incremental_dump is not supported for HybridStorage whose tiers "
                "have different score-word counts."
            )
        do_multi_rank_gather = (
            pg is not None
            and dist.is_initialized()
            and dist.get_world_size(group=pg) > 1
        )
        all_keys = []
        all_values = []
        all_slots = []
        all_opts: List[Optional[Tensor]] = []
        all_scores = []
        # tier index into self.tables: 0 = HBM tier, 1 = host tier. slot_index
        # packs this tier bit (bit 63) so replay can tell the two tiers apart.
        for tier, s in enumerate(states_to_dump):
            keys, named_scores, indices = s.key_index_map.incremental_dump(
                {s.incremental_score_name: threshold},
                pg=pg,
                return_index=True,
                table_id=table_id,
            )
            scores_batch = named_scores[s.incremental_score_name]
            flat_rows = _flat_row_indices_from_slots_and_scores(
                s, indices, scores_batch
            )
            values = load_from_flat_single_table(s, flat_rows, table_id)
            value, opt = _split_value_row(s, table_id, values)
            key = keys.to(s.device) if keys.device.type != "cuda" else keys
            slot = _encode_slot_index(s, indices, flat_rows, tier=tier)
            score_block = _dump_score_block(
                s, table_id, indices, scores_batch, timestamp
            )
            if not do_multi_rank_gather:
                value = value.cpu()
                key = key.cpu() if key.is_cuda else key
                slot = slot.cpu()
                score_block = score_block.cpu()
                opt = opt.cpu() if opt is not None else None
            all_keys.append(key)
            all_values.append(value)
            all_slots.append(slot)
            all_opts.append(opt)
            all_scores.append(score_block)
        device_for_gather = states_to_dump[0].device
        emb_dim_t = states_to_dump[0].table_emb_dims_cpu[table_id]
        # Checkpoint width -- see the note in DynamicEmbStorage.incremental_dump.
        st0 = states_to_dump[0]
        opt_dim_t = (
            st0.optimizer.get_ckpt_state_dim(emb_dim_t)
            if st0.table_value_dims_cpu[table_id] > emb_dim_t
            else 0
        )
        has_opt = bool(all_opts) and all_opts[0] is not None
        if all_keys:
            keys_cat = torch.cat(all_keys)
            values_cat = torch.cat(all_values, dim=0)
            slots_cat = torch.cat(all_slots)
            scores_cat = torch.cat(all_scores, dim=0)
            opts_cat = torch.cat(all_opts, dim=0) if has_opt else None
        else:
            dev = device_for_gather if do_multi_rank_gather else "cpu"
            dt = self.embedding_dtype()
            keys_cat = torch.empty(0, dtype=torch.int64, device=dev)
            values_cat = torch.empty(0, emb_dim_t, dtype=dt, device=dev)
            slots_cat = torch.empty(0, dtype=torch.int64, device=dev)
            scores_cat = torch.empty(0, num_scores, dtype=SCORE_TYPE, device=dev)
            opts_cat = (
                torch.empty(0, opt_dim_t, dtype=dt, device=dev)
                if opt_dim_t > 0
                else None
            )
        if do_multi_rank_gather:
            # Every rank runs the same table config, so ``opts_cat is None``
            # agrees across ranks and the column list below is the same length
            # everywhere -- an asymmetric all_gather would hang.
            columns = [keys_cat, values_cat, slots_cat, scores_cat]
            if opts_cat is not None:
                columns.append(opts_cat)
            gathered = _all_gather_dumped_columns(
                [c.to(device_for_gather) for c in columns], pg
            )
            keys_cat, values_cat, slots_cat, scores_cat = gathered[:4]
            opts_cat = gathered[4] if opts_cat is not None else None
        elif keys_cat.device.type == "cuda":
            keys_cat = keys_cat.cpu()
            values_cat = values_cat.cpu()
            slots_cat = slots_cat.cpu()
            scores_cat = scores_cat.cpu()
            opts_cat = opts_cat.cpu() if opts_cat is not None else None
        return keys_cat, values_cat, slots_cat, opts_cat, scores_cat

    # -- Replay --

    def replay_increment(
        self,
        table_id: int,
        keys: Tensor,
        values: Tensor,
        optimizer_states: Optional[Tensor],
        scores: Optional[Tensor],
        slot_index: Tensor,
        insert_score: int,
        content: ReplayContent = ReplayContent.ALL,
        timestamp: Optional[int] = None,
    ) -> ReplayStats:
        """Write an ``incremental_dump`` delta back into one table.

        Each key is routed to the tier it came from -- ``slot_index`` bit 63
        selects HBM (0) or host (1) -- and written at that tier's slot. Nothing
        is promoted or spilled between tiers: the point is to reproduce the
        source's layout, and the source already decided which tier each key
        belongs to.

        Args:
            table_id: logical table within this storage.
            keys / values / optimizer_states / scores: the delta's columns for
                this table, row aligned. The last two may be ``None`` when
                *content* does not ask for them.
            slot_index: the source's tier-tagged key slot (bit 63 = tier).
            insert_score: the score a restored key gets in every non-recency
                score word when *content* omits ``SCORE``.
            content: which parts of each row to write; see
                :class:`ReplayContent`.
            timestamp: reference clock. Rebases the delta's timestamp ages when
                ``SCORE`` is requested, and stamps restored keys as
                freshly-inserted when it is not. Defaults to this device's clock.


        Batched at ``threads_in_wave`` keys per tier, as
        :meth:`DynamicEmbStorage.replay_increment` explains.

        Returns:
            :class:`ReplayStats` with ``upserted`` summed over both tiers.

        Raises:
            RuntimeError: a key could not be placed at its source slot.
        """
        stats = ReplayStats()
        n = keys.numel()
        if n == 0:
            return stats
        timestamp = device_timestamp() if timestamp is None else timestamp

        # Both tiers sit on the same device, so one size covers them.
        batch_size = self.tables[0].threads_in_wave
        tier_bit = slot_index.to(torch.int64) < 0  # bit 63 set -> host tier
        key_slots = slot_index.to(torch.int64) & 0x7FFFFFFFFFFFFFFF
        for tier, state in enumerate(self.tables):
            sel = tier_bit if tier else torch.logical_not(tier_bit)
            if not bool(sel.any()):
                continue
            sel_keys, sel_values = keys[sel], values[sel]
            sel_opts = optimizer_states[sel] if optimizer_states is not None else None
            sel_scores = scores[sel] if scores is not None else None
            sel_slots = key_slots[sel]
            sel_n = sel_keys.numel()
            for start in range(0, sel_n, batch_size):
                end = min(start + batch_size, sel_n)
                stats.merge(
                    _replay_state_increment(
                        state,
                        table_id,
                        sel_keys[start:end],
                        sel_values[start:end],
                        _slice(sel_opts, start, end),
                        _slice(sel_scores, start, end),
                        sel_slots[start:end],
                        timestamp,
                        insert_score,
                        content,
                    )
                )
        return stats

    def erase_keys(
        self,
        table_id: int,
        keys: Tensor,
        mode: EvictedItemMode = EvictedItemMode.DISCARD,
    ) -> int:
        """Erase keys from both tiers (a key may live in either).

        Returns how many were actually present. Summing across tiers is right
        here, not double counting: the tiers hold disjoint key sets, so a key
        erased from one is reported as absent by the other. *mode* applies to
        both, so a key is recorded wherever it was found.
        """
        removed = 0
        for state in self.tables:
            removed += _erase_state_keys(state, table_id, keys, mode)
        return removed

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
        if (
            self._host.key_index_map.num_scores_ > 1
            or self._hbm.key_index_map.num_scores_ > 1
        ):
            # Hybrid dump appends the HBM tier after the host tier into the same
            # score file; the tiers can have different num_scores (e.g. host
            # LruLfu = 2 words, HBM cache TIMESTAMP = 1 word), so a uniform score
            # file is ill-defined. Multi-word dump/load for HybridStorage needs
            # per-tier score files -- left as a follow-up.
            raise NotImplementedError(
                "dump is not yet supported for HybridStorage with multi-word "
                "score layouts (e.g. the (TIMESTAMP, LFU) compound score with caching)."
            )
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
        if (
            self._host.key_index_map.num_scores_ > 1
            or self._hbm.key_index_map.num_scores_ > 1
        ):
            # Mirror of the dump guard: multi-word (e.g. LruLfu) checkpoints for
            # HybridStorage need per-tier score files -- left as a follow-up.
            raise NotImplementedError(
                "load is not yet supported for HybridStorage with multi-word "
                "score layouts (e.g. the (TIMESTAMP, LFU) compound score with caching)."
            )
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
        for batch in export_keys_values_iter(self._hbm, device, batch_size, table_id):
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

        yield from export_keys_values_iter(self._host, device, batch_size, table_id)


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
