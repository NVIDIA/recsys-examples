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

import collections
import contextlib
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.checkpoint.files import (  # noqa: F401 - preserve checkpoint helper exports
    META_EMBEDDING_DIM,
    META_EMBEDDING_DTYPE,
    META_OPT_STATE_DTYPE,
    _iter_batches_from_files,
    _tensor_from_bytes,
    dtype_from_meta_name,
    dtype_to_meta_name,
    load_from_json,
    _raw_bytes,
    resolve_checkpoint_value_layout,
    save_to_json,
)
from dynamicemb.checkpoint.state_dump import (  # noqa: F401 (re-exported)
    _all_gather_dumped_columns,
    _dump_table,
    export_keys_values_iter,
)
from dynamicemb.checkpoint.state_load import (  # noqa: F401 (re-exported)
    _count_owned_keys,
    _load_key_values,
    _validate_load_meta,
    check_dist_type_migration,
    suppress_layout_warnings,
)
from dynamicemb.checkpoint.state_replay import (  # noqa: F401 (re-exported)
    _replay_at_slots,
    _replay_state_increment,
    _replay_write_values,
)
from dynamicemb.dynamicemb_config import (
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EvictedItemMode,
    ReplayContent,
    ScoreStrategy,
    align_to_table_size,
    get_physical_score_order,
    get_score_part,
    has_no_eviction,
    score_dump_permutation,
    score_load_permutation,
)
from dynamicemb.extendable_tensor import (
    DeviceExtendableBuffer,
    ExtendableBuffer,
    HostExtendableBuffer,
)
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb.score.checkpoint_columns import (  # noqa: F401 (re-exported)
    _age_to_scores,
    _dump_score_block,
    _fresh_score_block,
    _replay_score_block,
    _scores_to_age,
    _timestamp_score_columns,
)
from dynamicemb.score.args import (  # noqa: F401 (re-exported)
    get_find_score_arg,
    get_insert_score_arg,
    score_source,
)
from dynamicemb.score.spec import get_score_policy  # noqa: F401 (re-exported)
from dynamicemb.state.value_rows import (  # noqa: F401 (re-exported)
    _row_differs_from_slot,
    load_from_flat,
    load_from_flat_by,
    load_from_flat_single_table,
    store_to_flat,
    store_to_flat_single_table,
)
from dynamicemb.types import (
    SCORE_TYPE,
    Cache,
    CopyMode,
    ReplayStats,
    Storage,
)
from dynamicemb.utils import checkpoint_dtype, decode_table_values, encode_table_values
from dynamicemb_extensions import EvictStrategy, device_timestamp, flagged_compact
from dynamicemb_extensions import segmented_sum_cuda, select_insert_failed_values
from torch import Tensor, nn  # usort:skip

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


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
    threads_in_wave: int
    score: Optional[int] = None
    training: bool = False
    # Overflow region fields (per-table, only set when overflow is enabled)
    overflow_caps: Optional[List[int]] = None
    # The table never evicts: it grows instead, its key map is kept sparse so a
    # bucket does not fill up, and its rows are not limited by ``max_capacity``.
    no_eviction: bool = False
    # Estimated per-table size (last_collected + accumulated unique since collection);
    # CPU tensor of shape (num_tables,), used to avoid key_index_map.size() when not needed.
    estimated_table_sizes: Optional[torch.Tensor] = None
    # The same for the rows handed out, kept only by a table that never evicts: its
    # key map grows with the keys it holds and its value buffer with the rows.
    estimated_row_counts: Optional[torch.Tensor] = None
    collect_table_sizes_flag: bool = False
    # How many times ``key_index_map`` has been replaced (an expansion builds a
    # new one and drops the old). A slot number means something only in the map it
    # was read from, so whoever walks the slots across calls -- an erase by score,
    # in chunks -- compares this to know when to start over.
    key_map_generation: int = 0
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
    # A table that stores a row per key also records where each erased key sat:
    # the (slot, row) pair of the slot it left reclaimed. A reclaimed
    # slot keeps its row, and a replica can only keep the same rows if it
    # recreates those slots, including for keys it never held.
    erased_slot_chunks: List[torch.Tensor] = field(default_factory=list)
    erased_slot_tid_chunks: List[torch.Tensor] = field(default_factory=list)
    # The logical tables that stored rows and erased keys without retaining them:
    # those slots were left reclaimed with their rows and nothing reports them,
    # so no delta from such a table can be replayed exactly any more. Sticky -- a
    # replica built from earlier deltas cannot be brought back in line. Per
    # table: the state covers every logical table of a storage, and one table's
    # untracked erase says nothing about the others.
    erase_untracked_tables: Set[int] = field(default_factory=set)


# The key map of a table that never evicts is kept at most this full, so that no
# bucket fills up; its value buffer then holds exactly that many rows, i.e. the
# key map is twice the size of the value buffer, at creation and after every
# expansion.
NO_EVICTION_LOAD_FACTOR = 0.5


def create_table_state(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    enable_overflow: bool = False,
    evicted_item_mode: EvictedItemMode = EvictedItemMode.DISCARD,
    use_rows: bool = False,
    global_bucket_layout: bool = True,
    use_ref_counter: bool = True,
) -> DynamicEmbTableState:
    """Build the key map and value buffers for a group of tables.

    ``use_rows`` stores a value row per key (see ``LinearBucketTable``) instead
    of using the hash slot as the row, which lets an expansion keep every value
    where it is. It is for tables that grow and keep their keys, so it cannot be
    combined with ``enable_overflow`` (a cache keeps slot == row). It is
    independent of the no-eviction mode, which keeps no row in its score.

    ``global_bucket_layout`` lets a table whose options carry a
    ``global_capacity`` (a ``hash_bucket`` table sized by the planner) hash
    keys into the whole table's bucket space, so the keys that share a bucket
    do not depend on the number of shards and a checkpoint reloads at any world
    size without a bucket overflowing, once it has grown to ``max_capacity /
    bucket_capacity`` buckets. It holds only when this state *is* the whole
    shard: a table that starts smaller folds the whole table's bucket map into
    the buckets it has (``g % num_buckets``, see ``home_bucket_of``) and, as it
    grows, splits them, so its keys stay on their rank. A cache or a hybrid tier
    is a rescaled part of the shard and passes ``False``; a no-eviction table
    sizes its key map at twice its rows and is excluded here for the same
    reason.
    ``use_ref_counter=False`` leaves out the per-slot reference counter (4 bytes
    a slot) for a table whose slots are never pinned: the storage under a cache,
    where the cache pins. It cannot be combined with ``enable_overflow``.
    """
    if not options:
        raise ValueError("options must be non-empty")

    base_opt = options[0]
    no_eviction = has_no_eviction(base_opt.score_strategy)
    if no_eviction and enable_overflow:
        raise ValueError(
            "enable_overflow is not supported when score_strategy is NO_EVICTION"
        )
    if use_rows and enable_overflow:
        raise ValueError("use_rows is not supported with enable_overflow")
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
        from dynamicemb.score.jit import register_score_function

        cc_major, cc_minor = torch.cuda.get_device_capability(device_idx)
        score_fn_key = register_score_function(
            base_opt.score_function, base_opt.score_strategy, cc_major, cc_minor
        )

    # NO_EVICTION: the key map starts at twice the value buffer. The two then grow
    # on their own (the key map with the keys held, the buffer with the rows handed
    # out, up to the key map's slots), see ``_expand_if_need_no_eviction``.
    bucket_capacity = base_opt.bucket_capacity
    if no_eviction:
        capacities = []
        for opt in options:
            if opt.init_capacity is not None:
                cap = math.ceil(opt.init_capacity / NO_EVICTION_LOAD_FACTOR)
                aligned = (
                    (cap + bucket_capacity - 1) // bucket_capacity
                ) * bucket_capacity
                capacities.append(aligned)
            else:
                capacities.append(opt.init_capacity)
    else:
        capacities = [opt.init_capacity for opt in options]

    global_capacities: List[Optional[int]] = []
    for opt in options:
        global_capacity = getattr(opt, "global_capacity", None)
        if global_capacity is not None and global_bucket_layout and not no_eviction:
            # The fused group hashes every table with options[0]'s bucket
            # capacity (above); a global capacity derived with another one
            # would send keys to buckets the input distributor does not expect.
            if opt.bucket_capacity != bucket_capacity:
                raise ValueError(
                    f"global_capacity {global_capacity} was derived with "
                    f"bucket_capacity {opt.bucket_capacity}, but the fused "
                    f"group hashes with bucket_capacity {bucket_capacity}"
                )
            global_capacities.append(global_capacity)
        else:
            global_capacities.append(None)

    key_index_map = get_scored_table(
        capacity=capacities,
        bucket_capacity=base_opt.bucket_capacity,
        key_type=base_opt.index_type,
        score_specs=[score_policy],
        device=device,
        enable_overflow=enable_overflow,
        score_fn_key=score_fn_key,
        use_rows=use_rows,
        global_capacities=global_capacities,
        use_ref_counter=use_ref_counter,
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

    # Table (embedding) capacity may differ from key_index_map in NO_EVICTION: the
    # buffer is NO_EVICTION_LOAD_FACTOR of the key map, which is the whole key
    # map's capacity after rounding to buckets (so it can exceed ``init_capacity``
    # by up to half a bucket), the same rule an expansion keeps it to.
    if no_eviction:
        table_caps = [
            math.ceil(c * NO_EVICTION_LOAD_FACTOR) for c in key_index_map_caps
        ]
    else:
        table_caps = list(key_index_map_caps)

    overflow_caps_list: Optional[List[int]] = None

    if enable_overflow:
        ovf_cap = key_index_map.overflow_bucket_capacity_
        overflow_caps_list = [ovf_cap] * num_tables

    # Which memory the values live in is the group's tier, not options[0]'s
    # budget: a host tier zeroes every table's budget, while an HBM tier may
    # hold a table with no budget of its own beside one that has some.
    on_host = all(opt.local_hbm_for_values == 0 for opt in options)
    tables: List[ExtendableBuffer] = []
    for i, (cap, vd) in enumerate(zip(table_caps, value_dims)):
        total_cap = cap
        if enable_overflow:
            total_cap += overflow_caps_list[i]
        shape = (total_cap, vd)
        if on_host:
            tables.append(HostExtendableBuffer(shape, emb_dtype, device))
        else:
            tables.append(DeviceExtendableBuffer(shape, emb_dtype, device))

    if use_rows:
        key_index_map.set_row_limits([t.shape[0] for t in tables])

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
        threads_in_wave=threads_in_wave,
        score=None,
        training=False,
        overflow_caps=overflow_caps_list,
        no_eviction=no_eviction,
        estimated_table_sizes=torch.zeros(
            num_tables, dtype=torch.int64, pin_memory=True
        ),
        estimated_row_counts=torch.zeros(num_tables, dtype=torch.int64, pin_memory=True)
        if no_eviction
        else None,
        evicted_item_mode=evicted_item_mode,
    )


def collect_table_sizes_to_device(state: DynamicEmbTableState) -> torch.Tensor:
    """Collect per-table sizes (main table only, no overflow) into a tensor on state.device.

    The keys each table holds. A table that never evicts also has the rows it has
    handed out, which can be more: an erased key's slot keeps its row until a key
    takes the slot again (see :func:`collect_row_counts_to_device`).

    Uses an async CUDA kernel when key_index_map exposes table_bucket_offsets_
    and bucket_sizes; otherwise falls back to a sync Python loop. No GPU-CPU
    synchronization when the kernel path is used.

    Returns:
        Tensor of shape (num_tables,) dtype torch.int64 on state.device.
    """
    km = state.key_index_map
    return segmented_sum_cuda(km.bucket_sizes, km.table_bucket_offsets_)


def collect_row_counts_to_device(state: DynamicEmbTableState) -> torch.Tensor:
    """The value rows each table has handed out, as a tensor on state.device.

    Only a table that stores a row per key has them. A slot owns at most one row,
    so this never passes the table's key map capacity -- the bound a value buffer
    needs to grow to and no further.
    """
    return state.key_index_map.row_counters_.to(torch.int64)


def collect_table_sizes_for_state(
    state: DynamicEmbTableState, non_blocking: bool = True
) -> None:
    """Copy device table sizes into ``state.estimated_table_sizes`` when flag is set."""
    if not state.collect_table_sizes_flag:
        return
    table_sizes = collect_table_sizes_to_device(state)
    state.estimated_table_sizes.copy_(table_sizes, non_blocking=non_blocking)
    if state.estimated_row_counts is not None:
        state.estimated_row_counts.copy_(
            collect_row_counts_to_device(state), non_blocking=non_blocking
        )


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
                values_batch = load_from_flat_single_table(state, src_slots, table_id)
                dst_values_list.append((dst_slots, table_id, values_batch))

        if new_key_index_map.use_rows_:
            # Reclaimed (erased) slots still own their rows; keep them too.
            new_key_index_map.carry_rows_from(old_key_index_map, table_id)

        for dst_indices, table_id, values_batch in dst_values_list:
            store_to_flat_single_table(state, dst_indices, table_id, values_batch)

    state.key_index_map = new_key_index_map
    state.key_map_generation += 1
    state.capacity = new_key_index_map.capacity()
    if new_key_index_map.use_rows_:
        new_key_index_map.set_row_limits([t.shape[0] for t in state.tables])


# ---------------------------------------------------------------------------
# Erase by score, in chunks
# ---------------------------------------------------------------------------

# The slots of a table one chunk of an erase by score scans, at most. A chunk is also
# the length of the kernel the training's own kernels can end up waiting behind, so
# the size is a trade between how much the scan disturbs the training and how long it
# takes. Measured on an H100, a table of 2**29 slots at half load, training steps of
# 0.7 ms, scans one after the other and two chunks in flight: 2**20 or 2**22 slots
# slow a step by about 10%, 2**24 by 110%, 2**26 by 8 times; in the same 300 steps
# 13-15 scans finish with 2**22, 115 with 2**24. A chunk of 2**22 slots is about 30
# microseconds of scan, so a smaller one is mostly the cost of launching it (2**18:
# 46 ms for the table, against 3.7 ms for 2**22). It is a parameter of the call.
ERASE_CHUNK_SLOTS = 1 << 22

# How many chunks of an erase by score may be launched and not yet finished on the
# device. Launching is some thirty times faster than a chunk runs, so without a limit
# the erase queues the whole table at once, and the prefetch that takes the table next
# -- whose stream waits for the erase's last event -- waits for all of it. With it, that
# wait is at most this many chunks (about a millisecond for two). Two keep the device
# busy: one chunk running, the next already queued.
ERASE_MAX_CHUNKS_IN_FLIGHT = 2


def next_erase_chunk_end(
    position: int,
    table_slots: int,
    bucket_capacity: int,
    target_slots: int = ERASE_CHUNK_SLOTS,
) -> int:
    """Where the chunk of an erase by score that starts at ``position`` ends.

    A chunk is a whole number of buckets and lies inside one table:

    * the size is counted in buckets, ``target_slots // bucket_capacity`` of them,
      so a bucket is never split;
    * never more than the table has, and never less than one bucket: a table smaller
      than a chunk is scanned as one, and a target below a bucket is a bucket;
    * it stops at the table's last slot (the last chunk is the short one), so it
      never reaches into the next table.

    ``position`` and the returned end are slots counted from the start of the table.
    """
    if bucket_capacity <= 0 or target_slots <= 0:
        raise ValueError("bucket_capacity and target_slots must be positive")
    if table_slots <= 0 or table_slots % bucket_capacity != 0:
        raise ValueError(
            f"a table is a whole number of buckets, got {table_slots} slots of "
            f"{bucket_capacity}-slot buckets"
        )
    if not 0 <= position < table_slots or position % bucket_capacity != 0:
        raise ValueError(
            f"position {position} is not the start of a bucket inside the table's "
            f"{table_slots} slots"
        )
    table_buckets = table_slots // bucket_capacity
    chunk_buckets = max(1, min(table_buckets, target_slots // bucket_capacity))
    return min(position + chunk_buckets * bucket_capacity, table_slots)


def _erase_state_by_score(
    state: DynamicEmbTableState,
    threshold: int,
    table_ids: Sequence[int],
    counts: Tensor,
    chunk_slots: int = ERASE_CHUNK_SLOTS,
    chunk_guard: Optional[Callable[[], ContextManager]] = None,
    max_in_flight: int = ERASE_MAX_CHUNKS_IN_FLIGHT,
    score_word: Optional[int] = None,
) -> None:
    """Erase the keys of ``table_ids`` whose score is below ``threshold``, a chunk
    of slots at a time (:func:`next_erase_chunk_end`). ``score_word`` is the word
    compared, as in :meth:`LinearBucketTable.erase_by_score`.

    No more than ``max_in_flight`` chunks are launched and unfinished at once: before
    launching one, the thread waits -- outside the guard, so holding nothing and not
    the GIL -- for the chunk ``max_in_flight`` before it to finish on the device. What a
    caller that takes the table after the guard has to wait for on the device is then
    bounded by those chunks, not by the rest of the scan.

    Each chunk runs inside ``chunk_guard()`` -- a context manager made anew for
    every chunk, which does nothing by default. That is where a caller that has to
    keep the table to itself for a chunk (and only a chunk) takes and drops its
    lock. Nothing about the table is planned ahead: the key map, the table's size
    and the chunk are all read inside the guard, because between two chunks
    another thread may have expanded the table, replacing the key map and freeing
    the old one.

    A replaced key map also moves every key to another slot, so where the scan had
    got to means nothing in the new one: when ``state.key_map_generation`` is not
    the one this table's scan started with, the table is scanned again from its first
    bucket. That is safe -- erasing by score is idempotent -- and a key an expansion
    moved past the scan is left for the next erase.

    ``counts`` is the device tensor of :meth:`LinearBucketTable.erase_by_score`,
    added to by every chunk.

    The keys it removes are not recorded (no ``RETAIN_KEY``), and a table that stores a
    row per key leaves each erased slot reclaimed with its row, which only a recorded
    erase lets an incremental dump report. So a table it removed any key from is marked,
    as :func:`_erase_state_keys` marks one (``erase_untracked_tables``): a delta dumped
    from it cannot be replayed exactly, and ``replay_increment`` refuses it. A table
    from which it removed nothing is not marked -- the mark is sticky. To know, each
    table is scanned into a count of its own, which the worker reads when the table is
    done (a wait for its own stream only) before adding it to ``counts``.
    """
    for table_id in table_ids:
        if not 0 <= table_id < state.num_tables:
            raise ValueError(
                f"table_id {table_id} is not one of the {state.num_tables} tables"
            )
    if max_in_flight < 1:
        raise ValueError(f"max_in_flight must be at least 1, got {max_in_flight}")
    guard = contextlib.nullcontext if chunk_guard is None else chunk_guard
    in_flight: "collections.deque[torch.cuda.Event]" = collections.deque()
    for table_id in table_ids:
        table_counts = torch.zeros(2, dtype=counts.dtype, device=counts.device)
        # Marked also when the scan fails half way: whatever it removed before is gone.
        try:
            position = 0
            generation: Optional[int] = None
            finished = False
            while not finished:
                if len(in_flight) >= max_in_flight:
                    in_flight.popleft().synchronize()
                with guard():
                    key_map = state.key_index_map
                    if generation != state.key_map_generation:
                        # The first chunk, or the key map was replaced since the last.
                        generation = state.key_map_generation
                        position = 0
                    table_slots = key_map.main_capacity(table_id)
                    end = next_erase_chunk_end(
                        position, table_slots, key_map.bucket_capacity_, chunk_slots
                    )
                    key_map.erase_by_score(
                        table_id,
                        threshold,
                        table_counts,
                        score_word=score_word,
                        begin=position,
                        end=end,
                    )
                    position = end
                    finished = position >= table_slots
                chunk_launched = torch.cuda.Event()
                chunk_launched.record()
                in_flight.append(chunk_launched)
        finally:
            counts.add_(table_counts)
            if state.key_index_map.use_rows_ and int(table_counts[0]) > 0:
                state.erase_untracked_tables.add(table_id)


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
        rows = state.tables[i].shape[0]
        target = _value_rows_target(
            rows, need_rows[i], state.key_index_map.per_table_capacity_[i]
        )
        if target > rows:
            state.tables[i].extend((target - rows, state.table_value_dims_cpu[i]))
            grown = True
    if grown:
        state.table_ptrs_dev.copy_(get_table_ptrs(state.tables, state.device))
        state.key_index_map.set_row_limits([t.shape[0] for t in state.tables])


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
            int(need[i]) > state.tables[i].shape[0]
            and state.tables[i].shape[0] < state.key_index_map.per_table_capacity_[i]
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


# ---------------------------------------------------------------------------
# Free functions operating on DynamicEmbTableState
# ---------------------------------------------------------------------------


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
      - a table that stores its rows (``use_rows``): key slot and value row are
        independent, so ``slot_index`` is an ``[N, 2]`` int64 tensor, column 0 =
        key slot, column 1 = value row.

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
    if not _row_differs_from_slot(state):
        return slot_indices  # key slot == value row
    flat_rows = flat_rows.to(device=state.device, dtype=torch.int64)
    return torch.stack([slot_indices, flat_rows], dim=1)


def _split_value_row(
    state: DynamicEmbTableState, table_id: int, values: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split a loaded ``[N, value_dim]`` row into (embedding, optimizer state).

    ``load_from_flat_single_table`` returns the table's own compact row, so the
    optimizer state is the trailing ``value_dim - emb_dim`` columns. That runtime
    width is not all payload: rowwise Adagrad reserves a fixed 16 bytes per row
    in the fused FBGEMM layout but only ever fills one accumulator scalar. The
    dumped block is therefore narrowed to the width the file checkpoint uses --
    the same ``optimizer.states_for_checkpoint`` ``_dump_table``
    applies -- so a delta and a checkpoint describe a row identically, and the
    padding is not paid for on every dump. Replay expands it back with
    ``optimizer.states_from_checkpoint``.

    Tables whose optimizer keeps no per-row state (e.g. plain SGD) get ``None``
    rather than a zero-width tensor, matching :func:`export_keys_values_iter`.
    """
    emb_dim = state.table_emb_dims_cpu[table_id]
    optim_state_dim = state.table_value_dims_cpu[table_id] - emb_dim
    emb = decode_table_values(values[:, :emb_dim], checkpoint_dtype(state.emb_dtype))
    if optim_state_dim <= 0:
        return emb, None
    opt = values[:, -optim_state_dim:]
    return (
        emb,
        state.optimizer.states_for_checkpoint(opt, emb_dim).contiguous(),
    )


def _find_keys(
    state: DynamicEmbTableState,
    unique_keys: torch.Tensor,
    table_ids: torch.Tensor,
    score_input: Optional[torch.Tensor] = None,
    *,
    const_lookup: bool = False,
    return_slots: bool = False,
) -> Tuple:
    """Key-only find: lookup in hash table, return missing info + indices.

    The indices are value rows for a table that stores them, hash slots otherwise.
    With ``return_slots`` the hash slots follow as a ninth element: the reference
    counters are keyed by slot.

    ``score_input`` is the one tensor a table's score is made from, as
    :func:`score_source` chose it (see :func:`get_find_score_arg`).
    """
    if unique_keys.dtype != state.key_index_map.key_type:
        unique_keys = unique_keys.to(state.key_index_map.key_type)

    batch = unique_keys.size(0)
    device = unique_keys.device

    score_arg = get_find_score_arg(
        state,
        batch,
        device,
        score_input,
        const_lookup=const_lookup,
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
            *((torch.empty(batch, dtype=torch.int64, device=device),) * return_slots),
        )

    km = state.key_index_map
    slots = None
    if getattr(km, "enable_overflow_", False):
        score_out, founds, indices = km.lookup_with_overflow(
            unique_keys, table_ids, score_arg
        )
        slots = indices  # a cache keeps slot == row
    elif return_slots:
        score_out, founds, indices, slots = km.lookup(
            unique_keys, table_ids, score_arg, return_slots=True
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
        *((slots,) * return_slots),
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
    slot_index: Optional[torch.Tensor] = None,
) -> None:
    """Append an explicit erase's keys to the state's erased-key buffer.

    Unlike :func:`_append_evicted` there is no count to read back: the caller has
    already masked the keys down to the ones the erase actually removed, so this
    costs no extra device sync. ``slot_index`` is the ``[M, 2]`` (slot, row) each
    erased key left behind, for a table that stores rows.
    """
    if keys.numel() == 0:
        return
    state.erased_key_chunks.append(keys.clone())
    state.erased_tid_chunks.append(
        torch.full((keys.numel(),), table_id, dtype=torch.int64, device=keys.device)
    )
    if slot_index is not None:
        state.erased_slot_chunks.append(slot_index.clone())
        state.erased_slot_tid_chunks.append(
            torch.full(
                (slot_index.size(0),), table_id, dtype=torch.int64, device=keys.device
            )
        )


def _pop_chunked_keys(
    key_chunks: List[torch.Tensor],
    tid_chunks: List[torch.Tensor],
    table_id: int,
    state: DynamicEmbTableState,
    dtype: Optional[torch.dtype] = None,
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
        empty = torch.empty(
            0,
            dtype=state.key_index_map.key_type if dtype is None else dtype,
            device=state.device,
        )
        return empty, [], []
    keys = torch.cat(key_chunks)
    tids = torch.cat(tid_chunks)
    mask = tids == table_id
    out = torch.unique(keys[mask], dim=0 if keys.dim() > 1 else None)
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


def _pop_state_erased_slot_index(
    state: DynamicEmbTableState, table_id: int
) -> torch.Tensor:
    """Pop (return + clear) the (slot, row) pair each explicit erase left behind,
    one per slot: a slot erased, refilled and erased again is one reclaimed slot
    with one row, however many keys passed through it."""
    out, state.erased_slot_chunks, state.erased_slot_tid_chunks = _pop_chunked_keys(
        state.erased_slot_chunks,
        state.erased_slot_tid_chunks,
        table_id,
        state,
        dtype=torch.int64,
    )
    return out.reshape(-1, 2)


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
    )
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
            collect_evicted=True,
        )
        _append_evicted(state, evicted_keys, evicted_table_ids, num_evicted)
    else:
        indices = state.key_index_map.insert(unique_keys, table_ids, score_arg)
    store_to_flat(state, indices, table_ids, unique_values)


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
        indices,
        num_evicted,
        evicted_keys,
        evicted_table_ids,
        evicted_indices,
        evicted_scores,
    )


# ---------------------------------------------------------------------------
# Replay – write an incremental_dump delta back into a table
# ---------------------------------------------------------------------------


def _slice(t: Optional[torch.Tensor], start: int, end: int) -> Optional[torch.Tensor]:
    """Slice an optional column, keeping ``None`` as ``None``."""
    return None if t is None else t[start:end]


def _erase_state_keys(
    state: DynamicEmbTableState,
    table_id: int,
    keys: torch.Tensor,
    mode: EvictedItemMode = EvictedItemMode.DISCARD,
    track_untracked: bool = True,
) -> int:
    """Erase keys from one table state. Returns how many were actually present.

    The erase kernel reports the slot it cleared, or ``-1`` for a key that was
    not there, so the count is the real number removed rather than the number
    asked for -- a delta can legitimately name keys this replica never held.

    *mode* is this call's, not the table's: retaining an erase costs only a copy
    of keys already in hand, so there is nothing to configure up front and each
    caller can decide whether its removals are worth reporting.

    A table that stores rows leaves each erased slot reclaimed with its row, which
    only a retained erase reports (see ``erase_untracked``). ``track_untracked``
    is off for a replica applying a delta's removals, which are not its own.
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
        slot_index = None
        if state.key_index_map.use_rows_:
            # The slot stays reclaimed with its row; record both so a replica can
            # recreate it, laid out the way a dumped slot_index is.
            slots = indices[removed]
            rows = state.key_index_map._rows_of(slots, tids[removed], False)
            slot_index = torch.stack([slots, rows], dim=1)
        _append_erased(state, keys[removed], table_id, slot_index)
    num_removed = int(removed.sum().item())
    if (
        track_untracked
        and num_removed > 0
        and state.key_index_map.use_rows_
        and EvictedItemMode.RETAIN_KEY not in mode
    ):
        state.erase_untracked_tables.add(table_id)
    return num_removed


# ---------------------------------------------------------------------------
# DynamicEmbCache – Cache interface (key-only find / insert_and_evict)
# ---------------------------------------------------------------------------


class DynamicEmbCache(Cache):
    def __init__(
        self,
        options: List[DynamicEmbTableOptions],
        optimizer: BaseDynamicEmbeddingOptimizer,
    ):
        self._state = create_table_state(
            options, optimizer, enable_overflow=True, global_bucket_layout=False
        )
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
        score_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lookup with overflow fallback. Returns (score_out, founds, indices).

        ``score_input`` is what the cache's score is made from (see
        :func:`score_source`); only a cache that owns scores (the NINE path) takes
        one. The cache of the inclusive path ranks by timestamp and takes none.
        """
        state = self._state
        score_arg = get_find_score_arg(
            state,
            unique_keys.size(0),
            unique_keys.device,
            score_input,
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
        report_refused: bool = False,
    ) -> Tuple:
        """Insert with counter-aware eviction and overflow fallback; with
        ``report_refused`` the number of keys it could not place follows."""
        state = self._state
        score_arg = get_insert_score_arg(state, keys.numel(), keys.device, scores)
        result = state.key_index_map.insert_and_evict_with_counter_and_overflow(
            keys, table_ids, score_arg, report_refused=report_refused
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
        return self._state.optimizer.get_initial_optimizer_state()

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
        use_ref_counter: bool = True,
    ):
        """``use_ref_counter=False`` for a storage that sits under a cache: the
        cache pins the slots a prefetched batch uses, never this storage's, so
        the per-slot counter would only take memory. Such a storage cannot
        :meth:`increment_counter` / :meth:`decrement_counter`."""
        self._state = create_table_state(
            options,
            optimizer,
            # DynamicEmbStorage is always a last tier (single tier or the backing
            # store under a cache), so it honors evicted_item_mode.
            evicted_item_mode=options[0].evicted_item_mode,
            # Every key keeps its own value row, so a rehash never moves values.
            use_rows=True,
            use_ref_counter=use_ref_counter,
        )

    @property
    def key_index_map(self):
        return self._state.key_index_map

    def erase_by_score(
        self,
        threshold: int,
        table_ids: Sequence[int],
        counts: Tensor,
        *,
        score_word: Optional[int] = None,
        chunk_slots: int = ERASE_CHUNK_SLOTS,
        chunk_guard: Optional[Callable[[], ContextManager]] = None,
        max_in_flight: int = ERASE_MAX_CHUNKS_IN_FLIGHT,
    ) -> None:
        """Erase the keys of logical tables ``table_ids`` whose score is below
        ``threshold``, sparing those a prefetched batch has pinned.

        The tables are scanned a chunk of at most ``chunk_slots`` slots at a time, each
        inside ``chunk_guard()``, with at most ``max_in_flight`` chunks launched and
        unfinished (see :func:`_erase_state_by_score`). ``counts`` is an
        int64 tensor of two on the device, ``[erased, spared]``, added to. The kernels run on
        the current stream and nothing waits for them, except that at the end of each table
        the count of its removals is read back, to know whether to mark it (below).

        One score word is compared, as in :meth:`LinearBucketTable.erase_by_score`:
        ``score_word`` names it for an ``LRU_LFU`` table (``0`` the last-access
        timestamp, ``1`` the frequency) and is left out for a single-score one. A table
        the scan removed a key from is marked (:meth:`erase_untracked`): the removals are
        not recorded, so a delta dumped from it cannot be replayed exactly.
        """
        _erase_state_by_score(
            self._state,
            threshold,
            table_ids,
            counts,
            chunk_slots,
            chunk_guard,
            max_in_flight,
            score_word,
        )

    @property
    def tables(self) -> List[DynamicEmbTableState]:
        """The tiers this storage is made of -- one, here.

        Mirrors :attr:`HybridStorage.tables` so that a caller asking a layout
        question ("how much capacity does table i have across this storage?")
        can iterate uniformly instead of branching on which storage it holds.
        """
        return [self._state]

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
                    score_input=score_source(state.evict_strategy, lfu_freq),
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

    def pop_erased_slot_index(self, table_id: int) -> torch.Tensor:
        """Return + clear the ``[M, 2]`` (slot, row) of each slot an explicit
        erase left reclaimed in ``table_id`` (empty unless the table stores
        rows)."""
        return _pop_state_erased_slot_index(self._state, table_id)

    # -- Storage interface --

    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        score_input: Optional[torch.Tensor] = None,
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
            self._state,
            unique_keys,
            table_ids,
            score_input,
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
        flat_rows = indices
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
        shard_index: Optional[int] = None,
        num_shards: Optional[int] = None,
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
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        # Every key this rank keeps may be a new one. The count has to be those
        # keys themselves: expand_if_need adds it to what the table holds and sizes
        # the table for that total, so passing only the part that overflows the
        # value buffer would undercount the keys it has to fit. And only this
        # rank's: the file can hold other ranks' keys too, and a table that never
        # evicts has no max_capacity to stop it growing for all of them.
        unique_size_per_table[table_id] = _count_owned_keys(
            emb_key_path,
            device,
            params.dist_type,
            shard_index=shard_index,
            num_shards=num_shards,
            global_capacity=params.global_capacity,
            bucket_capacity=params.bucket_capacity,
        )
        self.expand_if_need(unique_size_per_table)

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
            params.dist_type,
            num_scores=num_scores,
            emb_dtype=params.emb_dtype,
            opt_state_dtype=params.opt_state_dtype,
            shard_index=shard_index,
            num_shards=num_shards,
            global_capacity=params.global_capacity,
            bucket_capacity=params.bucket_capacity,
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
        ``slot_index`` is the key-slot/value-row used by replay, ``[N, 2]`` for a
        table that stores rows (see
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
            # ``indices`` are the rows of a ``use_rows`` table, hash slots
            # otherwise; the (slot, row) slot_index and the score blocks need slots.
            keys, named_scores, indices, slots = s.key_index_map.incremental_dump(
                {s.incremental_score_name: threshold},
                pg=pg,
                return_index=True,
                return_slots=True,
                table_id=table_id,
            )
            scores_batch = named_scores[s.incremental_score_name]
            flat_rows = indices
            values = load_from_flat_single_table(s, flat_rows, table_id)
            value, opt = _split_value_row(s, table_id, values)
            key = keys.to(s.device) if keys.device.type != "cuda" else keys
            slot = _encode_slot_index(s, slots, flat_rows)
            score_block = _dump_score_block(s, table_id, slots, scores_batch, timestamp)
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
            values_cat = torch.empty(
                0, emb_dim_t, dtype=checkpoint_dtype(state.emb_dtype), device=dev
            )
            slots_cat = torch.empty(
                (0, 2) if _row_differs_from_slot(state) else (0,),
                dtype=torch.int64,
                device=dev,
            )
            scores_cat = torch.empty(0, num_scores, dtype=SCORE_TYPE, device=dev)
            opts_cat = (
                torch.empty(
                    0,
                    opt_dim_t,
                    dtype=state.optimizer.get_state_dtype(state.emb_dtype),
                    device=dev,
                )
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
            slot_index: the source's key-slot/value-row (``[n, 2]`` for a table that
                stores rows).
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
        from_replay: bool = False,
    ) -> int:
        """Erase keys from the table (used to replay a delta's removals).

        Returns how many of them were actually present. *mode* decides whether
        the keys actually removed are recorded for :meth:`pop_erased_keys`.

        A table that stores a row per key leaves each erased slot reclaimed with
        its row, and only a retained erase (``RETAIN_KEY``) lets an incremental
        dump tell a replica to recreate it. Erasing without it is allowed, but
        marks the table (:meth:`erase_untracked`): a delta dumped from it can no
        longer be replayed exactly, and ``replay_increment`` refuses one.
        ``from_replay`` is for a replica applying a delta's removals, which are
        not this table's own to report.
        """
        return _erase_state_keys(
            self._state, table_id, keys, mode, track_untracked=not from_replay
        )

    def erase_untracked(self, table_id: int) -> bool:
        """Whether keys of ``table_id`` were erased without being retained (see
        ``DynamicEmbTableState.erase_untracked_tables``); sticky."""
        return table_id in self._state.erase_untracked_tables

    def restore_reclaimed_slots(self, table_id: int, slot_index: Tensor) -> None:
        """Recreate the reclaimed slots a delta reports, with their rows.

        ``slot_index`` is the ``[M, 2]`` (slot, row) that
        :meth:`pop_erased_slot_index` returned on the source. Applied after the
        delta's removals and before its upserts, so a slot an erased key left and
        another key then took ends up holding that key, while one nobody took
        stays reclaimed with its row.
        """
        self._state.key_index_map.restore_reclaimed_slots(
            slot_index.to(device=self._state.device, dtype=torch.int64), table_id
        )

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
        return self._state.optimizer.get_initial_optimizer_state()

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
        # Both tiers are rescaled parts of the shard, so neither holds the
        # exact bucket count the global bucket map needs (see
        # create_table_state); a hybrid table keeps today's per-tier hashing.
        # Neither tier is ever pinned (HybridStorage takes the generic path, with
        # no PrefetchState), so neither keeps a per-slot reference counter.
        self._hbm = create_table_state(
            hbm_options, optimizer, global_bucket_layout=False, use_ref_counter=False
        )
        self._host = create_table_state(
            host_options,
            optimizer,
            evicted_item_mode=host_options[0].evicted_item_mode,
            global_bucket_layout=False,
            use_ref_counter=False,
        )
        self.optimizer = optimizer

    @property
    def tables(self) -> List[DynamicEmbTableState]:
        """The tiers this storage is made of: HBM first, then host.

        Order matters to callers that want a single tier's property rather than
        a sum -- bucket capacity and score-word count are read off ``tables[0]``,
        the HBM tier.
        """
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
        return self._hbm.optimizer.get_initial_optimizer_state()

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
        score_input: Optional[torch.Tensor] = None,
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
        if (
            score_input is not None
            and self._hbm.evict_strategy == EvictStrategy.KCustomized
        ):
            raise NotImplementedError(
                "A score per key is not supported by HybridStorage."
            )
        result_hbm = _find_keys(
            self._hbm,
            unique_keys,
            table_ids,
            score_input,
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

        flat_hbm = indices_hbm
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

        flat_host = indices_host
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
            and score_input is not None
        ):
            if global_missing_indices.numel() == 0:
                assert (
                    global_missing_scores is None or global_missing_scores.numel() == 0
                )
            else:
                assert global_missing_scores is not None
                assert global_missing_scores.numel() == global_missing_indices.numel()
                expected = score_input[global_missing_indices]
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
                flat_hbm = indices_hbm
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
                    flat_host = indices_host
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
            flat_rows = indices
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
            values_cat = torch.empty(
                0, emb_dim_t, dtype=checkpoint_dtype(dt), device=dev
            )
            slots_cat = torch.empty(0, dtype=torch.int64, device=dev)
            scores_cat = torch.empty(0, num_scores, dtype=SCORE_TYPE, device=dev)
            opts_cat = (
                torch.empty(
                    0, opt_dim_t, dtype=st0.optimizer.get_state_dtype(dt), device=dev
                )
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
        from_replay: bool = False,
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
        if self._host.emb_dtype != self._hbm.emb_dtype:
            # The HBM tier is appended into the host tier's value files, under a
            # single meta describing one dtype, so two precisions in one file
            # would be unreadable. Both tiers are built from the same table
            # options today, so this is a guard, not a supported configuration.
            raise NotImplementedError(
                "dump is not supported for HybridStorage whose tiers store "
                f"different precisions (host {self._host.emb_dtype}, HBM "
                f"{self._hbm.emb_dtype}); they share one value file."
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
        shard_index: Optional[int] = None,
        num_shards: Optional[int] = None,
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
            # The HBM tier hashes with its own (larger) bucket capacity; the
            # host tier kept the shard's, which the routing rule is stated in.
            routing_options=self._host.options_list[table_id],
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
            params.dist_type,
            emb_dtype=params.emb_dtype,
            opt_state_dtype=params.opt_state_dtype,
            shard_index=shard_index,
            num_shards=num_shards,
            global_capacity=params.global_capacity,
            bucket_capacity=params.bucket_capacity,
        ):
            if keys.numel() == 0:
                continue

            if opt_states is None and params.runtime_optstate_dim > 0:
                opt_states = torch.empty(
                    keys.numel(),
                    params.runtime_optstate_dim,
                    dtype=self._hbm.emb_dtype,
                    device=device,
                )
                self._hbm.optimizer.reset_optimizer_states(
                    opt_states,
                    emb_dims=self._hbm.table_emb_dims_cpu[table_id],
                )
            elif opt_states is not None and params.runtime_optstate_dim > 0:
                opt_states = self._hbm.optimizer.states_from_checkpoint(
                    opt_states,
                    self._hbm.table_emb_dims_cpu[table_id],
                    self._hbm.emb_dtype,
                    device,
                )

            vtype = self._hbm.emb_dtype
            values = (
                torch.cat(
                    [encode_table_values(embeddings, vtype), opt_states.to(vtype)],
                    dim=-1,
                )
                if opt_states is not None
                else encode_table_values(embeddings, vtype)
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


def _select(values: torch.Tensor, rows: torch.Tensor, all_rows: bool) -> torch.Tensor:
    """``values[rows]`` -- ``flagged_compact`` only takes 1-D inputs.

    ``all_rows`` says the caller's mask kept every row; ``flagged_compact`` emits
    its index in ascending order, so ``rows`` is then the identity and the gather
    is skipped. The row count alone cannot say that: an index of the same length
    may still permute the rows.
    """
    return values if all_rows else values[rows]


def _write_back_evicted(
    storage: DynamicEmbStorage,
    next_tier: Optional[Storage],
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Return what the cache evicted to the tier below it.

    A key still in storage keeps its score, so only the value moves. One storage
    has evicted since goes to ``next_tier``; with none it is dropped, kept as an
    evicted key when storage retains them.
    """
    st = storage._state
    in_storage = _find_keys(st, keys, table_ids, const_lookup=True)[5]
    with torch.cuda.nvtx.range("op:flagged_compact"):
        num_present, present, (present_keys, present_tids) = flagged_compact(
            in_storage, [keys, table_ids]
        )
    with torch.cuda.nvtx.range("op:storage_insert"):
        if num_present > 0:
            storage.insert(
                present_keys,
                present_tids,
                _select(values, present, num_present == keys.numel()),
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
        if next_tier is not None:
            next_tier.insert(gone_keys, gone_tids, gone_values)
        elif EvictedItemMode.RETAIN_KEY in st.evicted_item_mode:
            _append_evicted(st, gone_keys, gone_tids, torch.tensor(gone_keys.numel()))


def flush_cache(
    cache: DynamicEmbCache,
    storage: Storage,
    external_storage: Optional[Storage] = None,
    *,
    cache_owns_scores: bool = False,
) -> None:
    """Push every cache entry's value down, as the prefetch path does on eviction.

    Inclusive path (the default): storage keeps the scores, so a key it still has
    only has its value updated. One it has evicted goes to ``external_storage``, or
    is dropped when there is none: it is not re-inserted, which would evict
    another row.

    NINE path (``cache_owns_scores``): ``storage`` is the external store itself, and
    the cache owns the scores, so each entry goes down with its score.
    """
    state = cache._state
    batch_size = state.threads_in_wave

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
            tid = torch.full((keys.numel(),), t, dtype=torch.int64, device=keys.device)
            values = load_from_flat(state, indices, tid, copy_mode=CopyMode.VALUE)
            if cache_owns_scores:
                storage.insert(keys, tid, values, named_scores[state.score_policy.name])
            else:
                _write_back_evicted(storage, external_storage, keys, tid, values)


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
        initializer(embs, keys, table_ids, missing_indices)

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
        initializer(
            storage_embs, missing_keys, missing_table_ids, missing_indices_in_storage
        )

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
