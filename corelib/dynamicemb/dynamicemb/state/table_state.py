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

"""The state of a group of fused tables: its buffers, its key map, its options, and how it is created."""

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

import torch
from dynamicemb.dynamicemb_config import (
    DynamicEmbTableOptions,
    EvictedItemMode,
    has_no_eviction,
)
from dynamicemb.extendable_tensor import (
    DeviceExtendableBuffer,
    ExtendableBuffer,
    HostExtendableBuffer,
)
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.score.spec import get_score_policy
from dynamicemb.scored_hashtable import ScorePolicy, ScoreSpec, get_scored_table
from dynamicemb_extensions import EvictStrategy, segmented_sum_cuda


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
