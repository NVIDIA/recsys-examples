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
import functools
import inspect
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch  # usort:skip
import torch.distributed as dist
from dynamicemb import sync_free as sf
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
from dynamicemb.growth_order import GrowthOrder
from dynamicemb.key_ownership import owned_key_mask
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.rehash_plan import (
    chunks_are_independent,
    lands_in_the_bucket_of_origin_or_a_new_one,
)
from dynamicemb.scored_hashtable import (
    REHASH_CHUNK_SLOTS,
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
    rehash_chunk_buckets,
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
    CUSTOMIZED) is carried verbatim.
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


# ---------------------------------------------------------------------------
# Checkpoint value layout
#
# Key / embedding / score / optimizer-state files are bare row-major blobs with
# no header, so the table's meta JSON is the only record of how to read them
# back. Embeddings and optimizer states are dumped at the table's own precision
# (``DynamicEmbTableState.emb_dtype``) rather than being upcast to fp32, which
# makes ``embedding_dtype`` and ``embedding_dim`` load-bearing rather than
# decorative. Checkpoints written before those keys existed are always fp32, and
# their dim is recoverable from the file size -- see
# :func:`resolve_checkpoint_value_layout`.
# ---------------------------------------------------------------------------

META_EMBEDDING_DTYPE = "embedding_dtype"
META_EMBEDDING_DIM = "embedding_dim"
META_OPT_STATE_DTYPE = "optim_state_dtype"


def dtype_to_meta_name(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``, the spelling checkpoint meta uses."""
    return str(dtype).split(".", 1)[-1]


def dtype_from_meta_name(name: Any, field_name: str) -> torch.dtype:
    """Inverse of :func:`dtype_to_meta_name`, rejecting anything not a dtype.

    Resolves against ``torch`` itself rather than a hand-kept table so any dtype
    a table can be built with round-trips, and a corrupt or hand-edited meta
    fails here instead of silently mis-striding the value file.
    """
    dtype = getattr(torch, name, None) if isinstance(name, str) else None
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"Checkpoint meta field {field_name!r} is {name!r}, which is not a "
            "torch dtype."
        )
    return dtype


def resolve_checkpoint_value_layout(
    meta_data: Dict[str, Any],
    embedding_file_path: str,
    num_keys: int,
    runtime_dim: int,
) -> Tuple[torch.dtype, torch.dtype, int]:
    """Decide how to read a value file: (embedding dtype, opt-state dtype, dim).

    Prefers what the dump recorded. For a checkpoint written before the meta
    carried a dtype, the historical answer is fp32 for both value files, and the
    row width is recovered from the value file's length -- exact, given the dtype
    and the key count. An empty shard is the one case the length cannot answer,
    so it defers to the runtime dim.

    Either way the result is cross-checked against the file size, which turns a
    wrong dtype assumption (or a truncated file) into an error here rather than
    a silently misparsed table.

    Raises:
        ValueError: the value file's size disagrees with the resolved layout, or
            a meta dtype field is not a torch dtype.
    """
    emb_dtype = (
        dtype_from_meta_name(meta_data[META_EMBEDDING_DTYPE], META_EMBEDDING_DTYPE)
        if META_EMBEDDING_DTYPE in meta_data
        else EMBEDDING_TYPE
    )
    opt_state_dtype = (
        dtype_from_meta_name(meta_data[META_OPT_STATE_DTYPE], META_OPT_STATE_DTYPE)
        if META_OPT_STATE_DTYPE in meta_data
        else OPT_STATE_TYPE
    )

    emb_bytes = os.path.getsize(embedding_file_path)
    ckpt_dim = meta_data.get(META_EMBEDDING_DIM, None)
    if ckpt_dim is None:
        if num_keys == 0:
            ckpt_dim = runtime_dim
        else:
            key_block = num_keys * emb_dtype.itemsize
            if emb_bytes % key_block != 0:
                raise ValueError(
                    f"Embedding file {embedding_file_path} size {emb_bytes} is not "
                    f"divisible by {key_block} (num_keys={num_keys}, dtype "
                    f"{emb_dtype}), so its row width cannot be recovered. This "
                    "checkpoint predates "
                    f"{META_EMBEDDING_DIM!r}/{META_EMBEDDING_DTYPE!r} in the meta "
                    "JSON, so it is assumed to be fp32."
                )
            ckpt_dim = emb_bytes // key_block
    ckpt_dim = int(ckpt_dim)

    expected_bytes = num_keys * ckpt_dim * emb_dtype.itemsize
    if emb_bytes != expected_bytes:
        raise ValueError(
            f"Embedding file {embedding_file_path} holds {emb_bytes} bytes but "
            f"{expected_bytes} were expected for {num_keys} keys of {ckpt_dim} "
            f"{emb_dtype} elements."
        )
    return emb_dtype, opt_state_dtype, ckpt_dim


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    """Little-endian raw bytes of *tensor*, for any dtype it can hold.

    Views as uint8 instead of going through ``numpy().tobytes()`` because numpy
    has no bfloat16, and value files now carry the table's own precision.
    """
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_from_bytes(
    buf: bytes, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Decode *buf* as a 1-D tensor of *dtype* on *device*.

    ``torch.frombuffer`` rather than ``np.frombuffer`` for the same bfloat16
    reason as :func:`_raw_bytes`. The ``bytearray`` copy makes the buffer writable
    so torch does not warn about aliasing read-only memory; the ``to(device)``
    would copy regardless.
    """
    if len(buf) == 0:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.frombuffer(bytearray(buf), dtype=dtype).to(device)


def get_score_policy(score_strategy):
    """Build the physical :class:`ScoreSpec` for a (validated) score strategy.

    *score_strategy* is either a single :class:`DynamicEmbScoreStrategy` or a
    supported compound tuple (currently only ``{TIMESTAMP, LFU}`` in either order).
    A compound strategy maps to the ``LruLfu`` policy whose physical layout is
    fixed at ``(timestamp, frequency)`` regardless of the user's tuple order; the
    user's ordering only affects checkpoint column order (see
    :func:`score_dump_permutation`). The no-eviction mode adds no score of its own:
    ``NO_EVICTION`` alone keeps none: its one word is a ``FLAG`` (1 while a key
    occupies the slot, 0 after an erase), which is not a score a user sets but
    what lets a full bucket give up an erased slot before a key in use. In a
    compound the score is the other strategy's.
    """
    score_strategy = get_score_part(score_strategy)
    if score_strategy is None:
        return ScoreSpec(name="no_eviction", policy=ScorePolicy.FLAG)
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
    elif score_strategy in (
        DynamicEmbScoreStrategy.CUSTOMIZED,
        DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED,
    ):
        # Stored the same way: what differs is only where a score comes from.
        return ScoreSpec(name="customized", policy=ScorePolicy.ASSIGN)
    elif score_strategy == DynamicEmbScoreStrategy.LFU:
        return ScoreSpec(name="frequency", policy=ScorePolicy.ACCUMULATE)
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


def _base_pointers(tables: List[ExtendableBuffer]) -> List[int]:
    """The base address of each value buffer of *tables*, as the kernels address them.

    A buffer keeps its address when it is extended (a device buffer maps new pages after
    the old ones, a host buffer commits pages inside the address range it reserved), so
    what a kernel queued earlier holds stays valid. See :func:`publish_table_ptrs`.
    """
    return [b.tensor().data_ptr() for b in tables]


def publish_table_ptrs(state: "DynamicEmbTableState") -> None:
    """Bring ``state.table_ptrs_dev`` up to date with the value buffers.

    The addresses are compared on the host: when none has changed, which is every growth
    of a buffer that never moves, nothing is uploaded. A buffer that did move (an
    extension built before buffers were address-stable) is published by a queued copy of
    the whole list on the current stream, the growth stream, inside the growth section.
    """
    ptrs = _base_pointers(state.tables)
    if ptrs == state.table_ptrs_host:
        return
    sf.copy_i64_into(state.table_ptrs_dev, ptrs)
    state.table_ptrs_host = ptrs


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
    # Per-table value buffer base pointers on ``device``, set on init and refreshed by
    # :func:`publish_table_ptrs` when a buffer moves.
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
    # What a state that can still grow decides its expansions from, without waiting
    # for the device (see :class:`SizeSnapshots`). ``None`` for a state that cannot
    # grow, which then has nothing to estimate.
    size_snapshots: Optional["SizeSnapshots"] = None
    # What orders a growth of this state against the streams that use it (see
    # :class:`GrowthOrder`); ``None`` for a state that cannot grow, which has
    # nothing to order.
    growth: Optional[GrowthOrder] = None
    # What ``table_ptrs_dev`` holds, as host integers: a growth compares the addresses
    # of the buffers with it and uploads only when one has changed.
    table_ptrs_host: List[int] = field(default_factory=list)
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
    # Evictions whose count has not been read back yet: [keys, table_ids, count,
    # ticket, stream] per entry (``stream`` is the ``cuda_stream`` handle of the
    # stream that made it), appended by ``_append_evicted_masked`` and folded into
    # the chunks above by ``_resolve_pending_evicted`` once ``count`` has landed on
    # the host (or, on a pop, by waiting for it). ``retain_ring`` is the pinned ring
    # those counts land in, allocated here so that no step allocates; only a table
    # that retains evicted keys has one. A copy of the state (``copy.deepcopy``,
    # pickle) gets a ring of its own, which does not know the tickets of the entries
    # still pending: pop before copying.
    evicted_pending: List[list] = field(default_factory=list)
    retain_ring: Optional[sf.DeferredRead] = None
    # Whether a step's miss path has to run (see ``sf.MissGate``). One per state:
    # it owns a ring of its own, so two states never share a ticket.
    miss_gate: Optional[sf.MissGate] = None
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

    def wait_for_growth(self) -> None:
        """Order the current stream after the last growth of this state, and register it
        as a user of the state (:class:`GrowthOrder`).

        A growth only queues its kernels on the stream that decided to grow and returns,
        so whoever reads or writes the key map or the value buffers from another stream
        calls this first. The forward, the backward and the evaluation of the module do,
        and so does every public method of the storages that reads the state. Nothing
        waits on the host, and a state that cannot grow does nothing.
        """
        if self.growth is not None:
            self.growth.wait_for_growth()


# The key map of a table that never evicts is kept at most this full, so that no
# bucket fills up; its value buffer then holds exactly that many rows, i.e. the
# key map is twice the size of the value buffer, at creation and after every
# expansion.
NO_EVICTION_LOAD_FACTOR = 0.5


def _host_reserve_factor() -> float:
    """``DYNAMICEMB_HOST_RESERVE_FACTOR``: how many times its ``max_capacity`` the value
    buffer of a host table that never evicts reserves address space for. Read when such
    a table is made, not when the package is imported, so that it can be set by the
    program that makes the tables."""
    value = os.environ.get("DYNAMICEMB_HOST_RESERVE_FACTOR", "8")
    try:
        factor = float(value)
    except ValueError:
        factor = 0.0
    # sync-ok: factor is a Python float, parsed from the environment
    if not (math.isfinite(factor) and factor >= 1.0):
        raise ValueError(
            "DYNAMICEMB_HOST_RESERVE_FACTOR must be a finite number of at least 1, "
            f"got {value!r}."
        )
    return factor


def _warn_if_exact_growth_is_set() -> None:
    """``DYNAMICEMB_EXACT_GROWTH=1`` made every expansion decision wait for the device
    sizes. No environment variable may make a step wait for the device, so it is gone;
    a script that still sets it keeps running, on the estimate."""
    if os.environ.get("DYNAMICEMB_EXACT_GROWTH", "0") == "1":
        warnings.warn(
            "DYNAMICEMB_EXACT_GROWTH was removed: a table grows from the size estimate "
            "and never makes a training step wait for the device. A checkpoint load "
            "still reads the exact sizes (once).",
            stacklevel=2,
        )


_warn_if_exact_growth_is_set()

# How much address space a host value buffer of a table that never evicts reserves when
# it is created, for it to grow in place: DYNAMICEMB_HOST_RESERVE_FACTOR times the
# table's max_capacity, and at least _HOST_RESERVE_FLOOR_BYTES. Address space only: it
# costs no memory, but it is not unlimited (every table of the process reserves its own,
# and a buffer made when the process has run out of it gets what is left, down to what it
# holds), which is why a table that evicts reserves exactly what its max_capacity can need.
_HOST_RESERVE_FLOOR_BYTES = 64 << 30


def _host_value_rows_reserve(
    opt: DynamicEmbTableOptions, rows_now: int, row_bytes: int, no_eviction: bool
) -> int:
    """The rows of address space the host value buffer of a table that can grow
    reserves (see :class:`HostExtendableBuffer`).

    A table that evicts never has more rows than its key map has slots, which
    :func:`get_expand_info` keeps to ``max_capacity`` (rounded up to a bucket, and a
    bucket more for a rounded target): that is exact. A table that never evicts is not
    capped by ``max_capacity`` on the key map, but its host value buffer is capped by
    what it reserves: ``DYNAMICEMB_HOST_RESERVE_FACTOR`` times it, and no less than
    64 GiB (at most the installed memory, which a pinned buffer cannot pass anyway).
    Growing past that is an error that names the knob, raised before anything is
    changed (:meth:`ExtendableBuffer.check_extend`).
    """
    bucket = opt.bucket_capacity
    if no_eviction or opt.max_capacity is None:
        planned = rows_now if opt.max_capacity is None else opt.max_capacity
        rows = max(
            int(_host_reserve_factor() * planned),
            -(-_HOST_RESERVE_FLOOR_BYTES // row_bytes),
        )
    else:
        rows = ((opt.max_capacity + bucket - 1) // bucket + 1) * bucket
    return max(rows_now, rows)


def create_table_state(
    options: List[DynamicEmbTableOptions],
    optimizer: BaseDynamicEmbeddingOptimizer,
    enable_overflow: bool = False,
    evicted_item_mode: EvictedItemMode = EvictedItemMode.DISCARD,
    use_rows: bool = False,
    use_ref_counter: bool = True,
) -> DynamicEmbTableState:
    """Build the key map and value buffers for a group of tables.

    ``use_rows`` stores a value row per key (see ``LinearBucketTable``) instead
    of using the hash slot as the row, which lets an expansion keep every value
    where it is. It is for tables that grow and keep their keys, so it cannot be
    combined with ``enable_overflow`` (a cache keeps slot == row). It is
    independent of the no-eviction mode, which keeps no row in its score.

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
        from dynamicemb.jit.score_jit import register_score_function

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

    key_index_map = get_scored_table(
        capacity=capacities,
        bucket_capacity=base_opt.bucket_capacity,
        key_type=base_opt.index_type,
        score_specs=[score_policy],
        device=device,
        enable_overflow=enable_overflow,
        score_fn_key=score_fn_key,
        use_rows=use_rows,
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
    can_still_grow = _tables_can_grow(options, no_eviction, key_index_map_caps)
    tables: List[ExtendableBuffer] = []
    for i, (cap, vd) in enumerate(zip(table_caps, value_dims)):
        total_cap = cap
        if enable_overflow:
            total_cap += overflow_caps_list[i]
        shape = (total_cap, vd)
        if on_host:
            # A host buffer grows in place, inside the address range it reserves now:
            # exactly what it has if it can never grow.
            reserve_rows = (
                _host_value_rows_reserve(
                    options[i], total_cap, vd * emb_dtype.itemsize, no_eviction
                )
                if can_still_grow
                else total_cap
            )
            tables.append(
                HostExtendableBuffer(
                    shape, emb_dtype, device, max_shape=(reserve_rows, vd)
                )
            )
        else:
            tables.append(DeviceExtendableBuffer(shape, emb_dtype, device))

    if use_rows:
        key_index_map.set_row_limits([t.shape[0] for t in tables])

    props = torch.cuda.get_device_properties(device_idx)
    threads_in_wave = (
        props.multi_processor_count * props.max_threads_per_multi_processor
    )
    table_ptrs = _base_pointers(tables)

    state = DynamicEmbTableState(
        options_list=options,
        num_tables=num_tables,
        device=device,
        score_policy=score_policy,
        incremental_score_name=incremental_score_name,
        evict_strategy=evict_strategy,
        key_index_map=key_index_map,
        capacity=capacity,
        tables=tables,
        table_ptrs_dev=sf.upload_i64(table_ptrs, device),
        table_ptrs_host=table_ptrs,
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
        # Its pinned ring is allocated here, once; a state that never grows (a cache,
        # a table at its max_capacity) gets no ring and no per-step work at all.
        size_snapshots=SizeSnapshots(num_tables, with_rows=no_eviction)
        if can_still_grow
        else None,
        # Likewise: only a state that can grow has streams to order its growth against.
        growth=GrowthOrder(device) if can_still_grow else None,
        evicted_item_mode=evicted_item_mode,
        miss_gate=sf.MissGate(),
        retain_ring=sf.DeferredRead((1,), torch.int64, slots=8)
        if EvictedItemMode.RETAIN_KEY in evicted_item_mode
        else None,
    )
    # The key map's bucket offsets, the row limits and the pointer table above are
    # uploaded by queued copies on this stream. A state that grows orders its users
    # after the last growth, but nothing orders the first use of a new state, on another
    # stream, after its own construction: one wait here, once per state, as the blocking
    # copies these replaced gave.
    # sync-ok: construction, once per state, never in a step
    torch.cuda.current_stream(device).synchronize()
    return state


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


def _tables_can_grow(
    options: Sequence[DynamicEmbTableOptions],
    no_eviction: bool,
    per_table_capacity: Sequence[int],
) -> bool:
    """Whether some table could still be expanded: the skips :func:`get_expand_info`
    makes for a table, less the one that depends on the batch (no new key)."""
    for opt, cap in zip(options, per_table_capacity):
        max_lf = NO_EVICTION_LOAD_FACTOR if no_eviction else opt.max_load_factor
        if max_lf <= 0:
            continue
        # A table that never evicts is not capped by max_capacity.
        if no_eviction or opt.max_capacity is None or cap < opt.max_capacity:
            return True
    return False


def can_grow(state: DynamicEmbTableState) -> bool:
    """Whether some table of ``state`` could still be expanded. A state that cannot
    (a cache, tables at their ``max_capacity``) has nothing to estimate."""
    return _tables_can_grow(
        state.options_list, state.no_eviction, state.key_index_map.per_table_capacity_
    )


class SizeSnapshots:
    """What a growable state decides its expansions from, without waiting for the device.

    The host needs two numbers to decide whether a batch makes a table grow: the keys
    each table holds and, for a table that never evicts, the value rows it has handed
    out. Both live on the device. Instead of reading them back at the moment of the
    decision, a snapshot of both is copied into a pinned ring slot as the state's
    inserts are enqueued, and the estimate is

        the newest snapshot that has landed
        + the bound of every batch issued after the inserts that snapshot saw.

    It is never too low, which is what lets a growth decision go without a confirming
    read: the snapshot is ordered, on the stream that performs the state's inserts,
    after every insert enqueued on that stream before it, and a later batch adds at
    most its bound to a table (its unique keys there). It can be too high -- a bound
    counts keys the table already has, and a snapshot that has not landed yet leaves
    its batches in the sum -- which makes a table grow earlier than an exact read
    would, by about the keys of the batches in flight.

    Where the snapshot is enqueued depends on where the state is inserted into. A
    cache's backing storage and an HBM-only storage are inserted into by the prefetch,
    so :func:`request_size_snapshot` is called there. A host or hybrid storage is
    inserted into by the forward, on another stream, and a batch is prefetched before
    it is forwarded: those batches are ``inflight`` until the forward calls
    ``note_forwarded``, and are left out of what a snapshot saw.

    Everything here is host bookkeeping on plain CPU tensors and the pinned ring,
    which is allocated in the constructor: a step allocates nothing and waits for
    nothing. All of it runs on the one thread that issues the batches.
    """

    def __init__(self, num_tables: int, with_rows: bool, slots: int = 4):
        self.T = num_tables
        self.with_rows = with_rows
        self.slots = slots
        # A snapshot is the keys per table, then the rows per table when it has them.
        self.ring = sf.DeferredRead(
            (2 * num_tables if with_rows else num_tables,), torch.int64, slots
        )
        # The sum of every bound given to ``expand_if_need``, and the part of it whose
        # inserts are still to come (a batch prefetched but not yet forwarded).
        self.issued = torch.zeros(num_tables, dtype=torch.int64)
        self.inflight = torch.zeros(num_tables, dtype=torch.int64)
        # The newest snapshot that has landed, and ``issued - inflight`` when it was
        # enqueued: the bounds of the batches it already counted.
        self.base_sizes = torch.zeros(num_tables, dtype=torch.int64)
        self.base_rows = torch.zeros(num_tables, dtype=torch.int64)
        self.base_issued = torch.zeros(num_tables, dtype=torch.int64)
        # (ring ticket, ``issued - inflight`` at enqueue), oldest first.
        self.pending = collections.deque()

    def enqueue(
        self, sizes_dev: torch.Tensor, rows_dev: Optional[torch.Tensor]
    ) -> None:
        """Start the copy of a snapshot. Call it on the stream that performs the
        state's inserts, after the inserts it should include. Never blocks; when every
        ring slot is still in flight the snapshot is dropped, which only leaves the
        base older (a looser estimate)."""
        packed = torch.cat([sizes_dev, rows_dev]) if self.with_rows else sizes_dev
        ticket = self.ring.enqueue(packed)
        if ticket is not None:
            self.pending.append((ticket, self.issued - self.inflight))

    def poll(self) -> None:
        """Adopt the newest snapshot that has landed, if any. Never blocks."""
        while self.pending and self.ring.ready(self.pending[0][0]):
            ticket, mark = self.pending.popleft()
            landed = self.ring.value(ticket)
            self.base_sizes.copy_(landed[: self.T])
            if self.with_rows:
                self.base_rows.copy_(landed[self.T :])
            self.base_issued.copy_(mark)
            self.ring.release(ticket)

    def estimate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(keys, rows)`` per table: an upper bound of what the tables hold once every
        batch issued so far has been inserted. ``rows`` means nothing for a state
        without them."""
        since = self.issued - self.base_issued
        return self.base_sizes + since, self.base_rows + since

    def rebase_exact(self, sizes: torch.Tensor, rows: Optional[torch.Tensor]) -> None:
        """Start over from numbers the host has just read from the device, blocking
        (a checkpoint load, a test): the snapshots still in flight are older than they
        are and are dropped.

        Inserts that were enqueued on another stream than the one that read the
        numbers may be missing from them; that is the exposure the blocking read has
        always had."""
        for ticket, _ in self.pending:
            self.ring.release(ticket)
        self.pending.clear()
        self.base_sizes.copy_(sizes)
        if rows is None:
            self.base_rows.zero_()
        else:
            self.base_rows.copy_(rows)
        self.base_issued.copy_(self.issued - self.inflight)

    def __reduce__(self):
        # A copy of the state (copy.deepcopy, pickle) gets a ring of its own, so
        # nothing is in flight in it. What was adopted so far still bounds its tables.
        return (
            _restore_size_snapshots,
            (
                self.T,
                self.with_rows,
                self.slots,
                self.issued,
                self.inflight,
                self.base_sizes,
                self.base_rows,
                self.base_issued,
            ),
        )


def _restore_size_snapshots(
    num_tables, with_rows, slots, issued, inflight, base_sizes, base_rows, base_issued
) -> SizeSnapshots:
    snapshots = SizeSnapshots(num_tables, with_rows, slots)
    snapshots.issued.copy_(issued)
    snapshots.inflight.copy_(inflight)
    snapshots.base_sizes.copy_(base_sizes)
    snapshots.base_rows.copy_(base_rows)
    snapshots.base_issued.copy_(base_issued)
    return snapshots


def request_size_snapshot(state: DynamicEmbTableState) -> None:
    """Start reading the sizes of ``state`` back, to decide expansions from later
    (see :class:`SizeSnapshots`). A few launches and an asynchronous copy, only for a
    state that can still grow. Call it on the stream that performs the state's
    inserts, after the ones it should count."""
    snapshots = state.size_snapshots
    if snapshots is None or not can_grow(state):
        return
    snapshots.enqueue(
        collect_table_sizes_to_device(state),
        collect_row_counts_to_device(state) if state.no_eviction else None,
    )


def _note_forwarded(state: DynamicEmbTableState, bound: torch.Tensor) -> None:
    """The forward of a batch whose keys ``bound`` bounded has inserted them, on the
    current stream: it is no longer in flight, and a snapshot taken now counts it."""
    snapshots = state.size_snapshots
    if snapshots is None:
        return
    snapshots.inflight -= bound
    request_size_snapshot(state)


def read_table_sizes_now(
    state: DynamicEmbTableState,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The keys every table holds and, for a table that never evicts, the rows it has
    handed out, read from the device NOW: ``(sizes, rows)`` as CPU int64 tensors, ``rows``
    ``None`` for a state that has none.

    One blocking read (:func:`dynamicemb.sync_free.read_now`, a single pinned transfer
    of both vectors), for a checkpoint load, which sizes the tables for the keys it is
    about to insert and is not on the training path. A step decides from the estimate
    (:class:`SizeSnapshots`) and never comes here.
    """
    on_device = [collect_table_sizes_to_device(state)]
    if state.no_eviction:
        on_device.append(collect_row_counts_to_device(state))
    on_host = sf.read_now(*on_device)
    return on_host[0], (on_host[1] if state.no_eviction else None)


def collect_table_sizes_for_state(
    state: DynamicEmbTableState, non_blocking: bool = True
) -> None:
    """Copy device table sizes into ``state.estimated_table_sizes`` when flag is set.

    A blocking copy (one :func:`read_table_sizes_now`) gives the exact numbers, which
    the size estimate of the state restarts from."""
    if not state.collect_table_sizes_flag:
        return
    if non_blocking:
        state.estimated_table_sizes.copy_(
            collect_table_sizes_to_device(state), non_blocking=True
        )
        if state.estimated_row_counts is not None:
            state.estimated_row_counts.copy_(
                collect_row_counts_to_device(state), non_blocking=True
            )
        return
    sizes, rows = read_table_sizes_now(state)
    state.estimated_table_sizes.copy_(sizes)
    if rows is not None and state.estimated_row_counts is not None:
        state.estimated_row_counts.copy_(rows)
    if state.size_snapshots is not None:
        state.size_snapshots.rebase_exact(
            state.estimated_table_sizes, state.estimated_row_counts
        )


# ---------------------------------------------------------------------------
# Storage expansion (expand before insert when needed)
# Used in: prefetch HBM direct, cache write-back, generic forward, HybridStorage.load
# ---------------------------------------------------------------------------


def _growth_section(state: DynamicEmbTableState, stream: torch.cuda.Stream):
    """The section one growth of ``state`` runs in, on ``stream`` (see
    :class:`GrowthOrder`): the other streams that use the state are waited for, on the
    device, before it, and wait for it after. Nothing for a state that cannot grow."""
    if state.growth is None:
        return contextlib.nullcontext()
    return state.growth.section(stream)


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
    such table, insert its old key column into the new key_index_map a chunk at a time
    (:meth:`LinearBucketTable.rehash_table_from`: the empty, reclaimed and locked slots hold
    reserved keys, which insert nothing, so nothing is compacted and no count is read),
    carrying the value rows, reference counters and score words of the keys along. A table
    that stores no row moves its values by slot: every chunk's values are loaded before
    any is stored, so that a store never lands on a row that was not read yet. For tables
    that do not expand, key_index_map for that table is copied directly from the old
    key_index_map (copy_table_from). Updates key_index_map, capacity (expansion does not
    change the key set). Mutates state. The kernels run on the current stream inside a
    growth section (:class:`GrowthOrder`); the host reads nothing back.

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
    new_caps = []
    for i in range(state.num_tables):
        # sync-ok: tables_to_expand is a list of Python bools, decided on the host
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
    # Built before the section: allocating and filling the new key map does not depend
    # on what the other streams queued, so it overlaps their work.
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
        # And whether it counts references at all.
        use_ref_counter=state.key_index_map.use_ref_counter_,
    )
    # What each value buffer has to grow by, and whether every one of them can: checked
    # before anything is changed, so that a buffer that cannot grow (a host buffer past
    # the address space it reserved) does not leave the others grown.
    extensions = _value_extensions(
        state, tables_to_expand, target_capacities, key_caps, new_key_index_map
    )
    _check_value_buffers(state, extensions)
    # The device memory the values of a table that stores no row are moved through (see
    # _ValueMove) is allocated before anything changes, too.
    moves = _stage_value_moves(state, new_key_index_map, tables_to_expand, key_caps)
    stream = torch.cuda.current_stream(device)
    with _growth_section(state, stream):
        _rehash_into(
            state, new_key_index_map, tables_to_expand, extensions, moves, stream
        )


# The device memory a value move stages values in, at most, while a table grows by a whole
# multiple of its size (see _ValueMove): the values of one chunk of the rehash.
REHASH_STAGING_BYTES = 256 << 20


def _allocate_staging(device: torch.device, nbytes: int) -> torch.Tensor:
    """The device memory a value move stages values in: ``nbytes`` of one flat tensor."""
    return torch.empty(nbytes, dtype=torch.uint8, device=device)


class _ValueMove:
    """Moves the values of a table whose value row is its hash slot (the host tier of a
    hybrid storage) from the slots of the key map a rehash replaces to the slots of the
    one that replaces it: the ``on_chunk`` of :meth:`LinearBucketTable.rehash_table_from`.

    A value stays where it is, so it has to be copied when its key gets another slot, and
    a store must not land on a row that has not been read yet. Every chunk of the rehash
    loads the values of its keys from their old slots into a staging tensor (a lane that
    carries no key has slot ``-1``, which the load and the store skip). Two cases, told
    apart on the host by the two sizes (:func:`~dynamicemb.rehash_plan.chunks_are_independent`):

    * **The new table is a whole multiple of the old one**, which is every growth that
      doubles. A key of old bucket ``b`` lands in bucket ``b + j * old_buckets`` of the new
      table: its own bucket, whose rows the chunk has just read, or a bucket past the end
      of the old table, whose rows hold nothing. So a chunk is stored before the next one
      is read, and the staging holds one chunk, at most ``REHASH_STAGING_BYTES``, however
      big the table. A device assert says that every key did land that way.
    * **Any other size** (the last growth of a table, which ``max_capacity`` clamps) sends
      the keys of a bucket to unrelated buckets: every chunk is loaded before any is
      stored, and the staging holds a row for every slot of the OLD table,
      ``old slots * value_dim * itemsize`` bytes of device memory, which is more than the
      values of the keys it holds.

    Either way the staging is allocated before the growth changes anything
    (:func:`_stage_value_moves`), so that a device that cannot give it fails with a message
    and the table as it was.
    """

    def __init__(
        self,
        state: DynamicEmbTableState,
        table_id: int,
        old_slots: int,
        new_slots: int,
        bucket_capacity: int,
    ):
        self.state = state
        self.table_id = table_id
        self.bucket_capacity = bucket_capacity
        self.old_buckets = old_slots // bucket_capacity
        value_dim = state.table_value_dims_cpu[table_id]
        self.row_bytes = value_dim * state.emb_dtype.itemsize
        self.chunk_by_chunk = chunks_are_independent(old_slots, new_slots)
        if self.chunk_by_chunk:
            self.chunk_slots = max(
                bucket_capacity,
                min(REHASH_CHUNK_SLOTS, REHASH_STAGING_BYTES // self.row_bytes),
            )
            self.staged_rows = (
                rehash_chunk_buckets(
                    self.old_buckets, bucket_capacity, self.chunk_slots
                )
                * bucket_capacity
            )
        else:
            self.chunk_slots = REHASH_CHUNK_SLOTS
            self.staged_rows = old_slots
        self.staged_bytes = self.staged_rows * self.row_bytes
        self.staging: Optional[torch.Tensor] = None
        self.offset = 0
        self.pending: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def bind(self, area: torch.Tensor) -> None:
        """Use the first ``staged_bytes`` of ``area`` (flat, uint8) as the staging."""
        value_dim = self.state.table_value_dims_cpu[self.table_id]
        self.staging = (
            area[: self.staged_bytes]
            .view(self.state.emb_dtype)
            .view(self.staged_rows, value_dim)
        )

    def __call__(self, src_slots: torch.Tensor, dst_slots: torch.Tensor) -> None:
        staging = self.staging
        # sync-ok: a host object (set by bind), not a tensor
        assert (
            staging is not None
        ), "the staging of a value move is bound before it runs"
        lanes = src_slots.numel()
        if self.chunk_by_chunk:
            values = staging[:lanes]
        else:
            values = staging[self.offset : self.offset + lanes]
            self.offset += lanes
        load_from_flat_single_table(self.state, src_slots, self.table_id, output=values)
        if not self.chunk_by_chunk:
            self.pending.append((dst_slots, values))
            return
        # Every key must have landed in the bucket it came from or in one the growth added
        # (see the class): then the store below only touches rows this chunk has read, or
        # rows that hold nothing.
        torch._assert_async(
            (
                (dst_slots < 0)
                | lands_in_the_bucket_of_origin_or_a_new_one(
                    src_slots, dst_slots, self.bucket_capacity, self.old_buckets
                )
            ).all()
        )
        store_to_flat_single_table(self.state, dst_slots, self.table_id, values)

    def finish(self) -> None:
        """Store what every chunk loaded, once the last one has (nothing, when each chunk
        was stored as it was loaded)."""
        pending, self.pending = self.pending, []
        for dst_slots, values in pending:
            store_to_flat_single_table(self.state, dst_slots, self.table_id, values)
        self.offset = 0


def _stage_value_moves(
    state: DynamicEmbTableState,
    new_key_index_map,
    tables_to_expand: List[bool],
    key_caps: List[int],
) -> Dict[int, _ValueMove]:
    """The value moves of a growth of a table that stores no row, by table, with the
    staging they need allocated, once and as large as the largest (the tables are moved
    one after the other). Allocated before the growth changes anything: a device that
    cannot give it raises :class:`torch.cuda.OutOfMemoryError` naming the table and the
    size, and the tables are as they were."""
    moves: Dict[int, _ValueMove] = {}
    if new_key_index_map.use_rows_:
        return moves
    for i in range(state.num_tables):
        # sync-ok: tables_to_expand is a list of Python bools, decided on the host
        if tables_to_expand[i]:
            moves[i] = _ValueMove(
                state,
                i,
                key_caps[i],
                new_key_index_map.per_table_capacity_[i],
                new_key_index_map.bucket_capacity_,
            )
    # sync-ok: a dict of host objects
    if not moves:
        return moves
    largest = max(moves, key=lambda i: moves[i].staged_bytes)
    nbytes = moves[largest].staged_bytes
    try:
        area = _allocate_staging(state.device, nbytes)
    except torch.cuda.OutOfMemoryError as exc:
        raise torch.cuda.OutOfMemoryError(
            f"growing table {largest} moves its values through device memory, because "
            "its value row is its hash slot: it needs "
            f"{moves[largest].staged_rows} rows of {moves[largest].row_bytes} bytes "
            f"({nbytes} bytes) at once, and the device cannot give them. Nothing was "
            "changed. Free device memory, or give the table a max_capacity that is its "
            "init_capacity times a power of two: a growth to a whole multiple of the "
            "table's size stages one chunk at a time, the last one, which max_capacity "
            "clamps, stages the whole old table."
        ) from exc
    for move in moves.values():
        move.bind(area)
    return moves


def _value_extensions(
    state: DynamicEmbTableState,
    tables_to_expand: List[bool],
    target_capacities: Optional[List[int]],
    key_caps: List[int],
    new_key_index_map,
) -> List[Tuple[int, Tuple[int, ...]]]:
    """The value buffers a rehash extends and by how much: ``(table, shape to add)``.

    A table that never evicts keeps its value buffer out of it, which follows the rows
    (:func:`_grow_values_only`). Host arithmetic only.
    """
    extensions: List[Tuple[int, Tuple[int, ...]]] = []
    if state.no_eviction:
        return extensions
    for i in range(state.num_tables):
        # sync-ok: tables_to_expand is a list of Python bools, decided on the host
        if not tables_to_expand[i]:
            continue
        if (
            target_capacities is not None
            and i < len(target_capacities)
            and target_capacities[i] >= 0
        ):
            # Hash buckets can round the requested target up. Grow values
            # by the actual key-map capacity change, including that padding.
            add_rows = new_key_index_map.per_table_capacity_[i] - key_caps[i]
            if add_rows > 0:
                extensions.append((i, (add_rows, state.table_value_dims_cpu[i])))
        else:
            extensions.append((i, state.tables[i].shape))
    return extensions


def _check_value_buffers(
    state: DynamicEmbTableState, extensions: List[Tuple[int, Tuple[int, ...]]]
) -> None:
    """Raise, with every buffer as it was, if one of ``extensions`` cannot be made
    (:meth:`ExtendableBuffer.check_extend`)."""
    for i, shape in extensions:
        state.tables[i].check_extend(shape)


def _rehash_into(
    state: DynamicEmbTableState,
    new_key_index_map,
    tables_to_expand: List[bool],
    extensions: List[Tuple[int, Tuple[int, ...]]],
    moves: Dict[int, _ValueMove],
    stream: torch.cuda.Stream,
) -> None:
    """The part of :func:`_expand_tables_impl` that runs inside the growth section, on
    ``stream``: extend the value buffers by ``extensions``, move every table into
    ``new_key_index_map`` (a table that stores no row moves its values by ``moves``) and
    make it the state's."""
    for i, shape in extensions:
        state.tables[i].extend(shape)
    publish_table_ptrs(state)

    old_key_index_map = state.key_index_map
    for table_id in range(state.num_tables):
        # sync-ok: tables_to_expand is a list of Python bools, decided on the host
        if not tables_to_expand[table_id]:
            new_key_index_map.copy_table_from(old_key_index_map, table_id)
            continue
        move = moves.get(table_id)
        # The padded rehash: no count is read, the table's own key column is the input.
        new_key_index_map.rehash_table_from(
            old_key_index_map,
            table_id,
            on_chunk=move,
            chunk_slots=REHASH_CHUNK_SLOTS if move is None else move.chunk_slots,
        )
        if new_key_index_map.use_rows_:
            # Reclaimed (erased) slots still own their rows; keep them too.
            new_key_index_map.carry_rows_from(old_key_index_map, table_id)
        if move is not None:
            move.finish()

    state.key_index_map = new_key_index_map
    state.key_map_generation += 1
    state.capacity = new_key_index_map.capacity()
    if new_key_index_map.use_rows_:
        new_key_index_map.set_row_limits([t.shape[0] for t in state.tables])
    # The replaced key map goes back to the allocator when this function returns, which
    # lets the stream it was allocated on take its blocks again, while the rehash that
    # reads it may not have run yet on ``stream``: the blocks wait for ``stream``.
    old_key_index_map.record_stream(stream)


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


def _value_growth(
    state: DynamicEmbTableState, need_rows: List[int], key_capacities: List[int]
) -> List[Tuple[int, Tuple[int, ...]]]:
    """What the value buffers of a table that never evicts grow by, ``(table, shape to
    add)``, for ``need_rows`` rows handed out and key maps of ``key_capacities`` slots (a
    slot owns one row, so a buffer never has more rows than that). Host arithmetic."""
    extensions: List[Tuple[int, Tuple[int, ...]]] = []
    for i in range(state.num_tables):
        rows = state.tables[i].shape[0]
        target = _value_rows_target(rows, need_rows[i], key_capacities[i])
        if target > rows:
            extensions.append((i, (target - rows, state.table_value_dims_cpu[i])))
    return extensions


def _check_values_can_grow(
    state: DynamicEmbTableState,
    expand: List[bool],
    targets: List[int],
    need_rows: List[int],
) -> None:
    """Before the key map of a decision that grows both it and the value buffers is
    replaced: raise, with nothing changed, if a value buffer cannot make the growth the
    decision will ask of it once the key map has its new size (``expand`` and ``targets``
    are :func:`get_expand_info`'s). The check of :func:`_grow_values_only` that follows
    the key map's growth is then the same one and passes."""
    key_capacities = [
        # sync-ok: expand is a list of Python bools, decided on the host
        targets[i] if expand[i] else state.key_index_map.per_table_capacity_[i]
        for i in range(state.num_tables)
    ]
    _check_value_buffers(state, _value_growth(state, need_rows, key_capacities))


def _grow_values_only(state: DynamicEmbTableState, need_rows: List[int]) -> None:
    """Grow the value buffer of a table that never evicts, leaving its key map.

    The key map is sized by the keys the table holds; the value buffer by the rows
    it has handed out, which can be more (an erased key's slot keeps its row until
    another key takes the slot) but never more than the key map's slots.
    """
    extensions = _value_growth(
        state, need_rows, state.key_index_map.per_table_capacity_
    )
    # sync-ok: a list of host tuples
    if not extensions:
        return
    # Every buffer that has to grow can, or none does (see _check_value_buffers).
    _check_value_buffers(state, extensions)
    with _growth_section(state, torch.cuda.current_stream(state.device)):
        for i, shape in extensions:
            state.tables[i].extend(shape)
        publish_table_ptrs(state)
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
                target_capacities[table_id] = min(
                    max_cap,
                    max(
                        cap * 2,
                        align_to_table_size(
                            math.ceil(new_total / max_lf), opt.bucket_capacity
                        ),
                    ),
                )
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
        keys, rows = read_table_sizes_now(state)
        keys_est.copy_(keys)
        rows_est.copy_(rows)
        results, targets = get_expand_info(state, keys, unique_size_per_table)
        need_rows = (rows + unique_size_per_table).tolist()
        # A value buffer that cannot make the growth is found before the key map is replaced.
        _check_values_can_grow(state, results, targets, need_rows)
        # The key map and the value buffers are one growth.
        with _growth_section(state, torch.cuda.current_stream(state.device)):
            if any(results):
                _expand_tables_impl(state, results, targets)
            # The value buffers follow the rows, whether or not the key map grew.
            _grow_values_only(state, need_rows)
    keys_est.add_(unique_size_per_table)
    rows_est.add_(unique_size_per_table)


def _expand_if_need_no_eviction_estimated(
    state: DynamicEmbTableState,
    keys_est: torch.Tensor,
    rows_est: torch.Tensor,
    unique_size_per_table: torch.Tensor,
) -> None:
    """:func:`_expand_if_need_no_eviction` from the size estimate of the state
    (:class:`SizeSnapshots`), with no confirming read of the device.

    ``keys_est`` and ``rows_est`` never fall below what the table holds, so a growth
    that is needed is never decided late; the device assert that a row is below the
    table's row limit stays as the safety net, and relies on that. They can be too
    high, and then the table grows earlier than it had to.
    """
    results, targets = get_expand_info(state, keys_est, unique_size_per_table)
    # sync-ok: the estimates are CPU tensors, the host's own numbers
    need_rows = (rows_est + unique_size_per_table).tolist()
    # See _expand_if_need_no_eviction: a buffer that has a row for every slot of the
    # key map has nothing more to grow to, so the rows it is asked for are no pressure.
    rows_short = any(
        need_rows[i] > state.tables[i].shape[0]
        and state.tables[i].shape[0] < state.key_index_map.per_table_capacity_[i]
        for i in range(state.num_tables)
    )
    if any(results) or rows_short:
        # A value buffer that cannot make the growth is found before the key map is replaced.
        _check_values_can_grow(state, results, targets, need_rows)
        # The key map and the value buffers are one growth.
        with _growth_section(state, torch.cuda.current_stream(state.device)):
            if any(results):
                _expand_tables_impl(state, results, targets)
            # The value buffers follow the rows, whether or not the key map grew.
            _grow_values_only(state, need_rows)


def _expand_if_need_exact(
    state: DynamicEmbTableState,
    unique_size_per_table: torch.Tensor,
) -> None:
    """Accumulate per-table unique counts, optionally collect size and expand.

    unique_size_per_table is the CPU tensor from segmented_unique (shape
    (num_tables,) with per-table unique counts).
    When a second opinion on table sizes is needed, calls
    :func:`collect_table_sizes_for_state` with ``non_blocking=False``: the decision
    waits for the device (one :func:`read_table_sizes_now`). This is what
    :func:`expand_if_need_impl` does for a checkpoint load.
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


def expand_if_need_impl(
    state: DynamicEmbTableState,
    unique_size_per_table: torch.Tensor,
    *,
    exact: bool = False,
    inserts_in_forward: bool = False,
) -> None:
    """Expand the tables of ``state`` that a batch's keys would overfill.

    ``unique_size_per_table`` (CPU, one int64 per table) is an upper bound of the
    keys the batch can add to each table -- its unique keys there, or any number that
    cannot be lower. The decision comes from the size estimate of the state
    (:class:`SizeSnapshots`): it never waits for the device, and never comes too
    late; it can come earlier than an exact read would have put it, by about the keys
    of the batches that are in flight. The expansion does not wait for the device
    either: its kernels are queued on the current stream, ordered against the other
    streams that use the table by :class:`GrowthOrder`.

    ``exact`` decides from the device sizes instead, waiting for them (one read,
    :func:`read_table_sizes_now`) when an estimate says a table might grow. A checkpoint
    load asks for it, which is a call that returns after host I/O anyway.

    ``inserts_in_forward`` says the batch's keys are inserted by its forward, not by
    this call's caller: until the forward reports back (:meth:`note_forwarded` of the
    storage), the batch is in flight.
    """
    assert (
        unique_size_per_table.device.type == "cpu"
    ), "unique_size_per_table must be a CPU tensor"
    snapshots = state.size_snapshots
    if exact:
        # The exact path used to be able to expand from its running sum, without
        # confirming, once an asynchronous copy at the start of every prefetch had
        # refreshed it (the flag below). Nothing refreshes it in the background any
        # more, so it confirms against the device whenever the sum says a table might
        # grow.
        state.collect_table_sizes_flag = False
        _expand_if_need_exact(state, unique_size_per_table)
        if snapshots is not None:
            # The batch still counts for the estimate. Where the exact path reads the
            # device sizes it restarts the estimate from them, through
            # collect_table_sizes_for_state; its own running sum is no tighter than
            # the estimate and is not taken over.
            snapshots.issued += unique_size_per_table
            if inserts_in_forward:
                snapshots.inflight += unique_size_per_table
        return
    if snapshots is None:
        # No table can grow: nothing to estimate and nothing to read.
        return
    snapshots.poll()
    keys_est, rows_est = snapshots.estimate()
    if state.no_eviction:
        _expand_if_need_no_eviction_estimated(
            state, keys_est, rows_est, unique_size_per_table
        )
    else:
        results, targets = get_expand_info(state, keys_est, unique_size_per_table)
        # From the over-estimate: never late, possibly early by the batches in flight.
        if any(results):
            _expand_tables_impl(state, results, targets)
    snapshots.issued += unique_size_per_table
    if inserts_in_forward:
        snapshots.inflight += unique_size_per_table
    # What the attributes mean everywhere else: the estimate, this batch included.
    if state.estimated_table_sizes is not None:
        state.estimated_table_sizes.copy_(keys_est + unique_size_per_table)
    if state.estimated_row_counts is not None:
        state.estimated_row_counts.copy_(rows_est + unique_size_per_table)


# ---------------------------------------------------------------------------
# Free functions operating on DynamicEmbTableState
# ---------------------------------------------------------------------------


def _row_differs_from_slot(state: DynamicEmbTableState) -> bool:
    """Whether a key's value row is not its hash slot, i.e. the table stores a row
    per key (``use_rows``). A dumped ``slot_index`` then has to carry both."""
    return getattr(state.key_index_map, "use_rows_", False)


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


def _fresh_score_block(
    state: DynamicEmbTableState,
    table_id: int,
    n: int,
    timestamp: int,
    insert_score: int,
) -> torch.Tensor:
    """``[N, num_scores]`` physical score block for keys written by replay.

    Used when the caller did not ask for ``ReplayContent.SCORE``. The delta does
    carry the source's scores, but they were not loaded, so a replayed key is
    scored as if it had just been inserted here instead: recency words get
    *timestamp*, every other word gets *insert_score*. The replica then ranks its
    restored keys by when it received them rather than by how the source ranked
    them -- acceptable because the score only orders future evictions, and a
    replica that evicts on its own schedule still holds correct embeddings.
    Ask for ``SCORE`` to inherit the source's ranking instead.

    *insert_score* is passed in rather than read off ``state.score``: that field
    is populated by the forward pass, and replay is the one write path that can
    run before a target has ever done a forward. The caller owns the table's
    score bookkeeping and always has a value. *n* is the number of keys.
    """
    device = state.device
    num_scores = state.key_index_map.num_scores_
    # Zeros, not ``empty``: the loop below fills one column per configured
    # strategy, while the width comes from the score policy. The two agree for
    # every strategy there is, but they are derived from different places, so a
    # future divergence should leave a quiet zero rather than whatever the
    # allocator handed back.
    block = torch.zeros((n, num_scores), device=device, dtype=SCORE_TYPE)
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
    the same ``optimizer.states_for_checkpoint`` ``_dump_table``
    applies -- so a delta and a checkpoint describe a row identically, and the
    padding is not paid for on every dump. Replay expands it back with
    ``optimizer.states_from_checkpoint``.

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
        state.optimizer.states_for_checkpoint(opt, emb_dim).contiguous(),
    )


def load_from_flat(
    state: DynamicEmbTableState,
    indices: torch.Tensor,
    table_ids: torch.Tensor,
    copy_mode: CopyMode,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather the rows ``indices`` of every table into one ``[N, dim]`` buffer.

    ``dim`` is ``state.emb_dim`` for ``CopyMode.EMBEDDING`` and ``state.value_dim``
    otherwise. A row whose index is negative is skipped: its output row is left
    as it was, uninitialised in a fresh buffer.

    ``output`` is an existing ``[N, dim]`` buffer of ``state.emb_dtype`` to load
    into instead of a fresh one. Only the rows with a non-negative index are
    written (the embedding at ``[0, edim_t)`` and, for ``CopyMode.VALUE``, the
    optimizer state at ``[max_emb_dim, ...)``), so a caller can fill the buffer
    with defaults first, load what the table has over them, and have only the
    rows the table lacks keep the defaults. It is the buffer that is returned.
    """
    # The key map hands rows back on the host after a dump.
    indices = indices.to(device=state.device, dtype=torch.int64)
    N = indices.numel()
    if copy_mode == CopyMode.EMBEDDING:
        max_dim = state.emb_dim
        _load = _load_emb
    else:
        max_dim = state.value_dim
        _load = _load_value
    if output is None:
        output = torch.empty(N, max_dim, dtype=state.emb_dtype, device=state.device)
    else:
        assert tuple(output.shape) == (N, max_dim), (
            f"load_from_flat: output has shape {tuple(output.shape)}, "
            f"expected {(N, max_dim)}"
        )
        assert (
            output.dtype == state.emb_dtype
        ), f"load_from_flat: output is {output.dtype}, the table holds {state.emb_dtype}"
        # The kernels address a row by ``stride(0)`` and its columns contiguously.
        assert (
            max_dim == 1 or output.stride(1) == 1
        ), "load_from_flat: the columns of output must be contiguous"
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
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Load full values for a single table. Returns compact [N, value_dim_t].

    ``output`` is an existing ``[N, value_dim_t]`` buffer of ``state.emb_dtype`` (whose rows
    are contiguous) to load into instead of a fresh one; a row whose index is negative is
    skipped and keeps what it held. It is the buffer that is returned.
    """
    indices = indices.to(device=state.device, dtype=torch.int64)
    N = indices.numel()
    vdim = state.table_value_dims_cpu[table_id]
    if output is None:
        output = torch.empty(N, vdim, dtype=state.emb_dtype, device=state.device)
    else:
        assert tuple(output.shape) == (N, vdim), (
            f"load_from_flat_single_table: output has shape {tuple(output.shape)}, "
            f"expected {(N, vdim)}"
        )
        assert output.dtype == state.emb_dtype and output.is_contiguous(), (
            "load_from_flat_single_table: output must be a contiguous buffer of "
            f"{state.emb_dtype}"
        )
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
    indices = indices.to(device=state.device, dtype=torch.int64)
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


def score_source(
    state: DynamicEmbTableState,
    frequency: Optional[torch.Tensor] = None,
    key_scores: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """The per-key values a table's score is made from, or ``None`` if it has none.

    The two inputs a key can carry are separate channels -- ``frequency``, how
    often the batch counted it, and ``key_scores``, what the input scored it
    with -- and this is the one place that says which of them a table takes: an
    LFU table the frequency, a CUSTOMIZED table the score of the input. Every
    path that hands a score to the key map (the find, the insert) asks here, so a
    table that takes both one day has one rule to change.

    ``None`` means "the table's own default" -- ``get_find_score_arg`` and
    ``get_insert_score_arg`` supply it.
    """
    if state.evict_strategy == EvictStrategy.KLfu:
        return frequency
    if state.evict_strategy == EvictStrategy.KCustomized:
        return key_scores
    return None


def get_find_score_arg(
    state: DynamicEmbTableState,
    num_keys: int,
    device: torch.device,
    score_input: Optional[torch.Tensor] = None,
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
      - CUSTOMIZED / STEP: ASSIGN with ``state.score``; a CUSTOMIZED table that
        takes a score per key is given those scores instead, one per key, and
        assigns each key its own.

    ``score_input`` is what :func:`score_source` chose for the table -- the
    frequency of an LFU table, the scores of the input for a CUSTOMIZED one --
    so this takes one tensor, whichever it is.
    """
    # A table that keeps no score (NO_EVICTION alone) has nothing to update.
    if state.score_policy.policy == ScorePolicy.FLAG:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    if const_lookup or not state.training:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    # LFU: use provided frequency for ACCUMULATE; must have length num_keys for lookup.
    if state.evict_strategy == EvictStrategy.KLfu:
        if score_input is not None and score_input.numel() == num_keys:
            scores = score_input.contiguous()
        else:
            # Fallback: each key counts as 1 so score accumulates by 1 per lookup
            scores = torch.ones(num_keys, device=device, dtype=torch.long)
    elif state.evict_strategy == EvictStrategy.KCustomized:
        if score_input is not None:
            assert score_input.numel() == num_keys
            scores = score_input.contiguous()
        else:
            scores = torch.empty(num_keys, device=device, dtype=torch.long)
            scores.fill_(state.score)
    else:
        scores = None

    return ScoreArg(
        name=state.score_policy.name,
        value=scores,
        policy=state.score_policy.policy,
    )


def get_insert_score_arg(
    state: DynamicEmbTableState,
    num_keys: int,
    device: torch.device,
    scores: Optional[torch.Tensor] = None,
    preserve_existing: bool = False,
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
    """
    if preserve_existing:
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.CONST
        )

    if state.score_policy.policy == ScorePolicy.FLAG:
        # NO_EVICTION alone keeps no score a user could set, but a new key still
        # sets its flag, so that it ranks above the slots an erase left.
        return ScoreArg(
            name=state.score_policy.name, value=None, policy=ScorePolicy.FLAG
        )

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
    score_input: Optional[torch.Tensor] = None,
    *,
    const_lookup: bool = False,
    return_slots: bool = False,
    compact: bool = True,
) -> Tuple:
    """Key-only find: lookup in hash table, return missing info + indices.

    The indices are value rows for a table that stores them, hash slots otherwise.
    With ``return_slots`` the hash slots follow as a ninth element: the reference
    counters are keyed by slot.

    ``score_input`` is the one tensor a table's score is made from, as
    :func:`score_source` chose it (see :func:`get_find_score_arg`).

    With ``compact=False`` the call is the lookup and nothing else: the misses are
    not compacted, so no count is read back and the host does not wait for the
    device. The first four elements are then ``None`` and the fifth is the
    full-length score tensor the lookup used (``None`` for an LRU or a const
    lookup), so that a later insert of the same keys can reuse it. A caller that
    takes this form works with masks over arrays of the length of
    ``unique_keys`` (``~founds`` is the misses); a key it must not touch is the
    reserved all-ones key, which the lookup reports as not found. An empty batch
    returns the same empty result with either ``compact``.
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

    if not compact:
        return (
            None,
            None,
            None,
            None,
            score_arg.value,
            founds,
            score_out,
            indices,
            *((slots,) * return_slots),
        )

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
    num_evicted: Union[int, torch.Tensor],
) -> None:
    """Append this insert's evicted (key, table_id) to the state's retain chunks.

    ``evicted_keys`` / ``evicted_table_ids`` are sized at the batch upper bound;
    only the first ``num_evicted`` entries are valid. No de-duplication here --
    that happens on pop.

    ``num_evicted`` on the host (an int or a CPU tensor) is exact: slice + clone,
    freeing the large per-batch buffers and keeping only the compact evicted rows.

    ``num_evicted`` on the device (what the insert kernels produce) is not read:
    reading it would make every insert into a retaining table wait for the
    device, even one that evicted nothing. The arrays are kept whole, with the
    entries past the count made inert, and folded into the chunks later (see
    :func:`_append_evicted_masked`); a pop folds what is still pending.
    """
    if not (isinstance(num_evicted, torch.Tensor) and num_evicted.is_cuda):
        n = int(num_evicted)
        if n > 0:
            state.evicted_key_chunks.append(evicted_keys[:n].clone())
            state.evicted_tid_chunks.append(evicted_table_ids[:n].clone())
        return
    # The kernels fill the first ``num_evicted`` entries (a prefix count): make the
    # rest inert (a no-op when the extension already did), then it is the same
    # as any array whose non-members carry the sentinels.
    keys, tids = sf.inert_collected(num_evicted, evicted_keys, evicted_table_ids)
    _append_evicted_masked(state, keys, tids)


def _wait_for_stream(handle: Optional[int]) -> None:
    """Order the current stream after what the stream with this ``cuda_stream``
    handle has been given so far (``None``: no stream). The handle is a plain int,
    so that whatever holds it (a state, a cache) can still be copied."""
    current = torch.cuda.current_stream()
    if handle is None or handle == current.cuda_stream:
        return
    default = torch.cuda.default_stream()
    if handle == default.cuda_stream:
        # The default stream's handle is 0, which ``ExternalStream`` takes for "no
        # stream given": it would return an idle stream of the pool, and waiting for
        # that orders nothing. Ask for the default stream by name.
        other = default
    else:
        other = torch.cuda.ExternalStream(handle)
    current.wait_stream(other)


def _append_evicted_masked(
    state: DynamicEmbTableState,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
) -> None:
    """Append evicted keys given as arrays with sentinels, without reading a count.

    ``keys`` / ``table_ids`` are N long and mark the entries that are not evicted
    keys as inert: the reserved all-ones key and table id ``-1``. The members may
    sit anywhere in the arrays, not only in a prefix, and the number of them is
    derived on the device from the table ids. It is never a parameter: a masked
    array must not be appended with a count and sliced ``[:n]``, because its
    members are scattered and the slice would retain the wrong keys.

    The count is copied to a pinned ring without waiting for it; an entry is
    folded into ``evicted_key_chunks`` / ``evicted_tid_chunks`` by
    :func:`_resolve_pending_evicted` once the count has landed, so the arrays
    stay alive until then (``16 * N`` bytes an entry).
    """
    assert state.retain_ring is not None, "the table does not retain evicted keys"
    if keys.numel() == 0:
        return
    _resolve_pending_evicted(state, block=False)  # fold what has already landed
    count = (table_ids >= 0).sum().reshape(1)  # int64[1], stays on the device
    # A full ring gives no ticket: the entry is retried by a later call, on this
    # stream (the one the count was produced on), or read by a blocking pop.
    state.evicted_pending.append(
        [
            keys,
            table_ids,
            count,
            state.retain_ring.enqueue(count),
            torch.cuda.current_stream().cuda_stream,
        ]
    )


def _resolve_pending_evicted(state: DynamicEmbTableState, block: bool) -> None:
    """Fold the pending evicted keys of ``state`` into its chunk lists.

    Without ``block`` only the entries whose count has already landed on the host
    are folded; nothing waits. With ``block`` every entry is folded, waiting for
    its count: for a pop, whose result has a data-dependent size anyway.

    An entry is folded by moving its members to the front with a stable sort of
    the membership flag (the order of the keys is kept) and cutting ``[:n]``.
    """
    if not state.evicted_pending:
        return
    ring = state.retain_ring
    cur = torch.cuda.current_stream()
    keep = []
    for keys, tids, count, ticket, stream in state.evicted_pending:
        if ticket is None and ring is not None and stream == cur.cuda_stream:
            # The ring was full when the entry was made. A ticket is requested only
            # on the stream that produced ``count``: the copy and its event are
            # ordered by that stream.
            ticket = ring.enqueue(count)
        if ticket is not None and not (block or ring.ready(ticket)):
            keep.append([keys, tids, count, ticket, stream])
            continue
        if ticket is not None:
            n = int((ring.wait(ticket) if block else ring.value(ticket))[0])
            ring.release(ticket)
        elif block:
            # No ticket and a pop that must have the number: one read. It runs on
            # this stream, so order it after the one that produced the count.
            _wait_for_stream(stream)
            n = int(sf.read_now(count)[0])
        else:
            keep.append([keys, tids, count, None, stream])
            continue
        if n > 0:
            # The arrays belong to the stream that made them; the sort and the
            # gathers below run on this one, and the arrays are dropped here.
            keys.record_stream(cur)
            tids.record_stream(cur)
            order = torch.sort(
                (tids >= 0).to(torch.int8), descending=True, stable=True
            ).indices[:n]
            state.evicted_key_chunks.append(sf.as_i64(keys)[order].view(keys.dtype))
            state.evicted_tid_chunks.append(tids[order])
    state.evicted_pending = keep


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
    # A row with table id -1 is a sentinel (an inert entry of a masked append), not
    # another table's key: leaving it in the remainder would keep it there for ever.
    keep = (tids != table_id) & (tids >= 0)
    if bool(keep.any()):
        return out, [keys[keep]], [tids[keep]]
    return out, [], []


def _pop_state_evicted_keys(state: DynamicEmbTableState, table_id: int) -> torch.Tensor:
    """Pop (return + clear) the unique keys ``table_id`` had evicted."""
    # A pop returns data of a data-dependent size, so waiting for the counts of
    # the evictions that are still pending is part of its cost.
    _resolve_pending_evicted(state, block=True)
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
    *,
    async_count: bool = False,
) -> Tuple[
    torch.Tensor,
    Union[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Key-only insert_and_evict. Returns (indices, num_evicted, evicted_keys,
    evicted_table_ids, evicted_indices, evicted_scores).
    Caller is responsible for loading evicted values and storing new values.

    *preserve_existing* is forwarded to :func:`get_insert_score_arg` (e.g. backward
    re-insert should use True so existing slot scores are not overwritten).

    By default ``num_evicted`` is a host int and the four evicted arrays hold
    exactly that many entries, which costs a read of the count from the device.
    With *async_count* nothing is read: ``num_evicted`` stays an ``int64[1]`` tensor
    on the device and the evicted arrays are as long as ``keys``, the entries
    past ``num_evicted`` inert (the reserved all-ones key, index and table id
    ``-1``). A caller that takes this form must not branch on ``num_evicted``.
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
    ) = state.key_index_map.insert_and_evict(
        keys, table_ids, score_arg, async_count=async_count
    )

    return (
        indices,
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

    Embeddings and optimizer states come out at the table's own precision
    (``state.emb_dtype``), matching what ``incremental_dump`` yields, so a
    checkpoint stores what the table actually holds instead of an fp32 widening
    of it. Callers that want fp32 cast on their own.
    """
    emb_dim_t = state.table_emb_dims_cpu[table_id]
    vdim = state.table_value_dims_cpu[table_id]
    optim_state_dim = vdim - emb_dim_t

    for (
        keys,
        named_scores,
        indices,
        slots,
    ) in state.key_index_map._batched_export_keys_scores(
        [state.score_policy.name],
        state.device,
        batch_size=batch_size,
        return_index=True,
        return_slots=True,
        table_id=table_id,
    ):
        scores = named_scores[state.score_policy.name]
        flat_rows = indices
        values = load_from_flat_single_table(state, flat_rows, table_id)
        embeddings = values[:, :emb_dim_t].contiguous()
        if optim_state_dim != 0:
            opt_states = values[:, -optim_state_dim:].contiguous().to(device)
        else:
            opt_states = None
        # Multi-word score layouts (e.g. LruLfu: timestamp + frequency) must
        # persist ALL words, not just the exported column, so the checkpoint can
        # restore them exactly. Gather the full [N, num_scores] block at the
        # exported slots; single-score tables keep the [N] shape unchanged.
        if state.key_index_map.num_scores_ > 1:
            # Score blocks are keyed by slot; ``indices`` are rows for a table
            # that stores them.
            out_scores = state.key_index_map.gather_score_blocks(table_id, slots)
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
        # How to read the value files back. They carry no header, so without
        # these a loader can only assume the historical fp32 and guess the row
        # width from the file length. Note this dict is rebuilt and rewritten
        # whole on every dump (by rank 0 only, and not by the appending second
        # tier of a hybrid dump): anything a loader needs has to be contributed
        # here, because there is no later pass that merges into the file.
        meta_data[META_EMBEDDING_DTYPE] = dtype_to_meta_name(state.emb_dtype)
        meta_data[META_EMBEDDING_DIM] = int(state.table_emb_dims_cpu[table_id])
        # Recorded separately from the embedding dtype because the two can
        # differ: rowwise Adagrad keeps its accumulator in fp32 in any table.
        meta_data[META_OPT_STATE_DTYPE] = dtype_to_meta_name(
            state.optimizer.get_state_dtype(state.emb_dtype)
        )

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
        fembedding.write(_raw_bytes(embeddings))
        if state.evict_strategy == EvictStrategy.KLru:
            scores = timestamp - scores
        fscore.write(scores.cpu().numpy().tobytes())
        if fopt_states and opt_states_batch is not None:
            to_write = state.optimizer.states_for_checkpoint(
                opt_states_batch,
                state.table_emb_dims_cpu[table_id],
            )
            fopt_states.write(_raw_bytes(to_write))

    fkey.close()
    fembedding.close()
    if fscore:
        fscore.close()
    if fopt_states:
        fopt_states.close()


def _resolve_shard(
    shard_index: Optional[int], num_shards: Optional[int]
) -> Tuple[int, int]:
    """The table's own shard geometry, or the global rank and world size (the
    row-wise answer) when either half is left ``None``."""
    if num_shards is None or shard_index is None:
        num_shards = dist.get_world_size() if dist.is_initialized() else 1
        shard_index = dist.get_rank() if dist.is_initialized() else 0
    return shard_index, num_shards


def _count_owned_keys(
    emb_key_path: str,
    device: torch.device,
    dist_type: str,
    batch_size: int = 65536,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> int:
    """How many of a checkpoint file's keys this rank keeps.

    Reads only the key file, with the filter :func:`_iter_batches_from_files`
    applies. A checkpoint can hold many ranks' keys while a rank keeps only its
    own share, so the file's size is not what a rank has to make room for.
    """
    num_keys = os.path.getsize(emb_key_path) // KEY_TYPE.itemsize
    shard_index, num_shards = _resolve_shard(shard_index, num_shards)
    if num_shards <= 1:
        return num_keys
    owned = 0
    with open(emb_key_path, "rb") as fkey:
        for start in range(0, num_keys, batch_size):
            n = min(num_keys - start, batch_size)
            keys = torch.tensor(
                np.frombuffer(
                    fkey.read(KEY_TYPE.itemsize * n),
                    dtype=torch_dtype_to_np_dtype[KEY_TYPE],
                ),
                dtype=KEY_TYPE,
                device=device,
            )
            mask = owned_key_mask(keys, shard_index, num_shards, dist_type)
            owned += int(mask.sum().item()) if mask is not None else n
    return owned


def _iter_batches_from_files(
    emb_key_path: str,
    embedding_file_path: str,
    score_file_path: Optional[str],
    opt_file_path: Optional[str],
    dim: int,
    optstate_dim: int,
    device: torch.device,
    dist_type: str,
    batch_size: int = 65536,
    num_scores: int = 1,
    emb_dtype: torch.dtype = EMBEDDING_TYPE,
    opt_state_dtype: torch.dtype = OPT_STATE_TYPE,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> Iterator[Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]]:
    """Yield (keys, embeddings, scores, opt_states) batches from checkpoint files.

    *shard_index* / *num_shards* are the table's own shard geometry (its position
    among the ranks that hold it, and how many there are): the world for a
    row-wise table, one node for a table-row-wise one. Left ``None`` they fall
    back to the global rank and world size, which is the row-wise answer.

    ``num_scores`` is the number of score words per key in the score file. When
    > 1 (e.g. LruLfu's timestamp+frequency) the yielded ``scores`` is
    [n, num_scores]; otherwise it is [n].

    *emb_dtype* / *opt_state_dtype* are the precisions the files were written at,
    which is how far apart the batches step through them -- get them wrong and
    the files misparse rather than fail. Callers holding the table's meta should
    pass what :func:`resolve_checkpoint_value_layout` resolved; the fp32 defaults
    are what every checkpoint written before the meta recorded a dtype used.
    The yielded tensors keep the file's dtype; the insert path casts to the
    table's.

    Handles file I/O, deserialization, and keeping only the keys this rank owns.

    *dist_type* selects the ownership rule, which is the table's own -- the load
    path checks it against the checkpoint's before getting here
    (``_validate_load_meta``). It matters in two ways. When the checkpoint's
    shard count matches the world size each rank reads only its own file, every
    key in it already belongs to that rank, and the filter is a no-op -- but
    only if the rule is the right one. When the counts differ the files are
    re-split across ranks and the filter is what does the resharding.
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

    shard_index, num_shards = _resolve_shard(shard_index, num_shards)

    try:
        for start in range(0, num_keys, batch_size):
            n = min(num_keys - start, batch_size)

            keys_bytes = fkey.read(KEY_TYPE.itemsize * n)
            keys = torch.tensor(
                np.frombuffer(keys_bytes, dtype=torch_dtype_to_np_dtype[KEY_TYPE]),
                dtype=KEY_TYPE,
                device=device,
            )

            emb_bytes = fembedding.read(emb_dtype.itemsize * dim * n)
            embeddings = _tensor_from_bytes(emb_bytes, emb_dtype, device).view(-1, dim)

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
                opt_bytes = fopt.read(opt_state_dtype.itemsize * optstate_dim * n)
                opt_states = _tensor_from_bytes(
                    opt_bytes, opt_state_dtype, device
                ).view(-1, optstate_dim)

            masks = owned_key_mask(keys, shard_index, num_shards, dist_type)
            if masks is not None:
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
    # The table's key -> rank rule, already checked against the checkpoint's.
    dist_type: str
    # Precisions the value files were written at, resolved from meta (fp32 for
    # checkpoints predating it). Pass both to ``_iter_batches_from_files``.
    emb_dtype: torch.dtype = EMBEDDING_TYPE
    opt_state_dtype: torch.dtype = OPT_STATE_TYPE


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

    Reads meta JSON, validates opt_type / evict_strategy / dist_type, resolves
    the value files' precision and row width, resolves include_optim, and checks
    file-size consistency.
    """
    meta_data = load_from_json(meta_json_file_path)
    opt_type = meta_data.get("opt_type", None)
    # The optimizer checks only matter when the caller asked for its states.
    if (
        include_optim
        and opt_type
        and state.optimizer.get_opt_args().get("opt_type", None) != opt_type
    ):
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

    if include_optim and (not opt_file_path or not os.path.exists(opt_file_path)):
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
    emb_dtype, opt_state_dtype, ckpt_emb_dim = resolve_checkpoint_value_layout(
        meta_data, embedding_file_path, num_keys, dim
    )
    if ckpt_emb_dim != dim:
        raise ValueError(
            f"Embedding dim mismatch: {embedding_file_path} holds rows of "
            f"{ckpt_emb_dim} but the runtime table is configured with dim {dim}."
        )
    if emb_dtype != state.emb_dtype:
        # Not fatal: the insert path casts, so this is a deliberate precision
        # migration as often as it is a mistake. Say which way it goes, since
        # narrowing loses bits that the checkpoint still has.
        warnings.warn(
            f"Checkpoint {embedding_file_path} stores {emb_dtype} embeddings but "
            f"the runtime table is {state.emb_dtype}; values will be converted on "
            "load.",
            UserWarning,
            stacklevel=2,
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
            row_block = num_keys * opt_state_dtype.itemsize
            if file_bytes % row_block != 0:
                raise ValueError(
                    f"Optimizer state file {opt_file_path} size {file_bytes} is not divisible by "
                    f"{row_block} (num_keys={num_keys}, dtype itemsize={opt_state_dtype.itemsize})."
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
        dist_type=runtime_dist_type,
        emb_dtype=emb_dtype,
        opt_state_dtype=opt_state_dtype,
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

    # Normalize the embeddings to the table's precision up front. A checkpoint
    # may carry a different precision than the table (load converts -- see
    # _validate_load_meta). The optimizer state is deliberately left alone:
    # optimizer.states_from_checkpoint owns its conversion, because it need not
    # be the table's dtype (rowwise Adagrad keeps an fp32 accumulator), and an
    # elementwise cast here would round that accumulator away. The branch below
    # only calls it when the state is non-empty, which is the case it casts.
    embeddings = embeddings.to(state.emb_dtype)

    if opt_states is None and runtime_optstate_dim > 0:
        opt_states = torch.empty(
            keys.numel(),
            runtime_optstate_dim,
            dtype=state.emb_dtype,
            device=embeddings.device,
        )
        state.optimizer.reset_optimizer_states(opt_states, emb_dims=emb_dim_cfg)
    elif opt_states is not None and runtime_optstate_dim > 0:
        opt_states = state.optimizer.states_from_checkpoint(
            opt_states,
            emb_dim_cfg,
            state.emb_dtype,
            embeddings.device,
        )

    # Both halves are already state.emb_dtype (normalized above), so this cat
    # cannot silently widen the row via type promotion.
    values = (
        torch.cat(
            [embeddings.view(-1, dim), opt_states.view(-1, runtime_optstate_dim)],
            dim=-1,
        )
        if opt_states is not None
        else embeddings
    )

    # Multi-word score layouts (LruLfu: timestamp + frequency) cannot be restored
    # by a single-value insert (which only writes word 0). Place the keys, then
    # scatter the full [N, num_scores] block. Placement uses the table's own
    # policy with a zero frequency delta so that a full bucket evicts through the
    # ranked LruLfu evictor (honoring a custom score_function); the scatter then
    # overwrites both words.
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
            name=state.score_policy.name,
            value=torch.zeros(keys.numel(), dtype=SCORE_TYPE, device=keys.device),
            policy=state.score_policy.policy,
        )
        indices = state.key_index_map.insert(keys, tid_tensor, place_arg)
        # The score blocks are keyed by slot, the values by row (the same thing
        # unless the table stores rows).
        state.key_index_map.scatter_score_blocks(
            table_id,
            state.key_index_map.slots_of(keys, tid_tensor),
            scores.to(SCORE_TYPE).contiguous(),
        )
        store_to_flat_single_table(state, indices, table_id, values)
        return

    policy = ScorePolicy.ASSIGN
    tid_tensor = torch.full(
        (keys.numel(),), table_id, dtype=torch.int64, device=keys.device
    )

    if state.score_policy.policy == ScorePolicy.FLAG:
        # A table that keeps no score (NO_EVICTION alone): none is restored, but the
        # keys loaded set their flag like any new key.
        policy, scores = ScorePolicy.FLAG, None
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

    indices = state.key_index_map.insert(keys, tid_tensor, score_arg_insert)
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
    the row base, so there is no way to write the tail without the head. The
    embedding is therefore always written, and the only choice is what the tail
    gets:

    - **With ``OPTIMIZER_STATE``**, the delta carries the whole row; write it as
      it was dumped. Nothing is inferred.
    - **Without it**, rows that already belonged to this same key keep their
      state, because writing just the embedding columns leaves the tail
      untouched. Every other row is new to this key, so its tail still holds the
      previous occupant's moments and has to be reset by the optimizer.
    """
    if rows.numel() == 0:
        return
    emb_dim_cfg = state.table_emb_dims_cpu[table_id]
    optstate_dim = state.optimizer.get_state_dim(emb_dim_cfg)
    embeddings = embeddings.to(dtype=state.emb_dtype)
    want_opt = ReplayContent.OPTIMIZER_STATE in content and optstate_dim > 0

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
        opt = state.optimizer.states_from_checkpoint(
            optimizer_states.to(device=embeddings.device),
            emb_dim_cfg,
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
        opt_states = torch.empty(
            (fresh_emb.size(0), optstate_dim),
            dtype=state.emb_dtype,
            device=fresh_emb.device,
        )
        state.optimizer.reset_optimizer_states(opt_states, emb_dims=emb_dim_cfg)
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

    if _row_differs_from_slot(state):
        # ``use_rows`` tables carry (key slot, value row) as ``[n, 2]``; every
        # other table uses one index for both.
        key_slots = slot_index[:, 0].contiguous()
        value_rows = slot_index[:, 1].contiguous()
    else:
        key_slots = slot_index
        value_rows = slot_index

    # The score words are written with the key, so they are chosen here: either
    # restored from the delta or minted as if the key had just been inserted.
    if ReplayContent.SCORE in content and scores is not None:
        score_block = _replay_score_block(state, table_id, scores, timestamp)
    else:
        score_block = _fresh_score_block(state, table_id, n, timestamp, insert_score)
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

    if state.key_index_map.use_rows_:
        # The key now sits at the source's slot; make it own the source's row.
        # The row counter is set by the caller from the delta's metadata -- the
        # rows named here are only the ones that changed.
        state.key_index_map.set_rows(key_slots, table_id, value_rows)

    _replay_write_values(
        state, table_id, value_rows, embeddings, optimizer_states, same_key, content
    )
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
        self._state = create_table_state(options, optimizer, enable_overflow=True)
        # The metrics live on the device and are read only when somebody asks (see
        # ``cache_metrics``): the lookups fill them without waiting for anything.
        # Allocated when recording is switched on, never inside a step.
        self._metrics_dev: Optional[torch.Tensor] = None
        # The ``cuda_stream`` handles of the streams the last lookup and the last
        # insert ran on, for a reader that is on another one.
        self._metrics_streams: List[Optional[int]] = [None, None]
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
            state,
            unique_keys.size(0),
            unique_keys.device,
            score_source(state, lfu_accumulated_frequency),
        )
        result = state.key_index_map.lookup_with_overflow(
            unique_keys, table_ids, score_arg
        )
        if self._record_cache_metrics:
            m = self._metrics_dev
            # Counted on the device. A reserved key (the inert padding of a batch)
            # is not a key, and it is reported as not found.
            m[0].copy_(sf.valid_keys(unique_keys).sum())
            m[1].copy_(result[1].sum())
            self._metrics_streams[0] = torch.cuda.current_stream().cuda_stream
        return result

    def insert_and_evict(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        *,
        async_count: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Union[int, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Insert with counter-aware eviction and overflow fallback.

        With ``async_count`` the count of evictions is not read: ``num_evicted``
        stays an ``int64[1]`` tensor on the device and the four evicted arrays are
        as long as ``keys``, their entries past ``num_evicted`` inert (the
        reserved all-ones key, index and table id ``-1``).
        """
        state = self._state
        score_arg = get_insert_score_arg(state, keys.numel(), keys.device, scores)
        result = state.key_index_map.insert_and_evict_with_counter_and_overflow(
            keys, table_ids, score_arg, async_count=async_count
        )
        if self._record_cache_metrics:
            m = self._metrics_dev
            m[2].copy_(sf.valid_keys(keys).sum())
            if async_count:
                m[3].copy_(result[1].reshape(()))  # num_evicted, on the device
            else:
                # ``m[3] = n`` copies a CPU scalar tensor to the device (a transfer
                # from pageable memory); ``fill_`` is a kernel taking ``n`` as an
                # argument.
                m[3].fill_(int(result[1]))
            self._metrics_streams[1] = torch.cuda.current_stream().cuda_stream
        return result

    def reset(self) -> None:
        self._state.key_index_map.reset()

    @property
    def cache_metrics(self) -> Optional[torch.Tensor]:
        """What the last lookups did, as a CPU ``int64[10]`` tensor; ``None`` unless
        recording was switched on with :meth:`set_record_cache_metrics`.

        ``[0]`` is the number of unique keys of the last lookup and ``[1]`` how many
        of them were found; ``[2]`` the number of keys of the last insert and ``[3]``
        how many entries it evicted. The counters are kept on the device, so
        reading them is where the host waits for the device, not the lookups: one
        batched copy per read. It is a snapshot, a later lookup does not change
        a tensor returned earlier.
        """
        if not self._record_cache_metrics:
            return None
        # The lookup and the insert may have run on streams other than the reader's
        # (and than each other's): take what each of them has done so far.
        for handle in set(self._metrics_streams):
            _wait_for_stream(handle)
        # read_now reuses one pinned buffer for every read, so keep a copy.
        return sf.read_now(self._metrics_dev)[0].clone()

    def set_record_cache_metrics(self, record: bool) -> None:
        if record and self._metrics_dev is None:
            self._metrics_dev = torch.zeros(
                10, dtype=torch.long, device=self._state.device
            )
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


class MaskedFind(NamedTuple):
    """What :meth:`DynamicEmbStorage.lookup_masked` (or the hybrid one) found.

    Nothing is compacted and no count is read back: every field is as long as
    the keys that were looked up, and a lane whose key is the reserved all-ones
    key (``sf.mask_keys``) is not found and owns no row, so a caller leaves a key
    out of a lookup, and out of the load that follows it, by masking the key.
    """

    founds: Tensor  # bool[N]
    # The full-length score tensor the lookups used (``None`` for an LRU table and
    # for a const lookup). A later insert, or the lookup of a second tier, takes
    # it as is: the kernels read ``score_input[i]`` for every lane, masked ones
    # included, so it must never be compacted.
    score_in: Optional[Tensor]
    scores: Tensor  # int64[N]
    # Opaque token for ``load_masked``: the value rows (a tensor, ``-1`` where
    # there is none), or one such tensor per tier for a ``HybridStorage``.
    rows: Any


def _compact_missing(
    state: DynamicEmbTableState,
    unique_keys: torch.Tensor,
    table_ids: torch.Tensor,
    founds: torch.Tensor,
    score_in: Optional[torch.Tensor],
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The one read of the public ``find``: compact what was not found.

    ``Storage.find`` returns the number of missing keys as a host int and the
    missing keys, their positions among ``unique_keys`` (ascending), table ids
    and scores sized by it, so this is a stream compaction with a count read
    back. A caller that can mask does not come here: it uses ``lookup_masked``.
    """
    # ``_find_keys`` compacted keys already converted to the key map's dtype.
    if unique_keys.dtype != state.key_index_map.key_type:
        unique_keys = unique_keys.to(state.key_index_map.key_type)
    (
        num_missing,
        missing_indices,
        (missing_keys, missing_table_ids, missing_scores),
    ) = flagged_compact(~founds, [unique_keys, table_ids, score_in])
    return (
        num_missing,
        missing_keys,
        missing_indices,
        missing_table_ids,
        missing_scores,
    )


def _select_refused_values(
    evicted_indices: torch.Tensor,
    evicted_table_ids: torch.Tensor,
    values: torch.Tensor,
    evicted_values: torch.Tensor,
) -> None:
    """Give a key the table refused its own row of ``evicted_values``.

    ``insert_and_evict`` reports a refused insert as an evicted entry whose index
    ``-(i + 1)`` names row ``i`` of the input, and ``load_from_flat`` leaves that
    entry's row of ``evicted_values`` unwritten; ``select_insert_failed_values``
    copies ``values[i]`` into it. Evicted arrays that end in an inert tail must
    not send the tail through that kernel: the tail's index is ``-1`` too, the
    refused insert of row 0 as far as the kernel can tell, and it takes ONE row
    stride, ``values.size(1)``, for the rows it reads and the rows it writes. Row
    0 would be copied over every tail row, and over rows of real evictions when
    ``values`` is narrower than ``evicted_values`` (``HybridStorage.load`` passes
    the compact row of its table, the state's rows are as wide as its widest
    table). The tail is told apart by its table id ``-1`` and hidden behind index
    0, which the kernel skips; the indices it resets are a copy nobody reads.
    """
    select_insert_failed_values(
        torch.where(evicted_table_ids >= 0, evicted_indices, 0),
        values,
        evicted_values,
    )


def _after_growth(method):
    """Make a public method of a storage order the current stream after the last growth
    of the state(s) it reads before it launches anything (``wait_for_growth``): a growth
    only queues its kernels on the stream that decided to grow.

    Not for a generator: its body runs at the first ``next()``, on whatever stream is
    current then, while this wrapper would order the stream that made the iterator. A
    generator calls ``self.wait_for_growth()`` itself, in its body."""
    # sync-ok: a Python bool about a function, decided when the class is made
    assert not inspect.isgeneratorfunction(method), (
        f"{method.__qualname__} is a generator: call self.wait_for_growth() in its body "
        "instead of decorating it with _after_growth"
    )

    @functools.wraps(method)
    def ordered(self, *args, **kwargs):
        self.wait_for_growth()
        return method(self, *args, **kwargs)

    return ordered


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

    def wait_for_growth(self) -> None:
        """Order the current stream after the last growth of the tables, and register it
        as one of their users (:meth:`DynamicEmbTableState.wait_for_growth`)."""
        self._state.wait_for_growth()

    @property
    def key_index_map(self):
        self._state.wait_for_growth()
        return self._state.key_index_map

    @_after_growth
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

    @_after_growth
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
                    score_input=score_source(state, lfu_freq),
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

    def expand_if_need(
        self,
        unique_size_per_table: torch.Tensor,
        *,
        exact: bool = False,
        inserts_in_forward: bool = False,
    ) -> None:
        """Expand the tables that a batch's keys would overfill.

        ``unique_size_per_table`` bounds, per table, the keys the batch can add. The
        decision does not wait for the device unless ``exact`` is set; see
        :func:`expand_if_need_impl`."""
        expand_if_need_impl(
            self._state,
            unique_size_per_table,
            exact=exact,
            inserts_in_forward=inserts_in_forward,
        )

    @_after_growth
    def request_size_snapshot(self) -> None:
        """Start reading the table sizes back for the next expansion decision; on the
        stream that inserts into this storage. Does nothing for a table that cannot
        grow."""
        request_size_snapshot(self._state)

    def note_forwarded(self, bound: torch.Tensor) -> None:
        """The forward of a batch bounded by ``bound`` has inserted its keys (see
        ``inserts_in_forward`` of :meth:`expand_if_need`)."""
        _note_forwarded(self._state, bound)

    @_after_growth
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

    @_after_growth
    def lookup_masked(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        key_scores: Optional[torch.Tensor] = None,
    ) -> MaskedFind:
        """:meth:`find` without the host count: one lookup of every key.

        The lookup scores exactly as :meth:`find` does (const outside training).
        Nothing is compacted and nothing is read back; a key that must not take
        part is replaced by the reserved key (``sf.mask_keys``) and comes back
        not found, with row ``-1`` and its score untouched.
        """
        r = _find_keys(
            self._state,
            unique_keys,
            table_ids,
            score_source(self._state, lfu_accumulated_frequency, key_scores),
            compact=False,
        )
        return MaskedFind(founds=r[5], score_in=r[4], scores=r[6], rows=r[7])

    @_after_growth
    def load_masked(
        self,
        found: MaskedFind,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Load the rows ``found`` names; a lane without a row (``-1``) is left alone.

        With ``output`` the rows are written into that ``[N, dim]`` buffer, whose
        other lanes keep what they hold, instead of into a new one.
        """
        return load_from_flat(
            self._state, found.rows, table_ids, copy_mode, output=output
        )

    @_after_growth
    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        key_scores: Optional[torch.Tensor] = None,
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
        # The host-int contract is the one read this costs (see _compact_missing);
        # a caller that does not need the count uses lookup_masked / load_masked.
        found = self.lookup_masked(
            unique_keys, table_ids, lfu_accumulated_frequency, key_scores
        )
        values = self.load_masked(found, table_ids, copy_mode)
        (
            h_num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
        ) = _compact_missing(
            self._state, unique_keys, table_ids, found.founds, found.score_in
        )

        return (
            h_num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            found.founds,
            found.scores,
            values,
        )

    @_after_growth
    def increment_counter(
        self,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> None:
        """Increment ref-counter at given per-table slot indices. table_ids must be provided and aligned with slot_indices."""
        self._state.key_index_map.increment_counter(slot_indices, table_ids)

    @_after_growth
    def decrement_counter(
        self,
        slot_indices: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> None:
        """Decrement ref-counter at given per-table slot indices. table_ids must be provided and aligned with slot_indices."""
        self._state.key_index_map.decrement_counter(slot_indices, table_ids)

    @_after_growth
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

    @_after_growth
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

    @_after_growth
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
        )
        # A checkpoint load waits for the exact sizes: it is not on the training path.
        self.expand_if_need(unique_size_per_table, exact=True)

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

    @_after_growth
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
            values_cat = torch.empty(0, emb_dim_t, dtype=state.emb_dtype, device=dev)
            slots_cat = torch.empty(
                (0, 2) if _row_differs_from_slot(state) else (0,),
                dtype=torch.int64,
                device=dev,
            )
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

    @_after_growth
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

    @_after_growth
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

    @_after_growth
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
        # A generator: the order is made when it runs, not when the iterator is made
        # (see _after_growth).
        self.wait_for_growth()
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

    @_after_growth
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
        # Neither tier is ever pinned (HybridStorage takes the generic path, with
        # no PrefetchState), so neither keeps a per-slot reference counter.
        self._hbm = create_table_state(hbm_options, optimizer, use_ref_counter=False)
        self._host = create_table_state(
            host_options,
            optimizer,
            evicted_item_mode=host_options[0].evicted_item_mode,
            use_ref_counter=False,
        )
        self.optimizer = optimizer

    def wait_for_growth(self) -> None:
        """Order the current stream after the last growth of the tiers (only the host
        tier grows), and register it as one of their users."""
        self._hbm.wait_for_growth()
        self._host.wait_for_growth()

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

    def expand_if_need(
        self,
        unique_size_per_table: torch.Tensor,
        *,
        exact: bool = False,
        inserts_in_forward: bool = False,
    ) -> None:
        """Expand the host tier if a batch's keys would overfill it (the HBM tier is not
        expanded). ``unique_size_per_table`` bounds, per table, the keys the batch can
        add; see :func:`expand_if_need_impl`."""
        expand_if_need_impl(
            self._host,
            unique_size_per_table,
            exact=exact,
            inserts_in_forward=inserts_in_forward,
        )

    @_after_growth
    def request_size_snapshot(self) -> None:
        """Start reading the host tier's sizes back for the next expansion decision;
        on the stream that inserts into it. Does nothing if it cannot grow."""
        request_size_snapshot(self._host)

    def note_forwarded(self, bound: torch.Tensor) -> None:
        """The forward of a batch bounded by ``bound`` has inserted its keys (see
        ``inserts_in_forward`` of :meth:`expand_if_need`)."""
        _note_forwarded(self._host, bound)

    @_after_growth
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

    @_after_growth
    def lookup_masked(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        key_scores: Optional[torch.Tensor] = None,
    ) -> MaskedFind:
        """:meth:`find` without the host counts: both tiers, no compaction.

        Both lookups score exactly as :meth:`find` always did (non-const while
        ``training``), so LRU timestamps and LFU counts of both tiers move the same
        way. ``MaskedFind.rows`` is the pair of the tiers' rows for
        :meth:`load_masked`.
        """
        if key_scores is not None:
            raise NotImplementedError(
                "A score per key is not supported by HybridStorage."
            )
        r_hbm = _find_keys(
            self._hbm,
            unique_keys,
            table_ids,
            score_source(self._hbm, lfu_accumulated_frequency),
            compact=False,
        )
        founds_hbm, score_in = r_hbm[5], r_hbm[4]
        # The host tier is asked only about what the HBM tier lacks: an HBM hit
        # becomes the reserved key, which is not found and touches no table, so
        # the host tier's scores move for exactly those keys without compacting
        # them first. The host lookup takes the HBM lookup's full-length score
        # tensor, never a compacted one (the kernels read it for every lane).
        r_host = _find_keys(
            self._host,
            sf.mask_keys(unique_keys, ~founds_hbm),
            table_ids,
            score_in,
            compact=False,
        )
        founds_host = r_host[5]
        return MaskedFind(
            founds=founds_hbm | founds_host,
            score_in=score_in,
            # a lane the host tier did not find keeps the HBM lookup's score
            scores=torch.where(founds_host, r_host[6], r_hbm[6]),
            rows=(r_hbm[7], r_host[7]),
        )

    @_after_growth
    def load_masked(
        self,
        found: MaskedFind,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Load the rows ``found`` names from both tiers.

        A key lives in one tier, so the host tier's load writes exactly the lanes
        the HBM tier's left alone (row ``-1``), into the same buffer: both tiers
        share their dims.
        """
        values = load_from_flat(
            self._hbm, found.rows[0], table_ids, copy_mode, output=output
        )
        return load_from_flat(
            self._host, found.rows[1], table_ids, copy_mode, output=values
        )

    @_after_growth
    def find(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
        copy_mode: CopyMode,
        lfu_accumulated_frequency: Optional[torch.Tensor] = None,
        key_scores: Optional[torch.Tensor] = None,
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
        # The missing arrays are those of the keys neither tier has, and the host
        # count is the one read of this call (see _compact_missing); a caller that
        # does not need it uses lookup_masked / load_masked.
        found = self.lookup_masked(
            unique_keys, table_ids, lfu_accumulated_frequency, key_scores
        )
        values = self.load_masked(found, table_ids, copy_mode)
        (
            h_num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
        ) = _compact_missing(
            self._hbm, unique_keys, table_ids, found.founds, found.score_in
        )
        return (
            h_num_missing,
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            found.founds,
            found.scores,
            values,
        )

    # -- Insert: HBM first, evictions to host --

    @_after_growth
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

        Nothing is read back, so the arrays may be longer than the keys that matter:
        a key that must not take part is the reserved key (``sf.mask_keys``), which
        no table accepts and no store writes.
        """
        if unique_keys.numel() == 0:
            return
        if preserve_existing:
            # Both lookups score as ``find`` does (non-const while ``training``).
            r_hbm = _find_keys(self._hbm, unique_keys, table_ids, compact=False)
            # A lane with no row (``-1``) is skipped by the store.
            store_to_flat(self._hbm, r_hbm[7], table_ids, unique_values)
            r_host = _find_keys(
                self._host,
                sf.mask_keys(unique_keys, ~r_hbm[5]),
                table_ids,
                r_hbm[4],
                compact=False,
            )
            store_to_flat(self._host, r_host[7], table_ids, unique_values)
            return

        (
            indices,
            _num_evicted,
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
            async_count=True,
        )

        # The count of evictions stays on the device: the evicted arrays are as long
        # as the batch and inert past the evictions (reserved key, table id and row
        # ``-1``), so the load, the store and the host insert below run over all of
        # them and skip the rest. The select would not: it is given the tail hidden.
        evicted_values = load_from_flat(
            self._hbm, evicted_indices, evicted_table_ids, copy_mode=CopyMode.VALUE
        )
        _select_refused_values(
            evicted_indices, evicted_table_ids, unique_values, evicted_values
        )
        store_to_flat(self._hbm, indices, table_ids, unique_values)

        _insert_key_values(
            self._host,
            evicted_keys,
            evicted_table_ids,
            evicted_values,
            evicted_scores,
        )

    @_after_growth
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

    @_after_growth
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

    @_after_growth
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

    @_after_growth
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

    @_after_growth
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

        # A checkpoint load waits for the exact sizes: it is not on the training path.
        self.expand_if_need(unique_size_per_table, exact=True)

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
                _num_evicted,
                evicted_keys,
                evicted_table_ids,
                evicted_indices,
                evicted_scores,
            ) = _insert_and_evict_keys(
                self._hbm, keys, tids, file_scores, async_count=True
            )

            # As in :meth:`insert`: the count of evictions stays on the device and
            # the evicted arrays are inert past it, so nothing is sliced or tested.
            # ``values`` is this table's compact row, narrower than the evicted rows
            # (the state's widest) for a table narrower than the widest of its
            # tier: the inert tail must stay out of the select.
            evicted_values = load_from_flat(
                self._hbm,
                evicted_indices,
                evicted_table_ids,
                copy_mode=CopyMode.VALUE,
            )
            _select_refused_values(
                evicted_indices, evicted_table_ids, values, evicted_values
            )
            store_to_flat_single_table(self._hbm, ins_indices, table_id, values)

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
        # A generator: the order is made when it runs, not when the iterator is made
        # (see _after_growth), and again where the consumer's loop body may have taken
        # any time: the erase and the export of the host tier come after it.
        self.wait_for_growth()
        hbm_key_parts: List[torch.Tensor] = []
        for batch in export_keys_values_iter(self._hbm, device, batch_size, table_id):
            keys, embeddings, opt_states, scores = batch
            hbm_key_parts.append(keys.detach())
            yield (keys, embeddings, opt_states, scores)

        if hbm_key_parts:
            hbm_unique = torch.unique(torch.cat(hbm_key_parts, dim=0))
        else:
            hbm_unique = torch.empty(0, dtype=torch.int64, device=device)

        self.wait_for_growth()
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
        next_tier.insert(gone_keys, gone_tids, gone_values)


def _write_back_evicted_masked(
    storage: DynamicEmbStorage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    values: torch.Tensor,
    slots: Optional[torch.Tensor],
) -> None:
    """:func:`_write_back_evicted` without a next tier, over arrays of any length.

    ``keys`` / ``table_ids`` / ``values`` are as long as the evicted arrays of an
    insert, an entry that is not an eviction being inert: the reserved all-ones
    key and table id ``-1``. The keys storage still has get their value written to
    the row they hold (``store_to_flat`` skips a row of ``-1``): storage keeps the
    score of a key, only the value moves, so this is what an insert that keeps
    existing scores does, without the insert. Without a store below it a key
    storage has evicted since is dropped, or retained when storage retains
    evicted keys.

    ``slots`` is the index the insert reported for each entry (``None`` for an
    array that has none, as in a flush). A negative one is not an eviction: the
    insert refused that key (its bucket and overflow are all pinned) and reports
    it as an evicted entry with index ``-(i + 1)``, a row of ``values`` that was
    never loaded. Such an entry is neither written back nor retained.
    """
    st = storage._state
    with torch.cuda.nvtx.range("op:write_back_evicted"):
        # A const lookup: it leaves the scores as they are.
        found = _find_keys(st, keys, table_ids, const_lookup=True, compact=False)
        found_rows = found[7]
        live = None if slots is None else (slots >= 0)
        rows = found_rows if live is None else torch.where(live, found_rows, -1)
        store_to_flat(st, rows, table_ids, values)
        if EvictedItemMode.RETAIN_KEY in st.evicted_item_mode:
            # Keys storage no longer has were evicted by it in the meantime.
            gone = (found_rows < 0) if live is None else ((found_rows < 0) & live)
            _append_evicted_masked(
                st, sf.mask_keys(keys, gone), torch.where(gone, table_ids, -1)
            )


def flush_cache(
    cache: DynamicEmbCache,
    storage: DynamicEmbStorage,
    external_storage: Optional[Storage] = None,
) -> None:
    """Push every cache entry's value down, as the prefetch path does on eviction.

    Storage keeps the scores, so a key it still has only has its value
    updated. One it has evicted goes to ``external_storage``, or is dropped
    when there is none: it is not re-inserted, which would evict another row.
    """
    # The storage below the cache may have grown on another stream (the prefetch).
    wait_for_growth = getattr(storage, "wait_for_growth", None)
    if wait_for_growth is not None:
        wait_for_growth()
    state = cache._state
    batch_size = state.threads_in_wave

    for t in range(state.num_tables):
        for keys, _, indices in state.key_index_map._batched_export_keys_scores(
            [state.score_policy.name],
            state.device,
            batch_size=batch_size,
            return_index=True,
            table_id=t,
        ):
            tid = torch.full((keys.numel(),), t, dtype=torch.int64, device=keys.device)
            values = load_from_flat(state, indices, tid, copy_mode=CopyMode.VALUE)
            _write_back_evicted(storage, external_storage, keys, tid, values)


# ---------------------------------------------------------------------------
# eval_lookup – unified eval path for storage-only and cache+storage
# ---------------------------------------------------------------------------


def _eval_lookup_storage(
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eval lookup of an in-tree storage: no host count, no compaction.

    ``valid`` (bool, one entry per key) marks the real keys of a batch whose tail
    is padding (the reserved key, table id ``-1``); ``None`` means every key is
    real. A key found nowhere takes the constant of the eval initializer, which
    is written over its row only: padding is neither loaded nor initialized.
    """
    found = storage.lookup_masked(keys, table_ids)  # const: not training in eval
    embs = storage.load_masked(found, table_ids, CopyMode.EMBEDDING)
    missing = ~found.founds if valid is None else valid & ~found.founds
    sf.initialize_masked(initializer, embs, keys, table_ids, missing)
    return embs


def _eval_lookup_storage_legacy(
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


def _eval_lookup_cached(
    cache: Cache,
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """:func:`_eval_lookup_storage` behind a cache, asking storage only about the
    cache's misses.

    A key the cache has (or padding) is replaced by the reserved key, which finds
    nothing and loads nothing, so the storage lookup and load run over every lane
    without overwriting the embeddings the cache gave.
    """
    _, founds, cache_indices = cache.lookup(keys, table_ids)
    embs = load_from_flat(
        cache._state, cache_indices, table_ids, copy_mode=CopyMode.EMBEDDING
    )
    miss = ~founds if valid is None else valid & ~founds
    found = storage.lookup_masked(sf.mask_keys(keys, miss), table_ids)
    storage.load_masked(found, table_ids, CopyMode.EMBEDDING, output=embs)
    sf.initialize_masked(initializer, embs, keys, table_ids, miss & ~found.founds)
    return embs


def _eval_lookup_cached_legacy(
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


def _masked_eval_applies(storage: Storage, cache: Optional[Cache]) -> bool:
    """Whether an eval lookup may take the masked path (host facts only).

    The masked functions need ``lookup_masked`` / ``load_masked``, which only the
    in-tree storages have, by exact type: a subclass may override ``find``. Any
    other ``Storage`` or ``Cache`` keeps the exact-length functions, as does
    ``DYNAMICEMB_SYNC_FREE=0``, the switch that restores the legacy paths. It is
    read from the environment at every call, not from the flag the prefetch
    dispatchers hold (that module imports this one): a test flips it with
    ``monkeypatch.setenv``.
    """
    return (
        os.environ.get("DYNAMICEMB_SYNC_FREE", "1") != "0"
        and type(storage) in (DynamicEmbStorage, HybridStorage)
        and (cache is None or type(cache) is DynamicEmbCache)
    )


def eval_lookup(
    storage: Storage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    initializer: Callable,
    cache: Optional[Cache] = None,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eval-only lookup (no insertion, no admission, no backward).

    When *cache* is ``None``, looks up directly from *storage*.
    When *cache* is provided, looks up from cache first, then falls back to
    *storage* for cache misses.  Only embedding columns are copied
    (``CopyMode.EMBEDDING``); optimizer states are never touched.

    *valid* (bool, one entry per key) marks the real keys of a batch padded to a
    fixed length: the others are the reserved key with table id ``-1``, and their
    rows are neither looked up nor written (they hold garbage). It needs the
    in-tree storage; ``None`` means every key is real.

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

    if valid is not None or _masked_eval_applies(storage, cache):
        if cache is None:
            return _eval_lookup_storage(storage, keys, table_ids, initializer, valid)
        return _eval_lookup_cached(
            cache,
            storage,
            keys,
            table_ids,
            initializer,
            valid,
        )

    if cache is None:
        return _eval_lookup_storage_legacy(storage, keys, table_ids, initializer)

    return _eval_lookup_cached_legacy(
        cache,
        storage,
        keys,
        table_ids,
        initializer,
    )
