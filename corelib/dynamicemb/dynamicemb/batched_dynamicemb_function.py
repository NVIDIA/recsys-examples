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

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, ContextManager, Optional, Tuple

import torch
from dynamicemb import sync_free as sf
from dynamicemb.async_erase import table_section
from dynamicemb.dynamicemb_config import EvictedItemMode
from dynamicemb.embedding_admission import IN_TREE_ADMITTERS
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.key_value_table import (
    Cache,
    DynamicEmbCache,
    DynamicEmbStorage,
    DynamicEmbTableState,
    HybridStorage,
    Storage,
    _append_evicted,
    _find_keys,
    _insert_and_evict_keys,
    _select,
    _write_back_evicted,
    eval_lookup,
    get_insert_score_arg,
    load_from_flat,
    score_source,
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


def largest_score_per_unique_key(
    scores: torch.Tensor, reverse_indices: torch.Tensor, num_unique: int
) -> torch.Tensor:
    """The score of each unique key: the largest of those its occurrences had.

    ``reverse_indices[i]`` is the unique key that input key ``i`` was merged
    into; a key that was not repeated keeps its own score.
    """
    unique_scores = torch.zeros(num_unique, dtype=torch.int64, device=scores.device)
    return unique_scores.scatter_reduce_(
        0, reverse_indices, scores, reduce="amax", include_self=False
    )


class _OutstandingEntry:
    """One prefetched batch's claim on the cache, see :class:`OutstandingKeys`."""

    __slots__ = ("bound", "total", "exact")

    def __init__(
        self, bound: int, total: Optional[torch.Tensor], exact: Optional[int]
    ) -> None:
        self.bound = bound
        self.total = total
        self.exact = exact


class OutstandingKeys:
    """The keys that prefetched batches, not forwarded yet, pin in the cache.

    A cache cannot evict a key a prefetched batch still has to use, so those batches
    must fit in it together, and prefetching more than fits is an error. What has to
    fit is their unique keys. Reading each batch's unique count back from the device
    would make every prefetch wait, so the check is made in two steps:

    * ``bound`` adds up what each batch says it can be at most -- a host number, the
      batch's key count. When that fits, the unique keys fit too, and nothing else
      is done.
    * Only when it does not fit is the exact count asked for from the device, late:
      through a pinned ring and an event that nothing waits for. A cache that really
      is over-subscribed raises, with the exact count, one or two prefetches after
      the batch that did it; a cache that only sees batches full of duplicates never
      raises.

    A batch whose unique count is known on the host (``exact``) is checked at once.
    """

    def __init__(self) -> None:
        self.bound = 0
        # entry id -> entry, in the order the batches were prefetched
        self._live = {}
        self._ring = sf.DeferredRead((1,), torch.int64, 4)
        # (ring ticket, the part of that count the host knew), for the counts in flight
        self._pend = []

    def reset(self) -> None:
        """Forget every batch: the cache was reset, nothing is pinned in it any more."""
        for ticket, _ in self._pend:
            self._ring.release(ticket)
        self.bound, self._live, self._pend = 0, {}, []

    def add(
        self, bound: int, total: Optional[torch.Tensor], exact: Optional[int]
    ) -> _OutstandingEntry:
        """Count a prefetched batch in. ``bound`` is a host int that its unique keys
        cannot pass; ``total`` is their count as a 0-d tensor on the device, or None
        when ``exact`` -- the same count as a host int -- is known. Returns the entry
        to hand to :meth:`release`."""
        entry = _OutstandingEntry(bound, total, exact)
        self.bound += bound
        self._live[id(entry)] = entry
        return entry

    def release(self, entry: Optional[_OutstandingEntry]) -> None:
        """The batch has been forwarded (whether or not its backward will run): it no
        longer pins anything that a later prefetch has to make room for."""
        if self._live.pop(id(entry), None) is not None:
            self.bound -= entry.bound

    @staticmethod
    def _raise(outstanding: int, capacity: int) -> None:
        raise RuntimeError(
            f"Outstanding prefetched keys ({outstanding}) "
            f"exceed cache capacity ({capacity}). "
            "Reduce prefetch pipeline depth or increase cache size."
        )

    def check(self, capacity: int) -> None:
        """Raise if the batches counted in do not fit in a cache of ``capacity`` keys.
        Never waits for the device and never raises for a cache that does fit."""
        # A count asked for earlier that has landed since.
        for pending in list(self._pend):
            ticket, host_part = pending
            if self._ring.ready(ticket):
                outstanding = int(self._ring.value(ticket)[0]) + host_part
                self._ring.release(ticket)
                self._pend.remove(pending)
                if outstanding > capacity:
                    self._raise(outstanding, capacity)
        if self.bound <= capacity:
            # The unique keys are at most the bound: they fit, nothing to count.
            return
        live = list(self._live.values())
        device_totals = [e.total.reshape(()) for e in live if e.exact is None]
        # Host ints only: a device tensor made from one would be a blocking upload.
        host_part = sum(e.exact for e in live if e.exact is not None)
        if not device_totals:
            # Every batch's count is known on the host: decide now.
            if host_part > capacity:
                self._raise(host_part, capacity)
            return
        # Not decided: count the unique keys exactly, one stacked sum, read late.
        if self._ring.free_slots() > 0:
            ticket = self._ring.enqueue(torch.stack(device_totals).sum().reshape(1))
            if ticket is not None:
                self._pend.append((ticket, host_part))

    def __reduce__(self):
        # A copy of the module (copy.deepcopy, pickle) starts with nothing outstanding
        # and a ring of its own.
        return (OutstandingKeys, ())


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
    # with. It differs from ``slot_indices`` for a table that stores a row per
    # key (a storage): rows are handed out densely while slots come from the
    # key's hash, and the key map of a table that never evicts is deliberately
    # larger than its value buffer, so indexing values by slot would read past the
    # end of it. A cache points this at the slot tensor itself rather than leaving it unset, so a
    # consumer never has to ask which of the two it is holding. Anything indexed
    # by hash slot (the ref counter) uses ``slot_indices``.
    value_row_indices: Optional[torch.Tensor] = None
    non_admitted_positions: Optional[torch.Tensor] = None
    # One int64 per unique key, for tables that take a score per key.
    unique_scores: Optional[torch.Tensor] = None
    num_prefetched_keys: int = 0
    outstanding_keys_ref: Optional[OutstandingKeys] = None
    # The key_map_generation (rehash count) of the table state the slots were found in: a later
    # prefetch may expand it before this batch's backward runs.
    key_map_generation: int = 0
    # This batch's entry in ``outstanding_keys_ref``, released by the forward.
    outstanding_entry: Optional[object] = None
    # Row validity of the unique arrays: bool per row, True for a real unique key.
    # None when the arrays are exactly as long as the unique count (every row is real).
    unique_valid: Optional[torch.Tensor] = None
    # Number of unique keys as a 0-d device tensor, which the host never reads.
    # None while the unique arrays are exactly as long as that count.
    unique_total: Optional[torch.Tensor] = None
    # bool per unique key, True for a key the admitter refused. A masked prefetch
    # path sets it when it ran its miss path with an admitter; the legacy paths set
    # ``non_admitted_positions`` instead. None: no key was refused.
    non_admitted_mask: Optional[torch.Tensor] = None
    # DEFAULT mode: the forward takes the mask-form path. Decided once in the
    # prefetch, from host facts, so that forward and backward agree on the layout.
    generic_masked: bool = False
    # What the growth of the storage was told this batch can add to each table (CPU
    # int64 per table); a forward that inserts the batch reports it back (see
    # ``note_forwarded`` of the storages).
    table_key_bounds: Optional[torch.Tensor] = None


def _slots_after_rehash(
    state: DynamicEmbTableState,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    slot_indices: torch.Tensor,
) -> torch.Tensor:
    """The slots of a batch's keys in ``state`` as it is now, for a batch
    whose prefetch found them before a rehash moved them: the keys are looked
    up again, read only (their ref counts kept them in the table). A key the
    prefetch did not admit stays without a slot. Value rows do not move: a
    storage keeps each key's row through a rehash, and a cache does not
    rehash."""
    # Only a table that stores a row per key keeps its value rows through a
    # rehash (``use_rows_``). Without it the row is the slot and the value moves
    # with the key, so ``value_row_indices`` would go stale too: not handled here.
    assert state.key_index_map.use_rows_, (
        "a rehash since the prefetch is only handled for a table that stores a "
        "value row per key (DynamicEmbStorage); a cache does not rehash"
    )
    # A read-only lookup of the slots alone: ``_find_keys`` would also compact the
    # misses, a host read for a result nobody uses here.
    slots = state.key_index_map.slots_of(unique_keys, unique_table_ids)
    return torch.where(slot_indices >= 0, slots, slot_indices)


def _is_hbm_storage(storage: Storage) -> bool:
    """True if values live in GPU device memory (HBM). Use buffer type, not
    tensor.is_cuda: host memory registered to CUDA address space can report is_cuda True.
    """
    return (
        isinstance(storage, DynamicEmbStorage)
        and storage._state.tables[0].is_device_buffer()
    )


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

        # Only a strategy with its own initializer needs the rejected rows (a host
        # fact); the compaction that finds them is a host read.
        if non_admitted_initializer is not None:
            non_admitted_mask = ~admit_mask
            with torch.cuda.nvtx.range("op:flagged_compact"):
                _, _, (non_admitted_indices,) = flagged_compact(
                    non_admitted_mask, [missing_indices]
                )
            if non_admitted_indices.numel() > 0:
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


def _load_storage_values(
    storage: DynamicEmbStorage,
    slots: torch.Tensor,
    miss_tids: torch.Tensor,
) -> torch.Tensor:
    """Values of the misses, from the rows a key-only storage lookup found.

    A miss storage lacks has a -1 row, which the load skips: its row is left
    unwritten.
    """
    st = storage._state
    rows = slots
    return load_from_flat(st, rows, miss_tids, copy_mode=CopyMode.VALUE)


def _lookup_next_tier(
    next_tier: Optional[Storage],
    miss_keys: torch.Tensor,
    miss_tids: torch.Tensor,
    in_storage: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Ask ``next_tier`` for the misses storage lacks; fill their ``values``.

    Returns the mask (over the misses) of those it had.
    """
    in_next = torch.zeros_like(in_storage)
    if next_tier is None:
        return in_next
    with torch.cuda.nvtx.range("op:flagged_compact"):
        num_asked, asked, (asked_keys, asked_tids) = flagged_compact(
            ~in_storage, [miss_keys, miss_tids]
        )
    if num_asked == 0:
        return in_next
    with torch.cuda.nvtx.range("op:next_tier_find"):
        result = next_tier.find(asked_keys, asked_tids, copy_mode=CopyMode.VALUE)
    found, found_values = result[5], result[7]
    # ``asked`` lists real, distinct positions, so both writes are plain scatters.
    # A boolean-mask index (``asked[found]``) or a Python scalar put through a CUDA
    # index would each wait on the host. ``found`` may come back from a host-side
    # store: it has to sit beside ``asked`` for the scatter.
    found = found.to(in_next.device)
    in_next[asked] = found
    values[asked] = torch.where(found.unsqueeze(1), found_values, values[asked])
    return in_next


def _initialize_rows(
    state,
    initializer: MultiTableInitializer,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
) -> torch.Tensor:
    """Fresh ``[len(keys), max_val_dim]`` rows: initialized embedding, reset optimizer state."""
    n = keys.numel()
    rows = torch.empty(n, max_val_dim, dtype=emb_dtype, device=keys.device)
    with torch.cuda.nvtx.range("op:initializer"):
        initializer(
            rows[:, :max_emb_dim],
            keys,
            table_ids,
            torch.arange(n, dtype=torch.int64, device=keys.device),
        )
    if max_val_dim != max_emb_dim:
        state.optimizer.reset_optimizer_states(
            rows[:, max_emb_dim:], emb_dims=state.table_emb_dims[table_ids]
        )
    return rows


def _insert_into_storage(
    storage: DynamicEmbStorage,
    next_tier: Optional[Storage],
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    values: torch.Tensor,
    scores: Optional[torch.Tensor],
) -> None:
    """Insert new keys into ``storage``; what it evicts for room moves to ``next_tier``."""
    if next_tier is None:
        storage.insert(keys, table_ids, values, scores)
        return
    # NO_EVICTION is deliberately not handled below: its storage never evicts
    # (it expands instead), so there is nothing to spill. The tables module
    # rejects NO_EVICTION together with an ``external_storage``.
    st = storage._state
    (
        slots,
        num_evicted,
        evicted_keys,
        evicted_table_ids,
        evicted_slots,
        _,
    ) = _insert_and_evict_keys(st, keys, table_ids, scores)
    if num_evicted > 0:
        evicted_values = load_from_flat(
            st, evicted_slots, evicted_table_ids, copy_mode=CopyMode.VALUE
        )
        # A key the storage refused (e.g. its bucket is full and the score does
        # not beat any stored one) comes back as an "evicted" row with a negative
        # slot: load_from_flat left its value unwritten, so take the key's own
        # input value instead, as HybridStorage.insert does.
        select_insert_failed_values(evicted_slots, values, evicted_values)
    store_to_flat(st, slots, table_ids, values)
    if num_evicted > 0:
        next_tier.insert(evicted_keys, evicted_table_ids, evicted_values)


# Which path a lookup group takes is decided from host facts only, never from data.
# The masked paths keep every array at the length of the batch instead of narrowing
# it to a count read from the device; only the miss gate (``sf.MissGate``) may still
# wait for the miss count. The legacy paths keep the host-sized arrays that an
# external storage or a user-defined class is written against; DYNAMICEMB_SYNC_FREE=0
# selects them for every group, which also makes them the reference the masked paths
# are tested against.
_SYNC_FREE = os.environ.get("DYNAMICEMB_SYNC_FREE", "1") != "0"
# A masked path holds two [N, value_dim] transients (the loaded values and the
# evicted ones); a call that would need more bytes than this takes the legacy path.
_MAX_BYTES = int(os.environ.get("DYNAMICEMB_SYNC_FREE_MAX_BYTES", str(4 << 30)))


def _in_tree_storage(storage) -> bool:
    """Exact types only: a subclass may change what the masked paths rely on."""
    return type(storage) in (DynamicEmbStorage, HybridStorage)


def _in_tree_cache(cache) -> bool:
    return type(cache) is DynamicEmbCache


def _in_tree_admitter(admitter) -> bool:
    """Whether the admitter may be asked about padded rows.

    A masked path hands ``admit`` arrays that keep the batch's length: the keys it
    is not asking about are replaced by the reserved EMPTY key. Both in-tree
    admitters accept that. A user class has to opt in on the class itself; an
    inherited attribute does not count, because a subclass that overrides ``admit``
    has not been written against EMPTY keys.
    """
    return (
        admitter is None
        or type(admitter) in IN_TREE_ADMITTERS
        or bool(type(admitter).__dict__.get("accepts_empty_keys", False))
    )


def _uses_rng(admitter) -> bool:
    """Whether ``admit`` draws random numbers. Its miss path is then never skipped
    on a timing-dependent poll (see ``sf.miss_gate_mode``): how many values a
    seeded stream has given out must not depend on timing."""
    return admitter is not None and bool(getattr(admitter, "uses_rng", False))


def _masked_fits(n: int, value_dim: int, itemsize: int) -> bool:
    return 2 * n * value_dim * itemsize <= _MAX_BYTES


def _masked_dispatch(
    mode: StorageMode,
    cache: Optional[Cache],
    storage: Storage,
    admitter: Optional[MultiTableAdmitter],
    external_storage: Optional[Storage],
    n: int,
    value_dim: int,
    itemsize: int,
) -> bool:
    """Whether a lookup group takes its masked path.

    ``n`` is the length of the arrays the path allocates its ``[n, value_dim]``
    transients for. A caller that routes on the answer asks once and hands it to the
    path function (``masked=``): asked again for arrays of another length, the size
    limit can answer differently.
    """
    if not (
        _SYNC_FREE
        and _in_tree_admitter(admitter)
        and _masked_fits(n, value_dim, itemsize)
    ):
        return False
    if mode == StorageMode.CACHE:
        return (
            external_storage is None
            and _in_tree_cache(cache)
            and type(storage) is DynamicEmbStorage
        )
    if mode == StorageMode.HBM_DIRECT:
        return type(storage) is DynamicEmbStorage
    return _in_tree_storage(storage)


def _miss_gate_of(storage):
    """The miss gate of the table state a DEFAULT-mode storage looks keys up in
    (the HBM tier's, for a HybridStorage)."""
    return (
        storage._state.miss_gate
        if hasattr(storage, "_state")
        else storage._hbm.miss_gate
    )


def _prefetch_cache_path_masked(
    cache: DynamicEmbCache,
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    valid: Optional[torch.Tensor],
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Cache prefetch that does not narrow its arrays to counts read from the device.

    The steps of ``_prefetch_cache_path_legacy``, over arrays that all keep the
    length of the unique batch. A key that takes no part in a step is replaced by
    the reserved EMPTY key, which every table kernel skips (found=false, index -1)
    without reading its table id; a slot or row that must not be touched is -1.
    ``valid`` marks the real rows of a padded batch (None: every row is real). A
    row that is not real looks like a miss to the lookups, so every miss mask ANDs
    it.

    Only for an in-tree cache and storage, without an external storage, and with an
    admitter that takes EMPTY keys (see ``_masked_dispatch``).

    Returns (slot_indices, non_admitted_mask): the cache slot of every key that is
    in the cache afterwards and -1 for the others, and the bool mask of the keys the
    admitter refused (None if none was asked).
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path"):
        state, device, n = cache._state, unique_keys.device, unique_keys.numel()
        if n == 0:
            return torch.empty(0, dtype=torch.int64, device=device), None
        gate = state.miss_gate

        # 1. Look the keys up in the cache and pin every hit before an insert can
        # evict it. The counter kernel returns on a negative slot before it reads
        # the table id, so misses and padding need no compaction.
        with torch.cuda.nvtx.range("op:cache_lookup"):
            _, founds, cache_indices = cache.lookup(unique_keys, unique_table_ids)
        cache.increment_counter(cache_indices, unique_table_ids)
        is_miss = ~founds if valid is None else valid & ~founds
        with torch.cuda.nvtx.range("op:miss_gate"):
            pending = gate.begin(is_miss, rng=_uses_rng(admitter))

        # 2. Storage is the only place scores live: a scoring lookup of every key,
        # hits included, updates them there without moving a value.
        # ``score_input`` is what storage's score is made from; the insert in step 6
        # takes it as it is, while the lookup builds its own score argument from it.
        score_input = score_source(storage._state, accumulated_frequency, key_scores)
        with torch.cuda.nvtx.range("op:storage_find"):
            mf = storage.lookup_masked(
                unique_keys, unique_table_ids, accumulated_frequency, key_scores
            )

        # 3. Whether anything missed. Gate "always" never skips; "poll" skips only
        # when an already landed count proves it; "read" waits for the count.
        with torch.cuda.nvtx.range("op:miss_gate"):
            need = gate.needed(pending)
        if not need:
            return cache_indices, None  # all hits, and they are pinned already

        # 4. Where each miss lives. A hit is in storage too, but it is neither
        # ``in_storage`` nor ``is_new``: both are ANDed with ``is_miss``.
        in_storage = is_miss & mf.founds
        is_new = is_miss & ~mf.founds

        # Reset the optimizer-state block first, then load the storage rows over
        # it: what a row carries overwrites the reset, whereas a reset after the
        # load would wipe the state just loaded. Only the new rows keep the reset.
        values = torch.empty(n, max_val_dim, dtype=emb_dtype, device=device)
        if max_val_dim != max_emb_dim:
            state.optimizer.reset_optimizer_states(
                values[:, max_emb_dim:],
                emb_dims=state.table_emb_dims[unique_table_ids],
            )
        with torch.cuda.nvtx.range("op:load_from_flat"):
            storage.load_masked(
                # The rows of hits and of padding are -1, which the load skips.
                mf._replace(rows=torch.where(is_miss, mf.rows, -1)),
                unique_table_ids,
                CopyMode.VALUE,
                output=values,
            )

        # 5. Admission: only the new keys are asked about.
        non_admitted_mask = None
        if admitter is None:
            admitted = is_new
        else:
            admit = admitter.admit(
                sf.mask_keys(unique_keys, is_new),
                unique_table_ids,
                accumulated_frequency,
            )
            # The answer for any other row means nothing (a counter hands an EMPTY
            # key back its input frequency), hence the AND.
            admitted, non_admitted_mask = is_new & admit, is_new & ~admit
        with torch.cuda.nvtx.range("op:initializer"):
            sf.initialize_masked(
                initializer,
                values[:, :max_emb_dim],
                unique_keys,
                unique_table_ids,
                admitted,
            )

        # 6. Storage learns the admitted keys, with their score, before the cache
        # sees them. A masked-out key is EMPTY: no row is allocated for it and the
        # store skips it.
        with torch.cuda.nvtx.range("op:storage_insert"):
            storage.insert(
                sf.mask_keys(unique_keys, admitted),
                unique_table_ids,
                values,
                score_input,
            )

        # 7. The cache takes every miss that lives in storage or was admitted. What
        # it evicts comes back with a count that stays on the device: arrays of the
        # batch's length whose entries past the count are inert.
        to_cache = in_storage | admitted
        with torch.cuda.nvtx.range("op:cache_insert_evict"):
            (
                cache_slots,
                _,
                evicted_keys,
                evicted_slots,
                _,
                evicted_tids,
            ) = cache.insert_and_evict(
                sf.mask_keys(unique_keys, to_cache),
                unique_table_ids,
                None,
                async_count=True,
            )

        # 8. The victims are read before their slots are overwritten (stream
        # order), the new rows are stored, then the victims are written back. An
        # evicted entry with a negative slot is a refused insert, not a victim; the
        # write-back leaves it alone.
        with torch.cuda.nvtx.range("op:load_from_flat"):
            evicted_values = load_from_flat(
                state, evicted_slots, evicted_tids, copy_mode=CopyMode.VALUE
            )
        with torch.cuda.nvtx.range("op:store_to_flat"):
            store_to_flat(state, cache_slots, unique_table_ids, values)
        _write_back_evicted(
            storage,
            None,
            evicted_keys,
            evicted_tids,
            evicted_values,
            slots=evicted_slots,
        )

        # 9. Pin the new entries. A hit keeps the slot its lookup found; a refused
        # insert and a padded row keep -1.
        cache.increment_counter(cache_slots, unique_table_ids)
        return (
            torch.where(cache_slots >= 0, cache_slots, cache_indices),
            non_admitted_mask,
        )


def _prefetch_cache_path(
    cache: DynamicEmbCache,
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    external_storage: Optional[Storage] = None,
    key_scores: Optional[torch.Tensor] = None,
    *,
    valid: Optional[torch.Tensor] = None,
    masked: Optional[bool] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Cache prefetch: the masked path when every participant is an in-tree class,
    else the legacy one, whose host-sized arrays an external storage or a user
    class is written against.

    Returns (slot_indices, non_admitted). The second value is the bool mask of the
    refused keys from the masked path and their positions from the legacy one.
    ``valid`` marks the real rows of padded unique arrays (None: every row is real);
    only the masked path takes padded arrays. ``masked`` is the decision of a
    caller that has already routed on it (``_masked_dispatch`` for the arrays it
    built); None takes the decision here, from the arrays given.
    """
    if masked is None:
        masked = _masked_dispatch(
            StorageMode.CACHE,
            cache,
            storage,
            admitter,
            external_storage,
            unique_keys.numel(),
            max_val_dim,
            emb_dtype.itemsize,
        )
    if masked:
        return _prefetch_cache_path_masked(
            cache,
            storage,
            unique_keys,
            unique_table_ids,
            valid,
            max_emb_dim,
            max_val_dim,
            emb_dtype,
            initializer,
            accumulated_frequency,
            admitter,
            key_scores=key_scores,
        )
    assert valid is None, "padded unique arrays need a masked prefetch path"
    return _prefetch_cache_path_legacy(
        cache,
        storage,
        unique_keys,
        unique_table_ids,
        max_emb_dim,
        max_val_dim,
        emb_dtype,
        initializer,
        evict_strategy,
        accumulated_frequency,
        admitter,
        external_storage,
        key_scores=key_scores,
    )


def _prefetch_cache_path_legacy(
    cache: DynamicEmbCache,
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    external_storage: Optional[Storage] = None,
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Cache prefetch with counter protection and overflow buffer.

    The host-count path: it reads from the device how many keys each step narrows
    to and works on arrays of exactly that size. It serves an external storage,
    user-defined classes and DYNAMICEMB_SYNC_FREE=0.

    ``storage`` is the tier that owns every key's score; the cache above it
    ranks by timestamp only, and holds the freshest value of a key until it is
    written back. When ``external_storage`` is given (a
    score-less store below ``storage``, e.g. a PS) it holds the keys ``storage`` has
    evicted: keys missing from ``storage`` are fetched from it, and what
    ``storage`` evicts is spilled to it.

    Only admitted keys are inserted into the cache.  Non-admitted keys get
    slot_indices = -1 and their positions (in the unique_keys array) are
    returned so the forward pass can lazily initialize their embeddings.

    Index spaces: ``unique_*`` arrays are indexed by position in the unique
    batch; everything after the cache lookup is over the *misses* only, indexed
    by miss number, and ``miss_pos`` maps a miss number back to its position
    in the unique batch.

    Returns (slot_indices, non_admitted_positions).
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path"):
        device = unique_keys.device
        state = cache._state

        if unique_keys.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, None

        # 1. Lookup with overflow fallback; pin what is found
        with torch.cuda.nvtx.range("op:cache_lookup"):
            _, founds, cache_indices = cache.lookup(unique_keys, unique_table_ids)
        slot_indices = cache_indices.clone()
        # Pin every row at once: the counter kernel returns on a negative slot
        # (a miss) before it reads the table id, and the slots of the hits differ.
        cache.increment_counter(cache_indices, unique_table_ids)

        # 2. Storage is the only place scores live: a key-only lookup of *every*
        # key -- hits included, and even when nothing missed -- updates its score
        # there without moving a value. The cache only ranks by timestamp.
        # ``score_input`` is what the storage's score is made from, chosen once
        # here and carried with the keys from now on; ``accumulated_frequency``
        # is only for admission below.
        score_input = score_source(storage._state, accumulated_frequency, key_scores)
        with torch.cuda.nvtx.range("op:storage_find"):
            # Only the lookup is wanted here, not the compaction of its misses.
            (_, _, _, _, _, st_founds, _, st_slots) = _find_keys(
                storage._state,
                unique_keys,
                unique_table_ids,
                score_input=score_input,
                compact=False,
            )

        # 3. The misses, with everything later steps need about them
        with torch.cuda.nvtx.range("op:flagged_compact"):
            (
                num_miss,
                miss_pos,
                (
                    miss_keys,
                    miss_tids,
                    miss_freq,
                    miss_score_input,
                    miss_slots,
                ),
            ) = flagged_compact(
                ~founds,
                [
                    unique_keys,
                    unique_table_ids,
                    accumulated_frequency,
                    score_input,
                    st_slots,
                ],
            )
        if num_miss == 0:
            return slot_indices, None

        # 4. Where each miss lives: storage, the tier below it, or nowhere (new).
        # ``values`` holds the first two's rows; new ones are filled in at 6.
        in_storage = st_founds[miss_pos]
        values = _load_storage_values(storage, miss_slots, miss_tids)
        in_next = _lookup_next_tier(
            external_storage, miss_keys, miss_tids, in_storage, values
        )

        # 5. Admit the new ones (a key from the tier below was admitted before)
        is_new = ~(in_storage | in_next)
        admitted = is_new.clone()
        non_admitted_positions: Optional[torch.Tensor] = None
        if admitter is not None:
            with torch.cuda.nvtx.range("op:flagged_compact"):
                (
                    num_new,
                    new_rows,
                    (new_keys, new_tids, new_freq, new_pos),
                ) = flagged_compact(is_new, [miss_keys, miss_tids, miss_freq, miss_pos])
            if num_new > 0:
                admit_mask = admitter.admit(new_keys, new_tids, new_freq)
                admitted[new_rows] = admit_mask
                with torch.cuda.nvtx.range("op:flagged_compact"):
                    num_refused, _, (refused_pos,) = flagged_compact(
                        ~admit_mask, [new_pos]
                    )
                if num_refused > 0:
                    # A refused key was a miss, so its slot is already -1, and it
                    # stays out of ``to_cache`` below.
                    non_admitted_positions = refused_pos

        to_cache = in_storage | in_next | admitted
        with torch.cuda.nvtx.range("op:flagged_compact"):
            (
                num_cache,
                into_cache,
                (cache_keys, cache_tids, cache_pos),
            ) = flagged_compact(to_cache, [miss_keys, miss_tids, miss_pos])
        if num_cache == 0:
            return slot_indices, non_admitted_positions

        # 6. Initialize the admitted-new rows, and put everything storage lacks
        # into it -- with its score -- before the cache sees it.
        with torch.cuda.nvtx.range("op:flagged_compact"):
            num_adm, adm_rows, (adm_keys, adm_tids) = flagged_compact(
                admitted, [miss_keys, miss_tids]
            )
        if num_adm > 0:
            values[adm_rows] = _initialize_rows(
                state,
                initializer,
                adm_keys,
                adm_tids,
                max_emb_dim,
                max_val_dim,
                emb_dtype,
            )
        with torch.cuda.nvtx.range("op:flagged_compact"):
            num_st, into_storage, (st_keys, st_tids, st_score_input) = flagged_compact(
                to_cache & ~in_storage, [miss_keys, miss_tids, miss_score_input]
            )
        if num_st > 0:
            with torch.cuda.nvtx.range("op:storage_insert"):
                _insert_into_storage(
                    storage,
                    external_storage,
                    st_keys,
                    st_tids,
                    _select(values, into_storage, num_st == values.size(0)),
                    st_score_input,
                )

        # 7. Bring the keys into the cache, ranked by timestamp only
        with torch.cuda.nvtx.range("op:cache_insert_evict"):
            (
                cache_slots,
                num_evicted,
                evicted_keys,
                evicted_slots,
                _,
                evicted_tids,
            ) = cache.insert_and_evict(cache_keys, cache_tids, None)

        # 8. What it evicted: read before the slots are overwritten, then write back
        with torch.cuda.nvtx.range("op:load_from_flat"):
            evicted_values = load_from_flat(
                state, evicted_slots, evicted_tids, copy_mode=CopyMode.VALUE
            )
        with torch.cuda.nvtx.range("op:store_to_flat"):
            store_to_flat(
                state,
                cache_slots,
                cache_tids,
                _select(values, into_cache, num_cache == values.size(0)),
            )
        if num_evicted > 0:
            _write_back_evicted(
                storage,
                external_storage,
                evicted_keys,
                evicted_tids,
                evicted_values,
                slots=evicted_slots,
            )

        # 9. Counter & slot mapping
        cache.increment_counter(cache_slots, cache_tids)
        slot_indices[cache_pos] = cache_slots
        return slot_indices, non_admitted_positions


def _prefetch_hbm_direct_path_masked(
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    valid: Optional[torch.Tensor],
    max_emb_dim: int,
    max_val_dim: int,
    initializer: MultiTableInitializer,
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """HBM-direct prefetch in mask form: no array is sized by a device count.

    The miss path runs over every row. A key that must not take part (a hit, a
    key the admitter refused, a padding row) is not taken out of the arrays: its
    key is replaced by the reserved all-ones key, which the table kernels skip
    without reading its table id, or its slot and row stay -1, which the load,
    store and counter kernels skip. Nothing here blocks on a count from the
    device, except for the miss gate: in mode ``read`` it waits for the number
    of misses, to decide whether the path is launched at all (``poll`` only takes
    a count that has already landed, ``always`` does not look at one).

    ``valid`` marks the real rows (``None``: every row is real). ``unique_keys``,
    ``unique_table_ids`` and the optional ``accumulated_frequency`` and
    ``key_scores`` have one entry per row; a table kernel reads a lane's score
    before it looks at its key.

    Returns ``(slot_indices, non_admitted_mask, value_row_indices)`` like the
    host-sized form, except that the refused keys are a bool mask over the rows
    (``True``: refused) and not their positions. The mask is ``None`` without an
    admitter, and when the miss path did not run.
    """
    with torch.cuda.nvtx.range("_prefetch_hbm_direct_path"):
        state = storage._state
        device = unique_keys.device
        n = unique_keys.numel()

        if n == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, None, empty

        gate = state.miss_gate

        # 1. Look every key up (a hit's score is updated), then pin the hits before
        # anything is inserted, so that no insert of this batch can evict them. A key
        # that was not found has slot -1: the counter skips it before it reads the
        # table id, so the slots go in as they are.
        with torch.cuda.nvtx.range("op:storage_find"):
            (
                _,
                _,
                _,
                _,
                score_value,
                founds,
                _,
                rows_found,
                slots_found,
            ) = _find_keys(
                state,
                unique_keys,
                unique_table_ids,
                score_input=score_source(state, accumulated_frequency, key_scores),
                return_slots=True,
                compact=False,
            )
        storage.increment_counter(slots_found, unique_table_ids)

        # 2. The misses. To the lookup a padding row is a miss like any other, so
        # ``valid`` is ANDed in. The gate only decides whether the path below is
        # launched: with no miss it is a no-op.
        is_miss = ~founds if valid is None else valid & ~founds
        with torch.cuda.nvtx.range("op:miss_gate"):
            pending = gate.begin(is_miss, rng=_uses_rng(admitter))
            need = gate.needed(pending)
        if not need:
            return slots_found, None, rows_found

        # 3. Admission. The admitter sees only the keys that missed, the others are
        # the reserved key. What it returns says nothing about the other lanes (the
        # frequency counter hands a reserved key's input frequency back, which can
        # pass its threshold), so both masks are restricted to the misses here.
        non_admitted_mask: Optional[torch.Tensor] = None
        if admitter is None:
            admitted = is_miss
        else:
            admit = admitter.admit(
                sf.mask_keys(unique_keys, is_miss),
                unique_table_ids,
                accumulated_frequency,
            )
            admitted = is_miss & admit
            non_admitted_mask = is_miss & ~admit

        # 4. Fresh rows. This block is a temporary: nothing is loaded into it and it
        # is stored only for the admitted keys, so the whole optimizer-state region
        # can be reset.
        init_values = torch.empty(n, max_val_dim, dtype=state.emb_dtype, device=device)
        with torch.cuda.nvtx.range("op:initializer"):
            sf.initialize_masked(
                initializer,
                init_values[:, :max_emb_dim],
                unique_keys,
                unique_table_ids,
                admitted,
            )
        if max_val_dim != max_emb_dim:
            state.optimizer.reset_optimizer_states(
                init_values[:, max_emb_dim:],
                emb_dims=state.table_emb_dims[unique_table_ids],
            )

        # 5. Insert the admitted keys, one lane per row. The score tensor is as long
        # as the key array: the kernel reads a lane's score before it looks at its
        # key.
        score_arg = get_insert_score_arg(state, n, device, score_value)
        insert_keys = sf.mask_keys(unique_keys, admitted)
        with torch.cuda.nvtx.range("op:storage_insert"):
            # HBM-direct is a last tier: this insert may evict, and the victims are
            # dropped here (unlike the cache path, which spills to storage). Retain
            # them when configured; their count stays on the device.
            if EvictedItemMode.RETAIN_KEY in state.evicted_item_mode:
                (
                    new_rows,
                    num_evicted,
                    evicted_keys,
                    evicted_table_ids,
                    new_slots,
                ) = state.key_index_map.insert(
                    insert_keys,
                    unique_table_ids,
                    score_arg,
                    collect_evicted=True,
                    return_slots=True,
                )
                _append_evicted(state, evicted_keys, evicted_table_ids, num_evicted)
            else:
                new_rows, new_slots = state.key_index_map.insert(
                    insert_keys, unique_table_ids, score_arg, return_slots=True
                )
        # A key that was not inserted (a hit, a refused key, a padding row) has row
        # -1 and slot -1: the store and the counter skip it. The value buffer is
        # addressed by row, the counter by slot.
        with torch.cuda.nvtx.range("op:store_to_flat"):
            store_to_flat(state, new_rows, unique_table_ids, init_values)
        storage.increment_counter(new_slots, unique_table_ids)

        # 6. A hit keeps the slot and row its lookup found, an inserted key takes the
        # new ones, any other key (refused, not inserted, padding) stays at -1.
        slot_indices = torch.where(founds, slots_found, new_slots)
        rows = (
            slot_indices
            if not state.key_index_map.use_rows_
            else torch.where(founds, rows_found, new_rows)
        )
        return slot_indices, non_admitted_mask, rows


def _prefetch_hbm_direct_path_legacy(
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    initializer: MultiTableInitializer,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """HBM-direct prefetch with counter protection.

    Only admitted keys are inserted.  Non-admitted keys get slot_indices = -1
    and their positions are returned for lazy initialization at forward time.

    Returns ``(slot_indices, non_admitted_positions, value_row_indices)``. The
    two index spaces are kept apart because they only coincide by accident:
    ``slot_indices`` addresses the hash table (and the ref counter beside it),
    ``value_row_indices`` addresses the flat value buffer. A storage stores a
    row per key, so they differ: rows are handed out densely while slots come
    from the key's hash, and the key map of a table that never evicts is sized
    ``1 / load factor`` times the value buffer, so a slot can point past the end
    of it. A table whose row is its slot returns the slot tensor itself as the
    rows, so the caller always gets something it can address values with.

    This is the host-sized form: it reads the number of missing keys and works
    on arrays compacted to it, over exact-length unique arrays. It runs for what
    the mask form does not take (a user admitter, a batch too large for the
    mask form's temporaries, the kill switch) and is the oracle the mask form is
    tested against.
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
                _,
                indices,
                slot_indices,
            ) = _find_keys(
                state,
                unique_keys,
                unique_table_ids,
                score_input=score_source(state, accumulated_frequency, key_scores),
                return_slots=True,
            )

        # ``indices`` are rows for a table that stores them, hash slots otherwise,
        # and ``slot_indices`` the hash slots either way (the same tensor when the
        # row is the slot). The ref counter is keyed by slot. A key that was not
        # found has slot -1, which the counter skips before it reads the table id,
        # so every key goes in as it is.
        storage.increment_counter(slot_indices, unique_table_ids)

        # Values are addressed by row, the counter by slot: a storage stores a row
        # per key, so these are different tensors and both are kept up to date.
        rows = indices

        if h_num_missing == 0:
            return slot_indices, None, rows

        # Determine admission for missing keys
        non_admitted_positions: Optional[torch.Tensor] = None

        if admitter is not None:
            freq_for_admission = (
                accumulated_frequency[missing_indices]
                if accumulated_frequency is not None
                else None
            )
            admit_mask = admitter.admit(
                missing_keys, missing_table_ids, freq_for_admission
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

            score_arg = get_insert_score_arg(state, n_admitted, device, admitted_scores)
            with torch.cuda.nvtx.range("op:storage_insert"):
                # HBM-direct is a last tier: this insert may evict, and the
                # victims are dropped here (unlike the cache path, which spills
                # to storage). Retain them when configured -- this is the
                # forward-path analogue of the _insert_key_values collection.
                if EvictedItemMode.RETAIN_KEY in state.evicted_item_mode:
                    (
                        new_indices,
                        num_evicted,
                        evicted_keys,
                        evicted_table_ids,
                        new_slots,
                    ) = state.key_index_map.insert(
                        admitted_keys,
                        admitted_tids,
                        score_arg,
                        collect_evicted=True,
                        return_slots=True,
                    )
                    _append_evicted(state, evicted_keys, evicted_table_ids, num_evicted)
                else:
                    new_indices, new_slots = state.key_index_map.insert(
                        admitted_keys,
                        admitted_tids,
                        score_arg,
                        return_slots=True,
                    )
            # The insert returns rows, and the slots as well. Keep both: the
            # counter is indexed by slot, the value buffer by row.
            new_rows = new_indices
            with torch.cuda.nvtx.range("op:store_to_flat"):
                store_to_flat(state, new_rows, admitted_tids, init_values)

            # A key the insert refused has slot -1, which the counter skips.
            storage.increment_counter(new_slots, admitted_tids)

            slot_indices[admitted_unique_positions] = new_slots
            rows[admitted_unique_positions] = new_rows

        # A non-admitted key was a miss, so its slot and row are already -1, and
        # it is not among ``admitted_unique_positions``.
        return slot_indices, non_admitted_positions, rows


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
    key_scores: Optional[torch.Tensor] = None,
    *,
    valid: Optional[torch.Tensor] = None,
    masked: Optional[bool] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """HBM-direct prefetch with counter protection: the mask form or the
    host-sized form, chosen from host facts only (never from data).

    The two return the refused keys differently (a bool mask over the rows, or
    the positions of the refused keys), so whoever stores the result has to know
    which one ran: that is one decision per batch, ``_masked_dispatch``. Its
    memory guard depends on the size of the batch, so a caller that decides on a
    size other than the length of the arrays it passes here has to hand its
    decision down as ``masked``: near the guard the two can differ. Left out, it
    is taken here, from the length of ``unique_keys``.

    ``valid`` marks the real rows of padded unique arrays. Only the mask form
    takes it: the host-sized form works on exact-length arrays and refuses it.

    Returns ``(slot_indices, non_admitted, value_row_indices)``: ``non_admitted``
    is a bool mask over the rows from the mask form, and the positions of the
    refused keys (``None`` if there were none) from the host-sized form.
    """
    if masked is None:
        masked = _masked_dispatch(
            StorageMode.HBM_DIRECT,
            None,  # cache
            storage,
            admitter,
            None,  # external_storage
            unique_keys.numel(),
            max_val_dim,
            storage.embedding_dtype().itemsize,
        )
    if masked:
        return _prefetch_hbm_direct_path_masked(
            storage,
            unique_keys,
            unique_table_ids,
            valid,
            max_emb_dim,
            max_val_dim,
            initializer,
            accumulated_frequency,
            admitter,
            key_scores=key_scores,
        )
    assert valid is None, (
        "padded unique arrays reached the host-sized HBM-direct prefetch, "
        "which takes exact-length arrays"
    )
    return _prefetch_hbm_direct_path_legacy(
        storage,
        unique_keys,
        unique_table_ids,
        max_emb_dim,
        max_val_dim,
        initializer,
        evict_strategy,
        accumulated_frequency,
        admitter,
        key_scores=key_scores,
    )


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
    outstanding_keys_ref: Optional[OutstandingKeys] = None,
    external_storage: Optional[Storage] = None,
    scores: Optional[torch.Tensor] = None,
    table_gate: Optional[Callable[[], ContextManager]] = None,
    *,
    table_key_bounds: Optional[torch.Tensor] = None,
) -> PrefetchState:
    """Unified prefetch for all storage types (cache, HBM-direct, generic).

    Returns a PrefetchState containing unique embeddings + optimizer states
    and the metadata needed by forward/backward.

    ``scores`` are one int64 per key, for tables that take a score per key. They
    follow the keys through the dedup on a channel of their own, beside the
    frequency counters: where a key repeats in the batch, the largest of its
    scores is the one the table gets.

    ``table_gate`` is what an erase running beside the training takes too (see
    ``async_erase.TableAccessGate``). Prefetch holds it while it reads or changes the
    tables -- from the expansion through the find and the counter increment -- so
    that a key it found is pinned before an erase can look at it.

    ``table_key_bounds`` (CPU int64, one per table) is what the caller knows, without
    the device, of how many keys each table's part of the batch can have: the key
    counts of its features. A front end that does not read the dedup count back
    sizes the growth of the storage from it; this one has the exact count of unique
    keys per table, which is never above the bound, and uses that.
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

        # The sizes the expansion below decides from are read back without waiting, on
        # the stream that inserts into the storage: this one, for a cache's storage
        # and an HBM storage. A host or hybrid storage is inserted into by the
        # forward, which takes its snapshot there (``note_forwarded``).
        if hasattr(storage, "request_size_snapshot") and (
            caching or _is_hbm_storage(storage)
        ):
            storage.request_size_snapshot()

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
        unique_scores = (
            None
            if scores is None
            else largest_score_per_unique_key(
                scores, reverse_indices, unique_keys.numel()
            )
        )

        num_prefetched_keys = unique_keys.numel()

        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range,
            unique_keys.numel(),
        )

        slot_indices = None
        non_admitted_positions = None
        value_row_indices = None
        key_map_generation = 0
        if caching:
            storage_mode = StorageMode.CACHE
        elif _is_hbm_storage(storage):
            storage_mode = StorageMode.HBM_DIRECT
        else:
            storage_mode = StorageMode.DEFAULT
        # The path of the group is decided once, from host facts, for the arrays as
        # they are now: exactly the unique count long, every row real (no ``valid``
        # mask). The cache path takes the decision as an argument; a DEFAULT-mode
        # batch carries it in its state, because its lookup runs in the forward.
        masked = _masked_dispatch(
            storage_mode,
            cache,
            storage,
            admitter,
            external_storage,
            unique_keys.numel(),
            max_val_dim,
            emb_dtype.itemsize,
        )

        outstanding_entry = None
        with table_section(table_gate):
            if hasattr(storage, "expand_if_need"):
                storage.expand_if_need(
                    unique_size_per_table,
                    inserts_in_forward=(storage_mode == StorageMode.DEFAULT),
                )

            if storage_mode == StorageMode.CACHE:
                if outstanding_keys_ref is not None:
                    # The unique count is known here, so this is checked at once.
                    outstanding_entry = outstanding_keys_ref.add(
                        num_prefetched_keys, None, num_prefetched_keys
                    )
                    outstanding_keys_ref.check(cache._state.capacity)
                (
                    slot_indices,
                    non_admitted_positions,
                ) = _prefetch_cache_path(
                    cache,
                    storage,
                    unique_keys,
                    unique_table_ids,
                    max_emb_dim,
                    max_val_dim,
                    emb_dtype,
                    initializer,
                    evict_strat,
                    lfu_accumulated_frequency,
                    admitter,
                    external_storage,
                    key_scores=unique_scores,
                    valid=None,
                    masked=masked,
                )
                # The cache addresses its own value buffer by cache slot: a cache is
                # never NO_EVICTION (it enables overflow, which NO_EVICTION rejects
                # at construction), so the two index spaces coincide here.
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
                    key_scores=unique_scores,
                )
                key_map_generation = storage._state.key_map_generation

        # A masked path returns the refused keys as a bool mask over all unique keys,
        # a legacy path as their int64 positions. What came back says which one ran,
        # so the label cannot disagree with the path.
        non_admitted_mask = None
        if (
            non_admitted_positions is not None
            and non_admitted_positions.dtype == torch.bool
        ):
            non_admitted_mask, non_admitted_positions = non_admitted_positions, None

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
            non_admitted_mask=non_admitted_mask,
            generic_masked=masked and storage_mode == StorageMode.DEFAULT,
            unique_scores=unique_scores,
            num_prefetched_keys=num_prefetched_keys,
            outstanding_keys_ref=outstanding_keys_ref,
            outstanding_entry=outstanding_entry,
            table_key_bounds=unique_size_per_table,
            value_row_indices=value_row_indices,
            key_map_generation=key_map_generation,
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
    key_scores: Optional[torch.Tensor] = None,
    *,
    valid: Optional[torch.Tensor] = None,
    masked: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Standalone forward for generic (DEFAULT) storage mode.

    ``masked`` picks the form: the mask form of an in-tree storage and
    admitter, or the legacy form over exact-length arrays for anything else.
    The caller passes the decision ``dynamicemb_prefetch`` recorded in
    ``PrefetchState.generic_masked``, so that a step is never prefetched for
    one form and run in the other. ``valid`` belongs to the mask form.

    Returns (unique_values, persisted). ``persisted`` says which unique keys
    ``storage`` holds after this forward (lookup hit or admitted insert), for
    the backward to re-insert: a bool mask over the unique rows (``None``:
    every row) in the mask form, an int64 vector of row indices in the legacy
    form.
    """
    if masked:
        return _generic_forward_path_masked(
            storage,
            optimizer,
            unique_keys,
            unique_table_ids,
            valid,
            max_emb_dim,
            max_val_dim,
            emb_dtype,
            initializer,
            accumulated_frequency,
            admitter,
            key_scores,
        )
    return _generic_forward_path_legacy(
        storage,
        optimizer,
        unique_keys,
        unique_table_ids,
        max_emb_dim,
        max_val_dim,
        emb_dtype,
        initializer,
        evict_strategy,
        accumulated_frequency,
        admitter,
        key_scores,
    )


def _generic_forward_path_masked(
    storage: Storage,
    optimizer: BaseDynamicEmbeddingOptimizer,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    valid: Optional[torch.Tensor],
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Generic (DEFAULT) forward of an in-tree storage, in mask form.

    Does what ``_generic_forward_path_legacy`` does (find, admit, initialize,
    insert), but over every unique key at once. A key a step must leave alone
    (a hit, a refused key, padding) is not compacted away, which would read
    the count of the survivors back to the host; it is hidden from the table
    operations as the reserved EMPTY key, which they all skip. Apart from the
    miss gate, which may skip the miss path once it knows nothing missed, the
    launches do not depend on the data.

    ``valid`` marks the real rows of a padded batch; ``None`` means every row
    is real. The padding is hidden from every table operation here, whatever
    its producer left in it, so a padding key is not found by the lookup, looks
    like a miss, and every miss mask is ANDed with ``valid``.

    Returns (unique_values, persisted_mask). ``persisted_mask`` is the bool
    mask over the unique rows of the keys ``storage`` holds after this forward
    (lookup hit or admitted insert), which the backward re-inserts: never true
    for padding. ``None`` (only for exact arrays, ``valid=None``) means every
    row is.
    """
    with torch.cuda.nvtx.range("_generic_forward_path"):
        n, device = unique_keys.numel(), unique_keys.device
        unique_values = torch.empty(n, max_val_dim, dtype=emb_dtype, device=device)
        if n == 0:
            return unique_values, None

        if valid is not None:
            # A key the table holds, left in the padding, would be looked up
            # (its score bumped) and written back by the backward with a row
            # nobody computed. The padding of a conforming batch is EMPTY
            # already, so this changes nothing for it.
            unique_keys = sf.mask_keys(unique_keys, valid)
        gate = _miss_gate_of(storage)
        with torch.cuda.nvtx.range("op:storage_find"):
            # A scoring lookup of every key, as the legacy find: it updates the
            # score of each key it finds. The rows are loaded below.
            mf = storage.lookup_masked(
                unique_keys,
                unique_table_ids,
                accumulated_frequency,
                key_scores=key_scores,
            )
        is_miss = ~mf.founds if valid is None else valid & ~mf.founds
        with torch.cuda.nvtx.range("op:miss_gate"):
            miss_path = gate.needed(gate.begin(is_miss, rng=_uses_rng(admitter)))

        if miss_path and max_val_dim != max_emb_dim:
            # Reset the state of every row first and load over it: the state a
            # found row loads replaces the reset one, a missing row keeps it.
            # Resetting after the load would wipe the state just loaded.
            optimizer.reset_optimizer_states(
                unique_values[:, max_emb_dim:],
                emb_dims=storage.embedding_dims(on_device=True)[unique_table_ids],
            )
        with torch.cuda.nvtx.range("op:load_from_flat"):
            storage.load_masked(
                mf, unique_table_ids, CopyMode.VALUE, output=unique_values
            )
        if not miss_path:
            # Nothing missed, so every real key is persisted.
            return unique_values, valid

        non_admitted_initializer = (
            admitter.non_admitted_initializer if admitter is not None else None
        )
        if admitter is None:
            admitted, refused = is_miss, None
        else:
            # The admitter answers for every lane, hidden ones included (a
            # reserved key can pass a frequency threshold), so only a key that
            # missed can be admitted or refused.
            admit = admitter.admit(
                sf.mask_keys(unique_keys, is_miss),
                unique_table_ids,
                accumulated_frequency,
            )
            admitted, refused = is_miss & admit, is_miss & ~admit

        emb = unique_values[:, :max_emb_dim]
        # When the admitter has an initializer of its own, the table's covers
        # only the admitted rows and the admitter's the refused ones; otherwise
        # the table's covers every missing row, the refused ones included.
        with torch.cuda.nvtx.range("op:initializer"):
            sf.initialize_masked(
                initializer,
                emb,
                unique_keys,
                unique_table_ids,
                is_miss if non_admitted_initializer is None else admitted,
            )
        if non_admitted_initializer is not None:
            with torch.cuda.nvtx.range("op:initializer"):
                sf.initialize_masked(
                    non_admitted_initializer,
                    emb,
                    unique_keys,
                    unique_table_ids,
                    refused,
                )

        with torch.cuda.nvtx.range("op:storage_insert"):
            # The score the find used, at full length: the insert reads
            # scores[i] before it checks the key, so a shorter tensor (the
            # admitted keys' scores alone) would be read out of bounds.
            storage.insert(
                sf.mask_keys(unique_keys, admitted),
                unique_table_ids,
                unique_values,
                mf.score_in,
            )
        # Without an admitter every miss was inserted, so every real key is
        # persisted; with one, the keys it refused are not.
        return unique_values, (valid if admitter is None else mf.founds | admitted)


def _generic_forward_path_legacy(
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
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standalone forward for generic (DEFAULT) storage mode.

    Does storage.find + admission + init + insert
    in a single pass with no PrefetchState.

    Works on exact-length arrays and reads host counts, so it also serves what
    the mask form (``_generic_forward_path_masked``) cannot: an external or
    user-defined storage or admitter, a batch over the mask form's memory
    limit, and ``DYNAMICEMB_SYNC_FREE=0``.

    Returns (unique_values, persisted_unique_indices). The latter is an int64
    vector of row indices into the unique-key tensors for keys that are stored
    in ``storage`` after this forward (lookup hit or admitted insert). Used in
    backward to ``storage.insert(..., preserve_existing=True)`` only for those
    rows, without compacting via a boolean mask again.
    """
    with torch.cuda.nvtx.range("_generic_forward_path"):
        device = unique_keys.device
        n = unique_keys.numel()

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
                # Only when there are some: an external store written against
                # the interface before per-key scores has no such parameter.
                **({} if key_scores is None else {"key_scores": key_scores}),
            )

        if h_num_missing == 0:
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
        if admitter is None:
            # Every missing key was inserted above, so every key is persisted
            # and there is nothing to compact.
            return unique_values, torch.arange(n, device=device, dtype=torch.int64)
        key_persisted = founds.clone()
        # index_fill_, not ``key_persisted[positions_in_unique] = True``: a
        # Python scalar put through a CUDA index tensor is a blocking H2D copy.
        key_persisted.index_fill_(0, positions_in_unique, True)
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
                        # Value buffer: address by row (the slot, for a cache).
                        prefetch_state.value_row_indices,
                        prefetch_state.unique_table_ids,
                        copy_mode=CopyMode.EMBEDDING,
                    )
                if prefetch_state.non_admitted_mask is not None:
                    # The prefetch had an in-tree admitter (a host fact); which
                    # keys it refused is device data, so the rows come as a
                    # mask and the initializer skips the rest.
                    non_admitted_initializer = (
                        admitter.non_admitted_initializer
                        if admitter is not None
                        else None
                    )
                    with torch.cuda.nvtx.range("op:initializer"):
                        # Whichever initializer owns rejected rows was settled
                        # at construction, so this is one call either way.
                        sf.initialize_masked(
                            non_admitted_initializer or initializer,
                            unique_embs[:, :max_emb_dim],
                            prefetch_state.unique_keys,
                            prefetch_state.unique_table_ids,
                            prefetch_state.non_admitted_mask,
                        )
                elif prefetch_state.non_admitted_positions is not None:
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
                persisted_mask = None
                generic_masked = False
            else:
                assert prefetch_state.storage_mode == StorageMode.DEFAULT
                evict_strat = (
                    EvictStrategy(evict_strategy.value) if evict_strategy else None
                )
                # The form the prefetch decided on, so that a step never runs
                # in a form its arrays were not made for.
                generic_masked = prefetch_state.generic_masked
                unique_values, persisted = _generic_forward_path(
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
                    key_scores=prefetch_state.unique_scores,
                    valid=prefetch_state.unique_valid,
                    masked=generic_masked,
                )
                # A bool mask over the unique rows from the mask form, row
                # indices from the legacy one.
                persisted_mask = persisted if generic_masked else None
                persisted_unique_indices = None if generic_masked else persisted
                # The generic path loads CopyMode.VALUE, so a row here is
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
            ctx.key_map_generation = prefetch_state.key_map_generation
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
            ctx.generic_masked = generic_masked
            ctx.persisted_mask = persisted_mask
            ctx.persisted_unique_indices = persisted_unique_indices
            ctx.outstanding_keys_ref = prefetch_state.outstanding_keys_ref
            ctx.num_prefetched_keys = prefetch_state.num_prefetched_keys

            # Recover outstanding count at end of forward so it is decremented
            # even when backward is not run, avoiding overflow.
            if prefetch_state.outstanding_keys_ref is not None:
                prefetch_state.outstanding_keys_ref.release(
                    prefetch_state.outstanding_entry
                )

            # A storage that is inserted into here, not by the prefetch, has the keys
            # of this batch now: tell its growth estimate, which takes a snapshot of
            # the sizes on this stream, after the inserts.
            if (
                prefetch_state.storage_mode == StorageMode.DEFAULT
                and prefetch_state.table_key_bounds is not None
                and hasattr(storage, "note_forwarded")
            ):
                storage.note_forwarded(prefetch_state.table_key_bounds)

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
                    # A later batch's prefetch may have rehashed the table
                    # since this one's prefetch (prefetch pipelines, several
                    # forwards before their backwards).
                    if ctx.key_map_generation != state.key_map_generation:
                        ctx.slot_indices = _slots_after_rehash(
                            state, ctx.unique_keys, unique_table_ids, ctx.slot_indices
                        )
                        ctx.key_map_generation = state.key_map_generation
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

                    counter_owner = (
                        cache if ctx.storage_mode == StorageMode.CACHE else storage
                    )

                    # Ref counter sits beside the hash table -> slots.
                    counter_owner.decrement_counter(
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
                            sf.padded_optimizer_table_ids(unique_table_ids),
                            table_emb_dims,
                            ctx.max_emb_dim,
                            ctx.max_val_dim,
                            all_dims_vec4,
                        )
                    if ctx.generic_masked:
                        # The same re-insert as below over every unique row: a
                        # key that is not persisted is hidden as EMPTY instead
                        # of gathered out, so no count is needed to size it.
                        keys_c = (
                            ctx.unique_keys
                            if ctx.persisted_mask is None
                            else sf.mask_keys(ctx.unique_keys, ctx.persisted_mask)
                        )
                        if keys_c.numel() > 0:
                            with torch.cuda.nvtx.range("op:storage_insert"):
                                storage.insert(
                                    keys_c,
                                    unique_table_ids,
                                    ctx.unique_values,
                                    preserve_existing=True,
                                )
                    else:
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
