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
from typing import Callable, ContextManager, Optional, Tuple

import torch
from dynamicemb import sync_free as sf
from dynamicemb.async_erase import table_section
from dynamicemb.dynamicemb_config import EvictedItemMode
from dynamicemb.embedding_admission import initializer_needs_dense_rows
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.key_value_table import (
    Cache,
    DynamicEmbCache,
    DynamicEmbStorage,
    DynamicEmbTableState,
    Storage,
    _append_evicted,
    _find_keys,
    _insert_and_evict_keys,
    _select_refused_values,
    _write_back_evicted,
    eval_lookup,
    get_insert_score_arg,
    load_from_flat,
    score_source,
    store_to_flat,
)
from dynamicemb.lookup_layout import EmbeddingLayout
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import (
    CopyMode,
    MaskedFind,
    MultiTableAdmitter,
    OldContractStorage,
    as_padded,
    has_padded_contract,
)
from dynamicemb_extensions import (
    EvictStrategy,
    expand_table_ids_cuda,
    gather_embedding,
    gather_embedding_pooled,
    get_table_range,
    reduce_grads,
    segmented_unique_cuda,
)


def segmented_unique(
    keys: torch.Tensor,
    segment_range: torch.Tensor,
    evict_strategy: Optional[EvictStrategy] = None,
    frequency_counts: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor
]:
    """
    Perform segmented unique operation on keys with segment_range.

    This function deduplicates keys within each table segment, using the
    GPU-accelerated segmented_unique_cuda kernel.

    The number of unique keys ``U`` exists only on the device, and reading it
    would make the host wait for everything queued before this call. So nothing
    is read and nothing is cut to ``U``: the outputs stay as long as the number
    of keys ``N`` the host already knows, and the part past the unique keys, rows
    ``[U, N)``, is padding. A build of the extension that fills it
    (``sync_free.TAIL_INERT``) leaves the reserved all-ones key there (``-1`` as
    int64, ``0xFF..FF`` as uint64) and frequency 0; ``sync_free.canonical_unique``
    gives the keys, and with them the table ids from ``expand_table_ids_cuda``,
    that convention for any build. Without the fill the frequency of the padding
    is unspecified, which is fine: it is only read.

    Args:
        keys: Input key tensor (int64 or uint64)
        segment_range: Table boundary offsets where segment_range[i] is the
                       start index for table i (int64)
        evict_strategy: Optional eviction strategy (for LFU mode)
        frequency_counts: Optional input frequency counts per key

    Returns:
        Tuple of (unique_keys, reverse_indices, unique_table_offsets,
                  output_scores, unique_total).
        unique_keys has N entries: the unique keys grouped by table, then the
        padding. reverse_indices has an entry per input key: the row of
        unique_keys that key was merged into, always a real row (below U).
        unique_table_offsets is int64[num_tables + 1] on the device; table t's
        unique keys are rows [offsets[t], offsets[t + 1]) and offsets[num_tables]
        is U. output_scores is the accumulated frequency of each unique key (N
        entries, padding included), or None when no frequency was asked for.
        unique_total is a 0-d int64 view of unique_table_offsets[num_tables], U
        on the device: reading it is the wait this function avoids.
        With no keys the arrays are empty and unique_table_offsets is zeros.
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
            return (
                empty_keys,
                empty_reverse_indices,
                d_table_range,
                None,
                d_table_range[num_tables],
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

        # The kernel allocates these at N, the most unique keys there can be.
        output_scores = freq_counters if need_frequency_output else None

        return (
            unique_keys,
            reverse_indices,
            table_offsets,
            output_scores,
            # A view of the offsets: no launch, and nothing reads it here.
            table_offsets[num_tables],
        )


def largest_score_per_unique_key(
    scores: torch.Tensor, reverse_indices: torch.Tensor, num_unique: int
) -> torch.Tensor:
    """The score of each unique key: the largest of those its occurrences had.

    ``reverse_indices[i]`` is the unique key that input key ``i`` was merged
    into; a key that was not repeated keeps its own score. ``num_unique`` is the
    length of the result and may exceed the number of unique keys (the unique
    arrays are padded to the number of keys): the rows nothing was merged into
    stay 0.
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
    # One int64 per unique key, for tables that take a score per key.
    unique_scores: Optional[torch.Tensor] = None
    # How many keys this batch counted for in the outstanding-keys check: its unique
    # keys when the arrays are exact, otherwise an upper bound of them, the number
    # of keys of the batch (``unique_total`` has the exact count, on the device).
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
    # bool per unique key, True for a key the admitter refused. A prefetch path sets
    # it when it ran its miss path with an admitter. None: no key was refused.
    non_admitted_mask: Optional[torch.Tensor] = None
    # What the growth of the storage was told this batch can add to each table (CPU
    # int64 per table); a forward that inserts the batch reports it back (see
    # ``note_forwarded`` of the storages).
    table_key_bounds: Optional[torch.Tensor] = None


def _note_table_users(storage: Storage, cache: Optional[Cache] = None) -> None:
    """Order the current stream after the last growth of the tables ``storage`` and
    ``cache`` keep, and register it as one of their users (``wait_for_growth`` of
    :class:`~dynamicemb.growth_order.GrowthOrder`).

    A growth only queues its kernels on the prefetch stream and returns, so the forward,
    the backward and the evaluation, which run on other streams, come first through
    here. An external storage or a user's class has no tables of ours and nothing to
    order.
    """
    for owner in (storage, cache):
        wait_for_growth = getattr(owner, "wait_for_growth", None)
        if wait_for_growth is not None:
            wait_for_growth()


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


def _insert_into_storage(
    storage: DynamicEmbStorage,
    next_tier: Optional[Storage],
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    values: torch.Tensor,
    scores: Optional[torch.Tensor],
) -> None:
    """Insert new keys into ``storage``, over arrays that keep the batch's length.

    A key that must not be inserted is the reserved key. What storage evicts for room
    moves to ``next_tier`` (a storage of the padded contract) as arrays of the same
    length, inert past the evictions; no count is read. Without a next tier it is
    ``storage.insert``.
    """
    if next_tier is None:
        storage.insert(keys, table_ids, values, scores)
        return
    # NO_EVICTION is deliberately not handled below: its storage never evicts
    # (it expands instead), so there is nothing to spill. The tables module
    # rejects NO_EVICTION together with an ``external_storage``.
    st = storage._state
    (
        slots,
        _,
        evicted_keys,
        evicted_table_ids,
        evicted_slots,
        _,
    ) = _insert_and_evict_keys(st, keys, table_ids, scores, async_count=True)
    # A row of the inert tail has slot -1, which the load skips.
    evicted_values = load_from_flat(
        st, evicted_slots, evicted_table_ids, copy_mode=CopyMode.VALUE
    )
    # A key the storage refused (e.g. its bucket is full and the score does not beat
    # any stored one) comes back as an "evicted" row with a negative slot:
    # load_from_flat left its value unwritten, so take the key's own input value
    # instead, as HybridStorage.insert does.
    _select_refused_values(evicted_slots, evicted_table_ids, values, evicted_values)
    store_to_flat(st, slots, table_ids, values)
    next_tier.insert_masked(evicted_keys, evicted_table_ids, evicted_values)


# One body serves every lookup group, over arrays that keep the length of the batch
# instead of narrowing them to a count read from the device. What the participants are
# decides how they are talked to, never which body runs: the cache and storage of the
# cache and HBM-direct paths must be in-tree (``_participants``); a storage that has not
# declared the padded contract, an admitter that has not and an initializer that has not
# (``types.as_padded``, ``MultiTableAdmitter.admit_masked``,
# ``embedding_admission.initializer_needs_dense_rows``) are served through their own
# host-sized methods after one read of how many rows are live, which is by their
# contract. Neither the size of the batch (a miss path whose buffers would not fit the
# budget runs in row chunks, ``sf.row_chunks``) nor the environment decides anything.


def _require(obj, cls, mode: StorageMode, role: str):
    """The cache and HBM-direct bodies read ``obj._state`` directly, so only the in-tree
    class qualifies; a subclass opts in by declaring ``accepts_empty_keys`` in its own
    body."""
    if isinstance(obj, cls) and (type(obj) is cls or has_padded_contract(obj)):
        return obj
    raise TypeError(
        f"the {role} of a {mode.name} lookup group must be a {cls.__name__} (or a "
        "subclass that sets accepts_empty_keys = True in its own class body), got "
        f"{type(obj).__name__}: this path reads its tables directly."
    )


def _participants(
    mode: StorageMode,
    cache: Optional[Cache],
    storage: Storage,
    external_storage: Optional[Storage],
) -> Tuple[Storage, Optional[Storage]]:
    """``(storage, next_tier)`` as the padded bodies talk to them.

    Decided by capability only, never by the size of the batch and never by the
    environment:

    * CACHE: the cache and the storage must be in-tree (``TypeError`` otherwise); the
      external storage, when there is one, is viewed through ``as_padded`` (itself when
      its class declared the padded contract, else an adapter).
    * HBM_DIRECT: the storage must be in-tree; there is no next tier (a PS only exists
      below a cache or as the only tier).
    * DEFAULT: any storage, viewed through ``as_padded``: the in-tree ones and a PS that
      declared the contract are themselves, a subclass of an in-tree class that did not
      redeclare it, and a class written against ``find`` / ``insert``, are adapted.
    """
    if mode == StorageMode.CACHE:
        _require(cache, DynamicEmbCache, mode, "cache")
        _require(storage, DynamicEmbStorage, mode, "storage")
        next_tier = None if external_storage is None else as_padded(external_storage)
        return storage, next_tier
    if mode == StorageMode.HBM_DIRECT:
        return _require(storage, DynamicEmbStorage, mode, "storage"), None
    return as_padded(storage), None


def _uses_rng(admitter) -> bool:
    """Whether ``admit`` draws random numbers. Its miss path is then never skipped
    on a timing-dependent poll (see ``sf.miss_gate_mode``): how many values a
    seeded stream has given out must not depend on timing."""
    return admitter is not None and bool(getattr(admitter, "uses_rng", False))


def _miss_gate_of(storage):
    """The miss gate of the table state a DEFAULT-mode storage looks keys up in
    (the HBM tier's, for a HybridStorage); a storage with no table state (an external
    one) has the stateless gate that always runs the miss path."""
    for attr in ("_state", "_hbm"):
        gate = getattr(getattr(storage, attr, None), "miss_gate", None)
        if gate is not None:
            return gate
    return sf.ALWAYS_GATE


def _rows(t: Optional[torch.Tensor], start: int, stop: int) -> Optional[torch.Tensor]:
    """The rows ``[start, stop)`` of a tensor with one entry per unique key (a view,
    no launch), or None for None."""
    return None if t is None else t[start:stop]


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
    *,
    external_storage: Optional[Storage] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Cache prefetch that does not narrow its arrays to counts read from the device.

    Every array keeps the length of the unique batch. A key that takes no part in a step is replaced by
    the reserved EMPTY key, which every table kernel skips (found=false, index -1)
    without reading its table id; a slot or row that must not be touched is -1.
    ``valid`` marks the real rows of a padded batch (None: every row is real). The
    other rows are hidden from the tables as the reserved key, whatever their
    producer left in them, so they look like misses to the lookups, and every miss
    mask ANDs ``valid``.

    The key-level work (the cache lookup and the pin of every hit, the scoring lookup
    in storage, the gate) is done once for the whole batch. The miss path, which
    allocates ``[rows, value_dim]`` buffers, runs in consecutive row chunks that keep
    those buffers within ``DYNAMICEMB_MISS_PATH_BUFFER_BYTES`` (``sf.row_chunks``):
    one chunk, the very kernels of an unchunked step, for every batch under the budget.
    Chunk ``k`` runs to completion (its keys are in the cache and pinned) before chunk
    ``k + 1`` starts, which is one of the orders the concurrent kernels of a single
    launch could have taken.

    The cache and storage must be in-tree (``_participants``); an admitter that has not
    declared ``accepts_empty_keys`` is served with dense keys (``admit_masked``).
    ``external_storage`` is the tier below the
    storage, any ``Storage``: it is asked for the keys the storage lacks, receives what
    the storage evicts for room, and what the cache evicts that the storage no longer
    has. One that has not declared the padded contract is served through its ``find`` /
    ``insert``; it needs the number of rows of each call on the host, which costs one
    read per call (see ``types.OldContractStorage``), and the whole path of a step that
    missed nothing is then skipped after one read of the miss count.

    Returns (slot_indices, non_admitted_mask): the cache slot of every key that is
    in the cache afterwards and -1 for the others, and the bool mask of the keys the
    admitter refused (None if none was asked).
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path"):
        state, device, n = cache._state, unique_keys.device, unique_keys.numel()
        if n == 0:
            return torch.empty(0, dtype=torch.int64, device=device), None
        storage, next_tier = _participants(
            StorageMode.CACHE, cache, storage, external_storage
        )
        # A tier written against find / insert is called with exact-length tensors, so
        # every call of it reads how many rows are live. The count of misses is read
        # first, once, which lets a step that missed nothing skip the tier entirely
        # (the gate would have launched the path on inert rows and paid three reads).
        counts_tier = isinstance(next_tier, OldContractStorage)
        if valid is not None:
            # A key the cache holds, left in the padding, would be pinned and its
            # slot updated by the backward from a gradient nobody computed (one only
            # the storage holds would get its score bumped). The padding of a
            # conforming batch is EMPTY already, so this changes nothing for it.
            unique_keys = sf.mask_keys(unique_keys, valid)
        gate = state.miss_gate

        # 1. Look the keys up in the cache and pin every hit before an insert can
        # evict it. The counter kernel returns on a negative slot before it reads
        # the table id, so misses and padding need no compaction.
        with torch.cuda.nvtx.range("op:cache_lookup"):
            _, founds, cache_indices = cache.lookup(unique_keys, unique_table_ids)
        cache.increment_counter(cache_indices, unique_table_ids)
        is_miss = ~founds if valid is None else valid & ~founds
        with torch.cuda.nvtx.range("op:miss_gate"):
            pending = (
                None if counts_tier else gate.begin(is_miss, rng=_uses_rng(admitter))
            )

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
        # when an already landed count proves it. A tier that counts its rows reads the
        # number of misses, and of the live keys it will be asked for, in one transfer.
        live_asked = None
        with torch.cuda.nvtx.range("op:miss_gate"):
            if counts_tier:
                misses, live_asked = _read_miss_counts(unique_keys, is_miss, mf.founds)
                need = misses > 0
            else:
                need = gate.needed(pending)
        if not need:
            return cache_indices, None  # all hits, and they are pinned already

        # 4. The miss path, in row chunks bounded by the buffer budget: two [rows,
        # value_dim] buffers (the loaded values, the evicted ones), four with a tier below
        # storage, counting what its lookup holds beside the loaded values (its answer,
        # and the two copies the answer of a host-sized class is merged through).
        chunks = sf.row_chunks(
            n,
            max_val_dim * emb_dtype.itemsize,
            buffers=2 if next_tier is None else 4,
        )
        if len(chunks) > 1:
            # The count is of the whole batch; a chunk asks the tier for its own rows.
            live_asked = None
        slot_parts, refused_parts = [], []
        for index, (start, stop) in enumerate(chunks):
            keys_c = unique_keys[start:stop]
            ids_c = unique_table_ids[start:stop]
            if index == 0:
                # Nothing has been inserted yet: the rows the whole-batch lookup found
                # are valid for this chunk.
                mf_c = MaskedFind(
                    mf.founds[start:stop],
                    _rows(mf.score_in, start, stop),
                    _rows(mf.scores, start, stop),
                    mf.rows[start:stop],
                )
            else:
                # An earlier chunk's insert into storage may have evicted a key this
                # chunk found there, and handed its row to the key that evicted it:
                # loading the old row would copy another key's value. Look the rows up
                # again, as storage is now (const: no score changes). A key that is gone
                # is asked of the tier below, which has it (the insert spilled it); with
                # none it is new to this step and initialized afresh, where the unchunked
                # step returns its trained row (DynamicEmb_APIs.md, Chunked miss path).
                r = _find_keys(
                    storage._state, keys_c, ids_c, const_lookup=True, compact=False
                )
                mf_c = MaskedFind(r[5], _rows(mf.score_in, start, stop), r[6], r[7])
            slots_c, refused_c = _cache_miss_chunk(
                cache,
                storage,
                next_tier,
                keys_c,
                ids_c,
                is_miss[start:stop],
                cache_indices[start:stop],
                mf_c,
                _rows(score_input, start, stop),
                _rows(accumulated_frequency, start, stop),
                max_emb_dim,
                max_val_dim,
                emb_dtype,
                initializer,
                admitter,
                accumulate=index > 0,
                live_asked=live_asked,
            )
            slot_parts.append(slots_c)
            refused_parts.append(refused_c)
        if len(slot_parts) == 1:
            return slot_parts[0], refused_parts[0]
        refused = None if refused_parts[0] is None else torch.cat(refused_parts)
        return torch.cat(slot_parts), refused


def _read_miss_counts(
    keys: torch.Tensor, is_miss: torch.Tensor, storage_founds: torch.Tensor
) -> Tuple[int, int]:
    """``(misses, live keys storage lacks)`` of a batch, in one blocking read.

    Only for a tier that has not declared the padded contract, whose calls need the
    number of rows they are made with on the host: knowing the number of misses first
    lets a step that missed nothing skip the tier, and the second number is handed to
    the adapter of its lookup, which would read it again otherwise.
    """
    asked = is_miss & ~storage_founds
    misses, live_asked = sf.read_now(
        is_miss.sum(), sf.valid_keys(sf.mask_keys(keys, asked)).sum()
    )
    return int(misses), int(live_asked)


def _cache_miss_chunk(
    cache: DynamicEmbCache,
    storage: DynamicEmbStorage,
    next_tier: Optional[Storage],
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    is_miss: torch.Tensor,
    cache_indices: torch.Tensor,
    mf: MaskedFind,
    score_input: Optional[torch.Tensor],
    frequency: Optional[torch.Tensor],
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    admitter: Optional[MultiTableAdmitter],
    accumulate: bool,
    live_asked: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The miss path of ``_prefetch_cache_path_masked`` for one chunk of rows.

    Every argument is the chunk's slice (``keys`` and ``table_ids`` already hide the
    padding). Returns the chunk's ``(slot_indices, non_admitted_mask)``. ``accumulate``
    says that an earlier chunk of this prefetch already recorded the cache metrics.
    ``next_tier`` is the tier below storage, a storage of the padded contract (a view,
    see ``_participants``) or None; ``live_asked`` is the number of keys it will be
    asked for when the caller has counted them (a host-sized tier then does not).
    """
    state, device, n = cache._state, keys.device, keys.numel()

    # 4. Where each miss lives: storage, the tier below it, or nowhere (new). A hit is
    # in storage too, but it is neither ``in_storage`` nor ``asked``: both are ANDed
    # with ``is_miss``.
    in_storage = is_miss & mf.founds
    asked = is_miss & ~mf.founds

    # Reset the optimizer-state block first, then load the storage rows over
    # it: what a row carries overwrites the reset, whereas a reset after the
    # load would wipe the state just loaded. Only the new rows keep the reset.
    values = torch.empty(n, max_val_dim, dtype=emb_dtype, device=device)
    if max_val_dim != max_emb_dim:
        state.optimizer.reset_optimizer_states(
            values[:, max_emb_dim:],
            emb_dims=state.table_emb_dims[table_ids],
        )
    with torch.cuda.nvtx.range("op:load_from_flat"):
        storage.load_masked(
            # The rows of hits and of padding are -1, which the load skips.
            mf._replace(rows=torch.where(is_miss, mf.rows, -1)),
            table_ids,
            CopyMode.VALUE,
            output=values,
        )
    in_next = None
    if next_tier is None:
        is_new = asked
    else:
        # The keys storage lacks are asked of the tier below it; the others are hidden
        # from it as the reserved key. What it has is loaded over the reset rows.
        with torch.cuda.nvtx.range("op:next_tier_find"):
            hint = {} if live_asked is None else {"live_count": live_asked}
            found_next = next_tier.lookup_masked(
                sf.mask_keys(keys, asked),
                table_ids,
                copy_mode=CopyMode.VALUE,
                **hint,
            )
            next_tier.load_masked(found_next, table_ids, CopyMode.VALUE, output=values)
        in_next = asked & found_next.founds
        is_new = asked & ~found_next.founds
        del found_next

    # 5. Admission: only the new keys are asked about (a key the tier below had was
    # admitted before).
    non_admitted_mask = None
    if admitter is None:
        admitted = is_new
    else:
        admit = admitter.admit_masked(sf.mask_keys(keys, is_new), table_ids, frequency)
        # The answer for any other row means nothing (a counter hands an EMPTY
        # key back its input frequency), hence the AND.
        admitted, non_admitted_mask = is_new & admit, is_new & ~admit
    with torch.cuda.nvtx.range("op:initializer"):
        sf.initialize_masked(
            initializer, values[:, :max_emb_dim], keys, table_ids, admitted
        )

    # 6. Storage learns what it lacks (the admitted keys, and the keys the tier below
    # had), with their score, before the cache sees them. A masked-out key is EMPTY:
    # no row is allocated for it and the store skips it. What storage evicts for room
    # goes to the tier below.
    to_storage = admitted if in_next is None else (in_next | admitted)
    with torch.cuda.nvtx.range("op:storage_insert"):
        _insert_into_storage(
            storage,
            next_tier,
            sf.mask_keys(keys, to_storage),
            table_ids,
            values,
            score_input,
        )

    # 7. The cache takes every miss that lives in storage, in the tier below it, or
    # was admitted. What it evicts comes back with a count that stays on the device:
    # arrays of the chunk's length whose entries past the count are inert.
    to_cache = in_storage | to_storage
    with torch.cuda.nvtx.range("op:cache_insert_evict"):
        (
            cache_slots,
            _,
            evicted_keys,
            evicted_slots,
            _,
            evicted_tids,
        ) = cache.insert_and_evict(
            sf.mask_keys(keys, to_cache),
            table_ids,
            None,
            async_count=True,
            accumulate=accumulate,
        )

    # 8. The victims are read before their slots are overwritten (stream
    # order), the new rows are stored, then the victims are written back (to storage,
    # or to the tier below it when storage no longer has them). An evicted entry with a
    # negative slot is a refused insert, not a victim; the write-back leaves it alone.
    with torch.cuda.nvtx.range("op:load_from_flat"):
        evicted_values = load_from_flat(
            state, evicted_slots, evicted_tids, copy_mode=CopyMode.VALUE
        )
    with torch.cuda.nvtx.range("op:store_to_flat"):
        store_to_flat(state, cache_slots, table_ids, values)
    _write_back_evicted(
        storage,
        next_tier,
        evicted_keys,
        evicted_tids,
        evicted_values,
        slots=evicted_slots,
    )

    # 9. Pin the new entries. A hit keeps the slot its lookup found; a refused
    # insert and a padded row keep -1.
    cache.increment_counter(cache_slots, table_ids)
    return (
        torch.where(cache_slots >= 0, cache_slots, cache_indices),
        non_admitted_mask,
    )


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
    device. The miss gate decides whether the path is launched at all: ``poll``
    only takes a count that has already landed, ``always`` does not look at one.

    The lookup and the pin of every hit are done once for the whole batch; the miss
    path, which allocates ``[rows, value_dim]`` buffers, runs in consecutive row chunks
    that keep them within ``DYNAMICEMB_MISS_PATH_BUFFER_BYTES`` (``sf.row_chunks``): one
    chunk, the very kernels of an unchunked step, for every batch under the budget. A
    chunk loads nothing from the table, and the hits are pinned by the reference
    counter, so the chunks need no second lookup.

    ``valid`` marks the real rows (``None``: every row is real). The other rows are
    hidden from the table as the reserved key before anything is looked up,
    whatever their producer left in them. ``unique_keys``, ``unique_table_ids`` and
    the optional ``accumulated_frequency`` and ``key_scores`` have one entry per
    row; a table kernel reads a lane's score before it looks at its key.

    Returns ``(slot_indices, non_admitted_mask, value_row_indices)``: the refused keys
    are a bool mask over the rows (``True``: refused), not their positions. The mask is
    ``None`` without an admitter, and when the miss path did not run.
    """
    with torch.cuda.nvtx.range("_prefetch_hbm_direct_path"):
        storage, _ = _participants(StorageMode.HBM_DIRECT, None, storage, None)
        state = storage._state
        device = unique_keys.device
        n = unique_keys.numel()

        if n == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, None, empty

        if valid is not None:
            # A key the table holds, left in the padding, would be found, pinned and
            # updated by the backward from a gradient nobody computed. The padding of
            # a conforming batch is EMPTY already, so this changes nothing for it.
            unique_keys = sf.mask_keys(unique_keys, valid)
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

        # 3. The miss path, in row chunks bounded by the buffer budget.
        chunks = sf.row_chunks(n, max_val_dim * state.emb_dtype.itemsize, buffers=2)
        parts = [
            _hbm_miss_chunk(
                storage,
                unique_keys[start:stop],
                unique_table_ids[start:stop],
                is_miss[start:stop],
                founds[start:stop],
                slots_found[start:stop],
                rows_found[start:stop],
                _rows(score_value, start, stop),
                _rows(accumulated_frequency, start, stop),
                max_emb_dim,
                max_val_dim,
                initializer,
                admitter,
            )
            for start, stop in chunks
        ]
        if len(parts) == 1:
            return parts[0]
        slots, refused, rows = zip(*parts)
        return (
            torch.cat(slots),
            None if refused[0] is None else torch.cat(refused),
            torch.cat(rows),
        )


def _hbm_miss_chunk(
    storage: DynamicEmbStorage,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    is_miss: torch.Tensor,
    founds: torch.Tensor,
    slots_found: torch.Tensor,
    rows_found: torch.Tensor,
    score_value: Optional[torch.Tensor],
    frequency: Optional[torch.Tensor],
    max_emb_dim: int,
    max_val_dim: int,
    initializer: MultiTableInitializer,
    admitter: Optional[MultiTableAdmitter],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """The miss path of ``_prefetch_hbm_direct_path_masked`` for one chunk of rows.

    Every argument is the chunk's slice. Returns the chunk's ``(slot_indices,
    non_admitted_mask, value_row_indices)``.
    """
    state, device, n = storage._state, keys.device, keys.numel()

    # 3. Admission. The admitter sees only the keys that missed, the others are
    # the reserved key. What it returns says nothing about the other lanes (the
    # frequency counter hands a reserved key's input frequency back, which can
    # pass its threshold), so both masks are restricted to the misses here.
    non_admitted_mask: Optional[torch.Tensor] = None
    if admitter is None:
        admitted = is_miss
    else:
        admit = admitter.admit_masked(sf.mask_keys(keys, is_miss), table_ids, frequency)
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
            keys,
            table_ids,
            admitted,
        )
    if max_val_dim != max_emb_dim:
        state.optimizer.reset_optimizer_states(
            init_values[:, max_emb_dim:],
            emb_dims=state.table_emb_dims[table_ids],
        )

    # 5. Insert the admitted keys, one lane per row. The score tensor is as long
    # as the key array: the kernel reads a lane's score before it looks at its
    # key.
    score_arg = get_insert_score_arg(state, n, device, score_value)
    insert_keys = sf.mask_keys(keys, admitted)
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
                table_ids,
                score_arg,
                collect_evicted=True,
                return_slots=True,
            )
            _append_evicted(state, evicted_keys, evicted_table_ids, num_evicted)
        else:
            new_rows, new_slots = state.key_index_map.insert(
                insert_keys, table_ids, score_arg, return_slots=True
            )
    # A key that was not inserted (a hit, a refused key, a padding row) has row
    # -1 and slot -1: the store and the counter skip it. The value buffer is
    # addressed by row, the counter by slot.
    with torch.cuda.nvtx.range("op:store_to_flat"):
        store_to_flat(state, new_rows, table_ids, init_values)
    storage.increment_counter(new_slots, table_ids)

    # 6. A hit keeps the slot and row its lookup found, an inserted key takes the
    # new ones, any other key (refused, not inserted, padding) stays at -1.
    slot_indices = torch.where(founds, slots_found, new_slots)
    rows = (
        slot_indices
        if not state.key_index_map.use_rows_
        else torch.where(founds, rows_found, new_rows)
    )
    return slot_indices, non_admitted_mask, rows


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
    counts of its features. The growth of the storage is sized from it (the exact number
    of unique keys per table is on the device and is never read), and every table is
    bounded by the whole batch when it is not given.

    The unique arrays are as long as ``indices``, not as the number of unique keys,
    which only the device knows; see ``segmented_unique`` and
    ``PrefetchState.unique_valid``.
    """
    with torch.cuda.nvtx.range("dynamicemb_prefetch"):
        _note_table_users(storage, cache)
        table_num = feature_offsets.numel() - 1
        assert table_num != 0
        assert (
            table_key_bounds is None or table_key_bounds.numel() == table_num
        ), "table_key_bounds has one entry per table"
        emb_dtype = storage.embedding_dtype()
        max_emb_dim = storage.max_embedding_dim()
        max_val_dim = storage.max_value_dim()

        evict_strat = EvictStrategy(evict_strategy.value) if evict_strategy else None
        # todo: double check
        frequency_counts_int64 = None
        if frequency_counters is not None:
            frequency_counts_int64 = frequency_counters.long()

        if cache is not None:
            storage_mode = StorageMode.CACHE
        elif _is_hbm_storage(storage):
            storage_mode = StorageMode.HBM_DIRECT
        else:
            storage_mode = StorageMode.DEFAULT
        # How many keys are unique is on the device: the unique arrays keep the length
        # of ``indices``, and ``unique_valid`` marks their real rows.
        num_keys = indices.numel()

        # The sizes the expansion below decides from are read back without waiting, on
        # the stream that inserts into the storage: this one, for a cache's storage
        # and an HBM storage. A host or hybrid storage is inserted into by the
        # forward, which takes its snapshot there (``note_forwarded``).
        if storage_mode != StorageMode.DEFAULT and hasattr(
            storage, "request_size_snapshot"
        ):
            storage.request_size_snapshot()

        indices_table_range = get_table_range(offsets, feature_offsets)
        (
            unique_keys,
            reverse_indices,
            unique_table_offsets,
            lfu_accumulated_frequency,
            unique_total,
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
        unique_table_ids = expand_table_ids_cuda(
            unique_table_offsets,
            unique_keys.numel(),
        )

        # Past the unique keys the arrays hold the all-ones key and table id -1,
        # which every table kernel skips; ``unique_valid`` is the mask of the rows
        # before them.
        unique_keys, unique_table_ids, unique_valid = sf.canonical_unique(
            unique_keys, unique_table_ids, table_num
        )
        # What each table's keys can add, as far as the host knows without the
        # count: its features' key counts if the caller has them, otherwise the
        # whole batch for every table (growth can then come early, never late).
        per_table_bound = (
            table_key_bounds
            if table_key_bounds is not None
            else torch.full((table_num,), num_keys, dtype=torch.int64)
        )
        # A bound, not the unique count; the exact one is on the device
        # (``unique_total``) and OutstandingKeys reads it late.
        num_prefetched_keys, exact_unique_count = num_keys, None

        slot_indices = None
        non_admitted = None
        value_row_indices = None
        key_map_generation = 0
        outstanding_entry = None
        with table_section(table_gate):
            if hasattr(storage, "expand_if_need"):
                storage.expand_if_need(
                    per_table_bound,
                    inserts_in_forward=(storage_mode == StorageMode.DEFAULT),
                )

            if storage_mode == StorageMode.CACHE:
                if outstanding_keys_ref is not None:
                    outstanding_entry = outstanding_keys_ref.add(
                        num_prefetched_keys, unique_total, exact_unique_count
                    )
                    outstanding_keys_ref.check(cache._state.capacity)
                (
                    slot_indices,
                    non_admitted,
                ) = _prefetch_cache_path_masked(
                    cache,
                    storage,
                    unique_keys,
                    unique_table_ids,
                    unique_valid,
                    max_emb_dim,
                    max_val_dim,
                    emb_dtype,
                    initializer,
                    lfu_accumulated_frequency,
                    admitter,
                    key_scores=unique_scores,
                    external_storage=external_storage,
                )
                # The cache addresses its own value buffer by cache slot: a cache is
                # never NO_EVICTION (it enables overflow, which NO_EVICTION rejects
                # at construction), so the two index spaces coincide here.
                value_row_indices = slot_indices
            elif storage_mode == StorageMode.HBM_DIRECT:
                (
                    slot_indices,
                    non_admitted,
                    value_row_indices,
                ) = _prefetch_hbm_direct_path_masked(
                    storage,
                    unique_keys,
                    unique_table_ids,
                    unique_valid,
                    max_emb_dim,
                    max_val_dim,
                    initializer,
                    lfu_accumulated_frequency,
                    admitter,
                    key_scores=unique_scores,
                )
                key_map_generation = storage._state.key_map_generation

        # The refused keys come back as a bool mask over all unique rows.
        return PrefetchState(
            unique_keys=unique_keys,
            reverse_indices=reverse_indices,
            unique_table_ids=unique_table_ids,
            lfu_accumulated_frequency=lfu_accumulated_frequency,
            table_num=table_num,
            emb_dtype=emb_dtype,
            slot_indices=slot_indices,
            storage_mode=storage_mode,
            non_admitted_mask=non_admitted,
            unique_valid=unique_valid,
            unique_total=unique_total,
            unique_scores=unique_scores,
            num_prefetched_keys=num_prefetched_keys,
            outstanding_keys_ref=outstanding_keys_ref,
            outstanding_entry=outstanding_entry,
            table_key_bounds=per_table_bound,
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
    """Eval-only forward for all storage configurations (no autograd).

    A pooled lookup deduplicates its keys like a training prefetch does: the unique
    arrays keep the length of the input and a ``valid`` mask marks their real rows, so
    nothing is read back from the device. A padded row is neither looked up nor
    written (``eval_lookup`` leaves it as the buffer was) and is never gathered:
    the pooling reads only the rows ``reverse_indices`` names. The ``[N, embedding_dim]``
    buffer of the unique rows is the one transient eval has: ``O(N)``, not ``O(unique
    keys)``, and not chunked (the gather would have to accumulate across chunks).
    """
    with torch.cuda.nvtx.range("dynamicemb_eval_forward"):
        _note_table_users(storage, cache)
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
            unique_table_offsets,
            _,
            _,
        ) = segmented_unique(
            indices,
            indices_table_range,
            evict_strat,
            frequency_counts_int64,
        )

        unique_table_ids = expand_table_ids_cuda(
            unique_table_offsets,
            unique_indices.numel(),
        )
        unique_indices, unique_table_ids, unique_valid = sf.canonical_unique(
            unique_indices, unique_table_ids, table_num
        )

        sf.warn_if_unchunked_over_budget(
            unique_indices.numel(),
            storage.max_embedding_dim() * emb_dtype.itemsize,
            "the unique rows of a pooled evaluation",
        )
        unique_embs = eval_lookup(
            storage,
            unique_indices,
            unique_table_ids,
            initializer,
            cache=cache,
            valid=unique_valid,
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
    """Generic (DEFAULT) forward of any storage, in mask form.

    Find, admit, initialize, insert, over every unique key at once. A key a step must
    leave alone (a hit, a refused key, padding) is not compacted away, which would read
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
    for padding. ``None`` (only when every row is real, ``valid=None``) means every
    row is.
    """
    with torch.cuda.nvtx.range("_generic_forward_path"):
        # The storage as the padded body talks to it: itself when its class declared
        # the padded contract (the in-tree storages, a PS that adopted it), else a view
        # that serves its find / insert after one read of the live rows per call.
        storage, _ = _participants(StorageMode.DEFAULT, None, storage, None)
        n, device = unique_keys.numel(), unique_keys.device
        sf.warn_if_unchunked_over_budget(
            n,
            max_val_dim * emb_dtype.itemsize,
            "the value block of a forward over a host or hybrid storage",
        )
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
            # A scoring lookup of every key, as ``find`` always did: it updates the
            # score of each key it finds. The rows are loaded below.
            mf = storage.lookup_masked(
                unique_keys,
                unique_table_ids,
                accumulated_frequency,
                key_scores=key_scores,
            )
        is_miss = ~mf.founds if valid is None else valid & ~mf.founds
        # sync-ok: a host fact, the count the class returned from its own find
        if isinstance(storage, OldContractStorage) and storage.nothing_missed(mf):
            # The class itself said (a host count it returned) that it lacked no key.
            miss_path = False
        else:
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
            admit = admitter.admit_masked(
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
                    # An initializer written for exact index lists gets them, after
                    # one read of how many rows are refused.
                    dense=initializer_needs_dense_rows(non_admitted_initializer),
                )

        with torch.cuda.nvtx.range("op:storage_insert"):
            # The score the find used, at full length: the insert reads
            # scores[i] before it checks the key, so a shorter tensor (the
            # admitted keys' scores alone) would be read out of bounds.
            storage.insert_masked(
                sf.mask_keys(unique_keys, admitted),
                unique_table_ids,
                unique_values,
                mf.score_in,
            )
        # Without an admitter every miss was inserted, so every real key is
        # persisted; with one, the keys it refused are not.
        return unique_values, (valid if admitter is None else mf.founds | admitted)


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
            _note_table_users(storage, cache)
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
                    # The prefetch ran its miss path with an admitter (a host
                    # fact); which keys it refused is device data, so the rows
                    # come as a mask and the initializer skips the rest.
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
                            # An initializer of the user's that is written for exact
                            # index lists gets them, after one read of how many.
                            dense=initializer_needs_dense_rows(
                                non_admitted_initializer
                            ),
                        )
                unique_values = None
                persisted_mask = None
            else:
                assert prefetch_state.storage_mode == StorageMode.DEFAULT
                unique_values, persisted_mask = _generic_forward_path_masked(
                    storage,
                    optimizer,
                    prefetch_state.unique_keys,
                    prefetch_state.unique_table_ids,
                    prefetch_state.unique_valid,
                    max_emb_dim,
                    max_val_dim,
                    emb_dtype,
                    initializer,
                    prefetch_state.lfu_accumulated_frequency,
                    admitter,
                    key_scores=prefetch_state.unique_scores,
                )
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
            ctx.persisted_mask = persisted_mask
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
            _note_table_users(storage, cache)
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
                    # Re-insert the rows the forward persisted with their updated values.
                    # A key that is not persisted is hidden as EMPTY instead of gathered
                    # out, so no count is needed to size it. preserve_existing=True: CONST
                    # score policy on insert: refresh embedding/optimizer buffers only; do
                    # not overwrite table scores (LFU / timestamp / etc.). For
                    # HybridStorage the HBM tier uses this on backward; existing scores
                    # stay. A storage written against the host-sized contract is served
                    # through its insert after one read of how many rows are persisted.
                    keys_c = (
                        ctx.unique_keys
                        if ctx.persisted_mask is None
                        else sf.mask_keys(ctx.unique_keys, ctx.persisted_mask)
                    )
                    if keys_c.numel() > 0:
                        with torch.cuda.nvtx.range("op:storage_insert"):
                            as_padded(storage).insert_masked(
                                keys_c,
                                unique_table_ids,
                                ctx.unique_values,
                                preserve_existing=True,
                            )

            return (None,) * 18
