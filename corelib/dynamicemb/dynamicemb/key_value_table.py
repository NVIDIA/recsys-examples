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

from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.checkpoint.files import (  # noqa: F401 - preserve checkpoint helper exports
    META_EMBEDDING_DIM,
    META_EMBEDDING_DTYPE,
    META_OPT_STATE_DTYPE,
    _iter_batches_from_files,
    _raw_bytes,
    _tensor_from_bytes,
    dtype_from_meta_name,
    dtype_to_meta_name,
    load_from_json,
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
    DynamicEmbTableOptions,
    EvictedItemMode,
    ReplayContent,
    score_load_permutation,
)
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.score.args import (  # noqa: F401 (re-exported)
    get_find_score_arg,
    get_insert_score_arg,
    score_source,
)
from dynamicemb.score.checkpoint_columns import (  # noqa: F401 (re-exported)
    _age_to_scores,
    _dump_score_block,
    _fresh_score_block,
    _replay_score_block,
    _scores_to_age,
    _timestamp_score_columns,
)
from dynamicemb.score.spec import get_score_policy  # noqa: F401 (re-exported)
from dynamicemb.state.erase import (  # noqa: F401 (re-exported)
    ERASE_CHUNK_SLOTS,
    ERASE_MAX_CHUNKS_IN_FLIGHT,
    _erase_state_by_score,
    _erase_state_keys,
    next_erase_chunk_end,
)
from dynamicemb.state.evicted_items import (  # noqa: F401 (re-exported)
    _append_erased,
    _append_evicted,
    _pop_chunked_keys,
    _pop_state_erased_keys,
    _pop_state_erased_slot_index,
    _pop_state_evicted_keys,
)
from dynamicemb.state.expansion import (  # noqa: F401 (re-exported)
    _expand_if_need_no_eviction,
    _expand_tables_impl,
    _grow_values_only,
    _value_rows_target,
    expand_if_need_impl,
    get_expand_info,
)
from dynamicemb.state.key_index_ops import (  # noqa: F401 (re-exported)
    _find_keys,
    _insert_and_evict_keys,
    _insert_key_values,
)
from dynamicemb.state.table_state import (  # noqa: F401 (re-exported)
    NO_EVICTION_LOAD_FACTOR,
    DynamicEmbTableState,
    collect_row_counts_to_device,
    collect_table_sizes_for_state,
    collect_table_sizes_to_device,
    create_table_state,
    get_uvm_tensor,
)
from dynamicemb.state.value_rows import (  # noqa: F401 (re-exported)
    _encode_slot_index,
    _row_differs_from_slot,
    _split_value_row,
)
from dynamicemb.types import SCORE_TYPE, Cache, CopyMode, ReplayStats, Storage
from dynamicemb.utils import (  # noqa: F401 (re-exported)
    _slice,
    checkpoint_dtype,
    decode_table_values,
    encode_table_values,
)
from dynamicemb_extensions import (
    EvictStrategy,
    device_timestamp,
    flagged_compact,
    select_insert_failed_values,
)
from torch import Tensor

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
        return self._state.buffers.dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._state.buffers.emb_dims_list[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._state.buffers.value_dims_list[table_id]

    def max_embedding_dim(self) -> int:
        return self._state.buffers.max_emb_dim

    def max_value_dim(self) -> int:
        return self._state.buffers.max_value_dim

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
        values = self._state.buffers.load_to_padded(
            flat_rows, table_ids, copy_mode=copy_mode
        )

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
            values = s.buffers.load(table_id, flat_rows)
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
        emb_dim_t = state.buffers.emb_dims_list[table_id]
        # Checkpoint width, matching what _split_value_row emits: an empty
        # batch still has to agree on column count with the non-empty ranks it
        # is all_gathered against.
        opt_dim_t = (
            state.optimizer.get_ckpt_state_dim(emb_dim_t)
            if state.buffers.value_dims_list[table_id] > emb_dim_t
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
                0, emb_dim_t, dtype=checkpoint_dtype(state.buffers.dtype), device=dev
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
                    dtype=state.optimizer.get_state_dtype(state.buffers.dtype),
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
        return self._state.buffers.dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._state.buffers.emb_dims_list[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._state.buffers.value_dims_list[table_id]

    def max_embedding_dim(self) -> int:
        return self._state.buffers.max_emb_dim

    def max_value_dim(self) -> int:
        return self._state.buffers.max_value_dim

    def embedding_dims(self, on_device: bool = False) -> torch.Tensor:
        return (
            self._state.buffers.emb_dims
            if on_device
            else self._state.buffers.emb_dims_host
        )

    def all_dims_vec4(self) -> bool:
        return self._state.buffers.all_dims_vec4

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
        return self._hbm.buffers.dtype

    def embedding_dim(self, table_id: int) -> int:
        return self._hbm.buffers.emb_dims_list[table_id]

    def value_dim(self, table_id: int) -> int:
        return self._hbm.buffers.value_dims_list[table_id]

    def max_embedding_dim(self) -> int:
        return self._hbm.buffers.max_emb_dim

    def max_value_dim(self) -> int:
        return self._hbm.buffers.max_value_dim

    def embedding_dims(self, on_device: bool = False) -> torch.Tensor:
        # find() builds the value buffer from the HBM tier (``self._hbm.buffers.load_to_padded``),
        # and max_*_dim above also report the HBM tier, so its dims are authoritative.
        return (
            self._hbm.buffers.emb_dims if on_device else self._hbm.buffers.emb_dims_host
        )

    def all_dims_vec4(self) -> bool:
        # HBM tier produces the value buffer in find(); its alignment governs
        # whether the vec4 padded-buffer kernels are safe for all rows.
        return self._hbm.buffers.all_dims_vec4

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
        values = self._hbm.buffers.load_to_padded(
            flat_hbm, table_ids, copy_mode=copy_mode
        )

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
        host_vals = self._host.buffers.load_to_padded(
            flat_host, missing_table_ids_hbm, copy_mode=copy_mode
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
          ``buffers.store_from_padded`` updates rows that already exist in HBM or host. Keys
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
                self._hbm.buffers.store_from_padded(
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
                    self._host.buffers.store_from_padded(
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

        evicted_values = self._hbm.buffers.load_to_padded(
            evicted_indices, evicted_table_ids, copy_mode=CopyMode.VALUE
        )
        select_insert_failed_values(evicted_indices, unique_values, evicted_values)
        self._hbm.buffers.store_from_padded(indices, table_ids, unique_values)

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
            values = s.buffers.load(table_id, flat_rows)
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
        emb_dim_t = states_to_dump[0].buffers.emb_dims_list[table_id]
        # Checkpoint width -- see the note in DynamicEmbStorage.incremental_dump.
        st0 = states_to_dump[0]
        opt_dim_t = (
            st0.optimizer.get_ckpt_state_dim(emb_dim_t)
            if st0.buffers.value_dims_list[table_id] > emb_dim_t
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
        if self._host.buffers.dtype != self._hbm.buffers.dtype:
            # The HBM tier is appended into the host tier's value files, under a
            # single meta describing one dtype, so two precisions in one file
            # would be unreadable. Both tiers are built from the same table
            # options today, so this is a guard, not a supported configuration.
            raise NotImplementedError(
                "dump is not supported for HybridStorage whose tiers store "
                f"different precisions (host {self._host.buffers.dtype}, HBM "
                f"{self._hbm.buffers.dtype}); they share one value file."
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
            - self._hbm.buffers.num_rows(table_id),
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
                    dtype=self._hbm.buffers.dtype,
                    device=device,
                )
                self._hbm.optimizer.reset_optimizer_states(
                    opt_states,
                    emb_dims=self._hbm.buffers.emb_dims_list[table_id],
                )
            elif opt_states is not None and params.runtime_optstate_dim > 0:
                opt_states = self._hbm.optimizer.states_from_checkpoint(
                    opt_states,
                    self._hbm.buffers.emb_dims_list[table_id],
                    self._hbm.buffers.dtype,
                    device,
                )

            vtype = self._hbm.buffers.dtype
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

            evicted_values = self._hbm.buffers.load_to_padded(
                evicted_indices,
                evicted_table_ids,
                copy_mode=CopyMode.VALUE,
            )
            select_insert_failed_values(evicted_indices, values, evicted_values)
            self._hbm.buffers.store(table_id, ins_indices, values)

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
            values = state.buffers.load_to_padded(
                indices, tid, copy_mode=CopyMode.VALUE
            )
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

    embs = cache._state.buffers.load_to_padded(
        cache_indices, table_ids, copy_mode=CopyMode.EMBEDDING
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
