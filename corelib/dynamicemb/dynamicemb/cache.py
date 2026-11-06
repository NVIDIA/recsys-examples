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

"""The GPU cache in front of a storage: which keys have a row, and which rows are in use.

See docs/lightweight_cache_design.md. ``CacheCore`` (C++/CUDA) holds the index; the
value rows are a ``MixedDimBuffers`` that the caller builds and hands over.
"""

from dataclasses import replace
from typing import Iterator, List, Optional, Tuple

import torch
from dynamicemb.mixed_dim_buffers import BuffersView, MixedDimBuffers
from dynamicemb.types import Cache, CacheMetrics, CopyMode
from dynamicemb_extensions import CacheCore, flagged_compact

# A slot with no key (all bits set, as in the hash tables); also the one key the
# cache cannot hold, and the storage cannot either.
_CACHE_EMPTY_KEY = -1


class DynamicEmbCache(Cache):
    """The GPU cache in front of a storage: which keys have a row, and which rows
    are in use. See docs/lightweight_cache_design.md.

    A key's row number is its *slot*. A slot is *pinned* while a step uses it
    (``lookup`` pins the slots it finds, ``insert_and_evict`` the ones it hands
    out) and cannot be taken until the step releases it (``release``).
    ``insert_and_evict`` hands out the first unpinned slots of each table in
    FIFO order, and fails only when a table has fewer unpinned slots than it
    needs. The cache holds no score: the storage owns every key's score, and
    what the cache evicts is written back by the caller.

    ``values`` holds the rows, one buffer per table, and the number of rows of a
    table is its number of slots. It must be a fixed pool (``kind="fixed"``),
    because a cache never grows. ``key_dtype`` is the dtype of the keys the
    caller uses (int64 or uint64): the index keeps int64 bit patterns and gives
    keys back in this dtype.
    """

    def __init__(
        self,
        values: MixedDimBuffers,
        key_dtype: torch.dtype = torch.int64,
    ):
        if values.kind != "fixed":
            raise ValueError(
                f"a cache's values must be a fixed pool (kind='fixed'), not {values.kind!r}"
            )
        if key_dtype not in (torch.int64, torch.uint64):
            raise ValueError(f"key_dtype is int64 or uint64, not {key_dtype}")
        self.values = values
        self._core = CacheCore(
            [values.num_rows(t) for t in range(values.num_tables)],
            values.device.index
            if values.device.index is not None
            else torch.cuda.current_device(),
        )
        self._key_dtype = key_dtype
        self._metrics = CacheMetrics()
        self._record_cache_metrics = False

    # -- key dtype: the index keeps int64 bit patterns; kernels want contiguous input --

    def _as_i64(self, keys: torch.Tensor) -> torch.Tensor:
        keys = keys.contiguous()
        return keys.view(torch.int64) if keys.dtype == torch.uint64 else keys

    def _to_key_dtype(self, keys: torch.Tensor) -> torch.Tensor:
        if self._key_dtype == torch.uint64:
            return keys.view(torch.uint64)
        return keys

    # -- Cache interface --

    def lookup(
        self,
        unique_keys: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(founds, slots)``; a missing key has slot -1.

        Every slot found is pinned (one more reference), so the keys must be
        unique, and whoever calls this releases them (``release``).
        """
        slots = self._core.lookup(self._as_i64(unique_keys), table_ids.contiguous())
        founds = slots >= 0
        if self._record_cache_metrics:
            self._metrics.num_lookups = unique_keys.size(0)
            self._metrics.num_hits = int(founds.sum().item())
        return founds, slots

    def release(self, slots: torch.Tensor, table_ids: torch.Tensor) -> None:
        """Take one reference back from each slot, aligned with ``table_ids``.

        A slot of -1 (a key ``lookup`` did not find) is skipped.
        """
        self._core.release(slots.contiguous(), table_ids.contiguous())

    def insert_and_evict(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Give each key (unique, not cached, grouped by table) the next unpinned
        slot of its table in FIFO order, and pin it (one reference).

        Returns ``(slots, num_evicted, evicted_keys, evicted_slots,
        evicted_table_ids)``: the key that held a slot is evicted, and the slot
        number is the row where its value still sits until the caller overwrites
        it. Raises, changing nothing, if a table has fewer unpinned slots than
        keys.
        """
        table_ids = table_ids.contiguous()
        slots, evicted = self._core.insert_and_evict(self._as_i64(keys), table_ids)
        n_ev, _, (ev_keys, ev_slots, ev_tids) = flagged_compact(
            evicted != _CACHE_EMPTY_KEY, [evicted, slots, table_ids]
        )
        if self._record_cache_metrics:
            self._metrics.num_inserts = keys.numel()
            self._metrics.num_evictions = n_ev
        return slots, n_ev, self._to_key_dtype(ev_keys), ev_slots, ev_tids

    def erase(self, keys: torch.Tensor, table_ids: torch.Tensor) -> None:
        """Drop the keys (those not cached are ignored); their slots become free."""
        self._core.erase(self._as_i64(keys), table_ids.contiguous())

    def export(
        self, table_id: int, batch_size: int = 1 << 20
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(keys, slots)`` of the keys the table holds, in chunks."""
        table_keys = self._core.keys_of(table_id)
        for start in range(0, table_keys.numel(), batch_size):
            chunk = table_keys[start : start + batch_size]
            idx = (chunk != _CACHE_EMPTY_KEY).nonzero().flatten()
            if idx.numel() > 0:
                yield self._to_key_dtype(chunk[idx]), idx + start

    def reset(self) -> None:
        self._core.reset()

    def check_errors(self) -> int:
        """Invariant violations the kernels saw (a release below zero, an erase of
        a pinned slot, a slot missing from its chain) since the last call. Syncs;
        for tests."""
        return self._core.check_errors()

    @property
    def cache_metrics(self) -> Optional[CacheMetrics]:
        return replace(self._metrics) if self._record_cache_metrics else None

    def set_record_cache_metrics(self, record: bool) -> None:
        self._record_cache_metrics = record

    # -- value rows: the cache's own, read and written by slot (see MixedDimBuffers) --

    def load_padded(
        self, slots: torch.Tensor, table_ids: torch.Tensor, copy_mode: CopyMode
    ) -> torch.Tensor:
        """Rows at ``slots`` of the given tables, padded to the widest table."""
        return self.values.load_padded(slots, table_ids, copy_mode)

    def store_padded(
        self, slots: torch.Tensor, table_ids: torch.Tensor, rows: torch.Tensor
    ) -> None:
        """Write padded rows to ``slots`` of the given tables."""
        self.values.store_padded(slots, table_ids, rows)

    def load_exact(self, slots: torch.Tensor, table: int) -> torch.Tensor:
        """Whole rows of one table at exactly its width: ``[n, value_dim]``."""
        return self.values.load_exact(slots, table)

    def store_exact(self, slots: torch.Tensor, table: int, rows: torch.Tensor) -> None:
        """Write whole rows of one table at exactly its width."""
        self.values.store_exact(slots, table, rows)

    @property
    def buffers_view(self) -> BuffersView:
        """Where the value rows are and how wide, for the optimizer."""
        return self.values.view

    # -- what the cache holds --

    @property
    def num_tables(self) -> int:
        return self.values.num_tables

    def capacity(self, table_id: Optional[int] = None) -> int:
        """Slots of one table or of all."""
        caps = self._core.capacities()
        return sum(caps) if table_id is None else int(caps[table_id])

    def size(self, table_id: Optional[int] = None) -> int:
        """Keys held, of one table or of all."""
        tables = range(self.num_tables) if table_id is None else [table_id]
        return sum(
            int((self._core.keys_of(t) != _CACHE_EMPTY_KEY).sum().item())
            for t in tables
        )

    def value_bytes(self) -> List[Tuple[int, int, int, int, int, int]]:
        """Per table, the bytes of the value rows: (total, embedding, optimizer
        state), then the same three again for the peak. A cache never grows, so the
        peak is the current size. The layout of the storage tiers' memory report."""
        rows = []
        for t in range(self.num_tables):
            total = self.values.buffers[t].num_bytes()
            emb = total * self.values.emb_dim(t) // self.values.value_dim(t)
            rows.append((total, emb, total - emb, total, emb, total - emb))
        return rows

    def meta_bytes(self) -> int:
        """Device bytes of the metadata, everything the cache keeps besides the
        value rows: the keys, the reference counts, the hash chains and buckets,
        the per-table offsets and cursors."""
        return self._core.meta_bytes()
