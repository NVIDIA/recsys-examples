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

"""Keep the DynamicEmb tables of TorchRec 2D sparse parallelism's replicas in agreement.

``DMPCollection.sync`` averages each fused TBE weight tensor across the replicas,
which works because a row id sits at the same offset in every replica. A
DynamicEmb table is a hash table: each replica places keys in its own slots and
holds its own key set, so ``DMPCollection.sync`` leaves it alone, and the
replicas are merged by key here instead. See docs/two_d_sparse_parallel_design.md.
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Sequence

import torch
import torch.distributed as dist
from dynamicemb.dump_load import get_dynamic_emb_module
from dynamicemb.dynamicemb_config import (
    score_strategy_has_timestamp_column,
)
from dynamicemb.key_value_table import (
    DynamicEmbStorage,
    get_find_score_arg,
)
from dynamicemb.replica_peer import ReplicaPeer
from dynamicemb.types import DynamicEmbInitializerMode
from dynamicemb.utils import encode_table_values
from dynamicemb_extensions import device_timestamp
from torch import nn
from torchrec.distributed.model_parallel import DMPCollection

# Bounds the simultaneous packed send and receive buffers, including pipeline overlap.
SYNC_CHUNK_BYTES = 256 * 1024**2
TOUCHED_KEY_SCAN_BATCH = 1 << 22


@dataclass
class TableSyncStats:
    """The keys one sync merged for a table, and how many of them this rank held."""

    table_name: str
    union_keys: int
    local_keys: int


def replica_sync_errors(
    kernels: Sequence[nn.Module], shard_ranks: List[int]
) -> List[str]:
    """Why each DynamicEmb kernel cannot be merged across replicas; empty when all can.

    ``shard_ranks`` are the global ranks of this rank's sharding group, the ranks
    a table of the replica is expected to be sharded over.
    """
    errors: List[str] = []
    for kernel in kernels:
        names = list(kernel.table_names)
        if kernel.cache is not None or not isinstance(kernel.tables, DynamicEmbStorage):
            errors.append(
                f"{names} keep values outside HBM ({type(kernel.tables).__name__}"
                f"{' with a cache' if kernel.cache is not None else ''}); the sync "
                "reads and writes rows in HBM only."
            )
            continue
        if kernel.shard_ranks != shard_ranks:
            errors.append(
                f"{names} are sharded over ranks {kernel.shard_ranks}, not over this "
                f"rank's sharding group {shard_ranks}; only row-wise tables of the "
                "DMPCollection's default context can be merged."
            )
            continue
        state = kernel.tables.tables[0]
        for table_id, name in enumerate(names):
            strategy = state.options_list[table_id].score_strategy
            if not score_strategy_has_timestamp_column(strategy):
                errors.append(
                    f"{name} has score_strategy {strategy}, which keeps no last-access "
                    "timestamp; the sync finds the keys touched since the last one by "
                    "it, so use TIMESTAMP or (TIMESTAMP, LFU)."
                )
    return errors


def gather_unique_keys(keys: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """The sorted union of ``keys`` over ``group``, the same tensor on every rank."""
    group_size = dist.get_world_size(group)
    local_count = torch.tensor([keys.numel()], dtype=torch.int64, device=keys.device)
    counts = torch.empty(group_size, dtype=torch.int64, device=keys.device)
    dist.all_gather_into_tensor(counts, local_count, group=group)
    per_rank = counts.tolist()
    max_count = max(per_rank)
    if max_count == 0:
        return keys[:0]
    padded = torch.zeros(max_count, dtype=keys.dtype, device=keys.device)
    padded[: keys.numel()] = keys
    gathered = torch.empty(group_size * max_count, dtype=keys.dtype, device=keys.device)
    dist.all_gather_into_tensor(gathered, padded, group=group)
    parts = [
        gathered[rank * max_count : rank * max_count + count]
        for rank, count in enumerate(per_rank)
    ]
    # Sorted, so that every replica merges the rows in the same order.
    return torch.unique(torch.cat(parts))


class _RowLayout(NamedTuple):
    embedding: int
    opt_columns: int
    opt_bytes: int
    bucket_capacity: int
    scores: int


def _buckets(state, table_id):
    key_map = state.key_index_map
    start, end = key_map.table_bucket_offsets_cpu_[table_id : table_id + 2].tolist()
    width = key_map.bucket_capacity_ * sum(key_map.fields_byte_)
    return key_map.table_storage_[start * width : end * width]


def _exchange(packet, group):
    replicas = dist.get_world_size(group)
    received = torch.empty(
        replicas, packet.numel(), device=packet.device, dtype=packet.dtype
    )
    work = dist.all_gather_into_tensor(
        received.view(-1), packet, group=group, async_op=True
    )
    return work, received


class DynamicEmbReplicaSync:
    """Merges the DynamicEmb tables of a ``DMPCollection``'s replicas by key.

    Call :meth:`sync` wherever ``DMPCollection.sync`` is called. Each sync takes
    the keys any replica touched since the last one (the table's last-access
    timestamp says which), averages the embedding, the optimizer state and the
    frequency score of each over the replica group, and writes the result into
    every replica, inserting the keys a replica did not hold.

    A replica that never held a key counts as holding a fresh row when the
    table's initializer is ``CONSTANT``: the initial value and a fresh optimizer
    state. That is the average a single, unreplicated table would reach, and the
    rule TorchRec 2D applies to a TBE row only some replicas updated. With a
    random initializer there is no one fresh row, and only the replicas that hold
    the key are averaged.

    Parameters
    ----------
    model : DMPCollection
        The 2D model whose DynamicEmb tables are kept in agreement. Its tables must
        be sharded row-wise over the default context's sharding group, keep their
        values in HBM (no cache, no host tier), and keep a timestamp score.
    max_staging_bytes : int
        Per-rank budget for communication payloads (default 256 MiB). All
        replicas use the smallest requested budget. A packed NCCL slab holds
        at least one row.
    """

    def __init__(
        self, model: DMPCollection, *, max_staging_bytes: int = SYNC_CHUNK_BYTES
    ) -> None:
        mesh = model.device_mesh
        self.replica_group = mesh.get_group("replicate")
        self.replica_count = dist.get_world_size(self.replica_group)
        budget = torch.tensor(max_staging_bytes, dtype=torch.long, device="cuda")
        dist.all_reduce(budget, op=dist.ReduceOp.MIN, group=self.replica_group)
        self.max_staging_bytes = budget.item()
        if self.max_staging_bytes <= 0:
            raise ValueError("max_staging_bytes must be positive on every replica")
        self._peer = (
            ReplicaPeer(self.replica_group, self.max_staging_bytes)
            if self.replica_count == 2
            else None
        )
        self.kernels = get_dynamic_emb_module(model)
        shard_ranks = dist.get_process_group_ranks(mesh.get_group("shard"))
        if errors := replica_sync_errors(self.kernels, shard_ranks):
            raise ValueError(
                "DynamicEmb tables cannot be merged across 2D replicas: "
                + " ".join(errors)
            )
        # Until mark_synced, a sync merges every key the tables hold.
        self.threshold = 0
        self._dense_routes = {}

    def mark_synced(self) -> None:
        """Take now as when the replicas last agreed; a sync covers keys touched since.

        Call it once all replicas hold the same rows without a sync, e.g. after
        every replica has loaded the same checkpoint.
        """
        torch.cuda.synchronize()
        self.threshold = device_timestamp()

    def sync(self) -> List[TableSyncStats]:
        """Merge every row touched on any replica since the last sync into all replicas."""
        stats = [
            self.sync_table(kernel, table_id, name)
            for kernel in self.kernels
            for table_id, name in enumerate(kernel.table_names)
        ]
        self.mark_synced()
        return stats

    def _route(self, kernel, table_id, chunk_rows):
        state = kernel.tables.tables[0]
        key_map = state.key_index_map
        token = (kernel, table_id)
        previous = self._dense_routes.get(token)
        if previous is not None:
            keys, rows, slots = previous
            start, end = key_map.table_bucket_offsets_cpu_[
                table_id : table_id + 2
            ].tolist()
            capacity = key_map.bucket_capacity_
            valid = (key_map.bucket_sizes[start:end].sum() == keys.numel()).int()
            torch.ops.dyn_emb.replica_validate_(
                _buckets(state, table_id),
                key_map.rows_[start * capacity : end * capacity],
                keys,
                rows,
                slots,
                capacity,
                key_map.num_scores_,
                self.threshold,
                valid,
            )
            dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=self.replica_group)
            if valid.item():
                counts = [
                    [min(chunk_rows, keys.numel() - i)] * self.replica_count
                    for i in range(0, keys.numel(), chunk_rows)
                ]
                return keys, None, rows, slots, counts
            del self._dense_routes[token]
            if self._peer is not None:
                self._peer.routes.pop(token, None)

        # Consume the same threshold scan as incremental_dump directly on the
        # device. Checkpoint-oriented incremental_dump stages its output on CPU.
        parts = [
            keys
            for keys, _, _ in key_map._batched_export_keys_scores(
                [state.incremental_score_name],
                state.device,
                table_id,
                [self.threshold],
                TOUCHED_KEY_SCAN_BATCH,
            )
        ]
        touched = (
            torch.cat(parts).to(torch.int64)
            if parts
            else torch.empty(0, dtype=torch.int64, device=state.device)
        )
        keys = gather_unique_keys(touched, self.replica_group)
        if keys.numel() == 0:
            return keys, None, None, None, []
        table_ids = torch.full_like(keys, table_id)
        lookup = get_find_score_arg(
            state, keys.numel(), state.device, const_lookup=True
        )
        _, held, rows, slots = key_map.lookup(
            keys.to(key_map.key_type), table_ids, lookup, return_slots=True
        )

        # Exchange all chunk counts in one collective and one device-to-host
        # transfer. A fully held chunk needs neither masks nor compaction.
        count = torch.stack([mask.sum() for mask in held.split(chunk_rows)])
        work, counts_gpu = _exchange(count, self.replica_group)
        work.wait()
        counts = counts_gpu.T.tolist()
        if all(
            min(per_rank) == min(chunk_rows, keys.numel() - i * chunk_rows)
            for i, per_rank in enumerate(counts)
        ):
            self._dense_routes[token] = keys, rows, slots
        return keys, held, rows, slots, counts

    def sync_table(self, kernel: nn.Module, table_id: int, name: str) -> TableSyncStats:
        """Merge one table's touched rows over the replica group and write them back."""
        storage = kernel.tables
        state = storage.tables[0]
        key_map = state.key_index_map
        layout = _RowLayout(
            state.table_emb_dims_cpu[table_id],
            state.optimizer.get_ckpt_state_dim(state.table_emb_dims_cpu[table_id]),
            state.optimizer.get_state_dtype(state.emb_dtype).itemsize,
            key_map.bucket_capacity_,
            key_map.num_scores_,
        )
        initializer = state.options_list[table_id].initializer_args
        constant = initializer.mode == DynamicEmbInitializerMode.CONSTANT
        fresh = torch.empty(
            1,
            state.table_value_dims_cpu[table_id],
            dtype=state.emb_dtype,
            device=state.device,
        )
        if constant:
            initial = torch.tensor([initializer.value], device=state.device)
            fresh[:, : layout.embedding] = encode_table_values(initial, state.emb_dtype)
        state.optimizer.reset_optimizer_states(
            fresh[:, layout.embedding :], emb_dims=layout.embedding
        )

        payload = (
            layout.embedding * fresh.element_size()
            + layout.opt_columns * layout.opt_bytes
        )
        wire_bytes = ((payload + 15) // 16) * 16 + (layout.scores - 1) * 8
        chunk_rows = max(
            1, self.max_staging_bytes // (2 * (self.replica_count + 1) * wire_bytes)
        )
        keys, held, rows, slots, counts = self._route(kernel, table_id, chunk_rows)
        if not counts:
            return TableSyncStats(name, 0, 0)
        replica = dist.get_rank(self.replica_group)
        local_keys = sum(c[replica] for c in counts)
        if (
            self._peer is not None
            and layout.scores == 1
            and (not layout.opt_columns or layout.opt_bytes == fresh.element_size())
            and payload == fresh.numel() * fresh.element_size()
            and state.emb_dtype in (torch.float32, torch.float16, torch.bfloat16)
            and payload % 16 == 0
            and payload * keys.numel() >= 8 << 20
            and all(sum(c[r] for c in counts) == keys.numel() for r in range(2))
            and self._peer.sync(
                (kernel, table_id),
                state.tables[table_id].tensor(),
                _buckets(state, table_id),
                rows,
                slots,
                layout.bucket_capacity,
            )
        ):
            return TableSyncStats(name, keys.numel(), local_keys)
        if self._peer is not None:
            self._peer.release_workspace()
        lookup = get_find_score_arg(
            state, keys.numel(), state.device, const_lookup=True
        )
        # Mutable score time comes from %globaltimer in the writeback kernel.
        now = 0
        pending = None

        def finish(pending):
            work, received, indices, destinations, positions = pending
            work.wait()
            torch.ops.dyn_emb.replica_merge_(
                state.tables[table_id].tensor(),
                _buckets(state, table_id),
                destinations,
                positions,
                received,
                indices,
                fresh,
                constant,
                now,
                layout,
            )

        refresh_lookup = False
        for chunk, counts_per_rank in enumerate(counts):
            begin = chunk * chunk_rows
            stop = min(begin + chunk_rows, keys.numel())
            selected = slice(begin, stop)
            chunk_keys = keys[selected].to(key_map.key_type)
            chunk_ids = None
            chunk_rows_local, chunk_slots = rows[selected], slots[selected]
            mask = held[selected] if held is not None else None
            if refresh_lookup or min(counts_per_rank) != stop - begin:
                chunk_ids = torch.full_like(keys[selected], table_id)
            if refresh_lookup:
                # Insertion in an earlier chunk may have grown a table or
                # evicted an as-yet unprocessed key. Refresh both coordinates
                # and held counts before packing, on every rank.
                _, mask, chunk_rows_local, chunk_slots = state.key_index_map.lookup(
                    chunk_keys, chunk_ids, lookup, return_slots=True
                )
                work, updated = _exchange(mask.sum().reshape(1), self.replica_group)
                work.wait()
                counts_per_rank = updated.flatten().tolist()
            dense = min(counts_per_rank) == stop - begin
            indices = None
            if dense:
                source_rows, source_slots = chunk_rows_local, chunk_slots
            else:
                if pending:
                    finish(pending)
                    pending = None
                source_rows, source_slots = chunk_rows_local[mask], chunk_slots[mask]
                work, masks = _exchange(mask.to(torch.uint8), self.replica_group)
                work.wait()
                indices = masks.cumsum(dim=1, dtype=torch.int32) - 1
                indices.masked_fill_(masks == 0, -1)
            packet = torch.ops.dyn_emb.replica_pack(
                state.tables[table_id].tensor(),
                _buckets(state, table_id),
                source_rows,
                source_slots,
                max(counts_per_rank),
                layout,
            )
            work, received = _exchange(packet, self.replica_group)
            # Enqueue the next chunk's all-gather before waiting on the last
            # one, allowing NCCL to run while the last chunk is averaged.
            if pending:
                finish(pending)
            if not dense and counts_per_rank[replica] < stop - begin:
                missing = (~mask) & masks.any(dim=0)
                # Index as signed bits; PyTorch lacks uint64 boolean indexing.
                missing_keys = keys[selected][missing].to(key_map.key_type)
                if missing_keys.numel():
                    per_table = torch.zeros(len(state.options_list), dtype=torch.int64)
                    per_table[table_id] = missing_keys.numel()
                    storage.expand_if_need(per_table)
                    storage.insert(
                        missing_keys,
                        chunk_ids[missing],
                        torch.zeros(
                            missing_keys.numel(),
                            state.value_dim,
                            device=state.device,
                            dtype=state.emb_dtype,
                        ),
                    )
                    _, _, chunk_rows_local, chunk_slots = state.key_index_map.lookup(
                        chunk_keys, chunk_ids, lookup, return_slots=True
                    )
            pending = work, received, indices, chunk_rows_local, chunk_slots
            if not dense:
                refresh_lookup = True
                finish(pending)
                pending = None
        if pending:
            finish(pending)
        return TableSyncStats(name, keys.numel(), local_keys)
