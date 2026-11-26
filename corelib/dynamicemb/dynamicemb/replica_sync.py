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
from typing import List, Sequence

import torch
import torch.distributed as dist
from dynamicemb.dump_load import get_dynamic_emb_module
from dynamicemb.dynamicemb_config import score_strategy_has_timestamp_column
from dynamicemb.key_value_table import DynamicEmbStorage
from dynamicemb.replica_peer import peer_state
from dynamicemb.types import DynamicEmbInitializerMode
from dynamicemb.utils import encode_table_values
from dynamicemb_extensions import device_timestamp, table_scatter_score_blocks
from torch import nn
from torchrec.distributed.model_parallel import DMPCollection

# Bounds the simultaneous packed send and receive buffers, including pipeline overlap.
SYNC_CHUNK_BYTES = 256 * 1024**2


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


def _install_missing(storage, keys, values, scores):
    """Grow the local storage once and install the rows computed by the batch op."""
    state = storage.tables[0]
    counts = [k.numel() for k in keys]
    total = sum(counts)
    if not total:
        return
    storage.expand_if_need(torch.tensor(counts, dtype=torch.int64, device="cpu"))
    key_map = state.key_index_map
    ids = torch.cat([torch.full_like(k, t) for t, k in enumerate(keys) if k.numel()])
    flat_keys = torch.cat(keys).to(key_map.key_type)
    padded = torch.empty(
        total, state.value_dim, device=state.device, dtype=state.emb_dtype
    )
    # Fused storage takes the optimizer columns at the maximum embedding width.
    offset = 0
    for emb_dim, rows in zip(state.table_emb_dims_cpu, values):
        if rows.size(0):
            target = padded.narrow(0, offset, rows.size(0))
            target[:, :emb_dim].copy_(rows[:, :emb_dim])
            opt_width = rows.size(1) - emb_dim
            target[:, state.emb_dim : state.emb_dim + opt_width].copy_(
                rows[:, emb_dim:]
            )
            offset += rows.size(0)
    storage.insert(flat_keys, ids, padded)
    slots = key_map.slots_of(flat_keys, ids)
    absolute = slots + key_map.table_bucket_offsets_[ids] * key_map.bucket_capacity_
    table_scatter_score_blocks(
        key_map.table_storage_,
        key_map.bucket_capacity_,
        key_map.num_scores_,
        0,
        absolute.masked_fill_(slots < 0, -1),
        torch.cat(scores),
        key_map.key_type,
    )


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
        self.kernels = get_dynamic_emb_module(model)
        shard_ranks = dist.get_process_group_ranks(mesh.get_group("shard"))
        if errors := replica_sync_errors(self.kernels, shard_ranks):
            raise ValueError(
                "DynamicEmb tables cannot be merged across 2D replicas: "
                + " ".join(errors)
            )
        # Until mark_synced, a sync merges every key the tables hold.
        self.threshold = 0
        self._routes = []
        self._peer = peer_state(self.replica_group, torch.cuda.current_device())
        self._generations = None
        self._names = []
        self._fresh = []
        self._layouts = []
        for kernel in self.kernels:
            state = kernel.tables.tables[0]
            key_map = state.key_index_map
            for i, name in enumerate(kernel.table_names):
                emb = state.table_emb_dims_cpu[i]
                initializer = state.options_list[i].initializer_args
                constant = initializer.mode == DynamicEmbInitializerMode.CONSTANT
                fresh = torch.empty(
                    (1, state.table_value_dims_cpu[i]),
                    dtype=state.emb_dtype,
                    device=state.device,
                )
                if constant:
                    initial = torch.tensor([initializer.value], device=state.device)
                    fresh[:, :emb] = encode_table_values(initial, state.emb_dtype)
                state.optimizer.reset_optimizer_states(fresh[:, emb:], emb_dims=emb)
                self._names.append(name)
                self._fresh.append(fresh)
                self._layouts.append(
                    (
                        emb,
                        state.optimizer.get_ckpt_state_dim(emb),
                        state.optimizer.get_state_dtype(state.emb_dtype).itemsize,
                        key_map.bucket_capacity_,
                        key_map.num_scores_,
                        constant,
                        key_map.global_capacities_[i],
                    )
                )

    def _refresh_views(self):
        """Refresh table slices when a rehash or value-buffer extension changes them."""
        states = [kernel.tables.tables[0] for kernel in self.kernels]
        generations = [
            (s.key_map_generation, s.value_buffer_generation) for s in states
        ]
        if generations == self._generations:
            return
        self._values, self._buckets, self._rows, self._sizes = [], [], [], []
        for state in states:
            key_map = state.key_index_map
            offsets = key_map.table_bucket_offsets_cpu_.tolist()
            cap = key_map.bucket_capacity_
            width = cap * sum(key_map.fields_byte_)
            for table, start, end in zip(state.tables, offsets, offsets[1:]):
                self._values.append(table.tensor())
                self._buckets.append(
                    key_map.table_storage_[start * width : end * width]
                )
                self._rows.append(key_map.rows_[start * cap : end * cap])
                self._sizes.append(key_map.bucket_sizes[start:end])
        self._generations = generations

    def mark_synced(self) -> None:
        """Take now as when the replicas last agreed; a sync covers keys touched since.

        Call it once all replicas hold the same rows without a sync, e.g. after
        every replica has loaded the same checkpoint.
        """
        torch.cuda.synchronize()
        self.threshold = device_timestamp()

    def sync(self) -> List[TableSyncStats]:
        """Merge every row touched on any replica since the last sync into all replicas."""
        if not self._names:
            self.mark_synced()
            return []
        self._refresh_views()
        (
            counts,
            self._routes,
            keys,
            values,
            scores,
            self._peer,
        ) = torch.ops.dyn_emb.replica_sync_batch_(
            self._values,
            self._buckets,
            self._rows,
            self._sizes,
            self._fresh,
            self._layouts,
            self._routes,
            self._peer,
            self.replica_group.group_name,
            self.threshold,
            self.max_staging_bytes,
        )
        stats = [
            TableSyncStats(name, union, local)
            for name, (union, local) in zip(self._names, counts.tolist())
        ]
        if any(entry.local_keys < entry.union_keys for entry in stats):
            offset = 0
            for kernel in self.kernels:
                end = offset + len(kernel.table_names)
                _install_missing(
                    kernel.tables,
                    keys[offset:end],
                    values[offset:end],
                    scores[offset:end],
                )
                offset = end
        self.mark_synced()
        return stats
