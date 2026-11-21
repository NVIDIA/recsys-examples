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
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist
from dynamicemb.dump_load import get_dynamic_emb_module
from dynamicemb.dynamicemb_config import (
    DynamicEmbScoreStrategy,
    get_physical_score_order,
    score_strategy_has_timestamp_column,
)
from dynamicemb.key_value_table import (
    DynamicEmbStorage,
    get_find_score_arg,
    load_from_flat_single_table,
    store_to_flat_single_table,
)
from dynamicemb.types import DynamicEmbInitializerMode
from dynamicemb.utils import checkpoint_dtype, decode_table_values, encode_table_values
from dynamicemb_extensions import device_timestamp
from torch import nn
from torchrec.distributed.model_parallel import DMPCollection

# Bounds the fp32 staging buffers of one exchange; a chunk of keys is merged at a time.
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


def gather_held_rows(
    rows: torch.Tensor, counts: List[int], group: dist.ProcessGroup
) -> List[torch.Tensor]:
    """Each rank's ``rows`` over ``group``, where rank r holds ``counts[r]`` of them."""
    max_count = max(counts)
    padded = torch.zeros(
        max_count, *rows.shape[1:], dtype=rows.dtype, device=rows.device
    )
    padded[: rows.size(0)] = rows
    gathered = torch.empty(
        len(counts) * max_count, *rows.shape[1:], dtype=rows.dtype, device=rows.device
    )
    if padded.numel():
        dist.all_gather_into_tensor(gathered, padded, group=group)
    return [
        gathered[rank * max_count : rank * max_count + count]
        for rank, count in enumerate(counts)
    ]


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
    """

    def __init__(self, model: DMPCollection) -> None:
        mesh = model.device_mesh
        self.replica_group = mesh.get_group("replicate")
        self.replica_count = dist.get_world_size(self.replica_group)
        self.kernels = get_dynamic_emb_module(model)
        shard_ranks = dist.get_process_group_ranks(mesh.get_group("shard"))
        if errors := replica_sync_errors(self.kernels, shard_ranks):
            raise ValueError(
                "DynamicEmb tables cannot be merged across 2D replicas: "
                + " ".join(errors)
            )
        # Until mark_synced, a sync merges every key the tables hold.
        self.threshold = 0

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

    def sync_table(self, kernel: nn.Module, table_id: int, name: str) -> TableSyncStats:
        """Merge one table's touched rows over the replica group and write them back."""
        storage = kernel.tables
        state = storage.tables[0]
        touched = state.key_index_map.incremental_dump(
            {state.incremental_score_name: self.threshold},
            table_id=table_id,
            batch_size=TOUCHED_KEY_SCAN_BATCH,
        )[0]
        union = gather_unique_keys(
            touched.to(device=state.device, dtype=torch.int64), self.replica_group
        )
        emb_dim = int(state.table_emb_dims_cpu[table_id])
        value_dim = int(state.table_value_dims_cpu[table_id])
        # (1, opt_dim): a fresh row's optimizer state at checkpoint width, in fp32.
        fresh_state = torch.empty(
            1, value_dim - emb_dim, dtype=state.emb_dtype, device=state.device
        )
        state.optimizer.reset_optimizer_states(fresh_state, emb_dims=emb_dim)
        fresh_opt = state.optimizer.states_for_checkpoint(fresh_state, emb_dim).float()
        initializer = state.options_list[table_id].initializer_args
        fresh_emb = None
        if initializer.mode == DynamicEmbInitializerMode.CONSTANT:
            # The initial value as the table stores it.
            fresh_emb = decode_table_values(
                encode_table_values(
                    torch.tensor([initializer.value], device=state.device),
                    state.emb_dtype,
                ),
                torch.float32,
            ).item()
        row_bytes = 4 * (emb_dim + fresh_opt.size(1)) + 16
        chunk_rows = max(1, SYNC_CHUNK_BYTES // row_bytes)
        local_keys = 0
        for begin in range(0, union.numel(), chunk_rows):
            local_keys += self.sync_chunk(
                storage,
                table_id,
                union[begin : begin + chunk_rows],
                fresh_opt,
                fresh_emb,
            )
        return TableSyncStats(name, int(union.numel()), local_keys)

    def sync_chunk(
        self,
        storage: DynamicEmbStorage,
        table_id: int,
        keys: torch.Tensor,
        fresh_opt: torch.Tensor,
        fresh_emb: Optional[float],
    ) -> int:
        """Merge and write back one chunk of keys; returns how many this rank held."""
        state = storage.tables[0]
        device = state.device
        num_keys = keys.numel()
        table_ids = torch.full((num_keys,), table_id, dtype=torch.int64, device=device)
        emb_dim = int(state.table_emb_dims_cpu[table_id])
        opt_dim = fresh_opt.size(1)
        key_map = state.key_index_map
        keys = keys.to(key_map.key_type)
        score_arg = get_find_score_arg(state, num_keys, device, const_lookup=True)
        _, founds, rows, slots = key_map.lookup(
            keys, table_ids, score_arg, return_slots=True
        )
        # (num_held, value_dim), at the table's storage precision.
        held = load_from_flat_single_table(state, rows[founds], table_id)
        # Embeddings travel at the precision a checkpoint holds them at, which is
        # lossless (fp16 for an fp8 table), and add up in fp32.
        held_emb = decode_table_values(
            held[:, :emb_dim], checkpoint_dtype(state.emb_dtype)
        )
        held_opt = state.optimizer.states_for_checkpoint(held[:, emb_dim:], emb_dim)
        held_scores = key_map.gather_score_blocks(table_id, slots[founds])
        # Only held rows travel, in key order, with a mask saying which keys they
        # are: the sync is bound by bandwidth.
        masks = torch.empty(
            self.replica_count, num_keys, dtype=torch.uint8, device=device
        )
        dist.all_gather_into_tensor(
            masks, founds.to(torch.uint8), group=self.replica_group
        )
        held_masks = masks.bool()  # (replica_count, num_keys)
        counts = held_masks.sum(dim=1).tolist()
        holders = held_masks.sum(dim=0)  # (num_keys,)
        emb_sum = torch.zeros(num_keys, emb_dim, dtype=torch.float32, device=device)
        opt_sum = torch.zeros(num_keys, opt_dim, dtype=torch.float32, device=device)
        scores = torch.zeros(
            num_keys, key_map.num_scores_, dtype=torch.int64, device=device
        )
        for total, local in (
            (emb_sum, held_emb),
            (opt_sum, held_opt.float()),
            (scores, held_scores.to(torch.int64)),
        ):
            parts = gather_held_rows(local.contiguous(), counts, self.replica_group)
            for replica, part in enumerate(parts):
                total[held_masks[replica]] += part.to(total.dtype)
        if fresh_emb is not None:
            missing = (self.replica_count - holders).unsqueeze(1).float()
            emb_sum += missing * fresh_emb
            opt_sum += missing * fresh_opt
            divisor = torch.full(
                (num_keys, 1), float(self.replica_count), device=device
            )
        else:
            divisor = holders.clamp(min=1).unsqueeze(1).float()
        keep = holders > 0
        emb_mean = emb_sum[keep] / divisor[keep]
        opt_mean = opt_sum[keep] / divisor[keep]
        keys, table_ids, scores = keys[keep], table_ids[keep], scores[keep]
        new_keys = int((keep & ~founds).sum().item())
        if new_keys:
            new_per_table = torch.zeros(len(state.options_list), dtype=torch.int64)
            new_per_table[table_id] = new_keys
            storage.expand_if_need(new_per_table)
        # The insert seats the keys this replica lacked; rows and scores are
        # written below, for every key.
        storage.insert(
            keys,
            table_ids,
            torch.zeros(
                keys.numel(), state.value_dim, dtype=state.emb_dtype, device=device
            ),
        )
        _, placed, rows, slots = key_map.lookup(
            keys, table_ids, score_arg, return_slots=True
        )
        # Both halves at the table's storage precision: the optimizer packs an
        # fp32 state into the words of an fp16 or fp8 row.
        opt = state.optimizer.states_from_checkpoint(
            opt_mean, emb_dim, state.emb_dtype, device
        )
        values = torch.cat(
            [encode_table_values(emb_mean, state.emb_dtype), opt.to(state.emb_dtype)],
            dim=1,
        )
        store_to_flat_single_table(state, rows[placed], table_id, values[placed])

        now = device_timestamp()
        physical = get_physical_score_order(state.options_list[table_id].score_strategy)
        for word, strategy in enumerate(physical):
            if strategy == DynamicEmbScoreStrategy.TIMESTAMP:
                scores[:, word] = now
            elif strategy == DynamicEmbScoreStrategy.LFU:
                # A key one replica saw once still ranks as a fresh insert (count 1).
                scores[:, word] = (scores[:, word] // self.replica_count).clamp(min=1)
            else:
                scores[:, word] = scores[:, word] // self.replica_count
        key_map.scatter_score_blocks(table_id, slots[placed], scores[placed])
        return int(founds.sum().item())
