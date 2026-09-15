# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Sequence

import torch
from dynamicemb.scored_hashtable import (
    LinearBucketTable,
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
)
from dynamicemb_extensions import table_partition

from . import _C


_SNAPSHOT_MAGIC = b"ECIDX001"
_SNAPSHOT_HEADER = struct.Struct("<8sIIqqQQQQQ")
_SNAPSHOT_SCHEMA_VERSION = 1


class IndexerSnapshotKind(IntEnum):
    # Persisted in the snapshot header; keep aligned with indexer_snapshot.h.
    LINEAR_HASH_MAP = 0
    FUSED_IDENTITY = 1


@dataclass
class IndexerSnapshot:
    kind: IndexerSnapshotKind
    table_storage: torch.Tensor
    table_bucket_offsets: torch.Tensor
    bucket_capacity: int
    miss_storage_indices: torch.Tensor
    valid_bases: torch.Tensor
    reserved_sizes: torch.Tensor
    next_fused_key: int

    def to(self, device: torch.device) -> "IndexerSnapshot":
        return IndexerSnapshot(
            kind=self.kind,
            table_storage=self.table_storage.to(device),
            table_bucket_offsets=self.table_bucket_offsets.to(device),
            bucket_capacity=self.bucket_capacity,
            miss_storage_indices=self.miss_storage_indices.to(device),
            valid_bases=self.valid_bases.to(device),
            reserved_sizes=self.reserved_sizes.to(device),
            next_fused_key=self.next_fused_key,
        )


def _write_tensor(stream: Any, tensor: torch.Tensor) -> None:
    stream.write(tensor.detach().cpu().contiguous().numpy().tobytes())


def dump_indexer_snapshot(snapshot: IndexerSnapshot, path: str | Path) -> None:
    path = Path(path)
    tensors = (
        snapshot.table_storage,
        snapshot.table_bucket_offsets,
        snapshot.miss_storage_indices,
        snapshot.valid_bases,
        snapshot.reserved_sizes,
    )
    lengths = [tensor.numel() for tensor in tensors]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(
            _SNAPSHOT_HEADER.pack(
                _SNAPSHOT_MAGIC,
                _SNAPSHOT_SCHEMA_VERSION,
                int(snapshot.kind),
                snapshot.bucket_capacity,
                snapshot.next_fused_key,
                *lengths,
            )
        )
        for tensor in tensors:
            _write_tensor(stream, tensor)


def load_embedding_collection_indexer_snapshot(
    path: str | Path, device: torch.device
) -> Any:
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return _C.load_indexer_snapshot(str(Path(path)), device_index)


def dump_embedding_collection_indexer_snapshot(
    *,
    collection_id: str,
    snapshot_id: int,
    snapshot: IndexerSnapshot,
    output_dir: str | Path,
) -> str:
    path = Path(output_dir).resolve() / f"{collection_id}.{snapshot_id}.bin"
    dump_indexer_snapshot(snapshot, path)
    return str(path)


def native_indexer_snapshot(snapshot: IndexerSnapshot) -> Any:
    return _C.IndexerSnapshot(
        int(snapshot.kind),
        snapshot.table_storage,
        snapshot.table_bucket_offsets,
        snapshot.bucket_capacity,
        snapshot.miss_storage_indices,
        snapshot.valid_bases,
        snapshot.reserved_sizes,
        snapshot.next_fused_key,
    )


def linear_hash_entries(
    snapshot: IndexerSnapshot,
) -> list[dict[int, int]]:
    storage = snapshot.table_storage.cpu().contiguous()
    offsets = snapshot.table_bucket_offsets.cpu().tolist()
    num_buckets = offsets[-1]
    keys, _digests, values = table_partition(
        storage,
        [torch.int64, torch.uint8, torch.uint64],
        snapshot.bucket_capacity,
        num_buckets,
    )
    result: list[dict[int, int]] = []
    for table_id in range(len(offsets) - 1):
        table_keys = keys[offsets[table_id] : offsets[table_id + 1]].reshape(-1)
        table_values = values[
            offsets[table_id] : offsets[table_id + 1]
        ].reshape(-1)
        present = table_keys != -1
        result.append(
            dict(
                zip(
                    table_keys[present].tolist(),
                    table_values[present].view(torch.int64).tolist(),
                )
            )
        )
    return result


@torch.no_grad()
def rebuild_linear_hash_snapshot(
    entries: Sequence[dict[int, int]],
    *,
    bucket_capacity: int,
    miss_storage_indices: torch.Tensor,
    next_fused_key: int,
    device: torch.device,
) -> IndexerSnapshot:
    logical_capacities = [max(1, len(table)) for table in entries]
    while True:
        table = LinearBucketTable(
            [
                max(bucket_capacity, 2 * capacity)
                for capacity in logical_capacities
            ],
            [ScoreSpec(name="fused_key", policy=ScorePolicy.ASSIGN)],
            key_type=torch.int64,
            bucket_capacity=bucket_capacity,
            device=device,
        )
        failed = False
        for table_id, mapping in enumerate(entries):
            if not mapping:
                continue
            feature_ids = torch.tensor(
                list(mapping), dtype=torch.int64, device=device
            )
            table_ids = torch.full_like(feature_ids, table_id)
            fused_keys = torch.tensor(
                list(mapping.values()), dtype=torch.int64, device=device
            )
            slots, evicted_count, _keys, _table_ids = table.insert(
                feature_ids,
                table_ids,
                ScoreArg(
                    name="fused_key",
                    value=fused_keys,
                    policy=ScorePolicy.ASSIGN,
                ),
                collect_evicted=True,
            )
            if (slots < 0).any() or int(evicted_count.item()) != 0:
                logical_capacities[table_id] *= 2
                failed = True
                break
        if not failed:
            break

    empty = torch.empty(0, dtype=torch.int64, device=device)
    return IndexerSnapshot(
        kind=IndexerSnapshotKind.LINEAR_HASH_MAP,
        table_storage=table.table_storage_,
        table_bucket_offsets=table.table_bucket_offsets_,
        bucket_capacity=table.bucket_capacity_,
        miss_storage_indices=miss_storage_indices.to(device),
        valid_bases=empty,
        reserved_sizes=empty,
        next_fused_key=next_fused_key,
    )
