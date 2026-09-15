# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
from dynamicemb.scored_hashtable import (
    LinearBucketTable,
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
)

from .config import (
    EmbeddingCollectionIndexerType,
    InferenceEmbeddingCollectionConfig,
)
from .indexer_snapshot import IndexerSnapshot, IndexerSnapshotKind


def _new_marker_id() -> int:
    return uuid.uuid4().int & ((1 << 63) - 1)


def _empty_i64(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64, device=device)


def _publish_native(marker: torch.Tensor, snapshot: IndexerSnapshot) -> None:
    torch.ops.INFERENCE_EMB.register_embedding_collection_indexer(
        marker,
        int(snapshot.kind),
        snapshot.table_storage,
        snapshot.table_bucket_offsets,
        snapshot.bucket_capacity,
        snapshot.miss_storage_indices,
        snapshot.valid_bases,
        snapshot.reserved_sizes,
        snapshot.next_fused_key,
    )


class EmbeddingCollectionIndexerBase(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self._failed_build_rows: set[tuple[int, int]] = set()

    @property
    def failed_build_rows(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._failed_build_rows)

    @property
    @abstractmethod
    def nve_num_embeddings(self) -> int:
        pass

    @property
    def snapshot(self) -> Optional[IndexerSnapshot]:
        return None

    @torch.no_grad()
    @abstractmethod
    def build_index(
        self, feature_ids: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        pass

    def finish_build(self) -> None:
        pass


class LinearHashMapIndexer(EmbeddingCollectionIndexerBase):
    """Map arbitrary per-table feature IDs into one stable linear key space."""

    def __init__(
        self,
        table_capacities: Sequence[int],
        *,
        bucket_capacity: int = 128,
        key_type: torch.dtype = torch.int64,
        state_sidecar: bool = True,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.table_capacities = tuple(int(value) for value in table_capacities)
        self.key_type = key_type
        self.state_sidecar = state_sidecar
        self._nve_num_embeddings = len(self.table_capacities) + sum(
            self.table_capacities
        )
        self.marker_value_ = _new_marker_id()
        self.register_buffer(
            "marker_tensor",
            torch.tensor([self.marker_value_], dtype=torch.int64, device=device),
            persistent=False,
        )
        self._build_table = LinearBucketTable(
            [
                max(bucket_capacity, 2 * capacity)
                for capacity in self.table_capacities
            ],
            [ScoreSpec(name="fused_key", policy=ScorePolicy.ASSIGN)],
            key_type=key_type,
            bucket_capacity=bucket_capacity,
            device=device,
        )
        self.bucket_capacity = self._build_table.bucket_capacity_
        empty = _empty_i64(device)
        self._snapshot = IndexerSnapshot(
            kind=IndexerSnapshotKind.LINEAR_HASH_MAP,
            table_storage=self._build_table.table_storage_,
            table_bucket_offsets=self._build_table.table_bucket_offsets_,
            bucket_capacity=self.bucket_capacity,
            miss_storage_indices=torch.arange(
                len(self.table_capacities), dtype=torch.int64, device=device
            ),
            valid_bases=empty,
            reserved_sizes=empty,
            next_fused_key=len(self.table_capacities),
        )
        if not self.state_sidecar:
            self.register_buffer(
                "table_storage_", self._snapshot.table_storage
            )
            self.register_buffer(
                "table_bucket_offsets_", self._snapshot.table_bucket_offsets
            )
            self.register_buffer(
                "miss_storage_indices_", self._snapshot.miss_storage_indices
            )
        self._failed_candidates: set[tuple[int, int]] = set()
        if self.state_sidecar:
            _publish_native(self.marker_tensor, self._snapshot)

    @property
    def nve_num_embeddings(self) -> int:
        return self._nve_num_embeddings

    @property
    def snapshot(self) -> Optional[IndexerSnapshot]:
        return self._snapshot if self.state_sidecar else None

    @property
    def miss_storage_indices(self) -> torch.Tensor:
        return self._snapshot.miss_storage_indices

    @torch.no_grad()
    def reset_for_build(self) -> None:
        self._build_table.reset()
        self._snapshot.next_fused_key = len(self.table_capacities)
        self._failed_candidates.clear()
        self._failed_build_rows.clear()

    @torch.no_grad()
    def build_index(
        self, feature_ids: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        stored_keys, found, _slots = self._build_table.lookup(
            feature_ids,
            table_ids,
            ScoreArg(name="fused_key", policy=ScorePolicy.CONST),
        )
        missing_ids = feature_ids[~found]
        missing_tables = table_ids[~found]
        if missing_ids.numel() > 0:
            start = self._snapshot.next_fused_key
            assigned = torch.arange(
                start,
                start + missing_ids.numel(),
                dtype=torch.int64,
                device=feature_ids.device,
            )
            self._snapshot.next_fused_key += missing_ids.numel()
            slots, evicted_count, evicted_keys, evicted_tables = (
                self._build_table.insert(
                    missing_ids,
                    missing_tables,
                    ScoreArg(
                        name="fused_key",
                        value=assigned,
                        policy=ScorePolicy.ASSIGN,
                    ),
                    collect_evicted=True,
                )
            )
            evicted_count_value = int(evicted_count.item())
            direct_failures = slots < 0
            if direct_failures.any() or evicted_count_value:
                print(
                    "[WARNING] LinearHashMapIndexer insertion could not retain "
                    f"{int(direct_failures.sum().item()) + evicted_count_value} rows"
                )
            self._failed_candidates.update(
                zip(
                    missing_tables[direct_failures].cpu().tolist(),
                    missing_ids[direct_failures].cpu().tolist(),
                )
            )
            self._failed_candidates.update(
                zip(
                    evicted_tables[:evicted_count_value].cpu().tolist(),
                    evicted_keys[:evicted_count_value].cpu().tolist(),
                )
            )
            stored_keys, found, _slots = self._build_table.lookup(
                feature_ids,
                table_ids,
                ScoreArg(name="fused_key", policy=ScorePolicy.CONST),
            )

        return torch.where(found, stored_keys, torch.full_like(stored_keys, -1))

    @torch.no_grad()
    def finish_build(self) -> None:
        candidates = sorted(self._failed_candidates)
        if candidates:
            table_ids = torch.tensor(
                [table_id for table_id, _ in candidates],
                dtype=torch.int64,
                device=self.marker_tensor.device,
            )
            feature_ids = torch.tensor(
                [feature_id for _, feature_id in candidates],
                dtype=self.key_type,
                device=self.marker_tensor.device,
            )
            _stored, found, _slots = self._build_table.lookup(
                feature_ids,
                table_ids,
                ScoreArg(name="fused_key", policy=ScorePolicy.CONST),
            )
            self._failed_build_rows = {
                row
                for row, retained in zip(candidates, found.cpu().tolist())
                if not retained
            }
        self._failed_candidates.clear()
        if self.state_sidecar:
            _publish_native(self.marker_tensor, self._snapshot)
        self._build_table = None

    def forward(
        self, feature_ids: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        if not self.state_sidecar:
            fused_keys, found, _slots = torch.ops.INFERENCE_EMB.table_lookup(
                self.table_storage_,
                self.table_bucket_offsets_,
                self.bucket_capacity,
                feature_ids,
                table_ids,
                None,
                int(ScorePolicy.CONST),
                None,
                0,
                None,
            )
            misses = torch.index_select(
                self.miss_storage_indices_, 0, table_ids
            )
            return torch.where(found, fused_keys, misses)
        return torch.ops.INFERENCE_EMB.embedding_collection_index(
            self.marker_tensor, self.marker_value_, feature_ids, table_ids
        )


class FusedIdentityIndexer(EmbeddingCollectionIndexerBase):
    """Keep table-local IDs and place each table in a reserved fused section."""

    def __init__(
        self,
        table_capacities: Sequence[int],
        *,
        state_sidecar: bool = True,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.state_sidecar = state_sidecar
        reserved_sizes = [int(value) for value in table_capacities]
        valid_bases = []
        next_row = 0
        for size in reserved_sizes:
            valid_bases.append(next_row + 1)
            next_row += size + 1
        self._nve_num_embeddings = next_row
        self.marker_value_ = _new_marker_id()
        self.register_buffer(
            "marker_tensor",
            torch.tensor(
                [self.marker_value_], dtype=torch.int64, device=device
            ),
            persistent=False,
        )
        empty = _empty_i64(device)
        self._snapshot = IndexerSnapshot(
            kind=IndexerSnapshotKind.FUSED_IDENTITY,
            table_storage=torch.empty(0, dtype=torch.uint8, device=device),
            table_bucket_offsets=empty,
            bucket_capacity=0,
            miss_storage_indices=empty,
            valid_bases=torch.tensor(
                valid_bases, dtype=torch.int64, device=device
            ),
            reserved_sizes=torch.tensor(
                reserved_sizes, dtype=torch.int64, device=device
            ),
            next_fused_key=next_row,
        )
        if self.state_sidecar:
            _publish_native(self.marker_tensor, self._snapshot)
        else:
            self.register_buffer("valid_bases_", self._snapshot.valid_bases)
            self.register_buffer("reserved_sizes_", self._snapshot.reserved_sizes)

    @property
    def nve_num_embeddings(self) -> int:
        return self._nve_num_embeddings

    @property
    def snapshot(self) -> Optional[IndexerSnapshot]:
        return self._snapshot if self.state_sidecar else None

    @property
    def miss_storage_indices(self) -> torch.Tensor:
        return self._snapshot.valid_bases - 1

    @torch.no_grad()
    def build_index(
        self, feature_ids: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.index_select(self._snapshot.valid_bases, 0, table_ids)
            + feature_ids
        )

    def forward(
        self, feature_ids: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        if not self.state_sidecar:
            bases = torch.index_select(self.valid_bases_, 0, table_ids)
            sizes = torch.index_select(self.reserved_sizes_, 0, table_ids)
            return torch.where(
                (feature_ids >= 0) & (feature_ids < sizes),
                bases + feature_ids,
                bases - 1,
            )
        return torch.ops.INFERENCE_EMB.embedding_collection_index(
            self.marker_tensor, self.marker_value_, feature_ids, table_ids
        )


class BitConcatIndexer(EmbeddingCollectionIndexerBase):
    def __init__(
        self,
        *,
        feature_id_bits: int,
        nve_num_embeddings: int,
    ) -> None:
        super().__init__()
        self.feature_id_bits = feature_id_bits
        self._nve_num_embeddings = nve_num_embeddings

    @property
    def nve_num_embeddings(self) -> int:
        return self._nve_num_embeddings

    @torch.no_grad()
    def build_index(
        self, feature_ids: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(feature_ids, table_ids)

    def forward(
        self, feature_ids: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        return (table_ids << self.feature_id_bits) | feature_ids


class IdentityIndexer(EmbeddingCollectionIndexerBase):
    def __init__(self, *, nve_num_embeddings: int) -> None:
        super().__init__()
        self._nve_num_embeddings = nve_num_embeddings

    @property
    def nve_num_embeddings(self) -> int:
        return self._nve_num_embeddings

    @torch.no_grad()
    def build_index(
        self, feature_ids: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        return feature_ids

    def forward(
        self, feature_ids: torch.Tensor, table_ids: torch.Tensor
    ) -> torch.Tensor:
        return feature_ids


def create_embedding_collection_indexer(
    config: InferenceEmbeddingCollectionConfig,
    table_capacities: Sequence[int],
    *,
    key_type: torch.dtype,
    device: torch.device,
) -> EmbeddingCollectionIndexerBase:
    capacities = tuple(int(value) for value in table_capacities)
    if config.indexer_type is EmbeddingCollectionIndexerType.LINEAR_HASH_MAP:
        return LinearHashMapIndexer(
            capacities,
            bucket_capacity=config.bucket_capacity,
            key_type=key_type,
            state_sidecar=config.indexer_state_sidecar,
            device=device,
        )
    if config.indexer_type is EmbeddingCollectionIndexerType.FUSED_IDENTITY:
        return FusedIdentityIndexer(
            capacities,
            state_sidecar=config.indexer_state_sidecar,
            device=device,
        )
    if config.indexer_type is EmbeddingCollectionIndexerType.BIT_CONCAT:
        if config.bit_concat is None:
            raise ValueError("BitConcatIndexer requires bit_concat")
        return BitConcatIndexer(
            feature_id_bits=config.bit_concat.feature_id_bits,
            nve_num_embeddings=sum(capacities),
        )
    if config.indexer_type is EmbeddingCollectionIndexerType.IDENTITY:
        return IdentityIndexer(nve_num_embeddings=sum(capacities))
    raise ValueError(f"Unsupported indexer: {config.indexer_type}")
