# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
from dynamicemb.incremental_dump import DeltaDumpResult

from .config import EmbeddingCollectionIndexerType
from .indexer_directory import (
    load_embedding_collection_indexer_state,
)
from .indexer_snapshot import (
    IndexerSnapshot,
    dump_embedding_collection_indexer_snapshot,
    linear_hash_entries,
    rebuild_linear_hash_snapshot,
)
from .nve_runtime import imported_nve_generation


@dataclass(frozen=True)
class EmbeddingCollectionUpdate:
    collection_id: str
    snapshot_id: int
    cache_update_keys: torch.Tensor
    cache_update_values: torch.Tensor
    indexer_snapshot_path: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "collection_id": self.collection_id,
                "snapshot_id": self.snapshot_id,
                "cache_update_keys": self.cache_update_keys.tolist(),
                "cache_update_values": self.cache_update_values.to(
                    dtype=torch.float32
                ).flatten().tolist(),
                "indexer_snapshot_path": self.indexer_snapshot_path,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "EmbeddingCollectionUpdate":
        value = json.loads(payload)
        return cls(
            collection_id=value["collection_id"],
            snapshot_id=value["snapshot_id"],
            cache_update_keys=torch.tensor(
                value["cache_update_keys"], dtype=torch.int64
            ),
            cache_update_values=torch.tensor(
                value["cache_update_values"], dtype=torch.float32
            ),
            indexer_snapshot_path=value.get("indexer_snapshot_path"),
        )


@dataclass(frozen=True)
class EmbeddingCollectionUpdateAck:
    collection_id: str
    snapshot_id: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "collection_id": self.collection_id,
                "snapshot_id": self.snapshot_id,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "EmbeddingCollectionUpdateAck":
        return cls(**json.loads(payload))


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if "float16" in name.lower() else torch.float32


def _make_parameter_server(config: dict[str, Any]) -> Any:
    from pynve.torch.nve_ps import NVEParameterServer

    return NVEParameterServer(
        num_embeddings=config.get("num_rows", 0),
        embedding_size=config["row_elements"],
        data_type=_torch_dtype(config["data_type"]),
        plugin_name=config["plugin_name"],
        factory_config=config.get("factory_config", {}),
        table_config=config.get("table_config", {}),
    )


def _open_parameter_servers(
    package_dir: Path,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = json.loads((package_dir / "metadata.json").read_text())
    by_path: dict[str, dict[str, Any]] = {}
    if isinstance(metadata, list):
        for layer in metadata:
            if "remote_ps_config" in layer:
                by_path[layer["module_path"]] = layer["remote_ps_config"]
    else:
        resources = metadata.get("resources", {}).get("remote_ps", {})
        for layer in metadata["layers"]:
            storage_ref = layer.get("storage_ref")
            if storage_ref in resources:
                by_path[layer["module_path"]] = resources[storage_ref]

    result = {}
    for collection_id, binding in bindings.items():
        config = by_path.get(binding.nve_layer_module_path)
        if config is not None and "redis" in config["plugin_name"].lower():
            result[collection_id] = _make_parameter_server(config)
    return result


class EmbeddingCollectionUpdateCoordinator:
    """Serialize DynamicEmb deltas into Redis and indexer publications."""

    def __init__(
        self,
        *,
        bindings: Mapping[str, Any],
        snapshots: Mapping[str, IndexerSnapshot],
        device: torch.device,
        parameter_servers: Mapping[str, Any],
        shared_update_dir: Path,
        subscriber_ids: Iterable[str],
    ) -> None:
        self.bindings = dict(bindings)
        self.snapshots = dict(snapshots)
        self.device = device
        self.parameter_servers = dict(parameter_servers)
        self.shared_update_dir = shared_update_dir
        self.subscriber_ids = set(subscriber_ids)
        self._snapshot_ids = {
            collection_id: 0 for collection_id in self.bindings
        }
        self._pending_deletes: dict[
            tuple[str, int], tuple[Any, torch.Tensor, set[str]]
        ] = {}

    @classmethod
    def open(
        cls,
        *,
        package_dir: str | Path,
        shared_update_dir: str | Path,
        device: torch.device,
        subscriber_ids: Iterable[str] = (),
    ) -> "EmbeddingCollectionUpdateCoordinator":
        if imported_nve_generation() < (26, 6):
            raise RuntimeError("Redis incremental load requires NVE 26.06 or later")
        package_dir = Path(package_dir).resolve()
        bindings, snapshots = load_embedding_collection_indexer_state(
            package_dir, device
        )
        update_dir = Path(shared_update_dir).resolve()
        update_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            bindings=bindings,
            snapshots=snapshots,
            device=device,
            parameter_servers=_open_parameter_servers(package_dir, bindings),
            shared_update_dir=update_dir,
            subscriber_ids=subscriber_ids,
        )

    def _next_snapshot_id(self, collection_id: str) -> int:
        self._snapshot_ids[collection_id] += 1
        return self._snapshot_ids[collection_id]

    def _linear_update(
        self,
        collection_id: str,
        delta: DeltaDumpResult,
        parameter_server: Any,
        snapshot_id: int,
    ) -> tuple[list[int], list[torch.Tensor], Optional[str], list[int]]:
        old_snapshot = self.snapshots[collection_id]
        entries = linear_hash_entries(old_snapshot)
        binding = self.bindings[collection_id]
        table_ids = {name: index for index, name in enumerate(binding.table_names)}
        cache_update_keys: list[int] = []
        cache_update_values: list[torch.Tensor] = []
        retired: list[int] = []
        changed_mapping = False
        next_fused_key = old_snapshot.next_fused_key

        for column, table_name in enumerate(delta.table_names):
            table_id = table_ids[table_name]
            keys = delta.keys[column].to(dtype=torch.int64, device="cpu")
            values = delta.values[column].to(device="cpu").contiguous()
            storage_keys = []
            for feature_id in keys.tolist():
                storage_key = entries[table_id].get(feature_id)
                if storage_key is None:
                    storage_key = next_fused_key
                    next_fused_key += 1
                    entries[table_id][feature_id] = storage_key
                    changed_mapping = True
                storage_keys.append(storage_key)
            storage_tensor = torch.tensor(storage_keys, dtype=torch.int64)
            if storage_tensor.numel() > 0:
                parameter_server.insert(storage_tensor, values)
                cache_update_keys.extend(storage_keys)
                cache_update_values.append(values.to(dtype=torch.float32))

            evicted = delta.evicted_keys[column]
            if evicted is not None:
                for feature_id in evicted.to(device="cpu").tolist():
                    storage_key = entries[table_id].pop(feature_id, None)
                    if storage_key is not None:
                        retired.append(storage_key)
                        changed_mapping = True

        snapshot_path = None
        if changed_mapping:
            snapshot = rebuild_linear_hash_snapshot(
                entries,
                bucket_capacity=old_snapshot.bucket_capacity,
                miss_storage_indices=old_snapshot.miss_storage_indices,
                next_fused_key=next_fused_key,
                device=self.device,
            )
            snapshot_path = dump_embedding_collection_indexer_snapshot(
                collection_id=collection_id.replace(".", "_"),
                snapshot_id=snapshot_id,
                snapshot=snapshot,
                output_dir=self.shared_update_dir,
            )
            # Advance the coordinator's shadow state so the next delta resolves
            # against snapshot N+1, not snapshot 0.
            self.snapshots[collection_id] = snapshot
        return (
            cache_update_keys,
            cache_update_values,
            snapshot_path,
            retired,
        )

    def _direct_update(
        self,
        collection_id: str,
        delta: DeltaDumpResult,
        parameter_server: Any,
    ) -> tuple[list[int], list[torch.Tensor]]:
        binding = self.bindings[collection_id]
        table_ids = {name: index for index, name in enumerate(binding.table_names)}
        snapshot = self.snapshots.get(collection_id)
        cache_update_keys: list[int] = []
        cache_update_values: list[torch.Tensor] = []
        for column, table_name in enumerate(delta.table_names):
            table_id = table_ids[table_name]
            feature_ids = delta.keys[column].to(dtype=torch.int64, device="cpu")
            if binding.indexer_type == EmbeddingCollectionIndexerType.BIT_CONCAT.value:
                feature_bits = binding.feature_id_bits
                assert feature_bits is not None
                storage_keys = (table_id << feature_bits) | feature_ids
            elif (
                binding.indexer_type
                == EmbeddingCollectionIndexerType.FUSED_IDENTITY.value
            ):
                assert snapshot is not None
                base = int(snapshot.valid_bases[table_id].item())
                reserved = int(snapshot.reserved_sizes[table_id].item())
                if feature_ids.numel() and int(feature_ids.max().item()) >= reserved:
                    raise ValueError(
                        f"feature ID exceeds the reserved range for {table_name}"
                    )
                storage_keys = base + feature_ids
            else:
                storage_keys = feature_ids
            if storage_keys.numel() > 0:
                values = delta.values[column].to(device="cpu").contiguous()
                parameter_server.insert(
                    storage_keys.contiguous(),
                    values,
                )
                cache_update_keys.extend(storage_keys.tolist())
                cache_update_values.append(values.to(dtype=torch.float32))

            evicted = delta.evicted_keys[column]
            if evicted is None:
                continue
            evicted = evicted.to(dtype=torch.int64, device="cpu")
            if binding.indexer_type == EmbeddingCollectionIndexerType.BIT_CONCAT.value:
                feature_bits = binding.feature_id_bits
                assert feature_bits is not None
                evicted_storage = (table_id << feature_bits) | evicted
            elif (
                binding.indexer_type
                == EmbeddingCollectionIndexerType.FUSED_IDENTITY.value
            ):
                assert snapshot is not None
                base = int(snapshot.valid_bases[table_id].item())
                evicted_storage = base + evicted
            else:
                evicted_storage = evicted
            parameter_server.erase(evicted_storage.contiguous())
        return cache_update_keys, cache_update_values

    def apply_delta(
        self, collection_id: str, delta: DeltaDumpResult
    ) -> EmbeddingCollectionUpdate:
        snapshot_id = self._next_snapshot_id(collection_id)
        parameter_server = self.parameter_servers[collection_id]
        indexer_type = self.bindings[collection_id].indexer_type
        if indexer_type == EmbeddingCollectionIndexerType.LINEAR_HASH_MAP.value:
            cache_update_keys, cache_update_values, snapshot_path, retired = (
                self._linear_update(
                    collection_id, delta, parameter_server, snapshot_id
                )
            )
            if retired:
                self._pending_deletes[(collection_id, snapshot_id)] = (
                    parameter_server,
                    torch.tensor(retired, dtype=torch.int64),
                    set(self.subscriber_ids),
                )
        else:
            cache_update_keys, cache_update_values = self._direct_update(
                collection_id, delta, parameter_server
            )
            snapshot_path = None
        if cache_update_values:
            update_values = torch.cat(cache_update_values, dim=0).flatten()
        else:
            update_values = torch.empty(0, dtype=torch.float32)
        return EmbeddingCollectionUpdate(
            collection_id=collection_id,
            snapshot_id=snapshot_id,
            cache_update_keys=torch.tensor(cache_update_keys, dtype=torch.int64),
            cache_update_values=update_values,
            indexer_snapshot_path=snapshot_path,
        )

    def acknowledge(
        self, subscriber_id: str, ack: EmbeddingCollectionUpdateAck
    ) -> None:
        pending = self._pending_deletes.get(
            (ack.collection_id, ack.snapshot_id)
        )
        if pending is None:
            return
        parameter_server, keys, waiting = pending
        waiting.discard(subscriber_id)
        if not waiting:
            parameter_server.erase(keys)
            del self._pending_deletes[(ack.collection_id, ack.snapshot_id)]
