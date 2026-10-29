# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

import torch
if TYPE_CHECKING:
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


def _prepare_delta_keys(
    delta: DeltaDumpResult,
    dense_capacities: Optional[Mapping[str, int]] = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Validate dense IDs and resolve removals before changing PS or indexer state."""
    columns = len(delta.table_names)
    evicted = delta.evicted_keys or [None] * columns
    erased = getattr(delta, "erased_keys", None) or [None] * columns
    if any(
        len(items) != columns for items in (delta.keys, delta.values, evicted, erased)
    ):
        raise ValueError("Delta columns must align with table names")
    result = []
    for name, keys, evictions, erases in zip(
        delta.table_names, delta.keys, evicted, erased
    ):
        keys = keys.to(dtype=torch.int64, device="cpu")
        feature_ids = keys.tolist()
        removed = set()
        for tensor in (evictions, erases):
            if tensor is not None:
                removed.update(tensor.to(dtype=torch.int64, device="cpu").tolist())
        if dense_capacities is not None:
            if name not in dense_capacities:
                raise ValueError(f"Unknown table: {name}")
            capacity = dense_capacities[name]
            if any(key < 0 or key >= capacity for key in chain(feature_ids, removed)):
                raise ValueError(f"Feature ID outside the reserved range for {name}")
        # The final dumped value wins over an earlier removal in the same window.
        removed.difference_update(feature_ids)
        result.append((keys, torch.tensor(sorted(removed), dtype=torch.int64)))
    return result


class EmbeddingCollectionUpdateCoordinator:
    """Prepare Redis or CPU HostLayer deltas and indexer publications."""

    def __init__(
        self,
        *,
        bindings: Mapping[str, Any],
        snapshots: Mapping[str, IndexerSnapshot],
        device: torch.device,
        parameter_servers: Mapping[str, Any],
        shared_update_dir: Path,
        subscriber_ids: Iterable[str],
        host_layers: Optional[Mapping[str, dict[str, Any]]] = None,
    ) -> None:
        self.bindings = dict(bindings)
        self.snapshots = dict(snapshots)
        self.device = device
        self.parameter_servers = dict(parameter_servers)
        self.host_layers = dict(host_layers or {})
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
        metadata = json.loads((package_dir / "metadata.json").read_text())
        layers = metadata if isinstance(metadata, list) else metadata["layers"]
        hosts = {layer["module_path"]: layer for layer in layers
                 if layer.get("layer_type") == "HostLayer"}
        return cls(
            host_layers={name: hosts[binding.nve_layer_module_path]
                         for name, binding in bindings.items()
                         if binding.nve_layer_module_path in hosts},
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
        table_keys: list[tuple[torch.Tensor, torch.Tensor]],
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
            keys, removed = table_keys[column]
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

            for feature_id in removed.tolist():
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
        table_keys: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[list[int], list[torch.Tensor]]:
        binding = self.bindings[collection_id]
        table_ids = {name: index for index, name in enumerate(binding.table_names)}
        snapshot = self.snapshots.get(collection_id)
        cache_update_keys: list[int] = []
        cache_update_values: list[torch.Tensor] = []
        for column, table_name in enumerate(delta.table_names):
            table_id = table_ids[table_name]
            feature_ids, removed = table_keys[column]
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

            if removed.numel() == 0:
                continue
            if binding.indexer_type == EmbeddingCollectionIndexerType.BIT_CONCAT.value:
                feature_bits = binding.feature_id_bits
                assert feature_bits is not None
                removed_storage = (table_id << feature_bits) | removed
            elif (
                binding.indexer_type
                == EmbeddingCollectionIndexerType.FUSED_IDENTITY.value
            ):
                assert snapshot is not None
                base = int(snapshot.valid_bases[table_id].item())
                removed_storage = base + removed
            else:
                removed_storage = removed
            parameter_server.erase(removed_storage.contiguous())
            # Direct mappings still resolve deleted keys. Reuse the existing
            # subscriber payload to replace stale values in every cache tier.
            cache_update_keys.extend(removed_storage.tolist())
            cache_update_values.append(
                torch.zeros(
                    (removed_storage.numel(), delta.values[column].shape[1]),
                    dtype=torch.float32,
                )
            )
        return cache_update_keys, cache_update_values

    def _host_update(
        self, collection_id: str, delta: DeltaDumpResult
    ) -> EmbeddingCollectionUpdate:
        if self.device.type != "cpu":
            raise ValueError("HostLayer incremental updates require CPU execution")
        binding = self.bindings[collection_id]
        config = self.host_layers[collection_id]
        capacities = list(binding.table_capacities)
        if len(capacities) != len(binding.table_names) or any(c < 0 for c in capacities):
            raise ValueError("Host updates require exported per-table capacities; re-export the package")
        kind = EmbeddingCollectionIndexerType(binding.indexer_type)
        linear = kind == EmbeddingCollectionIndexerType.LINEAR_HASH_MAP
        if kind not in (EmbeddingCollectionIndexerType.IDENTITY,
                        EmbeddingCollectionIndexerType.FUSED_IDENTITY,
                        EmbeddingCollectionIndexerType.LINEAR_HASH_MAP):
            raise ValueError("Unsupported HostLayer update indexer")
        if kind == EmbeddingCollectionIndexerType.IDENTITY and len(capacities) != 1:
            raise ValueError("Identity requires a single table")
        dim = int(config["embedding_size"])
        num_rows = int(config["num_embeddings"])
        expected_rows = sum(capacities) + (0 if kind == EmbeddingCollectionIndexerType.IDENTITY else len(capacities))
        if num_rows != expected_rows:
            raise ValueError("Host storage size does not match the indexer capacities")
        names = list(delta.table_names)
        if len(set(names)) != len(names):
            raise ValueError("Delta table names must be unique")
        erased = getattr(delta, "erased_keys", None) or [None] * len(names)
        if any(len(items) != len(names) for items in
               (delta.keys, delta.values, delta.evicted_keys, erased)):
            raise ValueError("Delta columns must align with table names")
        table_ids = {name: i for i, name in enumerate(binding.table_names)}
        # Validate the entire delta before advancing shadow state or writing a snapshot.
        changes = {}
        for column, name in enumerate(names):
            if name not in table_ids:
                raise ValueError(f"Unknown table: {name}")
            table = table_ids[name]
            keys, values = delta.keys[column], delta.values[column]
            removed = []
            for tensor in (keys, delta.evicted_keys[column], erased[column]):
                if tensor is None:
                    continue
                if tensor.device.type != "cpu" or tensor.dtype != torch.int64 or tensor.ndim != 1:
                    raise ValueError("Host delta keys must be one-dimensional CPU int64 tensors")
                ids = tensor.tolist()
                if linear:
                    if any(-4 <= key <= -1 for key in ids):
                        raise ValueError("Delta key is reserved by the hash indexer")
                elif any(key < 0 or key >= capacities[table] for key in ids):
                    raise ValueError(f"Feature ID outside the reserved range for {name}")
            if (values.device.type != "cpu" or values.ndim != 2
                    or tuple(values.shape) != (keys.numel(), dim)
                    or values.dtype not in (torch.float16, torch.float32)):
                raise ValueError("Host delta values must be CPU float32/float16 [num_keys, embedding_dim]")
            for tensor in (delta.evicted_keys[column], erased[column]):
                if tensor is not None:
                    removed.extend(tensor.tolist())
            # A final dumped value wins over an earlier eviction/erase of that key.
            # Repeated upserts use the last value, matching initial Host loading.
            upserts = dict(zip(keys.tolist(), values.to(torch.float32).unbind()))
            changes[table] = (upserts, set(removed) - upserts.keys())

        row_values = {}
        zero = torch.zeros(dim, dtype=torch.float32)
        next_snapshot = None
        if linear:
            old = self.snapshots.get(collection_id)
            if old is None:
                raise ValueError("Host LinearHashMap updates require sidecar indexer state")
            entries = linear_hash_entries(old)
            changed_mapping = False
            for table, (upserts, removed) in changes.items():
                for key in removed:
                    row = entries[table].pop(key, None)
                    if row is not None:
                        row_values[row] = zero
                        changed_mapping = True
                if len(entries[table].keys() | upserts.keys()) > capacities[table]:
                    raise ValueError(f"Delta exceeds capacity of table {binding.table_names[table]}")
            used = {row for mapping in entries for row in mapping.values()}
            free_rows = (row for row in range(len(capacities), num_rows) if row not in used)
            next_row = old.next_fused_key
            for table, (upserts, _) in changes.items():
                for key, value in upserts.items():
                    if key not in entries[table]:
                        row = next(free_rows)
                        entries[table][key] = row
                        next_row = max(next_row, row + 1)
                        changed_mapping = True
                    row_values[entries[table][key]] = value
            if changed_mapping:
                next_snapshot = rebuild_linear_hash_snapshot(
                    entries, bucket_capacity=old.bucket_capacity,
                    miss_storage_indices=old.miss_storage_indices,
                    next_fused_key=next_row, device=torch.device("cpu"),
                )
        else:
            base = 0
            for table, capacity in enumerate(capacities):
                if kind == EmbeddingCollectionIndexerType.FUSED_IDENTITY:
                    base += 1  # Preserve the zero miss row before each table.
                upserts, removed = changes.get(table, ({}, set()))
                row_values.update({base + key: zero for key in removed})
                row_values.update({base + key: value for key, value in upserts.items()})
                base += capacity

        snapshot_id = self._snapshot_ids[collection_id] + 1
        snapshot_path = None
        if next_snapshot is not None:
            snapshot_path = dump_embedding_collection_indexer_snapshot(
                collection_id=collection_id.replace(".", "_"),
                snapshot_id=snapshot_id, snapshot=next_snapshot,
                output_dir=self.shared_update_dir,
            )
        update = EmbeddingCollectionUpdate(
            collection_id, snapshot_id,
            torch.tensor(list(row_values), dtype=torch.int64),
            torch.stack(list(row_values.values())).flatten() if row_values
            else torch.empty(0, dtype=torch.float32),
            snapshot_path,
        )
        if next_snapshot is not None:
            self.snapshots[collection_id] = next_snapshot
        self._snapshot_ids[collection_id] = snapshot_id
        return update

    def apply_delta(
        self, collection_id: str, delta: DeltaDumpResult
    ) -> EmbeddingCollectionUpdate:
        if collection_id in self.host_layers:
            return self._host_update(collection_id, delta)
        binding = self.bindings[collection_id]
        indexer_type = binding.indexer_type
        dense_capacities = None
        if indexer_type in (
            EmbeddingCollectionIndexerType.IDENTITY.value,
            EmbeddingCollectionIndexerType.FUSED_IDENTITY.value,
        ):
            capacities = binding.table_capacities
            if len(capacities) != len(binding.table_names) or any(
                c < 0 for c in capacities
            ):
                raise ValueError(
                    "Dense updates require exported per-table capacities; re-export the package"
                )
            dense_capacities = dict(zip(binding.table_names, capacities))
        table_keys = _prepare_delta_keys(delta, dense_capacities)
        snapshot_id = self._next_snapshot_id(collection_id)
        parameter_server = self.parameter_servers[collection_id]
        if indexer_type == EmbeddingCollectionIndexerType.LINEAR_HASH_MAP.value:
            cache_update_keys, cache_update_values, snapshot_path, retired = (
                self._linear_update(
                    collection_id, delta, parameter_server, snapshot_id, table_keys
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
                collection_id, delta, parameter_server, table_keys
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
