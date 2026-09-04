# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
import threading
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .config import EmbeddingCollectionIndexerType
from .indexers import (
    BitConcatIndexer,
    EmbeddingCollectionIndexerBase,
    FusedIdentityIndexer,
    IdentityIndexer,
    IndexerSnapshot,
    LinearHashMapIndexer,
    _publish_native,
)


_SIDECAR_DIR = "embedding_collection_indexers"
_MANIFEST_NAME = "manifest.json"
_SNAPSHOT_MAGIC = b"ECIDX001"
_SNAPSHOT_HEADER = struct.Struct("<8sIIqqQQQQQ")
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmbeddingCollectionBinding:
    table_names: list[str]
    indexer_type: str
    indexer_module_path: str
    nve_layer_module_path: str
    snapshot_path: Optional[str] = None
    feature_id_bits: Optional[int] = None
    marker_value: Optional[int] = None


@dataclass
class _DirectoryEntry:
    marker: torch.Tensor
    snapshot: IndexerSnapshot
    module: Optional[EmbeddingCollectionIndexerBase] = None


def _indexer_type(indexer: EmbeddingCollectionIndexerBase) -> str:
    if isinstance(indexer, LinearHashMapIndexer):
        return EmbeddingCollectionIndexerType.LINEAR_HASH_MAP.value
    if isinstance(indexer, FusedIdentityIndexer):
        return EmbeddingCollectionIndexerType.FUSED_IDENTITY.value
    if isinstance(indexer, BitConcatIndexer):
        return EmbeddingCollectionIndexerType.BIT_CONCAT.value
    if isinstance(indexer, IdentityIndexer):
        return EmbeddingCollectionIndexerType.IDENTITY.value
    raise TypeError(f"Unsupported indexer: {type(indexer).__name__}")


def _write_tensor(stream: Any, tensor: torch.Tensor) -> None:
    stream.write(tensor.detach().cpu().contiguous().numpy().tobytes())


def _write_snapshot(snapshot: IndexerSnapshot, path: Path) -> None:
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
                _SCHEMA_VERSION,
                snapshot.kind,
                snapshot.bucket_capacity,
                snapshot.next_fused_key,
                *lengths,
            )
        )
        for tensor in tensors:
            _write_tensor(stream, tensor)


def _read_values(
    stream: Any, count: int, dtype: torch.dtype, item_size: int
) -> torch.Tensor:
    data = bytearray(stream.read(count * item_size))
    if not data:
        return torch.empty(0, dtype=dtype)
    return torch.frombuffer(data, dtype=dtype).clone()


def load_embedding_collection_indexer_snapshot(
    path: str | Path, device: torch.device
) -> IndexerSnapshot:
    with Path(path).open("rb") as stream:
        (
            magic,
            version,
            kind,
            bucket_capacity,
            next_fused_key,
            storage_size,
            offsets_size,
            miss_size,
            bases_size,
            reserved_size,
        ) = _SNAPSHOT_HEADER.unpack(stream.read(_SNAPSHOT_HEADER.size))
        if magic != _SNAPSHOT_MAGIC or version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported indexer snapshot: {path}")
        snapshot = IndexerSnapshot(
            kind=kind,
            table_storage=_read_values(stream, storage_size, torch.uint8, 1),
            table_bucket_offsets=_read_values(
                stream, offsets_size, torch.int64, 8
            ),
            bucket_capacity=bucket_capacity,
            miss_storage_indices=_read_values(
                stream, miss_size, torch.int64, 8
            ),
            valid_bases=_read_values(stream, bases_size, torch.int64, 8),
            reserved_sizes=_read_values(
                stream, reserved_size, torch.int64, 8
            ),
            next_fused_key=next_fused_key,
        )
    return snapshot.to(device)


def dump_embedding_collection_indexer_snapshot(
    *,
    collection_id: str,
    snapshot_id: int,
    snapshot: IndexerSnapshot,
    output_dir: str | Path,
) -> str:
    path = Path(output_dir).resolve() / f"{collection_id}.{snapshot_id}.bin"
    _write_snapshot(snapshot, path)
    return str(path)


class _InferenceCall(AbstractContextManager["_InferenceCall"]):
    def __init__(self, directory: "EmbeddingCollectionIndexerDirectory") -> None:
        self._directory = directory

    def __enter__(self) -> "_InferenceCall":
        with self._directory._condition:
            self._directory._active_enqueues += 1
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        event = None
        if exc_type is None and torch.cuda.is_available():
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self._directory.device))
        with self._directory._condition:
            if event is not None:
                self._directory._current_events.append(event)
            self._directory._active_enqueues -= 1
            self._directory._condition.notify_all()


class EmbeddingCollectionIndexerDirectory:
    """Own data-backed indexer bindings for eager or exported inference."""

    def __init__(
        self,
        *,
        bindings: Mapping[str, EmbeddingCollectionBinding],
        entries: Mapping[str, _DirectoryEntry],
        device: torch.device,
        owns_native_registrations: bool,
    ) -> None:
        self.bindings = dict(bindings)
        self._entries = dict(entries)
        self.device = device
        self._owns_native_registrations = owns_native_registrations
        self._condition = threading.Condition()
        self._active_enqueues = 0
        self._current_events: list[torch.cuda.Event] = []
        self._retired: dict[
            tuple[str, int], tuple[Optional[IndexerSnapshot], list[torch.cuda.Event]]
        ] = {}

    @classmethod
    def from_model(
        cls, model: torch.nn.Module
    ) -> "EmbeddingCollectionIndexerDirectory":
        bindings: dict[str, EmbeddingCollectionBinding] = {}
        entries: dict[str, _DirectoryEntry] = {}
        device = torch.device("cuda", torch.cuda.current_device())
        collection_number = 0
        for module_path, module in model.named_modules():
            indexer = getattr(module, "indexer_", None)
            if not isinstance(indexer, EmbeddingCollectionIndexerBase):
                continue
            collection_id = getattr(module, "collection_id_", None)
            if not collection_id:
                collection_id = module_path or f"collection_{collection_number}"
            collection_number += 1
            indexer_path = (
                f"{module_path}.indexer_" if module_path else "indexer_"
            )
            nve_path = (
                f"{module_path}.nve_embedding_"
                if module_path
                else "nve_embedding_"
            )
            bindings[collection_id] = EmbeddingCollectionBinding(
                table_names=list(module.table_names_),
                indexer_type=_indexer_type(indexer),
                indexer_module_path=indexer_path,
                nve_layer_module_path=nve_path,
                feature_id_bits=getattr(indexer, "feature_id_bits", None),
                marker_value=(
                    int(indexer.marker_tensor.item())
                    if indexer.snapshot is not None
                    else None
                ),
            )
            if indexer.snapshot is not None:
                entries[collection_id] = _DirectoryEntry(
                    marker=indexer.marker_tensor,
                    snapshot=indexer.snapshot,
                    module=indexer,
                )
                device = indexer.marker_tensor.device
        return cls(
            bindings=bindings,
            entries=entries,
            device=device,
            owns_native_registrations=False,
        )

    def inference_call(self) -> _InferenceCall:
        return _InferenceCall(self)

    def bind_aoti(self, loader: Any) -> None:
        constant_names = set(loader.get_constant_fqns())
        for collection_id, entry in self._entries.items():
            fqn = self.bindings[collection_id].indexer_module_path + ".marker_tensor"
            if fqn not in constant_names:
                raise RuntimeError(f"Indexer marker is not an AOTI constant: {fqn}")
            for use_inactive in (False, True):
                loader.load_constants(
                    {fqn: entry.marker}, use_inactive, False, True
                )

    def bind_exported_module(self, module: torch.nn.Module) -> None:
        for collection_id, entry in self._entries.items():
            module_path = self.bindings[collection_id].indexer_module_path
            module.get_submodule(module_path).marker_tensor = entry.marker

    def publish(
        self,
        collection_id: str,
        snapshot: Optional[IndexerSnapshot],
        snapshot_id: int,
    ) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._active_enqueues == 0)
            entry = self._entries.get(collection_id)
            old_snapshot = entry.snapshot if entry is not None else None
            if snapshot is not None:
                if entry is None:
                    raise KeyError(collection_id)
                _publish_native(entry.marker, snapshot)
                entry.snapshot = snapshot
                if entry.module is not None:
                    entry.module._snapshot = snapshot
            events, self._current_events = self._current_events, []
            self._retired[(collection_id, snapshot_id)] = (
                old_snapshot if snapshot is not None else None,
                events,
            )

    def wait_for_retirement(self, collection_id: str, snapshot_id: int) -> None:
        _snapshot, events = self._retired.pop((collection_id, snapshot_id), (None, []))
        for event in events:
            event.synchronize()

    def close(self) -> None:
        if not self._owns_native_registrations:
            return
        for entry in self._entries.values():
            torch.ops.INFERENCE_EMB.unregister_embedding_collection_indexer(
                entry.marker
            )
        self._owns_native_registrations = False


def dump_embedding_collection_indexers(
    directory: EmbeddingCollectionIndexerDirectory,
    package_dir: str | Path,
) -> Path:
    root = Path(package_dir).resolve() / _SIDECAR_DIR
    root.mkdir(parents=True, exist_ok=True)
    bindings: dict[str, EmbeddingCollectionBinding] = {}
    snapshot_number = 0
    for collection_id, binding in directory.bindings.items():
        entry = directory._entries.get(collection_id)
        if entry is None:
            bindings[collection_id] = binding
            continue
        relative_path = f"snapshot_{snapshot_number}.bin"
        _write_snapshot(entry.snapshot, root / relative_path)
        bindings[collection_id] = replace(
            binding, snapshot_path=relative_path
        )
        snapshot_number += 1

    document = {
        "schema_version": _SCHEMA_VERSION,
        "collections": {
            collection_id: asdict(binding)
            for collection_id, binding in bindings.items()
        },
    }
    manifest_path = root / _MANIFEST_NAME
    manifest_path.write_text(json.dumps(document, indent=2) + "\n")
    return manifest_path


def load_embedding_collection_indexers(
    package_dir: str | Path, device: torch.device
) -> EmbeddingCollectionIndexerDirectory:
    root = Path(package_dir).resolve() / _SIDECAR_DIR
    document = json.loads((root / _MANIFEST_NAME).read_text())
    bindings = {
        collection_id: EmbeddingCollectionBinding(**binding)
        for collection_id, binding in document["collections"].items()
    }
    entries: dict[str, _DirectoryEntry] = {}
    for collection_id, binding in bindings.items():
        if binding.snapshot_path is None:
            continue
        snapshot = load_embedding_collection_indexer_snapshot(
            root / binding.snapshot_path, device
        )
        marker = torch.tensor(
            [binding.marker_value], dtype=torch.int64, device=device
        )
        _publish_native(marker, snapshot)
        entries[collection_id] = _DirectoryEntry(marker, snapshot)
    return EmbeddingCollectionIndexerDirectory(
        bindings=bindings,
        entries=entries,
        device=device,
        owns_native_registrations=True,
    )
