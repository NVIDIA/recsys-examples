# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

from . import _C
from .config import EmbeddingCollectionIndexerType
from .indexer import (
    BitConcatIndexer,
    EmbeddingCollectionIndexerBase,
    FusedIdentityIndexer,
    IdentityIndexer,
    LinearHashMapIndexer,
)
from .indexer_snapshot import (
    IndexerSnapshot,
    dump_indexer_snapshot,
    load_embedding_collection_indexer_snapshot,
    native_indexer_snapshot,
)


_SIDECAR_DIR = "embedding_collection_indexers"
_MANIFEST_NAME = "manifest.json"
_DIRECTORY_SCHEMA_VERSION = 1


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


def _binding(
    *,
    table_names: list[str],
    indexer_type: str,
    indexer_module_path: str,
    nve_layer_module_path: str,
    snapshot_path: Optional[str] = None,
    feature_id_bits: Optional[int] = None,
    marker_value: Optional[int] = None,
) -> Any:
    result = _C.EmbeddingCollectionBinding()
    result.table_names = table_names
    result.indexer_type = indexer_type
    result.indexer_module_path = indexer_module_path
    result.nve_layer_module_path = nve_layer_module_path
    result.snapshot_path = snapshot_path
    result.feature_id_bits = feature_id_bits
    result.marker_value = marker_value
    return result


def embedding_collection_indexers_from_model(
    model: torch.nn.Module,
) -> Any:
    bindings = {}
    markers = {}
    snapshots = {}
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
        indexer_path = f"{module_path}.indexer_" if module_path else "indexer_"
        nve_path = (
            f"{module_path}.nve_embedding_" if module_path else "nve_embedding_"
        )
        snapshot = indexer.snapshot
        bindings[collection_id] = _binding(
            table_names=list(module.table_names_),
            indexer_type=_indexer_type(indexer),
            indexer_module_path=indexer_path,
            nve_layer_module_path=nve_path,
            feature_id_bits=getattr(indexer, "feature_id_bits", None),
            marker_value=(
                int(indexer.marker_tensor.item()) if snapshot is not None else None
            ),
        )
        if snapshot is not None:
            markers[collection_id] = indexer.marker_tensor
            snapshots[collection_id] = native_indexer_snapshot(snapshot)
            device = indexer.marker_tensor.device
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return _C.EmbeddingCollectionIndexerDirectory.create(
        bindings, markers, snapshots, device_index
    )


def dump_embedding_collection_indexers(
    directory: Any,
    package_dir: str | Path,
) -> Path:
    root = Path(package_dir).resolve() / _SIDECAR_DIR
    root.mkdir(parents=True, exist_ok=True)
    bindings: dict[str, Any] = {}
    snapshot_number = 0
    for collection_id, binding in directory.bindings.items():
        snapshot = directory.snapshot(collection_id)
        if snapshot is None:
            bindings[collection_id] = binding
            continue
        relative_path = f"snapshot_{snapshot_number}.bin"
        dump_indexer_snapshot(snapshot, root / relative_path)
        bindings[collection_id] = _binding(
            table_names=binding.table_names,
            indexer_type=binding.indexer_type,
            indexer_module_path=binding.indexer_module_path,
            nve_layer_module_path=binding.nve_layer_module_path,
            snapshot_path=relative_path,
            feature_id_bits=binding.feature_id_bits,
            marker_value=binding.marker_value,
        )
        snapshot_number += 1

    document = {
        "schema_version": _DIRECTORY_SCHEMA_VERSION,
        "collections": {
            collection_id: {
                "table_names": binding.table_names,
                "indexer_type": binding.indexer_type,
                "indexer_module_path": binding.indexer_module_path,
                "nve_layer_module_path": binding.nve_layer_module_path,
                "snapshot_path": binding.snapshot_path,
                "feature_id_bits": binding.feature_id_bits,
                "marker_value": binding.marker_value,
            }
            for collection_id, binding in bindings.items()
        },
    }
    manifest_path = root / _MANIFEST_NAME
    manifest_path.write_text(json.dumps(document, indent=2) + "\n")
    return manifest_path


def load_embedding_collection_indexer_state(
    package_dir: str | Path, device: torch.device
) -> tuple[dict[str, Any], dict[str, IndexerSnapshot]]:
    root = Path(package_dir).resolve() / _SIDECAR_DIR
    document = json.loads((root / _MANIFEST_NAME).read_text())
    bindings = {
        collection_id: _binding(**value)
        for collection_id, value in document["collections"].items()
    }
    snapshots = {
        collection_id: load_embedding_collection_indexer_snapshot(
            root / binding.snapshot_path, device
        )
        for collection_id, binding in bindings.items()
        if binding.snapshot_path is not None
    }
    return bindings, snapshots


def load_embedding_collection_indexers(
    package_dir: str | Path, device: torch.device
) -> Any:
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return _C.EmbeddingCollectionIndexerDirectory.load(
        str(Path(package_dir).resolve()), device_index
    )
