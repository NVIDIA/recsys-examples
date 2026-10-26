# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Register fake kernels only after the native schemas have been loaded.
# isort: off
from . import _C
from dynamicemb import lookup_meta as _lookup_meta  # noqa: F401

# isort: on
from .config import (
    BitConcatConfig,
    EmbeddingCollectionIndexerType,
    InferenceEmbeddingCollectionConfig,
)

from .indexer_directory import (
    dump_embedding_collection_indexers,
    embedding_collection_indexers_from_model,
    load_embedding_collection_indexers,
)
from .indexer import (
    BitConcatIndexer,
    EmbeddingCollectionIndexerBase,
    FusedIdentityIndexer,
    IdentityIndexer,
    LinearHashMapIndexer,
)
from .indexer_snapshot import dump_embedding_collection_indexer_snapshot
from .nve_runtime import (
    export_embedding_collection_aot,
    imported_nve_generation,
    load_embedding_collection_aot,
    register_nve_export_compat,
)

EmbeddingCollectionBinding = _C.EmbeddingCollectionBinding
EmbeddingCollectionIndexerDirectory = _C.EmbeddingCollectionIndexerDirectory
EmbeddingCollectionUpdateSubscriber = _C.EmbeddingCollectionUpdateSubscriber


__all__ = [
    "BitConcatConfig",
    "BitConcatIndexer",
    "EmbeddingCollectionBinding",
    "EmbeddingCollectionIndexerBase",
    "EmbeddingCollectionIndexerDirectory",
    "EmbeddingCollectionIndexerType",
    "EmbeddingCollectionUpdate",
    "EmbeddingCollectionUpdateAck",
    "EmbeddingCollectionUpdateCoordinator",
    "EmbeddingCollectionUpdateSubscriber",
    "FusedIdentityIndexer",
    "IdentityIndexer",
    "InferenceEmbeddingCollectionConfig",
    "LinearHashMapIndexer",
    "dump_embedding_collection_indexer_snapshot",
    "dump_embedding_collection_indexers",
    "embedding_collection_indexers_from_model",
    "export_embedding_collection_aot",
    "imported_nve_generation",
    "load_embedding_collection_aot",
    "load_embedding_collection_indexers",
    "register_nve_export_compat",
]


def __getattr__(name):
    if name in {
        "EmbeddingCollectionUpdate",
        "EmbeddingCollectionUpdateAck",
        "EmbeddingCollectionUpdateCoordinator",
    }:
        from importlib import import_module

        value = getattr(import_module(".incremental_update", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
