# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class EmbeddingCollectionIndexerType(str, Enum):
    LINEAR_HASH_MAP = "linear_hash_map"
    FUSED_IDENTITY = "fused_identity"
    BIT_CONCAT = "bit_concat"
    IDENTITY = "identity"


@dataclass(frozen=True)
class BitConcatConfig:
    table_id_bits: int
    feature_id_bits: int


@dataclass(frozen=True)
class InferenceEmbeddingCollectionConfig:
    indexer_type: EmbeddingCollectionIndexerType
    nve_layer_type: str
    indexer_state_sidecar: bool = True
    bucket_capacity: int = 128
    bit_concat: Optional[BitConcatConfig] = None
    gpu_cache_size: Optional[int] = None
    host_cache_size: int = 0
    parameter_server: Optional[Any] = None


def validate_collection_config(config: InferenceEmbeddingCollectionConfig) -> None:
    if config.nve_layer_type not in {"gpu", "linear_uvm", "hierarchical"}:
        raise ValueError(f"Unsupported NVE layer type: {config.nve_layer_type}")
    if config.nve_layer_type == "hierarchical" and config.parameter_server is None:
        raise ValueError("Hierarchical NVE requires parameter_server")
    if config.nve_layer_type != "hierarchical" and config.parameter_server is not None:
        raise ValueError("parameter_server is only used by hierarchical NVE")
    if (
        config.indexer_type is EmbeddingCollectionIndexerType.BIT_CONCAT
        and config.nve_layer_type != "hierarchical"
    ):
        raise ValueError("BitConcatIndexer is supported only with hierarchical NVE")
    if config.nve_layer_type != "gpu" and config.gpu_cache_size is None:
        raise ValueError("LinearUVM and hierarchical NVE require gpu_cache_size")
    if config.bucket_capacity <= 0:
        raise ValueError("bucket_capacity must be positive")
