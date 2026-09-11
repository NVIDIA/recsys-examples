# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exportable inference embedding-collection conversion and population."""

import itertools
import os
from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from dynamicemb.batched_dynamicemb_tables import (
    encode_meta_json_file_path,
    get_loading_files,
)
from dynamicemb.exportable_embedding.config import (
    BitConcatConfig,
    EmbeddingCollectionIndexerType,
    InferenceEmbeddingCollectionConfig,
    validate_collection_config,
)
from dynamicemb.exportable_embedding.indexers import (
    BitConcatIndexer,
    EmbeddingCollectionIndexerBase,
    FusedIdentityIndexer,
    IdentityIndexer,
    LinearHashMapIndexer,
    create_embedding_collection_indexer,
)
from dynamicemb.exportable_embedding.nve_runtime import (
    create_nve_layer,
    insert_parameter_server,
    parameter_server_backend,
)
from dynamicemb.key_value_table import _iter_batches_from_files, load_from_json
from torch.nn import ModuleDict
from torch.nn.modules.sparse import Embedding
from torchrec.modules.embedding_configs import EmbeddingConfig


def _derive_grouped_offsets(feature_table_map: Sequence[int]) -> list[int]:
    offsets = [0]
    previous = feature_table_map[0]
    for index, table_id in enumerate(feature_table_map[1:], start=1):
        if table_id != previous:
            offsets.append(index)
            previous = table_id
    offsets.append(len(feature_table_map))
    return offsets


class InferenceEmbeddingCollection(torch.nn.Module):
    """One configurable indexer feeding one exportable NVE layer."""

    def __init__(
        self,
        embedding_configs: Sequence[EmbeddingConfig],
        *,
        config: InferenceEmbeddingCollectionConfig,
        indexer: EmbeddingCollectionIndexerBase,
        nve_embedding: Optional[torch.nn.Module],
        pooling_mode: int,
        table_names: Sequence[str],
        feature_names: Sequence[str],
        feature_table_map: Sequence[int],
        output_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.embedding_configs = list(embedding_configs)
        self.config_ = config
        self.indexer_ = indexer
        self.nve_embedding_ = nve_embedding
        self.parameter_server_ = config.parameter_server
        self.pooling_mode_ = pooling_mode
        self.table_names_ = list(table_names)
        self.feature_names_ = list(feature_names)
        self.num_tables_ = len(self.embedding_configs)
        self.table_capacities_ = tuple(
            int(table.num_embeddings) for table in self.embedding_configs
        )
        self.emb_dim_ = int(self.embedding_configs[0].embedding_dim)
        self.output_dtype_ = output_dtype
        self.device = device
        self.collection_id_: Optional[str] = None
        self._capture_nvhashmap_sidecar = (
            config.nve_layer_type == "hierarchical"
            and parameter_server_backend(config.parameter_server) == "nvhashmap"
        )
        self._hierarchical_population_batches: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []
        self.register_buffer(
            "feature_offsets_",
            torch.tensor(
                _derive_grouped_offsets(feature_table_map),
                dtype=torch.int64,
                device=device,
            ),
        )

    @property
    def failed_build_rows(self) -> frozenset[tuple[int, int]]:
        return self.indexer_.failed_build_rows

    def _nve_embedding(self) -> torch.nn.Module:
        if self.nve_embedding_ is None:
            raise RuntimeError("Load the collection source before lookup")
        return self.nve_embedding_

    @torch.no_grad()
    def _write_storage_rows(
        self, storage_keys: torch.Tensor, embeddings: torch.Tensor
    ) -> None:
        if storage_keys.numel() == 0:
            return
        if self.config_.nve_layer_type == "hierarchical":
            keys_cpu = storage_keys.detach().to(
                device="cpu", dtype=torch.int64
            ).contiguous()
            values_cpu = embeddings.detach().to(
                device="cpu", dtype=self.output_dtype_
            ).contiguous()
            insert_parameter_server(
                self.parameter_server_, keys_cpu, values_cpu, self.output_dtype_
            )
            if self._capture_nvhashmap_sidecar:
                self._hierarchical_population_batches.append(
                    (keys_cpu.clone(), values_cpu.clone())
                )
            return

        weight = self._nve_embedding().weight.data
        weight.index_copy_(
            0,
            storage_keys.to(device=weight.device, dtype=torch.int64),
            embeddings.to(device=weight.device, dtype=weight.dtype),
        )

    @torch.no_grad()
    def _build_and_write_rows(
        self,
        feature_ids: torch.Tensor,
        table_ids: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        storage_keys = self.indexer_.build_index(feature_ids, table_ids)
        retained = storage_keys >= 0
        self._write_storage_rows(
            storage_keys[retained],
            embeddings[retained.to(device=embeddings.device)],
        )

    @torch.no_grad()
    def _begin_source_load(self) -> None:
        if isinstance(self.indexer_, LinearHashMapIndexer):
            self.indexer_.reset_for_build()
        if self.config_.nve_layer_type == "hierarchical":
            self._hierarchical_population_batches.clear()
        else:
            self._nve_embedding().weight.data.zero_()

        if isinstance(
            self.indexer_, (LinearHashMapIndexer, FusedIdentityIndexer)
        ):
            self._write_storage_rows(
                self.indexer_.miss_storage_indices,
                torch.zeros(
                    (self.num_tables_, self.emb_dim_),
                    dtype=self.output_dtype_,
                    device=self.device,
                ),
            )

    @torch.no_grad()
    def _finish_source_load(self) -> None:
        self.indexer_.finish_build()
        if self.config_.nve_layer_type == "hierarchical":
            self.nve_embedding_ = create_nve_layer(
                num_embeddings=self.indexer_.nve_num_embeddings,
                embedding_dim=self.emb_dim_,
                dtype=self.output_dtype_,
                pooling_mode=self.pooling_mode_,
                config=self.config_,
                device=self.device,
            )

    @torch.no_grad()
    def load_from_embedding_table(self, table_weights: torch.Tensor) -> None:
        """Populate from table-concatenated TorchRec checkpoint weights."""
        self._begin_source_load()
        source_offset = 0
        for table_id, capacity in enumerate(self.table_capacities_):
            feature_ids = torch.arange(
                capacity, dtype=torch.int64, device=self.device
            )
            table_ids = torch.full_like(feature_ids, table_id)
            self._build_and_write_rows(
                feature_ids,
                table_ids,
                table_weights[source_offset : source_offset + capacity],
            )
            source_offset += capacity
        self._finish_source_load()

    @torch.no_grad()
    def load_from_dynamicemb_file(
        self,
        save_dir: str,
        table_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Populate from a complete DynamicEmb dump."""
        if not os.path.exists(save_dir):
            raise RuntimeError(f"Save directory does not exist: {save_dir}")

        selected_names = set(
            self.table_names_ if table_names is None else table_names
        )
        self._begin_source_load()
        for table_id, table_name in enumerate(self.table_names_):
            if table_name not in selected_names:
                continue
            meta_path = encode_meta_json_file_path(save_dir, table_name)
            if os.path.exists(meta_path):
                load_from_json(meta_path)
            (
                key_files,
                value_files,
                score_files,
                _optimizer_files,
                _counter_key_files,
                _counter_frequency_files,
            ) = get_loading_files(save_dir, table_name, rank=0, world_size=1)
            for file_index, (key_file, value_file) in enumerate(
                zip(key_files, value_files)
            ):
                score_file = (
                    score_files[file_index]
                    if file_index < len(score_files)
                    else None
                )
                for feature_ids, embeddings, _scores, _optimizer_states in (
                    _iter_batches_from_files(
                        key_file,
                        value_file,
                        score_file,
                        None,
                        self.emb_dim_,
                        0,
                        self.device,
                    )
                ):
                    table_ids = torch.full(
                        (feature_ids.numel(),),
                        table_id,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    self._build_and_write_rows(
                        feature_ids, table_ids, embeddings
                    )
        self._finish_source_load()

    def forward(
        self,
        keys: torch.Tensor,
        offsets: torch.Tensor,
        pooling_offsets: Optional[torch.Tensor] = None,
        per_sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        table_ids = torch.ops.INFERENCE_EMB.expand_table_ids(
            offsets,
            keys,
            self.feature_offsets_,
            self.num_tables_,
            1,
        )
        storage_keys = self.indexer_(keys, table_ids)
        nve_embedding = self._nve_embedding()
        if self.pooling_mode_ < 0:
            return nve_embedding(storage_keys)
        return nve_embedding(
            storage_keys, pooling_offsets, per_sample_weights
        )


def create_inference_embedding_collection(
    embedding_configs: Sequence[EmbeddingConfig],
    *,
    pooling_mode: int,
    config: InferenceEmbeddingCollectionConfig,
    output_dtype: torch.dtype = torch.float32,
    key_type: torch.dtype = torch.int64,
    device: Optional[torch.device] = None,
) -> InferenceEmbeddingCollection:
    configs = list(embedding_configs)
    if not configs:
        raise ValueError("embedding_configs must not be empty")
    if pooling_mode not in (-1, 1, 2):
        raise ValueError("pooling_mode must be -1, 1, or 2")
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    validate_collection_config(config)

    embedding_dims = {int(table.embedding_dim) for table in configs}
    if len(embedding_dims) != 1:
        raise ValueError("Tables in one collection must share embedding_dim")
    embedding_dim = embedding_dims.pop()
    table_names = [table.name for table in configs]
    feature_names_by_table = [
        list(table.feature_names) if table.feature_names else [table.name]
        for table in configs
    ]
    feature_names = list(itertools.chain.from_iterable(feature_names_by_table))
    feature_table_map = list(
        itertools.chain.from_iterable(
            [table_id] * len(names)
            for table_id, names in enumerate(feature_names_by_table)
        )
    )
    indexer = create_embedding_collection_indexer(
        config,
        [int(table.num_embeddings) for table in configs],
        key_type=key_type,
        device=device,
    )
    nve_embedding = None
    if config.nve_layer_type != "hierarchical":
        nve_embedding = create_nve_layer(
            num_embeddings=indexer.nve_num_embeddings,
            embedding_dim=embedding_dim,
            dtype=output_dtype,
            pooling_mode=pooling_mode,
            config=config,
            device=device,
        )

    return InferenceEmbeddingCollection(
        configs,
        config=config,
        indexer=indexer,
        nve_embedding=nve_embedding,
        pooling_mode=pooling_mode,
        table_names=table_names,
        feature_names=feature_names,
        feature_table_map=feature_table_map,
        output_dtype=output_dtype,
        device=device,
    )


def _resolve_collection_config(
    embedding_configs: Sequence[EmbeddingConfig],
    embedding_collection_configs: Mapping[
        str, InferenceEmbeddingCollectionConfig
    ],
) -> InferenceEmbeddingCollectionConfig:
    configs = [embedding_collection_configs[table.name] for table in embedding_configs]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Tables in one physical collection must share one config")
    return configs[0]


def _resolve_pooling_mode(embedding_configs: Sequence[EmbeddingConfig]) -> int:
    pooling = getattr(embedding_configs[0], "pooling", "NONE")
    pooling_name = getattr(pooling, "name", pooling)
    if pooling_name == "NONE":
        return -1
    if pooling_name == "SUM":
        return 1
    if pooling_name == "MEAN":
        return 2
    raise ValueError(f"Unsupported pooling config: {pooling}")


def _replace_submodule(
    model: torch.nn.Module,
    module_path: str,
    replacement: torch.nn.Module,
) -> None:
    parent_path, separator, child_name = module_path.rpartition(".")
    parent = model.get_submodule(parent_path) if separator else model
    setattr(parent, child_name, replacement)


def apply_inference_embedding_collection(
    model: torch.nn.Module,
    embedding_collection_configs: Mapping[
        str, InferenceEmbeddingCollectionConfig
    ],
    trained_emb_table_sizes: Mapping[str, int],
) -> torch.nn.Module:
    """Replace TorchRec embedding collections with configured NVE collections."""
    checked_modules: set[str] = set()
    while True:
        candidate_name = None
        for name, module in model.named_modules():
            if not isinstance(module, ModuleDict) or name in checked_modules:
                continue
            children = [
                child
                for child_name, child in module.named_modules()
                if child_name
            ]
            if (
                children
                and len({type(child) for child in children}) == 1
                and isinstance(children[0], Embedding)
            ):
                candidate_name = name
                break
        if candidate_name is None:
            break

        parent_name = candidate_name.removesuffix(".embeddings")
        parent_module = model.get_submodule(parent_name)
        embedding_configs = list(parent_module.embedding_configs())
        checked_modules.update((candidate_name, parent_name))
        for table in embedding_configs:
            table.num_embeddings = trained_emb_table_sizes.get(
                table.name, table.num_embeddings
            )

        collection_config = _resolve_collection_config(
            embedding_configs, embedding_collection_configs
        )
        replacement = create_inference_embedding_collection(
            embedding_configs,
            pooling_mode=_resolve_pooling_mode(embedding_configs),
            config=collection_config,
        )
        replacement.collection_id_ = parent_name
        _replace_submodule(model, parent_name, replacement)
        print(
            f"[INFO] converted {parent_name}: "
            f"{collection_config.indexer_type.value} + "
            f"{collection_config.nve_layer_type}"
        )
    return model


__all__ = [
    "BitConcatConfig",
    "BitConcatIndexer",
    "EmbeddingCollectionIndexerBase",
    "EmbeddingCollectionIndexerType",
    "FusedIdentityIndexer",
    "IdentityIndexer",
    "InferenceEmbeddingCollection",
    "InferenceEmbeddingCollectionConfig",
    "LinearHashMapIndexer",
    "apply_inference_embedding_collection",
    "create_inference_embedding_collection",
]
