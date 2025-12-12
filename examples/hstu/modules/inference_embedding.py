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

import os

# pyre-strict
from typing import Dict, List, Optional

import torch
from configs import EmbeddingBackend, InferenceEmbeddingConfig
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from torchrec.modules.embedding_configs import EmbeddingConfig, dtype_to_data_type
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class ParameterServer(torch.nn.Module):
    pass


class DummyParameterServer(ParameterServer):
    def __init__(self, embedding_configs):
        super().__init__()
        self._embedding_collection = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name=config.table_name,
                    embedding_dim=config.dim,
                    num_embeddings=config.vocab_size,
                    feature_names=config.feature_names,
                    data_type=dtype_to_data_type(torch.float32),
                )
                for config in embedding_configs
            ],
            device=torch.device("meta"),
        )

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        return self._embedding_collection(features)


def create_dynamic_embedding_tables(
    embedding_configs: List[InferenceEmbeddingConfig],
    output_dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    ps: Optional[ParameterServer] = None,
    sparse_shareables=None,
):
    table_options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            dim=config.dim,
            max_capacity=config.vocab_size,
            local_hbm_for_values=0,
            bucket_capacity=128,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.NORMAL,
            ),
        )
        for config in embedding_configs
    ]

    table_names = [config.table_name for config in embedding_configs]

    return BatchedDynamicEmbeddingTablesV2(
        table_options=table_options,
        table_names=table_names,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        output_dtype=output_dtype,
    )


class InferenceDynamicEmbeddingCollection(torch.nn.Module):
    def __init__(
        self,
        embedding_configs,
        ps: Optional[ParameterServer] = None,
        enable_cache: bool = False,
        sparse_shareables=None,
    ):
        super().__init__()

        self._embedding_tables = create_dynamic_embedding_tables(
            embedding_configs, ps=ps, sparse_shareables=sparse_shareables
        )

        self._cache = (
            create_dynamic_embedding_tables(
                embedding_configs, device=torch.cuda.current_device()
            )
            if enable_cache
            else None
        )

        self._feature_names = [
            feature for config in embedding_configs for feature in config.feature_names
        ]

        self._features_split_sizes: List[int] = []
        self._features_split_indices: List[int] = []

    def set_feature_splits(self, features_split_size, features_split_indices):
        self._features_split_sizes = features_split_size
        self._features_split_indices = features_split_indices

    def load_checkpoint(self, checkpoint_dir):
        if checkpoint_dir is None:
            return

        embedding_table_dir = os.path.join(
            checkpoint_dir,
            "dynamicemb_module",
            "model._embedding_collection._model_parallel_embedding_collection",
        )

        try:
            for idx, table_name in enumerate(self._embedding_tables.table_names):
                self._embedding_tables.load(
                    embedding_table_dir, optim=False, table_names=[table_name]
                )
        except ValueError as e:
            warnings.warn(
                f"FAILED TO LOAD dynamic embedding tables failed due to ValueError:\n\t{e}\n\n"
                "Please check if the checkpoint is version 1. The loading of this old version is disabled."
            )

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        with torch.no_grad():
            features_split = features.split(self._features_split_sizes)
            features = KeyedJaggedTensor.concat(
                [features_split[idx] for idx in self._features_split_indices]
            )
            embeddings = self._embedding_tables(features.values(), features.offsets())
        embeddings_kjt = KeyedJaggedTensor(
            values=embeddings,
            keys=features.keys(),
            lengths=features.lengths(),
            offsets=features.offsets(),
        )
        return embeddings_kjt.to_dict()


def create_embedding_collection(configs, backend, use_static: bool = False, **kwargs):
    if backend == EmbeddingBackend.TORCHREC:
        assert (
            use_static == True
        ), "Do not support dynamic embedding table with TorchRec backend"
        return EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name=config.table_name,
                    embedding_dim=config.dim,
                    num_embeddings=config.vocab_size,
                    feature_names=config.feature_names,
                    data_type=dtype_to_data_type(torch.float32),
                )
                for config in configs
            ],
            device=torch.cuda.current_device(),
        )
    elif backend == EmbeddingBackend.DYNAMICEMB:
        assert (
            use_static == False
        ), "Only support dynamic embedding table with DynamicEmb backend"
        ps = kwargs.get("ps", None)
        enable_cache = kwargs.get("enable_cache", False)
        sparse_shareables = kwargs.get("sparse_shareables", False)
        return InferenceDynamicEmbeddingCollection(
            configs, ps, enable_cache, sparse_shareables
        )
    elif backend == EmbeddingBackend.NVEMB:
        from modules.nve_embeddingcollection import InferenceNVEEmbeddingCollection

        assert (
            InferenceNVEEmbeddingCollection is not None
        ), "Cannot create embedding collection for NV-Embeddings backend"
        sparse_shareables = kwargs.get("sparse_shareables", False)
        return InferenceNVEEmbeddingCollection(
            configs=[
                EmbeddingConfig(
                    name=config.table_name,
                    embedding_dim=config.dim,
                    num_embeddings=config.vocab_size,
                    feature_names=config.feature_names,
                    data_type=torch.float32,
                )
                for config in configs
            ],
            device=torch.cuda.current_device(),
            use_gpu_only=use_static,
            gpu_cache_ratio=kwargs.get("gpu_cache_ratio", 0.1),
            is_weighted=kwargs.get("is_weighted", False),
            sparse_shareables=sparse_shareables,
        )
    else:
        raise Exception("Unsupported embedding backend: {}".format(backend))


class InferenceEmbedding(torch.nn.Module):
    """
    InferenceEmbedding is a module for embeddings in the inference stage.

    Args:
        embedding_configs (List[InferenceEmbeddingConfig]): Configuration for the hstu (sharded) embedding.
        embedding_backend (EmbeddingBackend): Embedding collection backend.
    """

    def __init__(
        self,
        embedding_configs: List[InferenceEmbeddingConfig],
        embedding_backend: Optional[EmbeddingBackend] = None,
        sparse_shareables=None,
    ):
        super(InferenceEmbedding, self).__init__()

        self.dynamic_embedding_configs = []
        self.static_embedding_configs = []
        for config in embedding_configs:
            if not config.use_dynamicemb:
                self.static_embedding_configs.append(config)
            else:
                self.dynamic_embedding_configs.append(config)

        self.dynamic_emb_backend = (
            EmbeddingBackend.DYNAMICEMB
            if embedding_backend is None
            else embedding_backend
        )
        self.static_emb_backend = (
            EmbeddingBackend.TORCHREC
            if embedding_backend is None
            else embedding_backend
        )
        self._dynamic_embedding_collection = create_embedding_collection(
            configs=self.dynamic_embedding_configs,
            backend=self.dynamic_emb_backend,
            use_static=False,
            ps=None,
            enable_cache=False,
            gpu_cache_ratio=0.05,
            sparse_shareables=sparse_shareables,
        )

        self._static_embedding_collection = create_embedding_collection(
            configs=self.static_embedding_configs,
            backend=self.static_emb_backend,
            use_static=True,
        )
        self._side_stream = torch.cuda.Stream()
        self._static_embedding_collection = self._static_embedding_collection.to(
            torch.cuda.current_device()
        )

        features_split_sizes, features_split_indices = self.get_features_splits(
            embedding_configs
        )
        self._dynamic_embedding_collection.set_feature_splits(
            features_split_sizes, features_split_indices
        )

    def load_checkpoint(self, checkpoint_dir, model_state_dict=None):
        if checkpoint_dir is None:
            return

        self._dynamic_embedding_collection.load_checkpoint(checkpoint_dir)

        if model_state_dict is None:
            model_state_dict_path = os.path.join(
                checkpoint_dir, "torch_module", "model.0.pth"
            )
            model_state_dict = torch.load(model_state_dict_path)["model_state_dict"]
        self.load_state_dict(model_state_dict, strict=False)

    def load_state_dict(self, model_state_dict, *args, **kwargs):
        new_state_dict = {}
        for k in model_state_dict:
            if k.startswith(
                "_embedding_collection._data_parallel_embedding_collection.embeddings."
            ):
                emb_table_names = k.split(".")[-1].removesuffix("_weights").split("/")
                old_emb_table_weights = model_state_dict[k].view(
                    -1, self.static_embedding_configs[0].dim
                )
                weight_offset = 0
                # TODO(junyiq): Use a more flexible way to skip contextual features.
                for name in emb_table_names:
                    for emb_config in self.static_embedding_configs:
                        if name == emb_config.table_name:
                            emb_table_size = emb_config.vocab_size
                    newk = "_static_embedding_collection.embeddings." + name + ".weight"
                    new_state_dict[newk] = old_emb_table_weights[
                        weight_offset : weight_offset + emb_table_size
                    ]
                    weight_offset += emb_table_size
            else:
                continue

        unloaded_modules = super().load_state_dict(new_state_dict, *args, **kwargs)

        if self.dynamic_emb_backend == EmbeddingBackend.DYNAMICEMB:
            assert set(unloaded_modules.missing_keys) == set(
                ["_dynamic_embedding_collection._embedding_tables._empty_tensor"]
            )
        elif self.dynamic_emb_backend == EmbeddingBackend.NVEMB:
            for key in unloaded_modules.missing_keys:
                assert key.startswith("_dynamic_embedding_collection.embeddings")
        assert unloaded_modules.unexpected_keys == []

    def get_features_splits(self, embedding_configs):
        last_dynamic = None
        last_index = -1
        features_split_sizes = []
        for idx, emb_config in enumerate(embedding_configs):
            use_dynamicemb = emb_config.use_dynamicemb
            if last_dynamic != emb_config.use_dynamicemb:
                if last_dynamic is not None:
                    features_split_sizes.append(idx - last_index)
                last_index = idx
            last_dynamic = use_dynamicemb
        features_split_sizes.append(len(embedding_configs) - last_index)

        index = 1 if len(embedding_configs) % 2 != 0 ^ last_dynamic else 0
        features_split_indices = list(range(index, len(features_split_sizes), 2))

        return (features_split_sizes, features_split_indices)

    # @output_nvtx_hook(nvtx_tag="InferenceEmbedding", hook_tensor_attr_name="_values")
    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Forward pass of the sharded embedding module.

        Args:
            kjt (`KeyedJaggedTensor <https://pytorch.org/torchrec/concepts.html#keyedjaggedtensor>`): The input tokens.

        Returns:
            `Dict[str, JaggedTensor <https://pytorch.org/torchrec/concepts.html#jaggedtensor>]`: The output embeddings.
        """

        dynamic_embeddings = self._dynamic_embedding_collection(kjt)
        if self._static_embedding_collection is not None:
            with torch.cuda.stream(self._side_stream):
                static_embeddings = self._static_embedding_collection(kjt)
            torch.cuda.current_stream().wait_stream(self._side_stream)
            embeddings = {**dynamic_embeddings, **static_embeddings}
        else:
            embeddings = dynamic_embeddings
        return embeddings


def get_inference_sparse_model(
    embedding_configs: List[InferenceEmbeddingConfig],
    embedding_backend=None,
    sparse_shareables=None,
):
    return InferenceEmbedding(
        embedding_configs,
        embedding_backend,
        sparse_shareables,
    )
