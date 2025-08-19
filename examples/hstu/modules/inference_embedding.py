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
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTables
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

    return BatchedDynamicEmbeddingTables(
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
    ):
        super().__init__()

        self._embedding_tables = create_dynamic_embedding_tables(
            embedding_configs, ps=ps
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

        self._has_uninitialized_input_dist = True
        self._features_order: List[int] = []
        self._features_order_tensor = torch.zeros(
            (len(self._feature_names)),
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )

    def get_input_dist(
        self,
        input_feature_names: List[str],
    ) -> int:
        input_features_order = []
        for f in self._feature_names:
            if f in input_feature_names:
                input_features_order.append(input_feature_names.index(f))

        num_input_features = len(input_features_order)

        if (
            self._has_uninitialized_input_dist
            or input_features_order != self._features_order
        ):
            self._features_order = (
                []
                if input_features_order == list(range(num_input_features))
                else input_features_order
            )
            if len(self._features_order) > 0:
                self._features_order_tensor[:num_input_features].copy_(
                    torch.tensor(
                        input_features_order,
                        device=torch.cuda.current_device(),
                        dtype=torch.int32,
                    )
                )

        if self._has_uninitialized_input_dist:
            self._has_uninitialized_input_dist = False

        return num_input_features

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        num_input_features = self.get_input_dist(input_feature_names=features.keys())
        with torch.no_grad():
            if self._features_order:
                features = features.permute(
                    self._features_order,
                    self._features_order_tensor[: len(self._features_order)],
                )
            features = features.split([num_input_features])[0]
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
        return InferenceDynamicEmbeddingCollection(configs, ps, enable_cache)
    elif backend == EmbeddingBackend.NVEMB:
        from modules.nve_embeddingcollection import InferenceNVEEmbeddingCollection

        assert (
            InferenceNVEEmbeddingCollection is not None
        ), "Cannot create embedding collection for NV-Embeddings backend"
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
    ):
        super(InferenceEmbedding, self).__init__()

        dynamic_embedding_configs = []
        static_embedding_configs = []
        for config in embedding_configs:
            if not config.use_dynamicemb:
                static_embedding_configs.append(config)
            else:
                dynamic_embedding_configs.append(config)

        dynamic_emb_backend = (
            EmbeddingBackend.DYNAMICEMB
            if embedding_backend is None
            else embedding_backend
        )
        static_emb_backend = (
            EmbeddingBackend.TORCHREC
            if embedding_backend is None
            else embedding_backend
        )
        self._dynamic_embedding_collection = create_embedding_collection(
            configs=dynamic_embedding_configs,
            backend=dynamic_emb_backend,
            use_static=False,
            ps=None,
            enable_cache=False,
            gpu_cache_ratio=0.1,
        )

        self._static_embedding_collection = create_embedding_collection(
            configs=static_embedding_configs,
            backend=static_emb_backend,
            use_static=True,
        )
        self._side_stream = torch.cuda.Stream()
        self._static_embedding_collection = self._static_embedding_collection.to(
            torch.cuda.current_device()
        )

        @torch.no_grad()
        def init_weights(m):
            for param in m.parameters():
                torch.nn.init.ones_(param)

        self._static_embedding_collection.apply(init_weights)

    # @output_nvtx_hook(nvtx_tag="InferenceEmbedding", hook_tensor_attr_name="_values")
    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Forward pass of the sharded embedding module.

        Args:
            kjt (`KeyedJaggedTensor <https://pytorch.org/torchrec/concepts.html#keyedjaggedtensor>`): The input tokens.

        Returns:
            `Dict[str, JaggedTensor <https://pytorch.org/torchrec/concepts.html#jaggedtensor>]`: The output embeddings.
        """

        kjt = kjt.to(device=torch.cuda.current_device())

        dynamic_embeddings = self._dynamic_embedding_collection(kjt)
        if self._static_embedding_collection is not None:
            with torch.cuda.stream(self._side_stream):
                static_embeddings = self._static_embedding_collection(kjt)
                for feat_key in static_embeddings:
                    static_embeddings[feat_key] = static_embeddings[feat_key].to(
                        device=torch.cuda.current_device(), non_blocking=True
                    )
            torch.cuda.current_stream().wait_stream(self._side_stream)
            embeddings = {**dynamic_embeddings, **static_embeddings}
        else:
            embeddings = dynamic_embeddings
        return embeddings
