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
from configs import InferenceEmbeddingConfig
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTables
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner as DynamicEmbeddingShardingPlanner,
)
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


def create_embedding_collection(configs):
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
        device=torch.device("meta"),
    )


def create_dynamic_embedding_collection(
    configs, ps: Optional[ParameterServer] = None, enable_cache: bool = False
):
    return InferenceDynamicEmbeddingCollection(configs, ps, enable_cache)


class InferenceEmbedding(torch.nn.Module):
    """
    InferenceEmbedding is a module for embeddings in the inference stage.

    Args:
        embedding_configs (List[InferenceEmbeddingConfig]): Configuration for the hstu (sharded) embedding.
    """

    def __init__(
        self,
        embedding_configs: List[InferenceEmbeddingConfig],
    ):
        super(InferenceEmbedding, self).__init__()

        dynamic_embedding_configs = []
        nondynamic_embedding_configs = []
        for config in embedding_configs:
            if not config.use_dynamicemb:
                nondynamic_embedding_configs.append(config)
            else:
                dynamic_embedding_configs.append(config)

        self._dynamic_embedding_collection = create_dynamic_embedding_collection(
            configs=dynamic_embedding_configs, ps=None, enable_cache=False
        )

        self._nondynamic_embedding_collection = create_embedding_collection(
            configs=nondynamic_embedding_configs
        )
        self._side_stream = torch.cuda.Stream()
        self._nondynamicemb_device = torch.device("cpu")

        @torch.no_grad()
        def init_weights(m):
            for param in m.parameters():
                torch.nn.init.ones_(param)

        self._nondynamic_embedding_collection.apply(init_weights)

    def to_empty(self, device: torch.device):
        self._nondynamicemb_device = device
        self._nondynamic_embedding_collection.to_empty(device=device)
        self._dynamic_embedding_collection.to_empty(device=torch.cuda.current_device())

    # @output_nvtx_hook(nvtx_tag="InferenceEmbedding", hook_tensor_attr_name="_values")
    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Forward pass of the sharded embedding module.

        Args:
            kjt (`KeyedJaggedTensor <https://pytorch.org/torchrec/concepts.html#keyedjaggedtensor>`): The input tokens.

        Returns:
            `Dict[str, JaggedTensor <https://pytorch.org/torchrec/concepts.html#jaggedtensor>]`: The output embeddings.
        """

        kjt_dynamic = (
            kjt.to(device=torch.cuda.current_device())
            if kjt.device() == torch.device("cpu")
            else kjt
        )
        kjt_nondynamic = (
            kjt
            if kjt.device() == self._nondynamicemb_device
            else kjt.to(device=self._nondynamicemb_device)
        )

        dynamic_embeddings = self._dynamic_embedding_collection(kjt_dynamic)
        if self._nondynamic_embedding_collection is not None:
            with torch.cuda.stream(self._side_stream):
                nondynamic_embeddings = self._nondynamic_embedding_collection(
                    kjt_nondynamic
                )
                for feat_key in nondynamic_embeddings:
                    nondynamic_embeddings[feat_key] = nondynamic_embeddings[
                        feat_key
                    ].to(device=torch.cuda.current_device(), non_blocking=True)
            torch.cuda.current_stream().wait_stream(self._side_stream)
            embeddings = {**dynamic_embeddings, **nondynamic_embeddings}
        else:
            embeddings = dynamic_embeddings
        return embeddings
