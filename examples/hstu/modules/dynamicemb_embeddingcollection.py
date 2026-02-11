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
from dataclasses import dataclass
from typing import Dict, List

import torch
from configs import InferenceEmbeddingConfig
from dynamicemb import (
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from torchrec.modules.embedding_configs import EmbeddingConfig, data_type_to_dtype
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

@dataclass
class DynamicemBackendConfig(InferenceEmbeddingConfig):
    caching: bool
    bucket_capacity: int = 128
    gpu_ratio_for_values: float = 0.0


class DynamicembEmbeddingCollection(torch.nn.Module):
    def __init__(
        self,
        embedding_configs: List[EmbeddingConfig],
        embedding_backend_config: InferenceEmbeddingConfig,
        dynamic_table_names: List[str],
    ):
        super().__init__()
        self.mappings = None

        self._dynamic_table_names = dynamic_table_names
        self._device = embedding_backend_config.device

        table_options = list()
        table_names = list()
        for config in embedding_configs:
            global_hbm_for_values = int(
                embedding_backend_config.gpu_ratio_for_values
                * config.num_embeddings 
                * config.embedding_dim
                * torch.tensor((), dtype=data_type_to_dtype(config.data_type)).element_size()
            )
            if config.name not in self._dynamic_table_names:
                global_hbm_for_values = int(
                    config.num_embeddings
                    * config.embedding_dim
                    * torch.tensor((), dtype=data_type_to_dtype(config.data_type)).element_size()
                )
            table_options.append(
                DynamicEmbTableOptions(
                    index_type=torch.int64,
                    embedding_dtype=data_type_to_dtype(config.data_type),  # torchrec to torch
                    dim=config.embedding_dim,
                    init_capacity=min(config.num_embeddings, 1024),
                    max_capacity=config.num_embeddings,
                    training=True,
                    caching=embedding_backend_config.caching,
                    global_hbm_for_values=global_hbm_for_values,
                    bucket_capacity=min(128, config.num_embeddings),
                )
            )
            table_names.append(config.name)

        self.embeddings = BatchedDynamicEmbeddingTablesV2(
            table_options=table_options,
            table_names=table_names,
            pooling_mode=DynamicEmbPoolingMode.NONE,
            device=self._device
        )

        # split features in static tables vs features in dynamic tables
        self._features_split_sizes: List[int] = []
        self._features_split_indices: List[int] = []

    def load_checkpoint(self, checkpoint_dir):
        pass

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
