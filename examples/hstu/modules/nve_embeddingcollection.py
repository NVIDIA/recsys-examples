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
from typing import Dict, List, Optional

import torch
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import get_embedding_names_by_table
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

try:
    import pynve.torch.nve_layers as nve_layers

    class InferenceNVEEmbeddingCollection(torch.nn.Module):
        def __init__(
            self,
            configs: List[EmbeddingConfig],
            device: Optional[torch.device] = None,
            use_gpu_only: bool = False,
            gpu_cache_ratio: float = 0.01,
            is_weighted: bool = False,
        ):
            super().__init__()
            self._is_weighted = is_weighted
            self.embeddings: torch.nn.ModuleDict = torch.nn.ModuleDict()
            self._embedding_configs = configs
            self._device: torch.device = (
                device if device is not None else torch.cuda.current_device()
            )
            self._lengths_per_embedding: List[int] = []

            table_names = set()
            for embedding_config in configs:
                if embedding_config.name in table_names:
                    raise ValueError(f"Duplicate table name {embedding_config.name}")
                table_names.add(embedding_config.name)
                if not use_gpu_only:
                    gpu_cache_size = int(
                        embedding_config.num_embeddings * gpu_cache_ratio
                    )
                    gpu_cache_size *= embedding_config.embedding_dim
                    gpu_cache_size *= torch.tensor(
                        [], dtype=embedding_config.data_type
                    ).element_size()
                    self.embeddings[embedding_config.name] = nve_layers.NVEmbedding(
                        num_embeddings=embedding_config.num_embeddings,
                        embedding_size=embedding_config.embedding_dim,
                        data_type=embedding_config.data_type,
                        cache_type=nve_layers.CacheType.LinearUVM,
                        gpu_cache_size=gpu_cache_size,
                        optimize_for_training=False,
                    )
                else:
                    self.embeddings[embedding_config.name] = nve_layers.NVEmbedding(
                        num_embeddings=embedding_config.num_embeddings,
                        embedding_size=embedding_config.embedding_dim,
                        data_type=embedding_config.data_type,
                        cache_type=nve_layers.CacheType.NoCache,
                        optimize_for_training=False,
                    )

                if not embedding_config.feature_names:
                    embedding_config.feature_names = [embedding_config.name]
                self._lengths_per_embedding.extend(
                    len(embedding_config.feature_names)
                    * [embedding_config.embedding_dim]
                )

            self._embedding_names: List[str] = [
                embedding
                for embeddings in get_embedding_names_by_table(configs)
                for embedding in embeddings
            ]
            self._feature_names: List[List[str]] = [
                table.feature_names for table in configs
            ]

        def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
            """
            Run the EmbeddingCollection forward pass. This method takes in a `KeyedJaggedTensor`
            and returns a `dict` of `JaggedTensor`, which is the result embeddings for each feature.

            Args:
                features (KeyedJaggedTensor): Input features
            Returns:
                Dict[str, JaggedTensor]
            """
            result_embeddings: Dict[str, torch.Tensor] = dict()
            feature_dict = features.to_dict()
            for i, embedding in enumerate(self.embeddings.values()):
                for feature_name in self._feature_names[i]:
                    f = feature_dict[feature_name]
                    res = embedding(f.values())
                    result_embeddings[feature_name] = JaggedTensor(
                        values=res, lengths=f.lengths()
                    )

            return result_embeddings

        def load(self):
            pass

        def embedding_configs(
            self,
        ):
            return self._embedding_configs

        def is_weighted(self) -> bool:
            return self._is_weighted

except:
    print("NV-Embeddings is not installed. NVEMB backend is not supported.")
    nve_layers = None
    InferenceNVEEmbeddingCollection = None  # type: ignore
