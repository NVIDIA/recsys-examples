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
from typing import Dict, List, Optional

import torch
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import get_embedding_names_by_table
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

try:
    import pynve.torch.nve_ps as nve_ps
    from pynve.torch.nve_layers import CacheType, NVEmbedding

    def get_nve_local_ps(vocab_size, embedding_dim, torch_dtype):
        return nve_ps.NVLocalParameterServer(vocab_size, embedding_dim, torch_dtype)

    class InferenceNVEEmbeddingCollection(torch.nn.Module):
        def __init__(
            self,
            configs: List[EmbeddingConfig],
            device: Optional[torch.device] = None,
            use_gpu_only: bool = False,
            gpu_cache_ratio: float = 0.01,
            is_weighted: bool = False,
            sparse_shareables=None,
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
                    self.embeddings[embedding_config.name] = NVEmbedding(
                        num_embeddings=embedding_config.num_embeddings,
                        embedding_size=embedding_config.embedding_dim,
                        data_type=embedding_config.data_type,
                        cache_type=CacheType.Hierarchical,
                        gpu_cache_size=gpu_cache_size,
                        host_cache_size=0,
                        optimize_for_training=False,
                        remote_interface=sparse_shareables[embedding_config.name]
                        if sparse_shareables
                        else get_nve_local_ps(
                            0, embedding_config.embedding_dim, torch.float32
                        ),
                    )
                else:
                    self.embeddings[embedding_config.name] = NVEmbedding(
                        num_embeddings=embedding_config.num_embeddings,
                        embedding_size=embedding_config.embedding_dim,
                        data_type=embedding_config.data_type,
                        cache_type=CacheType.NoCache,
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

        def set_feature_splits(self, features_split_size, features_split_indices):
            pass

        def load_checkpoint(self, checkpoint_dir, model_state_dict=None):
            pass

        @classmethod
        def load_checkpoint_into_ps(
            self, ps_dict, checkpoint_dir=None, rank=0, world_size=1
        ):
            if checkpoint_dir is None:
                return

            for table_name in ps_dict:
                ps_dict[table_name].load_from_numpy_file(
                    os.path.join(
                        checkpoint_dir,
                        "ps_module",
                        f"{table_name}_emb_keys.rank_{rank}.world_size_{world_size}.npy",
                    ),
                    os.path.join(
                        checkpoint_dir,
                        "ps_module",
                        f"{table_name}_emb_values.rank_{rank}.world_size_{world_size}.npy",
                    ),
                )

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
