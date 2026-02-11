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
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torchrec.modules.embedding_configs import (
    EmbeddingConfig,
)
from configs import InferenceEmbeddingConfig
from torchrec.modules.embedding_modules import get_embedding_names_by_table
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

try:
    import pynve.torch.nve_ps as nve_ps
    from pynve.torch.nve_layers import CacheType, NVEmbedding


    @dataclass
    class NVEBackendConfig(InferenceEmbeddingConfig):
        use_gpu_only: bool
        gpu_cache_ratio: float
        host_cache_ratio: float
        weight_init: Optional[torch.Tensor]
        memblock: Optional[nve.Tensor]

        dynamic_gpu_cache_size: int
        dynamic_host_cache_size: int
        dynamic_weight_init: Optional[torch.Tensor]
        dynamic_remote_interface: Optional[nve.Table | nve_ps.NVLocalParameterServer]
    

    def get_nve_local_ps(vocab_size, embedding_dim, torch_dtype):
        return nve_ps.NVLocalParameterServer(vocab_size, embedding_dim, torch_dtype)

    class NVEEmbeddingCollection(torch.nn.Module):
        def __init__(
            self,
            embedding_configs: List[EmbeddingConfig],
            embedding_backend_config: InferenceEmbeddingConfig,
            dynamic_table_names: List[str],
        ):
            super().__init__()
            self.mappings = None
            self.embeddings: torch.nn.ModuleDict = torch.nn.ModuleDict()
            self.dynamic_embeddings: torch.nn.ModuleDict = torch.nn.ModuleDict()

            self._embedding_configs = embedding_configs
            self._dynamic_table_names = dynamic_table_names
            self._device = embedding_backend_config.device

            # create embedding tables for fusion
            for config in embedding_configs:
                if config.name in self._dynamic_table_names:
                    continue

                if embedding_backend_config.use_gpu_only and config.name not in self._dynamic_table_names: # use gpu nve layer
                    self.embeddings[config.name] = NVEmbedding(
                        num_embeddings=config.num_embeddings,
                        embedding_size=config.embedding_dim,
                        data_type=config.data_type,
                        cache_type=CacheType.NoCache,
                        optimize_for_training=False,
                        device=self._device,
                    )
                elif config.name not in self._dynamic_table_names: # use linear nve layer
                    gpu_cache_size = int(
                        embedding_backend_config.gpu_cache_ratio
                        * config.num_embeddings
                        * config.embedding_dim
                        * torch.tensor([], dtype=config.data_type).element_size()
                    )
                    self.embeddings[config.name] = NVEmbedding(
                        num_embeddings=config.num_embeddings, 
                        embedding_size=config.embedding_dim, 
                        data_type=config.data_type, 
                        cache_type=nve_layers.CacheType.LinearUVM, 
                        gpu_cache_size=gpu_cache_size, 
                        weight_init=embedding_backend_config.weight_init, 
                        memblock=embedding_backend_config.memblock,
                        optimize_for_training=False,
                        device=self._device,
                    )
                else:
                    self.embeddings[config.name] = NVEmbedding(
                        num_embeddings=config.num_embeddings,
                        embedding_size=config.embedding_dim,
                        data_type=config.data_type,
                        gpu_cache_size=embedding_backend_config.dynamic_gpu_cache_size, 
                        host_cache_size=embedding_backend_config.dynamic_host_cache_size, 
                        remote_interface=embedding_backend_config.dynamic_remote_interface, 
                        optimize_for_training=False, 
                        device=self._device, 
                    )

            self._feature_names: List[List[str]] = [
                table.feature_names for table in _embedding_configs
            ]

        def embedding_configs(self) -> List[EmbeddingConfig]:
            return self._embedding_configs

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
                ps_dict[table_name].load_from_file(
                    os.path.join(
                        checkpoint_dir,
                        "ps_module",
                        f"{table_name}_emb_keys.rank_{rank}.world_size_{world_size}.dyn",
                    ),
                    os.path.join(
                        checkpoint_dir,
                        "ps_module",
                        f"{table_name}_emb_values.rank_{rank}.world_size_{world_size}.dyn",
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


except:
    print("NV-Embeddings is not installed. NVEMB backend is not supported.")
    nve_layers = None
    NVEEmbeddingCollection = None  # type: ignore
