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
import warnings

# pyre-strict
from typing import Dict, List, Optional, Object

import torch
from configs import (
    EmbeddingBackend, 
    EmbeddingBackendConfig, 
    InferenceEmbeddingConfig,
)
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from modules.nve_embeddingcollection import NVEEmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingConfig, dtype_to_data_type
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


EMBEDDING_COLLECTION_MODULE_NAME = "_embedding_collections"


@dataclass
class DynamicemBackendConfig(EmbeddingBackendConfig):
    caching: bool
    bucket_capacity: int = 128
    gpu_ratio_for_values: float = 0.0


class DynamicembEmbeddingCollection(torch.nn.Module):
    def __init__(
        self,
        embedding_configs: List[EmbeddingConfig],
        embedding_backend_config: EmbeddingBackendConfig,
        dynamic_table_names: List[str],
    ):
        super().__init__()
        self.mappings = None

        self._dynamic_table_names = dynamic_table_names
        self._device = embedding_backend_config.device

        table_options = list()
        table_names = list()
        for config in embedding_configs:
            if config.name in self._dynamic_table_names:
                continue
            
            global_hbm_for_values = int(
                embedding_backend_config.gpu_ratio_for_values
                * config.num_embeddings 
                * config.embedding_dim
            )
            table_options.append(
                DynamicEmbTableOptions(
                    index_type=torch.int64,
                    embedding_dtype=dtype_to_data_type(config.data_type),
                    dim=config.embedding_dim,
                    training=False,
                    caching=embedding_backend_config.caching,
                    bucket_capacity=config.num_embeddings,
                    global_hbm_for_values=global_hbm_for_values,
                )
            )
            table_names.append(config.name)

        self.embeddings = BatchedDynamicEmbeddingTablesV2(
            table_options=table_options,
            table_names=table_names,
            pooling_mode=DynamicEmbPoolingMode.NONE,
            output_dtype=embedding_backend_config.output_dtype,
            device=self._device
        )

        table_options = list()
        table_names = list()
        for config in embedding_configs:
            if config.name not in self._dynamic_table_names:
                continue

            global_hbm_for_values = int(
                embedding_backend_config.gpu_ratio_for_values
                * config.num_embeddings 
                * config.embedding_dim
            )
            table_options.append(
                DynamicEmbTableOptions(
                    index_type=torch.int64,
                    embedding_dtype=dtype_to_data_type(config.data_type),
                    dim=config.embedding_dim,
                    training=False,
                    caching=embedding_backend_config.caching,
                    bucket_capacity=config.num_embeddings,
                    global_hbm_for_values=embedding_backend_config.hbm_ratio,
                )
            )
            table_names.append(config.name)

        self.dynamic_embeddings = BatchedDynamicEmbeddingTablesV2(
            table_options=table_options,
            table_names=table_names,
            pooling_mode=DynamicEmbPoolingMode.NONE,
            output_dtype=embedding_backend_config.output_dtype,
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


class Embedding(torch.nn.Module):
    """
    InferenceEmbedding is a module for embeddings in the inference stage.

    Args:
        embedding_configs (List[InferenceEmbeddingConfig]): Configuration for the hstu (sharded) embedding.
    """

    def __init__(
        self,
        embedding_configs: List[List[EmbeddingConfig]],
    ):
        super(Embedding, self).__init__()

        self._embedding_collections = ModuleList([
            EmbeddingCollection(
                tables=configs,
                device=torch.device("meta"),
            ) for configs in embedding_configs
        ])
        self._embedding_configs = embedding_configs

    def load_checkpoint(self, checkpoint_dir, model_state_dict=None):
        pass

    def load_state_dict(self, model_state_dict, *args, **kwargs):
        pass

    # @output_nvtx_hook(nvtx_tag="InferenceEmbedding", hook_tensor_attr_name="_values")
    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Forward pass of embedding module.

        Args:
            kjt (`KeyedJaggedTensor <https://pytorch.org/torchrec/concepts.html#keyedjaggedtensor>`): The input tokens.

        Returns:
            `Dict[str, JaggedTensor <https://pytorch.org/torchrec/concepts.html#jaggedtensor>]`: The output embeddings.
        """

        embeddings = { **embc(kjt) for embc in self._embedding_collections }
        return embeddings


def create_torchrec_embedding(
    embedding_collection: EmbeddingCollection,
    embedding_configs: List[EmbeddingConfig],
    embedding_backend_config: EmbeddingBackendConfig,
    dynamic_table_names: List[str],
) -> torch.nn.Module:
    if len(embedding_configs) == 0:
        return None
    return embedding_collection.to(device=embedding_backend_config.device)
    
def create_dynamicemb(
    embedding_collection: EmbeddingCollection,
    embedding_configs: List[EmbeddingConfig],
    embedding_backend_config: EmbeddingBackendConfig,
    dynamic_table_names: List[str],
) -> torch.nn.Module:
    if len(embedding_configs) == 0:
        return None
    return DynamicembEmbeddingCollection(
        configs=embedding_configs,
        embedding_backend_config=embedding_backend_config,
        dynamic_table_names=dynamic_table_names,
    )

def create_nve(
    embedding_collection: EmbeddingCollection,
    embedding_configs: List[EmbeddingConfig],
    embedding_backend_config: EmbeddingBackendConfig,
    dynamic_table_names: List[str],
) -> torch.nn.Module:
    if len(embedding_configs) == 0:
        return None
    return NVEEmbeddingCollection(
        configs=embedding_configs,
        embedding_backend_config=embedding_backend_config,
        dynamic_table_names=dynamic_table_names,
    )


def select_embedding(backend: EmbeddingBackend):
    if backend == EmbeddingBackend.TORCHREC:
        return create_torchrec_embedding
    elif backend == EmbeddingBackend.DYNAMICEMB:
        return create_dynamicemb
    elif backend == EmbeddingBackend.NVEMB:
        return create_nve
    else:
        raise InvalidValueError(f"No support for embedding backend {backend}")


def apply_inference_embedding(
    model: torch.nn.Module, 
    embedding_backend_configs: List[EmbeddingBackendConfig],
    dynamic_table_names: List[str],
) -> torch.nn.Module:

    def get_module_by_name(module_name) -> Optional[torch.nn.Module]:
        for name, module in model.named_modules():
            if name == module_name:
                return module
        return None
    
    embc_parent_module_names = set()
    for name, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            parent_name = name.rsplit('.', 1)[0]
            embc_parent_module_names.add(parent_name)
            # eb_configs.append(module.embedding_configs())
    assert len(embc_parent_module_names) == 1

    emb_module_name = list(embc_parent_module_names)[0]
    emb_module = get_module_by_name(embedding_module_name)
    inference_embedding_collections = list()
    for idx, embc in enumerate(emb_module._embedding_collections):
        emb_backend = embedding_backend_configs[idx].backend
        create_embedding_collection = select_embedding(emb_backend)
        inference_embc = create_embedding_collection(
            embc,
            embc.embedding_configs(),
            embedding_backend_configs[idx],
            dynamic_table_names,
        )
        inference_embedding_collections.append(inference_embc)
    setattr(
        emb_module,
        EMBEDDING_COLLECTION_MODULE_NAME,
        inference_embedding_collections,
    )
    
    return model
