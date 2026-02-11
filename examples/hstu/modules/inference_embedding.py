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
from typing import Dict, List

import torch
from configs import (
    EmbeddingBackend, 
    InferenceEmbeddingConfig,
)
from modules.dynamicemb_embeddingcollection import DynamicembEmbeddingCollection
from modules.nve_embeddingcollection import NVEEmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class InferenceEmbedding(torch.nn.Module):
    """
    InferenceEmbedding is a module for embeddings in the inference stage.

    Args:
        embedding_configs (List[InferenceEmbeddingConfig]): Configuration for the hstu (sharded) embedding.
    """

    def __init__(
        self,
        embedding_configs: List[List[EmbeddingConfig]],
    ):
        super(InferenceEmbedding, self).__init__()

        self._embedding_collections = torch.nn.ModuleList([
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

        embeddings = { 
            emb_name: emb 
            for emb_result in [ embc(kjt) for embc in self._embedding_collections ] 
            for emb_name, emb in emb_result.items()
        }
        return embeddings


def create_torchrec_embedding(
    embedding_collection: EmbeddingCollection,
    embedding_configs: List[EmbeddingConfig],
    embedding_backend_config: InferenceEmbeddingConfig,
    dynamic_table_names: List[str],
) -> torch.nn.Module:
    if len(embedding_configs) == 0:
        return None
    return embedding_collection.to(device=embedding_backend_config.device)
    
def create_dynamicemb(
    embedding_collection: EmbeddingCollection,
    embedding_configs: List[EmbeddingConfig],
    embedding_backend_config: InferenceEmbeddingConfig,
    dynamic_table_names: List[str],
) -> torch.nn.Module:
    if len(embedding_configs) == 0:
        return None
    return DynamicembEmbeddingCollection(
        embedding_configs=embedding_configs,
        embedding_backend_config=embedding_backend_config,
        dynamic_table_names=dynamic_table_names,
    )

def create_nve(
    embedding_collection: EmbeddingCollection,
    embedding_configs: List[EmbeddingConfig],
    embedding_backend_config: InferenceEmbeddingConfig,
    dynamic_table_names: List[str],
) -> torch.nn.Module:
    if len(embedding_configs) == 0:
        return None
    return NVEEmbeddingCollection(
        embedding_configs=embedding_configs,
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
    embedding_backend_configs: Dict[str, List[InferenceEmbeddingConfig]],
    dynamic_table_names: List[str],
) -> torch.nn.Module:
    embc_module_names = set()
    for name, module in model.named_modules():
        if name not in embedding_backend_configs:
            continue
        embc_modulelist = module._embedding_collections
        embc_configs = embedding_backend_configs[name]
        for idx, embc in enumerate(embc_modulelist):
            emb_backend = embc_configs[idx].backend
            create_embedding_collection = select_embedding(emb_backend)
            embc_modulelist[idx] = create_embedding_collection(
                embc,
                embc.embedding_configs(),
                embc_configs[idx],
                dynamic_table_names,
            )
    
    return model
