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

import math
from dataclasses import dataclass

# pyre-strict
from typing import Dict, List, Optional, Union

import torch
from dynamicemb import DynamicEmbPoolingMode, DynamicEmbTableOptions
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb.utils import TORCHREC_TYPES
from torchrec.modules.embedding_configs import (
    EmbeddingConfig,
    data_type_to_dtype,
    dtype_to_data_type,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
    get_embedding_names_by_table,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
import torch.nn.functional as F

def power_of_2(x: int) -> int:
    return 16 if x < 16 else 2 ** math.ceil(math.log2(x))


class InferenceEmbeddingCollection(torch.nn.Module):
    def __init__(
        self,
        ebc: EmbeddingCollection,
    ):
        super().__init__()
        embedding_configs = ebc.embedding_configs()
        self._feature_names = []
        self._feature_name_to_table_name = {}
        self._table_name_to_preprocessing_table = {}
        self._table_name_to_embedding_weights = {}
        for config in embedding_configs:
            print("config", config)
            print("16 * math.ceil(config.num_embeddings / 16)", 16 * math.ceil(config.num_embeddings / 16))
            self._table_name_to_preprocessing_table[config.name] = get_scored_table(
                capacity=power_of_2(config.num_embeddings),
                bucket_capacity=16 * math.ceil(config.num_embeddings / 16),
                key_type=torch.int64,
                score_specs=[ScoreSpec(name="score", policy=ScorePolicy.GLOBAL_TIMER)],
                device=torch.cuda.current_device(),
            )
            self._table_name_to_embedding_weights[config.name] = torch.empty(config.num_embeddings, config.embedding_dim, device=torch.cuda.current_device(), dtype=torch.float32)
            self._nve_layer = 
            if config.init_fn is not None:
                config.init_fn(self._table_name_to_embedding_weights[config.name])
            self._feature_names.extend(config.feature_names)
            for feature_name in config.feature_names:
                self._feature_name_to_table_name[feature_name] = config.name
        self._has_uninitialized_input_dist = True
        self._device = torch.cuda.current_device()

    def _create_input_dist(
        self,
        input_feature_names: List[str],
    ) -> None:
        nonpreprocessing_feature_names = [f for f in input_feature_names if f not in self._feature_names]
        self._features_order: List[int] = []
        print("input_feature_names", input_feature_names)
        print("self._feature_names", self._feature_names)
        for f in self._feature_names + nonpreprocessing_feature_names:
            self._features_order.append(input_feature_names.index(f))
        print("self._features_order", self._features_order)
        self._features_order = (
            []
            if self._features_order == list(range(len(self._features_order)))
            else self._features_order
        )
        self.register_buffer(
            "_features_order_tensor",
            torch.tensor(self._features_order, device=self._device, dtype=torch.int32),
            persistent=False,
        )
        self._feature_splits = [len(self._feature_names), len(input_feature_names) - len(self._feature_names)]
        print("self._feature_splits", self._feature_splits)

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(input_feature_names=features.keys())
            self._has_uninitialized_input_dist = False
        
        jt_dict = {}
        with torch.no_grad():
            if self._features_order:
                features = features.permute(
                    self._features_order, self._features_order_tensor
                )
            preprocessing_features, _ = features.split(self._feature_splits)
            for feature_name, jt in preprocessing_features.to_dict().items():
                table_name = self._feature_name_to_table_name[feature_name]
                table = self._table_name_to_preprocessing_table[table_name]
                keys = jt.values().to(table.key_type)
                founds = torch.zeros(
                    keys.numel(),
                    device=jt.values().device,
                    dtype=torch.bool,
                )
                indices = torch.empty(
                    jt.values().numel(),
                    device=jt.values().device,
                    dtype=table.index_type,
                )
                table.lookup(
                    keys, 
                    scores=[ScoreArg(name="score")], 
                    founds=founds, 
                    indices=indices,
                )
                indices = torch.where(founds, indices, 0)
                jt_dict[feature_name] = JaggedTensor(
                    lengths=jt.lengths(), 
                    values=F.embedding(indices, self._table_name_to_embedding_weights[table_name])
                )
                self._nve_layer = NVEmbedding()
                num_embeddings=config.num_embeddings,
                customized_emebedding(self._nve_layer, )
        return jt_dict


@dataclass
class InferenceOptions:
    global_hbm_for_values: int = 0


# TODO: remove this after nvembedding is OSS.
class InferenceDynamicEmbedding(torch.nn.Module):
    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        dynamicemb_options: Optional[DynamicEmbTableOptions],
    ):
        super().__init__()
        if dynamicemb_options is None:
            self.embedding = EmbeddingCollection(
                tables=[embedding_config],
                device=torch.cuda.current_device(),
            )
        else:
            assert (
                dynamicemb_options.training is False
            ), "Training mode is not supported for inference."
            self.embedding = BatchedDynamicEmbeddingTablesV2(
                table_options=[dynamicemb_options],
                table_names=[embedding_config.table_name],
                pooling_mode=DynamicEmbPoolingMode.NONE,
                output_dtype=data_type_to_dtype(embedding_config.data_type),
            )

    def set_feature_splits(self, features_split_size, features_split_indices):
        self._features_split_sizes = features_split_size
        self._features_split_indices = features_split_indices

    def forward(self, features: JaggedTensor) -> torch.Tensor:
        with torch.no_grad():
            return self.embedding(features.values(), features.offsets())[0]


# class InferenceEmbeddingCollection(torch.nn.Module):
#     """
#     InferenceEmbedding is a module for embeddings in the inference stage.

#     Args:
#         embedding_configs (List[InferenceEmbeddingConfig]): Configuration for the hstu (sharded) embedding.
#         embedding_backend (EmbeddingBackend): Embedding collection backend.
#     """

#     def __init__(
#         self,
#         configs: List[EmbeddingConfig],
#         dynamicemb_options_dict: Dict[str, InferenceOptions],
#         device: Optional[torch.device] = None,
#     ):
#         super().__init__()
#         self.embeddings: torch.nn.ModuleDict = torch.nn.ModuleDict()
#         self._embedding_configs = configs
#         self._device: torch.device = (
#             device if device is not None else torch.cuda.current_device()
#         )

#         table_names = set()
#         for embedding_config in configs:
#             if embedding_config.name in table_names:
#                 raise ValueError(f"Duplicate table name {embedding_config.name}")
#             table_names.add(embedding_config.name)

#             self.embeddings[embedding_config.name] = create_inference_embedding_tables(
#                 config, dynamicemb_options_dict.get(embedding_config.name, None)
#             )

#             self._embedding_names: List[str] = [
#                 embedding
#                 for embeddings in get_embedding_names_by_table(configs)
#                 for embedding in embeddings
#             ]
#             self._feature_names: List[List[str]] = [
#                 table.feature_names for table in configs
#             ]

#         self._side_stream = torch.cuda.Stream()
#         self._torchrec_embedding_collection = EmbeddingCollection(
#             tables=[
#                 EmbeddingConfig(
#                     name=config.table_name,
#                     embedding_dim=config.dim,
#                     num_embeddings=config.vocab_size,
#                     feature_names=config.feature_names,
#                     data_type=dtype_to_data_type(torch.float32),
#                 )
#                 for config in task_config.embedding_configs
#             ],
#             device=torch.device("meta"),
#         )

    #     self._static_embedding_collection = self._static_embedding_collection.to(
    #         torch.cuda.current_device()
    #     )

    #     features_split_sizes, features_split_indices = self.get_features_splits(
    #         embedding_configs
    #     )
    #     if isinstance(self._dynamic_embedding_collection, InferenceDynamicEmbeddingCollection):
    #         self._dynamic_embedding_collection.set_feature_splits(
    #             features_split_sizes, features_split_indices
    #         )

    # def get_features_splits(self, embedding_configs):
    #     last_dynamic = None
    #     last_index = -1
    #     features_split_sizes = []
    #     for idx, emb_config in enumerate(embedding_configs):
    #         use_dynamicemb = emb_config.use_dynamicemb
    #         if last_dynamic != emb_config.use_dynamicemb:
    #             if last_dynamic is not None:
    #                 features_split_sizes.append(idx - last_index)
    #             last_index = idx
    #         last_dynamic = use_dynamicemb
    #     features_split_sizes.append(len(embedding_configs) - last_index)

    #     index = 1 if len(embedding_configs) % 2 != 0 ^ last_dynamic else 0
    #     features_split_indices = list(range(index, len(features_split_sizes), 2))

    #     return (features_split_sizes, features_split_indices)

    # # @output_nvtx_hook(nvtx_tag="InferenceEmbedding", hook_tensor_attr_name="_values")
    # def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
    #     """
    #     Forward pass of the sharded embedding module.

    #     Args:
    #         kjt (`KeyedJaggedTensor <https://pytorch.org/torchrec/concepts.html#keyedjaggedtensor>`): The input tokens.

    #     Returns:
    #         `Dict[str, JaggedTensor <https://pytorch.org/torchrec/concepts.html#jaggedtensor>]`: The output embeddings.
    #     """

    #     dynamic_embeddings = self._dynamic_embedding_collection(kjt)
    #     if self._static_embedding_collection is not None:
    #         with torch.cuda.stream(self._side_stream):
    #             static_embeddings = self._static_embedding_collection(kjt)
    #         torch.cuda.current_stream().wait_stream(self._side_stream)
    #         embeddings = {**dynamic_embeddings, **static_embeddings}
    #     else:
    #         embeddings = dynamic_embeddings
    #     return embeddings


# def apply_nvembedding_inference(
#     model: torch.nn.Module, options_dict: Dict[str, InferenceOptions]
# ) -> torch.nn.Module:
#     print("before apply_dynamicemb_inference", model, type(model._embedding_collection))
#     embedding_collection_module_names = []
#     embedding_collection_modules = []
#     for k, module in model.named_modules():
#         if type(module) is EmbeddingBagCollection:
#             raise Exception("EmbeddingBagCollection is not supported for inference.")
#         if type(module) in TORCHREC_TYPES:
#             embedding_collection_module_names.append(k)
#             embedding_collection_modules.append(module)

#     for module_name, ebc in zip(
#         embedding_collection_module_names, embedding_collection_modules
#     ):
#         embedding_table_names = [
#             config.table_name for config in ebc.embedding_configs()
#         ]
#         if not any(
#             embedding_table_name in dynamicemb_options_dict
#             for embedding_table_name in embedding_table_names
#         ):
#             continue
#         module_name_list = module_name.split(".")
#         father_module = model
#         for name in module_name_list[:-1]:
#             father_module = getattr(father_module, name)
#         print("father_module", father_module)
#         infer_module = InferenceEmbeddingCollection(
#             ebc.embedding_configs(), dynamicemb_options_dict=dynamicemb_options_dict
#         )
#         setattr(father_module, module_name_list[-1], infer_module)
#     # model = quantize_embeddings(
#     #     model,
#     #     dtype=torch.qint8,
#     #     inplace=True
#     # )
#     print("after apply_dynamicemb_inference", model, type(model._embedding_collection))
#     raise
#     return model


def apply_dynamicemb_preprocessing(
    model: torch.nn.Module,
) -> torch.nn.Module:
    print("before apply_dynamicemb_inference", model)
    embedding_collection_module_names = []
    embedding_collection_modules = []
    for k, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            embedding_collection_module_names.append(k)
            embedding_collection_modules.append(module)

    for module_name, ebc in zip(
        embedding_collection_module_names, embedding_collection_modules
    ):
        infer_module = InferenceEmbeddingCollection(ebc)
        setattr(model, module_name, infer_module)
    print("after apply_dynamicemb_inference", model)
    return model
