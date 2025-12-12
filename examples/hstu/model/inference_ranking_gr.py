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
from typing import List, Tuple, Union

import torch
from configs import InferenceHSTUConfig, KVCacheConfig, RankingConfig
from dataset.utils import Batch
from modules.inference_dense_module import InferenceDenseModule
from modules.inference_embedding import InferenceEmbedding


class InferenceRankingGR(torch.nn.Module):
    """
    A class representing the ranking model inference.

    Args:
        sparse_module (InferenceHSTUConfig): The HSTU configuration.
        dense_module (RankingConfig): The ranking task configuration.
    """

    def __init__(
        self,
        sparse_module: torch.nn.Module,
        dense_module: torch.nn.Module,
    ):
        super().__init__()
        self.sparse_module = sparse_module
        self.dense_module = dense_module

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self.dense_module.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self.dense_module.half()
        return self

    def get_num_class(self):
        return self.dense_module.get_num_class()

    def get_num_tasks(self):
        return self.dense_module.get_num_tasks()

    def get_metric_types(self):
        return self.dense_module.get_metric_types()

    def load_checkpoint(self, checkpoint_dir):
        if checkpoint_dir is None:
            return

        model_state_dict_path = os.path.join(
            checkpoint_dir, "torch_module", "model.0.pth"
        )
        model_state_dict = torch.load(model_state_dict_path)["model_state_dict"]

        self.sparse_module.load_checkpoint(checkpoint_dir, model_state_dict)
        self.dense_module.load_state_dict(model_state_dict, strict=False)

    def get_user_kvdata_info(
        self,
        user_ids: Union[List[int], torch.Tensor],
        allow_bubble: bool = False,
        dbg_print: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dense_module.get_user_kvdata_info(user_ids, allow_bubble, dbg_print)

    def forward(
        self,
        batch: Batch,
        user_ids: torch.Tensor = None,
        user_start_pos: torch.Tensor = None,
    ):
        with torch.inference_mode():
            user_start_pos_cuda, kvcache_metadata = self.dense_module.kvcache_prelogue(
                batch, user_ids, user_start_pos
            )
            embeddings = self.sparse_module(batch.features)
            logits = self.dense_module(
                batch, embeddings, user_ids, user_start_pos_cuda, kvcache_metadata
            )

        return logits


def get_inference_ranking_gr(
    hstu_config: InferenceHSTUConfig,
    kvcache_config: KVCacheConfig,
    task_config: RankingConfig,
    use_cudagraph=False,
    cudagraph_configs=None,
    sparse_shareables=None,
):
    for ebc_config in task_config.embedding_configs:
        assert (
            ebc_config.dim == hstu_config.hidden_size
        ), "hstu layer hidden size should equal to embedding dim"

    inference_sparse = InferenceEmbedding(
        task_config.embedding_configs,
        sparse_shareables,
    )
    inference_dense = InferenceDenseModule(
        hstu_config,
        kvcache_config,
        task_config,
        use_cudagraph,
        cudagraph_configs,
    )

    return InferenceRankingGR(inference_sparse, inference_dense)
