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
from typing import Dict

import torch
from commons.datasets.hstu_batch import HSTUBatch
from configs import InferenceHSTUConfig, RankingConfig
from modules.inference_dense_module import InferenceDenseModule
from modules.inference_embedding import InferenceEmbedding
from recsys_kvcache_manager.kvcache_config import KVCacheConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


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

    def strip_cached_tokens(self, batch, origin_num_cached):
        torch.cuda.nvtx.range_push("strip_cached_tokens")

        num_context = len(batch.contextual_feature_names)

        num_cached = torch.clamp_min(origin_num_cached - num_context, 0).to(torch.int32)
        num_cached_action = num_cached // 2
        num_cached_item = num_cached - num_cached_action
        num_hist_cached = torch.concat([num_cached_item, num_cached_action], dim=0)

        old_offsets = batch.features.offsets().cpu()
        old_lengths = batch.features.lengths().cpu()

        item_offset = num_context * batch.batch_size

        new_lengths = torch.zeros_like(old_lengths)
        new_lengths[:item_offset] = torch.where(
            (origin_num_cached == 0).view(-1, batch.batch_size),
            old_lengths[:item_offset].view(-1, batch.batch_size),
            new_lengths[:item_offset].view(-1, batch.batch_size),
        ).view(-1)
        new_lengths[item_offset:] = old_lengths[item_offset:] - num_hist_cached

        startpos = (
            old_offsets[item_offset : item_offset + 2 * batch.batch_size]
            + num_hist_cached
        )
        endpos = old_offsets[item_offset + 1 :]

        old_values = batch.features.values()
        new_hist_value = [
            old_values[startpos[idx] : endpos[idx]]
            for idx in range(2 * batch.batch_size)
        ]

        new_context_value = [
            old_values[idx : idx + 1]
            for idx in range(num_context * batch.batch_size)
            if int(new_lengths[idx]) > 0
        ]

        new_features = KeyedJaggedTensor(
            values=torch.cat(new_context_value + new_hist_value, dim=0),
            lengths=new_lengths.cuda(),
            keys=batch.features.keys(),
        )
        batch.features = new_features

        torch.cuda.nvtx.range_pop()
        return batch

    def forward_with_kvcache(
        self,
        batch: HSTUBatch,
        user_ids: torch.Tensor,
        total_history_lengths: torch.Tensor,
    ):
        with torch.inference_mode():
            # lookup and allocate for kv cache
            index_meta, lookup_res = self.dense_module.kvcache.lookup_kvcache(
                user_ids,
                total_history_lengths,
            )
            kvcache_metadata = self.dense_module.kvcache.allocate_kvcache(
                index_meta, lookup_res
            )

            # asynchronous kvdata onboard, overlapping with strip_cached and embedding lookup
            self.dense_module.kvcache.onboard_launch(
                index_meta, lookup_res, kvcache_metadata
            )

            old_cached_lengths = lookup_res.cached_lengths
            striped_batch = self.strip_cached_tokens(
                batch,
                old_cached_lengths,
            )

            torch.cuda.nvtx.range_push("HSTU embedding")
            embeddings = self.sparse_module(striped_batch.features)
            torch.cuda.nvtx.range_pop()

            kvcache_info = (index_meta, lookup_res, kvcache_metadata)
            logits = self.dense_module.forward_with_kvcache(
                striped_batch,
                embeddings,
                user_ids,
                total_history_lengths,
                kvcache_info,
            )

        return logits

    def forward_nokvcache(
        self,
        batch: HSTUBatch,
    ):
        with torch.inference_mode():
            torch.cuda.nvtx.range_push("HSTU embedding")
            embeddings = self.sparse_module(batch.features)
            torch.cuda.nvtx.range_pop()
            logits = self.dense_module.forward_nokvcache(batch, embeddings)

        return logits

    def forward(
        self,
        batch: HSTUBatch,
    ):
        with torch.inference_mode():
            torch.cuda.nvtx.range_push("HSTU embedding")
            embeddings = self.sparse_module(batch.features)
            torch.cuda.nvtx.range_pop()
            logits = self.dense_module(batch, embeddings)
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


def apply_inference(
    training_model: torch.nn.Module,
    dynamic_table_configs: Dict[str, int],
    trained_emb_table_sizes: Dict[str, int],
    checkpoint_dir: str,
):
    from dynamicemb.exportable_tables import apply_inference_embedding_collection
    from modules.exportable_embedding import apply_inference_sparse
    from modules.inference_dense_module import apply_inference_hstu_dense

    # Step.1 - [General] Convert ModuleDict[Embedding] to InferenceEmbeddingCollection
    model = apply_inference_embedding_collection(
        training_model,
        dynamic_table_configs,
        trained_emb_table_sizes,
    )

    # Step.2 - [Recsys Example Structure Specific] Apply model specific training to inference conversion
    #          Convert ShardingEmbedding into ExportableEmbedding
    #          Convert dense module into InferenceDenseModule
    #          Convert RankingGR into InferenceRankingGR
    sparse_module = apply_inference_sparse(model._embedding_collection)
    dense_module = apply_inference_hstu_dense(
        hstu_config=model._hstu_config,
        kvcache_config=None,  # kvcache_config is not needed for export
        task_config=model._task_config,
        use_exportable=True,
        hstu_block=model._hstu_block,
        mlp=model._mlp,
    )

    inference_model = InferenceRankingGR(
        sparse_module,
        dense_module,
    )
    if model._hstu_config.bf16:
        inference_model.bfloat16()
    elif model._hstu_config.fp16:
        inference_model.half()

    inference_model.load_checkpoint(checkpoint_dir)
    inference_model = inference_model.eval()

    return inference_model
