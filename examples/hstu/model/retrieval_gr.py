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
from typing import Any, Tuple

import torch
from commons.modules.embedding import ShardedEmbedding
from commons.ops.length_to_offsets import length_to_complete_offsets
from commons.ops.triton_ops.triton_jagged import (  # type: ignore[attr-defined]
    triton_split_2D_jagged,
)
from commons.utils.nvtx_op import output_nvtx_hook
from configs import HSTUConfig, RetrievalConfig
from datasets.utils import RetrievalBatch
from megatron.core import parallel_state
from model.base_model import BaseModel
from modules.hstu_block import HSTUBlock
from modules.negatives_sampler import InBatchNegativesSampler
from modules.output_postprocessors import L2NormEmbeddingPostprocessor
from modules.sampled_softmax_loss import SampledSoftmaxLoss
from modules.similarity.dot_product import DotProductSimilarity


class RetrievalGR(BaseModel):
    """
    A class representing the retrieval model. Inherits from BaseModel. A retrieval model consists of
    a sparse architecture and a dense architecture. The loss for retrieval is computed using sampled softmax loss.

    Args:
        hstu_config (HSTUConfig): The HSTU configuration.
        task_config (RetrievalConfig): The retrieval task configuration.
        ddp_config (Optional[DistributedDataParallelConfig]): The distributed data parallel configuration. If not provided, will use default value.
    """

    def __init__(
        self,
        hstu_config: HSTUConfig,
        task_config: RetrievalConfig,
    ):
        super().__init__()
        self._tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert (
            self._tp_size == 1
        ), "RetrievalGR does not support tensor model parallel, because of the sampled softmax loss and evaluation"
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._hstu_config = hstu_config
        self._task_config = task_config

        self._embedding_dim = hstu_config.hidden_size
        for ebc_config in task_config.embedding_configs:
            assert (
                ebc_config.dim == self._embedding_dim
            ), "hstu layer hidden size should equal to embedding dim"
        self._embedding_collection = ShardedEmbedding(task_config.embedding_configs)

        self._hstu_block = HSTUBlock(hstu_config)

        self._loss_module = SampledSoftmaxLoss(
            num_to_sample=task_config.num_negatives,
            softmax_temperature=task_config.temperature,
            negatives_sampler=InBatchNegativesSampler(
                norm_func=L2NormEmbeddingPostprocessor(
                    embedding_dim=self._embedding_dim, eps=task_config.l2_norm_eps
                ),
                dedup_embeddings=True,
            ),
            interaction_module=DotProductSimilarity(
                dtype=torch.bfloat16 if hstu_config.bf16 else torch.float16
            ),
        )
        self._item_feature_name = None

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RetrievalGR: The model with bfloat16 precision.
        """
        self._hstu_block.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RetrievalGR: The model with half precision.
        """
        self._hstu_block.half()
        return self

    def get_logit_and_labels(
        self, batch: RetrievalBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the logits and labels for the batch.

        Args:
            batch (RetrievalBatch): The batch of retrieval data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The logits, supervision item IDs, and supervision embeddings.
        """
        assert (
            batch.max_num_candidates == 0
        ), "num candidates is not supported for retrieval"

        embeddings = self._embedding_collection(batch.features)
        jagged_data, seqlen_after_preprocessor = self._hstu_block(
            embeddings=embeddings,
            batch=batch,
        )
        pred_item_embeddings = jagged_data.values
        pred_item_max_seqlen = jagged_data.max_seqlen
        pred_item_seqlen_offsets = jagged_data.seqlen_offsets
        pred_item_seqlen = jagged_data.seqlen

        supervision_item_embeddings = embeddings[batch.item_feature_name].values()
        supervision_item_ids = batch.features[batch.item_feature_name].values()

        shift_pred_item_seqlen_offsets = length_to_complete_offsets(
            torch.clamp(pred_item_seqlen - 1, min=0)
        )
        first_n_pred_item_embeddings, _ = triton_split_2D_jagged(
            pred_item_embeddings,
            pred_item_max_seqlen,
            offsets_a=shift_pred_item_seqlen_offsets,
            offsets_b=pred_item_seqlen_offsets - shift_pred_item_seqlen_offsets,
        )

        _, last_n_supervision_item_embeddings = triton_split_2D_jagged(
            supervision_item_embeddings,
            pred_item_max_seqlen,
            offsets_a=pred_item_seqlen_offsets - shift_pred_item_seqlen_offsets,
            offsets_b=shift_pred_item_seqlen_offsets,
        )
        _, last_n_supervision_item_ids = triton_split_2D_jagged(
            supervision_item_ids.view(-1, 1),
            pred_item_max_seqlen,
            offsets_a=pred_item_seqlen_offsets - shift_pred_item_seqlen_offsets,
            offsets_b=shift_pred_item_seqlen_offsets,
        )
        return (
            first_n_pred_item_embeddings.view(-1, self._embedding_dim),
            seqlen_after_preprocessor,
            last_n_supervision_item_ids.view(-1),
            last_n_supervision_item_embeddings.view(-1, self._embedding_dim),
        )

    @output_nvtx_hook(nvtx_tag="RetrievalModel", backward=False)
    def forward(  # type: ignore[override]
        self,
        batch: RetrievalBatch,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]]:
        """
        Perform the forward pass of the model.

        Args:
            batch (RetrievalBatch): The batch of retrieval data.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]]: The losses and a tuple of losses, logits, and supervision embeddings.
        """
        if self._item_feature_name is None:
            self._item_feature_name = batch.item_feature_name

        (
            jagged_item_logit,
            seqlen_after_preprocessor,
            supervision_item_ids,
            supervision_emb,
        ) = self.get_logit_and_labels(batch)

        losses = self._loss_module(
            jagged_item_logit.float(), supervision_item_ids, supervision_emb.float()
        )
        return losses, (
            losses.detach(),
            jagged_item_logit.detach(),
            supervision_item_ids.detach(),
            seqlen_after_preprocessor,  # used to compute achieved flops/s
        )

    # used for evaluation
    def get_item_feature_table_name(self) -> str:
        for embedding_config in self._task_config.embedding_configs:
            if self._item_feature_name in embedding_config.feature_names:
                table_name = embedding_config.table_name
        return table_name
