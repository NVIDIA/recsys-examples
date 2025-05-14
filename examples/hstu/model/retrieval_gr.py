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
from collections import OrderedDict
from typing import Optional, Tuple

import torch
from commons.utils.nvtx_op import output_nvtx_hook
from configs import HSTUConfig, RetrievalConfig
from dataset.utils import RetrievalBatch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import (
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.transformer.module import Float16Module
from model.base_model import BaseModel
from modules.embedding import ShardedEmbedding
from modules.hstu_block import HSTUBlock
from modules.metrics.metric_modules import RetrievalTaskMetricWithSampling
from modules.negatives_sampler import InBatchNegativesSampler
from modules.output_postprocessors import L2NormEmbeddingPostprocessor
from modules.sampled_softmax_loss import SampledSoftmaxLoss
from modules.similarity.dot_product import DotProductSimilarity
from ops.triton_ops.triton_jagged import (  # type: ignore[attr-defined]
    triton_split_2D_jagged,
)


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
        ddp_config: Optional[DistributedDataParallelConfig] = None,
    ):
        super().__init__()
        self._tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert (
            self._tp_size == 1
        ), "RetrievalGR does not support tensor model parallel for now"
        self._device = torch.cuda.current_device()
        self._hstu_config = hstu_config
        self._task_config = task_config
        self._ddp_config = ddp_config

        self._embedding_dim = hstu_config.hidden_size
        for ebc_config in task_config.embedding_configs:
            assert (
                ebc_config.dim == self._embedding_dim
            ), "hstu layer hidden size should equal to embedding dim"

        self._embedding_collection = ShardedEmbedding(task_config.embedding_configs)

        self._hstu_block = HSTUBlock(hstu_config).cuda()

        self._dense_module = self._hstu_block
        # TODO, add ddp optimizer flag
        if hstu_config.bf16 or hstu_config.fp16:
            self._dense_module = Float16Module(hstu_config, self._hstu_block)
        if ddp_config is None:
            ddp_config = DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                use_distributed_optimizer=False,
                check_for_nan_in_grad=False,
                bucket_size=True,
            )
        self._dense_module = DDP(
            hstu_config,
            ddp_config,
            self._dense_module,
        )
        self._dense_module.broadcast_params()
        hstu_config.finalize_model_grads_func = finalize_model_grads

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
        self._metric_module = RetrievalTaskMetricWithSampling(
            metric_types=task_config.eval_metrics, MAX_K=500
        )

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RetrievalGR: The model with bfloat16 precision.
        """
        self._dense_module.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RetrievalGR: The model with half precision.
        """
        self._dense_module.half()
        return self

    def get_logit_and_labels(
        self, batch: RetrievalBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the logits and labels for the batch.

        Args:
            batch (RetrievalBatch): The batch of retrieval data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The logits, supervision item IDs, and supervision embeddings.
        """
        embeddings = self._embedding_collection(batch.features)
        jagged_data = self._hstu_block.hstu_preprocess(
            embeddings=embeddings,
            batch=batch,
        )
        jagged_data = self._dense_module(jagged_data)
        pred_item_embeddings = jagged_data.values
        pred_item_max_seqlen = jagged_data.max_seqlen
        pred_item_seqlen_offsets = jagged_data.seqlen_offsets

        supervision_item_embeddings = embeddings[batch.item_feature_name].values()
        supervision_item_ids = batch.features[batch.item_feature_name].values()
        if batch.max_num_candidates > 0:
            supervision_item_max_seqlen = batch.feature_to_max_seqlen[
                batch.item_feature_name
            ]
            supervision_item_seqlen_offsets = batch.features[
                batch.item_feature_name
            ].offsets()
            _, supervision_item_embeddings = triton_split_2D_jagged(
                supervision_item_embeddings,
                supervision_item_max_seqlen,
                offsets_a=supervision_item_seqlen_offsets - pred_item_seqlen_offsets,
                offsets_b=pred_item_seqlen_offsets,
            )
            _, supervision_item_ids = triton_split_2D_jagged(
                supervision_item_ids.view(-1, 1),
                supervision_item_max_seqlen,
                offsets_a=supervision_item_seqlen_offsets - pred_item_seqlen_offsets,
                offsets_b=pred_item_seqlen_offsets,
            )

        shift_pred_item_seqlen_offsets = torch.clamp(
            pred_item_seqlen_offsets - 1, min=0
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
            last_n_supervision_item_ids.view(-1),
            last_n_supervision_item_embeddings.view(-1, self._embedding_dim),
        )

    @output_nvtx_hook(nvtx_tag="RetrievalModel", backward=False)
    def forward(  # type: ignore[override]
        self,
        batch: RetrievalBatch,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Perform the forward pass of the model.

        Args:
            batch (RetrievalBatch): The batch of retrieval data.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: The losses and a tuple of losses, logits, and supervision embeddings.
        """
        (
            jagged_item_logit,
            supervision_item_ids,
            supervision_emb,
        ) = self.get_logit_and_labels(batch)

        losses = self._loss_module(
            jagged_item_logit.float(), supervision_item_ids, supervision_emb.float()
        )
        return losses, (
            losses.detach(),
            jagged_item_logit.detach(),
            supervision_emb.detach(),
        )

    def evaluate_one_batch(self, batch: RetrievalBatch) -> None:
        """
        Evaluate one batch of data.

        Args:
            batch (RetrievalBatch): The batch of retrieval data.
        """
        with torch.no_grad():
            jagged_item_logit, supervision_item_ids, _ = self.get_logit_and_labels(
                batch
            )
            self._metric_module(jagged_item_logit.float(), supervision_item_ids)
        self._item_feature_name = batch.item_feature_name

    def compute_metric(self) -> "OrderedDict":
        """
        Compute the evaluation metrics.

        Returns:
            OrderedDict: The computed metrics.
        """
        for embedding_config in self._task_config.embedding_configs:
            if self._item_feature_name in embedding_config.feature_names:
                table_name = embedding_config.table_name
        eval_dict, _, _ = self._metric_module.compute(
            self._embedding_collection, table_name=table_name
        )
        return eval_dict
