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
from typing import Tuple

import torch
from commons.utils.nvtx_op import output_nvtx_hook
from configs import HSTUConfig, RankingConfig
from dataset.utils import RankingBatch
from megatron.core import parallel_state
from model.base_model import BaseModel
from modules.embedding import ShardedEmbedding
from modules.hstu_block import HSTUBlock
from modules.metrics import get_multi_event_metric_module
from modules.mlp import MLP
from modules.multi_task_loss_module import MultiTaskLossModule
from modules.multi_task_over_arch import MultiTaskOverArch


class RankingGR(BaseModel):
    """
    A class representing the ranking model. Inherits from BaseModel. A ranking model consists of
    a sparse architecture and a dense architecture. A ranking model is able to process multiple labels
    and thus has multiple logit dimensions. Each label is associated with a loss functoin (e.g. BCE, CE).

    Args:
        hstu_config (HSTUConfig): The HSTU configuration.
        task_config (RankingConfig): The ranking task configuration.
    """

    def __init__(
        self,
        hstu_config: HSTUConfig,
        task_config: RankingConfig,
    ):
        super().__init__()
        self._tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert (
            self._tp_size == 1
        ), "RankingGR does not support tensor model parallel for now"
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._hstu_config = hstu_config
        self._task_config = task_config

        self._embedding_dim = hstu_config.hidden_size
        for ebc_config in task_config.embedding_configs:
            assert (
                ebc_config.dim == self._embedding_dim
            ), "hstu layer hidden size should equal to embedding dim"

        self._logit_dim_list = [
            layer_sizes[-1] for layer_sizes in task_config.prediction_head_arch
        ]
        self._embedding_collection = ShardedEmbedding(task_config.embedding_configs)

        self._hstu_block = HSTUBlock(hstu_config)
        self._multi_task_over_arch = MultiTaskOverArch(
            [
                MLP(
                    self._embedding_dim,
                    layer_sizes,
                    has_bias,
                    head_act_type,
                    device=self._device,
                )
                for layer_sizes, head_act_type, has_bias in zip(
                    task_config.prediction_head_arch,
                    task_config.prediction_head_act_type,
                    task_config.prediction_head_bias,  # type: ignore[arg-type]
                )
            ]
        )

        # TODO, make reduction configurable
        self._loss_module = MultiTaskLossModule(
            logit_dim_list=self._logit_dim_list, reduction="none"
        )
        self._metric_module = get_multi_event_metric_module(
            self._logit_dim_list,
            metric_types=task_config.eval_metrics,
            comm_pg=parallel_state.get_data_parallel_group(with_context_parallel=True),
        )

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self._hstu_block.bfloat16()
        self._multi_task_over_arch.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self._hstu_block.half()
        self._multi_task_over_arch.half()
        return self

    def get_logit_and_labels(
        self, batch: RankingBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the logits and labels for the batch.

        Args:
            batch (RankingBatch): The batch of ranking data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The logits and labels.
        """
        embeddings = self._embedding_collection(batch.features)
        hidden_states = self._hstu_block(
            embeddings=embeddings,
            batch=batch,
        )

        return self._multi_task_over_arch(hidden_states).values, batch.labels

    @output_nvtx_hook(nvtx_tag="RankingModel", backward=False)
    def forward(  # type: ignore[override]
        self,
        batch: RankingBatch,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Perform the forward pass of the model.

        Args:
            batch (RankingBatch): The batch of ranking data.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: The losses and a tuple of losses, logits, and labels.
        """
        jagged_item_logit, labels = self.get_logit_and_labels(batch)
        losses = self._loss_module(jagged_item_logit.float(), labels)
        return losses, (
            losses.detach(),
            jagged_item_logit.detach(),
            labels.detach(),
        )

    def evaluate_one_batch(self, batch: RankingBatch) -> None:
        """
        Evaluate one batch of data.

        Args:
            batch (RankingBatch): The batch of ranking data.
        """
        with torch.no_grad():
            jagged_item_logit, redistributed_labels = self.get_logit_and_labels(batch)
            self._metric_module(jagged_item_logit.float(), redistributed_labels)

    def compute_metric(self) -> "OrderedDict":
        """
        Compute the evaluation metrics.

        Returns:
            OrderedDict: The computed metrics.
        """
        ret_dict = self._metric_module.compute()
        return ret_dict
