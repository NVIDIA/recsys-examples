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
from typing import List

import torch
from commons.utils.nvtx_op import output_nvtx_hook
import torch.nn.functional as F


def _compute_loss(
    mt_logits: torch.Tensor,
    mt_labels: torch.Tensor,
) -> torch.Tensor:
    mt_losses = F.binary_cross_entropy_with_logits(
        input=mt_logits,
        target=mt_labels,
        reduction="none",
    )
    return mt_losses.sum(-1)


class MultiTaskLossModule(torch.nn.Module):
    """
    Multi-task loss module for handling multiple loss functions. A loss head is either a
      BCEWithLogitsLoss or CrossEntropyLoss.

    Args:
        logit_dim_list (List[int]): List of dimensions for each loss's logits.
        reduction (str, optional): Specifies the reduction operation to apply to the losses
          ``'none'`` | ``'mean'`` | ``'sum'``. Defaults to ``'none'``.
    """

    def __init__(self, logit_dim_list: List[int], reduction="none"):
        super().__init__()
        self._loss_modules = torch.nn.ModuleList()
        self._logit_dim_list = logit_dim_list
        for dim_size in self._logit_dim_list:
            self._loss_modules.append(
                torch.nn.BCEWithLogitsLoss(reduction=reduction)
                if dim_size == 1
                else torch.nn.CrossEntropyLoss(reduction=reduction)
            )
        self._loss_modules.cuda()

    @output_nvtx_hook(nvtx_tag="loss computation")
    def forward(self, merged_logits, labels):
        """
        Forward pass of the MultiTaskLossModule.

        Args:
            merged_logits (torch.Tensor): The merged logits tensor. Must be 2D tensor of float dtype.
            labels (torch.Tensor): The labels tensor.

        Returns:
            torch.Tensor: The computed losses for each task.
        """
        assert merged_logits.dim() == 2, "loss module expects 2D logit"
        assert merged_logits.dtype == torch.float, "merged_logits dtype should be float"
        assert (
            labels.dtype == torch.int32 or labels.dtype == torch.int64
        ), "labels dtype should be integer"
        logits_per_head = torch.split(merged_logits, self._logit_dim_list, dim=-1)
        labels_per_head = torch.split(labels, 1, dim=1)

        assert len(logits_per_head) == len(
            labels_per_head
        ), f"num head should match the input label"
        losses = []

        for head_logits, head_labels, loss_module in zip(
            logits_per_head, labels_per_head, self._loss_modules
        ):
            # for bce, the target must be float
            # for CE,  the target must be Long
            head_labels = (
                head_labels.float() if head_logits.size(1) == 1 else head_labels.long()
            )
            loss = loss_module(head_logits.squeeze(-1), head_labels.view(-1))
            losses.append(loss)
        return torch.stack(losses, dim=1)
