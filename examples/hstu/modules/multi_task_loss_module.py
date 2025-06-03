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

import torch
from commons.utils.nvtx_op import output_nvtx_hook


class MultiTaskLossModule(torch.nn.Module):
    """
    Multi-task loss module for handling multiple loss functions. A loss head is either a
      BCEWithLogitsLoss or CrossEntropyLoss.

    Args:
        logit_dim_list (List[int]): List of dimensions for each loss's logits.
        reduction (str, optional): Specifies the reduction operation to apply to the losses
          ``'none'`` | ``'mean'`` | ``'sum'``. Defaults to ``'none'``.
    """

    def __init__(self, num_classes: int, num_tasks: int, reduction="none"):
        super().__init__()
        self._loss_modules = torch.nn.ModuleList()
        self._num_tasks = num_tasks
        self._num_classes = num_classes
        if self._num_classes == self._num_tasks:
            for _ in range(self._num_tasks):
                self._loss_modules.append(
                    torch.nn.BCEWithLogitsLoss(reduction=reduction)
                )
        else:
            assert (
                self._num_tasks == 1
            ), "num_tasks should be 1 for multi-class classification"
            self._loss_modules.append(torch.nn.CrossEntropyLoss(reduction=reduction))

    @output_nvtx_hook(nvtx_tag="loss computation")
    def forward(self, merged_logits, labels):
        """
        Forward pass of the MultiTaskLossModule.

        Args:
            merged_logits (torch.Tensor): (N, num_tasks),The merged logits tensor. Must be 2D tensor of float dtype.
            labels (torch.Tensor): (N,), The labels tensor.

        Returns:
            torch.Tensor: The computed losses for each task.
        """
        assert merged_logits.dim() == 2, "loss module expects 2D logit"
        assert merged_logits.dtype == torch.float, "merged_logits dtype should be float"
        assert (
            labels.dtype == torch.int32 or labels.dtype == torch.int64
        ), "labels dtype should be integer"

        if self._num_classes == self._num_tasks:
            losses = []
            for task_idx in range(self._num_tasks):
                task_logits = merged_logits[:, task_idx]
                task_labels = (torch.bitwise_and(labels, 1 << task_idx) > 0).to(
                    torch.float
                )
                loss = self._loss_modules[task_idx](task_logits, task_labels)
                losses.append(loss)
            return torch.stack(losses, dim=1)
        else:
            return self._loss_modules[0](merged_logits, labels)
