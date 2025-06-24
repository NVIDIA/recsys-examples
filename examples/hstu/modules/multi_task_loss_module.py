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


def _decode_bits(encoded_labels: torch.Tensor, bit_width: int) -> torch.Tensor:
    """
    decode an integer to a list of bits

    Args:
        encoded_labels (torch.Tensor): (N,), The encoded labels tensor.
        bit_width (int): The bit width of the encoded labels.

    Returns:
        torch.Tensor: (N, bit_width), The decoded labels tensor.

    e.g [2,1,0,3], bit_width = 2, then the output is [[0,1], [1,0], [0,0], [1,1]]
    Most-significant bit is the last task, least-significant bit is the first task
    """
    bit_positions = torch.arange(bit_width, device=encoded_labels.device)

    encoded_labels = encoded_labels.unsqueeze(-1)
    return (encoded_labels >> bit_positions) & 1


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
        self._loss_modules: torch.nn.Module
        self._num_tasks = num_tasks
        self._num_classes = num_classes
        if self._num_classes == self._num_tasks:
            self._loss_modules = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            assert (
                self._num_tasks == 1
            ), "num_tasks should be 1 for multi-class classification"
            self._loss_modules = torch.nn.CrossEntropyLoss(reduction=reduction)

    @output_nvtx_hook(nvtx_tag="loss computation")
    def forward(self, merged_logits, labels) -> torch.Tensor:
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
            decoded_labels = _decode_bits(labels, self._num_tasks).float()
            losses = self._loss_modules(merged_logits, decoded_labels)
            return losses
        else:
            return self._loss_modules(merged_logits, labels)
