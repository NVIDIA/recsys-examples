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


class GPTSIDLossModule(torch.nn.Module):
    """
    Multi-task loss module for handling multiple loss functions. A loss head is either a
      BCEWithLogitsLoss or CrossEntropyLoss.
    """

    def __init__(self, reduction="none"):
        super().__init__()
        self._loss_modules = torch.nn.CrossEntropyLoss(reduction=reduction)

    @output_nvtx_hook(nvtx_tag="loss computation")
    def forward(self, merged_logits, labels) -> torch.Tensor:
        """
        Forward pass of the GPTSIDLossModule.

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

        return self._loss_modules(merged_logits, labels)
