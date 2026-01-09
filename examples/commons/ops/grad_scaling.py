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


class GradScale(torch.autograd.Function):
    """
    A custom autograd function for gradient scaling.

    Example:
        >>> data = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> scale = 0.1
        >>> scaled_data = GradScale.apply(data, scale)
        >>> scaled_data.backward(torch.ones_like(data))
        >>> data.grad
        tensor([0.1000, 0.1000, 0.1000])
    """

    @staticmethod
    def forward(ctx, data, scale):
        """"""
        ctx.scale = scale
        return data

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        return grad_output * ctx.scale, None


grad_scaling = GradScale.apply
