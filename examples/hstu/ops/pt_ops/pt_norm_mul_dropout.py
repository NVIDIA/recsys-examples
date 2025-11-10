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
import torch.nn.functional as F


def pytorch_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    concat_ux: bool = False,
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
) -> torch.Tensor:
    """
    All op is performed in fp32.
    """
    dtype = x.dtype
    x = x.to(torch.float32)
    u = u.reshape(u.size(0), -1).to(torch.float32)
    if group_norm:
        y = u * F.group_norm(
            x.view(-1, num_heads, linear_dim),
            num_groups=num_heads,
            weight=weight.to(torch.float32) if weight is not None else None,
            bias=bias.to(torch.float32) if bias is not None else None,
            eps=eps,
        ).view(-1, num_heads * linear_dim)
    else:
        y = u * F.layer_norm(
            x,
            normalized_shape=(x.shape[-1],),
            weight=weight.to(torch.float32) if weight is not None else None,
            bias=bias.to(torch.float32) if bias is not None else None,
            eps=eps,
        )
    if concat_ux:
        y = torch.cat([u, x, y], dim=1)
    y = F.dropout(
        y,
        p=dropout_ratio,
        training=training,
    )
    return y.to(dtype)
