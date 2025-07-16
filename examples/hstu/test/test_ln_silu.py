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
import pytest
import torch
import torch.nn.functional as F
from commons.utils import initialize as init
from ops.triton_ops.triton_layer_norm import (  # triton_rms_norm,
    triton_layer_norm,
    triton_swish_layer_norm,
)


def ref_layernorm(x, weight, bias, eps, swish=False):
    dtype = x.dtype
    x = x.to(torch.float32)
    y = F.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        bias=bias,
        eps=eps,
    )
    if swish:
        y = x * F.sigmoid(y)
    return y.to(dtype)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("swish", [True, False])
@pytest.mark.parametrize("hidden_dim", [128, 512])
def test_layernorm_swish(input_dtype, swish, hidden_dim):
    init.initialize_distributed()
    init.set_random_seed(1234)
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        pytest.skip("Skip test in distributed mode")
    device = torch.cuda.current_device()
    eps = 1e-5
    batchsize = 128

    ln_weight = torch.nn.Parameter(torch.ones(hidden_dim, device=device))
    ln_bias = torch.nn.Parameter(torch.zeros(hidden_dim, device=device))

    ref_weight = ln_weight.detach().clone().requires_grad_(True)
    ref_bias = ln_bias.detach().clone().requires_grad_(True)

    x = (
        torch.empty(batchsize, hidden_dim, device=device, dtype=input_dtype)
        .uniform_(-0.1, 0.1)
        .requires_grad_(True)
    )

    ref_x = x.detach().clone().requires_grad_(True)

    triton_norm = triton_layer_norm if not swish else triton_swish_layer_norm
    y = triton_norm(x, ln_weight, ln_bias, eps)
    ref_y = ref_layernorm(ref_x, ref_weight, ref_bias, eps, swish)

    torch.testing.assert_close(y, ref_y)

    dout = torch.empty_like(y).uniform_(-0.1, 0.1)

    y.backward(dout)
    ref_y.backward(dout)
    torch.testing.assert_close(ln_weight.grad, ref_weight.grad)
    torch.testing.assert_close(ln_bias.grad, ref_bias.grad)
