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
from commons.utils import initialize as init
from ops.pt_ops.pt_norm_mul_dropout import pytorch_norm_mul_dropout
from ops.triton_ops.triton_layer_norm import (
    triton_weighted_layer_norm_bwd,
    triton_weighted_layer_norm_fwd,
)
from ops.triton_ops.triton_norm_mul_dropout import triton_norm_mul_dropout


@pytest.mark.parametrize("training", [True])
@pytest.mark.parametrize("concat_ux", [False])
@pytest.mark.parametrize("dropout_ratio", [0.0, 0.5])
@pytest.mark.parametrize("hidden_dim", [128, 512])
@pytest.mark.parametrize("seed", [1234])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_ln_mul_dropout(
    input_dtype, training, concat_ux, dropout_ratio, hidden_dim, seed
):
    torch.backends.cuda.matmul.allow_tf32 = False
    init.initialize_distributed()
    init.set_random_seed(1234)
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
    u = (
        torch.empty(batchsize, hidden_dim, device=device, dtype=input_dtype)
        .fill_(0.5)
        .requires_grad_(True)
    )

    ref_x = x.detach().clone().requires_grad_(True)
    ref_u = u.detach().clone().requires_grad_(True)

    y = triton_norm_mul_dropout(
        x, u, ln_weight, ln_bias, eps, dropout_ratio, training, concat_ux, seed=seed
    )
    ref_y = pytorch_norm_mul_dropout(
        ref_x, ref_u, ref_weight, ref_bias, eps, dropout_ratio, training, concat_ux
    )
    if dropout_ratio == 0.0:
        torch.testing.assert_close(y, ref_y)

    dout = torch.empty_like(y).uniform_(-0.1, 0.1)

    y.backward(dout)
    ref_y.backward(dout)

    if dropout_ratio == 0.0:
        torch.testing.assert_close(ln_weight.grad, ref_weight.grad)
        torch.testing.assert_close(ln_bias.grad, ref_bias.grad)
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(u.grad, ref_u.grad)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize("batch_size", [200, 500, 1000])
@pytest.mark.parametrize("hidden_dim", [128, 256, 512])
def test_triton_layer_norm_with_grad_residual(dtype, batch_size, hidden_dim):
    input = (
        torch.empty(batch_size, hidden_dim, device="cuda", dtype=dtype)
        .uniform_(-1, 1)
        .requires_grad_(True)
    )
    ref_input = input.detach().clone().requires_grad_(True)

    weight = torch.nn.Parameter(
        torch.ones(hidden_dim, dtype=dtype).uniform_(-0.1, 0.1).cuda()
    )
    bias = torch.nn.Parameter(
        torch.zeros(hidden_dim, dtype=dtype).uniform_(-0.1, 0.1).cuda()
    )
    eps = 1e-5

    ref_output = (
        torch.nn.functional.layer_norm(ref_input, (hidden_dim,), weight, bias, eps)
        + ref_input
    )
    grad_output = torch.ones_like(ref_output) / hidden_dim
    ref_output.backward(grad_output)

    with torch.no_grad():
        output, mean, rstd, BLOCK_D, num_warps = triton_weighted_layer_norm_fwd(
            x=input,
            weight=weight,
            bias=bias,
            eps=eps,
        )
        output = output + ref_input
        dx, dweight, dbias = triton_weighted_layer_norm_bwd(
            dy=grad_output,
            x=input,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            learnable=True,
            eps=eps,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
            dx_accumulate=grad_output,
        )

    torch.testing.assert_close(output, ref_output)
    torch.testing.assert_close(dx, ref_input.grad)
