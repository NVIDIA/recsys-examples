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
from megatron.core import parallel_state
from modules.tp_layer_norm import (
    TPLayerNorm,
    TPLayerNormMulDropout,
    _divide_with_exception,
)
from ops.pt_ops.pt_norm_mul_dropout import pytorch_norm_mul_dropout


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


@pytest.mark.parametrize("hidden_dim", [32, 128, 256])
@pytest.mark.parametrize("tp_size", [2, 4, 8, 1])
@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("shard_weight", [True, False])
def test_tp_layernorm(
    hidden_dim,
    tp_size,
    trainable,
    shard_weight,
):
    init.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    if tp_size > world_size:
        pytest.skip("Skip tp size is greater than world size")
    if shard_weight and not trainable:
        pytest.skip("Skip shard weight is True but trainable is False")
    device = torch.cuda.current_device()
    init.initialize_model_parallel(tp_size)
    init.set_random_seed(1234)
    # no need to broadcast the weight and bias because they are initialized the same on all ranks
    ref_weight = (
        torch.nn.Parameter(torch.ones(hidden_dim, device=device)) if trainable else None
    )
    ref_bias = (
        torch.nn.Parameter(torch.zeros(hidden_dim, device=device))
        if trainable
        else None
    )
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    hidden_dim_per_partition = _divide_with_exception(hidden_dim, tp_size)
    if shard_weight:
        tp_weight = (
            ref_weight[
                tp_rank
                * hidden_dim_per_partition : (tp_rank + 1)
                * hidden_dim_per_partition
            ]
            .detach()
            .clone()
            .requires_grad_(True)
            if trainable
            else None
        )
        tp_bias = (
            ref_bias[
                tp_rank
                * hidden_dim_per_partition : (tp_rank + 1)
                * hidden_dim_per_partition
            ]
            .detach()
            .clone()
            .requires_grad_(True)
            if trainable
            else None
        )
    else:
        tp_weight = (
            ref_weight.detach().clone().requires_grad_(True) if trainable else None
        )
        tp_bias = ref_bias.detach().clone().requires_grad_(True) if trainable else None

    batchsize = 128
    eps = 1e-5
    # do not gather output!
    tp_layernorm = TPLayerNorm(
        hidden_dim,
        eps,
        trainable=trainable,
        shard_weight=shard_weight,
        gather_output=False,
    ).cuda()
    if trainable:
        tp_layernorm.weight.data.copy_(tp_weight)
        tp_layernorm.bias.data.copy_(tp_bias)
    # test 10 forward-backward
    for i in range(10):
        ref_x = torch.empty(
            batchsize, hidden_dim, device=device, dtype=torch.bfloat16
        ).uniform_(-0.1, 0.1)
        # to ensure the same input within tp group
        torch.distributed.broadcast(
            ref_x,
            parallel_state.get_tensor_model_parallel_src_rank(),
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        ref_x = ref_x.requires_grad_(True)
        tp_x = (
            ref_x[
                ...,
                tp_rank
                * hidden_dim_per_partition : (tp_rank + 1)
                * hidden_dim_per_partition,
            ]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        ref_y = ref_layernorm(ref_x, ref_weight, ref_bias, eps, False)
        ref_tp_y = ref_y[
            ...,
            tp_rank
            * hidden_dim_per_partition : (tp_rank + 1)
            * hidden_dim_per_partition,
        ]
        tp_y = tp_layernorm(tp_x)
        torch.testing.assert_close(tp_y, ref_tp_y)

        dout = torch.empty_like(ref_y).uniform_(-0.1, 0.1)
        # to ensure the same dout within tp group
        torch.distributed.broadcast(
            dout,
            parallel_state.get_tensor_model_parallel_src_rank(),
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        tp_dout = dout[
            ...,
            tp_rank
            * hidden_dim_per_partition : (tp_rank + 1)
            * hidden_dim_per_partition,
        ].contiguous()

        tp_y.backward(tp_dout)
        ref_y.backward(dout)
        if trainable:
            if shard_weight:
                ref_tp_weight_grad = ref_weight.grad[
                    tp_rank
                    * hidden_dim_per_partition : (tp_rank + 1)
                    * hidden_dim_per_partition
                ]
                ref_tp_bias_grad = ref_bias.grad[
                    tp_rank
                    * hidden_dim_per_partition : (tp_rank + 1)
                    * hidden_dim_per_partition
                ]
            else:
                ref_tp_weight_grad = ref_weight.grad
                ref_tp_bias_grad = ref_bias.grad
            torch.testing.assert_close(tp_layernorm.weight.grad, ref_tp_weight_grad)
            torch.testing.assert_close(tp_layernorm.bias.grad, ref_tp_bias_grad)
        ref_tp_x_grad = ref_x.grad[
            ...,
            tp_rank
            * hidden_dim_per_partition : (tp_rank + 1)
            * hidden_dim_per_partition,
        ]

        torch.testing.assert_close(tp_x.grad, ref_tp_x_grad)


@pytest.mark.parametrize("hidden_dim", [32, 128, 256])
@pytest.mark.parametrize("tp_size", [2, 4, 8, 1])
@pytest.mark.parametrize("shard_weight", [True, False])
def test_tp_layernorm_dropout_mul(
    hidden_dim,
    tp_size,
    shard_weight,
):
    trainable = True  # must be trainable because reference torch op
    dropout_ratio = 0.0
    init.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    if tp_size > world_size:
        pytest.skip("Skip tp size is greater than world size")
    if shard_weight and not trainable:
        pytest.skip("Skip shard weight is True but trainable is False")
    device = torch.cuda.current_device()
    init.initialize_model_parallel(tp_size)
    init.set_random_seed(1234)
    # no need to broadcast the weight and bias because they are initialized the same on all ranks
    ref_weight = (
        torch.nn.Parameter(torch.ones(hidden_dim, device=device)) if trainable else None
    )
    ref_bias = (
        torch.nn.Parameter(torch.zeros(hidden_dim, device=device))
        if trainable
        else None
    )
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    hidden_dim_per_partition = _divide_with_exception(hidden_dim, tp_size)
    if shard_weight:
        tp_weight = (
            ref_weight[
                tp_rank
                * hidden_dim_per_partition : (tp_rank + 1)
                * hidden_dim_per_partition
            ]
            .detach()
            .clone()
            .requires_grad_(True)
            if trainable
            else None
        )
        tp_bias = (
            ref_bias[
                tp_rank
                * hidden_dim_per_partition : (tp_rank + 1)
                * hidden_dim_per_partition
            ]
            .detach()
            .clone()
            .requires_grad_(True)
            if trainable
            else None
        )
    else:
        tp_weight = (
            ref_weight.detach().clone().requires_grad_(True) if trainable else None
        )
        tp_bias = ref_bias.detach().clone().requires_grad_(True) if trainable else None

    batchsize = 128
    eps = 1e-5
    # do not gather output!
    tp_layernorm_dropout_mul = TPLayerNormMulDropout(
        hidden_dim,
        eps,
        dropout_ratio,
        trainable=trainable,
        shard_weight=shard_weight,
        gather_output=False,
    ).cuda()
    if trainable:
        tp_layernorm_dropout_mul.weight.data.copy_(tp_weight)
        tp_layernorm_dropout_mul.bias.data.copy_(tp_bias)
    # test 10 forward-backward
    for i in range(10):
        ref_x = torch.empty(
            batchsize, hidden_dim, device=device, dtype=torch.bfloat16
        ).uniform_(-0.1, 0.1)
        ref_u = torch.empty(
            batchsize, hidden_dim, device=device, dtype=torch.bfloat16
        ).uniform_(-0.1, 0.1)
        # to ensure the same input within tp group
        torch.distributed.broadcast(
            ref_x,
            parallel_state.get_tensor_model_parallel_src_rank(),
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        torch.distributed.broadcast(
            ref_u,
            parallel_state.get_tensor_model_parallel_src_rank(),
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        ref_x = ref_x.requires_grad_(True)
        ref_u = ref_u.requires_grad_(True)
        tp_x = (
            ref_x[
                ...,
                tp_rank
                * hidden_dim_per_partition : (tp_rank + 1)
                * hidden_dim_per_partition,
            ]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        tp_u = (
            ref_u[
                ...,
                tp_rank
                * hidden_dim_per_partition : (tp_rank + 1)
                * hidden_dim_per_partition,
            ]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        ref_y = pytorch_norm_mul_dropout(
            ref_x, ref_u, ref_weight, ref_bias, eps, dropout_ratio, training=True
        )
        ref_tp_y = ref_y[
            ...,
            tp_rank
            * hidden_dim_per_partition : (tp_rank + 1)
            * hidden_dim_per_partition,
        ]
        tp_y = tp_layernorm_dropout_mul(tp_x, tp_u)
        torch.testing.assert_close(tp_y, ref_tp_y)
        dout = torch.empty_like(ref_y).uniform_(-0.1, 0.1)
        # to ensure the same dout within tp group
        torch.distributed.broadcast(
            dout,
            parallel_state.get_tensor_model_parallel_src_rank(),
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        tp_dout = dout[
            ...,
            tp_rank
            * hidden_dim_per_partition : (tp_rank + 1)
            * hidden_dim_per_partition,
        ].contiguous()

        tp_y.backward(tp_dout)
        ref_y.backward(dout)
        if trainable:
            if shard_weight:
                ref_tp_weight_grad = ref_weight.grad[
                    tp_rank
                    * hidden_dim_per_partition : (tp_rank + 1)
                    * hidden_dim_per_partition
                ]
                ref_tp_bias_grad = ref_bias.grad[
                    tp_rank
                    * hidden_dim_per_partition : (tp_rank + 1)
                    * hidden_dim_per_partition
                ]
            else:
                ref_tp_weight_grad = ref_weight.grad
                ref_tp_bias_grad = ref_bias.grad
            torch.testing.assert_close(
                tp_layernorm_dropout_mul.weight.grad, ref_tp_weight_grad
            )
            torch.testing.assert_close(
                tp_layernorm_dropout_mul.bias.grad, ref_tp_bias_grad
            )
        ref_tp_u_grad = ref_u.grad[
            ...,
            tp_rank
            * hidden_dim_per_partition : (tp_rank + 1)
            * hidden_dim_per_partition,
        ]
        ref_tp_x_grad = ref_x.grad[
            ...,
            tp_rank
            * hidden_dim_per_partition : (tp_rank + 1)
            * hidden_dim_per_partition,
        ]

        torch.testing.assert_close(tp_x.grad, ref_tp_x_grad)
        torch.testing.assert_close(tp_u.grad, ref_tp_u_grad)
    init.destroy_global_state()
