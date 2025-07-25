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


import commons.utils.initialize as init
import fbgemm_gpu  # pylint: disable-unused-import
import pytest
import torch
from commons.utils.hstu_assert_close import assert_hstu_close
from configs import get_hstu_config
from configs.hstu_config import HSTULayerType, KernelBackend
from megatron.core.transformer.module import Float16Module
from modules.debug.debug_hstu_layer import HSTULayer as DebugHSTULayer
from modules.fused_hstu_layer import FusedHSTULayer
from modules.jagged_data import JaggedData
from ops.length_to_offsets import length_to_complete_offsets
from test_utils import init_fused_weights_from_debug


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        #    torch.float16
    ],
)
@pytest.mark.parametrize("batchsize", [2])
@pytest.mark.parametrize("max_history_seqlen", [128, 200])
@pytest.mark.parametrize("max_num_targets", [16])
@pytest.mark.parametrize("max_num_contextuals", [2, 0])
@pytest.mark.parametrize("num_heads", [8, 1])
@pytest.mark.parametrize("hidden_dim_per_head", [64, 128])
@pytest.mark.parametrize("dropout_ratio", [0.0])
@pytest.mark.parametrize("attn_backend", [KernelBackend.CUTLASS])
@pytest.mark.parametrize("target_group_size", [1])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("learnable_ln", [True])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("input_sparsity", [0.75])
@pytest.mark.parametrize("async_wgrad", [True, False])
def test_fused_hstu_layer(
    dtype: torch.dtype,
    batchsize: int,
    max_history_seqlen: int,  # N
    max_num_targets: int,
    max_num_contextuals: int,
    num_heads: int,
    hidden_dim_per_head: int,
    dropout_ratio: float,
    attn_backend: KernelBackend,
    target_group_size: int,
    causal: bool,
    learnable_ln: bool,
    residual: bool,
    input_sparsity: float,
    async_wgrad: bool,
):
    init.initialize_distributed()
    init.set_random_seed(1234)
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        return
    device = torch.cuda.current_device()
    ln_eps = 1e-5
    hstu_config = get_hstu_config(
        hidden_size=hidden_dim_per_head * num_heads,
        kv_channels=hidden_dim_per_head,
        num_attention_heads=num_heads,
        num_layers=1,
        dtype=dtype,
        hidden_dropout=dropout_ratio,
        norm_epsilon=ln_eps,
        is_causal=causal,
        kernel_backend=attn_backend,  # attn_backend
        target_group_size=target_group_size,
        hstu_layer_type=HSTULayerType.DEBUG,
        learnable_input_layernorm=learnable_ln,
        residual=residual,
        async_wgrad=async_wgrad,
    )
    # hstu_config.kernel_backend = KernelBackend.PYTORCH
    ref_hstu_layer = DebugHSTULayer(hstu_config)
    # to create fused hstu layer
    hstu_config.hstu_layer_type = HSTULayerType.FUSED

    fused_hstu_layer = FusedHSTULayer(hstu_config)
    fused_hstu_layer.cuda()
    ref_hstu_layer.cuda()

    hstu_config.kernel_backend = KernelBackend.PYTORCH
    hstu_config.dtype = torch.float32
    hstu_config.hstu_layer_type = HSTULayerType.DEBUG
    fp32_ref_hstu_layer = DebugHSTULayer(hstu_config)

    fp32_ref_hstu_layer.cuda()
    fp32_ref_hstu_layer.load_state_dict(ref_hstu_layer.state_dict())

    init_fused_weights_from_debug(
        debug_module=ref_hstu_layer, fused_module=fused_hstu_layer, num_heads=num_heads
    )
    if dtype != torch.float32:
        ref_hstu_layer = Float16Module(hstu_config, ref_hstu_layer)
        fused_hstu_layer = Float16Module(hstu_config, fused_hstu_layer)
    ref_hstu_layer.cuda()

    # generate input
    # TODO: this is not exact, but should be close
    max_seqlen = max_history_seqlen + max_num_targets + max_num_contextuals
    lengths = torch.randint(
        low=1, high=max_seqlen + 1, size=(batchsize,), device=device, dtype=torch.int
    )

    seq_offsets = length_to_complete_offsets(lengths)

    L = int(seq_offsets[-1].item())

    if attn_backend == KernelBackend.TRITON and max_num_contextuals > 0:
        pytest.skip("TRITON does not support contextuals")

    if attn_backend == KernelBackend.TRITON and target_group_size > 1:
        pytest.skip("TRITON does not support target grouped attention")

    num_targets = None
    num_contextuals = None

    if max_num_targets > 0:
        num_targets = torch.randint(
            low=0,
            high=max_num_targets + 1,
            size=(batchsize,),
            device=device,
            dtype=torch.int32,
        )
        num_targets = torch.clamp(
            num_targets, max=lengths - 1, min=torch.zeros_like(num_targets)
        )  # at least 1 history

    if max_num_contextuals > 0:
        num_contextuals = torch.randint(
            low=0,
            high=max_num_contextuals + 1,
            size=(batchsize,),
            device=device,
            dtype=torch.int32,
        )
        num_contextuals = torch.clamp(
            num_contextuals,
            max=lengths - 1 - num_targets if num_targets is not None else lengths - 1,
            min=torch.zeros_like(num_contextuals),
        )  # at least 1 history!!

    input = torch.empty(
        (L, hidden_dim_per_head * num_heads),
        dtype=dtype,
        device=device,
    ).uniform_(-0.1, 0.1)
    # sparse the input
    with torch.no_grad():
        input = torch.nn.functional.dropout(input, p=input_sparsity, training=True)
    input.requires_grad_()
    ref_input = input.detach().clone().requires_grad_()
    fp32_ref_input = input.float().detach().clone().requires_grad_()

    ctor_nograd_dict = {
        "seqlen": lengths,
        "seqlen_offsets": seq_offsets,
        "max_seqlen": max_seqlen,
        "max_num_candidates": max_num_targets,
        "num_candidates": num_targets,
        "num_candidates_offsets": length_to_complete_offsets(num_targets)
        if num_targets is not None
        else None,
        "contextual_max_seqlen": max_num_contextuals,
        "contextual_seqlen": num_contextuals,
        "contextual_seqlen_offsets": length_to_complete_offsets(num_contextuals)
        if num_contextuals is not None
        else None,
    }
    jd = JaggedData(values=input, **ctor_nograd_dict)
    ref_jd = JaggedData(values=ref_input, **ctor_nograd_dict)
    fp32_ref_jd = JaggedData(values=fp32_ref_input, **ctor_nograd_dict)

    out_native = ref_hstu_layer(ref_jd).values
    out_fused = fused_hstu_layer(jd).values
    fp32_ref_out_native = fp32_ref_hstu_layer(fp32_ref_jd).values

    assert_hstu_close(out_fused, out_native, fp32_ref_out_native, fwd=True)

    # make the grad_output sparse
    with torch.no_grad():
        dout = torch.ones_like(out_native) / (2**8)
        dout = torch.nn.functional.dropout(dout, p=input_sparsity, training=True)

    # dropout
    out_native.backward(dout)
    out_fused.backward(dout)
    fp32_ref_out_native.backward(dout.float())
    grad_native = ref_input.grad
    fp32_grad_ref_native = fp32_ref_input.grad
    grad_fused = input.grad

    assert_hstu_close(grad_fused, grad_native, fp32_grad_ref_native, fwd=False)
