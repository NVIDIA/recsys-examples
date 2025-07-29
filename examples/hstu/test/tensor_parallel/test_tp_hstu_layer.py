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
from commons.checkpoint import get_unwrapped_module
from commons.utils.distributed_utils import collective_assert, collective_assert_tensor
from commons.utils.hstu_assert_close import assert_hstu_close, hstu_close
from configs import get_hstu_config
from configs.hstu_config import HSTULayerType, KernelBackend
from distributed.finalize_model_grads import finalize_model_grads
from megatron.core import parallel_state
from megatron.core.transformer.module import Float16Module
from modules.debug.debug_hstu_layer import HSTULayer as DebugHSTULayer
from modules.jagged_data import JaggedData
from modules.native_hstu_layer import HSTULayer
from ops.length_to_offsets import length_to_complete_offsets
from test_utils import (
    compare_tpN_to_debug_weights,
    create_hstu_layer_and_optimizer,
    init_module_from,
    init_tpN_weights_from_debug,
)


def generate_jagged_data_list(
    batchsize: int,
    dtype: torch.dtype,
    hidden_dim_per_head: int,
    num_heads: int,
    num_batches: int,
    replicate_batches: bool = True,
):
    max_history_seqlen = 100
    max_num_targets = 50
    max_num_contextuals = 2
    device = torch.cuda.current_device()
    max_seqlen = max_history_seqlen + max_num_targets + max_num_contextuals
    ret_list = []
    ref_ret_list = []
    fp32_ref_ret_list = []
    random_batches = 1 if replicate_batches else num_batches
    for i in range(random_batches):
        lengths = torch.randint(
            low=1,
            high=max_seqlen + 1,
            size=(batchsize,),
            device=device,
            dtype=torch.int,
        )
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
        lengths = torch.randint(
            low=1,
            high=max_seqlen + 1,
            size=(batchsize,),
            device=device,
            dtype=torch.int,
        )
        seq_offsets = length_to_complete_offsets(lengths)
        L = int(seq_offsets[-1].item())

        input = torch.empty(
            (L, hidden_dim_per_head * num_heads),
            dtype=dtype,
            device=device,
        ).uniform_(-0.1, 0.1)
        input.requires_grad_()
        ref_input = input.detach().clone().requires_grad_()
        fp32_ref_input = input.float().detach().clone().requires_grad_()

        ctor_nograd_dict = {
            "seqlen": lengths,
            "seqlen_offsets": seq_offsets,
            "max_seqlen": max_seqlen,
            "max_num_candidates": max_num_targets,
            "num_candidates": num_targets,
            "num_candidates_offsets": length_to_complete_offsets(num_targets),
            "contextual_max_seqlen": max_num_contextuals,
            "contextual_seqlen": num_contextuals,
            "contextual_seqlen_offsets": length_to_complete_offsets(num_contextuals),
        }
        jd = JaggedData(values=input, **ctor_nograd_dict)
        ref_jd = JaggedData(values=ref_input, **ctor_nograd_dict)
        fp32_ref_jd = JaggedData(values=fp32_ref_input, **ctor_nograd_dict)
        ret_list.append(jd)
        ref_ret_list.append(ref_jd)
        fp32_ref_ret_list.append(fp32_ref_jd)

    if replicate_batches:
        ret_list = ret_list * num_batches
        ref_ret_list = ref_ret_list * num_batches
        fp32_ref_ret_list = fp32_ref_ret_list * num_batches

    return ret_list, ref_ret_list, fp32_ref_ret_list


@pytest.mark.parametrize(
    "batchsize",
    [
        32,
    ],
)
@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("hidden_dim_per_head", [32, 128])  #
@pytest.mark.parametrize("tp_size", [2, 4, 8, 1])
def test_tp_hstu_layer_forward_backward(
    batchsize,
    num_heads,
    hidden_dim_per_head,
    tp_size,
):
    init.initialize_distributed()
    world_size = torch.distributed.get_world_size()

    if world_size < tp_size:
        pytest.skip("TP size is larger than world size")
    if num_heads % tp_size != 0:
        pytest.skip("num_heads should be divisible by tp_size")
    init.initialize_model_parallel(tp_size)
    init.set_random_seed(1234)

    ln_eps = 1e-5
    attn_backend = KernelBackend.CUTLASS
    dropout_ratio = 0.0  # triton dropout is not consistent with torch.nn.dropout
    dtype = torch.bfloat16
    hidden_size = hidden_dim_per_head * num_heads
    hstu_config = get_hstu_config(
        hidden_size=hidden_size,
        kv_channels=hidden_dim_per_head,
        num_attention_heads=num_heads,
        num_layers=1,
        dtype=dtype,
        hidden_dropout=dropout_ratio,
        norm_epsilon=ln_eps,
        is_causal=True,
        kernel_backend=attn_backend,  # attn_backend
        target_group_size=1,
        hstu_layer_type=HSTULayerType.DEBUG,
        learnable_input_layernorm=True,
        residual=True,
    )
    torch.cuda.current_device()

    debug_hstu_layer = DebugHSTULayer(hstu_config).cuda()
    debug_hstu_layer = Float16Module(hstu_config, debug_hstu_layer)

    hstu_config.hstu_layer_type = HSTULayerType.NATIVE
    tp_hstu_layer = HSTULayer(hstu_config).cuda()
    tp_hstu_layer = Float16Module(hstu_config, tp_hstu_layer)

    hstu_config.hstu_layer_type = HSTULayerType.DEBUG
    hstu_config.kernel_backend = KernelBackend.PYTORCH
    fp32_debug_hstu_layer = DebugHSTULayer(hstu_config).cuda()

    init_tpN_weights_from_debug(debug_hstu_layer.module, tp_hstu_layer.module)
    jd_list, ref_jd_list, fp32_ref_jd_list = generate_jagged_data_list(
        batchsize, dtype, hidden_dim_per_head, num_heads, 100, False
    )
    for i, (jd, ref_jd, fp32_ref_jd) in enumerate(
        zip(jd_list, ref_jd_list, fp32_ref_jd_list)
    ):
        out_legacy = debug_hstu_layer(ref_jd).values
        fp32_out_legacy = fp32_debug_hstu_layer(fp32_ref_jd).values
        tp_out = tp_hstu_layer(jd).values

        assert_hstu_close(tp_out, out_legacy, fp32_out_legacy, fwd=True)
        # use normal distribution
        dout = torch.empty_like(out_legacy)
        dout.normal_() / 2**2
        out_legacy.backward(dout)
        tp_out.backward(dout)
        fp32_out_legacy.backward(dout.float())
        grad_legacy = ref_jd.values.grad
        grad_tp = jd.values.grad
        grad_fp32_legacy = fp32_ref_jd.values.grad
        assert_hstu_close(grad_tp, grad_legacy, grad_fp32_legacy, fwd=False)
    init.destroy_global_state()


@pytest.mark.parametrize(
    "batchsize",
    [128],
)
@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("hidden_dim_per_head", [128])  #
@pytest.mark.parametrize("tp_size", [2, 4, 8, 1])
@pytest.mark.parametrize("optimizer_type_str", ["adam", "sgd"])
def test_tp_hstu_layer_forward_backward_update(
    batchsize,
    num_heads,
    hidden_dim_per_head,
    tp_size,
    optimizer_type_str,
):
    init.initialize_distributed()
    world_size = torch.distributed.get_world_size()

    if world_size < tp_size:
        pytest.skip("TP size is larger than world size")
    if num_heads % tp_size != 0:
        pytest.skip("num_heads should be divisible by tp_size")
    init.initialize_model_parallel(tp_size)
    init.set_random_seed(1234)
    dtype = torch.bfloat16
    hidden_size = hidden_dim_per_head * num_heads
    torch.cuda.current_device()

    debug_hstu_layer, debug_dense_optimizer = create_hstu_layer_and_optimizer(
        dtype=dtype,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        kv_channels=hidden_dim_per_head,
        optimizer_type_str=optimizer_type_str,
        hstu_layer_type=HSTULayerType.DEBUG,
        kernel_backend=KernelBackend.PYTORCH,
        learnable_input_layernorm=True,
    )

    tp_hstu_layer, tp_dense_optimizer = create_hstu_layer_and_optimizer(
        dtype=dtype,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        kv_channels=hidden_dim_per_head,
        optimizer_type_str=optimizer_type_str,
        hstu_layer_type=HSTULayerType.NATIVE,  # use native hstu layer for tp
        kernel_backend=KernelBackend.PYTORCH,
        learnable_input_layernorm=True,
    )

    fp32_debug_hstu_layer, fp32_debug_dense_optimizer = create_hstu_layer_and_optimizer(
        dtype=torch.float32,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        kv_channels=hidden_dim_per_head,
        optimizer_type_str=optimizer_type_str,
        hstu_layer_type=HSTULayerType.DEBUG,
        kernel_backend=KernelBackend.PYTORCH,
        learnable_input_layernorm=True,
    )

    init_module_from(fp32_debug_hstu_layer, debug_hstu_layer)
    init_tpN_weights_from_debug(debug_hstu_layer, tp_hstu_layer)
    tp_dense_optimizer.reload_model_params()
    debug_dense_optimizer.reload_model_params()
    fp32_debug_dense_optimizer.reload_model_params()

    def zero_grad(optimizer, model):
        if hasattr(model.module, "zero_grad_buffer"):
            model.module.zero_grad_buffer()
        optimizer.zero_grad()

    def optimizer_step(optimizer, model):
        finalize_model_grads([model], None)
        optimizer.step()

    tp_model = get_unwrapped_module(tp_hstu_layer)
    debug_model = get_unwrapped_module(debug_hstu_layer)
    debug_model_fp32 = get_unwrapped_module(fp32_debug_hstu_layer)

    jd_list, ref_jd_list, fp32_ref_jd_list = generate_jagged_data_list(
        batchsize,
        dtype,
        hidden_dim_per_head,
        num_heads,
        num_batches=100,
        replicate_batches=False,
    )
    multiplier = 2
    for i, (jd, ref_jd, fp32_ref_jd) in enumerate(
        zip(jd_list, ref_jd_list, fp32_ref_jd_list)
    ):
        zero_grad(debug_dense_optimizer, debug_hstu_layer)
        zero_grad(tp_dense_optimizer, tp_hstu_layer)
        zero_grad(fp32_debug_dense_optimizer, fp32_debug_hstu_layer)

        logits = debug_hstu_layer(ref_jd).values
        logits_fp32 = fp32_debug_hstu_layer(fp32_ref_jd).values
        tp_logits = tp_hstu_layer(jd).values

        collective_assert_tensor(
            logits_fp32,
            compare_type="equal",
            pg=parallel_state.get_tensor_model_parallel_group(),
        )
        collective_assert_tensor(
            logits,
            compare_type="equal",
            pg=parallel_state.get_tensor_model_parallel_group(),
        )
        collective_assert_tensor(
            tp_logits,
            compare_type="equal",
            pg=parallel_state.get_tensor_model_parallel_group(),
        )

        collective_assert(
            hstu_close(tp_logits, logits, logits_fp32, multiplier=multiplier),
            f"logits mismatch at iter {i}, diff {(tp_logits - logits_fp32).abs().max()} vs {(logits - logits_fp32).abs().max()} vs hey {(tp_logits - logits).abs().max()}",
        )
        # use normal distribution
        dout = torch.empty_like(tp_logits)
        dout.normal_() / 2**2
        logits.backward(dout)
        tp_logits.backward(dout)
        logits_fp32.backward(dout.float())

        # optimizer step
        optimizer_step(tp_dense_optimizer, tp_hstu_layer)
        optimizer_step(debug_dense_optimizer, debug_hstu_layer)
        optimizer_step(fp32_debug_dense_optimizer, fp32_debug_hstu_layer)
        compare_tpN_to_debug_weights(tp_model, debug_model, debug_model_fp32)

        grad_legacy = ref_jd.values.grad
        grad_tp = jd.values.grad
        grad_fp32_legacy = fp32_ref_jd.values.grad
        collective_assert(
            hstu_close(grad_tp, grad_legacy, grad_fp32_legacy, multiplier=multiplier),
            f"grads mismatch at iter {i}, diff {(grad_tp - grad_fp32_legacy).abs().max()} vs {(grad_legacy - grad_fp32_legacy).abs().max()} vs hey {(grad_tp - grad_legacy).abs().max()}",
        )
        print(
            f"[tp{parallel_state.get_tensor_model_parallel_rank()}, dp{parallel_state.get_data_parallel_rank()}] [iter {i} is good]"
        )
        multiplier = 2 if i == 0 else multiplier
    init.destroy_global_state()
