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

from typing import Optional

import commons.utils.initialize as init
import fbgemm_gpu  # pylint: disable-unused-import
import pytest
import torch
from commons.checkpoint import get_unwrapped_module
from commons.utils.distributed_utils import collective_assert, collective_assert_tensor
from commons.utils.hstu_assert_close import hstu_close
from configs.hstu_config import HSTULayerType, KernelBackend
from distributed.finalize_model_grads import finalize_model_grads
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from modules.jagged_data import JaggedData, pad_jd_values, unpad_jd_values
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
    banned_seqlen_divisor: Optional[
        int
    ] = None,  # we deliberately make the sequence length not divisible by banned_seqlen_divisor to test the padding logic
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
            low=2,
            high=max_seqlen + 1,
            size=(batchsize,),
            device=device,
            dtype=torch.int,
        )
        # we don't want the sequence length to be divisible by banned_seqlen_divisor
        # when banned_seqlen_divisor = 1, minus is safe
        if (
            banned_seqlen_divisor is not None
            and lengths.sum() % banned_seqlen_divisor == 0
        ):
            lengths[-1] -= 1
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
        # don't share seqlen either
        ref_jd = jd.copy_others_but_set_values(values=ref_input)
        fp32_ref_jd = jd.copy_others_but_set_values(values=fp32_ref_input)

        ret_list.append(jd)
        ref_ret_list.append(ref_jd)
        fp32_ref_ret_list.append(fp32_ref_jd)

    if replicate_batches:
        ret_list = ret_list * num_batches
        ref_ret_list = ref_ret_list * num_batches
        fp32_ref_ret_list = fp32_ref_ret_list * num_batches

    return ret_list, ref_ret_list, fp32_ref_ret_list


# set backend as PYTORCH
@pytest.mark.parametrize(
    "batchsize",
    [32],
)
@pytest.mark.parametrize("num_heads", [4, 1])
@pytest.mark.parametrize("hidden_dim_per_head", [32, 128])  #
@pytest.mark.parametrize("tp_size", [2, 4, 8, 1])
@pytest.mark.parametrize("optimizer_type_str", ["adam", "sgd"])
@pytest.mark.parametrize("sequence_parallel", [True, False])
def test_tp_hstu_layer_forward_backward_update(
    batchsize,
    num_heads,
    hidden_dim_per_head,
    tp_size,
    optimizer_type_str,
    sequence_parallel: bool,
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
    learnable_input_layernorm = True  # if optimizer_type_str == "sgd" else False
    learnable_output_layernorm = True  # if optimizer_type_str == "sgd" else False
    debug_hstu_layer, debug_dense_optimizer = create_hstu_layer_and_optimizer(
        dtype=dtype,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        kv_channels=hidden_dim_per_head,
        optimizer_type_str=optimizer_type_str,
        hstu_layer_type=HSTULayerType.DEBUG,
        kernel_backend=KernelBackend.PYTORCH,
        learnable_input_layernorm=learnable_input_layernorm,
        learnable_output_layernorm=learnable_output_layernorm,
        sequence_parallel=False,  # debug hstu layer does not support sequence parallel
    )

    tp_hstu_layer, tp_dense_optimizer = create_hstu_layer_and_optimizer(
        dtype=dtype,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        kv_channels=hidden_dim_per_head,
        optimizer_type_str=optimizer_type_str,
        hstu_layer_type=HSTULayerType.NATIVE,  # use native hstu layer for tp
        kernel_backend=KernelBackend.PYTORCH,
        learnable_input_layernorm=learnable_input_layernorm,
        learnable_output_layernorm=learnable_output_layernorm,
        sequence_parallel=sequence_parallel,
    )

    fp32_debug_hstu_layer, fp32_debug_dense_optimizer = create_hstu_layer_and_optimizer(
        dtype=torch.float32,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        kv_channels=hidden_dim_per_head,
        optimizer_type_str=optimizer_type_str,
        hstu_layer_type=HSTULayerType.DEBUG,
        kernel_backend=KernelBackend.PYTORCH,
        learnable_input_layernorm=learnable_input_layernorm,
        learnable_output_layernorm=learnable_output_layernorm,
        sequence_parallel=False,  # debug hstu layer does not support sequence parallel
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
        num_batches=50,
        replicate_batches=False,  # if True.
    )
    fwd_multiplier = 2
    bwd_multiplier = 2
    compare_tpN_to_debug_weights(tp_model, debug_model, debug_model_fp32)
    with init.auto_destroy_global_state():
        for i, (jd, ref_jd, fp32_ref_jd) in enumerate(
            zip(jd_list, ref_jd_list, fp32_ref_jd_list)
        ):
            zero_grad(debug_dense_optimizer, debug_hstu_layer)
            zero_grad(tp_dense_optimizer, tp_hstu_layer)
            zero_grad(fp32_debug_dense_optimizer, fp32_debug_hstu_layer)
            padded_jd = jd
            # when sequence parallel is on, we need to pad the jagged data to the tp size
            # and scatter the values to the sequence parallel region
            if sequence_parallel:
                padded_jd = pad_jd_values(jd, pad_base=tp_size)
                padded_jd.values = scatter_to_sequence_parallel_region(padded_jd.values)
            jagged_tp_out = tp_hstu_layer(padded_jd)
            # when sequence parallel is on, outputs are scatter among TP ranks, so we need to gather the outputs back for check
            if sequence_parallel:
                jagged_tp_out.values = gather_from_sequence_parallel_region(
                    jagged_tp_out.values, False
                )  # False -> output grad not RS but S
                jagged_tp_out = unpad_jd_values(jagged_tp_out)

            logits = debug_hstu_layer(ref_jd).values
            logits_fp32 = fp32_debug_hstu_layer(fp32_ref_jd).values
            tp_logits = jagged_tp_out.values
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
                hstu_close(tp_logits, logits, logits_fp32, multiplier=fwd_multiplier),
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
            # adam update is quite instable at the first step
            # compare the first two iterations when optimizer is sgd
            if i < 2 and optimizer_type_str == "sgd":
                compare_tpN_to_debug_weights(
                    tp_model, debug_model, debug_model_fp32, msg=f"iter {i}"
                )

            grad_debug = ref_jd.values.grad
            grad_tp = jd.values.grad
            grad_fp32_debug = fp32_ref_jd.values.grad
            collective_assert(
                hstu_close(
                    grad_tp, grad_debug, grad_fp32_debug, multiplier=bwd_multiplier
                ),
                f"grads mismatch at iter {i}, diff {(grad_tp - grad_fp32_debug).abs().max()} vs {(grad_debug - grad_fp32_debug).abs().max()} vs hey {(grad_tp - grad_debug).abs().max()}",
            )
            bwd_multiplier = 5 if i >= 2 else fwd_multiplier
