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
from megatron.core import parallel_state
from megatron.core.transformer.module import Float16Module
from modules.fused_hstu_layer import FusedHSTULayer
from modules.jagged_data import JaggedData
from modules.legacy.native_hstu_layer import HSTULayer as LegacyHSTULayer
from modules.native_hstu_layer import HSTULayer
from ops.length_to_offsets import length_to_complete_offsets


def init_fused_weights_from_native_legacy(
    native_module: LegacyHSTULayer, fused_module: FusedHSTULayer
):
    import re

    for name, param in native_module.named_parameters():
        # linear layer weight is transposed in the fused module
        fused_accessor = name.replace(".weight", "_weight").replace(".bias", "_bias")
        src_data = (
            param.data.t()
            if re.match(r".*linear\w*_weight$", fused_accessor)
            else param.data
        )
        if param.requires_grad:
            fused_module.state_dict()[fused_accessor].data.copy_(src_data)


# allgather weights from tp1 to tpN (slice tp1 to tpN)
def init_tpN_weights_from_native_legacy(
    legacy_module: LegacyHSTULayer, tpN_module: HSTULayer
):
    import re

    for name, param in legacy_module.state_dict().items():
        src = param.data
        # col parallel linear weight
        if re.match(r".*_linear_uvqk.weight$", name):
            # TP u,v,q,k layout is different from native
            # (output_size, input_size) => (num_heads, sum(split_arg_list), input_size) => transpose(0, 1) => (sum(split_arg_list), num_heads, input_size) => reshape(-1, input_size)
            # split_arg_list = legacy_module._split_arg_list
            output_size = src.size(0)
            input_size = src.size(1)
            src = (
                src.view(4, legacy_module._num_heads, -1, input_size)
                .transpose(0, 1)
                .reshape(output_size, input_size)
            )
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_slice = tp_rank * output_size // tp_size
            tp_slice_end = (tp_rank + 1) * output_size // tp_size
            src = src[tp_slice:tp_slice_end, :]
        # row wise linear weight
        elif re.match(r".*_linear_proj.weight$", name):
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            input_size = src.size(1)
            tp_slice = tp_rank * input_size // tp_size
            tp_slice_end = (tp_rank + 1) * input_size // tp_size
            src = src[:, tp_slice:tp_slice_end]
        # output layernorm weight and bias are TP split
        # colparallel linear bias is also TP split when config.use_cpu_initialization is True
        # see https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/module/linear.py#L1104, https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/module/linear.py#L1037
        elif re.match(r".*_output_layernorm.*$|.*_linear_uvqk.bias$", name):
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_slice = tp_rank * param.shape[0] // tp_size
            tp_slice_end = (tp_rank + 1) * param.shape[0] // tp_size
            src = src[tp_slice:tp_slice_end, ...]
        tpN_module.state_dict()[name].data.copy_(src)


def get_batch_on_this_tp_rank(batch: JaggedData):
    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )

    _broadcast(batch.values)
    _broadcast(batch.seqlen)
    _broadcast(batch.seqlen_offsets)
    _broadcast(batch.max_seqlen)
    _broadcast(batch.num_candidates)
    _broadcast(batch.num_candidates_offsets)
    _broadcast(batch.contextual_seqlen)
    _broadcast(batch.contextual_seqlen_offsets)
    return batch


@pytest.mark.parametrize(
    "batchsize",
    [
        2,
    ],
)
@pytest.mark.parametrize("num_heads", [4, 8, 1])
@pytest.mark.parametrize("hidden_dim_per_head", [32, 64, 128])  #
@pytest.mark.parametrize("tp_size", [2])
def test_tp_hstu_layer(
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

    def generate_input():
        input_sparsity = 0.75
        max_history_seqlen = 5
        max_num_targets = 2
        max_num_contextuals = 2
        device = torch.cuda.current_device()
        max_seqlen = max_history_seqlen + max_num_targets + max_num_contextuals
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
            "num_candidates_offsets": length_to_complete_offsets(num_targets),
            "contextual_max_seqlen": max_num_contextuals,
            "contextual_seqlen": num_contextuals,
            "contextual_seqlen_offsets": length_to_complete_offsets(num_contextuals),
        }
        jd = JaggedData(values=input, **ctor_nograd_dict)
        ref_jd = JaggedData(values=ref_input, **ctor_nograd_dict)
        fp32_ref_jd = JaggedData(values=fp32_ref_input, **ctor_nograd_dict)
        return jd, ref_jd, fp32_ref_jd

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
        hstu_layer_type=HSTULayerType.NATIVE,
        learnable_input_layernorm=True,
        residual=True,
    )
    torch.cuda.current_device()

    legacy_hstu_layer = LegacyHSTULayer(hstu_config).cuda()
    tp_hstu_layer = HSTULayer(hstu_config).cuda()

    legacy_hstu_layer = Float16Module(hstu_config, legacy_hstu_layer)
    tp_hstu_layer = Float16Module(hstu_config, tp_hstu_layer)
    hstu_config.kernel_backend = KernelBackend.PYTORCH
    fp32_legacy_hstu_layer = LegacyHSTULayer(hstu_config).cuda()

    init_tpN_weights_from_native_legacy(legacy_hstu_layer.module, tp_hstu_layer.module)
    jd, ref_jd, fp32_ref_jd = generate_input()

    out_legacy = legacy_hstu_layer(ref_jd).values
    fp32_out_legacy = fp32_legacy_hstu_layer(fp32_ref_jd).values
    tp_out = tp_hstu_layer(jd).values
    # torch.testing.assert_close(tp_out, out_legacy)
    assert_hstu_close(tp_out, out_legacy, fp32_out_legacy, fwd=True)

    # make the grad_output sparse
    with torch.no_grad():
        dout = torch.ones_like(out_legacy) / (2**8)
        dout = torch.nn.functional.dropout(dout, p=0.7, training=True)
    out_legacy.backward(dout)
    tp_out.backward(dout)
    fp32_out_legacy.backward(dout.float())
    grad_legacy = ref_jd.values.grad
    grad_tp = jd.values.grad
    grad_fp32_legacy = fp32_ref_jd.values.grad
    # torch.testing.assert_close(grad_tp, grad_legacy)
    assert_hstu_close(grad_tp, grad_legacy, grad_fp32_legacy, fwd=False)


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
        hstu_layer_type=HSTULayerType.NATIVE,
        learnable_input_layernorm=learnable_ln,
        residual=residual,
        async_wgrad=async_wgrad,
    )
    # hstu_config.kernel_backend = KernelBackend.PYTORCH
    ref_hstu_layer = LegacyHSTULayer(hstu_config)
    # to create fused hstu layer
    hstu_config.hstu_layer_type = HSTULayerType.FUSED

    fused_hstu_layer = FusedHSTULayer(hstu_config)
    fused_hstu_layer.cuda()
    ref_hstu_layer.cuda()

    hstu_config.kernel_backend = KernelBackend.PYTORCH
    hstu_config.dtype = torch.float32
    hstu_config.hstu_layer_type = HSTULayerType.NATIVE
    fp32_ref_hstu_layer = LegacyHSTULayer(hstu_config)

    fp32_ref_hstu_layer.cuda()
    fp32_ref_hstu_layer.load_state_dict(ref_hstu_layer.state_dict())

    init_fused_weights_from_native_legacy(
        native_module=ref_hstu_layer, fused_module=fused_hstu_layer
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
