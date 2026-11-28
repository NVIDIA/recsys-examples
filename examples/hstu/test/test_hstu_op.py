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
from collections import OrderedDict
from typing import Optional

import commons.utils.initialize as init
import fbgemm_gpu  # pylint: disable-unused-import
import pytest
import torch
from commons.ops.length_to_offsets import length_to_complete_offsets
from commons.utils.hstu_assert_close import assert_hstu_close
from configs import get_hstu_config
from configs.hstu_config import HSTULayerType, KernelBackend
from megatron.core.transformer.module import Float16Module
from modules.debug.debug_hstu_layer import HSTULayer
from modules.hstu_attention import create_hstu_attention
from modules.jagged_data import JaggedData
from ops.fused_hstu_op import fused_hstu_op


def generate_or_copy_parameters(
    embedding_dim: int,
    num_heads: int,
    hidden_dim_per_head: int,
    dtype: torch.dtype,
    device: torch.device,
    input_module: Optional[torch.nn.Module] = None,
    learnable_ln: bool = False,
):
    if input_module is None:
        linear_uvqk_weight = torch.nn.Parameter(
            torch.empty(
                (embedding_dim, hidden_dim_per_head * 4 * num_heads),
                dtype=dtype,
                device=device,
            )
        )
        torch.nn.init.xavier_uniform_(linear_uvqk_weight)
        linear_uvqk_bias = torch.nn.Parameter(
            torch.empty(
                (hidden_dim_per_head * 4 * num_heads,),
                dtype=dtype,
                device=device,
            ).uniform_(-0.01, 0.01)
        )
        linear_proj_weight = torch.nn.Parameter(
            torch.empty(
                (hidden_dim_per_head * num_heads, embedding_dim),
                dtype=dtype,
                device=device,
            ).uniform_(-0.1, 0.1)
        )
        torch.nn.init.xavier_uniform_(linear_proj_weight)
        if learnable_ln:
            input_norm_weight = torch.nn.Parameter(
                torch.empty(
                    (embedding_dim,),
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
            )
            input_norm_bias = torch.nn.Parameter(
                torch.zeros(
                    (embedding_dim,),
                    dtype=dtype,
                    device=device,
                )
            ).uniform_(-0.01, 0.01)
            output_norm_weight = torch.nn.Parameter(
                torch.empty(
                    (embedding_dim,),
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
            )
            output_norm_bias = torch.nn.Parameter(
                torch.zeros(
                    (embedding_dim,),
                    dtype=dtype,
                    device=device,
                ).uniform_(-0.01, 0.01)
            )
        else:
            input_norm_weight = None
            input_norm_bias = None
            # the output norm weight and bias are mandatory
            output_norm_weight = torch.ones(embedding_dim, device=device).to(dtype)
            output_norm_bias = torch.zeros(embedding_dim, device=device).to(dtype)

    else:
        with torch.no_grad():
            input_norm_weight = (
                torch.empty_like(input_module._input_layernorm_weight.data)
                .copy_(input_module._input_layernorm_weight.data)
                .to(dtype)
                .requires_grad_()
                if learnable_ln
                else None
            )
            input_norm_bias = (
                torch.empty_like(input_module._input_layernorm_bias.data)
                .copy_(input_module._input_layernorm_bias.data)
                .to(dtype)
                .requires_grad_()
                if learnable_ln
                else None
            )
            output_norm_weight = (
                torch.empty_like(input_module._output_layernorm_weight.data)
                .copy_(input_module._output_layernorm_weight.data)
                .to(dtype)
                .requires_grad_()
            )
            output_norm_bias = (
                torch.empty_like(input_module._output_layernorm_bias.data)
                .copy_(input_module._output_layernorm_bias.data)
                .to(dtype)
                .requires_grad_()
            )
            src_linear_uvqk_weight = input_module._linear_uvqk.weight.data
            input_size = src_linear_uvqk_weight.size(1)
            output_size = src_linear_uvqk_weight.size(0)
            linear_uvqk_weight = (
                torch.empty_like(src_linear_uvqk_weight)
                .copy_(src_linear_uvqk_weight)
                .reshape(num_heads, 4, -1, input_size)
                .transpose(0, 1)
                .reshape(output_size, input_size)
                .t()
                .to(dtype)
                .requires_grad_()
            )
            linear_uvqk_bias = (
                torch.empty_like(input_module._linear_uvqk.bias.data)
                .copy_(input_module._linear_uvqk.bias.data)
                .to(dtype)
                .requires_grad_()
            )
            linear_proj_weight = (
                torch.empty_like(input_module._linear_proj.weight.data)
                .copy_(input_module._linear_proj.weight.data)
                .t()
                .to(dtype)
                .requires_grad_()
            )
    return (
        input_norm_weight,
        input_norm_bias,
        output_norm_weight,
        output_norm_bias,
        linear_uvqk_weight,
        linear_uvqk_bias,
        linear_proj_weight,
    )


@pytest.mark.parametrize("heads", [8])
@pytest.mark.parametrize("hidden_dim", [128])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        #    torch.float16
    ],
)
@pytest.mark.parametrize(
    "max_seqlen,max_num_candidates,max_num_contextuals,scaling_seqlen",
    [
        (1024, 128, 6, -1),
        (1024, 128, 6, 2048),
        (32, 0, 0, -1),
        (32, 0, 0, 1024),
        (1024, 128, 0, -1),
        (1024, 128, 0, 512),
    ],
)
@pytest.mark.parametrize("target_group_size", [2, 16, 256, 1])
@pytest.mark.parametrize("batchsize", [32])
@pytest.mark.parametrize(
    "is_causal,kernel_backend",
    [
        (True, KernelBackend.CUTLASS),
        (False, KernelBackend.CUTLASS),
    ],
)
@pytest.mark.parametrize("from_scratch", [True, False])
def test_hstu_attn(
    heads,
    hidden_dim,
    is_causal,
    dtype,
    max_seqlen,
    scaling_seqlen,
    max_num_candidates,
    max_num_contextuals,
    target_group_size,
    batchsize,
    kernel_backend,
    from_scratch,
):
    if kernel_backend == KernelBackend.TRITON and target_group_size > 1:
        pytest.skip("Triton is not supported when target_group_size > 1")
    if kernel_backend == KernelBackend.TRITON and max_num_contextuals > 0:
        pytest.skip("Triton is not supported when max_num_contextuals > 0")
    # TODO: uncomment this once cutlass supports causal attention
    if not is_causal and max_num_contextuals > 0:
        pytest.skip("Only causal attention is supported when max_num_contextuals > 0")

    init.initialize_distributed()
    init.set_random_seed(1234)
    device = torch.cuda.current_device()
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        return
    if not is_causal:
        max_num_candidates = 0

    ref_hstu_attn = create_hstu_attention(
        KernelBackend.PYTORCH,
        num_heads=heads,
        attention_dim=hidden_dim,
        linear_dim=hidden_dim,
        is_causal=is_causal,
    )
    hstu_attn = create_hstu_attention(
        kernel_backend,
        num_heads=heads,
        attention_dim=hidden_dim,
        linear_dim=hidden_dim,
        is_causal=is_causal,
    )
    linear_uvqk_weight = torch.nn.Parameter(
        torch.empty(
            (hidden_dim * heads, hidden_dim * 4 * heads),
            dtype=dtype,
            device=device,
        )
    )
    torch.nn.init.xavier_uniform_(linear_uvqk_weight)
    linear_uvqk_weight = torch.nn.functional.dropout(
        linear_uvqk_weight, p=0.8, training=True
    )
    linear_uvqk_bias = torch.nn.Parameter(
        torch.empty(
            (hidden_dim * 4 * heads,),
            dtype=dtype,
            device=device,
        ).uniform_(-0.01, 0.01)
    )

    def init_input(input):
        normad_input = torch.nn.functional.layer_norm(
            input, (hidden_dim * heads,), eps=1e-5
        )
        linear = normad_input.matmul(linear_uvqk_weight) + linear_uvqk_bias
        silu = torch.nn.functional.silu(linear)
        return silu

    def generate_input(from_scratch: bool = True):
        if from_scratch:
            input = (
                torch.empty(
                    (L, hidden_dim * heads),
                    dtype=dtype,
                    device=device,
                )
                .uniform_(-0.01, 0.01)
                .requires_grad_()
            )
            x = init_input(input)
            u, q, k, v = torch.split(
                x,
                [
                    hidden_dim * heads,
                    hidden_dim * heads,
                    hidden_dim * heads,
                    hidden_dim * heads,
                ],
                dim=-1,
            )
            q = q.detach().clone().requires_grad_()
            k = k.detach().clone().requires_grad_()
            v = v.detach().clone().requires_grad_()
        else:
            x = torch.empty(
                (L, heads, hidden_dim * 3),
                dtype=dtype,
                device=torch.device("cuda"),
            ).uniform_(-0.1, 0.1)

            q, k, v = torch.split(x, [hidden_dim, hidden_dim, hidden_dim], dim=-1)
            q = q.requires_grad_(True)
            k = k.requires_grad_(True)
            v = v.requires_grad_(True)

        return q, k, v

    if scaling_seqlen == -1:
        scaling_seqlen = max_seqlen
    for _ in range(100):
        lengths = torch.randint(
            1, max_seqlen + 1, (batchsize,), device=device, dtype=torch.int
        )
        seq_offsets = length_to_complete_offsets(lengths)
        L = int(seq_offsets[-1].item())
        if max_num_candidates == 0:
            num_candidates = None
        else:
            num_candidates = torch.randint(
                0, max_num_candidates + 1, (batchsize,), device=device
            )
            num_candidates = torch.clamp(
                num_candidates, max=lengths - 1, min=torch.zeros_like(num_candidates)
            )  # at least 1 history
        if max_num_contextuals == 0:
            num_contextuals = None
        else:
            num_contextuals = torch.randint(
                0, max_num_contextuals + 1, (batchsize,), device=device, dtype=torch.int
            )
            num_contextuals = torch.clamp(
                num_contextuals,
                max=lengths - 1 - num_candidates
                if num_candidates is not None
                else lengths - 1,
                min=torch.zeros_like(num_contextuals),
            )  # at least 1 history!!

        q, k, v = generate_input(from_scratch)
        q_fp32, k_fp32, v_fp32 = (
            q.detach().clone().float().requires_grad_(),
            k.detach().clone().float().requires_grad_(),
            v.detach().clone().float().requires_grad_(),
        )

        ref_out = ref_hstu_attn(
            q,
            k,
            v,
            seq_offsets,
            num_candidates=num_candidates,
            num_contextuals=num_contextuals,
            max_seqlen=max_seqlen,
            scaling_seqlen=scaling_seqlen,
            target_group_size=target_group_size,
        )
        ref_out_fp32 = ref_hstu_attn(
            q_fp32,
            k_fp32,
            v_fp32,
            seq_offsets,
            num_candidates=num_candidates,
            num_contextuals=num_contextuals,
            max_seqlen=max_seqlen,
            scaling_seqlen=scaling_seqlen,
            target_group_size=target_group_size,
        )
        dout = torch.randn_like(ref_out) * 0.01
        ref_out.backward(dout)
        ref_out_fp32.backward(dout.float())
        ref_dq = q.grad.clone()
        ref_dk = k.grad.clone()
        ref_dv = v.grad.clone()

        q = q.detach().clone().requires_grad_()
        k = k.detach().clone().requires_grad_()
        v = v.detach().clone().requires_grad_()
        dout = dout.detach().clone()
        out = hstu_attn(
            q,
            k,
            v,
            seq_offsets,
            num_candidates=num_candidates,
            num_contextuals=num_contextuals,
            max_seqlen=max_seqlen,
            scaling_seqlen=scaling_seqlen,
            target_group_size=target_group_size,
        )
        out.backward(dout)

        assert_hstu_close(out, ref_out, ref_out_fp32, fwd=True)
        assert_hstu_close(q.grad, ref_dq, q_fp32.grad, fwd=False)
        assert_hstu_close(k.grad, ref_dk, k_fp32.grad, fwd=False)
        assert_hstu_close(v.grad, ref_dv, v_fp32.grad, fwd=False)
    init.destroy_global_state()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        #    torch.float16
    ],
)
@pytest.mark.parametrize("batchsize", [2])
@pytest.mark.parametrize("max_history_seqlen", [32, 200])
@pytest.mark.parametrize("max_num_targets", [4, 0])
@pytest.mark.parametrize("max_num_contextuals", [0, 4, 6])
@pytest.mark.parametrize("num_heads", [2, 8, 1])
@pytest.mark.parametrize("hidden_dim_per_head", [128])
@pytest.mark.parametrize("dropout_ratio", [0.0])
@pytest.mark.parametrize("training", [True])
@pytest.mark.parametrize("attn_backend", [KernelBackend.CUTLASS])
@pytest.mark.parametrize("target_group_size", [1, 4])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("seed", [None])
@pytest.mark.parametrize("learnable_ln", [True])
@pytest.mark.parametrize("upcast_reference", [False])
@pytest.mark.parametrize("residual", [False])
@pytest.mark.parametrize("async_wgrad", [True, False])
@pytest.mark.parametrize("recompute_input_layernorm", [True, False])
@pytest.mark.parametrize("recompute_input_silu", [True, False])
@pytest.mark.parametrize("only_targets", [True, False])
@pytest.mark.parametrize("scaling_seqlen", [-1, 200, 512])
def test_fused_hstu_op(
    dtype: torch.dtype,
    batchsize: int,
    max_history_seqlen: int,  # N
    max_num_targets: int,
    max_num_contextuals: int,
    num_heads: int,
    hidden_dim_per_head: int,
    dropout_ratio: float,
    training: bool,
    attn_backend: KernelBackend,
    target_group_size: int,
    causal: bool,
    seed: Optional[int],
    learnable_ln: bool,
    upcast_reference: bool,
    residual: bool,
    async_wgrad: bool,
    recompute_input_layernorm: bool,
    recompute_input_silu: bool,
    only_targets: bool,
    scaling_seqlen: int,
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
    ref_hstu_layer = HSTULayer(hstu_config)

    hstu_config.kernel_backend = KernelBackend.PYTORCH
    hstu_config.dtype = torch.float32
    fp32_ref_hstu_layer = HSTULayer(hstu_config)
    fp32_ref_hstu_layer.load_state_dict(ref_hstu_layer.state_dict())
    fp32_ref_hstu_layer.cuda()
    ref_hstu_layer.cuda()
    fp32_input_module = fp32_ref_hstu_layer

    if not upcast_reference and dtype != torch.float32:
        ref_hstu_layer = Float16Module(hstu_config, ref_hstu_layer)
        input_module = ref_hstu_layer.module
    else:
        input_module = ref_hstu_layer

    if hidden_dim_per_head < 32:
        pytest.skip("Hidden dim per head must be >= 32")
    embedding_dim = hidden_dim_per_head * num_heads
    # TODO: this is not exact, but should be close
    max_seqlen = max_history_seqlen + max_num_targets + max_num_contextuals
    lengths = torch.randint(
        low=1, high=max_seqlen + 1, size=(batchsize,), device=device, dtype=torch.int
    )
    if scaling_seqlen == -1:
        scaling_seqlen = max_seqlen

    seq_offsets = length_to_complete_offsets(lengths)

    L = int(seq_offsets[-1].item())

    if attn_backend == KernelBackend.TRITON and max_num_contextuals > 0:
        pytest.skip("TRITON does not support contextuals")

    if attn_backend == KernelBackend.TRITON and target_group_size > 1:
        pytest.skip("TRITON does not support target grouped attention")
    num_targets = None
    num_contextuals = None

    if max_num_targets > 0:
        if only_targets:
            num_targets = lengths.clone()
        else:
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
        input = torch.nn.functional.dropout(input, p=0.6, training=True)

    input.requires_grad_()
    ref_input = input.detach().clone().requires_grad_()
    fp32_ref_input = input.float().detach().clone().requires_grad_()

    (
        input_norm_weight,
        input_norm_bias,
        output_norm_weight,
        output_norm_bias,
        linear_uvqk_weight,
        linear_uvqk_bias,
        linear_proj_weight,
    ) = generate_or_copy_parameters(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim_per_head=hidden_dim_per_head,
        dtype=dtype,
        device=device,
        learnable_ln=learnable_ln,
        input_module=input_module,
    )

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
        "scaling_seqlen": scaling_seqlen,
    }
    jd = JaggedData(values=ref_input, **ctor_nograd_dict)
    fp32_ref_jd = JaggedData(values=fp32_ref_input, **ctor_nograd_dict)
    # Float16Module converts output tensor object to fp32!, but here we return JaggedData object
    ref_out = ref_hstu_layer(jd)
    fp32_ref_out = fp32_ref_hstu_layer(fp32_ref_jd)

    out = fused_hstu_op(
        input=input,
        seqlen_offsets=seq_offsets,
        max_seqlen=max_seqlen,
        scaling_seqlen=scaling_seqlen,
        linear_uvqk_weight=linear_uvqk_weight,
        linear_uvqk_bias=linear_uvqk_bias,
        linear_proj_weight=linear_proj_weight,
        num_heads=num_heads,
        linear_dim_per_head=hidden_dim_per_head,
        attention_dim_per_head=hidden_dim_per_head,
        ln_eps=ln_eps,
        dropout_ratio=dropout_ratio,
        training=training,
        input_norm_weight=input_norm_weight,
        input_norm_bias=input_norm_bias,
        output_norm_weight=output_norm_weight,
        output_norm_bias=output_norm_bias,
        attn_backend=attn_backend,
        num_targets=num_targets,
        num_contextuals=num_contextuals,
        target_group_size=target_group_size,
        alpha=1.0 / (hidden_dim_per_head**0.5),
        causal=causal,
        seed=seed,
        residual=residual,
        wgrad_stream=None,
        wgrad_event=None,
        recompute_input_layernorm=recompute_input_layernorm,
        recompute_input_silu=recompute_input_silu,
    )
    ref_out = ref_out.values
    fp32_ref_out = fp32_ref_out.values
    diff = (out - ref_out).abs()
    print(f"\n fwd diff max : {diff.max()}, min {diff.min()}, mean : {diff.mean()}")
    # assert_hstu_close(out, ref_out, fp32_ref_out, fwd=True)
    dout = torch.empty_like(out).uniform_(-0.1, 0.1)
    # sparse the dout
    with torch.no_grad():
        dout = torch.nn.functional.dropout(dout, p=0.6, training=True)
    ref_out.backward(dout.detach().clone())  # bf16
    fp32_ref_out.backward(dout.detach().clone().float())
    with torch.no_grad():
        out.backward(dout.detach().clone())
    # dst
    input_size = linear_uvqk_weight.size(0)
    output_size = linear_uvqk_weight.size(1)
    grad_to_compared = OrderedDict(
        {
            "input_norm_weight": (
                input_norm_weight.grad if learnable_ln else None,
                input_module._input_layernorm_weight.grad if learnable_ln else None,
                fp32_input_module._input_layernorm_weight.grad
                if learnable_ln
                else None,
            ),
            "input_norm_bias": (
                input_norm_bias.grad if learnable_ln else None,
                input_module._input_layernorm_bias.grad if learnable_ln else None,
                fp32_input_module._input_layernorm_bias.grad if learnable_ln else None,
            ),
            # uvqk weight and bias layout is different from debug.
            "linear_uvqk_weight": (
                linear_uvqk_weight.grad,
                input_module._linear_uvqk.weight.grad.reshape(
                    num_heads, 4, -1, input_size
                )
                .transpose(0, 1)
                .reshape(output_size, input_size)
                .t(),
                fp32_input_module._linear_uvqk.weight.grad.reshape(
                    num_heads, 4, -1, input_size
                )
                .transpose(0, 1)
                .reshape(output_size, input_size)
                .t(),
            ),
            "linear_uvqk_bias": (
                linear_uvqk_bias.grad,
                input_module._linear_uvqk.bias.grad.reshape(num_heads, 4, -1)
                .transpose(0, 1)
                .reshape(output_size, -1),
                fp32_input_module._linear_uvqk.bias.grad.reshape(num_heads, 4, -1)
                .transpose(0, 1)
                .reshape(output_size, -1),
            ),
            "output_norm_weight": (
                output_norm_weight.grad if learnable_ln else None,
                input_module._output_layernorm_weight.grad if learnable_ln else None,
                fp32_input_module._output_layernorm_weight.grad
                if learnable_ln
                else None,
            ),
            "output_norm_bias": (
                output_norm_bias.grad if learnable_ln else None,
                input_module._output_layernorm_bias.grad if learnable_ln else None,
                fp32_input_module._output_layernorm_bias.grad if learnable_ln else None,
            ),
            "linear_proj_weight": (
                linear_proj_weight.grad,
                input_module._linear_proj.weight.grad.t(),
                fp32_input_module._linear_proj.weight.grad.t(),
            ),
        }
    )
    for tensor_name, (grad_fused, grad_ref, fp32_ref_grad) in reversed(
        grad_to_compared.items()
    ):
        print(
            (
                f"{tensor_name} abs diff max : {torch.abs(grad_ref - grad_fused).max()}, min {torch.abs(grad_ref - grad_fused).min()}, mean : {torch.abs(grad_ref - grad_fused).mean()}"
            )
        )
        assert_hstu_close(grad_fused, grad_ref, fp32_ref_grad, fwd=False)
        print(f"{tensor_name} assert passed")
        # torch.testing.assert_close(grad_ref, grad_fused)
    # torch.testing.assert_close(ref_input.grad, input.grad)
    assert_hstu_close(input.grad, ref_input.grad, fp32_ref_input.grad, fwd=False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="FP8 fused HSTU training requires Hopper",
)
@pytest.mark.parametrize(
    "up_fp8,down_fp8,attention_quant_mode,scaling,fp8_backward",
    [
        pytest.param(False, False, 0, "per-tensor", False, id="attention-unscaled"),
        pytest.param(False, False, 1, "per-tensor", False, id="attention-mode-1"),
        pytest.param(False, False, 2, "per-tensor", False, id="attention-mode-2"),
        pytest.param(False, False, 3, "per-tensor", False, id="attention-mode-3"),
        pytest.param(False, False, 4, "per-tensor", False, id="attention-mode-4"),
        pytest.param(False, False, 5, "per-tensor", False, id="attention-mode-5"),
        pytest.param(True, False, -1, "per-tensor", True, id="up-projection"),
        pytest.param(False, True, -1, "per-tensor", True, id="down-projection"),
        pytest.param(True, True, 5, "per-tensor", True, id="combined"),
        pytest.param(True, True, -1, "rowwise", False, id="rowwise-forward-only"),
    ],
)
def test_fused_hstu_op_fp8(
    monkeypatch, up_fp8, down_fp8, attention_quant_mode, scaling, fp8_backward
):
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    num_heads = 4
    head_dim = 64
    hidden_size = num_heads * head_dim
    # Cross both Hopper FP8 tile boundaries and include an empty sequence.
    # Mode 1 loads transposed operands with aligned per-sequence addresses.
    lengths = (
        (32, 0, 80, 144, 208) if attention_quant_mode == 1 else (23, 0, 65, 129, 201)
    )
    offsets = length_to_complete_offsets(
        torch.tensor(lengths, device=device, dtype=torch.int32)
    )
    with torch.random.fork_rng(devices=[device.index]), torch.no_grad():
        torch.manual_seed(1234)
        weights = generate_or_copy_parameters(
            embedding_dim=hidden_size,
            num_heads=num_heads,
            hidden_dim_per_head=head_dim,
            dtype=dtype,
            device=device,
            learnable_ln=True,
        )
        inputs = torch.empty(
            (sum(lengths), hidden_size), device=device, dtype=dtype
        ).uniform_(-0.1, 0.1)
        dout = torch.empty_like(inputs).uniform_(-0.1, 0.1)
    tensor_names = (
        "input",
        "input_norm_weight",
        "input_norm_bias",
        "output_norm_weight",
        "output_norm_bias",
        "linear_uvqk_weight",
        "linear_uvqk_bias",
        "linear_proj_weight",
    )

    def run(*, empty=False, training=True, **kwargs):
        tensors = {
            name: tensor.detach().clone().requires_grad_()
            for name, tensor in zip(
                tensor_names,
                (inputs[:0] if empty else inputs, *weights),
                strict=True,
            )
        }
        output = fused_hstu_op(
            **tensors,
            seqlen_offsets=torch.zeros_like(offsets) if empty else offsets,
            max_seqlen=max(lengths),
            scaling_seqlen=211,
            num_heads=num_heads,
            linear_dim_per_head=head_dim,
            attention_dim_per_head=head_dim,
            ln_eps=1e-6,
            dropout_ratio=0.0,
            training=training,
            attn_backend=KernelBackend.CUTLASS,
            alpha=head_dim**-0.5,
            causal=True,
            residual=False,
            **kwargs,
        )
        if not training:
            return (output,)
        gradients = torch.autograd.grad(
            output, tuple(tensors.values()), dout[:0] if empty else dout
        )
        return (output.detach(), *gradients)

    attention_calls = []

    def capture_attention(name, q_index):
        operator = getattr(torch.ops.fbgemm, name)

        def call(*args, **kwargs):
            attention_calls.append((name, args[q_index].dtype, kwargs["quant_mode"]))
            return operator(*args, **kwargs)

        monkeypatch.setattr(torch.ops.fbgemm, name, call)

    capture_attention("hstu_varlen_fwd_90", q_index=0)
    capture_attention("hstu_varlen_bwd_90", q_index=2)
    bf16_gemms = 0
    mm = torch.mm

    def capture_mm(a, b):
        nonlocal bf16_gemms
        assert a.is_cuda and b.is_cuda
        assert a.dtype == b.dtype == dtype
        bf16_gemms += 1
        return mm(a, b)

    monkeypatch.setattr(torch, "mm", capture_mm)
    te_gemms = []
    rowwise_gemms = []
    if (up_fp8 or down_fp8) and scaling == "per-tensor":
        te = pytest.importorskip("transformer_engine.pytorch")
        te_gemm = pytest.importorskip("transformer_engine.pytorch.cpp_extensions")
        general_gemm = te_gemm.general_gemm

        def capture_gemm(a, b, *args, **kwargs):
            assert isinstance(a, te.Float8Tensor)
            assert isinstance(b, te.Float8Tensor)
            assert kwargs["out_dtype"] == dtype
            te_gemms.append((kwargs["layout"], a.shape))
            return general_gemm(a, b, *args, **kwargs)

        monkeypatch.setattr(te_gemm, "general_gemm", capture_gemm)
    if scaling == "rowwise":
        scaled_mm = torch.nn.functional.scaled_mm

        def capture_scaled_mm(
            a, b, scale_a, scale_recipe_a, scale_b, scale_recipe_b, *args, **kwargs
        ):
            assert a.dtype == b.dtype == torch.float8_e4m3fn
            assert scale_a.numel() == a.size(0)
            assert scale_b.numel() == b.size(1)
            assert (
                scale_recipe_a
                == scale_recipe_b
                == torch.nn.functional.ScalingType.RowWise
            )
            assert kwargs["output_dtype"] == dtype
            rowwise_gemms.append(b.size(1))
            return scaled_mm(
                a, b, scale_a, scale_recipe_a, scale_b, scale_recipe_b, *args, **kwargs
            )

        monkeypatch.setattr(torch.nn.functional, "scaled_mm", capture_scaled_mm)
    bf16 = run()
    assert not te_gemms and not rowwise_gemms
    assert bf16_gemms == 4
    bf16_gemms = 0
    fp8 = run(
        up_projection_fp8=up_fp8,
        down_projection_fp8=down_fp8,
        attention_quant_mode=attention_quant_mode,
        linear_fp8_scaling=scaling,
        linear_fp8_backward=fp8_backward,
    )

    attention_dtype = torch.float8_e4m3fn if attention_quant_mode >= 0 else dtype
    assert attention_calls == [
        ("hstu_varlen_fwd_90", dtype, -1),
        ("hstu_varlen_bwd_90", dtype, -1),
        ("hstu_varlen_fwd_90", attention_dtype, attention_quant_mode),
        ("hstu_varlen_bwd_90", attention_dtype, attention_quant_mode),
    ]
    widths = [
        width
        for enabled, width in ((up_fp8, 4 * hidden_size), (down_fp8, hidden_size))
        if enabled
    ]
    if scaling == "rowwise":
        assert rowwise_gemms == widths
    else:
        assert [shape[1] for layout, shape in te_gemms if layout == "NN"] == widths
        backward_layouts = [layout for layout, _ in te_gemms if layout != "NN"]
        assert backward_layouts == (["TN", "NT"] * len(widths) if fp8_backward else [])
    assert bf16_gemms == 4 - (2 * len(widths) if fp8_backward else 0)
    labels = ("output", *(f"{name} gradient" for name in tensor_names))
    for name, actual, expected in zip(labels, fp8, bf16, strict=True):
        assert actual.dtype == expected.dtype == dtype, name
        assert torch.isfinite(actual).all(), name
        if attention_quant_mode != 0:
            relative_error = (actual.float() - expected.float()).norm() / (
                expected.float().norm()
            )
            tolerance = 0.12 if name == "output" else 0.18
            assert (
                relative_error < tolerance
            ), f"{name} relative L2 error {relative_error}"

    if attention_quant_mode == 1:
        offsets[1:3] += 1
        with pytest.raises(ValueError, match="mode 1 requires .* multiples of 16"):
            run(attention_quant_mode=1)

    if up_fp8 and down_fp8 and attention_quant_mode == 5:
        attention_calls.clear()
        te_gemms.clear()
        empty_fp8 = dict(
            empty=True,
            up_projection_fp8=True,
            down_projection_fp8=True,
            attention_quant_mode=5,
            linear_fp8_backward=True,
        )
        with torch.no_grad():
            (empty_eval,) = run(**empty_fp8, training=False)
        empty_train, *gradients = run(**empty_fp8)
        for empty_output in (empty_eval, empty_train):
            assert empty_output.shape == (0, hidden_size)
            assert empty_output.dtype == dtype
        for name, grad in zip(tensor_names, gradients, strict=True):
            assert grad.dtype == dtype, name
            assert torch.count_nonzero(grad) == 0, name
        assert attention_calls == [
            ("hstu_varlen_fwd_90", dtype, -1),
            ("hstu_varlen_fwd_90", dtype, -1),
            ("hstu_varlen_bwd_90", dtype, -1),
        ]
        assert not te_gemms
