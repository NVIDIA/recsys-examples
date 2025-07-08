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
from commons.utils.hstu_assert_close import assert_hstu_close
from configs import get_hstu_config
from configs.hstu_config import HSTULayerType, KernelBackend
from megatron.core.transformer.module import Float16Module
from modules.debug.debug_hstu_layer import HSTULayer
from modules.hstu_attention import create_hstu_attention
from modules.jagged_data import JaggedData
from ops.fused_hstu_op import fused_hstu_op
from ops.length_to_offsets import length_to_complete_offsets


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
    "max_seqlen,max_num_candidates,max_num_contextuals",
    [
        (1024, 128, 6),
        (32, 0, 0),
        (1024, 128, 0),
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
