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
from typing import Optional, Tuple, Union

import hstu_attn_2_cuda as flash_attn_cuda_ampere

try:
    import hstu_hopper_cuda as flash_attn_cuda_hopper
except ImportError:
    pass
import nvtx
import torch
from commons.utils.clear_tensor_data import clear_tensor_data
from configs import KernelBackend
from ops.pt_ops.torch_addmm import torch_addmm_silu_fwd
from ops.triton_ops.triton_addmm import triton_addmm_silu_bwd, triton_addmm_silu_fwd
from ops.triton_ops.triton_hstu_attention import (
    triton_hstu_attention_bwd,
    triton_hstu_attention_fwd,
)
from ops.triton_ops.triton_layer_norm import (
    triton_weighted_layer_norm_bwd,
    triton_weighted_layer_norm_fwd,
)
from ops.triton_ops.triton_norm_mul_dropout import (
    triton_layer_norm_mul_dropout_bwd,
    triton_layer_norm_mul_dropout_fwd,
)


class FusedHSTULayerFunction(torch.autograd.Function):
    """
    This function has better precision performance than the native HSTULayer.

    y = layer_norm(input, input_norm_weight, input_norm_bias)
    y = linear_uvqk(y, linear_uvqk_weight, linear_uvqk_bias)
    y = silu(y)
    u,v,q,k = split(y)
    attn_out = hstu_attn(q,k,v)
    y = norm_mul_dropout(attn_out, u, output_norm_weight, output_norm_bias)
    y = linear_proj(y, linear_proj_weight) + x

    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,  # [T, hidden_size]
        seqlen_offsets: torch.Tensor,  # [batchsize]
        max_seqlen: int,  # N
        linear_uvqk_weight: torch.Tensor,
        linear_uvqk_bias: torch.Tensor,
        linear_proj_weight: torch.Tensor,
        num_heads: int,
        linear_dim_per_head: int,
        attention_dim_per_head: int,
        ln_eps: float,
        dropout_ratio: float,
        training: bool,
        # layer norm weight and bias
        input_norm_weight: Optional[torch.Tensor] = None,
        input_norm_bias: Optional[torch.Tensor] = None,
        output_norm_weight: Optional[torch.Tensor] = None,
        output_norm_bias: Optional[torch.Tensor] = None,
        # attn related
        attn_backend: KernelBackend = KernelBackend.CUTLASS,
        num_targets: Optional[torch.Tensor] = None,
        num_contextuals: Union[int, Optional[torch.Tensor]] = None,
        target_group_size: int = 1,
        alpha: float = 1.0,
        causal: bool = True,
        # dropout related
        seed: Optional[int] = None,
        # only for debug purpose!
        residual: bool = True,
        wgrad_stream: Optional[torch.cuda.Stream] = None,
        wgrad_event: Optional[torch.cuda.Event] = None,
        recompute_input_layernorm: bool = False,
        recompute_input_silu: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the fused HSTU layer.
        Args:
            input (torch.Tensor): Input tensor of shape [T, hidden_size]
            seqlen_offsets (torch.Tensor): Sequence length offsets tensor of shape [batchsize,]
            max_seqlen (int): Maximum sequence length.
            linear_uvqk_weight (torch.Tensor): Weight matrix for linear UVQK.
            linear_uvqk_bias (torch.Tensor): Bias vector for linear UVQK.
            linear_proj_weight (torch.Tensor): Weight matrix for final linear projection.
            num_heads (int): Number of attention heads.
            linear_dim_per_head (int): Linear dimension per head.
            attention_dim_per_head (int): Attention dimension per head.
            ln_eps (float): Layer norm epsilon
            dropout_ratio (float): Dropout probability
            training (bool): Whether in training mode
            input_norm_weight (Optional[torch.Tensor]): Input layer norm weight. Defaults to None.
            input_norm_bias (Optional[torch.Tensor]): Input layer norm bias. Defaults to None.
            output_norm_weight (Optional[torch.Tensor]): Output layer norm weight. Defaults to None.
            output_norm_bias (Optional[torch.Tensor]): Output layer norm bias. Defaults to None.
            attn_backend (KernelBackend): Attention kernel backend. KernelBackend.CUTLASS | KernelBackend.TRITON. Defaults to KernelBackend.CUTLASS.
            num_targets (Optional[torch.Tensor]): Number of targets. Defaults to None.
            num_contextuals (Union[int, Optional[torch.Tensor]]): Number of contextual tokens. Defaults to None.
            target_group_size (int): Target group size. Defaults to 1.
            alpha (float): Attention scaling factor. Defaults to 1.0.
            causal (bool): Whether to use causal attention. Defaults to True.
            seed (Optional[int]): Random seed for dropout(required by triton dropout). Defaults to None.
            residual (bool): Whether to use residual connection. Defaults to True.
            wgrad_stream (Optional[torch.cuda.Stream]): CUDA stream for weight gradient computation. Defaults to None.
            wgrad_event (Optional[torch.cuda.Event]): CUDA event for weight gradient computation. Defaults to None.
            recompute_input_layernorm (bool): Whether to recompute the input layer norm. Defaults to False.
            recompute_input_silu (bool): Whether to recompute the input silu. Defaults to False.
        Returns:
            torch.Tensor: Output tensor of shape [T, hidden_size]
        """
        ctx.attn_backend = attn_backend
        ctx.learnable_input_norm = input_norm_weight is not None
        ctx.learnable_output_norm = output_norm_weight is not None
        ctx.eps = ln_eps
        ctx.dropout_ratio = dropout_ratio
        ctx.num_heads = num_heads
        ctx.linear_dim_per_head = linear_dim_per_head
        ctx.attention_dim_per_head = attention_dim_per_head
        ctx.causal = causal
        ctx.alpha = alpha
        ctx.training = training
        ctx.residual = residual
        ctx.wgrad_stream = wgrad_stream
        ctx.wgrad_event = wgrad_event
        ctx.recompute_input_layernorm = recompute_input_layernorm
        ctx.recompute_input_silu = recompute_input_silu
        saved_tensor_map = OrderedDict()
        if num_contextuals is None and attn_backend == KernelBackend.TRITON:
            num_contextuals = 0
        if attn_backend == KernelBackend.TRITON:
            assert isinstance(num_contextuals, int)
        assert input.dim() == 2, "input tensor must be 2D"
        assert linear_uvqk_bias.dim() == 1, "linear_uvqk_bias must be 1D"

        assert not ctx.learnable_input_norm or input_norm_bias is not None
        assert not ctx.learnable_output_norm or output_norm_bias is not None

        _split_arg_list = [
            linear_dim_per_head * num_heads,
            linear_dim_per_head * num_heads,
            attention_dim_per_head * num_heads,
            attention_dim_per_head * num_heads,
        ]
        ctx.split_arg_list = _split_arg_list

        def _ln_linear_silu_fwd(
            input, ln_weight, ln_bias, linear_weight, linear_bias, ln_eps
        ):
            # 1. layer norm
            (
                normed_input,
                input_mean,
                input_rstd,
                input_BLOCK_D,
                input_num_warps,
            ) = triton_weighted_layer_norm_fwd(
                x=input,
                weight=ln_weight,
                bias=ln_bias,
                eps=ln_eps,
            )

            saved_tensor_map.update(
                {
                    "input": input,
                    "input_ln_weight": ln_weight,
                    "input_ln_bias": ln_bias,
                    "input_ln_mean": input_mean,
                    "input_ln_rstd": input_rstd,
                }
            )

            ctx.input_BLOCK_D = input_BLOCK_D
            ctx.input_num_warps = input_num_warps
            sm = torch.cuda.get_device_properties(0).major
            if sm == 8:
                addmm_silu_fwd_impl = triton_addmm_silu_fwd
            elif sm == 9:
                addmm_silu_fwd_impl = torch_addmm_silu_fwd
            else:
                raise ValueError(f"Unsupported SM major version: {sm}")
            # 2. linear & silu
            # bias is 1D
            linear_uvqk, silu_linear_uvqk = addmm_silu_fwd_impl(
                x=normed_input,
                w=linear_weight,
                y=linear_bias,
                silu=True,
            )
            # for gemm backward
            saved_tensor_map.update(
                {
                    "linear_uvqk_input": normed_input
                    if not recompute_input_layernorm
                    else None,
                    "linear_uvqk_weight": linear_weight,
                }
            )
            if recompute_input_layernorm:
                clear_tensor_data(normed_input, clear_storage=True)
                del normed_input
            saved_tensor_map.update(
                {
                    "silu_input": linear_uvqk,
                }
            )
            return silu_linear_uvqk

        def _hstu_attn_triton_fwd(
            N: int,
            alpha: float,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seq_offsets: torch.Tensor,
            causal: bool,
            num_targets: Optional[torch.Tensor],
            contextual_seq_len: int,
        ):
            saved_tensor_map.update(
                {
                    "q": q if not recompute_input_silu else None,
                    "k": k if not recompute_input_silu else None,
                    "v": v if not recompute_input_silu else None,
                    "seq_offsets": seq_offsets,
                    "num_contexts": None,
                    "num_targets": num_targets,
                }
            )

            ctx.has_multiple_targets = num_targets is not None
            ctx.N = N
            ctx.contextual_seq_len = contextual_seq_len

            jagged_attn_output = triton_hstu_attention_fwd(
                N=N,
                alpha=alpha,
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                causal=causal,
                num_targets=num_targets,
                max_attn_len=0,
                contextual_seq_len=contextual_seq_len,
                sort_by_length_indices=None,
            ).reshape(-1, num_heads * attention_dim_per_head)
            return jagged_attn_output

        def _hstu_attn_cutlass_fwd(
            q,
            k,
            v,
            seq_offsets_q,
            max_seqlen_q,
            num_contexts,
            num_targets,
            target_group_size,
            alpha,
        ):
            sm_major_version = torch.cuda.get_device_properties(0).major
            extension_args = ()
            if sm_major_version == 8:
                cutlass_hstu_varlen_fwd = flash_attn_cuda_ampere.varlen_fwd
                ampere_paged_kv_args = (None, None, None, None, None)
                extension_args = ampere_paged_kv_args
            elif sm_major_version == 9:
                cutlass_hstu_varlen_fwd = flash_attn_cuda_hopper.varlen_fwd
                hopper_fp8_args = (None, None, None)
                extension_args = hopper_fp8_args

            else:
                raise ValueError(f"Unsupported SM major version: {sm_major_version}")
            assert q.dim() == 3, "q shape should be (L, num_heads, head_dim)"
            assert k.dim() == 3, "k shape should be (L, num_heads, head_dim)"
            assert v.dim() == 3, "v shape should be (L, num_heads, hidden_dim)"
            seq_offsets_q = seq_offsets_q.to(torch.int32)
            num_contexts = (
                num_contexts.to(torch.int32) if num_contexts is not None else None
            )
            num_targets = (
                num_targets.to(torch.int32) if num_targets is not None else None
            )
            jagged_attn_output, _ = cutlass_hstu_varlen_fwd(
                q,
                k,
                v,
                seq_offsets_q,
                seq_offsets_q,
                max_seqlen_q,
                max_seqlen_q,
                num_contexts,
                num_targets,
                target_group_size,
                -1,  # window_size_left
                0,  # window_size_right
                alpha,
                None,  # rab
                False,  # is_delta_q
                *extension_args,
            )
            # in case of padding
            P = jagged_attn_output[:, :, :linear_dim_per_head].reshape(
                -1, num_heads * linear_dim_per_head
            )
            saved_tensor_map.update(
                {
                    "q": q if not recompute_input_silu else None,
                    "k": k if not recompute_input_silu else None,
                    "v": v if not recompute_input_silu else None,
                    "seq_offsets_q": seq_offsets_q,
                    "num_contexts": num_contexts,
                    "num_targets": num_targets,
                }
            )
            del jagged_attn_output
            ctx.max_seqlen_q = max_seqlen_q
            ctx.target_group_size = target_group_size
            ctx.window_size_left = -1
            ctx.window_size_right = 0
            ctx.has_drab = False
            ctx.is_delta_q = False

            return P

        def _norm_mul_dropout_fwd(
            x: torch.Tensor,
            u: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            eps: float,
            dropout_ratio: float,
            training: bool,
            dropout_seed: Optional[int] = None,
        ):
            (
                y,
                mean,
                rstd,
                BLOCK_D,
                num_warps,
                ret_seed,
            ) = triton_layer_norm_mul_dropout_fwd(
                x=x,
                u=u,
                weight=weight,
                bias=bias,
                eps=eps,
                dropout_ratio=dropout_ratio,
                training=training,
                concat_ux=False,
                seed=dropout_seed,
            )

            ctx.dropout_seed = ret_seed
            ctx.output_BLOCK_D = BLOCK_D
            ctx.output_num_warps = num_warps
            saved_tensor_map.update(
                {
                    "out_ln_input": x,
                    "u": u if not recompute_input_silu else None,
                    "out_ln_weight": weight,
                    "out_ln_bias": bias,
                    "out_ln_mean": mean,
                    "out_ln_rstd": rstd,
                }
            )
            return y

        def _linear_residual_fwd(
            residual,
            x,
            w,
        ):
            sm = torch.cuda.get_device_properties(0).major
            if sm == 8:
                addmm_silu_fwd_impl = triton_addmm_silu_fwd
            elif sm == 9:
                addmm_silu_fwd_impl = torch_addmm_silu_fwd
            else:
                raise ValueError(f"Unsupported SM major version: {sm}")
            y, _ = addmm_silu_fwd_impl(
                x=x,
                w=w,
                y=residual,
                silu=False,
            )
            saved_tensor_map.update(
                {
                    "linear_proj_input": x,
                    "linear_proj_weight": w,
                }
            )
            return y

        with nvtx.annotate("hstu ln+linear_bias+silu fwd", color="RED"):
            act_linear_uvqk = _ln_linear_silu_fwd(
                input=input,
                ln_weight=input_norm_weight,
                ln_bias=input_norm_bias,
                linear_weight=linear_uvqk_weight,
                linear_bias=linear_uvqk_bias,
                ln_eps=ln_eps,
            )
            tu, tv, tq, tk = torch.split(
                act_linear_uvqk,
                _split_arg_list,
                dim=-1,
            )
            tv = tv.view(-1, num_heads, linear_dim_per_head)
            tq = tq.view(-1, num_heads, attention_dim_per_head)
            tk = tk.view(-1, num_heads, attention_dim_per_head)

        with nvtx.annotate("hstu attn fwd", color="BLUE"):
            if ctx.attn_backend == KernelBackend.CUTLASS:
                # attn_output: [T, num_heads * attention_dim_per_head]
                attn_output = _hstu_attn_cutlass_fwd(
                    q=tq,
                    k=tk,
                    v=tv,
                    seq_offsets_q=seqlen_offsets,
                    max_seqlen_q=max_seqlen,
                    num_contexts=num_contextuals,
                    num_targets=num_targets,
                    target_group_size=target_group_size,
                    alpha=ctx.alpha,
                )
            else:
                assert isinstance(
                    num_contextuals, int
                ), "num_contextuals must be an int when kernel backend is triton"
                assert (
                    target_group_size == 1
                ), "target_group_size must be 1 when kernel backend is triton"
                attn_output = _hstu_attn_triton_fwd(
                    N=max_seqlen,
                    alpha=ctx.alpha,
                    q=tq,
                    k=tk,
                    v=tv,
                    seq_offsets=seqlen_offsets,
                    causal=ctx.causal,
                    num_targets=num_targets,
                    contextual_seq_len=num_contextuals,
                )
        with nvtx.annotate("hstu norm mul dropout fwd", color="GREEN"):
            # dropout ratio and seed are set in ctx

            assert output_norm_weight is not None, "output_norm_weight must be provided"
            assert output_norm_bias is not None, "output_norm_bias must be provided"
            # register is in fp32
            y = _norm_mul_dropout_fwd(
                x=attn_output,
                u=tu,
                weight=output_norm_weight,
                bias=output_norm_bias,
                eps=ctx.eps,
                dropout_ratio=ctx.dropout_ratio,
                training=ctx.training,
                dropout_seed=seed,
            )
        if ctx.recompute_input_silu:
            clear_tensor_data(tu, tq, tk, tv)
            del tu, tq, tk, tv
        # if recompute is disabled, we cannot clear storage because saved u,q,k,v refers to the same storage
        clear_tensor_data(act_linear_uvqk, clear_storage=ctx.recompute_input_silu)
        del act_linear_uvqk
        with nvtx.annotate("hstu linear_residual fwd", color="YELLOW"):
            # Note that when residual is off, there might be slightly perf loss due to the tensor construction
            residual_tensor = torch.zeros_like(input) if not ctx.residual else input
            out = _linear_residual_fwd(
                residual=residual_tensor,
                x=y,
                w=linear_proj_weight,
            )
        ctx.save_for_backward(*saved_tensor_map.values())
        ctx.saved_tensor_name = list(saved_tensor_map.keys())
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ) -> Tuple[
        torch.Tensor,
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        def _linear_residual_bwd(
            grad_output,
            x,
            w,
            wgrad_stream: Optional[torch.cuda.Stream] = None,
            wgrad_event: Optional[torch.cuda.Event] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # triton_addmm_silu_bwd are cublas-based even though the name starts with triton
            grad_x, grad_w, grad_residual = triton_addmm_silu_bwd(
                x=x,
                w=w,
                z=None,  # silu is False, and thus z is not used
                grad_output=grad_output,
                is_y_1d=False,
                wgrad_stream=wgrad_stream,
                wgrad_event=wgrad_event,
            )

            return grad_x, grad_w, grad_residual

        def _norm_mul_dropout_bwd(
            dy: torch.Tensor,
            x: torch.Tensor,
            u: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            mean: torch.Tensor,
            rstd: torch.Tensor,
            BLOCK_D: int,
            num_warps: int,
            eps: float,
            training: bool,
            dropout_ratio: float,
            seed: Optional[int] = None,
            wait_event: Optional[torch.cuda.Event] = None,
            du: Optional[torch.Tensor] = None,
        ):
            dx, du, dweight, dbias, _ = triton_layer_norm_mul_dropout_bwd(
                dy=dy,
                x=x,
                u=u,
                weight=weight,
                bias=bias,
                mean=mean,
                rstd=rstd,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                eps=eps,
                training=training,
                dropout_ratio=dropout_ratio,
                seed=seed,
                concat_ux=False,
                compute_y=False,
                wait_event=wait_event,
                du=du,
            )
            return dx, du, dweight, dbias

        def _hstu_attn_cutlass_bwd(
            dout: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seq_offsets_q: torch.Tensor,
            max_seqlen_q: int,
            num_contexts: Optional[torch.Tensor],  # b
            num_targets: Optional[torch.Tensor],  # b
            target_group_size: int,
            window_size_left: int,
            window_size_right: int,
            alpha: float,
            dq: Optional[torch.Tensor] = None,
            dk: Optional[torch.Tensor] = None,
            dv: Optional[torch.Tensor] = None,
        ):
            sm_major_version = torch.cuda.get_device_properties(0).major
            if sm_major_version == 8:
                cutlass_hstu_varlen_bwd = flash_attn_cuda_ampere.varlen_bwd
            elif sm_major_version == 9:
                cutlass_hstu_varlen_bwd = flash_attn_cuda_hopper.varlen_bwd
            else:
                raise ValueError(f"Unsupported SM major version: {sm_major_version}")
            assert dout.dim() == 3
            dq, dk, dv, _ = cutlass_hstu_varlen_bwd(
                dout,
                q,
                k,
                v,
                dq,
                dk,
                dv,
                seq_offsets_q,
                seq_offsets_q,
                max_seqlen_q,
                max_seqlen_q,
                num_contexts,
                num_targets,
                target_group_size,
                window_size_left,
                window_size_right,
                alpha,
                None,  # rab_padded
                False,  # has_drab
                False,  # is_delta_q
                False,  # deterministic
            )

            return dq, dk, dv

        def _hstu_attn_triton_bwd(
            dout: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seq_offsets: torch.Tensor,
            num_targets: Optional[torch.Tensor],
            N: int,
            alpha: float,
            causal: float,
            contextual_seq_len: int,
            dqkv: Optional[torch.Tensor] = None,
        ):
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq, dk, dv = triton_hstu_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=N,
                alpha=alpha,
                max_attn_len=0,
                causal=causal,
                contextual_seq_len=contextual_seq_len,
                sort_by_length_indices=None,
            )
            return dq, dk, dv

        def _ln_linear_silu_bwd(
            grad_output,
            # ln
            input,
            ln_weight: Optional[torch.Tensor],
            ln_bias: Optional[torch.Tensor],
            learnable: bool,
            ln_mean,
            ln_rstd,
            ln_eps,
            BLOCK_D: int,
            num_warps: int,
            # linear
            linear_input,
            linear_weight,
            # silu
            silu_input,
            # grad residual out
            dx_residual: Optional[torch.Tensor] = None,
            wgrad_stream: Optional[torch.cuda.Stream] = None,
            wgrad_event: Optional[torch.cuda.Event] = None,
        ):
            assert (
                grad_output.dim() == 2
            ), "grad_output shape should be (T, num_heads * attention_dim_per_head)"
            # 1. silu + linear
            (
                grad_linear_input,
                grad_linear_weight,
                grad_lienar_bias,
            ) = triton_addmm_silu_bwd(
                x=linear_input,
                w=linear_weight,
                z=silu_input,
                grad_output=grad_output,
                is_y_1d=True,
                silu=True,
                wgrad_stream=wgrad_stream,
                wgrad_event=wgrad_event,
            )
            # # 2. ln
            grad_input, grad_ln_weight, grad_ln_bias = triton_weighted_layer_norm_bwd(
                dy=grad_linear_input,
                x=input,
                weight=ln_weight,
                bias=ln_bias,
                mean=ln_mean,
                rstd=ln_rstd,
                learnable=learnable,
                eps=ln_eps,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                dx_accumulate=dx_residual,
                wait_event=wgrad_event,
            )
            return (
                grad_input,
                grad_ln_weight,
                grad_ln_bias,
                grad_linear_weight,
                grad_lienar_bias,
            )

        saved_tensor_name = ctx.saved_tensor_name
        saved_tensors = ctx.saved_tensors
        saved_tensor_map = OrderedDict(zip(saved_tensor_name, saved_tensors))
        duvqk = torch.empty_like(saved_tensor_map["silu_input"])
        pre_du, pre_dv, pre_dq, pre_dk = duvqk.split(ctx.split_arg_list, dim=-1)
        with nvtx.annotate("hstu linear_residual bwd", color="YELLOW"):
            (
                grad_output,
                grad_linear_proj_weight,
                grad_proj_residual,
            ) = _linear_residual_bwd(
                grad_output=grad_output,
                x=saved_tensor_map["linear_proj_input"],
                w=saved_tensor_map["linear_proj_weight"],
                wgrad_stream=ctx.wgrad_stream,
                wgrad_event=ctx.wgrad_event,
            )
        if ctx.recompute_input_silu:
            silu_input = saved_tensor_map["silu_input"]
            merged_logit = torch.ops.aten.silu(silu_input)
            # split does nothing
            u, v, q, k = merged_logit.split(ctx.split_arg_list, dim=-1)
            saved_tensor_map["u"] = u
            saved_tensor_map["v"] = v.view(
                -1, ctx.num_heads, ctx.attention_dim_per_head
            )
            saved_tensor_map["q"] = q.view(
                -1, ctx.num_heads, ctx.attention_dim_per_head
            )
            saved_tensor_map["k"] = k.view(
                -1, ctx.num_heads, ctx.attention_dim_per_head
            )
        with nvtx.annotate("norm_mul_dropout bwd", color="GREEN"):
            (
                grad_output,
                grad_u,
                grad_out_ln_weight,
                grad_out_ln_bias,
            ) = _norm_mul_dropout_bwd(
                dy=grad_output,
                x=saved_tensor_map["out_ln_input"],
                u=saved_tensor_map["u"],
                weight=saved_tensor_map["out_ln_weight"],
                bias=saved_tensor_map["out_ln_bias"],
                mean=saved_tensor_map["out_ln_mean"],
                rstd=saved_tensor_map["out_ln_rstd"],
                BLOCK_D=ctx.output_BLOCK_D,
                num_warps=ctx.output_num_warps,
                eps=ctx.eps,
                training=ctx.training,
                dropout_ratio=ctx.dropout_ratio,
                seed=ctx.dropout_seed,
                wait_event=ctx.wgrad_event,
                du=pre_du,
            )
        with nvtx.annotate("hstu attn bwd", color="BLUE"):
            if ctx.attn_backend == KernelBackend.CUTLASS:
                grad_q, grad_k, grad_v = _hstu_attn_cutlass_bwd(
                    dout=grad_output.view(
                        -1, ctx.num_heads, ctx.attention_dim_per_head
                    ),
                    q=saved_tensor_map["q"],
                    k=saved_tensor_map["k"],
                    v=saved_tensor_map["v"],
                    seq_offsets_q=saved_tensor_map["seq_offsets_q"],
                    max_seqlen_q=ctx.max_seqlen_q,
                    num_contexts=saved_tensor_map["num_contexts"],
                    num_targets=saved_tensor_map["num_targets"],
                    target_group_size=ctx.target_group_size,
                    window_size_left=ctx.window_size_left,
                    window_size_right=ctx.window_size_right,
                    alpha=ctx.alpha,
                    dq=pre_dq.view(-1, ctx.num_heads, ctx.attention_dim_per_head),
                    dk=pre_dk.view(-1, ctx.num_heads, ctx.attention_dim_per_head),
                    dv=pre_dv.view(-1, ctx.num_heads, ctx.attention_dim_per_head),
                )
                grad_output = duvqk
            else:
                grad_q, grad_k, grad_v = _hstu_attn_triton_bwd(
                    dout=grad_output.view(
                        -1, ctx.num_heads, ctx.attention_dim_per_head
                    ),
                    q=saved_tensor_map["q"],
                    k=saved_tensor_map["k"],
                    v=saved_tensor_map["v"],
                    seq_offsets=saved_tensor_map["seq_offsets"],
                    num_targets=saved_tensor_map["num_targets"],
                    N=ctx.N,  # => max_seqlen_q
                    alpha=ctx.alpha,
                    causal=ctx.causal,
                    contextual_seq_len=ctx.contextual_seq_len,  # saved_tensor_map["num_contexts"] == None,
                )
                grad_q = grad_q.view(-1, ctx.num_heads * ctx.attention_dim_per_head)
                grad_k = grad_k.view(-1, ctx.num_heads * ctx.attention_dim_per_head)
                grad_v = grad_v.view(-1, ctx.num_heads * ctx.attention_dim_per_head)
                grad_u = grad_u.view(-1, ctx.num_heads * ctx.attention_dim_per_head)
                grad_output = torch.cat(
                    [grad_u, grad_v, grad_q, grad_k], dim=-1
                ).contiguous()
        with nvtx.annotate("ln_linear_silu bwd", color="RED"):
            if ctx.recompute_input_layernorm:
                (
                    normed_input,
                    _,
                    _,
                    _,
                    _,
                ) = triton_weighted_layer_norm_fwd(
                    x=saved_tensor_map["input"],
                    weight=saved_tensor_map["input_ln_weight"],
                    bias=saved_tensor_map["input_ln_bias"],
                    eps=ctx.eps,
                    mean=saved_tensor_map["input_ln_mean"],
                    rstd=saved_tensor_map["input_ln_rstd"],
                )
                saved_tensor_map["linear_uvqk_input"] = normed_input
            (
                grad_input,
                grad_input_ln_weight,
                grad_input_ln_bias,
                grad_linear_uqkv_weight,
                grad_linear_uqkv_bias,
            ) = _ln_linear_silu_bwd(
                grad_output=grad_output,
                input=saved_tensor_map["input"],
                ln_weight=saved_tensor_map["input_ln_weight"],  # Optional[torch.Tensor]
                ln_bias=saved_tensor_map["input_ln_bias"],  # Optional[torch.Tensor]
                learnable=ctx.learnable_input_norm,
                ln_mean=saved_tensor_map["input_ln_mean"],
                ln_rstd=saved_tensor_map["input_ln_rstd"],
                ln_eps=ctx.eps,
                BLOCK_D=ctx.input_BLOCK_D,
                num_warps=ctx.input_num_warps,
                linear_input=saved_tensor_map["linear_uvqk_input"],
                linear_weight=saved_tensor_map["linear_uvqk_weight"],
                silu_input=saved_tensor_map["silu_input"],
                dx_residual=grad_proj_residual if ctx.residual else None,
                wgrad_stream=ctx.wgrad_stream,
                wgrad_event=ctx.wgrad_event,
            )
        del saved_tensor_map

        return (
            grad_input,
            None,
            None,
            grad_linear_uqkv_weight,
            grad_linear_uqkv_bias,
            grad_linear_proj_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_input_ln_weight,
            grad_input_ln_bias,
            grad_out_ln_weight,
            grad_out_ln_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_hstu_op(
    input: torch.Tensor,  # [T, hidden_size]
    seqlen_offsets: torch.Tensor,  # [batchsize]
    max_seqlen: int,  # N
    linear_uvqk_weight: torch.Tensor,
    linear_uvqk_bias: torch.Tensor,
    linear_proj_weight: torch.Tensor,
    num_heads: int,
    linear_dim_per_head: int,
    attention_dim_per_head: int,
    ln_eps: float,
    dropout_ratio: float,
    training: bool,
    # layer norm weight and bias
    input_norm_weight: Optional[torch.Tensor] = None,
    input_norm_bias: Optional[torch.Tensor] = None,
    output_norm_weight: Optional[torch.Tensor] = None,
    output_norm_bias: Optional[torch.Tensor] = None,
    # attn related
    attn_backend: KernelBackend = KernelBackend.CUTLASS,
    num_targets: Optional[torch.Tensor] = None,
    num_contextuals: Union[int, Optional[torch.Tensor]] = None,
    target_group_size: int = 1,
    alpha: float = 1.0,
    causal: bool = True,
    # dropout related
    seed: Optional[int] = None,
    # only for debug purpose!
    residual: bool = True,
    wgrad_stream: Optional[torch.cuda.Stream] = None,
    wgrad_event: Optional[torch.cuda.Event] = None,
    recompute_input_layernorm: bool = False,
    recompute_input_silu: bool = False,
):
    out = FusedHSTULayerFunction.apply(
        input,
        seqlen_offsets,
        max_seqlen,
        linear_uvqk_weight,
        linear_uvqk_bias,
        linear_proj_weight,
        num_heads,
        linear_dim_per_head,
        attention_dim_per_head,
        ln_eps,
        dropout_ratio,
        training,
        input_norm_weight,
        input_norm_bias,
        output_norm_weight,
        output_norm_bias,
        attn_backend,
        num_targets,
        num_contextuals,
        target_group_size,
        alpha,
        causal,
        seed,
        residual,
        wgrad_stream,
        wgrad_event,
        recompute_input_layernorm,
        recompute_input_silu,
    )

    return out
