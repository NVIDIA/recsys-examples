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

import os
from functools import partial

import nvtx
import torch
import torch.nn.functional as F
from commons.ops.collective_ops import gather_along_last_dim, split_along_first_dim
from commons.utils.clear_tensor_data import clear_tensor_data
from commons.utils.nvtx_op import output_nvtx_hook, register_setter_and_getter_for_nvtx
from configs import HSTUConfig
from configs.hstu_config import HSTULayerType
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import divide
from modules.hstu_attention import create_hstu_attention
from modules.jagged_data import JaggedData
from modules.tp_layer_norm import TPLayerNormMulDropout
from ops.triton_ops.triton_layer_norm import triton_layer_norm


class HSTULayer(MegatronModule):
    """
    One basic unit of HSTUBlock. Input and output are all JaggedData.
    This module support TP (TEColumnParallelLinear & TERowParallelLinear). And the uvqk split layout is different from legacy.

    Args:
        config (HSTUConfig): Configuration for the HSTU layer.
    """

    def __init__(self, config: HSTUConfig):
        assert (
            config.hstu_layer_type == HSTULayerType.NATIVE
        ), "HSTULayer expects native hstu layer type"
        self._tp_size = parallel_state.get_tensor_model_parallel_world_size()
        super().__init__(config=config)
        self._embedding_dim: int = config.hidden_size
        # per head dim;
        self._linear_dim_per_head: int = config.kv_channels
        self._attention_dim_per_head: int = config.kv_channels
        self._eps = config.layernorm_epsilon
        # dropout on proj_linear
        self._dropout_ratio: float = config.hidden_dropout
        # dropout on QK; not used now
        self._num_heads: int = config.num_attention_heads
        if self._tp_size > self._num_heads:
            raise ValueError("tp size should <= num_attention_heads")
        self._num_heads_per_partition = divide(self._num_heads, self._tp_size)
        # TODO, support packed qkv attention
        self._split_arg_list = [
            self._linear_dim_per_head,
            self._linear_dim_per_head,
            self._attention_dim_per_head,
            self._attention_dim_per_head,
        ]
        self._recompute_input_layernorm = config.recompute_input_layernorm
        if self._recompute_input_layernorm:
            self.input_layernorm_checkpoint = CheckpointWithoutOutput()
        self._recompute_input_silu = config.recompute_input_silu
        if self._recompute_input_silu:
            self.silu_checkpoint = CheckpointWithoutOutput()
        self._residual = config.residual
        device = torch.cuda.current_device()
        self._sequence_parallel = config.sequence_parallel
        # input layernorm
        if config.learnable_input_layernorm:
            self._input_layernorm_weight = torch.nn.Parameter(
                torch.ones(self._embedding_dim, device=device)
            )
            self._input_layernorm_bias = torch.nn.Parameter(
                torch.zeros(self._embedding_dim, device=device)
            )
            # this flag will be used in finalize_model_grads!
            # see https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0/megatron/core/distributed/finalize_model_grads.py#L286-L288
            self._input_layernorm_weight.sequence_parallel = self._sequence_parallel
            self._input_layernorm_bias.sequence_parallel = self._sequence_parallel
        else:
            self._input_layernorm_weight = None
            self._input_layernorm_bias = None

        self._learnable_output_layernorm = config.learnable_output_layernorm
        self._output_ln_dropout_mul = TPLayerNormMulDropout(
            hidden_size=self._num_heads * self._linear_dim_per_head,
            eps=self._eps,
            trainable=self._learnable_output_layernorm,
            dropout_ratio=self._dropout_ratio,
            fusion=config.fuse_norm_mul_dropout,
            sequence_parallel=config.sequence_parallel,
        )
        # [embedding_dim, 4 * num_head * head_dim]
        self._linear_uvqk = TEColumnParallelLinear(
            input_size=self._embedding_dim,
            output_size=sum(self._split_arg_list) * self._num_heads,
            init_method=config.init_method,
            config=config,  # sequence_parallel is handled in TEColumnParallelLinear, it will help ag along seq dim.
            bias=config.add_uvqk_bias,
            gather_output=False,
            skip_bias_add=False,  # note: TEColumnParallelLinear does not support bias fusion!
            is_expert=False,
        )

        self._debug_shortcut_proj_linear = (
            os.environ.get("DEBUG_SHORTCUT_PROJ_LINEAR", "0") == "1"
        )
        self._debug_shortcut_output_ln_mul_dropout = (
            os.environ.get("DEBUG_SHORTCUT_OUTPUT_LN_MUL_DROPOUT", "0") == "1"
        )
        if self._debug_shortcut_proj_linear:
            assert (
                self._embedding_dim == self._linear_dim_per_head * self._num_heads
            ), "when shortcut proj linear is on, embedding dim must be equal to linear dim per head * num heads"
        self._linear_proj = TERowParallelLinear(
            input_size=self._linear_dim_per_head * self._num_heads,
            output_size=self._embedding_dim,
            init_method=config.init_method,
            config=config,  # sequence_parallel is handled in TEColumnParallelLinear
            input_is_parallel=True,
            bias=False,  # there is no bias
            skip_bias_add=False,
            is_expert=False,
        )

        self._target_group_size = config.target_group_size

        self._attn_func = create_hstu_attention(
            kernel_backend=config.kernel_backend,
            num_heads=self._num_heads_per_partition,
            attention_dim=self._attention_dim_per_head,
            linear_dim=self._linear_dim_per_head,
            is_causal=config.is_causal,
        )
        register_setter_and_getter_for_nvtx(
            HSTULayer.forward, key_or_attr_name="values"
        )

    def get_user_value_query_key_tensors(self, hidden_states: torch.Tensor):
        """
        Splits the hidden states into user, value, query, and key tensors.

        Args:
            hidden_states (torch.Tensor): The hidden states tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The user, value, query, and key tensors.
        """

        # TODO: fuse linear, bias, and silu?
        with nvtx.annotate("hstu linear_uvqk fwd", color="RED"):
            mixed_uvqk, _ = self._linear_uvqk(hidden_states)

        with nvtx.annotate("hstu silu fwd", color="BLUE"):
            if self._recompute_input_silu:
                silu_uvqk = self.silu_checkpoint.checkpoint(
                    F.silu,
                    mixed_uvqk,
                )
            else:
                silu_uvqk = F.silu(mixed_uvqk)
            # silu will upcast to fp32 in register
            silu_uvqk = silu_uvqk.view(
                -1, self._num_heads_per_partition, sum(self._split_arg_list)
            )
        if self._recompute_input_layernorm:
            # when mixed_uvqk grad(silu) is computed, trigger the recompute of the input layernorm
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                mixed_uvqk
            )

        with nvtx.annotate("hstu split fwd", color="BLUE"):
            (user, value, query, key) = torch.split(
                silu_uvqk,
                self._split_arg_list,
                dim=-1,
            )

        clear_tensor_data(silu_uvqk)
        return user, value, query, key

    @output_nvtx_hook(nvtx_tag="HSTULayer")
    def forward(self, jd: JaggedData) -> JaggedData:
        """
        Forward pass of the HSTULayer

        Args:
            jd (JaggedData): The input jagged data

        Returns:
            Tensor: The output embeddings [\*, D]
        """
        # input is [*, h]
        x = jd.values
        with nvtx.annotate("hstu input layernorm fwd", color="RED"):
            if self._recompute_input_layernorm:
                normed_x = self.input_layernorm_checkpoint.checkpoint(
                    partial(triton_layer_norm, eps=self._eps),
                    x,
                    self._input_layernorm_weight,
                    self._input_layernorm_bias,
                )
            else:
                normed_x = F.layer_norm(
                    x,
                    normalized_shape=[self._embedding_dim],
                    weight=self._input_layernorm_weight,
                    bias=self._input_layernorm_bias,
                    eps=self._eps,
                )
        with nvtx.annotate("hstu uvqk linear_silu fwd", color="BLUE"):
            tu, tv, tq, tk = self.get_user_value_query_key_tensors(normed_x)
        # TODO: remove contiguous once cutlass backend is ready
        with nvtx.annotate("hstu attn fwd", color="BLUE"):
            # [ T, hidden_size_per_partition]
            jagged_attn_output = self._attn_func(
                tq,
                tk,
                tv,
                jd.seqlen_offsets,
                num_contextuals=jd.contextual_seqlen,
                num_candidates=jd.num_candidates,
                max_seqlen=jd.max_seqlen,
                scaling_seqlen=jd.scaling_seqlen,
                target_group_size=self._target_group_size,
            )
            # TODO handle the padding in HSTU kernel
            padding_length = jd.padding_length
            if padding_length > 0:
                last_valid_index = jd.seqlen_offsets[-1]
                jagged_attn_output = (
                    jagged_attn_output.clone()
                )  # we need this clone otherwise it will be inplace operation
                jagged_attn_output[
                    last_valid_index : last_valid_index + padding_length, ...
                ] = 0.0

                def reset_padding_grad_hook(grad):
                    grad[
                        last_valid_index : last_valid_index + padding_length, ...
                    ] = 0.0
                    return grad

                jagged_attn_output.register_hook(reset_padding_grad_hook)
        with nvtx.annotate("hstu norm mul dropout fwd", color="GREEN"):
            if self._debug_shortcut_output_ln_mul_dropout:
                parallel_input = jagged_attn_output
            else:
                parallel_input = self._output_ln_dropout_mul(
                    jagged_attn_output, tu
                )  # [ T, hidden_size_per_partition]
        if self._recompute_input_silu:
            # when output grad (gemm dgrad) is computed, trigger the recompute of the silu
            # we discard here after tu is used
            self.silu_checkpoint.discard_output_and_register_recompute(parallel_input)

        with nvtx.annotate("hstu linear_residual fwd", color="YELLOW"):
            # shortcut for debug
            if self._debug_shortcut_proj_linear:
                output = gather_along_last_dim(
                    parallel_input, parallel_state.get_tensor_model_parallel_group()
                )  # [ T , hidden_size]
                if self._sequence_parallel:
                    # scatter
                    output = split_along_first_dim(
                        output, parallel_state.get_tensor_model_parallel_group()
                    )  # [ T / tp_size, hidden_size]
            else:
                output, _ = self._linear_proj(
                    parallel_input
                )  # when sequence parallel is on, the output is scattered

            if self._residual:
                output = output + x
        return JaggedData(
            values=output,
            seqlen=jd.seqlen,
            seqlen_offsets=jd.seqlen_offsets,
            padding_length=jd.padding_length,  # inherit from the input jagged data
            max_seqlen=jd.max_seqlen,
            max_num_candidates=jd.max_num_candidates,
            num_candidates=jd.num_candidates,
            num_candidates_offsets=jd.num_candidates_offsets,
            contextual_max_seqlen=jd.contextual_max_seqlen,
            contextual_seqlen=jd.contextual_seqlen,
            contextual_seqlen_offsets=jd.contextual_seqlen_offsets,
            has_interleaved_action=jd.has_interleaved_action,
            scaling_seqlen=jd.scaling_seqlen,
        )
