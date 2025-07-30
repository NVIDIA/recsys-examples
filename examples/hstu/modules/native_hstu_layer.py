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

import nvtx
import torch
import torch.nn.functional as F
from commons.utils.nvtx_op import output_nvtx_hook, register_setter_and_getter_for_nvtx
from commons.utils.clear_tensor_data import clear_tensor_data
from configs import HSTUConfig
from configs.hstu_config import HSTULayerType
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import divide
from modules.hstu_attention import create_hstu_attention
from modules.jagged_data import JaggedData
from ops.triton_ops.triton_norm_mul_dropout import triton_norm_mul_dropout


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
        self._residual = config.residual
        device = torch.cuda.current_device()
        if config.learnable_input_layernorm:
            self._input_layernorm_weight = torch.nn.Parameter(
                torch.ones(self._embedding_dim, device=device)
            )
            self._input_layernorm_bias = torch.nn.Parameter(
                torch.zeros(self._embedding_dim, device=device)
            )
        else:
            self._input_layernorm_weight = None
            self._input_layernorm_bias = None
        # output norm weight and bias are mandatory
        # this should be a TP LayerNorm
        self._output_layernorm_weight = torch.nn.Parameter(
            torch.ones(
                self._num_heads_per_partition * self._linear_dim_per_head, device=device
            )
        )
        self._output_layernorm_bias = torch.nn.Parameter(
            torch.zeros(
                self._num_heads_per_partition * self._linear_dim_per_head, device=device
            )
        )
        # [embedding_dim, 4 * num_head * head_dim]
        self._linear_uvqk = TEColumnParallelLinear(
            input_size=self._embedding_dim,
            output_size=self._tp_size * sum(self._split_arg_list) * self._num_heads,
            init_method=config.init_method,
            config=config,
            bias=True,
            gather_output=False,
            skip_bias_add=False,  # note: TEColumnParallelLinear does not support bias fusion!
            is_expert=False,
        )

        # torch.nn.Linear(
        #     self._embedding_dim,
        #     sum(self._split_arg_list),
        #     bias=True,
        # ).apply(init_mlp_weights_optional_bias)

        self._linear_proj = TERowParallelLinear(
            input_size=self._linear_dim_per_head * self._num_heads_per_partition,
            output_size=self._embedding_dim,
            init_method=config.init_method,
            config=config,
            input_is_parallel=True,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        # torch.nn.Linear(
        #     self._linear_dim_per_head * self._num_heads,
        #     self._embedding_dim,
        #     bias=False,
        # ).apply(init_mlp_weights_optional_bias)

        self._eps = config.layernorm_epsilon
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
        mixed_uvqk, _ = self._linear_uvqk(hidden_states)
        # silu will upcast to fp32 in register
        mixed_uvqk = F.silu(mixed_uvqk).view(
            -1, self._num_heads_per_partition, sum(self._split_arg_list)
        )
        (user, value, query, key) = torch.split(
            mixed_uvqk,
            self._split_arg_list,
            dim=-1,
        )
        value = value.reshape(
            -1, self._num_heads_per_partition * self._linear_dim_per_head
        )
        query = query.reshape(
            -1, self._num_heads_per_partition * self._attention_dim_per_head
        )
        key = key.reshape(
            -1, self._num_heads_per_partition * self._attention_dim_per_head
        )
        # this contiguous is inevitable, because output layout is (T, head_dim * 4, num_heads)
        user = user.reshape(
            -1, self._num_heads_per_partition * self._linear_dim_per_head
        )
        clear_tensor_data(mixed_uvqk)
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
        with nvtx.annotate("hstu ln+linear_bias+silu fwd", color="RED"):
            normed_x = F.layer_norm(
                x,
                normalized_shape=[self._embedding_dim],
                weight=self._input_layernorm_weight,
                bias=self._input_layernorm_bias,
                eps=self._eps,
            )
            tu, tv, tq, tk = self.get_user_value_query_key_tensors(normed_x)
        # TODO: remove contiguous once cutlass backend is ready
        with nvtx.annotate("hstu attn fwd", color="BLUE"):
            jagged_attn_output = self._attn_func(
                tq,
                tk,
                tv,
                jd.seqlen_offsets,
                num_contextuals=jd.contextual_seqlen,
                num_candidates=jd.num_candidates,
                max_seqlen=jd.max_seqlen,
                target_group_size=self._target_group_size,
            )
        with nvtx.annotate("hstu norm mul dropout fwd", color="GREEN"):
            parallel_input = triton_norm_mul_dropout(
                jagged_attn_output,
                tu,
                self._output_layernorm_weight,
                self._output_layernorm_bias,
                self._eps,
                self._dropout_ratio,
                self.training,
            )
        with nvtx.annotate("hstu linear_residual fwd", color="YELLOW"):
            # Note that output is a pair of (output, bias)
            output, _ = self._linear_proj(parallel_input)
            if self._residual:
                output = output + x
        return JaggedData(
            values=output,
            seqlen=jd.seqlen,
            seqlen_offsets=jd.seqlen_offsets,
            max_seqlen=jd.max_seqlen,
            max_num_candidates=jd.max_num_candidates,
            num_candidates=jd.num_candidates,
            num_candidates_offsets=jd.num_candidates_offsets,
            contextual_max_seqlen=jd.contextual_max_seqlen,
            contextual_seqlen=jd.contextual_seqlen,
            contextual_seqlen_offsets=jd.contextual_seqlen_offsets,
            has_interleaved_action=jd.has_interleaved_action,
        )
