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
import torch
from commons.utils.nvtx_op import output_nvtx_hook
from configs import HSTUConfig
from configs.hstu_config import HSTULayerType
from modules.jagged_module import JaggedData, JaggedModule
from ops.fused_hstu_op import fused_hstu_op


class FusedHSTULayer(JaggedModule):
    """
    x = ln(x)
    u,v,q,k = silu(linear_bias(x))
    attn_output = hstu_attn.hstu_attn_varlen_func(q,k,v,offsets,max_seqlen)
    normed_out = ln_mul_dropout(attn_output)
    out = linear_residual(normed_out)

    One basic unit of HSTUBlock. Input and output are all JaggedData.
    hstu_mha is cutlass , others are triton
    """

    def __init__(self, config: HSTUConfig):
        assert (
            config.hstu_layer_type == HSTULayerType.FUSED
        ), "FusedHSTULayer expects fused hstu layer type"
        assert (
            config.tensor_model_parallel_size == 1
        ), "FusedHSTULayer does not support tensor model parallel"
        super().__init__(config=config)
        self._embedding_dim: int = config.hidden_size
        # per head dim;
        self._linear_dim_per_head: int = config.kv_channels
        self._attention_dim_per_head: int = config.kv_channels

        # dropout
        self._seed = None
        # dropout on proj_linear output
        self._dropout_ratio: float = config.hidden_dropout
        # dropout on QK; not used now
        self._num_heads: int = config.num_attention_heads

        self._eps = config.layernorm_epsilon
        self._is_causal = config.is_causal
        self._target_group_size = config.target_group_size
        self._alpha = 1.0
        self._residual = config.residual
        self._attn_backend = config.kernel_backend

        # stream and event are shared across all layers
        self._wgrad_stream = config.async_wgrad_stream
        self._wgrad_event = config.async_wgrad_event
        # all weights and biases are float32 unless module.to(dtype) is called
        self._linear_uvqk_weight = torch.nn.Parameter(
            torch.empty(
                (
                    self._embedding_dim,
                    (self._linear_dim_per_head * 2 + self._attention_dim_per_head * 2)
                    * self._num_heads,
                )
            ),
        )
        self._linear_uvqk_bias = torch.nn.Parameter(
            torch.zeros(
                (
                    (self._linear_dim_per_head * 2 + self._attention_dim_per_head * 2)
                    * self._num_heads,
                )
            )
        )
        torch.nn.init.xavier_uniform_(self._linear_uvqk_weight)

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
        self._output_layernorm_weight = torch.nn.Parameter(
            torch.ones(self._num_heads * self._linear_dim_per_head, device=device)
        )
        self._output_layernorm_bias = torch.nn.Parameter(
            torch.zeros(self._num_heads * self._linear_dim_per_head, device=device)
        )

        # linear_proj no bias
        self._linear_proj_weight = torch.nn.Parameter(
            torch.empty(
                (
                    self._linear_dim_per_head * self._num_heads,
                    self._embedding_dim,
                )
            )
        )

        torch.nn.init.xavier_uniform_(self._linear_proj_weight)

    @output_nvtx_hook(nvtx_tag="FusedHSTULayer", hook_tensor_attr_name="values")
    def forward(self, jd: JaggedData) -> JaggedData:
        input = jd.values
        output = fused_hstu_op(
            input=input,
            seqlen_offsets=jd.seqlen_offsets,
            max_seqlen=jd.max_seqlen,
            # linear weights and biases
            linear_uvqk_weight=self._linear_uvqk_weight,
            linear_uvqk_bias=self._linear_uvqk_bias,
            linear_proj_weight=self._linear_proj_weight,
            num_heads=self._num_heads,
            linear_dim_per_head=self._linear_dim_per_head,
            attention_dim_per_head=self._attention_dim_per_head,
            ln_eps=self._eps,
            dropout_ratio=self._dropout_ratio,
            training=self.training,
            # layer norm weight and bias
            input_norm_weight=self._input_layernorm_weight,
            input_norm_bias=self._input_layernorm_bias,
            output_norm_weight=self._output_layernorm_weight,
            output_norm_bias=self._output_layernorm_bias,
            # attn related
            attn_backend=self._attn_backend,
            num_targets=jd.num_candidates,
            num_contextuals=jd.contextual_seqlen,
            target_group_size=self._target_group_size,
            alpha=self._alpha,
            causal=self._is_causal,
            seed=self._seed,
            residual=self._residual,
            wgrad_stream=self._wgrad_stream,
            wgrad_event=self._wgrad_event,
        )
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
