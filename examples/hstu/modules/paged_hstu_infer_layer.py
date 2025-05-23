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
import torch.nn.functional as F
import flashinfer
from commons.utils.nvtx_op import output_nvtx_hook
from configs import HSTUConfig
from configs.hstu_config import HSTULayerType
from modules.jagged_module import JaggedData, JaggedModule
from ops.triton_ops.triton_paged_hstu_attention import (
    triton_history_hstu_mha,
    triton_candidate_hstu_mha
)


class PagedHSTUInferLayer(JaggedModule):
    """
    x = ln(x)
    u,v,q,k = silu(linear_bias(x))
    attn_output = hstu_attn.hstu_attn_varlen_func(q,k,v,offsets,max_seqlen)
    normed_out = ln_mul_dropout(attn_output)
    out = linear_residual(normed_out)

    One basic unit of PagedHSTUBlock. Input and output are all JaggedData.
    """

    def __init__(self, config: HSTUConfig, layer_idx: int):
        assert (
            config.tensor_model_parallel_size == 1
        ), "PagedHSTUBlock does not support tensor model parallel"
        super().__init__(config=config)
        self.layer_idx = layer_idx
        self._embedding_dim: int = config.hidden_size
        # per head dim;
        self._linear_dim_per_head: int = config.kv_channels
        self._attention_dim_per_head: int = config.kv_channels

        self._num_heads: int = config.num_attention_heads

        self._eps = config.layernorm_epsilon
        self._is_causal = config.is_causal
        self._target_group_size = config.target_group_size
        self._alpha = 1.0
        self._residual = config.residual

        self._split_arg_list = [
            self._linear_dim_per_head * self._num_heads,
            self._linear_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
        ]

        dtype = torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else torch.float32
        device = torch.cuda.current_device()

        # linear_uvqk
        self._linear_uvqk = torch.nn.Linear(
            self._embedding_dim,
            (self._linear_dim_per_head * 2 + self._attention_dim_per_head * 2)
            * self._num_heads,
            bias=True,
            dtype=dtype,
        )

        # input norm 
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
        
        # output norm 
        self._output_layernorm_weight = torch.nn.Parameter(
            torch.ones(self._num_heads * self._linear_dim_per_head, device=device)
        )
        self._output_layernorm_bias = torch.nn.Parameter(
            torch.zeros(self._num_heads * self._linear_dim_per_head, device=device)
        )

        # linear_proj
        self._linear_proj = torch.nn.Linear(
            self._linear_dim_per_head * self._num_heads,
            self._embedding_dim,
            bias=False,
            dtype=dtype,
        )

        # self.paged_attn_func
    
    def load_variable(self,):
        pass

    @torch.inference_mode()
    def forward(self, jd: JaggedData, kv_cache_handler) -> JaggedData:
        input = jd.values
        normed_input = F.layer_norm(
            input,
            normalized_shape=[self._embedding_dim],
            weight=self._input_layernorm_weight,
            bias=self._input_layernorm_bias,
            eps=self._eps,
        )
        
        mixed_uvqk = F.silu(self._linear_uvqk(normed_input))
        (user, value, query, key) = torch.split(
            mixed_uvqk,
            self._split_arg_list,
            dim=-1,
        )

        value = value.view(-1, self._num_heads, self._linear_dim_per_head)
        query = query.view(-1, self._num_heads, self._attention_dim_per_head)
        key = key.view(-1, self._num_heads, self._attention_dim_per_head)

        (gpu_kv_cache_manager, kv_cache_metadata) = kv_cache_handler
        gpu_kv_cache_manager.append_paged_kv_data(
            self.layer_idx, 
            key, 
            value, 
            kv_cache_metadata,
            jd.seqlen_offsets,
            jd.num_candidates)

        jagged_attn_output = torch.zeros_like(value)
        triton_history_hstu_mha(
            jd.max_seqlen,
            self._alpha,
            query,
            jagged_attn_output,
            kv_cache_metadata.max_delta_history_length,
            gpu_kv_cache_manager.get_buffers(self.layer_idx),
            kv_cache_metadata.kv_indices,
            kv_cache_metadata.kv_indptr,
            kv_cache_metadata.kv_last_page_len,
            gpu_kv_cache_manager.get_page_size(),
            jd.seqlen_offsets,
            jd.num_candidates,
            kv_cache_metadata.total_history_lengths,
            False,
        )
        triton_candidate_hstu_mha(
            jd.max_seqlen,
            self._alpha,
            query,
            key,
            value,
            jagged_attn_output,
            kv_cache_metadata.max_num_candidate,
            gpu_kv_cache_manager.get_buffers(self.layer_idx),
            kv_cache_metadata.kv_indices,
            kv_cache_metadata.kv_indptr,
            kv_cache_metadata.kv_last_page_len,
            gpu_kv_cache_manager.get_page_size(),
            jd.seqlen_offsets,
            jd.num_candidates,
            kv_cache_metadata.total_history_lengths,
            False,
        )
        jagged_attn_output = jagged_attn_output.view(-1, self._num_heads*self._linear_dim_per_head)

        parallel_input = user * F.layer_norm(
            jagged_attn_output,
            normalized_shape=jagged_attn_output.shape[1:],
            weight=self._output_layernorm_weight.to(torch.bfloat16),
            bias=self._output_layernorm_bias.to(torch.bfloat16),
            eps=self._eps,
        )

        output = self._linear_proj(parallel_input)
        if self._residual:
            output = output + input
        
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
