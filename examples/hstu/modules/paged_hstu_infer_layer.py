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
import hstu_attn
import paged_kvcache_ops
import torch
import torch.nn.functional as F
from configs import InferenceHSTUConfig, KVCacheConfig
from modules.jagged_data import JaggedData
from ops.pt_ops.torch_addmm import torch_addmm_silu_fwd
from ops.triton_ops.triton_addmm import triton_addmm_silu_fwd
from ops.triton_ops.triton_layer_norm import triton_weighted_layer_norm_fwd
from ops.triton_ops.triton_norm_mul_dropout import triton_layer_norm_mul_dropout_fwd

import numpy as np

def init():
    global dmp
    dmp = False

class PagedHSTUInferLayer(torch.nn.Module):
    """
    x = ln(x)
    u,v,q,k = silu(linear_bias(x))
    attn_output = hstu_attn.hstu_attn_varlen_func(q,k,v,offsets,max_seqlen)
    normed_out = ln_mul_dropout(attn_output)
    out = linear_residual(normed_out)

    One basic unit of PagedHSTUBlock. Input and output are all JaggedData.
    """

    def __init__(
        self,
        config: InferenceHSTUConfig,
        kv_cache_config: KVCacheConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._embedding_dim: int = config.hidden_size
        # per head dim;
        self._linear_dim_per_head: int = config.head_dim
        self._attention_dim_per_head: int = config.head_dim

        self._num_heads: int = config.num_heads

        self._eps = config.layernorm_epsilon
        self._is_causal = config.is_causal
        self._target_group_size = config.target_group_size
        self._alpha = 1.0 / (self._attention_dim_per_head**0.5)
        self._residual = config.residual

        self._split_arg_list = [
            self._linear_dim_per_head * self._num_heads,
            self._linear_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
        ]
        self._max_seqlen = kv_cache_config.max_seq_len

        dtype = (
            torch.bfloat16
            if config.bf16
            else torch.float16
            if config.fp16
            else torch.float32
        )
        device = torch.cuda.current_device()

        # linear_uvqk
        self._linear_uvqk = torch.nn.Linear(
            self._embedding_dim,
            (self._linear_dim_per_head * 2 + self._attention_dim_per_head * 2)
            * self._num_heads,
            bias=True,
            dtype=dtype,
            device=device,
        )
        for param in self._linear_uvqk.parameters():
            param.requires_grad = False
            param.copy_(torch.empty_like(param).uniform_(-0.5, 0.5))
        self._linear_uvqk_weight = self._linear_uvqk.weight.T.contiguous()

        # input norm
        if config.learnable_input_layernorm:
            self._input_layernorm_weight = torch.nn.Parameter(
                torch.ones(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
            self._input_layernorm_bias = torch.nn.Parameter(
                torch.zeros(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self._input_layernorm_weight = None
            self._input_layernorm_bias = None

        # output norm
        self._output_layernorm_weight = torch.nn.Parameter(
            torch.ones(
                self._num_heads * self._linear_dim_per_head, dtype=dtype, device=device
            ),
            requires_grad=False,
        )
        self._output_layernorm_bias = torch.nn.Parameter(
            torch.zeros(
                self._num_heads * self._linear_dim_per_head, dtype=dtype, device=device
            ),
            requires_grad=False,
        )

        # linear_proj
        self._linear_proj = torch.nn.Linear(
            self._linear_dim_per_head * self._num_heads,
            self._embedding_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )

        for param in self._linear_proj.parameters():
            param.requires_grad = False
            param.copy_(torch.randn_like(param))
        self._linear_proj_weight = self._linear_proj.weight.T.contiguous()

        # output buffer
        max_num_tokens = kv_cache_config.max_batch_size * kv_cache_config.max_seq_len
        self.output_buffer_ = torch.empty(
            (max_num_tokens, config.hidden_size),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        self.uvqk_buffer_ = torch.empty(
            (
                max_num_tokens,
                (self._linear_dim_per_head * 2 + self._attention_dim_per_head * 2)
                * self._num_heads,
            ),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

        sm = torch.cuda.get_device_properties(0).major
        if sm == 8:
            self.addmm_silu_impl = triton_addmm_silu_fwd
        elif sm == 9:
            self.addmm_silu_impl = torch_addmm_silu_fwd
        else:
            raise ValueError(f"Unsupported SM major version: {sm}")

    def uvqk_addmm_impl(self, input_data, num_tokens):
        if num_tokens >= 2048:  # fusion impl
            _, silu_output_data = self.addmm_silu_impl(
                x=input_data,
                w=self._linear_uvqk_weight,  # transposed
                y=self._linear_uvqk.bias,
                silu=True,
            )
        else:  # non fusion impl
            silu_output_data = self._linear_uvqk(input_data)
            F.silu(silu_output_data, inplace=True)

        return silu_output_data

    def uvqk_addmm_inplace_impl(self, input_data, silu_output_data, num_tokens):
        if num_tokens >= 1024:  # fusion impl
            self.addmm_silu_impl(
                x=input_data,
                w=self._linear_uvqk_weight,  # transposed
                y=self._linear_uvqk.bias,
                silu=True,
                keep_unfused_out=False,
                silu_out=silu_output_data,
            )
        else:  # non fusion impl
            torch.addmm(
                self._linear_uvqk.bias,
                input_data,
                self._linear_uvqk_weight,
                out=silu_output_data,
            )
            F.silu(silu_output_data, inplace=True)

    def proj_addmm_impl(self, input_data, residual, num_tokens):
        if num_tokens >= 2048:  # fusion impl
            output_data, _ = self.addmm_silu_impl(
                x=input_data,
                w=self._linear_proj_weight,  # transposed
                y=residual,
                silu=False,
            )
        else:  # non fusion impl
            output_data = self._linear_proj(input_data)
            torch.add(output_data, residual, out=output_data)

            # output_data = torch.addmm(residual, input_data, self._linear_proj_weight)

        return output_data

    def proj_addmm_inplace_impl(self, input_data, residual, output_data, num_tokens):
        if num_tokens >= 1024:  # fusion impl
            self.addmm_silu_impl(
                x=input_data,
                w=self._linear_proj_weight,  # transposed
                y=residual,
                silu=False,
                out=output_data,
            )
        else:  # non fusion impl
            torch.add(self._linear_proj(input_data), residual, out=output_data)

            # output_data = torch.addmm(residual, input_data, self._linear_proj_weight)

        return output_data

    def norm_mul_impl(self, jagged_attn_output, user, enable_fusion):
        if enable_fusion:  # fusion impl
            parallel_input, _, _, _, _, _ = triton_layer_norm_mul_dropout_fwd(
                x=jagged_attn_output,
                u=user,
                weight=self._output_layernorm_weight,
                bias=self._output_layernorm_bias,
                eps=self._eps,
                dropout_ratio=0.0,
                training=False,
            )
        else:  # non fusion impl
            parallel_input = user * F.layer_norm(
                jagged_attn_output,
                normalized_shape=[self._num_heads * self._linear_dim_per_head],
                weight=self._output_layernorm_weight,
                bias=self._output_layernorm_bias,
                eps=self._eps,
            )

        return parallel_input

    def layer_output(num_tokens):
        return self.output_buffer_[:num_tokens, ...]

    def load_variable(
        self,
    ):
        pass

    @torch.inference_mode()
    def forward_naive(
        self,
        batch_size: int,
        num_tokens: int,
        layer_input: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> JaggedData:
        normed_input = F.layer_norm(
            layer_input,
            normalized_shape=[self._embedding_dim],
            weight=self._input_layernorm_weight,
            bias=self._input_layernorm_bias,
            eps=self._eps,
        )

        # mixed_uvqk = self.uvqk_addmm_impl(normed_input, 0)
        mixed_uvqk = F.silu(torch.matmul(normed_input, self._linear_uvqk_weight) + self._linear_uvqk.bias)
        (user, value, query, key) = torch.split(
            mixed_uvqk,
            self._split_arg_list,
            dim=-1,
        )

        value = value.view(-1, self._num_heads, self._linear_dim_per_head)
        query = query.view(-1, self._num_heads, self._attention_dim_per_head)
        key = key.view(-1, self._num_heads, self._attention_dim_per_head)

        if kv_cache_metadata is not None:
            kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
            (paged_k_cache, paged_v_cache) = kv_cache_table.unbind(dim=1)
            paged_kvcache_ops.append_kvcache(
                key,
                value,
                kv_cache_metadata.batch_indices,
                kv_cache_metadata.position,
                jd.num_candidates_offsets[: batch_size + 1],
                kv_cache_metadata.new_history_nnz_cuda,
                kv_cache_metadata.new_history_nnz,
                paged_k_cache,
                paged_v_cache,
                kv_cache_metadata.kv_indices,
                kv_cache_metadata.kv_indptr,
                kv_cache_metadata.kv_last_page_len,
                0,  # NHD layout
            )

            kv_cache_metadata.kv_onload_handle.wait_host(self.layer_idx)
            jagged_attn_output = hstu_attn.hstu_attn_varlen_func(
                query,
                key,
                value,
                jd.seqlen_offsets[: batch_size + 1],
                kv_cache_metadata.total_history_offsets[: batch_size + 1],
                jd.max_seqlen,
                jd.max_seqlen, # kv_cache_metadata.max_seqlen,
                num_contexts = None,
                num_targets=jd.num_candidates[:batch_size],
                target_group_size=1,
                window_size=(-1, 0),
                alpha=self._alpha,
                rab=None,
                has_drab=False,
                kv_cache=kv_cache_table,
                page_offsets=kv_cache_metadata.kv_indptr,
                page_ids=kv_cache_metadata.kv_indices,
                last_page_lens=kv_cache_metadata.kv_last_page_len,
                cu_seqlens_t=jd.num_candidates_offsets[: batch_size + 1],
                scaling_seqlen=jd.scaling_seqlen,
            )
        else:
            jagged_attn_output = hstu_attn.hstu_attn_varlen_func(
                query,
                key,
                value,
                jd.seqlen_offsets[: batch_size + 1],
                jd.seqlen_offsets[: batch_size + 1],
                jd.max_seqlen,
                jd.max_seqlen,
                num_contexts = None,
                num_targets=jd.num_candidates[:batch_size],
                target_group_size=1,
                window_size=(-1, 0),
                alpha=self._alpha,
                rab=None,
                has_drab=False,
                scaling_seqlen=jd.scaling_seqlen,
            )


        jagged_attn_output = jagged_attn_output.view(
            -1, self._num_heads * self._linear_dim_per_head
        )

        parallel_input = self.norm_mul_impl(
            jagged_attn_output, user, False #num_tokens >= 2048
        )

        if self._residual:
            # layer_output = self.proj_addmm_impl(parallel_input, layer_input, 0)
            layer_output = (torch.matmul(parallel_input, self._linear_proj_weight) + layer_input)
        else:
            # layer_output = self._linear_proj(parallel_input)
            layer_output = torch.matmul(parallel_input, self._linear_proj_weight)

        return layer_output

    @torch.inference_mode()
    def forward_input(
        self,
        batch_size: int,
        num_tokens: int,
        input_buffer: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> JaggedData:
        input_tensor = input_buffer[:num_tokens, ...]
        normed_input, _, _, _, _ = triton_weighted_layer_norm_fwd(
            x=input_tensor,
            weight=self._input_layernorm_weight,
            bias=self._input_layernorm_bias,
            eps=self._eps,
        )

        self.uvqk_addmm_inplace_impl(
            normed_input, self.uvqk_buffer_[:num_tokens, ...], num_tokens
        )
        (user, value, query, key) = torch.split(
            self.uvqk_buffer_[:num_tokens, ...],
            self._split_arg_list,
            dim=-1,
        )

        value = value.view(-1, self._num_heads, self._linear_dim_per_head)
        key = key.view(-1, self._num_heads, self._attention_dim_per_head)

        kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
        (paged_k_cache, paged_v_cache) = kv_cache_table.unbind(dim=1)
        paged_kvcache_ops.append_kvcache(
            key,
            value,
            kv_cache_metadata.batch_indices,
            kv_cache_metadata.position,
            jd.num_candidates_offsets[: batch_size + 1],
            kv_cache_metadata.new_history_nnz_cuda,
            num_tokens,  # kv_cache_metadata.new_history_nnz
            paged_k_cache,
            paged_v_cache,
            kv_cache_metadata.kv_indices,
            kv_cache_metadata.kv_indptr,
            kv_cache_metadata.kv_last_page_len,
            0,  # NHD layout
        )

        return self.uvqk_buffer_[:num_tokens, ...]

    @torch.inference_mode()
    def forward_output(
        self,
        batch_size: int,
        num_tokens: int,
        input_buffer: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> JaggedData:
        (user, value, query, key) = torch.split(
            self.uvqk_buffer_[:num_tokens, ...],
            self._split_arg_list,
            dim=-1,
        )

        value = value.view(-1, self._num_heads, self._linear_dim_per_head)
        query = query.view(-1, self._num_heads, self._attention_dim_per_head)
        key = key.view(-1, self._num_heads, self._attention_dim_per_head)

        kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
        jagged_attn_output = hstu_attn.hstu_attn_varlen_func(
            query,
            key,
            value,
            jd.seqlen_offsets[: batch_size + 1],
            kv_cache_metadata.total_history_offsets[: batch_size + 1],
            self._max_seqlen,
            self._max_seqlen,
            num_contexts=jd.contextual_seqlen[:batch_size]
            if jd.contextual_seqlen is not None
            else None,
            num_targets=jd.num_candidates[:batch_size],
            target_group_size=1,
            window_size=(-1, 0),
            alpha=self._alpha,
            rab=None,
            has_drab=False,
            kv_cache=kv_cache_table,
            page_offsets=kv_cache_metadata.kv_indptr,
            page_ids=kv_cache_metadata.kv_indices,
            last_page_lens=kv_cache_metadata.kv_last_page_len,
            cu_seqlens_t=jd.num_candidates_offsets[: batch_size + 1],
            scaling_seqlen=self._max_seqlen,
        )

        jagged_attn_output = jagged_attn_output.view(
            -1, self._num_heads * self._linear_dim_per_head
        )
        parallel_input = self.norm_mul_impl(jagged_attn_output, user, True)

        if self._residual:
            self.proj_addmm_inplace_impl(
                parallel_input,
                input_buffer[:num_tokens, ...],
                self.output_buffer_[:num_tokens, ...],
                num_tokens,
            )
        else:
            self.output_buffer_[:num_tokens, ...] = self._linear_proj(parallel_input)

        return self.output_buffer_[:num_tokens, ...]
