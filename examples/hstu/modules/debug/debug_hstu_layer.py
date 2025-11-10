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
from typing import Callable

import nvtx
import torch
import torch.nn.functional as F
from commons.utils.distributed_utils import (
    collective_assert_tensor,
    grad_collective_equal_assert_hook,
)
from commons.utils.nvtx_op import output_nvtx_hook
from configs import HSTUConfig
from configs.hstu_config import HSTULayerType
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from modules.hstu_attention import create_hstu_attention
from modules.jagged_data import JaggedData
from modules.utils import init_mlp_weights_optional_bias
from ops.collective_ops import gather_along_last_dim, split_along_last_dim
from ops.pt_ops.pt_norm_mul_dropout import pytorch_norm_mul_dropout
from ops.triton_ops.triton_norm_mul_dropout import triton_norm_mul_dropout


def _mock_tp_output_ln_mul_dropout(
    jagged_attn_output: torch.Tensor,
    tu: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    norm_mul_dropout_func: Callable,
) -> torch.Tensor:
    jagged_attn_output = gather_along_last_dim(
        jagged_attn_output, parallel_state.get_tensor_model_parallel_group()
    )
    tu = gather_along_last_dim(
        tu.contiguous(), parallel_state.get_tensor_model_parallel_group()
    )
    full_output = norm_mul_dropout_func(
        jagged_attn_output,
        tu,
        weight,
        bias,
        eps,
        dropout_ratio,
        training,
    )
    this_rank_output = split_along_last_dim(
        full_output, parallel_state.get_tensor_model_parallel_group()
    )
    return this_rank_output


# TODO move this layer to test folder! And remove the layer type: DEBUG
class HSTULayer(MegatronModule):
    """
    One basic unit of HSTUBlock. Input and output are all JaggedData.
    This module does not support TP.

    Args:
        config (HSTUConfig): Configuration for the HSTU layer.
    """

    def __init__(
        self,
        config: HSTUConfig,
    ):
        assert (
            config.hstu_layer_type == HSTULayerType.DEBUG
        ), "HSTULayer expects native hstu layer type"
        super().__init__(config=config)
        self._embedding_dim: int = config.hidden_size
        # per head dim;
        self._linear_dim_per_head: int = config.kv_channels
        self._attention_dim_per_head: int = config.kv_channels
        # dropout on proj_linear
        self._dropout_ratio: float = config.hidden_dropout
        # dropout on QK; not used now
        self._num_heads: int = config.num_attention_heads

        self._debug_mock_tp = os.environ.get("DEBUG_MOCK_TP", "0") == "1"
        self._debug_check_tp_equal = os.environ.get("DEBUG_CHECK_TP_EQUAL", "0") == "1"
        if self._debug_mock_tp:
            assert (
                not self._debug_check_tp_equal
            ), "when mock tp is on, tp equality check is not available"
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

        self._tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self._tp_size = parallel_state.get_tensor_model_parallel_world_size()

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

        self._output_layernorm_weight = (
            torch.nn.Parameter(
                torch.ones(self._num_heads * self._linear_dim_per_head, device=device)
            )
            if config.learnable_output_layernorm
            else None
        )

        self._output_layernorm_bias = (
            torch.nn.Parameter(
                torch.zeros(self._num_heads * self._linear_dim_per_head, device=device)
            )
            if config.learnable_output_layernorm
            else None
        )
        # [embedding_dim, 4 * num_head * head_dim]
        self._linear_uvqk = torch.nn.Linear(
            self._embedding_dim,
            sum(self._split_arg_list) * self._num_heads,
            bias=config.add_uvqk_bias,
        ).apply(init_mlp_weights_optional_bias)

        self._linear_proj = torch.nn.Linear(
            self._linear_dim_per_head * self._num_heads,
            self._embedding_dim,
            bias=False,
        ).apply(init_mlp_weights_optional_bias)

        self._eps = config.layernorm_epsilon
        self._target_group_size = config.target_group_size

        self._attn_func = create_hstu_attention(
            kernel_backend=config.kernel_backend,
            num_heads=self._num_heads
            if not self._debug_mock_tp
            else self._num_heads // self._tp_size,
            attention_dim=self._attention_dim_per_head,
            linear_dim=self._linear_dim_per_head,
            is_causal=config.is_causal,
        )
        self._fuse_norm_mul_dropout = config.fuse_norm_mul_dropout
        self._norm_mul_dropout_func = (
            pytorch_norm_mul_dropout
            if not self._fuse_norm_mul_dropout
            else triton_norm_mul_dropout
        )

        # register hook to check if the param is bit-wise equal across tp ranks
        if self._debug_check_tp_equal:
            for name, param in self.named_parameters():
                if isinstance(param.data, torch.Tensor):
                    param.register_hook(
                        partial(
                            grad_collective_equal_assert_hook,
                            pg=parallel_state.get_tensor_model_parallel_group(),
                            msg=f"param {name}",
                        )
                    )

    def get_user_value_query_key_tensors(self, hidden_states: torch.Tensor):
        """
        Splits the hidden states into user, value, query, and key tensors.

        Args:
            hidden_states (torch.Tensor): The hidden states tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The user, value, query, and key tensors.
        """

        if self._debug_mock_tp:
            slice_size_per_rank = self._linear_uvqk.weight.shape[0] // self._tp_size
            this_rank_weight_slice_start = self._tp_rank * slice_size_per_rank
            this_rank_weight_slice_end = (self._tp_rank + 1) * slice_size_per_rank
            current_tp_weight = self._linear_uvqk.weight[
                this_rank_weight_slice_start:this_rank_weight_slice_end, ...
            ]
            current_tp_bias = (
                self._linear_uvqk.bias[
                    this_rank_weight_slice_start:this_rank_weight_slice_end
                ]
                if self._linear_uvqk.bias is not None
                else None
            )
            if current_tp_bias is not None:
                mixed_uvqk = torch.addmm(
                    current_tp_bias, hidden_states, current_tp_weight.t()
                )
                # we need to reduce the grad from tp ranks
            else:
                mixed_uvqk = torch.matmul(hidden_states, current_tp_weight.t())
            hidden_states.register_hook(
                lambda grad: reduce_from_tensor_model_parallel_region(grad)
                if grad is not None
                else None
            )
            num_heads_compute = self._num_heads // self._tp_size
        else:
            mixed_uvqk = self._linear_uvqk(hidden_states)
            num_heads_compute = self._num_heads

        if self._debug_check_tp_equal:
            # fwd
            collective_assert_tensor(
                mixed_uvqk,
                compare_type="equal",
                msg="linear uvqk fwd",
                pg=parallel_state.get_tensor_model_parallel_group(),
            )
            # bwd
            hidden_states.register_hook(
                partial(
                    grad_collective_equal_assert_hook,
                    pg=parallel_state.get_tensor_model_parallel_group(),
                    msg="linear uvqk dgrad",
                )
            )
        # maybe elevate to fp32 for higher precision
        mixed_uvqk = F.silu(mixed_uvqk).view(
            -1, num_heads_compute, sum(self._split_arg_list)
        )
        (user, value, query, key) = torch.split(
            mixed_uvqk,
            self._split_arg_list,
            dim=-1,
        )
        value = value.reshape(-1, num_heads_compute * self._linear_dim_per_head)
        query = query.reshape(-1, num_heads_compute * self._attention_dim_per_head)
        key = key.reshape(-1, num_heads_compute * self._attention_dim_per_head)
        user = user.reshape(-1, num_heads_compute * self._linear_dim_per_head)
        del mixed_uvqk
        return user, value, query, key

    @output_nvtx_hook(nvtx_tag="HSTULayer", hook_key_or_attr_name="values")
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
            if self._debug_check_tp_equal:
                # fwd
                collective_assert_tensor(
                    normed_x,
                    compare_type="equal",
                    msg="ln fwd",
                    pg=parallel_state.get_tensor_model_parallel_group(),
                )
                # bwd
                x.register_hook(
                    partial(
                        grad_collective_equal_assert_hook,
                        pg=parallel_state.get_tensor_model_parallel_group(),
                        msg="ln dgrad",
                    )
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
                scaling_seqlen=jd.scaling_seqlen,
                target_group_size=self._target_group_size,
            )
            if self._debug_check_tp_equal:
                # fwd
                collective_assert_tensor(
                    jagged_attn_output,
                    compare_type="equal",
                    msg="attn fwd",
                    pg=parallel_state.get_tensor_model_parallel_group(),
                )
                # bwd
                tq.register_hook(
                    partial(
                        grad_collective_equal_assert_hook,
                        pg=parallel_state.get_tensor_model_parallel_group(),
                        msg="tq dgrad",
                    )
                )
                tk.register_hook(
                    partial(
                        grad_collective_equal_assert_hook,
                        pg=parallel_state.get_tensor_model_parallel_group(),
                        msg="tk dgrad",
                    )
                )
                tv.register_hook(
                    partial(
                        grad_collective_equal_assert_hook,
                        pg=parallel_state.get_tensor_model_parallel_group(),
                        msg="tv dgrad",
                    )
                )
                tu.register_hook(
                    partial(
                        grad_collective_equal_assert_hook,
                        pg=parallel_state.get_tensor_model_parallel_group(),
                        msg="tu dgrad",
                    )
                )
        with nvtx.annotate("hstu norm mul dropout fwd", color="GREEN"):
            output_layernorm_weight = self._output_layernorm_weight
            output_layernorm_bias = self._output_layernorm_bias
            if self._debug_shortcut_output_ln_mul_dropout:
                parallel_input = jagged_attn_output
            elif self._debug_mock_tp:
                parallel_input = _mock_tp_output_ln_mul_dropout(
                    jagged_attn_output,
                    tu,
                    output_layernorm_weight,
                    output_layernorm_bias,
                    self._eps,
                    self._dropout_ratio,
                    self.training,
                    self._norm_mul_dropout_func,
                )
            else:
                parallel_input = self._norm_mul_dropout_func(
                    jagged_attn_output,
                    tu,
                    output_layernorm_weight,
                    output_layernorm_bias,
                    self._eps,
                    self._dropout_ratio,
                    self.training,
                )
            if self._debug_check_tp_equal:
                # fwd
                collective_assert_tensor(
                    parallel_input,
                    compare_type="equal",
                    msg="ln mul dropout fwd",
                    pg=parallel_state.get_tensor_model_parallel_group(),
                )
                # bwd
                jagged_attn_output.register_hook(
                    partial(
                        grad_collective_equal_assert_hook,
                        pg=parallel_state.get_tensor_model_parallel_group(),
                        msg="ln mul dropout dgrad",
                    )
                )
        with nvtx.annotate("hstu linear_residual fwd", color="YELLOW"):
            if self._debug_mock_tp and not self._debug_shortcut_proj_linear:
                slice_size_per_rank = self._linear_proj.weight.shape[1] // self._tp_size
                this_rank_weight_slice_start = self._tp_rank * slice_size_per_rank
                this_rank_weight_slice_end = (self._tp_rank + 1) * slice_size_per_rank
                this_rank_weight = self._linear_proj.weight[
                    ..., this_rank_weight_slice_start:this_rank_weight_slice_end
                ]
                output = torch.matmul(parallel_input, this_rank_weight.t())
                output = reduce_from_tensor_model_parallel_region(output)
            if self._debug_mock_tp and self._debug_shortcut_proj_linear:
                output = gather_along_last_dim(
                    parallel_input, parallel_state.get_tensor_model_parallel_group()
                )

            # this is the regular/default behavior
            if not self._debug_mock_tp and not self._debug_shortcut_proj_linear:
                output = self._linear_proj(parallel_input)
            if not self._debug_mock_tp and self._debug_shortcut_proj_linear:
                output = parallel_input
            if self._debug_check_tp_equal:
                # fwd
                collective_assert_tensor(
                    output,
                    compare_type="equal",
                    msg="proj linear fwd",
                    pg=parallel_state.get_tensor_model_parallel_group(),
                )
                # bwd
                parallel_input.register_hook(
                    partial(
                        grad_collective_equal_assert_hook,
                        pg=parallel_state.get_tensor_model_parallel_group(),
                        msg="proj linear dgrad",
                    )
                )
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
            scaling_seqlen=jd.scaling_seqlen,
        )
