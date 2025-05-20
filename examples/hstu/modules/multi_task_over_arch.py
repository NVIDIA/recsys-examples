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
# from typing import List

# import torch
# from commons.utils.nvtx_op import output_nvtx_hook
# from megatron.core.transformer.module import MegatronModule
# from modules.jagged_module import JaggedData, JaggedModule


# class MultiTaskOverArch(JaggedModule):
#     """
#     This is used for multi-task training that is a downstream of a generative model.
#     Usually, each task is a DNN.

#     Args:
#         mlp_layers (List[JaggedModule]): List of MLP layers for each task.
#     """

#     def __init__(self, mlp_layers: List[JaggedModule]):
#         super(MegatronModule, self).__init__()

#         self._multi_task_prediction_layers = torch.nn.ModuleList(mlp_layers)

#     @output_nvtx_hook(nvtx_tag="top network", hook_tensor_attr_name="values")
#     def forward(
#         self,
#         jagged_input: JaggedData,
#     ) -> JaggedData:
#         """
#         Forward pass for the MultiTaskOverArch module.

#         Args:
#             jagged_input (JaggedData): Input data with jagged dimensions.

#         Returns:
#             JaggedData: Output data with concatenated logits.
#         """
#         logits = []
#         for task_module in self._multi_task_prediction_layers:
#             logits.append(task_module(jagged_input).values)

#         return JaggedData(
#             # [T, logit_dim_sum]
#             values=torch.concat(logits, dim=-1),
#             # others can be none
#             seqlen=jagged_input.seqlen,
#             seqlen_offsets=jagged_input.seqlen_offsets,
#             max_seqlen=jagged_input.max_seqlen,
#         )
