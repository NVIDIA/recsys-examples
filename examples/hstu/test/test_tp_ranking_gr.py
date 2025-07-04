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
import re
from typing import Dict, List

import commons.utils as init
import pytest
import torch
from configs import HSTULayerType
from megatron.core import parallel_state
from test_utils import (
    collective_assert_tensor,
    create_model,
    debug_module_path_to_tpN_module_path,
)
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)


# @pytest.mark.parametrize("tp_size", [2, 4, 8, 1])
@pytest.mark.parametrize("tp_size", [2])
def test_gr_tp_ranking_initialization(tp_size: int):
    contextual_feature_names: List[str] = []
    max_num_candidates: int = 10
    # we must use static embedding for
    use_dynamic_emb: bool = False
    pipeline_type: str = "none"
    optimizer_type_str: str = "sgd"
    dtype: torch.dtype = torch.bfloat16

    init.initialize_distributed()
    init.initialize_model_parallel(tp_size)
    debug_model, dense_optimizer, history_batches = create_model(
        task_type="ranking",
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        use_dynamic_emb=use_dynamic_emb,
        pipeline_type="none",
        dtype=dtype,
        seed=1234,
        hstu_layer_type=HSTULayerType.DEBUG,
    )
    tp_model, tp_dense_optimizer, _ = create_model(
        task_type="ranking",
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        dtype=dtype,
        use_dynamic_emb=use_dynamic_emb,
        pipeline_type=pipeline_type,
        seed=1234,
        hstu_layer_type=HSTULayerType.NATIVE,
    )
    debug_tensor_shape_to_assert: Dict[str, torch.Size] = {}
    for name, param in debug_model.named_parameters():
        # The layernorm weights, mlp and data-parallel embedding should be initialized the same across whole world.
        if re.match(
            r".*data_parallel_embedding_collection.*|.*_mlp.*|.*layernorm.*|.*bias.*",
            name,
        ):
            collective_assert_tensor(param.data, compare_type="equal", pg=None)
        # model-parallel embedding collection should be initialized differently on each rank
        elif isinstance(param, TableBatchedEmbeddingSlice):
            collective_assert_tensor(param.data, compare_type="not_equal", pg=None)
        else:
            # other parameters should be initialized the same across data-parallel group. and not the same across model-parallel group.
            # i.e. TE*Linear
            collective_assert_tensor(
                param.data,
                compare_type="equal",
                pg=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            # For DEBUG type, we need to assert the parameters are initialized the same across model-parallel group.
            collective_assert_tensor(
                param.data,
                compare_type="equal",
                pg=parallel_state.get_tensor_model_parallel_group(),
            )

        if re.match(r".*_output_layernorm.*$", name):
            child_name = name.split(".")[-1]
            name = name.replace(
                child_name, debug_module_path_to_tpN_module_path[child_name]
            )
            debug_tensor_shape_to_assert[name] = param.shape
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    # TODO: The output layernorm weights might be sharded! But they are initialized the same across whole world. i.e. weight=1, bias=0
    # linear bias is zero'd
    for name, param in tp_model.named_parameters():
        if re.match(
            r".*data_parallel_embedding_collection.*|.*_mlp.*|.*layernorm.*|.*_output_ln_dropout_mul.*|.*bias.*",
            name,
        ):
            collective_assert_tensor(param.data, compare_type="equal", pg=None)
        elif isinstance(param, TableBatchedEmbeddingSlice):
            collective_assert_tensor(param.data, compare_type="not_equal", pg=None)
        else:
            collective_assert_tensor(
                param.data,
                compare_type="equal",
                pg=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            collective_assert_tensor(
                param.data,
                compare_type="not_equal",
                pg=parallel_state.get_tensor_model_parallel_group(),
            )
        if name in debug_tensor_shape_to_assert:
            tp_shape = param.shape
            # ColParallel Linear
            if re.match(r".*_linear_uvqk.weight$|.*_linear_uvqk.bias$", name):
                tp_shape[0] = tp_shape[0] * tp_size
            # RowParallel Linear
            if re.match(r".*_linear_proj.*$", name):
                tp_shape[-1] = tp_shape[-1] * tp_size
            assert (
                tp_shape == debug_tensor_shape_to_assert[name]
            ), f"[tp{parallel_state.get_tensor_model_parallel_rank()},dp{parallel_state.get_data_parallel_rank()}] {name} shape mismatch"


# @pytest.mark.parametrize("contextual_feature_names", [["user0", "user1"], []])
# @pytest.mark.parametrize("max_num_candidates", [10, 0])
# @pytest.mark.parametrize(
#     "optimizer_type_str", ["sgd", "none"]
# )  # adam does not work since torchrec does not save the optimizer state `step`.
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("tp_size", [2, 4, 8, 1])
# def test_tp_gr_ranking_forward_backward(
#     contextual_feature_names: List[str],
#     max_num_candidates: int,
#     optimizer_type_str: str,
#     dtype: torch.dtype,
#     tp_size: int,
# ):
#     # we must use static embedding for
#     use_dynamic_emb = False
#     pipeline_type = "none"
#     init.initialize_distributed()
#     init.initialize_model_parallel(tp_size)
#     model, dense_optimizer, history_batches = create_model(
#         task_type="ranking",
#         contextual_feature_names=contextual_feature_names,
#         max_num_candidates=max_num_candidates,
#         optimizer_type_str=optimizer_type_str,
#         use_dynamic_emb=use_dynamic_emb,
#         pipeline_type="none",
#         dtype=dtype,
#         seed=1234,
#     )
#     pipelined_model, pipelined_dense_optimizer, _ = create_model(
#         task_type="ranking",
#         contextual_feature_names=contextual_feature_names,
#         max_num_candidates=max_num_candidates,
#         optimizer_type_str=optimizer_type_str,
#         dtype=dtype,
#         use_dynamic_emb=use_dynamic_emb,
#         pipeline_type=pipeline_type,
#         seed=1234,
#     )

#     # we will use ckpt to initialize the pipelined model
#     # state_dict is not supported for dynamic embedding!
#     for batch in history_batches:
#         model.module.zero_grad_buffer()
#         dense_optimizer.zero_grad()
#         loss, _ = model(batch)
#         collective_assert(not torch.isnan(loss).any(), f"loss has nan")
#         loss.sum().backward()
#         finalize_model_grads([model.module], None)
#         dense_optimizer.step()

#     save_path = "./gr_checkpoint"
#     if dist.get_rank() == 0:
#         if os.path.exists(save_path):
#             shutil.rmtree(save_path)
#     dist.barrier(device_ids=[torch.cuda.current_device()])

#     if dist.get_rank() == 0:
#         os.makedirs(save_path, exist_ok=True)
#     dist.barrier(device_ids=[torch.cuda.current_device()])
#     checkpoint.save(save_path, model, dense_optimizer=dense_optimizer)
#     checkpoint.load(
#         save_path, pipelined_model, dense_optimizer=pipelined_dense_optimizer
#     )
#     dist.barrier(device_ids=[torch.cuda.current_device()])
#     if dist.get_rank() == 0:
#         shutil.rmtree(save_path)

#     no_pipeline = JaggedMegatronTrainNonePipeline(
#         model,
#         dense_optimizer,
#         device=torch.device("cuda", torch.cuda.current_device()),
#     )
#     if pipeline_type == "native":
#         target_pipeline = JaggedMegatronTrainPipelineSparseDist(
#             pipelined_model,
#             pipelined_dense_optimizer,
#             device=torch.device("cuda", torch.cuda.current_device()),
#         )
#     else:
#         target_pipeline = JaggedMegatronPrefetchTrainPipelineSparseDist(
#             pipelined_model,
#             pipelined_dense_optimizer,
#             device=torch.device("cuda", torch.cuda.current_device()),
#         )
#     iter_history_batches = iter(history_batches)
#     no_pipeline_batches = iter(history_batches)
#     for i, batch in enumerate(history_batches):
#         reporting_loss, (_, logits, _) = no_pipeline.progress(no_pipeline_batches)
#         pipelined_reporting_loss, (_, pipelined_logits, _) = target_pipeline.progress(
#             iter_history_batches
#         )
#         collective_assert(
#             torch.allclose(pipelined_reporting_loss, reporting_loss),
#             f"reporting loss mismatch",
#         )
#         collective_assert(torch.allclose(pipelined_logits, logits), f"logits mismatch")

#     init.destroy_global_state()
