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
import re
from typing import Dict, List

import commons.utils as init
import pytest
import torch
from commons.checkpoint import get_unwrapped_module
from commons.utils.distributed_utils import collective_assert, collective_assert_tensor
from commons.utils.hstu_assert_close import hstu_close
from configs import HSTULayerType, KernelBackend
from megatron.core import parallel_state
from pipeline.train_pipeline import JaggedMegatronTrainNonePipeline
from test_utils import (
    compare_tpN_to_debug_weights,
    create_model,
    debug_module_path_to_tpN_module_path,
    init_module_from,
    init_tpN_weights_from_debug,
    zero_bias,
)
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)


@pytest.mark.parametrize("tp_size", [2, 4, 8, 1])
def test_gr_tp_ranking_initialization(tp_size: int):
    contextual_feature_names: List[str] = []
    max_num_candidates: int = 10
    # we must use static embedding for
    use_dynamic_emb: bool = False
    pipeline_type: str = "none"
    optimizer_type_str: str = "sgd"
    dtype: torch.dtype = torch.bfloat16

    init.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    if world_size < tp_size:
        pytest.skip("TP size is larger than world size")
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
    # import pdb;pdb.set_trace()
    tp_model.state_dict()[
        "_embedding_collection._model_parallel_embedding_collection.embeddings.item.weight"
    ].local_tensor()
    # print(f'[tp{parallel_state.get_tensor_model_parallel_rank()},dp{parallel_state.get_data_parallel_rank()}] {local_shard.shape}')
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


@pytest.mark.parametrize("contextual_feature_names", [["user0", "user1"]])
@pytest.mark.parametrize("max_num_candidates", [0])
@pytest.mark.parametrize("optimizer_type_str", ["adam", "sgd"])  # TODO: add adam
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("tp_size", [2, 4, 8, 1])
@pytest.mark.parametrize(
    "replicate_batches", [True, False]
)  # same batch or various batches
@pytest.mark.parametrize(
    "kernel_backend", [KernelBackend.PYTORCH]
)  # Note that only pytorch supports fp32
def test_tp_gr_ranking_forward_backward_update(
    contextual_feature_names: List[str],
    max_num_candidates: int,
    optimizer_type_str: str,
    dtype: torch.dtype,
    tp_size: int,
    replicate_batches: bool,
    kernel_backend: KernelBackend,
):
    # we use mock debug tp layer to test the correctness.
    os.environ["DEBUG_MOCK_TP"] = "1"

    # we must use static embedding for reproducibility
    use_dynamic_emb = False
    init.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    num_heads = 8
    if world_size < tp_size:
        pytest.skip("TP size is larger than world size")
    if num_heads % tp_size != 0:
        pytest.skip("num_heads should be divisible by tp_size")
    init.initialize_model_parallel(tp_size)
    tp_model, tp_dense_optimizer, _ = create_model(
        task_type="ranking",
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        use_dynamic_emb=use_dynamic_emb,
        pipeline_type="none",
        dtype=dtype,
        seed=1234,
        hstu_layer_type=HSTULayerType.NATIVE,  # only native supports TP
        kernel_backend=kernel_backend,  # only pytorch supports fp32
        num_heads=num_heads,
    )
    (
        debug_model,
        debug_dense_optimizer,
        history_batches,
    ) = create_model(  # replicate all weights!
        task_type="ranking",
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        use_dynamic_emb=use_dynamic_emb,
        pipeline_type="none",
        dtype=dtype,
        seed=1234,
        hstu_layer_type=HSTULayerType.DEBUG,  # no TP, i.e. does not shard weights
        num_batches=40 if optimizer_type_str == "sgd" else 80,
        replicate_batches=replicate_batches,
        kernel_backend=kernel_backend,  # only pytorch supports fp32
        num_heads=num_heads,
    )
    debug_model_fp32, debug_dense_optimizer_fp32, _ = create_model(
        task_type="ranking",
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        use_dynamic_emb=use_dynamic_emb,
        pipeline_type="none",
        dtype=torch.float32,
        seed=1234,
        hstu_layer_type=HSTULayerType.DEBUG,
        kernel_backend=KernelBackend.PYTORCH,  # only pytorch supports fp32
        num_heads=num_heads,
    )

    tp_ranking_gr = get_unwrapped_module(tp_model)
    debug_ranking_gr = get_unwrapped_module(debug_model)
    debug_ranking_gr_fp32 = get_unwrapped_module(debug_model_fp32)

    # set bias to zero for better debugging
    zero_bias(debug_model_fp32)
    # init debug from fp32
    init_module_from(debug_model_fp32, debug_model)
    # init tp from debug
    init_tpN_weights_from_debug(debug_model, tp_model)
    # this is a must because the master weight needs to be updated as well
    tp_dense_optimizer.reload_model_params()
    debug_dense_optimizer.reload_model_params()
    debug_dense_optimizer_fp32.reload_model_params()

    debug_pipeline = JaggedMegatronTrainNonePipeline(
        debug_model,
        debug_dense_optimizer,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    debug_pipeline_fp32 = JaggedMegatronTrainNonePipeline(
        debug_model_fp32,
        debug_dense_optimizer_fp32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    tp_pipeline = JaggedMegatronTrainNonePipeline(
        tp_model,
        tp_dense_optimizer,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    iter_history_batches = iter(history_batches)
    debug_pipeline_batches = iter(history_batches)
    debug_pipeline_batches_fp32 = iter(history_batches)
    # initial state check
    compare_tpN_to_debug_weights(tp_ranking_gr, debug_ranking_gr, debug_ranking_gr_fp32)
    multiplier = 2

    for i, batch in enumerate(history_batches):
        _, (losses, logits, _) = debug_pipeline.progress(debug_pipeline_batches)
        _, (losses_fp32, logits_fp32, _) = debug_pipeline_fp32.progress(
            debug_pipeline_batches_fp32
        )
        _, (tp_losses, tp_logits, _) = tp_pipeline.progress(iter_history_batches)
        compare_tpN_to_debug_weights(
            tp_ranking_gr, debug_ranking_gr, debug_ranking_gr_fp32
        )

        # # the logits must be bit-wise aligned across tp ranks
        collective_assert_tensor(
            logits_fp32,
            compare_type="equal",
            pg=parallel_state.get_tensor_model_parallel_group(),
        )
        collective_assert_tensor(
            logits,
            compare_type="equal",
            pg=parallel_state.get_tensor_model_parallel_group(),
        )
        collective_assert_tensor(
            tp_logits,
            compare_type="equal",
            pg=parallel_state.get_tensor_model_parallel_group(),
        )

        # assert loss & logits element-wise
        collective_assert(
            hstu_close(tp_losses, losses, losses_fp32, multiplier=multiplier),
            f"losses mismatch at iter {i}, diff {(tp_losses - losses_fp32).abs().max()} vs {(losses - losses_fp32).abs().max()} vs hey {(tp_losses - losses).abs().max()}",
        )
        collective_assert(
            hstu_close(tp_logits, logits, logits_fp32, multiplier=multiplier),
            f"logits mismatch at iter {i}, diff {(tp_logits - logits_fp32).abs().max()} vs {(logits - logits_fp32).abs().max()} vs hey {(tp_logits - logits).abs().max()}",
        )
        print(
            f"[tp{parallel_state.get_tensor_model_parallel_rank()}, dp{parallel_state.get_data_parallel_rank()}] [iter {i} is good]"
        )
        # relax the threshold except for the first iteration
        multiplier = 5 if i == 0 else multiplier

    init.destroy_global_state()
