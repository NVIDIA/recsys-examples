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
import shutil
from typing import List

import commons.checkpoint as checkpoint
import commons.utils as init
import pytest
import torch
import torch.distributed as dist
from commons.utils.distributed_utils import collective_assert
from distributed.finalize_model_grads import finalize_model_grads
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from test_utils import create_model


@pytest.mark.parametrize("contextual_feature_names", [["user0", "user1"], []])
@pytest.mark.parametrize("max_num_candidates", [10, 0])
@pytest.mark.parametrize(
    "optimizer_type_str", ["sgd"]
)  # adam does not work since torchrec does not save the optimizer state `step`.
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_dynamic_emb", [True, False])
@pytest.mark.parametrize("pipeline_type", ["prefetch", "native"])
def test_pipeline(
    pipeline_type: str,
    contextual_feature_names: List[str],
    max_num_candidates: int,
    optimizer_type_str: str,
    dtype: torch.dtype,
    use_dynamic_emb: bool,
):
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    model, dense_optimizer, history_batches = create_model(
        task_type="ranking",
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        use_dynamic_emb=use_dynamic_emb,
        pipeline_type="none",
        dtype=dtype,
        seed=1234,
    )
    pipelined_model, pipelined_dense_optimizer, _ = create_model(
        task_type="ranking",
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        dtype=dtype,
        use_dynamic_emb=use_dynamic_emb,
        pipeline_type=pipeline_type,
        seed=1234,
    )

    # we will use ckpt to initialize the pipelined model
    # state_dict is not supported for dynamic embedding!
    for batch in history_batches:
        model.module.zero_grad_buffer()
        dense_optimizer.zero_grad()
        loss, _ = model(batch)
        collective_assert(not torch.isnan(loss).any(), f"loss has nan")
        loss.sum().backward()
        finalize_model_grads([model.module], None)
        dense_optimizer.step()

    save_path = "./gr_checkpoint"
    if dist.get_rank() == 0:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    dist.barrier(device_ids=[torch.cuda.current_device()])

    if dist.get_rank() == 0:
        os.makedirs(save_path, exist_ok=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    checkpoint.save(save_path, model, dense_optimizer=dense_optimizer)
    checkpoint.load(
        save_path, pipelined_model, dense_optimizer=pipelined_dense_optimizer
    )
    dist.barrier(device_ids=[torch.cuda.current_device()])
    if dist.get_rank() == 0:
        shutil.rmtree(save_path)
    no_pipeline = JaggedMegatronTrainNonePipeline(
        model,
        dense_optimizer,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    if pipeline_type == "native":
        target_pipeline = JaggedMegatronTrainPipelineSparseDist(
            pipelined_model,
            pipelined_dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    else:
        target_pipeline = JaggedMegatronPrefetchTrainPipelineSparseDist(
            pipelined_model,
            pipelined_dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    iter_history_batches = iter(history_batches)
    no_pipeline_batches = iter(history_batches)
    for i, batch in enumerate(history_batches):
        reporting_loss, (_, logits, _, _) = no_pipeline.progress(no_pipeline_batches)
        pipelined_reporting_loss, (
            _,
            pipelined_logits,
            _,
            _,
        ) = target_pipeline.progress(iter_history_batches)
        collective_assert(
            torch.allclose(pipelined_reporting_loss, reporting_loss),
            f"reporting loss mismatch",
        )
        collective_assert(torch.allclose(pipelined_logits, logits), f"logits mismatch")

    init.destroy_global_state()
