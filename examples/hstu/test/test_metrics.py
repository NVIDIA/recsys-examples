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


import random

import commons.utils.initialize as init
import pytest
import torch
from commons.ops.collective_ops import grouped_allgatherv_tensor_list
from megatron.core import parallel_state, tensor_parallel
from modules.metrics.metric_modules import RetrievalTaskMetricWithSampling


@pytest.mark.parametrize("num_embeddings", [10000])
@pytest.mark.parametrize("embedding_dim", [8])
@pytest.mark.parametrize("tp", [1, 2])
def test_distributed_topk(num_embeddings: int, embedding_dim: int, tp: int, max_k=5):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(1234)
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        return
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    metric_module = RetrievalTaskMetricWithSampling(MAX_K=max_k)

    # Create the JaggedTensor
    with tensor_parallel.get_cuda_rng_tracker().fork("sharded-embedding-group-seed"):
        keys_array = (
            torch.randint(num_embeddings, (100000,), device=device).cpu().numpy()
        )
        values_array = torch.rand(100000, embedding_dim, device=device).cpu().numpy()

    sum_seqlen = random.randint(0, 100)
    query_embeddings = torch.rand(sum_seqlen, embedding_dim, device=device)
    target_ids = torch.randint(num_embeddings, (sum_seqlen,), device=device)

    metric_module(query_embeddings, target_ids)
    eval_dict_all, global_topk_logits, global_topk_keys = metric_module.compute(
        keys_array, values_array
    )

    (all_query_embeddings, all_target_ids) = grouped_allgatherv_tensor_list(
        [query_embeddings, target_ids],
        pg=parallel_state.get_data_parallel_group(with_context_parallel=True),
    )

    # 2. export local embedding
    local_keys = torch.tensor(keys_array, device=device)
    local_values = torch.tensor(values_array, device=device)
    (global_keys, global_values) = grouped_allgatherv_tensor_list(
        [local_keys, local_values],
        pg=torch.distributed.group.WORLD,
    )

    all_logits = torch.mm(all_query_embeddings, global_values.T)
    ref_top_k_logits, ref_top_k_indices = torch.topk(
        all_logits,
        dim=1,
        k=max_k,
        sorted=True,
        largest=True,
    )  # (B, k,)

    ref_top_k_keys = global_keys[ref_top_k_indices]
    assert torch.allclose(
        ref_top_k_keys, global_topk_keys
    ), f"global topk keys {global_topk_keys} should match reference {ref_top_k_keys}"
    assert torch.allclose(
        ref_top_k_logits, global_topk_logits, rtol=1e-3, atol=1e-2
    ), f"global topk logits {global_topk_logits} should match reference {ref_top_k_logits}"

    init.destroy_global_state()
