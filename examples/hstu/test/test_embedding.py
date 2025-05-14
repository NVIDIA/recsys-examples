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


import commons.checkpoint as checkpoint
import commons.utils.initialize as init
import pytest
import torch
from modules.embedding import (
    DynamicShardedEmbeddingConfig,
    EmbeddingOptimizerParam,
    ShardedEmbedding,
    ShardedEmbeddingConfig,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@pytest.mark.parametrize("num_embeddings", [10000])
@pytest.mark.parametrize("embedding_dim", [128])
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("optimizer_type_str", ["adam", "sgd"])
def test_embedding(
    num_embeddings: int,
    embedding_dim: int,
    tp: int,
    optimizer_type_str: str,
):
    init.initialize_distributed()
    world_size = torch.distributed.get_world_size()
    if world_size < tp:
        print(f"no enough GPUs to run tp={tp}, will skip")
        return
    init.initialize_model_parallel(tp)
    init.set_random_seed(1234)
    device = torch.cuda.current_device()

    embedding_optimizer_param = EmbeddingOptimizerParam(
        optimizer_str=optimizer_type_str,
        learning_rate=1e-3,
    )
    emb_configs = [
        ShardedEmbeddingConfig(
            feature_names=["feature0"],
            table_name="table0",
            vocab_size=num_embeddings,
            dim=embedding_dim,
            sharding_type="model_parallel",
            optimizer_param=embedding_optimizer_param,
        ),
        ShardedEmbeddingConfig(
            feature_names=["feature1"],
            table_name="table1",
            vocab_size=num_embeddings,
            dim=embedding_dim,
            sharding_type="data_parallel",
            optimizer_param=embedding_optimizer_param,
        ),
        DynamicShardedEmbeddingConfig(
            feature_names=["feature2"],
            table_name="table2",
            vocab_size=num_embeddings,
            dim=embedding_dim,
            optimizer_param=embedding_optimizer_param,
            global_hbm_for_values=0,
        ),
    ]
    embedding = ShardedEmbedding(emb_configs)
    for _, v in embedding._plan.plan.items():
        for param_name, param_sharding in v.items():
            if param_name == "table0":
                assert param_sharding.sharding_type == "row_wise"
                assert param_sharding.compute_kernel == "fused"
            if param_name == "table1":
                assert param_sharding.sharding_type == "data_parallel"
                assert param_sharding.compute_kernel == "dense"
            if param_name == "table2":
                assert param_sharding.sharding_type == "row_wise"
                assert param_sharding.compute_kernel == "customized_kernel"

    world_size = torch.distributed.get_world_size()
    sharded_module = checkpoint.find_sharded_modules(embedding)
    for _, _, m in sharded_module:
        for n, p in m.named_parameters():
            if "table2" in n:
                continue
            output = torch.empty(world_size * p.numel(), device=p.device, dtype=p.dtype)
            torch.distributed.all_gather_into_tensor(output, p.contiguous())
            sliced_shape = [world_size, p.numel()]
            output = output.reshape(sliced_shape)
            for r in range(world_size):
                if r == torch.distributed.get_rank():
                    continue
                if "table1" in n:
                    assert torch.allclose(
                        p.flatten(), output[r].flatten()
                    ), "embedding table should be initialized same on each rank."
                else:
                    assert not torch.allclose(
                        p.flatten(), output[r].flatten()
                    ), "embedding table should be initialized differently on each rank for mp and dynamic embedding."

    # Create the JaggedTensor
    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["feature0", "feature1", "feature2"],
        values=torch.tensor([0, 2, 3, 4, 5, 6, 7, 8], device=device),
        lengths=torch.tensor([1, 1, 1, 1, 1, 3], device=device),
    )

    emb_dict = embedding(kjt)
    output = torch.concat([jt.values() for jt in emb_dict.values()])
    output.sum().backward()
    if hasattr(embedding, "_nonfused_embedding_optimizer"):
        embedding._nonfused_embedding_optimizer.step()
    for emb in emb_dict.values():
        assert not torch.isnan(emb.values()).any(), "emb contains NaN values"
        assert not torch.isnan(emb.lengths()).any(), "emb contains NaN values"

    init.destroy_global_state()
