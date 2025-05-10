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

import commons.checkpoint as checkpoint
import commons.utils as init
import configs
import dataset
import model
import modules
import pytest
import torch
import torch.distributed as dist
from commons.utils.distributed_utils import collective_assert
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from modules.embedding import EmbeddingOptimizerParam, ShardedEmbedding
from torch.distributed._shard.sharded_tensor import ShardedTensor


def flatten_state_dict(state_dict):
    search_list = [("", state_dict)]

    while len(search_list) > 0:
        prefix, s = search_list.pop()
        if isinstance(s, list):
            search_list.extend([(i, v) for i, v in enumerate(s)])
            continue
        if isinstance(s, dict):
            for name, v in s.items():
                subname = str(prefix) + ("." if prefix else "") + str(name)
                search_list.append((subname, v))
            continue
        yield prefix, s


def assert_equal_two_state_dict(a_state_dict, b_state_dict):
    flatten_a_state_dict = dict(flatten_state_dict(a_state_dict))
    flatten_b_state_dict = dict(flatten_state_dict(b_state_dict))
    for k, v in flatten_a_state_dict.items():
        assert k in flatten_b_state_dict, f"{k} not loadded"
        r = flatten_b_state_dict[k]
        if isinstance(v, torch.Tensor):
            if isinstance(v, ShardedTensor):
                v = v.local_tensor()
                r = r.local_tensor()
            assert torch.allclose(v, r), f"for {k}, tensor {v} != {r}"
        else:
            assert v == r, f"for {k}, value {v} != {r}"


def create_model(
    task_type,
    contextual_feature_names,
    max_num_candidates,
    optimizer_type_str: str,
    dtype: torch.dtype,
    forward=False,
    *,
    seed: int,
):
    init.set_random_seed(seed)
    device = torch.cuda.current_device()
    embdim = 128
    hstu_config = configs.get_hstu_config(
        hidden_size=embdim,
        kv_channels=128,
        num_attention_heads=4,
        num_layers=1,
        hidden_dropout=0.2,
        dtype=dtype,
    )
    embedding_optimizer_param = EmbeddingOptimizerParam(
        optimizer_str=optimizer_type_str,
        learning_rate=1e-3,
    )

    item_feature_name = "item_feat"
    action_feature_name = "action_feat"
    item_emb_size = 1000
    action_vocab_size = 10
    emb_configs = [
        configs.ShardedEmbeddingConfig(
            feature_names=[action_feature_name],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=embdim,
            sharding_type="data_parallel",
            optimizer_param=embedding_optimizer_param,
        ),
        configs.DynamicShardedEmbeddingConfig(
            feature_names=contextual_feature_names + [item_feature_name],
            table_name="item",
            vocab_size=item_emb_size,
            dim=embdim,
            optimizer_param=embedding_optimizer_param,
            global_hbm_for_values=0,
        ),
    ]
    batch_kwargs = dict(
        batch_size=32,
        feature_configs=[
            dataset.utils.FeatureConfig(
                feature_names=contextual_feature_names,
                max_item_ids=[item_emb_size for _ in contextual_feature_names],
                max_sequence_length=10,
                is_jagged=True,
            ),
            dataset.utils.FeatureConfig(
                feature_names=[item_feature_name, action_feature_name],
                max_item_ids=[item_emb_size, action_vocab_size],
                max_sequence_length=100,
                is_jagged=True,
            ),
        ],
        item_feature_name=item_feature_name,
        contextual_feature_names=contextual_feature_names,
        action_feature_name=action_feature_name,
        max_num_candidates=max_num_candidates,
        device=device,
    )
    if task_type == "ranking":
        num_tasks = 1
        task_config = configs.RankingConfig(
            embedding_configs=emb_configs,
            prediction_head_arch=[[128, 10, 1] for _ in range(num_tasks)],
        )
        model_train = model.RankingGR(hstu_config=hstu_config, task_config=task_config)
        batch = dataset.utils.RankingBatch.random(num_tasks=num_tasks, **batch_kwargs)
    else:
        assert task_type == "retrieval"
        task_config = configs.RetrievalConfig(embedding_configs=emb_configs)
        model_train = model.RetrievalGR(
            hstu_config=hstu_config, task_config=task_config
        )
        batch = dataset.utils.RetrievalBatch.random(**batch_kwargs)
    dense_optimizer_config = OptimizerConfig(
        optimizer=optimizer_type_str,
        lr=1e-3,
        params_dtype=dtype,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        use_distributed_optimizer=False,
    )
    megatron_module = [m for n, m in checkpoint.filter_megatron_module(model_train)]
    dense_optimizer = get_megatron_optimizer(
        dense_optimizer_config,
        megatron_module,
    )
    nonfused_embedding_optimizers = list(
        modules.embedding.get_nonfused_embedding_optimizer(model_train)
    )

    model_train.train()
    history_batches = []
    if forward:
        for i in range(10):
            history_batches.append(batch)
            model_train._dense_module.zero_grad_buffer()
            dense_optimizer.zero_grad()
            for optim in nonfused_embedding_optimizers:
                optim.zero_grad()

            loss, _ = model_train(batch)
            collective_assert(not torch.isnan(loss).any(), f"iter {i} loss has nan")

            loss.sum().backward()
            dense_optimizer.step()
            for optim in nonfused_embedding_optimizers:
                optim.step()
    return model_train, dense_optimizer, history_batches


@pytest.mark.parametrize("task_type", ["ranking", "retrieval"])
@pytest.mark.parametrize("contextual_feature_names", [["user0", "user1"], []])
@pytest.mark.parametrize("max_num_candidates", [10, 0])
@pytest.mark.parametrize("optimizer_type_str", ["adam", "sgd"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_checkpoint_model(
    task_type: str,
    contextual_feature_names,
    max_num_candidates,
    optimizer_type_str: str,
    dtype: torch.dtype,
):
    init.initialize_distributed()
    init.initialize_model_parallel(1)

    model, dense_optimizer, history_batches = create_model(
        task_type=task_type,
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        dtype=dtype,
        forward=True,
        seed=1234,
    )
    new_model, new_dense_optimizer, _ = create_model(
        task_type=task_type,
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        dtype=dtype,
        seed=2345,
    )

    save_path = "./gr_checkpoint"
    if dist.get_rank() == 0:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    dist.barrier(device_ids=[torch.cuda.current_device()])

    os.makedirs(save_path, exist_ok=True)

    checkpoint.save(save_path, model, dense_optimizer=dense_optimizer)

    checkpoint.load(save_path, new_model, dense_optimizer=new_dense_optimizer)

    model.eval()
    new_model.eval()
    for batch in history_batches:
        with torch.random.fork_rng():  # randomness negative sampling
            loss, _ = model(batch)
        with torch.random.fork_rng():
            new_loss, _ = new_model(batch)
        assert torch.allclose(
            loss, new_loss
        ), f"loaded model should have same output with original model {loss} vs. {new_loss}"

    assert_equal_two_state_dict(
        dense_optimizer.state_dict(), new_dense_optimizer.state_dict()
    )
    assert_equal_two_state_dict(
        new_dense_optimizer.state_dict(), dense_optimizer.state_dict()
    )
    for a, b in zip(
        checkpoint.find_sharded_modules(model),
        checkpoint.find_sharded_modules(new_model),
    ):
        _, _, sharded_module = a
        _, _, new_sharded_module = b
        assert_equal_two_state_dict(
            sharded_module.fused_optimizer.state_dict(),
            new_sharded_module.fused_optimizer.state_dict(),
        )

    nonfused_embedding_optimizers = list(
        modules.embedding.get_nonfused_embedding_optimizer(model)
    )
    new_nonfused_embedding_optimizers = list(
        modules.embedding.get_nonfused_embedding_optimizer(new_model)
    )
    for optim, new_optim in zip(
        nonfused_embedding_optimizers, new_nonfused_embedding_optimizers
    ):
        assert_equal_two_state_dict(optim.state_dict(), new_optim.state_dict())
        assert_equal_two_state_dict(optim.state_dict(), new_optim.state_dict())

    init.destroy_global_state()


@pytest.mark.parametrize("optimizer_type_str", ["adam", "sgd"])
def test_checkpoint_embedding(
    optimizer_type_str: str,
):
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(1234)

    embdim = 128
    embedding_optimizer_param = EmbeddingOptimizerParam(
        optimizer_str=optimizer_type_str,
        learning_rate=1e-3,
    )
    emb_configs = [
        configs.ShardedEmbeddingConfig(
            feature_names=["feature0"],
            table_name="table0",
            vocab_size=10000,
            dim=embdim,
            sharding_type="model_parallel",
            optimizer_param=embedding_optimizer_param,
        ),
        configs.ShardedEmbeddingConfig(
            feature_names=["feature1"],
            table_name="table1",
            vocab_size=10000,
            dim=embdim,
            sharding_type="data_parallel",
            optimizer_param=embedding_optimizer_param,
        ),
        configs.DynamicShardedEmbeddingConfig(
            feature_names=["feature2"],
            table_name="table2",
            vocab_size=10000,
            dim=embdim,
            optimizer_param=embedding_optimizer_param,
            global_hbm_for_values=0,
        ),
    ]
    embedding = ShardedEmbedding(emb_configs)

    save_path = "./embedding_checkpoint"
    if dist.get_rank() == 0:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    dist.barrier(device_ids=[torch.cuda.current_device()])

    os.makedirs(save_path, exist_ok=True)

    checkpoint.save(save_path, embedding)
    checkpoint.load(save_path, embedding)

    init.destroy_global_state()
