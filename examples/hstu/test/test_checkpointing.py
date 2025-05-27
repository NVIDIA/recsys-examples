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
import pytest
import torch
import torch.distributed as dist
from commons.utils.distributed_utils import collective_assert
from configs import OptimizerParam
from distributed.sharding import make_optimizer_and_shard
from dynamicemb import DynamicEmbTableOptions
from megatron.core import tensor_parallel
from megatron.core.distributed import finalize_model_grads
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)


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
    device = torch.device("cuda", torch.cuda.current_device())
    embdim = 128
    hstu_config = configs.get_hstu_config(
        hidden_size=embdim,
        kv_channels=128,
        num_attention_heads=4,
        num_layers=1,
        hidden_dropout=0.2,
        dtype=dtype,
    )

    item_feature_name = "item_feat"
    action_feature_name = "action_feat"
    contextual_emb_size = 1000
    item_emb_size = 1000
    action_vocab_size = 1000
    emb_configs = [
        configs.ShardedEmbeddingConfig(
            feature_names=[action_feature_name],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=embdim,
            sharding_type="data_parallel",
        ),
        configs.ShardedEmbeddingConfig(
            feature_names=[item_feature_name],
            table_name="item",
            vocab_size=item_emb_size,
            dim=embdim,
            sharding_type="model_parallel",
        ),
    ]
    feature_configs = [
        dataset.utils.FeatureConfig(
            feature_names=[item_feature_name, action_feature_name],
            max_item_ids=[item_emb_size, action_vocab_size],
            max_sequence_length=100,
            is_jagged=True,
        )
    ]
    if len(contextual_feature_names) > 0:
        feature_configs.append(
            dataset.utils.FeatureConfig(
                feature_names=contextual_feature_names,
                max_item_ids=[
                    contextual_emb_size for _ in range(len(contextual_feature_names))
                ],
                max_sequence_length=10,
                is_jagged=True,
            )
        )
        emb_configs.append(
            configs.ShardedEmbeddingConfig(
                feature_names=contextual_feature_names,
                table_name="context",
                vocab_size=contextual_emb_size,
                dim=embdim,
                sharding_type="model_parallel",
            )
        )

    batch_kwargs = dict(
        batch_size=32,
        feature_configs=feature_configs,
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

        history_batches = []
        with tensor_parallel.get_cuda_rng_tracker().fork():
            batch = dataset.utils.RankingBatch.random(
                num_tasks=num_tasks, **batch_kwargs
            )
            for i in range(10):
                history_batches.append(batch)
    else:
        assert task_type == "retrieval"
        task_config = configs.RetrievalConfig(embedding_configs=emb_configs)
        model_train = model.RetrievalGR(
            hstu_config=hstu_config, task_config=task_config
        )

        history_batches = []
        with tensor_parallel.get_cuda_rng_tracker().fork():
            batch = dataset.utils.RetrievalBatch.random(**batch_kwargs)
            for i in range(10):
                history_batches.append(batch)
    optimizer_param = OptimizerParam(
        optimizer_str=optimizer_type_str,
        learning_rate=1e-3,
    )
    model_train, dense_optimizer = make_optimizer_and_shard(
        model_train,
        config=hstu_config,
        dynamicemb_options_dict={
            "item": DynamicEmbTableOptions(
                global_hbm_for_values=0,
            )
        },
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
    )

    world_size = torch.distributed.get_world_size()
    for n, p in model_train.named_parameters():
        output = torch.empty(world_size * p.numel(), device=p.device, dtype=p.dtype)
        torch.distributed.all_gather_into_tensor(output, p.contiguous())
        sliced_shape = [world_size, p.numel()]
        output = output.reshape(sliced_shape)
        for r in range(world_size):
            if r == torch.distributed.get_rank():
                continue
            if p.numel() == 0:
                continue
            if isinstance(p, TableBatchedEmbeddingSlice):
                assert not torch.allclose(
                    p.flatten(), output[r].flatten()
                ), "embedding table should be initialized differently on each rank for mp and dynamic embedding."
            else:
                assert torch.allclose(
                    p.flatten(), output[r].flatten()
                ), f"parameter {n} shape {p.shape} should be initialized the same on each rank."

    model_train.train()
    if forward:
        for i in range(10):
            model_train.module.zero_grad_buffer()
            dense_optimizer.zero_grad()
            loss, _ = model_train(batch)
            collective_assert(not torch.isnan(loss).any(), f"iter {i} loss has nan")

            loss.sum().backward()
            finalize_model_grads([model_train.module], None)
            dense_optimizer.step()
    return model_train, dense_optimizer, history_batches


@pytest.mark.parametrize(
    "task_type",
    ["ranking", "retrieval"],
)
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
        forward=True,
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

    # from commons.checkpoint import get_unwrapped_module
    # eval_module = get_unwrapped_module(model)
    # new_eval_module = get_unwrapped_module(new_model)
    # for batch in history_batches:
    #     eval_module.evaluate_one_batch(batch)
    #     new_eval_module.evaluate_one_batch(batch)
    # eval_result = eval_module.compute_metric()
    # new_eval_result = new_eval_module.compute_metric()

    # assert (
    #     eval_result == new_eval_result
    # ), "loaded model should have same eval result with original model"
    init.destroy_global_state()


from modules.embedding import DataParallelEmbeddingCollection
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import BoundsCheckMode, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig, dtype_to_data_type
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def test_data_parallel_embedding_collection():
    init.initialize_distributed()
    init.initialize_model_parallel(1)

    embedding_configs = [
        EmbeddingConfig(
            name="item",
            embedding_dim=128,
            num_embeddings=10000,
            feature_names=["item_feat"],
            data_type=dtype_to_data_type(torch.float32),
        ),
        EmbeddingConfig(
            name="context",
            embedding_dim=128,
            num_embeddings=10000,
            feature_names=["context_feat"],
            data_type=dtype_to_data_type(torch.float32),
        ),
        EmbeddingConfig(
            name="action",
            embedding_dim=128,
            num_embeddings=10000,
            feature_names=["action_feat"],
            data_type=dtype_to_data_type(torch.float32),
        ),
    ]

    embedding_collection = EmbeddingCollection(
        tables=embedding_configs,
        device=torch.device("meta"),
    )
    constraints = {}
    for config in embedding_configs:
        constraints[config.name] = ParameterConstraints(
            sharding_types=[ShardingType.DATA_PARALLEL.value],
            bounds_check_mode=BoundsCheckMode.NONE,
        )
    planner = EmbeddingShardingPlanner(constraints=constraints)

    plan = planner.collective_plan(embedding_collection)
    sharding_plan = plan.plan[""]
    data_parallel_embedding_collection = DataParallelEmbeddingCollection(
        data_parallel_embedding_collection=embedding_collection,
        data_parallel_sharding_plan=sharding_plan,
        env=ShardingEnv.from_process_group(dist.group.WORLD),
        device=torch.device("cuda"),
    )

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["item_feat", "action_feat", "user0", "user1", "context_feat"],
        lengths=torch.tensor([5, 10, 15, 20, 25]),
        values=torch.randint(0, 10000, (135,)),
    ).to(torch.device("cuda"))
    output = data_parallel_embedding_collection(kjt)
    assert "item_feat" in output, "item_feat should be in output"
    assert "action_feat" in output, "action_feat should be in output"
    assert "user0" not in output, "user0 should not be in output"
    assert "user1" not in output, "user1 should not be in output"
    assert "context_feat" in output, "context_feat should be in output"
    for feature_name, table_name in zip(
        ["item_feat", "action_feat", "context_feat"], ["item", "action", "context"]
    ):
        weights = data_parallel_embedding_collection.embedding_weights[table_name].data
        feature = kjt[feature_name]
        res_embedding = output[feature_name]
        ref_embedding = weights[feature.values().long(), :]
        assert torch.allclose(
            feature.lengths(), res_embedding.lengths()
        ), f"lengths of {feature_name} should be the same"
        assert torch.allclose(
            ref_embedding, res_embedding.values()
        ), f"values of {feature_name} should be the same"
