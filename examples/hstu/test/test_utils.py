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


import commons.utils as init
import configs
import dataset
import model
import torch
from configs import OptimizerParam
from distributed.sharding import make_optimizer_and_shard
from dynamicemb import DynamicEmbTableOptions
from megatron.core import tensor_parallel
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)


def _flatten_state_dict(state_dict):
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
    flatten_a_state_dict = dict(_flatten_state_dict(a_state_dict))
    flatten_b_state_dict = dict(_flatten_state_dict(b_state_dict))
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
    pipeline_type: str = "none",
    use_dynamic_emb: bool = True,
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
        hidden_dropout=0.0,  # disable dropout
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
            prediction_head_arch=[128, 10, num_tasks],
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
        }
        if use_dynamic_emb
        else {},
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        pipeline_type=pipeline_type,
        device=device,
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

    return model_train, dense_optimizer, history_batches
