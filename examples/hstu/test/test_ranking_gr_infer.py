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
import commons.utils.initialize as init
import pytest
import torch
from configs import (
    DynamicShardedEmbeddingConfig,
    RankingConfig,
    RetrievalConfig,
    ShardedEmbeddingConfig,
    get_hstu_config,
    get_kvcache_config,
)
from dataset.utils import FeatureConfig, RankingBatch, RetrievalBatch
from megatron.core import tensor_parallel
from model.ranking_gr_infer import RankingGRInfer
from modules.embedding import EmbeddingOptimizerParam

@pytest.mark.parametrize("model_type", ["ranking"])
@pytest.mark.parametrize("batchsize_per_rank", [32])
@pytest.mark.parametrize("max_contextual_seqlen", [6, 0])
@pytest.mark.parametrize(
    "item_max_seqlen,max_num_candidates",
    [
        (2, 10),
        (20, 10),
        (200, 10),
    ],
)
@pytest.mark.parametrize("dim_size", [128,])
def test_gr_infer(
    model_type,
    batchsize_per_rank,
    item_max_seqlen,
    dim_size,
    max_num_candidates,
    max_contextual_seqlen,
):
    init.initialize_single_rank()
    init.initialize_model_parallel(1)
    init.set_random_seed(1234)
    device = torch.cuda.current_device()

    hstu_config = get_hstu_config(
        hidden_size=dim_size,
        kv_channels=128,
        num_attention_heads=4,
        num_layers=3,
        dtype=torch.bfloat16,
    )

    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=51200,
        tokens_per_block=32,
        max_batch_size=batchsize_per_rank,
        max_seq_len=4096,
    )

    context_emb_size = 1000
    item_emb_size = 1000
    action_vocab_size = 10
    num_tasks = 2
    embedding_optimizer_param = EmbeddingOptimizerParam(
        optimizer_str="adam", learning_rate=0.0001
    )
    emb_configs = [
        ShardedEmbeddingConfig(
            feature_names=["act_feat"],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=dim_size,
            sharding_type="data_parallel",
            optimizer_param=embedding_optimizer_param,
        ),
        DynamicShardedEmbeddingConfig(
            feature_names=["context_feat", "item_feat"]
            if max_contextual_seqlen > 0
            else ["item_feat"],
            table_name="item",
            vocab_size=item_emb_size,
            dim=dim_size,
            optimizer_param=embedding_optimizer_param,
            global_hbm_for_values=1000,
        ),
    ]
    feature_configs = [
        FeatureConfig(
            feature_names=["item_feat", "act_feat"],
            max_item_ids=[item_emb_size, action_vocab_size],
            max_sequence_length=item_max_seqlen + max_num_candidates,
            is_jagged=True,
        ),
    ]
    if max_contextual_seqlen > 0:
        feature_configs.append(
            FeatureConfig(
                feature_names=["context_feat"],
                max_item_ids=[context_emb_size],
                max_sequence_length=10,
                is_jagged=True,
            ),
        )
    batch_kwargs = dict(
        batch_size=batchsize_per_rank,
        feature_configs=feature_configs,
        item_feature_name="item_feat",
        contextual_feature_names=["context_feat"] if max_contextual_seqlen > 0 else [],
        action_feature_name="act_feat",
        max_num_candidates=max_num_candidates,
        device=device,
    )
    if model_type == "ranking":
        task_config = RankingConfig(
            embedding_configs=emb_configs,
            prediction_head_arch=[[128, 10, 1] for _ in range(num_tasks)],
        )
        model_predict = RankingGRInfer(
            hstu_config=hstu_config, 
            kvcache_config=kv_cache_config, 
            task_config=task_config)
        with tensor_parallel.get_cuda_rng_tracker().fork():
            batch = RankingBatch.random(num_tasks=num_tasks, **batch_kwargs)
        user_ids = torch.arange(batchsize_per_rank)

    output_logit = model_predict(batch, user_ids)

    init.destroy_global_state()
