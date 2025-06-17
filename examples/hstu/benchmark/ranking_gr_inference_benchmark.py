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
import torch
from configs import (
    RankingConfig,
    ShardedEmbeddingConfig,
    get_hstu_config,
    get_kvcache_config,
)
from dataset.utils import FeatureConfig, RankingBatch, RetrievalBatch
from megatron.core import tensor_parallel
from model.ranking_gr_infer import RankingGRInferenceModel
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Optional, List

def generate_benchmark_ranking_batch(
        batch_size: int,
        feature_configs: List[FeatureConfig],
        item_feature_name: str,
        contextual_feature_names: List[str] = [],
        action_feature_name: Optional[str] = None,
        max_num_candidates: int = 0,
        num_tasks: int = 1,
        *,
        device: torch.device,
    ) -> "RankingBatch":
        """
        Generate a random RankingBatch.

        Args:
            batch_size (int): The number of elements in the batch.
            feature_configs (List[FeatureConfig]): List of configurations for each feature.
            item_feature_name (str): The name of the item feature.
            contextual_feature_names (List[str], optional): List of names for the contextual features. Defaults to [].
            action_feature_name (Optional[str], optional): The name of the action feature. Defaults to None.
            max_num_candidates (int, optional): The maximum number of candidate items. Defaults to 0.
            num_tasks (int): The number of tasks. Defaults to 1.
            device (torch.device): The device on which the batch will be generated.

        Returns:
            RankingBatch: The generated random RankingBatch.
        """
        assert num_tasks is not None, "num_tasks is required for RankingBatch"

        keys = []
        values = []
        lengths = []
        num_candidates = None
        feature_to_max_seqlen = {}
        for fc in feature_configs:
            seqlen = torch.full((batch_size,), fc.max_sequence_length, dtype=torch.int64, device=device)
            cur_seqlen_sum = torch.sum(seqlen).item()

            for feature_name, max_item_id in zip(fc.feature_names, fc.max_item_ids):
                value = torch.randint(max_item_id, (cur_seqlen_sum,), device=device)
                keys.append(feature_name)
                values.append(value)
                lengths.append(seqlen)
                if max_num_candidates > 0 and feature_name == item_feature_name:
                    non_candidates_seqlen = seqlen - max_num_candidates
                    num_candidates = seqlen - non_candidates_seqlen
                feature_to_max_seqlen[feature_name] = fc.max_sequence_length
        
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=keys,
            values=torch.concat(values).to(device),
            lengths=torch.concat(lengths).to(device).long(),
        )

        if num_candidates is not None:
            total_num_labels = torch.sum(num_candidates)
        else:
            total_num_labels = torch.sum(features[item_feature_name].lengths())
        labels = torch.randint(1 << num_tasks, (total_num_labels,), device=device)
        return RankingBatch(
            features=features,
            batch_size=batch_size,
            feature_to_max_seqlen=feature_to_max_seqlen,
            contextual_feature_names=contextual_feature_names,
            item_feature_name=item_feature_name,
            action_feature_name=action_feature_name,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates,
            labels=labels,
        )

def bench_gr_infer(
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
        num_layers=8,
        dtype=torch.bfloat16,
    )

    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=10240,
        tokens_per_block=32,
        max_batch_size=batchsize_per_rank,
        max_seq_len=4096,
    )

    context_emb_size = 1000
    item_emb_size = 1000
    action_vocab_size = 10
    num_tasks = 1
    # embedding_optimizer_param = EmbeddingOptimizerParam(
    #     optimizer_str="adam", learning_rate=0.0001
    # )
    emb_configs = [
        ShardedEmbeddingConfig(
            feature_names=["act_feat"],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=dim_size,
            sharding_type="data_parallel",
        ),
        ShardedEmbeddingConfig(
            feature_names=["context_feat", "item_feat"]
            if max_contextual_seqlen > 0
            else ["item_feat"],
            table_name="item",
            vocab_size=item_emb_size,
            dim=dim_size,
            sharding_type="model_parallel",
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
        model_predict = RankingGRInferenceModel(
            hstu_config=hstu_config, 
            kvcache_config=kv_cache_config, 
            task_config=task_config,
            use_cudagraph=True)
        model_predict.bfloat16()
        model_predict.eval()

        input_batch = [
            generate_benchmark_ranking_batch(**batch_kwargs) for _ in range(10)
        ]
        # batch = generate_benchmark_ranking_batch(**batch_kwargs)
        user_ids = torch.arange(batchsize_per_rank)

    num_batch = len(input_batch)
    for idx in range(1):
        output_logit = model_predict(input_batch[idx % num_batch], user_ids)
    # torch.cuda.profiler.stop()
    
    user_ids = user_ids + batchsize_per_rank
    for idx in range(100):
        torch.cuda.profiler.start()
        output_logit = model_predict(input_batch[idx % num_batch], user_ids)
        torch.cuda.profiler.stop()
    
    init.destroy_global_state()

if __name__ == "__main__":
    for bs in [32]:
        for ncand in [16, 112, 240, 496]:
            bench_gr_infer(
                model_type = "ranking",
                batchsize_per_rank = bs,
                item_max_seqlen = 16,
                dim_size = 128,
                max_num_candidates = ncand,
                max_contextual_seqlen = 0,
            )
            print()