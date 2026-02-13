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
import argparse
import sys

import torch
from commons.datasets import get_data_loader
from commons.datasets.hstu_batch import FeatureConfig
from commons.datasets.random_inference_dataset import RandomInferenceDataset
from configs import (
    InferenceEmbeddingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)

sys.path.append("./model/")
from inference_ranking_gr import get_inference_ranking_gr


def run_ranking_gr_inference(disable_kvcache: bool):
    max_batch_size = 16
    max_num_history = 2048
    max_num_candidates = 256
    max_incremental_seqlen = 128
    max_seqlen = max_num_history * 2 + max_num_candidates

    # context_emb_size = 1000
    item_fea_name, item_vocab_size = "item_feat", 10000
    action_fea_name, action_vocab_size = "act_feat", 128
    feature_configs = [
        FeatureConfig(
            feature_names=[item_fea_name, action_fea_name],
            max_item_ids=[item_vocab_size - 1, action_vocab_size - 1],
            max_sequence_length=max_seqlen,
            is_jagged=False,
        ),
    ]
    max_contextual_seqlen = 0
    # total_max_seqlen = sum(
    #     [fc.max_sequence_length * len(fc.feature_names) for fc in feature_configs]
    # )

    hidden_dim_size = 1024
    num_heads = 4
    head_dim = 256
    num_layers = 8
    inference_dtype = torch.bfloat16
    hstu_cudagraph_configs = {
        "batch_size": [1, 2, 4, 8],
        "length_per_sequence": [i * 256 for i in range(2, 18)],
    }

    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seqlen,
        dtype=inference_dtype,
    )

    _blocks_in_primary_pool = 10240
    _page_size = 32
    _offload_chunksize = 8192
    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=_blocks_in_primary_pool,
        page_size=_page_size,
        offload_chunksize=_offload_chunksize,
    )
    emb_configs = [
        InferenceEmbeddingConfig(
            feature_names=["context_feat", "item_feat"]
            if max_contextual_seqlen > 0
            else ["item_feat"],
            table_name="item",
            vocab_size=item_vocab_size,
            dim=hidden_dim_size,
            use_dynamicemb=True,
        ),
        InferenceEmbeddingConfig(
            feature_names=["act_feat"],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=hidden_dim_size,
            use_dynamicemb=False,
        ),
    ]
    num_tasks = 3
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=[128, 10, 1],
        num_tasks=num_tasks,
    )

    with torch.inference_mode():
        model_predict = get_inference_ranking_gr(
            hstu_config=hstu_config,
            kvcache_config=kv_cache_config,
            task_config=task_config,
            use_cudagraph=True,
            cudagraph_configs=hstu_cudagraph_configs,
        )
        model_predict.bfloat16()
        model_predict.eval()

        dataset = RandomInferenceDataset(
            feature_configs=feature_configs,
            item_feature_name=item_fea_name,
            contextual_feature_names=[],
            action_feature_name=action_fea_name,
            max_num_users=1,
            max_batch_size=1,  # test batch size
            max_history_length=max_num_history,
            max_num_candidates=max_num_candidates,
            max_incremental_seqlen=max_incremental_seqlen,
            max_num_cached_batches=16,
            full_mode=True,
        )

        dataloader = get_data_loader(dataset)

        # Warm up
        for batch, user_ids, total_history_lengths in dataloader:
            model_predict.forward_nokvcache(batch)

        dataloader = get_data_loader(dataset)
        ts_start, ts_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ts_start.record()
        for batch, user_ids, total_history_lengths in dataloader:
            if not disable_kvcache:
                model_predict.forward(batch, user_ids, total_history_lengths)
            else:
                model_predict.forward_nokvcache(batch)
        ts_end.record()
        predict_time = ts_start.elapsed_time(ts_end)
        print("Total time(ms):", predict_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference KVCache Demo Benchmark")
    parser.add_argument("--disable_kvcache", action="store_true")

    args = parser.parse_args()
    run_ranking_gr_inference(args.disable_kvcache)
