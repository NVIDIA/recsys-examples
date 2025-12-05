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
import sys

import torch
from configs import (
    InferenceEmbeddingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from dataset.random_inference_dataset import RandomInferenceDataGenerator
from dataset.utils import FeatureConfig

sys.path.append("./model/")
from inference_ranking_gr import get_inference_ranking_gr


def run_ranking_gr_inference():
    max_batch_size = 16
    max_seqlen = 4096
    max_num_candidates = 256
    max_incremental_seqlen = 128

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
    total_max_seqlen = sum(
        [fc.max_sequence_length * len(fc.feature_names) for fc in feature_configs]
    )

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
        max_seq_len=total_max_seqlen,
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

        data_generator = RandomInferenceDataGenerator(
            feature_configs=feature_configs,
            item_feature_name=item_fea_name,
            contextual_feature_names=[],
            action_feature_name=action_fea_name,
            max_num_users=16,
            max_batch_size=8,  # test batch size
            max_seqlen=2304,
            max_num_candidates=max_num_candidates,
            max_incremental_seqlen=max_incremental_seqlen,
            full_mode=True,
        )

        num_warmup_batches = 16
        for idx in range(num_warmup_batches):
            uids = data_generator.get_inference_batch_user_ids()

            if uids is None:
                break

            cached_start_pos, cached_len = model_predict.get_user_kvdata_info(uids)
            truncate_start_pos = cached_start_pos + cached_len
            batch = data_generator.get_random_inference_batch(uids, truncate_start_pos)
            batch = batch.to(device=torch.cuda.current_device())

            model_predict.forward(batch, uids, truncate_start_pos)

        num_test_batches = 16
        ts_start, ts_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        predict_time = 0.0
        for idx in range(num_test_batches):
            uids = data_generator.get_inference_batch_user_ids()

            if uids is None:
                break

            cached_start_pos, cached_len = model_predict.get_user_kvdata_info(uids)
            truncate_start_pos = cached_start_pos + cached_len
            batch = data_generator.get_random_inference_batch(uids, truncate_start_pos)
            batch = batch.to(device=torch.cuda.current_device())

            torch.cuda.synchronize()
            ts_start.record()
            model_predict.forward(batch, uids, truncate_start_pos)
            ts_end.record()
            torch.cuda.synchronize()
            predict_time += ts_start.elapsed_time(ts_end)
        print("Total time(ms):", predict_time)


if __name__ == "__main__":
    run_ranking_gr_inference()
