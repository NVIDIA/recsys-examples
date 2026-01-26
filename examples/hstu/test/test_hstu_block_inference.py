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
from commons.datasets.hstu_batch import FeatureConfig
from commons.datasets.random_inference_dataset import RandomInferenceDataGenerator
from configs import (
    InferenceEmbeddingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)

sys.path.append("./model/")
from inference_ranking_gr import InferenceRankingGR


def get_test_setup():
    max_batch_size = 2
    max_seqlen = 1024

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

    hidden_dim_size = 512
    num_heads = 4
    head_dim = 128
    num_layers = 4

    hstu_config = get_inference_hstu_config(
        hidden_dim_size,
        num_layers,
        num_heads,
        head_dim,
        max_batch_size,
        max_seqlen,
    )

    _blocks_in_primary_pool = 10240
    _page_size = 32
    _offload_chunksize = 128
    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=_blocks_in_primary_pool,
        page_size=_page_size,
        offload_chunksize=_offload_chunksize,
    )
    emb_configs = [
        InferenceEmbeddingConfig(
            feature_names=["act_feat"],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=hidden_dim_size,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=["context_feat", "item_feat"]
            if max_contextual_seqlen > 0
            else ["item_feat"],
            table_name="item",
            vocab_size=item_vocab_size,
            dim=hidden_dim_size,
            use_dynamicemb=True,
        ),
    ]
    num_tasks = 3
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=[[128, 10, 1] for _ in range(num_tasks)],
    )

    model = InferenceRankingGR(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config,
        task_config=task_config,
        use_cudagraph=False,
    )
    model.bfloat16()
    model.eval()

    return model, feature_configs


def test_hstu_process_inference():
    max_batch_size = 2
    max_seqlen = 1024
    max_num_candidates = 128
    max_incremental_seqlen = 64

    item_fea_name = "item_feat"
    action_fea_name = "act_feat"

    with torch.inference_mode():
        model_predict, feature_configs = get_test_setup()

        data_generator = RandomInferenceDataGenerator(
            feature_configs,
            item_fea_name,
            [],
            action_fea_name,
            1024,
            max_batch_size,
            max_seqlen,
            max_num_candidates,
            max_incremental_seqlen,
            False,
        )

        num_test_batches = 100

        for idx in range(num_test_batches):
            uids = data_generator.get_inference_batch_user_ids()

            cached_start_pos, cached_len = model_predict.get_user_kvdata_info(uids)
            truncate_start_pos = cached_start_pos + cached_len

            batch = data_generator.get_random_inference_batch(uids, truncate_start_pos)

            kvc_mtdt = model_predict.prepare_kv_cache(batch, uids, truncate_start_pos)

            embs = model_predict._embedding_collection(batch.features)

            jd = model_predict._hstu_block._preprocessor(embs, batch)

            history_lens = [
                (jd.seqlen[i].item() - jd.num_candidates[i].item()) // 2
                for i in range(batch.batch_size)
            ]

            original_items = torch.tensor(
                [
                    embs["item_feat"].offsets()[i].item() + token_idx
                    for i in range(batch.batch_size)
                    for token_idx in range(history_lens[i])
                ]
            ).long()
            new_items = torch.tensor(
                [
                    2 * token_idx + jd.seqlen_offsets[i].item()
                    for i in range(batch.batch_size)
                    for token_idx in range(history_lens[i])
                ]
            ).long()
            assert torch.allclose(
                embs["item_feat"].values()[original_items].to(torch.bfloat16),
                jd.values[new_items],
            )

            original_actions = torch.tensor(
                [
                    embs["act_feat"].offsets()[i].item() + token_idx
                    for i in range(batch.batch_size)
                    for token_idx in range(history_lens[i])
                ]
            ).long()
            new_actions = torch.tensor(
                [
                    2 * token_idx + 1 + jd.seqlen_offsets[i].item()
                    for i in range(batch.batch_size)
                    for token_idx in range(history_lens[i])
                ]
            ).long()
            assert torch.allclose(
                embs["act_feat"].values()[original_actions].to(torch.bfloat16),
                jd.values[new_actions],
            )

            original_candidates = torch.tensor(
                [
                    embs["item_feat"].offsets()[i].item() + history_lens[i] + token_idx
                    for i in range(batch.batch_size)
                    for token_idx in range(jd.num_candidates[i].item())
                ]
            ).long()
            new_candidates = torch.tensor(
                [
                    jd.seqlen_offsets[i].item() + history_lens[i] * 2 + token_idx
                    for i in range(batch.batch_size)
                    for token_idx in range(jd.num_candidates[i].item())
                ]
            ).long()
            assert torch.allclose(
                embs["item_feat"].values()[original_candidates].to(torch.bfloat16),
                jd.values[new_candidates],
            )

            # post process
            post_jd = model_predict._hstu_block._postprocessor(jd)
            original_candidates = torch.tensor(
                [
                    embs["item_feat"].offsets()[i].item() + history_lens[i] + token_idx
                    for i in range(batch.batch_size)
                    for token_idx in range(jd.num_candidates[i].item())
                ]
            ).long()
            new_candidates = torch.tensor(
                [
                    post_jd.seqlen_offsets[i].item() + token_idx
                    for i in range(batch.batch_size)
                    for token_idx in range(jd.num_candidates[i].item())
                ]
            ).long()
            post_embs = (
                embs["item_feat"].values()[original_candidates].to(torch.bfloat16)
            )
            post_embs = post_embs / torch.linalg.norm(
                post_embs, ord=2, dim=-1, keepdim=True
            ).clamp(min=1e-6)
            assert torch.allclose(post_embs, post_jd.values)

            model_predict.offload_kv_cache(uids, kvc_mtdt)
