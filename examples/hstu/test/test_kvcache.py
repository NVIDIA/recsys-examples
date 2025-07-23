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
from dataset.utils import Batch, FeatureConfig

sys.path.append("./model/")
from inference_ranking_gr import InferenceRankingGR
from torchrec import KeyedJaggedTensor


def get_test_model(num_layers, blocks_in_primary_pool, page_size, offload_chunksize):
    # requires to test sequentientially
    max_batch_size = 8
    max_seqlen = 512

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
    inference_dtype = torch.bfloat16

    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        dtype=inference_dtype,
    )

    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=blocks_in_primary_pool,
        page_size=page_size,
        offload_chunksize=offload_chunksize,
        max_batch_size=max_batch_size,
        max_seq_len=max_seqlen,
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

    return model


def test_kvcache_offload_onload():
    keys = ["item_feat", "act_feat"]
    feat_max_len = {"item_feat": 256, "act_feat": 256}

    num_layers = 4
    blocks_in_primary_pool = 4096
    page_size = 32
    offload_chunksize = 64

    with torch.inference_mode():
        model_predict = get_test_model(
            num_layers, blocks_in_primary_pool, page_size, offload_chunksize
        )
        for idx in range(num_layers):
            model_predict._gpu_kv_cache_manager.get_kvcache_table(idx).uniform_(
                -0.5, 0.5
            )
        torch.cuda.synchronize()

        max_uid = 0
        for batch_size, seq_len, num_candidate in [
            (3, 200, 90),
            (5, 100, 50),
            (6, 100, 50),
        ]:
            uids = torch.arange(max_uid, max_uid + batch_size).long()
            max_uid += batch_size

            seqlen = torch.full((batch_size,), seq_len)
            num_candidates = torch.full((batch_size,), num_candidate)
            cur_seqlen_sum = torch.sum(seqlen * 2 - num_candidates).item()

            (
                start_positions,
                lengths,
            ) = model_predict._gpu_kv_cache_manager.get_batch_kvdata_info(uids)
            start_positions = torch.clamp(start_positions, min=0)
            start_positions += lengths

            batch = Batch(
                features=KeyedJaggedTensor.from_lengths_sync(
                    keys=keys,
                    values=torch.randint(100, (cur_seqlen_sum,)),
                    lengths=torch.concat([seqlen, seqlen - num_candidates]).long(),
                ),
                batch_size=batch_size,
                feature_to_max_seqlen=feat_max_len,
                contextual_feature_names=[],
                item_feature_name="item_feat",
                action_feature_name="act_feat",
                max_num_candidates=128,
                num_candidates=num_candidates,
            )

            kv_cache_metadata = model_predict.prepare_kv_cache(
                batch, uids, start_positions
            )
            torch.cuda.synchronize()
            model_predict.offload_kv_cache(uids, kv_cache_metadata)
            offload_chunks = (seq_len - num_candidate) * 2 // offload_chunksize
            offload_pages = offload_chunks * offload_chunksize // page_size

            page_indptr = []
            for idx in range(batch_size):
                page_start = kv_cache_metadata.kv_indptr[idx].item()
                page_indptr.append(torch.arange(page_start, page_start + offload_pages))
            page_indptr = torch.concat(page_indptr).cpu()
            gather_pages = kv_cache_metadata.kv_indices[page_indptr]

            (
                onload_length,
                _,
                _,
            ) = model_predict._host_kv_storage_manager.lookup_kvdata(
                uids, torch.zeros_like(uids), torch.zeros_like(uids)
            )

            offload_length = batch_size * offload_chunks * offload_chunksize
            offload_pages = offload_length // page_size

            for layer_idx in range(num_layers):
                assert torch.allclose(
                    kv_cache_metadata.kv_cache_table[layer_idx][gather_pages].cpu(),
                    model_predict._host_kv_storage_manager.static_kvdata_buffer_[
                        layer_idx, :offload_pages
                    ],
                )

            onload_length = offload_length
            onload_pages = offload_pages
            kv_cache_metadata.onload_history_kv_events = (
                model_predict._kvcache_metadata.onload_history_kv_events[:]
            )
            model_predict._gpu_kv_cache_manager.onload(
                model_predict._host_kv_storage_manager.get_lookup_buffer(),
                onload_length,
                kv_cache_metadata,
            )
            torch.cuda.synchronize()
            for layer_idx in range(num_layers):
                assert torch.allclose(
                    kv_cache_metadata.kv_cache_table[layer_idx][gather_pages],
                    kv_cache_metadata.kv_cache_table[layer_idx][
                        blocks_in_primary_pool : blocks_in_primary_pool + onload_pages
                    ],
                )


if __name__ == "__main__":
    test_kvcache_offload_onload()
