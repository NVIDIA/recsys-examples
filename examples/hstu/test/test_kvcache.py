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
import math
import random

import paged_kvcache_ops
import torch
from commons.datasets.hstu_batch import FeatureConfig
from configs import KVCacheMetadata, get_inference_hstu_config, get_kvcache_config
from modules.async_kvcache_manager import AsyncHSTUKVCacheManager


def get_test_kvcache_mgr(
    num_layers,
    blocks_in_primary_pool,
    page_size,
    offload_chunksize,
    enable_nvcomp=False,
):
    # requires to test sequentientially
    max_batch_size = 8
    max_seqlen = 10240

    item_fea_name, item_vocab_size = "item_feat", 100
    action_fea_name, action_vocab_size = "act_feat", 128
    feature_configs = [
        FeatureConfig(
            feature_names=[item_fea_name, action_fea_name],
            max_item_ids=[item_vocab_size - 1, action_vocab_size - 1],
            max_sequence_length=max_seqlen,
            is_jagged=False,
        ),
    ]

    hidden_dim_size = 512
    num_heads = 4
    head_dim = 128
    inference_dtype = torch.bfloat16

    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seqlen,
        dtype=inference_dtype,
    )

    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=blocks_in_primary_pool,
        page_size=page_size,
        offload_chunksize=offload_chunksize,
    )

    async_kvcache_mgr = AsyncHSTUKVCacheManager(
        hstu_config.num_layers,
        hstu_config.num_heads,
        hstu_config.head_dim,
        kv_cache_config.page_size,
        kv_cache_config.blocks_in_primary_pool,
        math.ceil(
            hstu_config.max_batch_size
            * hstu_config.max_seq_len
            / kv_cache_config.page_size
        ),
        4 * hstu_config.max_batch_size * hstu_config.max_seq_len,
        kv_cache_config.offload_chunksize,
        -1,
        hstu_config.max_seq_len,
        hstu_config.max_batch_size,
        4 * hstu_config.max_batch_size * hstu_config.max_seq_len,
        1,
        8,
        8,
        enable_nvcomp,
    )

    # randomnize kvcache data
    for idx in range(num_layers):
        async_kvcache_mgr.cache_table[idx].uniform_(-0.5, 0.5)

    return async_kvcache_mgr


def init_random_kvcache_status():
    pass


def reset_kvcache_status():
    pass


def get_test_userids_and_metadata(
    min_user_id, seq_lengths, num_layers, gpu_kvcache_mgr
):
    batch_size = len(seq_lengths)
    page_size = 32
    chunk_size = 1024
    blocks_in_primary_pool = 10240

    uids = list(range(min_user_id * 2, (min_user_id + batch_size) * 2))
    random.shuffle(uids)
    user_ids = torch.tensor(uids[:batch_size]).long().cuda()
    offload_user_ids = user_ids.clone().cpu()

    num_pages = torch.floor(seq_lengths / chunk_size).int() * int(
        chunk_size / page_size
    )
    num_pages = num_pages.cpu().tolist()
    offload_page_ids = torch.cat(
        [
            torch.randint(blocks_in_primary_pool, (int(num_pages[idx]),))
            for idx in range(batch_size)
        ],
        0,
    ).int()

    kv_offload_handle = paged_kvcache_ops.KVOffloadHandle(
        num_layers, gpu_kvcache_mgr, True
    )

    # zero start
    new_offload_startpos = torch.zeros((batch_size,)).int().cpu()
    new_offload_lengths = torch.floor(seq_lengths / chunk_size).int().cpu() * int(
        chunk_size
    )

    kvcache_metadata = KVCacheMetadata(
        offload_user_ids=offload_user_ids,
        offload_page_ids=offload_page_ids,
        kv_offload_handle=kv_offload_handle,
        new_offload_startpos=new_offload_startpos,
        new_offload_lengths=new_offload_lengths,
    )

    return user_ids, kvcache_metadata


def test_kvcache_offload_onload():
    num_layers = 4
    blocks_in_primary_pool = 10240
    page_size = 32
    offload_chunksize = 1024

    with torch.inference_mode():
        async_kvcache_mgr = get_test_kvcache_mgr(
            num_layers, blocks_in_primary_pool, page_size, offload_chunksize
        )

        uid_min_limit = 0
        for batch_size, seq_len in [
            (3, 5000),
            (5, 10000),
            (6, 8000),
        ]:
            uids, kv_metadata = get_test_userids_and_metadata(
                uid_min_limit,
                torch.tensor([seq_len] * batch_size).int().cuda(),
                num_layers,
                async_kvcache_mgr.gpu_kvcache_mgr,
            )
            async_kvcache_mgr.offload_kvcache(kv_metadata)

            for layer_idx in range(num_layers):
                kv_metadata.kv_offload_handle.mark_ready(layer_idx)

            while async_kvcache_mgr.gpu_kvcache_mgr.is_busy_offloading():
                pass

            async_kvcache_mgr.static_onload_handle.reset()
            async_kvcache_mgr.gpu_kvcache_mgr.onload_kvcache(
                uids.tolist(), async_kvcache_mgr.static_onload_handle
            )

            for layer_idx in range(num_layers):
                async_kvcache_mgr.static_onload_handle.wait_host(layer_idx)

            # check data
            total_onload_pages = len(kv_metadata.offload_page_ids)
            origin_kvdata = async_kvcache_mgr.cache_table[
                :, kv_metadata.offload_page_ids, ...
            ]
            onload_kvdata = async_kvcache_mgr.cache_table[
                :,
                blocks_in_primary_pool : blocks_in_primary_pool + total_onload_pages,
                ...,
            ]
            assert torch.allclose(onload_kvdata, origin_kvdata)

            torch.cuda.synchronize()

            uid_min_limit += batch_size


if __name__ == "__main__":
    random.seed(1)
    torch.manual_seed(0)
    test_kvcache_offload_onload()
