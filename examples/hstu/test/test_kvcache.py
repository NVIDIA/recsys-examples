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
from typing import Optional
import paged_kvcache_ops
import pytest
import torch
from commons.datasets.hstu_batch import FeatureConfig
from configs import KVCacheMetadata, get_inference_hstu_config, get_kvcache_config
from modules.async_kvcache_manager import AsyncHSTUKVCacheManager
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for kvcache tests.",
)
def _shutdown_mgr(async_kvcache_mgr: AsyncHSTUKVCacheManager) -> None:
    # Current manager has no explicit shutdown() in your branch.
    # Explicitly close thread pools to avoid dangling threads in tests.
    if hasattr(async_kvcache_mgr, "executor"):
        async_kvcache_mgr.executor.shutdown(wait=False)
    if hasattr(async_kvcache_mgr, "onload_worker"):
        async_kvcache_mgr.onload_worker.shutdown(wait=False)
def get_test_kvcache_mgr(
    num_layers,
    blocks_in_primary_pool,
    page_size,
    offload_chunksize,
    enable_nvcomp=False,
):
    # Original stress config kept for offload/onload regression
    max_batch_size = 8
    max_seqlen = 10240
    item_fea_name, item_vocab_size = "item_feat", 100
    action_fea_name, action_vocab_size = "act_feat", 128
    _ = [
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
    # randomize kvcache data
    for idx in range(num_layers):
        async_kvcache_mgr.cache_table[idx].uniform_(-0.5, 0.5)
    return async_kvcache_mgr
def get_small_test_kvcache_mgr(
    secondary_kvcache_manager: Optional[object] = None,
    namespace_mode: str = "uid",
    namespace_base: str = "recsys_hstu",
):
    # Lightweight config for fast unit tests
    max_batch_size = 4
    max_seqlen = 256
    num_layers = 2
    num_heads = 2
    head_dim = 64
    page_size = 32
    blocks_in_primary_pool = 512
    offload_chunksize = 128
    inference_dtype = torch.bfloat16
    hstu_config = get_inference_hstu_config(
        hidden_size=num_heads * head_dim,
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
        namespace_mode=namespace_mode,
        namespace_base=namespace_base,
    )
    mgr = AsyncHSTUKVCacheManager(
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
        0,
        kv_cache_config.offload_chunksize,
        -1,
        hstu_config.max_seq_len,
        hstu_config.max_batch_size,
        4 * hstu_config.max_batch_size * hstu_config.max_seq_len,
        kv_cache_config.num_onload_buffer_chunks,
        kv_cache_config.num_offload_buffer_chunks,
        kv_cache_config.num_memcpy_workers,
        kv_cache_config.enable_nvcomp,
        secondary_kvcache_manager,
        kv_cache_config.namespace_mode,
        kv_cache_config.namespace_base,
    )
    return mgr, hstu_config, kv_cache_config
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
        try:
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
        finally:
            _shutdown_mgr(async_kvcache_mgr)

#test the correctness of split prepare_kvcache_async
def test_prepare_kvcache_async_legacy_contract():
    mgr, _, _ = get_small_test_kvcache_mgr()
    try:
        batch_size = 2
        user_ids = [101, 202]
        total_history_lengths = [64, 96]
        result = mgr.prepare_kvcache_async(
            batch_size,
            user_ids,
            total_history_lengths,
            mgr.static_page_ids_gpu_buffer,
            mgr.static_offload_page_ids_gpu_buffer,
            mgr.static_metadata_gpu_buffer,
            mgr.static_onload_handle,
        )
        assert isinstance(result, list)
        assert len(result) == 7
        assert isinstance(result[0], list)  # old_cached_lengths
        assert len(result[0]) == batch_size
        assert int(result[1]) >= 0  # new_tokens
        assert hasattr(result[5], "result")  # kvcache_metadata_fut
        assert hasattr(result[6], "result")  # onload_fut
        # make sure futures are consumable
        result[5].result(timeout=30)
        result[6].result(timeout=30)
    finally:
        _shutdown_mgr(mgr)

#test the equivalence of split lookup and allocate with native prepare_kvcache_async
def test_lookup_allocate_equivalence_with_prepare():
    mgr_split, _, _ = get_small_test_kvcache_mgr()
    mgr_prepare, _, _ = get_small_test_kvcache_mgr()
    try:
        batch_size = 2
        user_ids = [301, 302]
        total_history_lengths = [80, 112]
        lookup = mgr_split.kv_cache_lookup(batch_size, user_ids, total_history_lengths)
        prepare_obj = mgr_split.kv_cache_allocate(
            lookup,
            mgr_split.static_page_ids_gpu_buffer,
            mgr_split.static_offload_page_ids_gpu_buffer,
            mgr_split.static_metadata_gpu_buffer,
            mgr_split.static_onload_handle,
        )
        legacy = mgr_prepare.prepare_kvcache_async(
            batch_size,
            user_ids,
            total_history_lengths,
            mgr_prepare.static_page_ids_gpu_buffer,
            mgr_prepare.static_offload_page_ids_gpu_buffer,
            mgr_prepare.static_metadata_gpu_buffer,
            mgr_prepare.static_onload_handle,
        )
        assert lookup.old_cached_lengths == legacy[0]
        assert int(prepare_obj.new_tokens) == int(legacy[1])
        prepare_obj.kvcache_metadata_fut.result(timeout=30)
        legacy[5].result(timeout=30)
    finally:
        _shutdown_mgr(mgr_split)
        _shutdown_mgr(mgr_prepare)

#test correctness of KVcache metadata after wait
def test_prepare_kvcache_wait_smoke():
    mgr, _, _ = get_small_test_kvcache_mgr()
    try:
        batch_size = 2
        user_ids = [401, 402]
        total_history_lengths = [48, 64]
        result = mgr.prepare_kvcache_async(
            batch_size,
            user_ids,
            total_history_lengths,
            mgr.static_page_ids_gpu_buffer,
            mgr.static_offload_page_ids_gpu_buffer,
            mgr.static_metadata_gpu_buffer,
            mgr.static_onload_handle,
        )
        metadata = mgr.prepare_kvcache_wait(
            result[6],  # onload_fut
            result[5],  # kvcache_metadata_fut
            batch_size,
            result[1],  # new_tokens
            mgr.static_page_ids_gpu_buffer,
            mgr.static_offload_page_ids_gpu_buffer,
            result[2],  # offload_uids_buffer
            result[3],  # metadata_host_buffer
            result[4],  # metadata_gpu_buffer
            mgr.static_onload_handle,
        )
        assert isinstance(metadata, KVCacheMetadata)
        assert metadata.kv_indices is not None
        assert metadata.kv_indptr is not None
        assert int(metadata.new_history_nnz) >= 0
    finally:
        _shutdown_mgr(mgr)

#from_config smoke test and nop secondary test
def test_from_config_smoke_and_nop_secondary():
    max_batch_size = 4
    max_seqlen = 256
    hstu_config = get_inference_hstu_config(
        hidden_size=128,
        num_layers=2,
        num_attention_heads=2,
        head_dim=64,
        max_batch_size=max_batch_size,
        max_seq_len=max_seqlen,
        dtype=torch.bfloat16,
    )
    kv_cfg = get_kvcache_config(
        blocks_in_primary_pool=512,
        page_size=32,
        offload_chunksize=128,
        namespace_mode="uid",
        namespace_base="phase1_test",
    )
    mgr = AsyncHSTUKVCacheManager.from_config(hstu_config, kv_cfg)
    try:
        assert mgr.namespace_mode == kv_cfg.namespace_mode
        assert mgr.namespace_base == kv_cfg.namespace_base
        lookup = mgr.kv_cache_lookup(
            1,
            [777],
            [32],
        )
        assert isinstance(lookup.secondary_lookup, dict)
        assert lookup.secondary_lookup.get("backend") == "nop"
    finally:
        _shutdown_mgr(mgr)
if __name__ == "__main__":
    random.seed(1)
    torch.manual_seed(0)
    test_kvcache_offload_onload()