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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

import os
from typing import Optional, Tuple

import torch
from configs.hstu_config import HSTUConfig
from configs.kv_cache_config import KVCacheConfig, KVCacheMetadata
from dataset.utils import RankingBatch
from modules.jagged_data import JaggedData

import flashinfer
import tensorrt_llm
KVCacheManagerImpl = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
KvCacheConfigCpp = tensorrt_llm.bindings.KvCacheConfig
DataType = tensorrt_llm.bindings.DataType


class HSTUGpuKVCacheManager:    
    def __init__(
            self,
            hstu_config: HSTUConfig,
            kv_cache_config: KVCacheConfig) -> None:

        self.num_layers = hstu_config.num_layers
        self.head_dim = hstu_config.kv_channels
        self.tokens_per_block = kv_cache_config.tokens_per_block
        self.num_cache_blocks = kv_cache_config.blocks_in_primary_pool
        self.max_batch_size = kv_cache_config.max_batch_size

        self.num_kv_heads_per_layer = [ hstu_config.num_attention_heads for _ in range(self.num_layers) ]

        if kv_cache_config.max_attention_window is None:
            self.max_attention_window = kv_cache_config.max_seq_len
        else:
            self.max_attention_window = max(kv_cache_config.max_attention_window)

        self._onload_stream = torch.cuda.Stream()
        self._offload_stream = torch.cuda.Stream()
        self.offload_chunksize = int(os.getenv("OFFLOAD_SIZE", 1024))
        kwargs = {
            'num_kv_heads_per_layer': self.num_kv_heads_per_layer,
            'size_per_head': self.head_dim,
            'tokens_per_block': self.tokens_per_block,
            'blocks_in_primary_pool': kv_cache_config.blocks_in_primary_pool,
            'blocks_in_secondary_pool': 0,
            'max_num_sequences': kv_cache_config.max_batch_size,
            'max_beam_width': 1,
            'max_attention_window': self.max_attention_window,
            'temporary_attention_window': 0,
            'sink_token_length': 0,
            'stream': self._offload_stream.cuda_stream,
            'max_sequence_length': kv_cache_config.max_seq_len,
            # 'enable_block_reuse': kv_cache_config.enable_block_reuse,
            # 'onboard_blocks': kv_cache_config.onboard_blocks,
            # 'offload_chunksize': self.offload_chunksize,
            # 'cache_type': CacheTypeCpp.SELF,
        }
        self.impl = KVCacheManagerImpl(**kwargs)

        kv_cache_dtype = DataType.BF16 if hstu_config.bf16 else DataType.HALF if hstu_config.fp16 else DataType.FLOAT
        self.impl.allocate_pools(kv_cache_dtype, False)
        # self.kv_cache_pool_pointers = self.impl.get_block_pool_pointers()
        # self.kv_cache_pool_mapping = self.impl.get_layer_to_pool_mapping()
        # self.num_pools = self.impl.num_pools
        # self.max_blocks_per_seq = self.impl.max_blocks_per_seq

        self.host_kvdata_gpu_buffer_ = torch.empty((),
            dtype=torch.bfloat16, device=torch.cuda.current_device())

    def allocate(self, batch: RankingBatch, user_ids: torch.Tensor):
        batch_size = user_ids.shape[0]
        seq_lengths = batch.features.lengths()[:batch_size] + torch.zeros_like(batch.features.lengths()[:batch_size])
        for idx in range(batch_size, batch.features.lengths().shape[0], batch_size):
            seq_lengths += batch.features.lengths()[idx:idx+batch_size]
        seq_lengths = seq_lengths.to('cpu')
        num_candidates = batch.num_candidates.to('cpu') * 2
        # allocate KV Cache
        for idx in range(batch_size):
            user_id = user_ids[idx].item()
            req_beam_width = 1
            delta_hist_length = seq_lengths[idx] - num_candidates[idx]
            self.impl.add_sequence_with_eviction(user_id, delta_hist_length, req_beam_width, None)
        
        return
    
    def append_paged_kv_data(self, 
        layer_idx: int,
        key: torch.Tensor, 
        value: torch.Tensor, 
        metadata: KVCacheMetadata,
        seqlen_offsets: torch.Tensor,
        num_candidates_offsets: torch.Tensor
    ):  
        
        return
    
    def evict(self, user_ids):
        for idx in range(user_ids.shape[0]):
            self.impl.remove_sequence(user_ids[idx].item(), None)
    
    def lookup(self, user_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = user_ids.shape[0]
        num_cached_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=torch.device('cpu'))
        num_offloaded_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=torch.device('cpu'))
        for idx in range(batch_size):
            user_id = user_ids[idx].item()
            num_cached_lengths[idx] = self.impl.get_num_tokens_cached(user_id)
            num_offloaded_lengths[idx] = self.impl.get_num_tokens_offload(user_id)

        return (num_cached_lengths, num_offloaded_lengths)
    
    def onboard(self, host_kv_data: torch.Tensor, onload_length: int, kv_cache_metadata):
        with torch.cuda.stream(self._onload_stream):
            for layer_idx in range(self.num_layers):
                kv_cache_metadata.onload_history_kv_buffer[layer_idx][:onload_length, ...].copy_(
                    host_kv_data[layer_idx, :onload_length, ...], non_blocking=True)
                kv_cache_metadata.onload_history_kv_events[layer_idx].record(self._onload_stream)
    
    def get_cache_page_metadata(self, batch: RankingBatch, user_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = user_ids.shape[0]
        
        page_ids = list()
        num_pages = 0
        kv_cache_page_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device=torch.device('cpu'))
        kv_cache_last_page_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=torch.device('cpu'))
        cached_history_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=torch.device('cpu'))

        last_fakeout_page_id = 4096
        for idx in range(batch_size):
            user_id = user_ids[idx]
            
            cached_history_length = int(os.getenv("CACHED_LEN")) + self.impl.get_num_tokens_cached(user_id)
            cached_history_lengths[idx] = cached_history_length
            last_page_length = cached_history_length % self.tokens_per_block
            last_page_length = self.tokens_per_block if last_page_length == 0 else last_page_length
            kv_cache_last_page_lengths[idx] = last_page_length

            kv_page_ids = self.impl.get_cache_block_ids(user_id)[0]
            fakeout_cached_pages_num = (cached_history_length - last_page_length)//self.tokens_per_block + 1 - len(kv_page_ids)
            page_ids.extend(kv_page_ids)
            page_ids.extend([ _ for _ in range(last_fakeout_page_id, last_fakeout_page_id+fakeout_cached_pages_num)])
            kv_cache_page_indptr[idx + 1] = len(page_ids)
            last_fakeout_page_id = last_fakeout_page_id+fakeout_cached_pages_num

        device = batch.features.values().device
        kv_cache_page_ids = torch.tensor(page_ids, dtype=torch.int32, device=device)
        kv_cache_page_indptr = kv_cache_page_indptr.to(device)
        kv_cache_last_page_lengths = kv_cache_last_page_lengths.to(device)
        cached_history_lengths = cached_history_lengths.to(device)
        
        seq_lengths = batch.features.lengths()[:batch_size] + torch.zeros_like(batch.features.lengths()[:batch_size])
        for idx in range(batch_size, batch.features.lengths().shape[0], batch_size):
            seq_lengths += batch.features.lengths()[idx:idx+batch_size]
        seq_lengths = seq_lengths.to('cpu')
        num_candidates = batch.num_candidates.to('cpu') * 2

        delta_history_length = seq_lengths - num_candidates

        delta_history_offsets = torch.zeros((batch_size+1,), dtype=torch.int32, device=torch.device("cpu"))
        torch.cumsum(delta_history_length, 0, out = delta_history_offsets[1:])
        delta_history_token_nnz = delta_history_offsets[-1].item()
        delta_history_offsets = delta_history_offsets.to(device=device)
        history_batch_indices, history_positions = flashinfer.page.get_batch_indices_positions(
            delta_history_offsets,
            cached_history_lengths,
            delta_history_token_nnz
        )
        new_history_nnz_cuda=torch.full((1,), delta_history_token_nnz, dtype=torch.int32, device=device)

        max_delta_history_length = torch.max(delta_history_length).item()
        max_num_candidate = torch.max(num_candidates).item()
        return KVCacheMetadata(
            kv_indices=kv_cache_page_ids,
            kv_indptr=kv_cache_page_indptr,
            kv_last_page_len=kv_cache_last_page_lengths,
            batch_indices=history_batch_indices,
            position=history_positions,
            delta_history_token_nnz=delta_history_token_nnz,
            new_history_nnz_cuda=new_history_nnz_cuda
        )
    
    def get_page_size(self) -> int:
        return self.tokens_per_block
    
    def get_buffers(self, layer_idx: int) -> Optional[torch.Tensor]:
        result = self.impl.get_primary_pool()
        result = result.view(result.shape[1], result.shape[0], result.shape[2], result.shape[3])
        result = result[layer_idx, ...]
        # result = self.impl.get_primary_pool_data(layer_idx)
        return result.reshape(result.shape[0], 2, self.tokens_per_block,
                              self.num_kv_heads_per_layer[layer_idx],
                              self.head_dim).to(torch.bfloat16)
