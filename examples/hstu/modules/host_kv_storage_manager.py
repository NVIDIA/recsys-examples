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

import random
import torch
from configs.hstu_config import HSTUConfig
from configs.kv_cache_config import KVCacheConfig, KVCacheMetadata
from modules.jagged_data import JaggedData

import flashinfer
import tensorrt_llm
KVCacheManagerImpl = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
KvCacheConfigCpp = tensorrt_llm.bindings.KvCacheConfig
DataType = tensorrt_llm.bindings.DataType


class HSTUHostKVStorageManager:    
    def __init__(
            self,
            hstu_config: HSTUConfig,
            kv_cache_config: KVCacheConfig) -> None:

        self.num_layers = hstu_config.num_layers
        self.head_dim = hstu_config.kv_channels
        self.num_heads = hstu_config.num_attention_heads
        self.tokens_per_block = kv_cache_config.tokens_per_block
        self.num_cache_blocks = kv_cache_config.blocks_in_primary_pool

        self.num_kv_heads_per_layer = [ hstu_config.num_attention_heads for _ in range(self.num_layers) ]

        if kv_cache_config.max_attention_window is None:
            self.max_attention_window = kv_cache_config.max_seq_len
        else:
            self.max_attention_window = max(kv_cache_config.max_attention_window)

        self.offload_chunksize = int(os.getenv("OFFLOAD_SIZE", 1024))
        self.max_batch_size = int(os.getenv("MAX_BATCHSIZE", 32))

        kv_cache_dtype = torch.bfloat16 if hstu_config.bf16 else torch.float16 if hstu_config.fp16 else torch.float32
        self.lookup_buffer_ = torch.empty((self.num_layers, self.max_batch_size * self.max_attention_window, self.num_heads * self.head_dim),
                                          dtype=kv_cache_dtype, device=torch.device("cpu"), pin_memory=True).uniform_(-0.05, 0.05)

        self.offloaded_history_seq_token_idx = dict()
        self.offloaded_history_seqlen_table = dict()
    
    def load(self, offloaded_history_seqlen, offloaded_history_seq_token_idx):
        pass
    
    def append(self, 
        layer_idx: int,
        user_id: int, 
        value: torch.Tensor, 
    ):
        pass
    
    def lookup(self, 
               user_ids: torch.Tensor, 
               cached_history_seq_token_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        onload_history_seq_length = torch.zeros_like(user_ids, dtype=torch.int32, device=torch.device("cpu"))
        onload_history_seqlen_offsets = torch.zeros((user_ids.shape[0]+1,), dtype=torch.int32, device=torch.device("cpu"))
        for idx in range(user_ids.shape[0]):
            user_id = user_ids[idx].item()
            onload_history_seq_length[idx] = self.get_onload_history_seqlen(user_id, cached_history_seq_token_idx[idx].item())
            
        # copy lookup results into buffer
        torch.cumsum(onload_history_seq_length, 0, out = onload_history_seqlen_offsets[1:])

        onload_length = onload_history_seqlen_offsets[-1]
        onload_kv_page_ids = torch.arange(
            start = self.num_cache_blocks,
            end = onload_length / self.tokens_per_block + self.num_cache_blocks,
            dtype=torch.int32, device=torch.device("cuda:0"))
        onload_kv_page_indptr = (onload_history_seqlen_offsets / self.tokens_per_block).to(dtype=torch.int32, device=torch.device("cuda:0"))

        return onload_kv_page_ids, onload_kv_page_indptr, onload_length
    
    def get_lookup_buffer(self) -> torch.Tensor:
        return self.lookup_buffer_
    
    def get_onload_history_seqlen(self, user_id: int, cached_history_seq_token_idx: int) -> int:
        # if user_id not in self.offloaded_history_seqlen_table:
        #     self.offloaded_history_seqlen_table[user_id] = 0
        # return min(self.offloaded_history_seqlen_table[user_id],
        #     cached_history_seq_token_idx - self.offloaded_history_seq_token_idx.get(user_id, 0))

        # # random test
        # return random.randint(0, 1024//self.tokens_per_block) * self.tokens_per_block

        # fix perf test
        return int(os.getenv("ONLOAD_LEN"))
