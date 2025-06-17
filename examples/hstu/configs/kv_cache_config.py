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
from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Optional

import torch
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig
from . import HSTUConfig
import tensorrt_llm
DataType = tensorrt_llm.bindings.DataType


@dataclass
class KVCacheMetadata:
    """
    KVCacheMetadata is a  data class for the HSTU KV cache metadata of a batch.
    """

    kv_indices: torch.Tensor
    kv_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
    batch_indices: torch.Tensor
    position: torch.Tensor
    # delta_history_offsets: torch.Tensor
    # total_history_lengths: torch.Tensor
    # max_delta_history_length: int
    # max_num_candidate: int
    delta_history_token_nnz: int
    new_history_nnz_cuda: torch.Tensor

    kv_cache_table: List[torch.Tensor] = None

    onload_history_kv_buffer: List[torch.Tensor] = None
    onload_history_kv_events: List[torch.cuda.Event] = None

    # onload_history_kv_page_ids: torch.Tensor = None
    # onload_history_kv_page_indptr: torch.Tensor = None
    # onload_seqlen: int = 0
    
    ts_onboard_start: torch.cuda.Event = None
    ts_first_layer_bp: torch.cuda.Event = None
    ts_last_layer_end: torch.cuda.Event = None

@dataclass
class KVCacheConfig:
    """
    KVCacheConfig is a configuration data class for the HSTU KV cache.

    Args:
        blocks_in_primary_pool (int): The number of cache pages per layer.
        tokens_per_block (int): The number of tokens per cache page.
        max_batch_size (int): The max batch size.
        max_seq_len (int): The upper bound of sequence length in cache.
        max_attention_window (int): The max attention window size.
    """

    blocks_in_primary_pool: int
    tokens_per_block: int
    max_batch_size: int
    max_seq_len: int
    max_attention_window: int = None


def get_kvcache_config(
    blocks_in_primary_pool,
    tokens_per_block,
    max_batch_size,
    max_seq_len,
    max_attention_window = None,
) -> KVCacheConfig:
    """
    Create the HSTU KV cache configuration.

    Args:
        blocks_in_primary_pool (int): The number of cache pages per layer.
        tokens_per_block (int): The number of tokens per cache page.
        max_batch_size (int): The max batch size.
        max_gpu_cache_seqlen (int): The upper bound of sequence length in gpu cache.
        max_host_cache_seqlen (int): The upper bound of sequence length in host cache.
        max_attention_window (int): The max attention window size.

    Returns:
        KVCacheConfig: The HSTU KV cache configuration object.
    """

    return KVCacheConfig(  # type: ignore
        blocks_in_primary_pool=blocks_in_primary_pool,
        tokens_per_block=tokens_per_block,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        max_attention_window=max_attention_window,
    )

def get_kvcache_metadata_buffer(hstu_config: HSTUConfig, kvcache_config: KVCacheConfig):
    device = torch.cuda.current_device()
    dtype = torch.bfloat16 if hstu_config.bf16 else torch.float16 if hstu_config.fp16 else torch.float32
    
    max_new_history_seqlen = kvcache_config.max_batch_size * kvcache_config.max_seq_len
    max_num_pages_per_seq = (kvcache_config.max_seq_len + kvcache_config.max_seq_len   # change to gpu/cpu len
        + kvcache_config.tokens_per_block - 1) // kvcache_config.tokens_per_block
    max_host_kv_buffer_size = (
        kvcache_config.max_batch_size * kvcache_config.max_seq_len,  # change to cpu len
        hstu_config.num_attention_heads * hstu_config.kv_channels
    )

    paged_indices_buffer = torch.randint(
        kvcache_config.blocks_in_primary_pool, (kvcache_config.max_batch_size * max_num_pages_per_seq,),
        dtype=torch.int32, device=device)
    page_indptr_buffer = torch.arange(
        kvcache_config.max_batch_size + 1, dtype=torch.int32, device=device) * max_num_pages_per_seq
    last_page_lens_buffer = torch.full(
        (kvcache_config.max_batch_size, ), kvcache_config.tokens_per_block, dtype=torch.int32, device=device)
    batch_indices_buffer = torch.zeros(
        (max_new_history_seqlen, ), dtype=torch.int32, device=device)
    position_buffer = torch.zeros(
        (max_new_history_seqlen, ), dtype=torch.int32, device=device)
    return KVCacheMetadata(
            kv_indices = paged_indices_buffer,
            kv_indptr = page_indptr_buffer,
            kv_last_page_len = last_page_lens_buffer,
            batch_indices = batch_indices_buffer,
            position = position_buffer,
            delta_history_token_nnz = max_new_history_seqlen,
            new_history_nnz_cuda = torch.ones((1), dtype=torch.int32, device=device),
            onload_history_kv_buffer = [ 
                torch.empty(max_host_kv_buffer_size, dtype=dtype, device=device)
                for _ in range(hstu_config.num_layers) ],
            onload_history_kv_events = [ torch.cuda.Event() for _ in range(hstu_config.num_layers) ]
        )

def copy_kvcache_metadata(dst_metadata, src_metata):

    def copy_tensor(dst, src):
        dst[:src.shape[0], ...].copy_(src, non_blocking=True)
    
    copy_tensor(dst_metadata.kv_indices, src_metata.kv_indices)
    copy_tensor(dst_metadata.kv_indptr, src_metata.kv_indptr)
    copy_tensor(dst_metadata.kv_last_page_len, src_metata.kv_last_page_len)
    copy_tensor(dst_metadata.batch_indices, src_metata.batch_indices)
    copy_tensor(dst_metadata.position, src_metata.position)
    copy_tensor(dst_metadata.new_history_nnz_cuda, src_metata.new_history_nnz_cuda)

    dst_metadata.delta_history_token_nnz = src_metata.delta_history_token_nnz