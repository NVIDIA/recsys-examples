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
from typing import Optional

import torch
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig
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
    delta_history_offsets: torch.Tensor
    total_history_lengths: torch.Tensor
    max_delta_history_length: int
    max_num_candidate: int

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
        max_seq_len (int): The upper bound of sequence length in cache.
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
