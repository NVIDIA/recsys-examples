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

from .hstu_config import HSTUPreprocessingConfig, PositionEncodingConfig


@unique
class EmbeddingBackend(Enum):
    """
    Enum class representing different embedding backends (for inference).

    Attributes:
      TORCHREC: Represents the TorchRec backend.
      DYNAMICEMB: Represents the DynamicEmb backend.
      NVEMB: Represents the NV-Embeddings backend.
    """

    TORCHREC = "TorchRec"
    DYNAMICEMB = "DynamicEmb"
    NVEMB = "NVEmb"


@dataclass
class InferenceEmbeddingConfig:
    backend: EmbeddingBackend
    device: torch.cuda.Device


@dataclass
class KVCacheMetadata:
    """
    KVCacheMetadata is a  data class for the HSTU KV cache metadata of a batch.
    """

    # paged cache metadata
    kv_indices: torch.Tensor = None
    kv_indptr: torch.Tensor = None
    kv_last_page_len: torch.Tensor = None
    total_history_lengths: torch.Tensor = None
    total_history_offsets: torch.Tensor = None

    # appending metadata
    batch_indices: torch.Tensor = None
    position: torch.Tensor = None
    new_history_nnz: int = 0
    new_history_nnz_cuda: torch.Tensor = None

    # onload utility
    onload_history_kv_buffer: Optional[List[torch.Tensor]] = None
    onload_history_kv_events: Optional[List[torch.cuda.Event]] = None

    # paged cache table pointers
    kv_cache_table: Optional[List[torch.Tensor]] = None


@dataclass
class KVCacheConfig:
    """
    KVCacheConfig is a configuration data class for the HSTU KV cache.

    Args:
        blocks_in_primary_pool (int): The number of cache pages per layer.
        page_size (int): The number of tokens per cache page.
        offload_chunksize (int): The size of basic offload data chunk.
        max_attention_window (int): (Optional) The maximum window size for HSTU attention calculation.
    """

    blocks_in_primary_pool: int
    page_size: int
    offload_chunksize: int
    max_attention_window: Optional[int] = None


def get_kvcache_config(
    blocks_in_primary_pool: int,
    page_size: int,
    offload_chunksize: int,
    max_attention_window: Optional[int] = None,
) -> KVCacheConfig:
    """
    Create the HSTU KV cache configuration.

    Args:
        blocks_in_primary_pool (int): The number of cache pages per layer.
        page_size (int): The number of tokens per cache page in the paged KV cache.
        offload_chunksize (int): The size of basic offload data chunk.
        max_attention_window (int): The max attention window size.

    Returns:
        KVCacheConfig: The HSTU KV cache configuration object.
    """

    return KVCacheConfig(  # type: ignore
        blocks_in_primary_pool=blocks_in_primary_pool,
        page_size=page_size,
        offload_chunksize=offload_chunksize,
        max_attention_window=max_attention_window,
    )


@dataclass
class InferenceHSTUConfig:
    """
    InferenceHSTUConfig is a configuration data class for the inference HSTU model.

    Args:
        hidden_size (int): The hidden states dimension size.
        num_layers (int): Number of attention layers.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of key-value channels (per attention head).
        max_batch_size (int): The maximum batch size for the inference input.
        max_seq_len (int): The upper bound of sequence length for each sequence in the inference batch.
        layernorm_epsilon (float): Epsilon value for normalization.
        bf16 (bool): Whether to inference in bfloat16.
        fp16 (bool): Whether to inference in float16.

        learnable_input_layernorm (bool): Whether to have input layernorm weights.
        residual (bool): Whether to add residual connection.
        is_causal (bool):Whether the attention is causal.
        target_group_size (int):  The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking).
        position_encoding_config (PositionEncodingConfig, optional): Position embedding config.
        hstu_preprocessing_config (HSTUPreprocessingConfig, optional): HSTU preprocessing config.
        contextual_max_seqlen (int): The (maximum) length of contextual features.
        embedding_backend (EmbeddingBackend, optional): Embedding backend to use.
    """

    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    max_batch_size: int
    max_seq_len: int
    layernorm_epsilon: float = 1e-5
    bf16: bool = True
    fp16: bool = False

    learnable_input_layernorm: bool = True
    residual: bool = True
    is_causal: bool = True
    target_group_size: int = 1
    position_encoding_config: Optional[PositionEncodingConfig] = None
    hstu_preprocessing_config: Optional[HSTUPreprocessingConfig] = None
    contextual_max_seqlen: int = 0
    scaling_seqlen: int = -1
    embedding_backend: Optional[EmbeddingBackend] = None

    def __post_init__(self):
        assert self.is_causal
        assert self.target_group_size == 1


def get_inference_hstu_config(
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    head_dim: int,
    max_batch_size: int,
    max_seq_len: int,
    norm_epsilon=1e-5,
    dtype: torch.dtype = torch.bfloat16,
    learnable_input_layernorm: bool = True,
    residual: bool = True,
    is_causal: bool = True,
    target_group_size: int = 1,
    position_encoding_config: Optional[PositionEncodingConfig] = None,
    contextual_max_seqlen: int = 0,
    scaling_seqlen: int = -1,
    embedding_backend=None,
) -> InferenceHSTUConfig:
    """
    Create the HSTU configuration.

    Args:
        hidden_size (int): The hidden dimension size.
        num_layers (int): Number of attention layers.
        num_attention_heads (int): Number of attention heads.
        head_dim (int): Number of key-value channels (per attention head).
        max_batch_size (int): The maximum batch size for the inference input.
        max_seq_len (int): The upper bound of sequence length for each sequence in the inference batch.
        norm_epsilon (float, optional): Epsilon value for normalization. Defaults to 1e-5.
        dtype (torch.dtype): Data type (e.g., torch.float16).
        learnable_input_layernorm (bool, optional): Whether to have input layernorm weights. Defaults to True.
        residual (bool, optional): Whether to add residual connection. Defaults to True.
        is_causal (bool, optional): Whether the attention is causal. Defaults to False.
        target_group_size (int, optional): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
        position_encoding_config (Optional[PositionEncodingConfig], optional): Position embedding config. Defaults to None.
        contextual_max_seqlen (int, optional): The (maximum) length of contextual features.
    Returns:
        HSTUConfig: The HSTU configuration object.
    """
    is_bf16 = dtype == torch.bfloat16
    is_fp16 = dtype == torch.float16
    return InferenceHSTUConfig(  # type: ignore
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_attention_heads,
        head_dim=head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        layernorm_epsilon=norm_epsilon,
        bf16=is_bf16,
        fp16=is_fp16,
        learnable_input_layernorm=learnable_input_layernorm,
        residual=residual,
        is_causal=is_causal,
        target_group_size=target_group_size,
        position_encoding_config=position_encoding_config,
        contextual_max_seqlen=contextual_max_seqlen,
        scaling_seqlen=scaling_seqlen,
        embedding_backend=embedding_backend,
    )


def get_kvcache_metadata_buffer(
    hstu_config: InferenceHSTUConfig, kvcache_config: KVCacheConfig
):
    device = torch.cuda.current_device()
    torch.bfloat16 if hstu_config.bf16 else torch.float16 if hstu_config.fp16 else torch.float32

    max_new_history_seqlen = hstu_config.max_batch_size * hstu_config.max_seq_len
    max_num_pages_per_seq = (
        hstu_config.max_seq_len + hstu_config.max_seq_len + kvcache_config.page_size - 1
    ) // kvcache_config.page_size
    max_host_kv_buffer_size = (
        hstu_config.max_batch_size * hstu_config.max_seq_len,
        hstu_config.num_heads * hstu_config.head_dim,
    )

    default_num_pages_per_seq = 4
    paged_indices_buffer = torch.randint(
        kvcache_config.blocks_in_primary_pool,
        (hstu_config.max_batch_size * max_num_pages_per_seq,),
        dtype=torch.int32,
        device=device,
    )
    page_indptr_buffer = (
        torch.arange(hstu_config.max_batch_size + 1, dtype=torch.int32, device=device)
        * default_num_pages_per_seq
    )
    last_page_lens_buffer = torch.full(
        (hstu_config.max_batch_size,),
        kvcache_config.page_size,
        dtype=torch.int32,
        device=device,
    )
    batch_indices_buffer = torch.zeros(
        (max_new_history_seqlen,), dtype=torch.int32, device=device
    )
    position_buffer = torch.zeros(
        (max_new_history_seqlen,), dtype=torch.int32, device=device
    )
    total_history_offsets_buffer = (
        torch.arange(hstu_config.max_batch_size + 1, dtype=torch.int32, device=device)
        * default_num_pages_per_seq
        * kvcache_config.page_size
    )
    return KVCacheMetadata(
        kv_indices=paged_indices_buffer,
        kv_indptr=page_indptr_buffer,
        kv_last_page_len=last_page_lens_buffer,
        batch_indices=batch_indices_buffer,
        position=position_buffer,
        new_history_nnz=max_new_history_seqlen,
        new_history_nnz_cuda=torch.ones((1,), dtype=torch.int32, device=device),
        total_history_offsets=total_history_offsets_buffer,
        onload_history_kv_buffer=[],
        onload_history_kv_events=[],
    )


def copy_kvcache_metadata(dst_metadata: KVCacheMetadata, src_metata: KVCacheMetadata):
    def copy_tensor(dst, src):
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = 0

    def copy_offsets(dst, src):
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = src[-1, ...]

    copy_tensor(dst_metadata.kv_indices, src_metata.kv_indices)
    copy_offsets(dst_metadata.kv_indptr, src_metata.kv_indptr)
    copy_tensor(dst_metadata.kv_last_page_len, src_metata.kv_last_page_len)
    copy_tensor(dst_metadata.batch_indices, src_metata.batch_indices)
    copy_tensor(dst_metadata.position, src_metata.position)
    copy_tensor(dst_metadata.new_history_nnz_cuda, src_metata.new_history_nnz_cuda)
    copy_offsets(dst_metadata.total_history_offsets, src_metata.total_history_offsets)

    dst_metadata.new_history_nnz = src_metata.new_history_nnz
