# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List, Optional

import torch


@dataclass
class KVCacheMetadata:
    """
    KVCacheMetadata is a  data class for the HSTU KV cache metadata of a batch.
    """

    page_ids_gpu_buffer: torch.Tensor
    metadata_gpu_buffer: torch.Tensor

    # paged cache metadata
    kv_indices: torch.Tensor = None  # num_pages
    kv_indptr: torch.Tensor = None  # num_seq + 1
    kv_last_page_len: torch.Tensor = None  # num_seq
    total_history_lengths: torch.Tensor = None  # num_seq
    total_history_offsets: torch.Tensor = None  # num_seq + 1
    new_history_offsets: torch.Tensor = None  # num_seq + 1

    # appending metadata
    batch_indices: torch.Tensor = None  # num_tokens
    position: torch.Tensor = None  # num_tokens
    new_history_nnz: int = 0
    new_history_nnz_cuda: torch.Tensor = None  # 1

    # attention kv seqlen & offsets
    kv_seqlens: Optional[torch.Tensor] = None  # num_seq + 1
    kv_seqlen_offsets: Optional[torch.Tensor] = None  # num_seq + 1

    # paged cache table pointers
    kv_cache_table: Optional[List[torch.Tensor]] = None

    # async attributes
    kv_onload_handle: Optional[object] = None
    # kv_offload_handle: Optional[object] = None

    max_seqlen: Optional[int] = 0

    onboard_slot_mappings: Optional[List[torch.Tensor]] = None
    onboard_task_ids: Optional[torch.Tensor] = None


def get_kvcache_metadata_buffer(
    batch_size: int,
    num_new_tokens: int,
    num_pages: int,
    page_ids_gpu_buffer: Optional[torch.Tensor] = None,
    metadata_gpu_buffer: Optional[torch.Tensor] = None,
    device: Optional[int] = None,
):
    if device is None:
        device = torch.cuda.current_device()

    page_ids_gpu_buffer = (
        torch.empty(
            (num_pages,),
            dtype=torch.int32,
            device=device,
        )
        if page_ids_gpu_buffer is None
        else page_ids_gpu_buffer
    )

    metadata_gpu_buffer = (
        torch.empty(
            (5 * batch_size + 4 + num_new_tokens * 2,),
            dtype=torch.int32,
            device=device,
        )
        if metadata_gpu_buffer is None
        else metadata_gpu_buffer
    )

    page_indptr_buffer = metadata_gpu_buffer.narrow(0, 0, batch_size + 1)
    last_page_lens_buffer = metadata_gpu_buffer.narrow(0, batch_size + 1, batch_size)
    total_history_lengths = metadata_gpu_buffer.narrow(
        0, batch_size * 2 + 1, batch_size
    )
    total_history_offsets = metadata_gpu_buffer.narrow(
        0, batch_size * 3 + 1, batch_size + 1
    )
    new_history_nnz_cuda = metadata_gpu_buffer.narrow(0, batch_size * 4 + 2, 1)
    new_history_offsets = metadata_gpu_buffer.narrow(
        0, batch_size * 4 + 3, batch_size + 1
    )
    batch_indices_buffer = metadata_gpu_buffer.narrow(
        0, batch_size * 5 + 4, num_new_tokens
    )
    position_buffer = metadata_gpu_buffer.narrow(
        0, batch_size * 5 + 4 + num_new_tokens, num_new_tokens
    )

    kv_seqlens = torch.empty_like(total_history_lengths)
    kv_seqlen_offsets = torch.empty_like(total_history_offsets)

    return KVCacheMetadata(
        page_ids_gpu_buffer=page_ids_gpu_buffer,
        metadata_gpu_buffer=metadata_gpu_buffer,
        kv_indices=page_ids_gpu_buffer,
        kv_indptr=page_indptr_buffer,
        kv_last_page_len=last_page_lens_buffer,
        total_history_lengths=total_history_lengths,
        total_history_offsets=total_history_offsets,
        new_history_offsets=new_history_offsets,
        batch_indices=batch_indices_buffer,
        position=position_buffer,
        new_history_nnz=num_new_tokens,
        new_history_nnz_cuda=new_history_nnz_cuda,
        kv_seqlens=kv_seqlens,
        kv_seqlen_offsets=kv_seqlen_offsets,
        kv_onload_handle=None,
    )


def copy_kvcache_metadata(dst_metadata: KVCacheMetadata, src_metadata: KVCacheMetadata):
    def copy_tensor(dst, src):
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = 0

    def copy_offsets(dst, src):
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = src[-1, ...]

    copy_tensor(dst_metadata.kv_indices, src_metadata.kv_indices)
    copy_offsets(dst_metadata.kv_indptr, src_metadata.kv_indptr)
    copy_tensor(dst_metadata.kv_last_page_len, src_metadata.kv_last_page_len)
    copy_tensor(dst_metadata.total_history_lengths, src_metadata.total_history_lengths)
    copy_tensor(dst_metadata.batch_indices, src_metadata.batch_indices)
    copy_tensor(dst_metadata.position, src_metadata.position)
    copy_tensor(dst_metadata.new_history_nnz_cuda, src_metadata.new_history_nnz_cuda)
    copy_offsets(dst_metadata.total_history_offsets, src_metadata.total_history_offsets)
    copy_offsets(dst_metadata.new_history_offsets, src_metadata.new_history_offsets)
    copy_tensor(dst_metadata.kv_seqlens, src_metadata.kv_seqlens)
    copy_offsets(dst_metadata.kv_seqlen_offsets, src_metadata.kv_seqlen_offsets)

    dst_metadata.new_history_nnz = src_metadata.new_history_nnz

    dst_metadata.kv_onload_handle = src_metadata.kv_onload_handle
    # dst_metadata.kv_offload_handle = src_metadata.kv_offload_handle
