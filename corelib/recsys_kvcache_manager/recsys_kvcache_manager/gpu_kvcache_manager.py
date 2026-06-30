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

from typing import List, Optional, Tuple, Union

import torch
from kvcache_cpp import GPUKVCacheManagerImpl

from .kvcache_metadata import KVCacheMetadata, get_kvcache_metadata_buffer
from .kvcache_utils import KVLookupResult


class DeviceKVCache:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_tokens_per_page: int,
        num_tokens_per_chunk: int,
        num_primary_cache_pages: int,
        num_buffer_pages: int,
        max_batch_size: int,
        max_sequence_length: int,
        dtype: torch.dtype,
        device_idx: int,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = num_tokens_per_page
        self.chunk_size = num_tokens_per_chunk
        self.num_primary_cache_pages = num_primary_cache_pages
        self.num_buffer_pages = num_buffer_pages
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        self.device_idx = device_idx

        self.gpu_kvcache_tensor: torch.Tensor = torch.empty(
            [
                self.num_layers,
                self.num_primary_cache_pages,
                2,  # k/v
                self.page_size,
                self.num_heads,
                self.head_dim,
            ],
            dtype=self.dtype,
            device=self.device_idx,
        )
        self.gpu_kvcache_tables = list(self.gpu_kvcache_tensor.unbind(dim=0))
        self.impl_ = GPUKVCacheManagerImpl(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.page_size,
            self.chunk_size,
            self.num_primary_cache_pages,
            self.num_buffer_pages,
            self.max_batch_size,
            self.max_sequence_length,
            self.device_idx,
        )
        # Auxiliary metadata
        self.num_sms = torch.cuda.get_device_properties(
            self.device_idx
        ).multi_processor_count

    def get_cache_tables(
        self, layer_idx: int = -1
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if layer_idx == -1:
            return self.gpu_kvcache_tables
        return self.gpu_kvcache_tables[layer_idx]

    def lookup(self, uids: torch.Tensor) -> KVLookupResult:
        cached_start_indices, cached_lengths = self.impl_.lookup(uids)
        return KVLookupResult(
            user_ids=uids,
            gpu_cached_start_indices=cached_start_indices,
            gpu_cached_lengths=cached_lengths,
        )

    def allocate(
        self,
        uids: torch.Tensor,
        seq_hist_lengths: torch.Tensor,  # total history lengths in the sequence of current batch
        lookup_results: KVLookupResult,
        output_kvcache_metadata: Optional[KVCacheMetadata] = None,
    ) -> KVCacheMetadata:
        new_hist_lengths = seq_hist_lengths - lookup_results.cached_lengths

        if output_kvcache_metadata is None:
            batch_size = uids.size(0)
            num_new_tokens = torch.sum(new_hist_lengths, dtype=torch.int32).item()
            num_total_pages = torch.sum(
                torch.ceil(seq_hist_lengths.float() / self.page_size).to(torch.int32)
            ).item()
            output_kvcache_metadata = get_kvcache_metadata_buffer(
                batch_size,
                num_new_tokens,
                num_total_pages,
            )
            output_kvcache_metadata.kv_cache_table = self.gpu_kvcache_tables
        output_kvcache_metadata.max_seqlen = max(seq_hist_lengths).item()
        self.impl_.allocate(
            uids,
            seq_hist_lengths,
            lookup_results.host_cached_lengths,
            output_kvcache_metadata.page_ids_gpu_buffer,
            output_kvcache_metadata.metadata_gpu_buffer,
        )
        return output_kvcache_metadata

    def evict(self, uids: torch.Tensor) -> None:
        for uid in uids.tolist():
            self.impl_.evict(uid)

    def evict_all(self) -> None:
        self.impl_.evict_all()

    # debug use interface. Use `paged_kvcache_ops.append_kvcache` directly.
    def put(
        self,
        k,
        v,
        layer_idx,
        kvcache_metadata: KVCacheMetadata,
        append_offsets: Optional[torch.Tensor] = None,
    ) -> None:
        import paged_kvcache_ops

        assert (
            k.shape == v.shape
        ), f"key and value shape mismatch: {k.shape} vs {v.shape}"
        if k.size(0) == self.num_layers:
            raise NotImplementedError("Only support layer-wise in this implementation.")
        kv_cache_table = self.gpu_kvcache_tables[layer_idx]
        (paged_k_cache, _) = kv_cache_table.unbind(dim=1)
        assert (
            k.shape[-2:] == paged_k_cache.shape[-2:]
        ), f"input k/v shape {k.shape} mismatch with cache shape {paged_k_cache.shape}"
        batch_size = kvcache_metadata.kv_indptr.size(0) - 1
        self.gpu_kvcache_tables[layer_idx] = paged_kvcache_ops.append_kvcache(
            k,
            v,
            kvcache_metadata.batch_indices,
            kvcache_metadata.position,
            append_offsets
            if append_offsets is not None
            else torch.zeros((batch_size,), dtype=torch.int32, device=self.device_idx),
            kvcache_metadata.new_history_nnz_cuda,
            kv_cache_table,
            kvcache_metadata.kv_indices,
            kvcache_metadata.kv_indptr,
            kvcache_metadata.kv_last_page_len,
            0,  # NHD layout
        )

    # debug use interface
    def get(
        self, page_ids, last_page_lens, layer_idx
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: debug interface (for symmetry). Not used in inference pipeline.
        (paged_k_cache, paged_v_cache) = self.gpu_kvcache_tables[layer_idx].unbind(
            dim=1
        )

        k = paged_k_cache[page_ids].view(-1, self.num_heads, self.head_dim).clone()
        v = paged_v_cache[page_ids].view(-1, self.num_heads, self.head_dim).clone()
        k = k[: k.size(0) - (self.page_size - int(last_page_lens)), ...]
        v = v[: v.size(0) - (self.page_size - int(last_page_lens)), ...]
        return k, v

    def revoke_onboard_pages(self, user_ids, onboard_page_starts, num_onboard_pages):
        self.impl_.revoke_onboard_pages(
            user_ids, onboard_page_starts, num_onboard_pages
        )

    # [[ offload critirea ]]
    def check_for_offload(self, uids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.impl_.check_for_offload(
            uids if uids is not None else torch.tensor([], dtype=torch.int64)
        )

    def acquire_offload_pages(
        self,
        uids: torch.Tensor,
        offloaded_lengths: torch.Tensor,
        always_offload: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        return self.impl_.acquire_offload_pages(uids, offloaded_lengths, always_offload)

    def release_offload_pages(
        self,
        uids: torch.Tensor,
        offload_start_indices: torch.Tensor,
        offload_lengths: torch.Tensor,
        offloaded: bool,
    ) -> None:
        return self.impl_.release_offload_pages(
            uids, offload_start_indices, offload_lengths, offloaded
        )


# Backward compatibility alias; prefer DeviceKVCache.
GPUKVCacheManager = DeviceKVCache
