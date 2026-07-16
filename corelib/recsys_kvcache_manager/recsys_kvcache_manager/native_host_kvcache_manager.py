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

import time
from typing import List, Optional, Tuple

import torch
from kvcache_cpp import HostKVStorageImpl, KVOffloadHandle, KVOnloadHandle

from .host_kvstorage_manager import (
    HostKVStorageBase,
    HostKVTaskHandle,
    HostKVTaskStatus,
    HostKVWaitResult,
)
from .kvcache_metadata import KVCacheMetadata
from .kvcache_utils import KVIndexMeta, KVLookupResult


class NativeHostKVStorage(HostKVStorageBase):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_tokens_per_page: int,
        num_tokens_per_chunk: int,
        bytes_capacity_per_layer: int,
        max_batch_size: int,
        max_sequence_length: int,
        onload_timeout_ms: float = 0.0,
        offload_timeout_ms: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
        device_idx: int = 0,
    ):
        self.backend_name = "native"

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = num_tokens_per_page
        self.chunk_size = num_tokens_per_chunk
        self.bytes_capacity_per_layer = bytes_capacity_per_layer
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        self.device_idx = device_idx

        self.impl_ = HostKVStorageImpl(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.page_size,
            self.chunk_size,
            self.bytes_capacity_per_layer,
            self.max_batch_size,
            self.max_sequence_length,
            self.device_idx,
        )

        self._onload_timeout_ms = onload_timeout_ms
        self._offload_timeout_ms = offload_timeout_ms

    def register_gpu_cache_tables(self, cache_table_list: List[torch.Tensor]):
        self.impl_.register_gpu_cache_table(cache_table_list)

    def build_index_meta(
        self, user_ids: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> KVIndexMeta:
        index_meta = KVIndexMeta(
            user_ids=user_ids,
            seq_lengths=sequence_lengths,
        )
        return index_meta

    def lookup_kvcache(self, index_meta: KVIndexMeta) -> KVLookupResult:
        cached_lengths = self.impl_.lookup(index_meta.user_ids)
        cached_start_indices = torch.zeros_like(cached_lengths)
        return KVLookupResult(
            user_ids=index_meta.user_ids,
            host_cached_start_indices=cached_start_indices,
            host_cached_lengths=cached_lengths,
        )

    def onboard_kvcache_launch(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata,
    ) -> HostKVTaskHandle:
        g_end_idxs = (
            lookup_result.gpu_cached_start_indices + lookup_result.gpu_cached_lengths
        )
        h_longer = g_end_idxs < lookup_result.host_cached_lengths

        onload_start_indices = torch.where(
            torch.logical_and(lookup_result.gpu_cached_start_indices == 0, h_longer),
            g_end_idxs,
            0,
        )
        onload_lengths = torch.where(
            h_longer,
            lookup_result.host_cached_lengths,
            lookup_result.gpu_cached_start_indices,
        )

        onload_paged_ids_list = [
            kvcache_metadata.kv_indices.narrow(
                0,  # dim
                kvcache_metadata.kv_indptr[seq_idx]
                + onload_start_indices[seq_idx].item()
                // self.page_size,  # start: gpu, this is allowed to be Tensor. TODO(junyiq): check if there is d2h
                onload_lengths[seq_idx].item() // self.page_size,  # length
            )
            for seq_idx in range(index_meta.user_ids.size(0))
        ]
        if torch.sum(onload_lengths).item() == 0:
            # Note: No data to onboard, skip the task.
            #       Whether to onboard for each used is decided in C++ implementation
            #       from `onload_paged_ids_list`.
            return HostKVTaskHandle(
                backend="native",
                handle=None,
                status=HostKVTaskStatus.SKIPPED,
            )

        native_handle = KVOnloadHandle(self.num_layers)
        self.impl_.onload_kvcache(
            index_meta.user_ids, onload_paged_ids_list, native_handle
        )
        return HostKVTaskHandle(
            backend="native",
            user_ids=index_meta.user_ids,
            handle=native_handle,
            status=HostKVTaskStatus.LAUNCHED,
            is_layerwise=True,
            metadata={
                "onboard_start_indices": onload_start_indices,
                "onboard_lengths": onload_lengths,
            },
        )

    def onboard_kvcache_wait(self, task_handle: HostKVTaskHandle) -> HostKVWaitResult:
        # Use layerwise data transfer and sync for native backend.
        return HostKVWaitResult(
            status=HostKVTaskStatus.SKIPPED,
            ready=False,
        )

    def onboard_kvcache_wait_by_layer(
        self, task_handle, layer_idx: int
    ):  # wait for a single layer
        task_handle.stream_wait_layer(layer_idx)
        return HostKVWaitResult(
            status=HostKVTaskStatus.EVENT_READY,  # the following get on the default stream will be synced
            ready=True,
        )

    def offload_kvcache_launch(
        self,
        offload_user_ids: torch.Tensor,
        offload_start_indices: torch.Tensor,
        offload_page_indices_list: List[torch.Tensor],
        index_meta: Optional[KVIndexMeta] = None,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ) -> Optional[HostKVTaskHandle]:
        native_handle = KVOffloadHandle(self.num_layers)
        ret = self.impl_.offload_kvcache(
            offload_user_ids,
            offload_start_indices,
            offload_page_indices_list,
            native_handle,
        )
        if not ret:
            return None

        return HostKVTaskHandle(
            backend="native",
            user_ids=offload_user_ids,
            handle=native_handle,
            status=HostKVTaskStatus.LAUNCHED,
            time_launched=time.perf_counter_ns(),
        )

    def offload_kvcache_wait(self, task_handle: HostKVTaskHandle) -> HostKVWaitResult:
        is_ready = task_handle.handle.try_wait_layer(-1)
        elapsed_time = (time.perf_counter_ns() - task_handle.time_launched) / 1000_000.0
        # print(f"[DEBUG] Offload elapsed time: {elapsed_time} ms")
        return HostKVWaitResult(
            status=HostKVTaskStatus.READY
            if is_ready
            else HostKVTaskStatus.TIMEOUT
            if (
                self._offload_timeout_ms > 0 and elapsed_time > self._offload_timeout_ms
            )
            else HostKVTaskStatus.LAUNCHED,
            ready=is_ready,
        )

    def finish_task(self, task_handle: HostKVTaskHandle) -> List[int]:
        if isinstance(task_handle.handle, KVOnloadHandle):
            raise NotImplementedError(
                "Finish onload by layer is supported, but not the whole task at once, since the native implementation uses layerwise synchronization."
            )
        elif isinstance(task_handle.handle, KVOffloadHandle):
            return self.impl_.finish_offload(task_handle.handle)
        else:
            raise ValueError(f"Unknown task handle type: {type(task_handle.handle)}")

        # Should not reach
        # return [0 for _ in range(task_handle.handle.get_user_ids().size(0))]

    def cancel_task(self, task_handle: HostKVTaskHandle) -> bool:
        if isinstance(task_handle.handle, KVOnloadHandle):
            raise NotImplementedError(
                "Cancel onload is not supported in the current native implementation."
            )
        elif isinstance(task_handle.handle, KVOffloadHandle):
            return self.impl_.cancel_offload(task_handle.handle)
        else:
            raise ValueError(f"Unknown task handle type: {type(task_handle.handle)}")

        # Should not reach
        # return False

    def evict(self, user_ids: torch.Tensor) -> None:
        for uid in user_ids.tolist():
            self.impl_.evict(uid)

    def evict_all(self) -> None:
        self.impl_.evict_all()

    @staticmethod
    def get_offload_handle_metadata(
        task_handle,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            task_handle.handle.get_user_ids(),
            task_handle.handle.get_start_indices(),
            task_handle.handle.get_lengths(),
        )


# Backward compatibility alias; prefer NativeHostKVStorage.
NativeHostKVCacheManager = NativeHostKVStorage
