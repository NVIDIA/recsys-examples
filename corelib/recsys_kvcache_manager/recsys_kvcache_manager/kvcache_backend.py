# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend abstraction for KV cache management."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from .host_kvstorage_manager import HostKVTaskHandle, HostKVWaitResult
from .kvcache_metadata import KVCacheMetadata
from .kvcache_utils import KVIndexMeta, KVLookupResult


class KVCacheBackend(ABC):
    @abstractmethod
    def lookup_kvcache(
        self,
        user_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> Tuple[KVIndexMeta, KVLookupResult]:
        ...

    @abstractmethod
    def allocate_kvcache(
        self,
        index_meta: KVIndexMeta,
        lookup_results: KVLookupResult,
        output_kvcache_metadata: Optional[KVCacheMetadata] = None,
    ) -> KVCacheMetadata:
        ...

    @abstractmethod
    def onboard_launch(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata,
    ) -> HostKVTaskHandle:
        ...

    @abstractmethod
    def onboard_try_wait(
        self,
        kv_index_meta: KVIndexMeta,
        task_handle: Optional[HostKVTaskHandle],
    ) -> Optional[HostKVWaitResult]:
        ...

    @abstractmethod
    def onboard_wait(
        self,
        kv_index_meta: KVIndexMeta,
        task_handle: Optional[HostKVTaskHandle],
    ) -> Optional[HostKVWaitResult]:
        ...

    @abstractmethod
    def offload_launch(
        self,
        index_meta: KVIndexMeta,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ):
        ...

    @abstractmethod
    def offload_try_wait(self) -> None:
        ...

    @abstractmethod
    def evict(
        self, user_ids: torch.Tensor, for_gpu: bool = False, for_host: bool = False
    ):
        ...

    @abstractmethod
    def evict_all(self, for_gpu: bool = False, for_host: bool = False):
        ...
