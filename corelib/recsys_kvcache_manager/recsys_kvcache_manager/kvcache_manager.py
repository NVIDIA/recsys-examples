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

from typing import Optional, Tuple

import torch

from .default_kvcache_backend import DefaultKVCacheBackend
from .kvcache_backend import KVCacheBackend
from .host_kvstorage_manager import (
    HostKVTaskHandle,
    HostKVWaitResult,
)
from .kvcache_metadata import KVCacheMetadata
from .kvcache_utils import KVIndexMeta, KVLookupResult


class KVCacheManager:
    """Public user-facing KVCache manager interface.

    The current implementation delegates all methods to DefaultKVCacheBackend.
    """

    def __init__(self, backend: KVCacheBackend):
        self.backend = backend

    def lookup_kvcache(
        self,
        user_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> Tuple[KVIndexMeta, KVLookupResult]:
        return self.backend.lookup_kvcache(user_ids, sequence_lengths)

    def allocate_kvcache(
        self,
        index_meta: KVIndexMeta,
        lookup_results: KVLookupResult,
        output_kvcache_metadata: Optional[KVCacheMetadata] = None,
    ) -> KVCacheMetadata:
        return self.backend.allocate_kvcache(
            index_meta, lookup_results, output_kvcache_metadata
        )

    def onboard_launch(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata,
    ) -> HostKVTaskHandle:
        return self.backend.onboard_launch(index_meta, lookup_result, kvcache_metadata)

    def onboard_try_wait(
        self,
        kv_index_meta: KVIndexMeta,
        task_handle: Optional[HostKVTaskHandle],
    ) -> Optional[HostKVWaitResult]:
        return self.backend.onboard_try_wait(kv_index_meta, task_handle)

    def onboard_wait(
        self,
        kv_index_meta: KVIndexMeta,
        task_handle: Optional[HostKVTaskHandle],
    ) -> Optional[HostKVWaitResult]:
        return self.backend.onboard_wait(kv_index_meta, task_handle)

    def offload_launch(
        self,
        index_meta: KVIndexMeta,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ):
        return self.backend.offload_launch(index_meta, kvcache_metadata)

    def offload_try_wait(self) -> None:
        self.backend.offload_try_wait()

    def evict(
        self, user_ids: torch.Tensor, for_gpu: bool = False, for_host: bool = False
    ):
        self.backend.evict(user_ids, for_gpu=for_gpu, for_host=for_host)

    def evict_all(self, for_gpu: bool = False, for_host: bool = False):
        self.backend.evict_all(for_gpu=for_gpu, for_host=for_host)

    @classmethod
    def from_config(cls, kvcache_config):
        return cls(DefaultKVCacheBackend.from_config(kvcache_config))
