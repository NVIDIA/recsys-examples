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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from .kvcache_metadata import KVCacheMetadata
from .kvcache_utils import KVIndexMeta, KVLookupResult


class HostKVTaskStatus(Enum):
    UNINITIALIZED = "uninitialized"
    SKIPPED = "skipped"
    LAUNCHED = "launched"
    READY = "ready"
    EVENT_READY = "event_ready"
    TIMEOUT = "timeout"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HostKVStorageErrorCode(str, Enum):
    SDK_IMPORT_FAILED = "sdk_import_failed"
    SDK_INIT_FAILED = "sdk_init_failed"
    LOOKUP_FAILED = "lookup_failed"
    LOOKUP_MISSING_TOKENS = "lookup_missing_tokens"
    ONBOARD_TASK_NOT_FOUND = "onboard_task_not_found"
    ONBOARD_WAIT_FAILED = "onboard_wait_failed"
    ONBOARD_TIMEOUT = "onboard_timeout"
    OFFLOAD_TASK_NOT_FOUND = "offload_task_not_found"
    OFFLOAD_WAIT_FAILED = "offload_wait_failed"
    OFFLOAD_TIMEOUT = "offload_timeout"
    CANCEL_FAILED = "cancel_failed"


@dataclass
class HostKVTaskHandle:
    backend: str
    user_ids: Optional[torch.Tensor] = None
    handle: Optional[Any] = None
    status: HostKVTaskStatus = HostKVTaskStatus.UNINITIALIZED
    metadata: Optional[Dict[str, Any]] = None
    time_launched: Optional[float] = None
    is_layerwise: bool = False
    onboard_wait_by_layer: Optional[Any] = None

    def __post_init__(self):
        if self.status not in {
            HostKVTaskStatus.UNINITIALIZED,
            HostKVTaskStatus.SKIPPED,
        }:
            assert (
                self.user_ids is not None
            ), "user_ids must be provided for initialized tasks"
            assert (
                self.handle is not None
            ), "underlying handle must be provided for initialized tasks"

    def stream_wait_layer(self, layer_idx: int) -> None:
        if self.is_layerwise:
            self.handle.wait_layer(layer_idx)

    def wait_layer(self, layer_idx: int):
        if self.onboard_wait_by_layer is not None:
            return self.onboard_wait_by_layer(self, layer_idx)
        self.stream_wait_layer(layer_idx)


@dataclass
class HostKVWaitResult:
    status: HostKVTaskStatus
    ready: bool
    message: str = ""
    failed_mask: Optional[torch.Tensor] = None
    failed_user_ids: Optional[List[int]] = None


class HostKVStorageManagerBase(ABC):
    @abstractmethod
    def register_gpu_cache_tables(self, cache_table_list: List[torch.Tensor]) -> None:
        ...

    @abstractmethod
    def build_index_meta(
        self, user_ids: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> KVIndexMeta:
        ...

    @abstractmethod
    def lookup_kvcache(self, index_meta: KVIndexMeta) -> KVLookupResult:
        ...

    @abstractmethod
    def onboard_kvcache_launch(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata,
    ) -> HostKVTaskHandle:
        ...

    @abstractmethod
    def onboard_kvcache_wait(self, task_handle: HostKVTaskHandle) -> HostKVWaitResult:
        ...

    def onboard_kvcache_wait_by_layer(
        self, task_handle: HostKVTaskHandle, layer_idx: int
    ) -> HostKVWaitResult:
        return HostKVWaitResult(
            status=HostKVTaskStatus.SKIPPED,
            ready=False,
        )

    @abstractmethod
    def offload_kvcache_launch(
        self,
        offload_user_ids: torch.Tensor,
        offload_start_indices: torch.Tensor,
        offload_page_indices_list: List[torch.Tensor],
        index_meta: Optional[KVIndexMeta] = None,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ) -> Optional[HostKVTaskHandle]:
        ...

    @abstractmethod
    def offload_kvcache_wait(self, task_handle: HostKVTaskHandle) -> HostKVWaitResult:
        ...

    @abstractmethod
    def finish_task(self, task_handle: HostKVTaskHandle) -> List[int]:
        ...

    @abstractmethod
    def cancel_task(self, task_handle: HostKVTaskHandle) -> bool:
        ...

    @abstractmethod
    def evict(self, user_ids: torch.Tensor) -> None:
        ...

    @abstractmethod
    def evict_all(self) -> None:
        ...

    @staticmethod
    def get_offload_handle_metadata(
        task_handle,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # return (offload_uids, offload_start_indices, offload_lengths)
        ...
