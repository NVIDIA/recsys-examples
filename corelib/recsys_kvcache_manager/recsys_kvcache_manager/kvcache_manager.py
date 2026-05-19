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

    @staticmethod
    def _build_host_kvstorage_manager_from_config(
        kvcache_config,
    ) -> HostKVStorageManagerBase:
        if kvcache_config.host_kvstorage_backend == "native":
            from .native_host_kvcache_manager import NativeHostKVCacheManager

            return NativeHostKVCacheManager(
                kvcache_config.num_layers,
                kvcache_config.num_heads,
                kvcache_config.head_dim,
                kvcache_config.page_size,
                kvcache_config.offload_chunksize,
                kvcache_config.host_capacity_per_layer,
                kvcache_config.max_batch_size,
                math.ceil(kvcache_config.max_seq_len / kvcache_config.page_size)
                * kvcache_config.page_size,
                kvcache_config.onload_timeout_ms,
                kvcache_config.offload_timeout_ms,
                kvcache_config.dtype,
                kvcache_config.device,
            )
        elif kvcache_config.host_kvstorage_backend == "flexkv":
            from .flex_kvcache_manager import FlexKVStorageManager

            extra = getattr(kvcache_config, "extra_configs", {}) or {}
            flexkv_mode = extra.get("flexkv_mode", "direct")
            flexkv_server_addr = extra.get("flexkv_server_addr", "")
            flexkv_server_port = int(extra.get("flexkv_server_port", 0))
            flexkv_config_path = extra.get("flexkv_config_path", None)
            flexkv_num_cpu_blocks = int(extra.get("flexkv_num_cpu_blocks", 4096))
            flexkv_num_local_blocks = int(extra.get("flexkv_num_local_blocks", 4096))
            flexkv_num_tmp_cpu_blocks = int(extra.get("flexkv_num_tmp_cpu_blocks", 256))
            flexkv_host_kvstorage_fail_policy = str(
                extra.get("flexkv_host_kvstorage_fail_policy", "fail_close")
            )
            flexkv_enable_mps_raw = extra.get("flexkv_enable_mps", 0)
            if isinstance(flexkv_enable_mps_raw, str):
                flexkv_enable_mps = flexkv_enable_mps_raw.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            else:
                flexkv_enable_mps = bool(flexkv_enable_mps_raw)
            flexkv_as_batch_raw = extra.get("flexkv_as_batch", 1)
            if isinstance(flexkv_as_batch_raw, str):
                flexkv_as_batch = flexkv_as_batch_raw.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            else:
                flexkv_as_batch = bool(flexkv_as_batch_raw)

            return FlexKVStorageManager(
                mode=flexkv_mode,
                server_addr=flexkv_server_addr,
                server_port=flexkv_server_port,
                num_layers=kvcache_config.num_layers,
                num_heads=kvcache_config.num_heads,
                head_dim=kvcache_config.head_dim,
                page_size=kvcache_config.page_size,
                num_cpu_blocks=flexkv_num_cpu_blocks,
                num_local_blocks=flexkv_num_local_blocks,
                num_tmp_cpu_blocks=flexkv_num_tmp_cpu_blocks,
                dtype=kvcache_config.dtype,
                enable_mps=flexkv_enable_mps,
                as_batch=flexkv_as_batch,
                host_kvstorage_fail_policy=flexkv_host_kvstorage_fail_policy,
                hostkv_wait_timeout_ms=int(kvcache_config.offload_timeout_ms),
                config_path=flexkv_config_path,
            )
        else:
            raise NotImplementedError(
                f"Unknown host kvcache backend {kvcache_config.host_kvstorage_backend}"
            )

    @classmethod
    def from_config(cls, kvcache_config):
        return cls(DefaultKVCacheBackend.from_config(kvcache_config))
