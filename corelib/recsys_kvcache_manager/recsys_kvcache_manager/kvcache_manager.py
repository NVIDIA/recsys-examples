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

import math
from typing import List, Optional, Tuple

import torch

from .gpu_kvcache_manager import GPUKVCacheManager
from .host_kvstorage_manager import (
    HostKVStorageManagerBase,
    HostKVTaskHandle,
    HostKVTaskStatus,
    HostKVWaitResult,
)
from .kvcache_metadata import KVCacheMetadata
from .kvcache_utils import KVCacheOffloadMode, KVIndexMeta, KVLookupResult


class KVCacheManager:
    def __init__(
        self,
        gpu_kvcache_manager: GPUKVCacheManager,
        host_kvstorage_manager: HostKVStorageManagerBase,
        offload_mode: str = "lazy",
        host_kvstorage_fail_policy: str = "fail_open",
    ):
        # self.num_layers = num_layers
        # self.num_heads = num_kv_heads
        # self.head_dim = kv_headdim
        # self.page_size = num_tokens_per_page
        # self.chunk_size = num_tokens_per_chunk
        # self.num_primary_cache_pages = num_primary_cache_pages
        # self.num_buffer_pages = num_buffer_pages
        # self.max_batch_size = max_batch_size

        self.gpu_kvcache_mgr = gpu_kvcache_manager
        self.dummy_empty_tensor = torch.tensor([], dtype=torch.int32)

        self.host_kvstorage_manager = host_kvstorage_manager
        self.host_kvstorage_manager.register_gpu_cache_tables(
            self.gpu_kvcache_mgr.get_cache_tables()
        )

        self.offload_mode = (
            KVCacheOffloadMode(offload_mode)
            if offload_mode in {m.value for m in KVCacheOffloadMode}
            else KVCacheOffloadMode.LAZY
        )
        self.host_kvstorage_fail_policy = host_kvstorage_fail_policy
        self.ongoing_onboard_tasks: List[HostKVTaskHandle] = []
        self.ongoing_offload_tasks: List[HostKVTaskHandle] = []

    def lookup_kvcache(
        self,
        user_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> Tuple[KVIndexMeta, KVLookupResult]:
        gpu_lookup_results = self.gpu_kvcache_mgr.lookup(user_ids)
        index_meta = self.host_kvstorage_manager.build_index_meta(
            user_ids, sequence_lengths
        )
        host_lookup_results = self.host_kvstorage_manager.lookup_kvcache(index_meta)

        lookup_results = KVLookupResult.merge(gpu_lookup_results, host_lookup_results)
        return index_meta, lookup_results

    def allocate_kvcache(
        self,
        index_meta: KVIndexMeta,
        lookup_results: KVLookupResult,
        output_kvcache_metadata: Optional[KVCacheMetadata] = None,
    ) -> KVCacheMetadata:
        return self.gpu_kvcache_mgr.allocate(
            index_meta.user_ids,
            index_meta.seq_lengths,
            lookup_results,
            output_kvcache_metadata=output_kvcache_metadata,
        )

    def onboard_launch(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata,
    ) -> HostKVTaskHandle:
        task_handle = self.host_kvstorage_manager.onboard_kvcache_launch(
            index_meta,
            lookup_result,
            kvcache_metadata,
        )
        kvcache_metadata.kv_onload_handle = task_handle

        # Skip recording the ongoing onboard tasks. For now there should be only one task.
        # self.ongoing_onboard_tasks.append(task_handle)
        return task_handle

    def onboard_try_wait(
        self,
        kv_index_meta: KVIndexMeta,
        task_handle: Optional[HostKVTaskHandle],
    ) -> Optional[HostKVWaitResult]:
        if self.host_kvstorage_manager.backend_name == "native":
            return self.host_kvstorage_manager.onboard_kvcache_wait(task_handle)
        elif self.host_kvstorage_manager.backend_name == "flexkv":
            print(
                "[WARNING] onboard_try_wait is not implemented for flexkv backend currently. Calling onboard_wait instead."
            )
            return self.onboard_wait(kv_index_meta, task_handle)
        else:
            raise NotImplementedError(
                f"Unknown host kvcache backend {self.host_kvstorage_manager.backend_name}"
            )

    def onboard_wait(
        self,
        kv_index_meta: KVIndexMeta,
        task_handle: Optional[HostKVTaskHandle],
    ) -> Optional[HostKVWaitResult]:
        if (
            task_handle is None
            or task_handle.handle is None
            or task_handle.status == HostKVTaskStatus.SKIPPED
        ):
            return HostKVWaitResult(
                status=HostKVTaskStatus.UNINITIALIZED,
                ready=False,
            )
        wait_result = self.host_kvstorage_manager.onboard_kvcache_wait(task_handle)
        # Note: wait_result.status == SKIPPED means waiting/sync here is not supported with backend in use.

        if wait_result.status in (
            HostKVTaskStatus.FAILED,
            HostKVTaskStatus.TIMEOUT,
            HostKVTaskStatus.CANCELLED,
        ):
            # Revoke affected pages on onboard failure.
            self.gpu_kvcache_mgr.revoke_onboard_pages(
                task_handle.user_ids,
                task_handle.metadata["onboard_start_indices"],
                task_handle.metadata["onboard_lengths"],
            )
            if task_handle.backend == "flexkv":
                if self.host_kvstorage_fail_policy == "fail_close":
                    raise RuntimeError(
                        f"Onboarding failed for {wait_result.failed_user_ids}: status={wait_result.status.value}, msg={wait_result.message}"
                    )
                else:
                    print(
                        f"[WARNING] Onboarding failed for {wait_result.failed_user_ids}, but continue with `fail_open` policy."
                    )
        return wait_result

    def offload_launch(
        self,
        index_meta: KVIndexMeta,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ):
        if self.host_kvstorage_manager.backend_name == "flexkv":
            assert (
                kvcache_metadata is not None
            ), "flexkv offload requires kvcache_metadata"

        # 1. Get the maximum batch for offloading from GPU
        # 2. Lookup host again in case of multi-GPU instances
        if self.host_kvstorage_manager.backend_name == "native":
            uids_to_offload = self.gpu_kvcache_mgr.check_for_offload(
                index_meta.user_ids
            )
            _index_meta = self.host_kvstorage_manager.build_index_meta(
                uids_to_offload,
                torch.empty(
                    0, dtype=torch.int32
                ),  # dummy seq lengths since they are not used for lookup
            )
            offloaded_lengths = self.host_kvstorage_manager.lookup_kvcache(
                _index_meta
            ).host_cached_lengths
        else:
            uids_to_offload = index_meta.user_ids
            offloaded_lengths = self.dummy_empty_tensor

        # 3. Acquire and lock GPU cache pages
        (
            offload_user_ids,
            offload_start_indices,
            offload_page_indices_list,
        ) = self.gpu_kvcache_mgr.acquire_offload_pages(
            uids_to_offload,
            offloaded_lengths,
            True if self.host_kvstorage_manager.backend_name == "flexkv" else False,
        )
        # returned with cache pages locked (per user).
        if offload_user_ids.size(0) == 0:
            return None

        # 4. Launch the offloading thru Host
        task_handle = self.host_kvstorage_manager.offload_kvcache_launch(
            offload_user_ids,
            offload_start_indices,
            offload_page_indices_list,
            index_meta=index_meta,
            kvcache_metadata=kvcache_metadata,
        )
        if (
            task_handle is None
            or task_handle.handle is None
            or task_handle.status == HostKVTaskStatus.SKIPPED
        ):
            # offload is rejected on the host side (e.g., due to overload), release the locks immediately.
            self.gpu_kvcache_mgr.release_offload_pages(
                offload_user_ids,
                offload_start_indices,
                self.dummy_empty_tensor,
                offloaded=[0 for _ in range(offload_user_ids.size(0))],
            )
            return None

        # Launch failure is not resolved here
        self.ongoing_offload_tasks.append(task_handle)
        return task_handle

    def offload_try_wait(self) -> None:
        remain_tasks = list()
        for task_handle in self.ongoing_offload_tasks:
            wait_result = self.host_kvstorage_manager.offload_kvcache_wait(task_handle)
            if wait_result.status == HostKVTaskStatus.LAUNCHED:
                remain_tasks.append(task_handle)
                continue
            if wait_result.status == HostKVTaskStatus.READY:
                offload_success = self.host_kvstorage_manager.finish_task(task_handle)
            elif wait_result.status == HostKVTaskStatus.SKIPPED:
                # No need to release GPU pages since offload is skipped
                print(f"Offload skipped for {task_handle.user_ids.tolist()}")
                continue
            elif wait_result.status in (
                HostKVTaskStatus.FAILED,
                HostKVTaskStatus.TIMEOUT,
                HostKVTaskStatus.CANCELLED,
            ):
                should_raise = self.host_kvstorage_fail_policy == "fail_close"
                if should_raise:
                    raise RuntimeError(
                        f"Offloading failed for {wait_result.failed_user_ids}, fail_policy={self.host_kvstorage_fail_policy}"
                    )

                offload_success = [0 for _ in range(task_handle.user_ids.size(0))]
                self.host_kvstorage_manager.cancel_task(task_handle)
            else:
                raise RuntimeError(
                    f"Unexpected offload wait result status: {wait_result.status.value}, msg={wait_result.message}"
                )
            self.gpu_kvcache_mgr.release_offload_pages(
                *(self.host_kvstorage_manager.get_offload_handle_metadata(task_handle)),
                offloaded=offload_success,
            )
        self.ongoing_offload_tasks = remain_tasks

    def evict(
        self, user_ids: torch.Tensor, for_gpu: bool = False, for_host: bool = False
    ):
        if for_gpu:
            self.gpu_kvcache_mgr.evict(user_ids)
        if for_host:
            self.host_kvstorage_manager.evict(user_ids)

    def evict_all(self, for_gpu: bool = False, for_host: bool = False):
        if for_gpu:
            self.gpu_kvcache_mgr.evict_all()
        if for_host:
            self.host_kvstorage_manager.evict_all()

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
                host_kvstorage_fail_policy=flexkv_host_kvstorage_fail_policy,
                hostkv_wait_timeout_ms=int(kvcache_config.offload_timeout_ms),
            )
        else:
            raise NotImplementedError(
                f"Unknown host kvcache backend {kvcache_config.host_kvstorage_backend}"
            )

    @classmethod
    def from_config(cls, kvcache_config):
        assert (
            kvcache_config.offload_chunksize % kvcache_config.page_size == 0
        ), "Require offload_chunksize to be multiple of page_size"
        gpu_kvcache_mgr = GPUKVCacheManager(
            kvcache_config.num_layers,
            kvcache_config.num_heads,
            kvcache_config.head_dim,
            kvcache_config.page_size,
            kvcache_config.offload_chunksize,
            kvcache_config.num_primary_cache_pages,
            kvcache_config.num_buffer_pages,
            kvcache_config.max_batch_size,
            math.ceil(kvcache_config.max_seq_len / kvcache_config.page_size)
            * kvcache_config.page_size,
            kvcache_config.dtype,
            kvcache_config.device,
        )

        host_kvcache_mgr = cls._build_host_kvstorage_manager_from_config(kvcache_config)

        return cls(
            gpu_kvcache_mgr,
            host_kvcache_mgr,
            kvcache_config.offload_mode,
            # kvcache_config.offload_timeout_ms,
            kvcache_config.host_kvstorage_fail_policy,
        )
