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

import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.server.client import KVTPClient

try:
    from flexkv.common.request import KVResponse, KVResponseStatus
except Exception:
    from flexkv.kvtask import KVResponseStatus  # type: ignore

    KVResponse = Any  # type: ignore

from .host_kvstorage_manager import (
    HostKVStorageManagerBase,
    HostKVTaskHandle,
    HostKVTaskStatus,
    HostKVWaitResult,
)
from .kvcache_metadata import KVCacheMetadata
from .kvcache_utils import KVIndexMeta, KVLookupResult


@dataclass
class FlexKVIndexMeta(KVIndexMeta):
    batch_size: int

    token_ids: List[torch.Tensor]
    token_mask: List[Optional[Any]]
    namespaces: List[List[str]]
    slot_mappings: Optional[List[torch.Tensor]] = None


@dataclass
class _FlexKVOnloadHandle:
    task_ids: List[int]
    uids: torch.Tensor
    slot_mappings: List[torch.Tensor]


@dataclass
class _FlexKVOffloadHandle:
    task_ids: List[int]
    uids: torch.Tensor
    seqlens: torch.Tensor
    responses: Optional[Dict[int, Any]] = None

    def __post_init__(self):
        if self.responses is None:
            self.responses = dict()


@dataclass
class FlexKVCacheLayout(KVCacheLayout):
    """Runtime KV layout: [layer, block, kv, token, head, dim]."""

    def __post_init__(self) -> None:
        self._kv_shape = torch.Size(
            [
                self.num_layer,
                self.num_block,
                self._kv_dim,
                self.tokens_per_block,
                self.num_head,
                self.head_size,
            ]
        )

    def get_layer_stride(self) -> int:
        return self.kv_shape[1:].numel()

    def get_block_stride(self) -> int:
        return self.kv_shape[2:].numel()

    def get_kv_stride(self) -> int:
        return self.kv_shape[3:].numel()


class FlexKVStorageManager(HostKVStorageManagerBase):
    def __init__(
        self,
        mode: str = "direct",
        server_addr: str = "",
        server_port: int = 0,
        num_layers: int = 0,
        num_heads: int = 0,
        head_dim: int = 0,
        page_size: int = 0,
        num_cpu_blocks: int = 4096,
        num_local_blocks: int = 4096,
        num_tmp_cpu_blocks: int = 256,
        dtype: torch.dtype = torch.bfloat16,
        enable_mps: bool = False,
        as_batch: bool = True,
        hostkv_wait_timeout_ms: int = 0,
        host_kvstorage_fail_policy: str = "fail_open",
    ) -> None:
        self.mode = mode
        self.server_addr = server_addr
        self.server_port = int(server_port)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.num_cpu_blocks = int(num_cpu_blocks)
        self.num_local_blocks = int(num_local_blocks)
        self.num_tmp_cpu_blocks = int(num_tmp_cpu_blocks)
        self.dtype = dtype
        self.hostkv_wait_timeout_ms = int(hostkv_wait_timeout_ms)
        self.host_kvstorage_fail_policy = host_kvstorage_fail_policy
        self.enable_mps = bool(enable_mps)
        self.as_batch = bool(as_batch)
        self.backend_name = "flexkv"

        self._gpu_cache_table_list: Optional[List[torch.Tensor]] = None
        self._gpu_register_port: str = os.environ.get(
            "FLEXKV_GPU_REGISTER_PORT",
            "ipc:///tmp/flexkv_server_gpu_register",
        )
        self._registered: bool = False
        self._adapter = FlexKVClientAdapter(mode, server_addr, server_port)
        self._client = None
        self._ready = False

    def register_gpu_cache_tables(self, cache_table_list: List[torch.Tensor]) -> None:
        assert (
            len(cache_table_list) == self.num_layers
        ), f"cache_table_list length {len(cache_table_list)} does not match num_layers {self.num_layers}"

        # Register runtime layout directly: [block, kv, block_size, head, head_dim].
        self._gpu_cache_table_list = list(cache_table_list)

        # Initialize FlexKV client only after GPU cache table is available.
        self._init_client()

        first_table = self._gpu_cache_table_list[0]
        device_id = int(
            first_table.device.index if first_table.device.index is not None else 0
        )
        gpu_layout = FlexKVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=len(self._gpu_cache_table_list),
            num_block=int(first_table.shape[0]),
            tokens_per_block=int(first_table.shape[2]),
            num_head=int(first_table.shape[3]),
            head_size=int(first_table.shape[4]),
            is_mla=False,
        )
        tp_client = KVTPClient(
            gpu_register_port=self._gpu_register_port,
            dp_client_id=0,
            device_id=device_id,
        )
        tp_client.register_to_server(
            kv_caches=self._gpu_cache_table_list, kv_layout=gpu_layout
        )
        self._registered = True

        # Client becomes operational only after transfer manager is ready.
        if hasattr(self._client, "is_ready"):
            deadline = time.time() + 45.0
            while not self._client.is_ready() and time.time() <= deadline:
                time.sleep(0.05)
        self._ready = True

    def _init_client(self) -> None:
        if self._client is not None:
            return
        try:
            from flexkv.common.config import CacheConfig, ModelConfig
            from flexkv.kvmanager import KVManager
        except Exception as e:
            raise RuntimeError(f"FlexKV SDK import failed: {e}") from e
        if "FLEXKV_ENABLE_MPS" not in os.environ:
            os.environ["FLEXKV_ENABLE_MPS"] = "1" if self.enable_mps else "0"
        model_cfg = ModelConfig(
            num_layers=self.num_layers,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            tp_size=1,
            dp_size=1,
            dtype=self.dtype,
        )
        cache_cfg_kwargs: Dict[str, Any] = {"tokens_per_block": self.page_size}
        # Configure CPU cache sizes if specified.
        if self.num_cpu_blocks > 0:
            cache_cfg_kwargs["num_cpu_blocks"] = self.num_cpu_blocks
        if self.num_local_blocks > 0:
            cache_cfg_kwargs["num_local_blocks"] = self.num_local_blocks
        if self.num_tmp_cpu_blocks > 0:
            cache_cfg_kwargs["num_tmp_cpu_blocks"] = self.num_tmp_cpu_blocks
        cache_cfg = CacheConfig(**cache_cfg_kwargs)
        self._client = KVManager(
            model_config=model_cfg,
            cache_config=cache_cfg,
            dp_client_id=0,
        )
        self._client.start()

    def build_index_meta(
        self,
        user_ids: torch.Tensor,  # CPU Tensor
        history_sequence_lengths: torch.Tensor,  # CPU Tensor
    ) -> FlexKVIndexMeta:
        user_ids_t = (
            user_ids if user_ids.dtype == torch.int64 else user_ids.to(torch.int64)
        )
        seq_lengths_t = history_sequence_lengths.to(dtype=torch.int32)
        bsz = user_ids_t.size(0)

        token_ids = [
            torch.arange(seq_lengths_t[i], dtype=torch.int64) for i in range(bsz)
        ]
        token_mask = [
            None for _ in range(bsz)
        ]  # No mask generation. For partial onboarding, we clip token_ids to mark the onboard length
        namespaces = [[f"uid:{int(uid)}"] for uid in user_ids_t.tolist()]
        return FlexKVIndexMeta(
            user_ids=user_ids_t,
            seq_lengths=seq_lengths_t,
            batch_size=bsz,
            token_ids=token_ids,
            token_mask=token_mask,
            namespaces=namespaces,
        )

    def lookup_kvcache(self, index_meta: KVIndexMeta) -> KVLookupResult:
        if getattr(index_meta, "namespaces", None) is None:
            index_meta.namespaces = [
                [f"uid:{int(uid)}"]
                for uid in index_meta.user_ids.detach().cpu().tolist()
            ]

        requests = self._adapter.to_get_match_requests(index_meta)
        task_ids: List[int] = []
        matched_lengths: List[int] = []
        for req in requests:
            # If no tokens, skip match and return empty hit mask.
            if req["token_ids"].size == 0:
                task_ids.append(-1)
                matched_lengths.append(0)
                continue

            task_id, matched_mask = self._client.get_match(
                token_ids=req["token_ids"],
                token_mask=req["token_mask"],
                namespace=req["namespace"],
            )

            matched_mask = np.asarray(matched_mask, dtype=np.bool_)
            task_ids.append(int(task_id))
            matched_lengths.append(int(matched_mask.sum()))

        matched_t = torch.tensor(matched_lengths, dtype=torch.int32)

        return KVLookupResult(
            user_ids=index_meta.user_ids,
            host_cached_start_indices=torch.zeros_like(matched_t),
            host_cached_lengths=matched_t,
            extra={
                "backend": "flexkv",
                "task_ids": task_ids,
            },
        )

    def _build_slot_mappings(
        self, kvcache_metadata: KVCacheMetadata
    ) -> List[torch.Tensor]:
        kv_indices = kvcache_metadata.kv_indices
        kv_indptr = kvcache_metadata.kv_indptr
        kv_indices_cpu = (
            kv_indices.detach().contiguous().to(device="cpu", dtype=torch.int64)
        )
        kv_indptr_cpu = kv_indptr.detach().contiguous().to(
            device="cpu", dtype=torch.int64
        )
        indptr = kv_indptr_cpu.tolist()
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        mappings: List[torch.Tensor] = []
        for i in range(len(indptr) - 1):
            start, end = indptr[i], indptr[i + 1]
            if start == end:
                mappings.append(torch.empty((0,), dtype=torch.int64))
            else:
                page_ids_i64 = kv_indices_cpu[start:end]
                mappings.append(
                    (page_ids_i64.unsqueeze(1) * self.page_size + offsets).reshape(-1)
                )
        return mappings

    def _slice_put_launch_slot_mapping(
        self,
        slot_mapping: torch.Tensor,
        return_mask: np.ndarray,
        aligned_token_count: int,
    ) -> Optional[torch.Tensor]:
        # Required for put_match + launch (BATCH_PUT): put_match defers slot fill until
        # launch, and return_mask marks tokens that still need host write.
        mask = np.asarray(return_mask, dtype=np.bool_)
        if aligned_token_count == 0 or mask.size == 0:
            return None
        # Normalize mask length to the aligned token_ids sent to put_match.
        if mask.size < aligned_token_count:
            mask = np.concatenate(
                [mask, np.zeros(aligned_token_count - mask.size, dtype=np.bool_)]
            )
        elif mask.size > aligned_token_count:
            mask = mask[:aligned_token_count]

        # True entries are tokens FlexKV will transfer on launch.
        num_unmatched = int(mask.sum())
        if num_unmatched == 0:
            return None

        # Slice slot_mapping to the unmatched block range (vLLM-style block offset).
        num_matched = aligned_token_count - num_unmatched
        num_matched_blocks = num_matched // self.page_size
        num_unmatched_blocks = num_unmatched // self.page_size
        if num_unmatched_blocks == 0:
            return None

        start = num_matched_blocks * self.page_size
        end = start + num_unmatched_blocks * self.page_size
        if end > slot_mapping.numel():
            return None
        return slot_mapping[start:end].contiguous()

    def onboard_kvcache_launch(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata,
    ) -> HostKVTaskHandle:
        index_meta.slot_mappings = self._build_slot_mappings(kvcache_metadata)

        # Step 1. Filter out uids not to onboard
        onboard_uids = list()
        onboard_task_ids = list()
        onboard_start_indices = list()
        onboard_lengths = list()
        onboard_slot_mappings = list()
        for i in range(index_meta.batch_size):
            if lookup_result.cached_lengths[i].item() == 0:
                continue
            if lookup_result.host_cached_lengths[i].item() == 0:
                continue
            # assert lookup_result.host_cached_start_indices[i].item() == 0

            # Case 1: GPU cache is shorter
            if (
                lookup_result.host_cached_lengths[i].item()
                > lookup_result.gpu_cached_start_indices[i]
                + lookup_result.gpu_cached_lengths[i].item()
            ):
                slot_mapping = index_meta.slot_mappings[i].contiguous()[
                    : int(index_meta.seq_lengths[i].item())
                ]
                onboard_uids.append(index_meta.user_ids[i].item())
                onboard_task_ids.append(lookup_result.extra["task_ids"][i])
                onboard_start_indices.append(0)
                onboard_lengths.append(index_meta.seq_lengths[i].item())
                onboard_slot_mappings.append(slot_mapping)
                continue

            # Case 2: GPU cache has evicted the offloaded tokens
            if lookup_result.gpu_cached_start_indices[i].item() > 0:
                # assert lookup_result.gpu_cached_lengths[i].item() > 0
                slot_mapping = index_meta.slot_mappings[i].contiguous()[
                    : int(index_meta.seq_lengths[i].item())
                ]
                onboard_uids.append(index_meta.user_ids[i].item())
                onboard_task_ids.append(lookup_result.extra["task_ids"][i])
                onboard_start_indices.append(0)
                onboard_lengths.append(index_meta.seq_lengths[i].item())
                onboard_slot_mappings.append(slot_mapping)
                continue

            # TODO(junyiq): Add optimization to onboard partial. For now on all cases, we onboard the full sequence.

        if len(onboard_task_ids) == 0:
            return HostKVTaskHandle(
                backend="flexkv",
                handle=None,
                status=HostKVTaskStatus.SKIPPED,
            )

        onload_handle = _FlexKVOnloadHandle(
            task_ids=onboard_task_ids,
            uids=torch.tensor(onboard_uids, dtype=torch.int64),
            slot_mappings=onboard_slot_mappings,
        )
        onload_task_handle = HostKVTaskHandle(
            backend="flexkv",
            user_ids=onload_handle.uids,
            handle=onload_handle,
            status=HostKVTaskStatus.LAUNCHED,
            metadata={
                "onboard_start_indices": torch.tensor(
                    onboard_start_indices, dtype=torch.int32
                ),
                "onboard_lengths": torch.tensor(onboard_lengths, dtype=torch.int32),
            },
        )

        use_batch = self.as_batch and len(onboard_task_ids) > 1
        self._client.launch(
            onload_handle.task_ids,
            onload_handle.slot_mappings,
            as_batch=use_batch,
        )
        return onload_task_handle

    def onboard_kvcache_wait(self, task_handle: HostKVTaskHandle) -> HostKVWaitResult:
        onboard_results: Dict[int, "KVResponse"] = self._client.wait(
            task_handle.handle.task_ids,
        )

        failed_flag = list()
        failed_user_ids = list()
        ready = True
        # Onboard for FlexKV may launch only a subset of batch users.
        # task_ids/uids lengths must be used here instead of full batch user_ids.
        for idx in range(len(task_handle.handle.task_ids)):
            task_id = task_handle.handle.task_ids[idx]
            res = onboard_results[task_id]
            if res.status == KVResponseStatus.SUCCESS:
                failed_flag.append(0)
            elif res.status == KVResponseStatus.UNREADY:
                # Flex KV wait should not return UNREADY
                ready = False
                continue
            else:
                failed_flag.append(1)
                failed_user_ids.append(task_handle.handle.uids[idx].item())

        if len(failed_user_ids) == 0:
            return HostKVWaitResult(
                status=HostKVTaskStatus.READY if ready else HostKVTaskStatus.LAUNCHED,
                ready=ready,
            )
        else:
            return HostKVWaitResult(
                status=HostKVTaskStatus.FAILED,
                ready=False,
                failed_mask=failed_flag,
                failed_user_ids=failed_user_ids,
            )

    def offload_kvcache_launch(
        self,
        offload_user_ids: torch.Tensor,
        offload_start_indices: torch.Tensor,
        offload_page_indices_list: List[torch.Tensor],
        index_meta: Optional[KVIndexMeta] = None,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ) -> Optional[HostKVTaskHandle]:
        assert index_meta is not None
        assert torch.equal(offload_user_ids, index_meta.user_ids), (
            "offload_user_ids must match index_meta.user_ids, "
            f"got offload_user_ids={offload_user_ids.tolist()}, "
            f"index_meta.user_ids={index_meta.user_ids.tolist()}"
        )

        slot_mappings = index_meta.slot_mappings
        slot_mappings = (
            slot_mappings
            if slot_mappings is not None
            else self._build_slot_mappings(kvcache_metadata)
        )

        use_batch = self.as_batch and offload_user_ids.size(0) > 1
        task_ids: List[int] = []
        batch_slot_mappings: List[torch.Tensor] = []
        for idx in range(offload_user_ids.size(0)):
            token_ids = index_meta.token_ids[idx]
            token_mask = index_meta.token_mask[idx]

            if token_mask is None:
                valid_len = int(token_ids.numel())
            else:
                valid_len = int(token_mask.sum())

            slot_mapping = slot_mappings[idx].contiguous()[:valid_len]

            if use_batch:
                if valid_len < self.page_size:
                    continue
                put_ids = token_ids[:valid_len]
                if put_ids.dtype != torch.int64:
                    put_ids = put_ids.to(torch.int64)
                put_token_ids = put_ids.contiguous().cpu().numpy()
                # Two-phase PUT: match host radix tree, defer transfer until launch.
                # Tail-block truncation is done inside FlexKV cache_engine.put.
                task_id, return_mask = self._client.put_match(
                    token_ids=put_token_ids,
                    token_mask=None,
                    namespace=index_meta.namespaces[idx],
                )
                # slot_mapping is per-token and block-aligned; slice uses the same
                # aligned prefix length FlexKV uses (not valid_len with tail tokens).
                aligned_len = (valid_len // self.page_size) * self.page_size
                launch_slot = self._slice_put_launch_slot_mapping(
                    slot_mapping, return_mask, aligned_len
                )
                if launch_slot is None:
                    self._client.cancel(int(task_id))
                    continue
                task_ids.append(int(task_id))
                batch_slot_mappings.append(launch_slot)
            else:
                task_id = self._client.put_async(
                    token_ids=index_meta.token_ids[idx],
                    token_mask=index_meta.token_mask[idx],
                    slot_mapping=slot_mapping,
                    namespace=index_meta.namespaces[idx],
                )
                task_ids.append(int(task_id))

        if use_batch and task_ids:
            batch_launch = len(task_ids) > 1
            self._client.launch(
                task_ids, batch_slot_mappings, as_batch=batch_launch
            )
        return HostKVTaskHandle(
            backend="flexkv",
            user_ids=index_meta.user_ids,
            handle=_FlexKVOffloadHandle(
                task_ids=task_ids,
                uids=index_meta.user_ids,
                seqlens=index_meta.seq_lengths,
            ),
            status=HostKVTaskStatus.LAUNCHED,
        )

    # TODO(junyiq): Applies timeout when hostkv_wait_timeout_ms > 0.
    def offload_kvcache_wait(self, task_handle: HostKVTaskHandle) -> HostKVWaitResult:
        task_ids = task_handle.handle.task_ids
        user_ids = task_handle.handle.uids

        remain_task_ids = [t for t in task_ids if t not in task_handle.handle.responses]
        if not remain_task_ids:
            return HostKVWaitResult(status=HostKVTaskStatus.READY, ready=True)

        responses = self._client.try_wait(remain_task_ids)

        # try_wait is non-blocking: empty or partial result means tasks are still pending.
        has_unready = False
        # has_timeout = False
        has_cancelled = False
        has_failed = False
        msgs: List[str] = []

        missing_task_ids = [t for t in remain_task_ids if t not in responses]
        if missing_task_ids:
            has_unready = True
            msgs.extend(
                f"{task_id}:{KVResponseStatus.UNREADY}" for task_id in missing_task_ids
            )

        for task_id, resp in responses.items():
            task_handle.handle.responses[task_id] = resp
            msgs.append(f"{task_id}:{resp.status}")
            if resp.status == KVResponseStatus.SUCCESS:
                continue
            elif resp.status == KVResponseStatus.UNREADY:
                has_unready = True
            # if resp.status == KVResponseStatus.TIMEOUT:
            #     has_timeout = True
            elif resp.status == KVResponseStatus.CANCELLED:
                has_cancelled = True
            else:
                has_failed = True
        # if has_timeout:
        #     return HostKVWaitResult(
        #         status=HostKVTaskStatus.TIMEOUT,
        #         ready=False,
        #         message=";".join(msgs),
        #         failed_user_ids=user_ids,
        #     )
        if has_failed or has_cancelled:
            return HostKVWaitResult(
                status=HostKVTaskStatus.CANCELLED
                if has_cancelled and not has_failed
                else HostKVTaskStatus.FAILED,
                ready=False,
                message=";".join(msgs),
                failed_user_ids=user_ids,
            )
        if has_unready:
            return HostKVWaitResult(
                status=HostKVTaskStatus.LAUNCHED,
                ready=False,
                message=";".join(msgs),
                failed_user_ids=user_ids,
            )
        return HostKVWaitResult(status=HostKVTaskStatus.READY, ready=True)

    def finish_task(self, task_handle: HostKVTaskHandle) -> List[int]:
        # FlexKV wait success equals finish.
        # offload_kvcache_wait() may already try_wait() to SUCCESS and release tasks in
        # FlexKV; waiting again on those ids logs "not submitted" (NOTFOUND). Only wait
        # on tasks that are not yet SUCCESS in handle.responses.
        # TODO(junyiq): Make sure gpu kvcache locks the pages when offloading for flexkv backend.
        responses = task_handle.handle.responses
        pending_task_ids = [
            task_id
            for task_id in task_handle.handle.task_ids
            if responses.get(task_id) is None
            or responses[task_id].status != KVResponseStatus.SUCCESS
        ]
        if pending_task_ids:
            self._client.wait(pending_task_ids, completely=True)
        return [1 for _ in range(task_handle.handle.uids.size(0))]

    def cancel_task(self, task_handle: HostKVTaskHandle) -> bool:
        if task_handle is None or task_handle.handle is None:
            return False
        self._client.cancel(task_handle.handle.task_ids)
        task_handle.status = HostKVTaskStatus.CANCELLED
        return True

    def evict(self, user_ids: torch.Tensor) -> None:
        warnings.warn(
            "FlexKV backend does not expose an explicit evict API; evict() is a no-op.",
            RuntimeWarning,
            stacklevel=2,
        )

    def evict_all(self) -> None:
        warnings.warn(
            "FlexKV backend does not expose an explicit evict_all API; evict_all() is a no-op.",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def get_offload_handle_metadata(
        task_handle,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            task_handle.handle.uids,
            torch.zeros_like(task_handle.handle.seqlens),
            task_handle.handle.seqlens,
        )


class FlexKVClientAdapter:
    def __init__(self, mode: str, server_addr: str = "", server_port: int = 0):
        self.mode = mode
        self.server_addr = server_addr
        self.server_port = server_port

    def to_get_match_requests(self, index_meta: KVIndexMeta) -> List[Dict[str, Any]]:
        reqs: List[Dict[str, Any]] = []
        for i in range(index_meta.batch_size):
            seq_ids = index_meta.token_ids[i]
            seq_mask = index_meta.token_mask[i]
            assert seq_mask is None

            if len(seq_ids) == 0:
                reqs.append(
                    {
                        "user_id": int(index_meta.user_ids[i]),
                        "namespace": index_meta.namespaces[i],
                        "token_ids": np.zeros((0,), dtype=np.int64),
                        "token_mask": np.zeros((0,), dtype=np.bool_),
                    }
                )
                continue

            reqs.append(
                {
                    "user_id": int(index_meta.user_ids[i]),
                    "namespace": index_meta.namespaces[i],
                    "token_ids": seq_ids,
                    "token_mask": seq_mask,
                }
            )
        return reqs
