import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import paged_kvcache_ops
import torch
from configs import KVCacheMetadata
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Any, List, Dict, Optional, Tuple, Union
from uuid import uuid4
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


class KVCacheOffloadMode(Enum):
    LAZY = "lazy"
    EAGER = "eager"

class SecondaryTaskStatus(Enum):
    SKIPPED = "skipped"
    LAUNCHED = "launched"
    READY = "ready"
    TIMEOUT = "timeout"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SecondaryErrorCode(str, Enum):
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
class KVLookupResult:
    request_id: str
    batch_size: int
    user_ids: List[int]
    total_history_lengths: List[int]
    old_cached_lengths: List[int]
    new_tokens_upper_bound: int
    token_ids: Optional[torch.Tensor] = None
    token_mask: Optional[torch.Tensor] = None
    secondary_lookup: Optional[Dict[str, Any]] = None

@dataclass
class KVPrepareResult:
    old_cached_lengths: List[int]
    new_tokens: int
    offload_uids_buffer: torch.Tensor
    metadata_host_buffer: torch.Tensor
    metadata_gpu_buffer: torch.Tensor
    kvcache_metadata_fut: Any
    onload_fut: Any

    def to_legacy_list(self):
        return [
            self.old_cached_lengths,
            self.new_tokens,
            self.offload_uids_buffer,
            self.metadata_host_buffer,
            self.metadata_gpu_buffer,
            self.kvcache_metadata_fut,
            self.onload_fut,
        ]

@dataclass
class KVIndexMeta:
    request_id: str
    batch_size: int
    user_ids: List[int]
    namespaces: List[str]
    total_history_lengths: List[int]
    old_cached_lengths: List[int]
    seq_start_indices: List[int]
    seq_lengths: List[int]
    new_tokens: int
    token_ids: Optional[torch.Tensor] = None
    token_mask: Optional[torch.Tensor] = None
    restore_slot_mapping: Optional[torch.Tensor] = None
    append_slot_mapping: Optional[torch.Tensor] = None
    append_slot_indptr: Optional[torch.Tensor] = None
    secondary_hit_mask: Optional[torch.Tensor] = None
    secondary_get_task_ids: Optional[List[int]] = None
    secondary_matched_lengths: Optional[List[int]] = None

@dataclass
class SecondaryTaskHandle:
    backend: str
    handle: Optional[Any]
    status: SecondaryTaskStatus = SecondaryTaskStatus.SKIPPED
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SecondaryWaitResult:
    status: SecondaryTaskStatus
    ready: bool
    error_code: Optional[str] = None
    message: str = ""
    failed_mask: Optional[torch.Tensor] = None
    failed_user_ids: Optional[List[int]] = None

class SecondaryKVCacheManagerBase(ABC):
    @abstractmethod
    def lookup_kvcache(self, index_meta: KVIndexMeta) -> Dict[str, Any]:
        ...

    @abstractmethod
    def onboard_launch_kvcache(
        self, index_meta: KVIndexMeta, restore_slot_mapping: Optional[torch.Tensor]
    ) -> SecondaryTaskHandle:
        ...

    @abstractmethod
    def onboard_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        ...

    @abstractmethod
    def offload_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        append_slot_mapping: Optional[torch.Tensor],
        append_slot_indptr: Optional[torch.Tensor] = None,
    ) -> SecondaryTaskHandle:
        ...

    @abstractmethod
    def offload_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        ...

    @abstractmethod
    def cancel_task(self, task_handle: SecondaryTaskHandle) -> None:
        ...

    def register_gpu_cache_tensors(self, cache_table_list: List[torch.Tensor]) -> None:
        # Optional hook for backends that require GPU KV cache registration.
        return None

class NopSecondaryKVCacheManager(SecondaryKVCacheManagerBase):
    def lookup_kvcache(self, index_meta: KVIndexMeta):
        return {"backend": "nop", "hit_mask": None}

    def onboard_launch_kvcache(self, index_meta, restore_slot_mapping):
        return SecondaryTaskHandle(
            backend="nop",
            handle=None,
            status=SecondaryTaskStatus.SKIPPED,
            metadata={"reason": "nop backend"},
        )

    def onboard_wait_kvcache(self, task_handle):
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.READY,
            ready=True,
            message="nop onboard wait ready",
        )

    def offload_launch_kvcache(
        self,
        index_meta,
        append_slot_mapping,
        append_slot_indptr=None,
    ):
        return SecondaryTaskHandle(
            backend="nop",
            handle=None,
            status=SecondaryTaskStatus.SKIPPED,
            metadata={"reason": "nop backend"},
        )

    def offload_wait_kvcache(self, task_handle):
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.READY,
            ready=True,
            message="nop offload wait ready",
        )

    def cancel_task(self, task_handle):
        return None


class KVCacheManager:
    def __init__(
        self,
        num_layers,
        num_kv_heads,
        kv_headdim,
        num_tokens_per_page,
        num_primary_cache_pages,
        num_onload_buffer_pages,
        num_reserved_buffer_pages,
        num_tokens_per_chunk,
        max_num_sequences,
        max_sequence_length,
        max_batch_size,
        max_queued_offload_tokens,
        num_onload_buffer_chunks=1,
        num_offload_buffer_chunks=8,
        num_memcpy_workers=8,
        enable_nvcomp=False,
        secondary_kvcache_manager: Optional[SecondaryKVCacheManagerBase] = None,
        offload_mode: str = "lazy",
        secondary_wait_timeout_ms: int = 0,
        secondary_fail_policy: str = "fail_open",

    ):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.onload_worker = ThreadPoolExecutor(max_workers=1)

        self.num_layers = num_layers
        self.num_heads = num_kv_heads
        self.head_dim = kv_headdim
        self.page_size = num_tokens_per_page
        self.num_primary_cache_pages = num_primary_cache_pages
        self.num_onload_buffer_pages = num_onload_buffer_pages
        self.num_reserved_buffer_pages = num_reserved_buffer_pages
        self.chunk_size = num_tokens_per_chunk
        self.max_num_sequences = max_num_sequences
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.max_num_pages_per_seq = math.ceil(
            self.max_sequence_length / self.page_size
        )

        self.num_cache_pages = num_primary_cache_pages + num_onload_buffer_pages
        self.cache_table = torch.empty(
            [
                num_layers,
                self.num_cache_pages,
                2,
                self.page_size,
                self.num_heads,
                self.head_dim,
            ],
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )

        self.host_kv_mgr = paged_kvcache_ops.HostKVStorageImpl(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.page_size,
            self.chunk_size,
        )
        self.gpu_kvcache_mgr = paged_kvcache_ops.GPUKVCacheMangerImpl(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.page_size,
            self.num_primary_cache_pages,
            self.num_onload_buffer_pages,
            self.num_reserved_buffer_pages,
            self.chunk_size,
            self.max_num_sequences,
            self.max_num_sequences,
            self.cache_table,
            self.host_kv_mgr,
            max_queued_offload_tokens,
            num_onload_buffer_chunks,
            num_offload_buffer_chunks,
            num_memcpy_workers,
            enable_nvcomp,
        )

        self.static_page_ids_gpu_buffer = torch.empty(
            [
                self.max_batch_size * self.max_num_pages_per_seq,
            ],
            dtype=torch.int32,
        ).cuda()
        self.static_offload_page_ids_gpu_buffer = torch.empty(
            [
                self.max_batch_size * self.max_num_pages_per_seq,
            ],
            dtype=torch.int32,
        ).cuda()
        self.static_metadata_gpu_buffer = torch.empty(
            [
                self.max_batch_size * 5
                + 4
                + self.max_batch_size * self.max_sequence_length * 2,
            ],
            dtype=torch.int32,
        ).cuda()
        self.static_onload_handle = paged_kvcache_ops.KVOnloadHandle(self.num_layers)
        self.static_empty_offload_handle = paged_kvcache_ops.KVOffloadHandle()

        self.cache_table_list = [
            self.cache_table[idx] for idx in range(self.num_layers)
        ]
        self.secondary_kvcache_manager = secondary_kvcache_manager if secondary_kvcache_manager is not None else NopSecondaryKVCacheManager()
        # Real FlexKV backend requires explicit GPU KV cache registration before ready.
        self.secondary_kvcache_manager.register_gpu_cache_tensors(self.cache_table_list)

        self.offload_mode = (
            KVCacheOffloadMode(offload_mode)
            if offload_mode in {m.value for m in KVCacheOffloadMode}
            else KVCacheOffloadMode.LAZY
        )
        self.secondary_wait_timeout_ms = int(secondary_wait_timeout_ms)
        self.secondary_fail_policy = secondary_fail_policy
        self.ongoing_onboard_tasks: Dict[str, SecondaryTaskHandle] = {}
        self.ongoing_offload_tasks: Dict[str, SecondaryTaskHandle] = {}
        self.request_to_task_handles: Dict[str, Dict[str, Optional[SecondaryTaskHandle]]] = {}

    def _page_ids_to_slot_mapping(self, page_ids: torch.Tensor) -> np.ndarray:
        page_ids_np = page_ids.to(torch.int64).detach().cpu().numpy()
        return np.repeat(page_ids_np * self.page_size, self.page_size).astype(np.int64)

    def materialize_restore_mapping_from_metadata(
        self,
        kv_index_meta: Optional[KVIndexMeta],
        kvcache_metadata: KVCacheMetadata,
    ) -> None:
        if kv_index_meta is None:
            return
        task_ids = kv_index_meta.secondary_get_task_ids or []
        matched_lengths = kv_index_meta.secondary_matched_lengths or []
        if len(task_ids) == 0 or len(matched_lengths) == 0:
            kv_index_meta.restore_slot_mapping = None
            return

        kv_indices = kvcache_metadata.kv_indices.to(torch.int64).detach().cpu()
        kv_indptr = kvcache_metadata.kv_indptr.to(torch.int64).detach().cpu()

        launch_task_ids: List[int] = []
        launch_slot_mappings: List[np.ndarray] = []

        for i in range(kv_index_meta.batch_size):
            if i >= len(task_ids):
                continue
            task_id = int(task_ids[i])
            matched_len = int(matched_lengths[i]) if i < len(matched_lengths) else 0
            old_cached_len = int(kv_index_meta.old_cached_lengths[i])

            start_block = old_cached_len // self.page_size
            end_block = matched_len // self.page_size
            if end_block <= start_block:
                continue

            seq_page_start = int(kv_indptr[i].item())
            seq_page_end = int(kv_indptr[i + 1].item())
            seq_pages = kv_indices[seq_page_start:seq_page_end]
            restore_pages = seq_pages[start_block:end_block]
            if restore_pages.numel() == 0:
                continue

            launch_task_ids.append(task_id)
            launch_slot_mappings.append(self._page_ids_to_slot_mapping(restore_pages))

        kv_index_meta.restore_slot_mapping = {
            "task_ids": launch_task_ids,
            "slot_mappings": launch_slot_mappings,
        }

    @staticmethod
    def _build_secondary_manager_from_config(
        secondary_backend: str,
        flexkv_mode: str,
        flexkv_server_addr: str,
        flexkv_server_port: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        secondary_wait_timeout_ms: int,
        secondary_fail_policy: str,
    ) -> SecondaryKVCacheManagerBase:
        if secondary_backend == "flexkv":
            return FlexKVStorageManager(
                mode=flexkv_mode,
                server_addr=flexkv_server_addr,
                server_port=flexkv_server_port,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                page_size=page_size,
                secondary_wait_timeout_ms=secondary_wait_timeout_ms,
                secondary_fail_policy=secondary_fail_policy,
            )
        return NopSecondaryKVCacheManager()

    def _build_namespace(self, uid: int) -> List[str]:
        return [f"uid:{uid}"]

    def _build_append_page_mapping_from_metadata(
        self,
        batch_size: int,
        offload_page_ids: torch.Tensor,
        new_offload_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        page_ids = offload_page_ids.to(torch.int32).detach().cpu()
        lengths = new_offload_lengths.to(torch.int64).detach().cpu()
        pages_per_req = torch.div(
            lengths + (self.page_size - 1),
            self.page_size,
            rounding_mode="floor",
        ).to(torch.int32)
        indptr = torch.empty(batch_size + 1, dtype=torch.int32)
        indptr[0] = 0
        indptr[1:] = torch.cumsum(pages_per_req, dim=0)
        if int(indptr[-1].item()) != int(page_ids.numel()):
            raise RuntimeError(
                f"append mapping mismatch: indptr[-1]={int(indptr[-1].item())}, "
                f"num_page_ids={int(page_ids.numel())}"
            )
        return page_ids, indptr
    
    def materialize_append_mapping_from_metadata(
        self,
        kv_index_meta: Optional[KVIndexMeta],
        kvcache_metadata: KVCacheMetadata,
    ) -> None:
        if kv_index_meta is None:
            return
        page_ids, indptr = self._build_append_page_mapping_from_metadata(
            batch_size=kv_index_meta.batch_size,
            offload_page_ids=kvcache_metadata.offload_page_ids,
            new_offload_lengths=kvcache_metadata.new_offload_lengths,
        )
        kv_index_meta.append_slot_mapping = page_ids
        kv_index_meta.append_slot_indptr = indptr

    def _normalize_user_ids_and_lengths(
        self,
        uid: Union[int, List[int], torch.Tensor],
        history_lengths_input: Union[int, List[int], torch.Tensor],
    ) -> Tuple[List[int], List[int]]:
        if isinstance(uid, torch.Tensor):
            user_ids = uid.detach().cpu().tolist()
        elif isinstance(uid, list):
            user_ids = uid
        else:
            user_ids = [uid]
        user_ids = [int(x) for x in user_ids]
        if isinstance(history_lengths_input, torch.Tensor):
            total_history_lengths = history_lengths_input.detach().cpu().tolist()
        elif isinstance(history_lengths_input, list):
            total_history_lengths = history_lengths_input
        else:
            total_history_lengths = [history_lengths_input]
        total_history_lengths = [int(x) for x in total_history_lengths]
        if len(total_history_lengths) == 1 and len(user_ids) > 1:
            total_history_lengths = total_history_lengths * len(user_ids)
        if len(user_ids) != len(total_history_lengths):
            raise ValueError(
                f"user_ids and sequence lengths size mismatch: {len(user_ids)} vs {len(total_history_lengths)}"
            )
        return user_ids, total_history_lengths
    def _build_index_meta_from_lookup(self, lookup: KVLookupResult) -> KVIndexMeta:
        seq_start_indices = [
            min(lookup.old_cached_lengths[i], lookup.total_history_lengths[i])
            for i in range(lookup.batch_size)
        ]
        seq_lengths = [
            max(lookup.total_history_lengths[i] - seq_start_indices[i], 0)
            for i in range(lookup.batch_size)
        ]
        namespaces = [self._build_namespace(uid)[0] for uid in lookup.user_ids]
        return KVIndexMeta(
            request_id=lookup.request_id,
            batch_size=lookup.batch_size,
            user_ids=list(lookup.user_ids),
            namespaces=namespaces,
            total_history_lengths=list(lookup.total_history_lengths),
            old_cached_lengths=list(lookup.old_cached_lengths),
            seq_start_indices=seq_start_indices,
            seq_lengths=seq_lengths,
            new_tokens=lookup.new_tokens_upper_bound,
            token_ids=lookup.token_ids,
            token_mask=lookup.token_mask,
        )

    def _normalize_lookup_token_inputs(
        self,
        token_ids: Optional[Union[torch.Tensor, List[List[int]], List[int]]],
        token_mask: Optional[Union[torch.Tensor, List[List[bool]], List[bool]]],
        batch_size: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if token_ids is None:
            return None, None
        row_lengths: Optional[List[int]] = None

        if isinstance(token_ids, torch.Tensor):
            token_ids_cpu = token_ids.detach().cpu()
            if token_ids_cpu.ndim == 1:
                token_ids_cpu = token_ids_cpu.unsqueeze(0)
            if token_ids_cpu.ndim != 2:
                raise ValueError(
                    f"token_ids tensor must be 1D/2D, got shape={tuple(token_ids_cpu.shape)}"
                )
            token_ids_2d = token_ids_cpu.to(torch.int64)
        elif isinstance(token_ids, list):
            if len(token_ids) == 0:
                token_ids_2d = torch.zeros((batch_size, 0), dtype=torch.int64)
                row_lengths = []
            elif isinstance(token_ids[0], list):
                row_count = len(token_ids)
                max_len = max(len(row) for row in token_ids) if row_count > 0 else 0
                token_ids_2d = torch.zeros((row_count, max_len), dtype=torch.int64)
                row_lengths = [len(row) for row in token_ids]
                for idx, row in enumerate(token_ids):
                    if len(row) == 0:
                        continue
                    token_ids_2d[idx, : len(row)] = torch.tensor(
                        row, dtype=torch.int64
                    )
            else:
                token_ids_2d = torch.tensor(token_ids, dtype=torch.int64).view(1, -1)
        else:
            raise ValueError(
                f"Unsupported token_ids type: {type(token_ids)}. "
                "Expected torch.Tensor or List[List[int]]."
            )

        if token_ids_2d.size(0) == 1 and batch_size > 1:
            token_ids_2d = token_ids_2d.repeat(batch_size, 1)
            if row_lengths is not None and len(row_lengths) == 1:
                row_lengths = row_lengths * batch_size
        if token_ids_2d.size(0) != batch_size:
            raise ValueError(
                f"token_ids batch dimension mismatch: token_ids_batch={token_ids_2d.size(0)}, "
                f"batch_size={batch_size}"
            )

        if token_mask is None:
            token_mask_2d = torch.ones_like(token_ids_2d, dtype=torch.bool)
            if row_lengths is not None and len(row_lengths) == batch_size:
                for idx, row_len in enumerate(row_lengths):
                    if row_len < token_ids_2d.size(1):
                        token_mask_2d[idx, row_len:] = False
            return token_ids_2d, token_mask_2d

        if isinstance(token_mask, torch.Tensor):
            token_mask_cpu = token_mask.detach().cpu()
            if token_mask_cpu.ndim == 1:
                token_mask_cpu = token_mask_cpu.unsqueeze(0)
            if token_mask_cpu.ndim != 2:
                raise ValueError(
                    f"token_mask tensor must be 1D/2D, got shape={tuple(token_mask_cpu.shape)}"
                )
            token_mask_2d = token_mask_cpu.to(torch.bool)
        elif isinstance(token_mask, list):
            if len(token_mask) == 0:
                token_mask_2d = torch.zeros_like(token_ids_2d, dtype=torch.bool)
            elif isinstance(token_mask[0], list):
                row_count = len(token_mask)
                max_len = max(len(row) for row in token_mask) if row_count > 0 else 0
                token_mask_2d = torch.zeros((row_count, max_len), dtype=torch.bool)
                for idx, row in enumerate(token_mask):
                    if len(row) == 0:
                        continue
                    token_mask_2d[idx, : len(row)] = torch.tensor(
                        row, dtype=torch.bool
                    )
            else:
                token_mask_2d = torch.tensor(token_mask, dtype=torch.bool).view(1, -1)
        else:
            raise ValueError(
                f"Unsupported token_mask type: {type(token_mask)}. "
                "Expected torch.Tensor or List[List[bool]]."
            )

        if token_mask_2d.size(0) == 1 and batch_size > 1:
            token_mask_2d = token_mask_2d.repeat(batch_size, 1)
        if token_mask_2d.size(0) != batch_size:
            raise ValueError(
                f"token_mask batch dimension mismatch: token_mask_batch={token_mask_2d.size(0)}, "
                f"batch_size={batch_size}"
            )
        if token_mask_2d.size(1) != token_ids_2d.size(1):
            raise ValueError(
                f"token_mask length mismatch: token_mask_width={token_mask_2d.size(1)}, "
                f"token_ids_width={token_ids_2d.size(1)}"
            )

        return token_ids_2d, token_mask_2d

    def lookup_kvcache(
        self,
        user_ids_input: Union[int, List[int], torch.Tensor],
        history_lengths_input: Union[int, List[int], torch.Tensor],
        token_ids: Optional[Union[torch.Tensor, List[List[int]], List[int]]] = None,
        token_mask: Optional[Union[torch.Tensor, List[List[bool]], List[bool]]] = None,
    ) -> KVLookupResult:
        user_ids, total_history_lengths = self._normalize_user_ids_and_lengths(
            user_ids_input, history_lengths_input
        )
        batch_size = len(user_ids)
        lookup_token_ids, lookup_token_mask = self._normalize_lookup_token_inputs(
            token_ids=token_ids,
            token_mask=token_mask,
            batch_size=batch_size,
        )
        old_cached_lengths = list(self.gpu_kvcache_mgr.get_total_cache_length(user_ids))
        new_tokens = max(
            sum(
                max(total_history_lengths[i] - old_cached_lengths[i], 0)
                for i in range(batch_size)
            ),
            0,
        )
        request_id = str(uuid4())
        lookup = KVLookupResult(
            request_id=request_id,
            batch_size=batch_size,
            user_ids=user_ids,
            total_history_lengths=total_history_lengths,
            old_cached_lengths=old_cached_lengths,
            new_tokens_upper_bound=int(new_tokens),
            token_ids=lookup_token_ids,
            token_mask=lookup_token_mask,
        )
        index_meta = self._build_index_meta_from_lookup(lookup)
        lookup.secondary_lookup = self.secondary_kvcache_manager.lookup_kvcache(index_meta)
        return lookup

    def allocate_kvcache(
        self,
        lookup_results: KVLookupResult,
        static_page_ids_gpu_buffer: Optional[torch.Tensor] = None,
        static_offload_page_ids_gpu_buffer: Optional[torch.Tensor] = None,
        static_metadata_gpu_buffer: Optional[torch.Tensor] = None,
        static_onload_handle: Optional[Any] = None,
    ) -> Tuple[KVIndexMeta, KVPrepareResult]:
        index_meta = self._build_index_meta_from_lookup(lookup_results)
        secondary = lookup_results.secondary_lookup or {}
        index_meta.secondary_hit_mask = secondary.get("hit_mask", None)
        index_meta.secondary_get_task_ids = secondary.get("task_ids")
        index_meta.secondary_matched_lengths = secondary.get("matched_lengths")
        index_meta.restore_slot_mapping = secondary.get("restore_slot_mapping", None)
        index_meta.append_slot_mapping = secondary.get("append_slot_mapping", None)
        page_ids_gpu_buffer = (
            static_page_ids_gpu_buffer
            if static_page_ids_gpu_buffer is not None
            else self.static_page_ids_gpu_buffer
        )
        offload_page_ids_gpu_buffer = (
            static_offload_page_ids_gpu_buffer
            if static_offload_page_ids_gpu_buffer is not None
            else self.static_offload_page_ids_gpu_buffer
        )
        metadata_gpu_buffer = (
            static_metadata_gpu_buffer
            if static_metadata_gpu_buffer is not None
            else self.static_metadata_gpu_buffer
        )
        onload_handle = (
            static_onload_handle
            if static_onload_handle is not None
            else self.static_onload_handle
        )
        offload_uids_buffer = torch.empty([lookup_results.batch_size], dtype=torch.int64)
        metadata_host_buffer = torch.empty(
            [lookup_results.batch_size * 7 + 7], dtype=torch.int, pin_memory=True
        )
        kvcache_metadata_fut = self.executor.submit(
            paged_kvcache_ops.prepare_kvcache,
            self.gpu_kvcache_mgr,
            self.host_kv_mgr,
            lookup_results.user_ids,
            lookup_results.total_history_lengths,
            page_ids_gpu_buffer,
            offload_page_ids_gpu_buffer,
            offload_uids_buffer,
            metadata_host_buffer,
            metadata_gpu_buffer,
        )
        onload_handle.reset()
        onload_fut = self.onload_worker.submit(
            self.gpu_kvcache_mgr.onload_kvcache,
            lookup_results.user_ids,
            onload_handle,
        )
        prepare = KVPrepareResult(
            old_cached_lengths=lookup_results.old_cached_lengths,
            new_tokens=lookup_results.new_tokens_upper_bound,
            offload_uids_buffer=offload_uids_buffer,
            metadata_host_buffer=metadata_host_buffer,
            metadata_gpu_buffer=metadata_gpu_buffer,
            kvcache_metadata_fut=kvcache_metadata_fut,
            onload_fut=onload_fut,
        )
        return index_meta, prepare
    
    def onboard_launch_kvcache(
        self,
        kv_index_meta: KVIndexMeta,
    ) -> SecondaryTaskHandle:
        task = self.secondary_kvcache_manager.onboard_launch_kvcache(
            kv_index_meta, kv_index_meta.restore_slot_mapping
        )
        rid = kv_index_meta.request_id
        self.ongoing_onboard_tasks[rid] = task
        self.request_to_task_handles.setdefault(rid, {})["onboard"] = task
        return task
    
    def onboard_try_wait_kvcache_or_fail(
        self,
        kv_index_meta: KVIndexMeta,
        task_handle: Optional[SecondaryTaskHandle],
    ) -> Optional[SecondaryWaitResult]:
        if task_handle is None:
            return SecondaryWaitResult(status=SecondaryTaskStatus.READY, ready=True)
        wait_result = self.secondary_kvcache_manager.onboard_wait_kvcache(task_handle)
        if wait_result.status in (
            SecondaryTaskStatus.FAILED,
            SecondaryTaskStatus.TIMEOUT,
            SecondaryTaskStatus.CANCELLED,
        ):
            self.secondary_kvcache_manager.cancel_task(task_handle)
            self.ongoing_onboard_tasks.pop(kv_index_meta.request_id, None)
            if self.secondary_fail_policy == "fail_close":
                raise RuntimeError(
                    f"onboard wait failed: status={wait_result.status.value}, msg={wait_result.message}"
                )
        elif wait_result.ready:
            self.ongoing_onboard_tasks.pop(kv_index_meta.request_id, None)
        return wait_result
    def lazy_offload_kvcache(
        self,
        kv_index_meta: KVIndexMeta,
    ) -> Optional[SecondaryTaskHandle]:
        if self.offload_mode != KVCacheOffloadMode.LAZY:
            return None
        task = self.secondary_kvcache_manager.offload_launch_kvcache(
            kv_index_meta, kv_index_meta.append_slot_mapping, kv_index_meta.append_slot_indptr,
        )
        rid = kv_index_meta.request_id
        self.ongoing_offload_tasks[rid] = task
        self.request_to_task_handles.setdefault(rid, {})["offload"] = task
        return task

    def finish_or_cancel_kvcache_ops(self, kv_index_meta=None) -> None:
        target_request_id = kv_index_meta.request_id if kv_index_meta is not None else None
        request_ids = (
            [target_request_id]
            if target_request_id is not None
            else list(self.ongoing_offload_tasks.keys())
        )
        for rid in request_ids:
            task = self.ongoing_offload_tasks.get(rid)
            if task is None:
                continue
            wait_result = self.secondary_kvcache_manager.offload_wait_kvcache(task)
            failed = wait_result.status in (
                SecondaryTaskStatus.FAILED,
                SecondaryTaskStatus.TIMEOUT,
                SecondaryTaskStatus.CANCELLED,
            )

            should_raise = failed and self.secondary_fail_policy == "fail_close"
            if failed:
                self.secondary_kvcache_manager.cancel_task(task)
            self.ongoing_offload_tasks.pop(rid, None)
            if rid in self.request_to_task_handles:
                self.request_to_task_handles[rid].pop("offload", None)
                if not self.request_to_task_handles[rid]:
                    self.request_to_task_handles.pop(rid, None)
            if should_raise:
                raise RuntimeError(
                    f"offload wait failed: status={wait_result.status.value}, "
                    f"error_code={wait_result.error_code}, msg={wait_result.message}"
                )

    def prepare_kvcache_wait(
        self,
        onload_fut,
        kvcache_metadata_fut,
        batch_size,
        new_tokens,
        static_page_ids_gpu_buffer,
        static_offload_page_ids_gpu_buffer,
        offload_uids_buffer,
        metadata_host_buffer,
        metadata_gpu_buffer,
        static_onload_handle,
    ):
        kvcache_metadata_fut.result()
        return self.get_kvcache_metadata_from_buffer(
            batch_size,
            new_tokens,
            static_page_ids_gpu_buffer,
            static_offload_page_ids_gpu_buffer,
            offload_uids_buffer,
            metadata_host_buffer,
            metadata_gpu_buffer,
            static_onload_handle,
        )

    def offload_kvcache(self, kvcache_metadata):
        num_offload_pages = len(kvcache_metadata.offload_page_ids)
        if num_offload_pages == 0:
            kvcache_metadata.kv_offload_handle.set_no_offload()
            return None

        self.gpu_kvcache_mgr.offload_kvcache(
            kvcache_metadata.kv_offload_handle,
            kvcache_metadata.offload_user_ids,
            kvcache_metadata.offload_page_ids,
            kvcache_metadata.new_offload_startpos,
            kvcache_metadata.new_offload_lengths,
        )

    def get_kvcache_metadata_from_buffer(
        self,
        batch_size,
        new_tokens,
        static_page_ids_gpu_buffer,
        static_offload_page_ids_gpu_buffer,
        offload_uids_buffer,
        metadata_host_buffer,
        metadata_gpu_buffer,
        static_onload_handle,
    ):
        offload_handle = self.static_empty_offload_handle
        if int(metadata_host_buffer[batch_size * 7 + 5]) > 0:
            offload_handle = paged_kvcache_ops.KVOffloadHandle(
                self.num_layers, self.gpu_kvcache_mgr, True
            )
        return KVCacheMetadata(
            kv_indices=static_page_ids_gpu_buffer[
                : metadata_host_buffer[batch_size * 7 + 4]
            ],
            kv_indptr=metadata_gpu_buffer[: batch_size + 1],
            kv_last_page_len=metadata_gpu_buffer[batch_size + 1 : batch_size * 2 + 1],
            total_history_lengths=metadata_gpu_buffer[
                batch_size * 2 + 1 : batch_size * 3 + 1
            ],
            total_history_offsets=metadata_gpu_buffer[
                batch_size * 3 + 1 : batch_size * 4 + 2
            ],
            batch_indices=metadata_gpu_buffer[
                batch_size * 5 + 4 : batch_size * 5 + 4 + new_tokens
            ],
            position=metadata_gpu_buffer[
                batch_size * 5 + 4 + new_tokens : batch_size * 5 + 4 + new_tokens * 2
            ],
            new_history_nnz=new_tokens,
            new_history_nnz_cuda=metadata_gpu_buffer[
                batch_size * 4 + 2 : batch_size * 4 + 3
            ],
            kv_cache_table=self.cache_table_list,
            kv_onload_handle=static_onload_handle,
            kv_offload_handle=offload_handle,
            offload_user_ids=offload_uids_buffer[
                : metadata_host_buffer[batch_size * 7 + 6]
            ],
            offload_page_ids=static_offload_page_ids_gpu_buffer[
                : int(metadata_host_buffer[batch_size * 7 + 5])
            ].clone(),
            new_offload_startpos=metadata_host_buffer[
                batch_size * 5 + 4 : batch_size * 6 + 4
            ],
            new_offload_lengths=metadata_host_buffer[
                batch_size * 6 + 4 : batch_size * 7 + 4
            ],
            max_seqlen=torch.max(
                metadata_host_buffer[batch_size * 2 + 1 : batch_size * 3 + 1]
            ).item(),
        )

    def strip_cached_tokens(self, batch, origin_num_cached):
        torch.cuda.nvtx.range_push("strip_cached_tokens")

        num_context = len(batch.contextual_feature_names)

        num_cached = torch.clamp_min(origin_num_cached - num_context, 0).to(torch.int32)
        num_cached_action = num_cached // 2
        num_cached_item = num_cached - num_cached_action
        num_hist_cached = torch.concat([num_cached_item, num_cached_action], dim=0)

        old_offsets = batch.features.offsets().cpu()
        old_lengths = batch.features.lengths().cpu()

        item_offset = num_context * batch.batch_size

        new_lengths = torch.zeros_like(old_lengths)
        new_lengths[:item_offset] = torch.where(
            (origin_num_cached == 0).view(-1, batch.batch_size),
            old_lengths[:item_offset].view(-1, batch.batch_size),
            new_lengths[:item_offset].view(-1, batch.batch_size),
        ).view(-1)
        new_lengths[item_offset:] = old_lengths[item_offset:] - num_hist_cached

        startpos = (
            old_offsets[item_offset : item_offset + 2 * batch.batch_size]
            + num_hist_cached
        )
        endpos = old_offsets[item_offset + 1 :]

        old_values = batch.features.values()
        new_hist_value = [
            old_values[startpos[idx] : endpos[idx]]
            for idx in range(2 * batch.batch_size)
        ]

        new_context_value = [
            old_values[idx : idx + 1]
            for idx in range(num_context * batch.batch_size)
            if int(new_lengths[idx]) > 0
        ]

        new_features = KeyedJaggedTensor(
            values=torch.cat(new_context_value + new_hist_value, dim=0),
            lengths=new_lengths.cuda(),
            keys=batch.features.keys(),
        )
        batch.features = new_features

        torch.cuda.nvtx.range_pop()
        return batch

    @classmethod
    def from_config(cls, hstu_config, kvcache_config):
        if kvcache_config.max_queued_offload_tokens is None:
            kvcache_config.max_queued_offload_tokens = (
                4 * hstu_config.max_batch_size * hstu_config.max_seq_len
            )

        secondary_mgr = cls._build_secondary_manager_from_config(
            secondary_backend=getattr(kvcache_config, "secondary_backend", "nop"),
            flexkv_mode=getattr(kvcache_config, "flexkv_mode", "direct"),
            flexkv_server_addr=getattr(kvcache_config, "flexkv_server_addr", ""),
            flexkv_server_port=getattr(kvcache_config, "flexkv_server_port", 0),
            num_layers=hstu_config.num_layers,
            num_heads=hstu_config.num_heads,
            head_dim=hstu_config.head_dim,
            page_size=kvcache_config.page_size,
            secondary_wait_timeout_ms=getattr(kvcache_config, "secondary_wait_timeout_ms", 0),
            secondary_fail_policy=getattr(kvcache_config, "secondary_fail_policy", "fail_open"),
        )
        return cls(
            hstu_config.num_layers,
            hstu_config.num_heads,
            hstu_config.head_dim,
            kvcache_config.page_size,
            kvcache_config.blocks_in_primary_pool,
            math.ceil(
                hstu_config.max_batch_size
                * hstu_config.max_seq_len
                / kvcache_config.page_size
        ),
        0,
        kvcache_config.offload_chunksize,
        -1,
        hstu_config.max_seq_len,
        hstu_config.max_batch_size,
        kvcache_config.max_queued_offload_tokens,
        kvcache_config.num_onload_buffer_chunks,
        kvcache_config.num_offload_buffer_chunks,
        kvcache_config.num_memcpy_workers,
        kvcache_config.enable_nvcomp,
        secondary_mgr,
        getattr(kvcache_config, "offload_mode", "lazy"),
        getattr(kvcache_config, "secondary_wait_timeout_ms", 0),
        getattr(kvcache_config, "secondary_fail_policy", "fail_open"),
    )


class FlexKVStorageManager(SecondaryKVCacheManagerBase):
    def __init__(
        self,
        mode: str = "direct",
        server_addr: str = "",
        server_port: int = 0,
        num_layers: int = 1,
        num_heads: int = 1,
        head_dim: int = 1,
        page_size: int = 16,
        secondary_wait_timeout_ms: int = 0,
        secondary_fail_policy: str = "fail_open",
    ):
        self.mode = mode
        self.server_addr = server_addr
        self.server_port = server_port
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.secondary_wait_timeout_ms = int(secondary_wait_timeout_ms)
        self.secondary_fail_policy = secondary_fail_policy
        self._gpu_register_port: str = ""
        self._ready: bool = False
        self._registered: bool = False
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._adapter = FlexKVClientAdapter(mode, server_addr, server_port)
        self._client = self._build_client()

    def _import_flexkv_sdk(self):
        try:
            from flexkv.kvmanager import KVManager
            from flexkv.common.config import ModelConfig, CacheConfig
            from flexkv.common.request import KVResponseStatus
            return KVManager, ModelConfig, CacheConfig, KVResponseStatus
        except Exception as e:
            raise RuntimeError(
                "FlexKV SDK import failed. "
                "Please install FlexKV in the runtime image/environment "
                "(for example via Dockerfile build step), and ensure ABI compatibility "
                f"(python/cuda/glibc). Original error: {e}"
            ) from e


    def _build_server_recv_port(self) -> str:
        if self.mode != "server_client":
            return ""
        if not self.server_addr:
            raise RuntimeError("server_client mode requires flexkv_server_addr")
        if str(self.server_addr).startswith(("ipc://", "inproc://")):
            return self.server_addr
        if str(self.server_addr).startswith("tcp://"):
            if self.server_port > 0 and self.server_addr.count(":") == 1:
                return f"{self.server_addr}:{self.server_port}"
            return self.server_addr
        if self.server_port <= 0:
            raise RuntimeError("server_client mode requires flexkv_server_port > 0")
        return f"tcp://{self.server_addr}:{self.server_port}"

    def _build_client(self):
        server_recv_port = self._build_server_recv_port()
        # NOTE: FlexKV's GLOBAL_CONFIG_FROM_ENV is initialized at import time.
        # Set env vars before importing SDK modules so runtime config is honored.
        if "FLEXKV_ENABLE_MPS" not in os.environ:
            # In many containers, MPS directories are unavailable for non-root users.
            # Defaulting to disabled avoids init timeout caused by MPS startup failure.
            os.environ["FLEXKV_ENABLE_MPS"] = "0"
        if self.mode == "server_client":
            os.environ["FLEXKV_SERVER_CLIENT_MODE"] = "1"
            os.environ["FLEXKV_SERVER_RECV_PORT"] = server_recv_port
        else:
            os.environ["FLEXKV_SERVER_CLIENT_MODE"] = "0"
            # Use per-instance endpoint in direct mode to avoid cross-test/process leakage.
            server_recv_port = f"ipc:///tmp/flexkv_server_{os.getpid()}_{id(self)}"
            os.environ["FLEXKV_SERVER_RECV_PORT"] = server_recv_port
        self._gpu_register_port = f"{server_recv_port}_gpu_register"

        KVManager, ModelConfig, CacheConfig, _ = self._import_flexkv_sdk()
        # Keep GLOBAL_CONFIG_FROM_ENV consistent even when module was imported earlier.
        try:
            from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

            GLOBAL_CONFIG_FROM_ENV.server_client_mode = (self.mode == "server_client")
            GLOBAL_CONFIG_FROM_ENV.server_recv_port = server_recv_port
            GLOBAL_CONFIG_FROM_ENV.enable_mps = bool(
                int(os.environ.get("FLEXKV_ENABLE_MPS", "0"))
            )
        except Exception:
            pass

        model_cfg = ModelConfig(
            num_layers=self.num_layers,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            tp_size=1,
            dp_size=1,
            dtype=torch.bfloat16,
        )
        cache_cfg = CacheConfig(tokens_per_block=self.page_size)

        client = KVManager(
            model_config=model_cfg,
            cache_config=cache_cfg,
            dp_client_id=0,
            server_recv_port=server_recv_port,
        )
        client.start()
        # IMPORTANT:
        # FlexKV transfer manager becomes ready only after KVTPClient registers GPU blocks.
        # Real registration is triggered by register_gpu_cache_tensors(), so do not block here.
        return client

    def _ensure_client_ready(self) -> None:
        if self._ready:
            return
        init_timeout_s = float(os.environ.get("FLEXKV_CLIENT_INIT_TIMEOUT_S", "45"))
        ready_grace_s = float(os.environ.get("FLEXKV_CLIENT_INIT_READY_GRACE_S", "30"))
        deadline = time.time() + init_timeout_s
        while not self._client.is_ready():
            if time.time() > deadline:
                # GPU blocks are already registered, but TransferEngine worker startup can
                # still take extra time (e.g., large CPU pinning). Give a short grace window
                # before treating it as hard timeout.
                if self._registered and ready_grace_s > 0:
                    grace_deadline = time.time() + ready_grace_s
                    while not self._client.is_ready() and time.time() <= grace_deadline:
                        time.sleep(0.05)
                    if self._client.is_ready():
                        self._ready = True
                        return
                raise RuntimeError(
                    "FlexKV client init timeout: is_ready=False, "
                    f"mode={self.mode}, timeout_s={init_timeout_s}, "
                    f"ready_grace_s={ready_grace_s}, "
                    f"enable_mps={os.environ.get('FLEXKV_ENABLE_MPS')}, "
                    f"server_recv_port={os.environ.get('FLEXKV_SERVER_RECV_PORT', 'ipc:///tmp/flexkv_server')}, "
                    f"gpu_registered={self._registered}, gpu_register_port={self._gpu_register_port}"
                )
            time.sleep(0.05)
        self._ready = True

    def register_gpu_cache_tensors(self, cache_table_list: List[torch.Tensor]) -> None:
        if self._registered:
            return
        if cache_table_list is None or len(cache_table_list) == 0:
            return
        try:
            from flexkv.server.client import KVTPClient
            from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

            # cache_table per-layer shape: [num_blocks, 2, tokens_per_block, num_heads, head_dim]
            # FlexKV worker registration expects [2, num_blocks, tokens_per_block, num_heads, head_dim].
            kv_caches = [layer.permute(1, 0, 2, 3, 4) for layer in cache_table_list]
            first = kv_caches[0]
            device_id = int(first.device.index if first.device.index is not None else 0)
            gpu_layout = KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=len(kv_caches),
                num_block=int(first.shape[1]),
                tokens_per_block=int(first.shape[2]),
                num_head=int(first.shape[3]),
                head_size=int(first.shape[4]),
                is_mla=False,
            )
            tp_client = KVTPClient(
                gpu_register_port=self._gpu_register_port,
                dp_client_id=0,
                device_id=device_id,
            )
            tp_client.register_to_server(kv_caches=kv_caches, kv_layout=gpu_layout)
            self._registered = True
            self._ensure_client_ready()
        except Exception as e:
            raise RuntimeError(f"FlexKV GPU cache registration failed: {e}") from e

    def _failed_wait_result(
        self,
        msg: str,
        error_code: str,
        failed_user_ids: Optional[List[int]] = None,
    ) -> SecondaryWaitResult:
        failed_mask = None
        if failed_user_ids:
            failed_mask = torch.ones((len(failed_user_ids),), dtype=torch.bool)
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.FAILED,
            ready=False,
            error_code=error_code,
            message=msg,
            failed_mask=failed_mask,
            failed_user_ids=failed_user_ids,
        )

    def _wait_task_ids(self, task_ids: List[int]) -> Dict[int, Any]:
        if len(task_ids) == 0:
            return {}
        if self.secondary_wait_timeout_ms > 0:
            timeout_s = float(self.secondary_wait_timeout_ms) / 1000.0
            return self._client.wait(task_ids, timeout=timeout_s, completely=True)
        return self._client.try_wait(task_ids)

    def _convert_wait_result(
        self,
        responses: Dict[int, Any],
        user_ids: List[int],
        timeout_code: str,
        failed_code: str,
    ) -> SecondaryWaitResult:
        if responses is None:
            return self._failed_wait_result(
                msg="wait returned None",
                error_code=failed_code,
                failed_user_ids=user_ids,
            )
        if len(responses) == 0:
            return SecondaryWaitResult(status=SecondaryTaskStatus.READY, ready=True)

        has_timeout = False
        has_cancelled = False
        has_failed = False
        status_msgs: List[str] = []
        for task_id, resp in responses.items():
            status_name = str(getattr(getattr(resp, "status", None), "name", "UNKNOWN"))
            status_msgs.append(f"{task_id}:{status_name}")
            if status_name == "SUCCESS":
                continue
            if status_name == "TIMEOUT":
                has_timeout = True
            elif status_name == "CANCELLED":
                has_cancelled = True
            else:
                has_failed = True

        if has_timeout:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.TIMEOUT,
                ready=False,
                error_code=timeout_code,
                message=";".join(status_msgs),
                failed_mask=torch.ones((len(user_ids),), dtype=torch.bool) if user_ids else None,
                failed_user_ids=user_ids if user_ids else None,
            )
        if has_failed or has_cancelled:
            status = SecondaryTaskStatus.CANCELLED if has_cancelled and not has_failed else SecondaryTaskStatus.FAILED
            return SecondaryWaitResult(
                status=status,
                ready=False,
                error_code=failed_code,
                message=";".join(status_msgs),
                failed_mask=torch.ones((len(user_ids),), dtype=torch.bool) if user_ids else None,
                failed_user_ids=user_ids if user_ids else None,
            )
        return SecondaryWaitResult(status=SecondaryTaskStatus.READY, ready=True)

    def lookup_kvcache(self, index_meta: KVIndexMeta) -> Dict[str, Any]:
        self._ensure_client_ready()
        if index_meta.token_ids is None or index_meta.token_mask is None:
            return {
                "backend": "flexkv",
                "error_code": SecondaryErrorCode.LOOKUP_MISSING_TOKENS.value,
                "task_ids": [],
                "matched_lengths": [0] * index_meta.batch_size,
                "hit_mask": None,
                "restore_slot_mapping": None,
                "append_slot_mapping": None,
            }
        try:
            requests = self._adapter.to_get_match_requests(index_meta)
            task_ids: List[int] = []
            matched_lengths: List[int] = []
            hit_masks: List[np.ndarray] = []
            for req in requests:
                if req["token_ids"].size == 0:
                    task_ids.append(-1)
                    matched_lengths.append(0)
                    hit_masks.append(np.zeros_like(req["token_mask"], dtype=np.bool_))
                    continue
                task_id, matched_mask = self._client.get_match(
                    token_ids=req["token_ids"],
                    token_mask=req["token_mask"],
                    namespace=req["namespace"],
                )
                matched_mask = np.asarray(matched_mask, dtype=np.bool_)
                task_ids.append(int(task_id))
                matched_lengths.append(int(matched_mask.sum()))
                hit_masks.append(matched_mask)
            return self._adapter.from_get_match_responses(
                index_meta=index_meta,
                task_ids=task_ids,
                matched_lengths=matched_lengths,
                hit_masks=hit_masks,
            )
        except Exception as e:
            return {
                "backend": "flexkv",
                "error": str(e),
                "error_code": SecondaryErrorCode.LOOKUP_FAILED.value,
                "task_ids": [],
                "matched_lengths": [0] * index_meta.batch_size,
                "hit_mask": None,
                "restore_slot_mapping": None,
                "append_slot_mapping": None,
            }

    def onboard_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        restore_slot_mapping: Optional[torch.Tensor],
    ) -> SecondaryTaskHandle:
        self._ensure_client_ready()
        endpoint = (
            f"{self.server_addr}:{self.server_port}"
            if self.mode == "server_client"
            else "local"
        )
        try:
            payload = self._adapter.to_onboard_launch_payload(index_meta, restore_slot_mapping)
            task_ids: List[int] = payload["task_ids"]
            slot_mappings: List[np.ndarray] = payload["slot_mappings"]
            all_get_task_ids = [int(x) for x in (index_meta.secondary_get_task_ids or []) if int(x) >= 0]

            if len(task_ids) == 0:
                if len(all_get_task_ids) > 0:
                    self._client.cancel(task_ids=all_get_task_ids)
                return SecondaryTaskHandle(
                    backend="flexkv",
                    handle=None,
                    status=SecondaryTaskStatus.SKIPPED,
                    metadata={"reason": "no onboard restore blocks"},
                )

            launched = self._client.launch(
                task_ids=task_ids,
                slot_mappings=slot_mappings,
                as_batch=True,
            )
            launched_ids = [int(x) for x in launched]
            to_cancel = [tid for tid in all_get_task_ids if tid not in task_ids]
            if len(to_cancel) > 0:
                self._client.cancel(task_ids=to_cancel)

            task_key = f"onboard:{index_meta.request_id}"
            self._tasks[task_key] = {
                "task_ids": launched_ids,
                "kind": "onboard",
                "mode": self.mode,
                "endpoint": endpoint,
                "user_ids": list(index_meta.user_ids),
            }
            return SecondaryTaskHandle(
                backend="flexkv",
                handle={
                    "task_key": task_key,
                    "task_ids": launched_ids,
                    "kind": "onboard",
                    "mode": self.mode,
                    "endpoint": endpoint,
                },
                status=SecondaryTaskStatus.LAUNCHED,
            )
        except Exception as e:
            return SecondaryTaskHandle(
                backend="flexkv",
                handle=None,
                status=SecondaryTaskStatus.FAILED,
                metadata={
                    "error": str(e),
                    "error_code": SecondaryErrorCode.ONBOARD_WAIT_FAILED.value,
                    "kind": "onboard",
                    "mode": self.mode,
                    "endpoint": endpoint,
                },
            )

    def onboard_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        if task_handle is None or task_handle.handle is None:
            return SecondaryWaitResult(status=SecondaryTaskStatus.SKIPPED, ready=True)
        try:
            task_key = task_handle.handle.get("task_key")
            state = self._tasks.get(task_key)
            if state is None:
                return self._failed_wait_result(
                    msg="onboard task not found",
                    error_code=SecondaryErrorCode.ONBOARD_TASK_NOT_FOUND.value,
                )
            task_ids = list(state.get("task_ids", []))
            responses = self._wait_task_ids(task_ids)
            return self._convert_wait_result(
                responses=responses,
                user_ids=list(state.get("user_ids", [])),
                timeout_code=SecondaryErrorCode.ONBOARD_TIMEOUT.value,
                failed_code=SecondaryErrorCode.ONBOARD_WAIT_FAILED.value,
            )
        except Exception as e:
            return self._failed_wait_result(
                msg=f"onboard_wait exception: {e}",
                error_code=SecondaryErrorCode.ONBOARD_WAIT_FAILED.value,
            )

    def offload_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        append_slot_mapping: Optional[torch.Tensor],
        append_slot_indptr: Optional[torch.Tensor] = None,
    ) -> SecondaryTaskHandle:
        self._ensure_client_ready()
        endpoint = (
            f"{self.server_addr}:{self.server_port}"
            if self.mode == "server_client"
            else "local"
        )
        try:
            reqs = self._adapter.to_offload_requests(
                index_meta=index_meta,
                append_slot_mapping=append_slot_mapping,
                append_slot_indptr=append_slot_indptr,
                tokens_per_block=self.page_size,
            )
            if len(reqs) == 0:
                return SecondaryTaskHandle(
                    backend="flexkv",
                    handle=None,
                    status=SecondaryTaskStatus.SKIPPED,
                    metadata={"reason": "no offload blocks"},
                )

            task_ids: List[int] = []
            req_user_ids: List[int] = []
            for req in reqs:
                task_id = self._client.put_async(
                    token_ids=req["token_ids"],
                    slot_mapping=req["slot_mapping"],
                    token_mask=req["token_mask"],
                    namespace=req["namespace"],
                )
                task_ids.append(int(task_id))
                req_user_ids.append(int(req["user_id"]))

            task_key = f"offload:{index_meta.request_id}"
            self._tasks[task_key] = {
                "task_ids": task_ids,
                "kind": "offload",
                "mode": self.mode,
                "endpoint": endpoint,
                "user_ids": req_user_ids,
            }
            return SecondaryTaskHandle(
                backend="flexkv",
                handle={
                    "task_key": task_key,
                    "task_ids": task_ids,
                    "kind": "offload",
                    "mode": self.mode,
                    "endpoint": endpoint,
                },
                status=SecondaryTaskStatus.LAUNCHED,
                metadata={"request_id": index_meta.request_id},
            )
        except Exception as e:
            return SecondaryTaskHandle(
                backend="flexkv",
                handle=None,
                status=SecondaryTaskStatus.FAILED,
                metadata={
                    "error": str(e),
                    "error_code": SecondaryErrorCode.OFFLOAD_WAIT_FAILED.value,
                    "kind": "offload",
                    "mode": self.mode,
                    "endpoint": endpoint,
                },
            )

    def offload_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        if task_handle is None or task_handle.handle is None:
            return SecondaryWaitResult(status=SecondaryTaskStatus.SKIPPED, ready=True)
        try:
            task_key = task_handle.handle.get("task_key")
            state = self._tasks.get(task_key)
            if state is None:
                return self._failed_wait_result(
                    msg="offload task not found",
                    error_code=SecondaryErrorCode.OFFLOAD_TASK_NOT_FOUND.value,
                )
            task_ids = list(state.get("task_ids", []))
            responses = self._wait_task_ids(task_ids)
            return self._convert_wait_result(
                responses=responses,
                user_ids=list(state.get("user_ids", [])),
                timeout_code=SecondaryErrorCode.OFFLOAD_TIMEOUT.value,
                failed_code=SecondaryErrorCode.OFFLOAD_WAIT_FAILED.value,
            )
        except Exception as e:
            return self._failed_wait_result(
                msg=f"offload_wait exception: {e}",
                error_code=SecondaryErrorCode.OFFLOAD_WAIT_FAILED.value,
            )

    def cancel_task(self, task_handle: SecondaryTaskHandle) -> None:
        if task_handle is None or task_handle.handle is None:
            return
        task_key = task_handle.handle.get("task_key")
        state = self._tasks.get(task_key)
        if state is None:
            return
        task_ids = list(state.get("task_ids", []))
        if len(task_ids) > 0:
            try:
                self._client.cancel(task_ids=task_ids)
            except Exception:
                pass
        self._tasks.pop(task_key, None)

class FlexKVClientAdapter:
    def __init__(self, mode: str, server_addr: str = "", server_port: int = 0):
        self.mode = mode
        self.server_addr = server_addr
        self.server_port = server_port

    def _to_numpy_2d(self, x: Optional[torch.Tensor], dtype) -> Optional[np.ndarray]:
        if x is None:
            return None
        arr = x.detach().cpu().numpy()
        return arr.astype(dtype)

    def to_get_match_requests(self, index_meta: KVIndexMeta) -> List[Dict[str, Any]]:
        token_ids_2d = self._to_numpy_2d(index_meta.token_ids, np.int64)
        token_mask_2d = self._to_numpy_2d(index_meta.token_mask, np.bool_)
        if token_ids_2d is None or token_mask_2d is None:
            return []
        reqs: List[Dict[str, Any]] = []
        for i in range(index_meta.batch_size):
            row_ids = token_ids_2d[i]
            row_mask = token_mask_2d[i]
            true_idx = np.where(row_mask)[0]
            if true_idx.size == 0:
                reqs.append(
                    {
                        "user_id": int(index_meta.user_ids[i]),
                        "namespace": [index_meta.namespaces[i]],
                        "token_ids": np.zeros((0,), dtype=np.int64),
                        "token_mask": np.zeros((0,), dtype=np.bool_),
                    }
                )
                continue
            end = int(true_idx[-1]) + 1
            reqs.append(
                {
                    "user_id": int(index_meta.user_ids[i]),
                    "namespace": [index_meta.namespaces[i]],
                    "token_ids": row_ids[:end].astype(np.int64),
                    "token_mask": row_mask[:end].astype(np.bool_),
                }
            )
        return reqs

    def from_get_match_responses(
        self,
        index_meta: KVIndexMeta,
        task_ids: List[int],
        matched_lengths: List[int],
        hit_masks: List[np.ndarray],
    ) -> Dict[str, Any]:
        batch_size = index_meta.batch_size
        max_len = 0
        if index_meta.token_mask is not None:
            max_len = int(index_meta.token_mask.shape[1])
        elif len(hit_masks) > 0:
            max_len = max(int(m.shape[0]) for m in hit_masks)
        hit_mask_2d = np.zeros((batch_size, max_len), dtype=np.bool_)
        for i, m in enumerate(hit_masks):
            if i >= batch_size:
                break
            upto = min(max_len, int(m.shape[0]))
            hit_mask_2d[i, :upto] = m[:upto]
        return {
            "backend": "flexkv",
            "task_ids": task_ids,
            "matched_lengths": matched_lengths,
            "hit_mask": torch.from_numpy(hit_mask_2d).to(torch.bool) if max_len > 0 else None,
            "restore_slot_mapping": None,
            "append_slot_mapping": None,
        }

    def to_onboard_launch_payload(
        self,
        index_meta: KVIndexMeta,
        restore_slot_mapping: Any,
    ) -> Dict[str, Any]:
        if not isinstance(restore_slot_mapping, dict):
            return {"task_ids": [], "slot_mappings": []}
        task_ids = [int(x) for x in restore_slot_mapping.get("task_ids", [])]
        slot_mappings = [
            np.asarray(x, dtype=np.int64) for x in restore_slot_mapping.get("slot_mappings", [])
        ]
        valid_task_ids: List[int] = []
        valid_slot_mappings: List[np.ndarray] = []
        for t, s in zip(task_ids, slot_mappings):
            if s.ndim != 1 or s.size == 0:
                continue
            valid_task_ids.append(t)
            valid_slot_mappings.append(s)
        return {"task_ids": valid_task_ids, "slot_mappings": valid_slot_mappings}

    def to_offload_requests(
        self,
        index_meta: KVIndexMeta,
        append_slot_mapping: Optional[torch.Tensor],
        append_slot_indptr: Optional[torch.Tensor],
        tokens_per_block: int,
    ) -> List[Dict[str, Any]]:
        if append_slot_mapping is None or append_slot_indptr is None:
            return []
        if index_meta.token_ids is None or index_meta.token_mask is None:
            return []

        page_ids = append_slot_mapping.to(torch.int64).detach().cpu().numpy()
        indptr = append_slot_indptr.to(torch.int64).detach().cpu().numpy()
        token_ids_2d = index_meta.token_ids.to(torch.int64).detach().cpu().numpy()
        token_mask_2d = index_meta.token_mask.to(torch.bool).detach().cpu().numpy()
        matched_lengths = index_meta.secondary_matched_lengths or []

        reqs: List[Dict[str, Any]] = []
        for i in range(index_meta.batch_size):
            p0 = int(indptr[i])
            p1 = int(indptr[i + 1])
            if p1 <= p0:
                continue
            req_pages = page_ids[p0:p1]
            slot_mapping_full = np.repeat(req_pages * tokens_per_block, tokens_per_block).astype(np.int64)

            row_ids = token_ids_2d[i]
            row_mask = token_mask_2d[i]
            valid_token_ids = row_ids[row_mask]

            old_cached = int(index_meta.old_cached_lengths[i])
            if i < len(matched_lengths):
                old_cached = max(old_cached, int(matched_lengths[i]))

            append_token_ids = valid_token_ids[old_cached:]
        
            aligned = (append_token_ids.size // tokens_per_block) * tokens_per_block
            aligned = min(aligned, slot_mapping_full.size)
            if aligned <= 0:
                continue

            reqs.append(
                {
                    "user_id": int(index_meta.user_ids[i]),
                    "namespace": [index_meta.namespaces[i]],
                    "token_ids": append_token_ids[:aligned].astype(np.int64),
                    "token_mask": np.ones((aligned,), dtype=np.bool_),
                    "slot_mapping": slot_mapping_full[:aligned].astype(np.int64),
                }
            )
        return reqs


AsyncHSTUKVCacheManager = KVCacheManager