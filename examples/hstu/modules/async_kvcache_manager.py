import math
from concurrent.futures import ThreadPoolExecutor

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

@dataclass
class KVLookupResult:
    request_id: str
    batch_size: int
    user_ids: List[int]
    total_history_lengths: List[int]
    old_cached_lengths: List[int]
    new_tokens_upper_bound: int
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
    restore_slot_mapping: Optional[torch.Tensor] = None
    append_slot_mapping: Optional[torch.Tensor] = None
    secondary_hit_mask: Optional[torch.Tensor] = None

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

class SecondaryKVCacheManagerBase(ABC):
    def __init__(self):
        #self.offload_mode = KVCacheOffloadMode.LAZY
        pass

    @abstractmethod
    def lookup_kvcache(self, index_meta: KVIndexMeta) -> Dict[str, Any]:
        pass

    @abstractmethod
    def onboard_launch_kvcache(
        self, index_meta: KVIndexMeta, restore_slot_mapping: Optional[torch.Tensor]
    ) -> SecondaryTaskHandle:
        pass

    @abstractmethod
    def onboard_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        pass

    @abstractmethod
    def offload_launch_kvcache(
        self, index_meta: KVIndexMeta, append_slot_mapping: Optional[torch.Tensor]
    ) -> SecondaryTaskHandle:
        pass

    @abstractmethod
    def offload_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        pass

    @abstractmethod
    def cancel_task(self, task_handle: SecondaryTaskHandle) -> None:
        pass

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

    def offload_launch_kvcache(self, index_meta, append_slot_mapping):
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
        namespace_mode: str = "uid",
        namespace_base: str = "recsys_hstu",
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
        self.namespace_mode = namespace_mode
        self.namespace_base = namespace_base

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

    def _build_namespace(self, uid: int) -> List[str]:
        """
        Phase 1 namespace helper.
        - uid mode:       uid:<uid>
        - default/fallback: <base>:uid=<uid>
        """
        if self.namespace_mode == "uid":
            return [f"uid:{uid}"]
        return [f"{self.namespace_base}:uid={uid}"]    

    def _normalize_uid_and_sequence(
        self,
        uid: Union[int, List[int], torch.Tensor],
        sequence_or_lengths: Union[int, List[int], torch.Tensor],
    ) -> Tuple[List[int], List[int]]:
        if isinstance(uid, torch.Tensor):
            user_ids = uid.detach().cpu().tolist()
        elif isinstance(uid, list):
            user_ids = uid
        else:
            user_ids = [uid]
        user_ids = [int(x) for x in user_ids]
        if isinstance(sequence_or_lengths, torch.Tensor):
            total_history_lengths = sequence_or_lengths.detach().cpu().tolist()
        elif isinstance(sequence_or_lengths, list):
            total_history_lengths = sequence_or_lengths
        else:
            total_history_lengths = [sequence_or_lengths]
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
        )

    def lookup_kvcache(
        self,
        uid_or_uids: Union[int, List[int], torch.Tensor],
        sequence_or_lengths: Union[int, List[int], torch.Tensor],
    ) -> KVLookupResult:
        user_ids, total_history_lengths = self._normalize_uid_and_sequence(
            uid_or_uids, sequence_or_lengths
        )
        batch_size = len(user_ids)
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
        )
        index_meta = self._build_index_meta_from_lookup(lookup)
        lookup.secondary_lookup = self.secondary_kvcache_manager.lookup_kvcache(index_meta)
        return lookup

    def allocate_kvcache(
        self,
        uid_or_uids: Union[int, List[int], torch.Tensor],
        lookup_results: KVLookupResult,
        static_page_ids_gpu_buffer: Optional[torch.Tensor] = None,
        static_offload_page_ids_gpu_buffer: Optional[torch.Tensor] = None,
        static_metadata_gpu_buffer: Optional[torch.Tensor] = None,
        static_onload_handle: Optional[Any] = None,
    ) -> Tuple[KVIndexMeta, KVPrepareResult]:
        index_meta = self._build_index_meta_from_lookup(lookup_results)
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
        uid_or_uids,
        kv_index_meta: KVIndexMeta,
        lookup_results: KVLookupResult,
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
        uid_or_uids,
        kv_index_meta: KVIndexMeta,
        lookup_results: KVLookupResult,
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
        uid_or_uids,
        kv_index_meta: KVIndexMeta,
        lookup_results: KVLookupResult,
    ) -> Optional[SecondaryTaskHandle]:
        if self.offload_mode != KVCacheOffloadMode.LAZY:
            return None
        task = self.secondary_kvcache_manager.offload_launch_kvcache(
            kv_index_meta, kv_index_meta.append_slot_mapping
        )
        rid = kv_index_meta.request_id
        self.ongoing_offload_tasks[rid] = task
        self.request_to_task_handles.setdefault(rid, {})["offload"] = task
        return task
    def finish_or_cancel_kvcache_ops(self, uid_or_uids=None, kv_index_meta=None) -> None:
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
            if wait_result.status in (
                SecondaryTaskStatus.FAILED,
                SecondaryTaskStatus.TIMEOUT,
                SecondaryTaskStatus.CANCELLED,
            ):
                self.secondary_kvcache_manager.cancel_task(task)
            self.ongoing_offload_tasks.pop(rid, None)
            if rid in self.request_to_task_handles:
                self.request_to_task_handles[rid].pop("offload", None)
                if not self.request_to_task_handles[rid]:
                    self.request_to_task_handles.pop(rid, None)

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
        metadata_gpu_buffer,  # input static
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
        metadata_gpu_buffer,  # input static
        static_onload_handle,
    ):
        # assert int(metadata_host_buffer[batch_size * 4 + 2]) == new_tokens
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
        #item_offset + batch.batch_size （Todo：junyi check this）

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
        None,
        kvcache_config.namespace_mode,
        kvcache_config.namespace_base,
            getattr(kvcache_config, "offload_mode", "lazy"),
            getattr(kvcache_config, "secondary_wait_timeout_ms", 0),
            getattr(kvcache_config, "secondary_fail_policy", "fail_open"),
    )
AsyncHSTUKVCacheManager = KVCacheManager