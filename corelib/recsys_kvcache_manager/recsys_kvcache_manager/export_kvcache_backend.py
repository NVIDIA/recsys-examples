# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export/AOTI backend for KV cache management."""

from typing import Optional, Tuple

import torch

from .host_kvstorage_manager import HostKVStorageBase, HostKVTaskHandle, HostKVWaitResult
from .kvcache_backend import KVCacheBackend
from .kvcache_metadata import KVCacheMetadata
from .kvcache_utils import KVIndexMeta, KVLookupResult


class ExportKVCacheBackend(KVCacheBackend):
	"""Export/AOTI backend stub.

	This module is the canonical home for the export backend implementation.
	The methods are intentionally thin at this stage and mirror the default
	backend surface so the public `KVCacheManager` can swap backends cleanly.
	"""

	def __init__(self, runtime=None):
		self.runtime = runtime

	def lookup_kvcache(
		self,
		user_ids: torch.Tensor,
		sequence_lengths: torch.Tensor,
	) -> Tuple[KVIndexMeta, KVLookupResult]:
		(
			cached_start_indices,
			cached_lengths,
			gpu_cached_start_indices,
			gpu_cached_lengths,
			host_cached_start_indices,
			host_cached_lengths,
			task_ids,
		) = torch.ops.kvcache_manager_ops.lookup_kvcache(user_ids, sequence_lengths)

		index_meta = KVIndexMeta(
			user_ids=user_ids,
			seq_lengths=sequence_lengths,
		)
		lookup_result = KVLookupResult(
			user_ids=user_ids,
			cached_start_indices=cached_start_indices,
			cached_lengths=cached_lengths,
			gpu_cached_start_indices=gpu_cached_start_indices,
			gpu_cached_lengths=gpu_cached_lengths,
			host_cached_start_indices=host_cached_start_indices,
			host_cached_lengths=host_cached_lengths,
			extra={
				"task_ids": task_ids
			}
		)
		return index_meta, lookup_result

	def allocate_kvcache(
		self,
		index_meta: KVIndexMeta,
		lookup_results: KVLookupResult,
		output_kvcache_metadata: Optional[KVCacheMetadata] = None,
	) -> KVCacheMetadata:
		# assert output_kvcache_metadata is None, "Pre-allocated KVCacheMetadata is not supported in ExportKVCacheBackend yet."

		(metadata_buffer, metadata_tensors) = torch.ops.kvcache_manager_ops.allocate_kvcache(
			index_meta.user_ids,
			index_meta.seq_lengths,
			lookup_results.cached_lengths,
			lookup_results.host_cached_lengths,
		)

		return KVCacheMetadata(
			page_ids_gpu_buffer=metadata_buffer[0],
			metadata_gpu_buffer=metadata_buffer[1],
			kv_indices=metadata_buffer[0],
			
			kv_indptr=metadata_tensors[0],
			kv_last_page_len=metadata_tensors[1],
			total_history_lengths=metadata_tensors[2],
			total_history_offsets=metadata_tensors[3],
			new_history_offsets=metadata_tensors[4],
			batch_indices=metadata_tensors[5],
			position=metadata_tensors[6],

			new_history_nnz=metadata_tensors[7],
			new_history_nnz_cuda=metadata_tensors[8],

			kv_seqlens=metadata_tensors[9],
			kv_seqlen_offsets=metadata_tensors[10],
			kv_onload_handle=None,
		)

	def onboard_launch(
		self,
		index_meta: KVIndexMeta,
		lookup_result: KVLookupResult,
		kvcache_metadata: KVCacheMetadata,
	) -> HostKVTaskHandle:
		slot_mappings = torch.ops.kvcache_manager_ops.onboard_kvcache_launch(
			index_meta.user_ids,
			index_meta.seq_lengths,
			lookup_result.cached_lengths,
			lookup_result.host_cached_lengths,
			lookup_result.gpu_cached_start_indices,
			lookup_result.gpu_cached_lengths,
			lookup_result.extra["task_ids"],
			kvcache_metadata.kv_indices,
			kvcache_metadata.kv_indptr,
		)

		# In export backend, all user_ids and slot mappings are recorded. User ids with no cache to onboard will have task_id of -1, and the corresponding slot mapping can be ignored in the downstream processing.
		onload_handle = _FlexKVOnloadHandle(
            task_ids=lookup_result.extra["task_ids"],
            uids=index_meta.user_ids,
            slot_mappings=slot_mappings,
        )
		return HostKVTaskHandle(
			backend="flexkv",
            user_ids=onload_handle.uids,
            handle=onload_handle,
            status=HostKVTaskStatus.LAUNCHED,
            # metadata={
            #     "onboard_start_indices": torch.tensor(
            #         onboard_start_indices, dtype=torch.int32
            #     ),
            #     "onboard_lengths": torch.tensor(onboard_lengths, dtype=torch.int32),
            # },
		)

	def onboard_try_wait(
		self,
		kv_index_meta: KVIndexMeta,
		task_handle: Optional[HostKVTaskHandle],
	) -> Optional[HostKVWaitResult]:
		raise NotImplementedError("ExportKVCacheBackend.onboard_try_wait is not implemented yet.")

	def onboard_wait(
		self,
		kv_index_meta: KVIndexMeta,
		task_handle: Optional[HostKVTaskHandle],
	) -> Optional[HostKVWaitResult]:
		torch.ops.kvcache_manager_ops.onboard_kvcache_wait(
			task_handle.handle.task_ids,
		)

	def offload_launch(
		self,
		index_meta: KVIndexMeta,
		kvcache_metadata: Optional[KVCacheMetadata] = None,
	):
		torch.ops.kvcache_manager_ops.offload_launch(
			task_handle.handle.task_ids,
		)

	def offload_try_wait(self) -> None:
		raise NotImplementedError("ExportKVCacheBackend.offload_try_wait is not implemented yet.")
	
	def offload_reap_completed(self) -> None:
		torch.ops.kvcache_manager_ops.offload_reap_completed()
	
	def offload_flush(self) -> None:
		raise NotImplementedError("ExportKVCacheBackend.offload_flush is not implemented yet.")

	def evict(
		self, user_ids: torch.Tensor, for_gpu: bool = False, for_host: bool = False
	):
		raise NotImplementedError("ExportKVCacheBackend.evict is not implemented yet.")

	def evict_all(self, for_gpu: bool = False, for_host: bool = False):
		raise NotImplementedError("ExportKVCacheBackend.evict_all is not implemented yet.")


__all__ = ["ExportKVCacheBackend"]
