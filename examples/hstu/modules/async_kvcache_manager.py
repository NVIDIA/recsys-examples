import torch
from concurrent.futures import ThreadPoolExecutor
import paged_kvcache_ops
import math
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from configs import KVCacheMetadata
import os

# def offload_callback(fut):
#     torch.cuda.empty_cache()

class AsyncHSTUKVCacheManager:
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
    ):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.onload_worker = ThreadPoolExecutor(max_workers=1)
        # self.offload_worker = ThreadPoolExecutor(max_workers=4)

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
        self.max_num_pages_per_seq = math.ceil(self.max_sequence_length / self.page_size)

        self.cache_table = torch.empty(
            [num_layers, (num_primary_cache_pages + num_onload_buffer_pages), 2, self.page_size, self.num_heads, self.head_dim],
            dtype=torch.bfloat16, device=torch.cuda.current_device()
        )

        self.host_kv_mgr = paged_kvcache_ops.HostKVStorageImpl(
            self.num_layers, self.num_heads, self.head_dim, self.page_size, self.chunk_size
        )
        self.gpu_kvcache_mgr = paged_kvcache_ops.GPUKVCacheMangerImpl(
            self.num_layers, self.num_heads, self.head_dim, self.page_size,
            self.num_primary_cache_pages, self.num_onload_buffer_pages,
            self.num_reserved_buffer_pages, self.chunk_size,
            self.max_num_sequences, self.max_num_sequences, 
            self.cache_table,
            self.host_kv_mgr
        )

        self.static_page_ids_gpu_buffer = torch.empty([self.max_batch_size * self.max_num_pages_per_seq,], dtype=torch.int32).cuda()
        self.static_offload_page_ids_gpu_buffer = torch.empty([self.max_batch_size * self.max_num_pages_per_seq,], dtype=torch.int32).cuda()
        # self.static_pinned_kv_buffer = torch.empty(
        #     [self.num_layers, self.max_batch_size * self.max_num_pages_per_seq, 2, self.page_size, self.num_heads, self.head_dim],
        #     dtype=torch.bfloat16, pin_memory=True
        # )
        self.static_onload_handle = paged_kvcache_ops.KVOnloadHandle(self.num_layers)

        self.cache_table_list = [ self.cache_table[idx] for idx in range(self.num_layers) ]

    def prepare_kvcache_async(self, 
        batch_size, 
        user_ids, 
        total_history_lengths, 
        static_page_ids_gpu_buffer, 
        static_offload_page_ids_gpu_buffer,
        # static_pinned_kv_buffer, 
        static_onload_handle,
    ):
        origin_cached_lengths = self.gpu_kvcache_mgr.get_total_cache_length(user_ids)
        new_tokens = sum([ total_history_lengths[idx] - origin_cached_lengths[idx] for idx in range(batch_size) ])
        if new_tokens <= 0:
            print(total_history_lengths)
            print(origin_cached_lengths)

        offload_uids_buffer = torch.empty([batch_size,], dtype=torch.int64)
        metadata_host_buffer = torch.empty([batch_size * 7 + 7,], dtype=torch.int, pin_memory=True)
        metadata_gpu_buffer = torch.empty([batch_size * 5 + 4 + new_tokens * 2,], dtype=torch.int, device = torch.cuda.current_device())

        kvcache_metadata_fut = self.executor.submit(paged_kvcache_ops.prepare_kvcache, 
            self.gpu_kvcache_mgr, self.host_kv_mgr, 
            user_ids, total_history_lengths,
            static_page_ids_gpu_buffer, static_offload_page_ids_gpu_buffer,
            offload_uids_buffer,
            metadata_host_buffer, metadata_gpu_buffer)

        static_onload_handle.reset()
        onload_fut = self.onload_worker.submit(self.gpu_kvcache_mgr.onload_kvcache, 
            user_ids, static_onload_handle)

        return origin_cached_lengths, new_tokens, offload_uids_buffer, metadata_host_buffer, metadata_gpu_buffer, kvcache_metadata_fut, onload_fut
    
    def prepare_kvcache_wait(self, 
        onload_fut,
        kvcache_metadata_fut,
        batch_size, 
        new_tokens,
        static_page_ids_gpu_buffer, 
        static_offload_page_ids_gpu_buffer,
        offload_uids_buffer,
        metadata_host_buffer, 
        metadata_gpu_buffer,
        static_onload_handle):

        # onload_fut.result()
        kvcache_metadata_fut.result()
        return self.get_kvcache_metadata_from_buffer(
            batch_size,
            new_tokens,
            static_page_ids_gpu_buffer,
            static_offload_page_ids_gpu_buffer,
            offload_uids_buffer,
            metadata_host_buffer,
            metadata_gpu_buffer,
            static_onload_handle)

    def offload_kvcache(self, kvcache_metadata):
        num_offload_pages = len(kvcache_metadata.offload_page_ids)
        if num_offload_pages == 0:
            return None
        
        kvcache_metadata.kv_offload_handle.record_ready()
        
        kvcache_metadata.gather_kv_gpu_buffer = torch.empty(
            [self.num_layers * num_offload_pages, 2, self.page_size, self.num_heads, self.head_dim],
            dtype = torch.bfloat16, device = torch.cuda.current_device(),
        )

        self.gpu_kvcache_mgr.offload_kvcache(
            kvcache_metadata.kv_offload_handle,
            kvcache_metadata.offload_user_ids,
            kvcache_metadata.offload_page_ids,
            kvcache_metadata.gather_kv_gpu_buffer,
            kvcache_metadata.new_offload_startpos,
            kvcache_metadata.new_offload_lengths,
        )

    def get_kvcache_metadata_from_buffer(self, 
        batch_size,
        new_tokens,
        static_page_ids_gpu_buffer,
        static_offload_page_ids_gpu_buffer,
        offload_uids_buffer,
        metadata_host_buffer,
        metadata_gpu_buffer,
        static_onload_handle):
        # assert int(metadata_host_buffer[batch_size * 4 + 2]) == new_tokens
        return KVCacheMetadata(
            kv_indices = static_page_ids_gpu_buffer[: metadata_host_buffer[batch_size * 7 + 4]],
            kv_indptr = metadata_gpu_buffer[: batch_size + 1],
            kv_last_page_len = metadata_gpu_buffer[batch_size + 1 : batch_size * 2 + 1],
            total_history_lengths = metadata_gpu_buffer[batch_size * 2 + 1 : batch_size * 3 + 1],
            total_history_offsets = metadata_gpu_buffer[batch_size * 3 + 1 : batch_size * 4 + 2],
            batch_indices = metadata_gpu_buffer[batch_size * 5 + 4:batch_size * 5 + 4 + new_tokens],
            position = metadata_gpu_buffer[batch_size * 5 + 4 + new_tokens:batch_size * 5 + 4 + new_tokens*2],
            new_history_nnz = new_tokens,
            new_history_nnz_cuda = metadata_gpu_buffer[batch_size * 4 + 2 : batch_size * 4 + 3],
            kv_cache_table = self.cache_table_list,
            kv_onload_handle = static_onload_handle,
            kv_offload_handle = paged_kvcache_ops.KVOffloadHandle(),
            offload_user_ids = offload_uids_buffer[: metadata_host_buffer[batch_size * 7 + 6]],
            offload_page_ids = static_offload_page_ids_gpu_buffer[: int(metadata_host_buffer[batch_size * 7 + 5])].clone(),
            new_offload_startpos = metadata_host_buffer[batch_size * 5 + 4 : batch_size * 6 + 4],
            new_offload_lengths = metadata_host_buffer[batch_size * 6 + 4 : batch_size * 7 + 4],
            max_seqlen = torch.max(metadata_host_buffer[batch_size * 2 + 1 : batch_size * 3 + 1]).item(),
        )
    

    def strip_cached_tokens(self, batch, origin_num_cached):
        torch.cuda.nvtx.range_push("strip_cached_tokens")

        num_context = len(batch.contextual_feature_names)

        num_cached = torch.maximum(origin_num_cached - num_context, torch.tensor([0], dtype=torch.int32))
        num_cached_action = num_cached // 2
        num_cached_item = num_cached - num_cached_action
        num_hist_cached = torch.concat([num_cached_item, num_cached_action], dim=0)

        old_offsets = batch.features.offsets().cpu()
        old_lengths = batch.features.lengths().cpu()

        item_offset = num_context * batch.batch_size
        act_offset = item_offset + batch.batch_size

        new_lengths = torch.zeros_like(old_lengths)
        new_lengths[:item_offset] = torch.where(
            (origin_num_cached == 0).view(-1, batch.batch_size), 
            old_lengths[:item_offset].view(-1, batch.batch_size),
            new_lengths[:item_offset].view(-1, batch.batch_size)).view(-1)
        new_lengths[item_offset:] = old_lengths[item_offset:] - num_hist_cached

        startpos = old_offsets[item_offset : item_offset + 2 * batch.batch_size] + num_hist_cached
        endpos = old_offsets[item_offset + 1 :]

        old_values = batch.features.values()
        new_hist_value = [
            old_values[startpos[idx]:endpos[idx]] for idx in range(2*batch.batch_size)
        ]

        new_context_value = [
            old_values[idx: idx + 1] for idx in range(num_context*batch.batch_size) if int(new_lengths[idx]) > 0
        ]

        new_features = KeyedJaggedTensor(
            values = torch.cat(new_context_value + new_hist_value, dim = 0),
            lengths = new_lengths.cuda(),
            keys = batch.features.keys()
        )
        batch.features = new_features

        torch.cuda.nvtx.range_pop()
        return batch
