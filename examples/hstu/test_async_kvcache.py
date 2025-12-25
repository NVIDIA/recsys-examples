import torch

import math
from modules.async_kvcache_manager import AsyncHSTUKVCacheManager
import random
import time

def forward(
        async_kvcache,
        batch_size: int,
        user_ids: torch.Tensor,
        total_history_lengths: torch.Tensor,
    ):
        with torch.inference_mode():
            # print("[DEBUG] total_history_lengths", total_history_lengths)
            user_ids_list = user_ids.tolist()

            prepare_kvcache_result = async_kvcache.prepare_kvcache_async(
                batch_size,
                user_ids_list,
                total_history_lengths.tolist(),
                async_kvcache.static_page_ids_gpu_buffer,
                async_kvcache.static_offload_page_ids_gpu_buffer,
                async_kvcache.static_onload_handle,
            )
            # print("[DEBUG] return from trigger\n", flush=True)

            (
                old_cached_lengths,
                num_history_tokens,
                offload_uids_buffer,
                metadata_host_buffer,
                metadata_gpu_buffer,
                kvcache_metadata_fut,
                onload_fut,
            ) = prepare_kvcache_result
            # print("[DEBUG] old_cached_lengths", old_cached_lengths)
            old_cached_lengths = torch.tensor(old_cached_lengths, dtype=torch.int32)


            kvcache_metadata = async_kvcache.prepare_kvcache_wait(
                onload_fut,
                kvcache_metadata_fut,
                batch_size,
                num_history_tokens,
                async_kvcache.static_page_ids_gpu_buffer,
                async_kvcache.static_offload_page_ids_gpu_buffer,
                offload_uids_buffer,
                metadata_host_buffer,
                metadata_gpu_buffer,
                async_kvcache.static_onload_handle,
            )

            for layer_idx in range(async_kvcache.num_layers):
                kvcache_metadata.kv_onload_handle.wait_host(layer_idx)

            async_kvcache.offload_kvcache(kvcache_metadata)

            # async_kvcache.onload_kvcache_finalize(user_ids_list)

        return None

if __name__ == "__main__":
    with torch.inference_mode():

        max_batch_size = 4
        max_seq_len = 20000
        kwargs = {
            "num_layers": 3,
            "num_kv_heads": 4,
            "kv_headdim": 128,
            "num_tokens_per_page": 32,
            "num_primary_cache_pages": 10240,
            "num_onload_buffer_pages": math.ceil(max_batch_size * max_seq_len / 32),
            "num_reserved_buffer_pages": 0,
            "num_tokens_per_chunk": 2048,
            "max_num_sequences": -1,
            "max_sequence_length": max_seq_len,
            "max_batch_size": max_batch_size,
        }
        kvc_mgr = AsyncHSTUKVCacheManager(**kwargs)

        max_num_users = 50
        user_ids_pool = list(range(max_num_users))

        init_user_ids = list(user_ids_pool)
        random.shuffle(init_user_ids)

        torch.cuda.profiler.start()

        running_batch_size = 1
        for ind in range(0, len(init_user_ids), running_batch_size):
            batch_size = running_batch_size
            user_ids = init_user_ids[ind:ind+batch_size]
            total_history_lengths = [ 6000 for _ in user_ids ]

            user_ids = torch.tensor(user_ids, dtype=torch.int64)
            total_history_lengths = torch.tensor(total_history_lengths, dtype=torch.int32)

            print("forward", init_user_ids[ind:ind+batch_size])

            forward(kvc_mgr, batch_size, user_ids, total_history_lengths)
        
        while (kvc_mgr.gpu_kvcache_mgr.is_busy_offloading()):
            pass

        # torch.cuda.profiler.stop()

        # for uid in range(max_num_users):
        #     print(uid, kvc_mgr.gpu_kvcache_mgr.get_total_cache_length([uid]))
        
        kvc_mgr.gpu_kvcache_mgr.evict_all()

        # for uid in range(max_num_users):
        #     print(uid, kvc_mgr.gpu_kvcache_mgr.get_total_cache_length([uid]))

        running_batch_size = 1
        appending_user_ids = list(init_user_ids)
        random.shuffle(appending_user_ids)
        appending_user_ids = appending_user_ids[:10]

        # torch.cuda.profiler.start()

        for ind in range(0, len(appending_user_ids), running_batch_size):
            batch_size = running_batch_size
            user_ids = appending_user_ids[ind:ind+batch_size]
            total_history_lengths = [ 6200 for _ in user_ids ]

            user_ids = torch.tensor(user_ids, dtype=torch.int64)
            total_history_lengths = torch.tensor(total_history_lengths, dtype=torch.int32)

            forward(kvc_mgr, batch_size, user_ids, total_history_lengths)
        
        while (kvc_mgr.gpu_kvcache_mgr.is_busy_offloading()):
            pass

        torch.cuda.profiler.stop()

    print("Done")

# nsys profile -f true -o ./bench_async_init10x6000_append10x200_offload -c cudaProfilerApi --cuda-graph-trace=node python3 ./test_async_kvcache.py