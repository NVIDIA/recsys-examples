/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Implementation based on FlashInfer library.
# 
******************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <barrier>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <nvtx3/nvtx3.hpp>

#define cudaCheck(ans) { cudaSuccesAssert((ans), __FILE__, __LINE__); }
inline void cudaSuccesAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    //   if (abort) exit(code);
   }
}

template <typename DType, typename IdType>
cudaError_t AppendPagedKVCache(DType* k_data,
                               DType* v_data,
                               IdType* indices,
                               IdType* indptr,
                               uint32_t num_heads,
                               uint32_t head_dim,
                               uint32_t page_size,
                               uint32_t stride_page,
                               uint32_t stride_n,
                               uint32_t stride_h,
                               DType* append_key, DType* append_value, IdType* batch_indices, 
                               IdType* positions, IdType* offsets, 
                               IdType* nnz_cuda, uint32_t nnz, 
                               size_t append_k_stride_n, size_t append_k_stride_h,
                               size_t append_v_stride_n, size_t append_v_stride_h,
                               cudaStream_t stream);

template <typename DType, typename IdType>
cudaError_t GatherPagedKVCache(DType* gather_kv,
                               IdType* page_ids,
                               uint32_t num_heads,
                               uint32_t head_dim,
                               uint32_t page_size,
                               uint32_t stride_page,
                               uint32_t stride_k2v,
                               uint32_t stride_n,
                               uint32_t stride_h,
                               DType* kv_cache,
                               uint32_t nnz,
                               cudaStream_t stream);

template <typename DType, typename IdType>
cudaError_t GatherPagedKVCacheAllLayers(DType* gather_kv,
                                        IdType* page_ids,
                                        uint32_t num_layers,
                                        uint32_t stride_gather,
                                        uint32_t stride_layer,
                                        uint32_t num_heads,
                                        uint32_t head_dim,
                                        uint32_t page_size,
                                        uint32_t stride_page,
                                        uint32_t stride_k2v,
                                        uint32_t stride_n,
                                        uint32_t stride_h,
                                        DType* kv_cache,
                                        uint32_t nnz,
                                        cudaStream_t stream);

cudaError_t GetPagedBatchIndicesPositions(
  int32_t batch_size,
  int32_t* append_indptr,
  int32_t* seq_lens_ptr,
  int32_t* batch_indices_ptr,
  int32_t* positions_ptr,
  cudaStream_t stream
);

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor seqlen_offsets, 
                           at::Tensor nnz_cuda, unsigned int nnz,
                           at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           int64_t kv_layout) {
  // unsigned int batch_size = kv_last_page_len.size(0);
  auto device = append_key.device();

  unsigned int num_heads, page_size, head_dim;
  head_dim = paged_k_cache.size(3);
  if (kv_layout == 1) {
    num_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_heads = paged_k_cache.size(2);
  }

  auto stride_page = paged_k_cache.stride(0);
  auto stride_n = (kv_layout == 1) ? head_dim : num_heads * head_dim;
  auto stride_h = (kv_layout == 1) ? page_size * head_dim : head_dim;

  // get kv_cache_strides
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");


  auto append_k_strides = append_key.strides();
  auto append_k_stride_n = append_k_strides[0];
  auto append_k_stride_h = append_k_strides[1];
  auto append_v_strides = append_value.strides();
  auto append_v_stride_n = append_v_strides[0];
  auto append_v_stride_h = append_v_strides[1];

  auto kv_scalar_dtype = paged_k_cache.scalar_type();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaError_t status;
  switch (kv_scalar_dtype) {
    case at::ScalarType::BFloat16:
        status =
        AppendPagedKVCache(static_cast<nv_bfloat16*>(paged_k_cache.data_ptr()),
                           static_cast<nv_bfloat16*>(paged_v_cache.data_ptr()),
                           static_cast<int32_t*>(kv_indices.data_ptr()),
                           static_cast<int32_t*>(kv_indptr.data_ptr()),
                           num_heads, head_dim, page_size, stride_page, stride_n, stride_h,
                           static_cast<nv_bfloat16*>(append_key.data_ptr()),
                           static_cast<nv_bfloat16*>(append_value.data_ptr()),
                           static_cast<int32_t*>(batch_indices.data_ptr()),
                           static_cast<int32_t*>(positions.data_ptr()), 
                           static_cast<int32_t*>(seqlen_offsets.data_ptr()), 
                           static_cast<int32_t*>(nnz_cuda.data_ptr()), 
                           nnz, append_k_stride_n, append_k_stride_h, 
                           append_v_stride_n, append_v_stride_h, stream);
        break;
    case at::ScalarType::Half:
        status =
        AppendPagedKVCache(static_cast<nv_half*>(paged_k_cache.data_ptr()), 
                           static_cast<nv_half*>(paged_v_cache.data_ptr()),
                           static_cast<int32_t*>(kv_indices.data_ptr()),
                           static_cast<int32_t*>(kv_indptr.data_ptr()),
                           num_heads, head_dim, page_size, stride_page, stride_n, stride_h,
                           static_cast<nv_half*>(append_key.data_ptr()),
                           static_cast<nv_half*>(append_value.data_ptr()),
                           static_cast<int32_t*>(batch_indices.data_ptr()),
                           static_cast<int32_t*>(positions.data_ptr()), 
                           static_cast<int32_t*>(seqlen_offsets.data_ptr()), 
                           static_cast<int32_t*>(nnz_cuda.data_ptr()), 
                           nnz, append_k_stride_n, append_k_stride_h, 
                           append_v_stride_n, append_v_stride_h, stream);
        break;
    default:
        TORCH_CHECK(false, "AppendPagedKVCache failed to dispatch with dtype ", kv_scalar_dtype);
  }
  TORCH_CHECK(status == cudaSuccess,
              "AppendPagedKVCache failed with error: ", cudaGetErrorString(status));
}

void gather_paged_kv_cache(at::Tensor gather_kv_gpu_buffer,
                           at::Tensor paged_kv_cache,
                           at::Tensor page_ids_to_offload,
                           unsigned int num_pages,
                           int64_t kv_layout) {
  auto device = paged_kv_cache.device();

  TORCH_CHECK(paged_kv_cache.ndimension() == 5, 
              "kv cache table must has 5 dimensions (num_pages, 2, page_size, num_head, head_dim).");
  
  unsigned int num_heads, page_size, head_dim;
  head_dim = paged_kv_cache.size(4);
  if (kv_layout == 1) {
    num_heads = paged_kv_cache.size(2);
    page_size = paged_kv_cache.size(3);
  } else {
    page_size = paged_kv_cache.size(2);
    num_heads = paged_kv_cache.size(3);
  }

  auto stride_page = paged_kv_cache.stride(0);
  auto stride_n = (kv_layout == 1) ? head_dim : num_heads * head_dim;
  auto stride_h = (kv_layout == 1) ? page_size * head_dim : head_dim;
  auto stride_k2v = paged_kv_cache.stride(1);

  // check input/output strides
  TORCH_CHECK(paged_kv_cache.strides() == gather_kv_gpu_buffer.strides(), 
              "input/output strides must be identical");
  TORCH_CHECK(paged_kv_cache.is_contiguous() && paged_kv_cache.is_contiguous(), 
              "buffer must be contiguous");
  
  auto kv_scalar_dtype = paged_kv_cache.scalar_type();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaError_t status;
  switch (kv_scalar_dtype) {
    case at::ScalarType::BFloat16:
        status = GatherPagedKVCache(
            static_cast<nv_bfloat16*>(gather_kv_gpu_buffer.data_ptr()),
            static_cast<int32_t*>(page_ids_to_offload.data_ptr()),
            num_heads, head_dim, page_size, 
            stride_page, stride_k2v, stride_n, stride_h,
            static_cast<nv_bfloat16*>(paged_kv_cache.data_ptr()),
            num_pages * page_size, stream);
        break;
    case at::ScalarType::Half:
        status = GatherPagedKVCache(
            static_cast<nv_half*>(gather_kv_gpu_buffer.data_ptr()),
            static_cast<int32_t*>(page_ids_to_offload.data_ptr()),
            num_heads, head_dim, page_size, 
            stride_page, stride_k2v, stride_n, stride_h,
            static_cast<nv_half*>(paged_kv_cache.data_ptr()),
            num_pages * page_size, stream);
        break;
    default:
        TORCH_CHECK(false, "GatherPagedKVCache failed to dispatch with dtype ", kv_scalar_dtype);
  }
  TORCH_CHECK(status == cudaSuccess,
              "GatherPagedKVCache failed with error: ", cudaGetErrorString(status));
}

void gather_paged_kv_cache_all_layers(uint16_t *gather_kv_gpu_buffer,
                                      uint16_t *paged_kv_cache,
                                      int *page_ids_to_offload,
                                      uint32_t num_layers,
                                      uint32_t stride_gather,
                                      uint32_t stride_layer,
                                      uint32_t num_heads,
                                      uint32_t head_dim,
                                      uint32_t page_size,
                                      uint32_t stride_page,
                                      uint32_t stride_k2v,
                                      uint32_t stride_n,
                                      uint32_t stride_h,
                                      uint32_t num_pages,
                                      cudaStream_t stream) {
  // auto device = paged_kv_cache.device();
  // const c10::cuda::OptionalCUDAGuard device_guard(device);

  cudaError_t status;
  status = GatherPagedKVCacheAllLayers(
      reinterpret_cast<nv_bfloat16*>(gather_kv_gpu_buffer),
      static_cast<int32_t*>(page_ids_to_offload),
      num_layers, stride_gather, stride_layer, 
      num_heads, head_dim, page_size, 
      stride_page, stride_k2v, stride_n, stride_h,
      reinterpret_cast<nv_bfloat16*>(paged_kv_cache),
      num_pages * page_size, stream);
  TORCH_CHECK(status == cudaSuccess,
              "GatherPagedKVCacheAllLayers failed with error: ", cudaGetErrorString(status));
}

namespace kvcache {

class HostKVStorageImpl
{
public:
    HostKVStorageImpl(
        int num_layers,
        int num_kv_heads,
        int kv_headdim,
        int num_tokens_per_page,
        int64_t num_tokens_per_chunk
    )
        : num_layers(num_layers)
        , num_kv_heads(num_kv_heads)
        , kv_headdim(kv_headdim)
        , page_size(num_tokens_per_page)
        , chunk_size(num_tokens_per_chunk)
        , _uid_to_chunk_id(num_layers, std::unordered_map<int64_t, std::vector<uintptr_t>>())
    {
        this->chunk_numel = num_tokens_per_chunk * 2 * num_kv_heads * kv_headdim;
        this->page_numel = 2 * page_size * num_kv_heads * kv_headdim;
        this->per_token_numel = 2 * num_kv_heads * kv_headdim;
    };

    ~HostKVStorageImpl()
    {}

    int64_t get_kvdata_length(int64_t user_id) {
        auto it = _uid_to_length.find(user_id);
        if (it ==  _uid_to_length.end()) return 0;
        return it->second;
    };

    void append_kvdata_v2(int64_t user_id, int64_t start_position, int64_t length, uint16_t *pinned_input_ptr, size_t gather_layer_stride) {
        assert(length % this->chunk_size == 0);
        if (start_position != 0) {
            assert(_uid_to_length[user_id] == start_position);
        }
        else {
            assert(_uid_to_length.find(user_id) == _uid_to_length.end());
            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
                _uid_to_chunk_id[layer_idx][user_id] = std::vector<uintptr_t>();
            _uid_to_mempool[user_id] = std::vector<uintptr_t>();
        }

        size_t num_chunks = length / chunk_size;
        size_t num_elem = length * per_token_numel;
        size_t kvdata_size = num_elem * sizeof(uint16_t);

        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            uint16_t* src_ptr = pinned_input_ptr + layer_idx * gather_layer_stride;
            
            for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx ++) {
                _uid_to_chunk_id[layer_idx][user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr + chunk_idx * this->chunk_numel));
            }

            _uid_to_mempool[user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr));
        }
        _uid_to_length[user_id] = start_position + length;
    };

    std::vector<uint16_t*> get_kvdata_v2(int64_t user_id, int64_t length, int64_t layer_idx) {
        // int64_t offloaded_length = get_kvdata_length(user_id);
        // assert(offloaded_length >= length);

        std::vector<uint16_t*> chunk_ptrs;
        if (length == 0) {
            return chunk_ptrs;
        }
        // assert(length % this->chunk_size == 0);
        size_t num_chunks = length / chunk_size;
        const size_t chunk_bytesize = this->chunk_numel * sizeof(uint16_t);
        const auto &chunk_ptr_list = _uid_to_chunk_id[layer_idx][user_id];
        // assert(chunk_ptr_list.size() >= num_chunks);
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx ++) {
            uint16_t* src_ptr = reinterpret_cast<uint16_t*>(chunk_ptr_list[chunk_idx]);
            chunk_ptrs.push_back(src_ptr);
        }
        return chunk_ptrs;
    };

public:
    std::vector<std::unordered_map<int64_t, std::vector<uintptr_t>>> _uid_to_chunk_id;
    std::unordered_map<int64_t, int64_t> _uid_to_length;
    std::unordered_map<int64_t, std::vector<uintptr_t>> _uid_to_mempool;

    const int num_layers;
    const int num_kv_heads;
    const int kv_headdim;
    const int page_size;

    const int64_t chunk_size;
    size_t chunk_numel;
    size_t page_numel;
    size_t per_token_numel;
    size_t layer_numel;
};

// class PinnedDoubleBuffer {
// public:
// public:
// };

class KVOnloadHandle {
public:
    KVOnloadHandle(
        int num_layers
    )
    : num_layers(num_layers)
    , event(std::vector<cudaEvent_t>(num_layers))
    , host_complete(num_layers, 0) {
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
            cudaEventCreate(&event[layer_idx]);
        }
    };

    ~KVOnloadHandle(){
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
            cudaEventDestroy(event[layer_idx]);
        }
    };

    // void record(int layer_idx, cudaStream_t stream) {
    //     cudaEventRecord(event[layer_idx], stream);
    // };

    // void wait(int layer_idx) {
    //     auto stream = at::cuda::getCurrentCUDAStream();
    //     cudaStreamWaitEvent(stream, event[layer_idx], 0);
    // };
    void reset() {
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
            host_complete[layer_idx] = 0;
        }
    }

    void complete_host(int layer_idx, cudaStream_t stream) {
        cudaEventRecord(event[layer_idx], stream);
        {
            std::unique_lock<std::mutex> lock(mtx_);
            host_complete[layer_idx] = 1;
        }
        cv_.notify_one();
    };

    void wait_host(int layer_idx) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this, layer_idx](){ return host_complete[layer_idx] == 1; });
        }
        auto stream = at::cuda::getCurrentCUDAStream();
        cudaStreamWaitEvent(stream, event[layer_idx], 0);
    };

    void wait_all(void);
public:
    int num_layers;
    std::vector<cudaEvent_t> event;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<int> host_complete;
};

class KVOffloadHandle {
public:

void record_ready(void) {
        auto stream = at::cuda::getCurrentCUDAStream();
        cudaEventCreate(&ready_event);
        cudaEventRecord(ready_event, stream);
    };

    // void wait_ready(cudaStream_t stream) {
    //     cudaStreamWaitEvent(stream, ready_event);
    // };

public:
    cudaEvent_t ready_event;
};

class GPUKVCacheMangerImpl
{
public:
    GPUKVCacheMangerImpl(
        int num_layers,
        int num_kv_heads,
        int kv_headdim,
        int num_tokens_per_page,
        int num_primary_cache_pages,
        int num_onload_buffer_pages,
        int num_reserved_buffer_pages,
        int num_tokens_per_chunk,
        int max_num_sequences,
        int max_sequence_length,
        at::Tensor cache_table_tensor,
        HostKVStorageImpl& host_kv_mgr)
     : num_layers(num_layers)
     , num_kv_heads(num_kv_heads)
     , kv_headdim(kv_headdim)
     , num_tokens_per_page(num_tokens_per_page)
     , num_primary_cache_pages(num_primary_cache_pages)
     , num_onload_buffer_pages(num_onload_buffer_pages)
     , num_reserved_buffer_pages(num_reserved_buffer_pages)
     , num_tokens_per_chunk(num_tokens_per_chunk)
     , max_num_sequences(max_num_sequences)
     , max_sequence_length(max_sequence_length)
     , cache_table(static_cast<uint16_t*>(cache_table_tensor.data_ptr()))
     , device(cache_table_tensor.device())
     , onload_pinned_buffers(2)
     , onload_memcpy_event(2)
     , offload_pinned_buffers(2)
     , offload_memcpy_event(2)
     , onload_memcpy_barrier_(std::barrier<>(3 + 1))
    {
        const c10::cuda::OptionalCUDAGuard device_guard(this->device);

        for (int page_id = 0; page_id < num_primary_cache_pages; page_id++)
            _empty_pages.push(page_id);
    
        page_stride = 2 * num_tokens_per_page * num_kv_heads * kv_headdim;
        k2v_stride = num_tokens_per_page * num_kv_heads * kv_headdim;
        layer_stride = (num_primary_cache_pages + num_onload_buffer_pages) * page_stride;

        per_token_kv_stride = 2 * num_kv_heads * kv_headdim;

        cudaStreamCreate(&worker_stream);
        cudaStreamCreate(&onload_stream);
        cudaStreamCreate(&offload_stream);

        this->host_kv_mgr = &host_kv_mgr;

        for (int i = 0; i < 2; i++) {
            cudaMallocHost((void**)&onload_pinned_buffers[i], host_kv_mgr.chunk_numel * sizeof(uint16_t));
            cudaEventCreate(&onload_memcpy_event[i]);
        }

        for (int i = 0; i < 2; i++) {
            cudaMallocHost((void**)&offload_pinned_buffers[i], host_kv_mgr.chunk_numel * sizeof(uint16_t));
            cudaEventCreate(&offload_memcpy_event[i]);
        }

        this->terminate_ = false;
        this->num_onload_memcpy_worker = 3;
        for (int i = 0; i < num_onload_memcpy_worker; i++) {
            this->onload_memcpy_worker.emplace_back(std::thread(
                &GPUKVCacheMangerImpl::onload_host_memcpy_loop, this 
            ));
        }
        // ;

        this->num_offload_memcpy_worker = 3;
        for (int i = 0; i < num_offload_memcpy_worker; i++) {
            this->offload_memcpy_worker.emplace_back(std::thread(
                &GPUKVCacheMangerImpl::offload_host_memcpy_loop, this 
            ));
        }

        this->queued_offload_tokens = 0;
        this->queued_offload_limits = num_reserved_buffer_pages * num_tokens_per_page;
        this->offload_busy_.store(false);
        this->offload_worker = std::thread(&GPUKVCacheMangerImpl::offload_loop, this);

    };

    ~GPUKVCacheMangerImpl() {
        {
            std::unique_lock<std::mutex> lock(onload_memcpy_task_mtx_);
            std::unique_lock<std::mutex> lock2(offload_memcpy_task_mtx_);
            std::unique_lock<std::mutex> lock3(offload_task_mutex_);
            this->terminate_ = true;
        }
        onload_memcpy_task_cv_.notify_all();
        offload_memcpy_task_cv_.notify_all();
        offload_task_cv_.notify_all();

        for (int i = 0; i < num_onload_memcpy_worker; i++) {
            this->onload_memcpy_worker[i].join();
        }

        for (int i = 0; i < num_offload_memcpy_worker; i++) {
            this->offload_memcpy_worker[i].join();
        }

        this->offload_worker.join();

        for (int i = 0; i < 2; i++) {
            cudaFree(onload_pinned_buffers[i]);
            cudaFree(offload_pinned_buffers[i]);
            cudaEventDestroy(onload_memcpy_event[i]);
            cudaEventDestroy(offload_memcpy_event[i]);
        }
    }

    int64_t getUIdToEvict(std::unordered_set<int64_t> extra_freezed_uids) {
        while (true) {
            int num_offloading_uids = 0;
            {
                std::unique_lock<std::mutex> lock(offload_freezed_uids_mtx_);
                num_offloading_uids = offload_freezed_uids_.size();
                // std::cout << "Saw " << num_offloading_uids << " freezed for offloading" << std::endl;

                for (auto it = std::rbegin(_lru_list); it != std::rend(_lru_list); ++it) {
                    if (offload_freezed_uids_.find((int64_t)*it) != offload_freezed_uids_.end())
                        continue;
                    if (extra_freezed_uids.find((int64_t)*it) != extra_freezed_uids.end())
                        continue;
                    return *it;
                }
            }
            if (num_offloading_uids == 0) assert(false);
            
            std::this_thread::yield();
        }

        return _lru_list.back();
    };

    std::vector<int32_t>& alloc(int64_t uid, int new_total_length, std::unordered_set<int64_t> freezed_uids) {
        int cur_cached_start = 0;
        int cur_cached_len = 0;
        // int padding_last_page = 0;

        bool found_in_cache = retain(uid);
        if (found_in_cache) {
            cur_cached_start = _uid_to_paged_cache_startpos[uid];
            cur_cached_len = _uid_to_paged_cache_length[uid];
        } else {
            _uid_to_page_id[uid] = std::vector<int32_t>();
            if (_uid_to_offloaded_length.find(uid) != _uid_to_offloaded_length.end()) {
                _uid_to_paged_cache_startpos[uid] = _uid_to_offloaded_length[uid];
                cur_cached_start = _uid_to_offloaded_length[uid];
            }
            else {
                _uid_to_paged_cache_startpos[uid] = 0;
            }
        }

        int new_cached_len = new_total_length - cur_cached_start;
        int cur_num_pages = (cur_cached_len + num_tokens_per_page - 1) / num_tokens_per_page;
        int new_num_pages = (new_cached_len + num_tokens_per_page - 1) / num_tokens_per_page;

        int num_append_pages = new_num_pages - cur_num_pages;
        // std::cout << " *** " << cur_num_pages << " " << cur_cached_len << std::endl;
        // std::cout << " *** " << new_num_pages << " " << new_cached_len << std::endl;

        while ((size_t)num_append_pages > _empty_pages.size()) {
            int64_t uid_to_evict = getUIdToEvict(freezed_uids);
            evict(uid_to_evict);
            // std::cout << "evict " << uid_to_evict << std::endl;
        }

        std::vector<int32_t>& page_ids = _uid_to_page_id[uid];
        // for (auto pid : page_ids) {
        //     // std::cout << " - " << pid << std::endl;
        // }
        for (int i = 0; i < num_append_pages; i++) {
            page_ids.push_back(_empty_pages.front());
            _empty_pages.pop();
        }
        _uid_to_paged_cache_length[uid] = new_cached_len;

        return page_ids;
    };

    std::vector<int32_t> get_total_cache_length(std::vector<int64_t>& uids) {
        int batch_size = uids.size();
        std::vector<int32_t> total_cached_lengths(batch_size);
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            int64_t uid = uids[seq_idx];
            if (_uid_to_paged_cache_startpos.find(uid) != _uid_to_paged_cache_startpos.end()) {
                total_cached_lengths[seq_idx] = _uid_to_paged_cache_startpos[uid] + _uid_to_paged_cache_length[uid];
            } else if (_uid_to_offloaded_length.find(uid) != _uid_to_offloaded_length.end())
                total_cached_lengths[seq_idx] = _uid_to_offloaded_length[uid];
            else {
                total_cached_lengths[seq_idx] = 0;
            }
        }
        return total_cached_lengths;
    };
    
    void evict(int64_t uid)
    {
        auto const tableIt = _lru_lookup_table.find(uid);
        assert(_lru_lookup_table.end() != tableIt);
        // if (_lru_lookup_table.end() != tableIt) {
            _lru_list.erase(tableIt->second);
            _lru_lookup_table.erase(tableIt);
            // assert(_uid_to_page_id[uid].size() > 0);

            for (auto page_id : _uid_to_page_id[uid]) {
                _empty_pages.push(page_id);
            }

            _uid_to_page_id.erase(uid);
            _uid_to_paged_cache_startpos.erase(uid);
            _uid_to_paged_cache_length.erase(uid);
        // }
    };

    void evict_all()
    {
        std::queue<int64_t> empty_pages;
        std::swap(_empty_pages, empty_pages);
        _lru_list.clear();
        _lru_lookup_table.clear();
        _uid_to_page_id.clear();
        _uid_to_paged_cache_startpos.clear();
        _uid_to_paged_cache_length.clear();

        for (int page_id = 0; page_id < this->num_primary_cache_pages; page_id++)
            _empty_pages.push(page_id);
    };

    void invalid(int64_t uid) {
        auto const tableIt = _lru_lookup_table.find(uid);
        if (_lru_lookup_table.end() != tableIt) {
            _lru_list.erase(tableIt->second);
            _lru_lookup_table.erase(tableIt);

            for (auto page_id : _uid_to_page_id[uid]) {
                _empty_pages.push(page_id);
            }

            _uid_to_page_id.erase(uid);
            _uid_to_paged_cache_startpos.erase(uid);
            _uid_to_paged_cache_length.erase(uid);
            _uid_to_offloaded_length.erase(uid);
        }
    };

    bool retain(int64_t uid)
    {
        auto const tableIt = _lru_lookup_table.find(uid);
        bool found = (_lru_lookup_table.end() != tableIt);
        if (found) {
            _lru_list.erase(tableIt->second);
        }
        _lru_list.push_front(uid);
        _lru_lookup_table[uid] = _lru_list.begin();
        return found;
    };

    uint16_t *get_cache_table(void) {
        return cache_table;
    };

    uint16_t *get_cache_table_by_layer(int layer_idx) {
        return cache_table + layer_idx * layer_stride;
    };

public:
    void onload_kvcache(
        std::vector<int64_t>& user_ids, 
        KVOnloadHandle& onloadhandle) {
        const c10::cuda::OptionalCUDAGuard device_guard(this->device);

        // std::cout << "onload_kvcache start" << std::endl << std::flush;
        const int batch_size = user_ids.size();

        std::vector<size_t> onload_length(batch_size);
        std::vector<size_t> onload_offsets(batch_size + 1);
        onload_offsets[0] = 0;
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            auto uid = user_ids[seq_idx];
            if (this->_uid_to_paged_cache_startpos.find(uid) != this->_uid_to_paged_cache_startpos.end())
                onload_length[seq_idx] = this->_uid_to_paged_cache_startpos[uid];
            else if (this->_uid_to_offloaded_length.find(uid) != this->_uid_to_offloaded_length.end())
                onload_length[seq_idx] = this->_uid_to_offloaded_length[uid];
            else
                onload_length[seq_idx] = 0;

            onload_offsets[seq_idx + 1] = onload_offsets[seq_idx] + onload_length[seq_idx];
        }
        size_t total_onload_length = onload_offsets[batch_size];
        if (total_onload_length == 0) {
            for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++)
                onloadhandle.complete_host(layer_idx, this->onload_stream);
            // std::cout << "onload_kvcache empty" << std::endl << std::flush;
            return;
        }
        // std::cout << "[Onload Launch] {" << user_ids[0] << "}: " << onload_length[0] << std::endl;

        const size_t chunk_numel_part = host_kv_mgr->chunk_numel / (this->num_onload_memcpy_worker + 1);

        int task_idx = 0;
        for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
            uint16_t *gpu_onload_buffer = this->get_cache_table_by_layer(layer_idx) + this->num_primary_cache_pages * this->page_stride;

            for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
                std::vector<uint16_t *> chunk_ptrs = host_kv_mgr->get_kvdata_v2(user_ids[0], onload_length[0], layer_idx);
                
                // std::cout << "[Onload] uid: " << user_ids[seq_idx] << " - " << chunk_ptrs.size() << std::endl;
                // std::cout << "\t" << host_kv_mgr->chunk_numel << " - " << host_kv_mgr->chunk_size * this->per_token_kv_stride << std::endl;
                for (int chunk_idx = 0; chunk_idx < chunk_ptrs.size(); chunk_idx++) {
                    // std::cout << "\t" << reinterpret_cast<uintptr_t>(chunk_ptrs[chunk_idx]) << " - " << (chunk_ptrs[chunk_idx] - chunk_ptrs[0]) << std::endl;

                    onload_host_memcpy(onload_pinned_buffers[task_idx%2], chunk_ptrs[chunk_idx], host_kv_mgr->chunk_numel * sizeof(uint16_t), chunk_numel_part * sizeof(uint16_t));
                    
                    cudaCheck(cudaMemcpyAsync(gpu_onload_buffer + onload_offsets[seq_idx] * this->per_token_kv_stride + chunk_idx * host_kv_mgr->chunk_numel, 
                        onload_pinned_buffers[task_idx%2], host_kv_mgr->chunk_numel * sizeof(uint16_t), cudaMemcpyHostToDevice, this->onload_stream));
                    cudaCheck(cudaEventRecord(onload_memcpy_event[task_idx%2], this->onload_stream));

                    if (task_idx > 0) {
                        cudaCheck(cudaEventSynchronize(onload_memcpy_event[(task_idx - 1)%2]));
                    }

                    task_idx++;
                }
            }

            onloadhandle.complete_host(layer_idx, this->onload_stream);
        }
        // std::cout << "onload_kvcache end" << std::endl << std::flush;
    };

    void offload_kvcache(
        KVOffloadHandle& offload_handle,
        at::Tensor offload_user_ids,      // host
        at::Tensor offload_page_ids,      // gpu
        at::Tensor gather_kv_gpu_buffer,  // gpu
        at::Tensor new_offload_startpos,  // host
        at::Tensor new_offload_lengths)   // host
    {
        const size_t num_offload_uids = offload_user_ids.numel();
        {
            std::unique_lock<std::mutex> lock(queued_offload_lastpos_mutex_);
            if (queued_offload_tokens >= queued_offload_limits) {
                return;
            }
            for (auto seq_idx = 0; seq_idx < num_offload_uids; seq_idx++) {
                queued_offload_tokens += ((int*)new_offload_lengths.data_ptr())[seq_idx];
            }
        }

        std::vector<int> offload_host_metadata(4*num_offload_uids);

        std::memcpy((void*)offload_host_metadata.data(), 
                    (void*)offload_user_ids.data_ptr(), num_offload_uids * sizeof(int64_t));
        std::memcpy(offload_host_metadata.data() + num_offload_uids * 2, 
                    new_offload_startpos.data_ptr(), num_offload_uids * sizeof(int));
        std::memcpy(offload_host_metadata.data() + num_offload_uids * 3, 
                    new_offload_lengths.data_ptr(), num_offload_uids * sizeof(int));
        
        {
            std::unique_lock<std::mutex> lock(offload_freezed_uids_mtx_);
            int64_t *offload_uids = reinterpret_cast<int64_t*>(offload_host_metadata.data());
            for (int idx = 0; idx < num_offload_uids; idx++) {
                int cur_freezed_times = offload_freezed_uids_[offload_uids[idx]];
                offload_freezed_uids_[offload_uids[idx]] = cur_freezed_times + 1;
                // std::cout << "Freezing " << offload_uids[idx] << " from " << cur_freezed_times << " to " << (cur_freezed_times + 1) << std::endl;
            }
        }
        {
            std::unique_lock<std::mutex> lock(offload_task_mutex_);
            offload_task_queue.push(std::make_tuple(
                offload_host_metadata,
                offload_page_ids, 
                gather_kv_gpu_buffer, 
                offload_handle.ready_event
            ));
        }

        offload_task_cv_.notify_one();
    };

    bool is_busy_offloading() {
        return !offload_task_queue.empty() || this->offload_busy_.load();
    }

private:
    void onload_host_memcpy(void* dst, void* src, size_t bytes, size_t bytes_part) {
        NVTX3_FUNC_RANGE();
        {
            std::unique_lock<std::mutex> lock(onload_memcpy_task_mtx_);
            // this->onload_memcpy_cmplt_flag = 0;
            for (int i = 1; i < (this->num_onload_memcpy_worker + 1); i++)
                onload_memcpy_task_queue.push(std::make_tuple(
                    (reinterpret_cast<char *>(dst) + i * bytes_part),
                    (reinterpret_cast<char *>(src) + i * bytes_part),
                    bytes_part
                ));
        }
        onload_memcpy_task_cv_.notify_all();
        std::memcpy(dst, src, bytes_part);

        // {
        //     std::unique_lock<std::mutex> lock(onload_memcpy_cmplt_mtx_);
        //     onload_memcpy_cmplt_cv_.wait(lock, [this] {
        //         return this->onload_memcpy_cmplt_flag == this->num_onload_memcpy_worker || this->terminate_;
        //     });
        // }
        onload_memcpy_barrier_.arrive_and_wait();
    }

    void onload_host_memcpy_loop() {
        while (true) {
            void *dst, *src;
            size_t bytes;
            {
                std::unique_lock<std::mutex> lock(onload_memcpy_task_mtx_);
                onload_memcpy_task_cv_.wait(lock, [this]{
                    return !this->onload_memcpy_task_queue.empty() || this->terminate_;
                });
                if (terminate_) return;
                std::tie(dst, src, bytes) = onload_memcpy_task_queue.front();
                onload_memcpy_task_queue.pop();
            }
            std::memcpy(dst, src, bytes);
            // {
            //     std::unique_lock<std::mutex> lock(onload_memcpy_cmplt_mtx_);
            //     onload_memcpy_cmplt_flag += 1;
            //     if (onload_memcpy_cmplt_flag == this->num_onload_memcpy_worker)
            //         onload_memcpy_cmplt_cv_.notify_one();
            // }
            onload_memcpy_barrier_.arrive_and_wait();
        }
    }

    void offload_host_memcpy(void* dst, void* src, size_t bytes, size_t bytes_part) {
        NVTX3_FUNC_RANGE();
        {
            std::unique_lock<std::mutex> lock(offload_memcpy_task_mtx_);
            this->offload_memcpy_cmplt_flag = 0;
            for (int i = 1; i < (this->num_offload_memcpy_worker + 1); i++)
                offload_memcpy_task_queue.push(std::make_tuple(
                    (reinterpret_cast<char *>(dst) + i * bytes_part),
                    (reinterpret_cast<char *>(src) + i * bytes_part),
                    bytes_part
                ));
        }
        offload_memcpy_task_cv_.notify_all();
        std::memcpy(dst, src, bytes_part);

        {
            std::unique_lock<std::mutex> lock(offload_memcpy_cmplt_mtx_);
            offload_memcpy_cmplt_cv_.wait(lock, [this] {
                return this->offload_memcpy_cmplt_flag == this->num_offload_memcpy_worker || this->terminate_;
            });
        }
    }

    void offload_host_memcpy_loop() {
        while (true) {
            void *dst, *src;
            size_t bytes;
            {
                std::unique_lock<std::mutex> lock(offload_memcpy_task_mtx_);
                offload_memcpy_task_cv_.wait(lock, [this]{
                    return !this->offload_memcpy_task_queue.empty() || this->terminate_;
                });
                if (terminate_) return;
                std::tie(dst, src, bytes) = offload_memcpy_task_queue.front();
                offload_memcpy_task_queue.pop();
            }
            std::memcpy(dst, src, bytes);
            {
                std::unique_lock<std::mutex> lock(offload_memcpy_cmplt_mtx_);
                offload_memcpy_cmplt_flag += 1;
                if (offload_memcpy_cmplt_flag == this->num_offload_memcpy_worker)
                    offload_memcpy_cmplt_cv_.notify_one();
            }
        }
    }

    void offload_loop()
    {
        const c10::cuda::OptionalCUDAGuard device_guard(this->device);

        while (true) {
            std::vector<int> host_metadata;
            at::Tensor offload_page_ids, gather_kv_gpu_buffer;
            cudaEvent_t offload_gpu_acq_event;
            {
                nvtx3::scoped_range r{"offload_prelogue"};

                std::unique_lock<std::mutex> lock(offload_task_mutex_);
                offload_task_cv_.wait(lock, [this] {
                    return !offload_task_queue.empty() || this->terminate_;
                });
                if (this->terminate_) {
                    break;
                }

                {
                    nvtx3::scoped_range r1{"offload_prelogue unpack_input"};

                    std::tie(
                        host_metadata, offload_page_ids, gather_kv_gpu_buffer, offload_gpu_acq_event
                    ) = offload_task_queue.front();
                }
                {
                    nvtx3::scoped_range r2{"offload_prelogue pop"};
                    offload_task_queue.pop();
                }

                this->offload_busy_.store(true);
            }

            int64_t *offload_uids = reinterpret_cast<int64_t *>(host_metadata.data());
            uint16_t *gather_kv_gpu_buffer_data_ptr = static_cast<uint16_t*>(gather_kv_gpu_buffer.data_ptr());
            const int num_offload_uids = host_metadata.size() / 4;
            const int num_offload_pages = offload_page_ids.numel();
            size_t gather_layer_stride = num_offload_pages * this->page_stride;

            int64_t dbg_uid;
            {
                int64_t *offload_uids = reinterpret_cast<int64_t *>(host_metadata.data());
                dbg_uid = offload_uids[0];
            }
            // std::cout << "[Offload Launch] {" << dbg_uid << "}: waiting event" << std::endl;
            cudaStreamWaitEvent(this->offload_stream, offload_gpu_acq_event);
            cudaEventDestroy(offload_gpu_acq_event);
            // std::cout << "[Offload Launch] {" << dbg_uid << "}: waited event" << std::endl;

            // gather
            gather_paged_kv_cache_all_layers(
                gather_kv_gpu_buffer_data_ptr,
                this->get_cache_table(),
                static_cast<int*>(offload_page_ids.data_ptr()),
                this->num_layers,
                gather_layer_stride,
                this->layer_stride,
                this->num_kv_heads,
                this->kv_headdim,
                this->num_tokens_per_page,
                this->page_stride,
                this->k2v_stride,
                this->num_kv_heads * this->kv_headdim,
                this->kv_headdim,
                num_offload_pages,
                this->offload_stream);
            cudaStreamSynchronize(this->offload_stream);
            // release on gpu kvcache
            {
                std::unique_lock<std::mutex> lock(offload_freezed_uids_mtx_);
                for (int idx = 0; idx < num_offload_uids; idx++) {
                    int cur_freezed_times = offload_freezed_uids_[offload_uids[idx]];
                    if (cur_freezed_times == 1) {
                        offload_freezed_uids_.erase(offload_uids[idx]);
                    } else {
                        offload_freezed_uids_[offload_uids[idx]] = cur_freezed_times - 1;
                    }
                    // std::cout << "Released " << offload_uids[idx] << " from " << cur_freezed_times << " to " << (cur_freezed_times - 1) << std::endl;
                }
            }
            // std::cout << "[Offload Launch] {" << dbg_uid << "}: gathered gpu" << std::endl;
            // skipped

            size_t pinned_bytes = this->num_layers * gather_layer_stride * sizeof(uint16_t);
            uint16_t *host_kv_ptr = static_cast<uint16_t *>(aligned_alloc(sysconf(_SC_PAGESIZE), pinned_bytes));

            size_t chunk_numel_part = host_kv_mgr->chunk_numel / (this->num_offload_memcpy_worker + 1);

            int num_chunks = (num_offload_pages * this->num_tokens_per_page) / this->num_tokens_per_chunk * this->num_layers;
            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                cudaMemcpyAsync(
                    offload_pinned_buffers[chunk_idx%2],
                    gather_kv_gpu_buffer_data_ptr + chunk_idx * host_kv_mgr->chunk_numel,
                    host_kv_mgr->chunk_numel * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost, 
                    this->offload_stream);
                cudaEventRecord(offload_memcpy_event[chunk_idx%2], this->offload_stream);

                if (chunk_idx > 0) {
                    cudaEventSynchronize(offload_memcpy_event[(chunk_idx-1)%2]);
                    offload_host_memcpy(
                        host_kv_ptr + (chunk_idx-1) * host_kv_mgr->chunk_numel,
                        offload_pinned_buffers[(chunk_idx-1)%2],
                        host_kv_mgr->chunk_numel * sizeof(uint16_t),
                        chunk_numel_part * sizeof(uint16_t));
                }
            }
            {
                cudaEventSynchronize(offload_memcpy_event[(num_chunks-1)%2]);
                offload_host_memcpy(
                    host_kv_ptr + (num_chunks-1) * host_kv_mgr->chunk_numel,
                    offload_pinned_buffers[(num_chunks-1)%2],
                    host_kv_mgr->chunk_numel * sizeof(uint16_t),
                    chunk_numel_part * sizeof(uint16_t));
            }
            // std::cout << "[Offload Launch] {" << dbg_uid << "}: copied data" << std::endl;

            // bookkeepping int host kv storage
            {
                nvtx3::scoped_range r{"offload_epilogue"};
                {
                    nvtx3::scoped_range r1{"offload_bookkeeping"};

                    size_t page_offset = 0;
                    int* offload_startpos = reinterpret_cast<int *>(host_metadata.data() + num_offload_uids * 2);
                    int* offload_lengths = reinterpret_cast<int *>(host_metadata.data() + num_offload_uids * 3);

                    for (int seq_idx = 0; seq_idx < num_offload_uids; seq_idx++) {
                        int64_t uid = offload_uids[seq_idx];
                        uint16_t *input_ptr = host_kv_ptr + page_offset * this->page_stride;
                        host_kv_mgr->append_kvdata_v2(uid, offload_startpos[seq_idx], offload_lengths[seq_idx], input_ptr, gather_layer_stride);
                        this->_uid_to_offloaded_length[uid] = offload_startpos[seq_idx] + offload_lengths[seq_idx];
                        page_offset += offload_lengths[seq_idx] / this->num_tokens_per_page;
                        {
                            std::unique_lock<std::mutex> lock(queued_offload_lastpos_mutex_);
                            if (offload_startpos[seq_idx] + offload_lengths[seq_idx] == queued_offload_lastpos[uid]) {
                                queued_offload_lastpos.erase(uid);
                            }
                            queued_offload_tokens -= offload_lengths[seq_idx];
                        }
                    }
                }

                this->offload_busy_.store(false);
            }
            // std::cout << "[Offload Launch] {" << dbg_uid << "}: offloading finish" << std::endl;
        }
    }

public:
    int num_layers;
    int num_kv_heads;
    int kv_headdim;
    int num_tokens_per_page;
    int num_primary_cache_pages;
    int num_onload_buffer_pages;
    int num_reserved_buffer_pages;
    int num_tokens_per_chunk;
    int max_num_sequences;
    int max_sequence_length;

    size_t layer_stride;
    size_t k2v_stride;
    size_t page_stride;
    size_t per_token_kv_stride;

public:
    std::list<int64_t> _lru_list;
    std::unordered_map<int64_t, 
                       typename std::list<int64_t>::iterator> _lru_lookup_table;
    std::queue<int64_t> _empty_pages;
    std::unordered_map<int64_t, std::vector<int32_t>> _uid_to_page_id;
    std::unordered_map<int64_t, int32_t> _uid_to_paged_cache_startpos;
    std::unordered_map<int64_t, int32_t> _uid_to_paged_cache_length;
    std::unordered_map<int64_t, int32_t> _uid_to_offloaded_length;

    int num_onload_memcpy_worker;
    std::vector<std::thread> onload_memcpy_worker;
    std::vector<uint16_t*> onload_pinned_buffers;
    std::vector<cudaEvent_t> onload_memcpy_event;

    std::queue<std::tuple<void*, void*, size_t>> onload_memcpy_task_queue;
    std::mutex onload_memcpy_task_mtx_;
    std::condition_variable onload_memcpy_task_cv_;

    // int onload_memcpy_cmplt_flag;
    // std::mutex onload_memcpy_cmplt_mtx_;
    // std::condition_variable onload_memcpy_cmplt_cv_;
    
    std::barrier<> onload_memcpy_barrier_;

    std::thread offload_worker;

    std::queue<std::tuple<std::vector<int>, at::Tensor, at::Tensor, cudaEvent_t>> offload_task_queue;
    std::mutex offload_task_mutex_;
    std::condition_variable offload_task_cv_;
    std::unordered_map<int64_t, int> queued_offload_lastpos;
    size_t queued_offload_tokens;
    std::mutex queued_offload_lastpos_mutex_;
    size_t queued_offload_limits;

    int num_offload_memcpy_worker;
    std::vector<std::thread> offload_memcpy_worker;
    std::vector<uint16_t*> offload_pinned_buffers;
    std::vector<cudaEvent_t> offload_memcpy_event;

    std::queue<std::tuple<void*, void*, size_t>> offload_memcpy_task_queue;
    std::mutex offload_memcpy_task_mtx_;
    std::condition_variable offload_memcpy_task_cv_;

    int offload_memcpy_cmplt_flag;
    std::mutex offload_memcpy_cmplt_mtx_;
    std::condition_variable offload_memcpy_cmplt_cv_;

    std::unordered_map<int64_t, int> offload_freezed_uids_;
    std::mutex offload_freezed_uids_mtx_;

    bool terminate_;
    std::atomic<bool> offload_busy_;

    cudaStream_t worker_stream;
    cudaStream_t onload_stream;
    cudaStream_t offload_stream;

    HostKVStorageImpl *host_kv_mgr;

public:
    uint16_t *cache_table;
    c10::Device device;
};


void prepare_kvcache(
    GPUKVCacheMangerImpl& gpu_mgr,
    HostKVStorageImpl& host_mgr,
    std::vector<int64_t>& user_ids,
    std::vector<int64_t>& total_hist_lens, // all histo w/o candi
    at::Tensor page_ids_gpu_buffer,
    at::Tensor offload_page_ids_gpu_buffer,
    at::Tensor offload_uids_buffer,
    at::Tensor metadata_host_buffer,
    at::Tensor metadata_gpu_buffer) {

    const c10::cuda::OptionalCUDAGuard device_guard(gpu_mgr.device);

    // std::cout << "prepare_kvcache start" << std::endl << std::flush;

    int batch_size = user_ids.size();

    std::vector<int32_t> old_history_lengths = gpu_mgr.get_total_cache_length(user_ids);

    std::vector<int> page_indices;
    std::vector<int> offload_page_ids;
    int64_t *offload_user_ids = static_cast<int64_t*>(offload_uids_buffer.data_ptr());

    int *host_bufptr = static_cast<int*>(metadata_host_buffer.data_ptr());

    int *page_indptr = host_bufptr + 0;
    int *last_page_len = host_bufptr + batch_size + 1;
    int *total_history_lengths = host_bufptr + batch_size * 2 + 1;
    int *total_history_offsets = host_bufptr + batch_size * 3 + 1;
    int *new_history_nnz_cuda = host_bufptr + batch_size * 4 + 2;
    int *new_history_offsets = host_bufptr + batch_size * 4 + 3;
    // === ^ GPU === v Host ===
    int *new_offload_startpos = host_bufptr + batch_size * 5 + 4;
    int *new_offload_lengths = host_bufptr + batch_size * 6 + 4;

    int *num_page_ids = host_bufptr + batch_size * 7 + 4;
    int *num_offload_page_ids = host_bufptr + batch_size * 7 + 5;
    int *num_offload_user_ids = host_bufptr + batch_size * 7 + 6;

    size_t onload_page_offset = gpu_mgr.num_primary_cache_pages;
    size_t num_offload_uids = 0;
    size_t num_offload_pages = 0;

    page_indptr[0] = 0;
    total_history_offsets[0] = 0;
    new_history_offsets[0] = 0;

    const std::unordered_set<int64_t> freezed_uids(user_ids.begin(), user_ids.end());
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids[seq_idx];
        int total_history_length = total_hist_lens[seq_idx];

        std::vector<int32_t>& page_ids = gpu_mgr.alloc(uid, total_history_length, freezed_uids);
        int gpu_cache_startpos = gpu_mgr._uid_to_paged_cache_startpos[uid];
        int gpu_cache_length = gpu_mgr._uid_to_paged_cache_length[uid];

        int num_onload_pages = gpu_cache_startpos / gpu_mgr.num_tokens_per_page;
        for (int i = 0; i < num_onload_pages; i++) page_indices.push_back(onload_page_offset+i);
        page_indices.insert(page_indices.end(), page_ids.begin(), page_ids.end());
        page_indptr[seq_idx + 1] = page_indptr[seq_idx] + page_ids.size() + num_onload_pages;
        last_page_len[seq_idx] = gpu_cache_length % gpu_mgr.num_tokens_per_page;
        onload_page_offset += num_onload_pages;

        total_history_lengths[seq_idx] = total_history_length;
        total_history_offsets[seq_idx + 1] = total_history_offsets[seq_idx] + total_history_length;
        new_history_offsets[seq_idx + 1] = new_history_offsets[seq_idx] + total_history_length - old_history_lengths[seq_idx];

        auto offloaded_length = 0;
        auto chunked_length = total_history_length - total_history_length % gpu_mgr.num_tokens_per_chunk;
        if (gpu_mgr._uid_to_offloaded_length.find(uid) != gpu_mgr._uid_to_offloaded_length.end())
            offloaded_length = gpu_mgr._uid_to_offloaded_length[uid];
        {
            std::unique_lock<std::mutex> lock(gpu_mgr.queued_offload_lastpos_mutex_);
            if (gpu_mgr.queued_offload_lastpos.find(uid) != gpu_mgr.queued_offload_lastpos.end()) {
                offloaded_length = gpu_mgr.queued_offload_lastpos[uid];
            }
            if (total_history_length - offloaded_length >= gpu_mgr.num_tokens_per_chunk) {
                gpu_mgr.queued_offload_lastpos[uid] = chunked_length;
            }
        }
        if (total_history_length - offloaded_length >= gpu_mgr.num_tokens_per_chunk) {
            // auto chunked_length = total_history_length - total_history_length % gpu_mgr.num_tokens_per_chunk;
            auto new_offload_page_start = (offloaded_length - gpu_cache_startpos) / gpu_mgr.num_tokens_per_page;

            offload_user_ids[num_offload_uids] = uid;
            new_offload_startpos[num_offload_uids] = offloaded_length;
            new_offload_lengths[num_offload_uids] = chunked_length - offloaded_length;
            auto num_pages = new_offload_lengths[num_offload_uids] / gpu_mgr.num_tokens_per_page;
            offload_page_ids.insert(offload_page_ids.end(), page_ids.begin() + new_offload_page_start, page_ids.begin() + new_offload_page_start + num_pages);
            
            num_offload_uids += 1;
            num_offload_pages += num_pages;
        }
    }
    auto new_tokens = new_history_offsets[batch_size];
    *new_history_nnz_cuda = new_tokens;
    *num_page_ids = page_indptr[batch_size];
    *num_offload_page_ids = num_offload_pages;
    *num_offload_user_ids = num_offload_uids;
    // std::cout << "num_offload_pages: " << num_offload_pages << std::endl << std::flush;

    cudaMemcpyAsync(page_ids_gpu_buffer.data_ptr(), page_indices.data(), page_indptr[batch_size] * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_mgr.worker_stream);
    cudaMemcpyAsync(offload_page_ids_gpu_buffer.data_ptr(), offload_page_ids.data(), num_offload_pages * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_mgr.worker_stream);

    size_t host_buffer_d2h_size = (batch_size * 5 + 4) * sizeof(int32_t);
    cudaMemcpyAsync(metadata_gpu_buffer.data_ptr(), metadata_host_buffer.data_ptr(), host_buffer_d2h_size, cudaMemcpyHostToDevice, gpu_mgr.worker_stream);
    
    int *gpu_bufptr = static_cast<int*>(metadata_gpu_buffer.data_ptr());

    int *total_history_lengths_dev = gpu_bufptr + batch_size * 2 + 1;
    int *new_history_offsets_dev = gpu_bufptr + batch_size * 4 + 3;
    int *batch_indices_dev = gpu_bufptr + batch_size * 5 + 4;
    int *position_dev = gpu_bufptr + batch_size * 5 + 4 + new_tokens;

    GetPagedBatchIndicesPositions(
        batch_size,
        new_history_offsets_dev,
        total_history_lengths_dev,
        batch_indices_dev,
        position_dev,
        gpu_mgr.worker_stream
    );

    cudaStreamSynchronize(gpu_mgr.worker_stream);
    // std::cout << "prepare_kvcache stop" << std::endl << std::flush;
 }

}  // namespace kvcache

PYBIND11_MODULE(paged_kvcache_ops, m) {
  m.def("append_kvcache", &append_paged_kv_cache, "append paged kv cache on GPU", py::call_guard<py::gil_scoped_release>());
  m.def("gather_kvcache", &gather_paged_kv_cache, "gather paged kv cache on GPU", py::call_guard<py::gil_scoped_release>());

  py::class_<kvcache::HostKVStorageImpl>(m, "HostKVStorageImpl")
    .def(py::init<int, int, int, int, int64_t>(), 
         py::arg("num_layers"),
         py::arg("num_kv_heads"),
         py::arg("kv_headdim"),
         py::arg("num_tokens_per_page"),
         py::arg("num_tokens_per_chunk"))
  ;

  py::class_<kvcache::GPUKVCacheMangerImpl>(m, "GPUKVCacheMangerImpl")
    .def(py::init<int, int, int, int, int, int, int, int, int, int, at::Tensor, kvcache::HostKVStorageImpl&>(),
         py::arg("num_layers"),
         py::arg("num_kv_heads"),
         py::arg("kv_headdim"),
         py::arg("num_tokens_per_page"),
         py::arg("num_primary_cache_pages"),
         py::arg("num_onload_buffer_pages"),
         py::arg("num_reserved_buffer_pages"),
         py::arg("num_tokens_per_chunk"), 
         py::arg("max_num_sequences"),
         py::arg("max_sequence_length"),
         py::arg("cache_table"),
         py::arg("host_kv_mgr"))
    .def("get_total_cache_length", &kvcache::GPUKVCacheMangerImpl::get_total_cache_length)
    .def("evict_all", &kvcache::GPUKVCacheMangerImpl::evict_all)
    .def("onload_kvcache", &kvcache::GPUKVCacheMangerImpl::onload_kvcache, py::call_guard<py::gil_scoped_release>())
    .def("offload_kvcache", &kvcache::GPUKVCacheMangerImpl::offload_kvcache, py::call_guard<py::gil_scoped_release>())
    .def("is_busy_offloading", &kvcache::GPUKVCacheMangerImpl::is_busy_offloading)
  ;

  py::class_<kvcache::KVOnloadHandle>(m, "KVOnloadHandle")
    .def(py::init<int>(), py::arg("num_layers"))
    // .def("wait", &kvcache::KVOnloadHandle::wait)
    .def("wait_host", &kvcache::KVOnloadHandle::wait_host)
    .def("reset", &kvcache::KVOnloadHandle::reset)
  ;

  py::class_<kvcache::KVOffloadHandle>(m, "KVOffloadHandle")
    .def(py::init())
    .def("record_ready", &kvcache::KVOffloadHandle::record_ready)
  ;

  m.def("prepare_kvcache", &kvcache::prepare_kvcache, "prepare_kvcache", py::call_guard<py::gil_scoped_release>());
}