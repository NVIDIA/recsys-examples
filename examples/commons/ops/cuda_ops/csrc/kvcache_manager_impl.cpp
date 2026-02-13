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

#include "kvcache_manager_impl.h"
#include <nvtx3/nvtx3.hpp>
#include "nvcomp/ans.h"

#define cudaCheck(ans) { cudaSuccesAssert((ans), __FILE__, __LINE__); }
inline void cudaSuccesAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

cudaError_t GetPagedBatchIndicesPositions(
  int32_t batch_size,
  int32_t* append_indptr,
  int32_t* seq_lens_ptr,
  int32_t* batch_indices_ptr,
  int32_t* positions_ptr,
  cudaStream_t stream);

void gather_paged_kv_cache_all_layers(
    uint16_t *gather_kv_gpu_buffer,
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
    const int num_sms,
    cudaStream_t stream);

namespace kvcache {

MultithreadMemcpyProcessor::MultithreadMemcpyProcessor(int num_workers)
: num_workers_(num_workers)
, start_barrier_(std::barrier<>(num_workers))
, end_barrier_(std::barrier<>(num_workers))
, dst_(nullptr)
, src_(nullptr)
, localbytes_(0)
, terminate_(false) {
    for (int i = 1; i < num_workers; i++) {
        this->workers_.emplace_back(std::thread(
            &MultithreadMemcpyProcessor::memcpy_coworker_loop, this, i
        ));
    }
}

MultithreadMemcpyProcessor::~MultithreadMemcpyProcessor() {
    this->terminate_ = true;
    
    start_barrier_.arrive_and_wait();
    for (int i = 0; i < this->num_workers_-1; i++) {
        this->workers_[i].join();
    }
}

inline const size_t MultithreadMemcpyProcessor::num_workers() const {
    return this->num_workers_;
}

void MultithreadMemcpyProcessor::memcpy(void* dst, void* src, size_t bytes, size_t bytes_part) {
    NVTX3_FUNC_RANGE();
    // auto start = std::chrono::high_resolution_clock::now();
    {
        this->dst_ = reinterpret_cast<char *>(dst);
        this->src_ = reinterpret_cast<char *>(src);
        this->localbytes_ = bytes_part;
        start_barrier_.arrive_and_wait();
    }
    std::memcpy(dst, src, bytes_part);
    {
        end_barrier_.arrive_and_wait();
        this->dst_ = this->src_ = nullptr;
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

void MultithreadMemcpyProcessor::memcpy_coworker_loop(const int idx) {
    while (true) {
        start_barrier_.arrive_and_wait();
        if (this->terminate_) break;
        std::memcpy(this->dst_ + idx * this->localbytes_, 
                    this->src_ + idx * this->localbytes_,
                    this->localbytes_);
        end_barrier_.arrive_and_wait();
    }
}


PinnedDoubleBuffer::PinnedDoubleBuffer(size_t buffer_bytes)
: ptr_(2), cuda_event_(2) {
    for (int i = 0; i < 2; i++) {
        cudaCheck(cudaMallocHost((void**)&ptr_[i], buffer_bytes));
        cudaCheck(cudaEventCreateWithFlags(&cuda_event_[i], cudaEventBlockingSync));
    }
}

PinnedDoubleBuffer::~PinnedDoubleBuffer() {
    for (int i = 0; i < 2; i++) {
        cudaEventDestroy(cuda_event_[i]);
        cudaFreeHost(ptr_[i]);
    }
}


KVCompressor::KVCompressor(int max_num_chunks, size_t chunk_numel, size_t chunk_bytes)
: max_num_chunks_(max_num_chunks), chunk_numel_(chunk_numel), chunk_bytes_(chunk_bytes) {
    if (max_num_chunks == 0) return;

    nvcompBatchedANSCompressGetMaxOutputChunkSize(chunk_bytes, k_comp_opts_, &max_comp_chunk_bytes_);

    std::vector<void*> ptrs(max_num_chunks);
    std::vector<size_t> bytes(max_num_chunks, chunk_bytes);

    // compress [offload] data struct
    cudaCheck(cudaMalloc((void**)&comp_out_buffer_, max_num_chunks * max_comp_chunk_bytes_));

    cudaCheck(cudaMalloc((void**)&comp_in_ptrs_, max_num_chunks * sizeof(void*)));  // setup once from gpu manager
    cudaCheck(cudaMalloc((void**)&comp_in_bytes_, max_num_chunks * sizeof(size_t)));
    cudaCheck(cudaMemcpy(comp_in_bytes_, bytes.data(), max_num_chunks * sizeof(size_t), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc((void**)&comp_out_ptrs_, max_num_chunks * sizeof(void*)));
    for (int i = 0; i < max_num_chunks; i++) ptrs[i] = (char *)comp_out_buffer_ + i * max_comp_chunk_bytes_;
    cudaCheck(cudaMemcpy(comp_out_ptrs_, ptrs.data(), max_num_chunks * sizeof(void*), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc((void**)&comp_out_bytes_, max_num_chunks * sizeof(size_t)));
    
    // decompress [onload] data struct
    cudaCheck(cudaMalloc((void**)&decomp_in_buffer_, max_num_chunks * max_comp_chunk_bytes_));

    cudaCheck(cudaMalloc((void**)&decomp_in_ptrs_, max_num_chunks * sizeof(void*)));
    for (int i = 0; i < max_num_chunks; i++) ptrs[i] = (char *)decomp_in_buffer_ + i * max_comp_chunk_bytes_;
    cudaCheck(cudaMemcpy(decomp_in_ptrs_, ptrs.data(), max_num_chunks * sizeof(void*), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc((void**)&decomp_in_bytes_, max_num_chunks * sizeof(size_t)));

    cudaCheck(cudaMalloc((void**)&decomp_out_ptrs_, max_num_chunks * sizeof(void*)));
    cudaCheck(cudaMalloc((void**)&decomp_out_bytes_, max_num_chunks * sizeof(size_t)));
    cudaCheck(cudaMalloc((void**)&decomp_buffer_bytes_, max_num_chunks * sizeof(size_t)));
    cudaCheck(cudaMemcpy(decomp_buffer_bytes_, bytes.data(), max_num_chunks * sizeof(size_t), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc((void**)&comp_status_, max_num_chunks * sizeof(nvcompStatus_t)));
    cudaCheck(cudaMallocHost((void**)&comp_status_cpu_, max_num_chunks * sizeof(nvcompStatus_t)));
    cudaCheck(cudaMalloc((void**)&decomp_status_, max_num_chunks * sizeof(nvcompStatus_t)));
    cudaCheck(cudaMallocHost((void**)&decomp_status_cpu_, max_num_chunks * sizeof(nvcompStatus_t)));

    nvcompBatchedANSCompressGetTempSizeAsync(
        max_num_chunks, chunk_bytes, k_comp_opts_, &comp_tmp_bytes_, max_num_chunks * chunk_bytes);
    cudaCheck(cudaMalloc(&comp_tmp_buffer_, comp_tmp_bytes_));

    nvcompBatchedANSDecompressGetTempSizeAsync(
        max_num_chunks, chunk_bytes, k_decomp_opts_, &decomp_tmp_bytes_, max_num_chunks * chunk_bytes);
    cudaCheck(cudaMalloc(&decomp_tmp_buffer_, decomp_tmp_bytes_));
}

KVCompressor::~KVCompressor() {
    if (max_num_chunks_ == 0) return;

    cudaFree(decomp_tmp_buffer_);
    cudaFree(comp_tmp_buffer_);
    cudaFreeHost(decomp_status_cpu_);
    cudaFree(decomp_status_);
    cudaFreeHost(comp_status_cpu_);
    cudaFree(comp_status_);
    cudaFree(decomp_buffer_bytes_);
    cudaFree(decomp_out_bytes_);
    cudaFree(decomp_out_ptrs_);
    cudaFree(decomp_in_bytes_);
    cudaFree(decomp_in_ptrs_);
    cudaFree(decomp_in_buffer_);
    cudaFree(comp_out_bytes_);
    cudaFree(comp_out_ptrs_);
    cudaFree(comp_in_bytes_);
    cudaFree(comp_in_ptrs_);
    cudaFree(comp_out_buffer_);
}

void KVCompressor::set_compress_input_buffer_ptrs(char *base_ptr, size_t num_chunks) {  // call once
    if (max_num_chunks_ == 0) return;
    std::vector<void*> ptrs(num_chunks);
    for (size_t idx = 0; idx < num_chunks; idx++) {
        ptrs[idx] = base_ptr + idx * chunk_bytes_;
    }
    cudaCheck(cudaMemcpy(comp_in_ptrs_, ptrs.data(), num_chunks * sizeof(void*), cudaMemcpyHostToDevice));
}

void KVCompressor::set_decompress_output_buffer_ptrs(char *base_ptr, size_t num_chunks, cudaStream_t stream) {  // call multiples
    std::vector<void*> ptrs(num_chunks);
    for (size_t idx = 0; idx < num_chunks; idx++) {
        ptrs[idx] = base_ptr + idx * chunk_bytes_;
    }
    cudaCheck(cudaMemcpyAsync(decomp_out_ptrs_, ptrs.data(), num_chunks * sizeof(void*), cudaMemcpyHostToDevice, stream));
}

void KVCompressor::compress(
    size_t *compressed_bytes_cpu,
    size_t num_chunks,
    cudaStream_t stream) {
    nvcompStatus_t comp_res = nvcompBatchedANSCompressAsync(
        comp_in_ptrs_, comp_in_bytes_,
        chunk_bytes_, num_chunks,
        comp_tmp_buffer_, comp_tmp_bytes_,
        comp_out_ptrs_, comp_out_bytes_,
        k_comp_opts_, comp_status_, stream);
    cudaCheck(cudaMemcpyAsync(compressed_bytes_cpu, comp_out_bytes_, num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost, stream));
}

void KVCompressor::decompress(
    size_t *compressed_bytes_cpu,
    size_t num_chunks,
    cudaStream_t stream) {
    cudaCheck(cudaMemcpyAsync(decomp_in_bytes_, compressed_bytes_cpu, num_chunks * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    nvcompStatus_t decomp_res = nvcompBatchedANSDecompressAsync(
        decomp_in_ptrs_, decomp_in_bytes_,
        decomp_buffer_bytes_, decomp_out_bytes_, num_chunks,
        decomp_tmp_buffer_, decomp_tmp_bytes_,
        decomp_out_ptrs_,
        k_decomp_opts_, decomp_status_, stream);
}

char *KVCompressor::comp_out_buffer() { return comp_out_buffer_; }
char *KVCompressor::comp_out_buffer(int index) { return comp_out_buffer_ + index * max_comp_chunk_bytes_; }

char *KVCompressor::decomp_in_buffer() { return decomp_in_buffer_; }
char *KVCompressor::decomp_in_buffer(int index) { return decomp_in_buffer_ + index * max_comp_chunk_bytes_; }

KVOnloadHandle::KVOnloadHandle() : no_onload(true) {}

KVOnloadHandle::KVOnloadHandle(
    int num_layers
)
: num_layers(num_layers)
, event(std::vector<cudaEvent_t>(num_layers))
, host_complete(num_layers, 0)
, no_onload(false) {
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        cudaCheck(cudaEventCreateWithFlags(&event[layer_idx], cudaEventBlockingSync));
    }
}

KVOnloadHandle::~KVOnloadHandle(){
    if (no_onload) return;
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        cudaEventDestroy(event[layer_idx]);
    }
};

void KVOnloadHandle::reset() {
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        host_complete[layer_idx] = 0;
    }
}

void KVOnloadHandle::complete_host(int layer_idx) {
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaCheck(cudaEventRecord(event[layer_idx], stream));
    {
        std::unique_lock<std::mutex> lock(mtx_);
        host_complete[layer_idx] = 1;
    }
    cv_.notify_one();
};

void KVOnloadHandle::complete_host(int layer_idx, cudaStream_t stream) {
    cudaCheck(cudaEventRecord(event[layer_idx], stream));
    {
        std::unique_lock<std::mutex> lock(mtx_);
        host_complete[layer_idx] = 1;
    }
    cv_.notify_one();
};

void KVOnloadHandle::wait_host(int layer_idx) {
    if (no_onload) return;
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this, layer_idx](){ return host_complete[layer_idx] == 1; });
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaCheck(cudaStreamWaitEvent(stream, event[layer_idx], 0));
};

KVOffloadHandle::KVOffloadHandle() : gpu_kv_mgr(nullptr), no_offload(true) { }

KVOffloadHandle::KVOffloadHandle(
    int num_layers,
    GPUKVCacheMangerImpl& gpu_kv_mgr,
    bool has_offload
)
: num_layers(num_layers)
, ready_event(std::vector<cudaEvent_t>(num_layers))
, no_offload(false) {
    this->gpu_kv_mgr = (has_offload) ? &gpu_kv_mgr : nullptr;
}

void KVOffloadHandle::mark_ready(int layer_idx) {
    if (this->gpu_kv_mgr == nullptr || this->no_offload) return;

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaCheck(cudaEventRecord(this->ready_event[layer_idx], stream));
    this->host_ready[layer_idx] = 1;
    this->gpu_kv_mgr->offload_ready_cv_.notify_one();
}

void KVOffloadHandle::set_no_offload() {
    this->no_offload = true;
}


HostKVStorageImpl::HostKVStorageImpl(
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
    , _uid_to_chunk_bytes(num_layers, std::unordered_map<int64_t, std::vector<size_t>>())
{
    this->chunk_numel = num_tokens_per_chunk * 2 * num_kv_heads * kv_headdim;
    this->page_numel = 2 * page_size * num_kv_heads * kv_headdim;
    this->per_token_numel = 2 * num_kv_heads * kv_headdim;
};

HostKVStorageImpl::~HostKVStorageImpl()
{}

int64_t HostKVStorageImpl::get_kvdata_length(int64_t user_id) {
    auto it = _uid_to_length.find(user_id);
    if (it ==  _uid_to_length.end()) return 0;
    return it->second;
};

void HostKVStorageImpl::append_kvdata(
    int64_t user_id, int64_t start_position, int64_t length, 
    uint16_t *kvdata_buffer, size_t buffer_layer_stride) {
    std::unique_lock<std::mutex> lock(host_kvcache_mutex_);
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

    const size_t num_chunks = length / chunk_size;
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        uint16_t* src_ptr = kvdata_buffer + layer_idx * buffer_layer_stride;
        
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx ++) {
            _uid_to_chunk_id[layer_idx][user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr + chunk_idx * this->chunk_numel));
        }

        _uid_to_mempool[user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr));
    }
    _uid_to_length[user_id] = start_position + length;
};

void HostKVStorageImpl::append_kvdata(
    int64_t user_id, int64_t start_position, int64_t length, 
    uint16_t *kvdata_buffer, size_t buffer_layer_stride,
    size_t *kvdata_bytes, size_t bytes_layer_stride) {
    std::unique_lock<std::mutex> lock(host_kvcache_mutex_);
    assert(length % this->chunk_size == 0);
    if (start_position != 0) {
        assert(_uid_to_length[user_id] == start_position);
    }
    else {
        assert(_uid_to_length.find(user_id) == _uid_to_length.end());
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            _uid_to_chunk_id[layer_idx][user_id] = std::vector<uintptr_t>();
            _uid_to_chunk_bytes[layer_idx][user_id] = std::vector<size_t>();
        }
        _uid_to_mempool[user_id] = std::vector<uintptr_t>();
    }

    const size_t num_chunks = length / chunk_size;
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        uint16_t* src_ptr = kvdata_buffer + layer_idx * buffer_layer_stride;
        
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx ++) {
            _uid_to_chunk_id[layer_idx][user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr + chunk_idx * this->chunk_numel));
            _uid_to_chunk_bytes[layer_idx][user_id].push_back(kvdata_bytes[layer_idx * bytes_layer_stride + chunk_idx]);
        }

        _uid_to_mempool[user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr));
    }
    _uid_to_length[user_id] = start_position + length;
};

std::vector<uint16_t*> HostKVStorageImpl::get_kvdata(int64_t user_id, int64_t length, int64_t layer_idx) {
    // int64_t offloaded_length = get_kvdata_length(user_id);
    // assert(offloaded_length >= length);
    std::unique_lock<std::mutex> lock(host_kvcache_mutex_);

    std::vector<uint16_t*> chunk_ptrs;
    if (length == 0) {
        return chunk_ptrs;
    }
    // assert(length % this->chunk_size == 0);
    size_t num_chunks = length / chunk_size;
    const auto &chunk_ptr_list = _uid_to_chunk_id[layer_idx][user_id];
    // assert(chunk_ptr_list.size() >= num_chunks);
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        uint16_t* src_ptr = reinterpret_cast<uint16_t*>(chunk_ptr_list[chunk_idx]);
        chunk_ptrs.push_back(src_ptr);
    }
    return chunk_ptrs;
};

std::vector<size_t> HostKVStorageImpl::get_kvdata_bytes(int64_t user_id, int64_t length, int64_t layer_idx) {
    std::unique_lock<std::mutex> lock(host_kvcache_mutex_);

    std::vector<size_t> chunk_bytes;
    if (length == 0 || _uid_to_chunk_bytes.size() == 0 || _uid_to_chunk_bytes[0].size() == 0) {
        return chunk_bytes;
    }
    // assert(length % this->chunk_size == 0);
    size_t num_chunks = length / chunk_size;
    const auto &chunk_bytes_list = _uid_to_chunk_bytes[layer_idx][user_id];
    // assert(chunk_ptr_list.size() >= num_chunks);
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        chunk_bytes.push_back(chunk_bytes_list[chunk_idx]);
    }
    return chunk_bytes;
};

std::vector<at::Tensor> HostKVStorageImpl::get_kvdata_tensor(std::vector<int64_t> user_ids, bool with_concat) {
    int batch_size = user_ids.size();

    std::vector<int64_t> seqlens(batch_size, 0);
    int64_t total_seqlen = 0;
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        seqlens[seq_idx] = get_kvdata_length(user_ids[seq_idx]);
        total_seqlen += seqlens[seq_idx];
    }
    int64_t num_total_pages = total_seqlen / this->page_size;

    at::Tensor tensor_res = at::ones({
        this->num_layers, num_total_pages, 2, this->page_size, this->num_kv_heads, this->kv_headdim}, torch::kBFloat16);
    
    int64_t seqlen_offset = 0;
    uint16_t *raw_ptr = static_cast<uint16_t*>(tensor_res.data_ptr());

    size_t chunk_bytes = this->chunk_numel * sizeof(uint16_t);

    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        uint16_t *seq_ptr = raw_ptr + (seqlen_offset / this->page_size) * tensor_res.stride(1);
        int num_chunks = seqlens[seq_idx] / this->chunk_size;

        for (int layer_idx = 0 ; layer_idx < this->num_layers; layer_idx++){
            auto host_ptrs = get_kvdata(user_ids[seq_idx], seqlens[seq_idx], layer_idx);
            for (int chk_idx = 0; chk_idx < num_chunks; chk_idx++) {
                std::memcpy(seq_ptr + layer_idx * tensor_res.stride(0) + chk_idx * this->chunk_numel, host_ptrs[chk_idx], chunk_bytes);
            }
        }

        seqlen_offset += seqlens[seq_idx];
    }
    
    std::vector<at::Tensor> res({tensor_res});
    return res;
}

void HostKVStorageImpl::init_random_kvdata(int64_t user_id, size_t num_tokens) {
    if (_uid_to_length.find(user_id) != _uid_to_length.end()) return;
    if (num_tokens == 0) return;
    
    size_t num_chunks = ((num_tokens + this->chunk_size - 1) / this->chunk_size);

    uint16_t *host_data_ptr = (uint16_t *)malloc(this->num_layers * num_chunks * this->chunk_numel * sizeof(uint16_t));

    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
        _uid_to_chunk_id[layer_idx][user_id] = std::vector<uintptr_t>();
    _uid_to_mempool[user_id] = std::vector<uintptr_t>();

    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        uint16_t* src_ptr = host_data_ptr + layer_idx * num_chunks * this->chunk_numel;
        
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx ++) {
            _uid_to_chunk_id[layer_idx][user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr + chunk_idx * this->chunk_numel));
        }
    }
    _uid_to_mempool[user_id].push_back(reinterpret_cast<uintptr_t>(host_data_ptr));
    _uid_to_length[user_id] = num_chunks * this->chunk_size;
}

GPUKVCacheMangerImpl::GPUKVCacheMangerImpl(
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
    HostKVStorageImpl& host_kv_mgr,
    size_t max_queued_offload_tokens,
    int onload_buffer_chunks,
    int offload_buffer_chunks,
    int num_memcpy_workers,
    bool enable_nvcomp)
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
    , queued_offload_limits(max_queued_offload_tokens)
    , num_onload_device_chunks(onload_buffer_chunks)
    , num_offload_device_chunks(offload_buffer_chunks)
    , enable_nvcomp(enable_nvcomp)
    , compressor(enable_nvcomp ? 8 : 0, host_kv_mgr.chunk_numel, host_kv_mgr.chunk_numel * sizeof(uint16_t))
    , onload_pin_buffer(host_kv_mgr.chunk_numel * sizeof(uint16_t))
    , offload_pin_buffer(host_kv_mgr.chunk_numel * sizeof(uint16_t))
    , onload_memcpy_workers(num_memcpy_workers)
    , offload_memcpy_workers(num_memcpy_workers)
    , cache_table(static_cast<uint16_t*>(cache_table_tensor.data_ptr()))
    , device(cache_table_tensor.device())
{
    const c10::cuda::OptionalCUDAGuard device_guard(this->device);

    for (int page_id = 0; page_id < num_primary_cache_pages; page_id++)
        _empty_pages.push(page_id);

    page_stride = 2 * num_tokens_per_page * num_kv_heads * kv_headdim;
    k2v_stride = num_tokens_per_page * num_kv_heads * kv_headdim;
    layer_stride = (num_primary_cache_pages + num_onload_buffer_pages) * page_stride;

    per_token_kv_stride = 2 * num_kv_heads * kv_headdim;

    cudaCheck(cudaStreamCreateWithFlags(&worker_stream, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&onload_stream, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&offload_stream, cudaStreamNonBlocking));

    this->host_kv_mgr = &host_kv_mgr;

    cudaCheck(cudaMalloc((void**)&onload_device_buffers, this->num_onload_device_chunks * host_kv_mgr.chunk_numel * sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void**)&offload_device_buffers, this->num_offload_device_chunks * host_kv_mgr.chunk_numel * sizeof(uint16_t)));
    compressor.set_compress_input_buffer_ptrs(
        reinterpret_cast<char*>(offload_device_buffers), this->num_offload_device_chunks);

    this->terminate_ = false;

    this->queued_offload_tokens = 0;
    this->queued_offload_limits = max_queued_offload_tokens;
    this->offload_busy_.store(false);
    this->offload_worker = std::thread(&GPUKVCacheMangerImpl::offload_loop, this);
};

GPUKVCacheMangerImpl::~GPUKVCacheMangerImpl() {
    {
        std::unique_lock<std::mutex> lock(offload_task_mutex_);
        this->terminate_ = true;
    }
    offload_task_cv_.notify_all();

    this->offload_worker.join();

    cudaFree(offload_device_buffers);
    cudaFree(onload_device_buffers);
}

int64_t GPUKVCacheMangerImpl::getUIdToEvict(std::unordered_set<int64_t> extra_freezed_uids) {
    while (true) {
        int num_offloading_uids = 0;
        {
            std::unique_lock<std::mutex> lock(offload_freezed_uids_mtx_);
            num_offloading_uids = offload_freezed_uids_.size();

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

std::vector<int32_t>& GPUKVCacheMangerImpl::alloc(int64_t uid, int new_total_length, std::unordered_set<int64_t> freezed_uids) {
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

    while ((size_t)num_append_pages > _empty_pages.size()) {
        int64_t uid_to_evict = getUIdToEvict(freezed_uids);
        evict(uid_to_evict);
    }

    std::vector<int32_t>& page_ids = _uid_to_page_id[uid];
    for (int i = 0; i < num_append_pages; i++) {
        page_ids.push_back(_empty_pages.front());
        _empty_pages.pop();
    }
    _uid_to_paged_cache_length[uid] = new_cached_len;

    return page_ids;
};

std::vector<int32_t> GPUKVCacheMangerImpl::get_total_cache_length(std::vector<int64_t>& uids) {
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
    
void GPUKVCacheMangerImpl::evict(int64_t uid)
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

void GPUKVCacheMangerImpl::evict_all()
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

    this->host_kv_mgr->_uid_to_chunk_id.clear();
    this->host_kv_mgr->_uid_to_length.clear();
    // for (auto p : this->host_kv_mgr->_uid_to_mempool) {
    //     for (auto ptr : p.second) {
    //         free(reinterpret_cast<uint16_t*>(ptr));
    //     }
    // }
    this->host_kv_mgr->_uid_to_mempool.clear();
};

void GPUKVCacheMangerImpl::invalid(int64_t uid) {
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

bool GPUKVCacheMangerImpl::retain(int64_t uid)
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

uint16_t *GPUKVCacheMangerImpl::get_cache_table(void) {
    return cache_table;
};

uint16_t *GPUKVCacheMangerImpl::get_cache_table_by_layer(int layer_idx) {
    return cache_table + layer_idx * layer_stride;
};

void GPUKVCacheMangerImpl::onload_kvcache(
    std::vector<int64_t>& user_ids, 
    KVOnloadHandle& onloadhandle) {
    const c10::cuda::OptionalCUDAGuard device_guard(this->device);

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
        return;
    }

    const size_t k_num_memcpy_workers = this->onload_memcpy_workers.num_workers();
    const size_t k_chunk_numel_part = host_kv_mgr->chunk_numel / this->onload_memcpy_workers.num_workers();

    int task_idx = 0;
    for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
        nvtx3::scoped_range r{"onload_layer_" + std::to_string(layer_idx)};
        uint16_t *gpu_onload_buffer = this->get_cache_table_by_layer(layer_idx) + this->num_primary_cache_pages * this->page_stride;

        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            std::vector<uint16_t *> chunk_ptrs = host_kv_mgr->get_kvdata(user_ids[seq_idx], onload_length[seq_idx], layer_idx);
            std::vector<size_t> chunk_bytes;
            if (this->enable_nvcomp)
                chunk_bytes = host_kv_mgr->get_kvdata_bytes(user_ids[seq_idx], onload_length[seq_idx], layer_idx);
            
            for (int chunk_idx = 0; chunk_idx < chunk_ptrs.size(); chunk_idx++) {

                cudaCheck(cudaEventSynchronize(onload_pin_buffer.cuda_event_[task_idx%2]));
                if (!this->enable_nvcomp) {
                    this->onload_memcpy_workers.memcpy(onload_pin_buffer.ptr_[task_idx%2], chunk_ptrs[chunk_idx], host_kv_mgr->chunk_numel * sizeof(uint16_t), k_chunk_numel_part * sizeof(uint16_t));
                    cudaCheck(cudaMemcpyAsync(gpu_onload_buffer + onload_offsets[seq_idx] * this->per_token_kv_stride + chunk_idx * host_kv_mgr->chunk_numel,
                        onload_pin_buffer.ptr_[task_idx%2], host_kv_mgr->chunk_numel * sizeof(uint16_t), cudaMemcpyHostToDevice, this->onload_stream));
                } else {
                    size_t chunk_bytes_part = (chunk_bytes[chunk_idx] + k_num_memcpy_workers - 1) / k_num_memcpy_workers;
                    this->onload_memcpy_workers.memcpy(onload_pin_buffer.ptr_[task_idx%2], chunk_ptrs[chunk_idx], chunk_bytes[chunk_idx], chunk_bytes_part);

                    cudaCheck(cudaMemcpyAsync(compressor.decomp_in_buffer(0),
                        onload_pin_buffer.ptr_[task_idx%2], chunk_bytes[chunk_idx], cudaMemcpyHostToDevice, this->onload_stream));
                    compressor.set_decompress_output_buffer_ptrs(
                        (char*)(gpu_onload_buffer + onload_offsets[seq_idx] * this->per_token_kv_stride + chunk_idx * host_kv_mgr->chunk_numel),
                        1, this->onload_stream);
                    compressor.decompress(&chunk_bytes[chunk_idx], 1, this->onload_stream);
                }
                cudaCheck(cudaEventRecord(onload_pin_buffer.cuda_event_[task_idx%2], this->onload_stream));

                task_idx++;
            }
        }

        onloadhandle.complete_host(layer_idx, this->onload_stream);
    }
};

void GPUKVCacheMangerImpl::offload_kvcache(
    KVOffloadHandle& offload_handle,
    at::Tensor offload_user_ids,      // host
    at::Tensor offload_page_ids,      // gpu
    at::Tensor new_offload_startpos,  // host
    at::Tensor new_offload_lengths)   // host
{
    NVTX3_FUNC_RANGE();
    const size_t num_offload_uids = offload_user_ids.numel();
    {
        std::unique_lock<std::mutex> lock(queued_offload_lastpos_mutex_);
        if (queued_offload_tokens >= queued_offload_limits) {
            return;
        }
        for (size_t seq_idx = 0; seq_idx < num_offload_uids; seq_idx++) {
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
    
    offload_handle.host_ready = new int[this->num_layers];
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        offload_handle.host_ready[layer_idx] = 0;
        cudaCheck(cudaEventCreateWithFlags(&offload_handle.ready_event[layer_idx], cudaEventBlockingSync));
    }
    
    {
        std::unique_lock<std::mutex> lock(offload_freezed_uids_mtx_);
        int64_t *offload_uids = reinterpret_cast<int64_t*>(offload_host_metadata.data());
        for (size_t idx = 0; idx < num_offload_uids; idx++) {
            int cur_freezed_times = offload_freezed_uids_[offload_uids[idx]];
            offload_freezed_uids_[offload_uids[idx]] = cur_freezed_times + 1;
        }
    }
    {
        std::unique_lock<std::mutex> lock(offload_task_mutex_);
        offload_task_queue.push(std::make_tuple(
            offload_host_metadata,
            offload_page_ids, 
            offload_handle.ready_event,
            offload_handle.host_ready
        ));
    }

    offload_task_cv_.notify_one();
};

bool GPUKVCacheMangerImpl::is_busy_offloading() {
    return !offload_task_queue.empty() || this->offload_busy_.load();
}

void GPUKVCacheMangerImpl::init_random_offload_status(int64_t user_id, size_t length) {
    _uid_to_offloaded_length[user_id] = length;
}

void GPUKVCacheMangerImpl::offload_loop()
{
    const c10::cuda::OptionalCUDAGuard device_guard(this->device);
    int dev_id = 0;
    int k_num_sms = 0;
    cudaCheck(cudaGetDevice(&dev_id));
    cudaCheck(cudaDeviceGetAttribute(&k_num_sms, cudaDevAttrMultiProcessorCount, dev_id));

    while (true) {
        std::vector<int> host_metadata;
        at::Tensor offload_page_ids;
        std::vector<cudaEvent_t> offload_gpu_acq_event;
        int *event_recorded;
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
                    host_metadata, offload_page_ids, offload_gpu_acq_event, event_recorded
                ) = offload_task_queue.front();
            }
            {
                nvtx3::scoped_range r2{"offload_prelogue pop"};
                offload_task_queue.pop();
            }

            this->offload_busy_.store(true);
        }

        int64_t *offload_uids = reinterpret_cast<int64_t *>(host_metadata.data());
        const int num_offload_uids = host_metadata.size() / 4;
        const int num_offload_pages = offload_page_ids.numel();
        const size_t gather_layer_stride = num_offload_pages * this->page_stride;

        size_t host_bytes = this->num_layers * gather_layer_stride * sizeof(uint16_t);
        uint16_t *host_kv_ptr = static_cast<uint16_t *>(aligned_alloc(sysconf(_SC_PAGESIZE), host_bytes));

        const size_t k_num_memcpy_workers = this->offload_memcpy_workers.num_workers();
        const size_t k_chunk_numel_part = host_kv_mgr->chunk_numel / this->offload_memcpy_workers.num_workers();

        const int num_chunks_per_layer = (num_offload_pages * this->num_tokens_per_page) / this->num_tokens_per_chunk;
        const int num_chunks = num_chunks_per_layer * this->num_layers;
        const int num_pages_per_chunk = this->num_tokens_per_chunk / this->num_tokens_per_page;
        std::vector<size_t> comp_bytes_per_chunk(num_chunks, 0);

        for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
            uint16_t *host_kv_layer_ptr = host_kv_ptr + layer_idx * gather_layer_stride;

            {
                std::unique_lock<std::mutex> lock(offload_ready_mtx_);
                this->offload_ready_cv_.wait(lock, [this, event_recorded, layer_idx] { return event_recorded[layer_idx] == 1; });
            }
            cudaCheck(cudaStreamWaitEvent(this->offload_stream, offload_gpu_acq_event[layer_idx]));
            cudaEventDestroy(offload_gpu_acq_event[layer_idx]);
            
            const bool last_layer = (layer_idx == this->num_layers - 1);


            for (int chunk_idx = 0; chunk_idx < num_chunks_per_layer; chunk_idx++) {
                if (chunk_idx % this->num_offload_device_chunks == 0) {
                    const int num_d2h_chunks = std::min((num_chunks_per_layer - chunk_idx), this->num_offload_device_chunks);

                    const int num_d2d_pages = num_d2h_chunks * num_pages_per_chunk;
                    const int page_ids_offsets = chunk_idx * num_pages_per_chunk;

                    // Gather: Device to device
                    gather_paged_kv_cache_all_layers(
                        offload_device_buffers,
                        this->get_cache_table_by_layer(layer_idx),
                        static_cast<int*>(offload_page_ids.data_ptr()) + page_ids_offsets,
                        1,
                        gather_layer_stride,
                        this->layer_stride,
                        this->num_kv_heads,
                        this->kv_headdim,
                        this->num_tokens_per_page,
                        this->page_stride,
                        this->k2v_stride,
                        this->num_kv_heads * this->kv_headdim,
                        this->kv_headdim,
                        num_d2d_pages,
                        k_num_sms,
                        this->offload_stream);
                    // release on gpu kvcache
                    if (last_layer && chunk_idx + this->num_offload_device_chunks >= num_chunks_per_layer) {
                        cudaCheck(cudaStreamSynchronize(this->offload_stream));
                        std::unique_lock<std::mutex> lock(offload_freezed_uids_mtx_);
                        for (int idx = 0; idx < num_offload_uids; idx++) {
                            int cur_freezed_times = offload_freezed_uids_[offload_uids[idx]];
                            if (cur_freezed_times == 1) {
                                offload_freezed_uids_.erase(offload_uids[idx]);
                            } else {
                                offload_freezed_uids_[offload_uids[idx]] = cur_freezed_times - 1;
                            }
                        }
                    }
                    if (this->enable_nvcomp) {
                        compressor.compress(
                            comp_bytes_per_chunk.data() + layer_idx * num_chunks_per_layer + chunk_idx,
                            num_d2h_chunks, this->offload_stream);
                    }
                }

                if (!this->enable_nvcomp) {
                    cudaCheck(cudaMemcpyAsync(
                        offload_pin_buffer.ptr_[chunk_idx%2],
                        offload_device_buffers + (chunk_idx % this->num_offload_device_chunks) * host_kv_mgr->chunk_numel,
                        host_kv_mgr->chunk_numel * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost, 
                        this->offload_stream));
                } else {
                    cudaCheck(cudaMemcpyAsync(
                        offload_pin_buffer.ptr_[chunk_idx%2],
                        compressor.comp_out_buffer(chunk_idx % this->num_offload_device_chunks),
                        comp_bytes_per_chunk[layer_idx * num_chunks_per_layer + chunk_idx],
                        cudaMemcpyDeviceToHost, 
                        this->offload_stream));
                }
                cudaCheck(cudaEventRecord(offload_pin_buffer.cuda_event_[chunk_idx%2], this->offload_stream));

                if (chunk_idx > 0) {
                    cudaCheck(cudaEventSynchronize(offload_pin_buffer.cuda_event_[(chunk_idx-1)%2]));
                    size_t chunk_bytes = host_kv_mgr->chunk_numel * sizeof(uint16_t);
                    size_t chunk_bytes_part = k_chunk_numel_part * sizeof(uint16_t);
                    if (this->enable_nvcomp) {
                        chunk_bytes_part = comp_bytes_per_chunk[layer_idx * num_chunks_per_layer + chunk_idx-1];
                        chunk_bytes_part = (chunk_bytes_part + k_num_memcpy_workers - 1) / k_num_memcpy_workers;
                        chunk_bytes = chunk_bytes_part * k_num_memcpy_workers;
                    }
                    this->offload_memcpy_workers.memcpy(
                        host_kv_layer_ptr + (chunk_idx-1) * host_kv_mgr->chunk_numel,
                        offload_pin_buffer.ptr_[(chunk_idx-1)%2],
                        chunk_bytes, chunk_bytes_part);
                }
            }

            {
                cudaCheck(cudaEventSynchronize(offload_pin_buffer.cuda_event_[(num_chunks_per_layer-1)%2]));
                size_t chunk_bytes = host_kv_mgr->chunk_numel * sizeof(uint16_t);
                size_t chunk_bytes_part = k_chunk_numel_part * sizeof(uint16_t);
                if (this->enable_nvcomp) {
                    chunk_bytes_part = comp_bytes_per_chunk[layer_idx * num_chunks_per_layer + num_chunks_per_layer-1];
                    chunk_bytes_part = (chunk_bytes_part + k_num_memcpy_workers - 1) / k_num_memcpy_workers;
                    chunk_bytes = chunk_bytes_part * k_num_memcpy_workers;
                }

                this->offload_memcpy_workers.memcpy(
                    host_kv_layer_ptr + (num_chunks_per_layer-1) * host_kv_mgr->chunk_numel,
                    offload_pin_buffer.ptr_[(num_chunks_per_layer-1)%2],
                    chunk_bytes, chunk_bytes_part);
            }
        } // close layer loop

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
                    if (!this->enable_nvcomp)
                        host_kv_mgr->append_kvdata(uid, offload_startpos[seq_idx], offload_lengths[seq_idx], input_ptr, gather_layer_stride);
                    else
                        host_kv_mgr->append_kvdata(uid, offload_startpos[seq_idx], offload_lengths[seq_idx], 
                                                        input_ptr, gather_layer_stride,
                                                        comp_bytes_per_chunk.data() + page_offset / num_pages_per_chunk, num_chunks_per_layer);
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

        delete[] event_recorded;
    }
}

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

    cudaCheck(cudaMemcpyAsync(page_ids_gpu_buffer.data_ptr(), page_indices.data(), page_indptr[batch_size] * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_mgr.worker_stream));
    cudaCheck(cudaMemcpyAsync(offload_page_ids_gpu_buffer.data_ptr(), offload_page_ids.data(), num_offload_pages * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_mgr.worker_stream));

    size_t host_buffer_d2h_size = (batch_size * 5 + 4) * sizeof(int32_t);
    cudaCheck(cudaMemcpyAsync(metadata_gpu_buffer.data_ptr(), metadata_host_buffer.data_ptr(), host_buffer_d2h_size, cudaMemcpyHostToDevice, gpu_mgr.worker_stream));
    
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

    cudaCheck(cudaStreamSynchronize(gpu_mgr.worker_stream));
 }

}  // namespace kvcache