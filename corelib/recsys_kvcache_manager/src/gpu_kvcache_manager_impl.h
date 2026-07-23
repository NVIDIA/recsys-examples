/******************************************************************************
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
// #include <torch/extension.h>
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

namespace kvcache {

class GPUKVCacheManagerImpl
{
public:
    GPUKVCacheManagerImpl(
        int num_layers,
        int num_kv_heads,
        int kv_headdim,
        int num_tokens_per_page,
        int num_tokens_per_chunk,
        int num_primary_cache_pages,
        int num_buffer_pages,
        int max_batch_size,
        int max_sequence_length,
        int device_idx);
    ~GPUKVCacheManagerImpl();

    std::vector<at::Tensor> lookup(at::Tensor uids);
    
    int64_t getUIdToEvict(std::unordered_set<int64_t> extra_freezed_uids);
    void evict(int64_t uid);
    void evict_offloaded(int64_t uid);
    void evict_all();
    bool retain(int64_t uid);

    std::vector<int>& alloc_single_sequence(
        int64_t uid, int new_total_length, int host_cached_startpos, int host_cached_length, std::unordered_set<int64_t> freezed_uids);
    void allocate(
        at::Tensor user_ids,
        at::Tensor total_hist_lens,  // all histo w/o candi
        at::Tensor host_cached_lengths,
        at::Tensor page_ids_gpu_buffer,
        at::Tensor metadata_gpu_buffer);
    void revoke_onboard_pages(
        at::Tensor& user_ids,
        at::Tensor& onboard_start_indices,
        at::Tensor& onboard_lengths);

    at::Tensor check_for_offload(at::Tensor& user_ids);
    std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> acquire_offload_pages(
        at::Tensor& user_ids,
        at::Tensor& offloaded_lengths,
        bool always_offload
    );
    void release_offload_pages(
        at::Tensor user_ids,
        at::Tensor offload_start_indices,
        at::Tensor offload_lengths,
        const std::vector<int>& offloaded
    );

public:
    int num_layers;
    int num_kv_heads;
    int kv_headdim;
    int num_tokens_per_page;
    int num_tokens_per_chunk;

    int num_primary_cache_pages;
    int num_buffer_pages;
    int total_offloaded_pages;

    int max_batch_size;
    int max_sequence_length;
    size_t max_offload_pages;

public:
    // kvcache bookkeeping
    std::list<int64_t> _lru_list;
    std::unordered_map<int64_t, 
                       typename std::list<int64_t>::iterator> _lru_lookup_table;
    std::queue<int64_t> _empty_pages;
    std::unordered_map<int64_t, std::vector<int>> _uid_to_page_id;
    std::unordered_map<int64_t, int> _uid_to_paged_cache_startpos;
    std::unordered_map<int64_t, int> _uid_to_paged_cache_length;
    std::unordered_map<int64_t, int> _uid_to_offloaded_length;

public:
    // allocation related
    cudaStream_t alloc_stream;
    void* metadata_host_buffer;   // preallocated pinned host buffer

public:
    std::unordered_map<int64_t, int> _uid_offload_lock;
    std::unordered_set<int64_t> _uid_inference_lock;

public:
    uint16_t *cache_table;
    at::Device device;  // check for c10::Device
};

}  // namespace kvcache