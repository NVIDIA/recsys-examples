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

#include "gpu_kvcache_manager_impl.h"
#include <nvtx3/nvtx3.hpp>

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
  int batch_size,
  int* append_indptr,
  int* seq_lens_ptr,
  int* batch_indices_ptr,
  int* positions_ptr,
  cudaStream_t stream);

namespace kvcache {

GPUKVCacheManagerImpl::GPUKVCacheManagerImpl(
    int num_layers,
    int num_kv_heads,
    int kv_headdim,
    int num_tokens_per_page,
    int num_tokens_per_chunk,
    int num_primary_cache_pages,
    int num_buffer_pages,
    int max_batch_size,
    int max_sequence_length,
    int device_idx
)
    : num_layers(num_layers)
    , num_kv_heads(num_kv_heads)
    , kv_headdim(kv_headdim)
    , num_tokens_per_page(num_tokens_per_page)
    , num_tokens_per_chunk(num_tokens_per_chunk)
    , num_primary_cache_pages(num_primary_cache_pages)
    , num_buffer_pages(num_buffer_pages)
    , total_offloaded_pages(0)
    , max_batch_size(max_batch_size)
    , max_sequence_length(max_sequence_length)
    , device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_idx))
{
    size_t padded_pages_per_seq = (max_sequence_length + num_tokens_per_page - 1) / num_tokens_per_page;
    this->max_offload_pages = max_batch_size * padded_pages_per_seq;

    const c10::cuda::OptionalCUDAGuard device_guard(this->device);

    for (int page_id = 0; page_id < num_primary_cache_pages; page_id++)
        _empty_pages.push(page_id);

    cudaCheck(cudaStreamCreateWithFlags(&alloc_stream, cudaStreamNonBlocking));
    cudaCheck(cudaMallocHost((void**)&this->metadata_host_buffer, 
        (5 * max_batch_size + 4) * sizeof(int) /* for new_history_nnz and new_history_offsets */));
}    

GPUKVCacheManagerImpl::~GPUKVCacheManagerImpl() {
    cudaCheck(cudaFreeHost(this->metadata_host_buffer));
    cudaCheck(cudaStreamDestroy(alloc_stream));
}

std::vector<at::Tensor> GPUKVCacheManagerImpl::lookup(at::Tensor uids) {
    int batch_size = uids.size(0);
    int64_t *user_ids_ptr = uids.data_ptr<int64_t>();
    at::Tensor cached_startpos = at::empty({batch_size}, at::dtype(torch::kInt32).device(at::kCPU));
    at::Tensor cached_lengths = at::empty({batch_size}, at::dtype(torch::kInt32).device(at::kCPU));
    int *cached_startpos_ptr = cached_startpos.data_ptr<int>();
    int *cached_lengths_ptr = cached_lengths.data_ptr<int>();
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids_ptr[seq_idx];
        if (_uid_to_paged_cache_startpos.find(uid) != _uid_to_paged_cache_startpos.end()) {
            cached_startpos_ptr[seq_idx] = _uid_to_paged_cache_startpos[uid];
            cached_lengths_ptr[seq_idx] = _uid_to_paged_cache_length[uid];
        } else {
            cached_startpos_ptr[seq_idx] = 0;
            cached_lengths_ptr[seq_idx] = 0;
        }
    }
    return {cached_startpos, cached_lengths};
};


int64_t GPUKVCacheManagerImpl::getUIdToEvict(std::unordered_set<int64_t> extra_freezed_uids) {
    int num_offloading_uids = _uid_offload_lock.size();
    for (auto it = std::begin(_lru_list); it != std::end(_lru_list); ++it) {
        if (_uid_offload_lock.find((int64_t)*it) != _uid_offload_lock.end())
            continue;
        if (extra_freezed_uids.find((int64_t)*it) != extra_freezed_uids.end())
            continue;
        return *it;
    }
    if (num_offloading_uids == 0) assert(false);
    return _lru_list.front();
};

void GPUKVCacheManagerImpl::evict(int64_t uid) {
    auto const tableIt = _lru_lookup_table.find(uid);
    if (_lru_lookup_table.end() != tableIt) {
        _lru_list.erase(tableIt->second);
        _lru_lookup_table.erase(tableIt);

        for (auto page_id : _uid_to_page_id[uid]) {
            _empty_pages.push(page_id);
        }

        total_offloaded_pages -= std::max(0, 
            (_uid_to_offloaded_length[uid] - _uid_to_paged_cache_startpos[uid]) / this->num_tokens_per_page);

        _uid_to_page_id.erase(uid);
        _uid_to_paged_cache_startpos.erase(uid);
        _uid_to_paged_cache_length.erase(uid);
        _uid_to_offloaded_length.erase(uid);
    }
};

void GPUKVCacheManagerImpl::evict_offloaded(int64_t uid)
{
    int num_offloaded_pages = std::max(0, 
        (_uid_to_offloaded_length[uid] - _uid_to_paged_cache_startpos[uid]) / this->num_tokens_per_page);
    if (num_offloaded_pages == 0) return;

    int num_pages = _uid_to_page_id[uid].size();
    for (int i = 0; i < num_offloaded_pages; i++) {
        _empty_pages.push(_uid_to_page_id[uid][i]);
    }
    _uid_to_page_id[uid].erase(
        _uid_to_page_id[uid].begin(), 
        _uid_to_page_id[uid].begin() + num_offloaded_pages);
    _uid_to_paged_cache_startpos[uid] += num_offloaded_pages * this->num_tokens_per_page;
    _uid_to_paged_cache_length[uid] -= num_offloaded_pages * this->num_tokens_per_page;
    total_offloaded_pages -= num_offloaded_pages;

    // CAVEAT: If _uid_to_page_id[uid].size() == 0 after evict_offloaded, a manual explicit 
    //         eviction of uid is required from the caller. No eviction here for flexible
    //         support for iteration on _lru_list with deletion.
};

void GPUKVCacheManagerImpl::evict_all()
{
    std::queue<int64_t> empty_pages;
    std::swap(_empty_pages, empty_pages);
    _lru_list.clear();
    _lru_lookup_table.clear();
    _uid_to_page_id.clear();
    _uid_to_paged_cache_startpos.clear();
    _uid_to_paged_cache_length.clear();
    _uid_to_offloaded_length.clear();
    _uid_offload_lock.clear();
    total_offloaded_pages = 0;

    for (int page_id = 0; page_id < this->num_primary_cache_pages; page_id++)
        _empty_pages.push(page_id);
};

bool GPUKVCacheManagerImpl::retain(int64_t uid)
{
    auto const tableIt = _lru_lookup_table.find(uid);
    bool found = (_lru_lookup_table.end() != tableIt);
    if (found) {
        _lru_list.erase(tableIt->second);
    }
    _lru_list.push_back(uid);
    _lru_lookup_table[uid] = std::prev(_lru_list.end());
    return found;
};

std::vector<int>& GPUKVCacheManagerImpl::alloc_single_sequence(
    int64_t uid, 
    int new_total_length, 
    int host_cached_startpos, 
    int host_cached_length, 
    std::unordered_set<int64_t> freezed_uids)
{
    (void)host_cached_startpos;  // assume to be zero in current implementation.

    int num_total_pages = int((new_total_length + this->num_tokens_per_page - 1) / this->num_tokens_per_page);  // duplicated computation

    int cur_cached_start = 0;
    int cur_cached_len = 0;
    bool found_in_gpu_cache = retain(uid);
    if (found_in_gpu_cache) {  // duplicated lookup
        cur_cached_start = _uid_to_paged_cache_startpos[uid];
        cur_cached_len = _uid_to_paged_cache_length[uid];
    }

    // 1. allocate pages for host cached data
    int num_onload_pages = 0;
    if (cur_cached_len == 0) {
        num_onload_pages = int((host_cached_length + this->num_tokens_per_page - 1) / this->num_tokens_per_page);
        // Note: We would allow host_cached_length not aligned with this->num_tokens_per_page.
        //       And in this case, num_cur_pages (below) will be 0. Thus, num_append_pages is correct.
    } else if (cur_cached_start > 0) {
        // cur_cached_start should align with this->num_tokens_per_page;
        num_onload_pages = int(cur_cached_start / this->num_tokens_per_page);
    } /* else {
        assert(cur_cached_start == 0 && cur_cached_len > 0);
        num_onload_pages = 0;
    } */
    
    // 2. allocate pages for new data to be appended
    int num_cur_pages = (cur_cached_len + this->num_tokens_per_page - 1) / this->num_tokens_per_page;
    // assert((num_cur_pages == 0 && _uid_to_page_id.find(uid) == _uid_to_page_id.end()) ||
    //        num_cur_pages == _uid_to_page_id[uid].size()                                  );
    int num_append_pages = num_total_pages - num_onload_pages - num_cur_pages;
    
    int num_required_pages = num_onload_pages + num_append_pages;

    // 3. Evict offloaded pages first; then evict by user until enough pages
    {
        // std::unique_lock<std::mutex> lock(offload_freezed_uids_mtx_);

        for (auto it = std::begin(_lru_list); it != std::end(_lru_list); ) {
            if ((size_t)num_required_pages <= _empty_pages.size())
                break;
            if (this->_uid_offload_lock.find((int64_t)*it) != _uid_offload_lock.end()) {
                ++it;
                continue;
            }
            if (freezed_uids.find((int64_t)*it) != freezed_uids.end()) {
                ++it;
                continue;
            }
            evict_offloaded(*it);
            if (_uid_to_paged_cache_length[*it] == 0) {
                auto uid_to_evict = *it;
                auto next_it = _lru_list.erase(it);
                _lru_lookup_table.erase(uid_to_evict);

                _uid_to_page_id.erase(uid_to_evict);
                _uid_to_paged_cache_startpos.erase(uid_to_evict);
                _uid_to_paged_cache_length.erase(uid_to_evict);
                _uid_to_offloaded_length.erase(uid_to_evict);
                it = next_it;
            } else {
                ++it;
            }
        }
    }
    while (num_required_pages > _empty_pages.size()) {
        int64_t uid_to_evict = getUIdToEvict(freezed_uids);
        evict(uid_to_evict);
    }

    std::vector<int> page_ids(num_total_pages);
    for (int i = 0; i < num_onload_pages; i++) {
        page_ids[i] = _empty_pages.front();
        _empty_pages.pop();
    }
    for (int i = num_onload_pages; i < num_onload_pages + num_cur_pages; i++) {
        page_ids[i] = _uid_to_page_id[uid][i - num_onload_pages];
    }
    for (int i = num_onload_pages + num_cur_pages; i < num_total_pages; i++) {
        page_ids[i] = _empty_pages.front();
        _empty_pages.pop();
    }
    _uid_to_page_id[uid] = page_ids;
    _uid_to_paged_cache_startpos[uid] = 0;
    _uid_to_paged_cache_length[uid] = new_total_length;

    return _uid_to_page_id[uid];
};

void GPUKVCacheManagerImpl::allocate(
    at::Tensor user_ids,
    at::Tensor total_hist_lens,  // all histo w/o candi
    at::Tensor host_cached_lengths,
    at::Tensor page_ids_gpu_buffer,
    at::Tensor metadata_gpu_buffer) {
    // TODO(junyiq): page_ids_gpu_buffer &  metadata_gpu_buffer are allocated in main thread
    //               try with pybind gil guard and allocate here in future

    const c10::cuda::OptionalCUDAGuard device_guard(this->device);

    int batch_size = user_ids.size(0);
    int64_t *user_ids_ptr = user_ids.data_ptr<int64_t>();
    int *host_cached_lengths_ptr = host_cached_lengths.data_ptr<int>();

    std::vector<int> page_indices;
    std::vector<int> offload_page_ids;

    int *host_bufptr = static_cast<int*>(this->metadata_host_buffer);

    int *page_indptr = host_bufptr + 0;
    int *last_page_len = host_bufptr + batch_size + 1;
    int *total_history_lengths = host_bufptr + batch_size * 2 + 1;
    int *total_history_offsets = host_bufptr + batch_size * 3 + 1;
    int *new_history_nnz_cuda = host_bufptr + batch_size * 4 + 2;
    int *new_history_offsets = host_bufptr + batch_size * 4 + 3;

    page_indptr[0] = 0;
    total_history_offsets[0] = 0;
    new_history_offsets[0] = 0;

    std::vector<int> cached_lengths;
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids_ptr[seq_idx];
        int gpu_cached_startpos = 0;
        int gpu_cached_length = 0;
        if (_uid_to_paged_cache_startpos.find(uid) != _uid_to_paged_cache_startpos.end()) {
            gpu_cached_startpos = _uid_to_paged_cache_startpos[uid];
            gpu_cached_length = _uid_to_paged_cache_length[uid];
        }
        if (gpu_cached_length > 0 && host_cached_lengths_ptr[seq_idx] < gpu_cached_startpos) {
            evict(uid);
            cached_lengths.push_back(0);
            continue;
        }
        
        cached_lengths.push_back(std::max(
            host_cached_lengths_ptr[seq_idx],
            _uid_to_paged_cache_startpos[uid] + _uid_to_paged_cache_length[uid]
        ));
        // [update offloaded length]
        // Note: host_cached_lengths is always larger
        this->_uid_to_offloaded_length[uid] = host_cached_lengths_ptr[seq_idx];
    }

    const std::unordered_set<int64_t> freezed_uids(user_ids_ptr, user_ids_ptr + batch_size);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids_ptr[seq_idx];
        int total_history_length = total_hist_lens[seq_idx].item<int>();

        // [attention/get metadata]; changed due to no separated onload block
        // Note: allocate pages for onboard kvcache, and revoke onboard pages upon failure.
        std::vector<int>& page_ids = alloc_single_sequence(uid, total_history_length, 0, host_cached_lengths_ptr[seq_idx], freezed_uids);
        page_indices.insert(page_indices.end(), page_ids.begin(), page_ids.end());
        page_indptr[seq_idx + 1] = page_indptr[seq_idx] + page_ids.size();
        last_page_len[seq_idx] = this->_uid_to_paged_cache_length[uid] % this->num_tokens_per_page;  // NOT duplicated lookup, updated
        if (last_page_len[seq_idx] == 0) last_page_len[seq_idx] = this->num_tokens_per_page;
        // int gpu_cache_startpos = this->_uid_to_paged_cache_startpos[uid];  // #1 -- NOT duplicated lookup, updated ; #2 -- assume to be zero after alloc and during inference

        // [history metadata]
        total_history_lengths[seq_idx] = total_history_length;  // used for [appending/put metadata]
        total_history_offsets[seq_idx + 1] = total_history_offsets[seq_idx] + total_history_length;  // used as [attention metadata]: k/v seqlen_offsets (need adding jagged_data.num_candidates_offsets)
        new_history_offsets[seq_idx + 1] = new_history_offsets[seq_idx] + total_history_length - cached_lengths[seq_idx];  // used for [appending/put metadata]
        // old_cached_lengths;  // used in [position encoding metadata]; from lookup results
    }

    cudaCheck(cudaMemcpyAsync(page_ids_gpu_buffer.data_ptr(), page_indices.data(), page_indptr[batch_size] * sizeof(int), cudaMemcpyHostToDevice, this->alloc_stream));

    // [appending/put metadata; for cudagraph only]
    auto new_tokens = new_history_offsets[batch_size];
    *new_history_nnz_cuda = new_tokens;

    size_t host_buffer_d2h_size = (batch_size * 5 + 4) * sizeof(int);
    cudaCheck(cudaMemcpyAsync(metadata_gpu_buffer.data_ptr(), this->metadata_host_buffer, host_buffer_d2h_size, cudaMemcpyHostToDevice, this->alloc_stream));
    
    // [appending/put metadata]
    int *gpu_bufptr = metadata_gpu_buffer.data_ptr<int>();
    int *total_history_lengths_dev = gpu_bufptr + batch_size * 2 + 1;
    int *new_history_offsets_dev = gpu_bufptr + batch_size * 4 + 3;
    int *batch_indices_dev = gpu_bufptr + batch_size * 5 + 4;
    int *position_dev = gpu_bufptr + batch_size * 5 + 4 + new_tokens;

    GetPagedBatchIndicesPositions(
        batch_size,
        new_history_offsets_dev,  // new_history_offsets
        total_history_lengths_dev,  // total_history_lengths
        batch_indices_dev,
        position_dev,
        this->alloc_stream
    );

    cudaCheck(cudaStreamSynchronize(this->alloc_stream));
}

at::Tensor GPUKVCacheManagerImpl::check_for_offload(
    at::Tensor& user_ids) {
    const auto batch_size = user_ids.size(0);

    std::vector<int64_t> offload_user_ids;
    std::unordered_set<int64_t> offload_uids_set;

    int num_pages_to_offload = 0;
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids.data_ptr<int64_t>()[seq_idx];
        int64_t offloaded_length = _uid_to_offloaded_length[uid];

        int cached_end_index = this->_uid_to_paged_cache_startpos[uid] + this->_uid_to_paged_cache_length[uid];
        if (cached_end_index - offloaded_length >= this->num_tokens_per_chunk) {
            offload_user_ids.push_back(uid);
            offload_uids_set.insert(uid);
        }

        num_pages_to_offload += (cached_end_index - offloaded_length) / this->num_tokens_per_page;
    }

    for (auto it = std::begin(_lru_list); it != std::end(_lru_list); ++it) {
        int64_t uid = *it;
        if (this->total_offloaded_pages + num_pages_to_offload + this->_empty_pages.size() > this->num_buffer_pages) {
            break;
        }
        if (offload_uids_set.find(uid) != offload_uids_set.end()) continue;

        int offloaded_length = this->_uid_to_offloaded_length[uid];
        int cached_startpos = this->_uid_to_paged_cache_startpos[uid];
        if (offloaded_length < cached_startpos) continue;  // should not happen

        int cached_end_index = cached_startpos + this->_uid_to_paged_cache_length[uid];
        if (cached_end_index - offloaded_length >= this->num_tokens_per_chunk) {
            num_pages_to_offload += (cached_end_index - offloaded_length) / this->num_tokens_per_page;
            if ((size_t)num_pages_to_offload > this->max_offload_pages) {
                break;
            }   
            offload_user_ids.push_back(uid);
            offload_uids_set.insert(uid);
        }
    }
    return at::from_blob(offload_user_ids.data(), {offload_user_ids.size()}, at::dtype(torch::kInt64)).clone();
}

void GPUKVCacheManagerImpl::revoke_onboard_pages(
    at::Tensor& user_ids,
    at::Tensor& onboard_start_indices,
    at::Tensor& onboard_lengths
) {
    const auto batch_size = user_ids.size(0);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids[seq_idx].item<int64_t>();
        int onboard_startpos = onboard_start_indices[seq_idx].item<int>();
        int onboard_length = onboard_lengths[seq_idx].item<int>();

        // Simplified impl. 
        // Assume: _uid_to_paged_cache_startpos[uid] is always 0, 
        //         onboard_startpos is always 0, and onboard_length are always aligned.
        // In this case, revoke_page_start is zero, and there will be no gap in cached pages.
        int revoke_page_start = onboard_startpos / this->num_tokens_per_page;
        int revoke_page_end = (onboard_startpos + onboard_length) / this->num_tokens_per_page;
        int num_revoke_pages = revoke_page_end - revoke_page_start;

        for (int jdx = revoke_page_start; jdx < revoke_page_end; jdx++) {
            _empty_pages.push(_uid_to_page_id[uid][jdx]);
        }
        _uid_to_paged_cache_startpos[uid] += num_revoke_pages * this->num_tokens_per_page;
        _uid_to_paged_cache_length[uid] -= num_revoke_pages * this->num_tokens_per_page;
        _uid_to_page_id[uid].erase(
            _uid_to_page_id[uid].begin() + revoke_page_start, 
            _uid_to_page_id[uid].begin() + revoke_page_start + num_revoke_pages);
        if (_uid_to_page_id[uid].size() == 0) {
            evict(uid);
        }
    }
}

std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> GPUKVCacheManagerImpl::acquire_offload_pages(
    at::Tensor& user_ids,
    at::Tensor& offloaded_lengths,
    bool always_offload) {
    const auto batch_size = user_ids.size(0);

    if (always_offload) {
        std::vector<int> offload_startpos(user_ids.size(0), 0);
        std::vector<at::Tensor> offload_page_ids_list;
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            int64_t uid = user_ids[seq_idx].item<int64_t>();
            if (this->_uid_to_page_id.find(uid) == this->_uid_to_page_id.end()) {
                offload_page_ids_list.push_back(at::empty({0}, at::dtype(torch::kInt32)));
                continue;
            }

            offload_page_ids_list.push_back(at::from_blob(
                this->_uid_to_page_id[uid].data(), {this->_uid_to_page_id[uid].size()}, at::dtype(torch::kInt32)
            ));
            this->_uid_offload_lock[uid] += 1;
        }
        return std::make_tuple(
            user_ids.clone(),
            at::from_blob(offload_startpos.data(), {offload_startpos.size()}, at::dtype(torch::kInt32)).clone(),
            offload_page_ids_list
        );
    }


    std::vector<int64_t> offload_user_ids;
    std::vector<int> offload_startpos;
    std::vector<at::Tensor> offload_page_ids_list;

    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids[seq_idx].item<int64_t>();
        if (offloaded_lengths.size(0) == batch_size) {
            this->_uid_to_offloaded_length[uid] = offloaded_lengths[seq_idx].item<int>();
        }
        int offloaded_length = this->_uid_to_offloaded_length[uid];

        int cached_startpos = this->_uid_to_paged_cache_startpos[uid];
        // if (offloaded_length < cached_startpos) continue;  // Should not have gap

        int cached_end_index = cached_startpos + this->_uid_to_paged_cache_length[uid];
        if (cached_end_index - offloaded_length >= this->num_tokens_per_chunk) {
            offload_user_ids.push_back(uid);
            offload_startpos.push_back(offloaded_length);

            const int pages_offload_start = (offloaded_length - cached_startpos) / this->num_tokens_per_page;
            const int pages_offload_num = (cached_end_index - offloaded_length) / this->num_tokens_per_page;
            offload_page_ids_list.push_back(
                at::from_blob(
                    this->_uid_to_page_id[uid].data() + pages_offload_start, 
                    {pages_offload_num}, 
                    at::dtype(torch::kInt32)
                )  // this is a slice of the _uid_to_page_id[uid] in gpu_kvcache_mgr, which will be locked during offloading.
            );
        }
        this->_uid_offload_lock[uid] += 1;
    }

    return std::make_tuple(
        at::from_blob(offload_user_ids.data(), {offload_user_ids.size()}, at::dtype(torch::kInt64)).clone(),
        at::from_blob(offload_startpos.data(), {offload_startpos.size()}, at::dtype(torch::kInt32)).clone(),
        offload_page_ids_list
    );
}


void GPUKVCacheManagerImpl::release_offload_pages(
    at::Tensor user_ids,
    at::Tensor offload_start_indices,
    at::Tensor offload_lengths,
    const std::vector<int>& offloaded) {
    for (auto idx = 0; idx < user_ids.size(0); idx++) {
        int64_t uid = user_ids[idx].item<int64_t>();
        if (this->_uid_offload_lock.find(uid) == this->_uid_offload_lock.end())
            continue;
        
        this->_uid_offload_lock[uid] -= 1;
        if (this->_uid_offload_lock[uid] <= 0) {
            this->_uid_offload_lock.erase(uid);
        }
        if (offloaded[idx]) {
            total_offloaded_pages -= std::max(0,
                (_uid_to_offloaded_length[uid] - _uid_to_paged_cache_startpos[uid]) / this->num_tokens_per_page);
            _uid_to_offloaded_length[uid] = offload_start_indices[idx].item<int>() + offload_lengths[idx].item<int>();
            total_offloaded_pages += std::max(0,
                (_uid_to_offloaded_length[uid] - _uid_to_paged_cache_startpos[uid]) / this->num_tokens_per_page);
        }
    }
    // TODO(junyiq): in future acq and rls pages by per-page lock.
    // When offloaded == true, offload_start_indices + offload_lengths can be used as already offloaded pages.
    // When offloaded == false, offload_lengths can be empty.
}


}  // namespace kvcache