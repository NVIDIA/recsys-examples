/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
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
******************************************************************************/

#pragma once

#include "types.cuh"
#include <cub/cub.cuh>

#include <cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace dyn_emb {

template <int ThreadBlockDim_, int ProbingGroupSize_, int ReductionGroupSize_,
          int CompactTileSize_, int NumScorePerThread_>
struct InsertKernelTraits {
  static constexpr int ThreadBlockDim = ThreadBlockDim_;
  static constexpr int ProbingGroupSize = ProbingGroupSize_;
  static constexpr int ReductionGroupSize = ReductionGroupSize_;
  static constexpr int CompactTileSize = CompactTileSize_;
  static constexpr int NumScorePerThread = NumScorePerThread_;
};

template <bool Pred> struct ExportPredFunctor {
  ScoreType threshold;
  ExportPredFunctor(ScoreType threshold) : threshold(threshold) {}

  __forceinline__ __device__ bool operator()(const ScoreType score) {
    if constexpr (Pred) {
      return score >= threshold;
    } else {
      return true;
    }
  }
};

// Increment the counter when matched
struct EvalAndCount {
  ScoreType threshold;
  CounterType *d_counter;

  EvalAndCount(ScoreType threshold, CounterType *d_counter)
      : threshold(threshold), d_counter(d_counter) {}

  template <int GroupSize>
  __forceinline__ __device__ void
  operator()(const ScoreType score, cg::thread_block_tile<GroupSize> &g,
             bool valid) {

    bool match = valid && (score >= threshold);

    uint32_t vote = g.ballot(match);
    int group_cnt = __popc(vote);
    if (g.thread_rank() == 0) {
      atomicAdd(d_counter, static_cast<CounterType>(group_cnt));
    }
  }
};

template <typename Table, int ProbingGroupSize>
__global__ void
table_lookup_kernel(Table table, int64_t batch,
                    typename Table::KeyType const *__restrict__ input_keys,
                    bool *__restrict__ founds, IndexType *__restrict__ indices,
                    ScoreType *__restrict__ scores, ScorePolicyType policy_type,
                    bool return_scores) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = input_keys[i];
    ScoreType score = ScorePolicy::get(policy_type, scores, i);

    Bucket bucket;
    KeyType hashcode = KeyType();
    int64_t bucket_id;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      uint64_t global_idx = static_cast<uint64_t>(hashcode % table.capacity());
      bucket_id = global_idx / table.bucket_capacity();
      // bucket_id = (hashcode % table.capacity()) / table.bucket_capacity();
      bucket = table[bucket_id];
    }
    Iter iter = Iter(hashcode % table.bucket_capacity());
    int step = 0;
    auto probe_res = bucket.probe<ProbingGroupSize>(key, iter, step);
    bool found = probe_res == Bucket::ProbeResult::Existed;
    IndexType index = -1;
    if (found) {

      if (policy_type == ScorePolicyType::Const) {
        score = *bucket.scores(iter);
      } else {
        KeyType expected_key = key;
        if (bucket.try_lock(iter, expected_key)) {
          ScorePolicy::update(policy_type, return_scores, bucket.scores(iter),
                              score);
          bucket.unlock(iter, key);
        } else {
          found = false; // only one update will succeed for duplicated keys.
          score = ScoreType();
        }
      }

      if (found) {
        index = bucket_id * bucket.capacity() + iter;
      }
    }
    ScorePolicy::set(return_scores, scores, i, score);
    if (founds) {
      founds[i] = found;
    }
    if (indices) {
      indices[i] = index;
    }
  }
}

template <typename Table, typename KernelTraits>
__global__ void
table_insert_kernel(Table table, int *__restrict__ bucket_sizes, int64_t batch,
                    typename Table::KeyType const *__restrict__ input_keys,
                    InsertResult *__restrict__ insert_results,
                    IndexType *__restrict__ indices,
                    ScoreType *__restrict__ scores, ScorePolicyType policy_type,
                    bool return_scores,
                    typename Table::KeyType **__restrict__ table_key_slots) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;
  using ProbeResult = typename Bucket::ProbeResult;

  static constexpr int BlockSize = KernelTraits::ThreadBlockDim;
  static constexpr int BufferDim = KernelTraits::NumScorePerThread;

  static constexpr int ProbingGroupSize = KernelTraits::ProbingGroupSize;
  static constexpr int ReductionGroupSize = KernelTraits::ReductionGroupSize;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  __shared__ ScoreType sm_scores[BlockSize * BufferDim];
  // extern __shared__ ScoreType sm_scores[];
  // cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = input_keys[i];
    ScoreType score = ScorePolicy::get(policy_type, scores, i);

    InsertResult result = InsertResult::Init;

    Bucket bucket = Bucket();
    KeyType hashcode = KeyType();
    uint64_t bucket_id;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      uint64_t global_idx = static_cast<uint64_t>(hashcode % table.capacity());
      bucket_id = global_idx / table.bucket_capacity();
      // bucket_id = (hashcode % table.capacity()) / table.bucket_capacity();
      bucket = table[bucket_id];
    } else {
      result = InsertResult::Illegal;
    }
    Iter iter = Iter(hashcode % table.bucket_capacity());
    ProbeResult probe_res = ProbeResult::Init;
    int step = 0;
    while (step != bucket.capacity()) {
      probe_res = bucket.probe<ProbingGroupSize>(key, iter, step);
      if (probe_res == ProbeResult::Existed) {
        KeyType expected_key = key;

        if (bucket.try_lock(iter, expected_key)) {
          result = InsertResult::Assign;
          // bucket.unlock(iter, key); // will not unlock, to avoid 2 threads
          // got the same slot.
        } // else: the key is evicted from the bucket(full), try to reintert by
          // eviction including reclaimed key.
        break;
      }
      if (probe_res == ProbeResult::Empty) {
        KeyType expected_key = Bucket::empty_key();

        if (bucket.try_lock(iter, expected_key)) {
          *bucket.digests(iter) = Bucket::key_to_digest(key);
          atomicAdd(&bucket_sizes[bucket_id], 1);
          result = InsertResult::Insert;
          break;
        } // else it was locked by another thread.
      }

      if (probe_res == ProbeResult::Failed) {
        result = InsertResult::Illegal;
        break;
      }
    }

    while (result == InsertResult::Init) {

      KeyType evict_key;
      ScoreType evict_score =
          ScorePolicy::score_for_compare(policy_type, score);

      bool succeed = bucket.template reduce<ReductionGroupSize, BufferDim>(
          iter, evict_key, evict_score, sm_scores);

      if (succeed) {

        if (bucket.try_lock(iter, evict_key)) {
          if (*bucket.scores(iter) != evict_score) {
            // that means when reduce we got a new key but old score.
            bucket.unlock(iter, evict_key);
          } else {
            *bucket.digests(iter) = Bucket::key_to_digest(key);
            if (evict_key == Bucket::reclaimed_key()) {
              atomicAdd(bucket_sizes + bucket_id, 1);
              result = InsertResult::Reclaim;
            } else {
              *bucket.scores(iter) = ScoreType();
              result = InsertResult::Evict;
            }
            break;
          }
        } // else it was locked by another thread.
      } else {
        result = InsertResult::Busy;
        break;
      }
    }

    IndexType index = -1;
    KeyType *table_key_slot = nullptr;
    if (isInsertSuccess(result)) {
      ScorePolicy::update(policy_type, return_scores, bucket.scores(iter),
                          score);
      index = bucket_id * bucket.capacity() + iter;
      table_key_slot = bucket.keys(iter);
    }
    ScorePolicy::set(return_scores, scores, i, score);
    // TODO: unlock using index.
    table_key_slots[i] = table_key_slot;
    if (indices) {
      indices[i] = index;
    }
    if (insert_results) {
      insert_results[i] = result;
    }
  }
}

template <typename Table, typename KernelTraits>
__global__ void table_insert_and_evict_kernel(
    Table table, int *__restrict__ bucket_sizes, int64_t batch,
    typename Table::KeyType const *__restrict__ input_keys,
    InsertResult *__restrict__ insert_results, IndexType *__restrict__ indices,
    ScoreType *__restrict__ scores, ScorePolicyType policy_type,
    bool return_scores, typename Table::KeyType **__restrict__ table_key_slots,
    CounterType *evicted_counter,
    typename Table::KeyType *__restrict__ evicted_keys,
    ScoreType *__restrict__ evicted_scores,
    IndexType *__restrict__ evicted_indices) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;
  using ProbeResult = typename Bucket::ProbeResult;

  static constexpr int BlockSize = KernelTraits::ThreadBlockDim;
  static constexpr int BufferDim = KernelTraits::NumScorePerThread;

  static constexpr int ProbingGroupSize = KernelTraits::ProbingGroupSize;
  static constexpr int ReductionGroupSize = KernelTraits::ReductionGroupSize;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  __shared__ ScoreType sm_scores[BlockSize * BufferDim];

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = input_keys[i];
    ScoreType score = ScorePolicy::get(policy_type, scores, i);

    InsertResult result = InsertResult::Init;

    Bucket bucket = Bucket();
    KeyType hashcode = KeyType();
    int64_t bucket_id;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      uint64_t global_idx = static_cast<uint64_t>(hashcode % table.capacity());
      bucket_id = global_idx / table.bucket_capacity();
      // bucket_id = (hashcode % table.capacity()) / table.bucket_capacity();
      bucket = table[bucket_id];
    } else {
      result = InsertResult::Illegal;
    }
    Iter iter = Iter(hashcode % table.bucket_capacity());
    ProbeResult probe_res = ProbeResult::Init;
    int step = 0;
    while (step != bucket.capacity()) {
      probe_res = bucket.probe<ProbingGroupSize>(key, iter, step);
      if (probe_res == ProbeResult::Existed) {
        KeyType expected_key = key;
        if (bucket.try_lock(iter, expected_key)) {
          result = InsertResult::Assign;
          // bucket.unlock(iter, key); // will not unlock, to avoid 2 threads
          // got the same slot.
        } // else: the key is evicted from the bucket(full), try to reintert by
          // eviction including reclaimed key.
        break;
      }
      if (probe_res == ProbeResult::Empty) {
        KeyType expected_key = Bucket::empty_key();
        if (bucket.try_lock(iter, expected_key)) {
          *bucket.digests(iter) = Bucket::key_to_digest(key);
          atomicAdd(&bucket_sizes[bucket_id], 1);
          result = InsertResult::Insert;
          break;
        } // else it was locked by another thread.
      }

      if (probe_res == ProbeResult::Failed) {
        result = InsertResult::Illegal;
        break;
      }
    }

    KeyType evict_key;
    ScoreType evict_score;

    while (result == InsertResult::Init) {

      evict_score = ScorePolicy::score_for_compare(policy_type, score);
      bool succeed = bucket.template reduce<ReductionGroupSize, BufferDim>(
          iter, evict_key, evict_score, sm_scores);

      if (succeed) {

        if (bucket.try_lock(iter, evict_key)) {
          if (*bucket.scores(iter) != evict_score) {
            // that means when reduce we got a new key but old score.
            bucket.unlock(iter, evict_key);
          } else {
            *bucket.digests(iter) = Bucket::key_to_digest(key);
            if (evict_key == Bucket::reclaimed_key()) {
              atomicAdd(&bucket_sizes[bucket_id], 1);
              result = InsertResult::Reclaim;
            } else {
              *bucket.scores(iter) = ScoreType();
              result = InsertResult::Evict;
            }
            break;
          }
        } // else it was locked by another thread.
      } else {
        result = InsertResult::Busy;
        evict_key = key;
        evict_score = score;
        break;
      }
    }

    auto g = cg::tiled_partition<KernelTraits::CompactTileSize>(
        cg::this_thread_block());
    bool evicted =
        (result == InsertResult::Evict or result == InsertResult::Busy) ? true
                                                                        : false;
    uint32_t vote = g.ballot(evicted);
    int group_cnt = __popc(vote);
    CounterType group_offset = 0;
    if (g.thread_rank() == 0) {
      group_offset =
          atomicAdd(evicted_counter, static_cast<CounterType>(group_cnt));
    }
    group_offset = g.shfl(group_offset, 0);

    int previous_cnt = group_cnt - __popc(vote >> g.thread_rank());
    int64_t out_id = group_offset + previous_cnt;

    if (evicted) {
      evicted_keys[out_id] = evict_key;
      if (evicted_scores) {
        evicted_scores[out_id] = evict_score;
      }
      if (evicted_indices) {
        IndexType index;
        if (result == InsertResult::Evict) {
          index = bucket_id * bucket.capacity() + iter;
        } else {
          index = -1;
        }
        evicted_indices[out_id] = index;
      }
    }

    IndexType index = -1;
    KeyType *table_key_slot = nullptr;
    if (isInsertSuccess(result)) {
      ScorePolicy::update(policy_type, return_scores, bucket.scores(iter),
                          score);
      index = bucket_id * bucket.capacity() + iter;
      table_key_slot = bucket.keys(iter);
    }
    ScorePolicy::set(return_scores, scores, i, score);
    table_key_slots[i] = table_key_slot;
    if (indices) {
      indices[i] = index;
    }
    if (insert_results) {
      insert_results[i] = result;
    }
  }
}

template <typename Table>
__global__ void
table_unlock_kernel(Table table, int64_t batch,
                    typename Table::KeyType const *__restrict__ input_keys,
                    typename Table::KeyType **__restrict__ table_key_slots) {
  using KeyType = typename Table::KeyType;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {
    KeyType key = input_keys[i];
    KeyType *key_slot = table_key_slots[i];
    if (key_slot) {
      *key_slot = key;
    }
  }
}

template <typename Table, int ProbingGroupSize>
__global__ void
table_erase_kernel(Table table, int *__restrict__ bucket_sizes, int64_t batch,
                   typename Table::KeyType const *__restrict__ input_keys,
                   IndexType *__restrict__ indices) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = input_keys[i];

    Bucket bucket;
    KeyType hashcode = KeyType();
    int64_t bucket_id;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      uint64_t global_idx = static_cast<uint64_t>(hashcode % table.capacity());
      bucket_id = global_idx / table.bucket_capacity();
      // bucket_id = (hashcode % table.capacity()) / table.bucket_capacity();
      bucket = table[bucket_id];
    }
    Iter iter = Iter(hashcode % table.bucket_capacity());
    int step = 0;
    auto probe_res = bucket.probe<ProbingGroupSize>(key, iter, step);
    bool found = probe_res == Bucket::ProbeResult::Existed;
    IndexType index = -1;
    if (found) {

      KeyType expected_key = key;
      if (bucket.try_lock(iter, expected_key)) {
        *bucket.scores(iter) = ScoreType();
        *bucket.digests(iter) = Bucket::empty_digest();

        bucket.unlock(iter, Bucket::reclaimed_key());
        atomicSub(bucket_sizes + bucket_id, 1);
      } else {
        found = false; // only one update will succeed for duplicated keys.
      }

      if (found) {
        index = bucket_id * bucket.capacity() + iter;
      }
    }
    if (indices) {
      indices[i] = index;
    }
  }
}

template <typename Table, typename PredFunctor, int TileSize>
__global__ void
table_export_batch_kernel(Table table, IndexType begin, IndexType end,
                          CounterType *__restrict__ counter,
                          typename Table::KeyType *__restrict__ keys,
                          ScoreType *__restrict__ scores, PredFunctor pred,
                          IndexType *__restrict__ indices) {
  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  auto g = cg::tiled_partition<TileSize>(cg::this_thread_block());

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = begin + tid; i < end; i += gridDim.x * blockDim.x) {

    int64_t bucket_id = i / table.bucket_capacity();

    Bucket bucket = table[bucket_id];

    Iter iter = Iter(i % bucket.capacity());

    const KeyType key = *bucket.keys(iter);
    const ScoreType score = *bucket.scores(iter);
    const IndexType index = i;

    bool valid = Bucket::is_valid(key);
    bool match = valid and pred.template operator()(score);
    // bool match = valid and pred(score);
    uint32_t vote = g.ballot(match);
    int group_cnt = __popc(vote);
    CounterType group_offset = 0;
    if (g.thread_rank() == 0) {
      group_offset = atomicAdd(counter, static_cast<CounterType>(group_cnt));
    }
    group_offset = g.shfl(group_offset, 0);

    int previous_cnt = group_cnt - __popc(vote >> g.thread_rank());
    int64_t out_id = group_offset + previous_cnt;

    if (match) {
      keys[out_id] = key;
      if (scores) {
        scores[out_id] = score;
      }
      if (indices) {
        indices[out_id] = index;
      }
    }
  }
}

template <typename Table, typename ExecFunctor, int TileSize>
__global__ void table_traverse_kernel(Table table, IndexType begin,
                                      IndexType end, ExecFunctor f) {
  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  cg::thread_block_tile<TileSize> g =
      cg::tiled_partition<TileSize>(cg::this_thread_block());

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = begin + tid; i < end; i += gridDim.x * blockDim.x) {

    int64_t bucket_id = i / table.bucket_capacity();

    Bucket bucket = table[bucket_id];

    Iter iter = Iter(i % bucket.capacity());

    const KeyType key = *bucket.keys(iter);
    const ScoreType score = *bucket.scores(iter);

    bool valid = Bucket::is_valid(key);
    f.template operator()<TileSize>(score, g, valid);
  }
}

} // namespace dyn_emb