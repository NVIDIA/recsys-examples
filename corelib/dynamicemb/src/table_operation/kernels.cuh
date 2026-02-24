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
          int CompactTileSize_, int NumScorePerThread_,
          ScorePolicyType PolicyType_ = ScorePolicyType::Const,
          bool OutputScore_ = false>
struct InsertKernelTraits {
  static constexpr int ThreadBlockDim = ThreadBlockDim_;
  static constexpr int ProbingGroupSize = ProbingGroupSize_;
  static constexpr int ReductionGroupSize = ReductionGroupSize_;
  static constexpr int CompactTileSize = CompactTileSize_;
  static constexpr int NumScorePerThread = NumScorePerThread_;
  static constexpr ScorePolicyType PolicyType = PolicyType_;
  static constexpr bool OutputScore = OutputScore_;
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

template <typename Table, int ProbingGroupSize, ScorePolicyType PolicyType>
__global__ void
table_lookup_kernel(Table table,
                    int64_t const *__restrict__ table_bucket_offsets,
                    int64_t batch,
                    typename Table::KeyType const *__restrict__ input_keys,
                    int64_t const *__restrict__ table_ids,
                    bool *__restrict__ founds, IndexType *__restrict__ indices,
                    ScoreType *__restrict__ score_input,
                    int64_t *__restrict__ score_output) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = input_keys[i];
    ScoreType score = ScorePolicy<PolicyType>::get(score_input, i);

    int64_t hashcode = 0;
    int64_t bucket_id = 0;
    int64_t table_cap = 0;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      int64_t t_id = table_ids[i];
      int64_t bkt_begin = table_bucket_offsets[t_id];
      int64_t bkt_end = table_bucket_offsets[t_id + 1];
      table_cap = (bkt_end - bkt_begin) * table.bucket_capacity();
      if (table_cap > 0) {
        int64_t local_idx = hashcode % table_cap;
        bucket_id = bkt_begin + local_idx / table.bucket_capacity();
      }
    }
    if (table_cap == 0) {
      score_output[i] = static_cast<int64_t>(score);
      founds[i] = false;
      indices[i] = -1;
      continue;
    }
    Bucket bucket = table[bucket_id];
    Iter iter = Iter(hashcode % table.bucket_capacity());
    int64_t step = 0;
    auto probe_res = bucket.probe<ProbingGroupSize>(key, iter, step);
    bool found = probe_res == Bucket::ProbeResult::Existed;
    IndexType index = -1;
    if (found) {
      if constexpr (PolicyType == ScorePolicyType::Const) {
        score = *bucket.scores(iter);
      } else {
        KeyType expected_key = key;
        if (bucket.try_lock(iter, expected_key)) {
          score = ScorePolicy<PolicyType>::update(bucket.scores(iter), score);
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
    score_output[i] = static_cast<int64_t>(score);
    founds[i] = found;
    indices[i] = index;
  }
}

template <int ProbingGroupSize, typename Bucket, typename KeyType>
__forceinline__ __device__ void insert_probe(
    Bucket &bucket, KeyType key, int *__restrict__ bucket_sizes, int64_t bucket_id,
    typename Bucket::Iterator iter_in,
    typename Bucket::Iterator *iter_out, InsertResult *result_out) {
  auto iter = iter_in;
  auto result = InsertResult::Init;
  using ProbeResult = typename Bucket::ProbeResult;
  ProbeResult probe_res = ProbeResult::Init;
  int64_t step = 0;
  while (step != bucket.capacity()) {
    probe_res = bucket.template probe<ProbingGroupSize>(key, iter, step);
    if (probe_res == ProbeResult::Existed) {
      KeyType expected_key = key;
      if (bucket.try_lock(iter, expected_key)) {
        result = InsertResult::Assign;
      }
      break;
    }
    if (probe_res == ProbeResult::Empty) {
      KeyType expected_key = Bucket::empty_key();
      if (bucket.try_lock(iter, expected_key)) {
        *bucket.digests(iter) = Bucket::key_to_digest(key);
        atomicAdd(&bucket_sizes[bucket_id], 1);
        result = InsertResult::Insert;
        break;
      }
    }
    if (probe_res == ProbeResult::Failed) {
      result = InsertResult::Illegal;
      break;
    }
  }
  *iter_out = iter;
  *result_out = result;
}

template <int ReductionGroupSize, int BufferDim, typename Policy, typename Bucket,
          typename KeyType>
__forceinline__ __device__ void insert(
    Bucket &bucket, KeyType key, ScoreType score,
    int *__restrict__ bucket_sizes, int64_t bucket_id,
    typename Bucket::Iterator iter_in, ScoreType *sm_scores,
    InsertResult result_in, typename Bucket::Iterator *iter_out,
    InsertResult *result_out, KeyType *evict_key_out,
    ScoreType *evict_score_out) {
  auto iter = iter_in;
  auto result = result_in;
  while (result == InsertResult::Init) {
    KeyType evict_key;
    ScoreType evict_score = Policy::score_for_compare(score);

    bool succeed = bucket.template reduce<ReductionGroupSize, BufferDim>(
        iter, evict_key, evict_score, sm_scores);

    if (succeed) {
      if (bucket.try_lock(iter, evict_key)) {
        if (*bucket.scores(iter) != evict_score) {
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
          if (evict_key_out) {
            *evict_key_out = evict_key;
          }
          if (evict_score_out) {
            *evict_score_out = evict_score;
          }
          break;
        }
      }
    } else {
      result = InsertResult::Busy;
      if (evict_key_out) {
        *evict_key_out = key;
      }
      if (evict_score_out) {
        *evict_score_out = score;
      }
      break;
    }
  }
  *iter_out = iter;
  *result_out = result;
}

template <int CompactTileSize, typename Bucket, typename KeyType>
__forceinline__ __device__ void insert_evict(
    InsertResult result, int64_t i, int64_t bucket_id, Bucket &bucket,
    typename Bucket::Iterator iter, KeyType evict_key, ScoreType evict_score,
    int64_t const *__restrict__ table_ids, CounterType *evicted_counter,
    KeyType *__restrict__ evicted_keys, int64_t *__restrict__ evicted_scores,
    IndexType *__restrict__ evicted_indices,
    int64_t *__restrict__ evicted_table_ids) {
  auto g = cg::tiled_partition<CompactTileSize>(cg::this_thread_block());
  bool evicted =
      (result == InsertResult::Evict or result == InsertResult::Busy) ? true
                                                                      : false;
  uint32_t vote = g.ballot(evicted);
  int group_cnt = __popc(vote);
  CounterType group_offset = 0;
  if (g.thread_rank() == 0) {
    group_offset = atomicAdd(evicted_counter, static_cast<CounterType>(group_cnt));
  }
  group_offset = g.shfl(group_offset, 0);

  int previous_cnt = group_cnt - __popc(vote >> g.thread_rank());
  int64_t out_id = group_offset + previous_cnt;

  if (evicted) {
    evicted_keys[out_id] = evict_key;
    evicted_scores[out_id] = static_cast<int64_t>(evict_score);
    IndexType index = (result == InsertResult::Evict)
                          ? bucket_id * bucket.capacity() + iter
                          : -static_cast<IndexType>(i + 1);
    evicted_indices[out_id] = index;
    evicted_table_ids[out_id] = table_ids[i];
  }
}

template <typename Table, typename KernelTraits>
__global__ void table_insert_kernel(
    Table table,
    int64_t const *__restrict__ table_bucket_offsets,
    int *__restrict__ bucket_sizes, int64_t batch,
    typename Table::KeyType const *__restrict__ input_keys,
    int64_t const *__restrict__ table_ids,
    InsertResult *__restrict__ insert_results, IndexType *__restrict__ indices,
    ScoreType *__restrict__ score_input, int64_t *__restrict__ score_output,
    typename Table::KeyType **__restrict__ table_key_slots) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  static constexpr int BlockSize = KernelTraits::ThreadBlockDim;
  static constexpr int BufferDim = KernelTraits::NumScorePerThread;

  static constexpr int ProbingGroupSize = KernelTraits::ProbingGroupSize;
  static constexpr int ReductionGroupSize = KernelTraits::ReductionGroupSize;
  static constexpr ScorePolicyType PolicyType = KernelTraits::PolicyType;
  static constexpr bool OutputScore = KernelTraits::OutputScore;

  using Policy = ScorePolicy<PolicyType>;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  __shared__ ScoreType sm_scores[BlockSize * BufferDim];

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = input_keys[i];
    ScoreType score = Policy::get(score_input, i);

    int64_t hashcode = 0;
    int64_t bucket_id = 0;
    int64_t table_cap = 0;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      int64_t t_id = table_ids[i];
      int64_t bkt_begin = table_bucket_offsets[t_id];
      int64_t bkt_end = table_bucket_offsets[t_id + 1];
      table_cap = (bkt_end - bkt_begin) * table.bucket_capacity();
      if (table_cap > 0) {
        int64_t local_idx = hashcode % table_cap;
        bucket_id = bkt_begin + local_idx / table.bucket_capacity();
      }
    }
    if (table_cap == 0) {
      if constexpr (OutputScore) {
        score_output[i] = static_cast<int64_t>(score);
      }
      table_key_slots[i] = nullptr;
      indices[i] = -1;
      if (insert_results) {
        insert_results[i] = InsertResult::Illegal;
      }
      continue;
    }

    InsertResult result;
    Bucket bucket = table[bucket_id];
    Iter iter = Iter(hashcode % table.bucket_capacity());
    insert_probe<ProbingGroupSize>(
        bucket, key, bucket_sizes, bucket_id, iter, &iter, &result);
    insert<ReductionGroupSize, BufferDim, Policy>(
        bucket, key, score, bucket_sizes, bucket_id, iter, sm_scores, result,
        &iter, &result, static_cast<KeyType *>(nullptr),
        static_cast<ScoreType *>(nullptr));

    IndexType index = -1;
    KeyType *table_key_slot = nullptr;
    if (isInsertSuccess(result)) {
      score = Policy::update(bucket.scores(iter), score);
      index = bucket_id * bucket.capacity() + iter;
      table_key_slot = bucket.keys(iter);
    }
    if constexpr (OutputScore) {
      score_output[i] = static_cast<int64_t>(score);
    }
    table_key_slots[i] = table_key_slot;
    indices[i] = index;
    if (insert_results) {
      insert_results[i] = result;
    }
  }
}

template <typename Table, typename KernelTraits>
__global__ void table_insert_and_evict_kernel(
    Table table,
    int64_t const *__restrict__ table_bucket_offsets,
    int *__restrict__ bucket_sizes, int64_t batch,
    typename Table::KeyType const *__restrict__ input_keys,
    int64_t const *__restrict__ table_ids,
    InsertResult *__restrict__ insert_results, IndexType *__restrict__ indices,
    ScoreType *__restrict__ score_input, int64_t *__restrict__ score_output,
    typename Table::KeyType **__restrict__ table_key_slots,
    CounterType *evicted_counter,
    typename Table::KeyType *__restrict__ evicted_keys,
    int64_t *__restrict__ evicted_scores,
    IndexType *__restrict__ evicted_indices,
    int64_t *__restrict__ evicted_table_ids) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  static constexpr int BlockSize = KernelTraits::ThreadBlockDim;
  static constexpr int BufferDim = KernelTraits::NumScorePerThread;

  static constexpr int ProbingGroupSize = KernelTraits::ProbingGroupSize;
  static constexpr int ReductionGroupSize = KernelTraits::ReductionGroupSize;
  static constexpr ScorePolicyType PolicyType = KernelTraits::PolicyType;
  static constexpr bool OutputScore = KernelTraits::OutputScore;

  using Policy = ScorePolicy<PolicyType>;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  __shared__ ScoreType sm_scores[BlockSize * BufferDim];

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = input_keys[i];
    ScoreType score = Policy::get(score_input, i);

    int64_t hashcode = 0;
    int64_t bucket_id = 0;
    int64_t table_cap = 0;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      int64_t t_id = table_ids[i];
      int64_t bkt_begin = table_bucket_offsets[t_id];
      int64_t bkt_end = table_bucket_offsets[t_id + 1];
      table_cap = (bkt_end - bkt_begin) * table.bucket_capacity();
      if (table_cap > 0) {
        int64_t local_idx = hashcode % table_cap;
        bucket_id = bkt_begin + local_idx / table.bucket_capacity();
      }
    }
    InsertResult result = InsertResult::Illegal;
    Bucket bucket = table[bucket_id];
    Iter iter = Iter();
    KeyType evict_key = KeyType();
    ScoreType evict_score = ScoreType();

    if (table_cap > 0) {
      iter = Iter(hashcode % table.bucket_capacity());
      insert_probe<ProbingGroupSize>(
          bucket, key, bucket_sizes, bucket_id, iter, &iter, &result);
      insert<ReductionGroupSize, BufferDim, Policy>(
          bucket, key, score, bucket_sizes, bucket_id, iter, sm_scores, result,
          &iter, &result, &evict_key, &evict_score);
    }

    // All threads must reach insert_evict so the cooperative-group ballot
    // in tiled_partition<CompactTileSize> sees every warp lane.
    // ILLEGAL threads participate with evicted=false.
    insert_evict<KernelTraits::CompactTileSize>(
        result, i, bucket_id, bucket, iter, evict_key, evict_score, table_ids,
        evicted_counter, evicted_keys, evicted_scores, evicted_indices,
        evicted_table_ids);

    IndexType index = -1;
    KeyType *table_key_slot = nullptr;
    if (isInsertSuccess(result)) {
      score = Policy::update(bucket.scores(iter), score);
      index = bucket_id * bucket.capacity() + iter;
      table_key_slot = bucket.keys(iter);
    }
    if constexpr (OutputScore) {
      score_output[i] = static_cast<int64_t>(score);
    }
    table_key_slots[i] = table_key_slot;
    indices[i] = index;
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
table_erase_kernel(Table table,
                   int64_t const *__restrict__ table_bucket_offsets,
                   int *__restrict__ bucket_sizes, int64_t batch,
                   typename Table::KeyType const *__restrict__ input_keys,
                   int64_t const *__restrict__ table_ids,
                   IndexType *__restrict__ indices) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = input_keys[i];

    int64_t hashcode = 0;
    int64_t bucket_id = 0;
    int64_t table_cap = 0;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      int64_t t_id = table_ids[i];
      int64_t bkt_begin = table_bucket_offsets[t_id];
      int64_t bkt_end = table_bucket_offsets[t_id + 1];
      table_cap = (bkt_end - bkt_begin) * table.bucket_capacity();
      if (table_cap > 0) {
        int64_t local_idx = hashcode % table_cap;
        bucket_id = bkt_begin + local_idx / table.bucket_capacity();
      }
    }
    if (table_cap == 0) {
      if (indices) {
        indices[i] = -1;
      }
      continue;
    }

    Bucket bucket = table[bucket_id];
    Iter iter = Iter(hashcode % table.bucket_capacity());
    int64_t step = 0;
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