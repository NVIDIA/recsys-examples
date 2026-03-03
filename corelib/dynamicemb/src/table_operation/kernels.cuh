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
          bool OutputScore_ = false,
          bool EnableOverflow_ = false>
struct InsertKernelTraits {
  static constexpr int ThreadBlockDim = ThreadBlockDim_;
  static constexpr int ProbingGroupSize = ProbingGroupSize_;
  static constexpr int ReductionGroupSize = ReductionGroupSize_;
  static constexpr int CompactTileSize = CompactTileSize_;
  static constexpr int NumScorePerThread = NumScorePerThread_;
  static constexpr ScorePolicyType PolicyType = PolicyType_;
  static constexpr bool OutputScore = OutputScore_;
  static constexpr bool EnableOverflow = EnableOverflow_;
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

template <typename Table, int ProbingGroupSize, ScorePolicyType PolicyType,
          bool EnableOverflow = false>
__global__ void
table_lookup_kernel(Table table,
                    int64_t const *__restrict__ table_bucket_offsets,
                    int64_t batch,
                    typename Table::KeyType const *__restrict__ input_keys,
                    int64_t const *__restrict__ table_ids,
                    bool *__restrict__ founds, IndexType *__restrict__ indices,
                    ScoreType *__restrict__ score_input,
                    int64_t *__restrict__ score_output,
                    Table ovf_table,
                    int64_t const *__restrict__ ovf_output_offsets) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = input_keys[i];
    ScoreType score = ScorePolicy<PolicyType>::get(score_input, i);

    int64_t hashcode = 0;
    int64_t bucket_id = 0;
    int64_t bkt_begin = 0;
    int64_t table_cap = 0;
    int64_t t_id = 0;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      t_id = table_ids[i];
      bkt_begin = table_bucket_offsets[t_id];
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
          found = false;
          score = ScoreType();
        }
      }

      if (found) {
        index = (bucket_id - bkt_begin) * bucket.capacity() + iter;
      }
    }

    if constexpr (EnableOverflow) {
      if (!found && Bucket::is_valid(key)) {
        Bucket ovf_bucket = ovf_table[t_id];
        Iter ovf_iter = Iter(hashcode % ovf_bucket.capacity());
        Iter ovf_out_iter;
        bool ovf_found = overflow_find(ovf_bucket, key,
                                       ovf_output_offsets[t_id],
                                       ovf_iter, &ovf_out_iter);
        if (ovf_found) {
          found = true;
          index = ovf_out_iter;
          if constexpr (PolicyType == ScorePolicyType::Const) {
            Iter local = ovf_out_iter - ovf_output_offsets[t_id];
            score = *ovf_bucket.scores(local);
          }
        }
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

template <int ReductionGroupSize, int BufferDim, typename Policy,
          typename Bucket, typename KeyType>
__forceinline__ __device__ void insert(
    Bucket &bucket, KeyType key, ScoreType score,
    int *__restrict__ bucket_sizes, int64_t bucket_id,
    typename Bucket::Iterator iter_in, ScoreType *sm_scores,
    InsertResult result_in, typename Bucket::Iterator *iter_out,
    InsertResult *result_out, KeyType *evict_key_out,
    ScoreType *evict_score_out,
    int32_t *__restrict__ counter,
    int64_t counter_offset) {
  auto iter = iter_in;
  auto result = result_in;
  while (result == InsertResult::Init) {
    KeyType evict_key;
    ScoreType evict_score = Policy::score_for_compare(score);

    bool succeed = bucket.template reduce<ReductionGroupSize, BufferDim>(
        iter, evict_key, evict_score, sm_scores, counter, counter_offset);

    if (succeed) {
      if (bucket.try_lock(iter, evict_key)) {
        if (evict_key != Bucket::reclaimed_key()) {
          int64_t flat_idx = counter_offset + iter;
          if (::atomicAdd(&counter[flat_idx], 0) > 0) {
            bucket.unlock(iter, evict_key);
            continue;
          }
        }
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
          if (evict_key_out) *evict_key_out = evict_key;
          if (evict_score_out) *evict_score_out = evict_score;
          break;
        }
      }
    } else {
      result = InsertResult::Busy;
      if (evict_key_out) *evict_key_out = key;
      if (evict_score_out) *evict_score_out = score;
      break;
    }
  }
  *iter_out = iter;
  *result_out = result;
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
    typename Table::KeyType **__restrict__ table_key_slots,
    int32_t *__restrict__ counter) {

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
    int64_t bkt_begin = 0;
    int64_t table_cap = 0;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      int64_t t_id = table_ids[i];
      bkt_begin = table_bucket_offsets[t_id];
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
    int64_t counter_offset = (bucket_id - bkt_begin) * bucket.capacity();
    insert<ReductionGroupSize, BufferDim, Policy>(
        bucket, key, score, bucket_sizes, bucket_id, iter, sm_scores, result,
        &iter, &result, static_cast<KeyType *>(nullptr),
        static_cast<ScoreType *>(nullptr),
        counter, counter_offset);

    IndexType index = -1;
    KeyType *table_key_slot = nullptr;
    if (isInsertSuccess(result)) {
      score = Policy::update(bucket.scores(iter), score);
      index = (bucket_id - bkt_begin) * bucket.capacity() + iter;
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
    int64_t *__restrict__ evicted_table_ids,
    int32_t *__restrict__ counter,
    Table ovf_table,
    int *__restrict__ ovf_bucket_sizes,
    int32_t *__restrict__ ovf_counter,
    int64_t const *__restrict__ ovf_output_offsets) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  static constexpr int BlockSize = KernelTraits::ThreadBlockDim;
  static constexpr int BufferDim = KernelTraits::NumScorePerThread;

  static constexpr int ProbingGroupSize = KernelTraits::ProbingGroupSize;
  static constexpr int ReductionGroupSize = KernelTraits::ReductionGroupSize;
  static constexpr ScorePolicyType PolicyType = KernelTraits::PolicyType;
  static constexpr bool OutputScore = KernelTraits::OutputScore;
  static constexpr bool UseOverflow = KernelTraits::EnableOverflow;

  using Policy = ScorePolicy<PolicyType>;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  __shared__ ScoreType sm_scores[BlockSize * BufferDim];

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = input_keys[i];
    ScoreType score = Policy::get(score_input, i);

    int64_t hashcode = 0;
    int64_t bucket_id = 0;
    int64_t bkt_begin = 0;
    int64_t table_cap = 0;
    int64_t t_id = 0;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      t_id = table_ids[i];
      bkt_begin = table_bucket_offsets[t_id];
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

      {
        int64_t counter_offset = (bucket_id - bkt_begin) * bucket.capacity();
        insert<ReductionGroupSize, BufferDim, Policy>(
            bucket, key, score, bucket_sizes, bucket_id, iter, sm_scores,
            result, &iter, &result, &evict_key, &evict_score,
            counter, counter_offset);
      }
    }

    IndexType index = -1;
    KeyType *table_key_slot = nullptr;
    if (isInsertSuccess(result)) {
      score = Policy::update(bucket.scores(iter), score);
      index = (bucket_id - bkt_begin) * bucket.capacity() + iter;
      table_key_slot = bucket.keys(iter);
    }

    // Overflow fallback for Busy results
    KeyType final_evict_key = evict_key;
    ScoreType final_evict_score = evict_score;
    IndexType final_evict_index = -static_cast<IndexType>(i + 1);
    InsertResult final_result = result;

    if constexpr (UseOverflow) {
      if (result == InsertResult::Busy && Bucket::is_valid(key)) {
        Bucket ovf_bucket = ovf_table[t_id];
        Iter ovf_iter = Iter(hashcode % ovf_bucket.capacity());
        InsertResult ovf_result = InsertResult::Init;
        KeyType ovf_evict_key = KeyType();
        overflow_insert_and_evict(
            ovf_bucket, key, ovf_bucket_sizes, t_id,
            ovf_counter, ovf_output_offsets[t_id],
            ovf_iter, &ovf_iter, &ovf_result, &ovf_evict_key);

        if (isInsertSuccess(ovf_result)) {
          index = ovf_iter;
          final_result = ovf_result;
          table_key_slot = nullptr;
          if (ovf_result == InsertResult::Evict) {
            Iter local = ovf_iter - ovf_output_offsets[t_id];
            table_key_slot = ovf_bucket.keys(local);
            final_evict_key = ovf_evict_key;
            final_evict_index = ovf_iter;
          } else if (ovf_result == InsertResult::Insert) {
            Iter local = ovf_iter - ovf_output_offsets[t_id];
            table_key_slot = ovf_bucket.keys(local);
          }
        }
      }

      if (result == InsertResult::Evict) {
        final_evict_index = (bucket_id - bkt_begin) * bucket.capacity() + iter;
      }
    }

    // Eviction compaction -- all threads participate in ballot
    {
      auto g = cg::tiled_partition<KernelTraits::CompactTileSize>(
          cg::this_thread_block());

      InsertResult cmp_result = UseOverflow ? final_result : result;
      bool evicted =
          (cmp_result == InsertResult::Evict ||
           cmp_result == InsertResult::Busy);
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
        if constexpr (UseOverflow) {
          evicted_keys[out_id] = final_evict_key;
          evicted_scores[out_id] = static_cast<int64_t>(final_evict_score);
          evicted_indices[out_id] = final_evict_index;
        } else {
          evicted_keys[out_id] = evict_key;
          evicted_scores[out_id] = static_cast<int64_t>(evict_score);
          IndexType evict_idx = (result == InsertResult::Evict)
              ? (bucket_id - bkt_begin) * bucket.capacity() + iter
              : -static_cast<IndexType>(i + 1);
          evicted_indices[out_id] = evict_idx;
        }
        evicted_table_ids[out_id] = table_ids[i];
      }
    }

    if constexpr (OutputScore) {
      score_output[i] = static_cast<int64_t>(score);
    }
    table_key_slots[i] = table_key_slot;
    indices[i] = index;
    if (insert_results) {
      insert_results[i] = UseOverflow ? final_result : result;
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
    int64_t bkt_begin = 0;
    int64_t table_cap = 0;
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      int64_t t_id = table_ids[i];
      bkt_begin = table_bucket_offsets[t_id];
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
        index = (bucket_id - bkt_begin) * bucket.capacity() + iter;
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
                          IndexType table_begin,
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
    const IndexType index = i - table_begin;

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


// ---------------------------------------------------------------------------
// Overflow buffer: find a key by linear scan.
// Returns true if found, sets *iter_out = local_pos + output_offset.
// ---------------------------------------------------------------------------
template <typename Bucket, typename KeyType>
__forceinline__ __device__ bool overflow_find(
    Bucket &ovf_bucket, KeyType key, int64_t output_offset,
    typename Bucket::Iterator start_iter,
    typename Bucket::Iterator *iter_out) {
  for (int64_t scan = 0; scan < ovf_bucket.capacity(); scan++) {
    auto pos = (start_iter + scan) % ovf_bucket.capacity();

    auto key_slot =
        reinterpret_cast<typename Bucket::AtomicKey *>(ovf_bucket.keys(pos));
    KeyType k = key_slot->load(cuda::std::memory_order_relaxed);

    if (k == key) {
      *iter_out = pos + output_offset;
      return true;
    }
    if (k == Bucket::empty_key()) {
      return false;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Overflow buffer: single-pass find-or-insert with counter-based eviction.
// Linear scan: key found -> Assign; empty -> Insert; counter==0 -> Evict.
// Outputs unified index (local + output_offset).
// ---------------------------------------------------------------------------
template <typename Bucket, typename KeyType>
__forceinline__ __device__ void overflow_insert_and_evict(
    Bucket &bucket, KeyType key,
    int *__restrict__ bucket_sizes, int64_t bucket_id,
    int32_t *__restrict__ counter,
    int64_t output_offset,
    typename Bucket::Iterator start_iter,
    typename Bucket::Iterator *iter_out, InsertResult *result_out,
    KeyType *evict_key_out) {

  for (int64_t scan = 0; scan < bucket.capacity(); scan++) {
    auto pos = (start_iter + scan) % bucket.capacity();

    auto key_slot =
        reinterpret_cast<typename Bucket::AtomicKey *>(bucket.keys(pos));
    KeyType k = key_slot->load(cuda::std::memory_order_relaxed);

    if (k == key) {
      *iter_out = pos + output_offset;
      *result_out = InsertResult::Assign;
      return;
    }

    if (k == Bucket::empty_key()) {
      KeyType expected = Bucket::empty_key();
      if (bucket.try_lock(pos, expected)) {
        *bucket.digests(pos) = Bucket::key_to_digest(key);
        atomicAdd(&bucket_sizes[bucket_id], 1);
        *iter_out = pos + output_offset;
        *result_out = InsertResult::Insert;
        return;
      }
      k = key_slot->load(cuda::std::memory_order_relaxed);
      if (k == key) {
        *iter_out = pos + output_offset;
        *result_out = InsertResult::Assign;
        return;
      }
      continue;
    }

    if (k == Bucket::LockedKey || k == Bucket::reclaimed_key()) continue;

    if (counter[pos] == 0) {
      if (bucket.try_lock(pos, k)) {
        if (::atomicAdd(&counter[pos], 0) > 0) {
          bucket.unlock(pos, k);
          continue;
        }
        *bucket.digests(pos) = Bucket::key_to_digest(key);
        *evict_key_out = k;
        *iter_out = pos + output_offset;
        *result_out = InsertResult::Evict;
        return;
      }
      continue;
    }
  }

  *result_out = InsertResult::Busy;
  *evict_key_out = key;
}

// ---------------------------------------------------------------------------
// Counter update kernel: atomically add delta to counter[slot] for each
// slot_indices entry that falls in [0, capacity).
// Defined in insert_and_evict.cu.
// ---------------------------------------------------------------------------
__global__ void update_counter_kernel(
    int32_t *__restrict__ counter, int64_t capacity,
    int64_t const *__restrict__ slot_indices, int64_t n, int32_t delta);


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