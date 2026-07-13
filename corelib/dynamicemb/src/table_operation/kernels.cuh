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
          bool OutputScore_ = false, bool EnableOverflow_ = false,
          bool FindOrInsert_ = false>
struct InsertKernelTraits {
  static constexpr int ThreadBlockDim = ThreadBlockDim_;
  static constexpr int ProbingGroupSize = ProbingGroupSize_;
  static constexpr int ReductionGroupSize = ReductionGroupSize_;
  static constexpr int CompactTileSize = CompactTileSize_;
  static constexpr int NumScorePerThread = NumScorePerThread_;
  static constexpr ScorePolicyType PolicyType = PolicyType_;
  static constexpr bool OutputScore = OutputScore_;
  static constexpr bool EnableOverflow = EnableOverflow_;
  static constexpr bool FindOrInsert = FindOrInsert_;
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
__global__ void table_lookup_kernel(
    Table table, int64_t const *__restrict__ table_bucket_offsets,
    int64_t batch, typename Table::KeyType const *__restrict__ input_keys,
    int64_t const *__restrict__ table_ids, bool *__restrict__ founds,
    IndexType *__restrict__ indices, ScoreType *__restrict__ score_input,
    int64_t *__restrict__ score_output, Table ovf_table,
    int64_t const *__restrict__ ovf_output_offsets,
    bool const *__restrict__ active_mask,
    int64_t const *__restrict__ active_count,
    int32_t *__restrict__ acquire_counter,
    int32_t *__restrict__ acquire_ovf_counter) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    if ((active_count != nullptr && i >= *active_count) ||
        (active_mask != nullptr && !active_mask[i])) {
      score_output[i] = 0;
      founds[i] = false;
      indices[i] = -1;
      continue;
    }

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
        if (acquire_counter != nullptr) {
          KeyType expected_key = key;
          if (bucket.try_lock(iter, expected_key)) {
            score = *bucket.scores(iter);
            atomicAdd(acquire_counter + bucket_id * bucket.capacity() + iter,
                      1);
            bucket.unlock(iter, key);
          } else {
            found = false;
            score = ScoreType();
          }
        } else {
          score = *bucket.scores(iter);
        }
      } else {
        KeyType expected_key = key;
        if (bucket.try_lock(iter, expected_key)) {
          score = ScorePolicy<PolicyType>::update(bucket.scores(iter), score);
          if (acquire_counter != nullptr) {
            atomicAdd(acquire_counter + bucket_id * bucket.capacity() + iter,
                      1);
          }
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
        bool ovf_found = overflow_find(
            ovf_bucket, key, ovf_output_offsets[t_id], ovf_iter, &ovf_out_iter);
        if (ovf_found) {
          found = true;
          index = ovf_out_iter;
          Iter local = ovf_out_iter - ovf_output_offsets[t_id];
          if constexpr (PolicyType == ScorePolicyType::Const) {
            if (acquire_ovf_counter != nullptr) {
              KeyType expected_key = key;
              if (ovf_bucket.try_lock(local, expected_key)) {
                score = *ovf_bucket.scores(local);
                atomicAdd(acquire_ovf_counter +
                              t_id * ovf_table.bucket_capacity() + local,
                          1);
                ovf_bucket.unlock(local, key);
              } else {
                found = false;
              }
            } else {
              score = *ovf_bucket.scores(local);
            }
          } else {
            KeyType expected_key = key;
            if (ovf_bucket.try_lock(local, expected_key)) {
              score = ScorePolicy<PolicyType>::update(
                  ovf_bucket.scores(local), score);
              if (acquire_ovf_counter != nullptr) {
                atomicAdd(acquire_ovf_counter +
                              t_id * ovf_table.bucket_capacity() + local,
                          1);
              }
              ovf_bucket.unlock(local, key);
            } else {
              found = false;
            }
          }
          if (!found) {
            index = -1;
            score = ScoreType();
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
__forceinline__ __device__ void
insert_probe(Bucket &bucket, KeyType key, int *__restrict__ bucket_sizes,
             int64_t bucket_id, typename Bucket::Iterator iter_in,
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
__forceinline__ __device__ void
insert(Bucket &bucket, KeyType key, ScoreType score,
       int *__restrict__ bucket_sizes, int64_t bucket_id,
       typename Bucket::Iterator iter_in, ScoreType *sm_scores,
       InsertResult result_in, typename Bucket::Iterator *iter_out,
       InsertResult *result_out, KeyType *evict_key_out,
       ScoreType *evict_score_out, int32_t *__restrict__ counter,
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
          if (evict_key_out)
            *evict_key_out = evict_key;
          if (evict_score_out)
            *evict_score_out = evict_score;
          break;
        }
      }
    } else {
      result = InsertResult::Busy;
      if (evict_key_out)
        *evict_key_out = key;
      if (evict_score_out)
        *evict_score_out = score;
      break;
    }
  }
  *iter_out = iter;
  *result_out = result;
}

template <typename Table, typename KernelTraits, bool PublishAndAcquire = false>
__global__ void table_insert_kernel(
    Table table, int64_t const *__restrict__ table_bucket_offsets,
    int *__restrict__ bucket_sizes, int64_t batch,
    typename Table::KeyType const *__restrict__ input_keys,
    int64_t const *__restrict__ table_ids,
    InsertResult *__restrict__ insert_results, IndexType *__restrict__ indices,
    ScoreType *__restrict__ score_input, int64_t *__restrict__ score_output,
    typename Table::KeyType **__restrict__ table_key_slots,
    int32_t *__restrict__ counter, bool const *__restrict__ active_mask,
    int64_t const *__restrict__ active_count,
    int64_t *__restrict__ row_counters) {

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

    if ((active_count != nullptr && i >= *active_count) ||
        (active_mask != nullptr && !active_mask[i])) {
      // Sparse score callers may provide uninitialized output storage.  The
      // generic score contract uses zero for skipped positions.  NO_EVICTION
      // row assignment and publish/acquire waves retain caller-owned output
      // so later deterministic waves cannot overwrite earlier results.
      if constexpr (OutputScore) {
        if (row_counters == nullptr && !PublishAndAcquire) {
          score_output[i] = 0;
        }
      }
      continue;
    }

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
    if (table_cap == 0) {
      if constexpr (OutputScore) {
        score_output[i] = row_counters == nullptr
                              ? static_cast<int64_t>(score)
                              : static_cast<int64_t>(-1);
      }
      indices[i] = -1;
      if (insert_results) {
        insert_results[i] = InsertResult::Illegal;
      }
      continue;
    }

    InsertResult result;
    Bucket bucket = table[bucket_id];
    Iter iter = Iter(hashcode % table.bucket_capacity());
    insert_probe<ProbingGroupSize>(bucket, key, bucket_sizes, bucket_id, iter,
                                   &iter, &result);
    int64_t counter_offset = bucket_id * bucket.capacity();
    insert<ReductionGroupSize, BufferDim, Policy>(
        bucket, key, score, bucket_sizes, bucket_id, iter, sm_scores, result,
        &iter, &result, static_cast<KeyType *>(nullptr),
        static_cast<ScoreType *>(nullptr), counter, counter_offset);

    IndexType index = -1;
    KeyType *table_key_slot = nullptr;
    if (isInsertSuccess(result)) {
      if (row_counters != nullptr) {
        if (result == InsertResult::Assign) {
          // Existing NO_EVICTION keys keep their stable physical row.
          score = *bucket.scores(iter);
        } else {
          // Allocate and publish the physical row while the key slot remains
          // locked.  This replaces the former insert -> assign -> score-update
          // sequence with one metadata kernel.
          int64_t row = atomicAdd(row_counters + t_id, int64_t{1});
          score = static_cast<ScoreType>(row);
          *bucket.scores(iter) = score;
        }
      } else {
        score = Policy::update(bucket.scores(iter), score);
      }
      index = (bucket_id - bkt_begin) * bucket.capacity() + iter;
      if constexpr (PublishAndAcquire) {
        // Pin the destination before making the key visible.  The exchange
        // value kernel releases this temporary ownership after its outbound
        // write, so another insertion cannot recycle the row in between.
        atomicAdd(counter + bucket_id * bucket.capacity() + iter, 1);
        auto atomic_key = reinterpret_cast<typename Bucket::AtomicKey *>(
            bucket.keys(iter));
        atomic_key->store(key, cuda::std::memory_order_release);
      } else {
        table_key_slot = bucket.keys(iter);
      }
    }
    if constexpr (OutputScore) {
      score_output[i] =
          (row_counters != nullptr && !isInsertSuccess(result))
              ? static_cast<int64_t>(-1)
              : static_cast<int64_t>(score);
    }
    indices[i] = index;
    if (insert_results) {
      insert_results[i] = result;
    }
    if constexpr (!PublishAndAcquire) {
      table_key_slots[i] = table_key_slot;
    }
  }
}

template <typename Table, typename KernelTraits>
__global__ void table_insert_and_evict_kernel(
    Table table, int64_t const *__restrict__ table_bucket_offsets,
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
    int64_t *__restrict__ evicted_table_ids, int32_t *__restrict__ counter,
    Table ovf_table, int *__restrict__ ovf_bucket_sizes,
    int32_t *__restrict__ ovf_counter,
    int64_t const *__restrict__ ovf_output_offsets,
    bool *__restrict__ founds, bool *__restrict__ evicted_mask,
    bool const *__restrict__ active_mask,
    int64_t const *__restrict__ active_count) {

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
  static constexpr bool FindOrInsert = KernelTraits::FindOrInsert;

  using Policy = ScorePolicy<PolicyType>;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  __shared__ ScoreType sm_scores[BlockSize * BufferDim];

  if constexpr (FindOrInsert) {
    // Phase 1: find and protect every hit before any thread may select an
    // eviction victim. This kernel is launched cooperatively so the grid-wide
    // barrier preserves lookup-all -> pin-hits -> insert-misses ordering.
    for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {
      bool active = (active_count == nullptr || i < *active_count) &&
                    (active_mask == nullptr || active_mask[i]);
      if (!active) {
        continue;
      }

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

      bool found = false;
      IndexType index = -1;
      if (table_cap > 0) {
        Bucket bucket = table[bucket_id];
        Iter iter = Iter(hashcode % table.bucket_capacity());
        int64_t step = 0;
        auto probe_res = bucket.template probe<ProbingGroupSize>(
            key, iter, step);
        found = probe_res == Bucket::ProbeResult::Existed;
        if (found) {
          KeyType expected_key = key;
          if (bucket.try_lock(iter, expected_key)) {
            if constexpr (PolicyType == ScorePolicyType::Const) {
              score = *bucket.scores(iter);
            } else {
              score = Policy::update(bucket.scores(iter), score);
            }
            atomicAdd(counter + bucket_id * bucket.capacity() + iter, 1);
            bucket.unlock(iter, key);
            index = (bucket_id - bkt_begin) * bucket.capacity() + iter;
          } else {
            found = false;
          }
        }

        if constexpr (UseOverflow) {
          if (!found && Bucket::is_valid(key)) {
            Bucket ovf_bucket = ovf_table[t_id];
            Iter ovf_iter = Iter(hashcode % ovf_bucket.capacity());
            Iter ovf_out_iter;
            found = overflow_find(ovf_bucket, key, ovf_output_offsets[t_id],
                                  ovf_iter, &ovf_out_iter);
            if (found) {
              Iter local = ovf_out_iter - ovf_output_offsets[t_id];
              KeyType expected_key = key;
              if (ovf_bucket.try_lock(local, expected_key)) {
                if constexpr (PolicyType == ScorePolicyType::Const) {
                  score = *ovf_bucket.scores(local);
                } else {
                  score = Policy::update(ovf_bucket.scores(local), score);
                }
                atomicAdd(ovf_counter +
                              t_id * ovf_table.bucket_capacity() + local,
                          1);
                ovf_bucket.unlock(local, key);
                index = ovf_out_iter;
              } else {
                found = false;
              }
            }
          }
        }
      }

      founds[i] = found;
      if (found) {
        indices[i] = index;
        if constexpr (OutputScore) {
          score_output[i] = static_cast<int64_t>(score);
        }
        if (insert_results) {
          insert_results[i] = InsertResult::Assign;
        }
      } else {
        indices[i] = -1;
      }
    }
    cg::this_grid().sync();
  }

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    // Do not early-return here: CompactTileSize may be a warp, and every lane
    // must reach the eviction ballot even when sparse masking is enabled.
    bool active = (active_count == nullptr || i < *active_count) &&
                  (active_mask == nullptr || active_mask[i]);
    if constexpr (FindOrInsert) {
      // find_or_insert may reuse one output buffer across deterministic waves.
      // Clear only this wave's active positions so earlier waves remain intact.
      if (active) {
        evicted_mask[i] = false;
      }
      active = active && !founds[i];
    }

    KeyType key = active ? input_keys[i] : Bucket::empty_key();
    ScoreType score = active ? Policy::get(score_input, i) : ScoreType();

    int64_t hashcode = 0;
    int64_t bucket_id = 0;
    int64_t bkt_begin = 0;
    int64_t table_cap = 0;
    int64_t t_id = 0;
    if (active && Bucket::is_valid(key)) {
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
      insert_probe<ProbingGroupSize>(bucket, key, bucket_sizes, bucket_id, iter,
                                     &iter, &result);

      {
        int64_t counter_offset = bucket_id * bucket.capacity();
        insert<ReductionGroupSize, BufferDim, Policy>(
            bucket, key, score, bucket_sizes, bucket_id, iter, sm_scores,
            result, &iter, &result, &evict_key, &evict_score, counter,
            counter_offset);
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
        int64_t ovf_bucket_capacity = ovf_bucket.capacity();
        Iter ovf_iter = Iter(hashcode % ovf_bucket_capacity);
        InsertResult ovf_result = InsertResult::Init;
        KeyType ovf_evict_key = KeyType();
        int32_t *ovf_counter_table = ovf_counter + t_id * ovf_bucket_capacity;
        overflow_insert_and_evict(ovf_bucket, key, ovf_bucket_sizes, t_id,
                                  ovf_counter_table, ovf_output_offsets[t_id],
                                  ovf_iter, &ovf_iter, &ovf_result,
                                  &ovf_evict_key);

        if (isInsertSuccess(ovf_result)) {
          index = ovf_iter;
          final_result = ovf_result;
          Iter local = ovf_iter - ovf_output_offsets[t_id];
          table_key_slot = ovf_bucket.keys(local);
          if (ovf_result == InsertResult::Evict) {
            final_evict_key = ovf_evict_key;
            final_evict_score = *ovf_bucket.scores(local);
            *ovf_bucket.scores(local) = ScoreType();
            final_evict_index = ovf_iter;
          }
          score = Policy::update(ovf_bucket.scores(local), score);
        }
      }

      if (result == InsertResult::Evict) {
        final_evict_index = (bucket_id - bkt_begin) * bucket.capacity() + iter;
      }
    }

    {
      InsertResult cmp_result = UseOverflow ? final_result : result;
      // Fused exchange consumes input-aligned sparse eviction metadata. Legacy
      // insert_and_evict retains compact output and its Busy convention.
      bool evicted = active &&
                     (cmp_result == InsertResult::Evict ||
                      (!FindOrInsert && cmp_result == InsertResult::Busy));
      if constexpr (FindOrInsert) {
        if (evicted) {
          if constexpr (UseOverflow) {
            evicted_keys[i] = final_evict_key;
            evicted_scores[i] = static_cast<int64_t>(final_evict_score);
            evicted_indices[i] = final_evict_index;
          } else {
            evicted_keys[i] = evict_key;
            evicted_scores[i] = static_cast<int64_t>(evict_score);
            evicted_indices[i] =
                (bucket_id - bkt_begin) * bucket.capacity() + iter;
          }
          evicted_table_ids[i] = table_ids[i];
          evicted_mask[i] = true;
        }
      } else {
        // All lanes participate because CompactTileSize may be a warp.
        auto g = cg::tiled_partition<KernelTraits::CompactTileSize>(
            cg::this_thread_block());
        uint32_t vote = g.ballot(evicted);
        int group_cnt = __popc(vote);
        CounterType group_offset = 0;
        if (g.thread_rank() == 0 && group_cnt != 0) {
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
            IndexType evict_idx =
                (result == InsertResult::Evict)
                    ? (bucket_id - bkt_begin) * bucket.capacity() + iter
                    : -static_cast<IndexType>(i + 1);
            evicted_indices[out_id] = evict_idx;
          }
          evicted_table_ids[out_id] = table_ids[i];
        }
      }
    }

    if (active) {
      if constexpr (FindOrInsert) {
        InsertResult observed_result = UseOverflow ? final_result : result;
        // A fused cache lookup must always provision a protected row. The
        // main-bucket Busy result has already had its overflow fallback above;
        // reaching Busy here means both tiers are unavailable. Continuing
        // would produce an untracked/transient update for a globally new key,
        // so make this violated cache-capacity invariant fatal. Use __trap()
        // rather than assert so release builds compiled with NDEBUG retain the
        // check.
        if (observed_result == InsertResult::Busy) {
          __trap();
        }
        founds[i] = observed_result == InsertResult::Assign;
        if (isInsertSuccess(observed_result)) {
          if constexpr (UseOverflow) {
            if (result == InsertResult::Busy) {
              int64_t local = index - ovf_output_offsets[t_id];
              atomicAdd(ovf_counter + t_id * ovf_table.bucket_capacity() + local,
                        1);
            } else {
              atomicAdd(counter + bucket_id * bucket.capacity() + iter, 1);
            }
          } else {
            atomicAdd(counter + bucket_id * bucket.capacity() + iter, 1);
          }
        }
      }
      if constexpr (OutputScore) {
        score_output[i] = static_cast<int64_t>(score);
      }
      if constexpr (!FindOrInsert) {
        table_key_slots[i] = table_key_slot;
      }
      indices[i] = index;
      if (insert_results) {
        insert_results[i] = UseOverflow ? final_result : result;
      }
      if constexpr (FindOrInsert) {
        // Counter ownership and eviction metadata are visible before the key
        // is published.  This keeps find_or_insert a single GPU kernel while
        // preserving the lock's release ordering.
        if (table_key_slot != nullptr) {
          auto atomic_key =
              reinterpret_cast<typename Bucket::AtomicKey *>(table_key_slot);
          atomic_key->store(key, cuda::std::memory_order_release);
        }
      }
    } else {
      if constexpr (!FindOrInsert) {
        table_key_slots[i] = nullptr;
      }
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
__global__ void table_erase_kernel(
    Table table, int64_t const *__restrict__ table_bucket_offsets,
    int *__restrict__ bucket_sizes, int64_t batch,
    typename Table::KeyType const *__restrict__ input_keys,
    int64_t const *__restrict__ table_ids, IndexType *__restrict__ indices,
    bool const *__restrict__ active_mask,
    int64_t const *__restrict__ active_count) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    if ((active_count != nullptr && i >= *active_count) ||
        (active_mask != nullptr && !active_mask[i])) {
      continue;
    }

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

// Reclaim provisional entries by their already-known per-table slot.  This
// avoids a second hash probe after admission and is safe for main-table probe
// chains because the slot is published as a tombstone, not an empty key.
template <typename Table, bool EnableOverflow = false>
__global__ void table_reclaim_by_slot_kernel(
    Table table, int64_t const *__restrict__ table_bucket_offsets,
    int *__restrict__ bucket_sizes, int64_t batch,
    typename Table::KeyType const *__restrict__ input_keys,
    int64_t const *__restrict__ table_ids,
    IndexType *__restrict__ slot_indices, bool const *__restrict__ mask,
    int32_t *__restrict__ counter, Table ovf_table,
    int *__restrict__ ovf_bucket_sizes, int32_t *__restrict__ ovf_counter,
    int64_t const *__restrict__ ovf_output_offsets) {
  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= batch || (mask != nullptr && !mask[i]))
    return;

  IndexType slot = slot_indices[i];
  if (slot < 0) {
    slot_indices[i] = -1;
    return;
  }

  int64_t t_id = table_ids[i];
  int64_t bkt_begin = table_bucket_offsets[t_id];
  int64_t bkt_end = table_bucket_offsets[t_id + 1];
  int64_t table_cap = (bkt_end - bkt_begin) * table.bucket_capacity();
  KeyType key = input_keys[i];

  if (slot < table_cap) {
    int64_t bucket_id = bkt_begin + slot / table.bucket_capacity();
    Iter iter = slot % table.bucket_capacity();
    Bucket bucket = table[bucket_id];
    KeyType expected = key;
    if (bucket.try_lock(iter, expected)) {
      *bucket.scores(iter) = ScoreType();
      *bucket.digests(iter) = Bucket::empty_digest();
      ::atomicExch(counter + bucket_id * table.bucket_capacity() + iter, 0);
      bucket.unlock(iter, Bucket::reclaimed_key());
      atomicSub(bucket_sizes + bucket_id, 1);
    }
  } else if constexpr (EnableOverflow) {
    int64_t local = slot - ovf_output_offsets[t_id];
    if (local >= 0 && local < ovf_table.bucket_capacity()) {
      Bucket bucket = ovf_table[t_id];
      Iter iter = local;
      KeyType expected = key;
      if (bucket.try_lock(iter, expected)) {
        *bucket.scores(iter) = ScoreType();
        *bucket.digests(iter) = Bucket::empty_digest();
        ::atomicExch(ovf_counter + t_id * ovf_table.bucket_capacity() + local,
                     0);
        bucket.unlock(iter, Bucket::reclaimed_key());
        atomicSub(ovf_bucket_sizes + t_id, 1);
      }
    }
  }
  slot_indices[i] = -1;
}

// Replace scores using known cache slots.  Storage exchange uses this for LFU
// hits so the cumulative backing-store score replaces the provisional cache
// score without another hash lookup.
template <typename Bucket>
__forceinline__ __device__ bool lock_expected_key_with_backoff(
    Bucket bucket, typename Bucket::Iterator iter,
    typename Bucket::KeyType key) {
  using KeyType = typename Bucket::KeyType;
  constexpr int kMaxAttempts = 32;
#pragma unroll 1
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    KeyType expected = key;
    if (bucket.try_lock(iter, expected))
      return true;
    // A different published key means the known slot is stale. Only a
    // transient lock is retryable; never write a replacement occupant's
    // score.
    if (expected != static_cast<KeyType>(Bucket::LockedKey))
      return false;
    const unsigned int backoff =
        32U << static_cast<unsigned int>(attempt < 5 ? attempt : 5);
    __nanosleep(backoff);
  }
  return false;
}

template <typename Table, bool EnableOverflow = false>
__global__ void table_update_score_by_slot_kernel(
    Table table, int64_t const *__restrict__ table_bucket_offsets,
    int64_t batch,
    typename Table::KeyType const *__restrict__ input_keys,
    IndexType const *__restrict__ slot_indices,
    int64_t const *__restrict__ table_ids,
    ScoreType const *__restrict__ input_scores,
    bool const *__restrict__ active_mask,
    int64_t const *__restrict__ active_count, Table ovf_table,
    int64_t const *__restrict__ ovf_output_offsets) {
  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= batch ||
      (active_count != nullptr && i >= *active_count) ||
      (active_mask != nullptr && !active_mask[i]))
    return;

  IndexType slot = slot_indices[i];
  if (slot < 0)
    return;
  int64_t t_id = table_ids[i];
  int64_t bkt_begin = table_bucket_offsets[t_id];
  int64_t bkt_end = table_bucket_offsets[t_id + 1];
  int64_t table_cap = (bkt_end - bkt_begin) * table.bucket_capacity();
  if (slot < table_cap) {
    int64_t bucket_id = bkt_begin + slot / table.bucket_capacity();
    auto bucket = table[bucket_id];
    Iter iter = slot % table.bucket_capacity();
    KeyType key = input_keys[i];
    if (lock_expected_key_with_backoff(bucket, iter, key)) {
      *bucket.scores(iter) = input_scores[i];
      bucket.unlock(iter, key);
    }
  } else if constexpr (EnableOverflow) {
    int64_t local = slot - ovf_output_offsets[t_id];
    if (local >= 0 && local < ovf_table.bucket_capacity()) {
      auto bucket = ovf_table[t_id];
      Iter iter = local;
      KeyType key = input_keys[i];
      if (lock_expected_key_with_backoff(bucket, iter, key)) {
        *bucket.scores(iter) = input_scores[i];
        bucket.unlock(iter, key);
      }
    }
  }
}

template <typename Table, typename PredFunctor, int TileSize>
__global__ void table_export_batch_kernel(
    Table table, IndexType begin, IndexType end, IndexType table_begin,
    CounterType *__restrict__ counter,
    typename Table::KeyType *__restrict__ keys, ScoreType *__restrict__ scores,
    PredFunctor pred, IndexType *__restrict__ indices) {
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
__forceinline__ __device__ bool
overflow_find(Bucket &ovf_bucket, KeyType key, int64_t output_offset,
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
    Bucket &bucket, KeyType key, int *__restrict__ bucket_sizes,
    int64_t bucket_id, int32_t *__restrict__ counter, int64_t output_offset,
    typename Bucket::Iterator start_iter, typename Bucket::Iterator *iter_out,
    InsertResult *result_out, KeyType *evict_key_out) {

  for (int64_t scan = 0; scan < bucket.capacity(); scan++) {
    auto pos = (start_iter + scan) % bucket.capacity();

    auto key_slot =
        reinterpret_cast<typename Bucket::AtomicKey *>(bucket.keys(pos));
    KeyType k = key_slot->load(cuda::std::memory_order_relaxed);

    if (k == key) {
      KeyType expected = key;
      if (bucket.try_lock(pos, expected)) {
        *iter_out = pos + output_offset;
        *result_out = InsertResult::Assign;
        return;
      }
      continue;
    }

    if (k == Bucket::empty_key() || k == Bucket::reclaimed_key()) {
      KeyType expected = k;
      if (bucket.try_lock(pos, expected)) {
        *bucket.digests(pos) = Bucket::key_to_digest(key);
        atomicAdd(&bucket_sizes[bucket_id], 1);
        *iter_out = pos + output_offset;
        *result_out = (k == Bucket::empty_key()) ? InsertResult::Insert
                                                : InsertResult::Reclaim;
        return;
      }
      k = key_slot->load(cuda::std::memory_order_relaxed);
      if (k == key) {
        expected = key;
        if (bucket.try_lock(pos, expected)) {
          *iter_out = pos + output_offset;
          *result_out = InsertResult::Assign;
          return;
        }
      }
      continue;
    }

    if (k == Bucket::LockedKey)
      continue;

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
