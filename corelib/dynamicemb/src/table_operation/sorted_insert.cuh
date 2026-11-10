/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

// The deterministic insert (DEMB_DETERMINISM_MODE): the keys of a batch are
// sorted by (home bucket, key) -- bucketize_keys_segmented -- and ONE thread
// inserts the keys of each bucket, one after another, smallest key first. A
// bucket therefore sees the same sequence of operations whatever the order of
// the batch and the scheduling of the GPU, which is what the waves of the
// earlier implementation (one launch per rank of a key in its bucket, a host
// loop) produced, without a loop bound that only the device knows and without a
// read.
//
// The per-key work is the code of the ordinary insert kernels, not a copy of
// it: insert_probe, insert (the eviction scan, DefaultEvictor), Policy::update
// and, for a cache, insert_overflow_tier. The invariants the kernels rely on:
//
// clang-format off
//  I1 Segments. ``segments`` is non-decreasing; a key that takes part has
//     segment == its home bucket < num_buckets; a key that does not has segment
//     == num_buckets and sorts after all of them. Inside a segment the keys
//     ascend (stable on input position).
//  I2 Exclusivity. In the main kernel the memory of bucket b (keys, digests,
//     scores, bucket_sizes[b]) is read and written only by the thread at the
//     head of b's run; in the overflow pass the overflow bucket of table t only
//     by thread t. The reference counters are only read.
//  I3 Lock discipline, the one of the ordinary insert kernels. A slot a key
//     takes (an Insert, a Reclaim, an Evict, or the Assign of a resident) stays
//     locked until every key of the run has been inserted, and is then released
//     (a release store of the key) by the thread that took it, as
//     table_unlock_kernel does at the end of an ordinary insert; the overflow
//     bucket likewise until the end of the overflow pass. A key is therefore
//     never evicted by a later key of its own call: the evictions of a call
//     name distinct slots and victims that were there before the call, which is
//     what the write-back of a cache and of HybridStorage rely on (they read
//     the value rows of the victims before the rows of the new keys are
//     stored). A bucket that gets more new keys than it has free or evictable
//     slots refuses the last of them (Busy), as the ordinary kernels do. The
//     one slot that is not held is that of a key found in the overflow bucket
//     (an Assign there is updated under a lock released at once,
//     insert_overflow_tier), in the ordinary kernel as here.
//  I4 Order. The sequence of (key, score) operations a bucket sees is its keys
//     in ascending order. The result is what the ordinary insert kernel
//     produces when its threads take the keys of a bucket one after the other
//     in that order.
//  I5 Coverage. Every sorted position is written once by its owner: the record
//     (all five fields, inert unless an eviction), the overflow flag, and the
//     insert result and output score of its input position.
//  I6 Inert rows. A reserved key, a key of a table without buckets and a key
//     of a table that does not exist do nothing and read no table id (reserved
//     keys) or no bucket (the others): result Illegal, output score = the
//     policy's input score, what insert_body does for table_cap == 0. Inert for
//     this insert only: the table_lookup that gives the indices afterwards
//     reads the bucket offsets of the table of every key that is not reserved,
//     so a caller passes real table ids (or the reserved padding), as it always
//     had to.
//  I7 No host dependence. Every shape and every launch parameter derives from
//     the length of the input and the geometry of the table.
// clang-format on
//
// Include order: no header that reaches table_operation/types.cuh may come
// before ../index_calculation.h in a translation unit. types.cuh declares
// dyn_emb::atomicAdd(int *, int) and (CounterType *, CounterType), which hide
// the CUDA atomicAdd(float *, float) that lookup_kernel.cuh (through
// index_calculation.h) calls from the members of its Vec4T specializations
// (bucketize.cu and insert_sorted.cu include them in that order).
#pragma once

#include "kernels.cuh"

// A fence between two keys of one run. All the accesses to a bucket in the main
// kernel are made by one thread, so program order already makes the digest and
// score stores of a key visible to the probe of the next key (plain 16-byte
// loads). The eviction scan reads the scores with cp.async, and whether that
// source read is ordered after the same thread's earlier plain stores is what
// test_sorted_insert_hw.py checks on hardware. What the answer could change is
// small: every slot this thread wrote in the call is still locked (I3), the
// scan skips a locked slot whatever score it read for it, and no slot the call
// has unlocked was written by it. The fence is kept until that test has passed
// on every architecture the library is built for; to measure without it, change
// the default below to 0 (setup.py has no knob for nvcc flags). A fence orders
// ordinary accesses, not the source read of an asynchronous copy, so it is not
// what would make a failing test pass.
#ifndef DEMB_SORTED_INSERT_THREADFENCE
#define DEMB_SORTED_INSERT_THREADFENCE 1
#endif

namespace dyn_emb {

// What an insert that evicted a key, or was refused, reports, at the key's
// SORTED position. The kernel writes all five arrays at every position (inert
// unless the flag is set), so that the compaction that follows reads defined
// memory. All null for a plain insert.
template <typename KeyType> struct EvictRecords {
  bool *flag;
  KeyType *keys;
  int64_t *scores;
  IndexType *indices;
  int64_t *table_ids;
};

// Put ``key`` back into the slot an insert locked: the release store of
// Bucket::unlock, from the address of the slot (what table_unlock_kernel does
// with the slots an ordinary insert reports).
template <typename Bucket, typename KeyType>
__device__ __forceinline__ void release_slot(KeyType *slot, KeyType key) {
  reinterpret_cast<typename Bucket::AtomicKey *>(slot)->store(
      key, cuda::std::memory_order_release);
}

// The first k in [0, count) with segments[busy[k]] >= bound. ``busy`` holds
// ascending sorted positions, so segments[busy[.]] is non-decreasing.
__device__ __forceinline__ int64_t
first_busy_at_or_after(int64_t const *segments, int64_t const *busy,
                       int64_t count, int64_t bound) {
  int64_t lo = 0, hi = count;
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (segments[busy[mid]] < bound) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// The deterministic insert. ``sorted_keys`` / ``segments`` / ``inverse`` come
// from bucketize_keys_segmented: position j holds the key that was at input
// position inverse[j], and segments[j] is its home bucket (num_buckets: takes
// no part). ``table_ids``, ``score_input``, ``insert_results`` and
// ``score_output`` are in INPUT order; ``score_output`` and ``insert_results``
// may be null. One thread per sorted position is launched, and only the thread
// at the head of the run of a bucket works on that bucket; it inserts the keys
// of the run in order. The other threads of the run exit; a position that takes
// no part is handled by its own thread.
//
// ``overflow_flag`` (null unless KernelTraits::EnableOverflow) is set for a key
// whose bucket answered Busy: the overflow tier decides in
// table_overflow_sorted_kernel, which then completes the record, insert_results
// and score_output of that key. ``counter`` is null for a table without
// reference counters. ``held_slots`` is scratch for ``batch`` pointers:
// position j holds the slot key j took (null when it took none), which stays
// locked until the run of its bucket is done (I3).
//
// GLOBAL_TIMER: two inserts of one thread may read the same timer value, and
// two equal scores would break the LRU order (the lowest slot wins a tie
// instead of the oldest key), which the waves, a launch apart, never had. The
// scores of a run are therefore kept strictly increasing: score = max(timer,
// previous + 1). A stored timestamp can run ahead of the timer by at most the
// number of keys of its run, in nanoseconds; it never reorders keys of
// different runs.
template <typename Table, typename KernelTraits>
__global__ void table_insert_sorted_kernel(
    Table table, int64_t const *__restrict__ table_bucket_offsets,
    int *__restrict__ bucket_sizes, int64_t batch,
    typename Table::KeyType const *__restrict__ sorted_keys,
    int64_t const *__restrict__ segments, int64_t const *__restrict__ inverse,
    int64_t const *__restrict__ table_ids, ScoreType *__restrict__ score_input,
    InsertResult *__restrict__ insert_results,
    int64_t *__restrict__ score_output, int32_t *__restrict__ counter,
    typename Table::KeyType **__restrict__ held_slots,
    EvictRecords<typename Table::KeyType> rec,
    bool *__restrict__ overflow_flag) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  static constexpr int BlockSize = KernelTraits::ThreadBlockDim;
  static constexpr int BufferDim = KernelTraits::NumScorePerThread;
  static constexpr int ProbingGroupSize = KernelTraits::ProbingGroupSize;
  static constexpr int ReductionGroupSize = KernelTraits::ReductionGroupSize;
  static constexpr ScorePolicyType PolicyType = KernelTraits::PolicyType;
  static constexpr bool UseOverflow = KernelTraits::EnableOverflow;
  static_assert(ProbingGroupSize == 1 && ReductionGroupSize == 1,
                "one thread works on a bucket: no tile");
  using Policy = ScorePolicy<PolicyType>;

  // reduce() indexes this by threadIdx.x: one private slice per thread, as in
  // insert_body.
  __shared__ ScoreType sm_scores[BlockSize * BufferDim];

  const int64_t num_buckets = static_cast<int64_t>(table.num_buckets_);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t head =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       head < batch; head += stride) {
    const int64_t segment = segments[head];

    if (segment >= num_buckets) {
      // Takes no part (I6): what insert_body does for table_cap == 0.
      const int64_t i = inverse[head];
      if (score_output != nullptr) {
        score_output[i] = static_cast<int64_t>(Policy::get(score_input, i));
      }
      if (insert_results != nullptr) {
        insert_results[i] = InsertResult::Illegal;
      }
      if (rec.flag != nullptr) {
        rec.flag[head] = false;
        rec.keys[head] = static_cast<KeyType>(Bucket::EmptyKey);
        rec.scores[head] = 0;
        rec.indices[head] = -1;
        rec.table_ids[head] = -1;
      }
      if (overflow_flag != nullptr) {
        overflow_flag[head] = false;
      }
      continue;
    }
    if (head > 0 && segments[head - 1] == segment) {
      continue; // the thread at the head of this run inserts it
    }

    ScoreType previous_score = 0; // GLOBAL_TIMER floor of this run
    int64_t j = head;
    for (; j < batch && segments[j] == segment; ++j) {
      const KeyType key = sorted_keys[j];
      const int64_t i = inverse[j];
      ScoreType score = Policy::get(score_input, i);
      if constexpr (PolicyType == ScorePolicyType::GlobalTimer) {
        if (score <= previous_score) {
          score = previous_score + 1;
        }
        previous_score = score;
      }

      // The bucket of a key that takes part IS its segment (I1); the rest is
      // the arithmetic of insert_body.
      const int64_t hashcode = Table::hash(key);
      const int64_t t_id = table_ids[i];
      const int64_t bkt_begin = table_bucket_offsets[t_id];
      const int64_t bucket_id = segment;
      Bucket bucket = table[bucket_id];
      Iter iter = Iter(hashcode % table.bucket_capacity());
      InsertResult result;
      insert_probe<ProbingGroupSize>(bucket, key, bucket_sizes, bucket_id, iter,
                                     &iter, &result);
      KeyType evict_key = KeyType();
      ScoreType evict_score = ScoreType();
      // Reference counters span all tables, as in insert_and_evict_body.
      insert<ReductionGroupSize, BufferDim, Policy, DefaultEvictor>(
          bucket, key, score, bucket_sizes, bucket_id, iter, sm_scores, result,
          &iter, &result, &evict_key, &evict_score, counter,
          bucket_id * bucket.capacity(), /*cur_ts=*/0);

      KeyType *table_key_slot = nullptr;
      if (isInsertSuccess(result)) {
        score = Policy::update(bucket.scores(iter), score);
        table_key_slot = bucket.keys(iter);
      }
      // I3: the slot stays locked, so that no later key of the run evicts this
      // one; it is released below, with the others of the run.
      held_slots[j] = table_key_slot;

      // Evictions, and refusals (Busy: the key itself is reported, with its
      // score and the encoding of the non-deterministic kernels, -(input
      // position + 1)). A Reclaim evicts no key.
      const bool refused = (result == InsertResult::Busy);
      const bool evicted = (result == InsertResult::Evict) || refused;
      if (rec.flag != nullptr) {
        rec.flag[j] = evicted;
        if (evicted) {
          rec.keys[j] = evict_key; // Busy: insert() set it to the key
          rec.scores[j] = static_cast<int64_t>(evict_score);
          rec.indices[j] =
              (result == InsertResult::Evict)
                  ? (bucket_id - bkt_begin) * bucket.capacity() + iter
                  : -(i + 1);
          rec.table_ids[j] = t_id;
        } else {
          rec.keys[j] = static_cast<KeyType>(Bucket::EmptyKey);
          rec.scores[j] = 0;
          rec.indices[j] = -1;
          rec.table_ids[j] = -1;
        }
      }
      const bool to_overflow = UseOverflow && refused;
      if (overflow_flag != nullptr) {
        overflow_flag[j] = to_overflow;
      }

      if (!to_overflow) {
        if (score_output != nullptr) {
          score_output[i] = static_cast<int64_t>(score);
        }
        if (insert_results != nullptr) {
          insert_results[i] = result;
        }
      }

#if DEMB_SORTED_INSERT_THREADFENCE
      __threadfence_block();
#endif
    }

    // The run is done: every slot it took goes back to its key.
    for (int64_t k = head; k < j; ++k) {
      if (held_slots[k] != nullptr) {
        release_slot<Bucket>(held_slots[k], sorted_keys[k]);
      }
    }
  }
}

// The overflow tier of a cache, applied in order. One thread per table: the
// keys whose bucket answered Busy are applied to the overflow bucket of their
// table in sorted order (bucket, key). The overflow bucket of table t is
// touched by thread t only, and the tables are independent, so the result is
// deterministic. The busy keys of table t are one contiguous range of ``busy``
// because the buckets of a table are contiguous and the positions are sorted by
// bucket.
//
// A slot a spill takes stays locked until every spill of the table has been
// applied (I3), as in the ordinary kernel, where it stays locked until
// table_unlock_kernel: a key this call put into the overflow bucket is never
// evicted by a later spill of the same call (its value row is only filled after
// the call). ``spill_slots`` holds, per busy position k, the locked slot of
// that key (null when it took none); it may be the buffer the main kernel used
// for ``held_slots``, which is released by then.
template <typename Table, typename KernelTraits>
__global__ void table_overflow_sorted_kernel(
    Table ovf_table, int64_t const *__restrict__ table_bucket_offsets,
    int64_t num_tables, int *__restrict__ ovf_bucket_sizes,
    int32_t *__restrict__ ovf_counter,
    int64_t const *__restrict__ ovf_output_offsets,
    typename Table::KeyType const *__restrict__ sorted_keys,
    int64_t const *__restrict__ segments, int64_t const *__restrict__ inverse,
    int64_t const *__restrict__ busy, int64_t const *__restrict__ num_busy,
    typename Table::KeyType **__restrict__ spill_slots,
    InsertResult *__restrict__ insert_results,
    int64_t *__restrict__ score_output,
    EvictRecords<typename Table::KeyType> rec) {

  using KeyType = typename Table::KeyType;
  using Bucket = typename Table::BucketType;
  using Policy = ScorePolicy<KernelTraits::PolicyType>;

  const int64_t t_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t_id >= num_tables) {
    return;
  }
  const int64_t count = *num_busy;
  const int64_t first =
      first_busy_at_or_after(segments, busy, count, table_bucket_offsets[t_id]);
  const int64_t last = first_busy_at_or_after(segments, busy, count,
                                              table_bucket_offsets[t_id + 1]);
  Bucket ovf_bucket = ovf_table[t_id];

  for (int64_t k = first; k < last; ++k) {
    const int64_t j = busy[k];
    const int64_t i = inverse[j];
    const KeyType key = sorted_keys[j];
    // A refused key's evict_score IS its score.
    ScoreType score = static_cast<ScoreType>(rec.scores[j]);
    IndexType index = -1;
    KeyType *table_key_slot = nullptr;
    InsertResult final_result = InsertResult::Busy;
    KeyType final_evict_key = rec.keys[j];        // the key itself
    IndexType final_evict_index = rec.indices[j]; // -(input position + 1)
    insert_overflow_tier<Policy>(
        ovf_bucket, key, Table::hash(key), ovf_bucket_sizes, t_id, ovf_counter,
        ovf_output_offsets[t_id], score, index, final_result, table_key_slot,
        final_evict_key, final_evict_index);
    spill_slots[k] = table_key_slot; // locked until the end of the pass (I3)
    // Insert / Assign: nothing is evicted. Evict: the overflow victim, with the
    // refused key's score (what insert_and_evict_body reports). Busy again: the
    // key stays refused, with the index of the main bucket's Busy.
    rec.flag[j] = (final_result == InsertResult::Evict) ||
                  (final_result == InsertResult::Busy);
    rec.keys[j] = final_evict_key;
    rec.indices[j] = final_evict_index;
    if (score_output != nullptr) {
      score_output[i] = static_cast<int64_t>(score);
    }
    if (insert_results != nullptr) {
      insert_results[i] = final_result;
    }
  }

  // Every spill of this table is placed: the keys go back into their slots.
  for (int64_t k = first; k < last; ++k) {
    if (spill_slots[k] != nullptr) {
      release_slot<Bucket>(spill_slots[k], sorted_keys[busy[k]]);
    }
  }
}

} // namespace dyn_emb
