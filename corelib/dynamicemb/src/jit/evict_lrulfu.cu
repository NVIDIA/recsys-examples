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

// LruLfu eviction cubin translation unit. Compiled twice by setup.py:
//   -DDEMB_EVICT_COMPARATOR=LexFreqTsComparator  -> complete fatbin (default,
//       no numba); exact (freq asc, ts asc) eviction.
//   -DDEMB_EVICT_COMPARATOR=UserFnComparator      -> LTO-IR fatbin with
//       user_score_fn undefined; nvJitLink links the numba-compiled user fn.
//
// It reuses the shared insert_and_evict_body<Table, KernelTraits, Evictor> from
// kernels.cuh with RankedEvictor<Comparator>, so the insert/evict logic is not
// duplicated. Traits are fixed for LruLfu: KeyType=int64, CompactTileSize=1
// (always correct), OutputScore=true (score_output null-checked at runtime),
// PolicyType=LruLfu. Only EnableOverflow varies, via two entry points.

#include "evict_abi.cuh"
#include "evict_comparators.cuh"
#include "table_operation/kernels.cuh"

namespace dyn_emb {

using KeyType = int64_t;
using Bucket = LinearBucket<KeyType>;
using EvictTable = LinearBucketTable<Bucket>;

#ifndef DEMB_EVICT_COMPARATOR
#define DEMB_EVICT_COMPARATOR LexFreqTsComparator
#endif
using EvictComparator = DEMB_EVICT_COMPARATOR;

__device__ __forceinline__ uint64_t read_globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t));
  return t;
}

template <bool EnableOverflow>
__device__ __forceinline__ void run_evict(const EvictParams &p,
                                          uint64_t cur_ts) {
  using KernelTraits =
      InsertKernelTraits<256, 1, 1, /*CompactTileSize=*/1,
                         /*NumScorePerThread=*/8, ScorePolicyType::LruLfu,
                         /*OutputScore=*/true, EnableOverflow>;

  EvictTable table(p.table_storage, p.num_buckets, p.bucket_capacity,
                   p.num_scores);
  EvictTable ovf_table =
      EnableOverflow ? EvictTable(p.ovf_storage, p.ovf_num_buckets,
                                  p.ovf_bucket_capacity, p.num_scores)
                     : EvictTable();

  insert_and_evict_body<EvictTable, KernelTraits, RankedEvictor<EvictComparator>>(
      table, p.table_bucket_offsets, p.bucket_sizes, p.batch, p.input_keys,
      p.table_ids, reinterpret_cast<InsertResult *>(p.insert_results),
      p.indices, p.score_input, p.score_output, p.table_key_slots,
      p.evicted_counter, p.evicted_keys, p.evicted_scores, p.evicted_indices,
      p.evicted_table_ids, p.counter, ovf_table, p.ovf_bucket_sizes,
      p.ovf_counter, p.ovf_output_offsets, cur_ts);
}

extern "C" __global__ void dyn_emb_evict_entry_ovf(EvictParams p) {
  run_evict<true>(p, read_globaltimer());
}

extern "C" __global__ void dyn_emb_evict_entry_noovf(EvictParams p) {
  run_evict<false>(p, read_globaltimer());
}

// Plain insert (no overflow tier, no evicted-output collection). A full bucket
// still evicts, but via RankedEvictor<Comparator> -- so a 2-word LruLfu table
// never falls back to the single-score reduce() in table_insert_kernel. Reuses
// EvictParams (evicted_* / ovf_* fields unused here).
__device__ __forceinline__ void run_insert(const EvictParams &p,
                                           uint64_t cur_ts) {
  using KernelTraits =
      InsertKernelTraits<256, 1, 1, /*CompactTileSize=*/1,
                         /*NumScorePerThread=*/8, ScorePolicyType::LruLfu,
                         /*OutputScore=*/true>;

  EvictTable table(p.table_storage, p.num_buckets, p.bucket_capacity,
                   p.num_scores);

  insert_body<EvictTable, KernelTraits, RankedEvictor<EvictComparator>>(
      table, p.table_bucket_offsets, p.bucket_sizes, p.batch, p.input_keys,
      p.table_ids, reinterpret_cast<InsertResult *>(p.insert_results),
      p.indices, p.score_input, p.score_output, p.table_key_slots, p.counter,
      cur_ts);
}

extern "C" __global__ void dyn_emb_insert_entry(EvictParams p) {
  run_insert(p, read_globaltimer());
}

} // namespace dyn_emb
