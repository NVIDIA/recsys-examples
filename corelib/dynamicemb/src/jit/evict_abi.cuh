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

// ABI between the LruLfu evict cubin (evict_lrulfu.cu, compiled by nvcc) and the
// C++ launcher that loads/launches it (jit_link.cpp, compiled by the host
// compiler). It MUST be includable by both, so it uses only <cstdint> and plain
// POD -- no CUDA device headers, no dynamicemb type aliases. The struct layout
// is the launch contract: one struct passed by value as the single kernel arg
// (cuLaunchKernel kernelParams = { &params }), so there is no hand-packed
// per-argument void* list to get wrong.
//
// KeyType/IndexType/CounterType are all int64, ScoreType is uint64, InsertResult
// is a uint8 enum -- fixed for LruLfu (§11.2), so this ABI is monomorphic. The
// cubin exposes exactly two entry points (one EnableOverflow variant each):
//   dyn_emb_evict_entry_ovf(EvictParams)     -- overflow-enabled tables
//   dyn_emb_evict_entry_noovf(EvictParams)   -- no-overflow tables
#pragma once

#include <cstdint>

namespace dyn_emb {

struct EvictParams {
  // --- main table (reconstruct LinearBucketTable device-side) ---
  uint8_t *table_storage;
  uint64_t num_buckets;
  int64_t bucket_capacity;
  int64_t num_scores; // == 2 for LruLfu

  // --- per-op inputs/outputs (mirror table_insert_and_evict_kernel args) ---
  const int64_t *table_bucket_offsets;
  int *bucket_sizes;
  int64_t batch;
  const int64_t *input_keys;
  const int64_t *table_ids;
  uint8_t *insert_results; // InsertResult*
  int64_t *indices;        // IndexType*
  uint64_t *score_input;   // ScoreType* (frequency deltas)
  int64_t *score_output;   // nullable
  int64_t **table_key_slots;
  int64_t *evicted_counter; // CounterType*
  int64_t *evicted_keys;
  int64_t *evicted_scores;
  int64_t *evicted_indices; // IndexType*
  int64_t *evicted_table_ids;
  int32_t *counter;

  // --- overflow tier (fields ignored / may be null for the _noovf entry) ---
  uint8_t *ovf_storage;
  uint64_t ovf_num_buckets;
  int64_t ovf_bucket_capacity;
  int *ovf_bucket_sizes;
  int32_t *ovf_counter;
  const int64_t *ovf_output_offsets;
};

} // namespace dyn_emb
