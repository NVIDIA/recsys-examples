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

#include "../check.h"
#include "torch_utils.h"
#include "types.cuh"

#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda/std/tuple>

#ifdef DEMB_USE_PYBIND11
#include <torch/extension.h>
#endif
#include <torch/torch.h>

#ifdef DEMB_USE_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

#define DISPATCH_KEY_TYPE(DATA_TYPE, HINT, ...)                                \
  switch (DATA_TYPE) {                                                         \
    CASE_TYPE_USING_HINT(DataType::Int64, int64_t, HINT, __VA_ARGS__)          \
    CASE_TYPE_USING_HINT(DataType::UInt64, uint64_t, HINT, __VA_ARGS__)        \
  default:                                                                     \
    throw std::runtime_error("Not supported key type.");                       \
  }

#define DISPATCH_SCORE_TYPE(DATA_TYPE, HINT, ...)                              \
  switch (DATA_TYPE) {                                                         \
    CASE_TYPE_USING_HINT(DataType::UInt64, uint64_t, HINT, __VA_ARGS__)        \
    CASE_TYPE_USING_HINT(DataType::UInt32, uint32_t, HINT, __VA_ARGS__)        \
  default:                                                                     \
    throw std::runtime_error("Not supported score type.");                     \
  }

#define DISPATCH_SCORE_POLICY(SCORE_POLICY, HINT, ...)                         \
  switch (SCORE_POLICY) {                                                      \
    CASE_ENUM_USING_HINT(ScorePolicyType::Const, HINT, __VA_ARGS__)            \
    CASE_ENUM_USING_HINT(ScorePolicyType::Assign, HINT, __VA_ARGS__)           \
    CASE_ENUM_USING_HINT(ScorePolicyType::Accumulate, HINT, __VA_ARGS__)       \
    CASE_ENUM_USING_HINT(ScorePolicyType::GlobalTimer, HINT, __VA_ARGS__)      \
    CASE_ENUM_USING_HINT(ScorePolicyType::LruLfu, HINT, __VA_ARGS__)           \
    CASE_ENUM_USING_HINT(ScorePolicyType::Flag, HINT, __VA_ARGS__)             \
  default:                                                                     \
    throw std::runtime_error("Not supported score policy.");                   \
  }

namespace dyn_emb {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
table_lookup(at::Tensor table_storage, at::Tensor table_bucket_offsets,
             int64_t bucket_capacity, at::Tensor keys, at::Tensor table_ids,
             std::optional<at::Tensor> score_input, ScorePolicyType policy_type,
             std::optional<at::Tensor> ovf_storage = std::nullopt,
             int64_t ovf_bucket_capacity = 0,
             std::optional<at::Tensor> ovf_output_offsets = std::nullopt,
             int64_t num_scores = 1);

at::Tensor table_insert(at::Tensor table_storage,
                        at::Tensor table_bucket_offsets,
                        int64_t bucket_capacity, at::Tensor bucket_sizes,
                        at::Tensor keys, at::Tensor table_ids,
                        std::optional<at::Tensor> score_input,
                        ScorePolicyType policy_type, std::optional<at::Tensor> counter,
                        std::optional<at::Tensor> insert_results = std::nullopt,
                        std::optional<at::Tensor> score_output = std::nullopt,
                        int64_t num_scores = 1, int64_t score_fn_key = 0);

// Plain insert that also retains each evicted victim's (key, table_id). Returns
// (indices, num_evicted[int64 scalar], evicted_keys, evicted_table_ids); the
// evicted_* tensors are sized at the batch upper bound -- slice to num_evicted.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
table_insert_collect_evicted(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, std::optional<at::Tensor> counter,
    std::optional<at::Tensor> insert_results = std::nullopt,
    std::optional<at::Tensor> score_output = std::nullopt,
    int64_t num_scores = 1, int64_t score_fn_key = 0);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor>
table_insert_and_evict(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, std::optional<at::Tensor> counter,
    std::optional<at::Tensor> insert_results = std::nullopt,
    std::optional<at::Tensor> score_output = std::nullopt,
    std::optional<at::Tensor> ovf_storage = std::nullopt,
    int64_t ovf_bucket_capacity = 0,
    std::optional<at::Tensor> ovf_bucket_sizes = std::nullopt,
    std::optional<at::Tensor> ovf_counter = std::nullopt,
    std::optional<at::Tensor> ovf_output_offsets = std::nullopt,
    int64_t num_scores = 1, int64_t score_fn_key = 0);

// The deterministic insert of DEMB_DETERMINISM_MODE (see sorted_insert.cuh):
// the keys are sorted by (home bucket, key) and the keys of each bucket are
// inserted one after another by one thread, smallest key first, so that a
// bucket sees the same operations whatever the order of the batch and the
// scheduling of the GPU. Nothing is read back and no shape depends on the data.
// A key that takes part in no insert (a reserved key, a key of a table without
// buckets or that does not exist) is Illegal. ``insert_results`` and
// ``score_output`` (input order) are filled for every input position when
// given; the indices are not returned (a lookup after the insert gives them).
// One score word only: LRU_LFU is refused.
void table_insert_sorted(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, std::optional<at::Tensor> counter,
    std::optional<at::Tensor> insert_results = std::nullopt,
    std::optional<at::Tensor> score_output = std::nullopt,
    int64_t num_scores = 1);

// The same insert that reports its evictions and refusals, in sorted (bucket,
// key) order, with the count left on the device. Returns (num_evicted [1],
// evicted_keys [N], evicted_indices [N], evicted_scores [N], evicted_table_ids
// [N]): only the first num_evicted entries are real, the rest is inert (key -1,
// index -1, score 0, table id -1). A refused key (Busy) is reported with the
// index -(its input position + 1). With ``ovf_storage`` (a cache) the keys
// whose bucket is Busy go to the overflow bucket of their table in sorted
// order.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
table_insert_and_evict_sorted(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, std::optional<at::Tensor> counter,
    std::optional<at::Tensor> insert_results = std::nullopt,
    std::optional<at::Tensor> score_output = std::nullopt,
    std::optional<at::Tensor> ovf_storage = std::nullopt,
    int64_t ovf_bucket_capacity = 0,
    std::optional<at::Tensor> ovf_bucket_sizes = std::nullopt,
    std::optional<at::Tensor> ovf_counter = std::nullopt,
    std::optional<at::Tensor> ovf_output_offsets = std::nullopt,
    int64_t num_scores = 1);

void table_erase_by_score(at::Tensor table_storage, torch::Dtype key_dtype,
                          int64_t bucket_capacity, at::Tensor bucket_sizes,
                          std::optional<at::Tensor> counter,
                          ScoreType threshold, int64_t begin, int64_t end,
                          int64_t num_scores, int64_t score_index,
                          at::Tensor counts);

void table_erase(at::Tensor table_storage, at::Tensor table_bucket_offsets,
                 int64_t bucket_capacity, at::Tensor bucket_sizes,
                 at::Tensor keys, at::Tensor table_ids,
                 std::optional<at::Tensor> indices, int64_t num_scores = 1);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
table_export_batch(at::Tensor table_storage, int64_t bucket_capacity,
                   int64_t batch, int64_t offset, torch::Dtype key_dtype,
                   std::optional<ScoreType> threshold = std::nullopt,
                   int64_t table_begin = 0, int64_t num_scores = 1,
                   int64_t score_index = 0);

at::Tensor table_count_matched(at::Tensor table_storage, torch::Dtype key_dtype,
                               int64_t bucket_capacity, ScoreType threshold,
                               int64_t begin = -1, int64_t end = -1,
                               int64_t num_scores = 1, int64_t score_index = 0);

void table_copy_score_blocks(at::Tensor src_storage, int64_t src_bucket_capacity,
                             at::Tensor dst_storage, int64_t dst_bucket_capacity,
                             int64_t num_scores, int64_t src_bkt_begin,
                             int64_t dst_bkt_begin, at::Tensor src_slots,
                             at::Tensor dst_slots, torch::Dtype key_dtype);

at::Tensor table_gather_score_blocks(at::Tensor table_storage,
                                     int64_t bucket_capacity, int64_t num_scores,
                                     int64_t bkt_begin, at::Tensor slots,
                                     torch::Dtype key_dtype);

void table_scatter_score_blocks(at::Tensor table_storage,
                                int64_t bucket_capacity, int64_t num_scores,
                                int64_t bkt_begin, at::Tensor slots,
                                at::Tensor values, torch::Dtype key_dtype);

// Write-back for replay_increment: place (key, score words) at the given
// table-relative slots. Returns (status, same_key) -- status is the slot
// written, or -1 when the key's home bucket does not contain that slot (the
// caller raises), same_key marks slots that already held the same key.
std::tuple<at::Tensor, at::Tensor> table_scatter_keys_at_slots(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, at::Tensor slots, at::Tensor scores,
    int64_t num_scores = 1);

std::vector<at::Tensor> table_partition(at::Tensor storage,
                                        std::vector<torch::Dtype> dtypes,
                                        int64_t bucket_capacity,
                                        int64_t num_buckets);

std::vector<at::Tensor> tensor_partition(at::Tensor input,
                                         std::vector<int64_t> byte_range,
                                         std::vector<torch::Dtype> dtypes);

std::vector<at::Tensor> bucketize_keys(at::Tensor keys, at::Tensor table_ids,
                                       at::Tensor table_bucket_offsets,
                                       int64_t num_buckets,
                                       int64_t bucket_capacity);

// The sort of bucketize_keys without its read back: returns (sorted keys,
// inverse, segment of every sorted position), where the segment is the key's
// home bucket, or num_buckets for a key that takes no part in an insert.
// Nothing is compacted, so no shape depends on the data (see bucketize.cu).
std::vector<at::Tensor>
bucketize_keys_segmented(at::Tensor keys, at::Tensor table_ids,
                         at::Tensor table_bucket_offsets, int64_t num_buckets,
                         int64_t bucket_capacity);

void table_update_counter_with_layout(
    at::Tensor counter, at::Tensor slot_indices, int32_t delta,
    at::Tensor table_bucket_offsets, int64_t bucket_capacity,
    int64_t main_capacity, int64_t num_tables,
    c10::optional<at::Tensor> table_ids,
    c10::optional<at::Tensor> overflow_output_offsets,
    int64_t overflow_bucket_capacity);

// Resolve table-relative hash slots to the value rows they own, taking the
// next row of the slot's table for a slot that has none yet when `allocate`;
// a row at or past the table's `row_limits` entry is asserted on the device.
at::Tensor table_resolve_rows(at::Tensor rows, at::Tensor row_counters,
                              at::Tensor row_limits,
                              at::Tensor table_bucket_offsets,
                              int64_t bucket_capacity, at::Tensor slots,
                              at::Tensor table_ids, bool allocate);

// Rehash support, with no count on the host (see table_rows.cu).
// table_carry_slot_state moves what is keyed by the slot -- the value row and the
// reference counter -- from the old table's slots of a chunk of its key column to the
// slots the keys got in the new table (`slots`, -1 for a reserved key, which carries
// nothing); a real key without a slot is asserted on the device. The row pair and the
// counter pair are each given whole or left out.
void table_carry_slot_state(at::Tensor keys, at::Tensor slots, int64_t dst_base,
                            std::optional<at::Tensor> src_rows,
                            std::optional<at::Tensor> dst_rows,
                            std::optional<at::Tensor> src_ref,
                            std::optional<at::Tensor> dst_ref);
// table_carry_reclaimed_rows gives the row of every reclaimed slot of the old
// table's bucket range [old_bkt_begin, old_bkt_end) that still owns one a free slot
// of the new table's range, marking the slot reclaimed: its key becomes ReclaimKey
// and its digest that key's. `count` (int64[1], on the device) is how many such rows
// there are: the table's row counter minus its live keys.
void table_carry_reclaimed_rows(at::Tensor old_keys, at::Tensor old_rows,
                                int64_t old_bkt_begin, int64_t old_bkt_end,
                                int64_t bucket_capacity, at::Tensor new_keys,
                                at::Tensor new_digests, at::Tensor new_rows,
                                int64_t new_bkt_begin, int64_t new_bkt_end,
                                at::Tensor count);

// Restore the reclaimed slots a dump's [n, 2] (slot, row) int64 names, owning
// their rows, on a replica: an empty slot becomes reclaimed with the row, a
// reclaimed one must already own it, and a live key still in the slot is erased
// first (see table_rows.cu).
void table_restore_reclaimed_slots(at::Tensor table_storage,
                                   at::Tensor table_bucket_offsets,
                                   int64_t bucket_capacity,
                                   at::Tensor bucket_sizes, at::Tensor rows,
                                   at::Tensor slot_index, int64_t table_id,
                                   int64_t num_scores, torch::Dtype key_dtype);

} // namespace dyn_emb
