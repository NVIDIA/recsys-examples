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

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <tuple>

namespace dyn_emb {

/**
 * @brief Segmented unique operation that deduplicates keys per table.
 *
 * This function performs unique operation on keys partitioned by table_ids.
 * Keys are deduplicated within each table independently, allowing the same key
 * to appear in different tables. Uses compound hashing on (key, table_id) pairs
 * with a single shared hash table for memory efficiency.
 *
 * NOTE: This function is fully asynchronous with no GPU-CPU synchronization.
 *
 * @param keys Input keys tensor (int64 or uint64)
 * @param table_ids Table ID for each key (int32, same length as keys,
 *                  must be in ascending order)
 * @param num_tables Total number of tables
 * @param device_sm_count Number of SMs on the device (used to determine
 *                        optimal grid size for kernel launches)
 * @param input_frequencies Controls frequency counting behavior:
 *                          - Undefined/empty tensor with numel()==0: Enable
 *                            frequency counting with each occurrence counted as
 * 1
 *                          - Tensor with numel()==num_keys: Use provided
 *                            frequencies for weighted counting
 *                          - Pass None from Python to disable frequency
 * counting entirely (output freq_counters will be empty)
 *
 * @return Tuple of (num_uniques, unique_keys, output_indices, table_offsets,
 *         freq_counters)
 *         - num_uniques: Tensor of size 1 containing total unique count
 *           (view of table_offsets[num_tables])
 *         - unique_keys: Compacted unique keys with size=num_keys (same as
 *           input). Only first num_uniques elements are valid.
 *         - output_indices: Index mapping (input idx -> global unique idx)
 *         - table_offsets: Tensor of size (num_tables + 1) with cumulative
 *           counts.
 *         - freq_counters: Frequency counts per unique key. Empty if frequency
 *           counting is disabled (input_frequencies was None).
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
segmented_unique_cuda(at::Tensor keys, at::Tensor table_ids, int64_t num_tables,
                      int64_t device_sm_count,
                      at::Tensor input_frequencies = at::Tensor());

/**
 * @brief Expand table IDs from offsets.
 *
 * Generates a table_id for each element based on offsets structure.
 * This is a helper function to prepare input for segmented_unique_cuda.
 *
 * @param offsets Jagged tensor offsets (int64)
 *                Size = num_features * local_batch_size + 1
 *                Indexed by (feature_id * local_batch_size + batch_id)
 *
 * @param table_offsets_in_feature Feature offsets per table (int64), or None
 *                Size = num_tables + 1
 *                Maps features to tables (adjacent features may share a table)
 *                table_offsets_in_feature[t] is the first feature index for
 * table t If None: each feature is treated as a separate table
 *
 * @param num_tables Number of tables (ignored if table_offsets_in_feature is
 * None)
 * @param local_batch_size Batch size per feature
 * @param num_elements Total number of elements (keys)
 * @param device_sm_count Number of SMs on the device
 *
 * @return table_ids tensor (int32) with same length as num_elements
 */
at::Tensor
expand_table_ids_cuda(at::Tensor offsets,
                      c10::optional<at::Tensor> table_offsets_in_feature,
                      int64_t num_tables, int64_t local_batch_size,
                      int64_t num_elements, int64_t device_sm_count);

/**
 * @brief Compute new lengths and offsets by evenly distributing unique keys.
 *
 * This is a GPU kernel that evenly distributes unique keys across (feature,
 * batch) buckets. For each table, unique keys are distributed so each bucket
 * gets (unique_count / num_buckets) keys, with the first (unique_count %
 * num_buckets) buckets getting one extra.
 *
 * @param unique_offsets Cumulative unique counts per table (int64, device)
 * @param table_offsets_in_feature Feature offsets per table (int64, device)
 * @param num_tables Number of tables
 * @param local_batch_size Batch size per feature
 * @param new_lengths_size Total size of output (num_features *
 * local_batch_size)
 *
 * @return Tuple of (new_lengths, new_offsets)
 */
std::tuple<at::Tensor, at::Tensor> compute_dedup_lengths_cuda(
    at::Tensor unique_offsets, at::Tensor table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size, int64_t new_lengths_size);

} // namespace dyn_emb

// Python binding
void bind_unique_op(pybind11::module &m);
