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
#ifdef DEMB_USE_PYBIND11
#include <pybind11/pybind11.h>
#endif
#include <tuple>

namespace dyn_emb {

/**
 * @brief Segmented unique operation that deduplicates keys per table.
 *
 * Keys must be pre-sorted by table: keys[segmented_range[t]:segmented_range[t+1]]
 * all belong to table t. Uses compound hashing on (key, table_id) pairs with a
 * single shared hash table for memory efficiency.
 *
 * NOTE: This function is fully asynchronous with no GPU-CPU synchronization.
 *
 * @param keys Input keys tensor (int64 or uint64), sorted by table.
 * @param segmented_range Table boundary offsets (int64, size = num_tables+1).
 *                        segmented_range[t] is the start index in keys for
 *                        table t; segmented_range[num_tables] == num_keys.
 * @param num_tables Total number of tables
 * @param input_frequencies Controls frequency counting behavior:
 *                          - Undefined/empty tensor with numel()==0: Enable
 *                            frequency counting with each occurrence counted as 1
 *                          - Tensor with numel()==num_keys: Use provided
 *                            frequencies for weighted counting
 *                          - Pass None from Python to disable frequency
 *                            counting entirely (output freq_counters will be empty)
 *
 * @return Tuple of (num_uniques, unique_keys, output_indices, table_offsets,
 *         freq_counters)
 *         - num_uniques: Tensor of size 1 containing total unique count
 *           (view of table_offsets[num_tables])
 *         - unique_keys: Compacted unique keys with size=num_keys (same as
 *           input). Only first num_uniques elements are valid.
 *         - output_indices: Index mapping (input idx -> global unique idx)
 *         - table_offsets: Tensor of size (num_tables + 1) with cumulative
 *           unique counts per table.
 *         - freq_counters: Frequency counts per unique key. Empty if frequency
 *           counting is disabled (input_frequencies was None).
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
segmented_unique_cuda(at::Tensor keys, at::Tensor segmented_range,
                      int64_t num_tables,
                      at::Tensor input_frequencies = at::Tensor());

/**
 * @brief Expand table IDs from offsets (identity mapping: one feature per table,
 * local_batch_size=1).
 *
 * Generates a table_id for each element via binary search on offsets.
 * num_tables is derived from offsets.size(0)-1.
 *
 * @param offsets Jagged tensor offsets (int64, size = num_tables + 1)
 *                offsets[t] is the start index for table t's keys.
 * @param num_elements Total number of elements (keys)
 *
 * @return table_ids tensor (int64) with same length as num_elements
 */
at::Tensor expand_table_ids_cuda(at::Tensor offsets, int64_t num_elements);

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
#ifdef DEMB_USE_PYBIND11
void bind_unique_op(pybind11::module &m);
#endif
