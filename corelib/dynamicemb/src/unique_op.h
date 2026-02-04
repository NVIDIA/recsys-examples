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
 * @brief Deduplicate keys using a GPU hash table.
 *
 * This function allocates internal hash table buffers automatically based on
 * input size. The buffers are temporary and freed after the operation.
 * Uses the current CUDA stream.
 *
 * @param keys Input keys tensor (int64 or uint64)
 * @param frequency_counters Optional: output frequency counter per unique key
 * @param input_frequencies Optional: input frequency per key (for weighted
 * counting)
 *
 * @return Tuple of (unique_keys, output_indices, num_unique)
 *         - unique_keys: Deduplicated keys
 *         - output_indices: Index mapping (input idx -> unique idx)
 *         - num_unique: Scalar tensor with count of unique keys
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor>
unique_cuda(at::Tensor keys, at::Tensor frequency_counters = at::Tensor(),
            at::Tensor input_frequencies = at::Tensor());

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
 *
 * @return Tuple of (unique_keys, output_indices, table_offsets)
 *         - unique_keys: Compacted unique keys with size=num_keys (same as
 *           input). Only first table_offsets[num_tables] elements are valid.
 *         - output_indices: Index mapping (input idx -> global unique idx)
 *         - table_offsets: Tensor of size (num_tables + 1) with cumulative
 *           counts. table_offsets[num_tables] contains total unique count.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor>
segmented_unique_cuda(at::Tensor keys, at::Tensor table_ids, int64_t num_tables,
                      int64_t device_sm_count);

} // namespace dyn_emb

// Python binding
void bind_unique_op(pybind11::module &m);
