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

} // namespace dyn_emb

// Python binding
void bind_unique_op(pybind11::module &m);
