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
#include <type_traits>
#include <vector>

#include <cuda/std/tuple>

#include <torch/extension.h>
#include <torch/torch.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
  default:                                                                     \
    throw std::runtime_error("Not supported score policy.");                   \
  }

namespace dyn_emb {

inline int get_size(torch::ScalarType scalar_type) {
  switch (scalar_type) {
  case torch::kUInt8:
    return 1;
  case torch::kInt8:
    return 1;
  case torch::kInt16:
    return 2;
  case torch::kInt32:
    return 4;
  case torch::kInt64:
    return 8;
  case torch::kFloat32:
    return 4;
  case torch::kFloat64:
    return 8;
  case torch::kBool:
    return 1;
  case torch::kBFloat16:
    return 2;
  case torch::kFloat16:
    return 2;
  case torch::kUInt16:
    return 2;
  case torch::kUInt32:
    return 4;
  case torch::kUInt64:
    return 8;
  default:
    throw std::runtime_error("Unsupported scalar type.");
  }
}

void table_lookup(at::Tensor table_storage, std::vector<torch::Dtype> dtypes,
                  int64_t bucket_capacity, at::Tensor keys,
                  std::vector<std::optional<at::Tensor>> scores,
                  std::vector<ScorePolicyType> policy_types,
                  std::vector<bool> is_returns, at::Tensor founds,
                  std::optional<at::Tensor> indices);

void table_insert(at::Tensor table_storage, std::vector<torch::Dtype> dtypes,
                  int64_t bucket_capacity, at::Tensor bucket_sizes,
                  at::Tensor keys,
                  std::vector<std::optional<at::Tensor>> scores,
                  std::vector<ScorePolicyType> policy_types,
                  std::vector<bool> is_returns,
                  std::optional<at::Tensor> indices,
                  std::optional<at::Tensor> insert_results);

void table_insert_and_evict(
    at::Tensor table_storage, std::vector<torch::Dtype> dtypes,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    std::vector<std::optional<at::Tensor>> scores,
    std::vector<ScorePolicyType> policy_types, std::vector<bool> is_returns,
    std::optional<at::Tensor> insert_results, std::optional<at::Tensor> indices,
    at::Tensor num_evicted, at::Tensor evicted_keys, at::Tensor evicted_indices,
    std::vector<at::Tensor> evicted_scores);

void table_erase(at::Tensor table_storage, std::vector<torch::Dtype> dtypes,
                 int64_t bucket_capacity, at::Tensor bucket_sizes,
                 at::Tensor keys, std::optional<at::Tensor> indices);

void table_export_batch(at::Tensor table_storage,
                        std::vector<torch::Dtype> dtypes,
                        int64_t bucket_capacity, int64_t batch, int64_t offset,
                        at::Tensor counter, at::Tensor keys,
                        std::vector<std::optional<at::Tensor>> scores,
                        std::optional<at::Tensor> indices);

std::vector<at::Tensor> table_partition(at::Tensor storage,
                                        std::vector<torch::Dtype> dtypes,
                                        int64_t bucket_capacity,
                                        int64_t num_buckets);

std::vector<at::Tensor> tensor_partition(at::Tensor input,
                                         std::vector<int64_t> byte_range,
                                         std::vector<torch::Dtype> dtypes);

} // namespace dyn_emb
