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
// The planner's CUB calls (planner.h): radix sorts and scans, in this one
// translation unit. Each sizes its scratch on the host first, then runs; the
// scratch comes from the caching allocator on the current stream.

#include <ATen/cuda/CUDAContext.h>
#include <thrust/iterator/transform_iterator.h>

#include <cstdint>
#include <cub/cub.cuh>

#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

at::Tensor scratch_of(size_t bytes, const at::Tensor &like) {
  return at::empty({(int64_t)bytes}, like.options().dtype(at::kByte));
}

template <typename Key>
void radix_sort_pairs(const at::Tensor &keys, const at::Tensor &values, at::Tensor &sorted_keys,
                      at::Tensor &sorted_values, int end_bit) {
  const int count = (int)keys.numel();
  size_t bytes = 0;
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, bytes, keys.data_ptr<Key>(), sorted_keys.data_ptr<Key>(), values.data_ptr<int32_t>(),
      sorted_values.data_ptr<int32_t>(), count, 0, end_bit, stream()));
  at::Tensor scratch = scratch_of(bytes, keys);
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      scratch.data_ptr(), bytes, keys.data_ptr<Key>(), sorted_keys.data_ptr<Key>(),
      values.data_ptr<int32_t>(), sorted_values.data_ptr<int32_t>(), count, 0, end_bit, stream()));
}

// A count as the offsets' type, as the scan reads it.
template <typename To>
struct CastTo {
  template <typename From>
  __host__ __device__ To operator()(From value) const {
    return (To)value;
  }
};

// offsets[1 .. n] = the inclusive sums of the n counts.
template <typename From, typename To>
void inclusive_sums(const at::Tensor &counts, const at::Tensor &offsets) {
  const int64_t count = counts.numel();
  auto in = thrust::make_transform_iterator(counts.data_ptr<From>(), CastTo<To>{});
  To *out = offsets.data_ptr<To>() + 1;
  size_t bytes = 0;
  AT_CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, bytes, in, out, count, stream()));
  at::Tensor scratch = scratch_of(bytes, counts);
  AT_CUDA_CHECK(cub::DeviceScan::InclusiveSum(scratch.data_ptr(), bytes, in, out, count, stream()));
}

template <typename To>
void inclusive_sums_to(const at::Tensor &counts, const at::Tensor &offsets) {
  switch (counts.scalar_type()) {
    case at::kBool:
      inclusive_sums<bool, To>(counts, offsets);
      break;
    case at::kInt:
      inclusive_sums<int32_t, To>(counts, offsets);
      break;
    case at::kLong:
      inclusive_sums<int64_t, To>(counts, offsets);
      break;
    default:
      TORCH_CHECK(false, "hier_output_dist: offsets_of sums bool, int32 or int64 counts, got ",
                  counts.scalar_type());
  }
}

}  // namespace

std::pair<at::Tensor, at::Tensor> sort_pairs(const at::Tensor &keys, const at::Tensor &values,
                                             int end_bit) {
  TORCH_CHECK(keys.is_cuda() && keys.is_contiguous() && values.is_contiguous() &&
                  values.scalar_type() == at::kInt && values.numel() == keys.numel() &&
                  keys.numel() < INT32_MAX && end_bit >= 1 &&
                  end_bit <= 8 * (int)keys.element_size(),
              "hier_output_dist: sort_pairs takes contiguous keys on the GPU, an int32 value "
              "per key, fewer than 2^31, and an end bit within the keys' width");
  at::Tensor sorted_keys = at::empty_like(keys), sorted_values = at::empty_like(values);
  if (keys.numel() == 0) return {sorted_keys, sorted_values};
  switch (keys.scalar_type()) {
    case at::kInt:
      radix_sort_pairs<int32_t>(keys, values, sorted_keys, sorted_values, end_bit);
      break;
    case at::kLong:
      radix_sort_pairs<int64_t>(keys, values, sorted_keys, sorted_values, end_bit);
      break;
    default:
      TORCH_CHECK(false, "hier_output_dist: sort_pairs sorts int32 or int64 keys, got ",
                  keys.scalar_type());
  }
  return {sorted_keys, sorted_values};
}

at::Tensor sort_keys(const at::Tensor &keys, int end_bit) {
  TORCH_CHECK(keys.is_cuda() && keys.is_contiguous() && keys.scalar_type() == at::kLong &&
                  keys.numel() < INT32_MAX && end_bit >= 1 && end_bit <= 64,
              "hier_output_dist: sort_keys sorts contiguous int64 keys on the GPU, fewer than "
              "2^31, by at most 64 bits");
  at::Tensor sorted = at::empty_like(keys);
  const int count = (int)keys.numel();
  if (count == 0) return sorted;
  size_t bytes = 0;
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys.data_ptr<int64_t>(),
                                               sorted.data_ptr<int64_t>(), count, 0, end_bit,
                                               stream()));
  at::Tensor scratch = scratch_of(bytes, keys);
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(scratch.data_ptr(), bytes, keys.data_ptr<int64_t>(),
                                               sorted.data_ptr<int64_t>(), count, 0, end_bit,
                                               stream()));
  return sorted;
}

at::Tensor offsets_of(const at::Tensor &counts, at::ScalarType dtype) {
  TORCH_CHECK(counts.is_cuda() && (dtype == at::kInt || dtype == at::kLong),
              "hier_output_dist: offsets_of takes counts on the GPU, and gives int32 or "
              "int64 offsets");
  const at::Tensor values = counts.contiguous().view(-1);
  const at::Tensor offsets = at::empty({values.numel() + 1}, values.options().dtype(dtype));
  AT_CUDA_CHECK(cudaMemsetAsync(offsets.data_ptr(), 0, offsets.element_size(), stream()));
  if (values.numel() == 0) return offsets;
  if (dtype == at::kInt)
    inclusive_sums_to<int32_t>(values, offsets);
  else
    inclusive_sums_to<int64_t>(values, offsets);
  return offsets;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
