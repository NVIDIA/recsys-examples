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
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <cub/cub.cuh>

#include "hier_output_dist/plan/sort.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// cub sizes its scratch on the host first, then sorts; the scratch comes from
// the caching allocator on the current stream.
template <typename Key>
void radix_sort(const at::Tensor &keys, const at::Tensor &values, at::Tensor &sorted_keys,
                at::Tensor &sorted_values, int end_bit) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int count = (int)keys.numel();
  size_t scratch_bytes = 0;
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, scratch_bytes, keys.data_ptr<Key>(), sorted_keys.data_ptr<Key>(),
      values.data_ptr<int32_t>(), sorted_values.data_ptr<int32_t>(), count, 0, end_bit, stream));
  at::Tensor scratch = at::empty({(int64_t)scratch_bytes}, keys.options().dtype(at::kByte));
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      scratch.data_ptr(), scratch_bytes, keys.data_ptr<Key>(), sorted_keys.data_ptr<Key>(),
      values.data_ptr<int32_t>(), sorted_values.data_ptr<int32_t>(), count, 0, end_bit, stream));
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
    case at::kShort:
      radix_sort<int16_t>(keys, values, sorted_keys, sorted_values, end_bit);
      break;
    case at::kInt:
      radix_sort<int32_t>(keys, values, sorted_keys, sorted_values, end_bit);
      break;
    case at::kLong:
      radix_sort<int64_t>(keys, values, sorted_keys, sorted_values, end_bit);
      break;
    default:
      TORCH_CHECK(false, "hier_output_dist: sort_pairs sorts int16, int32 or int64 keys, got ",
                  keys.scalar_type());
  }
  return {sorted_keys, sorted_values};
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
