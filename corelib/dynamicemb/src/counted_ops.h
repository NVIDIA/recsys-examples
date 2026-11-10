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

#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <optional>
#ifdef DEMB_USE_PYBIND11
#include <pybind11/pybind11.h>
#endif

namespace dyn_emb {

// Per-key arrays are allocated at a capacity the host knows (the number of
// input keys); the number of live leading entries is a device tensor
// `num_valid`: int64, exactly one element (any view with numel() == 1, e.g.
// table_offsets[T:T+1]). Entries from n = clamp(*num_valid, 0, capacity) on are
// undefined memory that no kernel reads or writes. Without a count (nullptr)
// every entry is live.

// n, read on the device.
__host__ __device__ __forceinline__ int64_t
counted_limit(const int64_t *num_valid, int64_t capacity) {
  if (num_valid == nullptr) {
    return capacity;
  }
  const int64_t n = *num_valid;
  return n < 0 ? 0 : (n > capacity ? capacity : n);
}

// Checks an optional `num_valid` tensor and returns the device pointer the
// kernels read, or nullptr when there is none. Takes c10::optional and
// std::optional alike.
template <typename OptionalTensor>
inline const int64_t *counted_num_valid_ptr(const OptionalTensor &num_valid,
                                            const at::Device &device) {
  if (!num_valid.has_value() || !num_valid.value().defined()) {
    return nullptr;
  }
  const at::Tensor &t = num_valid.value();
  TORCH_CHECK(t.is_cuda() && t.device() == device,
              "num_valid must be a CUDA tensor on the device of the data");
  TORCH_CHECK(t.scalar_type() == at::kLong && t.numel() == 1,
              "num_valid must be a one-element int64 tensor");
  return t.data_ptr<int64_t>();
}

// Glue ops on capacity-N arrays (counted_ops.cu). `idx` is int64[N]; the ops
// touch entries i < n only, and skip those with idx[i] < 0. 1-D tensors, or 2-D
// tensors whose rows are contiguous (the row stride is free), of dtype bool /
// int8 / int32 / int64 / uint64 / float16 / bfloat16 / float32.

// out[i] = src[idx[i]]. `out` (rows >= N) is allocated, uninitialised, when
// absent.
at::Tensor counted_gather(at::Tensor src, at::Tensor idx,
                          std::optional<at::Tensor> num_valid,
                          std::optional<at::Tensor> out);

// dst[idx[i]] = src[i] (src has >= N rows). Repeated indices: an arbitrary one
// wins.
void counted_scatter(at::Tensor dst, at::Tensor idx, at::Tensor src,
                     std::optional<at::Tensor> num_valid);

// dst[idx[i]] = value, converted to the dtype of dst (an error if it does not
// fit).
void counted_scatter_fill(at::Tensor dst, at::Tensor idx, double value,
                          std::optional<at::Tensor> num_valid);

// The number of true flags among the first n, as an int64[1] device tensor.
at::Tensor counted_count_true(at::Tensor flags,
                              std::optional<at::Tensor> num_valid);

} // namespace dyn_emb

#ifdef DEMB_USE_PYBIND11
void bind_counted_ops(pybind11::module &m);
#endif
