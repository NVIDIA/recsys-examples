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

#include "counted_limit.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
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
// every entry is live. Kernels read n with counted_limit (counted_limit.cuh).

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

// Blocks of a grid-stride launch that wants `blocks` of them (to cover its
// threads, or one per row): the same number, but capped the way the vec4
// kernels cap theirs (4 waves of resident blocks), so that an empty batch costs
// one cheap launch, not a block per capacity entry.
inline int counted_grid(int64_t blocks, int block_size) {
  const cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
  // The hardware keeps at most 32 blocks of a kernel resident on one SM (a
  // block-per-row kernel can have a block of a few threads).
  const int64_t blocks_per_sm = std::min(
      32, std::max(1, prop->maxThreadsPerMultiProcessor /
                          std::max(1, block_size)));
  const int64_t max_grid =
      static_cast<int64_t>(prop->multiProcessorCount) * blocks_per_sm * 4;
  return static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(blocks, max_grid)));
}

// Glue ops on capacity-N arrays (counted_ops.cu). `idx` is int64[N]; the ops
// touch entries i < n only, and skip those with idx[i] < 0. counted_gather and
// counted_scatter take 1-D tensors, or 2-D tensors whose rows are contiguous
// (the row stride is free), of dtype bool / int8 / int32 / int64 / uint64 /
// float16 / bfloat16 / float32.

// out[i] = src[idx[i]], into a new uninitialised tensor with the rows of idx.
at::Tensor counted_gather(at::Tensor src, at::Tensor idx,
                          std::optional<at::Tensor> num_valid);

// dst[idx[i]] = src[i] (src has >= N rows). Repeated indices: an arbitrary one
// wins.
void counted_scatter(at::Tensor dst, at::Tensor idx, at::Tensor src,
                     std::optional<at::Tensor> num_valid);

// dst[idx[i]] = value, for a 1-D bool (true when value != 0) or int64 dst.
void counted_scatter_fill(at::Tensor dst, at::Tensor idx, int64_t value,
                          std::optional<at::Tensor> num_valid);

} // namespace dyn_emb

#ifdef DEMB_USE_PYBIND11
void bind_counted_ops(pybind11::module &m);
#endif
