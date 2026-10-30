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
// What every host file of the hierarchical dist shares: the error checks of
// the CUDA runtime, the CUDA driver and NCCL, and small helpers.
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>

#include "hier_output_dist/common/constants.h"

#define HIER_NCCL_CHECK(cmd)                                                \
  do {                                                                      \
    ncclResult_t result_ = (cmd);                                           \
    TORCH_CHECK(result_ == ncclSuccess, "hier_output_dist: NCCL failure '", \
                ncclGetErrorString(result_), "' at " #cmd);                 \
  } while (0)
#define HIER_CUDA_CHECK(cmd)                                                                       \
  do {                                                                                             \
    cudaError_t err_ = (cmd);                                                                      \
    TORCH_CHECK(err_ == cudaSuccess, "hier_output_dist: CUDA failure '", cudaGetErrorString(err_), \
                "' at " #cmd);                                                                     \
  } while (0)
#define HIER_CU_CHECK(cmd)                                                                       \
  do {                                                                                           \
    CUresult result_ = (cmd);                                                                    \
    TORCH_CHECK(result_ == CUDA_SUCCESS, "hier_output_dist: CUDA driver failure ", (int)result_, \
                " at " #cmd);                                                                    \
  } while (0)

namespace dyn_emb {
namespace hier_output_dist {

inline int sms_of(int device) {
  int sms;
  HIER_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  return sms;
}
// CTAs of kThreads threads for `work` items, a thread each, at most two per SM.
inline int grid_for(int64_t work, int sms) {
  return (int)std::min<int64_t>((std::max<int64_t>(work, 1) + kThreads - 1) / kThreads,
                                2 * (int64_t)sms);
}
// Bits that hold the values [0, value).
inline int bits_for(int64_t value) {
  int bits = 1;
  while ((int64_t{1} << bits) < value) ++bits;
  return bits;
}
// Whether `tensor` is a contiguous CUDA tensor of `count` elements of `dtype`.
inline bool is_cuda_array(const at::Tensor &tensor, at::ScalarType dtype, int64_t count) {
  return tensor.is_cuda() && tensor.is_contiguous() && tensor.scalar_type() == dtype &&
         tensor.numel() == count;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
