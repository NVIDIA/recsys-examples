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

#include <cstdint>
#include <cuda_runtime.h>

namespace dyn_emb {

// n = clamp(*num_valid, 0, capacity), read on the device: how many leading
// entries of a per-key array of capacity `capacity` are live; all of them
// without a count (nullptr). The contract of `num_valid` is in counted_ops.h.
// This header includes no torch header, so that table_operation/kernels.cuh,
// which the evict fatbin builds with plain nvcc (setup.py), can use it.
__host__ __device__ __forceinline__ int64_t
counted_limit(const int64_t *num_valid, int64_t capacity) {
  if (num_valid == nullptr) {
    return capacity;
  }
  const int64_t n = *num_valid;
  return n < 0 ? 0 : (n > capacity ? capacity : n);
}

} // namespace dyn_emb
