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
// Rows of the output dist: a warp copy of rows of 16-byte vectors, the
// vectors of each row dtype as fp32 values, and a product rounded to nearest.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <type_traits>

#include "hier_output_dist/device/warp.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// Every lane holds one row of up to 32: `src`/`dst` byte pointers (lanes with
// lane >= num_rows hold anything). Copies num_rows rows of vec 16-byte vectors
// each, 8 vectors in flight per lane. Loads go through L2 only
// (ld.global.cg): a row another GPU wrote must not come from a stale L1 line.
__device__ __forceinline__ void warp_copy_rows(const char *src, char *dst, int num_rows, int vec,
                                               int lane) {
  constexpr int kUnroll = 8;
  const int total = num_rows * vec;
  for (int base = 0; base < total; base += 32 * kUnroll) {
    int4 vecs[kUnroll];
#pragma unroll
    for (int step = 0; step < kUnroll; ++step) {
      const int idx = base + step * 32 + lane;
      const int row = idx / vec;
      const char *src_row = from_lane(src, row & 31);
      if (idx < total) vecs[step] = __ldcg((const int4 *)src_row + (idx - row * vec));
    }
#pragma unroll
    for (int step = 0; step < kUnroll; ++step) {
      const int idx = base + step * 32 + lane;
      const int row = idx / vec;
      char *dst_row = from_lane(dst, row & 31);
      if (idx < total) *((int4 *)dst_row + (idx - row * vec)) = vecs[step];
    }
  }
}

// A row value as fp32 (exactly), and an fp32 value rounded once to the row
// dtype (to nearest even).
__device__ __forceinline__ float to_float(float value) { return value; }
__device__ __forceinline__ float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
__device__ __forceinline__ float to_float(__half value) { return __half2float(value); }
template <typename RowT>
__device__ __forceinline__ RowT from_float(float value) {
  if constexpr (std::is_same<RowT, __nv_bfloat16>::value)
    return __float2bfloat16_rn(value);
  else if constexpr (std::is_same<RowT, __half>::value)
    return __float2half_rn(value);
  else
    return value;
}

// 16-byte vectors of the row dtype as kValues fp32 values: get converts
// them, add adds them in fp32, pack rounds them back.
template <typename RowT>
struct Vec16 {
  static constexpr int kValues = 16 / sizeof(RowT);
  __device__ static void get(float *vals, int4 vec) {
    const RowT *elems = reinterpret_cast<const RowT *>(&vec);
#pragma unroll
    for (int i = 0; i < kValues; ++i) vals[i] = to_float(elems[i]);
  }
  __device__ static void add(float *acc, int4 vec) {
    const RowT *elems = reinterpret_cast<const RowT *>(&vec);
#pragma unroll
    for (int i = 0; i < kValues; ++i) acc[i] += to_float(elems[i]);
  }
  __device__ static int4 pack(const float *vals) {
    int4 vec;
    RowT *elems = reinterpret_cast<RowT *>(&vec);
#pragma unroll
    for (int i = 0; i < kValues; ++i) elems[i] = from_float<RowT>(vals[i]);
    return vec;
  }
};

// value * scale rounded to nearest, never fused into a multiply-add and with
// subnormals kept, whatever the build's fast-math flags: what PyTorch's
// division of a CUDA tensor by a CPU scalar computes (a multiplication by the
// reciprocal, in fp32), bit for bit.
__device__ __forceinline__ float mul_rn(float value, float scale) {
  float product;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(product) : "f"(value), "f"(scale));
  return product;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
