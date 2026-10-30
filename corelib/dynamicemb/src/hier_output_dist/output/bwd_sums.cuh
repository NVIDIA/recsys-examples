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
// The backward's sums of gradient rows, in fp32 and in a fixed order, and its
// scaled copies of them.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/device/rows.cuh"
#include "hier_output_dist/device/warp.cuh"

namespace dyn_emb {
namespace hier_output_dist {

template <typename RowT>
__device__ __forceinline__ int4 scaled(float *vals, float scale) {
#pragma unroll
  for (int i = 0; i < Vec16<RowT>::kValues; ++i) vals[i] = mul_rn(vals[i], scale);
  return Vec16<RowT>::pack(vals);
}

// Vector `col` of the gradient rows of original rows exp_idx[begin .. end),
// begin < end, summed in fp32 in that order into acc.
template <typename RowT>
__device__ __forceinline__ void sum_vec(const char *grad, const int32_t *exp_idx, int64_t begin,
                                        int64_t end, int row_bytes, int col, float *acc) {
  Vec16<RowT>::get(acc, __ldcg((const int4 *)(grad + (int64_t)exp_idx[begin] * row_bytes) + col));
  for (int64_t entry = begin + 1; entry < end; entry += 4) {
    int4 vecs[4];
#pragma unroll
    for (int step = 0; step < 4; ++step)
      vecs[step] =
          entry + step < end
              ? __ldcg((const int4 *)(grad + (int64_t)exp_idx[entry + step] * row_bytes) + col)
              : make_int4(0, 0, 0, 0);
#pragma unroll
    for (int step = 0; step < 4; ++step)
      if (entry + step < end) Vec16<RowT>::add(acc, vecs[step]);
  }
}

// fp32 values vals[0 .. Values) into vector `col` (`Values` floats) of the
// fp32 row at `row`.
template <int Values>
__device__ __forceinline__ void store_fp32(float *row, int col, const float *vals) {
#pragma unroll
  for (int i = 0; i < Values; i += 4)
    *(float4 *)(row + col * Values + i) =
        make_float4(vals[i], vals[i + 1], vals[i + 2], vals[i + 3]);
}

// Vector `col` (`Values` floats) of fp32 rows part[0], part[dim], ...
// part[(count - 1) * dim], count >= 1, summed in that order into acc, 4 rows'
// loads in flight.
template <int Values>
__device__ __forceinline__ void sum_fp32(const float *part, int64_t count, int dim, int col,
                                         float *acc) {
#pragma unroll
  for (int i = 0; i < Values; i += 4) {
    const float4 val = __ldcg((const float4 *)(part + col * Values + i));
    acc[i] = val.x;
    acc[i + 1] = val.y;
    acc[i + 2] = val.z;
    acc[i + 3] = val.w;
  }
  for (int64_t j = 1; j < count; j += 4) {
    float4 vals[4][Values / 4];
#pragma unroll
    for (int step = 0; step < 4; ++step)
#pragma unroll
      for (int i = 0; i < Values; i += 4)
        vals[step][i / 4] =
            j + step < count ? __ldcg((const float4 *)(part + (j + step) * dim + col * Values + i))
                             : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      if (j + step >= count) continue;
#pragma unroll
      for (int i = 0; i < Values; i += 4) {
        acc[i] += vals[step][i / 4].x;
        acc[i + 1] += vals[step][i / 4].y;
        acc[i + 2] += vals[step][i / 4].z;
        acc[i + 3] += vals[step][i / 4].w;
      }
    }
  }
}

// Each lane's row at its src, scaled, into its dst: the lanes over (row,
// vector) pairs as in warp_copy_rows, 4 vectors in flight per lane. A row
// whose dst is nullptr is skipped, and one whose src alone is nullptr gets
// zeros.
template <typename RowT>
__device__ __forceinline__ void scaled_copy_rows(const char *src, char *dst, int num_rows, int vec,
                                                 int lane, float scale) {
  const int total = num_rows * vec;
  for (int base = 0; base < total; base += 32 * 4) {
    int4 vecs[4];
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int idx = base + step * 32 + lane;
      const int row = idx / vec;
      const char *src_row = from_lane(src, row & 31);
      vecs[step] = idx < total && src_row != nullptr
                       ? __ldcg((const int4 *)src_row + (idx - row * vec))
                       : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int idx = base + step * 32 + lane;
      const int row = idx / vec;
      char *dst_row = from_lane(dst, row & 31);
      if (idx < total && dst_row != nullptr) {
        float vals[Vec16<RowT>::kValues];
        Vec16<RowT>::get(vals, vecs[step]);
        *((int4 *)dst_row + (idx - row * vec)) = scaled<RowT>(vals, scale);
      }
    }
  }
}

// Vector `col` (when `live`) of the rows of reducer-list entries [begin, end)
// in proxy_rows, summed in fp32 in list order into acc: rows of the key's
// type, or (Partial) a heavy key's fp32 partial rows, those of requesters
// without one (kEntryNoPush) skipped. Each lane loads its entry of each 32,
// the warp then 8 rows at a time.
template <typename RowT, bool Partial>
__device__ __forceinline__ void sum_entries(const char *proxy_rows, const int64_t *ent,
                                            int64_t begin, int64_t end, int row_bytes, int col,
                                            bool live, int lane, float *acc) {
  constexpr int kValues = Vec16<RowT>::kValues,
                kLoads = Partial ? kValues / 4 : 1;  // 16-byte loads per vector
  bool any = false;
  for (int64_t block0 = begin; block0 < end; block0 += 32) {
    const int block_len = (int)min((int64_t)32, end - block0);
    const uint64_t mine = lane < block_len ? (uint64_t)ent[block0 + lane] : 0;
    for (int i = 0; i < block_len; i += 8) {
      int4 vecs[8][kLoads];
      bool use[8];
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        const uint64_t entry = __shfl_sync(kFull, mine, (i + step) & 31);
        use[step] = i + step < block_len && (!Partial || (int32_t)(entry >> 32) != kEntryNoPush);
        const int4 *src =
            (const int4 *)(proxy_rows + (int64_t)(entry & ~kEntryPartial) * row_bytes) +
            col * kLoads;
#pragma unroll
        for (int load = 0; load < kLoads; ++load)
          vecs[step][load] = use[step] && live ? __ldcg(src + load) : make_int4(0, 0, 0, 0);
      }
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        if (!use[step]) continue;
        float vals[kValues];
        if constexpr (Partial)
          memcpy(vals, vecs[step], sizeof(vals));
        else
          Vec16<RowT>::get(vals, vecs[step][0]);
#pragma unroll
        for (int elem = 0; elem < kValues; ++elem)
          acc[elem] = any ? acc[elem] + vals[elem] : vals[elem];
        any = true;
      }
    }
  }
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
