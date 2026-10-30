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
// Row copies of the forward under index dedup and for heavy keys: a row to
// each of its original rows, and rows of one source read once.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/device/rows.cuh"
#include "hier_output_dist/device/warp.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// Copies of the row at `src` into original rows exp_idx[begin .. end) of
// `out`. The row is read once: lanes hold its 16-byte vectors, and a short
// row is written to several copies at a time.
__device__ __forceinline__ void fanout_row(const char *src, int64_t begin, int64_t end, int vec,
                                           int lane, const int32_t *exp_idx, char *out,
                                           int row_bytes) {
  if (vec <= 32) {
    const int per = 32 / vec, col = lane % vec, slot = lane / vec;
    if (slot >= per) return;
    const int4 val = __ldcg((const int4 *)src + col);
    for (int64_t j = begin + slot; j < end; j += per)
      *((int4 *)(out + (int64_t)exp_idx[j] * row_bytes) + col) = val;
    return;
  }
  for (int col = lane; col < vec; col += 32) {
    const int4 val = __ldcg((const int4 *)src + col);
    for (int64_t j = begin; j < end; ++j)
      *((int4 *)(out + (int64_t)exp_idx[j] * row_bytes) + col) = val;
  }
}

// Index dedup: each lane's row (at `src`, its deduplicated row `dedup_row`)
// goes to every original row of that deduplicated row. All rows go to their
// first original row in one warp copy; the few that repeat are then copied on
// from there (local memory, whatever the source was). Heavy rows are left to
// their sub-tasks, but for their first original row when `heavy_first` (their
// sub-tasks then copy from there).
__device__ __forceinline__ void warp_copy_fanout(const char *src, int64_t dedup_row, int num_rows,
                                                 int vec, int lane, const Fanout &fanout, char *out,
                                                 int row_bytes, bool heavy_first) {
  int64_t first_entry = 0, cnt = 0;
  char *dst = nullptr;
  if (lane < num_rows) {
    first_entry = fanout.exp_off[dedup_row];
    cnt = fanout.exp_off[dedup_row + 1] - first_entry;
    if (cnt > 0 && (cnt <= kHeavy || heavy_first))
      dst = out + (int64_t)fanout.exp_idx[first_entry] * row_bytes;
  }
  if (__ballot_sync(kFull, lane < num_rows && dst == nullptr) == 0) {
    warp_copy_rows(src, dst, num_rows, vec, lane);
  } else {
    const int total = num_rows * vec;
    for (int base = 0; base < total; base += 32) {
      const int idx = base + lane;
      const int row = idx / vec;
      const char *src_row = from_lane(src, row & 31);
      char *dst_row = from_lane(dst, row & 31);
      if (idx < total && dst_row != nullptr)
        *((int4 *)dst_row + (idx - row * vec)) = __ldcg((const int4 *)src_row + (idx - row * vec));
    }
  }
  unsigned many = __ballot_sync(kFull, cnt > 1 && cnt <= kHeavy);
  if (!many) return;
  __syncwarp();
  while (many) {
    const int src_lane = __ffs(many) - 1;
    many &= many - 1;
    const int64_t lane_begin = __shfl_sync(kFull, first_entry, src_lane);
    const int64_t lane_count = __shfl_sync(kFull, cnt, src_lane);
    const char *first = from_lane(dst, src_lane);
    fanout_row(first, lane_begin + 1, lane_begin + lane_count, vec, lane, fanout.exp_idx, out,
               row_bytes);
  }
}

// Lanes whose rows have the same source (the occurrences of a heavy key, all
// at its one row in proxy_rows) read it once: its vectors in the lanes, stored
// to each of their destinations.
__device__ __forceinline__ void warp_copy_shared(const char *src, char *dst, int num_rows, int vec,
                                                 int lane) {
  for (unsigned todo = __ballot_sync(kFull, lane < num_rows); todo;) {
    const char *src_row = from_lane(src, __ffs(todo) - 1);
    const unsigned same = __ballot_sync(kFull, lane < num_rows && src == src_row);
    todo &= ~same;
    for (int col_base = 0; col_base < vec; col_base += 32) {
      const int col = col_base + lane;
      const int4 val = col < vec ? __ldcg((const int4 *)src_row + col) : make_int4(0, 0, 0, 0);
      for (unsigned mask = same; mask; mask &= mask - 1) {
        char *dst_row = from_lane(dst, __ffs(mask) - 1);
        if (col < vec) *((int4 *)dst_row + col) = val;
      }
    }
  }
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
