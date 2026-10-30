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
// A binary search: the segment a value falls in.
#pragma once

#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

// Smallest i in [0, count) with bounds[(i + 1) * stride] > value, for
// non-decreasing bounds[0], bounds[stride], ..., bounds[count * stride] with
// bounds[0] <= value < bounds[count * stride]: the segment value falls in,
// empty segments skipped. bounds[count * stride] is never read, so it need not
// exist.
template <typename BoundT>
__device__ __forceinline__ int seg_search(const BoundT *bounds, int count, int64_t value,
                                          int stride = 1) {
  int lo = 0, hi = count - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if ((int64_t)bounds[(int64_t)mid * stride] <= value)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
