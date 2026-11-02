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
// Small helpers of the planner's steps (plan/): ATen ops on the current
// stream, none of which waits for the device. Every table of a step is sized
// by the op's sizes (Sizes, from its configuration: host/sizes.h), never by
// a device value. host/common.h brings bits_for and the constants, after the
// CUDA headers a host compiler needs for them.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>

#include "hier_output_dist/common/sizes.h"
#include "hier_output_dist/host/common.h"

namespace dyn_emb {
namespace hier_output_dist {

// The ends of n counts, [0, c0, c0 + c1, ...]: n + 1 entries, int64 (or
// int32 when their sum stays below 2^31).
inline at::Tensor exclusive_ends(const at::Tensor &counts, at::ScalarType dtype = at::kLong) {
  return at::cat({at::zeros({1}, counts.options().dtype(dtype)), counts.cumsum(0, dtype)});
}

// value / divisor rounded up, for values of at least 0. (On an integer tensor
// `/` divides exactly and gives floating point.)
inline at::Tensor ceil_div(const at::Tensor &value, int64_t divisor) {
  return (value + (divisor - 1)).div(divisor, "floor");
}
inline at::Tensor ceil_div(const at::Tensor &value, const at::Tensor &divisor) {
  return (value + divisor - 1).div(divisor, "floor");
}

// Where each entry of a permutation of [0, n) is: result[permutation[i]] = i,
// in the permutation's dtype.
inline at::Tensor inverse_permutation(const at::Tensor &permutation) {
  return at::empty_like(permutation)
      .scatter_(0, permutation.to(at::kLong),
                at::arange(permutation.numel(), permutation.options()));
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
