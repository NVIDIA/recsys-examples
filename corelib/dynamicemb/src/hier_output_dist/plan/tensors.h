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
// stream, none of which waits for the device; and the sizes the steps across
// nodes lay their tables out by. host/common.h brings bits_for and the
// constants, after the CUDA headers a host compiler needs for them.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>

#include "hier_output_dist/common/topology.h"
#include "hier_output_dist/host/common.h"

namespace dyn_emb {
namespace hier_output_dist {

// The sizes of an op's calls across nodes, from its configuration
// (host/layout.h): every table of a step is sized by them, never by a device
// value.
struct PlanSizes {
  Topo topo;
  int64_t num_features;
  int64_t lookup_keys, remote_keys;  // the capacities (entries)
  int64_t num_proxy_rows;            // rows of proxy_rows
  int64_t rounds_cap;                // the most rounds of a call (rounds_for(remote_keys,
                                     // num_proxy_rows)), what a round table holds
  int64_t partial_rows;              // rows of proxy_rows a requester's fp32 partial row takes
  int64_t chunk_rows;                // rows of an IB chunk, a multiple of kTaskRows
  int64_t record_bytes;              // bytes of a key record: 8, or 16 with its weight
};

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

// A field of a call's Meta (common/meta.h) at word first_word: `count` int64
// words, or `count` int32 entries packed in whole words.
inline at::Tensor meta_words(const at::Tensor &meta, int64_t first_word, int64_t count) {
  return meta.narrow(0, first_word, count);
}
inline at::Tensor meta_int32(const at::Tensor &meta, int64_t first_word, int64_t count) {
  return meta.narrow(0, first_word, (count + 1) / 2).view(at::kInt).narrow(0, 0, count);
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
