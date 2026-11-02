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
// The rows section of the window (RowsLayout): the out channel's data.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/sizes.h"

namespace dyn_emb {
namespace hier_output_dist {

// The rows section, its buffers in this order (those marked num_nodes > 1
// exist across nodes only):
//   lookup_rows  the owner's lookup output, row i for its KJT's key i; the
//                backward writes the gradient of that output over it
//   proxy_rows   (num_nodes > 1) the proxy's rows, round by round (formats.h):
//                the rows of the node-unique keys the rail owners put
//                (forward), the gradient rows of its node's requesters and
//                their sums (backward)
//   push_flags   (num_nodes > 1) the requesters' push-task flags, zeroed when
//                the window is made: those of round rnd are in half rnd % 2
// Each accessor returns its buffer in the copy at `base`.
class RowsLayout {
 public:
  __host__ __device__ RowsLayout(const Sizes &sizes, char *base)
      : base_(base),
        row_bytes_(sizes.row_bytes),
        half_flags_(sizes.multi_node() ? half_flags(sizes.num_proxy_rows, sizes.topo.local_size)
                                       : 0) {
    const bool multi = sizes.multi_node();
    proxy_rows_ = buffer_end(sizes.lookup_keys * row_bytes_);
    push_flags_ = proxy_rows_ + buffer_end(multi ? sizes.num_proxy_rows * row_bytes_ : 0);
    bytes_ = push_flags_ + buffer_end(push_flags_bytes());
  }
  __host__ __device__ uint64_t bytes() const { return bytes_; }
  // Each buffer's name and the bytes it takes (HierOutputDist::memory).
  std::vector<std::pair<const char *, uint64_t>> buffers() const {
    return {{"lookup_rows", proxy_rows_},
            {"proxy_rows", push_flags_ - proxy_rows_},
            {"push_flags", bytes_ - push_flags_}};
  }

  __host__ __device__ char *lookup_rows() const { return base_; }
  __host__ __device__ char *lookup_row(int64_t row) const {
    return lookup_rows() + row * row_bytes_;
  }
  __host__ __device__ char *proxy_rows() const { return base_ + proxy_rows_; }
  __host__ __device__ char *proxy_row(int64_t slot) const {
    return proxy_rows() + slot * row_bytes_;
  }
  // The flags of round rnd's half.
  __host__ __device__ uint64_t *push_flags(int64_t rnd) const {
    return (uint64_t *)(base_ + push_flags_) + (rnd & 1) * half_flags_;
  }
  // The bytes of both halves, from push_flags(0).
  __host__ __device__ uint64_t push_flags_bytes() const { return 16 * (uint64_t)half_flags_; }

 private:
  char *base_;
  int64_t row_bytes_, half_flags_;
  uint64_t proxy_rows_, push_flags_, bytes_;  // lookup_rows starts at base
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
