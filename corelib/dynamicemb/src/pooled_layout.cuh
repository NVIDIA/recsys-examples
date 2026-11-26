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

#ifndef POOLED_LAYOUT_CUH
#define POOLED_LAYOUT_CUH

#include "utils.h"

namespace dyn_emb {

// Bag geometry of a pooled lookup, device side. The spec lives in
// dynamicemb/lookup/embedding_layout.py -- read that first; this mirrors the pooled
// subset of its EmbeddingLayout, member for member, so the two cannot drift.
// Only the pooled subset: a sequence lookup emits one row per key, has no bags
// to number, and never reads any of this.
//
// The one-line version: a pooled lookup has F*B bags, one per (feature,
// sample), and the input and the output number them differently. The input
// (offsets) is feature-major, s = f*B + b. The output is batch-major,
// r = b*F + f. They are transposes of each other, and mixing them up is the
// classic bug in this code, so go through these helpers instead of writing the
// arithmetic.
//
// POD by design: passed to kernels by value, every accessor forceinline. Do
// not give it a constructor, virtuals, or anything that stops it being an
// aggregate.
struct PooledLayout {
  int batch_size;  // B
  int feature_num; // F
  int total_D;     // sum of every feature's dim; the pooled row width
  // Column width of one feature when they all share a dim. Equal to the
  // storage's row width, which is why gather_embedding_pooled can pass either
  // max_D or the source tensor's width here.
  int max_D;
  // [F+1] prefix sums, or nullptr when every feature has the same dim.
  const int *__restrict__ D_offsets;

  HOST_DEVICE_INLINE bool mixed_D() const { return D_offsets != nullptr; }
  HOST_DEVICE_INLINE int num_bags() const { return feature_num * batch_size; }

  // -- numbering a bag ---------------------------------------------------

  HOST_DEVICE_INLINE int input_index(int f, int b) const {
    return f * batch_size + b;
  }
  HOST_DEVICE_INLINE int output_index(int f, int b) const {
    return b * feature_num + f;
  }
  HOST_DEVICE_INLINE int feature_of_input(int s) const { return s / batch_size; }
  HOST_DEVICE_INLINE int sample_of_input(int s) const { return s % batch_size; }
  HOST_DEVICE_INLINE int feature_of_output(int r) const {
    return r % feature_num;
  }
  HOST_DEVICE_INLINE int sample_of_output(int r) const {
    return r / feature_num;
  }
  // The transpose. Backward applies it to every key when it builds the
  // GradInfo that says which row of the incoming gradient that key reads.
  HOST_DEVICE_INLINE int output_index_of_input(int s) const {
    return output_index(feature_of_input(s), sample_of_input(s));
  }

  // -- placing a feature in the pooled output ----------------------------

  HOST_DEVICE_INLINE int col_begin(int f) const {
    return D_offsets ? D_offsets[f] : f * max_D;
  }
  HOST_DEVICE_INLINE int col_width(int f) const {
    return D_offsets ? (D_offsets[f + 1] - D_offsets[f]) : max_D;
  }
};

} // namespace dyn_emb
#endif // POOLED_LAYOUT_CUH
