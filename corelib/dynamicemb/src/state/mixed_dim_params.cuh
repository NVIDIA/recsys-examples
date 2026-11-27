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

// What a kernel takes about tables of different embedding dimensions, passed by
// value. Each struct is the device-side half of one view in mixed_dim_view.h
// (PaddedBufferView::params(), FlatTablesView::params()); every field is
// always set. Neither holds the element type, which is a template parameter the
// host picks, or all_dims_vec4, which only picks the kernel.

#pragma once

#include <cstdint>

namespace dyn_emb {

// A padded buffer: [n, max_value_dim] rows, each the embedding padded to
// max_emb_dim and followed by the optimizer state.
struct PaddedBufferParams {
  int64_t const *emb_dims; // [num_tables]
  int64_t max_emb_dim;     // where the optimizer state starts in a row
  int64_t max_value_dim;   // row stride of the buffer
};

// One buffer per table, each [rows, value_dim] with its own value_dim.
struct FlatTablesParams {
  int64_t const *ptrs;       // [num_tables], base address of each table's buffer
  int64_t const *value_dims; // [num_tables]
  int64_t const *emb_dims;   // [num_tables]
  int64_t max_emb_dim;
  int64_t max_value_dim;
};

} // namespace dyn_emb
