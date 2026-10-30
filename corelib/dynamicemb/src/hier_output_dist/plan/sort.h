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
// The planner's one sort: cub's radix sort of keys that carry int32 values.
#pragma once

#include <ATen/ATen.h>

#include <utility>

namespace dyn_emb {
namespace hier_output_dist {

// A stable sort of `keys` (int16, int32 or int64, contiguous) by their bits
// [0, end_bit), which carries `values` (int32, one per key), on the current
// stream: the sorted keys, and the values in their order. With end_bit below
// the keys' width the keys must not be negative.
std::pair<at::Tensor, at::Tensor> sort_pairs(const at::Tensor &keys, const at::Tensor &values,
                                             int end_bit);

}  // namespace hier_output_dist
}  // namespace dyn_emb
