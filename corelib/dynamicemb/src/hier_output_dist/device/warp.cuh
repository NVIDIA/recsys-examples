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
// Warp-wide helpers: the mask of every lane, and a lane's pointer in every
// lane.
#pragma once

namespace dyn_emb {
namespace hier_output_dist {

constexpr unsigned kFull = 0xffffffffu;

// Lane src_lane's pointer, in every lane of the warp (every lane calls it).
template <typename T>
__device__ __forceinline__ T *from_lane(T *ptr, int src_lane) {
  return (T *)__shfl_sync(kFull, (unsigned long long)ptr, src_lane);
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
