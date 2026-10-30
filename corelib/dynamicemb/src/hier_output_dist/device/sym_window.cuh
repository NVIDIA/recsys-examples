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
// The regions of the symmetric windows: a node peer's through its LSA pointer.
// A rail peer's region is reached through GIN by its window handle and
// offset in Win (handle[R], off[R]).
#pragma once

#include <nccl_device.h>

#include "hier_output_dist/common/windows.h"

namespace dyn_emb {
namespace hier_output_dist {

// Region R of the symmetric windows, of T elements. It sits at the same
// offset of every rank's window, so a node peer's region is at its LSA
// pointer (NVLink).
template <Region R, typename T = char>
struct SymRegion {
  const Win &win;
  __device__ __forceinline__ explicit SymRegion(const Win &windows) : win(windows) {}
  // Node peer lsa_peer's region (this rank's own included).
  __device__ __forceinline__ T *peer(int lsa_peer) const {
    return (T *)ncclGetLsaPointer(win.handle[R], win.off[R], lsa_peer);
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
