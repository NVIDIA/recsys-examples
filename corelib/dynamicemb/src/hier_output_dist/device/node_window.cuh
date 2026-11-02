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
// The op's window as the GPUs of a node see it (NodeWindow).
#pragma once

#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/common/sizes.h"
#include "hier_output_dist/common/window.h"

namespace dyn_emb {
namespace hier_output_dist {

// Every GPU of the node maps every node peer's copy of the window (NCCL's LSA
// team, over NVLink). A kernel lays the window out over this rank's copy
// (layout()) and names places in it by plain pointers; peer() is the same
// place in a node peer's copy, and offset() where it is in every copy, which
// GIN puts take.
struct NodeWindow {
  ncclWindow_t window;
  char *base;  // this rank's copy, as its node peers map it
  __device__ __forceinline__ NodeWindow(ncclWindow_t handle, int local_rank)
      : window(handle), base((char *)ncclGetLsaPointer(handle, 0, local_rank)) {}
  __device__ __forceinline__ WindowLayout layout(const Sizes &sizes) const {
    return WindowLayout(sizes, base);
  }
  // Node peer lsa_peer's copy of what `mine` points to in this rank's copy.
  template <typename T>
  __device__ __forceinline__ T *peer(T *mine, int lsa_peer) const {
    return (T *)ncclGetLsaPointer(window, offset(mine), lsa_peer);
  }
  // Where `mine` is in the window, in bytes.
  __device__ __forceinline__ size_t offset(const void *mine) const {
    return (size_t)((const char *)mine - base);
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
