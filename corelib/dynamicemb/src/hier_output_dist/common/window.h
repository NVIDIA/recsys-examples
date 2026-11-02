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
// The op's window: this rank's copy of its symmetric memory, one NCCL window
// laid out the same on every rank (WindowLayout).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hier_output_dist/common/ctrl.h"
#include "hier_output_dist/common/keys.h"
#include "hier_output_dist/common/rows.h"
#include "hier_output_dist/common/sizes.h"

namespace dyn_emb {
namespace hier_output_dist {

// The window: the rows section, the ctrl section, then the keys section, each
// from a multiple of kBufferAlign. The out kernels' sections come first, so
// the places of their buffers take the least to compute. The host lays the
// window out over its copy (the memory it allocated) to size it and to view
// its buffers as tensors; a kernel over the same copy as the node's GPUs map
// it (NodeWindow). A place in a node peer's copy is the same place in this
// rank's: kernels name it by a pointer into their own copy, and GIN puts by
// its offset in the window.
struct WindowLayout {
  RowsLayout rows;
  CtrlLayout ctrl;
  KeysLayout keys;
  __host__ __device__ WindowLayout(const Sizes &sizes, char *base)
      : rows(sizes, base),
        ctrl(sizes, base + rows.bytes()),
        keys(sizes, base + rows.bytes() + buffer_end(ctrl.bytes())) {}
  // The window's bytes.
  __host__ __device__ static uint64_t bytes(const Sizes &sizes) {
    return RowsLayout(sizes, nullptr).bytes() + buffer_end(CtrlLayout(sizes, nullptr).bytes()) +
           KeysLayout(sizes, nullptr).bytes();
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
