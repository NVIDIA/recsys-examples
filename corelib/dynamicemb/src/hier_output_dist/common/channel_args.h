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
// The part of the kernels' arguments that both channels share and that is
// fixed for the op's life (ChannelStatic): its sizes, by which a kernel lays
// out its window and its tables, and the window itself. The op builds it
// once, after its window is registered; the out channel's static part
// extends it (OutStatic), the in channel's comm kernels take it with the
// call's sequence number (CommArgs), and the exit round's kernel takes it
// alone.
#pragma once

#include <nccl_device.h>

#include <cstdint>
#include <type_traits>

#include "hier_output_dist/common/sizes.h"

namespace dyn_emb {
namespace hier_output_dist {

struct ChannelStatic : Sizes {
  ncclWindow_t window;  // the op's window (common/window.h)
  uint64_t timeout_ns;  // how long a wait for a peer may take
  ncclDevComm dev;      // GIN across nodes
};
static_assert(std::is_trivially_copyable<ChannelStatic>::value,
              "a kernel parameter is copied as bytes");

}  // namespace hier_output_dist
}  // namespace dyn_emb
