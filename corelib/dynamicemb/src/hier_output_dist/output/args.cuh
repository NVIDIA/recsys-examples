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
// The argument of the out kernels (OutArgs): what the op fixed when it was
// made (OutStatic, which extends the part both channels share,
// ChannelStatic) and what each kernel adds (OutCall), with the plan of its
// call (OutPlan).
#pragma once

#include <cstdint>
#include <type_traits>

#include "hier_output_dist/common/channel_args.h"
#include "hier_output_dist/common/ctrl.h"
#include "hier_output_dist/common/formats.h"

namespace dyn_emb {
namespace hier_output_dist {

// One output dist kernel of a call. The requester's rows are its bucketized
// rows in [0, block_start[world_size]) (its block for an owner:
// [block_start[owner], block_start[owner + 1])). The forward reads a row at
// src[row], the backward writes its gradient at dst[row]: for a same-node
// owner both are its row of the owner's lookup_rows; for a remote owner, rows
// of the proxy's proxy_rows (the node-unique key's row, and the row's own
// slot). inv[row] is its row of the requester's output, or under index dedup
// its deduplicated row, whose original rows the fan-out lists. The tables of
// the call's Meta (meta) and, across nodes, the requester's push order
// (push_order), heavy groups and their sub-tasks, and the proxy's reducer and
// put lists say the rest.
struct OutPlan {
  const int64_t *block_start;  // [world_size + 1]
  const int32_t *rows;         // [sent] forward: src; backward: dst
  const int32_t *inv;          // [sent]
  Fanout fanout;               // fanout.exp_off: index dedup
  const int64_t *meta;         // the call's small tables (Meta)
  // across nodes
  const int32_t *push_order;       // [sent] remote rows by proxy and push order
  const HeavyGroup *heavy_groups;  // backward: heavy groups, positions in push_order
  const int32_t *group_first_sub;  // backward: each heavy group's first sub-task
  const int32_t *group_of_sub;     // backward: the heavy group of each sub-task
  const int32_t *reducer_start;    // backward, proxy: the reducer list (plan/proxy.h)
  const int64_t *reducer_entries;
  const PutSeg *bwd_put_list;  // backward, proxy: the put list (plan/put_list.h)
  int64_t *ready;              // backward, proxy: [chunks] keys of each IB chunk
                               // reduced; forward, requester: [local_size] rows it
                               // pulled from each proxy (index dedup's heavy rows
                               // wait for them); zeroed when the kernel starts
  // rows
  char *out;             // forward: [num_out, row] the requester's output
  const char *grad;      // backward: [num_out, row] its gradient
  float *partial;        // backward: [sub-tasks, dim] fp32 sums of the heavy
                         // rows' sub-tasks (index dedup), then of the heavy
                         // groups' (kStatusHeavySubs, then kStatusGroupSubs)
  uint64_t *heavy_done;  // backward: [heavy rows] sub-tasks of each one done
  uint64_t *group_done;  // backward: [heavy groups] the same (both zeroed when
                         // the kernel starts)
  int64_t heavy_subs;    // backward: the heavy rows' sub-tasks
  int64_t group_subs;    // backward: the heavy groups' sub-tasks
  float scale;           // backward: 1 / world_size under TorchRec's gradient
                         // division, else 1
};

// What every out kernel of the op takes, fixed when the op is made.
struct OutStatic : ChannelStatic {
  int row_bytes, vec;  // a row of lookup_rows, in bytes and in 16-byte vectors
};

// What one out kernel adds: its sequence number, its CTAs' roles and
// arrivals, its timeline, and its call's plan.
struct OutCall {
  Channel ch;                 // the out channel, at this kernel's out_seq
  int reducers;               // backward across nodes: the last CTAs, which reduce
  int64_t arrive0;            // Ctrl::arrive before this kernel
  unsigned long long *trace;  // [kTraceWords] this kernel's timeline, or nullptr
  OutPlan plan;
};

// The out kernels' parameter, by value: OutChannel builds it per kernel from
// its OutStatic and the kernel's OutCall.
struct OutArgs : OutStatic, OutCall {};
static_assert(std::is_trivially_copyable<OutArgs>::value, "a kernel parameter is copied as bytes");
static_assert(sizeof(OutArgs) <= 4096, "the out kernels' parameter fits the parameter space");

}  // namespace hier_output_dist
}  // namespace dyn_emb
