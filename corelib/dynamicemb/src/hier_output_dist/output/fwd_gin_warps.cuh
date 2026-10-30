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
// The forward's GIN warps (across nodes, the last two warps of CTA 0, which
// skip the entry barrier): the proxy's relay and the owner's issuer.
#pragma once

#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/gin_signals.h"
#include "hier_output_dist/common/meta.h"
#include "hier_output_dist/common/trace_slots.h"
#include "hier_output_dist/common/windows.h"
#include "hier_output_dist/device/ctrl_words.cuh"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/rail.cuh"
#include "hier_output_dist/device/spin.cuh"
#include "hier_output_dist/device/warp.cuh"
#include "hier_output_dist/output/args.cuh"
#include "hier_output_dist/output/protocol.cuh"
#include "hier_output_dist/output/timeline.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace fwd {

// The proxy's relay in the forward (across nodes; a warp of CTA 0, a lane
// per owner node). It grants the rail owners rounds 0 and 1 of proxy_rows if
// the previous out kernel has not (the first forward). Then:
//   - it publishes the rows each owner node puts into proxy_rows as they
//     land, in landed_rows[owner_node], where the node's requesters read it.
//     They land in the order of that node's node-unique keys, round after
//     round, and add up on its fwd_rows signal, a running total over
//     forwards (fwd_tot: what it held when this forward began; rows may land
//     before the kernel starts, as the grant came at the end of the previous
//     one);
//   - with more than one round (so at least three, rounds_for), it grants
//     round next + 2 into the half of proxy_rows round `next` took, once
//     every requester of the node (a lane each) has pulled its rows of round
//     `next` and of every round before. A requester's done word of a round's
//     parity is a running total of what it pulled of those rounds, so the
//     rounds so far are in once it has grown by its rows of them
//     (push_start) since the forward began. A requester pulls nothing before
//     the relay has published some rows, so the words read first (before any
//     lane publishes) are what they held when the forward began. And no
//     requester pulls a row of round next + 2 before it is granted, so until
//     then a word grows by rounds up to `next` alone.
struct ProxyRelay {
  // Every lane of the relay warp.
  __device__ static void run(const OutArgs &args);
};

__device__ void ProxyRelay::run(const OutArgs &args) {
  const int lane = threadIdx.x & 31, num_nodes = args.topo.num_nodes,
            local_size = args.topo.local_size, owner_node = lane;
  const bool live = owner_node < num_nodes && owner_node != args.topo.node;
  const CtrlWords mine = CtrlWords::of(args, args.topo.local_rank);
  ncclGin rows(args.dev, kCtxFwd), cred(args.dev, kCtxCredit);
  const ncclTeam rail = ncclTeamRail(args.dev);
  const Credits credits(cred, rail);
  const OutTimeline timeline(args);
  if (live)
    credits.raise(owner_node, sig_grant(args.topo.node), grant_of(args.ch.seq, 2),
                  mine.grant_sent(owner_node));
  const int64_t num_uniq = live ? args.plan.meta[args.meta_at.uniq_out + owner_node] : 0;
  const uint64_t base = live ? *mine.fwd_tot(owner_node) : 0;  // its fwd_rows signal then
  const int num_rounds = (int)args.plan.meta[args.meta_at.misc + kMiscRounds];
  // requester `lane`'s round starts in its push order here
  const int32_t *start = (const int32_t *)(args.plan.meta + args.meta_at.push_start) +
                         lane * (args.win.rounds_cap + 1);
  // requester `lane`'s done words once the rounds so far are in
  uint64_t want[2] = {0, 0};
  if (num_rounds > 1 && lane < local_size) {
    want[0] = ld_acquire_sys(mine.done(lane, 0));
    want[1] = ld_acquire_sys(mine.done(lane, 1));
  }
  __syncwarp();
  int next = 0;                    // rounds every requester has pulled
  int64_t seen = -1, counted = 0;  // rows seen landed, and those the timeline has (tracing)
  bool landed = !live;             // every row of the owner node has landed
  Spin spin(args.timeout_ns);
  for (;;) {
    bool moved = false;
    // the rows of the lane's owner node that landed, when that changed
    if (!landed) {
      const int64_t got = min((int64_t)(rows.readSignal(sig_rows(owner_node)) - base), num_uniq);
      if (got != seen) {
        seen = got;
        LandedRows::publish(mine.landed_rows(owner_node), args.ch.seq, got);
        moved = true;
        if (timeline.on() && got > counted) {
          timeline.landed((unsigned long long)(got - counted), counted == 0);
          counted = got;
        }
      }
      landed = got == num_uniq;
    }
    // the rounds whose halves every requester has pulled
    while (next + 2 < num_rounds) {
      const int half = next & 1;
      bool in = true;
      uint64_t need = 0;
      if (lane < local_size) {
        need = want[half] + (uint64_t)(start[next + 1] - start[next]);
        in = ld_acquire_sys(mine.done(lane, half)) >= need;
      }
      if (!__all_sync(kFull, in)) break;
      want[half] = need;
      ++next;
      if (live)
        credits.raise(owner_node, sig_grant(args.topo.node), grant_of(args.ch.seq, next + 2),
                      mine.grant_sent(owner_node));
      moved = true;
    }
    if (__all_sync(kFull, landed)) break;
    // the warp polls again at once while any lane moved
    if (!__any_sync(kFull, moved))
      spin.pause("rows of an owner node", owner_node, num_uniq, args.topo.rank);
  }
  if (live) *mine.fwd_tot(owner_node) = base + num_uniq;
  __syncwarp();
  if (lane == 0) timeline.stamp_latest(kTraceLanded);
  credits.flush_warp();  // the grants
}

// The owner's issuer in the forward (across nodes; a warp of CTA 0, a lane per
// rail proxy, so one QP each). Each lane puts the rows of lookup_rows the
// proxy's node asks for (the forward put list: per round, a segment per
// feature), round by round once the proxy granted it, in puts of at most
// chunk_rows rows, each adding its rows to the proxy's fwd_rows signal; then
// the warp flushes. The lookup wrote lookup_rows before the kernel: one
// system fence, then puts that make no release of their own. No node barrier
// is needed first.
class OwnerIssuer {
 public:
  // Every lane of the issuer warp.
  __device__ static void run(const OutArgs &args);

 private:
  // Before the first put of round rnd to the rail proxy on proxy_node: waits
  // until that proxy granted the round.
  __device__ __forceinline__ static void wait_grant(const OutArgs &args, const ncclGin &cred,
                                                    int proxy_node, int rnd) {
    Spin spin(args.timeout_ns);
    const uint64_t want = grant_of(args.ch.seq, rnd + 1);
    while (cred.readSignal(sig_grant(proxy_node)) < want)
      spin.pause("grant of proxy_rows", proxy_node, rnd, args.topo.rank);
  }
};

__device__ void OwnerIssuer::run(const OutArgs &args) {
  const int lane = threadIdx.x & 31, num_nodes = args.topo.num_nodes, node = args.topo.node,
            num_features = args.num_features;
  ncclGin rows(args.dev, kCtxFwd), cred(args.dev, kCtxCredit);
  const OutTimeline timeline(args);
  bool posted = false;  // this lane put rows
  const int proxy_node = lane;
  if (proxy_node < num_nodes && proxy_node != node) {
    const ncclTeam rail = ncclTeamRail(args.dev);
    const int64_t *meta = args.plan.meta;
    const int num_rounds = (int)meta[args.meta_at.slice_in + 2 * proxy_node + 1];
    const PutSeg *fwd_put_list = (const PutSeg *)(meta + args.meta_at.fwd_put_list) +
                                 (int64_t)proxy_node * args.win.rounds_cap * num_features;
    const ncclWindow_t win = args.win.handle[kLookupRows];
    const uint64_t lookup_base = args.win.off[kLookupRows], proxy_base = args.win.off[kProxyRows];
    const int64_t row_bytes = args.row_bytes;
    const RowPut put{rows, rail, win, proxy_base, lookup_base, row_bytes, sig_rows(node)};
    bool first = true;
    put.fence();
    for (int rnd = 0; rnd < num_rounds; ++rnd) {
      bool granted = false;
      for (int feature = 0; feature < num_features; ++feature) {
        const PutSeg seg = fwd_put_list[rnd * num_features + feature];
        for (int offset = 0; offset < seg.rows; offset += args.chunk_rows) {
          if (!granted) {
            wait_grant(args, cred, proxy_node, rnd);
            granted = true;
          }
          const int put_rows = min(args.chunk_rows, seg.rows - offset);
          if (first) timeline.stamp_earliest(kTraceFirstPut);
          first = false;
          put.put(proxy_node, seg.dst + offset, seg.src + offset, put_rows, false, 0);
          if (timeline.on()) timeline.posted((unsigned long long)put_rows);
        }
      }
    }
    if (!first) timeline.stamp_latest(kTraceLastPut);
    posted = !first;
  }
  __syncwarp();
  rows.flush(ncclCoopWarp{});
  // the puts have completed
  if (timeline.on() && __any_sync(kFull, posted) && lane == 0)
    timeline.stamp_latest(kTracePutsDone);
}

}  // namespace fwd

}  // namespace hier_output_dist
}  // namespace dyn_emb
