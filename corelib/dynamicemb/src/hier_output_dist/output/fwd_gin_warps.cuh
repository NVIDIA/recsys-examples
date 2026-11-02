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
#include "hier_output_dist/common/trace_slots.h"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/node_window.cuh"
#include "hier_output_dist/device/rail.cuh"
#include "hier_output_dist/device/spin.cuh"
#include "hier_output_dist/device/warp.cuh"
#include "hier_output_dist/output/args.cuh"
#include "hier_output_dist/output/protocol.cuh"
#include "hier_output_dist/output/timeline.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace fwd {

// The proxy's relay in the forward (across nodes; a warp of CTA 0, whose
// lanes take the owner nodes, and the requesters of the node, in turn: lane l
// those numbered l, l + 32, ...). It grants the rail owners rounds 0 and 1 of
// proxy_rows if the previous out kernel has not (the first forward). Then it
// polls, waiting for no single one:
//   - it publishes the rows each owner node puts into proxy_rows as they
//     land, in landed_rows[owner_node], where the node's requesters read it.
//     They land in the order of that node's node-unique keys, round after
//     round, and add up on its fwd_rows signal, a running total over
//     forwards (fwd_tot: what it held when this forward began; rows may land
//     before the kernel starts, as the grant came at the end of the previous
//     one);
//   - with more than one round (so at least three, rounds_for), it grants
//     round next + 2 into the half of proxy_rows round `next` took, once
//     every requester of the node has pulled its rows of round `next` and of
//     every round before. A requester's done word of a round's parity is a
//     running total of what it pulled of those rounds, so the rounds so far
//     are in once it has grown by its rows of them (push_start) since the
//     forward began (relay_want keeps what it waits for). A requester pulls
//     nothing before the relay has published some rows, so the words read
//     first (before any lane publishes) are what they held when the forward
//     began. And no requester pulls a row of round next + 2 before it is
//     granted, so until then a word grows by rounds up to `next` alone.
// What it keeps of an owner node between polls is in this rank's words: the
// rows it published (landed_rows, tagged with this out_seq) and fwd_tot.
struct ProxyRelay {
  // Every lane of the relay warp.
  __device__ static void run(const OutArgs &args);
};

__device__ void ProxyRelay::run(const OutArgs &args) {
  const int lane = threadIdx.x & 31, num_nodes = args.topo.num_nodes, node = args.topo.node,
            local_size = args.topo.local_size;
  const WindowLayout window = NodeWindow(args.window, args.topo.local_rank).layout(args);
  ncclGin rows(args.dev, kCtxFwd), cred(args.dev, kCtxCredit);
  const ncclTeam rail = ncclTeamRail(args.dev);
  const Credits credits(cred, rail);
  const OutTimeline timeline(args);
  auto grant = [&](int rounds) {  // rounds [0, rounds) to every rail owner of the lane
    for (int owner_node = lane; owner_node < num_nodes; owner_node += 32)
      if (owner_node != node)
        credits.raise(owner_node, sig_grant(node), grant_of(args.ch.seq, rounds),
                      window.ctrl.grant_sent(owner_node));
  };
  grant(2);
  const int num_rounds = (int)*args.plan.num_rounds;
  // a requester's rows of round rnd at this proxy
  auto round_rows = [&](int requester, int rnd) {
    const int32_t *start = args.plan.push_starts(requester);
    return (uint64_t)(start[rnd + 1] - start[rnd]);
  };
  if (num_rounds > 1)
    for (int requester = lane; requester < local_size; requester += 32)
      for (int half = 0; half < 2; ++half)
        *window.ctrl.relay_want(requester, half) =
            ld_acquire_sys(window.ctrl.done(requester, half));
  __syncwarp();
  int next = 0;  // rounds every requester has pulled
  Spin spin(args.timeout_ns);
  for (;;) {
    bool moved = false, landed = true;  // every row of the lane's owner nodes has landed
    // the rows of each owner node that landed, when that changed
    for (int owner_node = lane; owner_node < num_nodes; owner_node += 32) {
      if (owner_node == node) continue;
      const int64_t num_uniq = args.plan.uniq_out[owner_node];
      const uint64_t word = *window.ctrl.landed_rows(owner_node);
      const int64_t seen =
          LandedRows::tag(word) == args.ch.seq ? (int64_t)LandedRows::payload(word) : -1;
      if (seen == num_uniq) continue;
      const int64_t got =
          min((int64_t)(rows.readSignal(sig_rows(owner_node)) - *window.ctrl.fwd_tot(owner_node)),
              num_uniq);
      if (got != seen) {
        LandedRows::publish(window.ctrl.landed_rows(owner_node), args.ch.seq, got);
        moved = true;
        const int64_t counted = max(seen, (int64_t)0);  // what the timeline has
        if (timeline.on() && got > counted)
          timeline.landed((unsigned long long)(got - counted), counted == 0);
      }
      landed = landed && got == num_uniq;
    }
    // the rounds whose halves every requester has pulled
    while (next + 2 < num_rounds) {
      const int half = next & 1;
      bool in = true;
      for (int requester = lane; requester < local_size; requester += 32)
        in = in && ld_acquire_sys(window.ctrl.done(requester, half)) >=
                       *window.ctrl.relay_want(requester, half) + round_rows(requester, next);
      if (!__all_sync(kFull, in)) break;
      for (int requester = lane; requester < local_size; requester += 32)
        *window.ctrl.relay_want(requester, half) += round_rows(requester, next);
      ++next;
      grant(next + 2);
      moved = true;
    }
    if (__all_sync(kFull, landed)) break;
    // the warp polls again at once while any lane moved
    if (!__any_sync(kFull, moved))
      spin.pause("rows of the owner nodes", next, lane, args.topo.rank);
  }
  for (int owner_node = lane; owner_node < num_nodes; owner_node += 32)
    if (owner_node != node)
      *window.ctrl.fwd_tot(owner_node) += (uint64_t)args.plan.uniq_out[owner_node];
  __syncwarp();
  if (lane == 0) timeline.stamp_latest(kTraceLanded);
  credits.flush_warp();  // the grants
}

// The owner's issuer in the forward (across nodes; a warp of CTA 0, whose
// lanes take the rail proxies in turn, so one QP each). For each proxy node
// it puts the rows of lookup_rows the node asks for (the forward put list:
// its segments, round by round, each in one IB chunk and one feature cell),
// a round once the proxy granted it, each segment in one put that adds its
// rows to the proxy's fwd_rows signal. It polls the proxy nodes in turn and
// waits for no single one; its place in each one's segments is in this
// rank's put_at words. Then the warp flushes. The lookup wrote lookup_rows
// before the kernel: one system fence, then puts that make no release of
// their own. No node barrier is needed first.
struct OwnerIssuer {
  // Every lane of the issuer warp.
  __device__ static void run(const OutArgs &args);
};

__device__ void OwnerIssuer::run(const OutArgs &args) {
  const int lane = threadIdx.x & 31, num_nodes = args.topo.num_nodes, node = args.topo.node;
  ncclGin rows(args.dev, kCtxFwd), cred(args.dev, kCtxCredit);
  const ncclTeam rail = ncclTeamRail(args.dev);
  const OutTimeline timeline(args);
  const NodeWindow node_window(args.window, args.topo.local_rank);
  const WindowLayout window = node_window.layout(args);
  // rows of lookup_rows into the rail proxies' proxy_rows
  const RowPut put{rows,
                   rail,
                   node_window,
                   window.rows.proxy_rows(),
                   window.rows.lookup_rows(),
                   args.row_bytes,
                   sig_rows(node)};
  for (int proxy_node = lane; proxy_node < num_nodes; proxy_node += 32) {
    window.ctrl.put_at(proxy_node)[0] = args.plan.fwd_puts(proxy_node)[0];
    window.ctrl.put_at(proxy_node)[1] = 0;
  }
  put.fence();
  bool posted = false;  // this lane put rows
  Spin spin(args.timeout_ns);
  for (;;) {
    bool moved = false, left = false;  // segments of the lane's proxy nodes are left
    for (int proxy_node = lane; proxy_node < num_nodes; proxy_node += 32) {
      if (proxy_node == node) continue;
      const int32_t *start = args.plan.fwd_puts(proxy_node);  // the node's segments, by round
      int64_t *at = window.ctrl.put_at(proxy_node);
      int64_t seg = at[0];
      int rnd = (int)at[1];
      const int64_t end = start[args.rounds_cap];
      for (; seg < end; ++seg) {
        while (seg >= start[rnd + 1]) ++rnd;  // the segment's round
        if (cred.readSignal(sig_grant(proxy_node)) < grant_of(args.ch.seq, rnd + 1)) break;
        const PutSeg segment = args.plan.put_list[seg];
        if (!posted) timeline.stamp_earliest(kTraceFirstPut);
        posted = true;
        put.put(proxy_node, segment.dst, segment.src, segment.rows, false, 0);
        if (timeline.on()) timeline.posted((unsigned long long)segment.rows);
        moved = true;
      }
      at[0] = seg;
      at[1] = rnd;
      left = left || seg < end;
    }
    if (!__any_sync(kFull, left)) break;
    if (!__any_sync(kFull, moved)) spin.pause("grant of proxy_rows", lane, 0, args.topo.rank);
  }
  if (posted) timeline.stamp_latest(kTraceLastPut);
  __syncwarp();
  rows.flush(ncclCoopWarp{});
  // the puts have completed
  if (timeline.on() && __any_sync(kFull, posted) && lane == 0)
    timeline.stamp_latest(kTracePutsDone);
}

}  // namespace fwd

}  // namespace hier_output_dist
}  // namespace dyn_emb
