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
// The output dist: one cooperative kernel per direction on the caller's
// stream, which only moves rows, by the maps the input dist made.
//   fwd_kernel (here)          every requester pulls its rows from the lookup
//                              output (lookup_rows) of each owner of its node,
//                              itself included, into its output; across nodes
//                              the owners first put the rows of the other
//                              nodes' node-unique keys into their rail
//                              proxies' rows (proxy_rows), from which the
//                              requesters pull those;
//   bwd_kernel (backward.cuh)  every requester pushes the gradient of those
//                              rows back: into the owners' lookup_rows, where
//                              the owner's lookup backward reads it
//                              (lookup_grad), or into its proxies' proxy_rows,
//                              which sum each key's and put the sums into the
//                              owners' lookup_rows.
// Both share the out channel (out_seq, its node barrier word) and these steps
// (protocol.cuh, schedule.cuh):
//   entry  CTA 0 publishes out_seq and every CTA waits until all node peers
//          have: a peer there has ended its lookup (which wrote its
//          lookup_rows before the kernel) and every earlier out kernel;
//   tasks  warps take 32-row tasks from one queue (Ctrl::tasks), owners (and
//          proxies) in turn, so a rank's warps reach every peer at once;
//   exit   every CTA arrives once it is done (Ctrl::arrive, a running total,
//          so nothing is reset), and the last one ends the kernel's protocol
//          for the rank.
// Across nodes the last warps of CTA 0 take the GIN roles and skip the entry
// barrier: in the forward the owner's issuer lanes (one per rail proxy, so one
// QP each) and the proxy's relay; in the backward the proxy's issuer lanes.
// Each kernel ends by sending the credits for the next one on context 3 (the
// next kernel takes them only if it is of the kind they serve), then flushes
// that context, so no credit is in flight once the kernel has ended.
// A proxy whose rows of a call do not fit proxy_rows has them in rounds, each
// in a half of proxy_rows (common/formats.h): a requester takes every task of
// a round, at every proxy, before any task of the next round, so no task
// waits behind one that needs a later round. In the forward it counts the
// rows it pulled of each round at the proxy, whose relay grants the rail
// owners the next round of a half once every requester of the node has
// pulled the round before; in the backward the proxy's issuer frees a half
// once the NIC has read the sums it put out of it, and the requesters push
// the next round there only then.
// In a build with DYNAMICEMB_HIER_TRACE=1 each kernel also stamps its
// timeline (TraceSlots): when its CTAs start and pass the entry barrier, its
// first and last row put over GIN, when the rows other nodes send it have
// landed, when this rank's rows have moved, and when the exit barrier lets it
// go.
#pragma once

#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/gin_signals.h"
#include "hier_output_dist/common/trace_slots.h"
#include "hier_output_dist/device/cta.cuh"
#include "hier_output_dist/device/ctrl_words.cuh"
#include "hier_output_dist/device/node_sync.cuh"
#include "hier_output_dist/device/rail.cuh"
#include "hier_output_dist/output/args.cuh"
#include "hier_output_dist/output/fwd_gin_warps.cuh"
#include "hier_output_dist/output/fwd_requester.cuh"
#include "hier_output_dist/output/protocol.cuh"
#include "hier_output_dist/output/schedule.cuh"
#include "hier_output_dist/output/tables.cuh"
#include "hier_output_dist/output/timeline.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace fwd {

// At the end of a forward, the credits for the next out kernel (warp 0 of the
// last CTA, one lane per rail peer), and a flush of their context. The exit
// barrier and the owner lanes' flushes mean no node peer and no NIC reads
// this rank's lookup_rows or proxy_rows any more. So the owner lets each rail
// proxy put the next kernel's gradient rows (bwd_ready), and the proxy grants
// each rail owner rounds 0 and 1 of proxy_rows. (At the end of a backward the
// proxy's issuer sends its grants itself, bwd::ProxyIssuer.)
__device__ void next_kernel_credits_warp(const OutArgs &args) {
  const int lane = threadIdx.x & 31, num_nodes = args.topo.num_nodes, node = args.topo.node,
            peer_node = lane;
  ncclGin cred(args.dev, kCtxCredit);
  if (peer_node < num_nodes && peer_node != node) {
    const CtrlWords mine = CtrlWords::of(args, args.topo.local_rank);
    const ncclTeam rail = ncclTeamRail(args.dev);
    const Credits credits(cred, rail);
    credits.raise(peer_node, sig_bwd_ready(num_nodes, node), (uint64_t)args.ch.seq + 1,
                  mine.bwd_ready_sent(peer_node));
    credits.raise(peer_node, sig_grant(node), grant_of(args.ch.seq + 1, 2),
                  mine.grant_sent(peer_node));
  }
  __syncwarp();
  cred.flush(ncclCoopWarp{});
}

// The end of a forward for this rank, in the last CTA to arrive: it sets the
// task queues back to 0, tells the node that this rank has pulled everything
// (the exit barrier, on Ctrl::pulled) and waits until every node peer has
// too. After it nobody reads this rank's lookup_rows or proxy_rows any more,
// so the next out kernel or lookup may write them; across nodes warp 0 then
// sends the credits for the next out kernel (not compiled in on one node,
// kMulti false). Every thread of the last CTA.
template <bool kMulti>
__device__ __forceinline__ void exit(const OutArgs &args) {
  const int tid = threadIdx.x;
  const OutTimeline timeline(args);
  const NodeBarrier<OutArgs> pulled(args, args.ctrl_at.pulled);
  const CtrlWords mine = CtrlWords::of(args, args.topo.local_rank);
  if (tid == 0) {
    timeline.stamp_latest(kTraceArrived);
    TaskQueue::reset_all(args);
    pulled.publish(mine);
  }
  if (tid < args.topo.local_size) pulled.wait(tid, "exit barrier of the forward");
  __syncthreads();
  if (kMulti && args.topo.num_nodes > 1 && (tid >> 5) == 0) next_kernel_credits_warp(args);
  if (tid == 0) timeline.stamp_latest(kTraceExited);
}

}  // namespace fwd

// The forward of the output dist: every requester pulls its rows from the
// lookup output (lookup_rows) of each owner of its node, itself included,
// into its output; across nodes the owners first put the rows of the other
// nodes' node-unique keys into their rail proxies' proxy_rows, from which the
// requesters pull those. Roles: across nodes the last two warps of CTA 0 are
// the proxy's relay and the owner's issuer, which skip the entry barrier;
// every other warp requests (fwd_requester.cuh): it copies the call's tables,
// zeroes the counters this kernel counts from 0 (before this rank publishes
// its out_seq), passes the entry barrier and takes its tasks. Every CTA then
// arrives, and the last one ends the forward for this rank (fwd::exit).
template <bool kMulti>
__global__ void __launch_bounds__(kThreads, 1) fwd_kernel(OutArgs args) {
  __shared__ OutTables tables;
  const OutPlan &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int local_size = args.topo.local_size, num_nodes = kMulti ? args.topo.num_nodes : 1,
            node = args.topo.node, me = args.topo.local_rank;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const OutTimeline timeline(args);
  timeline.begin_cta();
  const int workers = num_nodes > 1 && blockIdx.x == 0 ? kThreads - 64 : kThreads;
  if (tid >= workers) {
    if (warp == kWarps - 1)
      fwd::ProxyRelay::run(args);
    else
      fwd::OwnerIssuer::run(args);
  } else {
    tables.load(args);
    const int64_t heavy = fanout.exp_off != nullptr ? fanout.count[1] : 0;
    if (num_nodes > 1 && heavy > 0 && blockIdx.x == 0) {
      // Counters this kernel counts from 0, before this rank publishes its
      // out_seq: everything that counts on them runs after the entry.
      if (tid < local_size) plan.ready[tid] = 0;
      cta_sync(workers);
    }
    pass_entry_barrier(args, workers);
    const int64_t *node_block_start = tables.block_start + node * local_size;  // this node's owners
    const int64_t *fwd_task_starts = plan.meta + args.meta_at.fwd_task_starts;
    const LocalRows local_rows{node_block_start, local_size, me};
    const int64_t rows = local_rows.count(), local = rows + heavy;
    const int max_rounds = num_nodes > 1 ? tables.max_rounds : 1;
    const int64_t remote = num_nodes > 1 ? fwd_task_starts[max_rounds] : 0;
    const int64_t total = local + remote + (num_nodes > 1 ? heavy : 0);
    // this warp's round
    FwdCursor cursor(num_nodes > 1 ? fwd_task_starts : nullptr, local, max_rounds);
    const TaskQueue queue(args, kQueueWork);
    for (int64_t task = queue.next_warp(); task < total; task = queue.next_warp()) {
      if (task >= local + remote) {  // the late pass of the heavy rows' sub-tasks
        fwd::heavy_sub_warp<kMulti>(args, tables, task - local - remote, true, lane);
        continue;
      }
      cursor.seek(task);
      int64_t index = 0;
      if (task < cursor.mid && Interleave{rows, heavy}.is_sub(cursor.local_index(task), index)) {
        fwd::heavy_sub_warp<kMulti>(args, tables, index, false, lane);
        continue;
      }
      fwd::Pull pull(me);
      if (task < cursor.mid) {
        if (!fwd::local_pull(args, tables, local_rows, index, lane, pull)) continue;
      } else {
        if (!fwd::remote_pull_warp(args, tables, cursor, task, lane, pull)) continue;
      }
      fwd::pull_warp(args, tables, pull, lane);
    }
  }
  if (!arrive_cta(args)) return;  // the last CTA goes on
  fwd::exit<kMulti>(args);
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
