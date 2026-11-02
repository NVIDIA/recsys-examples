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
// The backward of the output dist (bwd_kernel): every requester pushes the
// gradient of its rows back where each one came from. Every row of an
// owner's gradient (lookup_grad, over its lookup_rows) is written exactly
// once, so nothing is zeroed and nothing scattered. A row's gradient is,
// under index dedup, first summed over its original rows (fan-in), in fp32
// and in the fan-out's order, which is fixed (increasing original row); under
// TorchRec's gradient division it is then multiplied by 1 / world_size in
// fp32 and rounded once to the row dtype, which is what TorchRec's
// div_(world_size) at the owner computes.
//   same-node owner: into its lookup_rows at the row's own place;
//   remote owner (across nodes): into the proxy's proxy_rows in the proxy's
//     push order, at the row's slot (its key's row, a duplicate's slot), each
//     32-row push task then setting its flag in the proxy's push_flags,
//     tagged with (out_seq, round); the occurrences of a heavy key (more than
//     kHeavy on the node) go as one fp32 sum per requester (heavy group) into
//     its partial slot, 32 groups per flag.
// The proxy's reducer CTAs (across nodes, the last ones) wait for the flags
// of each key's reducer-list entries, sum the keys of several in fp32 in list
// order and round once into the key's row, and count each IB chunk's keys
// ready; its issuer lanes (the last warp of CTA 0) put each chunk, once
// ready, into the owner's lookup_rows. Only the keys of several rows are
// touched: the NIC reads the others where the requester's NVLink stores left
// them. Every sum has a fixed order, so the gradient is the same bit for bit
// from run to run. At the end the last CTA tells every owner of the node
// that this rank's rows are in and waits until every requester has told it
// the same, and (across nodes) until the gradient rows of every rail proxy
// have landed: its lookup_grad is then complete.
#pragma once

#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/gin_signals.h"
#include "hier_output_dist/common/trace_flag.h"
#include "hier_output_dist/common/trace_slots.h"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/device/cta.cuh"
#include "hier_output_dist/device/node_sync.cuh"
#include "hier_output_dist/device/node_window.cuh"
#include "hier_output_dist/device/spin.cuh"
#include "hier_output_dist/output/args.cuh"
#include "hier_output_dist/output/bwd_gin_warps.cuh"
#include "hier_output_dist/output/bwd_reducer.cuh"
#include "hier_output_dist/output/bwd_requester.cuh"
#include "hier_output_dist/output/protocol.cuh"
#include "hier_output_dist/output/schedule.cuh"
#include "hier_output_dist/output/tables.cuh"
#include "hier_output_dist/output/timeline.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace bwd {

// Across nodes, the end of a backward waits (warp 0 of the last CTA, its
// lanes taking the rail peers in turn) for the gradient rows the proxy on
// each other node puts into this owner's lookup_rows: its bwd_rows signal
// against the running total (bwd_tot, of this rank's control words `ctrl`).
__device__ __forceinline__ void wait_rail_rows_warp(const OutArgs &args, const CtrlLayout &ctrl) {
  const int lane = threadIdx.x & 31;
  // Tracing: CTA 0 is the last CTA when its watcher stopped before every
  // row had landed; another one comes after it saw them all
  // (LandingWatcher).
  if (OutTimeline(args).on() && blockIdx.x == 0) LandingWatcher::rest(args, lane);
  ncclGin rows(args.dev, kCtxBwd);
  for (int peer_node = lane; peer_node < args.topo.num_nodes; peer_node += 32) {
    if (peer_node == args.topo.node) continue;
    const uint64_t want = *ctrl.bwd_tot(peer_node) + (uint64_t)args.plan.uniq_in[peer_node];
    Spin spin(args.timeout_ns);
    while (rows.readSignal(sig_rows(peer_node)) < want)
      spin.pause("gradient rows of a proxy node", peer_node, (long long)want, args.topo.rank);
    *ctrl.bwd_tot(peer_node) = want;
  }
}

// The end of a backward for this rank, in the last CTA to arrive: it sets
// the task queues back to 0, tells every owner of the node that this rank's
// gradient rows are in (its flag pushed[local_rank] there) and waits until
// every requester of the node has told it the same; across nodes warp 0 then
// waits until the gradient rows of every rail proxy have landed. This rank's
// lookup_grad is then complete. Every thread of the last CTA.
__device__ __forceinline__ void exit(const OutArgs &args) {
  const int tid = threadIdx.x;
  const OutTimeline timeline(args);
  const NodeWindow node(args.window, args.topo.local_rank);
  const WindowLayout window = node.layout(args);
  const PeerFlags<OutArgs> pushed(args, node, window.ctrl.pushed());
  if (tid == 0) {
    timeline.stamp_latest(kTraceArrived);
    TaskQueue::reset_all(args);
    for (int peer = 0; peer < args.topo.local_size; ++peer) pushed.tell(peer);
  }
  for (int peer = tid; peer < args.topo.local_size; peer += blockDim.x)
    pushed.wait(peer, "gradient rows pushed");
  if (args.topo.num_nodes > 1 && (tid >> 5) == 0) wait_rail_rows_warp(args, window.ctrl);
  if (timeline.on()) {
    __syncthreads();
    if (tid == 0) timeline.stamp_latest(kTraceExited);
  }
}

}  // namespace bwd

// The backward kernel. Roles: across nodes the last warp of CTA 0 is the
// proxy's issuer and, in a build with kTrace, the warp before it watches
// this rank's landings; both skip the entry barrier. Across nodes the last
// CTAs reduce (bwd::ProxyReducer). Every other warp requests
// (bwd_requester.cuh): it zeroes the counters this kernel counts from 0
// (before this rank publishes its out_seq), passes the entry barrier and
// takes its tasks. Every CTA then arrives, and the last one ends the
// backward for this rank (bwd::exit).
template <typename RowT>
__global__ void __launch_bounds__(kThreads, 1) bwd_kernel(OutArgs args) {
  const OutPlan &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int local_size = args.topo.local_size, num_nodes = args.topo.num_nodes,
            node = args.topo.node, me = args.topo.local_rank;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const OutTimeline timeline(args);
  timeline.begin_cta();
  const int gin = num_nodes == 1 || blockIdx.x != 0 ? 0 : timeline.on() ? 64 : 32;
  const int workers = kThreads - gin;
  if (tid >= workers) {
    if (warp == kWarps - 1)
      bwd::ProxyIssuer::run(args);
    else if constexpr (kTrace)
      bwd::LandingWatcher::run(args);
  } else if (num_nodes > 1 && blockIdx.x >= gridDim.x - args.reducers) {
    pass_entry_barrier(args, kThreads);
    bwd::ProxyReducer<RowT>::run(args, lane);
  } else {
    const OutTables tables(args);
    // half_seen starts at 0; the entry barrier's sync publishes it
    for (int proxy = tid; proxy < local_size; proxy += workers) *tables.half_seen(proxy) = 0;
    int64_t groups = 0;  // this rank's heavy groups, at all its proxies
    if (num_nodes > 1)
      groups = plan.push_entry(local_size - 1, tables.rounds(local_size - 1))->groups;
    if (blockIdx.x == 0) {
      // Counters this kernel counts from 0, before this rank publishes its
      // out_seq: everything that counts on them runs after the entry.
      const int64_t slots = fanout.exp_off != nullptr ? fanout.count[0] : 0;
      for (int64_t i = tid; i < slots; i += workers) plan.heavy_done[i] = 0;
      for (int64_t i = tid; i < groups; i += workers) plan.group_done[i] = 0;
      if (num_nodes > 1)
        for (int64_t i = tid; i < *plan.num_chunks; i += workers) plan.ready[i] = 0;
      cta_sync(workers);
    }
    pass_entry_barrier(args, workers);
    const int64_t *node_block_start = plan.block_start + node * local_size;
    // Every heavy row's sub-task has a slot before the remote tasks (across
    // nodes), which those of rows to other nodes take, and one among the row
    // tasks, which those of the node's rows take.
    const int64_t heavy_subs = fanout.exp_off != nullptr ? fanout.count[1] : 0,
                  early_subs = num_nodes > 1 ? heavy_subs : 0;
    const int64_t subs = early_subs + plan.group_subs,
                  remote = subs + (num_nodes > 1 ? plan.bwd_task_starts[2 * tables.max_rounds] : 0);
    const LocalRows local_rows{node_block_start, local_size, me};
    const int64_t rows = local_rows.count(), total = remote + rows + heavy_subs;
    // The round of this warp's remote tasks (one node: none).
    RoundCursor cursor(num_nodes > 1 ? plan.bwd_task_starts : nullptr, 2);
    const TaskQueue queue(args, kQueueWork);
    for (int64_t task = queue.next_warp(); task < total; task = queue.next_warp()) {
      if (task < early_subs) {
        bwd::heavy_sub_warp<RowT>(args, tables, task, true, lane);
        continue;
      }
      if (task < subs) {
        bwd::group_sub_warp<RowT>(args, task - early_subs, lane);
        continue;
      }
      if (task < remote) {
        const int64_t remote_task = task - subs;
        cursor.seek(remote_task);
        bwd::ProxyPush push;
        if (!bwd::remote_push(args, tables, cursor, remote_task, push)) continue;
        if (push.is_group_task)
          bwd::push_groups_task_warp<RowT>(args, tables, push, lane);
        else
          bwd::push_rows_task_warp<RowT>(args, tables, push, lane);
        continue;
      }
      int64_t index;
      if (Interleave{rows, heavy_subs}.is_sub(task - remote, index)) {
        bwd::heavy_sub_warp<RowT>(args, tables, index, false, lane);
        continue;
      }
      bwd::local_push_warp<RowT>(args, tables, local_rows, index, lane);
    }
  }
  if (!arrive_cta(args)) return;  // the last CTA goes on
  bwd::exit(args);
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
