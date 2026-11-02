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
// The backward's GIN warps (across nodes, CTA 0, which skip the entry
// barrier): the proxy's issuer, and when tracing the warp that watches this
// rank's landings.
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

namespace bwd {

// The bit of a chunk's ready counter that the issuer sets once the chunk's
// puts are posted (only that lane reads it).
constexpr int kChunkPutBit = 62;

// The proxy's issuer in the backward (across nodes; the last warp of CTA 0,
// whose lanes take the owner nodes in turn, so one QP each). As owner, each
// lane first lets the rail proxies on its nodes put this backward's rows
// into its lookup_rows, if the previous kernel (a backward too) could not:
// its lookup_grad was still live. Then, for each owner node once the owner
// there let it (bwd_ready), it puts the IB chunks of that node's keys round
// after round, each as soon as the reducers counted all its keys ready, in
// whatever order they complete. It polls the owner nodes in turn and waits
// for no single one; its place in each one's segments is in this rank's
// put_at words. A chunk takes one system fence, then its segments from
// proxy_rows to their rows of the owner's lookup_rows, each adding its rows
// to the owner's bwd_rows signal and, with more than one round, counting on
// the put counter of its half of proxy_rows once the NIC has read it. With
// more than one round each lane also tells lane 0 (shared memory) the rounds
// its owner nodes have all put and its puts out of each half. Once every
// lane has put round rnd and the counter of its half reaches the puts posted
// to that half, the NIC has read round rnd's sums: lane 0 frees the half for
// round rnd + 2 (half_free; the half's later rounds are not ready before).
// Once the NIC has read proxy_rows (one round: the warp's flush, as in the
// forward; more: both counters reach the puts posted), the lanes grant each
// rail owner rounds 0 and 1 of the next kernel, and the warp flushes the
// credits' context.
struct ProxyIssuer {
  // Every lane of the issuer warp.
  __device__ static void run(const OutArgs &args);
};

__device__ void ProxyIssuer::run(const OutArgs &args) {
  __shared__ int at_round[32];         // each lane's round: it has put the rounds before
  __shared__ unsigned at_half[32][2];  // its puts out of each half
  const OutPlan &plan = args.plan;
  const int lane = threadIdx.x & 31, num_nodes = args.topo.num_nodes, node = args.topo.node;
  const NodeWindow node_window(args.window, args.topo.local_rank);
  const WindowLayout window = node_window.layout(args);
  ncclGin rows(args.dev, kCtxBwd), cred(args.dev, kCtxCredit);
  const ncclTeam rail = ncclTeamRail(args.dev);
  const Credits credits(cred, rail);
  const OutTimeline timeline(args);
  bool live = false;  // the lane has owner nodes
  for (int owner_node = lane; owner_node < num_nodes; owner_node += 32) {
    if (owner_node == node) continue;
    live = true;
    credits.raise(owner_node, sig_bwd_ready(num_nodes, node), (uint64_t)args.ch.seq,
                  window.ctrl.bwd_ready_sent(owner_node));
  }
  Spin spin(args.timeout_ns);
  // The ready counters were zeroed before this rank published its out_seq.
  if (lane == 0)
    while ((int64_t)ld_acquire_gpu(window.ctrl.out_started()) < args.ch.seq)
      spin.pause("entry of this rank", 0, args.ch.seq, args.topo.rank);
  const int num_rounds = (int)*plan.num_rounds;
  for (int owner_node = lane; owner_node < num_nodes; owner_node += 32) {
    window.ctrl.put_at(owner_node)[0] = plan.bwd_puts(owner_node)[0];
    window.ctrl.put_at(owner_node)[1] = 0;
  }
  // the puts this proxy posted to each half in earlier calls
  const uint64_t posted[2] = {*window.ctrl.posted(0), *window.ctrl.posted(1)};
  volatile int *lane_rounds = at_round;
  volatile unsigned *lane_half_puts = &at_half[0][0];
  lane_rounds[lane] = live ? 0 : num_rounds;
  lane_half_puts[2 * lane] = lane_half_puts[2 * lane + 1] = 0;
  __syncwarp();
  // rows of proxy_rows into the rail owners' lookup_rows
  const RowPut put{rows,
                   rail,
                   node_window,
                   window.rows.lookup_rows(),
                   window.rows.proxy_rows(),
                   args.row_bytes,
                   sig_rows(node)};
  unsigned puts[2] = {0, 0};  // this lane's puts out of each half
  int freed = 0;              // lane 0: rounds whose half the requesters may fill again
  bool first = true;          // the lane has put nothing yet
  for (;;) {
    bool moved = false;
    int lane_through = num_rounds;  // rounds every owner node of the lane has put
    for (int owner_node = lane; owner_node < num_nodes; owner_node += 32) {
      if (owner_node == node) continue;
      const int32_t *start = plan.bwd_puts(owner_node);  // the node's segments, by round
      int64_t *at = window.ctrl.put_at(owner_node);
      int64_t done = at[0];  // segments [start of round rnd, done) are put
      int rnd = (int)at[1];
      while (rnd < num_rounds && done == start[rnd + 1]) ++rnd;
      if (rnd < num_rounds &&
          (int64_t)credits.read(sig_bwd_ready(num_nodes, owner_node)) >= args.ch.seq) {
        // each chunk the reducers counted ready; a chunk's segments are
        // consecutive
        const int64_t end = start[rnd + 1];
        for (int64_t seg_start = done; seg_start < end;) {
          const int32_t chunk = plan.put_list[seg_start].chunk;
          int64_t seg_end = seg_start, keys = 0;
          for (; seg_end < end && plan.put_list[seg_end].chunk == chunk; ++seg_end)
            keys += plan.put_list[seg_end].rows;
          uint64_t *ready = (uint64_t *)&plan.ready[chunk];
          const uint64_t word = ld_acquire_sys(ready);
          if (!(word >> kChunkPutBit) && (int64_t)word >= keys) {
            if (first) timeline.stamp_earliest(kTraceFirstPut);
            first = false;
            put.fence();
            for (int64_t i = seg_start; i < seg_end; ++i) {
              const PutSeg seg = plan.put_list[i];
              put.put(owner_node, seg.dst, seg.src, seg.rows, num_rounds != 1, rnd & 1);
            }
            puts[rnd & 1] += (unsigned)(seg_end - seg_start);
            if (timeline.on()) timeline.posted((unsigned long long)keys);
            *ready = word | (1ull << kChunkPutBit);
            moved = true;
          }
          if (seg_start == done && (*ready >> kChunkPutBit)) done = seg_end;
          seg_start = seg_end;
        }
      }
      at[0] = done;
      at[1] = rnd;
      lane_through = min(lane_through, rnd < num_rounds && done == start[rnd + 1] ? rnd + 1 : rnd);
    }
    if (num_rounds > 1) {
      // its puts, then its rounds: lane 0 reads them in the other order
      lane_half_puts[2 * lane] = puts[0];
      lane_half_puts[2 * lane + 1] = puts[1];
      __threadfence_block();
      lane_rounds[lane] = lane_through;
    }
    // Lane 0, with more than one round: the rounds every lane has put
    // (through), and the half of each such round freed once the NIC has read
    // the puts posted to it.
    int through = num_rounds;
    if (lane == 0 && num_rounds > 1) {
      for (int other = 0; other < 32; ++other) through = min(through, lane_rounds[other]);
      __threadfence_block();
      while (freed < through && freed + 2 < num_rounds) {
        const int half = freed & 1;
        uint64_t want = posted[half];
        for (int other = 0; other < 32; ++other) want += lane_half_puts[2 * other + half];
        if (rows.readCounter(half) < want) break;
        st_release_sys(window.ctrl.half_free(), grant_of(args.ch.seq, freed + 2));
        ++freed;
        moved = true;
      }
    }
    if (lane_through == num_rounds && through == num_rounds) break;
    if (!moved) spin.pause("reduced chunks", lane_through, lane, args.topo.rank);
  }
  if (!first) timeline.stamp_latest(kTraceLastPut);
  // The NIC has read every put out of proxy_rows: one round, the warp's flush
  // (its puts count on no counter); more, lane 0 waits until each half's
  // counter reaches the puts posted to it, and keeps those totals.
  if (num_rounds == 1) {
    __syncwarp();
    rows.flush(ncclCoopWarp{});
  } else {
    const uint64_t all[2] = {posted[0] + __reduce_add_sync(kFull, puts[0]),
                             posted[1] + __reduce_add_sync(kFull, puts[1])};
    if (lane == 0) {
      for (int half = 0; half < 2; ++half) {
        *window.ctrl.posted(half) = all[half];
        while (rows.readCounter(half) < all[half])
          spin.pause("puts read from proxy_rows", half, (long long)all[half], args.topo.rank);
      }
    }
  }
  // the puts have completed
  if (timeline.on() && __any_sync(kFull, !first) && lane == 0)
    timeline.stamp_latest(kTracePutsDone);
  __syncwarp();
  for (int owner_node = lane; owner_node < num_nodes; owner_node += 32)
    if (owner_node != node)
      credits.raise(owner_node, sig_grant(node), grant_of(args.ch.seq + 1, 2),
                    window.ctrl.grant_sent(owner_node));
  __syncwarp();
  credits.flush_warp();
}

// The gradient rows the proxies on a lane's nodes (l, l + 32, ...) put into
// this rank's lookup_rows: how many have landed during this backward (each
// node's bwd_rows signal against the running total, bwd_tot), and all of
// them.
__device__ __forceinline__ void lane_landed_rows(const OutArgs &args, const ncclGin &rows, int lane,
                                                 uint64_t &got, uint64_t &need) {
  const WindowLayout window = NodeWindow(args.window, args.topo.local_rank).layout(args);
  got = need = 0;
  for (int peer_node = lane; peer_node < args.topo.num_nodes; peer_node += 32) {
    if (peer_node == args.topo.node) continue;
    const uint64_t rows_of_node = (uint64_t)args.plan.uniq_in[peer_node];
    got +=
        min(rows.readSignal(sig_rows(peer_node)) - *window.ctrl.bwd_tot(peer_node), rows_of_node);
    need += rows_of_node;
  }
}

// Tracing, across nodes: a warp of CTA 0 that takes no tasks bins, as they
// land, the gradient rows the proxies on the other nodes put into this
// rank's lookup_rows (each lane those of its nodes, lane_landed_rows; rows
// that landed before the kernel started count at the first look), until
// every one has landed or every other CTA has arrived. It keeps how many each
// lane saw (TraceSmSlots): CTA 0 is then the last CTA, and its wait for those
// rows at the end bins the rest (rest()). So the timeline holds every
// landing, and the kernel still ends when its last CTA has them all.
class LandingWatcher {
 public:
  // Every lane of the watcher warp.
  __device__ static void run(const OutArgs &args);
  // The backward's last CTA when it is CTA 0 (the watcher stopped before
  // every row had landed), every lane of its warp 0: the rows of the lane's
  // nodes the watcher had not seen, binned into the kernel's row as they land
  // (the CTA's shared copy is folded already), and when the last one landed.
  __device__ static void rest(const OutArgs &args, int lane);
};

__device__ void LandingWatcher::run(const OutArgs &args) {
  const int lane = threadIdx.x & 31;
  const WindowLayout window = NodeWindow(args.window, args.topo.local_rank).layout(args);
  ncclGin rows(args.dev, kCtxBwd);
  const OutTimeline timeline(args);
  const uint64_t others = (uint64_t)(args.arrive0 + gridDim.x - 1);
  uint64_t seen = 0, got = 0, need = 0;
  lane_landed_rows(args, rows, lane, got, need);
  Spin spin(args.timeout_ns);
  for (;;) {
    if (seen < need) {
      lane_landed_rows(args, rows, lane, got, need);
      if (got > seen) {
        timeline.landed(got - seen, seen == 0);
        seen = got;
        if (seen == need) timeline.stamp_latest(kTraceLanded);
      }
    }
    if (__all_sync(kFull, seen == need)) break;
    bool rest = false;
    if (lane == 0) rest = ld_acquire_gpu(window.ctrl.arrive()) >= others;
    if (__shfl_sync(kFull, rest, 0)) break;
    spin.pause("gradient rows of the proxy nodes", lane, (long long)need, args.topo.rank);
  }
  timeline.shared()[kSmLanded + lane] = seen;
}

__device__ void LandingWatcher::rest(const OutArgs &args, int lane) {
  const OutTimeline timeline(args);
  ncclGin rows(args.dev, kCtxBwd);
  uint64_t seen = timeline.shared()[kSmLanded + lane], got = 0, need = 0;
  lane_landed_rows(args, rows, lane, got, need);
  if (seen >= need) return;
  Spin spin(args.timeout_ns);
  while (seen < need) {
    lane_landed_rows(args, rows, lane, got, need);
    if (got == seen) {
      spin.pause("gradient rows of the proxy nodes", lane, (long long)need, args.topo.rank);
      continue;
    }
    timeline.landed_late(got - seen, seen == 0);
    seen = got;
  }
  timeline.stamp_latest(kTraceLanded);
}

}  // namespace bwd

}  // namespace hier_output_dist
}  // namespace dyn_emb
