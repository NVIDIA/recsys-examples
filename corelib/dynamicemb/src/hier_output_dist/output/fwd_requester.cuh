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
// The forward's requester warps: a pull task (fwd::Pull) and the steps of
// their task loop. Every warp but, across nodes, CTA 0's two GIN warps
// requests. Its tasks:
//   - the row tasks of the node's owners, with index dedup's heavy rows'
//     sub-tasks spread evenly among them (each sub-task reads its row where
//     the pull would and writes kHeavy of its copies). Those of rows of the
//     node's owners take these slots: the pulls keep NVLink busy while the
//     sub-tasks write HBM;
//   - across nodes, the tasks of the node's proxies, round after round
//     (fwd_task_starts): in a round each proxy's in its push order, first its
//     rows of keys with a row each, then those of heavy keys (32 positions
//     each; occurrences of one key read its row once), proxies in turn;
//   - then a slot per heavy row sub-task again, which those of rows of other
//     nodes take. Taken first, they would hold warps waiting for rows to land
//     while the pulls of rows that have landed wait behind them.
// A round takes its slice of the row tasks, then its remote tasks
// (FwdCursor). So the requesters pull a round soon after it lands, and the
// relay grants the round after next while the owners still put. A remote row
// is pulled once the relay of its proxy says it has landed. At a proxy of
// more than one round the task then adds its rows to this rank's done word
// there of the round's parity. No task waits for a later one. On one node
// (kMulti false) only the row tasks and the sub-tasks among them are
// compiled in: the multi-node paths would make the one-node loop about 10%
// slower. The kernel runs the task loop (fwd_kernel); these functions are
// its steps.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/rows.cuh"
#include "hier_output_dist/device/spin.cuh"
#include "hier_output_dist/device/warp.cuh"
#include "hier_output_dist/output/args.cuh"
#include "hier_output_dist/output/fwd_copy.cuh"
#include "hier_output_dist/output/protocol.cuh"
#include "hier_output_dist/output/schedule.cuh"
#include "hier_output_dist/output/tables.cuh"
#include "hier_output_dist/output/timeline.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace fwd {

// One pull task of a warp: at most kTaskRows rows, a lane each, from a node
// peer's lookup_rows (a row task) or a proxy's proxy_rows (a remote task),
// into the requester's output.
struct Pull {
  bool count;   // at a proxy of more than one round: count the rows pulled
  bool shared;  // occurrences of heavy keys, a source per key: read once
  int peer;     // the node peer whose window the task reads (a remote task: its proxy)
  int rnd;      // a remote task: the round
  int num_rows;
  int64_t out_row;  // the lane's row of the output (index dedup: its deduplicated row)
  const char *src;  // its row
  __device__ __forceinline__ explicit Pull(int me)
      : count(false), shared(false), peer(me), rnd(0), out_row(0), src(nullptr) {}
};

// Across nodes: the lane's row is a remote row of `proxy`, the row of one of
// the node-unique keys of `owner_node` in that proxy's proxy_rows (src: its
// row there, in `piece` of its round). The warp waits until the proxy's
// relay says every such row of the warp has landed (an owner node's rows land
// in the order of its node-unique keys, round after round). Every lane of
// the warp.
__device__ __forceinline__ void wait_landed_warp(const OutArgs &args, const OutTables &tables,
                                                 int proxy, bool live, int owner_node,
                                                 const RoundPiece &piece, int32_t src, int lane) {
  const unsigned on = __ballot_sync(kFull, live);
  if (live) {
    const unsigned peers = __match_any_sync(on, owner_node);
    const int need = __reduce_max_sync(peers, (unsigned)(src - piece.off + piece.uniq_lo + 1));
    if (lane == __ffs(peers) - 1) {
      const uint64_t want = LandedRows::of(args.ch.seq, need);
      Spin spin(args.timeout_ns);
      while (ld_acquire_sys(&tables.landed_rows[proxy][owner_node]) < want)
        spin.pause("rows landed at a proxy", proxy, need, args.topo.rank);
    }
  }
  __syncwarp();
}

// Index dedup, heavy row sub-task `sub`: kHeavy of its row's original rows,
// the row read where the pull would read it. Its slot in each of the two
// passes of the sub-tasks (`late`) runs it in one of them: a row of the
// node's owners in the first, among the row tasks; a row of another node's in
// the last, once every remote row's task was taken. There, at a proxy of one
// round its row has landed in proxy_rows by then, or soon, and proxy_rows
// keeps it until the kernel ends. A proxy of more than one round fills its
// halves of proxy_rows again while the kernel runs, so the row's remote task
// copied it to its first original row, and once every row this rank pulls
// from that proxy is in (ready[rail]) the sub-task copies it on from there
// (staged). Every lane of the warp.
template <bool kMulti>
__device__ void heavy_sub_warp(const OutArgs &args, const OutTables &tables, int64_t sub, bool late,
                               int lane) {
  const OutPlan &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int local_size = args.topo.local_size, row_bytes = args.row_bytes;
  const int64_t dedup_row = fanout.sub[2 * sub], first = fanout.exp_off[dedup_row];
  const int64_t begin = fanout.sub[2 * sub + 1],
                end = min(begin + kHeavy, (int64_t)fanout.exp_off[dedup_row + 1]);
  const int32_t row = fanout.pos[fanout.slot[dedup_row]];
  const int owner = tables.owner_of(args, row), rail = owner % local_size,
            owner_node = owner / local_size;
  const bool remote = kMulti && owner_node != args.topo.node,
             staged = remote && tables.rounds[rail] > 1;
  if (remote != late) return;
  const char *src = (remote ? tables.proxy_rows[rail] : tables.lookup_rows[rail]) +
                    (int64_t)plan.rows[row] * row_bytes;
  int64_t from = begin;
  if (staged) {
    if (lane == 0) {
      const int64_t *push_order_off = plan.meta + args.meta_at.push_order_off;
      Spin spin(args.timeout_ns);
      while ((int64_t)ld_acquire_gpu((const uint64_t *)&plan.ready[rail]) <
             push_order_off[rail + 1] - push_order_off[rail])
        spin.pause("rows pulled from a proxy", rail, dedup_row, args.topo.rank);
    }
    __syncwarp();
    src = plan.out + (int64_t)fanout.exp_idx[first] * row_bytes;
    from = max(begin, first + 1);
  } else if (remote) {
    wait_landed_warp(args, tables, rail, lane == 0, owner_node,
                     tables.piece_of(args, rail, 0, owner_node), plan.rows[row], lane);
  }
  const OutTimeline timeline(args);
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  fanout_row(src, from, end, args.vec, lane, fanout.exp_idx, plan.out, row_bytes);
  if (timeline.on() && !staged && lane == 0) {
    // the row is read once, from a node peer's window or this rank's own
    const bool peer = rail != args.topo.local_rank;
    timeline.task(start, peer ? row_bytes : 0, peer ? 0 : row_bytes);
  }
}

// Row task `index` of the node's owners: its rows of the owner's
// lookup_rows. False when the task is past the end of that owner's block.
__device__ __forceinline__ bool local_pull(const OutArgs &args, const OutTables &tables,
                                           const LocalRows &local_rows, int64_t index, int lane,
                                           Pull &pull) {
  const OutPlan &plan = args.plan;
  const int row_bytes = args.row_bytes;
  const char *src = nullptr;
  int64_t out_row = 0;
  const RowTask job = local_rows.task(index);
  if (job.count == 0) return false;
  if (lane < job.count) {
    const int64_t row = job.row0 + lane;
    src = tables.lookup_rows[job.owner] + (int64_t)plan.rows[row] * row_bytes;
    out_row = plan.inv[row];
  }
  pull.src = src;
  pull.out_row = out_row;
  pull.num_rows = job.count;
  pull.peer = job.owner;
  return true;
}

// Remote task `task` (across nodes; `cursor` at its round): of the proxy in
// turn from this rank's own one, its rows of this rank's push order there,
// once they have landed. False when there is nothing to pull. Every lane of
// the warp.
__device__ __forceinline__ bool remote_pull_warp(const OutArgs &args, const OutTables &tables,
                                                 const FwdCursor &cursor, int64_t task, int lane,
                                                 Pull &pull) {
  const OutPlan &plan = args.plan;
  const int local_size = args.topo.local_size, me = args.topo.local_rank,
            row_bytes = args.row_bytes;
  const char *src = nullptr;
  int64_t out_row = 0;
  bool shared = false;
  const int rnd = cursor.rnd;
  const int64_t slot = task - cursor.mid;
  const int proxy = (int)((me + slot) % local_size);
  const int64_t nth = slot / local_size;
  if (rnd >= tables.rounds[proxy]) return false;
  ReqEntry entry, next;
  tables.req_of(args, proxy, rnd, entry, next);
  const int64_t plain = ceil_div<kTaskRows>(entry.heavy - entry.start);
  int64_t pos0, end;
  if (nth < plain) {
    pos0 = entry.start + nth * kTaskRows;
    end = entry.heavy;
  } else {
    pos0 = entry.heavy + (nth - plain) * kTaskRows;
    end = next.start;
    shared = true;
  }
  const int num_rows = (int)min((int64_t)kTaskRows, end - pos0);
  if (num_rows <= 0) return false;
  int owner_node = 0;
  int32_t at = 0;
  RoundPiece piece{0, 0, 0, 0};
  if (lane < num_rows) {
    const int32_t row = plan.push_order[pos0 + lane];
    owner_node = tables.owner_of(args, row) / local_size;
    at = plan.rows[row];
    piece = tables.piece_of(args, proxy, rnd, owner_node);
    src = tables.proxy_rows[proxy] + (int64_t)at * row_bytes;
    out_row = plan.inv[row];
  }
  wait_landed_warp(args, tables, proxy, lane < num_rows, owner_node, piece, at, lane);
  pull.src = src;
  pull.out_row = out_row;
  pull.num_rows = num_rows;
  pull.rnd = rnd;
  pull.peer = proxy;
  pull.shared = shared;
  pull.count = tables.rounds[proxy] > 1;
  return true;
}

// Every lane of the warp: the task's rows into the output; at a proxy of
// more than one round, counted pulled; and the task into the timeline.
__device__ __forceinline__ void pull_warp(const OutArgs &args, const OutTables &tables,
                                          const Pull &pull, int lane) {
  const OutPlan &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int me = args.topo.local_rank, row_bytes = args.row_bytes;
  const char *src = pull.src;
  const int64_t out_row = pull.out_row;
  const int num_rows = pull.num_rows, rnd = pull.rnd, peer = pull.peer;
  const bool shared = pull.shared, count = pull.count;
  const OutTimeline timeline(args);
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  if (fanout.exp_off != nullptr) {
    warp_copy_fanout(src, out_row, num_rows, args.vec, lane, fanout, plan.out, row_bytes, count);
  } else {
    char *dst = nullptr;
    if (lane < num_rows) dst = plan.out + out_row * row_bytes;
    if (shared)
      warp_copy_shared(src, dst, num_rows, args.vec, lane);
    else
      warp_copy_rows(src, dst, num_rows, args.vec, lane);
  }
  if (count) {
    // The task's rows are out of proxy_rows (the warp's loads, before the
    // release).
    __syncwarp();
    if (lane == 0) {
      red_release_sys(tables.done[peer] + (rnd & 1), (uint64_t)num_rows);
      if (tables.staged) red_release_gpu((uint64_t *)&plan.ready[peer], (uint64_t)num_rows);
    }
  }
  if (timeline.on()) {
    // rows read: occurrences of one heavy key read its row once
    int read = num_rows;
    if (shared && fanout.exp_off == nullptr) {
      const unsigned same = __match_any_sync(kFull, (unsigned long long)src);
      read = __popc(__ballot_sync(kFull, lane < num_rows && lane == __ffs(same) - 1));
    }
    const unsigned long long bytes = (unsigned long long)read * row_bytes;
    if (lane == 0) timeline.task(start, peer != me ? bytes : 0, peer != me ? 0 : bytes);
  }
}

}  // namespace fwd

}  // namespace hier_output_dist
}  // namespace dyn_emb
