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
// The output dist's forward kernel: its row copies, its requester's
// pulls, its GIN warps (the proxy's relay and the owner's issuer), and the
// kernel.
#pragma once

#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/args.cuh"
#include "hier_output_dist/common.h"
#include "hier_output_dist/device.cuh"
#include "hier_output_dist/out_kernels.cuh"
#include "hier_output_dist/window.h"

namespace dyn_emb {
namespace hier_output_dist {

// Row copies of the forward under index dedup and for heavy keys: a row to
// each of its original rows, and rows of one source read once.

// Copies of the row at `src` into original rows exp_idx[begin .. end) of
// `out`. The row is read once: lanes hold its 16-byte vectors, and a short
// row is written to several copies at a time.
__device__ __forceinline__ void fanout_row(const char *src, int64_t begin, int64_t end, int vec,
                                           int lane, const int32_t *exp_idx, char *out,
                                           int row_bytes) {
  if (vec <= 32) {
    const int per = 32 / vec, col = lane % vec, slot = lane / vec;
    if (slot >= per) return;
    const int4 val = __ldcg((const int4 *)src + col);
    for (int64_t j = begin + slot; j < end; j += per)
      *((int4 *)(out + (int64_t)exp_idx[j] * row_bytes) + col) = val;
    return;
  }
  for (int col = lane; col < vec; col += 32) {
    const int4 val = __ldcg((const int4 *)src + col);
    for (int64_t j = begin; j < end; ++j)
      *((int4 *)(out + (int64_t)exp_idx[j] * row_bytes) + col) = val;
  }
}

// Index dedup: each lane's row (at `src`, its deduplicated row `dedup_row`)
// goes to every original row of that deduplicated row. All rows go to their
// first original row in one warp copy; the few that repeat are then copied on
// from there (local memory, whatever the source was). Heavy rows are left to
// their sub-tasks, but for their first original row when `heavy_first` (their
// sub-tasks then copy from there).
__device__ __forceinline__ void warp_copy_fanout(const char *src, int64_t dedup_row, int num_rows,
                                                 int vec, int lane, const Fanout &fanout, char *out,
                                                 int row_bytes, bool heavy_first) {
  int64_t first_entry = 0, cnt = 0;
  char *dst = nullptr;
  if (lane < num_rows) {
    first_entry = fanout.exp_off[dedup_row];
    cnt = fanout.exp_off[dedup_row + 1] - first_entry;
    if (cnt > 0 && (cnt <= kHeavy || heavy_first))
      dst = out + (int64_t)fanout.exp_idx[first_entry] * row_bytes;
  }
  if (__ballot_sync(kFull, lane < num_rows && dst == nullptr) == 0) {
    warp_copy_rows(src, dst, num_rows, vec, lane);
  } else {
    const int total = num_rows * vec;
    for (int base = 0; base < total; base += 32) {
      const int idx = base + lane;
      const int row = idx / vec;
      const char *src_row = from_lane(src, row & 31);
      char *dst_row = from_lane(dst, row & 31);
      if (idx < total && dst_row != nullptr)
        *((int4 *)dst_row + (idx - row * vec)) = __ldcg((const int4 *)src_row + (idx - row * vec));
    }
  }
  unsigned many = __ballot_sync(kFull, cnt > 1 && cnt <= kHeavy);
  if (!many) return;
  __syncwarp();
  while (many) {
    const int src_lane = __ffs(many) - 1;
    many &= many - 1;
    const int64_t lane_begin = __shfl_sync(kFull, first_entry, src_lane);
    const int64_t lane_count = __shfl_sync(kFull, cnt, src_lane);
    const char *first = from_lane(dst, src_lane);
    fanout_row(first, lane_begin + 1, lane_begin + lane_count, vec, lane, fanout.exp_idx, out,
               row_bytes);
  }
}

// Lanes whose rows have the same source (the occurrences of a heavy key, all
// at its one row in proxy_rows) read it once: its vectors in the lanes, stored
// to each of their destinations.
__device__ __forceinline__ void warp_copy_shared(const char *src, char *dst, int num_rows, int vec,
                                                 int lane) {
  for (unsigned todo = __ballot_sync(kFull, lane < num_rows); todo;) {
    const char *src_row = from_lane(src, __ffs(todo) - 1);
    const unsigned same = __ballot_sync(kFull, lane < num_rows && src == src_row);
    todo &= ~same;
    for (int col_base = 0; col_base < vec; col_base += 32) {
      const int col = col_base + lane;
      const int4 val = col < vec ? __ldcg((const int4 *)src_row + col) : make_int4(0, 0, 0, 0);
      for (unsigned mask = same; mask; mask &= mask - 1) {
        char *dst_row = from_lane(dst, __ffs(mask) - 1);
        if (col < vec) *((int4 *)dst_row + col) = val;
      }
    }
  }
}

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
      while (ld_acquire_sys(tables.landed_rows(proxy, owner_node)) < want)
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
  const int owner = tables.owner_of(row), rail = owner % local_size,
            owner_node = owner / local_size;
  const bool remote = kMulti && owner_node != args.topo.node,
             staged = remote && tables.rounds(rail) > 1;
  if (remote != late) return;
  const char *src =
      remote ? tables.proxy_row(rail, plan.rows[row]) : tables.lookup_row(rail, plan.rows[row]);
  int64_t from = begin;
  if (staged) {
    if (lane == 0) {
      Spin spin(args.timeout_ns);
      while ((int64_t)ld_acquire_gpu((const uint64_t *)&plan.ready[rail]) <
             plan.push_order_off[rail + 1] - plan.push_order_off[rail])
        spin.pause("rows pulled from a proxy", rail, dedup_row, args.topo.rank);
    }
    __syncwarp();
    src = plan.out + (int64_t)fanout.exp_idx[first] * row_bytes;
    from = max(begin, first + 1);
  } else if (remote) {
    wait_landed_warp(args, tables, rail, lane == 0, owner_node,
                     tables.piece_of(rail, 0, owner_node), plan.rows[row], lane);
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
  const char *src = nullptr;
  int64_t out_row = 0;
  const RowTask job = local_rows.task(index);
  if (job.count == 0) return false;
  if (lane < job.count) {
    const int64_t row = job.row0 + lane;
    src = tables.lookup_row(job.owner, plan.rows[row]);
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
  const int local_size = args.topo.local_size, me = args.topo.local_rank;
  const char *src = nullptr;
  int64_t out_row = 0;
  bool shared = false;
  const int rnd = cursor.rnd;
  const int64_t slot = task - cursor.mid;
  const int proxy = (int)((me + slot) % local_size);
  const int64_t nth = slot / local_size;
  if (rnd >= tables.rounds(proxy)) return false;
  ReqEntry entry, next;
  tables.req_of(proxy, rnd, entry, next);
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
    owner_node = tables.owner_of(row) / local_size;
    at = plan.rows[row];
    piece = tables.piece_of(proxy, rnd, owner_node);
    src = tables.proxy_row(proxy, at);
    out_row = plan.inv[row];
  }
  wait_landed_warp(args, tables, proxy, lane < num_rows, owner_node, piece, at, lane);
  pull.src = src;
  pull.out_row = out_row;
  pull.num_rows = num_rows;
  pull.rnd = rnd;
  pull.peer = proxy;
  pull.shared = shared;
  pull.count = tables.rounds(proxy) > 1;
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
      red_release_sys(tables.done(peer, rnd), (uint64_t)num_rows);
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

// The forward's GIN warps (across nodes, the last two warps of CTA 0, which
// skip the entry barrier): the proxy's relay and the owner's issuer.

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
//   tasks  warps take 32-row tasks from one queue (a task word), owners (and
//          proxies) in turn, so a rank's warps reach every peer at once;
//   exit   every CTA arrives once it is done (the arrive word, a running
//          total, so nothing is reset), and the last one ends the kernel's
//          protocol for the rank.
// Across nodes the last warps of CTA 0 take the GIN roles and skip the entry
// barrier: in the forward the owner's issuer lanes (one per rail proxy, so one
// QP each) and the proxy's relay; in the backward the proxy's issuer lanes.
// Each kernel ends by sending the credits for the next one on context 3 (the
// next kernel takes them only if it is of the kind they serve), then flushes
// that context, so no credit is in flight once the kernel has ended.
// A proxy whose rows of a call do not fit proxy_rows has them in rounds, each
// in a half of proxy_rows (common.h): a requester takes every task of
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

namespace fwd {

// At the end of a forward, the credits for the next out kernel (warp 0 of the
// last CTA, its lanes taking the rail peers in turn), and a flush of their
// context. The exit barrier and the owner lanes' flushes mean no node peer
// and no NIC reads this rank's lookup_rows or proxy_rows any more. So the
// owner lets each rail proxy put the next kernel's gradient rows
// (bwd_ready), and the proxy grants each rail owner rounds 0 and 1 of
// proxy_rows. (At the end of a backward the proxy's issuer sends its grants
// itself, bwd::ProxyIssuer.)
__device__ void next_kernel_credits_warp(const OutArgs &args) {
  const int lane = threadIdx.x & 31, num_nodes = args.topo.num_nodes, node = args.topo.node;
  ncclGin cred(args.dev, kCtxCredit);
  const WindowLayout window = NodeWindow(args.window, args.topo.local_rank).layout(args);
  const ncclTeam rail = ncclTeamRail(args.dev);
  const Credits credits(cred, rail);
  for (int peer_node = lane; peer_node < num_nodes; peer_node += 32) {
    if (peer_node == node) continue;
    credits.raise(peer_node, sig_bwd_ready(num_nodes, node), (uint64_t)args.ch.seq + 1,
                  window.ctrl.bwd_ready_sent(peer_node));
    credits.raise(peer_node, sig_grant(node), grant_of(args.ch.seq + 1, 2),
                  window.ctrl.grant_sent(peer_node));
  }
  __syncwarp();
  cred.flush(ncclCoopWarp{});
}

// The end of a forward for this rank, in the last CTA to arrive: it sets the
// task queues back to 0, tells the node that this rank has pulled everything
// (the exit barrier, on its pulled word) and waits until every node peer has
// too. After it nobody reads this rank's lookup_rows or proxy_rows any more,
// so the next out kernel or lookup may write them; across nodes warp 0 then
// sends the credits for the next out kernel (not compiled in on one node,
// kMulti false). Every thread of the last CTA.
template <bool kMulti>
__device__ __forceinline__ void exit(const OutArgs &args) {
  const int tid = threadIdx.x;
  const OutTimeline timeline(args);
  const NodeWindow node(args.window, args.topo.local_rank);
  const NodeBarrier<OutArgs> pulled(args, node, node.layout(args).ctrl.pulled());
  if (tid == 0) {
    timeline.stamp_latest(kTraceArrived);
    TaskQueue::reset_all(args);
    pulled.publish();
  }
  for (int peer = tid; peer < args.topo.local_size; peer += blockDim.x)
    pulled.wait(peer, "exit barrier of the forward");
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
// every other warp requests (fwd_requester.cuh): it zeroes the counters this
// kernel counts from 0 (before this rank publishes its out_seq), passes the
// entry barrier and takes its tasks. Every CTA then arrives, and the last one
// ends the forward for this rank (fwd::exit).
template <bool kMulti>
__global__ void __launch_bounds__(kThreads, 1) fwd_kernel(OutArgs args) {
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
    const OutTables tables(args);
    const int64_t heavy = fanout.exp_off != nullptr ? fanout.count[1] : 0;
    if (num_nodes > 1 && heavy > 0 && blockIdx.x == 0) {
      // Counters this kernel counts from 0, before this rank publishes its
      // out_seq: everything that counts on them runs after the entry.
      for (int proxy = tid; proxy < local_size; proxy += workers) plan.ready[proxy] = 0;
      cta_sync(workers);
    }
    pass_entry_barrier(args, workers);
    const int64_t *node_block_start = plan.block_start + node * local_size;  // this node's owners
    const LocalRows local_rows{node_block_start, local_size, me};
    const int64_t rows = local_rows.count(), local = rows + heavy;
    const int max_rounds = num_nodes > 1 ? tables.max_rounds : 1;
    const int64_t remote = num_nodes > 1 ? plan.fwd_task_starts[max_rounds] : 0;
    const int64_t total = local + remote + (num_nodes > 1 ? heavy : 0);
    // this warp's round
    FwdCursor cursor(num_nodes > 1 ? plan.fwd_task_starts : nullptr, local, max_rounds);
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
