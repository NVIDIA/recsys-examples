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

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/output_dist/args.cuh"
#include "hier_output_dist/output_dist/out_kernels.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// Row copies of the forward under index dedup: a row to each of its
// original rows.

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
// sub-tasks then copy from there). The warp copy keeps kVectors vectors in
// flight per lane (warp_copy_rows).
template <int kVectors>
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
    warp_copy_rows<kVectors>(src, dst, num_rows, vec, lane);
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

// The forward's requester warps: a pull task (fwd::Pull) and the steps of
// their task loop. Every warp but, across nodes, CTA 0's two GIN warps
// requests. Its tasks come from two queues (ReadyFirst):
//   - local: the row tasks of the node's owners, with index dedup's heavy
//     rows' sub-tasks spread evenly among them (each sub-task reads its row
//     where the pull would and writes kHeavy of its copies). Those of rows of
//     the node's owners take these slots: the pulls keep NVLink busy while
//     the sub-tasks write HBM;
//   - remote (across nodes): the tasks of the node's proxies, chunk after
//     chunk of their streams (task_starts): in a chunk each proxy's in its
//     push order, 32 positions each, proxies in turn; then a slot per heavy
//     row sub-task again, which those of rows of other nodes take. Taken
//     before the pulls, they would hold warps waiting for rows to land while
//     the pulls of rows that have landed wait behind them.
// A warp takes the next remote task once its chunk has landed, else a local
// task, so the pulls empty the proxies' rings as their rows land, while the
// local rows fill the time between; the late sub-tasks wait for the local
// tasks. A remote task's rows are pulled once the relay of its proxy says
// its chunk has landed. At a proxy whose stream wraps its ring the task then
// adds its rows to that chunk's pulled_rows word there, which tells the relay
// when the chunk is done with. No task waits for a later one of its queue. On
// one node (kMulti false) only the row tasks and the sub-tasks among them are
// compiled in: the multi-node paths would make the one-node loop about 10%
// slower. The kernel runs the task loop (fwd_kernel); these functions are its
// steps.

namespace fwd {

// The vectors in flight per lane of a pull's warp copy (warp_copy_rows): 8 on
// one node. Across nodes 4, which leaves the task loop the registers of its
// multi-node paths, so it has no spills. NVLink moves rows as fast with 4 as
// with 8 in flight per lane (32 CTAs of 16 warps).
constexpr int kPullVectors = 8, kPullVectorsMulti = 4;

// One pull task of a warp: at most kTaskRows rows, a lane each, from a node
// peer's lookup_rows (a row task) or a proxy's proxy_rows (a remote task),
// into the requester's output.
struct Pull {
  bool remote;    // a remote task: from a proxy's proxy_rows
  int peer;       // the node peer whose window the task reads (a remote task: its proxy)
  int64_t chunk;  // a remote task: its chunk of the proxy's stream
  int num_rows;
  int64_t out_row;  // the lane's deduplicated row, whose original rows its row goes to
  const char *src;  // its row
  __device__ __forceinline__ explicit Pull(int me)
      : remote(false), peer(me), chunk(0), out_row(0), src(nullptr) {}
};

// Across nodes: every row of chunk `chunk` of a proxy's stream has landed in
// its proxy_rows in forward seq, as the proxy's relay says. Every lane of
// the warp.
__device__ __forceinline__ void wait_landed_warp(const Sizes &sizes, int64_t seq,
                                                 const OutTables &tables, int proxy, int64_t chunk,
                                                 int lane) {
  if (lane == 0) {
    Spin spin(sizes.timeout_ns);
    while ((int64_t)ld_acquire_sys(tables.landed(proxy, chunk)) < seq)
      spin.pause("rows landed at a proxy", proxy, chunk, sizes.topo.rank);
  }
  __syncwarp();
}

// Index dedup, heavy row sub-task `sub`: kHeavy of its row's original rows,
// the row read where the pull would read it. Its slot in each of the two
// passes of the sub-tasks (kLate) runs it in one of them: a row of the
// node's owners in the first, among the row tasks; a row of another node's in
// the last, once every remote row's task was taken. There, at a proxy whose
// stream fits its ring the row is read in proxy_rows once its chunk has
// landed (by then, or soon), and proxy_rows keeps it until the kernel ends. A
// proxy whose stream wraps the ring takes other rows there again while the
// kernel runs, so the row's remote task copied it to its first original row,
// and once every row this rank pulls from that proxy is in (ready[rail]) the
// sub-task copies it on from there (staged). Every lane of the warp.
template <bool kMulti, bool kLate>
__device__ void heavy_sub_warp(const Sizes &sizes, int64_t seq, const FwdArgs &args,
                               const OutTables &tables, int64_t sub, int lane) {
  const FwdPlanView &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int local_size = sizes.topo.local_size, row_bytes = sizes.row_bytes;
  const int64_t dedup_row = fanout.sub[2 * sub], first = fanout.exp_off[dedup_row];
  const int64_t begin = fanout.sub[2 * sub + 1],
                end = min(begin + kHeavy, (int64_t)fanout.exp_off[dedup_row + 1]);
  const int32_t row = fanout.pos[fanout.slot[dedup_row]];
  const int owner = tables.owner_of(row), rail = owner % local_size;
  const bool remote = kMulti && owner / local_size != sizes.topo.node,
             staged = remote && plan.wraps(rail);
  if (remote != kLate) return;
  const char *src =
      remote ? tables.proxy_row(rail, plan.src[row]) : tables.lookup_row(rail, plan.src[row]);
  int64_t from = begin;
  if (staged) {
    if (lane == 0) {
      Spin spin(sizes.timeout_ns);
      while ((int64_t)ld_acquire_gpu((const uint64_t *)&args.ready[rail]) <
             plan.push_order_off[rail + 1] - plan.push_order_off[rail])
        spin.pause("rows pulled from a proxy", rail, dedup_row, sizes.topo.rank);
    }
    __syncwarp();
    src = args.out + (int64_t)fanout.exp_idx[first] * row_bytes;
    from = max(begin, first + 1);
  } else if (remote) {
    wait_landed_warp(sizes, seq, tables, rail, plan.chunk_at(rail, plan.src[row]), lane);
  }
  const OutTimeline timeline{args.trace, local_size};
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  fanout_row(src, from, end, sizes.row_vecs(), lane, fanout.exp_idx, args.out, row_bytes);
  if (timeline.on() && !staged && lane == 0) {
    // the row is read once, from a node peer's window or this rank's own
    const bool peer = rail != sizes.topo.local_rank;
    timeline.task(start, peer ? row_bytes : 0, peer ? 0 : row_bytes);
  }
}

// Row task `index` of the node's owners: its rows of the owner's
// lookup_rows. False when the task is past the end of that owner's block.
__device__ __forceinline__ bool local_pull(const FwdArgs &args, const OutTables &tables,
                                           const LocalRows &local_rows, int64_t index, int lane,
                                           Pull &pull) {
  const FwdPlanView &plan = args.plan;
  const char *src = nullptr;
  int64_t out_row = 0;
  const RowTask job = local_rows.task(index);
  if (job.count == 0) return false;
  if (lane < job.count) {
    const int64_t row = job.row0 + lane;
    src = tables.lookup_row(job.owner, plan.src[row]);
    out_row = plan.inv[row];
  }
  pull.src = src;
  pull.out_row = out_row;
  pull.num_rows = job.count;
  pull.peer = job.owner;
  return true;
}

// Remote task `task` (across nodes): of its chunk and of the proxy in turn
// from this rank's own one, its rows of this rank's push order there, once
// the chunk has landed. False when there is nothing to pull. Every lane of
// the warp.
__device__ __forceinline__ bool remote_pull_warp(const Sizes &sizes, int64_t seq,
                                                 const FwdArgs &args, const OutTables &tables,
                                                 int64_t task, int lane, Pull &pull) {
  const FwdPlanView &plan = args.plan;
  const int local_size = sizes.topo.local_size, me = sizes.topo.local_rank;
  const RemoteTask place(plan.task_starts, plan.max_chunks * local_size, task);
  const int64_t chunk = place.entry / local_size;
  const int proxy = (int)((me + place.entry % local_size) % local_size);
  const int64_t pos0 = plan.push_start(proxy, chunk) + place.nth * kTaskRows;
  const int num_rows = (int)min((int64_t)kTaskRows, plan.push_start(proxy, chunk + 1) - pos0);
  if (num_rows <= 0) return false;
  const char *src = nullptr;
  int64_t out_row = 0;
  if (lane < num_rows) {
    const int32_t row = plan.push_order[pos0 + lane];
    src = tables.proxy_row(proxy, plan.src[row]);
    out_row = plan.inv[row];
  }
  wait_landed_warp(sizes, seq, tables, proxy, chunk, lane);
  pull.remote = true;
  pull.src = src;
  pull.out_row = out_row;
  pull.num_rows = num_rows;
  pull.chunk = chunk;
  pull.peer = proxy;
  return true;
}

// Every lane of the warp: the task's rows into the output; a remote task's,
// when its proxy's stream wraps the ring, counted pulled at the proxy and
// (staged: index dedup's heavy rows across nodes) at this rank; and the task
// into the timeline. The warp copy keeps kVectors vectors in flight per lane.
template <int kVectors>
__device__ __forceinline__ void pull_warp(const Sizes &sizes, const FwdArgs &args,
                                          const OutTables &tables, const Pull &pull, bool staged,
                                          int lane) {
  const FwdPlanView &plan = args.plan;
  const int me = sizes.topo.local_rank, row_bytes = sizes.row_bytes, vec = sizes.row_vecs();
  const int num_rows = pull.num_rows, peer = pull.peer;
  const bool remote = pull.remote;
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  warp_copy_fanout<kVectors>(pull.src, pull.out_row, num_rows, vec, lane, plan.fanout, args.out,
                             row_bytes, remote);
  if (remote && plan.wraps(peer)) {
    // The task's rows are out of proxy_rows (the warp's loads, before the
    // release).
    __syncwarp();
    if (lane == 0) {
      red_release_sys(tables.pulled_rows(peer, pull.chunk), (uint64_t)num_rows);
      if (staged) red_release_gpu((uint64_t *)&args.ready[peer], (uint64_t)num_rows);
    }
  }
  if (timeline.on()) {
    const unsigned long long bytes = (unsigned long long)num_rows * row_bytes;
    if (lane == 0) timeline.task(start, peer != me ? bytes : 0, peer != me ? 0 : bytes);
  }
}

}  // namespace fwd

// The forward's GIN warps (across nodes, the last two warps of CTA 0, which
// skip the entry barrier): the proxy's relay and the owner's issuer.

namespace fwd {

// The rows of an owner node's chunks: the end of chunk j among the node's
// keys (its last key's place, plus one), j in [0, chunks).
struct NodeChunks {
  const FwdPlanView &plan;
  int64_t owner_node, first, chunks;
  __device__ NodeChunks(const FwdPlanView &call_plan, int64_t node)
      : plan(call_plan),
        owner_node(node),
        first(call_plan.chunk_offset_per_bucket[node]),
        chunks(call_plan.chunk_offset_per_bucket[node + 1] -
               call_plan.chunk_offset_per_bucket[node]) {}
  __device__ int64_t place(int64_t j) const { return plan.chunk_inv[first + j]; }
  __device__ int64_t keys_end(int64_t j) const {
    const Chunk &chunk = plan.chunks[place(j)];
    return (int64_t)chunk.key_lo + chunk.keys - plan.node_offset_per_bucket[owner_node];
  }
  // The first chunk whose keys end past `keys` (chunks when none does).
  __device__ int64_t first_past(int64_t keys) const {
    int64_t lo = 0, hi = chunks;
    while (lo < hi) {
      const int64_t mid = (lo + hi) / 2;
      if (keys_end(mid) <= keys)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }
};

// The proxy's relay in the forward (across nodes; a warp of CTA 0, whose
// lanes take the owner nodes in turn: lane l those numbered l, l + 32, ...).
// It grants every rail owner the ring's rows of the stream (positions below
// num_proxy_rows) if the previous out kernel has not (the first forward).
// When its stream wraps the ring it first zeroes the pulled_rows word of each
// of its chunks. Then it polls, waiting for no single one:
//   - it publishes the rows each owner node puts into proxy_rows as they
//     land, in landed_rows[owner_node]; each chunk all of whose rows have
//     landed gets this out_seq in its landed word, where the node's
//     requesters read it. An owner node's rows land in the order of its
//     node-unique keys, chunk after chunk, and add up on its fwd_rows signal,
//     a running total over forwards (fwd_tot: what it held when this forward
//     began; rows may land before the kernel starts, as the grant came at
//     the end of the previous one);
//   - when its stream wraps the ring, lane 0 marks chunks done with in
//     order, once the node's requesters pulled all their rows of them
//     (pulled_rows reaches the chunk's pulls), and grants each chunk whose
//     rows that frees to its owner: the rows of the stream below its end.
// What it keeps of an owner node between polls is in this rank's words: the
// rows it published (landed_rows, tagged with this out_seq) and fwd_tot.
struct ProxyRelay {
  // Every lane of the relay warp, in forward seq.
  template <ncclGinType_t kGin>
  __device__ static void run(ncclWindow_t window, const ncclDevComm &dev, const Sizes &sizes,
                             int64_t seq, const FwdArgs &args);
};

template <ncclGinType_t kGin>
__device__ void ProxyRelay::run(ncclWindow_t window, const ncclDevComm &dev, const Sizes &sizes,
                                int64_t seq, const FwdArgs &args) {
  const FwdPlanView &plan = args.plan;
  const int lane = threadIdx.x & 31, num_nodes = sizes.topo.num_nodes, node = sizes.topo.node;
  const int64_t ring = sizes.num_proxy_rows;
  const CtrlLayout ctrl = NodeWindow(window, sizes.topo.local_rank).layout(sizes).ctrl;
  ncclGin rows = gin_context<kGin>(dev, kCtxFwd), cred = gin_context<kGin>(dev, kCtxCredit);
  const ncclTeam rail = ncclTeamRail(dev);
  const Credits credits(cred, rail);
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
  for (int owner_node = lane; owner_node < num_nodes; owner_node += 32)
    if (owner_node != node)
      credits.raise(owner_node, sig_grant(node), tag_of(seq, ring), ctrl.grant_sent(owner_node));
  const int64_t count = *plan.num_chunks;
  const bool wraps = count > 0 && plan.chunks[count - 1].end > ring;
  if (wraps) {
    for (int64_t chunk = lane; chunk < count; chunk += 32) *ctrl.pulled_rows(chunk) = 0;
    __threadfence_system();
  }
  __syncwarp();
  int64_t done = 0;     // lane 0: chunks done with
  int64_t granted = 0;  // lane 0: chunks granted (those that end within the ring need none)
  if (wraps && lane == 0)
    while (granted < count && plan.chunks[granted].end <= ring) ++granted;
  Spin spin(sizes.timeout_ns);
  for (;;) {
    bool moved = false, landed = true;  // every row of the lane's owner nodes has landed
    // the rows of each owner node that landed, when that changed, and the
    // chunks that thereby landed
    for (int owner_node = lane; owner_node < num_nodes; owner_node += 32) {
      if (owner_node == node) continue;
      const int64_t num_uniq = plan.node_length_per_bucket[owner_node];
      const uint64_t word = *ctrl.landed_rows(owner_node);
      const int64_t seen = LandedRows::tag(word) == seq ? (int64_t)LandedRows::payload(word) : -1;
      if (seen == num_uniq) continue;
      const int64_t got = min(
          (int64_t)(rows.readSignal(sig_rows(owner_node)) - *ctrl.fwd_tot(owner_node)), num_uniq);
      if (got != seen) {
        const NodeChunks chunks(plan, owner_node);
        for (int64_t j = chunks.first_past(max(seen, (int64_t)0)); j < chunks.chunks; ++j) {
          if (chunks.keys_end(j) > got) break;
          st_release_sys(ctrl.landed(chunks.place(j)), (uint64_t)seq);
        }
        LandedRows::publish(ctrl.landed_rows(owner_node), seq, got);
        moved = true;
        const int64_t counted = max(seen, (int64_t)0);  // what the timeline has
        if (timeline.on() && got > counted)
          timeline.landed((unsigned long long)(got - counted), counted == 0);
      }
      landed = landed && got == num_uniq;
    }
    // the chunks done with, and the grants of the rows they free
    if (wraps && lane == 0) {
      while (done < count &&
             (int64_t)ld_acquire_sys(ctrl.pulled_rows(done)) >= plan.chunks[done].pulls) {
        ++done;
        const int64_t free_end = (done < count ? (int64_t)plan.chunks[done].start
                                               : (int64_t)plan.chunks[count - 1].end) +
                                 ring;
        for (; granted < count && (int64_t)plan.chunks[granted].end <= free_end; ++granted) {
          const Chunk &chunk = plan.chunks[granted];
          credits.raise(chunk.owner_node, sig_grant(node), tag_of(seq, chunk.end),
                        ctrl.grant_sent(chunk.owner_node));
        }
        moved = true;
      }
    }
    if (__all_sync(kFull, landed)) break;
    // the warp polls again at once while any lane moved
    if (!__any_sync(kFull, moved))
      spin.pause("rows of the owner nodes", done, lane, sizes.topo.rank);
  }
  for (int owner_node = lane; owner_node < num_nodes; owner_node += 32)
    if (owner_node != node)
      *ctrl.fwd_tot(owner_node) += (uint64_t)plan.node_length_per_bucket[owner_node];
  __syncwarp();
  if (lane == 0) timeline.stamp_latest(kTraceLanded);
  credits.flush_warp();  // the grants
}

// The owner's issuer in the forward (across nodes; a warp of CTA 0, whose
// lanes take the rail proxies in turn, so one QP each). For each proxy node
// it puts the rows of lookup_rows the node asks for (the forward put list:
// its segments in stream order, each in one chunk and one feature),
// each once the proxy granted the rows of its stream up to the end of the
// segment's chunk, in one put that adds its rows to the proxy's fwd_rows
// signal. It polls the proxy nodes in turn and waits for no single one; its
// place in each one's segments is in this rank's put_at words. Then the warp
// flushes. The lookup wrote lookup_rows before the kernel: one system fence,
// then puts that make no release of their own. No node barrier is needed
// first.
struct OwnerIssuer {
  // Every lane of the issuer warp, in forward seq.
  template <ncclGinType_t kGin>
  __device__ static void run(ncclWindow_t window, const ncclDevComm &dev, const Sizes &sizes,
                             int64_t seq, const FwdArgs &args);
};

template <ncclGinType_t kGin>
__device__ void OwnerIssuer::run(ncclWindow_t window, const ncclDevComm &dev, const Sizes &sizes,
                                 int64_t seq, const FwdArgs &args) {
  const FwdPlanView &plan = args.plan;
  const int lane = threadIdx.x & 31, num_nodes = sizes.topo.num_nodes, node = sizes.topo.node;
  ncclGin rows = gin_context<kGin>(dev, kCtxFwd, NCCL_GIN_RESOURCE_SHARING_CTA);
  ncclGin cred = gin_context<kGin>(dev, kCtxCredit);
  const ncclTeam rail = ncclTeamRail(dev);
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
  const NodeWindow node_window(window, sizes.topo.local_rank);
  const WindowLayout layout = node_window.layout(sizes);
  // rows of lookup_rows into the rail proxies' proxy_rows
  const RowPut put{rows,
                   rail,
                   node_window,
                   layout.rows.proxy_rows(),
                   layout.rows.lookup_rows(),
                   sizes.row_bytes,
                   sig_rows(node)};
  for (int proxy_node = lane; proxy_node < num_nodes; proxy_node += 32)
    *layout.ctrl.put_at(proxy_node) = plan.fwd_puts(proxy_node);
  put.fence();
  bool posted = false;  // this lane put rows
  Spin spin(sizes.timeout_ns);
  for (;;) {
    bool moved = false, left = false;  // segments of the lane's proxy nodes are left
    for (int proxy_node = lane; proxy_node < num_nodes; proxy_node += 32) {
      if (proxy_node == node) continue;
      int64_t *at = layout.ctrl.put_at(proxy_node);
      int64_t seg = *at;
      const int64_t end = plan.fwd_puts(proxy_node + 1);
      for (; seg < end; ++seg) {
        const PutSeg segment = plan.put_list[seg];
        if (cred.readSignal(sig_grant(proxy_node)) <
            tag_of(seq, plan.fwd_chunks[segment.chunk].end))
          break;
        if (!posted) timeline.stamp_earliest(kTraceFirstPut);
        posted = true;
        put.put(proxy_node, segment.dst, segment.src, segment.rows, false, 0);
        if (timeline.on()) timeline.posted((unsigned long long)segment.rows);
        moved = true;
      }
      *at = seg;
      left = left || seg < end;
    }
    if (!__any_sync(kFull, left)) break;
    if (!__any_sync(kFull, moved)) spin.pause("grant of proxy_rows", lane, 0, sizes.topo.rank);
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
// stream, which only moves rows, by the plans the input dist made.
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
// (out_kernels.cuh):
//   entry  CTA 0 publishes out_seq and every CTA waits until all node peers
//          have: a peer there has ended its lookup (which wrote its
//          lookup_rows before the kernel) and every earlier out kernel;
//   tasks  warps take 32-row tasks from two queues (task words): the node's
//          owners' rows, and across nodes the rows that cross nodes, each
//          remote task when it would not wait (ReadyFirst); owners (and
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
// A proxy's rows of a call are a stream of chunks through its ring,
// proxy_rows (common.h). Every agent takes the chunks in stream order: a
// requester takes its tasks chunk after chunk, at every proxy, and the owners
// put and the reducers sum them in that order, so no task waits behind one
// that needs a later chunk. When a stream wraps the ring, in the forward the
// requesters count the rows they pulled of each chunk at the proxy, whose
// relay grants the rail owners the rows a chunk frees once every requester of
// the node has pulled it; in the backward the proxy's issuer frees a chunk's
// rows once the NIC has read the sums it put out of them, and the requesters
// push the chunks that take those rows only then.
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
// (bwd_ready), and the proxy grants each rail owner the ring's rows of the
// next kernel's stream. (At the end of a backward the proxy's issuer sends
// its grants itself, bwd::ProxyIssuer.)
template <ncclGinType_t kGin>
__device__ void next_kernel_credits_warp(ncclWindow_t window, const ncclDevComm &dev,
                                         const Sizes &sizes, int64_t seq) {
  const int lane = threadIdx.x & 31, num_nodes = sizes.topo.num_nodes, node = sizes.topo.node;
  ncclGin cred = gin_context<kGin>(dev, kCtxCredit);
  const CtrlLayout ctrl = NodeWindow(window, sizes.topo.local_rank).layout(sizes).ctrl;
  const ncclTeam rail = ncclTeamRail(dev);
  const Credits credits(cred, rail);
  for (int peer_node = lane; peer_node < num_nodes; peer_node += 32) {
    if (peer_node == node) continue;
    credits.raise(peer_node, sig_bwd_ready(num_nodes, node), (uint64_t)seq + 1,
                  ctrl.bwd_ready_sent(peer_node));
    credits.raise(peer_node, sig_grant(node), tag_of(seq + 1, (uint64_t)sizes.num_proxy_rows),
                  ctrl.grant_sent(peer_node));
  }
  __syncwarp();
  cred.flush(ncclCoopWarp{});
}

// The end of forward seq for this rank, in the last CTA to arrive: it sets
// the task queues back to 0, tells the node that this rank has pulled
// everything (the exit barrier, on its pulled word) and waits until every
// node peer has too. After it nobody reads this rank's lookup_rows or
// proxy_rows any more, so the next out kernel or lookup may write them;
// across nodes warp 0 then sends the credits for the next out kernel (not
// compiled in on one node, kGin NCCL_GIN_TYPE_NONE). Every thread of the last
// CTA.
template <ncclGinType_t kGin>
__device__ __forceinline__ void exit(ncclWindow_t window, const ncclDevComm &dev,
                                     const Sizes &sizes, int64_t seq, const FwdArgs &args) {
  const int tid = threadIdx.x;
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
  const NodeWindow node(window, sizes.topo.local_rank);
  const CtrlLayout ctrl = node.layout(sizes).ctrl;
  const NodeBarrier pulled{sizes, node, ctrl.pulled(), seq};
  if (tid == 0) {
    timeline.stamp_latest(kTraceArrived);
    TaskQueue::reset_all(ctrl);
    pulled.publish();
  }
  for (int peer = tid; peer < sizes.topo.local_size; peer += blockDim.x)
    pulled.wait(peer, "exit barrier of the forward");
  __syncthreads();
  if constexpr (kGin != NCCL_GIN_TYPE_NONE)
    if (sizes.topo.num_nodes > 1 && (tid >> 5) == 0)
      next_kernel_credits_warp<kGin>(window, dev, sizes, seq);
  if (tid == 0) timeline.stamp_latest(kTraceExited);
}

}  // namespace fwd

// The forward of the output dist: every requester pulls its rows from the
// lookup output (lookup_rows) of each owner of its node, itself included,
// into its output; across nodes the owners first put the rows of the other
// nodes' node-unique keys into their rail proxies' proxy_rows, from which the
// requesters pull those. Roles: across nodes the last two warps of CTA 0 are
// the proxy's relay and the owner's issuer, which skip the entry barrier;
// every other warp requests (the fwd:: steps above): it zeroes the counters this
// kernel counts from 0 (before this rank publishes its out_seq), passes the
// entry barrier and takes its tasks. Every CTA then arrives, and the last one
// ends the forward for this rank (fwd::exit). It takes the op's window, its
// GIN device communicator and its sizes, its out_seq (seq) and the arrive
// word before it (arrive0), and the call (args). It is built once for each
// GIN backend (kGin, gin_context); NCCL_GIN_TYPE_NONE is the kernel for one
// node (kMulti false).
template <ncclGinType_t kGin>
__global__ void __launch_bounds__(kThreads, 1)
    fwd_kernel(ncclWindow_t window, ncclDevComm dev, Sizes sizes, int64_t seq, int64_t arrive0,
               FwdArgs args) {
  constexpr bool kMulti = kGin != NCCL_GIN_TYPE_NONE;
  const FwdPlanView &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int local_size = sizes.topo.local_size, num_nodes = kMulti ? sizes.topo.num_nodes : 1,
            node = sizes.topo.node, me = sizes.topo.local_rank;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const OutTimeline timeline{args.trace, local_size};
  timeline.begin_cta();
  const int workers = num_nodes > 1 && blockIdx.x == 0 ? kThreads - 64 : kThreads;
  if (tid >= workers) {
    if constexpr (kMulti) {
      if (warp == kWarps - 1)
        fwd::ProxyRelay::run<kGin>(window, dev, sizes, seq, args);
      else
        fwd::OwnerIssuer::run<kGin>(window, dev, sizes, seq, args);
    }
  } else {
    const OutTables tables(window, sizes, args.plan);
    const int64_t heavy = fanout.count[1];
    // index dedup's heavy rows of other nodes at a proxy whose stream wraps
    // the ring copy on from their first original row (fwd::heavy_sub_warp)
    const bool staged = num_nodes > 1 && heavy > 0;
    if (staged && blockIdx.x == 0) {
      // Counters this kernel counts from 0, before this rank publishes its
      // out_seq: everything that counts on them runs after the entry.
      for (int proxy = tid; proxy < local_size; proxy += workers) args.ready[proxy] = 0;
      cta_sync(workers);
    }
    pass_entry_barrier(window, sizes, seq, timeline, workers);
    // this node's owners' rows
    const int64_t *local_offset_per_bucket = plan.offset_per_bucket + node * local_size;
    const LocalRows local_rows{local_offset_per_bucket, local_size, me};
    const int64_t rows = local_rows.count(), local = rows + heavy;
    const int64_t remote = num_nodes > 1 ? plan.task_starts[plan.max_chunks * local_size] : 0;
    const int64_t late = num_nodes > 1 ? heavy : 0;
    const CtrlLayout ctrl = tables.layout().ctrl;
    ReadyFirst tasks{TaskQueue{ctrl.task(kQueueLocal)}, TaskQueue{ctrl.task(kQueueRemote)}, local,
                     remote + late};
    // Lane 0: a remote pull is ready once its chunk has landed at its proxy;
    // the late sub-tasks wait for the local tasks. The warp keeps the last
    // (chunk, proxy) entry it saw landed.
    int64_t landed_entry = -1;
    const auto ready = [&](int64_t next) {
      if (next >= remote) return false;
      const RemoteTask place(plan.task_starts, plan.max_chunks * local_size, next);
      if (place.entry == landed_entry) return true;
      const int proxy = (int)((me + place.entry % local_size) % local_size);
      if ((int64_t)ld_acquire_sys(tables.landed(proxy, place.entry / local_size)) < seq)
        return false;
      landed_entry = place.entry;
      return true;
    };
    for (int64_t task = tasks.next_warp(ready); task >= 0; task = tasks.next_warp(ready)) {
      if (task >= local + remote) {  // the late pass of the heavy rows' sub-tasks
        fwd::heavy_sub_warp<kMulti, true>(sizes, seq, args, tables, task - local - remote, lane);
        continue;
      }
      int64_t index = 0;
      if (task < local && Interleave{rows, heavy}.is_sub(task, index)) {
        fwd::heavy_sub_warp<kMulti, false>(sizes, seq, args, tables, index, lane);
        continue;
      }
      fwd::Pull pull(me);
      if (task < local) {
        if (!fwd::local_pull(args, tables, local_rows, index, lane, pull)) continue;
      } else {
        if (!fwd::remote_pull_warp(sizes, seq, args, tables, task - local, lane, pull)) continue;
      }
      fwd::pull_warp<kMulti ? fwd::kPullVectorsMulti : fwd::kPullVectors>(sizes, args, tables, pull,
                                                                          staged, lane);
    }
  }
  if (!arrive_cta(window, sizes, arrive0, timeline)) return;  // the last CTA goes on
  fwd::exit<kGin>(window, dev, sizes, seq, args);
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
