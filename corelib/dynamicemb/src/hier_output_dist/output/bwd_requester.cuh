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
// The backward's requester warps: the done counters of sub-tasks
// (SubTaskDone), a push task to a proxy (bwd::ProxyPush) and the steps of
// their task loop. Every warp but, across nodes, CTA 0's GIN warps and the
// reducer CTAs requests. It takes its tasks in this order:
//   1. Across nodes, the sub-tasks whose sums go to other nodes: index
//      dedup's heavy rows' sub-tasks (each sums kHeavy original rows into an
//      fp32 partial of its own) and the heavy groups' sub-tasks (kHeavy rows
//      each).
//   2. Round after round (bwd_task_starts), the push tasks of every proxy's
//      heavy groups, then those of every proxy's rows (32 positions of its
//      push order each), proxies in turn in both, so the IB chunks fill in
//      push order. The heavy groups come first: first need numbers a
//      feature's most asked keys about first, and a feature spans few IB
//      chunks, so most chunks hold heavy keys, and none is ready before their
//      partials.
//   3. The row tasks of the node's owners, with the sub-tasks of the node's
//      heavy rows spread evenly among them (the pushes keep NVLink busy while
//      the sub-tasks read HBM). The last sub-task of such a row to end adds
//      them up and writes the row, which its row task leaves out.
// A task that needs sub-tasks waits until their done counters say they
// ended. They come earlier in the queue, so a running warp has them, and they
// wait for nothing. A push into round rnd >= 2 of a proxy waits until the
// proxy freed that half of proxy_rows (half_free), which needs only the
// pushes of round rnd - 2, earlier in the queue. The kernel runs the task
// loop (bwd_kernel); these functions are its steps.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/gin_signals.h"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/rows.cuh"
#include "hier_output_dist/device/spin.cuh"
#include "hier_output_dist/device/warp.cuh"
#include "hier_output_dist/output/args.cuh"
#include "hier_output_dist/output/bwd_sums.cuh"
#include "hier_output_dist/output/schedule.cuh"
#include "hier_output_dist/output/tables.cuh"
#include "hier_output_dist/output/timeline.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The done counter of a heavy row's sub-tasks (index dedup) or of a heavy
// group's: each sub-task counts itself done once its warp wrote its fp32 sum,
// and what adds the sums up waits until all have.
struct SubTaskDone {
  uint64_t *word;
  // One lane: waits until `count` sub-tasks counted themselves done.
  __device__ __forceinline__ void wait_lane(const OutArgs &args, int64_t count,
                                            const char *what) const {
    Spin spin(args.timeout_ns);
    while ((int64_t)ld_acquire_gpu(word) < count)
      spin.pause(what, (long long)*word, count, args.topo.rank);
  }
  // Every lane of the warp, once its sum is written: counts it done.
  __device__ __forceinline__ void count_warp(int lane) const {
    __syncwarp();
    if (lane == 0) red_release_gpu(word, 1);
  }
  // One lane, once its warp's sum is written: counts it done and returns how
  // many had counted before (acquire-release: the last one sees every sum).
  __device__ __forceinline__ uint64_t count_lane() const {
    return atom_add_acq_rel_gpu(word, (uint64_t)1);
  }
};

namespace bwd {

// One push task to a proxy (across nodes): the proxy, the round, the task's
// place among this rank's tasks of its kind there (nth), and this rank's
// entries of that proxy's push table for the round (entry; next holds the
// ends).
struct ProxyPush {
  bool is_group_task;  // a task of heavy groups, else of rows
  int proxy, rnd;
  int64_t nth;
  int64_t plain_tasks;  // this rank's tasks of rows at the proxy in the round
  ReqEntry entry, next;
};

// Before a push into round rnd of the proxy's proxy_rows: rounds 0 and 1 are
// free from the start (the previous out kernel was done with proxy_rows
// before this one's entry), round rnd from when the proxy freed its half
// (half_free), once the NIC had read the sums of round rnd - 2 out of it.
// What one warp of the CTA saw is kept for the others (half_seen), so most
// tasks read no word of the proxy. Every lane of the warp.
__device__ __forceinline__ void wait_half_warp(const OutArgs &args, OutTables &tables, int proxy,
                                               int rnd, int lane) {
  if (rnd < 2) return;
  if (lane == 0) {
    const uint64_t want = grant_of(args.ch.seq, rnd);
    if (ld_acquire_cta(&tables.half_seen[proxy]) < want) {
      Spin spin(args.timeout_ns);
      uint64_t word;
      while ((word = ld_acquire_sys(tables.half_free[proxy])) < want)
        spin.pause("a half of a proxy's rows", proxy, rnd, args.topo.rank);
      red_max_release_cta(&tables.half_seen[proxy], word);
    }
  }
  __syncwarp();
}

// A push task's rows (or partials) of round rnd are in the proxy's
// proxy_rows: its flag `flag` in the round's half of that proxy's flags,
// tagged with (out_seq, rnd). Every lane of the warp.
__device__ __forceinline__ void push_flag_warp(const OutArgs &args, const OutTables &tables,
                                               int proxy, int rnd, int64_t flag, int lane) {
  __syncwarp();
  if (lane == 0)
    st_release_sys(&tables.push_flags[proxy][(rnd & 1) * half_flags(args.win.num_proxy_rows,
                                                                    args.topo.local_size) +
                                             flag],
                   grant_of(args.ch.seq, rnd));
}

// Index dedup, a heavy row (more than kHeavy original rows): once its
// sub-tasks are done, where their sums are (num_subs of them, in sub-task
// order). Every lane of the warp.
template <typename RowT>
__device__ __forceinline__ const float *heavy_row_warp(const OutArgs &args, int64_t dedup_row,
                                                       int64_t cnt, int lane, int64_t &num_subs) {
  const Fanout &fanout = args.plan.fanout;
  const int32_t slot = fanout.slot[dedup_row];
  num_subs = ceil_div<kHeavy>(cnt);
  if (lane == 0)
    SubTaskDone{args.plan.heavy_done + slot}.wait_lane(args, num_subs, "sub-tasks of a heavy row");
  __syncwarp();
  return args.plan.partial + (int64_t)fanout.first[slot] * (args.vec * Vec16<RowT>::kValues);
}

// Index dedup: heavy row sub-task `sub` sums kHeavy of its original rows
// into an fp32 partial of its own, then counts itself done for its row. It
// runs in its row's pass: `remote` for a row an owner on another node gets
// (through a proxy), whose push task adds up the row's sub-tasks; else for a
// row of one of the node's owners, where the last of the row's sub-tasks to
// end adds them up, in sub-task order, and writes the row into the owner's
// lookup_rows. Every lane of the warp.
template <typename RowT>
__device__ void heavy_sub_warp(const OutArgs &args, const OutTables &tables, int64_t sub,
                               bool remote, int lane) {
  constexpr int kValues = Vec16<RowT>::kValues;
  const OutPlan &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int dim = args.vec * kValues, local_size = args.topo.local_size;
  const int64_t dedup_row = fanout.sub[2 * sub], begin = fanout.sub[2 * sub + 1],
                row_end = fanout.exp_off[dedup_row + 1];
  const int32_t slot = fanout.slot[dedup_row], row = fanout.pos[slot];
  const int owner = tables.owner_of(args, row);
  if ((owner / local_size != args.topo.node) != remote) return;
  const OutTimeline timeline(args);
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  const int64_t end = min(begin + kHeavy, row_end);
  float *part = plan.partial + sub * dim;
  for (int col = lane; col < args.vec; col += 32) {
    float acc[kValues];
    sum_vec<RowT>(plan.grad, fanout.exp_idx, begin, end, args.row_bytes, col, acc);
    store_fp32<kValues>(part, col, acc);
  }
  if (remote) {
    SubTaskDone{plan.heavy_done + slot}.count_warp(lane);
    return;
  }
  __syncwarp();
  uint64_t before = 0;
  if (lane == 0) before = SubTaskDone{plan.heavy_done + slot}.count_lane();
  __syncwarp();
  const int64_t num_subs = ceil_div<kHeavy>(row_end - fanout.exp_off[dedup_row]);
  if ((int64_t)__shfl_sync(kFull, before, 0) + 1 < num_subs) return;
  // the last one: every sub-task of the row has ended (acquired above)
  const float *sums = plan.partial + (int64_t)fanout.first[slot] * dim;
  char *dst = tables.lookup_rows[owner % local_size] + (int64_t)plan.rows[row] * args.row_bytes;
  for (int col = lane; col < args.vec; col += 32) {
    float acc[kValues];
    sum_fp32<kValues>(sums, num_subs, dim, col, acc);
    *((int4 *)dst + col) = scaled<RowT>(acc, plan.scale);
  }
  if (timeline.on() && lane == 0) {
    const bool peer = owner % local_size != args.topo.local_rank;
    timeline.task(start, peer ? args.row_bytes : 0, peer ? 0 : args.row_bytes);
  }
}

// Across nodes, heavy group sub-task `sub`: the fp32 sum, in position order,
// of the scaled gradients of its rows (at most kHeavy positions of
// push_order), into an fp32 partial of its own; then it counts itself done
// for its group. Every lane of the warp.
template <typename RowT>
__device__ void group_sub_warp(const OutArgs &args, int64_t sub, int lane) {
  constexpr int kValues = Vec16<RowT>::kValues;
  static_assert(kHeavy % 32 == 0, "a sub-task's rows are spread over the lanes");
  const OutPlan &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int row_bytes = args.row_bytes, vec = args.vec, dim = vec * kValues;
  const int32_t group = plan.group_of_sub[sub];
  const HeavyGroup span = plan.heavy_groups[group];
  const int64_t begin = span.start + (sub - plan.group_first_sub[group]) * kHeavy,
                end = min(begin + kHeavy, (int64_t)span.end);
  float *part = plan.partial + (plan.heavy_subs + sub) * dim;
  if (fanout.exp_off == nullptr) {
    // each lane holds the gradient rows of positions begin + lane, begin + 32 + lane, ...
    int64_t grad_rows[kHeavy / 32];
#pragma unroll
    for (int j = 0; j < kHeavy / 32; ++j)
      grad_rows[j] =
          begin + 32 * j + lane < end ? plan.inv[plan.push_order[begin + 32 * j + lane]] : 0;
    const int count = (int)(end - begin);
    for (int col_base = 0; col_base < vec; col_base += 32) {
      const int col = col_base + lane;
      float acc[kValues];
#pragma unroll
      for (int j = 0; j < kHeavy / 32; ++j)
        for (int i = 32 * j; i < min(count, 32 * j + 32); i += 8) {
          int4 vecs[8];
#pragma unroll
          for (int step = 0; step < 8; ++step) {
            const int64_t out_row = __shfl_sync(kFull, grad_rows[j], (i + step) & 31);
            vecs[step] = i + step < count && col < vec
                             ? __ldcg((const int4 *)(plan.grad + out_row * row_bytes) + col)
                             : make_int4(0, 0, 0, 0);
          }
#pragma unroll
          for (int step = 0; step < 8; ++step) {
            if (i + step >= count) continue;
            float vals[kValues];
            Vec16<RowT>::get(vals, vecs[step]);
#pragma unroll
            for (int elem = 0; elem < kValues; ++elem) {
              vals[elem] = mul_rn(vals[elem], plan.scale);
              acc[elem] = i + step == 0 ? vals[elem] : acc[elem] + vals[elem];
            }
          }
        }
      if (col < vec) store_fp32<kValues>(part, col, acc);
    }
    SubTaskDone{plan.group_done + group}.count_warp(lane);
    return;
  }
  // index dedup: each row is a deduplicated row's fan-in
  for (int64_t pos = begin; pos < end; ++pos) {
    const int64_t dedup_row = plan.inv[plan.push_order[pos]];
    const int64_t first_entry = fanout.exp_off[dedup_row],
                  cnt = fanout.exp_off[dedup_row + 1] - first_entry;
    int64_t num_subs = 0;
    const float *heavy_part =
        cnt > kHeavy ? heavy_row_warp<RowT>(args, dedup_row, cnt, lane, num_subs) : nullptr;
    for (int col = lane; col < vec; col += 32) {
      float vals[kValues];
      if (heavy_part != nullptr)
        sum_fp32<kValues>(heavy_part, num_subs, dim, col, vals);
      else
        sum_vec<RowT>(plan.grad, fanout.exp_idx, first_entry, first_entry + cnt, row_bytes, col,
                      vals);
#pragma unroll
      for (int i = 0; i < kValues; ++i) {
        vals[i] = mul_rn(vals[i], plan.scale);
        if (pos != begin) vals[i] += part[col * kValues + i];
      }
      store_fp32<kValues>(part, col, vals);
    }
  }
  SubTaskDone{plan.group_done + group}.count_warp(lane);
}

// Each lane's `dst` gets the gradient of its row `dedup_row`: the
// requester's output gradient row `dedup_row`, or under index dedup the sum
// over the original rows of deduplicated row `dedup_row`. Rows of one
// original row (most) in one warp copy, those of 2..kHeavy then one at a
// time, heavy ones from their sub-tasks' sums, or, rows of the node's owners
// (`local`), not at all: their last sub-task writes them (heavy_sub_warp).
// Returns how many rows it wrote. Every lane of the warp.
template <typename RowT>
__device__ int push_rows_warp(const OutArgs &args, int64_t dedup_row, char *dst, int num_rows,
                              bool local, int lane) {
  const OutPlan &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int row_bytes = args.row_bytes, vec = args.vec, dim = vec * Vec16<RowT>::kValues;
  if (fanout.exp_off == nullptr) {
    scaled_copy_rows<RowT>(lane < num_rows ? plan.grad + dedup_row * row_bytes : nullptr, dst,
                           num_rows, vec, lane, plan.scale);
    return num_rows;
  }
  int64_t first_entry = 0, cnt = 0;
  const char *src = nullptr;
  if (lane < num_rows) {
    first_entry = fanout.exp_off[dedup_row];
    cnt = fanout.exp_off[dedup_row + 1] - first_entry;
    if (cnt == 1) src = plan.grad + (int64_t)fanout.exp_idx[first_entry] * row_bytes;
  }
  scaled_copy_rows<RowT>(src, cnt <= 1 ? dst : nullptr, num_rows, vec, lane, plan.scale);
  const bool left = local && cnt > kHeavy;
  for (unsigned many = __ballot_sync(kFull, cnt > 1 && !left); many; many &= many - 1) {
    const int src_lane = __ffs(many) - 1;
    const int64_t lane_begin = __shfl_sync(kFull, first_entry, src_lane);
    const int64_t lane_count = __shfl_sync(kFull, cnt, src_lane);
    const int64_t lane_dedup_row = __shfl_sync(kFull, dedup_row, src_lane);
    char *lane_dst = from_lane(dst, src_lane);
    if (lane_count <= kHeavy) {
      for (int col = lane; col < vec; col += 32) {
        float acc[Vec16<RowT>::kValues];
        sum_vec<RowT>(plan.grad, fanout.exp_idx, lane_begin, lane_begin + lane_count, row_bytes,
                      col, acc);
        *((int4 *)lane_dst + col) = scaled<RowT>(acc, plan.scale);
      }
      continue;
    }
    int64_t num_subs;
    const float *part = heavy_row_warp<RowT>(args, lane_dedup_row, lane_count, lane, num_subs);
    for (int col = lane; col < vec; col += 32) {
      float acc[Vec16<RowT>::kValues];
      sum_fp32<Vec16<RowT>::kValues>(part, num_subs, dim, col, acc);
      *((int4 *)lane_dst + col) = scaled<RowT>(acc, plan.scale);
    }
  }
  return num_rows - __popc(__ballot_sync(kFull, left));
}

// Across nodes, heavy groups [group_lo, group_hi) of a push task to `proxy`
// (at most 32): each group's sub-task sums, once done, added in sub-task
// order into its fp32 partial slot in that proxy's proxy_rows (where all of
// the group's rows point; it spans partial_rows rows). Each lane looks up one
// group (group_lo + lane) and waits for its sub-tasks. Every lane of the
// warp.
template <typename RowT>
__device__ void push_groups_warp(const OutArgs &args, const OutTables &tables, int proxy,
                                 int64_t group_lo, int64_t group_hi, int lane) {
  const OutPlan &plan = args.plan;
  const int dim = args.vec * Vec16<RowT>::kValues;
  int64_t num_subs = 0;
  const float *part = nullptr;
  float *slot = nullptr;
  if (group_lo + lane < group_hi) {
    const int64_t group = group_lo + lane;
    const HeavyGroup span = plan.heavy_groups[group];
    num_subs = ceil_div<kHeavy>(span.end - span.start);
    part = plan.partial + (plan.heavy_subs + plan.group_first_sub[group]) * dim;
    slot = (float *)(tables.proxy_rows[proxy] +
                     (int64_t)plan.rows[plan.push_order[span.start]] * args.row_bytes);
    SubTaskDone{plan.group_done + group}.wait_lane(args, num_subs, "sub-tasks of a heavy group");
  }
  __syncwarp();
  for (int j = 0; j < group_hi - group_lo; ++j) {
    const int64_t lane_subs = __shfl_sync(kFull, num_subs, j);
    const float *lane_part = from_lane(part, j);
    float *lane_slot = from_lane(slot, j);
    for (int col = lane; col < dim / 4; col += 32) {
      float acc[4];
      sum_fp32<4>(lane_part, lane_subs, dim, col, acc);
      store_fp32<4>(lane_slot, col, acc);
    }
  }
}

// Remote task `remote_task` (across nodes; `cursor` at its round): a push
// task to the proxy in turn from this rank's own one. False when that proxy
// has no such round.
__device__ __forceinline__ bool remote_push(const OutArgs &args, const OutTables &tables,
                                            const RoundCursor &cursor, int64_t remote_task,
                                            ProxyPush &push) {
  const int local_size = args.topo.local_size, me = args.topo.local_rank;
  const int rnd = cursor.rnd;
  const bool is_group_task = remote_task < cursor.mid;  // a heavy groups' push task
  const int64_t slot = remote_task - (is_group_task ? cursor.lo : cursor.mid);
  const int proxy = (int)((me + slot) % local_size);
  const int64_t nth = slot / local_size;
  if (rnd >= tables.rounds[proxy]) return false;
  ReqEntry entry, next;
  tables.req_of(args, proxy, rnd, entry, next);
  const int64_t plain_tasks = ceil_div<kTaskRows>(entry.heavy - entry.start);
  push.is_group_task = is_group_task;
  push.proxy = proxy;
  push.rnd = rnd;
  push.nth = nth;
  push.plain_tasks = plain_tasks;
  push.entry = entry;
  push.next = next;
  return true;
}

// A push task of this rank's heavy groups at the proxy (32 at most): their
// sums into their partial slots, then the task's flag. Every lane of the
// warp.
template <typename RowT>
__device__ __forceinline__ void push_groups_task_warp(const OutArgs &args, OutTables &tables,
                                                      const ProxyPush &push, int lane) {
  const int me = args.topo.local_rank;
  const OutTimeline timeline(args);
  const int proxy = push.proxy, rnd = push.rnd;
  const int64_t nth = push.nth, plain_tasks = push.plain_tasks;
  const ReqEntry &entry = push.entry, &next = push.next;
  const int64_t group_lo = entry.groups + nth * kTaskRows;
  if (group_lo >= next.groups) return;
  wait_half_warp(args, tables, proxy, rnd, lane);
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  const int64_t group_hi = min(group_lo + kTaskRows, (int64_t)next.groups);
  push_groups_warp<RowT>(args, tables, proxy, group_lo, group_hi, lane);
  push_flag_warp(args, tables, proxy, rnd, entry.flags + plain_tasks + nth, lane);
  if (timeline.on() && lane == 0) {
    // an fp32 partial row per group
    const unsigned long long bytes =
        (unsigned long long)(group_hi - group_lo) * 4 * args.vec * Vec16<RowT>::kValues;
    timeline.task(start, proxy != me ? bytes : 0, proxy != me ? 0 : bytes);
  }
}

// A push task of this rank's rows at the proxy (32 positions of its push
// order there): their gradients into their rows of proxy_rows, then the
// task's flag. Every lane of the warp.
template <typename RowT>
__device__ __forceinline__ void push_rows_task_warp(const OutArgs &args, OutTables &tables,
                                                    const ProxyPush &push, int lane) {
  const OutPlan &plan = args.plan;
  const int me = args.topo.local_rank, row_bytes = args.row_bytes;
  const OutTimeline timeline(args);
  const int proxy = push.proxy, rnd = push.rnd;
  const int64_t nth = push.nth, plain_tasks = push.plain_tasks;
  const ReqEntry &entry = push.entry;
  if (nth >= plain_tasks) return;
  wait_half_warp(args, tables, proxy, rnd, lane);
  const int64_t pos0 = entry.start + nth * kTaskRows;
  const int num_rows = (int)min((int64_t)kTaskRows, entry.heavy - pos0);
  int64_t dedup_row = 0;
  char *dst = nullptr;
  if (lane < num_rows) {
    const int32_t row = plan.push_order[pos0 + lane];
    dedup_row = plan.inv[row];
    dst = tables.proxy_rows[proxy] + (int64_t)plan.rows[row] * row_bytes;
  }
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  push_rows_warp<RowT>(args, dedup_row, dst, num_rows, false, lane);
  push_flag_warp(args, tables, proxy, rnd, entry.flags + nth, lane);
  if (timeline.on() && lane == 0) {
    const unsigned long long bytes = (unsigned long long)num_rows * row_bytes;
    timeline.task(start, proxy != me ? bytes : 0, proxy != me ? 0 : bytes);
  }
}

// Row task `index` of the node's owners: its rows into the owner's
// lookup_rows, but those of heavy rows (their last sub-task writes them).
// Every lane of the warp.
template <typename RowT>
__device__ __forceinline__ void local_push_warp(const OutArgs &args, const OutTables &tables,
                                                const LocalRows &local_rows, int64_t index,
                                                int lane) {
  const OutPlan &plan = args.plan;
  const int me = args.topo.local_rank, row_bytes = args.row_bytes;
  const OutTimeline timeline(args);
  const RowTask job = local_rows.task(index);
  if (job.count == 0) return;
  const int num_rows = job.count;
  int64_t dedup_row = 0;
  char *dst = nullptr;
  if (lane < job.count) {
    const int64_t row = job.row0 + lane;
    dedup_row = plan.inv[row];
    dst = tables.lookup_rows[job.owner] + (int64_t)plan.rows[row] * row_bytes;
  }
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  const int wrote = push_rows_warp<RowT>(args, dedup_row, dst, num_rows, true, lane);
  if (timeline.on() && lane == 0) {
    const unsigned long long bytes = (unsigned long long)wrote * row_bytes;
    timeline.task(start, job.owner != me ? bytes : 0, job.owner != me ? 0 : bytes);
  }
}

}  // namespace bwd

}  // namespace hier_output_dist
}  // namespace dyn_emb
