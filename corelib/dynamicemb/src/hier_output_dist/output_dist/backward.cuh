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
// The output dist's backward kernel: its row sums, its requester's
// pushes, the proxy's reducer, its GIN warps (the proxy's issuer and the
// landing watcher), and the kernel.
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

// The backward's sums of gradient rows, in fp32 and in a fixed order, and its
// scaled copies of them.

template <typename RowT>
__device__ __forceinline__ int4 scaled(float *vals, float scale) {
#pragma unroll
  for (int i = 0; i < Vec16<RowT>::kValues; ++i) vals[i] = mul_rn(vals[i], scale);
  return Vec16<RowT>::pack(vals);
}

// Vector `col` of the gradient rows of original rows exp_idx[begin .. end),
// begin < end, summed in fp32 in that order into acc.
template <typename RowT>
__device__ __forceinline__ void sum_vec(const char *grad, const int32_t *exp_idx, int64_t begin,
                                        int64_t end, int row_bytes, int col, float *acc) {
  Vec16<RowT>::get(acc, __ldcg((const int4 *)(grad + (int64_t)exp_idx[begin] * row_bytes) + col));
  for (int64_t entry = begin + 1; entry < end; entry += 4) {
    int4 vecs[4];
#pragma unroll
    for (int step = 0; step < 4; ++step)
      vecs[step] =
          entry + step < end
              ? __ldcg((const int4 *)(grad + (int64_t)exp_idx[entry + step] * row_bytes) + col)
              : make_int4(0, 0, 0, 0);
#pragma unroll
    for (int step = 0; step < 4; ++step)
      if (entry + step < end) Vec16<RowT>::add(acc, vecs[step]);
  }
}

// fp32 values vals[0 .. Values) into vector `col` (`Values` floats) of the
// fp32 row at `row`.
template <int Values>
__device__ __forceinline__ void store_fp32(float *row, int col, const float *vals) {
#pragma unroll
  for (int i = 0; i < Values; i += 4)
    *(float4 *)(row + col * Values + i) =
        make_float4(vals[i], vals[i + 1], vals[i + 2], vals[i + 3]);
}

// Vector `col` (`Values` floats) of fp32 rows part[0], part[dim], ...
// part[(count - 1) * dim], count >= 1, summed in that order into acc, 4 rows'
// loads in flight.
template <int Values>
__device__ __forceinline__ void sum_fp32(const float *part, int64_t count, int dim, int col,
                                         float *acc) {
#pragma unroll
  for (int i = 0; i < Values; i += 4) {
    const float4 val = __ldcg((const float4 *)(part + col * Values + i));
    acc[i] = val.x;
    acc[i + 1] = val.y;
    acc[i + 2] = val.z;
    acc[i + 3] = val.w;
  }
  for (int64_t j = 1; j < count; j += 4) {
    float4 vals[4][Values / 4];
#pragma unroll
    for (int step = 0; step < 4; ++step)
#pragma unroll
      for (int i = 0; i < Values; i += 4)
        vals[step][i / 4] =
            j + step < count ? __ldcg((const float4 *)(part + (j + step) * dim + col * Values + i))
                             : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      if (j + step >= count) continue;
#pragma unroll
      for (int i = 0; i < Values; i += 4) {
        acc[i] += vals[step][i / 4].x;
        acc[i + 1] += vals[step][i / 4].y;
        acc[i + 2] += vals[step][i / 4].z;
        acc[i + 3] += vals[step][i / 4].w;
      }
    }
  }
}

// Each lane's row at its src, scaled, into its dst: the lanes over (row,
// vector) pairs as in warp_copy_rows, 4 vectors in flight per lane. A row
// whose dst is nullptr is skipped, and one whose src alone is nullptr gets
// zeros.
template <typename RowT>
__device__ __forceinline__ void scaled_copy_rows(const char *src, char *dst, int num_rows, int vec,
                                                 int lane, float scale) {
  const int total = num_rows * vec;
  for (int base = 0; base < total; base += 32 * 4) {
    int4 vecs[4];
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int idx = base + step * 32 + lane;
      const int row = idx / vec;
      const char *src_row = from_lane(src, row & 31);
      vecs[step] = idx < total && src_row != nullptr
                       ? __ldcg((const int4 *)src_row + (idx - row * vec))
                       : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int idx = base + step * 32 + lane;
      const int row = idx / vec;
      char *dst_row = from_lane(dst, row & 31);
      if (idx < total && dst_row != nullptr) {
        float vals[Vec16<RowT>::kValues];
        Vec16<RowT>::get(vals, vecs[step]);
        *((int4 *)dst_row + (idx - row * vec)) = scaled<RowT>(vals, scale);
      }
    }
  }
}

// Vector `col` (when `live`) of the rows of reducer-list entries [begin, end)
// in this rank's proxy_rows (`rows`), summed in fp32 in list order into acc:
// rows of the key's type, or (Partial) a heavy key's fp32 partial rows, those
// of requesters without one (kEntryNoPush) skipped. Each lane loads its entry
// of each 32, the warp then 8 rows at a time.
template <typename RowT, bool Partial>
__device__ __forceinline__ void sum_entries(const RowsLayout &rows, const int64_t *ent,
                                            int64_t begin, int64_t end, int col, bool live,
                                            int lane, float *acc) {
  constexpr int kValues = Vec16<RowT>::kValues,
                kLoads = Partial ? kValues / 4 : 1;  // 16-byte loads per vector
  bool any = false;
  for (int64_t block0 = begin; block0 < end; block0 += 32) {
    const int block_len = (int)min((int64_t)32, end - block0);
    const uint64_t mine = lane < block_len ? (uint64_t)ent[block0 + lane] : 0;
    for (int i = 0; i < block_len; i += 8) {
      int4 vecs[8][kLoads];
      bool use[8];
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        const uint64_t entry = __shfl_sync(kFull, mine, (i + step) & 31);
        use[step] = i + step < block_len && (!Partial || (int32_t)(entry >> 32) != kEntryNoPush);
        const int4 *src =
            (const int4 *)rows.proxy_row((int64_t)(entry & ~kEntryPartial)) + col * kLoads;
#pragma unroll
        for (int load = 0; load < kLoads; ++load)
          vecs[step][load] = use[step] && live ? __ldcg(src + load) : make_int4(0, 0, 0, 0);
      }
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        if (!use[step]) continue;
        float vals[kValues];
        if constexpr (Partial)
          memcpy(vals, vecs[step], sizeof(vals));
        else
          Vec16<RowT>::get(vals, vecs[step][0]);
#pragma unroll
        for (int elem = 0; elem < kValues; ++elem)
          acc[elem] = any ? acc[elem] + vals[elem] : vals[elem];
        any = true;
      }
    }
  }
}

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

// The done counter of a heavy row's sub-tasks (index dedup) or of a heavy
// group's: each sub-task counts itself done once its warp wrote its fp32 sum,
// and what adds the sums up waits until all have.
struct SubTaskDone {
  uint64_t *word;
  // One lane: waits until `count` sub-tasks counted themselves done.
  __device__ __forceinline__ void wait_lane(const BwdArgs &args, int64_t count,
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
__device__ __forceinline__ void wait_half_warp(const BwdArgs &args, const OutTables &tables,
                                               int proxy, int rnd, int lane) {
  if (rnd < 2) return;
  if (lane == 0) {
    const uint64_t want = grant_of(args.ch.seq, rnd);
    if (ld_acquire_cta(tables.half_seen(proxy)) < want) {
      Spin spin(args.timeout_ns);
      uint64_t word;
      while ((word = ld_acquire_sys(tables.half_free(proxy))) < want)
        spin.pause("a half of a proxy's rows", proxy, rnd, args.topo.rank);
      red_max_release_cta(tables.half_seen(proxy), word);
    }
  }
  __syncwarp();
}

// A push task's rows (or partials) of round rnd are in the proxy's
// proxy_rows: its flag `flag` in the round's half of that proxy's flags,
// tagged with (out_seq, rnd). Every lane of the warp.
__device__ __forceinline__ void push_flag_warp(const BwdArgs &args, const OutTables &tables,
                                               int proxy, int rnd, int64_t flag, int lane) {
  __syncwarp();
  if (lane == 0) st_release_sys(tables.push_flag(proxy, rnd, flag), grant_of(args.ch.seq, rnd));
}

// Index dedup, a heavy row (more than kHeavy original rows): once its
// sub-tasks are done, where their sums are (num_subs of them, in sub-task
// order). Every lane of the warp.
template <typename RowT>
__device__ __forceinline__ const float *heavy_row_warp(const BwdArgs &args, int64_t dedup_row,
                                                       int64_t cnt, int lane, int64_t &num_subs) {
  const Fanout &fanout = args.plan.fanout;
  const int32_t slot = fanout.slot[dedup_row];
  num_subs = ceil_div<kHeavy>(cnt);
  if (lane == 0)
    SubTaskDone{args.heavy_done + slot}.wait_lane(args, num_subs, "sub-tasks of a heavy row");
  __syncwarp();
  return args.partial + (int64_t)fanout.first[slot] * (args.vec * Vec16<RowT>::kValues);
}

// Index dedup: heavy row sub-task `sub` sums kHeavy of its original rows
// into an fp32 partial of its own, then counts itself done for its row. It
// runs in its row's pass: `remote` for a row an owner on another node gets
// (through a proxy), whose push task adds up the row's sub-tasks; else for a
// row of one of the node's owners, where the last of the row's sub-tasks to
// end adds them up, in sub-task order, and writes the row into the owner's
// lookup_rows. Every lane of the warp.
template <typename RowT>
__device__ void heavy_sub_warp(const BwdArgs &args, const OutTables &tables, int64_t sub,
                               bool remote, int lane) {
  constexpr int kValues = Vec16<RowT>::kValues;
  const BwdPlanView &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int dim = args.vec * kValues, local_size = args.topo.local_size;
  const int64_t dedup_row = fanout.sub[2 * sub], begin = fanout.sub[2 * sub + 1],
                row_end = fanout.exp_off[dedup_row + 1];
  const int32_t slot = fanout.slot[dedup_row], row = fanout.pos[slot];
  const int owner = tables.owner_of(row);
  if ((owner / local_size != args.topo.node) != remote) return;
  const OutTimeline timeline(args);
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  const int64_t end = min(begin + kHeavy, row_end);
  float *part = args.partial + sub * dim;
  for (int col = lane; col < args.vec; col += 32) {
    float acc[kValues];
    sum_vec<RowT>(args.grad, fanout.exp_idx, begin, end, args.row_bytes, col, acc);
    store_fp32<kValues>(part, col, acc);
  }
  if (remote) {
    SubTaskDone{args.heavy_done + slot}.count_warp(lane);
    return;
  }
  __syncwarp();
  uint64_t before = 0;
  if (lane == 0) before = SubTaskDone{args.heavy_done + slot}.count_lane();
  __syncwarp();
  const int64_t num_subs = ceil_div<kHeavy>(row_end - fanout.exp_off[dedup_row]);
  if ((int64_t)__shfl_sync(kFull, before, 0) + 1 < num_subs) return;
  // the last one: every sub-task of the row has ended (acquired above)
  const float *sums = args.partial + (int64_t)fanout.first[slot] * dim;
  char *dst = tables.lookup_row(owner % local_size, plan.dst[row]);
  for (int col = lane; col < args.vec; col += 32) {
    float acc[kValues];
    sum_fp32<kValues>(sums, num_subs, dim, col, acc);
    *((int4 *)dst + col) = scaled<RowT>(acc, args.scale);
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
__device__ void group_sub_warp(const BwdArgs &args, int64_t sub, int lane) {
  constexpr int kValues = Vec16<RowT>::kValues;
  static_assert(kHeavy % 32 == 0, "a sub-task's rows are spread over the lanes");
  const BwdPlanView &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int row_bytes = args.row_bytes, vec = args.vec, dim = vec * kValues;
  const int32_t group = plan.group_of_sub[sub];
  const HeavyGroup span = plan.heavy_groups[group];
  const int64_t begin = span.start + (sub - plan.group_first_sub[group]) * kHeavy,
                end = min(begin + kHeavy, (int64_t)span.end);
  float *part = args.partial + (args.heavy_subs + sub) * dim;
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
                             ? __ldcg((const int4 *)(args.grad + out_row * row_bytes) + col)
                             : make_int4(0, 0, 0, 0);
          }
#pragma unroll
          for (int step = 0; step < 8; ++step) {
            if (i + step >= count) continue;
            float vals[kValues];
            Vec16<RowT>::get(vals, vecs[step]);
#pragma unroll
            for (int elem = 0; elem < kValues; ++elem) {
              vals[elem] = mul_rn(vals[elem], args.scale);
              acc[elem] = i + step == 0 ? vals[elem] : acc[elem] + vals[elem];
            }
          }
        }
      if (col < vec) store_fp32<kValues>(part, col, acc);
    }
    SubTaskDone{args.group_done + group}.count_warp(lane);
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
        sum_vec<RowT>(args.grad, fanout.exp_idx, first_entry, first_entry + cnt, row_bytes, col,
                      vals);
#pragma unroll
      for (int i = 0; i < kValues; ++i) {
        vals[i] = mul_rn(vals[i], args.scale);
        if (pos != begin) vals[i] += part[col * kValues + i];
      }
      store_fp32<kValues>(part, col, vals);
    }
  }
  SubTaskDone{args.group_done + group}.count_warp(lane);
}

// Each lane's `dst` gets the gradient of its row `dedup_row`: the
// requester's output gradient row `dedup_row`, or under index dedup the sum
// over the original rows of deduplicated row `dedup_row`. Rows of one
// original row (most) in one warp copy, those of 2..kHeavy then one at a
// time, heavy ones from their sub-tasks' sums, or, rows of the node's owners
// (`local`), not at all: their last sub-task writes them (heavy_sub_warp).
// Returns how many rows it wrote. Every lane of the warp.
template <typename RowT>
__device__ int push_rows_warp(const BwdArgs &args, int64_t dedup_row, char *dst, int num_rows,
                              bool local, int lane) {
  const BwdPlanView &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int row_bytes = args.row_bytes, vec = args.vec, dim = vec * Vec16<RowT>::kValues;
  if (fanout.exp_off == nullptr) {
    scaled_copy_rows<RowT>(lane < num_rows ? args.grad + dedup_row * row_bytes : nullptr, dst,
                           num_rows, vec, lane, args.scale);
    return num_rows;
  }
  int64_t first_entry = 0, cnt = 0;
  const char *src = nullptr;
  if (lane < num_rows) {
    first_entry = fanout.exp_off[dedup_row];
    cnt = fanout.exp_off[dedup_row + 1] - first_entry;
    if (cnt == 1) src = args.grad + (int64_t)fanout.exp_idx[first_entry] * row_bytes;
  }
  scaled_copy_rows<RowT>(src, cnt <= 1 ? dst : nullptr, num_rows, vec, lane, args.scale);
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
        sum_vec<RowT>(args.grad, fanout.exp_idx, lane_begin, lane_begin + lane_count, row_bytes,
                      col, acc);
        *((int4 *)lane_dst + col) = scaled<RowT>(acc, args.scale);
      }
      continue;
    }
    int64_t num_subs;
    const float *part = heavy_row_warp<RowT>(args, lane_dedup_row, lane_count, lane, num_subs);
    for (int col = lane; col < vec; col += 32) {
      float acc[Vec16<RowT>::kValues];
      sum_fp32<Vec16<RowT>::kValues>(part, num_subs, dim, col, acc);
      *((int4 *)lane_dst + col) = scaled<RowT>(acc, args.scale);
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
__device__ void push_groups_warp(const BwdArgs &args, const OutTables &tables, int proxy,
                                 int64_t group_lo, int64_t group_hi, int lane) {
  const BwdPlanView &plan = args.plan;
  const int dim = args.vec * Vec16<RowT>::kValues;
  int64_t num_subs = 0;
  const float *part = nullptr;
  float *slot = nullptr;
  if (group_lo + lane < group_hi) {
    const int64_t group = group_lo + lane;
    const HeavyGroup span = plan.heavy_groups[group];
    num_subs = ceil_div<kHeavy>(span.end - span.start);
    part = args.partial + (args.heavy_subs + plan.group_first_sub[group]) * dim;
    slot = (float *)tables.proxy_row(proxy, plan.dst[plan.push_order[span.start]]);
    SubTaskDone{args.group_done + group}.wait_lane(args, num_subs, "sub-tasks of a heavy group");
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
__device__ __forceinline__ bool remote_push(const BwdArgs &args, const OutTables &tables,
                                            const RoundCursor &cursor, int64_t remote_task,
                                            ProxyPush &push) {
  const int local_size = args.topo.local_size, me = args.topo.local_rank;
  const int rnd = cursor.rnd;
  const bool is_group_task = remote_task < cursor.mid;  // a heavy groups' push task
  const int64_t slot = remote_task - (is_group_task ? cursor.lo : cursor.mid);
  const int proxy = (int)((me + slot) % local_size);
  const int64_t nth = slot / local_size;
  if (rnd >= tables.rounds(proxy)) return false;
  ReqEntry entry, next;
  tables.req_of(proxy, rnd, entry, next);
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
__device__ __forceinline__ void push_groups_task_warp(const BwdArgs &args, const OutTables &tables,
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
__device__ __forceinline__ void push_rows_task_warp(const BwdArgs &args, const OutTables &tables,
                                                    const ProxyPush &push, int lane) {
  const BwdPlanView &plan = args.plan;
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
    dst = tables.proxy_row(proxy, plan.dst[row]);
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
__device__ __forceinline__ void local_push_warp(const BwdArgs &args, const OutTables &tables,
                                                const LocalRows &local_rows, int64_t index,
                                                int lane) {
  const BwdPlanView &plan = args.plan;
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
    dst = tables.lookup_row(job.owner, plan.dst[row]);
  }
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  const int wrote = push_rows_warp<RowT>(args, dedup_row, dst, num_rows, true, lane);
  if (timeline.on() && lane == 0) {
    const unsigned long long bytes = (unsigned long long)wrote * row_bytes;
    timeline.task(start, job.owner != me ? bytes : 0, job.owner != me ? 0 : bytes);
  }
}

}  // namespace bwd

// The proxy's reducers in the backward (across nodes, the last CTAs): their
// tasks (bwd::ProxyReducer).

namespace bwd {

// The proxy's reducers (across nodes; the last CTAs of the kernel). A task
// is 32 node-unique keys of one IB chunk of an owner node. The tasks go round
// after round (reduce_task_starts); in a round the chunks follow the
// requesters' push order (chunk-major, then owner node). The task's keys'
// reducer lists are consecutive. The warp waits for the push task of each
// entry (the lanes over the entries, 32 apart; the flags of the round's half,
// tagged with the round). Then it sums, for each key of several entries,
// their rows in fp32 in list order (a key's occurrences by requester and
// position; a heavy key's requesters' partials in requester order) and rounds
// the sum once into the key's row. Then it counts the task's keys ready in
// the chunk's counter, where the issuer looks. A key of one row is not
// touched: the NIC reads it where its requester's NVLink stores left it
// (released by the task flag, acquired here, released again by the counter).
template <typename RowT>
struct ProxyReducer {
  // Every thread of a reducer CTA: the CTA's warps take the tasks.
  __device__ static void run(const BwdArgs &args, int lane);
};

template <typename RowT>
__device__ void ProxyReducer<RowT>::run(const BwdArgs &args, int lane) {
  const BwdPlanView &plan = args.plan;
  const int num_nodes = args.topo.num_nodes, node = args.topo.node, vec = args.vec;
  const int64_t chunk_tasks = args.chunk_rows / kTaskRows;  // tasks per chunk
  // this rank's copy of the window: its proxy_rows and push flags
  const WindowLayout window = NodeWindow(args.window, args.topo.local_rank).layout(args);
  const int64_t total = plan.reduce_task_starts[*plan.num_rounds];
  RoundCursor cursor(plan.reduce_task_starts, 1);  // the round of this warp's tasks
  const TaskQueue queue(args, kQueueReduce);
  for (int64_t task = queue.next_warp(); task < total; task = queue.next_warp()) {
    cursor.seek(task);
    // The task: keys [uniq_lo, uniq_lo + num_keys) of owner node owner_node,
    // in IB chunk `chunk` of its keys of round rnd (piece). None for a chunk
    // of this node's own keys, or past the end of the chunk's keys.
    const int rnd = cursor.rnd;
    const int64_t slot = task - cursor.lo;
    const int owner_node = (int)(slot / chunk_tasks % num_nodes);
    const int64_t chunk = slot / chunk_tasks / num_nodes, sub = slot % chunk_tasks;
    if (owner_node == node) continue;
    // from its round table: where the owner node's keys of the round sit in
    // proxy_rows, and their first IB chunk
    const RoundPiece piece = plan.round_piece(rnd, owner_node);
    const int64_t uniq_lo = piece.uniq_lo + chunk * args.chunk_rows + sub * kTaskRows;
    const int64_t chunk_end =
        min((int64_t)piece.uniq_hi, piece.uniq_lo + (chunk + 1) * args.chunk_rows);
    const int num_keys = (int)min((int64_t)kTaskRows, chunk_end - uniq_lo);
    if (num_keys <= 0) continue;
    // The lane's key (a lane per key): its reducer list [begin, end) (heavy:
    // the first entry is the heavy key's own row) and its row in proxy_rows.
    // Then the wait for the push task of each entry of the task's lists.
    const uint64_t tag = grant_of(args.ch.seq, rnd);
    const uint64_t *flags = window.rows.push_flags(rnd);
    const int64_t uniq_global = plan.uniq_off[owner_node] + uniq_lo;  // in the reducer list
    int64_t begin = 0, end = 0;
    bool heavy = false;
    char *row = nullptr;
    if (lane < num_keys) {
      begin = plan.reducer_start[uniq_global + lane];
      end = plan.reducer_start[uniq_global + lane + 1];
      heavy = (int32_t)(plan.reducer_entries[begin] >> 32) == kEntryOwnRow;
      row = window.rows.proxy_row(piece.off + uniq_lo + lane - piece.uniq_lo);
    }
    const int64_t ent_end = __shfl_sync(kFull, end, num_keys - 1);
    Spin spin(args.timeout_ns);
    for (int64_t i = __shfl_sync(kFull, begin, 0) + lane; i < ent_end; i += 32) {
      const int32_t flag = (int32_t)(plan.reducer_entries[i] >> 32);
      if (flag < 0) continue;  // kEntryNoPush, kEntryOwnRow: no push task
      while (ld_acquire_sys(&flags[flag]) < tag)
        spin.pause("pushed gradient rows", flag, uniq_lo, args.topo.rank);
    }
    __syncwarp();
    // The keys of several entries, each summed in fp32 in list order (a heavy
    // key's own row gets the sum), and rounded once into the key's row.
    for (unsigned many = __ballot_sync(kFull, lane < num_keys && (heavy || end - begin > 1)); many;
         many &= many - 1) {
      const int src_lane = __ffs(many) - 1;
      const int64_t lane_begin = __shfl_sync(kFull, begin, src_lane),
                    lane_end = __shfl_sync(kFull, end, src_lane);
      const bool lane_heavy = __shfl_sync(kFull, heavy, src_lane);
      char *dst = from_lane(row, src_lane);
      for (int col_base = 0; col_base < vec; col_base += 32) {
        const int col = col_base + lane;
        float acc[Vec16<RowT>::kValues];
        if (lane_heavy)  // its own row (the first entry) gets the sum
          sum_entries<RowT, true>(window.rows, plan.reducer_entries, lane_begin + 1, lane_end, col,
                                  col < vec, lane, acc);
        else
          sum_entries<RowT, false>(window.rows, plan.reducer_entries, lane_begin, lane_end, col,
                                   col < vec, lane, acc);
        if (col < vec) *((int4 *)dst + col) = Vec16<RowT>::pack(acc);
      }
    }
    __syncwarp();
    // The task's keys are reduced: counted ready in the chunk's counter.
    if (lane == 0)
      red_release_sys((uint64_t *)&args.ready[piece.chunk + chunk], (uint64_t)num_keys);
  }
}

}  // namespace bwd

// The backward's GIN warps (across nodes, CTA 0, which skip the entry
// barrier): the proxy's issuer, and when tracing the warp that watches this
// rank's landings.

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
  __device__ static void run(const BwdArgs &args);
};

__device__ void ProxyIssuer::run(const BwdArgs &args) {
  __shared__ int at_round[32];         // each lane's round: it has put the rounds before
  __shared__ unsigned at_half[32][2];  // its puts out of each half
  const BwdPlanView &plan = args.plan;
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
          uint64_t *ready = (uint64_t *)&args.ready[chunk];
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
__device__ __forceinline__ void lane_landed_rows(const BwdArgs &args, const ncclGin &rows, int lane,
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
  __device__ static void run(const BwdArgs &args);
  // The backward's last CTA when it is CTA 0 (the watcher stopped before
  // every row had landed), every lane of its warp 0: the rows of the lane's
  // nodes the watcher had not seen, binned into the kernel's row as they land
  // (the CTA's shared copy is folded already), and when the last one landed.
  __device__ static void rest(const BwdArgs &args, int lane);
};

__device__ void LandingWatcher::run(const BwdArgs &args) {
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

__device__ void LandingWatcher::rest(const BwdArgs &args, int lane) {
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

namespace bwd {

// Across nodes, the end of a backward waits (warp 0 of the last CTA, its
// lanes taking the rail peers in turn) for the gradient rows the proxy on
// each other node puts into this owner's lookup_rows: its bwd_rows signal
// against the running total (bwd_tot, of this rank's control words `ctrl`).
__device__ __forceinline__ void wait_rail_rows_warp(const BwdArgs &args, const CtrlLayout &ctrl) {
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
__device__ __forceinline__ void exit(const BwdArgs &args) {
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
// CTAs reduce (bwd::ProxyReducer). Every other warp requests (the bwd::
// steps above): it zeroes the counters this kernel counts from 0
// (before this rank publishes its out_seq), passes the entry barrier and
// takes its tasks. Every CTA then arrives, and the last one ends the
// backward for this rank (bwd::exit).
template <typename RowT>
__global__ void __launch_bounds__(kThreads, 1) bwd_kernel(BwdArgs args) {
  const BwdPlanView &plan = args.plan;
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
    const OutTables tables(args, args.plan);
    // half_seen starts at 0; the entry barrier's sync publishes it
    for (int proxy = tid; proxy < local_size; proxy += workers) *tables.half_seen(proxy) = 0;
    int64_t groups = 0;  // this rank's heavy groups, at all its proxies
    if (num_nodes > 1)
      groups = plan.push_entry(local_size - 1, tables.rounds(local_size - 1))->groups;
    if (blockIdx.x == 0) {
      // Counters this kernel counts from 0, before this rank publishes its
      // out_seq: everything that counts on them runs after the entry.
      const int64_t slots = fanout.exp_off != nullptr ? fanout.count[0] : 0;
      for (int64_t i = tid; i < slots; i += workers) args.heavy_done[i] = 0;
      for (int64_t i = tid; i < groups; i += workers) args.group_done[i] = 0;
      if (num_nodes > 1)
        for (int64_t i = tid; i < *plan.num_chunks; i += workers) args.ready[i] = 0;
      cta_sync(workers);
    }
    pass_entry_barrier(args, workers);
    const int64_t *node_block_start = plan.block_start + node * local_size;
    // Every heavy row's sub-task has a slot before the remote tasks (across
    // nodes), which those of rows to other nodes take, and one among the row
    // tasks, which those of the node's rows take.
    const int64_t heavy_subs = fanout.exp_off != nullptr ? fanout.count[1] : 0,
                  early_subs = num_nodes > 1 ? heavy_subs : 0;
    const int64_t subs = early_subs + args.group_subs,
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
