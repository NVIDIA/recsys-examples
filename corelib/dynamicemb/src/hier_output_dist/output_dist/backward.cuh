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
// of each 32, the warp then 8 rows at a time (only partials need the flag).
template <typename RowT, bool Partial>
__device__ __forceinline__ void sum_entries(const RowsLayout &rows, const ReducerEntry *ent,
                                            int64_t begin, int64_t end, int col, bool live,
                                            int lane, float *acc) {
  constexpr int kValues = Vec16<RowT>::kValues,
                kLoads = Partial ? kValues / 4 : 1;  // 16-byte loads per vector
  bool any = false;
  for (int64_t block0 = begin; block0 < end; block0 += 32) {
    const int block_len = (int)min((int64_t)32, end - block0);
    const ReducerEntry mine = lane < block_len ? ent[block0 + lane] : ReducerEntry{0, 0};
    for (int i = 0; i < block_len; i += 8) {
      int4 vecs[8][kLoads];
      bool use[8];
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        ReducerEntry entry;
        entry.slot = __shfl_sync(kFull, mine.slot, (i + step) & 31);
        entry.flag = Partial ? __shfl_sync(kFull, mine.flag, (i + step) & 31) : 0;
        use[step] = i + step < block_len && (!Partial || entry.flag != kEntryNoPush);
        const int4 *src = (const int4 *)rows.proxy_row(entry.proxy_row()) + col * kLoads;
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
//   2. Chunk after chunk of the proxies' streams (bwd_task_starts), the push
//      tasks of every proxy's heavy groups, then those of every proxy's rows
//      (32 positions of its push order each), proxies in turn in both, so
//      the chunks fill in stream order. The heavy groups come first: none of
//      a chunk's keys is ready before their partials.
//   3. The row tasks of the node's owners, with the sub-tasks of the node's
//      heavy rows spread evenly among them (the pushes keep NVLink busy while
//      the sub-tasks read HBM). The last sub-task of such a row to end adds
//      them up and writes the row, which its row task leaves out.
// A task that needs sub-tasks waits until their done counters say they
// ended. They come earlier in the queue, so a running warp has them, and they
// wait for nothing. A push into a chunk whose rows are past the ring's first
// lap waits until the proxy freed them (slot_free), which needs only the
// pushes of earlier chunks, earlier in the queue. The kernel runs the task
// loop (bwd_kernel); these functions are its steps.

// The done counter of a heavy row's sub-tasks (index dedup) or of a heavy
// group's: each sub-task counts itself done once its warp wrote its fp32 sum,
// and what adds the sums up waits until all have.
struct SubTaskDone {
  uint64_t *word;
  // One lane: waits until `count` sub-tasks counted themselves done.
  __device__ __forceinline__ void wait_lane(const Sizes &sizes, int64_t count,
                                            const char *what) const {
    Spin spin(sizes.timeout_ns);
    while ((int64_t)ld_acquire_gpu(word) < count)
      spin.pause(what, (long long)*word, count, sizes.topo.rank);
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

// One push task to a proxy (across nodes): the proxy, the chunk, the task's
// place among this rank's tasks of its kind there (nth), and this rank's
// entries of that proxy's push table for the chunk (entry; next holds the
// ends).
struct ProxyPush {
  bool is_group_task;  // a task of heavy groups, else of rows
  int proxy;
  int64_t chunk, nth;
  int64_t plain_tasks;  // this rank's tasks of rows at the proxy in the chunk
  ReqEntry entry, next;
};

// Before a push into chunk `chunk` of a proxy's stream: rows of the ring's
// first lap (the stream's positions below num_proxy_rows) are free from the
// start (the previous out kernel was done with proxy_rows before this one's
// entry); the chunk's are free once the proxy freed the rows of its stream
// up to the chunk's end (slot_free), when the NIC had read the sums of the
// chunks there before. What one warp of the CTA saw is kept for the others
// (free_seen), so most tasks read no word of the proxy. seq: the backward's
// out_seq. Every lane of the warp.
__device__ __forceinline__ void wait_free_warp(const Sizes &sizes, int64_t seq,
                                               const OutTables &tables, int proxy, int64_t chunk,
                                               int lane) {
  const int64_t end = tables.plan.proxy_end(proxy, chunk);
  if (end <= sizes.num_proxy_rows) return;
  if (lane == 0) {
    const uint64_t want = tag_of(seq, (uint64_t)end);
    if (ld_acquire_cta(tables.free_seen(proxy)) < want) {
      Spin spin(sizes.timeout_ns);
      uint64_t word;
      while ((word = ld_acquire_sys(tables.slot_free(proxy))) < want)
        spin.pause("rows of a proxy's ring", proxy, chunk, sizes.topo.rank);
      red_max_release_cta(tables.free_seen(proxy), word);
    }
  }
  __syncwarp();
}

// A push task's rows (or partials) are in the proxy's proxy_rows: its flag
// `flag` in that proxy's push flags, tagged with seq. Every lane of the warp.
__device__ __forceinline__ void push_flag_warp(int64_t seq, const OutTables &tables, int proxy,
                                               int64_t flag, int lane) {
  __syncwarp();
  if (lane == 0) st_release_sys(tables.push_flag(proxy, flag), (uint64_t)seq);
}

// Index dedup, a heavy row (more than kHeavy original rows): once its
// sub-tasks are done, where their sums are (num_subs of them, in sub-task
// order). Every lane of the warp.
template <typename RowT>
__device__ __forceinline__ const float *heavy_row_warp(const Sizes &sizes, const BwdArgs &args,
                                                       int64_t dedup_row, int64_t cnt, int lane,
                                                       int64_t &num_subs) {
  const Fanout &fanout = args.plan.fanout;
  const int32_t slot = fanout.slot[dedup_row];
  num_subs = ceil_div<kHeavy>(cnt);
  if (lane == 0)
    SubTaskDone{args.heavy_done + slot}.wait_lane(sizes, num_subs, "sub-tasks of a heavy row");
  __syncwarp();
  return args.partial + (int64_t)fanout.first[slot] * (sizes.row_vecs() * Vec16<RowT>::kValues);
}

// Index dedup: heavy row sub-task `sub` sums kHeavy of its original rows
// into an fp32 partial of its own, then counts itself done for its row. It
// runs in its row's pass: kRemote for a row an owner on another node gets
// (through a proxy), whose push task adds up the row's sub-tasks; else for a
// row of one of the node's owners, where the last of the row's sub-tasks to
// end adds them up, in sub-task order, and writes the row into the owner's
// lookup_rows. Every lane of the warp.
template <typename RowT, bool kRemote>
__device__ void heavy_sub_warp(const Sizes &sizes, const BwdArgs &args, const OutTables &tables,
                               int64_t sub, int lane) {
  constexpr int kValues = Vec16<RowT>::kValues;
  const BwdPlanView &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int vec = sizes.row_vecs(), dim = vec * kValues, local_size = sizes.topo.local_size;
  const int64_t dedup_row = fanout.sub[2 * sub], begin = fanout.sub[2 * sub + 1],
                row_end = fanout.exp_off[dedup_row + 1];
  const int32_t slot = fanout.slot[dedup_row], row = fanout.pos[slot];
  const int owner = tables.owner_of(row);
  if ((owner / local_size != sizes.topo.node) != kRemote) return;
  const OutTimeline timeline{args.trace, local_size};
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  const int64_t end = min(begin + kHeavy, row_end);
  float *part = args.partial + sub * dim;
  for (int col = lane; col < vec; col += 32) {
    float acc[kValues];
    sum_vec<RowT>(args.grad, fanout.exp_idx, begin, end, sizes.row_bytes, col, acc);
    store_fp32<kValues>(part, col, acc);
  }
  if constexpr (kRemote) {
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
  for (int col = lane; col < vec; col += 32) {
    float acc[kValues];
    sum_fp32<kValues>(sums, num_subs, dim, col, acc);
    *((int4 *)dst + col) = scaled<RowT>(acc, args.scale);
  }
  if (timeline.on() && lane == 0) {
    const bool peer = owner % local_size != sizes.topo.local_rank;
    timeline.task(start, peer ? sizes.row_bytes : 0, peer ? 0 : sizes.row_bytes);
  }
}

// Across nodes, heavy group sub-task `sub`: the fp32 sum, in position order,
// of the scaled gradients of its rows (at most kHeavy positions of
// push_order), into an fp32 partial of its own; then it counts itself done
// for its group. Every lane of the warp.
template <typename RowT>
__device__ void group_sub_warp(const Sizes &sizes, const BwdArgs &args, int64_t sub, int lane) {
  constexpr int kValues = Vec16<RowT>::kValues;
  static_assert(kHeavy % 32 == 0, "a sub-task's rows are spread over the lanes");
  const BwdPlanView &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int row_bytes = sizes.row_bytes, vec = sizes.row_vecs(), dim = vec * kValues;
  const int32_t group = plan.group_of_sub[sub];
  const HeavyGroup span = plan.heavy_groups[group];
  const int64_t begin = span.start + (sub - plan.group_first_sub[group]) * kHeavy,
                end = min(begin + kHeavy, (int64_t)span.end);
  float *part = args.partial + (plan.heavy_subs + sub) * dim;
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
        cnt > kHeavy ? heavy_row_warp<RowT>(sizes, args, dedup_row, cnt, lane, num_subs) : nullptr;
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
// (kLocal), not at all: their last sub-task writes them (heavy_sub_warp).
// Returns how many rows it wrote. Every lane of the warp.
template <typename RowT, bool kLocal>
__device__ int push_rows_warp(const Sizes &sizes, const BwdArgs &args, int64_t dedup_row, char *dst,
                              int num_rows, int lane) {
  const BwdPlanView &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int row_bytes = sizes.row_bytes, vec = sizes.row_vecs(), dim = vec * Vec16<RowT>::kValues;
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
  const bool left = kLocal && cnt > kHeavy;
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
    const float *part =
        heavy_row_warp<RowT>(sizes, args, lane_dedup_row, lane_count, lane, num_subs);
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
__device__ void push_groups_warp(const Sizes &sizes, const BwdArgs &args, const OutTables &tables,
                                 int proxy, int64_t group_lo, int64_t group_hi, int lane) {
  const BwdPlanView &plan = args.plan;
  const int dim = sizes.row_vecs() * Vec16<RowT>::kValues;
  int64_t num_subs = 0;
  const float *part = nullptr;
  float *slot = nullptr;
  if (group_lo + lane < group_hi) {
    const int64_t group = group_lo + lane;
    const HeavyGroup span = plan.heavy_groups[group];
    num_subs = ceil_div<kHeavy>(span.end - span.start);
    part = args.partial + (plan.heavy_subs + plan.group_first_sub[group]) * dim;
    slot = (float *)tables.proxy_row(proxy, plan.dst[plan.push_order[span.start]]);
    SubTaskDone{args.group_done + group}.wait_lane(sizes, num_subs, "sub-tasks of a heavy group");
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

// Remote task `remote_task` (across nodes): a push task of its chunk, of
// heavy groups or of rows, to the proxy in turn from this rank's own one.
__device__ __forceinline__ void remote_push(const Sizes &sizes, const BwdPlanView &plan,
                                            const OutTables &tables, int64_t remote_task,
                                            ProxyPush &push) {
  const int local_size = sizes.topo.local_size, me = sizes.topo.local_rank;
  const RemoteTask place(plan.bwd_task_starts, 2 * plan.max_chunks * local_size, remote_task);
  push.chunk = place.entry / (2 * local_size);
  push.is_group_task = place.entry / local_size % 2 == 0;
  push.proxy = (int)((me + place.entry % local_size) % local_size);
  push.nth = place.nth;
  tables.req_of(push.proxy, push.chunk, push.entry, push.next);
  push.plain_tasks = ceil_div<kTaskRows>(push.entry.heavy - push.entry.start);
}

// A push task of this rank's heavy groups at the proxy (32 at most): their
// sums into their partial slots, then the task's flag. Every lane of the
// warp.
template <typename RowT>
__device__ __forceinline__ void push_groups_task_warp(const Sizes &sizes, int64_t seq,
                                                      const BwdArgs &args, const OutTables &tables,
                                                      const ProxyPush &push, int lane) {
  const int me = sizes.topo.local_rank;
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
  const int proxy = push.proxy;
  const int64_t nth = push.nth, plain_tasks = push.plain_tasks;
  const ReqEntry &entry = push.entry, &next = push.next;
  const int64_t group_lo = entry.groups + nth * kTaskRows;
  if (group_lo >= next.groups) return;
  wait_free_warp(sizes, seq, tables, proxy, push.chunk, lane);
  const unsigned long long start = timeline.on() ? now_ns() : 0;
  const int64_t group_hi = min(group_lo + kTaskRows, (int64_t)next.groups);
  push_groups_warp<RowT>(sizes, args, tables, proxy, group_lo, group_hi, lane);
  push_flag_warp(seq, tables, proxy, entry.flags + plain_tasks + nth, lane);
  if (timeline.on() && lane == 0) {
    // an fp32 partial row per group
    const unsigned long long bytes =
        (unsigned long long)(group_hi - group_lo) * 4 * sizes.row_vecs() * Vec16<RowT>::kValues;
    timeline.task(start, proxy != me ? bytes : 0, proxy != me ? 0 : bytes);
  }
}

// A push task of this rank's rows at the proxy (32 positions of its push
// order there): their gradients into their rows of proxy_rows, then the
// task's flag. Every lane of the warp.
template <typename RowT>
__device__ __forceinline__ void push_rows_task_warp(const Sizes &sizes, int64_t seq,
                                                    const BwdArgs &args, const OutTables &tables,
                                                    const ProxyPush &push, int lane) {
  const BwdPlanView &plan = args.plan;
  const int me = sizes.topo.local_rank, row_bytes = sizes.row_bytes;
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
  const int proxy = push.proxy;
  const int64_t nth = push.nth, plain_tasks = push.plain_tasks;
  const ReqEntry &entry = push.entry;
  if (nth >= plain_tasks) return;
  wait_free_warp(sizes, seq, tables, proxy, push.chunk, lane);
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
  push_rows_warp<RowT, false>(sizes, args, dedup_row, dst, num_rows, lane);
  push_flag_warp(seq, tables, proxy, entry.flags + nth, lane);
  if (timeline.on() && lane == 0) {
    const unsigned long long bytes = (unsigned long long)num_rows * row_bytes;
    timeline.task(start, proxy != me ? bytes : 0, proxy != me ? 0 : bytes);
  }
}

// Row task `index` of the node's owners: its rows into the owner's
// lookup_rows, but those of heavy rows (their last sub-task writes them).
// Every lane of the warp.
template <typename RowT>
__device__ __forceinline__ void local_push_warp(const Sizes &sizes, const BwdArgs &args,
                                                const OutTables &tables,
                                                const LocalRows &local_rows, int64_t index,
                                                int lane) {
  const BwdPlanView &plan = args.plan;
  const int me = sizes.topo.local_rank, row_bytes = sizes.row_bytes;
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
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
  const int wrote = push_rows_warp<RowT, true>(sizes, args, dedup_row, dst, num_rows, lane);
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
// is 32 node-unique keys of one chunk of its stream. The tasks go chunk after
// chunk (reduce_task_starts), the order the requesters push them in. The
// task's keys' reducer lists are consecutive. The warp waits for the push
// task of each entry (the lanes over the entries, 32 apart; flags tagged with
// the backward's out_seq). Then it sums, for each key of several entries,
// their rows in fp32 in list order (a key's occurrences by requester and
// position; a heavy key's requesters' partials in requester order) and rounds
// the sum once into the key's row. Then it counts the task's keys ready in
// the chunk's counter, where the issuer looks. A key of one row is not
// touched: the NIC reads it where its requester's NVLink stores left it
// (released by the task flag, acquired here, released again by the counter).
template <typename RowT>
struct ProxyReducer {
  // Every thread of a reducer CTA, in backward seq: the CTA's warps take the
  // tasks.
  __device__ static void run(ncclWindow_t window, const Sizes &sizes, int64_t seq,
                             const BwdArgs &args, int lane);
};

template <typename RowT>
__device__ void ProxyReducer<RowT>::run(ncclWindow_t window, const Sizes &sizes, int64_t seq,
                                        const BwdArgs &args, int lane) {
  const BwdPlanView &plan = args.plan;
  const int vec = sizes.row_vecs();
  // this rank's copy of the window: its proxy_rows and push flags
  const WindowLayout layout = NodeWindow(window, sizes.topo.local_rank).layout(sizes);
  const uint64_t *flags = layout.rows.push_flags();
  const int64_t total = plan.reduce_task_starts[plan.max_chunks];
  const TaskQueue queue{layout.ctrl.task(kQueueReduce)};
  for (int64_t task = queue.next_warp(); task < total; task = queue.next_warp()) {
    // The task: keys [key_lo + sub * 32, ...) of its chunk, at most 32.
    const RemoteTask place(plan.reduce_task_starts, plan.max_chunks, task);
    const Chunk chunk = plan.chunks[place.entry];
    const int64_t uniq_lo = chunk.key_lo + place.nth * kTaskRows;
    const int num_keys = (int)min((int64_t)kTaskRows, (int64_t)chunk.key_lo + chunk.keys - uniq_lo);
    if (num_keys <= 0) continue;
    // The lane's key (a lane per key): its reducer list [begin, end) (heavy:
    // the first entry is the heavy key's own row) and its row in proxy_rows
    // (where its chunk lies). Then the wait for the push task of each entry
    // of the task's lists.
    const int64_t at = (int64_t)chunk.start % sizes.num_proxy_rows;
    int64_t begin = 0, end = 0;
    bool heavy = false;
    char *row = nullptr;
    if (lane < num_keys) {
      begin = plan.reducer_start[uniq_lo + lane];
      end = plan.reducer_start[uniq_lo + lane + 1];
      heavy = plan.reducer_entries[begin].own_row();
      row = layout.rows.proxy_row(at + uniq_lo + lane - chunk.key_lo);
    }
    const int64_t ent_end = __shfl_sync(kFull, end, num_keys - 1);
    Spin spin(sizes.timeout_ns);
    for (int64_t i = __shfl_sync(kFull, begin, 0) + lane; i < ent_end; i += 32) {
      const ReducerEntry entry = plan.reducer_entries[i];
      if (!entry.pushed()) continue;  // kEntryNoPush, kEntryOwnRow: no push task
      while ((int64_t)ld_acquire_sys(&flags[entry.flag]) < seq)
        spin.pause("pushed gradient rows", entry.flag, uniq_lo, sizes.topo.rank);
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
          sum_entries<RowT, true>(layout.rows, plan.reducer_entries, lane_begin + 1, lane_end, col,
                                  col < vec, lane, acc);
        else
          sum_entries<RowT, false>(layout.rows, plan.reducer_entries, lane_begin, lane_end, col,
                                   col < vec, lane, acc);
        if (col < vec) *((int4 *)dst + col) = Vec16<RowT>::pack(acc);
      }
    }
    __syncwarp();
    // The task's keys are reduced: counted ready in the chunk's counter.
    if (lane == 0) red_release_sys((uint64_t *)&args.ready[place.entry], (uint64_t)num_keys);
  }
}

}  // namespace bwd

// The backward's GIN warps (across nodes, CTA 0, which skip the entry
// barrier): the proxy's issuer, and when tracing the warp that watches this
// rank's landings.

namespace bwd {

// A chunk's ready word (args.ready[chunk]): the keys of the chunk the
// reducers have counted ready, and the bit kPostedBit, which the issuer sets
// once the chunk's puts are posted (only that lane reads it).
struct ChunkReady {
  static constexpr int kPostedBit = 62;
  uint64_t word;
  __device__ int64_t keys() const { return (int64_t)(word & ((1ull << kPostedBit) - 1)); }
  __device__ bool posted() const { return (word >> kPostedBit) != 0; }
  __device__ uint64_t with_posted() const { return word | (1ull << kPostedBit); }
};

// The proxy's issuer in the backward (across nodes; the last warp of CTA 0,
// whose lanes take the owner nodes in turn, so one QP each). As owner, each
// lane first lets the rail proxies on its nodes put this backward's rows
// into its lookup_rows, if the previous kernel (a backward too) could not:
// its lookup_grad was still live. Then, for each owner node once the owner
// there let it (bwd_ready), it puts that node's chunks, each as soon as the
// reducers counted all its keys ready, in whatever order they complete. It
// polls the owner nodes in turn and waits for no single one; its place in
// each one's segments is in this rank's put_at words. A chunk takes one
// system fence, then its segments from proxy_rows to their rows of the
// owner's lookup_rows, each adding its rows to the owner's bwd_rows signal.
// When the stream wraps the ring, each also counts on the put counter of its
// chunk (chunk % counters) once the NIC has read it, and a chunk is put only
// within `counters` chunks of the first not yet read, so no two chunks in
// flight share a counter. Lane 0 then marks chunks read in order, once their
// puts are posted and their counter reaches the puts posted on it, and frees
// the rows of the stream they hold (slot_free). Once the NIC has read
// proxy_rows (no wrap: the warp's flush, as in the forward; a wrap: every
// chunk read), the lanes grant each rail owner the ring's rows of the next
// kernel's stream, and the warp flushes the credits' context.
struct ProxyIssuer {
  // Every lane of the issuer warp, in backward seq.
  __device__ static void run(ncclWindow_t window, const ncclDevComm &dev, const Sizes &sizes,
                             int64_t seq, const BwdArgs &args);
};

__device__ void ProxyIssuer::run(ncclWindow_t window, const ncclDevComm &dev, const Sizes &sizes,
                                 int64_t seq, const BwdArgs &args) {
  __shared__ int64_t read_chunks;  // lane 0's chunks read, for the other lanes
  const BwdPlanView &plan = args.plan;
  const int lane = threadIdx.x & 31, num_nodes = sizes.topo.num_nodes, node = sizes.topo.node;
  const int64_t ring = sizes.num_proxy_rows, counters = sizes.counters();
  const NodeWindow node_window(window, sizes.topo.local_rank);
  const WindowLayout layout = node_window.layout(sizes);
  ncclGin rows(dev, kCtxBwd), cred(dev, kCtxCredit);
  const ncclTeam rail = ncclTeamRail(dev);
  const Credits credits(cred, rail);
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
  for (int owner_node = lane; owner_node < num_nodes; owner_node += 32)
    if (owner_node != node)
      credits.raise(owner_node, sig_bwd_ready(num_nodes, node), (uint64_t)seq,
                    layout.ctrl.bwd_ready_sent(owner_node));
  Spin spin(sizes.timeout_ns);
  // The ready counters were zeroed before this rank published its out_seq.
  if (lane == 0)
    while ((int64_t)ld_acquire_gpu(layout.ctrl.out_started()) < seq)
      spin.pause("entry of this rank", 0, seq, sizes.topo.rank);
  const int64_t count = *plan.num_chunks;
  const bool wraps = count > 0 && plan.chunks[count - 1].end > ring;
  for (int owner_node = lane; owner_node < num_nodes; owner_node += 32)
    *layout.ctrl.put_at(owner_node) = plan.bwd_puts(owner_node);
  volatile int64_t *lane_read = &read_chunks;
  if (lane == 0) *lane_read = 0;
  __syncwarp();
  // rows of proxy_rows into the rail owners' lookup_rows
  const RowPut put{rows,
                   rail,
                   node_window,
                   layout.rows.lookup_rows(),
                   layout.rows.proxy_rows(),
                   sizes.row_bytes,
                   sig_rows(node)};
  int64_t read = 0;   // lane 0: chunks the NIC has read, in order
  bool first = true;  // the lane has put nothing yet
  for (;;) {
    bool moved = false, left = false;  // segments of the lane's owner nodes are left
    const int64_t limit = wraps ? *lane_read + counters : count;  // chunks it may put
    for (int owner_node = lane; owner_node < num_nodes; owner_node += 32) {
      if (owner_node == node) continue;
      int64_t *at = layout.ctrl.put_at(owner_node);
      int64_t done = *at;  // segments before it are put
      const int64_t end = plan.bwd_puts(owner_node + 1);
      if (done < end && (int64_t)credits.read(sig_bwd_ready(num_nodes, owner_node)) >= seq) {
        // each chunk the reducers counted ready; a chunk's segments are
        // consecutive
        for (int64_t seg_start = done; seg_start < end;) {
          const int32_t chunk = plan.put_list[seg_start].chunk;
          if (chunk >= limit) break;
          int64_t seg_end = seg_start, keys = 0;
          for (; seg_end < end && plan.put_list[seg_end].chunk == chunk; ++seg_end)
            keys += plan.put_list[seg_end].rows;
          uint64_t *ready = (uint64_t *)&args.ready[chunk];
          const ChunkReady state{ld_acquire_sys(ready)};
          if (!state.posted() && state.keys() >= keys) {
            if (first) timeline.stamp_earliest(kTraceFirstPut);
            first = false;
            put.fence();
            for (int64_t i = seg_start; i < seg_end; ++i) {
              const PutSeg seg = plan.put_list[i];
              put.put(owner_node, seg.dst, seg.src, seg.rows, wraps, (int)(chunk % counters));
            }
            if (wraps) *layout.ctrl.posted(chunk % counters) += (uint64_t)(seg_end - seg_start);
            if (timeline.on()) timeline.posted((unsigned long long)keys);
            *ready = state.with_posted();
            moved = true;
          }
          if (seg_start == done && ChunkReady{*ready}.posted()) done = seg_end;
          seg_start = seg_end;
        }
      }
      *at = done;
      left = left || done < end;
    }
    __syncwarp();
    // Lane 0, when the stream wraps: the chunks the NIC has read, in order
    // (their puts posted, and their counter at the puts posted on it), and
    // the rows of the stream they free.
    if (wraps && lane == 0) {
      while (read < count) {
        if (!ChunkReady{((volatile uint64_t *)args.ready)[read]}.posted()) break;
        const int64_t counter = read % counters;
        if (rows.readCounter((int)counter) < *layout.ctrl.posted(counter)) break;
        ++read;
        const int64_t free_end = (read < count ? (int64_t)plan.chunks[read].start
                                               : (int64_t)plan.chunks[count - 1].end) +
                                 ring;
        st_release_sys(layout.ctrl.slot_free(), tag_of(seq, (uint64_t)free_end));
        moved = true;
      }
      *lane_read = read;
    }
    __syncwarp();
    const bool all_read = !wraps || *lane_read == count;
    if (!__any_sync(kFull, left) && all_read) break;
    if (!__any_sync(kFull, moved))
      spin.pause("reduced chunks", (long long)read, lane, sizes.topo.rank);
  }
  if (!first) timeline.stamp_latest(kTraceLastPut);
  // The NIC has read every put out of proxy_rows: no wrap, the warp's flush
  // (its puts count on no counter); a wrap, every chunk read.
  if (!wraps) {
    __syncwarp();
    rows.flush(ncclCoopWarp{});
  }
  // the puts have completed
  if (timeline.on() && __any_sync(kFull, !first) && lane == 0)
    timeline.stamp_latest(kTracePutsDone);
  __syncwarp();
  for (int owner_node = lane; owner_node < num_nodes; owner_node += 32)
    if (owner_node != node)
      credits.raise(owner_node, sig_grant(node), tag_of(seq + 1, (uint64_t)ring),
                    layout.ctrl.grant_sent(owner_node));
  __syncwarp();
  credits.flush_warp();
}

// The gradient rows the proxies on a lane's nodes (l, l + 32, ...) put into
// this rank's lookup_rows: how many have landed during this backward (each
// node's bwd_rows signal against the running total, bwd_tot), and all of
// them.
__device__ __forceinline__ void lane_landed_rows(ncclWindow_t window, const Sizes &sizes,
                                                 const BwdArgs &args, const ncclGin &rows, int lane,
                                                 uint64_t &got, uint64_t &need) {
  const CtrlLayout ctrl = NodeWindow(window, sizes.topo.local_rank).layout(sizes).ctrl;
  got = need = 0;
  for (int peer_node = lane; peer_node < sizes.topo.num_nodes; peer_node += 32) {
    if (peer_node == sizes.topo.node) continue;
    const uint64_t rows_of_node = (uint64_t)args.plan.uniq_in[peer_node];
    got += min(rows.readSignal(sig_rows(peer_node)) - *ctrl.bwd_tot(peer_node), rows_of_node);
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
  // Every lane of the watcher warp (arrive0: the arrive word before this
  // kernel).
  __device__ static void run(ncclWindow_t window, const ncclDevComm &dev, const Sizes &sizes,
                             int64_t arrive0, const BwdArgs &args);
  // The backward's last CTA when it is CTA 0 (the watcher stopped before
  // every row had landed), every lane of its warp 0: the rows of the lane's
  // nodes the watcher had not seen, binned into the kernel's row as they land
  // (the CTA's shared copy is folded already), and when the last one landed.
  __device__ static void rest(ncclWindow_t window, const ncclDevComm &dev, const Sizes &sizes,
                              const BwdArgs &args, int lane);
};

__device__ void LandingWatcher::run(ncclWindow_t window, const ncclDevComm &dev, const Sizes &sizes,
                                    int64_t arrive0, const BwdArgs &args) {
  const int lane = threadIdx.x & 31;
  const CtrlLayout ctrl = NodeWindow(window, sizes.topo.local_rank).layout(sizes).ctrl;
  ncclGin rows(dev, kCtxBwd);
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
  const uint64_t others = (uint64_t)(arrive0 + gridDim.x - 1);
  uint64_t seen = 0, got = 0, need = 0;
  lane_landed_rows(window, sizes, args, rows, lane, got, need);
  Spin spin(sizes.timeout_ns);
  for (;;) {
    if (seen < need) {
      lane_landed_rows(window, sizes, args, rows, lane, got, need);
      if (got > seen) {
        timeline.landed(got - seen, seen == 0);
        seen = got;
        if (seen == need) timeline.stamp_latest(kTraceLanded);
      }
    }
    if (__all_sync(kFull, seen == need)) break;
    bool rest = false;
    if (lane == 0) rest = ld_acquire_gpu(ctrl.arrive()) >= others;
    if (__shfl_sync(kFull, rest, 0)) break;
    spin.pause("gradient rows of the proxy nodes", lane, (long long)need, sizes.topo.rank);
  }
  timeline.shared()[kSmLanded + lane] = seen;
}

__device__ void LandingWatcher::rest(ncclWindow_t window, const ncclDevComm &dev,
                                     const Sizes &sizes, const BwdArgs &args, int lane) {
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
  ncclGin rows(dev, kCtxBwd);
  uint64_t seen = timeline.shared()[kSmLanded + lane], got = 0, need = 0;
  lane_landed_rows(window, sizes, args, rows, lane, got, need);
  if (seen >= need) return;
  Spin spin(sizes.timeout_ns);
  while (seen < need) {
    lane_landed_rows(window, sizes, args, rows, lane, got, need);
    if (got == seen) {
      spin.pause("gradient rows of the proxy nodes", lane, (long long)need, sizes.topo.rank);
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
//     tagged with the out_seq; the occurrences of a heavy key (more than
//     heavy_at on the node) go as one fp32 sum per requester (heavy group)
//     into its partial slot, 32 groups per flag.
// The proxy's reducer CTAs (across nodes, the last ones) wait for the flags
// of each key's reducer-list entries, sum the keys of several in fp32 in list
// order and round once into the key's row, and count each chunk's keys
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
__device__ __forceinline__ void wait_rail_rows_warp(ncclWindow_t window, const ncclDevComm &dev,
                                                    const Sizes &sizes, const BwdArgs &args,
                                                    const CtrlLayout &ctrl) {
  const int lane = threadIdx.x & 31;
  // Tracing: CTA 0 is the last CTA when its watcher stopped before every
  // row had landed; another one comes after it saw them all
  // (LandingWatcher).
  if (OutTimeline{args.trace, sizes.topo.local_size}.on() && blockIdx.x == 0)
    LandingWatcher::rest(window, dev, sizes, args, lane);
  ncclGin rows(dev, kCtxBwd);
  for (int peer_node = lane; peer_node < sizes.topo.num_nodes; peer_node += 32) {
    if (peer_node == sizes.topo.node) continue;
    const uint64_t want = *ctrl.bwd_tot(peer_node) + (uint64_t)args.plan.uniq_in[peer_node];
    Spin spin(sizes.timeout_ns);
    while (rows.readSignal(sig_rows(peer_node)) < want)
      spin.pause("gradient rows of a proxy node", peer_node, (long long)want, sizes.topo.rank);
    *ctrl.bwd_tot(peer_node) = want;
  }
}

// The end of backward seq for this rank, in the last CTA to arrive: it sets
// the task queues back to 0, tells every owner of the node that this rank's
// gradient rows are in (its flag pushed[local_rank] there) and waits until
// every requester of the node has told it the same; across nodes warp 0 then
// waits until the gradient rows of every rail proxy have landed. This rank's
// lookup_grad is then complete. Every thread of the last CTA.
__device__ __forceinline__ void exit(ncclWindow_t window, const ncclDevComm &dev,
                                     const Sizes &sizes, int64_t seq, const BwdArgs &args) {
  const int tid = threadIdx.x;
  const OutTimeline timeline{args.trace, sizes.topo.local_size};
  const NodeWindow node(window, sizes.topo.local_rank);
  const CtrlLayout ctrl = node.layout(sizes).ctrl;
  const PeerFlags pushed{sizes, node, ctrl.pushed(), seq};
  if (tid == 0) {
    timeline.stamp_latest(kTraceArrived);
    TaskQueue::reset_all(ctrl);
    for (int peer = 0; peer < sizes.topo.local_size; ++peer) pushed.tell(peer);
  }
  for (int peer = tid; peer < sizes.topo.local_size; peer += blockDim.x)
    pushed.wait(peer, "gradient rows pushed");
  if (sizes.topo.num_nodes > 1 && (tid >> 5) == 0)
    wait_rail_rows_warp(window, dev, sizes, args, ctrl);
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
// backward for this rank (bwd::exit). It takes what fwd_kernel takes, with
// its call (args).
template <typename RowT>
__global__ void __launch_bounds__(kThreads, 1)
    bwd_kernel(ncclWindow_t window, ncclDevComm dev, Sizes sizes, int64_t seq, int64_t arrive0,
               BwdArgs args) {
  const BwdPlanView &plan = args.plan;
  const Fanout &fanout = plan.fanout;
  const int local_size = sizes.topo.local_size, num_nodes = sizes.topo.num_nodes,
            node = sizes.topo.node, me = sizes.topo.local_rank;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const OutTimeline timeline{args.trace, local_size};
  timeline.begin_cta();
  const int gin = num_nodes == 1 || blockIdx.x != 0 ? 0 : timeline.on() ? 64 : 32;
  const int workers = kThreads - gin;
  if (tid >= workers) {
    if (warp == kWarps - 1)
      bwd::ProxyIssuer::run(window, dev, sizes, seq, args);
    else if constexpr (kTrace)
      bwd::LandingWatcher::run(window, dev, sizes, arrive0, args);
  } else if (num_nodes > 1 && blockIdx.x >= gridDim.x - args.reducers) {
    pass_entry_barrier(window, sizes, seq, timeline, kThreads);
    bwd::ProxyReducer<RowT>::run(window, sizes, seq, args, lane);
  } else {
    const OutTables tables(window, sizes, args.plan);
    // free_seen starts at 0; the entry barrier's sync publishes it
    for (int proxy = tid; proxy < local_size; proxy += workers) *tables.free_seen(proxy) = 0;
    int64_t groups = 0;  // this rank's heavy groups, at all its proxies
    if (num_nodes > 1) groups = plan.push_entry(local_size - 1, plan.max_chunks)->groups;
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
    pass_entry_barrier(window, sizes, seq, timeline, workers);
    const int64_t *node_offset_per_bucket = plan.offset_per_bucket + node * local_size;
    // Every heavy row's sub-task has a slot before the remote tasks (across
    // nodes), which those of rows to other nodes take, and one among the row
    // tasks, which those of the node's rows take.
    const int64_t heavy_subs = fanout.exp_off != nullptr ? fanout.count[1] : 0,
                  early_subs = num_nodes > 1 ? heavy_subs : 0;
    const int64_t subs = early_subs + plan.group_subs,
                  remote =
                      subs +
                      (num_nodes > 1 ? plan.bwd_task_starts[2 * plan.max_chunks * local_size] : 0);
    const LocalRows local_rows{node_offset_per_bucket, local_size, me};
    const int64_t rows = local_rows.count(), total = remote + rows + heavy_subs;
    const TaskQueue queue{tables.layout().ctrl.task(kQueueWork)};
    for (int64_t task = queue.next_warp(); task < total; task = queue.next_warp()) {
      if (task < early_subs) {
        bwd::heavy_sub_warp<RowT, true>(sizes, args, tables, task, lane);
        continue;
      }
      if (task < subs) {
        bwd::group_sub_warp<RowT>(sizes, args, task - early_subs, lane);
        continue;
      }
      if (task < remote) {
        bwd::ProxyPush push;
        bwd::remote_push(sizes, plan, tables, task - subs, push);
        if (push.is_group_task)
          bwd::push_groups_task_warp<RowT>(sizes, seq, args, tables, push, lane);
        else
          bwd::push_rows_task_warp<RowT>(sizes, seq, args, tables, push, lane);
        continue;
      }
      int64_t index;
      if (Interleave{rows, heavy_subs}.is_sub(task - remote, index)) {
        bwd::heavy_sub_warp<RowT, false>(sizes, args, tables, index, lane);
        continue;
      }
      bwd::local_push_warp<RowT>(sizes, args, tables, local_rows, index, lane);
    }
  }
  if (!arrive_cta(window, sizes, arrive0, timeline)) return;  // the last CTA goes on
  bwd::exit(window, dev, sizes, seq, args);
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
