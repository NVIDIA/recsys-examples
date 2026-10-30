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
// The proxy's reducers in the backward (across nodes, the last CTAs): their
// shared tables (ReduceTables) and their tasks (bwd::ProxyReducer).
#pragma once

#include <cstdint>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/gin_signals.h"
#include "hier_output_dist/common/meta.h"
#include "hier_output_dist/common/windows.h"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/spin.cuh"
#include "hier_output_dist/device/sym_window.cuh"
#include "hier_output_dist/device/warp.cuh"
#include "hier_output_dist/output/args.cuh"
#include "hier_output_dist/output/bwd_sums.cuh"
#include "hier_output_dist/output/schedule.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The proxy's reducers' shared tables: round 0 of its round table (where each
// owner node's keys sit in proxy_rows, and their first IB chunk), and where
// each owner node's keys start in the reducer list (uniq_off).
struct ReduceTables {
  RoundPiece pieces[kMaxNodes];
  int32_t uniq_off[kMaxNodes + 1];

  // One thread: the copy (round_table: the proxy's round table). The CTA
  // syncs after.
  __device__ __forceinline__ void load(const OutArgs &args, const RoundPiece *round_table,
                                       int num_nodes) {
    const int32_t *uniq_off_at = (const int32_t *)(args.plan.meta + args.meta_at.uniq_off);
    for (int owner_node = 0; owner_node < num_nodes; ++owner_node) {
      pieces[owner_node] = round_table[owner_node];
      uniq_off[owner_node] = uniq_off_at[owner_node];
    }
    uniq_off[num_nodes] = uniq_off_at[num_nodes];
  }
  // Where an owner node's node-unique keys of round rnd sit in the proxy's
  // proxy_rows, and their first IB chunk.
  __device__ __forceinline__ RoundPiece piece_of(const RoundPiece *round_table, int rnd,
                                                 int owner_node, int num_nodes) const {
    return rnd == 0 ? pieces[owner_node] : round_table[(int64_t)rnd * num_nodes + owner_node];
  }
};

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
  // Every thread of a reducer CTA: the shared tables into `red`, then the
  // CTA's warps take the tasks.
  __device__ static void run(const OutArgs &args, ReduceTables &red, int lane);
};

template <typename RowT>
__device__ void ProxyReducer<RowT>::run(const OutArgs &args, ReduceTables &red, int lane) {
  const OutPlan &plan = args.plan;
  const int num_nodes = args.topo.num_nodes, node = args.topo.node, tid = threadIdx.x,
            row_bytes = args.row_bytes, vec = args.vec;
  const int64_t chunk_tasks = args.chunk_rows / kTaskRows;  // tasks per chunk
  const RoundPiece *round_table = (const RoundPiece *)(plan.meta + args.meta_at.round_table);
  const int64_t *reduce_task_starts = plan.meta + args.meta_at.reduce_task_starts;
  if (tid == 0) red.load(args, round_table, num_nodes);
  __syncthreads();
  char *proxy_rows = SymRegion<kProxyRows>(args.win).peer(args.topo.local_rank);
  const uint64_t *flags = SymRegion<kPushFlags, uint64_t>(args.win).peer(args.topo.local_rank);
  const int64_t flags_per_half = half_flags(args.win.num_proxy_rows, args.topo.local_size);
  const int64_t total = reduce_task_starts[plan.meta[args.meta_at.misc + kMiscRounds]];
  RoundCursor cursor(reduce_task_starts, 1);  // the round of this warp's tasks
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
    const RoundPiece piece = red.piece_of(round_table, rnd, owner_node, num_nodes);
    const int64_t uniq_lo = piece.uniq_lo + chunk * args.chunk_rows + sub * kTaskRows;
    const int64_t chunk_end =
        min((int64_t)piece.uniq_hi, piece.uniq_lo + (chunk + 1) * args.chunk_rows);
    const int num_keys = (int)min((int64_t)kTaskRows, chunk_end - uniq_lo);
    if (num_keys <= 0) continue;
    // The lane's key (a lane per key): its reducer list [begin, end) (heavy:
    // the first entry is the heavy key's own row) and its row in proxy_rows.
    // Then the wait for the push task of each entry of the task's lists.
    const uint64_t tag = grant_of(args.ch.seq, rnd);
    const uint64_t *half_flags_at = flags + (rnd & 1) * flags_per_half;
    const int64_t uniq_global = red.uniq_off[owner_node] + uniq_lo;
    int64_t begin = 0, end = 0;
    bool heavy = false;
    char *row = nullptr;
    if (lane < num_keys) {
      begin = plan.reducer_start[uniq_global + lane];
      end = plan.reducer_start[uniq_global + lane + 1];
      heavy = (int32_t)(plan.reducer_entries[begin] >> 32) == kEntryOwnRow;
      row = proxy_rows + (int64_t)(piece.off + uniq_lo + lane - piece.uniq_lo) * row_bytes;
    }
    const int64_t ent_end = __shfl_sync(kFull, end, num_keys - 1);
    Spin spin(args.timeout_ns);
    for (int64_t i = __shfl_sync(kFull, begin, 0) + lane; i < ent_end; i += 32) {
      const int32_t flag = (int32_t)(plan.reducer_entries[i] >> 32);
      if (flag < 0) continue;  // kEntryNoPush, kEntryOwnRow: no push task
      while (ld_acquire_sys(&half_flags_at[flag]) < tag)
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
          sum_entries<RowT, true>(proxy_rows, plan.reducer_entries, lane_begin + 1, lane_end,
                                  row_bytes, col, col < vec, lane, acc);
        else
          sum_entries<RowT, false>(proxy_rows, plan.reducer_entries, lane_begin, lane_end,
                                   row_bytes, col, col < vec, lane, acc);
        if (col < vec) *((int4 *)dst + col) = Vec16<RowT>::pack(acc);
      }
    }
    __syncwarp();
    // The task's keys are reduced: counted ready in the chunk's counter.
    if (lane == 0)
      red_release_sys((uint64_t *)&plan.ready[piece.chunk + chunk], (uint64_t)num_keys);
  }
}

}  // namespace bwd

}  // namespace hier_output_dist
}  // namespace dyn_emb
