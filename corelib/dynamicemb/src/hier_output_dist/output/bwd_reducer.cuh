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
// tasks (bwd::ProxyReducer).
#pragma once

#include <cstdint>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/gin_signals.h"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/node_window.cuh"
#include "hier_output_dist/device/spin.cuh"
#include "hier_output_dist/device/warp.cuh"
#include "hier_output_dist/output/args.cuh"
#include "hier_output_dist/output/bwd_sums.cuh"
#include "hier_output_dist/output/schedule.cuh"

namespace dyn_emb {
namespace hier_output_dist {

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
  __device__ static void run(const OutArgs &args, int lane);
};

template <typename RowT>
__device__ void ProxyReducer<RowT>::run(const OutArgs &args, int lane) {
  const OutPlan &plan = args.plan;
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
      red_release_sys((uint64_t *)&plan.ready[piece.chunk + chunk], (uint64_t)num_keys);
  }
}

}  // namespace bwd

}  // namespace hier_output_dist
}  // namespace dyn_emb
