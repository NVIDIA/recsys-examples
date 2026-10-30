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
// The call's tables as a requester's warps read them: their copy in shared
// memory (OutTables), and this rank's entries of each proxy's tables.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/windows.h"
#include "hier_output_dist/device/ctrl_words.cuh"
#include "hier_output_dist/device/search.cuh"
#include "hier_output_dist/device/sym_window.cuh"
#include "hier_output_dist/output/args.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// What a requester's warps read of the call, in shared memory: the block
// starts, every same-node lookup_rows and (across nodes) proxy_rows, and per
// same-node proxy its rounds and the round 0 entries of two tables: its round
// table (where the rows of each owner node's keys sit in its proxy_rows) and
// this rank's push table there (ReqEntry; the next entry holds the ends).
// Later rounds' entries are read from the call's tables (meta). The methods
// take the kernel's argument, whose call this is.
struct OutTables {
  int64_t block_start[kMaxRanks + 1];
  char *lookup_rows[kMaxLocalRanks];
  char *proxy_rows[kMaxLocalRanks];
  const uint64_t *landed_rows[kMaxLocalRanks];  // each proxy's relay words
  uint64_t *push_flags[kMaxLocalRanks];         // each proxy's push-task flags
  uint64_t *done[kMaxLocalRanks];               // [2] this rank's done words at each proxy
  const uint64_t *half_free[kMaxLocalRanks];    // each proxy's half_free word
  uint64_t half_seen[kMaxLocalRanks];           // the most of it this CTA's warps saw
  int32_t rounds[kMaxLocalRanks];               // each proxy's rounds
  int max_rounds;                               // the most of them
  bool staged;  // index dedup's heavy rows at a proxy of more than one round
  RoundPiece pieces[kMaxLocalRanks * kMaxNodes];  // [proxy][owner_node]
  ReqEntry req[kMaxLocalRanks][2];                // [proxy][round 0, its end]

  // The threads of the CTA that read them: the copy. There is no barrier
  // here: the group's next sync (the entry barrier's) publishes it.
  __device__ __forceinline__ void load(const OutArgs &args) {
    const int local_size = args.topo.local_size, num_nodes = args.topo.num_nodes,
              world_size = args.topo.world_size, tid = threadIdx.x;
    for (int i = tid; i <= world_size; i += blockDim.x) block_start[i] = args.plan.block_start[i];
    if (tid < local_size) {
      lookup_rows[tid] = SymRegion<kLookupRows>(args.win).peer(tid);
      if (num_nodes > 1) {
        const CtrlWords ctrl = CtrlWords::of(args, tid);
        proxy_rows[tid] = SymRegion<kProxyRows>(args.win).peer(tid);
        landed_rows[tid] = ctrl.landed_rows();
        push_flags[tid] = SymRegion<kPushFlags, uint64_t>(args.win).peer(tid);
        done[tid] = ctrl.done(args.topo.local_rank);
        half_free[tid] = ctrl.half_free();
        half_seen[tid] = 0;
        rounds[tid] = (int32_t)args.plan.meta[args.meta_at.proxy_rounds + tid];
        const ReqEntry *push_table = (const ReqEntry *)(args.plan.meta + args.meta_at.push_tables) +
                                     tid * (args.win.rounds_cap + 1);
        req[tid][0] = push_table[0];
        req[tid][1] = push_table[1];
      }
    }
    if (tid == 0) {
      int most_rounds = 1;
      for (int proxy = 0; proxy < local_size && num_nodes > 1; ++proxy)
        most_rounds = max(most_rounds, (int)args.plan.meta[args.meta_at.proxy_rounds + proxy]);
      max_rounds = most_rounds;
      staged =
          args.plan.fanout.exp_off != nullptr && most_rounds > 1 && args.plan.fanout.count[1] > 0;
    }
    if (num_nodes > 1) {
      const RoundPiece *proxy_round_tables =
          (const RoundPiece *)(args.plan.meta + args.meta_at.proxy_round_tables);
      for (int i = tid; i < local_size * num_nodes; i += blockDim.x)
        pieces[i] = proxy_round_tables[(int64_t)(i / num_nodes) * args.win.rounds_cap * num_nodes +
                                       i % num_nodes];
    }
  }
  // This rank's push order at a proxy in a round (positions of push_order)
  // and the next round's entry, which holds the ends.
  __device__ __forceinline__ void req_of(const OutArgs &args, int proxy, int rnd, ReqEntry &entry,
                                         ReqEntry &next) const {
    if (rnd == 0) {
      entry = req[proxy][0];
      next = req[proxy][1];
      return;
    }
    const ReqEntry *push_table = (const ReqEntry *)(args.plan.meta + args.meta_at.push_tables) +
                                 proxy * (args.win.rounds_cap + 1) + rnd;
    entry = push_table[0];
    next = push_table[1];
  }
  // Where an owner node's node-unique keys of a round sit in a proxy's
  // proxy_rows.
  __device__ __forceinline__ RoundPiece piece_of(const OutArgs &args, int proxy, int rnd,
                                                 int owner_node) const {
    if (rnd == 0) return pieces[proxy * args.topo.num_nodes + owner_node];
    return ((const RoundPiece *)(args.plan.meta + args.meta_at.proxy_round_tables))
        [((int64_t)proxy * args.win.rounds_cap + rnd) * args.topo.num_nodes + owner_node];
  }
  // The owner of one of this rank's rows (its bucketized rows by owner).
  __device__ __forceinline__ int owner_of(const OutArgs &args, int64_t row) const {
    return seg_search(block_start, args.topo.world_size, row);
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
