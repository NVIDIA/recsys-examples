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
// Formats of the tables the kernels of a call share (host and device): index
// dedup's fan-out, the proxy's round table and push orders, its reducer
// entries and heavy groups, and put lists.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/constants.h"

namespace dyn_emb {
namespace hier_output_dist {

// Index dedup's fan-out of the requester's deduplicated rows to its original
// rows (part 1): a row's are exp_idx[exp_off[row] .. exp_off[row + 1]), in
// increasing order. A row with more than kHeavy is heavy, and split into
// sub-tasks of kHeavy original rows.
struct Fanout {
  const int32_t *exp_off;  // [rows + 1], nullptr without index dedup
  const int32_t *exp_idx;  // [original rows]
  const int32_t *slot;     // [rows] heavy slot of a row, -1 if it is not heavy
  const int64_t *sub;      // [2 * subs] (row, first entry) of each heavy sub-task,
                           // a row's sub-tasks consecutive and in entry order
  const int64_t *count;    // [2] heavy rows (slots), heavy sub-tasks
  const int32_t *pos;      // [slots] the bucketized row of each heavy row
  const int32_t *first;    // [slots] its first sub-task
};

// ── the proxy's round table and push orders (num_nodes > 1) ──────────────────
// The proxy lays its rows out in proxy_rows by rounds: round rnd of num_rounds
// uses half rnd % 2 of proxy_rows (all of it when num_rounds = 1), owner node
// after owner node; an owner node's piece of a round holds its node-unique keys
// [uniq_lo, uniq_hi): their rows (the representatives) at
// off + uniq - uniq_lo, then their duplicates' slots, then the fp32 partial
// slots of its heavy keys (local_size each). Rounds 0 and 1 are free when an
// out kernel starts; round rnd + 2 refills round rnd's half once that is done
// with: in the forward once every requester of the node pulled round rnd (the
// in-kernel grants), in the backward once the NIC read round rnd's sums
// (half_free).
struct RoundPiece {
  int32_t uniq_lo, uniq_hi;  // the node-unique keys of the piece
  int32_t off;               // slot of its first row in proxy_rows
  int32_t chunk;             // the proxy's own table (Meta::round_table): the piece's first IB
                             // chunk of the backward, numbered over all pieces; else 0
};
// The push-task flags of one round in push_flags: those of round rnd start at
// (rnd % 2) * half_flags. A round holds at most half of proxy_rows (all of it
// when num_rounds = 1), and its flags each cover 32 of its rows or heavy groups
// (which take more than a row each), per requester rounded up.
__host__ __device__ inline int64_t half_flags(int64_t num_proxy_rows, int local_size) {
  return num_proxy_rows / kTaskRows + 2 * local_size;
}
// The rounds for `total` rows in proxy_rows: 1 if they fit, else enough of
// 90% of a half. A round then holds less than half of proxy_rows, so a proxy
// has one round or at least three, never two.
__host__ __device__ inline int64_t rounds_for(int64_t total, int64_t num_proxy_rows) {
  if (total <= num_proxy_rows) return 1;
  const int64_t room = (int64_t)(0.9 * (double)(num_proxy_rows / 2));
  return (total + room - 1) / room;
}
// The rows proxy_rows needs for every round to fit its half: cut fills a
// round with up to 90% of a half, its piece of each other owner node up to
// kHeavy rows over that owner node's share (a key, of at most kHeavy rows, is
// never split), and the other 10% must hold that much.
__host__ __device__ inline int64_t min_proxy_rows_for(int num_nodes) {
  return 20 * (int64_t)kHeavy * (num_nodes - 1);
}
// A requester's rows for one proxy, sorted by their position in its push
// order (push_pos): round after round, in each round its rows of keys with a
// row each, by (IB chunk, owner node), then its heavy groups: its occurrences
// of each heavy key of the round, in position order. Per round:
struct ReqEntry {
  int32_t start;   // first position of the round (round num_rounds: the end of its
                   // rows)
  int32_t heavy;   // first position of its heavy groups
  int32_t flags;   // its first push-task flag of the round (in the round's half of
                   // push_flags): ceil(rows / 32) tasks of rows, then ceil(groups / 32)
                   // of heavy partials, after those of the lower requesters
  int32_t groups;  // its first heavy group (index into the heavy groups;
                   // round num_rounds: the end of its groups)
};
// An entry of a proxy's reducer list (plan/proxy.h, reducer_entries): a slot of
// proxy_rows in the low word, with kEntryPartial for a heavy key's fp32
// partial row; in the high word the flag of the push task that fills it, or
// kEntryNoPush (a requester without a partial) or kEntryOwnRow (a heavy key's
// own row, which gets the sum).
constexpr int32_t kEntryNoPush = -1;
constexpr int32_t kEntryOwnRow = -2;
constexpr uint32_t kEntryPartial = 0x80000000u;
// A heavy group: positions [start, end) of a requester's rows for one proxy.
// Its backward sums them in sub-tasks of at most kHeavy rows each (the
// requester's group sub-tasks, numbered group after group), then those sums
// in sub-task order.
struct HeavyGroup {
  int32_t start, end;
};
// A segment of a put list: `rows` rows from src to dst (rows of the source and
// destination regions), all in one feature's cell of the owner's KJT; for the
// backward also its IB chunk, whose ready counter it waits for.
struct PutSeg {
  int32_t src, dst, rows, chunk;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
