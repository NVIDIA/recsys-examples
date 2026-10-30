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
// The trace of a build with DYNAMICEMB_HIER_TRACE=1 (kTrace): the slots of
// the out kernels' rows and of a CTA's shared copy of one, and the in
// channel's steps.
#pragma once

#include "hier_output_dist/common/constants.h"

namespace dyn_emb {
namespace hier_output_dist {

// The timeline of the out kernels, a row of kTraceWords uint64 words per
// kernel, in a ring of kTraceRing rows per direction. Times are %globaltimer
// stamps in ns: every GPU counts them on a clock of its own (no common
// origin, and drifting by up to tens of microseconds per second), so only
// stamps of one GPU compare directly. Each stamp is made by one thread per
// CTA (or one lane of a GIN role) and keeps the earliest (atomicMin) or the
// latest (atomicMax) of its CTAs'; the counters are sums. The workers, the
// relay, the issuers and the backward's watcher keep theirs in shared memory
// (TraceSmSlots) and the CTA folds them in when it arrives.
enum TraceSlots : int {
  // the earliest of the CTAs' (the row starts with these at UINT64_MAX)
  kTraceStart = 0,    // the first CTA started
  kTraceFirstPut,     // (num_nodes > 1) this rank posted its first row put over GIN
  kTraceFirstLanded,  // (num_nodes > 1) rows another node sent this rank had landed
  kTraceEnterFirst,   // the first CTA passed the entry barrier
  kTraceNvFirst,      // a worker started its first task that moved rows over NVLink
  kNumEarliest,
  // the latest (the rest of the row starts at 0)
  kTraceEntered = kNumEarliest,  // the last CTA passed the entry barrier
  kTraceLastPut,                 // (num_nodes > 1) this rank posted its last row put
  kTracePutsDone,                // (num_nodes > 1) every row put it posted had completed (the
                                 // issuer's flush; a backward of more than one round: the
                                 // put counters)
  kTraceLanded,                  // (num_nodes > 1) every row other nodes send this rank had landed
  kTraceNvLast,                  // a worker ended its last task that moved rows over NVLink
  kTraceArrived,                 // the last CTA arrived: this rank's rows have all moved
  kTraceExited,                  // it passed the exit barrier: every node peer's have too
                                 // (backward: and its lookup_grad is complete)
  kTraceStartLast,               // the last CTA started
  // sums
  kTraceNvBytes,     // bytes the workers moved over NVLink: rows loaded from node
                     // peers' lookup_rows and proxy_rows (forward), stored into them
                     // (backward)
  kTraceLocalBytes,  // bytes of rows the workers moved within this GPU (its own
                     // lookup_rows or proxy_rows to its output, its gradient to its
                     // own lookup_rows or proxy_rows)
  kTraceIbIn,        // (num_nodes > 1) rows other nodes sent this rank, as the relay
                     // (forward) or the watcher (backward) saw them land
  kTraceIbOut,       // (num_nodes > 1) rows this rank put over GIN
  kNumTrace
};
// Then three timelines of kTraceBins bins, bin i holding what happened from
// i to i + 1 bin widths (kTraceBinNs) after the start of the CTA that saw it,
// the last bin also what came later: bytes moved over NVLink, rows landed
// from other nodes, rows posted to other nodes. They span 5.12 ms.
constexpr int kTraceBins = 512;
constexpr unsigned long long kTraceBinNs = 10000;  // width of a bin: 10 us
constexpr int kTraceBinNv = kNumTrace;
constexpr int kTraceBinIbIn = kTraceBinNv + kTraceBins;
constexpr int kTraceBinIbOut = kTraceBinIbIn + kTraceBins;
constexpr int kTraceWords = kTraceBinIbOut + kTraceBins;
constexpr int kTraceRing = 64;  // rows of each direction's ring
// A CTA's shared copy (dynamic shared memory, only when tracing): its start,
// then the slots it keeps (first and last NVLink task, first landing, the
// counters), the rows of each node the backward's watcher saw land (not
// folded into the row) and the three timelines.
enum TraceSmSlots : int {
  kSmStart = 0,
  kSmNvFirst,
  kSmFirstLanded,
  kSmNvLast,
  kSmNvBytes,
  kSmLocalBytes,
  kSmIbIn,
  kSmIbOut,
  kSmLanded,                        // [kMaxNodes]
  kSmBins = kSmLanded + kMaxNodes,  // [3][kTraceBins], as in the row
  kTraceSmWords = kSmBins + 3 * kTraceBins
};

// The in channel's steps (InChannel::start), in the order a call runs them: a
// row of kInSteps step times per call, each the ns from the call's start on
// the side stream to the step's end (-1: a step the call does not run), in a
// ring of kTraceRing rows on the host (row in_seq % kTraceRing). One node
// runs the first four, kStepKjt and kStepStatus; across nodes every step but
// kStepKjt.
enum InTraceSteps : int {
  kStepCounts = 0,      // the count vector and node_count_exchange
  kStepLayout,          // the node layout
  kStepPush,            // node_key_push
  kStepDone,            // node_done
  kStepKjt,             // the owner's KJT (one node)
  kStepDedup,           // the proxy's node-unique keys, their rows, rounds and slots
  kStepPushOrder,       // the proxy's push order, push tables and reducer list
  kStepMessage,         // the count message
  kStepRailCounts,      // rail_count_exchange
  kStepRailLayout,      // the rail layout
  kStepRailKeys,        // rail_key_exchange
  kStepStatusExchange,  // node_status_exchange and the agreed needs
  kStepGathers,         // the two peer_gathers: the proxy headers, the remote_map words
  kStepRequester,       // the requester tables
  kStepOwner,           // the owner tables
  kStepPutList,         // the proxy's backward put list
  kStepStatus,          // Meta, the status and its pinned copy
  kInSteps
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
