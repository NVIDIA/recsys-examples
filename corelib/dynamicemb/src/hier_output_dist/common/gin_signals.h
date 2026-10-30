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
// GIN between nodes: its contexts, and the signals and counters of each.
#pragma once

#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

// GIN between nodes (num_nodes > 1): 4 contexts on the rail connection, each
// with 3 signals per node and 2 counters, all starting at 0 (checked once),
// every one a running total that is never reset. A signal belongs to one
// context, whose QP to a peer executes in order, and is named after the
// sender's node:
//   context 0 (forward rows)   fwd_rows[owner_node]: rows the owner on that node
//                              put into this proxy's proxy_rows;
//   context 1 (backward rows)  bwd_rows[node]: gradient rows the proxy on that
//                              node put into this owner's lookup_rows; counter
//                              sent[half]: this proxy's backward puts out of
//                              that half of proxy_rows whose source the NIC has
//                              read (in calls of more than one round; one
//                              round's puts count on none);
//   context 2 (in channel)     count[node] counts the count messages from the
//                              rail peer on that node, keys[node] its key
//                              streams (one put, or a payload-free signal, per
//                              call each);
//   context 3 (credits)        grant[node] = (out_seq << 20) | rounds of
//                              proxy_rows the proxy on that node lets this
//                              owner fill, bwd_ready[owner_node] = the out_seq
//                              of the backward whose rows the owner on that
//                              node lets this proxy put into its lookup_rows,
//                              closing[node]: the rail peer on that node tears
//                              down.
// Credits are payload-free SignalAdds of the difference to the value the
// sender sent before (Ctrl grant_sent, bwd_ready_sent), and name the out
// kernel they admit.
constexpr int kGinContexts = 4;
constexpr int kGinSignalsPerNode = 3;
constexpr int kGinCounters = 2;
constexpr int kCtxFwd = 0;
constexpr int kCtxBwd = 1;
constexpr int kCtxIn = 2;
constexpr int kCtxCredit = 3;
__host__ __device__ inline int sig_count(int node) { return node; }
__host__ __device__ inline int sig_keys(int num_nodes, int node) { return num_nodes + node; }
// fwd_rows (context 0), bwd_rows (context 1)
__host__ __device__ inline int sig_rows(int node) { return node; }
__host__ __device__ inline int sig_grant(int node) { return node; }
__host__ __device__ inline int sig_bwd_ready(int num_nodes, int owner_node) {
  return num_nodes + owner_node;
}
__host__ __device__ inline int sig_closing(int num_nodes, int node) { return 2 * num_nodes + node; }
__host__ __device__ inline uint64_t grant_of(int64_t out_seq, int rounds) {
  return ((uint64_t)out_seq << 20) | (uint64_t)rounds;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
