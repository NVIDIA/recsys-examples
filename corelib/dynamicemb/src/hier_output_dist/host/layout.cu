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
#include <algorithm>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/status.h"
#include "hier_output_dist/host/layout.h"

namespace dyn_emb {
namespace hier_output_dist {

int64_t num_proxy_rows_for(const Config &config) {
  const int64_t least = min_proxy_rows_for(config.num_nodes());
  return std::min(std::max(config.capacities.remote_keys, least),
                  config.capacities.proxy_bytes / config.row_bytes());
}

// IB chunks are 2 MiB of rows, a multiple of kTaskRows.
int chunk_rows_for(int64_t row_bytes) {
  return (int)std::max<int64_t>(kTaskRows, (2 << 20) / row_bytes / kTaskRows * kTaskRows);
}

// A requester's fp32 partial row of a heavy key takes 4 * dim bytes of proxy_rows.
int partial_rows_for(int64_t dim, int64_t row_bytes) {
  return (int)((4 * dim + row_bytes - 1) / row_bytes);
}

Ctrl ctrl_layout(int local_size, int num_nodes, int num_features) {
  const bool multi = num_nodes > 1;
  Ctrl ctrl_at{};
  int next = 0;
  auto take = [&](int words) {
    const int at = next;
    next += words;
    return at;
  };
  ctrl_at.in_started = take(1);
  ctrl_at.in_meta[0] = take(local_size * (num_features + 1));
  ctrl_at.in_meta[1] = take(local_size * (num_features + 1));
  ctrl_at.in_written = take(local_size);
  ctrl_at.cell_start = take(multi ? local_size * num_features : 0);
  ctrl_at.in_done = take(multi ? 1 + kNumNeeds : 0);
  ctrl_at.msg_words = (msg_len(num_nodes, num_features) + 1) / 2 * 2;  // whole 16-byte units
  ctrl_at.count_msg_out = take(multi ? ctrl_at.msg_words : 0);
  ctrl_at.count_msgs = take(multi ? 2 * num_nodes * ctrl_at.msg_words : 0);
  ctrl_at.out_started = take(1);
  ctrl_at.pushed = take(local_size);
  ctrl_at.pulled = take(1);
  ctrl_at.arrive = take(1);
  ctrl_at.tasks = take(kNumQueues);
  ctrl_at.landed_rows = take(multi ? num_nodes : 0);
  ctrl_at.fwd_tot = take(multi ? num_nodes : 0);
  ctrl_at.bwd_tot = take(multi ? num_nodes : 0);
  ctrl_at.grant_sent = take(multi ? num_nodes : 0);
  ctrl_at.bwd_ready_sent = take(multi ? num_nodes : 0);
  ctrl_at.posted = take(multi ? 2 : 0);
  ctrl_at.done = take(multi ? 2 * local_size : 0);
  ctrl_at.half_free = take(multi ? 1 : 0);
  ctrl_at.closing = take(1);
  ctrl_at.closed = take(1);
  ctrl_at.words = next;
  return ctrl_at;
}

Layout::Layout(const Config &config) {
  const int local_size = config.local_size, num_nodes = config.num_nodes();
  topo = Topo{config.world_size,          local_size,  num_nodes,
              local_size + num_nodes - 1, config.rank, config.rank % local_size,
              config.rank / local_size};
  num_features = config.num_features();
  row_bytes = config.row_bytes();
  partial_rows = partial_rows_for(config.dim, row_bytes);
  chunk_rows = chunk_rows_for(row_bytes);
  ctrl_at = ctrl_layout(local_size, num_nodes, num_features);
  lookup_keys = config.capacities.lookup_keys;
  remote_keys = num_proxy_rows = 0;
  rounds_cap = record_bytes = 0;
  if (num_nodes > 1) {
    remote_keys = config.capacities.remote_keys;
    num_proxy_rows = num_proxy_rows_for(config);
    rounds_cap = (int)rounds_for(remote_keys, num_proxy_rows);
    record_bytes = config.out_weights ? 16 : 8;
  }
  // node_keys holds an own part of lookup_keys entries, then (across nodes)
  // a remote part of remote_keys.
  const uint64_t node_entries = (uint64_t)(lookup_keys + remote_keys);
  const uint64_t pieces = (uint64_t)(num_nodes - 1) * rounds_cap * sizeof(RoundPiece);
  uint64_t *bytes = region_bytes;
  std::fill(bytes, bytes + kNumRegions, 0);
  bytes[kCtrl] = 8 * (uint64_t)ctrl_at.words;
  bytes[kNodeKeys] = 8 * node_entries;
  bytes[kNodeWeights] = config.weights ? 4 * node_entries : 0;
  bytes[kLookupRows] = (uint64_t)lookup_keys * row_bytes;
  if (num_nodes > 1) {
    bytes[kRemoteCells] = 2 * (uint64_t)remote_keys;
    bytes[kRemoteMap] = 12 * (uint64_t)remote_keys;
    // its rounds, its round table and each requester's push tables (int32 words)
    bytes[kProxyHdr] = 4 * (4 + 4 * (uint64_t)rounds_cap * num_nodes +
                            4 * (uint64_t)local_size * (rounds_cap + 1));
    // the slices, then 16 bytes that take what no key writes
    bytes[kKeyStage] = pieces + (uint64_t)remote_keys * record_bytes + 16;
    bytes[kRailKeys] = (uint64_t)lookup_keys * record_bytes + pieces;
    bytes[kProxyRows] = (uint64_t)num_proxy_rows * row_bytes;
    bytes[kPushFlags] = 8 * 2 * (uint64_t)half_flags(num_proxy_rows, local_size);
  }
  meta_at = meta_layout(num_nodes, local_size, num_features, meta_rounds());
  // The regions of each window in Region order, each at a multiple of
  // kRegionAlign.
  for (int window = 0; window < kNumWins; ++window) {
    uint64_t total = 0;
    for (int region = 0; region < kNumRegions; ++region) {
      if (kRegionWin[region] != window) continue;
      region_off[region] = total;
      total += (bytes[region] + kRegionAlign - 1) / kRegionAlign * kRegionAlign;
    }
    window_bytes[window] = total;
  }
}

PlanSizes Layout::plan_sizes() const {
  return PlanSizes{topo,       num_features, lookup_keys, remote_keys, num_proxy_rows,
                   rounds_cap, partial_rows, chunk_rows,  record_bytes};
}

std::vector<std::pair<std::string, int64_t>> meta_fields(const Meta &meta_at) {
  return {{"proxy_round_tables", meta_at.proxy_round_tables},
          {"push_tables", meta_at.push_tables},
          {"push_order_off", meta_at.push_order_off},
          {"proxy_rounds", meta_at.proxy_rounds},
          {"fwd_task_starts", meta_at.fwd_task_starts},
          {"bwd_task_starts", meta_at.bwd_task_starts},
          {"slice_in", meta_at.slice_in},
          {"uniq_in", meta_at.uniq_in},
          {"fwd_put_list", meta_at.fwd_put_list},
          {"uniq_off", meta_at.uniq_off},
          {"uniq_out", meta_at.uniq_out},
          {"round_table", meta_at.round_table},
          {"bwd_put_start", meta_at.bwd_put_start},
          {"push_start", meta_at.push_start},
          {"reduce_task_starts", meta_at.reduce_task_starts},
          {"misc", meta_at.misc},
          {"words", meta_at.words}};
}

const std::vector<std::string> &region_names() {
  static const std::vector<std::string> names = {
      "ctrl",      "node_keys", "node_weights", "remote_cells", "remote_map", "proxy_hdr",
      "key_stage", "rail_keys", "lookup_rows",  "proxy_rows",   "push_flags"};
  return names;
}

const std::vector<std::string> &window_names() {
  static const std::vector<std::string> names = {"ctrl", "keys", "rows"};
  return names;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
