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
// Every size and offset of an op, from its configuration alone: the ranks,
// the control words, a call's small tables and the regions of the windows.
// The capacities are fixed, so all of them are too.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "hier_output_dist/common/ctrl.h"
#include "hier_output_dist/common/meta.h"
#include "hier_output_dist/common/topology.h"
#include "hier_output_dist/common/windows.h"
#include "hier_output_dist/host/config.h"
#include "hier_output_dist/plan/tensors.h"

namespace dyn_emb {
namespace hier_output_dist {

// The window of each region, and whether the region must read 0 before its
// window is registered (the control words: sequence numbers start at 0; the
// push-task flags: tags start at 0). Every other region is written before it
// is read, so it is never initialized.
constexpr WinId kRegionWin[kNumRegions] = {kWinCtrl, kWinKeys, kWinKeys, kWinKeys,
                                           kWinKeys, kWinKeys, kWinKeys, kWinKeys,
                                           kWinRows, kWinRows, kWinRows};
constexpr bool kRegionZero[kNumRegions] = {true,  false, false, false, false, false,
                                           false, false, false, false, true};
// Each region starts at a multiple of this inside its window.
constexpr uint64_t kRegionAlign = 4096;

struct Layout {
  explicit Layout(const Config &config);

  Topo topo;
  int num_features;
  int64_t row_bytes;
  int partial_rows;  // rows of proxy_rows a requester's fp32 partial row takes
  int chunk_rows;    // rows of an IB chunk (2 MiB), a multiple of kTaskRows
  Ctrl ctrl_at;      // the control words (Ctrl)
  Meta meta_at;      // a call's small tables, for rounds_cap rounds
  // Capacities in place: as configured, and across nodes what follows from
  // them (0 on one node).
  int64_t lookup_keys, remote_keys;
  int64_t num_proxy_rows;  // rows of proxy_rows
  int rounds_cap;          // rounds a round table holds
  int record_bytes;        // bytes of a key record: 8, or 16 with its weight
  uint64_t region_bytes[kNumRegions];
  uint64_t region_off[kNumRegions];  // in its window
  uint64_t window_bytes[kNumWins];   // as allocated

  bool multi_node() const { return topo.num_nodes > 1; }
  // The rounds a call's meta tables are laid out for (at least 1).
  int meta_rounds() const { return rounds_cap > 0 ? rounds_cap : 1; }
  // The sizes the planner's steps lay a call's tables out by.
  PlanSizes plan_sizes() const;
};

// The sizing rules of the capacities that follow from the configured ones
// (Layout's, and Config::validate's), across nodes:
//   num_proxy_rows  min(max(remote_keys, the least rows every round needs),
//                   proxy_bytes / row_bytes)
//   rounds          the rounds remote_keys rows take in that many rows
int64_t num_proxy_rows_for(const Config &config);
// rows of an IB chunk, and of proxy_rows a requester's fp32 partial row takes
int chunk_rows_for(int64_t row_bytes);
int partial_rows_for(int64_t dim, int64_t row_bytes);

// The control words' layout of an op of these ranks and features.
Ctrl ctrl_layout(int local_size, int num_nodes, int num_features);
// A call's small tables (Meta) by name, in field order.
std::vector<std::pair<std::string, int64_t>> meta_fields(const Meta &meta_at);
// The names of the regions (Region order) and of the windows (WinId order).
const std::vector<std::string> &region_names();
const std::vector<std::string> &window_names();

}  // namespace hier_output_dist
}  // namespace dyn_emb
