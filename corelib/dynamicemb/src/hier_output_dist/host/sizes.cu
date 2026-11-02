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
#include "hier_output_dist/host/sizes.h"

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

std::vector<int> dense_tables(const std::vector<int> &feature_tables) {
  std::vector<int> seen, dense;
  for (int table : feature_tables) {
    const auto it = std::find(seen.begin(), seen.end(), table);
    dense.push_back((int)(it - seen.begin()));
    if (it == seen.end()) seen.push_back(table);
  }
  return dense;
}

Sizes sizes_of(const Config &config) {
  const int local_size = config.local_size, num_nodes = config.num_nodes();
  Sizes sizes{};
  sizes.topo = Topo{config.world_size,          local_size,  num_nodes,
                    local_size + num_nodes - 1, config.rank, config.rank % local_size,
                    config.rank / local_size};
  sizes.num_features = config.num_features();
  const std::vector<int> tables = dense_tables(config.feature_tables);
  sizes.num_tables = tables.empty() ? 0 : *std::max_element(tables.begin(), tables.end()) + 1;
  sizes.lookup_keys = config.capacities.lookup_keys;
  sizes.row_bytes = config.row_bytes();
  sizes.partial_rows = partial_rows_for(config.dim, sizes.row_bytes);
  sizes.chunk_rows = chunk_rows_for(sizes.row_bytes);
  sizes.weights = config.weights;
  if (num_nodes > 1) {
    sizes.remote_keys = config.capacities.remote_keys;
    sizes.num_proxy_rows = num_proxy_rows_for(config);
    sizes.rounds_cap = rounds_for(sizes.remote_keys, sizes.num_proxy_rows);
    sizes.record_bytes = config.out_weights ? 16 : 8;
  }
  return sizes;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
