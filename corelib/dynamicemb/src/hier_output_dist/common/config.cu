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
// Config's checks, and an op's sizes from its configuration (config.h).

#include <c10/util/StringUtil.h>

#include <algorithm>
#include <stdexcept>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

std::string unsupported(int world_size, int local_size, int num_features, int64_t row_bytes) {
  if (world_size < 1 || local_size < 1 || num_features < 1)
    return c10::str("it needs ranks and features, got ", world_size, " ranks, ", local_size,
                    " GPUs per node and ", num_features, " features");
  if (world_size % local_size != 0)
    return c10::str("the sharding's ", world_size, " ranks are not whole nodes of ", local_size,
                    " GPUs");
  // The planner numbers the cells (owner, feature) of a call with int32.
  if ((int64_t)world_size * num_features >= kMaxEntries)
    return c10::str("it numbers the (rank, feature) pairs with int32, and ", world_size,
                    " ranks times ", num_features, " features are too many");
  // The out kernels move rows as 16-byte vectors.
  if (row_bytes <= 0 || row_bytes % 16 != 0)
    return c10::str("it moves rows of the lookup output as 16-byte vectors, and these are ",
                    row_bytes, " bytes");
  return "";
}

namespace {

template <typename... Parts>
[[noreturn]] void invalid(const Parts &...parts) {
  throw std::invalid_argument(c10::str(parts...));
}

void check_capacity(const char *name, int64_t value) {
  if (value < 1 || value >= kMaxEntries)
    invalid("the capacity ", name, " must be in [1, 2^31 - 1024), got ", value);
}

}  // namespace

void Config::validate() const {
  const std::string why = unsupported(world_size, local_size, num_features(), row_bytes());
  if (!why.empty()) invalid(why);
  if (dtype != at::kFloat && dtype != at::kBFloat16 && dtype != at::kHalf)
    invalid("rows of the lookup output must be fp32, bf16 or fp16, got ", dtype);
  check_capacity("lookup_keys", capacities.lookup_keys);
  if (num_nodes() > 1) {
    check_capacity("remote_keys", capacities.remote_keys);
    const int64_t rows = capacities.proxy_bytes / row_bytes();
    const int64_t least = min_proxy_rows_for(num_nodes());
    if (rows < least)
      invalid("proxy_rows of at most ", (double)capacities.proxy_bytes / (1 << 20), " MiB holds ",
              rows, " rows of ", row_bytes(), " bytes, and across ", num_nodes(),
              " nodes every round fits its half only with at least ", least, " (",
              (least * row_bytes() + (1 << 20) - 1) >> 20, " MiB); raise DYNAMICEMB_GIN_PROXY_MB");
    const int64_t num_proxy_rows = num_proxy_rows_for(*this);
    const int64_t rounds = rounds_for(capacities.remote_keys, num_proxy_rows);
    if (rounds > kMaxRounds)
      invalid("proxy_rows of ", num_proxy_rows, " rows holds the ", capacities.remote_keys,
              " keys of other nodes' owners a proxy may deduplicate in ", rounds,
              " rounds, more than the ", kMaxRounds,
              " a call can take; raise DYNAMICEMB_GIN_PROXY_MB");
    const int bits = push_key_of(sizes_of(*this)).bits;
    if (bits > 63)
      invalid("the proxy sorts its push order by a key of ", bits,
              " bits (its fields: the GPUs of a node, the rounds, and the keys and IB chunks of ",
              "remote_keys), more than 63; lower remote_keys");
  }
  if (!(timeout_s > 0)) invalid("the timeout must be positive, got ", timeout_s);
  if (ctas < 0 || reduce_ctas < 1)
    invalid("out kernel CTAs must not be negative, and the backward needs a reducing CTA");
}

// ── the sizes ──

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
