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

namespace dyn_emb {
namespace hier_output_dist {

std::string unsupported(int world_size, int local_size, int num_features, int64_t row_bytes) {
  if (world_size < 1 || local_size < 1 || num_features < 1)
    return c10::str("it needs ranks and features, got ", world_size, " ranks, ", local_size,
                    " GPUs per node and ", num_features, " features");
  if (world_size % local_size != 0)
    return c10::str("the sharding's ", world_size, " ranks are not whole nodes of ", local_size,
                    " GPUs");
  // The planner numbers the bucket features (owner, feature) of a call
  // with int32.
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
[[noreturn]] void refuse(const Parts &...parts) {
  throw std::invalid_argument(c10::str(parts...));
}

void check_capacity(const char *name, int64_t value) {
  if (value < 1 || value >= kMaxEntries)
    refuse("the capacity ", name, " must be in [1, 2^31 - 1024), got ", value);
}

}  // namespace

void Config::validate() const {
  const std::string why = unsupported(world_size, local_size, num_features(), row_bytes());
  if (!why.empty()) refuse(why);
  if (dtype != at::kFloat && dtype != at::kBFloat16 && dtype != at::kHalf)
    refuse("rows of the lookup output must be fp32, bf16 or fp16, got ", dtype);
  check_capacity("lookup_keys", capacities.lookup_keys);
  if (num_nodes() > 1) {
    check_capacity("remote_keys", capacities.remote_keys);
    const Sizes sizes = sizes_of(*this);
    const int64_t rows = capacities.proxy_bytes / row_bytes();
    const int64_t least = min_proxy_rows_for(sizes.max_key_rows());
    if (rows < least)
      refuse("proxy_rows of at most ", (double)capacities.proxy_bytes / (1 << 20), " MiB holds ",
             rows, " rows of ", row_bytes(),
             " bytes, and its ring takes two chunks of keys of the ", "most rows: at least ", least,
             " (", (least * row_bytes() + (1 << 20) - 1) >> 20,
             " MiB); raise DYNAMICEMB_GIN_PROXY_MB");
    // the positions of a proxy's stream, unwrapped, stay below 2^32 (tag_of)
    if (2 * capacities.remote_keys + sizes.num_proxy_rows > (int64_t)UINT32_MAX)
      refuse("a proxy's stream of remote_keys rows through proxy_rows of ", sizes.num_proxy_rows,
             " rows may pass position 2^32; lower remote_keys");
  }
  if (!(timeout_s > 0)) refuse("the timeout must be positive, got ", timeout_s);
  if (ctas < 0 || reduce_ctas < 1)
    refuse("out kernel CTAs must not be negative, and the backward needs a reducing CTA");
}

std::string invalid(const Config &config) {
  try {
    config.validate();
  } catch (const std::invalid_argument &error) {
    return error.what();
  }
  return "";
}

Config config_of(int64_t rank, int64_t world_size, int64_t local_size,
                 const std::vector<int64_t> &feature_tables, const at::Tensor &block_sizes,
                 const at::Tensor &dist_types, int64_t dim, at::ScalarType dtype, bool weights,
                 bool node_dedup, int64_t lookup_keys, int64_t remote_keys, int64_t proxy_bytes,
                 double timeout_s, int64_t ctas, int64_t reduce_ctas) {
  Config config;
  config.rank = (int)rank;
  config.world_size = (int)world_size;
  config.local_size = (int)local_size;
  config.feature_tables.assign(feature_tables.begin(), feature_tables.end());
  config.block_sizes = block_sizes;
  config.dist_types = dist_types;
  config.dim = dim;
  config.dtype = dtype;
  config.weights = weights;
  config.node_dedup = node_dedup;
  config.capacities = Capacities{lookup_keys, remote_keys, proxy_bytes};
  config.timeout_s = timeout_s;
  config.ctas = (int)ctas;
  config.reduce_ctas = (int)reduce_ctas;
  return config;
}

// ── the sizes ──

int64_t max_chunks_for(const Sizes &sizes) {
  return sizes.remote_keys / sizes.chunk_room() + sizes.topo.num_nodes;
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
  sizes.weights = config.weights;
  sizes.timeout_ns = (uint64_t)(std::min(config.timeout_s, 1e9) * 1e9);
  if (num_nodes > 1) {
    sizes.remote_keys = config.capacities.remote_keys;
    const int64_t key_rows = sizes.max_key_rows();
    sizes.num_proxy_rows = std::min(std::max(sizes.remote_keys, min_proxy_rows_for(key_rows)),
                                    config.capacities.proxy_bytes / sizes.row_bytes);
    // 2 MiB of rows, at most half the ring: the ring holds two chunks at least
    const int64_t two_mib = std::max<int64_t>(kTaskRows, (2 << 20) / sizes.row_bytes);
    sizes.chunk_rows = std::max(std::min(two_mib, sizes.num_proxy_rows / 2), 2 * key_rows);
    sizes.max_chunks = max_chunks_for(sizes);
    sizes.record_bytes = config.weights ? 16 : 8;
  }
  return sizes;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
