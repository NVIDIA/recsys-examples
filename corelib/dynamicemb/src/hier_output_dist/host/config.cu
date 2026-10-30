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
#include <c10/util/StringUtil.h>

#include <algorithm>
#include <stdexcept>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/host/config.h"
#include "hier_output_dist/host/layout.h"

namespace dyn_emb {
namespace hier_output_dist {

std::string unsupported(int world_size, int local_size, int num_features, int64_t row_bytes) {
  if (world_size < 1 || local_size < 1 || local_size > kMaxLocalRanks || num_features < 1 ||
      num_features > kMaxFeatures || world_size / local_size > kMaxNodes)
    return c10::str("it supports at most ", kMaxLocalRanks, " GPUs per node, ", kMaxNodes,
                    " nodes and ", kMaxFeatures, " features per sharding, got ", local_size, ", ",
                    world_size / std::max(local_size, 1), " and ", num_features);
  if (world_size % local_size != 0)
    return c10::str("the sharding's ", world_size, " ranks are not whole nodes of ", local_size,
                    " GPUs");
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
  // The proxy's identity tags hold a table in 6 bits.
  for (int table : feature_tables)
    if (table < 0 || table >= kMaxFeatures)
      invalid("a table per feature, numbered in [0, ", kMaxFeatures, "), got ", table);
  if (dtype != at::kFloat && dtype != at::kBFloat16 && dtype != at::kHalf)
    invalid("rows of the lookup output must be fp32, bf16 or fp16, got ", dtype);
  check_capacity("lookup_keys", capacities.lookup_keys);
  if (num_nodes() > 1) {
    check_capacity("remote_keys", capacities.remote_keys);
    const int64_t rows = capacities.proxy_bytes / row_bytes();
    const int64_t least = min_proxy_rows_for(num_nodes());
    if (rows < least)
      invalid("a proxy rows region of at most ", (double)capacities.proxy_bytes / (1 << 20),
              " MiB holds ", rows, " rows of ", row_bytes(), " bytes, and across ", num_nodes(),
              " nodes every round fits its half only with at least ", least, " (",
              (least * row_bytes() + (1 << 20) - 1) >> 20, " MiB); raise DYNAMICEMB_GIN_PROXY_MB");
    const int64_t num_proxy_rows = num_proxy_rows_for(*this);
    const int64_t rounds = rounds_for(capacities.remote_keys, num_proxy_rows);
    if (rounds > kMaxRounds)
      invalid("a proxy rows region of ", num_proxy_rows, " rows holds the ", capacities.remote_keys,
              " keys of other nodes' owners a proxy may deduplicate in ", rounds,
              " rounds, more than the ", kMaxRounds,
              " a call can take; raise DYNAMICEMB_GIN_PROXY_MB");
  }
  if (!(timeout_s > 0)) invalid("the timeout must be positive, got ", timeout_s);
  if (ctas < 0 || reduce_ctas < 1)
    invalid("out kernel CTAs must not be negative, and the backward needs a reducing CTA");
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
