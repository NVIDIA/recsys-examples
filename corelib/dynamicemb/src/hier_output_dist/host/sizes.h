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
// The sizes of an op (Sizes, common/sizes.h) from its configuration alone,
// and the rules of the capacities that follow from the configured ones. The
// capacities are fixed, so the sizes are too.
#pragma once

#include <cstdint>
#include <vector>

#include "hier_output_dist/common/sizes.h"
#include "hier_output_dist/host/config.h"

namespace dyn_emb {
namespace hier_output_dist {

// The sizes of an op of `config`: its ranks, features and rows, and its
// capacities in place: as configured, and across nodes what follows from
// them (0 on one node).
Sizes sizes_of(const Config &config);

// The sizing rules of the capacities that follow from the configured ones
// (sizes_of's, and Config::validate's), across nodes:
//   num_proxy_rows  min(max(remote_keys, the least rows every round needs),
//                   proxy_bytes / row_bytes)
//   rounds          the rounds remote_keys rows take in that many rows
int64_t num_proxy_rows_for(const Config &config);
// rows of an IB chunk, and of proxy_rows a requester's fp32 partial row takes
int chunk_rows_for(int64_t row_bytes);
int partial_rows_for(int64_t dim, int64_t row_bytes);

// The table of each feature, numbered from 0 in the order the tables first
// appear: equal tables keep equal numbers.
std::vector<int> dense_tables(const std::vector<int> &feature_tables);

}  // namespace hier_output_dist
}  // namespace dyn_emb
