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
// The ranks of a sharding: its nodes and each rank's place in them.
#pragma once

namespace dyn_emb {
namespace hier_output_dist {

// Who is who. Ranks are node-contiguous: rank = node * local_size +
// local_rank. The owner's KJT has num_sources = local_size + num_nodes - 1
// columns: its node's requesters, then the other nodes.
struct Topo {
  int world_size, local_size, num_nodes, num_sources;
  int rank, local_rank, node;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
