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
// Part 1 of a call (the requester's, local, on the caller's stream): its keys
// partitioned by owner, and under index dedup the fan-out of its
// deduplicated rows. ATen ops and one cub sort; nothing waits for the device.
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <vector>

namespace dyn_emb {
namespace hier_output_dist {

// The keys of a call as part 1 takes them: the KJT the input dist was given
// and what construct_jagged_tensors reads. Under index dedup the KJT holds
// the deduplicated keys: only its first keys are valid, and only the device
// knows how many (its offsets say).
struct PlanInput {
  at::Tensor values;                  // its keys, int32 or int64
  c10::optional<at::Tensor> offsets;  // [num_features * batch_size + 1], or
  c10::optional<at::Tensor> lengths;  // [num_features * batch_size] when it has none
  int64_t batch_size = 0;
  c10::optional<at::Tensor> weights;        // fp32, one per key
  c10::optional<at::Tensor> input_offsets;  // the input KJT's offsets, when it is another KJT
  int64_t input_batch_size = 0;
  c10::optional<at::Tensor> reverse_indices;  // index dedup: original row -> row of the KJT
};

// The rule that places the keys of each feature on their owners, the
// bucketize's: a DistType code (src/utils.h; codes other than kCyclic and
// kHashedCyclic are kBlock) and the block size kBlock divides by.
struct OwnerRules {
  at::Tensor dist_types;             // [num_features] int32 on the GPU
  at::Tensor block_sizes;            // [num_features] int64 on the GPU
  std::vector<int> host_dist_types;  // the codes on the host: only the rules present run
};

// A call's keys by owner, as the bucketize orders them:
// [owner][feature][sample][position]. B is values.numel(); the valid keys
// are block_start[world_size] of them (fewer under index dedup).
struct Partition {
  at::Tensor keys;         // int64 [B]: each valid key as its owner stores it
  at::Tensor key_weights;  // fp32 [B], or undefined: their weights
  at::Tensor inv;          // int32 [B]: the KJT row of each (a permutation; row past the valid)
  at::Tensor row_cell;     // int32 [B]: the cell owner * num_features + feature of each,
                           // world_size * num_features past the valid
  at::Tensor cell_start;   // int64 [world_size * num_features + 1]: where each cell starts
  at::Tensor block_start;  // int64 [world_size + 1]: where each owner's block starts
  at::Tensor input_length_per_key;  // int64 [num_features]: keys per feature of the input KJT
};
Partition partition_keys(const PlanInput &input, const OwnerRules &rules, int64_t world_size);

// Index dedup's fan-out of the deduplicated rows to the original rows, in the
// format of Fanout (common/formats.h): a row's original rows are
// exp_idx[exp_off[row] .. exp_off[row + 1]), in increasing order. A row of
// more than kHeavy is heavy: it takes a slot (by increasing row) and
// consecutive sub-tasks of kHeavy original rows. Entries past `count` are
// clamped garbage nobody reads.
struct FanoutTables {
  at::Tensor exp_off;  // int32 [rows + 1]
  at::Tensor exp_idx;  // int32 [original rows]
  at::Tensor slot;     // int32 [rows]: the slot of a heavy row, else -1
  at::Tensor sub;      // int64 [2 * max_subs]: (row, first entry in exp_idx) of each sub-task
  at::Tensor count;    // int64 [2]: heavy rows, sub-tasks
  at::Tensor pos;      // int32 [max_slots]: the row of part 1's keys each slot's row went to
  at::Tensor first;    // int32 [max_slots]: the first sub-task of each slot
};
// From reverse_indices (original row -> deduplicated row) and the partition's
// inv, whose size is the number of deduplicated rows.
FanoutTables fanout_tables(const at::Tensor &reverse_indices, const at::Tensor &inv);

}  // namespace hier_output_dist
}  // namespace dyn_emb
