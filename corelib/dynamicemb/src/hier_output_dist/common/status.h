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
// A call's status, which the host reads once per call: what it needs of
// each capacity, the same on every rank, and this rank's counts.
#pragma once

namespace dyn_emb {
namespace hier_output_dist {

// What a call needs of each capacity; the status carries the most any rank
// needs of each, the same on every rank. A call fits when no need is above
// its capacity.
//   lookup_keys  keys an owner looks up (lookup_rows, the KJT)
//   remote_keys  remote occurrences a proxy deduplicates (the remote part of
//                node_keys, remote_map, key_stage; num_nodes > 1)
//   rounds       rounds of a proxy (num_nodes > 1; never an overflow)
enum Need : int { kNeedLookupKeys = 0, kNeedRemoteKeys, kNeedRounds, kNumNeeds };

// The status of one call (int64 words), copied to the host once.
enum StatusSlots : int {
  kStatusNeed = 0,                           // [kNumNeeds] the most any rank needs, by Need
  kStatusNumRecv = kStatusNeed + kNumNeeds,  // keys this owner looks up
  kStatusHeavySubs,                          // sub-tasks of this requester's heavy rows
                                             // (index dedup), what its backward sums in
  kStatusGroupSubs,                          // (num_nodes > 1) sub-tasks of its heavy groups
  kStatusSent,                               // rows this requester sends, the valid ones
                                             // of its bound (index dedup: fewer)
  kStatusLengthPerKey,                       // [num_features] keys this owner looks up per
                                             // feature, then [num_features] keys per feature
                                             // of the input KJT that construct_jagged_tensors
                                             // reads
};
__host__ __device__ inline int status_words(int num_features) {
  return kStatusLengthPerKey + 2 * num_features;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
