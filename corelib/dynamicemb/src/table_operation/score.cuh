/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#pragma once

#include <cstdint>

#include <vector>

#include <cuda_runtime.h>

namespace dyn_emb {

using ScoreType = uint64_t;

enum class ScorePolicyType : uint8_t {
  Const = 0,
  Assign = 1,
  Accumulate = 2,
  GlobalTimer = 3,
  // Compound policy occupying two contiguous (AoS) score words per key:
  //   word 0 = last-access timestamp (device global timer)
  //   word 1 = access frequency (accumulated)
  // Eviction ranks by the frequency word; the timestamp word is used only to
  // select keys for time-based incremental dump. Used for the compound
  // (TIMESTAMP, LFU) score strategy.
  LruLfu = 4,
};

// Number of contiguous score words a policy occupies per key. LruLfu stores a
// (timestamp, frequency) pair; every other policy is a single word.
static __host__ __device__ __forceinline__ int
score_policy_num_scores(ScorePolicyType policy) {
  return policy == ScorePolicyType::LruLfu ? 2 : 1;
}

template <ScorePolicyType PolicyType> struct ScorePolicy {

  static __device__ __forceinline__ ScoreType get(ScoreType *scores,
                                                  int64_t index) {
    if constexpr (PolicyType == ScorePolicyType::Const) {
      return ScoreType();
    } else if constexpr (PolicyType == ScorePolicyType::GlobalTimer) {
      ScoreType score;
      asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(score));
      return score;
    } else {
      // Assign / Accumulate / LruLfu all take their input from `scores`
      // (LruLfu: the per-key frequency delta).
      return scores[index];
    }
  }

  static __device__ __forceinline__ ScoreType
  score_for_compare(ScoreType score) {
    return UINT64_MAX;
  }

  // Update the slot and return the output score. `table_score` points to the
  // key's first score word (AoS base); LruLfu writes both words.
  static __device__ __forceinline__ ScoreType update(ScoreType *table_score,
                                                     ScoreType score) {
    if constexpr (PolicyType == ScorePolicyType::Const) {
      return *table_score;
    } else if constexpr (PolicyType == ScorePolicyType::Accumulate) {
      score += *table_score;
      *table_score = score;
      return score;
    } else if constexpr (PolicyType == ScorePolicyType::LruLfu) {
      ScoreType ts;
      asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ts));
      table_score[0] = ts;         // word 0: last-access timestamp
      score += table_score[1];     // word 1: accumulated frequency
      table_score[1] = score;
      return score;                // output = frequency
    } else {
      *table_score = score;
      return score;
    }
  }
};

} // namespace dyn_emb