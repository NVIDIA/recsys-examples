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
};

struct ScorePolicy {

  static __device__ __forceinline__ ScoreType get(ScorePolicyType policy_type,
                                                  ScoreType *scores,
                                                  int64_t index) {

    if (policy_type == ScorePolicyType::Const) {
      return ScoreType();
    }
    if (policy_type == ScorePolicyType::GlobalTimer) {
      ScoreType score;
      asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(score));
      return score;
    } else {
      return scores[index];
    }
  }

  static __device__ __forceinline__ ScoreType
  score_for_compare(ScorePolicyType policy_type, ScoreType score) {
    return UINT64_MAX;
  }

  static __device__ __forceinline__ void update(ScorePolicyType policy_type,
                                                bool is_return,
                                                ScoreType *table_score,
                                                ScoreType &score) {

    if (policy_type == ScorePolicyType::Const) {
      if (is_return) {
        score = *table_score;
      }
      return;
    }
    if (policy_type == ScorePolicyType::Accumulate) {
      score += *table_score;
      *table_score = score;
    } else {
      *table_score = score;
    }
  }

  static __device__ __forceinline__ void set(bool is_return, ScoreType *scores,
                                             int64_t index, ScoreType score) {
    if (is_return) {
      scores[index] = score;
    }
  }
};

} // namespace dyn_emb