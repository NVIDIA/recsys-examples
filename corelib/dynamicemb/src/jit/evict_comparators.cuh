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

// Eviction comparators for the LruLfu (num_scores == 2) layout. LinearBucket::
// reduce_ranked<Comparator>() computes Comparator::Rank for each candidate from a
// pointer to the key's prefetched [ts, freq] score pair (word 0 = timestamp,
// word 1 = frequency) and keeps the Comparator::less() minimum. Each comparator
// is selected at compile time in evict_lrulfu.cu (one variant per cubin), so the
// async-prefetch scan skeleton is shared but the ranking differs.
#pragma once

#include <cmath>
#include <cstdint>

namespace dyn_emb {

// Resolved at JIT-link time from the user's numba-compiled LTO-IR (custom cubin).
// Left undefined in the LTO-IR fatbin build; nvJitLink links the user function.
// Contract: scores[0] = last-access timestamp, scores[1] = frequency (raw uint64
// AoS words in shared memory). The numba side must cast to float64 before any
// subtraction (uint64 ts - cur_ts would underflow). It should return a FINITE
// value; a NaN return is clamped to -inf (evict-first) below.
extern "C" __device__ double user_score_fn(const uint64_t *scores,
                                            uint64_t cur_ts);

// Default (no score_function): exact lexicographic (frequency asc, then
// timestamp asc) -- lowest frequency evicts, ties broken by the older key. Pure
// integer compare, no double, no numba.
struct LexFreqTsComparator {
  struct Rank {
    uint64_t freq;
    uint64_t ts;
  };
  __device__ __forceinline__ static Rank worst() {
    return Rank{~uint64_t(0), ~uint64_t(0)};
  }
  __device__ __forceinline__ static bool less(const Rank &a, const Rank &b) {
    return a.freq < b.freq || (a.freq == b.freq && a.ts < b.ts);
  }
  __device__ __forceinline__ Rank rank(const uint64_t *scores,
                                       uint64_t /*cur_ts*/) const {
    return Rank{scores[1], scores[0]}; // freq = word 1, ts = word 0
  }
};

// Custom: rank by the user decay score (lower = evict first).
struct UserFnComparator {
  using Rank = double;
  __device__ __forceinline__ static Rank worst() {
    return __longlong_as_double(0x7FF0000000000000LL); // +inf
  }
  __device__ __forceinline__ static bool less(Rank a, Rank b) { return a < b; }
  __device__ __forceinline__ Rank rank(const uint64_t *scores,
                                       uint64_t cur_ts) const {
    double s = user_score_fn(scores, cur_ts);
    // A NaN score (a misbehaving user function) compares false against every
    // candidate, so it could never be selected as the eviction minimum -- and an
    // all-NaN bucket would find no victim and silently drop the incoming key.
    // Clamp NaN to -inf so a broken score ranks lowest (deterministically
    // evict-first), keeping inserts making progress.
    return isnan(s) ? __longlong_as_double(0xFFF0000000000000LL) : s;
  }
};

} // namespace dyn_emb
