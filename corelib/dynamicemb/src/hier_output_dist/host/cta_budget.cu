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
#include <algorithm>
#include <atomic>

#include "hier_output_dist/host/common.h"
#include "hier_output_dist/host/cta_budget.h"

namespace dyn_emb {
namespace hier_output_dist {

// Ops alive in the process.
static std::atomic<int> g_live_ops{0};

CtaBudget::CtaBudget(const Config &config, int sms)
    : sms_(sms),
      num_nodes_(config.num_nodes()),
      ctas_(config.ctas),
      reduce_ctas_(config.reduce_ctas) {}

void CtaBudget::count() { ++g_live_ops; }

void CtaBudget::uncount() { --g_live_ops; }

// DYNAMICEMB_GIN_CTAS if set. Else on one node 32 for the forward and 64 for
// the backward: inside a node pulls saturate NVLink at about 32 CTAs, and
// pushes keep improving up to about 64. Across nodes 32 for both: InfiniBand
// bounds them under uniform keys, and under skew the hot owner's NVLink and
// the heavy keys' sums do (more CTAs crowd the backward's pushes). Never more
// than the budget. Across nodes the backward needs CTA 0 (its issuer) and a
// reducing CTA at least.
int CtaBudget::out_ctas(bool forward) const {
  const bool multi = num_nodes_ > 1;
  const int want = ctas_ > 0 ? ctas_ : (forward || multi ? 32 : 64);
  const int ops = g_live_ops.load();
  const int least = multi && !forward ? 2 : 1;
  const int ctas = std::max(least, std::min(want, sms_ - 1 - kNcclReserve - ops));
  if (ctas < want)
    TORCH_WARN_ONCE("hier_output_dist: an out kernel runs on ", ctas, " CTAs instead of ", want,
                    ": the GPU has ", sms_, " SMs, ", ops,
                    " hierarchical dists may each hold one for their input dist, and NCCL keeps ",
                    kNcclReserve);
  return ctas;
}

int CtaBudget::reducers(int ctas) const { return std::max(1, std::min(reduce_ctas_, ctas - 1)); }

}  // namespace hier_output_dist
}  // namespace dyn_emb
