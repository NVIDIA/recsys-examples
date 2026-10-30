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
#include "hier_output_dist/common/trace_flag.h"
#include "hier_output_dist/common/trace_slots.h"
#include "hier_output_dist/host/common.h"
#include "hier_output_dist/host/trace.h"

namespace dyn_emb {
namespace hier_output_dist {

Trace::Trace(int device) {
  if constexpr (kTrace) {
    out_ = at::zeros({2, kTraceRing, kTraceWords},
                     at::TensorOptions().device(at::kCUDA, device).dtype(at::kLong));
    in_ = at::full({kTraceRing, kInSteps}, -1, at::kLong);
  }
}

// The earliest stamps start at UINT64_MAX, the rest at 0.
size_t Trace::start_out(bool forward, OutCall &call, cudaStream_t stream) {
  if constexpr (kTrace) {
    const int direction = forward ? 0 : 1;
    const int64_t at = direction * kTraceRing + launches_[direction] % kTraceRing;
    int64_t *row = out_.data_ptr<int64_t>() + at * kTraceWords;
    HIER_CUDA_CHECK(cudaMemsetAsync(row, 0xff, kNumEarliest * sizeof(int64_t), stream));
    HIER_CUDA_CHECK(cudaMemsetAsync(row + kNumEarliest, 0,
                                    (kTraceWords - kNumEarliest) * sizeof(int64_t), stream));
    call.trace = (unsigned long long *)row;
    return kTraceSmWords * sizeof(unsigned long long);
  }
  return 0;
}

void Trace::launched(bool forward) {
  if constexpr (kTrace) ++launches_[forward ? 0 : 1];
}

void Trace::keep_step_times(int64_t seq, const std::vector<at::cuda::CUDAEvent> &events) {
  if constexpr (kTrace) {
    int64_t *row = in_.data_ptr<int64_t>() + seq % kTraceRing * kInSteps;
    for (int step = 0; step < kInSteps; ++step) {
      const at::cuda::CUDAEvent &end = events[1 + step];
      row[step] = end.isCreated() ? (int64_t)(events[0].elapsed_time(end) * 1e6) : -1;
    }
  }
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
