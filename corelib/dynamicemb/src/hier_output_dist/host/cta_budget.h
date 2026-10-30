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
// The SMs the kernels that wait for peers share. Every such kernel of the
// process must be resident at once, with an SM to spare. They are one out
// kernel (out_channel.cu runs them one at a time in the process), one CTA of
// the in channel of each live op (its comm kernels that wait run on one CTA,
// one at a time on its side stream), and NCCL's own kernels (kNcclReserve
// SMs). ATen, CUB, push and gather kernels never wait. So every launch ends:
// each such kernel can always be resident, and every rank issues them in the
// same order.
#pragma once

#include "hier_output_dist/host/config.h"

namespace dyn_emb {
namespace hier_output_dist {

// SMs left to NCCL's own kernels.
constexpr int kNcclReserve = 32;

class CtaBudget {
 public:
  CtaBudget(const Config &config, int sms);

  // Counts the op among the process's live ops, which share the budget; the
  // kernels issued before (counted without it) must have ended first.
  void count();
  // Takes the op out of the count; its kernels must have ended first.
  void uncount();
  // CTAs of an out kernel, and of the backward's across nodes those that
  // reduce.
  int out_ctas(bool forward) const;
  int reducers(int ctas) const;

 private:
  int sms_, num_nodes_;
  int ctas_, reduce_ctas_;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
