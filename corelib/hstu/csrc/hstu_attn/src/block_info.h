/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct HstuBlockInfo {
  template <typename Params>
  __device__ HstuBlockInfo(const Params& params, const int bidb)
      : sum_s_q(params.cu_seqlens_q[bidb]),
        sum_s_k(params.cu_seqlens_k[bidb]),
        actual_seqlen_c(params.num_contexts == nullptr ? 0 : params.num_contexts[bidb]),
        actual_seqlen_q(params.cu_seqlens_q[bidb + 1] - sum_s_q),
        actual_seqlen_k(params.cu_seqlens_k[bidb + 1] - sum_s_k),
        actual_seqlen_t(params.num_targets == nullptr ? 0 : params.num_targets[bidb]) {}
    
  template <typename index_t>
  __forceinline__ __device__ index_t q_offset(const index_t row_stride) const {
    return uint32_t(sum_s_q) * row_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t k_offset(const index_t row_stride) const {
    return uint32_t(sum_s_k) * row_stride;
  }

  const int sum_s_q;
  const int sum_s_k;

  const int actual_seqlen_c;
  const int actual_seqlen_q;
  const int actual_seqlen_k;
  const int actual_seqlen_t;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
