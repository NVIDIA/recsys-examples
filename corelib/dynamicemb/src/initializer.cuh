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

#include "check.h"
#include "lookup_kernel.cuh"
#include "torch_utils.h"
#include "utils.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <pybind11/pybind11.h>
#include <random>
#include <string>

namespace dyn_emb {

DEVICE_INLINE unsigned int worker_id() {
  auto grid = cooperative_groups::this_grid();
  return grid.thread_rank();
}

// The parameters a generator reads. Both forms serve a fused module's buffer,
// which holds several logical tables; what differs is the parameters, shared
// by every table or held one row per table. Each kernel is built for exactly
// one of the two: the shared form keeps its parameters in registers and never
// reads memory for them, and neither form carries the other's fields.
template <bool kPerTable, int kNumParams> struct InitParams;

template <int kNumParams> struct InitParams<false, kNumParams> {
  float values[kNumParams];

  DEVICE_INLINE float get(int64_t vec_id, int slot) const {
    return values[slot];
  }
};

template <int kNumParams> struct InitParams<true, kNumParams> {
  const float *table_params; // [num_tables, kNumParams], row-major
  const int64_t *table_ids;  // buffer row -> table, the convention keys uses

  DEVICE_INLINE float get(int64_t vec_id, int slot) const {
    return table_params[table_ids[vec_id] * kNumParams + slot];
  }
};

template <bool kPerTable> struct UniformEmbeddingGenerator {
  static constexpr int kNumParams = 2; // {lower, upper}
  using Params = InitParams<kPerTable, kNumParams>;

  struct Args {
    curandState *state;
    Params params;
  };

  DEVICE_INLINE UniformEmbeddingGenerator(Args args)
      : load_(false), state_(args.state), params_(args.params) {}

  DEVICE_INLINE float generate(int64_t vec_id) {
    if (!load_) {
      localState_ = state_[worker_id()];
      load_ = true;
    }
    auto tmp = curand_uniform_double(&this->localState_);
    float lower = params_.get(vec_id, 0);
    float upper = params_.get(vec_id, 1);
    return static_cast<float>((upper - lower) * tmp + lower);
  }

  DEVICE_INLINE void destroy() {
    if (load_) {
      state_[worker_id()] = localState_;
    }
  }

  bool load_;
  curandState localState_;
  curandState *state_;
  Params params_;
};

template <bool kPerTable> struct NormalEmbeddingGenerator {
  static constexpr int kNumParams = 2; // {mean, std_dev}
  using Params = InitParams<kPerTable, kNumParams>;

  struct Args {
    curandState *state;
    Params params;
  };

  DEVICE_INLINE
  NormalEmbeddingGenerator(Args args)
      : load_(false), state_(args.state), params_(args.params) {}

  DEVICE_INLINE
  float generate(int64_t vec_id) {
    if (!load_) {
      localState_ = state_[worker_id()];
      load_ = true;
    }
    auto tmp = curand_normal_double(&this->localState_);
    float mean = params_.get(vec_id, 0);
    float std_dev = params_.get(vec_id, 1);
    return static_cast<float>(std_dev * tmp + mean);
  }

  DEVICE_INLINE void destroy() {
    if (load_) {
      state_[worker_id()] = localState_;
    }
  }

  bool load_;
  curandState localState_;
  curandState *state_;
  Params params_;
};

template <bool kPerTable> struct TruncatedNormalEmbeddingGenerator {
  static constexpr int kNumParams = 4; // {mean, std_dev, lower, upper}
  using Params = InitParams<kPerTable, kNumParams>;

  struct Args {
    curandState *state;
    Params params;
  };

  DEVICE_INLINE
  TruncatedNormalEmbeddingGenerator(Args args)
      : load_(false), state_(args.state), params_(args.params) {}

  DEVICE_INLINE
  float generate(int64_t vec_id) {
    if (!load_) {
      localState_ = state_[worker_id()];
      load_ = true;
    }
    float mean = params_.get(vec_id, 0);
    float std_dev = params_.get(vec_id, 1);
    float lower = params_.get(vec_id, 2);
    float upper = params_.get(vec_id, 3);
    // Inverse CDF, so the result is the normal conditioned on [lower, upper]
    // rather than one clipped to it -- no mass piles up on the bounds.
    auto l = normcdf((lower - mean) / std_dev);
    auto u = normcdf((upper - mean) / std_dev);
    u = 2 * u - 1;
    l = 2 * l - 1;
    float tmp = curand_uniform_double(&this->localState_);
    tmp = tmp * (u - l) + l;
    tmp = erfinv(tmp);
    tmp *= scale * std_dev;
    tmp += mean;
    tmp = max(tmp, lower);
    tmp = min(tmp, upper);
    return tmp;
  }

  DEVICE_INLINE void destroy() {
    if (load_) {
      state_[worker_id()] = localState_;
    }
  }

  bool load_;
  curandState localState_;
  curandState *state_;
  Params params_;
  double scale = sqrt(2.0f);
};

template <bool kPerTable> struct ConstEmbeddingGenerator {
  static constexpr int kNumParams = 1; // {value}
  using Params = InitParams<kPerTable, kNumParams>;

  struct Args {
    Params params;
  };

  DEVICE_INLINE
  ConstEmbeddingGenerator(Args args) : params_(args.params) {}

  DEVICE_INLINE
  float generate(int64_t vec_id) { return params_.get(vec_id, 0); }

  DEVICE_INLINE void destroy() {}

  Params params_;
};

// DEBUG derives its value from the key alone and takes no parameters, so its
// tables cannot disagree and it needs no per-table form.
template <typename K> struct MappingEmbeddingGenerator {
  struct Args {
    const K *keys;
    uint64_t mod;
  };

  DEVICE_INLINE
  MappingEmbeddingGenerator(Args args) : mod(args.mod), keys(args.keys) {}

  DEVICE_INLINE
  float generate(int64_t vec_id) {
    K key = keys[vec_id];
    return static_cast<float>(key % mod);
  }

  DEVICE_INLINE void destroy() {}

  uint64_t mod;
  const K *keys;
};

} // namespace dyn_emb
