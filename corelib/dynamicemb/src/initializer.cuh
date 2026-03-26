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

namespace dyn_emb {

DEVICE_INLINE unsigned int worker_id() {
  auto grid = cooperative_groups::this_grid();
  return grid.thread_rank();
}

struct UniformEmbeddingGenerator {
  struct Args {
    curandState *state;
    float lower;
    float upper;
  };

  DEVICE_INLINE UniformEmbeddingGenerator(Args args)
      : load_(false), state_(args.state), lower(args.lower), upper(args.upper) {
  }

  DEVICE_INLINE float generate(int64_t vec_id) {
    if (!load_) {
      localState_ = state_[worker_id()];
      load_ = true;
    }
    auto tmp = curand_uniform_double(&this->localState_);
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
  float lower;
  float upper;
};

struct NormalEmbeddingGenerator {
  struct Args {
    curandState *state;
    float mean;
    float std_dev;
  };

  DEVICE_INLINE
  NormalEmbeddingGenerator(Args args)
      : load_(false), state_(args.state), mean(args.mean),
        std_dev(args.std_dev) {}

  DEVICE_INLINE
  float generate(int64_t vec_id) {
    if (!load_) {
      localState_ = state_[worker_id()];
      load_ = true;
    }
    auto tmp = curand_normal_double(&this->localState_);
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
  float mean;
  float std_dev;
};

struct TruncatedNormalEmbeddingGenerator {
  struct Args {
    curandState *state;
    float mean;
    float std_dev;
    float lower;
    float upper;
  };

  DEVICE_INLINE
  TruncatedNormalEmbeddingGenerator(Args args)
      : load_(false), state_(args.state), mean(args.mean),
        std_dev(args.std_dev), lower(args.lower), upper(args.upper) {}

  DEVICE_INLINE
  float generate(int64_t vec_id) {
    if (!load_) {
      localState_ = state_[worker_id()];
      load_ = true;
    }
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
  float mean;
  float std_dev;
  float lower;
  float upper;
  double scale = sqrt(2.0f);
};

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

struct ConstEmbeddingGenerator {
  struct Args {
    float val;
  };

  DEVICE_INLINE
  ConstEmbeddingGenerator(Args args) : val(args.val) {}

  DEVICE_INLINE
  float generate(int64_t vec_id) { return val; }

  DEVICE_INLINE void destroy() {}

  float val;
};

} // namespace dyn_emb
