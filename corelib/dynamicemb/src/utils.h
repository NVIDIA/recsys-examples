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

#ifndef UTILS_H
#define UTILS_H
#include "check.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

namespace dyn_emb {

enum class DataType : uint32_t {
  Float32 = 0,
  Float16,
  BFloat16,
  Int64,
  UInt64,
  Int32,
  UInt32,
  Size_t,
};

// The EvictStrategy is consistent with HKV's EvictStrategy. If modifications
// are needed, please refer to the HKV documentation.
// TODO: need be changed to the form
// static_cast<uint32_t>(nv::merlin::EvictStrategy::EvictStrategyEnum::kLru)
enum class EvictStrategy : uint32_t {
  kLru = 0,
  kLfu = 1,      // dynamicemb don't use
  kEpochLru = 2, // dynamicemb don't use
  kEpochLfu = 3, // dynamicemb don't use
  kCustomized = 4,
};

#define CASE_TYPE_USING_HINT(enum_type, type, HINT, ...)                       \
  case (enum_type): {                                                          \
    using HINT = type;                                                         \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }

#define CASE_ENUM_USING_HINT(enum_type, HINT, ...)                             \
  case (enum_type): {                                                          \
    constexpr auto HINT = enum_type;                                           \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }

#define DISPATCH_INTEGER_DATATYPE_FUNCTION(DATA_TYPE, HINT, ...)               \
  switch (DATA_TYPE) {                                                         \
    CASE_TYPE_USING_HINT(DataType::Int64, int64_t, HINT, __VA_ARGS__)          \
    CASE_TYPE_USING_HINT(DataType::UInt64, uint64_t, HINT, __VA_ARGS__)        \
  default:                                                                     \
    exit(EXIT_FAILURE);                                                        \
  }

#define DISPATCH_OFFSET_INT_TYPE(DATA_TYPE, HINT, ...)                         \
  switch (DATA_TYPE) {                                                         \
    CASE_TYPE_USING_HINT(DataType::Int64, int64_t, HINT, __VA_ARGS__)          \
    CASE_TYPE_USING_HINT(DataType::UInt64, uint64_t, HINT, __VA_ARGS__)        \
    CASE_TYPE_USING_HINT(DataType::Int32, int, HINT, __VA_ARGS__)              \
    CASE_TYPE_USING_HINT(DataType::UInt32, uint32_t, HINT, __VA_ARGS__)        \
  default:                                                                     \
    exit(EXIT_FAILURE);                                                        \
  }

#define DISPATCH_FLOAT_DATATYPE_FUNCTION(DATA_TYPE, HINT, ...)                 \
  switch (DATA_TYPE) {                                                         \
    CASE_TYPE_USING_HINT(DataType::Float32, float, HINT, __VA_ARGS__)          \
    CASE_TYPE_USING_HINT(DataType::Float16, __half, HINT, __VA_ARGS__)         \
    CASE_TYPE_USING_HINT(DataType::BFloat16, __nv_bfloat16, HINT, __VA_ARGS__) \
  default:                                                                     \
    exit(EXIT_FAILURE);                                                        \
  }

#define DISPATCH_FLOAT_ACCUM_TYPE_FUNC(ACCUM_TYPE, HINT, ...)                  \
  switch (ACCUM_TYPE) {                                                        \
    CASE_TYPE_USING_HINT(DataType::Float32, float, HINT, __VA_ARGS__)          \
  default:                                                                     \
    exit(EXIT_FAILURE);                                                        \
  }

#define DISPATCH_EVICTYPE_FUNCTION(EVICT_TYPE, HINT, ...)                      \
  switch (EVICT_TYPE) {                                                        \
    CASE_ENUM_USING_HINT(EvictStrategy::kLru, HINT, __VA_ARGS__)               \
    CASE_ENUM_USING_HINT(EvictStrategy::kCustomized, HINT, __VA_ARGS__)        \
    CASE_ENUM_USING_HINT(EvictStrategy::kLfu, HINT, __VA_ARGS__)               \
  default:                                                                     \
    exit(EXIT_FAILURE);                                                        \
  }

#define DISPATCH_BOOLEAN(flag, HINT, ...)                                      \
  if (flag) {                                                                  \
    constexpr bool HINT = true;                                                \
    __VA_ARGS__();                                                             \
  } else {                                                                     \
    constexpr bool HINT = false;                                               \
    __VA_ARGS__();                                                             \
  }

enum class OptimizerType : int {
  Null = 0, // used in inference mode.
  SGD,
  Adam,
  AdaGrad,
  RowWiseAdaGrad,
};

template <typename T>
uint32_t get_optimizer_state_dim(OptimizerType opt_type, uint32_t emb_dim) {
  uint32_t optstate_dim = 0;
  switch (opt_type) {
  case OptimizerType::Null:
  case OptimizerType::SGD: {
    break;
  }
  case OptimizerType::Adam: {
    optstate_dim = emb_dim * 2;
    break;
  }
  case OptimizerType::AdaGrad: {
    optstate_dim = emb_dim;
    break;
  }
  case OptimizerType::RowWiseAdaGrad: {
    optstate_dim = 16 / sizeof(T);
    break;
  }
  default:
    throw std::invalid_argument("Unsupported optimizer type.");
  }
  return optstate_dim;
}

#define HOST_INLINE __host__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x,                      \
               step = blockDim.x * gridDim.x;                                  \
       i < (n); i += step)

class DeviceProp {
public:
  static DeviceProp &getDeviceProp(int device_id = 0);

  // DeviceProp(const DeviceProp&) = delete; //TODO: whether to remove
  DeviceProp &operator=(const DeviceProp &) = delete;

  int num_sms;
  int warp_size;
  int max_thread_per_sm;
  int max_thread_per_block;
  int total_threads;

private:
  explicit DeviceProp(int device_id);
  ~DeviceProp() = default;
};

template <typename TOUT, typename TIN> struct TypeConvertFunc;

template <> struct TypeConvertFunc<__half, float> {
  static __forceinline__ __device__ __half convert(float val) {
    return __float2half(val);
  }
};

template <> struct TypeConvertFunc<float, __half> {
  static __forceinline__ __device__ float convert(__half val) {
    return __half2float(val);
  }
};

template <> struct TypeConvertFunc<nv_bfloat16, float> {
  static __forceinline__ __device__ nv_bfloat16 convert(float val) {
    return __float2bfloat16(val);
  }
};

template <> struct TypeConvertFunc<float, nv_bfloat16> {
  static __forceinline__ __device__ float convert(nv_bfloat16 val) {
    return __bfloat162float(val);
  }
};

template <> struct TypeConvertFunc<nv_bfloat16, __half> {
  static __forceinline__ __device__ nv_bfloat16 convert(__half val) {
    float temp = __half2float(val);
    return __float2bfloat16(temp);
  }
};

template <> struct TypeConvertFunc<__half, nv_bfloat16> {
  static __forceinline__ __device__ __half convert(nv_bfloat16 val) {
    float temp = __bfloat162float(val);
    return __float2half(temp);
  }
};

template <> struct TypeConvertFunc<float, float> {
  static __forceinline__ __device__ float convert(float val) { return val; }
};

template <> struct TypeConvertFunc<__half, __half> {
  static __forceinline__ __device__ __half convert(__half val) { return val; }
};

template <> struct TypeConvertFunc<nv_bfloat16, nv_bfloat16> {
  static __forceinline__ __device__ nv_bfloat16 convert(nv_bfloat16 val) {
    return val;
  }
};

template <> struct TypeConvertFunc<float, long long> {
  static __forceinline__ __device__ float convert(long long val) {
    return static_cast<float>(val);
  }
};

template <> struct TypeConvertFunc<float, unsigned int> {
  static __forceinline__ __device__ float convert(unsigned int val) {
    return static_cast<float>(val);
  }
};

template <> struct TypeConvertFunc<int, long long> {
  static __forceinline__ __device__ int convert(long long val) {
    return static_cast<int>(val);
  }
};

template <> struct TypeConvertFunc<int, unsigned int> {
  static __forceinline__ __device__ int convert(unsigned int val) {
    return static_cast<int>(val);
  }
};

class DeviceCounter {
public:
  DeviceCounter() {
    CUDACHECK(cudaMalloc((void **)&d_counter, sizeof(uint64_t)));
  }

  ~DeviceCounter() { CUDACHECK(cudaFree(d_counter)); }

  DeviceCounter &reset(const cudaStream_t &stream) {
    CUDACHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint64_t), stream));
    return *this;
  }

  uint64_t *get() { return d_counter; }

  DeviceCounter &sync(const cudaStream_t &stream) {
    CUDACHECK(cudaMemcpyAsync(&h_counter, d_counter, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaGetLastError());
    return *this;
  }

  uint64_t result() { return h_counter; }

private:
  uint64_t *d_counter{nullptr};
  uint64_t h_counter{0};
};

} // namespace dyn_emb

#endif // UTILS_H
