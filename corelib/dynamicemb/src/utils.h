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
#include <cuda_fp8.h>
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
  Float8,
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

// How a bag of embeddings is combined into one output row.  This is the single
// source of truth for the numbering: dynamicemb.DynamicEmbPoolingMode takes its
// values from the bound enum, so a pooling mode reaches the kernels with no
// translation on the way.
enum class PoolingMode : int32_t {
  kSum = 0,
  kMean = 1,
  kNone = 2, // sequence lookup: one output row per key, nothing to combine
};

// 8-bit table storage: an e4m3 code holding value * 2^8. The shift moves e4m3's
// range to [2^-17, 1.75] -- exponent bias 15, the format FBGEMM's FP8 comms use
// -- which fits trained embeddings (|w| is typically 1e-3..0.4) where plain
// e4m3fn would flush everything below 2^-9 and leave a third of the values
// subnormal. Writes round stochastically: an optimizer step is often smaller
// than half an e4m3 step (1/16 of |w|), and round-to-nearest would discard it,
// freezing rows after a few dozen updates. Values already on the e4m3 grid
// (e.g. loaded from an fp16 dump of this table) are stored exactly.
#ifdef __CUDACC__
constexpr float kFp8StorageScale = 256.0f;
constexpr float kFp8StorageInvScale = 1.0f / 256.0f;

__device__ __forceinline__ uint32_t fp8_rounding_noise(float value) {
  uint32_t h = __float_as_uint(value) ^ static_cast<uint32_t>(clock64()) ^
               ((threadIdx.x + blockIdx.x * blockDim.x) * 0x9E3779B9u);
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;
  return h;
}

__device__ __forceinline__ __nv_fp8_storage_t float_to_fp8_storage(float value) {
  float scaled = value * kFp8StorageScale;
  const float magnitude = fabsf(scaled);
  if (magnitude < 448.0f) {
    const uint32_t noise = fp8_rounding_noise(scaled);
    if (magnitude >= 0.015625f) {
      // Normal e4m3 keeps 3 of fp32's 23 mantissa bits: dither the 20 dropped
      // bits, then truncate.
      uint32_t bits = __float_as_uint(scaled);
      bits = (bits + (noise & 0xFFFFFu)) & ~0xFFFFFu;
      scaled = __uint_as_float(bits);
    } else {
      // Subnormal e4m3 is a fixed grid of 2^-9.
      constexpr float kSubnormalStep = 0.001953125f;
      scaled = floorf(scaled / kSubnormalStep +
                      static_cast<float>(noise >> 8) * (1.0f / 16777216.0f)) *
               kSubnormalStep;
    }
  }
  return __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float fp8_storage_to_float(__nv_fp8_storage_t bits) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3))) *
         kFp8StorageInvScale;
}

struct dyn_fp8_t {
  __nv_fp8_storage_t bits;

  dyn_fp8_t() = default;
  __device__ __forceinline__ explicit dyn_fp8_t(float value)
      : bits(float_to_fp8_storage(value)) {}
  __device__ __forceinline__ operator float() const {
    return fp8_storage_to_float(bits);
  }
};
static_assert(sizeof(dyn_fp8_t) == 1, "dyn_fp8_t must be one byte");

// Up to four consecutive fp8 codes, moved as one 32-bit word when the address
// allows it. Host-resident tables pay a PCIe transaction per access, so byte
// accesses make an fp8 table slower than an fp16 one.
__device__ __forceinline__ void load_fp8_codes(const dyn_fp8_t *src, int n,
                                               dyn_fp8_t *codes) {
  if (n == 4 && (reinterpret_cast<uintptr_t>(src) & 3u) == 0) {
    const uint32_t word = *reinterpret_cast<const uint32_t *>(src);
    for (int i = 0; i < 4; ++i)
      codes[i].bits = static_cast<__nv_fp8_storage_t>((word >> (8 * i)) & 0xFFu);
  } else {
    for (int i = 0; i < n && i < 4; ++i)
      codes[i] = src[i];
  }
}

__device__ __forceinline__ void store_fp8_codes(dyn_fp8_t *dst, int n,
                                                const dyn_fp8_t *codes) {
  if (n == 4 && (reinterpret_cast<uintptr_t>(dst) & 3u) == 0) {
    uint32_t word = 0;
    for (int i = 0; i < 4; ++i)
      word |= static_cast<uint32_t>(codes[i].bits) << (8 * i);
    *reinterpret_cast<uint32_t *>(dst) = word;
  } else {
    for (int i = 0; i < n && i < 4; ++i)
      dst[i] = codes[i];
  }
}
#endif // __CUDACC__

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

// Every type a table can store its values in: the float types plus the 8-bit
// dyn_fp8_t storage format, which no gradient or accumulator ever uses.
#define DISPATCH_VALUE_DATATYPE_FUNCTION(DATA_TYPE, HINT, ...)                 \
  switch (DATA_TYPE) {                                                         \
    CASE_TYPE_USING_HINT(DataType::Float32, float, HINT, __VA_ARGS__)          \
    CASE_TYPE_USING_HINT(DataType::Float16, __half, HINT, __VA_ARGS__)         \
    CASE_TYPE_USING_HINT(DataType::BFloat16, __nv_bfloat16, HINT, __VA_ARGS__) \
    CASE_TYPE_USING_HINT(DataType::Float8, dyn_fp8_t, HINT, __VA_ARGS__)       \
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
  int64_t totalGlobalMem;

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

#ifdef __CUDACC__
template <> struct TypeConvertFunc<dyn_fp8_t, float> {
  static __forceinline__ __device__ dyn_fp8_t convert(float val) {
    return dyn_fp8_t(val);
  }
};

template <> struct TypeConvertFunc<float, dyn_fp8_t> {
  static __forceinline__ __device__ float convert(dyn_fp8_t val) {
    return static_cast<float>(val);
  }
};

template <> struct TypeConvertFunc<dyn_fp8_t, __half> {
  static __forceinline__ __device__ dyn_fp8_t convert(__half val) {
    return dyn_fp8_t(__half2float(val));
  }
};

template <> struct TypeConvertFunc<__half, dyn_fp8_t> {
  static __forceinline__ __device__ __half convert(dyn_fp8_t val) {
    return __float2half(static_cast<float>(val));
  }
};

template <> struct TypeConvertFunc<dyn_fp8_t, nv_bfloat16> {
  static __forceinline__ __device__ dyn_fp8_t convert(nv_bfloat16 val) {
    return dyn_fp8_t(__bfloat162float(val));
  }
};

template <> struct TypeConvertFunc<nv_bfloat16, dyn_fp8_t> {
  static __forceinline__ __device__ nv_bfloat16 convert(dyn_fp8_t val) {
    return __float2bfloat16(static_cast<float>(val));
  }
};

template <> struct TypeConvertFunc<dyn_fp8_t, dyn_fp8_t> {
  static __forceinline__ __device__ dyn_fp8_t convert(dyn_fp8_t val) {
    return val;
  }
};
#endif // __CUDACC__

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
