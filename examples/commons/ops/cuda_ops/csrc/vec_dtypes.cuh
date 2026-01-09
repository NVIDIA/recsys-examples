/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#ifndef VEC_DTYPES_CUH_
#define VEC_DTYPES_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <type_traits>

#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))
#define HW_FP8_CONVERSION_ENABLED
#endif

#define INLINE inline __attribute__((always_inline)) __device__


template <typename float_t, size_t vec_size>
struct vec_t {
  INLINE static void memcpy(float_t* dst, const float_t* src);
};

/******************* vec_t<half> *******************/

// half x 1
template <>
struct vec_t<half, 1> {
  INLINE static void memcpy(half* dst, const half* src);
};

INLINE void vec_t<half, 1>::memcpy(half* dst, const half* src) { *dst = *src; }

// half x 2
template <>
struct vec_t<half, 2> {
  INLINE static void memcpy(half* dst, const half* src);
};

INLINE void vec_t<half, 2>::memcpy(half* dst, const half* src) {
  *((half2*)dst) = *((half2*)src);
}

// half x 4

template <>
struct vec_t<half, 4> {
  INLINE static void memcpy(half* dst, const half* src);
};

INLINE void vec_t<half, 4>::memcpy(half* dst, const half* src) {
  *((uint2*)dst) = *((uint2*)src);
}

// half x 8 or more

template <size_t vec_size>
struct vec_t<half, vec_size> {
  static_assert(vec_size % 8 == 0, "Invalid vector size");
  INLINE static void memcpy(half* dst, const half* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((int4*)dst)[i] = ((int4*)src)[i];
    }
  }
};

/******************* vec_t<nv_bfloat16> *******************/

// nv_bfloat16 x 1
template <>
struct vec_t<nv_bfloat16, 1> {
  INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src);
};

INLINE void vec_t<nv_bfloat16, 1>::memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
  *dst = *src;
}

// nv_bfloat16 x 2
template <>
struct vec_t<nv_bfloat16, 2> {
  INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src);
};

INLINE void vec_t<nv_bfloat16, 2>::memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
  *((nv_bfloat162*)dst) = *((nv_bfloat162*)src);
}

// nv_bfloat16 x 4

template <>
struct vec_t<nv_bfloat16, 4> {
  INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src);
};

INLINE void vec_t<nv_bfloat16, 4>::memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
  *((uint2*)dst) = *((uint2*)src);
}

// nv_bfloat16 x 8 or more

template <size_t vec_size>
struct vec_t<nv_bfloat16, vec_size> {
  static_assert(vec_size % 8 == 0, "Invalid vector size");
  INLINE static void memcpy(nv_bfloat16* dst, const nv_bfloat16* src) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((int4*)dst)[i] = ((int4*)src)[i];
    }
  }
};

/******************* vec_t<float> *******************/

// float x 1

template <>
struct vec_t<float, 1> {
  INLINE static void memcpy(float* dst, const float* src);
};

INLINE void vec_t<float, 1>::memcpy(float* dst, const float* src) { *dst = *src; }

// float x 2

template <>
struct vec_t<float, 2> {
  INLINE static void memcpy(float* dst, const float* src);
};

INLINE void vec_t<float, 2>::memcpy(float* dst, const float* src) {
  *((float2*)dst) = *((float2*)src);
}

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  static_assert(vec_size % 4 == 0, "Invalid vector size");
  INLINE static void memcpy(float* dst, const float* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)dst)[i] = ((float4*)src)[i];
    }
  }
};

#endif  // VEC_DTYPES_CUH_