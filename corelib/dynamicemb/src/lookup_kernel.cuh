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

#ifndef LOOKUP_KERNEL_CUH
#define LOOKUP_KERNEL_CUH

#include "check.h"
#include "utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
namespace dyn_emb {
#define EV_NUM 32
#define WARP_SIZE 32

DEVICE_INLINE unsigned int GlobalThreadId() {
  unsigned int smid;
  unsigned int warpid;
  unsigned int laneid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  asm("mov.u32 %0, %%warpid;" : "=r"(warpid));
  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  /// TODO: align with device property.
  return smid * 2048 + warpid * 32 + laneid;
}

template <typename T> struct Vec4T {};

template <> struct Vec4T<__half> {
  union U {
    float2 f;
    __half2 h[2];
  } value;

  DEVICE_INLINE Vec4T() {
    value.h[0].x = 0.f;
    value.h[0].y = 0.f;
    value.h[1].x = 0.f;
    value.h[1].y = 0.f;
  }

  DEVICE_INLINE void reset() {
    value.h[0].x = 0.f;
    value.h[0].y = 0.f;
    value.h[1].x = 0.f;
    value.h[1].y = 0.f;
  }

  DEVICE_INLINE void reset(const __half initial_value) {
    value.h[0].x = initial_value;
    value.h[0].y = initial_value;
    value.h[1].x = initial_value;
    value.h[1].y = initial_value;
  }

  DEVICE_INLINE void load(const float *p, int n) {
    if (n == 4) {
      float4 f = *(reinterpret_cast<const float4 *>(p));
      float2 firstf{f.x, f.y};
      float2 secondf{f.z, f.w};
      value.h[0] = __float22half2_rn(firstf);
      value.h[1] = __float22half2_rn(secondf);
    } else {
      if (n > 0)
        value.h[0].x = __float2half(p[0]);
      if (n > 1)
        value.h[0].y = __float2half(p[1]);
      if (n > 2)
        value.h[1].x = __float2half(p[2]);
    }
  }

  DEVICE_INLINE void load(const __half *p, int n) {
    if (n == 4) {
      value.f = *(reinterpret_cast<const float2 *>(p));
    } else {
      if (n > 0)
        value.h[0].x = p[0];
      if (n > 1)
        value.h[0].y = p[1];
      if (n > 2)
        value.h[1].x = p[2];
    }
  }

  DEVICE_INLINE void load(const __nv_bfloat16 *p, int n) {
    if (n == 4) {
      float2 f = *(reinterpret_cast<const float2 *>(p));
      __nv_bfloat162 first = __float2bfloat162_rn(f.x);
      __nv_bfloat162 second = __float2bfloat162_rn(f.y);
      value.h[0] =
          __half2(TypeConvertFunc<__half, __nv_bfloat16>::convert(first.x),
                  TypeConvertFunc<__half, __nv_bfloat16>::convert(first.y));
      value.h[1] =
          __half2(TypeConvertFunc<__half, __nv_bfloat16>::convert(second.x),
                  TypeConvertFunc<__half, __nv_bfloat16>::convert(second.y));
    } else {
      if (n > 0)
        value.h[0].x = TypeConvertFunc<__half, __nv_bfloat16>::convert(p[0]);
      if (n > 1)
        value.h[0].y = TypeConvertFunc<__half, __nv_bfloat16>::convert(p[1]);
      if (n > 2)
        value.h[1].x = TypeConvertFunc<__half, __nv_bfloat16>::convert(p[2]);
    }
  }

  DEVICE_INLINE void load(const float *p) {
    float4 f = *(reinterpret_cast<const float4 *>(p));
    float2 firstf{f.x, f.y};
    float2 secondf{f.z, f.w};
    value.h[0] = __float22half2_rn(firstf);
    value.h[1] = __float22half2_rn(secondf);
  }

  DEVICE_INLINE void load(const __half *p) {
    value.f = *(reinterpret_cast<const float2 *>(p));
  }

  DEVICE_INLINE void load(const __nv_bfloat16 *p) {
    float2 f = *(reinterpret_cast<const float2 *>(p));
    __nv_bfloat162 first = __float2bfloat162_rn(f.x);
    __nv_bfloat162 second = __float2bfloat162_rn(f.y);
    value.h[0] =
        __half2(TypeConvertFunc<__half, __nv_bfloat16>::convert(first.x),
                TypeConvertFunc<__half, __nv_bfloat16>::convert(first.y));
    value.h[1] =
        __half2(TypeConvertFunc<__half, __nv_bfloat16>::convert(second.x),
                TypeConvertFunc<__half, __nv_bfloat16>::convert(second.y));
  }

  DEVICE_INLINE void atomic_store_accum(float *dst, int n) {
    if (n > 0)
      atomicAdd(dst, __half2float(value.h[0].x));
    if (n > 1)
      atomicAdd(dst + 1, __half2float(value.h[0].y));
    if (n > 2)
      atomicAdd(dst + 2, __half2float(value.h[1].x));
    if (n > 3)
      atomicAdd(dst + 3, __half2float(value.h[1].y));
  }

  DEVICE_INLINE void atomic_store_accum(__half *dst, int n) {
    if (n == 4) {
      atomicAdd((reinterpret_cast<__half2 *>(dst)), value.h[0]);
      atomicAdd((reinterpret_cast<__half2 *>(dst + 2)), value.h[1]);
    } else {
      if (n > 0)
        atomicAdd(dst, value.h[0].x);
      if (n > 1)
        atomicAdd(dst + 1, value.h[0].y);
      if (n > 2)
        atomicAdd(dst + 2, value.h[1].x);
    }
  }

  DEVICE_INLINE void atomic_store_accum(__nv_bfloat16 *dst, int n) {
    if (n == 4) {
      __nv_bfloat162 h0 = __nv_bfloat162(
          TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[0].x),
          TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[0].y));
      __nv_bfloat162 h1 = __nv_bfloat162(
          TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[1].x),
          TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[1].y));
      atomicAdd(reinterpret_cast<__nv_bfloat162 *>(dst), h0);
      atomicAdd(reinterpret_cast<__nv_bfloat162 *>(dst + 2), h1);
    } else {
      if (n > 0)
        atomicAdd(
            dst, TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[0].x));
      if (n > 1)
        atomicAdd(dst + 1, TypeConvertFunc<__nv_bfloat16, __half>::convert(
                               value.h[0].y));
      if (n > 2)
        atomicAdd(dst + 2, TypeConvertFunc<__nv_bfloat16, __half>::convert(
                               value.h[1].x));
    }
  }

  DEVICE_INLINE void store(float *dst, int n) {
    if (n == 4) {
      float4 f;
      f.x = __half2float(value.h[0].x);
      f.y = __half2float(value.h[0].y);
      f.z = __half2float(value.h[1].x);
      f.w = __half2float(value.h[1].y);
      *(reinterpret_cast<float4 *>(dst)) = f;
    } else {
      if (n > 0)
        dst[0] = __half2float(value.h[0].x);
      if (n > 1)
        dst[1] = __half2float(value.h[0].y);
      if (n > 2)
        dst[2] = __half2float(value.h[1].x);
    }
  }

  DEVICE_INLINE void store(__half *dst, int n) {
    if (n == 4) {
      *(reinterpret_cast<float2 *>(dst)) = value.f;
    } else {
      if (n > 0)
        dst[0] = value.h[0].x;
      if (n > 1)
        dst[1] = value.h[0].y;
      if (n > 2)
        dst[2] = value.h[1].x;
    }
  }

  DEVICE_INLINE void store(__nv_bfloat16 *dst, int n) {
    if (n == 4) {
      union {
        float2 f;
        __nv_bfloat162 h[2];
      } tmp;
      tmp.h[0].x =
          TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[0].x);
      tmp.h[0].y =
          TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[0].y);
      tmp.h[1].x =
          TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[1].x);
      tmp.h[1].y =
          TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[1].y);
      *(reinterpret_cast<float2 *>(dst)) = tmp.f;
    } else {
      if (n > 0)
        dst[0] = TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[0].x);
      if (n > 1)
        dst[1] = TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[0].y);
      if (n > 2)
        dst[2] = TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[1].x);
    }
  }

  DEVICE_INLINE void store(float *dst) {
    float4 f;
    f.x = __half2float(value.h[0].x);
    f.y = __half2float(value.h[0].y);
    f.z = __half2float(value.h[1].x);
    f.w = __half2float(value.h[1].y);
    *(reinterpret_cast<float4 *>(dst)) = f;
  }

  DEVICE_INLINE void store(__half *dst) {
    *(reinterpret_cast<float2 *>(dst)) = value.f;
  }

  DEVICE_INLINE void store(__nv_bfloat16 *dst) {
    union {
      float2 f;
      __nv_bfloat162 h[2];
    } tmp;
    tmp.h[0].x = TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[0].x);
    tmp.h[0].y = TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[0].y);
    tmp.h[1].x = TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[1].x);
    tmp.h[1].y = TypeConvertFunc<__nv_bfloat16, __half>::convert(value.h[1].y);
    *(reinterpret_cast<float2 *>(dst)) = tmp.f;
  }
};

template <> struct Vec4T<__nv_bfloat16> {
  union U {
    float2 f;
    __nv_bfloat162 h[2];
  } value;

  DEVICE_INLINE Vec4T() {
    value.h[0].x = 0.f;
    value.h[0].y = 0.f;
    value.h[1].x = 0.f;
    value.h[1].y = 0.f;
  }

  DEVICE_INLINE void reset() {
    value.h[0].x = 0.f;
    value.h[0].y = 0.f;
    value.h[1].x = 0.f;
    value.h[1].y = 0.f;
  }

  DEVICE_INLINE void reset(const __nv_bfloat16 initial_value) {
    value.h[0].x = initial_value;
    value.h[0].y = initial_value;
    value.h[1].x = initial_value;
    value.h[1].y = initial_value;
  }

  DEVICE_INLINE void load(const float *p, int n) {
    if (n == 4) {
      float4 f = *(reinterpret_cast<const float4 *>(p));
      float2 firstf{f.x, f.y};
      float2 secondf{f.z, f.w};
      value.h[0] = __float22bfloat162_rn(firstf);
      value.h[1] = __float22bfloat162_rn(secondf);
    } else {
      if (n > 0)
        value.h[0].x = __float2bfloat16(p[0]);
      if (n > 1)
        value.h[0].y = __float2bfloat16(p[1]);
      if (n > 2)
        value.h[1].x = __float2bfloat16(p[2]);
    }
  }

  DEVICE_INLINE void load(const __half *p, int n) {
    if (n == 4) {
      float2 f = *(reinterpret_cast<const float2 *>(p));
      value.h[0] = __float2bfloat162_rn(f.x);
      value.h[1] = __float2bfloat162_rn(f.y);
    } else {
      if (n > 0)
        value.h[0].x = TypeConvertFunc<__nv_bfloat16, __half>::convert(p[0]);
      if (n > 1)
        value.h[0].y = TypeConvertFunc<__nv_bfloat16, __half>::convert(p[1]);
      if (n > 2)
        value.h[1].x = TypeConvertFunc<__nv_bfloat16, __half>::convert(p[2]);
    }
  }

  DEVICE_INLINE void load(const __nv_bfloat16 *p, int n) {
    if (n == 4) {
      value.f = *(reinterpret_cast<const float2 *>(p));
    } else {
      if (n > 0)
        value.h[0].x = p[0];
      if (n > 1)
        value.h[0].y = p[1];
      if (n > 2)
        value.h[1].x = p[2];
    }
  }

  DEVICE_INLINE void load(const float *p) {
    float4 f = *(reinterpret_cast<const float4 *>(p));
    float2 firstf{f.x, f.y};
    float2 secondf{f.z, f.w};
    value.h[0] = __float22bfloat162_rn(firstf);
    value.h[1] = __float22bfloat162_rn(secondf);
  }

  DEVICE_INLINE void load(const __half *p) {
    float2 f = *(reinterpret_cast<const float2 *>(p));
    value.h[0] = __float2bfloat162_rn(f.x);
    value.h[1] = __float2bfloat162_rn(f.y);
  }

  DEVICE_INLINE void load(const __nv_bfloat16 *p) {
    value.f = *(reinterpret_cast<const float2 *>(p));
  }

  DEVICE_INLINE void atomic_store_accum(float *dst, int n) {
    if (n > 0)
      atomicAdd(dst, __bfloat162float(value.h[0].x));
    if (n > 1)
      atomicAdd(dst + 1, __bfloat162float(value.h[0].y));
    if (n > 2)
      atomicAdd(dst + 2, __bfloat162float(value.h[1].x));
    if (n > 3)
      atomicAdd(dst + 3, __bfloat162float(value.h[1].y));
  }

  DEVICE_INLINE void atomic_store_accum(__half *dst, int n) {
    if (n == 4) {
      __half2 h0 = __half2(
          TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[0].x),
          TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[0].y));
      __half2 h1 = __half2(
          TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[1].x),
          TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[1].y));
      atomicAdd((reinterpret_cast<__half2 *>(dst)), h0);
      atomicAdd((reinterpret_cast<__half2 *>(dst + 2)), h1);
    } else {
      if (n > 0)
        atomicAdd(
            dst, TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[0].x));
      if (n > 1)
        atomicAdd(dst + 1, TypeConvertFunc<__half, __nv_bfloat16>::convert(
                               value.h[0].y));
      if (n > 2)
        atomicAdd(dst + 2, TypeConvertFunc<__half, __nv_bfloat16>::convert(
                               value.h[1].x));
    }
  }

  DEVICE_INLINE void atomic_store_accum(__nv_bfloat16 *dst, int n) {
    if (n == 4) {
      atomicAdd((reinterpret_cast<__nv_bfloat162 *>(dst)), value.h[0]);
      atomicAdd((reinterpret_cast<__nv_bfloat162 *>(dst + 2)), value.h[1]);
    } else {
      if (n > 0)
        atomicAdd(dst, value.h[0].x);
      if (n > 1)
        atomicAdd(dst + 1, value.h[0].y);
      if (n > 2)
        atomicAdd(dst + 2, value.h[1].x);
    }
  }

  DEVICE_INLINE void store(float *dst, int n) {
    if (n == 4) {
      float4 f;
      f.x = __bfloat162float(value.h[0].x);
      f.y = __bfloat162float(value.h[0].y);
      f.z = __bfloat162float(value.h[1].x);
      f.w = __bfloat162float(value.h[1].y);
      *(reinterpret_cast<float4 *>(dst)) = f;
    } else {
      if (n > 0)
        dst[0] = __bfloat162float(value.h[0].x);
      if (n > 1)
        dst[1] = __bfloat162float(value.h[0].y);
      if (n > 2)
        dst[2] = __bfloat162float(value.h[1].x);
    }
  }

  DEVICE_INLINE void store(__half *dst, int n) {
    if (n == 4) {
      union {
        float2 f;
        __half2 h[2];
      } tmp;
      __half2 h[2];
      tmp.h[0].x =
          TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[0].x);
      tmp.h[0].y =
          TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[0].y);
      tmp.h[1].x =
          TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[1].x);
      tmp.h[1].y =
          TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[1].y);
      *(reinterpret_cast<float2 *>(dst)) = tmp.f;
    } else {
      if (n > 0)
        dst[0] = TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[0].x);
      if (n > 1)
        dst[1] = TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[0].y);
      if (n > 2)
        dst[2] = TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[1].x);
    }
  }

  DEVICE_INLINE void store(__nv_bfloat16 *dst, int n) {
    if (n == 4) {
      *(reinterpret_cast<float2 *>(dst)) = value.f;
    } else {
      if (n > 0)
        dst[0] = value.h[0].x;
      if (n > 1)
        dst[1] = value.h[0].y;
      if (n > 2)
        dst[2] = value.h[1].x;
    }
  }

  DEVICE_INLINE void store(float *dst) {
    float4 f;
    f.x = __bfloat162float(value.h[0].x);
    f.y = __bfloat162float(value.h[0].y);
    f.z = __bfloat162float(value.h[1].x);
    f.w = __bfloat162float(value.h[1].y);
    *(reinterpret_cast<float4 *>(dst)) = f;
  }

  DEVICE_INLINE void store(__half *dst) {
    union {
      float2 f;
      __half2 h[2];
    } tmp;
    tmp.h[0].x = TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[0].x);
    tmp.h[0].y = TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[0].y);
    tmp.h[1].x = TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[1].x);
    tmp.h[1].y = TypeConvertFunc<__half, __nv_bfloat16>::convert(value.h[1].y);
    *(reinterpret_cast<float2 *>(dst)) = tmp.f;
  }

  DEVICE_INLINE void store(__nv_bfloat16 *dst) {
    *(reinterpret_cast<float2 *>(dst)) = value.f;
  }
};

template <> struct Vec4T<float> {
  float4 val;

  DEVICE_INLINE Vec4T() {
    val.x = 0.f;
    val.y = 0.f;
    val.z = 0.f;
    val.w = 0.f;
  }

  DEVICE_INLINE void reset() {
    val.x = 0.f;
    val.y = 0.f;
    val.z = 0.f;
    val.w = 0.f;
  }

  DEVICE_INLINE void reset(const float initial_value) {
    val.x = initial_value;
    val.y = initial_value;
    val.z = initial_value;
    val.w = initial_value;
  }

  DEVICE_INLINE void load(const float *p, int n) {
    if (n == 4) {
      val = *((const float4 *)p);
    } else {
      if (n > 0)
        val.x = p[0];
      if (n > 1)
        val.y = p[1];
      if (n > 2)
        val.z = p[2];
    }
  }

  DEVICE_INLINE void load(const __half *p, int n) {
    if (n == 4) {
      Vec4T<__half> h;
      h.load(p, n);
      val.x = __half2float(h.value.h[0].x);
      val.y = __half2float(h.value.h[0].y);
      val.z = __half2float(h.value.h[1].x);
      val.w = __half2float(h.value.h[1].y);
    } else {
      if (n > 0)
        val.x = __half2float(p[0]);
      if (n > 1)
        val.y = __half2float(p[1]);
      if (n > 2)
        val.z = __half2float(p[2]);
    }
  }

  DEVICE_INLINE void load(const __nv_bfloat16 *p, int n) {
    if (n == 4) {
      Vec4T<__nv_bfloat16> h;
      h.load(p);
      val.x = __bfloat162float(h.value.h[0].x);
      val.y = __bfloat162float(h.value.h[0].y);
      val.z = __bfloat162float(h.value.h[1].x);
      val.w = __bfloat162float(h.value.h[1].y);
    } else {
      if (n > 0)
        val.x = __bfloat162float(p[0]);
      if (n > 1)
        val.y = __bfloat162float(p[1]);
      if (n > 2)
        val.z = __bfloat162float(p[2]);
    }
  }

  DEVICE_INLINE void load(const float *p) { val = *((const float4 *)p); }

  DEVICE_INLINE void load(const __half *p) {
    Vec4T<__half> h;
    h.load(p);
    val.x = __half2float(h.value.h[0].x);
    val.y = __half2float(h.value.h[0].y);
    val.z = __half2float(h.value.h[1].x);
    val.w = __half2float(h.value.h[1].y);
  }

  DEVICE_INLINE void load(const __nv_bfloat16 *p) {
    Vec4T<__nv_bfloat16> h;
    h.load(p);
    val.x = __bfloat162float(h.value.h[0].x);
    val.y = __bfloat162float(h.value.h[0].y);
    val.z = __bfloat162float(h.value.h[1].x);
    val.w = __bfloat162float(h.value.h[1].y);
  }

  DEVICE_INLINE void store(float *dst, int n) {
    if (n == 4) {
      *(reinterpret_cast<float4 *>(dst)) = val;
    } else {
      if (n > 0)
        dst[0] = val.x;
      if (n > 1)
        dst[1] = val.y;
      if (n > 2)
        dst[2] = val.z;
    }
  }

  DEVICE_INLINE void store(__half *dst, int n) {
    if (n == 4) {
      Vec4T<__half> h;
      h.load(reinterpret_cast<float *>(&val), 4);
      h.store(dst, 4);
    } else {
      if (n > 0)
        dst[0] = __float2half(val.x);
      if (n > 1)
        dst[1] = __float2half(val.y);
      if (n > 2)
        dst[2] = __float2half(val.z);
    }
  }

  DEVICE_INLINE void store(__nv_bfloat16 *dst, int n) {
    if (n == 4) {
      Vec4T<__nv_bfloat16> h;
      h.load(reinterpret_cast<float *>(&val), 4);
      h.store(dst, 4);
    } else {
      if (n > 0)
        dst[0] = __float2bfloat16(val.x);
      if (n > 1)
        dst[1] = __float2bfloat16(val.y);
      if (n > 2)
        dst[2] = __float2bfloat16(val.z);
    }
  }

  DEVICE_INLINE void atomic_store_accum(float *dst, int n) {
    if (n > 0)
      atomicAdd(dst, val.x);
    if (n > 1)
      atomicAdd(dst + 1, val.y);
    if (n > 2)
      atomicAdd(dst + 2, val.z);
    if (n > 3)
      atomicAdd(dst + 3, val.w);
  }

  DEVICE_INLINE void atomic_store_accum(__half *dst, int n) {
    if (n == 4) {
      __half2 tmp1;
      __half2 tmp2;
      tmp1.x = __float2half(val.x);
      tmp1.y = __float2half(val.y);
      tmp2.x = __float2half(val.z);
      tmp2.y = __float2half(val.w);
      atomicAdd((reinterpret_cast<__half2 *>(dst)), tmp1);
      atomicAdd((reinterpret_cast<__half2 *>(dst + 2)), tmp2);
    } else {
      if (n > 0)
        atomicAdd(dst, __float2half(val.x));
      if (n > 1)
        atomicAdd(dst + 1, __float2half(val.y));
      if (n > 2)
        atomicAdd(dst + 2, __float2half(val.z));
    }
  }

  DEVICE_INLINE void atomic_store_accum(__nv_bfloat16 *dst, int n) {
    if (n == 4) {
      __nv_bfloat162 tmp1;
      __nv_bfloat162 tmp2;
      tmp1.x = __float2bfloat16(val.x);
      tmp1.y = __float2bfloat16(val.y);
      tmp2.x = __float2bfloat16(val.z);
      tmp2.y = __float2bfloat16(val.w);
      atomicAdd(reinterpret_cast<__nv_bfloat162 *>(dst), tmp1);
      atomicAdd(reinterpret_cast<__nv_bfloat162 *>(dst + 2), tmp2);
    } else {
      if (n > 0)
        atomicAdd(dst, __float2bfloat16(val.x));
      if (n > 1)
        atomicAdd(dst + 1, __float2bfloat16(val.y));
      if (n > 2)
        atomicAdd(dst + 2, __float2bfloat16(val.z));
    }
  }

  DEVICE_INLINE void store(float *dst) {
    *(reinterpret_cast<float4 *>(dst)) = val;
  }

  DEVICE_INLINE void store(__half *dst) {
    Vec4T<__half> h;
    h.load(reinterpret_cast<float *>(&val), 4);
    h.store(dst, 4);
  }

  DEVICE_INLINE void store(__nv_bfloat16 *dst) {
    Vec4T<__nv_bfloat16> h;
    h.load(reinterpret_cast<float *>(&val), 4);
    h.store(dst, 4);
  }

  DEVICE_INLINE void accumulate(const Vec4T<float> &other) {
    val.x += other.val.x;
    val.y += other.val.y;
    val.z += other.val.z;
    val.w += other.val.w;
  }

  DEVICE_INLINE void accumulate(const Vec4T<__half> &other) {
    val.x += __half2float(other.value.h[0].x);
    val.y += __half2float(other.value.h[0].y);
    val.z += __half2float(other.value.h[1].x);
    val.w += __half2float(other.value.h[1].y);
  }

  DEVICE_INLINE void accumulate(const Vec4T<__nv_bfloat16> &other) {
    val.x += __bfloat162float(other.value.h[0].x);
    val.y += __bfloat162float(other.value.h[0].y);
    val.z += __bfloat162float(other.value.h[1].x);
    val.w += __bfloat162float(other.value.h[1].y);
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<float> &other,
                                         float weight) {
    val.x += (other.val.x * weight);
    val.y += (other.val.y * weight);
    val.z += (other.val.z * weight);
    val.w += (other.val.w * weight);
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<float> &other,
                                         __half weight) {
    val.x += (other.val.x * __half2float(weight));
    val.y += (other.val.y * __half2float(weight));
    val.z += (other.val.z * __half2float(weight));
    val.w += (other.val.w * __half2float(weight));
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<float> &other,
                                         __nv_bfloat16 weight) {
    val.x += (other.val.x * __bfloat162float(weight));
    val.y += (other.val.y * __bfloat162float(weight));
    val.z += (other.val.z * __bfloat162float(weight));
    val.w += (other.val.w * __bfloat162float(weight));
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<__half> &other,
                                         float weight) {
    val.x += (__half2float(other.value.h[0].x) * weight);
    val.y += (__half2float(other.value.h[0].y) * weight);
    val.z += (__half2float(other.value.h[1].x) * weight);
    val.w += (__half2float(other.value.h[1].y) * weight);
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<__half> &other,
                                         __half weight) {
    val.x += (__half2float(other.value.h[0].x) * __half2float(weight));
    val.y += (__half2float(other.value.h[0].y) * __half2float(weight));
    val.z += (__half2float(other.value.h[1].x) * __half2float(weight));
    val.w += (__half2float(other.value.h[1].y) * __half2float(weight));
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<__half> &other,
                                         __nv_bfloat16 weight) {
    val.x += (__half2float(other.value.h[0].x) * __bfloat162float(weight));
    val.y += (__half2float(other.value.h[0].y) * __bfloat162float(weight));
    val.z += (__half2float(other.value.h[1].x) * __bfloat162float(weight));
    val.w += (__half2float(other.value.h[1].y) * __bfloat162float(weight));
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<__nv_bfloat16> &other,
                                         float weight) {
    val.x += (__bfloat162float(other.value.h[0].x) * weight);
    val.y += (__bfloat162float(other.value.h[0].y) * weight);
    val.z += (__bfloat162float(other.value.h[1].x) * weight);
    val.w += (__bfloat162float(other.value.h[1].y) * weight);
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<__nv_bfloat16> &other,
                                         __half weight) {
    val.x += (__bfloat162float(other.value.h[0].x) * __half2float(weight));
    val.y += (__bfloat162float(other.value.h[0].y) * __half2float(weight));
    val.z += (__bfloat162float(other.value.h[1].x) * __half2float(weight));
    val.w += (__bfloat162float(other.value.h[1].y) * __half2float(weight));
  }

  DEVICE_INLINE void accumulate_multiply(const Vec4T<__nv_bfloat16> &other,
                                         __nv_bfloat16 weight) {
    val.x += (__bfloat162float(other.value.h[0].x) * __bfloat162float(weight));
    val.y += (__bfloat162float(other.value.h[0].y) * __bfloat162float(weight));
    val.z += (__bfloat162float(other.value.h[1].x) * __bfloat162float(weight));
    val.w += (__bfloat162float(other.value.h[1].y) * __bfloat162float(weight));
  }
};

template <typename T>
HOST_DEVICE_INLINE int64_t bs_upper_bound_sub_one(const T *const arr,
                                                  int64_t num, T target) {
  int64_t start = 0;
  int64_t end = num;
  while (start < end) {
    int64_t middle = start + (end - start) / 2;
    T value = arr[middle];
    if (value <= target) {
      start = middle + 1;
    } else {
      end = middle;
    }
  }
  return (start == num && arr[start - 1] != target) ? num : start - 1;
}

template <typename CopyDesc>
__global__ void one_to_one_warp_per_ev_kernel(CopyDesc copy_desc) {
  using vec_length_type = int;
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;

  const vec_length_type ev_size = copy_desc.get_vec_length();
  for (int i_ev = blockIdx.x; i_ev < copy_desc.num_vec_; i_ev += gridDim.x) {
    auto *dst_ptr = copy_desc.get_dst_ptr(i_ev);
    auto *src_ptr = copy_desc.get_src_ptr(i_ev);
    for (int i = threadIdx.x; i < ev_size; i += blockDim.x) {
      dst_ptr[i] = TypeConvertFunc<dst_type, src_type>::convert(src_ptr[i]);
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void one_to_one_warp_per_ev_vec4_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;
  const vec_length_type ev_size = copy_desc.get_vec_length();

  Vec4T<src_type> src_elem;
  for (int i_ev = blockIdx.x * blockDim.y + warp_id; i_ev < copy_desc.num_vec_;
       i_ev += (gridDim.x * blockDim.y)) {

    const src_type *src_ptr = copy_desc.get_src_ptr(i_ev);
    dst_type *dst_ptr = copy_desc.get_dst_ptr(i_ev);

#pragma unroll kMaxElemPerThread
    for (int i = 0;
         i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < ev_size;
         ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(ev_size - idx4, copy_width);
      src_elem.load(src_ptr + idx4, n);
      src_elem.store(dst_ptr + idx4, n);
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void multi_to_one_cta_per_ev_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;
  int i_ev = blockIdx.x;

  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    int average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);
    dst_type *dst_ev = copy_desc.get_dst_ptr(i_ev);

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);

    float accum[kMaxElemPerThread] = {0.f};
    for (int r = 0; r < (end - start); ++r) {
      const src_type *src_ev = copy_desc.get_src_ptr(r + start);
#pragma unroll kMaxElemPerThread
      for (int i = 0;
           i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < vec_length;
           ++i) {
        accum[i] += (float)(src_ev[blockDim.x * i + threadIdx.x]);
      }
    }
    if (average_pooling_factor > 0) {
#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread; ++i) {
        accum[i] /= average_pooling_factor;
      }
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0;
         i < kMaxElemPerThread && blockDim.x * i + threadIdx.x < vec_length;
         ++i) {
      dst_ev[blockDim.x * i + threadIdx.x] = (dst_type)(accum[i]);
    }
  }
}

template <typename CopyDesc, int kMaxElemPerThread>
__global__ void multi_to_one_warp_per_ev_vec4_kernel(CopyDesc copy_desc) {
  using src_type = typename CopyDesc::SrcT;
  using dst_type = typename CopyDesc::DstT;
  using vec_length_type = int;

  constexpr int copy_width = 4;
  constexpr int kWarpSize = 32;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int i_ev = blockIdx.x * blockDim.y + warp_id;
  if (i_ev < copy_desc.num_vec_) {
    vec_length_type vec_length = copy_desc.get_vec_length(i_ev);
    int average_pooling_factor = copy_desc.get_average_pooling_factor(i_ev);

    int start = copy_desc.get_offset(i_ev);
    int end = copy_desc.get_offset(i_ev + 1);

    dst_type *dst_ev = copy_desc.get_dst_ptr(i_ev);

    Vec4T<float> accum[kMaxElemPerThread];
    int L = end - start;
    for (int r = 0; r < L; r += kWarpSize) {
      int l = r + lane_id < L ? start + r + lane_id : 0;

      for (int j = 0; j < kWarpSize && r + j < L; ++j) {
        int j_ev = __shfl_sync(0xFFFFFFFF, l, j);
        const src_type *src_ev = copy_desc.get_src_ptr(j_ev);

#pragma unroll kMaxElemPerThread
        for (int i = 0; i < kMaxElemPerThread &&
                        4 * kWarpSize * i + 4 * lane_id < vec_length;
             ++i) {
          Vec4T<src_type> src_elem;
          int idx4 = 4 * kWarpSize * i + 4 * lane_id;
          int n = min(vec_length - idx4, copy_width);
          src_elem.load(src_ev + idx4, n);
          accum[i].accumulate(src_elem);
        }
      }
    }

    if (average_pooling_factor > 0) {
#pragma unroll kMaxElemPerThread
      for (int i = 0; i < kMaxElemPerThread; ++i) {
        accum[i].val.x /= average_pooling_factor;
        accum[i].val.y /= average_pooling_factor;
        accum[i].val.z /= average_pooling_factor;
        accum[i].val.w /= average_pooling_factor;
      }
    }

#pragma unroll kMaxElemPerThread
    for (int i = 0;
         i < kMaxElemPerThread && 4 * kWarpSize * i + 4 * lane_id < vec_length;
         ++i) {
      int idx4 = 4 * kWarpSize * i + 4 * lane_id;
      int n = min(vec_length - idx4, copy_width);
      accum[i].store(dst_ev + idx4, n);
    }
  }
}

template <typename CopyDesc>
void copy_multi_to_one(CopyDesc &copy_desc, int ev_size, cudaStream_t stream) {
  if (copy_desc.num_vec_ == 0)
    return;
  if (ev_size % 4 != 0 || copy_desc.accum_D % 4 != 0 ||
      copy_desc.total_D % 4 != 0) {
    //  need to optimize for small ev_size
    constexpr int MAX_THREADS_PER_BLOCK = 1024;
    int grid_dim = copy_desc.num_vec_;
    int block_dim =
        ev_size < MAX_THREADS_PER_BLOCK ? ev_size : MAX_THREADS_PER_BLOCK;

    multi_to_one_cta_per_ev_kernel<CopyDesc, 1>
        <<<grid_dim, block_dim, 0, stream>>>(copy_desc);
  } else {
    if (ev_size <= 128) {
      int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
      dim3 block_size{32, 2};
      multi_to_one_warp_per_ev_vec4_kernel<CopyDesc, 1>
          <<<grid_size, block_size, 0, stream>>>(copy_desc);
    } else if (ev_size <= 256) {
      int grid_size = (copy_desc.num_vec_ - 1) / 2 + 1;
      dim3 block_size{32, 2};
      multi_to_one_warp_per_ev_vec4_kernel<CopyDesc, 2>
          <<<grid_size, block_size, 0, stream>>>(copy_desc);
    } else if (ev_size <= 1024) {
      multi_to_one_cta_per_ev_kernel<CopyDesc, 1>
          <<<copy_desc.num_vec_, ev_size, 0, stream>>>(copy_desc);
    } else {
      throw std::runtime_error(
          "dyn emb does not support emb vector size > 1024");
    }
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename CopyDesc>
void copy_one_to_one(CopyDesc &copy_desc, int ev_size, int num_sms,
                     cudaStream_t stream) {
  if (copy_desc.num_vec_ == 0)
    return;
  constexpr int MAX_THREADS_PER_BLOCK = 1024;
  if (ev_size % 4 != 0) {
    //  need to optimize for small ev_size
    int grid_dim = copy_desc.num_vec_;
    int block_dim =
        ev_size < MAX_THREADS_PER_BLOCK ? ev_size : MAX_THREADS_PER_BLOCK;

    one_to_one_warp_per_ev_kernel<CopyDesc>
        <<<grid_dim, block_dim, 0, stream>>>(copy_desc);
  } else {
    if (ev_size <= 128) {
      int grid_size = num_sms * 32; // 2048/64 =32
      if (copy_desc.num_vec_ < grid_size)
        grid_size = copy_desc.num_vec_;
      dim3 block_size{32, 2};
      one_to_one_warp_per_ev_vec4_kernel<CopyDesc, 1>
          <<<grid_size, block_size, 0, stream>>>(copy_desc);
    } else if (ev_size <= 256) {
      int grid_size = num_sms * 32; // 2048/64 =32
      if (copy_desc.num_vec_ < grid_size)
        grid_size = copy_desc.num_vec_;
      dim3 block_size{32, 2};
      one_to_one_warp_per_ev_vec4_kernel<CopyDesc, 2>
          <<<grid_size, block_size, 0, stream>>>(copy_desc);
    } else if (ev_size <= 1024) {
      int grid_dim = copy_desc.num_vec_;
      int block_dim =
          ev_size < MAX_THREADS_PER_BLOCK ? ev_size : MAX_THREADS_PER_BLOCK;
      one_to_one_warp_per_ev_kernel<CopyDesc>
          <<<grid_dim, block_dim, 0, stream>>>(copy_desc);
    } else {
      throw std::runtime_error(
          "dynamic emb does not support emb vector size > 1024");
    }
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename src_t, typename dst_t>
__global__ void add_offset_kernel(const src_t *src, dst_t *dst, int idx) {
  dst[idx + 1] = dst[idx] + src[idx];
}

template <typename offset_t, typename length_t>
__global__ void get_new_length_and_offsets_kernel(
    uint64_t *d_unique_offsets, int64_t *d_table_offsets_in_feature,
    int table_num, int64_t new_lengths_size, int local_batch_size,
    offset_t *new_offsets, length_t *new_lenghths) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= new_lengths_size)
    return;
  int feature_id = i / local_batch_size;
  int64_t table_id =
      bs_upper_bound_sub_one(d_table_offsets_in_feature,
                             (int64_t)(table_num + 1), (int64_t)feature_id);

  int64_t table_buckets = (d_table_offsets_in_feature[table_id + 1] -
                           d_table_offsets_in_feature[table_id]) *
                          local_batch_size;
  int64_t bucket_id =
      i - (d_table_offsets_in_feature[table_id]) * local_batch_size;
  uint64_t unique_num =
      d_unique_offsets[table_id + 1] - d_unique_offsets[table_id];

  uint64_t bucket_value = unique_num / table_buckets;
  uint64_t remainder = unique_num % table_buckets;
  uint64_t tmp_length = bucket_value;
  uint64_t tmp_offset = d_unique_offsets[table_id];
  if (bucket_id < remainder)
    tmp_length += 1;
  tmp_offset += (bucket_id)*bucket_value +
                (bucket_id < remainder ? bucket_id : remainder);
  ;

  new_lenghths[i] = tmp_length;
  new_offsets[i] = tmp_offset;
  if (i == new_lengths_size - 1) {
    new_offsets[new_lengths_size] = tmp_offset + bucket_value;
  }
}
} // namespace dyn_emb
#endif // LOOKUP_KERNEL_CUH
