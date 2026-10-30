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
// Loads, stores, reductions, atomics and fences with an explicit memory order
// and scope (inline PTX).
#pragma once

#include <cstdint>

namespace dyn_emb {
namespace hier_output_dist {

// Flags that another GPU writes are read with system-scope acquire, and the
// data they guard with ld.global.cg (never a stale L1 line); words only this
// GPU's CTAs write, with gpu-scope acquire.
__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t *addr) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(addr) : "memory");
  return value;
}
__device__ __forceinline__ uint64_t ld_acquire_gpu(const uint64_t *addr) {
  uint64_t value;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"(addr) : "memory");
  return value;
}
__device__ __forceinline__ void st_release_sys(uint64_t *addr, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
}
__device__ __forceinline__ void red_release_sys(uint64_t *addr, uint64_t value) {
  asm volatile("red.release.sys.global.add.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
}
__device__ __forceinline__ void red_release_gpu(uint64_t *addr, uint64_t value) {
  asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
}
// A word the warps of one CTA share (generic address, shared memory included).
__device__ __forceinline__ uint64_t ld_acquire_cta(const uint64_t *addr) {
  uint64_t value;
  asm volatile("ld.acquire.cta.u64 %0, [%1];" : "=l"(value) : "l"(addr) : "memory");
  return value;
}
__device__ __forceinline__ void red_max_release_cta(uint64_t *addr, uint64_t value) {
  asm volatile("red.release.cta.max.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
}
__device__ __forceinline__ uint64_t atom_add_acq_rel_gpu(uint64_t *addr, uint64_t value) {
  uint64_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;"
               : "=l"(old)
               : "l"(addr), "l"(value)
               : "memory");
  return old;
}
__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }

}  // namespace hier_output_dist
}  // namespace dyn_emb
