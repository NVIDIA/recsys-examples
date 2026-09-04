/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

namespace dyn_emb {

// MurmurHash3's 64-bit finalizer, and nothing else: the avalanche step that
// spreads a key's bits before something narrows them down.
//
// Two callers narrow it differently -- picking an owning rank takes it modulo
// the world size, picking a hash bucket masks it to a non-negative int64 -- and
// those tails belong to the callers, not here. Only the avalanche is shared,
// which is why this header pulls in nothing but <cstdint>.
//
// Anything that has to agree with this on the host (``murmur3_fmix64`` in
// dynamicemb/scored_hashtable.py) is a translation of exactly this function.
__host__ __device__ __forceinline__ uint64_t murmur3_fmix64(uint64_t key) {
  key ^= key >> 33;
  key *= UINT64_C(0xff51afd7ed558ccd);
  key ^= key >> 33;
  key *= UINT64_C(0xc4ceb9fe1a85ec53);
  key ^= key >> 33;
  return key;
}

} // namespace dyn_emb
