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

// Runtime loader/linker for the LruLfu eviction cubins. The default (Lex)
// evictor is a prebuilt complete fatbin loaded directly; a custom score_function
// is a numba-compiled LTO-IR linked into the custom fatbin via nvJitLink. Both
// expose the dyn_emb_evict_entry_{ovf,noovf} entries and are launched the same
// way. Includable by both nvcc (insert_and_evict.cu routing) and the host
// compiler (jit_link.cpp) -- only cuda.h + the POD EvictParams.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>

#include "evict_abi.cuh"

namespace dyn_emb {

// Register the default (Lex) evictor fatbin bytes (complete SASS). Called once
// from Python for any LruLfu table; no numba, and the custom fatbin is NOT
// needed for the default eviction path.
void demb_set_lex_fatbin(const void *lex, size_t lex_size);

// Link a numba-compiled user_score_fn (LTO-IR) into the custom evict fatbin
// (passed here only when a score_function is actually used) for the given device
// compute capability, and cache the resulting entries under `key`. Idempotent
// per key. Throws std::runtime_error on link/load failure.
void demb_register_score_function(int64_t key, const void *ltoir,
                                  size_t ltoir_size, const void *cust,
                                  size_t cust_size, int cc_major, int cc_minor);

// Evict entry CUfunction for (key, overflow). key == 0 selects the default Lex
// evictor (loaded lazily from the packaged fatbin; no numba). key != 0 must have
// been registered via demb_register_score_function.
CUfunction demb_get_evict_fn(int64_t key, bool overflow);

// Launch the evict kernel over `batch` keys (block 256), passing `params` as the
// single by-value argument, on `stream`. Throws on launch error.
void demb_launch_evict(CUfunction fn, EvictParams params, int64_t batch,
                       CUstream stream);

} // namespace dyn_emb
