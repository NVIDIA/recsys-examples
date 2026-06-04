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

#include "check.h"
#include "lookup_forward.h"
#include "torch_utils.h"
#include "unique_op.h"
#include "utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
namespace cg = cooperative_groups;
#ifdef DEMB_USE_PYBIND11
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#endif

#include <cassert>
#include <cstdlib>
#include <limits>

#ifdef DEMB_USE_PYBIND11
namespace py = pybind11;
#endif

namespace dyn_emb {

constexpr int BLOCK_SIZE = 256;

// MurmurHash3_32 hash function
template <typename Key, uint32_t m_seed = 0> struct MurmurHash3_32 {
  __forceinline__ __host__ __device__ static uint32_t rotl32(uint32_t x,
                                                             int8_t r) {
    return (x << r) | (x >> (32 - r));
  }

  __forceinline__ __host__ __device__ static uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  __forceinline__ __host__ __device__ static uint32_t hash(const Key &key) {
    constexpr int len = sizeof(Key);
    const uint8_t *const data = reinterpret_cast<const uint8_t *>(&key);
    constexpr int nblocks = len / 4;
    uint32_t h1 = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;

    const uint32_t *const blocks =
        reinterpret_cast<const uint32_t *>(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
      uint32_t k1 = blocks[i];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }

    const uint8_t *tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
      [[fallthrough]];
    case 2:
      k1 ^= tail[1] << 8;
      [[fallthrough]];
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
    }

    h1 ^= len;
    return fmix32(h1);
  }

  // Combine two hash values (for compound keys)
  __forceinline__ __host__ __device__ static uint32_t
  hash_combine(uint32_t h1, uint32_t h2) {
    h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    return h1;
  }
};

// Atomic operation overloads for 64-bit types
__forceinline__ __device__ long atomicAdd(long *address, long val) {
  return static_cast<long>(
      ::atomicAdd(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(val)));
}

__forceinline__ __device__ long long atomicAdd(long long *address,
                                               long long val) {
  return static_cast<long long>(
      ::atomicAdd(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(val)));
}

__forceinline__ __device__ unsigned long atomicAdd(unsigned long *address,
                                                   unsigned long val) {
  return static_cast<unsigned long>(
      ::atomicAdd(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(val)));
}

__forceinline__ __device__ uint64_t atomicCAS(uint64_t *address,
                                              uint64_t compare, uint64_t val) {
  return static_cast<uint64_t>(
      ::atomicCAS(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(compare),
                  static_cast<unsigned long long>(val)));
}

__forceinline__ __device__ int64_t atomicCAS(int64_t *address, int64_t compare,
                                             int64_t val) {
  return static_cast<int64_t>(
      ::atomicCAS(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(compare),
                  static_cast<unsigned long long>(val)));
}

// Type dispatch helper
template <typename Func>
void dispatch_key_type(at::ScalarType key_type, Func &&func) {
  if (key_type == at::kLong) {
    func.template operator()<int64_t>();
  } else if (key_type == at::kUInt64) {
    func.template operator()<uint64_t>();
  } else {
    throw std::invalid_argument(
        "Unsupported key dtype: must be int64 or uint64");
  }
}

// ============================================================================
// Segmented Unique Implementation
// ============================================================================

// ============================================================================
// Packed value encoding for segmented unique
// ============================================================================
// Pack table_id (high 32 bits) and local_unique_idx (low 32 bits) into int64_t
// This allows us to use only 2 arrays (hash_keys, hash_vals) instead of 3

__device__ __forceinline__ int64_t pack_table_val(int64_t table_id,
                                                  int32_t local_idx) {
  // Use uint32_t cast to avoid sign extension issues
  return (static_cast<int64_t>(static_cast<int32_t>(table_id)) << 32) |
         static_cast<uint32_t>(local_idx);
}

__device__ __forceinline__ int64_t unpack_table_id(int64_t packed) {
  return static_cast<int64_t>(static_cast<int32_t>(packed >> 32));
}

__device__ __forceinline__ int32_t unpack_local_idx(int64_t packed) {
  return static_cast<int32_t>(packed & 0xFFFFFFFF);
}

// Initialize segmented hash table kernel (strided loop version)
// Uses packed (table_id, local_idx) in hash_vals for memory efficiency
template <typename KeyType,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__global__ void segmented_init_kernel(KeyType *hash_keys, int64_t *hash_vals,
                                      int64_t *table_counters, size_t capacity,
                                      int64_t num_tables) {
  const size_t stride = blockDim.x * gridDim.x;
  // Initialize hash table entries
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < capacity;
       idx += stride) {
    hash_keys[idx] = empty_key;
    hash_vals[idx] = empty_val;
  }
  // Initialize per-table counters
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < static_cast<size_t>(num_tables); idx += stride) {
    table_counters[idx] = 0;
  }
}

// Segmented unique kernel - deduplicates (key, table_id) pairs (strided loop
// version) Uses packed (table_id, local_idx) encoding in hash_vals for
// efficiency Only hash_vals needs volatile reads - hash_keys uses CAS for
// synchronization Supports optional frequency counting for LFU eviction
// strategy
template <typename KeyType, typename Hasher,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__global__ void __launch_bounds__(BLOCK_SIZE, 8)
segmented_unique_kernel(const KeyType *d_keys, const int64_t *d_table_ids,
                        KeyType *d_unique_keys, int64_t *d_output_indices,
                        size_t num_keys, KeyType *hash_keys, int64_t *hash_vals,
                        size_t capacity, int64_t *table_counters,
                        const int64_t *d_segmented_range,
                        int64_t *frequency_counters,
                        const int64_t *input_frequencies) {
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_keys;
       idx += stride) {
    const KeyType key = d_keys[idx];
    const int table_id = static_cast<int>(d_table_ids[idx]);
    const int64_t input_freq = input_frequencies ? input_frequencies[idx] : 1;

    // Hash the (key, table_id) pair.  capacity = 2*num_keys fits 32-bit
    // (num_keys < INT32_MAX is enforced), so keep the probe index in 32-bit
    // registers and use a 32-bit modulo: cheaper than a 64-bit modulo.
    const uint32_t cap = static_cast<uint32_t>(capacity);
    uint32_t key_hash = Hasher::hash(key);
    uint32_t tid_hash = Hasher::hash(static_cast<uint32_t>(table_id));
    uint32_t combined_hash = Hasher::hash_combine(key_hash, tid_hash);
    uint32_t hash_index = combined_hash % cap;

    bool done = false;
    for (uint32_t probe = 0; probe < cap && !done; ++probe) {
      const KeyType existing_key = hash_keys[hash_index];

      if (existing_key == empty_key) {
        // Try to claim this slot using CAS on hash_keys
        const KeyType old_key =
            atomicCAS(&hash_keys[hash_index], empty_key, key);

        if (old_key == empty_key) {
          // Successfully claimed the slot; get this table's local unique index.
          // Warp-aggregate the counter bump: threads in this warp that are
          // claiming a new key for the SAME table_id share one atomicAdd (of
          // the group size), then each takes a distinct slot from the reserved
          // range.  table_counters[table_id] is highly contended (num_tables is
          // small), so this collapses up to 32 atomics into 1 per table_id.
          auto active = cg::coalesced_threads();
          auto grp = cg::labeled_partition(active, static_cast<int>(table_id));
          int32_t base = 0;
          if (grp.thread_rank() == 0)
            base = static_cast<int32_t>(
                atomicAdd(&table_counters[table_id], grp.size()));
          base = grp.shfl(base, 0);
          int32_t local_unique_idx = base + grp.thread_rank();

          // Store unique key in partitioned layout using segmented_range offsets
          size_t output_pos =
              static_cast<size_t>(d_segmented_range[table_id]) +
              local_unique_idx;
          d_unique_keys[output_pos] = key;

          // Pack and store (table_id, local_idx) - this signals completion
          // Use volatile write to ensure visibility
          *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]) =
              pack_table_val(table_id, local_unique_idx);

          d_output_indices[idx] = local_unique_idx;

          // Update frequency counter for new unique key
          if (frequency_counters) {
            atomicAdd(&frequency_counters[output_pos], input_freq);
          }
          done = true;
        } else if (old_key == key) {
          // Another thread claimed with same key, wait for packed value
          int64_t packed_val;
          do {
            packed_val =
                *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]);
            __nanosleep(1);
          } while (packed_val == empty_val);

          // Check if table_id matches
          if (unpack_table_id(packed_val) == table_id) {
            // Same (key, table_id) pair - use existing index
            int32_t local_idx = unpack_local_idx(packed_val);
            d_output_indices[idx] = local_idx;

            // Update frequency counter for duplicate key
            if (frequency_counters) {
              size_t output_pos =
                  static_cast<size_t>(d_segmented_range[table_id]) +
                  local_idx;
              atomicAdd(&frequency_counters[output_pos], input_freq);
            }
            done = true;
          }
          // Different table_id with same key, continue probing
        }
        // Different key claimed this slot, continue probing
      } else if (existing_key == key) {
        // Slot has same key - read packed value to check table_id
        int64_t packed_val;
        do {
          packed_val =
              *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]);
          __nanosleep(1);
        } while (packed_val == empty_val);

        if (unpack_table_id(packed_val) == table_id) {
          // Exact (key, table_id) match found
          int32_t local_idx = unpack_local_idx(packed_val);
          d_output_indices[idx] = local_idx;

          // Update frequency counter for duplicate key
          if (frequency_counters) {
            size_t output_pos =
                static_cast<size_t>(d_segmented_range[table_id]) + local_idx;
            atomicAdd(&frequency_counters[output_pos], input_freq);
          }
          done = true;
        }
        // Different table_id with same key, continue probing
      }

      // Linear probing
      hash_index = (hash_index + 1) % cap;
    }
    assert(done && "segmented_unique_kernel: hash table full");
  }
}

// ============================================================================
// Two-level segmented unique: shared-memory first-level dedup, then merge into
// the global hash table.
// ============================================================================
// Block size and persistent shared hash-table size for the flush-on-fill
// segmented unique kernel.  BLOCK_SIZE is opened to the H100 maximum (1024).
// The shared table is PERSISTENT across the block's whole key stream (not
// reset per tile); it is flushed to the global table and cleared only when it
// is about to fill.  SHARED_TABLE_SIZE must be a power of two (the probe index
// is masked with S-1) and >= 2*SHARED_BLOCK_SIZE so a freshly cleared table
// always has room for a full wave.  At 24 bytes/slot (key8 + freq8 + tab4 +
// gidx4): 4096 slots = 96KB -> 2 blocks/SM (full occupancy); 8192 = 192KB ->
// 1 block/SM (50% occupancy, but fewer flushes).
constexpr int SHARED_BLOCK_SIZE = 1024;
constexpr int SHARED_TABLE_SIZE = 4096;
// Flush when the next wave might push the distinct count past this fraction of
// the table.  Below 1.0 keeps linear-probe chains short (cheap shared probing)
// at the cost of slightly more frequent flushes -- the load-factor knob.
constexpr int SHARED_FLUSH_CAP =
    static_cast<int>(SHARED_TABLE_SIZE * 0.7);

// Insert/lookup (key, table_id) in the GLOBAL hash table and return its local
// unique index within the table.  Shared between the per-key kernel and the
// two-level kernel's merge pass.  `freq_add` is added to the unique key's
// frequency counter (the tile-accumulated count in the two-level path, or 1 in
// the per-key path).  Mirrors the original inline probe logic exactly:
//   - claim an empty slot via CAS on hash_keys, warp-aggregate the per-table
//     counter bump, write the unique key + packed (table_id, local_idx),
//   - on a same-key slot, spin until the packed value is published, then match
//     table_id (probe past on mismatch).
template <typename KeyType, typename Hasher,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__device__ __forceinline__ int32_t global_insert(
    KeyType key, int table_id, int64_t freq_add, KeyType *hash_keys,
    int64_t *hash_vals, uint32_t cap, int64_t *table_counters,
    const int64_t *d_segmented_range, KeyType *d_unique_keys,
    int64_t *frequency_counters) {
  uint32_t key_hash = Hasher::hash(key);
  uint32_t tid_hash = Hasher::hash(static_cast<uint32_t>(table_id));
  uint32_t combined_hash = Hasher::hash_combine(key_hash, tid_hash);
  uint32_t hash_index = combined_hash % cap;

  for (uint32_t probe = 0; probe < cap; ++probe) {
    const KeyType existing_key = hash_keys[hash_index];

    if (existing_key == empty_key) {
      const KeyType old_key =
          atomicCAS(&hash_keys[hash_index], empty_key, key);
      if (old_key == empty_key) {
        auto active = cg::coalesced_threads();
        auto grp = cg::labeled_partition(active, table_id);
        int32_t base = 0;
        if (grp.thread_rank() == 0)
          base = static_cast<int32_t>(
              atomicAdd(&table_counters[table_id], grp.size()));
        base = grp.shfl(base, 0);
        int32_t local_unique_idx = base + grp.thread_rank();

        size_t output_pos =
            static_cast<size_t>(d_segmented_range[table_id]) + local_unique_idx;
        d_unique_keys[output_pos] = key;
        *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]) =
            pack_table_val(table_id, local_unique_idx);
        if (frequency_counters)
          atomicAdd(&frequency_counters[output_pos], freq_add);
        return local_unique_idx;
      } else if (old_key == key) {
        int64_t packed_val;
        do {
          packed_val =
              *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]);
          __nanosleep(1);
        } while (packed_val == empty_val);
        if (unpack_table_id(packed_val) == table_id) {
          int32_t local_idx = unpack_local_idx(packed_val);
          if (frequency_counters) {
            size_t output_pos =
                static_cast<size_t>(d_segmented_range[table_id]) + local_idx;
            atomicAdd(&frequency_counters[output_pos], freq_add);
          }
          return local_idx;
        }
      }
    } else if (existing_key == key) {
      int64_t packed_val;
      do {
        packed_val =
            *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]);
        __nanosleep(1);
      } while (packed_val == empty_val);
      if (unpack_table_id(packed_val) == table_id) {
        int32_t local_idx = unpack_local_idx(packed_val);
        if (frequency_counters) {
          size_t output_pos =
              static_cast<size_t>(d_segmented_range[table_id]) + local_idx;
          atomicAdd(&frequency_counters[output_pos], freq_add);
        }
        return local_idx;
      }
    }
    hash_index = (hash_index + 1) % cap;
  }
  return -1; // table full (guarded against by capacity = 2*num_keys)
}

// Flush the persistent shared table into the global table and clear it.
// Called collectively by all threads (contains __syncthreads).  Three phases
// privatize the table_counters atomic: the per-table NEW-key count is summed in
// SHARED during phase 1, then committed to the GLOBAL counter with ONE atomic
// per table in phase 2 -- cutting global atomics from O(claims) to
// O(num_tables) per flush.
//
//   Phase 1 (no global counter atomic, claim path never blocks): probe the
//     global table for each occupied slot.  A CAS-claim publishes the table_id
//     immediately with a PENDING (-1) index so concurrent probers can still
//     disambiguate the composite (key, table_id) by reading the table_id, while
//     the real index is deferred.  New claims bump the SHARED s_tcount[table].
//   Phase 2: one global atomicAdd(table_counters[t], s_tcount[t]) reserves a
//     contiguous index range per table -> s_tnext[t].
//   Phase 3: claimers draw their local idx from the SHARED s_tnext[t], write
//     d_unique_keys and overwrite hash_vals with the final index; duplicates
//     spin until the owner publishes a real (>=0) index.  Claimers publish
//     before any spin, so the block always makes progress -> deadlock-free.
//
// Then scatter idx to this epoch's input positions [lo,hi) and clear the table.
// MUST run before the table is reused, because clearing recycles slot ids.
template <typename KeyType, typename Hasher,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__device__ void su_flush(KeyType *s_keys, int64_t *s_freq, int *s_tabs,
                         int *s_gidx, uint32_t *s_ghash, int *s_tcount,
                         int *s_tnext, int *s_distinct, int S, int num_tables,
                         bool count_freq, size_t lo, size_t hi,
                         const int *key_slot, int64_t *d_output_indices,
                         KeyType *hash_keys, int64_t *hash_vals, uint32_t cap,
                         int64_t *table_counters,
                         const int64_t *d_segmented_range, KeyType *d_unique_keys,
                         int64_t *frequency_counters) {
  const int B = blockDim.x;
  const int tid = threadIdx.x;
  constexpr int CLAIM = -1, EXIST = -2;
  __syncthreads();

  for (int t = tid; t < num_tables; t += B)
    s_tcount[t] = 0;
  __syncthreads();

  // Phase 1: claim-or-find in the global table; count new keys in shared.
  for (int s = tid; s < S; s += B) {
    if (s_keys[s] == empty_key)
      continue;
    const KeyType key = s_keys[s];
    const int t = s_tabs[s];
    uint32_t kh = Hasher::hash(key);
    uint32_t th = Hasher::hash(static_cast<uint32_t>(t));
    uint32_t gi = Hasher::hash_combine(kh, th) % cap;
    for (uint32_t probe = 0; probe < cap; ++probe) {
      const KeyType ek = hash_keys[gi];
      if (ek == empty_key) {
        const KeyType old = atomicCAS(&hash_keys[gi], empty_key, key);
        if (old == empty_key) {
          // Claimed: publish table_id now (PENDING index), defer the index.
          *reinterpret_cast<volatile int64_t *>(&hash_vals[gi]) =
              pack_table_val(t, -1);
          ::atomicAdd(&s_tcount[t], 1);
          s_ghash[s] = gi;
          s_gidx[s] = CLAIM;
          break;
        } else if (old == key) {
          int64_t pv;
          do {
            pv = *reinterpret_cast<volatile int64_t *>(&hash_vals[gi]);
          } while (pv == empty_val);
          if (unpack_table_id(pv) == t) {
            s_ghash[s] = gi;
            s_gidx[s] = EXIST;
            break;
          }
        }
      } else if (ek == key) {
        int64_t pv;
        do {
          pv = *reinterpret_cast<volatile int64_t *>(&hash_vals[gi]);
        } while (pv == empty_val);
        if (unpack_table_id(pv) == t) {
          s_ghash[s] = gi;
          s_gidx[s] = EXIST;
          break;
        }
      }
      gi = (gi + 1) % cap;
    }
  }
  __syncthreads();

  // Phase 2: one global atomic per table reserves a contiguous index range.
  for (int t = tid; t < num_tables; t += B) {
    const int cnt = s_tcount[t];
    s_tnext[t] = cnt > 0 ? static_cast<int>(atomicAdd(
                               &table_counters[t], static_cast<int64_t>(cnt)))
                         : 0;
  }
  __syncthreads();

  // Phase 3: assign indices for claims (publish final), resolve duplicates.
  for (int s = tid; s < S; s += B) {
    if (s_keys[s] == empty_key)
      continue;
    const int t = s_tabs[s];
    const uint32_t gi = s_ghash[s];
    const int64_t freq_add = count_freq ? s_freq[s] : 0;
    int local_idx;
    if (s_gidx[s] == CLAIM) {
      local_idx = ::atomicAdd(&s_tnext[t], 1); // shared, within reserved range
      const size_t opos = static_cast<size_t>(d_segmented_range[t]) + local_idx;
      d_unique_keys[opos] = s_keys[s];
      *reinterpret_cast<volatile int64_t *>(&hash_vals[gi]) =
          pack_table_val(t, local_idx);
      if (count_freq)
        atomicAdd(&frequency_counters[opos], freq_add);
    } else { // EXIST: wait for the owner's final (>=0) index.
      int64_t pv;
      do {
        pv = *reinterpret_cast<volatile int64_t *>(&hash_vals[gi]);
      } while (unpack_local_idx(pv) < 0);
      local_idx = unpack_local_idx(pv);
      if (count_freq) {
        const size_t opos =
            static_cast<size_t>(d_segmented_range[t]) + local_idx;
        atomicAdd(&frequency_counters[opos], freq_add);
      }
    }
    s_gidx[s] = local_idx;
  }
  __syncthreads();

  // Scatter the global idx to this epoch's input positions.
  for (size_t i = lo + tid; i < hi; i += B) {
    d_output_indices[i] = s_gidx[key_slot[i]];
  }
  __syncthreads();

  // Clear the table for the next epoch.
  for (int s = tid; s < S; s += B) {
    s_keys[s] = empty_key;
    s_tabs[s] = -1;
    if (count_freq)
      s_freq[s] = 0;
  }
  if (tid == 0)
    *s_distinct = 0;
  __syncthreads();
}

// Flush-on-fill segmented unique kernel.  Each block owns a CONTIGUOUS chunk of
// keys and accumulates their distinct (key, table_id) pairs into a PERSISTENT
// shared-memory hash table, collapsing duplicates in fast shared memory.  The
// table is flushed to the global table (su_flush) and cleared only when it is
// about to fill -- not once per tile -- so a block with many duplicates touches
// the global table just once.  This cuts scattered global probes from num_keys
// down to the number of distinct keys, attacking the long_scoreboard bound,
// while amortizing the barrier/reset cost across the whole chunk.
//
// Intermediate state lives in three places:
//   - distinct (key,table) + accumulated freq -> persistent shared table
//     (s_keys/s_tabs/s_freq), survives across waves, cleared only on flush;
//   - per-slot global idx -> s_gidx (filled at flush, consumed by the scatter);
//   - per-input -> slot map -> GLOBAL key_slot[num_keys] (an epoch's total key
//     count can exceed S, so it cannot live in shared); scattered to
//     d_output_indices before each clear.
//
// Dynamic shared layout (bytes), 8-byte types first for alignment:
//   s_keys [S] KeyType   - persistent table keys (empty_key sentinel)
//   s_freq [S] int64     - accumulated frequency per slot
//   s_tabs [S] int32     - table_id per slot (-1 = unpublished, producer/consumer)
//   s_gidx [S] int32     - global local unique idx per slot (filled at flush)
template <typename KeyType, typename Hasher,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__global__ void __launch_bounds__(SHARED_BLOCK_SIZE)
segmented_unique_shared_kernel(
    const KeyType *d_keys, const int64_t *d_table_ids, KeyType *d_unique_keys,
    int64_t *d_output_indices, size_t num_keys, KeyType *hash_keys,
    int64_t *hash_vals, size_t capacity, int64_t *table_counters,
    const int64_t *d_segmented_range, int64_t *frequency_counters,
    const int64_t *input_frequencies, int *key_slot, int64_t num_tables) {
  constexpr int S = SHARED_TABLE_SIZE;
  const int B = blockDim.x;
  const int tid = threadIdx.x;
  const uint32_t cap = static_cast<uint32_t>(capacity);
  const bool count_freq = frequency_counters != nullptr;

  // Dynamic shared layout: persistent table (s_keys/s_freq/s_tabs/s_gidx),
  // s_ghash[S] remembers each occupied slot's global probe index across the
  // 3-phase flush, and s_tcount/s_tnext[num_tables] privatize the per-table
  // counter (shared count in phase 1 -> one global reserve in phase 2).
  extern __shared__ char smem[];
  KeyType *s_keys = reinterpret_cast<KeyType *>(smem);
  int64_t *s_freq = reinterpret_cast<int64_t *>(s_keys + S);
  int *s_tabs = reinterpret_cast<int *>(s_freq + S);
  int *s_gidx = reinterpret_cast<int *>(s_tabs + S);
  uint32_t *s_ghash = reinterpret_cast<uint32_t *>(s_gidx + S);
  int *s_tcount = reinterpret_cast<int *>(s_ghash + S);
  int *s_tnext = reinterpret_cast<int *>(s_tcount + num_tables);
  __shared__ int s_distinct;

  // This block owns the contiguous chunk [cstart, cend).  Contiguous (not
  // strided) so each flush epoch is a contiguous range [epoch_start, wave) that
  // the scatter can walk, and so the d_keys loads coalesce.
  const size_t chunk = (num_keys + gridDim.x - 1) / gridDim.x;
  const size_t cstart = static_cast<size_t>(blockIdx.x) * chunk;
  if (cstart >= num_keys)
    return;
  const size_t cend = (cstart + chunk < num_keys) ? cstart + chunk : num_keys;

  // Initialize the persistent table once.
  for (int i = tid; i < S; i += B) {
    s_keys[i] = empty_key;
    s_tabs[i] = -1;
    if (count_freq)
      s_freq[i] = 0;
  }
  if (tid == 0)
    s_distinct = 0;
  __syncthreads();

  size_t epoch_start = cstart;
  for (size_t wave = cstart; wave < cend; wave += B) {
    // Flush before a wave that could overflow the table.  s_distinct is the
    // count carried over from prior waves (same value for all threads after the
    // end-of-loop sync), so this branch is uniform across the block.
    if (s_distinct + B > SHARED_FLUSH_CAP) {
      su_flush<KeyType, Hasher, empty_key, empty_val>(
          s_keys, s_freq, s_tabs, s_gidx, s_ghash, s_tcount, s_tnext,
          &s_distinct, S, static_cast<int>(num_tables), count_freq, epoch_start,
          wave, key_slot, d_output_indices, hash_keys, hash_vals, cap,
          table_counters, d_segmented_range, d_unique_keys, frequency_counters);
      epoch_start = wave;
    }

    const size_t idx = wave + tid;
    if (idx < cend) {
      const KeyType key = d_keys[idx];
      const int table_id = static_cast<int>(d_table_ids[idx]);
      const int64_t input_freq = input_frequencies ? input_frequencies[idx] : 1;

      uint32_t key_hash = Hasher::hash(key);
      uint32_t tid_hash = Hasher::hash(static_cast<uint32_t>(table_id));
      uint32_t combined_hash = Hasher::hash_combine(key_hash, tid_hash);
      uint32_t si = combined_hash & (S - 1);

      int slot = -1;
      for (uint32_t probe = 0; probe < static_cast<uint32_t>(S); ++probe) {
        const KeyType ek = s_keys[si];
        if (ek == empty_key) {
          const KeyType old = atomicCAS(&s_keys[si], empty_key, key);
          if (old == empty_key) {
            // Claimed a new distinct slot: publish table_id (volatile so
            // spinners observe it) and bump the distinct counter.
            *reinterpret_cast<volatile int *>(&s_tabs[si]) = table_id;
            ::atomicAdd(&s_distinct, 1); // built-in int overload (dyn_emb:: hides it)
            slot = si;
            break;
          } else if (old == key) {
            int t;
            do {
              t = *reinterpret_cast<volatile int *>(&s_tabs[si]);
            } while (t < 0);
            if (t == table_id) {
              slot = si;
              break;
            }
          }
        } else if (ek == key) {
          int t;
          do {
            t = *reinterpret_cast<volatile int *>(&s_tabs[si]);
          } while (t < 0);
          if (t == table_id) {
            slot = si;
            break;
          }
        }
        si = (si + 1) & (S - 1);
      }
      key_slot[idx] = slot;
      if (count_freq && slot >= 0)
        atomicAdd(&s_freq[slot], input_freq);
    }
    __syncthreads(); // all inserts of this wave land before the next flush check
  }

  // Final flush of the last epoch.
  su_flush<KeyType, Hasher, empty_key, empty_val>(
      s_keys, s_freq, s_tabs, s_gidx, s_ghash, s_tcount, s_tnext, &s_distinct, S,
      static_cast<int>(num_tables), count_freq, epoch_start, cend, key_slot,
      d_output_indices, hash_keys, hash_vals, cap, table_counters,
      d_segmented_range, d_unique_keys, frequency_counters);
}

// Binary search helper for compaction
__device__ __forceinline__ int binary_search_upper_bound(const int64_t *arr,
                                                         int n, int64_t val) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (arr[mid] <= val) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo - 1;
}

// Fused kernel to compact both keys and frequency counters from partitioned
// layout Shares binary search computation between both operations, reducing
// overhead d_total_unique is a device pointer to avoid GPU-CPU synchronization
// partitioned_freq and output_freq can be nullptr if frequency counting is
// disabled
template <typename KeyType>
__global__ void compact_keys_and_freq_kernel(
    const KeyType *partitioned_keys, const int64_t *partitioned_freq,
    const int64_t *d_segmented_range, const int64_t *table_offsets,
    int64_t num_tables, KeyType *output_keys, int64_t *output_freq,
    const int64_t *d_total_unique) {
  const int64_t total_unique = *d_total_unique;
  const int64_t stride = blockDim.x * gridDim.x;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_unique;
       idx += stride) {
    // Find which table this output index belongs to via output table_offsets
    int table_id =
        binary_search_upper_bound(table_offsets, num_tables + 1, idx);

    // Offset within this table's unique keys
    int64_t local_idx = idx - table_offsets[table_id];

    // Source position uses segmented_range (input partition layout)
    size_t src_pos =
        static_cast<size_t>(d_segmented_range[table_id]) + local_idx;

    // Compact keys
    output_keys[idx] = partitioned_keys[src_pos];

    // Compact frequency counters if enabled
    if (partitioned_freq != nullptr) {
      output_freq[idx] = partitioned_freq[src_pos];
    }
  }
}

// ============================================================================
// Helper kernel to expand table IDs from jagged offsets
// ============================================================================

// Expand jagged offsets to per-element table_ids (identity mapping,
// local_batch_size=1). For each idx, find largest t such that offsets[t] <= idx
// via binary_search_upper_bound (defined above).
__global__ void expand_table_ids_kernel(const int64_t *offsets,
                                        int64_t *table_ids, int num_tables,
                                        int64_t num_elements) {
  const int64_t stride = blockDim.x * gridDim.x;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;
       idx += stride) {
    table_ids[idx] = binary_search_upper_bound(offsets, num_tables + 1, idx);
  }
}

// Adjust output indices to global indices using table offsets (strided loop
// version)
__global__ void adjust_output_indices_kernel(const int64_t *d_table_ids,
                                             const int64_t *table_offsets,
                                             int64_t *d_output_indices,
                                             size_t num_keys) {
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_keys;
       idx += stride) {
    int64_t table_id = d_table_ids[idx];
    d_output_indices[idx] += table_offsets[table_id];
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
segmented_unique_cuda(at::Tensor keys, at::Tensor segmented_range,
                      int64_t num_tables, at::Tensor input_frequencies) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const int64_t num_keys = keys.numel();
  const auto device = keys.device();
  const auto key_dtype = keys.scalar_type();
  const int device_sm_count = DeviceProp::getDeviceProp(device.index()).num_sms;

  TORCH_CHECK(segmented_range.numel() == num_tables + 1,
              "segmented_range must have num_tables+1 elements");
  TORCH_CHECK(segmented_range.scalar_type() == at::kLong,
              "segmented_range must be int64");
  TORCH_CHECK(segmented_range.device() == device,
              "segmented_range must be on the same device as keys");
  TORCH_CHECK(segmented_range.is_contiguous(),
              "segmented_range must be contiguous");
  TORCH_CHECK(num_tables > 0, "num_tables must be positive");
  TORCH_CHECK(num_keys < std::numeric_limits<int32_t>::max(),
              "num_keys must be less than std::numeric_limits<int32_t>::max()");
  TORCH_CHECK(
      num_tables < std::numeric_limits<int32_t>::max(),
      "num_tables must be less than std::numeric_limits<int32_t>::max()");

  // Frequency counting behavior:
  // - input_frequencies not defined (None): disable frequency counting entirely
  // - input_frequencies defined with numel()==0: enable counting, each key
  // counts as 1
  // - input_frequencies defined with numel()>0: use provided frequencies (must
  // match num_keys)
  const bool enable_freq_counting = input_frequencies.defined();
  const bool has_input_freq =
      enable_freq_counting && input_frequencies.numel() > 0;

  if (has_input_freq) {
    TORCH_CHECK(input_frequencies.numel() == num_keys,
                "input_frequencies must have same length as keys");
  }

  // Debug validation of segmented_range (enabled via DYNAMICEMB_DEBUG=1).
  // Checks: starts at 0, ends at num_keys, monotonically non-decreasing.
  if (std::getenv("DYNAMICEMB_DEBUG")) {
    at::Tensor sr_cpu = segmented_range.to(at::kCPU);
    const int64_t *sr = sr_cpu.data_ptr<int64_t>();
    TORCH_CHECK(sr[0] == 0,
                "segmented_range[0] must be 0, got ", sr[0]);
    TORCH_CHECK(sr[num_tables] == num_keys,
                "segmented_range[num_tables] must equal num_keys (", num_keys,
                "), got ", sr[num_tables]);
    for (int64_t t = 0; t < num_tables; ++t) {
      TORCH_CHECK(sr[t + 1] >= sr[t],
                  "segmented_range must be non-decreasing: "
                  "segmented_range[", t + 1, "]=", sr[t + 1],
                  " < segmented_range[", t, "]=", sr[t]);
    }
  }

  // Handle empty input
  if (num_keys == 0) {
    at::Tensor table_offsets = at::zeros(
        {num_tables + 1}, at::TensorOptions().dtype(at::kLong).device(device));
    at::Tensor num_uniques = table_offsets.slice(0, num_tables, num_tables + 1);
    return std::make_tuple(
        num_uniques, at::empty({0}, keys.options()),
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)),
        table_offsets,
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)));
  }

  // Size the grid to fill the SMs.  These kernels (esp. segmented_unique_kernel)
  // are latency-bound on scattered global probe loads (~88% long_scoreboard
  // stalls), which are hidden by having many resident warps.  With
  // BLOCK_SIZE=256 (8 warps/block), 8 blocks/SM = 64 warps/SM = full occupancy
  // on sm_90 (regs=32 fit).  Cap at full occupancy and never exceed the work.
  // The old SM*4 x 64-thread launch sat at ~12% occupancy / 0.12 waves, leaving
  // nothing for the scheduler to hide the memory latency behind.
  constexpr int BLOCKS_PER_SM = 8;
  int64_t grid64 = (num_keys + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int64_t full_grid = static_cast<int64_t>(device_sm_count) * BLOCKS_PER_SM;
  if (grid64 > full_grid)
    grid64 = full_grid;
  if (grid64 < 1)
    grid64 = 1;
  const int grid_size = static_cast<int>(grid64);

  // Generate per-element table_ids from segmented_range for hash logic and
  // adjust_output_indices_kernel. Keys must be sorted by table:
  // keys[segmented_range[t]:segmented_range[t+1]] all belong to table t.
  at::Tensor table_ids = at::empty(
      {num_keys}, at::TensorOptions().dtype(at::kLong).device(device));
  expand_table_ids_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      get_pointer<const int64_t>(segmented_range),
      get_pointer<int64_t>(table_ids), num_tables, num_keys);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  // Partitioned buffer uses segmented_range as per-table offsets, so total
  // size is num_keys instead of the worst-case num_tables * num_keys.
  at::Tensor partitioned_unique_keys = at::empty({num_keys}, keys.options());

  // Allocate output indices (local indices within each table, adjusted later)
  at::Tensor output_indices = at::empty(
      {num_keys}, at::TensorOptions().dtype(at::kLong).device(device));

  // Per-table unique counters
  at::Tensor table_counters = at::zeros(
      {num_tables}, at::TensorOptions().dtype(at::kLong).device(device));

  // Allocate partitioned frequency counters if needed
  at::Tensor partitioned_freq_counters;
  if (enable_freq_counting) {
    partitioned_freq_counters = at::zeros(
        {num_keys}, at::TensorOptions().dtype(at::kLong).device(device));
  }

  // Allocate shared hash table for (key, table_id) pairs
  // capacity = 2 * num_keys for good load factor
  // hash_vals stores packed (table_id << 32 | local_idx)
  const int64_t capacity = num_keys * 2;
  at::Tensor hash_keys = at::empty({capacity}, keys.options());
  at::Tensor hash_vals = at::empty(
      {capacity}, at::TensorOptions().dtype(at::kLong).device(device));

  // Per-input -> shared-slot map for the flush-on-fill kernel.  An epoch's key
  // count can exceed the shared table size, so this linkage cannot live in
  // shared; it is written once per key and read once at each flush (int32 is
  // enough: a slot index is < SHARED_TABLE_SIZE).
  at::Tensor key_slot = at::empty(
      {num_keys}, at::TensorOptions().dtype(at::kInt).device(device));

  // Grid for the flush-on-fill shared kernel: each block owns a contiguous
  // chunk of keys.  Cap at SHARED_BLOCKS_PER_SM blocks/SM (the 96KB persistent
  // table at S=4096 allows 2 resident blocks of 1024 threads = full occupancy
  // on sm_90; a larger table drops this to 1).  Never launch more blocks than
  // there is work for (>= 1 wave each).
  constexpr int SHARED_BLOCKS_PER_SM = 2;
  int64_t shared_grid64 =
      (num_keys + SHARED_BLOCK_SIZE - 1) / SHARED_BLOCK_SIZE;
  const int64_t shared_full_grid =
      static_cast<int64_t>(device_sm_count) * SHARED_BLOCKS_PER_SM;
  if (shared_grid64 > shared_full_grid)
    shared_grid64 = shared_full_grid;
  if (shared_grid64 < 1)
    shared_grid64 = 1;
  const int shared_grid = static_cast<int>(shared_grid64);

  dispatch_key_type(key_dtype, [&]<typename KeyType>() {
    // Initialize hash table and counters
    segmented_init_kernel<KeyType><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        get_pointer<KeyType>(hash_keys), get_pointer<int64_t>(hash_vals),
        get_pointer<int64_t>(table_counters), capacity, num_tables);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();

    // Dynamic shared memory: persistent table s_keys[S]+s_freq[S]+s_tabs[S]+
    // s_gidx[S] (24B/slot) + s_ghash[S] (4B/slot, global probe index across the
    // 3-phase flush) + s_tcount[num_tables]+s_tnext[num_tables] (privatized
    // per-table counter).  Exceeds the 48KB default, so opt in explicitly.
    const size_t smem_bytes =
        static_cast<size_t>(SHARED_TABLE_SIZE) *
            (sizeof(KeyType) + sizeof(int64_t) + 3 * sizeof(int)) +
        static_cast<size_t>(num_tables) * 2 * sizeof(int);
    auto kernel =
        segmented_unique_shared_kernel<KeyType, MurmurHash3_32<KeyType>>;
    cudaFuncSetAttribute(kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));

    // Run flush-on-fill segmented unique kernel with optional freq counting
    kernel<<<shared_grid, SHARED_BLOCK_SIZE, smem_bytes, stream>>>(
        get_pointer<const KeyType>(keys),
        get_pointer<const int64_t>(table_ids),
        get_pointer<KeyType>(partitioned_unique_keys),
        get_pointer<int64_t>(output_indices), num_keys,
        get_pointer<KeyType>(hash_keys), get_pointer<int64_t>(hash_vals),
        capacity, get_pointer<int64_t>(table_counters),
        get_pointer<const int64_t>(segmented_range),
        enable_freq_counting
            ? get_pointer<int64_t>(partitioned_freq_counters)
            : nullptr,
        has_input_freq ? get_pointer<const int64_t>(input_frequencies)
                       : nullptr,
        get_pointer<int32_t>(key_slot), num_tables);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  // Compute table offsets using inclusive scan
  at::Tensor table_offsets = at::zeros(
      {num_tables + 1}, at::TensorOptions().dtype(at::kLong).device(device));

  // Use CUB for inclusive scan
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      nullptr, temp_storage_bytes, get_pointer<int64_t>(table_counters),
      get_pointer<int64_t>(table_offsets) + 1, num_tables, stream);
  at::Tensor temp_storage =
      at::empty({static_cast<int64_t>(temp_storage_bytes)},
                at::TensorOptions().dtype(at::kByte).device(device));
  cub::DeviceScan::InclusiveSum(temp_storage.data_ptr(), temp_storage_bytes,
                                get_pointer<int64_t>(table_counters),
                                get_pointer<int64_t>(table_offsets) + 1,
                                num_tables, stream);

  // Allocate compacted output with size num_keys (worst case: all keys unique)
  // Actual count is table_offsets[num_tables], available on device
  at::Tensor unique_keys = at::empty({num_keys}, keys.options());
  at::Tensor output_freq_counters;
  if (enable_freq_counting) {
    output_freq_counters = at::empty(
        {num_keys}, at::TensorOptions().dtype(at::kLong).device(device));
  } else {
    // Return an empty tensor when frequency counting is disabled
    output_freq_counters =
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device));
  }

  // Compact keys and frequency counters in a single fused kernel
  dispatch_key_type(key_dtype, [&]<typename KeyType>() {
    compact_keys_and_freq_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        get_pointer<const KeyType>(partitioned_unique_keys),
        enable_freq_counting
            ? get_pointer<const int64_t>(partitioned_freq_counters)
            : nullptr,
        get_pointer<const int64_t>(segmented_range),
        get_pointer<const int64_t>(table_offsets),
        num_tables, get_pointer<KeyType>(unique_keys),
        enable_freq_counting ? get_pointer<int64_t>(output_freq_counters)
                             : nullptr,
        get_pointer<const int64_t>(table_offsets) + num_tables);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  // Adjust output indices to global indices
  adjust_output_indices_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      get_pointer<const int64_t>(table_ids),
      get_pointer<const int64_t>(table_offsets),
      get_pointer<int64_t>(output_indices), num_keys);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  // Extract num_uniques as a separate tensor (view of
  // table_offsets[num_tables])
  at::Tensor num_uniques = table_offsets.slice(0, num_tables, num_tables + 1);

  return std::make_tuple(num_uniques, unique_keys, output_indices,
                         table_offsets, output_freq_counters);
}

// Expand table IDs from offsets (identity mapping, local_batch_size=1).
// offsets: size = num_tables + 1; offsets[t] is the start index for table t.
// num_tables is derived from offsets.size(0)-1.
at::Tensor expand_table_ids_cuda(at::Tensor offsets, int64_t num_elements) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const auto device = offsets.device();
  const int device_sm_count = DeviceProp::getDeviceProp(device.index()).num_sms;

  TORCH_CHECK(offsets.is_cuda(), "offsets must be on CUDA device");

  // Handle empty input
  if (num_elements == 0) {
    return at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device));
  }

  // num_tables derived from offsets; local_batch_size is always 1
  const int64_t num_tables = offsets.size(0) - 1;

  // Compute grid size based on SM count
  constexpr int BLOCKS_PER_SM = 4;
  const int grid_size =
      std::min((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
               static_cast<int64_t>(device_sm_count * BLOCKS_PER_SM));

  at::Tensor table_ids = at::empty(
      {num_elements}, at::TensorOptions().dtype(at::kLong).device(device));

  expand_table_ids_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      get_pointer<const int64_t>(offsets),
      get_pointer<int64_t>(table_ids), num_tables, num_elements);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  return table_ids;
}

// Compute dedup lengths and offsets using GPU kernel
std::tuple<at::Tensor, at::Tensor> compute_dedup_lengths_cuda(
    at::Tensor unique_offsets, at::Tensor table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size, int64_t new_lengths_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const auto device = unique_offsets.device();

  TORCH_CHECK(unique_offsets.is_cuda(),
              "unique_offsets must be on CUDA device");
  TORCH_CHECK(table_offsets_in_feature.is_cuda(),
              "table_offsets_in_feature must be on CUDA device");

  // Handle empty case
  if (new_lengths_size == 0) {
    return std::make_tuple(
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)),
        at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(device)));
  }

  // Allocate output tensors
  at::Tensor new_lengths = at::empty(
      {new_lengths_size}, at::TensorOptions().dtype(at::kLong).device(device));
  at::Tensor new_offsets =
      at::empty({new_lengths_size + 1},
                at::TensorOptions().dtype(at::kLong).device(device));

  // Convert unique_offsets to uint64_t for the kernel
  // The kernel expects uint64_t*, but int64_t is bit-compatible
  get_new_length_and_offsets(
      reinterpret_cast<uint64_t *>(get_pointer<int64_t>(unique_offsets)),
      get_pointer<int64_t>(table_offsets_in_feature), num_tables,
      new_lengths_size, local_batch_size, DataType::Int64, DataType::Int64,
      get_pointer<int64_t>(new_offsets), get_pointer<int64_t>(new_lengths),
      stream);

  return std::make_tuple(new_lengths, new_offsets);
}

} // namespace dyn_emb

// Python bindings
#ifdef DEMB_USE_PYBIND11
void bind_unique_op(py::module &m) {
  m.def(
      "segmented_unique_cuda",
      [](at::Tensor keys, at::Tensor segmented_range, int64_t num_tables,
         const c10::optional<at::Tensor> &input_frequencies) {
        // Convert optional to tensor:
        // - None -> undefined tensor (disables frequency counting)
        // - Some(tensor) -> that tensor (enables frequency counting)
        at::Tensor freq_tensor;
        if (input_frequencies.has_value()) {
          freq_tensor = input_frequencies.value();
        }
        // If input_frequencies was None, freq_tensor remains undefined
        // which will disable frequency counting in the C++ implementation
        return dyn_emb::segmented_unique_cuda(keys, segmented_range, num_tables,
                                              freq_tensor);
      },
      R"doc(
Segmented unique: deduplicate keys per table using GPU hash table.

Keys must be pre-sorted by table: keys[segmented_range[t]:segmented_range[t+1]]
all belong to table t. Keys are deduplicated within each table independently.
The same key can appear in different tables.

NOTE: This function is fully asynchronous with no GPU-CPU synchronization.

Args:
    keys: Input keys tensor (int64 or uint64), sorted by table.
    segmented_range: Table boundary offsets (int64, size=num_tables+1).
                     segmented_range[t] is the start index in keys for table t;
                     segmented_range[num_tables] must equal len(keys).
    num_tables: Total number of tables
    input_frequencies: Controls frequency counting behavior:
                       - None: Disable frequency counting (output freq_counters empty)
                       - Empty tensor (numel==0): Enable counting, each key counts as 1
                       - Tensor with numel==num_keys: Use provided frequencies

Returns:
    Tuple of (num_uniques, unique_keys, output_indices, table_offsets, frequency_counters)
    - num_uniques: Tensor of size 1 with total unique count (on device)
    - unique_keys: Compacted unique keys with size=len(keys). Only first
                   num_uniques elements are valid.
    - output_indices: Index mapping (input idx -> global unique idx)
    - table_offsets: Tensor of size (num_tables + 1) with cumulative unique counts
                     table_offsets[i] is the start index for table i in unique_keys
    - frequency_counters: Per-unique-key frequency counts (empty if disabled)
)doc",
      py::arg("keys"), py::arg("segmented_range"), py::arg("num_tables"),
      py::arg("input_frequencies") = py::none());

  m.def(
      "expand_table_ids_cuda",
      [](at::Tensor offsets, int64_t num_elements) {
        return dyn_emb::expand_table_ids_cuda(offsets, num_elements);
      },
      R"doc(
Expand table IDs from offsets (identity mapping, local_batch_size=1).

Generates a table_id for each element via binary search on offsets.
num_tables is derived from offsets.size(0)-1.

Args:
    offsets: Table boundary offsets (int64, size = num_tables + 1)
             offsets[t] is the start index for table t's keys.
    num_elements: Total number of elements (keys)

Returns:
    table_ids tensor (int64) with same length as num_elements
)doc",
      py::arg("offsets"), py::arg("num_elements") = 0);

  m.def(
      "compute_dedup_lengths_cuda",
      [](at::Tensor unique_offsets, at::Tensor table_offsets_in_feature,
         int64_t num_tables, int64_t local_batch_size,
         int64_t new_lengths_size) {
        return dyn_emb::compute_dedup_lengths_cuda(
            unique_offsets, table_offsets_in_feature, num_tables,
            local_batch_size, new_lengths_size);
      },
      R"doc(
Compute new lengths and offsets by evenly distributing unique keys.

This is a GPU kernel that evenly distributes unique keys across (feature, batch)
buckets. For each table, unique keys are distributed so each bucket gets
(unique_count / num_buckets) keys, with the first (unique_count % num_buckets)
buckets getting one extra.

Args:
    unique_offsets: Cumulative unique counts per table (int64, device)
    table_offsets_in_feature: Feature offsets per table (int64, device)
    num_tables: Number of tables
    local_batch_size: Batch size per feature
    new_lengths_size: Total output size (num_features * local_batch_size)

Returns:
    Tuple of (new_lengths, new_offsets)
    - new_lengths: Length for each bucket (int64)
    - new_offsets: Offset for each bucket (int64)
)doc",
      py::arg("unique_offsets"), py::arg("table_offsets_in_feature"),
      py::arg("num_tables"), py::arg("local_batch_size"),
      py::arg("new_lengths_size"));
}
#endif
