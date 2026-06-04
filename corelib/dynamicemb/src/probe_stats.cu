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

// Diagnostic-only: instruments the segmented_unique probe behaviour without
// touching the production kernel in unique_op.cu.  For every input key it
// reproduces the same composite-(key,table_id) open-addressing probe and
// records (a) how the key resolved and (b) how many slots it probed.
//
//   outcome[i]:  0 = CLAIMED  (this thread inserted the unique (key,table))
//                1 = DUPLICATE (the (key,table) was already present)
//   probe_count[i]: number of slots inspected until the key resolved
//                   (1 == resolved on the very first hashed slot).
//
// Not wired into segmented_unique_cuda; exposed only as the standalone op
// segmented_unique_probe_stats_cuda for offline analysis.

#include "check.h"
#include "torch_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <limits>
#ifdef DEMB_USE_PYBIND11
#include <pybind11/pybind11.h>
#include <torch/extension.h>
namespace py = pybind11;
#endif

namespace dyn_emb {
namespace {

constexpr int PS_BLOCK = 256;

// Same MurmurHash3_32 as unique_op.cu (kept local so this file is standalone).
template <typename Key, uint32_t m_seed = 0> struct PSMurmur {
  __forceinline__ __device__ static uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
  }
  __forceinline__ __device__ static uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }
  __forceinline__ __device__ static uint32_t hash(const Key &key) {
    constexpr int len = sizeof(Key);
    const uint8_t *const data = reinterpret_cast<const uint8_t *>(&key);
    constexpr int nblocks = len / 4;
    uint32_t h1 = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51, c2 = 0x1b873593;
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
  __forceinline__ __device__ static uint32_t hash_combine(uint32_t h1,
                                                          uint32_t h2) {
    h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    return h1;
  }
};

__device__ __forceinline__ int ps_table_id(const int64_t *seg, int n_tables,
                                            int64_t idx) {
  // largest t with seg[t] <= idx  (seg has n_tables+1 entries, monotone)
  int lo = 0, hi = n_tables;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (seg[mid] <= idx)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

__device__ __forceinline__ uint64_t atomicCAS_u64(uint64_t *a, uint64_t c,
                                                  uint64_t v) {
  return static_cast<uint64_t>(
      ::atomicCAS(reinterpret_cast<unsigned long long *>(a),
                  static_cast<unsigned long long>(c),
                  static_cast<unsigned long long>(v)));
}

template <typename KeyType, KeyType empty_key = std::numeric_limits<KeyType>::max()>
__global__ void
probe_stats_kernel(const KeyType *d_keys, const int64_t *d_seg, int num_tables,
                   size_t num_keys, KeyType *hash_keys, int32_t *slot_tid,
                   size_t capacity, int8_t *outcome, int32_t *probe_count) {
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_keys;
       idx += stride) {
    const KeyType key = d_keys[idx];
    const int tid = ps_table_id(d_seg, num_tables, static_cast<int64_t>(idx));

    uint32_t kh = PSMurmur<KeyType>::hash(key);
    uint32_t th = PSMurmur<KeyType>::hash(static_cast<uint32_t>(tid));
    uint32_t combined = PSMurmur<KeyType>::hash_combine(kh, th);
    size_t hi = combined % capacity;

    int32_t probes = 0;
    int8_t res = -1;
    for (size_t p = 0; p < capacity; ++p) {
      ++probes;
      const KeyType ek = hash_keys[hi];
      if (ek == empty_key) {
        KeyType old = static_cast<KeyType>(atomicCAS_u64(
            reinterpret_cast<uint64_t *>(&hash_keys[hi]),
            static_cast<uint64_t>(empty_key), static_cast<uint64_t>(key)));
        if (old == empty_key) {
          // We claimed this slot for (key, tid).
          *reinterpret_cast<volatile int32_t *>(&slot_tid[hi]) = tid;
          res = 0; // CLAIMED
          break;
        } else if (old == key) {
          // Someone else claimed the same key first; confirm same table.
          int32_t owner;
          do {
            owner = *reinterpret_cast<volatile int32_t *>(&slot_tid[hi]);
            __nanosleep(1);
          } while (owner < 0);
          if (owner == tid) {
            res = 1; // DUPLICATE
            break;
          }
          // same key, different table -> hash collision, keep probing
        }
        // different key won this slot -> keep probing
      } else if (ek == key) {
        int32_t owner;
        do {
          owner = *reinterpret_cast<volatile int32_t *>(&slot_tid[hi]);
          __nanosleep(1);
        } while (owner < 0);
        if (owner == tid) {
          res = 1; // DUPLICATE
          break;
        }
        // same key, different table -> keep probing
      }
      hi = (hi + 1) % capacity;
    }
    outcome[idx] = res;
    probe_count[idx] = probes;
  }
}

template <typename KeyType>
__global__ void ps_init_kernel(KeyType *hash_keys, int32_t *slot_tid,
                               size_t capacity, KeyType empty_key) {
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < capacity;
       i += stride) {
    hash_keys[i] = empty_key;
    slot_tid[i] = -1;
  }
}

} // namespace

std::tuple<at::Tensor, at::Tensor>
segmented_unique_probe_stats_cuda(at::Tensor keys, at::Tensor segmented_range,
                                  int64_t num_tables) {
  TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
  TORCH_CHECK(segmented_range.numel() == num_tables + 1,
              "segmented_range must have num_tables+1 entries");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t num_keys = keys.numel();
  const auto device = keys.device();

  at::Tensor outcome =
      at::empty({num_keys}, at::TensorOptions().dtype(at::kChar).device(device));
  at::Tensor probe_count = at::empty(
      {num_keys}, at::TensorOptions().dtype(at::kInt).device(device));
  if (num_keys == 0)
    return std::make_tuple(outcome, probe_count);

  const int64_t capacity = num_keys * 2;
  at::Tensor hash_keys = at::empty({capacity}, keys.options());
  at::Tensor slot_tid = at::empty(
      {capacity}, at::TensorOptions().dtype(at::kInt).device(device));

  const int sm = DeviceProp::getDeviceProp(device.index()).num_sms;
  const int grid = sm * 8;

  auto kt = keys.scalar_type();
  TORCH_CHECK(kt == at::kLong || kt == at::kUInt64, "keys must be int64/uint64");

  if (kt == at::kLong) {
    using K = int64_t;
    K ek = std::numeric_limits<K>::max();
    ps_init_kernel<K><<<grid, PS_BLOCK, 0, stream>>>(
        get_pointer<K>(hash_keys), get_pointer<int32_t>(slot_tid), capacity, ek);
    probe_stats_kernel<K><<<grid, PS_BLOCK, 0, stream>>>(
        get_pointer<const K>(keys), get_pointer<const int64_t>(segmented_range),
        static_cast<int>(num_tables), num_keys, get_pointer<K>(hash_keys),
        get_pointer<int32_t>(slot_tid), capacity, get_pointer<int8_t>(outcome),
        get_pointer<int32_t>(probe_count));
  } else {
    using K = uint64_t;
    K ek = std::numeric_limits<K>::max();
    ps_init_kernel<K><<<grid, PS_BLOCK, 0, stream>>>(
        get_pointer<K>(hash_keys), get_pointer<int32_t>(slot_tid), capacity, ek);
    probe_stats_kernel<K><<<grid, PS_BLOCK, 0, stream>>>(
        get_pointer<const K>(keys), get_pointer<const int64_t>(segmented_range),
        static_cast<int>(num_tables), num_keys, get_pointer<K>(hash_keys),
        get_pointer<int32_t>(slot_tid), capacity, get_pointer<int8_t>(outcome),
        get_pointer<int32_t>(probe_count));
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(outcome, probe_count);
}

} // namespace dyn_emb

#ifdef DEMB_USE_PYBIND11
void bind_probe_stats(py::module &m) {
  m.def("segmented_unique_probe_stats_cuda",
        &dyn_emb::segmented_unique_probe_stats_cuda,
        R"doc(Diagnostic: per-key probe statistics for segmented_unique.

Returns (outcome, probe_count), both length len(keys):
  outcome: int8, 0 = CLAIMED (this key is the unique representative inserted
           here), 1 = DUPLICATE (the (key, table) was already present).
  probe_count: int32, number of slots probed before the key resolved
               (1 = resolved on the first hashed slot, higher = collisions).

Mirrors the production composite-(key,table_id) probe but only measures it;
it does not affect segmented_unique_cuda.)doc",
        py::arg("keys"), py::arg("segmented_range"), py::arg("num_tables"));
}
#endif
