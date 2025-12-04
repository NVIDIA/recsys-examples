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

#include <array>
#include <cstdint>
#include <stddef.h>
#include <type_traits>
#include <utility>

#include <cuda/atomic>
#include <cuda/pipeline>
#include <cuda/std/semaphore>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include "score.cuh"

extern "C" __device__ size_t __cvta_generic_to_shared(const void *);

namespace dyn_emb {

using CounterType = int64_t;
using DigestType = uint8_t;
using IndexType = int64_t;

__forceinline__ __device__ int atomicAdd(int *address, int val) {
  return ::atomicAdd(address, val);
}

__device__ __forceinline__ CounterType atomicAdd(CounterType *address,
                                                 CounterType val) {
  return (CounterType)::atomicAdd((unsigned long long *)address,
                                  (unsigned long long)val);
}

enum class InsertResult : uint8_t {
  Insert,     // Insert into an empty slot.
  Reclaim,    // Insert into a reclaimed slot.
  Assign,     // Hit and assign.
  Evict,      // Evict a key and insert into the evicted slot.
  Duplicated, // Meet duplicated keys on the fly.
  Busy,       // Insert failed as all slots busy.
  Init,
};

__device__ __forceinline__ bool isInsertSuccess(InsertResult result) {
  if (static_cast<uint8_t>(result) <=
      static_cast<uint8_t>(InsertResult::Evict)) {
    return true;
  }
  return false;
}

// Select from double buffer.
// If i % 2 == 0, select buffer 0, else buffer 1.
__forceinline__ __device__ int same_buf(int i) { return (i & 0x01) ^ 0; }
// If i % 2 == 0, select buffer 1, else buffer 0.
__forceinline__ __device__ int diff_buf(int i) { return (i & 0x01) ^ 1; }

template <typename T, int N, int Stride,
          typename = std::enable_if_t<sizeof(T) * Stride <= 16>>
__forceinline__ __device__ void async_copy_bulk(T *dst, T const *src) {
  static_assert(N % Stride == 0);
  // dst = (ScoreType*)__cvta_generic_to_shared((void*)dst);
#pragma unroll
  for (int i = 0; i < N; i += Stride) {
    __pipeline_memcpy_async(dst + i, src + i, sizeof(T) * Stride);
  }
}

template <typename KeyType_,
          typename = std::enable_if_t<std::is_integral_v<KeyType_> &&
                                      sizeof(KeyType_) == 8>>
struct LinearBucket {

  __forceinline__ __device__ LinearBucket(uint8_t *storage, uint32_t capacity)
      : storage_(storage), capacity_(capacity) {}

  __forceinline__ __device__ LinearBucket() : LinearBucket(nullptr, 0) {}

  /*
  Iterator:
  */
  using Iterator = uint32_t;

  template <uint32_t AlignSize>
  static __forceinline__ __device__ int align(Iterator &iter) {
    // iter - (iter % AlignSize)
    constexpr uint32_t MASK = 0xffffffffU - (AlignSize - 1);
    return iter & MASK;
  }

  /*
  Keys:
  */
  using KeyType = KeyType_;
  using AtomicKey = cuda::atomic<KeyType, cuda::thread_scope_device>;

  static constexpr uint64_t EmptyKey = UINT64_C(0xFFFFFFFFFFFFFFFF);
  static constexpr uint64_t LockedKey = UINT64_C(0xFFFFFFFFFFFFFFFD);
  static constexpr uint64_t ReclaimKey = UINT64_C(0xFFFFFFFFFFFFFFFE);

  static constexpr uint64_t ReserveKeyMask = UINT64_C(0xFFFFFFFFFFFFFFFC);

  static __device__ __forceinline__ uint64_t hash(uint64_t key) {
    uint64_t k = key;
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return static_cast<uint64_t>(k);
  }

  static __device__ __forceinline__ KeyType empty_key() { return EmptyKey; }

  static __device__ __forceinline__ KeyType reclaimed_key() {
    return ReclaimKey;
  }

  static __device__ __forceinline__ DigestType empty_digest() {
    auto hashcode = hash(EmptyKey);
    return hashcode_to_digest(hashcode);
  }

  static __device__ __forceinline__ bool is_valid(uint64_t const &key) {
    return (key & ReserveKeyMask) != ReserveKeyMask;
  }

  __device__ __forceinline__ bool is_empty(Iterator &iter) const {
    auto key_slot = reinterpret_cast<AtomicKey *>(keys(iter));
    auto slot_key = key_slot->load(cuda::std::memory_order_relaxed);
    return slot_key == EmptyKey;
  }

  __device__ __forceinline__ bool is_locked(Iterator &iter) const {
    auto key_slot = reinterpret_cast<AtomicKey *>(keys(iter));
    auto slot_key = key_slot->load(cuda::std::memory_order_relaxed);
    return slot_key == LockedKey;
  }

  __device__ __forceinline__ bool try_lock(Iterator &iter, KeyType &key) {
    auto key_slot = reinterpret_cast<AtomicKey *>(keys(iter));
    return key_slot->compare_exchange_strong(
        key, static_cast<KeyType>(LockedKey), cuda::std::memory_order_acquire,
        cuda::std::memory_order_relaxed);
  }

  __device__ __forceinline__ void unlock(Iterator &iter, KeyType key) {
    auto key_slot = reinterpret_cast<AtomicKey *>(keys(iter));
    key_slot->store(key, cuda::std::memory_order_release);
  }

  /*
  Digest:
  */
  using DigestVector = uint32_t; // used for comparison
  using DigestBuffer = uint4;    // used for loading
  using ComparedResult = int;

  static constexpr int VectorDim = sizeof(DigestVector) / sizeof(DigestType);
  static constexpr int BufferDim = sizeof(DigestBuffer) / sizeof(DigestType);
  static constexpr int NumVectorPerBuffer =
      sizeof(DigestBuffer) / sizeof(DigestVector);

  struct VectorComparator {
    static __device__ __forceinline__ ComparedResult compare(DigestVector lhs,
                                                             DigestVector rhs) {
      // Perform a vectorized comparison by byte,
      // and if they are equal, set the corresponding byte in the result to
      // 0xff.
      ComparedResult cmp_result = __vcmpeq4(lhs, rhs);
      cmp_result &= 0x01010101;
      return cmp_result;
    }

    static __device__ __forceinline__ int
    equal_index(ComparedResult &cmp_result) {
      if (cmp_result == 0)
        return -1;
      // CUDA uses little endian,
      // and the lowest byte in register stores in the lowest address.
      uint32_t index = (__ffs(cmp_result) - 1) >> 3;
      cmp_result &= (cmp_result - 1);
      return index;
    }
  };

  static __device__ __forceinline__ DigestType
  hashcode_to_digest(uint64_t hashcode) {
    return static_cast<DigestType>(hashcode >> 32);
  }

  static __device__ __forceinline__ DigestType key_to_digest(KeyType key) {
    auto hashcode = hash(key);
    return hashcode_to_digest(hashcode);
  }

  static __device__ __forceinline__ DigestVector
  digest_to_vector(DigestType digest) {
    return static_cast<DigestVector>(__byte_perm(digest, digest, 0x0000));
  }

  static __device__ __forceinline__ void
  digest_buffer_to_vector(DigestBuffer const &digest_buffer,
                          DigestVector digest_vec[NumVectorPerBuffer]) {
    digest_vec[0] = digest_buffer.x;
    digest_vec[1] = digest_buffer.y;
    digest_vec[2] = digest_buffer.z;
    digest_vec[3] = digest_buffer.w;
  }

  /*
  Scores:
  */
  static constexpr uint64_t EmptyScore = UINT64_C(0);
  static constexpr uint64_t MaxScore = UINT64_C(0xFFFFFFFFFFFFFFFF);
  using ScoreVector = uint4;
  static constexpr int NumScorePerVector =
      sizeof(ScoreVector) / sizeof(ScoreType);

  /*
   */
  static constexpr int KeyOffset = 0;
  static constexpr int DigestOffset = KeyOffset + sizeof(KeyType);
  static constexpr int ScoreOffset = DigestOffset + sizeof(DigestType);
  static constexpr int BucketBytes = ScoreOffset + sizeof(ScoreType);

  static __device__ __forceinline__ uint64_t memory_usage(int size) {
    return BucketBytes * size;
  }

  __forceinline__ __device__ uint32_t capacity() const { return capacity_; }

  __forceinline__ __device__ KeyType *keys(const Iterator &iter) const {
    return reinterpret_cast<KeyType *>(storage_ + KeyOffset * capacity_) + iter;
  }

  __forceinline__ __device__ DigestType *digests(const Iterator &iter) const {
    return reinterpret_cast<DigestType *>(storage_ + DigestOffset * capacity_) +
           iter;
  }

  __forceinline__ __device__ ScoreType *scores(const Iterator &iter) const {
    return reinterpret_cast<ScoreType *>(storage_ + ScoreOffset * capacity_) +
           iter;
  }

  enum class ProbeResult : uint8_t {
    Init = 0,
    Existed = 1,
    Empty = 2,
    Exhausted = 3,
    Failed = 4,
    Absent = 5,
  };
  /*
  Let iter and step have a state, and if they have been probed, they will not be
  probed again
  */
  template <int GroupSize = 1>
  __forceinline__ __device__ ProbeResult probe(KeyType key, Iterator &iter,
                                               int &step) const {
    static_assert(GroupSize == 1);
    if (not storage_) {
      step = capacity_;
      return ProbeResult::Failed;
    }

    if (step == capacity_) {
      return ProbeResult::Exhausted;
    }

    auto hashcode = hash(key);
    auto digest = hashcode_to_digest(hashcode);
    auto digest_vec = digest_to_vector(digest);

    // bool early_stop = false; // used when GroupSize > 1
    if (storage_ == nullptr or capacity_ == 0) {
      // early_stop = true;
      return ProbeResult::Failed;
    }

    if (iter < 0 or iter > capacity_) {
      iter = hashcode % capacity_;
    }

    constexpr int Stride = BufferDim;
    iter = align<Stride>(iter);

    auto empty_digest = key_to_digest(EmptyKey);
    auto empty_vec = digest_to_vector(empty_digest);

    ProbeResult result = ProbeResult::Init;

    for (; step < capacity_; step += Stride) {

      auto buffer = *(reinterpret_cast<DigestBuffer *>(digests(iter)));
      // DigestBuffer buffer;

      constexpr int Length = NumVectorPerBuffer;

      DigestVector vec[Length] = {buffer.x, buffer.y, buffer.z, buffer.w};
      // digest_buffer_to_vector(buffer, vec);

      // vec[0] = buffer.x;
      // vec[1] = buffer.y;
      // vec[2] = buffer.z;
      // vec[3] = buffer.w;

      for (int i = 0; i < Length; i++) {

        int cmp_res = VectorComparator::compare(vec[i], digest_vec);
        while (true) {
          int offset = VectorComparator::equal_index(cmp_res);
          if (offset < 0)
            break;

          auto possible_iter = iter + i * VectorDim + offset;

          auto possible_key_slot =
              reinterpret_cast<AtomicKey *>(keys(possible_iter));

          auto possible_key =
              possible_key_slot->load(cuda::std::memory_order_relaxed);

          if (possible_key == key) {
            iter = possible_iter;
            return ProbeResult::Existed;
          }
        }
        cmp_res = VectorComparator::compare(vec[i], empty_vec);
        while (true) {
          int offset = VectorComparator::equal_index(cmp_res);
          if (offset < 0)
            break;

          auto possible_iter = iter + i * VectorDim + offset;

          auto possible_key_slot =
              reinterpret_cast<AtomicKey *>(keys(possible_iter));

          auto possible_key =
              possible_key_slot->load(cuda::std::memory_order_relaxed);

          if (possible_key == EmptyKey) {
            iter = possible_iter;
            return ProbeResult::Empty;
          }
        }
      }
      iter = (iter + Stride) % capacity_;
    }
    return ProbeResult::Exhausted;
  }

  template <int GroupSize, int BufferDim>
  __forceinline__ __device__ bool reduce(Iterator &dst_iter, KeyType &dst_key,
                                         ScoreType &dst_score,
                                         ScoreType *sm_buffers) const {

    static_assert(GroupSize == 1);

    static constexpr int BulkDim = BufferDim / 2;
    static_assert(BulkDim == 4);

    static constexpr int Stride = NumScorePerVector;

    Iterator iter = 0;
    int rank = threadIdx.x;

    // ScoreType* sm_buffers =
    // (ScoreType*)__cvta_generic_to_shared(sm_buffers_); sm_buffers =
    // reinterpret_cast<ScoreType
    // *>(__cvta_generic_to_shared((void*)sm_buffers));

    // asm("cvta.to.shared.u64 %0, %1;" : "=l"(sm_buffers) : "l"(sm_buffers));

    async_copy_bulk<ScoreType, BulkDim, Stride>(&sm_buffers[rank * BufferDim],
                                                scores(iter));
    __pipeline_commit();

    bool succeed = false;

    for (; iter < capacity_; iter += BulkDim) {
      if (iter < capacity_ - BulkDim) {
        async_copy_bulk<ScoreType, BulkDim, Stride>(
            &sm_buffers[rank * BufferDim] + diff_buf(iter / BulkDim) * BulkDim,
            scores(iter) + BulkDim);
      }
      __pipeline_commit();
      __pipeline_wait_prior(1);
      ScoreType temp_scores[Stride];
      ScoreType *src =
          sm_buffers + rank * BufferDim + same_buf(iter / BulkDim) * BulkDim;
#pragma unroll
      for (int k = 0; k < BulkDim; k += Stride) {
        *reinterpret_cast<ScoreVector *>(temp_scores) =
            *reinterpret_cast<ScoreVector *>(src + k);
#pragma unroll
        for (int j = 0; j < Stride; j += 1) {
          ScoreType temp_score = temp_scores[j];
          if (temp_score < dst_score) {
            auto temp_key_slot =
                reinterpret_cast<AtomicKey *>(keys(iter + k + j));

            auto temp_key =
                temp_key_slot->load(cuda::std::memory_order_relaxed);

            if (temp_key != LockedKey && temp_key != EmptyKey) {
              dst_iter = iter + k + j;
              dst_key = temp_key;
              dst_score = temp_score;
              succeed = true;
            }
          }
        }
      }
    }
    return succeed;
  }

  uint8_t *__restrict__ storage_;
  uint32_t capacity_;
};

template <typename BucketType_> struct LinearBucketTable {
  using BucketType = BucketType_;
  using KeyType = typename BucketType::KeyType;

  LinearBucketTable(uint8_t *storage, uint64_t num_buckets,
                    uint32_t bucket_capacity)
      : storage_(storage), num_buckets_(num_buckets),
        bucket_capacity_(bucket_capacity) {}

  static __device__ __forceinline__ uint64_t hash(uint64_t key) {
    return BucketType::hash(key);
  }

  __device__ __forceinline__ BucketType operator[](uint64_t idx) const {
    // assert(idx < num_buckets_);
    auto bucket_raw_data =
        storage_ + BucketType::memory_usage(bucket_capacity_) * idx;
    return BucketType(bucket_raw_data, bucket_capacity_);
  }

  __device__ __forceinline__ uint64_t capacity() const {
    return num_buckets_ * bucket_capacity_;
  }

  __device__ __forceinline__ uint32_t bucket_capacity() const {
    return bucket_capacity_;
  }

  __device__ __forceinline__ BucketType get_bucket(KeyType key) const {
    auto hashcode = hash(key);
    auto idx = hashcode / bucket_capacity_;
    auto bucket_raw_data =
        storage_ + BucketType::memory_usage(bucket_capacity_) * idx;
    return BucketType(bucket_raw_data, bucket_capacity_);
  }

  uint8_t *__restrict__ storage_;
  uint64_t num_buckets_;
  uint32_t bucket_capacity_;
};

} // namespace dyn_emb