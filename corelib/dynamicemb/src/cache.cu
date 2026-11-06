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

// Lightweight cache index. See docs/lightweight_cache_design.md.
//
// Per table: a pool of C slots (slot number == row of the value buffer), the
// key a slot holds, a reference count, and a hash index key -> slot made of
// singly linked bucket chains. Slot allocation is independent of the index: a
// per-table eviction cursor marks where the last allocation ended, and the new
// keys take the first unpinned slots from there in order (FIFO), so an insert
// succeeds whenever the table has enough unpinned slots, however the pins are
// spread. Key r of a table first looks at the slot r places after the cursor;
// a key whose slot is pinned takes slots from further on, claimed a warp at a
// time with a compare-and-swap on a counter of how far the search has gone.
//
// A slot is pinned by a reference count: `lookup` adds one for every key it
// finds, `insert_and_evict` gives every new slot a count of one, and `release`
// takes one back. A pinned slot is never handed out again.
//
// The Python side owns the value rows; this object owns only the index.

#include "check.h"
#include "murmur_hash.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#ifdef DEMB_USE_PYBIND11
#include <torch/extension.h>
#endif

namespace py = pybind11;

namespace dyn_emb {

namespace {

constexpr int64_t kEmptyKey = -1; // all bits set: the hash table's EmptyKey
constexpr uint32_t kNil = 0x7FFFFFFFu;  // end of chain; also the link mask
constexpr uint32_t kLock = 0x80000000u; // bit 31 of a bucket head

// Plain device-pointer view of the cache, passed to kernels by value. All
// tables live in flat arrays; slot_off / bucket_off give each table's start.
struct CacheView {
  int64_t *keys;             // [sum C]  kEmptyKey = no key
  int32_t *refs;             // [sum C]
  uint32_t *next;            // [sum C]  table-local slot, kNil = end
  uint32_t *heads;           // [sum B]  bit 31 = lock, low bits = slot or kNil
  const int64_t *slot_off;   // [T + 1]
  const int64_t *bucket_off; // [T + 1]
  const int64_t *bmask;      // [T]      B - 1
  int32_t *err;              // [1]      invariant violations seen by kernels
};

__device__ __forceinline__ uint32_t bucket_of(const CacheView &v, int64_t t,
                                              int64_t key) {
  return static_cast<uint32_t>(murmur3_fmix64(static_cast<uint64_t>(key)) &
                               static_cast<uint64_t>(v.bmask[t]));
}

__device__ __forceinline__ uint32_t lock_bucket(uint32_t *h) {
  uint32_t old;
  do {
    old = atomicOr(h, kLock);
  } while (old & kLock);
  return old & kNil; // the head while the lock is held
}
__device__ __forceinline__ void unlock_bucket(uint32_t *h, uint32_t head) {
  __threadfence();
  atomicExch(h, head); // bit 31 cleared
}

// ------------------------------------------------------------------ kernels --

__global__ void lookup_kernel(CacheView v, const int64_t *__restrict__ keys,
                              const int64_t *__restrict__ tids, int64_t n,
                              int64_t *__restrict__ slots) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n)
    return;
  int64_t t = tids[i];
  int64_t k = keys[i];
  int64_t base = v.slot_off[t];
  uint32_t b = bucket_of(v, t, k);
  uint32_t s = __ldg(&v.heads[v.bucket_off[t] + b]) & kNil;
  while (s != kNil) {
    if (__ldg(&v.keys[base + s]) == k) {
      atomicAdd(&v.refs[base + s], 1); // pin what was found
      slots[i] = s;
      return;
    }
    s = __ldg(&v.next[base + s]);
  }
  slots[i] = -1;
}

// Takes one reference back from each slot (slots of -1, the keys a lookup did
// not find, are skipped).
__global__ void release_kernel(CacheView v, const int64_t *slots,
                               const int64_t *tids, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n)
    return;
  int64_t s = slots[i];
  if (s < 0)
    return;
  // the update of the row this step made is visible before the release
  __threadfence();
  int32_t old = atomicSub(&v.refs[v.slot_off[tids[i]] + s], 1);
  if (old <= 0)
    atomicAdd(v.err, 1);
}

// What the kernels of one insert_and_evict call write for each other and for
// the host, in one int64 buffer (zero at the start) that the host reads back in
// one go at the end. The fields are views into it; `over` lays them out, on the
// device buffer for the kernels and on its host copy for the host.
struct AllocWork {
  // table t owns the keys [table_range[t], table_range[t + 1]); table_range[T] = n
  int64_t *table_range; // [T + 1]
  // 0 = fine, kIdsNotAscending, kIdsOutOfRange; written by table_range_kernel
  int64_t *ids_status; // [1]
  // how far the search for table t has advanced beyond the first N positions
  // (keys whose own position was pinned look there); the next search starts at
  // position N + advanced[t]
  int64_t *advanced; // [T]
  // 1 = table t has fewer unpinned slots than keys; written by alloc_slots_kernel
  int64_t *fail; // [T]
  // 1 = the call goes on to change the index (the ids are fine, no table
  // failed); written by finalize_alloc_kernel, read by update_index_kernel
  int64_t *ok; // [1]

  static int64_t size(int64_t T) { return (T + 1) + 1 + T + T + 1; }

  __host__ __device__ static AllocWork over(int64_t *base, int64_t T) {
    AllocWork w;
    w.table_range = base;
    w.ids_status = w.table_range + (T + 1);
    w.advanced = w.ids_status + 1;
    w.fail = w.advanced + T;
    w.ok = w.fail + T;
    return w;
  }
};

constexpr int64_t kIdsNotAscending = 1;
constexpr int64_t kIdsOutOfRange = 2;

// One pass over the table ids, which must be grouped by table in ascending
// order: fills w.table_range (the first position with table id >= t, for t = 0..T)
// and w.ids_status. A thread whose id differs from its predecessor's starts a
// table, and also the tables between the two ids, which are empty.
__global__ void table_range_kernel(const int64_t *tids, int64_t n, int64_t T,
                                    AllocWork w) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n)
    return;
  int64_t cur = tids[i];
  if (cur < 0 || cur >= T) {
    atomicMax(reinterpret_cast<unsigned long long *>(w.ids_status),
              static_cast<unsigned long long>(kIdsOutOfRange));
    return;
  }
  int64_t prev = i == 0 ? -1 : tids[i - 1];
  if (cur < prev) {
    atomicMax(reinterpret_cast<unsigned long long *>(w.ids_status),
              static_cast<unsigned long long>(kIdsNotAscending));
    return;
  }
  for (int64_t t = prev < 0 ? 0 : prev + 1; t <= cur; t++)
    w.table_range[t] = i;
  if (i == n - 1)
    for (int64_t t = cur + 1; t <= T; t++)
      w.table_range[t] = n;
}

constexpr unsigned kFullMask = 0xFFFFFFFFu;

// Gives every key a slot of its table. Positions are offsets from the table's
// cursor, 0 .. C - 1, and each is looked at once. Key r (its number among its
// table's N keys) first looks at position r: free, it takes that slot. A key
// that finds it pinned is "needy" and takes slots from the positions from N on,
// which a counter advanced[t] hands out in order: the needy keys of a warp that
// belong to one table read the counter, load the next 32 reference counts
// together, and one compare-and-swap moves the counter past the free slots they
// take (or past all 32 if there are fewer than they need). The slots handed out
// are exactly the first unpinned slots from the cursor, and the cursor ends just
// after the last of them. Only slots seen pinned are ever skipped. A table that
// runs out of positions is marked failed; nothing else is changed here.
__global__ void alloc_slots_kernel(CacheView v, const int64_t *caps,
                                   const int64_t *cursors, AllocWork w,
                                   const int64_t *tids, int64_t n,
                                   int64_t *slots) {
  if (*w.ids_status != 0) // the table ids are unusable (the same for every thread)
    return;
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int lane = threadIdx.x & 31;

  int64_t t = -1, N = 0, C = 0, P = 0;
  bool need = false;
  if (i < n) {
    t = tids[i];
    N = w.table_range[t + 1] - w.table_range[t];
    C = caps[t];
    if (N > C) {
      w.fail[t] = 1;
    } else {
      P = cursors[t];
      int64_t slot = (P + (i - w.table_range[t])) % C;
      if (__ldcg(&v.refs[v.slot_off[t] + slot]) == 0)
        slots[i] = slot;
      else
        need = true;
    }
  }

  // Every lane of the warp takes part in each round, so that the warp can work
  // on one table at a time. A round whose compare-and-swap lost changes nothing
  // for the keys, so the next round starts from the counter value the failed
  // swap returned instead of reading it again.
  int64_t cur = 0;
  bool have_cur = false;
  while (true) {
    unsigned pending = __ballot_sync(kFullMask, need);
    if (pending == 0)
      break;
    const int ldr = __ffs(pending) - 1;
    const int64_t gt = __shfl_sync(kFullMask, t, ldr);
    const unsigned group = __ballot_sync(kFullMask, need && t == gt);
    const int64_t gN = __shfl_sync(kFullMask, N, ldr);
    const int64_t gC = __shfl_sync(kFullMask, C, ldr);
    const int64_t gP = __shfl_sync(kFullMask, P, ldr);
    if (!have_cur) {
      if (lane == ldr)
        cur = *reinterpret_cast<volatile int64_t *>(&w.advanced[gt]);
      cur = __shfl_sync(kFullMask, cur, ldr);
    }
    have_cur = false;

    const int64_t start = gN + cur; // first position this round looks at
    if (start >= gC) {              // every position has been looked at
      if (lane == ldr)
        w.fail[gt] = 1;
      if (need && t == gt)
        need = false;
      continue;
    }
    const int64_t q = start + lane;
    const bool is_free =
        q < gC && __ldcg(&v.refs[v.slot_off[gt] + (gP + q) % gC]) == 0;
    const unsigned fm = __ballot_sync(kFullMask, is_free);
    const int m = __popc(group); // slots the group still needs
    const int f = __popc(fm);    // free slots in the window
    const int k = m < f ? m : f; // slots this round hands out
    int64_t step;
    if (f >= m) { // up to and including the m-th free slot
      unsigned x = fm;
      for (int s = 1; s < m; s++)
        x &= x - 1;
      step = __ffs(x);
    } else { // all the free slots, and the pinned ones among them: the window
      step = gC - start < 32 ? gC - start : 32;
    }
    int won = 0;
    long long seen = 0;
    if (lane == ldr) {
      unsigned long long old = atomicCAS(
          reinterpret_cast<unsigned long long *>(&w.advanced[gt]),
          static_cast<unsigned long long>(cur),
          static_cast<unsigned long long>(cur + step));
      won = old == static_cast<unsigned long long>(cur);
      seen = static_cast<long long>(old);
    }
    won = __shfl_sync(kFullMask, won, ldr);
    if (!won) { // the counter moved on: try again from the value it has now
      cur = __shfl_sync(kFullMask, seen, ldr);
      have_cur = true;
    }
    if (won && need && t == gt) {
      const int rank = __popc(group & ((1u << lane) - 1u));
      if (rank < k) {
        unsigned x = fm;
        for (int s = 0; s < rank; s++)
          x &= x - 1;
        slots[i] = (gP + start + (__ffs(x) - 1)) % gC;
        need = false;
      }
    }
  }
}

// One block. Decides whether the call goes on (the table ids are fine and no
// table failed) and, if so, moves the cursor of every table that allocated to
// just after its last slot. A failed call leaves the cursors alone.
__global__ void finalize_alloc_kernel(int64_t *cursors, const int64_t *caps,
                                      AllocWork w, int64_t T) {
  __shared__ int bad;
  if (threadIdx.x == 0)
    bad = *w.ids_status != 0;
  __syncthreads();
  for (int64_t t = threadIdx.x; t < T; t += blockDim.x)
    if (w.fail[t])
      atomicOr(&bad, 1);
  __syncthreads();
  if (threadIdx.x == 0)
    *w.ok = bad ? 0 : 1;
  if (bad)
    return;
  for (int64_t t = threadIdx.x; t < T; t += blockDim.x) {
    int64_t N = w.table_range[t + 1] - w.table_range[t];
    if (N > 0)
      cursors[t] = (cursors[t] + N + w.advanced[t]) % caps[t];
  }
}

// For each new key: unlink the slot's old key from its bucket, write the new
// key, link it into its own bucket. One bucket lock held at a time.
// Does nothing unless *ok: a failed allocation changes nothing.
__global__ void update_index_kernel(CacheView v, const int64_t *slots,
                                    const int64_t *new_keys,
                                    const int64_t *tids, int64_t n,
                                    int64_t *evicted_keys, const int64_t *ok) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n || *ok == 0)
    return;
  int64_t t = tids[i];
  int64_t base = v.slot_off[t];
  uint32_t *heads = v.heads + v.bucket_off[t];
  volatile uint32_t *nx = v.next + base;
  uint32_t s = static_cast<uint32_t>(slots[i]);

  int64_t old = v.keys[base + s];
  evicted_keys[i] = old;
  // The slot was found unpinned and nothing else raises a count while this call
  // runs, so a plain store pins it for the caller.
  v.refs[base + s] = 1;
  if (old != kEmptyKey) {
    uint32_t b = bucket_of(v, t, old);
    uint32_t h = lock_bucket(&heads[b]);
    uint32_t newh = h;
    if (h == s) {
      newh = nx[s];
    } else {
      uint32_t p = h;
      while (p != kNil && nx[p] != s)
        p = nx[p];
      if (p == kNil)
        atomicAdd(v.err, 1); // the slot was not on its chain
      else
        nx[p] = nx[s];
    }
    unlock_bucket(&heads[b], newh);
  }
  int64_t nk = new_keys[i];
  v.keys[base + s] = nk;
  uint32_t nb = bucket_of(v, t, nk);
  uint32_t h = lock_bucket(&heads[nb]);
  nx[s] = h;
  unlock_bucket(&heads[nb], s);
}

__global__ void erase_kernel(CacheView v, const int64_t *keys,
                             const int64_t *tids, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n)
    return;
  int64_t t = tids[i];
  int64_t k = keys[i];
  int64_t base = v.slot_off[t];
  uint32_t *heads = v.heads + v.bucket_off[t];
  volatile uint32_t *nx = v.next + base;
  uint32_t b = bucket_of(v, t, k);
  uint32_t h = lock_bucket(&heads[b]);
  uint32_t prev = kNil, p = h;
  while (p != kNil && v.keys[base + p] != k) {
    prev = p;
    p = nx[p];
  }
  uint32_t newh = h;
  if (p != kNil) {
    if (prev == kNil)
      newh = nx[p];
    else
      nx[prev] = nx[p];
    nx[p] = kNil;
    v.keys[base + p] = kEmptyKey;
    // The caller erases keys no step holds; the count is left alone (it belongs
    // to the step that pinned the slot, whose release is still to come), but the
    // violation is counted.
    if (v.refs[base + p] != 0)
      atomicAdd(v.err, 1);
  }
  unlock_bucket(&heads[b], newh);
}

inline int blocks_for(int64_t n, int threads = 256) {
  int64_t b = (n + threads - 1) / threads;
  return static_cast<int>(b < 1 ? 1 : (b > (1 << 30) ? (1 << 30) : b));
}

} // namespace

// ------------------------------------------------------------------- class --

class CacheCore {
public:
  CacheCore(std::vector<int64_t> capacities, int64_t device_index)
      : caps_(std::move(capacities)), device_(device_index) {
    TORCH_CHECK(!caps_.empty(), "CacheCore: no tables");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    const int64_t T = static_cast<int64_t>(caps_.size());
    slot_off_h_.assign(T + 1, 0);
    bucket_off_h_.assign(T + 1, 0);
    std::vector<int64_t> bmask(T);
    for (int64_t t = 0; t < T; t++) {
      TORCH_CHECK(caps_[t] > 0 && caps_[t] < static_cast<int64_t>(kNil),
                  "CacheCore: capacity of table ", t,
                  " must be in [1, 2^31-2], got ", caps_[t]);
      int64_t B = 1;
      while (B < caps_[t])
        B <<= 1;
      slot_off_h_[t + 1] = slot_off_h_[t] + caps_[t];
      bucket_off_h_[t + 1] = bucket_off_h_[t] + B;
      bmask[t] = B - 1;
    }
    auto i64 = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device_);
    auto i32 = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, device_);
    auto cpu64 = at::TensorOptions().dtype(at::kLong);
    keys_ = at::empty({slot_off_h_[T]}, i64);
    refs_ = at::empty({slot_off_h_[T]}, i32);
    next_ = at::empty({slot_off_h_[T]}, i32);
    heads_ = at::empty({bucket_off_h_[T]}, i32);
    slot_off_ = at::from_blob(slot_off_h_.data(), {T + 1}, cpu64)
                    .to(i64, /*non_blocking=*/false, /*copy=*/true);
    bucket_off_ = at::from_blob(bucket_off_h_.data(), {T + 1}, cpu64)
                      .to(i64, false, true);
    bmask_ = at::from_blob(bmask.data(), {T}, cpu64).to(i64, false, true);
    caps_dev_ = at::from_blob(caps_.data(), {T}, cpu64).to(i64, false, true);
    cursors_ = at::zeros({T}, i64);
    err_ = at::zeros({1}, i32);
    reset();
  }

  void reset() {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    keys_.fill_(kEmptyKey);
    refs_.zero_();
    next_.fill_(static_cast<int64_t>(kNil));
    heads_.fill_(static_cast<int64_t>(kNil));
    cursors_.zero_();
    err_.zero_();
  }

  // slots[i] = table-local slot of keys[i], -1 if not cached. Every slot found
  // gets one more reference (the keys must be unique).
  at::Tensor lookup(at::Tensor keys, at::Tensor table_ids) {
    check_pair(keys, table_ids);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t n = keys.numel();
    at::Tensor slots = at::empty({n}, keys.options());
    if (n == 0)
      return slots;
    lookup_kernel<<<blocks_for(n), 256, 0, stream>>>(
        view(), keys.data_ptr<int64_t>(), table_ids.data_ptr<int64_t>(), n,
        slots.data_ptr<int64_t>());
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    return slots;
  }

  // Takes one reference back from each slot (-1 entries are skipped).
  void release(at::Tensor slots, at::Tensor table_ids) {
    check_pair(slots, table_ids);
    if (slots.numel() == 0)
      return;
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    release_kernel<<<blocks_for(slots.numel()), 256, 0, stream>>>(
        view(), slots.data_ptr<int64_t>(), table_ids.data_ptr<int64_t>(),
        slots.numel());
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // Gives every key a slot: the first unpinned slots in FIFO order of its
  // table. `keys` must be grouped by table in ascending table id (keys that
  // are not in the cache and are unique). Returns (slots, evicted_keys): the
  // table-local slot of each key, and the key that slot held before
  // (kEmptyKey if none). Each new slot is pinned (one reference).
  // Throws, changing nothing, when a table has fewer unpinned slots than keys.
  std::tuple<at::Tensor, at::Tensor> insert_and_evict(at::Tensor keys,
                                                      at::Tensor table_ids) {
    check_pair(keys, table_ids);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t n = keys.numel();
    const int64_t T = static_cast<int64_t>(caps_.size());
    at::Tensor slots = at::empty({n}, keys.options());
    at::Tensor evicted = at::empty({n}, keys.options());
    if (n == 0)
      return {slots, evicted};

    // The kernels of the call pass their results to each other in `work` (see
    // AllocWork); the host reads it back once at the end.
    at::Tensor work = at::zeros({AllocWork::size(T)}, keys.options());
    const AllocWork w = AllocWork::over(work.data_ptr<int64_t>(), T);
    table_range_kernel<<<blocks_for(n), 256, 0, stream>>>(
        table_ids.data_ptr<int64_t>(), n, T, w);
    alloc_slots_kernel<<<blocks_for(n), 256, 0, stream>>>(
        view(), caps_dev_.data_ptr<int64_t>(), cursors_.data_ptr<int64_t>(), w,
        table_ids.data_ptr<int64_t>(), n, slots.data_ptr<int64_t>());
    finalize_alloc_kernel<<<1, 256, 0, stream>>>(
        cursors_.data_ptr<int64_t>(), caps_dev_.data_ptr<int64_t>(), w, T);
    update_index_kernel<<<blocks_for(n), 256, 0, stream>>>(
        view(), slots.data_ptr<int64_t>(), keys.data_ptr<int64_t>(),
        table_ids.data_ptr<int64_t>(), n, evicted.data_ptr<int64_t>(), w.ok);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();

    at::Tensor work_host = work.to(at::kCPU); // the one sync
    const AllocWork r = AllocWork::over(work_host.data_ptr<int64_t>(), T);
    TORCH_CHECK(*r.ids_status != kIdsOutOfRange,
                "CacheCore::insert_and_evict: table id out of range [0, ", T,
                ")");
    TORCH_CHECK(*r.ids_status != kIdsNotAscending,
                "CacheCore::insert_and_evict: keys must be grouped by "
                "table in ascending table id");
    for (int64_t t = 0; t < T; t++) {
      if (r.fail[t] == 0)
        continue;
      const int64_t N = r.table_range[t + 1] - r.table_range[t];
      TORCH_CHECK(N <= caps_[t], "CacheCore: table ", t, " needs ", N,
                  " slots but has only ", caps_[t]);
      const int64_t free_slots = (refs_of(t) == 0).sum().item<int64_t>();
      TORCH_CHECK(false, "CacheCore: table ", t, " needs ", N,
                  " free slots but only ", free_slots, " of its ", caps_[t],
                  " slots are not pinned by in-flight steps. Increase the "
                  "cache capacity or reduce the prefetch depth / batch size.");
    }
    return {slots, evicted};
  }

  // Drops the keys from the index (keys that are not cached are ignored).
  // The slots become free for the cursor to reach; their rows are untouched.
  void erase(at::Tensor keys, at::Tensor table_ids) {
    check_pair(keys, table_ids);
    if (keys.numel() == 0)
      return;
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    erase_kernel<<<blocks_for(keys.numel()), 256, 0, stream>>>(
        view(), keys.data_ptr<int64_t>(), table_ids.data_ptr<int64_t>(),
        keys.numel());
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // Number of invariant violations kernels have seen since the last call
  // (a release below zero, a slot missing from its chain, an erase of a pinned
  // slot). Syncs.
  int64_t check_errors() {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    int64_t e = err_.item<int32_t>();
    err_.zero_();
    return e;
  }

  // Views of one table's arrays (no copy), for export / size / tests.
  at::Tensor keys_of(int64_t t) { return slice(keys_, slot_off_h_, t); }
  at::Tensor refs_of(int64_t t) { return slice(refs_, slot_off_h_, t); }
  at::Tensor next_of(int64_t t) { return slice(next_, slot_off_h_, t); }
  at::Tensor heads_of(int64_t t) { return slice(heads_, bucket_off_h_, t); }
  at::Tensor cursors() { return cursors_; }
  std::vector<int64_t> capacities() const { return caps_; }
  // device bytes of everything but the values: the keys, the reference counts,
  // the hash chains and buckets, the per-table offsets and cursors
  int64_t meta_bytes() const {
    return keys_.nbytes() + refs_.nbytes() + next_.nbytes() + heads_.nbytes() +
           slot_off_.nbytes() + bucket_off_.nbytes() + bmask_.nbytes() +
           caps_dev_.nbytes() + cursors_.nbytes() + err_.nbytes();
  }

private:
  void check_pair(const at::Tensor &a, const at::Tensor &tids) const {
    TORCH_CHECK(a.is_cuda() && tids.is_cuda(),
                "CacheCore: tensors must be on CUDA");
    TORCH_CHECK(a.device().index() == device_,
                "CacheCore: tensors are on a different device");
    TORCH_CHECK(a.dtype() == at::kLong && tids.dtype() == at::kLong,
                "CacheCore: keys/slots and table_ids must be int64");
    TORCH_CHECK(a.is_contiguous() && tids.is_contiguous(),
                "CacheCore: tensors must be contiguous");
    TORCH_CHECK(a.numel() == tids.numel(),
                "CacheCore: keys/slots and table_ids differ in length");
  }
  CacheView view() {
    CacheView v;
    v.keys = keys_.data_ptr<int64_t>();
    v.refs = refs_.data_ptr<int32_t>();
    v.next = reinterpret_cast<uint32_t *>(next_.data_ptr<int32_t>());
    v.heads = reinterpret_cast<uint32_t *>(heads_.data_ptr<int32_t>());
    v.slot_off = slot_off_.data_ptr<int64_t>();
    v.bucket_off = bucket_off_.data_ptr<int64_t>();
    v.bmask = bmask_.data_ptr<int64_t>();
    v.err = err_.data_ptr<int32_t>();
    return v;
  }
  static at::Tensor slice(const at::Tensor &x, const std::vector<int64_t> &off,
                          int64_t t) {
    TORCH_CHECK(t >= 0 && t + 1 < static_cast<int64_t>(off.size()),
                "table id out of range");
    return x.narrow(0, off[t], off[t + 1] - off[t]);
  }

  std::vector<int64_t> caps_;
  int64_t device_;
  std::vector<int64_t> slot_off_h_, bucket_off_h_;
  at::Tensor keys_, refs_, next_, heads_;
  at::Tensor slot_off_, bucket_off_, bmask_, caps_dev_;
  at::Tensor cursors_, err_;
};

} // namespace dyn_emb

void bind_cache(py::module &m) {
  using dyn_emb::CacheCore;
  py::class_<CacheCore>(m, "CacheCore")
      .def(py::init<std::vector<int64_t>, int64_t>(), py::arg("capacities"),
           py::arg("device_index"))
      .def("lookup", &CacheCore::lookup, py::arg("keys"),
           py::arg("table_ids"))
      .def("release", &CacheCore::release, py::arg("slots"),
           py::arg("table_ids"))
      .def("insert_and_evict", &CacheCore::insert_and_evict,
           py::arg("keys"), py::arg("table_ids"))
      .def("erase", &CacheCore::erase, py::arg("keys"),
           py::arg("table_ids"))
      .def("reset", &CacheCore::reset)
      .def("check_errors", &CacheCore::check_errors)
      .def("keys_of", &CacheCore::keys_of, py::arg("table_id"))
      .def("refs_of", &CacheCore::refs_of, py::arg("table_id"))
      .def("next_of", &CacheCore::next_of, py::arg("table_id"))
      .def("heads_of", &CacheCore::heads_of, py::arg("table_id"))
      .def("cursors", &CacheCore::cursors)
      .def("capacities", &CacheCore::capacities)
      .def("meta_bytes", &CacheCore::meta_bytes);
}
