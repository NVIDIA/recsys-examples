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

// ---------------------------------------------------------------------------
// The queue of nodes that may be taken for reuse.
//
// Initialised to arange(capacity), which collapses two structures into one:
// there is no free list of never-used slots alongside an eviction queue of used
// ones. Everything in here is simply "a node nobody is referencing". Some have
// never held data; some hold a valid row whose last user has finished, and
// those are still cache hits right up until the moment they are popped. The
// allocator does not care which it gets.
//
// Which is why this is not an eviction queue: half of what it holds has nothing
// to evict.
//
// ---------------------------------------------------------------------------
// The constraint that makes this simple
// ---------------------------------------------------------------------------
//
// At any moment this queue is touched by popping kernels or by pushing
// kernels, never both. Stream ordering enforces it.
//
// Without that it is the textbook MPMC ring hazard: a pusher bumps `tail` to
// reserve a slot, and before it stores the node id a popper reads `tail`, sees
// that slot as available, and consumes whatever was in it last time round.
// Fixing it properly wants a per-slot sequence number, which is a large step up
// in complexity and in how hard it is to tell when it is wrong.
//
// The hazard is one-directional, which is why one event dependency suffices:
//
//   pop running, push starts   — safe. The pusher writes at a position >= the
//                                popper's tail snapshot; the popper only reads
//                                below it.
//   push in flight, pop starts — unsafe. Exactly the window above.
//
// So an allocating kernel must wait for every already-issued releasing kernel.
// The reverse costs nothing for the kernel itself: a release that starts after
// the popper's bound is fixed writes above it. This holds at any prefetch
// depth; a deeper pipeline only means a released node becomes available later,
// which is a capacity question rather than a correctness one.
//
// The bound a popper honours is therefore a snapshot of `tail`, and it has to be
// one value for the whole kernel. Reading `tail` inside the kernel does not give
// that: blocks start at different times, and a block that starts after a pusher
// has begun would read a `tail` covering slots reserved but not yet written —
// the unsafe window above, reached through the safe direction. So the snapshot
// is taken before the kernel exists, by a copy ordered on the stream, and the
// kernel only reads it.
//
// That copy is itself a boundary between the two phases, and it has to be ordered
// on both sides. A release issued before it must have finished, or the copy can
// read a `tail` that includes reserved-but-unwritten slots. A release issued
// after it must not have started, for the same reason: "issued" is the host's
// order, not the device's, and nothing stops a release launched later from
// running earlier than a copy that is queued behind other work. In stream terms,
// the allocating stream waits for the releasing stream, takes the snapshot, and
// records an event the releasing stream then waits on. Releases that wait on
// that event run freely alongside the popping kernel. See
// ReclaimQueueStorage::take_snapshot.
//
// ---------------------------------------------------------------------------
// Why there are twice as many slots as nodes
// ---------------------------------------------------------------------------
//
// Position p lives in slot p & mask, so it is overwritten when a pusher reaches
// position p + slots. The in_ring bit keeps the number of live entries at or
// below the node count C, and it is tempting to stop there and size the slot
// array to C. It is not enough, because live entries are not what a push has to
// stay clear of — the span from the oldest unread position to `tail` is.
//
// A popper takes a ticket and reads its slot a little later. In between, other
// poppers drop or claim entries behind it, and those entries leave holes in the
// middle of the span; a push goes to `tail`, never into a hole. So the span can
// be wider than the live count, and with only C slots a push can wrap onto a
// position whose popper has its ticket but has not yet read it. That popper then
// reads somebody else's node, and the node whose entry was overwritten still
// carries in_ring, so it is never queued again: a slot lost for the rest of the
// run.
//
// This is not a corner. A releasing kernel overlapping the next allocate is the
// pipeline working as intended, and the case needs only a full queue and one
// slow popper.
//
// The span has a hard bound. At the start of a popping kernel it is at most C,
// the live count. A pusher that starts mid-pop writes at positions at or above
// the popper's tail snapshot, and it can push each node at most once — a pushed
// node holds in_ring, and its new entry is above the snapshot, so no popper
// clears it. That adds at most C more. The span never exceeds 2C, so 2C slots
// cannot be overwritten before they are read.
//
// Rounded up to a power of two so that indexing is a mask rather than a 64-bit
// division, which has no native instruction. The cost is four bytes per slot,
// and the slot array is the only thing here that grows with the node count.
//
// ---------------------------------------------------------------------------
// FIFO only
// ---------------------------------------------------------------------------
//
// No second-chance policy here. CLOCK needs somewhere to put a node it has
// spared — it cannot go straight back on the queue, because writing `tail` from
// inside a popping kernel is exactly the overlap ruled out above — so it needs
// a landing buffer and a follow-up kernel to drain it. That machinery is where
// the nastiest defect in the previous implementation lived, and it misbehaved
// only at full speed, never under a race checker.
//
// Nothing is lost by waiting. NodeState keeps its `used` bit, set for free
// inside the pin, and the fraction of popped nodes carrying it is exactly where
// CLOCK's benefit would come from — measurable while running FIFO, and it
// already excludes the pinned nodes that stale-dropping protects under either
// policy. Build it when that number says to.
// ---------------------------------------------------------------------------

#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "allocator.h"
#include "ec_assert.h"
#include "launch.h"
#include "reserve.h"

#ifndef __CUDACC__
#error \
    "embedding_cache/reclaim_queue.h launches kernels and must be compiled by \
nvcc. Including it from a .cpp will not work; make the translation unit a .cu."
#endif

namespace embedding_cache {

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// ReclaimQueue — the view, passed to kernels by value
// ---------------------------------------------------------------------------

template <typename NodeIdT = int32_t,
          NodeIdT kNoNodeV = static_cast<NodeIdT>(-1)>
struct ReclaimQueue {
  using NodeId = NodeIdT;

  // Returned by pop() when the queue is exhausted, and it has to be the same
  // value HashMap uses for "no node": the two travel together through a caller
  // that cannot tell whose it is holding.
  //
  // Parameters rather than fixed constants so that agreement can be derived
  // instead of restated —
  //
  //   using Queue = ReclaimQueue<Map::NodeId, Map::kNoNode>;
  //
  // which is one place to change instead of two to keep in step.
  static constexpr NodeId kNoNode = kNoNodeV;

  static_assert(std::is_integral<NodeIdT>::value,
                "NodeId must be integral: it indexes the node pool");

  // -- state, all borrowed --------------------------------------------------

  NodeId* slots;  // [mask + 1]

  // Monotonic, and 64-bit so they never wrap in any real run. Counters rather
  // than wrapped indices, because then `tail - head` is the live count with no
  // ambiguity between full and empty.
  uint64_t* head;
  uint64_t* tail;
  uint64_t* snapshot;  // written only by ReclaimQueueStorage::take_snapshot

  // The slot count minus one. The slot count is a power of two, at least twice
  // the node count; see the header for why it is that large. Holding the mask
  // and not the count means there is one field to keep right rather than two.
  uint64_t mask;

  // -- block-collective scratch ---------------------------------------------
  //
  // One per block, in shared memory:
  //
  //   __shared__ ReclaimQueue<>::Scratch scratch;
  //
  // Aliased rather than redeclared, because reserving a slot on this queue and
  // reserving one in a compaction output are the same operation — see
  // reserve.h. Left uninitialised on purpose: reserve() zeroes it.
  using Scratch = BlockReserve::Scratch;

  // -- the interface --------------------------------------------------------
  //
  // BOTH ARE BLOCK-COLLECTIVE. Every thread in the block has to call them, the
  // same number of times, in the same order — they contain __syncthreads(), and
  // a thread that skips one hangs the block. `want` is how a thread that has
  // nothing to do still takes part:
  //
  //   for (int64_t base = blockIdx.x * blockDim.x; base < n;
  //        base += gridDim.x * blockDim.x) {
  //     const int64_t i = base + threadIdx.x;
  //     queue.push(i < n && owns_enqueue(i), node_of(i), scratch);
  //   }
  //
  // Note the shape: the loop bound is the same for every thread, and the tail
  // of the range is handled by `want` rather than by exiting early. This is the
  // same discipline unique_op.cu's segmented_unique_core uses, for the same
  // reason.

  // Enqueue a node whose last reference just went away.
  //
  // Pass want=true only when NodeState::unpin said this thread owns the enqueue
  // — that call, not this one, is what decides ownership, and it decides it in
  // the same atomic that took the count to zero.
  //
  // No overflow check, and none can be needed: the in_queue bit admits at most
  // one entry per node, so live entries never exceed the node count.
  //
  // Aggregated two levels deep — see reserve.h. Up to 256 threads end up behind
  // one atomic on `tail`, which matters because every one of them targets the
  // same address, and same-address atomics serialise in L2.
  __forceinline__ __device__ void push(bool want, NodeId node, Scratch& s) {
    const uint64_t pos = BlockReserve::reserve(
        want, reinterpret_cast<unsigned long long*>(tail), s);
    if (want) slots[pos & mask] = node;
  }

  // The bound for a popping kernel: `tail` as it was when take_snapshot ran,
  // the same for every thread. A plain load is enough — the copy that wrote it
  // finished before this kernel started.
  [[nodiscard]] __forceinline__ __device__ uint64_t snapshot_tail() const {
    return *snapshot;
  }

  // Take a node for reuse, or report the queue exhausted.
  //
  // `try_take(node)` decides whether a candidate can be had — in practice
  // AtomicNodeState::try_claim, which refuses a node referenced again since it
  // was enqueued. Passing it in rather than calling it directly is what keeps
  // this file from knowing what a node's state looks like: the retry loop is
  // queue logic, the decision is not.
  //
  // A refused entry is dropped, never requeued. Requeueing would livelock the
  // moment every node is pinned — threads would circle forever finding nothing
  // takeable. Dropping shrinks the queue monotonically until either a node is
  // taken or it runs dry and the caller gets a definite failure. Nothing leaks:
  // try_claim leaves in_queue clear, so the node re-enqueues on its next
  // release.
  //
  // Exhaustion means every node is pinned by an in-flight step, which is a
  // capacity planning error. It is reported, never worked around quietly.
  // Only the first ticket is aggregated. A thread whose candidate is refused
  // needs another one, and how many it needs is not known to anybody else, so
  // the retries cannot be block-collective — a barrier inside that loop would
  // deadlock the moment two threads wanted different numbers of rounds. They
  // fall back to one global atomic each, which is what the old shape did for
  // every ticket.
  //
  // That is the right split: the first attempt succeeds unless the entry is
  // stale, so the aggregated path carries the common case and the unaggregated
  // one carries the exception.
  template <typename TryTake>
  [[nodiscard]] __forceinline__ __device__ NodeId pop(bool want,
                                                      uint64_t tail_snapshot,
                                                      Scratch& s,
                                                      TryTake&& try_take) {
    const uint64_t first = BlockReserve::reserve(
        want, reinterpret_cast<unsigned long long*>(head), s);

    NodeId node = kNoNode;
    bool settled = !want;

    if (want) {
      const uint64_t pos = first;
      if (pos >= tail_snapshot) {
        // Past the bound. Hand the position back, aggregated the same way, or
        // `head` runs past `tail` and the live count underflows. CUDA has no
        // 64-bit atomicSub — only int and unsigned int — so this adds the two's
        // complement instead.
        //
        // `head` cannot end below the snapshot: only a thread that pushed it
        // above brings it back down, and each hands back exactly what it took.
        auto g = cg::coalesced_threads();
        if (g.thread_rank() == 0) {
          atomicAdd(reinterpret_cast<unsigned long long*>(head),
                    ~static_cast<unsigned long long>(g.size()) + 1ull);
        }
        settled = true;  // exhausted, and kNoNode is the answer
      } else {
        const NodeId candidate = slots[pos & mask];
        if (try_take(candidate)) {
          node = candidate;
          settled = true;
        }
      }
    }

    // Retries: one thread, one atomic, no barrier.
    while (!settled) {
      const uint64_t pos =
          atomicAdd(reinterpret_cast<unsigned long long*>(head), 1ULL);
      if (pos >= tail_snapshot) {
        atomicAdd(reinterpret_cast<unsigned long long*>(head), ~0ULL);
        break;
      }
      const NodeId candidate = slots[pos & mask];
      if (try_take(candidate)) {
        node = candidate;
        break;
      }
    }
    return node;
  }
};

// ---------------------------------------------------------------------------
// Initialisation kernel
// ---------------------------------------------------------------------------

namespace detail {

// A template rather than a plain __global__ so that it can live in a header
// without nvcc emitting a duplicate symbol in every translation unit that
// includes it. (`inline` on a non-template __global__ is ignored, with a
// warning that is easy to miss.)
template <typename NodeIdT>
__global__ void queue_arange_kernel(NodeIdT* slots, int64_t capacity) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < capacity;
       i += int64_t(gridDim.x) * blockDim.x) {
    slots[i] = static_cast<NodeIdT>(i);
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// ReclaimQueueStorage
// ---------------------------------------------------------------------------
//
// Owns the slot array and the two counters.
//
// Unlike the node pool there is nothing here belonging to another component, so
// this one can initialise everything it allocates: arange for the first
// `capacity` slots, zero for head, capacity for tail. The rest of the slots are
// not touched; a position is only ever read after a push has written it, or
// after the arange has. A freshly built queue is a complete queue.

template <typename NodeIdT = int32_t,
          NodeIdT kNoNodeV = static_cast<NodeIdT>(-1),
          typename AllocT = CudaAllocator>
class ReclaimQueueStorage {
 public:
  using Queue = ReclaimQueue<NodeIdT, kNoNodeV>;
  using NodeId = NodeIdT;
  using Allocator = AllocT;
  using Buffer = typename AllocT::Buffer;

  static constexpr NodeId kNoNode = Queue::kNoNode;

  static_assert(std::is_trivially_copyable<Queue>::value,
                "the queue view reaches kernels by value");

  // `stream` is kept rather than passed per call, and the constructor is
  // asynchronous with respect to it: it enqueues the arange and returns. The
  // download_* diagnostics are ordered on the same stream and synchronise, so
  // they stay safe to call immediately.
  explicit ReclaimQueueStorage(int64_t capacity, AllocT allocator = AllocT{},
                               cudaStream_t stream = nullptr)
      : capacity_(capacity), stream_(stream), alloc_(std::move(allocator)) {
    if (capacity <= 0) {
      throw std::invalid_argument("ReclaimQueueStorage: capacity must be > 0");
    }
    if (capacity > static_cast<int64_t>(INT32_MAX)) {
      throw std::invalid_argument(
          "ReclaimQueueStorage: capacity exceeds the 32-bit node id range");
    }
    slot_count_ = round_up_pow2(2 * capacity_);
    slots_ = alloc_.allocate(sizeof(NodeId) * size_t(slot_count_));
    // Head, tail and the snapshot in one allocation: they are read together by
    // every kernel that touches the queue, and neighbouring counters share a
    // line.
    counters_ = alloc_.allocate(sizeof(uint64_t) * 3u);
    reset();
  }

  ReclaimQueueStorage(const ReclaimQueueStorage&) = delete;
  ReclaimQueueStorage& operator=(const ReclaimQueueStorage&) = delete;

  [[nodiscard]] Queue queue() const {
    return Queue{d_slots(), d_head(), d_tail(), d_snapshot(),
                 static_cast<uint64_t>(slot_count_ - 1)};
  }

  // Every node queued, in order, nothing taken yet.
  //
  // The caller has to put the node pool into the state that matches — every
  // node with in_queue set — or the two disagree and a node ends up queued
  // twice. That pairing cannot happen here: this component does not know what a
  // node looks like.
  void reset() {
    detail::queue_arange_kernel<<<grid_for(capacity_), kBlock, 0, stream_>>>(
        d_slots(), capacity_);
    check(cudaGetLastError(), "arange launch");

    const uint64_t counters[3] = {0u, static_cast<uint64_t>(capacity_),
                                  static_cast<uint64_t>(capacity_)};
    check(cudaMemcpyAsync(d_head(), counters, sizeof(counters),
                          cudaMemcpyHostToDevice, stream_),
          "counter upload");
    check(cudaStreamSynchronize(stream_), "counter upload sync");
  }

  // Fix the bound the next popping kernel will honour: copy `tail` into the
  // snapshot, on `stream`.
  //
  // This is the boundary between the push phase and the pop phase, and it is
  // ordered on both sides by the caller:
  //
  //   before   `stream` has waited for every release issued so far
  //   after    every release issued from here on waits for an event recorded on
  //            `stream` right after this call
  //
  // Between the two, a release may overlap the popping kernel freely: it writes
  // above the snapshot, which that kernel never reads. Skip either wait and the
  // copy can read a `tail` that covers slots a running release has reserved but
  // not yet written.
  void take_snapshot(cudaStream_t stream) {
    check(cudaMemcpyAsync(d_snapshot(), d_tail(), sizeof(uint64_t),
                          cudaMemcpyDeviceToDevice, stream),
          "snapshot");
  }

  [[nodiscard]] int64_t capacity() const { return capacity_; }
  [[nodiscard]] cudaStream_t stream() const { return stream_; }
  [[nodiscard]] const Allocator& allocator() const { return alloc_; }

  [[nodiscard]] size_t bytes() const {
    return sizeof(NodeId) * size_t(slot_count_) + sizeof(uint64_t) * 3u;
  }

  // -- inspection -----------------------------------------------------------

  // Every slot, including those no live position maps to. Only the positions
  // in [head, tail) mean anything, at slot (position & mask).
  [[nodiscard]] std::vector<NodeId> download_slots() const {
    return download(d_slots(), size_t(slot_count_));
  }

  // {head, tail}.
  [[nodiscard]] std::vector<uint64_t> download_counters() const {
    return download(d_head(), 2u);
  }

  // Entries currently live. Only meaningful when nothing is running.
  [[nodiscard]] int64_t size() const {
    const auto c = download_counters();
    return static_cast<int64_t>(c[1] - c[0]);
  }

 private:
  static int64_t round_up_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
  }
  static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("ReclaimQueueStorage: ") + what +
                               ": " + cudaGetErrorString(e));
    }
  }

  template <typename T>
  std::vector<T> download(const T* src, size_t n) const {
    std::vector<T> out(n);
    check(cudaMemcpyAsync(out.data(), src, n * sizeof(T),
                          cudaMemcpyDeviceToHost, stream_),
          "download");
    check(cudaStreamSynchronize(stream_), "download sync");
    return out;
  }

  NodeId* d_slots() const { return static_cast<NodeId*>(slots_.data()); }
  uint64_t* d_head() const { return static_cast<uint64_t*>(counters_.data()); }
  uint64_t* d_tail() const { return d_head() + 1; }
  uint64_t* d_snapshot() const { return d_head() + 2; }

  int64_t capacity_ = 0;
  int64_t slot_count_ = 0;
  cudaStream_t stream_ = nullptr;
  AllocT alloc_;
  Buffer slots_, counters_;
};

}  // namespace embedding_cache
