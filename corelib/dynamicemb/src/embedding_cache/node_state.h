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
// One node's state, as a value.
//
// Four facts packed into 32 bits: what the node's row is doing, how many
// in-flight steps refer to it, whether it is sitting in the reclaim ring, and
// whether it has been touched since it was last considered for eviction.
//
// There are no atomics in this file, and that is the point. Every operation is
// a pure function from one NodeState to another, so the concurrency layer above
// becomes load / decode / decide / compare-exchange — where "decide" is
// something a host test can enumerate with no GPU, no threads and no atomic.
// What is left for that layer to get right is memory ordering, which is hard
// enough on its own.
//
// ---------------------------------------------------------------------------
// Why one word
// ---------------------------------------------------------------------------
//
// Almost every transition here has a single writer. Keys are deduplicated
// before any of this runs, so one thread handles one key and therefore one
// node; the ring's in_ring bit admits at most one entry per node, so two
// threads never pop the same one; and pinning and claiming are separated by a
// kernel boundary, so they cannot interleave at all.
//
// One pair is left, and it decides the shape of this type: pinning runs on the
// prefetch stream and releasing runs on the backward pass, and nothing orders
// them against each other. The only cross-stream constraint in the design is
// that an allocate waits for already-issued releases — deliberately
// one-directional, because the overlap is what the pipeline is for. So for any
// key that appears in two in-flight steps, which is to say any key worth
// caching, refs+1 and refs-1 can land on the same word at the same moment.
//
// Non-atomically that loses one of the two updates. Lose the release and refs
// never returns to zero, so the node is never re-enqueued and the cache is one
// slot smaller for the rest of the run. Lose the pin and refs reads zero while
// a step is still using the row, so the node is claimed, rebound to another key
// and overwritten underneath its reader.
//
// Everything else follows. refs has to be an atomic read-modify-write. Claiming
// has to test refs and change residency without a release slipping in between,
// which means one compare-exchange over both — and that is what puts them in
// one word. Whether the fields could be split now that pin and claim are phase
// separated is a live question; they are together because claim and release
// still race, not because pin and claim do.
//
// ---------------------------------------------------------------------------
// The reference-count protocol, and what it does not support
// ---------------------------------------------------------------------------
//
//   prefetch (forward)   refs + 1, exactly once per unique key per step
//   backward             refs - 1, exactly once per unique key per step
//
// So refs is the number of in-flight steps holding the node. Zero does not mean
// the row is dead: an unreferenced RESIDENT node is still a cache hit and keeps
// serving until the ring hands it to somebody else.
//
// The protocol assumes forward and backward run the same number of times for a
// given step. Two ordinary PyTorch patterns break that assumption and are NOT
// supported:
//
//   * backward(retain_graph=True), with the embedding on more than one backward
//     path — one increment, several decrements
//   * torch.utils.checkpoint around a region containing the embedding lookup —
//     the forward is replayed during backward, so several increments and one
//     decrement
//
// The first underflows, the second leaks.
//
// ---------------------------------------------------------------------------
// What the assertions buy, and what they do not
// ---------------------------------------------------------------------------
//
// pinned() and released() assert that the resulting count is representable.
// They are EC_ASSERT, so a release build has no check at all — see ec_assert.h
// for why a device-side assert must not survive into a real run.
//
// They are therefore a development aid and not a guard. In a release build a
// release from zero is a subtraction on the whole word, and the borrow runs up
// out of the refs field into used, in_ring and residency, leaving the node in a
// state the machine above has no case for. Bit 29 is reserved and could be
// spent on containing that — held at one it would absorb the borrow and leave
// the fields above it intact — but nothing would read such a bit today, so it
// stays reserved until something does.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>

#include "ec_assert.h"

namespace embedding_cache {

class NodeState {
 public:
  // The storage. A node declares its field as NodeState::Raw rather than as a
  // NodeState, so the concurrency layer can bind a cuda::atomic_ref straight to
  // it; naming the type here is what says which component owns that field.
  using Raw = uint32_t;

  // What the node's row is doing. Two bits, three values in use; the fourth is
  // free, and a fifth would mean taking a bit from refs.
  enum Residency : Raw {
    // Never held data. Claimable, and nothing to write back when claimed.
    kEmpty = 0u,
    // Bound to a key, contents not yet valid — a transfer or an initializer is
    // in flight. Neither readable nor claimable.
    //
    // This is also what a node becomes the instant it is claimed. Claiming and
    // rebinding are one thread in one kernel, so nobody can observe the gap
    // between them, and a separate state for it would be one nothing reads: a
    // pin and a claim refuse a LOADING node exactly as they would refuse that.
    kLoading = 1u,
    // Contents valid. refs > 0 means in use by an in-flight step; refs == 0
    // means idle, and still a cache hit.
    kResident = 2u,
  };

  //   31 30 | 29       | 28      | 27   | 26 ................ 0
  //   resid | reserved | in_ring | used | refs
  static constexpr int kResidencyShift = 30;
  static constexpr Raw kResidencyMask = 3u << kResidencyShift;
  static constexpr Raw kReservedMask = 1u << 29;
  static constexpr Raw kInRingMask = 1u << 28;
  static constexpr Raw kUsedMask = 1u << 27;
  static constexpr Raw kRefsMask = (1u << 27) - 1u;

  // 27 bits, which is what was left rather than a budget anyone plans against:
  // a node's count is (unique keys referring to it) x (prefetch depth), single
  // digits in any configuration that fits in memory.
  static constexpr uint32_t kMaxRefs = kRefsMask;

  // Trivially default constructible and trivially copyable, both required: a
  // node lives in an array staged through std::vector on the host and moved
  // with cudaMemcpy. Default construction leaves the word unset, which is
  // right — what a fresh node holds is the caller's to say, see make().
  NodeState() = default;

  // [[nodiscard]] on everything below, and it earns its keep here more than
  // usual. Nothing in this class mutates, so `s.pinned();` as a statement is a
  // no-op that reads like an action — the characteristic mistake of a value
  // type whose operations are named after verbs. The attribute turns it into a
  // compile error.
  [[nodiscard]] static constexpr __forceinline__ __host__ __device__ NodeState
  from_raw(Raw w) {
    return NodeState(w);
  }

  // The general constructor. There is deliberately no initial() or reset():
  // what a never-used node should look like depends on whether it is already
  // sitting in the reclaim ring, and this component knows nothing about the
  // ring. Whoever initialises the node pool states it outright, which is also
  // where the reason for it belongs.
  [[nodiscard]] static constexpr __forceinline__ __host__ __device__ NodeState
  make(Residency r, uint32_t refs, bool in_ring, bool used) {
    EC_ASSERT(refs <= kMaxRefs);
    return NodeState((static_cast<Raw>(r) << kResidencyShift) |
                     (in_ring ? kInRingMask : 0u) | (used ? kUsedMask : 0u) |
                     (refs & kRefsMask));
  }

  [[nodiscard]] constexpr __forceinline__ __host__ __device__ Raw raw() const {
    return w_;
  }

  // -- queries --------------------------------------------------------------

  [[nodiscard]] constexpr __forceinline__ __host__ __device__ Residency
  residency() const {
    return static_cast<Residency>((w_ & kResidencyMask) >> kResidencyShift);
  }
  [[nodiscard]] constexpr __forceinline__ __host__ __device__ uint32_t refs()
      const {
    return w_ & kRefsMask;
  }
  [[nodiscard]] constexpr __forceinline__ __host__ __device__ bool in_ring()
      const {
    return (w_ & kInRingMask) != 0u;
  }
  [[nodiscard]] constexpr __forceinline__ __host__ __device__ bool used()
      const {
    return (w_ & kUsedMask) != 0u;
  }
  [[nodiscard]] constexpr __forceinline__ __host__ __device__ bool has_refs()
      const {
    return refs() != 0u;
  }

  // -- transitions ----------------------------------------------------------
  //
  // Each returns the state that would result and changes nothing. That is what
  // lets the layer above read a word once, compute the word it wants, and hand
  // both to a single compare-exchange.

  // The state after a pin: one more reference, and `used` set.
  //
  // `used` rides along in the same word deliberately. It is what a
  // second-chance replacement policy reads, and folding it in here means that
  // recording recency costs no extra memory traffic on the hottest path in the
  // cache — the one that runs for every key that hits.
  //
  // refs is the low field, so the increment cannot disturb anything above it as
  // long as it does not overflow, which is what the assertion covers.
  [[nodiscard]] constexpr __forceinline__ __host__ __device__ NodeState pinned()
      const {
    EC_ASSERT(refs() < kMaxRefs);
    return NodeState((w_ | kUsedMask) + 1u);
  }

  // The state after a release: one fewer reference.
  [[nodiscard]] constexpr __forceinline__ __host__ __device__ NodeState
  released() const {
    EC_ASSERT(has_refs());
    return NodeState(w_ - 1u);
  }

  [[nodiscard]] constexpr __forceinline__ __host__ __device__ NodeState
  with_refs(uint32_t n) const {
    EC_ASSERT(n <= kMaxRefs);
    return NodeState((w_ & ~kRefsMask) | (n & kRefsMask));
  }
  [[nodiscard]] constexpr __forceinline__ __host__ __device__ NodeState
  with_residency(Residency r) const {
    return NodeState((w_ & ~kResidencyMask) |
                     (static_cast<Raw>(r) << kResidencyShift));
  }
  [[nodiscard]] constexpr __forceinline__ __host__ __device__ NodeState
  with_in_ring(bool v) const {
    return NodeState(v ? (w_ | kInRingMask) : (w_ & ~kInRingMask));
  }
  [[nodiscard]] constexpr __forceinline__ __host__ __device__ NodeState
  with_used(bool v) const {
    return NodeState(v ? (w_ | kUsedMask) : (w_ & ~kUsedMask));
  }

  [[nodiscard]] constexpr __forceinline__ __host__ __device__ bool operator==(
      NodeState o) const {
    return w_ == o.w_;
  }
  [[nodiscard]] constexpr __forceinline__ __host__ __device__ bool operator!=(
      NodeState o) const {
    return w_ != o.w_;
  }

 private:
  explicit constexpr __forceinline__ __host__ __device__ NodeState(Raw w)
      : w_(w) {}

  Raw w_;
};

static_assert(sizeof(NodeState) == sizeof(NodeState::Raw),
              "NodeState must be exactly its word: it is what a node stores");
static_assert((NodeState::kResidencyMask | NodeState::kReservedMask |
               NodeState::kInRingMask | NodeState::kUsedMask |
               NodeState::kRefsMask) == 0xFFFFFFFFu,
              "the fields must tile the word with no gap and no overlap");

// ---------------------------------------------------------------------------
// AtomicNodeState — the word as it lives in a node
// ---------------------------------------------------------------------------
//
// NodeState is a value; this is the storage. A node declares one of these, and
// every modification goes through it.
//
// The split is what the two halves can be checked with. NodeState is pure, so a
// host test enumerates it: four residencies against both bits against the ends
// of the refs range, and every field-setter proved not to disturb the other
// three. What is left here is memory ordering and behaviour under contention,
// which needs a GPU and cannot be reached any other way.
//
// The methods that modify are instance methods, deliberately: they act on this
// object, at this address, and there is nothing to discard or forget to store
// back. The pure transitions stay on NodeState, where operating on a copy is
// the point rather than a mistake.
//
// One sharp edge remains. This type has to stay trivially copyable — a node
// travels through std::vector and cudaMemcpy, and HashMapStorage asserts on it
// — so copy construction cannot be deleted, and in device code
//
//     AtomicNodeState copy = node.state;   // then copy.try_pin()
//
// would pin a register. Reaching that takes a deliberate copy: load() hands
// back a NodeState, not one of these, so ordinary code never holds a loose one.
//
// ---------------------------------------------------------------------------
// Orders
// ---------------------------------------------------------------------------
//
//   try_pin    acquire on success — the caller is about to read the row, and
//              must see what publish() released
//   release    release on success — the backward pass has just written the row
//              and whoever picks the node up next has to see it
//   try_claim  acquire on success — the caller is about to read the victim's
//              row in order to write it back
//   publish    release — the row has landed, and must be visible before the
//              residency that advertises it
//
// Written for what each operation means rather than for what today's kernel
// boundaries happen to make unnecessary. Ordering that is correct only because
// of a boundary somewhere else is ordering nobody will remember to revisit when
// the boundary moves.

class AtomicNodeState {
 public:
  using Raw = NodeState::Raw;

  // Trivially default constructible and trivially copyable, both required by
  // the node array. Default construction leaves the word unset; see set().
  AtomicNodeState() = default;

  // Plain, non-atomic write, for initialising a node pool before anything can
  // reach it. Not for use once threads are running.
  __forceinline__ __host__ __device__ void set(NodeState s) { w_ = s.raw(); }

  // Plain, non-atomic read, for a host that has the array to itself —
  // diagnostics, checkpointing, tests.
  [[nodiscard]] __forceinline__ __host__ __device__ NodeState get() const {
    return NodeState::from_raw(w_);
  }

  [[nodiscard]] __forceinline__ __device__ NodeState
  load(cuda::std::memory_order order = cuda::std::memory_order_relaxed) const {
    return NodeState::from_raw(ref().load(order));
  }

  // Take a reference on a cache hit: refs+1, and `used` set.
  //
  // Fails, rather than waiting, if the node is not RESIDENT — an allocator got
  // there first, or the row is still loading. Spinning here would mean spinning
  // on an operation that may contain a PCIe transfer.
  //
  // What a failed pin means for the caller is not "miss, and one extra
  // transfer". The node is bound to the key, so treating the key as a miss makes
  // the allocator bind it to a second node. That cannot happen while pinning and
  // rebinding stay in separate kernels, and it is what that separation is for.
  //
  // For the same reason a successful pin needs no check that the node still
  // holds the key that was looked up: the word cannot tell a node that was
  // evicted and rebound from one that was not, because it can return to a
  // bit-identical value, so the only protection is that no rebind runs between
  // the lookup and this call.
  [[nodiscard]] __forceinline__ __device__ bool try_pin() {
    auto a = ref();
    Raw cur = a.load(cuda::std::memory_order_relaxed);
    for (;;) {
      const NodeState s = NodeState::from_raw(cur);
      if (s.residency() != NodeState::kResident) return false;
      if (a.compare_exchange_weak(cur, s.pinned().raw(),
                                  cuda::std::memory_order_acquire,
                                  cuda::std::memory_order_relaxed)) {
        return true;
      }
    }
  }

  // Give up a reference: refs-1.
  //
  // Returns whether this thread now owns enqueueing the node — true when it
  // took the count to zero and the node was not already in the ring. Deciding
  // that and clearing the way for it happen in the same compare-exchange as the
  // decrement, so no other thread can act on the node in between; a decrement
  // followed by a separate test-and-set of in_ring would leave a window in
  // which a claim on another stream could take the node and this thread would
  // then advertise it as reclaimable anyway.
  //
  // The bit is set here and the queue is somebody else's: this type owns
  // in_ring's atomicity, the ring owns what the answer means.
  [[nodiscard]] __forceinline__ __device__ bool release() {
    auto a = ref();
    Raw cur = a.load(cuda::std::memory_order_relaxed);
    for (;;) {
      const NodeState s = NodeState::from_raw(cur);
      const bool owns_enqueue = (s.refs() == 1u && !s.in_ring());
      NodeState next = s.released();
      if (owns_enqueue) next = next.with_in_ring(true);
      if (a.compare_exchange_weak(cur, next.raw(),
                                  cuda::std::memory_order_release,
                                  cuda::std::memory_order_relaxed)) {
        return owns_enqueue;
      }
    }
  }

  // Take the node for reuse: EMPTY or RESIDENT with no references becomes
  // LOADING, with one reference held by this thread.
  //
  // `*was` receives the state at the instant of the claim, which is what tells
  // the caller whether there is anything to write back — an EMPTY node never
  // held data, a RESIDENT one is by construction newer than the backing copy.
  //
  // Fails on a node that has references (it was hit again after being enqueued,
  // so the entry that led here is stale), and on one that is LOADING. `used` is
  // cleared on success: the node is about to hold something else, and carries
  // no recency into that.
  //
  // CALL THIS ONLY ON A NODE JUST TAKEN OFF THE RECLAIM QUEUE, because it
  // clears in_queue WHETHER OR NOT THE CLAIM SUCCEEDS. The entry that led here
  // has been consumed by the pop either way, and the flag has to say so — a
  // node still flagged as queued is one that release() will decline to enqueue,
  // on the grounds that an entry already exists. If the flag outlives the entry
  // the node is never queued again and the cache quietly loses a slot for the
  // rest of the run.
  //
  // That is not hypothetical. It is what the first device run of this file
  // found: with capacity tight enough for stale entries to be common, every
  // node ended up flagged-but-absent and the queue drained to nothing, while
  // the comment in reclaim_queue.h claimed this method already did this.
  [[nodiscard]] __forceinline__ __device__ bool try_claim(NodeState* was) {
    auto a = ref();
    Raw cur = a.load(cuda::std::memory_order_relaxed);
    for (;;) {
      const NodeState s = NodeState::from_raw(cur);
      const NodeState::Residency r = s.residency();
      const bool claimable =
          !s.has_refs() && (r == NodeState::kEmpty || r == NodeState::kResident);

      if (!claimable) {
        // Refused. Consume the queue entry and leave everything else alone.
        if (!s.in_ring()) return false;  // already clear; nothing to write
        if (a.compare_exchange_weak(cur, s.with_in_ring(false).raw(),
                                    cuda::std::memory_order_relaxed,
                                    cuda::std::memory_order_relaxed)) {
          return false;
        }
        continue;
      }

      const NodeState next = s.with_residency(NodeState::kLoading)
                                 .with_in_ring(false)
                                 .with_used(false)
                                 .with_refs(1u);
      if (a.compare_exchange_weak(cur, next.raw(),
                                  cuda::std::memory_order_acquire,
                                  cuda::std::memory_order_relaxed)) {
        *was = s;
        return true;
      }
    }
  }

  // Advertise a loaded row: LOADING becomes RESIDENT.
  //
  // A store rather than a compare-exchange, because the node has one owner
  // while it is LOADING — a pin needs RESIDENT and a claim needs no references,
  // and the claiming thread holds one. Nothing else can be writing this word.
  __forceinline__ __device__ void publish() {
    auto a = ref();
    const NodeState s = NodeState::from_raw(
        a.load(cuda::std::memory_order_relaxed));
    EC_ASSERT(s.residency() == NodeState::kLoading);
    a.store(s.with_residency(NodeState::kResident).raw(),
            cuda::std::memory_order_release);
  }

  // Drop an idle node's binding: RESIDENT with no references becomes EMPTY.
  //
  // For a key the backing has evicted. The key has left the system, so its row has
  // nothing to be written back, and EMPTY is exactly "claimable, nothing to write
  // back" — an allocate that pops this node later finds it as any other empty one.
  //
  // The caller guarantees RESIDENT and refs == 0, and this asserts it rather than
  // checking. The guarantee is the eviction guard's: the backing does not take a
  // key whose node has references, so a referenced node never reaches here.
  //
  // in_ring is left alone. An idle node always has its ring entry — a count
  // reaching zero is what enqueues it — and that entry is what lets an allocate
  // find the node now. used is cleared: the node carries no recency into whatever
  // it holds next.
  //
  // A store rather than a compare-exchange, for the reason publish() gives: with
  // no references nothing can write this word in this phase. A pin needs a lookup,
  // a claim needs an allocate, and a release needs a reference to drop.
  __forceinline__ __device__ void invalidate() {
    auto a = ref();
    const NodeState s = NodeState::from_raw(
        a.load(cuda::std::memory_order_relaxed));
    EC_ASSERT(s.residency() == NodeState::kResident && !s.has_refs());
    a.store(s.with_residency(NodeState::kEmpty).with_used(false).raw(),
            cuda::std::memory_order_release);
  }

 private:
  using Ref = cuda::atomic_ref<Raw, cuda::thread_scope_device>;

  // Free to build: an atomic_ref is a typed pointer and nothing more.
  __forceinline__ __device__ Ref ref() const {
    return Ref(const_cast<Raw&>(w_));
  }

  Raw w_;
};

static_assert(sizeof(AtomicNodeState) == sizeof(NodeState::Raw),
              "AtomicNodeState is the word and nothing else");

}  // namespace embedding_cache
