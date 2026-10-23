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
// key <-> node, and nothing else. This is the view: the structure a kernel reads
// and writes. hashmap.h adds the storage that owns it.
//
// It is a header of its own because the backing's eviction kernels, which are
// built separately and know nothing of torch or of this directory's launch code,
// have to read it too — see guard.h.
//
// This component knows that a node has a key and two chain pointers. It does
// not know what else a node has, where nodes are allocated, or that any of the
// rest of the cache exists. Everything else is a template parameter, so it
// compiles and tests on its own.
//
// ---------------------------------------------------------------------------
// The interface
// ---------------------------------------------------------------------------
//
//   lookup(key)        -> node id, or kNoNode
//   insert(node, key)      bind a node to a key
//   remove(node)           unbind it — no key, the node already holds it
//   clear_links(Node&)     put one node's links in the not-linked state
//
// That is the whole of it. There is no key_of(node) or holds(node, key): the
// caller holds the node array and node.key is a public field, so those would be
// this component's name wrapped around the caller's own member access, and one
// more entry in a contract that is easier to keep short.
//
// ---------------------------------------------------------------------------
// Why the node type is a parameter
// ---------------------------------------------------------------------------
//
// A chain walk reads a node's key (to compare) and its next pointer (to
// advance) — the same node, both times. Split across two arrays that is two
// cache lines per hop; in one struct it is one. The caller then gets to put
// whatever else it wants alongside, and a lookup followed by an acquire touches
// one line rather than three.
//
// The caller writes the node type and derives it from HashMapNode, which is
// where key, prev and next live:
//
//   struct Node : HashMapNode<int64_t, int32_t> {
//     uint32_t state;      // belongs to the concurrency protocol
//     int32_t  ring_hint;  // belongs to the ring
//   };
//
//   HashMapStorage<Node> storage(capacity);
//
// Derivation, not a member, for two reasons. The map's fields read as
// `node.key`, not `node.chain.key`, at every call site including the caller's
// own. And the base carries both types, so HashMap<Node> deduces Key and NodeId
// rather than being told them a second time — a second statement of a type is a
// second chance to state it differently.
//
// The cost is that this component no longer owns the key field, so a caller
// *can* write node.key directly. That corrupts the map silently — the node
// stays in the bucket its old key hashed to, and every later lookup for either
// key walks the wrong chain. insert() is the only correct way to change a
// binding. Since it can no longer be enforced, check_bindings() below exists to
// detect it.
//
// ---------------------------------------------------------------------------
// What a caller must guarantee
// ---------------------------------------------------------------------------
//
//   1. lookup must not run concurrently with anything at all
//   2. node.key is written only by insert()
//
// Rule 1 is a property of *this* implementation, not a universal truth. A
// different one — open addressing, say — may relax it.
//
// Everything else may overlap freely: many threads may insert, remove or both
// at once, which is what an allocating kernel does when one thread rebinds a
// victim out of one bucket while another binds a new key into it. insert and
// remove hold the same bucket lock across the whole mutation, and every node a
// mutation touches — the victim's neighbours, the incumbent head — is in the
// bucket being locked, because a chain never spans buckets. That is what turns
// the general lock-free doubly-linked list problem into a short critical
// section, and why the two do not have to be put in separate phases.
//
// Rule 1 is the load-bearing one, and the reason is worth stating: a lookup
// that races a mutation does not merely return a stale answer, it returns
// kNoNode for a key that is present. remove() clears the victim's next pointer,
// so a walk standing on that node stops there and never sees the rest of the
// chain; insert() prepends, so a walk that already passed the head never sees
// the new node. Either way the caller reads a miss, allocates a node, and
// inserts the same key a second time. Two nodes then hold one key, each with
// its own row, and they diverge silently — whichever is written back last wins.
// A missed lookup is not a lost lookup; it is a duplicated key.
//
// Rule 1 says "anything at all", not "insert or remove", and the extra breadth
// is deliberate. It is what lets lookup() read bucket heads as plain NodeId
// while lock() reads them as atomics: two access kinds on one object, legal
// only because they never overlap. Weaken the rule to cover just this
// component's mutators and the read path has to go back to atomics, which costs
// an L1 bypass per hop on the hottest path there is.
//
// ---------------------------------------------------------------------------
// Requires sm_70+
// ---------------------------------------------------------------------------
//
// Two threads touching one bucket serialise on a short lock held in the bucket
// head itself. Two reasons the floor is Volta:
//
//   * the lock spins, so it relies on independent thread scheduling for a warp
//     where one lane holds the lock and the others wait; on older
//     architectures the waiters would starve the holder
//   * the acquire/release ordering below is expressed with scoped atomics,
//     which compile to real acquire/release forms only from sm_70 on
//
// Both are already the repository's floor (CMakeLists sets 70 as the lowest
// architecture) and libcu++ is already a dependency — see
// table_operation/types.cuh, which locks its key slots the same way.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "ec_assert.h"

namespace embedding_cache {


// This header must be compiled by nvcc. It launches kernels, and there is no
// fallback that does the same work on the host: the code this belongs to runs
// on a GPU, and a second implementation to keep a GPU-free build alive would be
// a second thing to keep correct.
#ifndef __CUDACC__
#error \
    "embedding_cache/hashmap.h launches kernels and must be compiled by nvcc. \
Including it from a .cpp will not work; make the translation unit a .cu."
#endif

// ---------------------------------------------------------------------------
// Default hash
// ---------------------------------------------------------------------------

// The MurmurHash3 fmix64 finalizer, which is what the rest of this repository
// already hashes keys with. Recommender ids are often small, or dense within a
// per-table range, so the low bits alone would pile into a handful of buckets;
// this is here to avalanche them before masking.
//
// NOTE: the same six lines, with the same constants, currently appear in at
// least four other places — table_operation/types.cuh (as KeyTraits::hash,
// additionally masked to INT64_MAX), sparse_block_bucketize_features.cu (as
// hash_key), scored_hashtable.py (as murmur3_hash_64bits), and one unit test.
// Nobody has put it in a header, so everyone redeclares it. Worth
// consolidating; this copy exists so that hashmap.h stays free of dependencies
// on the rest of the tree, which is the property that lets it be tested alone.
struct Murmur3Hash64 {
  template <typename KeyT>
  __forceinline__ __host__ __device__ uint64_t operator()(KeyT key) const {
    uint64_t k = static_cast<uint64_t>(key);
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return k;
  }
};

// ---------------------------------------------------------------------------
// The fields this component owns
// ---------------------------------------------------------------------------

// Derive the node type from this. Everything else in a node belongs to somebody
// else; these three belong to the map, and grouping them says so.
//
// Deliberately an aggregate — no constructors, no default member initialisers.
// It has to stay trivially copyable, because the node array is moved with
// cudaMemcpy, and trivially default constructible, because HashMapStorage
// stages that array through a std::vector on the host.
//
// Note for anyone adding fields to a derived node: a base class with trailing
// padding does not lend it to the derived class, so a base whose size is not a
// multiple of its alignment wastes those bytes. HashMapNode<int64_t, int32_t> is
// 16 bytes with none, which is the combination this cache uses.
template <typename KeyT, typename NodeIdT>
struct HashMapNode {
  KeyT key;
  NodeIdT prev;
  NodeIdT next;
};

namespace detail {

// Recovers the HashMapNode a node type derives from, and with it the key and
// node-id types. Never defined: it exists only to be deduced against.
template <typename KeyT, typename NodeIdT>
HashMapNode<KeyT, NodeIdT> hash_map_node_base(const HashMapNode<KeyT, NodeIdT>&);

// The placeholders on the failure branch are there so that a node type which
// does not derive from HashMapNode still leaves the rest of HashMap well-formed
// — the static_assert is then what the caller sees, rather than a cascade from
// numeric_limits<void>.
template <typename T, typename Base = void>
struct node_traits_impl {
  static constexpr bool derived = false;
  using key_type = int;
  using id_type = int32_t;
};

template <typename T, typename KeyT, typename NodeIdT>
struct node_traits_impl<T, HashMapNode<KeyT, NodeIdT>> {
  static constexpr bool derived = true;
  using key_type = KeyT;
  using id_type = NodeIdT;
};

// Deduction fails, and `derived` stays false, if the node derives from no
// HashMapNode, from more than one, or derives privately.
template <typename T, typename = void>
struct node_traits : node_traits_impl<T> {};

template <typename T>
struct node_traits<
    T, std::void_t<decltype(hash_map_node_base(std::declval<const T&>()))>>
    : node_traits_impl<T,
                       decltype(hash_map_node_base(std::declval<const T&>()))> {
};

template <typename T>
using node_id_t = typename node_traits<T>::id_type;

}  // namespace detail

// ---------------------------------------------------------------------------
// HashMap
// ---------------------------------------------------------------------------
//
// Trivially copyable: passed to kernels by value, holds pointers and scalars,
// never ownership.

template <typename NodeT, typename HashFn = Murmur3Hash64,
          detail::node_id_t<NodeT> kNoNodeV =
              static_cast<detail::node_id_t<NodeT>>(-1)>
struct HashMap {
  using Node = NodeT;
  using NodeId = typename detail::node_traits<NodeT>::id_type;
  using Key = typename detail::node_traits<NodeT>::key_type;
  using Hash = HashFn;

  static_assert(detail::node_traits<NodeT>::derived,
                "the node type must derive publicly from HashMapNode<Key, "
                "NodeId>, from exactly one of them; that base is what supplies "
                "key/prev/next and what Key and NodeId are deduced from");
  static_assert(std::is_trivially_copyable<NodeT>::value,
                "the node type must be trivially copyable: the node array is "
                "moved with cudaMemcpy and staged through std::vector");
  static_assert(std::is_integral<NodeId>::value,
                "NodeId must be integral: it indexes the node pool and is the "
                "subject of the bucket-head atomics");
  static_assert(sizeof(NodeId) == 4,
                "NodeId must be 32-bit: the bucket lock is an exchange on a "
                "bucket head, and 32 bits covers any node pool that fits in "
                "memory");

  // The hash is not stored. It is a template parameter and every hash in this
  // repository is stateless — MurmurHash3_32 takes its seed as a template
  // parameter too — so an instance would carry no information while costing
  // eight bytes of a struct that reaches every kernel by value.
  static_assert(std::is_empty<HashFn>::value,
                "HashFn must be stateless: HashMap does not store it. A seeded "
                "hash should take its seed as a template parameter, as "
                "MurmurHash3_32 does");
  static_assert(std::is_trivially_default_constructible<HashFn>::value,
                "HashFn must be trivially default-constructible: it is "
                "instantiated at each call site rather than held");

  static constexpr NodeId kNoNode = kNoNodeV;

  // The lock marker must be a value no bucket head can legitimately hold. Node
  // ids live in [0, capacity) and kNoNode is outside that by construction, so
  // one step further out is outside it too.
  static constexpr NodeId kLocked = static_cast<NodeId>(kNoNodeV - 1);
  static_assert(kLocked != kNoNodeV, "the two sentinels must differ");

  // -- state, all borrowed --------------------------------------------------

  NodeId* heads;  // [num_buckets], kNoNode when empty
  Node* nodes;    // [capacity], owned by whoever owns the node pool

  // The count, not a mask alongside it. Two fields that must agree can
  // disagree, and a mask wider than the array is an out-of-bounds write; with
  // one field the other is derived and cannot drift.
  //
  // Must be 2^k, which HashMapStorage enforces. Masking rather than a modulo is
  // deliberate: one integer division here would cost more than everything else
  // in bucket_of put together. If it is not 2^k the index still lands inside
  // the array — `hash & (n-1) <= n-1` for any n — but the holes in the mask
  // leave most buckets unreachable, which costs hit rate rather than
  // correctness.
  int64_t num_buckets;

  __forceinline__ __host__ __device__ int64_t bucket_of(Key key) const {
    return static_cast<int64_t>(HashFn{}(key) &
                                static_cast<uint64_t>(num_buckets - 1));
  }

  // -- the interface --------------------------------------------------------

  // Resolve a key. kNoNode if no node is bound to it.
  //
  // Each hop reads the node's key and its next pointer, which is why they share
  // a struct: one line per hop instead of two.
  //
  // Concurrent with nothing. Not with insert, not with remove, not with another
  // caller mutating the map behind this component's back — a lookup phase is a
  // phase in which the map does not change. Everything else here follows from
  // that: no lock, no atomic, no volatile, every read a plain load. This is the
  // hot path — one call per key per step, and the map is consulted before
  // anything else happens — so the difference is not academic. A volatile or
  // relaxed-atomic read bypasses L1, and this walk would pay that on the head
  // and again on every hop, for an ordering guarantee that has nothing to
  // order.
  __forceinline__ __device__ NodeId lookup(Key key) const {
    NodeId n = heads[bucket_of(key)];
    while (n != kNoNode) {
      const Node& node = nodes[n];
      if (node.key == key) return n;
      n = node.next;
    }
    return kNoNode;
  }

  // Bind `node` to `key` and place it in the right bucket.
  //
  // The key is a parameter because this is the operation that *sets* it. The
  // two happen together deliberately: writing the key without moving the node
  // is the one way to corrupt this structure, so this is the only operation
  // that should ever write node.key.
  //
  // The node must be unbound — freshly initialised, or just removed. Inserting
  // one that is still linked would splice a second bucket onto its chain.
  __forceinline__ __device__ void insert(NodeId node, Key key) {
    Node& n = nodes[node];
    EC_ASSERT(n.prev == kNoNode && n.next == kNoNode &&
              "HashMap::insert: the node is still linked into a bucket; "
              "remove() it before rebinding");
    n.key = key;
    const int64_t b = bucket_of(key);
    const NodeId head = lock(b);

    n.prev = kNoNode;
    n.next = head;
    if (head != kNoNode) nodes[head].prev = node;

    unlock(b, node);
  }

  // Unbind `node`.
  //
  // No key parameter, unlike insert. The key is not an input here — the node
  // already holds it, and it is the only correct value. Passing it in would put
  // the same fact in two places, and the copy in the caller's hand is the one
  // that can be wrong: the key picks which bucket to lock and decides whether
  // this node is the head, so a stale one both drops mutual exclusion against
  // the node's real bucket and leaves that bucket's head pointing at a node
  // that has just been unlinked.
  //
  // Reading it here costs nothing worth counting. The load is before the lock,
  // not inside it, and it lands on a line the caller has already touched —
  // node.key sits in the same struct as the state word that decided this node
  // could be evicted at all.
  //
  // The key field is left set. A removed node sits in no bucket, so nothing can
  // reach it to read the key, and a caller that wants the outgoing key reads it
  // before removing — the natural order anyway.
  __forceinline__ __device__ void remove(NodeId node) {
    Node& n = nodes[node];
    const int64_t b = bucket_of(n.key);
    const NodeId head = lock(b);

    const NodeId p = n.prev;
    const NodeId nx = n.next;

    // A node with no neighbours is either the only occupant of its bucket or in
    // no bucket at all, and the head is what tells the two apart. This catches
    // a second remove() of the same node, which would otherwise quietly write
    // some other bucket's head back over itself and look like it worked.
    EC_ASSERT((p != kNoNode || nx != kNoNode || head == node) &&
              "HashMap::remove: the node is not linked into the bucket its key "
              "hashes to — removed twice, or never inserted");

    if (p != kNoNode) nodes[p].next = nx;
    if (nx != kNoNode) nodes[nx].prev = p;
    n.prev = kNoNode;
    n.next = kNoNode;

    // Removing the head promotes its successor; otherwise the head stands.
    unlock(b, (head == node) ? nx : head);
  }

  // Put one node's links into the not-linked state.
  //
  // The links, and not the key. There is no "unbound" value for a key here and
  // there should not be one: reserving a key value costs the caller a key it can
  // never use, and it would encode a fact this component does not own. Whether a
  // node holds anything is the caller's state word's business. Until insert()
  // runs, node.key is simply meaningless — and unreachable, since a node with no
  // links is in no bucket.
  __forceinline__ __host__ __device__ static void clear_links(Node& n) {
    n.prev = kNoNode;
    n.next = kNoNode;
  }

 private:
  // Atomic access to a bucket head.
  //
  // atomic_ref, not an array declared as cuda::atomic<NodeId>: the storage stays
  // a plain NodeId array, so the host side — clear_buckets, download_heads,
  // check_bindings — needs no special handling, and unlike
  // reinterpret_cast<cuda::atomic*> (which is what table_operation/types.cuh
  // does) it is not an aliasing violation. It is the tool for exactly this
  // case: atomic access to storage that is not itself an atomic type.
  //
  // Device scope, not system: everything here lives in device memory and is
  // touched only by device threads.
  //
  // lookup() reads these same words as plain NodeId, which would be a data race
  // if the two could overlap. They cannot — that is exactly what rule 1 of the
  // caller contract buys, and it is the reason the read path costs nothing.
  using AtomicHead = cuda::atomic_ref<NodeId, cuda::thread_scope_device>;

  // Returns the head as it was before locking.
  //
  // The exchange carries `acquire`, and it has to. A relaxed exchange would
  // make the head itself coherent and say nothing about any other address, so
  // the plain loads of n.prev / n.next that follow could be served from an L1
  // line this SM cached before the previous holder's writes existed — L1 is not
  // coherent across SMs. Paired with the `release` in unlock(), those writes are
  // ordered before the head becomes visible and this acquire makes them visible
  // here. Neither half works alone.
  //
  // acquire/release rather than __threadfence(): a fence is seq_cst at device
  // scope, which is strictly more than this needs and prices accordingly.
  __forceinline__ __device__ NodeId lock(int64_t b) const {
    AtomicHead head(heads[b]);
    for (;;) {
      const NodeId prev =
          head.exchange(kLocked, cuda::std::memory_order_acquire);
      if (prev != kLocked) return prev;
      // Test-and-test-and-set. A failed exchange still writes, so retrying it
      // directly would have every waiter issuing stores against the one line
      // the holder has to write to make progress. Spin on a relaxed load
      // instead and only attempt the exchange once the lock looks free.
      while (head.load(cuda::std::memory_order_relaxed) == kLocked) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
      }
    }
  }

  // A plain store, not an exchange: this thread holds the lock, so nobody else
  // can be writing the head, and there is nothing to read back. `release`
  // publishes the chain edits above before the head that leads to them.
  __forceinline__ __device__ void unlock(int64_t b, NodeId new_head) const {
    AtomicHead(heads[b]).store(new_head, cuda::std::memory_order_release);
  }
};


}  // namespace embedding_cache
