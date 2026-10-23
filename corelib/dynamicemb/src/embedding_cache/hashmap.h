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
// HashMapStorage: the buckets and the node array behind a HashMap view.
//
// The view itself, with the contract a caller has to keep, is in hashmap_view.h.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "allocator.h"
#include "ec_assert.h"
#include "hashmap_view.h"
#include "launch.h"

namespace embedding_cache {

// ---------------------------------------------------------------------------
// Initialisation kernels
// ---------------------------------------------------------------------------
//
// Templates rather than plain __global__ functions, so that they can live in a
// header without nvcc emitting a duplicate symbol in every translation unit
// that includes it. (`inline` on a non-template __global__ is ignored, with a
// warning that is easy to miss.)
//
// Grid-stride, so the launch geometry never has to be exact.

namespace detail {

template <typename T>
__global__ void fill_kernel(T* dst, int64_t n, T value) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    dst[i] = value;
  }
}

// Writes prev and next in place. Note what it does *not* do: read the node
// first. The host-staged version this replaced had to, because it round-tripped
// whole nodes through a std::vector and had to put back the fields it does not
// own — 24 bytes each way, for a two-word write.
template <typename MapT, typename NodeT>
__global__ void clear_links_kernel(NodeT* nodes, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    MapT::clear_links(nodes[i]);
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// HashMapStorage
// ---------------------------------------------------------------------------
//
// Owns the bucket heads and the node array.
//
// It allocates the nodes because it is the component that knows the node type,
// and every other component reaches them through nodes(). But note what it
// cannot do with them: a node carries fields belonging elsewhere — a state
// word, whatever else the cache needs — and this component has no idea what
// those should hold. So allocation and initialisation come apart:
//
//   clear_links()  sets prev and next, which is all this component owns at
//                  construction — node.key means nothing until insert() runs
//   <caller>       sets the rest, through nodes()
//
// There is no reset_all(), and there should not be one: it could only ever be
// half a reset.
//
// Every allocation goes through `AllocT`; see allocator.h for why that is worth
// a template parameter rather than a call to cudaMalloc.

template <typename NodeT, typename HashFn = Murmur3Hash64,
          detail::node_id_t<NodeT> kNoNodeV =
              static_cast<detail::node_id_t<NodeT>>(-1),
          typename AllocT = CudaAllocator>
class HashMapStorage {
 public:
  using Map = HashMap<NodeT, HashFn, kNoNodeV>;
  using Node = NodeT;
  using NodeId = typename Map::NodeId;
  using Key = typename Map::Key;
  using Allocator = AllocT;
  using Buffer = typename AllocT::Buffer;

  static constexpr NodeId kNoNode = Map::kNoNode;

  // `num_buckets` rounds up to a power of two, and defaults to twice the node
  // count, which keeps chains near length one for little: a bucket head is four
  // bytes against a node's row, which is orders of magnitude larger.
  //
  // `stream` is kept, not passed per call. Every operation this class performs
  // is initialisation — it runs once, at construction, or on an explicit reset —
  // so there is one stream in the picture and storing it means the constructor
  // and a later clear cannot end up on different ones. A caller that wants
  // initialisation somewhere else should call Map::clear_links from its own
  // kernel and skip these entirely.
  //
  // The constructor is now asynchronous with respect to that stream: it enqueues
  // two kernels and returns. The map is usable in stream order, not on return.
  // The download_* diagnostics below are ordered on the same stream and
  // synchronise, so they stay safe.
  explicit HashMapStorage(int64_t capacity, int64_t num_buckets = 0,
                          AllocT allocator = AllocT{},
                          cudaStream_t stream = nullptr)
      : capacity_(capacity), stream_(stream), alloc_(std::move(allocator)) {
    if (capacity <= 0) {
      throw std::invalid_argument("HashMapStorage: capacity must be > 0");
    }
    if (capacity > static_cast<int64_t>(INT32_MAX)) {
      throw std::invalid_argument(
          "HashMapStorage: capacity exceeds the 32-bit node id range");
    }
    num_buckets_ = round_up_pow2(num_buckets > 0 ? num_buckets : 2 * capacity);
    // The map masks rather than divides, so this has to be 2^k. round_up_pow2
    // makes it so; the check is here because it is the one place a change could
    // quietly stop making it so.
    if ((num_buckets_ & (num_buckets_ - 1)) != 0) {
      throw std::invalid_argument(
          "HashMapStorage: bucket count must be a power of two");
    }

    heads_ = alloc_.allocate(sizeof(NodeId) * size_t(num_buckets_));
    nodes_ = alloc_.allocate(sizeof(Node) * size_t(capacity_));
    clear_buckets();
    clear_links();
  }

  HashMapStorage(const HashMapStorage&) = delete;
  HashMapStorage& operator=(const HashMapStorage&) = delete;

  Map map() const { return Map{d_heads(), d_nodes(), num_buckets_}; }

  const Allocator& allocator() const { return alloc_; }

  // Empties every bucket. Enqueued on the storage's stream.
  //
  // Half of a reset, and only useful as half: the nodes keep the links they had,
  // so what was reachable before is now merely unreachable. Pair it with
  // clear_links(), which is what the constructor does. insert() asserts on a
  // node that is still linked, so the unpaired form does not stay quiet long.
  void clear_buckets() {
    detail::fill_kernel<<<grid_for(num_buckets_), kBlock, 0, stream_>>>(
        d_heads(), num_buckets_, kNoNode);
    check(cudaGetLastError(), "clear_buckets launch");
  }

  // Sets prev and next on every node, and touches nothing else — not the key,
  // which means nothing until insert() writes it, and not the fields belonging
  // to other components. Enqueued on the storage's stream.
  //
  // A caller with its own initialisation kernel should call Map::clear_links
  // from there and skip this.
  void clear_links() {
    detail::clear_links_kernel<Map><<<grid_for(capacity_), kBlock, 0, stream_>>>(
        d_nodes(), capacity_);
    check(cudaGetLastError(), "clear_links launch");
  }

  cudaStream_t stream() const { return stream_; }

  int64_t capacity() const { return capacity_; }
  int64_t num_buckets() const { return num_buckets_; }
  // The node array, for the components that own the other fields.
  Node* nodes() const { return d_nodes(); }

  size_t bytes() const {
    return sizeof(NodeId) * size_t(num_buckets_) +
           sizeof(Node) * size_t(capacity_);
  }

  // -- inspection -----------------------------------------------------------

  std::vector<NodeId> download_heads() const {
    return download(d_heads(), size_t(num_buckets_));
  }
  std::vector<Node> download_nodes() const {
    return download(d_nodes(), size_t(capacity_));
  }

  // Every node sitting in a bucket must be one whose key hashes to that bucket,
  // and the chain it sits in must agree with itself in both directions.
  //
  // This is the replacement for an invariant that used to be enforced: when the
  // map owned the key array, writing a key without moving buckets was
  // unreachable. Sharing the node type with other components gives that ability
  // back, so what is left is detection. Anything but zero means somebody wrote
  // node.key directly, or edited a chain without holding the lock.
  //
  // Driven from the buckets rather than from the nodes, because there is no way
  // to ask a node whether it is bound — that fact lives in the caller's state
  // word, not here. The consequence is one blind spot: a node that *should* be
  // in a chain and is in none looks exactly like a node that was never inserted.
  // Everything that is in a chain and wrong is seen.
  int64_t check_bindings() const {
    const auto nodes = download_nodes();
    const auto heads = download_heads();
    std::vector<char> seen(static_cast<size_t>(capacity_), 0);
    int64_t bad = 0;
    for (int64_t b = 0; b < num_buckets_; ++b) {
      NodeId cur = heads[static_cast<size_t>(b)];
      NodeId expect_prev = kNoNode;
      while (cur != kNoNode) {
        // Catches a head left locked as well as an id off the end of the pool.
        if (cur < 0 || static_cast<int64_t>(cur) >= capacity_) {
          ++bad;
          break;
        }
        // A node reached twice is a cycle, or a node in two buckets at once.
        if (seen[static_cast<size_t>(cur)]) {
          ++bad;
          break;
        }
        seen[static_cast<size_t>(cur)] = 1;
        const Node& n = nodes[static_cast<size_t>(cur)];
        if (bucket_of_host(n.key) != b) ++bad;
        if (n.prev != expect_prev) ++bad;
        expect_prev = cur;
        cur = n.next;
      }
    }
    return bad;
  }

 private:
  int64_t bucket_of_host(Key key) const {
    return static_cast<int64_t>(HashFn{}(key) &
                                static_cast<uint64_t>(num_buckets_ - 1));
  }
  static int64_t round_up_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
  }
  static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("HashMapStorage: ") + what + ": " +
                               cudaGetErrorString(e));
    }
  }

  template <typename T>
  std::vector<T> download(const T* src, size_t n) const {
    std::vector<T> out(n);
    check(cudaMemcpyAsync(out.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost,
                          stream_),
          "download");
    // These are diagnostics, called from the host, and the caller expects the
    // answer in the vector it is handed. Ordering them on the storage's stream
    // is what makes them safe to call right after an asynchronous constructor.
    check(cudaStreamSynchronize(stream_), "download sync");
    return out;
  }
  NodeId* d_heads() const { return static_cast<NodeId*>(heads_.data()); }
  Node* d_nodes() const { return static_cast<Node*>(nodes_.data()); }

  int64_t capacity_ = 0;
  int64_t num_buckets_ = 0;
  cudaStream_t stream_ = nullptr;
  AllocT alloc_;
  Buffer heads_, nodes_;
};

}  // namespace embedding_cache
