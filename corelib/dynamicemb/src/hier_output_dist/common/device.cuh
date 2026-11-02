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
// The kernels' building blocks: loads, stores and fences with a memory
// order, bounded waits, warp and CTA helpers, a search, row vectors, the
// window as the node's GPUs map it (NodeWindow), the syncs on its control
// words, and GIN on the rail.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <nccl_device.h>

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/window.h"

namespace dyn_emb {
namespace hier_output_dist {

// Loads, stores, reductions, atomics and fences with an explicit memory order
// and scope (inline PTX).

// Flags that another GPU writes are read with system-scope acquire, and the
// data they guard with ld.global.cg (never a stale L1 line); words only this
// GPU's CTAs write, with gpu-scope acquire.
__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t *addr) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(addr) : "memory");
  return value;
}
__device__ __forceinline__ uint64_t ld_acquire_gpu(const uint64_t *addr) {
  uint64_t value;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"(addr) : "memory");
  return value;
}
__device__ __forceinline__ void st_release_sys(uint64_t *addr, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
}
__device__ __forceinline__ void red_release_sys(uint64_t *addr, uint64_t value) {
  asm volatile("red.release.sys.global.add.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
}
__device__ __forceinline__ void red_release_gpu(uint64_t *addr, uint64_t value) {
  asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
}
// A word the warps of one CTA share (generic address, shared memory included).
__device__ __forceinline__ uint64_t ld_acquire_cta(const uint64_t *addr) {
  uint64_t value;
  asm volatile("ld.acquire.cta.u64 %0, [%1];" : "=l"(value) : "l"(addr) : "memory");
  return value;
}
__device__ __forceinline__ void red_max_release_cta(uint64_t *addr, uint64_t value) {
  asm volatile("red.release.cta.max.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
}
__device__ __forceinline__ uint64_t atom_add_acq_rel_gpu(uint64_t *addr, uint64_t value) {
  uint64_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;"
               : "=l"(old)
               : "l"(addr), "l"(value)
               : "memory");
  return old;
}
__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }

// The GPU's clock (%globaltimer) and bounded spin-waits.

__device__ __forceinline__ uint64_t now_ns() {
  uint64_t ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
  return ns;
}

// Waits report instead of hanging forever: after `limit` ns the kernel prints
// what it was waiting for and traps.
struct Spin {
  uint64_t start_ns, limit;
  uint32_t polls = 0;
  __device__ explicit Spin(uint64_t limit_ns) : start_ns(now_ns()), limit(limit_ns) {}
  __device__ __forceinline__ void pause(const char *what, long long first, long long second,
                                        int rank) {
    if (((++polls) & 1023) == 0 && now_ns() - start_ns > limit) {
      printf("[hier_output_dist] rank %d timed out waiting for %s (%lld, %lld)\n", rank, what,
             first, second);
      __trap();
    }
    __nanosleep(64);
  }
};

// Warp-wide helpers: the mask of every lane, and a lane's pointer in every
// lane.

constexpr unsigned kFull = 0xffffffffu;

// Lane src_lane's pointer, in every lane of the warp (every lane calls it).
template <typename T>
__device__ __forceinline__ T *from_lane(T *ptr, int src_lane) {
  return (T *)__shfl_sync(kFull, (unsigned long long)ptr, src_lane);
}

// The threads of a CTA: a sync of some of them.

// Threads [0, threads) of a CTA of kThreads threads (whole warps) wait for
// each other: all of them, or fewer on named barrier 1 while the other warps
// run on. Every thread of the group calls it.
__device__ __forceinline__ void cta_sync(int threads) {
  if (threads == kThreads)
    __syncthreads();
  else
    asm volatile("bar.sync 1, %0;" ::"r"(threads) : "memory");
}

// A binary search: the segment a value falls in.

// Smallest i in [0, count) with bounds[(i + 1) * stride] > value, for
// non-decreasing bounds[0], bounds[stride], ..., bounds[count * stride] with
// bounds[0] <= value < bounds[count * stride]: the segment value falls in,
// empty segments skipped. bounds[count * stride] is never read, so it need not
// exist.
template <typename BoundT>
__device__ __forceinline__ int seg_search(const BoundT *bounds, int count, int64_t value,
                                          int stride = 1) {
  int lo = 0, hi = count - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if ((int64_t)bounds[(int64_t)mid * stride] <= value)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

// Rows of the output dist: a warp copy of rows of 16-byte vectors, the
// vectors of each row dtype as fp32 values, and a product rounded to nearest.

// Every lane holds one row of up to 32: `src`/`dst` byte pointers (lanes with
// lane >= num_rows hold anything). Copies num_rows rows of vec 16-byte vectors
// each, 8 vectors in flight per lane. Loads go through L2 only
// (ld.global.cg): a row another GPU wrote must not come from a stale L1 line.
__device__ __forceinline__ void warp_copy_rows(const char *src, char *dst, int num_rows, int vec,
                                               int lane) {
  constexpr int kUnroll = 8;
  const int total = num_rows * vec;
  for (int base = 0; base < total; base += 32 * kUnroll) {
    int4 vecs[kUnroll];
#pragma unroll
    for (int step = 0; step < kUnroll; ++step) {
      const int idx = base + step * 32 + lane;
      const int row = idx / vec;
      const char *src_row = from_lane(src, row & 31);
      if (idx < total) vecs[step] = __ldcg((const int4 *)src_row + (idx - row * vec));
    }
#pragma unroll
    for (int step = 0; step < kUnroll; ++step) {
      const int idx = base + step * 32 + lane;
      const int row = idx / vec;
      char *dst_row = from_lane(dst, row & 31);
      if (idx < total) *((int4 *)dst_row + (idx - row * vec)) = vecs[step];
    }
  }
}

// A row value as fp32 (exactly), and an fp32 value rounded once to the row
// dtype (to nearest even).
__device__ __forceinline__ float to_float(float value) { return value; }
__device__ __forceinline__ float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
__device__ __forceinline__ float to_float(__half value) { return __half2float(value); }
template <typename RowT>
__device__ __forceinline__ RowT from_float(float value) {
  if constexpr (std::is_same<RowT, __nv_bfloat16>::value)
    return __float2bfloat16_rn(value);
  else if constexpr (std::is_same<RowT, __half>::value)
    return __float2half_rn(value);
  else
    return value;
}

// 16-byte vectors of the row dtype as kValues fp32 values: get converts
// them, add adds them in fp32, pack rounds them back.
template <typename RowT>
struct Vec16 {
  static constexpr int kValues = 16 / sizeof(RowT);
  __device__ static void get(float *vals, int4 vec) {
    const RowT *elems = reinterpret_cast<const RowT *>(&vec);
#pragma unroll
    for (int i = 0; i < kValues; ++i) vals[i] = to_float(elems[i]);
  }
  __device__ static void add(float *acc, int4 vec) {
    const RowT *elems = reinterpret_cast<const RowT *>(&vec);
#pragma unroll
    for (int i = 0; i < kValues; ++i) acc[i] += to_float(elems[i]);
  }
  __device__ static int4 pack(const float *vals) {
    int4 vec;
    RowT *elems = reinterpret_cast<RowT *>(&vec);
#pragma unroll
    for (int i = 0; i < kValues; ++i) elems[i] = from_float<RowT>(vals[i]);
    return vec;
  }
};

// value * scale rounded to nearest, never fused into a multiply-add and with
// subnormals kept, whatever the build's fast-math flags: what PyTorch's
// division of a CUDA tensor by a CPU scalar computes (a multiplication by the
// reciprocal, in fp32), bit for bit.
__device__ __forceinline__ float mul_rn(float value, float scale) {
  float product;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(product) : "f"(value), "f"(scale));
  return product;
}

// The op's window as the GPUs of a node see it (NodeWindow).

// Every GPU of the node maps every node peer's copy of the window (NCCL's LSA
// team, over NVLink). A kernel lays the window out over this rank's copy
// (layout()) and names places in it by plain pointers; peer() is the same
// place in a node peer's copy, and offset() where it is in every copy, which
// GIN puts take.
struct NodeWindow {
  ncclWindow_t window;
  char *base;  // this rank's copy, as its node peers map it
  __device__ __forceinline__ NodeWindow(ncclWindow_t handle, int local_rank)
      : window(handle), base((char *)ncclGetLsaPointer(handle, 0, local_rank)) {}
  __device__ __forceinline__ WindowLayout layout(const Sizes &sizes) const {
    return WindowLayout(sizes, base);
  }
  // Node peer lsa_peer's copy of what `mine` points to in this rank's copy.
  template <typename T>
  __device__ __forceinline__ T *peer(T *mine, int lsa_peer) const {
    return (T *)ncclGetLsaPointer(window, offset(mine), lsa_peer);
  }
  // Where `mine` is in the window, in bytes.
  __device__ __forceinline__ size_t offset(const void *mine) const {
    return (size_t)((const char *)mine - base);
  }
};

// The part of the kernels' arguments that both channels share and that is
// fixed for the op's life: its sizes, by which a kernel lays out its window
// and its tables, and the window itself. The op makes it once, after its
// window is registered. The in channel's comm kernels that wait take it with
// the call's sequence number (CommArgs), the out kernels with theirs
// (OutArgs), and the exit round's kernel alone.
struct ChannelStatic : Sizes {
  ncclWindow_t window;  // the op's window (window.h)
  uint64_t timeout_ns;  // how long a wait for a peer may take
  ncclDevComm dev;      // GIN across nodes
};
static_assert(std::is_trivially_copyable<ChannelStatic>::value,
              "a kernel parameter is copied as bytes");

// Syncs between the ranks of a node through their control words: the node
// barrier (each rank publishes its own word and waits for every peer's),
// flags per pair of node peers (each rank sets its flag in every peer's words
// and waits for every peer's in its own), and words that carry a payload
// beside the sequence number (TaggedWord). The words hold the channel's
// sequence number, so none is ever reset: a later kernel waits for a larger
// one.

// The node barrier on one control word (`word`, this rank's copy of it), for
// a channel's kernel (Args, CommArgs or OutArgs, gives its Channel, Topo and
// timeout): one thread publishes this rank's sequence number in its word;
// threads [0, local_size) then wait, each until one node peer's word has it.
// That peer has then done everything it did before it published (its
// release, our acquire): all its earlier work on the channel too (its kernels
// run in issue order).
template <typename Args>
struct NodeBarrier {
  const Args &args;
  const NodeWindow &node;
  uint64_t *word;
  __device__ __forceinline__ NodeBarrier(const Args &channel_args, const NodeWindow &window,
                                         uint64_t *mine)
      : args(channel_args), node(window), word(mine) {}
  // One thread: this rank's word gets the sequence number.
  __device__ __forceinline__ void publish() const { st_release_sys(word, (uint64_t)args.ch.seq); }
  // Waits until node peer `peer` has published (spin reports `what`).
  __device__ __forceinline__ void wait(int peer, const char *what) const {
    Spin spin(args.timeout_ns);
    const uint64_t *theirs = node.peer(word, peer);
    while ((int64_t)ld_acquire_sys(theirs) < args.ch.seq)
      spin.pause(what, peer, args.ch.seq, args.topo.rank);
  }
};

// Flags of the node's ranks, one per pair: node peer r's flag in a rank's
// copy is flags[r] (`flags`: this rank's copy). Each rank tells every peer
// (sets its flag in the peer's copy to the sequence number), then waits
// until every peer told it. Args as for NodeBarrier.
template <typename Args>
struct PeerFlags {
  const Args &args;
  const NodeWindow &node;
  uint64_t *flags;
  __device__ __forceinline__ PeerFlags(const Args &channel_args, const NodeWindow &window,
                                       uint64_t *mine)
      : args(channel_args), node(window), flags(mine) {}
  // This rank tells node peer `peer`.
  __device__ __forceinline__ void tell(int peer) const {
    st_release_sys(node.peer(flags + args.topo.local_rank, peer), (uint64_t)args.ch.seq);
  }
  // Waits until node peer `peer` told this rank.
  __device__ __forceinline__ void wait(int peer, const char *what) const {
    Spin spin(args.timeout_ns);
    while ((int64_t)ld_acquire_sys(flags + peer) < args.ch.seq)
      spin.pause(what, peer, args.ch.seq, args.topo.rank);
  }
};

// A control word that holds a channel's sequence number in its high bits and
// a payload in its low PayloadBits bits, which its writer publishes (release)
// and readers wait for (acquire): landed_rows (rows, 32 bits).
template <int PayloadBits>
struct TaggedWord {
  __device__ __forceinline__ static uint64_t of(int64_t seq, uint64_t payload) {
    return ((uint64_t)seq << PayloadBits) | payload;
  }
  __device__ __forceinline__ static int64_t tag(uint64_t word) {
    return (int64_t)(word >> PayloadBits);
  }
  __device__ __forceinline__ static uint64_t payload(uint64_t word) {
    return word & ((1ull << PayloadBits) - 1);
  }
  // One thread: *word gets sequence number seq and the payload.
  __device__ __forceinline__ static void publish(uint64_t *word, int64_t seq, uint64_t payload) {
    st_release_sys(word, of(seq, payload));
  }
  // Waits until *word holds sequence number seq or a later one, and returns
  // it (a timeout reports `what`, the peer, seq and this rank).
  __device__ __forceinline__ static uint64_t wait(const uint64_t *word, int64_t seq, Spin &spin,
                                                  const char *what, int peer, int rank) {
    uint64_t value;
    while (((value = ld_acquire_sys(word)) >> PayloadBits) < (uint64_t)seq)
      spin.pause(what, peer, seq, rank);
    return value;
  }
};

// The rail: GIN between this rank and the ranks of its local index on the
// other nodes. The credits rail peers raise each other, puts of rows, and the
// in channel's messages. A GIN warp's lane l works with the rail peers on
// nodes l, l + 32, l + 64, ...

// Credits on the rail: payload-free signals of GIN context kCtxCredit (the
// caller's `gin`) whose value only grows. A sender adds the difference to the
// value it sent before, which it keeps in a word of its own, so its peer
// reads the latest value.
struct Credits {
  const ncclGin &gin;
  const ncclTeam &rail;
  __device__ __forceinline__ Credits(const ncclGin &context, const ncclTeam &team)
      : gin(context), rail(team) {}
  // Raises this rank's credit `sig` at the rail peer on node peer_node to
  // `value` (nothing when *sent, what it sent before, is that much already).
  // The signal makes a system-scope release of its own: what this thread did
  // before is ordered before the credit.
  __device__ __forceinline__ void raise(int peer_node, int sig, uint64_t value,
                                        uint64_t *sent) const {
    if (value <= *sent) return;
    gin.signal(rail, peer_node, ncclGin_StrongSignalAdd{(ncclGinSignal_t)sig, value - *sent},
               ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread,
               cuda::thread_scope_system);
    *sent = value;
  }
  // The credit `sig` that rail peers raised at this rank (acquire).
  __device__ __forceinline__ uint64_t read(int sig) const { return gin.readSignal(sig); }
  // Every lane of the warp: once it returns, no credit the warp raised is in
  // flight.
  __device__ __forceinline__ void flush_warp() const { gin.flush(ncclCoopWarp{}); }
};

// A put on the rail without a signal: `bytes` from `src` in this rank's copy
// of the window into the place of `dst` in the rail peer's copy on node
// peer_node (both pointers name places in this rank's copy). It makes no
// release of its own: the caller makes one system-scope fence after the
// writes it carries and before it.
__device__ __forceinline__ void rail_put(const ncclGin &gin, const ncclTeam &rail,
                                         const NodeWindow &node, int peer_node, const void *dst,
                                         const void *src, size_t bytes) {
  gin.put(rail, peer_node, node.window, node.offset(dst), node.window, node.offset(src), bytes,
          ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
          cuda::thread_scope_system, cuda::thread_scope_system);
}

// Puts of rows on the rail, from the rows at `src_rows` into those at
// `dst_rows` of a rail peer (both in this rank's copy of the window: the
// peer's at the same place), each adding its rows to the peer's signal
// `sig`. The signal is strong: a peer that sees it also sees this put and
// every earlier one on the context to it. The puts make no release of their
// own: they are told the caller made a system-scope release, which is the
// scope they need. So the caller makes one system-scope fence (fence())
// after the writes they carry and before them.
struct RowPut {
  const ncclGin &gin;
  const ncclTeam &rail;
  const NodeWindow &node;
  char *dst_rows;
  const char *src_rows;
  int64_t row_bytes;
  int sig;
  __device__ __forceinline__ void fence() const { fence_sc_sys(); }
  // Rows [src_row, src_row + rows) into [dst_row, dst_row + rows) of the rail
  // peer on node peer_node; with `count`, the put also counts on this rank's
  // counter `counter` once the NIC has read its rows.
  __device__ __forceinline__ void put(int peer_node, int dst_row, int src_row, int rows, bool count,
                                      int counter) const {
    const size_t dst = node.offset(dst_rows + (int64_t)dst_row * row_bytes);
    const size_t src = node.offset(src_rows + (int64_t)src_row * row_bytes);
    const ncclGin_StrongSignalAdd add{(ncclGinSignal_t)sig, (uint64_t)rows};
    if (!count)
      gin.put(rail, peer_node, node.window, dst, node.window, src, (size_t)rows * row_bytes, add,
              ncclGin_None{}, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_system,
              cuda::thread_scope_system);
    else
      gin.put(rail, peer_node, node.window, dst, node.window, src, (size_t)rows * row_bytes, add,
              ncclGin_WeakCounterInc{(ncclGinCounter_t)counter}, ncclCoopThread{}, ncclGin_None{},
              cuda::thread_scope_system, cuda::thread_scope_system);
  }
};

// One kind of the in channel's messages on the rail (GIN context kCtxIn, the
// caller's `gin`): a count message or a key stream. Every call sends each
// rail peer one message of each kind, which adds 1 to the peer's signal of
// that kind for this rank's node (`sig`). So a signal counts the calls whose
// message from that node has landed, a running total that reaches in_seq
// once this call's has. A message is one put, without a release of its own
// (as RowPut's: the caller makes one system-scope fence after the writes it
// carries and before the put); a put without a signal before it on the same
// context lands before it.
struct RailMessage {
  const ncclGin &gin;
  const ncclTeam &rail;
  const NodeWindow &node;
  const ncclGin_StrongSignalAdd sig;
  __device__ __forceinline__ RailMessage(const ncclGin &context, const ncclTeam &team,
                                         const NodeWindow &window, int signal)
      : gin(context), rail(team), node(window), sig{(ncclGinSignal_t)signal, 1} {}
  // `bytes` from `src` into the place of `dst` in the rail peer's copy on
  // node peer_node (both pointers name places in this rank's copy).
  __device__ __forceinline__ void put(int peer_node, const void *dst, const void *src,
                                      size_t bytes) const {
    gin.put(rail, peer_node, node.window, node.offset(dst), node.window, node.offset(src), bytes,
            sig, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_system,
            cuda::thread_scope_system);
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
