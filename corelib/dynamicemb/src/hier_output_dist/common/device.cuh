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
// A poll of a flag, when the reader makes a fence after it before what the
// flag guards (an acquire pattern). An acquire load drops the whole L1 cache
// of its SM (CCTL.IVALL): a warp that polls with it in a fast loop makes the
// other warps of its SM miss in L1.
__device__ __forceinline__ uint64_t ld_relaxed_sys(const uint64_t *addr) {
  uint64_t value;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(addr) : "memory");
  return value;
}
__device__ __forceinline__ void st_release_sys(uint64_t *addr, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
}
// A store of a flag that another GPU reads, when the writer made a fence
// before it after what the flag guards (a release pattern).
__device__ __forceinline__ void st_relaxed_sys(uint64_t *addr, uint64_t value) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
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
// A system-scope fence that is not sequentially consistent (MEMBAR.ALL, where
// fence.sc is MEMBAR.SC). After relaxed polls of flags (ld_relaxed_sys) it is
// an acquire of what they guard, and before relaxed stores of flags
// (st_relaxed_sys) or puts it is a release of what this thread saw and wrote
// before it. Like an acquire load, it drops the L1 cache of the SM.
__device__ __forceinline__ void fence_acq_rel_sys() {
  asm volatile("fence.acq_rel.sys;" ::: "memory");
}

// The GPU's clock (%globaltimer) and bounded spin-waits.

__device__ __forceinline__ uint64_t now_ns() {
  uint64_t ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
  return ns;
}

// Waits report instead of hanging forever: after `limit` ns the kernel prints
// what it was waiting for and traps (timed_out). The report is a separate
// function: when it was inline, its printf used registers at each wait, also
// at the waits in the task loops of the out kernels.
[[noreturn]] __device__ __noinline__ inline void timed_out(const char *what, long long first,
                                                           long long second, int rank) {
  printf("[hier_output_dist] rank %d timed out waiting for %s (%lld, %lld)\n", rank, what, first,
         second);
  __trap();
}
struct Spin {
  uint64_t start_ns, limit;
  uint32_t polls = 0;
  __device__ explicit Spin(uint64_t limit_ns) : start_ns(now_ns()), limit(limit_ns) {}
  __device__ __forceinline__ void pause(const char *what, long long first, long long second,
                                        int rank) {
    if (((++polls) & 1023) == 0 && now_ns() - start_ns > limit)
      timed_out(what, first, second, rank);
    __nanosleep(64);
  }
};

// Warp-wide helpers: the mask of every lane, a lane's pointer in every lane,
// and the run of a ballot.

constexpr unsigned kFull = 0xffffffffu;

// Lane src_lane's pointer, in every lane of the warp (every lane calls it).
template <typename T>
__device__ __forceinline__ T *from_lane(T *ptr, int src_lane) {
  return (T *)__shfl_sync(kFull, (unsigned long long)ptr, src_lane);
}

// The lanes of a ballot that voted true from lane 0 on, before the first
// that did not (32 when all did).
__device__ __forceinline__ int ballot_run(unsigned ballot) { return __clz(__brev(~ballot)); }

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
// each, kVectors vectors in flight per lane (in registers: 4 each). Loads go
// through L2 only (ld.global.cg): a row another GPU wrote must not come from a
// stale L1 line.
template <int kVectors>
__device__ __forceinline__ void warp_copy_rows(const char *src, char *dst, int num_rows, int vec,
                                               int lane) {
  const int total = num_rows * vec;
  for (int base = 0; base < total; base += 32 * kVectors) {
    int4 vecs[kVectors];
#pragma unroll
    for (int step = 0; step < kVectors; ++step) {
      const int idx = base + step * 32 + lane;
      const int row = idx / vec;
      const char *src_row = from_lane(src, row & 31);
      if (idx < total) vecs[step] = __ldcg((const int4 *)src_row + (idx - row * vec));
    }
#pragma unroll
    for (int step = 0; step < kVectors; ++step) {
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

// Syncs between the ranks of a node through their control words: the node
// barrier (each rank publishes its own word and waits for every peer's),
// flags per pair of node peers (each rank sets its flag in every peer's words
// and waits for every peer's in its own), and words that carry a payload
// beside the sequence number (TaggedWord). The words hold the sequence number
// of the kernel that writes them (`seq`: in_seq or out_seq), so none is ever
// reset: a later kernel waits for a larger one.

// The node barrier on one control word (`word`, this rank's copy of it), of
// the kernel with sequence number `seq`: one thread publishes seq in this
// rank's word; threads [0, local_size) then wait, each until one node peer's
// word has it. That peer has then done everything it did before it
// published (its release, our acquire): all its earlier kernels of the same
// sequence too (they run in issue order).
struct NodeBarrier {
  const Sizes &sizes;
  const NodeWindow &node;
  uint64_t *word;
  int64_t seq;
  // One thread: this rank's word gets the sequence number.
  __device__ __forceinline__ void publish() const { st_release_sys(word, (uint64_t)seq); }
  // Waits until node peer `peer` has published (spin reports `what`).
  __device__ __forceinline__ void wait(int peer, const char *what) const {
    Spin spin(sizes.timeout_ns);
    const uint64_t *theirs = node.peer(word, peer);
    while ((int64_t)ld_acquire_sys(theirs) < seq) spin.pause(what, peer, seq, sizes.topo.rank);
  }
};

// Flags of the node's ranks, one per pair: node peer r's flag in a rank's
// copy is flags[r] (`flags`: this rank's copy). Each rank tells every peer
// (sets its flag in the peer's copy to the sequence number), then waits
// until every peer told it. Its fields as NodeBarrier's.
struct PeerFlags {
  const Sizes &sizes;
  const NodeWindow &node;
  uint64_t *flags;
  int64_t seq;
  // This rank tells node peer `peer`.
  __device__ __forceinline__ void tell(int peer) const {
    st_release_sys(node.peer(flags + sizes.topo.local_rank, peer), (uint64_t)seq);
  }
  // Waits until node peer `peer` told this rank.
  __device__ __forceinline__ void wait(int peer, const char *what) const {
    Spin spin(sizes.timeout_ns);
    while ((int64_t)ld_acquire_sys(flags + peer) < seq)
      spin.pause(what, peer, seq, sizes.topo.rank);
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

// The out kernels are built once for each GIN backend (kGin, an
// ncclGinType_t; NCCL_GIN_TYPE_NONE on one node, where the kernels use no
// GIN). The host launches the kernels of the backend that all GIN contexts
// of its communicator have (Communicator::gin_type). NCCL compiles each GIN
// call for all backends and selects one at run time from the backend of the
// context. A context that holds kGin as a constant keeps only the code of
// kGin. Together, the code of all the backends used more registers than the
// out kernels have. `mode`: who uses its QPs (one each per rail peer). GPU,
// the default, lets the warps of every CTA use them. In GPU mode, each put
// takes its QP's slots with GPU-scope atomics and a GPU-scope acquire fence,
// which drops the L1 cache of the SM (the other warps of CTA 0 then miss in
// L1). An issuer warp of CTA 0 is the only user of its rows context's QPs
// (issuer_sharing).
template <ncclGinType_t kGin>
__device__ __forceinline__ ncclGin
gin_context(const ncclDevComm &dev, int index,
            ncclGinResourceSharingMode mode = NCCL_GIN_RESOURCE_SHARING_GPU) {
  static_assert(kGin != NCCL_GIN_TYPE_NONE, "one node has no GIN context");
  ncclGin gin(dev, index, mode);
  gin._ginBackend = (uint32_t)kGin;
  return gin;
}

// The mode of an issuer's rows context (kCtxFwd, kCtxBwd). The issuer warp
// of CTA 0 is the only user of its QPs, and one lane posts on each QP: lane
// l on those of the rail peers on nodes l, l + 32, ... The warp's flush
// comes after a __syncwarp. On GDAKI, THREAD: a QP has one posting thread,
// so a put takes its slots and rings the doorbell with plain loads and
// stores, without atomics, a lock or CTA fences. GPI has one channel per
// context for all peers, on which every lane of a warp's flush posts: CTA
// there, as DeepEP does for a QP that one SM uses. The proxy backend does
// not use the mode.
template <ncclGinType_t kGin>
__device__ __forceinline__ constexpr ncclGinResourceSharingMode issuer_sharing() {
  return kGin == NCCL_GIN_TYPE_GDAKI ? NCCL_GIN_RESOURCE_SHARING_THREAD
                                     : NCCL_GIN_RESOURCE_SHARING_CTA;
}

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
  // The same without acquire (ld_relaxed_sys): the caller makes a fence after
  // it before what the credit allows, or reads nothing the credit guards (the
  // forward's issuer: fwd::OwnerIssuer).
  __device__ __forceinline__ uint64_t read_relaxed(int sig) const {
    return gin.readSignal(sig, 64, cuda::memory_order_relaxed);
  }
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
// after the writes they carry and before them. The same fence is the
// acquire of the caller's relaxed polls before it.
struct RowPut {
  const ncclGin &gin;
  const ncclTeam &rail;
  const NodeWindow &node;
  char *dst_rows;
  const char *src_rows;
  int64_t row_bytes;
  int sig;
  __device__ __forceinline__ void fence() const { fence_acq_rel_sys(); }
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
