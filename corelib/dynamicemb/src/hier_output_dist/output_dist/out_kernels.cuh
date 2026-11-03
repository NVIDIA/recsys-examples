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
// What both out kernels share: their timeline, the call's tables as a
// requester's warps read them, their task schedule, and their protocol.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/output_dist/args.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The out kernels' timeline (a build with DYNAMICEMB_HIER_TRACE=1, kTrace):
// stamps, counters and bins, which each CTA keeps in shared memory and folds
// into the kernel's row. Without kTrace it compiles away.

// The timeline of an out kernel: its row (`trace`, the call's args.trace,
// nullptr without kTrace) and the CTA's shared copy (TraceSmSlots, in the
// kernel's dynamic shared memory after free_seen, out_shared). Stamps check
// on() themselves; the rest is for when it is on. Each member does its work
// only with kTrace.
struct OutTimeline {
  unsigned long long *trace;
  int local_size;  // free_seen's words, before the shared copy
  __device__ __forceinline__ bool on() const { return kTrace && trace != nullptr; }

  // One thread: `slot` keeps the earliest (or the latest) time any CTA
  // stamps it with.
  __device__ __forceinline__ void stamp_earliest(int slot) const { stamp(slot, true); }
  __device__ __forceinline__ void stamp_latest(int slot) const { stamp(slot, false); }

  // The CTA's shared copy.
  __device__ __forceinline__ unsigned long long *shared() const {
    return out_shared() + local_size;
  }
  // At the top of the kernel, every thread of the CTA (on() is the same for
  // all): the CTA's start, the kernel's first and last CTA start, and the
  // shared copy set to its start values.
  __device__ __forceinline__ void begin_cta() const {
    if constexpr (kTrace) {
      if (trace == nullptr) return;
      unsigned long long *sm = shared();
      for (int i = threadIdx.x; i < kTraceSmWords; i += blockDim.x)
        sm[i] = i == kSmNvFirst || i == kSmFirstLanded ? ~0ull : 0ull;
      __syncthreads();
      if (threadIdx.x == 0) {
        const unsigned long long now = now_ns();
        sm[kSmStart] = now;
        atomicMin(trace + kTraceStart, now);
        atomicMax(trace + kTraceStartLast, now);
      }
      __syncthreads();
    }
  }
  // A worker's task (lane 0), started at `start`: it moved `nv` bytes over
  // NVLink and `local` bytes of rows within this GPU.
  __device__ __forceinline__ void task(unsigned long long start, unsigned long long nv,
                                       unsigned long long local) const {
    if constexpr (kTrace) {
      unsigned long long *sm = shared();
      if (nv > 0) {
        const unsigned long long now = now_ns();
        atomicMin(sm + kSmNvFirst, start);
        atomicMax(sm + kSmNvLast, now);
        atomicAdd(sm + kSmNvBytes, nv);
        add(sm + kSmBins, 0, now, nv);
      }
      if (local > 0) atomicAdd(sm + kSmLocalBytes, local);
    }
  }
  // Rows from another node landed (`rows` more, the first ones when `first`).
  __device__ __forceinline__ void landed(unsigned long long rows, bool first) const {
    if constexpr (kTrace) {
      unsigned long long *sm = shared();
      const unsigned long long now = now_ns();
      if (first) atomicMin(sm + kSmFirstLanded, now);
      atomicAdd(sm + kSmIbIn, rows);
      add(sm + kSmBins, 1, now, rows);
    }
  }
  // The same once the CTA folded its shared copy (the backward's last CTA):
  // straight into the kernel's row.
  __device__ __forceinline__ void landed_late(unsigned long long rows, bool first) const {
    if constexpr (kTrace) {
      const unsigned long long now = now_ns();
      if (first) atomicMin(trace + kTraceFirstLanded, now);
      atomicAdd(trace + kTraceIbIn, rows);
      add(trace + kTraceBinNv, 1, now, rows);
    }
  }
  // Rows put to another node (`rows` more).
  __device__ __forceinline__ void posted(unsigned long long rows) const {
    if constexpr (kTrace) {
      atomicAdd(shared() + kSmIbOut, rows);
      add(shared() + kSmBins, 2, now_ns(), rows);
    }
  }
  // Every thread of the CTA, once every warp of the CTA is done with the
  // shared copy: thread 0 folds the stamps and counters into the kernel's
  // row, every thread some bins.
  __device__ __forceinline__ void fold_cta() const {
    if constexpr (kTrace) {
      const unsigned long long *sm = shared();
      unsigned long long *row = trace;
      if (threadIdx.x == 0) {
        if (sm[kSmNvFirst] != ~0ull) atomicMin(row + kTraceNvFirst, sm[kSmNvFirst]);
        if (sm[kSmFirstLanded] != ~0ull) atomicMin(row + kTraceFirstLanded, sm[kSmFirstLanded]);
        if (sm[kSmNvLast] != 0) atomicMax(row + kTraceNvLast, sm[kSmNvLast]);
        const int sums[4][2] = {{kSmNvBytes, kTraceNvBytes},
                                {kSmLocalBytes, kTraceLocalBytes},
                                {kSmIbIn, kTraceIbIn},
                                {kSmIbOut, kTraceIbOut}};
        for (int i = 0; i < 4; ++i)
          if (sm[sums[i][0]] != 0) atomicAdd(row + sums[i][1], sm[sums[i][0]]);
      }
      for (int i = threadIdx.x; i < 3 * kTraceBins; i += blockDim.x)
        if (sm[kSmBins + i] != 0) atomicAdd(row + kTraceBinNv + i, sm[kSmBins + i]);
    }
  }

 private:
  __device__ __forceinline__ void stamp(int slot, bool earliest) const {
    if constexpr (kTrace) {
      if (trace == nullptr) return;
      const unsigned long long now = now_ns();
      if (earliest)
        atomicMin(trace + slot, now);
      else
        atomicMax(trace + slot, now);
    }
  }
  // Timeline `which` (0: NVLink bytes, 1: rows landed, 2: rows posted) of
  // `bins` (the shared copy's or the kernel row's) gets `amount` at `time`,
  // in the bin of `time` after this CTA's start.
  __device__ __forceinline__ void add(unsigned long long *bins, int which, unsigned long long time,
                                      unsigned long long amount) const {
    const unsigned long long start = shared()[kSmStart];
    const unsigned long long bin = time > start ? (time - start) / kTraceBinNs : 0;
    atomicAdd(bins + which * kTraceBins + (int)min(bin, (unsigned long long)kTraceBins - 1),
              amount);
  }
};

// The call's tables as a requester's warps read them.

// What a requester's warps read of the call: from the plan (what both
// kernels' plans hold), the block starts and its tables of the node's
// proxies (their streams' chunks and ends, and its push tables there); from
// the window, the node peers' buffers and words, each the same place as in
// this rank's copy (layout()). The one word per proxy a CTA keeps,
// free_seen, is in its shared memory. Every thread of a warp makes the same
// one.
struct OutTables {
  const Sizes &sizes;
  const PlanView &plan;
  NodeWindow node;

  __device__ OutTables(ncclWindow_t window, const Sizes &op_sizes, const PlanView &call_plan)
      : sizes(op_sizes), plan(call_plan), node(window, op_sizes.topo.local_rank) {}
  __device__ __forceinline__ WindowLayout layout() const { return node.layout(sizes); }
  // Node peers' buffers and words.
  __device__ __forceinline__ char *lookup_rows(int peer) const {
    return node.peer(layout().rows.lookup_rows(), peer);
  }
  __device__ __forceinline__ char *proxy_rows(int proxy) const {
    return node.peer(layout().rows.proxy_rows(), proxy);
  }
  // Row `row` of a node peer's lookup_rows, slot `slot` of a proxy's proxy_rows.
  __device__ __forceinline__ char *lookup_row(int peer, int64_t row) const {
    return node.peer(layout().rows.lookup_row(row), peer);
  }
  __device__ __forceinline__ char *proxy_row(int proxy, int64_t slot) const {
    return node.peer(layout().rows.proxy_row(slot), proxy);
  }
  // A push task's flag at a proxy.
  __device__ __forceinline__ uint64_t *push_flag(int proxy, int64_t flag) const {
    return node.peer(layout().rows.push_flags() + flag, proxy);
  }
  // A proxy's words of its chunk `chunk`: when it landed, and the rows its
  // node's requesters pulled of it.
  __device__ __forceinline__ const uint64_t *landed(int proxy, int64_t chunk) const {
    return node.peer(layout().ctrl.landed(chunk), proxy);
  }
  __device__ __forceinline__ uint64_t *pulled_rows(int proxy, int64_t chunk) const {
    return node.peer(layout().ctrl.pulled_rows(chunk), proxy);
  }
  __device__ __forceinline__ const uint64_t *slot_free(int proxy) const {
    return node.peer(layout().ctrl.slot_free(), proxy);
  }
  // The most of a proxy's slot_free word that this CTA's warps saw.
  __device__ __forceinline__ uint64_t *free_seen(int proxy) const {
    return (uint64_t *)out_shared() + proxy;
  }
  // This rank's push order at a proxy in a chunk (positions of push_order)
  // and the next chunk's entry, which holds the ends.
  __device__ __forceinline__ void req_of(int proxy, int64_t chunk, ReqEntry &entry,
                                         ReqEntry &next) const {
    const ReqEntry *push_entry = plan.push_entry(proxy, chunk);
    entry = push_entry[0];
    next = push_entry[1];
  }
  // The owner of one of this rank's rows (its bucketized rows by owner).
  __device__ __forceinline__ int owner_of(int64_t row) const {
    return seg_search(plan.block_start, sizes.topo.world_size, row);
  }
};

// How the warps of an out kernel take their tasks: the task queues, the row
// tasks of the node's owners and sub-tasks spread among them, and the remote
// tasks laid out by chunk.

// A task queue of the out kernel (`word`, its task word in this rank's ctrl
// section, CtrlLayout::task): its warps take the tasks in order, one at a
// time; the last CTA sets every queue back to 0 for the next kernel.
struct TaskQueue {
  uint64_t *word;
  // Every lane of the warp: the warp's next task.
  __device__ __forceinline__ int64_t next_warp() const {
    int64_t task = 0;
    if ((threadIdx.x & 31) == 0) task = (int64_t)atomicAdd((unsigned long long *)word, 1ull);
    return __shfl_sync(kFull, task, 0);
  }
  // One thread of the last CTA, once every other CTA has taken its last task.
  __device__ __forceinline__ static void reset_all(const CtrlLayout &ctrl) {
    for (int queue = 0; queue < kNumQueues; ++queue) *ctrl.task(queue) = 0;
  }
};

// A row task: kTaskRows rows of an owner's block (from row0), owners in turn
// from this rank's own one, so the ranks of a node start on different peers.
// count = 0 when the task is past the end of that owner's block.
struct RowTask {
  int owner, count;
  int64_t row0;
};
// The row tasks of the node's owners (block_start: theirs, this rank `me`
// of local_size): every owner gets as many as the largest block needs.
struct LocalRows {
  const int64_t *block_start;
  int local_size, me;
  __device__ __forceinline__ int64_t count() const {
    int64_t most = 0;
    for (int owner = 0; owner < local_size; ++owner)
      most = max(most, block_start[owner + 1] - block_start[owner]);
    return local_size * ceil_div<kTaskRows>(most);
  }
  __device__ __forceinline__ RowTask task(int64_t task) const {
    RowTask job;
    job.owner = (int)((me + task) % local_size);
    job.row0 = block_start[job.owner] + task / local_size * kTaskRows;
    job.count =
        (int)min((int64_t)kTaskRows, max(block_start[job.owner + 1] - job.row0, (int64_t)0));
    return job;
  }
};
// num_tasks + num_subs slots, the num_subs sub-tasks spread evenly among the
// num_tasks others.
struct Interleave {
  int64_t num_tasks, num_subs;
  // Whether slot `slot` is a sub-task, and its index among its kind (index).
  __device__ __forceinline__ bool is_sub(int64_t slot, int64_t &index) const {
    const int64_t before = slot * num_subs / (num_tasks + num_subs);  // sub-tasks in [0, slot)
    const bool sub = (slot + 1) * num_subs / (num_tasks + num_subs) > before;
    index = sub ? before : slot - before;
    return sub;
  }
};

// A task of the remote tasks laid out by (chunk, proxy in turn): its entry
// in their task starts (starts [entries + 1]: the tasks before each entry),
// the entry it falls in, and its place there.
struct RemoteTask {
  int64_t entry, nth;
  __device__ __forceinline__ RemoteTask(const int64_t *starts, int64_t entries, int64_t task)
      : entry(seg_search(starts, (int)entries, task)), nth(task - starts[entry]) {}
};

// The out channel's protocol, which both out kernels follow: the entry
// barrier, a CTA's arrival at its end, and the landed_rows words.

// The out channel's entry barrier, on its node barrier word (out_started):
// CTA 0 publishes this rank's out_seq, after a system
// fence for what earlier kernels wrote (lookup_rows, which node peers read)
// and what CTA 0 reset for this kernel; every CTA then waits until all node
// peers have published theirs. A peer there has ended its lookup (which wrote
// its lookup_rows before the kernel) and every earlier out kernel. Every
// thread of threads [0, threads) (cta_sync), in every CTA; seq is the
// kernel's out_seq.
__device__ __forceinline__ void pass_entry_barrier(ncclWindow_t window, const Sizes &sizes,
                                                   int64_t seq, const OutTimeline &timeline,
                                                   int threads) {
  const int tid = threadIdx.x, local_size = sizes.topo.local_size;
  const NodeWindow node(window, sizes.topo.local_rank);
  const NodeBarrier barrier{sizes, node, node.layout(sizes).ctrl.out_started(), seq};
  if (blockIdx.x == 0 && tid == 0) {
    fence_sc_sys();
    barrier.publish();
  }
  for (int peer = tid; peer < local_size; peer += threads)
    barrier.wait(peer, "out-channel node barrier");
  cta_sync(threads);
  if (tid == 0) {
    timeline.stamp_earliest(kTraceEnterFirst);
    timeline.stamp_latest(kTraceEntered);
  }
}

// The end of a CTA of an out kernel: each CTA arrives once in its arrive word, a
// running total over the rank's out kernels (so nothing is reset), and the
// one whose arrival brings it to arrive0 (the word before this kernel) +
// gridDim.x is this kernel's last. Everything a CTA wrote and every load it
// made are ordered before its arrival, so the last CTA sees everything the
// others did. Every thread of the CTA (tracing: the CTA first folds its
// shared copy of the timeline into the kernel's row). Returns, to every
// thread, whether this CTA is the last.
__device__ __forceinline__ bool arrive_cta(ncclWindow_t window, const Sizes &sizes, int64_t arrive0,
                                           const OutTimeline &timeline) {
  const int64_t target = arrive0 + gridDim.x;
  __shared__ int last;
  __syncthreads();
  if (timeline.on()) timeline.fold_cta();
  if (threadIdx.x == 0) {
    fence_sc_sys();
    uint64_t *word = NodeWindow(window, sizes.topo.local_rank).layout(sizes).ctrl.arrive();
    last = (int64_t)atomicAdd((unsigned long long *)word, 1ull) + 1 == target;
    if (last) fence_sc_sys();
  }
  __syncthreads();
  return last;
}

// A proxy's landed_rows word (one per owner node): out_seq, and the rows of
// that owner node that have landed during forward out_seq.
using LandedRows = TaggedWord<32>;

}  // namespace hier_output_dist
}  // namespace dyn_emb
