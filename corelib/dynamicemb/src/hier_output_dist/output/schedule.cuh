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
// How the warps of an out kernel take their tasks: the task queues, the row
// tasks of the node's owners and sub-tasks spread among them, and a warp's
// place in tasks laid out round after round.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/device/ctrl_words.cuh"
#include "hier_output_dist/device/warp.cuh"
#include "hier_output_dist/output/args.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// Task queue `queue` of the out kernel (Ctrl::tasks): its warps take the
// tasks in order, one at a time; the last CTA sets every queue back to 0 for
// the next kernel.
struct TaskQueue {
  const OutArgs &args;
  int queue;
  __device__ __forceinline__ TaskQueue(const OutArgs &out_args, int which)
      : args(out_args), queue(which) {}
  // Every lane of the warp: the warp's next task.
  __device__ __forceinline__ int64_t next_warp() const {
    int64_t task = 0;
    if ((threadIdx.x & 31) == 0)
      task = (int64_t)atomicAdd(
          (unsigned long long *)CtrlWords::of(args, args.topo.local_rank).task(queue), 1ull);
    return __shfl_sync(kFull, task, 0);
  }
  // One thread of the last CTA, once every other CTA has taken its last task.
  __device__ __forceinline__ static void reset_all(const OutArgs &args) {
    const CtrlWords mine = CtrlWords::of(args, args.topo.local_rank);
    for (int queue = 0; queue < kNumQueues; ++queue) *mine.task(queue) = 0;
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

// A warp's place in tasks laid out round after round: round rnd's are
// [start[rnd * stride], start[(rnd + 1) * stride]), and with stride 2 those
// before start[rnd * stride + 1] come first. A warp takes its tasks in order,
// so it only moves on, and reads the table only when its round changes.
struct RoundCursor {
  const int64_t *start;
  int stride, rnd = 0;
  int64_t lo, mid, hi;  // round rnd's tasks
  __device__ RoundCursor(const int64_t *task_starts, int task_stride)
      : start(task_starts),
        stride(task_stride),
        lo(task_starts[0]),
        mid(task_starts[task_stride - 1]),
        hi(task_starts[task_stride]) {}
  __device__ __forceinline__ void seek(int64_t task) {
    while (task >= hi) {
      ++rnd;
      lo = hi;
      mid = start[(int64_t)rnd * stride + stride - 1];
      hi = start[(int64_t)(rnd + 1) * stride];
    }
  }
};

// The forward's tasks of rows, round after round: round rnd's slice of the
// local tasks (slice rnd of max_rounds; all of them with one round), then its
// remote tasks (fwd_task_starts, absent on one node). A warp takes its tasks
// in order, so it only moves on.
struct FwdCursor {
  const int64_t *task_starts;  // nullptr: no remote tasks
  int64_t local;
  int max_rounds, rnd = 0;
  int64_t local_lo;     // round rnd's first local task
  int64_t lo, mid, hi;  // its local slice's slots [lo, mid), its remote tasks' [mid, hi)
  __device__ FwdCursor(const int64_t *remote_tasks, int64_t local_tasks, int most_rounds)
      : task_starts(remote_tasks), local(local_tasks), max_rounds(most_rounds) {
    set();
  }
  __device__ __forceinline__ int64_t before(int rnd) const {
    return task_starts ? task_starts[rnd] : 0;
  }
  __device__ __forceinline__ void set() {
    local_lo = rnd * local / max_rounds;
    lo = local_lo + before(rnd);
    mid = lo + (rnd + 1) * local / max_rounds - local_lo;
    hi = mid + before(rnd + 1) - before(rnd);
  }
  __device__ __forceinline__ void seek(int64_t task) {
    while (task >= hi && rnd + 1 < max_rounds) {
      ++rnd;
      set();
    }
  }
  // A task of the local slice: its index among the local tasks.
  __device__ __forceinline__ int64_t local_index(int64_t task) const {
    return local_lo + task - lo;
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
