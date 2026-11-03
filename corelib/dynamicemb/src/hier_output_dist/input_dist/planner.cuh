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
// What the planner's kernels share (planner.h's steps, in the .cu files of
// this folder): how they are launched, searches in sorted tables, the starts
// of the runs of a sorted table, and a CTA's scans and reductions.
//
// A step's kernels are of two kinds. A kernel over items (keys, rows,
// occurrences) runs a thread per item in a grid-stride loop (item_grid).
// A kernel over a step's small tables (sizes of the nodes, ranks, features
// and rounds) runs on one CTA of kThreads threads (kOneCta), which loops over
// them and syncs with __syncthreads; its scans are cub::BlockScan, a tile at
// a time. No kernel waits for another CTA: a step that needs every item done
// goes on in its next kernel.
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <cub/cub.cuh>
#include <cuda/functional>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"

namespace dyn_emb {
namespace hier_output_dist {

// ── launching ──

// The CTAs of a kernel over `items` items on the current GPU.
inline int item_grid(int64_t items) {
  return grid_for(items, at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
}
// A kernel over small tables: one CTA of kThreads.
constexpr int kOneCta = 1;
inline cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

// A thread's items in a kernel over items: first_item(), then every
// item_stride() after it.
__device__ __forceinline__ int64_t first_item() {
  return blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
}
__device__ __forceinline__ int64_t item_stride() { return (int64_t)gridDim.x * blockDim.x; }

// A tensor's data, or nullptr when it is undefined.
template <typename T>
T *data_or_null(const at::Tensor &tensor) {
  return tensor.defined() ? tensor.data_ptr<T>() : nullptr;
}

// ── searches in a sorted table ──

// In table[0, n), sorted ascending: the number of entries below `value`
// (std::lower_bound's index), and the number of entries at most `value`
// (std::upper_bound's).
template <typename T, typename V>
__device__ __forceinline__ int64_t lower_bound(const T *table, int64_t n, V value) {
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (table[mid] < value)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}
template <typename T, typename V>
__device__ __forceinline__ int64_t upper_bound(const T *table, int64_t n, V value) {
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (table[mid] <= value)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

// ── the starts of sorted runs ──

// The starts of the runs of `sorted` (n values in [0, buckets], ascending),
// a thread per bucket in [0, buckets]: start[bucket] is the first position
// whose value is at least `bucket` (n when none is), by a search. Every
// stride-th start also goes to coarse[bucket / stride] (coarse nullptr:
// none).
template <typename Value, typename Start>
__global__ void __launch_bounds__(kThreads)
    run_starts_kernel(const Value *sorted, int64_t n, int64_t buckets, Start *start, int64_t stride,
                      Start *coarse) {
  for (int64_t bucket = first_item(); bucket <= buckets; bucket += item_stride()) {
    const Start position = (Start)lower_bound(sorted, n, bucket);
    start[bucket] = position;
    if (coarse != nullptr && bucket % stride == 0) coarse[bucket / stride] = position;
  }
}
// start: buckets + 1 entries; coarse: buckets / stride + 1, or undefined.
template <typename Value, typename Start>
void run_starts(const at::Tensor &sorted, int64_t buckets, const at::Tensor &start,
                int64_t stride = 1, const at::Tensor &coarse = at::Tensor()) {
  const int64_t n = sorted.numel();
  run_starts_kernel<<<item_grid(buckets + 1), kThreads, 0, current_stream()>>>(
      sorted.data_ptr<Value>(), n, buckets, start.data_ptr<Start>(), stride,
      data_or_null<Start>(coarse));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ── a CTA's scans and reductions (kernels of one CTA of kThreads) ──

// Every thread of the CTA: ends[index] = value(0) + ... + value(index - 1)
// for every index in [0, n], n + 1 entries, a tile of kThreads values at a
// time; value(index) is called once for each index below n. Returns the
// total to every thread.
template <typename T, typename Value>
__device__ T cta_exclusive_ends(int64_t n, Value value, T *ends) {
  using Scan = cub::BlockScan<T, kThreads>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ T carry;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (int64_t first = 0; first < n; first += kThreads) {
    const int64_t index = first + threadIdx.x;
    T before, total;
    Scan(scan).ExclusiveSum(index < n ? (T)value(index) : (T)0, before, total);
    if (index < n) ends[index] = carry + before;
    __syncthreads();
    if (threadIdx.x == 0) carry += total;
    __syncthreads();
  }
  const T total = carry;
  if (threadIdx.x == 0) ends[n] = total;
  __syncthreads();
  return total;
}

// Every thread of the CTA: the largest of value(index) for index in [0, n)
// (`least` when n is 0), to every thread.
template <typename T, typename Value>
__device__ T cta_max(int64_t n, Value value, T least) {
  using Reduce = cub::BlockReduce<T, kThreads>;
  __shared__ typename Reduce::TempStorage reduce;
  __shared__ T result;
  T mine = least;
  for (int64_t index = threadIdx.x; index < n; index += kThreads) mine = max(mine, (T)value(index));
  const T most = Reduce(reduce).Reduce(mine, cuda::maximum<>{});
  if (threadIdx.x == 0) result = most;
  __syncthreads();
  const T out = result;
  __syncthreads();
  return out;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
