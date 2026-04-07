/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#include "../check.h"
#include "table.cuh"
#include "types.cuh"

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace dyn_emb {

// Max num_tables for which we use shared-memory path (2 * num_tables * 8
// bytes). 1024 tables -> 16KB shared, well under 48KB limit.
constexpr int64_t kMaxNumTablesShm = 1024;

__global__ void no_eviction_assign_scores_kernel_global(
    int64_t *__restrict__ no_eviction_next_index_dev,
    int64_t const *__restrict__ table_ids, uint64_t *__restrict__ scores,
    int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t tid = table_ids[i];
  int64_t old_val = atomicAdd(no_eviction_next_index_dev + tid, 1);
  scores[i] = static_cast<uint64_t>(old_val);
}

__global__ void no_eviction_assign_scores_kernel_shm(
    int64_t *__restrict__ no_eviction_next_index_dev,
    int64_t const *__restrict__ table_ids, uint64_t *__restrict__ scores,
    int64_t n, int64_t num_tables) {
  extern __shared__ char shm_raw[];
  int64_t *hist = reinterpret_cast<int64_t *>(shm_raw);
  int64_t *base =
      reinterpret_cast<int64_t *>(shm_raw + num_tables * sizeof(int64_t));

  // Zero histogram (cooperative)
  for (int64_t t = threadIdx.x; t < num_tables; t += blockDim.x) {
    hist[t] = 0;
  }
  __syncthreads();

  // Block-local histogram
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t tid = table_ids[i];
    atomicAdd(hist + tid, 1);
  }
  __syncthreads();

  // Reserve range per table from global counter; store base in shared
  for (int64_t t = threadIdx.x; t < num_tables; t += blockDim.x) {
    if (hist[t] > 0) {
      base[t] = atomicAdd(no_eviction_next_index_dev + t, hist[t]);
    }
  }
  __syncthreads();

  // Assign scores: each thread uses local offset within block's reserved range
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t tid = table_ids[i];
    int64_t local_off = atomicAdd(hist + tid, 1);
    scores[i] = static_cast<uint64_t>(base[tid] + local_off);
  }
}

at::Tensor no_eviction_assign_scores(at::Tensor no_eviction_next_index_dev,
                                     at::Tensor table_ids) {
  TORCH_CHECK(no_eviction_next_index_dev.is_cuda());
  TORCH_CHECK(table_ids.is_cuda());
  TORCH_CHECK(no_eviction_next_index_dev.scalar_type() == at::kLong);
  TORCH_CHECK(table_ids.scalar_type() == at::kLong);
  int64_t n = table_ids.numel();
  int64_t num_tables = no_eviction_next_index_dev.numel();
  at::Tensor scores =
      torch::empty({n}, table_ids.options().dtype(torch::kUInt64));
  if (n == 0)
    return scores;
  constexpr int BLOCK = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // if (num_tables > 0 && num_tables <= kMaxNumTablesShm) {
  //   size_t shm_size = 2 * static_cast<size_t>(num_tables) * sizeof(int64_t);
  //   no_eviction_assign_scores_kernel_shm<<<(n + BLOCK - 1) / BLOCK, BLOCK,
  //                                          shm_size, stream>>>(
  //       no_eviction_next_index_dev.data_ptr<int64_t>(),
  //       table_ids.data_ptr<int64_t>(), scores.data_ptr<uint64_t>(), n,
  //       num_tables);
  // } else {
    no_eviction_assign_scores_kernel_global<<<(n + BLOCK - 1) / BLOCK, BLOCK, 0,
                                              stream>>>(
        no_eviction_next_index_dev.data_ptr<int64_t>(),
        table_ids.data_ptr<int64_t>(), scores.data_ptr<uint64_t>(), n);
  // }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

} // namespace dyn_emb
