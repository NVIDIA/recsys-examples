/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
#
# Implementation based on FlashInfer library.
# 
******************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
// #include <torch/extension.h>
#include <torch/serialize/tensor.h>


template <typename DType, typename IdType>
cudaError_t ScatterPagedKVCache(DType* continuous_kv,
                               IdType* page_ids,
                               uint32_t num_heads,
                               uint32_t head_dim,
                               uint32_t page_size,
                               uint32_t stride_page,
                               uint32_t stride_k2v,
                               uint32_t stride_n,
                               uint32_t stride_h,
                               DType* kv_cache,
                               uint32_t nnz,
                               int num_sms,
                               cudaStream_t stream);

template <typename DType, typename IdType>
cudaError_t GatherPagedKVCache(DType* gather_kv,
                               IdType* page_ids,
                               uint32_t num_heads,
                               uint32_t head_dim,
                               uint32_t page_size,
                               uint32_t stride_page,
                               uint32_t stride_k2v,
                               uint32_t stride_n,
                               uint32_t stride_h,
                               DType* kv_cache,
                               uint32_t nnz,
                               int num_sms,
                               cudaStream_t stream);


void scatter_paged_kvcache(
    uint16_t *paged_kvcache_table,    // output
    uint16_t *continuous_gpu_buffer,  // input: gpu onload buffer
    int *page_ids,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t page_size,
    uint32_t stride_page,
    uint32_t stride_k2v,
    uint32_t stride_n,
    uint32_t stride_h,
    uint32_t num_pages,
    const int num_sms,
    cudaStream_t stream) {

  cudaError_t status;
  status = ScatterPagedKVCache(
      reinterpret_cast<nv_bfloat16*>(continuous_gpu_buffer),
      static_cast<int32_t*>(page_ids),
      num_heads, head_dim, page_size, 
      stride_page, stride_k2v, stride_n, stride_h,
      reinterpret_cast<nv_bfloat16*>(paged_kvcache_table),
      num_pages * page_size, num_sms, stream);
  TORCH_CHECK(status == cudaSuccess,
              "ScatterPagedKVCache failed with error: ", cudaGetErrorString(status));
}

void gather_paged_kvcache(
    uint16_t *gather_gpu_buffer,    // output: gpu offload buffer
    uint16_t *paged_kvcache_table,  // input
    int *page_ids,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t page_size,
    uint32_t stride_page,
    uint32_t stride_k2v,
    uint32_t stride_n,
    uint32_t stride_h,
    uint32_t num_pages,
    const int num_sms,
    cudaStream_t stream) {

  cudaError_t status;
  status = GatherPagedKVCache(
      reinterpret_cast<nv_bfloat16*>(gather_gpu_buffer),
      static_cast<int32_t*>(page_ids),
      num_heads, head_dim, page_size, 
      stride_page, stride_k2v, stride_n, stride_h,
      reinterpret_cast<nv_bfloat16*>(paged_kvcache_table),
      num_pages * page_size, num_sms, stream);
  TORCH_CHECK(status == cudaSuccess,
              "GatherPagedKVCache failed with error: ", cudaGetErrorString(status));
}
