/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// #include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

template <typename DType, typename IdType>
cudaError_t AppendPagedKVCache(DType* k_data,
                               DType* v_data,
                               IdType* indices,
                               IdType* indptr,
                               uint32_t num_heads,
                               uint32_t head_dim,
                               uint32_t page_size,
                               uint32_t stride_page,
                               uint32_t stride_n,
                               uint32_t stride_h,
                               DType* append_key, DType* append_value, IdType* batch_indices, 
                               IdType* positions, IdType* offsets, 
                               IdType* nnz_cuda, uint32_t nnz, 
                               size_t append_k_stride_n, size_t append_k_stride_h,
                               size_t append_v_stride_n, size_t append_v_stride_h,
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
                               cudaStream_t stream);

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor seqlen_offsets, 
                           at::Tensor nnz_cuda, unsigned int nnz,
                           at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           int64_t kv_layout) {
  // unsigned int batch_size = kv_last_page_len.size(0);
  auto device = append_key.device();

  unsigned int num_heads, page_size, head_dim;
  head_dim = paged_k_cache.size(3);
  if (kv_layout == 1) {
    num_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_heads = paged_k_cache.size(2);
  }

  auto stride_page = paged_k_cache.stride(0);
  auto stride_n = (kv_layout == 1) ? head_dim : num_heads * head_dim;
  auto stride_h = (kv_layout == 1) ? page_size * head_dim : head_dim;

  // get kv_cache_strides
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");


  auto append_k_strides = append_key.strides();
  auto append_k_stride_n = append_k_strides[0];
  auto append_k_stride_h = append_k_strides[1];
  auto append_v_strides = append_value.strides();
  auto append_v_stride_n = append_v_strides[0];
  auto append_v_stride_h = append_v_strides[1];

  auto kv_scalar_dtype = paged_k_cache.scalar_type();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaError_t status;
  switch (kv_scalar_dtype) {
    case at::ScalarType::BFloat16:
        status =
        AppendPagedKVCache(static_cast<nv_bfloat16*>(paged_k_cache.data_ptr()),
                           static_cast<nv_bfloat16*>(paged_v_cache.data_ptr()),
                           static_cast<int32_t*>(kv_indices.data_ptr()),
                           static_cast<int32_t*>(kv_indptr.data_ptr()),
                           num_heads, head_dim, page_size, stride_page, stride_n, stride_h,
                           static_cast<nv_bfloat16*>(append_key.data_ptr()),
                           static_cast<nv_bfloat16*>(append_value.data_ptr()),
                           static_cast<int32_t*>(batch_indices.data_ptr()),
                           static_cast<int32_t*>(positions.data_ptr()), 
                           static_cast<int32_t*>(seqlen_offsets.data_ptr()), 
                           static_cast<int32_t*>(nnz_cuda.data_ptr()), 
                           nnz, append_k_stride_n, append_k_stride_h, 
                           append_v_stride_n, append_v_stride_h, stream);
        break;
    case at::ScalarType::Half:
        status =
        AppendPagedKVCache(static_cast<nv_half*>(paged_k_cache.data_ptr()), 
                           static_cast<nv_half*>(paged_v_cache.data_ptr()),
                           static_cast<int32_t*>(kv_indices.data_ptr()),
                           static_cast<int32_t*>(kv_indptr.data_ptr()),
                           num_heads, head_dim, page_size, stride_page, stride_n, stride_h,
                           static_cast<nv_half*>(append_key.data_ptr()),
                           static_cast<nv_half*>(append_value.data_ptr()),
                           static_cast<int32_t*>(batch_indices.data_ptr()),
                           static_cast<int32_t*>(positions.data_ptr()), 
                           static_cast<int32_t*>(seqlen_offsets.data_ptr()), 
                           static_cast<int32_t*>(nnz_cuda.data_ptr()), 
                           nnz, append_k_stride_n, append_k_stride_h, 
                           append_v_stride_n, append_v_stride_h, stream);
        break;
    default:
        TORCH_CHECK(false, "AppendPagedKVCache failed to dispatch with dtype ", kv_scalar_dtype);
  }
  TORCH_CHECK(status == cudaSuccess,
              "AppendPagedKVCache failed with error: ", cudaGetErrorString(status));
}

void gather_paged_kv_cache(at::Tensor gather_kv_gpu_buffer,
                           at::Tensor paged_kv_cache,
                           at::Tensor page_ids_to_offload,
                           unsigned int num_pages,
                           int64_t kv_layout) {
  auto device = paged_kv_cache.device();

  TORCH_CHECK(paged_kv_cache.ndimension() == 5, 
              "kv cache table must has 5 dimensions (num_pages, 2, page_size, num_head, head_dim).");
  
  unsigned int num_heads, page_size, head_dim;
  head_dim = paged_kv_cache.size(4);
  if (kv_layout == 1) {
    num_heads = paged_kv_cache.size(2);
    page_size = paged_kv_cache.size(3);
  } else {
    page_size = paged_kv_cache.size(2);
    num_heads = paged_kv_cache.size(3);
  }

  auto stride_page = paged_kv_cache.stride(0);
  auto stride_n = (kv_layout == 1) ? head_dim : num_heads * head_dim;
  auto stride_h = (kv_layout == 1) ? page_size * head_dim : head_dim;
  auto stride_k2v = paged_kv_cache.stride(1);

  // check input/output strides
  TORCH_CHECK(paged_kv_cache.strides() == gather_kv_gpu_buffer.strides(), 
              "input/output strides must be identical");
  TORCH_CHECK(paged_kv_cache.is_contiguous() && paged_kv_cache.is_contiguous(), 
              "buffer must be contiguous");
  
  auto kv_scalar_dtype = paged_kv_cache.scalar_type();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaError_t status;
  switch (kv_scalar_dtype) {
    case at::ScalarType::BFloat16:
        status = GatherPagedKVCache(
            static_cast<nv_bfloat16*>(gather_kv_gpu_buffer.data_ptr()),
            static_cast<int32_t*>(page_ids_to_offload.data_ptr()),
            num_heads, head_dim, page_size, 
            stride_page, stride_k2v, stride_n, stride_h,
            static_cast<nv_bfloat16*>(paged_kv_cache.data_ptr()),
            num_pages * page_size, stream);
        break;
    case at::ScalarType::Half:
        status = GatherPagedKVCache(
            static_cast<nv_half*>(gather_kv_gpu_buffer.data_ptr()),
            static_cast<int32_t*>(page_ids_to_offload.data_ptr()),
            num_heads, head_dim, page_size, 
            stride_page, stride_k2v, stride_n, stride_h,
            static_cast<nv_half*>(paged_kv_cache.data_ptr()),
            num_pages * page_size, stream);
        break;
    default:
        TORCH_CHECK(false, "GatherPagedKVCache failed to dispatch with dtype ", kv_scalar_dtype);
  }
  TORCH_CHECK(status == cudaSuccess,
              "GatherPagedKVCache failed with error: ", cudaGetErrorString(status));
}

PYBIND11_MODULE(paged_kvcache_ops, m) {
  m.def("append_kvcache", &append_paged_kv_cache, "append paged kv cache on GPU");
  m.def("gather_kvcache", &gather_paged_kv_cache, "gather paged kv cache on GPU");
}