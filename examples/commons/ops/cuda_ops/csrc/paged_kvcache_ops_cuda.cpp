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
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#ifdef WITH_PYBIND11
#include <torch/extension.h>
#endif
#include <torch/library.h>

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

template <typename DType, typename IdType>
cudaError_t GatherPagedKVCacheAllLayers(DType* gather_kv,
                                        IdType* page_ids,
                                        uint32_t num_layers,
                                        uint32_t stride_gather,
                                        uint32_t stride_layer,
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

cudaError_t GetPagedBatchIndicesPositions(
  int32_t batch_size,
  int32_t* append_indptr,
  int32_t* seq_lens_ptr,
  int32_t* batch_indices_ptr,
  int32_t* positions_ptr,
  cudaStream_t stream
);

namespace {

int resolve_num_sms(const at::Device& device) {
  const char* env_value = std::getenv("PAGED_KVCACHE_NUM_SMS");
  if (env_value != nullptr && std::string(env_value).size() > 0) {
    return std::stoi(env_value);
  }

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  int device_index = device.index();
  if (device_index < 0) {
    auto status = cudaGetDevice(&device_index);
    TORCH_CHECK(status == cudaSuccess,
                "cudaGetDevice failed with error: ",
                cudaGetErrorString(status));
  }

  cudaDeviceProp props;
  auto status = cudaGetDeviceProperties(&props, device_index);
  TORCH_CHECK(status == cudaSuccess,
              "cudaGetDeviceProperties failed with error: ",
              cudaGetErrorString(status));
  return props.multiProcessorCount;
}

void check_cuda_status(cudaError_t status, const char* message) {
  TORCH_CHECK(status == cudaSuccess, message, ": ", cudaGetErrorString(status));
}

} // namespace

at::Tensor append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                                 at::Tensor positions, at::Tensor seqlen_offsets, 
                                 at::Tensor nnz_cuda,
                                 at::Tensor kv_cache_table,
                                 at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                                 int64_t kv_layout) {
  TORCH_CHECK(nnz_cuda.is_cuda(), "nnz_cuda must be a CUDA tensor");
  TORCH_CHECK(nnz_cuda.scalar_type() == at::ScalarType::Int,
              "nnz_cuda must have dtype int32, got ", nnz_cuda.scalar_type());
  TORCH_CHECK(nnz_cuda.numel() == 1, "nnz_cuda must contain exactly one element");
  auto paged_k_cache = kv_cache_table.select(1, 0);
  auto paged_v_cache = kv_cache_table.select(1, 1);
  const auto nnz = static_cast<unsigned int>(nnz_cuda.item<int32_t>());
  if (nnz == 0) {
    return kv_cache_table;
  }
  // TORCH_CHECK(nnz == append_key.size(0) - seqlen_offsets[-1].item<int32_t>(),  // skipped this check due to the D2H transfer overhead
  //             "nnz must be equal to the number of tokens in append_key excluding skipped tokens summed as seqlen_offsets[-1]"
  //             "got nnz=", nnz, " append_key.size(0)=", append_key.size(0), " seqlen_offsets[-1]=", seqlen_offsets[-1].item<int32_t>());

  auto device = append_key.device();
  const auto num_sms = resolve_num_sms(device);

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
  // check_cuda_status(cudaGetLastError(), "AppendPagedKVCache found a pre-existing CUDA error before launch");

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
                           append_v_stride_n, append_v_stride_h, num_sms, stream);
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
                           append_v_stride_n, append_v_stride_h, num_sms, stream);
        break;
    default:
        TORCH_CHECK(false, "AppendPagedKVCache failed to dispatch with dtype ", kv_scalar_dtype);
  }
  check_cuda_status(status, "AppendPagedKVCache launch failed");

  return kv_cache_table;
}

void gather_paged_kv_cache(at::Tensor gather_kv_gpu_buffer,
                           at::Tensor paged_kv_cache,
                           at::Tensor page_ids_to_offload,
                           unsigned int num_pages,
                           int64_t kv_layout,
                           const int num_sms) {
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
            num_pages * page_size, num_sms, stream);
        break;
    case at::ScalarType::Half:
        status = GatherPagedKVCache(
            static_cast<nv_half*>(gather_kv_gpu_buffer.data_ptr()),
            static_cast<int32_t*>(page_ids_to_offload.data_ptr()),
            num_heads, head_dim, page_size, 
            stride_page, stride_k2v, stride_n, stride_h,
            static_cast<nv_half*>(paged_kv_cache.data_ptr()),
            num_pages * page_size, num_sms, stream);
        break;
    default:
        TORCH_CHECK(false, "GatherPagedKVCache failed to dispatch with dtype ", kv_scalar_dtype);
  }
  TORCH_CHECK(status == cudaSuccess,
              "GatherPagedKVCache failed with error: ", cudaGetErrorString(status));
}

void gather_paged_kv_cache_all_layers(uint16_t *gather_kv_gpu_buffer,
                                      uint16_t *paged_kv_cache,
                                      int *page_ids_to_offload,
                                      uint32_t num_layers,
                                      uint32_t stride_gather,
                                      uint32_t stride_layer,
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
  status = GatherPagedKVCacheAllLayers(
      reinterpret_cast<nv_bfloat16*>(gather_kv_gpu_buffer),
      static_cast<int32_t*>(page_ids_to_offload),
      num_layers, stride_gather, stride_layer, 
      num_heads, head_dim, page_size, 
      stride_page, stride_k2v, stride_n, stride_h,
      reinterpret_cast<nv_bfloat16*>(paged_kv_cache),
      num_pages * page_size, num_sms, stream);
  TORCH_CHECK(status == cudaSuccess,
              "GatherPagedKVCacheAllLayers failed with error: ", cudaGetErrorString(status));
}

#ifdef WITH_PYBIND11

PYBIND11_MODULE(paged_kvcache_ops, m) {
  m.def("append_kvcache", &append_paged_kv_cache, "append paged kv cache on GPU", py::call_guard<py::gil_scoped_release>());
  m.def("gather_kvcache", &gather_paged_kv_cache, "gather paged kv cache on GPU", py::call_guard<py::gil_scoped_release>());
}
#endif

TORCH_LIBRARY_FRAGMENT(paged_kvcache_ops, m) {
  m.def("append_kvcache(Tensor append_key, Tensor append_value, Tensor batch_indices, Tensor positions, Tensor seqlen_offsets, Tensor nnz_cuda, Tensor kv_cache_table, Tensor kv_indices, Tensor kv_indptr, Tensor kv_last_page_len, int kv_layout) -> Tensor");
}

TORCH_LIBRARY_IMPL(paged_kvcache_ops, CUDA, m) {
  m.impl("append_kvcache", &append_paged_kv_cache);
}