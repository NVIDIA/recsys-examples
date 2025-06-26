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

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor seqlen_offsets, 
                           at::Tensor nnz_cuda, unsigned int nnz,
                           at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           int64_t kv_layout) {
  unsigned int batch_size = kv_last_page_len.size(0);
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

  auto stride_page = num_heads * page_size * head_dim;
  auto stride_n = (kv_layout == 1) ? head_dim : num_heads * head_dim;
  auto stride_h = (kv_layout == 1) ? page_size * head_dim : head_dim;

  // get kv_cache_strides
  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();


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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &append_paged_kv_cache, "append paged kv cache on GPU");
}