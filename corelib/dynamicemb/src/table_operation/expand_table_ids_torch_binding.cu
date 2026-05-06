#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include "unique_op.h"

TORCH_LIBRARY_FRAGMENT(INFERENCE_EMB, m) {
    m.def("expand_table_ids(Tensor offsets, Tensor indices, Tensor? table_offsets_in_feature=None, int num_tables=0, int local_batch_size=1, int num_elements=0) -> Tensor");
}

namespace dyn_emb {

// expand_table_ids_cuda (unique_op.cu) now only supports identity mapping
// (local_batch_size=1, no table_offsets_in_feature). The inference op needs the
// full generality (feature_offsets + local_batch_size), so implement the kernel
// here and adapt the body without changing the external interface.
__global__ void expand_table_ids_inference_kernel(
    const int64_t *offsets,
    const int64_t *feature_offsets,  // nullptr = identity mapping
    int64_t *table_ids,
    int num_tables,
    int local_batch_size,
    int64_t num_elements) {
  const int64_t stride = blockDim.x * gridDim.x;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;
       idx += stride) {
    int lo = 0, hi = num_tables;
    while (lo < hi) {
      int mid = (lo + hi + 1) / 2;
      int64_t feat = feature_offsets ? feature_offsets[mid] : mid;
      if (offsets[feat * local_batch_size] <= idx) lo = mid;
      else hi = mid - 1;
    }
    table_ids[idx] = lo;
  }
}

at::Tensor expand_table_ids_cuda_impl(
    at::Tensor offsets, at::Tensor indices, c10::optional<at::Tensor> table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size, int64_t num_elements) {
    TORCH_CHECK(offsets.is_cuda(), "INFERENCE_EMB::expand_table_ids expects CUDA offsets.");
    if (table_offsets_in_feature.has_value() &&
        table_offsets_in_feature.value().defined()) {
        TORCH_CHECK(
            table_offsets_in_feature.value().is_cuda(),
            "INFERENCE_EMB::expand_table_ids expects CUDA table_offsets_in_feature when provided."
        );
    }

    if (num_elements == 0) {
        return at::empty({0}, at::TensorOptions().dtype(at::kLong).device(offsets.device()));
    }

    TORCH_CHECK(local_batch_size > 0,
        "INFERENCE_EMB::expand_table_ids expects local_batch_size > 0");

    const int64_t *feature_offsets_ptr = nullptr;
    if (table_offsets_in_feature.has_value() &&
        table_offsets_in_feature.value().defined() &&
        table_offsets_in_feature.value().numel() > 0) {
        feature_offsets_ptr = table_offsets_in_feature.value().data_ptr<int64_t>();
        // num_tables kept as caller-supplied value, matching original behavior
    } else {
        num_tables = (offsets.size(0) - 1) / local_batch_size;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int BLOCK_SIZE = 64;
    constexpr int BLOCKS_PER_SM = 4;
    const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int grid_size = static_cast<int>(
        std::min((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 static_cast<int64_t>(sm_count * BLOCKS_PER_SM)));

    at::Tensor table_ids = at::empty(
        {num_elements}, at::TensorOptions().dtype(at::kLong).device(offsets.device()));

    expand_table_ids_inference_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        offsets.data_ptr<int64_t>(), feature_offsets_ptr,
        table_ids.data_ptr<int64_t>(), num_tables, local_batch_size, num_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return table_ids;
}

at::Tensor expand_table_ids_cpu_impl(
    at::Tensor offsets, at::Tensor indices, c10::optional<at::Tensor> table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size, int64_t num_elements) {
    TORCH_WARN_ONCE(
        "INFERENCE_EMB::expand_table_ids has no CPU kernel. "
        "Please move inputs to CUDA and load inference_emb_ops.so before calling this operator."
    );
    TORCH_CHECK(
        false,
        "INFERENCE_EMB::expand_table_ids is CUDA-only. Got CPU dispatch."
    );
    return at::Tensor();
}

} // namespace dyn_emb

TORCH_LIBRARY_IMPL(INFERENCE_EMB, CUDA, m) {
    m.impl("expand_table_ids", &dyn_emb::expand_table_ids_cuda_impl);
}

TORCH_LIBRARY_IMPL(INFERENCE_EMB, CPU, m) {
    m.impl("expand_table_ids", &dyn_emb::expand_table_ids_cpu_impl);
}
