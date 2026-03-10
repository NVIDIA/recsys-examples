#include <ATen/ATen.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(INFERENCE_EMB, m) {
    m.def("get_table_range(Tensor offsets, Tensor feature_offsets) -> Tensor");
}

namespace dyn_emb {

at::Tensor get_table_range(at::Tensor offsets, at::Tensor feature_offsets);

at::Tensor get_table_range_cuda_impl(at::Tensor offsets,
                                     at::Tensor feature_offsets) {
    TORCH_CHECK(offsets.is_cuda(), "INFERENCE_EMB::get_table_range expects CUDA offsets.");
    TORCH_CHECK(feature_offsets.is_cuda(), "INFERENCE_EMB::get_table_range expects CUDA feature_offsets.");
    return get_table_range(offsets, feature_offsets);
}

at::Tensor get_table_range_cpu_impl(at::Tensor offsets,
                                    at::Tensor feature_offsets) {
    TORCH_WARN_ONCE(
        "INFERENCE_EMB::get_table_range has no CPU kernel. "
        "Please move inputs to CUDA and load inference_emb_ops.so before calling this operator."
    );
    TORCH_CHECK(
        false,
        "INFERENCE_EMB::get_table_range is CUDA-only. Got CPU dispatch."
    );
    return at::Tensor();
}

} // namespace dyn_emb

TORCH_LIBRARY_IMPL(INFERENCE_EMB, CUDA, m) {
    m.impl("get_table_range", &dyn_emb::get_table_range_cuda_impl);
}

TORCH_LIBRARY_IMPL(INFERENCE_EMB, CPU, m) {
    m.impl("get_table_range", &dyn_emb::get_table_range_cpu_impl);
}
