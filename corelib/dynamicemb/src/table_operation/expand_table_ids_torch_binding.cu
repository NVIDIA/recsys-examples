#include <ATen/ATen.h>
#include <torch/library.h>

#include "unique_op.h"

TORCH_LIBRARY_FRAGMENT(INFERENCE_EMB, m) {
    m.def("expand_table_ids(Tensor offsets, Tensor? table_offsets_in_feature=None, int num_tables=0, int local_batch_size=1, int num_elements=0) -> Tensor");
}

namespace dyn_emb {

at::Tensor expand_table_ids_cuda_impl(
    at::Tensor offsets, c10::optional<at::Tensor> table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size, int64_t num_elements) {
    TORCH_CHECK(offsets.is_cuda(), "INFERENCE_EMB::expand_table_ids expects CUDA offsets.");
    if (table_offsets_in_feature.has_value() &&
        table_offsets_in_feature.value().defined()) {
        TORCH_CHECK(
            table_offsets_in_feature.value().is_cuda(),
            "INFERENCE_EMB::expand_table_ids expects CUDA table_offsets_in_feature when provided."
        );
    }
    return expand_table_ids_cuda(offsets, table_offsets_in_feature, num_tables,
                                 local_batch_size, num_elements);
}

at::Tensor expand_table_ids_cpu_impl(
    at::Tensor offsets, c10::optional<at::Tensor> table_offsets_in_feature,
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
