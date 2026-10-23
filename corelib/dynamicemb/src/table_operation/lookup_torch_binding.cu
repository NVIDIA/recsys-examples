#include <torch/library.h>
#include "kernels.cuh"
#include "table.cuh"

// This translation unit contributes one schema fragment to the shared
// INFERENCE_EMB namespace. Use TORCH_LIBRARY_FRAGMENT so split bindings across
// multiple files remain symmetric and avoid making this file the sole primary
// owner of the namespace definition.
TORCH_LIBRARY_FRAGMENT(INFERENCE_EMB, m) {
    // NOTE: In Torch operator schema, integral scalar arguments are declared as
    // "int" and are represented as int64_t on the C++ kernel side.
    m.def("table_lookup(Tensor table_storage, Tensor table_bucket_offsets, int bucket_capacity, Tensor keys, Tensor table_ids, Tensor? score_input, int policy_type) -> (Tensor, Tensor, Tensor)");
}


// forward declaration
namespace dyn_emb {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
table_lookup(at::Tensor table_storage, at::Tensor table_bucket_offsets,
             int64_t bucket_capacity, at::Tensor keys,
             at::Tensor table_ids,
             std::optional<at::Tensor> score_input,
             ScorePolicyType policy_type,
             int64_t num_scores);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
table_lookup_cuda_impl(at::Tensor table_storage, at::Tensor table_bucket_offsets,
             int64_t bucket_capacity, at::Tensor keys,
             at::Tensor table_ids,
             std::optional<at::Tensor> score_input,
             int64_t policy) {
    TORCH_CHECK(table_storage.is_cuda(), "INFERENCE_EMB::table_lookup expects CUDA table_storage.");
    TORCH_CHECK(table_bucket_offsets.is_cuda(), "INFERENCE_EMB::table_lookup expects CUDA table_bucket_offsets.");
    TORCH_CHECK(keys.is_cuda(), "INFERENCE_EMB::table_lookup expects CUDA keys.");
    TORCH_CHECK(table_ids.is_cuda(), "INFERENCE_EMB::table_lookup expects CUDA table_ids.");
    if (score_input.has_value() && score_input.value().defined()) {
        TORCH_CHECK(score_input.value().is_cuda(), "INFERENCE_EMB::table_lookup expects CUDA score_input when provided.");
    }

    return table_lookup(
        table_storage, table_bucket_offsets, bucket_capacity, keys, table_ids,
        score_input, static_cast<ScorePolicyType>(policy));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
table_lookup_cpu_impl(at::Tensor table_storage, at::Tensor table_bucket_offsets,
             int64_t bucket_capacity, at::Tensor keys,
             at::Tensor table_ids,
             std::optional<at::Tensor> score_input,
             int64_t policy) {
    TORCH_WARN_ONCE(
        "INFERENCE_EMB::table_lookup has no CPU kernel. "
        "Please move inputs to CUDA and load inference_emb_ops.so before calling this operator."
    );
    TORCH_CHECK(
        false,
        "INFERENCE_EMB::table_lookup is CUDA-only. Got CPU dispatch."
    );
    return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
}

} // namespace dyn_emb

TORCH_LIBRARY_IMPL(INFERENCE_EMB, CUDA, m) {
    m.impl("table_lookup", &dyn_emb::table_lookup_cuda_impl);
}

TORCH_LIBRARY_IMPL(INFERENCE_EMB, CPU, m) {
    m.impl("table_lookup", &dyn_emb::table_lookup_cpu_impl);
}
