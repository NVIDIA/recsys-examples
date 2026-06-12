#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include "kvcache_manager_context.h"
#include "interface_kvcache_runtime.h"

TORCH_LIBRARY_FRAGMENT(kvcache_manager_ops, m) {
    m.def("lookup(Tensor user_ids, Tensor seqlens) -> Tensor[]");
}

namespace kvcache_manager {

std::vector<at::Tensor> lookup_impl(at::Tensor user_ids, at::Tensor seqlens) {
    TORCH_CHECK(!user_ids.is_cuda(), "kvcache_manager_ops::lookup expects CPU tensors.");
    TORCH_CHECK(!seqlens.is_cuda(), "kvcache_manager_ops::lookup expects CPU tensors.");

    auto runtime = KVCacheRuntimeContext::instance().manager();
    return runtime->lookup_kvcache(user_ids, seqlens);
}

// std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
std::vector<at::Tensor>
allocate_impl(at::Tensor user_ids, at::Tensor seqlens, at::Tensor merged_cached_lengths, at::Tensor host_cached_lengths) {
    TORCH_CHECK(!user_ids.is_cuda(), "kvcache_manager_ops::lookup expects CPU tensors.");
    TORCH_CHECK(!seqlens.is_cuda(), "kvcache_manager_ops::lookup expects CPU tensors.");

    auto runtime = KVCacheRuntimeContext::instance().manager();
    return runtime->allocate_kvcache(user_ids, seqlens, merged_cached_lengths, host_cached_lengths);
}

std::vector<at::Tensor> onboard_launch_impl(
    at::Tensor user_ids,
    at::Tensor seqlens,
    std::vector<at::Tensor> lookup_results,
    at::Tensor kv_page_indices,
    at::Tensor kv_page_indptr
) {
    TORCH_CHECK(!user_ids.is_cuda(), "kvcache_manager_ops::onboard_launch expects CPU tensors.");
    TORCH_CHECK(!seqlens.is_cuda(), "kvcache_manager_ops::onboard_launch expects CPU tensors.");

    auto runtime = KVCacheRuntimeContext::instance().manager();
    auto &task_ids = lookup_results[6];
    return runtime->onboard_kvcache_launch(
        user_ids, seqlens, 
        lookup_results[1], lookup_results[5], lookup_results[2], lookup_results[3],
        task_ids,
        kv_page_indices, kv_page_indptr);
}

void onboard_wait_impl(
    at::Tensor task_ids
) {
    TORCH_CHECK(!task_ids.is_cuda(), "kvcache_manager_ops::onboard_wait expects CPU tensors.");

    auto runtime = KVCacheRuntimeContext::instance().manager();
    return runtime->onboard_kvcache_wait(task_ids);
}

// std::vector<at::Tensor> lookup_meta_impl(at::Tensor user_ids, at::Tensor seqlens) {
//     return {
//         at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//         at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//         at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//         at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//         at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//         at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//     };
// }

// std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
// allocate_meta_impl(at::Tensor user_ids, at::Tensor seqlens, at::Tensor merged_cached_lengths, at::Tensor host_cached_lengths) {
//     return std::make_tuple(
//         {
//             at::empty({0}, at::dtype(torch::kInt32).device(at::kCUDA)),
//             at::empty({0}, at::dtype(torch::kInt32).device(at::kCUDA)),
//         },
//         {
//             at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//             at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//             at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//             at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//             at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//             at::empty_like(user_ids, at::TensorOptions().dtype(at::kInt32)),
//         }
//     );
// }

} // namespace kvcache_manager

TORCH_LIBRARY_IMPL(kvcache_manager_ops, CUDA, m) {
    m.impl("lookup", &kvcache_manager::lookup_impl);
    m.impl("allocate", &kvcache_manager::allocate_impl);
    m.impl("onboard_launch", &kvcache_manager::onboard_launch_impl);
    m.impl("onboard_wait", &kvcache_manager::onboard_wait_impl);
//     m.impl("offload_launch", &kvcache_manager::offload_launch_impl);
//     m.impl("offload_reap_completed", &kvcache_manager::offload_reap_completed_impl);
}

TORCH_LIBRARY_IMPL(kvcache_manager_ops, CPU, m) {
    m.impl("lookup", &kvcache_manager::lookup_impl);
    m.impl("allocate", &kvcache_manager::allocate_impl);
    m.impl("onboard_launch", &kvcache_manager::onboard_launch_impl);
    m.impl("onboard_wait", &kvcache_manager::onboard_wait_impl);
    // m.impl("offload_launch", &kvcache_manager::offload_launch_impl);
    // m.impl("offload_reap_completed", &kvcache_manager::offload_reap_completed_impl);
}
