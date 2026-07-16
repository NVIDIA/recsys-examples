#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <torch/types.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "kvcache_manager_context.h"
#include "interface_kvcache_runtime.h"

TORCH_LIBRARY_FRAGMENT(kvcache_manager_ops, m) {
    m.def("init_kvcache(Tensor ordering_tensor) -> int");
    m.def("shutdown_runtime(Tensor ordering_tensor) -> ()");
    m.def("lookup(Tensor user_ids, Tensor seqlens, Tensor ordering_tensor) -> Tensor[]");
    m.def("allocate(Tensor user_ids, Tensor seqlens, Tensor merged_cached_lengths, Tensor host_cached_lengths) -> Tensor[]");
    m.def("onboard_launch(Tensor user_ids, Tensor seqlens, Tensor[] lookup_results, Tensor kv_page_indices, Tensor kv_page_indptr) -> (Tensor, Tensor, Tensor)");
    m.def("onboard_wait(Tensor task_ids, Tensor? ordering_tensor=None) -> Tensor[]");
    m.def("offload_launch(Tensor user_ids, Tensor seqlens, Tensor merged_cached_lengths, Tensor host_cached_lengths, Tensor gpu_cached_startpos, Tensor gpu_cached_lengths, Tensor kv_page_indices, Tensor kv_page_indptr, Tensor[] slot_mappings, Tensor ordering_tensor) -> Tensor");
    m.def("offload_reap_completed(Tensor ordering_tensor) -> Tensor");
    m.def("get_cache_tables(Tensor ordering_tensor) -> Tensor[]");

    m.def("offload_flush(Tensor ordering_tensor) -> Tensor");
    m.def("evict_kvcache(Tensor user_ids, bool evict_gpu_only, Tensor ordering_tensor) -> Tensor");
}

namespace kvcache_manager {

namespace {

#ifndef KVCACHE_MANAGER_OP_DEBUG
#define KVCACHE_MANAGER_OP_DEBUG 0
#endif

void log_op(const char* name, const char* phase) {
#if KVCACHE_MANAGER_OP_DEBUG
    std::cout << "[KVCACHE][op] thread=" << std::this_thread::get_id()
              << " " << name << " " << phase << std::endl;
#else
    (void)name;
    (void)phase;
#endif
}

void log_tensor(const char* name, const at::Tensor& tensor) {
#if KVCACHE_MANAGER_OP_DEBUG
    std::cout << "[KVCACHE][op]   " << name
              << " sizes=" << tensor.sizes()
              << " dtype=" << tensor.scalar_type()
              << " device=" << tensor.device()
              << std::endl;
#else
    (void)name;
    (void)tensor;
#endif
}

void log_message(const std::string& message) {
#if KVCACHE_MANAGER_OP_DEBUG
    std::cout << "[KVCACHE][op] " << message << std::endl;
#else
    (void)message;
#endif
}

} // namespace

int64_t init_kvcache(at::Tensor ordering_tensor) {
    // Force initialization of the runtime context and its manager before any operator calls.
    log_op("init_kvcache", "enter");
    log_tensor("ordering_tensor", ordering_tensor);
    (void)KVCacheRuntimeContext::instance().manager();
    log_op("init_kvcache", "exit");
    return 0;
}

void shutdown_runtime_impl(at::Tensor ordering_tensor) {
    (void)ordering_tensor;
    log_op("shutdown_runtime", "enter");
    KVCacheRuntimeContext::instance().clear_manager();
    log_op("shutdown_runtime", "exit");
}

std::vector<at::Tensor> lookup_impl(
    at::Tensor user_ids,
    at::Tensor seqlens,
    at::Tensor ordering_tensor
) {
    log_op("lookup", "enter");
    log_tensor("user_ids", user_ids);
    log_tensor("seqlens", seqlens);
    log_tensor("ordering_tensor", ordering_tensor);
    TORCH_CHECK(!user_ids.is_cuda(), "kvcache_manager_ops::lookup expects CPU tensors.");
    TORCH_CHECK(!seqlens.is_cuda(), "kvcache_manager_ops::lookup expects CPU tensors.");
    (void)ordering_tensor;

    auto runtime = KVCacheRuntimeContext::instance().manager();
    auto result = runtime->lookup_kvcache(user_ids, seqlens);
    log_message("lookup exit outputs=" + std::to_string(result.size()));
    return result;
}

// std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
std::vector<at::Tensor>
allocate_impl(at::Tensor user_ids, at::Tensor seqlens, at::Tensor merged_cached_lengths, at::Tensor host_cached_lengths) {
    log_op("allocate", "enter");
    log_tensor("user_ids", user_ids);
    log_tensor("seqlens", seqlens);
    log_tensor("merged_cached_lengths", merged_cached_lengths);
    log_tensor("host_cached_lengths", host_cached_lengths);
    TORCH_CHECK(!user_ids.is_cuda(), "kvcache_manager_ops::allocate expects CPU tensors.");
    TORCH_CHECK(!seqlens.is_cuda(), "kvcache_manager_ops::allocate expects CPU tensors.");
    TORCH_CHECK(!merged_cached_lengths.is_cuda(), "kvcache_manager_ops::allocate expects CPU merged_cached_lengths tensor.");
    TORCH_CHECK(!host_cached_lengths.is_cuda(), "kvcache_manager_ops::allocate expects CPU host_cached_lengths tensor.");

    auto runtime = KVCacheRuntimeContext::instance().manager();
    auto result = runtime->allocate_kvcache(user_ids, seqlens, merged_cached_lengths, host_cached_lengths);
    log_message("allocate exit outputs=" + std::to_string(result.size()));
    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> onboard_launch_impl(
    at::Tensor user_ids,
    at::Tensor seqlens,
    std::vector<at::Tensor> lookup_results,
    at::Tensor kv_page_indices,
    at::Tensor kv_page_indptr
) {
    log_op("onboard_launch", "enter");
    log_tensor("user_ids", user_ids);
    log_tensor("seqlens", seqlens);
    log_tensor("kv_page_indices", kv_page_indices);
    log_tensor("kv_page_indptr", kv_page_indptr);
    TORCH_CHECK(!user_ids.is_cuda(), "kvcache_manager_ops::onboard_launch expects CPU tensors.");
    TORCH_CHECK(!seqlens.is_cuda(), "kvcache_manager_ops::onboard_launch expects CPU tensors.");
    TORCH_CHECK(lookup_results.size() == 7, "kvcache_manager_ops::onboard_launch expects 7 lookup result tensors.");
    at::Tensor task_ids = lookup_results[6];
    TORCH_CHECK(!task_ids.is_cuda(), "kvcache_manager_ops::onboard_launch expects CPU task_ids tensor.");

    auto runtime = KVCacheRuntimeContext::instance().manager();
    std::vector<at::Tensor> slot_mappings = runtime->onboard_kvcache_launch(
        user_ids, seqlens, 
        lookup_results[1], lookup_results[5], lookup_results[2], lookup_results[3],
        task_ids,
        kv_page_indices, kv_page_indptr);
    
    auto concat_slot_mappings = at::cat(slot_mappings, 0);
    std::vector<int64_t> slot_mapping_offsets;
    slot_mapping_offsets.push_back(0);
    for (const auto &slot_mapping : slot_mappings) {
        slot_mapping_offsets.push_back(slot_mapping_offsets.back() + slot_mapping.size(0));
    }

    log_message(
        "onboard_launch exit slot_mappings=" + std::to_string(slot_mappings.size())
        + " task_ids_sizes=" + c10::str(task_ids.sizes())
    );
    return std::make_tuple(
        concat_slot_mappings,
        at::tensor(slot_mapping_offsets, at::dtype(torch::kInt64)),
        task_ids
    );
}

std::vector<at::Tensor> onboard_wait_impl(
    at::Tensor task_ids,
    c10::optional<at::Tensor> ordering_tensor
) {
    log_op("onboard_wait", "enter");
    log_tensor("task_ids", task_ids);
    // if (ordering_tensor.has_value()) {
    //     log_tensor("ordering_tensor", ordering_tensor.value());
    // }
    TORCH_CHECK(!task_ids.is_cuda(), "kvcache_manager_ops::onboard_wait expects CPU tensors.");
    (void)ordering_tensor;

    auto runtime = KVCacheRuntimeContext::instance().manager();
    runtime->onboard_kvcache_wait(task_ids);
    auto result = runtime->get_kvcache_tables();
    log_message("onboard_wait exit cache_tables=" + std::to_string(result.size()));
    return result;
}

at::Tensor offload_launch_impl(
    at::Tensor user_ids,
    at::Tensor seqlens,
    at::Tensor merged_cached_lengths,
    at::Tensor host_cached_lengths,
    at::Tensor gpu_cached_startpos,
    at::Tensor gpu_cached_lengths,
    at::Tensor kv_page_indices,
    at::Tensor kv_page_indptr,
    std::vector<at::Tensor> slot_mappings,
    at::Tensor ordering_tensor
) {
    log_op("offload_launch", "enter");
    log_tensor("user_ids", user_ids);
    log_tensor("seqlens", seqlens);
    log_tensor("merged_cached_lengths", merged_cached_lengths);
    log_tensor("host_cached_lengths", host_cached_lengths);
    log_tensor("gpu_cached_startpos", gpu_cached_startpos);
    log_tensor("gpu_cached_lengths", gpu_cached_lengths);
    log_tensor("kv_page_indices", kv_page_indices);
    log_tensor("kv_page_indptr", kv_page_indptr);
    TORCH_CHECK(!user_ids.is_cuda(), "kvcache_manager_ops::offload_launch expects CPU tensors.");
    TORCH_CHECK(!seqlens.is_cuda(), "kvcache_manager_ops::offload_launch expects CPU tensors.");
    (void)ordering_tensor;

    auto runtime = KVCacheRuntimeContext::instance().manager();
    auto result = runtime->offload_kvcache_launch(
        user_ids, 
        seqlens, 
        merged_cached_lengths, 
        host_cached_lengths, 
        gpu_cached_startpos, 
        gpu_cached_lengths, 
        kv_page_indices,
        kv_page_indptr,
        slot_mappings);
    log_tensor("offload_task_ids", result);
    log_op("offload_launch", "exit");
    return result;
}

at::Tensor offload_reap_completed_impl(at::Tensor ordering_tensor) {
    log_op("offload_reap_completed", "enter");
    log_tensor("ordering_tensor", ordering_tensor);
    (void)ordering_tensor;
    auto runtime = KVCacheRuntimeContext::instance().manager();
    auto result = runtime->offload_kvcache_reap_completed();
    log_tensor("reap_result", result);
    log_op("offload_reap_completed", "exit");
    return result;
}

std::vector<at::Tensor> get_cache_tables_impl(at::Tensor ordering_tensor) {
    log_op("get_cache_tables", "enter");
    (void)ordering_tensor;
    auto runtime = KVCacheRuntimeContext::instance().manager();
    auto result = runtime->get_kvcache_tables();
    log_message("get_cache_tables exit cache_tables=" + std::to_string(result.size()));
    return result;
}


at::Tensor offload_flush_impl(at::Tensor ordering_tensor) {
    log_op("offload_flush", "enter");
    auto runtime = KVCacheRuntimeContext::instance().manager();
    at::Tensor reap_result;
    while (runtime->get_num_offload_tasks() > 0) {
        reap_result = runtime->offload_kvcache_reap_completed();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // sleep for a while before checking again
        log_tensor("offload_flush_result", reap_result);
    }
    log_op("offload_flush", "exit");
    return reap_result;
}

at::Tensor evict_kvcache_impl(
    at::Tensor user_ids,
    bool evict_gpu_only,
    at::Tensor ordering_tensor
) {
    log_op("evict_kvcache", "enter");
    log_tensor("user_ids", user_ids);
    TORCH_CHECK(!user_ids.is_cuda(), "kvcache_manager_ops::evict_kvcache expects CPU tensors.");
    (void)ordering_tensor;

    auto runtime = KVCacheRuntimeContext::instance().manager();
    runtime->evict_kvcache(user_ids, evict_gpu_only);
    log_op("evict_kvcache", "exit");
    return at::empty({1}, at::dtype(torch::kInt32));  // return empty tensor as placeholder
}


} // namespace kvcache_manager

TORCH_LIBRARY_IMPL(kvcache_manager_ops, CUDA, m) {
    m.impl("init_kvcache", &kvcache_manager::init_kvcache);
    m.impl("shutdown_runtime", &kvcache_manager::shutdown_runtime_impl);
    m.impl("lookup", &kvcache_manager::lookup_impl);
    m.impl("allocate", &kvcache_manager::allocate_impl);
    m.impl("onboard_launch", &kvcache_manager::onboard_launch_impl);
    m.impl("onboard_wait", &kvcache_manager::onboard_wait_impl);
    m.impl("offload_launch", &kvcache_manager::offload_launch_impl);
    m.impl("offload_reap_completed", &kvcache_manager::offload_reap_completed_impl);
    m.impl("get_cache_tables", &kvcache_manager::get_cache_tables_impl);

    m.impl("offload_flush", &kvcache_manager::offload_flush_impl);
    m.impl("evict_kvcache", &kvcache_manager::evict_kvcache_impl);
}

TORCH_LIBRARY_IMPL(kvcache_manager_ops, CPU, m) {
    m.impl("init_kvcache", &kvcache_manager::init_kvcache);
    m.impl("shutdown_runtime", &kvcache_manager::shutdown_runtime_impl);
    m.impl("lookup", &kvcache_manager::lookup_impl);
    m.impl("allocate", &kvcache_manager::allocate_impl);
    m.impl("onboard_launch", &kvcache_manager::onboard_launch_impl);
    m.impl("onboard_wait", &kvcache_manager::onboard_wait_impl);
    m.impl("offload_launch", &kvcache_manager::offload_launch_impl);
    m.impl("offload_reap_completed", &kvcache_manager::offload_reap_completed_impl);
    m.impl("get_cache_tables", &kvcache_manager::get_cache_tables_impl);

    m.impl("offload_flush", &kvcache_manager::offload_flush_impl);
    m.impl("evict_kvcache", &kvcache_manager::evict_kvcache_impl);
}
