#pragma once

#include <memory>

namespace kvcache_manager {

class IKVCacheRuntime {
public:
    IKVCacheRuntime() = default;
    virtual ~IKVCacheRuntime() = default;

    virtual std::vector<at::Tensor> lookup_kvcache(
        at::Tensor user_ids,
        at::Tensor seqlens
    ) = 0;
    // virtual std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> 
    virtual std::vector<at::Tensor> allocate_kvcache(
        at::Tensor user_ids,
        at::Tensor seqlens,
        at::Tensor merged_cached_lengths,
        at::Tensor host_cached_lengths
    ) = 0;
    virtual std::vector<at::Tensor> onboard_kvcache_launch(
        at::Tensor user_ids,
        at::Tensor seqlens,
        at::Tensor merged_cached_lengths,
        at::Tensor host_cached_lengths,
        at::Tensor gpu_cached_startpos,
        at::Tensor gpu_cached_lengths,
        at::Tensor& task_ids,  // from `get_match` in lookup and reuse in onboarding only, returned as masked
        at::Tensor kv_page_indices,
        at::Tensor kv_page_indptr
    ) = 0;
    virtual void onboard_kvcache_wait(at::Tensor user_ids) = 0;
    virtual at::Tensor offload_kvcache_launch(
        at::Tensor user_ids,
        at::Tensor seqlens,
        at::Tensor merged_cached_lengths,
        at::Tensor host_cached_lengths,
        at::Tensor gpu_cached_startpos,
        at::Tensor gpu_cached_lengths,
        at::Tensor kv_page_indices,
        at::Tensor kv_page_indptr,
        std::vector<at::Tensor> slot_mappings
    ) = 0;
    virtual at::Tensor offload_kvcache_reap_completed() = 0;
    virtual std::vector<at::Tensor> get_kvcache_tables() = 0;

public:
    virtual void evict_kvcache(at::Tensor user_ids, bool evict_gpu_only) = 0;
};

} // namespace kvcache_manager

