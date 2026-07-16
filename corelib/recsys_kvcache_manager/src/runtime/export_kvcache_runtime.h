#pragma once

#include <memory>
#include "gpu_kvcache_manager_impl.h"
#include "flexkv_cpp_client.h"
#include "interface_kvcache_runtime.h"

namespace kvcache_manager {

class ExportKVCacheRuntime : public IKVCacheRuntime {
public:
    ExportKVCacheRuntime();
    virtual ~ExportKVCacheRuntime();

    std::vector<at::Tensor> lookup_kvcache(
        at::Tensor user_ids,
        at::Tensor seqlens
    ) override;
    // std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> 
    std::vector<at::Tensor> allocate_kvcache(
        at::Tensor user_ids,
        at::Tensor seqlens,
        at::Tensor merged_cached_lengths,
        at::Tensor host_cached_lengths
    ) override;
    std::vector<at::Tensor> onboard_kvcache_launch(
        at::Tensor user_ids,
        at::Tensor seqlens,
        at::Tensor merged_cached_lengths,
        at::Tensor host_cached_lengths,
        at::Tensor gpu_cached_startpos,
        at::Tensor gpu_cached_lengths,
        at::Tensor& task_ids,  // from `get_match` in lookup and reuse in onboarding only, returned as masked
        at::Tensor kv_page_indices,
        at::Tensor kv_page_indptr
    ) override;
    void onboard_kvcache_wait(at::Tensor user_ids) override;
    at::Tensor offload_kvcache_launch(
        at::Tensor user_ids,
        at::Tensor seqlens,
        at::Tensor merged_cached_lengths,
        at::Tensor host_cached_lengths,
        at::Tensor gpu_cached_startpos,
        at::Tensor gpu_cached_lengths,
        at::Tensor kv_page_indices,
        at::Tensor kv_page_indptr,
        std::vector<at::Tensor> slot_mappings
    ) override;
    at::Tensor offload_kvcache_reap_completed() override;
    std::vector<at::Tensor> get_kvcache_tables() override;
    int64_t get_num_offload_tasks() override;

public:
    void evict_kvcache(at::Tensor user_ids, bool evict_gpu_only) override;

private:
    std::unique_ptr<kvcache::GPUKVCacheManagerImpl> gpu_kvcache_;
    std::unique_ptr<FlexKVCppClient> flexkv_client_;
    std::unique_ptr<FlexKVGPURegistrator> flexkv_gpu_registrator_;

    std::list<std::tuple<int64_t, int64_t, int32_t, int32_t>> offload_tasks_;

private:
    at::Tensor lookup_token_index_;  // in recsys only namespace matters, so we can pre-allocate a tensor for the token index in get_match.
    at::Tensor token_slot_mappings_in_page_;  // reuse for slot mapping building in onboard_kvcache_launch
    at::Tensor cache_tensor_;
    std::vector<at::Tensor> cache_tables_;
    int num_layers_;
};

} // namespace kvcache_manager

