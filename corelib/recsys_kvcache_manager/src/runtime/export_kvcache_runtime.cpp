#include "export_kvcache_runtime.h"

#include <ATen/ATen.h>
#include <utility>

using namespace kvcache;

namespace kvcache_manager {

ExportKVCacheRuntime::ExportKVCacheRuntime() {
    std::cout << "[DEV] enter ExportKVCacheRuntime" << std::endl;
    gpu_kvcache_ = std::make_unique<GPUKVCacheManagerImpl>(
        /*num_layers=*/2,
        /*num_kv_heads=*/32,
        /*kv_headdim=*/128,
        /*num_tokens_per_page=*/32,
        /*num_tokens_per_chunk=*/1024,
        /*num_primary_cache_pages=*/10240,
        /*num_buffer_pages=*/1024,
        /*max_batch_size=*/8,
        /*max_sequence_length=*/4096,
        /*device_idx=*/0);

    auto server_recv_port = std::string(std::getenv("SERVER_RECV_PORT"));
    auto gpu_register_port = std::string(std::getenv("GPU_REGISTER_PORT"));
    std::cout << "[DEV] server_recv_port: " << server_recv_port << "\n\n" << std::endl;
    std::cout << "[DEV] gpu_register_port: " << gpu_register_port << "\n\n" << std::endl;
    flexkv_client_ = std::make_unique<FlexKVCppClient>(server_recv_port, /*dpClientId=*/0, /*tpSize=*/1);

    flexkv_client_->start_server_and_register();

    flexkv_gpu_registrator_ = std::make_unique<FlexKVGPURegistrator>(gpu_register_port, /*dpClientId=*/0, /*device_id=*/0);
    FlexKVCacheLayoutSpec layout{
        FlexKVLayoutType::LayerFirst,
        1, 1024, 32, 4, 128, 
        false,
    };
    at::Tensor cache_tensor = at::empty({1, 1024, 2, 32, 4, 128}, at::dtype(torch::kFloat16).device(at::kCUDA));
    flexkv_gpu_registrator_->register_to_server({cache_tensor}, layout);

    auto start_time = std::chrono::steady_clock::now();
    while (!flexkv_client_->is_ready()) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > std::chrono::seconds(30)) {
            throw std::runtime_error("FlexKV client failed to become ready within 30 seconds");
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    const bool is_ready = flexkv_client_->is_ready();

    auto max_seqlen = 4096;
    lookup_token_index_ = at::arange(0, max_seqlen, at::TensorOptions().dtype(torch::kInt64).device(at::kCPU));
    token_slot_mappings_in_page_ = at::arange(0, gpu_kvcache_->num_tokens_per_page, at::TensorOptions().dtype(torch::kInt64).device(at::kCPU));
}

ExportKVCacheRuntime::~ExportKVCacheRuntime() {
    flexkv_client_->shutdown();
}

std::vector<at::Tensor> ExportKVCacheRuntime::lookup_kvcache(
    at::Tensor user_ids, at::Tensor seqlens) {

    auto gpu_lookup_result = gpu_kvcache_->lookup(user_ids);
    auto gpu_cached_startpos = gpu_lookup_result[0];
    auto gpu_cached_lengths = gpu_lookup_result[1];

    const auto batch_size = user_ids.size(0);
    auto task_ids = at::empty({batch_size}, at::dtype(torch::kInt64).device(at::kCPU));
    auto host_cached_startpos = at::zeros_like(gpu_cached_startpos, at::dtype(torch::kInt32).device(at::kCPU));
    auto host_cached_lengths = at::empty_like(gpu_cached_lengths, at::dtype(torch::kInt32).device(at::kCPU));

    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids.data_ptr<int64_t>()[seq_idx];
        int64_t seqlen = seqlens.data_ptr<int64_t>()[seq_idx];

        auto user_namespace = "uid:" + std::to_string(uid);
        auto [task_id, cache_result] = flexkv_client_->get_match(lookup_token_index_.slice(0, 0, seqlen), std::nullopt, 1, {user_namespace});
        task_ids.data_ptr<int64_t>()[seq_idx] = task_id;

        // int cached_startpos = 0;
        int cached_length = cache_result.sum().item<int>();
        host_cached_lengths.data_ptr<int>()[seq_idx] = cached_length;

        // printf("User %lld: cached_startpos=%d, cached_length=%d\n", uid, cached_startpos, cached_length);
    }

    auto merge_cached_startpos = at::where(host_cached_lengths > 0, host_cached_startpos, gpu_cached_startpos);
    auto merge_cached_lengths = at::where(gpu_cached_lengths > 0, 
        at::max(gpu_cached_startpos, host_cached_startpos) + gpu_cached_lengths - merge_cached_startpos,
        host_cached_lengths);

    return {
        merge_cached_startpos,
        merge_cached_lengths,
        gpu_cached_startpos,
        gpu_cached_lengths,
        host_cached_startpos,
        host_cached_lengths,
        task_ids,
    };
}

// std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> 
std::vector<at::Tensor>
ExportKVCacheRuntime::allocate_kvcache(
    at::Tensor user_ids,
    at::Tensor seqlens,
    at::Tensor merged_cached_lengths,
    at::Tensor host_cached_lengths) {

    const auto batch_size = user_ids.size(0);
    int num_total_pages = at::ceil(seqlens / this->gpu_kvcache_->num_tokens_per_page).to(torch::kInt32).sum().item<int>();
    int num_new_tokens = (seqlens - merged_cached_lengths).sum().item<int>();
    
    at::Tensor page_ids_gpu_buffer = at::empty({num_total_pages}, at::dtype(torch::kInt32).device(at::kCUDA));
    at::Tensor metadata_gpu_buffer = at::empty({5 * batch_size + 4 + num_new_tokens * 2}, at::dtype(torch::kInt32).device(at::kCUDA));

    gpu_kvcache_->allocate(
        user_ids, 
        seqlens, 
        host_cached_lengths,
        page_ids_gpu_buffer,
        metadata_gpu_buffer
    );
    
    // auto new_history_nnz = metadata_gpu_buffer.slice(0, batch_size * 4 + 2, 1).cpu();
    auto new_history_nnz = at::tensor({num_new_tokens}, at::dtype(torch::kInt32).device(at::kCPU));

    // return std::make_tuple(
    return {
            page_ids_gpu_buffer,
            metadata_gpu_buffer,
        // },
        // {
            metadata_gpu_buffer.narrow(0, 0, batch_size + 1),  // page_indptr
            metadata_gpu_buffer.narrow(0, batch_size + 1, batch_size),  // last_page_lens
            metadata_gpu_buffer.narrow(0, batch_size * 2 + 1, batch_size),  // total_history_lengths
            metadata_gpu_buffer.narrow(0, batch_size * 3 + 1, batch_size + 1),  // total_history_offsets
            metadata_gpu_buffer.narrow(0, batch_size * 4 + 3, batch_size + 1),  // new_history_offsets
            metadata_gpu_buffer.narrow(0, batch_size * 5 + 4, num_new_tokens),  // batch_indices
            metadata_gpu_buffer.narrow(0, batch_size * 5 + 4 + num_new_tokens, num_new_tokens),  // positions

            new_history_nnz,
            metadata_gpu_buffer.narrow(0, batch_size * 4 + 2, 1),  // new_history_nnz_cuda

            at::empty({batch_size}, at::dtype(torch::kInt32).device(at::kCPU)),  // kv_seqlens
            at::empty({batch_size + 1}, at::dtype(torch::kInt32).device(at::kCPU)),  // kv_seqlen_offsets
        };
    // );
}

std::vector<at::Tensor> ExportKVCacheRuntime::onboard_kvcache_launch(
    at::Tensor user_ids,
    at::Tensor seqlens,
    at::Tensor merged_cached_lengths,
    at::Tensor host_cached_lengths,
    at::Tensor gpu_cached_startpos,
    at::Tensor gpu_cached_lengths,
    at::Tensor& task_ids,  // from `get_match` in lookup and reuse in onboarding only, returned as masked
    at::Tensor kv_page_indices,
    at::Tensor kv_page_indptr
) {
    const auto batch_size = user_ids.size(0);

    at::Tensor kv_page_indices_cpu = kv_page_indices.cpu();
    std::vector<at::Tensor> slot_mappings;
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        auto page_ids = kv_page_indices_cpu.slice(0, kv_page_indptr[seq_idx].item<int>(), kv_page_indptr[seq_idx + 1].item<int>());

        if (page_ids.numel() > 0) {
            auto slot_mapping = page_ids.unsqueeze(1) * this->gpu_kvcache_->num_tokens_per_page + token_slot_mappings_in_page_.unsqueeze(0);
            slot_mappings.push_back(slot_mapping.reshape(-1));
        } else {
            slot_mappings.push_back(at::empty({0}, at::dtype(torch::kInt64).device(at::kCPU)));
        }
    }

    std::vector<int64_t> task_ids_tolaunch;
    std::vector<at::Tensor> slot_mappings_tolaunch;
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        if (merged_cached_lengths.data_ptr<int>()[seq_idx] == 0 ||
            host_cached_lengths.data_ptr<int>()[seq_idx] == 0      ) {
            task_ids.data_ptr<int64_t>()[seq_idx] = -1;  // -1 indicates no cache to onboard for this sequence
            continue;
        };  // skip sequences without cache to onboard

        bool tolaunch = false;
        tolaunch = tolaunch || (host_cached_lengths.data_ptr<int>()[seq_idx] > gpu_cached_startpos.data_ptr<int>()[seq_idx] + gpu_cached_lengths.data_ptr<int>()[seq_idx]);
        tolaunch = tolaunch || (gpu_cached_startpos.data_ptr<int>()[seq_idx] > 0);

        if (tolaunch) {
            task_ids_tolaunch.push_back(task_ids.data_ptr<int64_t>()[seq_idx]);
            slot_mappings_tolaunch.push_back(slot_mappings[seq_idx]);
        } else {
            task_ids.data_ptr<int64_t>()[seq_idx] = -1;  // -1 indicates no cache to onboard for this sequence
        }
    }

    flexkv_client_->launch_tasks(task_ids_tolaunch, slot_mappings_tolaunch, true);

    return slot_mappings;
}

void ExportKVCacheRuntime::onboard_kvcache_wait(
    at::Tensor task_ids  // masked task ids from onboarding launch
) {
    const auto batch_size = task_ids.size(0);

    std::vector<int64_t> task_ids_launched;
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        if (task_ids.data_ptr<int64_t>()[seq_idx] == -1) continue;  // skip sequences without cache to onboard for this sequence
        task_ids_launched.push_back(task_ids.data_ptr<int64_t>()[seq_idx]);
    }
    if (task_ids_launched.empty()) return;  // no onboard task launched, skip waiting

    auto wait_results = flexkv_client_->wait(task_ids_launched, 1000.f, true);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        if (task_ids.data_ptr<int64_t>()[seq_idx] == -1) continue;  // skip sequences without cache to onboard for this sequence
        int64_t task_id = task_ids.data_ptr<int64_t>()[seq_idx];
        auto it = wait_results.find(task_id);
        if (it == wait_results.end()) {
            throw std::runtime_error("Did not receive result for onboard task_id " + std::to_string(task_id));
        }
        auto& result = it->second;
        if (result.status != KVResponseStatus::SUCCESS) {
            throw std::runtime_error("Onboard task_id " + std::to_string(task_id) + " failed with status: " + std::to_string(static_cast<int>(result.status)));
        }
    }

    // skipped returning status for now
}

void ExportKVCacheRuntime::offload_kvcache_launch(
    at::Tensor user_ids,
    at::Tensor seqlens,
    at::Tensor merged_cached_lengths,
    at::Tensor host_cached_lengths,
    at::Tensor gpu_cached_startpos,
    at::Tensor gpu_cached_lengths,
    at::Tensor& task_ids,  // from `get_match` in lookup and reused in offloading only, returned as masked
    std::vector<at::Tensor>& slot_mappings
) {
    auto zero_tensor = at::zeros_like(host_cached_lengths, at::dtype(torch::kInt32).device(at::kCPU));
    auto [offload_user_ids, offload_start_indices, offload_page_indices_list] = gpu_kvcache_->acquire_offload_pages(user_ids, zero_tensor, true);

    auto batch_size = user_ids.size(0);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t seqlen = seqlens.data_ptr<int64_t>()[seq_idx];
        int64_t valid_len = int64_t(seqlen / this->gpu_kvcache_->num_tokens_per_page) * this->gpu_kvcache_->num_tokens_per_page; // align to page boundary

        int64_t uid = user_ids.data_ptr<int64_t>()[seq_idx];
        auto user_namespace = "uid:" + std::to_string(uid);
        int64_t task_id = flexkv_client_->put_async(
            lookup_token_index_.slice(0, 0, valid_len),
            slot_mappings[seq_idx].slice(0, 0, valid_len),
            std::nullopt,
            {user_namespace}
        );
        offload_tasks_.push_back(
            std::make_tuple(
                task_id,
                user_ids.data_ptr<int64_t>()[seq_idx],
                offload_start_indices.data_ptr<int>()[seq_idx],
                valid_len
            )
        );
    }
}

void ExportKVCacheRuntime::offload_kvcache_reap_completed() {
    std::list<std::tuple<int64_t, int64_t, int32_t, int32_t>> remaining_tasks;
    for (auto& task : offload_tasks_) {
        auto [task_id, user_id, offload_start_index, offload_length] = task;
        auto wait_results = flexkv_client_->try_wait({task_id});
        if (wait_results.find(task_id) == wait_results.end()) {
            remaining_tasks.push_back(task);  // not completed yet, keep waiting
            continue;
        }
        if (wait_results[task_id].status == KVResponseStatus::UNREADY) {
            remaining_tasks.push_back(task);  // not completed yet, keep waiting
            continue;  // completed successfully, can be reaped
        }

        int offload_succeeded = (wait_results[task_id].status == KVResponseStatus::SUCCESS);
        if (wait_results[task_id].status != KVResponseStatus::SUCCESS) {
            // throw std::runtime_error("Offload task_id " + std::to_string(task_id) + " failed with status: " + std::to_string(static_cast<int>(wait_results[task_id].status)));
        }

        gpu_kvcache_->release_offload_pages(
            at::from_blob(&user_id, {1}, at::dtype(torch::kInt64).device(at::kCPU)),
            at::from_blob(&offload_start_index, {1}, at::dtype(torch::kInt32).device(at::kCPU)),
            at::from_blob(&offload_length, {1}, at::dtype(torch::kInt32).device(at::kCPU)),
            { offload_succeeded, }
        );
    }
    offload_tasks_ = std::move(remaining_tasks);
}

} // namespace kvcache_manager

