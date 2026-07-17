#include "export_kvcache_runtime.h"
#include "kvcache_runtime_config.h"

#include <ATen/ATen.h>
#include <iostream>
#include <string>
#include <thread>

using namespace kvcache;

namespace kvcache_manager {

namespace {

#ifndef KVCACHE_MANAGER_RUNTIME_DEBUG
#define KVCACHE_MANAGER_RUNTIME_DEBUG 0
#endif

void log_runtime_message(const std::string& message) {
#if KVCACHE_MANAGER_RUNTIME_DEBUG
    std::cout << "[KVCACHE][runtime] " << message << std::endl;
#else
    (void)message;
#endif
}

} // namespace

ExportKVCacheRuntime::ExportKVCacheRuntime() {
    log_runtime_message(
        "thread=" + c10::str(std::this_thread::get_id()) + " constructor enter"
    );
    const auto config_path = find_config_path();
    const auto config = load_yaml_config(config_path);
    const int num_layers = get_config_int(config, "num_layers");
    const int num_kv_heads = get_config_int(config, "num_kv_heads");
    const int head_size = get_config_int(config, "head_size");
    const int tokens_per_page = get_config_int(config, "tokens_per_page");
    const int tokens_per_chunk = get_config_int(config, "tokens_per_chunk");
    const int num_primary_cache_pages = get_config_int(config, "num_primary_cache_pages");
    const int num_buffer_pages = get_config_int(config, "num_buffer_pages");
    const int max_batch_size = get_config_int(config, "max_batch_size");
    const int max_sequence_length = get_config_int(config, "max_sequence_length");
    const int device_idx = get_config_int(config, "device_idx");
    const at::ScalarType cache_dtype = get_config_dtype(config);
    num_layers_ = num_layers;

    log_runtime_message(
        "config_path=" + config_path.string()
        + " layers=" + std::to_string(num_layers)
        + " kv_heads=" + std::to_string(num_kv_heads)
        + " head_size=" + std::to_string(head_size)
        + " page=" + std::to_string(tokens_per_page)
        + " chunk=" + std::to_string(tokens_per_chunk)
        + " primary_pages=" + std::to_string(num_primary_cache_pages)
        + " buffer_pages=" + std::to_string(num_buffer_pages)
        + " max_bs=" + std::to_string(max_batch_size)
        + " max_seq=" + std::to_string(max_sequence_length)
        + " device=" + std::to_string(device_idx)
        + " dtype=" + c10::str(cache_dtype)
    );

    log_runtime_message("creating GPUKVCacheManagerImpl");
    gpu_kvcache_ = std::make_unique<GPUKVCacheManagerImpl>(
        /*num_layers=*/num_layers,
        /*num_kv_heads=*/num_kv_heads,
        /*kv_headdim=*/head_size,
        /*num_tokens_per_page=*/tokens_per_page,
        /*num_tokens_per_chunk=*/tokens_per_chunk,
        /*num_primary_cache_pages=*/num_primary_cache_pages,
        /*num_buffer_pages=*/num_buffer_pages,
        /*max_batch_size=*/max_batch_size,
        /*max_sequence_length=*/max_sequence_length,
        /*device_idx=*/device_idx);

    auto server_recv_port = get_required_config_string(config, "server_recv_port");
    auto gpu_register_port = get_required_config_string(config, "gpu_register_port");
    log_runtime_message(
        "creating FlexKVCppClient server_recv_port=" + server_recv_port
    );
    flexkv_client_ = std::make_unique<FlexKVCppClient>(server_recv_port, /*dpClientId=*/0, /*tpSize=*/1);

    log_runtime_message("start_server_and_register begin");
    flexkv_client_->start_server_and_register();
    log_runtime_message("start_server_and_register done");

    log_runtime_message(
        "creating GPU registrator port=" + gpu_register_port
    );
    flexkv_gpu_registrator_ = std::make_unique<FlexKVGPURegistrator>(gpu_register_port, /*dpClientId=*/0, /*device_id=*/0);
    FlexKVCacheLayoutSpec layout{
        FlexKVLayoutType::LayerFirst,
        num_layers, num_primary_cache_pages, tokens_per_page, num_kv_heads, head_size,
        false,
    };
    cache_tensor_ = at::empty(
        {num_layers, num_primary_cache_pages, 2, tokens_per_page, num_kv_heads, head_size},
        at::TensorOptions().dtype(cache_dtype).device(at::Device(at::kCUDA, device_idx)));
    cache_tables_ = cache_tensor_.unbind(0);
    log_runtime_message(
        "register_to_server begin cache_tensor=" + c10::str(cache_tensor_.sizes())
    );
    flexkv_gpu_registrator_->register_to_server({cache_tensor_}, layout);
    log_runtime_message("register_to_server done");

    auto start_time = std::chrono::steady_clock::now();
    while (!flexkv_client_->is_ready()) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > std::chrono::seconds(30)) {
            throw std::runtime_error("FlexKV client failed to become ready within 30 seconds");
        }
        log_runtime_message(
            "waiting for FlexKV client ready elapsed_ms="
            + std::to_string(
                std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
            )
        );
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    log_runtime_message(
        "FlexKV client ready=" + std::to_string(flexkv_client_->is_ready())
    );

    lookup_token_index_ = at::arange(0, max_sequence_length, at::TensorOptions().dtype(torch::kInt64).device(at::kCPU));
    token_slot_mappings_in_page_ = at::arange(0, gpu_kvcache_->num_tokens_per_page, at::TensorOptions().dtype(torch::kInt64).device(at::kCPU));
    log_runtime_message("constructor exit");
}

ExportKVCacheRuntime::~ExportKVCacheRuntime() {
    log_runtime_message("destructor enter");
    if (flexkv_client_) {
        flexkv_client_->shutdown();
    }
    log_runtime_message("destructor exit");
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
        auto [task_id, cache_result] = flexkv_client_->get_match(
            lookup_token_index_.slice(0, 0, seqlen),
            std::nullopt,
            num_layers_,
            {user_namespace});
        task_ids.data_ptr<int64_t>()[seq_idx] = task_id;

        int cached_length = cache_result.sum().item<int>();
        host_cached_lengths.data_ptr<int>()[seq_idx] = cached_length;
    }

    auto merge_cached_startpos = at::where(host_cached_lengths > 0, host_cached_startpos, gpu_cached_startpos);
    auto merge_cached_lengths = at::where(gpu_cached_lengths > 0, 
        at::max(gpu_cached_startpos, host_cached_startpos) + gpu_cached_lengths - merge_cached_startpos,
        host_cached_lengths);
    auto non_valid_caching = (gpu_cached_startpos > host_cached_lengths) & (gpu_cached_lengths > 0);
    non_valid_caching = non_valid_caching | (merge_cached_startpos > 0);
    merge_cached_lengths = at::where(non_valid_caching, 0, merge_cached_lengths);
    merge_cached_startpos = at::where(non_valid_caching, 0, merge_cached_startpos);

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
    
    auto new_history_nnz = at::tensor({num_new_tokens}, at::dtype(torch::kInt32).device(at::kCPU));

    return {
            page_ids_gpu_buffer,
            metadata_gpu_buffer,
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
        auto seqlen = seqlens.data_ptr<int64_t>()[seq_idx];
        auto page_ids = kv_page_indices_cpu.slice(0, kv_page_indptr[seq_idx].item<int>(), kv_page_indptr[seq_idx + 1].item<int>());

        if (page_ids.numel() > 0) {
            auto slot_mapping = page_ids.unsqueeze(1) * this->gpu_kvcache_->num_tokens_per_page + token_slot_mappings_in_page_.unsqueeze(0);
            slot_mappings.push_back(slot_mapping.reshape(-1).slice(0, 0, seqlen));
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

    if (!task_ids_tolaunch.empty()) {
        flexkv_client_->launch_tasks(task_ids_tolaunch, slot_mappings_tolaunch, true);
    }

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
    if (task_ids_launched.empty()) {
        return;  // no onboard task launched, skip waiting
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

at::Tensor ExportKVCacheRuntime::offload_kvcache_launch(
    at::Tensor user_ids,
    at::Tensor seqlens,
    at::Tensor merged_cached_lengths,
    at::Tensor host_cached_lengths,
    at::Tensor gpu_cached_startpos,
    at::Tensor gpu_cached_lengths,
    at::Tensor kv_page_indices,
    at::Tensor kv_page_indptr,
    std::vector<at::Tensor> slot_mappings
) {
    auto [offload_user_ids, offload_start_indices, offload_page_indices_list] = gpu_kvcache_->acquire_offload_pages(user_ids, host_cached_lengths, true);

    std::vector<at::Tensor> new_slot_mappings;
    if (slot_mappings.size() == 1 && slot_mappings[0].data_ptr<int64_t>()[0] == -1) {
        for (int seq_idx = 0; seq_idx < user_ids.size(0); seq_idx++) {
            auto seqlen = seqlens.data_ptr<int64_t>()[seq_idx];
            auto page_ids = kv_page_indices.cpu().slice(0, kv_page_indptr[seq_idx].item<int>(), kv_page_indptr[seq_idx + 1].item<int>());
            if (page_ids.numel() > 0) {
                auto slot_mapping = page_ids.unsqueeze(1) * this->gpu_kvcache_->num_tokens_per_page + token_slot_mappings_in_page_.unsqueeze(0);
                new_slot_mappings.push_back(slot_mapping.reshape(-1).slice(0, 0, seqlen));
            } else {
                new_slot_mappings.push_back(at::empty({0}, at::dtype(torch::kInt64).device(at::kCPU)));
            }
        }
    } else if (slot_mappings.size() != user_ids.size(0)) {
        throw std::runtime_error("slot_mappings size does not match batch size");
    } else {
        new_slot_mappings = slot_mappings;
    }

    auto batch_size = user_ids.size(0);
    at::Tensor offload_task_ids = at::empty({batch_size}, at::dtype(torch::kInt64).device(at::kCPU));
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids.data_ptr<int64_t>()[seq_idx];
        int64_t seqlen = seqlens.data_ptr<int64_t>()[seq_idx];
        int64_t valid_len = int64_t(seqlen / this->gpu_kvcache_->num_tokens_per_page) * this->gpu_kvcache_->num_tokens_per_page; // align to page boundary
        if (valid_len == 0) {
            offload_task_ids.data_ptr<int64_t>()[seq_idx] = -1;  // -1 indicates no cache to offload for this sequence
            this->gpu_kvcache_->release_offload_pages(
                at::from_blob(&uid, {1}, at::dtype(torch::kInt64).device(at::kCPU)),
                at::from_blob(&offload_start_indices.data_ptr<int>()[seq_idx], {1}, at::dtype(torch::kInt32).device(at::kCPU)),
                at::from_blob(&valid_len, {1}, at::dtype(torch::kInt32).device(at::kCPU)),
                { true, }
            );
            continue;
        }  // skip sequences without cache to offload for this sequence
        auto user_namespace = "uid:" + std::to_string(uid);
        int64_t task_id = flexkv_client_->put_async(
            lookup_token_index_.slice(0, 0, valid_len),
            new_slot_mappings[seq_idx].slice(0, 0, valid_len),
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
        offload_task_ids.data_ptr<int64_t>()[seq_idx] = task_id;
    }
    return offload_task_ids;
}

at::Tensor ExportKVCacheRuntime::offload_kvcache_reap_completed() {
    std::vector<int64_t> completed_tasks;
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
        completed_tasks.push_back(user_id);
        completed_tasks.push_back(task_id);
        completed_tasks.push_back(static_cast<int64_t>(wait_results[task_id].status));

        gpu_kvcache_->release_offload_pages(
            at::from_blob(&user_id, {1}, at::dtype(torch::kInt64).device(at::kCPU)),
            at::from_blob(&offload_start_index, {1}, at::dtype(torch::kInt32).device(at::kCPU)),
            at::from_blob(&offload_length, {1}, at::dtype(torch::kInt32).device(at::kCPU)),
            { offload_succeeded, }
        );
    }
    offload_tasks_ = std::move(remaining_tasks);
    if (completed_tasks.empty()) {
        return at::empty({0, 3}, at::dtype(torch::kInt64).device(at::kCPU));  // return empty tensor with shape (0, 3) if no completed task
    }
    return at::from_blob(completed_tasks.data(), {(int64_t)completed_tasks.size()}, at::dtype(torch::kInt64).device(at::kCPU)).clone().reshape({-1, 3});  // reshape to Nx3 for easier parsing (user_id, task_id, status)
}

std::vector<at::Tensor> ExportKVCacheRuntime::get_kvcache_tables() {
    return cache_tables_;
}

int64_t ExportKVCacheRuntime::get_num_offload_tasks() {
    return offload_tasks_.size();
}

void ExportKVCacheRuntime::evict_kvcache(at::Tensor user_ids, bool evict_gpu_only) {
    const auto batch_size = user_ids.size(0);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids.data_ptr<int64_t>()[seq_idx];
        gpu_kvcache_->evict(uid);
            
        if (!evict_gpu_only) {
        }
    }
}

} // namespace kvcache_manager

