#include "export_kvcache_runtime.h"

#include <ATen/ATen.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

using namespace kvcache;

namespace kvcache_manager {

namespace {

int get_env_int(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || std::string(value).empty()) {
        return default_value;
    }
    return std::stoi(value);
}

std::string get_required_env_string(const char* name) {
    const char* value = std::getenv(name);
    TORCH_CHECK(value != nullptr && !std::string(value).empty(), "Missing required env var: ", name);
    return std::string(value);
}

at::ScalarType get_env_dtype(const char* name, at::ScalarType default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || std::string(value).empty()) {
        return default_value;
    }
    std::string dtype(value);
    if (dtype == "bfloat16" || dtype == "bf16") {
        return at::kBFloat16;
    }
    if (dtype == "float16" || dtype == "fp16" || dtype == "half") {
        return at::kHalf;
    }
    TORCH_CHECK(false, "Unsupported KVCACHE_MANAGER_DTYPE: ", dtype);
}

} // namespace

ExportKVCacheRuntime::ExportKVCacheRuntime() {
    std::cout << "[KVCACHE][runtime] thread=" << std::this_thread::get_id()
              << " constructor enter" << std::endl;
    const int num_layers = get_env_int("KVCACHE_MANAGER_NUM_LAYERS", 2);
    const int num_kv_heads = get_env_int("KVCACHE_MANAGER_NUM_KV_HEADS", 32);
    const int head_size = get_env_int("KVCACHE_MANAGER_HEAD_SIZE", 128);
    const int tokens_per_page = get_env_int("KVCACHE_MANAGER_TOKENS_PER_PAGE", 32);
    const int tokens_per_chunk = get_env_int("KVCACHE_MANAGER_TOKENS_PER_CHUNK", 1024);
    const int num_primary_cache_pages = get_env_int("KVCACHE_MANAGER_NUM_PRIMARY_CACHE_PAGES", 10240);
    const int num_buffer_pages = get_env_int("KVCACHE_MANAGER_NUM_BUFFER_PAGES", 1024);
    const int max_batch_size = get_env_int("KVCACHE_MANAGER_MAX_BATCH_SIZE", 8);
    const int max_sequence_length = get_env_int("KVCACHE_MANAGER_MAX_SEQUENCE_LENGTH", 4096);
    const int device_idx = get_env_int("KVCACHE_MANAGER_DEVICE_IDX", 0);
    const at::ScalarType cache_dtype = get_env_dtype("KVCACHE_MANAGER_DTYPE", at::kHalf);
    num_layers_ = num_layers;

    std::cout << "[KVCACHE][runtime] config layers=" << num_layers
              << " kv_heads=" << num_kv_heads
              << " head_size=" << head_size
              << " page=" << tokens_per_page
              << " chunk=" << tokens_per_chunk
              << " primary_pages=" << num_primary_cache_pages
              << " buffer_pages=" << num_buffer_pages
              << " max_bs=" << max_batch_size
              << " max_seq=" << max_sequence_length
              << " device=" << device_idx
              << " dtype=" << cache_dtype
              << std::endl;

    std::cout << "[KVCACHE][runtime] creating GPUKVCacheManagerImpl" << std::endl;
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

    auto server_recv_port = get_required_env_string("SERVER_RECV_PORT");
    auto gpu_register_port = get_required_env_string("GPU_REGISTER_PORT");
    std::cout << "[KVCACHE][runtime] creating FlexKVCppClient server_recv_port="
              << server_recv_port << std::endl;
    flexkv_client_ = std::make_unique<FlexKVCppClient>(server_recv_port, /*dpClientId=*/0, /*tpSize=*/1);

    std::cout << "[KVCACHE][runtime] start_server_and_register begin" << std::endl;
    flexkv_client_->start_server_and_register();
    std::cout << "[KVCACHE][runtime] start_server_and_register done" << std::endl;

    std::cout << "[KVCACHE][runtime] creating GPU registrator port=" << gpu_register_port << std::endl;
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
    std::cout << "[KVCACHE][runtime] register_to_server begin cache_tensor="
              << cache_tensor_.sizes() << std::endl;
    flexkv_gpu_registrator_->register_to_server({cache_tensor_}, layout);
    std::cout << "[KVCACHE][runtime] register_to_server done" << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    while (!flexkv_client_->is_ready()) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > std::chrono::seconds(30)) {
            throw std::runtime_error("FlexKV client failed to become ready within 30 seconds");
        }
        std::cout << "[KVCACHE][runtime] waiting for FlexKV client ready elapsed_ms="
                  << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                  << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    const bool is_ready = flexkv_client_->is_ready();
    std::cout << "[KVCACHE][runtime] FlexKV client ready=" << is_ready << std::endl;

    lookup_token_index_ = at::arange(0, max_sequence_length, at::TensorOptions().dtype(torch::kInt64).device(at::kCPU));
    token_slot_mappings_in_page_ = at::arange(0, gpu_kvcache_->num_tokens_per_page, at::TensorOptions().dtype(torch::kInt64).device(at::kCPU));
    std::cout << "[KVCACHE][runtime] constructor exit" << std::endl;
}

ExportKVCacheRuntime::~ExportKVCacheRuntime() {
    std::cout << "[KVCACHE][runtime] destructor enter" << std::endl;
    if (flexkv_client_) {
        flexkv_client_->shutdown();
    }
    std::cout << "[KVCACHE][runtime] destructor exit" << std::endl;
}

std::vector<at::Tensor> ExportKVCacheRuntime::lookup_kvcache(
    at::Tensor user_ids, at::Tensor seqlens) {
    std::cout << "[KVCACHE][runtime] lookup_kvcache enter batch=" << user_ids.size(0)
              << std::endl;

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
    std::cout << "[KVCACHE][runtime] lookup seq=" << seq_idx
          << " uid=" << uid
          << " seqlen=" << seqlen
          << " get_match begin" << std::endl;
        auto [task_id, cache_result] = flexkv_client_->get_match(
            lookup_token_index_.slice(0, 0, seqlen),
            std::nullopt,
            num_layers_,
            {user_namespace});
        task_ids.data_ptr<int64_t>()[seq_idx] = task_id;

        // int cached_startpos = 0;
        int cached_length = cache_result.sum().item<int>();
        host_cached_lengths.data_ptr<int>()[seq_idx] = cached_length;
        std::cout << "[KVCACHE][runtime] lookup seq=" << seq_idx
              << " uid=" << uid
              << " task_id=" << task_id
              << " host_cached_length=" << cached_length
              << std::endl;

        // printf("User %lld: cached_startpos=%d, cached_length=%d\n", uid, cached_startpos, cached_length);
    }

    auto merge_cached_startpos = at::where(host_cached_lengths > 0, host_cached_startpos, gpu_cached_startpos);
    auto merge_cached_lengths = at::where(gpu_cached_lengths > 0, 
        at::max(gpu_cached_startpos, host_cached_startpos) + gpu_cached_lengths - merge_cached_startpos,
        host_cached_lengths);

    std::cout << "[KVCACHE][runtime] lookup_kvcache exit" << std::endl;
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
    std::cout << "[KVCACHE][runtime] allocate_kvcache enter batch=" << user_ids.size(0)
              << std::endl;

    const auto batch_size = user_ids.size(0);
    int num_total_pages = at::ceil(seqlens / this->gpu_kvcache_->num_tokens_per_page).to(torch::kInt32).sum().item<int>();
    int num_new_tokens = (seqlens - merged_cached_lengths).sum().item<int>();
    std::cout << "[KVCACHE][runtime] allocate pages=" << num_total_pages
              << " new_tokens=" << num_new_tokens << std::endl;
    
    at::Tensor page_ids_gpu_buffer = at::empty({num_total_pages}, at::dtype(torch::kInt32).device(at::kCUDA));
    at::Tensor metadata_gpu_buffer = at::empty({5 * batch_size + 4 + num_new_tokens * 2}, at::dtype(torch::kInt32).device(at::kCUDA));

    gpu_kvcache_->allocate(
        user_ids, 
        seqlens, 
        host_cached_lengths,
        page_ids_gpu_buffer,
        metadata_gpu_buffer
    );
    std::cout << "[KVCACHE][runtime] allocate_kvcache gpu allocate done" << std::endl;
    
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
            std::cout << "[INFO] Sequence " << seq_idx << " user_id " << user_ids.data_ptr<int64_t>()[seq_idx]
                      << " has no cache to onboard (merged_cached_lengths=" 
                      << merged_cached_lengths.data_ptr<int>()[seq_idx] << ", host_cached_lengths=" 
                      << host_cached_lengths.data_ptr<int>()[seq_idx] << "). Skipping onboarding." << std::endl;
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
        std::cout << "[INFO] Launching onboard tasks for " << task_ids_tolaunch.size() << " sequences." << std::endl;
        flexkv_client_->launch_tasks(task_ids_tolaunch, slot_mappings_tolaunch, true);
    } else {
        std::cout << "[INFO] No onboard tasks to launch." << std::endl;
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
        std::cout << "[INFO] No onboard tasks were launched. Skipping wait." << std::endl;
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
    std::cout << "[KVCACHE][runtime] offload_kvcache_launch enter batch=" << user_ids.size(0)
              << " slot_mappings=" << slot_mappings.size() << std::endl;
    // auto zero_tensor = at::zeros_like(host_cached_lengths, at::dtype(torch::kInt32).device(at::kCPU));
    auto [offload_user_ids, offload_start_indices, offload_page_indices_list] = gpu_kvcache_->acquire_offload_pages(user_ids, host_cached_lengths, true);
    std::cout << "[KVCACHE][runtime] acquire_offload_pages done page_lists="
              << offload_page_indices_list.size() << std::endl;

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
        int64_t seqlen = seqlens.data_ptr<int64_t>()[seq_idx];
        int64_t valid_len = int64_t(seqlen / this->gpu_kvcache_->num_tokens_per_page) * this->gpu_kvcache_->num_tokens_per_page; // align to page boundary
        std::cout << "[DEBUG] Offloading user_id: " << user_ids.data_ptr<int64_t>()[seq_idx] << ", seqlen: " << seqlen << ", valid_len: " << valid_len << std::endl;
        if (valid_len == 0) {
            offload_task_ids.data_ptr<int64_t>()[seq_idx] = -1;  // -1 indicates no cache to offload for this sequence
            continue;
        }  // skip sequences without cache to offload for this sequence
        int64_t uid = user_ids.data_ptr<int64_t>()[seq_idx];
        auto user_namespace = "uid:" + std::to_string(uid);
        int64_t task_id = flexkv_client_->put_async(
            lookup_token_index_.slice(0, 0, valid_len),
            new_slot_mappings[seq_idx].slice(0, 0, valid_len),
            std::nullopt,
            {user_namespace}
        );
        std::cout << "[KVCACHE][runtime] offload put_async uid=" << uid
                  << " task_id=" << task_id
                  << " start=" << offload_start_indices.data_ptr<int>()[seq_idx]
                  << " len=" << valid_len << std::endl;
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
    std::cout << "[KVCACHE][runtime] offload_kvcache_launch exit pending_tasks="
              << offload_tasks_.size() << std::endl;
    return offload_task_ids;
}

at::Tensor ExportKVCacheRuntime::offload_kvcache_reap_completed() {
    std::cout << "[KVCACHE][runtime] offload_reap_completed enter pending_tasks="
              << offload_tasks_.size() << std::endl;
    std::vector<int64_t> completed_tasks;
    std::list<std::tuple<int64_t, int64_t, int32_t, int32_t>> remaining_tasks;
    for (auto& task : offload_tasks_) {
        auto [task_id, user_id, offload_start_index, offload_length] = task;
        std::cout << "[KVCACHE][runtime] reap try_wait task_id=" << task_id
                  << " uid=" << user_id
                  << " start=" << offload_start_index
                  << " len=" << offload_length << std::endl;
        auto wait_results = flexkv_client_->try_wait({task_id});
        if (wait_results.find(task_id) == wait_results.end()) {
            std::cout << "[KVCACHE][runtime] reap task_id=" << task_id
                      << " no result" << std::endl;
            remaining_tasks.push_back(task);  // not completed yet, keep waiting
            continue;
        }
        if (wait_results[task_id].status == KVResponseStatus::UNREADY) {
            std::cout << "[KVCACHE][runtime] reap task_id=" << task_id
                      << " status=UNREADY" << std::endl;
            remaining_tasks.push_back(task);  // not completed yet, keep waiting
            continue;  // completed successfully, can be reaped
        }

        int offload_succeeded = (wait_results[task_id].status == KVResponseStatus::SUCCESS);
        std::cout << "[KVCACHE][runtime] reap task_id=" << task_id
                  << " status=" << static_cast<int>(wait_results[task_id].status)
                  << " release_pages begin" << std::endl;
        if (wait_results[task_id].status != KVResponseStatus::SUCCESS) {
            // throw std::runtime_error("Offload task_id " + std::to_string(task_id) + " failed with status: " + std::to_string(static_cast<int>(wait_results[task_id].status)));
        }
        completed_tasks.push_back(user_id);
        completed_tasks.push_back(task_id);
        completed_tasks.push_back(static_cast<int64_t>(wait_results[task_id].status));

        gpu_kvcache_->release_offload_pages(
            at::from_blob(&user_id, {1}, at::dtype(torch::kInt64).device(at::kCPU)),
            at::from_blob(&offload_start_index, {1}, at::dtype(torch::kInt32).device(at::kCPU)),
            at::from_blob(&offload_length, {1}, at::dtype(torch::kInt32).device(at::kCPU)),
            { offload_succeeded, }
        );
        std::cout << "[KVCACHE][runtime] reap task_id=" << task_id
                  << " release_pages done" << std::endl;
    }
    offload_tasks_ = std::move(remaining_tasks);
    if (completed_tasks.empty()) {
        std::cout << "[KVCACHE][runtime] offload_reap_completed exit completed=0 pending="
                  << offload_tasks_.size() << std::endl;
        return at::empty({0, 3}, at::dtype(torch::kInt64).device(at::kCPU));  // return empty tensor with shape (0, 3) if no completed task
    }
    std::cout << "[KVCACHE][runtime] offload_reap_completed exit completed="
              << completed_tasks.size() / 3
              << " pending=" << offload_tasks_.size() << std::endl;
    return at::from_blob(completed_tasks.data(), {(int64_t)completed_tasks.size()}, at::dtype(torch::kInt64).device(at::kCPU)).clone().reshape({-1, 3});  // reshape to Nx3 for easier parsing (user_id, task_id, status)
}

std::vector<at::Tensor> ExportKVCacheRuntime::get_kvcache_tables() {
    return cache_tables_;
}

void ExportKVCacheRuntime::evict_kvcache(at::Tensor user_ids, bool evict_gpu_only) {
    const auto batch_size = user_ids.size(0);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids.data_ptr<int64_t>()[seq_idx];
        gpu_kvcache_->evict(uid);
            
        if (!evict_gpu_only) {
            // auto user_namespace = "uid:" + std::to_string(uid);
            // flexkv_client_->evict(user_namespace);
        }
    }
}

} // namespace kvcache_manager

