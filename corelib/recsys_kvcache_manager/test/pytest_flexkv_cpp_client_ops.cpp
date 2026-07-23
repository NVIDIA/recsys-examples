#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <torch/library.h>


#include "flexkv_cpp_client.h"

namespace py = pybind11;
using namespace kvcache_manager;

static std::jthread tp_thread;

void run_tp_client(
    std::stop_token stop_token,
    torch::Tensor& gpu_tensor,
    const std::string& gpu_register_port,
    int dp_client_id,
    int tp_size,
    int device_id) {
    // auto gpu_tensor = torch::zeros(
    //     {1024, 2, 32, 4, 128},
    //     torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, device_id));
    FlexKVGPURegistrator tp_client(gpu_register_port, dp_client_id, device_id);
    FlexKVCacheLayoutSpec layout{
        FlexKVLayoutType::LayerFirst,
        1,
        1024,
        32,
        4,
        128,
        false,
    };
    tp_client.register_to_server({gpu_tensor}, layout);

    while (true) {
        if (stop_token.stop_requested()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void shutdown_tp_client() {
    tp_thread.request_stop();
    tp_thread.join();
    std::cout << "[DEV] TP client thread has been shutdown" << std::endl;
    std::cout << "[DEV] TP client thread has been shutdown" << std::endl;
    std::cout << "[DEV] TP client thread has been shutdown" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10)); // ensure the shutdown message is printed before main thread exits
}

py::dict run_cpp_client_smoke(
    const std::string& server_recv_port,
    const std::string& gpu_register_port,
    int dp_client_id,
    int tp_size,
    int device_id) {
    py::dict result;

    FlexKVCppClient dp_client(server_recv_port, dp_client_id, tp_size);
    dp_client.start_server_and_register();

    auto gpu_tensor = torch::zeros(
        {1024, 2, 32, 4, 128},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, device_id));
    tp_thread = std::jthread(run_tp_client, std::ref(gpu_tensor), gpu_register_port, dp_client_id, tp_size, device_id);
    shutdown_tp_client();

    auto start_time = std::chrono::steady_clock::now();
    while (!dp_client.is_ready()) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > std::chrono::seconds(30)) {
            throw std::runtime_error("DP client failed to become ready within 30 seconds");
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    const bool is_ready = dp_client.is_ready();
    std::cout << "[DEV] DP client is ready, proceeding with get_match request" << std::endl;

    auto token_ids = torch::tensor({11, 22, 33}, torch::TensorOptions().dtype(torch::kInt64));
    auto get_match_result1 = dp_client.get_match(token_ids, std::nullopt, 1, {"uid:11"});
    auto get_match_result2 = dp_client.get_match(token_ids, std::nullopt, 1, {"uid:47"});

    {
        auto token_ids = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}, torch::TensorOptions().dtype(torch::kInt64));
        auto slot_mappings = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}, torch::TensorOptions().dtype(torch::kInt64));
        auto offload_task_id = dp_client.put_async(token_ids, slot_mappings, std::nullopt, {"uid:23"});

        auto offload_raw_resp = dp_client.wait({offload_task_id}, 1000.f, true);
        if (offload_raw_resp.end() == offload_raw_resp.find(offload_task_id)) {
            throw std::runtime_error("Failed to get offload response for task " + std::to_string(offload_task_id));
        }
        auto offload_resp = offload_raw_resp[offload_task_id];
        std::cout << "[DEV] offload status: " << static_cast<int>(offload_resp.status) << std::endl;
    }

    {
        auto token_ids = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}, torch::TensorOptions().dtype(torch::kInt64));
        auto slot_mappings = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}, torch::TensorOptions().dtype(torch::kInt64));
        slot_mappings = slot_mappings + 64;
        auto [onboard_task_id, cached_result] = dp_client.get_match(token_ids, std::nullopt, 1, {"uid:23"});
        std::cout << "[DEV] cached length: " << cached_result.sum().item() << std::endl;

        dp_client.launch_tasks({onboard_task_id}, {slot_mappings}, true);
        auto onboard_raw_resp = dp_client.wait({onboard_task_id}, 1000.f, true);
        if (onboard_raw_resp.end() == onboard_raw_resp.find(onboard_task_id)) {
            throw std::runtime_error("Failed to get onboard response for task " + std::to_string(onboard_task_id));
        }
        auto onboard_resp = onboard_raw_resp[onboard_task_id];
        std::cout << "[DEV] onboard status: " << static_cast<int>(onboard_resp.status) << std::endl;
    }

    // shutdown_tp_client();
    dp_client.shutdown();

    result["ready"] = is_ready;
    result["task_id_1"] = get_match_result1.first;
    result["mask_1"] = get_match_result1.second.to(torch::kCPU).to(torch::kUInt8);
    result["task_id_2"] = get_match_result2.first;
    result["mask_2"] = get_match_result2.second.to(torch::kCPU).to(torch::kUInt8);
    result["client_recv_port"] = dp_client.client_recv_port();
    return result;
}

PYBIND11_MODULE(flexkv_cpp_client_test_ext, m) {
    m.def("run_cpp_client_smoke", &run_cpp_client_smoke);
}