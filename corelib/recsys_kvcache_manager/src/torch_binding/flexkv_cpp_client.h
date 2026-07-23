#pragma once

#include <ATen/ATen.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "flexkv_aoti_protocol.h"

namespace kvcache_manager {

/**
 * @class FlexKVCppClient
 * @brief Python-free C++ client for FlexKV server-client mode.
 *
 * This client uses the additive raw AOTI protocol implemented in
 * `FlexKV/flexkv/server/aoti_protocol.py`. Existing Python `KVDPClient`
 * instances continue to use pickle on the same server socket.
 */
class FlexKVCppClient {
public:
    /**
     * @param serverAddr ZMQ server receive endpoint
     * @param dpClientId data-parallel client ID
     * @param tpSize tensor-parallel size for registration bookkeeping
     */
    explicit FlexKVCppClient(const std::string& serverAddr, int32_t dpClientId, int32_t tpSize = 1);
    ~FlexKVCppClient();

    FlexKVCppClient(const FlexKVCppClient&) = delete;
    FlexKVCppClient& operator=(const FlexKVCppClient&) = delete;

    void register_to_server();

    void start_server_and_register();

    const std::string& client_recv_port() const;

    std::pair<int64_t, at::Tensor> get_match(
        const at::Tensor& token_ids,
        const std::optional<at::Tensor>& token_mask,
        int layer_granularity = -1,
        const std::vector<std::string>& namespace_list = {});

    int64_t put_async(
        const at::Tensor& token_ids,
        const at::Tensor& slot_mapping,
        const std::optional<at::Tensor>& token_mask,
        const std::vector<std::string>& namespace_list);

    int64_t get_async(
        const at::Tensor& token_ids,
        const at::Tensor& slot_mapping,
        const std::optional<at::Tensor>& token_mask,
        int layer_granularity = -1,
        const std::vector<std::string>& namespace_list = {});

    std::vector<int64_t> launch_tasks(
        const std::vector<int64_t>& task_ids,
        const std::vector<at::Tensor>& slot_mappings,
        bool as_batch = false);

    std::unordered_map<int, KVTaskResponse> wait(
        const std::vector<int64_t>& task_ids,
        float timeout_sec = 20.0f,
        bool completely = false);

    std::unordered_map<int, KVTaskResponse> try_wait(
        const std::vector<int64_t>& task_ids);

    void cancel(const std::vector<int64_t>& task_ids);

    bool is_ready();

    void shutdown();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class FlexKVGPURegistrator {
public:
    FlexKVGPURegistrator(const std::string& gpuRegisterPort, int32_t dpClientId, int32_t deviceId);
    ~FlexKVGPURegistrator();

    FlexKVGPURegistrator(const FlexKVGPURegistrator&) = delete;
    FlexKVGPURegistrator& operator=(const FlexKVGPURegistrator&) = delete;

    void register_to_server(
        const std::vector<at::Tensor>& kvCaches,
        const FlexKVCacheLayoutSpec& layout,
        std::optional<int32_t> overrideDeviceId = std::nullopt);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kvcache_manager
