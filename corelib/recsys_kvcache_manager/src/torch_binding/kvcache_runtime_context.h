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
 * @class KVCacheRuntimeContext
 * @brief Python-free C++ client for FlexKV server-client mode.
 *
 * This client uses the additive raw AOTI protocol implemented in
 * `FlexKV/flexkv/server/aoti_protocol.py`. Existing Python `KVDPClient`
 * instances continue to use pickle on the same server socket.
 */
class KVCacheRuntimeContext {
public:
    /**
     * @param serverAddr ZMQ server receive endpoint
     * @param dpClientId data-parallel client ID
     * @param tpSize tensor-parallel size for registration bookkeeping
     */
    explicit KVCacheRuntimeContext(const std::string& serverAddr, int32_t dpClientId, int32_t tpSize = 1);
    ~KVCacheRuntimeContext();

    KVCacheRuntimeContext(const KVCacheRuntimeContext&) = delete;
    KVCacheRuntimeContext& operator=(const KVCacheRuntimeContext&) = delete;

    static KVCacheRuntimeContext& instance();

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
