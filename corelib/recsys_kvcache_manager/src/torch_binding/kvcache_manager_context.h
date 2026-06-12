#pragma once

#include <memory>

namespace kvcache_manager {

class IKVCacheRuntime;

class __attribute__((visibility("default"))) KVCacheRuntimeContext {
public:
    static KVCacheRuntimeContext& instance();
    std::shared_ptr<IKVCacheRuntime> manager() const;

private:
    void set_manager(std::shared_ptr<IKVCacheRuntime> manager);
    bool has_manager() const;
    std::shared_ptr<IKVCacheRuntime> get_manager(
        
    );
    void clear_manager();

private:
    KVCacheRuntimeContext() = default;
    ~KVCacheRuntimeContext() = default;
    KVCacheRuntimeContext(const KVCacheRuntimeContext&) = delete;
    KVCacheRuntimeContext& operator=(const KVCacheRuntimeContext&) = delete;

    static thread_local std::shared_ptr<IKVCacheRuntime> kvcache_manager_;
};

} // namespace kvcache_manager

