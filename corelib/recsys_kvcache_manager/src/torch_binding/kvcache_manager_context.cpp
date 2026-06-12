#include "kvcache_manager_context.h"
#include "export_kvcache_runtime.h"

#include <ATen/ATen.h>
#include <utility>

#ifndef KVCACHE_MANAGER_ENABLE_SCOPED_CONTEXT_GUARD
#define KVCACHE_MANAGER_ENABLE_SCOPED_CONTEXT_GUARD 0
#endif

namespace kvcache_manager {

#if KVCACHE_MANAGER_ENABLE_SCOPED_CONTEXT_GUARD
namespace {

// Optional helper: keep this local to the .cpp so it can be removed freely.
class ScopedKVCacheRuntimeContextGuard final {
public:
    explicit ScopedKVCacheRuntimeContextGuard(std::shared_ptr<IKVCacheRuntime> manager)
        : previous_(KVCacheRuntimeContext::instance().has_manager()
                        ? KVCacheRuntimeContext::instance().get_manager()
                        : nullptr) {
        KVCacheRuntimeContext::instance().set_manager(std::move(manager));
    }

    ~ScopedKVCacheRuntimeContextGuard() {
        if (restore_previous_) {
            KVCacheRuntimeContext::instance().set_manager(std::move(previous_));
        }
    }

    ScopedKVCacheRuntimeContextGuard(const ScopedKVCacheRuntimeContextGuard&) = delete;
    ScopedKVCacheRuntimeContextGuard& operator=(const ScopedKVCacheRuntimeContextGuard&) = delete;

    ScopedKVCacheRuntimeContextGuard(ScopedKVCacheRuntimeContextGuard&&) = delete;
    ScopedKVCacheRuntimeContextGuard& operator=(ScopedKVCacheRuntimeContextGuard&&) = delete;

    void dismiss() {
        restore_previous_ = false;
    }

private:
    std::shared_ptr<IKVCacheRuntime> previous_;
    bool restore_previous_ = true;
};

} // namespace
#endif

thread_local std::shared_ptr<IKVCacheRuntime> KVCacheRuntimeContext::kvcache_manager_ = nullptr;

KVCacheRuntimeContext& KVCacheRuntimeContext::instance() {
    static KVCacheRuntimeContext context;
    if (!context.has_manager())
        context.set_manager(std::make_shared<ExportKVCacheRuntime>());
    return context;
}

void KVCacheRuntimeContext::set_manager(std::shared_ptr<IKVCacheRuntime> manager) {
    kvcache_manager_ = std::move(manager);
}

bool KVCacheRuntimeContext::has_manager() const {
    return static_cast<bool>(kvcache_manager_);
}

std::shared_ptr<IKVCacheRuntime> KVCacheRuntimeContext::manager() const {
    TORCH_CHECK(
        kvcache_manager_ != nullptr,
        "KVCacheRuntimeContext is empty for current thread. "
        "Call set_manager() before invoking kvcache_manager_ops.");
    return kvcache_manager_;
}

std::shared_ptr<IKVCacheRuntime> KVCacheRuntimeContext::get_manager() {
    return manager();
}

void KVCacheRuntimeContext::clear_manager() {
    kvcache_manager_.reset();
}

} // namespace kvcache_manager

