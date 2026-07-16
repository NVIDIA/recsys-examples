# Export KVCache Design

## 1. Scope and Goals
This document captures the finalized class naming and architecture for an export-friendly KV cache stack.

Goals:
1. Keep one stable Python user interface.
2. Support both default and export paths with shared workflow.
3. Keep stateful runtime logic in C++ for export and AOTI compatibility.
4. Keep torch custom ops thin and tensor-only.
5. Use thread-local runtime context instead of explicit context_id in this phase.

## 2. Final Naming

### 2.1 Python layer
1. `KVCacheManager`: user-facing API (kept for compatibility).
2. `KVCacheBackend`: abstract backend interface.
3. `DefaultKVCacheBackend`: default implementation using current Python manager stack.
4. `DeviceKVCache`: device-side KV cache component.
5. `NativeHostKVStorage`: native host storage component.
6. `FlexKVStorage`: FlexKV-backed host storage component.

### 2.2 C++ layer
1. `KVCacheRuntimeContext`: thread-local context holder for current runtime instance.
2. `IKVCacheRuntime`: abstract C++ runtime interface.
3. `ExportKVCacheRuntime`: concrete C++ runtime implementation.

## 3. High-Level Architecture and Ownership

## 3.1 Layer responsibilities
1. `KVCacheManager`
- Owns orchestration workflow used by model/inference code.
- Delegates backend-specific operations to `KVCacheBackend`.

2. `KVCacheBackend`
- Defines backend contract for operations such as lookup and allocate.
- Keeps `KVCacheManager` independent of implementation details.

3. `DefaultKVCacheBackend`
- Uses current Python components (`DeviceKVCache`, `NativeHostKVStorage` or `FlexKVStorage`).
- Preserves current behavior as the default path.

4. `ExportKVCacheRuntime`
- Owns runtime state and policy in C++.
- Owns references to device and host storage runtime components.

5. `KVCacheRuntimeContext`
- Holds the current `IKVCacheRuntime` pointer in `thread_local` storage.
- Serves runtime resolution for all kvcache torch ops on the active thread.

## 3.2 Ownership model
1. One inference instance owns one `ExportKVCacheRuntime`.
2. The runtime is set into `KVCacheRuntimeContext` on inference entry.
3. Torch ops read runtime from context and invoke methods.
4. Runtime owns mutable state and backend handles.
5. Context is cleared at inference exit.

## 4. Key Relations (Composition + Shared Workflow)
1. `KVCacheManager` has-a `KVCacheBackend`.
2. `DefaultKVCacheBackend` has-a `DeviceKVCache` and host storage component.
3. Export path calls ops from `torch.ops.kvcache_manager_ops`
4. `ExportKVCacheRuntime` implements `IKVCacheRuntime`.
5. `KVCacheRuntimeContext` stores `std::shared_ptr<IKVCacheRuntime>` per thread.

## 5. lookup_kvcache: Implementation Outline

## 5.1 Public API contract
`KVCacheManager.lookup_kvcache(user_ids, sequence_lengths)` returns lookup metadata/results with the same logical schema regardless of backend.

## 5.2 Default path (non-export)
1. `KVCacheManager.lookup_kvcache` delegates to `DefaultKVCacheBackend.lookup_kvcache`.
2. Backend calls `DeviceKVCache.lookup(user_ids)`.
3. Backend builds index metadata and calls host storage lookup.
4. Backend merges host and device lookup results.
5. Backend returns merged result to `KVCacheManager`.

## 5.3 Export path (AOTI and torch.export focus)
1. Inference initialization sets `ExportKVCacheRuntime` into `KVCacheRuntimeContext`.
2. Lookup in model forward calls `torch.ops.kvcache_manager_ops.lookup_kvcache(...)`.
3. C++ op shim resolves runtime from `KVCacheRuntimeContext`.
4. C++ op shim calls `ExportKVCacheRuntime::lookup_kvcache(...)`.
5. Runtime performs device lookup and host lookup, merges outputs.
6. Runtime returns tensor-only outputs to op shim.
7. Op shim returns outputs to Python backend.

## 5.4 Export path pseudocode
```python
# Python
class ExportableModel:
    def forward(self, user_ids, sequence_lengths, ...):
        lookup_results = torch.ops.kvcache_manager_ops.lookup_kvcache(user_ids, sequence_lengths, ordering_tensor)
```

```cpp
// C++ op shim
std::vector<at::Tensor> lookup_kvcache_op(const at::Tensor& user_ids,
                                          const at::Tensor& sequence_lengths) {
    auto runtime = KVCacheRuntimeContext::instance().runtime();
    return runtime->lookup_kvcache(user_ids, sequence_lengths);
}
```

```cpp
// C++ runtime
std::vector<at::Tensor> ExportKVCacheRuntime::lookup_kvcache(
    const at::Tensor& user_ids,
    const at::Tensor& sequence_lengths) {
    auto gpu = device_kvcache_->lookup(user_ids);
    auto idx = host_kvstorage_->build_index_meta(user_ids, sequence_lengths);
    auto host = host_kvstorage_->lookup_kvcache(idx);
    return merge_lookup_outputs(gpu, host);
}
```

## 6. Thread-Local Context Decision
1. Current phase uses thread-local context and does not use explicit `context_id` op arguments.
2. This is valid under these assumptions:
- Each inference instance has its own device KV cache table.
- Host backend handles concurrency outside this layer.
3. If future execution interleaves multiple inference instances on the same thread, add explicit `context_id` to op schemas.

## 7. Torch Op Design Constraints
1. Torch ops remain thin adapters only.
2. Torch op inputs and outputs are tensor-first and export-friendly.
3. Runtime policy and mutable state live in C++ runtime, not in op wrappers.
4. Every exported op should provide shape-compatible Meta behavior.

## 8. Migration Notes
1. Keep existing public API name `KVCacheManager` for compatibility.
2. Introduce backend interface and two implementations first.
3. Route `lookup_kvcache` through backend as the first migrated API.
4. Migrate `allocate` and onboard/offload APIs incrementally.
5. Keep fallback to `DefaultKVCacheBackend` while validating export path.

## 9. Optional Additions
1. Add an RAII helper around thread-local context set and restore in C++ implementation.
2. Add a backend selection config flag to switch default and export implementations.
3. Add parity tests that compare default and export lookup outputs for identical inputs.
