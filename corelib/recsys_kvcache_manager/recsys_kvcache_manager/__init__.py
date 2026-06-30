# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recsys KVCache Manager - Dynamic KV-cache management for LLM inference."""

from importlib import import_module

from .fake_kvcache_manager_ops import register_fake_kvcache_manager_ops

__all__ = [
    "KVCacheManager",
    "KVCacheBackend",
    "DefaultKVCacheBackend",
    "ExportKVCacheBackend",
    "DeviceKVCache",
    "HostKVStorageBase",
    "NativeHostKVStorage",
    # Backward compatible exports
    "GPUKVCacheManager",
    "HostKVStorageManagerBase",
    "NativeHostKVCacheManager",
    "KVCacheConfig",
    "KVCacheOffloadMode",
    "register_fake_kvcache_manager_ops",
]

_LAZY_IMPORTS = {
    "KVCacheManager": (".kvcache_manager", "KVCacheManager"),
    "KVCacheBackend": (".kvcache_backend", "KVCacheBackend"),
    "DefaultKVCacheBackend": (".default_kvcache_backend", "DefaultKVCacheBackend"),
    "ExportKVCacheBackend": (".export_kvcache_backend", "ExportKVCacheBackend"),
    "DeviceKVCache": (".gpu_kvcache_manager", "DeviceKVCache"),
    "GPUKVCacheManager": (".gpu_kvcache_manager", "GPUKVCacheManager"),
    "HostKVStorageBase": (".host_kvstorage_manager", "HostKVStorageBase"),
    "HostKVStorageManagerBase": (".host_kvstorage_manager", "HostKVStorageManagerBase"),
    "NativeHostKVCacheManager": (".native_host_kvcache_manager", "NativeHostKVCacheManager"),
    "NativeHostKVStorage": (".native_host_kvcache_manager", "NativeHostKVStorage"),
    "KVCacheConfig": (".kvcache_config", "KVCacheConfig"),
    "KVCacheOffloadMode": (".kvcache_utils", "KVCacheOffloadMode"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

register_fake_kvcache_manager_ops()
