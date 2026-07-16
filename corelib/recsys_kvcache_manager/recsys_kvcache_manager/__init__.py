# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recsys KVCache Manager - Dynamic KV-cache management for LLM inference."""

import os
from importlib import import_module
from pathlib import Path

import torch

_OPS_LOADED = False
_RUNTIME_OPS_BASENAME = "kvcache_manager_ops.so"


def load_kvcache_manager_ops(strict: bool = False) -> bool:
    global _OPS_LOADED

    if _OPS_LOADED:
        return True

    package_root = Path(__file__).resolve().parent
    candidate_paths = []

    env_dir = os.getenv("KVCACHE_MANAGER_OPS_LIB_DIR", "")
    if env_dir:
        candidate_paths.append(Path(env_dir) / _RUNTIME_OPS_BASENAME)

    env_path = os.getenv("KVCACHE_MANAGER_OPS_LIB") or os.getenv(
        "KVCACHE_MANAGER_OPS_LIBRARY"
    )
    if env_path:
        candidate_paths.append(Path(env_path))

    candidate_paths.extend(
        [
            package_root / _RUNTIME_OPS_BASENAME,
            package_root.parent / "build" / _RUNTIME_OPS_BASENAME,
            package_root.parent / _RUNTIME_OPS_BASENAME,
        ]
    )

    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        torch.ops.load_library(str(candidate))
        _OPS_LOADED = True
        return True

    if strict:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            f"Unable to locate {_RUNTIME_OPS_BASENAME}. Searched: {searched}"
        )

    return False


load_kvcache_manager_ops(strict=False)

from .fake_kvcache_manager_ops import register_fake_kvcache_manager_ops

__all__ = [
    "KVCacheManager",
    "KVCacheBackend",
    "DefaultKVCacheBackend",
    "DeviceKVCache",
    "HostKVStorageBase",
    "NativeHostKVStorage",
    # Backward compatible exports
    "GPUKVCacheManager",
    "HostKVStorageManagerBase",
    "NativeHostKVCacheManager",
    "KVCacheConfig",
    "KVCacheOffloadMode",
    "load_kvcache_manager_ops",
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
    "NativeHostKVCacheManager": (
        ".native_host_kvcache_manager",
        "NativeHostKVCacheManager",
    ),
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


if _OPS_LOADED:
    register_fake_kvcache_manager_ops()
