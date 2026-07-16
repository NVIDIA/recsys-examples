# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from flexkv.common.config import (  # noqa: E402
    CacheConfig,
    ModelConfig,
    UserConfig,
    update_default_config_from_user_config,
)
from flexkv.server.server import KVServer  # noqa: E402
from kvcache_runtime_config import (  # noqa: E402
    DEFAULT_KVCACHE_CONFIG_FILE,
    config_float,
    config_int,
    config_optional_int,
    config_torch_dtype,
    extra_flexkv_configs,
    load_kvcache_runtime_yaml,
    required_config,
    reset_ipc_socket,
)


@dataclass
class KVCacheConfig:
    server_recv_port: str
    gpu_register_port: str
    num_layers: int
    num_heads: int
    head_dim: int
    page_size: int
    offload_chunksize: int
    num_primary_cache_pages: int
    num_buffer_pages: int
    host_capacity_per_layer: int
    max_batch_size: int
    max_seq_len: int
    dtype: torch.dtype
    device: int
    host_kvstorage_backend: str = "flexkv"
    onload_timeout_ms: float = 0.0
    offload_timeout_ms: float = 100.0
    offload_mode: str = "lazy"
    host_kvstorage_fail_policy: str = "fail_open"
    cpu_cache_gb: Optional[int] = None
    ssd_cache_gb: int = 0
    extra_configs: Dict[str, Any] = field(default_factory=dict)


_INT_CONFIG_FIELDS = {
    "num_layers": "num_layers",
    "num_kv_heads": "num_heads",
    "head_size": "head_dim",
    "tokens_per_page": "page_size",
    "tokens_per_chunk": "offload_chunksize",
    "num_primary_cache_pages": "num_primary_cache_pages",
    "num_buffer_pages": "num_buffer_pages",
    "max_batch_size": "max_batch_size",
    "max_sequence_length": "max_seq_len",
    "device_idx": "device",
}


def _dtype_size_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.bfloat16, torch.float16):
        return 2
    raise ValueError(f"Unsupported kvcache dtype for export runtime: {dtype}")


def _host_capacity_per_layer(config_values: Dict[str, Any], dtype: torch.dtype) -> int:
    override = config_values.get("host_capacity_per_layer")
    if override is not None:
        return int(override)
    return (
        config_values["num_primary_cache_pages"]
        * 2
        * config_values["page_size"]
        * config_values["num_heads"]
        * config_values["head_dim"]
        * _dtype_size_bytes(dtype)
    )


def _make_kvcache_config_from_yaml(
    config_file: Optional[str],
) -> tuple[KVCacheConfig, str]:
    config, resolved_config_file = load_kvcache_runtime_yaml(config_file)
    config_values = {
        field_name: config_int(config, yaml_name)
        for yaml_name, field_name in _INT_CONFIG_FIELDS.items()
    }
    dtype = config_torch_dtype(config)
    capacity_values = {
        **config_values,
        "host_capacity_per_layer": config_optional_int(
            config, "host_capacity_per_layer"
        ),
    }
    host_capacity_per_layer = _host_capacity_per_layer(capacity_values, dtype)
    return KVCacheConfig(
        **config_values,
        server_recv_port=str(required_config(config, "server_recv_port")),
        gpu_register_port=str(required_config(config, "gpu_register_port")),
        host_capacity_per_layer=host_capacity_per_layer,
        dtype=dtype,
        host_kvstorage_backend=str(config.get("host_kvstorage_backend", "flexkv")),
        onload_timeout_ms=config_float(config, "onload_timeout_ms", 0.0),
        offload_timeout_ms=config_float(config, "offload_timeout_ms", 100.0),
        offload_mode=str(config.get("offload_mode", "lazy")),
        host_kvstorage_fail_policy=str(
            config.get("host_kvstorage_fail_policy", "fail_open")
        ),
        cpu_cache_gb=config_optional_int(config, "cpu_cache_gb"),
        ssd_cache_gb=config_int(config, "ssd_cache_gb")
        if config.get("ssd_cache_gb") is not None
        else 0,
        extra_configs=extra_flexkv_configs(config),
    ), str(resolved_config_file)


def _start_flexkv_server(kvcache_config: KVCacheConfig):
    model_config = ModelConfig()
    cache_config = CacheConfig()
    user_config = UserConfig()

    model_config.num_layers = kvcache_config.num_layers
    model_config.num_kv_heads = kvcache_config.num_heads
    model_config.head_size = kvcache_config.head_dim
    model_config.dtype = kvcache_config.dtype
    model_config.use_mla = False
    model_config.tp_size = 1
    model_config.dp_size = 1
    cache_config.tokens_per_block = kvcache_config.page_size

    cpu_cache_gb = kvcache_config.cpu_cache_gb or max(
        1,
        math.ceil(
            kvcache_config.host_capacity_per_layer
            * kvcache_config.num_layers
            / (1024**3)
        ),
    )
    ssd_cache_gb = kvcache_config.ssd_cache_gb
    user_config.cpu_cache_gb = cpu_cache_gb
    user_config.ssd_cache_gb = ssd_cache_gb
    for name, value in (kvcache_config.extra_configs or {}).items():
        setattr(user_config, name, value)

    update_default_config_from_user_config(model_config, cache_config, user_config)

    reset_ipc_socket(kvcache_config.server_recv_port)
    reset_ipc_socket(kvcache_config.gpu_register_port)

    server_handle = KVServer.create_server(
        model_config=model_config,
        cache_config=cache_config,
        gpu_register_port=kvcache_config.gpu_register_port,
        server_recv_port=kvcache_config.server_recv_port,
        total_clients=1,
        inherit_env=True,
    )
    print("[INFO] Started FlexKV server for kvcache C++ demo", flush=True)
    time.sleep(3)
    return server_handle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the FlexKV server required by the KV-cache C++ AOTI demo."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=str(DEFAULT_KVCACHE_CONFIG_FILE),
        help="Static YAML config shared with the C++ KV-cache runtime.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to start the FlexKV export runtime server."
        )

    server_handle = None
    try:
        kvcache_config, config_file = _make_kvcache_config_from_yaml(args.config_file)
        os.environ["KVCACHE_MANAGER_CONFIG_FILE"] = str(config_file)
        server_handle = _start_flexkv_server(kvcache_config)
        print(f"[INFO] Loaded KV-cache runtime config: {config_file}", flush=True)
        print(f"[INFO] SERVER_RECV_PORT={kvcache_config.server_recv_port}", flush=True)
        print(
            f"[INFO] GPU_REGISTER_PORT={kvcache_config.gpu_register_port}",
            flush=True,
        )
        print(
            "[INFO] Run the C++ demo with this config path:",
            flush=True,
        )
        print(
            f"       export KVCACHE_MANAGER_CONFIG_FILE={config_file}",
            flush=True,
        )
        print("[INFO] FlexKV server is running. Press Ctrl+C to stop.", flush=True)

        stop = False

        def _handle_signal(signum, _frame):
            nonlocal stop
            print(
                f"[INFO] Received signal {signum}; shutting down FlexKV server.",
                flush=True,
            )
            stop = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        start_time = time.time()
        while not stop:
            time.sleep(1)
            if time.time() - start_time >= 10:
                print(
                    "[INFO] FlexKV server has been running for 10 seconds.", flush=True
                )
                start_time = time.time()

        if server_handle is not None:
            server_handle.shutdown()
        return 0
    except BaseException as exc:
        print("[ERROR] FlexKV C++ demo server helper failed:", flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        if server_handle is not None:
            try:
                print(
                    "[INFO] Attempting to shut down FlexKV server after error.",
                    flush=True,
                )
                server_handle.shutdown()
            except BaseException as shutdown_exc:
                print("[ERROR] FlexKV server shutdown also failed:", flush=True)
                traceback.print_exception(
                    type(shutdown_exc),
                    shutdown_exc,
                    shutdown_exc.__traceback__,
                    file=sys.stderr,
                )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
