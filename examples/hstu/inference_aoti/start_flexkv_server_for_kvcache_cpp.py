# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import os
import signal
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent

from flexkv.common.config import (  # noqa: E402
    CacheConfig,
    ModelConfig,
    UserConfig,
    update_default_config_from_user_config,
)
from flexkv.server.server import KVServer  # noqa: E402


@dataclass
class KVCacheConfig:
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
    extra_configs: Dict[str, Any] = field(default_factory=dict)


_INT_ENV_FIELDS = {
    "KVCACHE_MANAGER_NUM_LAYERS": "num_layers",
    "KVCACHE_MANAGER_NUM_KV_HEADS": "num_heads",
    "KVCACHE_MANAGER_HEAD_SIZE": "head_dim",
    "KVCACHE_MANAGER_TOKENS_PER_PAGE": "page_size",
    "KVCACHE_MANAGER_TOKENS_PER_CHUNK": "offload_chunksize",
    "KVCACHE_MANAGER_NUM_PRIMARY_CACHE_PAGES": "num_primary_cache_pages",
    "KVCACHE_MANAGER_NUM_BUFFER_PAGES": "num_buffer_pages",
    "KVCACHE_MANAGER_MAX_BATCH_SIZE": "max_batch_size",
    "KVCACHE_MANAGER_MAX_SEQUENCE_LENGTH": "max_seq_len",
    "KVCACHE_MANAGER_DEVICE_IDX": "device",
}

_ENV_FILE_NAMES = [
    "SERVER_RECV_PORT",
    "GPU_REGISTER_PORT",
    *_INT_ENV_FIELDS.keys(),
    "KVCACHE_MANAGER_DTYPE",
    "KVCACHE_MANAGER_HOST_CAPACITY_PER_LAYER",
    "KVCACHE_MANAGER_ONLOAD_TIMEOUT_MS",
    "KVCACHE_MANAGER_OFFLOAD_TIMEOUT_MS",
    "KVCACHE_MANAGER_OFFLOAD_MODE",
    "KVCACHE_MANAGER_HOST_KVSTORAGE_BACKEND",
    "KVCACHE_MANAGER_HOST_KVSTORAGE_FAIL_POLICY",
    "FLEXKV_CPU_CACHE_GB",
    "FLEXKV_SSD_CACHE_GB",
    "FLEXKV_ENABLE_P2P_CPU",
    "FLEXKV_ENABLE_P2P_SSD",
    "FLEXKV_ENABLE_3RD_REMOTE",
    "FLEXKV_REDIS_HOST",
    "FLEXKV_REDIS_PORT",
    "FLEXKV_LOCAL_IP",
    "FLEXKV_REDIS_PASSWORD",
]


def _write_env_file(path: Path, env_values: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as env_file:
        for name in _ENV_FILE_NAMES:
            value = (
                env_values.get(name) if env_values is not None else os.environ.get(name)
            )
            if value is not None:
                env_file.write(f"export {name}={value!r}\n")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _env_int(name: str) -> int:
    try:
        return int(_required_env(name))
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer, got {os.environ.get(name)!r}"
        ) from exc


def _env_optional_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _env_bool(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_torch_dtype(name: str) -> torch.dtype:
    value = _required_env(name).strip().lower()
    if value in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    if value in ("float16", "fp16", "half", "torch.float16"):
        return torch.float16
    raise ValueError(f"{name} must be bfloat16 or float16, got {value!r}")


def _dtype_size_bytes(dtype: torch.dtype) -> int:
    if dtype in (torch.bfloat16, torch.float16):
        return 2
    raise ValueError(f"Unsupported kvcache dtype for export runtime: {dtype}")


def _host_capacity_per_layer(config_values: Dict[str, Any], dtype: torch.dtype) -> int:
    override = _env_optional_int("KVCACHE_MANAGER_HOST_CAPACITY_PER_LAYER")
    if override is not None:
        return override
    return (
        config_values["num_primary_cache_pages"]
        * 2
        * config_values["page_size"]
        * config_values["num_heads"]
        * config_values["head_dim"]
        * _dtype_size_bytes(dtype)
    )


def _extra_flexkv_configs() -> Dict[str, Any]:
    configs: Dict[str, Any] = {}
    bool_envs = {
        "FLEXKV_ENABLE_P2P_CPU": "enable_p2p_cpu",
        "FLEXKV_ENABLE_P2P_SSD": "enable_p2p_ssd",
        "FLEXKV_ENABLE_3RD_REMOTE": "enable_3rd_remote",
    }
    for env_name, config_name in bool_envs.items():
        value = _env_bool(env_name)
        if value is not None:
            configs[config_name] = value

    string_envs = {
        "FLEXKV_REDIS_HOST": "redis_host",
        "FLEXKV_LOCAL_IP": "local_ip",
        "FLEXKV_REDIS_PASSWORD": "redis_password",
    }
    for env_name, config_name in string_envs.items():
        str_value = os.environ.get(env_name)
        if str_value:
            configs[config_name] = str_value

    redis_port = _env_optional_int("FLEXKV_REDIS_PORT")
    if redis_port is not None:
        configs["redis_port"] = redis_port
    return configs


def _runtime_env_values(kvcache_config: KVCacheConfig) -> Dict[str, Any]:
    values: Dict[str, Any] = {
        "KVCACHE_MANAGER_NUM_LAYERS": kvcache_config.num_layers,
        "KVCACHE_MANAGER_NUM_KV_HEADS": kvcache_config.num_heads,
        "KVCACHE_MANAGER_HEAD_SIZE": kvcache_config.head_dim,
        "KVCACHE_MANAGER_TOKENS_PER_PAGE": kvcache_config.page_size,
        "KVCACHE_MANAGER_TOKENS_PER_CHUNK": kvcache_config.offload_chunksize,
        "KVCACHE_MANAGER_NUM_PRIMARY_CACHE_PAGES": kvcache_config.num_primary_cache_pages,
        "KVCACHE_MANAGER_NUM_BUFFER_PAGES": kvcache_config.num_buffer_pages,
        "KVCACHE_MANAGER_MAX_BATCH_SIZE": kvcache_config.max_batch_size,
        "KVCACHE_MANAGER_MAX_SEQUENCE_LENGTH": kvcache_config.max_seq_len,
        "KVCACHE_MANAGER_DEVICE_IDX": kvcache_config.device,
        "KVCACHE_MANAGER_DTYPE": "bfloat16"
        if kvcache_config.dtype == torch.bfloat16
        else "float16",
        "KVCACHE_MANAGER_HOST_CAPACITY_PER_LAYER": kvcache_config.host_capacity_per_layer,
        "KVCACHE_MANAGER_ONLOAD_TIMEOUT_MS": kvcache_config.onload_timeout_ms,
        "KVCACHE_MANAGER_OFFLOAD_TIMEOUT_MS": kvcache_config.offload_timeout_ms,
        "KVCACHE_MANAGER_OFFLOAD_MODE": kvcache_config.offload_mode,
        "KVCACHE_MANAGER_HOST_KVSTORAGE_BACKEND": kvcache_config.host_kvstorage_backend,
        "KVCACHE_MANAGER_HOST_KVSTORAGE_FAIL_POLICY": kvcache_config.host_kvstorage_fail_policy,
    }
    for env_name in (
        "FLEXKV_ENABLE_P2P_CPU",
        "FLEXKV_ENABLE_P2P_SSD",
        "FLEXKV_ENABLE_3RD_REMOTE",
        "FLEXKV_REDIS_HOST",
        "FLEXKV_REDIS_PORT",
        "FLEXKV_LOCAL_IP",
        "FLEXKV_REDIS_PASSWORD",
    ):
        if os.environ.get(env_name) is not None:
            values[env_name] = os.environ[env_name]
    return values


def _set_runtime_env(env_values: Dict[str, Any]) -> None:
    for name, value in env_values.items():
        os.environ[name] = str(value)


def _make_kvcache_config_from_env() -> KVCacheConfig:
    config_values = {
        field_name: _env_int(env_name)
        for env_name, field_name in _INT_ENV_FIELDS.items()
    }
    dtype = _env_torch_dtype("KVCACHE_MANAGER_DTYPE")
    host_capacity_per_layer = _host_capacity_per_layer(config_values, dtype)
    return KVCacheConfig(
        **config_values,
        host_capacity_per_layer=host_capacity_per_layer,
        dtype=dtype,
        host_kvstorage_backend=os.environ.get(
            "KVCACHE_MANAGER_HOST_KVSTORAGE_BACKEND", "flexkv"
        ),
        onload_timeout_ms=_env_float("KVCACHE_MANAGER_ONLOAD_TIMEOUT_MS", 0.0),
        offload_timeout_ms=_env_float("KVCACHE_MANAGER_OFFLOAD_TIMEOUT_MS", 100.0),
        offload_mode=os.environ.get("KVCACHE_MANAGER_OFFLOAD_MODE", "lazy"),
        host_kvstorage_fail_policy=os.environ.get(
            "KVCACHE_MANAGER_HOST_KVSTORAGE_FAIL_POLICY", "fail_open"
        ),
        extra_configs=_extra_flexkv_configs(),
    )


def _ipc_endpoint(name: str) -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{name}_"))
    return f"ipc://{temp_dir / 'sock'}"


def _start_flexkv_server(kvcache_config: KVCacheConfig, env_file: Path):
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

    cpu_cache_gb = _env_optional_int("FLEXKV_CPU_CACHE_GB") or max(
        1,
        math.ceil(
            kvcache_config.host_capacity_per_layer
            * kvcache_config.num_layers
            / (1024**3)
        ),
    )
    ssd_cache_gb = _env_optional_int("FLEXKV_SSD_CACHE_GB") or 0
    user_config.cpu_cache_gb = cpu_cache_gb
    user_config.ssd_cache_gb = ssd_cache_gb
    for name, value in (kvcache_config.extra_configs or {}).items():
        setattr(user_config, name, value)

    update_default_config_from_user_config(model_config, cache_config, user_config)

    server_recv_port = _ipc_endpoint("flexkv_server_sock")
    gpu_register_port = server_recv_port + "_gpu_register"
    env_values = {
        "SERVER_RECV_PORT": server_recv_port,
        "GPU_REGISTER_PORT": gpu_register_port,
        **_runtime_env_values(kvcache_config),
        "FLEXKV_CPU_CACHE_GB": cpu_cache_gb,
        "FLEXKV_SSD_CACHE_GB": ssd_cache_gb,
    }
    _set_runtime_env(env_values)
    _write_env_file(env_file, env_values)
    print(f"[INFO] Wrote C++ runtime env file: {env_file}", flush=True)

    server_handle = KVServer.create_server(
        model_config=model_config,
        cache_config=cache_config,
        gpu_register_port=gpu_register_port,
        server_recv_port=server_recv_port,
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
        "--env_file",
        type=str,
        default=str(SCRIPT_DIR / "kvcache_cpp_runtime.env"),
        help="Shell env file to source before running the C++ demo.",
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
        kvcache_config = _make_kvcache_config_from_env()
        env_file = Path(args.env_file).resolve()
        server_handle = _start_flexkv_server(kvcache_config, env_file)
        print(
            f"[INFO] SERVER_RECV_PORT={os.environ.get('SERVER_RECV_PORT')}", flush=True
        )
        print(
            f"[INFO] GPU_REGISTER_PORT={os.environ.get('GPU_REGISTER_PORT')}",
            flush=True,
        )
        print("[INFO] Source this file before running the C++ demo:", flush=True)
        print(f"       source {env_file}", flush=True)
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
