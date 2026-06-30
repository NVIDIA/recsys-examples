import os
# import sys
import tempfile
# import threading
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from flexkv.server.server import KVServer
from flexkv.common.config import (
    ModelConfig, CacheConfig, UserConfig,
    update_default_config_from_user_config, parse_path_list,
    GLOBAL_CONFIG_FROM_ENV,
)

KVCCACHE_ROOT = Path(__file__).resolve().parents[1]
# REPO_ROOT = Path(__file__).resolve().parents[1]
# KVCCACHE_ROOT = REPO_ROOT / "corelib" / "recsys_kvcache_manager"
SRC_ROOT = KVCCACHE_ROOT / "src"
TORCH_BINDING_ROOT = SRC_ROOT / "torch_binding"


def _ipc_endpoint(name: str) -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{name}_"))
    return f"ipc://{temp_dir / 'sock'}"

def get_config(config_path: str):
    """Load config with distributed KVCache support.

    Extends the standard load_config to handle distributed-specific fields:
      enable_p2p_cpu, enable_p2p_ssd, enable_3rd_remote,
      redis_host, redis_port, local_ip, redis_password,
      server_client_mode, etc.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    print(f"Loaded config: {config}")

    model_config = ModelConfig()
    cache_config = CacheConfig()
    user_config = UserConfig()

    # Model config
    model_config.num_layers = config["num_layers"]
    model_config.num_kv_heads = config["num_kv_heads"]
    model_config.head_size = config["head_size"]
    model_config.dtype = eval(f"torch.{config['dtype']}")
    model_config.use_mla = config["use_mla"]
    model_config.tp_size = config["tp_size"]
    model_config.dp_size = config["dp_size"]
    cache_config.tokens_per_block = config["tokens_per_block"]

    # Cache size config
    if "cpu_cache_gb" in config:
        user_config.cpu_cache_gb = config["cpu_cache_gb"]
    if "ssd_cache_gb" in config:
        user_config.ssd_cache_gb = config["ssd_cache_gb"]
    if "ssd_cache_dir" in config:
        user_config.ssd_cache_dir = parse_path_list(config["ssd_cache_dir"])
    if "enable_gds" in config:
        user_config.enable_gds = config["enable_gds"]

    # Distributed KVCache config
    if "enable_p2p_cpu" in config:
        user_config.enable_p2p_cpu = config["enable_p2p_cpu"]
    if "enable_p2p_ssd" in config:
        user_config.enable_p2p_ssd = config["enable_p2p_ssd"]
    if "enable_3rd_remote" in config:
        user_config.enable_3rd_remote = config["enable_3rd_remote"]

    # Redis config
    if "redis_host" in config:
        user_config.redis_host = config["redis_host"]
    if "redis_port" in config:
        user_config.redis_port = config["redis_port"]
    if "local_ip" in config:
        user_config.local_ip = config["local_ip"]
    if "redis_password" in config:
        user_config.redis_password = config["redis_password"]

    # Auto-generate mooncake config JSON and set MOONCAKE_CONFIG_PATH if P2P is enabled
    if config.get("enable_p2p_cpu", False) or config.get("enable_p2p_ssd", False):
        if "MOONCAKE_CONFIG_PATH" not in os.environ:
            mooncake_config = {
                "engine_ip": config.get("mooncake_engine_ip", config.get("local_ip", "127.0.0.1")),
                "engine_port": config.get("mooncake_engine_port", 5555),
                "metadata_backend": config.get("mooncake_metadata_backend", "redis"),
                "metadata_server": config.get("mooncake_metadata_server",
                    f"redis://{config.get('redis_host', '127.0.0.1')}:{config.get('redis_port', 6379)}"),
                "metadata_server_auth": config.get("mooncake_metadata_server_auth",
                    config.get("redis_password", "")),
                "protocol": config.get("mooncake_protocol", "tcp"),
                "device_name": config.get("mooncake_device_name", ""),
            }
            # Write to a temp file that persists until process exits
            mooncake_config_fd, mooncake_config_path = tempfile.mkstemp(
                suffix=".json", prefix="mooncake_config_"
            )
            with os.fdopen(mooncake_config_fd, "w") as f:
                json.dump(mooncake_config, f, indent=2)
            os.environ["MOONCAKE_CONFIG_PATH"] = mooncake_config_path
            print(f"[INFO] Auto-generated mooncake config at: {mooncake_config_path}")
            print(f"[INFO] Mooncake config: {json.dumps(mooncake_config, indent=2)}")
        else:
            mooncake_config_path = os.environ['MOONCAKE_CONFIG_PATH']
            print(f"[INFO] Using existing MOONCAKE_CONFIG_PATH: {mooncake_config_path}")

        # Store mooncake_config_path in cache_config so it survives spawn subprocesses via pickle
        cache_config.mooncake_config_path = mooncake_config_path

    update_default_config_from_user_config(model_config, cache_config, user_config)

    # Handle server_client_mode from config
    if config.get("server_client_mode", False):
        os.environ["FLEXKV_SERVER_CLIENT_MODE"] = "1"
        GLOBAL_CONFIG_FROM_ENV.server_client_mode = True

    return model_config, cache_config

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for FlexKVGPURegistrator smoke test"

    with open(str(KVCCACHE_ROOT / "test" / "pytest_flexkv_cpp_client_ops.cpp"), "r") as fcode:
        cpp_codes = fcode.readlines()

    build_dir = Path(tempfile.mkdtemp("flexkv_cpp_client_ext"))
    wrapper_path = build_dir / "flexkv_cpp_client_test_bindings.cpp"
    wrapper_path.write_text("".join(cpp_codes), encoding="utf-8")

    flexkv_cpp_test_ext = load(
        name="flexkv_cpp_client_test_ext",
        sources=[
            str(wrapper_path),
            str(TORCH_BINDING_ROOT / "flexkv_cpp_client.cpp"),
            str(TORCH_BINDING_ROOT / "flexkv_aoti_protocol.cpp"),
        ],
        extra_include_paths=[
            str(SRC_ROOT),
            str(TORCH_BINDING_ROOT),
        ],
        extra_cflags=["-O3", "-std=c++20"],
        extra_ldflags=["-lzmq", "-lcudart"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=False,
    )


    model_config, cache_config = get_config(str(KVCCACHE_ROOT / "test" / "bindings_test_config.yml"))
    server_recv_port = _ipc_endpoint("flexkv_server_sock")
    gpu_register_port = server_recv_port + "_gpu_register"
    print("server_recv_port: ", server_recv_port)
    print("gpu_register_port: ", gpu_register_port)

    server_handle = KVServer.create_server(
        model_config=model_config,
        cache_config=cache_config,
        gpu_register_port=gpu_register_port,
        server_recv_port=server_recv_port,
        total_clients=1,
        inherit_env=False)
    print("[DEV] Started KVServer")

    result = flexkv_cpp_test_ext.run_cpp_client_smoke(
        server_recv_port,
        gpu_register_port,
        0,
        1,
        torch.cuda.current_device(),
    )

    for key, value in result.items():
        print(f"[DEV] Result - {key}: {value}")
