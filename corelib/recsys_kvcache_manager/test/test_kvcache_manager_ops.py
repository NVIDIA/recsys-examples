import json
import os
import tempfile
import time
from pathlib import Path

import torch
from flexkv.common.config import (
    GLOBAL_CONFIG_FROM_ENV,
    CacheConfig,
    ModelConfig,
    UserConfig,
    parse_path_list,
    update_default_config_from_user_config,
)
from flexkv.server.server import KVServer
from torch.export import Dim

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# if str(PACKAGE_ROOT) not in sys.path:
#     sys.path.insert(0, str(PACKAGE_ROOT))

from recsys_kvcache_manager import (
    load_kvcache_manager_ops,
    register_fake_kvcache_manager_ops,
)

load_kvcache_manager_ops(strict=True)


def _dtype_env_value(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"Unsupported dtype for export runtime env: {dtype}")


def set_export_kvcache_runtime_env(
    model_config: ModelConfig,
    cache_config: CacheConfig,
) -> None:
    env_values = {
        "KVCACHE_MANAGER_NUM_LAYERS": model_config.num_layers,
        "KVCACHE_MANAGER_NUM_KV_HEADS": model_config.num_kv_heads,
        "KVCACHE_MANAGER_HEAD_SIZE": model_config.head_size,
        "KVCACHE_MANAGER_TOKENS_PER_PAGE": cache_config.tokens_per_block,
        "KVCACHE_MANAGER_TOKENS_PER_CHUNK": 128,
        "KVCACHE_MANAGER_NUM_PRIMARY_CACHE_PAGES": 512,
        "KVCACHE_MANAGER_NUM_BUFFER_PAGES": 0,
        "KVCACHE_MANAGER_MAX_BATCH_SIZE": 8,
        "KVCACHE_MANAGER_MAX_SEQUENCE_LENGTH": 2048,
        "KVCACHE_MANAGER_DEVICE_IDX": torch.cuda.current_device(),
        "KVCACHE_MANAGER_DTYPE": _dtype_env_value(model_config.dtype),
    }
    for name, value in env_values.items():
        os.environ[name] = str(value)


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
                "engine_ip": config.get(
                    "mooncake_engine_ip", config.get("local_ip", "127.0.0.1")
                ),
                "engine_port": config.get("mooncake_engine_port", 5555),
                "metadata_backend": config.get("mooncake_metadata_backend", "redis"),
                "metadata_server": config.get(
                    "mooncake_metadata_server",
                    f"redis://{config.get('redis_host', '127.0.0.1')}:{config.get('redis_port', 6379)}",
                ),
                "metadata_server_auth": config.get(
                    "mooncake_metadata_server_auth", config.get("redis_password", "")
                ),
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
            mooncake_config_path = os.environ["MOONCAKE_CONFIG_PATH"]
            print(f"[INFO] Using existing MOONCAKE_CONFIG_PATH: {mooncake_config_path}")

        # Store mooncake_config_path in cache_config so it survives spawn subprocesses via pickle
        cache_config.mooncake_config_path = mooncake_config_path

    update_default_config_from_user_config(model_config, cache_config, user_config)

    # Handle server_client_mode from config
    if config.get("server_client_mode", False):
        os.environ["FLEXKV_SERVER_CLIENT_MODE"] = "1"
        GLOBAL_CONFIG_FROM_ENV.server_client_mode = True

    return model_config, cache_config


class KVCacheManagerTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("dummy", torch.zeros(0, dtype=torch.int64))

    def forward(self, seqlens, uids):
        with torch.no_grad():
            # Init kvcache manager context and manager before lookup
            # torch.ops.kvcache_manager_ops.init_kvcache(self.dummy)

            # Lookup
            lookup_res = torch.ops.kvcache_manager_ops.lookup(uids, seqlens, uids)

            # Allocate
            alloc_result = torch.ops.kvcache_manager_ops.allocate(
                uids, seqlens, lookup_res[1], lookup_res[5]
            )

            # # Offload launch
            offload_task_ids = torch.ops.kvcache_manager_ops.offload_launch(
                uids,
                seqlens,
                lookup_res[1],
                lookup_res[5],
                lookup_res[2],
                lookup_res[3],
                alloc_result[0],
                alloc_result[2],
                [torch.ones(1, dtype=torch.int64) * -1],
                self.dummy,
            )

            # # Reap completed offload tasks
            reap_result = torch.ops.kvcache_manager_ops.offload_reap_completed(
                self.dummy
            )
            flush_result = torch.ops.kvcache_manager_ops.offload_flush(reap_result)

            # Lookup
            lookup_res_ = torch.ops.kvcache_manager_ops.lookup(
                uids, seqlens, flush_result
            )

            # Evict GPU cache only
            evict_res = torch.ops.kvcache_manager_ops.evict_kvcache(
                uids, True, lookup_res_[0]
            )

            # Lookup
            lookup_res2 = torch.ops.kvcache_manager_ops.lookup(uids, seqlens, evict_res)
            # Allocate (new)
            alloc_result2 = torch.ops.kvcache_manager_ops.allocate(
                uids, seqlens, lookup_res2[1], lookup_res2[5]
            )

            # Onboard launch
            onboard_slot_mappings = torch.ops.kvcache_manager_ops.onboard_launch(
                uids, seqlens, lookup_res2, alloc_result2[0], alloc_result2[2]
            )
            masked_onboard_task_ids = onboard_slot_mappings[2]

            # Onboard wait
            cache_tables = torch.ops.kvcache_manager_ops.onboard_wait(
                masked_onboard_task_ids,
                alloc_result2[0],
            )

        return (
            lookup_res,
            alloc_result,
            offload_task_ids,
            reap_result,
            flush_result,
            lookup_res_,
            evict_res,
            lookup_res2,
            alloc_result2,
            onboard_slot_mappings,
            cache_tables,
        )


def test_print(
    uids,
    lookup_res,
    alloc_result,
    offload_task_ids,
    reap_result,
    flush_result,
    lookup_res_,
    lookup_res2,
    alloc_result2,
    onboard_slot_mappings,
    cache_tables,
):
    # Lookup
    print("[CHECK] Lookup result:")
    print(f"\tMerged cached startpos: {lookup_res[0]}")
    print(f"\tMerged cached lengths: {lookup_res[1]}")
    print(f"\tGPU cached startpos: {lookup_res[2]}")
    print(f"\tGPU cached lengths: {lookup_res[3]}")
    print(f"\tHost cached startpos: {lookup_res[4]}")
    print(f"\tHost cached lengths: {lookup_res[5]}")

    # Allocate
    print(f"[CHECK] Allocate result:")
    print(f"\tPage indices: {alloc_result[0]}")
    print(f"\tPage indptr: {alloc_result[2]}")

    # Offload launch
    print(f"[CHECK] Offload result:")
    print(f"\tTask IDs: {offload_task_ids}")

    # Reap completed offload tasks
    print(f"[CHECK] Offload reap result: {reap_result}, shape: {reap_result.shape}")
    # print(f"[CHECK] Offload flush result: {flush_result}, shape: {flush_result.shape}")

    # Lookup
    print("[CHECK] Lookup result:")
    print(f"\tMerged cached startpos: {lookup_res_[0]}")
    print(f"\tMerged cached lengths: {lookup_res_[1]}")

    print(f"\tGPU cached startpos: {lookup_res_[2]}")
    print(f"\tGPU cached lengths: {lookup_res_[3]}")
    print(f"\tHost cached startpos: {lookup_res_[4]}")
    print(f"\tHost cached lengths: {lookup_res_[5]}")

    # Evict GPU cache only
    print("[CHECK] Evicted GPU cache for user_ids:", uids)

    # Lookup
    print("[CHECK] Lookup result:")
    print(f"\tMerged cached startpos: {lookup_res2[0]}")
    print(f"\tMerged cached lengths: {lookup_res2[1]}")
    print(f"\tGPU cached startpos: {lookup_res2[2]}")
    print(f"\tGPU cached lengths: {lookup_res2[3]}")
    print(f"\tHost cached startpos: {lookup_res2[4]}")
    print(f"\tHost cached lengths: {lookup_res2[5]}")

    # Allocate (new)
    print(f"[CHECK] Allocate result:")
    print(f"\tPage indices: {alloc_result2[0]}")
    print(f"\tPage indptr: {alloc_result2[2]}")

    # Onboard launch
    print(f"[CHECK] Onboard result:")
    print(
        f"\tLaunched tasks: {lookup_res2[6]}"
    )  # Onboard task IDs returned in lookup result
    print(f"\tSlot mappings: {onboard_slot_mappings}")

    # Onboard wait
    print(f"[CHECK] Onboard tasks completed: {lookup_res2[6]}")
    print(f"[CHECK] Cache tables after onboard wait: {len(cache_tables)} tensors")
    return


if __name__ == "__main__":
    model_config, cache_config = get_config(
        str(PACKAGE_ROOT / "test" / "bindings_test_config.yml")
    )
    set_export_kvcache_runtime_env(model_config, cache_config)
    server_recv_port = _ipc_endpoint("flexkv_server_sock")
    gpu_register_port = server_recv_port + "_gpu_register"
    os.environ["SERVER_RECV_PORT"] = server_recv_port
    os.environ["GPU_REGISTER_PORT"] = gpu_register_port

    server_handle = KVServer.create_server(
        model_config=model_config,
        cache_config=cache_config,
        gpu_register_port=gpu_register_port,
        server_recv_port=server_recv_port,
        total_clients=1,
        inherit_env=True,
    )
    print("[CHECK] Started KVServer")
    time.sleep(3)  # Wait a bit for server to be fully ready

    test_model = KVCacheManagerTestModel()

    # Warmup
    tokens = torch.arange(0, 100, dtype=torch.int64)
    uids = torch.tensor([11, 12], dtype=torch.int64)
    seqlens = torch.tensor([tokens.size(0), tokens.size(0)], dtype=torch.int64)

    (
        lookup_res,
        alloc_result,
        offload_task_ids,
        reap_result,
        flush_result,
        lookup_res_,
        evict_res,
        lookup_res2,
        alloc_result2,
        onboard_slot_mappings,
        cache_tables,
    ) = test_model(seqlens, uids)
    del evict_res
    test_print(
        uids,
        lookup_res,
        alloc_result,
        offload_task_ids,
        reap_result,
        flush_result,
        lookup_res_,
        lookup_res2,
        alloc_result2,
        onboard_slot_mappings,
        cache_tables,
    )

    # Export
    register_fake_kvcache_manager_ops()

    uids = torch.tensor([23, 24], dtype=torch.int64)
    # num_tokens = Dim("num_tokens", min=1, max=40000)
    batch = Dim("batch", min=1, max=8)
    dynamic_shapes = {
        "seqlens": {0: batch},
        "uids": {0: batch},
    }

    save_dir = PACKAGE_ROOT
    exported = torch.export.export(
        test_model, (seqlens, uids), dynamic_shapes=dynamic_shapes
    )
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=os.path.join(save_dir, "model.pt2"),
        inductor_configs={},
    )
    print(
        f"[CHECK] Exported and compiled model to {os.path.join(save_dir, 'model.pt2')}."
    )

    # Test
    export_dir = PACKAGE_ROOT
    compiled_model = torch._inductor.aoti_load_package(
        os.path.join(export_dir, "model.pt2")
    )
    print(
        f"[CHECK] Loaded compiled model from {os.path.join(export_dir, 'model.pt2')}."
    )

    seqlens = torch.tensor([100], dtype=torch.int64)
    uids = torch.tensor([37], dtype=torch.int64)
    with torch.no_grad():
        (
            lookup_res,
            alloc_result,
            offload_task_ids,
            reap_result,
            reap_result2,
            lookup_res_,
            evict_res,
            lookup_res2,
            alloc_result2,
            onboard_slot_mappings,
            cache_tables,
        ) = compiled_model((seqlens, uids))
        del evict_res
        test_print(
            uids,
            lookup_res,
            alloc_result,
            offload_task_ids,
            reap_result,
            reap_result2,
            lookup_res_,
            lookup_res2,
            alloc_result2,
            onboard_slot_mappings,
            cache_tables,
        )

    server_handle.shutdown()
    start_time = time.time()
    while server_handle._is_alive():
        time.sleep(1)
        if time.time() - start_time > 10:
            print("[ERROR] Server did not shut down within 10 seconds, forcing exit")
            break
    print("[CHECK] KVServer shutdown complete")
