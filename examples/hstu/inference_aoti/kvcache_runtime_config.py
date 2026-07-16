import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml  # type: ignore[import-untyped]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_KVCACHE_CONFIG_FILE = SCRIPT_DIR / "kvcache_cpp_runtime.yaml"


def resolve_kvcache_config_file(config_file: Optional[str] = None) -> Path:
    path = config_file or os.environ.get("KVCACHE_MANAGER_CONFIG_FILE")
    return Path(path).resolve() if path else DEFAULT_KVCACHE_CONFIG_FILE.resolve()


def load_kvcache_runtime_yaml(
    config_file: Optional[str] = None,
) -> tuple[Dict[str, Any], Path]:
    path = resolve_kvcache_config_file(config_file)
    with path.open("r", encoding="utf-8") as config_stream:
        config = yaml.safe_load(config_stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"KV-cache config must be a YAML mapping: {path}")
    return config, path


def required_config(config: Dict[str, Any], name: str) -> Any:
    value = config.get(name)
    if value is None or value == "":
        raise ValueError(f"Missing required KV-cache config value: {name}")
    return value


def config_int(config: Dict[str, Any], name: str) -> int:
    try:
        return int(required_config(config, name))
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer, got {config.get(name)!r}"
        ) from exc


def config_optional_int(config: Dict[str, Any], name: str) -> Optional[int]:
    value = config.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def config_float(config: Dict[str, Any], name: str, default: float) -> float:
    value = config.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def config_bool(config: Dict[str, Any], name: str) -> Optional[bool]:
    value = config.get(name)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def config_torch_dtype(config: Dict[str, Any]) -> torch.dtype:
    value = str(required_config(config, "dtype")).strip().lower()
    if value in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    if value in ("float16", "fp16", "half", "torch.float16"):
        return torch.float16
    raise ValueError(f"dtype must be bfloat16 or float16, got {value!r}")


def ipc_socket_path(endpoint: str) -> Optional[Path]:
    prefix = "ipc://"
    if not endpoint.startswith(prefix):
        return None
    return Path(endpoint[len(prefix) :])


def reset_ipc_socket(endpoint: str) -> None:
    socket_path = ipc_socket_path(endpoint)
    if socket_path is None:
        return
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass


def extra_flexkv_configs(config: Dict[str, Any]) -> Dict[str, Any]:
    configs: Dict[str, Any] = {}
    for name in ("enable_p2p_cpu", "enable_p2p_ssd", "enable_3rd_remote"):
        value = config_bool(config, name)
        if value is not None:
            configs[name] = value
    for name in ("redis_host", "local_ip", "redis_password"):
        value = config.get(name)
        if value:
            configs[name] = str(value)
    redis_port = config_optional_int(config, "redis_port")
    if redis_port is not None:
        configs["redis_port"] = redis_port
    return configs
