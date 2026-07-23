# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime config helpers shared by Python-side KV-cache export shims."""

from __future__ import annotations

import os
from pathlib import Path


def _strip_yaml_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _remove_yaml_comment(line: str) -> str:
    in_single_quote = False
    in_double_quote = False
    for index, char in enumerate(line):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            return line[:index]
    return line


def load_runtime_config_from_env(
    env_name: str = "KVCACHE_MANAGER_CONFIG_FILE",
) -> dict[str, str]:
    config_file = os.getenv(env_name)
    if not config_file:
        raise RuntimeError(f"{env_name} must be set to the KV-cache YAML config path")

    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"{env_name} does not exist: {path}")

    config: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = _remove_yaml_comment(raw_line).strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key:
                config[key] = _strip_yaml_quotes(value)
    return config


__all__ = ["load_runtime_config_from_env"]
