# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared JSON-like payload validation helpers."""

from __future__ import annotations

from typing import Any, Mapping


def required_field(payload: Mapping[str, Any], name: str) -> Any:
    if name not in payload:
        raise ValueError(f"missing required field: {name}")
    return payload[name]


def required_str(payload: Mapping[str, Any], name: str) -> str:
    value = required_field(payload, name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def required_int(payload: Mapping[str, Any], name: str) -> int:
    value = required_field(payload, name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def optional_int(payload: Mapping[str, Any], name: str) -> int | None:
    if name not in payload or payload[name] is None:
        return None
    if isinstance(payload[name], bool):
        raise ValueError(f"{name} must be an integer")
    return int(payload[name])


def payload_list(
    payload: Mapping[str, Any],
    name: str,
) -> tuple[Mapping[str, Any], ...]:
    values = required_field(payload, name)
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    if not all(isinstance(value, Mapping) for value in values):
        raise ValueError(f"{name} must contain JSON objects")
    return tuple(values)
