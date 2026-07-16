# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draft fake implementations for ``kvcache_manager_ops``.

These registrations are shape-only shims for export/fake-tensor flows. They do
not model mutable cache state, task lifecycle, or host/device cache coherence.
The goal is to preserve operator schemas and output structure closely enough for
``torch.export`` and other meta/fake tracing paths.

The C++ runtime currently determines several output lengths from hidden runtime
configuration, most notably the GPU page size and the number of newly allocated
tokens. Those lengths are represented here with fresh symbolic sizes.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Sequence, Tuple

import torch

_OPS_NAMESPACE = "kvcache_manager_ops"
_REGISTERED = False


def _has_op(op_name: str) -> bool:
    return hasattr(torch.ops.kvcache_manager_ops, op_name)


def _register_fake(op_name: str, func: Callable) -> None:
    torch.library.register_fake(f"{_OPS_NAMESPACE}::{op_name}")(func)


def _check_1d_cpu_tensor(tensor: torch.Tensor, name: str) -> None:
    torch._check(tensor.dim() == 1, lambda: f"{name} must be 1-D")
    torch._check(not tensor.is_cuda, lambda: f"{name} must be a CPU tensor")


def _check_same_length(
    lhs: torch.Tensor, rhs: torch.Tensor, lhs_name: str, rhs_name: str
) -> None:
    torch._check(
        lhs.shape[0] == rhs.shape[0],
        lambda: f"{lhs_name} and {rhs_name} must have the same batch dimension",
    )


def _new_dynamic_size(min_value: int = 0) -> int:
    return torch.library.get_ctx().new_dynamic_size(min=min_value)


def _empty_cpu(shape: Sequence[int], *, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(tuple(shape), dtype=dtype, device="cpu")


def _empty_cuda(shape: Sequence[int], *, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(tuple(shape), dtype=dtype, device="cuda")


def _fake_num_layers() -> int:
    value = os.getenv("KVCACHE_MANAGER_NUM_LAYERS")
    if value is None or value == "":
        return 1
    return max(int(value), 1)


def _fake_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return max(int(value), 1)


def _fake_cache_dtype() -> torch.dtype:
    value = os.getenv("KVCACHE_MANAGER_DTYPE", "bfloat16").lower()
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    return torch.bfloat16


def _fake_cache_table_shape() -> tuple[int, int, int, int, int]:
    return (
        _fake_env_int("KVCACHE_MANAGER_NUM_PRIMARY_CACHE_PAGES", 1),
        2,
        _fake_env_int("KVCACHE_MANAGER_TOKENS_PER_PAGE", 1),
        _fake_env_int("KVCACHE_MANAGER_NUM_KV_HEADS", 1),
        _fake_env_int("KVCACHE_MANAGER_HEAD_SIZE", 1),
    )


def _lookup_fake(
    user_ids: torch.Tensor, seqlens: torch.Tensor, sync_point: torch.Tensor
) -> list[torch.Tensor]:
    # _check_1d_cpu_tensor(user_ids, "user_ids")
    # _check_1d_cpu_tensor(seqlens, "seqlens")
    # _check_same_length(user_ids, seqlens, "user_ids", "seqlens")

    batch_size = user_ids.shape[0]
    int32_shape = (batch_size,)

    return [
        _empty_cpu(int32_shape, dtype=torch.int32),
        _empty_cpu(int32_shape, dtype=torch.int32),
        _empty_cpu(int32_shape, dtype=torch.int32),
        _empty_cpu(int32_shape, dtype=torch.int32),
        _empty_cpu(int32_shape, dtype=torch.int32),
        _empty_cpu(int32_shape, dtype=torch.int32),
        _empty_cpu((batch_size,), dtype=torch.int64),
    ]


def _allocate_fake(
    user_ids: torch.Tensor,
    seqlens: torch.Tensor,
    merged_cached_lengths: torch.Tensor,
    host_cached_lengths: torch.Tensor,
) -> list[torch.Tensor]:
    # _check_1d_cpu_tensor(user_ids, "user_ids")
    # _check_1d_cpu_tensor(seqlens, "seqlens")
    # _check_1d_cpu_tensor(merged_cached_lengths, "merged_cached_lengths")
    # _check_1d_cpu_tensor(host_cached_lengths, "host_cached_lengths")
    # _check_same_length(user_ids, seqlens, "user_ids", "seqlens")
    # _check_same_length(user_ids, merged_cached_lengths, "user_ids", "merged_cached_lengths")
    # _check_same_length(user_ids, host_cached_lengths, "user_ids", "host_cached_lengths")

    batch_size = user_ids.shape[0]
    num_total_pages = _new_dynamic_size()
    num_new_tokens = _new_dynamic_size()
    metadata_buffer_len = _new_dynamic_size()

    return [
        _empty_cuda((num_total_pages,), dtype=torch.int32),
        _empty_cuda((metadata_buffer_len,), dtype=torch.int32),
        _empty_cuda((batch_size + 1,), dtype=torch.int32),
        _empty_cuda((batch_size,), dtype=torch.int32),
        _empty_cuda((batch_size,), dtype=torch.int32),
        _empty_cuda((batch_size + 1,), dtype=torch.int32),
        _empty_cuda((batch_size + 1,), dtype=torch.int32),
        _empty_cuda((num_new_tokens,), dtype=torch.int32),
        _empty_cuda((num_new_tokens,), dtype=torch.int32),
        _empty_cpu((1,), dtype=torch.int32),
        _empty_cuda((1,), dtype=torch.int32),
        _empty_cpu((batch_size,), dtype=torch.int32),
        _empty_cpu((batch_size + 1,), dtype=torch.int32),
    ]


def _onboard_launch_fake(
    user_ids: torch.Tensor,
    seqlens: torch.Tensor,
    lookup_results: Sequence[torch.Tensor],
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_1d_cpu_tensor(user_ids, "user_ids")
    _check_1d_cpu_tensor(seqlens, "seqlens")
    # _check_same_length(user_ids, seqlens, "user_ids", "seqlens")
    torch._check(
        len(lookup_results) == 7, lambda: "lookup_results must contain 7 tensors"
    )
    _check_1d_cpu_tensor(lookup_results[6], "lookup_results[6]")
    torch._check(kv_page_indices.dim() == 1, lambda: "kv_page_indices must be 1-D")
    torch._check(kv_page_indptr.dim() == 1, lambda: "kv_page_indptr must be 1-D")
    # torch._check(
    #     kv_page_indptr.shape[0] == user_ids.shape[0] + 1,
    #     lambda: "kv_page_indptr must have shape [batch_size + 1]",
    # )

    batch_size = user_ids.shape[0]
    return (
        _empty_cpu((_new_dynamic_size(),), dtype=torch.int64),
        _empty_cpu((batch_size + 1,), dtype=torch.int64),
        _empty_cpu((batch_size,), dtype=torch.int64),
    )


def _onboard_wait_fake(
    task_ids: torch.Tensor,
    dummy_dependency: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    _check_1d_cpu_tensor(task_ids, "task_ids")
    del dummy_dependency
    return [
        _empty_cuda(_fake_cache_table_shape(), dtype=_fake_cache_dtype())
        for _ in range(_fake_num_layers())
    ]


def _offload_launch_fake(
    user_ids: torch.Tensor,
    seqlens: torch.Tensor,
    merged_cached_lengths: torch.Tensor,
    host_cached_lengths: torch.Tensor,
    gpu_cached_startpos: torch.Tensor,
    gpu_cached_lengths: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    slot_mappings: Sequence[torch.Tensor],
    dummy_dependency: torch.Tensor,
) -> torch.Tensor:
    # _check_1d_cpu_tensor(user_ids, "user_ids")
    # _check_1d_cpu_tensor(seqlens, "seqlens")
    # _check_1d_cpu_tensor(merged_cached_lengths, "merged_cached_lengths")
    # _check_1d_cpu_tensor(host_cached_lengths, "host_cached_lengths")
    # _check_1d_cpu_tensor(gpu_cached_startpos, "gpu_cached_startpos")
    # _check_1d_cpu_tensor(gpu_cached_lengths, "gpu_cached_lengths")
    # _check_same_length(user_ids, seqlens, "user_ids", "seqlens")
    # _check_same_length(user_ids, merged_cached_lengths, "user_ids", "merged_cached_lengths")
    # _check_same_length(user_ids, host_cached_lengths, "user_ids", "host_cached_lengths")
    # _check_same_length(user_ids, gpu_cached_startpos, "user_ids", "gpu_cached_startpos")
    # _check_same_length(user_ids, gpu_cached_lengths, "user_ids", "gpu_cached_lengths")
    # torch._check(kv_page_indices.dim() == 1, lambda: "kv_page_indices must be 1-D")
    # torch._check(kv_page_indptr.dim() == 1, lambda: "kv_page_indptr must be 1-D")
    # torch._check(
    #     len(slot_mappings) in (0, user_ids.shape[0]),
    #     lambda: "slot_mappings must be empty or match batch size",
    # )

    del dummy_dependency
    return _empty_cpu((user_ids.shape[0],), dtype=torch.int64)


def _offload_reap_completed_fake(dummy: torch.Tensor) -> torch.Tensor:
    # torch._check(dummy.dim() <= 1, lambda: "dummy must be a scalar or 1-D tensor")
    return _empty_cpu((_new_dynamic_size(), 3), dtype=torch.int64)


def _get_cache_tables_fake(dummy: torch.Tensor) -> list[torch.Tensor]:
    del dummy
    return [
        _empty_cuda(_fake_cache_table_shape(), dtype=_fake_cache_dtype())
        for _ in range(_fake_num_layers())
    ]


def _offload_wait_fake(task_ids: torch.Tensor) -> torch.Tensor:
    _check_1d_cpu_tensor(task_ids, "task_ids")
    return _empty_cpu((task_ids.shape[0], 3), dtype=torch.int64)


def _evict_kvcache_fake(
    user_ids: torch.Tensor, evict_gpu_only: bool, sync_point: torch.Tensor
) -> torch.Tensor:
    del evict_gpu_only
    _check_1d_cpu_tensor(user_ids, "user_ids")
    return _empty_cpu((1,), dtype=torch.int32)


def _init_kvcache_fake(dummy: torch.Tensor) -> int:
    # torch._check(dummy.dim() <= 1, lambda: "dummy must be a scalar or 1-D tensor")
    return 0


def _shutdown_runtime_fake(dummy: torch.Tensor) -> None:
    del dummy
    return None


def register_fake_kvcache_manager_ops() -> bool:
    global _REGISTERED

    if _REGISTERED:
        return True

    required_ops: Iterable[str] = (
        "init_kvcache",
        "shutdown_runtime",
        "lookup",
        "allocate",
        "onboard_launch",
        "onboard_wait",
        "offload_launch",
        "offload_reap_completed",
        "get_cache_tables",
        "offload_wait",
        "evict_kvcache",
    )
    if not all(_has_op(op_name) for op_name in required_ops):
        return False

    _register_fake("init_kvcache", _init_kvcache_fake)
    _register_fake("shutdown_runtime", _shutdown_runtime_fake)
    _register_fake("lookup", _lookup_fake)
    _register_fake("allocate", _allocate_fake)
    _register_fake("onboard_launch", _onboard_launch_fake)
    _register_fake("onboard_wait", _onboard_wait_fake)
    _register_fake("offload_launch", _offload_launch_fake)
    _register_fake("offload_reap_completed", _offload_reap_completed_fake)
    _register_fake("get_cache_tables", _get_cache_tables_fake)
    _register_fake("offload_wait", _offload_wait_fake)
    _register_fake("evict_kvcache", _evict_kvcache_fake)

    _REGISTERED = True
    return True


__all__ = ["register_fake_kvcache_manager_ops"]
