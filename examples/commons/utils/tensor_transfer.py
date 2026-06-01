# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any as _Any
from typing import Optional as _Optional
from typing import Union as _Union

import torch as _torch


def _to_device(
    tensor: _torch.Tensor,
    device: _torch.device,
    dtype: _Optional[_torch.dtype] = None,
    non_blocking: bool = False,
) -> _torch.Tensor:
    if dtype is None:
        return tensor.to(device=device, non_blocking=non_blocking)
    return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)


def copy_tensor_to_pinned_cpu(tensor: _torch.Tensor) -> _torch.Tensor:
    """Copy a tensor to pinned CPU memory."""
    source = tensor.detach()
    if source.device.type == "cpu" and source.is_pinned():
        return source
    copy_source = (
        source if source.is_cuda or source.device.type == "cpu" else source.cpu()
    )
    cpu = _torch.empty(
        tuple(copy_source.shape),
        dtype=copy_source.dtype,
        device=_torch.device("cpu"),
        pin_memory=True,
    )
    cpu.copy_(copy_source, non_blocking=source.is_cuda)
    if source.is_cuda:
        _torch.cuda.current_stream(source.device).synchronize()
    return cpu


def tensor_to_cpu_list(tensor: _torch.Tensor) -> _Any:
    if tensor.is_cuda:
        return copy_tensor_to_pinned_cpu(tensor).tolist()
    source = tensor.detach()
    return source.tolist() if source.device.type == "cpu" else source.cpu().tolist()


def tensor_from_cpu_array_like(
    values: _Any,
    *,
    dtype: _Optional[_torch.dtype] = None,
    device: _Optional[_Union[_torch.device, str]] = None,
    pinned_staging: bool = True,
    non_blocking: bool = True,
) -> _torch.Tensor:
    """Create a tensor from CPU-side data, using pinned staging for CUDA copies."""
    target_device = _torch.device("cpu" if device is None else device)

    if not isinstance(values, _torch.Tensor):
        if target_device.type == "cuda" and pinned_staging:
            # Build pinned host memory directly, then enqueue a non-blocking H2D
            # copy. With pinned staging, cudaMemcpyAsync can return to the host
            # after submitting the copy instead of waiting for the copy to finish.
            # The returned CUDA tensor must be consumed on the same current stream.
            pinned = _torch.tensor(
                values,
                dtype=dtype,
                device=_torch.device("cpu"),
                pin_memory=True,
            )
            return pinned.to(device=target_device, non_blocking=non_blocking)
        return _torch.as_tensor(values, dtype=dtype, device=target_device)

    source = values.detach()
    if source.device.type != "cpu":
        raise ValueError("tensor_from_cpu_array_like expects CPU-side data")

    if target_device.type != "cuda":
        return _to_device(source, target_device, dtype=dtype)

    if not pinned_staging:
        return _to_device(
            source,
            target_device,
            dtype=dtype,
            non_blocking=non_blocking,
        )

    target_dtype = source.dtype if dtype is None else dtype
    if source.is_pinned() and source.dtype == target_dtype:
        pinned = source
    else:
        pinned = _torch.empty(
            tuple(source.shape),
            dtype=target_dtype,
            device=_torch.device("cpu"),
            pin_memory=True,
        )
        pinned.copy_(source)
    # With pinned staging, cudaMemcpyAsync can return to the host after
    # submitting the H2D copy instead of waiting for the copy to finish. The
    # copy is ordered only on the current CUDA stream, so callers must consume
    # the returned tensor on the same stream.
    return pinned.to(device=target_device, non_blocking=non_blocking)
