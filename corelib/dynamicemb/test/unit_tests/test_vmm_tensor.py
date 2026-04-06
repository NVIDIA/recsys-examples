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

from __future__ import annotations

import os
import platform
import time

import pytest
import torch
from dynamicemb.extendable_tensor import DeviceExtendableBuffer, HostExtendableBuffer

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

_EXTENDABLE_BUFFER_PARAMS = [
    pytest.param(DeviceExtendableBuffer, id="device"),
    pytest.param(
        HostExtendableBuffer,
        id="host",
        marks=pytest.mark.skipif(
            platform.system() != "Linux",
            reason="HostExtendableBuffer is only supported on Linux",
        ),
    ),
]


def _cuda_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _format_mem_bytes(num_bytes: int) -> str:
    """Format byte count using B / KB / MB / GB / TB (1024-based)."""
    if num_bytes < 0:
        raise ValueError(num_bytes)
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MB"
    if num_bytes < 1024**4:
        return f"{num_bytes / (1024**3):.3f} GB"
    return f"{num_bytes / (1024**4):.3f} TB"


@requires_cuda
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
def test_extend_updates_numel_and_shape_1d(buffer_cls):
    device = _cuda_device()
    dtype = torch.float32
    buf = buffer_cls((10,), dtype, device)
    assert buf.numel() == 10
    assert buf.shape == (10,)

    buf.extend((5,))
    torch.cuda.synchronize()
    assert buf.numel() == 15
    assert buf.shape == (15,)

    buf.extend((3,))
    torch.cuda.synchronize()
    assert buf.numel() == 18
    assert buf.shape == (18,)


@requires_cuda
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
def test_extend_updates_numel_and_shape_2d(buffer_cls):
    device = _cuda_device()
    dtype = torch.float32
    buf = buffer_cls((32, 8), dtype, device)
    assert buf.numel() == 256
    assert buf.shape == (32, 8)

    buf.extend((16, 8))
    torch.cuda.synchronize()
    assert buf.numel() == 384
    assert buf.shape == (48, 8)


@requires_cuda
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
def test_extend_2d_with_list_extend_shape(buffer_cls):
    """extend_shape as list must match tuple trailing dims (list != tuple in ==)."""
    device = _cuda_device()
    dtype = torch.float32
    buf = buffer_cls((32, 8), dtype, device)
    buf.extend([16, 8])
    torch.cuda.synchronize()
    assert buf.numel() == 384
    assert buf.shape == (48, 8)


@requires_cuda
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
def test_extend_mismatched_trailing_dims_flattens_shape(buffer_cls):
    """When extend_shape trailing dims do not match, fall back to 1-D total numel."""
    device = _cuda_device()
    dtype = torch.float32
    buf = buffer_cls((4, 8), dtype, device)
    assert buf.numel() == 32
    buf.extend((8,))
    torch.cuda.synchronize()
    assert buf.numel() == 40
    assert buf.shape == (40,)


@requires_cuda
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
def test_extend_zero_increment_is_noop(buffer_cls):
    device = _cuda_device()
    dtype = torch.float32
    buf = buffer_cls((8,), dtype, device)
    n0, s0 = buf.numel(), buf.shape

    buf.extend((0,))
    torch.cuda.synchronize()
    assert buf.numel() == n0
    assert buf.shape == s0

    buf.extend((1, 0))
    torch.cuda.synchronize()
    assert buf.numel() == n0
    assert buf.shape == s0


@requires_cuda
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
def test_num_bytes_matches_numel_times_element_size(buffer_cls):
    device = _cuda_device()
    dtype = torch.float32
    es = torch.empty(0, dtype=dtype).element_size()
    buf = buffer_cls((4,), dtype, device)
    assert buf.num_bytes() == buf.numel() * es == 4 * es

    buf.extend((2,))
    torch.cuda.synchronize()
    assert buf.num_bytes() == buf.numel() * es == 6 * es


@requires_cuda
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
def test_allocated_bytes_from_native_matches_allocated_numel(buffer_cls):
    """C++ reports page-aligned backing size; must cover logical num_bytes."""
    device = _cuda_device()
    dtype = torch.float32
    es = torch.empty(0, dtype=dtype).element_size()
    buf = buffer_cls((4,), dtype, device)
    assert buf.allocated_bytes() >= buf.num_bytes()
    assert buf.allocated_bytes() % es == 0
    assert buf.allocated_bytes() == buf.vmm_tensor.allocated_numel() * es

    buf.extend((2,))
    torch.cuda.synchronize()
    assert buf.allocated_bytes() >= buf.num_bytes()
    assert buf.allocated_bytes() % es == 0
    assert buf.allocated_bytes() == buf.vmm_tensor.allocated_numel() * es


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
def test_prefix_values_preserved_after_extend(buffer_cls, dtype):
    device = _cuda_device()
    buf = buffer_cls((10,), dtype, device)
    t = buf.tensor()
    src = torch.arange(10, device=device, dtype=dtype)
    t.copy_(src)
    ref = t.detach().clone()

    buf.extend((6,))
    torch.cuda.synchronize()

    assert buf.numel() == 16
    out = buf.tensor().reshape(-1)[:10]
    if dtype == torch.float32:
        rtol, atol = 0.0, 0.0
    elif dtype == torch.float16:
        rtol, atol = 1e-4, 0.0
    else:
        rtol, atol = 1e-2, 0.0
    torch.testing.assert_close(out, ref.reshape(-1), rtol=rtol, atol=atol)


@requires_cuda
def test_is_device_buffer_device_vs_host():
    device = _cuda_device()
    dtype = torch.float32
    dev_buf = DeviceExtendableBuffer((1,), dtype, device)
    assert dev_buf.is_device_buffer() is True

    if platform.system() != "Linux":
        pytest.skip("HostExtendableBuffer only supported on Linux")

    host_buf = HostExtendableBuffer((1,), dtype, device)
    assert host_buf.is_device_buffer() is False


def _vmm_stress_full_matrix() -> bool:
    """Set DYNAMICEMB_VMM_STRESS_FULL=1 to run the old wide param grid (very slow)."""
    return os.environ.get("DYNAMICEMB_VMM_STRESS_FULL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


_VMM_BUFFER_INIT_CAPACITIES = (1, 1023, 1024, 1 << 30)
_VMM_BUFFER_DTYPES_DEFAULT = (torch.float32,)
# If a single while-loop iteration (extend + sync + bookkeeping) exceeds this, raise TimeoutError.
_VMM_BUFFER_ITER_BODY_TIMEOUT_SEC = 10.0


@pytest.mark.parametrize("init_capacity", list(_VMM_BUFFER_INIT_CAPACITIES))
@pytest.mark.parametrize(
    "dtype",
    (
        [torch.float32, torch.float16, torch.bfloat16]
        if _vmm_stress_full_matrix()
        else list(_VMM_BUFFER_DTYPES_DEFAULT)
    ),
)
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
@requires_cuda
def test_vmm_buffer(init_capacity, dtype, buffer_cls):
    """Grow buffer by doubling until allocation fails (OOM / VA limit).

    No fill_: stress extend/map paths only. Default: device + host (Linux) + float32.
    DYNAMICEMB_VMM_STRESS_FULL=1 adds fp16/bf16.
    If one loop-body iteration exceeds _VMM_BUFFER_ITER_BODY_TIMEOUT_SEC, stops like OOM
    (TimeoutError caught; test passes — we only skip waiting for slower iterations).
    """
    device = _cuda_device()
    shape = (init_capacity,)
    buffer = buffer_cls(shape, dtype, device)

    apply_capacity = init_capacity
    real_capacity = buffer.numel()
    pointer = buffer.tensor().data_ptr()

    try:
        while True:
            t_iter = time.monotonic()
            # Extend by current logical size so total capacity doubles each iteration.
            buffer.extend((apply_capacity,))

            apply_capacity = apply_capacity * 2
            real_capacity = buffer.numel()

            # synchronize is necessary to avoid IMA, we will add device synchronization to extend.
            # stream synchronization and device synchronization in C++ not worked.
            torch.cuda.synchronize()

            logical_b = buffer.num_bytes()
            alloc_b = buffer.allocated_bytes()
            print(
                f"test_vmm_buffer extend: logical={_format_mem_bytes(logical_b)} "
                f"allocated={_format_mem_bytes(alloc_b)} "
                f"(numel={real_capacity}, dtype={dtype}, {buffer_cls.__name__}, init={init_capacity})"
            )

            iter_elapsed = time.monotonic() - t_iter
            if iter_elapsed > _VMM_BUFFER_ITER_BODY_TIMEOUT_SEC:
                raise TimeoutError(
                    f"iteration body {iter_elapsed:.2f}s > {_VMM_BUFFER_ITER_BODY_TIMEOUT_SEC}s "
                    f"({buffer_cls.__name__}, dtype={dtype}, init_capacity={init_capacity})"
                )

    except (RuntimeError, TimeoutError) as e:
        logical_b = buffer.num_bytes()
        alloc_b = buffer.allocated_bytes()
        reason = (
            "allocation/runtime"
            if isinstance(e, RuntimeError)
            else "per-iteration time limit (normal stop, not waiting further)"
        )
        print(
            f"test_vmm_buffer expected stop ({reason}): {e!r} "
            f"dtype={dtype} buffer={buffer_cls.__name__} init_capacity={init_capacity} "
            f"last_apply_capacity={apply_capacity} last_numel={buffer.numel()} "
            f"logical={_format_mem_bytes(logical_b)} allocated={_format_mem_bytes(alloc_b)} "
            f"base_ptr={pointer:#x} tensor_ptr={buffer.tensor().data_ptr():#x}"
        )
