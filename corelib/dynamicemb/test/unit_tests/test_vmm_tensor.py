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
import subprocess
import sys
import time

import pytest
import torch
from dynamicemb import extendable_tensor
from dynamicemb.extendable_tensor import DeviceExtendableBuffer, HostExtendableBuffer

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# A host buffer that grows in place needs the extension built with it.
requires_in_place_host = pytest.mark.skipif(
    platform.system() != "Linux" or not extendable_tensor._HOST_IN_PLACE,
    reason="HostExtendableBuffer growth in place needs Linux and the rebuilt extension",
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


# Long enough that the work queued behind it still waits when the buffer is
# dropped: half a second of GPU clock, while the host reaches the drop within
# microseconds.
_GPU_DELAY_SEC = 0.5
_DROP_FILL_VALUE = 7
_DROP_DONE_MESSAGE = "queued work finished"
_DROP_CHILD_TIMEOUT_SEC = 300.0
_BUFFER_CLASSES_BY_NAME = {
    cls.__name__: cls for cls in (DeviceExtendableBuffer, HostExtendableBuffer)
}
# Where the queued work runs: on the current stream or on a side stream of the
# current GPU, or on GPU 1 while GPU 0 is current at the drop.
_QUEUED_ON_PARAMS = [
    "current_stream",
    "side_stream",
    pytest.param(
        "other_gpu",
        marks=pytest.mark.skipif(
            torch.cuda.device_count() < 2, reason="2 GPUs required"
        ),
    ),
]


def _gpu_delay_cycles(device: torch.device) -> int:
    # torch.cuda._sleep counts GPU clock cycles; clock_rate is in kHz.
    clock_khz = torch.cuda.get_device_properties(device).clock_rate
    return int(_GPU_DELAY_SEC * clock_khz * 1000)


def _drop_buffer_with_queued_gpu_work(buffer_cls, queued_on: str) -> None:
    """Drop a buffer while GPU work that uses it still waits in a stream.

    A long GPU-side sleep holds back a fill of the buffer and a copy out of it,
    so the last reference to the buffer goes away before they run. Freeing the
    buffer has to wait for them, on whichever stream and GPU they run; if it
    does not, they touch unmapped memory and fail with an illegal memory access.
    The drop must also leave the current GPU as it was.
    """
    if queued_on == "other_gpu":
        # Give GPU 0 an active context, so that it is the current GPU of the
        # driver, not only of PyTorch, when the buffer on GPU 1 is dropped.
        torch.cuda.set_device(0)
        torch.zeros(1, device="cuda:0")
        device = torch.device("cuda", 1)
    else:
        device = _cuda_device()
    with torch.cuda.device(device):
        buf = buffer_cls((1 << 20,), torch.int32, device)
        view = buf.tensor()
        stream = (
            torch.cuda.Stream(device)
            if queued_on == "side_stream"
            else torch.cuda.current_stream(device)
        )
        with torch.cuda.stream(stream):
            # Launch each kernel once first: with lazy loading, the first launch
            # of a kernel can wait for the kernels that are already running.
            torch.cuda._sleep(1)
            view.fill_(0)
            view.clone()
            torch.cuda.synchronize()

            torch.cuda._sleep(_gpu_delay_cycles(device))
            view.fill_(_DROP_FILL_VALUE)
            snapshot = view.clone()
    current_device = torch.cuda.current_device()
    del view, buf
    assert torch.cuda.current_device() == current_device, "the drop changed the GPU"
    torch.cuda.synchronize(device)
    assert torch.equal(snapshot, torch.full_like(snapshot, _DROP_FILL_VALUE))
    print(_DROP_DONE_MESSAGE)


@requires_cuda
@pytest.mark.parametrize("queued_on", _QUEUED_ON_PARAMS)
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
def test_drop_waits_for_queued_gpu_work(buffer_cls, queued_on):
    """Dropping a buffer must not free memory that queued GPU work still uses.

    An illegal memory access is sticky: it breaks the CUDA context for the rest
    of the process. So the case runs in a child process, and the test checks
    the child's exit status and output. The child gets the import path of this
    process, so it imports the same dynamicemb build as the test.
    """
    child = subprocess.run(
        [sys.executable, __file__, buffer_cls.__name__, queued_on],
        capture_output=True,
        text=True,
        timeout=_DROP_CHILD_TIMEOUT_SEC,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    output = child.stdout + child.stderr
    assert "illegal memory access" not in output, output
    assert child.returncode == 0, output
    assert _DROP_DONE_MESSAGE in child.stdout, output


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
            # (That was the state of extend when this was written. It no longer waits for the
            # device: test_extend_does_not_wait_for_a_running_kernel and the hardware gates
            # in growth/gates look at it under load. The synchronize stays so that a fault
            # shows in the iteration that caused it.)
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


@requires_cuda
@requires_in_place_host
def test_host_buffer_grows_in_place():
    """A host buffer commits pages inside the address range it reserved: the base address
    of its tensor is the same after every growth (and with it every pointer table a kernel
    holds), the values of the first pages are where they were, and a kernel can read and
    write across the boundaries of the pieces the range is registered in."""
    device = _cuda_device()
    buf = HostExtendableBuffer((1024,), torch.float32, device, max_shape=(1 << 16,))
    pointer = buf.tensor().data_ptr()
    assert buf.reserved_bytes() >= (1 << 16) * 4
    buf.tensor().copy_(torch.arange(1024, device=device, dtype=torch.float32))
    total, allocated = 1024, buf.allocated_bytes()
    for step in range(20):  # sub-page and multi-page growths
        buf.extend((1 + 150 * step,))
        total += 1 + 150 * step
        torch.cuda.synchronize()
        assert buf.tensor().data_ptr() == pointer
        assert buf.numel() == total
        assert buf.allocated_bytes() >= max(allocated, total * 4)
        allocated = buf.allocated_bytes()
    first = torch.arange(1024, device=device, dtype=torch.float32)
    torch.testing.assert_close(buf.tensor()[:1024], first)
    # Kernels over the whole buffer, which cross every boundary: the pieces are registered
    # one by one, and only a kernel is indifferent to that (a copy engine operation across
    # two pieces is refused by the driver, see
    # test_a_copy_across_two_pieces_is_refused_cleanly_and_kernels_are_not_affected).
    everything = buf.tensor().reshape(-1)
    indices = torch.arange(total, device=device, dtype=torch.float32)
    everything.mul_(0.0).add_(indices)
    everything.add_(1.0)
    torch.cuda.synchronize()
    assert torch.equal(buf.tensor().reshape(-1), indices + 1.0)
    # the whole buffer read through a kernel that makes its own tensor
    assert torch.equal(buf.tensor().reshape(-1) * 1.0, indices + 1.0)


@requires_cuda
@requires_in_place_host
def test_a_copy_across_two_pieces_is_refused_cleanly_and_kernels_are_not_affected():
    """A host buffer is registered with CUDA one piece per growth. A kernel addresses the
    range as the one range it is, but ``cudaMemcpyAsync`` (what ``copy_``, ``clone`` and
    ``cpu`` of a contiguous range come to) does not follow a copy from one registration
    into the next: the driver answers ``cudaErrorInvalidValue`` at the call
    (``CUDA_LOG_FILE=stderr`` says why: "Copy is larger than memobj size", the size of the
    registration the pointer is in). That is a limit of growing in place, measured on an
    H100. Nothing in the library copies a value buffer that way (rows are read and written
    by kernels through the pointer table); this says what happens if a caller does: an
    error that names the argument, and a context and a buffer that are as usable as before.
    A copy inside one piece, a memset and every kernel work across any number of boundaries.
    """
    device = _cuda_device()
    page = os.sysconf("SC_PAGE_SIZE")
    per_page = page // 4
    buf = HostExtendableBuffer(
        (per_page,), torch.float32, device, max_shape=(8 * per_page,)
    )
    buf.extend((per_page,))  # the second piece
    buf.extend((per_page,))  # the third
    view = buf.tensor().reshape(-1)
    source = torch.arange(3 * per_page, device=device, dtype=torch.float32)

    view.mul_(0.0).add_(source)  # kernels across both boundaries
    torch.cuda.synchronize()
    assert torch.equal(view, source)
    view[per_page : 2 * per_page].copy_(source[:per_page] + 1.0)  # inside one piece
    view.zero_()  # a memset
    torch.cuda.synchronize()
    assert not bool(view.any())

    try:
        view.copy_(source)
        torch.cuda.synchronize()
        refused = False
    except RuntimeError as exc:  # torch.AcceleratorError is one
        refused = True
        assert "invalid argument" in str(exc).lower(), exc
    if not refused:
        pytest.skip("this driver follows a copy from one registration into the next")

    # the refusal is clean: nothing was written, and kernels and one-piece copies go on
    assert not bool(view.any())
    view.add_(source)  # a kernel across both boundaries
    view[:per_page].copy_(source[:per_page])  # inside one piece
    torch.cuda.synchronize()
    assert torch.equal(view, source)


@requires_cuda
@requires_in_place_host
def test_host_buffer_cannot_outgrow_its_reservation_and_stays_usable():
    device = _cuda_device()
    buf = HostExtendableBuffer((1024,), torch.float32, device, max_shape=(4096,))
    pointer = buf.tensor().data_ptr()
    buf.tensor().fill_(3.0)
    torch.cuda.synchronize()

    with pytest.raises(RuntimeError, match="address space") as info:
        buf.extend((1 << 20,))
    # the error says what to turn
    assert "DYNAMICEMB_HOST_RESERVE_FACTOR" in str(info.value)

    assert buf.numel() == 1024 and buf.shape == (1024,)
    assert buf.tensor().data_ptr() == pointer
    buf.extend((1024,))  # still inside the reservation
    torch.cuda.synchronize()
    assert buf.numel() == 2048
    assert bool((buf.tensor().reshape(-1)[:1024] == 3.0).all())


@requires_cuda
def test_a_stale_extension_says_to_rebuild(monkeypatch):
    if platform.system() != "Linux":
        pytest.skip("HostExtendableBuffer is only supported on Linux")
    monkeypatch.setattr(extendable_tensor, "_HOST_IN_PLACE", False)
    buf = HostExtendableBuffer((16,), torch.float32, _cuda_device())
    with pytest.raises(RuntimeError, match="rebuild"):
        buf.extend((16,))
    assert buf.numel() == 16


_MEMLOCK_SCRIPT = """
import os, resource, sys, torch
from dynamicemb.extendable_tensor import HostExtendableBuffer

device = torch.device("cuda", torch.cuda.current_device())
page = os.sysconf("SC_PAGE_SIZE")
buf = HostExtendableBuffer((page // 4,), torch.float32, device, max_shape=(1 << 22,))
buf.tensor().fill_(5.0)
torch.cuda.synchronize()

def locked():
    for line in open("/proc/self/status"):
        if line.startswith("VmLck:"):
            return int(line.split()[1]) * 1024

limit = locked() + 4 * page
resource.setrlimit(resource.RLIMIT_MEMLOCK, (limit, resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]))
try:
    buf.extend((64 * page // 4,))  # 64 more pages: over the limit
except RuntimeError as exc:
    assert "mlock" in str(exc), exc
    print("REFUSED")
else:
    print("NOT_ENFORCED")  # root, or CAP_IPC_LOCK: the limit does not apply
    sys.exit(0)
assert buf.numel() == page // 4, buf.numel()
assert bool((buf.tensor() == 5.0).all())
buf.extend((page // 4,))  # one more page fits
torch.cuda.synchronize()
assert buf.numel() == 2 * (page // 4)
assert bool((buf.tensor().reshape(-1)[: page // 4] == 5.0).all())
print("USABLE")
"""


@requires_cuda
@requires_in_place_host
def test_a_failed_growth_leaves_the_buffer_usable():
    """RLIMIT_MEMLOCK refuses the pages of a growth: the error comes out of extend, the
    range it tried is address space again, the buffer has the size and the values it had,
    and a growth that fits works. In a child process: the limit is the process's."""
    child = subprocess.run(
        [sys.executable, "-c", _MEMLOCK_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300.0,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert child.returncode == 0, child.stdout + child.stderr
    if "NOT_ENFORCED" in child.stdout:
        pytest.skip("RLIMIT_MEMLOCK is not enforced for this user")
    assert "REFUSED" in child.stdout and "USABLE" in child.stdout, (
        child.stdout + child.stderr
    )


@requires_cuda
@pytest.mark.parametrize("buffer_cls", _EXTENDABLE_BUFFER_PARAMS)
@pytest.mark.parametrize("growth", [1 << 10, 1 << 22], ids=["small", "many_pages"])
def test_extend_does_not_wait_for_a_running_kernel(buffer_cls, growth):
    """``extend`` returns while a kernel that is running on the buffer has not finished.

    Whether a driver call waits for the kernels in flight is not something to read off a
    profiler's names, so the test looks at the time the host comes back: a long sleep runs
    on a stream, a kernel that reads and writes every old element is queued behind it, and
    the host grows the buffer meanwhile. If ``extend`` waited for the device the sleep
    would be over when it returned. The pages that were mapped before must not be hurt
    (the kernel after the sleep runs on them) and the new ones must be usable at once, by a
    kernel on another stream.
    """
    if buffer_cls is HostExtendableBuffer and not extendable_tensor._HOST_IN_PLACE:
        pytest.skip("growing a host buffer needs the rebuilt extension")
    device = _cuda_device()
    old = 1 << 20
    buf = buffer_cls((old,), torch.float32, device)
    stream, other = torch.cuda.Stream(), torch.cuda.Stream()
    view = buf.tensor().reshape(-1)
    # launch every kernel once first: with lazy loading the first launch of a kernel can
    # wait for the kernels that are already running
    view.zero_()
    view.add_(1.0)
    view.zero_()
    scratch = torch.empty(16, device=device)
    scratch.fill_(0.0)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        torch.cuda._sleep(_gpu_delay_cycles(device))
        view.add_(
            1.0
        )  # queued behind the sleep: runs on the pages after extend returned
        done = torch.cuda.Event()
        done.record(stream)
    buf.extend((growth,))
    returned_while_running = not done.query()
    tail = buf.tensor().reshape(-1)[old:]
    with torch.cuda.stream(other):
        tail.fill_(3.0)
    torch.cuda.synchronize()

    assert returned_while_running, "extend waited for the device"
    assert buf.numel() == old + growth
    everything = buf.tensor().reshape(-1)
    assert bool((everything[:old] == 1.0).all()), "the old pages were hurt by extend"
    assert bool((everything[old:] == 3.0).all()), "the new pages are not usable"


if __name__ == "__main__":
    # The child process of test_drop_waits_for_queued_gpu_work.
    _drop_buffer_with_queued_gpu_work(
        _BUFFER_CLASSES_BY_NAME[sys.argv[1]], queued_on=sys.argv[2]
    )
