# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GATE E2 -- growing a HOST value buffer in place (``HostVMMTensor``) under load.

``HostVMMTensor`` reserves its address range when it is created and a growth commits, locks
and registers (``cudaHostRegister``) only the new tail; nothing is unregistered or moved until
the buffer is destroyed. What hardware has to say:

* (a) does ``cudaHostRegister`` of a tail WAIT for kernels in flight that read the buffer?
  A train of kernels on stream A reads and writes the whole old buffer through the mapped
  pointer for about ``GATE_E2_KERNEL_MS`` (1.8 s, far longer than any growth) while the host
  thread extends by 64 MiB. The call is BLOCKED when it returned after A had finished, or took
  more than half of the time A had left when it was called (a call that waits takes about
  that; one that does not takes what pinning the pages needs, touch + ``mlock`` + registration,
  tens of milliseconds for 64 MiB, which is printed and is not a criterion). PASS when at most
  1% of the iterations are blocked. ``test_e2a_growth_keeps_the_data`` is the same with short
  kernels (``GATE_E2_DATA_KERNEL_MS``, 100 ms) and no synchronize between the iterations: PASS
  when there is no CUDA error, every old element has had each of A's adds once, and every
  tail holds what a kernel on stream B wrote right after the growth that made it.
* (b) is a buffer registered in many pieces one contiguous range for the kernels, and does its
  address stay put? 1000 growths of one to three pages, the address checked after every one;
  then kernels gather, write and gather again across every boundary of the pieces and at
  random. (A kernel is meant: the driver does not follow a ``cudaMemcpyAsync`` from one
  registered piece into the next; (f) prints what it does as information, and
  ``test_a_copy_across_two_pieces_is_refused_cleanly...`` of ``test_vmm_tensor.py`` pins it
  down.)
* (c) is a buffer in 20 pieces as fast as one in a single registration? The random-gather
  throughput of a 1 GiB buffer built both ways must agree within 5%.
* (d) can 128 tables reserve their address space (64 GiB each, 8 TiB in all)? The reservation
  halves when the kernel refuses; PASS when every table was made, with at least its size.
* (e) prints ``cudaDevAttrCanUseHostPointerForRegisteredMem`` (informational).

Run it (one H100 node, ``ulimit -l unlimited``, the extension built with ``DEMB_ARCHS=90``)::

    DEMB_GROWTH_GATES=1 python -m pytest -s -q \\
        corelib/dynamicemb/test/unit_tests/growth/gates/test_gate_e2_host_in_place.py
    # what the driver was asked: one cudaHostRegister per growth, no cudaHostUnregister
    # while the buffer lives
    DEMB_GROWTH_GATES=1 GATE_E2_ITERS=6 nsys profile --trace=cuda -o e2 python -m pytest -s -q \\
        -k e2a_growth_does_not_wait <the file>; nsys stats --report cuda_api_sum e2.nsys-rep | grep -i hostregister
    DEMB_GROWTH_GATES=1 GATE_E2_DATA_ITERS=16 GATE_E2_DATA_KERNEL_MS=20 compute-sanitizer --tool memcheck \\
        --error-exitcode 99 python -m pytest -s -q -k "keeps_the_data or e2b" <the file>

Every verdict is one printed line: ``E2 PASS ...`` or ``E2 FAIL ...``.
"""

import ctypes
import glob
import os
import statistics
import sys
import time

import pytest
import torch
from dynamicemb import extendable_tensor
from dynamicemb_extensions import HostVMMTensor

pytestmark = pytest.mark.skipif(
    os.environ.get("DEMB_GROWTH_GATES") != "1"
    or not torch.cuda.is_available()
    or not extendable_tensor._HOST_IN_PLACE,
    reason="a hardware gate on an extension that grows host buffers in place: "
    "set DEMB_GROWTH_GATES=1 on a GPU",
)

MB = 1 << 20
PAGE = os.sysconf("SC_PAGE_SIZE")
FPP = PAGE // 4  # float32 elements per page
# the wait run: a kernel of seconds, a synchronize after every iteration
ITERS = int(os.environ.get("GATE_E2_ITERS", "12"))
KERNEL_MS = float(os.environ.get("GATE_E2_KERNEL_MS", "1800"))
# the data run: short kernels, nothing waits in between
DATA_ITERS = int(os.environ.get("GATE_E2_DATA_ITERS", "40"))
DATA_KERNEL_MS = float(os.environ.get("GATE_E2_DATA_KERNEL_MS", "100"))
GROW_MB = int(os.environ.get("GATE_E2_GROW_MB", "64"))
RECREATE_EVERY = int(os.environ.get("GATE_E2_RECREATE_EVERY", "8"))
OLD_MB = int(os.environ.get("GATE_E2_OLD_MB", "256"))
MAX_LAUNCHES = 900  # a stream that has more queued makes the host wait in the launch
MAX_BLOCKED = 0.01
BLOCKED_SHARE = 0.5  # a blocked call took more than this share of what A had left


def verdict(ok: bool, label: str, detail: str = "") -> None:
    print(
        "E2 %s %s%s"
        % ("PASS" if ok else "FAIL", label, (": " + detail) if detail else "")
    )
    assert ok, "E2 failed: %s %s" % (label, detail)


def cuda_attribute(device: int, attribute: int):
    names = ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    names += glob.glob(
        os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*")
    )
    for name in names:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        value = ctypes.c_int(-1)
        if lib.cudaDeviceGetAttribute(ctypes.byref(value), attribute, device) == 0:
            return value.value
    return None


def calibrate_adds(base: torch.Tensor, target_ms: float) -> int:
    """How many ``add_`` over ``base`` take about ``target_ms`` (at most MAX_LAUNCHES)."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        base.add_(1.0)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record(stream)
        for _ in range(3):
            base.add_(1.0)
        end.record(stream)
    end.synchronize()
    base.sub_(4.0)
    return max(
        1, min(MAX_LAUNCHES, int(target_ms / max(start.elapsed_time(end) / 3, 1e-3)))
    )


def percentile(sorted_values, share: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(share * len(sorted_values)))]


# --- (a) --------------------------------------------------------------------------------


def run_a(wait: bool):
    """``wait``: one long kernel A per iteration and a synchronize after it, the share of the
    blocked calls is the verdict; otherwise short kernels piling up, the data is."""
    iters = ITERS if wait else DATA_ITERS
    target_ms = KERNEL_MS if wait else DATA_KERNEL_MS
    device = torch.cuda.current_device()
    n_old, grow = OLD_MB * MB // 4, GROW_MB * MB // 4
    reserve = n_old + (RECREATE_EVERY + 1) * grow
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    latencies, kernels, blocked, problems = [], [], 0, []
    buf = base = None
    k = 0
    adds, fills, total = 0, [], 0

    def check():
        torch.cuda.synchronize()
        if not bool((base[:n_old] == float(adds)).all()):
            problems.append("an old element does not hold %d" % adds)
        view = buf.data()
        for lo, hi, value in fills:
            if not bool((view[lo:hi] == value).all()):
                problems.append("the tail [%d, %d) does not hold %s" % (lo, hi, value))
                break

    try:
        for it in range(iters):
            if buf is None or it % RECREATE_EVERY == 0:
                if buf is not None:
                    check()
                torch.cuda.synchronize()
                buf = HostVMMTensor(n_old, torch.float32, device, reserve)
                base = buf.data()
                base.zero_()
                torch.cuda.synchronize()
                total, adds, fills = n_old, 0, []
                k = k or calibrate_adds(base, target_ms)
            if wait:
                torch.cuda.synchronize()  # kernel A starts on an idle device
            launched = time.perf_counter()
            with torch.cuda.stream(stream_a):
                a_start = torch.cuda.Event(enable_timing=True)
                a_start.record(stream_a)
                for _ in range(k):
                    base.add_(1.0)
                a_end = torch.cuda.Event(enable_timing=True)
                a_end.record(stream_a)
            adds += k
            start = time.perf_counter()
            total += grow
            buf.extend(total)
            latency = time.perf_counter() - start
            finished = a_end.query()
            latencies.append(latency)
            value = float(it % 97 + 1)
            with torch.cuda.stream(stream_b):
                buf.data()[total - grow : total].fill_(value)
            fills.append((total - grow, total, value))
            if wait:
                torch.cuda.synchronize()
                kernel_s = a_start.elapsed_time(a_end) / 1e3
                kernels.append(kernel_s * 1e3)
                left = kernel_s - (start - launched)
                blocked += int(finished or latency > BLOCKED_SHARE * left)
            else:
                blocked += int(finished)
        if buf is not None:
            check()
    except Exception as exc:  # noqa: BLE001 -- an error is the verdict
        problems.append("%s: %s" % (type(exc).__name__, str(exc)[:300]))
    torch.cuda.synchronize()
    lat = sorted(latencies) or [0.0]
    fraction = blocked / max(1, len(latencies))
    if wait and fraction > MAX_BLOCKED:
        problems.append(
            "extend was blocked (it returned after kernel A had finished, or took more than "
            "half of what A had left) in %.1f%% of the iterations (at most %.0f%%)"
            % (100 * fraction, 100 * MAX_BLOCKED)
        )
    verdict(
        not problems,
        "(a) %s grow=%dMiB iters=%d %s=%.2f%% extend: median=%.2fms p99=%.2fms max=%.2fms"
        " (kernel A %.0fms)"
        % (
            "wait" if wait else "data",
            GROW_MB,
            len(latencies),
            # in a data run kernel A is short, and a slow call can return after it without
            # having waited for it: the share is information there
            "blocked" if wait else "returned_after_A(info)",
            100 * fraction,
            statistics.median(lat) * 1e3,
            percentile(lat, 0.99) * 1e3,
            lat[-1] * 1e3,
            statistics.median(kernels) if kernels else target_ms,
        ),
        "; ".join(problems),
    )


def test_e2a_growth_does_not_wait_for_a_running_kernel():
    run_a(wait=True)


def test_e2a_growth_keeps_the_data():
    run_a(wait=False)


# --- (b) --------------------------------------------------------------------------------


def test_e2b_pieces_are_one_contiguous_range_and_the_address_never_changes():
    device = torch.cuda.current_device()
    growths = 1000
    reserve = (growths * 3 + 8) * FPP
    buf = HostVMMTensor(FPP, torch.float32, device, reserve)
    pointer = buf.data().data_ptr()
    total, boundaries, problems = FPP, [], []
    for g in range(growths):
        total += (1 + g % 3) * FPP
        buf.extend(total)
        boundaries.append(buf.allocated_numel())
        if buf.data().data_ptr() != pointer:
            problems.append("the base address changed at growth %d" % g)
            break
    view = buf.data()
    n = view.numel()
    assert n < (1 << 24), "float32 indices must stay exact"
    # kernels (not ``copy_``, which is a cudaMemcpyAsync: see the end of this test)
    view.mul_(0.0).add_(torch.arange(n, device="cuda", dtype=torch.float32))
    torch.cuda.synchronize()

    windows = torch.unique(
        torch.cat(
            [
                torch.arange(max(0, b - 1024), min(n, b + 1024), device="cuda")
                for b in boundaries[:-1]
            ]
        )
    )
    if not torch.equal(view[windows], windows.to(torch.float32)):
        problems.append("a gather across the boundary of two pieces read wrong values")
    view.index_copy_(0, windows, windows.to(torch.float32) * 2)
    torch.cuda.synchronize()
    if not torch.equal(view[windows], windows.to(torch.float32) * 2):
        problems.append("a write across the boundary of two pieces was lost")
    rest = torch.ones(n, dtype=torch.bool, device="cuda")
    rest[windows] = False
    if not torch.equal(
        view[rest], torch.arange(n, device="cuda", dtype=torch.float32)[rest]
    ):
        problems.append("a write across a boundary changed an element outside it")
    random = torch.randint(0, n, (1 << 22,), device="cuda")
    expected = torch.where(
        rest[random], random.to(torch.float32), random.to(torch.float32) * 2
    )
    if not torch.equal(view[random], expected):
        problems.append("a random gather over all pieces read wrong values")
    verdict(
        not problems,
        "(b) %d growths, %d pieces' boundaries, address stable"
        % (growths, len(boundaries)),
        "; ".join(problems),
    )


def test_e2f_a_copy_across_a_boundary_is_information():
    """What a copy engine operation across the boundary of two pieces does, printed. A test
    of its own because ``compute-sanitizer`` reports the driver's refusal as an API error,
    which would fail the sanitizer run of (b)."""
    device = torch.cuda.current_device()
    buf = HostVMMTensor(FPP, torch.float32, device, 4 * FPP)
    buf.extend(2 * FPP)
    view = buf.data()
    try:
        view[FPP - 8 : FPP + 8].copy_(torch.zeros(16, device="cuda"))
        torch.cuda.synchronize()
        outcome = "worked"
    except RuntimeError as exc:
        outcome = "refused (%s)" % str(exc).splitlines()[0][:60]
    print("E2 INFO (f) cudaMemcpyAsync across the boundary of two pieces: %s" % outcome)


# --- (c) --------------------------------------------------------------------------------


def gather_time_ms(view: torch.Tensor, idx: torch.Tensor, reps: int = 15) -> float:
    view[idx]  # warm up
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        view[idx]
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def test_e2c_twenty_pieces_are_as_fast_as_one_registration():
    device = torch.cuda.current_device()
    size = (1 << 30) // 4  # 1 GiB of float32
    pieces = 20
    one = HostVMMTensor(size, torch.float32, device, size)
    many = HostVMMTensor(size // pieces, torch.float32, device, size)
    for p in range(2, pieces + 1):
        many.extend(p * (size // pieces))
    many.extend(size)
    idx = torch.randint(0, size, (1 << 24,), device="cuda")
    t_one = gather_time_ms(one.data(), idx)
    t_many = gather_time_ms(many.data(), idx)
    ratio = t_many / t_one
    verdict(
        ratio <= 1.05,
        "(c) random gather of 1 GiB: one registration %.3f ms, %d pieces %.3f ms, ratio %.3f"
        % (t_one, pieces, t_many, ratio),
        "" if ratio <= 1.05 else "slower than 5%",
    )


# --- (d) --------------------------------------------------------------------------------


def vm_kb(field: str) -> int:
    for line in open("/proc/self/status"):
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return -1


def test_e2d_128_tables_reserve_their_address_space():
    import resource

    if resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY:
        pytest.skip("RLIMIT_AS is set: the reservation halves by design")
    device = torch.cuda.current_device()
    reserve_floats = (64 << 30) // 4
    before = vm_kb("VmSize")
    tables = [
        HostVMMTensor(FPP, torch.float32, device, reserve_floats) for _ in range(128)
    ]
    after = vm_kb("VmSize")
    reserved = [t.reserved_bytes() for t in tables]
    ok = all(r >= PAGE for r in reserved)
    verdict(
        ok,
        "(d) 128 tables: VmSize %+d GiB, VmPeak %d GiB, each reserved %.1f..%.1f GiB"
        % (
            (after - before) // (1 << 20),
            vm_kb("VmPeak") // (1 << 20),
            min(reserved) / (1 << 30),
            max(reserved) / (1 << 30),
        ),
    )
    del tables


# --- (e) --------------------------------------------------------------------------------


def test_e2e_attribute():
    value = cuda_attribute(torch.cuda.current_device(), 91)
    print("E2 INFO (e) cudaDevAttrCanUseHostPointerForRegisteredMem = %s" % value)


if __name__ == "__main__":
    os.environ.setdefault("DEMB_GROWTH_GATES", "1")
    sys.exit(pytest.main(["-s", "-q", __file__]))
