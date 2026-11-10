# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GATE E1 -- growing a DEVICE value buffer (``VMMTensor.extend``) without waiting for the device.

The question: may ``DeviceExtendableBuffer.extend`` drop its ``torch.cuda.synchronize()``?
``VMMTensor::extend`` maps new physical pages after the old ones (``cuMemCreate``, ``cuMemMap``,
``cuMemSetAccess`` of the new range only) and never unmaps, remaps or moves an old page. What
code cannot show is whether those driver calls wait for kernels in flight (``cuMemSetAccess``
carries the header note "exhibits synchronous behavior for most use cases"), whether a kernel
launched right after sees the new tail, and whether a kernel running on the old pages is hurt.

Two things are different things and are measured by different runs:

* WAITING for the device (``test_e1_extend_does_not_wait``). Kernel A is in flight on stream A
  for about ``GATE_E1_WAIT_KERNEL_MS`` (1.8 s, far longer than any ``extend``), either a
  ``torch.cuda._sleep`` (kind ``sleep``: no memory traffic) or a train of ``base.add_(1.0)``
  over a 4 GiB buffer that reads and writes every old element (kind ``memory``), and the host
  thread extends by 2, 6 or 64 MiB meanwhile (below a granule, a few granules, many). The
  call is BLOCKED when it returned after A had finished, or took more than half of the time
  A had left when it was called: a call that waits for the device takes about that time, one
  that does not takes what the driver needs to map and zero the pages (tens of milliseconds,
  more while a bandwidth-bound kernel competes for the memory: allocator-class latency, which
  is information here, not a criterion). PASS when at most 1% of the iterations are blocked.
  The controls ``GATE_E1_SYNC=before`` (``torch.cuda.synchronize()`` before each ``extend``, what
  the product did) and ``inside`` (the same inside the timed region: a call that takes as long
  as the kernel has left) wait by construction and must show every iteration blocked, or the
  gate cannot tell a wait from the lack of one. ``after`` puts it after: it gates nothing.
* DATA (``test_e1_extend_keeps_the_data``): the same, with kernels of about
  ``GATE_E1_KERNEL_MS`` (50 ms) and many iterations with no synchronize in between, so that
  growths and kernels pile up. PASS when there is no CUDA error, every old element has had
  each of A's adds exactly once (fp32 sums are exact below 2**24) and every tail holds what a
  kernel on stream B wrote right after the ``extend`` that made it. The latency is printed.

Optionally a second host thread launches kernels on stream C all the time (the GIL is released
inside ``extend``, so launches and the driver calls can overlap).

The third test, ``recorded_doubling_loop``, is the one the recorded failure was seen in. ``test_vmm_buffer``
(``test/unit_tests/test_vmm_tensor.py``) doubles a buffer of each class until the driver or the
reservation refuses and says, next to the ``torch.cuda.synchronize()`` it calls after every
``extend``: "synchronize is necessary to avoid IMA ... stream synchronization and device
synchronization in C++ not worked". No kernel runs in that loop, and the synchronize came from
Python, so the kernels of the runs above cannot stand in for it. This test runs the same loop
for both buffer classes (``DeviceExtendableBuffer`` and ``HostExtendableBuffer``) with NO
synchronize between the ``extend`` calls and one at the end, where a fault of any of them shows,
and then writes and reads three elements of the grown buffer through the mapping. PASS when the
loop ends by the driver's or the reservation's refusal (the normal stop of ``test_vmm_buffer``),
the synchronize finds no error and the elements read back; anything else (an illegal memory
access, a registration whose device pointer does not continue the buffer) is FAIL, and for the
device class it means the answer of the design is copy-on-grow, not "no wait": do not merge the
commit that deletes the synchronize. ``DYNAMICEMB_VMM_STRESS_FULL=1`` adds fp16 and bf16, as it does
for ``test_vmm_buffer``.

Run it (one H100, the extension built for it with ``DEMB_ARCHS=90``)::

    DEMB_GROWTH_GATES=1 python -m pytest -s -q \\
        corelib/dynamicemb/test/unit_tests/growth/gates/test_gate_e1_device_extend.py
    # the control, which must show every iteration blocked
    DEMB_GROWTH_GATES=1 GATE_E1_SYNC=before python -m pytest -s -q -k does_not_wait <the file>
    # the sanitizer, a reduced run of the data test with the control and the candidate
    DEMB_GROWTH_GATES=1 GATE_E1_ITERS=40 GATE_E1_SYNC=before compute-sanitizer --tool memcheck \\
        --error-exitcode 99 python -m pytest -s -q -k keeps_the_data <the file>
    DEMB_GROWTH_GATES=1 GATE_E1_ITERS=40 compute-sanitizer --tool memcheck \\
        --error-exitcode 99 python -m pytest -s -q -k keeps_the_data <the file>
    # the recorded loop of test_vmm_buffer without its synchronize, twenty times, with and
    # without the sanitizer (one failure of any run is a FAIL)
    for i in $(seq 20); do DEMB_GROWTH_GATES=1 DYNAMICEMB_VMM_STRESS_FULL=1 \\
        python -m pytest -s -q -k recorded_doubling_loop <the file> || break; done
    DEMB_GROWTH_GATES=1 compute-sanitizer --tool memcheck --error-exitcode 99 \\
        python -m pytest -s -q -k recorded_doubling_loop <the file>

Every verdict is one printed line: ``E1 PASS ...`` or ``E1 FAIL ...``.
"""

import os
import platform
import statistics
import sys
import threading
import time

import pytest
import torch
from dynamicemb import extendable_tensor
from dynamicemb_extensions import VMMTensor

pytestmark = pytest.mark.skipif(
    os.environ.get("DEMB_GROWTH_GATES") != "1" or not torch.cuda.is_available(),
    reason="a hardware gate: set DEMB_GROWTH_GATES=1 on a GPU",
)

MB = 1 << 20
# the data runs: short kernels, many iterations, nothing waits for the device in between
ITERS = int(os.environ.get("GATE_E1_ITERS", "100"))
OLD_MB = int(os.environ.get("GATE_E1_OLD_MB", "256"))
KERNEL_MS = float(os.environ.get("GATE_E1_KERNEL_MS", "50"))
# the wait runs: kernels of seconds (a call that waits takes about that long, one that does not
# a few tens of milliseconds), a few iterations; the old buffer is large enough for the kernel
# of the kind ``memory`` to be a few hundred launches (a stream queues about a thousand)
WAIT_ITERS = int(os.environ.get("GATE_E1_WAIT_ITERS", "12"))
WAIT_OLD_MB = int(os.environ.get("GATE_E1_WAIT_OLD_MB", "4096"))
WAIT_KERNEL_MS = float(os.environ.get("GATE_E1_WAIT_KERNEL_MS", "1800"))
MAX_LAUNCHES = 900
GROW_MB = [int(x) for x in os.environ.get("GATE_E1_GROW_MB", "2,6,64").split(",")]
# the buffer is made again every this many iterations, so that the mapped memory stays
# bounded (a buffer never gives pages back): 256 MiB + 16 x 64 MiB
RECREATE_EVERY = int(os.environ.get("GATE_E1_RECREATE_EVERY", "16"))
SYNC = os.environ.get("GATE_E1_SYNC", "none")  # none | before | inside | after

# The acceptance criterion of the design: a call that waits for kernel A comes back after it,
# or takes about as long as A has left; at most this share of the calls may.
MAX_BLOCKED = 0.01
# What a blocked call is: it took more than this share of the time A had left.
BLOCKED_SHARE = 0.5


def kernel_adds(base: torch.Tensor, target_ms: float) -> int:
    """How many ``add_`` over the old region take about ``target_ms`` (at most
    MAX_LAUNCHES: a stream that has more queued makes the host wait in the launch)."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        base.add_(1.0)  # load the kernel
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record(stream)
        for _ in range(10):
            base.add_(1.0)
        end.record(stream)
    end.synchronize()
    base.sub_(11.0)
    per_add = max(start.elapsed_time(end) / 10, 1e-3)
    return max(1, min(MAX_LAUNCHES, int(target_ms / per_add)))


def sleep_cycles(target_ms: float) -> int:
    """How many cycles of ``torch.cuda._sleep`` take about ``target_ms`` (measured: the clock
    of the device is not something PyTorch reports on every version)."""
    stream = torch.cuda.Stream()
    probe = 20_000_000
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000)  # load the kernel
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record(stream)
        torch.cuda._sleep(probe)
        end.record(stream)
    end.synchronize()
    return max(1, int(probe * target_ms / max(start.elapsed_time(end), 1e-3)))


def percentile(sorted_values, share: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(share * len(sorted_values)))]


class Result:
    def __init__(self, mode: str):
        self.mode = mode
        self.latencies = []
        self.blocked = 0
        self.problems = []
        self.iterations = 0
        self.kernel_ms = []  # what kernel A took in each iteration of a wait run
        self.target_ms = 0.0  # what it was meant to take

    def verdict(self, label: str) -> bool:
        lat = sorted(self.latencies) or [0.0]
        median = statistics.median(lat)
        p99, worst = percentile(lat, 0.99), lat[-1]
        fraction = self.blocked / max(1, self.iterations)
        problems = list(self.problems)
        if self.mode == "wait" and SYNC == "none":
            if fraction > MAX_BLOCKED:
                problems.append(
                    "extend was blocked (it returned after kernel A had finished, or took "
                    "more than half of what A had left) in %.1f%% of the iterations "
                    "(at most %.0f%%)" % (100 * fraction, 100 * MAX_BLOCKED)
                )
        elif self.mode == "wait" and SYNC in ("before", "inside"):
            # the controls wait by construction: they have to be seen
            if fraction < 0.99:
                problems.append(
                    "the control (a synchronize %s every extend) was blocked in only "
                    "%.1f%% of the iterations: the gate cannot see a wait"
                    % (
                        "before" if SYNC == "before" else "inside the timing of",
                        100 * fraction,
                    )
                )
        kernel = statistics.median(self.kernel_ms) if self.kernel_ms else self.target_ms
        line = (
            "%s mode=%s iters=%d %s=%.2f%% extend: median=%.3fms p99=%.3fms max=%.3fms "
            "(kernel A %.0fms) sync=%s"
            % (
                label,
                self.mode,
                self.iterations,
                # in a data run kernel A is short, and a slow call can return after it
                # without having waited for it: the share is information there
                "blocked" if self.mode == "wait" else "returned_after_A(info)",
                100 * fraction,
                median * 1e3,
                p99 * 1e3,
                worst * 1e3,
                kernel,
                SYNC,
            )
        )
        if problems:
            print("E1 FAIL %s: %s" % (line, "; ".join(problems)))
            return False
        print("E1 PASS %s" % line)
        return True


def run(grow_mb: int, kind: str, with_thread: bool, mode: str) -> Result:
    """``mode`` is ``wait`` (one long kernel A per iteration, a synchronize after each
    iteration) or ``data`` (short kernels, no synchronize until a buffer is replaced).
    """
    wait = mode == "wait"
    iters = WAIT_ITERS if wait else ITERS
    device = torch.cuda.current_device()
    result = Result(mode)
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    n_old = (WAIT_OLD_MB if wait and kind == "memory" else OLD_MB) * MB // 4
    grow = grow_mb * MB // 4
    target_ms = WAIT_KERNEL_MS if wait else KERNEL_MS
    result.target_ms = target_ms
    cycles = sleep_cycles(target_ms) if kind == "sleep" else 0

    stop = threading.Event()
    thread_errors = []
    thread_state = {"launches": 0}
    thread_tensor = torch.zeros(1 << 18, device="cuda")

    def launcher():
        stream_c = torch.cuda.Stream()
        try:
            with torch.cuda.stream(stream_c):
                while not stop.is_set():
                    thread_tensor.add_(1.0)
                    thread_state["launches"] += 1
                    time.sleep(1e-4)
        except Exception as exc:  # noqa: BLE001 -- any error is the verdict
            thread_errors.append(repr(exc))

    worker = threading.Thread(target=launcher) if with_thread else None
    if worker:
        worker.start()

    buf, base, total, adds, fills, k = None, None, 0, 0, [], 0
    try:
        for it in range(iters):
            if buf is None or it % RECREATE_EVERY == 0:
                if buf is not None:
                    check(result, buf, base, n_old, adds, fills)
                torch.cuda.synchronize()
                buf = VMMTensor(n_old, torch.float32, device)
                base = buf.data()  # a pointer taken before any growth: never stale
                base.zero_()
                torch.cuda.synchronize()
                total, adds, fills = n_old, 0, []
                if kind == "memory" and k == 0:
                    k = kernel_adds(base, target_ms)
            if wait:
                torch.cuda.synchronize()  # kernel A starts on an idle device
            launched = time.perf_counter()
            with torch.cuda.stream(stream_a):
                a_start = torch.cuda.Event(enable_timing=True)
                a_start.record(stream_a)
                if kind == "memory":
                    for _ in range(k):
                        base.add_(1.0)
                else:
                    torch.cuda._sleep(cycles)
                a_end = torch.cuda.Event(enable_timing=True)
                a_end.record(stream_a)
            adds += k if kind == "memory" else 0
            if SYNC == "before":
                torch.cuda.synchronize()
            start = time.perf_counter()
            if SYNC == "inside":
                torch.cuda.synchronize()  # a call that waits for the device, as timed
            total += grow
            buf.extend(total)  # NO device synchronize anywhere in the candidate
            latency = time.perf_counter() - start
            finished = a_end.query()
            result.latencies.append(latency)
            result.iterations += 1
            view = buf.data()  # a kernel launched right after sees the new tail
            value = float(it % 97 + 1)
            with torch.cuda.stream(stream_b):
                view[total - grow : total].fill_(value)
            fills.append((total - grow, total, value))
            if wait:
                torch.cuda.synchronize()
                kernel_s = a_start.elapsed_time(a_end) / 1e3
                result.kernel_ms.append(kernel_s * 1e3)
                left = kernel_s - (start - launched)
                result.blocked += int(finished or latency > BLOCKED_SHARE * left)
            else:
                result.blocked += int(finished)
            if SYNC == "after":
                torch.cuda.synchronize()
        if buf is not None:
            check(result, buf, base, n_old, adds, fills)
    except Exception as exc:  # noqa: BLE001 -- a CUDA error (an IMA) is the verdict
        result.problems.append("%s: %s" % (type(exc).__name__, str(exc)[:300]))
    finally:
        stop.set()
        if worker:
            worker.join()
        try:
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            result.problems.append("synchronize: %s" % str(exc)[:300])
    if thread_errors:
        result.problems.append("launching thread: %s" % thread_errors[0])
    if with_thread and thread_state["launches"] > 0:
        expected = float(thread_state["launches"])
        if float(thread_tensor[0]) != expected:
            result.problems.append(
                "the second thread's kernels: %s of %d adds"
                % (float(thread_tensor[0]), expected)
            )
    return result


def check(result, buf, base, n_old, adds, fills):
    torch.cuda.synchronize()
    old = base[:n_old]
    if not bool((old == float(adds)).all()):
        result.problems.append(
            "an old element does not hold %d (kernel A's adds): its pages were hurt"
            % adds
        )
    view = buf.data()
    for lo, hi, value in fills:
        if not bool((view[lo:hi] == value).all()):
            result.problems.append(
                "the tail [%d, %d) does not hold %s: the kernel after extend did not see it"
                % (lo, hi, value)
            )
            break


def label_of(grow_mb: int, kind: str, with_thread: bool) -> str:
    return "%s/%s/grow%dMiB" % ("thread" if with_thread else "alone", kind, grow_mb)


@pytest.mark.parametrize(
    "with_thread", [False, True], ids=["alone", "with_launching_thread"]
)
@pytest.mark.parametrize("kind", ["memory", "sleep"])
@pytest.mark.parametrize("grow_mb", GROW_MB)
def test_e1_extend_does_not_wait(grow_mb, kind, with_thread):
    result = run(grow_mb, kind, with_thread, "wait")
    ok = result.verdict(label_of(grow_mb, kind, with_thread))
    assert ok, "E1 failed; see the E1 FAIL line above"


@pytest.mark.parametrize(
    "with_thread", [False, True], ids=["alone", "with_launching_thread"]
)
@pytest.mark.parametrize("kind", ["memory", "sleep"])
@pytest.mark.parametrize("grow_mb", GROW_MB)
def test_e1_extend_keeps_the_data(grow_mb, kind, with_thread):
    result = run(grow_mb, kind, with_thread, "data")
    ok = result.verdict(label_of(grow_mb, kind, with_thread))
    assert ok, "E1 failed; see the E1 FAIL line above"


# --- the loop of test_vmm_buffer, with no synchronize between the extends -------------------

# test_vmm_buffer's: the capacities a buffer starts from, and how long one iteration may take
# before the loop stops as it does at an out-of-memory (a normal stop, not waiting further)
BUFFER_INIT_CAPACITIES = (1, 1023, 1024, 1 << 30)
BUFFER_ITERATION_LIMIT_S = 10.0
# What the driver or the reservation says when they refuse: the loop ends, which is the point
# of it. A fault, or a registration the device cannot address contiguously, is not one.
NORMAL_STOPS = (
    "out of memory",
    "exceeds reserved",
    "address space",
    "exceeds total physical memory",
    "mlock",
    "mprotect",
    "mmap",
)
FAULTS = (
    "illegal memory access",
    "unspecified launch failure",
    "device-side assert",
    "operation not supported",
    "misaligned",
)


def buffer_dtypes():
    if os.environ.get("DYNAMICEMB_VMM_STRESS_FULL") == "1":
        return [torch.float32, torch.float16, torch.bfloat16]
    return [torch.float32]


def classify_stop(message: str) -> str:
    """ "fault", "normal" or "unexpected" for the text of an error that ended the loop."""
    text = message.lower()
    if any(marker in text for marker in FAULTS):
        return "fault"
    if any(marker in text for marker in NORMAL_STOPS):
        return "normal"
    return "unexpected"


@pytest.mark.parametrize("dtype", buffer_dtypes())
@pytest.mark.parametrize("init_capacity", BUFFER_INIT_CAPACITIES)
@pytest.mark.parametrize("buffer_class", ["device", "host"])
def test_e1_recorded_doubling_loop_without_a_synchronize(
    buffer_class, init_capacity, dtype
):
    host = buffer_class == "host"
    if host and (platform.system() != "Linux" or not extendable_tensor._HOST_IN_PLACE):
        pytest.skip("a host buffer needs Linux and an extension that grows it in place")
    cls = (
        extendable_tensor.HostExtendableBuffer
        if host
        else extendable_tensor.DeviceExtendableBuffer
    )
    device = torch.device("cuda", torch.cuda.current_device())
    label = "recorded_loop/%s/%s/init%d" % (buffer_class, str(dtype)[6:], init_capacity)
    problems, stop, iterations = [], "the loop was cut by its time limit", 0

    buffer = cls((init_capacity,), dtype, device)
    apply_capacity = init_capacity
    try:
        while True:
            started = time.monotonic()
            buffer.extend((apply_capacity,))  # NO synchronize, anywhere in the loop
            apply_capacity *= 2
            iterations += 1
            if time.monotonic() - started > BUFFER_ITERATION_LIMIT_S:
                break
    except RuntimeError as exc:
        stop = "%s: %s" % (classify_stop(str(exc)), str(exc)[:160])
        if classify_stop(str(exc)) != "normal":
            problems.append("the loop ended by an error that is not a refusal: " + stop)

    # one synchronize, at the end: a fault of any of the extends shows here
    try:
        torch.cuda.synchronize()
        flat = buffer.tensor().reshape(-1)
        last = flat.numel() - 1
        for index in (0, last // 2, last):
            flat[index] = index % 97 + 1
        torch.cuda.synchronize()
        for index in (0, last // 2, last):
            if float(flat[index]) != float(index % 97 + 1):
                problems.append(
                    "element %d of the grown buffer reads back %s"
                    % (index, flat[index])
                )
    except Exception as exc:  # noqa: BLE001 -- an IMA is the verdict, whatever its type
        problems.append("%s: %s" % (type(exc).__name__, str(exc)[:300]))

    if iterations == 0 and not problems:
        problems.append("not one extend succeeded")
    line = "%s extends=%d final_numel=%d stop=%s" % (
        label,
        iterations,
        buffer.numel(),
        stop,
    )
    if problems:
        print("E1 FAIL %s: %s" % (line, "; ".join(problems)))
    else:
        print("E1 PASS %s" % line)
    assert not problems, "E1 failed; see the E1 FAIL line above"


if __name__ == "__main__":
    os.environ.setdefault("DEMB_GROWTH_GATES", "1")
    passed = True
    for g in GROW_MB:
        for kind in ("memory", "sleep"):
            for thread in (False, True):
                for mode in ("wait", "data"):
                    passed &= run(g, kind, thread, mode).verdict(
                        label_of(g, kind, thread)
                    )
    print("E1 %s" % ("PASS" if passed else "FAIL"))
    sys.exit(0 if passed else 1)
