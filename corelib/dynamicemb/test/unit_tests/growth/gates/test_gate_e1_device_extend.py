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

The script drives the C++ buffer directly (no Python synchronize, whatever the tree has) with

* kernel A in flight on stream A: ``base.add_(1.0)`` k times, which reads and writes every old
  element for about 50 ms (kind ``memory``), or ``torch.cuda._sleep`` of the same length (kind
  ``sleep``: no memory traffic, to tell "waits for the kernel" from "contention");
* ``extend`` on the host thread meanwhile, by 2, 6 and 64 MiB (below a granule, a few
  granules, many);
* a kernel on stream B on the new tail right after ``extend`` returned;
* optionally a second host thread that launches kernels on stream C all the time (the GIL is
  released inside ``extend``, so launches and the driver calls can overlap).

PASS when, for every (growth, kind): no CUDA error; every old element has had each of A's adds
exactly once (fp32 sums are exact below 2**24) and every tail holds what B wrote; ``extend``
returned while A was still running in at least 99% of the iterations; the median ``extend``
takes at most 2 ms and the 99th percentile at most 10 ms (``GATE_E1_MAX_MEDIAN_MS`` and
``GATE_E1_MAX_P99_MS`` loosen those two for a driver that is only slow).

``GATE_E1_SYNC=before`` repeats the run with ``torch.cuda.synchronize()`` before every
``extend`` (what the product does today: the control; its latency is that of an idle device),
``after`` puts it after (the comment of ``test_vmm_buffer``: "synchronize is necessary to
avoid IMA"); neither gates anything.

The second test, ``recorded_doubling_loop``, is the one the recorded failure was seen in. ``test_vmm_buffer``
(``test/unit_tests/test_vmm_tensor.py``) doubles a buffer of each class until the driver or the
reservation refuses and says, next to the ``torch.cuda.synchronize()`` it calls after every
``extend``: "synchronize is necessary to avoid IMA ... stream synchronization and device
synchronization in C++ not worked". No kernel runs in that loop, and the synchronize came from
Python, so the kernels of the loop above cannot stand in for it. This test runs the same loop
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

    DEMB_GROWTH_GATES=1 GATE_E1_ITERS=2000 python -m pytest -s -q \\
        corelib/dynamicemb/test/unit_tests/growth/gates/test_gate_e1_device_extend.py
    # the sanitizer, with the control and the candidate; compare their reports
    DEMB_GROWTH_GATES=1 GATE_E1_ITERS=40 GATE_E1_SYNC=before compute-sanitizer --tool memcheck \\
        --error-exitcode 99 python -m pytest -s -q <the file>
    DEMB_GROWTH_GATES=1 GATE_E1_ITERS=40 compute-sanitizer --tool memcheck \\
        --error-exitcode 99 python -m pytest -s -q <the file>
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
ITERS = int(os.environ.get("GATE_E1_ITERS", "100"))
OLD_MB = int(os.environ.get("GATE_E1_OLD_MB", "256"))
GROW_MB = [int(x) for x in os.environ.get("GATE_E1_GROW_MB", "2,6,64").split(",")]
# the buffer is made again every this many iterations, so that the mapped memory stays
# bounded (a buffer never gives pages back): 256 MiB + 16 x 64 MiB
RECREATE_EVERY = int(os.environ.get("GATE_E1_RECREATE_EVERY", "16"))
SYNC = os.environ.get("GATE_E1_SYNC", "none")  # none | before | after
KERNEL_MS = float(os.environ.get("GATE_E1_KERNEL_MS", "50"))

# The acceptance criteria of the design (a driver call that waits for kernel A takes about
# KERNEL_MS, 25 times the median allowed here); the two latency limits can be loosened for
# a driver that is only slow, never the share of calls that returned after A had finished.
MAX_BLOCKED = 0.01
MAX_MEDIAN_S = float(os.environ.get("GATE_E1_MAX_MEDIAN_MS", "2")) * 1e-3
MAX_P99_S = float(os.environ.get("GATE_E1_MAX_P99_MS", "10")) * 1e-3


def kernel_adds(base: torch.Tensor) -> int:
    """How many ``add_`` over the old region take about KERNEL_MS."""
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
    return max(1, int(KERNEL_MS / per_add))


def sleep_cycles() -> int:
    """How many cycles of ``torch.cuda._sleep`` take about KERNEL_MS (measured: the clock of
    the device is not something PyTorch reports on every version)."""
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
    return max(1, int(probe * KERNEL_MS / max(start.elapsed_time(end), 1e-3)))


class Result:
    def __init__(self):
        self.latencies = []
        self.blocked = 0
        self.problems = []
        self.iterations = 0

    def verdict(self, label: str) -> bool:
        lat = sorted(self.latencies) or [0.0]
        median = statistics.median(lat)
        p99 = lat[min(len(lat) - 1, int(0.99 * len(lat)))]
        fraction = self.blocked / max(1, self.iterations)
        problems = list(self.problems)
        if SYNC == "none":
            if fraction > MAX_BLOCKED:
                problems.append(
                    "extend returned only after kernel A had finished in %.1f%% of "
                    "the iterations (at most %.0f%%)"
                    % (100 * fraction, 100 * MAX_BLOCKED)
                )
            if median > MAX_MEDIAN_S:
                problems.append(
                    "median extend %.2f ms > %.1f ms"
                    % (median * 1e3, MAX_MEDIAN_S * 1e3)
                )
            if p99 > MAX_P99_S:
                problems.append(
                    "p99 extend %.2f ms > %.1f ms" % (p99 * 1e3, MAX_P99_S * 1e3)
                )
        line = "%s iters=%d blocked=%.2f%% median=%.3fms p99=%.3fms sync=%s" % (
            label,
            self.iterations,
            100 * fraction,
            median * 1e3,
            p99 * 1e3,
            SYNC,
        )
        if problems:
            print("E1 FAIL %s: %s" % (line, "; ".join(problems)))
            return False
        print("E1 PASS %s" % line)
        return True


def run(grow_mb: int, kind: str, with_thread: bool, iters: int = None) -> Result:
    iters = ITERS if iters is None else iters
    device = torch.cuda.current_device()
    result = Result()
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    n_old = OLD_MB * MB // 4
    grow = grow_mb * MB // 4
    cycles = sleep_cycles() if kind == "sleep" else 0

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
                    k = kernel_adds(base)
            with torch.cuda.stream(stream_a):
                if kind == "memory":
                    for _ in range(k):
                        base.add_(1.0)
                else:
                    torch.cuda._sleep(cycles)
                done = torch.cuda.Event()
                done.record(stream_a)
            adds += k if kind == "memory" else 0
            if SYNC == "before":
                torch.cuda.synchronize()
            start = time.perf_counter()
            total += grow
            buf.extend(total)  # NO device synchronize anywhere in the candidate
            result.latencies.append(time.perf_counter() - start)
            result.blocked += int(done.query())
            result.iterations += 1
            view = buf.data()  # a kernel launched right after sees the new tail
            value = float(it % 97 + 1)
            with torch.cuda.stream(stream_b):
                view[total - grow : total].fill_(value)
            fills.append((total - grow, total, value))
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


@pytest.mark.parametrize(
    "with_thread", [False, True], ids=["alone", "with_launching_thread"]
)
@pytest.mark.parametrize("kind", ["memory", "sleep"])
@pytest.mark.parametrize("grow_mb", GROW_MB)
def test_e1_extend_under_load(grow_mb, kind, with_thread):
    result = run(grow_mb, kind, with_thread)
    ok = result.verdict(
        "%s/%s/%s/grow%dMiB"
        % ("thread" if with_thread else "alone", kind, SYNC, grow_mb)
    )
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
                passed &= run(g, kind, thread).verdict(
                    "%s/%s/%s/grow%dMiB"
                    % ("thread" if thread else "alone", kind, SYNC, g)
                )
    print("E1 %s" % ("PASS" if passed else "FAIL"))
    sys.exit(0 if passed else 1)
