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

"""Hardware checks behind the designs of the explicit APIs (dump, load, replay, pops).

Not part of the CI list (``unit_test.sh``): each check says PASS or FAIL for one assumption
the code makes about the stack it runs on, and the answer is a property of the machine
(an H100 with the CUDA 13.3 / PyTorch 2.13 image, in the coordinator's runs) and not of the
code. Run it alone, and again under the sanitizer::

    python -m pytest -s test/unit_tests/no_host_sync/test_hw_explicit_api_copies.py
    compute-sanitizer --tool memcheck python -m pytest -x -q \\
        test/unit_tests/no_host_sync/test_hw_explicit_api_copies.py

What is checked, each with a kernel that keeps the stream busy for about a second (so a host
call that waits for the device shows up as a call that takes about a second, and, more to
the point, as a call that returns after the device finished what was queued before it):

* a copy of a strided CUDA tensor (the embedding columns of a value row, the prefix of one)
  into pinned memory with ``non_blocking=True`` returns at once and delivers the right
  bytes -- the dump copies ``values[:n, :emb_dim]`` that way, with no contiguous copy of
  the whole batch;
* the same for a contiguous prefix (the keys, the scores);
* a staged upload (``sync_free.StagedUpload``) of a host tensor returns at once, and what
  arrives is the host tensor, also when the host tensor is overwritten right after;
* ``sync_free.pull_to_host`` waits for the work queued before it on the current stream,
  and for nothing else, and what it returns is what that work wrote: a kernel that writes
  the values after a long spin kernel is not overtaken, on the default stream, on a side
  stream, for a strided source and for several tensors at once;
* a ring of two staging buffers waits for a buffer when the device is still busy with the
  transfer two uploads back: the floor that load and replay state (about one wait per
  upload while the device is the bottleneck, none while the host is).

How the checks measure, and what went wrong once: the first launch of a kernel in a process
waits for the device (``CUDA_MODULE_LOADING=LAZY``, the default: the module of the kernel
is loaded behind whatever is running). A check that queues a kernel it has never launched
behind the spin kernel therefore blocks in that *launch*, before the call it means to time:
the strided copy of the dump looked as if it blocked for the whole second, and ``x.add_(5)``
ahead of a ``pull_to_host`` made it look as if ``pull_to_host`` did not wait (the stream was
idle when it was called). Neither was the call's doing: with ``CUDA_MODULE_LOADING=EAGER``,
or with the kernel launched once before, the same calls behave. So every check launches
what it uses once before the spin kernel, and the instrument is self-checking: an event
recorded behind the spin kernel (and behind the work queued after it) must still be pending
when the call starts (else the call is timed against an idle device and proves nothing), and
is the evidence of whether the call waited when it returns (a call that does not wait
returns while it is pending, one that does returns after it completed). A control copy into
pageable memory, which does wait, shows that the instrument sees a wait.
"""

import importlib
import time

import pytest

try:
    import torch
except ImportError:
    pytest.skip("needs torch", allow_module_level=True)

BUSY_CYCLES = 2 * 10**9  # about a second of an H100
HOST_LIMIT = 0.25  # seconds: what a call that does not wait may take
ROWS, WIDTH, KEEP = 4096, 64, 24


def _need(name):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.skip("%s is not importable here: %s" % (name, e))


def _timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _verdict(name, ok, detail):
    print("%s: %s (%s)" % ("PASS" if ok else "FAIL", name, detail))
    assert ok, "%s: %s" % (name, detail)


def _spin(cycles=BUSY_CYCLES):
    """Queue a kernel that keeps the current stream busy for about ``cycles`` GPU cycles.

    The device is idle before it, so that its start is the moment of this call; the caller
    queues what it wants behind it and calls :func:`_behind`."""
    torch.cuda.synchronize()
    torch.cuda._sleep(cycles)


def _behind():
    """An event behind everything queued so far on the current stream."""
    event = torch.cuda.Event()
    event.record()
    return event


def _device_is_busy(behind, what):
    """The instrument's own check, made right before the call under test: the device is
    still running what was queued before it. If it is not, the host waited somewhere
    on its way here (the first launch of a kernel does, see the module docstring) and a
    short call proves nothing."""
    busy = not behind.query()
    _verdict(
        "the device is still busy when %s starts" % what,
        busy,
        "the spin kernel is still running"
        if busy
        else "the spin kernel had finished already: a launch before the call waited for "
        "it (a kernel that was never launched in this process? warm it up first)",
    )


def _did_not_wait(name, elapsed, behind):
    """The verdict on a call that must not wait for the device: it came back quickly, and
    while what was queued before it (``behind``) was still running. The second is the
    evidence; the first is only a number to read."""
    busy = not behind.query()
    ok = elapsed < HOST_LIMIT and busy
    _verdict(
        name,
        ok,
        "the host call took %.3f s behind a kernel of about 1 s, and returned %s%s"
        % (
            elapsed,
            "while the device was busy" if busy else "after it was done",
            ""
            if ok
            else " (a kernel the call launches for the first time in this process waits "
            "for the device: warm the call up before the spin kernel)",
        ),
    )


def test_the_instrument_sees_a_host_wait(gpu):
    """The control: a copy into pageable memory waits for the stream, and the instrument
    reports it as a call of about the rest of the spin that returns after the device is
    done. Without it a call that does not wait would read the same as a broken instrument.
    """
    column = torch.arange(ROWS, dtype=torch.int64, device="cuda")
    pageable = torch.empty(1000, dtype=torch.int64)
    pageable.copy_(column[:1000], non_blocking=True)  # warm up
    torch.cuda.synchronize()
    _spin()
    behind = _behind()
    _device_is_busy(behind, "the control copy")
    _, elapsed = _timed(lambda: pageable.copy_(column[:1000], non_blocking=True))
    _verdict(
        "a copy into pageable memory waits for the device",
        elapsed > 0.5 and behind.query(),
        "the host call took %.3f s behind a kernel of about 1 s" % elapsed,
    )


def test_a_strided_copy_into_pinned_memory_does_not_wait(gpu):
    rows = torch.arange(ROWS * WIDTH, dtype=torch.float32, device="cuda").view(
        ROWS, WIDTH
    )
    view = rows[:1000, :KEEP]  # the embedding columns of the first 1000 value rows
    assert not view.is_contiguous()
    host = torch.empty(tuple(view.shape), dtype=view.dtype, pin_memory=True)
    host.copy_(view, non_blocking=True)  # warm up: the kernel that makes it contiguous
    torch.cuda.synchronize()
    host.zero_()
    _spin()
    behind = _behind()
    _device_is_busy(behind, "the strided copy")
    _, elapsed = _timed(lambda: host.copy_(view, non_blocking=True))
    _did_not_wait("strided D2H into pinned memory is asynchronous", elapsed, behind)
    behind.synchronize()
    _verdict(
        "strided D2H delivers the bytes", torch.equal(host, view.cpu()), "compared"
    )


def test_a_contiguous_prefix_copy_into_pinned_memory_does_not_wait(gpu):
    column = torch.arange(ROWS, dtype=torch.int64, device="cuda")
    host = torch.empty(1000, dtype=torch.int64, pin_memory=True)
    host.copy_(column[:1000], non_blocking=True)  # warm up
    torch.cuda.synchronize()
    host.zero_()
    _spin()
    behind = _behind()
    _device_is_busy(behind, "the prefix copy")
    _, elapsed = _timed(lambda: host.copy_(column[:1000], non_blocking=True))
    _did_not_wait("contiguous D2H into pinned memory is asynchronous", elapsed, behind)
    behind.synchronize()
    _verdict(
        "contiguous D2H delivers the bytes",
        torch.equal(host, column[:1000].cpu()),
        "compared",
    )


def test_a_staged_upload_does_not_wait_and_delivers_the_host_tensor(gpu):
    sf = _need("dynamicemb.sync_free")
    ring = sf.StagedUpload(torch.device("cuda", torch.cuda.current_device()))
    host = torch.randn(ROWS, 8)
    expected = host.clone()
    ring.upload(host)  # warm up: the staging buffers and the allocator
    ring.upload(host)
    torch.cuda.synchronize()
    _spin()
    behind = _behind()
    _device_is_busy(behind, "the staged upload")
    device, elapsed = _timed(lambda: ring.upload(host))
    _did_not_wait("a staged upload is asynchronous", elapsed, behind)
    host.zero_()  # the staging buffer, not this tensor, is what the copy reads
    torch.cuda.synchronize()
    _verdict(
        "a staged upload delivers the tensor",
        torch.equal(device.cpu(), expected),
        "compared",
    )


def test_pull_to_host_waits_for_the_work_before_it_and_gives_its_values(gpu):
    sf = _need("dynamicemb.sync_free")
    x = torch.zeros(1000, dtype=torch.int64, device="cuda")
    # every kernel launched behind the spin kernel, and the call itself, once before
    x.add_(1)
    sf.pull_to_host(x)
    torch.cuda.synchronize()
    x.zero_()
    _spin(BUSY_CYCLES // 2)
    x.add_(5)  # behind the long kernel
    behind = _behind()
    _device_is_busy(behind, "pull_to_host")
    (host,), elapsed = _timed(lambda: sf.pull_to_host(x))
    _verdict(
        "pull_to_host waits for the kernels queued before it",
        elapsed > 0.25 and behind.query(),
        "the call took %.3f s behind a kernel of about 0.5 s, and returned %s"
        % (elapsed, "after the device was done" if behind.query() else "too early"),
    )
    _verdict("pull_to_host gives the final values", bool((host == 5).all()), "compared")


def test_pull_to_host_gives_what_a_kernel_wrote_after_a_long_spin(gpu):
    """Data integrity, on the default stream: the values do not exist until the kernel
    behind the spin has run. A ``pull_to_host`` that returned before it would hand back the
    zeros that were there; here several tensors, one of them a strided view (the embedding
    columns of a value block), are written after the spin and pulled in one call."""
    sf = _need("dynamicemb.sync_free")
    base = torch.arange(1, 1001, device="cuda")
    keys = torch.zeros(1000, dtype=torch.int64, device="cuda")
    rows = torch.zeros(ROWS, WIDTH, dtype=torch.float32, device="cuda")
    columns = rows[:1000, :KEEP]
    assert not columns.is_contiguous()
    torch.mul(base, 3, out=keys)  # warm up: the kernels below and the call itself
    rows.fill_(1.0)
    sf.pull_to_host(keys, columns)
    torch.cuda.synchronize()
    keys.zero_()
    rows.zero_()
    _spin(BUSY_CYCLES // 2)
    torch.mul(base, 3, out=keys)  # the writes, behind the spin kernel
    rows.fill_(7.5)
    behind = _behind()
    _device_is_busy(behind, "pull_to_host")
    (host_keys, host_columns), elapsed = _timed(lambda: sf.pull_to_host(keys, columns))
    done = behind.query()
    _verdict(
        "pull_to_host returned after the writes were done",
        done,
        "the call took %.3f s behind a kernel of about 0.5 s" % elapsed,
    )
    expected_keys = torch.arange(1, 1001) * 3
    _verdict(
        "the keys it gives are the ones the kernel wrote",
        torch.equal(host_keys, expected_keys),
        "the first few are %s, expected %s"
        % (host_keys[:3].tolist(), expected_keys[:3].tolist()),
    )
    _verdict(
        "the strided columns it gives are the ones the kernel wrote",
        bool((host_columns == 7.5).all()) and host_columns.shape == (1000, KEEP),
        "shape %s, first value %s"
        % (tuple(host_columns.shape), host_columns[0, 0].item()),
    )
    _verdict(
        "what it gives lives in pinned memory",
        host_keys.is_pinned() and host_columns.is_pinned(),
        "the destinations are pinned",
    )


def test_pull_to_host_waits_for_the_current_stream_and_for_nothing_else(gpu):
    """The stream it waits for is the one that is current when it is called: a side stream
    inside ``torch.cuda.stream`` is waited for (and its values arrive), while a busy stream
    that is not current is not."""
    sf = _need("dynamicemb.sync_free")
    side, other = torch.cuda.Stream(), torch.cuda.Stream()
    x = torch.zeros(1000, dtype=torch.int64, device="cuda")
    with torch.cuda.stream(side):  # warm up on the stream the checks use
        x.fill_(1)
        sf.pull_to_host(x)
    torch.cuda.synchronize()

    # (1) the side stream is current: the call waits for its spin kernel and gets its values
    with torch.cuda.stream(side):
        _spin(BUSY_CYCLES // 2)
        x.fill_(9)
        behind = _behind()
        _device_is_busy(behind, "pull_to_host on a side stream")
        (host,), elapsed = _timed(lambda: sf.pull_to_host(x))
        waited = behind.query()
    _verdict(
        "pull_to_host waits for the current (side) stream",
        elapsed > 0.25 and waited,
        "the call took %.3f s behind a kernel of about 0.5 s" % elapsed,
    )
    _verdict(
        "pull_to_host gives what the side stream wrote",
        bool((host == 9).all()),
        "compared",
    )

    # (2) another stream is busy, the current one is not: nothing to wait for
    with torch.cuda.stream(other):
        _spin()
        busy = _behind()
    torch.cuda.current_stream().synchronize()
    _, elapsed = _timed(lambda: sf.pull_to_host(x))
    _did_not_wait(
        "pull_to_host does not wait for a stream that is not current", elapsed, busy
    )
    torch.cuda.synchronize()


def test_a_staging_ring_of_two_waits_for_a_buffer_when_the_device_is_busy(gpu):
    """The floor that load and replay state: the third upload of a ring of two reuses the
    first buffer, and the device is still busy with that transfer."""
    sf = _need("dynamicemb.sync_free")
    ring = sf.StagedUpload(torch.device("cuda", torch.cuda.current_device()), ring=2)
    host = torch.randn(ROWS, 8)
    for _ in range(4):  # warm up: both buffers, their events, the allocator
        ring.upload(host)
    torch.cuda.synchronize()
    _spin()  # every transfer below is queued behind this
    behind = _behind()
    _device_is_busy(behind, "the first upload")
    _, first = _timed(lambda: ring.upload(host))
    _, second = _timed(lambda: ring.upload(host))
    _verdict(
        "the first two uploads of a ring do not wait",
        max(first, second) < HOST_LIMIT and not behind.query(),
        "they took %.3f s and %.3f s" % (first, second),
    )
    _, third = _timed(lambda: ring.upload(host))
    _verdict(
        "the third upload waits for the first buffer while the device is busy",
        third > 0.05 and behind.query(),
        "it took %.3f s" % third,
    )
    torch.cuda.synchronize()
