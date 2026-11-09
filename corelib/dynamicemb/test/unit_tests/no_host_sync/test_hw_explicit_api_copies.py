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
call that waits for the device shows up as a call that takes about a second):

* a copy of a strided CUDA tensor (the embedding columns of a value row, the prefix of one)
  into pinned memory with ``non_blocking=True`` returns at once and delivers the right
  bytes -- the dump copies ``values[:n, :emb_dim]`` that way, with no contiguous copy of
  the whole batch;
* the same for a contiguous prefix (the keys, the scores);
* a staged upload (``sync_free.StagedUpload``) of a host tensor returns at once, and what
  arrives is the host tensor, also when the host tensor is overwritten right after;
* ``sync_free.pull_to_host`` waits for the work queued before it, and for nothing else;
* a ring of two staging buffers waits for a buffer when the device is still busy with the
  transfer two uploads back: the floor that load and replay state (a wait at most once a
  ring turn while the device is the bottleneck, none while the host is).
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


def test_a_strided_copy_into_pinned_memory_does_not_wait(gpu):
    rows = torch.arange(ROWS * WIDTH, dtype=torch.float32, device="cuda").view(
        ROWS, WIDTH
    )
    view = rows[:1000, :KEEP]  # the embedding columns of the first 1000 value rows
    assert not view.is_contiguous()
    host = torch.empty(tuple(view.shape), dtype=view.dtype, pin_memory=True)
    torch.cuda.synchronize()
    torch.cuda._sleep(BUSY_CYCLES)
    _, elapsed = _timed(lambda: host.copy_(view, non_blocking=True))
    done = torch.cuda.Event()
    done.record()
    _verdict(
        "strided D2H into pinned memory is asynchronous",
        elapsed < HOST_LIMIT,
        "the host call took %.3f s behind a kernel of about 1 s" % elapsed,
    )
    done.synchronize()
    _verdict(
        "strided D2H delivers the bytes", torch.equal(host, view.cpu()), "compared"
    )


def test_a_contiguous_prefix_copy_into_pinned_memory_does_not_wait(gpu):
    column = torch.arange(ROWS, dtype=torch.int64, device="cuda")
    host = torch.empty(1000, dtype=torch.int64, pin_memory=True)
    torch.cuda.synchronize()
    torch.cuda._sleep(BUSY_CYCLES)
    _, elapsed = _timed(lambda: host.copy_(column[:1000], non_blocking=True))
    done = torch.cuda.Event()
    done.record()
    _verdict(
        "contiguous D2H into pinned memory is asynchronous",
        elapsed < HOST_LIMIT,
        "the host call took %.3f s" % elapsed,
    )
    done.synchronize()
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
    torch.cuda.synchronize()
    torch.cuda._sleep(BUSY_CYCLES)
    device, elapsed = _timed(lambda: ring.upload(host))
    host.zero_()  # the staging buffer, not this tensor, is what the copy reads
    _verdict(
        "a staged upload is asynchronous",
        elapsed < HOST_LIMIT,
        "the host call took %.3f s behind a kernel of about 1 s" % elapsed,
    )
    torch.cuda.synchronize()
    _verdict(
        "a staged upload delivers the tensor",
        torch.equal(device.cpu(), expected),
        "compared",
    )


def test_pull_to_host_waits_for_the_work_before_it_and_gives_its_values(gpu):
    sf = _need("dynamicemb.sync_free")
    x = torch.zeros(1000, dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    torch.cuda._sleep(BUSY_CYCLES // 2)
    x.add_(5)  # behind the long kernel
    (host,), elapsed = _timed(lambda: sf.pull_to_host(x))
    _verdict(
        "pull_to_host waits for the kernels queued before it",
        elapsed > 0.05,
        "the call took %.3f s behind a kernel of about 0.5 s" % elapsed,
    )
    _verdict("pull_to_host gives the final values", bool((host == 5).all()), "compared")


def test_a_staging_ring_of_two_waits_for_a_buffer_when_the_device_is_busy(gpu):
    """The floor that load and replay state: the third upload of a ring of two reuses the
    first buffer, and the device is still busy with that transfer."""
    sf = _need("dynamicemb.sync_free")
    ring = sf.StagedUpload(torch.device("cuda", torch.cuda.current_device()), ring=2)
    host = torch.randn(ROWS, 8)
    torch.cuda.synchronize()
    torch.cuda._sleep(BUSY_CYCLES)  # every transfer below is queued behind this
    _, first = _timed(lambda: ring.upload(host))
    _, second = _timed(lambda: ring.upload(host))
    _, third = _timed(lambda: ring.upload(host))
    _verdict(
        "the first two uploads of a ring do not wait",
        max(first, second) < HOST_LIMIT,
        "they took %.3f s and %.3f s" % (first, second),
    )
    _verdict(
        "the third upload waits for the first buffer while the device is busy",
        third > 0.05,
        "it took %.3f s" % third,
    )
    torch.cuda.synchronize()
