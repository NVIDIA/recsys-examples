# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``HostVMMTensor`` on the CPU: the real class, a fake CUDA runtime, no GPU.

``src/vmm_tensor.cu`` cannot be compiled without nvcc, but ``HostVMMTensor`` (the host
buffer that grows in place) only needs ``mmap``, ``mprotect``, ``mlock`` and
``cudaHostRegister``. ``extract_host_class.py`` cuts the class out of the source,
``host_stubs.h`` stands in for the CUDA runtime, the driver API and ATen (and records
every registration, fails one on request, returns a device pointer that is not contiguous,
or takes its time) and ``test_host_vmm.cpp`` runs it:

* ``grow``: growth inside the reserved range, the base address and the first pages
  unchanged, one registration per new tail and none unregistered until the destructor;
* ``rollback``: a registration, a device pointer and an ``mlock`` that fail leave the
  buffer as it was and a retry works;
* ``reservation``: the default, the clamp to the installed memory, a reservation below what
  the buffer needs, and the halving under ``RLIMIT_AS``;
* ``ctor``: a constructor that fails gives its range back;
* ``concurrent``, ``readers``: what another thread sees while a buffer grows, and that a
  reader never waits for a commit (the thread that asks holds the GIL).

Needs a C++ compiler and nothing else (no torch).
"""

import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
VMM_SOURCE = os.path.join(ROOT, "src", "vmm_tensor.cu")
CASES = ["grow", "rollback", "reservation", "ctor", "concurrent", "readers"]


def _compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    compiler = _compiler()
    if compiler is None or sys.platform != "linux":
        pytest.skip("needs a C++ compiler on Linux")
    out = tmp_path_factory.mktemp("host_only_vmm")
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "extract_host_class.py"),
            VMM_SOURCE,
            str(out),
        ],
        check=True,
    )
    exe = str(out / "test_host_vmm")
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-Wall",
            "-Wextra",
            "-I",
            HERE,
            "-I",
            str(out),
            os.path.join(HERE, "test_host_vmm.cpp"),
            "-o",
            exe,
            "-pthread",
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr[-4000:]
    return exe


@pytest.mark.parametrize("case", CASES)
def test_the_real_host_buffer_against_a_fake_cuda_runtime(binary, case):
    run = subprocess.run([binary, case], capture_output=True, text=True, timeout=180)
    assert run.returncode == 0 and "ALL PASSED" in run.stdout, (
        run.stdout[-3000:] + run.stderr[-2000:]
    )
