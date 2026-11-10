# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rehash kernels on the CPU: the real source, host stand-ins, no GPU, no torch.

``table_carry_slot_state`` and ``table_carry_reclaimed_rows`` (``src/table_operation/
table_rows.cu``) move what is keyed by the slot of a key, and the rows of the erased slots,
when a table is rehashed without a count read on the host. nvcc is not needed to check what
they compute: ``extract_rehash_kernels.py`` cuts them out of the source, ``kernel_stubs.h``
stands in for CUDA and ATen and runs every thread of every block one after the other, and
``test_rehash_kernels.cpp`` compares the reclaimed-row carry with the pair of kernels it
replaced (``replaced_reclaimed_rows_kernels.inc``, vendored from the tree before the padded
rehash) on random tables, checks the properties that do not need that pair, and walks the
edges (a count of zero or below, a count that is too high or too low, an empty or a too small
range, a real key without a slot, the checks of the wrappers).

It cannot show that the code compiles under nvcc, or what a warp does with an atomic: that is
the GPU's. Needs a C++ compiler and nothing else.
"""

import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
TABLE_ROWS = os.path.join(ROOT, "src", "table_operation", "table_rows.cu")


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
    out = tmp_path_factory.mktemp("host_only_rehash_kernels")
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "extract_rehash_kernels.py"),
            TABLE_ROWS,
            str(out),
        ],
        check=True,
    )
    exe = str(out / "test_rehash_kernels")
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-Wall",
            "-I",
            HERE,
            "-I",
            str(out),
            os.path.join(HERE, "test_rehash_kernels.cpp"),
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr[-4000:]
    return exe


def test_the_real_rehash_kernels_on_the_cpu(binary):
    run = subprocess.run([binary], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and "ALL PASSED" in run.stdout, (
        run.stdout[-3000:] + run.stderr[-2000:]
    )
    # the three parts ran
    for part in ("e8 == collect+place", "e8 edges done", "e7:"):
        assert part in run.stdout, run.stdout
