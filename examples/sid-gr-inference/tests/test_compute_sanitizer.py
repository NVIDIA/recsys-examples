# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SANITIZED_TESTS = [
    "tests/test_decode_kv_writer.py::test_cuda_beam_kv_writer_handles_distinct_k_v_strides",
    "tests/test_decode_kv_writer.py::test_cuda_beam_kv_write_selects_vectorized_scalar_or_pytorch_copy",
    "tests/test_decode_kv_writer.py::test_cuda_beam_kv_write_vectorized_only_writes_nothing_it_cannot_vectorize",
]


@pytest.mark.parametrize("tool", ["memcheck", "racecheck"])
def test_cuda_kernel_tests_are_clean_under_compute_sanitizer(tool) -> None:
    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        pytest.skip("compute-sanitizer is not installed")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    result = subprocess.run(
        [
            sanitizer,
            "--tool",
            tool,
            "--error-exitcode",
            "1",
            sys.executable,
            "-m",
            "pytest",
            "-k",
            "not batch_over_grid_limit",
            *SANITIZED_TESTS,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert " passed" in result.stdout and "skipped" not in result.stdout
