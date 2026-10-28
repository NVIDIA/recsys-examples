# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SANITIZED_TESTS = [
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_is_bit_identical_to_the_public_three_kernel_path",
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_is_bit_identical_at_any_beam_width",
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_is_bit_identical_for_every_context_layout",
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_is_bit_identical_at_long_decode_depths",
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_never_follows_top_k_entries_past_decode_nums",
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_handles_zero_decode_nums",
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_writes_into_out",
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_runs_the_public_path_for_head_dims_it_is_not_validated_for",
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_runs_the_public_path_for_int64_top_k",
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_runs_the_public_path_where_context_attention_splits",
    "tests/test_fused_beam_decode_gpu.py::test_fused_beam_decode_follows_the_public_backend_choice",
    "tests/test_fused_beam_decode_gpu.py::test_backend_runs_the_fused_kernel_bit_identically",
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
            "--report-api-errors",
            "no",
            sys.executable,
            "-m",
            "pytest",
            *SANITIZED_TESTS,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert " passed" in result.stdout and "skipped" not in result.stdout
