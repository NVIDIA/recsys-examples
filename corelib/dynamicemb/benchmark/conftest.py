# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import os

import pytest
import torch
import torch.distributed as dist
from benchmark_utils import GPUTimer


def pytest_addoption(parser):
    parser.addoption(
        "--profile",
        action="store",
        default=None,
        choices=["torch", "nsys", "ncu-gen", "ncu"],
        help=(
            "Profiling mode: 'torch' for torch.profiler, 'nsys' for NVTX only, "
            "'ncu-gen' to print ncu commands without running tests, "
            "'ncu' to run a single-iteration benchmark under ncu."
        ),
    )
    parser.addoption(
        "--correctness",
        action="store_true",
        default=False,
        help=(
            "Force BenchmarkConfig.correctness=True on every parametrized "
            "config, enabling the forward-only TBE vs DynamicEmb comparison "
            "alongside the normal (profile=none) reporting/timing run. "
            "Configs that already set correctness=True in code are unaffected."
        ),
    )
    parser.addoption(
        "--num-iterations",
        action="store",
        type=int,
        default=None,
        help=(
            "Override BenchmarkConfig.num_iterations on every config (default "
            "100).  This sets the number of sampled batches, which also bounds "
            "the warmup/reporting loop and how many iterations each profile "
            "mode covers -- useful to keep `--profile ncu` tractable, e.g. "
            "`--profile ncu --num-iterations 3`."
        ),
    )
    parser.addoption(
        "--ncu-kernel-regex",
        action="store",
        default=None,
        help=(
            "Optional kernel-name regex for the generated ncu command "
            "(--profile ncu-gen), emitted verbatim as --kernel-name "
            "'regex:<value>', e.g. 'segmented_unique|table_insert'.  If omitted, "
            "no --kernel-name filter is emitted, so every kernel within the "
            "--nvtx-include scope is profiled -- use this to capture all kernels "
            "of an op range."
        ),
    )
    parser.addoption(
        "--ncu-iterations",
        action="store",
        default=None,
        help=(
            "Only used with `--profile ncu` (pass it to `--profile ncu-gen`): "
            "select which iterations ncu captures, widening the default of "
            "iteration 0 only.  Accepts a comma list ('0,3,7') or a Python-style "
            "slice 'begin:end:step' (end exclusive, parts optional: ':10', '5:', "
            "'::2', '2:20:3').  Implemented as one --nvtx-include "
            "'ncu_iter/iter_{i}/' per selected iter in the generated command; "
            "all iterations still run, unselected ones are filtered out by NVTX."
        ),
    )
    parser.addoption(
        "--num-tables",
        action="store",
        type=int,
        default=None,
        help=(
            "Override the number of tables on every parametrized config (e.g. "
            "1000).  The per-suite total capacity and total batch are held "
            "fixed, so each table gets total_cap//N rows and batch_size "
            "total_batch//N -- stresses the many-tables path without changing "
            "the overall HBM footprint or total key count."
        ),
    )
    parser.addoption(
        "--sparse-key-range",
        action="store",
        type=int,
        default=None,
        help=(
            "Override the sparse-key sampling range: each table draws indices "
            "from [.., N) instead of [.., per-table cap), controlling the "
            "duplicate rate independently of table capacity (smaller N -> more "
            "duplicates).  Should be <= per-table cap for in-range embedding "
            "lookups.  Default (None) samples over the per-table cap."
        ),
    )
    parser.addoption(
        "--no-torchrec",
        action="store_true",
        default=False,
        help=(
            "Skip the TorchRec/FBGEMM TBE baseline entirely (neither built nor "
            "run); only DynamicEmb is exercised and trc_* metrics are null.  "
            "Incompatible with --correctness (which needs the baseline)."
        ),
    )


@pytest.fixture(scope="session", autouse=True)
def dist_group():
    dist.init_process_group(backend="nccl")
    yield
    dist.barrier()
    dist.destroy_process_group()


@pytest.fixture(scope="session")
def device():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


@pytest.fixture(scope="session")
def timer():
    return GPUTimer()


@pytest.fixture(scope="session")
def profile_mode(request):
    return request.config.getoption("--profile")


@pytest.fixture(scope="session")
def correctness_flag(request):
    """Session-wide override for BenchmarkConfig.correctness from --correctness."""
    return request.config.getoption("--correctness")


@pytest.fixture(scope="session")
def num_iterations(request):
    """Session-wide override for BenchmarkConfig.num_iterations (None = keep config default)."""
    return request.config.getoption("--num-iterations")


@pytest.fixture(scope="session")
def ncu_iterations(request):
    """Session-wide --ncu-iterations spec for which iterations ncu captures (None = iter 0)."""
    return request.config.getoption("--ncu-iterations")


@pytest.fixture(scope="session")
def ncu_kernel_regex(request):
    """Session-wide --ncu-kernel-regex: user-supplied kernel-name regex for ncu-gen."""
    return request.config.getoption("--ncu-kernel-regex")


@pytest.fixture(scope="session")
def num_tables(request):
    """Session-wide override for the table count (None = keep config default)."""
    return request.config.getoption("--num-tables")


@pytest.fixture(scope="session")
def sparse_key_range(request):
    """Session-wide override for the sparse-key sampling range (None = per-table cap)."""
    return request.config.getoption("--sparse-key-range")


@pytest.fixture(scope="session")
def no_torchrec(request):
    """Session-wide flag: when True, skip the TorchRec baseline."""
    return request.config.getoption("--no-torchrec")


@pytest.fixture(autouse=True)
def _gpu_mem_cleanup():
    """Reclaim CUDA memory between tests, even when a test raises.

    run_single_benchmark's own `del + empty_cache` is skipped on exception
    paths, so a single OOM was cascading into every subsequent suite. This
    finally-block runs unconditionally and lets each test start from a clean
    allocator state.
    """
    yield
    gc.collect()
    torch.cuda.empty_cache()
