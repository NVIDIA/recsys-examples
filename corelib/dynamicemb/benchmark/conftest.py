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
        choices=["torch", "nsys", "ncu-gen", "ncu-run"],
        help=(
            "Profiling mode: 'torch' for torch.profiler, 'nsys' for NVTX only, "
            "'ncu-gen' to print ncu commands without running tests, "
            "'ncu-run' to run a single-iteration benchmark under ncu."
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
