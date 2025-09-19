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
import re
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

subprocess.run(
    ["git", "submodule", "update", "--init", "../../third_party/HierarchicalKV"]
)

# TODO: update when torchrec release compatible commit.
compatible_versions = "1.2.0"


def check_torchrec_version():
    try:
        import torchrec

        version = re.match(r"^\d+\.\d+\.\d+", torchrec.__version__).group()
        if version >= compatible_versions:
            print(f"torchrec version {version} is installed.")
            return
        else:
            raise RuntimeError(
                f"torchrec version {version} is installed, but version >= {compatible_versions} is required."
            )
    except ImportError:
        raise RuntimeError("torchrec is not installed.")


def find_source_files(directory, extension_pattern, exclude_dirs=[]):
    source_files = []
    pattern = re.compile(extension_pattern)
    for root, dirs, files in os.walk(directory):
        if any(os.path.basename(dir) in exclude_dirs for dir in dirs):
            continue

        for file in files:
            if pattern.search(file):
                full_path = os.path.join(root, file)
                source_files.append(full_path)
    return source_files


check_torchrec_version()

library_name = "dynamicemb"

root_path: Path = Path(__file__).resolve().parent


def get_extensions():
    extra_link_args = []
    extra_compile_args = {
        "cxx": ["-O3", "-fdiagnostics-color=always", "-w"],
        "nvcc": [
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-gencode",
            "arch=compute_70,code=sm_70",
            "-gencode",
            "arch=compute_75,code=sm_75",
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_90,code=sm_90",
            "-w",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-UCUDA_ERROR_CHECK",  # this is to disable HKV error check
        ],
    }

    cuda_sources = find_source_files(
        os.path.join(root_path, "src"),
        r".*\.cu$|.*\.cpp$|.*\.c$|.*\.cxx$",
    )

    include_dirs = [
        root_path / "../../third_party/HierarchicalKV" / "include",
        root_path / "src",
    ]
    cuda_sources = [str(path) for path in cuda_sources]
    include_dirs = [str(path) for path in include_dirs]

    ext_modules = [
        CUDAExtension(
            f"{library_name}_extensions",
            cuda_sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=include_dirs,
            # libraries=['torch', 'c10'],
        )
    ]

    return ext_modules


package = find_packages(exclude=("*test",))

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf8") as f:
    readme = f.read()
import time


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (
                1024**3
            )  # free memory in GB
            max_num_jobs_memory = int(
                free_memory_gb / 9
            )  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

    def run(self):
        start_time = time.time()
        super().run()
        end_time = time.time()
        compilation_time = end_time - start_time
        print(f"compilation_time: {compilation_time}")


setup(
    name=library_name,
    version="0.0.1",
    author="NVIDIA Corporation.",
    maintainer="zehuanw",
    maintainer_email="zehuanw@nvidia.com",
    description="Plugin for Dynamic Embedding in TorchREC",
    packages=package,
    ext_modules=get_extensions(),
    license="BSD-3",
    keywords=[
        "pytorch",
        "torchrec",
        "recommendation systems",
        "dynamic embedding",
    ],
    python_requires=">=3.9",
    cmdclass={"build_ext": NinjaBuildExtension},
    install_requires=["torch"],
)
