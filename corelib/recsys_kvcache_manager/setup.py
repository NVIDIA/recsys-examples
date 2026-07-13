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
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

library_name = "recsys_kvcache_manager"
root_path: Path = Path(__file__).resolve().parent
runtime_ops_name = "kvcache_manager_ops.so"


def get_version():
    """Get version from git or environment variable."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(cmd, cwd=str(root_path)).decode("ascii").strip()
    except Exception:
        sha = None

    if "BUILD_VERSION" in os.environ:
        version = os.environ["BUILD_VERSION"]
    else:
        # Use a default version if no version file exists
        version = "0.1.0"

    if sha is None:
        sha = "Unknown"
    return version, sha


def nvcc_threads_args():
    """Get NVCC threads configuration."""
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return ["--threads", nvcc_threads]


def parse_torch_cuda_arch_list() -> Optional[str]:
    arch_list = os.getenv("TORCH_CUDA_ARCH_LIST", "").replace(";", " ").split()
    if not arch_list:
        return None

    arch_map = {
        "7.0": "70",
        "7.5": "75",
        "8.0": "80",
        "8.6": "86",
        "8.9": "89",
        "9.0": "90",
        "9.0a": "90",
        "10.0": "100",
        "12.0": "120",
    }

    cmake_arches = []
    for arch in arch_list:
        normalized = arch.lower().removesuffix("+ptx")
        mapped = arch_map.get(normalized)
        if mapped and mapped not in cmake_arches:
            cmake_arches.append(mapped)

    return ";".join(cmake_arches) if cmake_arches else None


def get_extensions():
    """Build CUDA extension modules."""
    nvcc_flags = [
        "-g",
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]

    cuda_sources = [
        "src/pybind.cpp",
        "src/gpu_kvcache_manager_impl.cpp",
        "src/native_host_kvcache_manager_impl.cpp",
        "src/gather_scatter.cpp",
        "src/gather_scatter_kernels.cu",
    ]

    include_dirs = [
        str(root_path / "src"),
        str(root_path / "src/runtime"),
        str(root_path / "src/torch_binding"),
    ]

    ext_modules = [
        CUDAExtension(
            name="kvcache_cpp",
            sources=cuda_sources,
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++20",
                    "-DWITH_PYBIND11=1",
                    "-fvisibility=hidden",
                ],
                "nvcc": nvcc_threads_args() + nvcc_flags,
            },
            include_dirs=include_dirs,
            libraries=["zmq"],
            extra_link_args=["-lzmq"],
        ),
    ]

    return ext_modules


class NinjaBuildExtension(BuildExtension):
    """Custom build extension with Ninja support and memory management."""

    def __init__(self, *args, **kwargs) -> None:
        # Do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            try:
                import psutil

                # Calculate the maximum allowed NUM_JOBS based on cores
                max_num_jobs_cores = max(1, os.cpu_count() // 2)

                # Calculate the maximum allowed NUM_JOBS based on free memory
                free_memory_gb = psutil.virtual_memory().available / (1024**3)
                # Each JOB peak can exceed 9GB with 4-arch nvcc; use 12GB to reduce OOM
                max_num_jobs_memory = int(free_memory_gb / 12)

                # Pick lower value to minimize OOM and swap usage
                max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
                os.environ["MAX_JOBS"] = str(max_jobs)
            except ImportError:
                pass

        super().__init__(*args, **kwargs)

    def run(self):
        super().run()
        self._build_and_stage_runtime_ops()

    def _build_and_stage_runtime_ops(self) -> None:
        build_dir = Path(self.build_temp) / "kvcache_manager_ops"
        build_dir.mkdir(parents=True, exist_ok=True)

        cmake_configure_cmd = [
            "cmake",
            "-S",
            str(root_path),
            "-B",
            str(build_dir),
        ]
        cmake_arches = parse_torch_cuda_arch_list()
        if cmake_arches:
            cmake_configure_cmd.append(
                f"-DCMAKE_CUDA_ARCHITECTURES={cmake_arches}"
            )

        subprocess.check_call(cmake_configure_cmd, cwd=str(root_path))

        max_jobs = os.getenv("MAX_JOBS") or str(os.cpu_count() or 1)
        subprocess.check_call(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                "kvcache_manager_ops",
                "-j",
                max_jobs,
            ],
            cwd=str(root_path),
        )

        built_runtime_ops = build_dir / runtime_ops_name
        if not built_runtime_ops.exists():
            raise FileNotFoundError(
                f"Expected built runtime ops library at {built_runtime_ops}"
            )

        package_output_dir = Path(self.build_lib) / library_name
        package_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_runtime_ops, package_output_dir / runtime_ops_name)


package = find_packages(exclude=("*test",))
version, sha = get_version()

# with open(os.path.join(root_path, "README.md"), encoding="utf8") as f:
#     readme = f.read()

setup(
    name=library_name,
    version=version,
    author="NVIDIA Corporation.",
    maintainer="recsys-team",
    description="Recsys KVCache Manager - Dynamic KV-cache management for LLM inference",
    # long_description=readme,
    long_description_content_type="text/markdown",
    packages=package,
    package_data={library_name: [runtime_ops_name]},
    ext_modules=get_extensions(),
    license="Apache-2.0",
    keywords=[
        "pytorch",
        "kvcache",
        "recommendation systems",
        "inference",
        "cuda",
    ],
    python_requires=">=3.9",
    cmdclass={"build_ext": NinjaBuildExtension},
    install_requires=["torch"],
)
