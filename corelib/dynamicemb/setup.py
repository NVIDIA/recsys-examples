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

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "dynamicemb",
        "--break-system-packages",
    ]
)
subprocess.run(["pip", "install", "ordered-set", "--break-system-packages"])

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


def find_source_files(
    directory,
    extension_pattern,
    exclude_dirs=[],
    exclude_files=[],
):
    source_files = []
    pattern = re.compile(extension_pattern)
    for root, dirs, files in os.walk(directory):
        if any(os.path.basename(dir) in exclude_dirs for dir in dirs):
            continue

        for file in files:
            if file in exclude_files:
                continue
            if pattern.search(file):
                full_path = os.path.join(root, file)
                source_files.append(full_path)
    return source_files


check_torchrec_version()

library_name = "dynamicemb"

root_path: Path = Path(__file__).resolve().parent


def get_version():
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(cmd, cwd=str(root_path)).decode("ascii").strip()
    except Exception:
        sha = None

    if "BUILD_VERSION" in os.environ:
        version = os.environ["BUILD_VERSION"]
    else:
        with open(os.path.join(root_path, "version.txt"), "r") as f:
            version = f.readline().strip()
        if sha is not None and "OFFICIAL_RELEASE" not in os.environ:
            version += "+" + sha[:7]

    if sha is None:
        sha = "Unknown"
    return version, sha


# Target GPU compute capabilities (no dot). Parsed in two places: the main
# extension's -gencode flags and the LruLfu evict fatbins. Override with
# DEMB_ARCHS="80;90;100".
DEMB_ARCHS = [
    a.strip()
    for a in os.environ.get("DEMB_ARCHS", "75;80;90;100;120")
    .replace(",", ";")
    .split(";")
    if a.strip()
]


def _gencode_flags(code_kind):
    """nvcc -gencode flags for DEMB_ARCHS. code_kind is 'sm' (complete SASS) or
    'lto' (LTO-IR, for the custom evict fatbin linked at runtime)."""
    flags = []
    for a in DEMB_ARCHS:
        flags += ["-gencode", f"arch=compute_{a},code={code_kind}_{a}"]
    return flags


def get_extensions():
    extra_link_args = [
        "-Wl,--no-as-needed",
        "-lcuda",  # CUDA drive API
        "-lnvJitLink",  # runtime link of numba LTO-IR into the LruLfu evict cubin
    ]
    extra_compile_args = {
        "cxx": ["-O3", "-fdiagnostics-color=always", "-w", "-DDEMB_USE_PYBIND11"],
        "nvcc": [
            "-O3",
            # SASS<->source line mapping for ncu's Source page / --import-source.
            # Does not disable device optimizations (unlike -G), so runtime
            # kernel performance is unaffected.
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            *_gencode_flags("sm"),
            "-w",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-DDEMB_USE_PYBIND11",
        ],
    }

    cuda_sources = find_source_files(
        os.path.join(root_path, "src"),
        r".*\.cu$|.*\.cpp$|.*\.c$|.*\.cxx$",
        exclude_files=[
            "lookup_torch_binding.cu",
            "get_table_range_torch_binding.cu",
            "expand_table_ids_torch_binding.cu",
            "indexer_directory.cpp",
            "indexer_snapshot.cpp",
            "indexer_ops.cu",
            "indexer_cpu.cpp",
            "incremental_update.cpp",
            "nve_runtime.cu",
            "nve_tier_pybind.cu",
            "exportable_embedding_pybind.cpp",
            "indexer_directory_pybind.cpp",
            "indexer_snapshot_pybind.cpp",
            "update_subscriber_pybind.cu",
            "update_subscriber_unavailable_pybind.cpp",
            # Built separately into standalone fatbins (Lex + custom LTO-IR),
            # shipped as package_data; NOT linked into the .so.
            "evict_lrulfu.cu",
        ],
    )

    include_dirs = [
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

EVICT_TU = "src/jit/evict_lrulfu.cu"


def compile_evict_fatbins():
    """Build the two LruLfu eviction fatbins shipped as package_data (for
    DEMB_ARCHS):
      - evict_lrulfu_lex.fatbin    : complete SASS fatbin (LexFreqTsComparator),
                                     the default evictor, no numba at runtime.
      - evict_lrulfu_custom.fatbin : multi-arch LTO-IR fatbin (UserFnComparator,
                                     user_score_fn undefined) for nvJitLink to
                                     link the numba-compiled score_function into.
    Both come from the same TU, selected by -DDEMB_EVICT_COMPARATOR."""
    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = os.path.join(CUDA_HOME or "/usr/local/cuda", "bin", "nvcc")
    src = str(root_path / EVICT_TU)
    out_dir = root_path / library_name / "jit"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Match the main extension's nvcc flags (get_extensions): the fatbin TU
    # compiles the same kernels.cuh / <cub/cub.cuh> / cooperative_groups headers,
    # so it needs the same relaxed-constexpr + extended-lambda support to stay
    # buildable across CUDA versions.
    common = [
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        f"-I{root_path / 'src'}",
    ]

    variants = [
        ("LexFreqTsComparator", "evict_lrulfu_lex.fatbin", "sm"),
        ("UserFnComparator", "evict_lrulfu_custom.fatbin", "lto"),
    ]
    for comparator, out_name, code_kind in variants:
        out = str(out_dir / out_name)
        cmd = [
            nvcc,
            "--fatbin",
            *_gencode_flags(code_kind),
            *common,
            f"-DDEMB_EVICT_COMPARATOR={comparator}",
            src,
            "-o",
            out,
        ]
        print(f"[dynamicemb] nvcc evict fatbin ({comparator}): {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[dynamicemb] {out_name}: {os.path.getsize(out)} bytes")


def build_inference_extensions(build_lib, inplace):
    """Reuse the CMake build and package its bindings with their local libraries."""
    package_dir = root_path / library_name / "exportable_embedding"
    destination = (
        package_dir
        if inplace
        else Path(build_lib) / library_name / "exportable_embedding"
    )
    libraries = (
        "_C.so",
        "_C_legacy.so",
        "inference_emb_ops.so",
        "inference_emb_update.so",
    )
    destination.mkdir(parents=True, exist_ok=True)
    if not inplace:
        # build_py can have copied artifacts from an earlier, different build.
        for name in libraries:
            (destination / name).unlink(missing_ok=True)
    nve_spec = importlib.util.find_spec("pynve")
    enabled = os.environ.get("DYNAMICEMB_BUILD_INFERENCE", "1" if nve_spec else "0")
    if enabled not in {"0", "1"}:
        raise ValueError("DYNAMICEMB_BUILD_INFERENCE must be 0 or 1")
    if enabled == "0":
        print(
            "[dynamicemb] inference bindings disabled; set DYNAMICEMB_BUILD_INFERENCE=1 to enable"
        )
        return
    if nve_spec is None:
        raise RuntimeError(
            "Inference bindings require pynve; install NVE before building DynamicEmb"
        )
    import pynve

    build_dir = root_path / "torch_binding_build"
    command = [
        "cmake",
        "-S",
        str(root_path),
        "-B",
        str(build_dir),
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DNVE_VERSION={pynve.__version__}",
        f"-DNVE_LIB_DIR={os.environ.get('NVE_LIB_DIR', str(Path(pynve.__file__).parent))}",
        "-DDYNAMICEMB_ENABLE_NVE_UPDATE=ON",
    ]
    for name in ("NVE_ROOT", "NVTX_INCLUDE_DIR"):
        if name in os.environ:
            command.append(f"-D{name}={os.environ[name]}")
    # Preserve an explicit CMake configuration (including Docker's GPU targets).
    if not (build_dir / "CMakeCache.txt").exists():
        subprocess.run(command, check=True)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", os.environ["MAX_JOBS"]],
        check=True,
    )
    sources = [
        package_dir / "_C.so",
        package_dir / "_C_legacy.so",
        build_dir / "inference_emb_ops.so",
    ]
    update_library = build_dir / "inference_emb_update.so"
    if update_library.exists():
        sources.append(update_library)
    for source in sources:
        target = destination / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        print(f"[dynamicemb] staged {source.name} -> {target}")


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
            # each JOB peak can exceed 9GB with 4-arch nvcc; use 12GB to reduce OOM/bad_alloc
            max_num_jobs_memory = int(free_memory_gb / 12)

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

    def run(self):
        start_time = time.time()
        compile_evict_fatbins()
        super().run()
        # build_py (which collects package_data) runs BEFORE build_ext, so the
        # fatbins we just generated are not yet staged into build_lib for a
        # wheel/install -- copy them in now. Inplace builds import straight from
        # the source tree and don't need this.
        if not self.inplace:
            src_dir = root_path / library_name / "jit"
            dst_dir = os.path.join(self.build_lib, library_name, "jit")
            os.makedirs(dst_dir, exist_ok=True)
            for fb in sorted(src_dir.glob("*.fatbin")):
                shutil.copy2(str(fb), dst_dir)
                print(f"[dynamicemb] staged {fb.name} -> {dst_dir}")
        build_inference_extensions(self.build_lib, self.inplace)
        end_time = time.time()
        compilation_time = end_time - start_time
        print(f"compilation_time: {compilation_time}")


version, sha = get_version()

setup(
    name=library_name,
    version=version,
    author="NVIDIA Corporation.",
    maintainer="zehuanw",
    maintainer_email="zehuanw@nvidia.com",
    description="Plugin for Dynamic Embedding in TorchREC",
    packages=package,
    ext_modules=get_extensions(),
    package_data={
        f"{library_name}.jit": ["*.fatbin"],
        f"{library_name}.exportable_embedding": [
            "_C.so",
            "_C_legacy.so",
            "inference_emb*.so",
        ],
    },
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
