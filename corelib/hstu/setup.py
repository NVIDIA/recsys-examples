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
# # Copyright (c) 2023, Tri Dao.
# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

import itertools
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import torch
from packaging.version import Version, parse
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
root_path: Path = Path(__file__).resolve().parent

cmd = ["git", "rev-parse", "HEAD"]
sha = subprocess.check_output(cmd, cwd=str(root_path)).decode("ascii").strip()

PACKAGE_NAME = "hstu_attn"

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        PACKAGE_NAME,
        "--break-system-packages",
    ]
)

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("HSTU_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("HSTU_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("HSTU_FORCE_CXX11_ABI", "FALSE") == "TRUE"

DISABLE_BACKWARD = os.getenv("HSTU_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_DETERMINISTIC = os.getenv("HSTU_DISABLE_DETERMINISTIC", "TRUE") == "TRUE"
DISABLE_BF16 = os.getenv("HSTU_DISABLE_BF16", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("HSTU_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_HDIM32 = os.getenv("HSTU_DISABLE_HDIM32", "FALSE") == "TRUE"
DISABLE_HDIM64 = os.getenv("HSTU_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("HSTU_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("HSTU_DISABLE_HDIM256", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("HSTU_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_CAUSAL = os.getenv("HSTU_DISABLE_CAUSAL", "FALSE") == "TRUE"
DISABLE_CONTEXT = os.getenv("HSTU_DISABLE_CONTEXT", "FALSE") == "TRUE"
DISABLE_TARGET = os.getenv("HSTU_DISABLE_TARGET", "FALSE") == "TRUE"
DISABLE_ARBITRARY = os.getenv("HSTU_DISABLE_ARBITRARY", "FALSE") == "TRUE"
ARBITRARY_NFUNC = int(os.getenv("HSTU_ARBITRARY_NFUNC", "0"))
DISABLE_RAB = os.getenv("HSTU_DISABLE_RAB", "FALSE") == "TRUE"
DISABLE_DRAB = os.getenv("HSTU_DISABLE_DRAB", "FALSE") == "TRUE"
DISABLE_86OR89 = os.getenv("HSTU_DISABLE_86OR89", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return ["--threads", nvcc_threads]


def generate_cuda_sources():
    ARCH_SM = ["80"] + (["89"] if not DISABLE_86OR89 else [])
    DTYPE_FWD_SM80 = (["bf16"] if not DISABLE_BF16 else []) + (
        ["fp16"] if not DISABLE_FP16 else []
    )
    DTYPE_BWD_SM80 = (["bf16"] if not DISABLE_BF16 else []) + (
        ["fp16"] if not DISABLE_FP16 else []
    )
    HEAD_DIMENSIONS = (
        []
        + ([32] if not DISABLE_HDIM32 else [])
        + ([64] if not DISABLE_HDIM64 else [])
        + ([128] if not DISABLE_HDIM128 else [])
        + ([256] if not DISABLE_HDIM256 else [])
    )
    RAB = [""] + (["_rab"] if not DISABLE_RAB else [])
    RAB_DRAB = [""] + (
        (["_rab_drab", "_rab"])
        if not DISABLE_DRAB
        else ["_rab"]
        if not DISABLE_RAB
        else []
    )
    MASK = [""]
    if not DISABLE_LOCAL:
        MASK += ["_local"]
    if not DISABLE_CAUSAL:
        CAUSAL_MASK = ["_causal"]
        CONTEXT_MASK = [""] + (["_context"] if not DISABLE_CONTEXT else [])
        TARGET_MASK = [""] + (["_target"] if not DISABLE_TARGET else [])
        MASK += [
            f"{c}{x}{t}"
            for c, x, t in itertools.product(CAUSAL_MASK, CONTEXT_MASK, TARGET_MASK)
        ]
    if not DISABLE_ARBITRARY:
        MASK += ["_arbitrary"]
    BWD_DETERMINISTIC = ["false"] + (["true"] if not DISABLE_DETERMINISTIC else [])
    dtype_to_str = {
        "bf16": "cutlass::bfloat16_t",
        "fp16": "cutlass::half_t",
    }

    subprocess.run(["rm", "-rf", "csrc/hstu_attn/src/generated/*"])
    sources_fwd_sm80 = []
    fwd_file_head = """
// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
// Splitting different head dimensions, data types and masks to different files to speed up
// compilation. This file is auto-generated. See generate_cuda_sources() in setup.py

#include "hstu_fwd.h"

template void run_hstu_fwd_<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>
                           (Hstu_fwd_params& params, cudaStream_t stream);

    """
    for hdim, dtype, rab, mask, arch_sm in itertools.product(
        HEAD_DIMENSIONS, DTYPE_FWD_SM80, RAB, MASK, ARCH_SM
    ):
        file_name = (
            f"csrc/hstu_attn/src/generated/flash_fwd_hdim{hdim}_{dtype}{rab}{mask}_fn{ARBITRARY_NFUNC}_sm{arch_sm}.cu"
            if "arbitrary" in mask
            else f"csrc/hstu_attn/src/generated/flash_fwd_hdim{hdim}_{dtype}{rab}{mask}_sm{arch_sm}.cu"
        )
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(
                    fwd_file_head.format(
                        arch_sm,
                        dtype_to_str[dtype],
                        hdim,
                        "true" if "_rab" in rab else "false",
                        "true" if "local" in mask else "false",
                        "true" if "causal" in mask else "false",
                        "true" if "context" in mask else "false",
                        "true" if "target" in mask else "false",
                        "true" if "arbitrary" in mask else "false",
                        str(ARBITRARY_NFUNC) if "arbitrary" in mask else "0",
                    )
                )
        sources_fwd_sm80.append(file_name)

    sources_bwd_sm80 = []
    bwd_file_head = """
// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
// Splitting different head dimensions, data types and masks to different files to speed up
// compilation. This file is auto-generated. See generate_cuda_sources() in setup.py

#include "hstu_bwd.h"

template void run_hstu_bwd_<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>
                           (Hstu_bwd_params& params, cudaStream_t stream);

    """
    if not DISABLE_BACKWARD:
        for (
            hdim,
            dtype,
            rab_drab,
            mask,
            bwd_deterministic,
            arch_sm,
        ) in itertools.product(
            HEAD_DIMENSIONS, DTYPE_BWD_SM80, RAB_DRAB, MASK, BWD_DETERMINISTIC, ARCH_SM
        ):
            file_name = (
                f"csrc/hstu_attn/src/generated/flash_bwd_hdim{hdim}_{dtype}{rab_drab}{mask}_fn{ARBITRARY_NFUNC}_{bwd_deterministic}_sm{arch_sm}.cu"
                if "arbitrary" in mask
                else f"csrc/hstu_attn/src/generated/flash_bwd_hdim{hdim}_{dtype}{rab_drab}{mask}_{bwd_deterministic}_sm{arch_sm}.cu"
            )
            if not os.path.exists(file_name):
                with open(file_name, "w") as f:
                    f.write(
                        bwd_file_head.format(
                            arch_sm,
                            dtype_to_str[dtype],
                            hdim,
                            "true" if "_rab" in rab_drab else "false",
                            "true" if "drab" in rab_drab else "false",
                            "true" if "local" in mask else "false",
                            "true" if "causal" in mask else "false",
                            "true" if "context" in mask else "false",
                            "true" if "target" in mask else "false",
                            "true" if "arbitrary" in mask else "false",
                            str(ARBITRARY_NFUNC) if "arbitrary" in mask else "0",
                            bwd_deterministic,
                        )
                    )
            sources_bwd_sm80.append(file_name)

    return sources_fwd_sm80 + sources_bwd_sm80


cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
subprocess.run(["git", "submodule", "update", "--init", "../../third_party/cutlass"])

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("hstu_attn")
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("11.6"):
        raise RuntimeError(
            "FlashAttention is only supported on CUDA 11.6 and above.  "
            "Note: make sure nvcc has a supported version by running nvcc -V."
        )
    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    # cc_flag.append("arch=compute_86,code=sm_86")

    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(this_dir)
    cutlass_dir = repo_dir / "../../third_party/cutlass"

    feature_args = (
        []
        + (["-DHSTU_DISABLE_BACKWARD"] if DISABLE_BACKWARD else [])
        + (["-DHSTU_DISABLE_DETERMINISTIC"] if DISABLE_DETERMINISTIC else [])
        + (["-DHSTU_DISABLE_BF16"] if DISABLE_BF16 else [])
        + (["-DHSTU_DISABLE_FP16"] if DISABLE_FP16 else [])
        + (["-DHSTU_DISABLE_HDIM32"] if DISABLE_HDIM32 else [])
        + (["-DHSTU_DISABLE_HDIM64"] if DISABLE_HDIM64 else [])
        + (["-DHSTU_DISABLE_HDIM128"] if DISABLE_HDIM128 else [])
        + (["-DHSTU_DISABLE_HDIM256"] if DISABLE_HDIM256 else [])
        + (["-DHSTU_DISABLE_LOCAL"] if DISABLE_LOCAL else [])
        + (["-DHSTU_DISABLE_CAUSAL"] if DISABLE_CAUSAL else [])
        + (["-DHSTU_DISABLE_CONTEXT"] if DISABLE_CONTEXT else [])
        + (["-DHSTU_DISABLE_TARGET"] if DISABLE_TARGET else [])
        + (["-DHSTU_DISABLE_ARBITRARY"] if DISABLE_ARBITRARY else [])
        + (["-DHSTU_ARBITRARY_NFUNC=" + str(ARBITRARY_NFUNC)])
        + (["-DHSTU_DISABLE_RAB"] if DISABLE_RAB else [])
        + (["-DHSTU_DISABLE_DRAB"] if DISABLE_DRAB else [])
        + (["-DHSTU_DISABLE_86OR89"] if DISABLE_86OR89 else [])
    )

    if DISABLE_BF16 and DISABLE_FP16:
        raise ValueError("At least one of DISABLE_BF16 or DISABLE_FP16 must be False")
    if DISABLE_HDIM32 and DISABLE_HDIM64 and DISABLE_HDIM128 and DISABLE_HDIM256:
        raise ValueError(
            "At least one of DISABLE_HDIM32, DISABLE_HDIM64, DISABLE_HDIM128, or DISABLE_HDIM256 must be False"
        )
    if DISABLE_RAB and not DISABLE_DRAB:
        raise ValueError("Cannot support drab without rab")
    if DISABLE_CAUSAL and not DISABLE_TARGET:
        raise ValueError("Cannot support target without causal")
    if not DISABLE_ARBITRARY and ARBITRARY_NFUNC % 2 == 0:
        raise ValueError("ARBITRARY_NFUNC must be odd")

    torch_cpp_sources = ["csrc/hstu_attn/hstu_api.cpp"]
    cuda_sources = generate_cuda_sources()

    nvcc_flags = [
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
        # "--ptxas-options=-v",
        "-lineinfo",
    ]
    include_dirs = [
        Path(this_dir) / "csrc" / "hstu_attn" / "src",
        cutlass_dir / "include",
    ]

    sources = torch_cpp_sources + cuda_sources

    ext_modules.append(
        CUDAExtension(
            name="hstu_attn_2_cuda",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + feature_args,
                "nvcc": nvcc_threads_args() + nvcc_flags + cc_flag + feature_args,
            },
            include_dirs=include_dirs,
        )
    )


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


setup(
    name=PACKAGE_NAME,
    version="0.1.0" + "+" + sha[:7] + ".cu" + str(bare_metal_version),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    author="NVIDIA-DevTech",
    py_modules=["hstu_attn_interface"],
    description="HSTU Attention for Generative Recommendation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
