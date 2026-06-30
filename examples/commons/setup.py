import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return ["--threads", nvcc_threads]


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

_ALL_EXT_MODULES = [
    CUDAExtension(
        name="hstu_cuda_ops",
        sources=[
            "ops/cuda_ops/csrc/jagged_tensor_op_cuda.cpp",
            "ops/cuda_ops/csrc/jagged_tensor_op_kernel.cu",
            "ops/cuda_ops/csrc/kjt_aux_op.cpp",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-DWITH_PYBIND11=1"],
            "nvcc": nvcc_threads_args() + nvcc_flags,
        },
    ),
    CppExtension(
        # Pure-CPU Karmarkar-Karp k-way partitioner.  Lives outside the
        # CUDA extensions so that pip / setup.py users without CUDA
        # toolchains can still build it, and so the .so has zero CUDA
        # runtime dependencies.
        name="kk_cpu_ops",
        sources=["perf_model/csrc/kk_partition.cpp"],
        extra_compile_args=["-O3", "-std=c++17", "-fvisibility=hidden"],
    ),
    CUDAExtension(
        name="paged_kvcache_ops",
        sources=[
            "ops/cuda_ops/csrc/paged_kvcache_ops_cuda.cpp",
            "ops/cuda_ops/csrc/paged_kvcache_ops_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20", "-fvisibility=hidden", "-DWITH_PYBIND11=1"]
            + [
                "-I/workspace/deps/nvcomp/include",
                "-L/workspace/deps/nvcomp/lib",
                "-lnvcomp_static",
            ],
            "nvcc": nvcc_threads_args() + nvcc_flags,
        },
        include_dirs=[
            "/workspace/deps/nvcomp/include",
        ],
        library_dirs=[
            "/workspace/deps/nvcomp/lib",
        ],
        libraries=["nvcomp_static"],
    ),
]

# ``BUILD_EXT_ONLY`` filters ``ext_modules`` down to a comma-separated allowlist
# of extension names.  Benchmark slurm jobs use ``BUILD_EXT_ONLY=kk_cpu_ops`` to
# rebuild just the CPU partitioner inside the container without spending
# minutes recompiling the CUDA extensions that the image already ships.
_BUILD_EXT_ONLY = os.environ.get("BUILD_EXT_ONLY", "").strip()
if _BUILD_EXT_ONLY:
    _allow = {n.strip() for n in _BUILD_EXT_ONLY.split(",") if n.strip()}
    ext_modules = [m for m in _ALL_EXT_MODULES if m.name in _allow]
    missing = _allow - {m.name for m in _ALL_EXT_MODULES}
    if missing:
        raise SystemExit(
            f"BUILD_EXT_ONLY referenced unknown extensions: {sorted(missing)}. "
            f"Known: {[m.name for m in _ALL_EXT_MODULES]}"
        )
else:
    ext_modules = _ALL_EXT_MODULES

setup(
    name="hstu_cuda_ops",
    description="HSTU CUDA ops",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
