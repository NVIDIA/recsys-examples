import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


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

setup(
    name="hstu_cuda_ops",
    description="HSTU CUDA ops",
    ext_modules=[
        CUDAExtension(
            name="hstu_cuda_ops",
            sources=[
                "ops/cuda_ops/csrc/jagged_tensor_op_cuda.cpp",
                "ops/cuda_ops/csrc/jagged_tensor_op_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_threads_args() + nvcc_flags,
            },
        ),
        CUDAExtension(
            name="paged_kvcache_ops",
            sources=[
                "ops/cuda_ops/csrc/paged_kvcache_ops_cuda.cpp",
                "ops/cuda_ops/csrc/paged_kvcache_ops_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-fvisibility=hidden"],
                "nvcc": nvcc_threads_args() + nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
