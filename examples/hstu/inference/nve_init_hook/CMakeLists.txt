# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds libnve_init_hook.so -- the NVE model-init hook dlopen()ed by the Triton
# PyTorch (AOTI) backend (see README.md). Build it in the same
# nvcr.io/nvidia/pytorch:XX.YY-py3 image whose libtorch matches the
# tritonserver:XX.YY-py3 you deploy on, with NVE available (pip-installed at
# .../dist-packages/pynve, headers under /workspace/deps/nve).

cmake_minimum_required(VERSION 3.18)
project(nve_init_hook LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(NVE_ROOT "/workspace/deps/nve"
    CACHE PATH "nv-embedding-cache source tree (for headers)")
set(NVE_LIB_DIR "/usr/local/lib/python3.12/dist-packages/pynve"
    CACHE PATH "Directory containing libnve-*.so")

find_library(NVE_TORCH_OPS_LIB nve-torch-ops
    PATHS ${NVE_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(NVE_COMMON_LIB nve-common
    PATHS ${NVE_LIB_DIR} NO_DEFAULT_PATH REQUIRED)

add_library(nve_init_hook SHARED nve_init_hook.cu)
set_target_properties(nve_init_hook PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES "90"
)
target_include_directories(nve_init_hook PRIVATE
  ${NVE_ROOT}
  ${NVE_ROOT}/include
  ${NVE_ROOT}/third_party/json/include/
)
target_link_libraries(nve_init_hook PRIVATE
  CUDA::cudart CUDA::cuda_driver ${NVE_TORCH_OPS_LIB} ${NVE_COMMON_LIB})

# NVE header feature flags (mirror examples/.../cpp_inference/CMakeLists.txt).
if(NOT NVE_CACHE_LINE_SIZE)
  execute_process(
    COMMAND getconf LEVEL1_DCACHE_LINESIZE
    OUTPUT_VARIABLE NVE_CACHE_LINE_SIZE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()
target_compile_options(nve_init_hook PRIVATE
  -DNVE_FEATURE_HT_PART_FNV1A=1
  -DNVE_FEATURE_HT_PART_MURMUR=1
  -DNVE_FEATURE_HT_PART_RRXMRRXMSX0=1
  -DNVE_FEATURE_HT_PART_STD_HASH=1
  -DNVE_FEATURE_HT_MASK_64=1
  -DNVE_FEATURE_HT_MASK_32=1
  -DNVE_FEATURE_HT_MASK_16=1
  -DNVE_FEATURE_HT_MASK_8=1
  -DNVE_FEATURE_HT_KEY_64=1
  -DNVE_FEATURE_HT_KEY_32=1
  -DNVE_FEATURE_HT_KEY_16=1
  -DNVE_FEATURE_HT_KEY_8=1
  -DNVE_CACHE_LINE_SIZE=${NVE_CACHE_LINE_SIZE}
)
