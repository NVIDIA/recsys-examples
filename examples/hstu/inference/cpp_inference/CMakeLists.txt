cmake_minimum_required(VERSION 3.18)
project(inferece_hstu_gr_ranking_exported_model LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{Torch_DIR}")

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Try to find fbgemm_gpu - optional but recommended
find_library(FBGEMM_GPU_LIB fbgemm_gpu_py PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)

# Try to find experimental HSTU FBGEMM ops library
find_library(FBGEMM_HSTU_EXPERIMENTAL_LIB fbgemm_gpu_experimental_hstu
  PATHS /usr/local/lib/python3.12/dist-packages/hstu ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH)

find_library(NVE_TORCH_OPS_LIB nve_torch_ops
  PATHS /usr/local/lib/python3.12/dist-packages/pynve/ ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH REQUIRED)

find_library(NVE_COMMON_LIB nve-common
  PATHS /usr/local/lib/python3.12/dist-packages/pynve/ ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH REQUIRED)

# Build a dedicated HSTU CUDA ops runtime library for C++ demos.
# This avoids loading CPython extension modules from a pure C++ process.
add_library(hstu_cuda_ops_runtime SHARED
  ../../../commons/ops/cuda_ops/csrc/kjt_aux_op.cpp
  ../../../commons/ops/cuda_ops/csrc/jagged_tensor_op_cuda.cpp
  ../../../commons/ops/cuda_ops/csrc/jagged_tensor_op_kernel.cu
)
target_link_libraries(hstu_cuda_ops_runtime PRIVATE ${TORCH_LIBRARIES})
target_compile_features(hstu_cuda_ops_runtime PRIVATE cxx_std_17)
set_target_properties(hstu_cuda_ops_runtime PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(inferece_hstu_gr_ranking_exported_model inferece_hstu_gr_ranking_exported_model.cpp)
set_source_files_properties(inferece_hstu_gr_ranking_exported_model.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(inferece_hstu_gr_ranking_exported_model PRIVATE CUDA::cudart)
target_link_libraries(inferece_hstu_gr_ranking_exported_model PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(inferece_hstu_gr_ranking_exported_model PRIVATE hstu_cuda_ops_runtime)
target_link_libraries(inferece_hstu_gr_ranking_exported_model PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(inferece_hstu_gr_ranking_exported_model PRIVATE ${NVE_TORCH_OPS_LIB} ${NVE_COMMON_LIB})
target_compile_definitions(inferece_hstu_gr_ranking_exported_model PRIVATE HSTU_CUDA_OPS_AVAILABLE=1)

target_include_directories(inferece_hstu_gr_ranking_exported_model PRIVATE
    ${TORCH_INCLUDE_DIRS}
    /home/junyiq/newscratch/spring26/trt-recsys/
    /home/junyiq/newscratch/spring26/trt-recsys/include
)

# If fbgemm_gpu is found, link against it to register fbgemm operators
if(FBGEMM_GPU_LIB)
  message(STATUS "Found fbgemm_gpu library: ${FBGEMM_GPU_LIB}")
  target_link_libraries(inferece_hstu_gr_ranking_exported_model PRIVATE ${FBGEMM_GPU_LIB})
  target_compile_definitions(inferece_hstu_gr_ranking_exported_model PRIVATE FBGEMM_GPU_AVAILABLE=1)
else()
  message(STATUS "fbgemm_gpu library not found. FBGEMM operators may not work.")
  message(STATUS "To use FBGEMM operators, install fbgemm_gpu and set appropriate CMake paths.")
endif()

if(FBGEMM_HSTU_EXPERIMENTAL_LIB)
  message(STATUS "Found fbgemm experimental hstu library: ${FBGEMM_HSTU_EXPERIMENTAL_LIB}")
  target_link_libraries(inferece_hstu_gr_ranking_exported_model PRIVATE ${FBGEMM_HSTU_EXPERIMENTAL_LIB})
else()
  message(STATUS "fbgemm experimental hstu library not found. Will rely on runtime dlopen path.")
endif()

target_compile_features(inferece_hstu_gr_ranking_exported_model PRIVATE cxx_std_17)
target_compile_options(inferece_hstu_gr_ranking_exported_model PRIVATE -w)

if(NOT NVE_CACHE_LINE_SIZE)
  if (WIN32)
    set(NVE_CACHE_LINE_SIZE "64")
  else()
    execute_process(
      COMMAND getconf LEVEL1_DCACHE_LINESIZE
      OUTPUT_VARIABLE NVE_CACHE_LINE_SIZE
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
endif()

target_compile_options(inferece_hstu_gr_ranking_exported_model PRIVATE 
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