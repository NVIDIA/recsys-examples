cmake_minimum_required(VERSION 3.18)
project(inference_hstu_gr_ranking_exported_model LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{Torch_DIR}")

set(NVE_INSTALL_ROOT "/opt/nve" CACHE PATH "NVE install root")
set(NVE_2605_ROOT "/workspace/deps/nve-26.05" CACHE PATH "NVE 26.05 source tree")
set(NVE_DEFAULT_ROOT "/workspace/deps/nve" CACHE PATH "Default NVE source tree")
set(NVE_2605_LIB_DIR "${NVE_INSTALL_ROOT}/26.05/python/pynve" CACHE PATH "NVE 26.05 native-library directory")
set(NVE_DEFAULT_LIB_DIR "${NVE_INSTALL_ROOT}/default/python/pynve" CACHE PATH "Default NVE native-library directory")
set(NVE_2605_REPLAY_DIR "${NVE_INSTALL_ROOT}/26.05/replay")
set(NVE_DEFAULT_REPLAY_DIR "${NVE_INSTALL_ROOT}/default/replay")
set(NVTX_INCLUDE_DIR "/workspace/deps/NVTX/c/include")

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Match the libstdc++ ABI used by PyTorch for the loader manager.
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_compile_options(${TORCH_CXX_FLAGS_LIST})

# Try to find fbgemm_gpu - optional but recommended
find_library(FBGEMM_GPU_LIB fbgemm_gpu_py PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)

# Try to find experimental HSTU FBGEMM ops library
find_library(FBGEMM_HSTU_EXPERIMENTAL_LIB fbgemm_gpu_experimental_hstu
  PATHS /usr/local/lib/python3.12/dist-packages/hstu ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH)

find_library(NVE_2605_TORCH_OPS_LIB nve-torch-ops
  PATHS "${NVE_2605_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(NVE_2605_COMMON_LIB nve-common
  PATHS "${NVE_2605_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(NVE_DEFAULT_TORCH_OPS_LIB nve-torch-ops
  PATHS "${NVE_DEFAULT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(NVE_DEFAULT_COMMON_LIB nve-common
  PATHS "${NVE_DEFAULT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

# Build a dedicated HSTU CUDA ops runtime library for C++ demos.
# This avoids loading CPython extension modules from a pure C++ process.
add_library(hstu_cuda_ops_runtime SHARED
  ../../../commons/ops/cuda_ops/csrc/kjt_aux_op.cpp
  ../../../commons/ops/cuda_ops/csrc/jagged_tensor_op_cuda.cpp
  ../../../commons/ops/cuda_ops/csrc/jagged_tensor_op_kernel.cu
)
target_link_libraries(hstu_cuda_ops_runtime PRIVATE ${TORCH_LIBRARIES})
target_include_directories(hstu_cuda_ops_runtime PRIVATE
  ${TORCH_INCLUDE_DIRS}
  "${NVTX_INCLUDE_DIR}"
)
target_compile_features(hstu_cuda_ops_runtime PRIVATE cxx_std_20)
set_target_properties(hstu_cuda_ops_runtime PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

add_library(paged_kvcache_ops_runtime SHARED
  ../../../commons/ops/cuda_ops/csrc/paged_kvcache_ops_cuda.cpp
  ../../../commons/ops/cuda_ops/csrc/paged_kvcache_ops_kernel.cu
)
target_link_libraries(paged_kvcache_ops_runtime PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(paged_kvcache_ops_runtime PRIVATE CUDA::cudart)
target_link_libraries(paged_kvcache_ops_runtime PRIVATE /workspace/deps/nvcomp/lib/libnvcomp_static.a)
target_include_directories(paged_kvcache_ops_runtime PRIVATE
  ${TORCH_INCLUDE_DIRS}
  /workspace/deps/nvcomp/include
  ../../../commons/ops/cuda_ops/csrc
)
target_compile_features(paged_kvcache_ops_runtime PRIVATE cxx_std_20)
set_target_properties(paged_kvcache_ops_runtime PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# This manager has no NVE include path or link dependency.
add_library(recsys_nve_loader_manager STATIC nve_loader_plugin.cpp)
target_link_libraries(recsys_nve_loader_manager PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(recsys_nve_loader_manager PRIVATE
  RECSYS_NVE_INSTALL_ROOT="${NVE_INSTALL_ROOT}"
)

add_executable(inference_hstu_gr_ranking_exported_model inference_hstu_gr_ranking_exported_model.cpp)
set_source_files_properties(inference_hstu_gr_ranking_exported_model.cpp PROPERTIES LANGUAGE CUDA)

add_executable(inference_hstu_gr_ranking_kvcache_exported_model inference_hstu_gr_ranking_kvcache_exported_model.cpp)
set_source_files_properties(inference_hstu_gr_ranking_kvcache_exported_model.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE CUDA::cudart)
target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE CUDA::cuda_driver)
target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE hstu_cuda_ops_runtime)
target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE recsys_nve_loader_manager)
target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(inference_hstu_gr_ranking_exported_model PRIVATE HSTU_CUDA_OPS_AVAILABLE=1)

target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE CUDA::cudart)
target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE CUDA::cuda_driver)
target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE hstu_cuda_ops_runtime)
target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE paged_kvcache_ops_runtime)
target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE recsys_nve_loader_manager)
target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE HSTU_CUDA_OPS_AVAILABLE=1)

target_include_directories(inference_hstu_gr_ranking_exported_model PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_include_directories(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE
  ${TORCH_INCLUDE_DIRS}
)

# If fbgemm_gpu is found, link against it to register fbgemm operators
if(FBGEMM_GPU_LIB)
  message(STATUS "Found fbgemm_gpu library: ${FBGEMM_GPU_LIB}")
  target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE ${FBGEMM_GPU_LIB})
  target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE ${FBGEMM_GPU_LIB})
  target_compile_definitions(inference_hstu_gr_ranking_exported_model PRIVATE FBGEMM_GPU_AVAILABLE=1)
  target_compile_definitions(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE FBGEMM_GPU_AVAILABLE=1)
else()
  message(STATUS "fbgemm_gpu library not found. FBGEMM operators may not work.")
  message(STATUS "To use FBGEMM operators, install fbgemm_gpu and set appropriate CMake paths.")
endif()

if(FBGEMM_HSTU_EXPERIMENTAL_LIB)
  message(STATUS "Found fbgemm experimental hstu library: ${FBGEMM_HSTU_EXPERIMENTAL_LIB}")
  target_link_libraries(inference_hstu_gr_ranking_exported_model PRIVATE ${FBGEMM_HSTU_EXPERIMENTAL_LIB})
  target_link_libraries(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE ${FBGEMM_HSTU_EXPERIMENTAL_LIB})
else()
  message(STATUS "fbgemm experimental hstu library not found. Will rely on runtime dlopen path.")
endif()

target_compile_features(inference_hstu_gr_ranking_exported_model PRIVATE cxx_std_20)
target_compile_options(inference_hstu_gr_ranking_exported_model PRIVATE -w)
target_compile_features(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE cxx_std_20)
target_compile_options(inference_hstu_gr_ranking_kvcache_exported_model PRIVATE -w)

if(NOT NVE_CACHE_LINE_SIZE)
  if (WIN32)
    set(NVE_CACHE_LINE_SIZE "64")
  else()
    execute_process(
      COMMAND getconf LEVEL1_DCACHE_LINESIZE
      OUTPUT_VARIABLE NVE_CACHE_LINE_SIZE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
endif()

function(add_nve_loader_plugin
    target nve_root nve_lib_dir replay_dir torch_ops_lib common_lib)
  add_library(${target} MODULE nve_loader_plugin_impl.cu)
  set(version_script "${CMAKE_CURRENT_BINARY_DIR}/${target}.map")
  file(GENERATE OUTPUT "${version_script}" CONTENT
    "{\n  global:\n    recsys_nve_loader_create_state;\n    recsys_nve_loader_destroy_state;\n  local: *;\n};\n")
  target_include_directories(${target} PRIVATE
    "${nve_root}"
    "${nve_root}/include"
    "${nve_root}/third_party/json/include"
    "${NVTX_INCLUDE_DIR}"
    ${TORCH_INCLUDE_DIRS}
  )
  target_compile_definitions(${target} PRIVATE
    NVE_FEATURE_HT_PART_FNV1A=1
    NVE_FEATURE_HT_PART_MURMUR=1
    NVE_FEATURE_HT_PART_RRXMRRXMSX0=1
    NVE_FEATURE_HT_PART_STD_HASH=1
    NVE_FEATURE_HT_MASK_64=1
    NVE_FEATURE_HT_MASK_32=1
    NVE_FEATURE_HT_MASK_16=1
    NVE_FEATURE_HT_MASK_8=1
    NVE_FEATURE_HT_KEY_64=1
    NVE_FEATURE_HT_KEY_32=1
    NVE_FEATURE_HT_KEY_16=1
    NVE_FEATURE_HT_KEY_8=1
    NVE_CACHE_LINE_SIZE=${NVE_CACHE_LINE_SIZE}
  )
  target_link_options(${target} PRIVATE
    "LINKER:--no-as-needed"
    "LINKER:--version-script=${version_script}"
  )
  target_link_libraries(${target} PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    ${TORCH_LIBRARIES}
    "${torch_ops_lib}"
    "${common_lib}"
  )
  set_target_properties(${target} PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    OUTPUT_NAME recsys_nve_loader
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${target}"
    BUILD_RPATH "${nve_lib_dir}"
    INSTALL_RPATH "\$ORIGIN/../python/pynve"
  )
  install(TARGETS ${target} LIBRARY DESTINATION "${replay_dir}")
endfunction()

add_nve_loader_plugin(
  recsys_nve_loader_2605
  "${NVE_2605_ROOT}" "${NVE_2605_LIB_DIR}" "${NVE_2605_REPLAY_DIR}"
  "${NVE_2605_TORCH_OPS_LIB}" "${NVE_2605_COMMON_LIB}"
)
target_compile_definitions(recsys_nve_loader_2605 PRIVATE NVE_VERSION=2605)
add_nve_loader_plugin(
  recsys_nve_loader_default
  "${NVE_DEFAULT_ROOT}" "${NVE_DEFAULT_LIB_DIR}" "${NVE_DEFAULT_REPLAY_DIR}"
  "${NVE_DEFAULT_TORCH_OPS_LIB}" "${NVE_DEFAULT_COMMON_LIB}"
)
set(RECSYS_NVE_LOADER_TARGETS
  recsys_nve_loader_2605 recsys_nve_loader_default)

add_dependencies(inference_hstu_gr_ranking_exported_model ${RECSYS_NVE_LOADER_TARGETS})
add_dependencies(inference_hstu_gr_ranking_kvcache_exported_model ${RECSYS_NVE_LOADER_TARGETS})
