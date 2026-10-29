cmake_minimum_required(VERSION 3.18)
project(dynamicemb_exportable_embedding_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NVE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../third_party/nv-embedding-cache" CACHE PATH "NVE source root")
set(NVE_LIB_DIR "/opt/nve/default/python/pynve" CACHE PATH "NVE library directory")
set(NVTX_INCLUDE_DIR "/workspace/deps/NVTX/c/include" CACHE PATH "NVTX include directory")
set(DYNAMICEMB_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "DynamicEmb source root")
set(DYNAMICEMB_LIB_DIR "${DYNAMICEMB_ROOT}/torch_binding_build" CACHE PATH "DynamicEmb library directory")
include("${DYNAMICEMB_ROOT}/cmake/NVETarget.cmake")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_PREFIX_STATUS
)
if(NOT TORCH_PREFIX_STATUS EQUAL 0)
  message(FATAL_ERROR "Python cannot locate the PyTorch CMake package")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_file(INFERENCE_EMB_OPS inference_emb_ops.so PATHS "${DYNAMICEMB_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(NVE_TORCH_OPS nve-torch-ops PATHS "${NVE_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(NVE_COMMON nve-common PATHS "${NVE_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

add_executable(replay_and_verify replay_and_verify.cpp)
set_source_files_properties(replay_and_verify.cpp PROPERTIES LANGUAGE CUDA)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(replay_and_verify PRIVATE ${TORCH_CXX_FLAGS_LIST} -w)
target_link_options(replay_and_verify PRIVATE "LINKER:--no-as-needed")
target_include_directories(replay_and_verify PRIVATE
    "${DYNAMICEMB_ROOT}/include"
    "${NVE_ROOT}"
    "${NVE_ROOT}/include"
    "${NVE_ROOT}/third_party/json/include"
    "${NVTX_INCLUDE_DIR}"
    ${TORCH_INCLUDE_DIRS}
)
dynamicemb_configure_nve_target(replay_and_verify)
target_link_libraries(replay_and_verify PRIVATE
    ${TORCH_LIBRARIES}
    Threads::Threads
    CUDA::cudart
    CUDA::cuda_driver
    "${INFERENCE_EMB_OPS}"
    "${NVE_TORCH_OPS}"
    "${NVE_COMMON}"
)

# Unset selects the repository NVE (26.07 or later); set 26.05 explicitly for compatibility.
if(
  NOT DEFINED NVE_VERSION
  OR NVE_VERSION STREQUAL ""
  OR NVE_VERSION VERSION_GREATER_EQUAL "26.06"
)
  find_file(INFERENCE_EMB_UPDATE inference_emb_update.so PATHS "${DYNAMICEMB_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  target_link_libraries(replay_and_verify PRIVATE "${INFERENCE_EMB_UPDATE}")
else()
  target_compile_definitions(replay_and_verify PRIVATE DYNAMICEMB_NVE_2605=1)
endif()

set_target_properties(replay_and_verify PROPERTIES
    BUILD_RPATH "${DYNAMICEMB_LIB_DIR};${NVE_LIB_DIR}"
)
