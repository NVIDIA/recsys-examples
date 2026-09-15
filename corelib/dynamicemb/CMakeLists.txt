cmake_minimum_required(VERSION 3.18)
project(DynamicEmbInferenceOps LANGUAGES CXX CUDA)

include(GNUInstallDirs)

set(NVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/nv-embedding-cache" CACHE PATH "NVE source root")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(NOT DEFINED CMAKE_PREFIX_PATH)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE TORCH_PREFIX_STATUS
    )
    if(TORCH_PREFIX_STATUS EQUAL 0 AND TORCH_CMAKE_PREFIX_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
    endif()
endif()

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(DEFINED SM)
        set(CMAKE_CUDA_ARCHITECTURES ${SM})
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 89 90 100 120)
    endif()
endif()

message(STATUS "Found Torch ${TORCH_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(INFERENCE_EMB_SOURCES
    src/exportable_embedding/indexer_directory.cpp
    src/exportable_embedding/indexer_snapshot.cpp
    src/exportable_embedding/indexer_ops.cu
    src/table_operation/lookup_torch_binding.cu
    src/table_operation/get_table_range_torch_binding.cu
    src/table_operation/expand_table_ids_torch_binding.cu
    src/table_operation/lookup.cu
    src/index_calculation.cu
    src/unique_op.cu
    src/lookup_forward.cu
    src/torch_utils.cu
    src/utils.cpp
)

add_library(inference_emb_ops SHARED ${INFERENCE_EMB_SOURCES})

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
if(TORCH_CXX_FLAGS_LIST)
    target_compile_options(inference_emb_ops PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()

target_include_directories(inference_emb_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/table_operation
    ${NVE_ROOT}/third_party/json/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(inference_emb_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -fdiagnostics-color=always
        -fvisibility=hidden
        -w
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        # SASS<->source line mapping for ncu (does not disable optimizations).
        -lineinfo
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --use_fast_math
        -Xcompiler=-fvisibility=hidden
        -w
        -U__CUDA_NO_HALF_OPERATORS__
        -U__CUDA_NO_HALF_CONVERSIONS__
        -U__CUDA_NO_HALF2_OPERATORS__
        -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    >
)

target_link_libraries(inference_emb_ops PRIVATE
    Python3::Python
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

set_target_properties(inference_emb_ops PROPERTIES
    PREFIX ""
    OUTPUT_NAME "inference_emb_ops"
    CUDA_SEPARABLE_COMPILATION OFF
)

install(TARGETS inference_emb_ops
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Unset selects the repository NVE (26.07 or later); set 26.05 explicitly for compatibility.
if(
    NOT DEFINED NVE_VERSION
    OR NVE_VERSION STREQUAL ""
    OR NVE_VERSION VERSION_GREATER_EQUAL "26.06"
)
    set(DYNAMICEMB_HAS_NVE_UPDATE ON)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/NVETarget.cmake)
    set(NVE_LIB_DIR "/opt/nve/default/python/pynve" CACHE PATH "NVE library directory")
    set(NVTX_INCLUDE_DIR "/workspace/deps/NVTX/c/include" CACHE PATH "NVTX include directory")
    find_library(NVE_TORCH_OPS_LIB nve-torch-ops PATHS "${NVE_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
    find_library(NVE_COMMON_LIB nve-common PATHS "${NVE_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

    add_library(inference_emb_update SHARED
        src/exportable_embedding/incremental_update.cpp
    )
    set_source_files_properties(
        src/exportable_embedding/incremental_update.cpp
        PROPERTIES LANGUAGE CUDA
    )
    target_include_directories(inference_emb_update PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        "${NVE_ROOT}"
        "${NVE_ROOT}/include"
        "${NVE_ROOT}/third_party/json/include"
        "${NVTX_INCLUDE_DIR}"
        ${TORCH_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    if(TORCH_CXX_FLAGS_LIST)
        target_compile_options(inference_emb_update PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    target_link_options(inference_emb_update PRIVATE "LINKER:--no-as-needed")
    dynamicemb_configure_nve_target(inference_emb_update)
    target_link_libraries(inference_emb_update PRIVATE
        inference_emb_ops
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::cuda_driver
        "${NVE_TORCH_OPS_LIB}"
        "${NVE_COMMON_LIB}"
    )
    set_target_properties(inference_emb_update PROPERTIES
        PREFIX ""
        OUTPUT_NAME "inference_emb_update"
        INSTALL_RPATH "${NVE_LIB_DIR}"
    )
    install(TARGETS inference_emb_update
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
else()
    set(DYNAMICEMB_HAS_NVE_UPDATE OFF)
endif()

find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    REQUIRED
)

set(EXPORTABLE_EMBEDDING_PYBIND_SOURCES
    src/exportable_embedding/exportable_embedding_pybind.cpp
    src/exportable_embedding/indexer_directory_pybind.cpp
    src/exportable_embedding/indexer_snapshot_pybind.cpp
)
if(DYNAMICEMB_HAS_NVE_UPDATE)
    list(APPEND EXPORTABLE_EMBEDDING_PYBIND_SOURCES
        src/exportable_embedding/update_subscriber_pybind.cu
    )
else()
    list(APPEND EXPORTABLE_EMBEDDING_PYBIND_SOURCES
        src/exportable_embedding/update_subscriber_unavailable_pybind.cpp
    )
endif()

add_library(dynamicemb_exportable_embedding_python MODULE
    ${EXPORTABLE_EMBEDDING_PYBIND_SOURCES}
)
target_include_directories(dynamicemb_exportable_embedding_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(dynamicemb_exportable_embedding_python PRIVATE
    TORCH_EXTENSION_NAME=_C
)
if(TORCH_CXX_FLAGS_LIST)
    target_compile_options(dynamicemb_exportable_embedding_python PRIVATE
        ${TORCH_CXX_FLAGS_LIST}
    )
endif()
target_link_libraries(dynamicemb_exportable_embedding_python PRIVATE
    inference_emb_ops
    Python3::Python
    ${TORCH_LIBRARIES}
    "${TORCH_PYTHON_LIBRARY}"
)

if(DYNAMICEMB_HAS_NVE_UPDATE)
    target_include_directories(dynamicemb_exportable_embedding_python PRIVATE
        "${NVE_ROOT}"
        "${NVE_ROOT}/include"
        "${NVE_ROOT}/third_party/json/include"
        "${NVTX_INCLUDE_DIR}"
    )
    dynamicemb_configure_nve_target(dynamicemb_exportable_embedding_python)
    target_link_libraries(dynamicemb_exportable_embedding_python PRIVATE
        inference_emb_update
        CUDA::cudart
        "${NVE_TORCH_OPS_LIB}"
        "${NVE_COMMON_LIB}"
    )
endif()

set_target_properties(dynamicemb_exportable_embedding_python PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_C"
    LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/dynamicemb/exportable_embedding"
    INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR};${NVE_LIB_DIR}"
)
install(TARGETS dynamicemb_exportable_embedding_python
    LIBRARY DESTINATION
        "${Python3_SITEARCH}/dynamicemb/exportable_embedding"
    RUNTIME DESTINATION
        "${Python3_SITEARCH}/dynamicemb/exportable_embedding"
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
