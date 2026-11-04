cmake_minimum_required(VERSION 3.18)

# CPU mode uses CPU-only libtorch. Without the Python package, it builds just
# the indexing operators, with no CUDA toolkit, cudart or Python dependency.
option(DYNAMICEMB_CPU_ONLY "Build CPU inference against CPU-only PyTorch" OFF)
if(DYNAMICEMB_CPU_ONLY)
    set(_build_python_default OFF)
else()
    set(_build_python_default ON)
endif()
option(DYNAMICEMB_BUILD_PYTHON "Build the full inference package (requires NVE)" ${_build_python_default})
if(DYNAMICEMB_CPU_ONLY)
    project(DynamicEmbInferenceOps LANGUAGES CXX)
else()
    project(DynamicEmbInferenceOps LANGUAGES CXX CUDA)
endif()

include(GNUInstallDirs)

# setup.py also ships these libraries together inside the Python package.
list(APPEND CMAKE_BUILD_RPATH "$ORIGIN" "$ORIGIN/../../pynve" "$ORIGIN/../../torch/lib")

set(NVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/nv-embedding-cache" CACHE PATH "NVE source root")
set(DYNAMICEMB_PYTHON_MODULE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/dynamicemb/exportable_embedding"
    CACHE PATH "Output directory for the inference Python bindings"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DYNAMICEMB_CPU_ONLY AND NOT DYNAMICEMB_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter)
else()
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif()

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
if(DYNAMICEMB_CPU_ONLY AND TARGET torch_cuda)
    message(FATAL_ERROR "DYNAMICEMB_CPU_ONLY requires CPU-only PyTorch/libtorch")
endif()
message(STATUS "Found Torch ${TORCH_VERSION}")
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

if(DYNAMICEMB_CPU_ONLY)
    add_library(inference_emb_ops SHARED
        src/exportable_embedding/indexer_snapshot.cpp
        src/exportable_embedding/indexer_cpu.cpp
        src/exportable_embedding/inference_emb_schemas.cpp
    )
    target_compile_options(inference_emb_ops PRIVATE
        ${TORCH_CXX_FLAGS_LIST} -O3 -fvisibility=hidden -w)
    target_include_directories(inference_emb_ops PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
    )
    # at::parallel_for is header-inlined; without OpenMP it runs serially.
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(inference_emb_ops PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(inference_emb_ops PRIVATE ${TORCH_LIBRARIES})
    target_link_options(inference_emb_ops PRIVATE "LINKER:--no-undefined")
    set_target_properties(inference_emb_ops PROPERTIES
        PREFIX ""
        OUTPUT_NAME "inference_emb_ops"
    )
    install(TARGETS inference_emb_ops
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    if(NOT DYNAMICEMB_BUILD_PYTHON)
        return()
    endif()
    # Directory loading and marker rebinding use only CPU libtorch.
    target_sources(inference_emb_ops PRIVATE src/exportable_embedding/indexer_directory.cpp)
    target_include_directories(inference_emb_ops PRIVATE ${NVE_ROOT}/third_party/json/include)

    # NVE's public binding headers still contain CUDA code. The native builder
    # needs nvcc; the resulting Host package needs only cudart at deployment.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
    # nvcc's C++17 processing of PyTorch 2.10's dependent List type drops a
    # required typename. C++20 accepts that type in the NVE-facing bindings.
    set(CMAKE_CUDA_STANDARD 20)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
else()

find_package(CUDAToolkit REQUIRED)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(DEFINED SM)
        set(CMAKE_CUDA_ARCHITECTURES ${SM})
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 89 90 100 120)
    endif()
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(INFERENCE_EMB_SOURCES
    src/exportable_embedding/indexer_directory.cpp
    src/exportable_embedding/indexer_snapshot.cpp
    src/exportable_embedding/indexer_ops.cu
    src/exportable_embedding/indexer_cpu.cpp
    src/exportable_embedding/inference_emb_schemas.cpp
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
target_compile_definitions(inference_emb_ops PRIVATE DYNAMICEMB_WITH_CUDA)

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
endif()

if(NOT DYNAMICEMB_BUILD_PYTHON)
    return()
endif()

option(DYNAMICEMB_ENABLE_NVE_UPDATE "Build the NVE incremental update subscriber" ON)

# Unset selects the repository NVE (26.06 or later); set 26.05 explicitly for compatibility.
if(DYNAMICEMB_ENABLE_NVE_UPDATE AND (
    NOT DEFINED NVE_VERSION
    OR NVE_VERSION STREQUAL ""
    OR NVE_VERSION VERSION_GREATER_EQUAL "26.06"
))
    set(DYNAMICEMB_HAS_NVE_UPDATE ON)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/NVETarget.cmake)
    set(NVE_LIB_DIR "/opt/nve/default/python/pynve" CACHE PATH "NVE library directory")
    if(NOT DYNAMICEMB_CPU_ONLY)
        find_path(NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
            HINTS /workspace/deps/NVTX/c/include ${CUDAToolkit_INCLUDE_DIRS}
            REQUIRED
        )
    endif()
    find_library(NVE_TORCH_OPS_LIB nve-torch-ops PATHS "${NVE_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
    find_library(NVE_COMMON_LIB nve-common PATHS "${NVE_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

    add_library(inference_emb_update SHARED src/exportable_embedding/incremental_update.cpp)
    if(NOT DYNAMICEMB_CPU_ONLY)
        target_sources(inference_emb_update PRIVATE src/exportable_embedding/nve_runtime.cu)
        target_compile_definitions(inference_emb_update PRIVATE DYNAMICEMB_WITH_CUDA)
    endif()
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

set(EXPORTABLE_EMBEDDING_INDEXER_SOURCES
    src/exportable_embedding/exportable_embedding_pybind.cpp
    src/exportable_embedding/indexer_directory_pybind.cpp
    src/exportable_embedding/indexer_snapshot_pybind.cpp
)
set(EXPORTABLE_EMBEDDING_PYBIND_SOURCES ${EXPORTABLE_EMBEDDING_INDEXER_SOURCES})
if(DYNAMICEMB_HAS_NVE_UPDATE)
    list(APPEND EXPORTABLE_EMBEDDING_PYBIND_SOURCES
        src/exportable_embedding/update_subscriber_pybind.cu
    )
    if(NOT DYNAMICEMB_CPU_ONLY)
        list(APPEND EXPORTABLE_EMBEDDING_PYBIND_SOURCES src/exportable_embedding/nve_tier_pybind.cu)
    endif()
else()
    list(APPEND EXPORTABLE_EMBEDDING_PYBIND_SOURCES
        src/exportable_embedding/update_subscriber_unavailable_pybind.cpp
    )
endif()

add_library(dynamicemb_exportable_embedding_python MODULE
    ${EXPORTABLE_EMBEDDING_PYBIND_SOURCES}
)
# NVE 26.05 must never dlopen the default extension's modern NVE dependencies.
# Both modules share the same indexer operators; only _C links the update runtime.
set(_binding_targets dynamicemb_exportable_embedding_python)
if(DYNAMICEMB_CPU_ONLY)
    target_compile_definitions(dynamicemb_exportable_embedding_python PRIVATE DYNAMICEMB_CPU_ONLY)
else()
    add_library(dynamicemb_exportable_embedding_legacy_python MODULE
        ${EXPORTABLE_EMBEDDING_INDEXER_SOURCES}
        src/exportable_embedding/update_subscriber_unavailable_pybind.cpp
    )
    list(APPEND _binding_targets dynamicemb_exportable_embedding_legacy_python)
    target_compile_definitions(dynamicemb_exportable_embedding_python PRIVATE DYNAMICEMB_WITH_CUDA)
    target_compile_definitions(dynamicemb_exportable_embedding_legacy_python PRIVATE TORCH_EXTENSION_NAME=_C_legacy)
    set_target_properties(dynamicemb_exportable_embedding_legacy_python PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_C_legacy"
        LIBRARY_OUTPUT_DIRECTORY "${DYNAMICEMB_PYTHON_MODULE_DIR}"
        INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}"
    )
endif()
foreach(binding_target IN LISTS _binding_targets)
    target_include_directories(${binding_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Python3_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    if(TORCH_CXX_FLAGS_LIST)
        target_compile_options(${binding_target} PRIVATE
            ${TORCH_CXX_FLAGS_LIST}
        )
    endif()
    target_link_libraries(${binding_target} PRIVATE
        inference_emb_ops
        Python3::Python
        ${TORCH_LIBRARIES}
        "${TORCH_PYTHON_LIBRARY}"
    )
endforeach()
target_compile_definitions(dynamicemb_exportable_embedding_python PRIVATE
    TORCH_EXTENSION_NAME=_C
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
    LIBRARY_OUTPUT_DIRECTORY "${DYNAMICEMB_PYTHON_MODULE_DIR}"
    INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR};${NVE_LIB_DIR}"
)
install(TARGETS
    ${_binding_targets}
    LIBRARY DESTINATION
        "${Python3_SITEARCH}/dynamicemb/exportable_embedding"
    RUNTIME DESTINATION
        "${Python3_SITEARCH}/dynamicemb/exportable_embedding"
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
