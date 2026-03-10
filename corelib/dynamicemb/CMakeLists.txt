cmake_minimum_required(VERSION 3.18)
project(DynamicEmbInferenceOps LANGUAGES CXX CUDA)

include(GNUInstallDirs)

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
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 89 90)
    endif()
endif()

message(STATUS "Found Torch ${TORCH_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(INFERENCE_EMB_SOURCES
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/table_operation
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

target_compile_definitions(inference_emb_ops PRIVATE
    TORCH_EXTENSION_NAME=inference_emb_ops
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
