cmake_minimum_required(VERSION 3.18)
project(KVCacheManagerOps LANGUAGES CXX CUDA)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

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
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90 100 120)
    endif()
endif()

message(STATUS "Found Torch ${TORCH_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(KVCACHE_MANAGER_OPS_SOURCES
    src/torch_binding/kvcache_manager_ops.cpp
    src/torch_binding/kvcache_manager_context.cpp
    src/runtime/kvcache_runtime_config.cpp
    src/runtime/export_kvcache_runtime.cpp
    src/gpu_kvcache_manager_impl.cpp
    src/torch_binding/flexkv_cpp_client.cpp
    src/torch_binding/flexkv_aoti_protocol.cpp
    src/gather_scatter.cpp
    src/gather_scatter_kernels.cu
)

add_library(kvcache_manager_ops SHARED ${KVCACHE_MANAGER_OPS_SOURCES})

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
if(TORCH_CXX_FLAGS_LIST)
    target_compile_options(kvcache_manager_ops PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()

target_include_directories(kvcache_manager_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/src/torch_binding
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(kvcache_manager_ops PRIVATE
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

target_compile_definitions(kvcache_manager_ops PRIVATE
    TORCH_EXTENSION_NAME=kvcache_manager_ops
)

target_link_libraries(kvcache_manager_ops PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    zmq
)

set_target_properties(kvcache_manager_ops PROPERTIES
    PREFIX ""
    OUTPUT_NAME "kvcache_manager_ops"
    CUDA_SEPARABLE_COMPILATION OFF
)

install(TARGETS kvcache_manager_ops
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
