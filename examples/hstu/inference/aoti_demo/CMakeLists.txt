cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(inference_embedding_aoti_e2e_demos LANGUAGES CXX)

find_package(Torch REQUIRED)

add_executable(inference_embedding_aoti_e2e_demo inference_e2e.cpp)

target_link_libraries(inference_embedding_aoti_e2e_demo PRIVATE "${TORCH_LIBRARIES}" ${CMAKE_DL_LIBS})
target_compile_features(inference_embedding_aoti_e2e_demo PRIVATE cxx_std_17)
set_property(TARGET inference_embedding_aoti_e2e_demo PROPERTY CXX_STANDARD 17)
set_property(TARGET inference_embedding_aoti_e2e_demo PROPERTY CXX_STANDARD_REQUIRED ON)

if(TORCH_CXX_FLAGS)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(inference_embedding_aoti_e2e_demo PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()

message(STATUS "Shared libraries are NOT auto-copied by CMake.")
message(STATUS "Manually copy inference_emb_ops.so and libnve_torch.so into ../lib (aoti_demo/lib) before running demos.")

# --- Pooled (sum/mean) demo ---
add_executable(inference_embedding_aoti_e2e_pooled_demo inference_e2e_pooled.cpp)
target_link_libraries(inference_embedding_aoti_e2e_pooled_demo PRIVATE "${TORCH_LIBRARIES}" ${CMAKE_DL_LIBS})
target_compile_features(inference_embedding_aoti_e2e_pooled_demo PRIVATE cxx_std_17)
set_property(TARGET inference_embedding_aoti_e2e_pooled_demo PROPERTY CXX_STANDARD 17)
set_property(TARGET inference_embedding_aoti_e2e_pooled_demo PROPERTY CXX_STANDARD_REQUIRED ON)

if(TORCH_CXX_FLAGS)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(inference_embedding_aoti_e2e_pooled_demo PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()
