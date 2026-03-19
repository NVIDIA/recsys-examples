cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(inference_embedding_aoti_demo LANGUAGES CXX)

find_package(Torch REQUIRED)

set(INFERENCE_EMB_OPS_SO
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../corelib/dynamicemb/torch_binding_build/inference_emb_ops.so")

add_executable(inference_embedding_aoti_demo inference.cpp)
add_executable(inference_embedding_aoti_e2e_demo inference_e2e.cpp)

target_link_libraries(inference_embedding_aoti_demo PRIVATE "${TORCH_LIBRARIES}" ${CMAKE_DL_LIBS})
target_link_libraries(inference_embedding_aoti_e2e_demo PRIVATE "${TORCH_LIBRARIES}" ${CMAKE_DL_LIBS})
target_compile_features(inference_embedding_aoti_demo PRIVATE cxx_std_17)
target_compile_features(inference_embedding_aoti_e2e_demo PRIVATE cxx_std_17)
set_property(TARGET inference_embedding_aoti_demo PROPERTY CXX_STANDARD 17)
set_property(TARGET inference_embedding_aoti_demo PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET inference_embedding_aoti_e2e_demo PROPERTY CXX_STANDARD 17)
set_property(TARGET inference_embedding_aoti_e2e_demo PROPERTY CXX_STANDARD_REQUIRED ON)

if(TORCH_CXX_FLAGS)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(inference_embedding_aoti_demo PRIVATE ${TORCH_CXX_FLAGS_LIST})
  target_compile_options(inference_embedding_aoti_e2e_demo PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()

# If model.pt2 already exists in the source directory, copy it next to the binary
# so the demo can be run as: ./inference_embedding_aoti_demo
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model.pt2")
  add_custom_command(
    TARGET inference_embedding_aoti_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/model.pt2"
            "$<TARGET_FILE_DIR:inference_embedding_aoti_demo>/model.pt2"
  )
endif()

if(EXISTS "${INFERENCE_EMB_OPS_SO}")
  add_custom_command(
    TARGET inference_embedding_aoti_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${INFERENCE_EMB_OPS_SO}"
            "$<TARGET_FILE_DIR:inference_embedding_aoti_demo>/inference_emb_ops.so"
  )
  add_custom_command(
    TARGET inference_embedding_aoti_e2e_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${INFERENCE_EMB_OPS_SO}"
            "$<TARGET_FILE_DIR:inference_embedding_aoti_e2e_demo>/inference_emb_ops.so"
  )
else()
  message(WARNING "Custom op library not found at ${INFERENCE_EMB_OPS_SO}; runtime will require passing the .so path manually.")
endif()
