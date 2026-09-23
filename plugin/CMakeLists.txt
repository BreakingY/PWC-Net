cmake_minimum_required(VERSION 3.18)

project(pwc_net_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TRT_PATH "" CACHE PATH "TensorRT installation path")

if(NOT TRT_PATH)
    message(FATAL_ERROR
        "TRT_PATH is not set. "
        "Example: -DTRT_PATH=/data/sunkx/TensorRT-10.4.0.26")
endif()

# ============================================================
# TensorRT
# ============================================================

include_directories(
    ${TRT_PATH}/include
    /usr/local/cuda/include
)

link_directories(
    ${TRT_PATH}/lib
    /usr/local/cuda/lib64
)


# ============================================================
# Plugin
# ============================================================

add_library(pwc_net_plugin SHARED
    pwc_net_plugin.cpp
    pwc_net_plugin.cu
)

target_compile_features(
    pwc_net_plugin
    PRIVATE cxx_std_17
)

set_target_properties(
    pwc_net_plugin
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)


target_link_libraries(
    pwc_net_plugin
    PRIVATE
        nvinfer
        cudart
)


# ============================================================
# Output
# ============================================================

set_target_properties(
    pwc_net_plugin
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)