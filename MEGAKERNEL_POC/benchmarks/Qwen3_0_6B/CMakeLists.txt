cmake_minimum_required(VERSION 3.16)
project(qwen06b_random_decode_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MEGAKERNEL_IMPLEMENTATION "Qwen06BPOC_prefill_separate_kernels" CACHE STRING
    "Megakernel implementation to benchmark from MEGAKERNEL_POC/megakernels")
set(MEGAKERNEL_IMPLEMENTATION_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../megakernels/${MEGAKERNEL_IMPLEMENTATION}")
if(NOT EXISTS "${MEGAKERNEL_IMPLEMENTATION_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Megakernel implementation '${MEGAKERNEL_IMPLEMENTATION}' not found")
endif()

find_package(OpenCL REQUIRED)
add_subdirectory("${MEGAKERNEL_IMPLEMENTATION_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/${MEGAKERNEL_IMPLEMENTATION}")
if(NOT TARGET ${MEGAKERNEL_IMPLEMENTATION})
    message(FATAL_ERROR "Megakernel implementation '${MEGAKERNEL_IMPLEMENTATION}' must define its target")
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdparty/gtest/gtest"
    "${CMAKE_CURRENT_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL)
add_executable(qwen06b_random_decode_benchmark decode_benchmark.cpp)
target_compile_definitions(qwen06b_random_decode_benchmark PRIVATE CL_TARGET_OPENCL_VERSION=300)
target_link_libraries(qwen06b_random_decode_benchmark PRIVATE ${MEGAKERNEL_IMPLEMENTATION} OpenCL::OpenCL gtest)
enable_testing()
add_test(NAME qwen06b_random_decode COMMAND qwen06b_random_decode_benchmark)