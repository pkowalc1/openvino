cmake_minimum_required(VERSION 3.16)

set(TARGET_NAME Qwen06BPOC_prefill_separate_kernels)

project(${TARGET_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TASK_SYSTEM_OPENCL_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../research/preloading_gemv/src/ocl"
    CACHE PATH "Path to the task-system OpenCL sources")

if(NOT TARGET OpenCL::OpenCL)
    find_package(OpenCL REQUIRED)
endif()

if(TARGET onednn_gpu_tgt)
    set(MEGAKERNEL_DNNL_TARGET onednn_gpu_tgt)
else()
    set(MEGAKERNEL_DNNL_TARGET dnnl)
endif()

if(NOT TARGET onednn_gpu_tgt AND NOT TARGET dnnl)
    set(DNNL_CPU_RUNTIME NONE CACHE STRING "Disable oneDNN CPU runtime" FORCE)
    set(DNNL_GPU_RUNTIME OCL CACHE STRING "Use OpenCL oneDNN GPU runtime" FORCE)
    set(DNNL_GPU_VENDOR INTEL CACHE STRING "Build oneDNN for Intel GPUs" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "Disable oneDNN examples")
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "Disable oneDNN tests")
    set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "Disable oneDNN graph component")
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../../src/plugins/intel_gpu/thirdparty/onednn_gpu"
        "${CMAKE_CURRENT_BINARY_DIR}/onednn" EXCLUDE_FROM_ALL)
endif()

add_library(${TARGET_NAME} SHARED
    megakernelImpl.cpp
    impl/qwen06BPOCRuntime.cpp
    "${TASK_SYSTEM_OPENCL_ROOT}/taskSystem/host/taskManagerHost.cpp")
add_library(megakernel::${TARGET_NAME} ALIAS ${TARGET_NAME})

include(GenerateExportHeader)
generate_export_header(${TARGET_NAME}
    EXPORT_FILE_NAME exportApi.h
    EXPORT_MACRO_NAME EXPORT_API)

target_compile_definitions(${TARGET_NAME} PRIVATE
    CL_TARGET_OPENCL_VERSION=300
    TASK_SYSTEM_OPENCL_ROOT="${TASK_SYSTEM_OPENCL_ROOT}")

target_include_directories(${TARGET_NAME}
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}"
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/impl"
        "${TASK_SYSTEM_OPENCL_ROOT}")

target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL ${MEGAKERNEL_DNNL_TARGET})

set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)