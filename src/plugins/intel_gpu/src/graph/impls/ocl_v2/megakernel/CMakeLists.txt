# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(MEGAKERNEL_IMPLEMENTATION "Qwen06BPOC_prefill_separate_kernels" CACHE STRING
        "Megakernel implementation to link from MEGAKERNEL_POC/megakernels")

set(MEGAKERNEL_ROOT "${CMAKE_SOURCE_DIR}/MEGAKERNEL_POC/megakernels")
set(MEGAKERNEL_IMPLEMENTATION_DIR "${MEGAKERNEL_ROOT}/${MEGAKERNEL_IMPLEMENTATION}")
set(MEGAKERNEL_RUNTIME_TARGET "${MEGAKERNEL_IMPLEMENTATION}")

if(NOT EXISTS "${MEGAKERNEL_IMPLEMENTATION_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
                "Megakernel implementation '${MEGAKERNEL_IMPLEMENTATION}' was not found in ${MEGAKERNEL_IMPLEMENTATION_DIR}")
endif()

# The graph transformation is model-specific and ships with the megakernel, but
# it needs the plugin-internal MegaKernel op, so it is compiled into the plugin
# instead of into the megakernel's shared library. Exported as cache variables
# because the plugin target is defined in a sibling directory.
file(GLOB MEGAKERNEL_TRANSFORMATION_SOURCES
        "${MEGAKERNEL_IMPLEMENTATION_DIR}/transformation/*.cpp")
if(NOT MEGAKERNEL_TRANSFORMATION_SOURCES)
        message(FATAL_ERROR
                "Megakernel implementation '${MEGAKERNEL_IMPLEMENTATION}' must provide "
                "transformation/*.cpp defining mk::InsertMegakernelTransformation")
endif()
set(MEGAKERNEL_TRANSFORMATION_SOURCES "${MEGAKERNEL_TRANSFORMATION_SOURCES}"
        CACHE INTERNAL "Megakernel graph transformation sources")
set(MEGAKERNEL_TRANSFORMATION_INCLUDE_DIR "${MEGAKERNEL_ROOT}"
        CACHE INTERNAL "Megakernel transformation interface include directory")

add_subdirectory("${MEGAKERNEL_IMPLEMENTATION_DIR}"
                                 "${CMAKE_CURRENT_BINARY_DIR}/${MEGAKERNEL_IMPLEMENTATION}"
                 EXCLUDE_FROM_ALL)

if(NOT TARGET ${MEGAKERNEL_RUNTIME_TARGET})
        message(FATAL_ERROR
                "Megakernel implementation '${MEGAKERNEL_IMPLEMENTATION}' must define target ${MEGAKERNEL_RUNTIME_TARGET}")
endif()

install(TARGETS ${MEGAKERNEL_IMPLEMENTATION}
        RUNTIME DESTINATION ${OV_CPACK_PLUGINSDIR}
        LIBRARY DESTINATION ${OV_CPACK_PLUGINSDIR}
        COMPONENT gpu)

set(MEGAKERNEL_RUNTIME_TARGET ${MEGAKERNEL_RUNTIME_TARGET} PARENT_SCOPE)