cmake_minimum_required(VERSION 3.11)
project(wcn_fullstack_compute_experiment C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(FS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(WCN_ROOT "${FS_ROOT}/../.." ABSOLUTE)

find_package(glfw3 QUIET CONFIG)
if(NOT glfw3_FOUND)
    find_package(glfw3 QUIET)
endif()
if(NOT glfw3_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW_PKG QUIET glfw3)
endif()

if(NOT glfw3_FOUND AND NOT GLFW_PKG_FOUND)
    message(FATAL_ERROR "GLFW not found for fullstack compute experiment")
endif()

find_library(WGPU_LIBRARY NAMES wgpu_native wgpu webgpu)
if(NOT WGPU_LIBRARY)
    find_library(
        WGPU_LIBRARY
        NAMES wgpu_native wgpu webgpu
        PATHS ${WCN_ROOT}/external/wgpu/lib
        PATH_SUFFIXES release debug
        NO_DEFAULT_PATH
    )
endif()
if(NOT WGPU_LIBRARY)
    message(FATAL_ERROR "WebGPU library not found for fullstack compute experiment")
endif()

add_subdirectory(examples)
