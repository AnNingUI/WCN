cmake_minimum_required(VERSION 3.11)
project(wcn_fullstack_compute_experiment C CXX)

include(CTest)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(FS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(WCN_ROOT "${FS_ROOT}/../.." ABSOLUTE)
set(FS_WGPU_LIB_DIR "${WCN_ROOT}/external/wgpu/lib")

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

find_library(
    FS_WGPU_LIBRARY
    NAMES wgpu_native wgpu webgpu
    PATHS ${FS_WGPU_LIB_DIR}
    PATH_SUFFIXES release debug
    NO_DEFAULT_PATH
)
if(NOT FS_WGPU_LIBRARY)
    message(FATAL_ERROR "Pinned WebGPU library not found under ${FS_WGPU_LIB_DIR}")
endif()
set(WGPU_LIBRARY "${FS_WGPU_LIBRARY}")
unset(WGPU_LIBRARY CACHE)

if(WIN32)
    set(FS_WGPU_RUNTIME_LIBRARY "${FS_WGPU_LIB_DIR}/wgpu_native.dll")
    if(NOT EXISTS "${FS_WGPU_RUNTIME_LIBRARY}")
        message(FATAL_ERROR "Pinned WebGPU runtime not found: ${FS_WGPU_RUNTIME_LIBRARY}")
    endif()
    add_custom_target(fs_wgpu_runtime
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FS_WGPU_RUNTIME_LIBRARY}"
            "${CMAKE_BINARY_DIR}/wgpu_native.dll"
        COMMENT "Deploying pinned wgpu-native runtime"
        VERBATIM
    )
endif()

add_library(fullstack_result STATIC
    src/fullstack_abi.c
    src/fullstack_result.c
)
target_include_directories(fullstack_result PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(fullstack_gpu STATIC
    src/fullstack_gpu.c
)
target_include_directories(fullstack_gpu
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${WCN_ROOT}/external/wgpu/include
)
target_link_libraries(fullstack_gpu PUBLIC fullstack_result PRIVATE ${WGPU_LIBRARY})

find_package(Threads REQUIRED)
add_library(fullstack_app_foundation STATIC
    src/app/fullstack_app_event.c
    src/app/fullstack_app_runtime.c
)
target_include_directories(fullstack_app_foundation
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/app
)
target_link_libraries(fullstack_app_foundation PUBLIC fullstack_result Threads::Threads)

add_library(fullstack_app STATIC
    src/app/fullstack_app_backend_registry.c
    src/app/fullstack_app.c
)
target_include_directories(fullstack_app
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/app
)
target_link_libraries(fullstack_app PUBLIC fullstack_app_foundation fullstack_gpu Threads::Threads)

add_library(fullstack_backend_mock STATIC
    impl/backends/mock/fullstack_mock_app_backend.c
)
target_include_directories(fullstack_backend_mock
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/app
)
target_link_libraries(fullstack_backend_mock PUBLIC fullstack_app)
add_subdirectory(examples)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
