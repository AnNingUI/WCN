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


add_library(fullstack_present STATIC
    src/present/fullstack_present.c
)
target_include_directories(fullstack_present PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${WCN_ROOT}/external/wgpu/include
)
target_link_libraries(fullstack_present PUBLIC fullstack_result PRIVATE ${WGPU_LIBRARY})
set(FS_CORE_RUNTIME_SOURCES
    src/fullstack_core.c
    src/fullstack_core_paint.c
    src/fullstack_core_path2d.c
    src/fullstack_core_transform.c
    src/fullstack_core_util.c
    src/fullstack_core_atlas.c
    src/fullstack_core_upload.c
    src/fullstack_core_canvas.c
    src/fullstack_core_gpu_common.c
    src/fullstack_core_clip_resources.c
    src/fullstack_core_text_glyph.c
    src/fullstack_core_text_image_font.c
    src/fullstack_core_text_api.c
    src/fullstack_core_style_api.c
    src/fullstack_core_style.c
    src/fullstack_effects.c
    src/fullstack_filters.c
    impl/fullstack_stb_image_backend.c
    impl/fullstack_stb_font_backend.c
    impl/fullstack_stb_cbdt_parser.c
    impl/fullstack_freetype2_font_backend.c
)
add_library(fullstack_core_runtime STATIC ${FS_CORE_RUNTIME_SOURCES})
target_include_directories(fullstack_core_runtime
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/impl
            ${WCN_ROOT}/include
            ${WCN_ROOT}/external/wgpu/include
            ${WCN_ROOT}/external/stb/include
)
target_link_libraries(fullstack_core_runtime PUBLIC fullstack_gpu fullstack_present PRIVATE ${WGPU_LIBRARY})
find_package(Freetype QUIET)
if(FREETYPE_FOUND)
    target_compile_definitions(fullstack_core_runtime PRIVATE FS_HAS_FREETYPE2=1)
    target_include_directories(fullstack_core_runtime PRIVATE ${FREETYPE_INCLUDE_DIRS})
    target_link_libraries(fullstack_core_runtime PRIVATE ${FREETYPE_LIBRARIES})
endif()
if(NOT MSVC)
    target_link_libraries(fullstack_core_runtime PRIVATE m)
endif()

add_library(fullstack_render STATIC
    src/render/fullstack_render_context.c
    src/render/fullstack_render_target.c
    src/render/fullstack_render_frame.c
    src/render/fullstack_frame_graph.c
)
target_include_directories(fullstack_render
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/render
            ${WCN_ROOT}/external/wgpu/include
)
target_link_libraries(fullstack_render
    PUBLIC fullstack_core_runtime fullstack_gpu fullstack_present
    PRIVATE ${WGPU_LIBRARY}
)
add_subdirectory(examples)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
