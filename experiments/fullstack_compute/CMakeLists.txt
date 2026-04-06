cmake_minimum_required(VERSION 3.11)
project(wcn_fullstack_compute_experiment C)

set(CMAKE_C_STANDARD 11)
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

add_executable(
    fullstack_compute_demo
    # ${FS_ROOT}/src/fullstack_core.replay.c
    ${FS_ROOT}/src/fullstack_core.c
    ${FS_ROOT}/src/fullstack_effects.c
    ${FS_ROOT}/src/fullstack_filters.c
    ${FS_ROOT}/src/fullstack_core_paint.c
    ${FS_ROOT}/src/fullstack_core_path2d.c
    ${FS_ROOT}/src/fullstack_core_transform.c
    ${FS_ROOT}/src/fullstack_core_util.c
    ${FS_ROOT}/src/fullstack_core_atlas.c
    ${FS_ROOT}/src/fullstack_core_upload.c
    ${FS_ROOT}/src/fullstack_core_canvas.c
    ${FS_ROOT}/src/fullstack_core_gpu_common.c
    ${FS_ROOT}/src/fullstack_core_clip_resources.c
    ${FS_ROOT}/src/fullstack_core_text_glyph.c
    ${FS_ROOT}/src/fullstack_core_text_image_font.c
    ${FS_ROOT}/src/fullstack_core_text_api.c
    ${FS_ROOT}/src/fullstack_core_style_api.c
    ${FS_ROOT}/src/fullstack_core_style.c
    ${FS_ROOT}/impl/fullstack_stb_image_backend.c
    ${FS_ROOT}/impl/fullstack_stb_font_backend.c
    ${FS_ROOT}/impl/fullstack_stb_cbdt_parser.c
    ${FS_ROOT}/impl/fullstack_freetype2_font_backend.c
    ${FS_ROOT}/impl/fullstack_glfw_backend.c
    ${FS_ROOT}/examples/fullstack_compute_demo.c
)

target_include_directories(
    fullstack_compute_demo
    PRIVATE
    ${WCN_ROOT}/include
    ${WCN_ROOT}/external/wgpu/include
    ${WCN_ROOT}/external/stb/include
    ${FS_ROOT}/include
    ${FS_ROOT}/src
    ${FS_ROOT}/impl
)

if(TARGET GLFW::GLFW)
    target_link_libraries(fullstack_compute_demo PRIVATE GLFW::GLFW)
elseif(GLFW_LIBRARIES)
    target_include_directories(fullstack_compute_demo PRIVATE ${GLFW_INCLUDE_DIRS})
    target_link_directories(fullstack_compute_demo PRIVATE ${GLFW_LINK_DIRS})
    target_link_libraries(fullstack_compute_demo PRIVATE ${GLFW_LIBRARIES})
else()
    target_link_libraries(fullstack_compute_demo PRIVATE glfw)
endif()

target_link_libraries(fullstack_compute_demo PRIVATE ${WGPU_LIBRARY})
if(NOT MSVC)
    target_link_libraries(fullstack_compute_demo PRIVATE m)
endif()

find_package(Freetype QUIET)
if(FREETYPE_FOUND)
    target_compile_definitions(fullstack_compute_demo PRIVATE FS_HAS_FREETYPE2=1)
    target_include_directories(fullstack_compute_demo PRIVATE ${FREETYPE_INCLUDE_DIRS})
    target_link_libraries(fullstack_compute_demo PRIVATE ${FREETYPE_LIBRARIES})
endif()

# ---------------------------------------------------------------------------
# fs_text_layout_demo — Pretext-style DOM-free text layout
# ---------------------------------------------------------------------------
add_executable(
    fs_text_layout_demo
    ${FS_ROOT}/src/fullstack_core.c
    ${FS_ROOT}/src/fullstack_effects.c
    ${FS_ROOT}/src/fullstack_filters.c
    ${FS_ROOT}/src/fullstack_core_paint.c
    ${FS_ROOT}/src/fullstack_core_path2d.c
    ${FS_ROOT}/src/fullstack_core_transform.c
    ${FS_ROOT}/src/fullstack_core_util.c
    ${FS_ROOT}/src/fullstack_core_atlas.c
    ${FS_ROOT}/src/fullstack_core_upload.c
    ${FS_ROOT}/src/fullstack_core_canvas.c
    ${FS_ROOT}/src/fullstack_core_gpu_common.c
    ${FS_ROOT}/src/fullstack_core_clip_resources.c
    ${FS_ROOT}/src/fullstack_core_text_glyph.c
    ${FS_ROOT}/src/fullstack_core_text_image_font.c
    ${FS_ROOT}/src/fullstack_core_text_api.c
    ${FS_ROOT}/src/fullstack_core_style_api.c
    ${FS_ROOT}/src/fullstack_core_style.c
    ${FS_ROOT}/impl/fullstack_stb_image_backend.c
    ${FS_ROOT}/impl/fullstack_stb_font_backend.c
    ${FS_ROOT}/impl/fullstack_stb_cbdt_parser.c
    ${FS_ROOT}/impl/fullstack_freetype2_font_backend.c
    ${FS_ROOT}/impl/fullstack_glfw_backend.c
    ${FS_ROOT}/examples/fs_text_layout_demo.c
)
target_include_directories(
    fs_text_layout_demo
    PRIVATE
    ${WCN_ROOT}/include
    ${WCN_ROOT}/external/wgpu/include
    ${WCN_ROOT}/external/stb/include
    ${FS_ROOT}/include
    ${FS_ROOT}/src
    ${FS_ROOT}/impl
)
if(TARGET GLFW::GLFW)
    target_link_libraries(fs_text_layout_demo PRIVATE GLFW::GLFW)
elseif(GLFW_LIBRARIES)
    target_include_directories(fs_text_layout_demo PRIVATE ${GLFW_INCLUDE_DIRS})
    target_link_directories(fs_text_layout_demo PRIVATE ${GLFW_LINK_DIRS})
    target_link_libraries(fs_text_layout_demo PRIVATE ${GLFW_LIBRARIES})
else()
    target_link_libraries(fs_text_layout_demo PRIVATE glfw)
endif()
target_link_libraries(fs_text_layout_demo PRIVATE ${WGPU_LIBRARY})
if(NOT MSVC)
    target_link_libraries(fs_text_layout_demo PRIVATE m)
endif()
if(FREETYPE_FOUND)
    target_compile_definitions(fs_text_layout_demo PRIVATE FS_HAS_FREETYPE2=1)
    target_include_directories(fs_text_layout_demo PRIVATE ${FREETYPE_INCLUDE_DIRS})
    target_link_libraries(fs_text_layout_demo PRIVATE ${FREETYPE_LIBRARIES})
endif()

# ---------------------------------------------------------------------------
# clay_layout_demo — Clay layout engine × WCN showcase
# ---------------------------------------------------------------------------
add_executable(
    clay_layout_demo
    ${FS_ROOT}/src/fullstack_core.c
    ${FS_ROOT}/src/fullstack_effects.c
    ${FS_ROOT}/src/fullstack_filters.c
    ${FS_ROOT}/src/fullstack_core_paint.c
    ${FS_ROOT}/src/fullstack_core_path2d.c
    ${FS_ROOT}/src/fullstack_core_transform.c
    ${FS_ROOT}/src/fullstack_core_util.c
    ${FS_ROOT}/src/fullstack_core_atlas.c
    ${FS_ROOT}/src/fullstack_core_upload.c
    ${FS_ROOT}/src/fullstack_core_canvas.c
    ${FS_ROOT}/src/fullstack_core_gpu_common.c
    ${FS_ROOT}/src/fullstack_core_clip_resources.c
    ${FS_ROOT}/src/fullstack_core_text_glyph.c
    ${FS_ROOT}/src/fullstack_core_text_image_font.c
    ${FS_ROOT}/src/fullstack_core_text_api.c
    ${FS_ROOT}/src/fullstack_core_style_api.c
    ${FS_ROOT}/src/fullstack_core_style.c
    ${FS_ROOT}/impl/fullstack_stb_image_backend.c
    ${FS_ROOT}/impl/fullstack_stb_font_backend.c
    ${FS_ROOT}/impl/fullstack_stb_cbdt_parser.c
    ${FS_ROOT}/impl/fullstack_freetype2_font_backend.c
    ${FS_ROOT}/impl/fullstack_glfw_backend.c
    ${FS_ROOT}/examples/clay_layout_demo.c
)
target_include_directories(
    clay_layout_demo
    PRIVATE
    ${WCN_ROOT}/include
    ${WCN_ROOT}/external/wgpu/include
    ${WCN_ROOT}/external/stb/include
    ${FS_ROOT}/include
    ${FS_ROOT}/src
    ${FS_ROOT}/impl
)
if(TARGET GLFW::GLFW)
    target_link_libraries(clay_layout_demo PRIVATE GLFW::GLFW)
elseif(GLFW_LIBRARIES)
    target_include_directories(clay_layout_demo PRIVATE ${GLFW_INCLUDE_DIRS})
    target_link_directories(clay_layout_demo PRIVATE ${GLFW_LINK_DIRS})
    target_link_libraries(clay_layout_demo PRIVATE ${GLFW_LIBRARIES})
else()
    target_link_libraries(clay_layout_demo PRIVATE glfw)
endif()
target_link_libraries(clay_layout_demo PRIVATE ${WGPU_LIBRARY})
if(NOT MSVC)
    target_link_libraries(clay_layout_demo PRIVATE m)
endif()
if(FREETYPE_FOUND)
    target_compile_definitions(clay_layout_demo PRIVATE FS_HAS_FREETYPE2=1)
    target_include_directories(clay_layout_demo PRIVATE ${FREETYPE_INCLUDE_DIRS})
    target_link_libraries(clay_layout_demo PRIVATE ${FREETYPE_LIBRARIES})
endif()
