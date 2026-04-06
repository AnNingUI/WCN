# Examples for fullstack_compute experiment
cmake_minimum_required(VERSION 3.11)

set(FS_EXAMPLES_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(FS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(WCN_ROOT "${FS_ROOT}/../.." ABSOLUTE)

# ---------------------------------------------------------------------------
# Shared library search / dependency setup
# ---------------------------------------------------------------------------
find_package(glfw3 QUIET CONFIG)
if(NOT glfw3_FOUND)
    find_package(glfw3 QUIET)
endif()
if(NOT glfw3_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW_PKG QUIET glfw3)
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

find_package(Freetype QUIET)

# ---------------------------------------------------------------------------
# Helper macro: link GLFW + WGPU + Freetype to a target
# ---------------------------------------------------------------------------
macro(fs_example_link target)
    if(TARGET GLFW::GLFW)
        target_link_libraries(${target} PRIVATE GLFW::GLFW)
    elseif(GLFW_LIBRARIES)
        target_include_directories(${target} PRIVATE ${GLFW_INCLUDE_DIRS})
        target_link_directories(${target} PRIVATE ${GLFW_LINK_DIRS})
        target_link_libraries(${target} PRIVATE ${GLFW_LIBRARIES})
    else()
        target_link_libraries(${target} PRIVATE glfw)
    endif()
    target_link_libraries(${target} PRIVATE ${WGPU_LIBRARY})
    if(NOT MSVC)
        target_link_libraries(${target} PRIVATE m)
    endif()
    if(FREETYPE_FOUND)
        target_compile_definitions(${target} PRIVATE FS_HAS_FREETYPE2=1)
        target_include_directories(${target} PRIVATE ${FREETYPE_INCLUDE_DIRS})
        target_link_libraries(${target} PRIVATE ${FREETYPE_LIBRARIES})
    endif()
endmacro()

# ---------------------------------------------------------------------------
# Shared source files
# ---------------------------------------------------------------------------
set(FS_CORE_SOURCES
    ${FS_ROOT}/src/fullstack_core.c
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
    ${FS_ROOT}/src/fullstack_filters.c
)

set(FS_BACKEND_SOURCES
    ${FS_ROOT}/impl/fullstack_stb_image_backend.c
    ${FS_ROOT}/impl/fullstack_stb_font_backend.c
    ${FS_ROOT}/impl/fullstack_stb_cbdt_parser.c
    ${FS_ROOT}/impl/fullstack_freetype2_font_backend.c
    ${FS_ROOT}/impl/fullstack_glfw_backend.c
)

set(FS_INCLUDE_DIRS
    PRIVATE
    ${WCN_ROOT}/include
    ${WCN_ROOT}/external/wgpu/include
    ${WCN_ROOT}/external/stb/include
    ${FS_ROOT}/include
    ${FS_ROOT}/src
    ${FS_ROOT}/impl
)

# ---------------------------------------------------------------------------
# fullstack_compute_demo
# ---------------------------------------------------------------------------
add_executable(fullstack_compute_demo
    ${FS_CORE_SOURCES}
    ${FS_BACKEND_SOURCES}
    ${FS_EXAMPLES_ROOT}/fullstack_compute_demo.c
    ${FS_EXAMPLES_ROOT}/fullstack_emoji_embedded.h
)
target_include_directories(fullstack_compute_demo ${FS_INCLUDE_DIRS})
fs_example_link(fullstack_compute_demo)

# ---------------------------------------------------------------------------
# Orbital Catcher
# ---------------------------------------------------------------------------
add_executable(orbital_catcher
    ${FS_CORE_SOURCES}
    ${FS_BACKEND_SOURCES}
    ${FS_EXAMPLES_ROOT}/orbital_catcher.c
)
target_include_directories(orbital_catcher ${FS_INCLUDE_DIRS})
fs_example_link(orbital_catcher)

# ---------------------------------------------------------------------------
# fs_text_layout Demo — Pretext-style DOM-free text layout
# ---------------------------------------------------------------------------
add_executable(fs_text_layout_demo
    ${FS_CORE_SOURCES}
    ${FS_BACKEND_SOURCES}
    ${FS_EXAMPLES_ROOT}/fs_text_layout_demo.c
)
target_include_directories(fs_text_layout_demo ${FS_INCLUDE_DIRS})
fs_example_link(fs_text_layout_demo)

# ---------------------------------------------------------------------------
# Clay Layout Demo
# ---------------------------------------------------------------------------
add_executable(clay_layout_demo
    ${FS_CORE_SOURCES}
    ${FS_BACKEND_SOURCES}
    ${FS_EXAMPLES_ROOT}/clay_layout_demo.c
)
target_include_directories(clay_layout_demo ${FS_INCLUDE_DIRS})
fs_example_link(clay_layout_demo)
