cmake_minimum_required(VERSION 3.11)
project(wcn_examples_glfw C)

# 避免 CMP0111 报很多开发警告（IMPORTED target 没有 location）
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 NEW)
endif()

# 尝试优先使用 CMake 的 glfw3 配置包（CONFIG 或 Find 模式）
set(GLFW_FOUND_BY "none")
find_package(glfw3 QUIET CONFIG)   # prefer config package
if(NOT glfw3_FOUND)
    find_package(glfw3 QUIET)      # fallback to FindGLFW module if present
endif()

# 如果上面没有找到，再使用 pkg-config 回退查找 (常见于 linux 的 libglfw3 + pkg-config)
if(NOT glfw3_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW_PKG QUIET glfw3)  # pkg-config name is "glfw3" on debian/ubuntu
    if(GLFW_PKG_FOUND)
        set(GLFW_FOUND_BY "pkg-config")
        set(GLFW_INCLUDE_DIRS ${GLFW_PKG_INCLUDE_DIRS})
        set(GLFW_LIBRARIES ${GLFW_PKG_LIBRARIES})
        set(GLFW_LINK_DIRS ${GLFW_PKG_LIBRARY_DIRS})
    endif()
else()
    set(GLFW_FOUND_BY "find_package")
endif()

if(NOT glfw3_FOUND AND NOT GLFW_PKG_FOUND)
    message(FATAL_ERROR "GLFW (glfw3) not found. Install libglfw3 + dev package or provide glfw3 CMake config.")
else()
    message(STATUS "Found GLFW via: ${GLFW_FOUND_BY}")
endif()

# Helper: function to link a target to GLFW in a cross-platform way
function(link_with_glfw tgt)
    if(TARGET GLFW::GLFW)
        target_link_libraries(${tgt} PRIVATE GLFW::GLFW)
    elseif (GLFW_LIBRARIES)
        # pkg-config 返回的可能是 "-lglfw -lX11 -lpthread ..." 之类
        target_include_directories(${tgt} PRIVATE ${GLFW_INCLUDE_DIRS})
        target_link_directories(${tgt} PRIVATE ${GLFW_LINK_DIRS})
        target_link_libraries(${tgt} PRIVATE ${GLFW_LIBRARIES})
    else()
        # 兜底：直接尝试链接 libglfw（在某些系统上库名是 glfw）
        target_link_libraries(${tgt} PRIVATE glfw)
    endif()
endfunction()

# ============================================================================ #
# 各种示例/测试二进制
# ============================================================================ #

macro(add_wcn_glfw_exe name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
    target_link_libraries(${name} PRIVATE wcn)   # 先链接 wcn
    link_with_glfw(${name})                       # 再链接 glfw（跨平台）
endmacro()

add_wcn_glfw_exe(simple_glfw_test     simple_glfw_test.c)
add_wcn_glfw_exe(minimal_test         minimal_test.c)
add_wcn_glfw_exe(transform_test       transform_test.c)
add_wcn_glfw_exe(line_cap_join_test   line_cap_join_test.c)
add_wcn_glfw_exe(sdf_text_test        sdf_text_test.c)
add_wcn_glfw_exe(simple_text_test     simple_text_test.c)
