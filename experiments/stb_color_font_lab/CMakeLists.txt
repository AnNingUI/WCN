cmake_minimum_required(VERSION 3.11)
project(stb_color_font_lab C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(LAB_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(WCN_ROOT "${LAB_ROOT}/../.." ABSOLUTE)

add_library(
    stb_cbdt_parser
    ${LAB_ROOT}/src/stb_cbdt_parser.c
)

target_include_directories(
    stb_cbdt_parser
    PUBLIC
    ${LAB_ROOT}/include
)

add_executable(
    stb_color_font_lab
    ${LAB_ROOT}/src/main.c
)

target_include_directories(
    stb_color_font_lab
    PRIVATE
    ${LAB_ROOT}/include
    ${WCN_ROOT}/external/stb/include
)

target_link_libraries(stb_color_font_lab PRIVATE stb_cbdt_parser)

