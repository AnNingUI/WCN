# 查找 GLFW 包
find_package(glfw3 REQUIRED)

# 创建可执行文件
add_executable(glfw_render_test render_test.c)
add_executable(transform_test transform_test.c)

# 链接库
target_link_libraries(glfw_render_test 
    wcn
    glfw
)

target_link_libraries(transform_test 
    wcn
    glfw
)

# 包含目录
target_include_directories(glfw_render_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_include_directories(transform_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)