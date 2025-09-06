# 查找 GLFW 包
find_package(glfw3 REQUIRED)

# 创建可执行文件
add_executable(transform_test transform_test.c)

target_link_libraries(transform_test 
    wcn
    glfw
)


# 包含目录
target_include_directories(transform_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)