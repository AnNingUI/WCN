cmake_minimum_required(VERSION 3.11)

project(WCN C)

# 设置 C 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 设置包含目录
include_directories(include)
include_directories(external/wgpu/include)
include_directories(src)

# 查找所有源文件
file(GLOB_RECURSE SOURCES "src/*.c")

# 如果没有源文件，则创建一个空的源文件列表
if(NOT SOURCES)
    set(SOURCES "")
endif()

# 创建主库
add_library(wcn ${SOURCES})

# 链接 WebGPU 库
find_library(WGPU_LIBRARY 
    NAMES wgpu_native wgpu webgpu
    HINTS external/wgpu/lib
    PATH_SUFFIXES release debug
)

if(WGPU_LIBRARY)
    target_link_libraries(wcn ${WGPU_LIBRARY})
else()
    message(WARNING "WebGPU library not found")
endif()

# WASM 构建配置
if(EMSCRIPTEN)
    # 创建 WASM 版本的库
    add_library(wcn_wasm ${SOURCES})
    
    # 设置 WASM 编译选项
    target_compile_options(wcn_wasm PRIVATE
        -sUSE_WEBGPU=1
        -sEXPORTED_FUNCTIONS=_malloc,_free
        -sEXPORTED_RUNTIME_METHODS=ccall,cwrap
        -sMODULARIZE=1
        -sEXPORT_NAME=WCN
    )
    
    target_link_options(wcn_wasm PRIVATE
        -sUSE_WEBGPU=1
        -sEXPORTED_FUNCTIONS=_malloc,_free
        -sEXPORTED_RUNTIME_METHODS=ccall,cwrap
        -sMODULARIZE=1
        -sEXPORT_NAME=WCN
        -sALLOW_MEMORY_GROWTH=1
    )
    
    # 导出需要的函数
    target_compile_definitions(wcn_wasm PRIVATE WCN_WASM_EXPORT)
endif()

# 添加 examples 子目录
add_subdirectory(examples)

# 安装规则
install(TARGETS wcn
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# 安装头文件到 include/WCN 目录
install(DIRECTORY include/ 
    DESTINATION include/WCN
    FILES_MATCHING PATTERN "*.h"
)

# 生成 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)