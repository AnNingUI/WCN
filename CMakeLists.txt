cmake_minimum_required(VERSION 3.11)

project(WCN C)

# 包含必要的 CMake 模块
include(CheckCCompilerFlag)

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

# SIMD 检测和配置选项
option(WCN_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(WCN_ENABLE_SSE "Enable SSE instructions" ON)
option(WCN_ENABLE_SSE2 "Enable SSE2 instructions" ON)
option(WCN_ENABLE_SSE3 "Enable SSE3 instructions" ON)
option(WCN_ENABLE_SSSE3 "Enable SSSE3 instructions" ON)
option(WCN_ENABLE_SSE41 "Enable SSE4.1 instructions" ON)
option(WCN_ENABLE_SSE42 "Enable SSE4.2 instructions" ON)
option(WCN_ENABLE_AVX "Enable AVX instructions" ON)
option(WCN_ENABLE_AVX2 "Enable AVX2 instructions" ON)
option(WCN_ENABLE_NEON "Enable NEON instructions" ON)

# SIMD 检测宏定义
if(WCN_ENABLE_SIMD)
    # 基本 SIMD 支持
    if(WCN_ENABLE_SSE)
        check_c_compiler_flag(-msse SUPPORTS_SSE)
        if(SUPPORTS_SSE)
            target_compile_options(wcn PRIVATE -msse)
            target_compile_definitions(wcn PRIVATE WCN_ENABLE_SSE)
        endif()
    endif()
    
    if(WCN_ENABLE_SSE2)
        check_c_compiler_flag(-msse2 SUPPORTS_SSE2)
        if(SUPPORTS_SSE2)
            target_compile_options(wcn PRIVATE -msse2)
            target_compile_definitions(wcn PRIVATE WCN_ENABLE_SSE2)
        endif()
    endif()
    
    if(WCN_ENABLE_SSE3)
        check_c_compiler_flag(-msse3 SUPPORTS_SSE3)
        if(SUPPORTS_SSE3)
            target_compile_options(wcn PRIVATE -msse3)
            target_compile_definitions(wcn PRIVATE WCN_ENABLE_SSE3)
        endif()
    endif()
    
    if(WCN_ENABLE_SSSE3)
        check_c_compiler_flag(-mssse3 SUPPORTS_SSSE3)
        if(SUPPORTS_SSSE3)
            target_compile_options(wcn PRIVATE -mssse3)
            target_compile_definitions(wcn PRIVATE WCN_ENABLE_SSSE3)
        endif()
    endif()
    
    if(WCN_ENABLE_SSE41)
        check_c_compiler_flag(-msse4.1 SUPPORTS_SSE41)
        if(SUPPORTS_SSE41)
            target_compile_options(wcn PRIVATE -msse4.1)
            target_compile_definitions(wcn PRIVATE WCN_ENABLE_SSE41)
        endif()
    endif()
    
    if(WCN_ENABLE_SSE42)
        check_c_compiler_flag(-msse4.2 SUPPORTS_SSE42)
        if(SUPPORTS_SSE42)
            target_compile_options(wcn PRIVATE -msse4.2)
            target_compile_definitions(wcn PRIVATE WCN_ENABLE_SSE42)
        endif()
    endif()
    
    if(WCN_ENABLE_AVX)
        check_c_compiler_flag(-mavx SUPPORTS_AVX)
        if(SUPPORTS_AVX)
            target_compile_options(wcn PRIVATE -mavx)
            target_compile_definitions(wcn PRIVATE WCN_ENABLE_AVX)
        endif()
    endif()
    
    if(WCN_ENABLE_AVX2)
        check_c_compiler_flag(-mavx2 SUPPORTS_AVX2)
        if(SUPPORTS_AVX2)
            target_compile_options(wcn PRIVATE -mavx2)
            target_compile_definitions(wcn PRIVATE WCN_ENABLE_AVX2)
        endif()
    endif()
    
    if(WCN_ENABLE_NEON)
        check_c_compiler_flag(-mfpu=neon SUPPORTS_NEON)
        if(SUPPORTS_NEON)
            target_compile_options(wcn PRIVATE -mfpu=neon)
            target_compile_definitions(wcn PRIVATE WCN_ENABLE_NEON)
        endif()
    endif()
    
    # 添加通用 SIMD 定义
    target_compile_definitions(wcn PRIVATE WCN_ENABLE_SIMD)
else()
    target_compile_definitions(wcn PRIVATE WMATH_DISABLE_SIMD)
endif()

# 平台特定的 SIMD 检测
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
    # x86/x64 平台
    target_compile_definitions(wcn PRIVATE WCN_ARCH_X86)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_definitions(wcn PRIVATE WCN_ARCH_X64)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM")
    # ARM 平台
    target_compile_definitions(wcn PRIVATE WCN_ARCH_ARM)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        target_compile_definitions(wcn PRIVATE WCN_ARCH_ARM64)
    endif()
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