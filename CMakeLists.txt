cmake_minimum_required(VERSION 3.11)

project(WCN C CXX)

# 包含必要的 CMake 模块
include(CheckCCompilerFlag)

# 设置 C 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 设置包含目录
include_directories(include)
include_directories(external/wgpu/include)
include_directories(external/stb/include)
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
# 先查找系统库
find_library(WGPU_LIBRARY NAMES wgpu_native wgpu webgpu)

# 如果没找到，再查找项目内的
if(NOT WGPU_LIBRARY)
    find_library(WGPU_LIBRARY NAMES wgpu_native wgpu webgpu
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/wgpu/lib
        PATH_SUFFIXES release debug
        NO_DEFAULT_PATH
    )
endif()

if(WGPU_LIBRARY)
    message(STATUS "Found WebGPU library: ${WGPU_LIBRARY}")
    target_link_libraries(wcn ${WGPU_LIBRARY} m)
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
option(WCN_ENABLE_FMA "Enable FMA instructions" ON)
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

    if(WCN_ENABLE_FMA)
        check_c_compiler_flag(-mfma SUPPORTS_FMA)
        if(SUPPORTS_FMA)
            target_compile_options(wcn PRIVATE -mfma)
            target_compile_definitions(wcn PRIVATE WCN_ENABLE_FMA)
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
option(WCN_BUILD_WASM "Build WebAssembly version" OFF)

if(EMSCRIPTEN OR WCN_BUILD_WASM)
    message(STATUS "Building WCN for WebAssembly")
    message(STATUS "WASM build will use Emscripten's Dawn WebGPU implementation")

    # 为 WASM 构建收集源文件
    file(GLOB WASM_SOURCES
        "src/math/wcn_math_base.c"
        "src/math/wcn_math_globals.c"
        "src/math/wcn_math_internal.h"
        "src/math/wcn_math_mat3.c"
        "src/math/wcn_math_mat4.c"
        "src/math/wcn_math_other.c"
        "src/math/wcn_math_quat.c"
        "src/math/wcn_math_vec2.c"
        "src/math/wcn_math_vec3.c"
        "src/math/wcn_math_vec4.c"
        "src/math/wcn_math_wasm.c"

        "src/wcn_context.c"
        "src/wcn_instance.c"
        "src/wcn_path.c"
        "src/wcn_rect.c"
        "src/wcn_renderer.c"
        "src/wcn_sdf_atlas.c"
        "src/wcn_style.c"
        "src/wcn_image.c"
        "src/wcn_text.c"
        "src/wcn_transform.c"
        "src/wcn_wasm_exports.c"
        "src/wcn_wasm_font_decoder.c"
        "src/wcn_wasm_image_decoder.c"
        "src/wcn_emcc_js.c"
    )

    message(STATUS "WASM sources: ${WASM_SOURCES}")

    # 创建 WASM 版本的库
    add_executable(wcn_wasm ${WASM_SOURCES})

    # 设置包含目录
    target_include_directories(wcn_wasm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/wgpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/stb/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # 手动定义导出的函数列表（与WCN_WASM.h中WCN_WASM_EXPORT标记的函数对应）
    set(EXPORTED_FUNCTIONS
        "_wcn_init_js"
        "_wcn_create_context"
        "_wcn_destroy_context"
        "_wcn_begin_frame"
        "_wcn_end_frame"
        "_wcn_begin_render_pass"
        "_wcn_end_render_pass"
        "_wcn_submit_commands"
        "_wcn_save"
        "_wcn_restore"
        "_wcn_register_font_decoder"
        "_wcn_register_image_decoder"
        "_wcn_clear_rect"
        "_wcn_fill_rect"
        "_wcn_stroke_rect"
        "_wcn_begin_path"
        "_wcn_close_path"
        "_wcn_move_to"
        "_wcn_line_to"
        "_wcn_arc"
        "_wcn_rect"
        "_wcn_fill"
        "_wcn_stroke"
        "_wcn_set_fill_style"
        "_wcn_set_stroke_style"
        "_wcn_set_line_width"
        "_wcn_set_line_cap"
        "_wcn_set_line_join"
        "_wcn_set_miter_limit"
        "_wcn_set_global_alpha"
        "_wcn_set_global_composite_operation"
        "_wcn_translate"
        "_wcn_rotate"
        "_wcn_scale"
        "_wcn_transform"
        "_wcn_set_transform"
        "_wcn_reset_transform"
        "_wcn_fill_text"
        "_wcn_stroke_text"
        "_wcn_measure_text"
        "_wcn_set_font"
        "_wcn_set_font_face"
        "_wcn_set_text_align"
        "_wcn_set_text_baseline"
        "_wcn_wasm_load_font"
        "_wcn_add_font_fallback"
        "_wcn_clear_font_fallbacks"
        "_wcn_draw_image"
        "_wcn_draw_image_scaled"
        "_wcn_draw_image_source"
        "_wcn_get_image_data"
        "_wcn_put_image_data"
        "_wcn_decode_image"
        "_wcn_destroy_image_data"
        "_wcn_get_surface_format"
        "_wcn_set_surface_format"
        "_wcn_wasm_get_font_decoder"
        "_wcn_wasm_create_default_font_face"
        "_wcn_wasm_get_image_decoder"

        # Math Functions
        "_wcn_math_set_epsilon"
        "_wcn_math_get_epsilon"
        "_wcn_math_Vec2_create_wasm"
        "_wcn_math_Vec3_create_wasm"
        "_wcn_math_Vec4_create_wasm"
        "_wcn_math_Quat_create_wasm"
        "_wcn_math_Mat3_create_wasm"
        "_wcn_math_Mat4_create_wasm"
        "_wcn_math_Vec3_WithAngleAxis_get_vec3_wasm"
        "_wcn_math_Vec3_WithAngleAxis_get_angle"
        "_wcn_math_Vec2_set_wasm"
        "_wcn_math_Vec2_set_x_wasm"
        "_wcn_math_Vec2_set_y_wasm"
        "_wcn_math_Vec2_copy_wasm"
        "_wcn_math_Vec2_zero_wasm"
        "_wcn_math_Vec2_identity_wasm"
        "_wcn_math_Vec2_ceil_wasm"
        "_wcn_math_Vec2_floor_wasm"
        "_wcn_math_Vec2_round_wasm"
        "_wcn_math_Vec2_clamp_wasm"
        "_wcn_math_Vec2_dot_wasm"
        "_wcn_math_Vec2_add_wasm"
        "_wcn_math_Vec2_add_scaled_wasm"
        "_wcn_math_Vec2_sub_wasm"
        "_wcn_math_Vec2_angle_wasm"
        "_wcn_math_Vec2_equals_approximately_wasm"
        "_wcn_math_Vec2_equals_wasm"
        "_wcn_math_Vec2_lerp_wasm"
        "_wcn_math_Vec2_lerp_v_wasm"
        "_wcn_math_Vec2_fmax_wasm"
        "_wcn_math_Vec2_fmin_wasm"
        "_wcn_math_Vec2_multiply_scalar_wasm"
        "_wcn_math_Vec2_multiply_wasm"
        "_wcn_math_Vec2_div_scalar_wasm"
        "_wcn_math_Vec2_div_wasm"
        "_wcn_math_Vec2_inverse_wasm"
        "_wcn_math_Vec2_cross_wasm"
        "_wcn_math_Vec2_length_wasm"
        "_wcn_math_Vec2_length_squared_wasm"
        "_wcn_math_Vec2_distance_wasm"
        "_wcn_math_Vec2_distance_squared_wasm"
        "_wcn_math_Vec2_negate_wasm"
        "_wcn_math_Vec2_random_wasm"
        "_wcn_math_Vec2_normalize_wasm"
        "_wcn_math_Vec2_rotate_wasm"
        "_wcn_math_Vec2_set_length_wasm"
        "_wcn_math_Vec2_truncate_wasm"
        "_wcn_math_Vec2_midpoint_wasm"
        "_wcn_math_Vec3_set_wasm"
        "_wcn_math_Vec3_set_x_wasm"
        "_wcn_math_Vec3_set_y_wasm"
        "_wcn_math_Vec3_set_z_wasm"
        "_wcn_math_Vec3_copy_wasm"
        "_wcn_math_Vec3_zero_wasm"
        "_wcn_math_Vec3_identity_wasm"
        "_wcn_math_Vec3_ceil_wasm"
        "_wcn_math_Vec3_floor_wasm"
        "_wcn_math_Vec3_round_wasm"
        "_wcn_math_Vec3_dot_wasm"
        "_wcn_math_Vec3_cross_wasm"
        "_wcn_math_Vec3_length_wasm"
        "_wcn_math_Vec3_length_squared_wasm"
        "_wcn_math_Vec3_normalize_wasm"
        "_wcn_math_Vec3_clamp_wasm"
        "_wcn_math_Vec3_add_wasm"
        "_wcn_math_Vec3_add_scaled_wasm"
        "_wcn_math_Vec3_sub_wasm"
        "_wcn_math_Vec3_angle_wasm"
        "_wcn_math_Vec3_equals_approximately_wasm"
        "_wcn_math_Vec3_equals_wasm"
        "_wcn_math_Vec3_lerp_wasm"
        "_wcn_math_Vec3_lerp_v_wasm"
        "_wcn_math_Vec3_fmax_wasm"
        "_wcn_math_Vec3_fmin_wasm"
        "_wcn_math_Vec3_multiply_scalar_wasm"
        "_wcn_math_Vec3_multiply_wasm"
        "_wcn_math_Vec3_div_scalar_wasm"
        "_wcn_math_Vec3_div_wasm"
        "_wcn_math_Vec3_inverse_wasm"
        "_wcn_math_Vec3_distance_wasm"
        "_wcn_math_Vec3_distance_squared_wasm"
        "_wcn_math_Vec3_negate_wasm"
        "_wcn_math_Vec3_random_wasm"
        "_wcn_math_Vec3_set_length_wasm"
        "_wcn_math_Vec3_truncate_wasm"
        "_wcn_math_Vec3_midpoint_wasm"
        "_wcn_math_Vec4_set_wasm"
        "_wcn_math_Vec4_set_x_wasm"
        "_wcn_math_Vec4_set_y_wasm"
        "_wcn_math_Vec4_set_z_wasm"
        "_wcn_math_Vec4_set_w_wasm"
        "_wcn_math_Vec4_copy_wasm"
        "_wcn_math_Vec4_zero_wasm"
        "_wcn_math_Vec4_identity_wasm"
        "_wcn_math_Vec4_ceil_wasm"
        "_wcn_math_Vec4_floor_wasm"
        "_wcn_math_Vec4_round_wasm"
        "_wcn_math_Vec4_clamp_wasm"
        "_wcn_math_Vec4_add_wasm"
        "_wcn_math_Vec4_add_scaled_wasm"
        "_wcn_math_Vec4_sub_wasm"
        "_wcn_math_Vec4_equals_approximately_wasm"
        "_wcn_math_Vec4_equals_wasm"
        "_wcn_math_Vec4_lerp_wasm"
        "_wcn_math_Vec4_lerp_v_wasm"
        "_wcn_math_Vec4_fmax_wasm"
        "_wcn_math_Vec4_fmin_wasm"
        "_wcn_math_Vec4_multiply_wasm"
        "_wcn_math_Vec4_multiply_scalar_wasm"
        "_wcn_math_Vec4_div_wasm"
        "_wcn_math_Vec4_div_scalar_wasm"
        "_wcn_math_Vec4_inverse_wasm"
        "_wcn_math_Vec4_dot_wasm"
        "_wcn_math_Vec4_length_squared_wasm"
        "_wcn_math_Vec4_length_wasm"
        "_wcn_math_Vec4_distance_wasm"
        "_wcn_math_Vec4_distance_squared_wasm"
        "_wcn_math_Vec4_normalize_wasm"
        "_wcn_math_Vec4_negate_wasm"
        "_wcn_math_Vec4_set_length_wasm"
        "_wcn_math_Vec4_truncate_wasm"
        "_wcn_math_Vec4_midpoint_wasm"
        "_wcn_math_Mat3_set_wasm"
        "_wcn_math_Mat3_set_with_index_wasm"
        "_wcn_math_Mat3_copy_wasm"
        "_wcn_math_Mat3_zero_wasm"
        "_wcn_math_Mat3_identity_wasm"
        "_wcn_math_Mat3_equals_wasm"
        "_wcn_math_Mat3_equals_approximately_wasm"
        "_wcn_math_Mat3_negate_wasm"
        "_wcn_math_Mat3_transpose_wasm"
        "_wcn_math_Mat3_add_wasm"
        "_wcn_math_Mat3_sub_wasm"
        "_wcn_math_Mat3_multiply_wasm"
        "_wcn_math_Mat3_multiply_scalar_wasm"
        "_wcn_math_Mat3_inverse_wasm"
        "_wcn_math_Mat3_determinant_wasm"
        "_wcn_math_Mat4_set_wasm"
        "_wcn_math_Mat4_set_with_index_wasm"
        "_wcn_math_Mat4_copy_wasm"
        "_wcn_math_Mat4_zero_wasm"
        "_wcn_math_Mat4_identity_wasm"
        "_wcn_math_Mat4_negate_wasm"
        "_wcn_math_Mat4_equals_wasm"
        "_wcn_math_Mat4_equals_approximately_wasm"
        "_wcn_math_Mat4_add_wasm"
        "_wcn_math_Mat4_sub_wasm"
        "_wcn_math_Mat4_multiply_wasm"
        "_wcn_math_Mat4_multiply_scalar_wasm"
        "_wcn_math_Mat4_inverse_wasm"
        "_wcn_math_Mat4_transpose_wasm"
        "_wcn_math_Mat4_determinant_wasm"
        "_wcn_math_Mat4_aim_wasm"
        "_wcn_math_Mat4_look_at_wasm"
        "_wcn_math_Mat4_ortho_wasm"
        "_wcn_math_Quat_set_wasm"
        "_wcn_math_Quat_set_x_wasm"
        "_wcn_math_Quat_set_y_wasm"
        "_wcn_math_Quat_set_z_wasm"
        "_wcn_math_Quat_set_w_wasm"
        "_wcn_math_Quat_zero_wasm"
        "_wcn_math_Quat_identity_wasm"
        "_wcn_math_Quat_copy_wasm"
        "_wcn_math_Quat_dot_wasm"
        "_wcn_math_Quat_lerp_wasm"
        "_wcn_math_Quat_slerp_wasm"
        "_wcn_math_Quat_sqlerp_wasm"
        "_wcn_math_Quat_length_wasm"
        "_wcn_math_Quat_length_squared_wasm"
        "_wcn_math_Quat_normalize_wasm"
        "_wcn_math_Quat_equals_approximately_wasm"
        "_wcn_math_Quat_equals_wasm"
        "_wcn_math_Quat_angle_wasm"
        "_wcn_math_Quat_rotation_to_wasm"
        "_wcn_math_Quat_multiply_wasm"
        "_wcn_math_Quat_multiply_scalar_wasm"
        "_wcn_math_Quat_sub_wasm"
        "_wcn_math_Quat_add_wasm"
        "_wcn_math_Quat_inverse_wasm"
        "_wcn_math_Quat_conjugate_wasm"
        "_wcn_math_Quat_div_scalar_wasm"
        "_wcn_math_Mat3_from_mat4_wasm"
        "_wcn_math_Mat3_from_quat_wasm"
        "_wcn_math_Mat4_from_mat3_wasm"
        "_wcn_math_Mat4_from_quat_wasm"
        "_wcn_math_Quat_from_axis_angle_wasm"
        "_wcn_math_Quat_to_axis_angle_wasm"
        "_wcn_math_Vec2_transform_mat4_wasm"
        "_wcn_math_Vec2_transform_mat3_wasm"
        "_wcn_math_Vec3_transform_mat4_wasm"
        "_wcn_math_Vec3_transform_mat4_upper3x3_wasm"
        "_wcn_math_Vec3_transform_mat3_wasm"
        "_wcn_math_Vec3_transform_quat_wasm"
        "_wcn_math_Quat_from_mat4_wasm"
        "_wcn_math_Quat_from_mat3_wasm"
        "_wcn_math_Quat_from_euler_wasm"
        "_wcn_math_Vec3_get_translation_wasm"
        "_wcn_math_Vec3_get_axis_wasm"
        "_wcn_math_Vec3_get_scale_wasm"
        "_wcn_math_Vec3_rotate_x_wasm"
        "_wcn_math_Vec3_rotate_y_wasm"
        "_wcn_math_Vec3_rotate_z_wasm"
        "_wcn_math_Vec4_transform_mat4_wasm"
        "_wcn_math_Quat_rotate_x_wasm"
        "_wcn_math_Quat_rotate_y_wasm"
        "_wcn_math_Quat_rotate_z_wasm"
        "_wcn_math_Mat3_rotate_wasm"
        "_wcn_math_Mat3_rotate_x_wasm"
        "_wcn_math_Mat3_rotate_y_wasm"
        "_wcn_math_Mat3_rotate_z_wasm"
        "_wcn_math_Mat3_rotation_wasm"
        "_wcn_math_Mat3_rotation_x_wasm"
        "_wcn_math_Mat3_rotation_y_wasm"
        "_wcn_math_Mat3_rotation_z_wasm"
        "_wcn_math_Mat3_get_axis_wasm"
        "_wcn_math_Mat3_set_axis_wasm"
        "_wcn_math_Mat3_get_scaling_wasm"
        "_wcn_math_Mat3_get_3D_scaling_wasm"
        "_wcn_math_Mat3_get_translation_wasm"
        "_wcn_math_Mat3_set_translation_wasm"
        "_wcn_math_Mat3_translation_wasm"
        "_wcn_math_Mat3_translate_wasm"
        "_wcn_math_Mat4_axis_rotate_wasm"
        "_wcn_math_Mat4_axis_rotation_wasm"
        "_wcn_math_Mat4_camera_aim_wasm"
        "_wcn_math_Mat4_frustum_wasm"
        "_wcn_math_Mat4_frustum_reverse_z_wasm"
        "_wcn_math_Mat4_get_axis_wasm"
        "_wcn_math_Mat4_set_axis_wasm"
        "_wcn_math_Mat4_get_translation_wasm"
        "_wcn_math_Mat4_set_translation_wasm"
        "_wcn_math_Mat4_translation_wasm"
        "_wcn_math_Mat4_perspective_wasm"
        "_wcn_math_Mat4_perspective_reverse_z_wasm"
        "_wcn_math_Mat4_translate_wasm"
        "_wcn_math_Mat4_rotate_wasm"
        "_wcn_math_Mat4_rotate_x_wasm"
        "_wcn_math_Mat4_rotate_y_wasm"
        "_wcn_math_Mat4_rotate_z_wasm"
        "_wcn_math_Mat4_rotation_wasm"
        "_wcn_math_Mat4_rotation_x_wasm"
        "_wcn_math_Mat4_rotation_y_wasm"
        "_wcn_math_Mat4_rotation_z_wasm"
        "_wcn_math_Vec2_scale_wasm"
        "_wcn_math_Vec3_scale_wasm"
        "_wcn_math_Quat_scale_wasm"
        "_wcn_math_Mat3_scale_wasm"
        "_wcn_math_Mat4_scale_wasm"
        "_wcn_math_Mat3_scale3D_wasm"
        "_wcn_math_Mat3_scaling_wasm"
        "_wcn_math_Mat3_scaling3D_wasm"
        "_wcn_math_Mat3_uniform_scale_wasm"
        "_wcn_math_Mat3_uniform_scale_3D_wasm"
        "_wcn_math_Mat3_uniform_scaling_wasm"
        "_wcn_math_Mat3_uniform_scaling_3D_wasm"
        "_wcn_math_Mat4_get_scaling_wasm"
        "_wcn_math_Mat4_scaling_wasm"
        "_wcn_math_Mat4_uniform_scale_wasm"
        "_wcn_math_Mat4_uniform_scaling_wasm"

        "_malloc"
        "_free"
    )
    # 将列表转换为逗号分隔的字符串
    string(REPLACE ";" "," EXPORTED_FUNCTIONS "${EXPORTED_FUNCTIONS}")
    set(WCN_WASM_RSP "${CMAKE_CURRENT_BINARY_DIR}/wcn_wasm_exports.rsp")
    file(WRITE "${WCN_WASM_RSP}" "-sEXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS}")
    message(STATUS "WASM exports: ${WCN_WASM_RSP}")

    # =========================================================================
    # WASM 编译和链接选项分离配置
    # =========================================================================

    # 1. 纯编译选项 (仅传给 C 编译器)
    set(WASM_ONLY_COMPILE_FLAGS
        -O3                              # 优化级别
    )

    # 2. 纯链接选项 (仅传给 Linker)
    # 所有 -s 开头的 Emscripten 特定参数都在这里
    set(WASM_ONLY_LINK_FLAGS
        -O3                              # 链接时也需要优化标志 (用于 LTO)
        -sUSE_WEBGPU=1                   # 使用 Emscripten 的 WebGPU (Dawn)
        -sWASM=1                         # 生成 WASM
        -sALLOW_MEMORY_GROWTH=1          # 允许内存增长
        -sMODULARIZE=1                   # 模块化输出
        -sEXPORT_NAME=createWCNModule    # 模块名称
        -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,UTF8ToString,stringToUTF8,lengthBytesUTF8,getValue,setValue
        -sENVIRONMENT=web                # 目标环境：浏览器
        -sMAXIMUM_MEMORY=2147483648      # 最大内存 (2GB)
        -sINITIAL_MEMORY=67108864        # 初始内存 (64MB)
        "@${WCN_WASM_RSP}"
    )

    # 3. 处理 SIMD (根据 CMake 选项动态添加)
    if(WCN_ENABLE_SIMD)
        message(STATUS "\n\n\n\n\nWASM SIMD Enabled\n\n\n\n\n")
        # 编译和链接都需要 SIMD 标志
        list(APPEND WASM_ONLY_COMPILE_FLAGS -msimd128)
        list(APPEND WASM_ONLY_LINK_FLAGS -msimd128)
    endif()

    # 4. Debug/Release 构建选项区分
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # 编译时增加调试信息
        list(APPEND WASM_ONLY_COMPILE_FLAGS -g)

        # 链接时增加调试支持、断言和安全检查
        list(APPEND WASM_ONLY_LINK_FLAGS
            -g
            -sASSERTIONS=1
            -sSAFE_HEAP=1
            -sSTACK_OVERFLOW_CHECK=2
            -sDEMANGLE_SUPPORT=1
        )
    else()
        # Release 版本保留基本检查以获得更好的错误信息
        list(APPEND WASM_ONLY_LINK_FLAGS
            -sASSERTIONS=1
            -sSAFE_HEAP=1
        )
    endif()

    # 5. 分别应用编译和链接选项
    target_compile_options(wcn_wasm PRIVATE ${WASM_ONLY_COMPILE_FLAGS})
    target_link_options(wcn_wasm PRIVATE ${WASM_ONLY_LINK_FLAGS})

    # 设置输出文件名
    set_target_properties(wcn_wasm PROPERTIES
        OUTPUT_NAME "wcn"
        SUFFIX ".js"
    )

    # 安装 WASM 文件
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/wcn.js
        ${CMAKE_CURRENT_BINARY_DIR}/wcn.wasm
        DESTINATION lib/wasm
        OPTIONAL
    )

    # 创建一个自定义目标来显示构建信息
    add_custom_command(TARGET wcn_wasm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "WebAssembly build complete:"
        COMMAND ${CMAKE_COMMAND} -E echo "  Output: ${CMAKE_CURRENT_BINARY_DIR}/wcn.js"
        COMMAND ${CMAKE_COMMAND} -E echo "  WASM:   ${CMAKE_CURRENT_BINARY_DIR}/wcn.wasm"
    )

    message(STATUS "WASM build configured. Use 'make wcn_wasm' to build.")
else()
    # 只在非 WASM 构建时添加 examples（examples 依赖 GLFW 等本地库）
    add_subdirectory(examples)
endif()

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