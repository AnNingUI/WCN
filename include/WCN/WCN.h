#ifndef WCN_H
#define WCN_H

#include "WCN/WCN_WGSL.h"
#include "webgpu/webgpu.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 基础类型定义
// ============================================================================

typedef enum WCN_Result {
    WCN_SUCCESS = 0,
    WCN_ERROR_INVALID_PARAMETER,
    WCN_ERROR_OUT_OF_MEMORY,
    WCN_ERROR_GPU_RESOURCE_CREATION_FAILED,
    WCN_ERROR_INVALID_OPERATION,
    WCN_ERROR_DECODER_NOT_REGISTERED
} WCN_Result;

typedef enum WCN_TextAlign {
    WCN_TEXT_ALIGN_LEFT,
    WCN_TEXT_ALIGN_CENTER,
    WCN_TEXT_ALIGN_RIGHT
} WCN_TextAlign;

typedef enum WCN_TextBaseline {
    WCN_TEXT_BASELINE_TOP,
    WCN_TEXT_BASELINE_MIDDLE,
    WCN_TEXT_BASELINE_BOTTOM,
    WCN_TEXT_BASELINE_ALPHABETIC
} WCN_TextBaseline;

typedef enum WCN_CompositeOperation {
    WCN_COMPOSITE_SOURCE_OVER,
    WCN_COMPOSITE_SOURCE_IN,
    WCN_COMPOSITE_SOURCE_OUT,
    WCN_COMPOSITE_SOURCE_ATOP,
    WCN_COMPOSITE_DESTINATION_OVER,
    WCN_COMPOSITE_DESTINATION_IN,
    WCN_COMPOSITE_DESTINATION_OUT,
    WCN_COMPOSITE_DESTINATION_ATOP,
    WCN_COMPOSITE_LIGHTER,
    WCN_COMPOSITE_COPY,
    WCN_COMPOSITE_XOR
} WCN_CompositeOperation;

typedef enum WCN_LineCap {
    WCN_LINE_CAP_BUTT,
    WCN_LINE_CAP_ROUND,
    WCN_LINE_CAP_SQUARE
} WCN_LineCap;

typedef enum WCN_LineJoin {
    WCN_LINE_JOIN_MITER,
    WCN_LINE_JOIN_ROUND,
    WCN_LINE_JOIN_BEVEL
} WCN_LineJoin;

// ============================================================================
// 核心数据结构
// ============================================================================

// WCN Canvas2D Context (前向声明)
typedef struct WCN_Context WCN_Context;

// WebGPU 资源结构
typedef struct WCN_GPUResources {
    WGPUInstance instance;
    WGPUDevice device;
    WGPUQueue queue;
    WGPUSurface surface;
} WCN_GPUResources;

// GPU 状态数据结构（Uniform对齐要求：256字节对齐用于动态偏移）
typedef struct WCN_GPUState {
    float transform_matrix[16];     // 当前变换矩阵 (64 bytes)
    uint32_t fill_color;           // 填充颜色 (RGBA)
    uint32_t stroke_color;         // 描边颜色 (RGBA)
    float stroke_width;            // 描边宽度
    float global_alpha;            // 全局透明度
    uint32_t blend_mode;           // 混合模式
    uint32_t state_flags;          // 状态标志位
    uint32_t line_cap;             // 线帽样式 (WCN_LineCap)
    uint32_t line_join;            // 线连接样式 (WCN_LineJoin)
    float miter_limit;             // 斜接限制（默认10.0）
    float reserved0;               // 保留字段
    uint8_t padding[148];          // 填充到256字节（256 - 108 = 148）
} __attribute__((aligned(256))) WCN_GPUState;

// 状态栈管理
typedef struct WCN_StateStack {
    WCN_GPUState* states;          // GPU 状态缓冲区
    uint32_t current_state;        // 当前状态索引
    uint32_t max_states;           // 最大状态数
    WGPUBuffer state_buffer;       // WebGPU 状态缓冲区
    WGPUBindGroup bind_group;      // 状态绑定组
} WCN_StateStack;

// 图像数据结构
typedef struct WCN_ImageData {
    uint8_t* data;                 // 图像像素数据
    uint32_t width;                // 图像宽度
    uint32_t height;               // 图像高度
    uint32_t format;               // 像素格式 (RGBA8, BGRA8, etc.)
    size_t data_size;              // 数据大小
} WCN_ImageData;

// 路径数据结构
typedef struct WCN_Path {
    float* points;                 // 路径点数组
    size_t point_count;            // 点数量
    uint8_t* commands;             // 路径命令 (moveTo, lineTo, etc.)
    size_t command_count;          // 命令数量
    bool is_closed;                // 路径是否闭合
} WCN_Path;

// 变换矩阵
typedef struct WCN_Transform {
    float matrix[16];              // 4x4 变换矩阵
} WCN_Transform;

// 文本测量结果
typedef struct WCN_TextMetrics {
    float width;
    float actual_bounding_box_ascent;
    float actual_bounding_box_descent;
    float em_height_ascent;
    float em_height_descent;
} WCN_TextMetrics;

// ============================================================================
// 字体和字形数据结构
// ============================================================================

// 字形轮廓点
typedef struct WCN_GlyphPoint {
    float x, y;           // 点坐标
    bool on_curve;        // 是否在曲线上（用于二次贝塞尔曲线）
} WCN_GlyphPoint;

// 字形轮廓
typedef struct WCN_GlyphContour {
    WCN_GlyphPoint* points;  // 轮廓点数组
    size_t point_count;      // 点数量
} WCN_GlyphContour;

// 字形数据
typedef struct WCN_Glyph {
    uint32_t codepoint;           // Unicode 码点
    WCN_GlyphContour* contours;   // 轮廓数组
    size_t contour_count;         // 轮廓数量
    
    // 度量信息
    float advance_width;          // 前进宽度
    float left_side_bearing;      // 左侧边距
    float bounding_box[4];        // 边界框 [x_min, y_min, x_max, y_max]
    
    // 三角化缓存（可选，由 WCN 管理）
    float* vertices;              // 三角化后的顶点
    uint32_t* indices;            // 索引
    size_t vertex_count;          // 顶点数量
    size_t index_count;           // 索引数量
    
    // 原始字体数据（用于 SDF 生成）
    void* raw_vertices;           // stb_truetype 的原始顶点数据
    int raw_vertex_count;         // 原始顶点数量
} WCN_Glyph;

// 字体信息
typedef struct WCN_FontFace {
    const char* family_name;      // 字体家族名称
    
    // 字体度量
    float ascent;                 // 上升高度
    float descent;                // 下降高度
    float line_gap;               // 行间距
    float units_per_em;           // EM 单位
    
    // 解码器私有数据
    void* user_data;
} WCN_FontFace;

// ============================================================================
// 解码器接口
// ============================================================================

// 字体解码器接口
typedef struct WCN_FontDecoder {
    // 加载字体文件
    bool (*load_font)(const void* font_data, size_t data_size, WCN_FontFace** out_face);
    
    // 获取字符字形（轮廓数据，用于三角化）
    bool (*get_glyph)(WCN_FontFace* face, uint32_t codepoint, WCN_Glyph** out_glyph);
    
    // 获取字符 SDF 位图（新增，用于 SDF 渲染）
    // 如果解码器不支持 SDF，可以设置为 NULL，WCN 将使用三角化渲染
    bool (*get_glyph_sdf)(WCN_FontFace* face, uint32_t codepoint, float font_size,
                          unsigned char** out_bitmap,
                          int* out_width, int* out_height,
                          float* out_offset_x, float* out_offset_y,
                          float* out_advance);
    
    // 释放 SDF 位图（新增）
    void (*free_glyph_sdf)(unsigned char* bitmap);
    
    // 测量文本
    bool (*measure_text)(WCN_FontFace* face, const char* text, float font_size, 
                        float* out_width, float* out_height);
    
    // 释放字形资源
    void (*free_glyph)(WCN_Glyph* glyph);
    
    // 释放字体资源
    void (*free_font)(WCN_FontFace* face);
    
    // 解码器名称
    const char* name;
} WCN_FontDecoder;

// 图像解码器接口
typedef struct WCN_ImageDecoder {
    bool (*decode)(const uint8_t* image_data, size_t data_size, WCN_ImageData* out_image);
    const char* name;
} WCN_ImageDecoder;

// ============================================================================
// 核心 API - 上下文管理
// ============================================================================

// 创建和销毁上下文
WCN_Context* wcn_create_context(WCN_GPUResources* gpu_resources);
void wcn_destroy_context(WCN_Context* ctx);

// 帧生命周期管理
void wcn_begin_frame(WCN_Context* ctx, uint32_t width, uint32_t height, WGPUTextureFormat surface_format);
void wcn_end_frame(WCN_Context* ctx);

// 渲染通道管理（高级API）
#ifdef __EMSCRIPTEN__
bool wcn_begin_render_pass(WCN_Context* ctx, int texture_view);
#else
bool wcn_begin_render_pass(WCN_Context* ctx, WGPUTextureView texture_view);
#endif

void wcn_end_render_pass(WCN_Context* ctx);
void wcn_submit_commands(WCN_Context* ctx);

// 状态管理
void wcn_save(WCN_Context* ctx);
void wcn_restore(WCN_Context* ctx);

// 解码器注册
void wcn_register_font_decoder(WCN_Context* ctx, WCN_FontDecoder* decoder);
void wcn_register_image_decoder(WCN_Context* ctx, WCN_ImageDecoder* decoder);

// ============================================================================
// 2D 绘图 API - 矩形操作
// ============================================================================

void wcn_clear_rect(WCN_Context* ctx, float x, float y, float width, float height);
void wcn_fill_rect(WCN_Context* ctx, float x, float y, float width, float height);
void wcn_stroke_rect(WCN_Context* ctx, float x, float y, float width, float height);

// ============================================================================
// 2D 绘图 API - 路径操作
// ============================================================================

void wcn_begin_path(WCN_Context* ctx);
void wcn_close_path(WCN_Context* ctx);
void wcn_move_to(WCN_Context* ctx, float x, float y);
void wcn_line_to(WCN_Context* ctx, float x, float y);
void wcn_arc(WCN_Context* ctx, float x, float y, float radius, float start_angle, float end_angle, bool anticlockwise);
void wcn_rect(WCN_Context* ctx, float x, float y, float width, float height);
void wcn_fill(WCN_Context* ctx);
void wcn_stroke(WCN_Context* ctx);

// ============================================================================
// 2D 绘图 API - 样式和颜色
// ============================================================================

void wcn_set_fill_style(WCN_Context* ctx, uint32_t color);
void wcn_set_stroke_style(WCN_Context* ctx, uint32_t color);
void wcn_set_line_width(WCN_Context* ctx, float width);
void wcn_set_line_cap(WCN_Context* ctx, WCN_LineCap cap);
void wcn_set_line_join(WCN_Context* ctx, WCN_LineJoin join);
void wcn_set_miter_limit(WCN_Context* ctx, float limit);
void wcn_set_global_alpha(WCN_Context* ctx, float alpha);
void wcn_set_global_composite_operation(WCN_Context* ctx, WCN_CompositeOperation operation);

// ============================================================================
// 2D 绘图 API - 变换操作
// ============================================================================

void wcn_translate(WCN_Context* ctx, float x, float y);
void wcn_rotate(WCN_Context* ctx, float angle);
void wcn_scale(WCN_Context* ctx, float x, float y);
void wcn_transform(WCN_Context* ctx, float a, float b, float c, float d, float e, float f);
void wcn_set_transform(WCN_Context* ctx, float a, float b, float c, float d, float e, float f);
void wcn_reset_transform(WCN_Context* ctx);

// ============================================================================
// 2D 绘图 API - 文本渲染
// ============================================================================

void wcn_fill_text(WCN_Context* ctx, const char* text, float x, float y);
void wcn_stroke_text(WCN_Context* ctx, const char* text, float x, float y);
WCN_TextMetrics wcn_measure_text(WCN_Context* ctx, const char* text);
void wcn_set_font(WCN_Context* ctx, const char* font_spec);
void wcn_set_font_face(WCN_Context* ctx, WCN_FontFace* face, float size);
bool wcn_add_font_fallback(WCN_Context* ctx, WCN_FontFace* face);
void wcn_clear_font_fallbacks(WCN_Context* ctx);
void wcn_set_text_align(WCN_Context* ctx, WCN_TextAlign align);
void wcn_set_text_baseline(WCN_Context* ctx, WCN_TextBaseline baseline);

// ============================================================================
// 2D 绘图 API - 图像操作
// ============================================================================

void wcn_draw_image(WCN_Context* ctx, WCN_ImageData* image, float dx, float dy);
void wcn_draw_image_scaled(WCN_Context* ctx, WCN_ImageData* image, float dx, float dy, float dw, float dh);
void wcn_draw_image_source(WCN_Context* ctx, WCN_ImageData* image, float sx, float sy, float sw, float sh, float dx, float dy, float dw, float dh);
WCN_ImageData* wcn_get_image_data(WCN_Context* ctx, float x, float y, float width, float height);
void wcn_put_image_data(WCN_Context* ctx, WCN_ImageData* image_data, float x, float y);
WCN_ImageData* wcn_decode_image(WCN_Context* ctx, const uint8_t* image_bytes, size_t data_size);
void wcn_destroy_image_data(WCN_ImageData* image_data);


// ============================================================================
// 辅助函数
// ============================================================================

// 获取 Surface 格式
WGPUTextureFormat wcn_get_surface_format(WCN_Context* ctx);

// 设置 Surface 格式
void wcn_set_surface_format(WCN_Context* ctx, WGPUTextureFormat format);

#ifdef __cplusplus
}
#endif

#endif // WCN_H
