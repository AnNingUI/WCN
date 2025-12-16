#ifndef WCN_INTERNAL_H
#define WCN_INTERNAL_H

#include "WCN/WCN.h"

// ============================================================================
// 统一渲染系统 (Unified Rendering System)
// ============================================================================

// 实例类型枚举
typedef enum {
    WCN_INSTANCE_TYPE_RECT = 0,
    WCN_INSTANCE_TYPE_TEXT = 1,
    WCN_INSTANCE_TYPE_PATH = 2,
    WCN_INSTANCE_TYPE_LINE = 3,
    WCN_INSTANCE_TYPE_IMAGE = 4,
    WCN_INSTANCE_TYPE_ARC = 5,           // GPU SDF 圆弧 (stroke)
    WCN_INSTANCE_TYPE_BEZIER = 6,        // GPU SDF 二次贝塞尔曲线
    WCN_INSTANCE_TYPE_CIRCLE_FILL = 7    // GPU SDF 圆形/扇形填充
} WCN_InstanceType;

// ============================================================================
// GPU Native 路径命令系统 (内部使用)
// ============================================================================

// 路径命令类型
typedef enum {
    WCN_CMD_MOVE_TO = 0,
    WCN_CMD_LINE_TO = 1,
    WCN_CMD_ARC = 2,           // 圆弧 (cx, cy, radius, start_angle, end_angle, anticlockwise)
    WCN_CMD_QUAD_TO = 3,       // 二次贝塞尔 (cpx, cpy, x, y)
    WCN_CMD_CUBIC_TO = 4,      // 三次贝塞尔 (cp1x, cp1y, cp2x, cp2y, x, y)
    WCN_CMD_CLOSE = 5
} WCN_PathCmdType;

// 路径命令结构 (最多存储 6 个参数)
typedef struct {
    uint8_t type;              // WCN_PathCmdType
    float params[6];           // 参数 (根据类型使用不同数量)
} WCN_PathCmd;

// GPU Native 路径结构 (内部使用，替代原有的 WCN_Path 点数组)
typedef struct {
    WCN_PathCmd* commands;     // 命令数组
    size_t command_count;
    size_t command_capacity;
    
    // 当前点 (用于相对命令)
    float current_x;
    float current_y;
    float start_x;             // 子路径起点 (用于 close)
    float start_y;
    
    bool is_closed;
} WCN_GPUNativePath;

// 统一实例结构 (64 字节，用于 GPU 实例化渲染)
// 字段排列以达到精确 64 字节，无额外填充
typedef struct {
    float position[2];      // 8 bytes: (x, y) 屏幕空间位置
    float size[2];          // 8 bytes: (width, height) 尺寸
    float uv[2];            // 8 bytes: UV 起始坐标
    float uvSize[2];        // 8 bytes: UV 尺寸
    float transform[4];     // 16 bytes: 2x2 变换矩阵 (行优先)
    uint32_t color;         // 4 bytes: RGBA 打包颜色 (8 bits per channel)
    uint32_t type;          // 4 bytes: 实例类型 (WCN_InstanceType)
    uint32_t flags;         // 4 bytes: 渲染标志
    float param0;           // 4 bytes: 类型特定参数 0
} WCN_Instance;             // Total: 60 bytes + 4 bytes padding = 64 bytes

typedef struct {
    float clip_position[4];
    float color[4];
    float uv[2];
    uint32_t instance_type;
    uint32_t flags;
    float local_pos[2];
    float params_x;
    float padding0;
    float size[2];
    float tri_v0[2];
    float tri_v1[2];
    float tri_v2[2];
} WCN_VertexGPU;

typedef struct {
    float viewport_size[2];
    uint32_t instance_count;
    uint32_t instance_offset;
} WCN_RendererUniforms;

// 实例缓冲区（CPU 端动态数组）
typedef struct {
    WCN_Instance* instances;  // 动态数组
    size_t count;             // 当前实例数量
    size_t capacity;          // 已分配容量
} WCN_InstanceBuffer;

// 统一渲染器结构
typedef struct WCN_Renderer {
    // WebGPU 资源
    WGPUDevice device;
    WGPUQueue queue;
    WGPURenderPipeline pipeline;
    WGPUComputePipeline compute_pipeline;
    WGPUBindGroupLayout bind_group_layout;          // Group 0: instances + uniforms
    WGPUBindGroupLayout sdf_bind_group_layout;      // Group 1: SDF atlas texture + sampler
    WGPUBindGroup bind_group;
    WGPUBindGroupLayout compute_bind_group_layout;
    WGPUBindGroup compute_bind_group;
    
    // 缓冲区
    WGPUBuffer instance_buffer;      // 存储缓冲区（实例数据）
    WGPUBuffer uniform_buffer;       // 统一缓冲区（窗口尺寸）
    WGPUBuffer vertex_buffer;        // 计算阶段生成的顶点缓冲
    size_t instance_buffer_size;     // 当前 GPU 缓冲区大小
    size_t vertex_buffer_size;
    size_t vertex_batch_instance_capacity;
    
    // CPU 端实例累积
    WCN_InstanceBuffer cpu_instances;
    
    // 视口
    uint32_t width;
    uint32_t height;
} WCN_Renderer;

// ============================================================================
// Old batch rendering system structures removed
// ============================================================================
// WCN_Vertex, WCN_RenderBatch, and WCN_VertexCollector have been removed
// as they are no longer needed with the unified rendering system.
// ============================================================================

// ============================================================================
// MSDF Atlas 系统
// ============================================================================

// 纹理图集中的字形条目
typedef struct {
    uint32_t codepoint;      // 字符码点
    WCN_FontFace* font_face; // 字体引用
    float font_size;         // 字号
    uint16_t x, y;           // 在图集中的位置（像素）
    uint16_t width, height;  // 字形大小（像素）
    float uv_min[2];         // UV 坐标最小值
    float uv_max[2];         // UV 坐标最大值
    float offset_x, offset_y; // 字形偏移
    float advance_width;     // 字形前进宽度
    bool is_valid;           // 是否有效
    bool is_color;           // 是否为真彩色位图
} WCN_AtlasGlyph;

// MSDF 纹理图集（Multi-channel Signed Distance Field）
typedef struct {
    WGPUTexture texture;
    WGPUTextureView texture_view;
    uint32_t width;          // 图集宽度
    uint32_t height;         // 图集高度
    uint32_t current_x;      // 当前打包位置 X
    uint32_t current_y;      // 当前打包位置 Y
    uint32_t row_height;     // 当前行高度

    // 字形缓存（哈希表）
    WCN_AtlasGlyph* glyphs;
    size_t glyph_count;
    size_t glyph_capacity;
    bool dirty;              // 是否需要刷新到 GPU
} WCN_SDFAtlas;

// 图像纹理图集
typedef struct {
    WCN_ImageData* source;
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
    float uv_min[2];
    float uv_max[2];
    bool is_valid;
} WCN_ImageCacheEntry;

typedef struct {
    WGPUTexture texture;
    WGPUTextureView texture_view;
    uint32_t width;
    uint32_t height;
    uint32_t current_x;
    uint32_t current_y;
    uint32_t row_height;
    WCN_ImageCacheEntry* entries;
    size_t entry_count;
    size_t entry_capacity;
} WCN_ImageAtlas;

// WCN_Context 完整定义（内部使用）
struct WCN_Context {
    // WebGPU 资源
    WGPUInstance instance;
    WGPUDevice device;
    WGPUQueue queue;
    WGPUSurface surface;
    WGPUTextureFormat surface_format;

    // 状态管理
    WCN_StateStack state_stack;
    uint32_t next_gpu_state_slot;  // 下一个可用的 GPU 状态槽位（用于批次渲染）

    // 解码器
    WCN_FontDecoder* font_decoder;
    WCN_ImageDecoder* image_decoder;

    // 当前路径 (保留用于 fill 的点数组)
    WCN_Path* current_path;
    
    // GPU Native 路径 (保留原始命令用于 stroke)
    WCN_GPUNativePath* gpu_path;
    
    // 文本渲染状态
    WCN_FontFace* current_font_face;
    float current_font_size;
    WCN_TextAlign text_align;
    WCN_TextBaseline text_baseline;
    WCN_FontFace** font_fallbacks;
    size_t font_fallback_count;
    size_t font_fallback_capacity;

    // 当前帧状态
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t width;  // 当前窗口宽度
    uint32_t height; // 当前窗口高度
    bool in_frame;

    // 资源管理（旧的即时渲染方式，保留用于兼容）
    WGPUBuffer vertex_buffer;
    WGPUBuffer index_buffer;
    WGPUBuffer uniform_buffer;

    // 渲染状态
#ifdef __EMSCRIPTEN__
    int current_texture_view_id;
#else
    WGPUTextureView current_texture_view;
#endif    
    WGPUCommandEncoder current_command_encoder;
    WGPURenderPassEncoder current_render_pass;
    bool render_pass_needs_begin;
    WGPULoadOp pending_color_load_op;
    WGPUColor pending_clear_color;

    // 顶点/索引数据管理（旧的即时渲染方式，保留用于兼容）
    size_t vertex_buffer_offset;
    size_t index_buffer_offset;

    // 渲染状态
    bool renderer_initialized;
    
    // 绘制调用计数（用于动态偏移）
    uint32_t current_draw_call;
    
    // 调试信息
    uint32_t frame_count;
    
    // 统一渲染系统（Unified Rendering System）
    WCN_Renderer* renderer;
    
    // MSDF Atlas（新增）
    WCN_SDFAtlas* sdf_atlas;  // 注意：虽然名为 sdf_atlas，但实际存储 MSDF 数据
    WGPUSampler sdf_sampler;
    WGPUBindGroup sdf_bind_group;
    WGPUBindGroupLayout sdf_bind_group_layout;
    WCN_ImageAtlas* image_atlas;
    WGPUSampler image_sampler;
    
    size_t text_command_count;
    size_t text_command_capacity;

    // 私有数据
    void* user_data;
};

// 渲染后端初始化函数
bool wcn_initialize_renderer(WCN_Context* ctx);

// 顶点/索引缓冲区管理（旧的即时渲染方式）
bool wcn_write_vertex_data(WCN_Context* ctx, const void* data, size_t size, size_t* out_offset);
bool wcn_write_index_data(WCN_Context* ctx, const void* data, size_t size, size_t* out_offset);

// ============================================================================
// Old batch rendering system function declarations removed
// ============================================================================
// Function declarations for wcn_init_vertex_collector, wcn_destroy_vertex_collector,
// wcn_clear_vertex_collector, wcn_add_quad, wcn_add_triangles, wcn_optimize_batches,
// and wcn_render_batches have been removed as they are no longer needed.
// ============================================================================

// ============================================================================
// MSDF Atlas 管理
// ============================================================================

// 创建和销毁 MSDF Atlas
WCN_SDFAtlas* wcn_create_sdf_atlas(WCN_Context* ctx, uint32_t width, uint32_t height);
void wcn_destroy_sdf_atlas(WCN_SDFAtlas* atlas);

// 刷新 Atlas 到 GPU
void wcn_flush_sdf_atlas(WCN_Context* ctx);

// 打包字形到 atlas（MSDF 格式：RGBA，每像素 4 字节）
bool wcn_atlas_pack_glyph(WCN_Context* ctx,
                          unsigned char* msdf_bitmap,
                          int width, int height,
                          float offset_x, float offset_y,
                          float advance,
                          uint32_t codepoint,
                          float font_size,
                          WCN_AtlasGlyph* out_glyph);

// 在缓存中查找字形
WCN_AtlasGlyph* wcn_find_glyph_in_atlas(WCN_SDFAtlas* atlas,
                                        WCN_FontFace* face,
                                        uint32_t codepoint,
                                        float font_size);

// 获取或创建字形
WCN_AtlasGlyph* wcn_get_or_create_glyph(WCN_Context* ctx,
                                        WCN_FontFace* face,
                                        uint32_t codepoint,
                                        float font_size);

// UTF-8 解码
uint32_t wcn_decode_utf8(const char** str);

// 图像图集管理
WCN_ImageAtlas* wcn_create_image_atlas(WCN_Context* ctx, uint32_t width, uint32_t height);
void wcn_destroy_image_atlas(WCN_ImageAtlas* atlas);
WCN_ImageCacheEntry* wcn_image_atlas_get_entry(WCN_Context* ctx, WCN_ImageData* image);
bool wcn_init_image_manager(WCN_Context* ctx);
void wcn_shutdown_image_manager(WCN_Context* ctx);

// ============================================================================
// 统一渲染器管理 (Unified Renderer Management)
// ============================================================================

// 创建和销毁渲染器
WCN_Renderer* wcn_create_renderer(
    WGPUDevice device,
    WGPUQueue queue,
    WGPUTextureFormat surface_format,
    uint32_t width,
    uint32_t height
);

void wcn_destroy_renderer(WCN_Renderer* renderer);

// 实例缓冲区管理
bool wcn_instance_buffer_init(WCN_InstanceBuffer* buffer, size_t initial_capacity);
void wcn_instance_buffer_destroy(WCN_InstanceBuffer* buffer);
void wcn_instance_buffer_clear(WCN_InstanceBuffer* buffer);
bool wcn_instance_buffer_grow(WCN_InstanceBuffer* buffer);
bool wcn_instance_buffer_add(WCN_InstanceBuffer* buffer, const WCN_Instance* instance);

// 添加矩形实例
void wcn_renderer_add_rect(
    WCN_Renderer* renderer,
    float x, float y, float width, float height,
    float radius,  // Added radius parameter
    uint32_t color,
    const float transform[4]
);

void wcn_renderer_add_image(
    WCN_Renderer* renderer,
    float x, float y,
    float width, float height,
    uint32_t color,
    const float transform[4],
    const float uv_min[2],
    const float uv_size[2]
);

void wcn_renderer_add_text(
    WCN_Renderer* renderer,
    WCN_Context* ctx,
    const char* text,
    float x, float y,
    float font_size,
    uint32_t color,
    const float transform[4]
);

// Simple vertex structure for triangle rendering (position only)
typedef struct {
    float position[2];
} WCN_SimpleVertex;

void wcn_renderer_add_triangles(
    WCN_Renderer* renderer,
    const WCN_SimpleVertex* vertices, size_t vertex_count,
    const uint16_t* indices, size_t index_count,
    uint32_t color,
    const float transform[4]
);

void wcn_renderer_add_line(
    WCN_Renderer* renderer,
    float x1, float y1,
    float x2, float y2,
    float width,
    uint32_t color,
    const float transform[4],
    uint32_t line_cap  // Line cap style (WCN_LineCap)
);

// 添加圆弧实例 (GPU SDF 渲染)
void wcn_renderer_add_arc(
    WCN_Renderer* renderer,
    float cx, float cy,           // 圆心
    float radius,                 // 半径
    float start_angle,            // 起始角度 (弧度)
    float end_angle,              // 结束角度 (弧度)
    float stroke_width,           // 描边宽度
    uint32_t color,
    const float transform[4],
    uint32_t flags                // bit 0: is_fill
);

// 添加圆形/扇形填充实例 (GPU SDF 渲染)
void wcn_renderer_add_circle_fill(
    WCN_Renderer* renderer,
    float cx, float cy,           // 圆心
    float radius,                 // 半径
    float start_angle,            // 起始角度 (弧度)
    float end_angle,              // 结束角度 (弧度)
    uint32_t color,
    const float transform[4]
);

// 添加二次贝塞尔曲线实例 (GPU SDF 渲染)
void wcn_renderer_add_quadratic_bezier(
    WCN_Renderer* renderer,
    float x0, float y0,           // 起点
    float cpx, float cpy,         // 控制点
    float x1, float y1,           // 终点
    float stroke_width,           // 描边宽度
    uint32_t color,
    const float transform[4],
    uint32_t flags
);

// 添加三次贝塞尔曲线 (分解为两个二次贝塞尔)
void wcn_renderer_add_cubic_bezier(
    WCN_Renderer* renderer,
    float x0, float y0,           // 起点
    float cp1x, float cp1y,       // 控制点1
    float cp2x, float cp2y,       // 控制点2
    float x1, float y1,           // 终点
    float stroke_width,           // 描边宽度
    uint32_t color,
    const float transform[4],
    uint32_t flags
);

// 渲染
void wcn_renderer_render(
    WCN_Context* ctx,
    WGPUTextureView sdf_atlas_view,
    WGPUTextureView image_atlas_view
);

// 工具函数
void wcn_renderer_clear(WCN_Renderer* renderer);
void wcn_renderer_resize(WCN_Renderer* renderer, uint32_t width, uint32_t height);

// ============================================================================
// 三角化函数（wcn_triangulate.c）
// ============================================================================

// 三角化单个轮廓
bool wcn_triangulate_contour(WCN_GlyphContour* contour,
                             float** out_vertices,
                             uint32_t** out_indices,
                             size_t* out_vertex_count,
                             size_t* out_index_count);

// 三角化字形（支持内轮廓/洞）
bool wcn_triangulate_glyph_with_holes(WCN_Glyph* glyph,
                                      float** out_vertices,
                                      uint32_t** out_indices,
                                      size_t* out_vertex_count,
                                      size_t* out_index_count);

#endif // WCN_INTERNAL_H
