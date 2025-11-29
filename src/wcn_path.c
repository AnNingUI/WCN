#include "wcn_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================================
// Line Join辅助函数（改进的三角形方法）
// ============================================================================

// 添加ROUND连接（高质量圆角）
static void add_round_join(WCN_Renderer* renderer,
                           float x, float y,
                           float perp1_x, float perp1_y,
                           float perp2_x, float perp2_y,
                           float half_width,
                           uint32_t color,
                           const float transform[4]) {
    float angle1 = atan2f(perp1_y, perp1_x);
    float angle2 = atan2f(perp2_y, perp2_x);
    
    float angle_diff = angle2 - angle1;
    if (angle_diff > M_PI) angle_diff -= 2.0f * M_PI;
    if (angle_diff < -M_PI) angle_diff += 2.0f * M_PI;
    
    int segments = (int)(fabsf(angle_diff) / (M_PI / 16.0f)) + 1;
    if (segments < 4) segments = 4;
    if (segments > 32) segments = 32;
    
    WCN_SimpleVertex* vertices = (WCN_SimpleVertex*)malloc((segments + 2) * sizeof(WCN_SimpleVertex));
    uint16_t* indices = (uint16_t*)malloc(segments * 3 * sizeof(uint16_t));
    
    if (!vertices || !indices) {
        free(vertices);
        free(indices);
        return;
    }
    
    vertices[0].position[0] = x;
    vertices[0].position[1] = y;
    
    for (int i = 0; i <= segments; i++) {
        float t = (float)i / (float)segments;
        float angle = angle1 + angle_diff * t;
        vertices[i + 1].position[0] = x + cosf(angle) * half_width;
        vertices[i + 1].position[1] = y + sinf(angle) * half_width;
    }
    
    for (int i = 0; i < segments; i++) {
        indices[i * 3 + 0] = 0;
        indices[i * 3 + 1] = i + 1;
        indices[i * 3 + 2] = i + 2;
    }
    
    wcn_renderer_add_triangles(renderer, vertices, segments + 2, indices, segments * 3, color, transform);
    
    free(vertices);
    free(indices);
}

// 添加BEVEL连接
static void add_bevel_join(WCN_Renderer* renderer,
                           float x, float y,
                           float perp1_x, float perp1_y,
                           float perp2_x, float perp2_y,
                           float half_width,
                           uint32_t color,
                           const float transform[4]) {
    WCN_SimpleVertex vertices[3];
    vertices[0].position[0] = x;
    vertices[0].position[1] = y;
    vertices[1].position[0] = x + perp1_x * half_width;
    vertices[1].position[1] = y + perp1_y * half_width;
    vertices[2].position[0] = x + perp2_x * half_width;
    vertices[2].position[1] = y + perp2_y * half_width;
    
    uint16_t indices[3] = {0, 1, 2};
    wcn_renderer_add_triangles(renderer, vertices, 3, indices, 3, color, transform);
}

// 添加MITER连接（尖角）
static void add_miter_join(WCN_Renderer* renderer,
                           float x, float y,
                           float perp1_x, float perp1_y,
                           float perp2_x, float perp2_y,
                           float half_width,
                           uint32_t color,
                           const float transform[4],
                           float miter_limit) {
    // 计算两条线外边缘的端点
    float p1_x = x + perp1_x * half_width;
    float p1_y = y + perp1_y * half_width;
    float p2_x = x + perp2_x * half_width;
    float p2_y = y + perp2_y * half_width;
    
    // 计算miter点（两条延长线的交点）
    // 使用线段交点公式
    float denom = perp1_x * perp2_y - perp1_y * perp2_x;
    
    if (fabsf(denom) < 0.001f) {
        // 平行线，退化为bevel
        add_bevel_join(renderer, x, y, perp1_x, perp1_y, perp2_x, perp2_y, 
                      half_width, color, transform);
        return;
    }
    
    // 计算从p1沿perp1方向到p2沿perp2方向的交点
    float dx = p2_x - p1_x;
    float dy = p2_y - p1_y;
    float t1 = (dx * perp2_y - dy * perp2_x) / denom;
    
    // Miter点
    float miter_x = p1_x + perp1_x * t1;
    float miter_y = p1_y + perp1_y * t1;
    
    // 检查miter长度
    float miter_dx = miter_x - x;
    float miter_dy = miter_y - y;
    float miter_length = sqrtf(miter_dx * miter_dx + miter_dy * miter_dy);
    
    if (miter_length > half_width * miter_limit) {
        // Miter太长，退化为bevel
        add_bevel_join(renderer, x, y, perp1_x, perp1_y, perp2_x, perp2_y, 
                      half_width, color, transform);
        return;
    }
    
    // 创建miter三角形：中心点 -> p1 -> miter点 -> p2
    WCN_SimpleVertex vertices[4];
    vertices[0].position[0] = x;
    vertices[0].position[1] = y;
    vertices[1].position[0] = p1_x;
    vertices[1].position[1] = p1_y;
    vertices[2].position[0] = miter_x;
    vertices[2].position[1] = miter_y;
    vertices[3].position[0] = p2_x;
    vertices[3].position[1] = p2_y;
    
    // 两个三角形：(0,1,2) 和 (0,2,3)
    uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
    
    wcn_renderer_add_triangles(renderer, vertices, 4, indices, 6, color, transform);
}

// ============================================================================
// 路径渲染合批化 - 已实现
// ============================================================================
// 路径渲染现在使用批次系统：
// 1. 三角化后的顶点添加到 WCN_VertexCollector
// 2. 使用 wcn_add_triangles() 函数添加到批次
// 3. 在 wcn_end_frame() 时统一渲染
//
// 优势：
// - 减少绘制调用次数
// - 更好的性能
// - 与文本和矩形渲染统一
// ============================================================================

// ============================================================================
// 路径操作私有函数
// ============================================================================

static WCN_Path* wcn_get_current_path(WCN_Context* ctx) {
    if (!ctx) return NULL;
    return ctx->current_path;
}

static void wcn_create_new_path(WCN_Context* ctx) {
    if (!ctx) return;

    // 释放现有的路径
    if (ctx->current_path) {
        if (ctx->current_path->points) {
            free(ctx->current_path->points);
        }
        if (ctx->current_path->commands) {
            free(ctx->current_path->commands);
        }
        free(ctx->current_path);
    }

    // 创建新的路径
    ctx->current_path = (WCN_Path*)malloc(sizeof(WCN_Path));
    if (ctx->current_path) {
        memset(ctx->current_path, 0, sizeof(WCN_Path));
        ctx->current_path->is_closed = false;
    }
}

static void wcn_path_add_point(WCN_Path* path, float x, float y, uint8_t command) {
    if (!path) return;

    // 扩展点数组
    size_t new_point_count = path->point_count + 2;
    float* new_points = (float*)realloc(path->points, new_point_count * sizeof(float));
    if (!new_points) return;

    path->points = new_points;
    path->points[path->point_count] = x;
    path->points[path->point_count + 1] = y;
    path->point_count = new_point_count;

    // 扩展命令数组
    size_t new_command_count = path->command_count + 1;
    uint8_t* new_commands = (uint8_t*)realloc(path->commands, new_command_count * sizeof(uint8_t));
    if (!new_commands) return;

    path->commands = new_commands;
    path->commands[path->command_count] = command;
    path->command_count = new_command_count;
}

static void wcn_path_close(WCN_Path* path) {
    if (!path) return;
    path->is_closed = true;
}

// 使用扇形三角化填充路径 - 使用统一渲染器
static void wcn_render_path_fill(WCN_Context* ctx, WCN_Path* path) {
    if (!ctx || !path || path->point_count < 6 || !ctx->renderer) return;

    WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];
    uint32_t color = state->fill_color;
    size_t num_points = path->point_count / 2;

    // For a proper triangle fill, we need to triangulate the path
    // For a simple convex polygon (like a triangle), we can use a fan triangulation
    
    // Create vertices for all points in the path
    WCN_SimpleVertex* vertices = (WCN_SimpleVertex*)malloc(num_points * sizeof(WCN_SimpleVertex));
    if (!vertices) return;

    // Fill vertices with path points
    for (size_t i = 0; i < num_points; i++) {
        vertices[i].position[0] = path->points[i*2];
        vertices[i].position[1] = path->points[i*2+1];
    }

    // Create indices for triangle fan triangulation
    // For a convex polygon, we can triangulate by connecting all vertices to the first vertex
    size_t num_triangles = num_points - 2;
    size_t num_indices = num_triangles * 3;
    uint16_t* indices = (uint16_t*)malloc(num_indices * sizeof(uint16_t));
    if (!indices) { 
        free(vertices); 
        return; 
    }

    // Generate triangle fan indices correctly
    // For a triangle fan, all triangles share the first vertex (0)
    // and each triangle uses two consecutive vertices from the remaining ones
    for (size_t i = 0; i < num_triangles; i++) {
        indices[i*3] = 0;        // First vertex (shared by all triangles)
        indices[i*3+1] = i+1;    // Current vertex
        indices[i*3+2] = i+2;    // Next vertex
    }

    float transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };

    wcn_renderer_add_triangles(ctx->renderer, vertices, num_points, indices, num_indices, color, transform);

    free(vertices);
    free(indices);
}

// 渲染路径描边（线段）- 使用统一渲染器，支持Line Join
static void wcn_render_path_stroke(WCN_Context* ctx, WCN_Path* path) {
    if (!ctx || !path || path->point_count < 4 || !ctx->renderer) return; // 至少需要2个点

    // 获取当前样式
    WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];
    uint32_t color = state->stroke_color;
    float line_width = state->stroke_width;
    float half_width = line_width * 0.5f;
    uint32_t line_join = state->line_join;
    float miter_limit = state->miter_limit;
    
    // 提取 2x2 变换矩阵（从 4x4 矩阵的左上角）
    float transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };
    
    size_t num_points = path->point_count / 2;
    size_t num_segments = path->is_closed ? num_points : (num_points - 1);

    // 渲染每条线段和连接点
    for (size_t i = 0; i < num_segments; i++) {
        size_t p1_idx = i;
        size_t p2_idx = (i + 1) % num_points;

        float x1 = path->points[p1_idx * 2];
        float y1 = path->points[p1_idx * 2 + 1];
        float x2 = path->points[p2_idx * 2];
        float y2 = path->points[p2_idx * 2 + 1];

        // 确定是否在端点渲染cap
        // 对于路径中间的线段，禁用cap（由line join处理）
        // 只在路径的真正端点才渲染cap
        bool is_first_segment = (i == 0);
        bool is_last_segment = (i == num_segments - 1);
        bool render_start_cap = is_first_segment && !path->is_closed;
        bool render_end_cap = is_last_segment && !path->is_closed;
        
        // 构建cap flags: bit 0-7 = cap style, bit 8 = start cap, bit 9 = end cap
        uint32_t cap_flags = state->line_cap & 0xFFu;
        if (render_start_cap) cap_flags |= (1u << 8);
        if (render_end_cap) cap_flags |= (1u << 9);

        // 添加线段到统一渲染器
        wcn_renderer_add_line(
            ctx->renderer,
            x1, y1,
            x2, y2,
            line_width,
            color,
            transform,
            cap_flags
        );
        
        // 添加连接点（如果不是最后一个线段，或者路径是闭合的）
        bool should_add_join = (i < num_segments - 1) || path->is_closed;
        
        if (should_add_join && num_points >= 3) {
            size_t p3_idx = (i + 2) % num_points;
            
            // 跳过闭合路径的最后一个连接（会在第一个点处理）
            if (!path->is_closed || i < num_segments - 1) {
                float x3 = path->points[p3_idx * 2];
                float y3 = path->points[p3_idx * 2 + 1];
                
                // 计算两条线段的方向向量
                float dx1 = x2 - x1;
                float dy1 = y2 - y1;
                float len1 = sqrtf(dx1 * dx1 + dy1 * dy1);
                
                float dx2 = x3 - x2;
                float dy2 = y3 - y2;
                float len2 = sqrtf(dx2 * dx2 + dy2 * dy2);
                
                if (len1 > 0.001f && len2 > 0.001f) {
                    // 归一化方向向量
                    dx1 /= len1; dy1 /= len1;
                    dx2 /= len2; dy2 /= len2;
                    
                    // 计算垂直向量
                    float perp1_x = -dy1;
                    float perp1_y = dx1;
                    float perp2_x = -dy2;
                    float perp2_y = dx2;
                    
                    // 使用三角形join
                    switch (line_join) {
                        case 0: // WCN_LINE_JOIN_MITER
                            add_miter_join(ctx->renderer, x2, y2, perp1_x, perp1_y,
                                         perp2_x, perp2_y, half_width, color, transform, miter_limit);
                            break;
                        case 1: // WCN_LINE_JOIN_ROUND
                            add_round_join(ctx->renderer, x2, y2, perp1_x, perp1_y,
                                         perp2_x, perp2_y, half_width, color, transform);
                            break;
                        case 2: // WCN_LINE_JOIN_BEVEL
                            add_bevel_join(ctx->renderer, x2, y2, perp1_x, perp1_y,
                                         perp2_x, perp2_y, half_width, color, transform);
                            break;
                    }
                }
            }
        }
    }
}

static void wcn_render_path(WCN_Context* ctx, WCN_Path* path, bool is_stroke) {
    if (!ctx || !ctx->in_frame || !path) return;

    if (is_stroke) {
        wcn_render_path_stroke(ctx, path);
    } else {
        wcn_render_path_fill(ctx, path);
    }
}

// ============================================================================
// 公共API实现
// ============================================================================

void wcn_begin_path(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;

    // 创建新的路径
    wcn_create_new_path(ctx);
}

void wcn_close_path(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* current_path = wcn_get_current_path(ctx);
    if (current_path) {
        wcn_path_close(current_path);
    }
}

void wcn_move_to(WCN_Context* ctx, float x, float y) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* current_path = wcn_get_current_path(ctx);
    if (current_path) {
        wcn_path_add_point(current_path, x, y, 0); // 0 = moveTo
    }
}

void wcn_line_to(WCN_Context* ctx, float x, float y) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* current_path = wcn_get_current_path(ctx);
    if (current_path) {
        wcn_path_add_point(current_path, x, y, 1); // 1 = lineTo
    }
}

void wcn_arc(WCN_Context* ctx, float x, float y, float radius,
             float start_angle, float end_angle, bool anticlockwise) {
    if (!ctx || !ctx->in_frame || radius <= 0.0f) return;

    WCN_Path* current_path = wcn_get_current_path(ctx);
    if (!current_path) return;

    float angle_diff = end_angle - start_angle;
    if (anticlockwise && angle_diff > 0) angle_diff -= 2.0f * M_PI;
    if (!anticlockwise && angle_diff < 0) angle_diff += 2.0f * M_PI;

    int segments = (int)(radius * fabsf(angle_diff) / 2.0f);
    if (segments < 4) segments = 4;
    if (segments > 256) segments = 256;

    for (int i = 0; i <= segments; i++) {
        float t = (float)i / segments;
        float angle = start_angle + angle_diff * t;
        float px = x + cosf(angle) * radius;
        float py = y + sinf(angle) * radius;

        uint8_t cmd = (i == 0 && current_path->point_count == 0) ? 0 : 1;
        wcn_path_add_point(current_path, px, py, cmd);
    }

    // 确保闭合路径（用于填充）
    if (!current_path->is_closed && fabsf(angle_diff) >= 2.0f * M_PI - 0.001f) {
        wcn_path_close(current_path);
    }
}


void wcn_rect(WCN_Context* ctx, float x, float y, float width, float height) {
    if (!ctx || !ctx->in_frame) return;

    // 创建矩形路径
    wcn_move_to(ctx, x, y);
    wcn_line_to(ctx, x + width, y);
    wcn_line_to(ctx, x + width, y + height);
    wcn_line_to(ctx, x, y + height);
    wcn_close_path(ctx);
}

void wcn_fill(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* current_path = wcn_get_current_path(ctx);
    if (current_path) {
        wcn_render_path(ctx, current_path, false);
    }
}

void wcn_stroke(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* current_path = wcn_get_current_path(ctx);
    if (current_path) {
        wcn_render_path(ctx, current_path, true);
    }
}
