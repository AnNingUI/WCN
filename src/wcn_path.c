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
    const float angle1 = atan2f(perp1_y, perp1_x);
    const float angle2 = atan2f(perp2_y, perp2_x);

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
        const float t = (float)i / (float)segments;
        const float angle = angle1 + angle_diff * t;
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
                           const float x, const float y,
                           const float perp1_x, const float perp1_y,
                           const float perp2_x, const float perp2_y,
                           const float half_width,
                           const uint32_t color,
                           const float transform[4]) {
    WCN_SimpleVertex vertices[3];
    vertices[0].position[0] = x;
    vertices[0].position[1] = y;
    vertices[1].position[0] = x + perp1_x * half_width;
    vertices[1].position[1] = y + perp1_y * half_width;
    vertices[2].position[0] = x + perp2_x * half_width;
    vertices[2].position[1] = y + perp2_y * half_width;

    const uint16_t indices[3] = {0, 1, 2};
    wcn_renderer_add_triangles(renderer, vertices, 3, indices, 3, color, transform);
}

// 添加MITER连接（尖角）
static void add_miter_join(WCN_Renderer* renderer,
                           const float x, const float y,
                           const float perp1_x, const float perp1_y,
                           const float perp2_x, const float perp2_y,
                           const float half_width,
                           const uint32_t color,
                           const float transform[4],
                           const float miter_limit) {
    // 计算两条线外边缘的端点
    const float p1_x = x + perp1_x * half_width;
    const float p1_y = y + perp1_y * half_width;
    const float p2_x = x + perp2_x * half_width;
    const float p2_y = y + perp2_y * half_width;

    // 计算miter点（两条延长线的交点）
    // 使用线段交点公式
    const float denom = perp1_x * perp2_y - perp1_y * perp2_x;

    if (fabsf(denom) < 0.001f) {
        // 平行线，退化为bevel
        add_bevel_join(renderer, x, y, perp1_x, perp1_y, perp2_x, perp2_y,
                      half_width, color, transform);
        return;
    }

    // 计算从p1沿perp1方向到p2沿perp2方向的交点
    const float dx = p2_x - p1_x;
    const float dy = p2_y - p1_y;
    const float t1 = (dx * perp2_y - dy * perp2_x) / denom;

    // Miter点
    const float miter_x = p1_x + perp1_x * t1;
    const float miter_y = p1_y + perp1_y * t1;

    // 检查miter长度
    const float miter_dx = miter_x - x;
    const float miter_dy = miter_y - y;
    const float miter_length = sqrtf(miter_dx * miter_dx + miter_dy * miter_dy);

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
    const uint16_t indices[6] = {0, 1, 2, 0, 2, 3};

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

    // 限制路径大小，防止内存过度增长
    const size_t MAX_PATH_POINTS = 10000;
    const size_t MAX_PATH_COMMANDS = 5000;

    if (path->point_count >= MAX_PATH_POINTS * 2 || path->command_count >= MAX_PATH_COMMANDS) {
        return; // 路径过大，拒绝添加新点
    }

    // 扩展点数组
    const size_t new_point_count = path->point_count + 2;
    float* new_points = (float*)realloc(path->points, new_point_count * sizeof(float));
    if (!new_points) return;

    path->points = new_points;
    path->points[path->point_count] = x;
    path->points[path->point_count + 1] = y;
    path->point_count = new_point_count;

    // 扩展命令数组
    const size_t new_command_count = path->command_count + 1;
    uint8_t* new_commands = (uint8_t*)realloc(path->commands, new_command_count * sizeof(uint8_t));
    if (!new_commands) {
        // 如果命令数组分配失败，回滚点数组
        path->points = (float*)realloc(path->points, path->point_count * sizeof(float));
        return;
    }

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

    const WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];
    const uint32_t color = state->fill_color;
    const size_t num_points = path->point_count / 2;

    // For a proper triangle fill, we need to triangulate the path
    // For a simple convex polygon (like a triangle), we can use a fan triangulation

    // Create vertices for all points in the path
    WCN_SimpleVertex* vertices = (WCN_SimpleVertex*)malloc(num_points * sizeof(WCN_SimpleVertex));
    if (!vertices) return;

    // Fill vertices with path points, applying the current transformation
    for (size_t i = 0; i < num_points; i++) {
        float x = path->points[i*2];
        float y = path->points[i*2+1];

        // Apply transformation: result = transformation_matrix * [x, y, 0, 1]
        // For 2D in 4x4 matrix (column-major order):
        // x' = m[0]*x + m[4]*y + m[12]
        // y' = m[1]*x + m[5]*y + m[13]
        vertices[i].position[0] = x * state->transform_matrix[0] + y * state->transform_matrix[4] + state->transform_matrix[12];
        vertices[i].position[1] = x * state->transform_matrix[1] + y * state->transform_matrix[5] + state->transform_matrix[13];
    }

    // Create indices for triangle fan triangulation
    // For a convex polygon, we can triangulate by connecting all vertices to the first vertex
    const size_t num_triangles = num_points - 2;
    const size_t num_indices = num_triangles * 3;
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

    const float transform[4] = {
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
    const WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];
    const uint32_t color = state->stroke_color;
    const float line_width = state->stroke_width;
    const float half_width = line_width * 0.5f;
    const uint32_t line_join = state->line_join;
    const float miter_limit = state->miter_limit;

    // 提取 2x2 变换矩阵（从 4x4 矩阵的左上角）
    const float transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };

    const size_t num_points = path->point_count / 2;
    const size_t num_segments = path->is_closed ? num_points : (num_points - 1);

    // 渲染每条线段和连接点
    for (size_t i = 0; i < num_segments; i++) {
        const size_t p1_idx = i;
        const size_t p2_idx = (i + 1) % num_points;

        // Apply transformation to path points
        // x' = m[0]*x + m[4]*y + m[12]
        // y' = m[1]*x + m[5]*y + m[13]
        const float x1 = path->points[p1_idx * 2] * state->transform_matrix[0] +
                         path->points[p1_idx * 2 + 1] * state->transform_matrix[4] +
                         state->transform_matrix[12];
        const float y1 = path->points[p1_idx * 2] * state->transform_matrix[1] +
                         path->points[p1_idx * 2 + 1] * state->transform_matrix[5] +
                         state->transform_matrix[13];
        const float x2 = path->points[p2_idx * 2] * state->transform_matrix[0] +
                         path->points[p2_idx * 2 + 1] * state->transform_matrix[4] +
                         state->transform_matrix[12];
        const float y2 = path->points[p2_idx * 2] * state->transform_matrix[1] +
                         path->points[p2_idx * 2 + 1] * state->transform_matrix[5] +
                         state->transform_matrix[13];

        // 确定是否在端点渲染cap
        // 对于路径中间的线段，禁用cap（由line join处理）
        // 只在路径的真正端点才渲染cap
        const bool is_first_segment = (i == 0);
        const bool is_last_segment = (i == num_segments - 1);
        const bool render_start_cap = is_first_segment && !path->is_closed;
        const bool render_end_cap = is_last_segment && !path->is_closed;

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
        const bool should_add_join = (i < num_segments - 1) || path->is_closed;

        if (should_add_join && num_points >= 3) {
            const size_t p3_idx = (i + 2) % num_points;

            // 跳过闭合路径的最后一个连接（会在第一个点处理）
            if (!path->is_closed || i < num_segments - 1) {
                // Apply transformation to path point
                const float x3 = path->points[p3_idx * 2] * state->transform_matrix[0] +
                                 path->points[p3_idx * 2 + 1] * state->transform_matrix[4] +
                                 state->transform_matrix[12];
                const float y3 = path->points[p3_idx * 2] * state->transform_matrix[1] +
                                 path->points[p3_idx * 2 + 1] * state->transform_matrix[5] +
                                 state->transform_matrix[13];

                // 计算两条线段的方向向量
                float dx1 = x2 - x1;
                float dy1 = y2 - y1;
                const float len1 = sqrtf(dx1 * dx1 + dy1 * dy1);

                float dx2 = x3 - x2;
                float dy2 = y3 - y2;
                const float len2 = sqrtf(dx2 * dx2 + dy2 * dy2);

                if (len1 > 0.001f && len2 > 0.001f) {
                    // 归一化方向向量
                    dx1 /= len1; dy1 /= len1;
                    dx2 /= len2; dy2 /= len2;

                    // 计算垂直向量
                    const float perp1_x = -dy1;
                    const float perp1_y = dx1;
                    const float perp2_x = -dy2;
                    const float perp2_y = dx2;

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
                        default:
                            break;
                    }
                }
            }
        }
    }
}

static void wcn_render_path(WCN_Context* ctx, WCN_Path* path, const bool is_stroke) {
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

void wcn_move_to(WCN_Context* ctx, const float x, const float y) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* current_path = wcn_get_current_path(ctx);
    if (current_path) {
        wcn_path_add_point(current_path, x, y, 0); // 0 = moveTo
    }
}

void wcn_line_to(WCN_Context* ctx, const float x, const float y) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* current_path = wcn_get_current_path(ctx);
    if (current_path) {
        wcn_path_add_point(current_path, x, y, 1); // 1 = lineTo
    }
}

void wcn_arc(WCN_Context* ctx, const float x, const float y, const float radius,
             const float start_angle, const float end_angle, const bool anticlockwise) {
    if (!ctx || !ctx->in_frame || radius <= 0.0f) return;

    WCN_Path* current_path = wcn_get_current_path(ctx);
    if (!current_path) return;

    float angle_diff = end_angle - start_angle;
    if (anticlockwise) {
        // 确保逆时针角度差是负的，且绝对值在 (0, 2*PI] 之间
        while (angle_diff >= 0.0f) angle_diff -= 2.0f * M_PI;
    } else {
        // 确保顺时针角度差是正的，且绝对值在 (0, 2*PI] 之间
        while (angle_diff <= 0.0f) angle_diff += 2.0f * M_PI;
    }

    // 确定细分段数
    // 依然使用圆弧长度的启发式估计，但可以保证精度
    int segments = (int)(radius * fabsf(angle_diff) / 2.0f);
    if (segments < 4) segments = 4;
    if (segments > 256) segments = 256;

    // 如果角度差很小，segments 可能为 0，防止除以 0
    if (segments == 0) return;

    // --- 优化核心：增量旋转 ---
    float delta_angle = angle_diff / (float)segments;

    // 仅计算一次三角函数
    const float cos_delta = cosf(delta_angle);
    const float sin_delta = sinf(delta_angle);

    // 初始点 (相对中心点的坐标)
    float current_rx = cosf(start_angle) * radius;
    float current_ry = sinf(start_angle) * radius;

    // 添加第一个点
    // 根据路径是否为空决定使用 move_to 还是 line_to
    uint8_t cmd = (current_path->point_count == 0) ? 0 : 1;
    wcn_path_add_point(current_path, x + current_rx, y + current_ry, cmd);

    // 循环添加剩余的点
    for (int i = 1; i <= segments; i++) {
        // 旋转矩阵应用于 (current_rx, current_ry)
        // new_x = r*cos(a+da) = r*cos(a)cos(da) - r*sin(a)sin(da)
        // new_y = r*sin(a+da) = r*sin(a)cos(da) + r*cos(a)sin(da)

        const float new_rx = current_rx * cos_delta - current_ry * sin_delta;
        const float new_ry = current_rx * sin_delta + current_ry * cos_delta;

        current_rx = new_rx;
        current_ry = new_ry;

        // 添加点 (始终使用 lineTo)
        wcn_path_add_point(current_path, x + current_rx, y + current_ry, 1);
    }
    // --- 增量旋转优化结束 ---

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
