#include "wcn_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================================
// GPU Native 路径管理
// ============================================================================

static WCN_GPUNativePath* wcn_gpu_path_create(void) {
    WCN_GPUNativePath* path = (WCN_GPUNativePath*)malloc(sizeof(WCN_GPUNativePath));
    if (!path) return NULL;
    
    memset(path, 0, sizeof(WCN_GPUNativePath));
    path->command_capacity = 64;
    path->commands = (WCN_PathCmd*)malloc(path->command_capacity * sizeof(WCN_PathCmd));
    if (!path->commands) {
        free(path);
        return NULL;
    }
    return path;
}

static void wcn_gpu_path_destroy(WCN_GPUNativePath* path) {
    if (!path) return;
    if (path->commands) free(path->commands);
    free(path);
}

static void wcn_gpu_path_clear(WCN_GPUNativePath* path) {
    if (!path) return;
    path->command_count = 0;
    path->current_x = 0.0f;
    path->current_y = 0.0f;
    path->start_x = 0.0f;
    path->start_y = 0.0f;
    path->is_closed = false;
}

static bool wcn_gpu_path_ensure_capacity(WCN_GPUNativePath* path) {
    if (path->command_count >= path->command_capacity) {
        size_t new_capacity = path->command_capacity * 2;
        WCN_PathCmd* new_cmds = (WCN_PathCmd*)realloc(path->commands, new_capacity * sizeof(WCN_PathCmd));
        if (!new_cmds) return false;
        path->commands = new_cmds;
        path->command_capacity = new_capacity;
    }
    return true;
}

static void wcn_gpu_path_add_cmd(WCN_GPUNativePath* path, uint8_t type, const float* params, int param_count) {
    if (!path || !wcn_gpu_path_ensure_capacity(path)) return;
    
    WCN_PathCmd* cmd = &path->commands[path->command_count++];
    cmd->type = type;
    for (int i = 0; i < param_count && i < 6; i++) {
        cmd->params[i] = params[i];
    }
}

// ============================================================================
// Line Join 辅助函数
// ============================================================================

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
    if (!vertices || !indices) { free(vertices); free(indices); return; }

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

    const uint16_t indices[3] = {0, 1, 2};
    wcn_renderer_add_triangles(renderer, vertices, 3, indices, 3, color, transform);
}

static void add_miter_join(WCN_Renderer* renderer,
                           float x, float y,
                           float perp1_x, float perp1_y,
                           float perp2_x, float perp2_y,
                           float half_width,
                           uint32_t color,
                           const float transform[4],
                           float miter_limit) {
    const float p1_x = x + perp1_x * half_width;
    const float p1_y = y + perp1_y * half_width;
    const float p2_x = x + perp2_x * half_width;
    const float p2_y = y + perp2_y * half_width;

    const float denom = perp1_x * perp2_y - perp1_y * perp2_x;
    if (fabsf(denom) < 0.001f) {
        add_bevel_join(renderer, x, y, perp1_x, perp1_y, perp2_x, perp2_y, half_width, color, transform);
        return;
    }

    const float dx = p2_x - p1_x;
    const float dy = p2_y - p1_y;
    const float t1 = (dx * perp2_y - dy * perp2_x) / denom;

    const float miter_x = p1_x + perp1_x * t1;
    const float miter_y = p1_y + perp1_y * t1;

    const float miter_dx = miter_x - x;
    const float miter_dy = miter_y - y;
    const float miter_length = sqrtf(miter_dx * miter_dx + miter_dy * miter_dy);

    if (miter_length > half_width * miter_limit) {
        add_bevel_join(renderer, x, y, perp1_x, perp1_y, perp2_x, perp2_y, half_width, color, transform);
        return;
    }

    WCN_SimpleVertex vertices[4];
    vertices[0].position[0] = x;
    vertices[0].position[1] = y;
    vertices[1].position[0] = p1_x;
    vertices[1].position[1] = p1_y;
    vertices[2].position[0] = miter_x;
    vertices[2].position[1] = miter_y;
    vertices[3].position[0] = p2_x;
    vertices[3].position[1] = p2_y;

    const uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
    wcn_renderer_add_triangles(renderer, vertices, 4, indices, 6, color, transform);
}

// ============================================================================
// 路径操作私有函数 (保留用于 fill 的点数组)
// ============================================================================

static WCN_Path* wcn_get_current_path(WCN_Context* ctx) {
    if (!ctx) return NULL;
    return ctx->current_path;
}

static void wcn_create_new_path(WCN_Context* ctx) {
    if (!ctx) return;

    // 释放现有的点数组路径
    if (ctx->current_path) {
        if (ctx->current_path->points) free(ctx->current_path->points);
        if (ctx->current_path->commands) free(ctx->current_path->commands);
        free(ctx->current_path);
    }

    ctx->current_path = (WCN_Path*)malloc(sizeof(WCN_Path));
    if (ctx->current_path) {
        memset(ctx->current_path, 0, sizeof(WCN_Path));
        ctx->current_path->is_closed = false;
    }
    
    // 创建或清空 GPU Native 路径
    if (!ctx->gpu_path) {
        ctx->gpu_path = wcn_gpu_path_create();
    } else {
        wcn_gpu_path_clear(ctx->gpu_path);
    }
}

static void wcn_path_add_point(WCN_Path* path, float x, float y, uint8_t command) {
    if (!path) return;

    const size_t MAX_PATH_POINTS = 10000;
    const size_t MAX_PATH_COMMANDS = 5000;

    if (path->point_count >= MAX_PATH_POINTS * 2 || path->command_count >= MAX_PATH_COMMANDS) {
        return;
    }

    const size_t new_point_count = path->point_count + 2;
    float* new_points = (float*)realloc(path->points, new_point_count * sizeof(float));
    if (!new_points) return;

    path->points = new_points;
    path->points[path->point_count] = x;
    path->points[path->point_count + 1] = y;
    path->point_count = new_point_count;

    const size_t new_command_count = path->command_count + 1;
    uint8_t* new_commands = (uint8_t*)realloc(path->commands, new_command_count * sizeof(uint8_t));
    if (!new_commands) {
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

// ============================================================================
// Fill 渲染 (使用点数组扇形三角化)
// ============================================================================

static void wcn_render_path_fill(WCN_Context* ctx, WCN_Path* path) {
    if (!ctx || !path || path->point_count < 6 || !ctx->renderer) return;

    const WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];
    const uint32_t color = state->fill_color;
    const size_t num_points = path->point_count / 2;

    WCN_SimpleVertex* vertices = (WCN_SimpleVertex*)malloc(num_points * sizeof(WCN_SimpleVertex));
    if (!vertices) return;

    for (size_t i = 0; i < num_points; i++) {
        float x = path->points[i*2];
        float y = path->points[i*2+1];
        vertices[i].position[0] = x * state->transform_matrix[0] + y * state->transform_matrix[4] + state->transform_matrix[12];
        vertices[i].position[1] = x * state->transform_matrix[1] + y * state->transform_matrix[5] + state->transform_matrix[13];
    }

    const size_t num_triangles = num_points - 2;
    const size_t num_indices = num_triangles * 3;
    uint16_t* indices = (uint16_t*)malloc(num_indices * sizeof(uint16_t));
    if (!indices) { free(vertices); return; }

    for (size_t i = 0; i < num_triangles; i++) {
        indices[i*3] = 0;
        indices[i*3+1] = i+1;
        indices[i*3+2] = i+2;
    }

    const float transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };

    wcn_renderer_add_triangles(ctx->renderer, vertices, num_points, indices, num_indices, color, transform);
    free(vertices);
    free(indices);
}

// ============================================================================
// Stroke 渲染 (GPU Native - 使用 SDF 实例)
// ============================================================================

static void wcn_render_path_stroke_gpu(WCN_Context* ctx, WCN_GPUNativePath* path) {
    if (!ctx || !path || path->command_count == 0 || !ctx->renderer) return;

    const WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];
    const uint32_t color = state->stroke_color;
    const float line_width = state->stroke_width;
    const uint32_t line_join = state->line_join;
    const uint32_t line_cap = state->line_cap;
    const float miter_limit = state->miter_limit;
    const float half_width = line_width * 0.5f;

    const float transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };
    
    // 变换辅助宏
    #define TRANSFORM_X(px, py) ((px) * state->transform_matrix[0] + (py) * state->transform_matrix[4] + state->transform_matrix[12])
    #define TRANSFORM_Y(px, py) ((px) * state->transform_matrix[1] + (py) * state->transform_matrix[5] + state->transform_matrix[13])

    float cur_x = 0.0f, cur_y = 0.0f;
    float prev_end_x = 0.0f, prev_end_y = 0.0f;
    float prev_dx = 0.0f, prev_dy = 0.0f;
    bool has_prev_segment = false;
    
    for (size_t i = 0; i < path->command_count; i++) {
        WCN_PathCmd* cmd = &path->commands[i];
        
        switch (cmd->type) {
            case WCN_CMD_MOVE_TO: {
                cur_x = cmd->params[0];
                cur_y = cmd->params[1];
                has_prev_segment = false;
                break;
            }
            
            case WCN_CMD_LINE_TO: {
                float x = cmd->params[0];
                float y = cmd->params[1];
                
                float tx1 = TRANSFORM_X(cur_x, cur_y);
                float ty1 = TRANSFORM_Y(cur_x, cur_y);
                float tx2 = TRANSFORM_X(x, y);
                float ty2 = TRANSFORM_Y(x, y);
                
                // 计算方向
                float dx = tx2 - tx1;
                float dy = ty2 - ty1;
                float len = sqrtf(dx * dx + dy * dy);
                
                if (len > 0.001f) {
                    dx /= len;
                    dy /= len;
                    
                    // Line Join
                    if (has_prev_segment) {
                        float perp1_x = -prev_dy;
                        float perp1_y = prev_dx;
                        float perp2_x = -dy;
                        float perp2_y = dx;
                        
                        switch (line_join) {
                            case 0: add_miter_join(ctx->renderer, prev_end_x, prev_end_y, perp1_x, perp1_y, perp2_x, perp2_y, half_width, color, transform, miter_limit); break;
                            case 1: add_round_join(ctx->renderer, prev_end_x, prev_end_y, perp1_x, perp1_y, perp2_x, perp2_y, half_width, color, transform); break;
                            case 2: add_bevel_join(ctx->renderer, prev_end_x, prev_end_y, perp1_x, perp1_y, perp2_x, perp2_y, half_width, color, transform); break;
                        }
                    }
                    
                    // 确定 cap flags
                    bool is_first = (i == 0 || path->commands[i-1].type == WCN_CMD_MOVE_TO);
                    bool is_last = (i == path->command_count - 1 || (i + 1 < path->command_count && path->commands[i+1].type == WCN_CMD_MOVE_TO));
                    uint32_t cap_flags = line_cap & 0xFF;
                    if (is_first && !path->is_closed) cap_flags |= (1u << 8);
                    if (is_last && !path->is_closed) cap_flags |= (1u << 9);
                    
                    wcn_renderer_add_line(ctx->renderer, tx1, ty1, tx2, ty2, line_width, color, transform, cap_flags);
                    
                    prev_end_x = tx2;
                    prev_end_y = ty2;
                    prev_dx = dx;
                    prev_dy = dy;
                    has_prev_segment = true;
                }
                
                cur_x = x;
                cur_y = y;
                break;
            }
            
            case WCN_CMD_ARC: {
                // GPU SDF 圆弧渲染
                float cx = cmd->params[0];
                float cy = cmd->params[1];
                float radius = cmd->params[2];
                float start_angle = cmd->params[3];
                float end_angle = cmd->params[4];
                
                // 计算圆弧起点
                float arc_start_x = cx + radius * cosf(start_angle);
                float arc_start_y = cy + radius * sinf(start_angle);
                
                // 如果当前点与圆弧起点不同，先画一条连接线
                if (fabsf(cur_x - arc_start_x) > 0.001f || fabsf(cur_y - arc_start_y) > 0.001f) {
                    float tx1 = TRANSFORM_X(cur_x, cur_y);
                    float ty1 = TRANSFORM_Y(cur_x, cur_y);
                    float tx2 = TRANSFORM_X(arc_start_x, arc_start_y);
                    float ty2 = TRANSFORM_Y(arc_start_x, arc_start_y);
                    
                    wcn_renderer_add_line(ctx->renderer, tx1, ty1, tx2, ty2, line_width, color, transform, line_cap & 0xFF);
                }
                
                float tcx = TRANSFORM_X(cx, cy);
                float tcy = TRANSFORM_Y(cx, cy);
                
                wcn_renderer_add_arc(ctx->renderer, tcx, tcy, radius, start_angle, end_angle, line_width, color, transform, 0);
                
                // 更新当前点为圆弧终点
                cur_x = cx + radius * cosf(end_angle);
                cur_y = cy + radius * sinf(end_angle);
                has_prev_segment = false; // 圆弧后重置 join 状态
                break;
            }
            
            case WCN_CMD_QUAD_TO: {
                // GPU SDF 二次贝塞尔渲染
                float cpx = cmd->params[0];
                float cpy = cmd->params[1];
                float x = cmd->params[2];
                float y = cmd->params[3];
                
                float tx0 = TRANSFORM_X(cur_x, cur_y);
                float ty0 = TRANSFORM_Y(cur_x, cur_y);
                float tcpx = TRANSFORM_X(cpx, cpy);
                float tcpy = TRANSFORM_Y(cpx, cpy);
                float tx1 = TRANSFORM_X(x, y);
                float ty1 = TRANSFORM_Y(x, y);
                
                wcn_renderer_add_quadratic_bezier(ctx->renderer, tx0, ty0, tcpx, tcpy, tx1, ty1, line_width, color, transform, 0);
                
                cur_x = x;
                cur_y = y;
                has_prev_segment = false;
                break;
            }
            
            case WCN_CMD_CUBIC_TO: {
                // GPU SDF 三次贝塞尔渲染 (分解为两个二次)
                float cp1x = cmd->params[0];
                float cp1y = cmd->params[1];
                float cp2x = cmd->params[2];
                float cp2y = cmd->params[3];
                float x = cmd->params[4];
                float y = cmd->params[5];
                
                float tx0 = TRANSFORM_X(cur_x, cur_y);
                float ty0 = TRANSFORM_Y(cur_x, cur_y);
                float tcp1x = TRANSFORM_X(cp1x, cp1y);
                float tcp1y = TRANSFORM_Y(cp1x, cp1y);
                float tcp2x = TRANSFORM_X(cp2x, cp2y);
                float tcp2y = TRANSFORM_Y(cp2x, cp2y);
                float tx1 = TRANSFORM_X(x, y);
                float ty1 = TRANSFORM_Y(x, y);
                
                wcn_renderer_add_cubic_bezier(ctx->renderer, tx0, ty0, tcp1x, tcp1y, tcp2x, tcp2y, tx1, ty1, line_width, color, transform, 0);
                
                cur_x = x;
                cur_y = y;
                has_prev_segment = false;
                break;
            }
            
            case WCN_CMD_CLOSE: {
                // 闭合路径：画线回到起点
                if (fabsf(cur_x - path->start_x) > 0.001f || fabsf(cur_y - path->start_y) > 0.001f) {
                    float tx1 = TRANSFORM_X(cur_x, cur_y);
                    float ty1 = TRANSFORM_Y(cur_x, cur_y);
                    float tx2 = TRANSFORM_X(path->start_x, path->start_y);
                    float ty2 = TRANSFORM_Y(path->start_x, path->start_y);
                    
                    wcn_renderer_add_line(ctx->renderer, tx1, ty1, tx2, ty2, line_width, color, transform, line_cap & 0xFF);
                }
                cur_x = path->start_x;
                cur_y = path->start_y;
                has_prev_segment = false;
                break;
            }
        }
    }
    
    #undef TRANSFORM_X
    #undef TRANSFORM_Y
}

// ============================================================================
// 公共 API 实现
// ============================================================================

void wcn_begin_path(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;
    wcn_create_new_path(ctx);
}

void wcn_close_path(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* path = wcn_get_current_path(ctx);
    if (path) wcn_path_close(path);
    
    if (ctx->gpu_path) {
        ctx->gpu_path->is_closed = true;
        wcn_gpu_path_add_cmd(ctx->gpu_path, WCN_CMD_CLOSE, NULL, 0);
    }
}

void wcn_move_to(WCN_Context* ctx, const float x, const float y) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* path = wcn_get_current_path(ctx);
    if (path) wcn_path_add_point(path, x, y, 0);
    
    if (ctx->gpu_path) {
        float params[2] = {x, y};
        wcn_gpu_path_add_cmd(ctx->gpu_path, WCN_CMD_MOVE_TO, params, 2);
        ctx->gpu_path->current_x = x;
        ctx->gpu_path->current_y = y;
        ctx->gpu_path->start_x = x;
        ctx->gpu_path->start_y = y;
    }
}

void wcn_line_to(WCN_Context* ctx, const float x, const float y) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* path = wcn_get_current_path(ctx);
    if (path) wcn_path_add_point(path, x, y, 1);
    
    if (ctx->gpu_path) {
        float params[2] = {x, y};
        wcn_gpu_path_add_cmd(ctx->gpu_path, WCN_CMD_LINE_TO, params, 2);
        ctx->gpu_path->current_x = x;
        ctx->gpu_path->current_y = y;
    }
}


void wcn_arc(WCN_Context* ctx, const float x, const float y, const float radius,
             const float start_angle, const float end_angle, const bool anticlockwise) {
    if (!ctx || !ctx->in_frame || radius <= 0.0f) return;

    WCN_Path* path = wcn_get_current_path(ctx);
    if (!path) return;

    float angle_diff = end_angle - start_angle;
    if (anticlockwise) {
        while (angle_diff >= 0.0f) angle_diff -= 2.0f * M_PI;
    } else {
        while (angle_diff <= 0.0f) angle_diff += 2.0f * M_PI;
    }

    // 为 fill 生成点数组 (保留原有逻辑)
    int segments = (int)(radius * fabsf(angle_diff) / 2.0f);
    if (segments < 4) segments = 4;
    if (segments > 256) segments = 256;

    float delta_angle = angle_diff / (float)segments;
    const float cos_delta = cosf(delta_angle);
    const float sin_delta = sinf(delta_angle);

    float current_rx = cosf(start_angle) * radius;
    float current_ry = sinf(start_angle) * radius;

    uint8_t cmd = (path->point_count == 0) ? 0 : 1;
    wcn_path_add_point(path, x + current_rx, y + current_ry, cmd);

    for (int i = 1; i <= segments; i++) {
        const float new_rx = current_rx * cos_delta - current_ry * sin_delta;
        const float new_ry = current_rx * sin_delta + current_ry * cos_delta;
        current_rx = new_rx;
        current_ry = new_ry;
        wcn_path_add_point(path, x + current_rx, y + current_ry, 1);
    }

    if (!path->is_closed && fabsf(angle_diff) >= 2.0f * M_PI - 0.001f) {
        wcn_path_close(path);
    }
    
    // 为 stroke 添加 GPU 命令
    if (ctx->gpu_path) {
        // 先添加 line_to 到圆弧起点 (如果需要)
        float arc_start_x = x + radius * cosf(start_angle);
        float arc_start_y = y + radius * sinf(start_angle);
        
        if (ctx->gpu_path->command_count > 0) {
            float params_line[2] = {arc_start_x, arc_start_y};
            wcn_gpu_path_add_cmd(ctx->gpu_path, WCN_CMD_LINE_TO, params_line, 2);
        } else {
            float params_move[2] = {arc_start_x, arc_start_y};
            wcn_gpu_path_add_cmd(ctx->gpu_path, WCN_CMD_MOVE_TO, params_move, 2);
            ctx->gpu_path->start_x = arc_start_x;
            ctx->gpu_path->start_y = arc_start_y;
        }
        
        // 添加圆弧命令
        float params[5] = {x, y, radius, start_angle, end_angle};
        wcn_gpu_path_add_cmd(ctx->gpu_path, WCN_CMD_ARC, params, 5);
        
        ctx->gpu_path->current_x = x + radius * cosf(end_angle);
        ctx->gpu_path->current_y = y + radius * sinf(end_angle);
    }
}

void wcn_rect(WCN_Context* ctx, float x, float y, float width, float height) {
    if (!ctx || !ctx->in_frame) return;

    wcn_move_to(ctx, x, y);
    wcn_line_to(ctx, x + width, y);
    wcn_line_to(ctx, x + width, y + height);
    wcn_line_to(ctx, x, y + height);
    wcn_close_path(ctx);
}

// 检测 GPU 路径是否为简单圆形/弧形 (只有一个 ARC 命令)
static bool wcn_is_simple_circle_path(WCN_GPUNativePath* gpu_path, 
                                       float* out_cx, float* out_cy, 
                                       float* out_radius,
                                       float* out_start_angle, float* out_end_angle) {
    if (!gpu_path || gpu_path->command_count == 0) return false;
    
    // 查找 ARC 命令
    int arc_count = 0;
    int other_count = 0;
    WCN_PathCmd* arc_cmd = NULL;
    
    for (size_t i = 0; i < gpu_path->command_count; i++) {
        WCN_PathCmd* cmd = &gpu_path->commands[i];
        if (cmd->type == WCN_CMD_ARC) {
            arc_count++;
            arc_cmd = cmd;
        } else if (cmd->type == WCN_CMD_MOVE_TO) {
            // MOVE_TO 可以忽略
        } else if (cmd->type == WCN_CMD_LINE_TO) {
            // LINE_TO 到圆弧起点可以忽略，其他情况不是简单圆
            other_count++;
        } else if (cmd->type == WCN_CMD_CLOSE) {
            // CLOSE 可以忽略
        } else {
            // 其他命令（贝塞尔等）不是简单圆
            return false;
        }
    }
    
    // 只有一个 ARC 命令，且没有其他复杂命令
    if (arc_count == 1 && arc_cmd && other_count <= 1) {
        *out_cx = arc_cmd->params[0];
        *out_cy = arc_cmd->params[1];
        *out_radius = arc_cmd->params[2];
        *out_start_angle = arc_cmd->params[3];
        *out_end_angle = arc_cmd->params[4];
        return true;
    }
    
    return false;
}

void wcn_fill(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;

    // 检测是否为简单圆形/弧形，使用 GPU SDF 渲染
    float cx, cy, radius, start_angle, end_angle;
    if (ctx->gpu_path && wcn_is_simple_circle_path(ctx->gpu_path, &cx, &cy, &radius, &start_angle, &end_angle)) {
        const WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];
        const uint32_t color = state->fill_color;
        const float transform[4] = {
            state->transform_matrix[0], state->transform_matrix[1],
            state->transform_matrix[4], state->transform_matrix[5]
        };
        
        // 变换圆心
        float tcx = cx * state->transform_matrix[0] + cy * state->transform_matrix[4] + state->transform_matrix[12];
        float tcy = cx * state->transform_matrix[1] + cy * state->transform_matrix[5] + state->transform_matrix[13];
        
        wcn_renderer_add_circle_fill(ctx->renderer, tcx, tcy, radius, start_angle, end_angle, color, transform);
        return;
    }

    // 非圆形路径，使用传统三角化
    WCN_Path* path = wcn_get_current_path(ctx);
    if (path) {
        wcn_render_path_fill(ctx, path);
    }
}

void wcn_stroke(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;

    // 使用 GPU Native 路径渲染
    if (ctx->gpu_path && ctx->gpu_path->command_count > 0) {
        wcn_render_path_stroke_gpu(ctx, ctx->gpu_path);
    }
}

// ============================================================================
// 贝塞尔曲线 API
// ============================================================================

static bool wcn_get_last_path_point(WCN_Context* ctx, float* out_x, float* out_y) {
    if (!ctx || !ctx->current_path || ctx->current_path->point_count < 2) {
        return false;
    }
    size_t last_idx = ctx->current_path->point_count - 2;
    *out_x = ctx->current_path->points[last_idx];
    *out_y = ctx->current_path->points[last_idx + 1];
    return true;
}

void wcn_quadratic_curve_to(WCN_Context* ctx, float cpx, float cpy, float x, float y) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* path = wcn_get_current_path(ctx);
    if (!path) return;

    float x0, y0;
    if (!wcn_get_last_path_point(ctx, &x0, &y0)) {
        x0 = cpx;
        y0 = cpy;
    }

    // 为 fill 生成点数组
    const int segments = 16;
    for (int i = 1; i <= segments; i++) {
        float t = (float)i / (float)segments;
        float mt = 1.0f - t;
        float px = mt * mt * x0 + 2.0f * mt * t * cpx + t * t * x;
        float py = mt * mt * y0 + 2.0f * mt * t * cpy + t * t * y;
        wcn_path_add_point(path, px, py, 1);
    }
    
    // 为 stroke 添加 GPU 命令
    if (ctx->gpu_path) {
        float params[4] = {cpx, cpy, x, y};
        wcn_gpu_path_add_cmd(ctx->gpu_path, WCN_CMD_QUAD_TO, params, 4);
        ctx->gpu_path->current_x = x;
        ctx->gpu_path->current_y = y;
    }
}

void wcn_bezier_curve_to(WCN_Context* ctx, float cp1x, float cp1y, float cp2x, float cp2y, float x, float y) {
    if (!ctx || !ctx->in_frame) return;

    WCN_Path* path = wcn_get_current_path(ctx);
    if (!path) return;

    float x0, y0;
    if (!wcn_get_last_path_point(ctx, &x0, &y0)) {
        x0 = cp1x;
        y0 = cp1y;
    }

    // 为 fill 生成点数组
    const int segments = 24;
    for (int i = 1; i <= segments; i++) {
        float t = (float)i / (float)segments;
        float mt = 1.0f - t;
        float px = mt * mt * mt * x0 + 3.0f * mt * mt * t * cp1x + 3.0f * mt * t * t * cp2x + t * t * t * x;
        float py = mt * mt * mt * y0 + 3.0f * mt * mt * t * cp1y + 3.0f * mt * t * t * cp2y + t * t * t * y;
        wcn_path_add_point(path, px, py, 1);
    }
    
    // 为 stroke 添加 GPU 命令
    if (ctx->gpu_path) {
        float params[6] = {cp1x, cp1y, cp2x, cp2y, x, y};
        wcn_gpu_path_add_cmd(ctx->gpu_path, WCN_CMD_CUBIC_TO, params, 6);
        ctx->gpu_path->current_x = x;
        ctx->gpu_path->current_y = y;
    }
}
