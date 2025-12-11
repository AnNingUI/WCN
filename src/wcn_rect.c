#include "wcn_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// 矩形绘制私有函数（使用合批渲染）
// ============================================================================

// ============================================================================
// 公共API实现
// ============================================================================

void wcn_clear_rect(WCN_Context* ctx, float x, float y, float width, float height) {
    if (!ctx || !ctx->in_frame) return;

    // 对于清除操作，使用黑色填充（透明背景）
    uint32_t original_fill_color = ctx->state_stack.states[ctx->state_stack.current_state].fill_color;
    ctx->state_stack.states[ctx->state_stack.current_state].fill_color = 0xFF000000; // 黑色

    // 使用填充矩形来清除区域
    wcn_fill_rect(ctx, x, y, width, height);

    // 恢复原始填充颜色
    ctx->state_stack.states[ctx->state_stack.current_state].fill_color = original_fill_color;
}

void wcn_fill_rect(WCN_Context* ctx, float x, float y, float width, float height) {
    if (!ctx || !ctx->in_frame || !ctx->renderer) return;

    WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];

    // 提取 2x2 变换矩阵（从 4x4 矩阵的左上角）- 用于实例变换
    float instance_transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };

    // 计算应用变换后的矩形位置
    // 将原始位置 (x, y) 应用变换矩阵的平移分量
    float transformed_x = x * state->transform_matrix[0] + y * state->transform_matrix[4] + state->transform_matrix[12];
    float transformed_y = x * state->transform_matrix[1] + y * state->transform_matrix[5] + state->transform_matrix[13];

    // 添加矩形实例到统一渲染器
    wcn_renderer_add_rect(
        ctx->renderer,
        transformed_x, transformed_y, width, height,
        0.0f, // radius
        state->fill_color,
        instance_transform
    );
}

void wcn_stroke_rect(WCN_Context* ctx, float x, float y, float width, float height) {
    // TODO: 实现描边矩形的合批渲染
    if (!ctx || !ctx->in_frame) return;
}