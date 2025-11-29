#include "wcn_internal.h"
#include <string.h>

// ============================================================================
// 样式设置私有函数
// ============================================================================

static void wcn_update_current_state(WCN_Context* ctx) {
    // 确保当前状态有效
    if (ctx->state_stack.current_state >= ctx->state_stack.max_states) {
        ctx->state_stack.current_state = ctx->state_stack.max_states - 1;
    }
}

// ============================================================================
// 公共API实现
// ============================================================================

void wcn_set_fill_style(WCN_Context* ctx, uint32_t color) {
    if (!ctx) return;

    wcn_update_current_state(ctx);
    ctx->state_stack.states[ctx->state_stack.current_state].fill_color = color;
}

void wcn_set_stroke_style(WCN_Context* ctx, uint32_t color) {
    if (!ctx) return;

    wcn_update_current_state(ctx);
    ctx->state_stack.states[ctx->state_stack.current_state].stroke_color = color;
}

void wcn_set_line_width(WCN_Context* ctx, float width) {
    if (!ctx || width <= 0.0f) return;

    wcn_update_current_state(ctx);
    ctx->state_stack.states[ctx->state_stack.current_state].stroke_width = width;
}

void wcn_set_line_cap(WCN_Context* ctx, WCN_LineCap cap) {
    if (!ctx) return;

    wcn_update_current_state(ctx);
    ctx->state_stack.states[ctx->state_stack.current_state].line_cap = (uint32_t)cap;
}

void wcn_set_line_join(WCN_Context* ctx, WCN_LineJoin join) {
    if (!ctx) return;

    wcn_update_current_state(ctx);
    ctx->state_stack.states[ctx->state_stack.current_state].line_join = (uint32_t)join;
}

void wcn_set_miter_limit(WCN_Context* ctx, float limit) {
    if (!ctx || limit <= 0.0f) return;

    wcn_update_current_state(ctx);
    ctx->state_stack.states[ctx->state_stack.current_state].miter_limit = limit;
}

void wcn_set_global_alpha(WCN_Context* ctx, float alpha) {
    if (!ctx || alpha < 0.0f || alpha > 1.0f) return;

    wcn_update_current_state(ctx);
    ctx->state_stack.states[ctx->state_stack.current_state].global_alpha = alpha;
}

void wcn_set_global_composite_operation(WCN_Context* ctx, WCN_CompositeOperation operation) {
    if (!ctx) return;

    wcn_update_current_state(ctx);
    ctx->state_stack.states[ctx->state_stack.current_state].blend_mode = (uint32_t)operation;
}