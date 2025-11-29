#include "wcn_internal.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// 变换操作私有函数
// ============================================================================

static void wcn_matrix_multiply(float* result, const float* a, const float* b) {
    // 4x4 矩阵乘法（列主序）
    // 对于列主序矩阵，索引是 matrix[col * 4 + row]
    // result = a * b
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            result[col * 4 + row] = 0.0f;
            for (int k = 0; k < 4; k++) {
                result[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
            }
        }
    }
}

static void wcn_create_translation_matrix(float* matrix, float x, float y) {
    // WGSL使用列主序矩阵，所以我们需要按列主序存储
    // 列主序的平移矩阵：
    // [1  0  0  x]     存储为: [1 0 0 0] [0 1 0 0] [0 0 1 0] [x y 0 1]
    // [0  1  0  y]             col0      col1      col2      col3
    // [0  0  1  0]
    // [0  0  0  1]
    memset(matrix, 0, 16 * sizeof(float));
    matrix[0] = 1.0f;   // col0[0]
    matrix[5] = 1.0f;   // col1[1]
    matrix[10] = 1.0f;  // col2[2]
    matrix[12] = x;     // col3[0] - 平移x
    matrix[13] = y;     // col3[1] - 平移y
    matrix[15] = 1.0f;  // col3[3]
}

static void wcn_create_rotation_matrix(float* matrix, float angle) {
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // WGSL列主序旋转矩阵：
    // [cos  -sin  0  0]     存储为: [cos sin 0 0] [-sin cos 0 0] [0 0 1 0] [0 0 0 1]
    // [sin   cos  0  0]             col0          col1            col2      col3
    // [0     0    1  0]
    // [0     0    0  1]
    memset(matrix, 0, 16 * sizeof(float));
    matrix[0] = cos_a;   // col0[0]
    matrix[1] = sin_a;   // col0[1]
    matrix[4] = -sin_a;  // col1[0]
    matrix[5] = cos_a;   // col1[1]
    matrix[10] = 1.0f;   // col2[2]
    matrix[15] = 1.0f;   // col3[3]
}

static void wcn_create_scale_matrix(float* matrix, float x, float y) {
    memset(matrix, 0, 16 * sizeof(float));
    matrix[0] = x;
    matrix[5] = y;
    matrix[10] = 1.0f;
    matrix[15] = 1.0f;
}

static void wcn_create_transform_matrix(float* matrix, float a, float b, float c, float d, float e, float f) {
    // Canvas2D transform(a, b, c, d, e, f) 对应矩阵：
    // [a  c  e]     在3D中扩展为: [a  c  0  e]
    // [b  d  f]                   [b  d  0  f]
    // [0  0  1]                   [0  0  1  0]
    //                             [0  0  0  1]
    // WGSL列主序存储: [a b 0 0] [c d 0 0] [0 0 1 0] [e f 0 1]
    memset(matrix, 0, 16 * sizeof(float));
    matrix[0] = a;    // col0[0]
    matrix[1] = b;    // col0[1]
    matrix[4] = c;    // col1[0]
    matrix[5] = d;    // col1[1]
    matrix[10] = 1.0f; // col2[2]
    matrix[12] = e;   // col3[0]
    matrix[13] = f;   // col3[1]
    matrix[15] = 1.0f; // col3[3]
}

static void wcn_update_current_transform(WCN_Context* ctx) {
    if (!ctx) return;

    // 确保当前状态有效
    if (ctx->state_stack.current_state >= ctx->state_stack.max_states) {
        ctx->state_stack.current_state = ctx->state_stack.max_states - 1;
    }
}

// ============================================================================
// 公共API实现
// ============================================================================

void wcn_translate(WCN_Context* ctx, float x, float y) {
    if (!ctx) return;

    wcn_update_current_transform(ctx);

    float current_matrix[16];
    memcpy(current_matrix, ctx->state_stack.states[ctx->state_stack.current_state].transform_matrix, 16 * sizeof(float));

    float translation_matrix[16];
    wcn_create_translation_matrix(translation_matrix, x, y);

    float result_matrix[16];
    wcn_matrix_multiply(result_matrix, current_matrix, translation_matrix);

    memcpy(ctx->state_stack.states[ctx->state_stack.current_state].transform_matrix, result_matrix, 16 * sizeof(float));
}

void wcn_rotate(WCN_Context* ctx, float angle) {
    if (!ctx) return;

    wcn_update_current_transform(ctx);

    float current_matrix[16];
    memcpy(current_matrix, ctx->state_stack.states[ctx->state_stack.current_state].transform_matrix, 16 * sizeof(float));

    float rotation_matrix[16];
    wcn_create_rotation_matrix(rotation_matrix, angle);

    float result_matrix[16];
    wcn_matrix_multiply(result_matrix, current_matrix, rotation_matrix);

    memcpy(ctx->state_stack.states[ctx->state_stack.current_state].transform_matrix, result_matrix, 16 * sizeof(float));
}

void wcn_scale(WCN_Context* ctx, float x, float y) {
    if (!ctx || x == 0.0f || y == 0.0f) return;

    wcn_update_current_transform(ctx);

    float current_matrix[16];
    memcpy(current_matrix, ctx->state_stack.states[ctx->state_stack.current_state].transform_matrix, 16 * sizeof(float));

    float scale_matrix[16];
    wcn_create_scale_matrix(scale_matrix, x, y);

    float result_matrix[16];
    wcn_matrix_multiply(result_matrix, current_matrix, scale_matrix);

    memcpy(ctx->state_stack.states[ctx->state_stack.current_state].transform_matrix, result_matrix, 16 * sizeof(float));
}

void wcn_transform(WCN_Context* ctx, float a, float b, float c, float d, float e, float f) {
    if (!ctx) return;

    wcn_update_current_transform(ctx);

    float current_matrix[16];
    memcpy(current_matrix, ctx->state_stack.states[ctx->state_stack.current_state].transform_matrix, 16 * sizeof(float));

    float transform_matrix[16];
    wcn_create_transform_matrix(transform_matrix, a, b, c, d, e, f);

    float result_matrix[16];
    wcn_matrix_multiply(result_matrix, current_matrix, transform_matrix);

    memcpy(ctx->state_stack.states[ctx->state_stack.current_state].transform_matrix, result_matrix, 16 * sizeof(float));
}

void wcn_set_transform(WCN_Context* ctx, float a, float b, float c, float d, float e, float f) {
    if (!ctx) return;

    wcn_update_current_transform(ctx);

    float transform_matrix[16];
    wcn_create_transform_matrix(transform_matrix, a, b, c, d, e, f);

    memcpy(ctx->state_stack.states[ctx->state_stack.current_state].transform_matrix, transform_matrix, 16 * sizeof(float));
}

void wcn_reset_transform(WCN_Context* ctx) {
    if (!ctx) return;

    wcn_update_current_transform(ctx);

    // 设置为单位矩阵
    float identity_matrix[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    memcpy(ctx->state_stack.states[ctx->state_stack.current_state].transform_matrix, identity_matrix, 16 * sizeof(float));
}