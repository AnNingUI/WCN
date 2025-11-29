#include "wcn_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

// ============================================================================
// UTF-8 解码
// ============================================================================

uint32_t wcn_decode_utf8(const char** ptr) {
    const uint8_t* bytes = (const uint8_t*)*ptr;

    if ((bytes[0] & 0x80) == 0) {
        *ptr += 1;
        return bytes[0];
    } else if ((bytes[0] & 0xE0) == 0xC0) {
        uint32_t codepoint = ((bytes[0] & 0x1F) << 6) | (bytes[1] & 0x3F);
        *ptr += 2;
        return codepoint;
    } else if ((bytes[0] & 0xF0) == 0xE0) {
        uint32_t codepoint = ((bytes[0] & 0x0F) << 12) | ((bytes[1] & 0x3F) << 6) | (bytes[2] & 0x3F);
        *ptr += 3;
        return codepoint;
    } else if ((bytes[0] & 0xF8) == 0xF0) {
        uint32_t codepoint = ((bytes[0] & 0x07) << 18) | ((bytes[1] & 0x3F) << 12) |
                            ((bytes[2] & 0x3F) << 6) | (bytes[3] & 0x3F);
        *ptr += 4;
        return codepoint;
    } else {
        *ptr += 1;
        return 0xFFFD; // Replacement character for invalid UTF-8
    }
}

// ============================================================================
// 文本渲染实现 - 使用 MSDF Atlas 合批渲染
// ============================================================================

void wcn_fill_text(WCN_Context* ctx, const char* text, float x, float y) {
    if (!ctx || !text || !ctx->in_frame || !ctx->renderer) return;

    // 检查是否有字体解码器和 MSDF 支持
    if (!ctx->font_decoder || !ctx->font_decoder->get_glyph_sdf) {
        printf("警告: 没有字体解码器或不支持 MSDF\n");
        return;
    }

    if (!ctx->current_font_face) {
        printf("警告: 没有当前字体\n");
        return;
    }

    // 计算对齐偏移
    float total_width = 0;
    if (ctx->text_align != WCN_TEXT_ALIGN_LEFT) {
        WCN_TextMetrics metrics = wcn_measure_text(ctx, text);
        total_width = metrics.width;
    }

    // 计算字体缩放比例
    float font_scale = ctx->current_font_size / ctx->current_font_face->units_per_em;
    
    // 基线偏移计算（注意：stb_truetype 的坐标系）
    // ascent 是正值（基线上方），descent 是负值（基线下方）
    float baseline_offset = 0;
    switch (ctx->text_baseline) {
        case WCN_TEXT_BASELINE_TOP:
            // 从顶部对齐：向下偏移 ascent
            baseline_offset = ctx->current_font_face->ascent * font_scale;
            break;
        case WCN_TEXT_BASELINE_MIDDLE:
            // 从中间对齐：向下偏移 (ascent + descent) / 2
            baseline_offset = (ctx->current_font_face->ascent + ctx->current_font_face->descent) * 0.5f * font_scale;
            break;
        case WCN_TEXT_BASELINE_BOTTOM:
            // 从底部对齐：向下偏移 descent（通常是负值）
            baseline_offset = ctx->current_font_face->descent * font_scale;
            break;
        case WCN_TEXT_BASELINE_ALPHABETIC:
        default:
            // 字母基线：不偏移
            baseline_offset = 0;
            break;
    }

    float align_offset = 0;
    switch (ctx->text_align) {
        case WCN_TEXT_ALIGN_CENTER:
            align_offset = -total_width * 0.5f;
            break;
        case WCN_TEXT_ALIGN_RIGHT:
            align_offset = -total_width;
            break;
        case WCN_TEXT_ALIGN_LEFT:
        default:
            align_offset = 0;
            break;
    }
    
    float start_x = x + align_offset;
    float start_y = y + baseline_offset;

    WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];
    
    // 提取 2x2 变换矩阵（从 4x4 矩阵的左上角）
    float transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };

    // 添加文本实例到统一渲染器
    wcn_renderer_add_text(
        ctx->renderer,
        ctx,
        text,
        start_x, start_y,
        ctx->current_font_size,
        state->fill_color,
        transform
    );
}

void wcn_stroke_text(WCN_Context* ctx, const char* text, float x, float y) {
    if (!ctx || !text || !ctx->in_frame || !ctx->renderer) return;

    // 检查是否有字体解码器和 MSDF 支持
    if (!ctx->font_decoder || !ctx->font_decoder->get_glyph_sdf) {
        printf("警告: 没有字体解码器或不支持 MSDF\n");
        return;
    }

    if (!ctx->current_font_face) {
        printf("警告: 没有当前字体\n");
        return;
    }

    // 计算对齐偏移
    float total_width = 0;
    if (ctx->text_align != WCN_TEXT_ALIGN_LEFT) {
        WCN_TextMetrics metrics = wcn_measure_text(ctx, text);
        total_width = metrics.width;
    }

    // 计算字体缩放比例
    float font_scale = ctx->current_font_size / ctx->current_font_face->units_per_em;
    
    // 基线偏移计算（注意：stb_truetype 的坐标系）
    // ascent 是正值（基线上方），descent 是负值（基线下方）
    float baseline_offset = 0;
    switch (ctx->text_baseline) {
        case WCN_TEXT_BASELINE_TOP:
            // 从顶部对齐：向下偏移 ascent
            baseline_offset = ctx->current_font_face->ascent * font_scale;
            break;
        case WCN_TEXT_BASELINE_MIDDLE:
            // 从中间对齐：向下偏移 (ascent + descent) / 2
            baseline_offset = (ctx->current_font_face->ascent + ctx->current_font_face->descent) * 0.5f * font_scale;
            break;
        case WCN_TEXT_BASELINE_BOTTOM:
            // 从底部对齐：向下偏移 descent（通常是负值）
            baseline_offset = ctx->current_font_face->descent * font_scale;
            break;
        case WCN_TEXT_BASELINE_ALPHABETIC:
        default:
            // 字母基线：不偏移
            baseline_offset = 0;
            break;
    }

    float align_offset = 0;
    switch (ctx->text_align) {
        case WCN_TEXT_ALIGN_CENTER:
            align_offset = -total_width * 0.5f;
            break;
        case WCN_TEXT_ALIGN_RIGHT:
            align_offset = -total_width;
            break;
        case WCN_TEXT_ALIGN_LEFT:
        default:
            align_offset = 0;
            break;
    }
    
    float start_x = x + align_offset;
    float start_y = y + baseline_offset;

    WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];
    
    // 提取 2x2 变换矩阵（从 4x4 矩阵的左上角）
    float transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };

    // 添加文本实例到统一渲染器（使用描边颜色）
    wcn_renderer_add_text(
        ctx->renderer,
        ctx,
        text,
        start_x, start_y,
        ctx->current_font_size,
        state->stroke_color,
        transform
    );
}

// ============================================================================
// 文本测量
// ============================================================================

WCN_TextMetrics wcn_measure_text(WCN_Context* ctx, const char* text) {
    WCN_TextMetrics metrics = {0};

    if (!ctx || !text || !ctx->font_decoder || !ctx->current_font_face) {
        return metrics;
    }

    if (ctx->font_decoder->measure_text) {
        float width, height;
        if (ctx->font_decoder->measure_text(ctx->current_font_face, text,
                                           ctx->current_font_size, &width, &height)) {
            metrics.width = width;
            return metrics;
        }
    }

    // 手动计算
    float scale = ctx->current_font_size / ctx->current_font_face->units_per_em;
    const char* ptr = text;
    float x_offset = 0;

    while (*ptr) {
        const uint8_t* bytes = (const uint8_t*)ptr;
        uint32_t codepoint = 0;

        if ((bytes[0] & 0x80) == 0) {
            codepoint = bytes[0];
            ptr += 1;
        } else if ((bytes[0] & 0xE0) == 0xC0) {
            codepoint = ((bytes[0] & 0x1F) << 6) | (bytes[1] & 0x3F);
            ptr += 2;
        } else if ((bytes[0] & 0xF0) == 0xE0) {
            codepoint = ((bytes[0] & 0x0F) << 12) | ((bytes[1] & 0x3F) << 6) | (bytes[2] & 0x3F);
            ptr += 3;
        } else if ((bytes[0] & 0xF8) == 0xF0) {
            codepoint = ((bytes[0] & 0x07) << 18) | ((bytes[1] & 0x3F) << 12) |
                        ((bytes[2] & 0x3F) << 6) | (bytes[3] & 0x3F);
            ptr += 4;
        } else {
            ptr += 1;
            continue;
        }

        if (codepoint == 0) break;

        WCN_Glyph* glyph = NULL;
        if (ctx->font_decoder->get_glyph(ctx->current_font_face, codepoint, &glyph)) {
            x_offset += glyph->advance_width * scale;

            if (ctx->font_decoder->free_glyph) {
                ctx->font_decoder->free_glyph(glyph);
            }
        }
    }

    metrics.width = x_offset;
    metrics.em_height_ascent = ctx->current_font_face->ascent * scale;
    metrics.em_height_descent = ctx->current_font_face->descent * scale;

    return metrics;
}

// ============================================================================
// 字体设置 API
// ============================================================================

void wcn_set_font(WCN_Context* ctx, const char* font_spec) {
    if (!ctx || !font_spec) return;

    // 解析字体规范（如 "48px Arial"）
    float font_size = 16.0f;
    const char* ptr = font_spec;

    while (*ptr == ' ' || *ptr == '\t') ptr++;

    if (isdigit(*ptr)) {
        font_size = 0;
        while (isdigit(*ptr) || *ptr == '.') {
            if (*ptr == '.') {
                ptr++;
                float decimal = 0.1f;
                while (isdigit(*ptr)) {
                    font_size += (*ptr - '0') * decimal;
                    decimal *= 0.1f;
                    ptr++;
                }
                break;
            } else {
                font_size = font_size * 10 + (*ptr - '0');
                ptr++;
            }
        }

        while (*ptr && !isspace(*ptr)) ptr++;
    }

    ctx->current_font_size = font_size;
}

void wcn_set_font_face(WCN_Context* ctx, WCN_FontFace* face, float size) {
    if (!ctx) return;

    ctx->current_font_face = face;
    ctx->current_font_size = size;
}

void wcn_set_text_align(WCN_Context* ctx, WCN_TextAlign align) {
    if (!ctx) return;
    ctx->text_align = align;
}

void wcn_set_text_baseline(WCN_Context* ctx, WCN_TextBaseline baseline) {
    if (!ctx) return;
    ctx->text_baseline = baseline;
}
