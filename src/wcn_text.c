#include "wcn_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

static bool wcn_is_variation_selector(uint32_t cp) {
    return (cp >= 0xFE00 && cp <= 0xFE0F) || (cp >= 0xE0100 && cp <= 0xE01EF);
}

static bool wcn_is_zero_width_joiner(uint32_t cp) {
    return cp == 0x200D;
}

static WCN_FontFace* wcn_get_primary_font_face(WCN_Context* ctx) {
    if (!ctx) {
        return NULL;
    }
    if (ctx->current_font_face) {
        return ctx->current_font_face;
    }
    if (ctx->font_fallback_count > 0) {
        return ctx->font_fallbacks[0];
    }
    return NULL;
}

static bool wcn_attempt_get_glyph(WCN_Context* ctx,
                                  WCN_FontFace* face,
                                  uint32_t codepoint,
                                  WCN_Glyph** out_glyph) {
    if (!ctx || !face || !ctx->font_decoder || !ctx->font_decoder->get_glyph) {
        return false;
    }
    WCN_Glyph* glyph = NULL;
    if (!ctx->font_decoder->get_glyph(face, codepoint, &glyph)) {
        return false;
    }
    if (out_glyph) {
        *out_glyph = glyph;
    } else if (ctx->font_decoder->free_glyph && glyph) {
        ctx->font_decoder->free_glyph(glyph);
    }
    return true;
}

static bool wcn_get_glyph_with_fallback(WCN_Context* ctx,
                                        uint32_t codepoint,
                                        WCN_FontFace* preferred_face,
                                        WCN_FontFace** out_face,
                                        WCN_Glyph** out_glyph) {
    if (!ctx || !ctx->font_decoder || !ctx->font_decoder->get_glyph) {
        return false;
    }

    if (preferred_face) {
        if (wcn_attempt_get_glyph(ctx, preferred_face, codepoint, out_glyph)) {
            if (out_face) {
                *out_face = preferred_face;
            }
            return true;
        }
    }

    WCN_FontFace* primary_face = ctx->current_font_face;
    if (!primary_face && ctx->font_fallback_count > 0) {
        primary_face = ctx->font_fallbacks[0];
    }

    if (primary_face && primary_face != preferred_face) {
        if (wcn_attempt_get_glyph(ctx, primary_face, codepoint, out_glyph)) {
            if (out_face) {
                *out_face = primary_face;
            }
            return true;
        }
    }

    for (size_t i = 0; i < ctx->font_fallback_count; i++) {
        WCN_FontFace* fallback = ctx->font_fallbacks[i];
        if (!fallback || fallback == preferred_face || fallback == primary_face) {
            continue;
        }
        if (wcn_attempt_get_glyph(ctx, fallback, codepoint, out_glyph)) {
            if (out_face) {
                *out_face = fallback;
            }
            return true;
        }
    }

    return false;
}

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

    WCN_FontFace* base_face = wcn_get_primary_font_face(ctx);
    if (!base_face) {
        printf("警告: 没有可用的字体\n");
        return;
    }

    // 计算对齐偏移
    float total_width = 0;
    if (ctx->text_align != WCN_TEXT_ALIGN_LEFT) {
        WCN_TextMetrics metrics = wcn_measure_text(ctx, text);
        total_width = metrics.width;
    }

    // 计算字体缩放比例
    float font_scale = ctx->current_font_size / base_face->units_per_em;
    
    // 基线偏移计算（注意：stb_truetype 的坐标系）
    // ascent 是正值（基线上方），descent 是负值（基线下方）
    float baseline_offset = 0;
    switch (ctx->text_baseline) {
        case WCN_TEXT_BASELINE_TOP:
            // 从顶部对齐：向下偏移 ascent
            baseline_offset = base_face->ascent * font_scale;
            break;
        case WCN_TEXT_BASELINE_MIDDLE:
            // 从中间对齐：向下偏移 (ascent + descent) / 2
            baseline_offset = (base_face->ascent + base_face->descent) * 0.5f * font_scale;
            break;
        case WCN_TEXT_BASELINE_BOTTOM:
            // 从底部对齐：向下偏移 descent（通常是负值）
            baseline_offset = base_face->descent * font_scale;
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
    
    WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];

    // 计算应用变换后的文本位置
    float original_x = x + align_offset;
    float original_y = y + baseline_offset;

    // 应用变换矩阵的平移分量到文本位置
    float transformed_x = original_x * state->transform_matrix[0] + original_y * state->transform_matrix[4] + state->transform_matrix[12];
    float transformed_y = original_x * state->transform_matrix[1] + original_y * state->transform_matrix[5] + state->transform_matrix[13];

    // 提取 2x2 变换矩阵（从 4x4 矩阵的左上角）- 用于实例变换
    float instance_transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };

    // 添加文本实例到统一渲染器
    wcn_renderer_add_text(
        ctx->renderer,
        ctx,
        text,
        transformed_x, transformed_y,
        ctx->current_font_size,
        state->fill_color,
        instance_transform
    );
}

void wcn_stroke_text(WCN_Context* ctx, const char* text, float x, float y) {
    if (!ctx || !text || !ctx->in_frame || !ctx->renderer) return;

    // 检查是否有字体解码器和 MSDF 支持
    if (!ctx->font_decoder || !ctx->font_decoder->get_glyph_sdf) {
        printf("警告: 没有字体解码器或不支持 MSDF\n");
        return;
    }

    WCN_FontFace* base_face = wcn_get_primary_font_face(ctx);
    if (!base_face) {
        printf("警告: 没有可用的字体\n");
        return;
    }

    // 计算对齐偏移
    float total_width = 0;
    if (ctx->text_align != WCN_TEXT_ALIGN_LEFT) {
        WCN_TextMetrics metrics = wcn_measure_text(ctx, text);
        total_width = metrics.width;
    }

    // 计算字体缩放比例
    float font_scale = ctx->current_font_size / base_face->units_per_em;
    
    // 基线偏移计算（注意：stb_truetype 的坐标系）
    // ascent 是正值（基线上方），descent 是负值（基线下方）
    float baseline_offset = 0;
    switch (ctx->text_baseline) {
        case WCN_TEXT_BASELINE_TOP:
            // 从顶部对齐：向下偏移 ascent
            baseline_offset = base_face->ascent * font_scale;
            break;
        case WCN_TEXT_BASELINE_MIDDLE:
            // 从中间对齐：向下偏移 (ascent + descent) / 2
            baseline_offset = (base_face->ascent + base_face->descent) * 0.5f * font_scale;
            break;
        case WCN_TEXT_BASELINE_BOTTOM:
            // 从底部对齐：向下偏移 descent（通常是负值）
            baseline_offset = base_face->descent * font_scale;
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
    
    WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];

    // 计算应用变换后的文本位置
    float original_x = x + align_offset;
    float original_y = y + baseline_offset;

    // 应用变换矩阵的平移分量到文本位置
    float transformed_x = original_x * state->transform_matrix[0] + original_y * state->transform_matrix[4] + state->transform_matrix[12];
    float transformed_y = original_x * state->transform_matrix[1] + original_y * state->transform_matrix[5] + state->transform_matrix[13];

    // 提取 2x2 变换矩阵（从 4x4 矩阵的左上角）- 用于实例变换
    float instance_transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };

    // 添加文本实例到统一渲染器（使用描边颜色）
    wcn_renderer_add_text(
        ctx->renderer,
        ctx,
        text,
        transformed_x, transformed_y,
        ctx->current_font_size,
        state->stroke_color,
        instance_transform
    );
}

// ============================================================================
// 文本测量
// ============================================================================

WCN_TextMetrics wcn_measure_text(WCN_Context* ctx, const char* text) {
    WCN_TextMetrics metrics = {0};

    WCN_FontFace* base_face = wcn_get_primary_font_face(ctx);
    if (!ctx || !text || !ctx->font_decoder || !base_face) {
        return metrics;
    }

    bool has_fallbacks = ctx->font_fallback_count > 0;
    if (!has_fallbacks && ctx->font_decoder->measure_text) {
        float width, height;
        if (ctx->font_decoder->measure_text(base_face, text,
                                           ctx->current_font_size, &width, &height)) {
            metrics.width = width;
            metrics.em_height_ascent = base_face->ascent * (ctx->current_font_size / base_face->units_per_em);
            metrics.em_height_descent = base_face->descent * (ctx->current_font_size / base_face->units_per_em);
            return metrics;
        }
    }

    float base_scale = ctx->current_font_size / base_face->units_per_em;
    const char* ptr = text;
    float x_offset = 0;
    WCN_FontFace* last_face = base_face;

    while (*ptr) {
        const char* run_start = ptr;
        uint32_t codepoint = wcn_decode_utf8(&ptr);
        if (codepoint == 0) {
            break;
        }

        if (wcn_is_variation_selector(codepoint) || wcn_is_zero_width_joiner(codepoint)) {
            continue;
        }

        WCN_FontFace* glyph_face = NULL;
        WCN_Glyph* glyph = NULL;
        if (!wcn_get_glyph_with_fallback(ctx, codepoint, last_face, &glyph_face, &glyph)) {
            x_offset += 8.0f;
            continue;
        }

        last_face = glyph_face ? glyph_face : last_face;
        WCN_FontFace* face_for_metrics = glyph_face ? glyph_face : base_face;
        float glyph_scale = ctx->current_font_size / face_for_metrics->units_per_em;
        x_offset += glyph->advance_width * glyph_scale;

        if (ctx->font_decoder->free_glyph && glyph) {
            ctx->font_decoder->free_glyph(glyph);
        }
    }

    metrics.width = x_offset;
    metrics.em_height_ascent = base_face->ascent * base_scale;
    metrics.em_height_descent = base_face->descent * base_scale;

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

bool wcn_add_font_fallback(WCN_Context* ctx, WCN_FontFace* face) {
    if (!ctx || !face) {
        return false;
    }

    for (size_t i = 0; i < ctx->font_fallback_count; i++) {
        if (ctx->font_fallbacks[i] == face) {
            return true;
        }
    }

    if (ctx->font_fallback_count >= ctx->font_fallback_capacity) {
        size_t new_capacity = ctx->font_fallback_capacity == 0 ? 4 : ctx->font_fallback_capacity * 2;
        WCN_FontFace** new_list = (WCN_FontFace**)realloc(ctx->font_fallbacks, new_capacity * sizeof(WCN_FontFace*));
        if (!new_list) {
            return false;
        }
        ctx->font_fallbacks = new_list;
        ctx->font_fallback_capacity = new_capacity;
    }

    ctx->font_fallbacks[ctx->font_fallback_count++] = face;
    return true;
}

void wcn_clear_font_fallbacks(WCN_Context* ctx) {
    if (!ctx) {
        return;
    }
    free(ctx->font_fallbacks);
    ctx->font_fallbacks = NULL;
    ctx->font_fallback_count = 0;
    ctx->font_fallback_capacity = 0;
}
