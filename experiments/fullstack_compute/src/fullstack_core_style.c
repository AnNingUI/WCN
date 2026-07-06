#include "fullstack_core_private.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool fs_style_copy_gradient_stops(
    FS_GradientStop* dst,
    uint32_t dst_max_count,
    uint32_t* out_count,
    const FS_GradientStop* src,
    uint32_t src_count
) {
    if (!dst || !out_count || !src || src_count < 2u || dst_max_count == 0u) {
        return false;
    }
    uint32_t copy_count = src_count;
    if (copy_count > dst_max_count) {
        copy_count = dst_max_count;
    }
    if (copy_count == src_count) {
        memcpy(dst, src, (size_t)copy_count * sizeof(FS_GradientStop));
    } else if (copy_count == 1u) {
        dst[0] = src[0];
    } else {
        for (uint32_t i = 0u; i < copy_count; ++i) {
            const uint32_t src_index = (uint32_t)(((uint64_t)i * (uint64_t)(src_count - 1u)) / (uint64_t)(copy_count - 1u));
            dst[i] = src[src_index];
        }
    }
    *out_count = copy_count;
    return true;
}

bool fs_ensure_state_stack_capacity(FS_InternalState* st, size_t required) {
    if (!st) {
        return false;
    }
    if (required <= st->state_stack_capacity) {
        return true;
    }
    size_t new_cap = st->state_stack_capacity ? st->state_stack_capacity : 16u;
    while (new_cap < required) {
        if (new_cap > (SIZE_MAX / 2u)) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_StateSnapshot* grown = (FS_StateSnapshot*)realloc(st->state_stack, new_cap * sizeof(FS_StateSnapshot));
    if (!grown) {
        return false;
    }
    st->state_stack = grown;
    st->state_stack_capacity = (uint32_t)new_cap;
    return true;
}

static void fs_style_snapshot_dispose(FS_StyleSnapshot* snap) {
    if (!snap) {
        return;
    }
    free(snap->dash_segments);
    snap->dash_segments = NULL;
    snap->dash_count = 0u;
    snap->dash_offset = 0.0f;
    fs_filter_chain_destroy(snap->filter_chain);
    snap->filter_chain = NULL;
}

bool fs_style_snapshot_capture(FS_StyleSnapshot* dst, const FS_InternalState* st) {
    if (!dst || !st) {
        return false;
    }
    memset(dst, 0, sizeof(*dst));
    dst->line_width = st->style_line_width;
    dst->miter_limit = st->style_miter_limit;
    dst->line_cap = st->style_line_cap;
    dst->line_join = st->style_line_join;
    dst->fill_rule = st->style_fill_rule;
    dst->composite_op = st->style_composite_op;
    dst->text_align = st->style_text_align;
    dst->text_baseline = st->style_text_baseline;
    dst->text_direction = st->style_text_direction;
    dst->font_kerning = st->style_font_kerning;
    dst->text_rendering = st->style_text_rendering;
    dst->font_stretch = st->style_font_stretch;
    dst->font_variant_caps = st->style_font_variant_caps;
    dst->font_size_px = st->style_font_size_px;
    memcpy(dst->font_family, st->style_font_family, sizeof(dst->font_family));
    dst->letter_spacing = st->style_letter_spacing;
    dst->word_spacing = st->style_word_spacing;
    dst->image_smoothing_enabled = st->style_image_smoothing_enabled;
    dst->image_smoothing_quality = st->style_image_smoothing_quality;
    dst->global_alpha = st->style_global_alpha;
    dst->fill_color_rgba8 = st->style_fill_color_rgba8;
    dst->stroke_color_rgba8 = st->style_stroke_color_rgba8;
    dst->fill_paint_type = st->style_fill_paint_type;
    dst->stroke_paint_type = st->style_stroke_paint_type;
    dst->fill_linear_gradient = st->style_fill_linear_gradient;
    dst->stroke_linear_gradient = st->style_stroke_linear_gradient;
    dst->fill_radial_gradient = st->style_fill_radial_gradient;
    dst->stroke_radial_gradient = st->style_stroke_radial_gradient;
    dst->fill_conic_gradient = st->style_fill_conic_gradient;
    dst->stroke_conic_gradient = st->style_stroke_conic_gradient;
    dst->fill_pattern = st->style_fill_pattern;
    dst->stroke_pattern = st->style_stroke_pattern;
    dst->shadow_color_rgba8 = st->style_shadow_color_rgba8;
    dst->shadow_blur = st->style_shadow_blur;
    dst->shadow_offset_x = st->style_shadow_offset_x;
    dst->shadow_offset_y = st->style_shadow_offset_y;
    dst->dash_offset = st->style_dash_offset;
    if (st->style_dash_count > 0u) {
        if (!st->style_dash_segments) {
            return false;
        }
        dst->dash_segments = (float*)malloc((size_t)st->style_dash_count * sizeof(float));
        if (!dst->dash_segments) {
            return false;
        }
        memcpy(dst->dash_segments, st->style_dash_segments, (size_t)st->style_dash_count * sizeof(float));
        dst->dash_count = st->style_dash_count;
    }
    dst->filter_chain = fs_filter_chain_clone(st->filter_chain);
    return true;
}

void fs_style_snapshot_apply(FS_InternalState* st, FS_StyleSnapshot* src) {
    if (!st || !src) {
        return;
    }
    free(st->style_dash_segments);
    st->style_line_width = src->line_width;
    st->style_miter_limit = src->miter_limit;
    st->style_line_cap = src->line_cap;
    st->style_line_join = src->line_join;
    st->style_fill_rule = src->fill_rule;
    st->style_composite_op = src->composite_op;
    st->style_text_align = src->text_align;
    st->style_text_baseline = src->text_baseline;
    st->style_text_direction = src->text_direction;
    st->style_font_kerning = src->font_kerning;
    st->style_text_rendering = src->text_rendering;
    st->style_font_stretch = src->font_stretch;
    st->style_font_variant_caps = src->font_variant_caps;
    st->style_font_size_px = src->font_size_px;
    memcpy(st->style_font_family, src->font_family, sizeof(st->style_font_family));
    st->style_letter_spacing = src->letter_spacing;
    st->style_word_spacing = src->word_spacing;
    st->style_image_smoothing_enabled = src->image_smoothing_enabled;
    st->style_image_smoothing_quality = src->image_smoothing_quality;
    st->style_global_alpha = src->global_alpha;
    st->style_fill_color_rgba8 = src->fill_color_rgba8;
    st->style_stroke_color_rgba8 = src->stroke_color_rgba8;
    st->style_fill_paint_type = src->fill_paint_type;
    st->style_stroke_paint_type = src->stroke_paint_type;
    st->style_fill_linear_gradient = src->fill_linear_gradient;
    st->style_stroke_linear_gradient = src->stroke_linear_gradient;
    st->style_fill_radial_gradient = src->fill_radial_gradient;
    st->style_stroke_radial_gradient = src->stroke_radial_gradient;
    st->style_fill_conic_gradient = src->fill_conic_gradient;
    st->style_stroke_conic_gradient = src->stroke_conic_gradient;
    st->style_fill_pattern = src->fill_pattern;
    st->style_stroke_pattern = src->stroke_pattern;
    st->style_shadow_color_rgba8 = src->shadow_color_rgba8;
    st->style_shadow_blur = src->shadow_blur;
    st->style_shadow_offset_x = src->shadow_offset_x;
    st->style_shadow_offset_y = src->shadow_offset_y;
    st->style_dash_segments = src->dash_segments;
    st->style_dash_count = src->dash_count;
    st->style_dash_offset = src->dash_offset;
    src->dash_segments = NULL;
    src->dash_count = 0u;
    src->dash_offset = 0.0f;
    fs_filter_chain_destroy(st->filter_chain);
    st->filter_chain = src->filter_chain;
    src->filter_chain = NULL;
}

void fs_state_snapshot_dispose(FS_StateSnapshot* snap) {
    if (!snap) {
        return;
    }
    fs_style_snapshot_dispose(&snap->style);
}

void fs_state_stack_clear(FS_InternalState* st) {
    if (!st) {
        return;
    }
    for (uint32_t i = 0u; i < st->state_stack_count; ++i) {
        fs_state_snapshot_dispose(&st->state_stack[i]);
    }
    st->state_stack_count = 0u;
}

void fs_style_reset_state(FS_InternalState* st) {
    if (!st) {
        return;
    }
    st->style_line_width = 1.0f;
    st->style_miter_limit = 10.0f;
    st->style_line_cap = (uint8_t)FS_LINE_CAP_ROUND;
    st->style_line_join = (uint8_t)FS_LINE_JOIN_MITER;
    st->style_fill_rule = (uint8_t)FS_FILL_RULE_NONZERO;
    st->style_composite_op = (uint8_t)FS_GLOBAL_COMPOSITE_SOURCE_OVER;
    st->style_text_align = (uint8_t)FS_TEXT_ALIGN_START;
    st->style_text_baseline = (uint8_t)FS_TEXT_BASELINE_ALPHABETIC;
    st->style_text_direction = (uint8_t)FS_TEXT_DIRECTION_LTR;
    st->style_font_kerning = (uint8_t)FS_FONT_KERNING_AUTO;
    st->style_text_rendering = (uint8_t)FS_TEXT_RENDERING_AUTO;
    st->style_font_stretch = (uint8_t)FS_FONT_STRETCH_NORMAL;
    st->style_font_variant_caps = (uint8_t)FS_FONT_VARIANT_CAPS_NORMAL;
    st->style_font_size_px = 16.0f;
    st->style_font_family[0] = '\0';
    st->style_letter_spacing = 0.0f;
    st->style_word_spacing = 0.0f;
    st->style_image_smoothing_enabled = 1u;
    st->style_image_smoothing_quality = (uint8_t)FS_IMAGE_SMOOTHING_QUALITY_LOW;
    st->style_global_alpha = 1.0f;
    st->style_fill_color_rgba8 = 0xFF000000u;
    st->style_stroke_color_rgba8 = 0xFF000000u;
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_SOLID;
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_SOLID;
    st->style_fill_linear_gradient.stop_count = 0u;
    st->style_stroke_linear_gradient.stop_count = 0u;
    st->style_fill_radial_gradient.stop_count = 0u;
    st->style_stroke_radial_gradient.stop_count = 0u;
    st->style_fill_conic_gradient.stop_count = 0u;
    st->style_stroke_conic_gradient.stop_count = 0u;
    memset(&st->style_fill_pattern, 0, sizeof(st->style_fill_pattern));
    memset(&st->style_stroke_pattern, 0, sizeof(st->style_stroke_pattern));
    st->style_fill_pattern.repeat_mode = (uint8_t)FS_PATTERN_REPEAT;
    st->style_stroke_pattern.repeat_mode = (uint8_t)FS_PATTERN_REPEAT;
    fs_affine_set_identity_2d(st->style_fill_pattern.xform);
    fs_affine_set_identity_2d(st->style_fill_pattern.inv_xform);
    st->style_fill_pattern.inv_valid = 1u;
    fs_affine_set_identity_2d(st->style_stroke_pattern.xform);
    fs_affine_set_identity_2d(st->style_stroke_pattern.inv_xform);
    st->style_stroke_pattern.inv_valid = 1u;
    st->style_shadow_color_rgba8 = 0u;
    st->style_shadow_blur = 0.0f;
    st->style_shadow_offset_x = 0.0f;
    st->style_shadow_offset_y = 0.0f;
    free(st->style_dash_segments);
    st->style_dash_segments = NULL;
    st->style_dash_count = 0u;
    st->style_dash_offset = 0.0f;
    fs_filter_chain_destroy(st->filter_chain);
    st->filter_chain = NULL;
}

bool fs_style_has_dash(const FS_InternalState* st) {
    return st && st->style_dash_segments && st->style_dash_count > 0u;
}

float fs_style_resolve_line_width(const FS_InternalState* st, float width) {
    if (width > 0.0f) {
        return width;
    }
    if (!st || st->style_line_width <= 0.0f) {
        return 1.0f;
    }
    return st->style_line_width;
}

void fs_resolve_text_vertical_metrics(
    const FS_InternalState* st,
    float font_size_px,
    float* out_em_ascent,
    float* out_em_descent,
    float* out_line_height
) {
    float em_ascent = (font_size_px > 0.0f) ? (font_size_px * 0.8f) : 0.0f;
    float em_descent = (font_size_px > 0.0f) ? (font_size_px * 0.2f) : 0.0f;
    float line_height = (font_size_px > 0.0f) ? (font_size_px * 1.25f) : 0.0f;
    if (st && st->font_backend && st->font_backend->get_vertical_metrics && st->font_count > 0u && st->fonts[0]) {
        FS_FontVerticalMetrics vm = {0};
        if (st->font_backend->get_vertical_metrics(st->fonts[0], font_size_px, &vm)) {
            if (isfinite(vm.ascent) && vm.ascent > 0.0f) {
                em_ascent = vm.ascent;
            }
            if (isfinite(vm.descent) && vm.descent >= 0.0f) {
                em_descent = vm.descent;
            }
            if (isfinite(vm.line_height) && vm.line_height > 0.0f) {
                line_height = vm.line_height;
            }
        }
    }
    if (!isfinite(line_height) || line_height <= 0.0f) {
        line_height = em_ascent + em_descent;
    } else if (line_height < em_ascent + em_descent) {
        line_height = em_ascent + em_descent;
    }
    if (out_em_ascent) {
        *out_em_ascent = em_ascent;
    }
    if (out_em_descent) {
        *out_em_descent = em_descent;
    }
    if (out_line_height) {
        *out_line_height = line_height;
    }
}

float fs_text_align_offset(const FS_InternalState* st, const FS_TextMetrics* metrics) {
    if (!st || !metrics) {
        return 0.0f;
    }
    const bool rtl = st->style_text_direction == (uint8_t)FS_TEXT_DIRECTION_RTL;
    switch ((FS_TextAlign)st->style_text_align) {
        case FS_TEXT_ALIGN_CENTER:
            return -0.5f * metrics->width;
        case FS_TEXT_ALIGN_RIGHT:
            return -metrics->width;
        case FS_TEXT_ALIGN_END:
            return rtl ? 0.0f : -metrics->width;
        case FS_TEXT_ALIGN_START:
            return rtl ? -metrics->width : 0.0f;
        case FS_TEXT_ALIGN_LEFT:
        default:
            return 0.0f;
    }
}

bool fs_is_word_spacing_codepoint(uint32_t cp) {
    switch (cp) {
        case 0x0009u:
        case 0x000Bu:
        case 0x000Cu:
        case 0x0020u:
        case 0x00A0u:
        case 0x1680u:
        case 0x2000u:
        case 0x2001u:
        case 0x2002u:
        case 0x2003u:
        case 0x2004u:
        case 0x2005u:
        case 0x2006u:
        case 0x2007u:
        case 0x2008u:
        case 0x2009u:
        case 0x200Au:
        case 0x202Fu:
        case 0x205Fu:
        case 0x3000u:
            return true;
        default:
            return false;
    }
}

float fs_text_baseline_offset(const FS_InternalState* st, const FS_TextMetrics* metrics, float font_size_px) {
    if (!st || !metrics || font_size_px <= 0.0f) {
        return 0.0f;
    }
    const float em_ascent = (metrics->em_height_ascent > 0.0f) ? metrics->em_height_ascent : (font_size_px * 0.8f);
    const float em_descent = (metrics->em_height_descent > 0.0f) ? metrics->em_height_descent : (font_size_px * 0.2f);
    const float em_middle = 0.5f * (em_ascent - em_descent);
    switch ((FS_TextBaseline)st->style_text_baseline) {
        case FS_TEXT_BASELINE_TOP:
            return em_ascent;
        case FS_TEXT_BASELINE_HANGING:
            return em_ascent * 0.8f;
        case FS_TEXT_BASELINE_MIDDLE:
            return em_middle;
        case FS_TEXT_BASELINE_IDEOGRAPHIC:
            return -em_descent;
        case FS_TEXT_BASELINE_BOTTOM:
            return -em_descent;
        case FS_TEXT_BASELINE_ALPHABETIC:
        default:
            return 0.0f;
    }
}

bool fs_is_text_kerning_enabled(const FS_InternalState* st) {
    if (!st) {
        return true;
    }
    const FS_FontKerning kerning = (FS_FontKerning)st->style_font_kerning;
    if (kerning == FS_FONT_KERNING_NONE) {
        return false;
    }
    const FS_TextRendering rendering = (FS_TextRendering)st->style_text_rendering;
    if (rendering == FS_TEXT_RENDERING_OPTIMIZE_SPEED) {
        return false;
    }
    return true;
}

bool fs_is_text_geometric_precision(const FS_InternalState* st) {
    if (!st) {
        return false;
    }
    return (FS_TextRendering)st->style_text_rendering == FS_TEXT_RENDERING_GEOMETRIC_PRECISION;
}

float fs_text_stretch_scale(const FS_InternalState* st) {
    if (!st) {
        return 1.0f;
    }
    switch ((FS_FontStretch)st->style_font_stretch) {
        case FS_FONT_STRETCH_ULTRA_CONDENSED:
            return 0.5f;
        case FS_FONT_STRETCH_EXTRA_CONDENSED:
            return 0.625f;
        case FS_FONT_STRETCH_CONDENSED:
            return 0.75f;
        case FS_FONT_STRETCH_SEMI_CONDENSED:
            return 0.875f;
        case FS_FONT_STRETCH_SEMI_EXPANDED:
            return 1.125f;
        case FS_FONT_STRETCH_EXPANDED:
            return 1.25f;
        case FS_FONT_STRETCH_EXTRA_EXPANDED:
            return 1.5f;
        case FS_FONT_STRETCH_ULTRA_EXPANDED:
            return 2.0f;
        case FS_FONT_STRETCH_NORMAL:
        default:
            return 1.0f;
    }
}

bool fs_is_text_small_caps_enabled(const FS_InternalState* st) {
    if (!st) {
        return false;
    }
    switch ((FS_FontVariantCaps)st->style_font_variant_caps) {
        case FS_FONT_VARIANT_CAPS_SMALL_CAPS:
        case FS_FONT_VARIANT_CAPS_ALL_SMALL_CAPS:
            return true;
        default:
            return false;
    }
}

void fs_text_variant_map_codepoint(const FS_InternalState* st, uint32_t cp, uint32_t* out_cp, float* out_size_scale) {
    uint32_t mapped_cp = cp;
    float mapped_scale = 1.0f;
    if (fs_is_text_small_caps_enabled(st)) {
        const FS_FontVariantCaps variant = (FS_FontVariantCaps)st->style_font_variant_caps;
        if (cp >= (uint32_t)'a' && cp <= (uint32_t)'z') {
            mapped_cp = cp - ((uint32_t)'a' - (uint32_t)'A');
            mapped_scale = 0.82f;
        } else if (variant == FS_FONT_VARIANT_CAPS_ALL_SMALL_CAPS &&
                   cp >= (uint32_t)'A' && cp <= (uint32_t)'Z') {
            mapped_cp = cp;
            mapped_scale = 0.82f;
        }
    }
    if (out_cp) {
        *out_cp = mapped_cp;
    }
    if (out_size_scale) {
        *out_size_scale = mapped_scale;
    }
}

void fs_clip_reset_state(FS_InternalState* st) {
    if (!st) {
        return;
    }
    st->clip_enabled = 0u;
    st->clip_path_enabled = 0u;
    st->clip_path_layer = 0u;
    st->clip_min_x = 0.0f;
    st->clip_min_y = 0.0f;
    st->clip_max_x = 0.0f;
    st->clip_max_y = 0.0f;
}

bool fs_style_copy_linear_gradient(FS_StyleLinearGradient* dst, const FS_LinearGradient* src) {
    if (!dst || !src || !isfinite(src->x0) || !isfinite(src->y0) || !isfinite(src->x1) || !isfinite(src->y1)) {
        return false;
    }
    if (!src->stops || src->stop_count < 2u) {
        return false;
    }
    dst->x0 = src->x0;
    dst->y0 = src->y0;
    dst->x1 = src->x1;
    dst->y1 = src->y1;
    return fs_style_copy_gradient_stops(
        dst->stops,
        FS_LINEAR_GRADIENT_MAX_STOPS,
        &dst->stop_count,
        src->stops,
        src->stop_count
    );
}

bool fs_style_copy_radial_gradient(FS_StyleRadialGradient* dst, const FS_RadialGradient* src) {
    if (!dst || !src || !isfinite(src->x0) || !isfinite(src->y0) || !isfinite(src->r0) || src->r0 < 0.0f ||
        !isfinite(src->x1) || !isfinite(src->y1) || !isfinite(src->r1) || src->r1 < 0.0f) {
        return false;
    }
    if (!src->stops || src->stop_count < 2u) {
        return false;
    }
    dst->x0 = src->x0;
    dst->y0 = src->y0;
    dst->r0 = src->r0;
    dst->x1 = src->x1;
    dst->y1 = src->y1;
    dst->r1 = src->r1;
    return fs_style_copy_gradient_stops(
        dst->stops,
        FS_RADIAL_GRADIENT_MAX_STOPS,
        &dst->stop_count,
        src->stops,
        src->stop_count
    );
}

bool fs_style_copy_conic_gradient(FS_StyleConicGradient* dst, const FS_ConicGradient* src) {
    if (!dst || !src || !isfinite(src->start_angle_radians) || !isfinite(src->cx) || !isfinite(src->cy)) {
        return false;
    }
    if (!src->stops || src->stop_count < 2u) {
        return false;
    }
    dst->start_angle_radians = src->start_angle_radians;
    dst->cx = src->cx;
    dst->cy = src->cy;
    return fs_style_copy_gradient_stops(
        dst->stops,
        FS_CONIC_GRADIENT_MAX_STOPS,
        &dst->stop_count,
        src->stops,
        src->stop_count
    );
}

bool fs_style_copy_pattern(FS_StylePattern* dst, const FS_Pattern* src) {
    if (!dst || !src) {
        return false;
    }
    const uint8_t repeat_mode = src->repeat_mode;
    if (repeat_mode > (uint8_t)FS_PATTERN_NO_REPEAT) {
        return false;
    }
    if (src->handle.width == 0u || src->handle.height == 0u) {
        return false;
    }
    *dst = (FS_StylePattern){
        .handle = src->handle,
        .xform = {src->xform[0], src->xform[1], src->xform[2], src->xform[3], src->xform[4], src->xform[5]},
        .inv_xform = {
            src->inv_xform[0],
            src->inv_xform[1],
            src->inv_xform[2],
            src->inv_xform[3],
            src->inv_xform[4],
            src->inv_xform[5]
        },
        .repeat_mode = repeat_mode,
        .inv_valid = src->inv_valid,
        ._pad0 = 0u,
        ._pad1 = 0u
    };
    if (!dst->inv_valid) {
        dst->inv_valid = fs_affine_try_invert_2d(dst->xform, dst->inv_xform) ? 1u : 0u;
        if (!dst->inv_valid) {
            return false;
        }
    }
    return true;
}

bool fs_style_set_font(FS_Core* core, const char* font_family, float font_size_px) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (font_size_px <= 0.0f) {
        font_size_px = 16.0f;
    }
    st->style_font_size_px = font_size_px;
    if (font_family) {
        size_t len = strlen(font_family);
        if (len >= sizeof(st->style_font_family)) {
            len = sizeof(st->style_font_family) - 1u;
        }
        memcpy(st->style_font_family, font_family, len);
        st->style_font_family[len] = '\0';
    } else {
        st->style_font_family[0] = '\0';
    }
    return true;
}

bool fs_style_get_font_family(const FS_Core* core, char* buffer, size_t buffer_size) {
    FS_InternalState* st = fs_state((FS_Core*)core);
    if (!st || !buffer || buffer_size == 0u) {
        return false;
    }
    size_t len = strlen(st->style_font_family);
    if (len >= buffer_size) {
        len = buffer_size - 1u;
    }
    memcpy(buffer, st->style_font_family, len);
    buffer[len] = '\0';
    return true;
}

float fs_style_get_font_size(const FS_Core* core) {
    FS_InternalState* st = fs_state((FS_Core*)core);
    if (!st) {
        return 16.0f;
    }
    return st->style_font_size_px;
}
