#include "fullstack_core_private.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void fs_style_reset(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    fs_style_reset_state(st);
}

bool fs_style_set_line_width(FS_Core* core, float width) {
    FS_InternalState* st = fs_state(core);
    if (!st || width <= 0.0f) {
        return false;
    }
    st->style_line_width = width;
    return true;
}

bool fs_style_set_line_cap(FS_Core* core, FS_LineCap cap) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (cap != FS_LINE_CAP_BUTT && cap != FS_LINE_CAP_ROUND && cap != FS_LINE_CAP_SQUARE) {
        return false;
    }
    st->style_line_cap = (uint8_t)cap;
    return true;
}

bool fs_style_set_line_join(FS_Core* core, FS_LineJoin join) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (join != FS_LINE_JOIN_MITER && join != FS_LINE_JOIN_ROUND && join != FS_LINE_JOIN_BEVEL) {
        return false;
    }
    st->style_line_join = (uint8_t)join;
    return true;
}

bool fs_style_set_fill_rule(FS_Core* core, FS_FillRule fill_rule) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (fill_rule != FS_FILL_RULE_NONZERO && fill_rule != FS_FILL_RULE_EVENODD) {
        return false;
    }
    st->style_fill_rule = (uint8_t)fill_rule;
    return true;
}

bool fs_style_set_miter_limit(FS_Core* core, float limit) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(limit) || limit <= 0.0f) {
        return false;
    }
    st->style_miter_limit = limit;
    return true;
}

float fs_style_get_miter_limit(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_miter_limit) || st->style_miter_limit <= 0.0f) {
        return 10.0f;
    }
    return st->style_miter_limit;
}

bool fs_style_set_global_alpha(FS_Core* core, float alpha) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(alpha)) {
        return false;
    }
    if (alpha < 0.0f) {
        alpha = 0.0f;
    } else if (alpha > 1.0f) {
        alpha = 1.0f;
    }
    st->style_global_alpha = alpha;
    return true;
}

float fs_style_get_global_alpha(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return 1.0f;
    }
    return st->style_global_alpha;
}

bool fs_style_set_global_composite_operation(FS_Core* core, FS_GlobalCompositeOperation op) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (op) {
        case FS_GLOBAL_COMPOSITE_SOURCE_OVER:
        case FS_GLOBAL_COMPOSITE_COPY:
        case FS_GLOBAL_COMPOSITE_LIGHTER:
        case FS_GLOBAL_COMPOSITE_DESTINATION_OVER:
        case FS_GLOBAL_COMPOSITE_SOURCE_IN:
        case FS_GLOBAL_COMPOSITE_SOURCE_OUT:
        case FS_GLOBAL_COMPOSITE_DESTINATION_IN:
        case FS_GLOBAL_COMPOSITE_DESTINATION_OUT:
        case FS_GLOBAL_COMPOSITE_XOR:
        case FS_GLOBAL_COMPOSITE_SOURCE_ATOP:
        case FS_GLOBAL_COMPOSITE_DESTINATION_ATOP:
            st->style_composite_op = (uint8_t)op;
            return true;
        default:
            return false;
    }
}

FS_GlobalCompositeOperation fs_style_get_global_composite_operation(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_GLOBAL_COMPOSITE_SOURCE_OVER;
    }
    switch ((FS_GlobalCompositeOperation)st->style_composite_op) {
        case FS_GLOBAL_COMPOSITE_COPY:
        case FS_GLOBAL_COMPOSITE_LIGHTER:
        case FS_GLOBAL_COMPOSITE_DESTINATION_OVER:
        case FS_GLOBAL_COMPOSITE_SOURCE_IN:
        case FS_GLOBAL_COMPOSITE_SOURCE_OUT:
        case FS_GLOBAL_COMPOSITE_DESTINATION_IN:
        case FS_GLOBAL_COMPOSITE_DESTINATION_OUT:
        case FS_GLOBAL_COMPOSITE_XOR:
        case FS_GLOBAL_COMPOSITE_SOURCE_ATOP:
        case FS_GLOBAL_COMPOSITE_DESTINATION_ATOP:
            return (FS_GlobalCompositeOperation)st->style_composite_op;
        case FS_GLOBAL_COMPOSITE_SOURCE_OVER:
        default:
            return FS_GLOBAL_COMPOSITE_SOURCE_OVER;
    }
}

bool fs_style_set_shadow_color(FS_Core* core, uint32_t color_rgba8) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->style_shadow_color_rgba8 = color_rgba8;
    return true;
}

uint32_t fs_style_get_shadow_color(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return 0u;
    }
    return st->style_shadow_color_rgba8;
}

bool fs_style_set_fill_color(FS_Core* core, uint32_t color_rgba8) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->style_fill_color_rgba8 = color_rgba8;
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_SOLID;
    return true;
}

uint32_t fs_style_get_fill_color(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return 0xFF000000u;
    }
    return st->style_fill_color_rgba8;
}

bool fs_style_set_stroke_color(FS_Core* core, uint32_t color_rgba8) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->style_stroke_color_rgba8 = color_rgba8;
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_SOLID;
    return true;
}

uint32_t fs_style_get_stroke_color(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return 0xFF000000u;
    }
    return st->style_stroke_color_rgba8;
}

bool fs_style_set_fill_linear_gradient(FS_Core* core, const FS_LinearGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_linear_gradient(&st->style_fill_linear_gradient, gradient)) {
        return false;
    }
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT;
    return true;
}

bool fs_style_set_stroke_linear_gradient(FS_Core* core, const FS_LinearGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_linear_gradient(&st->style_stroke_linear_gradient, gradient)) {
        return false;
    }
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT;
    return true;
}

bool fs_style_set_fill_radial_gradient(FS_Core* core, const FS_RadialGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_radial_gradient(&st->style_fill_radial_gradient, gradient)) {
        return false;
    }
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT;
    return true;
}

bool fs_style_set_stroke_radial_gradient(FS_Core* core, const FS_RadialGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_radial_gradient(&st->style_stroke_radial_gradient, gradient)) {
        return false;
    }
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT;
    return true;
}

bool fs_style_set_fill_conic_gradient(FS_Core* core, const FS_ConicGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_conic_gradient(&st->style_fill_conic_gradient, gradient)) {
        return false;
    }
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_CONIC_GRADIENT;
    return true;
}

bool fs_style_set_stroke_conic_gradient(FS_Core* core, const FS_ConicGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_conic_gradient(&st->style_stroke_conic_gradient, gradient)) {
        return false;
    }
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_CONIC_GRADIENT;
    return true;
}

bool fs_style_set_fill_pattern(FS_Core* core, const FS_Pattern* pattern) {
    FS_InternalState* st = fs_state(core);
    if (!st || !pattern) {
        return false;
    }
    if (!fs_style_copy_pattern(&st->style_fill_pattern, pattern)) {
        return false;
    }
    uint32_t atlas_x = 0u;
    uint32_t atlas_y = 0u;
    if (!fs_image_handle_resolve_atlas_origin(core, &st->style_fill_pattern.handle, &atlas_x, &atlas_y)) {
        return false;
    }
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_PATTERN;
    return true;
}

bool fs_style_set_stroke_pattern(FS_Core* core, const FS_Pattern* pattern) {
    FS_InternalState* st = fs_state(core);
    if (!st || !pattern) {
        return false;
    }
    if (!fs_style_copy_pattern(&st->style_stroke_pattern, pattern)) {
        return false;
    }
    uint32_t atlas_x = 0u;
    uint32_t atlas_y = 0u;
    if (!fs_image_handle_resolve_atlas_origin(core, &st->style_stroke_pattern.handle, &atlas_x, &atlas_y)) {
        return false;
    }
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_PATTERN;
    return true;
}

bool fs_style_set_shadow_blur(FS_Core* core, float blur_px) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(blur_px) || blur_px < 0.0f) {
        return false;
    }
    if (blur_px > FS_SHADOW_BLUR_MAX) {
        blur_px = FS_SHADOW_BLUR_MAX;
    }
    st->style_shadow_blur = blur_px;
    return true;
}

float fs_style_get_shadow_blur(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_shadow_blur) || st->style_shadow_blur < 0.0f) {
        return 0.0f;
    }
    return st->style_shadow_blur;
}

bool fs_style_set_shadow_offset(FS_Core* core, float offset_x, float offset_y) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(offset_x) || !isfinite(offset_y)) {
        return false;
    }
    st->style_shadow_offset_x = offset_x;
    st->style_shadow_offset_y = offset_y;
    return true;
}

float fs_style_get_shadow_offset_x(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_shadow_offset_x)) {
        return 0.0f;
    }
    return st->style_shadow_offset_x;
}

float fs_style_get_shadow_offset_y(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_shadow_offset_y)) {
        return 0.0f;
    }
    return st->style_shadow_offset_y;
}

bool fs_style_set_text_align(FS_Core* core, FS_TextAlign align) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (align) {
        case FS_TEXT_ALIGN_START:
        case FS_TEXT_ALIGN_LEFT:
        case FS_TEXT_ALIGN_CENTER:
        case FS_TEXT_ALIGN_RIGHT:
        case FS_TEXT_ALIGN_END:
            st->style_text_align = (uint8_t)align;
            return true;
        default:
            return false;
    }
}

FS_TextAlign fs_style_get_text_align(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_TEXT_ALIGN_START;
    }
    switch ((FS_TextAlign)st->style_text_align) {
        case FS_TEXT_ALIGN_LEFT:
        case FS_TEXT_ALIGN_CENTER:
        case FS_TEXT_ALIGN_RIGHT:
        case FS_TEXT_ALIGN_END:
            return (FS_TextAlign)st->style_text_align;
        case FS_TEXT_ALIGN_START:
        default:
            return FS_TEXT_ALIGN_START;
    }
}

bool fs_style_set_text_baseline(FS_Core* core, FS_TextBaseline baseline) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (baseline) {
        case FS_TEXT_BASELINE_TOP:
        case FS_TEXT_BASELINE_HANGING:
        case FS_TEXT_BASELINE_MIDDLE:
        case FS_TEXT_BASELINE_ALPHABETIC:
        case FS_TEXT_BASELINE_IDEOGRAPHIC:
        case FS_TEXT_BASELINE_BOTTOM:
            st->style_text_baseline = (uint8_t)baseline;
            return true;
        default:
            return false;
    }
}

FS_TextBaseline fs_style_get_text_baseline(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_TEXT_BASELINE_ALPHABETIC;
    }
    switch ((FS_TextBaseline)st->style_text_baseline) {
        case FS_TEXT_BASELINE_TOP:
        case FS_TEXT_BASELINE_HANGING:
        case FS_TEXT_BASELINE_MIDDLE:
        case FS_TEXT_BASELINE_IDEOGRAPHIC:
        case FS_TEXT_BASELINE_BOTTOM:
            return (FS_TextBaseline)st->style_text_baseline;
        case FS_TEXT_BASELINE_ALPHABETIC:
        default:
            return FS_TEXT_BASELINE_ALPHABETIC;
    }
}

bool fs_style_set_text_direction(FS_Core* core, FS_TextDirection direction) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (direction) {
        case FS_TEXT_DIRECTION_INHERIT:
        case FS_TEXT_DIRECTION_LTR:
        case FS_TEXT_DIRECTION_RTL:
            st->style_text_direction = (uint8_t)direction;
            return true;
        default:
            return false;
    }
}

FS_TextDirection fs_style_get_text_direction(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_TEXT_DIRECTION_LTR;
    }
    switch ((FS_TextDirection)st->style_text_direction) {
        case FS_TEXT_DIRECTION_INHERIT:
        case FS_TEXT_DIRECTION_LTR:
        case FS_TEXT_DIRECTION_RTL:
            return (FS_TextDirection)st->style_text_direction;
        default:
            return FS_TEXT_DIRECTION_LTR;
    }
}

bool fs_style_set_font_kerning(FS_Core* core, FS_FontKerning kerning) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (kerning) {
        case FS_FONT_KERNING_AUTO:
        case FS_FONT_KERNING_NORMAL:
        case FS_FONT_KERNING_NONE:
            st->style_font_kerning = (uint8_t)kerning;
            return true;
        default:
            return false;
    }
}

FS_FontKerning fs_style_get_font_kerning(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_FONT_KERNING_AUTO;
    }
    switch ((FS_FontKerning)st->style_font_kerning) {
        case FS_FONT_KERNING_AUTO:
        case FS_FONT_KERNING_NORMAL:
        case FS_FONT_KERNING_NONE:
            return (FS_FontKerning)st->style_font_kerning;
        default:
            return FS_FONT_KERNING_AUTO;
    }
}

bool fs_style_set_text_rendering(FS_Core* core, FS_TextRendering rendering) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (rendering) {
        case FS_TEXT_RENDERING_AUTO:
        case FS_TEXT_RENDERING_OPTIMIZE_SPEED:
        case FS_TEXT_RENDERING_OPTIMIZE_LEGIBILITY:
        case FS_TEXT_RENDERING_GEOMETRIC_PRECISION:
            st->style_text_rendering = (uint8_t)rendering;
            return true;
        default:
            return false;
    }
}

FS_TextRendering fs_style_get_text_rendering(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_TEXT_RENDERING_AUTO;
    }
    switch ((FS_TextRendering)st->style_text_rendering) {
        case FS_TEXT_RENDERING_AUTO:
        case FS_TEXT_RENDERING_OPTIMIZE_SPEED:
        case FS_TEXT_RENDERING_OPTIMIZE_LEGIBILITY:
        case FS_TEXT_RENDERING_GEOMETRIC_PRECISION:
            return (FS_TextRendering)st->style_text_rendering;
        default:
            return FS_TEXT_RENDERING_AUTO;
    }
}

bool fs_style_set_font_stretch(FS_Core* core, FS_FontStretch stretch) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (stretch) {
        case FS_FONT_STRETCH_ULTRA_CONDENSED:
        case FS_FONT_STRETCH_EXTRA_CONDENSED:
        case FS_FONT_STRETCH_CONDENSED:
        case FS_FONT_STRETCH_SEMI_CONDENSED:
        case FS_FONT_STRETCH_NORMAL:
        case FS_FONT_STRETCH_SEMI_EXPANDED:
        case FS_FONT_STRETCH_EXPANDED:
        case FS_FONT_STRETCH_EXTRA_EXPANDED:
        case FS_FONT_STRETCH_ULTRA_EXPANDED:
            st->style_font_stretch = (uint8_t)stretch;
            return true;
        default:
            return false;
    }
}

FS_FontStretch fs_style_get_font_stretch(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_FONT_STRETCH_NORMAL;
    }
    switch ((FS_FontStretch)st->style_font_stretch) {
        case FS_FONT_STRETCH_ULTRA_CONDENSED:
        case FS_FONT_STRETCH_EXTRA_CONDENSED:
        case FS_FONT_STRETCH_CONDENSED:
        case FS_FONT_STRETCH_SEMI_CONDENSED:
        case FS_FONT_STRETCH_NORMAL:
        case FS_FONT_STRETCH_SEMI_EXPANDED:
        case FS_FONT_STRETCH_EXPANDED:
        case FS_FONT_STRETCH_EXTRA_EXPANDED:
        case FS_FONT_STRETCH_ULTRA_EXPANDED:
            return (FS_FontStretch)st->style_font_stretch;
        default:
            return FS_FONT_STRETCH_NORMAL;
    }
}

bool fs_style_set_font_variant_caps(FS_Core* core, FS_FontVariantCaps variant_caps) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (variant_caps) {
        case FS_FONT_VARIANT_CAPS_NORMAL:
        case FS_FONT_VARIANT_CAPS_SMALL_CAPS:
        case FS_FONT_VARIANT_CAPS_ALL_SMALL_CAPS:
        case FS_FONT_VARIANT_CAPS_PETITE_CAPS:
        case FS_FONT_VARIANT_CAPS_ALL_PETITE_CAPS:
        case FS_FONT_VARIANT_CAPS_UNICASE:
        case FS_FONT_VARIANT_CAPS_TITLING_CAPS:
            st->style_font_variant_caps = (uint8_t)variant_caps;
            return true;
        default:
            return false;
    }
}

FS_FontVariantCaps fs_style_get_font_variant_caps(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_FONT_VARIANT_CAPS_NORMAL;
    }
    switch ((FS_FontVariantCaps)st->style_font_variant_caps) {
        case FS_FONT_VARIANT_CAPS_NORMAL:
        case FS_FONT_VARIANT_CAPS_SMALL_CAPS:
        case FS_FONT_VARIANT_CAPS_ALL_SMALL_CAPS:
        case FS_FONT_VARIANT_CAPS_PETITE_CAPS:
        case FS_FONT_VARIANT_CAPS_ALL_PETITE_CAPS:
        case FS_FONT_VARIANT_CAPS_UNICASE:
        case FS_FONT_VARIANT_CAPS_TITLING_CAPS:
            return (FS_FontVariantCaps)st->style_font_variant_caps;
        default:
            return FS_FONT_VARIANT_CAPS_NORMAL;
    }
}

bool fs_style_set_letter_spacing(FS_Core* core, float spacing_px) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(spacing_px)) {
        return false;
    }
    st->style_letter_spacing = spacing_px;
    return true;
}

float fs_style_get_letter_spacing(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_letter_spacing)) {
        return 0.0f;
    }
    return st->style_letter_spacing;
}

bool fs_style_set_word_spacing(FS_Core* core, float spacing_px) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(spacing_px)) {
        return false;
    }
    st->style_word_spacing = spacing_px;
    return true;
}

float fs_style_get_word_spacing(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_word_spacing)) {
        return 0.0f;
    }
    return st->style_word_spacing;
}

bool fs_style_set_image_smoothing_enabled(FS_Core* core, bool enabled) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->style_image_smoothing_enabled = enabled ? 1u : 0u;
    return true;
}

bool fs_style_get_image_smoothing_enabled(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return true;
    }
    return st->style_image_smoothing_enabled != 0u;
}

bool fs_style_set_image_smoothing_quality(FS_Core* core, FS_ImageSmoothingQuality quality) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (quality) {
        case FS_IMAGE_SMOOTHING_QUALITY_LOW:
        case FS_IMAGE_SMOOTHING_QUALITY_MEDIUM:
        case FS_IMAGE_SMOOTHING_QUALITY_HIGH:
            st->style_image_smoothing_quality = (uint8_t)quality;
            return true;
        default:
            return false;
    }
}

FS_ImageSmoothingQuality fs_style_get_image_smoothing_quality(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_IMAGE_SMOOTHING_QUALITY_LOW;
    }
    switch ((FS_ImageSmoothingQuality)st->style_image_smoothing_quality) {
        case FS_IMAGE_SMOOTHING_QUALITY_LOW:
        case FS_IMAGE_SMOOTHING_QUALITY_MEDIUM:
        case FS_IMAGE_SMOOTHING_QUALITY_HIGH:
            return (FS_ImageSmoothingQuality)st->style_image_smoothing_quality;
        default:
            return FS_IMAGE_SMOOTHING_QUALITY_LOW;
    }
}

void fs_style_clear_dash(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return;
    }
    free(st->style_dash_segments);
    st->style_dash_segments = NULL;
    st->style_dash_count = 0u;
    st->style_dash_offset = 0.0f;
}

bool fs_style_set_dash(FS_Core* core, const float* segments, uint32_t segment_count, float offset) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (!segments || segment_count == 0u) {
        fs_style_clear_dash(core);
        st->style_dash_offset = offset;
        return true;
    }
    float* sanitized = (float*)malloc((size_t)segment_count * sizeof(float));
    if (!sanitized) {
        return false;
    }
    uint32_t count = 0u;
    for (uint32_t i = 0u; i < segment_count; ++i) {
        const float v = segments[i];
        if (v > 1e-6f) {
            sanitized[count++] = v;
        }
    }
    if (count == 0u) {
        free(sanitized);
        fs_style_clear_dash(core);
        st->style_dash_offset = offset;
        return true;
    }
    float* shrunk = (float*)realloc(sanitized, (size_t)count * sizeof(float));
    if (!shrunk) {
        shrunk = sanitized;
    }
    free(st->style_dash_segments);
    st->style_dash_segments = shrunk;
    st->style_dash_count = count;
    st->style_dash_offset = offset;
    return true;
}

bool fs_style_get_dash(
    const FS_Core* core,
    float* out_segments,
    uint32_t max_segments,
    uint32_t* out_count,
    float* out_offset
) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!out_count) {
        return false;
    }
    if (!st) {
        *out_count = 0u;
        if (out_offset) {
            *out_offset = 0.0f;
        }
        return false;
    }
    const uint32_t count = st->style_dash_count;
    if (out_offset) {
        *out_offset = st->style_dash_offset;
    }
    *out_count = count;
    if (!out_segments || max_segments == 0u || count == 0u || !st->style_dash_segments) {
        return true;
    }
    const uint32_t copy_count = (count < max_segments) ? count : max_segments;
    memcpy(out_segments, st->style_dash_segments, (size_t)copy_count * sizeof(float));
    return true;
}
