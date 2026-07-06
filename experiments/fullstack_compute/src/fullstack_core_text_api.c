#include "fullstack_core_private.h"

#include <math.h>
#include <string.h>

static bool fs_push_text_command(
    FS_Core* core,
    const FS_GlyphEntry* glyph,
    float draw_x,
    float draw_y,
    float draw_w,
    float draw_h,
    uint32_t color,
    uint32_t extra_text_flags,
    float stroke_width
) {
    if (!core || !glyph || draw_w <= 0.0f || draw_h <= 0.0f) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    const float scale_y = (glyph->atlas_height > 0.0f) ? (draw_h / glyph->atlas_height) : 1.0f;
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = draw_x;
    cmd.p0[1] = draw_y;
    cmd.p0[2] = draw_w;
    cmd.p0[3] = draw_h;
    if (fs_transform_requires_oriented_quad(t)) {
        fs_command_set_oriented_quad_from_rect(&cmd, t, draw_x, draw_y, draw_w, draw_h);
        cmd.flags |= FS_RENDER_FLAG_ORIENTED_QUAD;
    } else {
        cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    }
    cmd.p1[0] = glyph->uv_min[0];
    cmd.p1[1] = glyph->uv_min[1];
    cmd.p1[2] = glyph->uv_max[0] - glyph->uv_min[0];
    cmd.p1[3] = glyph->uv_max[1] - glyph->uv_min[1];
    cmd.p2[0] = glyph->sdf_radius_px;
    cmd.p2[1] = glyph->sdf_onedge;
    cmd.p2[2] = glyph->sdf_pixel_dist_scale;
    cmd.p2[3] = stroke_width > 0.0f ? stroke_width : 0.0f;
    cmd.scalar = scale_y;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_TEXT;
    cmd.flags |= (glyph->text_flags | extra_text_flags);
    return fs_push_command(core, &cmd);
}

#define FS_TEXT_STYLE_COLOR_NONE 0u
#define FS_TEXT_STYLE_COLOR_FILL 1u
#define FS_TEXT_STYLE_COLOR_STROKE 2u

static uint32_t fs_resolve_text_draw_color(
    const FS_InternalState* st,
    const FS_GlyphEntry* glyph,
    uint32_t fallback_color,
    uint32_t style_color_mode,
    float draw_x,
    float draw_y,
    float draw_w,
    float draw_h
) {
    if (!st || style_color_mode == FS_TEXT_STYLE_COLOR_NONE) {
        return fallback_color;
    }
    if (glyph && (glyph->text_flags & FS_TEXT_FLAG_COLOR_GLYPH)) {
        return fallback_color;
    }
    const float cx = draw_x + draw_w * 0.5f;
    const float cy = draw_y + draw_h * 0.5f;
    if (style_color_mode == FS_TEXT_STYLE_COLOR_FILL) {
        return fs_style_resolve_fill_color_at(st, cx, cy);
    }
    if (style_color_mode == FS_TEXT_STYLE_COLOR_STROKE) {
        return fs_style_resolve_stroke_color_at(st, cx, cy);
    }
    return fallback_color;
}

bool fs_cmd_text_glyph(FS_Core* core, float x, float y, float w, float h, uint32_t codepoint, uint32_t color) {
    if (!core) {
        return false;
    }
    FS_GlyphEntry* glyph = NULL;
    float glyph_scale = 1.0f;
    if (!fs_find_or_create_glyph(core, codepoint, 0u, h > 0.0f ? h : 16.0f, &glyph, &glyph_scale) || !glyph) {
        return false;
    }

    const float default_w = glyph->atlas_width * glyph_scale;
    const float default_h = glyph->atlas_height * glyph_scale;
    const float draw_w = (w > 0.0f) ? w : default_w;
    const float draw_h = (h > 0.0f) ? h : default_h;
    if (draw_w <= 0.0f || draw_h <= 0.0f) {
        return true;
    }
    return fs_push_text_command(core, glyph, x, y, draw_w, draw_h, color, 0u, 0.0f);
}

static bool fs_cmd_text_utf8_internal(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    uint32_t color,
    float max_width,
    uint32_t extra_text_flags,
    float stroke_width,
    uint32_t style_color_mode
) {
    if (!core || !utf8) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (font_size_px <= 0.0f) {
        font_size_px = (st && st->style_font_size_px > 0.0f) ? st->style_font_size_px : 16.0f;
    }
    const FS_Transform2D* text_transform = st ? &st->current_transform : NULL;
    float line_advance = font_size_px * 1.25f;
    fs_resolve_text_vertical_metrics(st, font_size_px, NULL, NULL, &line_advance);
    const bool kerning_enabled = fs_is_text_kerning_enabled(st);
    const float stretch_x = fs_text_stretch_scale(st);
    const bool small_caps_enabled = fs_is_text_small_caps_enabled(st);
    const bool allow_snap =
        !fs_is_text_geometric_precision(st) &&
        !fs_transform_requires_oriented_quad(text_transform);
    const bool snap_x = allow_snap;
    const bool snap_y = false;
    const float letter_spacing = (st && isfinite(st->style_letter_spacing)) ? st->style_letter_spacing : 0.0f;
    const float word_spacing = (st && isfinite(st->style_word_spacing)) ? st->style_word_spacing : 0.0f;
    const bool has_letter_spacing = fabsf(letter_spacing) > 1e-6f;
    const bool has_word_spacing = fabsf(word_spacing) > 1e-6f;
    const bool text_fill_pattern =
        st &&
        style_color_mode == FS_TEXT_STYLE_COLOR_FILL &&
        st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN &&
        st->style_fill_pattern.handle.width > 0u &&
        st->style_fill_pattern.handle.height > 0u;
    const bool text_stroke_pattern =
        st &&
        style_color_mode == FS_TEXT_STYLE_COLOR_STROKE &&
        st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN &&
        st->style_stroke_pattern.handle.width > 0u &&
        st->style_stroke_pattern.handle.height > 0u;
    const bool text_pattern_per_fragment = text_fill_pattern || text_stroke_pattern;
    const uint32_t text_pattern_flag = text_pattern_per_fragment ? FS_RENDER_FLAG_PATTERN_SHADE : 0u;
    const uint32_t pattern_base_color = 0xFFFFFFFFu;

    const bool can_shape =
        FS_ENABLE_EXPERIMENTAL_SHAPING &&
        st &&
        st->image_font_count == 0u &&
        st->font_count == 1u &&
        st->fonts[0] &&
        st->font_backend &&
        st->font_backend->shape_text_utf8 &&
        st->font_backend->free_shaped_text &&
        st->font_backend->get_glyph_sdf_by_index &&
        !has_word_spacing &&
        kerning_enabled &&
        fabsf(stretch_x - 1.0f) <= 1e-6f &&
        !small_caps_enabled;

    float text_origin_x = x;
    float text_baseline_y = baseline_y;
    if (st) {
        FS_TextMetrics anchor_metrics = {0};
        if (fs_measure_text_utf8(core, font_size_px, utf8, max_width, &anchor_metrics)) {
            text_origin_x += fs_text_align_offset(st, &anchor_metrics);
            text_baseline_y += fs_text_baseline_offset(st, &anchor_metrics, font_size_px);
        }
    }
    if (allow_snap) {
        text_baseline_y = floorf(text_baseline_y + 0.5f);
    }

    if (can_shape && strchr(utf8, '\n') == NULL && strchr(utf8, '\r') == NULL) {
        const float bake_px = fs_get_text_bake_px(font_size_px);
        const float text_scale = font_size_px / bake_px;
        FS_ShapedTextRun run = {0};
        if (st->font_backend->shape_text_utf8(st->fonts[0], utf8, bake_px, &run)) {
            float pen_x = text_origin_x;
            float pen_y = text_baseline_y;
            bool has_prev_glyph = false;
            for (uint32_t i = 0u; i < run.glyph_count; ++i) {
                const FS_ShapedGlyph* shaped = &run.glyphs[i];
                if (has_prev_glyph && has_letter_spacing) {
                    pen_x += letter_spacing;
                }
                FS_GlyphEntry* glyph = NULL;
                float glyph_scale = 1.0f;
                if (!fs_find_or_create_glyph(core, shaped->glyph_index, 1u, font_size_px, &glyph, &glyph_scale) || !glyph) {
                    pen_x += ((float)shaped->x_advance_26d6 / 64.0f) * text_scale;
                    pen_y -= ((float)shaped->y_advance_26d6 / 64.0f) * text_scale;
                    has_prev_glyph = true;
                    continue;
                }

                if (glyph->atlas_width > 0.0f && glyph->atlas_height > 0.0f) {
                    float draw_x =
                        pen_x +
                        ((float)shaped->x_offset_26d6 / 64.0f) * text_scale +
                        glyph->bearing_x * glyph_scale;
                    float draw_y =
                        pen_y -
                        ((float)shaped->y_offset_26d6 / 64.0f) * text_scale +
                        glyph->bearing_y * glyph_scale;
                    if (snap_x) {
                        draw_x = floorf(draw_x + 0.5f);
                    }
                    if (snap_y) {
                        draw_y = floorf(draw_y + 0.5f);
                    }
                    const float draw_w = glyph->atlas_width * glyph_scale;
                    const float draw_h = glyph->atlas_height * glyph_scale;
                    if (max_width <= 0.0f || (draw_x - text_origin_x) <= max_width) {
                        const uint32_t draw_color = fs_resolve_text_draw_color(
                            st,
                            glyph,
                            color,
                            style_color_mode,
                            draw_x,
                            draw_y,
                            draw_w,
                            draw_h
                        );
                        if (!fs_push_text_command(
                                core,
                                glyph,
                                draw_x,
                                draw_y,
                                draw_w,
                                draw_h,
                                draw_color,
                                extra_text_flags,
                                stroke_width
                            )) {
                            st->font_backend->free_shaped_text(&run);
                            return false;
                        }
                    } else {
                        st->font_backend->free_shaped_text(&run);
                        return true;
                    }
                }
                pen_x += ((float)shaped->x_advance_26d6 / 64.0f) * text_scale;
                pen_y -= ((float)shaped->y_advance_26d6 / 64.0f) * text_scale;
                has_prev_glyph = true;
            }
            st->font_backend->free_shaped_text(&run);
            return true;
        }
    }

    float pen_x = text_origin_x;
    float pen_y = text_baseline_y;
    const char* ptr = utf8;
    uint32_t prev_cp = 0u;
    bool has_prev_cp = false;
    uint8_t prev_font_slot = 0u;
    float prev_font_px = font_size_px;
    bool line_has_glyph = false;

    while (*ptr != '\0') {
        if (st && st->image_font_count > 0u) {
            uint32_t image_font_id = 0u;
            uint32_t image_glyph_id = 0u;
            size_t image_seq_bytes = 0u;
            if (fs_find_image_sequence_match(st, ptr, &image_font_id, &image_glyph_id, &image_seq_bytes) &&
                image_seq_bytes > 0u) {
                uint32_t image_key = 0u;
                bool has_key = fs_pack_image_glyph_key(image_font_id, image_glyph_id, &image_key);
                FS_GlyphEntry* image_glyph = NULL;
                float image_scale = 1.0f;
                bool loaded = has_key &&
                              fs_find_or_create_glyph(
                                  core,
                                  image_key,
                                  FS_IMAGE_FONT_KIND,
                                  font_size_px,
                                  &image_glyph,
                                  &image_scale
                              ) &&
                              image_glyph;
                if (loaded) {
                    if (line_has_glyph && has_letter_spacing) {
                        pen_x += letter_spacing;
                    }
                    if (max_width > 0.0f && (pen_x - text_origin_x) > max_width) {
                        break;
                    }
                    if (image_glyph->atlas_width > 0.0f && image_glyph->atlas_height > 0.0f) {
                        float draw_x = pen_x + image_glyph->bearing_x * image_scale * stretch_x;
                        float draw_y = pen_y + image_glyph->bearing_y * image_scale;
                        if (snap_x) {
                            draw_x = floorf(draw_x + 0.5f);
                        }
                        if (snap_y) {
                            draw_y = floorf(draw_y + 0.5f);
                        }
                        const float draw_w = image_glyph->atlas_width * image_scale * stretch_x;
                        const float draw_h = image_glyph->atlas_height * image_scale;
                        const uint32_t draw_color = fs_resolve_text_draw_color(
                            st,
                            image_glyph,
                            color,
                            style_color_mode,
                            draw_x,
                            draw_y,
                            draw_w,
                            draw_h
                        );
                        const bool color_glyph = (image_glyph->text_flags & FS_TEXT_FLAG_COLOR_GLYPH) != 0u;
                        const uint32_t final_color = (text_pattern_per_fragment && !color_glyph) ? pattern_base_color : draw_color;
                        uint32_t draw_text_flags = extra_text_flags;
                        if (text_pattern_per_fragment && !color_glyph) {
                            draw_text_flags |= text_pattern_flag;
                        }
                        if (!fs_push_text_command(
                                core,
                                image_glyph,
                                draw_x,
                                draw_y,
                                draw_w,
                                draw_h,
                                final_color,
                                draw_text_flags,
                                stroke_width
                            )) {
                            return false;
                        }
                    }
                    pen_x += image_glyph->advance * image_scale * stretch_x;
                    line_has_glyph = true;
                } else {
                    (void)fs_push_missing_image_glyph(st, image_font_id, image_glyph_id, ptr, image_seq_bytes);
                }
                ptr += image_seq_bytes;
                has_prev_cp = false;
                continue;
            }
        }

        uint32_t cp = fs_decode_utf8(&ptr);
        if (cp == 0u) {
            break;
        }
        if (cp == '\r') {
            continue;
        }
        if (cp == '\n') {
            pen_x = text_origin_x;
            pen_y += line_advance;
            has_prev_cp = false;
            line_has_glyph = false;
            continue;
        }

        uint32_t glyph_cp = cp;
        float variant_size_scale = 1.0f;
        fs_text_variant_map_codepoint(st, cp, &glyph_cp, &variant_size_scale);
        const float glyph_font_px = font_size_px * variant_size_scale;

        FS_GlyphEntry* glyph = NULL;
        float glyph_scale = 1.0f;
        if (!fs_find_or_create_glyph(core, glyph_cp, 0u, glyph_font_px, &glyph, &glyph_scale) || !glyph) {
            has_prev_cp = false;
            continue;
        }

        if (kerning_enabled && has_prev_cp && prev_font_slot == glyph->font_slot) {
            const float kern_font_px = (prev_font_px < glyph_font_px) ? prev_font_px : glyph_font_px;
            pen_x += fs_get_kerning_advance(core, prev_cp, glyph_cp, kern_font_px, glyph->font_slot) * stretch_x;
        }
        if (line_has_glyph && has_letter_spacing) {
            pen_x += letter_spacing;
        }

        if (max_width > 0.0f && (pen_x - text_origin_x) > max_width) {
            break;
        }

        if (glyph->atlas_width > 0.0f && glyph->atlas_height > 0.0f) {
            float draw_x = pen_x + glyph->bearing_x * glyph_scale * stretch_x;
            float draw_y = pen_y + glyph->bearing_y * glyph_scale;
            if (snap_x) {
                draw_x = floorf(draw_x + 0.5f);
            }
            if (snap_y) {
                draw_y = floorf(draw_y + 0.5f);
            }
            const float draw_w = glyph->atlas_width * glyph_scale * stretch_x;
            const float draw_h = glyph->atlas_height * glyph_scale;
            const uint32_t draw_color = fs_resolve_text_draw_color(
                st,
                glyph,
                color,
                style_color_mode,
                draw_x,
                draw_y,
                draw_w,
                draw_h
            );
            const bool color_glyph = (glyph->text_flags & FS_TEXT_FLAG_COLOR_GLYPH) != 0u;
            const uint32_t final_color = (text_pattern_per_fragment && !color_glyph) ? pattern_base_color : draw_color;
            uint32_t draw_text_flags = extra_text_flags;
            if (text_pattern_per_fragment && !color_glyph) {
                draw_text_flags |= text_pattern_flag;
            }
            if (!fs_push_text_command(
                    core,
                    glyph,
                    draw_x,
                    draw_y,
                    draw_w,
                    draw_h,
                    final_color,
                    draw_text_flags,
                    stroke_width
                )) {
                return false;
            }
        }
        pen_x += glyph->advance * glyph_scale * stretch_x;
        if (has_word_spacing && fs_is_word_spacing_codepoint(cp)) {
            pen_x += word_spacing;
        }
        prev_cp = glyph_cp;
        prev_font_slot = glyph->font_slot;
        prev_font_px = glyph_font_px;
        has_prev_cp = true;
        line_has_glyph = true;
    }
    return true;
}

bool fs_cmd_text_utf8(FS_Core* core, float x, float baseline_y, float font_size_px, const char* utf8, uint32_t color, float max_width) {
    return fs_cmd_text_utf8_internal(
        core,
        x,
        baseline_y,
        font_size_px,
        utf8,
        color,
        max_width,
        0u,
        0.0f,
        FS_TEXT_STYLE_COLOR_NONE
    );
}

bool fs_cmd_stroke_text_utf8(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    uint32_t color,
    float max_width,
    float stroke_width
) {
    if (!core || !utf8) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    const float resolved_width = fs_style_resolve_line_width(st, stroke_width);
    if (resolved_width <= 0.0f) {
        return false;
    }
    return fs_cmd_text_utf8_internal(
        core,
        x,
        baseline_y,
        font_size_px,
        utf8,
        color,
        max_width,
        FS_TEXT_FLAG_STROKE,
        resolved_width,
        FS_TEXT_STYLE_COLOR_NONE
    );
}

bool fs_fill_text_utf8(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    float max_width
) {
    FS_InternalState* st = fs_state(core);
    if (st && (
            (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT &&
             st->style_fill_linear_gradient.stop_count >= 2u) ||
            (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT &&
             st->style_fill_radial_gradient.stop_count >= 2u))) {
        const uint32_t fallback_color = st->style_fill_color_rgba8;
        return fs_cmd_text_utf8_internal(
            core,
            x,
            baseline_y,
            font_size_px,
            utf8,
            fallback_color,
            max_width,
            0u,
            0.0f,
            FS_TEXT_STYLE_COLOR_FILL
        );
    }
    const uint32_t color = fs_style_resolve_fill_color_at(st, x, baseline_y);
    return fs_cmd_text_utf8(core, x, baseline_y, font_size_px, utf8, color, max_width);
}

bool fs_stroke_text_utf8(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    float max_width,
    float stroke_width
) {
    FS_InternalState* st = fs_state(core);
    const float resolved_width = fs_style_resolve_line_width(st, stroke_width);
    if (resolved_width <= 0.0f) {
        return false;
    }
    if (st && (
            (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT &&
             st->style_stroke_linear_gradient.stop_count >= 2u) ||
            (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT &&
             st->style_stroke_radial_gradient.stop_count >= 2u))) {
        const uint32_t fallback_color = st->style_stroke_color_rgba8;
        return fs_cmd_text_utf8_internal(
            core,
            x,
            baseline_y,
            font_size_px,
            utf8,
            fallback_color,
            max_width,
            FS_TEXT_FLAG_STROKE,
            resolved_width,
            FS_TEXT_STYLE_COLOR_STROKE
        );
    }
    const uint32_t color = fs_style_resolve_stroke_color_at(st, x, baseline_y);
    return fs_cmd_stroke_text_utf8(
        core,
        x,
        baseline_y,
        font_size_px,
        utf8,
        color,
        max_width,
        resolved_width
    );
}

bool fs_measure_text_utf8(
    FS_Core* core,
    float font_size_px,
    const char* utf8,
    float max_width,
    FS_TextMetrics* out_metrics
) {
    if (!core || !utf8 || !out_metrics) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (font_size_px <= 0.0f) {
        font_size_px = (st->style_font_size_px > 0.0f) ? st->style_font_size_px : 16.0f;
    }
    memset(out_metrics, 0, sizeof(*out_metrics));

    float em_ascent = 0.0f;
    float em_descent = 0.0f;
    float line_advance = font_size_px * 1.25f;
    fs_resolve_text_vertical_metrics(st, font_size_px, &em_ascent, &em_descent, &line_advance);

    float pen_x = 0.0f;
    float pen_y = 0.0f;
    float line_start_x = 0.0f;
    float max_line_width = 0.0f;
    uint32_t line_count = 1u;
    uint32_t glyph_count = 0u;

    bool has_bounds = false;
    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
    const float letter_spacing = (isfinite(st->style_letter_spacing)) ? st->style_letter_spacing : 0.0f;
    const float word_spacing = (isfinite(st->style_word_spacing)) ? st->style_word_spacing : 0.0f;
    const bool has_letter_spacing = fabsf(letter_spacing) > 1e-6f;
    const bool has_word_spacing = fabsf(word_spacing) > 1e-6f;
    const bool kerning_enabled = fs_is_text_kerning_enabled(st);
    const float stretch_x = fs_text_stretch_scale(st);

    const bool snap_x = !fs_is_text_geometric_precision(st);
    const bool snap_y = false;
    const char* ptr = utf8;
    uint32_t prev_cp = 0u;
    bool has_prev_cp = false;
    uint8_t prev_font_slot = 0u;
    float prev_font_px = font_size_px;
    bool line_has_glyph = false;

    while (*ptr != '\0') {
        if (st->image_font_count > 0u) {
            uint32_t image_font_id = 0u;
            uint32_t image_glyph_id = 0u;
            size_t image_seq_bytes = 0u;
            if (fs_find_image_sequence_match(st, ptr, &image_font_id, &image_glyph_id, &image_seq_bytes) &&
                image_seq_bytes > 0u) {
                uint32_t image_key = 0u;
                bool has_key = fs_pack_image_glyph_key(image_font_id, image_glyph_id, &image_key);
                FS_GlyphEntry* image_glyph = NULL;
                float image_scale = 1.0f;
                bool loaded = has_key &&
                              fs_find_or_create_glyph(
                                  core,
                                  image_key,
                                  FS_IMAGE_FONT_KIND,
                                  font_size_px,
                                  &image_glyph,
                                  &image_scale
                              ) &&
                              image_glyph;
                if (loaded) {
                    if (line_has_glyph && has_letter_spacing) {
                        pen_x += letter_spacing;
                    }
                    if (max_width > 0.0f && (pen_x - line_start_x) > max_width) {
                        break;
                    }
                    if (image_glyph->atlas_width > 0.0f && image_glyph->atlas_height > 0.0f) {
                        float draw_x = pen_x + image_glyph->bearing_x * image_scale * stretch_x;
                        float draw_y = pen_y + image_glyph->bearing_y * image_scale;
                        if (snap_x) {
                            draw_x = floorf(draw_x + 0.5f);
                        }
                        if (snap_y) {
                            draw_y = floorf(draw_y + 0.5f);
                        }
                        const float draw_w = image_glyph->atlas_width * image_scale * stretch_x;
                        const float draw_h = image_glyph->atlas_height * image_scale;
                        if (draw_w > 0.0f && draw_h > 0.0f) {
                            const float bx0 = draw_x;
                            const float by0 = draw_y;
                            const float bx1 = draw_x + draw_w;
                            const float by1 = draw_y + draw_h;
                            if (!has_bounds) {
                                min_x = bx0;
                                min_y = by0;
                                max_x = bx1;
                                max_y = by1;
                                has_bounds = true;
                            } else {
                                if (bx0 < min_x) min_x = bx0;
                                if (by0 < min_y) min_y = by0;
                                if (bx1 > max_x) max_x = bx1;
                                if (by1 > max_y) max_y = by1;
                            }
                            glyph_count += 1u;
                        }
                    }
                    pen_x += image_glyph->advance * image_scale * stretch_x;
                    const float line_width = pen_x - line_start_x;
                    if (line_width > max_line_width) {
                        max_line_width = line_width;
                    }
                    line_has_glyph = true;
                }
                ptr += image_seq_bytes;
                has_prev_cp = false;
                continue;
            }
        }

        uint32_t cp = fs_decode_utf8(&ptr);
        if (cp == 0u) {
            break;
        }
        if (cp == '\r') {
            continue;
        }
        if (cp == '\n') {
            const float line_width = pen_x - line_start_x;
            if (line_width > max_line_width) {
                max_line_width = line_width;
            }
            pen_x = 0.0f;
            pen_y += line_advance;
            line_start_x = pen_x;
            has_prev_cp = false;
            line_has_glyph = false;
            line_count += 1u;
            continue;
        }

        uint32_t glyph_cp = cp;
        float variant_size_scale = 1.0f;
        fs_text_variant_map_codepoint(st, cp, &glyph_cp, &variant_size_scale);
        const float glyph_font_px = font_size_px * variant_size_scale;

        FS_GlyphEntry* glyph = NULL;
        float glyph_scale = 1.0f;
        if (!fs_find_or_create_glyph(core, glyph_cp, 0u, glyph_font_px, &glyph, &glyph_scale) || !glyph) {
            has_prev_cp = false;
            continue;
        }

        if (kerning_enabled && has_prev_cp && prev_font_slot == glyph->font_slot) {
            const float kern_font_px = (prev_font_px < glyph_font_px) ? prev_font_px : glyph_font_px;
            pen_x += fs_get_kerning_advance(core, prev_cp, glyph_cp, kern_font_px, glyph->font_slot) * stretch_x;
        }
        if (line_has_glyph && has_letter_spacing) {
            pen_x += letter_spacing;
        }

        if (max_width > 0.0f && (pen_x - line_start_x) > max_width) {
            break;
        }

        if (glyph->atlas_width > 0.0f && glyph->atlas_height > 0.0f) {
            float draw_x = pen_x + glyph->bearing_x * glyph_scale * stretch_x;
            float draw_y = pen_y + glyph->bearing_y * glyph_scale;
            if (snap_x) {
                draw_x = floorf(draw_x + 0.5f);
            }
            if (snap_y) {
                draw_y = floorf(draw_y + 0.5f);
            }
            const float draw_w = glyph->atlas_width * glyph_scale * stretch_x;
            const float draw_h = glyph->atlas_height * glyph_scale;
            const float bx0 = draw_x;
            const float by0 = draw_y;
            const float bx1 = draw_x + draw_w;
            const float by1 = draw_y + draw_h;
            if (!has_bounds) {
                min_x = bx0;
                min_y = by0;
                max_x = bx1;
                max_y = by1;
                has_bounds = true;
            } else {
                if (bx0 < min_x) min_x = bx0;
                if (by0 < min_y) min_y = by0;
                if (bx1 > max_x) max_x = bx1;
                if (by1 > max_y) max_y = by1;
            }
            glyph_count += 1u;
        }

        pen_x += glyph->advance * glyph_scale * stretch_x;
        if (has_word_spacing && fs_is_word_spacing_codepoint(cp)) {
            pen_x += word_spacing;
        }
        const float line_width = pen_x - line_start_x;
        if (line_width > max_line_width) {
            max_line_width = line_width;
        }
        prev_cp = glyph_cp;
        prev_font_slot = glyph->font_slot;
        prev_font_px = glyph_font_px;
        has_prev_cp = true;
        line_has_glyph = true;
    }

    out_metrics->width = max_line_width;
    out_metrics->glyph_count = glyph_count;
    out_metrics->line_count = line_count;
    out_metrics->em_height_ascent = em_ascent;
    out_metrics->em_height_descent = em_descent;
    if (has_bounds) {
        out_metrics->actual_bounding_box_left = -min_x;
        out_metrics->actual_bounding_box_right = max_x;
        out_metrics->actual_bounding_box_ascent = -min_y;
        out_metrics->actual_bounding_box_descent = max_y;
    } else {
        out_metrics->actual_bounding_box_left = 0.0f;
        out_metrics->actual_bounding_box_right = 0.0f;
        out_metrics->actual_bounding_box_ascent = 0.0f;
        out_metrics->actual_bounding_box_descent = 0.0f;
    }
    return true;
}
