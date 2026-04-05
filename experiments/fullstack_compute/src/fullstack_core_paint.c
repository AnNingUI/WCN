#include "fullstack_core_private.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint8_t fs_color_r_u8(uint32_t color) {
    return (uint8_t)(color & 0xFFu);
}

static uint8_t fs_color_g_u8(uint32_t color) {
    return (uint8_t)((color >> 8u) & 0xFFu);
}

static uint8_t fs_color_b_u8(uint32_t color) {
    return (uint8_t)((color >> 16u) & 0xFFu);
}

static uint8_t fs_color_a_u8(uint32_t color) {
    return (uint8_t)((color >> 24u) & 0xFFu);
}

static uint32_t fs_color_rgba8_pack_u8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t)a << 24u) | ((uint32_t)b << 16u) | ((uint32_t)g << 8u) | (uint32_t)r;
}

static uint32_t fs_gradient_stops_sample_rgba8(const FS_GradientStop* stops, uint32_t stop_count, float t) {
    if (!stops || stop_count == 0u) {
        return 0xFF000000u;
    }
    if (stop_count == 1u) {
        return stops[0].color_rgba8;
    }
    if (t <= stops[0].offset_0_to_1) {
        return stops[0].color_rgba8;
    }
    const uint32_t last_index = stop_count - 1u;
    if (t >= stops[last_index].offset_0_to_1) {
        return stops[last_index].color_rgba8;
    }
    for (uint32_t i = 1u; i < stop_count; ++i) {
        const FS_GradientStop* b = &stops[i];
        if (t > b->offset_0_to_1) {
            continue;
        }
        const FS_GradientStop* a = &stops[i - 1u];
        const float denom = b->offset_0_to_1 - a->offset_0_to_1;
        float k = 0.0f;
        if (denom > 1e-8f) {
            k = (t - a->offset_0_to_1) / denom;
        }
        if (k < 0.0f) {
            k = 0.0f;
        } else if (k > 1.0f) {
            k = 1.0f;
        }
        const float ar = (float)fs_color_r_u8(a->color_rgba8);
        const float ag = (float)fs_color_g_u8(a->color_rgba8);
        const float ab = (float)fs_color_b_u8(a->color_rgba8);
        const float aa = (float)fs_color_a_u8(a->color_rgba8);
        const float br = (float)fs_color_r_u8(b->color_rgba8);
        const float bg = (float)fs_color_g_u8(b->color_rgba8);
        const float bb = (float)fs_color_b_u8(b->color_rgba8);
        const float ba = (float)fs_color_a_u8(b->color_rgba8);
        const uint8_t r = (uint8_t)lroundf(ar + (br - ar) * k);
        const uint8_t g = (uint8_t)lroundf(ag + (bg - ag) * k);
        const uint8_t bch = (uint8_t)lroundf(ab + (bb - ab) * k);
        const uint8_t ach = (uint8_t)lroundf(aa + (ba - aa) * k);
        return fs_color_rgba8_pack_u8(r, g, bch, ach);
    }
    return stops[last_index].color_rgba8;
}

bool fs_image_handle_resolve_atlas_origin(
    const FS_Core* core,
    const FS_ImageHandle* handle,
    uint32_t* out_atlas_x,
    uint32_t* out_atlas_y
) {
    if (!core || !handle || !out_atlas_x || !out_atlas_y) {
        return false;
    }
    if (handle->width == 0u || handle->height == 0u) {
        return false;
    }
    if (handle->layer >= core->image_atlas_layers) {
        return false;
    }
    if (handle->generation != core->image_atlas_generation[handle->layer]) {
        return false;
    }

    uint32_t atlas_x = handle->atlas_x;
    uint32_t atlas_y = handle->atlas_y;
    if (atlas_x + handle->width > core->image_atlas_width ||
        atlas_y + handle->height > core->image_atlas_height) {
        if (!isfinite(handle->uv_min[0]) || !isfinite(handle->uv_min[1])) {
            return false;
        }
        atlas_x = (uint32_t)floorf(handle->uv_min[0] * (float)core->image_atlas_width + 0.5f);
        atlas_y = (uint32_t)floorf(handle->uv_min[1] * (float)core->image_atlas_height + 0.5f);
    }
    if (atlas_x + handle->width > core->image_atlas_width ||
        atlas_y + handle->height > core->image_atlas_height) {
        return false;
    }
    *out_atlas_x = atlas_x;
    *out_atlas_y = atlas_y;
    return true;
}

void fs_command_state_clear_pattern(FS_CommandStateGPU* state) {
    if (!state) {
        return;
    }
    memset(state->pattern_inv0, 0, sizeof(state->pattern_inv0));
    memset(state->pattern_inv1, 0, sizeof(state->pattern_inv1));
    memset(state->pattern_meta, 0, sizeof(state->pattern_meta));
}

bool fs_command_state_set_pattern(FS_Core* core, FS_CommandStateGPU* state, const FS_StylePattern* pattern) {
    if (!core || !state || !pattern) {
        return false;
    }
    if (pattern->handle.width == 0u || pattern->handle.height == 0u || !pattern->inv_valid) {
        return false;
    }
    uint32_t atlas_x = 0u;
    uint32_t atlas_y = 0u;
    if (!fs_image_handle_resolve_atlas_origin(core, &pattern->handle, &atlas_x, &atlas_y)) {
        return false;
    }
    state->pattern_inv0[0] = pattern->inv_xform[0];
    state->pattern_inv0[1] = pattern->inv_xform[1];
    state->pattern_inv0[2] = pattern->inv_xform[2];
    state->pattern_inv0[3] = pattern->inv_xform[3];
    state->pattern_inv1[0] = pattern->inv_xform[4];
    state->pattern_inv1[1] = pattern->inv_xform[5];
    state->pattern_inv1[2] = (float)atlas_x;
    state->pattern_inv1[3] = (float)atlas_y;
    state->pattern_meta[0] = (float)pattern->handle.width;
    state->pattern_meta[1] = (float)pattern->handle.height;
    state->pattern_meta[2] = (float)pattern->handle.layer;
    state->pattern_meta[3] = (float)pattern->repeat_mode;
    return true;
}

bool fs_style_pattern_sample_rgba8(
    const FS_InternalState* st,
    const FS_StylePattern* pattern,
    float px,
    float py,
    uint32_t* out_color
) {
    if (!st || !pattern || !out_color || !isfinite(px) || !isfinite(py)) {
        return false;
    }
    const FS_Core* core = st->owner_core;
    if (!core || !core->image_atlas_shadow_rgba || core->image_atlas_shadow_size == 0u) {
        return false;
    }
    const uint32_t w = pattern->handle.width;
    const uint32_t h = pattern->handle.height;
    if (w == 0u || h == 0u) {
        return false;
    }
    uint32_t atlas_x = 0u;
    uint32_t atlas_y = 0u;
    if (!fs_image_handle_resolve_atlas_origin(core, &pattern->handle, &atlas_x, &atlas_y)) {
        return false;
    }

    const uint8_t repeat_mode = pattern->repeat_mode;
    const bool repeat_x =
        repeat_mode == (uint8_t)FS_PATTERN_REPEAT || repeat_mode == (uint8_t)FS_PATTERN_REPEAT_X;
    const bool repeat_y =
        repeat_mode == (uint8_t)FS_PATTERN_REPEAT || repeat_mode == (uint8_t)FS_PATTERN_REPEAT_Y;

    float pattern_x = px;
    float pattern_y = py;
    if (pattern->inv_valid) {
        fs_affine_apply_point_2d(pattern->inv_xform, px, py, &pattern_x, &pattern_y);
    }
    int64_t sx = (int64_t)floorf(pattern_x);
    int64_t sy = (int64_t)floorf(pattern_y);
    if (!repeat_x) {
        if (sx < 0 || sx >= (int64_t)w) {
            return false;
        }
    } else {
        sx %= (int64_t)w;
        if (sx < 0) {
            sx += (int64_t)w;
        }
    }
    if (!repeat_y) {
        if (sy < 0 || sy >= (int64_t)h) {
            return false;
        }
    } else {
        sy %= (int64_t)h;
        if (sy < 0) {
            sy += (int64_t)h;
        }
    }

    const uint32_t sample_x = atlas_x + (uint32_t)sx;
    const uint32_t sample_y = atlas_y + (uint32_t)sy;
    if (sample_x >= core->image_atlas_width || sample_y >= core->image_atlas_height) {
        return false;
    }
    const size_t atlas_w = (size_t)core->image_atlas_width;
    const size_t atlas_h = (size_t)core->image_atlas_height;
    const size_t layer_stride_px = atlas_w * atlas_h;
    const size_t px_index =
        (size_t)pattern->handle.layer * layer_stride_px +
        (size_t)sample_y * atlas_w +
        (size_t)sample_x;
    const uint8_t* rgba = core->image_atlas_shadow_rgba + px_index * 4u;
    *out_color = fs_color_rgba8_pack_u8(rgba[0], rgba[1], rgba[2], rgba[3]);
    return true;
}

uint32_t fs_linear_gradient_sample_rgba8(const FS_StyleLinearGradient* grad, float px, float py) {
    if (!grad) {
        return 0xFF000000u;
    }
    const float dx = grad->x1 - grad->x0;
    const float dy = grad->y1 - grad->y0;
    const float len_sq = dx * dx + dy * dy;
    float t = 0.0f;
    if (len_sq > 1e-8f) {
        t = ((px - grad->x0) * dx + (py - grad->y0) * dy) / len_sq;
    }
    return fs_gradient_stops_sample_rgba8(grad->stops, grad->stop_count, t);
}

uint32_t fs_conic_gradient_sample_rgba8(const FS_StyleConicGradient* grad, float px, float py) {
    if (!grad) {
        return 0xFF000000u;
    }
    const float two_pi = 6.2831853071795864769f;
    float angle = atan2f(py - grad->cy, px - grad->cx) - grad->start_angle_radians;
    angle = fmodf(angle, two_pi);
    if (angle < 0.0f) {
        angle += two_pi;
    }
    const float t = angle / two_pi;
    return fs_gradient_stops_sample_rgba8(grad->stops, grad->stop_count, t);
}

uint32_t fs_radial_gradient_sample_rgba8(const FS_StyleRadialGradient* grad, float px, float py) {
    if (!grad) {
        return 0xFF000000u;
    }
    const float sx = px - grad->x0;
    const float sy = py - grad->y0;
    const float dx = grad->x1 - grad->x0;
    const float dy = grad->y1 - grad->y0;
    const float dr = grad->r1 - grad->r0;
    const float a = dx * dx + dy * dy - dr * dr;
    const float b = -2.0f * (sx * dx + sy * dy + grad->r0 * dr);
    const float c = sx * sx + sy * sy - grad->r0 * grad->r0;

    float t = 0.0f;
    if (fabsf(a) <= 1e-8f) {
        if (fabsf(b) > 1e-8f) {
            t = -c / b;
        }
    } else {
        const float disc = b * b - 4.0f * a * c;
        if (disc < 0.0f) {
            t = -b / (2.0f * a);
        } else {
            const float sqrt_disc = sqrtf(disc);
            const float inv_2a = 0.5f / a;
            const float t0 = (-b - sqrt_disc) * inv_2a;
            const float t1 = (-b + sqrt_disc) * inv_2a;
            const bool t0_in = (t0 >= 0.0f && t0 <= 1.0f);
            const bool t1_in = (t1 >= 0.0f && t1 <= 1.0f);
            if (t0_in && t1_in) {
                t = (t0 > t1) ? t0 : t1;
            } else if (t0_in) {
                t = t0;
            } else if (t1_in) {
                t = t1;
            } else {
                const float d0 = fabsf(t0 - 0.5f);
                const float d1 = fabsf(t1 - 0.5f);
                t = (d0 <= d1) ? t0 : t1;
            }
        }
    }
    if (!isfinite(t)) {
        t = 0.0f;
    }
    return fs_gradient_stops_sample_rgba8(grad->stops, grad->stop_count, t);
}

uint32_t fs_style_resolve_fill_color_at(const FS_InternalState* st, float px, float py) {
    if (!st) {
        return 0xFF000000u;
    }
    if (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT &&
        st->style_fill_linear_gradient.stop_count >= 2u) {
        return fs_linear_gradient_sample_rgba8(&st->style_fill_linear_gradient, px, py);
    }
    if (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT &&
        st->style_fill_radial_gradient.stop_count >= 2u) {
        return fs_radial_gradient_sample_rgba8(&st->style_fill_radial_gradient, px, py);
    }
    if (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_CONIC_GRADIENT &&
        st->style_fill_conic_gradient.stop_count >= 2u) {
        return fs_conic_gradient_sample_rgba8(&st->style_fill_conic_gradient, px, py);
    }
    if (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN) {
        uint32_t sampled = 0u;
        if (fs_style_pattern_sample_rgba8(st, &st->style_fill_pattern, px, py, &sampled)) {
            return sampled;
        }
        return 0u;
    }
    return st->style_fill_color_rgba8;
}

uint32_t fs_style_resolve_stroke_color_at(const FS_InternalState* st, float px, float py) {
    if (!st) {
        return 0xFF000000u;
    }
    if (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT &&
        st->style_stroke_linear_gradient.stop_count >= 2u) {
        return fs_linear_gradient_sample_rgba8(&st->style_stroke_linear_gradient, px, py);
    }
    if (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT &&
        st->style_stroke_radial_gradient.stop_count >= 2u) {
        return fs_radial_gradient_sample_rgba8(&st->style_stroke_radial_gradient, px, py);
    }
    if (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_CONIC_GRADIENT &&
        st->style_stroke_conic_gradient.stop_count >= 2u) {
        return fs_conic_gradient_sample_rgba8(&st->style_stroke_conic_gradient, px, py);
    }
    if (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN) {
        uint32_t sampled = 0u;
        if (fs_style_pattern_sample_rgba8(st, &st->style_stroke_pattern, px, py, &sampled)) {
            return sampled;
        }
        return 0u;
    }
    return st->style_stroke_color_rgba8;
}

FS_LinearGradient* fs_linear_gradient_create(float x0, float y0, float x1, float y1) {
    if (!isfinite(x0) || !isfinite(y0) || !isfinite(x1) || !isfinite(y1)) {
        return NULL;
    }
    FS_LinearGradient* gradient = (FS_LinearGradient*)calloc(1u, sizeof(FS_LinearGradient));
    if (!gradient) {
        return NULL;
    }
    gradient->x0 = x0;
    gradient->y0 = y0;
    gradient->x1 = x1;
    gradient->y1 = y1;
    return gradient;
}

void fs_linear_gradient_destroy(FS_LinearGradient* gradient) {
    if (!gradient) {
        return;
    }
    free(gradient->stops);
    gradient->stops = NULL;
    gradient->stop_count = 0u;
    gradient->stop_capacity = 0u;
    free(gradient);
}

bool fs_linear_gradient_add_color_stop(FS_LinearGradient* gradient, float offset_0_to_1, uint32_t color_rgba8) {
    if (!gradient || !isfinite(offset_0_to_1) || offset_0_to_1 < 0.0f || offset_0_to_1 > 1.0f) {
        return false;
    }
    if (gradient->stop_count + 1u > gradient->stop_capacity) {
        uint32_t new_capacity = gradient->stop_capacity ? gradient->stop_capacity * 2u : 8u;
        if (new_capacity < gradient->stop_count + 1u) {
            new_capacity = gradient->stop_count + 1u;
        }
        FS_GradientStop* grown =
            (FS_GradientStop*)realloc(gradient->stops, (size_t)new_capacity * sizeof(FS_GradientStop));
        if (!grown) {
            return false;
        }
        gradient->stops = grown;
        gradient->stop_capacity = new_capacity;
    }
    uint32_t insert_at = gradient->stop_count;
    while (insert_at > 0u && gradient->stops[insert_at - 1u].offset_0_to_1 > offset_0_to_1) {
        insert_at -= 1u;
    }
    if (insert_at < gradient->stop_count) {
        memmove(
            &gradient->stops[insert_at + 1u],
            &gradient->stops[insert_at],
            (size_t)(gradient->stop_count - insert_at) * sizeof(FS_GradientStop)
        );
    }
    gradient->stops[insert_at].offset_0_to_1 = offset_0_to_1;
    gradient->stops[insert_at].color_rgba8 = color_rgba8;
    gradient->stop_count += 1u;
    return true;
}

FS_Pattern* fs_pattern_create_image(const FS_ImageHandle* handle, FS_PatternRepeat repeat_mode) {
    if (!handle || handle->width == 0u || handle->height == 0u) {
        return NULL;
    }
    if (repeat_mode < FS_PATTERN_REPEAT || repeat_mode > FS_PATTERN_NO_REPEAT) {
        return NULL;
    }
    FS_Pattern* pattern = (FS_Pattern*)calloc(1u, sizeof(FS_Pattern));
    if (!pattern) {
        return NULL;
    }
    pattern->handle = *handle;
    pattern->repeat_mode = (uint8_t)repeat_mode;
    fs_affine_set_identity_2d(pattern->xform);
    fs_affine_set_identity_2d(pattern->inv_xform);
    pattern->inv_valid = 1u;
    return pattern;
}

void fs_pattern_destroy(FS_Pattern* pattern) {
    free(pattern);
}

bool fs_pattern_set_transform(FS_Pattern* pattern, float a, float b, float c, float d, float e, float f) {
    if (!pattern || !isfinite(a) || !isfinite(b) || !isfinite(c) || !isfinite(d) || !isfinite(e) || !isfinite(f)) {
        return false;
    }
    pattern->xform[0] = a;
    pattern->xform[1] = b;
    pattern->xform[2] = c;
    pattern->xform[3] = d;
    pattern->xform[4] = e;
    pattern->xform[5] = f;
    pattern->inv_valid = fs_affine_try_invert_2d(pattern->xform, pattern->inv_xform) ? 1u : 0u;
    return pattern->inv_valid != 0u;
}

FS_RadialGradient* fs_radial_gradient_create(float x0, float y0, float r0, float x1, float y1, float r1) {
    if (!isfinite(x0) || !isfinite(y0) || !isfinite(r0) || r0 < 0.0f ||
        !isfinite(x1) || !isfinite(y1) || !isfinite(r1) || r1 < 0.0f) {
        return NULL;
    }
    FS_RadialGradient* gradient = (FS_RadialGradient*)calloc(1u, sizeof(FS_RadialGradient));
    if (!gradient) {
        return NULL;
    }
    gradient->x0 = x0;
    gradient->y0 = y0;
    gradient->r0 = r0;
    gradient->x1 = x1;
    gradient->y1 = y1;
    gradient->r1 = r1;
    return gradient;
}

void fs_radial_gradient_destroy(FS_RadialGradient* gradient) {
    if (!gradient) {
        return;
    }
    free(gradient->stops);
    gradient->stops = NULL;
    gradient->stop_count = 0u;
    gradient->stop_capacity = 0u;
    free(gradient);
}

bool fs_radial_gradient_add_color_stop(FS_RadialGradient* gradient, float offset_0_to_1, uint32_t color_rgba8) {
    if (!gradient || !isfinite(offset_0_to_1) || offset_0_to_1 < 0.0f || offset_0_to_1 > 1.0f) {
        return false;
    }
    if (gradient->stop_count + 1u > gradient->stop_capacity) {
        uint32_t new_capacity = gradient->stop_capacity ? gradient->stop_capacity * 2u : 8u;
        if (new_capacity < gradient->stop_count + 1u) {
            new_capacity = gradient->stop_count + 1u;
        }
        FS_GradientStop* grown =
            (FS_GradientStop*)realloc(gradient->stops, (size_t)new_capacity * sizeof(FS_GradientStop));
        if (!grown) {
            return false;
        }
        gradient->stops = grown;
        gradient->stop_capacity = new_capacity;
    }
    uint32_t insert_at = gradient->stop_count;
    while (insert_at > 0u && gradient->stops[insert_at - 1u].offset_0_to_1 > offset_0_to_1) {
        insert_at -= 1u;
    }
    if (insert_at < gradient->stop_count) {
        memmove(
            &gradient->stops[insert_at + 1u],
            &gradient->stops[insert_at],
            (size_t)(gradient->stop_count - insert_at) * sizeof(FS_GradientStop)
        );
    }
    gradient->stops[insert_at].offset_0_to_1 = offset_0_to_1;
    gradient->stops[insert_at].color_rgba8 = color_rgba8;
    gradient->stop_count += 1u;
    return true;
}

FS_ConicGradient* fs_conic_gradient_create(float start_angle_radians, float cx, float cy) {
    if (!isfinite(start_angle_radians) || !isfinite(cx) || !isfinite(cy)) {
        return NULL;
    }
    FS_ConicGradient* gradient = (FS_ConicGradient*)calloc(1u, sizeof(FS_ConicGradient));
    if (!gradient) {
        return NULL;
    }
    gradient->start_angle_radians = start_angle_radians;
    gradient->cx = cx;
    gradient->cy = cy;
    return gradient;
}

void fs_conic_gradient_destroy(FS_ConicGradient* gradient) {
    if (!gradient) {
        return;
    }
    free(gradient->stops);
    gradient->stops = NULL;
    gradient->stop_count = 0u;
    gradient->stop_capacity = 0u;
    free(gradient);
}

bool fs_conic_gradient_add_color_stop(FS_ConicGradient* gradient, float offset_0_to_1, uint32_t color_rgba8) {
    if (!gradient || !isfinite(offset_0_to_1) || offset_0_to_1 < 0.0f || offset_0_to_1 > 1.0f) {
        return false;
    }
    if (gradient->stop_count + 1u > gradient->stop_capacity) {
        uint32_t new_capacity = gradient->stop_capacity ? gradient->stop_capacity * 2u : 8u;
        if (new_capacity < gradient->stop_count + 1u) {
            new_capacity = gradient->stop_count + 1u;
        }
        FS_GradientStop* grown =
            (FS_GradientStop*)realloc(gradient->stops, (size_t)new_capacity * sizeof(FS_GradientStop));
        if (!grown) {
            return false;
        }
        gradient->stops = grown;
        gradient->stop_capacity = new_capacity;
    }
    uint32_t insert_at = gradient->stop_count;
    while (insert_at > 0u && gradient->stops[insert_at - 1u].offset_0_to_1 > offset_0_to_1) {
        insert_at -= 1u;
    }
    if (insert_at < gradient->stop_count) {
        memmove(
            &gradient->stops[insert_at + 1u],
            &gradient->stops[insert_at],
            (size_t)(gradient->stop_count - insert_at) * sizeof(FS_GradientStop)
        );
    }
    gradient->stops[insert_at].offset_0_to_1 = offset_0_to_1;
    gradient->stops[insert_at].color_rgba8 = color_rgba8;
    gradient->stop_count += 1u;
    return true;
}

bool fs_draw_linear_gradient_rect_cells(
    FS_Core* core,
    float x,
    float y,
    float w,
    float h,
    const FS_StyleLinearGradient* grad
) {
    if (!core || !grad || grad->stop_count < 2u || !isfinite(x) || !isfinite(y) || !isfinite(w) || !isfinite(h)) {
        return false;
    }
    float x0 = x;
    float y0 = y;
    float x1 = x + w;
    float y1 = y + h;
    if (x1 < x0) {
        float t = x0;
        x0 = x1;
        x1 = t;
    }
    if (y1 < y0) {
        float t = y0;
        y0 = y1;
        y1 = t;
    }
    const float rw = x1 - x0;
    const float rh = y1 - y0;
    if (rw <= 1e-6f || rh <= 1e-6f) {
        return true;
    }

    const float gdx = grad->x1 - grad->x0;
    const float gdy = grad->y1 - grad->y0;
    const float agx = fabsf(gdx);
    const float agy = fabsf(gdy);
    uint32_t nx = 1u;
    uint32_t ny = 1u;
    if (agx > agy * 2.0f) {
        nx = (uint32_t)ceilf(rw / 18.0f);
        if (nx < 4u) {
            nx = 4u;
        } else if (nx > 64u) {
            nx = 64u;
        }
    } else if (agy > agx * 2.0f) {
        ny = (uint32_t)ceilf(rh / 18.0f);
        if (ny < 4u) {
            ny = 4u;
        } else if (ny > 64u) {
            ny = 64u;
        }
    } else {
        nx = (uint32_t)ceilf(rw / 22.0f);
        ny = (uint32_t)ceilf(rh / 22.0f);
        if (nx < 3u) {
            nx = 3u;
        } else if (nx > 32u) {
            nx = 32u;
        }
        if (ny < 3u) {
            ny = 3u;
        } else if (ny > 32u) {
            ny = 32u;
        }
    }

    for (uint32_t iy = 0u; iy < ny; ++iy) {
        const float ya = y0 + rh * ((float)iy / (float)ny);
        const float yb = y0 + rh * ((float)(iy + 1u) / (float)ny);
        const float cy = (ya + yb) * 0.5f;
        for (uint32_t ix = 0u; ix < nx; ++ix) {
            const float xa = x0 + rw * ((float)ix / (float)nx);
            const float xb = x0 + rw * ((float)(ix + 1u) / (float)nx);
            const float cx = (xa + xb) * 0.5f;
            const uint32_t c = fs_linear_gradient_sample_rgba8(grad, cx, cy);
            if (!fs_cmd_rect(core, xa, ya, xb - xa, yb - ya, 0.0f, c)) {
                return false;
            }
        }
    }
    return true;
}

bool fs_draw_radial_gradient_rect_cells(
    FS_Core* core,
    float x,
    float y,
    float w,
    float h,
    const FS_StyleRadialGradient* grad
) {
    if (!core || !grad || grad->stop_count < 2u || !isfinite(x) || !isfinite(y) || !isfinite(w) || !isfinite(h)) {
        return false;
    }
    float x0 = x;
    float y0 = y;
    float x1 = x + w;
    float y1 = y + h;
    if (x1 < x0) {
        float t = x0;
        x0 = x1;
        x1 = t;
    }
    if (y1 < y0) {
        float t = y0;
        y0 = y1;
        y1 = t;
    }
    const float rw = x1 - x0;
    const float rh = y1 - y0;
    if (rw <= 1e-6f || rh <= 1e-6f) {
        return true;
    }

    uint32_t nx = (uint32_t)ceilf(rw / 16.0f);
    uint32_t ny = (uint32_t)ceilf(rh / 16.0f);
    if (nx < 4u) {
        nx = 4u;
    } else if (nx > 72u) {
        nx = 72u;
    }
    if (ny < 4u) {
        ny = 4u;
    } else if (ny > 72u) {
        ny = 72u;
    }

    for (uint32_t iy = 0u; iy < ny; ++iy) {
        const float ya = y0 + rh * ((float)iy / (float)ny);
        const float yb = y0 + rh * ((float)(iy + 1u) / (float)ny);
        const float cy = (ya + yb) * 0.5f;
        for (uint32_t ix = 0u; ix < nx; ++ix) {
            const float xa = x0 + rw * ((float)ix / (float)nx);
            const float xb = x0 + rw * ((float)(ix + 1u) / (float)nx);
            const float cx = (xa + xb) * 0.5f;
            const uint32_t c = fs_radial_gradient_sample_rgba8(grad, cx, cy);
            if (!fs_cmd_rect(core, xa, ya, xb - xa, yb - ya, 0.0f, c)) {
                return false;
            }
        }
    }
    return true;
}

bool fs_draw_conic_gradient_rect_cells(
    FS_Core* core,
    float x,
    float y,
    float w,
    float h,
    const FS_StyleConicGradient* grad
) {
    if (!core || !grad || grad->stop_count < 2u || !isfinite(x) || !isfinite(y) || !isfinite(w) || !isfinite(h)) {
        return false;
    }
    float x0 = x;
    float y0 = y;
    float x1 = x + w;
    float y1 = y + h;
    if (x1 < x0) {
        float t = x0;
        x0 = x1;
        x1 = t;
    }
    if (y1 < y0) {
        float t = y0;
        y0 = y1;
        y1 = t;
    }
    const float rw = x1 - x0;
    const float rh = y1 - y0;
    if (rw <= 1e-6f || rh <= 1e-6f) {
        return true;
    }

    uint32_t nx = (uint32_t)ceilf(rw / 16.0f);
    uint32_t ny = (uint32_t)ceilf(rh / 16.0f);
    if (nx < 4u) {
        nx = 4u;
    } else if (nx > 72u) {
        nx = 72u;
    }
    if (ny < 4u) {
        ny = 4u;
    } else if (ny > 72u) {
        ny = 72u;
    }

    for (uint32_t iy = 0u; iy < ny; ++iy) {
        const float ya = y0 + rh * ((float)iy / (float)ny);
        const float yb = y0 + rh * ((float)(iy + 1u) / (float)ny);
        const float cy = (ya + yb) * 0.5f;
        for (uint32_t ix = 0u; ix < nx; ++ix) {
            const float xa = x0 + rw * ((float)ix / (float)nx);
            const float xb = x0 + rw * ((float)(ix + 1u) / (float)nx);
            const float cx = (xa + xb) * 0.5f;
            const uint32_t c = fs_conic_gradient_sample_rgba8(grad, cx, cy);
            if (!fs_cmd_rect(core, xa, ya, xb - xa, yb - ya, 0.0f, c)) {
                return false;
            }
        }
    }
    return true;
}

bool fs_emit_fill_triangle_fan(FS_Core* core, const FS_Point2* points, uint32_t count, uint32_t color) {
    if (!core || !points || count < 3u) {
        return true;
    }
    const FS_Point2 p0 = points[0];
    for (uint32_t i = 1u; i + 1u < count; ++i) {
        const FS_Point2 p1 = points[i];
        const FS_Point2 p2 = points[i + 1u];
        const float area2 =
            (p1.x - p0.x) * (p2.y - p0.y) -
            (p1.y - p0.y) * (p2.x - p0.x);
        if (fabsf(area2) <= 1e-6f) {
            continue;
        }
        if (!fs_cmd_triangle(core, p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, color)) {
            return false;
        }
    }
    return true;
}

bool fs_fill_points_reserve(FS_Point2** io_points, uint32_t* io_capacity, uint32_t required) {
    if (!io_points || !io_capacity) {
        return false;
    }
    if (required <= *io_capacity) {
        return true;
    }
    uint32_t new_cap = (*io_capacity > 0u) ? *io_capacity : 64u;
    while (new_cap < required) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_Point2* grown = (FS_Point2*)realloc(*io_points, (size_t)new_cap * sizeof(FS_Point2));
    if (!grown) {
        return false;
    }
    *io_points = grown;
    *io_capacity = new_cap;
    return true;
}

bool fs_fill_points_push_unique(
    FS_Point2** io_points,
    uint32_t* io_count,
    uint32_t* io_capacity,
    float x,
    float y
) {
    if (!io_points || !io_count || !io_capacity) {
        return false;
    }
    if (*io_count > 0u) {
        const FS_Point2* prev = &(*io_points)[*io_count - 1u];
        if (fabsf(prev->x - x) <= 1e-4f && fabsf(prev->y - y) <= 1e-4f) {
            return true;
        }
    }
    const uint32_t required = *io_count + 1u;
    if (!fs_fill_points_reserve(io_points, io_capacity, required)) {
        return false;
    }
    FS_Point2* dst = *io_points;
    dst[*io_count].x = x;
    dst[*io_count].y = y;
    *io_count = required;
    return true;
}

void fs_fill_contour_clear(FS_FillContour* contour) {
    if (!contour) {
        return;
    }
    free(contour->points);
    contour->points = NULL;
    contour->count = 0u;
    contour->capacity = 0u;
    contour->area2 = 0.0f;
    contour->abs_area2 = 0.0f;
    contour->parent = -1;
    contour->depth = 0u;
    contour->is_hole = false;
    contour->owner_outer = -1;
}

bool fs_fill_contours_reserve(FS_FillContour** io_contours, uint32_t* io_capacity, uint32_t required) {
    if (!io_contours || !io_capacity) {
        return false;
    }
    if (required <= *io_capacity) {
        return true;
    }
    uint32_t new_cap = (*io_capacity > 0u) ? *io_capacity : 8u;
    while (new_cap < required) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_FillContour* grown = (FS_FillContour*)realloc(*io_contours, (size_t)new_cap * sizeof(FS_FillContour));
    if (!grown) {
        return false;
    }
    if (new_cap > *io_capacity) {
        memset(grown + *io_capacity, 0, (size_t)(new_cap - *io_capacity) * sizeof(FS_FillContour));
    }
    *io_contours = grown;
    *io_capacity = new_cap;
    return true;
}

float fs_polygon_signed_area2(const FS_Point2* points, uint32_t count) {
    if (!points || count < 3u) {
        return 0.0f;
    }
    float area2 = 0.0f;
    for (uint32_t i = 0u; i < count; ++i) {
        const FS_Point2* a = &points[i];
        const FS_Point2* b = &points[(i + 1u) % count];
        area2 += (a->x * b->y) - (b->x * a->y);
    }
    return area2;
}

float fs_cross2(const FS_Point2* a, const FS_Point2* b, const FS_Point2* c) {
    return (b->x - a->x) * (c->y - a->y) - (b->y - a->y) * (c->x - a->x);
}

bool fs_point_in_contour(const FS_Point2* points, uint32_t count, const FS_Point2* p) {
    if (!points || !p || count < 3u) {
        return false;
    }
    bool inside = false;
    for (uint32_t i = 0u, j = count - 1u; i < count; j = i++) {
        const FS_Point2* a = &points[i];
        const FS_Point2* b = &points[j];
        const bool intersects =
            ((a->y > p->y) != (b->y > p->y)) &&
            (p->x < (b->x - a->x) * (p->y - a->y) / ((b->y - a->y) + 1e-12f) + a->x);
        if (intersects) {
            inside = !inside;
        }
    }
    return inside;
}

void fs_points_reverse(FS_Point2* points, uint32_t count) {
    if (!points || count < 2u) {
        return;
    }
    uint32_t i = 0u;
    uint32_t j = count - 1u;
    while (i < j) {
        FS_Point2 tmp = points[i];
        points[i] = points[j];
        points[j] = tmp;
        ++i;
        --j;
    }
}

bool fs_contour_finalize(FS_FillContour* contour) {
    if (!contour || contour->count < 3u) {
        return false;
    }
    if (!fs_polygon_compact_in_place(contour->points, &contour->count)) {
        return false;
    }
    contour->area2 = fs_polygon_signed_area2(contour->points, contour->count);
    contour->abs_area2 = fabsf(contour->area2);
    return contour->abs_area2 > 1e-5f;
}

bool fs_point_in_triangle_or_edge(
    const FS_Point2* p,
    const FS_Point2* a,
    const FS_Point2* b,
    const FS_Point2* c
) {
    const float e0 = fs_cross2(a, b, p);
    const float e1 = fs_cross2(b, c, p);
    const float e2 = fs_cross2(c, a, p);
    const bool has_neg = (e0 < -1e-6f) || (e1 < -1e-6f) || (e2 < -1e-6f);
    const bool has_pos = (e0 > 1e-6f) || (e1 > 1e-6f) || (e2 > 1e-6f);
    return !(has_neg && has_pos);
}

uint32_t fs_find_rightmost_point(const FS_Point2* points, uint32_t count) {
    uint32_t idx = 0u;
    for (uint32_t i = 1u; i < count; ++i) {
        if (points[i].x > points[idx].x + 1e-6f ||
            (fabsf(points[i].x - points[idx].x) <= 1e-6f && points[i].y < points[idx].y)) {
            idx = i;
        }
    }
    return idx;
}

int fs_orient2d(const FS_Point2* a, const FS_Point2* b, const FS_Point2* c) {
    const float v = (b->x - a->x) * (c->y - a->y) - (b->y - a->y) * (c->x - a->x);
    if (v > 1e-6f) {
        return 1;
    }
    if (v < -1e-6f) {
        return -1;
    }
    return 0;
}

bool fs_point_on_segment(const FS_Point2* p, const FS_Point2* a, const FS_Point2* b) {
    if (!p || !a || !b) {
        return false;
    }
    if (fabsf(fs_cross2(a, b, p)) > 1e-6f) {
        return false;
    }
    const float min_x = fminf(a->x, b->x) - 1e-6f;
    const float max_x = fmaxf(a->x, b->x) + 1e-6f;
    const float min_y = fminf(a->y, b->y) - 1e-6f;
    const float max_y = fmaxf(a->y, b->y) + 1e-6f;
    return p->x >= min_x && p->x <= max_x && p->y >= min_y && p->y <= max_y;
}

bool fs_segments_intersect(const FS_Point2* a, const FS_Point2* b, const FS_Point2* c, const FS_Point2* d) {
    const int o1 = fs_orient2d(a, b, c);
    const int o2 = fs_orient2d(a, b, d);
    const int o3 = fs_orient2d(c, d, a);
    const int o4 = fs_orient2d(c, d, b);
    if (o1 != o2 && o3 != o4) {
        return true;
    }
    if (o1 == 0 && fs_point_on_segment(c, a, b)) {
        return true;
    }
    if (o2 == 0 && fs_point_on_segment(d, a, b)) {
        return true;
    }
    if (o3 == 0 && fs_point_on_segment(a, c, d)) {
        return true;
    }
    if (o4 == 0 && fs_point_on_segment(b, c, d)) {
        return true;
    }
    return false;
}

bool fs_bridge_visible(
    const FS_Point2* outer,
    uint32_t outer_count,
    uint32_t outer_idx,
    const FS_Point2* hole,
    uint32_t hole_count,
    uint32_t hole_idx
) {
    if (!outer || !hole || outer_count < 3u || hole_count < 3u || outer_idx >= outer_count || hole_idx >= hole_count) {
        return false;
    }

    const FS_Point2* hp = &hole[hole_idx];
    const FS_Point2* op = &outer[outer_idx];
    const FS_Point2 seg_a = *hp;
    const FS_Point2 seg_b = *op;

    const FS_Point2 near_hole = {
        .x = hp->x + (op->x - hp->x) * 1e-3f,
        .y = hp->y + (op->y - hp->y) * 1e-3f
    };
    if (fs_point_in_contour(hole, hole_count, &near_hole)) {
        return false;
    }

    const FS_Point2 mid = {
        .x = 0.5f * (hp->x + op->x),
        .y = 0.5f * (hp->y + op->y)
    };
    if (!fs_point_in_contour(outer, outer_count, &mid)) {
        return false;
    }

    for (uint32_t i = 0u; i < outer_count; ++i) {
        const uint32_t j = (i + 1u) % outer_count;
        if (i == outer_idx || j == outer_idx) {
            continue;
        }
        if (fs_segments_intersect(&seg_a, &seg_b, &outer[i], &outer[j])) {
            return false;
        }
    }

    for (uint32_t i = 0u; i < hole_count; ++i) {
        const uint32_t j = (i + 1u) % hole_count;
        if (i == hole_idx || j == hole_idx) {
            continue;
        }
        if (fs_segments_intersect(&seg_a, &seg_b, &hole[i], &hole[j])) {
            return false;
        }
    }
    return true;
}

bool fs_find_outer_bridge_point(
    const FS_Point2* outer,
    uint32_t outer_count,
    const FS_Point2* hole,
    uint32_t hole_count,
    uint32_t hole_idx,
    uint32_t* out_outer_idx
) {
    if (!outer || !hole || !out_outer_idx || outer_count < 3u || hole_count < 3u || hole_idx >= hole_count) {
        return false;
    }

    const FS_Point2* hole_point = &hole[hole_idx];
    const float hx = hole_point->x;
    const float hy = hole_point->y;
    const float eps = 1e-6f;

    bool ray_hit = false;
    float best_ix = 1e30f;
    uint32_t best_ei = 0u;
    uint32_t best_ej = 0u;
    for (uint32_t i = 0u; i < outer_count; ++i) {
        const uint32_t j = (i + 1u) % outer_count;
        const FS_Point2* a = &outer[i];
        const FS_Point2* b = &outer[j];
        const float ay = a->y;
        const float by = b->y;
        if (fabsf(ay - by) <= eps) {
            continue;
        }
        if ((hy < fminf(ay, by)) || (hy > fmaxf(ay, by))) {
            continue;
        }
        const float t = (hy - ay) / (by - ay);
        if (t < -eps || t > 1.0f + eps) {
            continue;
        }
        const float ix = a->x + t * (b->x - a->x);
        if (ix <= hx + eps) {
            continue;
        }
        if (!ray_hit || ix < best_ix) {
            ray_hit = true;
            best_ix = ix;
            best_ei = i;
            best_ej = j;
        }
    }

    if (ray_hit) {
        uint32_t primary = best_ei;
        uint32_t secondary = best_ej;
        if (outer[best_ej].x > outer[best_ei].x) {
            primary = best_ej;
            secondary = best_ei;
        }
        if (fs_bridge_visible(outer, outer_count, primary, hole, hole_count, hole_idx)) {
            *out_outer_idx = primary;
            return true;
        }
        if (fs_bridge_visible(outer, outer_count, secondary, hole, hole_count, hole_idx)) {
            *out_outer_idx = secondary;
            return true;
        }

        uint32_t interval_best = 0u;
        float interval_score = 1e30f;
        bool interval_found = false;
        for (uint32_t i = 0u; i < outer_count; ++i) {
            if (outer[i].x < hx - eps || outer[i].x > best_ix + eps) {
                continue;
            }
            if (!fs_bridge_visible(outer, outer_count, i, hole, hole_count, hole_idx)) {
                continue;
            }
            const float dx = outer[i].x - hx;
            const float dy = outer[i].y - hy;
            const float score = dx * dx + dy * dy;
            if (!interval_found || score < interval_score) {
                interval_found = true;
                interval_best = i;
                interval_score = score;
            }
        }
        if (interval_found) {
            *out_outer_idx = interval_best;
            return true;
        }
    }

    uint32_t fallback_best = 0u;
    float fallback_score = 1e30f;
    bool fallback_right = false;

    uint32_t visible_best = 0u;
    float visible_score = 1e30f;
    bool visible_right = false;
    bool found_visible = false;

    for (uint32_t i = 0u; i < outer_count; ++i) {
        const float dx = outer[i].x - hole_point->x;
        const float dy = outer[i].y - hole_point->y;
        const float d2 = dx * dx + dy * dy;
        const bool right = dx >= -1e-4f;

        if (right) {
            if (!fallback_right || d2 < fallback_score) {
                fallback_right = true;
                fallback_best = i;
                fallback_score = d2;
            }
        } else if (!fallback_right && d2 < fallback_score) {
            fallback_best = i;
            fallback_score = d2;
        }

        if (!fs_bridge_visible(outer, outer_count, i, hole, hole_count, hole_idx)) {
            continue;
        }
        if (right) {
            if (!visible_right || d2 < visible_score) {
                visible_right = true;
                visible_best = i;
                visible_score = d2;
                found_visible = true;
            }
        } else if (!visible_right && (!found_visible || d2 < visible_score)) {
            visible_best = i;
            visible_score = d2;
            found_visible = true;
        }
    }

    *out_outer_idx = found_visible ? visible_best : fallback_best;
    return true;
}

bool fs_merge_hole_into_polygon(
    FS_Point2** io_poly,
    uint32_t* io_count,
    uint32_t* io_capacity,
    FS_Point2* hole,
    uint32_t hole_count,
    uint32_t hole_right_idx
) {
    if (!io_poly || !io_count || !io_capacity || !hole || hole_count < 3u || *io_count < 3u) {
        return false;
    }
    if (hole_right_idx >= hole_count) {
        return false;
    }
    uint32_t oi = 0u;
    if (!fs_find_outer_bridge_point(*io_poly, *io_count, hole, hole_count, hole_right_idx, &oi)) {
        return false;
    }

    const uint32_t old_count = *io_count;
    const uint32_t required = old_count + hole_count + 2u;
    FS_Point2* merged = NULL;
    uint32_t merged_capacity = 0u;
    if (!fs_fill_points_reserve(&merged, &merged_capacity, required)) {
        return false;
    }

    uint32_t out_count = 0u;
    for (uint32_t i = 0u; i <= oi; ++i) {
        merged[out_count++] = (*io_poly)[i];
    }
    for (uint32_t s = 0u; s < hole_count; ++s) {
        const uint32_t hi = (hole_right_idx + s) % hole_count;
        merged[out_count++] = hole[hi];
    }
    merged[out_count++] = hole[hole_right_idx];
    for (uint32_t i = oi; i < old_count; ++i) {
        merged[out_count++] = (*io_poly)[i];
    }

    free(*io_poly);
    *io_poly = merged;
    *io_count = out_count;
    *io_capacity = merged_capacity;
    return true;
}

bool fs_emit_fill_triangles_ear_clip(FS_Core* core, const FS_Point2* points, uint32_t count, uint32_t color, bool allow_fan_fallback) {
    if (!core || !points || count < 3u) {
        return true;
    }

    uint32_t* indices = (uint32_t*)malloc((size_t)count * sizeof(uint32_t));
    if (!indices) {
        return false;
    }
    for (uint32_t i = 0u; i < count; ++i) {
        indices[i] = i;
    }

    uint32_t remaining = count;
    const bool ccw = fs_polygon_signed_area2(points, count) >= 0.0f;
    bool ok = true;
    uint32_t guard = 0u;
    const uint32_t guard_max = count * count * 2u + 16u;

    while (ok && remaining > 3u && guard < guard_max) {
        bool ear_found = false;
        for (uint32_t i = 0u; i < remaining; ++i) {
            const uint32_t ip = (i + remaining - 1u) % remaining;
            const uint32_t in = (i + 1u) % remaining;
            const FS_Point2* a = &points[indices[ip]];
            const FS_Point2* b = &points[indices[i]];
            const FS_Point2* c = &points[indices[in]];
            const float cross = fs_cross2(a, b, c);
            const bool is_convex = ccw ? (cross > 1e-6f) : (cross < -1e-6f);
            if (!is_convex) {
                continue;
            }

            bool contains_other = false;
            for (uint32_t j = 0u; j < remaining; ++j) {
                if (j == ip || j == i || j == in) {
                    continue;
                }
                const FS_Point2* p = &points[indices[j]];
                if (fs_point_in_triangle_or_edge(p, a, b, c)) {
                    contains_other = true;
                    break;
                }
            }
            if (contains_other) {
                continue;
            }

            if (!fs_cmd_triangle(core, a->x, a->y, b->x, b->y, c->x, c->y, color)) {
                ok = false;
                break;
            }

            if (i + 1u < remaining) {
                memmove(&indices[i], &indices[i + 1u], (size_t)(remaining - i - 1u) * sizeof(uint32_t));
            }
            remaining -= 1u;
            ear_found = true;
            break;
        }

        if (!ear_found) {
            break;
        }
        guard += 1u;
    }

    if (ok && remaining == 3u) {
        const FS_Point2* a = &points[indices[0]];
        const FS_Point2* b = &points[indices[1]];
        const FS_Point2* c = &points[indices[2]];
        if (fabsf(fs_cross2(a, b, c)) > 1e-6f) {
            ok = fs_cmd_triangle(core, a->x, a->y, b->x, b->y, c->x, c->y, color);
        }
    } else if (ok && remaining > 3u && allow_fan_fallback) {
        ok = fs_emit_fill_triangle_fan(core, points, count, color);
    } else if (ok && remaining > 3u) {
        ok = false;
    }

    free(indices);
    return ok;
}
