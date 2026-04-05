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
