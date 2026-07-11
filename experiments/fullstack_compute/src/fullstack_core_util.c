#include "fullstack_core_private.h"

#include <math.h>

uint32_t fs_align_up_u32(uint32_t value, uint32_t alignment) {
    if (alignment == 0u) {
        return value;
    }
    return (value + alignment - 1u) & ~(alignment - 1u);
}

uint32_t fs_round_up_pow2_u32(uint32_t v) {
    if (v <= 1u) {
        return 1u;
    }
    v -= 1u;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    return v + 1u;
}

float fs_get_text_bake_px(float requested_font_px) {
    const float requested = requested_font_px < 1.0f ? 1.0f : requested_font_px;
    const float clamped = requested < FS_TEXT_BAKE_MIN_PX ? FS_TEXT_BAKE_MIN_PX : requested;
    const uint32_t rounded = fs_round_up_pow2_u32((uint32_t)ceilf(clamped));
    return (float)rounded * FS_TEXT_BAKE_SCALE;
}

uint32_t fs_compute_mip_count(uint32_t width, uint32_t height) {
    uint32_t max_dim = width > height ? width : height;
    uint32_t levels = 1u;
    while (max_dim > 1u) {
        max_dim >>= 1u;
        levels++;
    }
    return levels;
}

uint64_t fs_hash64_mix(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

uint64_t fs_hash64_u32(uint64_t hash, uint32_t value) {
    return fs_hash64_mix(hash, (uint64_t)value);
}

uint64_t fs_hash64_f32(uint64_t hash, float value) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.f = value;
    return fs_hash64_u32(hash, conv.u);
}

static void fs_swap_round_radius(FS_RoundRadius* a, FS_RoundRadius* b) {
    const FS_RoundRadius tmp = *a;
    *a = *b;
    *b = tmp;
}

bool fs_normalize_round_rect(
    float x,
    float y,
    float w,
    float h,
    const FS_RoundRectRadii* radii,
    FS_NormalizedRoundRect* out_rect
) {
    if (!radii || !out_rect || !isfinite(x) || !isfinite(y) ||
        !isfinite(w) || !isfinite(h)) {
        return false;
    }

    FS_RoundRectRadii resolved = *radii;
    FS_RoundRadius* values[4] = {
        &resolved.top_left,
        &resolved.top_right,
        &resolved.bottom_right,
        &resolved.bottom_left
    };
    for (uint32_t i = 0u; i < 4u; ++i) {
        if (!isfinite(values[i]->x) || !isfinite(values[i]->y) ||
            values[i]->x < 0.0f || values[i]->y < 0.0f) {
            return false;
        }
    }

    if (w < 0.0f) {
        x += w;
        w = -w;
        fs_swap_round_radius(&resolved.top_left, &resolved.top_right);
        fs_swap_round_radius(&resolved.bottom_left, &resolved.bottom_right);
    }
    if (h < 0.0f) {
        y += h;
        h = -h;
        fs_swap_round_radius(&resolved.top_left, &resolved.bottom_left);
        fs_swap_round_radius(&resolved.top_right, &resolved.bottom_right);
    }

    float scale = 1.0f;
#define FS_LIMIT_RADIUS_SUM(limit, sum) \
    do { \
        const float fs_sum_value = (sum); \
        if (fs_sum_value > 0.0f) { \
            scale = fminf(scale, (limit) / fs_sum_value); \
        } \
    } while (0)
    FS_LIMIT_RADIUS_SUM(w, resolved.top_left.x + resolved.top_right.x);
    FS_LIMIT_RADIUS_SUM(w, resolved.bottom_left.x + resolved.bottom_right.x);
    FS_LIMIT_RADIUS_SUM(h, resolved.top_left.y + resolved.bottom_left.y);
    FS_LIMIT_RADIUS_SUM(h, resolved.top_right.y + resolved.bottom_right.y);
#undef FS_LIMIT_RADIUS_SUM
    scale = fmaxf(0.0f, fminf(scale, 1.0f));

    FS_RoundRadius* normalized[4] = {
        &resolved.top_left,
        &resolved.top_right,
        &resolved.bottom_right,
        &resolved.bottom_left
    };
    for (uint32_t i = 0u; i < 4u; ++i) {
        normalized[i]->x *= scale;
        normalized[i]->y *= scale;
    }

    out_rect->x = x;
    out_rect->y = y;
    out_rect->w = w;
    out_rect->h = h;
    out_rect->radii = resolved;
    return true;
}
