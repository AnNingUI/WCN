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
