#include "fullstack_core_private.h"

#include <math.h>

void fs_apply_emoji_layout_from_bitmap(FS_GlyphEntry* entry, float bake_px, uint32_t bitmap_w, uint32_t bitmap_h) {
    if (!entry) {
        return;
    }
    const float emoji_h = bake_px * FS_EMOJI_SCALE_BIAS;
    float aspect = 1.0f;
    if (bitmap_h > 0u) {
        aspect = (float)bitmap_w / (float)bitmap_h;
    }
    if (aspect < 0.75f) {
        aspect = 0.75f;
    } else if (aspect > 1.25f) {
        aspect = 1.25f;
    }
    const float emoji_w = emoji_h * aspect;
    float emoji_advance = emoji_w * (1.0f + FS_EMOJI_ADVANCE_BIAS);
    const float advance_quant = bake_px * FS_EMOJI_ADVANCE_ALIGN;
    if (advance_quant > 1e-3f) {
        emoji_advance = floorf((emoji_advance / advance_quant) + 0.5f) * advance_quant;
    }
    entry->advance = emoji_advance;
    entry->bearing_x = emoji_w * FS_EMOJI_ADVANCE_BIAS * FS_EMOJI_ADVANCE_ALIGN;
    entry->bearing_y = -emoji_h * FS_EMOJI_ASCENT;
    entry->atlas_width = emoji_w;
    entry->atlas_height = emoji_h;
    entry->sdf_radius_px = 1.0f;
    entry->sdf_onedge = 0.5f;
    entry->sdf_pixel_dist_scale = 1.0f;
    entry->text_flags = FS_TEXT_FLAG_COLOR_GLYPH;
}

bool fs_image_atlas_shadow_bounds_ok(const FS_Core* core, uint32_t layer, uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    if (!core || !core->image_atlas_shadow_rgba || core->image_atlas_shadow_size == 0u) {
        return false;
    }
    if (layer >= core->image_atlas_layers || width == 0u || height == 0u) {
        return false;
    }
    if (x > core->image_atlas_width || y > core->image_atlas_height) {
        return false;
    }
    if (width > core->image_atlas_width - x || height > core->image_atlas_height - y) {
        return false;
    }
    return true;
}

bool fs_alloc_from_atlas(
    uint32_t atlas_w,
    uint32_t atlas_h,
    uint32_t* cursor_x,
    uint32_t* cursor_y,
    uint32_t* row_h,
    uint32_t padding,
    uint32_t payload_w,
    uint32_t payload_h,
    uint32_t* out_x,
    uint32_t* out_y
) {
    if (!cursor_x || !cursor_y || !row_h || !out_x || !out_y || payload_w == 0u || payload_h == 0u) {
        return false;
    }
    const uint32_t alloc_w = payload_w + 2u * padding;
    const uint32_t alloc_h = payload_h + 2u * padding;
    if (alloc_w > atlas_w || alloc_h > atlas_h) {
        return false;
    }
    if (*cursor_x + alloc_w > atlas_w) {
        *cursor_x = 0u;
        *cursor_y += *row_h;
        *row_h = 0u;
    }
    if (*cursor_y + alloc_h > atlas_h) {
        return false;
    }
    *out_x = *cursor_x + padding;
    *out_y = *cursor_y + padding;
    *cursor_x += alloc_w;
    if (alloc_h > *row_h) {
        *row_h = alloc_h;
    }
    return true;
}
