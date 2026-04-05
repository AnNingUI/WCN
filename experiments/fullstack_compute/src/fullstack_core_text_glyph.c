#include "fullstack_core_private.h"

#include <stdlib.h>
#include <string.h>


uint32_t fs_decode_utf8(const char** cursor) {
    if (!cursor || !*cursor) {
        return 0u;
    }
    const unsigned char* s = (const unsigned char*)(*cursor);
    if (*s == 0u) {
        return 0u;
    }
    uint32_t cp = 0u;
    if (s[0] < 0x80u) {
        cp = s[0];
        *cursor += 1;
        return cp;
    }
    if ((s[0] & 0xE0u) == 0xC0u && s[1] != 0u) {
        cp = ((uint32_t)(s[0] & 0x1Fu) << 6) | (uint32_t)(s[1] & 0x3Fu);
        *cursor += 2;
        return cp;
    }
    if ((s[0] & 0xF0u) == 0xE0u && s[1] != 0u && s[2] != 0u) {
        cp = ((uint32_t)(s[0] & 0x0Fu) << 12) |
             ((uint32_t)(s[1] & 0x3Fu) << 6) |
             (uint32_t)(s[2] & 0x3Fu);
        *cursor += 3;
        return cp;
    }
    if ((s[0] & 0xF8u) == 0xF0u && s[1] != 0u && s[2] != 0u && s[3] != 0u) {
        cp = ((uint32_t)(s[0] & 0x07u) << 18) |
             ((uint32_t)(s[1] & 0x3Fu) << 12) |
             ((uint32_t)(s[2] & 0x3Fu) << 6) |
             (uint32_t)(s[3] & 0x3Fu);
        *cursor += 4;
        return cp;
    }
    *cursor += 1;
    return 0xFFFDu;
}

bool fs_pack_image_glyph_key(uint32_t image_font_id, uint32_t glyph_id, uint32_t* out_key) {
    if (!out_key) {
        return false;
    }
    if (image_font_id >= (1u << FS_IMAGE_KEY_FONT_BITS) || glyph_id > FS_IMAGE_KEY_GLYPH_MASK) {
        return false;
    }
    *out_key = (image_font_id << FS_IMAGE_KEY_GLYPH_BITS) | glyph_id;
    return true;
}

uint32_t fs_unpack_image_key_font_id(uint32_t key) {
    return key >> FS_IMAGE_KEY_GLYPH_BITS;
}

uint32_t fs_unpack_image_key_glyph_id(uint32_t key) {
    return key & FS_IMAGE_KEY_GLYPH_MASK;
}

bool fs_is_emoji_modifier(uint32_t cp) {
    return cp >= 0x1F3FBu && cp <= 0x1F3FFu;
}

bool fs_is_emoji_variation_selector(uint32_t cp) {
    return cp == 0xFE0Fu || cp == 0xFE0Eu;
}

bool fs_is_zwj(uint32_t cp) {
    return cp == 0x200Du;
}

bool fs_is_keycap_combiner(uint32_t cp) {
    return cp == 0x20E3u;
}

bool fs_canonicalize_utf8_sequence(
    const char* utf8,
    uint32_t* out_cps,
    uint8_t* out_cp_count
) {
    if (!utf8 || !out_cps || !out_cp_count) {
        return false;
    }
    *out_cp_count = 0u;
    const char* cursor = utf8;
    while (*cursor != '\0') {
        uint32_t cp = fs_decode_utf8(&cursor);
        if (cp == 0u) {
            break;
        }
        if (fs_is_emoji_variation_selector(cp)) {
            continue;
        }
        if (*out_cp_count >= FS_IMAGE_SEQ_MAX_CP) {
            return false;
        }
        out_cps[*out_cp_count] = cp;
        *out_cp_count += 1u;
    }
    return *out_cp_count > 0u;
}

bool fs_match_sequence_prefix(
    const FS_ImageSequenceEntry* seq,
    const char* utf8_ptr,
    size_t* out_bytes
) {
    if (!seq || !utf8_ptr || !out_bytes || seq->cp_count == 0u) {
        return false;
    }
    const char* cursor = utf8_ptr;
    uint32_t matched = 0u;
    bool odd = false;
    while (*cursor != '\0' && matched < seq->cp_count) {
        const char* probe = cursor;
        const uint32_t cp = fs_decode_utf8(&probe);
        if (cp == 0u) {
            return false;
        }
        if (fs_is_emoji_variation_selector(cp)) {
            cursor = probe;
            continue;
        }

        bool consume = false;
        if (!odd) {
            consume = true;
            odd = true;
        } else if (fs_is_zwj(cp)) {
            consume = true;
            odd = false;
        } else if (fs_is_emoji_modifier(cp) || fs_is_keycap_combiner(cp)) {
            consume = true;
        } else {
            break;
        }

        if (!consume || cp != seq->cps[matched]) {
            return false;
        }
        matched += 1u;
        cursor = probe;
    }
    if (matched != seq->cp_count) {
        return false;
    }
    *out_bytes = (size_t)(cursor - utf8_ptr);
    return *out_bytes > 0u;
}

bool fs_ensure_image_font_capacity(FS_InternalState* st, uint32_t required) {
    if (!st) {
        return false;
    }
    if (required <= st->image_font_capacity) {
        return true;
    }
    uint32_t new_cap = st->image_font_capacity ? st->image_font_capacity : 4u;
    while (new_cap < required) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_ImageFontState* grown = (FS_ImageFontState*)realloc(st->image_fonts, (size_t)new_cap * sizeof(FS_ImageFontState));
    if (!grown) {
        return false;
    }
    if (new_cap > st->image_font_capacity) {
        memset(
            grown + st->image_font_capacity,
            0,
            (size_t)(new_cap - st->image_font_capacity) * sizeof(FS_ImageFontState)
        );
    }
    st->image_fonts = grown;
    st->image_font_capacity = new_cap;
    return true;
}

FS_ImageFontState* fs_find_image_font(FS_InternalState* st, uint32_t image_font_id) {
    if (!st || image_font_id == 0u) {
        return NULL;
    }
    for (uint32_t i = 0u; i < st->image_font_count; ++i) {
        if (st->image_fonts[i].id == image_font_id) {
            return &st->image_fonts[i];
        }
    }
    return NULL;
}

bool fs_image_font_ensure_glyph_slots(FS_ImageFontState* font, uint32_t required_slots) {
    if (!font) {
        return false;
    }
    if (required_slots <= font->glyph_slot_count) {
        return true;
    }
    uint32_t new_count = font->glyph_slot_count ? font->glyph_slot_count : 8u;
    while (new_count < required_slots) {
        if (new_count > UINT32_MAX / 2u) {
            new_count = required_slots;
            break;
        }
        new_count *= 2u;
    }
    FS_ImageGlyphSlot* grown =
        (FS_ImageGlyphSlot*)realloc(font->glyph_slots, (size_t)new_count * sizeof(FS_ImageGlyphSlot));
    if (!grown) {
        return false;
    }
    if (new_count > font->glyph_slot_count) {
        memset(
            grown + font->glyph_slot_count,
            0,
            (size_t)(new_count - font->glyph_slot_count) * sizeof(FS_ImageGlyphSlot)
        );
    }
    font->glyph_slots = grown;
    font->glyph_slot_count = new_count;
    return true;
}

bool fs_push_missing_image_glyph(
    FS_InternalState* st,
    uint32_t image_font_id,
    uint32_t glyph_id,
    const char* utf8,
    size_t utf8_len
) {
    if (!st || !utf8 || utf8_len == 0u) {
        return false;
    }
    for (uint32_t i = 0u; i < st->missing_count; ++i) {
        const FS_MissingImageGlyph* m = &st->missing_image_glyphs[i];
        if (m->image_font_id == image_font_id && m->glyph_id == glyph_id) {
            return true;
        }
    }
    if (st->missing_count + 1u > st->missing_capacity) {
        uint32_t new_cap = st->missing_capacity ? st->missing_capacity : 16u;
        while (new_cap < st->missing_count + 1u) {
            if (new_cap > UINT32_MAX / 2u) {
                new_cap = st->missing_count + 1u;
                break;
            }
            new_cap *= 2u;
        }
        FS_MissingImageGlyph* grown =
            (FS_MissingImageGlyph*)realloc(st->missing_image_glyphs, (size_t)new_cap * sizeof(FS_MissingImageGlyph));
        if (!grown) {
            return false;
        }
        st->missing_image_glyphs = grown;
        st->missing_capacity = new_cap;
    }
    FS_MissingImageGlyph* out = &st->missing_image_glyphs[st->missing_count++];
    memset(out, 0, sizeof(*out));
    out->image_font_id = image_font_id;
    out->glyph_id = glyph_id;
    const size_t copy_len = utf8_len < (sizeof(out->sequence_utf8) - 1u) ? utf8_len : (sizeof(out->sequence_utf8) - 1u);
    memcpy(out->sequence_utf8, utf8, copy_len);
    out->sequence_utf8[copy_len] = '\0';
    return true;
}

bool fs_find_image_sequence_match(
    const FS_InternalState* st,
    const char* utf8_ptr,
    uint32_t* out_image_font_id,
    uint32_t* out_glyph_id,
    size_t* out_consumed_bytes
) {
    if (!st || !utf8_ptr || !out_image_font_id || !out_glyph_id || !out_consumed_bytes) {
        return false;
    }
    const FS_ImageSequenceEntry* best_seq = NULL;
    uint32_t best_font_id = 0u;
    size_t best_bytes = 0u;
    uint8_t best_cp_count = 0u;

    for (uint32_t i = 0u; i < st->image_font_count; ++i) {
        const FS_ImageFontState* font = &st->image_fonts[i];
        for (uint32_t j = 0u; j < font->sequence_count; ++j) {
            const FS_ImageSequenceEntry* seq = &font->sequences[j];
            size_t consumed = 0u;
            if (!fs_match_sequence_prefix(seq, utf8_ptr, &consumed)) {
                continue;
            }
            if (!best_seq ||
                seq->cp_count > best_cp_count ||
                (seq->cp_count == best_cp_count && consumed > best_bytes)) {
                best_seq = seq;
                best_font_id = font->id;
                best_bytes = consumed;
                best_cp_count = seq->cp_count;
            }
        }
    }
    if (!best_seq) {
        return false;
    }
    *out_image_font_id = best_font_id;
    *out_glyph_id = best_seq->glyph_id;
    *out_consumed_bytes = best_bytes;
    return true;
}

uint32_t fs_glyph_cache_hash_key(uint32_t glyph_key, uint32_t key_kind, uint32_t bake_px_q) {
    uint32_t x = glyph_key * 0x9E3779B1u;
    x ^= key_kind * 0x85EBCA77u;
    x ^= bake_px_q * 0xC2B2AE3Du;
    x ^= x >> 16u;
    return x;
}

size_t fs_glyph_cache_next_pow2(size_t v) {
    if (v <= 1u) {
        return 1u;
    }
    --v;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
#if SIZE_MAX > 0xFFFFFFFFu
    v |= v >> 32u;
#endif
    return v + 1u;
}

bool fs_glyph_cache_rebuild(FS_InternalState* st, size_t min_capacity) {
    if (!st) {
        return false;
    }
    size_t target = min_capacity;
    const size_t needed_for_entries = st->glyph_count > 0u ? (st->glyph_count * 2u) : 0u;
    if (target < needed_for_entries) {
        target = needed_for_entries;
    }
    if (target < 64u) {
        target = 64u;
    }
    target = fs_glyph_cache_next_pow2(target);
    if (target < st->glyph_count) {
        return false;
    }

    uint32_t* table = (uint32_t*)calloc(target, sizeof(uint32_t));
    if (!table) {
        return false;
    }
    const size_t mask = target - 1u;
    for (size_t i = 0u; i < st->glyph_count; ++i) {
        const FS_GlyphEntry* g = &st->glyphs[i];
        size_t slot = (size_t)fs_glyph_cache_hash_key(g->glyph_key, g->key_kind, g->bake_px_q) & mask;
        for (size_t probe = 0u; probe < target; ++probe) {
            if (table[slot] == 0u) {
                table[slot] = (uint32_t)(i + 1u);
                break;
            }
            slot = (slot + 1u) & mask;
        }
    }

    free(st->glyph_hash_slots);
    st->glyph_hash_slots = table;
    st->glyph_hash_capacity = target;
    return true;
}

void fs_glyph_cache_clear_index(FS_InternalState* st) {
    if (!st || !st->glyph_hash_slots || st->glyph_hash_capacity == 0u) {
        return;
    }
    memset(st->glyph_hash_slots, 0, st->glyph_hash_capacity * sizeof(uint32_t));
}

FS_GlyphEntry* fs_glyph_cache_find(
    FS_InternalState* st,
    uint32_t glyph_key,
    uint32_t key_kind,
    uint32_t bake_px_q
) {
    if (!st || st->glyph_count == 0u) {
        return NULL;
    }
    if (!st->glyph_hash_slots || st->glyph_hash_capacity == 0u) {
        if (!fs_glyph_cache_rebuild(st, st->glyph_count * 2u)) {
            return NULL;
        }
    }

    const size_t cap = st->glyph_hash_capacity;
    const size_t mask = cap - 1u;
    size_t slot = (size_t)fs_glyph_cache_hash_key(glyph_key, key_kind, bake_px_q) & mask;
    for (size_t probe = 0u; probe < cap; ++probe) {
        const uint32_t packed = st->glyph_hash_slots[slot];
        if (packed == 0u) {
            return NULL;
        }
        const size_t idx = (size_t)(packed - 1u);
        if (idx < st->glyph_count) {
            FS_GlyphEntry* g = &st->glyphs[idx];
            if (g->glyph_key == glyph_key && g->key_kind == key_kind && g->bake_px_q == bake_px_q) {
                return g;
            }
        }
        slot = (slot + 1u) & mask;
    }
    return NULL;
}

bool fs_glyph_cache_insert_index(FS_InternalState* st, size_t glyph_index) {
    if (!st || glyph_index >= st->glyph_count) {
        return false;
    }
    if (!st->glyph_hash_slots || st->glyph_hash_capacity == 0u ||
        (st->glyph_count * 10u) >= (st->glyph_hash_capacity * 7u)) {
        size_t target = st->glyph_hash_capacity ? st->glyph_hash_capacity * 2u : 64u;
        if (target < st->glyph_count * 2u) {
            target = st->glyph_count * 2u;
        }
        if (!fs_glyph_cache_rebuild(st, target)) {
            return false;
        }
    }

    const FS_GlyphEntry* g = &st->glyphs[glyph_index];
    const size_t cap = st->glyph_hash_capacity;
    const size_t mask = cap - 1u;
    size_t slot = (size_t)fs_glyph_cache_hash_key(g->glyph_key, g->key_kind, g->bake_px_q) & mask;
    for (size_t probe = 0u; probe < cap; ++probe) {
        const uint32_t packed = st->glyph_hash_slots[slot];
        if (packed == 0u) {
            st->glyph_hash_slots[slot] = (uint32_t)(glyph_index + 1u);
            return true;
        }
        const size_t idx = (size_t)(packed - 1u);
        if (idx < st->glyph_count) {
            const FS_GlyphEntry* cur = &st->glyphs[idx];
            if (cur->glyph_key == g->glyph_key && cur->key_kind == g->key_kind && cur->bake_px_q == g->bake_px_q) {
                st->glyph_hash_slots[slot] = (uint32_t)(glyph_index + 1u);
                return true;
            }
        }
        slot = (slot + 1u) & mask;
    }
    return false;
}

void fs_invalidate_cached_glyph(FS_InternalState* st, uint32_t key_kind, uint32_t glyph_key) {
    if (!st || st->glyph_count == 0u) {
        return;
    }
    size_t write = 0u;
    for (size_t read = 0u; read < st->glyph_count; ++read) {
        FS_GlyphEntry g = st->glyphs[read];
        if (g.key_kind == key_kind && g.glyph_key == glyph_key) {
            continue;
        }
        if (write != read) {
            st->glyphs[write] = g;
        }
        write += 1u;
    }
    st->glyph_count = write;
    if (st->glyph_count == 0u) {
        fs_glyph_cache_clear_index(st);
    } else {
        (void)fs_glyph_cache_rebuild(st, st->glyph_hash_capacity);
    }
}

int fs_sequence_entry_sort_desc(const void* lhs, const void* rhs) {
    const FS_ImageSequenceEntry* a = (const FS_ImageSequenceEntry*)lhs;
    const FS_ImageSequenceEntry* b = (const FS_ImageSequenceEntry*)rhs;
    if (a->cp_count == b->cp_count) {
        return 0;
    }
    return (a->cp_count > b->cp_count) ? -1 : 1;
}

bool fs_ensure_glyph_capacity(FS_InternalState* st, size_t required) {
    if (!st) {
        return false;
    }
    if (required <= st->glyph_capacity) {
        return true;
    }
    size_t new_cap = st->glyph_capacity ? st->glyph_capacity : 512u;
    while (new_cap < required) {
        if (new_cap > (SIZE_MAX / 2u)) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_GlyphEntry* grown = (FS_GlyphEntry*)realloc(st->glyphs, new_cap * sizeof(FS_GlyphEntry));
    if (!grown) {
        return false;
    }
    st->glyphs = grown;
    st->glyph_capacity = new_cap;
    return true;
}

bool fs_find_or_create_glyph(
    FS_Core* core,
    uint32_t glyph_key,
    uint32_t key_kind,
    float font_px,
    FS_GlyphEntry** out_glyph,
    float* out_layout_scale
) {
    if (!core || !out_glyph) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (key_kind != FS_IMAGE_FONT_KIND && st->font_count == 0u) {
        return false;
    }

    const float requested_px = font_px < 1.0f ? 1.0f : font_px;
    const float bake_px = fs_get_text_bake_px(requested_px);
    const uint32_t bake_px_q = fs_quantize_font_size(bake_px);
    const float layout_scale = requested_px / bake_px;

    FS_GlyphEntry* cached = fs_glyph_cache_find(st, glyph_key, key_kind, bake_px_q);
    if (cached) {
        *out_glyph = cached;
        if (out_layout_scale) {
            *out_layout_scale = layout_scale;
        }
        return true;
    }

    if (!fs_ensure_glyph_capacity(st, st->glyph_count + 1u)) {
        return false;
    }

    FS_GlyphEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.glyph_key = glyph_key;
    entry.key_kind = key_kind;
    entry.bake_px_q = bake_px_q;
    entry.bake_px = bake_px;
    entry.sdf_radius_px = 8.0f;
    entry.sdf_onedge = 0.5f;
    entry.sdf_pixel_dist_scale = 1.0f;
    entry.text_flags = 0u;
    entry.font_slot = 0u;

    FS_FontGlyphBitmap glyph_bitmap;
    memset(&glyph_bitmap, 0, sizeof(glyph_bitmap));
    bool need_free_pixels = false;

    if (key_kind == FS_IMAGE_FONT_KIND) {
        const uint32_t image_font_id = fs_unpack_image_key_font_id(glyph_key);
        const uint32_t image_glyph_id = fs_unpack_image_key_glyph_id(glyph_key);
        FS_ImageFontState* image_font = fs_find_image_font(st, image_font_id);
        if (!image_font || image_glyph_id >= image_font->glyph_slot_count) {
            return false;
        }
        FS_ImageGlyphSlot* slot = &image_font->glyph_slots[image_glyph_id];
        if (!slot->loaded || !slot->rgba || slot->width == 0u || slot->height == 0u) {
            return false;
        }

        glyph_bitmap.pixels = slot->rgba;
        glyph_bitmap.width = slot->width;
        glyph_bitmap.height = slot->height;
        glyph_bitmap.pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8;
        glyph_bitmap.sdf_radius_px = 1.0f;
        glyph_bitmap.sdf_onedge = 0.5f;
        glyph_bitmap.sdf_pixel_dist_scale = 1.0f;
        fs_apply_emoji_layout_from_bitmap(&entry, bake_px, slot->width, slot->height);
    } else {
        if (!st->font_backend || !st->font_backend->get_glyph_sdf) {
            return false;
        }
        bool glyph_ok = false;
        uint32_t selected_font_slot = 0u;
        if (key_kind == 1u && st->font_backend->get_glyph_sdf_by_index) {
            if (st->font_count == 0u) {
                return false;
            }
            if (st->font_backend->get_glyph_sdf_by_index(st->fonts[0], glyph_key, bake_px, &glyph_bitmap)) {
                glyph_ok = true;
                selected_font_slot = 0u;
            }
        } else {
            const uint32_t count = st->font_count > FS_MAX_FONT_FALLBACKS ? FS_MAX_FONT_FALLBACKS : st->font_count;
            for (uint32_t fi = 0u; fi < count; ++fi) {
                if (!st->fonts[fi]) {
                    continue;
                }
                if (st->font_backend->get_glyph_sdf(st->fonts[fi], glyph_key, bake_px, &glyph_bitmap)) {
                    glyph_ok = true;
                    selected_font_slot = fi;
                    break;
                }
            }
        }
        if (!glyph_ok) {
            return false;
        }
        entry.font_slot = (uint8_t)selected_font_slot;
        need_free_pixels = true;
        if (glyph_bitmap.pixel_format != FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8) {
            glyph_bitmap.pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_SDF_R8;
            entry.text_flags = 0u;
            entry.advance = glyph_bitmap.advance;
            entry.bearing_x = (float)glyph_bitmap.offset_x;
            entry.bearing_y = (float)glyph_bitmap.offset_y;
            entry.atlas_width = (float)glyph_bitmap.width;
            entry.atlas_height = (float)glyph_bitmap.height;
            entry.sdf_radius_px = glyph_bitmap.sdf_radius_px;
            entry.sdf_onedge = glyph_bitmap.sdf_onedge;
            entry.sdf_pixel_dist_scale = glyph_bitmap.sdf_pixel_dist_scale;
        } else {
            fs_apply_emoji_layout_from_bitmap(&entry, bake_px, glyph_bitmap.width, glyph_bitmap.height);
        }
    }

    if (glyph_bitmap.pixels && glyph_bitmap.width > 0u && glyph_bitmap.height > 0u) {
        uint32_t gx = 0, gy = 0;
        if (!fs_alloc_from_atlas(
                core->glyph_atlas_width,
                core->glyph_atlas_height,
                &core->glyph_atlas_cursor_x,
                &core->glyph_atlas_cursor_y,
                &core->glyph_atlas_row_height,
                FS_GLYPH_ATLAS_PADDING,
                glyph_bitmap.width,
                glyph_bitmap.height,
                &gx,
                &gy
            )) {
            if (need_free_pixels && st->font_backend->free_glyph_pixels) {
                st->font_backend->free_glyph_pixels(glyph_bitmap.pixels);
            }
            return false;
        }

        if (!fs_upload_glyph_with_mips(
                core,
                gx,
                gy,
                glyph_bitmap.width,
                glyph_bitmap.height,
                glyph_bitmap.pixels,
                glyph_bitmap.pixel_format,
                glyph_bitmap.sdf_onedge,
                glyph_bitmap.sdf_pixel_dist_scale
            )) {
            if (need_free_pixels && st->font_backend->free_glyph_pixels) {
                st->font_backend->free_glyph_pixels(glyph_bitmap.pixels);
            }
            return false;
        }
        if (need_free_pixels && st->font_backend->free_glyph_pixels) {
            st->font_backend->free_glyph_pixels(glyph_bitmap.pixels);
        }

        entry.uv_min[0] = (float)gx / (float)core->glyph_atlas_width;
        entry.uv_min[1] = (float)gy / (float)core->glyph_atlas_height;
        entry.uv_max[0] = (float)(gx + glyph_bitmap.width) / (float)core->glyph_atlas_width;
        entry.uv_max[1] = (float)(gy + glyph_bitmap.height) / (float)core->glyph_atlas_height;
    } else if (need_free_pixels && glyph_bitmap.pixels && st->font_backend->free_glyph_pixels) {
        st->font_backend->free_glyph_pixels(glyph_bitmap.pixels);
    }

    const size_t new_index = st->glyph_count;
    st->glyphs[new_index] = entry;
    *out_glyph = &st->glyphs[new_index];
    if (out_layout_scale) {
        *out_layout_scale = layout_scale;
    }
    st->glyph_count += 1u;
    if (!fs_glyph_cache_insert_index(st, new_index)) {
        (void)fs_glyph_cache_rebuild(st, st->glyph_hash_capacity);
    }
    return true;
}

float fs_get_kerning_advance(
    FS_Core* core,
    uint32_t left_codepoint,
    uint32_t right_codepoint,
    float font_px,
    uint8_t font_slot
) {
    if (!core || font_px <= 0.0f) {
        return 0.0f;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || st->font_count == 0u || !st->font_backend || !st->font_backend->get_kerning_advance) {
        return 0.0f;
    }
    if ((uint32_t)font_slot >= st->font_count || !st->fonts[font_slot]) {
        return 0.0f;
    }
    return st->font_backend->get_kerning_advance(st->fonts[font_slot], left_codepoint, right_codepoint, font_px);
}
