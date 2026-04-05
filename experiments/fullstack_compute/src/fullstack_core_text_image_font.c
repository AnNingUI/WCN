#include "fullstack_core_private.h"

#include <stdlib.h>
#include <string.h>

static char* fs_strdup_owned(const char* s) {
    if (!s) {
        return NULL;
    }
    const size_t len = strlen(s);
    char* out = (char*)malloc(len + 1u);
    if (!out) {
        return NULL;
    }
    memcpy(out, s, len + 1u);
    return out;
}

bool fs_core_register_image_font(
    FS_Core* core,
    const FS_ImageFontSequence* sequences,
    uint32_t sequence_count,
    uint32_t* out_font_id
) {
    if (!core || !sequences || sequence_count == 0u || !out_font_id) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }

    if (!fs_ensure_image_font_capacity(st, st->image_font_count + 1u)) {
        return false;
    }

    FS_ImageSequenceEntry* entries =
        (FS_ImageSequenceEntry*)calloc((size_t)sequence_count, sizeof(FS_ImageSequenceEntry));
    if (!entries) {
        return false;
    }

    for (uint32_t i = 0u; i < sequence_count; ++i) {
        const FS_ImageFontSequence* src = &sequences[i];
        if (!src->utf8 || src->utf8[0] == '\0') {
            for (uint32_t j = 0u; j < i; ++j) {
                free(entries[j].utf8);
            }
            free(entries);
            return false;
        }
        entries[i].glyph_id = src->glyph_id;
        entries[i].utf8 = fs_strdup_owned(src->utf8);
        if (!entries[i].utf8) {
            for (uint32_t j = 0u; j < i; ++j) {
                free(entries[j].utf8);
            }
            free(entries);
            return false;
        }
        uint8_t cp_count = 0u;
        if (!fs_canonicalize_utf8_sequence(entries[i].utf8, entries[i].cps, &cp_count) || cp_count == 0u) {
            for (uint32_t j = 0u; j <= i; ++j) {
                free(entries[j].utf8);
            }
            free(entries);
            return false;
        }
        entries[i].cp_count = cp_count;
    }

    qsort(entries, sequence_count, sizeof(FS_ImageSequenceEntry), fs_sequence_entry_sort_desc);

    uint32_t next_font_id = 1u;
    for (uint32_t i = 0u; i < st->image_font_count; ++i) {
        if (st->image_fonts[i].id >= next_font_id) {
            next_font_id = st->image_fonts[i].id + 1u;
        }
    }
    if (next_font_id == 0u) {
        for (uint32_t i = 0u; i < sequence_count; ++i) {
            free(entries[i].utf8);
        }
        free(entries);
        return false;
    }

    FS_ImageFontState* font = &st->image_fonts[st->image_font_count++];
    memset(font, 0, sizeof(*font));
    font->id = next_font_id;
    font->sequences = entries;
    font->sequence_count = sequence_count;
    font->glyph_slots = NULL;
    font->glyph_slot_count = 0u;

    *out_font_id = next_font_id;
    return true;
}

bool fs_core_load_image_glyph_rgba8(
    FS_Core* core,
    uint32_t image_font_id,
    uint32_t glyph_id,
    const uint8_t* rgba_pixels,
    uint32_t width,
    uint32_t height
) {
    if (!core || image_font_id == 0u || !rgba_pixels || width == 0u || height == 0u) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    FS_ImageFontState* font = fs_find_image_font(st, image_font_id);
    if (!font) {
        return false;
    }
    if (!fs_image_font_ensure_glyph_slots(font, glyph_id + 1u)) {
        return false;
    }

    const size_t pixel_bytes = (size_t)width * (size_t)height * 4u;
    uint8_t* copied = (uint8_t*)malloc(pixel_bytes);
    if (!copied) {
        return false;
    }
    memcpy(copied, rgba_pixels, pixel_bytes);

    FS_ImageGlyphSlot* slot = &font->glyph_slots[glyph_id];
    free(slot->rgba);
    slot->rgba = copied;
    slot->width = width;
    slot->height = height;
    slot->loaded = true;

    uint32_t packed_key = 0u;
    if (fs_pack_image_glyph_key(image_font_id, glyph_id, &packed_key)) {
        fs_invalidate_cached_glyph(st, FS_IMAGE_FONT_KIND, packed_key);
    }

    if (st->missing_count > 0u) {
        uint32_t write = 0u;
        for (uint32_t read = 0u; read < st->missing_count; ++read) {
            FS_MissingImageGlyph m = st->missing_image_glyphs[read];
            if (m.image_font_id == image_font_id && m.glyph_id == glyph_id) {
                continue;
            }
            if (write != read) {
                st->missing_image_glyphs[write] = m;
            }
            write += 1u;
        }
        st->missing_count = write;
    }
    return true;
}

bool fs_core_load_image_glyph_png_memory(
    FS_Core* core,
    uint32_t image_font_id,
    uint32_t glyph_id,
    const uint8_t* encoded_bytes,
    size_t encoded_size
) {
    if (!core || image_font_id == 0u || !encoded_bytes || encoded_size == 0u) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || !st->image_backend || !st->image_backend->decode_memory || !st->image_backend->free_image) {
        return false;
    }
    uint8_t* rgba = NULL;
    uint32_t w = 0u;
    uint32_t h = 0u;
    if (!st->image_backend->decode_memory(encoded_bytes, encoded_size, &rgba, &w, &h)) {
        return false;
    }
    const bool ok = fs_core_load_image_glyph_rgba8(core, image_font_id, glyph_id, rgba, w, h);
    st->image_backend->free_image(rgba);
    return ok;
}

uint32_t fs_core_get_missing_image_glyph_count(const FS_Core* core) {
    if (!core || !core->internal_state) {
        return 0u;
    }
    const FS_InternalState* st = (const FS_InternalState*)core->internal_state;
    return st->missing_count;
}

bool fs_core_get_missing_image_glyph(
    const FS_Core* core,
    uint32_t index,
    FS_MissingImageGlyph* out_missing
) {
    if (!core || !core->internal_state || !out_missing) {
        return false;
    }
    const FS_InternalState* st = (const FS_InternalState*)core->internal_state;
    if (index >= st->missing_count) {
        return false;
    }
    *out_missing = st->missing_image_glyphs[index];
    return true;
}

void fs_core_clear_missing_image_glyphs(FS_Core* core) {
    if (!core || !core->internal_state) {
        return;
    }
    FS_InternalState* st = (FS_InternalState*)core->internal_state;
    st->missing_count = 0u;
}
