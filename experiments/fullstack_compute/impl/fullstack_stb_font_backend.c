#include "fullstack_stb_font_backend.h"
#include "fullstack_stb_cbdt_parser.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"
#include "stb_image.h"

#define FS_STB_TEXT_SCALE_BOOST 1.08f

typedef struct FS_StbColrBaseGlyphRecord {
    uint16_t glyph_id;
    uint16_t first_layer_index;
    uint16_t layer_count;
} FS_StbColrBaseGlyphRecord;

typedef struct FS_StbColrLayerRecord {
    uint16_t glyph_id;
    uint16_t palette_index;
} FS_StbColrLayerRecord;

typedef struct FS_StbCbdtCacheEntry {
    uint16_t glyph_id;
    uint8_t* rgba;
    uint32_t width;
    uint32_t height;
    FS_StbCbdtMetrics metrics;
} FS_StbCbdtCacheEntry;

#define FS_STB_CBDT_HASH_EMPTY 0u

typedef struct FS_StbFontHandle {
    stbtt_fontinfo info;
    bool has_stbtt;
    unsigned char* bytes;
    size_t bytes_size;
    bool prefer_bitmap_rgba;
    uint16_t colr_version;
    FS_StbColrBaseGlyphRecord* colr_base_glyphs;
    uint16_t colr_base_glyph_count;
    FS_StbColrLayerRecord* colr_layers;
    uint16_t colr_layer_count;
    uint32_t* cpal_palette_rgba;
    uint16_t cpal_palette_entry_count;
    FS_StbCbdtFont cbdt;
    bool has_cbdt;
    FS_StbCbdtCacheEntry* cbdt_cache;
    uint32_t* cbdt_cache_hash;
    uint32_t cbdt_cache_hash_capacity;
    uint32_t cbdt_cache_count;
    uint32_t cbdt_cache_capacity;
    bool cbdt_cache_debug;
    uint64_t cbdt_cache_hits;
    uint64_t cbdt_cache_misses;
    uint64_t cbdt_cache_inserts;
    uint64_t cbdt_cache_decode_failures;
    uint8_t* rgba_expand_scratch;
    size_t rgba_expand_scratch_capacity;
} FS_StbFontHandle;

static bool fs_stb_str_contains_nocase(const char* haystack, const char* needle) {
    if (!haystack || !needle || !needle[0]) {
        return false;
    }
    const size_t nh = strlen(haystack);
    const size_t nn = strlen(needle);
    if (nn > nh) {
        return false;
    }
    for (size_t i = 0; i + nn <= nh; ++i) {
        bool ok = true;
        for (size_t j = 0; j < nn; ++j) {
            const int a = tolower((unsigned char)haystack[i + j]);
            const int b = tolower((unsigned char)needle[j]);
            if (a != b) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return true;
        }
    }
    return false;
}

static bool fs_stb_is_probably_emoji_codepoint(uint32_t cp) {
    return
        (cp >= 0x1F000u && cp <= 0x1FAFFu) ||
        (cp >= 0x2600u && cp <= 0x27BFu) ||
        (cp >= 0xFE00u && cp <= 0xFE0Fu) ||
        cp == 0x200Du;
}

static float fs_stb_effective_text_px(float font_px) {
    const float clamped = font_px < 1.0f ? 1.0f : font_px;
    return clamped * FS_STB_TEXT_SCALE_BOOST;
}

static bool fs_stb_range_ok(size_t total, size_t offset, size_t size) {
    return offset <= total && size <= (total - offset);
}

static bool fs_stb_read_be_u16(const uint8_t* bytes, size_t bytes_size, size_t offset, uint16_t* out_value) {
    if (!out_value || !fs_stb_range_ok(bytes_size, offset, 2u)) {
        return false;
    }
    *out_value = (uint16_t)(((uint16_t)bytes[offset] << 8u) | (uint16_t)bytes[offset + 1u]);
    return true;
}

static bool fs_stb_read_be_u32(const uint8_t* bytes, size_t bytes_size, size_t offset, uint32_t* out_value) {
    if (!out_value || !fs_stb_range_ok(bytes_size, offset, 4u)) {
        return false;
    }
    *out_value =
        ((uint32_t)bytes[offset] << 24u) |
        ((uint32_t)bytes[offset + 1u] << 16u) |
        ((uint32_t)bytes[offset + 2u] << 8u) |
        (uint32_t)bytes[offset + 3u];
    return true;
}

static bool fs_stb_find_sfnt_table(
    const uint8_t* bytes,
    size_t bytes_size,
    uint32_t fontstart,
    const char tag[4],
    uint32_t* out_offset,
    uint32_t* out_length
) {
    if (!bytes || !tag || !out_offset || !out_length) {
        return false;
    }
    const size_t start = (size_t)fontstart;
    uint16_t num_tables = 0u;
    if (!fs_stb_read_be_u16(bytes, bytes_size, start + 4u, &num_tables)) {
        return false;
    }
    const size_t table_dir = start + 12u;
    const size_t rec_size = 16u;
    for (size_t i = 0u; i < (size_t)num_tables; ++i) {
        const size_t rec = table_dir + i * rec_size;
        if (!fs_stb_range_ok(bytes_size, rec, rec_size)) {
            return false;
        }
        if (bytes[rec + 0u] != (uint8_t)tag[0] ||
            bytes[rec + 1u] != (uint8_t)tag[1] ||
            bytes[rec + 2u] != (uint8_t)tag[2] ||
            bytes[rec + 3u] != (uint8_t)tag[3]) {
            continue;
        }
        uint32_t table_offset = 0u;
        uint32_t table_length = 0u;
        if (!fs_stb_read_be_u32(bytes, bytes_size, rec + 8u, &table_offset) ||
            !fs_stb_read_be_u32(bytes, bytes_size, rec + 12u, &table_length)) {
            return false;
        }
        if (!fs_stb_range_ok(bytes_size, (size_t)table_offset, (size_t)table_length)) {
            return false;
        }
        *out_offset = table_offset;
        *out_length = table_length;
        return true;
    }
    return false;
}

static void fs_stb_free_colr_cpal(FS_StbFontHandle* handle) {
    if (!handle) {
        return;
    }
    free(handle->colr_base_glyphs);
    handle->colr_base_glyphs = NULL;
    handle->colr_base_glyph_count = 0u;
    free(handle->colr_layers);
    handle->colr_layers = NULL;
    handle->colr_layer_count = 0u;
    free(handle->cpal_palette_rgba);
    handle->cpal_palette_rgba = NULL;
    handle->cpal_palette_entry_count = 0u;
    handle->colr_version = 0u;
}

static bool fs_stb_parse_cpal_table(FS_StbFontHandle* handle) {
    if (!handle || !handle->bytes || handle->bytes_size == 0u) {
        return false;
    }
    uint32_t cpal_offset = 0u;
    uint32_t cpal_length = 0u;
    if (!fs_stb_find_sfnt_table(
            handle->bytes,
            handle->bytes_size,
            (uint32_t)handle->info.fontstart,
            "CPAL",
            &cpal_offset,
            &cpal_length
        )) {
        return false;
    }
    const size_t base = (size_t)cpal_offset;
    const size_t len = (size_t)cpal_length;
    if (len < 12u) {
        return false;
    }

    uint16_t version = 0u;
    uint16_t num_palette_entries = 0u;
    uint16_t num_palettes = 0u;
    uint16_t num_color_records = 0u;
    uint32_t first_color_record_offset = 0u;
    if (!fs_stb_read_be_u16(handle->bytes, handle->bytes_size, base + 0u, &version) ||
        !fs_stb_read_be_u16(handle->bytes, handle->bytes_size, base + 2u, &num_palette_entries) ||
        !fs_stb_read_be_u16(handle->bytes, handle->bytes_size, base + 4u, &num_palettes) ||
        !fs_stb_read_be_u16(handle->bytes, handle->bytes_size, base + 6u, &num_color_records) ||
        !fs_stb_read_be_u32(handle->bytes, handle->bytes_size, base + 8u, &first_color_record_offset)) {
        return false;
    }
    (void)version;
    if (num_palette_entries == 0u || num_palettes == 0u || num_color_records == 0u) {
        return false;
    }

    const size_t color_index_array = base + 12u;
    const size_t color_index_bytes = (size_t)num_palettes * 2u;
    if (!fs_stb_range_ok(handle->bytes_size, color_index_array, color_index_bytes) ||
        !fs_stb_range_ok(base + len, color_index_array, color_index_bytes)) {
        return false;
    }
    const uint16_t palette_index = 0u;
    uint16_t palette_record_start = 0u;
    if (!fs_stb_read_be_u16(
            handle->bytes,
            handle->bytes_size,
            color_index_array + (size_t)palette_index * 2u,
            &palette_record_start
        )) {
        return false;
    }

    if ((uint32_t)palette_record_start + (uint32_t)num_palette_entries > (uint32_t)num_color_records) {
        return false;
    }

    const size_t color_record_array = base + (size_t)first_color_record_offset;
    const size_t color_record_bytes = (size_t)num_color_records * 4u;
    if (!fs_stb_range_ok(handle->bytes_size, color_record_array, color_record_bytes) ||
        !fs_stb_range_ok(base + len, color_record_array, color_record_bytes)) {
        return false;
    }

    uint32_t* palette_rgba = (uint32_t*)calloc((size_t)num_palette_entries, sizeof(uint32_t));
    if (!palette_rgba) {
        return false;
    }
    for (uint32_t i = 0u; i < (uint32_t)num_palette_entries; ++i) {
        const uint32_t rec_index = (uint32_t)palette_record_start + i;
        const size_t rec_off = color_record_array + (size_t)rec_index * 4u;
        const uint8_t b = handle->bytes[rec_off + 0u];
        const uint8_t g = handle->bytes[rec_off + 1u];
        const uint8_t r = handle->bytes[rec_off + 2u];
        const uint8_t a = handle->bytes[rec_off + 3u];
        palette_rgba[i] = ((uint32_t)r << 24u) | ((uint32_t)g << 16u) | ((uint32_t)b << 8u) | (uint32_t)a;
    }

    handle->cpal_palette_rgba = palette_rgba;
    handle->cpal_palette_entry_count = num_palette_entries;
    return true;
}

static bool fs_stb_parse_colr_table(FS_StbFontHandle* handle) {
    if (!handle || !handle->bytes || handle->bytes_size == 0u) {
        return false;
    }
    uint32_t colr_offset = 0u;
    uint32_t colr_length = 0u;
    if (!fs_stb_find_sfnt_table(
            handle->bytes,
            handle->bytes_size,
            (uint32_t)handle->info.fontstart,
            "COLR",
            &colr_offset,
            &colr_length
        )) {
        return false;
    }
    const size_t base = (size_t)colr_offset;
    const size_t len = (size_t)colr_length;
    if (len < 14u) {
        return false;
    }

    uint16_t version = 0u;
    uint16_t num_base_records = 0u;
    uint32_t base_records_offset = 0u;
    uint32_t layer_records_offset = 0u;
    uint16_t num_layer_records = 0u;
    if (!fs_stb_read_be_u16(handle->bytes, handle->bytes_size, base + 0u, &version) ||
        !fs_stb_read_be_u16(handle->bytes, handle->bytes_size, base + 2u, &num_base_records) ||
        !fs_stb_read_be_u32(handle->bytes, handle->bytes_size, base + 4u, &base_records_offset) ||
        !fs_stb_read_be_u32(handle->bytes, handle->bytes_size, base + 8u, &layer_records_offset) ||
        !fs_stb_read_be_u16(handle->bytes, handle->bytes_size, base + 12u, &num_layer_records)) {
        return false;
    }
    if (num_base_records == 0u || num_layer_records == 0u) {
        return false;
    }

    const size_t base_array = base + (size_t)base_records_offset;
    const size_t base_array_size = (size_t)num_base_records * 6u;
    const size_t layer_array = base + (size_t)layer_records_offset;
    const size_t layer_array_size = (size_t)num_layer_records * 4u;
    if (!fs_stb_range_ok(handle->bytes_size, base_array, base_array_size) ||
        !fs_stb_range_ok(base + len, base_array, base_array_size) ||
        !fs_stb_range_ok(handle->bytes_size, layer_array, layer_array_size) ||
        !fs_stb_range_ok(base + len, layer_array, layer_array_size)) {
        return false;
    }

    FS_StbColrBaseGlyphRecord* base_glyphs =
        (FS_StbColrBaseGlyphRecord*)calloc((size_t)num_base_records, sizeof(FS_StbColrBaseGlyphRecord));
    FS_StbColrLayerRecord* layers =
        (FS_StbColrLayerRecord*)calloc((size_t)num_layer_records, sizeof(FS_StbColrLayerRecord));
    if (!base_glyphs || !layers) {
        free(base_glyphs);
        free(layers);
        return false;
    }

    for (uint32_t i = 0u; i < (uint32_t)num_base_records; ++i) {
        const size_t rec = base_array + (size_t)i * 6u;
        uint16_t gid = 0u;
        uint16_t first_layer = 0u;
        uint16_t layer_count = 0u;
        if (!fs_stb_read_be_u16(handle->bytes, handle->bytes_size, rec + 0u, &gid) ||
            !fs_stb_read_be_u16(handle->bytes, handle->bytes_size, rec + 2u, &first_layer) ||
            !fs_stb_read_be_u16(handle->bytes, handle->bytes_size, rec + 4u, &layer_count)) {
            free(base_glyphs);
            free(layers);
            return false;
        }
        if ((uint32_t)first_layer + (uint32_t)layer_count > (uint32_t)num_layer_records) {
            free(base_glyphs);
            free(layers);
            return false;
        }
        base_glyphs[i].glyph_id = gid;
        base_glyphs[i].first_layer_index = first_layer;
        base_glyphs[i].layer_count = layer_count;
    }

    for (uint32_t i = 0u; i < (uint32_t)num_layer_records; ++i) {
        const size_t rec = layer_array + (size_t)i * 4u;
        uint16_t gid = 0u;
        uint16_t palette_index = 0u;
        if (!fs_stb_read_be_u16(handle->bytes, handle->bytes_size, rec + 0u, &gid) ||
            !fs_stb_read_be_u16(handle->bytes, handle->bytes_size, rec + 2u, &palette_index)) {
            free(base_glyphs);
            free(layers);
            return false;
        }
        layers[i].glyph_id = gid;
        layers[i].palette_index = palette_index;
    }

    handle->colr_base_glyphs = base_glyphs;
    handle->colr_base_glyph_count = num_base_records;
    handle->colr_layers = layers;
    handle->colr_layer_count = num_layer_records;
    handle->colr_version = version;
    return true;
}

static bool fs_stb_parse_colr_cpal(FS_StbFontHandle* handle) {
    if (!handle) {
        return false;
    }
    fs_stb_free_colr_cpal(handle);
    if (!fs_stb_parse_cpal_table(handle)) {
        return false;
    }
    if (!fs_stb_parse_colr_table(handle)) {
        fs_stb_free_colr_cpal(handle);
        return false;
    }
    return true;
}

static const FS_StbColrBaseGlyphRecord* fs_stb_find_colr_base_glyph(
    const FS_StbFontHandle* handle,
    uint16_t glyph_id
) {
    if (!handle || !handle->colr_base_glyphs || handle->colr_base_glyph_count == 0u) {
        return NULL;
    }
    for (uint32_t i = 0u; i < (uint32_t)handle->colr_base_glyph_count; ++i) {
        if (handle->colr_base_glyphs[i].glyph_id == glyph_id) {
            return &handle->colr_base_glyphs[i];
        }
    }
    return NULL;
}

static void fs_stb_colr_palette_lookup(
    const FS_StbFontHandle* handle,
    uint16_t palette_index,
    uint8_t* out_r,
    uint8_t* out_g,
    uint8_t* out_b,
    uint8_t* out_a
) {
    if (!out_r || !out_g || !out_b || !out_a) {
        return;
    }
    if (!handle || !handle->cpal_palette_rgba || handle->cpal_palette_entry_count == 0u ||
        palette_index == 0xFFFFu || palette_index >= handle->cpal_palette_entry_count) {
        *out_r = 255u;
        *out_g = 255u;
        *out_b = 255u;
        *out_a = 255u;
        return;
    }
    const uint32_t rgba = handle->cpal_palette_rgba[palette_index];
    *out_r = (uint8_t)((rgba >> 24u) & 0xFFu);
    *out_g = (uint8_t)((rgba >> 16u) & 0xFFu);
    *out_b = (uint8_t)((rgba >> 8u) & 0xFFu);
    *out_a = (uint8_t)(rgba & 0xFFu);
}

static bool fs_stb_try_get_colr_glyph_rgba(
    FS_StbFontHandle* handle,
    uint32_t glyph_index,
    float scale,
    FS_FontGlyphBitmap* out_glyph
) {
    if (!handle || !out_glyph || glyph_index > 0xFFFFu || scale <= 0.0f) {
        return false;
    }
    if (!handle->colr_base_glyphs || !handle->colr_layers || !handle->cpal_palette_rgba) {
        return false;
    }
    const FS_StbColrBaseGlyphRecord* base =
        fs_stb_find_colr_base_glyph(handle, (uint16_t)glyph_index);
    if (!base || base->layer_count == 0u) {
        return false;
    }
    if ((uint32_t)base->first_layer_index + (uint32_t)base->layer_count > (uint32_t)handle->colr_layer_count) {
        return false;
    }

    int32_t union_x0 = INT_MAX;
    int32_t union_y0 = INT_MAX;
    int32_t union_x1 = INT_MIN;
    int32_t union_y1 = INT_MIN;
    bool has_box = false;
    for (uint32_t i = 0u; i < (uint32_t)base->layer_count; ++i) {
        const FS_StbColrLayerRecord layer = handle->colr_layers[(uint32_t)base->first_layer_index + i];
        int x0 = 0;
        int y0 = 0;
        int x1 = 0;
        int y1 = 0;
        stbtt_GetGlyphBitmapBoxSubpixel(
            &handle->info,
            (int)layer.glyph_id,
            scale,
            scale,
            0.0f,
            0.0f,
            &x0,
            &y0,
            &x1,
            &y1
        );
        if (x1 <= x0 || y1 <= y0) {
            continue;
        }
        if (!has_box) {
            union_x0 = x0;
            union_y0 = y0;
            union_x1 = x1;
            union_y1 = y1;
            has_box = true;
        } else {
            if (x0 < union_x0) union_x0 = x0;
            if (y0 < union_y0) union_y0 = y0;
            if (x1 > union_x1) union_x1 = x1;
            if (y1 > union_y1) union_y1 = y1;
        }
    }
    if (!has_box || union_x1 <= union_x0 || union_y1 <= union_y0) {
        return false;
    }
    const int32_t w = union_x1 - union_x0;
    const int32_t h = union_y1 - union_y0;
    if (w <= 0 || h <= 0) {
        return false;
    }
    const size_t pixel_count = (size_t)w * (size_t)h;
    if (w > 32768 || h > 32768 || pixel_count > (SIZE_MAX / 4u)) {
        return false;
    }

    // Temporary buffer stores premultiplied RGBA while compositing layers.
    uint8_t* rgba = (uint8_t*)calloc(pixel_count, 4u);
    if (!rgba) {
        return false;
    }

    bool composited_any_layer = false;
    for (uint32_t i = 0u; i < (uint32_t)base->layer_count; ++i) {
        const FS_StbColrLayerRecord layer = handle->colr_layers[(uint32_t)base->first_layer_index + i];
        int bw = 0;
        int bh = 0;
        int xoff = 0;
        int yoff = 0;
        unsigned char* bitmap =
            stbtt_GetGlyphBitmap(&handle->info, scale, scale, (int)layer.glyph_id, &bw, &bh, &xoff, &yoff);
        if (!bitmap || bw <= 0 || bh <= 0) {
            if (bitmap) {
                stbtt_FreeBitmap(bitmap, NULL);
            }
            continue;
        }

        uint8_t lr = 255u;
        uint8_t lg = 255u;
        uint8_t lb = 255u;
        uint8_t la = 255u;
        fs_stb_colr_palette_lookup(handle, layer.palette_index, &lr, &lg, &lb, &la);
        if (la == 0u) {
            stbtt_FreeBitmap(bitmap, NULL);
            continue;
        }

        const int32_t dst_x0 = (int32_t)xoff - union_x0;
        const int32_t dst_y0 = (int32_t)yoff - union_y0;
        for (int32_t y = 0; y < (int32_t)bh; ++y) {
            const int32_t dy = dst_y0 + y;
            if (dy < 0 || dy >= h) {
                continue;
            }
            for (int32_t x = 0; x < (int32_t)bw; ++x) {
                const int32_t dx = dst_x0 + x;
                if (dx < 0 || dx >= w) {
                    continue;
                }
                const uint8_t m = bitmap[(size_t)y * (size_t)bw + (size_t)x];
                if (m == 0u) {
                    continue;
                }

                const uint8_t src_a = (uint8_t)(((uint32_t)m * (uint32_t)la + 127u) / 255u);
                if (src_a == 0u) {
                    continue;
                }
                const uint8_t src_pr = (uint8_t)(((uint32_t)lr * (uint32_t)src_a + 127u) / 255u);
                const uint8_t src_pg = (uint8_t)(((uint32_t)lg * (uint32_t)src_a + 127u) / 255u);
                const uint8_t src_pb = (uint8_t)(((uint32_t)lb * (uint32_t)src_a + 127u) / 255u);

                const size_t di = ((size_t)dy * (size_t)w + (size_t)dx) * 4u;
                const uint8_t inv_a = (uint8_t)(255u - src_a);
                const uint8_t dst_pr = rgba[di + 0u];
                const uint8_t dst_pg = rgba[di + 1u];
                const uint8_t dst_pb = rgba[di + 2u];
                const uint8_t dst_a = rgba[di + 3u];

                rgba[di + 0u] = (uint8_t)(src_pr + (((uint32_t)dst_pr * (uint32_t)inv_a + 127u) / 255u));
                rgba[di + 1u] = (uint8_t)(src_pg + (((uint32_t)dst_pg * (uint32_t)inv_a + 127u) / 255u));
                rgba[di + 2u] = (uint8_t)(src_pb + (((uint32_t)dst_pb * (uint32_t)inv_a + 127u) / 255u));
                rgba[di + 3u] = (uint8_t)(src_a + (((uint32_t)dst_a * (uint32_t)inv_a + 127u) / 255u));
            }
        }
        composited_any_layer = true;
        stbtt_FreeBitmap(bitmap, NULL);
    }

    if (!composited_any_layer) {
        free(rgba);
        return false;
    }

    for (size_t i = 0u; i < pixel_count; ++i) {
        const size_t di = i * 4u;
        const uint8_t a = rgba[di + 3u];
        if (a == 0u) {
            rgba[di + 0u] = 0u;
            rgba[di + 1u] = 0u;
            rgba[di + 2u] = 0u;
            continue;
        }
        rgba[di + 0u] = (uint8_t)(((uint32_t)rgba[di + 0u] * 255u + (uint32_t)a / 2u) / (uint32_t)a);
        rgba[di + 1u] = (uint8_t)(((uint32_t)rgba[di + 1u] * 255u + (uint32_t)a / 2u) / (uint32_t)a);
        rgba[di + 2u] = (uint8_t)(((uint32_t)rgba[di + 2u] * 255u + (uint32_t)a / 2u) / (uint32_t)a);
    }

    out_glyph->pixels = rgba;
    out_glyph->width = (uint32_t)w;
    out_glyph->height = (uint32_t)h;
    out_glyph->pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8;
    out_glyph->offset_x = union_x0;
    out_glyph->offset_y = union_y0;
    out_glyph->sdf_radius_px = 1.0f;
    out_glyph->sdf_onedge = 0.5f;
    out_glyph->sdf_pixel_dist_scale = 1.0f;
    return true;
}

static uint32_t fs_stb_u16_hash(uint16_t key) {
    uint32_t x = (uint32_t)key;
    x ^= x >> 7u;
    x *= 0x9E3779B1u;
    x ^= x >> 11u;
    return x;
}

static uint32_t fs_stb_next_pow2_u32(uint32_t x) {
    if (x <= 1u) {
        return 1u;
    }
    --x;
    x |= x >> 1u;
    x |= x >> 2u;
    x |= x >> 4u;
    x |= x >> 8u;
    x |= x >> 16u;
    return x + 1u;
}

static bool fs_stb_cbdt_cache_rehash(FS_StbFontHandle* handle, uint32_t requested_capacity) {
    if (!handle || requested_capacity == 0u) {
        return false;
    }
    const uint32_t cap = fs_stb_next_pow2_u32(requested_capacity);
    if (cap < 16u) {
        return false;
    }
    if (handle->cbdt_cache_count > cap) {
        return false;
    }

    uint32_t* table = (uint32_t*)calloc((size_t)cap, sizeof(uint32_t));
    if (!table) {
        return false;
    }
    const uint32_t mask = cap - 1u;
    for (uint32_t i = 0u; i < handle->cbdt_cache_count; ++i) {
        const uint16_t gid = handle->cbdt_cache[i].glyph_id;
        uint32_t slot = fs_stb_u16_hash(gid) & mask;
        for (;;) {
            if (table[slot] == FS_STB_CBDT_HASH_EMPTY) {
                table[slot] = i + 1u;
                break;
            }
            slot = (slot + 1u) & mask;
        }
    }

    free(handle->cbdt_cache_hash);
    handle->cbdt_cache_hash = table;
    handle->cbdt_cache_hash_capacity = cap;
    return true;
}

static FS_StbCbdtCacheEntry* fs_stb_cbdt_cache_find(FS_StbFontHandle* handle, uint16_t glyph_id) {
    if (!handle || !handle->cbdt_cache || handle->cbdt_cache_count == 0u ||
        !handle->cbdt_cache_hash || handle->cbdt_cache_hash_capacity == 0u) {
        return NULL;
    }
    const uint32_t cap = handle->cbdt_cache_hash_capacity;
    const uint32_t mask = cap - 1u;
    uint32_t slot = fs_stb_u16_hash(glyph_id) & mask;
    for (uint32_t probe = 0u; probe < cap; ++probe) {
        const uint32_t packed = handle->cbdt_cache_hash[slot];
        if (packed == FS_STB_CBDT_HASH_EMPTY) {
            return NULL;
        }
        const uint32_t idx = packed - 1u;
        if (idx < handle->cbdt_cache_count && handle->cbdt_cache[idx].glyph_id == glyph_id) {
            return &handle->cbdt_cache[idx];
        }
        slot = (slot + 1u) & mask;
    }
    return NULL;
}

static FS_StbCbdtCacheEntry* fs_stb_cbdt_cache_insert(
    FS_StbFontHandle* handle,
    uint16_t glyph_id,
    const uint8_t* rgba,
    uint32_t width,
    uint32_t height,
    const FS_StbCbdtMetrics* metrics
) {
    if (!handle || !rgba || width == 0u || height == 0u || !metrics) {
        return NULL;
    }

    FS_StbCbdtCacheEntry* existing = fs_stb_cbdt_cache_find(handle, glyph_id);
    if (existing) {
        return existing;
    }

    if (handle->cbdt_cache_count >= handle->cbdt_cache_capacity) {
        uint32_t new_cap = handle->cbdt_cache_capacity ? handle->cbdt_cache_capacity * 2u : 64u;
        if (new_cap < handle->cbdt_cache_count + 1u) {
            new_cap = handle->cbdt_cache_count + 1u;
        }
        FS_StbCbdtCacheEntry* grown =
            (FS_StbCbdtCacheEntry*)realloc(handle->cbdt_cache, (size_t)new_cap * sizeof(FS_StbCbdtCacheEntry));
        if (!grown) {
            return NULL;
        }
        if (new_cap > handle->cbdt_cache_capacity) {
            memset(
                grown + handle->cbdt_cache_capacity,
                0,
                (size_t)(new_cap - handle->cbdt_cache_capacity) * sizeof(FS_StbCbdtCacheEntry)
            );
        }
        handle->cbdt_cache = grown;
        handle->cbdt_cache_capacity = new_cap;
    }

    if (handle->cbdt_cache_hash_capacity == 0u ||
        (handle->cbdt_cache_count + 1u) * 10u >= handle->cbdt_cache_hash_capacity * 7u) {
        uint32_t target = handle->cbdt_cache_hash_capacity ? handle->cbdt_cache_hash_capacity * 2u : 128u;
        if (target < (handle->cbdt_cache_count + 1u) * 2u) {
            target = (handle->cbdt_cache_count + 1u) * 2u;
        }
        if (!fs_stb_cbdt_cache_rehash(handle, target)) {
            return NULL;
        }
    }

    const size_t bytes = (size_t)width * (size_t)height * 4u;
    uint8_t* copy = (uint8_t*)malloc(bytes);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, rgba, bytes);

    const uint32_t entry_idx = handle->cbdt_cache_count++;
    FS_StbCbdtCacheEntry* entry = &handle->cbdt_cache[entry_idx];
    entry->glyph_id = glyph_id;
    entry->rgba = copy;
    entry->width = width;
    entry->height = height;
    entry->metrics = *metrics;

    const uint32_t cap = handle->cbdt_cache_hash_capacity;
    const uint32_t mask = cap - 1u;
    uint32_t slot = fs_stb_u16_hash(glyph_id) & mask;
    for (uint32_t probe = 0u; probe < cap; ++probe) {
        if (handle->cbdt_cache_hash[slot] == FS_STB_CBDT_HASH_EMPTY) {
            handle->cbdt_cache_hash[slot] = entry_idx + 1u;
            return entry;
        }
        slot = (slot + 1u) & mask;
    }

    free(entry->rgba);
    entry->rgba = NULL;
    entry->glyph_id = 0u;
    entry->width = 0u;
    entry->height = 0u;
    memset(&entry->metrics, 0, sizeof(entry->metrics));
    --handle->cbdt_cache_count;
    (void)fs_stb_cbdt_cache_rehash(handle, cap);
    return NULL;
}

static void fs_stb_cbdt_cache_clear(FS_StbFontHandle* handle) {
    if (!handle) {
        return;
    }
    if (handle->cbdt_cache) {
        for (uint32_t i = 0u; i < handle->cbdt_cache_count; ++i) {
            free(handle->cbdt_cache[i].rgba);
            handle->cbdt_cache[i].rgba = NULL;
        }
        free(handle->cbdt_cache);
        handle->cbdt_cache = NULL;
    }
    free(handle->cbdt_cache_hash);
    handle->cbdt_cache_hash = NULL;
    handle->cbdt_cache_hash_capacity = 0u;
    handle->cbdt_cache_count = 0u;
    handle->cbdt_cache_capacity = 0u;
}

static bool fs_stb_try_get_cbdt_glyph_rgba(
    FS_StbFontHandle* handle,
    uint32_t glyph_index,
    float font_px,
    FS_FontGlyphBitmap* out_glyph
) {
    if (!handle || !out_glyph || !handle->has_cbdt || glyph_index > 0xFFFFu || font_px <= 0.0f) {
        return false;
    }
    const uint16_t gid = (uint16_t)glyph_index;
    uint32_t w = 0u;
    uint32_t h = 0u;
    FS_StbCbdtMetrics metrics = {0};

    FS_StbCbdtCacheEntry* cached = fs_stb_cbdt_cache_find(handle, gid);
    if (cached) {
        ++handle->cbdt_cache_hits;
    } else {
        ++handle->cbdt_cache_misses;
    }
    if (!cached || !cached->rgba || cached->width == 0u || cached->height == 0u) {
        FS_StbCbdtGlyphPngView view = {0};
        if (!fs_stb_cbdt_get_glyph_png(&handle->cbdt, gid, &view)) {
            ++handle->cbdt_cache_decode_failures;
            return false;
        }
        int iw = 0;
        int ih = 0;
        int channels = 0;
        unsigned char* decoded = stbi_load_from_memory(
            view.png_bytes,
            (int)view.png_size,
            &iw,
            &ih,
            &channels,
            4
        );
        if (!decoded || iw <= 0 || ih <= 0) {
            ++handle->cbdt_cache_decode_failures;
            if (decoded) {
                stbi_image_free(decoded);
            }
            return false;
        }

        const uint32_t dw = (uint32_t)iw;
        const uint32_t dh = (uint32_t)ih;
        const FS_StbCbdtMetrics dm = view.metrics;
        cached = fs_stb_cbdt_cache_insert(handle, gid, decoded, dw, dh, &dm);
        if (!cached) {
            ++handle->cbdt_cache_decode_failures;
            stbi_image_free(decoded);
            return false;
        }
        ++handle->cbdt_cache_inserts;
        stbi_image_free(decoded);
    }

    w = cached->width;
    h = cached->height;
    metrics = cached->metrics;
    const size_t pixel_bytes = (size_t)w * (size_t)h * 4u;
    uint8_t* rgba = (uint8_t*)malloc(pixel_bytes);
    if (!rgba) {
        return false;
    }
    memcpy(rgba, cached->rgba, pixel_bytes);

    const float strike_ppem = (float)fs_stb_cbdt_strike_ppem_y(&handle->cbdt);
    const float metric_scale = strike_ppem > 1e-3f ? (font_px / strike_ppem) : 1.0f;

    out_glyph->pixels = rgba;
    out_glyph->width = w;
    out_glyph->height = h;
    out_glyph->pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8;
    if (metrics.has_metrics) {
        out_glyph->offset_x = (int32_t)lroundf((float)metrics.bearing_x * metric_scale);
        out_glyph->offset_y = (int32_t)lroundf(-(float)metrics.bearing_y * metric_scale);
        out_glyph->advance = (float)metrics.advance * metric_scale;
    } else {
        out_glyph->offset_x = 0;
        out_glyph->offset_y = 0;
        out_glyph->advance = (float)w * metric_scale;
    }
    out_glyph->sdf_radius_px = 1.0f;
    out_glyph->sdf_onedge = 0.5f;
    out_glyph->sdf_pixel_dist_scale = 1.0f;
    return true;
}

static void fs_stb_pick_sdf_params(
    float font_px,
    int* out_padding,
    unsigned char* out_onedge_value,
    float* out_pixel_dist_scale
) {
    const float px = font_px < 1.0f ? 1.0f : font_px;
    int padding = 8;
    unsigned char onedge = 180u;
    float dist_scale = 10.0f;

    if (px <= 32.0f) {
        padding = 12;
        onedge = 188u;
        dist_scale = 18.0f;
    } else if (px <= 64.0f) {
        padding = 10;
        onedge = 184u;
        dist_scale = 14.0f;
    } else if (px <= 128.0f) {
        padding = 8;
        onedge = 176u;
        dist_scale = 10.0f;
    } else {
        padding = 6;
        onedge = 168u;
        dist_scale = 8.0f;
    }

    if (out_padding) {
        *out_padding = padding;
    }
    if (out_onedge_value) {
        *out_onedge_value = onedge;
    }
    if (out_pixel_dist_scale) {
        *out_pixel_dist_scale = dist_scale;
    }
}

static bool fs_read_file_bytes(const char* path, uint8_t** out_data, size_t* out_size) {
    if (!path || !out_data || !out_size) {
        return false;
    }
    *out_data = NULL;
    *out_size = 0u;

    FILE* f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    long sz = ftell(f);
    if (sz <= 0) {
        fclose(f);
        return false;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    uint8_t* data = (uint8_t*)malloc((size_t)sz);
    if (!data) {
        fclose(f);
        return false;
    }
    const size_t n = fread(data, 1, (size_t)sz, f);
    fclose(f);
    if (n != (size_t)sz) {
        free(data);
        return false;
    }
    *out_data = data;
    *out_size = (size_t)sz;
    return true;
}

static void* fs_stb_load_font_file(const char* path) {
    uint8_t* bytes = NULL;
    size_t size = 0u;
    if (!fs_read_file_bytes(path, &bytes, &size)) {
        return NULL;
    }
    FS_StbFontHandle* handle = (FS_StbFontHandle*)calloc(1, sizeof(FS_StbFontHandle));
    if (!handle) {
        free(bytes);
        return NULL;
    }
    handle->bytes = bytes;
    handle->bytes_size = size;
    {
        const char* dbg = getenv("FS_STB_CBDT_CACHE_DEBUG");
        handle->cbdt_cache_debug = (dbg && dbg[0] != '\0' && dbg[0] != '0');
    }
    handle->has_stbtt = stbtt_InitFont(&handle->info, bytes, 0) != 0;
    handle->prefer_bitmap_rgba =
        fs_stb_str_contains_nocase(path, "emoji") || fs_stb_str_contains_nocase(path, "color");
    if (handle->has_stbtt) {
        const size_t face_off = (size_t)handle->info.fontstart;
        const uint8_t* face_bytes = (face_off < size) ? (bytes + face_off) : bytes;
        const size_t face_size = (face_off < size) ? (size - face_off) : size;
        handle->has_cbdt = fs_stb_cbdt_init(&handle->cbdt, face_bytes, face_size, 128u);
        (void)fs_stb_parse_colr_cpal(handle);
    } else {
        // Some color emoji fonts are not accepted by stbtt_InitFont but still expose CBDT/CBLC.
        handle->has_cbdt = fs_stb_cbdt_init(&handle->cbdt, bytes, size, 128u);
    }
    if (!handle->has_stbtt && !handle->has_cbdt) {
        free(bytes);
        free(handle);
        return NULL;
    }
    if (handle->cbdt_cache_debug) {
        fprintf(
            stderr,
            "[stb-font-load] path=%s has_stbtt=%d has_cbdt=%d prefer_bitmap=%d\n",
            path ? path : "(null)",
            handle->has_stbtt ? 1 : 0,
            handle->has_cbdt ? 1 : 0,
            handle->prefer_bitmap_rgba ? 1 : 0
        );
    }
    return handle;
}

static void* fs_stb_load_font_memory(const uint8_t* data, size_t size) {
    if (!data || size == 0) {
        return NULL;
    }
    uint8_t* bytes = (uint8_t*)malloc(size);
    if (!bytes) {
        return NULL;
    }
    memcpy(bytes, data, size);

    FS_StbFontHandle* handle = (FS_StbFontHandle*)calloc(1, sizeof(FS_StbFontHandle));
    if (!handle) {
        free(bytes);
        return NULL;
    }
    handle->bytes = bytes;
    handle->bytes_size = size;
    {
        const char* dbg = getenv("FS_STB_CBDT_CACHE_DEBUG");
        handle->cbdt_cache_debug = (dbg && dbg[0] != '\0' && dbg[0] != '0');
    }
    handle->has_stbtt = stbtt_InitFont(&handle->info, bytes, 0) != 0;
    handle->prefer_bitmap_rgba = false;
    if (handle->has_stbtt) {
        const size_t face_off = (size_t)handle->info.fontstart;
        const uint8_t* face_bytes = (face_off < size) ? (bytes + face_off) : bytes;
        const size_t face_size = (face_off < size) ? (size - face_off) : size;
        handle->has_cbdt = fs_stb_cbdt_init(&handle->cbdt, face_bytes, (size_t)face_size, 128u);
        (void)fs_stb_parse_colr_cpal(handle);
    } else {
        handle->has_cbdt = fs_stb_cbdt_init(&handle->cbdt, bytes, size, 128u);
    }
    if (!handle->has_stbtt && !handle->has_cbdt) {
        free(bytes);
        free(handle);
        return NULL;
    }
    return handle;
}

static void fs_stb_destroy_font(void* font_handle) {
    FS_StbFontHandle* handle = (FS_StbFontHandle*)font_handle;
    if (!handle) {
        return;
    }
    if (handle->cbdt_cache_debug && handle->has_cbdt) {
        fprintf(
            stderr,
            "[stb-cbdt-cache] entries=%u hits=%llu misses=%llu inserts=%llu decode_failures=%llu\n",
            (unsigned)handle->cbdt_cache_count,
            (unsigned long long)handle->cbdt_cache_hits,
            (unsigned long long)handle->cbdt_cache_misses,
            (unsigned long long)handle->cbdt_cache_inserts,
            (unsigned long long)handle->cbdt_cache_decode_failures
        );
    }
    fs_stb_cbdt_cache_clear(handle);
    if (handle->has_cbdt) {
        fs_stb_cbdt_deinit(&handle->cbdt);
        handle->has_cbdt = false;
    }
    fs_stb_free_colr_cpal(handle);
    free(handle->bytes);
    handle->bytes = NULL;
    handle->bytes_size = 0u;
    free(handle->rgba_expand_scratch);
    handle->rgba_expand_scratch = NULL;
    handle->rgba_expand_scratch_capacity = 0u;
    free(handle);
}

static bool fs_stb_get_glyph_sdf(
    void* font_handle,
    uint32_t codepoint,
    float font_px,
    FS_FontGlyphBitmap* out_glyph
) {
    if (!font_handle || !out_glyph || font_px <= 0.0f) {
        return false;
    }
    FS_StbFontHandle* handle = (FS_StbFontHandle*)font_handle;
    out_glyph->pixels = NULL;
    out_glyph->width = 0u;
    out_glyph->height = 0u;
    out_glyph->pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_SDF_R8;
    out_glyph->offset_x = 0;
    out_glyph->offset_y = 0;
    out_glyph->advance = 0.0f;
    out_glyph->sdf_radius_px = 8.0f;
    out_glyph->sdf_onedge = 0.5f;
    out_glyph->sdf_pixel_dist_scale = 1.0f;

    if (!handle->has_stbtt) {
        if (!handle->has_cbdt) {
            return false;
        }
        uint16_t glyph_id = 0u;
        if (!fs_stb_cbdt_find_glyph_for_codepoint(&handle->cbdt, codepoint, &glyph_id) || glyph_id == 0u) {
            return false;
        }
        return fs_stb_try_get_cbdt_glyph_rgba(handle, (uint32_t)glyph_id, font_px, out_glyph);
    }

    const float effective_px = fs_stb_effective_text_px(font_px);
    const float scale = stbtt_ScaleForPixelHeight(&handle->info, effective_px);
    const int glyph_index_i = stbtt_FindGlyphIndex(&handle->info, (int)codepoint);
    if (glyph_index_i == 0 && codepoint != 0u) {
        // Missing glyph: let caller skip it instead of advancing with .notdef metrics.
        return false;
    }
    int advance = 0;
    int lsb = 0;
    (void)lsb;
    stbtt_GetCodepointHMetrics(&handle->info, (int)codepoint, &advance, &lsb);
    out_glyph->advance = (float)advance * scale;

    const uint32_t glyph_index = (uint32_t)glyph_index_i;
    if (fs_stb_try_get_colr_glyph_rgba(handle, glyph_index, scale, out_glyph)) {
        return true;
    }
    if (fs_stb_try_get_cbdt_glyph_rgba(handle, glyph_index, font_px, out_glyph)) {
        return true;
    }

    const bool prefer_color_path = handle->prefer_bitmap_rgba || fs_stb_is_probably_emoji_codepoint(codepoint);
    if (prefer_color_path) {
        int bw = 0;
        int bh = 0;
        int xoff = 0;
        int yoff = 0;
        unsigned char* bitmap = stbtt_GetCodepointBitmap(&handle->info, scale, scale, (int)codepoint, &bw, &bh, &xoff, &yoff);
        if (bitmap && bw > 0 && bh > 0) {
            size_t needed = (size_t)bw * (size_t)bh * 4u;
            if (needed > handle->rgba_expand_scratch_capacity) {
                free(handle->rgba_expand_scratch);
                handle->rgba_expand_scratch = (uint8_t*)malloc(needed);
                handle->rgba_expand_scratch_capacity = needed;
            }
            uint8_t* rgba = handle->rgba_expand_scratch;
            if (rgba) {
                for (int i = 0; i < bw * bh; ++i) {
                    const uint8_t a = bitmap[i];
                    rgba[i * 4 + 0] = 255u;
                    rgba[i * 4 + 1] = 255u;
                    rgba[i * 4 + 2] = 255u;
                    rgba[i * 4 + 3] = a;
                }
                out_glyph->pixels = rgba;
                out_glyph->width = (uint32_t)bw;
                out_glyph->height = (uint32_t)bh;
                out_glyph->pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8;
                out_glyph->offset_x = xoff;
                out_glyph->offset_y = yoff;
                out_glyph->sdf_radius_px = 1.0f;
                out_glyph->sdf_onedge = 0.5f;
                out_glyph->sdf_pixel_dist_scale = 1.0f;
                stbtt_FreeBitmap(bitmap, NULL);
                return true;
            }
        }
        if (bitmap) {
            stbtt_FreeBitmap(bitmap, NULL);
        }
    }

    int padding = 8;
    unsigned char onedge_value = 180u;
    float pixel_dist_scale = 10.0f;
    fs_stb_pick_sdf_params(effective_px, &padding, &onedge_value, &pixel_dist_scale);
    out_glyph->sdf_radius_px = (float)padding;
    out_glyph->sdf_onedge = (float)onedge_value / 255.0f;
    out_glyph->sdf_pixel_dist_scale = pixel_dist_scale;

    int w = 0;
    int h = 0;
    int xoff = 0;
    int yoff = 0;
    unsigned char* sdf = stbtt_GetCodepointSDF(
        &handle->info,
        scale,
        (int)codepoint,
        padding,
        onedge_value,
        pixel_dist_scale,
        &w,
        &h,
        &xoff,
        &yoff
    );
    if (!sdf || w <= 0 || h <= 0) {
        if (sdf) {
            stbtt_FreeSDF(sdf, NULL);
        }
        return true;
    }
    out_glyph->pixels = sdf;
    out_glyph->width = (uint32_t)w;
    out_glyph->height = (uint32_t)h;
    out_glyph->offset_x = xoff;
    out_glyph->offset_y = yoff;
    return true;
}

static bool fs_stb_get_glyph_sdf_by_index(
    void* font_handle,
    uint32_t glyph_index,
    float font_px,
    FS_FontGlyphBitmap* out_glyph
) {
    if (!font_handle || !out_glyph || font_px <= 0.0f) {
        return false;
    }
    FS_StbFontHandle* handle = (FS_StbFontHandle*)font_handle;
    out_glyph->pixels = NULL;
    out_glyph->width = 0u;
    out_glyph->height = 0u;
    out_glyph->pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_SDF_R8;
    out_glyph->offset_x = 0;
    out_glyph->offset_y = 0;
    out_glyph->advance = 0.0f;
    out_glyph->sdf_radius_px = 8.0f;
    out_glyph->sdf_onedge = 0.5f;
    out_glyph->sdf_pixel_dist_scale = 1.0f;

    if (!handle->has_stbtt) {
        return fs_stb_try_get_cbdt_glyph_rgba(handle, glyph_index, font_px, out_glyph);
    }

    const float effective_px = fs_stb_effective_text_px(font_px);
    const float scale = stbtt_ScaleForPixelHeight(&handle->info, effective_px);
    int advance = 0;
    int lsb = 0;
    stbtt_GetGlyphHMetrics(&handle->info, (int)glyph_index, &advance, &lsb);
    (void)lsb;
    out_glyph->advance = (float)advance * scale;

    if (fs_stb_try_get_colr_glyph_rgba(handle, glyph_index, scale, out_glyph)) {
        return true;
    }
    if (fs_stb_try_get_cbdt_glyph_rgba(handle, glyph_index, font_px, out_glyph)) {
        return true;
    }

    if (handle->prefer_bitmap_rgba) {
        int bw = 0;
        int bh = 0;
        int xoff = 0;
        int yoff = 0;
        unsigned char* bitmap =
            stbtt_GetGlyphBitmap(&handle->info, scale, scale, (int)glyph_index, &bw, &bh, &xoff, &yoff);
        if (bitmap && bw > 0 && bh > 0) {
            size_t needed = (size_t)bw * (size_t)bh * 4u;
            if (needed > handle->rgba_expand_scratch_capacity) {
                free(handle->rgba_expand_scratch);
                handle->rgba_expand_scratch = (uint8_t*)malloc(needed);
                handle->rgba_expand_scratch_capacity = needed;
            }
            uint8_t* rgba = handle->rgba_expand_scratch;
            if (rgba) {
                for (int i = 0; i < bw * bh; ++i) {
                    const uint8_t a = bitmap[i];
                    rgba[i * 4 + 0] = 255u;
                    rgba[i * 4 + 1] = 255u;
                    rgba[i * 4 + 2] = 255u;
                    rgba[i * 4 + 3] = a;
                }
                out_glyph->pixels = rgba;
                out_glyph->width = (uint32_t)bw;
                out_glyph->height = (uint32_t)bh;
                out_glyph->pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8;
                out_glyph->offset_x = xoff;
                out_glyph->offset_y = yoff;
                out_glyph->sdf_radius_px = 1.0f;
                out_glyph->sdf_onedge = 0.5f;
                out_glyph->sdf_pixel_dist_scale = 1.0f;
                stbtt_FreeBitmap(bitmap, NULL);
                return true;
            }
        }
        if (bitmap) {
            stbtt_FreeBitmap(bitmap, NULL);
        }
    }

    int padding = 8;
    unsigned char onedge_value = 180u;
    float pixel_dist_scale = 10.0f;
    fs_stb_pick_sdf_params(effective_px, &padding, &onedge_value, &pixel_dist_scale);
    out_glyph->sdf_radius_px = (float)padding;
    out_glyph->sdf_onedge = (float)onedge_value / 255.0f;
    out_glyph->sdf_pixel_dist_scale = pixel_dist_scale;

    int w = 0;
    int h = 0;
    int xoff = 0;
    int yoff = 0;
    unsigned char* sdf = stbtt_GetGlyphSDF(
        &handle->info,
        scale,
        (int)glyph_index,
        padding,
        onedge_value,
        pixel_dist_scale,
        &w,
        &h,
        &xoff,
        &yoff
    );
    if (!sdf || w <= 0 || h <= 0) {
        if (sdf) {
            stbtt_FreeSDF(sdf, NULL);
        }
        return true;
    }
    out_glyph->pixels = sdf;
    out_glyph->width = (uint32_t)w;
    out_glyph->height = (uint32_t)h;
    out_glyph->offset_x = xoff;
    out_glyph->offset_y = yoff;
    return true;
}

static float fs_stb_get_kerning_advance(
    void* font_handle,
    uint32_t left_codepoint,
    uint32_t right_codepoint,
    float font_px
) {
    if (!font_handle || font_px <= 0.0f) {
        return 0.0f;
    }
    FS_StbFontHandle* handle = (FS_StbFontHandle*)font_handle;
    if (!handle->has_stbtt) {
        return 0.0f;
    }
    const float effective_px = fs_stb_effective_text_px(font_px);
    const float scale = stbtt_ScaleForPixelHeight(&handle->info, effective_px);
    const int kern = stbtt_GetCodepointKernAdvance(
        &handle->info,
        (int)left_codepoint,
        (int)right_codepoint
    );
    return (float)kern * scale;
}

static bool fs_stb_get_vertical_metrics(
    void* font_handle,
    float font_px,
    FS_FontVerticalMetrics* out_metrics
) {
    if (!font_handle || !out_metrics || font_px <= 0.0f) {
        return false;
    }
    FS_StbFontHandle* handle = (FS_StbFontHandle*)font_handle;
    if (!handle->has_stbtt) {
        return false;
    }
    const float effective_px = fs_stb_effective_text_px(font_px);
    const float scale = stbtt_ScaleForPixelHeight(&handle->info, effective_px);
    int ascent_i = 0;
    int descent_i = 0;
    int line_gap_i = 0;
    stbtt_GetFontVMetrics(&handle->info, &ascent_i, &descent_i, &line_gap_i);
    const float ascent = (float)ascent_i * scale;
    const float descent = (float)(-descent_i) * scale;
    float line_height = (float)(ascent_i - descent_i + line_gap_i) * scale;
    if (!(ascent > 0.0f) || !(descent >= 0.0f)) {
        return false;
    }
    if (!(line_height > 0.0f)) {
        line_height = ascent + descent;
    } else if (line_height < ascent + descent) {
        line_height = ascent + descent;
    }
    out_metrics->ascent = ascent;
    out_metrics->descent = descent;
    out_metrics->line_height = line_height;
    return true;
}

static void fs_stb_free_glyph_pixels(uint8_t* pixels) {
    if (pixels) {
        free(pixels);
    }
}

static const FS_FontBackend g_stb_font_backend = {
    .name = "stb_truetype",
    .load_font_file = fs_stb_load_font_file,
    .load_font_memory = fs_stb_load_font_memory,
    .destroy_font = fs_stb_destroy_font,
    .get_glyph_sdf = fs_stb_get_glyph_sdf,
    .get_glyph_sdf_by_index = fs_stb_get_glyph_sdf_by_index,
    .get_kerning_advance = fs_stb_get_kerning_advance,
    .get_vertical_metrics = fs_stb_get_vertical_metrics,
    .shape_text_utf8 = NULL,
    .free_shaped_text = NULL,
    .free_glyph_pixels = fs_stb_free_glyph_pixels
};

const FS_FontBackend* fs_get_stb_font_backend(void) {
    return &g_stb_font_backend;
}
