#include "fullstack_stb_cbdt_parser.h"

#include <stdlib.h>
#include <string.h>

static bool range_ok(size_t size, size_t offset, size_t need) {
    return offset <= size && need <= (size - offset);
}

static bool read_be_u16(const uint8_t* p, size_t size, size_t off, uint16_t* out_v) {
    if (!out_v || !range_ok(size, off, 2u)) {
        return false;
    }
    *out_v = (uint16_t)(((uint16_t)p[off] << 8u) | (uint16_t)p[off + 1u]);
    return true;
}

static bool read_be_u32(const uint8_t* p, size_t size, size_t off, uint32_t* out_v) {
    if (!out_v || !range_ok(size, off, 4u)) {
        return false;
    }
    *out_v =
        ((uint32_t)p[off] << 24u) |
        ((uint32_t)p[off + 1u] << 16u) |
        ((uint32_t)p[off + 2u] << 8u) |
        (uint32_t)p[off + 3u];
    return true;
}

static bool find_sfnt_table(
    const uint8_t* sfnt,
    size_t sfnt_size,
    const char tag[4],
    uint32_t* out_offset,
    uint32_t* out_length
) {
    if (!sfnt || !tag || !out_offset || !out_length || sfnt_size < 12u) {
        return false;
    }
    uint16_t num_tables = 0u;
    if (!read_be_u16(sfnt, sfnt_size, 4u, &num_tables)) {
        return false;
    }
    const size_t dir = 12u;
    for (size_t i = 0u; i < (size_t)num_tables; ++i) {
        const size_t rec = dir + i * 16u;
        if (!range_ok(sfnt_size, rec, 16u)) {
            return false;
        }
        if (sfnt[rec + 0u] != (uint8_t)tag[0] ||
            sfnt[rec + 1u] != (uint8_t)tag[1] ||
            sfnt[rec + 2u] != (uint8_t)tag[2] ||
            sfnt[rec + 3u] != (uint8_t)tag[3]) {
            continue;
        }
        uint32_t off = 0u;
        uint32_t len = 0u;
        if (!read_be_u32(sfnt, sfnt_size, rec + 8u, &off) ||
            !read_be_u32(sfnt, sfnt_size, rec + 12u, &len)) {
            return false;
        }
        if (!range_ok(sfnt_size, (size_t)off, (size_t)len)) {
            return false;
        }
        *out_offset = off;
        *out_length = len;
        return true;
    }
    return false;
}

static bool cmap_lookup_format12(
    const uint8_t* cmap_subtable,
    size_t subtable_size,
    uint32_t codepoint,
    uint16_t* out_gid
) {
    if (!cmap_subtable || !out_gid || subtable_size < 16u) {
        return false;
    }
    const uint32_t n_groups =
        ((uint32_t)cmap_subtable[12u] << 24u) |
        ((uint32_t)cmap_subtable[13u] << 16u) |
        ((uint32_t)cmap_subtable[14u] << 8u) |
        (uint32_t)cmap_subtable[15u];
    if (16u + (size_t)n_groups * 12u > subtable_size) {
        return false;
    }
    for (uint32_t i = 0u; i < n_groups; ++i) {
        const size_t off = 16u + (size_t)i * 12u;
        const uint32_t start =
            ((uint32_t)cmap_subtable[off + 0u] << 24u) |
            ((uint32_t)cmap_subtable[off + 1u] << 16u) |
            ((uint32_t)cmap_subtable[off + 2u] << 8u) |
            (uint32_t)cmap_subtable[off + 3u];
        const uint32_t end =
            ((uint32_t)cmap_subtable[off + 4u] << 24u) |
            ((uint32_t)cmap_subtable[off + 5u] << 16u) |
            ((uint32_t)cmap_subtable[off + 6u] << 8u) |
            (uint32_t)cmap_subtable[off + 7u];
        const uint32_t start_gid =
            ((uint32_t)cmap_subtable[off + 8u] << 24u) |
            ((uint32_t)cmap_subtable[off + 9u] << 16u) |
            ((uint32_t)cmap_subtable[off + 10u] << 8u) |
            (uint32_t)cmap_subtable[off + 11u];
        if (codepoint >= start && codepoint <= end) {
            const uint32_t gid = start_gid + (codepoint - start);
            if (gid > 0xFFFFu) {
                return false;
            }
            *out_gid = (uint16_t)gid;
            return true;
        }
    }
    return false;
}

static bool cmap_lookup_format4(
    const uint8_t* cmap_subtable,
    size_t subtable_size,
    uint32_t codepoint,
    uint16_t* out_gid
) {
    if (!cmap_subtable || !out_gid || subtable_size < 24u || codepoint > 0xFFFFu) {
        return false;
    }
    const uint16_t seg_count_x2 = (uint16_t)(((uint16_t)cmap_subtable[6u] << 8u) | (uint16_t)cmap_subtable[7u]);
    if ((seg_count_x2 & 1u) != 0u || seg_count_x2 == 0u) {
        return false;
    }
    const uint16_t seg_count = (uint16_t)(seg_count_x2 / 2u);
    const size_t end_code_off = 14u;
    const size_t start_code_off = end_code_off + (size_t)seg_count * 2u + 2u;
    const size_t id_delta_off = start_code_off + (size_t)seg_count * 2u;
    const size_t id_range_off_off = id_delta_off + (size_t)seg_count * 2u;
    if (id_range_off_off + (size_t)seg_count * 2u > subtable_size) {
        return false;
    }

    const uint16_t cp16 = (uint16_t)codepoint;
    for (uint16_t i = 0u; i < seg_count; ++i) {
        const size_t e_off = end_code_off + (size_t)i * 2u;
        const size_t s_off = start_code_off + (size_t)i * 2u;
        const size_t d_off = id_delta_off + (size_t)i * 2u;
        const size_t r_off = id_range_off_off + (size_t)i * 2u;
        const uint16_t end_code = (uint16_t)(((uint16_t)cmap_subtable[e_off] << 8u) | (uint16_t)cmap_subtable[e_off + 1u]);
        const uint16_t start_code = (uint16_t)(((uint16_t)cmap_subtable[s_off] << 8u) | (uint16_t)cmap_subtable[s_off + 1u]);
        if (cp16 < start_code || cp16 > end_code) {
            continue;
        }
        const int16_t id_delta = (int16_t)(((uint16_t)cmap_subtable[d_off] << 8u) | (uint16_t)cmap_subtable[d_off + 1u]);
        const uint16_t id_range_offset = (uint16_t)(((uint16_t)cmap_subtable[r_off] << 8u) | (uint16_t)cmap_subtable[r_off + 1u]);
        if (id_range_offset == 0u) {
            *out_gid = (uint16_t)((cp16 + (uint16_t)id_delta) & 0xFFFFu);
            return *out_gid != 0u;
        }
        const size_t glyph_index_off =
            r_off + (size_t)id_range_offset + (size_t)(cp16 - start_code) * 2u;
        if (!range_ok(subtable_size, glyph_index_off, 2u)) {
            return false;
        }
        uint16_t glyph_id = (uint16_t)(((uint16_t)cmap_subtable[glyph_index_off] << 8u) |
                                       (uint16_t)cmap_subtable[glyph_index_off + 1u]);
        if (glyph_id == 0u) {
            return false;
        }
        glyph_id = (uint16_t)((glyph_id + (uint16_t)id_delta) & 0xFFFFu);
        *out_gid = glyph_id;
        return true;
    }
    return false;
}

static uint32_t abs_u32_diff(uint32_t a, uint32_t b) {
    return (a > b) ? (a - b) : (b - a);
}

static bool parse_strike_subtables(
    FS_StbCbdtFont* font,
    size_t strike_base
) {
    uint32_t index_array_rel = 0u;
    uint32_t subtable_count = 0u;
    if (!read_be_u32(font->sfnt, font->sfnt_size, strike_base + 0u, &index_array_rel) ||
        !read_be_u32(font->sfnt, font->sfnt_size, strike_base + 8u, &subtable_count)) {
        return false;
    }
    if (subtable_count == 0u) {
        return false;
    }
    const size_t cblc_base = (size_t)font->cblc_offset;
    const size_t index_array = cblc_base + (size_t)index_array_rel;
    if (!range_ok(font->sfnt_size, index_array, (size_t)subtable_count * 8u)) {
        return false;
    }

    FS_StbCbdtSubtable* subtables =
        (FS_StbCbdtSubtable*)calloc((size_t)subtable_count, sizeof(FS_StbCbdtSubtable));
    if (!subtables) {
        return false;
    }

    uint32_t write_count = 0u;
    for (uint32_t i = 0u; i < subtable_count; ++i) {
        const size_t rec = index_array + (size_t)i * 8u;
        uint16_t first = 0u;
        uint16_t last = 0u;
        uint32_t add_off = 0u;
        if (!read_be_u16(font->sfnt, font->sfnt_size, rec + 0u, &first) ||
            !read_be_u16(font->sfnt, font->sfnt_size, rec + 2u, &last) ||
            !read_be_u32(font->sfnt, font->sfnt_size, rec + 4u, &add_off)) {
            free(subtables);
            return false;
        }
        if (last < first) {
            continue;
        }
        const size_t sub_off = index_array + (size_t)add_off;
        uint16_t index_fmt = 0u;
        uint16_t image_fmt = 0u;
        uint32_t image_data_off = 0u;
        if (!read_be_u16(font->sfnt, font->sfnt_size, sub_off + 0u, &index_fmt) ||
            !read_be_u16(font->sfnt, font->sfnt_size, sub_off + 2u, &image_fmt) ||
            !read_be_u32(font->sfnt, font->sfnt_size, sub_off + 4u, &image_data_off)) {
            free(subtables);
            return false;
        }
        if (!(index_fmt == 1u || index_fmt == 3u)) {
            continue;
        }
        if (!(image_fmt == 17u || image_fmt == 18u || image_fmt == 19u)) {
            continue;
        }

        FS_StbCbdtSubtable st = {0};
        st.first_glyph = first;
        st.last_glyph = last;
        st.index_format = index_fmt;
        st.image_format = image_fmt;
        st.image_data_offset = image_data_off;
        st.location_count = (uint32_t)last - (uint32_t)first + 2u;
        st.location_array_offset = (uint32_t)(sub_off + 8u);
        const size_t loc_item_size = (index_fmt == 1u) ? 4u : 2u;
        if (!range_ok(font->sfnt_size, (size_t)st.location_array_offset, (size_t)st.location_count * loc_item_size)) {
            continue;
        }
        subtables[write_count++] = st;
    }

    if (write_count == 0u) {
        free(subtables);
        return false;
    }
    font->strike.subtables = subtables;
    font->strike.subtable_count = write_count;
    return true;
}

bool fs_stb_cbdt_init(
    FS_StbCbdtFont* font,
    const uint8_t* sfnt_bytes,
    size_t sfnt_size,
    uint16_t requested_ppem
) {
    if (!font || !sfnt_bytes || sfnt_size < 64u) {
        return false;
    }
    memset(font, 0, sizeof(*font));
    font->sfnt = sfnt_bytes;
    font->sfnt_size = sfnt_size;

    if (!find_sfnt_table(sfnt_bytes, sfnt_size, "CBLC", &font->cblc_offset, &font->cblc_length) ||
        !find_sfnt_table(sfnt_bytes, sfnt_size, "CBDT", &font->cbdt_offset, &font->cbdt_length)) {
        fs_stb_cbdt_deinit(font);
        return false;
    }

    const size_t cblc = (size_t)font->cblc_offset;
    uint32_t num_sizes = 0u;
    if (!read_be_u32(sfnt_bytes, sfnt_size, cblc + 4u, &num_sizes) || num_sizes == 0u) {
        fs_stb_cbdt_deinit(font);
        return false;
    }
    if (!range_ok(sfnt_size, cblc + 8u, (size_t)num_sizes * 48u)) {
        fs_stb_cbdt_deinit(font);
        return false;
    }

    uint32_t best_idx = 0u;
    uint32_t best_delta = 0xFFFFFFFFu;
    for (uint32_t i = 0u; i < num_sizes; ++i) {
        const size_t s = cblc + 8u + (size_t)i * 48u;
        const uint8_t ppem_y = sfnt_bytes[s + 45u];
        const uint32_t d = abs_u32_diff((uint32_t)ppem_y, (uint32_t)requested_ppem);
        if (d < best_delta) {
            best_delta = d;
            best_idx = i;
        }
    }

    const size_t strike_base = cblc + 8u + (size_t)best_idx * 48u;
    font->strike.ppem_x = sfnt_bytes[strike_base + 44u];
    font->strike.ppem_y = sfnt_bytes[strike_base + 45u];
    font->strike.bit_depth = sfnt_bytes[strike_base + 46u];
    if (!parse_strike_subtables(font, strike_base)) {
        fs_stb_cbdt_deinit(font);
        return false;
    }
    return true;
}

void fs_stb_cbdt_deinit(FS_StbCbdtFont* font) {
    if (!font) {
        return;
    }
    free(font->strike.subtables);
    font->strike.subtables = NULL;
    font->strike.subtable_count = 0u;
    font->strike.ppem_x = 0u;
    font->strike.ppem_y = 0u;
    font->strike.bit_depth = 0u;
    font->sfnt = NULL;
    font->sfnt_size = 0u;
    font->cblc_offset = 0u;
    font->cblc_length = 0u;
    font->cbdt_offset = 0u;
    font->cbdt_length = 0u;
}

static bool read_location_pair(
    const FS_StbCbdtFont* font,
    const FS_StbCbdtSubtable* st,
    uint16_t glyph_id,
    uint32_t* out_off0,
    uint32_t* out_off1
) {
    if (!font || !st || !out_off0 || !out_off1) {
        return false;
    }
    if (glyph_id < st->first_glyph || glyph_id > st->last_glyph) {
        return false;
    }
    const uint32_t index = (uint32_t)glyph_id - (uint32_t)st->first_glyph;
    if (index + 1u >= st->location_count) {
        return false;
    }
    const size_t base = (size_t)st->location_array_offset;
    if (st->index_format == 1u) {
        uint32_t a = 0u;
        uint32_t b = 0u;
        if (!read_be_u32(font->sfnt, font->sfnt_size, base + (size_t)index * 4u, &a) ||
            !read_be_u32(font->sfnt, font->sfnt_size, base + (size_t)(index + 1u) * 4u, &b)) {
            return false;
        }
        *out_off0 = a;
        *out_off1 = b;
        return true;
    }
    if (st->index_format == 3u) {
        uint16_t a = 0u;
        uint16_t b = 0u;
        if (!read_be_u16(font->sfnt, font->sfnt_size, base + (size_t)index * 2u, &a) ||
            !read_be_u16(font->sfnt, font->sfnt_size, base + (size_t)(index + 1u) * 2u, &b)) {
            return false;
        }
        *out_off0 = (uint32_t)a;
        *out_off1 = (uint32_t)b;
        return true;
    }
    return false;
}

static bool parse_metrics_and_png(
    const uint8_t* blob,
    size_t blob_size,
    uint16_t image_format,
    FS_StbCbdtGlyphPngView* out_png
) {
    if (!blob || !out_png) {
        return false;
    }
    memset(out_png, 0, sizeof(*out_png));
    uint32_t png_len = 0u;
    size_t png_off = 0u;
    if (image_format == 17u) {
        if (!range_ok(blob_size, 0u, 9u)) {
            return false;
        }
        out_png->metrics.width = blob[0u];
        out_png->metrics.height = blob[1u];
        out_png->metrics.bearing_x = (int8_t)blob[2u];
        out_png->metrics.bearing_y = (int8_t)blob[3u];
        out_png->metrics.advance = blob[4u];
        out_png->metrics.has_metrics = true;
        png_len =
            ((uint32_t)blob[5u] << 24u) |
            ((uint32_t)blob[6u] << 16u) |
            ((uint32_t)blob[7u] << 8u) |
            (uint32_t)blob[8u];
        png_off = 9u;
    } else if (image_format == 18u) {
        if (!range_ok(blob_size, 0u, 12u)) {
            return false;
        }
        out_png->metrics.width = blob[0u];
        out_png->metrics.height = blob[1u];
        out_png->metrics.bearing_x = (int8_t)blob[2u];
        out_png->metrics.bearing_y = (int8_t)blob[3u];
        out_png->metrics.advance = blob[4u];
        out_png->metrics.has_metrics = true;
        png_len =
            ((uint32_t)blob[8u] << 24u) |
            ((uint32_t)blob[9u] << 16u) |
            ((uint32_t)blob[10u] << 8u) |
            (uint32_t)blob[11u];
        png_off = 12u;
    } else if (image_format == 19u) {
        if (!range_ok(blob_size, 0u, 4u)) {
            return false;
        }
        png_len =
            ((uint32_t)blob[0u] << 24u) |
            ((uint32_t)blob[1u] << 16u) |
            ((uint32_t)blob[2u] << 8u) |
            (uint32_t)blob[3u];
        png_off = 4u;
        out_png->metrics.has_metrics = false;
    } else {
        return false;
    }
    if (png_len == 0u || !range_ok(blob_size, png_off, (size_t)png_len)) {
        return false;
    }
    out_png->png_bytes = blob + png_off;
    out_png->png_size = png_len;
    out_png->image_format = image_format;
    return true;
}

bool fs_stb_cbdt_get_glyph_png(
    const FS_StbCbdtFont* font,
    uint16_t glyph_id,
    FS_StbCbdtGlyphPngView* out_png
) {
    if (!font || !out_png || !font->sfnt || !font->strike.subtables) {
        return false;
    }
    for (uint32_t i = 0u; i < font->strike.subtable_count; ++i) {
        const FS_StbCbdtSubtable* st = &font->strike.subtables[i];
        uint32_t off0 = 0u;
        uint32_t off1 = 0u;
        if (!read_location_pair(font, st, glyph_id, &off0, &off1)) {
            continue;
        }
        if (off1 <= off0) {
            continue;
        }
        const size_t cbdt_base = (size_t)font->cbdt_offset + (size_t)st->image_data_offset;
        const size_t glyph_off = cbdt_base + (size_t)off0;
        const size_t glyph_end = cbdt_base + (size_t)off1;
        if (glyph_end <= glyph_off || !range_ok(font->sfnt_size, glyph_off, glyph_end - glyph_off)) {
            continue;
        }
        FS_StbCbdtGlyphPngView view = {0};
        if (!parse_metrics_and_png(font->sfnt + glyph_off, glyph_end - glyph_off, st->image_format, &view)) {
            continue;
        }
        view.ppem_x = font->strike.ppem_x;
        view.ppem_y = font->strike.ppem_y;
        *out_png = view;
        return true;
    }
    return false;
}

bool fs_stb_cbdt_find_glyph_for_codepoint(
    const FS_StbCbdtFont* font,
    uint32_t codepoint,
    uint16_t* out_glyph_id
) {
    if (!font || !font->sfnt || !out_glyph_id) {
        return false;
    }
    uint32_t cmap_off = 0u;
    uint32_t cmap_len = 0u;
    if (!find_sfnt_table(font->sfnt, font->sfnt_size, "cmap", &cmap_off, &cmap_len)) {
        return false;
    }
    const size_t base = (size_t)cmap_off;
    if (!range_ok(font->sfnt_size, base, (size_t)cmap_len) || cmap_len < 4u) {
        return false;
    }
    uint16_t num_tables = 0u;
    if (!read_be_u16(font->sfnt, font->sfnt_size, base + 2u, &num_tables)) {
        return false;
    }
    if (!range_ok(font->sfnt_size, base + 4u, (size_t)num_tables * 8u)) {
        return false;
    }

    const uint8_t* best_fmt12 = NULL;
    size_t best_fmt12_size = 0u;
    const uint8_t* best_fmt4 = NULL;
    size_t best_fmt4_size = 0u;

    for (uint16_t i = 0u; i < num_tables; ++i) {
        const size_t rec = base + 4u + (size_t)i * 8u;
        uint16_t platform_id = 0u;
        uint16_t encoding_id = 0u;
        uint32_t sub_rel = 0u;
        if (!read_be_u16(font->sfnt, font->sfnt_size, rec + 0u, &platform_id) ||
            !read_be_u16(font->sfnt, font->sfnt_size, rec + 2u, &encoding_id) ||
            !read_be_u32(font->sfnt, font->sfnt_size, rec + 4u, &sub_rel)) {
            return false;
        }
        const size_t sub = base + (size_t)sub_rel;
        const size_t cmap_end = base + (size_t)cmap_len;
        if (!range_ok(font->sfnt_size, sub, 4u) || sub > cmap_end) {
            continue;
        }
        uint16_t fmt = 0u;
        if (!read_be_u16(font->sfnt, font->sfnt_size, sub + 0u, &fmt)) {
            continue;
        }
        size_t sub_size = 0u;
        if (fmt == 12u) {
            uint32_t len32 = 0u;
            if (!read_be_u32(font->sfnt, font->sfnt_size, sub + 4u, &len32) || len32 < 16u) {
                continue;
            }
            sub_size = (size_t)len32;
        } else if (fmt == 4u) {
            uint16_t len16 = 0u;
            if (!read_be_u16(font->sfnt, font->sfnt_size, sub + 2u, &len16) || len16 < 24u) {
                continue;
            }
            sub_size = (size_t)len16;
        } else {
            continue;
        }
        if (!range_ok(font->sfnt_size, sub, sub_size)) {
            continue;
        }
        const bool is_unicode =
            (platform_id == 0u) ||
            (platform_id == 3u && (encoding_id == 1u || encoding_id == 10u));
        if (!is_unicode) {
            continue;
        }

        if (fmt == 12u && !best_fmt12) {
            best_fmt12 = font->sfnt + sub;
            best_fmt12_size = sub_size;
        } else if (fmt == 4u && !best_fmt4) {
            best_fmt4 = font->sfnt + sub;
            best_fmt4_size = sub_size;
        }
    }

    uint16_t gid = 0u;
    if (best_fmt12 && cmap_lookup_format12(best_fmt12, best_fmt12_size, codepoint, &gid) && gid != 0u) {
        *out_glyph_id = gid;
        return true;
    }
    if (best_fmt4 && cmap_lookup_format4(best_fmt4, best_fmt4_size, codepoint, &gid) && gid != 0u) {
        *out_glyph_id = gid;
        return true;
    }
    return false;
}

uint8_t fs_stb_cbdt_strike_ppem_x(const FS_StbCbdtFont* font) {
    return font ? font->strike.ppem_x : 0u;
}

uint8_t fs_stb_cbdt_strike_ppem_y(const FS_StbCbdtFont* font) {
    return font ? font->strike.ppem_y : 0u;
}
