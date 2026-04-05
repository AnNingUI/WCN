#ifndef WCN_FULLSTACK_STB_CBDT_PARSER_H
#define WCN_FULLSTACK_STB_CBDT_PARSER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct FS_StbCbdtMetrics {
    uint8_t width;
    uint8_t height;
    int8_t bearing_x;
    int8_t bearing_y;
    uint8_t advance;
    bool has_metrics;
} FS_StbCbdtMetrics;

typedef struct FS_StbCbdtGlyphPngView {
    const uint8_t* png_bytes;
    uint32_t png_size;
    uint16_t image_format;
    uint8_t ppem_x;
    uint8_t ppem_y;
    FS_StbCbdtMetrics metrics;
} FS_StbCbdtGlyphPngView;

typedef struct FS_StbCbdtSubtable {
    uint16_t first_glyph;
    uint16_t last_glyph;
    uint16_t index_format;
    uint16_t image_format;
    uint32_t image_data_offset;
    uint32_t location_array_offset;
    uint32_t location_count;
} FS_StbCbdtSubtable;

typedef struct FS_StbCbdtStrike {
    uint8_t ppem_x;
    uint8_t ppem_y;
    uint8_t bit_depth;
    uint8_t _pad0;
    uint32_t subtable_count;
    FS_StbCbdtSubtable* subtables;
} FS_StbCbdtStrike;

typedef struct FS_StbCbdtFont {
    const uint8_t* sfnt;
    size_t sfnt_size;
    uint32_t cblc_offset;
    uint32_t cblc_length;
    uint32_t cbdt_offset;
    uint32_t cbdt_length;
    FS_StbCbdtStrike strike;
} FS_StbCbdtFont;

bool fs_stb_cbdt_init(
    FS_StbCbdtFont* font,
    const uint8_t* sfnt_bytes,
    size_t sfnt_size,
    uint16_t requested_ppem
);

void fs_stb_cbdt_deinit(FS_StbCbdtFont* font);

bool fs_stb_cbdt_get_glyph_png(
    const FS_StbCbdtFont* font,
    uint16_t glyph_id,
    FS_StbCbdtGlyphPngView* out_png
);

bool fs_stb_cbdt_find_glyph_for_codepoint(
    const FS_StbCbdtFont* font,
    uint32_t codepoint,
    uint16_t* out_glyph_id
);

uint8_t fs_stb_cbdt_strike_ppem_x(const FS_StbCbdtFont* font);
uint8_t fs_stb_cbdt_strike_ppem_y(const FS_StbCbdtFont* font);

#endif
