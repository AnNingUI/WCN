#ifndef STB_CBDT_PARSER_H
#define STB_CBDT_PARSER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct STBCBDTMetrics {
    uint8_t width;
    uint8_t height;
    int8_t bearing_x;
    int8_t bearing_y;
    uint8_t advance;
    bool has_metrics;
} STBCBDTMetrics;

typedef struct STBCBDTGlyphPngView {
    const uint8_t* png_bytes;
    uint32_t png_size;
    uint16_t image_format;
    uint8_t ppem_x;
    uint8_t ppem_y;
    STBCBDTMetrics metrics;
} STBCBDTGlyphPngView;

typedef struct STBCBDTSubtable {
    uint16_t first_glyph;
    uint16_t last_glyph;
    uint16_t index_format;
    uint16_t image_format;
    uint32_t image_data_offset;
    uint32_t location_array_offset;
    uint32_t location_count;
} STBCBDTSubtable;

typedef struct STBCBDTStrike {
    uint8_t ppem_x;
    uint8_t ppem_y;
    uint8_t bit_depth;
    uint8_t _pad0;
    uint32_t subtable_count;
    STBCBDTSubtable* subtables;
} STBCBDTStrike;

typedef struct STBCBDTFont {
    const uint8_t* sfnt;
    size_t sfnt_size;
    uint32_t cblc_offset;
    uint32_t cblc_length;
    uint32_t cbdt_offset;
    uint32_t cbdt_length;
    STBCBDTStrike strike;
} STBCBDTFont;

bool stbcbdt_init(
    STBCBDTFont* font,
    const uint8_t* sfnt_bytes,
    size_t sfnt_size,
    uint16_t requested_ppem
);

void stbcbdt_deinit(STBCBDTFont* font);

bool stbcbdt_get_glyph_png(
    const STBCBDTFont* font,
    uint16_t glyph_id,
    STBCBDTGlyphPngView* out_png
);

bool stbcbdt_find_glyph_for_codepoint(
    const STBCBDTFont* font,
    uint32_t codepoint,
    uint16_t* out_glyph_id
);

uint8_t stbcbdt_strike_ppem_x(const STBCBDTFont* font);
uint8_t stbcbdt_strike_ppem_y(const STBCBDTFont* font);
uint8_t stbcbdt_strike_bit_depth(const STBCBDTFont* font);

#endif
