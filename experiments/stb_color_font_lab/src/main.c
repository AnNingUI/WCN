#include "../include/stb_cbdt_parser.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_u32_arg(const char* s, uint32_t* out_v) {
    if (!s || !out_v) {
        return 0;
    }
    char* end = NULL;
    const unsigned long v = strtoul(s, &end, 0);
    if (!end || *end != '\0') {
        return 0;
    }
    *out_v = (uint32_t)v;
    return 1;
}

static int read_file_bytes(const char* path, uint8_t** out_bytes, size_t* out_size) {
    if (!path || !out_bytes || !out_size) {
        return 0;
    }
    *out_bytes = NULL;
    *out_size = 0u;
    FILE* f = fopen(path, "rb");
    if (!f) {
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    const long sz = ftell(f);
    if (sz <= 0) {
        fclose(f);
        return 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }
    uint8_t* bytes = (uint8_t*)malloc((size_t)sz);
    if (!bytes) {
        fclose(f);
        return 0;
    }
    if (fread(bytes, 1u, (size_t)sz, f) != (size_t)sz) {
        free(bytes);
        fclose(f);
        return 0;
    }
    fclose(f);
    *out_bytes = bytes;
    *out_size = (size_t)sz;
    return 1;
}

static int write_file_bytes(const char* path, const uint8_t* bytes, size_t size) {
    if (!path || !bytes || size == 0u) {
        return 0;
    }
    FILE* f = fopen(path, "wb");
    if (!f) {
        return 0;
    }
    const size_t n = fwrite(bytes, 1u, size, f);
    fclose(f);
    return n == size;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <font.ttf> <codepoint(hex or dec)> [output.png] [ppem]\n", argv[0]);
        return 2;
    }

    const char* font_path = argv[1];
    uint32_t codepoint = 0u;
    if (!parse_u32_arg(argv[2], &codepoint)) {
        fprintf(stderr, "Invalid codepoint: %s\n", argv[2]);
        return 2;
    }
    const char* out_path = (argc >= 4) ? argv[3] : "glyph.png";
    uint32_t requested_ppem = 128u;
    if (argc >= 5) {
        (void)parse_u32_arg(argv[4], &requested_ppem);
    }

    uint8_t* bytes = NULL;
    size_t size = 0u;
    if (!read_file_bytes(font_path, &bytes, &size)) {
        fprintf(stderr, "Failed to read font: %s\n", font_path);
        return 1;
    }

    STBCBDTFont cbdt = {0};
    if (!stbcbdt_init(&cbdt, bytes, size, (uint16_t)requested_ppem)) {
        fprintf(stderr, "stbcbdt_init failed (no usable CBDT/CBLC)\n");
        free(bytes);
        return 1;
    }

    uint16_t glyph_index = 0u;
    if (!stbcbdt_find_glyph_for_codepoint(&cbdt, codepoint, &glyph_index) || glyph_index == 0u) {
        fprintf(stderr, "Glyph missing in cmap for U+%04X\n", (unsigned)codepoint);
        stbcbdt_deinit(&cbdt);
        free(bytes);
        return 1;
    }
    printf("codepoint=U+%04X glyph_index=%u\n", (unsigned)codepoint, (unsigned)glyph_index);
    printf(
        "selected_strike: ppem=(%u,%u) bitDepth=%u\n",
        (unsigned)stbcbdt_strike_ppem_x(&cbdt),
        (unsigned)stbcbdt_strike_ppem_y(&cbdt),
        (unsigned)stbcbdt_strike_bit_depth(&cbdt)
    );

    STBCBDTGlyphPngView view = {0};
    if (!stbcbdt_get_glyph_png(&cbdt, glyph_index, &view)) {
        fprintf(stderr, "No CBDT glyph PNG for glyph_index=%u\n", (unsigned)glyph_index);
        stbcbdt_deinit(&cbdt);
        free(bytes);
        return 1;
    }

    printf(
        "imageFormat=%u pngSize=%u metrics=%s w=%u h=%u bx=%d by=%d adv=%u\n",
        (unsigned)view.image_format,
        (unsigned)view.png_size,
        view.metrics.has_metrics ? "yes" : "no",
        (unsigned)view.metrics.width,
        (unsigned)view.metrics.height,
        (int)view.metrics.bearing_x,
        (int)view.metrics.bearing_y,
        (unsigned)view.metrics.advance
    );

    if (!write_file_bytes(out_path, view.png_bytes, (size_t)view.png_size)) {
        fprintf(stderr, "Failed writing PNG: %s\n", out_path);
        stbcbdt_deinit(&cbdt);
        free(bytes);
        return 1;
    }
    printf("wrote: %s\n", out_path);

    stbcbdt_deinit(&cbdt);
    free(bytes);
    return 0;
}
