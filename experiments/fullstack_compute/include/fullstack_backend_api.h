#ifndef WCN_FULLSTACK_BACKEND_API_H
#define WCN_FULLSTACK_BACKEND_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct FS_ImageBackend {
    const char* name;
    bool (*decode_memory)(
        const uint8_t* encoded_bytes,
        size_t encoded_size,
        uint8_t** out_rgba_pixels,
        uint32_t* out_width,
        uint32_t* out_height
    );
    bool (*decode_file)(
        const char* path,
        uint8_t** out_rgba_pixels,
        uint32_t* out_width,
        uint32_t* out_height
    );
    void (*free_image)(uint8_t* rgba_pixels);
} FS_ImageBackend;

typedef struct FS_FontGlyphBitmap {
    uint8_t* pixels;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    int32_t offset_x;
    int32_t offset_y;
    float advance;
    float sdf_radius_px;
    float sdf_onedge;
    float sdf_pixel_dist_scale;
} FS_FontGlyphBitmap;

enum {
    FS_FONT_GLYPH_PIXEL_FORMAT_SDF_R8 = 0u,
    FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8 = 1u
};

typedef struct FS_ShapedGlyph {
    uint32_t glyph_index;
    int32_t x_offset_26d6;
    int32_t y_offset_26d6;
    int32_t x_advance_26d6;
    int32_t y_advance_26d6;
} FS_ShapedGlyph;

typedef struct FS_ShapedTextRun {
    FS_ShapedGlyph* glyphs;
    uint32_t glyph_count;
} FS_ShapedTextRun;

typedef struct FS_FontVerticalMetrics {
    float ascent;
    float descent;
    float line_height;
} FS_FontVerticalMetrics;

typedef struct FS_FontBackend {
    const char* name;
    void* (*load_font_file)(const char* path);
    void* (*load_font_memory)(const uint8_t* data, size_t size);
    void (*destroy_font)(void* font_handle);
    bool (*get_glyph_sdf)(
        void* font_handle,
        uint32_t codepoint,
        float font_px,
        FS_FontGlyphBitmap* out_glyph
    );
    bool (*get_glyph_sdf_by_index)(
        void* font_handle,
        uint32_t glyph_index,
        float font_px,
        FS_FontGlyphBitmap* out_glyph
    );
    float (*get_kerning_advance)(
        void* font_handle,
        uint32_t left_codepoint,
        uint32_t right_codepoint,
        float font_px
    );
    bool (*get_vertical_metrics)(
        void* font_handle,
        float font_px,
        FS_FontVerticalMetrics* out_metrics
    );
    bool (*shape_text_utf8)(
        void* font_handle,
        const char* utf8,
        float font_px,
        FS_ShapedTextRun* out_run
    );
    void (*free_shaped_text)(FS_ShapedTextRun* run);
    void (*free_glyph_pixels)(uint8_t* pixels);
} FS_FontBackend;

#endif
