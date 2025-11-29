#include "wcn_internal.h"
#include "WCN/WCN_WASM.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

// ============================================================================
// WCN WASM Font Decoder Implementation
// This is a minimal font decoder implementation for WebAssembly builds
// It uses JavaScript interop to handle font loading and glyph rendering
// ============================================================================

// Font data structure for WASM
typedef struct {
    char* font_name;
    float font_size;
    void* js_font_object;  // JavaScript font object reference
} WCN_WASM_FontData;

// Glyph data structure for WASM
typedef struct {
    uint32_t codepoint;
    float advance_width;
    float left_side_bearing;
    float bounding_box[4];  // [x_min, y_min, x_max, y_max]
    void* js_glyph_object;  // JavaScript glyph object reference
} WCN_WASM_GlyphData;

// ============================================================================
// JavaScript Interop Functions
// ============================================================================

#ifdef __EMSCRIPTEN__

// Load font from JavaScript
EM_JS(bool, js_load_font, (const char* font_name, float font_size, void** out_js_font_object), {
    const fontName = UTF8ToString(font_name);
    try {
        // In a real implementation, this would communicate with JavaScript
        // to load the actual font and return a reference to it
        // console.log('[WCN WASM] Loading font:', fontName, 'size:', font_size);
        
        // For now, we'll just create a mock object
        const fontObj = {
            name: fontName,
            size: font_size,
            loaded: true
        };
        
        // Store reference and return handle
        if (typeof window.WCNJS === 'undefined') {
            window.WCNJS = {};
        }
        if (typeof window.WCNJS.fonts === 'undefined') {
            window.WCNJS.fonts = {};
            window.WCNJS.nextFontId = 1;
        }
        
        const fontId = window.WCNJS.nextFontId++;
        window.WCNJS.fonts[fontId] = fontObj;
        
        // Store the font ID in the out parameter
        setValue(out_js_font_object, fontId, 'i32');
        
        return true;
    } catch (error) {
        // console.error('[WCN WASM] Failed to load font:', error);
        return false;
    }
});

// Get glyph from JavaScript
EM_JS(bool, js_get_glyph, (void* js_font_object, uint32_t codepoint, void** out_js_glyph_object,
                          float* out_advance_width, float* out_left_side_bearing,
                          float* out_bounding_box), {
    const fontId = js_font_object;
    try {
        // Retrieve font object
        const fontObj = window.WCNJS?.fonts?.[fontId];
        if (!fontObj) {
            console.error('[WCN WASM] Font not found for ID:', fontId);
            return false;
        }
        
        // console.log('[WCN WASM] Getting glyph for codepoint:', codepoint, 'in font:', fontObj.name);
        
        // Create mock glyph data
        // In a real implementation, this would get actual glyph metrics from JavaScript
        const glyphObj = {
            codepoint: codepoint,
            fontId: fontId,
            // Mock metrics - in a real implementation these would come from the actual font
            advanceWidth: 12.0,
            leftSideBearing: 1.0,
            boundingBox: [-1.0, -8.0, 11.0, 2.0]
        };
        
        // Store reference and return handle
        if (typeof window.WCNJS.glyphs === 'undefined') {
            window.WCNJS.glyphs = {};
            window.WCNJS.nextGlyphId = 1;
        }
        
        const glyphId = window.WCNJS.nextGlyphId++;
        window.WCNJS.glyphs[glyphId] = glyphObj;
        
        // Store the glyph ID in the out parameter
        setValue(out_js_glyph_object, glyphId, 'i32');
        
        // Set output metrics
        setValue(out_advance_width, glyphObj.advanceWidth, 'float');
        setValue(out_left_side_bearing, glyphObj.leftSideBearing, 'float');
        
        // Set bounding box (4 floats)
        setValue(out_bounding_box, glyphObj.boundingBox[0], 'float');
        setValue(out_bounding_box + 4, glyphObj.boundingBox[1], 'float');
        setValue(out_bounding_box + 8, glyphObj.boundingBox[2], 'float');
        setValue(out_bounding_box + 12, glyphObj.boundingBox[3], 'float');
        
        return true;
    } catch (error) {
        // console.error('[WCN WASM] Failed to get glyph:', error);
        return false;
    }
});

// Get glyph SDF from JavaScript
EM_JS(bool, js_get_glyph_sdf, (void* js_font_object, uint32_t codepoint, float font_size,
                              unsigned char** out_bitmap, int* out_width, int* out_height,
                              float* out_offset_x, float* out_offset_y, float* out_advance), {
    const fontId = js_font_object;
    try {
        // Retrieve font object
        const fontObj = window.WCNJS?.fonts?.[fontId];
        if (!fontObj) {
            console.error('[WCN WASM] Font not found for ID:', fontId);
            return false;
        }
        
        // console.log('[WCN WASM] Getting SDF for glyph:', codepoint, 'in font:', fontObj.name);
        
        // Mock SDF data - in a real implementation, this would generate or retrieve actual SDF
        const width = 32;
        const height = 32;
        
        // Allocate memory for the bitmap (RGBA format: 4 bytes per pixel)
        const bytesPerPixel = 4;
        const bitmapSize = width * height * bytesPerPixel;
        const bitmapPtr = _malloc(bitmapSize);
        if (!bitmapPtr) {
            // console.error('[WCN WASM] Failed to allocate memory for SDF bitmap');
            return false;
        }
        
        // Fill with mock data (in a real implementation, this would be actual SDF values)
        const heapU8 = new Uint8Array(Module.HEAPU8.buffer, bitmapPtr, bitmapSize);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const i = (y * width + x) * bytesPerPixel;
                // Create a simple circle shape for mock SDF
                const centerX = width / 2;
                const centerY = height / 2;
                const distance = Math.sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                const radius = Math.min(width, height) / 3;
                // Convert distance to SDF value (0-255)
                const sdfValue = Math.max(0, Math.min(255, 128 - (distance - radius) * 10));
                
                // RGBA format
                heapU8[i] = sdfValue;     // R
                heapU8[i + 1] = sdfValue; // G
                heapU8[i + 2] = sdfValue; // B
                heapU8[i + 3] = 255;      // A
            }
        }
        
        // Set output values
        setValue(out_bitmap, bitmapPtr, 'i32');
        setValue(out_width, width, 'i32');
        setValue(out_height, height, 'i32');
        setValue(out_offset_x, 0.0, 'float');
        setValue(out_offset_y, 0.0, 'float');
        setValue(out_advance, 12.0, 'float'); // Mock advance width
        
        return true;
    } catch (error) {
        // console.error('[WCN WASM] Failed to get glyph SDF:', error);
        return false;
    }
});

// Free SDF bitmap
EM_JS(void, js_free_glyph_sdf, (unsigned char* bitmap), {
    if (bitmap) {
        _free(bitmap);
    }
});

// Measure text
EM_JS(bool, js_measure_text, (void* js_font_object, const char* text, float font_size,
                             float* out_width, float* out_height), {
    const fontId = js_font_object;
    const textStr = UTF8ToString(text);
    try {
        // Retrieve font object
        const fontObj = window.WCNJS?.fonts?.[fontId];
        if (!fontObj) {
            console.error('[WCN WASM] Font not found for ID:', fontId);
            return false;
        }
        
        // console.log('[WCN WASM] Measuring text:', textStr, 'in font:', fontObj.name);
        
        // Mock measurement - in a real implementation, this would use actual font metrics
        // Simple estimation: average character width * text length
        const avgCharWidth = 12.0;
        const textWidth = textStr.length * avgCharWidth;
        const textHeight = font_size;
        
        setValue(out_width, textWidth, 'float');
        setValue(out_height, textHeight, 'float');
        
        return true;
    } catch (error) {
        // console.error('[WCN WASM] Failed to measure text:', error);
        return false;
    }
});

// Free glyph
EM_JS(void, js_free_glyph, (void* js_glyph_object), {
    const glyphId = js_glyph_object;
    if (window.WCNJS?.glyphs?.[glyphId]) {
        delete window.WCNJS.glyphs[glyphId];
        // console.log('[WCN WASM] Freed glyph:', glyphId);
    }
});

// Free font
EM_JS(void, js_free_font, (void* js_font_object), {
    const fontId = js_font_object;
    if (window.WCNJS?.fonts?.[fontId]) {
        delete window.WCNJS.fonts[fontId];
        // console.log('[WCN WASM] Freed font:', fontId);
    }
});

#endif // __EMSCRIPTEN__

// ============================================================================
// WASM Font Decoder Implementation
// ============================================================================

// Load font
static bool wcn_wasm_load_font(const void* font_data, size_t data_size, WCN_FontFace** out_face) {
    if (!font_data || data_size == 0 || !out_face) {
        return false;
    }
    
    // Extract font name from font data (first null-terminated string)
    const char* font_name = (const char*)font_data;
    
    // printf("[WCN WASM] Loading font: %s\n", font_name);
    
    // Create font face
    WCN_FontFace* face = (WCN_FontFace*)malloc(sizeof(WCN_FontFace));
    if (!face) {
        return false;
    }
    
    // Initialize font face
    memset(face, 0, sizeof(WCN_FontFace));
    
    // Allocate private data
    WCN_WASM_FontData* font_priv = (WCN_WASM_FontData*)malloc(sizeof(WCN_WASM_FontData));
    if (!font_priv) {
        free(face);
        return false;
    }
    
    // Copy font name
    font_priv->font_name = (char*)malloc(strlen(font_name) + 1);
    if (!font_priv->font_name) {
        free(font_priv);
        free(face);
        return false;
    }
    strcpy(font_priv->font_name, font_name);
    
    font_priv->font_size = 16.0f; // Default size
    font_priv->js_font_object = NULL;
    
#ifdef __EMSCRIPTEN__
    // Try to load the font through JavaScript interop
    void* js_font_object = NULL;
    if (!js_load_font(font_name, font_priv->font_size, &js_font_object)) {
        free(font_priv->font_name);
        free(font_priv);
        free(face);
        return false;
    }
    
    font_priv->js_font_object = js_font_object;
#else
    // For native builds, we can't load fonts through JavaScript
    // In a real implementation, you might want to load fonts differently
    font_priv->js_font_object = NULL;
#endif
    
    // Set font face properties
    face->family_name = font_priv->font_name;
    face->ascent = font_priv->font_size * 0.8f;   // Mock ascent
    face->descent = font_priv->font_size * 0.2f;  // Mock descent
    face->line_gap = font_priv->font_size * 0.1f; // Mock line gap
    face->units_per_em = 1000.0f;                 // Standard for most fonts
    face->user_data = font_priv;
    
    *out_face = face;
    return true;
}

// Get glyph
static bool wcn_wasm_get_glyph(WCN_FontFace* face, uint32_t codepoint, WCN_Glyph** out_glyph) {
    if (!face || !out_glyph) {
        return false;
    }
    
    WCN_WASM_FontData* font_data = (WCN_WASM_FontData*)face->user_data;
    if (!font_data) {
        return false;
    }
    
    // printf("[WCN WASM] Getting glyph for codepoint: %u\n", codepoint);
    
    // Create glyph
    WCN_Glyph* glyph = (WCN_Glyph*)malloc(sizeof(WCN_Glyph));
    if (!glyph) {
        return false;
    }
    
    memset(glyph, 0, sizeof(WCN_Glyph));
    
#ifdef __EMSCRIPTEN__
    // Get glyph data through JavaScript interop
    void* js_glyph_object = NULL;
    float advance_width, left_side_bearing;
    float bounding_box[4];
    
    if (!js_get_glyph(font_data->js_font_object, codepoint, &js_glyph_object,
                      &advance_width, &left_side_bearing, bounding_box)) {
        free(glyph);
        return false;
    }
    
    // Fill glyph data
    glyph->codepoint = codepoint;
    glyph->contours = NULL;
    glyph->contour_count = 0;
    glyph->advance_width = advance_width;
    glyph->left_side_bearing = left_side_bearing;
    glyph->bounding_box[0] = bounding_box[0];
    glyph->bounding_box[1] = bounding_box[1];
    glyph->bounding_box[2] = bounding_box[2];
    glyph->bounding_box[3] = bounding_box[3];
    glyph->vertices = NULL;
    glyph->indices = NULL;
    glyph->vertex_count = 0;
    glyph->index_count = 0;
    glyph->raw_vertices = NULL;
    glyph->raw_vertex_count = 0;
    
    // Store JavaScript object reference in a separate structure if needed
    // Since WCN_Glyph doesn't have user_data, we'll manage this separately
    WCN_WASM_GlyphData* glyph_priv = (WCN_WASM_GlyphData*)malloc(sizeof(WCN_WASM_GlyphData));
    if (glyph_priv) {
        glyph_priv->codepoint = codepoint;
        glyph_priv->advance_width = advance_width;
        glyph_priv->left_side_bearing = left_side_bearing;
        glyph_priv->bounding_box[0] = bounding_box[0];
        glyph_priv->bounding_box[1] = bounding_box[1];
        glyph_priv->bounding_box[2] = bounding_box[2];
        glyph_priv->bounding_box[3] = bounding_box[3];
        glyph_priv->js_glyph_object = js_glyph_object;
        
        // We can't store this in glyph->user_data since it doesn't exist
        // In a real implementation, we might need a different approach
        // For now, we'll just free this in the free_glyph function
    }
#else
    // For native builds, create mock glyph data
    glyph->codepoint = codepoint;
    glyph->contours = NULL;
    glyph->contour_count = 0;
    glyph->advance_width = 12.0f;  // Mock advance width
    glyph->left_side_bearing = 1.0f;  // Mock left side bearing
    glyph->bounding_box[0] = -1.0f;  // x_min
    glyph->bounding_box[1] = -8.0f;  // y_min
    glyph->bounding_box[2] = 11.0f;  // x_max
    glyph->bounding_box[3] = 2.0f;   // y_max
    glyph->vertices = NULL;
    glyph->indices = NULL;
    glyph->vertex_count = 0;
    glyph->index_count = 0;
    glyph->raw_vertices = NULL;
    glyph->raw_vertex_count = 0;
#endif
    
    *out_glyph = glyph;
    return true;
}

// Get glyph SDF
static bool wcn_wasm_get_glyph_sdf(WCN_FontFace* face, uint32_t codepoint, float font_size,
                                  unsigned char** out_bitmap,
                                  int* out_width, int* out_height,
                                  float* out_offset_x, float* out_offset_y,
                                  float* out_advance) {
    if (!face || !out_bitmap || !out_width || !out_height) {
        return false;
    }
    
    WCN_WASM_FontData* font_data = (WCN_WASM_FontData*)face->user_data;
    if (!font_data) {
        return false;
    }
    
    // printf("[WCN WASM] Getting SDF for glyph: %u\n", codepoint);
    
#ifdef __EMSCRIPTEN__
    // Get SDF through JavaScript interop
    return js_get_glyph_sdf(font_data->js_font_object, codepoint, font_size,
                           out_bitmap, out_width, out_height,
                           out_offset_x, out_offset_y, out_advance);
#else
    // For native builds, create mock SDF data
    const int width = 32;
    const int height = 32;
    
    // Allocate memory for the bitmap
    const int bitmapSize = width * height;
    unsigned char* bitmap = (unsigned char*)malloc(bitmapSize);
    if (!bitmap) {
        return false;
    }
    
    // Fill with mock data
    for (int i = 0; i < bitmapSize; i++) {
        bitmap[i] = (unsigned char)((i * 255) / bitmapSize);
    }
    
    // Set output values
    *out_bitmap = bitmap;
    *out_width = width;
    *out_height = height;
    *out_offset_x = 0.0f;
    *out_offset_y = 0.0f;
    *out_advance = 12.0f; // Mock advance width
    
    return true;
#endif
}

// Free SDF bitmap
static void wcn_wasm_free_glyph_sdf(unsigned char* bitmap) {
    if (bitmap) {
#ifdef __EMSCRIPTEN__
        js_free_glyph_sdf(bitmap);
#else
        free(bitmap);
#endif
    }
}

// Measure text
static bool wcn_wasm_measure_text(WCN_FontFace* face, const char* text, float font_size,
                                 float* out_width, float* out_height) {
    if (!face || !text || !out_width) {
        return false;
    }
    
    WCN_WASM_FontData* font_data = (WCN_WASM_FontData*)face->user_data;
    if (!font_data) {
        return false;
    }
    
    // printf("[WCN WASAM] Measuring text: %s\n", text);
    
#ifdef __EMSCRIPTEN__
    // Measure through JavaScript interop
    return js_measure_text(font_data->js_font_object, text, font_size, out_width, out_height);
#else
    // For native builds, create mock measurement
    // Simple estimation: average character width * text length
    const float avgCharWidth = 12.0f;
    const float textWidth = (float)strlen(text) * avgCharWidth;
    const float textHeight = font_size;
    
    *out_width = textWidth;
    if (out_height) {
        *out_height = textHeight;
    }
    
    return true;
#endif
}

// Free glyph
static void wcn_wasm_free_glyph(WCN_Glyph* glyph) {
    if (glyph) {
        // Note: We can't access glyph->user_data because it doesn't exist in the structure
        // In a real implementation, we would need a different way to track private data
        
        // Free contours, vertices, indices, etc.
        if (glyph->contours) {
            for (size_t i = 0; i < glyph->contour_count; i++) {
                free(glyph->contours[i].points);
            }
            free(glyph->contours);
        }
        
        free(glyph->vertices);
        free(glyph->indices);
        free(glyph->raw_vertices);
        free(glyph);
    }
}

// Free font
static void wcn_wasm_free_font(WCN_FontFace* face) {
    if (face) {
        WCN_WASM_FontData* font_data = (WCN_WASM_FontData*)face->user_data;
        if (font_data) {
#ifdef __EMSCRIPTEN__
            if (font_data->js_font_object) {
                js_free_font(font_data->js_font_object);
            }
#endif
            free(font_data->font_name);
            free(font_data);
        }
        free(face);
    }
}

// ============================================================================
// Global WASM Font Decoder Instance
// ============================================================================

// Global decoder instance
static WCN_FontDecoder wcn_wasm_font_decoder = {
    .load_font = wcn_wasm_load_font,
    .get_glyph = wcn_wasm_get_glyph,
    .get_glyph_sdf = wcn_wasm_get_glyph_sdf,
    .free_glyph_sdf = wcn_wasm_free_glyph_sdf,
    .measure_text = wcn_wasm_measure_text,
    .free_glyph = wcn_wasm_free_glyph,
    .free_font = wcn_wasm_free_font,
    .name = "wasm_font_decoder"
};

// Get decoder instance
WCN_FontDecoder* wcn_get_wasm_font_decoder(void) {
    return &wcn_wasm_font_decoder;
}

// Export function for WASM
#ifdef __EMSCRIPTEN__
WCN_WASM_EXPORT WCN_FontDecoder* wcn_wasm_get_font_decoder(void) {
    return wcn_get_wasm_font_decoder();
}

// Create a default font face
WCN_WASM_EXPORT WCN_FontFace* wcn_wasm_create_default_font_face(void) {
    // Create font face
    WCN_FontFace* face = (WCN_FontFace*)malloc(sizeof(WCN_FontFace));
    if (!face) {
        return NULL;
    }
    
    // Initialize font face
    memset(face, 0, sizeof(WCN_FontFace));
    
    // Allocate private data
    WCN_WASM_FontData* font_priv = (WCN_WASM_FontData*)malloc(sizeof(WCN_WASM_FontData));
    if (!font_priv) {
        free(face);
        return NULL;
    }
    
    // Set default font name
    const char* default_font_name = "Arial";
    font_priv->font_name = (char*)malloc(strlen(default_font_name) + 1);
    if (!font_priv->font_name) {
        free(font_priv);
        free(face);
        return NULL;
    }
    strcpy(font_priv->font_name, default_font_name);
    
    font_priv->font_size = 16.0f; // Default size
    font_priv->js_font_object = NULL;
    
#ifdef __EMSCRIPTEN__
    // Try to load the font through JavaScript interop
    void* js_font_object = NULL;
    if (js_load_font(default_font_name, font_priv->font_size, &js_font_object)) {
        font_priv->js_font_object = js_font_object;
    }
#endif
    
    // Set font face properties
    face->family_name = font_priv->font_name;
    face->ascent = font_priv->font_size * 0.8f;   // Mock ascent
    face->descent = font_priv->font_size * 0.2f;  // Mock descent
    face->line_gap = font_priv->font_size * 0.1f; // Mock line gap
    face->units_per_em = 1000.0f;                 // Standard for most fonts
    face->user_data = font_priv;
    
    return face;
}

#endif
