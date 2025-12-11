#ifndef WCN_WASM_H
#define WCN_WASM_H

// ============================================================================
// WebAssembly Export Macros for WCN
// ============================================================================
// This file provides macros to export WCN functions for WebAssembly builds.
// Usage: Include this file and use WCN_WASM_EXPORT_ALL() in your implementation.
// ============================================================================

#include "webgpu/webgpu.h"
#include "WCN/WCN_PLATFORM_MACROS.h"
// ============================================================================
// Export Macro Definitions
// ============================================================================

// Context Management
#define WCN_WASM_EXPORT_CONTEXT() \
    WCN_WASM_EXPORT WCN_Context* wcn_create_context(WCN_GPUResources* gpu_resources); \
    WCN_WASM_EXPORT void wcn_destroy_context(WCN_Context* ctx); \
    WCN_WASM_EXPORT void wcn_begin_frame(WCN_Context* ctx, uint32_t width, uint32_t height, WGPUTextureFormat surface_format); \
    WCN_WASM_EXPORT void wcn_end_frame(WCN_Context* ctx); \
    WCN_WASM_EXPORT bool wcn_begin_render_pass(WCN_Context* ctx, int texture_view); \
    WCN_WASM_EXPORT void wcn_end_render_pass(WCN_Context* ctx); \
    WCN_WASM_EXPORT void wcn_submit_commands(WCN_Context* ctx);

// State Management
#define WCN_WASM_EXPORT_STATE() \
    WCN_WASM_EXPORT void wcn_save(WCN_Context* ctx); \
    WCN_WASM_EXPORT void wcn_restore(WCN_Context* ctx);

// Decoder Registration
#define WCN_WASM_EXPORT_DECODERS() \
    WCN_WASM_EXPORT void wcn_register_font_decoder(WCN_Context* ctx, WCN_FontDecoder* decoder); \
    WCN_WASM_EXPORT void wcn_register_image_decoder(WCN_Context* ctx, WCN_ImageDecoder* decoder);

// Rectangle Operations
#define WCN_WASM_EXPORT_RECT() \
    WCN_WASM_EXPORT void wcn_clear_rect(WCN_Context* ctx, float x, float y, float width, float height); \
    WCN_WASM_EXPORT void wcn_fill_rect(WCN_Context* ctx, float x, float y, float width, float height); \
    WCN_WASM_EXPORT void wcn_stroke_rect(WCN_Context* ctx, float x, float y, float width, float height);

// Path Operations
#define WCN_WASM_EXPORT_PATH() \
    WCN_WASM_EXPORT void wcn_begin_path(WCN_Context* ctx); \
    WCN_WASM_EXPORT void wcn_close_path(WCN_Context* ctx); \
    WCN_WASM_EXPORT void wcn_move_to(WCN_Context* ctx, float x, float y); \
    WCN_WASM_EXPORT void wcn_line_to(WCN_Context* ctx, float x, float y); \
    WCN_WASM_EXPORT void wcn_arc(WCN_Context* ctx, float x, float y, float radius, float start_angle, float end_angle, bool anticlockwise); \
    WCN_WASM_EXPORT void wcn_rect(WCN_Context* ctx, float x, float y, float width, float height); \
    WCN_WASM_EXPORT void wcn_fill(WCN_Context* ctx); \
    WCN_WASM_EXPORT void wcn_stroke(WCN_Context* ctx);

// Style and Color
#define WCN_WASM_EXPORT_STYLE() \
    WCN_WASM_EXPORT void wcn_set_fill_style(WCN_Context* ctx, uint32_t color); \
    WCN_WASM_EXPORT void wcn_set_stroke_style(WCN_Context* ctx, uint32_t color); \
    WCN_WASM_EXPORT void wcn_set_line_width(WCN_Context* ctx, float width); \
    WCN_WASM_EXPORT void wcn_set_line_cap(WCN_Context* ctx, WCN_LineCap cap); \
    WCN_WASM_EXPORT void wcn_set_line_join(WCN_Context* ctx, WCN_LineJoin join); \
    WCN_WASM_EXPORT void wcn_set_miter_limit(WCN_Context* ctx, float limit); \
    WCN_WASM_EXPORT void wcn_set_global_alpha(WCN_Context* ctx, float alpha); \
    WCN_WASM_EXPORT void wcn_set_global_composite_operation(WCN_Context* ctx, WCN_CompositeOperation operation);

// Transform Operations
#define WCN_WASM_EXPORT_TRANSFORM() \
    WCN_WASM_EXPORT void wcn_translate(WCN_Context* ctx, float x, float y); \
    WCN_WASM_EXPORT void wcn_rotate(WCN_Context* ctx, float angle); \
    WCN_WASM_EXPORT void wcn_scale(WCN_Context* ctx, float x, float y); \
    WCN_WASM_EXPORT void wcn_transform(WCN_Context* ctx, float a, float b, float c, float d, float e, float f); \
    WCN_WASM_EXPORT void wcn_set_transform(WCN_Context* ctx, float a, float b, float c, float d, float e, float f); \
    WCN_WASM_EXPORT void wcn_reset_transform(WCN_Context* ctx);

// Text Rendering
#define WCN_WASM_EXPORT_TEXT() \
    WCN_WASM_EXPORT void wcn_fill_text(WCN_Context* ctx, const char* text, float x, float y); \
    WCN_WASM_EXPORT void wcn_stroke_text(WCN_Context* ctx, const char* text, float x, float y); \
    WCN_WASM_EXPORT WCN_TextMetrics wcn_measure_text(WCN_Context* ctx, const char* text); \
    WCN_WASM_EXPORT void wcn_set_font(WCN_Context* ctx, const char* font_spec); \
    WCN_WASM_EXPORT void wcn_set_font_face(WCN_Context* ctx, WCN_FontFace* face, float size); \
    WCN_WASM_EXPORT void wcn_set_text_align(WCN_Context* ctx, WCN_TextAlign align); \
    WCN_WASM_EXPORT void wcn_set_text_baseline(WCN_Context* ctx, WCN_TextBaseline baseline); \
    WCN_WASM_EXPORT bool wcn_add_font_fallback(WCN_Context* ctx, WCN_FontFace* face); \
    WCN_WASM_EXPORT void wcn_clear_font_fallbacks(WCN_Context* ctx);

// Image Operations
#define WCN_WASM_EXPORT_IMAGE() \
    WCN_WASM_EXPORT void wcn_draw_image(WCN_Context* ctx, WCN_ImageData* image, float dx, float dy); \
    WCN_WASM_EXPORT void wcn_draw_image_scaled(WCN_Context* ctx, WCN_ImageData* image, float dx, float dy, float dw, float dh); \
    WCN_WASM_EXPORT void wcn_draw_image_source(WCN_Context* ctx, WCN_ImageData* image, float sx, float sy, float sw, float sh, float dx, float dy, float dw, float dh); \
    WCN_WASM_EXPORT WCN_ImageData* wcn_get_image_data(WCN_Context* ctx, float x, float y, float width, float height); \
    WCN_WASM_EXPORT void wcn_put_image_data(WCN_Context* ctx, WCN_ImageData* image_data, float x, float y); \
    WCN_WASM_EXPORT WCN_ImageData* wcn_decode_image(WCN_Context* ctx, const uint8_t* image_bytes, size_t data_size); \
    WCN_WASM_EXPORT void wcn_destroy_image_data(WCN_ImageData* image_data);

typedef WGPUTextureView (*GetWGPUTextureViewCallback)();
// Helper Functions
#define WCN_WASM_EXPORT_HELPERS() \
    WCN_WASM_EXPORT WGPUTextureFormat wcn_get_surface_format(WCN_Context* ctx); \
    WCN_WASM_EXPORT void wcn_set_surface_format(WCN_Context* ctx, WGPUTextureFormat format); \
    WCN_WASM_EXPORT bool wcn_wasm_load_font(const void* font_data, size_t data_size, WCN_FontFace** out_face); \

// ============================================================================
// Export All Functions
// ============================================================================

// Use this macro to export all WCN functions at once
#define WCN_WASM_EXPORT_ALL() \
    WCN_WASM_EXPORT_CONTEXT() \
    WCN_WASM_EXPORT_STATE() \
    WCN_WASM_EXPORT_DECODERS() \
    WCN_WASM_EXPORT_RECT() \
    WCN_WASM_EXPORT_PATH() \
    WCN_WASM_EXPORT_STYLE() \
    WCN_WASM_EXPORT_TRANSFORM() \
    WCN_WASM_EXPORT_TEXT() \
    WCN_WASM_EXPORT_IMAGE() \
    WCN_WASM_EXPORT_HELPERS()

#endif // WCN_WASM_H
