#ifndef WCN_FULLSTACK_CORE_H
#define WCN_FULLSTACK_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <webgpu/wgpu.h>
#include "fullstack_backend_api.h"

typedef struct FS_Core FS_Core;
typedef struct FS_Path2D FS_Path2D;
typedef struct FS_LinearGradient FS_LinearGradient;
typedef struct FS_RadialGradient FS_RadialGradient;
typedef struct FS_ConicGradient FS_ConicGradient;
typedef struct FS_Pattern FS_Pattern;

typedef struct FS_ImageHandle {
    float uv_min[2];
    float uv_max[2];
    uint32_t width;
    uint32_t height;
    uint32_t layer;
    uint32_t generation;
    uint32_t atlas_x;
    uint32_t atlas_y;
} FS_ImageHandle;

typedef struct FS_ImageFontSequence {
    const char* utf8;
    uint32_t glyph_id;
} FS_ImageFontSequence;

typedef enum FS_LineCap {
    FS_LINE_CAP_BUTT = 0,
    FS_LINE_CAP_ROUND = 1,
    FS_LINE_CAP_SQUARE = 2
} FS_LineCap;

typedef enum FS_LineJoin {
    FS_LINE_JOIN_MITER = 0,
    FS_LINE_JOIN_ROUND = 1,
    FS_LINE_JOIN_BEVEL = 2
} FS_LineJoin;

typedef enum FS_FillRule {
    FS_FILL_RULE_NONZERO = 0,
    FS_FILL_RULE_EVENODD = 1
} FS_FillRule;

typedef enum FS_GlobalCompositeOperation {
    FS_GLOBAL_COMPOSITE_SOURCE_OVER = 0,
    FS_GLOBAL_COMPOSITE_COPY = 1,
    FS_GLOBAL_COMPOSITE_LIGHTER = 2,
    FS_GLOBAL_COMPOSITE_DESTINATION_OVER = 3,
    FS_GLOBAL_COMPOSITE_SOURCE_IN = 4,
    FS_GLOBAL_COMPOSITE_SOURCE_OUT = 5,
    FS_GLOBAL_COMPOSITE_DESTINATION_IN = 6,
    FS_GLOBAL_COMPOSITE_DESTINATION_OUT = 7,
    FS_GLOBAL_COMPOSITE_XOR = 8,
    FS_GLOBAL_COMPOSITE_SOURCE_ATOP = 9,
    FS_GLOBAL_COMPOSITE_DESTINATION_ATOP = 10
} FS_GlobalCompositeOperation;

typedef enum FS_TextAlign {
    FS_TEXT_ALIGN_START = 0,
    FS_TEXT_ALIGN_LEFT = 1,
    FS_TEXT_ALIGN_CENTER = 2,
    FS_TEXT_ALIGN_RIGHT = 3,
    FS_TEXT_ALIGN_END = 4
} FS_TextAlign;

typedef enum FS_TextBaseline {
    FS_TEXT_BASELINE_TOP = 0,
    FS_TEXT_BASELINE_HANGING = 1,
    FS_TEXT_BASELINE_MIDDLE = 2,
    FS_TEXT_BASELINE_ALPHABETIC = 3,
    FS_TEXT_BASELINE_IDEOGRAPHIC = 4,
    FS_TEXT_BASELINE_BOTTOM = 5
} FS_TextBaseline;

typedef enum FS_TextDirection {
    FS_TEXT_DIRECTION_INHERIT = 0,
    FS_TEXT_DIRECTION_LTR = 1,
    FS_TEXT_DIRECTION_RTL = 2
} FS_TextDirection;

typedef enum FS_FontKerning {
    FS_FONT_KERNING_AUTO = 0,
    FS_FONT_KERNING_NORMAL = 1,
    FS_FONT_KERNING_NONE = 2
} FS_FontKerning;

typedef enum FS_TextRendering {
    FS_TEXT_RENDERING_AUTO = 0,
    FS_TEXT_RENDERING_OPTIMIZE_SPEED = 1,
    FS_TEXT_RENDERING_OPTIMIZE_LEGIBILITY = 2,
    FS_TEXT_RENDERING_GEOMETRIC_PRECISION = 3
} FS_TextRendering;

typedef enum FS_FontStretch {
    FS_FONT_STRETCH_ULTRA_CONDENSED = 0,
    FS_FONT_STRETCH_EXTRA_CONDENSED = 1,
    FS_FONT_STRETCH_CONDENSED = 2,
    FS_FONT_STRETCH_SEMI_CONDENSED = 3,
    FS_FONT_STRETCH_NORMAL = 4,
    FS_FONT_STRETCH_SEMI_EXPANDED = 5,
    FS_FONT_STRETCH_EXPANDED = 6,
    FS_FONT_STRETCH_EXTRA_EXPANDED = 7,
    FS_FONT_STRETCH_ULTRA_EXPANDED = 8
} FS_FontStretch;

typedef enum FS_FontVariantCaps {
    FS_FONT_VARIANT_CAPS_NORMAL = 0,
    FS_FONT_VARIANT_CAPS_SMALL_CAPS = 1,
    FS_FONT_VARIANT_CAPS_ALL_SMALL_CAPS = 2,
    FS_FONT_VARIANT_CAPS_PETITE_CAPS = 3,
    FS_FONT_VARIANT_CAPS_ALL_PETITE_CAPS = 4,
    FS_FONT_VARIANT_CAPS_UNICASE = 5,
    FS_FONT_VARIANT_CAPS_TITLING_CAPS = 6
} FS_FontVariantCaps;

typedef enum FS_PatternRepeat {
    FS_PATTERN_REPEAT = 0,
    FS_PATTERN_REPEAT_X = 1,
    FS_PATTERN_REPEAT_Y = 2,
    FS_PATTERN_NO_REPEAT = 3
} FS_PatternRepeat;

typedef enum FS_ImageSmoothingQuality {
    FS_IMAGE_SMOOTHING_QUALITY_LOW = 0,
    FS_IMAGE_SMOOTHING_QUALITY_MEDIUM = 1,
    FS_IMAGE_SMOOTHING_QUALITY_HIGH = 2
} FS_ImageSmoothingQuality;

typedef struct FS_MissingImageGlyph {
    uint32_t image_font_id;
    uint32_t glyph_id;
    char sequence_utf8[64];
} FS_MissingImageGlyph;

typedef struct FS_TextMetrics {
    float width;
    float actual_bounding_box_left;
    float actual_bounding_box_right;
    float actual_bounding_box_ascent;
    float actual_bounding_box_descent;
    float em_height_ascent;
    float em_height_descent;
    uint32_t glyph_count;
    uint32_t line_count;
} FS_TextMetrics;

typedef struct FS_ContextAttributes {
    bool alpha;
    bool premultiplied_alpha;
    bool antialias;
    bool depth;
    bool stencil;
    bool preserve_drawing_buffer;
} FS_ContextAttributes;

FS_LinearGradient* fs_linear_gradient_create(float x0, float y0, float x1, float y1);
void fs_linear_gradient_destroy(FS_LinearGradient* gradient);
bool fs_linear_gradient_add_color_stop(FS_LinearGradient* gradient, float offset_0_to_1, uint32_t color_rgba8);
FS_RadialGradient* fs_radial_gradient_create(float x0, float y0, float r0, float x1, float y1, float r1);
void fs_radial_gradient_destroy(FS_RadialGradient* gradient);
bool fs_radial_gradient_add_color_stop(FS_RadialGradient* gradient, float offset_0_to_1, uint32_t color_rgba8);
FS_ConicGradient* fs_conic_gradient_create(float start_angle_radians, float cx, float cy);
void fs_conic_gradient_destroy(FS_ConicGradient* gradient);
bool fs_conic_gradient_add_color_stop(FS_ConicGradient* gradient, float offset_0_to_1, uint32_t color_rgba8);
FS_Pattern* fs_pattern_create_image(const FS_ImageHandle* handle, FS_PatternRepeat repeat_mode);
void fs_pattern_destroy(FS_Pattern* pattern);
bool fs_pattern_set_transform(FS_Pattern* pattern, float a, float b, float c, float d, float e, float f);

FS_Core* fs_core_create(
    WGPUDevice device,
    WGPUQueue queue,
    WGPUTextureFormat target_format,
    uint32_t width,
    uint32_t height
);
void fs_core_destroy(FS_Core* core);

bool fs_core_init(
    FS_Core* core,
    WGPUDevice device,
    WGPUQueue queue,
    WGPUTextureFormat target_format,
    uint32_t width,
    uint32_t height
);

void fs_core_shutdown(FS_Core* core);
void fs_core_resize(FS_Core* core, uint32_t width, uint32_t height);
void fs_core_begin_commands(FS_Core* core);
void fs_context_reset(FS_Core* core);
bool fs_core_is_context_lost(const FS_Core* core);
bool fs_core_get_context_attributes(const FS_Core* core, FS_ContextAttributes* out_attributes);

bool fs_core_encode(
    FS_Core* core,
    WGPUCommandEncoder encoder,
    WGPUTexture target_texture,
    WGPUTextureView target_view,
    float clear_r,
    float clear_g,
    float clear_b,
    float clear_a
);
void fs_core_notify_submission(FS_Core* core, WGPUSubmissionIndex submission_index);

bool fs_cmd_rect(FS_Core* core, float x, float y, float w, float h, float radius, uint32_t color);
bool fs_cmd_clear_rect(FS_Core* core, float x, float y, float w, float h);
bool fs_cmd_image(FS_Core* core, float x, float y, float w, float h, float uv_x, float uv_y, float uv_w, float uv_h, uint32_t color);
bool fs_cmd_text_glyph(FS_Core* core, float x, float y, float w, float h, uint32_t codepoint, uint32_t color);
bool fs_cmd_line(FS_Core* core, float x0, float y0, float x1, float y1, float width, uint32_t color);
bool fs_cmd_path_segment(FS_Core* core, float x0, float y0, float x1, float y1, float width, uint32_t color);
bool fs_cmd_circle(FS_Core* core, float cx, float cy, float radius, uint32_t color);
bool fs_cmd_arc(FS_Core* core, float cx, float cy, float radius, float thickness, float start_angle, float end_angle, uint32_t color);
bool fs_cmd_bezier_quad(FS_Core* core, float x0, float y0, float cx, float cy, float x1, float y1, float width, uint32_t color);
bool fs_cmd_rect_stroke(FS_Core* core, float x, float y, float w, float h, float radius, float stroke_width, uint32_t color);
bool fs_cmd_ellipse(FS_Core* core, float cx, float cy, float radius_x, float radius_y, uint32_t color);
bool fs_cmd_bezier_cubic(FS_Core* core, float x0, float y0, float cx0, float cy0, float cx1, float cy1, float x1, float y1, float width, uint32_t color);
bool fs_cmd_triangle(FS_Core* core, float x0, float y0, float x1, float y1, float x2, float y2, uint32_t color);
bool fs_cmd_image_handle(FS_Core* core, float x, float y, float w, float h, const FS_ImageHandle* handle, uint32_t color);
bool fs_cmd_text_utf8(FS_Core* core, float x, float baseline_y, float font_size_px, const char* utf8, uint32_t color, float max_width);
bool fs_cmd_stroke_text_utf8(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    uint32_t color,
    float max_width,
    float stroke_width
);
bool fs_measure_text_utf8(
    FS_Core* core,
    float font_size_px,
    const char* utf8,
    float max_width,
    FS_TextMetrics* out_metrics
);

void fs_state_save(FS_Core* core);
bool fs_state_restore(FS_Core* core);
void fs_transform_reset(FS_Core* core);
bool fs_translate(FS_Core* core, float tx, float ty);
bool fs_rotate(FS_Core* core, float radians);
bool fs_scale(FS_Core* core, float sx, float sy);
bool fs_transform(FS_Core* core, float a, float b, float c, float d, float e, float f);
bool fs_set_transform(FS_Core* core, float a, float b, float c, float d, float e, float f);
bool fs_get_transform(const FS_Core* core, float out_matrix_2x3[6]);
bool fs_clip_rect(FS_Core* core, float x, float y, float w, float h);
bool fs_clip_path(FS_Core* core);
bool fs_clip_path_with_fill_rule(FS_Core* core, FS_FillRule fill_rule);

void fs_style_reset(FS_Core* core);
bool fs_style_set_line_width(FS_Core* core, float width);
bool fs_style_set_line_cap(FS_Core* core, FS_LineCap cap);
bool fs_style_set_line_join(FS_Core* core, FS_LineJoin join);
bool fs_style_set_fill_rule(FS_Core* core, FS_FillRule fill_rule);
bool fs_style_set_miter_limit(FS_Core* core, float limit);
float fs_style_get_miter_limit(const FS_Core* core);
bool fs_style_set_global_alpha(FS_Core* core, float alpha);
float fs_style_get_global_alpha(const FS_Core* core);
bool fs_style_set_global_composite_operation(FS_Core* core, FS_GlobalCompositeOperation op);
FS_GlobalCompositeOperation fs_style_get_global_composite_operation(const FS_Core* core);
bool fs_style_set_shadow_color(FS_Core* core, uint32_t color_rgba8);
uint32_t fs_style_get_shadow_color(const FS_Core* core);
bool fs_style_set_fill_color(FS_Core* core, uint32_t color_rgba8);
uint32_t fs_style_get_fill_color(const FS_Core* core);
bool fs_style_set_stroke_color(FS_Core* core, uint32_t color_rgba8);
uint32_t fs_style_get_stroke_color(const FS_Core* core);
bool fs_style_set_fill_linear_gradient(FS_Core* core, const FS_LinearGradient* gradient);
bool fs_style_set_stroke_linear_gradient(FS_Core* core, const FS_LinearGradient* gradient);
bool fs_style_set_fill_radial_gradient(FS_Core* core, const FS_RadialGradient* gradient);
bool fs_style_set_stroke_radial_gradient(FS_Core* core, const FS_RadialGradient* gradient);
bool fs_style_set_fill_conic_gradient(FS_Core* core, const FS_ConicGradient* gradient);
bool fs_style_set_stroke_conic_gradient(FS_Core* core, const FS_ConicGradient* gradient);
bool fs_style_set_fill_pattern(FS_Core* core, const FS_Pattern* pattern);
bool fs_style_set_stroke_pattern(FS_Core* core, const FS_Pattern* pattern);
bool fs_style_set_shadow_blur(FS_Core* core, float blur_px);
float fs_style_get_shadow_blur(const FS_Core* core);
bool fs_style_set_shadow_offset(FS_Core* core, float offset_x, float offset_y);
float fs_style_get_shadow_offset_x(const FS_Core* core);
float fs_style_get_shadow_offset_y(const FS_Core* core);
bool fs_style_set_text_align(FS_Core* core, FS_TextAlign align);
FS_TextAlign fs_style_get_text_align(const FS_Core* core);
bool fs_style_set_text_baseline(FS_Core* core, FS_TextBaseline baseline);
FS_TextBaseline fs_style_get_text_baseline(const FS_Core* core);
bool fs_style_set_text_direction(FS_Core* core, FS_TextDirection direction);
FS_TextDirection fs_style_get_text_direction(const FS_Core* core);
bool fs_style_set_font_kerning(FS_Core* core, FS_FontKerning kerning);
FS_FontKerning fs_style_get_font_kerning(const FS_Core* core);
bool fs_style_set_text_rendering(FS_Core* core, FS_TextRendering rendering);
FS_TextRendering fs_style_get_text_rendering(const FS_Core* core);
bool fs_style_set_font_stretch(FS_Core* core, FS_FontStretch stretch);
FS_FontStretch fs_style_get_font_stretch(const FS_Core* core);
bool fs_style_set_font_variant_caps(FS_Core* core, FS_FontVariantCaps variant_caps);
FS_FontVariantCaps fs_style_get_font_variant_caps(const FS_Core* core);
bool fs_style_set_letter_spacing(FS_Core* core, float spacing_px);
float fs_style_get_letter_spacing(const FS_Core* core);
bool fs_style_set_word_spacing(FS_Core* core, float spacing_px);
float fs_style_get_word_spacing(const FS_Core* core);
bool fs_style_set_image_smoothing_enabled(FS_Core* core, bool enabled);
bool fs_style_get_image_smoothing_enabled(const FS_Core* core);
bool fs_style_set_image_smoothing_quality(FS_Core* core, FS_ImageSmoothingQuality quality);
FS_ImageSmoothingQuality fs_style_get_image_smoothing_quality(const FS_Core* core);
bool fs_style_set_dash(FS_Core* core, const float* segments, uint32_t segment_count, float offset);
bool fs_style_get_dash(
    const FS_Core* core,
    float* out_segments,
    uint32_t max_segments,
    uint32_t* out_count,
    float* out_offset
);
void fs_style_clear_dash(FS_Core* core);

void fs_path_begin(FS_Core* core);
bool fs_path_move_to(FS_Core* core, float x, float y);
bool fs_path_line_to(FS_Core* core, float x, float y);
bool fs_path_quadratic_curve_to(FS_Core* core, float cx, float cy, float x, float y);
bool fs_path_bezier_curve_to(FS_Core* core, float cx0, float cy0, float cx1, float cy1, float x, float y);
bool fs_path_arc(FS_Core* core, float cx, float cy, float radius, float start_angle, float end_angle, bool counterclockwise);
bool fs_path_ellipse(
    FS_Core* core,
    float cx,
    float cy,
    float radius_x,
    float radius_y,
    float rotation,
    float start_angle,
    float end_angle,
    bool counterclockwise
);
bool fs_path_arc_to(FS_Core* core, float x1, float y1, float x2, float y2, float radius);
bool fs_path_rect(FS_Core* core, float x, float y, float w, float h);
bool fs_path_round_rect(FS_Core* core, float x, float y, float w, float h, float radius);
bool fs_path_close(FS_Core* core);
bool fs_path_stroke(FS_Core* core, float width, uint32_t color);
bool fs_path_fill(FS_Core* core, uint32_t color);
bool fs_stroke(FS_Core* core, float width);
bool fs_fill(FS_Core* core);
bool fs_stroke_rect(FS_Core* core, float x, float y, float w, float h, float radius, float stroke_width);
bool fs_fill_rect(FS_Core* core, float x, float y, float w, float h, float radius);
bool fs_stroke_text_utf8(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    float max_width,
    float stroke_width
);
bool fs_fill_text_utf8(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    float max_width
);
bool fs_is_point_in_path(FS_Core* core, float x, float y);
bool fs_is_point_in_path_with_fill_rule(FS_Core* core, float x, float y, FS_FillRule fill_rule);
bool fs_is_point_in_stroke(FS_Core* core, float x, float y);

FS_Path2D* fs_path2d_create(void);
void fs_path2d_destroy(FS_Path2D* path);
void fs_path2d_reset(FS_Path2D* path);
bool fs_path2d_move_to(FS_Path2D* path, float x, float y);
bool fs_path2d_line_to(FS_Path2D* path, float x, float y);
bool fs_path2d_quadratic_curve_to(FS_Path2D* path, float cx, float cy, float x, float y);
bool fs_path2d_bezier_curve_to(FS_Path2D* path, float cx0, float cy0, float cx1, float cy1, float x, float y);
bool fs_path2d_arc(FS_Path2D* path, float cx, float cy, float radius, float start_angle, float end_angle, bool counterclockwise);
bool fs_path2d_ellipse(
    FS_Path2D* path,
    float cx,
    float cy,
    float radius_x,
    float radius_y,
    float rotation,
    float start_angle,
    float end_angle,
    bool counterclockwise
);
bool fs_path2d_arc_to(FS_Path2D* path, float x1, float y1, float x2, float y2, float radius);
bool fs_path2d_rect(FS_Path2D* path, float x, float y, float w, float h);
bool fs_path2d_round_rect(FS_Path2D* path, float x, float y, float w, float h, float radius);
bool fs_path2d_close(FS_Path2D* path);
bool fs_path2d_add_path(FS_Path2D* path, const FS_Path2D* other);
bool fs_path2d_add_path_with_transform(FS_Path2D* path, const FS_Path2D* other, const float matrix_2x3[6]);
bool fs_clip_path2d(FS_Core* core, const FS_Path2D* path);
bool fs_clip_path2d_with_fill_rule(FS_Core* core, const FS_Path2D* path, FS_FillRule fill_rule);
bool fs_path_fill_path2d(FS_Core* core, const FS_Path2D* path, uint32_t color);
bool fs_path_stroke_path2d(FS_Core* core, const FS_Path2D* path, float width, uint32_t color);
bool fs_is_point_in_path2d(FS_Core* core, const FS_Path2D* path, float x, float y);
bool fs_is_point_in_path2d_with_fill_rule(FS_Core* core, const FS_Path2D* path, float x, float y, FS_FillRule fill_rule);
bool fs_is_point_in_stroke_path2d(FS_Core* core, const FS_Path2D* path, float x, float y);

bool fs_core_upload_image_rgba8(
    FS_Core* core,
    const uint8_t* rgba_pixels,
    uint32_t width,
    uint32_t height,
    FS_ImageHandle* out_handle
);

bool fs_core_decode_image_memory(
    FS_Core* core,
    const uint8_t* encoded_bytes,
    size_t encoded_size,
    FS_ImageHandle* out_handle
);

bool fs_core_decode_image_file(
    FS_Core* core,
    const char* path,
    FS_ImageHandle* out_handle
);

bool fs_core_put_image_data_rgba8(
    FS_Core* core,
    const FS_ImageHandle* handle,
    const uint8_t* rgba_pixels,
    size_t rgba_size
);

bool fs_core_get_image_data_rgba8(
    const FS_Core* core,
    const FS_ImageHandle* handle,
    uint8_t* out_rgba_pixels,
    size_t out_rgba_size
);

bool fs_core_create_image_data_rgba8(
    uint32_t width,
    uint32_t height,
    uint8_t* out_rgba_pixels,
    size_t out_rgba_size
);

bool fs_core_put_canvas_image_data_rgba8(
    FS_Core* core,
    int32_t dst_x,
    int32_t dst_y,
    uint32_t width,
    uint32_t height,
    const uint8_t* rgba_pixels,
    size_t rgba_size
);

bool fs_core_get_canvas_image_data_rgba8(
    const FS_Core* core,
    int32_t src_x,
    int32_t src_y,
    uint32_t width,
    uint32_t height,
    uint8_t* out_rgba_pixels,
    size_t out_rgba_size
);

bool fs_core_load_font_file(FS_Core* core, const char* path);
bool fs_core_register_image_font(
    FS_Core* core,
    const FS_ImageFontSequence* sequences,
    uint32_t sequence_count,
    uint32_t* out_font_id
);
bool fs_core_load_image_glyph_rgba8(
    FS_Core* core,
    uint32_t image_font_id,
    uint32_t glyph_id,
    const uint8_t* rgba_pixels,
    uint32_t width,
    uint32_t height
);
bool fs_core_load_image_glyph_png_memory(
    FS_Core* core,
    uint32_t image_font_id,
    uint32_t glyph_id,
    const uint8_t* encoded_bytes,
    size_t encoded_size
);
uint32_t fs_core_get_missing_image_glyph_count(const FS_Core* core);
bool fs_core_get_missing_image_glyph(
    const FS_Core* core,
    uint32_t index,
    FS_MissingImageGlyph* out_missing
);
void fs_core_clear_missing_image_glyphs(FS_Core* core);

bool fs_core_set_image_backend(FS_Core* core, const FS_ImageBackend* backend);
bool fs_core_set_font_backend(FS_Core* core, const FS_FontBackend* backend);
const char* fs_core_get_image_backend_name(const FS_Core* core);
const char* fs_core_get_font_backend_name(const FS_Core* core);

#endif
