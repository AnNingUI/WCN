#ifndef WCN_FULLSTACK_CORE_PRIVATE_H
#define WCN_FULLSTACK_CORE_PRIVATE_H

#include "fullstack_core.h"
#include "fullstack_backend_api.h"
#include "fullstack_core_gpu_layout.h"

#define FS_IMAGE_ATLAS_SIZE 2048u
#define FS_IMAGE_ATLAS_MAX_LAYERS 4u
#define FS_GLYPH_ATLAS_SIZE 2048u
#define FS_IMAGE_ATLAS_PADDING 1u
#define FS_GLYPH_ATLAS_PADDING 4u
#define FS_ATLAS_PADDING 1u
#define FS_TEXT_BAKE_MIN_PX 24.0f
#define FS_TEXT_BAKE_SCALE 1.5f
#define FS_ENABLE_EXPERIMENTAL_SHAPING 0
#define FS_RENDER_PREMULTIPLIED_ALPHA 1
#define FS_IMAGE_SEQ_MAX_CP 32u
#define FS_EMOJI_SCALE_BIAS 1.11f
#define FS_EMOJI_ADVANCE_BIAS 0.125f
#define FS_EMOJI_ADVANCE_ALIGN 0.25f
#define FS_EMOJI_ASCENT 0.84f
#define FS_IMAGE_FONT_KIND 2u
#define FS_IMAGE_KEY_FONT_BITS 10u
#define FS_IMAGE_KEY_GLYPH_BITS (32u - FS_IMAGE_KEY_FONT_BITS)
#define FS_IMAGE_KEY_GLYPH_MASK ((1u << FS_IMAGE_KEY_GLYPH_BITS) - 1u)
#define FS_MAX_FONT_FALLBACKS 8u
#define FS_RENDER_MSAA_SAMPLES 4u
#define FS_SHADOW_BLUR_MAX 15.0f
#define FS_SHADOW_SEPARABLE_THRESHOLD 5.0f
#define FS_SHADOW_SEPARABLE_STEP_SCALE 0.55f
#define FS_LINEAR_GRADIENT_MAX_STOPS 16u
#define FS_RADIAL_GRADIENT_MAX_STOPS 16u
#define FS_CONIC_GRADIENT_MAX_STOPS 16u

typedef struct FS_GlyphEntry {
    uint32_t glyph_key;
    uint32_t key_kind;
    uint32_t bake_px_q;
    uint32_t text_flags;
    uint8_t font_slot;
    uint8_t _pad0;
    uint8_t _pad1;
    uint8_t _pad2;
    float uv_min[2];
    float uv_max[2];
    float atlas_width;
    float atlas_height;
    float bearing_x;
    float bearing_y;
    float advance;
    float sdf_radius_px;
    float sdf_onedge;
    float sdf_pixel_dist_scale;
    float bake_px;
} FS_GlyphEntry;

typedef struct FS_ImageGlyphSlot {
    bool loaded;
    uint8_t* rgba;
    uint32_t width;
    uint32_t height;
} FS_ImageGlyphSlot;

typedef struct FS_ImageSequenceEntry {
    uint32_t glyph_id;
    uint8_t cp_count;
    uint32_t cps[FS_IMAGE_SEQ_MAX_CP];
    char* utf8;
} FS_ImageSequenceEntry;

typedef struct FS_ImageFontState {
    uint32_t id;
    FS_ImageSequenceEntry* sequences;
    uint32_t sequence_count;
    FS_ImageGlyphSlot* glyph_slots;
    uint32_t glyph_slot_count;
} FS_ImageFontState;

typedef enum FS_PathSegType {
    FS_PATH_SEG_LINE = 0,
    FS_PATH_SEG_QUAD = 1,
    FS_PATH_SEG_CUBIC = 2
} FS_PathSegType;

typedef struct FS_PathSegment {
    uint8_t type;
    uint8_t _pad0;
    uint8_t _pad1;
    uint8_t _pad2;
    float x0;
    float y0;
    float cx0;
    float cy0;
    float cx1;
    float cy1;
    float x1;
    float y1;
} FS_PathSegment;

struct FS_Path2D {
    FS_PathSegment* segments;
    uint32_t count;
    uint32_t capacity;
    bool has_current;
    bool has_subpath_start;
    float current_x;
    float current_y;
    float subpath_start_x;
    float subpath_start_y;
};

typedef struct FS_PathStateBorrow {
    FS_PathSegment* segments;
    uint32_t count;
    uint32_t capacity;
    bool has_current;
    bool has_subpath_start;
    float current_x;
    float current_y;
    float subpath_start_x;
    float subpath_start_y;
} FS_PathStateBorrow;

typedef struct FS_GradientStop {
    float offset_0_to_1;
    uint32_t color_rgba8;
} FS_GradientStop;

struct FS_LinearGradient {
    float x0;
    float y0;
    float x1;
    float y1;
    FS_GradientStop* stops;
    uint32_t stop_count;
    uint32_t stop_capacity;
};

struct FS_RadialGradient {
    float x0;
    float y0;
    float r0;
    float x1;
    float y1;
    float r1;
    FS_GradientStop* stops;
    uint32_t stop_count;
    uint32_t stop_capacity;
};

struct FS_ConicGradient {
    float start_angle_radians;
    float cx;
    float cy;
    FS_GradientStop* stops;
    uint32_t stop_count;
    uint32_t stop_capacity;
};

struct FS_Pattern {
    FS_ImageHandle handle;
    float xform[6];
    float inv_xform[6];
    uint8_t repeat_mode;
    uint8_t inv_valid;
    uint8_t _pad0;
    uint8_t _pad1;
};

typedef struct FS_StyleLinearGradient {
    float x0;
    float y0;
    float x1;
    float y1;
    FS_GradientStop stops[FS_LINEAR_GRADIENT_MAX_STOPS];
    uint32_t stop_count;
} FS_StyleLinearGradient;

typedef struct FS_StyleRadialGradient {
    float x0;
    float y0;
    float r0;
    float x1;
    float y1;
    float r1;
    FS_GradientStop stops[FS_RADIAL_GRADIENT_MAX_STOPS];
    uint32_t stop_count;
} FS_StyleRadialGradient;

typedef struct FS_StyleConicGradient {
    float start_angle_radians;
    float cx;
    float cy;
    FS_GradientStop stops[FS_CONIC_GRADIENT_MAX_STOPS];
    uint32_t stop_count;
} FS_StyleConicGradient;

typedef struct FS_StylePattern {
    FS_ImageHandle handle;
    float xform[6];
    float inv_xform[6];
    uint8_t repeat_mode;
    uint8_t inv_valid;
    uint8_t _pad0;
    uint8_t _pad1;
} FS_StylePattern;

typedef enum FS_StylePaintType {
    FS_STYLE_PAINT_SOLID = 0,
    FS_STYLE_PAINT_LINEAR_GRADIENT = 1,
    FS_STYLE_PAINT_RADIAL_GRADIENT = 2,
    FS_STYLE_PAINT_CONIC_GRADIENT = 3,
    FS_STYLE_PAINT_PATTERN = 4
} FS_StylePaintType;

typedef struct FS_Transform2D {
    float a;
    float b;
    float c;
    float d;
    float e;
    float f;
} FS_Transform2D;

typedef struct FS_StyleSnapshot {
    float line_width;
    float miter_limit;
    uint8_t line_cap;
    uint8_t line_join;
    uint8_t fill_rule;
    uint8_t composite_op;
    uint8_t text_align;
    uint8_t text_baseline;
    uint8_t text_direction;
    uint8_t font_kerning;
    uint8_t text_rendering;
    uint8_t font_stretch;
    uint8_t font_variant_caps;
    float letter_spacing;
    float word_spacing;
    uint8_t image_smoothing_enabled;
    uint8_t image_smoothing_quality;
    uint8_t _sampling_pad0;
    uint8_t _sampling_pad1;
    float global_alpha;
    uint32_t fill_color_rgba8;
    uint32_t stroke_color_rgba8;
    uint8_t fill_paint_type;
    uint8_t stroke_paint_type;
    uint8_t _paint_pad0;
    uint8_t _paint_pad1;
    FS_StyleLinearGradient fill_linear_gradient;
    FS_StyleLinearGradient stroke_linear_gradient;
    FS_StyleRadialGradient fill_radial_gradient;
    FS_StyleRadialGradient stroke_radial_gradient;
    FS_StyleConicGradient fill_conic_gradient;
    FS_StyleConicGradient stroke_conic_gradient;
    FS_StylePattern fill_pattern;
    FS_StylePattern stroke_pattern;
    uint32_t shadow_color_rgba8;
    float shadow_blur;
    float shadow_offset_x;
    float shadow_offset_y;
    float* dash_segments;
    uint32_t dash_count;
    float dash_offset;
} FS_StyleSnapshot;

typedef struct FS_StateSnapshot {
    FS_Transform2D transform;
    uint8_t clip_enabled;
    uint8_t clip_path_enabled;
    uint8_t clip_path_layer;
    uint8_t _pad0;
    float clip_min_x;
    float clip_min_y;
    float clip_max_x;
    float clip_max_y;
    FS_StyleSnapshot style;
} FS_StateSnapshot;

typedef struct FS_Point2 {
    float x;
    float y;
} FS_Point2;

typedef struct FS_FillContour {
    FS_Point2* points;
    uint32_t count;
    uint32_t capacity;
    float area2;
    float abs_area2;
    int32_t parent;
    uint32_t depth;
    bool is_hole;
    int32_t owner_outer;
} FS_FillContour;

typedef struct FS_HitFillContext {
    float px;
    float py;
    float edge_epsilon;
    int32_t winding;
    uint32_t parity;
    bool on_edge;
    bool evenodd;
} FS_HitFillContext;

typedef struct FS_InternalState {
    const FS_ImageBackend* image_backend;
    const FS_FontBackend* font_backend;
    FS_Core* owner_core;
    void* fonts[FS_MAX_FONT_FALLBACKS];
    uint32_t font_count;

    FS_GlyphEntry* glyphs;
    size_t glyph_count;
    size_t glyph_capacity;
    uint32_t* glyph_hash_slots;
    size_t glyph_hash_capacity;

    FS_ImageFontState* image_fonts;
    uint32_t image_font_count;
    uint32_t image_font_capacity;

    FS_MissingImageGlyph* missing_image_glyphs;
    uint32_t missing_count;
    uint32_t missing_capacity;

    FS_PathSegment* path_segments;
    uint32_t path_count;
    uint32_t path_capacity;
    bool path_has_current;
    bool path_has_subpath_start;
    float path_current_x;
    float path_current_y;
    float path_subpath_start_x;
    float path_subpath_start_y;

    FS_Transform2D current_transform;
    FS_StateSnapshot* state_stack;
    uint32_t state_stack_count;
    uint32_t state_stack_capacity;

    float style_line_width;
    float style_miter_limit;
    uint8_t style_line_cap;
    uint8_t style_line_join;
    uint8_t style_fill_rule;
    uint8_t style_composite_op;
    uint8_t style_text_align;
    uint8_t style_text_baseline;
    uint8_t style_text_direction;
    uint8_t style_font_kerning;
    uint8_t style_text_rendering;
    uint8_t style_font_stretch;
    uint8_t style_font_variant_caps;
    float style_letter_spacing;
    float style_word_spacing;
    uint8_t style_image_smoothing_enabled;
    uint8_t style_image_smoothing_quality;
    uint8_t _style_sampling_pad0;
    uint8_t _style_sampling_pad1;
    float style_global_alpha;
    uint32_t style_fill_color_rgba8;
    uint32_t style_stroke_color_rgba8;
    uint8_t style_fill_paint_type;
    uint8_t style_stroke_paint_type;
    uint8_t _style_paint_pad0;
    uint8_t _style_paint_pad1;
    FS_StyleLinearGradient style_fill_linear_gradient;
    FS_StyleLinearGradient style_stroke_linear_gradient;
    FS_StyleRadialGradient style_fill_radial_gradient;
    FS_StyleRadialGradient style_stroke_radial_gradient;
    FS_StyleConicGradient style_fill_conic_gradient;
    FS_StyleConicGradient style_stroke_conic_gradient;
    FS_StylePattern style_fill_pattern;
    FS_StylePattern style_stroke_pattern;
    uint32_t style_shadow_color_rgba8;
    float style_shadow_blur;
    float style_shadow_offset_x;
    float style_shadow_offset_y;
    uint8_t clip_enabled;
    uint8_t clip_path_enabled;
    uint8_t clip_path_layer;
    uint8_t _clip_pad0;
    uint8_t _clip_pad1;
    float clip_min_x;
    float clip_min_y;
    float clip_max_x;
    float clip_max_y;
    float* style_dash_segments;
    uint32_t style_dash_count;
    float style_dash_offset;
} FS_InternalState;

typedef struct FS_PendingTextureUpload {
    WGPUTexture texture;
    uint32_t mip_level;
    uint32_t layer;
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
    uint32_t padded_row_bytes;
    size_t src_offset;
} FS_PendingTextureUpload;

typedef struct FS_MapReadbackContext {
    volatile uint32_t done;
    volatile uint32_t success;
} FS_MapReadbackContext;

struct FS_Core {
    WGPUDevice device;
    WGPUQueue queue;
    WGPUTextureFormat target_format;
    uint32_t width;
    uint32_t height;

    WGPUBuffer command_buffer;
    size_t command_buffer_size;
    WGPUBuffer command_state_buffer;
    size_t command_state_buffer_size;
    WGPUBuffer vertex_buffer;
    size_t vertex_buffer_size;
    WGPUBuffer uniform_buffer;

    WGPUTexture image_atlas_texture;
    WGPUTextureView image_atlas_view;
    WGPUSampler image_atlas_sampler;
    uint32_t image_atlas_width;
    uint32_t image_atlas_height;
    uint32_t image_atlas_layers;
    uint32_t image_atlas_active_layer;
    uint32_t image_atlas_cursor_x[FS_IMAGE_ATLAS_MAX_LAYERS];
    uint32_t image_atlas_cursor_y[FS_IMAGE_ATLAS_MAX_LAYERS];
    uint32_t image_atlas_row_height[FS_IMAGE_ATLAS_MAX_LAYERS];
    uint32_t image_atlas_generation[FS_IMAGE_ATLAS_MAX_LAYERS];
    uint8_t* image_atlas_shadow_rgba;
    size_t image_atlas_shadow_size;
    uint8_t* canvas_shadow_rgba;
    size_t canvas_shadow_size;
    WGPUBuffer canvas_readback_buffer;
    size_t canvas_readback_buffer_size;
    uint32_t canvas_readback_row_bytes;
    uint32_t canvas_readback_padded_row_bytes;
    uint32_t canvas_readback_width;
    uint32_t canvas_readback_height;
    uint64_t canvas_readback_serial;
    uint64_t canvas_shadow_serial;
    WGPUSubmissionIndex canvas_readback_submission;
    uint8_t canvas_readback_submission_valid;
    uint8_t canvas_readback_mapped;
    FS_ImageHandle canvas_image_data_handle;
    uint8_t canvas_image_data_handle_valid;

    WGPUTexture glyph_atlas_texture;
    WGPUTextureView glyph_atlas_view;
    WGPUSampler glyph_atlas_sampler;
    uint32_t glyph_atlas_width;
    uint32_t glyph_atlas_height;
    uint32_t glyph_atlas_mip_count;
    uint32_t glyph_atlas_cursor_x;
    uint32_t glyph_atlas_cursor_y;
    uint32_t glyph_atlas_row_height;

    WGPUTexture clip_mask_texture;
    WGPUTextureView clip_mask_view;
    WGPUSampler clip_mask_sampler;
    uint32_t clip_mask_width;
    uint32_t clip_mask_height;
    uint32_t clip_mask_layers;
    uint32_t clip_mask_next_layer;
    WGPUTexture msaa_color_texture;
    WGPUTextureView msaa_color_view;
    uint32_t render_sample_count;
    uint8_t* clip_mask_layer_has_data;
    uint32_t* clip_mask_layer_min_x;
    uint32_t* clip_mask_layer_min_y;
    uint32_t* clip_mask_layer_max_x;
    uint32_t* clip_mask_layer_max_y;
    uint64_t* clip_mask_layer_hash;
    uint8_t* clip_mask_layer_hash_valid;
    uint32_t* clip_mask_layer_parent;
    uint32_t* clip_mask_layer_last_used_frame;
    void* clip_edge_cpu;
    size_t clip_edge_count;
    size_t clip_edge_capacity;
    void* clip_job_cpu;
    void* clip_job_xform_cpu;
    size_t clip_job_count;
    size_t clip_job_capacity;
    WGPUBuffer clip_edge_local_buffer;
    size_t clip_edge_local_buffer_size;
    WGPUBuffer clip_edge_buffer;
    size_t clip_edge_buffer_size;
    WGPUBuffer clip_job_buffer;
    size_t clip_job_buffer_size;
    WGPUBuffer clip_job_xform_buffer;
    size_t clip_job_xform_buffer_size;
    WGPUBuffer clip_dispatch_uniform_buffer;
    WGPUBuffer clip_layer_uniform_buffer;
    WGPUBindGroupLayout clip_compute_bgl;
    WGPUBindGroupLayout clip_edge_transform_bgl;
    WGPUBindGroup clip_compute_bg;
    WGPUComputePipeline clip_compute_pipeline;
    WGPUComputePipeline clip_edge_transform_pipeline;

    WGPUBindGroupLayout compute_bgl;
    WGPUBindGroup compute_bg;
    WGPUBindGroupLayout render_bgl;
    WGPUBindGroup render_bg;
    WGPUComputePipeline compute_pipeline;
    WGPURenderPipeline render_pipelines[FS_RENDER_PIPELINE_COUNT];
    int32_t clip_aa_mode_override;
    bool context_lost;
    bool clip_cache_enabled;
    uint32_t clip_requests_this_frame;
    uint32_t clip_cache_hits_this_frame;
    uint32_t clip_jobs_enqueued_this_frame;
    uint32_t clip_layer_reuses_this_frame;
    uint32_t clip_failures_this_frame;
    uint32_t clip_layers_used_this_frame;
    uint32_t clip_last_failure_reason;
    uint32_t clip_last_failure_path_segments;
    uint32_t clip_last_failure_edge_count;
    uint32_t clip_dispatch_batches_this_frame;
    uint32_t clip_dispatch_valid_jobs_this_frame;
    uint64_t clip_dispatch_pixels_ideal_this_frame;
    uint64_t clip_dispatch_pixels_estimated_this_frame;
    uint64_t clip_dispatch_pixels_waste_this_frame;
    uint32_t clip_dispatch_bucket_jobs_this_frame[6];
    uint32_t clip_oriented_quad_commands_this_frame;
    uint32_t clip_oriented_quad_clipped_this_frame;
    uint32_t clip_frame_index;
    uint32_t clip_layer_reuse_reserve;
    FS_ContextAttributes context_attributes;

    FS_Command* commands;
    FS_CommandStateGPU* command_states;
    size_t command_count;
    size_t command_capacity;
    size_t command_state_capacity;

    uint8_t* upload_staging_cpu;
    size_t upload_staging_cpu_capacity;
    size_t upload_staging_used;
    WGPUBuffer upload_staging_gpu;
    size_t upload_staging_gpu_capacity;
    void* pending_uploads;
    size_t pending_upload_count;
    size_t pending_upload_capacity;

    uint8_t* glyph_scratch_rgba[2];
    size_t glyph_scratch_rgba_capacity[2];
    uint8_t* glyph_scratch_alpha[2];
    size_t glyph_scratch_alpha_capacity[2];

    void* internal_state;
};

bool fs_transform_requires_oriented_quad(const FS_Transform2D* t);
void fs_transform_apply_point(const FS_Transform2D* t, float x, float y, float* out_x, float* out_y);
FS_Transform2D fs_transform_identity_value(void);
FS_Transform2D fs_transform_mul(const FS_Transform2D* lhs, const FS_Transform2D* rhs);
void fs_transform_rect_to_aabb(
    const FS_Transform2D* t,
    float x,
    float y,
    float w,
    float h,
    float* out_x,
    float* out_y,
    float* out_w,
    float* out_h
);
void fs_affine_set_identity_2d(float m[6]);
bool fs_affine_try_invert_2d(const float m[6], float out_inv[6]);
void fs_affine_apply_point_2d(const float m[6], float x, float y, float* out_x, float* out_y);
void fs_command_set_oriented_quad_from_rect(FS_Command* cmd, const FS_Transform2D* t, float x, float y, float w, float h);
bool fs_vec2_normalize(float x, float y, float* out_x, float* out_y);
bool fs_transform_points_aabb(
    const FS_Transform2D* t,
    const float* xy,
    uint32_t point_count,
    float* out_min_x,
    float* out_min_y,
    float* out_max_x,
    float* out_max_y
);
void fs_eval_quad_point(float x0, float y0, float cx, float cy, float x1, float y1, float t, float* out_x, float* out_y);
void fs_eval_cubic_point(
    float x0,
    float y0,
    float cx0,
    float cy0,
    float cx1,
    float cy1,
    float x1,
    float y1,
    float t,
    float* out_x,
    float* out_y
);
float fs_transform_metric_scale_cpu(const FS_Transform2D* t);
float fs_distance_sq_point_segment(float px, float py, float x0, float y0, float x1, float y1, float* out_t);
uint32_t fs_align_up_u32(uint32_t value, uint32_t alignment);
uint32_t fs_round_up_pow2_u32(uint32_t v);
float fs_get_text_bake_px(float requested_font_px);
uint32_t fs_compute_mip_count(uint32_t width, uint32_t height);
uint64_t fs_hash64_u32(uint64_t hash, uint32_t value);
uint64_t fs_hash64_mix(uint64_t hash, uint64_t value);
uint64_t fs_hash64_f32(uint64_t hash, float value);
bool fs_ensure_state_stack_capacity(FS_InternalState* st, size_t required);
bool fs_style_snapshot_capture(FS_StyleSnapshot* dst, const FS_InternalState* st);
void fs_style_snapshot_apply(FS_InternalState* st, FS_StyleSnapshot* src);
void fs_state_snapshot_dispose(FS_StateSnapshot* snap);
void fs_state_stack_clear(FS_InternalState* st);
void fs_style_reset_state(FS_InternalState* st);
bool fs_style_has_dash(const FS_InternalState* st);
float fs_style_resolve_line_width(const FS_InternalState* st, float width);
void fs_resolve_text_vertical_metrics(
    const FS_InternalState* st,
    float font_size_px,
    float* out_em_ascent,
    float* out_em_descent,
    float* out_line_height
);
float fs_text_align_offset(const FS_InternalState* st, const FS_TextMetrics* metrics);
bool fs_is_word_spacing_codepoint(uint32_t cp);
float fs_text_baseline_offset(const FS_InternalState* st, const FS_TextMetrics* metrics, float font_size_px);
bool fs_is_text_kerning_enabled(const FS_InternalState* st);
bool fs_is_text_geometric_precision(const FS_InternalState* st);
float fs_text_stretch_scale(const FS_InternalState* st);
bool fs_is_text_small_caps_enabled(const FS_InternalState* st);
void fs_text_variant_map_codepoint(const FS_InternalState* st, uint32_t cp, uint32_t* out_cp, float* out_size_scale);
void fs_clip_reset_state(FS_InternalState* st);
bool fs_style_copy_linear_gradient(FS_StyleLinearGradient* dst, const FS_LinearGradient* src);
bool fs_style_copy_radial_gradient(FS_StyleRadialGradient* dst, const FS_RadialGradient* src);
bool fs_style_copy_conic_gradient(FS_StyleConicGradient* dst, const FS_ConicGradient* src);
bool fs_style_copy_pattern(FS_StylePattern* dst, const FS_Pattern* src);
void fs_apply_emoji_layout_from_bitmap(FS_GlyphEntry* entry, float bake_px, uint32_t bitmap_w, uint32_t bitmap_h);
bool fs_image_atlas_shadow_bounds_ok(
    const FS_Core* core,
    uint32_t layer,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height
);
bool fs_alloc_from_atlas(
    uint32_t atlas_w,
    uint32_t atlas_h,
    uint32_t* cursor_x,
    uint32_t* cursor_y,
    uint32_t* row_h,
    uint32_t padding,
    uint32_t payload_w,
    uint32_t payload_h,
    uint32_t* out_x,
    uint32_t* out_y
);
void fs_discard_pending_uploads_for_texture(FS_Core* core, WGPUTexture texture);
bool fs_ensure_upload_staging_capacity(FS_Core* core, size_t required);
bool fs_ensure_pending_upload_capacity(FS_Core* core, size_t required);
bool fs_queue_write_texture_2d(
    FS_Core* core,
    WGPUTexture texture,
    uint32_t layer,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    const uint8_t* pixels,
    uint32_t bytes_per_pixel
);
bool fs_flush_pending_texture_uploads(FS_Core* core, WGPUCommandEncoder encoder);
bool fs_image_atlas_shadow_read_rgba(
    const FS_Core* core,
    uint32_t layer,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    uint8_t* out_rgba_pixels
);
bool fs_ensure_canvas_shadow(FS_Core* core);
bool fs_target_format_is_bgra(WGPUTextureFormat format);
void fs_canvas_readback_map_callback(
    WGPUMapAsyncStatus status,
    WGPUStringView message,
    void* userdata1,
    void* userdata2
);
bool fs_refresh_canvas_shadow_from_readback(FS_Core* core);
bool fs_ensure_canvas_image_data_handle(FS_Core* core, uint32_t width, uint32_t height, FS_ImageHandle* out_handle);
bool fs_encode_canvas_readback_copy(FS_Core* core, WGPUCommandEncoder encoder, WGPUTexture target_texture);
WGPUBuffer fs_create_buffer(WGPUDevice device, const char* label, WGPUBufferUsage usage, size_t size);
bool fs_ensure_clip_edge_gpu_capacity(FS_Core* core, size_t required);
bool fs_ensure_clip_job_gpu_capacity(FS_Core* core, size_t required);
bool fs_ensure_clip_job_transform_gpu_capacity(FS_Core* core, size_t required);
bool fs_ensure_clip_edge_cpu_capacity(FS_Core* core, size_t required);
bool fs_ensure_clip_job_cpu_capacity(FS_Core* core, size_t required);
bool fs_recreate_clip_compute_bind_group(FS_Core* core);
WGPUBindGroup fs_create_clip_compute_bind_group_range(FS_Core* core, uint64_t job_offset_bytes, uint64_t job_size_bytes);
WGPUBindGroup fs_create_clip_edge_transform_bind_group_range(
    FS_Core* core,
    uint64_t job_offset_bytes,
    uint64_t job_size_bytes,
    uint64_t xform_offset_bytes,
    uint64_t xform_size_bytes
);
FS_InternalState* fs_state(FS_Core* core);
bool fs_push_command(FS_Core* core, const FS_Command* cmd);
uint32_t fs_quantize_font_size(float font_px);
uint32_t fs_decode_utf8(const char** cursor);
bool fs_pack_image_glyph_key(uint32_t image_font_id, uint32_t glyph_id, uint32_t* out_key);
uint32_t fs_unpack_image_key_font_id(uint32_t key);
uint32_t fs_unpack_image_key_glyph_id(uint32_t key);
bool fs_is_emoji_modifier(uint32_t cp);
bool fs_is_emoji_variation_selector(uint32_t cp);
bool fs_is_zwj(uint32_t cp);
bool fs_is_keycap_combiner(uint32_t cp);
bool fs_canonicalize_utf8_sequence(const char* utf8, uint32_t* out_cps, uint8_t* out_cp_count);
bool fs_match_sequence_prefix(const FS_ImageSequenceEntry* seq, const char* utf8_ptr, size_t* out_bytes);
bool fs_ensure_image_font_capacity(FS_InternalState* st, uint32_t required);
bool fs_ensure_missing_capacity(FS_InternalState* st, uint32_t required);
FS_ImageFontState* fs_find_image_font(FS_InternalState* st, uint32_t image_font_id);
bool fs_image_font_ensure_glyph_slots(FS_ImageFontState* font, uint32_t required_slots);
bool fs_push_missing_image_glyph(FS_InternalState* st, uint32_t image_font_id, uint32_t glyph_id, const char* utf8, size_t utf8_len);
bool fs_find_image_sequence_match(
    const FS_InternalState* st,
    const char* utf8_ptr,
    uint32_t* out_image_font_id,
    uint32_t* out_glyph_id,
    size_t* out_consumed_bytes
);
uint32_t fs_glyph_cache_hash_key(uint32_t glyph_key, uint32_t key_kind, uint32_t bake_px_q);
size_t fs_glyph_cache_next_pow2(size_t v);
bool fs_glyph_cache_rebuild(FS_InternalState* st, size_t min_capacity);
void fs_glyph_cache_clear_index(FS_InternalState* st);
FS_GlyphEntry* fs_glyph_cache_find(FS_InternalState* st, uint32_t glyph_key, uint32_t key_kind, uint32_t bake_px_q);
bool fs_glyph_cache_insert_index(FS_InternalState* st, size_t glyph_index);
void fs_invalidate_cached_glyph(FS_InternalState* st, uint32_t key_kind, uint32_t glyph_key);
int fs_sequence_entry_sort_desc(const void* lhs, const void* rhs);
bool fs_ensure_glyph_capacity(FS_InternalState* st, size_t required);
bool fs_find_or_create_glyph(
    FS_Core* core,
    uint32_t glyph_key,
    uint32_t key_kind,
    float font_px,
    FS_GlyphEntry** out_glyph,
    float* out_layout_scale
);
float fs_get_kerning_advance(
    FS_Core* core,
    uint32_t left_codepoint,
    uint32_t right_codepoint,
    float font_px,
    uint8_t font_slot
);
bool fs_upload_glyph_with_mips(FS_Core* core, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const uint8_t* pixels, uint32_t pixel_format, float sdf_onedge, float sdf_pixel_dist_scale);
bool fs_image_handle_resolve_atlas_origin(
    const FS_Core* core,
    const FS_ImageHandle* handle,
    uint32_t* out_atlas_x,
    uint32_t* out_atlas_y
);
void fs_command_state_clear_pattern(FS_CommandStateGPU* state);
bool fs_command_state_set_pattern(FS_Core* core, FS_CommandStateGPU* state, const FS_StylePattern* pattern);
bool fs_style_pattern_sample_rgba8(const FS_InternalState* st, const FS_StylePattern* pattern, float px, float py, uint32_t* out_color);
uint32_t fs_linear_gradient_sample_rgba8(const FS_StyleLinearGradient* grad, float px, float py);
uint32_t fs_conic_gradient_sample_rgba8(const FS_StyleConicGradient* grad, float px, float py);
uint32_t fs_radial_gradient_sample_rgba8(const FS_StyleRadialGradient* grad, float px, float py);
uint32_t fs_style_resolve_fill_color_at(const FS_InternalState* st, float px, float py);
uint32_t fs_style_resolve_stroke_color_at(const FS_InternalState* st, float px, float py);
bool fs_draw_linear_gradient_rect_cells(FS_Core* core, float x, float y, float w, float h, const FS_StyleLinearGradient* grad);
bool fs_draw_radial_gradient_rect_cells(FS_Core* core, float x, float y, float w, float h, const FS_StyleRadialGradient* grad);
bool fs_draw_conic_gradient_rect_cells(FS_Core* core, float x, float y, float w, float h, const FS_StyleConicGradient* grad);
bool fs_emit_fill_triangle_fan(FS_Core* core, const FS_Point2* points, uint32_t count, uint32_t color);
bool fs_emit_fill_triangles_ear_clip(FS_Core* core, const FS_Point2* points, uint32_t count, uint32_t color, bool allow_fan_fallback);
float fs_polygon_signed_area2(const FS_Point2* points, uint32_t count);
float fs_cross2(const FS_Point2* a, const FS_Point2* b, const FS_Point2* c);
bool fs_point_in_contour(const FS_Point2* points, uint32_t count, const FS_Point2* p);
void fs_points_reverse(FS_Point2* points, uint32_t count);
bool fs_point_in_triangle_or_edge(const FS_Point2* p, const FS_Point2* a, const FS_Point2* b, const FS_Point2* c);
uint32_t fs_find_rightmost_point(const FS_Point2* points, uint32_t count);
int fs_orient2d(const FS_Point2* a, const FS_Point2* b, const FS_Point2* c);
bool fs_point_on_segment(const FS_Point2* p, const FS_Point2* a, const FS_Point2* b);
bool fs_segments_intersect(const FS_Point2* a, const FS_Point2* b, const FS_Point2* c, const FS_Point2* d);
bool fs_bridge_visible(const FS_Point2* outer, uint32_t outer_count, uint32_t outer_idx, const FS_Point2* hole, uint32_t hole_count, uint32_t hole_idx);
bool fs_find_outer_bridge_point(const FS_Point2* outer, uint32_t outer_count, const FS_Point2* hole, uint32_t hole_count, uint32_t hole_idx, uint32_t* out_outer_idx);
bool fs_merge_hole_into_polygon(FS_Point2** io_poly, uint32_t* io_count, uint32_t* io_capacity, FS_Point2* hole, uint32_t hole_count, uint32_t hole_right_idx);
bool fs_fill_points_reserve(FS_Point2** io_points, uint32_t* io_capacity, uint32_t required);
bool fs_fill_points_push_unique(FS_Point2** io_points, uint32_t* io_count, uint32_t* io_capacity, float x, float y);
void fs_fill_contour_clear(FS_FillContour* contour);
bool fs_fill_contours_reserve(FS_FillContour** io_contours, uint32_t* io_capacity, uint32_t required);
bool fs_contour_finalize(FS_FillContour* contour);
bool fs_polygon_compact_in_place(FS_Point2* points, uint32_t* io_count);
float fs_arc_resolve_delta(float start_angle, float end_angle, bool counterclockwise);
void fs_path_state_borrow(const FS_InternalState* st, FS_PathStateBorrow* out_state);
void fs_path_state_bind_path2d(FS_InternalState* st, const FS_Path2D* path);
void fs_path_state_restore(FS_InternalState* st, const FS_PathStateBorrow* saved);
void fs_path_state_begin_temporary(FS_InternalState* st, FS_PathStateBorrow* out_saved);
void fs_path_state_end_temporary(FS_InternalState* st, const FS_PathStateBorrow* saved);
bool fs_path_append_arc_sweep(FS_Core* core, float cx, float cy, float radius, float a0, float a1);

// Style API
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
void fs_style_clear_dash(FS_Core* core);
bool fs_style_set_dash(FS_Core* core, const float* segments, uint32_t segment_count, float offset);
bool fs_style_get_dash(const FS_Core* core, float* out_segments, uint32_t max_segments, uint32_t* out_count, float* out_offset);

#endif
