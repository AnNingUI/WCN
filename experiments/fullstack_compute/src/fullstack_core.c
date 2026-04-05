#include "fullstack_core.h"
#include "fullstack_core_debug.h"
#include "fullstack_shaders.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FS_IMAGE_ATLAS_SIZE 2048u
#define FS_IMAGE_ATLAS_MAX_LAYERS 4u
#define FS_GLYPH_ATLAS_SIZE 2048u
#define FS_RENDER_PIPELINE_COUNT 11u
#define FS_IMAGE_ATLAS_PADDING 1u
#define FS_GLYPH_ATLAS_PADDING 4u
#define FS_ATLAS_PADDING 1u
#define FS_CLIP_MASK_LAYERS 64u
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

#define FS_CMD_CLIP_RECT_BIT 1u
#define FS_CMD_CLIP_PATH_BIT 2u
#define FS_RENDER_FLAG_PATTERN_FILL_HINT (1u << 0u)
#define FS_RENDER_FLAG_IMAGE_NEAREST (1u << 1u)
#define FS_TRI_FLAG_AA_EDGE0 (1u << 2u)
#define FS_TRI_FLAG_AA_EDGE1 (1u << 3u)
#define FS_TRI_FLAG_AA_EDGE2 (1u << 4u)
#define FS_TRI_FLAG_AA_ALL (FS_TRI_FLAG_AA_EDGE0 | FS_TRI_FLAG_AA_EDGE1 | FS_TRI_FLAG_AA_EDGE2)
#define FS_LINE_FLAG_BUTT (1u << 5u)
#define FS_LINE_FLAG_NO_AA_START (1u << 6u)
#define FS_LINE_FLAG_NO_AA_END (1u << 7u)
#define FS_RENDER_FLAG_SHADOW_BLUR_SHIFT 8u
#define FS_RENDER_FLAG_SHADOW_BLUR_MASK (0xFu << FS_RENDER_FLAG_SHADOW_BLUR_SHIFT)
#define FS_RENDER_FLAG_SHADOW (1u << 12u)
#define FS_RENDER_FLAG_PATTERN_SHADE (1u << 13u)
#define FS_RENDER_FLAG_ORIENTED_QUAD (1u << 14u)
#define FS_RENDER_FLAG_LOCAL_SPACE (1u << 15u)

#define FS_RENDER_FLAG_USER_MASK 0x0000FFFFu
#define FS_RENDER_FLAG_COMPOSITE_SHIFT 8u
#define FS_RENDER_FLAG_COMPOSITE_MASK (0xFu << FS_RENDER_FLAG_COMPOSITE_SHIFT)
#define FS_RENDER_FLAG_CLIP_MASK (1u << 16u)
#define FS_RENDER_FLAG_CLIP_PARENT_SHIFT 17u
#define FS_RENDER_FLAG_CLIP_PARENT_MASK (0x7Fu << FS_RENDER_FLAG_CLIP_PARENT_SHIFT)
#define FS_RENDER_FLAG_CLIP_LAYER_SHIFT 24u
#define FS_RENDER_FLAG_CLIP_LAYER_MASK (0xFFu << FS_RENDER_FLAG_CLIP_LAYER_SHIFT)
#define FS_CLIP_FILL_MODE_COVERAGE 0u
#define FS_CLIP_FILL_MODE_SDF 1u
#define FS_CLIP_FILL_MODE_ROUND_RECT 2u
#define FS_SHADOW_BLUR_MAX 15.0f
#define FS_SHADOW_SEPARABLE_THRESHOLD 5.0f
#define FS_SHADOW_SEPARABLE_STEP_SCALE 0.55f

typedef enum FS_CommandType {
    FS_CMD_RECT = 0,
    FS_CMD_IMAGE = 1,
    FS_CMD_TEXT = 2,
    FS_CMD_LINE = 3,
    FS_CMD_PATH_SEGMENT = 4,
    FS_CMD_CIRCLE = 5,
    FS_CMD_ARC = 6,
    FS_CMD_BEZIER_QUAD = 7,
    FS_CMD_RECT_STROKE = 8,
    FS_CMD_ELLIPSE = 9,
    FS_CMD_BEZIER_CUBIC = 10,
    FS_CMD_TRIANGLE = 11
} FS_CommandType;

typedef struct FS_Command {
    float p0[4];
    float p1[4];
    float p2[4];
    float quad0[4];
    float quad1[4];
    float clip_min[2];
    float clip_max[2];
    uint32_t clip_enabled;
    uint32_t state_index;
    uint32_t color_rgba8;
    uint32_t type;
    uint32_t flags;
    float scalar;
    uint32_t _tail_pad[2];
} FS_Command;

typedef struct FS_Uniforms {
    float viewport[2];
    uint32_t command_count;
    uint32_t clip_enabled;
    float clip_min[2];
    float clip_max[2];
} FS_Uniforms;

typedef struct FS_CommandStateGPU {
    float clip_rect[4];
    uint32_t clip_meta[4];
    float xform0[4];
    float xform1[4];
    float pattern_inv0[4];
    float pattern_inv1[4];
    float pattern_meta[4];
} FS_CommandStateGPU;

enum {
    FS_TEXT_FLAG_COLOR_GLYPH = 1u << 0,
    FS_TEXT_FLAG_STROKE = 1u << 1
};

typedef struct FS_VertexGPU {
    float clip_pos[4];
    float color[4];
    float uv[2];
    float world_pos[2];
    uint32_t cmd_type;
    uint32_t flags;
    uint32_t state_index;
    uint32_t _pad_state_index;
    float extra0[4];
    float extra1[4];
    float extra2[4];
} FS_VertexGPU;

_Static_assert(sizeof(FS_Command) == 128u, "FS_Command must match WGSL Command stride (128 bytes)");
_Static_assert(sizeof(FS_CommandStateGPU) == 112u, "FS_CommandStateGPU must match WGSL CommandState stride (112 bytes)");

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

typedef struct FS_Path2D {
    FS_PathSegment* segments;
    uint32_t count;
    uint32_t capacity;
    bool has_current;
    bool has_subpath_start;
    float current_x;
    float current_y;
    float subpath_start_x;
    float subpath_start_y;
} FS_Path2D;

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

#define FS_LINEAR_GRADIENT_MAX_STOPS 16u
#define FS_RADIAL_GRADIENT_MAX_STOPS 16u
#define FS_CONIC_GRADIENT_MAX_STOPS 16u

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

typedef struct FS_ClipEdgeGPU {
    float x0;
    float y0;
    float x1;
    float y1;
} FS_ClipEdgeGPU;

typedef struct FS_ClipJobGPU {
    uint32_t edge_offset;
    uint32_t edge_count;
    uint32_t layer;
    uint32_t fill_rule;
    uint32_t fill_min_x;
    uint32_t fill_min_y;
    uint32_t fill_max_x;
    uint32_t fill_max_y;
    uint32_t clear_min_x;
    uint32_t clear_min_y;
    uint32_t clear_max_x;
    uint32_t clear_max_y;
    uint32_t parent_layer;
    uint32_t has_parent;
    uint32_t scale_hint_bits;
    uint32_t fill_mode;
} FS_ClipJobGPU;

typedef struct FS_ClipJobTransformGPU {
    float xform0[4];
    float xform1[4];
} FS_ClipJobTransformGPU;

typedef struct FS_ClipDispatchUniforms {
    uint32_t job_count;
    uint32_t job_offset;
    uint32_t viewport_width;
    uint32_t viewport_height;
    uint32_t aa_mode;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
} FS_ClipDispatchUniforms;

typedef struct FS_ClipLayerUniforms {
    uint32_t parent[FS_CLIP_MASK_LAYERS];
    uint32_t min_x[FS_CLIP_MASK_LAYERS];
    uint32_t min_y[FS_CLIP_MASK_LAYERS];
    uint32_t max_x[FS_CLIP_MASK_LAYERS];
    uint32_t max_y[FS_CLIP_MASK_LAYERS];
} FS_ClipLayerUniforms;

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

_Static_assert(sizeof(FS_ClipEdgeGPU) == 16u, "FS_ClipEdgeGPU must match WGSL ClipEdge stride (16 bytes)");
_Static_assert(sizeof(FS_ClipJobGPU) == 64u, "FS_ClipJobGPU must match WGSL ClipJob stride (64 bytes)");
_Static_assert(sizeof(FS_ClipJobTransformGPU) == 32u, "FS_ClipJobTransformGPU must match WGSL ClipJobTransform stride (32 bytes)");
_Static_assert(sizeof(FS_ClipDispatchUniforms) == 32u, "FS_ClipDispatchUniforms must match WGSL ClipDispatch layout");
_Static_assert(sizeof(FS_ClipLayerUniforms) == (size_t)FS_CLIP_MASK_LAYERS * 5u * sizeof(uint32_t),
               "FS_ClipLayerUniforms must match WGSL ClipLayers layout");


// ========== MISSING FORWARD DECLARATIONS (restored) ==========
static FS_InternalState* fs_state(FS_Core* core);
static void fs_transform_apply_point(const FS_Transform2D* t, float x, float y, float* out_x, float* out_y);
static FS_Transform2D fs_transform_identity_value(void);
static FS_Transform2D fs_transform_mul(const FS_Transform2D* lhs, const FS_Transform2D* rhs);
static bool fs_ensure_transform_stack_capacity(FS_InternalState* st, uint32_t required);
static void fs_affine_set_identity_2d(float m[6]);
static bool fs_push_command(FS_Core* core, const FS_Command* cmd);
static void fs_clip_diag_reset_frame(FS_Core* core);
static void fs_path_state_begin_temporary(FS_InternalState* st, FS_PathStateBorrow* out_saved);
static void fs_path_state_end_temporary(FS_InternalState* st, const FS_PathStateBorrow* saved);
static bool fs_emit_styled_line_segment_with_flags(FS_Core* core, float x0, float y0, float x1, float y1, float width, uint32_t color, uint8_t line_cap, uint32_t extra_line_flags);
static bool fs_cmd_path_segment_with_flags(FS_Core* core, float x0, float y0, float x1, float y1, float width, uint32_t color, uint32_t user_flags);
static bool fs_cmd_triangle_with_edge_mask(FS_Core* core, float x0, float y0, float x1, float y1, float x2, float y2, uint32_t color, uint32_t tri_aa_mask, uint32_t extra_user_flags);
static bool fs_cmd_circle_with_flags(FS_Core* core, float cx, float cy, float radius, uint32_t color, uint32_t user_flags);
static bool fs_cmd_rect_with_flags(FS_Core* core, float x, float y, float w, float h, float radius, uint32_t color, uint32_t extra_flags);
static bool fs_path_append_arc_sweep(FS_Core* core, float cx, float cy, float radius, float a0, float a1);
static bool fs_emit_styled_line_segment_compute_coverage(
    FS_Core* core,
    float x0,
    float y0,
    float x1,
    float y1,
    float width,
    uint32_t color,
    uint8_t line_cap,
    uint32_t extra_line_flags
);
static bool fs_cmd_bezier_quad_with_flags(FS_Core* core, float x0, float y0, float cx, float cy, float x1, float y1, float width, uint32_t color, uint32_t user_flags);
static bool fs_cmd_bezier_cubic_with_flags(FS_Core* core, float x0, float y0, float cx0, float cy0, float cx1, float cy1, float x1, float y1, float width, uint32_t color, uint32_t user_flags);
static WGPUShaderModule fs_create_shader_module(WGPUDevice device, const char* wgsl_code, const char* label);
static bool fs_recreate_clip_compute_bind_group(FS_Core* core);
static void fs_discard_pending_uploads_for_texture(FS_Core* core, WGPUTexture texture);
static bool fs_encode_canvas_readback_copy(FS_Core* core, WGPUCommandEncoder encoder, WGPUTexture target_texture);
static void fs_canvas_readback_map_callback(
    WGPUMapAsyncStatus status,
    WGPUStringView message,
    void* userdata1,
    void* userdata2
);
static bool fs_ensure_canvas_image_data_handle(FS_Core* core, uint32_t width, uint32_t height, FS_ImageHandle* out_handle);
static bool fs_ensure_canvas_shadow(FS_Core* core);
static bool fs_ensure_clip_edge_gpu_capacity(FS_Core* core, size_t required);
static bool fs_ensure_clip_job_gpu_capacity(FS_Core* core, size_t required);
static bool fs_ensure_clip_job_transform_gpu_capacity(FS_Core* core, size_t required);
static bool fs_ensure_clip_edge_cpu_capacity(FS_Core* core, size_t required);
static bool fs_ensure_clip_job_cpu_capacity(FS_Core* core, size_t required);
static bool fs_ensure_state_stack_capacity(FS_InternalState* st, size_t required);
static void fs_state_stack_clear(FS_InternalState* st);
static bool fs_style_has_dash(const FS_InternalState* st);
static bool fs_flush_pending_texture_uploads(FS_Core* core, WGPUCommandEncoder encoder);
static bool fs_image_atlas_shadow_read_rgba(const FS_Core* core, uint32_t layer, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t* out_rgba_pixels);
static uint64_t fs_hash64_u32(uint64_t hash, uint32_t value);
static uint64_t fs_hash64_mix(uint64_t hash, uint64_t value);
static uint64_t fs_hash64_f32(uint64_t hash, float value);
static bool fs_queue_write_texture_2d(FS_Core* core, WGPUTexture texture, uint32_t layer, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const uint8_t* pixels, uint32_t bytes_per_pixel);
static bool fs_refresh_canvas_shadow_from_readback(FS_Core* core);
static void fs_command_set_oriented_quad_from_rect(FS_Command* cmd, const FS_Transform2D* t, float x, float y, float w, float h);
static bool fs_cmd_ellipse_compute_coverage_stroke_with_flags(FS_Core* core, float x0, float y0, float x1, float y1, float width, uint32_t color, uint32_t user_flags);
static bool fs_cmd_polygon_compute_coverage_fill_with_flags(FS_Core* core, const float* xy, uint32_t point_count, uint32_t color, uint32_t extra_flags);
static bool fs_polygon_compact_in_place(FS_Point2* points, uint32_t* in_out_count);
static bool fs_upload_glyph_with_mips(FS_Core* core, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const uint8_t* pixels, uint32_t pixel_format, float sdf_onedge, float sdf_pixel_dist_scale);
// ============================================================

// ========== MISSING FUNCTION DEFINITIONS (restored) ==========

static uint32_t fs_align_up_u32(uint32_t value, uint32_t alignment) {
    if (alignment == 0u) {
        return value;
    }
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static uint32_t fs_round_up_pow2_u32(uint32_t v) {
    if (v <= 1u) {
        return 1u;
    }
    v -= 1u;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    return v + 1u;
}

static float fs_get_text_bake_px(float requested_font_px) {
    const float requested = requested_font_px < 1.0f ? 1.0f : requested_font_px;
    const float clamped = requested < FS_TEXT_BAKE_MIN_PX ? FS_TEXT_BAKE_MIN_PX : requested;
    const uint32_t rounded = fs_round_up_pow2_u32((uint32_t)ceilf(clamped));
    return (float)rounded * FS_TEXT_BAKE_SCALE;
}

static uint32_t fs_compute_mip_count(uint32_t width, uint32_t height) {
    uint32_t max_dim = width > height ? width : height;
    uint32_t levels = 1u;
    while (max_dim > 1u) {
        max_dim >>= 1u;
        levels++;
    }
    return levels;
}

static bool fs_transform_requires_oriented_quad(const FS_Transform2D* t) {
    if (!t) {
        return false;
    }
    const float eps = 1e-6f;
    if (fabsf(t->b) > eps || fabsf(t->c) > eps) {
        return true;
    }
    if (t->a < -eps || t->d < -eps) {
        return true;
    }
    return false;
}

static FS_InternalState* fs_state(FS_Core* core) {
    return core ? core->internal_state : NULL;
}

static void fs_transform_apply_point(const FS_Transform2D* t, float x, float y, float* out_x, float* out_y) {
    if (out_x) {
        *out_x = t ? (t->a * x + t->c * y + t->e) : x;
    }
    if (out_y) {
        *out_y = t ? (t->b * x + t->d * y + t->f) : y;
    }
}

static FS_Transform2D fs_transform_identity_value(void) {
    FS_Transform2D result = {.a = 1.0f, .b = 0.0f, .c = 0.0f, .d = 1.0f, .e = 0.0f, .f = 0.0f};
    return result;
}

static FS_Transform2D fs_transform_mul(const FS_Transform2D* lhs, const FS_Transform2D* rhs) {
    FS_Transform2D result;
    if (!lhs || !rhs) {
        result.a = 1.0f; result.b = 0.0f; result.c = 0.0f;
        result.d = 1.0f; result.e = 0.0f; result.f = 0.0f;
        return result;
    }
    result.a = lhs->a * rhs->a + lhs->c * rhs->b;
    result.b = lhs->b * rhs->a + lhs->d * rhs->b;
    result.c = lhs->a * rhs->c + lhs->c * rhs->d;
    result.d = lhs->b * rhs->c + lhs->d * rhs->d;
    result.e = lhs->a * rhs->e + lhs->c * rhs->f + lhs->e;
    result.f = lhs->b * rhs->e + lhs->d * rhs->f + lhs->f;
    return result;
}

static bool fs_cmd_path_segment_with_flags(
    FS_Core* core,
    float x0,
    float y0,
    float x1,
    float y1,
    float width,
    uint32_t color,
    uint32_t user_flags
) {
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    if (t && fs_transform_requires_oriented_quad(t)) {
        return fs_cmd_ellipse_compute_coverage_stroke_with_flags(core, x0, y0, x1, y1, width, color, user_flags);
    }
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = x0;
    cmd.p0[1] = y0;
    cmd.p0[2] = x1;
    cmd.p0[3] = y1;
    cmd.scalar = width;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_PATH_SEGMENT;
    cmd.flags = user_flags & FS_RENDER_FLAG_USER_MASK;
    cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    return fs_push_command(core, &cmd);
}

static bool fs_cmd_triangle_with_edge_mask(
    FS_Core* core,
    float x0, float y0,
    float x1, float y1,
    float x2, float y2,
    uint32_t color,
    uint32_t tri_aa_mask,
    uint32_t extra_user_flags
) {
    if (!core) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    if (t && fs_transform_requires_oriented_quad(t)) {
        const float tri[6] = {x0, y0, x1, y1, x2, y2};
        return fs_cmd_polygon_compute_coverage_fill_with_flags(core, tri, 3u, color, extra_user_flags & FS_RENDER_FLAG_PATTERN_SHADE);
    }
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = x0;
    cmd.p0[1] = y0;
    cmd.p0[2] = x1;
    cmd.p0[3] = y1;
    cmd.p1[0] = x2;
    cmd.p1[1] = y2;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_TRIANGLE;
    uint32_t flags = (tri_aa_mask | extra_user_flags) & FS_RENDER_FLAG_USER_MASK;
    flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    cmd.flags = flags;
    return fs_push_command(core, &cmd);
}

static void fs_apply_emoji_layout_from_bitmap(FS_GlyphEntry* entry, float bake_px, uint32_t bitmap_w, uint32_t bitmap_h) {
    if (!entry) {
        return;
    }
    const float emoji_h = bake_px * FS_EMOJI_SCALE_BIAS;
    float aspect = 1.0f;
    if (bitmap_h > 0u) {
        aspect = (float)bitmap_w / (float)bitmap_h;
    }
    if (aspect < 0.75f) {
        aspect = 0.75f;
    } else if (aspect > 1.25f) {
        aspect = 1.25f;
    }
    const float emoji_w = emoji_h * aspect;
    float emoji_advance = emoji_w * (1.0f + FS_EMOJI_ADVANCE_BIAS);
    const float advance_quant = bake_px * FS_EMOJI_ADVANCE_ALIGN;
    if (advance_quant > 1e-3f) {
        emoji_advance = floorf((emoji_advance / advance_quant) + 0.5f) * advance_quant;
    }
    entry->advance = emoji_advance;
    entry->bearing_x = emoji_w * FS_EMOJI_ADVANCE_BIAS * FS_EMOJI_ADVANCE_ALIGN;
    entry->bearing_y = -emoji_h * FS_EMOJI_ASCENT;
    entry->atlas_width = emoji_w;
    entry->atlas_height = emoji_h;
    entry->sdf_radius_px = 1.0f;
    entry->sdf_onedge = 0.5f;
    entry->sdf_pixel_dist_scale = 1.0f;
    entry->text_flags = FS_TEXT_FLAG_COLOR_GLYPH;
}

static bool fs_image_atlas_shadow_bounds_ok(const FS_Core* core, uint32_t layer, uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    if (!core || !core->image_atlas_shadow_rgba || core->image_atlas_shadow_size == 0u) {
        return false;
    }
    if (layer >= core->image_atlas_layers || width == 0u || height == 0u) {
        return false;
    }
    if (x > core->image_atlas_width || y > core->image_atlas_height) {
        return false;
    }
    if (width > core->image_atlas_width - x || height > core->image_atlas_height - y) {
        return false;
    }
    return true;
}

static bool fs_alloc_from_atlas(uint32_t atlas_w, uint32_t atlas_h, uint32_t* cursor_x, uint32_t* cursor_y, uint32_t* row_h, uint32_t padding, uint32_t payload_w, uint32_t payload_h, uint32_t* out_x, uint32_t* out_y) {
    if (!cursor_x || !cursor_y || !row_h || !out_x || !out_y || payload_w == 0u || payload_h == 0u) {
        return false;
    }
    const uint32_t alloc_w = payload_w + 2u * padding;
    const uint32_t alloc_h = payload_h + 2u * padding;
    if (alloc_w > atlas_w || alloc_h > atlas_h) {
        return false;
    }
    if (*cursor_x + alloc_w > atlas_w) {
        *cursor_x = 0u;
        *cursor_y += *row_h;
        *row_h = 0u;
    }
    if (*cursor_y + alloc_h > atlas_h) {
        return false;
    }
    *out_x = *cursor_x + padding;
    *out_y = *cursor_y + padding;
    *cursor_x += alloc_w;
    if (alloc_h > *row_h) {
        *row_h = alloc_h;
    }
    return true;
}

static WGPUBuffer fs_create_buffer(WGPUDevice device, const char* label, WGPUBufferUsage usage, size_t size) {
    WGPUBufferDescriptor desc = {
        .nextInChain = NULL,
        .label = label,
        .usage = usage,
        .size = size,
        .mappedAtCreation = false
    };
    return wgpuDeviceCreateBuffer(device, &desc);
}

static WGPUShaderModule fs_create_shader_module(WGPUDevice device, const char* wgsl_code, const char* label) {
    WGPUStringView code_view = {
        .data = wgsl_code,
        .length = strlen(wgsl_code)
    };
    WGPUShaderSourceWGSL source = {
        .chain = {
            .next = NULL,
            .sType = WGPUSType_ShaderSourceWGSL
        },
        .code = code_view
    };
    WGPUShaderModuleDescriptor desc = {
        .nextInChain = &source.chain,
        .label = {
            .data = label,
            .length = strlen(label)
        }
    };
    return wgpuDeviceCreateShaderModule(device, &desc);
}

static void fs_discard_pending_uploads_for_texture(FS_Core* core, WGPUTexture texture) {
    if (!core || !texture || core->pending_upload_count == 0u || !core->pending_uploads) {
        return;
    }
    if (!core->upload_staging_cpu) {
        core->pending_upload_count = 0u;
        core->upload_staging_used = 0u;
        return;
    }

    FS_PendingTextureUpload* uploads = (FS_PendingTextureUpload*)core->pending_uploads;
    size_t write_idx = 0u;
    size_t new_used = 0u;

    for (size_t i = 0u; i < core->pending_upload_count; ++i) {
        FS_PendingTextureUpload u = uploads[i];
        if (u.texture == texture) {
            continue;
        }
        const size_t upload_size = (size_t)u.padded_row_bytes * (size_t)u.height;
        const size_t new_offset = (size_t)fs_align_up_u32((uint32_t)new_used, 256u);
        if (new_offset != u.src_offset) {
            memmove(
                core->upload_staging_cpu + new_offset,
                core->upload_staging_cpu + u.src_offset,
                upload_size
            );
        }
        u.src_offset = new_offset;
        uploads[write_idx++] = u;
        new_used = new_offset + upload_size;
    }

    core->pending_upload_count = write_idx;
    core->upload_staging_used = (write_idx > 0u) ? new_used : 0u;
}

static bool fs_ensure_upload_staging_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    if (required <= core->upload_staging_cpu_capacity) {
        return true;
    }
    size_t new_capacity = core->upload_staging_cpu_capacity ? core->upload_staging_cpu_capacity : 4096u;
    while (new_capacity < required) {
        if (new_capacity > (SIZE_MAX / 2u)) {
            new_capacity = required;
            break;
        }
        new_capacity *= 2u;
    }
    uint8_t* grown = (uint8_t*)realloc(core->upload_staging_cpu, new_capacity);
    if (!grown) {
        return false;
    }
    core->upload_staging_cpu = grown;
    core->upload_staging_cpu_capacity = new_capacity;
    return true;
}

static bool fs_ensure_pending_upload_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    if (required <= core->pending_upload_capacity) {
        return true;
    }
    size_t new_capacity = core->pending_upload_capacity ? core->pending_upload_capacity : 128u;
    while (new_capacity < required) {
        if (new_capacity > (SIZE_MAX / 2u)) {
            new_capacity = required;
            break;
        }
        new_capacity *= 2u;
    }
    FS_PendingTextureUpload* grown =
        (FS_PendingTextureUpload*)realloc(core->pending_uploads, new_capacity * sizeof(FS_PendingTextureUpload));
    if (!grown) {
        return false;
    }
    core->pending_uploads = grown;
    core->pending_upload_capacity = new_capacity;
    return true;
}

static bool fs_queue_write_texture_2d(
    FS_Core* core,
    WGPUTexture texture,
    uint32_t layer,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    const uint8_t* pixels,
    uint32_t bytes_per_pixel
) {
    if (!core || !texture || !pixels || width == 0u || height == 0u || bytes_per_pixel == 0u) {
        return false;
    }
    const uint64_t tight_row = (uint64_t)width * (uint64_t)bytes_per_pixel;
    if (tight_row == 0u || tight_row > UINT32_MAX) {
        return false;
    }
    const uint32_t padded_row = fs_align_up_u32((uint32_t)tight_row, 256u);
    const size_t upload_size = (size_t)padded_row * (size_t)height;
    const size_t upload_offset = core->upload_staging_used;
    const size_t required_size = upload_offset + upload_size;
    if (required_size < upload_offset) {
        return false;
    }
    if (!fs_ensure_upload_staging_capacity(core, required_size)) {
        return false;
    }
    if (!fs_ensure_pending_upload_capacity(core, core->pending_upload_count + 1u)) {
        return false;
    }

    uint8_t* dst = core->upload_staging_cpu + upload_offset;
    const size_t src_row = (size_t)width * (size_t)bytes_per_pixel;
    for (uint32_t row = 0u; row < height; ++row) {
        memcpy(dst + (size_t)row * (size_t)padded_row, pixels + (size_t)row * src_row, src_row);
        if (padded_row > src_row) {
            memset(dst + (size_t)row * (size_t)padded_row + src_row, 0, (size_t)padded_row - src_row);
        }
    }

    if (texture == core->image_atlas_texture && bytes_per_pixel == 4u &&
        fs_image_atlas_shadow_bounds_ok(core, layer, x, y, width, height)) {
        const size_t layer_stride = (size_t)core->image_atlas_width * (size_t)core->image_atlas_height * 4u;
        uint8_t* shadow_base = core->image_atlas_shadow_rgba + (size_t)layer * layer_stride;
        const size_t shadow_row = (size_t)core->image_atlas_width * 4u;
        for (uint32_t row = 0u; row < height; ++row) {
            memcpy(
                shadow_base + ((size_t)(y + row) * shadow_row) + (size_t)x * 4u,
                pixels + (size_t)row * src_row,
                src_row
            );
        }
    }

    FS_PendingTextureUpload* uploads = (FS_PendingTextureUpload*)core->pending_uploads;
    uploads[core->pending_upload_count++] = (FS_PendingTextureUpload){
        .texture = texture,
        .mip_level = 0u,
        .layer = layer,
        .x = x,
        .y = y,
        .width = width,
        .height = height,
        .padded_row_bytes = padded_row,
        .src_offset = upload_offset
    };
    core->upload_staging_used = required_size;
    return true;
}

static bool fs_flush_pending_texture_uploads(FS_Core* core, WGPUCommandEncoder encoder) {
    if (!core) {
        return false;
    }
    if (core->pending_upload_count == 0u) {
        core->upload_staging_used = 0u;
        return true;
    }
    if (!encoder || !core->queue || !core->device || !core->upload_staging_cpu) {
        return false;
    }
    if (core->upload_staging_used > core->upload_staging_gpu_capacity) {
        size_t new_size = core->upload_staging_gpu_capacity ? core->upload_staging_gpu_capacity : 4096u;
        while (new_size < core->upload_staging_used) {
            if (new_size > (SIZE_MAX / 2u)) {
                new_size = core->upload_staging_used;
                break;
            }
            new_size *= 2u;
        }
        WGPUBuffer new_buf = fs_create_buffer(
            core->device,
            "FS Upload Staging Buffer",
            WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
            new_size
        );
        if (!new_buf) {
            return false;
        }
        if (core->upload_staging_gpu) {
            wgpuBufferRelease(core->upload_staging_gpu);
        }
        core->upload_staging_gpu = new_buf;
        core->upload_staging_gpu_capacity = new_size;
    }
    if (!core->upload_staging_gpu) {
        return false;
    }

    wgpuQueueWriteBuffer(core->queue, core->upload_staging_gpu, 0u, core->upload_staging_cpu, core->upload_staging_used);

    FS_PendingTextureUpload* uploads = (FS_PendingTextureUpload*)core->pending_uploads;
    for (size_t i = 0u; i < core->pending_upload_count; ++i) {
        const FS_PendingTextureUpload* upload = &uploads[i];
        WGPUTexelCopyBufferInfo src = {
            .layout = {
                .offset = upload->src_offset,
                .bytesPerRow = upload->padded_row_bytes,
                .rowsPerImage = upload->height
            },
            .buffer = core->upload_staging_gpu
        };
        WGPUTexelCopyTextureInfo dst = {
            .texture = upload->texture,
            .mipLevel = upload->mip_level,
            .origin = {upload->x, upload->y, upload->layer},
            .aspect = WGPUTextureAspect_All
        };
        WGPUExtent3D extent = {upload->width, upload->height, 1u};
        wgpuCommandEncoderCopyBufferToTexture(encoder, &src, &dst, &extent);
    }

    core->pending_upload_count = 0u;
    core->upload_staging_used = 0u;
    return true;
}

static bool fs_ensure_clip_edge_gpu_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    const size_t min_count = 256u;
    size_t target = required > min_count ? required : min_count;
    size_t needed = target * sizeof(FS_ClipEdgeGPU);
    if (needed <= core->clip_edge_buffer_size && needed <= core->clip_edge_local_buffer_size &&
        core->clip_edge_buffer && core->clip_edge_local_buffer) {
        return true;
    }
    size_t new_size = core->clip_edge_buffer_size > core->clip_edge_local_buffer_size
                          ? core->clip_edge_buffer_size
                          : core->clip_edge_local_buffer_size;
    if (new_size < sizeof(FS_ClipEdgeGPU) * min_count) {
        new_size = sizeof(FS_ClipEdgeGPU) * min_count;
    }
    while (new_size < needed) {
        if (new_size > (SIZE_MAX / 2u)) {
            new_size = needed;
            break;
        }
        new_size *= 2u;
    }
    WGPUBuffer local_buf = fs_create_buffer(
        core->device,
        "FS Clip Edge Local Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        new_size
    );
    if (!local_buf) {
        return false;
    }
    WGPUBuffer device_buf = fs_create_buffer(
        core->device,
        "FS Clip Edge Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        new_size
    );
    if (!device_buf) {
        wgpuBufferRelease(local_buf);
        return false;
    }
    if (core->clip_edge_local_buffer) {
        wgpuBufferRelease(core->clip_edge_local_buffer);
    }
    if (core->clip_edge_buffer) {
        wgpuBufferRelease(core->clip_edge_buffer);
    }
    core->clip_edge_local_buffer = local_buf;
    core->clip_edge_local_buffer_size = new_size;
    core->clip_edge_buffer = device_buf;
    core->clip_edge_buffer_size = new_size;
    if (core->clip_compute_bgl && core->clip_job_buffer && core->clip_dispatch_uniform_buffer) {
        (void)fs_recreate_clip_compute_bind_group(core);
    }
    return true;
}

static bool fs_ensure_clip_job_gpu_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    const size_t min_count = 64u;
    size_t target = required > min_count ? required : min_count;
    size_t needed = target * sizeof(FS_ClipJobGPU);
    if (needed <= core->clip_job_buffer_size && core->clip_job_buffer) {
        return true;
    }
    size_t new_size = core->clip_job_buffer_size ? core->clip_job_buffer_size : sizeof(FS_ClipJobGPU) * min_count;
    while (new_size < needed) {
        if (new_size > (SIZE_MAX / 2u)) {
            new_size = needed;
            break;
        }
        new_size *= 2u;
    }
    WGPUBuffer new_buf = fs_create_buffer(
        core->device,
        "FS Clip Job Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        new_size
    );
    if (!new_buf) {
        return false;
    }
    if (core->clip_job_buffer) {
        wgpuBufferRelease(core->clip_job_buffer);
    }
    core->clip_job_buffer = new_buf;
    core->clip_job_buffer_size = new_size;
    if (core->clip_compute_bgl && core->clip_edge_buffer && core->clip_dispatch_uniform_buffer) {
        (void)fs_recreate_clip_compute_bind_group(core);
    }
    return true;
}

static bool fs_ensure_clip_job_transform_gpu_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    const size_t min_count = 64u;
    size_t target = required > min_count ? required : min_count;
    size_t needed = target * sizeof(FS_ClipJobTransformGPU);
    if (needed <= core->clip_job_xform_buffer_size && core->clip_job_xform_buffer) {
        return true;
    }
    size_t new_size = core->clip_job_xform_buffer_size ? core->clip_job_xform_buffer_size : sizeof(FS_ClipJobTransformGPU) * min_count;
    while (new_size < needed) {
        if (new_size > (SIZE_MAX / 2u)) {
            new_size = needed;
            break;
        }
        new_size *= 2u;
    }
    WGPUBuffer new_buf = fs_create_buffer(
        core->device,
        "FS Clip Job Transform Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        new_size
    );
    if (!new_buf) {
        return false;
    }
    if (core->clip_job_xform_buffer) {
        wgpuBufferRelease(core->clip_job_xform_buffer);
    }
    core->clip_job_xform_buffer = new_buf;
    core->clip_job_xform_buffer_size = new_size;
    return true;
}

static bool fs_ensure_canvas_shadow(FS_Core* core) {
    if (!core) {
        return false;
    }
    const size_t needed = (size_t)core->width * (size_t)core->height * 4u;
    if (needed == 0u) {
        free(core->canvas_shadow_rgba);
        core->canvas_shadow_rgba = NULL;
        core->canvas_shadow_size = 0u;
        return true;
    }
    if (needed <= core->canvas_shadow_size && core->canvas_shadow_rgba) {
        return true;
    }
    uint8_t* grown = (uint8_t*)realloc(core->canvas_shadow_rgba, needed);
    if (!grown) {
        return false;
    }
    if (needed > core->canvas_shadow_size) {
        memset(grown + core->canvas_shadow_size, 0, needed - core->canvas_shadow_size);
    }
    core->canvas_shadow_rgba = grown;
    core->canvas_shadow_size = needed;
    return true;
}

static bool fs_image_atlas_shadow_read_rgba(
    const FS_Core* core,
    uint32_t layer,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    uint8_t* out_rgba_pixels
) {
    if (!out_rgba_pixels || !fs_image_atlas_shadow_bounds_ok(core, layer, x, y, width, height)) {
        return false;
    }
    const size_t atlas_row = (size_t)core->image_atlas_width * 4u;
    const size_t out_row = (size_t)width * 4u;
    const size_t layer_stride = (size_t)core->image_atlas_width * (size_t)core->image_atlas_height * 4u;
    const uint8_t* src_base = core->image_atlas_shadow_rgba + (size_t)layer * layer_stride;
    for (uint32_t row = 0u; row < height; ++row) {
        memcpy(
            out_rgba_pixels + (size_t)row * out_row,
            src_base + ((size_t)(y + row) * atlas_row) + (size_t)x * 4u,
            out_row
        );
    }
    return true;
}

static bool fs_ensure_canvas_image_data_handle(FS_Core* core, uint32_t width, uint32_t height, FS_ImageHandle* out_handle) {
    if (!core || width == 0u || height == 0u) {
        return false;
    }
    if (!core->canvas_image_data_handle_valid || core->canvas_image_data_handle.width != width ||
        core->canvas_image_data_handle.height != height ||
        core->canvas_image_data_handle.layer >= core->image_atlas_layers ||
        core->canvas_image_data_handle.generation !=
            core->image_atlas_generation[core->canvas_image_data_handle.layer]) {
        FS_ImageHandle handle;
        const size_t zero_size = (size_t)width * (size_t)height * 4u;
        uint8_t* zero = (uint8_t*)calloc(1u, zero_size);
        if (!zero) {
            return false;
        }
        bool ok = fs_core_upload_image_rgba8(core, zero, width, height, &handle);
        free(zero);
        if (!ok) {
            return false;
        }
        core->canvas_image_data_handle = handle;
        core->canvas_image_data_handle_valid = 1u;
    }
    if (out_handle) {
        *out_handle = core->canvas_image_data_handle;
    }
    return true;
}

static bool fs_encode_canvas_readback_copy(FS_Core* core, WGPUCommandEncoder encoder, WGPUTexture target_texture) {
    if (!core || !encoder || !target_texture || core->width == 0u || core->height == 0u) {
        return false;
    }
    const uint32_t row_bytes = core->width * 4u;
    const uint32_t padded_row = fs_align_up_u32(row_bytes, 256u);
    const size_t needed = (size_t)padded_row * (size_t)core->height;
    if (needed == 0u) {
        return false;
    }
    if (!core->canvas_readback_buffer || core->canvas_readback_buffer_size < needed ||
        core->canvas_readback_width != core->width || core->canvas_readback_height != core->height) {
        if (core->canvas_readback_buffer) {
            if (core->canvas_readback_mapped) {
                wgpuBufferUnmap(core->canvas_readback_buffer);
                core->canvas_readback_mapped = 0u;
            }
            wgpuBufferRelease(core->canvas_readback_buffer);
        }
        core->canvas_readback_buffer = fs_create_buffer(
            core->device,
            "FS Canvas Readback Buffer",
            WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
            needed
        );
        if (!core->canvas_readback_buffer) {
            core->canvas_readback_buffer_size = 0u;
            return false;
        }
        core->canvas_readback_buffer_size = needed;
        core->canvas_readback_width = core->width;
        core->canvas_readback_height = core->height;
    }
    core->canvas_readback_row_bytes = row_bytes;
    core->canvas_readback_padded_row_bytes = padded_row;
    core->canvas_readback_serial += 1u;
    core->canvas_readback_submission = 0u;
    core->canvas_readback_submission_valid = 0u;

    WGPUTexelCopyTextureInfo src = {
        .texture = target_texture,
        .mipLevel = 0u,
        .origin = {0u, 0u, 0u},
        .aspect = WGPUTextureAspect_All
    };
    WGPUTexelCopyBufferInfo dst = {
        .layout = {
            .offset = 0u,
            .bytesPerRow = padded_row,
            .rowsPerImage = core->height
        },
        .buffer = core->canvas_readback_buffer
    };
    WGPUExtent3D extent = {core->width, core->height, 1u};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);
    return true;
}

static bool fs_target_format_is_bgra(WGPUTextureFormat format) {
    return format == WGPUTextureFormat_BGRA8Unorm || format == WGPUTextureFormat_BGRA8UnormSrgb;
}

static void fs_canvas_readback_map_callback(
    WGPUMapAsyncStatus status,
    WGPUStringView message,
    void* userdata1,
    void* userdata2
) {
    (void)message;
    (void)userdata2;
    FS_MapReadbackContext* ctx = (FS_MapReadbackContext*)userdata1;
    if (!ctx) {
        return;
    }
    ctx->success = (status == WGPUMapAsyncStatus_Success) ? 1u : 0u;
    ctx->done = 1u;
}

static bool fs_refresh_canvas_shadow_from_readback(FS_Core* core) {
    if (!core || !core->canvas_readback_buffer || core->canvas_readback_serial == 0u) {
        return false;
    }
    if (core->canvas_shadow_serial == core->canvas_readback_serial) {
        return true;
    }
    if (!fs_ensure_canvas_shadow(core)) {
        return false;
    }
    if (!core->canvas_shadow_rgba || core->canvas_shadow_size == 0u) {
        return false;
    }

    FS_MapReadbackContext map_ctx = {0u, 0u};
    WGPUBufferMapCallbackInfo map_info = {
        .nextInChain = NULL,
        .mode = WGPUCallbackMode_AllowSpontaneous,
        .callback = fs_canvas_readback_map_callback,
        .userdata1 = &map_ctx,
        .userdata2 = NULL
    };
    wgpuBufferMapAsync(
        core->canvas_readback_buffer,
        WGPUMapMode_Read,
        0u,
        core->canvas_readback_buffer_size,
        map_info
    );
    for (uint32_t i = 0u; i < 8u && map_ctx.done == 0u; ++i) {
        (void)wgpuDevicePoll(core->device, true, NULL);
    }
    if (map_ctx.done == 0u || map_ctx.success == 0u) {
        return false;
    }

    const uint8_t* mapped = (const uint8_t*)wgpuBufferGetConstMappedRange(
        core->canvas_readback_buffer,
        0u,
        core->canvas_readback_buffer_size
    );
    if (!mapped) {
        return false;
    }

    const bool is_bgra = fs_target_format_is_bgra(core->target_format);
    const uint32_t copy_width = core->canvas_readback_width;
    const uint32_t copy_height = core->canvas_readback_height;
    for (uint32_t row = 0u; row < copy_height; ++row) {
        const uint8_t* src = mapped + (size_t)row * (size_t)core->canvas_readback_padded_row_bytes;
        uint8_t* dst = core->canvas_shadow_rgba + (size_t)row * (size_t)core->canvas_readback_row_bytes;
        if (!is_bgra) {
            memcpy(dst, src, (size_t)copy_width * 4u);
        } else {
            for (uint32_t x = 0u; x < copy_width; ++x) {
                const uint8_t* s = src + (size_t)x * 4u;
                uint8_t* d = dst + (size_t)x * 4u;
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            }
        }
    }
    wgpuBufferUnmap(core->canvas_readback_buffer);
    core->canvas_shadow_serial = core->canvas_readback_serial;
    return true;
}

static bool fs_upload_glyph_with_mips(
    FS_Core* core,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    const uint8_t* pixels,
    uint32_t pixel_format,
    float sdf_onedge,
    float sdf_pixel_dist_scale
) {
    (void)sdf_pixel_dist_scale;
    if (!core || !pixels || width == 0u || height == 0u) {
        return false;
    }
    if (pixel_format == FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8) {
        return fs_queue_write_texture_2d(core, core->glyph_atlas_texture, 0u, x, y, width, height, pixels, 4u);
    }
    // SDF glyphs are R8 but atlas is RGBA8Unorm. Expand to RGBA.
    const size_t rgba_size = (size_t)width * (size_t)height * 4u;
    uint8_t* rgba = (uint8_t*)malloc(rgba_size);
    if (!rgba) {
        return false;
    }
    for (size_t i = 0u; i < (size_t)width * (size_t)height; ++i) {
        rgba[i * 4u + 0u] = pixels[i];
        rgba[i * 4u + 1u] = pixels[i];
        rgba[i * 4u + 2u] = pixels[i];
        rgba[i * 4u + 3u] = pixels[i];  // Shader reads sampled.a for SDF value
    }
    (void)sdf_onedge;
    bool ok = fs_queue_write_texture_2d(core, core->glyph_atlas_texture, 0u, x, y, width, height, rgba, 4u);
    free(rgba);
    return ok;
}


static void fs_transform_rect_to_aabb(
    const FS_Transform2D* t,
    float x,
    float y,
    float w,
    float h,
    float* out_x,
    float* out_y,
    float* out_w,
    float* out_h
) {
    if (!out_x || !out_y || !out_w || !out_h) {
        return;
    }
    if (!t) {
        *out_x = x;
        *out_y = y;
        *out_w = w;
        *out_h = h;
        return;
    }
    float px[4];
    float py[4];
    fs_transform_apply_point(t, x, y, &px[0], &py[0]);
    fs_transform_apply_point(t, x + w, y, &px[1], &py[1]);
    fs_transform_apply_point(t, x, y + h, &px[2], &py[2]);
    fs_transform_apply_point(t, x + w, y + h, &px[3], &py[3]);
    float min_x = px[0];
    float min_y = py[0];
    float max_x = px[0];
    float max_y = py[0];
    for (uint32_t i = 1u; i < 4u; ++i) {
        if (px[i] < min_x) {
            min_x = px[i];
        }
        if (py[i] < min_y) {
            min_y = py[i];
        }
        if (px[i] > max_x) {
            max_x = px[i];
        }
        if (py[i] > max_y) {
            max_y = py[i];
        }
    }
    *out_x = min_x;
    *out_y = min_y;
    *out_w = max_x - min_x;
    *out_h = max_y - min_y;
}

static void fs_command_set_oriented_quad_from_rect(FS_Command* cmd, const FS_Transform2D* t, float x, float y, float w, float h) {
    if (!cmd) {
        return;
    }
    float q0x = x;
    float q0y = y;
    float q1x = x + w;
    float q1y = y;
    float q2x = x;
    float q2y = y + h;
    if (t) {
        fs_transform_apply_point(t, x, y, &q0x, &q0y);
        fs_transform_apply_point(t, x + w, y, &q1x, &q1y);
        fs_transform_apply_point(t, x, y + h, &q2x, &q2y);
    }
    cmd->quad0[0] = q0x;
    cmd->quad0[1] = q0y;
    cmd->quad0[2] = q1x - q0x;
    cmd->quad0[3] = q1y - q0y;
    cmd->quad1[0] = q2x - q0x;
    cmd->quad1[1] = q2y - q0y;
    cmd->quad1[2] = 0.0f;
    cmd->quad1[3] = 0.0f;
}

static float fs_cross2(const FS_Point2* a, const FS_Point2* b, const FS_Point2* c);
static bool fs_polygon_compact_in_place(FS_Point2* points, uint32_t* io_count) {
    if (!points || !io_count || *io_count < 3u) {
        return false;
    }
    const float eps_dist2 = 1e-8f;
    const float eps_cross = 1e-5f;
    uint32_t count = *io_count;

    // Pass 1: remove adjacent duplicate points.
    uint32_t write = 0u;
    for (uint32_t i = 0u; i < count; ++i) {
        if (write == 0u) {
            points[write++] = points[i];
            continue;
        }
        const float dx = points[i].x - points[write - 1u].x;
        const float dy = points[i].y - points[write - 1u].y;
        if (dx * dx + dy * dy <= eps_dist2) {
            continue;
        }
        points[write++] = points[i];
    }
    count = write;
    if (count < 3u) {
        *io_count = count;
        return false;
    }
    {
        const float dx = points[0].x - points[count - 1u].x;
        const float dy = points[0].y - points[count - 1u].y;
        if (dx * dx + dy * dy <= eps_dist2) {
            count -= 1u;
        }
    }
    if (count < 3u) {
        *io_count = count;
        return false;
    }

    // Pass 2: iteratively remove spikes and near-collinear vertices.
    bool changed = true;
    uint32_t guard = 0u;
    const uint32_t guard_max = count * 6u + 32u;
    while (changed && count >= 3u && guard < guard_max) {
        changed = false;
        for (uint32_t i = 0u; i < count; ++i) {
            const uint32_t ip = (i + count - 1u) % count;
            const uint32_t in = (i + 1u) % count;
            const FS_Point2* p = &points[ip];
            const FS_Point2* c = &points[i];
            const FS_Point2* n = &points[in];

            const float dxpn = n->x - p->x;
            const float dypn = n->y - p->y;
            if (dxpn * dxpn + dypn * dypn <= eps_dist2) {
                if (i + 1u < count) {
                    memmove(&points[i], &points[i + 1u], (size_t)(count - i - 1u) * sizeof(FS_Point2));
                }
                count -= 1u;
                changed = true;
                break;
            }

            const float cross = fabsf(fs_cross2(p, c, n));
            if (cross <= eps_cross) {
                if (i + 1u < count) {
                    memmove(&points[i], &points[i + 1u], (size_t)(count - i - 1u) * sizeof(FS_Point2));
                }
                count -= 1u;
                changed = true;
                break;
            }
        }
        guard += 1u;
    }

    *io_count = count;
    return count >= 3u;
}

static uint64_t fs_hash64_u32(uint64_t hash, uint32_t value) {
    return fs_hash64_mix(hash, (uint64_t)value);
}

static uint64_t fs_hash64_mix(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

static uint64_t fs_hash64_f32(uint64_t hash, float value) {
    union { float f; uint32_t u; } conv;
    conv.f = value;
    return fs_hash64_u32(hash, conv.u);
}

static bool fs_ensure_clip_edge_cpu_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    if (required <= core->clip_edge_capacity) {
        return true;
    }
    size_t new_cap = core->clip_edge_capacity ? core->clip_edge_capacity : 256u;
    while (new_cap < required) {
        if (new_cap > (SIZE_MAX / 2u)) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_ClipEdgeGPU* grown = (FS_ClipEdgeGPU*)realloc(core->clip_edge_cpu, new_cap * sizeof(FS_ClipEdgeGPU));
    if (!grown) {
        return false;
    }
    core->clip_edge_cpu = grown;
    core->clip_edge_capacity = new_cap;
    return true;
}

static bool fs_ensure_clip_job_cpu_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    if (required <= core->clip_job_capacity && core->clip_job_cpu && core->clip_job_xform_cpu) {
        return true;
    }
    size_t new_cap = core->clip_job_capacity ? core->clip_job_capacity : 64u;
    while (new_cap < required) {
        if (new_cap > (SIZE_MAX / 2u)) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }

    FS_ClipJobGPU* grown_jobs = (FS_ClipJobGPU*)malloc(new_cap * sizeof(FS_ClipJobGPU));
    if (!grown_jobs) {
        return false;
    }
    FS_ClipJobTransformGPU* grown_xforms =
        (FS_ClipJobTransformGPU*)malloc(new_cap * sizeof(FS_ClipJobTransformGPU));
    if (!grown_xforms) {
        free(grown_jobs);
        return false;
    }

    size_t copy_count = core->clip_job_count;
    if (copy_count > new_cap) {
        copy_count = new_cap;
    }
    if (core->clip_job_cpu && copy_count > 0u) {
        memcpy(grown_jobs, core->clip_job_cpu, copy_count * sizeof(FS_ClipJobGPU));
    }
    if (core->clip_job_xform_cpu && copy_count > 0u) {
        memcpy(grown_xforms, core->clip_job_xform_cpu, copy_count * sizeof(FS_ClipJobTransformGPU));
    } else if (copy_count > 0u) {
        const FS_ClipJobTransformGPU identity = {
            .xform0 = {1.0f, 0.0f, 0.0f, 1.0f},
            .xform1 = {0.0f, 0.0f, 0.0f, 0.0f}
        };
        for (size_t i = 0u; i < copy_count; ++i) {
            grown_xforms[i] = identity;
        }
    }

    free(core->clip_job_cpu);
    free(core->clip_job_xform_cpu);
    core->clip_job_cpu = grown_jobs;
    core->clip_job_xform_cpu = grown_xforms;
    core->clip_job_capacity = new_cap;
    return true;
}

static void fs_state_reset(FS_InternalState* st) {
    if (!st) {
        return;
    }
    fs_state_stack_clear(st);
    st->path_count = 0u;
    st->glyph_count = 0u;
}

static bool fs_ensure_state_stack_capacity(FS_InternalState* st, size_t required) {
    if (!st) {
        return false;
    }
    if (required <= st->state_stack_capacity) {
        return true;
    }
    size_t new_cap = st->state_stack_capacity ? st->state_stack_capacity : 16u;
    while (new_cap < required) {
        if (new_cap > (SIZE_MAX / 2u)) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_StateSnapshot* grown = (FS_StateSnapshot*)realloc(st->state_stack, new_cap * sizeof(FS_StateSnapshot));
    if (!grown) {
        return false;
    }
    st->state_stack = grown;
    st->state_stack_capacity = (uint32_t)new_cap;
    return true;
}

static void fs_style_snapshot_dispose(FS_StyleSnapshot* snap) {
    if (!snap) {
        return;
    }
    free(snap->dash_segments);
    snap->dash_segments = NULL;
    snap->dash_count = 0u;
    snap->dash_offset = 0.0f;
}

static bool fs_style_snapshot_capture(FS_StyleSnapshot* dst, const FS_InternalState* st) {
    if (!dst || !st) {
        return false;
    }
    memset(dst, 0, sizeof(*dst));
    dst->line_width = st->style_line_width;
    dst->miter_limit = st->style_miter_limit;
    dst->line_cap = st->style_line_cap;
    dst->line_join = st->style_line_join;
    dst->fill_rule = st->style_fill_rule;
    dst->composite_op = st->style_composite_op;
    dst->text_align = st->style_text_align;
    dst->text_baseline = st->style_text_baseline;
    dst->text_direction = st->style_text_direction;
    dst->font_kerning = st->style_font_kerning;
    dst->text_rendering = st->style_text_rendering;
    dst->font_stretch = st->style_font_stretch;
    dst->font_variant_caps = st->style_font_variant_caps;
    dst->letter_spacing = st->style_letter_spacing;
    dst->word_spacing = st->style_word_spacing;
    dst->image_smoothing_enabled = st->style_image_smoothing_enabled;
    dst->image_smoothing_quality = st->style_image_smoothing_quality;
    dst->global_alpha = st->style_global_alpha;
    dst->fill_color_rgba8 = st->style_fill_color_rgba8;
    dst->stroke_color_rgba8 = st->style_stroke_color_rgba8;
    dst->fill_paint_type = st->style_fill_paint_type;
    dst->stroke_paint_type = st->style_stroke_paint_type;
    dst->fill_linear_gradient = st->style_fill_linear_gradient;
    dst->stroke_linear_gradient = st->style_stroke_linear_gradient;
    dst->fill_radial_gradient = st->style_fill_radial_gradient;
    dst->stroke_radial_gradient = st->style_stroke_radial_gradient;
    dst->fill_conic_gradient = st->style_fill_conic_gradient;
    dst->stroke_conic_gradient = st->style_stroke_conic_gradient;
    dst->fill_pattern = st->style_fill_pattern;
    dst->stroke_pattern = st->style_stroke_pattern;
    dst->shadow_color_rgba8 = st->style_shadow_color_rgba8;
    dst->shadow_blur = st->style_shadow_blur;
    dst->shadow_offset_x = st->style_shadow_offset_x;
    dst->shadow_offset_y = st->style_shadow_offset_y;
    dst->dash_offset = st->style_dash_offset;
    if (st->style_dash_count > 0u) {
        if (!st->style_dash_segments) {
            return false;
        }
        dst->dash_segments = (float*)malloc((size_t)st->style_dash_count * sizeof(float));
        if (!dst->dash_segments) {
            return false;
        }
        memcpy(dst->dash_segments, st->style_dash_segments, (size_t)st->style_dash_count * sizeof(float));
        dst->dash_count = st->style_dash_count;
    }
    return true;
}

static void fs_style_snapshot_apply(FS_InternalState* st, FS_StyleSnapshot* src) {
    if (!st || !src) {
        return;
    }
    free(st->style_dash_segments);
    st->style_line_width = src->line_width;
    st->style_miter_limit = src->miter_limit;
    st->style_line_cap = src->line_cap;
    st->style_line_join = src->line_join;
    st->style_fill_rule = src->fill_rule;
    st->style_composite_op = src->composite_op;
    st->style_text_align = src->text_align;
    st->style_text_baseline = src->text_baseline;
    st->style_text_direction = src->text_direction;
    st->style_font_kerning = src->font_kerning;
    st->style_text_rendering = src->text_rendering;
    st->style_font_stretch = src->font_stretch;
    st->style_font_variant_caps = src->font_variant_caps;
    st->style_letter_spacing = src->letter_spacing;
    st->style_word_spacing = src->word_spacing;
    st->style_image_smoothing_enabled = src->image_smoothing_enabled;
    st->style_image_smoothing_quality = src->image_smoothing_quality;
    st->style_global_alpha = src->global_alpha;
    st->style_fill_color_rgba8 = src->fill_color_rgba8;
    st->style_stroke_color_rgba8 = src->stroke_color_rgba8;
    st->style_fill_paint_type = src->fill_paint_type;
    st->style_stroke_paint_type = src->stroke_paint_type;
    st->style_fill_linear_gradient = src->fill_linear_gradient;
    st->style_stroke_linear_gradient = src->stroke_linear_gradient;
    st->style_fill_radial_gradient = src->fill_radial_gradient;
    st->style_stroke_radial_gradient = src->stroke_radial_gradient;
    st->style_fill_conic_gradient = src->fill_conic_gradient;
    st->style_stroke_conic_gradient = src->stroke_conic_gradient;
    st->style_fill_pattern = src->fill_pattern;
    st->style_stroke_pattern = src->stroke_pattern;
    st->style_shadow_color_rgba8 = src->shadow_color_rgba8;
    st->style_shadow_blur = src->shadow_blur;
    st->style_shadow_offset_x = src->shadow_offset_x;
    st->style_shadow_offset_y = src->shadow_offset_y;
    st->style_dash_segments = src->dash_segments;
    st->style_dash_count = src->dash_count;
    st->style_dash_offset = src->dash_offset;
    src->dash_segments = NULL;
    src->dash_count = 0u;
    src->dash_offset = 0.0f;
}

static void fs_state_snapshot_dispose(FS_StateSnapshot* snap) {
    if (!snap) {
        return;
    }
    fs_style_snapshot_dispose(&snap->style);
}

static void fs_state_stack_clear(FS_InternalState* st) {
    if (!st) {
        return;
    }
    for (uint32_t i = 0u; i < st->state_stack_count; ++i) {
        fs_state_snapshot_dispose(&st->state_stack[i]);
    }
    st->state_stack_count = 0u;
}

static void fs_style_reset_state(FS_InternalState* st) {
    if (!st) {
        return;
    }
    st->style_line_width = 1.0f;
    st->style_miter_limit = 10.0f;
    st->style_line_cap = (uint8_t)FS_LINE_CAP_ROUND;
    st->style_line_join = (uint8_t)FS_LINE_JOIN_MITER;
    st->style_fill_rule = (uint8_t)FS_FILL_RULE_NONZERO;
    st->style_composite_op = (uint8_t)FS_GLOBAL_COMPOSITE_SOURCE_OVER;
    st->style_text_align = (uint8_t)FS_TEXT_ALIGN_START;
    st->style_text_baseline = (uint8_t)FS_TEXT_BASELINE_ALPHABETIC;
    st->style_text_direction = (uint8_t)FS_TEXT_DIRECTION_LTR;
    st->style_font_kerning = (uint8_t)FS_FONT_KERNING_AUTO;
    st->style_text_rendering = (uint8_t)FS_TEXT_RENDERING_AUTO;
    st->style_font_stretch = (uint8_t)FS_FONT_STRETCH_NORMAL;
    st->style_font_variant_caps = (uint8_t)FS_FONT_VARIANT_CAPS_NORMAL;
    st->style_letter_spacing = 0.0f;
    st->style_word_spacing = 0.0f;
    st->style_image_smoothing_enabled = 1u;
    st->style_image_smoothing_quality = (uint8_t)FS_IMAGE_SMOOTHING_QUALITY_LOW;
    st->style_global_alpha = 1.0f;
    st->style_fill_color_rgba8 = 0xFF000000u;
    st->style_stroke_color_rgba8 = 0xFF000000u;
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_SOLID;
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_SOLID;
    st->style_fill_linear_gradient.stop_count = 0u;
    st->style_stroke_linear_gradient.stop_count = 0u;
    st->style_fill_radial_gradient.stop_count = 0u;
    st->style_stroke_radial_gradient.stop_count = 0u;
    st->style_fill_conic_gradient.stop_count = 0u;
    st->style_stroke_conic_gradient.stop_count = 0u;
    memset(&st->style_fill_pattern, 0, sizeof(st->style_fill_pattern));
    memset(&st->style_stroke_pattern, 0, sizeof(st->style_stroke_pattern));
    st->style_fill_pattern.repeat_mode = (uint8_t)FS_PATTERN_REPEAT;
    st->style_stroke_pattern.repeat_mode = (uint8_t)FS_PATTERN_REPEAT;
    fs_affine_set_identity_2d(st->style_fill_pattern.xform);
    fs_affine_set_identity_2d(st->style_fill_pattern.inv_xform);
    st->style_fill_pattern.inv_valid = 1u;
    fs_affine_set_identity_2d(st->style_stroke_pattern.xform);
    fs_affine_set_identity_2d(st->style_stroke_pattern.inv_xform);
    st->style_stroke_pattern.inv_valid = 1u;
    st->style_shadow_color_rgba8 = 0u;
    st->style_shadow_blur = 0.0f;
    st->style_shadow_offset_x = 0.0f;
    st->style_shadow_offset_y = 0.0f;
    free(st->style_dash_segments);
    st->style_dash_segments = NULL;
    st->style_dash_count = 0u;
    st->style_dash_offset = 0.0f;
}

static void fs_clip_reset_state(FS_InternalState* st) {
    if (!st) {
        return;
    }
    st->clip_enabled = 0u;
    st->clip_path_enabled = 0u;
    st->clip_path_layer = 0u;
    st->clip_min_x = 0.0f;
    st->clip_min_y = 0.0f;
    st->clip_max_x = 0.0f;
    st->clip_max_y = 0.0f;
}

static float fs_style_resolve_line_width(const FS_InternalState* st, float width) {
    if (width > 0.0f) {
        return width;
    }
    if (!st || st->style_line_width <= 0.0f) {
        return 1.0f;
    }
    return st->style_line_width;
}

static bool fs_style_has_dash(const FS_InternalState* st) {
    return st && st->style_dash_segments && st->style_dash_count > 0u;
}

static void fs_resolve_text_vertical_metrics(
    const FS_InternalState* st,
    float font_size_px,
    float* out_em_ascent,
    float* out_em_descent,
    float* out_line_height
) {
    float em_ascent = (font_size_px > 0.0f) ? (font_size_px * 0.8f) : 0.0f;
    float em_descent = (font_size_px > 0.0f) ? (font_size_px * 0.2f) : 0.0f;
    float line_height = (font_size_px > 0.0f) ? (font_size_px * 1.25f) : 0.0f;
    if (st && st->font_backend && st->font_backend->get_vertical_metrics && st->font_count > 0u && st->fonts[0]) {
        FS_FontVerticalMetrics vm = {0};
        if (st->font_backend->get_vertical_metrics(st->fonts[0], font_size_px, &vm)) {
            if (isfinite(vm.ascent) && vm.ascent > 0.0f) {
                em_ascent = vm.ascent;
            }
            if (isfinite(vm.descent) && vm.descent >= 0.0f) {
                em_descent = vm.descent;
            }
            if (isfinite(vm.line_height) && vm.line_height > 0.0f) {
                line_height = vm.line_height;
            }
        }
    }
    if (!isfinite(line_height) || line_height <= 0.0f) {
        line_height = em_ascent + em_descent;
    } else if (line_height < em_ascent + em_descent) {
        line_height = em_ascent + em_descent;
    }
    if (out_em_ascent) {
        *out_em_ascent = em_ascent;
    }
    if (out_em_descent) {
        *out_em_descent = em_descent;
    }
    if (out_line_height) {
        *out_line_height = line_height;
    }
}

static float fs_text_align_offset(const FS_InternalState* st, const FS_TextMetrics* metrics) {
    if (!st || !metrics) {
        return 0.0f;
    }
    const bool rtl = st->style_text_direction == (uint8_t)FS_TEXT_DIRECTION_RTL;
    switch ((FS_TextAlign)st->style_text_align) {
        case FS_TEXT_ALIGN_CENTER:
            return -0.5f * metrics->width;
        case FS_TEXT_ALIGN_RIGHT:
            return -metrics->width;
        case FS_TEXT_ALIGN_END:
            return rtl ? 0.0f : -metrics->width;
        case FS_TEXT_ALIGN_START:
            return rtl ? -metrics->width : 0.0f;
        case FS_TEXT_ALIGN_LEFT:
        default:
            return 0.0f;
    }
}

static bool fs_is_word_spacing_codepoint(uint32_t cp) {
    switch (cp) {
        case 0x0009u:
        case 0x000Bu:
        case 0x000Cu:
        case 0x0020u:
        case 0x00A0u:
        case 0x1680u:
        case 0x2000u:
        case 0x2001u:
        case 0x2002u:
        case 0x2003u:
        case 0x2004u:
        case 0x2005u:
        case 0x2006u:
        case 0x2007u:
        case 0x2008u:
        case 0x2009u:
        case 0x200Au:
        case 0x202Fu:
        case 0x205Fu:
        case 0x3000u:
            return true;
        default:
            return false;
    }
}

static float fs_text_baseline_offset(const FS_InternalState* st, const FS_TextMetrics* metrics, float font_size_px) {
    if (!st || !metrics || font_size_px <= 0.0f) {
        return 0.0f;
    }
    const float em_ascent = (metrics->em_height_ascent > 0.0f) ? metrics->em_height_ascent : (font_size_px * 0.8f);
    const float em_descent = (metrics->em_height_descent > 0.0f) ? metrics->em_height_descent : (font_size_px * 0.2f);
    const float em_middle = 0.5f * (em_ascent - em_descent);
    switch ((FS_TextBaseline)st->style_text_baseline) {
        case FS_TEXT_BASELINE_TOP:
            return em_ascent;
        case FS_TEXT_BASELINE_HANGING:
            return em_ascent * 0.8f;
        case FS_TEXT_BASELINE_MIDDLE:
            return em_middle;
        case FS_TEXT_BASELINE_IDEOGRAPHIC:
            return -em_descent;
        case FS_TEXT_BASELINE_BOTTOM:
            return -em_descent;
        case FS_TEXT_BASELINE_ALPHABETIC:
        default:
            return 0.0f;
    }
}

static bool fs_is_text_kerning_enabled(const FS_InternalState* st) {
    if (!st) {
        return true;
    }
    const FS_FontKerning kerning = (FS_FontKerning)st->style_font_kerning;
    if (kerning == FS_FONT_KERNING_NONE) {
        return false;
    }
    const FS_TextRendering rendering = (FS_TextRendering)st->style_text_rendering;
    if (rendering == FS_TEXT_RENDERING_OPTIMIZE_SPEED) {
        return false;
    }
    return true;
}

static bool fs_is_text_geometric_precision(const FS_InternalState* st) {
    if (!st) {
        return false;
    }
    return (FS_TextRendering)st->style_text_rendering == FS_TEXT_RENDERING_GEOMETRIC_PRECISION;
}

static float fs_text_stretch_scale(const FS_InternalState* st) {
    if (!st) {
        return 1.0f;
    }
    switch ((FS_FontStretch)st->style_font_stretch) {
        case FS_FONT_STRETCH_ULTRA_CONDENSED:
            return 0.5f;
        case FS_FONT_STRETCH_EXTRA_CONDENSED:
            return 0.625f;
        case FS_FONT_STRETCH_CONDENSED:
            return 0.75f;
        case FS_FONT_STRETCH_SEMI_CONDENSED:
            return 0.875f;
        case FS_FONT_STRETCH_SEMI_EXPANDED:
            return 1.125f;
        case FS_FONT_STRETCH_EXPANDED:
            return 1.25f;
        case FS_FONT_STRETCH_EXTRA_EXPANDED:
            return 1.5f;
        case FS_FONT_STRETCH_ULTRA_EXPANDED:
            return 2.0f;
        case FS_FONT_STRETCH_NORMAL:
        default:
            return 1.0f;
    }
}

static bool fs_is_text_small_caps_enabled(const FS_InternalState* st) {
    if (!st) {
        return false;
    }
    switch ((FS_FontVariantCaps)st->style_font_variant_caps) {
        case FS_FONT_VARIANT_CAPS_SMALL_CAPS:
        case FS_FONT_VARIANT_CAPS_ALL_SMALL_CAPS:
            return true;
        default:
            return false;
    }
}

static void fs_text_variant_map_codepoint(const FS_InternalState* st, uint32_t cp, uint32_t* out_cp, float* out_size_scale) {
    uint32_t mapped_cp = cp;
    float mapped_scale = 1.0f;
    if (fs_is_text_small_caps_enabled(st)) {
        const FS_FontVariantCaps variant = (FS_FontVariantCaps)st->style_font_variant_caps;
        if (cp >= (uint32_t)'a' && cp <= (uint32_t)'z') {
            mapped_cp = cp - ((uint32_t)'a' - (uint32_t)'A');
            mapped_scale = 0.82f;
        } else if (variant == FS_FONT_VARIANT_CAPS_ALL_SMALL_CAPS &&
                   cp >= (uint32_t)'A' && cp <= (uint32_t)'Z') {
            mapped_cp = cp;
            mapped_scale = 0.82f;
        }
    }
    if (out_cp) {
        *out_cp = mapped_cp;
    }
    if (out_size_scale) {
        *out_size_scale = mapped_scale;
    }
}

static bool fs_vec2_normalize(float x, float y, float* out_x, float* out_y) {
    const float len = hypotf(x, y);
    if (len <= 1e-6f) {
        return false;
    }
    const float inv_len = 1.0f / len;
    *out_x = x * inv_len;
    *out_y = y * inv_len;
    return true;
}

static bool fs_transform_points_aabb(
    const FS_Transform2D* t,
    const float* xy,
    uint32_t point_count,
    float* out_min_x,
    float* out_min_y,
    float* out_max_x,
    float* out_max_y
) {
    if (!xy || point_count == 0u || !out_min_x || !out_min_y || !out_max_x || !out_max_y) {
        return false;
    }
    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
    for (uint32_t i = 0u; i < point_count; ++i) {
        float tx = xy[i * 2u + 0u];
        float ty = xy[i * 2u + 1u];
        fs_transform_apply_point(t, tx, ty, &tx, &ty);
        if (i == 0u) {
            min_x = max_x = tx;
            min_y = max_y = ty;
        } else {
            if (tx < min_x) min_x = tx;
            if (ty < min_y) min_y = ty;
            if (tx > max_x) max_x = tx;
            if (ty > max_y) max_y = ty;
        }
    }
    *out_min_x = min_x;
    *out_min_y = min_y;
    *out_max_x = max_x;
    *out_max_y = max_y;
    return true;
}

static bool fs_path_append_ellipse_loop(FS_Core* core, float cx, float cy, float rx, float ry) {
    if (!core || rx <= 0.0f || ry <= 0.0f) {
        return false;
    }
    const float k = 0.552284749831f;
    const float kx = rx * k;
    const float ky = ry * k;
    if (!fs_path_move_to(core, cx + rx, cy)) {
        return false;
    }
    if (!fs_path_bezier_curve_to(core, cx + rx, cy + ky, cx + kx, cy + ry, cx, cy + ry)) {
        return false;
    }
    if (!fs_path_bezier_curve_to(core, cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, cy)) {
        return false;
    }
    if (!fs_path_bezier_curve_to(core, cx - rx, cy - ky, cx - kx, cy - ry, cx, cy - ry)) {
        return false;
    }
    if (!fs_path_bezier_curve_to(core, cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, cy)) {
        return false;
    }
    return fs_path_close(core);
}

static bool fs_cmd_polygon_compute_coverage_fill_with_flags(
    FS_Core* core,
    const float* xy,
    uint32_t point_count,
    uint32_t color,
    uint32_t extra_flags
) {
    if (!core || !xy || point_count < 3u) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    const FS_Transform2D* t = &st->current_transform;
    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
    if (!fs_transform_points_aabb(t, xy, point_count, &min_x, &min_y, &max_x, &max_y)) {
        return false;
    }
    FS_PathStateBorrow path_saved;
    fs_path_state_begin_temporary(st, &path_saved);

    fs_state_save(core);
    fs_path_begin(core);
    bool ok = fs_path_move_to(core, xy[0], xy[1]);
    for (uint32_t i = 1u; ok && i < point_count; ++i) {
        ok = fs_path_line_to(core, xy[i * 2u + 0u], xy[i * 2u + 1u]);
    }
    if (ok) {
        ok = fs_path_close(core);
    }
    if (ok) {
        ok = fs_clip_path_with_fill_rule(core, FS_FILL_RULE_NONZERO);
    }
    if (ok) {
        const float pad = 1.5f;
        fs_transform_reset(core);
        ok = fs_cmd_rect_with_flags(
            core,
            min_x - pad,
            min_y - pad,
            (max_x - min_x) + pad * 2.0f,
            (max_y - min_y) + pad * 2.0f,
            0.0f,
            color,
            extra_flags & FS_RENDER_FLAG_PATTERN_SHADE
        );
    }
    (void)fs_state_restore(core);
    fs_path_state_end_temporary(st, &path_saved);
    return ok;
}

static bool fs_cmd_polygon_compute_coverage_fill(FS_Core* core, const float* xy, uint32_t point_count, uint32_t color) {
    return fs_cmd_polygon_compute_coverage_fill_with_flags(core, xy, point_count, color, 0u);
}

static bool fs_cmd_ellipse_compute_coverage_fill_with_flags(
    FS_Core* core,
    float cx,
    float cy,
    float rx,
    float ry,
    uint32_t color,
    uint32_t extra_flags
) {
    if (!core || rx <= 0.0f || ry <= 0.0f) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    FS_PathStateBorrow path_saved;
    fs_path_state_begin_temporary(st, &path_saved);
    const FS_Transform2D* t = &st->current_transform;
    float tx = cx - rx;
    float ty = cy - ry;
    float tw = rx * 2.0f;
    float th = ry * 2.0f;
    fs_transform_rect_to_aabb(t, cx - rx, cy - ry, rx * 2.0f, ry * 2.0f, &tx, &ty, &tw, &th);

    fs_state_save(core);
    fs_path_begin(core);
    bool ok = fs_path_append_ellipse_loop(core, cx, cy, rx, ry);
    if (ok) {
        ok = fs_clip_path_with_fill_rule(core, FS_FILL_RULE_NONZERO);
    }
    if (ok) {
        const float pad = 1.5f;
        fs_transform_reset(core);
        ok = fs_cmd_rect_with_flags(
            core,
            tx - pad,
            ty - pad,
            tw + pad * 2.0f,
            th + pad * 2.0f,
            0.0f,
            color,
            extra_flags & FS_RENDER_FLAG_PATTERN_SHADE
        );
    }
    (void)fs_state_restore(core);
    fs_path_state_end_temporary(st, &path_saved);
    return ok;
}

static bool fs_cmd_ellipse_compute_coverage_fill(FS_Core* core, float cx, float cy, float rx, float ry, uint32_t color) {
    return fs_cmd_ellipse_compute_coverage_fill_with_flags(core, cx, cy, rx, ry, color, 0u);
}

static bool fs_emit_styled_line_segment_compute_coverage(
    FS_Core* core,
    float x0,
    float y0,
    float x1,
    float y1,
    float width,
    uint32_t color,
    uint8_t line_cap,
    uint32_t extra_line_flags
) {
    if (!core || width <= 0.0f) {
        return false;
    }

    float sx0 = x0;
    float sy0 = y0;
    float sx1 = x1;
    float sy1 = y1;
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float len = sqrtf(dx * dx + dy * dy);
    if (len <= 1e-6f) {
        return true;
    }

    const float hw = width * 0.5f;
    if (line_cap == (uint8_t)FS_LINE_CAP_SQUARE) {
        const float ex = (dx / len) * hw;
        const float ey = (dy / len) * hw;
        sx0 -= ex;
        sy0 -= ey;
        sx1 += ex;
        sy1 += ey;
    }

    const float inv_len = 1.0f / len;
    const float nx = -dy * inv_len * hw;
    const float ny = dx * inv_len * hw;
    const float quad_xy[8] = {
        sx0 + nx, sy0 + ny,
        sx1 + nx, sy1 + ny,
        sx1 - nx, sy1 - ny,
        sx0 - nx, sy0 - ny
    };
    const uint32_t coverage_flags = extra_line_flags & FS_RENDER_FLAG_PATTERN_SHADE;
    if (!fs_cmd_polygon_compute_coverage_fill_with_flags(core, quad_xy, 4u, color, coverage_flags)) {
        return false;
    }
    if (line_cap == (uint8_t)FS_LINE_CAP_ROUND) {
        if ((extra_line_flags & FS_LINE_FLAG_NO_AA_START) == 0u) {
            if (!fs_cmd_ellipse_compute_coverage_fill_with_flags(core, sx0, sy0, hw, hw, color, coverage_flags)) {
                return false;
            }
        }
        if ((extra_line_flags & FS_LINE_FLAG_NO_AA_END) == 0u) {
            if (!fs_cmd_ellipse_compute_coverage_fill_with_flags(core, sx1, sy1, hw, hw, color, coverage_flags)) {
                return false;
            }
        }
    }
    return true;
}

static bool fs_cmd_ellipse_compute_coverage_stroke_with_flags(
    FS_Core* core,
    float x0,
    float y0,
    float x1,
    float y1,
    float width,
    uint32_t color,
    uint32_t user_flags
) {
    if (!core || width <= 0.0f) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    const uint8_t line_cap = st ? st->style_line_cap : (uint8_t)FS_LINE_CAP_ROUND;
    return fs_emit_styled_line_segment_compute_coverage(core, x0, y0, x1, y1, width, color, line_cap, user_flags);
}

static bool fs_cmd_arc_compute_coverage_stroke(
    FS_Core* core,
    float cx,
    float cy,
    float radius,
    float thickness,
    float start_angle,
    float end_angle,
    uint32_t color
) {
    if (!core || radius <= 0.0f || thickness <= 0.0f) {
        return false;
    }
    const float pi = 3.14159265358979323846f;
    const float two_pi = 2.0f * pi;
    const float raw_sweep = end_angle - start_angle;
    if (fabsf(raw_sweep) <= 1e-6f) {
        return true;
    }
    float start = fmodf(start_angle, two_pi);
    float end = fmodf(end_angle, two_pi);
    if (start < 0.0f) start += two_pi;
    if (end < 0.0f) end += two_pi;
    float sweep = end - start;
    if (sweep <= 0.0f) {
        sweep += two_pi;
    }
    const bool full_ring = fabsf(raw_sweep) >= (two_pi - 1e-4f);
    if (full_ring) {
        sweep = two_pi;
    }

    const float outer_r = fmaxf(radius + thickness * 0.5f, 0.0f);
    const float inner_r = fmaxf(radius - thickness * 0.5f, 0.0f);
    if (outer_r <= 1e-6f) {
        return true;
    }

    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    FS_PathStateBorrow path_saved;
    fs_path_state_begin_temporary(st, &path_saved);
    const FS_Transform2D* t = &st->current_transform;
    float tx = cx - outer_r;
    float ty = cy - outer_r;
    float tw = outer_r * 2.0f;
    float th = outer_r * 2.0f;
    fs_transform_rect_to_aabb(t, cx - outer_r, cy - outer_r, outer_r * 2.0f, outer_r * 2.0f, &tx, &ty, &tw, &th);

    fs_state_save(core);
    fs_path_begin(core);
    bool ok = true;
    const float end_abs = start + sweep;
    const float sx = cx + cosf(start) * outer_r;
    const float sy = cy + sinf(start) * outer_r;
    if (!fs_path_move_to(core, sx, sy)) {
        ok = false;
    }
    if (ok) {
        ok = fs_path_append_arc_sweep(core, cx, cy, outer_r, start, end_abs);
    }
    if (ok) {
        if (inner_r > 1e-6f) {
            const float ix1 = cx + cosf(end_abs) * inner_r;
            const float iy1 = cy + sinf(end_abs) * inner_r;
            ok = fs_path_line_to(core, ix1, iy1);
            if (ok) {
                ok = fs_path_append_arc_sweep(core, cx, cy, inner_r, end_abs, start);
            }
        } else {
            ok = fs_path_line_to(core, cx, cy);
        }
    }
    if (ok) {
        ok = fs_path_close(core);
    }
    if (ok) {
        const FS_FillRule fill_rule = (full_ring && inner_r > 1e-6f) ? FS_FILL_RULE_EVENODD : FS_FILL_RULE_NONZERO;
        ok = fs_clip_path_with_fill_rule(core, fill_rule);
    }
    if (ok) {
        const float pad = 1.5f;
        fs_transform_reset(core);
        ok = fs_cmd_rect(core, tx - pad, ty - pad, tw + pad * 2.0f, th + pad * 2.0f, 0.0f, color);
    }
    (void)fs_state_restore(core);
    fs_path_state_end_temporary(st, &path_saved);
    return ok;
}


static bool fs_path_segment_start_dir(const FS_PathSegment* seg, float* out_dx, float* out_dy) {
    if (!seg || !out_dx || !out_dy) {
        return false;
    }
    switch ((FS_PathSegType)seg->type) {
        case FS_PATH_SEG_LINE:
            return fs_vec2_normalize(seg->x1 - seg->x0, seg->y1 - seg->y0, out_dx, out_dy);
        case FS_PATH_SEG_QUAD:
            if (fs_vec2_normalize(seg->cx0 - seg->x0, seg->cy0 - seg->y0, out_dx, out_dy)) {
                return true;
            }
            if (fs_vec2_normalize(seg->x1 - seg->cx0, seg->y1 - seg->cy0, out_dx, out_dy)) {
                return true;
            }
            return fs_vec2_normalize(seg->x1 - seg->x0, seg->y1 - seg->y0, out_dx, out_dy);
        case FS_PATH_SEG_CUBIC:
            if (fs_vec2_normalize(seg->cx0 - seg->x0, seg->cy0 - seg->y0, out_dx, out_dy)) {
                return true;
            }
            if (fs_vec2_normalize(seg->cx1 - seg->cx0, seg->cy1 - seg->cy0, out_dx, out_dy)) {
                return true;
            }
            if (fs_vec2_normalize(seg->x1 - seg->cx1, seg->y1 - seg->cy1, out_dx, out_dy)) {
                return true;
            }
            return fs_vec2_normalize(seg->x1 - seg->x0, seg->y1 - seg->y0, out_dx, out_dy);
        default:
            return false;
    }
}

static bool fs_path_segment_end_dir(const FS_PathSegment* seg, float* out_dx, float* out_dy) {
    if (!seg || !out_dx || !out_dy) {
        return false;
    }
    switch ((FS_PathSegType)seg->type) {
        case FS_PATH_SEG_LINE:
            return fs_vec2_normalize(seg->x1 - seg->x0, seg->y1 - seg->y0, out_dx, out_dy);
        case FS_PATH_SEG_QUAD:
            if (fs_vec2_normalize(seg->x1 - seg->cx0, seg->y1 - seg->cy0, out_dx, out_dy)) {
                return true;
            }
            if (fs_vec2_normalize(seg->cx0 - seg->x0, seg->cy0 - seg->y0, out_dx, out_dy)) {
                return true;
            }
            return fs_vec2_normalize(seg->x1 - seg->x0, seg->y1 - seg->y0, out_dx, out_dy);
        case FS_PATH_SEG_CUBIC:
            if (fs_vec2_normalize(seg->x1 - seg->cx1, seg->y1 - seg->cy1, out_dx, out_dy)) {
                return true;
            }
            if (fs_vec2_normalize(seg->cx1 - seg->cx0, seg->cy1 - seg->cy0, out_dx, out_dy)) {
                return true;
            }
            if (fs_vec2_normalize(seg->cx0 - seg->x0, seg->cy0 - seg->y0, out_dx, out_dy)) {
                return true;
            }
            return fs_vec2_normalize(seg->x1 - seg->x0, seg->y1 - seg->y0, out_dx, out_dy);
        default:
            return false;
    }
}

static bool fs_emit_path_join(
    FS_Core* core,
    float px,
    float py,
    float in_dx,
    float in_dy,
    float out_dx,
    float out_dy,
    float stroke_width,
    uint32_t color,
    uint8_t line_join,
    float miter_limit,
    uint32_t extra_user_flags
) {
    if (!core || stroke_width <= 0.0f) {
        return false;
    }
    if (!fs_vec2_normalize(in_dx, in_dy, &in_dx, &in_dy) || !fs_vec2_normalize(out_dx, out_dy, &out_dx, &out_dy)) {
        return true;
    }
    const float dot = in_dx * out_dx + in_dy * out_dy;
    if (dot > 0.9995f) {
        return true;
    }
    const float turn = in_dx * out_dy - in_dy * out_dx;
    if (fabsf(turn) <= 1e-5f) {
        return true;
    }
    const float hw = stroke_width * 0.5f;
    // Screen-space Y grows downward; for outer join side we need the opposite sign
    // compared to Cartesian math conventions.
    const float side = (turn > 0.0f) ? -1.0f : 1.0f;
    const float nin_x = side * (-in_dy);
    const float nin_y = side * in_dx;
    const float nout_x = side * (-out_dy);
    const float nout_y = side * out_dx;
    const float ax = px + nin_x * hw;
    const float ay = py + nin_y * hw;
    const float bx = px + nout_x * hw;
    const float by = py + nout_y * hw;
    FS_InternalState* st = fs_state(core);
    const bool oriented_transform = st && fs_transform_requires_oriented_quad(&st->current_transform);

    const uint32_t coverage_flags = extra_user_flags & FS_RENDER_FLAG_PATTERN_SHADE;
    if (line_join == (uint8_t)FS_LINE_JOIN_BEVEL) {
        if (oriented_transform) {
            const float tri[6] = {px, py, ax, ay, bx, by};
            return fs_cmd_polygon_compute_coverage_fill_with_flags(core, tri, 3u, color, coverage_flags);
        }
        // For join wedges, only edge AB-B C (outer edge) should be antialiased.
        return fs_cmd_triangle_with_edge_mask(core, px, py, ax, ay, bx, by, color, FS_TRI_FLAG_AA_EDGE1, extra_user_flags);
    }

    if (line_join == (uint8_t)FS_LINE_JOIN_MITER) {
        const float denom = in_dx * out_dy - in_dy * out_dx;
        if (fabsf(denom) > 1e-6f) {
            const float qpx = bx - ax;
            const float qpy = by - ay;
            const float t = (qpx * out_dy - qpy * out_dx) / denom;
            const float mx = ax + in_dx * t;
            const float my = ay + in_dy * t;
            float resolved_limit = miter_limit;
            if (!isfinite(resolved_limit) || resolved_limit <= 0.0f) {
                resolved_limit = 10.0f;
            }
            if (resolved_limit < 1.0f) {
                resolved_limit = 1.0f;
            }
            const float miter_len = hypotf(mx - px, my - py) / fmaxf(hw, 1e-6f);
            if (isfinite(miter_len) && miter_len <= resolved_limit) {
                if (oriented_transform) {
                    const float tri0[6] = {px, py, ax, ay, mx, my};
                    const float tri1[6] = {px, py, mx, my, bx, by};
                    if (!fs_cmd_polygon_compute_coverage_fill_with_flags(core, tri0, 3u, color, coverage_flags)) {
                        return false;
                    }
                    if (!fs_cmd_polygon_compute_coverage_fill_with_flags(core, tri1, 3u, color, coverage_flags)) {
                        return false;
                    }
                } else {
                    if (!fs_cmd_triangle_with_edge_mask(
                            core,
                            px,
                            py,
                            ax,
                            ay,
                            mx,
                            my,
                            color,
                            FS_TRI_FLAG_AA_EDGE1,
                            extra_user_flags
                        )) {
                        return false;
                    }
                    if (!fs_cmd_triangle_with_edge_mask(
                            core,
                            px,
                            py,
                            mx,
                            my,
                            bx,
                            by,
                            color,
                            FS_TRI_FLAG_AA_EDGE1,
                            extra_user_flags
                        )) {
                        return false;
                    }
                }
                return true;
            }
        }
        if (oriented_transform) {
            const float tri[6] = {px, py, ax, ay, bx, by};
        return fs_cmd_polygon_compute_coverage_fill_with_flags(core, tri, 3u, color, coverage_flags);
        }
        return fs_cmd_triangle_with_edge_mask(core, px, py, ax, ay, bx, by, color, FS_TRI_FLAG_AA_EDGE1, extra_user_flags);
    }
    return true;
}

static bool fs_command_supports_shadow(uint32_t cmd_type) {
    switch (cmd_type) {
        case FS_CMD_RECT:
        case FS_CMD_IMAGE:
        case FS_CMD_TEXT:
        case FS_CMD_LINE:
        case FS_CMD_PATH_SEGMENT:
        case FS_CMD_CIRCLE:
        case FS_CMD_ARC:
        case FS_CMD_BEZIER_QUAD:
        case FS_CMD_RECT_STROKE:
        case FS_CMD_ELLIPSE:
        case FS_CMD_BEZIER_CUBIC:
        case FS_CMD_TRIANGLE:
            return true;
        default:
            return false;
    }
}

static uint32_t fs_shadow_blur_to_flag_bits(float blur_px) {
    if (!isfinite(blur_px) || blur_px <= 0.0f) {
        return 0u;
    }
    float clamped = blur_px;
    if (clamped > FS_SHADOW_BLUR_MAX) {
        clamped = FS_SHADOW_BLUR_MAX;
    }
    uint32_t q = (uint32_t)lroundf(clamped);
    if (q > 15u) {
        q = 15u;
    }
    return (q << FS_RENDER_FLAG_SHADOW_BLUR_SHIFT) & FS_RENDER_FLAG_SHADOW_BLUR_MASK;
}

static uint32_t fs_color_scale_alpha_rgba8(uint32_t color_rgba8, float scale) {
    if (!isfinite(scale) || scale <= 0.0f) {
        return color_rgba8 & 0x00FFFFFFu;
    }
    if (scale >= 1.0f) {
        return color_rgba8;
    }
    const uint32_t alpha = (color_rgba8 >> 24u) & 0xFFu;
    uint32_t scaled_alpha = (uint32_t)lroundf((float)alpha * scale);
    if (scaled_alpha > 0xFFu) {
        scaled_alpha = 0xFFu;
    }
    return (color_rgba8 & 0x00FFFFFFu) | (scaled_alpha << 24u);
}

static uint8_t fs_color_r_u8(uint32_t color) {
    return (uint8_t)(color & 0xFFu);
}

static uint8_t fs_color_g_u8(uint32_t color) {
    return (uint8_t)((color >> 8u) & 0xFFu);
}

static uint8_t fs_color_b_u8(uint32_t color) {
    return (uint8_t)((color >> 16u) & 0xFFu);
}

static uint8_t fs_color_a_u8(uint32_t color) {
    return (uint8_t)((color >> 24u) & 0xFFu);
}

static uint32_t fs_color_rgba8_pack_u8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t)a << 24u) | ((uint32_t)b << 16u) | ((uint32_t)g << 8u) | (uint32_t)r;
}

static bool fs_style_copy_gradient_stops(
    FS_GradientStop* dst,
    uint32_t dst_max_count,
    uint32_t* out_count,
    const FS_GradientStop* src,
    uint32_t src_count
) {
    if (!dst || !out_count || !src || src_count < 2u || dst_max_count == 0u) {
        return false;
    }
    uint32_t copy_count = src_count;
    if (copy_count > dst_max_count) {
        copy_count = dst_max_count;
    }
    if (copy_count == src_count) {
        memcpy(dst, src, (size_t)copy_count * sizeof(FS_GradientStop));
    } else if (copy_count == 1u) {
        dst[0] = src[0];
    } else {
        for (uint32_t i = 0u; i < copy_count; ++i) {
            const uint32_t src_index = (uint32_t)(((uint64_t)i * (uint64_t)(src_count - 1u)) / (uint64_t)(copy_count - 1u));
            dst[i] = src[src_index];
        }
    }
    *out_count = copy_count;
    return true;
}

static uint32_t fs_gradient_stops_sample_rgba8(const FS_GradientStop* stops, uint32_t stop_count, float t) {
    if (!stops || stop_count == 0u) {
        return 0xFF000000u;
    }
    if (stop_count == 1u) {
        return stops[0].color_rgba8;
    }
    if (t <= stops[0].offset_0_to_1) {
        return stops[0].color_rgba8;
    }
    const uint32_t last_index = stop_count - 1u;
    if (t >= stops[last_index].offset_0_to_1) {
        return stops[last_index].color_rgba8;
    }
    for (uint32_t i = 1u; i < stop_count; ++i) {
        const FS_GradientStop* b = &stops[i];
        if (t > b->offset_0_to_1) {
            continue;
        }
        const FS_GradientStop* a = &stops[i - 1u];
        const float denom = b->offset_0_to_1 - a->offset_0_to_1;
        float k = 0.0f;
        if (denom > 1e-8f) {
            k = (t - a->offset_0_to_1) / denom;
        }
        if (k < 0.0f) {
            k = 0.0f;
        } else if (k > 1.0f) {
            k = 1.0f;
        }
        const float ar = (float)fs_color_r_u8(a->color_rgba8);
        const float ag = (float)fs_color_g_u8(a->color_rgba8);
        const float ab = (float)fs_color_b_u8(a->color_rgba8);
        const float aa = (float)fs_color_a_u8(a->color_rgba8);
        const float br = (float)fs_color_r_u8(b->color_rgba8);
        const float bg = (float)fs_color_g_u8(b->color_rgba8);
        const float bb = (float)fs_color_b_u8(b->color_rgba8);
        const float ba = (float)fs_color_a_u8(b->color_rgba8);
        const uint8_t r = (uint8_t)lroundf(ar + (br - ar) * k);
        const uint8_t g = (uint8_t)lroundf(ag + (bg - ag) * k);
        const uint8_t bch = (uint8_t)lroundf(ab + (bb - ab) * k);
        const uint8_t ach = (uint8_t)lroundf(aa + (ba - aa) * k);
        return fs_color_rgba8_pack_u8(r, g, bch, ach);
    }
    return stops[last_index].color_rgba8;
}

static bool fs_style_copy_linear_gradient(FS_StyleLinearGradient* dst, const FS_LinearGradient* src) {
    if (!dst || !src || !isfinite(src->x0) || !isfinite(src->y0) || !isfinite(src->x1) || !isfinite(src->y1)) {
        return false;
    }
    if (!src->stops || src->stop_count < 2u) {
        return false;
    }
    dst->x0 = src->x0;
    dst->y0 = src->y0;
    dst->x1 = src->x1;
    dst->y1 = src->y1;
    return fs_style_copy_gradient_stops(
        dst->stops,
        FS_LINEAR_GRADIENT_MAX_STOPS,
        &dst->stop_count,
        src->stops,
        src->stop_count
    );
}

static bool fs_style_copy_radial_gradient(FS_StyleRadialGradient* dst, const FS_RadialGradient* src) {
    if (!dst || !src || !isfinite(src->x0) || !isfinite(src->y0) || !isfinite(src->r0) || src->r0 < 0.0f ||
        !isfinite(src->x1) || !isfinite(src->y1) || !isfinite(src->r1) || src->r1 < 0.0f) {
        return false;
    }
    if (!src->stops || src->stop_count < 2u) {
        return false;
    }
    dst->x0 = src->x0;
    dst->y0 = src->y0;
    dst->r0 = src->r0;
    dst->x1 = src->x1;
    dst->y1 = src->y1;
    dst->r1 = src->r1;
    return fs_style_copy_gradient_stops(
        dst->stops,
        FS_RADIAL_GRADIENT_MAX_STOPS,
        &dst->stop_count,
        src->stops,
        src->stop_count
    );
}

static bool fs_style_copy_conic_gradient(FS_StyleConicGradient* dst, const FS_ConicGradient* src) {
    if (!dst || !src || !isfinite(src->start_angle_radians) || !isfinite(src->cx) || !isfinite(src->cy)) {
        return false;
    }
    if (!src->stops || src->stop_count < 2u) {
        return false;
    }
    dst->start_angle_radians = src->start_angle_radians;
    dst->cx = src->cx;
    dst->cy = src->cy;
    return fs_style_copy_gradient_stops(
        dst->stops,
        FS_CONIC_GRADIENT_MAX_STOPS,
        &dst->stop_count,
        src->stops,
        src->stop_count
    );
}

static void fs_affine_set_identity_2d(float m[6]) {
    if (!m) {
        return;
    }
    m[0] = 1.0f;
    m[1] = 0.0f;
    m[2] = 0.0f;
    m[3] = 1.0f;
    m[4] = 0.0f;
    m[5] = 0.0f;
}

static bool fs_affine_try_invert_2d(const float m[6], float out_inv[6]) {
    if (!m || !out_inv) {
        return false;
    }
    const float a = m[0];
    const float b = m[1];
    const float c = m[2];
    const float d = m[3];
    const float e = m[4];
    const float f = m[5];
    if (!isfinite(a) || !isfinite(b) || !isfinite(c) || !isfinite(d) || !isfinite(e) || !isfinite(f)) {
        return false;
    }
    const float det = a * d - b * c;
    if (!isfinite(det) || fabsf(det) <= 1e-8f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    out_inv[0] = d * inv_det;
    out_inv[1] = -b * inv_det;
    out_inv[2] = -c * inv_det;
    out_inv[3] = a * inv_det;
    out_inv[4] = (c * f - d * e) * inv_det;
    out_inv[5] = (b * e - a * f) * inv_det;
    return true;
}

static void fs_affine_apply_point_2d(const float m[6], float x, float y, float* out_x, float* out_y) {
    if (!m || !out_x || !out_y) {
        return;
    }
    *out_x = m[0] * x + m[2] * y + m[4];
    *out_y = m[1] * x + m[3] * y + m[5];
}

static bool fs_style_copy_pattern(FS_StylePattern* dst, const FS_Pattern* src) {
    if (!dst || !src) {
        return false;
    }
    const uint8_t repeat_mode = src->repeat_mode;
    if (repeat_mode > (uint8_t)FS_PATTERN_NO_REPEAT) {
        return false;
    }
    if (src->handle.width == 0u || src->handle.height == 0u) {
        return false;
    }
    *dst = (FS_StylePattern){
        .handle = src->handle,
        .xform = {src->xform[0], src->xform[1], src->xform[2], src->xform[3], src->xform[4], src->xform[5]},
        .inv_xform = {
            src->inv_xform[0],
            src->inv_xform[1],
            src->inv_xform[2],
            src->inv_xform[3],
            src->inv_xform[4],
            src->inv_xform[5]
        },
        .repeat_mode = repeat_mode,
        .inv_valid = src->inv_valid,
        ._pad0 = 0u,
        ._pad1 = 0u
    };
    if (!dst->inv_valid) {
        dst->inv_valid = fs_affine_try_invert_2d(dst->xform, dst->inv_xform) ? 1u : 0u;
        if (!dst->inv_valid) {
            return false;
        }
    }
    return true;
}

static bool fs_image_handle_resolve_atlas_origin(
    const FS_Core* core,
    const FS_ImageHandle* handle,
    uint32_t* out_atlas_x,
    uint32_t* out_atlas_y
) {
    if (!core || !handle || !out_atlas_x || !out_atlas_y) {
        return false;
    }
    if (handle->width == 0u || handle->height == 0u) {
        return false;
    }
    if (handle->layer >= core->image_atlas_layers) {
        return false;
    }
    if (handle->generation != core->image_atlas_generation[handle->layer]) {
        return false;
    }

    uint32_t atlas_x = handle->atlas_x;
    uint32_t atlas_y = handle->atlas_y;
    if (atlas_x + handle->width > core->image_atlas_width ||
        atlas_y + handle->height > core->image_atlas_height) {
        if (!isfinite(handle->uv_min[0]) || !isfinite(handle->uv_min[1])) {
            return false;
        }
        atlas_x = (uint32_t)floorf(handle->uv_min[0] * (float)core->image_atlas_width + 0.5f);
        atlas_y = (uint32_t)floorf(handle->uv_min[1] * (float)core->image_atlas_height + 0.5f);
    }
    if (atlas_x + handle->width > core->image_atlas_width ||
        atlas_y + handle->height > core->image_atlas_height) {
        return false;
    }
    *out_atlas_x = atlas_x;
    *out_atlas_y = atlas_y;
    return true;
}

static void fs_command_state_clear_pattern(FS_CommandStateGPU* state) {
    if (!state) {
        return;
    }
    memset(state->pattern_inv0, 0, sizeof(state->pattern_inv0));
    memset(state->pattern_inv1, 0, sizeof(state->pattern_inv1));
    memset(state->pattern_meta, 0, sizeof(state->pattern_meta));
}

static bool fs_command_state_set_pattern(FS_Core* core, FS_CommandStateGPU* state, const FS_StylePattern* pattern) {
    if (!core || !state || !pattern) {
        return false;
    }
    if (pattern->handle.width == 0u || pattern->handle.height == 0u || !pattern->inv_valid) {
        return false;
    }
    uint32_t atlas_x = 0u;
    uint32_t atlas_y = 0u;
    if (!fs_image_handle_resolve_atlas_origin(core, &pattern->handle, &atlas_x, &atlas_y)) {
        return false;
    }
    state->pattern_inv0[0] = pattern->inv_xform[0];
    state->pattern_inv0[1] = pattern->inv_xform[1];
    state->pattern_inv0[2] = pattern->inv_xform[2];
    state->pattern_inv0[3] = pattern->inv_xform[3];
    state->pattern_inv1[0] = pattern->inv_xform[4];
    state->pattern_inv1[1] = pattern->inv_xform[5];
    state->pattern_inv1[2] = (float)atlas_x;
    state->pattern_inv1[3] = (float)atlas_y;
    state->pattern_meta[0] = (float)pattern->handle.width;
    state->pattern_meta[1] = (float)pattern->handle.height;
    state->pattern_meta[2] = (float)pattern->handle.layer;
    state->pattern_meta[3] = (float)pattern->repeat_mode;
    return true;
}

static bool fs_style_pattern_sample_rgba8(
    const FS_InternalState* st,
    const FS_StylePattern* pattern,
    float px,
    float py,
    uint32_t* out_color
) {
    if (!st || !pattern || !out_color || !isfinite(px) || !isfinite(py)) {
        return false;
    }
    const FS_Core* core = st->owner_core;
    if (!core || !core->image_atlas_shadow_rgba || core->image_atlas_shadow_size == 0u) {
        return false;
    }
    const uint32_t w = pattern->handle.width;
    const uint32_t h = pattern->handle.height;
    if (w == 0u || h == 0u) {
        return false;
    }
    uint32_t atlas_x = 0u;
    uint32_t atlas_y = 0u;
    if (!fs_image_handle_resolve_atlas_origin(core, &pattern->handle, &atlas_x, &atlas_y)) {
        return false;
    }

    const uint8_t repeat_mode = pattern->repeat_mode;
    const bool repeat_x =
        repeat_mode == (uint8_t)FS_PATTERN_REPEAT || repeat_mode == (uint8_t)FS_PATTERN_REPEAT_X;
    const bool repeat_y =
        repeat_mode == (uint8_t)FS_PATTERN_REPEAT || repeat_mode == (uint8_t)FS_PATTERN_REPEAT_Y;

    float pattern_x = px;
    float pattern_y = py;
    if (pattern->inv_valid) {
        fs_affine_apply_point_2d(pattern->inv_xform, px, py, &pattern_x, &pattern_y);
    }
    int64_t sx = (int64_t)floorf(pattern_x);
    int64_t sy = (int64_t)floorf(pattern_y);
    if (!repeat_x) {
        if (sx < 0 || sx >= (int64_t)w) {
            return false;
        }
    } else {
        sx %= (int64_t)w;
        if (sx < 0) {
            sx += (int64_t)w;
        }
    }
    if (!repeat_y) {
        if (sy < 0 || sy >= (int64_t)h) {
            return false;
        }
    } else {
        sy %= (int64_t)h;
        if (sy < 0) {
            sy += (int64_t)h;
        }
    }

    const uint32_t sample_x = atlas_x + (uint32_t)sx;
    const uint32_t sample_y = atlas_y + (uint32_t)sy;
    if (sample_x >= core->image_atlas_width || sample_y >= core->image_atlas_height) {
        return false;
    }
    const size_t atlas_w = (size_t)core->image_atlas_width;
    const size_t atlas_h = (size_t)core->image_atlas_height;
    const size_t layer_stride_px = atlas_w * atlas_h;
    const size_t px_index =
        (size_t)pattern->handle.layer * layer_stride_px +
        (size_t)sample_y * atlas_w +
        (size_t)sample_x;
    const uint8_t* rgba = core->image_atlas_shadow_rgba + px_index * 4u;
    *out_color = fs_color_rgba8_pack_u8(rgba[0], rgba[1], rgba[2], rgba[3]);
    return true;
}

static uint32_t fs_linear_gradient_sample_rgba8(const FS_StyleLinearGradient* grad, float px, float py) {
    if (!grad) {
        return 0xFF000000u;
    }
    const float dx = grad->x1 - grad->x0;
    const float dy = grad->y1 - grad->y0;
    const float len_sq = dx * dx + dy * dy;
    float t = 0.0f;
    if (len_sq > 1e-8f) {
        t = ((px - grad->x0) * dx + (py - grad->y0) * dy) / len_sq;
    }
    return fs_gradient_stops_sample_rgba8(grad->stops, grad->stop_count, t);
}

static uint32_t fs_conic_gradient_sample_rgba8(const FS_StyleConicGradient* grad, float px, float py) {
    if (!grad) {
        return 0xFF000000u;
    }
    const float two_pi = 6.2831853071795864769f;
    float angle = atan2f(py - grad->cy, px - grad->cx) - grad->start_angle_radians;
    angle = fmodf(angle, two_pi);
    if (angle < 0.0f) {
        angle += two_pi;
    }
    const float t = angle / two_pi;
    return fs_gradient_stops_sample_rgba8(grad->stops, grad->stop_count, t);
}

static uint32_t fs_radial_gradient_sample_rgba8(const FS_StyleRadialGradient* grad, float px, float py) {
    if (!grad) {
        return 0xFF000000u;
    }
    const float sx = px - grad->x0;
    const float sy = py - grad->y0;
    const float dx = grad->x1 - grad->x0;
    const float dy = grad->y1 - grad->y0;
    const float dr = grad->r1 - grad->r0;
    const float a = dx * dx + dy * dy - dr * dr;
    const float b = -2.0f * (sx * dx + sy * dy + grad->r0 * dr);
    const float c = sx * sx + sy * sy - grad->r0 * grad->r0;

    float t = 0.0f;
    if (fabsf(a) <= 1e-8f) {
        if (fabsf(b) > 1e-8f) {
            t = -c / b;
        }
    } else {
        const float disc = b * b - 4.0f * a * c;
        if (disc < 0.0f) {
            t = -b / (2.0f * a);
        } else {
            const float sqrt_disc = sqrtf(disc);
            const float inv_2a = 0.5f / a;
            const float t0 = (-b - sqrt_disc) * inv_2a;
            const float t1 = (-b + sqrt_disc) * inv_2a;
            const bool t0_in = (t0 >= 0.0f && t0 <= 1.0f);
            const bool t1_in = (t1 >= 0.0f && t1 <= 1.0f);
            if (t0_in && t1_in) {
                t = (t0 > t1) ? t0 : t1;
            } else if (t0_in) {
                t = t0;
            } else if (t1_in) {
                t = t1;
            } else {
                const float d0 = fabsf(t0 - 0.5f);
                const float d1 = fabsf(t1 - 0.5f);
                t = (d0 <= d1) ? t0 : t1;
            }
        }
    }
    if (!isfinite(t)) {
        t = 0.0f;
    }
    return fs_gradient_stops_sample_rgba8(grad->stops, grad->stop_count, t);
}

static uint32_t fs_style_resolve_fill_color_at(const FS_InternalState* st, float px, float py) {
    if (!st) {
        return 0xFF000000u;
    }
    if (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT &&
        st->style_fill_linear_gradient.stop_count >= 2u) {
        return fs_linear_gradient_sample_rgba8(&st->style_fill_linear_gradient, px, py);
    }
    if (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT &&
        st->style_fill_radial_gradient.stop_count >= 2u) {
        return fs_radial_gradient_sample_rgba8(&st->style_fill_radial_gradient, px, py);
    }
    if (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_CONIC_GRADIENT &&
        st->style_fill_conic_gradient.stop_count >= 2u) {
        return fs_conic_gradient_sample_rgba8(&st->style_fill_conic_gradient, px, py);
    }
    if (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN) {
        uint32_t sampled = 0u;
        if (fs_style_pattern_sample_rgba8(st, &st->style_fill_pattern, px, py, &sampled)) {
            return sampled;
        }
        return 0u;
    }
    return st->style_fill_color_rgba8;
}

static uint32_t fs_style_resolve_stroke_color_at(const FS_InternalState* st, float px, float py) {
    if (!st) {
        return 0xFF000000u;
    }
    if (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT &&
        st->style_stroke_linear_gradient.stop_count >= 2u) {
        return fs_linear_gradient_sample_rgba8(&st->style_stroke_linear_gradient, px, py);
    }
    if (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT &&
        st->style_stroke_radial_gradient.stop_count >= 2u) {
        return fs_radial_gradient_sample_rgba8(&st->style_stroke_radial_gradient, px, py);
    }
    if (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_CONIC_GRADIENT &&
        st->style_stroke_conic_gradient.stop_count >= 2u) {
        return fs_conic_gradient_sample_rgba8(&st->style_stroke_conic_gradient, px, py);
    }
    if (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN) {
        uint32_t sampled = 0u;
        if (fs_style_pattern_sample_rgba8(st, &st->style_stroke_pattern, px, py, &sampled)) {
            return sampled;
        }
        return 0u;
    }
    return st->style_stroke_color_rgba8;
}

static bool fs_path_bounds_local(
    const FS_InternalState* st,
    float* out_min_x,
    float* out_min_y,
    float* out_max_x,
    float* out_max_y
) {
    if (!st || !st->path_segments || st->path_count == 0u || !out_min_x || !out_min_y || !out_max_x || !out_max_y) {
        return false;
    }
    float min_x = INFINITY;
    float min_y = INFINITY;
    float max_x = -INFINITY;
    float max_y = -INFINITY;
    for (uint32_t i = 0u; i < st->path_count; ++i) {
        const FS_PathSegment* seg = &st->path_segments[i];
        const float pts[8] = {
            seg->x0, seg->y0,
            seg->cx0, seg->cy0,
            seg->cx1, seg->cy1,
            seg->x1, seg->y1
        };
        for (uint32_t p = 0u; p < 4u; ++p) {
            const float x = pts[p * 2u + 0u];
            const float y = pts[p * 2u + 1u];
            if (!isfinite(x) || !isfinite(y)) {
                continue;
            }
            if (x < min_x) {
                min_x = x;
            }
            if (y < min_y) {
                min_y = y;
            }
            if (x > max_x) {
                max_x = x;
            }
            if (y > max_y) {
                max_y = y;
            }
        }
    }
    if (!isfinite(min_x) || !isfinite(min_y) || !isfinite(max_x) || !isfinite(max_y)) {
        return false;
    }
    *out_min_x = min_x;
    *out_min_y = min_y;
    *out_max_x = max_x;
    *out_max_y = max_y;
    return true;
}

static bool fs_draw_linear_gradient_rect_cells(
    FS_Core* core,
    float x,
    float y,
    float w,
    float h,
    const FS_StyleLinearGradient* grad
) {
    if (!core || !grad || grad->stop_count < 2u || !isfinite(x) || !isfinite(y) || !isfinite(w) || !isfinite(h)) {
        return false;
    }
    float x0 = x;
    float y0 = y;
    float x1 = x + w;
    float y1 = y + h;
    if (x1 < x0) {
        float t = x0;
        x0 = x1;
        x1 = t;
    }
    if (y1 < y0) {
        float t = y0;
        y0 = y1;
        y1 = t;
    }
    const float rw = x1 - x0;
    const float rh = y1 - y0;
    if (rw <= 1e-6f || rh <= 1e-6f) {
        return true;
    }

    const float gdx = grad->x1 - grad->x0;
    const float gdy = grad->y1 - grad->y0;
    const float agx = fabsf(gdx);
    const float agy = fabsf(gdy);
    uint32_t nx = 1u;
    uint32_t ny = 1u;
    if (agx > agy * 2.0f) {
        nx = (uint32_t)ceilf(rw / 18.0f);
        if (nx < 4u) {
            nx = 4u;
        } else if (nx > 64u) {
            nx = 64u;
        }
    } else if (agy > agx * 2.0f) {
        ny = (uint32_t)ceilf(rh / 18.0f);
        if (ny < 4u) {
            ny = 4u;
        } else if (ny > 64u) {
            ny = 64u;
        }
    } else {
        nx = (uint32_t)ceilf(rw / 22.0f);
        ny = (uint32_t)ceilf(rh / 22.0f);
        if (nx < 3u) {
            nx = 3u;
        } else if (nx > 32u) {
            nx = 32u;
        }
        if (ny < 3u) {
            ny = 3u;
        } else if (ny > 32u) {
            ny = 32u;
        }
    }

    for (uint32_t iy = 0u; iy < ny; ++iy) {
        const float ya = y0 + rh * ((float)iy / (float)ny);
        const float yb = y0 + rh * ((float)(iy + 1u) / (float)ny);
        const float cy = (ya + yb) * 0.5f;
        for (uint32_t ix = 0u; ix < nx; ++ix) {
            const float xa = x0 + rw * ((float)ix / (float)nx);
            const float xb = x0 + rw * ((float)(ix + 1u) / (float)nx);
            const float cx = (xa + xb) * 0.5f;
            const uint32_t c = fs_linear_gradient_sample_rgba8(grad, cx, cy);
            if (!fs_cmd_rect(core, xa, ya, xb - xa, yb - ya, 0.0f, c)) {
                return false;
            }
        }
    }
    return true;
}


static bool fs_draw_radial_gradient_rect_cells(
    FS_Core* core,
    float x,
    float y,
    float w,
    float h,
    const FS_StyleRadialGradient* grad
) {
    if (!core || !grad || grad->stop_count < 2u || !isfinite(x) || !isfinite(y) || !isfinite(w) || !isfinite(h)) {
        return false;
    }
    float x0 = x;
    float y0 = y;
    float x1 = x + w;
    float y1 = y + h;
    if (x1 < x0) {
        float t = x0;
        x0 = x1;
        x1 = t;
    }
    if (y1 < y0) {
        float t = y0;
        y0 = y1;
        y1 = t;
    }
    const float rw = x1 - x0;
    const float rh = y1 - y0;
    if (rw <= 1e-6f || rh <= 1e-6f) {
        return true;
    }

    uint32_t nx = (uint32_t)ceilf(rw / 16.0f);
    uint32_t ny = (uint32_t)ceilf(rh / 16.0f);
    if (nx < 4u) {
        nx = 4u;
    } else if (nx > 72u) {
        nx = 72u;
    }
    if (ny < 4u) {
        ny = 4u;
    } else if (ny > 72u) {
        ny = 72u;
    }

    for (uint32_t iy = 0u; iy < ny; ++iy) {
        const float ya = y0 + rh * ((float)iy / (float)ny);
        const float yb = y0 + rh * ((float)(iy + 1u) / (float)ny);
        const float cy = (ya + yb) * 0.5f;
        for (uint32_t ix = 0u; ix < nx; ++ix) {
            const float xa = x0 + rw * ((float)ix / (float)nx);
            const float xb = x0 + rw * ((float)(ix + 1u) / (float)nx);
            const float cx = (xa + xb) * 0.5f;
            const uint32_t c = fs_radial_gradient_sample_rgba8(grad, cx, cy);
            if (!fs_cmd_rect(core, xa, ya, xb - xa, yb - ya, 0.0f, c)) {
                return false;
            }
        }
    }
    return true;
}

static bool fs_draw_conic_gradient_rect_cells(
    FS_Core* core,
    float x,
    float y,
    float w,
    float h,
    const FS_StyleConicGradient* grad
) {
    if (!core || !grad || grad->stop_count < 2u || !isfinite(x) || !isfinite(y) || !isfinite(w) || !isfinite(h)) {
        return false;
    }
    float x0 = x;
    float y0 = y;
    float x1 = x + w;
    float y1 = y + h;
    if (x1 < x0) {
        float t = x0;
        x0 = x1;
        x1 = t;
    }
    if (y1 < y0) {
        float t = y0;
        y0 = y1;
        y1 = t;
    }
    const float rw = x1 - x0;
    const float rh = y1 - y0;
    if (rw <= 1e-6f || rh <= 1e-6f) {
        return true;
    }

    uint32_t nx = (uint32_t)ceilf(rw / 16.0f);
    uint32_t ny = (uint32_t)ceilf(rh / 16.0f);
    if (nx < 4u) {
        nx = 4u;
    } else if (nx > 72u) {
        nx = 72u;
    }
    if (ny < 4u) {
        ny = 4u;
    } else if (ny > 72u) {
        ny = 72u;
    }

    for (uint32_t iy = 0u; iy < ny; ++iy) {
        const float ya = y0 + rh * ((float)iy / (float)ny);
        const float yb = y0 + rh * ((float)(iy + 1u) / (float)ny);
        const float cy = (ya + yb) * 0.5f;
        for (uint32_t ix = 0u; ix < nx; ++ix) {
            const float xa = x0 + rw * ((float)ix / (float)nx);
            const float xb = x0 + rw * ((float)(ix + 1u) / (float)nx);
            const float cx = (xa + xb) * 0.5f;
            const uint32_t c = fs_conic_gradient_sample_rgba8(grad, cx, cy);
            if (!fs_cmd_rect(core, xa, ya, xb - xa, yb - ya, 0.0f, c)) {
                return false;
            }
        }
    }
    return true;
}

static void fs_command_translate(FS_Command* cmd, float dx, float dy) {
    if (!cmd || (!isfinite(dx) && !isfinite(dy))) {
        return;
    }
    if (!isfinite(dx)) {
        dx = 0.0f;
    }
    if (!isfinite(dy)) {
        dy = 0.0f;
    }
    if (fabsf(dx) < 1e-7f && fabsf(dy) < 1e-7f) {
        return;
    }

    switch (cmd->type) {
        case FS_CMD_RECT:
        case FS_CMD_RECT_STROKE:
        case FS_CMD_IMAGE:
        case FS_CMD_TEXT:
            cmd->p0[0] += dx;
            cmd->p0[1] += dy;
            break;
        case FS_CMD_LINE:
        case FS_CMD_PATH_SEGMENT:
            cmd->p0[0] += dx;
            cmd->p0[1] += dy;
            cmd->p0[2] += dx;
            cmd->p0[3] += dy;
            break;
        case FS_CMD_CIRCLE:
        case FS_CMD_ELLIPSE:
        case FS_CMD_ARC:
            cmd->p0[0] += dx;
            cmd->p0[1] += dy;
            break;
        case FS_CMD_BEZIER_QUAD:
            cmd->p0[0] += dx;
            cmd->p0[1] += dy;
            cmd->p0[2] += dx;
            cmd->p0[3] += dy;
            cmd->p1[0] += dx;
            cmd->p1[1] += dy;
            break;
        case FS_CMD_BEZIER_CUBIC:
            cmd->p0[0] += dx;
            cmd->p0[1] += dy;
            cmd->p0[2] += dx;
            cmd->p0[3] += dy;
            cmd->p1[0] += dx;
            cmd->p1[1] += dy;
            cmd->p1[2] += dx;
            cmd->p1[3] += dy;
            break;
        case FS_CMD_TRIANGLE:
            cmd->p0[0] += dx;
            cmd->p0[1] += dy;
            cmd->p0[2] += dx;
            cmd->p0[3] += dy;
            cmd->p1[0] += dx;
            cmd->p1[1] += dy;
            break;
        default:
            cmd->p0[0] += dx;
            cmd->p0[1] += dy;
            break;
    }

    cmd->quad0[0] += dx;
    cmd->quad0[1] += dy;
}

static bool fs_emit_shadow_commands(
    FS_Core* core,
    const FS_Command* cmd,
    const FS_InternalState* st,
    uint32_t shadow_color,
    uint32_t shadow_blur_bits
) {
    if (!core || !cmd || !st) {
        return false;
    }

    const float offset_x = st->style_shadow_offset_x;
    const float offset_y = st->style_shadow_offset_y;
    float blur_px = st->style_shadow_blur;
    if (!isfinite(blur_px) || blur_px < 0.0f) {
        blur_px = 0.0f;
    }

    if (blur_px < FS_SHADOW_SEPARABLE_THRESHOLD) {
        FS_Command shadow_cmd = *cmd;
        shadow_cmd.flags &= ~FS_RENDER_FLAG_SHADOW_BLUR_MASK;
        shadow_cmd.flags &= ~FS_RENDER_FLAG_PATTERN_SHADE;
        shadow_cmd.flags |= FS_RENDER_FLAG_SHADOW | shadow_blur_bits;
        shadow_cmd.color_rgba8 = shadow_color;
        fs_command_translate(&shadow_cmd, offset_x, offset_y);
        return fs_push_command(core, &shadow_cmd);
    }

    static const float k_tap_offsets[3] = {-1.0f, 0.0f, 1.0f};
    static const float k_tap_weights[3] = {0.25f, 0.50f, 0.25f};
    const float tap_step = fmaxf(blur_px * FS_SHADOW_SEPARABLE_STEP_SCALE, 1.0f);
    const uint32_t tap_blur_bits = fs_shadow_blur_to_flag_bits(fmaxf(blur_px * 0.25f, 1.0f));

    for (uint32_t iy = 0u; iy < 3u; ++iy) {
        for (uint32_t ix = 0u; ix < 3u; ++ix) {
            const float weight = k_tap_weights[ix] * k_tap_weights[iy];
            if (weight <= 1e-6f) {
                continue;
            }
            FS_Command shadow_cmd = *cmd;
            shadow_cmd.flags &= ~FS_RENDER_FLAG_SHADOW_BLUR_MASK;
            shadow_cmd.flags &= ~FS_RENDER_FLAG_PATTERN_SHADE;
            shadow_cmd.flags |= FS_RENDER_FLAG_SHADOW | tap_blur_bits;
            shadow_cmd.color_rgba8 = fs_color_scale_alpha_rgba8(shadow_color, weight);
            if (((shadow_cmd.color_rgba8 >> 24u) & 0xFFu) == 0u) {
                continue;
            }
            fs_command_translate(
                &shadow_cmd,
                offset_x + k_tap_offsets[ix] * tap_step,
                offset_y + k_tap_offsets[iy] * tap_step
            );
            if (!fs_push_command(core, &shadow_cmd)) {
                return false;
            }
        }
    }
    return true;
}

static uint32_t fs_composite_op_to_pipeline_index(FS_GlobalCompositeOperation op) {
    switch (op) {
        case FS_GLOBAL_COMPOSITE_COPY:
            return 1u;
        case FS_GLOBAL_COMPOSITE_LIGHTER:
            return 2u;
        case FS_GLOBAL_COMPOSITE_DESTINATION_OVER:
            return 3u;
        case FS_GLOBAL_COMPOSITE_SOURCE_IN:
            return 4u;
        case FS_GLOBAL_COMPOSITE_SOURCE_OUT:
            return 5u;
        case FS_GLOBAL_COMPOSITE_DESTINATION_IN:
            return 6u;
        case FS_GLOBAL_COMPOSITE_DESTINATION_OUT:
            return 7u;
        case FS_GLOBAL_COMPOSITE_XOR:
            return 8u;
        case FS_GLOBAL_COMPOSITE_SOURCE_ATOP:
            return 9u;
        case FS_GLOBAL_COMPOSITE_DESTINATION_ATOP:
            return 10u;
        case FS_GLOBAL_COMPOSITE_SOURCE_OVER:
        default:
            return 0u;
    }
}

static WGPUBlendState fs_make_blend_state_for_pipeline(uint32_t pipeline_index) {
    WGPUBlendState blend;
#if FS_RENDER_PREMULTIPLIED_ALPHA
    switch (pipeline_index) {
        case 1u: // copy
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_One;
            blend.color.dstFactor = WGPUBlendFactor_Zero;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_One;
            blend.alpha.dstFactor = WGPUBlendFactor_Zero;
            break;
        case 2u: // lighter
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_One;
            blend.color.dstFactor = WGPUBlendFactor_One;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_One;
            blend.alpha.dstFactor = WGPUBlendFactor_One;
            break;
        case 3u: // destination-over
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_One;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_One;
            break;
        case 4u: // source-in
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_DstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_Zero;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_DstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_Zero;
            break;
        case 5u: // source-out
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_Zero;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_Zero;
            break;
        case 6u: // destination-in
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_Zero;
            blend.color.dstFactor = WGPUBlendFactor_SrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_Zero;
            blend.alpha.dstFactor = WGPUBlendFactor_SrcAlpha;
            break;
        case 7u: // destination-out
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_Zero;
            blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_Zero;
            blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            break;
        case 8u: // xor
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            break;
        case 9u: // source-atop
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_DstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_DstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            break;
        case 10u: // destination-atop
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_SrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_SrcAlpha;
            break;
        default: // source-over
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_One;
            blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_One;
            blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            break;
    }
#else
    switch (pipeline_index) {
        case 1u: // copy
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
            blend.color.dstFactor = WGPUBlendFactor_Zero;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_One;
            blend.alpha.dstFactor = WGPUBlendFactor_Zero;
            break;
        case 2u: // lighter
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
            blend.color.dstFactor = WGPUBlendFactor_One;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_One;
            blend.alpha.dstFactor = WGPUBlendFactor_One;
            break;
        case 3u: // destination-over
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_One;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_One;
            break;
        case 4u: // source-in
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_DstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_Zero;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_DstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_Zero;
            break;
        case 5u: // source-out
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_Zero;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_Zero;
            break;
        case 6u: // destination-in
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_Zero;
            blend.color.dstFactor = WGPUBlendFactor_SrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_Zero;
            blend.alpha.dstFactor = WGPUBlendFactor_SrcAlpha;
            break;
        case 7u: // destination-out
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_Zero;
            blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_Zero;
            blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            break;
        case 8u: // xor
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            break;
        case 9u: // source-atop
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_DstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_DstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            break;
        case 10u: // destination-atop
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.color.dstFactor = WGPUBlendFactor_SrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_OneMinusDstAlpha;
            blend.alpha.dstFactor = WGPUBlendFactor_SrcAlpha;
            break;
        default: // source-over
            blend.color.operation = WGPUBlendOperation_Add;
            blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
            blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            blend.alpha.operation = WGPUBlendOperation_Add;
            blend.alpha.srcFactor = WGPUBlendFactor_One;
            blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
            break;
    }
#endif
    return blend;
}

static void fs_mark_context_lost(FS_Core* core) {
    if (core) {
        core->context_lost = true;
    }
}

static bool fs_emit_styled_line_segment(
    FS_Core* core,
    float x0,
    float y0,
    float x1,
    float y1,
    float width,
    uint32_t color,
    uint8_t line_cap
) {
    return fs_emit_styled_line_segment_with_flags(core, x0, y0, x1, y1, width, color, line_cap, 0u);
}

static bool fs_emit_styled_line_segment_with_flags(
    FS_Core* core,
    float x0,
    float y0,
    float x1,
    float y1,
    float width,
    uint32_t color,
    uint8_t line_cap,
    uint32_t extra_line_flags
) {
    if (!core || width <= 0.0f) {
        return false;
    }
    float sx0 = x0;
    float sy0 = y0;
    float sx1 = x1;
    float sy1 = y1;
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float len = sqrtf(dx * dx + dy * dy);
    if (len <= 1e-6f) {
        return true;
    }
    if (line_cap == (uint8_t)FS_LINE_CAP_SQUARE) {
        const float ex = (dx / len) * (width * 0.5f);
        const float ey = (dy / len) * (width * 0.5f);
        sx0 -= ex;
        sy0 -= ey;
        sx1 += ex;
        sy1 += ey;
    }
    uint32_t seg_flags = 0u;
    if (line_cap == (uint8_t)FS_LINE_CAP_BUTT || line_cap == (uint8_t)FS_LINE_CAP_SQUARE) {
        seg_flags |= FS_LINE_FLAG_BUTT;
    }
    seg_flags |= (extra_line_flags & (FS_LINE_FLAG_NO_AA_START | FS_LINE_FLAG_NO_AA_END | FS_RENDER_FLAG_PATTERN_SHADE));
    return fs_cmd_path_segment_with_flags(core, sx0, sy0, sx1, sy1, width, color, seg_flags);
}

static bool fs_emit_dashed_line_segment(
    FS_Core* core,
    float x0,
    float y0,
    float x1,
    float y1,
    float width,
    uint32_t color,
    uint8_t line_cap,
    const float* dash,
    uint32_t dash_count,
    float dash_total,
    float* io_phase,
    uint32_t extra_render_flags
) {
    if (!core || !dash || dash_count == 0u || dash_total <= 1e-6f || !io_phase) {
        return false;
    }
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float len = sqrtf(dx * dx + dy * dy);
    if (len <= 1e-6f) {
        return true;
    }
    const float dir_x = dx / len;
    const float dir_y = dy / len;

    const uint32_t period_count = (dash_count & 1u) ? (dash_count * 2u) : dash_count;
    float phase = fmodf(*io_phase, dash_total);
    if (phase < 0.0f) {
        phase += dash_total;
    }

    uint32_t idx = 0u;
    float seg_pos = 0.0f;
    while (idx < period_count) {
        const float seg_len = dash[idx % dash_count];
        if (phase < seg_pos + seg_len || idx + 1u == period_count) {
            break;
        }
        seg_pos += seg_len;
        idx += 1u;
    }
    float dash_cursor = phase - seg_pos;
    if (dash_cursor < 0.0f) {
        dash_cursor = 0.0f;
    }
    float dist_cursor = 0.0f;
    uint32_t dash_idx = idx;

    while (dist_cursor < len - 1e-6f) {
        float seg_len = dash[dash_idx % dash_count];
        if (seg_len <= 1e-6f) {
            dash_idx = (dash_idx + 1u) % period_count;
            dash_cursor = 0.0f;
            continue;
        }
        const float remain_dash = seg_len - dash_cursor;
        if (remain_dash <= 1e-6f) {
            dash_idx = (dash_idx + 1u) % period_count;
            dash_cursor = 0.0f;
            continue;
        }
        float step = remain_dash;
        const float remain_line = len - dist_cursor;
        if (step > remain_line) {
            step = remain_line;
        }
        const bool draw = ((dash_idx & 1u) == 0u);
        if (draw && step > 1e-6f) {
            const float seg0 = dist_cursor;
            const float seg1 = dist_cursor + step;
            const float sx0 = x0 + dir_x * seg0;
            const float sy0 = y0 + dir_y * seg0;
            const float sx1 = x0 + dir_x * seg1;
            const float sy1 = y0 + dir_y * seg1;
            if (!fs_emit_styled_line_segment_with_flags(
                    core,
                    sx0,
                    sy0,
                    sx1,
                    sy1,
                    width,
                    color,
                    line_cap,
                    extra_render_flags
                )) {
                return false;
            }
        }
        dist_cursor += step;
        dash_cursor += step;
        if (dash_cursor >= seg_len - 1e-6f) {
            dash_idx = (dash_idx + 1u) % period_count;
            dash_cursor = 0.0f;
        }
    }

    *io_phase = phase + len;
    if (*io_phase >= dash_total) {
        *io_phase = fmodf(*io_phase, dash_total);
    }
    return true;
}

static void fs_eval_quad_point(
    float x0,
    float y0,
    float cx,
    float cy,
    float x1,
    float y1,
    float t,
    float* out_x,
    float* out_y
) {
    const float u = 1.0f - t;
    const float tt = t * t;
    const float uu = u * u;
    if (out_x) {
        *out_x = uu * x0 + 2.0f * u * t * cx + tt * x1;
    }
    if (out_y) {
        *out_y = uu * y0 + 2.0f * u * t * cy + tt * y1;
    }
}

static void fs_eval_cubic_point(
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
) {
    const float u = 1.0f - t;
    const float tt = t * t;
    const float uu = u * u;
    const float ttt = tt * t;
    const float uuu = uu * u;
    if (out_x) {
        *out_x = uuu * x0 + 3.0f * uu * t * cx0 + 3.0f * u * tt * cx1 + ttt * x1;
    }
    if (out_y) {
        *out_y = uuu * y0 + 3.0f * uu * t * cy0 + 3.0f * u * tt * cy1 + ttt * y1;
    }
}

static float fs_transform_metric_scale_cpu(const FS_Transform2D* t) {
    if (!t) {
        return 1.0f;
    }
    const float sx = hypotf(t->a, t->b);
    const float sy = hypotf(t->c, t->d);
    const float s = sx * sy;
    return sqrtf(fmaxf(s, 1e-8f));
}

static float fs_distance_sq_point_segment(
    float px,
    float py,
    float x0,
    float y0,
    float x1,
    float y1,
    float* out_t
) {
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float len_sq = dx * dx + dy * dy;
    if (len_sq <= 1e-12f) {
        if (out_t) {
            *out_t = 0.0f;
        }
        const float ex = px - x0;
        const float ey = py - y0;
        return ex * ex + ey * ey;
    }
    float t = ((px - x0) * dx + (py - y0) * dy) / len_sq;
    if (t < 0.0f) {
        t = 0.0f;
    } else if (t > 1.0f) {
        t = 1.0f;
    }
    if (out_t) {
        *out_t = t;
    }
    const float qx = x0 + dx * t;
    const float qy = y0 + dy * t;
    const float ex = px - qx;
    const float ey = py - qy;
    return ex * ex + ey * ey;
}

static bool fs_point_in_triangle(
    float px,
    float py,
    float ax,
    float ay,
    float bx,
    float by,
    float cx,
    float cy,
    float eps
) {
    const float c0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    const float c1 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
    const float c2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
    const bool has_neg = (c0 < -eps) || (c1 < -eps) || (c2 < -eps);
    const bool has_pos = (c0 > eps) || (c1 > eps) || (c2 > eps);
    return !(has_neg && has_pos);
}

static bool fs_hit_segment_stroke(
    float px,
    float py,
    float x0,
    float y0,
    float x1,
    float y1,
    float half_width,
    uint8_t line_cap
) {
    if (!isfinite(px) || !isfinite(py) ||
        !isfinite(x0) || !isfinite(y0) || !isfinite(x1) || !isfinite(y1) ||
        !isfinite(half_width) || half_width <= 0.0f) {
        return false;
    }

    if (line_cap == (uint8_t)FS_LINE_CAP_SQUARE) {
        const float dx = x1 - x0;
        const float dy = y1 - y0;
        const float len = hypotf(dx, dy);
        if (len > 1e-6f) {
            const float ex = (dx / len) * half_width;
            const float ey = (dy / len) * half_width;
            x0 -= ex;
            y0 -= ey;
            x1 += ex;
            y1 += ey;
        }
    }

    const float eps = fmaxf(1e-5f, half_width * 1e-4f);
    const float dist_sq = fs_distance_sq_point_segment(px, py, x0, y0, x1, y1, NULL);
    const float radius = half_width + eps;
    return dist_sq <= radius * radius;
}

static bool fs_hit_dashed_segment_stroke(
    float px,
    float py,
    float x0,
    float y0,
    float x1,
    float y1,
    float half_width,
    uint8_t line_cap,
    const float* dash,
    uint32_t dash_count,
    float dash_total,
    float* io_phase
) {
    if (!dash || dash_count == 0u || dash_total <= 1e-6f || !io_phase) {
        return false;
    }
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float len = sqrtf(dx * dx + dy * dy);
    if (len <= 1e-6f) {
        return false;
    }
    const float dir_x = dx / len;
    const float dir_y = dy / len;

    const uint32_t period_count = (dash_count & 1u) ? (dash_count * 2u) : dash_count;
    float phase = fmodf(*io_phase, dash_total);
    if (phase < 0.0f) {
        phase += dash_total;
    }

    uint32_t idx = 0u;
    float seg_pos = 0.0f;
    while (idx < period_count) {
        const float seg_len = dash[idx % dash_count];
        if (phase < seg_pos + seg_len || idx + 1u == period_count) {
            break;
        }
        seg_pos += seg_len;
        idx += 1u;
    }
    float dash_cursor = phase - seg_pos;
    if (dash_cursor < 0.0f) {
        dash_cursor = 0.0f;
    }
    float dist_cursor = 0.0f;
    uint32_t dash_idx = idx;

    while (dist_cursor < len - 1e-6f) {
        float seg_len = dash[dash_idx % dash_count];
        if (seg_len <= 1e-6f) {
            dash_idx = (dash_idx + 1u) % period_count;
            dash_cursor = 0.0f;
            continue;
        }
        const float remain_dash = seg_len - dash_cursor;
        if (remain_dash <= 1e-6f) {
            dash_idx = (dash_idx + 1u) % period_count;
            dash_cursor = 0.0f;
            continue;
        }
        float step = remain_dash;
        const float remain_line = len - dist_cursor;
        if (step > remain_line) {
            step = remain_line;
        }
        const bool draw = ((dash_idx & 1u) == 0u);
        if (draw && step > 1e-6f) {
            const float seg0 = dist_cursor;
            const float seg1 = dist_cursor + step;
            const float sx0 = x0 + dir_x * seg0;
            const float sy0 = y0 + dir_y * seg0;
            const float sx1 = x0 + dir_x * seg1;
            const float sy1 = y0 + dir_y * seg1;
            if (fs_hit_segment_stroke(px, py, sx0, sy0, sx1, sy1, half_width, line_cap)) {
                return true;
            }
        }
        dist_cursor += step;
        dash_cursor += step;
        if (dash_cursor >= seg_len - 1e-6f) {
            dash_idx = (dash_idx + 1u) % period_count;
            dash_cursor = 0.0f;
        }
    }

    *io_phase = phase + len;
    if (*io_phase >= dash_total) {
        *io_phase = fmodf(*io_phase, dash_total);
    }
    return false;
}

static bool fs_hit_path_join(
    float px,
    float py,
    float join_x,
    float join_y,
    float in_dx,
    float in_dy,
    float out_dx,
    float out_dy,
    float stroke_width,
    uint8_t line_join,
    float miter_limit
) {
    if (stroke_width <= 0.0f) {
        return false;
    }
    if (!fs_vec2_normalize(in_dx, in_dy, &in_dx, &in_dy) || !fs_vec2_normalize(out_dx, out_dy, &out_dx, &out_dy)) {
        return false;
    }
    const float dot = in_dx * out_dx + in_dy * out_dy;
    if (dot > 0.9995f) {
        return false;
    }
    const float turn = in_dx * out_dy - in_dy * out_dx;
    if (fabsf(turn) <= 1e-5f) {
        return false;
    }

    const float hw = stroke_width * 0.5f;
    const float side = (turn > 0.0f) ? -1.0f : 1.0f;
    const float nin_x = side * (-in_dy);
    const float nin_y = side * in_dx;
    const float nout_x = side * (-out_dy);
    const float nout_y = side * out_dx;
    const float ax = join_x + nin_x * hw;
    const float ay = join_y + nin_y * hw;
    const float bx = join_x + nout_x * hw;
    const float by = join_y + nout_y * hw;
    const float eps = fmaxf(1e-4f, hw * 1e-4f);

    if (line_join == (uint8_t)FS_LINE_JOIN_ROUND) {
        const float dx = px - join_x;
        const float dy = py - join_y;
        const float r = hw + eps;
        return dx * dx + dy * dy <= r * r;
    }

    if (line_join == (uint8_t)FS_LINE_JOIN_BEVEL) {
        return fs_point_in_triangle(px, py, join_x, join_y, ax, ay, bx, by, eps);
    }

    if (line_join == (uint8_t)FS_LINE_JOIN_MITER) {
        const float denom = in_dx * out_dy - in_dy * out_dx;
        if (fabsf(denom) > 1e-6f) {
            const float qpx = bx - ax;
            const float qpy = by - ay;
            const float t = (qpx * out_dy - qpy * out_dx) / denom;
            const float mx = ax + in_dx * t;
            const float my = ay + in_dy * t;
            float resolved_limit = miter_limit;
            if (!isfinite(resolved_limit) || resolved_limit <= 0.0f) {
                resolved_limit = 10.0f;
            }
            if (resolved_limit < 1.0f) {
                resolved_limit = 1.0f;
            }
            const float miter_len = hypotf(mx - join_x, my - join_y) / fmaxf(hw, 1e-6f);
            if (isfinite(miter_len) && miter_len <= resolved_limit) {
                if (fs_point_in_triangle(px, py, join_x, join_y, ax, ay, mx, my, eps)) {
                    return true;
                }
                if (fs_point_in_triangle(px, py, join_x, join_y, mx, my, bx, by, eps)) {
                    return true;
                }
                return false;
            }
        }
        return fs_point_in_triangle(px, py, join_x, join_y, ax, ay, bx, by, eps);
    }

    return false;
}

static void fs_hit_fill_accumulate_edge(FS_HitFillContext* ctx, float x0, float y0, float x1, float y1) {
    if (!ctx || ctx->on_edge) {
        return;
    }
    const float eps = ctx->edge_epsilon;
    const float dist_sq = fs_distance_sq_point_segment(ctx->px, ctx->py, x0, y0, x1, y1, NULL);
    if (dist_sq <= eps * eps) {
        ctx->on_edge = true;
        return;
    }

    const bool upward = (y0 <= ctx->py) && (y1 > ctx->py);
    const bool downward = (y0 > ctx->py) && (y1 <= ctx->py);
    if (!(upward || downward)) {
        return;
    }
    const float dy = y1 - y0;
    if (fabsf(dy) <= 1e-8f) {
        return;
    }
    const float x_intersect = x0 + (ctx->py - y0) * (x1 - x0) / dy;
    if (fabsf(x_intersect - ctx->px) <= eps) {
        ctx->on_edge = true;
        return;
    }
    if (x_intersect > ctx->px) {
        if (ctx->evenodd) {
            ctx->parity ^= 1u;
        } else {
            ctx->winding += upward ? 1 : -1;
        }
    }
}

static bool fs_hit_test_fill_path_device(
    const FS_PathSegment* segments,
    uint32_t segment_count,
    const FS_Transform2D* transform,
    float px,
    float py,
    FS_FillRule fill_rule
) {
    if (!segments || segment_count == 0u || !isfinite(px) || !isfinite(py)) {
        return false;
    }

    FS_HitFillContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.px = px;
    ctx.py = py;
    ctx.edge_epsilon = 1e-4f;
    ctx.evenodd = (fill_rule == FS_FILL_RULE_EVENODD);

    bool have_prev_end = false;
    bool have_subpath = false;
    float prev_end_x = 0.0f;
    float prev_end_y = 0.0f;
    float subpath_start_x = 0.0f;
    float subpath_start_y = 0.0f;

    for (uint32_t i = 0u; i < segment_count; ++i) {
        const FS_PathSegment* seg = &segments[i];
        const bool contour_break =
            !have_prev_end ||
            fabsf(prev_end_x - seg->x0) > 1e-4f ||
            fabsf(prev_end_y - seg->y0) > 1e-4f;

        if (contour_break) {
            if (have_subpath) {
                float ex0 = 0.0f;
                float ey0 = 0.0f;
                float ex1 = 0.0f;
                float ey1 = 0.0f;
                fs_transform_apply_point(transform, prev_end_x, prev_end_y, &ex0, &ey0);
                fs_transform_apply_point(transform, subpath_start_x, subpath_start_y, &ex1, &ey1);
                fs_hit_fill_accumulate_edge(&ctx, ex0, ey0, ex1, ey1);
                if (ctx.on_edge) {
                    return true;
                }
            }
            subpath_start_x = seg->x0;
            subpath_start_y = seg->y0;
            have_subpath = true;
        }

        if (seg->type == (uint8_t)FS_PATH_SEG_LINE) {
            float x0 = 0.0f;
            float y0 = 0.0f;
            float x1 = 0.0f;
            float y1 = 0.0f;
            fs_transform_apply_point(transform, seg->x0, seg->y0, &x0, &y0);
            fs_transform_apply_point(transform, seg->x1, seg->y1, &x1, &y1);
            fs_hit_fill_accumulate_edge(&ctx, x0, y0, x1, y1);
            if (ctx.on_edge) {
                return true;
            }
        } else if (seg->type == (uint8_t)FS_PATH_SEG_QUAD) {
            const float len_a = hypotf(seg->cx0 - seg->x0, seg->cy0 - seg->y0);
            const float len_b = hypotf(seg->x1 - seg->cx0, seg->y1 - seg->cy0);
            uint32_t steps = (uint32_t)((len_a + len_b) / 14.0f) + 8u;
            if (steps < 8u) {
                steps = 8u;
            } else if (steps > 96u) {
                steps = 96u;
            }
            float prev_x = seg->x0;
            float prev_y = seg->y0;
            for (uint32_t s = 1u; s <= steps; ++s) {
                const float u = (float)s / (float)steps;
                float cur_x = 0.0f;
                float cur_y = 0.0f;
                fs_eval_quad_point(seg->x0, seg->y0, seg->cx0, seg->cy0, seg->x1, seg->y1, u, &cur_x, &cur_y);
                float tx0 = 0.0f;
                float ty0 = 0.0f;
                float tx1 = 0.0f;
                float ty1 = 0.0f;
                fs_transform_apply_point(transform, prev_x, prev_y, &tx0, &ty0);
                fs_transform_apply_point(transform, cur_x, cur_y, &tx1, &ty1);
                fs_hit_fill_accumulate_edge(&ctx, tx0, ty0, tx1, ty1);
                if (ctx.on_edge) {
                    return true;
                }
                prev_x = cur_x;
                prev_y = cur_y;
            }
        } else if (seg->type == (uint8_t)FS_PATH_SEG_CUBIC) {
            const float len_a = hypotf(seg->cx0 - seg->x0, seg->cy0 - seg->y0);
            const float len_b = hypotf(seg->cx1 - seg->cx0, seg->cy1 - seg->cy0);
            const float len_c = hypotf(seg->x1 - seg->cx1, seg->y1 - seg->cy1);
            uint32_t steps = (uint32_t)((len_a + len_b + len_c) / 12.0f) + 10u;
            if (steps < 10u) {
                steps = 10u;
            } else if (steps > 144u) {
                steps = 144u;
            }
            float prev_x = seg->x0;
            float prev_y = seg->y0;
            for (uint32_t s = 1u; s <= steps; ++s) {
                const float u = (float)s / (float)steps;
                float cur_x = 0.0f;
                float cur_y = 0.0f;
                fs_eval_cubic_point(
                    seg->x0,
                    seg->y0,
                    seg->cx0,
                    seg->cy0,
                    seg->cx1,
                    seg->cy1,
                    seg->x1,
                    seg->y1,
                    u,
                    &cur_x,
                    &cur_y
                );
                float tx0 = 0.0f;
                float ty0 = 0.0f;
                float tx1 = 0.0f;
                float ty1 = 0.0f;
                fs_transform_apply_point(transform, prev_x, prev_y, &tx0, &ty0);
                fs_transform_apply_point(transform, cur_x, cur_y, &tx1, &ty1);
                fs_hit_fill_accumulate_edge(&ctx, tx0, ty0, tx1, ty1);
                if (ctx.on_edge) {
                    return true;
                }
                prev_x = cur_x;
                prev_y = cur_y;
            }
        }

        prev_end_x = seg->x1;
        prev_end_y = seg->y1;
        have_prev_end = true;
    }

    if (have_subpath) {
        float ex0 = 0.0f;
        float ey0 = 0.0f;
        float ex1 = 0.0f;
        float ey1 = 0.0f;
        fs_transform_apply_point(transform, prev_end_x, prev_end_y, &ex0, &ey0);
        fs_transform_apply_point(transform, subpath_start_x, subpath_start_y, &ex1, &ey1);
        fs_hit_fill_accumulate_edge(&ctx, ex0, ey0, ex1, ey1);
    }

    if (ctx.on_edge) {
        return true;
    }
    if (ctx.evenodd) {
        return (ctx.parity & 1u) != 0u;
    }
    return ctx.winding != 0;
}

static bool fs_hit_test_stroke_path_device(
    const FS_PathSegment* segments,
    uint32_t segment_count,
    const FS_Transform2D* transform,
    uint8_t line_cap,
    uint8_t line_join,
    float miter_limit,
    const float* dash,
    uint32_t dash_count,
    float dash_offset,
    float stroke_width,
    float px,
    float py
) {
    if (!segments || segment_count == 0u ||
        !isfinite(px) || !isfinite(py) ||
        !isfinite(stroke_width) || stroke_width <= 0.0f) {
        return false;
    }

    const float device_scale = fs_transform_metric_scale_cpu(transform);
    const float stroke_width_device = stroke_width * device_scale;
    if (!isfinite(stroke_width_device) || stroke_width_device <= 0.0f) {
        return false;
    }
    const float half_width = stroke_width_device * 0.5f;

    bool use_dash = (dash && dash_count > 0u);
    float dash_total = 0.0f;
    float dash_phase = dash_offset;
    if (use_dash) {
        for (uint32_t i = 0u; i < dash_count; ++i) {
            dash_total += dash[i];
        }
        if (dash_total <= 1e-6f) {
            use_dash = false;
        }
    }

    bool has_prev_end = false;
    float prev_end_x = 0.0f;
    float prev_end_y = 0.0f;
    bool has_prev_end_dir = false;
    float prev_end_dx = 0.0f;
    float prev_end_dy = 0.0f;

    for (uint32_t i = 0u; i < segment_count; ++i) {
        const FS_PathSegment* seg = &segments[i];
        const bool connected =
            has_prev_end &&
            fabsf(prev_end_x - seg->x0) <= 1e-4f &&
            fabsf(prev_end_y - seg->y0) <= 1e-4f;

        float start_dx = 0.0f;
        float start_dy = 0.0f;
        const bool has_start_dir = fs_path_segment_start_dir(seg, &start_dx, &start_dy);
        if (connected && has_prev_end_dir && has_start_dir) {
            float join_x = 0.0f;
            float join_y = 0.0f;
            fs_transform_apply_point(transform, seg->x0, seg->y0, &join_x, &join_y);
            if (fs_hit_path_join(
                    px,
                    py,
                    join_x,
                    join_y,
                    prev_end_dx,
                    prev_end_dy,
                    start_dx,
                    start_dy,
                    stroke_width_device,
                    line_join,
                    miter_limit
                )) {
                return true;
            }
        }

        if (seg->type == (uint8_t)FS_PATH_SEG_LINE) {
            float x0 = 0.0f;
            float y0 = 0.0f;
            float x1 = 0.0f;
            float y1 = 0.0f;
            fs_transform_apply_point(transform, seg->x0, seg->y0, &x0, &y0);
            fs_transform_apply_point(transform, seg->x1, seg->y1, &x1, &y1);
            if (use_dash) {
                if (fs_hit_dashed_segment_stroke(
                        px,
                        py,
                        x0,
                        y0,
                        x1,
                        y1,
                        half_width,
                        line_cap,
                        dash,
                        dash_count,
                        dash_total,
                        &dash_phase
                    )) {
                    return true;
                }
            } else if (fs_hit_segment_stroke(px, py, x0, y0, x1, y1, half_width, line_cap)) {
                return true;
            }
        } else if (seg->type == (uint8_t)FS_PATH_SEG_QUAD || seg->type == (uint8_t)FS_PATH_SEG_CUBIC) {
            uint32_t steps = 0u;
            if (seg->type == (uint8_t)FS_PATH_SEG_QUAD) {
                const float len_a = hypotf(seg->cx0 - seg->x0, seg->cy0 - seg->y0);
                const float len_b = hypotf(seg->x1 - seg->cx0, seg->y1 - seg->cy0);
                steps = (uint32_t)((len_a + len_b) / 14.0f) + 8u;
                if (steps < 8u) {
                    steps = 8u;
                } else if (steps > 128u) {
                    steps = 128u;
                }
            } else {
                const float len_a = hypotf(seg->cx0 - seg->x0, seg->cy0 - seg->y0);
                const float len_b = hypotf(seg->cx1 - seg->cx0, seg->cy1 - seg->cy0);
                const float len_c = hypotf(seg->x1 - seg->cx1, seg->y1 - seg->cy1);
                steps = (uint32_t)((len_a + len_b + len_c) / 12.0f) + 10u;
                if (steps < 10u) {
                    steps = 10u;
                } else if (steps > 192u) {
                    steps = 192u;
                }
            }
            float prev_x = seg->x0;
            float prev_y = seg->y0;
            for (uint32_t s = 1u; s <= steps; ++s) {
                const float t = (float)s / (float)steps;
                float cur_x = 0.0f;
                float cur_y = 0.0f;
                if (seg->type == (uint8_t)FS_PATH_SEG_QUAD) {
                    fs_eval_quad_point(seg->x0, seg->y0, seg->cx0, seg->cy0, seg->x1, seg->y1, t, &cur_x, &cur_y);
                } else {
                    fs_eval_cubic_point(
                        seg->x0,
                        seg->y0,
                        seg->cx0,
                        seg->cy0,
                        seg->cx1,
                        seg->cy1,
                        seg->x1,
                        seg->y1,
                        t,
                        &cur_x,
                        &cur_y
                    );
                }
                float x0 = 0.0f;
                float y0 = 0.0f;
                float x1 = 0.0f;
                float y1 = 0.0f;
                fs_transform_apply_point(transform, prev_x, prev_y, &x0, &y0);
                fs_transform_apply_point(transform, cur_x, cur_y, &x1, &y1);
                if (use_dash) {
                    if (fs_hit_dashed_segment_stroke(
                            px,
                            py,
                            x0,
                            y0,
                            x1,
                            y1,
                            half_width,
                            line_cap,
                            dash,
                            dash_count,
                            dash_total,
                            &dash_phase
                        )) {
                        return true;
                    }
                } else {
                    uint8_t seg_cap = (uint8_t)FS_LINE_CAP_BUTT;
                    if (s == 1u || s == steps) {
                        seg_cap = line_cap;
                    }
                    if (fs_hit_segment_stroke(px, py, x0, y0, x1, y1, half_width, seg_cap)) {
                        return true;
                    }
                }
                prev_x = cur_x;
                prev_y = cur_y;
            }
        }

        prev_end_x = seg->x1;
        prev_end_y = seg->y1;
        has_prev_end = true;
        has_prev_end_dir = fs_path_segment_end_dir(seg, &prev_end_dx, &prev_end_dy);
    }

    return false;
}

static uint32_t fs_quantize_font_size(float font_px) {
    const float clamped = font_px < 1.0f ? 1.0f : font_px;
    const float q = clamped * 64.0f;
    return (uint32_t)(q + 0.5f);
}

static bool fs_release_compute_binding(FS_Core* core) {
    if (!core) {
        return false;
    }
    if (core->compute_bg) {
        wgpuBindGroupRelease(core->compute_bg);
        core->compute_bg = NULL;
    }
    return true;
}

static bool fs_clear_image_atlas_layer(FS_Core* core, uint32_t layer) {
    if (!core || layer >= core->image_atlas_layers) {
        return false;
    }
    const uint32_t row_bytes = core->image_atlas_width * 4u;
    const uint32_t padded_row = fs_align_up_u32(row_bytes, 256u);
    const size_t upload_size = (size_t)padded_row * (size_t)core->image_atlas_height;
    uint8_t* zero = (uint8_t*)calloc(1, upload_size);
    if (!zero) {
        return false;
    }
    WGPUTexelCopyTextureInfo dst = {
        .texture = core->image_atlas_texture,
        .mipLevel = 0,
        .origin = {0u, 0u, layer},
        .aspect = WGPUTextureAspect_All
    };
    WGPUTexelCopyBufferLayout layout = {
        .offset = 0,
        .bytesPerRow = padded_row,
        .rowsPerImage = core->image_atlas_height
    };
    WGPUExtent3D extent = {core->image_atlas_width, core->image_atlas_height, 1u};
    wgpuQueueWriteTexture(core->queue, &dst, zero, upload_size, &layout, &extent);
    if (fs_image_atlas_shadow_bounds_ok(core, layer, 0u, 0u, core->image_atlas_width, core->image_atlas_height)) {
        const size_t layer_px = (size_t)core->image_atlas_width * (size_t)core->image_atlas_height;
        memset(core->image_atlas_shadow_rgba + (size_t)layer * layer_px * 4u, 0, layer_px * 4u);
    }
    free(zero);
    return true;
}

static bool fs_alloc_image_slot(
    FS_Core* core,
    uint32_t width,
    uint32_t height,
    uint32_t* out_layer,
    uint32_t* out_x,
    uint32_t* out_y
) {
    if (!core || !out_layer || !out_x || !out_y) {
        return false;
    }
    for (uint32_t sweep = 0; sweep < core->image_atlas_layers; ++sweep) {
        const uint32_t layer = (core->image_atlas_active_layer + sweep) % core->image_atlas_layers;
        if (fs_alloc_from_atlas(
                core->image_atlas_width,
                core->image_atlas_height,
                &core->image_atlas_cursor_x[layer],
                &core->image_atlas_cursor_y[layer],
                &core->image_atlas_row_height[layer],
                FS_IMAGE_ATLAS_PADDING,
                width,
                height,
                out_x,
                out_y
            )) {
            core->image_atlas_active_layer = layer;
            *out_layer = layer;
            return true;
        }
    }

    const uint32_t recycle_layer = (core->image_atlas_active_layer + 1u) % core->image_atlas_layers;
    if (!fs_clear_image_atlas_layer(core, recycle_layer)) {
        return false;
    }
    core->image_atlas_cursor_x[recycle_layer] = 0u;
    core->image_atlas_cursor_y[recycle_layer] = 0u;
    core->image_atlas_row_height[recycle_layer] = 0u;
    core->image_atlas_generation[recycle_layer] += 1u;

    if (!fs_alloc_from_atlas(
            core->image_atlas_width,
            core->image_atlas_height,
            &core->image_atlas_cursor_x[recycle_layer],
            &core->image_atlas_cursor_y[recycle_layer],
            &core->image_atlas_row_height[recycle_layer],
            FS_IMAGE_ATLAS_PADDING,
            width,
            height,
            out_x,
            out_y
        )) {
        return false;
    }
    core->image_atlas_active_layer = recycle_layer;
    *out_layer = recycle_layer;
    return true;
}

static bool fs_recreate_compute_bind_group(FS_Core* core) {
    if (!core || !core->compute_bgl || !core->command_buffer || !core->command_state_buffer ||
        !core->vertex_buffer || !core->uniform_buffer || !core->clip_layer_uniform_buffer) {
        return false;
    }
    fs_release_compute_binding(core);
    WGPUBindGroupEntry entries[] = {
        {
            .binding = 0,
            .buffer = core->command_buffer,
            .offset = 0,
            .size = core->command_buffer_size
        },
        {
            .binding = 1,
            .buffer = core->command_state_buffer,
            .offset = 0,
            .size = core->command_state_buffer_size
        },
        {
            .binding = 2,
            .buffer = core->vertex_buffer,
            .offset = 0,
            .size = core->vertex_buffer_size
        },
        {
            .binding = 3,
            .buffer = core->uniform_buffer,
            .offset = 0,
            .size = sizeof(FS_Uniforms)
        },
        {
            .binding = 4,
            .buffer = core->clip_layer_uniform_buffer,
            .offset = 0,
            .size = sizeof(FS_ClipLayerUniforms)
        }
    };
    WGPUBindGroupDescriptor desc = {
        .nextInChain = NULL,
        .label = "FS Compute Bind Group",
        .layout = core->compute_bgl,
        .entryCount = 5,
        .entries = entries
    };
    core->compute_bg = wgpuDeviceCreateBindGroup(core->device, &desc);
    return core->compute_bg != NULL;
}

static bool fs_create_image_atlas(FS_Core* core) {
    if (!core) {
        return false;
    }
    core->image_atlas_width = FS_IMAGE_ATLAS_SIZE;
    core->image_atlas_height = FS_IMAGE_ATLAS_SIZE;
    core->image_atlas_layers = FS_IMAGE_ATLAS_MAX_LAYERS;
    core->image_atlas_active_layer = 0u;
    for (uint32_t i = 0; i < core->image_atlas_layers; ++i) {
        core->image_atlas_cursor_x[i] = 0u;
        core->image_atlas_cursor_y[i] = 0u;
        core->image_atlas_row_height[i] = 0u;
        core->image_atlas_generation[i] = 1u;
    }
    const size_t shadow_size =
        (size_t)core->image_atlas_width * (size_t)core->image_atlas_height * (size_t)core->image_atlas_layers * 4u;
    core->image_atlas_shadow_rgba = (uint8_t*)calloc(1u, shadow_size);
    core->image_atlas_shadow_size = core->image_atlas_shadow_rgba ? shadow_size : 0u;
    if (!core->image_atlas_shadow_rgba) {
        return false;
    }

    WGPUTextureDescriptor tex_desc = {
        .nextInChain = NULL,
        .label = "FS Image Atlas",
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        .dimension = WGPUTextureDimension_2D,
        .size = {core->image_atlas_width, core->image_atlas_height, core->image_atlas_layers},
        .format = WGPUTextureFormat_RGBA8Unorm,
        .mipLevelCount = 1,
        .sampleCount = 1,
        .viewFormatCount = 0,
        .viewFormats = NULL
    };
    core->image_atlas_texture = wgpuDeviceCreateTexture(core->device, &tex_desc);
    if (!core->image_atlas_texture) {
        return false;
    }

    WGPUTextureViewDescriptor view_desc = {
        .nextInChain = NULL,
        .label = "FS Image Atlas View",
        .format = WGPUTextureFormat_RGBA8Unorm,
        .dimension = WGPUTextureViewDimension_2DArray,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = core->image_atlas_layers,
        .aspect = WGPUTextureAspect_All
    };
    core->image_atlas_view = wgpuTextureCreateView(core->image_atlas_texture, &view_desc);
    if (!core->image_atlas_view) {
        return false;
    }

    WGPUSamplerDescriptor sampler_desc = {
        .nextInChain = NULL,
        .label = "FS Image Atlas Sampler",
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .addressModeW = WGPUAddressMode_ClampToEdge,
        .magFilter = WGPUFilterMode_Linear,
        .minFilter = WGPUFilterMode_Linear,
        .mipmapFilter = WGPUMipmapFilterMode_Linear,
        .lodMinClamp = 0.0f,
        .lodMaxClamp = (float)(core->glyph_atlas_mip_count - 1u),
        .maxAnisotropy = 1,
        .compare = WGPUCompareFunction_Undefined
    };
    core->image_atlas_sampler = wgpuDeviceCreateSampler(core->device, &sampler_desc);
    if (!core->image_atlas_sampler) {
        return false;
    }
    return true;
}

static bool fs_create_glyph_atlas(FS_Core* core) {
    if (!core) {
        return false;
    }
    core->glyph_atlas_width = FS_GLYPH_ATLAS_SIZE;
    core->glyph_atlas_height = FS_GLYPH_ATLAS_SIZE;
    core->glyph_atlas_mip_count = fs_compute_mip_count(core->glyph_atlas_width, core->glyph_atlas_height);
    core->glyph_atlas_cursor_x = 0u;
    core->glyph_atlas_cursor_y = 0u;
    core->glyph_atlas_row_height = 0u;

    WGPUTextureDescriptor tex_desc = {
        .nextInChain = NULL,
        .label = "FS Glyph Atlas",
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        .dimension = WGPUTextureDimension_2D,
        .size = {core->glyph_atlas_width, core->glyph_atlas_height, 1},
        .format = WGPUTextureFormat_RGBA8Unorm,
        .mipLevelCount = core->glyph_atlas_mip_count,
        .sampleCount = 1,
        .viewFormatCount = 0,
        .viewFormats = NULL
    };
    core->glyph_atlas_texture = wgpuDeviceCreateTexture(core->device, &tex_desc);
    if (!core->glyph_atlas_texture) {
        return false;
    }

    WGPUTextureViewDescriptor view_desc = {
        .nextInChain = NULL,
        .label = "FS Glyph Atlas View",
        .format = WGPUTextureFormat_RGBA8Unorm,
        .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = core->glyph_atlas_mip_count,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All
    };
    core->glyph_atlas_view = wgpuTextureCreateView(core->glyph_atlas_texture, &view_desc);
    if (!core->glyph_atlas_view) {
        return false;
    }

    WGPUSamplerDescriptor sampler_desc = {
        .nextInChain = NULL,
        .label = "FS Glyph Atlas Sampler",
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .addressModeW = WGPUAddressMode_ClampToEdge,
        .magFilter = WGPUFilterMode_Linear,
        .minFilter = WGPUFilterMode_Linear,
        .mipmapFilter = WGPUMipmapFilterMode_Linear,
        .lodMinClamp = 0.0f,
        .lodMaxClamp = 0.0f,
        .maxAnisotropy = 1,
        .compare = WGPUCompareFunction_Undefined
    };
    core->glyph_atlas_sampler = wgpuDeviceCreateSampler(core->device, &sampler_desc);
    if (!core->glyph_atlas_sampler) {
        return false;
    }
    return true;
}

static bool fs_create_clip_mask(FS_Core* core) {
    if (!core) {
        return false;
    }
    core->clip_mask_width = core->width > 0u ? core->width : 1u;
    core->clip_mask_height = core->height > 0u ? core->height : 1u;
    core->clip_mask_layers = FS_CLIP_MASK_LAYERS;
    if (core->clip_layer_reuse_reserve >= core->clip_mask_layers) {
        core->clip_layer_reuse_reserve = (core->clip_mask_layers > 0u) ? (core->clip_mask_layers - 1u) : 0u;
    }
    core->clip_mask_next_layer = 0u;

    if (!core->clip_mask_layer_has_data) {
        core->clip_mask_layer_has_data = (uint8_t*)calloc((size_t)core->clip_mask_layers, sizeof(uint8_t));
    }
    if (!core->clip_mask_layer_min_x) {
        core->clip_mask_layer_min_x = (uint32_t*)calloc((size_t)core->clip_mask_layers, sizeof(uint32_t));
    }
    if (!core->clip_mask_layer_min_y) {
        core->clip_mask_layer_min_y = (uint32_t*)calloc((size_t)core->clip_mask_layers, sizeof(uint32_t));
    }
    if (!core->clip_mask_layer_max_x) {
        core->clip_mask_layer_max_x = (uint32_t*)calloc((size_t)core->clip_mask_layers, sizeof(uint32_t));
    }
    if (!core->clip_mask_layer_max_y) {
        core->clip_mask_layer_max_y = (uint32_t*)calloc((size_t)core->clip_mask_layers, sizeof(uint32_t));
    }
    if (!core->clip_mask_layer_hash) {
        core->clip_mask_layer_hash = (uint64_t*)calloc((size_t)core->clip_mask_layers, sizeof(uint64_t));
    }
    if (!core->clip_mask_layer_hash_valid) {
        core->clip_mask_layer_hash_valid = (uint8_t*)calloc((size_t)core->clip_mask_layers, sizeof(uint8_t));
    }
    if (!core->clip_mask_layer_parent) {
        core->clip_mask_layer_parent = (uint32_t*)calloc((size_t)core->clip_mask_layers, sizeof(uint32_t));
    }
    if (!core->clip_mask_layer_last_used_frame) {
        core->clip_mask_layer_last_used_frame = (uint32_t*)calloc((size_t)core->clip_mask_layers, sizeof(uint32_t));
    }
    if (!core->clip_mask_layer_has_data || !core->clip_mask_layer_min_x || !core->clip_mask_layer_min_y ||
        !core->clip_mask_layer_max_x || !core->clip_mask_layer_max_y ||
        !core->clip_mask_layer_hash || !core->clip_mask_layer_hash_valid || !core->clip_mask_layer_parent ||
        !core->clip_mask_layer_last_used_frame) {
        return false;
    }
    memset(core->clip_mask_layer_has_data, 0, (size_t)core->clip_mask_layers * sizeof(uint8_t));
    memset(core->clip_mask_layer_hash_valid, 0, (size_t)core->clip_mask_layers * sizeof(uint8_t));
    memset(core->clip_mask_layer_last_used_frame, 0, (size_t)core->clip_mask_layers * sizeof(uint32_t));
    for (uint32_t i = 0u; i < core->clip_mask_layers; ++i) {
        core->clip_mask_layer_parent[i] = UINT32_MAX;
    }

    WGPUTextureDescriptor tex_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Mask",
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopyDst,
        .dimension = WGPUTextureDimension_2D,
        .size = {core->clip_mask_width, core->clip_mask_height, core->clip_mask_layers},
        .format = WGPUTextureFormat_RGBA8Unorm,
        .mipLevelCount = 1,
        .sampleCount = 1,
        .viewFormatCount = 0,
        .viewFormats = NULL
    };
    core->clip_mask_texture = wgpuDeviceCreateTexture(core->device, &tex_desc);
    if (!core->clip_mask_texture) {
        return false;
    }

    WGPUTextureViewDescriptor view_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Mask View",
        .format = WGPUTextureFormat_RGBA8Unorm,
        .dimension = WGPUTextureViewDimension_2DArray,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = core->clip_mask_layers,
        .aspect = WGPUTextureAspect_All
    };
    core->clip_mask_view = wgpuTextureCreateView(core->clip_mask_texture, &view_desc);
    if (!core->clip_mask_view) {
        return false;
    }

    WGPUSamplerDescriptor sampler_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Mask Sampler",
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .addressModeW = WGPUAddressMode_ClampToEdge,
        .magFilter = WGPUFilterMode_Linear,
        .minFilter = WGPUFilterMode_Linear,
        .mipmapFilter = WGPUMipmapFilterMode_Nearest,
        .lodMinClamp = 0.0f,
        .lodMaxClamp = 0.0f,
        .maxAnisotropy = 1,
        .compare = WGPUCompareFunction_Undefined
    };
    core->clip_mask_sampler = wgpuDeviceCreateSampler(core->device, &sampler_desc);
    if (!core->clip_mask_sampler) {
        return false;
    }
    return true;
}

static bool fs_create_msaa_color_target(FS_Core* core) {
    if (!core || !core->device) {
        return false;
    }
    if (core->msaa_color_view) {
        wgpuTextureViewRelease(core->msaa_color_view);
        core->msaa_color_view = NULL;
    }
    if (core->msaa_color_texture) {
        wgpuTextureRelease(core->msaa_color_texture);
        core->msaa_color_texture = NULL;
    }
    core->render_sample_count = 1u;
    if (FS_RENDER_MSAA_SAMPLES <= 1u) {
        return true;
    }
    const uint32_t tex_w = core->width > 0u ? core->width : 1u;
    const uint32_t tex_h = core->height > 0u ? core->height : 1u;
    WGPUTextureDescriptor tex_desc = {
        .nextInChain = NULL,
        .label = "FS MSAA Color",
        .usage = WGPUTextureUsage_RenderAttachment,
        .dimension = WGPUTextureDimension_2D,
        .size = {tex_w, tex_h, 1u},
        .format = core->target_format,
        .mipLevelCount = 1,
        .sampleCount = FS_RENDER_MSAA_SAMPLES,
        .viewFormatCount = 0,
        .viewFormats = NULL
    };
    core->msaa_color_texture = wgpuDeviceCreateTexture(core->device, &tex_desc);
    if (!core->msaa_color_texture) {
        return false;
    }
    WGPUTextureViewDescriptor view_desc = {
        .nextInChain = NULL,
        .label = "FS MSAA Color View",
        .format = core->target_format,
        .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = 0u,
        .mipLevelCount = 1u,
        .baseArrayLayer = 0u,
        .arrayLayerCount = 1u,
        .aspect = WGPUTextureAspect_All
    };
    core->msaa_color_view = wgpuTextureCreateView(core->msaa_color_texture, &view_desc);
    if (!core->msaa_color_view) {
        wgpuTextureRelease(core->msaa_color_texture);
        core->msaa_color_texture = NULL;
        return false;
    }
    core->render_sample_count = FS_RENDER_MSAA_SAMPLES;
    return true;
}

static bool fs_upload_default_image(FS_Core* core) {
    static const uint8_t pixels[] = {
        255, 255, 255, 255,
        220, 220, 220, 255,
        220, 220, 220, 255,
        255, 255, 255, 255
    };
    FS_ImageHandle handle = {0};
    return fs_core_upload_image_rgba8(core, pixels, 2u, 2u, &handle);
}

static bool fs_recreate_render_bind_group(FS_Core* core) {
    if (!core || !core->render_bgl || !core->image_atlas_view || !core->image_atlas_sampler ||
        !core->glyph_atlas_view || !core->glyph_atlas_sampler ||
        !core->clip_mask_view || !core->clip_mask_sampler || !core->uniform_buffer ||
        !core->command_state_buffer ||
        !core->clip_layer_uniform_buffer) {
        return false;
    }
    if (core->render_bg) {
        wgpuBindGroupRelease(core->render_bg);
        core->render_bg = NULL;
    }
    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .textureView = core->image_atlas_view},
        {.binding = 1, .sampler = core->image_atlas_sampler},
        {.binding = 2, .textureView = core->glyph_atlas_view},
        {.binding = 3, .sampler = core->glyph_atlas_sampler},
        {.binding = 4, .buffer = core->uniform_buffer, .offset = 0, .size = sizeof(FS_Uniforms)},
        {.binding = 5, .textureView = core->clip_mask_view},
        {.binding = 6, .sampler = core->clip_mask_sampler},
        {.binding = 7, .buffer = core->clip_layer_uniform_buffer, .offset = 0, .size = sizeof(FS_ClipLayerUniforms)},
        {.binding = 8, .buffer = core->command_state_buffer, .offset = 0, .size = core->command_state_buffer_size}
    };
    WGPUBindGroupDescriptor desc = {
        .nextInChain = NULL,
        .label = "FS Render Bind Group",
        .layout = core->render_bgl,
        .entryCount = 9,
        .entries = entries
    };
    core->render_bg = wgpuDeviceCreateBindGroup(core->device, &desc);
    return core->render_bg != NULL;
}

static bool fs_recreate_clip_compute_bind_group(FS_Core* core) {
    if (!core || !core->clip_compute_bgl || !core->clip_edge_buffer || !core->clip_job_buffer ||
        !core->clip_mask_view || !core->clip_dispatch_uniform_buffer) {
        return false;
    }
    if (core->clip_compute_bg) {
        wgpuBindGroupRelease(core->clip_compute_bg);
        core->clip_compute_bg = NULL;
    }
    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = core->clip_edge_buffer, .offset = 0, .size = core->clip_edge_buffer_size},
        {.binding = 1, .buffer = core->clip_job_buffer, .offset = 0, .size = core->clip_job_buffer_size},
        {.binding = 2, .textureView = core->clip_mask_view},
        {.binding = 3, .buffer = core->clip_dispatch_uniform_buffer, .offset = 0, .size = sizeof(FS_ClipDispatchUniforms)}
    };
    WGPUBindGroupDescriptor desc = {
        .nextInChain = NULL,
        .label = "FS Clip Compute Bind Group",
        .layout = core->clip_compute_bgl,
        .entryCount = 4,
        .entries = entries
    };
    core->clip_compute_bg = wgpuDeviceCreateBindGroup(core->device, &desc);
    return core->clip_compute_bg != NULL;
}

static WGPUBindGroup fs_create_clip_compute_bind_group_range(
    FS_Core* core,
    uint64_t job_offset_bytes,
    uint64_t job_size_bytes
) {
    if (!core || !core->clip_compute_bgl || !core->clip_edge_buffer || !core->clip_job_buffer ||
        !core->clip_mask_view || !core->clip_dispatch_uniform_buffer) {
        return NULL;
    }
    if (job_size_bytes == 0u) {
        return NULL;
    }
    if (job_offset_bytes + job_size_bytes > core->clip_job_buffer_size) {
        return NULL;
    }

    const uint64_t edge_size_bytes = (core->clip_edge_buffer_size > 0u)
                                         ? core->clip_edge_buffer_size
                                         : (uint64_t)sizeof(FS_ClipEdgeGPU);
    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = core->clip_edge_buffer, .offset = 0u, .size = edge_size_bytes},
        {.binding = 1, .buffer = core->clip_job_buffer, .offset = job_offset_bytes, .size = job_size_bytes},
        {.binding = 2, .textureView = core->clip_mask_view},
        {.binding = 3, .buffer = core->clip_dispatch_uniform_buffer, .offset = 0u, .size = sizeof(FS_ClipDispatchUniforms)}
    };
    WGPUBindGroupDescriptor desc = {
        .nextInChain = NULL,
        .label = "FS Clip Compute Bind Group Range",
        .layout = core->clip_compute_bgl,
        .entryCount = 4,
        .entries = entries
    };
    return wgpuDeviceCreateBindGroup(core->device, &desc);
}

static WGPUBindGroup fs_create_clip_edge_transform_bind_group_range(
    FS_Core* core,
    uint64_t job_offset_bytes,
    uint64_t job_size_bytes,
    uint64_t xform_offset_bytes,
    uint64_t xform_size_bytes
) {
    if (!core || !core->clip_edge_transform_bgl ||
        !core->clip_edge_local_buffer || !core->clip_job_buffer || !core->clip_job_xform_buffer ||
        !core->clip_edge_buffer) {
        return NULL;
    }
    if (job_size_bytes == 0u || xform_size_bytes == 0u) {
        return NULL;
    }
    if (job_offset_bytes + job_size_bytes > core->clip_job_buffer_size) {
        return NULL;
    }
    if (xform_offset_bytes + xform_size_bytes > core->clip_job_xform_buffer_size) {
        return NULL;
    }
    const uint64_t edge_local_size_bytes = (core->clip_edge_local_buffer_size > 0u)
                                               ? core->clip_edge_local_buffer_size
                                               : (uint64_t)sizeof(FS_ClipEdgeGPU);
    const uint64_t edge_device_size_bytes = (core->clip_edge_buffer_size > 0u)
                                                ? core->clip_edge_buffer_size
                                                : (uint64_t)sizeof(FS_ClipEdgeGPU);
    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = core->clip_edge_local_buffer, .offset = 0u, .size = edge_local_size_bytes},
        {.binding = 1, .buffer = core->clip_job_buffer, .offset = job_offset_bytes, .size = job_size_bytes},
        {.binding = 2, .buffer = core->clip_job_xform_buffer, .offset = xform_offset_bytes, .size = xform_size_bytes},
        {.binding = 3, .buffer = core->clip_edge_buffer, .offset = 0u, .size = edge_device_size_bytes}
    };
    WGPUBindGroupDescriptor desc = {
        .nextInChain = NULL,
        .label = "FS Clip Edge Transform Bind Group Range",
        .layout = core->clip_edge_transform_bgl,
        .entryCount = 4,
        .entries = entries
    };
    return wgpuDeviceCreateBindGroup(core->device, &desc);
}

static bool fs_create_pipelines_and_bindings(FS_Core* core) {
    WGPUShaderModule compute_shader = fs_create_shader_module(core->device, FS_COMPUTE_WGSL, "FS Compute Shader");
    WGPUShaderModule render_shader = fs_create_shader_module(core->device, FS_RENDER_WGSL, "FS Render Shader");
    WGPUShaderModule clip_shader = fs_create_shader_module(core->device, FS_CLIP_MASK_WGSL, "FS Clip Compute Shader");
    WGPUShaderModule clip_edge_transform_shader = fs_create_shader_module(
        core->device,
        FS_CLIP_EDGE_TRANSFORM_WGSL,
        "FS Clip Edge Transform Shader"
    );
    if (!compute_shader || !render_shader || !clip_shader || !clip_edge_transform_shader) {
        if (compute_shader) {
            wgpuShaderModuleRelease(compute_shader);
        }
        if (render_shader) {
            wgpuShaderModuleRelease(render_shader);
        }
        if (clip_shader) {
            wgpuShaderModuleRelease(clip_shader);
        }
        if (clip_edge_transform_shader) {
            wgpuShaderModuleRelease(clip_edge_transform_shader);
        }
        return false;
    }

    WGPUBindGroupLayoutEntry compute_entries[] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_Command)
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_CommandStateGPU)
            }
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_VertexGPU)
            }
        },
        {
            .binding = 3,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_Uniforms)
            }
        },
        {
            .binding = 4,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_ClipLayerUniforms)
            }
        }
    };
    WGPUBindGroupLayoutDescriptor compute_bgl_desc = {
        .nextInChain = NULL,
        .label = "FS Compute BGL",
        .entryCount = 5,
        .entries = compute_entries
    };
    core->compute_bgl = wgpuDeviceCreateBindGroupLayout(core->device, &compute_bgl_desc);
    if (!core->compute_bgl) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuShaderModuleRelease(render_shader);
        wgpuShaderModuleRelease(clip_shader);
        wgpuShaderModuleRelease(clip_edge_transform_shader);
        return false;
    }

    WGPUBindGroupLayoutEntry render_entries[] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Fragment,
            .texture = {
                .sampleType = WGPUTextureSampleType_Float,
                .viewDimension = WGPUTextureViewDimension_2DArray,
                .multisampled = false
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Fragment,
            .sampler = {
                .type = WGPUSamplerBindingType_Filtering
            }
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Fragment,
            .texture = {
                .sampleType = WGPUTextureSampleType_Float,
                .viewDimension = WGPUTextureViewDimension_2D,
                .multisampled = false
            }
        },
        {
            .binding = 3,
            .visibility = WGPUShaderStage_Fragment,
            .sampler = {
                .type = WGPUSamplerBindingType_Filtering
            }
        },
        {
            .binding = 4,
            .visibility = WGPUShaderStage_Fragment,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_Uniforms)
            }
        },
        {
            .binding = 5,
            .visibility = WGPUShaderStage_Fragment,
            .texture = {
                .sampleType = WGPUTextureSampleType_Float,
                .viewDimension = WGPUTextureViewDimension_2DArray,
                .multisampled = false
            }
        },
        {
            .binding = 6,
            .visibility = WGPUShaderStage_Fragment,
            .sampler = {
                .type = WGPUSamplerBindingType_Filtering
            }
        },
        {
            .binding = 7,
            .visibility = WGPUShaderStage_Fragment,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_ClipLayerUniforms)
            }
        },
        {
            .binding = 8,
            .visibility = WGPUShaderStage_Fragment,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_CommandStateGPU)
            }
        }
    };
    WGPUBindGroupLayoutDescriptor render_bgl_desc = {
        .nextInChain = NULL,
        .label = "FS Render BGL",
        .entryCount = 9,
        .entries = render_entries
    };
    core->render_bgl = wgpuDeviceCreateBindGroupLayout(core->device, &render_bgl_desc);
    if (!core->render_bgl) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuShaderModuleRelease(render_shader);
        wgpuShaderModuleRelease(clip_shader);
        wgpuShaderModuleRelease(clip_edge_transform_shader);
        return false;
    }

    WGPUStringView compute_entry = {.data = "main", .length = 4};
    WGPUPipelineLayoutDescriptor compute_layout_desc = {
        .nextInChain = NULL,
        .label = "FS Compute Pipeline Layout",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &core->compute_bgl
    };
    WGPUPipelineLayout compute_layout = wgpuDeviceCreatePipelineLayout(core->device, &compute_layout_desc);
    if (!compute_layout) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuShaderModuleRelease(render_shader);
        wgpuShaderModuleRelease(clip_shader);
        wgpuShaderModuleRelease(clip_edge_transform_shader);
        return false;
    }
    WGPUComputePipelineDescriptor compute_pipe_desc = {
        .nextInChain = NULL,
        .label = "FS Compute Pipeline",
        .layout = compute_layout,
        .compute = {
            .module = compute_shader,
            .entryPoint = compute_entry,
            .constantCount = 0,
            .constants = NULL
        }
    };
    core->compute_pipeline = wgpuDeviceCreateComputePipeline(core->device, &compute_pipe_desc);
    wgpuPipelineLayoutRelease(compute_layout);

    WGPUBindGroupLayoutEntry clip_entries[] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_ClipEdgeGPU)
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_ClipJobGPU)
            }
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .storageTexture = {
                .access = WGPUStorageTextureAccess_WriteOnly,
                .format = WGPUTextureFormat_RGBA8Unorm,
                .viewDimension = WGPUTextureViewDimension_2DArray
            }
        },
        {
            .binding = 3,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_ClipDispatchUniforms)
            }
        }
    };
    WGPUBindGroupLayoutDescriptor clip_bgl_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Compute BGL",
        .entryCount = 4,
        .entries = clip_entries
    };
    core->clip_compute_bgl = wgpuDeviceCreateBindGroupLayout(core->device, &clip_bgl_desc);
    if (!core->clip_compute_bgl) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuShaderModuleRelease(render_shader);
        wgpuShaderModuleRelease(clip_shader);
        wgpuShaderModuleRelease(clip_edge_transform_shader);
        return false;
    }
    WGPUPipelineLayoutDescriptor clip_layout_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Compute Pipeline Layout",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &core->clip_compute_bgl
    };
    WGPUPipelineLayout clip_layout = wgpuDeviceCreatePipelineLayout(core->device, &clip_layout_desc);
    if (!clip_layout) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuShaderModuleRelease(render_shader);
        wgpuShaderModuleRelease(clip_shader);
        wgpuShaderModuleRelease(clip_edge_transform_shader);
        return false;
    }
    WGPUComputePipelineDescriptor clip_pipe_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Compute Pipeline",
        .layout = clip_layout,
        .compute = {
            .module = clip_shader,
            .entryPoint = compute_entry,
            .constantCount = 0,
            .constants = NULL
        }
    };
    core->clip_compute_pipeline = wgpuDeviceCreateComputePipeline(core->device, &clip_pipe_desc);
    wgpuPipelineLayoutRelease(clip_layout);

    WGPUBindGroupLayoutEntry clip_edge_transform_entries[] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_ClipEdgeGPU)
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_ClipJobGPU)
            }
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_ClipJobTransformGPU)
            }
        },
        {
            .binding = 3,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(FS_ClipEdgeGPU)
            }
        }
    };
    WGPUBindGroupLayoutDescriptor clip_edge_transform_bgl_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Edge Transform BGL",
        .entryCount = 4,
        .entries = clip_edge_transform_entries
    };
    core->clip_edge_transform_bgl = wgpuDeviceCreateBindGroupLayout(core->device, &clip_edge_transform_bgl_desc);
    if (!core->clip_edge_transform_bgl) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuShaderModuleRelease(render_shader);
        wgpuShaderModuleRelease(clip_shader);
        wgpuShaderModuleRelease(clip_edge_transform_shader);
        return false;
    }
    WGPUPipelineLayoutDescriptor clip_edge_transform_layout_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Edge Transform Pipeline Layout",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &core->clip_edge_transform_bgl
    };
    WGPUPipelineLayout clip_edge_transform_layout =
        wgpuDeviceCreatePipelineLayout(core->device, &clip_edge_transform_layout_desc);
    if (!clip_edge_transform_layout) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuShaderModuleRelease(render_shader);
        wgpuShaderModuleRelease(clip_shader);
        wgpuShaderModuleRelease(clip_edge_transform_shader);
        return false;
    }
    WGPUComputePipelineDescriptor clip_edge_transform_pipe_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Edge Transform Pipeline",
        .layout = clip_edge_transform_layout,
        .compute = {
            .module = clip_edge_transform_shader,
            .entryPoint = compute_entry,
            .constantCount = 0,
            .constants = NULL
        }
    };
    core->clip_edge_transform_pipeline =
        wgpuDeviceCreateComputePipeline(core->device, &clip_edge_transform_pipe_desc);
    wgpuPipelineLayoutRelease(clip_edge_transform_layout);

    WGPUVertexAttribute attrs[] = {
        {.shaderLocation = 0, .format = WGPUVertexFormat_Float32x4, .offset = 0},
        {.shaderLocation = 1, .format = WGPUVertexFormat_Float32x4, .offset = 16},
        {.shaderLocation = 2, .format = WGPUVertexFormat_Float32x2, .offset = 32},
        {.shaderLocation = 3, .format = WGPUVertexFormat_Float32x2, .offset = 40},
        {.shaderLocation = 4, .format = WGPUVertexFormat_Uint32, .offset = 48},
        {.shaderLocation = 5, .format = WGPUVertexFormat_Uint32, .offset = 52},
        {.shaderLocation = 6, .format = WGPUVertexFormat_Uint32, .offset = 56},
        {.shaderLocation = 7, .format = WGPUVertexFormat_Float32x4, .offset = 64},
        {.shaderLocation = 8, .format = WGPUVertexFormat_Float32x4, .offset = 80},
        {.shaderLocation = 9, .format = WGPUVertexFormat_Float32x4, .offset = 96}
    };
    WGPUVertexBufferLayout vb_layout = {
        .arrayStride = sizeof(FS_VertexGPU),
        .stepMode = WGPUVertexStepMode_Vertex,
        .attributeCount = sizeof(attrs) / sizeof(attrs[0]),
        .attributes = attrs
    };
    WGPUStringView vs_entry = {.data = "vs_main", .length = 7};
    WGPUStringView fs_entry = {.data = "fs_main", .length = 7};
    WGPUVertexState vertex_state = {
        .nextInChain = NULL,
        .module = render_shader,
        .entryPoint = vs_entry,
        .constantCount = 0,
        .constants = NULL,
        .bufferCount = 1,
        .buffers = &vb_layout
    };
    WGPUColorTargetState target = {
        .format = core->target_format,
        .blend = NULL,
        .writeMask = WGPUColorWriteMask_All
    };
    WGPUFragmentState fragment_state = {
        .nextInChain = NULL,
        .module = render_shader,
        .entryPoint = fs_entry,
        .constantCount = 0,
        .constants = NULL,
        .targetCount = 1,
        .targets = &target
    };

    WGPUPipelineLayoutDescriptor render_layout_desc = {
        .nextInChain = NULL,
        .label = "FS Render Pipeline Layout",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &core->render_bgl
    };
    WGPUPipelineLayout render_layout = wgpuDeviceCreatePipelineLayout(core->device, &render_layout_desc);
    if (!render_layout) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuShaderModuleRelease(render_shader);
        wgpuShaderModuleRelease(clip_shader);
        return false;
    }
    memset(core->render_pipelines, 0, sizeof(core->render_pipelines));
    for (uint32_t i = 0u; i < FS_RENDER_PIPELINE_COUNT; ++i) {
        WGPUBlendState blend = fs_make_blend_state_for_pipeline(i);
        target.blend = &blend;
        const uint32_t sample_count = core->render_sample_count > 0u ? core->render_sample_count : 1u;
        WGPURenderPipelineDescriptor render_pipe_desc = {
            .nextInChain = NULL,
            .label = "FS Render Pipeline",
            .layout = render_layout,
            .vertex = vertex_state,
            .primitive = {
                .topology = WGPUPrimitiveTopology_TriangleList,
                .stripIndexFormat = WGPUIndexFormat_Undefined,
                .frontFace = WGPUFrontFace_CCW,
                .cullMode = WGPUCullMode_None
            },
            .depthStencil = NULL,
            .multisample = {
                .count = sample_count,
                .mask = 0xFFFFFFFFu,
                .alphaToCoverageEnabled = false
            },
            .fragment = &fragment_state
        };
        core->render_pipelines[i] = wgpuDeviceCreateRenderPipeline(core->device, &render_pipe_desc);
        if (!core->render_pipelines[i]) {
            break;
        }
    }
    wgpuPipelineLayoutRelease(render_layout);

    wgpuShaderModuleRelease(compute_shader);
    wgpuShaderModuleRelease(render_shader);
    wgpuShaderModuleRelease(clip_shader);
    wgpuShaderModuleRelease(clip_edge_transform_shader);

    bool has_all_render_pipelines = true;
    for (uint32_t i = 0u; i < FS_RENDER_PIPELINE_COUNT; ++i) {
        if (!core->render_pipelines[i]) {
            has_all_render_pipelines = false;
            break;
        }
    }
    if (!core->compute_pipeline || !has_all_render_pipelines || !core->clip_compute_pipeline ||
        !core->clip_edge_transform_pipeline) {
        return false;
    }
    if (!fs_recreate_render_bind_group(core)) {
        return false;
    }
    return fs_recreate_compute_bind_group(core);
}

static void fs_clear_loaded_fonts(FS_InternalState* st) {
    if (!st || !st->font_backend || !st->font_backend->destroy_font) {
        if (st) {
            st->font_count = 0u;
            memset(st->fonts, 0, sizeof(st->fonts));
        }
        return;
    }
    for (uint32_t i = 0u; i < st->font_count && i < FS_MAX_FONT_FALLBACKS; ++i) {
        if (st->fonts[i]) {
            st->font_backend->destroy_font(st->fonts[i]);
            st->fonts[i] = NULL;
        }
    }
    st->font_count = 0u;
}

static void fs_free_internal_state(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return;
    }
    fs_clear_loaded_fonts(st);
    free(st->glyphs);
    st->glyphs = NULL;
    st->glyph_count = 0;
    st->glyph_capacity = 0;
    free(st->glyph_hash_slots);
    st->glyph_hash_slots = NULL;
    st->glyph_hash_capacity = 0u;

    if (st->image_fonts) {
        for (uint32_t i = 0u; i < st->image_font_count; ++i) {
            FS_ImageFontState* font = &st->image_fonts[i];
            if (font->sequences) {
                for (uint32_t j = 0u; j < font->sequence_count; ++j) {
                    free(font->sequences[j].utf8);
                    font->sequences[j].utf8 = NULL;
                }
            }
            free(font->sequences);
            font->sequences = NULL;
            font->sequence_count = 0u;

            if (font->glyph_slots) {
                for (uint32_t j = 0u; j < font->glyph_slot_count; ++j) {
                    free(font->glyph_slots[j].rgba);
                    font->glyph_slots[j].rgba = NULL;
                    font->glyph_slots[j].loaded = false;
                }
            }
            free(font->glyph_slots);
            font->glyph_slots = NULL;
            font->glyph_slot_count = 0u;
        }
    }
    free(st->image_fonts);
    st->image_fonts = NULL;
    st->image_font_count = 0u;
    st->image_font_capacity = 0u;

    free(st->missing_image_glyphs);
    st->missing_image_glyphs = NULL;
    st->missing_count = 0u;
    st->missing_capacity = 0u;

    free(st->path_segments);
    st->path_segments = NULL;
    st->path_count = 0u;
    st->path_capacity = 0u;
    st->path_has_current = false;
    st->path_has_subpath_start = false;
    fs_state_stack_clear(st);
    free(st->state_stack);
    st->state_stack = NULL;
    st->state_stack_count = 0u;
    st->state_stack_capacity = 0u;
    free(st->style_dash_segments);
    st->style_dash_segments = NULL;
    st->style_dash_count = 0u;
    st->style_dash_offset = 0.0f;

    free(st);
    core->internal_state = NULL;
}

static void fs_release_resources(FS_Core* core) {
    if (!core) {
        return;
    }
    if (core->render_bg) {
        wgpuBindGroupRelease(core->render_bg);
        core->render_bg = NULL;
    }
    if (core->compute_bg) {
        wgpuBindGroupRelease(core->compute_bg);
        core->compute_bg = NULL;
    }
    if (core->clip_compute_bg) {
        wgpuBindGroupRelease(core->clip_compute_bg);
        core->clip_compute_bg = NULL;
    }
    for (uint32_t i = 0u; i < FS_RENDER_PIPELINE_COUNT; ++i) {
        if (core->render_pipelines[i]) {
            wgpuRenderPipelineRelease(core->render_pipelines[i]);
            core->render_pipelines[i] = NULL;
        }
    }
    if (core->compute_pipeline) {
        wgpuComputePipelineRelease(core->compute_pipeline);
        core->compute_pipeline = NULL;
    }
    if (core->clip_compute_pipeline) {
        wgpuComputePipelineRelease(core->clip_compute_pipeline);
        core->clip_compute_pipeline = NULL;
    }
    if (core->clip_edge_transform_pipeline) {
        wgpuComputePipelineRelease(core->clip_edge_transform_pipeline);
        core->clip_edge_transform_pipeline = NULL;
    }
    if (core->render_bgl) {
        wgpuBindGroupLayoutRelease(core->render_bgl);
        core->render_bgl = NULL;
    }
    if (core->compute_bgl) {
        wgpuBindGroupLayoutRelease(core->compute_bgl);
        core->compute_bgl = NULL;
    }
    if (core->clip_compute_bgl) {
        wgpuBindGroupLayoutRelease(core->clip_compute_bgl);
        core->clip_compute_bgl = NULL;
    }
    if (core->clip_edge_transform_bgl) {
        wgpuBindGroupLayoutRelease(core->clip_edge_transform_bgl);
        core->clip_edge_transform_bgl = NULL;
    }

    if (core->glyph_atlas_sampler) {
        wgpuSamplerRelease(core->glyph_atlas_sampler);
        core->glyph_atlas_sampler = NULL;
    }
    if (core->glyph_atlas_view) {
        wgpuTextureViewRelease(core->glyph_atlas_view);
        core->glyph_atlas_view = NULL;
    }
    if (core->glyph_atlas_texture) {
        wgpuTextureRelease(core->glyph_atlas_texture);
        core->glyph_atlas_texture = NULL;
    }
    if (core->clip_mask_sampler) {
        wgpuSamplerRelease(core->clip_mask_sampler);
        core->clip_mask_sampler = NULL;
    }
    if (core->clip_mask_view) {
        wgpuTextureViewRelease(core->clip_mask_view);
        core->clip_mask_view = NULL;
    }
    if (core->clip_mask_texture) {
        wgpuTextureRelease(core->clip_mask_texture);
        core->clip_mask_texture = NULL;
    }
    if (core->msaa_color_view) {
        wgpuTextureViewRelease(core->msaa_color_view);
        core->msaa_color_view = NULL;
    }
    if (core->msaa_color_texture) {
        wgpuTextureRelease(core->msaa_color_texture);
        core->msaa_color_texture = NULL;
    }
    core->render_sample_count = 1u;
    if (core->image_atlas_sampler) {
        wgpuSamplerRelease(core->image_atlas_sampler);
        core->image_atlas_sampler = NULL;
    }
    if (core->image_atlas_view) {
        wgpuTextureViewRelease(core->image_atlas_view);
        core->image_atlas_view = NULL;
    }
    if (core->image_atlas_texture) {
        wgpuTextureRelease(core->image_atlas_texture);
        core->image_atlas_texture = NULL;
    }
    free(core->image_atlas_shadow_rgba);
    core->image_atlas_shadow_rgba = NULL;
    core->image_atlas_shadow_size = 0u;
    free(core->canvas_shadow_rgba);
    core->canvas_shadow_rgba = NULL;
    core->canvas_shadow_size = 0u;
    if (core->canvas_readback_buffer) {
        if (core->canvas_readback_mapped) {
            wgpuBufferUnmap(core->canvas_readback_buffer);
            core->canvas_readback_mapped = 0u;
        }
        wgpuBufferRelease(core->canvas_readback_buffer);
        core->canvas_readback_buffer = NULL;
    }
    core->canvas_readback_buffer_size = 0u;
    core->canvas_readback_row_bytes = 0u;
    core->canvas_readback_padded_row_bytes = 0u;
    core->canvas_readback_width = 0u;
    core->canvas_readback_height = 0u;
    core->canvas_readback_serial = 0u;
    core->canvas_shadow_serial = 0u;
    core->canvas_readback_submission = 0u;
    core->canvas_readback_submission_valid = 0u;
    core->canvas_readback_mapped = 0u;
    core->canvas_image_data_handle_valid = 0u;
    memset(&core->canvas_image_data_handle, 0, sizeof(core->canvas_image_data_handle));
    if (core->clip_mask_layer_hash_valid) {
        memset(core->clip_mask_layer_hash_valid, 0, (size_t)core->clip_mask_layers * sizeof(uint8_t));
    }
    free(core->clip_mask_layer_has_data);
    core->clip_mask_layer_has_data = NULL;
    free(core->clip_mask_layer_min_x);
    core->clip_mask_layer_min_x = NULL;
    free(core->clip_mask_layer_min_y);
    core->clip_mask_layer_min_y = NULL;
    free(core->clip_mask_layer_max_x);
    core->clip_mask_layer_max_x = NULL;
    free(core->clip_mask_layer_max_y);
    core->clip_mask_layer_max_y = NULL;
    free(core->clip_mask_layer_hash);
    core->clip_mask_layer_hash = NULL;
    free(core->clip_mask_layer_hash_valid);
    core->clip_mask_layer_hash_valid = NULL;
    free(core->clip_mask_layer_parent);
    core->clip_mask_layer_parent = NULL;
    free(core->clip_mask_layer_last_used_frame);
    core->clip_mask_layer_last_used_frame = NULL;

    if (core->uniform_buffer) {
        wgpuBufferRelease(core->uniform_buffer);
        core->uniform_buffer = NULL;
    }
    if (core->clip_dispatch_uniform_buffer) {
        wgpuBufferRelease(core->clip_dispatch_uniform_buffer);
        core->clip_dispatch_uniform_buffer = NULL;
    }
    if (core->clip_layer_uniform_buffer) {
        wgpuBufferRelease(core->clip_layer_uniform_buffer);
        core->clip_layer_uniform_buffer = NULL;
    }
    if (core->clip_job_buffer) {
        wgpuBufferRelease(core->clip_job_buffer);
        core->clip_job_buffer = NULL;
    }
    core->clip_job_buffer_size = 0u;
    if (core->clip_job_xform_buffer) {
        wgpuBufferRelease(core->clip_job_xform_buffer);
        core->clip_job_xform_buffer = NULL;
    }
    core->clip_job_xform_buffer_size = 0u;
    if (core->clip_edge_buffer) {
        wgpuBufferRelease(core->clip_edge_buffer);
        core->clip_edge_buffer = NULL;
    }
    core->clip_edge_buffer_size = 0u;
    if (core->clip_edge_local_buffer) {
        wgpuBufferRelease(core->clip_edge_local_buffer);
        core->clip_edge_local_buffer = NULL;
    }
    core->clip_edge_local_buffer_size = 0u;
    if (core->vertex_buffer) {
        wgpuBufferRelease(core->vertex_buffer);
        core->vertex_buffer = NULL;
    }
    if (core->command_state_buffer) {
        wgpuBufferRelease(core->command_state_buffer);
        core->command_state_buffer = NULL;
    }
    core->command_state_buffer_size = 0u;
    if (core->command_buffer) {
        wgpuBufferRelease(core->command_buffer);
        core->command_buffer = NULL;
    }
    core->command_buffer_size = 0u;
    core->vertex_buffer_size = 0u;

    free(core->upload_staging_cpu);
    core->upload_staging_cpu = NULL;
    core->upload_staging_cpu_capacity = 0u;
    core->upload_staging_used = 0u;
    if (core->upload_staging_gpu) {
        wgpuBufferRelease(core->upload_staging_gpu);
        core->upload_staging_gpu = NULL;
    }
    core->upload_staging_gpu_capacity = 0u;
    free(core->pending_uploads);
    core->pending_uploads = NULL;
    core->pending_upload_count = 0u;
    core->pending_upload_capacity = 0u;

    for (uint32_t i = 0u; i < 2u; ++i) {
        free(core->glyph_scratch_rgba[i]);
        core->glyph_scratch_rgba[i] = NULL;
        core->glyph_scratch_rgba_capacity[i] = 0u;
        free(core->glyph_scratch_alpha[i]);
        core->glyph_scratch_alpha[i] = NULL;
        core->glyph_scratch_alpha_capacity[i] = 0u;
    }

    free(core->commands);
    core->commands = NULL;
    free(core->command_states);
    core->command_states = NULL;
    core->command_count = 0;
    core->command_capacity = 0;
    core->command_state_capacity = 0;
    free(core->clip_edge_cpu);
    core->clip_edge_cpu = NULL;
    core->clip_edge_count = 0u;
    core->clip_edge_capacity = 0u;
    free(core->clip_job_cpu);
    core->clip_job_cpu = NULL;
    free(core->clip_job_xform_cpu);
    core->clip_job_xform_cpu = NULL;
    core->clip_job_count = 0u;
    core->clip_job_capacity = 0u;

    fs_free_internal_state(core);
}

static bool fs_ensure_cpu_capacity(FS_Core* core, size_t required_count) {
    if (!core) {
        return false;
    }
    if (required_count <= core->command_capacity) {
        return true;
    }
    size_t new_capacity = core->command_capacity ? core->command_capacity : 1024;
    while (new_capacity < required_count) {
        if (new_capacity > (SIZE_MAX / 2)) {
            new_capacity = required_count;
            break;
        }
        new_capacity *= 2;
    }
    FS_Command* grown = (FS_Command*)realloc(core->commands, new_capacity * sizeof(FS_Command));
    if (!grown) {
        return false;
    }
    core->commands = grown;
    FS_CommandStateGPU* grown_states =
        (FS_CommandStateGPU*)realloc(core->command_states, new_capacity * sizeof(FS_CommandStateGPU));
    if (!grown_states) {
        return false;
    }
    core->command_states = grown_states;
    core->command_capacity = new_capacity;
    core->command_state_capacity = new_capacity;
    return true;
}

static bool fs_ensure_gpu_capacity(FS_Core* core, size_t command_count) {
    if (!core) {
        return false;
    }
    const size_t min_cmd_bytes = sizeof(FS_Command) * 1024u;
    const size_t min_state_bytes = sizeof(FS_CommandStateGPU) * 1024u;
    const size_t min_vtx_bytes = sizeof(FS_VertexGPU) * 6u * 1024u;
    const size_t needed_cmd_bytes = command_count ? (command_count * sizeof(FS_Command)) : min_cmd_bytes;
    const size_t needed_state_bytes = command_count ? (command_count * sizeof(FS_CommandStateGPU)) : min_state_bytes;
    const size_t needed_vtx_bytes = command_count ? (command_count * 6u * sizeof(FS_VertexGPU)) : min_vtx_bytes;

    bool resized = false;
    if (needed_cmd_bytes > core->command_buffer_size) {
        size_t new_size = core->command_buffer_size ? core->command_buffer_size : min_cmd_bytes;
        while (new_size < needed_cmd_bytes) {
            new_size *= 2u;
        }
        WGPUBuffer new_buf = fs_create_buffer(
            core->device,
            "FS Command Buffer",
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            new_size
        );
        if (!new_buf) {
            return false;
        }
        if (core->command_buffer) {
            wgpuBufferRelease(core->command_buffer);
        }
        core->command_buffer = new_buf;
        core->command_buffer_size = new_size;
        resized = true;
    }

    if (needed_state_bytes > core->command_state_buffer_size) {
        size_t new_size = core->command_state_buffer_size ? core->command_state_buffer_size : min_state_bytes;
        while (new_size < needed_state_bytes) {
            new_size *= 2u;
        }
        WGPUBuffer new_buf = fs_create_buffer(
            core->device,
            "FS Command State Buffer",
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            new_size
        );
        if (!new_buf) {
            return false;
        }
        if (core->command_state_buffer) {
            wgpuBufferRelease(core->command_state_buffer);
        }
        core->command_state_buffer = new_buf;
        core->command_state_buffer_size = new_size;
        resized = true;
    }

    if (needed_vtx_bytes > core->vertex_buffer_size) {
        size_t new_size = core->vertex_buffer_size ? core->vertex_buffer_size : min_vtx_bytes;
        while (new_size < needed_vtx_bytes) {
            new_size *= 2u;
        }
        WGPUBuffer new_buf = fs_create_buffer(
            core->device,
            "FS Vertex Buffer",
            WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
            new_size
        );
        if (!new_buf) {
            return false;
        }
        if (core->vertex_buffer) {
            wgpuBufferRelease(core->vertex_buffer);
        }
        core->vertex_buffer = new_buf;
        core->vertex_buffer_size = new_size;
        resized = true;
    }

    if (resized) {
        bool ok = fs_recreate_compute_bind_group(core);
        if (!fs_recreate_render_bind_group(core)) {
            ok = false;
        }
        return ok;
    }
    return true;
}

static bool fs_push_command(FS_Core* core, const FS_Command* cmd) {
    if (!core || !cmd) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    const bool is_shadow_command = (cmd->flags & FS_RENDER_FLAG_SHADOW) != 0u;
    bool emit_shadow = false;
    uint32_t shadow_blur_bits = 0u;
    uint32_t shadow_color = 0u;
    bool has_shadow_offset = false;
    if (!is_shadow_command && st && fs_command_supports_shadow(cmd->type)) {
        shadow_blur_bits = fs_shadow_blur_to_flag_bits(st->style_shadow_blur);
        shadow_color = st->style_shadow_color_rgba8;
        const uint32_t shadow_alpha = (shadow_color >> 24u) & 0xFFu;
        has_shadow_offset =
            isfinite(st->style_shadow_offset_x) &&
            isfinite(st->style_shadow_offset_y) &&
            (fabsf(st->style_shadow_offset_x) > 1e-5f || fabsf(st->style_shadow_offset_y) > 1e-5f);
        emit_shadow = ((shadow_blur_bits != 0u) || has_shadow_offset) && (shadow_alpha != 0u);
    }

    const size_t needed = core->command_count + (emit_shadow ? 2u : 1u);
    if (!fs_ensure_cpu_capacity(core, needed)) {
        return false;
    }
    if (core->command_count > (size_t)UINT32_MAX) {
        return false;
    }

    if (emit_shadow && !fs_emit_shadow_commands(core, cmd, st, shadow_color, shadow_blur_bits)) {
        return false;
    }

    FS_Command baked = *cmd;
    memset(baked.clip_min, 0, sizeof(baked.clip_min));
    memset(baked.clip_max, 0, sizeof(baked.clip_max));
    baked.clip_enabled = 0u;
    baked.flags &= FS_RENDER_FLAG_USER_MASK;
    baked.state_index = (uint32_t)core->command_count;

    FS_CommandStateGPU state;
    memset(&state, 0, sizeof(state));
    state.clip_meta[1] = UINT32_MAX;
    state.clip_meta[2] = UINT32_MAX;
    state.xform0[0] = 1.0f;
    state.xform0[3] = 1.0f;
    state.xform1[2] = (float)FS_IMAGE_SMOOTHING_QUALITY_LOW;
    state.xform1[3] = 1.0f;
    fs_command_state_clear_pattern(&state);

    state.clip_meta[3] =
        st ? fs_composite_op_to_pipeline_index((FS_GlobalCompositeOperation)st->style_composite_op) : 0u;
    if (state.clip_meta[3] >= FS_RENDER_PIPELINE_COUNT) {
        state.clip_meta[3] = 0u;
    }
    if (st) {
        const FS_Transform2D* t = &st->current_transform;
        state.xform0[0] = t->a;
        state.xform0[1] = t->b;
        state.xform0[2] = t->c;
        state.xform0[3] = t->d;
        state.xform1[0] = t->e;
        state.xform1[1] = t->f;
        state.xform1[2] = (float)st->style_image_smoothing_quality;
        state.xform1[3] = st->style_global_alpha;
    }

    if (st && st->clip_enabled) {
        state.clip_meta[0] |= FS_CMD_CLIP_RECT_BIT;
        state.clip_rect[0] = st->clip_min_x;
        state.clip_rect[1] = st->clip_min_y;
        state.clip_rect[2] = st->clip_max_x;
        state.clip_rect[3] = st->clip_max_y;
    }
    if (st && st->clip_path_enabled) {
        state.clip_meta[0] |= FS_CMD_CLIP_PATH_BIT;
        state.clip_meta[1] = (uint32_t)st->clip_path_layer;
        uint32_t parent_layer = UINT32_MAX;
        if (core->clip_mask_layer_parent && (uint32_t)st->clip_path_layer < core->clip_mask_layers) {
            parent_layer = core->clip_mask_layer_parent[st->clip_path_layer];
        }
        state.clip_meta[2] = parent_layer;
    }
    if ((baked.flags & FS_RENDER_FLAG_PATTERN_SHADE) != 0u) {
        const FS_StylePattern* style_pattern = NULL;
        if (st) {
            bool prefer_fill_pattern = (baked.flags & FS_RENDER_FLAG_PATTERN_FILL_HINT) != 0u;
            if (!prefer_fill_pattern && baked.type == FS_CMD_TEXT) {
                prefer_fill_pattern = (baked.flags & FS_TEXT_FLAG_STROKE) == 0u;
            }
            if (prefer_fill_pattern) {
                if (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN) {
                    style_pattern = &st->style_fill_pattern;
                }
            } else if (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN) {
                style_pattern = &st->style_stroke_pattern;
            }
        }
        if (!fs_command_state_set_pattern(core, &state, style_pattern)) {
            baked.flags &= ~FS_RENDER_FLAG_PATTERN_SHADE;
        }
    }
    if (st) {
        bool use_nearest = st->style_image_smoothing_enabled == 0u;
        if (use_nearest && (baked.type == FS_CMD_IMAGE || (baked.flags & FS_RENDER_FLAG_PATTERN_SHADE) != 0u)) {
            baked.flags |= FS_RENDER_FLAG_IMAGE_NEAREST;
        }
    }
    if ((baked.flags & FS_RENDER_FLAG_ORIENTED_QUAD) != 0u) {
        core->clip_oriented_quad_commands_this_frame += 1u;
        if (state.clip_meta[0] != 0u) {
            core->clip_oriented_quad_clipped_this_frame += 1u;
        }
    }
    core->commands[core->command_count] = baked;
    core->command_states[core->command_count] = state;
    core->command_count += 1u;
    return true;
}

static uint32_t fs_decode_utf8(const char** cursor) {
    if (!cursor || !*cursor) {
        return 0u;
    }
    const unsigned char* s = (const unsigned char*)(*cursor);
    if (*s == 0u) {
        return 0u;
    }
    uint32_t cp = 0u;
    if (s[0] < 0x80u) {
        cp = s[0];
        *cursor += 1;
        return cp;
    }
    if ((s[0] & 0xE0u) == 0xC0u && s[1] != 0u) {
        cp = ((uint32_t)(s[0] & 0x1Fu) << 6) | (uint32_t)(s[1] & 0x3Fu);
        *cursor += 2;
        return cp;
    }
    if ((s[0] & 0xF0u) == 0xE0u && s[1] != 0u && s[2] != 0u) {
        cp = ((uint32_t)(s[0] & 0x0Fu) << 12) |
             ((uint32_t)(s[1] & 0x3Fu) << 6) |
             (uint32_t)(s[2] & 0x3Fu);
        *cursor += 3;
        return cp;
    }
    if ((s[0] & 0xF8u) == 0xF0u && s[1] != 0u && s[2] != 0u && s[3] != 0u) {
        cp = ((uint32_t)(s[0] & 0x07u) << 18) |
             ((uint32_t)(s[1] & 0x3Fu) << 12) |
             ((uint32_t)(s[2] & 0x3Fu) << 6) |
             (uint32_t)(s[3] & 0x3Fu);
        *cursor += 4;
        return cp;
    }
    *cursor += 1;
    return 0xFFFDu;
}

static char* fs_strdup_owned(const char* s) {
    if (!s) {
        return NULL;
    }
    const size_t len = strlen(s);
    char* out = (char*)malloc(len + 1u);
    if (!out) {
        return NULL;
    }
    memcpy(out, s, len + 1u);
    return out;
}

static bool fs_pack_image_glyph_key(uint32_t image_font_id, uint32_t glyph_id, uint32_t* out_key) {
    if (!out_key) {
        return false;
    }
    if (image_font_id >= (1u << FS_IMAGE_KEY_FONT_BITS) || glyph_id > FS_IMAGE_KEY_GLYPH_MASK) {
        return false;
    }
    *out_key = (image_font_id << FS_IMAGE_KEY_GLYPH_BITS) | glyph_id;
    return true;
}

static uint32_t fs_unpack_image_key_font_id(uint32_t key) {
    return key >> FS_IMAGE_KEY_GLYPH_BITS;
}

static uint32_t fs_unpack_image_key_glyph_id(uint32_t key) {
    return key & FS_IMAGE_KEY_GLYPH_MASK;
}

static bool fs_is_emoji_modifier(uint32_t cp) {
    return cp >= 0x1F3FBu && cp <= 0x1F3FFu;
}

static bool fs_is_emoji_variation_selector(uint32_t cp) {
    return cp == 0xFE0Fu || cp == 0xFE0Eu;
}

static bool fs_is_zwj(uint32_t cp) {
    return cp == 0x200Du;
}

static bool fs_is_keycap_combiner(uint32_t cp) {
    return cp == 0x20E3u;
}

static bool fs_canonicalize_utf8_sequence(
    const char* utf8,
    uint32_t* out_cps,
    uint8_t* out_cp_count
) {
    if (!utf8 || !out_cps || !out_cp_count) {
        return false;
    }
    *out_cp_count = 0u;
    const char* cursor = utf8;
    while (*cursor != '\0') {
        uint32_t cp = fs_decode_utf8(&cursor);
        if (cp == 0u) {
            break;
        }
        if (fs_is_emoji_variation_selector(cp)) {
            continue;
        }
        if (*out_cp_count >= FS_IMAGE_SEQ_MAX_CP) {
            return false;
        }
        out_cps[*out_cp_count] = cp;
        *out_cp_count += 1u;
    }
    return *out_cp_count > 0u;
}

static bool fs_match_sequence_prefix(
    const FS_ImageSequenceEntry* seq,
    const char* utf8_ptr,
    size_t* out_bytes
) {
    if (!seq || !utf8_ptr || !out_bytes || seq->cp_count == 0u) {
        return false;
    }
    const char* cursor = utf8_ptr;
    uint32_t matched = 0u;
    bool odd = false;
    while (*cursor != '\0' && matched < seq->cp_count) {
        const char* probe = cursor;
        const uint32_t cp = fs_decode_utf8(&probe);
        if (cp == 0u) {
            return false;
        }
        if (fs_is_emoji_variation_selector(cp)) {
            cursor = probe;
            continue;
        }

        bool consume = false;
        if (!odd) {
            consume = true;
            odd = true;
        } else if (fs_is_zwj(cp)) {
            consume = true;
            odd = false;
        } else if (fs_is_emoji_modifier(cp) || fs_is_keycap_combiner(cp)) {
            consume = true;
        } else {
            break;
        }

        if (!consume || cp != seq->cps[matched]) {
            return false;
        }
        matched += 1u;
        cursor = probe;
    }
    if (matched != seq->cp_count) {
        return false;
    }
    *out_bytes = (size_t)(cursor - utf8_ptr);
    return *out_bytes > 0u;
}

static bool fs_ensure_image_font_capacity(FS_InternalState* st, uint32_t required) {
    if (!st) {
        return false;
    }
    if (required <= st->image_font_capacity) {
        return true;
    }
    uint32_t new_cap = st->image_font_capacity ? st->image_font_capacity : 4u;
    while (new_cap < required) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_ImageFontState* grown = (FS_ImageFontState*)realloc(st->image_fonts, (size_t)new_cap * sizeof(FS_ImageFontState));
    if (!grown) {
        return false;
    }
    if (new_cap > st->image_font_capacity) {
        memset(
            grown + st->image_font_capacity,
            0,
            (size_t)(new_cap - st->image_font_capacity) * sizeof(FS_ImageFontState)
        );
    }
    st->image_fonts = grown;
    st->image_font_capacity = new_cap;
    return true;
}

static bool fs_ensure_missing_capacity(FS_InternalState* st, uint32_t required) {
    if (!st) {
        return false;
    }
    if (required <= st->missing_capacity) {
        return true;
    }
    uint32_t new_cap = st->missing_capacity ? st->missing_capacity : 16u;
    while (new_cap < required) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_MissingImageGlyph* grown =
        (FS_MissingImageGlyph*)realloc(st->missing_image_glyphs, (size_t)new_cap * sizeof(FS_MissingImageGlyph));
    if (!grown) {
        return false;
    }
    st->missing_image_glyphs = grown;
    st->missing_capacity = new_cap;
    return true;
}

static bool fs_ensure_path_capacity(FS_InternalState* st, uint32_t required) {
    if (!st) {
        return false;
    }
    if (required <= st->path_capacity) {
        return true;
    }
    uint32_t new_cap = st->path_capacity ? st->path_capacity : 64u;
    while (new_cap < required) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_PathSegment* grown =
        (FS_PathSegment*)realloc(st->path_segments, (size_t)new_cap * sizeof(FS_PathSegment));
    if (!grown) {
        return false;
    }
    st->path_segments = grown;
    st->path_capacity = new_cap;
    return true;
}

static bool fs_append_path_segment(FS_InternalState* st, const FS_PathSegment* segment) {
    if (!st || !segment) {
        return false;
    }
    if (!fs_ensure_path_capacity(st, st->path_count + 1u)) {
        return false;
    }
    st->path_segments[st->path_count++] = *segment;
    return true;
}

static bool fs_path2d_ensure_capacity(FS_Path2D* path, uint32_t required) {
    if (!path) {
        return false;
    }
    if (required <= path->capacity) {
        return true;
    }
    uint32_t new_cap = path->capacity ? path->capacity : 64u;
    while (new_cap < required) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_PathSegment* grown = (FS_PathSegment*)realloc(path->segments, (size_t)new_cap * sizeof(FS_PathSegment));
    if (!grown) {
        return false;
    }
    path->segments = grown;
    path->capacity = new_cap;
    return true;
}

static bool fs_path2d_append_segment(FS_Path2D* path, const FS_PathSegment* segment) {
    if (!path || !segment) {
        return false;
    }
    if (!fs_path2d_ensure_capacity(path, path->count + 1u)) {
        return false;
    }
    path->segments[path->count++] = *segment;
    return true;
}

static void fs_path_state_borrow(const FS_InternalState* st, FS_PathStateBorrow* out_state) {
    if (!out_state) {
        return;
    }
    memset(out_state, 0, sizeof(*out_state));
    if (!st) {
        return;
    }
    out_state->segments = st->path_segments;
    out_state->count = st->path_count;
    out_state->capacity = st->path_capacity;
    out_state->has_current = st->path_has_current;
    out_state->has_subpath_start = st->path_has_subpath_start;
    out_state->current_x = st->path_current_x;
    out_state->current_y = st->path_current_y;
    out_state->subpath_start_x = st->path_subpath_start_x;
    out_state->subpath_start_y = st->path_subpath_start_y;
}

static void fs_path_state_bind_path2d(FS_InternalState* st, const FS_Path2D* path) {
    if (!st) {
        return;
    }
    st->path_segments = path ? path->segments : NULL;
    st->path_count = path ? path->count : 0u;
    st->path_capacity = path ? path->capacity : 0u;
    st->path_has_current = path ? path->has_current : false;
    st->path_has_subpath_start = path ? path->has_subpath_start : false;
    st->path_current_x = path ? path->current_x : 0.0f;
    st->path_current_y = path ? path->current_y : 0.0f;
    st->path_subpath_start_x = path ? path->subpath_start_x : 0.0f;
    st->path_subpath_start_y = path ? path->subpath_start_y : 0.0f;
}

static void fs_path_state_bind_empty(FS_InternalState* st) {
    if (!st) {
        return;
    }
    st->path_segments = NULL;
    st->path_count = 0u;
    st->path_capacity = 0u;
    st->path_has_current = false;
    st->path_has_subpath_start = false;
    st->path_current_x = 0.0f;
    st->path_current_y = 0.0f;
    st->path_subpath_start_x = 0.0f;
    st->path_subpath_start_y = 0.0f;
}

static void fs_path_state_restore(FS_InternalState* st, const FS_PathStateBorrow* saved) {
    if (!st || !saved) {
        return;
    }
    st->path_segments = saved->segments;
    st->path_count = saved->count;
    st->path_capacity = saved->capacity;
    st->path_has_current = saved->has_current;
    st->path_has_subpath_start = saved->has_subpath_start;
    st->path_current_x = saved->current_x;
    st->path_current_y = saved->current_y;
    st->path_subpath_start_x = saved->subpath_start_x;
    st->path_subpath_start_y = saved->subpath_start_y;
}

static void fs_path_state_begin_temporary(FS_InternalState* st, FS_PathStateBorrow* out_saved) {
    if (!st || !out_saved) {
        return;
    }
    fs_path_state_borrow(st, out_saved);
    fs_path_state_bind_empty(st);
}

static void fs_path_state_end_temporary(FS_InternalState* st, const FS_PathStateBorrow* saved) {
    if (!st || !saved) {
        return;
    }
    FS_PathSegment* temp_segments = st->path_segments;
    fs_path_state_restore(st, saved);
    if (temp_segments && temp_segments != saved->segments) {
        free(temp_segments);
    }
}

static FS_ImageFontState* fs_find_image_font(FS_InternalState* st, uint32_t image_font_id) {
    if (!st || image_font_id == 0u) {
        return NULL;
    }
    for (uint32_t i = 0u; i < st->image_font_count; ++i) {
        if (st->image_fonts[i].id == image_font_id) {
            return &st->image_fonts[i];
        }
    }
    return NULL;
}

static bool fs_image_font_ensure_glyph_slots(FS_ImageFontState* font, uint32_t required_slots) {
    if (!font) {
        return false;
    }
    if (required_slots <= font->glyph_slot_count) {
        return true;
    }
    uint32_t new_count = font->glyph_slot_count ? font->glyph_slot_count : 8u;
    while (new_count < required_slots) {
        if (new_count > UINT32_MAX / 2u) {
            new_count = required_slots;
            break;
        }
        new_count *= 2u;
    }
    FS_ImageGlyphSlot* grown =
        (FS_ImageGlyphSlot*)realloc(font->glyph_slots, (size_t)new_count * sizeof(FS_ImageGlyphSlot));
    if (!grown) {
        return false;
    }
    if (new_count > font->glyph_slot_count) {
        memset(
            grown + font->glyph_slot_count,
            0,
            (size_t)(new_count - font->glyph_slot_count) * sizeof(FS_ImageGlyphSlot)
        );
    }
    font->glyph_slots = grown;
    font->glyph_slot_count = new_count;
    return true;
}

static bool fs_push_missing_image_glyph(
    FS_InternalState* st,
    uint32_t image_font_id,
    uint32_t glyph_id,
    const char* utf8,
    size_t utf8_len
) {
    if (!st || !utf8 || utf8_len == 0u) {
        return false;
    }
    for (uint32_t i = 0u; i < st->missing_count; ++i) {
        const FS_MissingImageGlyph* m = &st->missing_image_glyphs[i];
        if (m->image_font_id == image_font_id && m->glyph_id == glyph_id) {
            return true;
        }
    }
    if (!fs_ensure_missing_capacity(st, st->missing_count + 1u)) {
        return false;
    }
    FS_MissingImageGlyph* out = &st->missing_image_glyphs[st->missing_count++];
    memset(out, 0, sizeof(*out));
    out->image_font_id = image_font_id;
    out->glyph_id = glyph_id;
    const size_t copy_len = utf8_len < (sizeof(out->sequence_utf8) - 1u) ? utf8_len : (sizeof(out->sequence_utf8) - 1u);
    memcpy(out->sequence_utf8, utf8, copy_len);
    out->sequence_utf8[copy_len] = '\0';
    return true;
}

static bool fs_find_image_sequence_match(
    const FS_InternalState* st,
    const char* utf8_ptr,
    uint32_t* out_image_font_id,
    uint32_t* out_glyph_id,
    size_t* out_consumed_bytes
) {
    if (!st || !utf8_ptr || !out_image_font_id || !out_glyph_id || !out_consumed_bytes) {
        return false;
    }
    const FS_ImageSequenceEntry* best_seq = NULL;
    uint32_t best_font_id = 0u;
    size_t best_bytes = 0u;
    uint8_t best_cp_count = 0u;

    for (uint32_t i = 0u; i < st->image_font_count; ++i) {
        const FS_ImageFontState* font = &st->image_fonts[i];
        for (uint32_t j = 0u; j < font->sequence_count; ++j) {
            const FS_ImageSequenceEntry* seq = &font->sequences[j];
            size_t consumed = 0u;
            if (!fs_match_sequence_prefix(seq, utf8_ptr, &consumed)) {
                continue;
            }
            if (!best_seq ||
                seq->cp_count > best_cp_count ||
                (seq->cp_count == best_cp_count && consumed > best_bytes)) {
                best_seq = seq;
                best_font_id = font->id;
                best_bytes = consumed;
                best_cp_count = seq->cp_count;
            }
        }
    }
    if (!best_seq) {
        return false;
    }
    *out_image_font_id = best_font_id;
    *out_glyph_id = best_seq->glyph_id;
    *out_consumed_bytes = best_bytes;
    return true;
}

static uint32_t fs_glyph_cache_hash_key(uint32_t glyph_key, uint32_t key_kind, uint32_t bake_px_q) {
    uint32_t x = glyph_key * 0x9E3779B1u;
    x ^= key_kind * 0x85EBCA77u;
    x ^= bake_px_q * 0xC2B2AE3Du;
    x ^= x >> 16u;
    return x;
}

static size_t fs_glyph_cache_next_pow2(size_t v) {
    if (v <= 1u) {
        return 1u;
    }
    --v;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
#if SIZE_MAX > 0xFFFFFFFFu
    v |= v >> 32u;
#endif
    return v + 1u;
}

static bool fs_glyph_cache_rebuild(FS_InternalState* st, size_t min_capacity) {
    if (!st) {
        return false;
    }
    size_t target = min_capacity;
    const size_t needed_for_entries = st->glyph_count > 0u ? (st->glyph_count * 2u) : 0u;
    if (target < needed_for_entries) {
        target = needed_for_entries;
    }
    if (target < 64u) {
        target = 64u;
    }
    target = fs_glyph_cache_next_pow2(target);
    if (target < st->glyph_count) {
        return false;
    }

    uint32_t* table = (uint32_t*)calloc(target, sizeof(uint32_t));
    if (!table) {
        return false;
    }
    const size_t mask = target - 1u;
    for (size_t i = 0u; i < st->glyph_count; ++i) {
        const FS_GlyphEntry* g = &st->glyphs[i];
        size_t slot = (size_t)fs_glyph_cache_hash_key(g->glyph_key, g->key_kind, g->bake_px_q) & mask;
        for (size_t probe = 0u; probe < target; ++probe) {
            if (table[slot] == 0u) {
                table[slot] = (uint32_t)(i + 1u);
                break;
            }
            slot = (slot + 1u) & mask;
        }
    }

    free(st->glyph_hash_slots);
    st->glyph_hash_slots = table;
    st->glyph_hash_capacity = target;
    return true;
}

static void fs_glyph_cache_clear_index(FS_InternalState* st) {
    if (!st || !st->glyph_hash_slots || st->glyph_hash_capacity == 0u) {
        return;
    }
    memset(st->glyph_hash_slots, 0, st->glyph_hash_capacity * sizeof(uint32_t));
}

static FS_GlyphEntry* fs_glyph_cache_find(
    FS_InternalState* st,
    uint32_t glyph_key,
    uint32_t key_kind,
    uint32_t bake_px_q
) {
    if (!st || st->glyph_count == 0u) {
        return NULL;
    }
    if (!st->glyph_hash_slots || st->glyph_hash_capacity == 0u) {
        if (!fs_glyph_cache_rebuild(st, st->glyph_count * 2u)) {
            return NULL;
        }
    }

    const size_t cap = st->glyph_hash_capacity;
    const size_t mask = cap - 1u;
    size_t slot = (size_t)fs_glyph_cache_hash_key(glyph_key, key_kind, bake_px_q) & mask;
    for (size_t probe = 0u; probe < cap; ++probe) {
        const uint32_t packed = st->glyph_hash_slots[slot];
        if (packed == 0u) {
            return NULL;
        }
        const size_t idx = (size_t)(packed - 1u);
        if (idx < st->glyph_count) {
            FS_GlyphEntry* g = &st->glyphs[idx];
            if (g->glyph_key == glyph_key && g->key_kind == key_kind && g->bake_px_q == bake_px_q) {
                return g;
            }
        }
        slot = (slot + 1u) & mask;
    }
    return NULL;
}

static bool fs_glyph_cache_insert_index(
    FS_InternalState* st,
    size_t glyph_index
) {
    if (!st || glyph_index >= st->glyph_count) {
        return false;
    }
    if (!st->glyph_hash_slots || st->glyph_hash_capacity == 0u ||
        (st->glyph_count * 10u) >= (st->glyph_hash_capacity * 7u)) {
        size_t target = st->glyph_hash_capacity ? st->glyph_hash_capacity * 2u : 64u;
        if (target < st->glyph_count * 2u) {
            target = st->glyph_count * 2u;
        }
        if (!fs_glyph_cache_rebuild(st, target)) {
            return false;
        }
    }

    const FS_GlyphEntry* g = &st->glyphs[glyph_index];
    const size_t cap = st->glyph_hash_capacity;
    const size_t mask = cap - 1u;
    size_t slot = (size_t)fs_glyph_cache_hash_key(g->glyph_key, g->key_kind, g->bake_px_q) & mask;
    for (size_t probe = 0u; probe < cap; ++probe) {
        const uint32_t packed = st->glyph_hash_slots[slot];
        if (packed == 0u) {
            st->glyph_hash_slots[slot] = (uint32_t)(glyph_index + 1u);
            return true;
        }
        const size_t idx = (size_t)(packed - 1u);
        if (idx < st->glyph_count) {
            const FS_GlyphEntry* cur = &st->glyphs[idx];
            if (cur->glyph_key == g->glyph_key && cur->key_kind == g->key_kind && cur->bake_px_q == g->bake_px_q) {
                st->glyph_hash_slots[slot] = (uint32_t)(glyph_index + 1u);
                return true;
            }
        }
        slot = (slot + 1u) & mask;
    }
    return false;
}

static void fs_invalidate_cached_glyph(FS_InternalState* st, uint32_t key_kind, uint32_t glyph_key) {
    if (!st || st->glyph_count == 0u) {
        return;
    }
    size_t write = 0u;
    for (size_t read = 0u; read < st->glyph_count; ++read) {
        FS_GlyphEntry g = st->glyphs[read];
        if (g.key_kind == key_kind && g.glyph_key == glyph_key) {
            continue;
        }
        if (write != read) {
            st->glyphs[write] = g;
        }
        write += 1u;
    }
    st->glyph_count = write;
    if (st->glyph_count == 0u) {
        fs_glyph_cache_clear_index(st);
    } else {
        (void)fs_glyph_cache_rebuild(st, st->glyph_hash_capacity);
    }
}

static int fs_sequence_entry_sort_desc(const void* lhs, const void* rhs) {
    const FS_ImageSequenceEntry* a = (const FS_ImageSequenceEntry*)lhs;
    const FS_ImageSequenceEntry* b = (const FS_ImageSequenceEntry*)rhs;
    if (a->cp_count == b->cp_count) {
        return 0;
    }
    return (a->cp_count > b->cp_count) ? -1 : 1;
}

static bool fs_ensure_glyph_capacity(FS_InternalState* st, size_t required) {
    if (!st) {
        return false;
    }
    if (required <= st->glyph_capacity) {
        return true;
    }
    size_t new_cap = st->glyph_capacity ? st->glyph_capacity : 512u;
    while (new_cap < required) {
        if (new_cap > (SIZE_MAX / 2u)) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_GlyphEntry* grown = (FS_GlyphEntry*)realloc(st->glyphs, new_cap * sizeof(FS_GlyphEntry));
    if (!grown) {
        return false;
    }
    st->glyphs = grown;
    st->glyph_capacity = new_cap;
    return true;
}

static bool fs_find_or_create_glyph(
    FS_Core* core,
    uint32_t glyph_key,
    uint32_t key_kind,
    float font_px,
    FS_GlyphEntry** out_glyph,
    float* out_layout_scale
) {
    if (!core || !out_glyph) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (key_kind != FS_IMAGE_FONT_KIND && st->font_count == 0u) {
        return false;
    }

    const float requested_px = font_px < 1.0f ? 1.0f : font_px;
    const float bake_px = fs_get_text_bake_px(requested_px);
    const uint32_t bake_px_q = fs_quantize_font_size(bake_px);
    const float layout_scale = requested_px / bake_px;

    FS_GlyphEntry* cached = fs_glyph_cache_find(st, glyph_key, key_kind, bake_px_q);
    if (cached) {
        *out_glyph = cached;
        if (out_layout_scale) {
            *out_layout_scale = layout_scale;
        }
        return true;
    }

    if (!fs_ensure_glyph_capacity(st, st->glyph_count + 1u)) {
        return false;
    }

    FS_GlyphEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.glyph_key = glyph_key;
    entry.key_kind = key_kind;
    entry.bake_px_q = bake_px_q;
    entry.bake_px = bake_px;
    entry.sdf_radius_px = 8.0f;
    entry.sdf_onedge = 0.5f;
    entry.sdf_pixel_dist_scale = 1.0f;
    entry.text_flags = 0u;
    entry.font_slot = 0u;

    FS_FontGlyphBitmap glyph_bitmap;
    memset(&glyph_bitmap, 0, sizeof(glyph_bitmap));
    bool need_free_pixels = false;

    if (key_kind == FS_IMAGE_FONT_KIND) {
        const uint32_t image_font_id = fs_unpack_image_key_font_id(glyph_key);
        const uint32_t image_glyph_id = fs_unpack_image_key_glyph_id(glyph_key);
        FS_ImageFontState* image_font = fs_find_image_font(st, image_font_id);
        if (!image_font || image_glyph_id >= image_font->glyph_slot_count) {
            return false;
        }
        FS_ImageGlyphSlot* slot = &image_font->glyph_slots[image_glyph_id];
        if (!slot->loaded || !slot->rgba || slot->width == 0u || slot->height == 0u) {
            return false;
        }

        glyph_bitmap.pixels = slot->rgba;
        glyph_bitmap.width = slot->width;
        glyph_bitmap.height = slot->height;
        glyph_bitmap.pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8;
        glyph_bitmap.sdf_radius_px = 1.0f;
        glyph_bitmap.sdf_onedge = 0.5f;
        glyph_bitmap.sdf_pixel_dist_scale = 1.0f;
        fs_apply_emoji_layout_from_bitmap(&entry, bake_px, slot->width, slot->height);
    } else {
        if (!st->font_backend || !st->font_backend->get_glyph_sdf) {
            return false;
        }
        bool glyph_ok = false;
        uint32_t selected_font_slot = 0u;
        if (key_kind == 1u && st->font_backend->get_glyph_sdf_by_index) {
            if (st->font_count == 0u) {
                return false;
            }
            if (st->font_backend->get_glyph_sdf_by_index(st->fonts[0], glyph_key, bake_px, &glyph_bitmap)) {
                glyph_ok = true;
                selected_font_slot = 0u;
            }
        } else {
            const uint32_t count = st->font_count > FS_MAX_FONT_FALLBACKS ? FS_MAX_FONT_FALLBACKS : st->font_count;
            for (uint32_t fi = 0u; fi < count; ++fi) {
                if (!st->fonts[fi]) {
                    continue;
                }
                if (st->font_backend->get_glyph_sdf(st->fonts[fi], glyph_key, bake_px, &glyph_bitmap)) {
                    glyph_ok = true;
                    selected_font_slot = fi;
                    break;
                }
            }
        }
        if (!glyph_ok) {
            return false;
        }
        entry.font_slot = (uint8_t)selected_font_slot;
        need_free_pixels = true;
        if (glyph_bitmap.pixel_format != FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8) {
            glyph_bitmap.pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_SDF_R8;
            entry.text_flags = 0u;
            entry.advance = glyph_bitmap.advance;
            entry.bearing_x = (float)glyph_bitmap.offset_x;
            entry.bearing_y = (float)glyph_bitmap.offset_y;
            entry.atlas_width = (float)glyph_bitmap.width;
            entry.atlas_height = (float)glyph_bitmap.height;
            entry.sdf_radius_px = glyph_bitmap.sdf_radius_px;
            entry.sdf_onedge = glyph_bitmap.sdf_onedge;
            entry.sdf_pixel_dist_scale = glyph_bitmap.sdf_pixel_dist_scale;
        } else {
            fs_apply_emoji_layout_from_bitmap(&entry, bake_px, glyph_bitmap.width, glyph_bitmap.height);
        }
    }

    if (glyph_bitmap.pixels && glyph_bitmap.width > 0u && glyph_bitmap.height > 0u) {
        uint32_t gx = 0, gy = 0;
        if (!fs_alloc_from_atlas(
                core->glyph_atlas_width,
                core->glyph_atlas_height,
                &core->glyph_atlas_cursor_x,
                &core->glyph_atlas_cursor_y,
                &core->glyph_atlas_row_height,
                FS_GLYPH_ATLAS_PADDING,
                glyph_bitmap.width,
                glyph_bitmap.height,
                &gx,
                &gy
            )) {
            if (need_free_pixels && st->font_backend->free_glyph_pixels) {
                st->font_backend->free_glyph_pixels(glyph_bitmap.pixels);
            }
            return false;
        }

        if (!fs_upload_glyph_with_mips(
                core,
                gx,
                gy,
                glyph_bitmap.width,
                glyph_bitmap.height,
                glyph_bitmap.pixels,
                glyph_bitmap.pixel_format,
                glyph_bitmap.sdf_onedge,
                glyph_bitmap.sdf_pixel_dist_scale
            )) {
            if (need_free_pixels && st->font_backend->free_glyph_pixels) {
                st->font_backend->free_glyph_pixels(glyph_bitmap.pixels);
            }
            return false;
        }
        if (need_free_pixels && st->font_backend->free_glyph_pixels) {
            st->font_backend->free_glyph_pixels(glyph_bitmap.pixels);
        }

        entry.uv_min[0] = (float)gx / (float)core->glyph_atlas_width;
        entry.uv_min[1] = (float)gy / (float)core->glyph_atlas_height;
        entry.uv_max[0] = (float)(gx + glyph_bitmap.width) / (float)core->glyph_atlas_width;
        entry.uv_max[1] = (float)(gy + glyph_bitmap.height) / (float)core->glyph_atlas_height;
    } else if (need_free_pixels && glyph_bitmap.pixels && st->font_backend->free_glyph_pixels) {
        st->font_backend->free_glyph_pixels(glyph_bitmap.pixels);
    }

    const size_t new_index = st->glyph_count;
    st->glyphs[new_index] = entry;
    *out_glyph = &st->glyphs[new_index];
    if (out_layout_scale) {
        *out_layout_scale = layout_scale;
    }
    st->glyph_count += 1u;
    if (!fs_glyph_cache_insert_index(st, new_index)) {
        (void)fs_glyph_cache_rebuild(st, st->glyph_hash_capacity);
    }
    return true;
}

static float fs_get_kerning_advance(
    FS_Core* core,
    uint32_t left_codepoint,
    uint32_t right_codepoint,
    float font_px,
    uint8_t font_slot
) {
    if (!core || font_px <= 0.0f) {
        return 0.0f;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || st->font_count == 0u || !st->font_backend || !st->font_backend->get_kerning_advance) {
        return 0.0f;
    }
    if ((uint32_t)font_slot >= st->font_count || !st->fonts[font_slot]) {
        return 0.0f;
    }
    return st->font_backend->get_kerning_advance(st->fonts[font_slot], left_codepoint, right_codepoint, font_px);
}

FS_Core* fs_core_create(
    WGPUDevice device,
    WGPUQueue queue,
    WGPUTextureFormat target_format,
    uint32_t width,
    uint32_t height
) {
    if (!device || !queue) {
        return NULL;
    }
    FS_Core* core = (FS_Core*)calloc(1u, sizeof(FS_Core));
    if (!core) {
        return NULL;
    }
    if (!fs_core_init(core, device, queue, target_format, width, height)) {
        free(core);
        return NULL;
    }
    return core;
}

void fs_core_destroy(FS_Core* core) {
    if (!core) {
        return;
    }
    fs_core_shutdown(core);
    free(core);
}

bool fs_core_init(
    FS_Core* core,
    WGPUDevice device,
    WGPUQueue queue,
    WGPUTextureFormat target_format,
    uint32_t width,
    uint32_t height
) {
    if (!core || !device || !queue) {
        return false;
    }
    memset(core, 0, sizeof(*core));
    core->device = device;
    core->queue = queue;
    core->target_format = target_format;
    core->width = width;
    core->height = height;
    core->render_sample_count = 1u;
    core->clip_aa_mode_override = -1;
    core->context_lost = false;
    core->context_attributes.alpha = true;
    core->context_attributes.premultiplied_alpha = (FS_RENDER_PREMULTIPLIED_ALPHA != 0);
    core->context_attributes.antialias = true;
    core->context_attributes.depth = false;
    core->context_attributes.stencil = false;
    core->context_attributes.preserve_drawing_buffer = false;
    core->clip_frame_index = 1u;
    core->clip_layer_reuse_reserve = 2u;
    core->clip_cache_enabled = false;
    fs_clip_diag_reset_frame(core);

    FS_InternalState* st = (FS_InternalState*)calloc(1, sizeof(FS_InternalState));
    if (!st) {
        return false;
    }
    core->internal_state = st;
    st->image_backend = NULL;
    st->font_backend = NULL;
    st->owner_core = core;
    st->current_transform = fs_transform_identity_value();
    fs_style_reset_state(st);
    fs_clip_reset_state(st);

    core->command_capacity = 1024u;
    core->commands = (FS_Command*)malloc(core->command_capacity * sizeof(FS_Command));
    core->command_state_capacity = core->command_capacity;
    core->command_states = (FS_CommandStateGPU*)malloc(core->command_state_capacity * sizeof(FS_CommandStateGPU));
    if (!core->commands || !core->command_states) {
        fs_release_resources(core);
        return false;
    }

    core->command_buffer_size = core->command_capacity * sizeof(FS_Command);
    core->command_state_buffer_size = core->command_state_capacity * sizeof(FS_CommandStateGPU);
    core->vertex_buffer_size = core->command_capacity * 6u * sizeof(FS_VertexGPU);
    core->command_buffer = fs_create_buffer(
        core->device,
        "FS Command Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        core->command_buffer_size
    );
    core->command_state_buffer = fs_create_buffer(
        core->device,
        "FS Command State Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        core->command_state_buffer_size
    );
    core->vertex_buffer = fs_create_buffer(
        core->device,
        "FS Vertex Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
        core->vertex_buffer_size
    );
    core->uniform_buffer = fs_create_buffer(
        core->device,
        "FS Uniform Buffer",
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        sizeof(FS_Uniforms)
    );
    core->clip_dispatch_uniform_buffer = fs_create_buffer(
        core->device,
        "FS Clip Dispatch Uniform Buffer",
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        sizeof(FS_ClipDispatchUniforms)
    );
    core->clip_layer_uniform_buffer = fs_create_buffer(
        core->device,
        "FS Clip Layer Uniform Buffer",
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        sizeof(FS_ClipLayerUniforms)
    );
    if (!core->command_buffer || !core->command_state_buffer || !core->vertex_buffer || !core->uniform_buffer ||
        !core->clip_dispatch_uniform_buffer || !core->clip_layer_uniform_buffer) {
        fs_release_resources(core);
        return false;
    }

    if (!fs_create_image_atlas(core) || !fs_create_glyph_atlas(core) || !fs_create_clip_mask(core) ||
        !fs_create_msaa_color_target(core)) {
        fs_release_resources(core);
        return false;
    }
    if (!fs_upload_default_image(core)) {
        fs_release_resources(core);
        return false;
    }
    if (!fs_create_pipelines_and_bindings(core)) {
        fs_release_resources(core);
        return false;
    }
    if (!fs_ensure_canvas_shadow(core)) {
        fs_release_resources(core);
        return false;
    }
    if (core->canvas_shadow_rgba && core->canvas_shadow_size > 0u) {
        const size_t clear_size = (size_t)core->width * (size_t)core->height * 4u;
        memset(core->canvas_shadow_rgba, 0, clear_size);
    }
    core->canvas_shadow_serial = 0u;
    core->canvas_shadow_serial = 0u;
    core->canvas_image_data_handle_valid = 0u;
    return true;
}

bool fs_core_is_context_lost(const FS_Core* core) {
    if (!core) {
        return true;
    }
    return core->context_lost;
}

bool fs_core_get_context_attributes(const FS_Core* core, FS_ContextAttributes* out_attributes) {
    if (!core || !out_attributes) {
        return false;
    }
    *out_attributes = core->context_attributes;
    return true;
}

bool fs_core_get_clip_diagnostics(const FS_Core* core, FS_ClipDiagnostics* out_diagnostics) {
    if (!core || !out_diagnostics) {
        return false;
    }
    memset(out_diagnostics, 0, sizeof(*out_diagnostics));
    out_diagnostics->requests_this_frame = core->clip_requests_this_frame;
    out_diagnostics->cache_hits_this_frame = core->clip_cache_hits_this_frame;
    out_diagnostics->jobs_enqueued_this_frame = core->clip_jobs_enqueued_this_frame;
    out_diagnostics->layer_reuses_this_frame = core->clip_layer_reuses_this_frame;
    out_diagnostics->failures_this_frame = core->clip_failures_this_frame;
    out_diagnostics->layers_used_this_frame = core->clip_layers_used_this_frame;
    out_diagnostics->layer_capacity = core->clip_mask_layers;
    out_diagnostics->last_failure_reason = (FS_ClipFailureReason)core->clip_last_failure_reason;
    out_diagnostics->last_failure_path_segments = core->clip_last_failure_path_segments;
    out_diagnostics->last_failure_edge_count = core->clip_last_failure_edge_count;
    out_diagnostics->dispatch_batches_this_frame = core->clip_dispatch_batches_this_frame;
    out_diagnostics->dispatch_valid_jobs_this_frame = core->clip_dispatch_valid_jobs_this_frame;
    out_diagnostics->dispatch_pixels_ideal_this_frame = core->clip_dispatch_pixels_ideal_this_frame;
    out_diagnostics->dispatch_pixels_estimated_this_frame = core->clip_dispatch_pixels_estimated_this_frame;
    out_diagnostics->dispatch_pixels_waste_this_frame = core->clip_dispatch_pixels_waste_this_frame;
    memcpy(
        out_diagnostics->dispatch_bucket_jobs_this_frame,
        core->clip_dispatch_bucket_jobs_this_frame,
        sizeof(out_diagnostics->dispatch_bucket_jobs_this_frame)
    );
    out_diagnostics->oriented_quad_commands_this_frame = core->clip_oriented_quad_commands_this_frame;
    out_diagnostics->oriented_quad_clipped_this_frame = core->clip_oriented_quad_clipped_this_frame;
    return true;
}

void fs_core_shutdown(FS_Core* core) {
    if (!core) {
        return;
    }
    fs_release_resources(core);
}

void fs_core_resize(FS_Core* core, uint32_t width, uint32_t height) {
    if (!core) {
        return;
    }
    if (core->width == width && core->height == height) {
        return;
    }
    core->width = width;
    core->height = height;
    if (!fs_ensure_canvas_shadow(core)) {
        fs_mark_context_lost(core);
        return;
    }
    if (core->canvas_shadow_rgba && core->canvas_shadow_size > 0u) {
        const size_t clear_size = (size_t)core->width * (size_t)core->height * 4u;
        memset(core->canvas_shadow_rgba, 0, clear_size);
    }
    core->canvas_shadow_serial = 0u;
    core->canvas_readback_serial = 0u;
    core->canvas_readback_submission = 0u;
    core->canvas_readback_submission_valid = 0u;
    if (core->canvas_readback_buffer &&
        (core->canvas_readback_width != width || core->canvas_readback_height != height)) {
        if (core->canvas_readback_mapped) {
            wgpuBufferUnmap(core->canvas_readback_buffer);
            core->canvas_readback_mapped = 0u;
        }
        wgpuBufferRelease(core->canvas_readback_buffer);
        core->canvas_readback_buffer = NULL;
        core->canvas_readback_buffer_size = 0u;
        core->canvas_readback_row_bytes = 0u;
        core->canvas_readback_padded_row_bytes = 0u;
        core->canvas_readback_width = 0u;
        core->canvas_readback_height = 0u;
    }
    core->canvas_image_data_handle_valid = 0u;
    FS_InternalState* st = fs_state(core);
    if (!st || !core->device || !core->render_bgl || !core->clip_mask_texture) {
        fs_mark_context_lost(core);
        return;
    }
    // Remove only uploads targeting the old clip texture; keep other atlas uploads intact.
    fs_discard_pending_uploads_for_texture(core, core->clip_mask_texture);
    core->clip_mask_next_layer = 0u;
    if (core->clip_mask_sampler) {
        wgpuSamplerRelease(core->clip_mask_sampler);
        core->clip_mask_sampler = NULL;
    }
    if (core->clip_mask_view) {
        wgpuTextureViewRelease(core->clip_mask_view);
        core->clip_mask_view = NULL;
    }
    if (core->clip_mask_texture) {
        wgpuTextureRelease(core->clip_mask_texture);
        core->clip_mask_texture = NULL;
    }
    if (fs_create_clip_mask(core) && fs_create_msaa_color_target(core)) {
        if (!fs_recreate_render_bind_group(core)) {
            fs_mark_context_lost(core);
        }
        if (core->clip_edge_buffer && core->clip_job_buffer && core->clip_dispatch_uniform_buffer) {
            if (!fs_recreate_clip_compute_bind_group(core)) {
                fs_mark_context_lost(core);
            }
        }
    } else {
        fs_mark_context_lost(core);
    }
    fs_clip_reset_state(st);
}

void fs_core_begin_commands(FS_Core* core) {
    if (!core) {
        return;
    }
    core->clip_frame_index += 1u;
    if (core->clip_frame_index == 0u) {
        core->clip_frame_index = 1u;
        if (core->clip_mask_layer_last_used_frame && core->clip_mask_layers > 0u) {
            memset(core->clip_mask_layer_last_used_frame, 0, (size_t)core->clip_mask_layers * sizeof(uint32_t));
        }
    }
    core->command_count = 0u;
    core->clip_mask_next_layer = 0u;
    core->clip_edge_count = 0u;
    core->clip_job_count = 0u;
    // Disable cross-frame hash hits, but keep layer occupancy/bounds metadata so
    // reused layers can clear union(old_bounds, new_bounds) and avoid stale masks.
    if (core->clip_mask_layer_hash_valid && core->clip_mask_layers > 0u) {
        memset(core->clip_mask_layer_hash_valid, 0, (size_t)core->clip_mask_layers * sizeof(uint8_t));
    }
    fs_clip_diag_reset_frame(core);
}

void fs_context_reset(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return;
    }
    fs_state_stack_clear(st);
    st->current_transform = fs_transform_identity_value();
    fs_style_reset_state(st);
    fs_clip_reset_state(st);
    fs_path_begin(core);
}

static bool fs_execute_clip_jobs(FS_Core* core, WGPUCommandEncoder encoder) {
    if (!core || !encoder) {
        return false;
    }
    if (core->clip_job_count == 0u) {
        return true;
    }
    if (!core->clip_compute_pipeline || !core->clip_compute_bgl || !core->clip_dispatch_uniform_buffer) {
        return false;
    }
    if (!fs_ensure_clip_edge_gpu_capacity(core, core->clip_edge_count)) {
        return false;
    }

    wgpuQueueWriteBuffer(
        core->queue,
        core->clip_edge_local_buffer,
        0u,
        core->clip_edge_cpu,
        core->clip_edge_count * sizeof(FS_ClipEdgeGPU)
    );

    const FS_ClipJobGPU* src_jobs = (const FS_ClipJobGPU*)core->clip_job_cpu;
    const FS_ClipJobTransformGPU* src_xforms = (const FS_ClipJobTransformGPU*)core->clip_job_xform_cpu;
    size_t valid_job_count = 0u;
    for (size_t i = 0u; i < core->clip_job_count; ++i) {
        const FS_ClipJobGPU* j = &src_jobs[i];
        const uint32_t w = (j->clear_max_x > j->clear_min_x) ? (j->clear_max_x - j->clear_min_x) : 0u;
        const uint32_t h = (j->clear_max_y > j->clear_min_y) ? (j->clear_max_y - j->clear_min_y) : 0u;
        if (w == 0u || h == 0u) {
            continue;
        }
        valid_job_count += 1u;
    }
    if (valid_job_count == 0u) {
        core->clip_edge_count = 0u;
        core->clip_job_count = 0u;
        return true;
    }

    FS_ClipJobGPU* valid_jobs = (FS_ClipJobGPU*)malloc(valid_job_count * sizeof(FS_ClipJobGPU));
    if (!valid_jobs) {
        return false;
    }
    FS_ClipJobTransformGPU* valid_xforms =
        (FS_ClipJobTransformGPU*)malloc(valid_job_count * sizeof(FS_ClipJobTransformGPU));
    if (!valid_xforms) {
        free(valid_jobs);
        return false;
    }
    uint8_t* valid_bucket_ids = (uint8_t*)malloc(valid_job_count * sizeof(uint8_t));
    if (!valid_bucket_ids) {
        free(valid_xforms);
        free(valid_jobs);
        return false;
    }

    enum { FS_CLIP_DISPATCH_BUCKET_COUNT = 6 };
    const uint32_t bucket_limits[FS_CLIP_DISPATCH_BUCKET_COUNT] = {64u, 128u, 256u, 512u, 1024u, UINT32_MAX};
    size_t bucket_counts[FS_CLIP_DISPATCH_BUCKET_COUNT] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t bucket_max_edges[FS_CLIP_DISPATCH_BUCKET_COUNT] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t bucket_max_w[FS_CLIP_DISPATCH_BUCKET_COUNT] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t bucket_max_h[FS_CLIP_DISPATCH_BUCKET_COUNT] = {0u, 0u, 0u, 0u, 0u, 0u};
    uint64_t ideal_pixels = 0u;

    size_t valid_index = 0u;
    for (size_t i = 0u; i < core->clip_job_count; ++i) {
        const FS_ClipJobGPU* j = &src_jobs[i];
        const uint32_t w = (j->clear_max_x > j->clear_min_x) ? (j->clear_max_x - j->clear_min_x) : 0u;
        const uint32_t h = (j->clear_max_y > j->clear_min_y) ? (j->clear_max_y - j->clear_min_y) : 0u;
        if (w == 0u || h == 0u) {
            continue;
        }
        uint32_t dim = (w > h) ? w : h;
        uint8_t bucket = (uint8_t)(FS_CLIP_DISPATCH_BUCKET_COUNT - 1u);
        for (uint32_t b = 0u; b < FS_CLIP_DISPATCH_BUCKET_COUNT; ++b) {
            if (dim <= bucket_limits[b]) {
                bucket = (uint8_t)b;
                break;
            }
        }
        valid_jobs[valid_index] = *j;
        if (src_xforms) {
            valid_xforms[valid_index] = src_xforms[i];
        } else {
            FS_ClipJobTransformGPU identity = {
                .xform0 = {1.0f, 0.0f, 0.0f, 1.0f},
                .xform1 = {0.0f, 0.0f, 0.0f, 0.0f}
            };
            valid_xforms[valid_index] = identity;
        }
        valid_bucket_ids[valid_index] = bucket;
        bucket_counts[bucket] += 1u;
        if (j->edge_count > bucket_max_edges[bucket]) {
            bucket_max_edges[bucket] = j->edge_count;
        }
        if (w > bucket_max_w[bucket]) {
            bucket_max_w[bucket] = w;
        }
        if (h > bucket_max_h[bucket]) {
            bucket_max_h[bucket] = h;
        }
        ideal_pixels += (uint64_t)w * (uint64_t)h;
        valid_index += 1u;
    }

    core->clip_dispatch_valid_jobs_this_frame +=
        (valid_job_count > (size_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)valid_job_count;
    core->clip_dispatch_pixels_ideal_this_frame += ideal_pixels;
    for (uint32_t b = 0u; b < FS_CLIP_DISPATCH_BUCKET_COUNT; ++b) {
        const uint32_t add = (bucket_counts[b] > (size_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)bucket_counts[b];
        core->clip_dispatch_bucket_jobs_this_frame[b] += add;
    }

    size_t bucket_starts[FS_CLIP_DISPATCH_BUCKET_COUNT] = {0u, 0u, 0u, 0u, 0u, 0u};
    size_t bucket_cursor[FS_CLIP_DISPATCH_BUCKET_COUNT] = {0u, 0u, 0u, 0u, 0u, 0u};
    size_t total_slots = 0u;
    for (uint32_t b = 0u; b < FS_CLIP_DISPATCH_BUCKET_COUNT; ++b) {
        // Align to 8 entries so both job buffer (64-byte stride) and xform buffer
        // (32-byte stride) produce 256-byte aligned storage offsets.
        total_slots = (total_slots + 7u) & ~(size_t)7u;
        bucket_starts[b] = total_slots;
        bucket_cursor[b] = total_slots;
        total_slots += bucket_counts[b];
    }
    if (total_slots == 0u) {
        free(valid_bucket_ids);
        free(valid_xforms);
        free(valid_jobs);
        core->clip_edge_count = 0u;
        core->clip_job_count = 0u;
        return true;
    }
    if (!fs_ensure_clip_job_gpu_capacity(core, total_slots)) {
        free(valid_bucket_ids);
        free(valid_xforms);
        free(valid_jobs);
        return false;
    }
    if (!fs_ensure_clip_job_transform_gpu_capacity(core, total_slots)) {
        free(valid_bucket_ids);
        free(valid_xforms);
        free(valid_jobs);
        return false;
    }

    FS_ClipJobGPU* ordered_jobs = (FS_ClipJobGPU*)calloc(total_slots, sizeof(FS_ClipJobGPU));
    if (!ordered_jobs) {
        free(valid_bucket_ids);
        free(valid_xforms);
        free(valid_jobs);
        return false;
    }
    FS_ClipJobTransformGPU* ordered_xforms =
        (FS_ClipJobTransformGPU*)calloc(total_slots, sizeof(FS_ClipJobTransformGPU));
    if (!ordered_xforms) {
        free(ordered_jobs);
        free(valid_bucket_ids);
        free(valid_xforms);
        free(valid_jobs);
        return false;
    }
    for (size_t i = 0u; i < valid_job_count; ++i) {
        const uint8_t b = valid_bucket_ids[i];
        const size_t dst = bucket_cursor[b]++;
        ordered_jobs[dst] = valid_jobs[i];
        ordered_xforms[dst] = valid_xforms[i];
    }
    free(valid_bucket_ids);
    free(valid_xforms);
    free(valid_jobs);

    wgpuQueueWriteBuffer(
        core->queue,
        core->clip_job_buffer,
        0u,
        ordered_jobs,
        total_slots * sizeof(FS_ClipJobGPU)
    );
    wgpuQueueWriteBuffer(
        core->queue,
        core->clip_job_xform_buffer,
        0u,
        ordered_xforms,
        total_slots * sizeof(FS_ClipJobTransformGPU)
    );

    FS_ClipDispatchUniforms dispatch_uniforms = {
        .job_count = 0u,
        .job_offset = 0u,
        .viewport_width = core->width,
        .viewport_height = core->height,
        .aa_mode = 1u,
        ._pad0 = 0u,
        ._pad1 = 0u,
        ._pad2 = 0u
    };
    if (core->clip_aa_mode_override >= 0) {
        uint32_t forced = (uint32_t)core->clip_aa_mode_override;
        if (forced > 3u) {
            forced = 3u;
        }
        dispatch_uniforms.aa_mode = forced;
    } else {
        const uint32_t max_dim = (core->width > core->height) ? core->width : core->height;
        if (max_dim >= 3000u) {
            dispatch_uniforms.aa_mode = 3u;
        } else if (max_dim >= 1800u) {
            dispatch_uniforms.aa_mode = 2u;
        } else if (max_dim >= 1100u) {
            dispatch_uniforms.aa_mode = 1u;
        } else {
            dispatch_uniforms.aa_mode = 0u;
        }
    }

    wgpuQueueWriteBuffer(
        core->queue,
        core->clip_dispatch_uniform_buffer,
        0u,
        &dispatch_uniforms,
        sizeof(dispatch_uniforms)
    );

    if (!core->clip_edge_transform_pipeline || !core->clip_edge_transform_bgl) {
        free(ordered_xforms);
        free(ordered_jobs);
        return false;
    }

    WGPUComputePassDescriptor edge_transform_pass_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Edge Transform Pass",
        .timestampWrites = NULL
    };
    WGPUComputePassEncoder edge_transform_pass =
        wgpuCommandEncoderBeginComputePass(encoder, &edge_transform_pass_desc);
    if (!edge_transform_pass) {
        free(ordered_xforms);
        free(ordered_jobs);
        return false;
    }
    wgpuComputePassEncoderSetPipeline(edge_transform_pass, core->clip_edge_transform_pipeline);
    bool edge_transform_ok = true;
    for (uint32_t b = 0u; b < FS_CLIP_DISPATCH_BUCKET_COUNT; ++b) {
        const size_t bucket_count = bucket_counts[b];
        const uint32_t max_edges = bucket_max_edges[b];
        if (bucket_count == 0u || max_edges == 0u) {
            continue;
        }

        const uint64_t job_offset_bytes = (uint64_t)bucket_starts[b] * (uint64_t)sizeof(FS_ClipJobGPU);
        const uint64_t xform_offset_bytes =
            (uint64_t)bucket_starts[b] * (uint64_t)sizeof(FS_ClipJobTransformGPU);
        const uint64_t job_size_bytes = (uint64_t)bucket_count * (uint64_t)sizeof(FS_ClipJobGPU);
        const uint64_t xform_size_bytes =
            (uint64_t)bucket_count * (uint64_t)sizeof(FS_ClipJobTransformGPU);

        WGPUBindGroup edge_bg = fs_create_clip_edge_transform_bind_group_range(
            core,
            job_offset_bytes,
            job_size_bytes,
            xform_offset_bytes,
            xform_size_bytes
        );
        if (!edge_bg) {
            edge_transform_ok = false;
            break;
        }

        wgpuComputePassEncoderSetBindGroup(edge_transform_pass, 0, edge_bg, 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(
            edge_transform_pass,
            (max_edges + 63u) / 64u,
            1u,
            (uint32_t)bucket_count
        );
        wgpuBindGroupRelease(edge_bg);
    }
    wgpuComputePassEncoderEnd(edge_transform_pass);
    if (!edge_transform_ok) {
        free(ordered_xforms);
        free(ordered_jobs);
        return false;
    }

    WGPUComputePassDescriptor clip_pass_desc = {
        .nextInChain = NULL,
        .label = "FS Clip Compute Pass",
        .timestampWrites = NULL
    };
    WGPUComputePassEncoder clip_pass = wgpuCommandEncoderBeginComputePass(encoder, &clip_pass_desc);
    if (!clip_pass) {
        free(ordered_xforms);
        free(ordered_jobs);
        return false;
    }
    wgpuComputePassEncoderSetPipeline(clip_pass, core->clip_compute_pipeline);

    bool pass_ok = true;
    uint32_t dispatch_batches = 0u;
    uint64_t estimated_pixels = 0u;
    for (uint32_t b = 0u; b < FS_CLIP_DISPATCH_BUCKET_COUNT; ++b) {
        const size_t bucket_count = bucket_counts[b];
        if (bucket_count == 0u) {
            continue;
        }
        const uint32_t max_w = bucket_max_w[b];
        const uint32_t max_h = bucket_max_h[b];
        if (max_w == 0u || max_h == 0u) {
            continue;
        }
        estimated_pixels +=
            (uint64_t)max_w * (uint64_t)max_h *
            ((bucket_count > (size_t)UINT32_MAX) ? (uint64_t)UINT32_MAX : (uint64_t)bucket_count);
        dispatch_batches += 1u;

        const uint64_t job_offset_bytes = (uint64_t)bucket_starts[b] * (uint64_t)sizeof(FS_ClipJobGPU);
        const uint64_t job_size_bytes = (uint64_t)bucket_count * (uint64_t)sizeof(FS_ClipJobGPU);
        WGPUBindGroup bucket_bg = fs_create_clip_compute_bind_group_range(core, job_offset_bytes, job_size_bytes);
        if (!bucket_bg) {
            pass_ok = false;
            break;
        }

        wgpuComputePassEncoderSetBindGroup(clip_pass, 0, bucket_bg, 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(
            clip_pass,
            (max_w + 7u) / 8u,
            (max_h + 7u) / 8u,
            (uint32_t)bucket_count
        );
        wgpuBindGroupRelease(bucket_bg);
    }
    wgpuComputePassEncoderEnd(clip_pass);
    core->clip_dispatch_batches_this_frame += dispatch_batches;
    core->clip_dispatch_pixels_estimated_this_frame += estimated_pixels;
    if (estimated_pixels > ideal_pixels) {
        core->clip_dispatch_pixels_waste_this_frame += (estimated_pixels - ideal_pixels);
    }

    free(ordered_xforms);
    free(ordered_jobs);
    if (!pass_ok) {
        return false;
    }
    core->clip_edge_count = 0u;
    core->clip_job_count = 0u;
    return true;
}

static bool fs_update_clip_layer_uniform(FS_Core* core) {
    if (!core || !core->clip_layer_uniform_buffer) {
        return false;
    }
    FS_ClipLayerUniforms ubo;
    for (uint32_t i = 0u; i < FS_CLIP_MASK_LAYERS; ++i) {
        ubo.parent[i] = UINT32_MAX;
        ubo.min_x[i] = 0u;
        ubo.min_y[i] = 0u;
        ubo.max_x[i] = 0u;
        ubo.max_y[i] = 0u;
        if (core->clip_mask_layer_parent && i < core->clip_mask_layers) {
            ubo.parent[i] = core->clip_mask_layer_parent[i];
        }
        if (i < core->clip_mask_layers &&
            core->clip_mask_layer_has_data && core->clip_mask_layer_has_data[i] &&
            core->clip_mask_layer_min_x && core->clip_mask_layer_min_y &&
            core->clip_mask_layer_max_x && core->clip_mask_layer_max_y) {
            ubo.min_x[i] = core->clip_mask_layer_min_x[i];
            ubo.min_y[i] = core->clip_mask_layer_min_y[i];
            ubo.max_x[i] = core->clip_mask_layer_max_x[i];
            ubo.max_y[i] = core->clip_mask_layer_max_y[i];
        }
    }
    wgpuQueueWriteBuffer(
        core->queue,
        core->clip_layer_uniform_buffer,
        0u,
        &ubo,
        sizeof(ubo)
    );
    return true;
}

bool fs_core_encode(
    FS_Core* core,
    WGPUCommandEncoder encoder,
    WGPUTexture target_texture,
    WGPUTextureView target_view,
    float clear_r,
    float clear_g,
    float clear_b,
    float clear_a
) {
    if (!core || !encoder || !target_view) {
        return false;
    }
    if (core->context_lost) {
        return false;
    }
    if (!fs_flush_pending_texture_uploads(core, encoder)) {
        fs_mark_context_lost(core);
        return false;
    }
    if (!fs_execute_clip_jobs(core, encoder)) {
        fs_mark_context_lost(core);
        return false;
    }
    if (!fs_update_clip_layer_uniform(core)) {
        fs_mark_context_lost(core);
        return false;
    }
    if (!fs_ensure_gpu_capacity(core, core->command_count)) {
        fs_mark_context_lost(core);
        return false;
    }
    if (!core->render_bg) {
        fs_mark_context_lost(core);
        return false;
    }
    if (core->command_count > 0u && (!core->compute_pipeline || !core->compute_bg)) {
        fs_mark_context_lost(core);
        return false;
    }

    if (core->command_count > 0u) {
        wgpuQueueWriteBuffer(
            core->queue,
            core->command_buffer,
            0,
            core->commands,
            core->command_count * sizeof(FS_Command)
        );
        wgpuQueueWriteBuffer(
            core->queue,
            core->command_state_buffer,
            0,
            core->command_states,
            core->command_count * sizeof(FS_CommandStateGPU)
        );
    }

    FS_Uniforms uniforms = {
        .viewport = {(float)core->width, (float)core->height},
        .command_count = (uint32_t)core->command_count,
        .clip_enabled = 0u,
        .clip_min = {0.0f, 0.0f},
        .clip_max = {0.0f, 0.0f}
    };
    wgpuQueueWriteBuffer(core->queue, core->uniform_buffer, 0, &uniforms, sizeof(uniforms));

    if (core->command_count > 0u) {
        WGPUComputePassDescriptor compute_desc = {
            .nextInChain = NULL,
            .label = "FS Compute Pass",
            .timestampWrites = NULL
        };
        WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, &compute_desc);
        if (!compute_pass) {
            fs_mark_context_lost(core);
            fs_mark_context_lost(core);
            return false;
        }
        wgpuComputePassEncoderSetPipeline(compute_pass, core->compute_pipeline);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, core->compute_bg, 0, NULL);
        const uint32_t vertex_count = (uint32_t)(core->command_count * 6u);
        const uint32_t workgroups = (vertex_count + 127u) / 128u;
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroups, 1, 1);
        wgpuComputePassEncoderEnd(compute_pass);
    }

    WGPUTextureView color_view = target_view;
    WGPUTextureView resolve_view = NULL;
    if (core->render_sample_count > 1u && core->msaa_color_view) {
        color_view = core->msaa_color_view;
        resolve_view = target_view;
    }
    WGPURenderPassColorAttachment color = {
        .view = color_view,
        .resolveTarget = resolve_view,
        .loadOp = WGPULoadOp_Clear,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = {.r = clear_r, .g = clear_g, .b = clear_b, .a = clear_a}
    };
    WGPURenderPassDescriptor render_desc = {
        .nextInChain = NULL,
        .label = "FS Render Pass",
        .colorAttachmentCount = 1,
        .colorAttachments = &color,
        .depthStencilAttachment = NULL,
        .occlusionQuerySet = NULL,
        .timestampWrites = NULL
    };
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_desc);
    if (!pass) {
        fs_mark_context_lost(core);
        fs_mark_context_lost(core);
        return false;
    }
    wgpuRenderPassEncoderSetBindGroup(pass, 0, core->render_bg, 0, NULL);
    if (core->command_count > 0u) {
        const uint64_t draw_vertices = (uint64_t)(core->command_count * 6u);
        const uint64_t bytes = draw_vertices * sizeof(FS_VertexGPU);
        wgpuRenderPassEncoderSetVertexBuffer(pass, 0, core->vertex_buffer, 0, bytes);
        size_t start_cmd = 0u;
        while (start_cmd < core->command_count) {
            uint32_t pipeline_index = core->command_states[start_cmd].clip_meta[3];
            if (pipeline_index >= FS_RENDER_PIPELINE_COUNT || !core->render_pipelines[pipeline_index]) {
                pipeline_index = 0u;
            }
            WGPURenderPipeline pipeline = core->render_pipelines[pipeline_index];
            if (!pipeline) {
                fs_mark_context_lost(core);
                fs_mark_context_lost(core);
                wgpuRenderPassEncoderEnd(pass);
                return false;
            }
            size_t end_cmd = start_cmd + 1u;
            while (end_cmd < core->command_count) {
                const uint32_t next_idx = core->command_states[end_cmd].clip_meta[3];
                if (next_idx != pipeline_index) {
                    break;
                }
                end_cmd += 1u;
            }
            const uint32_t first_vertex = (uint32_t)(start_cmd * 6u);
            const uint32_t vertex_count = (uint32_t)((end_cmd - start_cmd) * 6u);
            wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            wgpuRenderPassEncoderDraw(pass, vertex_count, 1, first_vertex, 0);
            start_cmd = end_cmd;
        }
    }
    wgpuRenderPassEncoderEnd(pass);
    if (target_texture) {
        if (!fs_encode_canvas_readback_copy(core, encoder, target_texture)) {
            core->canvas_shadow_serial = 0u;
        }
    }
    return true;
}

void fs_core_notify_submission(FS_Core* core, WGPUSubmissionIndex submission_index) {
    if (!core) {
        return;
    }
    core->canvas_readback_submission = submission_index;
    core->canvas_readback_submission_valid = (submission_index != 0u) ? 1u : 0u;
}

bool fs_core_upload_image_rgba8(
    FS_Core* core,
    const uint8_t* rgba_pixels,
    uint32_t width,
    uint32_t height,
    FS_ImageHandle* out_handle
) {
    if (!core || !rgba_pixels || width == 0u || height == 0u) {
        return false;
    }

    uint32_t layer = 0u;
    uint32_t atlas_x = 0u;
    uint32_t atlas_y = 0u;
    if (!fs_alloc_image_slot(
            core,
            width,
            height,
            &layer,
            &atlas_x,
            &atlas_y
        )) {
        return false;
    }

    if (!fs_queue_write_texture_2d(
            core,
            core->image_atlas_texture,
            layer,
            atlas_x,
            atlas_y,
            width,
            height,
            rgba_pixels,
            4u
        )) {
        return false;
    }

    if (out_handle) {
        out_handle->uv_min[0] = (float)atlas_x / (float)core->image_atlas_width;
        out_handle->uv_min[1] = (float)atlas_y / (float)core->image_atlas_height;
        out_handle->uv_max[0] = (float)(atlas_x + width) / (float)core->image_atlas_width;
        out_handle->uv_max[1] = (float)(atlas_y + height) / (float)core->image_atlas_height;
        out_handle->width = width;
        out_handle->height = height;
        out_handle->layer = layer;
        out_handle->generation = core->image_atlas_generation[layer];
        out_handle->atlas_x = atlas_x;
        out_handle->atlas_y = atlas_y;
    }
    return true;
}

bool fs_core_decode_image_memory(
    FS_Core* core,
    const uint8_t* encoded_bytes,
    size_t encoded_size,
    FS_ImageHandle* out_handle
) {
    if (!core || !encoded_bytes || encoded_size == 0u) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || !st->image_backend || !st->image_backend->decode_memory || !st->image_backend->free_image) {
        return false;
    }
    uint8_t* rgba = NULL;
    uint32_t w = 0u;
    uint32_t h = 0u;
    if (!st->image_backend->decode_memory(encoded_bytes, encoded_size, &rgba, &w, &h)) {
        return false;
    }
    bool ok = fs_core_upload_image_rgba8(core, rgba, w, h, out_handle);
    st->image_backend->free_image(rgba);
    return ok;
}

bool fs_core_decode_image_file(
    FS_Core* core,
    const char* path,
    FS_ImageHandle* out_handle
) {
    if (!core || !path) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || !st->image_backend || !st->image_backend->decode_file || !st->image_backend->free_image) {
        return false;
    }
    uint8_t* rgba = NULL;
    uint32_t w = 0u;
    uint32_t h = 0u;
    if (!st->image_backend->decode_file(path, &rgba, &w, &h)) {
        return false;
    }
    bool ok = fs_core_upload_image_rgba8(core, rgba, w, h, out_handle);
    st->image_backend->free_image(rgba);
    return ok;
}

bool fs_core_put_image_data_rgba8(
    FS_Core* core,
    const FS_ImageHandle* handle,
    const uint8_t* rgba_pixels,
    size_t rgba_size
) {
    if (!core || !handle || !rgba_pixels || handle->width == 0u || handle->height == 0u) {
        return false;
    }
    const size_t needed = (size_t)handle->width * (size_t)handle->height * 4u;
    if (rgba_size < needed) {
        return false;
    }
    if (handle->layer >= core->image_atlas_layers) {
        return false;
    }
    if (handle->generation != core->image_atlas_generation[handle->layer]) {
        return false;
    }

    uint32_t atlas_x = handle->atlas_x;
    uint32_t atlas_y = handle->atlas_y;
    if (atlas_x + handle->width > core->image_atlas_width ||
        atlas_y + handle->height > core->image_atlas_height) {
        atlas_x = (uint32_t)floorf(handle->uv_min[0] * (float)core->image_atlas_width + 0.5f);
        atlas_y = (uint32_t)floorf(handle->uv_min[1] * (float)core->image_atlas_height + 0.5f);
    }

    return fs_queue_write_texture_2d(
        core,
        core->image_atlas_texture,
        handle->layer,
        atlas_x,
        atlas_y,
        handle->width,
        handle->height,
        rgba_pixels,
        4u
    );
}

bool fs_core_get_image_data_rgba8(
    const FS_Core* core,
    const FS_ImageHandle* handle,
    uint8_t* out_rgba_pixels,
    size_t out_rgba_size
) {
    if (!core || !handle || !out_rgba_pixels || handle->width == 0u || handle->height == 0u) {
        return false;
    }
    const size_t needed = (size_t)handle->width * (size_t)handle->height * 4u;
    if (out_rgba_size < needed) {
        return false;
    }
    if (handle->layer >= core->image_atlas_layers) {
        return false;
    }
    if (handle->generation != core->image_atlas_generation[handle->layer]) {
        return false;
    }

    uint32_t atlas_x = handle->atlas_x;
    uint32_t atlas_y = handle->atlas_y;
    if (atlas_x + handle->width > core->image_atlas_width ||
        atlas_y + handle->height > core->image_atlas_height) {
        atlas_x = (uint32_t)floorf(handle->uv_min[0] * (float)core->image_atlas_width + 0.5f);
        atlas_y = (uint32_t)floorf(handle->uv_min[1] * (float)core->image_atlas_height + 0.5f);
    }

    return fs_image_atlas_shadow_read_rgba(
        core,
        handle->layer,
        atlas_x,
        atlas_y,
        handle->width,
        handle->height,
        out_rgba_pixels
    );
}

bool fs_core_create_image_data_rgba8(
    uint32_t width,
    uint32_t height,
    uint8_t* out_rgba_pixels,
    size_t out_rgba_size
) {
    if (!out_rgba_pixels || width == 0u || height == 0u) {
        return false;
    }
    const size_t needed = (size_t)width * (size_t)height * 4u;
    if (needed == 0u || out_rgba_size < needed) {
        return false;
    }
    memset(out_rgba_pixels, 0, needed);
    return true;
}

bool fs_core_put_canvas_image_data_rgba8(
    FS_Core* core,
    int32_t dst_x,
    int32_t dst_y,
    uint32_t width,
    uint32_t height,
    const uint8_t* rgba_pixels,
    size_t rgba_size
) {
    if (!core || !rgba_pixels || width == 0u || height == 0u) {
        return false;
    }
    const size_t needed = (size_t)width * (size_t)height * 4u;
    if (needed == 0u || rgba_size < needed) {
        return false;
    }
    if (core->width == 0u || core->height == 0u) {
        return true;
    }
    if (!fs_ensure_canvas_shadow(core)) {
        return false;
    }

    const int64_t x0 = (int64_t)dst_x;
    const int64_t y0 = (int64_t)dst_y;
    const int64_t x1 = x0 + (int64_t)width;
    const int64_t y1 = y0 + (int64_t)height;
    const int64_t cx0 = (x0 < 0) ? 0 : x0;
    const int64_t cy0 = (y0 < 0) ? 0 : y0;
    const int64_t cx1 = (x1 > (int64_t)core->width) ? (int64_t)core->width : x1;
    const int64_t cy1 = (y1 > (int64_t)core->height) ? (int64_t)core->height : y1;
    if (cx1 <= cx0 || cy1 <= cy0) {
        return true;
    }

    const uint32_t clip_x = (uint32_t)cx0;
    const uint32_t clip_y = (uint32_t)cy0;
    const uint32_t clip_w = (uint32_t)(cx1 - cx0);
    const uint32_t clip_h = (uint32_t)(cy1 - cy0);
    const uint32_t src_x = (uint32_t)(cx0 - x0);
    const uint32_t src_y = (uint32_t)(cy0 - y0);

    const size_t shadow_row_bytes = (size_t)core->width * 4u;
    const size_t src_row_bytes = (size_t)width * 4u;
    const size_t copy_row_bytes = (size_t)clip_w * 4u;
    for (uint32_t row = 0u; row < clip_h; ++row) {
        const uint8_t* src = rgba_pixels + ((size_t)(src_y + row) * src_row_bytes) + (size_t)src_x * 4u;
        uint8_t* dst = core->canvas_shadow_rgba + ((size_t)(clip_y + row) * shadow_row_bytes) + (size_t)clip_x * 4u;
        memcpy(dst, src, copy_row_bytes);
    }

    const size_t blit_size = (size_t)clip_w * (size_t)clip_h * 4u;
    uint8_t* blit = (uint8_t*)malloc(blit_size);
    if (!blit) {
        return false;
    }
    for (uint32_t row = 0u; row < clip_h; ++row) {
        const uint8_t* src = rgba_pixels + ((size_t)(src_y + row) * src_row_bytes) + (size_t)src_x * 4u;
        uint8_t* dst = blit + (size_t)row * copy_row_bytes;
        memcpy(dst, src, copy_row_bytes);
    }

    FS_ImageHandle handle;
    bool ok = fs_ensure_canvas_image_data_handle(core, clip_w, clip_h, &handle);
    if (!ok) {
        free(blit);
        return false;
    }
    ok = fs_queue_write_texture_2d(
        core,
        core->image_atlas_texture,
        handle.layer,
        handle.atlas_x,
        handle.atlas_y,
        clip_w,
        clip_h,
        blit,
        4u
    );
    free(blit);
    if (!ok) {
        return false;
    }

    const float uv_x = handle.uv_min[0];
    const float uv_y = handle.uv_min[1];
    const float uv_w = handle.uv_max[0] - handle.uv_min[0];
    const float uv_h = handle.uv_max[1] - handle.uv_min[1];
    const float prev_alpha = fs_style_get_global_alpha(core);
    const FS_GlobalCompositeOperation prev_comp = fs_style_get_global_composite_operation(core);
    const uint32_t prev_shadow_color = fs_style_get_shadow_color(core);
    const float prev_shadow_blur = fs_style_get_shadow_blur(core);
    const float prev_shadow_off_x = fs_style_get_shadow_offset_x(core);
    const float prev_shadow_off_y = fs_style_get_shadow_offset_y(core);

    fs_state_save(core);
    fs_transform_reset(core);
    fs_clip_reset_state(fs_state(core));
    fs_style_set_global_alpha(core, 1.0f);
    fs_style_set_global_composite_operation(core, FS_GLOBAL_COMPOSITE_SOURCE_OVER);
    fs_style_set_shadow_color(core, 0u);
    fs_style_set_shadow_blur(core, 0.0f);
    fs_style_set_shadow_offset(core, 0.0f, 0.0f);
    ok = fs_cmd_image(
        core,
        (float)clip_x,
        (float)clip_y,
        (float)clip_w,
        (float)clip_h,
        uv_x,
        uv_y,
        uv_w,
        uv_h,
        0xFFFFFFFFu
    );
    (void)fs_state_restore(core);
    fs_style_set_global_alpha(core, prev_alpha);
    fs_style_set_global_composite_operation(core, prev_comp);
    fs_style_set_shadow_color(core, prev_shadow_color);
    fs_style_set_shadow_blur(core, prev_shadow_blur);
    fs_style_set_shadow_offset(core, prev_shadow_off_x, prev_shadow_off_y);
    return ok;
}

bool fs_core_get_canvas_image_data_rgba8(
    const FS_Core* core,
    int32_t src_x,
    int32_t src_y,
    uint32_t width,
    uint32_t height,
    uint8_t* out_rgba_pixels,
    size_t out_rgba_size
) {
    if (!core || !out_rgba_pixels || width == 0u || height == 0u) {
        return false;
    }
    const size_t needed = (size_t)width * (size_t)height * 4u;
    if (needed == 0u || out_rgba_size < needed) {
        return false;
    }
    memset(out_rgba_pixels, 0, needed);
    // Best-effort sync from the latest GPU-presented frame into canvas shadow.
    // If this fails, we keep the previous shadow/fallback behavior.
    (void)fs_refresh_canvas_shadow_from_readback((FS_Core*)core);
    if (!core->canvas_shadow_rgba || core->canvas_shadow_size == 0u || core->width == 0u || core->height == 0u) {
        return true;
    }

    const int64_t x0 = (int64_t)src_x;
    const int64_t y0 = (int64_t)src_y;
    const int64_t x1 = x0 + (int64_t)width;
    const int64_t y1 = y0 + (int64_t)height;
    const int64_t cx0 = (x0 < 0) ? 0 : x0;
    const int64_t cy0 = (y0 < 0) ? 0 : y0;
    const int64_t cx1 = (x1 > (int64_t)core->width) ? (int64_t)core->width : x1;
    const int64_t cy1 = (y1 > (int64_t)core->height) ? (int64_t)core->height : y1;
    if (cx1 <= cx0 || cy1 <= cy0) {
        return true;
    }

    const uint32_t clip_x = (uint32_t)cx0;
    const uint32_t clip_y = (uint32_t)cy0;
    const uint32_t clip_w = (uint32_t)(cx1 - cx0);
    const uint32_t clip_h = (uint32_t)(cy1 - cy0);
    const uint32_t dst_x = (uint32_t)(cx0 - x0);
    const uint32_t dst_y = (uint32_t)(cy0 - y0);

    const size_t shadow_row_bytes = (size_t)core->width * 4u;
    const size_t out_row_bytes = (size_t)width * 4u;
    const size_t copy_row_bytes = (size_t)clip_w * 4u;
    for (uint32_t row = 0u; row < clip_h; ++row) {
        const uint8_t* src = core->canvas_shadow_rgba + ((size_t)(clip_y + row) * shadow_row_bytes) + (size_t)clip_x * 4u;
        uint8_t* dst = out_rgba_pixels + ((size_t)(dst_y + row) * out_row_bytes) + (size_t)dst_x * 4u;
        memcpy(dst, src, copy_row_bytes);
    }
    return true;
}

bool fs_core_load_font_file(FS_Core* core, const char* path) {
    if (!core || !path) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }

    if (!st->font_backend || !st->font_backend->load_font_file) {
        return false;
    }
    if (st->font_count >= FS_MAX_FONT_FALLBACKS) {
        return false;
    }
    void* loaded = st->font_backend->load_font_file(path);
    if (!loaded) {
        return false;
    }
    st->fonts[st->font_count++] = loaded;
    return true;
}

bool fs_core_register_image_font(
    FS_Core* core,
    const FS_ImageFontSequence* sequences,
    uint32_t sequence_count,
    uint32_t* out_font_id
) {
    if (!core || !sequences || sequence_count == 0u || !out_font_id) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }

    if (!fs_ensure_image_font_capacity(st, st->image_font_count + 1u)) {
        return false;
    }

    FS_ImageSequenceEntry* entries =
        (FS_ImageSequenceEntry*)calloc((size_t)sequence_count, sizeof(FS_ImageSequenceEntry));
    if (!entries) {
        return false;
    }

    for (uint32_t i = 0u; i < sequence_count; ++i) {
        const FS_ImageFontSequence* src = &sequences[i];
        if (!src->utf8 || src->utf8[0] == '\0') {
            for (uint32_t j = 0u; j < i; ++j) {
                free(entries[j].utf8);
            }
            free(entries);
            return false;
        }
        entries[i].glyph_id = src->glyph_id;
        entries[i].utf8 = fs_strdup_owned(src->utf8);
        if (!entries[i].utf8) {
            for (uint32_t j = 0u; j < i; ++j) {
                free(entries[j].utf8);
            }
            free(entries);
            return false;
        }
        uint8_t cp_count = 0u;
        if (!fs_canonicalize_utf8_sequence(entries[i].utf8, entries[i].cps, &cp_count) || cp_count == 0u) {
            for (uint32_t j = 0u; j <= i; ++j) {
                free(entries[j].utf8);
            }
            free(entries);
            return false;
        }
        entries[i].cp_count = cp_count;
    }

    qsort(entries, sequence_count, sizeof(FS_ImageSequenceEntry), fs_sequence_entry_sort_desc);

    uint32_t next_font_id = 1u;
    for (uint32_t i = 0u; i < st->image_font_count; ++i) {
        if (st->image_fonts[i].id >= next_font_id) {
            next_font_id = st->image_fonts[i].id + 1u;
        }
    }
    if (next_font_id == 0u) {
        for (uint32_t i = 0u; i < sequence_count; ++i) {
            free(entries[i].utf8);
        }
        free(entries);
        return false;
    }

    FS_ImageFontState* font = &st->image_fonts[st->image_font_count++];
    memset(font, 0, sizeof(*font));
    font->id = next_font_id;
    font->sequences = entries;
    font->sequence_count = sequence_count;
    font->glyph_slots = NULL;
    font->glyph_slot_count = 0u;

    *out_font_id = next_font_id;
    return true;
}

bool fs_core_load_image_glyph_rgba8(
    FS_Core* core,
    uint32_t image_font_id,
    uint32_t glyph_id,
    const uint8_t* rgba_pixels,
    uint32_t width,
    uint32_t height
) {
    if (!core || image_font_id == 0u || !rgba_pixels || width == 0u || height == 0u) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    FS_ImageFontState* font = fs_find_image_font(st, image_font_id);
    if (!font) {
        return false;
    }
    if (!fs_image_font_ensure_glyph_slots(font, glyph_id + 1u)) {
        return false;
    }

    const size_t pixel_bytes = (size_t)width * (size_t)height * 4u;
    uint8_t* copied = (uint8_t*)malloc(pixel_bytes);
    if (!copied) {
        return false;
    }
    memcpy(copied, rgba_pixels, pixel_bytes);

    FS_ImageGlyphSlot* slot = &font->glyph_slots[glyph_id];
    free(slot->rgba);
    slot->rgba = copied;
    slot->width = width;
    slot->height = height;
    slot->loaded = true;

    uint32_t packed_key = 0u;
    if (fs_pack_image_glyph_key(image_font_id, glyph_id, &packed_key)) {
        fs_invalidate_cached_glyph(st, FS_IMAGE_FONT_KIND, packed_key);
    }

    if (st->missing_count > 0u) {
        uint32_t write = 0u;
        for (uint32_t read = 0u; read < st->missing_count; ++read) {
            FS_MissingImageGlyph m = st->missing_image_glyphs[read];
            if (m.image_font_id == image_font_id && m.glyph_id == glyph_id) {
                continue;
            }
            if (write != read) {
                st->missing_image_glyphs[write] = m;
            }
            write += 1u;
        }
        st->missing_count = write;
    }
    return true;
}

bool fs_core_load_image_glyph_png_memory(
    FS_Core* core,
    uint32_t image_font_id,
    uint32_t glyph_id,
    const uint8_t* encoded_bytes,
    size_t encoded_size
) {
    if (!core || image_font_id == 0u || !encoded_bytes || encoded_size == 0u) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || !st->image_backend || !st->image_backend->decode_memory || !st->image_backend->free_image) {
        return false;
    }
    uint8_t* rgba = NULL;
    uint32_t w = 0u;
    uint32_t h = 0u;
    if (!st->image_backend->decode_memory(encoded_bytes, encoded_size, &rgba, &w, &h)) {
        return false;
    }
    const bool ok = fs_core_load_image_glyph_rgba8(core, image_font_id, glyph_id, rgba, w, h);
    st->image_backend->free_image(rgba);
    return ok;
}

uint32_t fs_core_get_missing_image_glyph_count(const FS_Core* core) {
    if (!core || !core->internal_state) {
        return 0u;
    }
    const FS_InternalState* st = (const FS_InternalState*)core->internal_state;
    return st->missing_count;
}

bool fs_core_get_missing_image_glyph(
    const FS_Core* core,
    uint32_t index,
    FS_MissingImageGlyph* out_missing
) {
    if (!core || !core->internal_state || !out_missing) {
        return false;
    }
    const FS_InternalState* st = (const FS_InternalState*)core->internal_state;
    if (index >= st->missing_count) {
        return false;
    }
    *out_missing = st->missing_image_glyphs[index];
    return true;
}

void fs_core_clear_missing_image_glyphs(FS_Core* core) {
    if (!core || !core->internal_state) {
        return;
    }
    FS_InternalState* st = (FS_InternalState*)core->internal_state;
    st->missing_count = 0u;
}

bool fs_core_set_image_backend(FS_Core* core, const FS_ImageBackend* backend) {
    if (!core || !backend || !backend->decode_memory || !backend->decode_file || !backend->free_image) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->image_backend = backend;
    return true;
}

bool fs_core_set_font_backend(FS_Core* core, const FS_FontBackend* backend) {
    if (!core || !backend || !backend->load_font_file || !backend->destroy_font ||
        !backend->get_glyph_sdf || !backend->free_glyph_pixels) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    fs_clear_loaded_fonts(st);
    st->glyph_count = 0u;
    fs_glyph_cache_clear_index(st);
    st->font_backend = backend;
    core->glyph_atlas_cursor_x = 0u;
    core->glyph_atlas_cursor_y = 0u;
    core->glyph_atlas_row_height = 0u;
    return true;
}

void fs_core_set_clip_aa_mode(FS_Core* core, int32_t mode) {
    if (!core) {
        return;
    }
    if (mode < -1) {
        mode = -1;
    } else if (mode > 3) {
        mode = 3;
    }
    core->clip_aa_mode_override = mode;
}

void fs_core_set_clip_layer_reuse_reserve(FS_Core* core, uint32_t reserve_layers) {
    if (!core) {
        return;
    }
    if (core->clip_mask_layers > 0u && reserve_layers >= core->clip_mask_layers) {
        reserve_layers = core->clip_mask_layers - 1u;
    }
    core->clip_layer_reuse_reserve = reserve_layers;
}

uint32_t fs_core_get_clip_layer_reuse_reserve(const FS_Core* core) {
    if (!core) {
        return 0u;
    }
    return core->clip_layer_reuse_reserve;
}

void fs_core_set_clip_cache_enabled(FS_Core* core, bool enabled) {
    if (!core) {
        return;
    }
    core->clip_cache_enabled = enabled;
    if (!enabled && core->clip_mask_layer_hash_valid && core->clip_mask_layers > 0u) {
        memset(core->clip_mask_layer_hash_valid, 0, (size_t)core->clip_mask_layers * sizeof(uint8_t));
    }
}

bool fs_core_get_clip_cache_enabled(const FS_Core* core) {
    if (!core) {
        return false;
    }
    return core->clip_cache_enabled;
}

const char* fs_core_get_image_backend_name(const FS_Core* core) {
    if (!core || !core->internal_state) {
        return "none";
    }
    const FS_InternalState* st = (const FS_InternalState*)core->internal_state;
    if (!st->image_backend || !st->image_backend->name) {
        return "none";
    }
    return st->image_backend->name;
}

const char* fs_core_get_font_backend_name(const FS_Core* core) {
    if (!core || !core->internal_state) {
        return "none";
    }
    const FS_InternalState* st = (const FS_InternalState*)core->internal_state;
    if (!st->font_backend || !st->font_backend->name) {
        return "none";
    }
    return st->font_backend->name;
}

FS_LinearGradient* fs_linear_gradient_create(float x0, float y0, float x1, float y1) {
    if (!isfinite(x0) || !isfinite(y0) || !isfinite(x1) || !isfinite(y1)) {
        return NULL;
    }
    FS_LinearGradient* gradient = (FS_LinearGradient*)calloc(1u, sizeof(FS_LinearGradient));
    if (!gradient) {
        return NULL;
    }
    gradient->x0 = x0;
    gradient->y0 = y0;
    gradient->x1 = x1;
    gradient->y1 = y1;
    return gradient;
}

void fs_linear_gradient_destroy(FS_LinearGradient* gradient) {
    if (!gradient) {
        return;
    }
    free(gradient->stops);
    gradient->stops = NULL;
    gradient->stop_count = 0u;
    gradient->stop_capacity = 0u;
    free(gradient);
}

bool fs_linear_gradient_add_color_stop(FS_LinearGradient* gradient, float offset_0_to_1, uint32_t color_rgba8) {
    if (!gradient || !isfinite(offset_0_to_1) || offset_0_to_1 < 0.0f || offset_0_to_1 > 1.0f) {
        return false;
    }
    if (gradient->stop_count + 1u > gradient->stop_capacity) {
        uint32_t new_capacity = gradient->stop_capacity ? gradient->stop_capacity * 2u : 8u;
        if (new_capacity < gradient->stop_count + 1u) {
            new_capacity = gradient->stop_count + 1u;
        }
        FS_GradientStop* grown =
            (FS_GradientStop*)realloc(gradient->stops, (size_t)new_capacity * sizeof(FS_GradientStop));
        if (!grown) {
            return false;
        }
        gradient->stops = grown;
        gradient->stop_capacity = new_capacity;
    }
    uint32_t insert_at = gradient->stop_count;
    while (insert_at > 0u && gradient->stops[insert_at - 1u].offset_0_to_1 > offset_0_to_1) {
        insert_at -= 1u;
    }
    if (insert_at < gradient->stop_count) {
        memmove(
            &gradient->stops[insert_at + 1u],
            &gradient->stops[insert_at],
            (size_t)(gradient->stop_count - insert_at) * sizeof(FS_GradientStop)
        );
    }
    gradient->stops[insert_at].offset_0_to_1 = offset_0_to_1;
    gradient->stops[insert_at].color_rgba8 = color_rgba8;
    gradient->stop_count += 1u;
    return true;
}

FS_Pattern* fs_pattern_create_image(const FS_ImageHandle* handle, FS_PatternRepeat repeat_mode) {
    if (!handle || handle->width == 0u || handle->height == 0u) {
        return NULL;
    }
    if (repeat_mode < FS_PATTERN_REPEAT || repeat_mode > FS_PATTERN_NO_REPEAT) {
        return NULL;
    }
    FS_Pattern* pattern = (FS_Pattern*)calloc(1u, sizeof(FS_Pattern));
    if (!pattern) {
        return NULL;
    }
    pattern->handle = *handle;
    pattern->repeat_mode = (uint8_t)repeat_mode;
    fs_affine_set_identity_2d(pattern->xform);
    fs_affine_set_identity_2d(pattern->inv_xform);
    pattern->inv_valid = 1u;
    return pattern;
}

void fs_pattern_destroy(FS_Pattern* pattern) {
    free(pattern);
}

bool fs_pattern_set_transform(FS_Pattern* pattern, float a, float b, float c, float d, float e, float f) {
    if (!pattern || !isfinite(a) || !isfinite(b) || !isfinite(c) || !isfinite(d) || !isfinite(e) || !isfinite(f)) {
        return false;
    }
    pattern->xform[0] = a;
    pattern->xform[1] = b;
    pattern->xform[2] = c;
    pattern->xform[3] = d;
    pattern->xform[4] = e;
    pattern->xform[5] = f;
    pattern->inv_valid = fs_affine_try_invert_2d(pattern->xform, pattern->inv_xform) ? 1u : 0u;
    return pattern->inv_valid != 0u;
}

FS_RadialGradient* fs_radial_gradient_create(float x0, float y0, float r0, float x1, float y1, float r1) {
    if (!isfinite(x0) || !isfinite(y0) || !isfinite(r0) || r0 < 0.0f ||
        !isfinite(x1) || !isfinite(y1) || !isfinite(r1) || r1 < 0.0f) {
        return NULL;
    }
    FS_RadialGradient* gradient = (FS_RadialGradient*)calloc(1u, sizeof(FS_RadialGradient));
    if (!gradient) {
        return NULL;
    }
    gradient->x0 = x0;
    gradient->y0 = y0;
    gradient->r0 = r0;
    gradient->x1 = x1;
    gradient->y1 = y1;
    gradient->r1 = r1;
    return gradient;
}

void fs_radial_gradient_destroy(FS_RadialGradient* gradient) {
    if (!gradient) {
        return;
    }
    free(gradient->stops);
    gradient->stops = NULL;
    gradient->stop_count = 0u;
    gradient->stop_capacity = 0u;
    free(gradient);
}

bool fs_radial_gradient_add_color_stop(FS_RadialGradient* gradient, float offset_0_to_1, uint32_t color_rgba8) {
    if (!gradient || !isfinite(offset_0_to_1) || offset_0_to_1 < 0.0f || offset_0_to_1 > 1.0f) {
        return false;
    }
    if (gradient->stop_count + 1u > gradient->stop_capacity) {
        uint32_t new_capacity = gradient->stop_capacity ? gradient->stop_capacity * 2u : 8u;
        if (new_capacity < gradient->stop_count + 1u) {
            new_capacity = gradient->stop_count + 1u;
        }
        FS_GradientStop* grown =
            (FS_GradientStop*)realloc(gradient->stops, (size_t)new_capacity * sizeof(FS_GradientStop));
        if (!grown) {
            return false;
        }
        gradient->stops = grown;
        gradient->stop_capacity = new_capacity;
    }
    uint32_t insert_at = gradient->stop_count;
    while (insert_at > 0u && gradient->stops[insert_at - 1u].offset_0_to_1 > offset_0_to_1) {
        insert_at -= 1u;
    }
    if (insert_at < gradient->stop_count) {
        memmove(
            &gradient->stops[insert_at + 1u],
            &gradient->stops[insert_at],
            (size_t)(gradient->stop_count - insert_at) * sizeof(FS_GradientStop)
        );
    }
    gradient->stops[insert_at].offset_0_to_1 = offset_0_to_1;
    gradient->stops[insert_at].color_rgba8 = color_rgba8;
    gradient->stop_count += 1u;
    return true;
}

FS_ConicGradient* fs_conic_gradient_create(float start_angle_radians, float cx, float cy) {
    if (!isfinite(start_angle_radians) || !isfinite(cx) || !isfinite(cy)) {
        return NULL;
    }
    FS_ConicGradient* gradient = (FS_ConicGradient*)calloc(1u, sizeof(FS_ConicGradient));
    if (!gradient) {
        return NULL;
    }
    gradient->start_angle_radians = start_angle_radians;
    gradient->cx = cx;
    gradient->cy = cy;
    return gradient;
}

void fs_conic_gradient_destroy(FS_ConicGradient* gradient) {
    if (!gradient) {
        return;
    }
    free(gradient->stops);
    gradient->stops = NULL;
    gradient->stop_count = 0u;
    gradient->stop_capacity = 0u;
    free(gradient);
}

bool fs_conic_gradient_add_color_stop(FS_ConicGradient* gradient, float offset_0_to_1, uint32_t color_rgba8) {
    if (!gradient || !isfinite(offset_0_to_1) || offset_0_to_1 < 0.0f || offset_0_to_1 > 1.0f) {
        return false;
    }
    if (gradient->stop_count + 1u > gradient->stop_capacity) {
        uint32_t new_capacity = gradient->stop_capacity ? gradient->stop_capacity * 2u : 8u;
        if (new_capacity < gradient->stop_count + 1u) {
            new_capacity = gradient->stop_count + 1u;
        }
        FS_GradientStop* grown =
            (FS_GradientStop*)realloc(gradient->stops, (size_t)new_capacity * sizeof(FS_GradientStop));
        if (!grown) {
            return false;
        }
        gradient->stops = grown;
        gradient->stop_capacity = new_capacity;
    }
    uint32_t insert_at = gradient->stop_count;
    while (insert_at > 0u && gradient->stops[insert_at - 1u].offset_0_to_1 > offset_0_to_1) {
        insert_at -= 1u;
    }
    if (insert_at < gradient->stop_count) {
        memmove(
            &gradient->stops[insert_at + 1u],
            &gradient->stops[insert_at],
            (size_t)(gradient->stop_count - insert_at) * sizeof(FS_GradientStop)
        );
    }
    gradient->stops[insert_at].offset_0_to_1 = offset_0_to_1;
    gradient->stops[insert_at].color_rgba8 = color_rgba8;
    gradient->stop_count += 1u;
    return true;
}

void fs_state_save(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return;
    }
    if (!fs_ensure_state_stack_capacity(st, st->state_stack_count + 1u)) {
        return;
    }
    FS_StateSnapshot snap;
    memset(&snap, 0, sizeof(snap));
    snap.transform = st->current_transform;
    snap.clip_enabled = st->clip_enabled;
    snap.clip_path_enabled = st->clip_path_enabled;
    snap.clip_path_layer = st->clip_path_layer;
    snap.clip_min_x = st->clip_min_x;
    snap.clip_min_y = st->clip_min_y;
    snap.clip_max_x = st->clip_max_x;
    snap.clip_max_y = st->clip_max_y;
    if (!fs_style_snapshot_capture(&snap.style, st)) {
        fs_state_snapshot_dispose(&snap);
        return;
    }
    st->state_stack[st->state_stack_count++] = snap;
}

bool fs_state_restore(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    if (!st || st->state_stack_count == 0u) {
        return false;
    }
    st->state_stack_count -= 1u;
    FS_StateSnapshot* snap = &st->state_stack[st->state_stack_count];
    st->current_transform = snap->transform;
    st->clip_enabled = snap->clip_enabled;
    st->clip_path_enabled = snap->clip_path_enabled;
    st->clip_path_layer = snap->clip_path_layer;
    st->clip_min_x = snap->clip_min_x;
    st->clip_min_y = snap->clip_min_y;
    st->clip_max_x = snap->clip_max_x;
    st->clip_max_y = snap->clip_max_y;
    fs_style_snapshot_apply(st, &snap->style);
    return true;
}

void fs_transform_reset(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return;
    }
    st->current_transform = fs_transform_identity_value();
}

bool fs_set_transform(FS_Core* core, float a, float b, float c, float d, float e, float f) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->current_transform.a = a;
    st->current_transform.b = b;
    st->current_transform.c = c;
    st->current_transform.d = d;
    st->current_transform.e = e;
    st->current_transform.f = f;
    return true;
}

bool fs_get_transform(const FS_Core* core, float out_matrix_2x3[6]) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !out_matrix_2x3) {
        return false;
    }
    out_matrix_2x3[0] = st->current_transform.a;
    out_matrix_2x3[1] = st->current_transform.b;
    out_matrix_2x3[2] = st->current_transform.c;
    out_matrix_2x3[3] = st->current_transform.d;
    out_matrix_2x3[4] = st->current_transform.e;
    out_matrix_2x3[5] = st->current_transform.f;
    return true;
}

bool fs_transform(FS_Core* core, float a, float b, float c, float d, float e, float f) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    FS_Transform2D rhs = {.a = a, .b = b, .c = c, .d = d, .e = e, .f = f};
    st->current_transform = fs_transform_mul(&st->current_transform, &rhs);
    return true;
}

bool fs_translate(FS_Core* core, float tx, float ty) {
    return fs_transform(core, 1.0f, 0.0f, 0.0f, 1.0f, tx, ty);
}

bool fs_rotate(FS_Core* core, float radians) {
    const float s = sinf(radians);
    const float c = cosf(radians);
    return fs_transform(core, c, s, -s, c, 0.0f, 0.0f);
}

bool fs_scale(FS_Core* core, float sx, float sy) {
    return fs_transform(core, sx, 0.0f, 0.0f, sy, 0.0f, 0.0f);
}

void fs_clip_reset(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    fs_clip_reset_state(st);
}

static bool fs_clip_intersect_aabb(FS_InternalState* st, float min_x, float min_y, float max_x, float max_y) {
    if (!st) {
        return false;
    }
    if (max_x <= min_x || max_y <= min_y) {
        // Represent an empty clip region.
        st->clip_enabled = 1u;
        st->clip_min_x = 1.0f;
        st->clip_min_y = 1.0f;
        st->clip_max_x = 0.0f;
        st->clip_max_y = 0.0f;
        return true;
    }
    if (!st->clip_enabled) {
        st->clip_enabled = 1u;
        st->clip_min_x = min_x;
        st->clip_min_y = min_y;
        st->clip_max_x = max_x;
        st->clip_max_y = max_y;
        return true;
    }
    if (min_x > st->clip_min_x) {
        st->clip_min_x = min_x;
    }
    if (min_y > st->clip_min_y) {
        st->clip_min_y = min_y;
    }
    if (max_x < st->clip_max_x) {
        st->clip_max_x = max_x;
    }
    if (max_y < st->clip_max_y) {
        st->clip_max_y = max_y;
    }
    return true;
}

bool fs_clip_rect(FS_Core* core, float x, float y, float w, float h) {
    FS_InternalState* st = fs_state(core);
    if (!st || w <= 0.0f || h <= 0.0f) {
        return false;
    }
    const FS_Transform2D* t = &st->current_transform;
    float tx = x;
    float ty = y;
    float tw = w;
    float th = h;
    fs_transform_rect_to_aabb(t, x, y, w, h, &tx, &ty, &tw, &th);
    (void)tw;
    (void)th;
    const float min_x = tx;
    const float min_y = ty;
    const float max_x = tx + tw;
    const float max_y = ty + th;
    return fs_clip_intersect_aabb(st, min_x, min_y, max_x, max_y);
}

static void fs_clip_mark_layer_chain(uint8_t* protected_layers, uint32_t layer_count, const uint32_t* parents, uint32_t layer) {
    if (!protected_layers || layer_count == 0u) {
        return;
    }
    uint32_t cur = layer;
    uint32_t guard = 0u;
    while (cur < layer_count) {
        protected_layers[cur] = 1u;
        if (!parents) {
            break;
        }
        const uint32_t parent = parents[cur];
        if (parent == UINT32_MAX || parent >= layer_count) {
            break;
        }
        cur = parent;
        guard += 1u;
        if (guard >= layer_count) {
            break;
        }
    }
}

static bool fs_clip_try_acquire_layer_with_policy(FS_Core* core, const FS_InternalState* st, uint32_t* out_layer) {
    if (!core || !out_layer || core->clip_mask_layers == 0u) {
        return false;
    }
    const uint32_t layer_count = core->clip_mask_layers;
    const bool can_allocate_fresh = core->clip_mask_next_layer < layer_count;
    bool prefer_reuse = false;
    if (can_allocate_fresh && core->clip_layer_reuse_reserve > 0u) {
        const uint32_t remaining = layer_count - core->clip_mask_next_layer;
        prefer_reuse = (remaining <= core->clip_layer_reuse_reserve);
    }
    if (can_allocate_fresh && !prefer_reuse) {
        *out_layer = core->clip_mask_next_layer++;
        return true;
    }

    uint8_t* protected_layers = (uint8_t*)calloc((size_t)layer_count, sizeof(uint8_t));
    if (!protected_layers) {
        return false;
    }

    for (size_t i = 0u; i < core->command_count; ++i) {
        const FS_Command* cmd = &core->commands[i];
        if ((cmd->flags & FS_RENDER_FLAG_CLIP_MASK) == 0u) {
            continue;
        }
        uint32_t layer = (cmd->flags & FS_RENDER_FLAG_CLIP_LAYER_MASK) >> FS_RENDER_FLAG_CLIP_LAYER_SHIFT;
        if (layer < layer_count) {
            fs_clip_mark_layer_chain(protected_layers, layer_count, core->clip_mask_layer_parent, layer);
        }
        const uint32_t parent_bits = (cmd->flags & FS_RENDER_FLAG_CLIP_PARENT_MASK) >> FS_RENDER_FLAG_CLIP_PARENT_SHIFT;
        if (parent_bits != 0u) {
            const uint32_t parent = parent_bits - 1u;
            if (parent < layer_count) {
                fs_clip_mark_layer_chain(protected_layers, layer_count, core->clip_mask_layer_parent, parent);
            }
        }
    }

    if (st) {
        if (st->clip_path_enabled && st->clip_path_layer < layer_count) {
            fs_clip_mark_layer_chain(protected_layers, layer_count, core->clip_mask_layer_parent, st->clip_path_layer);
        }
        for (uint32_t i = 0u; i < st->state_stack_count; ++i) {
            const FS_StateSnapshot* snap = &st->state_stack[i];
            if (snap->clip_path_enabled && snap->clip_path_layer < layer_count) {
                fs_clip_mark_layer_chain(protected_layers, layer_count, core->clip_mask_layer_parent, snap->clip_path_layer);
            }
        }
    }

    const uint32_t victim_search_end = can_allocate_fresh ? core->clip_mask_next_layer : layer_count;
    int32_t victim = -1;
    uint32_t oldest_stamp = UINT32_MAX;
    for (uint32_t i = 0u; i < victim_search_end; ++i) {
        if (protected_layers[i] != 0u) {
            continue;
        }
        if (core->clip_mask_layer_has_data && core->clip_mask_layer_has_data[i] == 0u) {
            victim = (int32_t)i;
            oldest_stamp = 0u;
            break;
        }
        const uint32_t stamp =
            core->clip_mask_layer_last_used_frame ? core->clip_mask_layer_last_used_frame[i] : 0u;
        if (victim < 0 || stamp < oldest_stamp) {
            victim = (int32_t)i;
            oldest_stamp = stamp;
        }
    }
    free(protected_layers);
    if (victim < 0) {
        if (can_allocate_fresh) {
            *out_layer = core->clip_mask_next_layer++;
            return true;
        }
        return false;
    }

    const uint32_t layer = (uint32_t)victim;
    if (core->clip_mask_layer_hash_valid) {
        core->clip_mask_layer_hash_valid[layer] = 0u;
    }
    if (core->clip_mask_layer_hash) {
        core->clip_mask_layer_hash[layer] = 0u;
    }
    if (core->clip_mask_layer_parent) {
        core->clip_mask_layer_parent[layer] = UINT32_MAX;
    }
    core->clip_layer_reuses_this_frame += 1u;
    *out_layer = layer;
    return true;
}

typedef struct FS_ClipEdge {
    float x0;
    float y0;
    float x1;
    float y1;
} FS_ClipEdge;

static bool fs_clip_edges_reserve(FS_ClipEdge** io_edges, uint32_t* io_capacity, uint32_t required) {
    if (!io_edges || !io_capacity) {
        return false;
    }
    if (required <= *io_capacity) {
        return true;
    }
    uint32_t new_cap = (*io_capacity > 0u) ? *io_capacity : 128u;
    while (new_cap < required) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_ClipEdge* grown = (FS_ClipEdge*)realloc(*io_edges, (size_t)new_cap * sizeof(FS_ClipEdge));
    if (!grown) {
        return false;
    }
    *io_edges = grown;
    *io_capacity = new_cap;
    return true;
}

static bool fs_clip_edges_push(
    FS_ClipEdge** io_edges,
    uint32_t* io_count,
    uint32_t* io_capacity,
    float x0,
    float y0,
    float x1,
    float y1,
    float* io_min_x,
    float* io_min_y,
    float* io_max_x,
    float* io_max_y,
    bool* io_has_bounds
) {
    if (!io_edges || !io_count || !io_capacity || !io_min_x || !io_min_y || !io_max_x || !io_max_y || !io_has_bounds) {
        return false;
    }
    const float dx = x1 - x0;
    const float dy = y1 - y0;
    if (fabsf(dx) <= 1e-6f && fabsf(dy) <= 1e-6f) {
        return true;
    }
    const uint32_t required = *io_count + 1u;
    if (!fs_clip_edges_reserve(io_edges, io_capacity, required)) {
        return false;
    }
    FS_ClipEdge* edges = *io_edges;
    edges[*io_count].x0 = x0;
    edges[*io_count].y0 = y0;
    edges[*io_count].x1 = x1;
    edges[*io_count].y1 = y1;
    *io_count = required;

    if (!*io_has_bounds) {
        *io_min_x = fminf(x0, x1);
        *io_min_y = fminf(y0, y1);
        *io_max_x = fmaxf(x0, x1);
        *io_max_y = fmaxf(y0, y1);
        *io_has_bounds = true;
    } else {
        if (x0 < *io_min_x) *io_min_x = x0;
        if (x1 < *io_min_x) *io_min_x = x1;
        if (y0 < *io_min_y) *io_min_y = y0;
        if (y1 < *io_min_y) *io_min_y = y1;
        if (x0 > *io_max_x) *io_max_x = x0;
        if (x1 > *io_max_x) *io_max_x = x1;
        if (y0 > *io_max_y) *io_max_y = y0;
        if (y1 > *io_max_y) *io_max_y = y1;
    }
    return true;
}

static void fs_clip_diag_reset_frame(FS_Core* core) {
    if (!core) {
        return;
    }
    core->clip_requests_this_frame = 0u;
    core->clip_cache_hits_this_frame = 0u;
    core->clip_jobs_enqueued_this_frame = 0u;
    core->clip_layer_reuses_this_frame = 0u;
    core->clip_failures_this_frame = 0u;
    core->clip_layers_used_this_frame = 0u;
    core->clip_last_failure_reason = (uint32_t)FS_CLIP_FAILURE_NONE;
    core->clip_last_failure_path_segments = 0u;
    core->clip_last_failure_edge_count = 0u;
    core->clip_dispatch_batches_this_frame = 0u;
    core->clip_dispatch_valid_jobs_this_frame = 0u;
    core->clip_dispatch_pixels_ideal_this_frame = 0u;
    core->clip_dispatch_pixels_estimated_this_frame = 0u;
    core->clip_dispatch_pixels_waste_this_frame = 0u;
    memset(
        core->clip_dispatch_bucket_jobs_this_frame,
        0,
        sizeof(core->clip_dispatch_bucket_jobs_this_frame)
    );
    core->clip_oriented_quad_commands_this_frame = 0u;
    core->clip_oriented_quad_clipped_this_frame = 0u;
}

static void fs_clip_diag_note_layer_usage(FS_Core* core, uint32_t layer) {
    if (!core) {
        return;
    }
    const uint32_t used = layer + 1u;
    if (used > core->clip_layers_used_this_frame) {
        core->clip_layers_used_this_frame = used;
    }
    if (core->clip_mask_layer_last_used_frame && layer < core->clip_mask_layers) {
        core->clip_mask_layer_last_used_frame[layer] = core->clip_frame_index;
    }
}

static void fs_clip_diag_note_failure(
    FS_Core* core,
    FS_ClipFailureReason reason,
    uint32_t path_count,
    uint32_t edge_count
) {
    if (!core) {
        return;
    }
    core->clip_failures_this_frame += 1u;
    core->clip_last_failure_reason = (uint32_t)reason;
    core->clip_last_failure_path_segments = path_count;
    core->clip_last_failure_edge_count = edge_count;
}

static bool fs_clip_point_matches_rect_corner(
    float x,
    float y,
    float min_x,
    float min_y,
    float max_x,
    float max_y,
    float eps
) {
    const bool c0 = fabsf(x - min_x) <= eps && fabsf(y - min_y) <= eps;
    const bool c1 = fabsf(x - max_x) <= eps && fabsf(y - min_y) <= eps;
    const bool c2 = fabsf(x - max_x) <= eps && fabsf(y - max_y) <= eps;
    const bool c3 = fabsf(x - min_x) <= eps && fabsf(y - max_y) <= eps;
    return c0 || c1 || c2 || c3;
}

static bool fs_clip_try_extract_axis_aligned_rect_aabb(
    const FS_InternalState* st,
    float* out_min_x,
    float* out_min_y,
    float* out_max_x,
    float* out_max_y
) {
    if (!st || !out_min_x || !out_min_y || !out_max_x || !out_max_y) {
        return false;
    }
    if (st->path_count != 4u || !st->path_segments) {
        return false;
    }

    const float eps_local = 1e-4f;
    const float eps_dev = 1e-3f;
    FS_Point2 p[5];

    for (uint32_t i = 0u; i < 4u; ++i) {
        const FS_PathSegment* seg = &st->path_segments[i];
        if (seg->type != (uint8_t)FS_PATH_SEG_LINE) {
            return false;
        }
        if (i == 0u) {
            p[0].x = seg->x0;
            p[0].y = seg->y0;
        } else {
            if (fabsf(seg->x0 - p[i].x) > eps_local || fabsf(seg->y0 - p[i].y) > eps_local) {
                return false;
            }
        }
        p[i + 1u].x = seg->x1;
        p[i + 1u].y = seg->y1;

        const float dx = p[i + 1u].x - p[i].x;
        const float dy = p[i + 1u].y - p[i].y;
        if (fabsf(dx) <= eps_local && fabsf(dy) <= eps_local) {
            return false;
        }
        if (fabsf(dx) > eps_local && fabsf(dy) > eps_local) {
            return false;
        }
    }

    if (fabsf(p[4].x - p[0].x) > eps_local || fabsf(p[4].y - p[0].y) > eps_local) {
        return false;
    }

    float local_min_x = p[0].x;
    float local_min_y = p[0].y;
    float local_max_x = p[0].x;
    float local_max_y = p[0].y;
    for (uint32_t i = 1u; i < 4u; ++i) {
        if (p[i].x < local_min_x) local_min_x = p[i].x;
        if (p[i].y < local_min_y) local_min_y = p[i].y;
        if (p[i].x > local_max_x) local_max_x = p[i].x;
        if (p[i].y > local_max_y) local_max_y = p[i].y;
    }
    if (local_max_x - local_min_x <= eps_local || local_max_y - local_min_y <= eps_local) {
        return false;
    }
    for (uint32_t i = 0u; i < 4u; ++i) {
        if (!fs_clip_point_matches_rect_corner(
                p[i].x, p[i].y, local_min_x, local_min_y, local_max_x, local_max_y, eps_local
            )) {
            return false;
        }
    }

    FS_Point2 tp[4];
    for (uint32_t i = 0u; i < 4u; ++i) {
        fs_transform_apply_point(&st->current_transform, p[i].x, p[i].y, &tp[i].x, &tp[i].y);
    }
    for (uint32_t i = 0u; i < 4u; ++i) {
        const uint32_t j = (i + 1u) & 3u;
        const float dx = tp[j].x - tp[i].x;
        const float dy = tp[j].y - tp[i].y;
        if (fabsf(dx) <= eps_dev && fabsf(dy) <= eps_dev) {
            return false;
        }
        if (fabsf(dx) > eps_dev && fabsf(dy) > eps_dev) {
            return false;
        }
    }

    float dev_min_x = tp[0].x;
    float dev_min_y = tp[0].y;
    float dev_max_x = tp[0].x;
    float dev_max_y = tp[0].y;
    for (uint32_t i = 1u; i < 4u; ++i) {
        if (tp[i].x < dev_min_x) dev_min_x = tp[i].x;
        if (tp[i].y < dev_min_y) dev_min_y = tp[i].y;
        if (tp[i].x > dev_max_x) dev_max_x = tp[i].x;
        if (tp[i].y > dev_max_y) dev_max_y = tp[i].y;
    }
    if (dev_max_x - dev_min_x <= eps_dev || dev_max_y - dev_min_y <= eps_dev) {
        return false;
    }
    for (uint32_t i = 0u; i < 4u; ++i) {
        if (!fs_clip_point_matches_rect_corner(
                tp[i].x, tp[i].y, dev_min_x, dev_min_y, dev_max_x, dev_max_y, eps_dev
            )) {
            return false;
        }
    }

    *out_min_x = dev_min_x;
    *out_min_y = dev_min_y;
    *out_max_x = dev_max_x;
    *out_max_y = dev_max_y;
    return true;
}

static bool fs_clip_try_extract_axis_aligned_round_rect_aabb(
    const FS_InternalState* st,
    float* out_min_x,
    float* out_min_y,
    float* out_max_x,
    float* out_max_y,
    float* out_radius
) {
    if (!st || !st->path_segments || st->path_count != 8u ||
        !out_min_x || !out_min_y || !out_max_x || !out_max_y || !out_radius) {
        return false;
    }
    const float eps_local = 1e-4f;
    const float eps_dev = 1e-4f;
    static const uint8_t kTypes[8] = {
        (uint8_t)FS_PATH_SEG_LINE, (uint8_t)FS_PATH_SEG_CUBIC,
        (uint8_t)FS_PATH_SEG_LINE, (uint8_t)FS_PATH_SEG_CUBIC,
        (uint8_t)FS_PATH_SEG_LINE, (uint8_t)FS_PATH_SEG_CUBIC,
        (uint8_t)FS_PATH_SEG_LINE, (uint8_t)FS_PATH_SEG_CUBIC
    };
    for (uint32_t i = 0u; i < 8u; ++i) {
        const FS_PathSegment* seg = &st->path_segments[i];
        if (seg->type != kTypes[i]) {
            return false;
        }
        if (i > 0u) {
            const FS_PathSegment* prev = &st->path_segments[i - 1u];
            if (fabsf(seg->x0 - prev->x1) > eps_local || fabsf(seg->y0 - prev->y1) > eps_local) {
                return false;
            }
        }
    }
    if (fabsf(st->path_segments[7].x1 - st->path_segments[0].x0) > eps_local ||
        fabsf(st->path_segments[7].y1 - st->path_segments[0].y0) > eps_local) {
        return false;
    }

    const FS_Transform2D* t = &st->current_transform;
    if (fabsf(t->b) > eps_dev || fabsf(t->c) > eps_dev) {
        return false;
    }
    const float sx = fabsf(t->a);
    const float sy = fabsf(t->d);
    if (sx <= eps_dev || sy <= eps_dev) {
        return false;
    }
    if (fabsf(sx - sy) > fmaxf(sx, sy) * 1e-4f) {
        return false;
    }

    float left = st->path_segments[0].x0;
    float top = st->path_segments[0].y0;
    float right = left;
    float bottom = top;
    for (uint32_t i = 0u; i < 8u; ++i) {
        const FS_PathSegment* seg = &st->path_segments[i];
        if (seg->x0 < left) left = seg->x0;
        if (seg->y0 < top) top = seg->y0;
        if (seg->x0 > right) right = seg->x0;
        if (seg->y0 > bottom) bottom = seg->y0;
        if (seg->x1 < left) left = seg->x1;
        if (seg->y1 < top) top = seg->y1;
        if (seg->x1 > right) right = seg->x1;
        if (seg->y1 > bottom) bottom = seg->y1;
    }
    const float w = right - left;
    const float h = bottom - top;
    if (w <= eps_local || h <= eps_local) {
        return false;
    }

    const FS_PathSegment* s0 = &st->path_segments[0];
    const FS_PathSegment* s2 = &st->path_segments[2];
    const FS_PathSegment* s4 = &st->path_segments[4];
    const FS_PathSegment* s6 = &st->path_segments[6];
    const float r0 = s0->x0 - left;
    const float r1 = right - s0->x1;
    const float r2 = s2->y0 - top;
    const float r3 = bottom - s2->y1;
    const float r4 = s4->x1 - left;
    const float r5 = s6->y1 - top;
    const float r = (r0 + r1 + r2 + r3 + r4 + r5) * (1.0f / 6.0f);
    if (r <= eps_local) {
        return false;
    }
    const float r_eps = fmaxf(1e-3f, r * 1e-2f);
    const float rvals[6] = {r0, r1, r2, r3, r4, r5};
    for (uint32_t i = 0u; i < 6u; ++i) {
        if (fabsf(rvals[i] - r) > r_eps) {
            return false;
        }
    }
    if (r > 0.5f * w + r_eps || r > 0.5f * h + r_eps) {
        return false;
    }

    for (uint32_t idx = 0u; idx < 8u; idx += 2u) {
        const FS_PathSegment* ls = &st->path_segments[idx];
        const float dx = ls->x1 - ls->x0;
        const float dy = ls->y1 - ls->y0;
        if (fabsf(dx) > eps_local && fabsf(dy) > eps_local) {
            return false;
        }
    }

    float dev_x = 0.0f;
    float dev_y = 0.0f;
    float dev_w = 0.0f;
    float dev_h = 0.0f;
    fs_transform_rect_to_aabb(t, left, top, w, h, &dev_x, &dev_y, &dev_w, &dev_h);
    if (dev_w <= eps_dev || dev_h <= eps_dev) {
        return false;
    }
    *out_min_x = dev_x;
    *out_min_y = dev_y;
    *out_max_x = dev_x + dev_w;
    *out_max_y = dev_y + dev_h;
    *out_radius = r * sx;
    return true;
}

static void fs_clip_clamp_fill_rect_to_parent_chain(
    const FS_Core* core,
    uint32_t parent_layer,
    int* io_x0,
    int* io_y0,
    int* io_x1,
    int* io_y1
) {
    if (!core || !io_x0 || !io_y0 || !io_x1 || !io_y1) {
        return;
    }
    if (parent_layer == UINT32_MAX || parent_layer >= core->clip_mask_layers) {
        return;
    }
    if (!core->clip_mask_layer_parent ||
        !core->clip_mask_layer_has_data ||
        !core->clip_mask_layer_min_x || !core->clip_mask_layer_min_y ||
        !core->clip_mask_layer_max_x || !core->clip_mask_layer_max_y) {
        *io_x0 = 1;
        *io_y0 = 1;
        *io_x1 = 0;
        *io_y1 = 0;
        return;
    }

    uint32_t cur = parent_layer;
    uint32_t guard = 0u;
    while (cur < core->clip_mask_layers && guard < core->clip_mask_layers) {
        if (core->clip_mask_layer_has_data[cur] == 0u) {
            *io_x0 = 1;
            *io_y0 = 1;
            *io_x1 = 0;
            *io_y1 = 0;
            return;
        }

        const int pmin_x = (int)core->clip_mask_layer_min_x[cur];
        const int pmin_y = (int)core->clip_mask_layer_min_y[cur];
        const int pmax_x = (int)core->clip_mask_layer_max_x[cur];
        const int pmax_y = (int)core->clip_mask_layer_max_y[cur];

        if (*io_x0 < pmin_x) *io_x0 = pmin_x;
        if (*io_y0 < pmin_y) *io_y0 = pmin_y;
        if (*io_x1 > pmax_x) *io_x1 = pmax_x;
        if (*io_y1 > pmax_y) *io_y1 = pmax_y;

        if (*io_x1 <= *io_x0 || *io_y1 <= *io_y0) {
            *io_x0 = 1;
            *io_y0 = 1;
            *io_x1 = 0;
            *io_y1 = 0;
            return;
        }

        const uint32_t next = core->clip_mask_layer_parent[cur];
        if (next == UINT32_MAX || next >= core->clip_mask_layers) {
            break;
        }
        cur = next;
        guard += 1u;
    }
}

static void fs_clip_bounds_include_point(
    const FS_Transform2D* t,
    float x,
    float y,
    bool* io_has_bounds,
    float* io_min_x,
    float* io_min_y,
    float* io_max_x,
    float* io_max_y
) {
    if (!t || !io_has_bounds || !io_min_x || !io_min_y || !io_max_x || !io_max_y) {
        return;
    }
    float tx = x;
    float ty = y;
    fs_transform_apply_point(t, x, y, &tx, &ty);
    if (!(*io_has_bounds)) {
        *io_has_bounds = true;
        *io_min_x = tx;
        *io_min_y = ty;
        *io_max_x = tx;
        *io_max_y = ty;
        return;
    }
    if (tx < *io_min_x) *io_min_x = tx;
    if (ty < *io_min_y) *io_min_y = ty;
    if (tx > *io_max_x) *io_max_x = tx;
    if (ty > *io_max_y) *io_max_y = ty;
}

static bool fs_clip_compute_path_device_bounds(
    const FS_InternalState* st,
    float* out_min_x,
    float* out_min_y,
    float* out_max_x,
    float* out_max_y
) {
    if (!st || !out_min_x || !out_min_y || !out_max_x || !out_max_y || st->path_count == 0u) {
        return false;
    }
    const FS_Transform2D* t = &st->current_transform;
    bool has_bounds = false;
    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;

    for (uint32_t i = 0u; i < st->path_count; ++i) {
        const FS_PathSegment* seg = &st->path_segments[i];
        if (seg->type == (uint8_t)FS_PATH_SEG_LINE) {
            fs_clip_bounds_include_point(t, seg->x0, seg->y0, &has_bounds, &min_x, &min_y, &max_x, &max_y);
            fs_clip_bounds_include_point(t, seg->x1, seg->y1, &has_bounds, &min_x, &min_y, &max_x, &max_y);
        } else if (seg->type == (uint8_t)FS_PATH_SEG_QUAD) {
            fs_clip_bounds_include_point(t, seg->x0, seg->y0, &has_bounds, &min_x, &min_y, &max_x, &max_y);
            fs_clip_bounds_include_point(t, seg->cx0, seg->cy0, &has_bounds, &min_x, &min_y, &max_x, &max_y);
            fs_clip_bounds_include_point(t, seg->x1, seg->y1, &has_bounds, &min_x, &min_y, &max_x, &max_y);
        } else if (seg->type == (uint8_t)FS_PATH_SEG_CUBIC) {
            fs_clip_bounds_include_point(t, seg->x0, seg->y0, &has_bounds, &min_x, &min_y, &max_x, &max_y);
            fs_clip_bounds_include_point(t, seg->cx0, seg->cy0, &has_bounds, &min_x, &min_y, &max_x, &max_y);
            fs_clip_bounds_include_point(t, seg->cx1, seg->cy1, &has_bounds, &min_x, &min_y, &max_x, &max_y);
            fs_clip_bounds_include_point(t, seg->x1, seg->y1, &has_bounds, &min_x, &min_y, &max_x, &max_y);
        } else {
            fs_clip_bounds_include_point(t, seg->x0, seg->y0, &has_bounds, &min_x, &min_y, &max_x, &max_y);
            fs_clip_bounds_include_point(t, seg->x1, seg->y1, &has_bounds, &min_x, &min_y, &max_x, &max_y);
        }
    }

    if (!has_bounds) {
        return false;
    }
    *out_min_x = min_x;
    *out_min_y = min_y;
    *out_max_x = max_x;
    *out_max_y = max_y;
    return true;
}

static bool fs_clip_apply_path_aabb_fallback(
    FS_Core* core,
    FS_InternalState* st,
    FS_ClipFailureReason reason,
    uint32_t edge_count
) {
    if (core) {
        fs_clip_diag_note_failure(core, reason, st ? st->path_count : 0u, edge_count);
    }
    if (!st || st->path_count == 0u) {
        return false;
    }

    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
    if (!fs_clip_compute_path_device_bounds(st, &min_x, &min_y, &max_x, &max_y)) {
        return fs_clip_intersect_aabb(st, 1.0f, 1.0f, 0.0f, 0.0f);
    }
    return fs_clip_intersect_aabb(st, min_x, min_y, max_x, max_y);
}

static void fs_clip_release_uncommitted_layer(
    FS_Core* core,
    uint32_t layer,
    bool layer_was_fresh,
    uint32_t previous_next_layer
) {
    if (!core || layer >= core->clip_mask_layers) {
        return;
    }
    if (core->clip_mask_layer_has_data) {
        core->clip_mask_layer_has_data[layer] = 0u;
    }
    if (core->clip_mask_layer_hash_valid) {
        core->clip_mask_layer_hash_valid[layer] = 0u;
    }
    if (core->clip_mask_layer_hash) {
        core->clip_mask_layer_hash[layer] = 0u;
    }
    if (core->clip_mask_layer_parent) {
        core->clip_mask_layer_parent[layer] = UINT32_MAX;
    }
    if (core->clip_mask_layer_last_used_frame) {
        core->clip_mask_layer_last_used_frame[layer] = 0u;
    }
    if (layer_was_fresh && core->clip_mask_next_layer == previous_next_layer + 1u && layer == previous_next_layer) {
        core->clip_mask_next_layer = previous_next_layer;
    }
}

static bool fs_clip_path_with_mode(
    FS_Core* core,
    uint32_t fill_mode,
    bool use_fill_rule_override,
    FS_FillRule fill_rule_override
) {
    FS_InternalState* st = fs_state(core);
    if (core) {
        core->clip_requests_this_frame += 1u;
    }

    if (!core || !st) {
        if (core) {
            fs_clip_diag_note_failure(core, FS_CLIP_FAILURE_INVALID_INPUT, 0u, 0u);
        }
        return false;
    }
    if (st->path_count == 0u) {
        fs_clip_diag_note_failure(core, FS_CLIP_FAILURE_EMPTY_PATH, 0u, 0u);
        return false;
    }
    if (!core->clip_mask_texture || core->clip_mask_width == 0u || core->clip_mask_height == 0u || core->clip_mask_layers == 0u) {
        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_INVALID_INPUT, 0u);
    }
    uint32_t clip_fill_mode = fill_mode;
    if (clip_fill_mode != FS_CLIP_FILL_MODE_COVERAGE && clip_fill_mode != FS_CLIP_FILL_MODE_SDF) {
        clip_fill_mode = FS_CLIP_FILL_MODE_COVERAGE;
    }

    FS_FillRule fill_rule =
        (st->style_fill_rule == (uint8_t)FS_FILL_RULE_EVENODD) ? FS_FILL_RULE_EVENODD : FS_FILL_RULE_NONZERO;
    if (use_fill_rule_override) {
        if (fill_rule_override != FS_FILL_RULE_NONZERO && fill_rule_override != FS_FILL_RULE_EVENODD) {
            fs_clip_diag_note_failure(core, FS_CLIP_FAILURE_INVALID_INPUT, st->path_count, 0u);
            return false;
        }
        fill_rule = fill_rule_override;
    }
    if (clip_fill_mode == FS_CLIP_FILL_MODE_COVERAGE) {
        float rect_min_x = 0.0f;
        float rect_min_y = 0.0f;
        float rect_max_x = 0.0f;
        float rect_max_y = 0.0f;
        if (fs_clip_try_extract_axis_aligned_rect_aabb(
                st, &rect_min_x, &rect_min_y, &rect_max_x, &rect_max_y
            )) {
            return fs_clip_intersect_aabb(st, rect_min_x, rect_min_y, rect_max_x, rect_max_y);
        }
    }
    if (!core->clip_mask_texture ||
        core->clip_mask_width == 0u || core->clip_mask_height == 0u || core->clip_mask_layers == 0u) {
        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_INVALID_INPUT, 0u);
    }

    bool analytic_round_rect = false;
    float rr_min_x = 0.0f;
    float rr_min_y = 0.0f;
    float rr_max_x = 0.0f;
    float rr_max_y = 0.0f;
    float rr_radius = 0.0f;
    if (clip_fill_mode == FS_CLIP_FILL_MODE_COVERAGE) {
        if (fs_clip_try_extract_axis_aligned_round_rect_aabb(
                st, &rr_min_x, &rr_min_y, &rr_max_x, &rr_max_y, &rr_radius
            )) {
            analytic_round_rect = true;
            clip_fill_mode = FS_CLIP_FILL_MODE_ROUND_RECT;
        }
    }

    uint64_t parent_hash = 0xA5A5A5A55A5A5A5Aull;
    uint32_t parent_layer = UINT32_MAX;
    if (st->clip_path_enabled &&
        st->clip_path_layer < core->clip_mask_layers &&
        core->clip_mask_layer_hash_valid &&
        core->clip_mask_layer_hash_valid[st->clip_path_layer]) {
        parent_hash = core->clip_mask_layer_hash[st->clip_path_layer];
        parent_layer = (uint32_t)st->clip_path_layer;
    }
    uint64_t clip_hash = 1469598103934665603ull;
    clip_hash = fs_hash64_u32(clip_hash, (uint32_t)fill_rule);
    clip_hash = fs_hash64_u32(clip_hash, clip_fill_mode);
    clip_hash = fs_hash64_mix(clip_hash, parent_hash);
    clip_hash = fs_hash64_u32(clip_hash, st->path_count);
    clip_hash = fs_hash64_f32(clip_hash, st->current_transform.a);
    clip_hash = fs_hash64_f32(clip_hash, st->current_transform.b);
    clip_hash = fs_hash64_f32(clip_hash, st->current_transform.c);
    clip_hash = fs_hash64_f32(clip_hash, st->current_transform.d);
    clip_hash = fs_hash64_f32(clip_hash, st->current_transform.e);
    clip_hash = fs_hash64_f32(clip_hash, st->current_transform.f);
    for (uint32_t i = 0u; i < st->path_count; ++i) {
        const FS_PathSegment* seg = &st->path_segments[i];
        clip_hash = fs_hash64_u32(clip_hash, (uint32_t)seg->type);
        clip_hash = fs_hash64_f32(clip_hash, seg->x0);
        clip_hash = fs_hash64_f32(clip_hash, seg->y0);
        clip_hash = fs_hash64_f32(clip_hash, seg->cx0);
        clip_hash = fs_hash64_f32(clip_hash, seg->cy0);
        clip_hash = fs_hash64_f32(clip_hash, seg->cx1);
        clip_hash = fs_hash64_f32(clip_hash, seg->cy1);
        clip_hash = fs_hash64_f32(clip_hash, seg->x1);
        clip_hash = fs_hash64_f32(clip_hash, seg->y1);
    }
    if (core->clip_cache_enabled && core->clip_mask_layer_hash_valid && core->clip_mask_layer_hash) {
        for (uint32_t layer_i = 0u; layer_i < core->clip_mask_layers; ++layer_i) {
            if (!core->clip_mask_layer_hash_valid[layer_i]) {
                continue;
            }
            if (core->clip_mask_layer_hash[layer_i] != clip_hash) {
                continue;
            }
            st->clip_path_enabled = 1u;
            st->clip_path_layer = (uint8_t)layer_i;
            if (core->clip_mask_next_layer <= layer_i) {
                core->clip_mask_next_layer = layer_i + 1u;
            }
            core->clip_cache_hits_this_frame += 1u;
            fs_clip_diag_note_layer_usage(core, layer_i);
            return true;
        }
    }

    const uint32_t previous_next_layer = core->clip_mask_next_layer;
    uint32_t layer = 0u;
    if (!fs_clip_try_acquire_layer_with_policy(core, st, &layer)) {
        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_LAYER_EXHAUSTED, 0u);
    }
    const bool layer_was_fresh = (layer == previous_next_layer && core->clip_mask_next_layer == previous_next_layer + 1u);

    FS_ClipEdge* edges = NULL;
    uint32_t edge_count = 0u;
    uint32_t edge_capacity = 0u;
    size_t edge_offset = core->clip_edge_count;

    float min_x = rr_min_x;
    float min_y = rr_min_y;
    float max_x = rr_max_x;
    float max_y = rr_max_y;
    bool has_bounds = analytic_round_rect;
    float edge_min_x = 0.0f;
    float edge_min_y = 0.0f;
    float edge_max_x = 0.0f;
    float edge_max_y = 0.0f;
    bool edge_has_bounds = false;
    bool have_prev_end = false;
    bool have_subpath = false;
    float prev_end_x = 0.0f;
    float prev_end_y = 0.0f;
    float subpath_start_x = 0.0f;
    float subpath_start_y = 0.0f;

    if (!analytic_round_rect) {
        for (uint32_t i = 0u; i < st->path_count; ++i) {
            const FS_PathSegment* seg = &st->path_segments[i];
            const bool contour_break =
                !have_prev_end ||
                fabsf(prev_end_x - seg->x0) > 1e-4f ||
                fabsf(prev_end_y - seg->y0) > 1e-4f;

            if (contour_break) {
                if (have_subpath) {
                    if (!fs_clip_edges_push(
                            &edges, &edge_count, &edge_capacity, prev_end_x, prev_end_y, subpath_start_x, subpath_start_y,
                            &edge_min_x, &edge_min_y, &edge_max_x, &edge_max_y, &edge_has_bounds
                        )) {
                        free(edges);
                        fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
                        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_EDGE_ALLOC, edge_count);
                    }
                }
                subpath_start_x = seg->x0;
                subpath_start_y = seg->y0;
                have_subpath = true;
            }

            if (seg->type == (uint8_t)FS_PATH_SEG_LINE) {
                if (!fs_clip_edges_push(
                        &edges, &edge_count, &edge_capacity, seg->x0, seg->y0, seg->x1, seg->y1,
                        &edge_min_x, &edge_min_y, &edge_max_x, &edge_max_y, &edge_has_bounds
                    )) {
                    free(edges);
                    fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
                    return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_EDGE_ALLOC, edge_count);
                }
            } else if (seg->type == (uint8_t)FS_PATH_SEG_QUAD) {
                const float len_a = hypotf(seg->cx0 - seg->x0, seg->cy0 - seg->y0);
                const float len_b = hypotf(seg->x1 - seg->cx0, seg->y1 - seg->cy0);
                uint32_t steps = (uint32_t)((len_a + len_b) / 14.0f) + 8u;
                if (steps < 8u) {
                    steps = 8u;
                } else if (steps > 96u) {
                    steps = 96u;
                }
                float prev_x = seg->x0;
                float prev_y = seg->y0;
                for (uint32_t s = 1u; s <= steps; ++s) {
                    const float u = (float)s / (float)steps;
                    float cur_x = 0.0f;
                    float cur_y = 0.0f;
                    fs_eval_quad_point(seg->x0, seg->y0, seg->cx0, seg->cy0, seg->x1, seg->y1, u, &cur_x, &cur_y);
                    if (!fs_clip_edges_push(
                            &edges, &edge_count, &edge_capacity, prev_x, prev_y, cur_x, cur_y,
                            &edge_min_x, &edge_min_y, &edge_max_x, &edge_max_y, &edge_has_bounds
                        )) {
                        free(edges);
                        fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
                        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_EDGE_ALLOC, edge_count);
                    }
                    prev_x = cur_x;
                    prev_y = cur_y;
                }
            } else if (seg->type == (uint8_t)FS_PATH_SEG_CUBIC) {
                const float len_a = hypotf(seg->cx0 - seg->x0, seg->cy0 - seg->y0);
                const float len_b = hypotf(seg->cx1 - seg->cx0, seg->cy1 - seg->cy0);
                const float len_c = hypotf(seg->x1 - seg->cx1, seg->y1 - seg->cy1);
                uint32_t steps = (uint32_t)((len_a + len_b + len_c) / 12.0f) + 10u;
                if (steps < 10u) {
                    steps = 10u;
                } else if (steps > 144u) {
                    steps = 144u;
                }
                float prev_x = seg->x0;
                float prev_y = seg->y0;
                for (uint32_t s = 1u; s <= steps; ++s) {
                    const float u = (float)s / (float)steps;
                    float cur_x = 0.0f;
                    float cur_y = 0.0f;
                    fs_eval_cubic_point(
                        seg->x0,
                        seg->y0,
                        seg->cx0,
                        seg->cy0,
                        seg->cx1,
                        seg->cy1,
                        seg->x1,
                        seg->y1,
                        u,
                        &cur_x,
                        &cur_y
                    );
                    if (!fs_clip_edges_push(
                            &edges, &edge_count, &edge_capacity, prev_x, prev_y, cur_x, cur_y,
                            &edge_min_x, &edge_min_y, &edge_max_x, &edge_max_y, &edge_has_bounds
                        )) {
                        free(edges);
                        fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
                        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_EDGE_ALLOC, edge_count);
                    }
                    prev_x = cur_x;
                    prev_y = cur_y;
                }
            }

            prev_end_x = seg->x1;
            prev_end_y = seg->y1;
            have_prev_end = true;
        }

        if (have_subpath) {
            if (!fs_clip_edges_push(
                    &edges, &edge_count, &edge_capacity, prev_end_x, prev_end_y, subpath_start_x, subpath_start_y,
                    &edge_min_x, &edge_min_y, &edge_max_x, &edge_max_y, &edge_has_bounds
                )) {
                free(edges);
                fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
                return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_EDGE_ALLOC, edge_count);
            }
        }

        if (!fs_clip_compute_path_device_bounds(st, &min_x, &min_y, &max_x, &max_y)) {
            free(edges);
            fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
            return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_INVALID_BOUNDS, edge_count);
        }
        has_bounds = true;
    }

    if (!has_bounds || (edge_count == 0u && !analytic_round_rect)) {
        free(edges);
        fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_EMPTY_PATH, edge_count);
    }

    int x0 = (int)floorf(min_x);
    int y0 = (int)floorf(min_y);
    int x1 = (int)ceilf(max_x);
    int y1 = (int)ceilf(max_y);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (int)core->clip_mask_width) x1 = (int)core->clip_mask_width;
    if (y1 > (int)core->clip_mask_height) y1 = (int)core->clip_mask_height;
    if (parent_layer != UINT32_MAX) {
        fs_clip_clamp_fill_rect_to_parent_chain(core, parent_layer, &x0, &y0, &x1, &y1);
    }
    const bool has_fill_rect = (x1 > x0) && (y1 > y0);
    if (!has_fill_rect) {
        free(edges);
        fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
        if (parent_layer != UINT32_MAX) {
            return fs_clip_intersect_aabb(st, 1.0f, 1.0f, 0.0f, 0.0f);
        }
        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_INVALID_BOUNDS, edge_count);
    }

    // Correctness + performance clear policy:
    // clear union(old_layer_bounds, new_fill_bounds) with a small guard-band.
    uint32_t clear_x0 = (uint32_t)x0;
    uint32_t clear_y0 = (uint32_t)y0;
    uint32_t clear_x1 = (uint32_t)x1;
    uint32_t clear_y1 = (uint32_t)y1;
    if (core->clip_mask_layer_has_data && layer < core->clip_mask_layers &&
        core->clip_mask_layer_has_data[layer] &&
        core->clip_mask_layer_min_x && core->clip_mask_layer_min_y &&
        core->clip_mask_layer_max_x && core->clip_mask_layer_max_y) {
        const uint32_t old_min_x = core->clip_mask_layer_min_x[layer];
        const uint32_t old_min_y = core->clip_mask_layer_min_y[layer];
        const uint32_t old_max_x = core->clip_mask_layer_max_x[layer];
        const uint32_t old_max_y = core->clip_mask_layer_max_y[layer];
        if (old_max_x > old_min_x && old_max_y > old_min_y) {
            if (old_min_x < clear_x0) clear_x0 = old_min_x;
            if (old_min_y < clear_y0) clear_y0 = old_min_y;
            if (old_max_x > clear_x1) clear_x1 = old_max_x;
            if (old_max_y > clear_y1) clear_y1 = old_max_y;
        }
    }
    if (clear_x0 > 0u) clear_x0 -= 1u;
    if (clear_y0 > 0u) clear_y0 -= 1u;
    if (clear_x1 < core->clip_mask_width) clear_x1 += 1u;
    if (clear_y1 < core->clip_mask_height) clear_y1 += 1u;
    if (clear_x1 > core->clip_mask_width) clear_x1 = core->clip_mask_width;
    if (clear_y1 > core->clip_mask_height) clear_y1 = core->clip_mask_height;
    if (clear_x1 <= clear_x0 || clear_y1 <= clear_y0) {
        free(edges);
        fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_INVALID_BOUNDS, edge_count);
    }

    if (!analytic_round_rect) {
        if (!fs_ensure_clip_edge_cpu_capacity(core, edge_offset + edge_count)) {
            free(edges);
            fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
            return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_EDGE_ALLOC, edge_count);
        }
        FS_ClipEdgeGPU* gpu_edges = (FS_ClipEdgeGPU*)core->clip_edge_cpu;
        for (uint32_t i = 0u; i < edge_count; ++i) {
            gpu_edges[edge_offset + i].x0 = edges[i].x0;
            gpu_edges[edge_offset + i].y0 = edges[i].y0;
            gpu_edges[edge_offset + i].x1 = edges[i].x1;
            gpu_edges[edge_offset + i].y1 = edges[i].y1;
        }
        core->clip_edge_count = edge_offset + edge_count;
    }

    if (!fs_ensure_clip_job_cpu_capacity(core, core->clip_job_count + 1u)) {
        free(edges);
        core->clip_edge_count = edge_offset;
        fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_JOB_ALLOC, edge_count);
    }
    FS_ClipJobGPU* jobs = (FS_ClipJobGPU*)core->clip_job_cpu;
    FS_ClipJobTransformGPU* job_xforms = (FS_ClipJobTransformGPU*)core->clip_job_xform_cpu;
    FS_ClipJobGPU* job = &jobs[core->clip_job_count++];
    FS_ClipJobTransformGPU* job_xform = job_xforms ? &job_xforms[core->clip_job_count - 1u] : NULL;
    job->edge_offset = analytic_round_rect ? 0u : (uint32_t)edge_offset;
    job->edge_count = analytic_round_rect ? 0u : edge_count;
    job->layer = layer;
    job->fill_rule = (uint32_t)fill_rule;
    job->fill_min_x = (uint32_t)x0;
    job->fill_min_y = (uint32_t)y0;
    job->fill_max_x = (uint32_t)x1;
    job->fill_max_y = (uint32_t)y1;
    job->clear_min_x = clear_x0;
    job->clear_min_y = clear_y0;
    job->clear_max_x = clear_x1;
    job->clear_max_y = clear_y1;
    job->parent_layer = parent_layer;
    job->has_parent = (parent_layer != UINT32_MAX) ? 1u : 0u;
    if (analytic_round_rect) {
        memcpy(&job->scale_hint_bits, &rr_radius, sizeof(uint32_t));
    } else {
        const float sx = sqrtf(st->current_transform.a * st->current_transform.a + st->current_transform.b * st->current_transform.b);
        const float sy = sqrtf(st->current_transform.c * st->current_transform.c + st->current_transform.d * st->current_transform.d);
        const float scale_hint = fmaxf(sx, sy);
        memcpy(&job->scale_hint_bits, &scale_hint, sizeof(uint32_t));
    }
    job->fill_mode = clip_fill_mode;
    if (job_xform) {
        job_xform->xform0[0] = st->current_transform.a;
        job_xform->xform0[1] = st->current_transform.b;
        job_xform->xform0[2] = st->current_transform.c;
        job_xform->xform0[3] = st->current_transform.d;
        job_xform->xform1[0] = st->current_transform.e;
        job_xform->xform1[1] = st->current_transform.f;
        job_xform->xform1[2] = 0.0f;
        job_xform->xform1[3] = 0.0f;
    }

    if (core->clip_mask_layer_has_data) {
        core->clip_mask_layer_has_data[layer] = 1u;
    }
    if (core->clip_mask_layer_min_x) {
        core->clip_mask_layer_min_x[layer] = (uint32_t)x0;
    }
    if (core->clip_mask_layer_min_y) {
        core->clip_mask_layer_min_y[layer] = (uint32_t)y0;
    }
    if (core->clip_mask_layer_max_x) {
        core->clip_mask_layer_max_x[layer] = (uint32_t)x1;
    }
    if (core->clip_mask_layer_max_y) {
        core->clip_mask_layer_max_y[layer] = (uint32_t)y1;
    }
    if (core->clip_mask_layer_hash && core->clip_mask_layer_hash_valid) {
        core->clip_mask_layer_hash[layer] = clip_hash;
        core->clip_mask_layer_hash_valid[layer] = 1u;
    }
    if (core->clip_mask_layer_parent) {
        core->clip_mask_layer_parent[layer] = parent_layer;
    }

    free(edges);

    st->clip_path_enabled = 1u;
    st->clip_path_layer = (uint8_t)layer;
    core->clip_jobs_enqueued_this_frame += 1u;
    fs_clip_diag_note_layer_usage(core, layer);
    return true;
}

bool fs_clip_path(FS_Core* core) {
    return fs_clip_path_with_mode(core, FS_CLIP_FILL_MODE_COVERAGE, false, FS_FILL_RULE_NONZERO);
}

bool fs_clip_path_with_fill_rule(FS_Core* core, FS_FillRule fill_rule) {
    return fs_clip_path_with_mode(core, FS_CLIP_FILL_MODE_COVERAGE, true, fill_rule);
}

void fs_style_reset(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    fs_style_reset_state(st);
}

bool fs_style_set_line_width(FS_Core* core, float width) {
    FS_InternalState* st = fs_state(core);
    if (!st || width <= 0.0f) {
        return false;
    }
    st->style_line_width = width;
    return true;
}

bool fs_style_set_line_cap(FS_Core* core, FS_LineCap cap) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (cap != FS_LINE_CAP_BUTT && cap != FS_LINE_CAP_ROUND && cap != FS_LINE_CAP_SQUARE) {
        return false;
    }
    st->style_line_cap = (uint8_t)cap;
    return true;
}

bool fs_style_set_line_join(FS_Core* core, FS_LineJoin join) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (join != FS_LINE_JOIN_MITER && join != FS_LINE_JOIN_ROUND && join != FS_LINE_JOIN_BEVEL) {
        return false;
    }
    st->style_line_join = (uint8_t)join;
    return true;
}

bool fs_style_set_fill_rule(FS_Core* core, FS_FillRule fill_rule) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (fill_rule != FS_FILL_RULE_NONZERO && fill_rule != FS_FILL_RULE_EVENODD) {
        return false;
    }
    st->style_fill_rule = (uint8_t)fill_rule;
    return true;
}

bool fs_style_set_miter_limit(FS_Core* core, float limit) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(limit) || limit <= 0.0f) {
        return false;
    }
    st->style_miter_limit = limit;
    return true;
}

float fs_style_get_miter_limit(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_miter_limit) || st->style_miter_limit <= 0.0f) {
        return 10.0f;
    }
    return st->style_miter_limit;
}

bool fs_style_set_global_alpha(FS_Core* core, float alpha) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(alpha)) {
        return false;
    }
    if (alpha < 0.0f) {
        alpha = 0.0f;
    } else if (alpha > 1.0f) {
        alpha = 1.0f;
    }
    st->style_global_alpha = alpha;
    return true;
}

float fs_style_get_global_alpha(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return 1.0f;
    }
    return st->style_global_alpha;
}

bool fs_style_set_global_composite_operation(FS_Core* core, FS_GlobalCompositeOperation op) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (op) {
        case FS_GLOBAL_COMPOSITE_SOURCE_OVER:
        case FS_GLOBAL_COMPOSITE_COPY:
        case FS_GLOBAL_COMPOSITE_LIGHTER:
        case FS_GLOBAL_COMPOSITE_DESTINATION_OVER:
        case FS_GLOBAL_COMPOSITE_SOURCE_IN:
        case FS_GLOBAL_COMPOSITE_SOURCE_OUT:
        case FS_GLOBAL_COMPOSITE_DESTINATION_IN:
        case FS_GLOBAL_COMPOSITE_DESTINATION_OUT:
        case FS_GLOBAL_COMPOSITE_XOR:
        case FS_GLOBAL_COMPOSITE_SOURCE_ATOP:
        case FS_GLOBAL_COMPOSITE_DESTINATION_ATOP:
            st->style_composite_op = (uint8_t)op;
            return true;
        default:
            return false;
    }
}

FS_GlobalCompositeOperation fs_style_get_global_composite_operation(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_GLOBAL_COMPOSITE_SOURCE_OVER;
    }
    switch ((FS_GlobalCompositeOperation)st->style_composite_op) {
        case FS_GLOBAL_COMPOSITE_COPY:
        case FS_GLOBAL_COMPOSITE_LIGHTER:
        case FS_GLOBAL_COMPOSITE_DESTINATION_OVER:
        case FS_GLOBAL_COMPOSITE_SOURCE_IN:
        case FS_GLOBAL_COMPOSITE_SOURCE_OUT:
        case FS_GLOBAL_COMPOSITE_DESTINATION_IN:
        case FS_GLOBAL_COMPOSITE_DESTINATION_OUT:
        case FS_GLOBAL_COMPOSITE_XOR:
        case FS_GLOBAL_COMPOSITE_SOURCE_ATOP:
        case FS_GLOBAL_COMPOSITE_DESTINATION_ATOP:
            return (FS_GlobalCompositeOperation)st->style_composite_op;
        case FS_GLOBAL_COMPOSITE_SOURCE_OVER:
        default:
            return FS_GLOBAL_COMPOSITE_SOURCE_OVER;
    }
}

bool fs_style_set_shadow_color(FS_Core* core, uint32_t color_rgba8) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->style_shadow_color_rgba8 = color_rgba8;
    return true;
}

uint32_t fs_style_get_shadow_color(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return 0u;
    }
    return st->style_shadow_color_rgba8;
}

bool fs_style_set_fill_color(FS_Core* core, uint32_t color_rgba8) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->style_fill_color_rgba8 = color_rgba8;
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_SOLID;
    return true;
}

uint32_t fs_style_get_fill_color(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return 0xFF000000u;
    }
    return st->style_fill_color_rgba8;
}

bool fs_style_set_stroke_color(FS_Core* core, uint32_t color_rgba8) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->style_stroke_color_rgba8 = color_rgba8;
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_SOLID;
    return true;
}

uint32_t fs_style_get_stroke_color(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return 0xFF000000u;
    }
    return st->style_stroke_color_rgba8;
}

bool fs_style_set_fill_linear_gradient(FS_Core* core, const FS_LinearGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_linear_gradient(&st->style_fill_linear_gradient, gradient)) {
        return false;
    }
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT;
    return true;
}

bool fs_style_set_stroke_linear_gradient(FS_Core* core, const FS_LinearGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_linear_gradient(&st->style_stroke_linear_gradient, gradient)) {
        return false;
    }
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT;
    return true;
}

bool fs_style_set_fill_radial_gradient(FS_Core* core, const FS_RadialGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_radial_gradient(&st->style_fill_radial_gradient, gradient)) {
        return false;
    }
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT;
    return true;
}

bool fs_style_set_stroke_radial_gradient(FS_Core* core, const FS_RadialGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_radial_gradient(&st->style_stroke_radial_gradient, gradient)) {
        return false;
    }
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT;
    return true;
}

bool fs_style_set_fill_conic_gradient(FS_Core* core, const FS_ConicGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_conic_gradient(&st->style_fill_conic_gradient, gradient)) {
        return false;
    }
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_CONIC_GRADIENT;
    return true;
}

bool fs_style_set_stroke_conic_gradient(FS_Core* core, const FS_ConicGradient* gradient) {
    FS_InternalState* st = fs_state(core);
    if (!st || !gradient) {
        return false;
    }
    if (!fs_style_copy_conic_gradient(&st->style_stroke_conic_gradient, gradient)) {
        return false;
    }
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_CONIC_GRADIENT;
    return true;
}

bool fs_style_set_fill_pattern(FS_Core* core, const FS_Pattern* pattern) {
    FS_InternalState* st = fs_state(core);
    if (!st || !pattern) {
        return false;
    }
    if (!fs_style_copy_pattern(&st->style_fill_pattern, pattern)) {
        return false;
    }
    uint32_t atlas_x = 0u;
    uint32_t atlas_y = 0u;
    if (!fs_image_handle_resolve_atlas_origin(core, &st->style_fill_pattern.handle, &atlas_x, &atlas_y)) {
        return false;
    }
    st->style_fill_paint_type = (uint8_t)FS_STYLE_PAINT_PATTERN;
    return true;
}

bool fs_style_set_stroke_pattern(FS_Core* core, const FS_Pattern* pattern) {
    FS_InternalState* st = fs_state(core);
    if (!st || !pattern) {
        return false;
    }
    if (!fs_style_copy_pattern(&st->style_stroke_pattern, pattern)) {
        return false;
    }
    uint32_t atlas_x = 0u;
    uint32_t atlas_y = 0u;
    if (!fs_image_handle_resolve_atlas_origin(core, &st->style_stroke_pattern.handle, &atlas_x, &atlas_y)) {
        return false;
    }
    st->style_stroke_paint_type = (uint8_t)FS_STYLE_PAINT_PATTERN;
    return true;
}

bool fs_style_set_shadow_blur(FS_Core* core, float blur_px) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(blur_px) || blur_px < 0.0f) {
        return false;
    }
    if (blur_px > FS_SHADOW_BLUR_MAX) {
        blur_px = FS_SHADOW_BLUR_MAX;
    }
    st->style_shadow_blur = blur_px;
    return true;
}

float fs_style_get_shadow_blur(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_shadow_blur) || st->style_shadow_blur < 0.0f) {
        return 0.0f;
    }
    return st->style_shadow_blur;
}

bool fs_style_set_shadow_offset(FS_Core* core, float offset_x, float offset_y) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(offset_x) || !isfinite(offset_y)) {
        return false;
    }
    st->style_shadow_offset_x = offset_x;
    st->style_shadow_offset_y = offset_y;
    return true;
}

float fs_style_get_shadow_offset_x(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_shadow_offset_x)) {
        return 0.0f;
    }
    return st->style_shadow_offset_x;
}

float fs_style_get_shadow_offset_y(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_shadow_offset_y)) {
        return 0.0f;
    }
    return st->style_shadow_offset_y;
}

bool fs_style_set_text_align(FS_Core* core, FS_TextAlign align) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (align) {
        case FS_TEXT_ALIGN_START:
        case FS_TEXT_ALIGN_LEFT:
        case FS_TEXT_ALIGN_CENTER:
        case FS_TEXT_ALIGN_RIGHT:
        case FS_TEXT_ALIGN_END:
            st->style_text_align = (uint8_t)align;
            return true;
        default:
            return false;
    }
}

FS_TextAlign fs_style_get_text_align(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_TEXT_ALIGN_START;
    }
    switch ((FS_TextAlign)st->style_text_align) {
        case FS_TEXT_ALIGN_LEFT:
        case FS_TEXT_ALIGN_CENTER:
        case FS_TEXT_ALIGN_RIGHT:
        case FS_TEXT_ALIGN_END:
            return (FS_TextAlign)st->style_text_align;
        case FS_TEXT_ALIGN_START:
        default:
            return FS_TEXT_ALIGN_START;
    }
}

bool fs_style_set_text_baseline(FS_Core* core, FS_TextBaseline baseline) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (baseline) {
        case FS_TEXT_BASELINE_TOP:
        case FS_TEXT_BASELINE_HANGING:
        case FS_TEXT_BASELINE_MIDDLE:
        case FS_TEXT_BASELINE_ALPHABETIC:
        case FS_TEXT_BASELINE_IDEOGRAPHIC:
        case FS_TEXT_BASELINE_BOTTOM:
            st->style_text_baseline = (uint8_t)baseline;
            return true;
        default:
            return false;
    }
}

FS_TextBaseline fs_style_get_text_baseline(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_TEXT_BASELINE_ALPHABETIC;
    }
    switch ((FS_TextBaseline)st->style_text_baseline) {
        case FS_TEXT_BASELINE_TOP:
        case FS_TEXT_BASELINE_HANGING:
        case FS_TEXT_BASELINE_MIDDLE:
        case FS_TEXT_BASELINE_IDEOGRAPHIC:
        case FS_TEXT_BASELINE_BOTTOM:
            return (FS_TextBaseline)st->style_text_baseline;
        case FS_TEXT_BASELINE_ALPHABETIC:
        default:
            return FS_TEXT_BASELINE_ALPHABETIC;
    }
}

bool fs_style_set_text_direction(FS_Core* core, FS_TextDirection direction) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (direction) {
        case FS_TEXT_DIRECTION_INHERIT:
        case FS_TEXT_DIRECTION_LTR:
        case FS_TEXT_DIRECTION_RTL:
            st->style_text_direction = (uint8_t)direction;
            return true;
        default:
            return false;
    }
}

FS_TextDirection fs_style_get_text_direction(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_TEXT_DIRECTION_LTR;
    }
    switch ((FS_TextDirection)st->style_text_direction) {
        case FS_TEXT_DIRECTION_INHERIT:
        case FS_TEXT_DIRECTION_LTR:
        case FS_TEXT_DIRECTION_RTL:
            return (FS_TextDirection)st->style_text_direction;
        default:
            return FS_TEXT_DIRECTION_LTR;
    }
}

bool fs_style_set_font_kerning(FS_Core* core, FS_FontKerning kerning) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (kerning) {
        case FS_FONT_KERNING_AUTO:
        case FS_FONT_KERNING_NORMAL:
        case FS_FONT_KERNING_NONE:
            st->style_font_kerning = (uint8_t)kerning;
            return true;
        default:
            return false;
    }
}

FS_FontKerning fs_style_get_font_kerning(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_FONT_KERNING_AUTO;
    }
    switch ((FS_FontKerning)st->style_font_kerning) {
        case FS_FONT_KERNING_AUTO:
        case FS_FONT_KERNING_NORMAL:
        case FS_FONT_KERNING_NONE:
            return (FS_FontKerning)st->style_font_kerning;
        default:
            return FS_FONT_KERNING_AUTO;
    }
}

bool fs_style_set_text_rendering(FS_Core* core, FS_TextRendering rendering) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (rendering) {
        case FS_TEXT_RENDERING_AUTO:
        case FS_TEXT_RENDERING_OPTIMIZE_SPEED:
        case FS_TEXT_RENDERING_OPTIMIZE_LEGIBILITY:
        case FS_TEXT_RENDERING_GEOMETRIC_PRECISION:
            st->style_text_rendering = (uint8_t)rendering;
            return true;
        default:
            return false;
    }
}

FS_TextRendering fs_style_get_text_rendering(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_TEXT_RENDERING_AUTO;
    }
    switch ((FS_TextRendering)st->style_text_rendering) {
        case FS_TEXT_RENDERING_AUTO:
        case FS_TEXT_RENDERING_OPTIMIZE_SPEED:
        case FS_TEXT_RENDERING_OPTIMIZE_LEGIBILITY:
        case FS_TEXT_RENDERING_GEOMETRIC_PRECISION:
            return (FS_TextRendering)st->style_text_rendering;
        default:
            return FS_TEXT_RENDERING_AUTO;
    }
}

bool fs_style_set_font_stretch(FS_Core* core, FS_FontStretch stretch) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (stretch) {
        case FS_FONT_STRETCH_ULTRA_CONDENSED:
        case FS_FONT_STRETCH_EXTRA_CONDENSED:
        case FS_FONT_STRETCH_CONDENSED:
        case FS_FONT_STRETCH_SEMI_CONDENSED:
        case FS_FONT_STRETCH_NORMAL:
        case FS_FONT_STRETCH_SEMI_EXPANDED:
        case FS_FONT_STRETCH_EXPANDED:
        case FS_FONT_STRETCH_EXTRA_EXPANDED:
        case FS_FONT_STRETCH_ULTRA_EXPANDED:
            st->style_font_stretch = (uint8_t)stretch;
            return true;
        default:
            return false;
    }
}

FS_FontStretch fs_style_get_font_stretch(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_FONT_STRETCH_NORMAL;
    }
    switch ((FS_FontStretch)st->style_font_stretch) {
        case FS_FONT_STRETCH_ULTRA_CONDENSED:
        case FS_FONT_STRETCH_EXTRA_CONDENSED:
        case FS_FONT_STRETCH_CONDENSED:
        case FS_FONT_STRETCH_SEMI_CONDENSED:
        case FS_FONT_STRETCH_NORMAL:
        case FS_FONT_STRETCH_SEMI_EXPANDED:
        case FS_FONT_STRETCH_EXPANDED:
        case FS_FONT_STRETCH_EXTRA_EXPANDED:
        case FS_FONT_STRETCH_ULTRA_EXPANDED:
            return (FS_FontStretch)st->style_font_stretch;
        default:
            return FS_FONT_STRETCH_NORMAL;
    }
}

bool fs_style_set_font_variant_caps(FS_Core* core, FS_FontVariantCaps variant_caps) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (variant_caps) {
        case FS_FONT_VARIANT_CAPS_NORMAL:
        case FS_FONT_VARIANT_CAPS_SMALL_CAPS:
        case FS_FONT_VARIANT_CAPS_ALL_SMALL_CAPS:
        case FS_FONT_VARIANT_CAPS_PETITE_CAPS:
        case FS_FONT_VARIANT_CAPS_ALL_PETITE_CAPS:
        case FS_FONT_VARIANT_CAPS_UNICASE:
        case FS_FONT_VARIANT_CAPS_TITLING_CAPS:
            st->style_font_variant_caps = (uint8_t)variant_caps;
            return true;
        default:
            return false;
    }
}

FS_FontVariantCaps fs_style_get_font_variant_caps(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_FONT_VARIANT_CAPS_NORMAL;
    }
    switch ((FS_FontVariantCaps)st->style_font_variant_caps) {
        case FS_FONT_VARIANT_CAPS_NORMAL:
        case FS_FONT_VARIANT_CAPS_SMALL_CAPS:
        case FS_FONT_VARIANT_CAPS_ALL_SMALL_CAPS:
        case FS_FONT_VARIANT_CAPS_PETITE_CAPS:
        case FS_FONT_VARIANT_CAPS_ALL_PETITE_CAPS:
        case FS_FONT_VARIANT_CAPS_UNICASE:
        case FS_FONT_VARIANT_CAPS_TITLING_CAPS:
            return (FS_FontVariantCaps)st->style_font_variant_caps;
        default:
            return FS_FONT_VARIANT_CAPS_NORMAL;
    }
}

bool fs_style_set_letter_spacing(FS_Core* core, float spacing_px) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(spacing_px)) {
        return false;
    }
    st->style_letter_spacing = spacing_px;
    return true;
}

float fs_style_get_letter_spacing(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_letter_spacing)) {
        return 0.0f;
    }
    return st->style_letter_spacing;
}

bool fs_style_set_word_spacing(FS_Core* core, float spacing_px) {
    FS_InternalState* st = fs_state(core);
    if (!st || !isfinite(spacing_px)) {
        return false;
    }
    st->style_word_spacing = spacing_px;
    return true;
}

float fs_style_get_word_spacing(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st || !isfinite(st->style_word_spacing)) {
        return 0.0f;
    }
    return st->style_word_spacing;
}

bool fs_style_set_image_smoothing_enabled(FS_Core* core, bool enabled) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->style_image_smoothing_enabled = enabled ? 1u : 0u;
    return true;
}

bool fs_style_get_image_smoothing_enabled(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return true;
    }
    return st->style_image_smoothing_enabled != 0u;
}

bool fs_style_set_image_smoothing_quality(FS_Core* core, FS_ImageSmoothingQuality quality) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    switch (quality) {
        case FS_IMAGE_SMOOTHING_QUALITY_LOW:
        case FS_IMAGE_SMOOTHING_QUALITY_MEDIUM:
        case FS_IMAGE_SMOOTHING_QUALITY_HIGH:
            st->style_image_smoothing_quality = (uint8_t)quality;
            return true;
        default:
            return false;
    }
}

FS_ImageSmoothingQuality fs_style_get_image_smoothing_quality(const FS_Core* core) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!st) {
        return FS_IMAGE_SMOOTHING_QUALITY_LOW;
    }
    switch ((FS_ImageSmoothingQuality)st->style_image_smoothing_quality) {
        case FS_IMAGE_SMOOTHING_QUALITY_LOW:
        case FS_IMAGE_SMOOTHING_QUALITY_MEDIUM:
        case FS_IMAGE_SMOOTHING_QUALITY_HIGH:
            return (FS_ImageSmoothingQuality)st->style_image_smoothing_quality;
        default:
            return FS_IMAGE_SMOOTHING_QUALITY_LOW;
    }
}

void fs_style_clear_dash(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return;
    }
    free(st->style_dash_segments);
    st->style_dash_segments = NULL;
    st->style_dash_count = 0u;
    st->style_dash_offset = 0.0f;
}

bool fs_style_set_dash(FS_Core* core, const float* segments, uint32_t segment_count, float offset) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (!segments || segment_count == 0u) {
        fs_style_clear_dash(core);
        st->style_dash_offset = offset;
        return true;
    }
    float* sanitized = (float*)malloc((size_t)segment_count * sizeof(float));
    if (!sanitized) {
        return false;
    }
    uint32_t count = 0u;
    for (uint32_t i = 0u; i < segment_count; ++i) {
        const float v = segments[i];
        if (v > 1e-6f) {
            sanitized[count++] = v;
        }
    }
    if (count == 0u) {
        free(sanitized);
        fs_style_clear_dash(core);
        st->style_dash_offset = offset;
        return true;
    }
    float* shrunk = (float*)realloc(sanitized, (size_t)count * sizeof(float));
    if (!shrunk) {
        shrunk = sanitized;
    }
    free(st->style_dash_segments);
    st->style_dash_segments = shrunk;
    st->style_dash_count = count;
    st->style_dash_offset = offset;
    return true;
}

bool fs_style_get_dash(
    const FS_Core* core,
    float* out_segments,
    uint32_t max_segments,
    uint32_t* out_count,
    float* out_offset
) {
    const FS_InternalState* st = (const FS_InternalState*)(core ? core->internal_state : NULL);
    if (!out_count) {
        return false;
    }
    if (!st) {
        *out_count = 0u;
        if (out_offset) {
            *out_offset = 0.0f;
        }
        return false;
    }
    const uint32_t count = st->style_dash_count;
    if (out_offset) {
        *out_offset = st->style_dash_offset;
    }
    *out_count = count;
    if (!out_segments || max_segments == 0u || count == 0u || !st->style_dash_segments) {
        return true;
    }
    const uint32_t copy_count = (count < max_segments) ? count : max_segments;
    memcpy(out_segments, st->style_dash_segments, (size_t)copy_count * sizeof(float));
    return true;
}

static bool fs_cmd_rect_compute_coverage_fill(
    FS_Core* core,
    float x,
    float y,
    float w,
    float h,
    float radius,
    uint32_t color,
    uint32_t extra_flags
) {
    if (!core) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || w == 0.0f || h == 0.0f) {
        return false;
    }
    FS_PathStateBorrow path_saved;
    fs_path_state_begin_temporary(st, &path_saved);
    const FS_Transform2D* t = &st->current_transform;
    float tx = x;
    float ty = y;
    float tw = w;
    float th = h;
    fs_transform_rect_to_aabb(t, x, y, w, h, &tx, &ty, &tw, &th);
    const float rr = fmaxf(0.0f, fminf(radius, fminf(fabsf(w), fabsf(h)) * 0.5f));

    fs_state_save(core);
    fs_path_begin(core);
    bool ok = (rr > 1e-5f) ? fs_path_round_rect(core, x, y, w, h, rr) : fs_path_rect(core, x, y, w, h);
    if (ok) {
        ok = fs_clip_path_with_fill_rule(core, FS_FILL_RULE_NONZERO);
    }
    if (ok) {
        const float pad = 1.5f;
        fs_transform_reset(core);
        ok = fs_cmd_rect_with_flags(
            core,
            tx - pad,
            ty - pad,
            tw + pad * 2.0f,
            th + pad * 2.0f,
            0.0f,
            color,
            extra_flags
        );
    }
    (void)fs_state_restore(core);
    fs_path_state_end_temporary(st, &path_saved);
    return ok;
}

static bool fs_cmd_rect_compute_coverage_stroke(
    FS_Core* core,
    float x,
    float y,
    float w,
    float h,
    float radius,
    float stroke_width,
    uint32_t color
) {
    if (!core || stroke_width <= 0.0f) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || w == 0.0f || h == 0.0f) {
        return false;
    }
    FS_PathStateBorrow path_saved;
    fs_path_state_begin_temporary(st, &path_saved);
    const FS_Transform2D* t = &st->current_transform;
    float tx = x;
    float ty = y;
    float tw = w;
    float th = h;
    fs_transform_rect_to_aabb(t, x, y, w, h, &tx, &ty, &tw, &th);

    const float abs_w = fabsf(w);
    const float abs_h = fabsf(h);
    const float outer_r = fmaxf(0.0f, fminf(radius, fminf(abs_w, abs_h) * 0.5f));
    const float sw = fmaxf(0.0f, stroke_width);
    const float inner_w = abs_w - sw * 2.0f;
    const float inner_h = abs_h - sw * 2.0f;
    const bool has_inner = inner_w > 1e-4f && inner_h > 1e-4f;

    fs_state_save(core);
    fs_path_begin(core);
    bool ok = (outer_r > 1e-5f) ? fs_path_round_rect(core, x, y, w, h, outer_r) : fs_path_rect(core, x, y, w, h);
    if (ok && has_inner) {
        const float sx = (w >= 0.0f) ? sw : -sw;
        const float sy = (h >= 0.0f) ? sw : -sw;
        const float ix = x + sx;
        const float iy = y + sy;
        const float iw = w - sx * 2.0f;
        const float ih = h - sy * 2.0f;
        const float inner_r = fmaxf(0.0f, outer_r - sw);
        ok = (inner_r > 1e-5f) ? fs_path_round_rect(core, ix, iy, iw, ih, inner_r) : fs_path_rect(core, ix, iy, iw, ih);
    }
    if (ok) {
        ok = fs_clip_path_with_fill_rule(core, has_inner ? FS_FILL_RULE_EVENODD : FS_FILL_RULE_NONZERO);
    }
    if (ok) {
        const float pad = 1.5f;
        fs_transform_reset(core);
        ok = fs_cmd_rect(core, tx - pad, ty - pad, tw + pad * 2.0f, th + pad * 2.0f, 0.0f, color);
    }
    (void)fs_state_restore(core);
    fs_path_state_end_temporary(st, &path_saved);
    return ok;
}

static bool fs_cmd_rect_with_flags(
    FS_Core* core,
    float x,
    float y,
    float w,
    float h,
    float radius,
    uint32_t color,
    uint32_t extra_flags
) {
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    if (core && st && t && fs_transform_requires_oriented_quad(t)) {
        return fs_cmd_rect_compute_coverage_fill(core, x, y, w, h, radius, color, extra_flags);
    }
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE | extra_flags;
    cmd.p0[0] = x;
    cmd.p0[1] = y;
    cmd.p0[2] = w;
    cmd.p0[3] = h;
    cmd.scalar = radius;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_RECT;
    return fs_push_command(core, &cmd);
}

bool fs_cmd_rect(FS_Core* core, float x, float y, float w, float h, float radius, uint32_t color) {
    return fs_cmd_rect_with_flags(core, x, y, w, h, radius, color, 0u);
}

bool fs_cmd_clear_rect(FS_Core* core, float x, float y, float w, float h) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    const float prev_alpha = st->style_global_alpha;
    const uint8_t prev_comp = st->style_composite_op;
    const uint32_t prev_shadow_color = st->style_shadow_color_rgba8;
    const float prev_shadow_blur = st->style_shadow_blur;
    const float prev_shadow_offset_x = st->style_shadow_offset_x;
    const float prev_shadow_offset_y = st->style_shadow_offset_y;

    // Canvas clearRect semantic: clear destination pixels independent from current paint color.
    st->style_global_alpha = 1.0f;
    st->style_composite_op = (uint8_t)FS_GLOBAL_COMPOSITE_COPY;
    st->style_shadow_color_rgba8 = 0u;
    st->style_shadow_blur = 0.0f;
    st->style_shadow_offset_x = 0.0f;
    st->style_shadow_offset_y = 0.0f;

    const bool ok = fs_cmd_rect(core, x, y, w, h, 0.0f, 0u);

    st->style_global_alpha = prev_alpha;
    st->style_composite_op = prev_comp;
    st->style_shadow_color_rgba8 = prev_shadow_color;
    st->style_shadow_blur = prev_shadow_blur;
    st->style_shadow_offset_x = prev_shadow_offset_x;
    st->style_shadow_offset_y = prev_shadow_offset_y;
    return ok;
}

bool fs_cmd_rect_stroke(FS_Core* core, float x, float y, float w, float h, float radius, float stroke_width, uint32_t color) {
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    if (core && st && t && fs_transform_requires_oriented_quad(t)) {
        return fs_cmd_rect_compute_coverage_stroke(core, x, y, w, h, radius, stroke_width, color);
    }
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = x;
    cmd.p0[1] = y;
    cmd.p0[2] = w;
    cmd.p0[3] = h;
    cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    cmd.p1[0] = radius;
    cmd.scalar = stroke_width;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_RECT_STROKE;
    return fs_push_command(core, &cmd);
}

bool fs_fill_rect(FS_Core* core, float x, float y, float w, float h, float radius) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    const bool fill_linear = st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT &&
                             st->style_fill_linear_gradient.stop_count >= 2u;
    const bool fill_radial = st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT &&
                             st->style_fill_radial_gradient.stop_count >= 2u;
    const bool fill_conic = st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_CONIC_GRADIENT &&
                            st->style_fill_conic_gradient.stop_count >= 2u;
    const bool fill_pattern = st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN &&
                              st->style_fill_pattern.handle.width > 0u &&
                              st->style_fill_pattern.handle.height > 0u;
    if (!fill_linear && !fill_radial && !fill_conic && !fill_pattern) {
        return fs_cmd_rect(core, x, y, w, h, radius, st->style_fill_color_rgba8);
    }
    if (!isfinite(radius) || radius <= 1e-6f) {
        if (fill_linear) {
            return fs_draw_linear_gradient_rect_cells(core, x, y, w, h, &st->style_fill_linear_gradient);
        }
        if (fill_radial) {
            return fs_draw_radial_gradient_rect_cells(core, x, y, w, h, &st->style_fill_radial_gradient);
        }
        if (fill_conic) {
            return fs_draw_conic_gradient_rect_cells(core, x, y, w, h, &st->style_fill_conic_gradient);
        }
        return fs_cmd_rect_with_flags(
            core,
            x,
            y,
            w,
            h,
            0.0f,
            0xFFFFFFFFu,
            FS_RENDER_FLAG_PATTERN_SHADE | FS_RENDER_FLAG_PATTERN_FILL_HINT
        );
    }

    FS_Path2D* clip_rr = fs_path2d_create();
    if (!clip_rr) {
        const float cx = x + w * 0.5f;
        const float cy = y + h * 0.5f;
        return fs_cmd_rect(core, x, y, w, h, radius, fs_style_resolve_fill_color_at(st, cx, cy));
    }
    fs_state_save(core);
    const bool clip_ok = fs_path2d_round_rect(clip_rr, x, y, w, h, radius) && fs_clip_path2d(core, clip_rr);
    bool draw_ok = false;
    if (clip_ok) {
        if (fill_linear) {
            draw_ok = fs_draw_linear_gradient_rect_cells(core, x, y, w, h, &st->style_fill_linear_gradient);
        } else if (fill_radial) {
            draw_ok = fs_draw_radial_gradient_rect_cells(core, x, y, w, h, &st->style_fill_radial_gradient);
        } else if (fill_conic) {
            draw_ok = fs_draw_conic_gradient_rect_cells(core, x, y, w, h, &st->style_fill_conic_gradient);
        } else {
            draw_ok = fs_cmd_rect_with_flags(
                core,
                x,
                y,
                w,
                h,
                0.0f,
                0xFFFFFFFFu,
                FS_RENDER_FLAG_PATTERN_SHADE | FS_RENDER_FLAG_PATTERN_FILL_HINT
            );
        }
    }
    const bool restore_ok = fs_state_restore(core);
    fs_path2d_destroy(clip_rr);
    return clip_ok && draw_ok && restore_ok;
}

bool fs_stroke_rect(FS_Core* core, float x, float y, float w, float h, float radius, float stroke_width) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    const float cx = x + w * 0.5f;
    const float cy = y + h * 0.5f;
    const uint32_t color = fs_style_resolve_stroke_color_at(st, cx, cy);
    return fs_cmd_rect_stroke(core, x, y, w, h, radius, stroke_width, color);
}

bool fs_cmd_image(FS_Core* core, float x, float y, float w, float h, float uv_x, float uv_y, float uv_w, float uv_h, uint32_t color) {
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = x;
    cmd.p0[1] = y;
    cmd.p0[2] = w;
    cmd.p0[3] = h;
    if (fs_transform_requires_oriented_quad(t)) {
        fs_command_set_oriented_quad_from_rect(&cmd, t, x, y, w, h);
        cmd.flags |= FS_RENDER_FLAG_ORIENTED_QUAD;
    } else {
        cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    }
    cmd.p1[0] = uv_x;
    cmd.p1[1] = uv_y;
    cmd.p1[2] = uv_w;
    cmd.p1[3] = uv_h;
    cmd.p2[0] = 0.0f;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_IMAGE;
    return fs_push_command(core, &cmd);
}

bool fs_cmd_image_handle(FS_Core* core, float x, float y, float w, float h, const FS_ImageHandle* handle, uint32_t color) {
    if (!core || !handle || handle->layer >= core->image_atlas_layers) {
        return false;
    }
    if (handle->generation != core->image_atlas_generation[handle->layer]) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = x;
    cmd.p0[1] = y;
    cmd.p0[2] = w;
    cmd.p0[3] = h;
    if (fs_transform_requires_oriented_quad(t)) {
        fs_command_set_oriented_quad_from_rect(&cmd, t, x, y, w, h);
        cmd.flags |= FS_RENDER_FLAG_ORIENTED_QUAD;
    } else {
        cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    }
    cmd.p1[0] = handle->uv_min[0];
    cmd.p1[1] = handle->uv_min[1];
    cmd.p1[2] = handle->uv_max[0] - handle->uv_min[0];
    cmd.p1[3] = handle->uv_max[1] - handle->uv_min[1];
    cmd.p2[0] = (float)handle->layer;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_IMAGE;
    return fs_push_command(core, &cmd);
}

static bool fs_push_text_command(
    FS_Core* core,
    const FS_GlyphEntry* glyph,
    float draw_x,
    float draw_y,
    float draw_w,
    float draw_h,
    uint32_t color,
    uint32_t extra_text_flags,
    float stroke_width
) {
    if (!core || !glyph || draw_w <= 0.0f || draw_h <= 0.0f) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    const float scale_y = (glyph->atlas_height > 0.0f) ? (draw_h / glyph->atlas_height) : 1.0f;
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = draw_x;
    cmd.p0[1] = draw_y;
    cmd.p0[2] = draw_w;
    cmd.p0[3] = draw_h;
    if (fs_transform_requires_oriented_quad(t)) {
        fs_command_set_oriented_quad_from_rect(&cmd, t, draw_x, draw_y, draw_w, draw_h);
        cmd.flags |= FS_RENDER_FLAG_ORIENTED_QUAD;
    } else {
        cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    }
    cmd.p1[0] = glyph->uv_min[0];
    cmd.p1[1] = glyph->uv_min[1];
    cmd.p1[2] = glyph->uv_max[0] - glyph->uv_min[0];
    cmd.p1[3] = glyph->uv_max[1] - glyph->uv_min[1];
    cmd.p2[0] = glyph->sdf_radius_px;
    cmd.p2[1] = glyph->sdf_onedge;
    cmd.p2[2] = glyph->sdf_pixel_dist_scale;
    cmd.p2[3] = stroke_width > 0.0f ? stroke_width : 0.0f;
    cmd.scalar = scale_y;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_TEXT;
    // Keep internal render bits (e.g. oriented-quad) and add text behavior flags.
    cmd.flags |= (glyph->text_flags | extra_text_flags);
    return fs_push_command(core, &cmd);
}

#define FS_TEXT_STYLE_COLOR_NONE 0u
#define FS_TEXT_STYLE_COLOR_FILL 1u
#define FS_TEXT_STYLE_COLOR_STROKE 2u

#define FS_TEXT_STYLE_COLOR_NONE 0u
#define FS_TEXT_STYLE_COLOR_FILL 1u
#define FS_TEXT_STYLE_COLOR_STROKE 2u

static uint32_t fs_resolve_text_draw_color(
    const FS_InternalState* st,
    const FS_GlyphEntry* glyph,
    uint32_t fallback_color,
    uint32_t style_color_mode,
    float draw_x,
    float draw_y,
    float draw_w,
    float draw_h
) {
    if (!st || style_color_mode == FS_TEXT_STYLE_COLOR_NONE) {
        return fallback_color;
    }
    if (glyph && (glyph->text_flags & FS_TEXT_FLAG_COLOR_GLYPH)) {
        return fallback_color;
    }
    const float cx = draw_x + draw_w * 0.5f;
    const float cy = draw_y + draw_h * 0.5f;
    if (style_color_mode == FS_TEXT_STYLE_COLOR_FILL) {
        return fs_style_resolve_fill_color_at(st, cx, cy);
    }
    if (style_color_mode == FS_TEXT_STYLE_COLOR_STROKE) {
        return fs_style_resolve_stroke_color_at(st, cx, cy);
    }
    return fallback_color;
}

bool fs_cmd_text_glyph(FS_Core* core, float x, float y, float w, float h, uint32_t codepoint, uint32_t color) {
    if (!core) {
        return false;
    }
    FS_GlyphEntry* glyph = NULL;
    float glyph_scale = 1.0f;
    if (!fs_find_or_create_glyph(core, codepoint, 0u, h > 0.0f ? h : 16.0f, &glyph, &glyph_scale) || !glyph) {
        return false;
    }

    const float default_w = glyph->atlas_width * glyph_scale;
    const float default_h = glyph->atlas_height * glyph_scale;
    const float draw_w = (w > 0.0f) ? w : default_w;
    const float draw_h = (h > 0.0f) ? h : default_h;
    if (draw_w <= 0.0f || draw_h <= 0.0f) {
        return true;
    }
    return fs_push_text_command(core, glyph, x, y, draw_w, draw_h, color, 0u, 0.0f);
}

static bool fs_cmd_text_utf8_internal(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    uint32_t color,
    float max_width,
    uint32_t extra_text_flags,
    float stroke_width,
    uint32_t style_color_mode
) {
    if (!core || !utf8 || font_size_px <= 0.0f) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* text_transform = st ? &st->current_transform : NULL;
    float line_advance = font_size_px * 1.25f;
    fs_resolve_text_vertical_metrics(st, font_size_px, NULL, NULL, &line_advance);
    const bool kerning_enabled = fs_is_text_kerning_enabled(st);
    const float stretch_x = fs_text_stretch_scale(st);
    const bool small_caps_enabled = fs_is_text_small_caps_enabled(st);
    // Pixel snapping is only stable under axis-aligned transforms.
    // Under rotation/shear/mirror it introduces visible per-glyph jitter.
    const bool allow_snap =
        !fs_is_text_geometric_precision(st) &&
        !fs_transform_requires_oriented_quad(text_transform);
    const bool snap_x = allow_snap;
    // Do not snap each glyph's Y independently; that causes visible baseline wobble
    // (e.g. "u/e", "C/L", digits drifting by ~1 px) due different bearings.
    // Keep baseline-aligned Y stable and only snap X.
    const bool snap_y = false;
    const float letter_spacing = (st && isfinite(st->style_letter_spacing)) ? st->style_letter_spacing : 0.0f;
    const float word_spacing = (st && isfinite(st->style_word_spacing)) ? st->style_word_spacing : 0.0f;
    const bool has_letter_spacing = fabsf(letter_spacing) > 1e-6f;
    const bool has_word_spacing = fabsf(word_spacing) > 1e-6f;
    const bool text_fill_pattern =
        st &&
        style_color_mode == FS_TEXT_STYLE_COLOR_FILL &&
        st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN &&
        st->style_fill_pattern.handle.width > 0u &&
        st->style_fill_pattern.handle.height > 0u;
    const bool text_stroke_pattern =
        st &&
        style_color_mode == FS_TEXT_STYLE_COLOR_STROKE &&
        st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN &&
        st->style_stroke_pattern.handle.width > 0u &&
        st->style_stroke_pattern.handle.height > 0u;
    const bool text_pattern_per_fragment = text_fill_pattern || text_stroke_pattern;
    const uint32_t text_pattern_flag = text_pattern_per_fragment ? FS_RENDER_FLAG_PATTERN_SHADE : 0u;
    const uint32_t pattern_base_color = 0xFFFFFFFFu;

    const bool can_shape =
        FS_ENABLE_EXPERIMENTAL_SHAPING &&
        st &&
        st->image_font_count == 0u &&
        st->font_count == 1u &&
        st->fonts[0] &&
        st->font_backend &&
        st->font_backend->shape_text_utf8 &&
        st->font_backend->free_shaped_text &&
        st->font_backend->get_glyph_sdf_by_index &&
        !has_word_spacing &&
        kerning_enabled &&
        fabsf(stretch_x - 1.0f) <= 1e-6f &&
        !small_caps_enabled;

    float text_origin_x = x;
    float text_baseline_y = baseline_y;
    if (st) {
        FS_TextMetrics anchor_metrics = {0};
        if (fs_measure_text_utf8(core, font_size_px, utf8, max_width, &anchor_metrics)) {
            text_origin_x += fs_text_align_offset(st, &anchor_metrics);
            text_baseline_y += fs_text_baseline_offset(st, &anchor_metrics, font_size_px);
        }
    }
    if (allow_snap) {
        text_baseline_y = floorf(text_baseline_y + 0.5f);
    }

    if (can_shape && strchr(utf8, '\n') == NULL && strchr(utf8, '\r') == NULL) {
        const float bake_px = fs_get_text_bake_px(font_size_px);
        const float text_scale = font_size_px / bake_px;
        FS_ShapedTextRun run = {0};
        if (st->font_backend->shape_text_utf8(st->fonts[0], utf8, bake_px, &run)) {
            float pen_x = text_origin_x;
            float pen_y = text_baseline_y;
            bool has_prev_glyph = false;
            for (uint32_t i = 0u; i < run.glyph_count; ++i) {
                const FS_ShapedGlyph* shaped = &run.glyphs[i];
                if (has_prev_glyph && has_letter_spacing) {
                    pen_x += letter_spacing;
                }
                FS_GlyphEntry* glyph = NULL;
                float glyph_scale = 1.0f;
                if (!fs_find_or_create_glyph(core, shaped->glyph_index, 1u, font_size_px, &glyph, &glyph_scale) || !glyph) {
                    pen_x += ((float)shaped->x_advance_26d6 / 64.0f) * text_scale;
                    pen_y -= ((float)shaped->y_advance_26d6 / 64.0f) * text_scale;
                    has_prev_glyph = true;
                    continue;
                }

                if (glyph->atlas_width > 0.0f && glyph->atlas_height > 0.0f) {
                    float draw_x =
                        pen_x +
                        ((float)shaped->x_offset_26d6 / 64.0f) * text_scale +
                        glyph->bearing_x * glyph_scale;
                    float draw_y =
                        pen_y -
                        ((float)shaped->y_offset_26d6 / 64.0f) * text_scale +
                        glyph->bearing_y * glyph_scale;
                    if (snap_x) {
                        draw_x = floorf(draw_x + 0.5f);
                    }
                    if (snap_y) {
                        draw_y = floorf(draw_y + 0.5f);
                    }
                    const float draw_w = glyph->atlas_width * glyph_scale;
                    const float draw_h = glyph->atlas_height * glyph_scale;
                    if (max_width <= 0.0f || (draw_x - text_origin_x) <= max_width) {
                        const uint32_t draw_color = fs_resolve_text_draw_color(
                            st,
                            glyph,
                            color,
                            style_color_mode,
                            draw_x,
                            draw_y,
                            draw_w,
                            draw_h
                        );
                        if (!fs_push_text_command(
                                core,
                                glyph,
                                draw_x,
                                draw_y,
                                draw_w,
                                draw_h,
                                draw_color,
                                extra_text_flags,
                                stroke_width
                            )) {
                            st->font_backend->free_shaped_text(&run);
                            return false;
                        }
                    } else {
                        st->font_backend->free_shaped_text(&run);
                        return true;
                    }
                }
                pen_x += ((float)shaped->x_advance_26d6 / 64.0f) * text_scale;
                pen_y -= ((float)shaped->y_advance_26d6 / 64.0f) * text_scale;
                has_prev_glyph = true;
            }
            st->font_backend->free_shaped_text(&run);
            return true;
        }
    }

    float pen_x = text_origin_x;
    float pen_y = text_baseline_y;
    const char* ptr = utf8;
    uint32_t prev_cp = 0u;
    bool has_prev_cp = false;
    uint8_t prev_font_slot = 0u;
    float prev_font_px = font_size_px;
    bool line_has_glyph = false;

    while (*ptr != '\0') {
        if (st && st->image_font_count > 0u) {
            uint32_t image_font_id = 0u;
            uint32_t image_glyph_id = 0u;
            size_t image_seq_bytes = 0u;
            if (fs_find_image_sequence_match(st, ptr, &image_font_id, &image_glyph_id, &image_seq_bytes) &&
                image_seq_bytes > 0u) {
                uint32_t image_key = 0u;
                bool has_key = fs_pack_image_glyph_key(image_font_id, image_glyph_id, &image_key);
                FS_GlyphEntry* image_glyph = NULL;
                float image_scale = 1.0f;
                bool loaded = has_key &&
                              fs_find_or_create_glyph(
                                  core,
                                  image_key,
                                  FS_IMAGE_FONT_KIND,
                                  font_size_px,
                                  &image_glyph,
                                  &image_scale
                              ) &&
                              image_glyph;
                if (loaded) {
                    if (line_has_glyph && has_letter_spacing) {
                        pen_x += letter_spacing;
                    }
                    if (max_width > 0.0f && (pen_x - text_origin_x) > max_width) {
                        break;
                    }
                    if (image_glyph->atlas_width > 0.0f && image_glyph->atlas_height > 0.0f) {
                        float draw_x = pen_x + image_glyph->bearing_x * image_scale * stretch_x;
                        float draw_y = pen_y + image_glyph->bearing_y * image_scale;
                        if (snap_x) {
                            draw_x = floorf(draw_x + 0.5f);
                        }
                        if (snap_y) {
                            draw_y = floorf(draw_y + 0.5f);
                        }
                        const float draw_w = image_glyph->atlas_width * image_scale * stretch_x;
                        const float draw_h = image_glyph->atlas_height * image_scale;
                        const uint32_t draw_color = fs_resolve_text_draw_color(
                            st,
                            image_glyph,
                            color,
                            style_color_mode,
                            draw_x,
                            draw_y,
                            draw_w,
                            draw_h
                        );
                        const bool color_glyph = (image_glyph->text_flags & FS_TEXT_FLAG_COLOR_GLYPH) != 0u;
                        const uint32_t final_color = (text_pattern_per_fragment && !color_glyph) ? pattern_base_color : draw_color;
                        uint32_t draw_text_flags = extra_text_flags;
                        if (text_pattern_per_fragment && !color_glyph) {
                            draw_text_flags |= text_pattern_flag;
                        }
                        if (!fs_push_text_command(
                                core,
                                image_glyph,
                                draw_x,
                                draw_y,
                                draw_w,
                                draw_h,
                                final_color,
                                draw_text_flags,
                                stroke_width
                            )) {
                            return false;
                        }
                    }
                    pen_x += image_glyph->advance * image_scale * stretch_x;
                    line_has_glyph = true;
                } else {
                    (void)fs_push_missing_image_glyph(st, image_font_id, image_glyph_id, ptr, image_seq_bytes);
                }
                ptr += image_seq_bytes;
                has_prev_cp = false;
                continue;
            }
        }

        uint32_t cp = fs_decode_utf8(&ptr);
        if (cp == 0u) {
            break;
        }
        if (cp == '\r') {
            continue;
        }
        if (cp == '\n') {
            pen_x = text_origin_x;
            pen_y += line_advance;
            has_prev_cp = false;
            line_has_glyph = false;
            continue;
        }

        uint32_t glyph_cp = cp;
        float variant_size_scale = 1.0f;
        fs_text_variant_map_codepoint(st, cp, &glyph_cp, &variant_size_scale);
        const float glyph_font_px = font_size_px * variant_size_scale;

        FS_GlyphEntry* glyph = NULL;
        float glyph_scale = 1.0f;
        if (!fs_find_or_create_glyph(core, glyph_cp, 0u, glyph_font_px, &glyph, &glyph_scale) || !glyph) {
            has_prev_cp = false;
            continue;
        }

        if (kerning_enabled && has_prev_cp && prev_font_slot == glyph->font_slot) {
            const float kern_font_px = (prev_font_px < glyph_font_px) ? prev_font_px : glyph_font_px;
            pen_x += fs_get_kerning_advance(core, prev_cp, glyph_cp, kern_font_px, glyph->font_slot) * stretch_x;
        }
        if (line_has_glyph && has_letter_spacing) {
            pen_x += letter_spacing;
        }

        if (max_width > 0.0f && (pen_x - text_origin_x) > max_width) {
            break;
        }

        if (glyph->atlas_width > 0.0f && glyph->atlas_height > 0.0f) {
            float draw_x = pen_x + glyph->bearing_x * glyph_scale * stretch_x;
            float draw_y = pen_y + glyph->bearing_y * glyph_scale;
            if (snap_x) {
                draw_x = floorf(draw_x + 0.5f);
            }
            if (snap_y) {
                draw_y = floorf(draw_y + 0.5f);
            }
            const float draw_w = glyph->atlas_width * glyph_scale * stretch_x;
            const float draw_h = glyph->atlas_height * glyph_scale;
            const uint32_t draw_color = fs_resolve_text_draw_color(
                st,
                glyph,
                color,
                style_color_mode,
                draw_x,
                draw_y,
                draw_w,
                draw_h
            );
            const bool color_glyph = (glyph->text_flags & FS_TEXT_FLAG_COLOR_GLYPH) != 0u;
            const uint32_t final_color = (text_pattern_per_fragment && !color_glyph) ? pattern_base_color : draw_color;
            uint32_t draw_text_flags = extra_text_flags;
            if (text_pattern_per_fragment && !color_glyph) {
                draw_text_flags |= text_pattern_flag;
            }
            if (!fs_push_text_command(
                    core,
                    glyph,
                    draw_x,
                    draw_y,
                    draw_w,
                    draw_h,
                    final_color,
                    draw_text_flags,
                    stroke_width
                )) {
                return false;
            }
        }
        pen_x += glyph->advance * glyph_scale * stretch_x;
        if (has_word_spacing && fs_is_word_spacing_codepoint(cp)) {
            pen_x += word_spacing;
        }
        prev_cp = glyph_cp;
        prev_font_slot = glyph->font_slot;
        prev_font_px = glyph_font_px;
        has_prev_cp = true;
        line_has_glyph = true;
    }
    return true;
}

bool fs_cmd_text_utf8(FS_Core* core, float x, float baseline_y, float font_size_px, const char* utf8, uint32_t color, float max_width) {
    return fs_cmd_text_utf8_internal(
        core,
        x,
        baseline_y,
        font_size_px,
        utf8,
        color,
        max_width,
        0u,
        0.0f,
        FS_TEXT_STYLE_COLOR_NONE
    );
}

bool fs_cmd_stroke_text_utf8(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    uint32_t color,
    float max_width,
    float stroke_width
) {
    if (!core || !utf8 || font_size_px <= 0.0f) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    const float resolved_width = fs_style_resolve_line_width(st, stroke_width);
    if (resolved_width <= 0.0f) {
        return false;
    }
    return fs_cmd_text_utf8_internal(
        core,
        x,
        baseline_y,
        font_size_px,
        utf8,
        color,
        max_width,
        FS_TEXT_FLAG_STROKE,
        resolved_width,
        FS_TEXT_STYLE_COLOR_NONE
    );
}

bool fs_fill_text_utf8(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    float max_width
) {
    FS_InternalState* st = fs_state(core);
    if (st && (
            (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT &&
             st->style_fill_linear_gradient.stop_count >= 2u) ||
            (st->style_fill_paint_type == (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT &&
             st->style_fill_radial_gradient.stop_count >= 2u))) {
        const uint32_t fallback_color = st->style_fill_color_rgba8;
        return fs_cmd_text_utf8_internal(
            core,
            x,
            baseline_y,
            font_size_px,
            utf8,
            fallback_color,
            max_width,
            0u,
            0.0f,
            FS_TEXT_STYLE_COLOR_FILL
        );
    }
    const uint32_t color = fs_style_resolve_fill_color_at(st, x, baseline_y);
    return fs_cmd_text_utf8(core, x, baseline_y, font_size_px, utf8, color, max_width);
}

bool fs_stroke_text_utf8(
    FS_Core* core,
    float x,
    float baseline_y,
    float font_size_px,
    const char* utf8,
    float max_width,
    float stroke_width
) {
    FS_InternalState* st = fs_state(core);
    const float resolved_width = fs_style_resolve_line_width(st, stroke_width);
    if (resolved_width <= 0.0f) {
        return false;
    }
    if (st && (
            (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT &&
             st->style_stroke_linear_gradient.stop_count >= 2u) ||
            (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT &&
             st->style_stroke_radial_gradient.stop_count >= 2u))) {
        const uint32_t fallback_color = st->style_stroke_color_rgba8;
        return fs_cmd_text_utf8_internal(
            core,
            x,
            baseline_y,
            font_size_px,
            utf8,
            fallback_color,
            max_width,
            FS_TEXT_FLAG_STROKE,
            resolved_width,
            FS_TEXT_STYLE_COLOR_STROKE
        );
    }
    const uint32_t color = fs_style_resolve_stroke_color_at(st, x, baseline_y);
    return fs_cmd_stroke_text_utf8(
        core,
        x,
        baseline_y,
        font_size_px,
        utf8,
        color,
        max_width,
        resolved_width
    );
}

bool fs_measure_text_utf8(
    FS_Core* core,
    float font_size_px,
    const char* utf8,
    float max_width,
    FS_TextMetrics* out_metrics
) {
    if (!core || !utf8 || font_size_px <= 0.0f || !out_metrics) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    memset(out_metrics, 0, sizeof(*out_metrics));

    float em_ascent = 0.0f;
    float em_descent = 0.0f;
    float line_advance = font_size_px * 1.25f;
    fs_resolve_text_vertical_metrics(st, font_size_px, &em_ascent, &em_descent, &line_advance);

    float pen_x = 0.0f;
    float pen_y = 0.0f;
    float line_start_x = 0.0f;
    float max_line_width = 0.0f;
    uint32_t line_count = 1u;
    uint32_t glyph_count = 0u;

    bool has_bounds = false;
    float min_x = 0.0f;
    float min_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
    const float letter_spacing = (isfinite(st->style_letter_spacing)) ? st->style_letter_spacing : 0.0f;
    const float word_spacing = (isfinite(st->style_word_spacing)) ? st->style_word_spacing : 0.0f;
    const bool has_letter_spacing = fabsf(letter_spacing) > 1e-6f;
    const bool has_word_spacing = fabsf(word_spacing) > 1e-6f;
    const bool kerning_enabled = fs_is_text_kerning_enabled(st);
    const float stretch_x = fs_text_stretch_scale(st);

    const bool snap_x = !fs_is_text_geometric_precision(st);
    const bool snap_y = false;
    const char* ptr = utf8;
    uint32_t prev_cp = 0u;
    bool has_prev_cp = false;
    uint8_t prev_font_slot = 0u;
    float prev_font_px = font_size_px;
    bool line_has_glyph = false;

    while (*ptr != '\0') {
        if (st->image_font_count > 0u) {
            uint32_t image_font_id = 0u;
            uint32_t image_glyph_id = 0u;
            size_t image_seq_bytes = 0u;
            if (fs_find_image_sequence_match(st, ptr, &image_font_id, &image_glyph_id, &image_seq_bytes) &&
                image_seq_bytes > 0u) {
                uint32_t image_key = 0u;
                bool has_key = fs_pack_image_glyph_key(image_font_id, image_glyph_id, &image_key);
                FS_GlyphEntry* image_glyph = NULL;
                float image_scale = 1.0f;
                bool loaded = has_key &&
                              fs_find_or_create_glyph(
                                  core,
                                  image_key,
                                  FS_IMAGE_FONT_KIND,
                                  font_size_px,
                                  &image_glyph,
                                  &image_scale
                              ) &&
                              image_glyph;
                if (loaded) {
                    if (line_has_glyph && has_letter_spacing) {
                        pen_x += letter_spacing;
                    }
                    if (max_width > 0.0f && (pen_x - line_start_x) > max_width) {
                        break;
                    }
                    if (image_glyph->atlas_width > 0.0f && image_glyph->atlas_height > 0.0f) {
                        float draw_x = pen_x + image_glyph->bearing_x * image_scale * stretch_x;
                        float draw_y = pen_y + image_glyph->bearing_y * image_scale;
                        if (snap_x) {
                            draw_x = floorf(draw_x + 0.5f);
                        }
                        if (snap_y) {
                            draw_y = floorf(draw_y + 0.5f);
                        }
                        const float draw_w = image_glyph->atlas_width * image_scale * stretch_x;
                        const float draw_h = image_glyph->atlas_height * image_scale;
                        if (draw_w > 0.0f && draw_h > 0.0f) {
                            const float bx0 = draw_x;
                            const float by0 = draw_y;
                            const float bx1 = draw_x + draw_w;
                            const float by1 = draw_y + draw_h;
                            if (!has_bounds) {
                                min_x = bx0;
                                min_y = by0;
                                max_x = bx1;
                                max_y = by1;
                                has_bounds = true;
                            } else {
                                if (bx0 < min_x) {
                                    min_x = bx0;
                                }
                                if (by0 < min_y) {
                                    min_y = by0;
                                }
                                if (bx1 > max_x) {
                                    max_x = bx1;
                                }
                                if (by1 > max_y) {
                                    max_y = by1;
                                }
                            }
                            glyph_count += 1u;
                        }
                    }
                    pen_x += image_glyph->advance * image_scale * stretch_x;
                    const float line_width = pen_x - line_start_x;
                    if (line_width > max_line_width) {
                        max_line_width = line_width;
                    }
                    line_has_glyph = true;
                }
                ptr += image_seq_bytes;
                has_prev_cp = false;
                continue;
            }
        }

        uint32_t cp = fs_decode_utf8(&ptr);
        if (cp == 0u) {
            break;
        }
        if (cp == '\r') {
            continue;
        }
        if (cp == '\n') {
            const float line_width = pen_x - line_start_x;
            if (line_width > max_line_width) {
                max_line_width = line_width;
            }
            pen_x = 0.0f;
            pen_y += line_advance;
            line_start_x = pen_x;
            has_prev_cp = false;
            line_has_glyph = false;
            line_count += 1u;
            continue;
        }

        uint32_t glyph_cp = cp;
        float variant_size_scale = 1.0f;
        fs_text_variant_map_codepoint(st, cp, &glyph_cp, &variant_size_scale);
        const float glyph_font_px = font_size_px * variant_size_scale;

        FS_GlyphEntry* glyph = NULL;
        float glyph_scale = 1.0f;
        if (!fs_find_or_create_glyph(core, glyph_cp, 0u, glyph_font_px, &glyph, &glyph_scale) || !glyph) {
            has_prev_cp = false;
            continue;
        }

        if (kerning_enabled && has_prev_cp && prev_font_slot == glyph->font_slot) {
            const float kern_font_px = (prev_font_px < glyph_font_px) ? prev_font_px : glyph_font_px;
            pen_x += fs_get_kerning_advance(core, prev_cp, glyph_cp, kern_font_px, glyph->font_slot) * stretch_x;
        }
        if (line_has_glyph && has_letter_spacing) {
            pen_x += letter_spacing;
        }

        if (max_width > 0.0f && (pen_x - line_start_x) > max_width) {
            break;
        }

        if (glyph->atlas_width > 0.0f && glyph->atlas_height > 0.0f) {
            float draw_x = pen_x + glyph->bearing_x * glyph_scale * stretch_x;
            float draw_y = pen_y + glyph->bearing_y * glyph_scale;
            if (snap_x) {
                draw_x = floorf(draw_x + 0.5f);
            }
            if (snap_y) {
                draw_y = floorf(draw_y + 0.5f);
            }
            const float draw_w = glyph->atlas_width * glyph_scale * stretch_x;
            const float draw_h = glyph->atlas_height * glyph_scale;
            const float bx0 = draw_x;
            const float by0 = draw_y;
            const float bx1 = draw_x + draw_w;
            const float by1 = draw_y + draw_h;
            if (!has_bounds) {
                min_x = bx0;
                min_y = by0;
                max_x = bx1;
                max_y = by1;
                has_bounds = true;
            } else {
                if (bx0 < min_x) {
                    min_x = bx0;
                }
                if (by0 < min_y) {
                    min_y = by0;
                }
                if (bx1 > max_x) {
                    max_x = bx1;
                }
                if (by1 > max_y) {
                    max_y = by1;
                }
            }
            glyph_count += 1u;
        }

        pen_x += glyph->advance * glyph_scale * stretch_x;
        if (has_word_spacing && fs_is_word_spacing_codepoint(cp)) {
            pen_x += word_spacing;
        }
        const float line_width = pen_x - line_start_x;
        if (line_width > max_line_width) {
            max_line_width = line_width;
        }
        prev_cp = glyph_cp;
        prev_font_slot = glyph->font_slot;
        prev_font_px = glyph_font_px;
        has_prev_cp = true;
        line_has_glyph = true;
    }

    out_metrics->width = max_line_width;
    out_metrics->glyph_count = glyph_count;
    out_metrics->line_count = line_count;
    out_metrics->em_height_ascent = em_ascent;
    out_metrics->em_height_descent = em_descent;
    if (has_bounds) {
        out_metrics->actual_bounding_box_left = -min_x;
        out_metrics->actual_bounding_box_right = max_x;
        out_metrics->actual_bounding_box_ascent = -min_y;
        out_metrics->actual_bounding_box_descent = max_y;
    } else {
        out_metrics->actual_bounding_box_left = 0.0f;
        out_metrics->actual_bounding_box_right = 0.0f;
        out_metrics->actual_bounding_box_ascent = 0.0f;
        out_metrics->actual_bounding_box_descent = 0.0f;
    }
    return true;
}

static bool fs_path_append_arc_cubic(FS_Core* core, float cx, float cy, float radius, float a0, float a1) {
    if (!core || radius <= 0.0f) {
        return false;
    }
    const float delta = a1 - a0;
    if (fabsf(delta) <= 1e-7f) {
        return true;
    }
    const float k = (4.0f / 3.0f) * tanf(delta * 0.25f);

    const float c0 = cosf(a0);
    const float s0 = sinf(a0);
    const float c1 = cosf(a1);
    const float s1 = sinf(a1);

    const float p0x = cx + c0 * radius;
    const float p0y = cy + s0 * radius;
    const float p3x = cx + c1 * radius;
    const float p3y = cy + s1 * radius;

    const float t0x = -s0;
    const float t0y = c0;
    const float t1x = -s1;
    const float t1y = c1;

    const float cp0x = p0x + t0x * (k * radius);
    const float cp0y = p0y + t0y * (k * radius);
    const float cp1x = p3x - t1x * (k * radius);
    const float cp1y = p3y - t1y * (k * radius);

    return fs_path_bezier_curve_to(core, cp0x, cp0y, cp1x, cp1y, p3x, p3y);
}

static bool fs_path_append_arc_sweep(FS_Core* core, float cx, float cy, float radius, float a0, float a1) {
    const float pi = 3.14159265358979323846f;
    const float sweep = a1 - a0;
    const float abs_sweep = fabsf(sweep);
    if (abs_sweep <= 1e-7f) {
        return true;
    }
    uint32_t segment_count = (uint32_t)ceilf(abs_sweep / (pi * 0.5f));
    if (segment_count < 1u) {
        segment_count = 1u;
    } else if (segment_count > 64u) {
        segment_count = 64u;
    }
    const float step = sweep / (float)segment_count;
    float a = a0;
    for (uint32_t i = 0u; i < segment_count; ++i) {
        const float b = a + step;
        if (!fs_path_append_arc_cubic(core, cx, cy, radius, a, b)) {
            return false;
        }
        a = b;
    }
    return true;
}



static bool fs_path2d_append_arc_cubic(FS_Path2D* path, float cx, float cy, float radius, float a0, float a1) {
    if (!path || radius <= 0.0f) {
        return false;
    }
    const float delta = a1 - a0;
    if (fabsf(delta) <= 1e-7f) {
        return true;
    }
    const float k = (4.0f / 3.0f) * tanf(delta * 0.25f);

    const float c0 = cosf(a0);
    const float s0 = sinf(a0);
    const float c1 = cosf(a1);
    const float s1 = sinf(a1);

    const float p0x = cx + c0 * radius;
    const float p0y = cy + s0 * radius;
    const float p3x = cx + c1 * radius;
    const float p3y = cy + s1 * radius;

    const float t0x = -s0;
    const float t0y = c0;
    const float t1x = -s1;
    const float t1y = c1;

    const float cp0x = p0x + t0x * (k * radius);
    const float cp0y = p0y + t0y * (k * radius);
    const float cp1x = p3x - t1x * (k * radius);
    const float cp1y = p3y - t1y * (k * radius);

    return fs_path2d_bezier_curve_to(path, cp0x, cp0y, cp1x, cp1y, p3x, p3y);
}

static bool fs_path2d_append_arc_sweep(FS_Path2D* path, float cx, float cy, float radius, float a0, float a1) {
    const float pi = 3.14159265358979323846f;
    const float sweep = a1 - a0;
    const float abs_sweep = fabsf(sweep);
    if (abs_sweep <= 1e-7f) {
        return true;
    }
    uint32_t segment_count = (uint32_t)ceilf(abs_sweep / (pi * 0.5f));
    if (segment_count < 1u) {
        segment_count = 1u;
    } else if (segment_count > 64u) {
        segment_count = 64u;
    }
    const float step = sweep / (float)segment_count;
    float a = a0;
    for (uint32_t i = 0u; i < segment_count; ++i) {
        const float b = a + step;
        if (!fs_path2d_append_arc_cubic(path, cx, cy, radius, a, b)) {
            return false;
        }
        a = b;
    }
    return true;
}

static float fs_arc_resolve_delta(float start_angle, float end_angle, bool counterclockwise) {
    const float pi = 3.14159265358979323846f;
    const float tau = 2.0f * pi;
    const float raw = end_angle - start_angle;
    if (fabsf(raw) >= tau) {
        return counterclockwise ? -tau : tau;
    }
    float delta = raw;
    if (!counterclockwise) {
        while (delta < 0.0f) {
            delta += tau;
        }
        while (delta > tau) {
            delta -= tau;
        }
    } else {
        while (delta > 0.0f) {
            delta -= tau;
        }
        while (delta < -tau) {
            delta += tau;
        }
    }
    return delta;
}

static bool fs_path_append_ellipse_arc_sweep(
    FS_Core* core,
    float cx,
    float cy,
    float rx,
    float ry,
    float rotation,
    float a0,
    float a1
) {
    const float pi = 3.14159265358979323846f;
    if (!core || rx <= 0.0f || ry <= 0.0f) {
        return false;
    }
    const float sweep = a1 - a0;
    const float abs_sweep = fabsf(sweep);
    if (abs_sweep <= 1e-7f) {
        return true;
    }
    uint32_t segment_count = (uint32_t)ceilf(abs_sweep / (pi * 0.5f));
    if (segment_count < 1u) {
        segment_count = 1u;
    } else if (segment_count > 64u) {
        segment_count = 64u;
    }
    const float step = sweep / (float)segment_count;
    const float cr = cosf(rotation);
    const float sr = sinf(rotation);
    float a = a0;
    for (uint32_t i = 0u; i < segment_count; ++i) {
        const float b = a + step;
        const float delta = b - a;
        const float k = (4.0f / 3.0f) * tanf(delta * 0.25f);

        const float c0 = cosf(a);
        const float s0 = sinf(a);
        const float c1 = cosf(b);
        const float s1 = sinf(b);

        const float p3x0 = rx * c1;
        const float p3y0 = ry * s1;
        const float d0x = -rx * s0;
        const float d0y = ry * c0;
        const float d1x = -rx * s1;
        const float d1y = ry * c1;

        const float cp0x0 = rx * c0 + d0x * k;
        const float cp0y0 = ry * s0 + d0y * k;
        const float cp1x0 = p3x0 - d1x * k;
        const float cp1y0 = p3y0 - d1y * k;

        const float cp0x = cx + cp0x0 * cr - cp0y0 * sr;
        const float cp0y = cy + cp0x0 * sr + cp0y0 * cr;
        const float cp1x = cx + cp1x0 * cr - cp1y0 * sr;
        const float cp1y = cy + cp1x0 * sr + cp1y0 * cr;
        const float p3x = cx + p3x0 * cr - p3y0 * sr;
        const float p3y = cy + p3x0 * sr + p3y0 * cr;

        if (!fs_path_bezier_curve_to(core, cp0x, cp0y, cp1x, cp1y, p3x, p3y)) {
            return false;
        }
        a = b;
    }
    return true;
}

static bool fs_path2d_append_ellipse_arc_sweep(
    FS_Path2D* path,
    float cx,
    float cy,
    float rx,
    float ry,
    float rotation,
    float a0,
    float a1
) {
    const float pi = 3.14159265358979323846f;
    if (!path || rx <= 0.0f || ry <= 0.0f) {
        return false;
    }
    const float sweep = a1 - a0;
    const float abs_sweep = fabsf(sweep);
    if (abs_sweep <= 1e-7f) {
        return true;
    }
    uint32_t segment_count = (uint32_t)ceilf(abs_sweep / (pi * 0.5f));
    if (segment_count < 1u) {
        segment_count = 1u;
    } else if (segment_count > 64u) {
        segment_count = 64u;
    }
    const float step = sweep / (float)segment_count;
    const float cr = cosf(rotation);
    const float sr = sinf(rotation);
    float a = a0;
    for (uint32_t i = 0u; i < segment_count; ++i) {
        const float b = a + step;
        const float delta = b - a;
        const float k = (4.0f / 3.0f) * tanf(delta * 0.25f);

        const float c0 = cosf(a);
        const float s0 = sinf(a);
        const float c1 = cosf(b);
        const float s1 = sinf(b);

        const float p3x0 = rx * c1;
        const float p3y0 = ry * s1;
        const float d0x = -rx * s0;
        const float d0y = ry * c0;
        const float d1x = -rx * s1;
        const float d1y = ry * c1;

        const float cp0x0 = rx * c0 + d0x * k;
        const float cp0y0 = ry * s0 + d0y * k;
        const float cp1x0 = p3x0 - d1x * k;
        const float cp1y0 = p3y0 - d1y * k;

        const float cp0x = cx + cp0x0 * cr - cp0y0 * sr;
        const float cp0y = cy + cp0x0 * sr + cp0y0 * cr;
        const float cp1x = cx + cp1x0 * cr - cp1y0 * sr;
        const float cp1y = cy + cp1x0 * sr + cp1y0 * cr;
        const float p3x = cx + p3x0 * cr - p3y0 * sr;
        const float p3y = cy + p3x0 * sr + p3y0 * cr;

        if (!fs_path2d_bezier_curve_to(path, cp0x, cp0y, cp1x, cp1y, p3x, p3y)) {
            return false;
        }
        a = b;
    }
    return true;
}

void fs_path_begin(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return;
    }
    st->path_count = 0u;
    st->path_has_current = false;
    st->path_has_subpath_start = false;
    st->path_current_x = 0.0f;
    st->path_current_y = 0.0f;
    st->path_subpath_start_x = 0.0f;
    st->path_subpath_start_y = 0.0f;
}

bool fs_path_move_to(FS_Core* core, float x, float y) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    st->path_has_current = true;
    st->path_has_subpath_start = true;
    st->path_current_x = x;
    st->path_current_y = y;
    st->path_subpath_start_x = x;
    st->path_subpath_start_y = y;
    return true;
}

bool fs_path_line_to(FS_Core* core, float x, float y) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (!st->path_has_current) {
        return fs_path_move_to(core, x, y);
    }
    FS_PathSegment seg;
    memset(&seg, 0, sizeof(seg));
    seg.type = (uint8_t)FS_PATH_SEG_LINE;
    seg.x0 = st->path_current_x;
    seg.y0 = st->path_current_y;
    seg.x1 = x;
    seg.y1 = y;
    if (!fs_append_path_segment(st, &seg)) {
        return false;
    }
    st->path_current_x = x;
    st->path_current_y = y;
    return true;
}

bool fs_path_quadratic_curve_to(FS_Core* core, float cx, float cy, float x, float y) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (!st->path_has_current) {
        return fs_path_move_to(core, x, y);
    }
    FS_PathSegment seg;
    memset(&seg, 0, sizeof(seg));
    seg.type = (uint8_t)FS_PATH_SEG_QUAD;
    seg.x0 = st->path_current_x;
    seg.y0 = st->path_current_y;
    seg.cx0 = cx;
    seg.cy0 = cy;
    seg.x1 = x;
    seg.y1 = y;
    if (!fs_append_path_segment(st, &seg)) {
        return false;
    }
    st->path_current_x = x;
    st->path_current_y = y;
    return true;
}

bool fs_path_bezier_curve_to(FS_Core* core, float cx0, float cy0, float cx1, float cy1, float x, float y) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (!st->path_has_current) {
        return fs_path_move_to(core, x, y);
    }
    FS_PathSegment seg;
    memset(&seg, 0, sizeof(seg));
    seg.type = (uint8_t)FS_PATH_SEG_CUBIC;
    seg.x0 = st->path_current_x;
    seg.y0 = st->path_current_y;
    seg.cx0 = cx0;
    seg.cy0 = cy0;
    seg.cx1 = cx1;
    seg.cy1 = cy1;
    seg.x1 = x;
    seg.y1 = y;
    if (!fs_append_path_segment(st, &seg)) {
        return false;
    }
    st->path_current_x = x;
    st->path_current_y = y;
    return true;
}

bool fs_path_arc(FS_Core* core, float cx, float cy, float radius, float start_angle, float end_angle, bool counterclockwise) {
    return fs_path_ellipse(core, cx, cy, radius, radius, 0.0f, start_angle, end_angle, counterclockwise);
}

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
) {
    FS_InternalState* st = fs_state(core);
    if (!st ||
        !isfinite(cx) || !isfinite(cy) ||
        !isfinite(radius_x) || !isfinite(radius_y) ||
        !isfinite(rotation) || !isfinite(start_angle) || !isfinite(end_angle) ||
        radius_x < 0.0f || radius_y < 0.0f) {
        return false;
    }

    const float cr = cosf(rotation);
    const float sr = sinf(rotation);
    const float cs = cosf(start_angle);
    const float ss = sinf(start_angle);
    const float sx = cx + (radius_x * cs) * cr - (radius_y * ss) * sr;
    const float sy = cy + (radius_x * cs) * sr + (radius_y * ss) * cr;

    if (!st->path_has_current) {
        if (!fs_path_move_to(core, sx, sy)) {
            return false;
        }
    } else {
        const float dx = st->path_current_x - sx;
        const float dy = st->path_current_y - sy;
        if (fabsf(dx) > 1e-6f || fabsf(dy) > 1e-6f) {
            if (!fs_path_line_to(core, sx, sy)) {
                return false;
            }
        }
    }

    const float delta = fs_arc_resolve_delta(start_angle, end_angle, counterclockwise);
    if (fabsf(delta) <= 1e-7f || radius_x <= 1e-7f || radius_y <= 1e-7f) {
        return true;
    }
    return fs_path_append_ellipse_arc_sweep(
        core,
        cx,
        cy,
        radius_x,
        radius_y,
        rotation,
        start_angle,
        start_angle + delta
    );
}

bool fs_path_arc_to(FS_Core* core, float x1, float y1, float x2, float y2, float radius) {
    const float pi = 3.14159265358979323846f;
    FS_InternalState* st = fs_state(core);
    if (!st || radius < 0.0f) {
        return false;
    }
    if (!st->path_has_current) {
        return fs_path_move_to(core, x1, y1);
    }

    const float x0 = st->path_current_x;
    const float y0 = st->path_current_y;
    const float dx01 = x0 - x1;
    const float dy01 = y0 - y1;
    const float dx21 = x2 - x1;
    const float dy21 = y2 - y1;
    const float len01 = hypotf(dx01, dy01);
    const float len21 = hypotf(dx21, dy21);
    if (radius <= 1e-6f || len01 <= 1e-6f || len21 <= 1e-6f) {
        return fs_path_line_to(core, x1, y1);
    }

    const float u1x = dx01 / len01;
    const float u1y = dy01 / len01;
    const float u2x = dx21 / len21;
    const float u2y = dy21 / len21;
    float dot = u1x * u2x + u1y * u2y;
    if (dot > 1.0f) dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;
    const float cross = u1x * u2y - u1y * u2x;
    const float angle = acosf(dot);
    if (fabsf(cross) <= 1e-7f || angle <= 1e-5f || fabsf(pi - angle) <= 1e-5f) {
        return fs_path_line_to(core, x1, y1);
    }

    float t = radius / tanf(angle * 0.5f);
    if (!isfinite(t) || t <= 1e-6f) {
        return fs_path_line_to(core, x1, y1);
    }
    const float max_t = fminf(len01, len21) - 1e-4f;
    if (max_t <= 1e-6f) {
        return fs_path_line_to(core, x1, y1);
    }
    if (t > max_t) {
        t = max_t;
    }

    const float tx1 = x1 + u1x * t;
    const float ty1 = y1 + u1y * t;
    const float tx2 = x1 + u2x * t;
    const float ty2 = y1 + u2y * t;

    if (!fs_path_line_to(core, tx1, ty1)) {
        return false;
    }

    const float bisx = u1x + u2x;
    const float bisy = u1y + u2y;
    const float bis_len = hypotf(bisx, bisy);
    if (bis_len <= 1e-6f) {
        return fs_path_line_to(core, tx2, ty2);
    }
    const float inv_bis = 1.0f / bis_len;
    const float bx = bisx * inv_bis;
    const float by = bisy * inv_bis;
    const float center_dist = radius / sinf(angle * 0.5f);
    const float cx = x1 + bx * center_dist;
    const float cy = y1 + by * center_dist;

    float a0 = atan2f(ty1 - cy, tx1 - cx);
    float a1 = atan2f(ty2 - cy, tx2 - cx);
    const bool ccw = cross < 0.0f;
    if (ccw) {
        while (a1 <= a0) {
            a1 += 2.0f * pi;
        }
    } else {
        while (a1 >= a0) {
            a1 -= 2.0f * pi;
        }
    }

    return fs_path_append_arc_sweep(core, cx, cy, radius, a0, a1);
}

bool fs_path_rect(FS_Core* core, float x, float y, float w, float h) {
    if (!core) {
        return false;
    }
    const float x0 = x;
    const float y0 = y;
    const float x1 = x + w;
    const float y1 = y + h;
    if (!fs_path_move_to(core, x0, y0)) {
        return false;
    }
    if (!fs_path_line_to(core, x1, y0)) {
        return false;
    }
    if (!fs_path_line_to(core, x1, y1)) {
        return false;
    }
    if (!fs_path_line_to(core, x0, y1)) {
        return false;
    }
    return fs_path_close(core);
}

bool fs_path_round_rect(FS_Core* core, float x, float y, float w, float h, float radius) {
    const float pi = 3.14159265358979323846f;
    if (!core) {
        return false;
    }
    if (fabsf(w) <= 1e-6f || fabsf(h) <= 1e-6f) {
        return fs_path_rect(core, x, y, w, h);
    }

    const float left = fminf(x, x + w);
    const float right = fmaxf(x, x + w);
    const float top = fminf(y, y + h);
    const float bottom = fmaxf(y, y + h);
    const float width = right - left;
    const float height = bottom - top;

    float r = radius;
    if (r < 0.0f) {
        r = 0.0f;
    }
    const float max_r = fminf(width, height) * 0.5f;
    if (r > max_r) {
        r = max_r;
    }
    if (r <= 1e-6f) {
        return fs_path_rect(core, x, y, w, h);
    }

    if (!fs_path_move_to(core, left + r, top)) {
        return false;
    }
    if (!fs_path_line_to(core, right - r, top)) {
        return false;
    }
    if (!fs_path_append_arc_sweep(core, right - r, top + r, r, -0.5f * pi, 0.0f)) {
        return false;
    }
    if (!fs_path_line_to(core, right, bottom - r)) {
        return false;
    }
    if (!fs_path_append_arc_sweep(core, right - r, bottom - r, r, 0.0f, 0.5f * pi)) {
        return false;
    }
    if (!fs_path_line_to(core, left + r, bottom)) {
        return false;
    }
    if (!fs_path_append_arc_sweep(core, left + r, bottom - r, r, 0.5f * pi, pi)) {
        return false;
    }
    if (!fs_path_line_to(core, left, top + r)) {
        return false;
    }
    if (!fs_path_append_arc_sweep(core, left + r, top + r, r, pi, 1.5f * pi)) {
        return false;
    }
    return fs_path_close(core);
}

bool fs_path_close(FS_Core* core) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (!st->path_has_current || !st->path_has_subpath_start) {
        return true;
    }
    const float dx = st->path_current_x - st->path_subpath_start_x;
    const float dy = st->path_current_y - st->path_subpath_start_y;
    if (fabsf(dx) <= 1e-6f && fabsf(dy) <= 1e-6f) {
        return true;
    }
    return fs_path_line_to(core, st->path_subpath_start_x, st->path_subpath_start_y);
}

static void fs_path_segment_sample_point(const FS_PathSegment* seg, float* out_x, float* out_y) {
    if (!seg || !out_x || !out_y) {
        return;
    }
    if (seg->type == (uint8_t)FS_PATH_SEG_LINE) {
        *out_x = (seg->x0 + seg->x1) * 0.5f;
        *out_y = (seg->y0 + seg->y1) * 0.5f;
        return;
    }
    if (seg->type == (uint8_t)FS_PATH_SEG_QUAD) {
        fs_eval_quad_point(seg->x0, seg->y0, seg->cx0, seg->cy0, seg->x1, seg->y1, 0.5f, out_x, out_y);
        return;
    }
    if (seg->type == (uint8_t)FS_PATH_SEG_CUBIC) {
        fs_eval_cubic_point(
            seg->x0,
            seg->y0,
            seg->cx0,
            seg->cy0,
            seg->cx1,
            seg->cy1,
            seg->x1,
            seg->y1,
            0.5f,
            out_x,
            out_y
        );
        return;
    }
    *out_x = seg->x0;
    *out_y = seg->y0;
}

static bool fs_path_stroke_internal(FS_Core* core, float width, uint32_t color, bool dynamic_stroke_style_color) {
    if (!core) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    const float stroke_width = fs_style_resolve_line_width(st, width);
    if (stroke_width <= 0.0f) {
        return false;
    }
    if (st->path_count == 0u) {
        return true;
    }
    const uint8_t line_cap = st->style_line_cap;
    const uint8_t line_join = st->style_line_join;
    const bool oriented_transform = fs_transform_requires_oriented_quad(&st->current_transform);
    const bool stroke_pattern_style =
        st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN &&
        st->style_stroke_pattern.handle.width > 0u &&
        st->style_stroke_pattern.handle.height > 0u;
    const bool stroke_pattern_per_fragment = dynamic_stroke_style_color && stroke_pattern_style;
    const uint32_t stroke_pattern_flag = stroke_pattern_per_fragment ? FS_RENDER_FLAG_PATTERN_SHADE : 0u;
    const uint32_t pattern_base_color = 0xFFFFFFFFu;
    bool use_dash = fs_style_has_dash(st);
    float dash_total = 0.0f;
    float dash_phase = st->style_dash_offset;
    if (use_dash) {
        for (uint32_t i = 0u; i < st->style_dash_count; ++i) {
            dash_total += st->style_dash_segments[i];
        }
        if (dash_total <= 1e-6f) {
            use_dash = false;
        }
    }

    bool has_prev_end = false;
    float prev_end_x = 0.0f;
    float prev_end_y = 0.0f;
    bool has_prev_end_dir = false;
    float prev_end_dx = 0.0f;
    float prev_end_dy = 0.0f;
    for (uint32_t i = 0u; i < st->path_count; ++i) {
        const FS_PathSegment* seg = &st->path_segments[i];
        bool connected_next = false;
        if (i + 1u < st->path_count) {
            const FS_PathSegment* next_seg = &st->path_segments[i + 1u];
            connected_next =
                fabsf(next_seg->x0 - seg->x1) <= 1e-4f &&
                fabsf(next_seg->y0 - seg->y1) <= 1e-4f;
        }
        const bool connected =
            has_prev_end && fabsf(prev_end_x - seg->x0) <= 1e-4f && fabsf(prev_end_y - seg->y0) <= 1e-4f;
        float start_dx = 0.0f;
        float start_dy = 0.0f;
        const bool has_start_dir = fs_path_segment_start_dir(seg, &start_dx, &start_dy);
        uint32_t join_color = color;
        if (dynamic_stroke_style_color) {
            join_color = stroke_pattern_per_fragment ? pattern_base_color : fs_style_resolve_stroke_color_at(st, seg->x0, seg->y0);
        }
        if (connected && has_prev_end_dir && has_start_dir) {
            if (line_join == (uint8_t)FS_LINE_JOIN_ROUND) {
                if (oriented_transform) {
                    if (!fs_cmd_ellipse_compute_coverage_fill_with_flags(
                            core,
                            seg->x0,
                            seg->y0,
                            stroke_width * 0.5f,
                            stroke_width * 0.5f,
                            join_color,
                            stroke_pattern_flag
                        )) {
                        return false;
                    }
                } else if (!fs_cmd_circle_with_flags(
                               core,
                               seg->x0,
                               seg->y0,
                               stroke_width * 0.5f,
                               join_color,
                               stroke_pattern_flag
                           )) {
                    return false;
                }
            } else if (line_join == (uint8_t)FS_LINE_JOIN_BEVEL || line_join == (uint8_t)FS_LINE_JOIN_MITER) {
                if (!fs_emit_path_join(
                        core,
                        seg->x0,
                        seg->y0,
                        prev_end_dx,
                        prev_end_dy,
                        start_dx,
                        start_dy,
                        stroke_width,
                        join_color,
                        line_join,
                        st->style_miter_limit,
                        stroke_pattern_flag
                    )) {
                    return false;
                }
            }
        }
        float seg_sx = 0.0f;
        float seg_sy = 0.0f;
        fs_path_segment_sample_point(seg, &seg_sx, &seg_sy);
        uint32_t seg_color = color;
        if (dynamic_stroke_style_color) {
            seg_color = stroke_pattern_per_fragment ? pattern_base_color : fs_style_resolve_stroke_color_at(st, seg_sx, seg_sy);
        }
        bool ok = false;
        if (seg->type == (uint8_t)FS_PATH_SEG_LINE) {
            if (use_dash) {
                ok = fs_emit_dashed_line_segment(
                    core,
                    seg->x0,
                    seg->y0,
                    seg->x1,
                    seg->y1,
                    stroke_width,
                    seg_color,
                    line_cap,
                    st->style_dash_segments,
                    st->style_dash_count,
                    dash_total,
                    &dash_phase,
                    stroke_pattern_flag
                );
            } else {
                uint32_t line_flags = 0u;
                if (line_cap != (uint8_t)FS_LINE_CAP_ROUND) {
                    if (connected) {
                        line_flags |= FS_LINE_FLAG_NO_AA_START;
                    }
                    if (connected_next) {
                        line_flags |= FS_LINE_FLAG_NO_AA_END;
                    }
                }
                line_flags |= stroke_pattern_flag;
                ok = fs_emit_styled_line_segment_with_flags(
                    core,
                    seg->x0,
                    seg->y0,
                    seg->x1,
                    seg->y1,
                    stroke_width,
                    seg_color,
                    line_cap,
                    line_flags
                );
            }
        } else if (seg->type == (uint8_t)FS_PATH_SEG_QUAD) {
            if (use_dash) {
                const float len_a = hypotf(seg->cx0 - seg->x0, seg->cy0 - seg->y0);
                const float len_b = hypotf(seg->x1 - seg->cx0, seg->y1 - seg->cy0);
                uint32_t steps = (uint32_t)((len_a + len_b) / 14.0f) + 8u;
                if (steps < 8u) {
                    steps = 8u;
                } else if (steps > 128u) {
                    steps = 128u;
                }
                float px = seg->x0;
                float py = seg->y0;
                ok = true;
                for (uint32_t s = 1u; s <= steps; ++s) {
                    const float t = (float)s / (float)steps;
                    float qx = 0.0f;
                    float qy = 0.0f;
                    fs_eval_quad_point(seg->x0, seg->y0, seg->cx0, seg->cy0, seg->x1, seg->y1, t, &qx, &qy);
                    uint32_t piece_color = seg_color;
                    if (dynamic_stroke_style_color) {
                        if (stroke_pattern_per_fragment) {
                            piece_color = pattern_base_color;
                        } else {
                            const float mx = (px + qx) * 0.5f;
                            const float my = (py + qy) * 0.5f;
                            piece_color = fs_style_resolve_stroke_color_at(st, mx, my);
                        }
                    }
                    if (!fs_emit_dashed_line_segment(
                            core,
                            px,
                            py,
                            qx,
                            qy,
                            stroke_width,
                            piece_color,
                            line_cap,
                            st->style_dash_segments,
                            st->style_dash_count,
                            dash_total,
                            &dash_phase,
                            stroke_pattern_flag
                        )) {
                        ok = false;
                        break;
                    }
                    px = qx;
                    py = qy;
                }
            } else {
                ok = fs_cmd_bezier_quad_with_flags(
                    core,
                    seg->x0,
                    seg->y0,
                    seg->cx0,
                    seg->cy0,
                    seg->x1,
                    seg->y1,
                    stroke_width,
                    seg_color,
                    stroke_pattern_flag
                );
            }
        } else if (seg->type == (uint8_t)FS_PATH_SEG_CUBIC) {
            if (use_dash) {
                const float len_a = hypotf(seg->cx0 - seg->x0, seg->cy0 - seg->y0);
                const float len_b = hypotf(seg->cx1 - seg->cx0, seg->cy1 - seg->cy0);
                const float len_c = hypotf(seg->x1 - seg->cx1, seg->y1 - seg->cy1);
                uint32_t steps = (uint32_t)((len_a + len_b + len_c) / 12.0f) + 10u;
                if (steps < 10u) {
                    steps = 10u;
                } else if (steps > 192u) {
                    steps = 192u;
                }
                float px = seg->x0;
                float py = seg->y0;
                ok = true;
                for (uint32_t s = 1u; s <= steps; ++s) {
                    const float t = (float)s / (float)steps;
                    float qx = 0.0f;
                    float qy = 0.0f;
                    fs_eval_cubic_point(
                        seg->x0,
                        seg->y0,
                        seg->cx0,
                        seg->cy0,
                        seg->cx1,
                        seg->cy1,
                        seg->x1,
                        seg->y1,
                        t,
                        &qx,
                        &qy
                    );
                    uint32_t piece_color = seg_color;
                    if (dynamic_stroke_style_color) {
                        if (stroke_pattern_per_fragment) {
                            piece_color = pattern_base_color;
                        } else {
                            const float mx = (px + qx) * 0.5f;
                            const float my = (py + qy) * 0.5f;
                            piece_color = fs_style_resolve_stroke_color_at(st, mx, my);
                        }
                    }
                    if (!fs_emit_dashed_line_segment(
                            core,
                            px,
                            py,
                            qx,
                            qy,
                            stroke_width,
                            piece_color,
                            line_cap,
                            st->style_dash_segments,
                            st->style_dash_count,
                            dash_total,
                            &dash_phase,
                            stroke_pattern_flag
                        )) {
                        ok = false;
                        break;
                    }
                    px = qx;
                    py = qy;
                }
            } else {
                ok = fs_cmd_bezier_cubic_with_flags(
                    core,
                    seg->x0,
                    seg->y0,
                    seg->cx0,
                    seg->cy0,
                    seg->cx1,
                    seg->cy1,
                    seg->x1,
                    seg->y1,
                    stroke_width,
                    seg_color,
                    stroke_pattern_flag
                );
            }
        }
        if (!ok) {
            return false;
        }
        prev_end_x = seg->x1;
        prev_end_y = seg->y1;
        has_prev_end = true;
        has_prev_end_dir = fs_path_segment_end_dir(seg, &prev_end_dx, &prev_end_dy);
    }
    return true;
}

bool fs_path_stroke(FS_Core* core, float width, uint32_t color) {
    return fs_path_stroke_internal(core, width, color, false);
}

static bool fs_fill_points_reserve(FS_Point2** io_points, uint32_t* io_capacity, uint32_t required) {
    if (!io_points || !io_capacity) {
        return false;
    }
    if (required <= *io_capacity) {
        return true;
    }
    uint32_t new_cap = (*io_capacity > 0u) ? *io_capacity : 64u;
    while (new_cap < required) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_Point2* grown = (FS_Point2*)realloc(*io_points, (size_t)new_cap * sizeof(FS_Point2));
    if (!grown) {
        return false;
    }
    *io_points = grown;
    *io_capacity = new_cap;
    return true;
}

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

static bool fs_fill_points_push_unique(
    FS_Point2** io_points,
    uint32_t* io_count,
    uint32_t* io_capacity,
    float x,
    float y
) {
    if (!io_points || !io_count || !io_capacity) {
        return false;
    }
    if (*io_count > 0u) {
        const FS_Point2* prev = &(*io_points)[*io_count - 1u];
        if (fabsf(prev->x - x) <= 1e-4f && fabsf(prev->y - y) <= 1e-4f) {
            return true;
        }
    }
    const uint32_t required = *io_count + 1u;
    if (!fs_fill_points_reserve(io_points, io_capacity, required)) {
        return false;
    }
    FS_Point2* dst = *io_points;
    dst[*io_count].x = x;
    dst[*io_count].y = y;
    *io_count = required;
    return true;
}

static void fs_fill_contour_clear(FS_FillContour* contour) {
    if (!contour) {
        return;
    }
    free(contour->points);
    contour->points = NULL;
    contour->count = 0u;
    contour->capacity = 0u;
    contour->area2 = 0.0f;
    contour->abs_area2 = 0.0f;
    contour->parent = -1;
    contour->depth = 0u;
    contour->is_hole = false;
    contour->owner_outer = -1;
}

static bool fs_fill_contours_reserve(FS_FillContour** io_contours, uint32_t* io_capacity, uint32_t required) {
    if (!io_contours || !io_capacity) {
        return false;
    }
    if (required <= *io_capacity) {
        return true;
    }
    uint32_t new_cap = (*io_capacity > 0u) ? *io_capacity : 8u;
    while (new_cap < required) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_FillContour* grown = (FS_FillContour*)realloc(*io_contours, (size_t)new_cap * sizeof(FS_FillContour));
    if (!grown) {
        return false;
    }
    if (new_cap > *io_capacity) {
        memset(grown + *io_capacity, 0, (size_t)(new_cap - *io_capacity) * sizeof(FS_FillContour));
    }
    *io_contours = grown;
    *io_capacity = new_cap;
    return true;
}




static float fs_polygon_signed_area2(const FS_Point2* points, uint32_t count) {
    if (!points || count < 3u) {
        return 0.0f;
    }
    float area2 = 0.0f;
    for (uint32_t i = 0u; i < count; ++i) {
        const FS_Point2* a = &points[i];
        const FS_Point2* b = &points[(i + 1u) % count];
        area2 += (a->x * b->y) - (b->x * a->y);
    }
    return area2;
}

static float fs_cross2(const FS_Point2* a, const FS_Point2* b, const FS_Point2* c) {
    return (b->x - a->x) * (c->y - a->y) - (b->y - a->y) * (c->x - a->x);
}

static bool fs_point_in_contour(const FS_Point2* points, uint32_t count, const FS_Point2* p) {
    if (!points || !p || count < 3u) {
        return false;
    }
    bool inside = false;
    for (uint32_t i = 0u, j = count - 1u; i < count; j = i++) {
        const FS_Point2* a = &points[i];
        const FS_Point2* b = &points[j];
        const bool intersects =
            ((a->y > p->y) != (b->y > p->y)) &&
            (p->x < (b->x - a->x) * (p->y - a->y) / ((b->y - a->y) + 1e-12f) + a->x);
        if (intersects) {
            inside = !inside;
        }
    }
    return inside;
}

static void fs_points_reverse(FS_Point2* points, uint32_t count) {
    if (!points || count < 2u) {
        return;
    }
    uint32_t i = 0u;
    uint32_t j = count - 1u;
    while (i < j) {
        FS_Point2 tmp = points[i];
        points[i] = points[j];
        points[j] = tmp;
        ++i;
        --j;
    }
}


static bool fs_contour_finalize(FS_FillContour* contour) {
    if (!contour || contour->count < 3u) {
        return false;
    }
    if (!fs_polygon_compact_in_place(contour->points, &contour->count)) {
        return false;
    }
    contour->area2 = fs_polygon_signed_area2(contour->points, contour->count);
    contour->abs_area2 = fabsf(contour->area2);
    return contour->abs_area2 > 1e-5f;
}

static bool fs_point_in_triangle_or_edge(
    const FS_Point2* p,
    const FS_Point2* a,
    const FS_Point2* b,
    const FS_Point2* c
) {
    const float e0 = fs_cross2(a, b, p);
    const float e1 = fs_cross2(b, c, p);
    const float e2 = fs_cross2(c, a, p);
    const bool has_neg = (e0 < -1e-6f) || (e1 < -1e-6f) || (e2 < -1e-6f);
    const bool has_pos = (e0 > 1e-6f) || (e1 > 1e-6f) || (e2 > 1e-6f);
    return !(has_neg && has_pos);
}

static bool fs_emit_fill_triangle_fan(FS_Core* core, const FS_Point2* points, uint32_t count, uint32_t color) {
    if (!core || !points || count < 3u) {
        return true;
    }
    const FS_Point2 p0 = points[0];
    for (uint32_t i = 1u; i + 1u < count; ++i) {
        const FS_Point2 p1 = points[i];
        const FS_Point2 p2 = points[i + 1u];
        const float area2 =
            (p1.x - p0.x) * (p2.y - p0.y) -
            (p1.y - p0.y) * (p2.x - p0.x);
        if (fabsf(area2) <= 1e-6f) {
            continue;
        }
        if (!fs_cmd_triangle(core, p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, color)) {
            return false;
        }
    }
    return true;
}

static uint32_t fs_find_rightmost_point(const FS_Point2* points, uint32_t count) {
    uint32_t idx = 0u;
    for (uint32_t i = 1u; i < count; ++i) {
        if (points[i].x > points[idx].x + 1e-6f ||
            (fabsf(points[i].x - points[idx].x) <= 1e-6f && points[i].y < points[idx].y)) {
            idx = i;
        }
    }
    return idx;
}

static int fs_orient2d(const FS_Point2* a, const FS_Point2* b, const FS_Point2* c) {
    const float v = (b->x - a->x) * (c->y - a->y) - (b->y - a->y) * (c->x - a->x);
    if (v > 1e-6f) {
        return 1;
    }
    if (v < -1e-6f) {
        return -1;
    }
    return 0;
}

static bool fs_point_on_segment(const FS_Point2* p, const FS_Point2* a, const FS_Point2* b) {
    if (!p || !a || !b) {
        return false;
    }
    if (fabsf(fs_cross2(a, b, p)) > 1e-6f) {
        return false;
    }
    const float min_x = fminf(a->x, b->x) - 1e-6f;
    const float max_x = fmaxf(a->x, b->x) + 1e-6f;
    const float min_y = fminf(a->y, b->y) - 1e-6f;
    const float max_y = fmaxf(a->y, b->y) + 1e-6f;
    return p->x >= min_x && p->x <= max_x && p->y >= min_y && p->y <= max_y;
}

static bool fs_segments_intersect(const FS_Point2* a, const FS_Point2* b, const FS_Point2* c, const FS_Point2* d) {
    const int o1 = fs_orient2d(a, b, c);
    const int o2 = fs_orient2d(a, b, d);
    const int o3 = fs_orient2d(c, d, a);
    const int o4 = fs_orient2d(c, d, b);
    if (o1 != o2 && o3 != o4) {
        return true;
    }
    if (o1 == 0 && fs_point_on_segment(c, a, b)) {
        return true;
    }
    if (o2 == 0 && fs_point_on_segment(d, a, b)) {
        return true;
    }
    if (o3 == 0 && fs_point_on_segment(a, c, d)) {
        return true;
    }
    if (o4 == 0 && fs_point_on_segment(b, c, d)) {
        return true;
    }
    return false;
}

static bool fs_bridge_visible(
    const FS_Point2* outer,
    uint32_t outer_count,
    uint32_t outer_idx,
    const FS_Point2* hole,
    uint32_t hole_count,
    uint32_t hole_idx
) {
    if (!outer || !hole || outer_count < 3u || hole_count < 3u || outer_idx >= outer_count || hole_idx >= hole_count) {
        return false;
    }

    const FS_Point2* hp = &hole[hole_idx];
    const FS_Point2* op = &outer[outer_idx];
    const FS_Point2 seg_a = *hp;
    const FS_Point2 seg_b = *op;

    // Ensure the bridge leaves the hole boundary to the outside.
    const FS_Point2 near_hole = {
        .x = hp->x + (op->x - hp->x) * 1e-3f,
        .y = hp->y + (op->y - hp->y) * 1e-3f
    };
    if (fs_point_in_contour(hole, hole_count, &near_hole)) {
        return false;
    }

    // Bridge midpoint should remain inside outer polygon.
    const FS_Point2 mid = {
        .x = 0.5f * (hp->x + op->x),
        .y = 0.5f * (hp->y + op->y)
    };
    if (!fs_point_in_contour(outer, outer_count, &mid)) {
        return false;
    }

    for (uint32_t i = 0u; i < outer_count; ++i) {
        const uint32_t j = (i + 1u) % outer_count;
        if (i == outer_idx || j == outer_idx) {
            continue;
        }
        if (fs_segments_intersect(&seg_a, &seg_b, &outer[i], &outer[j])) {
            return false;
        }
    }

    for (uint32_t i = 0u; i < hole_count; ++i) {
        const uint32_t j = (i + 1u) % hole_count;
        if (i == hole_idx || j == hole_idx) {
            continue;
        }
        if (fs_segments_intersect(&seg_a, &seg_b, &hole[i], &hole[j])) {
            return false;
        }
    }
    return true;
}

static bool fs_find_outer_bridge_point(
    const FS_Point2* outer,
    uint32_t outer_count,
    const FS_Point2* hole,
    uint32_t hole_count,
    uint32_t hole_idx,
    uint32_t* out_outer_idx
) {
    if (!outer || !hole || !out_outer_idx || outer_count < 3u || hole_count < 3u || hole_idx >= hole_count) {
        return false;
    }

    const FS_Point2* hole_point = &hole[hole_idx];
    const float hx = hole_point->x;
    const float hy = hole_point->y;
    const float eps = 1e-6f;

    // Earcut-style: cast horizontal ray to +X and pick the nearest intersected outer edge.
    bool ray_hit = false;
    float best_ix = 1e30f;
    uint32_t best_ei = 0u;
    uint32_t best_ej = 0u;
    for (uint32_t i = 0u; i < outer_count; ++i) {
        const uint32_t j = (i + 1u) % outer_count;
        const FS_Point2* a = &outer[i];
        const FS_Point2* b = &outer[j];
        const float ay = a->y;
        const float by = b->y;
        if (fabsf(ay - by) <= eps) {
            continue;
        }
        if ((hy < fminf(ay, by)) || (hy > fmaxf(ay, by))) {
            continue;
        }
        const float t = (hy - ay) / (by - ay);
        if (t < -eps || t > 1.0f + eps) {
            continue;
        }
        const float ix = a->x + t * (b->x - a->x);
        if (ix <= hx + eps) {
            continue;
        }
        if (!ray_hit || ix < best_ix) {
            ray_hit = true;
            best_ix = ix;
            best_ei = i;
            best_ej = j;
        }
    }

    if (ray_hit) {
        uint32_t primary = best_ei;
        uint32_t secondary = best_ej;
        if (outer[best_ej].x > outer[best_ei].x) {
            primary = best_ej;
            secondary = best_ei;
        }
        if (fs_bridge_visible(outer, outer_count, primary, hole, hole_count, hole_idx)) {
            *out_outer_idx = primary;
            return true;
        }
        if (fs_bridge_visible(outer, outer_count, secondary, hole, hole_count, hole_idx)) {
            *out_outer_idx = secondary;
            return true;
        }

        // If edge endpoints fail visibility, choose any visible vertex on/near the ray interval.
        uint32_t interval_best = 0u;
        float interval_score = 1e30f;
        bool interval_found = false;
        for (uint32_t i = 0u; i < outer_count; ++i) {
            if (outer[i].x < hx - eps || outer[i].x > best_ix + eps) {
                continue;
            }
            if (!fs_bridge_visible(outer, outer_count, i, hole, hole_count, hole_idx)) {
                continue;
            }
            const float dx = outer[i].x - hx;
            const float dy = outer[i].y - hy;
            const float score = dx * dx + dy * dy;
            if (!interval_found || score < interval_score) {
                interval_found = true;
                interval_best = i;
                interval_score = score;
            }
        }
        if (interval_found) {
            *out_outer_idx = interval_best;
            return true;
        }
    }

    // Fallback: nearest visible vertex with preference to +X side.
    uint32_t fallback_best = 0u;
    float fallback_score = 1e30f;
    bool fallback_right = false;

    uint32_t visible_best = 0u;
    float visible_score = 1e30f;
    bool visible_right = false;
    bool found_visible = false;

    for (uint32_t i = 0u; i < outer_count; ++i) {
        const float dx = outer[i].x - hole_point->x;
        const float dy = outer[i].y - hole_point->y;
        const float d2 = dx * dx + dy * dy;
        const bool right = dx >= -1e-4f;

        if (right) {
            if (!fallback_right || d2 < fallback_score) {
                fallback_right = true;
                fallback_best = i;
                fallback_score = d2;
            }
        } else if (!fallback_right && d2 < fallback_score) {
            fallback_best = i;
            fallback_score = d2;
        }

        if (!fs_bridge_visible(outer, outer_count, i, hole, hole_count, hole_idx)) {
            continue;
        }
        if (right) {
            if (!visible_right || d2 < visible_score) {
                visible_right = true;
                visible_best = i;
                visible_score = d2;
                found_visible = true;
            }
        } else if (!visible_right && (!found_visible || d2 < visible_score)) {
            visible_best = i;
            visible_score = d2;
            found_visible = true;
        }
    }

    *out_outer_idx = found_visible ? visible_best : fallback_best;
    return true;
}

static bool fs_merge_hole_into_polygon(
    FS_Point2** io_poly,
    uint32_t* io_count,
    uint32_t* io_capacity,
    FS_Point2* hole,
    uint32_t hole_count,
    uint32_t hole_right_idx
) {
    if (!io_poly || !io_count || !io_capacity || !hole || hole_count < 3u || *io_count < 3u) {
        return false;
    }
    if (hole_right_idx >= hole_count) {
        return false;
    }
    uint32_t oi = 0u;
    if (!fs_find_outer_bridge_point(*io_poly, *io_count, hole, hole_count, hole_right_idx, &oi)) {
        return false;
    }

    const uint32_t old_count = *io_count;
    const uint32_t required = old_count + hole_count + 2u;
    FS_Point2* merged = NULL;
    uint32_t merged_capacity = 0u;
    if (!fs_fill_points_reserve(&merged, &merged_capacity, required)) {
        return false;
    }

    uint32_t out_count = 0u;
    for (uint32_t i = 0u; i <= oi; ++i) {
        merged[out_count++] = (*io_poly)[i];
    }
    for (uint32_t s = 0u; s < hole_count; ++s) {
        const uint32_t hi = (hole_right_idx + s) % hole_count;
        merged[out_count++] = hole[hi];
    }
    merged[out_count++] = hole[hole_right_idx];
    for (uint32_t i = oi; i < old_count; ++i) {
        merged[out_count++] = (*io_poly)[i];
    }

    free(*io_poly);
    *io_poly = merged;
    *io_count = out_count;
    *io_capacity = merged_capacity;
    return true;
}

static bool fs_emit_fill_triangles_ear_clip(FS_Core* core, const FS_Point2* points, uint32_t count, uint32_t color, bool allow_fan_fallback) {
    if (!core || !points || count < 3u) {
        return true;
    }

    uint32_t* indices = (uint32_t*)malloc((size_t)count * sizeof(uint32_t));
    if (!indices) {
        return false;
    }
    for (uint32_t i = 0u; i < count; ++i) {
        indices[i] = i;
    }

    uint32_t remaining = count;
    const bool ccw = fs_polygon_signed_area2(points, count) >= 0.0f;
    bool ok = true;
    uint32_t guard = 0u;
    const uint32_t guard_max = count * count * 2u + 16u;

    while (ok && remaining > 3u && guard < guard_max) {
        bool ear_found = false;
        for (uint32_t i = 0u; i < remaining; ++i) {
            const uint32_t ip = (i + remaining - 1u) % remaining;
            const uint32_t in = (i + 1u) % remaining;
            const FS_Point2* a = &points[indices[ip]];
            const FS_Point2* b = &points[indices[i]];
            const FS_Point2* c = &points[indices[in]];
            const float cross = fs_cross2(a, b, c);
            const bool is_convex = ccw ? (cross > 1e-6f) : (cross < -1e-6f);
            if (!is_convex) {
                continue;
            }

            bool contains_other = false;
            for (uint32_t j = 0u; j < remaining; ++j) {
                if (j == ip || j == i || j == in) {
                    continue;
                }
                const FS_Point2* p = &points[indices[j]];
                if (fs_point_in_triangle_or_edge(p, a, b, c)) {
                    contains_other = true;
                    break;
                }
            }
            if (contains_other) {
                continue;
            }

            if (!fs_cmd_triangle(core, a->x, a->y, b->x, b->y, c->x, c->y, color)) {
                ok = false;
                break;
            }

            if (i + 1u < remaining) {
                memmove(&indices[i], &indices[i + 1u], (size_t)(remaining - i - 1u) * sizeof(uint32_t));
            }
            remaining -= 1u;
            ear_found = true;
            break;
        }

        if (!ear_found) {
            break;
        }
        guard += 1u;
    }

    if (ok && remaining == 3u) {
        const FS_Point2* a = &points[indices[0]];
        const FS_Point2* b = &points[indices[1]];
        const FS_Point2* c = &points[indices[2]];
        if (fabsf(fs_cross2(a, b, c)) > 1e-6f) {
            ok = fs_cmd_triangle(core, a->x, a->y, b->x, b->y, c->x, c->y, color);
        }
    } else if (ok && remaining > 3u && allow_fan_fallback) {
        // Fallback keeps rendering alive even for self-intersection/degenerate input.
        ok = fs_emit_fill_triangle_fan(core, points, count, color);
    } else if (ok && remaining > 3u) {
        ok = false;
    }

    free(indices);
    return ok;
}

static bool fs_fill_current_path_via_clip_mask(FS_Core* core, uint32_t color, uint32_t fill_mode) {
    FS_InternalState* st = fs_state(core);
    if (!core || !st || st->path_count == 0u || core->width == 0u || core->height == 0u) {
        return false;
    }

    fs_state_save(core);
    const bool clipped = fs_clip_path_with_mode(core, fill_mode, false, FS_FILL_RULE_NONZERO);
    if (!clipped) {
        (void)fs_state_restore(core);
        return false;
    }

    // Fill happens in device-space rect; the active path mask carries the shape.
    fs_transform_reset(core);
    const bool draw_ok = fs_cmd_rect(core, 0.0f, 0.0f, (float)core->width, (float)core->height, 0.0f, color);
    const bool restore_ok = fs_state_restore(core);
    return clipped && draw_ok && restore_ok;
}

bool fs_path_fill(FS_Core* core, uint32_t color) {
    if (!core) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (st->path_count == 0u) {
        return true;
    }

    // Preferred path fill: GPU clip-mask fill (SDF edge solve in clip compute shader).
    // Falls back to CPU triangulation when clip resources/layers are unavailable.
    if (fs_fill_current_path_via_clip_mask(core, color, FS_CLIP_FILL_MODE_SDF)) {
        return true;
    }

    FS_FillContour* contours = NULL;
    uint32_t contour_count = 0u;
    uint32_t contour_capacity = 0u;

    FS_FillContour current = {0};
    bool have_current = false;
    bool have_prev_end = false;
    float prev_end_x = 0.0f;
    float prev_end_y = 0.0f;
    bool ok = true;

    for (uint32_t i = 0u; i < st->path_count && ok; ++i) {
        const FS_PathSegment* seg = &st->path_segments[i];
        const bool contour_break =
            !have_prev_end ||
            fabsf(prev_end_x - seg->x0) > 1e-4f ||
            fabsf(prev_end_y - seg->y0) > 1e-4f;

        if (contour_break && have_current) {
            if (fs_contour_finalize(&current)) {
                if (!fs_fill_contours_reserve(&contours, &contour_capacity, contour_count + 1u)) {
                    ok = false;
                    break;
                }
                contours[contour_count++] = current;
                memset(&current, 0, sizeof(current));
            } else {
                fs_fill_contour_clear(&current);
            }
            have_current = false;
        }

        if (!have_current) {
            have_current = true;
            if (!fs_fill_points_push_unique(&current.points, &current.count, &current.capacity, seg->x0, seg->y0)) {
                ok = false;
                break;
            }
        }

        if (seg->type == (uint8_t)FS_PATH_SEG_LINE) {
            ok = fs_fill_points_push_unique(&current.points, &current.count, &current.capacity, seg->x1, seg->y1);
        } else if (seg->type == (uint8_t)FS_PATH_SEG_QUAD) {
            const float len_a = hypotf(seg->cx0 - seg->x0, seg->cy0 - seg->y0);
            const float len_b = hypotf(seg->x1 - seg->cx0, seg->y1 - seg->cy0);
            uint32_t steps = (uint32_t)((len_a + len_b) / 16.0f) + 8u;
            if (steps < 8u) {
                steps = 8u;
            } else if (steps > 96u) {
                steps = 96u;
            }
            for (uint32_t s = 1u; s <= steps && ok; ++s) {
                const float t = (float)s / (float)steps;
                float qx = 0.0f;
                float qy = 0.0f;
                fs_eval_quad_point(seg->x0, seg->y0, seg->cx0, seg->cy0, seg->x1, seg->y1, t, &qx, &qy);
                ok = fs_fill_points_push_unique(&current.points, &current.count, &current.capacity, qx, qy);
            }
        } else if (seg->type == (uint8_t)FS_PATH_SEG_CUBIC) {
            const float len_a = hypotf(seg->cx0 - seg->x0, seg->cy0 - seg->y0);
            const float len_b = hypotf(seg->cx1 - seg->cx0, seg->cy1 - seg->cy0);
            const float len_c = hypotf(seg->x1 - seg->cx1, seg->y1 - seg->cy1);
            uint32_t steps = (uint32_t)((len_a + len_b + len_c) / 12.0f) + 10u;
            if (steps < 10u) {
                steps = 10u;
            } else if (steps > 144u) {
                steps = 144u;
            }
            for (uint32_t s = 1u; s <= steps && ok; ++s) {
                const float t = (float)s / (float)steps;
                float qx = 0.0f;
                float qy = 0.0f;
                fs_eval_cubic_point(
                    seg->x0,
                    seg->y0,
                    seg->cx0,
                    seg->cy0,
                    seg->cx1,
                    seg->cy1,
                    seg->x1,
                    seg->y1,
                    t,
                    &qx,
                    &qy
                );
                ok = fs_fill_points_push_unique(&current.points, &current.count, &current.capacity, qx, qy);
            }
        }

        prev_end_x = seg->x1;
        prev_end_y = seg->y1;
        have_prev_end = true;
    }

    if (ok && have_current) {
        if (fs_contour_finalize(&current)) {
            if (!fs_fill_contours_reserve(&contours, &contour_capacity, contour_count + 1u)) {
                ok = false;
            } else {
                contours[contour_count++] = current;
                memset(&current, 0, sizeof(current));
            }
        } else {
            fs_fill_contour_clear(&current);
        }
    }

    if (!ok || contour_count == 0u) {
        fs_fill_contour_clear(&current);
        for (uint32_t i = 0u; i < contour_count; ++i) {
            fs_fill_contour_clear(&contours[i]);
        }
        free(contours);
        return ok;
    }

    for (uint32_t i = 0u; i < contour_count; ++i) {
        int32_t parent = -1;
        float parent_area = 1e30f;
        const FS_Point2 probe = contours[i].points[0];
        for (uint32_t j = 0u; j < contour_count; ++j) {
            if (j == i || contours[j].abs_area2 <= contours[i].abs_area2 + 1e-5f) {
                continue;
            }
            if (!fs_point_in_contour(contours[j].points, contours[j].count, &probe)) {
                continue;
            }
            if (contours[j].abs_area2 < parent_area) {
                parent_area = contours[j].abs_area2;
                parent = (int32_t)j;
            }
        }
        contours[i].parent = parent;
    }

    for (uint32_t i = 0u; i < contour_count; ++i) {
        uint32_t depth = 0u;
        int32_t p = contours[i].parent;
        while (p >= 0 && depth < 1024u) {
            depth += 1u;
            p = contours[(uint32_t)p].parent;
        }
        contours[i].depth = depth;
    }

    const FS_FillRule fill_rule =
        (st->style_fill_rule == (uint8_t)FS_FILL_RULE_EVENODD) ? FS_FILL_RULE_EVENODD : FS_FILL_RULE_NONZERO;
    for (uint32_t i = 0u; i < contour_count; ++i) {
        if (contours[i].parent < 0) {
            contours[i].is_hole = false;
            continue;
        }
        if (fill_rule == FS_FILL_RULE_EVENODD) {
            contours[i].is_hole = ((contours[i].depth & 1u) == 1u);
        } else {
            const int32_t p = contours[i].parent;
            const bool sign_self = contours[i].area2 >= 0.0f;
            const bool sign_parent = contours[(uint32_t)p].area2 >= 0.0f;
            contours[i].is_hole = (sign_self != sign_parent);
        }
    }

    for (uint32_t i = 0u; i < contour_count; ++i) {
        if (!contours[i].is_hole) {
            contours[i].owner_outer = (int32_t)i;
            continue;
        }
        int32_t p = contours[i].parent;
        int32_t owner = -1;
        while (p >= 0) {
            if (!contours[(uint32_t)p].is_hole) {
                owner = p;
                break;
            }
            p = contours[(uint32_t)p].parent;
        }
        contours[i].owner_outer = owner;
    }

    for (uint32_t i = 0u; i < contour_count && ok; ++i) {
        if (contours[i].is_hole) {
            continue;
        }

        FS_Point2* poly = NULL;
        uint32_t poly_count = 0u;
        uint32_t poly_capacity = 0u;
        if (!fs_fill_points_reserve(&poly, &poly_capacity, contours[i].count)) {
            ok = false;
            break;
        }
        memcpy(poly, contours[i].points, (size_t)contours[i].count * sizeof(FS_Point2));
        poly_count = contours[i].count;
        if (contours[i].area2 < 0.0f) {
            fs_points_reverse(poly, poly_count);
        }

        uint32_t hole_local_count = 0u;
        for (uint32_t h = 0u; h < contour_count; ++h) {
            if (contours[h].is_hole && contours[h].owner_outer == (int32_t)i) {
                hole_local_count += 1u;
            }
        }
        uint32_t* hole_indices = NULL;
        uint32_t* hole_right_indices = NULL;
        float* hole_right_x = NULL;
        if (hole_local_count > 0u) {
            hole_indices = (uint32_t*)malloc((size_t)hole_local_count * sizeof(uint32_t));
            hole_right_indices = (uint32_t*)malloc((size_t)hole_local_count * sizeof(uint32_t));
            hole_right_x = (float*)malloc((size_t)hole_local_count * sizeof(float));
            if (!hole_indices || !hole_right_indices || !hole_right_x) {
                ok = false;
            } else {
                uint32_t cursor = 0u;
                for (uint32_t h = 0u; h < contour_count; ++h) {
                    if (!contours[h].is_hole || contours[h].owner_outer != (int32_t)i) {
                        continue;
                    }
                    FS_Point2* hole = contours[h].points;
                    uint32_t hole_count = contours[h].count;
                    if (hole_count < 3u) {
                        continue;
                    }
                    if (contours[h].area2 > 0.0f) {
                        fs_points_reverse(hole, hole_count);
                        contours[h].area2 = -contours[h].area2;
                    }
                    const uint32_t hr = fs_find_rightmost_point(hole, hole_count);
                    hole_indices[cursor] = h;
                    hole_right_indices[cursor] = hr;
                    hole_right_x[cursor] = hole[hr].x;
                    cursor += 1u;
                }
                hole_local_count = cursor;
                for (uint32_t a = 0u; a + 1u < hole_local_count; ++a) {
                    for (uint32_t b = a + 1u; b < hole_local_count; ++b) {
                        if (hole_right_x[b] > hole_right_x[a]) {
                            const float tx = hole_right_x[a];
                            hole_right_x[a] = hole_right_x[b];
                            hole_right_x[b] = tx;
                            const uint32_t ti = hole_indices[a];
                            hole_indices[a] = hole_indices[b];
                            hole_indices[b] = ti;
                            const uint32_t tr = hole_right_indices[a];
                            hole_right_indices[a] = hole_right_indices[b];
                            hole_right_indices[b] = tr;
                        }
                    }
                }
            }
        }

        for (uint32_t h = 0u; h < hole_local_count && ok; ++h) {
            const uint32_t hole_ci = hole_indices[h];
            FS_Point2* hole = contours[hole_ci].points;
            const uint32_t hole_count = contours[hole_ci].count;
            const uint32_t hole_right = hole_right_indices[h];
            if (!fs_merge_hole_into_polygon(&poly, &poly_count, &poly_capacity, hole, hole_count, hole_right)) {
                ok = false;
                break;
            }
            if (!fs_polygon_compact_in_place(poly, &poly_count)) {
                ok = false;
                break;
            }
        }
        free(hole_indices);
        free(hole_right_indices);
        free(hole_right_x);

        if (ok) {
            ok = fs_polygon_compact_in_place(poly, &poly_count);
        }

        if (ok) {
            ok = fs_emit_fill_triangles_ear_clip(core, poly, poly_count, color, hole_local_count == 0u);
        }
        free(poly);
    }

    for (uint32_t i = 0u; i < contour_count; ++i) {
        fs_fill_contour_clear(&contours[i]);
    }
    free(contours);
    return ok;
}

bool fs_fill(FS_Core* core) {
    return fs_path_fill(core, fs_style_get_fill_color(core));
}

bool fs_is_point_in_path_with_fill_rule(FS_Core* core, float x, float y, FS_FillRule fill_rule) {
    if (!core || !isfinite(x) || !isfinite(y)) {
        return false;
    }
    if (fill_rule != FS_FILL_RULE_NONZERO && fill_rule != FS_FILL_RULE_EVENODD) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || !st->path_segments || st->path_count == 0u) {
        return false;
    }
    return fs_hit_test_fill_path_device(st->path_segments, st->path_count, &st->current_transform, x, y, fill_rule);
}

bool fs_is_point_in_path(FS_Core* core, float x, float y) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    const FS_FillRule fill_rule =
        (st->style_fill_rule == (uint8_t)FS_FILL_RULE_EVENODD) ? FS_FILL_RULE_EVENODD : FS_FILL_RULE_NONZERO;
    return fs_is_point_in_path_with_fill_rule(core, x, y, fill_rule);
}

bool fs_is_point_in_stroke(FS_Core* core, float x, float y) {
    if (!core || !isfinite(x) || !isfinite(y)) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || !st->path_segments || st->path_count == 0u) {
        return false;
    }
    const float stroke_width = fs_style_resolve_line_width(st, 0.0f);
    if (!isfinite(stroke_width) || stroke_width <= 0.0f) {
        return false;
    }
    return fs_hit_test_stroke_path_device(
        st->path_segments,
        st->path_count,
        &st->current_transform,
        st->style_line_cap,
        st->style_line_join,
        st->style_miter_limit,
        st->style_dash_segments,
        st->style_dash_count,
        st->style_dash_offset,
        stroke_width,
        x,
        y
    );
}

FS_Path2D* fs_path2d_create(void) {
    FS_Path2D* path = (FS_Path2D*)calloc(1u, sizeof(FS_Path2D));
    return path;
}

void fs_path2d_destroy(FS_Path2D* path) {
    if (!path) {
        return;
    }
    free(path->segments);
    path->segments = NULL;
    path->count = 0u;
    path->capacity = 0u;
    free(path);
}

void fs_path2d_reset(FS_Path2D* path) {
    if (!path) {
        return;
    }
    path->count = 0u;
    path->has_current = false;
    path->has_subpath_start = false;
    path->current_x = 0.0f;
    path->current_y = 0.0f;
    path->subpath_start_x = 0.0f;
    path->subpath_start_y = 0.0f;
}

bool fs_path2d_move_to(FS_Path2D* path, float x, float y) {
    if (!path) {
        return false;
    }
    path->has_current = true;
    path->has_subpath_start = true;
    path->current_x = x;
    path->current_y = y;
    path->subpath_start_x = x;
    path->subpath_start_y = y;
    return true;
}

bool fs_path2d_line_to(FS_Path2D* path, float x, float y) {
    if (!path) {
        return false;
    }
    if (!path->has_current) {
        return fs_path2d_move_to(path, x, y);
    }
    FS_PathSegment seg;
    memset(&seg, 0, sizeof(seg));
    seg.type = (uint8_t)FS_PATH_SEG_LINE;
    seg.x0 = path->current_x;
    seg.y0 = path->current_y;
    seg.x1 = x;
    seg.y1 = y;
    if (!fs_path2d_append_segment(path, &seg)) {
        return false;
    }
    path->current_x = x;
    path->current_y = y;
    return true;
}

bool fs_path2d_quadratic_curve_to(FS_Path2D* path, float cx, float cy, float x, float y) {
    if (!path) {
        return false;
    }
    if (!path->has_current) {
        return fs_path2d_move_to(path, x, y);
    }
    FS_PathSegment seg;
    memset(&seg, 0, sizeof(seg));
    seg.type = (uint8_t)FS_PATH_SEG_QUAD;
    seg.x0 = path->current_x;
    seg.y0 = path->current_y;
    seg.cx0 = cx;
    seg.cy0 = cy;
    seg.x1 = x;
    seg.y1 = y;
    if (!fs_path2d_append_segment(path, &seg)) {
        return false;
    }
    path->current_x = x;
    path->current_y = y;
    return true;
}

bool fs_path2d_bezier_curve_to(FS_Path2D* path, float cx0, float cy0, float cx1, float cy1, float x, float y) {
    if (!path) {
        return false;
    }
    if (!path->has_current) {
        return fs_path2d_move_to(path, x, y);
    }
    FS_PathSegment seg;
    memset(&seg, 0, sizeof(seg));
    seg.type = (uint8_t)FS_PATH_SEG_CUBIC;
    seg.x0 = path->current_x;
    seg.y0 = path->current_y;
    seg.cx0 = cx0;
    seg.cy0 = cy0;
    seg.cx1 = cx1;
    seg.cy1 = cy1;
    seg.x1 = x;
    seg.y1 = y;
    if (!fs_path2d_append_segment(path, &seg)) {
        return false;
    }
    path->current_x = x;
    path->current_y = y;
    return true;
}

bool fs_path2d_arc_to(FS_Path2D* path, float x1, float y1, float x2, float y2, float radius) {
    const float pi = 3.14159265358979323846f;
    if (!path || radius < 0.0f) {
        return false;
    }
    if (!path->has_current) {
        return fs_path2d_move_to(path, x1, y1);
    }

    const float x0 = path->current_x;
    const float y0 = path->current_y;
    const float dx01 = x0 - x1;
    const float dy01 = y0 - y1;
    const float dx21 = x2 - x1;
    const float dy21 = y2 - y1;
    const float len01 = hypotf(dx01, dy01);
    const float len21 = hypotf(dx21, dy21);
    if (radius <= 1e-6f || len01 <= 1e-6f || len21 <= 1e-6f) {
        return fs_path2d_line_to(path, x1, y1);
    }

    const float u1x = dx01 / len01;
    const float u1y = dy01 / len01;
    const float u2x = dx21 / len21;
    const float u2y = dy21 / len21;
    float dot = u1x * u2x + u1y * u2y;
    if (dot > 1.0f) dot = 1.0f;
    if (dot < -1.0f) dot = -1.0f;
    const float cross = u1x * u2y - u1y * u2x;
    const float angle = acosf(dot);
    if (fabsf(cross) <= 1e-7f || angle <= 1e-5f || fabsf(pi - angle) <= 1e-5f) {
        return fs_path2d_line_to(path, x1, y1);
    }

    float t = radius / tanf(angle * 0.5f);
    if (!isfinite(t) || t <= 1e-6f) {
        return fs_path2d_line_to(path, x1, y1);
    }
    const float max_t = fminf(len01, len21) - 1e-4f;
    if (max_t <= 1e-6f) {
        return fs_path2d_line_to(path, x1, y1);
    }
    if (t > max_t) {
        t = max_t;
    }

    const float tx1 = x1 + u1x * t;
    const float ty1 = y1 + u1y * t;
    const float tx2 = x1 + u2x * t;
    const float ty2 = y1 + u2y * t;
    if (!fs_path2d_line_to(path, tx1, ty1)) {
        return false;
    }

    const float bisx = u1x + u2x;
    const float bisy = u1y + u2y;
    const float bis_len = hypotf(bisx, bisy);
    if (bis_len <= 1e-6f) {
        return fs_path2d_line_to(path, tx2, ty2);
    }
    const float inv_bis = 1.0f / bis_len;
    const float bx = bisx * inv_bis;
    const float by = bisy * inv_bis;
    const float center_dist = radius / sinf(angle * 0.5f);
    const float cx = x1 + bx * center_dist;
    const float cy = y1 + by * center_dist;

    float a0 = atan2f(ty1 - cy, tx1 - cx);
    float a1 = atan2f(ty2 - cy, tx2 - cx);
    const bool ccw = cross < 0.0f;
    if (ccw) {
        while (a1 <= a0) {
            a1 += 2.0f * pi;
        }
    } else {
        while (a1 >= a0) {
            a1 -= 2.0f * pi;
        }
    }
    return fs_path2d_append_arc_sweep(path, cx, cy, radius, a0, a1);
}

bool fs_path2d_rect(FS_Path2D* path, float x, float y, float w, float h) {
    if (!path) {
        return false;
    }
    const float x0 = x;
    const float y0 = y;
    const float x1 = x + w;
    const float y1 = y + h;
    if (!fs_path2d_move_to(path, x0, y0)) {
        return false;
    }
    if (!fs_path2d_line_to(path, x1, y0)) {
        return false;
    }
    if (!fs_path2d_line_to(path, x1, y1)) {
        return false;
    }
    if (!fs_path2d_line_to(path, x0, y1)) {
        return false;
    }
    return fs_path2d_close(path);
}

bool fs_path2d_round_rect(FS_Path2D* path, float x, float y, float w, float h, float radius) {
    const float pi = 3.14159265358979323846f;
    if (!path) {
        return false;
    }
    if (fabsf(w) <= 1e-6f || fabsf(h) <= 1e-6f) {
        return fs_path2d_rect(path, x, y, w, h);
    }
    const float left = fminf(x, x + w);
    const float right = fmaxf(x, x + w);
    const float top = fminf(y, y + h);
    const float bottom = fmaxf(y, y + h);
    const float width = right - left;
    const float height = bottom - top;

    float r = radius;
    if (r < 0.0f) {
        r = 0.0f;
    }
    const float max_r = fminf(width, height) * 0.5f;
    if (r > max_r) {
        r = max_r;
    }
    if (r <= 1e-6f) {
        return fs_path2d_rect(path, x, y, w, h);
    }

    if (!fs_path2d_move_to(path, left + r, top)) {
        return false;
    }
    if (!fs_path2d_line_to(path, right - r, top)) {
        return false;
    }
    if (!fs_path2d_append_arc_sweep(path, right - r, top + r, r, -0.5f * pi, 0.0f)) {
        return false;
    }
    if (!fs_path2d_line_to(path, right, bottom - r)) {
        return false;
    }
    if (!fs_path2d_append_arc_sweep(path, right - r, bottom - r, r, 0.0f, 0.5f * pi)) {
        return false;
    }
    if (!fs_path2d_line_to(path, left + r, bottom)) {
        return false;
    }
    if (!fs_path2d_append_arc_sweep(path, left + r, bottom - r, r, 0.5f * pi, pi)) {
        return false;
    }
    if (!fs_path2d_line_to(path, left, top + r)) {
        return false;
    }
    if (!fs_path2d_append_arc_sweep(path, left + r, top + r, r, pi, 1.5f * pi)) {
        return false;
    }
    return fs_path2d_close(path);
}

bool fs_path2d_close(FS_Path2D* path) {
    if (!path) {
        return false;
    }
    if (!path->has_current || !path->has_subpath_start) {
        return true;
    }
    const float dx = path->current_x - path->subpath_start_x;
    const float dy = path->current_y - path->subpath_start_y;
    if (fabsf(dx) <= 1e-6f && fabsf(dy) <= 1e-6f) {
        return true;
    }
    return fs_path2d_line_to(path, path->subpath_start_x, path->subpath_start_y);
}

bool fs_path2d_arc(
    FS_Path2D* path,
    float cx,
    float cy,
    float radius,
    float start_angle,
    float end_angle,
    bool counterclockwise
) {
    return fs_path2d_ellipse(path, cx, cy, radius, radius, 0.0f, start_angle, end_angle, counterclockwise);
}

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
) {
    if (!path ||
        !isfinite(cx) || !isfinite(cy) ||
        !isfinite(radius_x) || !isfinite(radius_y) ||
        !isfinite(rotation) || !isfinite(start_angle) || !isfinite(end_angle) ||
        radius_x < 0.0f || radius_y < 0.0f) {
        return false;
    }

    const float cr = cosf(rotation);
    const float sr = sinf(rotation);
    const float cs = cosf(start_angle);
    const float ss = sinf(start_angle);
    const float sx = cx + (radius_x * cs) * cr - (radius_y * ss) * sr;
    const float sy = cy + (radius_x * cs) * sr + (radius_y * ss) * cr;

    if (!path->has_current) {
        if (!fs_path2d_move_to(path, sx, sy)) {
            return false;
        }
    } else {
        const float dx = path->current_x - sx;
        const float dy = path->current_y - sy;
        if (fabsf(dx) > 1e-6f || fabsf(dy) > 1e-6f) {
            if (!fs_path2d_line_to(path, sx, sy)) {
                return false;
            }
        }
    }

    const float delta = fs_arc_resolve_delta(start_angle, end_angle, counterclockwise);
    if (fabsf(delta) <= 1e-7f || radius_x <= 1e-7f || radius_y <= 1e-7f) {
        return true;
    }
    return fs_path2d_append_ellipse_arc_sweep(
        path,
        cx,
        cy,
        radius_x,
        radius_y,
        rotation,
        start_angle,
        start_angle + delta
    );
}

bool fs_path2d_add_path(FS_Path2D* path, const FS_Path2D* other) {
    if (!path || !other) {
        return false;
    }
    if (other->count == 0u) {
        if (other->has_current) {
            return fs_path2d_move_to(path, other->current_x, other->current_y);
        }
        return true;
    }
    if (!other->segments) {
        return false;
    }
    if (!fs_path2d_ensure_capacity(path, path->count + other->count)) {
        return false;
    }
    memcpy(
        &path->segments[path->count],
        other->segments,
        (size_t)other->count * sizeof(FS_PathSegment)
    );
    path->count += other->count;
    if (other->has_current) {
        path->has_current = true;
        path->current_x = other->current_x;
        path->current_y = other->current_y;
        path->has_subpath_start = other->has_subpath_start;
        path->subpath_start_x = other->subpath_start_x;
        path->subpath_start_y = other->subpath_start_y;
    } else {
        const FS_PathSegment* last = &path->segments[path->count - 1u];
        path->has_current = true;
        path->current_x = last->x1;
        path->current_y = last->y1;
        path->has_subpath_start = true;
        path->subpath_start_x = last->x0;
        path->subpath_start_y = last->y0;
    }
    return true;
}

bool fs_path2d_add_path_with_transform(FS_Path2D* path, const FS_Path2D* other, const float matrix_2x3[6]) {
    if (!path || !other || !matrix_2x3) {
        return false;
    }
    FS_Transform2D t;
    t.a = matrix_2x3[0];
    t.b = matrix_2x3[1];
    t.c = matrix_2x3[2];
    t.d = matrix_2x3[3];
    t.e = matrix_2x3[4];
    t.f = matrix_2x3[5];
    if (!isfinite(t.a) || !isfinite(t.b) || !isfinite(t.c) || !isfinite(t.d) || !isfinite(t.e) || !isfinite(t.f)) {
        return false;
    }

    if (other->count == 0u) {
        if (other->has_current) {
            float tx = 0.0f;
            float ty = 0.0f;
            fs_transform_apply_point(&t, other->current_x, other->current_y, &tx, &ty);
            return fs_path2d_move_to(path, tx, ty);
        }
        return true;
    }
    if (!other->segments) {
        return false;
    }
    if (!fs_path2d_ensure_capacity(path, path->count + other->count)) {
        return false;
    }

    for (uint32_t i = 0u; i < other->count; ++i) {
        FS_PathSegment seg = other->segments[i];
        fs_transform_apply_point(&t, seg.x0, seg.y0, &seg.x0, &seg.y0);
        if (seg.type == (uint8_t)FS_PATH_SEG_QUAD || seg.type == (uint8_t)FS_PATH_SEG_CUBIC) {
            fs_transform_apply_point(&t, seg.cx0, seg.cy0, &seg.cx0, &seg.cy0);
        }
        if (seg.type == (uint8_t)FS_PATH_SEG_CUBIC) {
            fs_transform_apply_point(&t, seg.cx1, seg.cy1, &seg.cx1, &seg.cy1);
        }
        fs_transform_apply_point(&t, seg.x1, seg.y1, &seg.x1, &seg.y1);
        path->segments[path->count++] = seg;
    }

    if (other->has_current) {
        float tx = 0.0f;
        float ty = 0.0f;
        fs_transform_apply_point(&t, other->current_x, other->current_y, &tx, &ty);
        path->has_current = true;
        path->current_x = tx;
        path->current_y = ty;
        if (other->has_subpath_start) {
            fs_transform_apply_point(&t, other->subpath_start_x, other->subpath_start_y, &tx, &ty);
            path->has_subpath_start = true;
            path->subpath_start_x = tx;
            path->subpath_start_y = ty;
        } else {
            path->has_subpath_start = false;
            path->subpath_start_x = 0.0f;
            path->subpath_start_y = 0.0f;
        }
    } else {
        const FS_PathSegment* last = &path->segments[path->count - 1u];
        path->has_current = true;
        path->current_x = last->x1;
        path->current_y = last->y1;
        path->has_subpath_start = true;
        path->subpath_start_x = last->x0;
        path->subpath_start_y = last->y0;
    }

    return true;
}

bool fs_clip_path2d(FS_Core* core, const FS_Path2D* path) {
    if (!core || !path) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    FS_PathStateBorrow saved;
    fs_path_state_borrow(st, &saved);
    fs_path_state_bind_path2d(st, path);
    const bool ok = fs_clip_path(core);
    fs_path_state_restore(st, &saved);
    return ok;
}

bool fs_path_fill_path2d(FS_Core* core, const FS_Path2D* path, uint32_t color) {
    if (!core || !path) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    FS_PathStateBorrow saved;
    fs_path_state_borrow(st, &saved);
    fs_path_state_bind_path2d(st, path);
    const bool ok = fs_path_fill(core, color);
    fs_path_state_restore(st, &saved);
    return ok;
}

bool fs_path_stroke_path2d(FS_Core* core, const FS_Path2D* path, float width, uint32_t color) {
    if (!core || !path) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    FS_PathStateBorrow saved;
    fs_path_state_borrow(st, &saved);
    fs_path_state_bind_path2d(st, path);
    const bool ok = fs_path_stroke(core, width, color);
    fs_path_state_restore(st, &saved);
    return ok;
}

bool fs_stroke(FS_Core* core, float width) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    const bool dynamic_stroke_style_color =
        (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_LINEAR_GRADIENT &&
         st->style_stroke_linear_gradient.stop_count >= 2u) ||
        (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_RADIAL_GRADIENT &&
         st->style_stroke_radial_gradient.stop_count >= 2u) ||
        (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_CONIC_GRADIENT &&
         st->style_stroke_conic_gradient.stop_count >= 2u) ||
        (st->style_stroke_paint_type == (uint8_t)FS_STYLE_PAINT_PATTERN &&
         st->style_stroke_pattern.handle.width > 0u &&
         st->style_stroke_pattern.handle.height > 0u);
    if (dynamic_stroke_style_color) {
        return fs_path_stroke_internal(core, width, st->style_stroke_color_rgba8, true);
    }
    return fs_path_stroke(core, width, st->style_stroke_color_rgba8);
}

bool fs_clip_path2d_with_fill_rule(FS_Core* core, const FS_Path2D* path, FS_FillRule fill_rule) {
    if (!core || !path) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    FS_PathStateBorrow saved;
    fs_path_state_borrow(st, &saved);
    fs_path_state_bind_path2d(st, path);
    const bool ok = fs_clip_path_with_fill_rule(core, fill_rule);
    fs_path_state_restore(st, &saved);
    return ok;
}

bool fs_is_point_in_path2d(FS_Core* core, const FS_Path2D* path, float x, float y) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    const FS_FillRule fill_rule =
        (st->style_fill_rule == (uint8_t)FS_FILL_RULE_EVENODD) ? FS_FILL_RULE_EVENODD : FS_FILL_RULE_NONZERO;
    return fs_is_point_in_path2d_with_fill_rule(core, path, x, y, fill_rule);
}

bool fs_is_point_in_path2d_with_fill_rule(FS_Core* core, const FS_Path2D* path, float x, float y, FS_FillRule fill_rule) {
    if (!core || !path || !isfinite(x) || !isfinite(y)) {
        return false;
    }
    if (fill_rule != FS_FILL_RULE_NONZERO && fill_rule != FS_FILL_RULE_EVENODD) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || !path->segments || path->count == 0u) {
        return false;
    }
    return fs_hit_test_fill_path_device(path->segments, path->count, &st->current_transform, x, y, fill_rule);
}

bool fs_is_point_in_stroke_path2d(FS_Core* core, const FS_Path2D* path, float x, float y) {
    if (!core || !path || !isfinite(x) || !isfinite(y)) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st || !path->segments || path->count == 0u) {
        return false;
    }
    const float stroke_width = fs_style_resolve_line_width(st, 0.0f);
    if (!isfinite(stroke_width) || stroke_width <= 0.0f) {
        return false;
    }
    return fs_hit_test_stroke_path_device(
        path->segments,
        path->count,
        &st->current_transform,
        st->style_line_cap,
        st->style_line_join,
        st->style_miter_limit,
        st->style_dash_segments,
        st->style_dash_count,
        st->style_dash_offset,
        stroke_width,
        x,
        y
    );
}


bool fs_cmd_line(FS_Core* core, float x0, float y0, float x1, float y1, float width, uint32_t color) {
    FS_InternalState* st = fs_state(core);
    const float resolved_width = fs_style_resolve_line_width(st, width);
    if (resolved_width <= 0.0f) {
        return false;
    }
    const uint8_t line_cap = st ? st->style_line_cap : (uint8_t)FS_LINE_CAP_ROUND;
    if (fs_style_has_dash(st)) {
        float dash_total = 0.0f;
        for (uint32_t i = 0u; i < st->style_dash_count; ++i) {
            dash_total += st->style_dash_segments[i];
        }
        if (dash_total > 1e-6f) {
            float phase = st->style_dash_offset;
            return fs_emit_dashed_line_segment(
                core,
                x0,
                y0,
                x1,
                y1,
                resolved_width,
                color,
                line_cap,
                st->style_dash_segments,
                st->style_dash_count,
                dash_total,
                &phase,
                0u
            );
        }
    }
    return fs_emit_styled_line_segment(core, x0, y0, x1, y1, resolved_width, color, line_cap);
}


bool fs_cmd_path_segment(FS_Core* core, float x0, float y0, float x1, float y1, float width, uint32_t color) {
    return fs_cmd_path_segment_with_flags(core, x0, y0, x1, y1, width, color, 0u);
}

static bool fs_cmd_circle_with_flags(FS_Core* core, float cx, float cy, float radius, uint32_t color, uint32_t user_flags) {
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    if (core && st && t && fs_transform_requires_oriented_quad(t)) {
        return fs_cmd_ellipse_compute_coverage_fill_with_flags(core, cx, cy, radius, radius, color, user_flags);
    }
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = cx;
    cmd.p0[1] = cy;
    cmd.flags = user_flags & FS_RENDER_FLAG_USER_MASK;
    cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    cmd.scalar = radius;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_CIRCLE;
    return fs_push_command(core, &cmd);
}

bool fs_cmd_circle(FS_Core* core, float cx, float cy, float radius, uint32_t color) {
    return fs_cmd_circle_with_flags(core, cx, cy, radius, color, 0u);
}

bool fs_cmd_ellipse(FS_Core* core, float cx, float cy, float radius_x, float radius_y, uint32_t color) {
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    if (core && st && t && fs_transform_requires_oriented_quad(t)) {
        return fs_cmd_ellipse_compute_coverage_fill(core, cx, cy, radius_x, radius_y, color);
    }
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = cx;
    cmd.p0[1] = cy;
    cmd.p0[2] = radius_x;
    cmd.p0[3] = radius_y;
    cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_ELLIPSE;
    return fs_push_command(core, &cmd);
}

bool fs_cmd_arc(FS_Core* core, float cx, float cy, float radius, float thickness, float start_angle, float end_angle, uint32_t color) {
    FS_InternalState* st = fs_state(core);
    const FS_Transform2D* t = st ? &st->current_transform : NULL;
    if (core && st && t && fs_transform_requires_oriented_quad(t)) {
        return fs_cmd_arc_compute_coverage_stroke(core, cx, cy, radius, thickness, start_angle, end_angle, color);
    }
    float theta = 0.0f;
    if (t) {
        theta = atan2f(t->b, t->a);
    }
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = cx;
    cmd.p0[1] = cy;
    cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    cmd.p1[0] = start_angle + theta;
    cmd.p1[1] = end_angle + theta;
    cmd.p1[3] = thickness;
    cmd.scalar = radius;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_ARC;
    return fs_push_command(core, &cmd);
}

static bool fs_cmd_bezier_quad_with_flags(
    FS_Core* core,
    float x0,
    float y0,
    float cx,
    float cy,
    float x1,
    float y1,
    float width,
    uint32_t color,
    uint32_t user_flags
) {
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = x0;
    cmd.p0[1] = y0;
    cmd.p0[2] = cx;
    cmd.p0[3] = cy;
    cmd.p1[0] = x1;
    cmd.p1[1] = y1;
    cmd.flags = user_flags & FS_RENDER_FLAG_USER_MASK;
    cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    cmd.scalar = width;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_BEZIER_QUAD;
    return fs_push_command(core, &cmd);
}

bool fs_cmd_bezier_quad(FS_Core* core, float x0, float y0, float cx, float cy, float x1, float y1, float width, uint32_t color) {
    return fs_cmd_bezier_quad_with_flags(core, x0, y0, cx, cy, x1, y1, width, color, 0u);
}

static bool fs_cmd_bezier_cubic_with_flags(
    FS_Core* core,
    float x0,
    float y0,
    float cx0,
    float cy0,
    float cx1,
    float cy1,
    float x1,
    float y1,
    float width,
    uint32_t color,
    uint32_t user_flags
) {
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = x0;
    cmd.p0[1] = y0;
    cmd.p0[2] = cx0;
    cmd.p0[3] = cy0;
    cmd.p1[0] = cx1;
    cmd.p1[1] = cy1;
    cmd.p1[2] = x1;
    cmd.p1[3] = y1;
    cmd.flags = user_flags & FS_RENDER_FLAG_USER_MASK;
    cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    cmd.scalar = width;
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_BEZIER_CUBIC;
    return fs_push_command(core, &cmd);
}

bool fs_cmd_bezier_cubic(FS_Core* core, float x0, float y0, float cx0, float cy0, float cx1, float cy1, float x1, float y1, float width, uint32_t color) {
    return fs_cmd_bezier_cubic_with_flags(core, x0, y0, cx0, cy0, cx1, cy1, x1, y1, width, color, 0u);
}


bool fs_cmd_triangle(FS_Core* core, float x0, float y0, float x1, float y1, float x2, float y2, uint32_t color) {
    return fs_cmd_triangle_with_edge_mask(core, x0, y0, x1, y1, x2, y2, color, FS_TRI_FLAG_AA_ALL, 0u);
}
