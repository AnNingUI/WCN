#ifndef WCN_FULLSTACK_CORE_GPU_LAYOUT_H
#define WCN_FULLSTACK_CORE_GPU_LAYOUT_H

#include "fullstack_core.h"

#define FS_RENDER_PIPELINE_COUNT 11u
#define FS_CLIP_MASK_LAYERS 64u

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

_Static_assert(sizeof(FS_Command) == 128u, "FS_Command must match WGSL Command stride (128 bytes)");
_Static_assert(sizeof(FS_CommandStateGPU) == 112u, "FS_CommandStateGPU must match WGSL CommandState stride (112 bytes)");
_Static_assert(sizeof(FS_ClipEdgeGPU) == 16u, "FS_ClipEdgeGPU must match WGSL ClipEdge stride (16 bytes)");
_Static_assert(sizeof(FS_ClipJobGPU) == 64u, "FS_ClipJobGPU must match WGSL ClipJob stride (64 bytes)");
_Static_assert(sizeof(FS_ClipJobTransformGPU) == 32u, "FS_ClipJobTransformGPU must match WGSL ClipJobTransform stride (32 bytes)");
_Static_assert(sizeof(FS_ClipDispatchUniforms) == 32u, "FS_ClipDispatchUniforms must match WGSL ClipDispatch layout");
_Static_assert(
    sizeof(FS_ClipLayerUniforms) == (size_t)FS_CLIP_MASK_LAYERS * 5u * sizeof(uint32_t),
    "FS_ClipLayerUniforms must match WGSL ClipLayers layout"
);

#endif
