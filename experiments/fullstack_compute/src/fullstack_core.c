#include "fullstack_core_private.h"
#include "fullstack_core_debug.h"
#include "fullstack_shaders.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



// ========== MISSING FORWARD DECLARATIONS (restored) ==========
static bool fs_ensure_transform_stack_capacity(FS_InternalState* st, uint32_t required);
bool fs_push_command(FS_Core* core, const FS_Command* cmd);
static void fs_clip_diag_reset_frame(FS_Core* core);
static bool fs_emit_styled_line_segment_with_flags(FS_Core* core, float x0, float y0, float x1, float y1, float width, uint32_t color, uint8_t line_cap, uint32_t extra_line_flags);
static bool fs_cmd_path_segment_with_flags(FS_Core* core, float x0, float y0, float x1, float y1, float width, uint32_t color, uint32_t user_flags);
static bool fs_cmd_triangle_with_edge_mask(FS_Core* core, float x0, float y0, float x1, float y1, float x2, float y2, uint32_t color, uint32_t tri_aa_mask, uint32_t extra_user_flags);
static bool fs_cmd_circle_with_flags(FS_Core* core, float cx, float cy, float radius, uint32_t color, uint32_t user_flags);
static bool fs_cmd_rect_with_flags(FS_Core* core, float x, float y, float w, float h, float radius, uint32_t color, uint32_t extra_flags);
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
static bool fs_cmd_polygon_compute_coverage_fill_with_flags(FS_Core* core, const float* xy, uint32_t point_count, uint32_t color, uint32_t extra_flags);
static bool fs_cmd_ellipse_compute_coverage_stroke_with_flags(FS_Core* core, float cx, float cy, float rx, float ry, float width, uint32_t color, uint32_t extra_flags);
// ============================================================

// ========== MISSING FUNCTION DEFINITIONS (restored) ==========

FS_InternalState* fs_state(FS_Core* core) {
    return core ? core->internal_state : NULL;
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

bool fs_upload_glyph_with_mips(
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
    if (core->glyph_scratch_rgba_capacity[0] < rgba_size) {
        uint8_t* grown = (uint8_t*)realloc(core->glyph_scratch_rgba[0], rgba_size);
        if (!grown) {
            return false;
        }
        core->glyph_scratch_rgba[0] = grown;
        core->glyph_scratch_rgba_capacity[0] = rgba_size;
    }
    uint8_t* rgba = core->glyph_scratch_rgba[0];
    for (size_t i = 0u; i < (size_t)width * (size_t)height; ++i) {
        rgba[i * 4u + 0u] = pixels[i];
        rgba[i * 4u + 1u] = pixels[i];
        rgba[i * 4u + 2u] = pixels[i];
        rgba[i * 4u + 3u] = pixels[i];  // Shader reads sampled.a for SDF value
    }
    (void)sdf_onedge;
    bool ok = fs_queue_write_texture_2d(core, core->glyph_atlas_texture, 0u, x, y, width, height, rgba, 4u);
    return ok;
}


bool fs_polygon_compact_in_place(FS_Point2* points, uint32_t* io_count) {
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

uint32_t fs_quantize_font_size(float font_px) {
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
        .label = { .data = "FS Compute Bind Group", .length = 21 },
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
        .label = { .data = "FS Image Atlas", .length = 14 },
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
        .label = { .data = "FS Image Atlas View", .length = 19 },
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
        .label = { .data = "FS Image Atlas Sampler", .length = 22 },
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
        .label = { .data = "FS Glyph Atlas", .length = 14 },
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
        .label = { .data = "FS Glyph Atlas View", .length = 19 },
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
        .label = { .data = "FS Glyph Atlas Sampler", .length = 22 },
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
        .label = { .data = "FS Clip Mask", .length = 12 },
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
        .label = { .data = "FS Clip Mask View", .length = 17 },
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
        .label = { .data = "FS Clip Mask Sampler", .length = 20 },
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
        .label = { .data = "FS MSAA Color", .length = 13 },
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
        .label = { .data = "FS MSAA Color View", .length = 18 },
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
        .label = { .data = "FS Render Bind Group", .length = 20 },
        .layout = core->render_bgl,
        .entryCount = 9,
        .entries = entries
    };
    core->render_bg = wgpuDeviceCreateBindGroup(core->device, &desc);
    return core->render_bg != NULL;
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
        .label = { .data = "FS Compute BGL", .length = 14 },
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
        .label = { .data = "FS Render BGL", .length = 13 },
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
        .label = { .data = "FS Compute Pipeline Layout", .length = 26 },
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
        .label = { .data = "FS Compute Pipeline", .length = 19 },
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
        .label = { .data = "FS Clip Compute BGL", .length = 19 },
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
        .label = { .data = "FS Clip Compute Pipeline Layout", .length = 31 },
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
        .label = { .data = "FS Clip Compute Pipeline", .length = 24 },
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
        .label = { .data = "FS Clip Edge Transform BGL", .length = 26 },
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
        .label = { .data = "FS Clip Edge Transform Pipeline Layout", .length = 38 },
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
        .label = { .data = "FS Clip Edge Transform Pipeline", .length = 31 },
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
        {.shaderLocation = 9, .format = WGPUVertexFormat_Float32x4, .offset = 96},
        {.shaderLocation = 10, .format = WGPUVertexFormat_Float32x4, .offset = 112}
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
        // Use RGBA8Unorm to match scene_texture (always used as render target).
        // presentation_pipeline handles canvas swap chain format (BGRA8UnormSrgb).
        .format = WGPUTextureFormat_RGBA8Unorm,
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
        .label = { .data = "FS Render Pipeline Layout", .length = 25 },
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
        const uint32_t sample_count = 1u;  // scene_texture has sampleCount=1
        WGPURenderPipelineDescriptor render_pipe_desc = {
            .nextInChain = NULL,
            .label = { .data = "FS Render Pipeline", .length = 18 },
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

    free(core->clip_path_edges_scratch);
    core->clip_path_edges_scratch = NULL;
    core->clip_path_edges_scratch_capacity = 0u;
    core->clip_path_edges_scratch_count = 0u;

    free(core->clip_layer_protected_scratch);
    core->clip_layer_protected_scratch = NULL;
    core->clip_layer_protected_scratch_capacity = 0u;

    free(core->clip_dispatch_valid_jobs_scratch);
    core->clip_dispatch_valid_jobs_scratch = NULL;
    core->clip_dispatch_valid_jobs_scratch_capacity = 0u;
    free(core->clip_dispatch_valid_xforms_scratch);
    core->clip_dispatch_valid_xforms_scratch = NULL;
    core->clip_dispatch_valid_xforms_scratch_capacity = 0u;
    free(core->clip_dispatch_valid_bucket_ids_scratch);
    core->clip_dispatch_valid_bucket_ids_scratch = NULL;
    core->clip_dispatch_valid_bucket_ids_scratch_capacity = 0u;
    free(core->clip_dispatch_ordered_jobs_scratch);
    core->clip_dispatch_ordered_jobs_scratch = NULL;
    core->clip_dispatch_ordered_jobs_scratch_capacity = 0u;
    free(core->clip_dispatch_ordered_xforms_scratch);
    core->clip_dispatch_ordered_xforms_scratch = NULL;
    core->clip_dispatch_ordered_xforms_scratch_capacity = 0u;

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

bool fs_push_command(FS_Core* core, const FS_Command* cmd) {
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

struct FS_EffectResources* fs_core_get_effects_resources(FS_Core* core) {
    if (!core) {
        return NULL;
    }
    return core->effects;
}

void fs_core_set_effects_resources(FS_Core* core, struct FS_EffectResources* effects) {
    if (!core) {
        return;
    }
    core->effects = effects;
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
    core->canvas_image_data_handle_valid = 0u;
    // Initialize effects subsystem (Gaussian blur, filter pipeline, shadow)
    if (!fs_effects_init(core)) {
        fs_release_resources(core);
        return false;
    }
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
    fs_effects_destroy(core);
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
    // Resize effects subsystem textures and recreate presentation pipeline
    if (!fs_effects_resize(core, core->width, core->height)) {
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

    if (core->clip_dispatch_valid_jobs_scratch_capacity < valid_job_count) {
        FS_ClipJobGPU* grown = (FS_ClipJobGPU*)realloc(
            core->clip_dispatch_valid_jobs_scratch, valid_job_count * sizeof(FS_ClipJobGPU));
        if (!grown) {
            return false;
        }
        core->clip_dispatch_valid_jobs_scratch = grown;
        core->clip_dispatch_valid_jobs_scratch_capacity = valid_job_count;
    }
    if (core->clip_dispatch_valid_xforms_scratch_capacity < valid_job_count) {
        FS_ClipJobTransformGPU* grown = (FS_ClipJobTransformGPU*)realloc(
            core->clip_dispatch_valid_xforms_scratch, valid_job_count * sizeof(FS_ClipJobTransformGPU));
        if (!grown) {
            return false;
        }
        core->clip_dispatch_valid_xforms_scratch = grown;
        core->clip_dispatch_valid_xforms_scratch_capacity = valid_job_count;
    }
    if (core->clip_dispatch_valid_bucket_ids_scratch_capacity < valid_job_count) {
        uint8_t* grown = (uint8_t*)realloc(
            core->clip_dispatch_valid_bucket_ids_scratch, valid_job_count * sizeof(uint8_t));
        if (!grown) {
            return false;
        }
        core->clip_dispatch_valid_bucket_ids_scratch = grown;
        core->clip_dispatch_valid_bucket_ids_scratch_capacity = valid_job_count;
    }
    FS_ClipJobGPU* valid_jobs = core->clip_dispatch_valid_jobs_scratch;
    FS_ClipJobTransformGPU* valid_xforms = core->clip_dispatch_valid_xforms_scratch;
    uint8_t* valid_bucket_ids = core->clip_dispatch_valid_bucket_ids_scratch;

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
        core->clip_edge_count = 0u;
        core->clip_job_count = 0u;
        return true;
    }
    if (!fs_ensure_clip_job_gpu_capacity(core, total_slots)) {
        return false;
    }
    if (!fs_ensure_clip_job_transform_gpu_capacity(core, total_slots)) {
        return false;
    }

    if (core->clip_dispatch_ordered_jobs_scratch_capacity < total_slots) {
        FS_ClipJobGPU* grown = (FS_ClipJobGPU*)realloc(
            core->clip_dispatch_ordered_jobs_scratch, total_slots * sizeof(FS_ClipJobGPU));
        if (!grown) {
            return false;
        }
        core->clip_dispatch_ordered_jobs_scratch = grown;
        core->clip_dispatch_ordered_jobs_scratch_capacity = total_slots;
    }
    if (core->clip_dispatch_ordered_xforms_scratch_capacity < total_slots) {
        FS_ClipJobTransformGPU* grown = (FS_ClipJobTransformGPU*)realloc(
            core->clip_dispatch_ordered_xforms_scratch, total_slots * sizeof(FS_ClipJobTransformGPU));
        if (!grown) {
            return false;
        }
        core->clip_dispatch_ordered_xforms_scratch = grown;
        core->clip_dispatch_ordered_xforms_scratch_capacity = total_slots;
    }
    FS_ClipJobGPU* ordered_jobs = core->clip_dispatch_ordered_jobs_scratch;
    FS_ClipJobTransformGPU* ordered_xforms = core->clip_dispatch_ordered_xforms_scratch;
    memset(ordered_jobs, 0, total_slots * sizeof(FS_ClipJobGPU));
    memset(ordered_xforms, 0, total_slots * sizeof(FS_ClipJobTransformGPU));
    for (size_t i = 0u; i < valid_job_count; ++i) {
        const uint8_t b = valid_bucket_ids[i];
        const size_t dst = bucket_cursor[b]++;
        ordered_jobs[dst] = valid_jobs[i];
        ordered_xforms[dst] = valid_xforms[i];
    }
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
        return false;
    }

    WGPUComputePassDescriptor edge_transform_pass_desc = {
        .nextInChain = NULL,
        .label = { .data = "FS Clip Edge Transform Pass", .length = 27 },
        .timestampWrites = NULL
    };
    WGPUComputePassEncoder edge_transform_pass =
        wgpuCommandEncoderBeginComputePass(encoder, &edge_transform_pass_desc);
    if (!edge_transform_pass) {
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
    wgpuComputePassEncoderRelease(edge_transform_pass);
    if (!edge_transform_ok) {
        return false;
    }

    WGPUComputePassDescriptor clip_pass_desc = {
        .nextInChain = NULL,
        .label = { .data = "FS Clip Compute Pass", .length = 20 },
        .timestampWrites = NULL
    };
    WGPUComputePassEncoder clip_pass = wgpuCommandEncoderBeginComputePass(encoder, &clip_pass_desc);
    if (!clip_pass) {
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
    wgpuComputePassEncoderRelease(clip_pass);
    core->clip_dispatch_batches_this_frame += dispatch_batches;
    core->clip_dispatch_pixels_estimated_this_frame += estimated_pixels;
    if (estimated_pixels > ideal_pixels) {
        core->clip_dispatch_pixels_waste_this_frame += (estimated_pixels - ideal_pixels);
    }

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
            .label = { .data = "FS Compute Pass", .length = 15 },
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
        wgpuComputePassEncoderRelease(compute_pass);
    }

    // Always render to scene_texture (RGBA8Unorm) — the main render pipeline is created
    // with scene_texture's format. Use presentation_pipeline to copy to canvas swap chain.
    FS_EffectResources* fx = fs_core_get_effects_resources(core);
    FS_FilterChain* active_chain = NULL;
    bool effects_active = false;
    if (fx && fx->enabled && fx->scene_view) {
        FS_InternalState* st = fs_state(core);
        if (st && st->filter_chain && st->filter_chain->head) {
            active_chain = st->filter_chain;
            effects_active = true;
        }
    }
    (void)effects_active;

    // Render to scene_texture (RGBA8Unorm) always — matches main pipeline format
    WGPUTextureView scene_view = fx ? fx->scene_view : NULL;
    WGPURenderPassColorAttachment color = {
        .view = scene_view ? scene_view : target_view,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        .resolveTarget = NULL,
        .loadOp = WGPULoadOp_Clear,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = {.r = clear_r, .g = clear_g, .b = clear_b, .a = clear_a}
    };
    WGPURenderPassDescriptor render_desc = {
        .nextInChain = NULL,
        .label = { .data = "FS Render Pass", .length = 14 },
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
    wgpuRenderPassEncoderSetViewport(pass, 0.0f, 0.0f, (float)core->width, (float)core->height, 0.0f, 1.0f);
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
                wgpuRenderPassEncoderRelease(pass);
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
    wgpuRenderPassEncoderRelease(pass);

    // Execute filter pipeline if effects are active
    // Filter pipeline reads from scene_texture (RGBA8Unorm) -> applies filters via ping-pong -> writes back to scene_texture
    // Then presentation_pipeline copies scene_texture -> canvas (with format conversion)
    if (effects_active && active_chain && fx && fx->scene_texture) {
        if (!fs_filter_chain_execute(core, encoder, fx->scene_texture, target_view, active_chain)) {
            fprintf(stderr, "[FS] Filter chain execution failed\n");
        }
        // Present the filtered result: presentation_pipeline samples scene_texture -> canvas
        if (fx->presentation_pipeline && fx->presentation_scene_bg && target_texture) {
            WGPURenderPassColorAttachment pres_att = {
                .view = target_view,
                .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
                .resolveTarget = NULL,
                .loadOp = WGPULoadOp_Clear,
                .storeOp = WGPUStoreOp_Store,
                .clearValue = {0.0f, 0.0f, 0.0f, 0.0f}
            };
            WGPURenderPassDescriptor pres_desc = {
                .nextInChain = NULL,
                .label = { .data = "FS Presentation Copy", .length = 20 },
                .colorAttachmentCount = 1,
                .colorAttachments = &pres_att,
                .depthStencilAttachment = NULL,
                .occlusionQuerySet = NULL,
                .timestampWrites = NULL
            };
            WGPURenderPassEncoder pres_pass = wgpuCommandEncoderBeginRenderPass(encoder, &pres_desc);
            if (pres_pass) {
                wgpuRenderPassEncoderSetViewport(pres_pass, 0.0f, 0.0f, (float)core->width, (float)core->height, 0.0f, 1.0f);
                wgpuRenderPassEncoderSetPipeline(pres_pass, fx->presentation_pipeline);
                wgpuRenderPassEncoderSetBindGroup(pres_pass, 0, fx->presentation_scene_bg, 0, NULL);
                wgpuRenderPassEncoderDraw(pres_pass, 3, 1, 0, 0);
                wgpuRenderPassEncoderEnd(pres_pass);
                wgpuRenderPassEncoderRelease(pres_pass);
            }
        }
    } else {
        // No effects: copy scene_texture -> canvas using presentation_pipeline
        if (fx && fx->presentation_pipeline && fx->presentation_scene_bg && target_texture) {
            WGPURenderPassColorAttachment pres_att = {
                .view = target_view,
                .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
                .resolveTarget = NULL,
                .loadOp = WGPULoadOp_Clear,
                .storeOp = WGPUStoreOp_Store,
                .clearValue = {0.0f, 0.0f, 0.0f, 0.0f}
            };
            WGPURenderPassDescriptor pres_desc = {
                .nextInChain = NULL,
                .label = { .data = "FS Presentation Copy", .length = 20 },
                .colorAttachmentCount = 1,
                .colorAttachments = &pres_att,
                .depthStencilAttachment = NULL,
                .occlusionQuerySet = NULL,
                .timestampWrites = NULL
            };
            WGPURenderPassEncoder pres_pass = wgpuCommandEncoderBeginRenderPass(encoder, &pres_desc);
            if (pres_pass) {
                wgpuRenderPassEncoderSetViewport(pres_pass, 0.0f, 0.0f, (float)core->width, (float)core->height, 0.0f, 1.0f);
                wgpuRenderPassEncoderSetPipeline(pres_pass, fx->presentation_pipeline);
                wgpuRenderPassEncoderSetBindGroup(pres_pass, 0, fx->presentation_scene_bg, 0, NULL);
                wgpuRenderPassEncoderDraw(pres_pass, 3, 1, 0, 0);
                wgpuRenderPassEncoderEnd(pres_pass);
                wgpuRenderPassEncoderRelease(pres_pass);
            }
        }
    }

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

    FS_ImageHandle handle;
    bool ok = fs_ensure_canvas_image_data_handle(core, width, height, &handle);
    if (!ok) {
        return false;
    }
    ok = fs_queue_write_texture_2d(
        core,
        core->image_atlas_texture,
        handle.layer,
        handle.atlas_x,
        handle.atlas_y,
        width,
        height,
        rgba_pixels,
        4u
    );
    if (!ok) {
        return false;
    }

    const float handle_uv_w = handle.uv_max[0] - handle.uv_min[0];
    const float handle_uv_h = handle.uv_max[1] - handle.uv_min[1];
    const float uv_x = handle.uv_min[0] + handle_uv_w * ((float)src_x / (float)width);
    const float uv_y = handle.uv_min[1] + handle_uv_h * ((float)src_y / (float)height);
    const float uv_w = handle_uv_w * ((float)clip_w / (float)width);
    const float uv_h = handle_uv_h * ((float)clip_h / (float)height);
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

bool fs_core_load_font_memory(FS_Core* core, const uint8_t* data, size_t size) {
    if (!core || !data || size == 0) {
        return false;
    }
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    if (!st->font_backend || !st->font_backend->load_font_memory) {
        return false;
    }
    if (st->font_count >= FS_MAX_FONT_FALLBACKS) {
        return false;
    }
    void* loaded = st->font_backend->load_font_memory(data, size);
    if (!loaded) {
        return false;
    }
    st->fonts[st->font_count++] = loaded;
    return true;
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

    if (core->clip_layer_protected_scratch_capacity < layer_count) {
        uint8_t* grown = (uint8_t*)realloc(core->clip_layer_protected_scratch, (size_t)layer_count);
        if (!grown) {
            return false;
        }
        core->clip_layer_protected_scratch = grown;
        core->clip_layer_protected_scratch_capacity = layer_count;
    }
    uint8_t* protected_layers = core->clip_layer_protected_scratch;
    memset(protected_layers, 0, (size_t)layer_count);

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

    FS_ClipEdge* edges = core->clip_path_edges_scratch;
    uint32_t edge_count = 0u;
    uint32_t edge_capacity = (uint32_t)core->clip_path_edges_scratch_capacity;
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
                        core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
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
                    core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
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
                        core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
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
                        core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
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
                core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
                fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
                return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_EDGE_ALLOC, edge_count);
            }
        }

        if (!fs_clip_compute_path_device_bounds(st, &min_x, &min_y, &max_x, &max_y)) {
            core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
            fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
            return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_INVALID_BOUNDS, edge_count);
        }
        has_bounds = true;
    }

    if (!has_bounds || (edge_count == 0u && !analytic_round_rect)) {
        core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
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
        core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
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
        core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
        fs_clip_release_uncommitted_layer(core, layer, layer_was_fresh, previous_next_layer);
        return fs_clip_apply_path_aabb_fallback(core, st, FS_CLIP_FAILURE_INVALID_BOUNDS, edge_count);
    }

    if (!analytic_round_rect) {
        if (!fs_ensure_clip_edge_cpu_capacity(core, edge_offset + edge_count)) {
            core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
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
        core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;
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

    core->clip_path_edges_scratch = edges, core->clip_path_edges_scratch_capacity = edge_capacity;

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

static void fs_pack_round_rect_radii(FS_Command* cmd, const FS_RoundRectRadii* radii) {
    cmd->p1[0] = radii->top_left.x;
    cmd->p1[1] = radii->top_right.x;
    cmd->p1[2] = radii->bottom_right.x;
    cmd->p1[3] = radii->bottom_left.x;
    cmd->p2[0] = radii->top_left.y;
    cmd->p2[1] = radii->top_right.y;
    cmd->p2[2] = radii->bottom_right.y;
    cmd->p2[3] = radii->bottom_left.y;
}

static float fs_round_rect_min_transform_scale(const FS_Transform2D* t) {
    if (!t) return 1.0f;
    const float m00 = t->a * t->a + t->b * t->b;
    const float m11 = t->c * t->c + t->d * t->d;
    const float m01 = t->a * t->c + t->b * t->d;
    const float trace = m00 + m11;
    const float delta = m00 - m11;
    const float discriminant = sqrtf(fmaxf(delta * delta + 4.0f * m01 * m01, 0.0f));
    const float lambda_min = 0.5f * (trace - discriminant);
    return sqrtf(fmaxf(lambda_min, 1e-8f));
}

static bool fs_cmd_round_rect_with_flags(
    FS_Core* core, float x, float y, float w, float h,
    const FS_RoundRectRadii* radii, FS_CornerProfile profile,
    uint32_t color, uint32_t extra_flags
) {
    FS_InternalState* st = fs_state(core);
    if (!core || !st) return false;
    FS_NormalizedRoundRect rr;
    if (!fs_normalize_round_rect(x, y, w, h, radii, &rr)) return false;

    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = rr.x;
    cmd.p0[1] = rr.y;
    cmd.p0[2] = rr.w;
    cmd.p0[3] = rr.h;
    fs_pack_round_rect_radii(&cmd, &rr.radii);
    cmd.flags = extra_flags;
    if (profile == FS_CORNER_PROFILE_CONTINUOUS) {
        cmd.flags |= FS_RENDER_FLAG_CONTINUOUS_CORNER;
    }
    const FS_Transform2D* t = &st->current_transform;
    if (fs_transform_requires_oriented_quad(t)) {
        const float min_scale = fs_round_rect_min_transform_scale(t);
        const float pad = 1.5f / min_scale;
        fs_command_set_oriented_quad_from_rect(&cmd, t,
            rr.x - pad, rr.y - pad, rr.w + pad * 2.0f, rr.h + pad * 2.0f);
        cmd.flags |= FS_RENDER_FLAG_ORIENTED_QUAD;
    } else {
        cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    }
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_RECT;
    return fs_push_command(core, &cmd);
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
    const FS_RoundRectRadii radii = fs_round_rect_uniform_radii(radius, radius);
    return fs_cmd_round_rect_with_flags(core, x, y, w, h, &radii,
                                        FS_CORNER_PROFILE_ROUND, color, extra_flags);
}

bool fs_cmd_round_rect(FS_Core* core, float x, float y, float w, float h,
                       const FS_RoundRectRadii* radii, FS_CornerProfile profile,
                       uint32_t color) {
    return fs_cmd_round_rect_with_flags(core, x, y, w, h, radii, profile, color, 0u);
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
    const FS_RoundRectRadii radii = fs_round_rect_uniform_radii(radius, radius);
    return fs_cmd_round_rect_stroke(core, x, y, w, h, &radii,
                                    FS_CORNER_PROFILE_ROUND, stroke_width, color);
}

bool fs_cmd_round_rect_stroke(
    FS_Core* core, float x, float y, float w, float h,
    const FS_RoundRectRadii* radii, FS_CornerProfile profile,
    float stroke_width, uint32_t color
) {
    FS_InternalState* st = fs_state(core);
    if (!core || !st || !isfinite(stroke_width) || stroke_width <= 0.0f) return false;
    FS_NormalizedRoundRect rr;
    if (!fs_normalize_round_rect(x, y, w, h, radii, &rr)) return false;
    FS_Command cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.p0[0] = rr.x;
    cmd.p0[1] = rr.y;
    cmd.p0[2] = rr.w;
    cmd.p0[3] = rr.h;
    fs_pack_round_rect_radii(&cmd, &rr.radii);
    cmd.scalar = stroke_width;
    if (profile == FS_CORNER_PROFILE_CONTINUOUS) {
        cmd.flags |= FS_RENDER_FLAG_CONTINUOUS_CORNER;
    }
    const FS_Transform2D* t = &st->current_transform;
    if (fs_transform_requires_oriented_quad(t)) {
        const float min_scale = fs_round_rect_min_transform_scale(t);
        const float pad = stroke_width * 0.5f + 1.5f / min_scale;
        fs_command_set_oriented_quad_from_rect(&cmd, t,
            rr.x - pad, rr.y - pad, rr.w + pad * 2.0f, rr.h + pad * 2.0f);
        cmd.flags |= FS_RENDER_FLAG_ORIENTED_QUAD;
    } else {
        cmd.flags |= FS_RENDER_FLAG_LOCAL_SPACE;
    }
    cmd.color_rgba8 = color;
    cmd.type = FS_CMD_RECT_STROKE;
    return fs_push_command(core, &cmd);
}

bool fs_fill_rect(FS_Core* core, float x, float y, float w, float h, float radius) {
    const FS_RoundRectRadii radii = fs_round_rect_uniform_radii(radius, radius);
    return fs_fill_round_rect(core, x, y, w, h, &radii, FS_CORNER_PROFILE_ROUND);
}

bool fs_fill_round_rect(FS_Core* core, float x, float y, float w, float h,
                        const FS_RoundRectRadii* radii,
                        FS_CornerProfile profile) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    FS_NormalizedRoundRect rr;
    if (!fs_normalize_round_rect(x, y, w, h, radii, &rr)) return false;
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
        return fs_cmd_round_rect(core, rr.x, rr.y, rr.w, rr.h,
                                 &rr.radii, profile, st->style_fill_color_rgba8);
    }
    const FS_RoundRadius* rv[4] = {
        &rr.radii.top_left, &rr.radii.top_right,
        &rr.radii.bottom_right, &rr.radii.bottom_left
    };
    bool has_radius = false;
    for (uint32_t i = 0u; i < 4u; ++i) {
        has_radius = has_radius || rv[i]->x > 1e-6f || rv[i]->y > 1e-6f;
    }
    if (!has_radius) {
        if (fill_linear) {
            return fs_draw_linear_gradient_rect_cells(core, rr.x, rr.y, rr.w, rr.h, &st->style_fill_linear_gradient);
        }
        if (fill_radial) {
            return fs_draw_radial_gradient_rect_cells(core, rr.x, rr.y, rr.w, rr.h, &st->style_fill_radial_gradient);
        }
        if (fill_conic) {
            return fs_draw_conic_gradient_rect_cells(core, rr.x, rr.y, rr.w, rr.h, &st->style_fill_conic_gradient);
        }
        return fs_cmd_rect_with_flags(
            core,
            rr.x,
            rr.y,
            rr.w,
            rr.h,
            0.0f,
            0xFFFFFFFFu,
            FS_RENDER_FLAG_PATTERN_SHADE | FS_RENDER_FLAG_PATTERN_FILL_HINT
        );
    }

    FS_Path2D* clip_rr = fs_path2d_create();
    if (!clip_rr) {
        const float cx = rr.x + rr.w * 0.5f;
        const float cy = rr.y + rr.h * 0.5f;
        return fs_cmd_round_rect(core, rr.x, rr.y, rr.w, rr.h, &rr.radii, profile,
                                 fs_style_resolve_fill_color_at(st, cx, cy));
    }
    fs_state_save(core);
    const bool clip_ok = fs_path2d_round_rect_radii(clip_rr, rr.x, rr.y, rr.w, rr.h,
                                                     &rr.radii, profile) && fs_clip_path2d(core, clip_rr);
    bool draw_ok = false;
    if (clip_ok) {
        if (fill_linear) {
            draw_ok = fs_draw_linear_gradient_rect_cells(core, rr.x, rr.y, rr.w, rr.h, &st->style_fill_linear_gradient);
        } else if (fill_radial) {
            draw_ok = fs_draw_radial_gradient_rect_cells(core, rr.x, rr.y, rr.w, rr.h, &st->style_fill_radial_gradient);
        } else if (fill_conic) {
            draw_ok = fs_draw_conic_gradient_rect_cells(core, rr.x, rr.y, rr.w, rr.h, &st->style_fill_conic_gradient);
        } else {
            draw_ok = fs_cmd_rect_with_flags(
                core,
                rr.x,
                rr.y,
                rr.w,
                rr.h,
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
    const FS_RoundRectRadii radii = fs_round_rect_uniform_radii(radius, radius);
    return fs_stroke_round_rect(core, x, y, w, h, &radii,
                                FS_CORNER_PROFILE_ROUND, stroke_width);
}

bool fs_stroke_round_rect(FS_Core* core, float x, float y, float w, float h,
                          const FS_RoundRectRadii* radii,
                          FS_CornerProfile profile, float stroke_width) {
    FS_InternalState* st = fs_state(core);
    if (!st) {
        return false;
    }
    const float cx = x + w * 0.5f;
    const float cy = y + h * 0.5f;
    const uint32_t color = fs_style_resolve_stroke_color_at(st, cx, cy);
    return fs_cmd_round_rect_stroke(core, x, y, w, h, radii, profile, stroke_width, color);
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

    uint32_t* hole_indices = (uint32_t*)malloc((size_t)contour_count * sizeof(uint32_t));
    uint32_t* hole_right_indices = (uint32_t*)malloc((size_t)contour_count * sizeof(uint32_t));
    float* hole_right_x = (float*)malloc((size_t)contour_count * sizeof(float));
    FS_Point2* poly = NULL;
    uint32_t poly_capacity = 0u;

    for (uint32_t i = 0u; i < contour_count && ok; ++i) {
        if (contours[i].is_hole) {
            continue;
        }

        uint32_t poly_count = 0u;
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
        if (hole_local_count > 0u) {
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
        if (ok) {
            ok = fs_polygon_compact_in_place(poly, &poly_count);
        }

        if (ok) {
            ok = fs_emit_fill_triangles_ear_clip(core, poly, poly_count, color, hole_local_count == 0u);
        }
    }

    free(hole_indices);
    free(hole_right_indices);
    free(hole_right_x);
    free(poly);

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
