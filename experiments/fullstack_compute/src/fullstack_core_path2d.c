#include "fullstack_core_private.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static FS_InternalState* fs_path_core_state(FS_Core* core) {
    return core ? (FS_InternalState*)core->internal_state : NULL;
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
    FS_PathSegment* grown = (FS_PathSegment*)realloc(st->path_segments, (size_t)new_cap * sizeof(FS_PathSegment));
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

void fs_path_state_borrow(const FS_InternalState* st, FS_PathStateBorrow* out_state) {
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

void fs_path_state_bind_path2d(FS_InternalState* st, const FS_Path2D* path) {
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

void fs_path_state_restore(FS_InternalState* st, const FS_PathStateBorrow* saved) {
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

void fs_path_state_begin_temporary(FS_InternalState* st, FS_PathStateBorrow* out_saved) {
    if (!st || !out_saved) {
        return;
    }
    fs_path_state_borrow(st, out_saved);
    fs_path_state_bind_empty(st);
}

void fs_path_state_end_temporary(FS_InternalState* st, const FS_PathStateBorrow* saved) {
    if (!st || !saved) {
        return;
    }
    FS_PathSegment* temp_segments = st->path_segments;
    fs_path_state_restore(st, saved);
    if (temp_segments && temp_segments != saved->segments) {
        free(temp_segments);
    }
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

bool fs_path_append_arc_sweep(FS_Core* core, float cx, float cy, float radius, float a0, float a1) {
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

float fs_arc_resolve_delta(float start_angle, float end_angle, bool counterclockwise) {
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

static bool fs_path2d_continuous_corner(FS_Path2D* path, float cx, float cy,
                                         float rx, float ry, uint32_t corner) {
    const float half_pi = 1.57079632679489661923f;
    const uint32_t segments = 12u;
    for (uint32_t i = 1u; i <= segments; ++i) {
        const float t = half_pi * ((float)i / (float)segments);
        const float st = sqrtf(fmaxf(sinf(t), 0.0f));
        const float ct = sqrtf(fmaxf(cosf(t), 0.0f));
        float px = cx, py = cy;
        switch (corner) {
            case 0u: px += rx * st; py -= ry * ct; break;
            case 1u: px += rx * ct; py += ry * st; break;
            case 2u: px -= rx * st; py += ry * ct; break;
            default: px -= rx * ct; py -= ry * st; break;
        }
        if (!fs_path2d_line_to(path, px, py)) return false;
    }
    return true;
}

static bool fs_path2d_corner(FS_Path2D* path, float cx, float cy, float rx, float ry,
                             float start, float end, uint32_t corner,
                             FS_CornerProfile profile) {
    if (rx <= 1e-6f || ry <= 1e-6f) {
        const float ex[4] = {cx + rx, cx + rx, cx - rx, cx};
        const float ey[4] = {cy, cy + ry, cy, cy - ry};
        return fs_path2d_line_to(path, ex[corner], ey[corner]);
    }
    if (profile == FS_CORNER_PROFILE_CONTINUOUS) {
        return fs_path2d_continuous_corner(path, cx, cy, rx, ry, corner);
    }
    return fs_path2d_ellipse(path, cx, cy, rx, ry, 0.0f, start, end, false);
}

bool fs_path2d_round_rect_radii(FS_Path2D* path, float x, float y, float w, float h,
                                const FS_RoundRectRadii* radii,
                                FS_CornerProfile profile) {
    if (!path) return false;
    FS_NormalizedRoundRect rr;
    if (!fs_normalize_round_rect(x, y, w, h, radii, &rr)) return false;
    if (rr.w <= 1e-6f || rr.h <= 1e-6f) return fs_path2d_rect(path, rr.x, rr.y, rr.w, rr.h);
    const float pi = 3.14159265358979323846f;
    const float left = rr.x, top = rr.y, right = rr.x + rr.w, bottom = rr.y + rr.h;
    const FS_RoundRadius tl = rr.radii.top_left;
    const FS_RoundRadius tr = rr.radii.top_right;
    const FS_RoundRadius br = rr.radii.bottom_right;
    const FS_RoundRadius bl = rr.radii.bottom_left;
    if (!fs_path2d_move_to(path, left + tl.x, top)) return false;
    if (!fs_path2d_line_to(path, right - tr.x, top)) return false;
    if (!fs_path2d_corner(path, right-tr.x, top+tr.y, tr.x, tr.y,
                          -0.5f*pi, 0.0f, 0u, profile)) return false;
    if (!fs_path2d_line_to(path, right, bottom - br.y)) return false;
    if (!fs_path2d_corner(path, right-br.x, bottom-br.y, br.x, br.y,
                          0.0f, 0.5f*pi, 1u, profile)) return false;
    if (!fs_path2d_line_to(path, left + bl.x, bottom)) return false;
    if (!fs_path2d_corner(path, left+bl.x, bottom-bl.y, bl.x, bl.y,
                          0.5f*pi, pi, 2u, profile)) return false;
    if (!fs_path2d_line_to(path, left, top + tl.y)) return false;
    if (!fs_path2d_corner(path, left+tl.x, top+tl.y, tl.x, tl.y,
                          pi, 1.5f*pi, 3u, profile)) return false;
    return fs_path2d_close(path);
}

bool fs_path2d_round_rect(FS_Path2D* path, float x, float y, float w, float h, float radius) {
    const FS_RoundRectRadii radii = fs_round_rect_uniform_radii(radius, radius);
    return fs_path2d_round_rect_radii(path, x, y, w, h, &radii, FS_CORNER_PROFILE_ROUND);
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

void fs_path_begin(FS_Core* core) {
    FS_InternalState* st = fs_path_core_state(core);
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
    FS_InternalState* st = fs_path_core_state(core);
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
    FS_InternalState* st = fs_path_core_state(core);
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
    FS_InternalState* st = fs_path_core_state(core);
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
    FS_InternalState* st = fs_path_core_state(core);
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
    FS_InternalState* st = fs_path_core_state(core);
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
    FS_InternalState* st = fs_path_core_state(core);
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
    const FS_RoundRectRadii radii = fs_round_rect_uniform_radii(radius, radius);
    return fs_path_round_rect_radii(core, x, y, w, h, &radii, FS_CORNER_PROFILE_ROUND);
#if 0
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

#endif
}
bool fs_path_close(FS_Core* core) {
    FS_InternalState* st = fs_path_core_state(core);
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
static bool fs_path_continuous_corner(FS_Core* core, float cx, float cy,
                                       float rx, float ry, uint32_t corner) {
    const float half_pi = 1.57079632679489661923f;
    const uint32_t segments = 12u;
    for (uint32_t i = 1u; i <= segments; ++i) {
        const float t = half_pi * ((float)i / (float)segments);
        const float st = sqrtf(fmaxf(sinf(t), 0.0f));
        const float ct = sqrtf(fmaxf(cosf(t), 0.0f));
        float px = cx, py = cy;
        switch (corner) {
            case 0u: px += rx * st; py -= ry * ct; break;
            case 1u: px += rx * ct; py += ry * st; break;
            case 2u: px -= rx * st; py += ry * ct; break;
            default: px -= rx * ct; py -= ry * st; break;
        }
        if (!fs_path_line_to(core, px, py)) return false;
    }
    return true;
}

static bool fs_path_corner(FS_Core* core, float cx, float cy, float rx, float ry,
                           float start, float end, uint32_t corner,
                           FS_CornerProfile profile) {
    if (rx <= 1e-6f || ry <= 1e-6f) {
        const float ex[4] = {cx + rx, cx + rx, cx - rx, cx};
        const float ey[4] = {cy, cy + ry, cy, cy - ry};
        return fs_path_line_to(core, ex[corner], ey[corner]);
    }
    if (profile == FS_CORNER_PROFILE_CONTINUOUS) {
        return fs_path_continuous_corner(core, cx, cy, rx, ry, corner);
    }
    return fs_path_ellipse(core, cx, cy, rx, ry, 0.0f, start, end, false);
}

bool fs_path_round_rect_radii(FS_Core* core, float x, float y, float w, float h,
                              const FS_RoundRectRadii* radii,
                              FS_CornerProfile profile) {
    if (!core) return false;
    FS_NormalizedRoundRect rr;
    if (!fs_normalize_round_rect(x, y, w, h, radii, &rr)) return false;
    if (rr.w <= 1e-6f || rr.h <= 1e-6f) return fs_path_rect(core, rr.x, rr.y, rr.w, rr.h);
    const float pi = 3.14159265358979323846f;
    const float left = rr.x, top = rr.y, right = rr.x + rr.w, bottom = rr.y + rr.h;
    const FS_RoundRadius tl = rr.radii.top_left;
    const FS_RoundRadius tr = rr.radii.top_right;
    const FS_RoundRadius br = rr.radii.bottom_right;
    const FS_RoundRadius bl = rr.radii.bottom_left;
    if (!fs_path_move_to(core, left + tl.x, top)) return false;
    if (!fs_path_line_to(core, right - tr.x, top)) return false;
    if (!fs_path_corner(core, right-tr.x, top+tr.y, tr.x, tr.y, -0.5f*pi, 0, 0u, profile)) return false;
    if (!fs_path_line_to(core, right, bottom-br.y)) return false;
    if (!fs_path_corner(core, right-br.x, bottom-br.y, br.x, br.y, 0, 0.5f*pi, 1u, profile)) return false;
    if (!fs_path_line_to(core, left+bl.x, bottom)) return false;
    if (!fs_path_corner(core, left+bl.x, bottom-bl.y, bl.x, bl.y, 0.5f*pi, pi, 2u, profile)) return false;
    if (!fs_path_line_to(core, left, top+tl.y)) return false;
    if (!fs_path_corner(core, left+tl.x, top+tl.y, tl.x, tl.y, pi, 1.5f*pi, 3u, profile)) return false;
    return fs_path_close(core);
}
