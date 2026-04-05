#include "fullstack_core_private.h"

#include <math.h>

bool fs_transform_requires_oriented_quad(const FS_Transform2D* t) {
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

void fs_transform_apply_point(const FS_Transform2D* t, float x, float y, float* out_x, float* out_y) {
    if (out_x) {
        *out_x = t ? (t->a * x + t->c * y + t->e) : x;
    }
    if (out_y) {
        *out_y = t ? (t->b * x + t->d * y + t->f) : y;
    }
}

FS_Transform2D fs_transform_identity_value(void) {
    FS_Transform2D result = {.a = 1.0f, .b = 0.0f, .c = 0.0f, .d = 1.0f, .e = 0.0f, .f = 0.0f};
    return result;
}

FS_Transform2D fs_transform_mul(const FS_Transform2D* lhs, const FS_Transform2D* rhs) {
    FS_Transform2D result;
    if (!lhs || !rhs) {
        result.a = 1.0f;
        result.b = 0.0f;
        result.c = 0.0f;
        result.d = 1.0f;
        result.e = 0.0f;
        result.f = 0.0f;
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

void fs_affine_set_identity_2d(float m[6]) {
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

bool fs_affine_try_invert_2d(const float m[6], float out_inv[6]) {
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

void fs_affine_apply_point_2d(const float m[6], float x, float y, float* out_x, float* out_y) {
    if (!m || !out_x || !out_y) {
        return;
    }
    *out_x = m[0] * x + m[2] * y + m[4];
    *out_y = m[1] * x + m[3] * y + m[5];
}

void fs_command_set_oriented_quad_from_rect(FS_Command* cmd, const FS_Transform2D* t, float x, float y, float w, float h) {
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

bool fs_vec2_normalize(float x, float y, float* out_x, float* out_y) {
    const float len = hypotf(x, y);
    if (len <= 1e-6f) {
        return false;
    }
    const float inv_len = 1.0f / len;
    *out_x = x * inv_len;
    *out_y = y * inv_len;
    return true;
}

bool fs_transform_points_aabb(
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

void fs_eval_quad_point(
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

float fs_transform_metric_scale_cpu(const FS_Transform2D* t) {
    if (!t) {
        return 1.0f;
    }
    const float sx = hypotf(t->a, t->b);
    const float sy = hypotf(t->c, t->d);
    const float s = sx * sy;
    return sqrtf(fmaxf(s, 1e-8f));
}

float fs_distance_sq_point_segment(
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
