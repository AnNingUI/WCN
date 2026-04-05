#include "fullstack_freetype2_font_backend.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef FS_HAS_FREETYPE2
#include <ft2build.h>
#include FT_FREETYPE_H
#ifdef FS_HAS_HARFBUZZ
#include <hb.h>
#include <hb-ft.h>
#endif

typedef struct FS_FT2FaceHandle {
    FT_Face face;
#ifdef FS_HAS_HARFBUZZ
    hb_font_t* hb_font;
#endif
} FS_FT2FaceHandle;

static FT_Library g_ft_library = NULL;
static int g_ft_ref_count = 0;
static const float FS_EDT_INF = 1e20f;

typedef struct FS_FT2SDFProfile {
    int padding;
    float onedge_value;
    float pixel_dist_scale;
    float refine_mix;
    float preprocess_strength;
    int solidify_passes;
    int postprocess_passes;
} FS_FT2SDFProfile;

static void fs_ft2_pick_sdf_profile(float font_px, FS_FT2SDFProfile* out_profile) {
    if (!out_profile) {
        return;
    }
    const float px = font_px < 1.0f ? 1.0f : font_px;
    FS_FT2SDFProfile p = {
        .padding = 8,
        .onedge_value = 176.0f,
        .pixel_dist_scale = 10.0f,
        .refine_mix = 0.68f,
        .preprocess_strength = 0.35f,
        .solidify_passes = 0,
        .postprocess_passes = 1
    };

    if (px <= 18.0f) {
        p.padding = 14;
        p.onedge_value = 194.0f;
        p.pixel_dist_scale = 24.0f;
        p.refine_mix = 0.78f;
        p.preprocess_strength = 0.90f;
        p.solidify_passes = 2;
        p.postprocess_passes = 2;
    } else if (px <= 32.0f) {
        p.padding = 12;
        p.onedge_value = 188.0f;
        p.pixel_dist_scale = 18.0f;
        p.refine_mix = 0.72f;
        p.preprocess_strength = 0.72f;
        p.solidify_passes = 1;
        p.postprocess_passes = 2;
    } else if (px <= 64.0f) {
        p.padding = 10;
        p.onedge_value = 184.0f;
        p.pixel_dist_scale = 14.0f;
        p.refine_mix = 0.70f;
        p.preprocess_strength = 0.58f;
        p.solidify_passes = 1;
        p.postprocess_passes = 1;
    } else if (px <= 128.0f) {
        p.padding = 8;
        p.onedge_value = 176.0f;
        p.pixel_dist_scale = 10.0f;
        p.refine_mix = 0.62f;
        p.preprocess_strength = 0.35f;
        p.solidify_passes = 0;
        p.postprocess_passes = 1;
    } else {
        p.padding = 6;
        p.onedge_value = 168.0f;
        p.pixel_dist_scale = 8.0f;
        p.refine_mix = 0.55f;
        p.preprocess_strength = 0.20f;
        p.solidify_passes = 0;
        p.postprocess_passes = 0;
    }
    *out_profile = p;
}

static void fs_edt_1d(const float* f, int n, float* d) {
    int* v = (int*)malloc((size_t)n * sizeof(int));
    float* z = (float*)malloc((size_t)(n + 1) * sizeof(float));
    if (!v || !z) {
        for (int i = 0; i < n; ++i) {
            d[i] = f[i];
        }
        free(v);
        free(z);
        return;
    }

    int k = 0;
    v[0] = 0;
    z[0] = -FS_EDT_INF;
    z[1] = FS_EDT_INF;
    for (int q = 1; q < n; ++q) {
        float s;
        do {
            const int vk = v[k];
            s = ((f[q] + (float)(q * q)) - (f[vk] + (float)(vk * vk))) / (2.0f * (float)(q - vk));
            if (s <= z[k]) {
                k--;
            }
        } while (s <= z[k]);
        k++;
        v[k] = q;
        z[k] = s;
        z[k + 1] = FS_EDT_INF;
    }

    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < (float)q) {
            k++;
        }
        const float dx = (float)q - (float)v[k];
        d[q] = dx * dx + f[v[k]];
    }

    free(v);
    free(z);
}

static bool fs_edt_2d(const float* f, int w, int h, float* d) {
    float* tmp = (float*)malloc((size_t)w * (size_t)h * sizeof(float));
    float* in_col = (float*)malloc((size_t)h * sizeof(float));
    float* out_col = (float*)malloc((size_t)h * sizeof(float));
    float* in_row = (float*)malloc((size_t)w * sizeof(float));
    float* out_row = (float*)malloc((size_t)w * sizeof(float));
    if (!tmp || !in_col || !out_col || !in_row || !out_row) {
        free(tmp);
        free(in_col);
        free(out_col);
        free(in_row);
        free(out_row);
        return false;
    }

    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            in_col[y] = f[(size_t)y * (size_t)w + (size_t)x];
        }
        fs_edt_1d(in_col, h, out_col);
        for (int y = 0; y < h; ++y) {
            tmp[(size_t)y * (size_t)w + (size_t)x] = out_col[y];
        }
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            in_row[x] = tmp[(size_t)y * (size_t)w + (size_t)x];
        }
        fs_edt_1d(in_row, w, out_row);
        for (int x = 0; x < w; ++x) {
            d[(size_t)y * (size_t)w + (size_t)x] = out_row[x];
        }
    }

    free(tmp);
    free(in_col);
    free(out_col);
    free(in_row);
    free(out_row);
    return true;
}

static bool fs_ft2_library_acquire(void) {
    if (g_ft_ref_count == 0) {
        if (FT_Init_FreeType(&g_ft_library) != 0) {
            g_ft_library = NULL;
            return false;
        }
    }
    g_ft_ref_count++;
    return true;
}

static void fs_ft2_library_release(void) {
    if (g_ft_ref_count <= 0) {
        return;
    }
    g_ft_ref_count--;
    if (g_ft_ref_count == 0 && g_ft_library) {
        FT_Done_FreeType(g_ft_library);
        g_ft_library = NULL;
    }
}

static float fs_ft2_coverage_at(const uint8_t* coverage, uint32_t width, uint32_t height, int32_t x, int32_t y) {
    if (x < 0) {
        x = 0;
    } else if ((uint32_t)x >= width) {
        x = (int32_t)width - 1;
    }
    if (y < 0) {
        y = 0;
    } else if ((uint32_t)y >= height) {
        y = (int32_t)height - 1;
    }
    return (float)coverage[(size_t)(uint32_t)y * (size_t)width + (size_t)(uint32_t)x] / 255.0f;
}

static float fs_ft2_subpixel_signed_distance(
    const uint8_t* coverage,
    uint32_t width,
    uint32_t height,
    uint32_t x,
    uint32_t y
) {
    const float a = fs_ft2_coverage_at(coverage, width, height, (int32_t)x, (int32_t)y);
    if (a <= 0.0f || a >= 1.0f) {
        return 0.0f;
    }

    const float gx =
        fs_ft2_coverage_at(coverage, width, height, (int32_t)x + 1, (int32_t)y) -
        fs_ft2_coverage_at(coverage, width, height, (int32_t)x - 1, (int32_t)y);
    const float gy =
        fs_ft2_coverage_at(coverage, width, height, (int32_t)x, (int32_t)y + 1) -
        fs_ft2_coverage_at(coverage, width, height, (int32_t)x, (int32_t)y - 1);
    const float grad = sqrtf(gx * gx + gy * gy) * 0.5f;
    if (grad < 1e-4f) {
        return 0.0f;
    }

    float d = (0.5f - a) / grad;
    if (d < -3.0f) {
        d = -3.0f;
    } else if (d > 3.0f) {
        d = 3.0f;
    }
    return d;
}

static uint8_t fs_ft2_clamp_u8_i32(int v) {
    if (v < 0) return 0u;
    if (v > 255) return 255u;
    return (uint8_t)v;
}

static float fs_ft2_signf(float v) {
    if (v < 0.0f) {
        return -1.0f;
    }
    if (v > 0.0f) {
        return 1.0f;
    }
    return 0.0f;
}

static bool fs_ft2_is_solid_coverage(float a) {
    return !(a > 0.0f && a < 1.0f);
}

static float fs_ft2_get_coverage_norm(
    const uint8_t* coverage,
    uint32_t stage_w,
    int pad,
    int inner_w,
    int inner_h,
    int x,
    int y
) {
    if (!coverage || x < 0 || y < 0 || x >= inner_w || y >= inner_h) {
        return 0.0f;
    }
    const size_t i = (size_t)(y + pad) * (size_t)stage_w + (size_t)(x + pad);
    return (float)coverage[i] / 255.0f;
}

static void fs_ft2_esdt_paint_stage(
    const uint8_t* coverage,
    uint32_t stage_w,
    uint32_t stage_h,
    int pad,
    float* outer,
    float* inner
) {
    if (!coverage || !outer || !inner) {
        return;
    }
    const size_t n = (size_t)stage_w * (size_t)stage_h;
    for (size_t i = 0; i < n; ++i) {
        outer[i] = FS_EDT_INF;
        inner[i] = 0.0f;
    }

    const int inner_w = (int)stage_w - 2 * pad;
    const int inner_h = (int)stage_h - 2 * pad;
    if (inner_w <= 0 || inner_h <= 0) {
        return;
    }

    for (int y = 0; y < inner_h; ++y) {
        for (int x = 0; x < inner_w; ++x) {
            const size_t j = (size_t)(y + pad) * (size_t)stage_w + (size_t)(x + pad);
            const uint8_t a = coverage[j];
            if (a == 0u) {
                continue;
            }
            if (a >= 254u) {
                outer[j] = 0.0f;
                inner[j] = FS_EDT_INF;
            } else {
                outer[j] = 0.0f;
                inner[j] = 0.0f;
            }
        }
    }
}

static bool fs_ft2_esdt_check_cross(
    float nx,
    float ny,
    float dc,
    float dl,
    float dr,
    float dxl,
    float dyl,
    float dxr,
    float dyr
) {
    return
        ((dxl * nx + dyl * ny) * (dc * dl) > 0.0f) &&
        ((dxr * nx + dyr * ny) * (dc * dr) > 0.0f) &&
        ((dxl * dxr + dyl * dyr) * (dl * dr) > 0.0f);
}

static void fs_ft2_esdt_paint_subpixel_offsets(
    const uint8_t* coverage,
    uint32_t stage_w,
    uint32_t stage_h,
    int pad,
    float* outer,
    float* inner,
    float* xo,
    float* yo,
    float* xi,
    float* yi,
    bool relax_pre
) {
    if (!coverage || !outer || !inner || !xo || !yo || !xi || !yi) {
        return;
    }

    const int inner_w = (int)stage_w - 2 * pad;
    const int inner_h = (int)stage_h - 2 * pad;
    if (inner_w <= 0 || inner_h <= 0) {
        return;
    }

    const size_t n = (size_t)stage_w * (size_t)stage_h;
    memset(xo, 0, n * sizeof(float));
    memset(yo, 0, n * sizeof(float));
    memset(xi, 0, n * sizeof(float));
    memset(yi, 0, n * sizeof(float));

    for (int y = 0; y < inner_h; ++y) {
        for (int x = 0; x < inner_w; ++x) {
            const size_t j = (size_t)(y + pad) * (size_t)stage_w + (size_t)(x + pad);
            const float c = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x, y);
            if (!fs_ft2_is_solid_coverage(c)) {
                const float dc = c - 0.5f;

                const float l = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x - 1, y);
                const float r = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x + 1, y);
                const float t = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x, y - 1);
                const float b = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x, y + 1);

                const float tl = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x - 1, y - 1);
                const float tr = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x + 1, y - 1);
                const float bl = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x - 1, y + 1);
                const float br = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x + 1, y + 1);

                const float ll = (tl + 2.0f * l + bl) * 0.25f;
                const float rr = (tr + 2.0f * r + br) * 0.25f;
                const float tt = (tl + 2.0f * t + tr) * 0.25f;
                const float bb = (bl + 2.0f * b + br) * 0.25f;

                float min_v = l;
                float max_v = l;
                const float samples[7] = {r, t, b, tl, tr, bl, br};
                for (int si = 0; si < 7; ++si) {
                    if (samples[si] < min_v) min_v = samples[si];
                    if (samples[si] > max_v) max_v = samples[si];
                }

                if (min_v > 0.0f) {
                    inner[j] = FS_EDT_INF;
                    continue;
                }
                if (max_v < 1.0f) {
                    outer[j] = FS_EDT_INF;
                    continue;
                }

                float dx = rr - ll;
                float dy = bb - tt;
                const float dn = sqrtf(dx * dx + dy * dy);
                if (dn > 1e-6f) {
                    dx /= dn;
                    dy /= dn;
                    xo[j] = -dc * dx;
                    yo[j] = -dc * dy;
                }
            } else if (c >= 1.0f) {
                const float l = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x - 1, y);
                const float r = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x + 1, y);
                const float t = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x, y - 1);
                const float b = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x, y + 1);

                if (l <= 0.0f) {
                    xo[j - 1u] = 0.4999f;
                    outer[j - 1u] = 0.0f;
                    inner[j - 1u] = 0.0f;
                }
                if (r <= 0.0f) {
                    xo[j + 1u] = -0.4999f;
                    outer[j + 1u] = 0.0f;
                    inner[j + 1u] = 0.0f;
                }
                if (t <= 0.0f) {
                    yo[j - stage_w] = 0.4999f;
                    outer[j - stage_w] = 0.0f;
                    inner[j - stage_w] = 0.0f;
                }
                if (b <= 0.0f) {
                    yo[j + stage_w] = -0.4999f;
                    outer[j + stage_w] = 0.0f;
                    inner[j + stage_w] = 0.0f;
                }
            }
        }
    }

    float* xs = xo;
    float* ys = yo;
    if (relax_pre) {
        for (int y = 0; y < inner_h; ++y) {
            for (int x = 0; x < inner_w; ++x) {
                const size_t j = (size_t)(y + pad) * (size_t)stage_w + (size_t)(x + pad);
                const float nx = xo[j];
                const float ny = yo[j];
                if (nx == 0.0f && ny == 0.0f) {
                    continue;
                }

                const float c = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x, y);
                const float l = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x - 1, y);
                const float r = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x + 1, y);
                const float t = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x, y - 1);
                const float b = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x, y + 1);

                const float dxl = xo[j - 1u];
                const float dxr = xo[j + 1u];
                const float dxt = xo[j - stage_w];
                const float dxb = xo[j + stage_w];
                const float dyl = yo[j - 1u];
                const float dyr = yo[j + 1u];
                const float dyt = yo[j - stage_w];
                const float dyb = yo[j + stage_w];

                float dx = nx;
                float dy = ny;
                float dw = 1.0f;

                const float dc = c - 0.5f;
                const float dl = l - 0.5f;
                const float dr = r - 0.5f;
                const float dt = t - 0.5f;
                const float db = b - 0.5f;

                if (!fs_ft2_is_solid_coverage(l) && !fs_ft2_is_solid_coverage(r)) {
                    if (fs_ft2_esdt_check_cross(nx, ny, dc, dl, dr, dxl, dyl, dxr, dyr)) {
                        dx += (dxl + dxr) * 0.5f;
                        dy += (dyl + dyr) * 0.5f;
                        dw += 1.0f;
                    }
                }
                if (!fs_ft2_is_solid_coverage(t) && !fs_ft2_is_solid_coverage(b)) {
                    if (fs_ft2_esdt_check_cross(nx, ny, dc, dt, db, dxt, dyt, dxb, dyb)) {
                        dx += (dxt + dxb) * 0.5f;
                        dy += (dyt + dyb) * 0.5f;
                        dw += 1.0f;
                    }
                }
                if (!fs_ft2_is_solid_coverage(l) && !fs_ft2_is_solid_coverage(t)) {
                    if (fs_ft2_esdt_check_cross(nx, ny, dc, dl, dt, dxl, dyl, dxt, dyt)) {
                        dx += (dxl + dxt - 1.0f) * 0.5f;
                        dy += (dyl + dyt - 1.0f) * 0.5f;
                        dw += 1.0f;
                    }
                }
                if (!fs_ft2_is_solid_coverage(r) && !fs_ft2_is_solid_coverage(t)) {
                    if (fs_ft2_esdt_check_cross(nx, ny, dc, dr, dt, dxr, dyr, dxt, dyt)) {
                        dx += (dxr + dxt + 1.0f) * 0.5f;
                        dy += (dyr + dyt - 1.0f) * 0.5f;
                        dw += 1.0f;
                    }
                }
                if (!fs_ft2_is_solid_coverage(l) && !fs_ft2_is_solid_coverage(b)) {
                    if (fs_ft2_esdt_check_cross(nx, ny, dc, dl, db, dxl, dyl, dxb, dyb)) {
                        dx += (dxl + dxb - 1.0f) * 0.5f;
                        dy += (dyl + dyb + 1.0f) * 0.5f;
                        dw += 1.0f;
                    }
                }
                if (!fs_ft2_is_solid_coverage(r) && !fs_ft2_is_solid_coverage(b)) {
                    if (fs_ft2_esdt_check_cross(nx, ny, dc, dr, db, dxr, dyr, dxb, dyb)) {
                        dx += (dxr + dxb + 1.0f) * 0.5f;
                        dy += (dyr + dyb + 1.0f) * 0.5f;
                        dw += 1.0f;
                    }
                }

                const float nn = sqrtf(nx * nx + ny * ny);
                if (nn <= 1e-6f) {
                    xi[j] = nx;
                    yi[j] = ny;
                    continue;
                }
                const float ll = (dx * nx + dy * ny) / nn;
                xi[j] = nx * ll / (dw * nn);
                yi[j] = ny * ll / (dw * nn);
            }
        }
        xs = xi;
        ys = yi;
    }

    for (int y = 0; y < inner_h; ++y) {
        for (int x = 0; x < inner_w; ++x) {
            const size_t j = (size_t)(y + pad) * (size_t)stage_w + (size_t)(x + pad);
            const float nx = xs[j];
            const float ny = ys[j];
            if (nx == 0.0f && ny == 0.0f) {
                continue;
            }
            const float nn = sqrtf(nx * nx + ny * ny);
            if (nn <= 1e-6f) {
                continue;
            }

            const float sx = (fabsf(nx / nn) - 0.5f) > 0.0f ? fs_ft2_signf(nx) : 0.0f;
            const float sy = (fabsf(ny / nn) - 0.5f) > 0.0f ? fs_ft2_signf(ny) : 0.0f;
            const float c = fs_ft2_get_coverage_norm(coverage, stage_w, pad, inner_w, inner_h, x, y);
            const float d = fs_ft2_get_coverage_norm(
                coverage,
                stage_w,
                pad,
                inner_w,
                inner_h,
                x + (int)sx,
                y + (int)sy
            );
            const float s = fs_ft2_signf(d - c);

            const float dlo = (nn + 0.4999f * s) / nn;
            const float dli = (nn - 0.4999f * s) / nn;

            xo[j] = nx * dlo;
            yo[j] = ny * dlo;
            xi[j] = nx * dli;
            yi[j] = ny * dli;
        }
    }
}

static void fs_ft2_esdt_1d(
    float* mask,
    float* xs,
    float* ys,
    int offset,
    int stride,
    int length,
    float* f,
    float* z,
    float* b,
    float* t,
    uint32_t* v
) {
    if (!mask || !xs || !ys || !f || !z || !b || !t || !v || length <= 0) {
        return;
    }
    v[0] = 0u;
    b[0] = xs[offset];
    t[0] = ys[offset];
    z[0] = -FS_EDT_INF;
    z[1] = FS_EDT_INF;
    f[0] = mask[offset] ? FS_EDT_INF : ys[offset] * ys[offset];

    int k = 0;
    for (int q = 1; q < length; ++q) {
        const int o = offset + q * stride;
        const float dx = xs[o];
        const float dy = ys[o];
        const float fq = (mask[o] ? FS_EDT_INF : dy * dy);
        f[q] = fq;
        t[q] = dy;
        const float qs = (float)q + dx;
        const float q2 = qs * qs;
        b[q] = qs;

        float s = 0.0f;
        do {
            const uint32_t r = v[k];
            const float rs = b[r];
            const float denom = qs - rs;
            if (fabsf(denom) <= 1e-6f) {
                s = FS_EDT_INF;
            } else {
                const float r2 = rs * rs;
                s = (fq - f[r] + q2 - r2) / denom * 0.5f;
            }
            if (!(s <= z[k])) {
                break;
            }
            --k;
        } while (k > -1);

        k++;
        v[k] = (uint32_t)q;
        z[k] = s;
        z[k + 1] = FS_EDT_INF;
    }

    for (int q = 0, kk = 0; q < length; ++q) {
        while (z[kk + 1] < (float)q) {
            kk++;
        }
        const uint32_t r = v[kk];
        const float rs = b[r];
        const float dy = t[r];
        const float rq = rs - (float)q;
        const int o = offset + q * stride;
        xs[o] = rq;
        ys[o] = dy;
        if ((int)r != q) {
            mask[o] = 0.0f;
        }
    }
}

static void fs_ft2_esdt_2d(
    float* mask,
    float* xs,
    float* ys,
    int w,
    int h,
    float* f,
    float* z,
    float* b,
    float* t,
    uint32_t* v
) {
    if (!mask || !xs || !ys || !f || !z || !b || !t || !v || w <= 0 || h <= 0) {
        return;
    }
    for (int x = 0; x < w; ++x) {
        fs_ft2_esdt_1d(mask, ys, xs, x, w, h, f, z, b, t, v);
    }
    for (int y = 0; y < h; ++y) {
        fs_ft2_esdt_1d(mask, xs, ys, y * w, 1, w, f, z, b, t, v);
    }
}

static float fs_ft2_relax_check_neighbor(
    float* xs,
    float* ys,
    uint32_t stage_w,
    int pad,
    int inner_w,
    int inner_h,
    int x,
    int y,
    float dx,
    float dy,
    float current_best,
    size_t j
) {
    if (!xs || !ys || x < 0 || y < 0 || x >= inner_w || y >= inner_h) {
        return current_best;
    }
    const size_t k = (size_t)(y + pad) * (size_t)stage_w + (size_t)(x + pad);
    const float dx2 = dx + xs[k];
    const float dy2 = dy + ys[k];
    const float d2 = sqrtf(dx2 * dx2 + dy2 * dy2);
    if (d2 < current_best) {
        xs[j] = dx2;
        ys[j] = dy2;
        return d2;
    }
    return current_best;
}

static void fs_ft2_relax_subpixel_offsets(
    uint32_t stage_w,
    uint32_t stage_h,
    int pad,
    float* xo,
    float* yo,
    float* xi,
    float* yi
) {
    (void)stage_h;
    const int inner_w = (int)stage_w - 2 * pad;
    const int inner_h = (int)stage_h - 2 * pad;
    if (inner_w <= 0 || inner_h <= 0) {
        return;
    }

    float* relax_sets[4] = {xo, yo, xi, yi};
    for (int si = 0; si < 2; ++si) {
        float* xs = relax_sets[si * 2];
        float* ys = relax_sets[si * 2 + 1];
        if (!xs || !ys) {
            continue;
        }
        for (int y = 0; y < inner_h; ++y) {
            for (int x = 0; x < inner_w; ++x) {
                const size_t j = (size_t)(y + pad) * (size_t)stage_w + (size_t)(x + pad);
                const float dx = xs[j];
                const float dy = ys[j];
                if (dx == 0.0f && dy == 0.0f) {
                    continue;
                }
                float d = sqrtf(dx * dx + dy * dy);
                if (d <= 1e-6f) {
                    continue;
                }
                const float ds = (d - 0.5f) / d;
                const float tx = (float)x + dx * ds;
                const float ty = (float)y + dy * ds;
                const int ix = (int)lroundf(tx);
                const int iy = (int)lroundf(ty);
                d = fs_ft2_relax_check_neighbor(xs, ys, stage_w, pad, inner_w, inner_h, ix + 1, iy, (float)(ix - x + 1), (float)(iy - y), d, j);
                d = fs_ft2_relax_check_neighbor(xs, ys, stage_w, pad, inner_w, inner_h, ix - 1, iy, (float)(ix - x - 1), (float)(iy - y), d, j);
                d = fs_ft2_relax_check_neighbor(xs, ys, stage_w, pad, inner_w, inner_h, ix, iy + 1, (float)(ix - x), (float)(iy - y + 1), d, j);
                d = fs_ft2_relax_check_neighbor(xs, ys, stage_w, pad, inner_w, inner_h, ix, iy - 1, (float)(ix - x), (float)(iy - y - 1), d, j);
                (void)d;
            }
        }
    }
}

static bool fs_ft2_build_sdf_from_coverage_esdt(
    const uint8_t* coverage,
    uint32_t width,
    uint32_t height,
    int padding,
    float onedge_value,
    float pixel_dist_scale,
    float refine_mix,
    int relax_postprocess_passes,
    uint8_t** out_sdf
) {
    if (!coverage || width == 0u || height == 0u || !out_sdf || padding < 0) {
        return false;
    }
    *out_sdf = NULL;
    const size_t n = (size_t)width * (size_t)height;
    const int max_dim = (width > height) ? (int)width : (int)height;
    if (max_dim <= 0) {
        return false;
    }

    float* outer = (float*)malloc(n * sizeof(float));
    float* inner = (float*)malloc(n * sizeof(float));
    float* xo = (float*)malloc(n * sizeof(float));
    float* yo = (float*)malloc(n * sizeof(float));
    float* xi = (float*)malloc(n * sizeof(float));
    float* yi = (float*)malloc(n * sizeof(float));
    float* f = (float*)malloc((size_t)(max_dim + 1) * sizeof(float));
    float* z = (float*)malloc((size_t)(max_dim + 2) * sizeof(float));
    float* b = (float*)malloc((size_t)(max_dim + 1) * sizeof(float));
    float* t = (float*)malloc((size_t)(max_dim + 1) * sizeof(float));
    uint32_t* v = (uint32_t*)malloc((size_t)(max_dim + 2) * sizeof(uint32_t));
    uint8_t* sdf = (uint8_t*)malloc(n);
    if (!outer || !inner || !xo || !yo || !xi || !yi || !f || !z || !b || !t || !v || !sdf) {
        free(outer);
        free(inner);
        free(xo);
        free(yo);
        free(xi);
        free(yi);
        free(f);
        free(z);
        free(b);
        free(t);
        free(v);
        free(sdf);
        return false;
    }

    fs_ft2_esdt_paint_stage(coverage, width, height, padding, outer, inner);
    const bool relax_pre = refine_mix >= 0.66f;
    fs_ft2_esdt_paint_subpixel_offsets(
        coverage,
        width,
        height,
        padding,
        outer,
        inner,
        xo,
        yo,
        xi,
        yi,
        relax_pre
    );
    fs_ft2_esdt_2d(outer, xo, yo, (int)width, (int)height, f, z, b, t, v);
    fs_ft2_esdt_2d(inner, xi, yi, (int)width, (int)height, f, z, b, t, v);

    for (int pass = 0; pass < relax_postprocess_passes; ++pass) {
        fs_ft2_relax_subpixel_offsets(width, height, padding, xo, yo, xi, yi);
    }

    for (size_t i = 0; i < n; ++i) {
        const float od = fmaxf(0.0f, sqrtf(xo[i] * xo[i] + yo[i] * yo[i]) - 0.5f);
        const float id = fmaxf(0.0f, sqrtf(xi[i] * xi[i] + yi[i] * yi[i]) - 0.5f);
        const float d = (od >= id) ? od : -id;
        const float sd = -d;
        float v8 = onedge_value + sd * pixel_dist_scale;
        if (v8 < 0.0f) {
            v8 = 0.0f;
        } else if (v8 > 255.0f) {
            v8 = 255.0f;
        }
        sdf[i] = (uint8_t)(v8 + 0.5f);
    }

    free(outer);
    free(inner);
    free(xo);
    free(yo);
    free(xi);
    free(yi);
    free(f);
    free(z);
    free(b);
    free(t);
    free(v);
    *out_sdf = sdf;
    return true;
}

static void fs_ft2_solidify_coverage(uint8_t* coverage, uint32_t width, uint32_t height, int passes) {
    if (!coverage || width == 0u || height == 0u || passes <= 0) {
        return;
    }
    const size_t n = (size_t)width * (size_t)height;
    uint8_t* tmp = (uint8_t*)malloc(n);
    if (!tmp) {
        return;
    }
    for (int pass = 0; pass < passes; ++pass) {
        memcpy(tmp, coverage, n);
        for (uint32_t y = 1u; y + 1u < height; ++y) {
            for (uint32_t x = 1u; x + 1u < width; ++x) {
                const size_t i = (size_t)y * (size_t)width + (size_t)x;
                const uint8_t c = coverage[i];
                if (c >= 8u) {
                    continue;
                }
                int strong_neighbors = 0;
                int neighbor_sum = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const size_t j = (size_t)(y + (uint32_t)dy) * (size_t)width + (size_t)(x + (uint32_t)dx);
                        const int v = (int)coverage[j];
                        neighbor_sum += v;
                        if (v >= 160) {
                            strong_neighbors++;
                        }
                    }
                }
                if (strong_neighbors >= 3) {
                    const int avg = neighbor_sum / 8;
                    tmp[i] = fs_ft2_clamp_u8_i32((avg + 64) / 2);
                }
            }
        }
        memcpy(coverage, tmp, n);
    }
    free(tmp);
}

static void fs_ft2_preprocess_coverage(uint8_t* coverage, uint32_t width, uint32_t height, float strength) {
    if (!coverage || width < 3u || height < 3u || strength <= 0.0f) {
        return;
    }
    if (strength > 1.0f) {
        strength = 1.0f;
    }
    const size_t n = (size_t)width * (size_t)height;
    uint8_t* tmp = (uint8_t*)malloc(n);
    if (!tmp) {
        return;
    }
    memcpy(tmp, coverage, n);
    for (uint32_t y = 1u; y + 1u < height; ++y) {
        for (uint32_t x = 1u; x + 1u < width; ++x) {
            const size_t i = (size_t)y * (size_t)width + (size_t)x;
            const int c = (int)coverage[i];
            if (c <= 0 || c >= 255) {
                continue;
            }
            const int n0 = (int)coverage[(size_t)(y - 1u) * (size_t)width + (size_t)x];
            const int n1 = (int)coverage[(size_t)(y + 1u) * (size_t)width + (size_t)x];
            const int n2 = (int)coverage[(size_t)y * (size_t)width + (size_t)(x - 1u)];
            const int n3 = (int)coverage[(size_t)y * (size_t)width + (size_t)(x + 1u)];
            const int d0 = (int)coverage[(size_t)(y - 1u) * (size_t)width + (size_t)(x - 1u)];
            const int d1 = (int)coverage[(size_t)(y - 1u) * (size_t)width + (size_t)(x + 1u)];
            const int d2 = (int)coverage[(size_t)(y + 1u) * (size_t)width + (size_t)(x - 1u)];
            const int d3 = (int)coverage[(size_t)(y + 1u) * (size_t)width + (size_t)(x + 1u)];
            const int blur = (4 * c + 2 * (n0 + n1 + n2 + n3) + d0 + d1 + d2 + d3 + 8) / 16;
            const float mixed = (1.0f - strength) * (float)c + strength * (float)blur;
            tmp[i] = fs_ft2_clamp_u8_i32((int)lroundf(mixed));
        }
    }
    memcpy(coverage, tmp, n);
    free(tmp);
}

static void fs_ft2_postprocess_sdf(
    uint8_t* sdf,
    uint32_t width,
    uint32_t height,
    float onedge_value,
    int passes
) {
    if (!sdf || width < 3u || height < 3u || passes <= 0) {
        return;
    }
    const size_t n = (size_t)width * (size_t)height;
    uint8_t* tmp = (uint8_t*)malloc(n);
    if (!tmp) {
        return;
    }
    const int edge = (int)lroundf(onedge_value);
    const int band = 56;
    for (int pass = 0; pass < passes; ++pass) {
        memcpy(tmp, sdf, n);
        for (uint32_t y = 1u; y + 1u < height; ++y) {
            for (uint32_t x = 1u; x + 1u < width; ++x) {
                const size_t i = (size_t)y * (size_t)width + (size_t)x;
                const int c = (int)sdf[i];
                if (abs(c - edge) > band) {
                    continue;
                }
                const int n0 = (int)sdf[(size_t)(y - 1u) * (size_t)width + (size_t)x];
                const int n1 = (int)sdf[(size_t)(y + 1u) * (size_t)width + (size_t)x];
                const int n2 = (int)sdf[(size_t)y * (size_t)width + (size_t)(x - 1u)];
                const int n3 = (int)sdf[(size_t)y * (size_t)width + (size_t)(x + 1u)];
                const int d0 = (int)sdf[(size_t)(y - 1u) * (size_t)width + (size_t)(x - 1u)];
                const int d1 = (int)sdf[(size_t)(y - 1u) * (size_t)width + (size_t)(x + 1u)];
                const int d2 = (int)sdf[(size_t)(y + 1u) * (size_t)width + (size_t)(x - 1u)];
                const int d3 = (int)sdf[(size_t)(y + 1u) * (size_t)width + (size_t)(x + 1u)];
                const int blur = (4 * c + 2 * (n0 + n1 + n2 + n3) + d0 + d1 + d2 + d3 + 8) / 16;
                tmp[i] = fs_ft2_clamp_u8_i32(blur);
            }
        }
        memcpy(sdf, tmp, n);
    }
    free(tmp);
}

static bool fs_ft2_build_sdf_from_coverage(
    const uint8_t* coverage,
    uint32_t width,
    uint32_t height,
    float onedge_value,
    float pixel_dist_scale,
    float refine_mix,
    uint8_t** out_sdf
) {
    if (!coverage || width == 0u || height == 0u || !out_sdf) {
        return false;
    }
    *out_sdf = NULL;
    const size_t n = (size_t)width * (size_t)height;

    float* f_in = (float*)malloc(n * sizeof(float));
    float* f_out = (float*)malloc(n * sizeof(float));
    float* d_in = (float*)malloc(n * sizeof(float));
    float* d_out = (float*)malloc(n * sizeof(float));
    uint8_t* sdf = (uint8_t*)malloc(n);
    if (!f_in || !f_out || !d_in || !d_out || !sdf) {
        free(f_in);
        free(f_out);
        free(d_in);
        free(d_out);
        free(sdf);
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        const bool inside = coverage[i] >= 128u;
        f_in[i] = inside ? 0.0f : FS_EDT_INF;
        f_out[i] = inside ? FS_EDT_INF : 0.0f;
    }

    if (!fs_edt_2d(f_in, (int)width, (int)height, d_in) ||
        !fs_edt_2d(f_out, (int)width, (int)height, d_out)) {
        free(f_in);
        free(f_out);
        free(d_in);
        free(d_out);
        free(sdf);
        return false;
    }

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t i = (size_t)y * (size_t)width + (size_t)x;
            float sd = sqrtf(d_out[i]) - sqrtf(d_in[i]);
            const uint8_t a8 = coverage[i];
            if (a8 > 0u && a8 < 255u) {
                const float refined = fs_ft2_subpixel_signed_distance(coverage, width, height, x, y);
                const float mix = refine_mix < 0.0f ? 0.0f : (refine_mix > 1.0f ? 1.0f : refine_mix);
                sd = sd * (1.0f - mix) + refined * mix;
            }
            float v = onedge_value + sd * pixel_dist_scale;
            if (v < 0.0f) {
                v = 0.0f;
            } else if (v > 255.0f) {
                v = 255.0f;
            }
            sdf[i] = (uint8_t)(v + 0.5f);
        }
    }

    free(f_in);
    free(f_out);
    free(d_in);
    free(d_out);
    *out_sdf = sdf;
    return true;
}

static void* fs_ft2_load_font_file(const char* path) {
    if (!path) {
        return NULL;
    }
    if (!fs_ft2_library_acquire()) {
        return NULL;
    }

    FT_Face face = NULL;
    if (FT_New_Face(g_ft_library, path, 0, &face) != 0) {
        fs_ft2_library_release();
        return NULL;
    }
    if (face->charmaps && face->num_charmaps > 0) {
        if (FT_Select_Charmap(face, FT_ENCODING_UNICODE) != 0) {
            for (int i = 0; i < face->num_charmaps; ++i) {
                if (face->charmaps[i] && face->charmaps[i]->encoding == FT_ENCODING_UNICODE) {
                    (void)FT_Set_Charmap(face, face->charmaps[i]);
                    break;
                }
            }
        }
    }

    FS_FT2FaceHandle* handle = (FS_FT2FaceHandle*)calloc(1, sizeof(FS_FT2FaceHandle));
    if (!handle) {
        FT_Done_Face(face);
        fs_ft2_library_release();
        return NULL;
    }
    handle->face = face;
#ifdef FS_HAS_HARFBUZZ
    handle->hb_font = hb_ft_font_create_referenced(face);
    if (handle->hb_font) {
        hb_ft_font_set_load_flags(handle->hb_font, FT_LOAD_DEFAULT);
    }
#endif
    return handle;
}

static void fs_ft2_destroy_font(void* font_handle) {
    FS_FT2FaceHandle* handle = (FS_FT2FaceHandle*)font_handle;
    if (!handle) {
        return;
    }
#ifdef FS_HAS_HARFBUZZ
    if (handle->hb_font) {
        hb_font_destroy(handle->hb_font);
        handle->hb_font = NULL;
    }
#endif
    if (handle->face) {
        FT_Done_Face(handle->face);
        handle->face = NULL;
    }
    free(handle);
    fs_ft2_library_release();
}

static uint8_t fs_ft2_unpremul_u8(uint8_t c, uint8_t a) {
    if (a == 0u) {
        return 0u;
    }
    int v = ((int)c * 255 + (int)a / 2) / (int)a;
    if (v < 0) {
        v = 0;
    } else if (v > 255) {
        v = 255;
    }
    return (uint8_t)v;
}

static bool fs_ft2_try_get_color_glyph_rgba(
    FS_FT2FaceHandle* handle,
    uint32_t glyph_index,
    FS_FontGlyphBitmap* out_glyph
) {
    if (!handle || !out_glyph || !handle->face) {
        return false;
    }
    FT_Face face = handle->face;
    if (FT_Load_Glyph(face, (FT_UInt)glyph_index, FT_LOAD_DEFAULT | FT_LOAD_COLOR) != 0) {
        return false;
    }

    FT_GlyphSlot slot = face->glyph;
    if (!slot) {
        return false;
    }
    if (slot->format != FT_GLYPH_FORMAT_BITMAP) {
        if (FT_Render_Glyph(slot, FT_RENDER_MODE_NORMAL) != 0) {
            return false;
        }
    }

    const uint32_t w = (uint32_t)slot->bitmap.width;
    const uint32_t h = (uint32_t)slot->bitmap.rows;
    if (w == 0u || h == 0u || !slot->bitmap.buffer) {
        return false;
    }
    // Only treat true BGRA glyph bitmaps as native color glyphs.
    // Gray/mono bitmaps belong to the normal SDF text path.
    if (slot->bitmap.pixel_mode != FT_PIXEL_MODE_BGRA) {
        return false;
    }

    uint8_t* rgba = (uint8_t*)malloc((size_t)w * (size_t)h * 4u);
    if (!rgba) {
        return false;
    }

    const int pitch = slot->bitmap.pitch;
    const int abs_pitch = pitch < 0 ? -pitch : pitch;
    const uint8_t mode = slot->bitmap.pixel_mode;
    for (uint32_t y = 0u; y < h; ++y) {
        const uint8_t* src_row = NULL;
        if (pitch >= 0) {
            src_row = slot->bitmap.buffer + (size_t)y * (size_t)abs_pitch;
        } else {
            src_row = slot->bitmap.buffer + (size_t)(h - 1u - y) * (size_t)abs_pitch;
        }
        uint8_t* dst_row = rgba + (size_t)y * (size_t)w * 4u;

        if (mode == FT_PIXEL_MODE_BGRA) {
            for (uint32_t x = 0u; x < w; ++x) {
                const uint8_t b = src_row[x * 4u + 0u];
                const uint8_t g = src_row[x * 4u + 1u];
                const uint8_t r = src_row[x * 4u + 2u];
                const uint8_t a = src_row[x * 4u + 3u];
                // FreeType color bitmap path often returns premultiplied BGRA; convert to straight RGBA.
                dst_row[x * 4u + 0u] = fs_ft2_unpremul_u8(r, a);
                dst_row[x * 4u + 1u] = fs_ft2_unpremul_u8(g, a);
                dst_row[x * 4u + 2u] = fs_ft2_unpremul_u8(b, a);
                dst_row[x * 4u + 3u] = a;
            }
        } else {
            free(rgba);
            return false;
        }
    }

    out_glyph->pixels = rgba;
    out_glyph->width = w;
    out_glyph->height = h;
    out_glyph->pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_RGBA8;
    out_glyph->offset_x = slot->bitmap_left;
    out_glyph->offset_y = -slot->bitmap_top;
    out_glyph->advance = (float)slot->advance.x / 64.0f;
    out_glyph->sdf_radius_px = 1.0f;
    out_glyph->sdf_onedge = 0.5f;
    out_glyph->sdf_pixel_dist_scale = 1.0f;
    return true;
}

static bool fs_ft2_set_font_px(FT_Face face, float font_px) {
    if (!face || font_px <= 0.0f) {
        return false;
    }
    const FT_UInt px = (FT_UInt)fmaxf(1.0f, ceilf(font_px));
    if (FT_Set_Pixel_Sizes(face, 0, px) == 0) {
        return true;
    }
    if (face->num_fixed_sizes <= 0 || !face->available_sizes) {
        return false;
    }

    int best_index = -1;
    float best_delta = 1e30f;
    for (int i = 0; i < face->num_fixed_sizes; ++i) {
        const FT_Bitmap_Size* sz = &face->available_sizes[i];
        float strike_px = (float)sz->y_ppem / 64.0f;
        if (strike_px <= 0.0f) {
            strike_px = (float)sz->height;
        }
        if (strike_px <= 0.0f) {
            continue;
        }
        const float d = fabsf(strike_px - font_px);
        if (d < best_delta) {
            best_delta = d;
            best_index = i;
        }
    }
    if (best_index < 0) {
        return false;
    }
    return FT_Select_Size(face, best_index) == 0;
}

static bool fs_ft2_get_glyph_sdf_from_index(
    FS_FT2FaceHandle* handle,
    uint32_t glyph_index,
    float font_px,
    FS_FontGlyphBitmap* out_glyph
) {
    if (!handle || !out_glyph || font_px <= 0.0f) {
        return false;
    }
    FT_Face face = handle->face;
    if (!face) {
        return false;
    }

    out_glyph->pixels = NULL;
    out_glyph->width = 0u;
    out_glyph->height = 0u;
    out_glyph->pixel_format = FS_FONT_GLYPH_PIXEL_FORMAT_SDF_R8;
    out_glyph->offset_x = 0;
    out_glyph->offset_y = 0;
    out_glyph->advance = 0.0f;
    out_glyph->sdf_radius_px = 8.0f;
    out_glyph->sdf_onedge = 0.5f;
    out_glyph->sdf_pixel_dist_scale = 1.0f;

    if (!fs_ft2_set_font_px(face, font_px)) {
        return false;
    }
    if (fs_ft2_try_get_color_glyph_rgba(handle, glyph_index, out_glyph)) {
        return true;
    }
    if (FT_Load_Glyph(face, (FT_UInt)glyph_index, FT_LOAD_RENDER | FT_LOAD_TARGET_NORMAL) != 0) {
        return false;
    }

    FT_GlyphSlot slot = face->glyph;
    out_glyph->advance = (float)slot->advance.x / 64.0f;
    const uint32_t src_w = (uint32_t)slot->bitmap.width;
    const uint32_t src_h = (uint32_t)slot->bitmap.rows;
    if (src_w == 0u || src_h == 0u) {
        return true;
    }

    FS_FT2SDFProfile profile;
    fs_ft2_pick_sdf_profile(font_px, &profile);
    const int padding = profile.padding;
    const float onedge_value = profile.onedge_value;
    const float pixel_dist_scale = profile.pixel_dist_scale;
    out_glyph->sdf_radius_px = (float)padding;
    const uint32_t w = src_w + (uint32_t)padding * 2u;
    const uint32_t h = src_h + (uint32_t)padding * 2u;
    uint8_t* coverage = (uint8_t*)calloc((size_t)w * (size_t)h, 1u);
    if (!coverage) {
        return false;
    }

    const int pitch = slot->bitmap.pitch;
    for (uint32_t y = 0; y < src_h; ++y) {
        const uint8_t* src_row = NULL;
        if (pitch >= 0) {
            src_row = slot->bitmap.buffer + (size_t)y * (size_t)pitch;
        } else {
            src_row = slot->bitmap.buffer + (size_t)(src_h - 1u - y) * (size_t)(-pitch);
        }
        uint8_t* dst_row = coverage + (size_t)(y + (uint32_t)padding) * (size_t)w + (size_t)padding;
        memcpy(dst_row, src_row, (size_t)src_w);
    }

    if (profile.solidify_passes > 0) {
        fs_ft2_solidify_coverage(coverage, w, h, profile.solidify_passes);
    }
    if (profile.preprocess_strength > 0.0f) {
        fs_ft2_preprocess_coverage(coverage, w, h, profile.preprocess_strength);
    }

    const float onedge_norm = onedge_value / 255.0f;
    uint8_t* sdf = NULL;
    if (!fs_ft2_build_sdf_from_coverage_esdt(
            coverage,
            w,
            h,
            padding,
            onedge_value,
            pixel_dist_scale,
            profile.refine_mix,
            profile.postprocess_passes,
            &sdf
        )) {
        free(coverage);
        return false;
    }
    free(coverage);

    out_glyph->pixels = sdf;
    out_glyph->width = w;
    out_glyph->height = h;
    out_glyph->offset_x = slot->bitmap_left - padding;
    out_glyph->offset_y = -slot->bitmap_top - padding;
    out_glyph->sdf_onedge = onedge_norm;
    out_glyph->sdf_pixel_dist_scale = pixel_dist_scale;
    return true;
}

static bool fs_ft2_get_glyph_sdf(
    void* font_handle,
    uint32_t codepoint,
    float font_px,
    FS_FontGlyphBitmap* out_glyph
) {
    if (!font_handle || !out_glyph || font_px <= 0.0f) {
        return false;
    }
    FS_FT2FaceHandle* handle = (FS_FT2FaceHandle*)font_handle;
    FT_Face face = handle->face;
    if (!face) {
        return false;
    }
    const FT_UInt glyph_index = FT_Get_Char_Index(face, codepoint);
    if (glyph_index == 0u && codepoint != 0u) {
        // Missing glyph: report miss to caller so layout can skip instead of consuming .notdef advance.
        return false;
    }
    return fs_ft2_get_glyph_sdf_from_index(handle, glyph_index, font_px, out_glyph);
}

static bool fs_ft2_get_glyph_sdf_by_index(
    void* font_handle,
    uint32_t glyph_index,
    float font_px,
    FS_FontGlyphBitmap* out_glyph
) {
    if (!font_handle || !out_glyph || font_px <= 0.0f) {
        return false;
    }
    FS_FT2FaceHandle* handle = (FS_FT2FaceHandle*)font_handle;
    return fs_ft2_get_glyph_sdf_from_index(handle, glyph_index, font_px, out_glyph);
}

static float fs_ft2_get_kerning_advance(
    void* font_handle,
    uint32_t left_codepoint,
    uint32_t right_codepoint,
    float font_px
) {
    if (!font_handle || font_px <= 0.0f) {
        return 0.0f;
    }
    FS_FT2FaceHandle* handle = (FS_FT2FaceHandle*)font_handle;
    FT_Face face = handle->face;
    if (!face || !FT_HAS_KERNING(face)) {
        return 0.0f;
    }
    if (!fs_ft2_set_font_px(face, font_px)) {
        return 0.0f;
    }
    const FT_UInt left_index = FT_Get_Char_Index(face, left_codepoint);
    const FT_UInt right_index = FT_Get_Char_Index(face, right_codepoint);
    if (left_index == 0u || right_index == 0u) {
        return 0.0f;
    }
    FT_Vector delta = {0};
    if (FT_Get_Kerning(face, left_index, right_index, FT_KERNING_DEFAULT, &delta) != 0) {
        return 0.0f;
    }
    return (float)delta.x / 64.0f;
}

static bool fs_ft2_get_vertical_metrics(
    void* font_handle,
    float font_px,
    FS_FontVerticalMetrics* out_metrics
) {
    if (!font_handle || !out_metrics || font_px <= 0.0f) {
        return false;
    }
    FS_FT2FaceHandle* handle = (FS_FT2FaceHandle*)font_handle;
    FT_Face face = handle->face;
    if (!face) {
        return false;
    }
    if (!fs_ft2_set_font_px(face, font_px)) {
        return false;
    }
    const float ascent = (float)face->size->metrics.ascender / 64.0f;
    const float descent = (float)(-face->size->metrics.descender) / 64.0f;
    float line_height = (float)face->size->metrics.height / 64.0f;
    if (!(ascent > 0.0f) || !(descent >= 0.0f)) {
        return false;
    }
    if (!(line_height > 0.0f)) {
        line_height = ascent + descent;
    } else if (line_height < ascent + descent) {
        line_height = ascent + descent;
    }
    out_metrics->ascent = ascent;
    out_metrics->descent = descent;
    out_metrics->line_height = line_height;
    return true;
}

static bool fs_ft2_shape_text_utf8(
    void* font_handle,
    const char* utf8,
    float font_px,
    FS_ShapedTextRun* out_run
) {
    if (!font_handle || !utf8 || !out_run || font_px <= 0.0f) {
        return false;
    }
    out_run->glyphs = NULL;
    out_run->glyph_count = 0u;
#ifndef FS_HAS_HARFBUZZ
    return false;
#else
    FS_FT2FaceHandle* handle = (FS_FT2FaceHandle*)font_handle;
    if (!handle->hb_font) {
        return false;
    }

    hb_buffer_t* buffer = hb_buffer_create();
    if (!buffer) {
        return false;
    }

    hb_buffer_add_utf8(buffer, utf8, -1, 0, -1);
    hb_buffer_guess_segment_properties(buffer);

    const int32_t px_26d6 = (int32_t)(font_px * 64.0f + 0.5f);
    hb_font_set_scale(handle->hb_font, px_26d6, px_26d6);
    hb_ft_font_changed(handle->hb_font);
    hb_shape(handle->hb_font, buffer, NULL, 0);

    unsigned int glyph_count = 0u;
    const hb_glyph_info_t* infos = hb_buffer_get_glyph_infos(buffer, &glyph_count);
    const hb_glyph_position_t* positions = hb_buffer_get_glyph_positions(buffer, &glyph_count);
    if (!infos || !positions || glyph_count == 0u) {
        hb_buffer_destroy(buffer);
        return true;
    }

    FS_ShapedGlyph* out_glyphs = (FS_ShapedGlyph*)calloc((size_t)glyph_count, sizeof(FS_ShapedGlyph));
    if (!out_glyphs) {
        hb_buffer_destroy(buffer);
        return false;
    }
    for (unsigned int i = 0u; i < glyph_count; ++i) {
        out_glyphs[i].glyph_index = infos[i].codepoint;
        out_glyphs[i].x_offset_26d6 = positions[i].x_offset;
        out_glyphs[i].y_offset_26d6 = positions[i].y_offset;
        out_glyphs[i].x_advance_26d6 = positions[i].x_advance;
        out_glyphs[i].y_advance_26d6 = positions[i].y_advance;
    }
    hb_buffer_destroy(buffer);

    out_run->glyphs = out_glyphs;
    out_run->glyph_count = glyph_count;
    return true;
#endif
}

static void fs_ft2_free_shaped_text(FS_ShapedTextRun* run) {
    if (!run) {
        return;
    }
    free(run->glyphs);
    run->glyphs = NULL;
    run->glyph_count = 0u;
}

static void fs_ft2_free_glyph_pixels(uint8_t* pixels) {
    free(pixels);
}

static const FS_FontBackend g_ft2_font_backend = {
    .name = "freetype2",
    .load_font_file = fs_ft2_load_font_file,
    .destroy_font = fs_ft2_destroy_font,
    .get_glyph_sdf = fs_ft2_get_glyph_sdf,
    .get_glyph_sdf_by_index = fs_ft2_get_glyph_sdf_by_index,
    .get_kerning_advance = fs_ft2_get_kerning_advance,
    .get_vertical_metrics = fs_ft2_get_vertical_metrics,
    .shape_text_utf8 = fs_ft2_shape_text_utf8,
    .free_shaped_text = fs_ft2_free_shaped_text,
    .free_glyph_pixels = fs_ft2_free_glyph_pixels
};

#else

static const FS_FontBackend g_ft2_font_backend = {
    .name = "freetype2(unavailable)",
    .load_font_file = NULL,
    .destroy_font = NULL,
    .get_glyph_sdf = NULL,
    .get_glyph_sdf_by_index = NULL,
    .get_kerning_advance = NULL,
    .get_vertical_metrics = NULL,
    .shape_text_utf8 = NULL,
    .free_shaped_text = NULL,
    .free_glyph_pixels = NULL
};

#endif

const FS_FontBackend* fs_get_freetype2_font_backend(void) {
    return &g_ft2_font_backend;
}
