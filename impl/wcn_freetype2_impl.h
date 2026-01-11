#ifndef WCN_FREETYPE2_IMPL_H
#define WCN_FREETYPE2_IMPL_H

#include "WCN/WCN.h"

#include <ft2build.h>
#include FT_FREETYPE_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// FreeType2 字体解码器实现
// ============================================================================

// 字体私有数据
typedef struct {
    FT_Library library;
    FT_Face face;
    unsigned char* font_buffer;
    size_t buffer_size;
} WCN_FT2_FontData;

// 辅助：简单的 2D 向量
typedef struct { float x, y; } WCN_FT2_Vec2;

// 辅助：计算平方距离
static float wcn_ft2_dist_sq(WCN_FT2_Vec2 a, WCN_FT2_Vec2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx*dx + dy*dy;
}

// 辅助：ESDT (Euclidean Signed Distance Transform) 算法
// 输入: alpha_map (w x h)
// 输出: dist_map (w x h), offset_map (w x h)
// spread: SDF 扩散半径
static void wcn_ft2_compute_esdt(const unsigned char* alpha_map, int w, int h, int spread,
                                 float* out_dist, WCN_FT2_Vec2* out_offsets) {
    // 初始化
    int count = w * h;
    const float INF = 1e9f;

    // 这里的网格用于存储最近的"边界像素"的坐标
    WCN_FT2_Vec2* grid_inside = (WCN_FT2_Vec2*)malloc(count * sizeof(WCN_FT2_Vec2));
    WCN_FT2_Vec2* grid_outside = (WCN_FT2_Vec2*)malloc(count * sizeof(WCN_FT2_Vec2));

    if (!grid_inside || !grid_outside) {
        free(grid_inside);
        free(grid_outside);
        return;
    }

    // 1. 初始化网格
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            unsigned char a = alpha_map[idx];

            grid_inside[idx] = (WCN_FT2_Vec2){INF, INF};
            grid_outside[idx] = (WCN_FT2_Vec2){INF, INF};

            if (a >= 128) {
                // 内部像素
                grid_inside[idx].x = (float)x;
                grid_inside[idx].y = (float)y;
            } else {
                // 外部像素
                grid_outside[idx].x = (float)x;
                grid_outside[idx].y = (float)y;
            }
        }
    }

    // 2. 传播距离 (Dead Reckoning / Chamfer Distance 的两遍扫描变体)
    // Pass 1: Forward (top-left to bottom-right)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            WCN_FT2_Vec2 p = {(float)x, (float)y};

            // 检查周围像素 (左，上，左上，右上)
            int neighbors[4][2] = {{-1, 0}, {0, -1}, {-1, -1}, {1, -1}};

            for (int k = 0; k < 4; k++) {
                int nx = x + neighbors[k][0];
                int ny = y + neighbors[k][1];

                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    int nidx = ny * w + nx;

                    // Update Inside Grid
                    if (grid_inside[nidx].x != INF) {
                        float d_curr = wcn_ft2_dist_sq(p, grid_inside[idx]);
                        float d_new = wcn_ft2_dist_sq(p, grid_inside[nidx]);
                        if (d_new < d_curr) grid_inside[idx] = grid_inside[nidx];
                    }

                    // Update Outside Grid
                    if (grid_outside[nidx].x != INF) {
                        float d_curr = wcn_ft2_dist_sq(p, grid_outside[idx]);
                        float d_new = wcn_ft2_dist_sq(p, grid_outside[nidx]);
                        if (d_new < d_curr) grid_outside[idx] = grid_outside[nidx];
                    }
                }
            }
        }
    }

    // Pass 2: Backward (bottom-right to top-left)
    for (int y = h - 1; y >= 0; y--) {
        for (int x = w - 1; x >= 0; x--) {
            int idx = y * w + x;
            WCN_FT2_Vec2 p = {(float)x, (float)y};

            // 检查周围像素 (右，下，右下，左下)
            int neighbors[4][2] = {{1, 0}, {0, 1}, {1, 1}, {-1, 1}};

            for (int k = 0; k < 4; k++) {
                int nx = x + neighbors[k][0];
                int ny = y + neighbors[k][1];

                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    int nidx = ny * w + nx;

                    // Update Inside Grid
                    if (grid_inside[nidx].x != INF) {
                        float d_curr = wcn_ft2_dist_sq(p, grid_inside[idx]);
                        float d_new = wcn_ft2_dist_sq(p, grid_inside[nidx]);
                        if (d_new < d_curr) grid_inside[idx] = grid_inside[nidx];
                    }

                    // Update Outside Grid
                    if (grid_outside[nidx].x != INF) {
                        float d_curr = wcn_ft2_dist_sq(p, grid_outside[idx]);
                        float d_new = wcn_ft2_dist_sq(p, grid_outside[nidx]);
                        if (d_new < d_curr) grid_outside[idx] = grid_outside[nidx];
                    }
                }
            }
        }
    }

    // 3. 合成 SDF
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            WCN_FT2_Vec2 p = {(float)x, (float)y};

            float dist_to_outside = sqrtf(wcn_ft2_dist_sq(p, grid_outside[idx]));
            float dist_to_inside = sqrtf(wcn_ft2_dist_sq(p, grid_inside[idx]));

            float sdf = 0.0f;
            WCN_FT2_Vec2 nearest = {0, 0};

            unsigned char alpha = alpha_map[idx];
            if (alpha >= 128) {
                sdf = dist_to_outside - 0.5f;
                nearest = grid_outside[idx];
            } else {
                sdf = -(dist_to_inside - 0.5f);
                nearest = grid_inside[idx];
            }

            out_dist[idx] = sdf;
            out_offsets[idx].x = nearest.x - p.x;
            out_offsets[idx].y = nearest.y - p.y;

            if (alpha > 0 && alpha < 255) {
                float alpha_dist = 0.5f - ((float)alpha / 255.0f);
                out_dist[idx] = alpha_dist;
            }
        }
    }

    free(grid_inside);
    free(grid_outside);
}

// 加载字体
static bool wcn_ft2_load_font(const void* font_data, size_t data_size, WCN_FontFace** out_face) {
    if (!font_data || data_size == 0 || !out_face) {
        return false;
    }

    WCN_FT2_FontData* font_priv = malloc(sizeof(WCN_FT2_FontData));
    if (!font_priv) {
        return false;
    }

    if (FT_Init_FreeType(&font_priv->library)) {
        free(font_priv);
        return false;
    }

    font_priv->font_buffer = malloc(data_size);
    if (!font_priv->font_buffer) {
        FT_Done_FreeType(font_priv->library);
        free(font_priv);
        return false;
    }
    memcpy(font_priv->font_buffer, font_data, data_size);
    font_priv->buffer_size = data_size;

    if (FT_New_Memory_Face(font_priv->library, font_priv->font_buffer, data_size, 0, &font_priv->face)) {
        free(font_priv->font_buffer);
        FT_Done_FreeType(font_priv->library);
        free(font_priv);
        return false;
    }

    WCN_FontFace* face = malloc(sizeof(WCN_FontFace));
    if (!face) {
        FT_Done_Face(font_priv->face);
        free(font_priv->font_buffer);
        FT_Done_FreeType(font_priv->library);
        free(font_priv);
        return false;
    }

    face->family_name = font_priv->face->family_name ? font_priv->face->family_name : "Unknown";
    face->ascent = (float)font_priv->face->ascender;
    face->descent = (float)font_priv->face->descender;
    face->line_gap = (float)font_priv->face->height - (face->ascent - face->descent);
    face->units_per_em = (float)font_priv->face->units_per_EM;
    face->user_data = font_priv;

    printf("FreeType2: 字体加载成功 '%s' (ascent=%.1f, descent=%.1f, units_per_em=%.1f)\n",
           face->family_name, face->ascent, face->descent, face->units_per_em);

    *out_face = face;
    return true;
}

// 获取字形（轮廓数据）
static bool wcn_ft2_get_glyph(WCN_FontFace* face, uint32_t codepoint, WCN_Glyph** out_glyph) {
    if (!face || !out_glyph) {
        return false;
    }

    WCN_FT2_FontData* font_data = (WCN_FT2_FontData*)face->user_data;

    FT_UInt glyph_index = FT_Get_Char_Index(font_data->face, codepoint);
    if (glyph_index == 0) {
        return false;
    }

    if (FT_Load_Glyph(font_data->face, glyph_index, FT_LOAD_NO_SCALE)) {
        return false;
    }

    WCN_Glyph* glyph = malloc(sizeof(WCN_Glyph));
    if (!glyph) {
        return false;
    }

    FT_GlyphSlot slot = font_data->face->glyph;

    glyph->codepoint = codepoint;
    glyph->contours = NULL;
    glyph->contour_count = 0;
    glyph->advance_width = (float)slot->advance.x;
    glyph->left_side_bearing = (float)slot->metrics.horiBearingX;
    glyph->bounding_box[0] = (float)slot->metrics.horiBearingX;
    glyph->bounding_box[1] = (float)slot->metrics.horiBearingY - slot->metrics.height;
    glyph->bounding_box[2] = (float)slot->metrics.horiBearingX + slot->metrics.width;
    glyph->bounding_box[3] = (float)slot->metrics.horiBearingY;
    glyph->vertices = NULL;
    glyph->indices = NULL;
    glyph->vertex_count = 0;
    glyph->index_count = 0;
    glyph->raw_vertices = NULL;
    glyph->raw_vertex_count = 0;

    *out_glyph = glyph;
    return true;
}

// 获取字形 SDF 位图
static bool wcn_ft2_get_glyph_sdf(WCN_FontFace* face, uint32_t codepoint, float font_size,
                                  unsigned char** out_bitmap,
                                  int* out_width, int* out_height,
                                  float* out_offset_x, float* out_offset_y,
                                  float* out_advance,
                                  bool* out_is_color) {
    if (!face || !out_bitmap || !out_width || !out_height) {
        return false;
    }

    if (out_is_color) *out_is_color = false;

    WCN_FT2_FontData* font_data = (WCN_FT2_FontData*)face->user_data;

    FT_UInt glyph_index = FT_Get_Char_Index(font_data->face, codepoint);
    if (glyph_index == 0) return false;

    if (FT_Set_Pixel_Sizes(font_data->face, 0, (FT_UInt)font_size)) {
        return false;
    }

    const int OVERSAMPLE = 4;
    FT_UInt high_res_size = (FT_UInt)(font_size * OVERSAMPLE);
    if (FT_Set_Pixel_Sizes(font_data->face, 0, high_res_size)) {
        return false;
    }

    if (FT_Load_Glyph(font_data->face, glyph_index, FT_LOAD_RENDER)) {
        return false;
    }

    FT_GlyphSlot slot = font_data->face->glyph;
    FT_Bitmap* bitmap = &slot->bitmap;

    if (bitmap->width == 0 || bitmap->rows == 0) {
        return false;
    }

    int spread = 4;
    int padding = spread + 1;

    int content_width = bitmap->width;
    int content_height = bitmap->rows;

    int high_res_w = content_width + padding * 2 * OVERSAMPLE;
    int high_res_h = content_height + padding * 2 * OVERSAMPLE;

    unsigned char* alpha_bitmap = (unsigned char*)calloc(high_res_w * high_res_h, 1);
    if (!alpha_bitmap) return false;

    for (unsigned int y = 0; y < bitmap->rows; y++) {
        for (unsigned int x = 0; x < bitmap->width; x++) {
            int dst_x = x + padding * OVERSAMPLE;
            int dst_y = y + padding * OVERSAMPLE;
            int dst_idx = dst_y * high_res_w + dst_x;
            int src_idx = y * bitmap->pitch + x;
            alpha_bitmap[dst_idx] = bitmap->buffer[src_idx];
        }
    }

    int high_res_spread = spread * OVERSAMPLE;

    float* dist_map = (float*)malloc(high_res_w * high_res_h * sizeof(float));
    WCN_FT2_Vec2* offsets_map = (WCN_FT2_Vec2*)malloc(high_res_w * high_res_h * sizeof(WCN_FT2_Vec2));

    if (!dist_map || !offsets_map) {
        free(alpha_bitmap);
        if(dist_map) free(dist_map);
        if(offsets_map) free(offsets_map);
        return false;
    }

    wcn_ft2_compute_esdt(alpha_bitmap, high_res_w, high_res_h, high_res_spread, dist_map, offsets_map);

    int target_w = (content_width / OVERSAMPLE) + padding * 2;
    int target_h = (content_height / OVERSAMPLE) + padding * 2;

    unsigned char* output_rgba = (unsigned char*)malloc(target_w * target_h * 4);
    if (!output_rgba) {
        free(alpha_bitmap);
        free(dist_map);
        free(offsets_map);
        return false;
    }

    float inv_range = 1.0f / (float)(spread * 2);

    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            int src_x = x * OVERSAMPLE + OVERSAMPLE / 2;
            int src_y = y * OVERSAMPLE + OVERSAMPLE / 2;

            if (src_x >= high_res_w) src_x = high_res_w - 1;
            if (src_y >= high_res_h) src_y = high_res_h - 1;

            int src_idx = src_y * high_res_w + src_x;

            float d_high = dist_map[src_idx];
            float off_x_high = offsets_map[src_idx].x;
            float off_y_high = offsets_map[src_idx].y;

            float d_target = d_high / OVERSAMPLE;
            float off_x_target = off_x_high / OVERSAMPLE;
            float off_y_target = off_y_high / OVERSAMPLE;

            float norm_d = 0.5f + (d_target * inv_range);
            norm_d = norm_d < 0.0f ? 0.0f : (norm_d > 1.0f ? 1.0f : norm_d);

            float norm_off_x = 0.5f + (off_x_target * inv_range);
            float norm_off_y = 0.5f + (off_y_target * inv_range);

            norm_off_x = norm_off_x < 0.0f ? 0.0f : (norm_off_x > 1.0f ? 1.0f : norm_off_x);
            norm_off_y = norm_off_y < 0.0f ? 0.0f : (norm_off_y > 1.0f ? 1.0f : norm_off_y);

            int dst_idx = y * target_w + x;
            output_rgba[dst_idx*4 + 0] = (unsigned char)(norm_d * 255.0f);
            output_rgba[dst_idx*4 + 1] = (unsigned char)(norm_off_x * 255.0f);
            output_rgba[dst_idx*4 + 2] = (unsigned char)(norm_off_y * 255.0f);
            output_rgba[dst_idx*4 + 3] = 255;
        }
    }

    free(alpha_bitmap);
    free(dist_map);
    free(offsets_map);

    *out_bitmap = output_rgba;
    *out_width = target_w;
    *out_height = target_h;

    float scale = font_size / face->units_per_em;
    *out_offset_x = (float)slot->bitmap_left / OVERSAMPLE - padding;
    *out_offset_y = (float)slot->bitmap_top / OVERSAMPLE - padding;
    *out_advance = (float)slot->advance.x / 64.0f / OVERSAMPLE;

    return true;
}

// 释放 SDF 位图
static void wcn_ft2_free_glyph_sdf(unsigned char* bitmap) {
    if (bitmap) {
        free(bitmap);
    }
}

// 测量文本
static bool wcn_ft2_measure_text(WCN_FontFace* face, const char* text, float font_size,
                                 float* out_width, float* out_height) {
    if (!face || !text || !out_width || !out_height) {
        return false;
    }

    WCN_FT2_FontData* font_data = (WCN_FT2_FontData*)face->user_data;

    if (FT_Set_Pixel_Sizes(font_data->face, 0, (FT_UInt)font_size)) {
        return false;
    }

    float width = 0;
    const char* ptr = text;

    while (*ptr) {
        int codepoint = *ptr++;

        FT_UInt glyph_index = FT_Get_Char_Index(font_data->face, codepoint);
        if (glyph_index == 0) continue;

        if (FT_Load_Glyph(font_data->face, glyph_index, FT_LOAD_DEFAULT)) {
            continue;
        }

        width += font_data->face->glyph->advance.x / 64.0f;
    }

    *out_width = width;
    *out_height = font_size;

    return true;
}

// 释放字形
static void wcn_ft2_free_glyph(WCN_Glyph* glyph) {
    if (glyph) {
        free(glyph->contours);
        free(glyph->vertices);
        free(glyph->indices);
        free(glyph->raw_vertices);
        free(glyph);
    }
}

// 释放字体
static void wcn_ft2_free_font(WCN_FontFace* face) {
    if (face) {
        WCN_FT2_FontData* font_data = (WCN_FT2_FontData*)face->user_data;
        if (font_data) {
            if (font_data->face) {
                FT_Done_Face(font_data->face);
            }
            if (font_data->library) {
                FT_Done_FreeType(font_data->library);
            }
            free(font_data->font_buffer);
            free(font_data);
        }
        free(face);
    }
}

// 全局解码器实例
static WCN_FontDecoder wcn_freetype2_decoder = {
    .load_font = wcn_ft2_load_font,
    .get_glyph = wcn_ft2_get_glyph,
    .get_glyph_sdf = wcn_ft2_get_glyph_sdf,
    .free_glyph_sdf = wcn_ft2_free_glyph_sdf,
    .measure_text = wcn_ft2_measure_text,
    .free_glyph = wcn_ft2_free_glyph,
    .free_font = wcn_ft2_free_font,
    .name = "FreeType2"
};

// 获取解码器实例
static inline WCN_FontDecoder* wcn_get_freetype2_decoder(void) {
    return &wcn_freetype2_decoder;
}

#endif // WCN_FREETYPE2_IMPL_H