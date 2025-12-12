#ifndef WCN_STB_TRUETYPE_IMPL_H
#define WCN_STB_TRUETYPE_IMPL_H

#include "WCN/WCN.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// stb_truetype 字体解码器实现
// ============================================================================

// 字体私有数据
typedef struct {
    stbtt_fontinfo font_info;
    unsigned char* font_buffer;
    size_t buffer_size;
} WCN_STB_FontData;

// 加载字体
static bool wcn_stb_load_font(const void* font_data, size_t data_size, WCN_FontFace** out_face) {
    if (!font_data || data_size == 0 || !out_face) {
        return false;
    }
    
    // 分配字体数据
    WCN_STB_FontData* font_priv = malloc(sizeof(WCN_STB_FontData));
    if (!font_priv) {
        return false;
    }
    
    // 复制字体数据
    font_priv->font_buffer = malloc(data_size);
    if (!font_priv->font_buffer) {
        free(font_priv);
        return false;
    }
    memcpy(font_priv->font_buffer, font_data, data_size);
    font_priv->buffer_size = data_size;
    
    // 初始化 stb_truetype
    if (!stbtt_InitFont(&font_priv->font_info, font_priv->font_buffer, 0)) {
        free(font_priv->font_buffer);
        free(font_priv);
        return false;
    }
    
    // 创建 WCN_FontFace
    WCN_FontFace* face = malloc(sizeof(WCN_FontFace));
    if (!face) {
        free(font_priv->font_buffer);
        free(font_priv);
        return false;
    }
    
    // 获取字体度量
    int ascent, descent, line_gap;
    stbtt_GetFontVMetrics(&font_priv->font_info, &ascent, &descent, &line_gap);
    
    // STB TrueType的度量值已经是字体单位，scale为1.0时对应units_per_em
    // 我们使用ScaleForPixelHeight来计算实际的units_per_em
    // 对于大多数字体，units_per_em通常是1000或2048
    // 我们可以通过反推得到：如果scale * units_per_em = pixel_height
    // 那么 units_per_em = pixel_height / scale
    // 但更简单的方法是：stb返回的度量值就是以units_per_em为单位的
    // 所以我们直接使用ascent作为参考，通常ascent接近units_per_em的80%左右
    
    face->family_name = "Unknown";  // stb_truetype 不提供字体名称
    face->ascent = (float)ascent;
    face->descent = (float)descent;
    face->line_gap = (float)line_gap;
    // 使用ascent + abs(descent)作为units_per_em的近似值
    // 这对大多数字体都是合理的
    face->units_per_em = (float)(ascent - descent);
    face->user_data = font_priv;
    
    *out_face = face;
    
    printf("stb_truetype: 字体加载成功 (ascent=%.1f, descent=%.1f)\n", 
           face->ascent, face->descent);
    
    return true;
}

// 获取字形（轮廓数据）
static bool wcn_stb_get_glyph(WCN_FontFace* face, uint32_t codepoint, WCN_Glyph** out_glyph) {
    if (!face || !out_glyph) {
        return false;
    }
    
    WCN_STB_FontData* font_data = (WCN_STB_FontData*)face->user_data;
    
    // 获取字形索引
    int glyph_index = stbtt_FindGlyphIndex(&font_data->font_info, codepoint);
    if (glyph_index == 0) {
        return false;  // 字形不存在
    }
    
    // 获取字形度量
    int advance, lsb;
    stbtt_GetGlyphHMetrics(&font_data->font_info, glyph_index, &advance, &lsb);
    
    // 获取边界框
    int x0, y0, x1, y1;
    stbtt_GetGlyphBox(&font_data->font_info, glyph_index, &x0, &y0, &x1, &y1);
    
    // 创建字形
    WCN_Glyph* glyph = malloc(sizeof(WCN_Glyph));
    if (!glyph) {
        return false;
    }
    
    glyph->codepoint = codepoint;
    glyph->contours = NULL;
    glyph->contour_count = 0;
    glyph->advance_width = (float)advance;
    glyph->left_side_bearing = (float)lsb;
    glyph->bounding_box[0] = (float)x0;
    glyph->bounding_box[1] = (float)y0;
    glyph->bounding_box[2] = (float)x1;
    glyph->bounding_box[3] = (float)y1;
    glyph->vertices = NULL;
    glyph->indices = NULL;
    glyph->vertex_count = 0;
    glyph->index_count = 0;
    glyph->raw_vertices = NULL;
    glyph->raw_vertex_count = 0;
    
    *out_glyph = glyph;
    return true;
}

// 辅助：简单的 2D 向量
typedef struct { float x, y; } WCN_Vec2;

// 辅助：计算平方距离
static float wcn_dist_sq(WCN_Vec2 a, WCN_Vec2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx*dx + dy*dy;
}

// 辅助：ESDT (Euclidean Signed Distance Transform) 算法
// 输入: alpha_map (w x h)
// 输出: dist_map (w x h), offset_map (w x h)
// spread: SDF 扩散半径
static void wcn_compute_esdt(const unsigned char* alpha_map, int w, int h, int spread,
                             float* out_dist, WCN_Vec2* out_offsets) {
    // 初始化
    int count = w * h;
    const float INF = 1e9f;
    
    // 这里的网格用于存储最近的"边界像素"的坐标
    WCN_Vec2* grid_inside = (WCN_Vec2*)malloc(count * sizeof(WCN_Vec2));
    WCN_Vec2* grid_outside = (WCN_Vec2*)malloc(count * sizeof(WCN_Vec2));
    
    if (!grid_inside || !grid_outside) {
        free(grid_inside);
        free(grid_outside);
        return;
    }

    // 1. 初始化网格
    // 这里的"边界"定义为 alpha 在 (0, 255) 之间的区域，或者是 alpha 阈值穿越的地方
    // 为了简化且高效，我们将 alpha >= 128 视为内部，< 128 视为外部
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            unsigned char a = alpha_map[idx];
            
            grid_inside[idx] = (WCN_Vec2){INF, INF};
            grid_outside[idx] = (WCN_Vec2){INF, INF};
            
            // 如果是边界像素
            // 简单的阈值判定法：
            if (a >= 128) {
                // 内部像素
                // 初始化 grid_inside 为自身坐标
                grid_inside[idx].x = (float)x;
                grid_inside[idx].y = (float)y;
            } else {
                // 外部像素
                // 初始化 grid_outside 为自身坐标
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
            WCN_Vec2 p = {(float)x, (float)y};
            
            // 检查周围像素 (左，上，左上，右上)
            int neighbors[4][2] = {{-1, 0}, {0, -1}, {-1, -1}, {1, -1}};
            
            for (int k = 0; k < 4; k++) {
                int nx = x + neighbors[k][0];
                int ny = y + neighbors[k][1];
                
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    int nidx = ny * w + nx;
                    
                    // Update Inside Grid
                    if (grid_inside[nidx].x != INF) {
                        float d_curr = wcn_dist_sq(p, grid_inside[idx]);
                        float d_new = wcn_dist_sq(p, grid_inside[nidx]);
                        if (d_new < d_curr) grid_inside[idx] = grid_inside[nidx];
                    }
                    
                    // Update Outside Grid
                    if (grid_outside[nidx].x != INF) {
                        float d_curr = wcn_dist_sq(p, grid_outside[idx]);
                        float d_new = wcn_dist_sq(p, grid_outside[nidx]);
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
            WCN_Vec2 p = {(float)x, (float)y};
            
            // 检查周围像素 (右，下，右下，左下)
            int neighbors[4][2] = {{1, 0}, {0, 1}, {1, 1}, {-1, 1}};
            
            for (int k = 0; k < 4; k++) {
                int nx = x + neighbors[k][0];
                int ny = y + neighbors[k][1];
                
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    int nidx = ny * w + nx;
                    
                    // Update Inside Grid
                    if (grid_inside[nidx].x != INF) {
                        float d_curr = wcn_dist_sq(p, grid_inside[idx]);
                        float d_new = wcn_dist_sq(p, grid_inside[nidx]);
                        if (d_new < d_curr) grid_inside[idx] = grid_inside[nidx];
                    }
                    
                    // Update Outside Grid
                    if (grid_outside[nidx].x != INF) {
                        float d_curr = wcn_dist_sq(p, grid_outside[idx]);
                        float d_new = wcn_dist_sq(p, grid_outside[nidx]);
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
            WCN_Vec2 p = {(float)x, (float)y};
            
            // 计算到最近的"外部像素"的距离 (dist to outside)
            float dist_to_outside = sqrtf(wcn_dist_sq(p, grid_outside[idx]));
            // 计算到最近的"内部像素"的距离 (dist to inside)
            float dist_to_inside = sqrtf(wcn_dist_sq(p, grid_inside[idx]));
            
            float sdf = 0.0f;
            WCN_Vec2 nearest = {0, 0};
            
            unsigned char alpha = alpha_map[idx];
            if (alpha >= 128) {
                // Inside: 距离是负的 (或者在 0.5-1.0 范围映射)
                // dist_to_outside 代表我们离背景有多远
                sdf = dist_to_outside - 0.5f; // -0.5 是因为 grid 指向的是像素中心
                nearest = grid_outside[idx];
            } else {
                // Outside: 距离是正的
                // dist_to_inside 代表我们离形状有多远
                sdf = -(dist_to_inside - 0.5f);
                nearest = grid_inside[idx];
            }
            
            // 存储距离 (Regular SDF)
            out_dist[idx] = sdf;
            
            // 存储 Offset (Vector to boundary)
            // 指向边界的向量
            out_offsets[idx].x = nearest.x - p.x;
            out_offsets[idx].y = nearest.y - p.y;
            
            // Hybrid Approach:
            // 1. Geometric SDF (sdf) gives robust global distance.
            // 2. Alpha value gives precise local distance for the edge.
            // If we blindly threshold at 128, thin strokes (max alpha < 128) disappear (broken paths).
            // If we blindly trust alpha, we get noise (hairy edges).
            // Solution: Use alpha-derived distance when near the "visual" edge (alpha > 0 and < 255).
            if (alpha > 0 && alpha < 255) {
                // Map alpha 0..255 to distance +0.5..-0.5
                // alpha 0   => dist +0.5 (outside)
                // alpha 128 => dist  0.0 (edge)
                // alpha 255 => dist -0.5 (inside)
                float alpha_dist = 0.5f - ((float)alpha / 255.0f);
                
                // Only replace if the geometric SDF is consistent with the alpha hint
                // or if we are dealing with a thin feature (where geometric SDF might be wrong).
                // For thin features, geometric SDF might say "+2.0" (outside) while alpha says "+0.1" (near edge).
                // We trust alpha if it indicates we are 'close' to the edge.
                out_dist[idx] = alpha_dist;
            }
        }
    }
    
    free(grid_inside);
    free(grid_outside);
}

// 获取字形 MSDF 位图（核心功能 - 替换为 Alpha+Offsets Pipeline）
static bool wcn_stb_get_glyph_sdf(WCN_FontFace* face, uint32_t codepoint, float font_size,
                                  unsigned char** out_bitmap,
                                  int* out_width, int* out_height,
                                  float* out_offset_x, float* out_offset_y,
                                  float* out_advance,
                                  bool* out_is_color) {
    if (!face || !out_bitmap || !out_width || !out_height) {
        return false;
    }

    // Default to SDF mode (monochrome)
    if (out_is_color) *out_is_color = false;
    
    WCN_STB_FontData* font_data = (WCN_STB_FontData*)face->user_data;
    
    // 1. 获取基础字形 Alpha 位图
    float scale = stbtt_ScaleForPixelHeight(&font_data->font_info, font_size);
    int glyph_index = stbtt_FindGlyphIndex(&font_data->font_info, codepoint);
    if (glyph_index == 0) return false;

    // Use 4x Oversampling for high-quality SDF generation
    const int OVERSAMPLE = 4;
    float high_res_scale = scale * OVERSAMPLE;

    int x0, y0, x1, y1;
    stbtt_GetGlyphBitmapBox(&font_data->font_info, glyph_index, scale, scale, &x0, &y0, &x1, &y1);
    
    int content_width = x1 - x0;
    int content_height = y1 - y0;
    if (content_width <= 0 || content_height <= 0) return false;
    
    // Spread in target pixels
    int spread = 4; 
    int padding = spread + 1;
    
    // Target dimensions
    int target_w = content_width + padding * 2;
    int target_h = content_height + padding * 2;
    
    // High-res dimensions
    int high_res_w = target_w * OVERSAMPLE;
    int high_res_h = target_h * OVERSAMPLE;
    
    unsigned char* alpha_bitmap = (unsigned char*)calloc(high_res_w * high_res_h, 1);
    if (!alpha_bitmap) return false;
    
    // Render Alpha at 4x resolution
    // Note: Padding is also scaled
    stbtt_MakeGlyphBitmap(&font_data->font_info, 
                          alpha_bitmap + (padding * OVERSAMPLE) * high_res_w + (padding * OVERSAMPLE), 
                          content_width * OVERSAMPLE, content_height * OVERSAMPLE, 
                          high_res_w, 
                          high_res_scale, high_res_scale, 
                          glyph_index);

    // 2. 运行 SDF 生成管线 (ESDT + Offsets) on High-Res Bitmap
    // The spread also needs to be scaled for the calculation
    int high_res_spread = spread * OVERSAMPLE;
    
    float* dist_map = (float*)malloc(high_res_w * high_res_h * sizeof(float));
    WCN_Vec2* offsets_map = (WCN_Vec2*)malloc(high_res_w * high_res_h * sizeof(WCN_Vec2));
    
    if (!dist_map || !offsets_map) {
        free(alpha_bitmap);
        if(dist_map) free(dist_map);
        if(offsets_map) free(offsets_map);
        return false;
    }
    
    wcn_compute_esdt(alpha_bitmap, high_res_w, high_res_h, high_res_spread, dist_map, offsets_map);
    
    // 3. 打包数据到 RGBA (Downsampling)
    unsigned char* output_rgba = (unsigned char*)malloc(target_w * target_h * 4);
    if (!output_rgba) {
        free(alpha_bitmap);
        free(dist_map);
        free(offsets_map);
        return false;
    }
    
    float inv_range = 1.0f / (float)(spread * 2); // Spread in target pixels
    // Note: high_res distance is in high_res pixels. 
    // dist_in_target_pixels = dist_in_high_res / OVERSAMPLE
    
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            // Sample the center of the 4x4 block
            // center_x = x * OVERSAMPLE + OVERSAMPLE/2
            int src_x = x * OVERSAMPLE + OVERSAMPLE / 2;
            int src_y = y * OVERSAMPLE + OVERSAMPLE / 2;
            
            // Clamp to be safe
            if (src_x >= high_res_w) src_x = high_res_w - 1;
            if (src_y >= high_res_h) src_y = high_res_h - 1;
            
            int src_idx = src_y * high_res_w + src_x;
            
            // Get High-Res values
            float d_high = dist_map[src_idx];
            float off_x_high = offsets_map[src_idx].x;
            float off_y_high = offsets_map[src_idx].y;
            
            // Convert to Target Space
            float d_target = d_high / OVERSAMPLE;
            float off_x_target = off_x_high / OVERSAMPLE;
            float off_y_target = off_y_high / OVERSAMPLE;
            
            // Normalize
            float norm_d = 0.5f + (d_target * inv_range);
            norm_d = norm_d < 0.0f ? 0.0f : (norm_d > 1.0f ? 1.0f : norm_d);
            
            float norm_off_x = 0.5f + (off_x_target * inv_range);
            float norm_off_y = 0.5f + (off_y_target * inv_range);
            
            norm_off_x = norm_off_x < 0.0f ? 0.0f : (norm_off_x > 1.0f ? 1.0f : norm_off_x);
            norm_off_y = norm_off_y < 0.0f ? 0.0f : (norm_off_y > 1.0f ? 1.0f : norm_off_y);
            
            int dst_idx = y * target_w + x;
            output_rgba[dst_idx*4 + 0] = (unsigned char)(norm_d * 255.0f);   // R
            output_rgba[dst_idx*4 + 1] = (unsigned char)(norm_off_x * 255.0f); // G
            output_rgba[dst_idx*4 + 2] = (unsigned char)(norm_off_y * 255.0f); // B
            output_rgba[dst_idx*4 + 3] = 255; // A
        }
    }
    
    // 4. 清理与返回
    free(alpha_bitmap);
    free(dist_map);
    free(offsets_map);
    
    *out_bitmap = output_rgba;
    *out_width = target_w;
    *out_height = target_h;
    
    *out_offset_x = (float)(x0 - padding);
    *out_offset_y = (float)(y0 - padding);
    
    int advance, lsb;
    stbtt_GetGlyphHMetrics(&font_data->font_info, glyph_index, &advance, &lsb);
    *out_advance = advance * scale;
    
    return true;
}

// 释放 MSDF 位图
static void wcn_stb_free_glyph_sdf(unsigned char* bitmap) {
    if (bitmap) {
        // 伪 MSDF 使用 malloc 分配，用 free 释放
        free(bitmap);
    }
}

// 测量文本
static bool wcn_stb_measure_text(WCN_FontFace* face, const char* text, float font_size,
                                 float* out_width, float* out_height) {
    if (!face || !text || !out_width || !out_height) {
        return false;
    }
    
    WCN_STB_FontData* font_data = (WCN_STB_FontData*)face->user_data;
    float scale = stbtt_ScaleForPixelHeight(&font_data->font_info, font_size);
    
    float width = 0;
    const char* ptr = text;
    
    while (*ptr) {
        // 简单的 ASCII 处理
        int codepoint = *ptr++;
        
        int glyph_index = stbtt_FindGlyphIndex(&font_data->font_info, codepoint);
        if (glyph_index == 0) continue;
        
        int advance, lsb;
        stbtt_GetGlyphHMetrics(&font_data->font_info, glyph_index, &advance, &lsb);
        width += advance * scale;
    }
    
    *out_width = width;
    *out_height = font_size;
    
    return true;
}

// 释放字形
static void wcn_stb_free_glyph(WCN_Glyph* glyph) {
    if (glyph) {
        free(glyph->contours);
        free(glyph->vertices);
        free(glyph->indices);
        free(glyph->raw_vertices);
        free(glyph);
    }
}

// 释放字体
static void wcn_stb_free_font(WCN_FontFace* face) {
    if (face) {
        WCN_STB_FontData* font_data = (WCN_STB_FontData*)face->user_data;
        if (font_data) {
            free(font_data->font_buffer);
            free(font_data);
        }
        free(face);
    }
}

// 全局解码器实例
static WCN_FontDecoder wcn_stb_truetype_decoder = {
    .load_font = wcn_stb_load_font,
    .get_glyph = wcn_stb_get_glyph,
    .get_glyph_sdf = wcn_stb_get_glyph_sdf,
    .free_glyph_sdf = wcn_stb_free_glyph_sdf,
    .measure_text = wcn_stb_measure_text,
    .free_glyph = wcn_stb_free_glyph,
    .free_font = wcn_stb_free_font,
    .name = "stb_truetype"
};

// 获取解码器实例
static inline WCN_FontDecoder* wcn_get_stb_truetype_decoder(void) {
    return &wcn_stb_truetype_decoder;
}

#endif // WCN_STB_TRUETYPE_IMPL_H
