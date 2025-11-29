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

// 获取字形 MSDF 位图（核心功能）
// 使用 stb_truetype 的 SDF 生成伪 MSDF（将单通道复制到 RGB 通道）
static bool wcn_stb_get_glyph_sdf(WCN_FontFace* face, uint32_t codepoint, float font_size,
                                  unsigned char** out_bitmap,
                                  int* out_width, int* out_height,
                                  float* out_offset_x, float* out_offset_y,
                                  float* out_advance) {
    if (!face || !out_bitmap || !out_width || !out_height) {
        return false;
    }
    
    WCN_STB_FontData* font_data = (WCN_STB_FontData*)face->user_data;
    
    // 计算缩放比例
    float scale = stbtt_ScaleForPixelHeight(&font_data->font_info, font_size);
    
    // 获取字形索引
    int glyph_index = stbtt_FindGlyphIndex(&font_data->font_info, codepoint);
    if (glyph_index == 0) {
        // 静默处理不存在的字形（避免刷屏）
        return false;
    }
    
    // 生成单通道 SDF
    // 关键参数调整：
    // - padding: 4 像素，平衡质量和空间（减少以节省 Atlas 空间）
    // - onedge_value: 128 (0.5 * 255)，边缘值
    // - pixel_dist_scale: 32.0，控制距离场的梯度（值越小越清晰）
    int xoff, yoff;
    unsigned char* sdf = stbtt_GetGlyphSDF(
        &font_data->font_info,
        scale,
        glyph_index,
        4,      // padding - 4 像素平衡质量和空间
        128,    // onedge_value (0.5 in 0-255 range)
        32.0f,  // pixel_dist_scale - 降低到 32 以获得更清晰的边缘
        out_width, out_height,
        &xoff, &yoff
    );
    
    if (!sdf) {
        // 静默处理 SDF 生成失败（通常是空格等不可见字符）
        return false;
    }
    
    // SDF 生成成功（调试信息已关闭）
    
    // 转换为伪 MSDF 格式（RGBA）
    // 将单通道 SDF 复制到 RGB 通道，A 通道设为 255
    int pixel_count = (*out_width) * (*out_height);
    unsigned char* msdf = (unsigned char*)malloc(pixel_count * 4);
    if (!msdf) {
        stbtt_FreeSDF(sdf, NULL);
        return false;
    }
    
    for (int i = 0; i < pixel_count; i++) {
        unsigned char value = sdf[i];
        msdf[i * 4 + 0] = value;  // R
        msdf[i * 4 + 1] = value;  // G
        msdf[i * 4 + 2] = value;  // B
        msdf[i * 4 + 3] = 255;    // A (不透明)
    }
    
    // 释放原始 SDF
    stbtt_FreeSDF(sdf, NULL);
    
    *out_bitmap = msdf;
    *out_offset_x = (float)xoff;
    // stb_truetype 的 Y 坐标系：yoff 是字形顶部相对于基线的偏移
    // 正值表示在基线上方，负值表示在基线下方
    // WebGPU 坐标系：Y 轴向下为正
    // 因此不需要取反，直接使用
    *out_offset_y = (float)yoff;
    
    // 获取前进宽度
    int advance, lsb;
    stbtt_GetGlyphHMetrics(&font_data->font_info, glyph_index, &advance, &lsb);
    *out_advance = advance * scale;
    
    // 仅在调试时打印
    // printf("stb_truetype: 生成 MSDF U+%04X 尺寸 %dx%d 偏移 (%.1f, %.1f) 前进 %.1f\n",
    //        codepoint, *out_width, *out_height, *out_offset_x, *out_offset_y, *out_advance);
    
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
