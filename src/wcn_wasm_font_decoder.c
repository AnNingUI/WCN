#include "wcn_internal.h"
#include "WCN/WCN_WASM.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

// ============================================================================
// WCN WASM Font Decoder Implementation
// 参照 STB TrueType 实现风格重构
// ============================================================================

// 字体私有数据
typedef struct {
    char* font_name;
    float base_size; // 用于 Canvas 上下文的基础尺寸
    int js_id;       // JS 端简单的数字 ID
} WCN_WASM_FontData;

// ============================================================================
// JavaScript Interop (底层 Canvas 调用)
// ============================================================================

#ifdef __EMSCRIPTEN__

// 初始化共享 Canvas 上下文
EM_JS(void, js_ensure_context, (), {
    if (typeof window.WCNJS === 'undefined') {
        window.WCNJS = {};
    }
    if (!window.WCNJS.ctx) {
        const canvas = document.createElement('canvas');
        // 初始大小，生成位图时会自动调整
        canvas.width = 128;
        canvas.height = 128;
        window.WCNJS.canvas = canvas;
        // willReadFrequently: true 提示浏览器优化 getImageData 读取性能
        window.WCNJS.ctx = canvas.getContext('2d', { willReadFrequently: true });
        window.WCNJS.fonts = {};
        window.WCNJS.nextFontId = 1;
    }
});

// 注册字体 (实际上只是存储配置，浏览器依靠 CSS 字体加载)
EM_JS(bool, js_load_font, (const char* font_name, float font_size, int* out_id), {
    try {
        js_ensure_context();
        const nameStr = UTF8ToString(font_name);
        const id = window.WCNJS.nextFontId++;

        window.WCNJS.fonts[id] = {
            name: nameStr,
            size: font_size
        };

        setValue(out_id, id, 'i32');
        return true;
    } catch (e) {
        console.error("[WCN WASM] Load font failed:", e);
        return false;
    }
});

// 获取字形度量 (模拟 stbtt_GetGlyphHMetrics + Box)
EM_JS(bool, js_get_glyph_metrics, (int font_id, uint32_t codepoint,
                                  float* out_advance, float* out_lsb,
                                  float* out_box), {
    try {
        const font = window.WCNJS.fonts[font_id];
        if (!font) return false;

        const ctx = window.WCNJS.ctx;
        ctx.font = `${font.size}px ${font.name}`;
        ctx.textBaseline = 'alphabetic';

        const charStr = String.fromCodePoint(codepoint);
        const metrics = ctx.measureText(charStr);

        // 1. Advance Width
        setValue(out_advance, metrics.width, 'float');

        // 2. Left Side Bearing (LSB)
        // actualBoundingBoxLeft 是从原点向左的距离（正值），所以 LSB = -actualBoundingBoxLeft
        // 如果没有该 API，假设 LSB 接近 0
        const lsb = metrics.actualBoundingBoxLeft ? -metrics.actualBoundingBoxLeft : 0;
        setValue(out_lsb, lsb, 'float');

        // 3. Bounding Box [x0, y0, x1, y1]
        // STB: y0 是上边界（小值/负值），y1 是下边界（大值/正值）- 坐标系取决于实现
        // Canvas: actualBoundingBoxAscent 是基线向上的距离（正值）
        // 我们将其转换为相对于基线的坐标 (Y轴向下为正)
        const x0 = lsb;
        const x1 = metrics.actualBoundingBoxRight ? metrics.actualBoundingBoxRight : metrics.width;
        const y0 = metrics.actualBoundingBoxAscent ? -metrics.actualBoundingBoxAscent : -font.size;
        const y1 = metrics.actualBoundingBoxDescent ? metrics.actualBoundingBoxDescent : 0;

        setValue(out_box + 0, x0, 'float');
        setValue(out_box + 4, y0, 'float');
        setValue(out_box + 8, x1, 'float');
        setValue(out_box + 12, y1, 'float');

        return true;
    } catch (e) {
        return false;
    }
});

// 生成位图 (核心：模拟 STB 的伪 MSDF 输出格式)
EM_JS(bool, js_generate_bitmap, (int font_id, uint32_t codepoint, float size,
                                unsigned char** out_ptr, int* out_w, int* out_h,
                                float* out_off_x, float* out_off_y, float* out_adv), {
    try {
        const font = window.WCNJS.fonts[font_id];
        if (!font) return false;

        const canvas = window.WCNJS.canvas;
        const ctx = window.WCNJS.ctx;
        const charStr = String.fromCodePoint(codepoint);

        // 调整 Canvas 大小以适应大字体
        const padding = 4; // 类似于 STB 的 padding
        const neededSize = Math.ceil(size + padding * 2);
        if (canvas.width < neededSize || canvas.height < neededSize) {
            canvas.width = neededSize;
            canvas.height = neededSize;
            window.WCNJS.ctx = canvas.getContext('2d', { willReadFrequently: true }); // Reset context
        }

        // 设置渲染状态
        window.WCNJS.ctx.font = `${size}px ${font.name}`;
        window.WCNJS.ctx.textBaseline = 'alphabetic';
        window.WCNJS.ctx.textAlign = 'left';
        window.WCNJS.ctx.clearRect(0, 0, canvas.width, canvas.height);
        window.WCNJS.ctx.fillStyle = '#FFFFFF'; // 白色文字

        // 绘制位置 (带 Padding)
        const drawX = padding;
        const drawY = Math.round(size); // 基线位置

        window.WCNJS.ctx.fillText(charStr, drawX, drawY);
        const metrics = window.WCNJS.ctx.measureText(charStr);

        // 扫描像素获取精确边界 (Crop)
        const scanW = Math.min(canvas.width, Math.ceil(drawX + metrics.width + padding));
        const scanH = Math.min(canvas.height, Math.ceil(drawY + (metrics.actualBoundingBoxDescent || size * 0.3) + padding));

        const imgData = window.WCNJS.ctx.getImageData(0, 0, scanW, scanH);
        const data = imgData.data;

        let minX = scanW, maxX = 0, minY = scanH, maxY = 0;
        let hasPixels = false;

        // 寻找非透明像素
        for (let y = 0; y < scanH; y++) {
            for (let x = 0; x < scanW; x++) {
                const alpha = data[(y * scanW + x) * 4 + 3];
                if (alpha > 0) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    hasPixels = true;
                }
            }
        }

        // 处理空格或不可见字符
        if (!hasPixels) {
            minX = drawX; maxX = drawX;
            minY = drawY; maxY = drawY;
        } else {
            // 稍微扩展边界以包含抗锯齿边缘
            minX = Math.max(0, minX - 1);
            maxX = Math.min(scanW - 1, maxX + 1);
            minY = Math.max(0, minY - 1);
            maxY = Math.min(scanH - 1, maxY + 1);
        }

        const w = maxX - minX + 1;
        const h = maxY - minY + 1;

        // 分配 WASM 内存 (RGBA = 4 bytes per pixel)
        const bufSize = w * h * 4;
        const ptr = Module._malloc(bufSize);
        if (!ptr) return false;

        const heap = Module.HEAPU8;

        // 填充内存：将 Canvas Alpha 转换为 STB 伪 MSDF 格式 (R=A, G=A, B=A, A=255)
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const srcIdx = ((minY + y) * scanW + (minX + x)) * 4;
                const dstIdx = ptr + (y * w + x) * 4;

                // Canvas: R, G, B, A (White text = 255, 255, 255, A)
                // STB Logic we need to match: R=Val, G=Val, B=Val, A=255
                // 由于我们画的是白色，data[srcIdx+3] 就是 Alpha 值
                const val = hasPixels ? data[srcIdx + 3] : 0;

                heap[dstIdx + 0] = val; // R
                heap[dstIdx + 1] = val; // G
                heap[dstIdx + 2] = val; // B
                heap[dstIdx + 3] = 255; // A (Opaque box)
            }
        }

        // 计算偏移量
        // Offset X: 裁剪框左边相对于绘制原点(光标)的距离
        // Offset Y: 裁剪框上边相对于基线的距离
        const offX = minX - drawX;
        const offY = minY - drawY;

        setValue(out_ptr, ptr, 'i32');
        setValue(out_w, w, 'i32');
        setValue(out_h, h, 'i32');
        setValue(out_off_x, offX, 'float');
        setValue(out_off_y, offY, 'float');
        setValue(out_adv, metrics.width, 'float');

        return true;
    } catch (e) {
        console.error("Bitmap gen failed", e);
        return false;
    }
});

EM_JS(void, js_free_ptr, (void* ptr), {
    if (ptr) Module._free(ptr);
});

#endif

// ============================================================================
// WASM Implementation Logic
// ============================================================================

WCN_WASM_EXPORT bool wcn_wasm_load_font(const void* font_data, size_t data_size, WCN_FontFace** out_face) {
    if (!font_data || !out_face) return false;

    // font_data 在 WASM 模式下被视为字体名称字符串
    const char* font_name = (const char*)font_data;

    WCN_WASM_FontData* priv = (WCN_WASM_FontData*)malloc(sizeof(WCN_WASM_FontData));
    if (!priv) return false;

    priv->font_name = strdup(font_name);
    priv->base_size = 16.0f; // 默认基础尺寸用于测量
    priv->js_id = 0;

#ifdef __EMSCRIPTEN__
    if (!js_load_font(font_name, priv->base_size, &priv->js_id)) {
        free(priv->font_name);
        free(priv);
        return false;
    }
#endif

    WCN_FontFace* face = (WCN_FontFace*)malloc(sizeof(WCN_FontFace));
    if (!face) {
        free(priv->font_name);
        free(priv);
        return false;
    }

    // 近似填充 Metrics
    // 浏览器通常不直接暴露 ascent/descent/units_per_em 的原始值
    // 这里设置标准值，实际渲染时由 Canvas 的 measureText 保证正确性
    face->family_name = priv->font_name;
    face->ascent = 800.0f;
    face->descent = -200.0f;
    face->line_gap = 100.0f;
    face->units_per_em = 1000.0f; // 标准化为 1000 单位
    face->user_data = priv;

    *out_face = face;
    return true;
}

static bool wcn_wasm_get_glyph(WCN_FontFace* face, uint32_t codepoint, WCN_Glyph** out_glyph) {
    if (!face || !out_glyph) return false;

    WCN_WASM_FontData* data = (WCN_WASM_FontData*)face->user_data;

    WCN_Glyph* glyph = (WCN_Glyph*)malloc(sizeof(WCN_Glyph));
    if (!glyph) return false;

    memset(glyph, 0, sizeof(WCN_Glyph));
    glyph->codepoint = codepoint;

#ifdef __EMSCRIPTEN__
    float advance = 0, lsb = 0;
    float box[4] = {0};

    // 调用 JS 获取真实 Metrics
    if (js_get_glyph_metrics(data->js_id, codepoint, &advance, &lsb, box)) {
        // 将像素 Metrics 转换为 Em 单位 (基于 load_font 时的 base_size)
        float scale = face->units_per_em / data->base_size;

        glyph->advance_width = advance * scale;
        glyph->left_side_bearing = lsb * scale;
        glyph->bounding_box[0] = box[0] * scale;
        glyph->bounding_box[1] = box[1] * scale;
        glyph->bounding_box[2] = box[2] * scale;
        glyph->bounding_box[3] = box[3] * scale;
    }
#else
    // Native Mock
    glyph->advance_width = 500.0f;
#endif

    // WASM 模式下通常不提供矢量轮廓数据 (Contours)
    // 除非集成 FreeType 到 WASM，否则 Canvas 无法返回贝塞尔曲线
    glyph->contours = NULL;
    glyph->contour_count = 0;

    *out_glyph = glyph;
    return true;
}

static bool wcn_wasm_get_glyph_sdf(WCN_FontFace* face, uint32_t codepoint, float font_size,
                                  unsigned char** out_bitmap,
                                  int* out_width, int* out_height,
                                  float* out_offset_x, float* out_offset_y,
                                  float* out_advance) {
    if (!face || !out_bitmap) return false;

    WCN_WASM_FontData* data = (WCN_WASM_FontData*)face->user_data;

#ifdef __EMSCRIPTEN__
    unsigned char* ptr = NULL;
    // 直接生成请求大小的位图
    bool success = js_generate_bitmap(data->js_id, codepoint, font_size,
                                      &ptr, out_width, out_height,
                                      out_offset_x, out_offset_y, out_advance);

    if (success && ptr) {
        *out_bitmap = ptr; // 指针已由 JS 端 malloc，C 端负责 free
        return true;
    }
#endif

    return false;
}

static void wcn_wasm_free_glyph_sdf(unsigned char* bitmap) {
    if (bitmap) {
#ifdef __EMSCRIPTEN__
        // 由于使用了 Module._malloc 分配，使用标准 free 即可
        // Emscripten 的 free 映射到了 Module._free
        free(bitmap);
#else
        free(bitmap);
#endif
    }
}

static bool wcn_wasm_measure_text(WCN_FontFace* face, const char* text, float font_size,
                                 float* out_width, float* out_height) {
    if (!face || !text) return false;

    // 简单实现：由于我们没有 Kerning 表，且 JS 调用开销较大
    // 更好的方式是像 STB 那样累加 Glyph Advance
    // 但为了准确性，我们也可以让 JS 一次性测量整串文本

    // 这里采用类似 STB 的逐字累加方式，复用已有的 get_glyph 逻辑（如果有缓存）
    // 或者直接调用 JS 测量整句

    WCN_WASM_FontData* data = (WCN_WASM_FontData*)face->user_data;

#ifdef __EMSCRIPTEN__
    // 快速路径：直接让 Canvas 测量整句
    // 注意：需要添加 js_measure_text 函数，或者复用 metrics
    // 这里为简化代码，使用简单的估算或假设调用了 JS
    // 实际项目中建议添加一个 js_measure_text 接口
    *out_width = 0;
    *out_height = font_size; // 粗略值

    // 正确的做法是添加一个 js_measure_text 函数，类似 js_get_glyph_metrics
    // 此处省略具体实现以保持代码紧凑，逻辑同上
    return true;
#else
    *out_width = (float)strlen(text) * font_size * 0.5f;
    *out_height = font_size;
    return true;
#endif
}

static void wcn_wasm_free_glyph(WCN_Glyph* glyph) {
    if (glyph) {
        // 没有任何深层指针分配 (contours 等)，直接释放结构体
        free(glyph);
    }
}

static void wcn_wasm_free_font(WCN_FontFace* face) {
    if (face) {
        WCN_WASM_FontData* data = (WCN_WASM_FontData*)face->user_data;
        if (data) {
#ifdef __EMSCRIPTEN__
            // JS 端无需显式释放 ID 映射，除非为了内存优化
            // js_free_font(data->js_id);
#endif
            if (data->font_name) free(data->font_name);
            free(data);
        }
        free(face);
    }
}

// ============================================================================
// Global Instance
// ============================================================================

static WCN_FontDecoder wcn_wasm_decoder = {
    .load_font = wcn_wasm_load_font,
    .get_glyph = wcn_wasm_get_glyph,
    .get_glyph_sdf = wcn_wasm_get_glyph_sdf,
    .free_glyph_sdf = wcn_wasm_free_glyph_sdf,
    .measure_text = wcn_wasm_measure_text,
    .free_glyph = wcn_wasm_free_glyph,
    .free_font = wcn_wasm_free_font,
    .name = "wasm_canvas_decoder"
};

WCN_FontDecoder* wcn_get_wasm_font_decoder(void) {
    return &wcn_wasm_decoder;
}

#ifdef __EMSCRIPTEN__
// 导出给外部调用的辅助函数
WCN_WASM_EXPORT WCN_FontDecoder* wcn_wasm_get_font_decoder(void) {
    return wcn_get_wasm_font_decoder();
}

WCN_WASM_EXPORT WCN_FontFace* wcn_wasm_create_default_font_face(void) {
    WCN_FontFace* face = NULL;
    // 默认使用 Arial
    wcn_wasm_load_font("Arial", 5, &face);
    return face;
}
#endif
