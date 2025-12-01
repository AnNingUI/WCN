#ifdef __EMSCRIPTEN__

#include "wcn_internal.h"
#include "WCN/WCN_WASM_Image.h"

#include <stdlib.h>
#include <string.h>
#include <wasm_simd128.h>

// --------------------------------------------------------------------------
// 数据结构定义
// --------------------------------------------------------------------------
typedef struct {
    uint8_t id_length;
    uint8_t color_map_type;
    uint8_t image_type;
    uint16_t color_map_origin;
    uint16_t color_map_length;
    uint8_t color_map_depth;
    uint16_t x_origin;
    uint16_t y_origin;
    uint16_t width;
    uint16_t height;
    uint8_t pixel_depth;
    uint8_t image_descriptor;
} __attribute__((packed)) WCN_TGAHeader;

// --------------------------------------------------------------------------
// 核心解码函数
// --------------------------------------------------------------------------

static bool wcn_wasm_decode_tga(
    const uint8_t* bytes,
    size_t size,
    WCN_ImageData* out_image
) {
    if (!bytes || !out_image || size < sizeof(WCN_TGAHeader)) return false;

    const WCN_TGAHeader* header = (const WCN_TGAHeader*)bytes;

    // 仅支持未压缩真彩色 (Type 2)
    if (header->color_map_type != 0 || header->image_type != 2) return false;

    const uint8_t depth = header->pixel_depth;
    if (depth != 24 && depth != 32) return false;

    const uint16_t width = header->width;
    const uint16_t height = header->height;
    if (width == 0 || height == 0) return false;

    const size_t pixel_stride = depth / 8;
    const size_t image_offset = sizeof(WCN_TGAHeader) + header->id_length;
    const size_t raw_data_size = (size_t)width * height * pixel_stride;

    if (size < image_offset + raw_data_size) return false;

    // 分配输出内存 (RGBA = 4 bytes)
    const size_t out_stride = (size_t)width * 4;
    const size_t out_size = out_stride * height;
    uint8_t* buffer = (uint8_t*)malloc(out_size);
    if (!buffer) return false;

    // 计算原点方向
    const bool origin_top = (header->image_descriptor & 0x20) != 0;
    const uint8_t* src_ptr = bytes + image_offset;
    uint8_t* dst_row_start;
    intptr_t dst_row_step;

    if (origin_top) {
        dst_row_start = buffer;
        dst_row_step = (intptr_t)out_stride;
    } else {
        dst_row_start = buffer + (size_t)(height - 1) * out_stride;
        dst_row_step = -(intptr_t)out_stride;
    }

    // ==================================================================================
    // 32-bit 解码路径 (BGRA -> RGBA)
    // ==================================================================================
    if (depth == 32) {
        // [FIX] 使用 wasm_i8x16_const 替代 wasm_v128_const
        const v128_t mask = wasm_i8x16_const(
            2, 1, 0, 3,  6, 5, 4, 7,  10, 9, 8, 11,  14, 13, 12, 15
        );

        // Unroll 4x: 每次处理 16 个像素
        uint16_t width_unrolled = width & ~15; 
        // SIMD normal: 每次处理 4 个像素
        uint16_t width_simd = width & ~3;

        for (uint16_t y = 0; y < height; ++y) {
            const uint8_t* row_src = src_ptr;
            uint8_t* row_dst = dst_row_start;
            uint16_t x = 0;

            // --- Level 1: 4x Unrolled SIMD (16 pixels) ---
            for (; x < width_unrolled; x += 16) {
                v128_t v0 = wasm_v128_load(row_src);
                v128_t v1 = wasm_v128_load(row_src + 16);
                v128_t v2 = wasm_v128_load(row_src + 32);
                v128_t v3 = wasm_v128_load(row_src + 48);

                // [FIX] 使用 wasm_i8x16_swizzle 替代 wasm_v8x16_swizzle
                v0 = wasm_i8x16_swizzle(v0, mask);
                v1 = wasm_i8x16_swizzle(v1, mask);
                v2 = wasm_i8x16_swizzle(v2, mask);
                v3 = wasm_i8x16_swizzle(v3, mask);

                wasm_v128_store(row_dst, v0);
                wasm_v128_store(row_dst + 16, v1);
                wasm_v128_store(row_dst + 32, v2);
                wasm_v128_store(row_dst + 48, v3);

                row_src += 64;
                row_dst += 64;
            }

            // --- Level 2: Single SIMD (4 pixels) ---
            for (; x < width_simd; x += 4) {
                v128_t v = wasm_v128_load(row_src);
                v = wasm_i8x16_swizzle(v, mask); // [FIX]
                wasm_v128_store(row_dst, v);
                row_src += 16;
                row_dst += 16;
            }

            // --- Level 3: Scalar Tail ---
            for (; x < width; ++x) {
                row_dst[0] = row_src[2];
                row_dst[1] = row_src[1];
                row_dst[2] = row_src[0];
                row_dst[3] = row_src[3];
                row_src += 4;
                row_dst += 4;
            }

            src_ptr += out_stride;
            dst_row_start += dst_row_step;
        }
    } 
    // ==================================================================================
    // 24-bit 解码路径 (BGR -> RGBA)
    // ==================================================================================
    else {
        const v128_t alpha_vec = wasm_i8x16_splat(0xFF);
        
        const uint8_t* safe_end_strict = bytes + size - 64;
        
        uint16_t width_unrolled = width & ~15;
        uint16_t width_simd = width & ~3;

        for (uint16_t y = 0; y < height; ++y) {
            const uint8_t* row_src = src_ptr;
            uint8_t* row_dst = dst_row_start;
            uint16_t x = 0;

            if (row_src <= safe_end_strict) {
                // --- Level 1: 4x Unrolled SIMD ---
                for (; x < width_unrolled; x += 16) {
                    v128_t v0 = wasm_v128_load(row_src);
                    v128_t v1 = wasm_v128_load(row_src + 12);
                    v128_t v2 = wasm_v128_load(row_src + 24);
                    v128_t v3 = wasm_v128_load(row_src + 36);

                    // [FIX] 使用 wasm_i8x16_shuffle 替代 wasm_v8x16_shuffle
                    v0 = wasm_i8x16_shuffle(v0, alpha_vec, 2,1,0,16, 5,4,3,16, 8,7,6,16, 11,10,9,16);
                    v1 = wasm_i8x16_shuffle(v1, alpha_vec, 2,1,0,16, 5,4,3,16, 8,7,6,16, 11,10,9,16);
                    v2 = wasm_i8x16_shuffle(v2, alpha_vec, 2,1,0,16, 5,4,3,16, 8,7,6,16, 11,10,9,16);
                    v3 = wasm_i8x16_shuffle(v3, alpha_vec, 2,1,0,16, 5,4,3,16, 8,7,6,16, 11,10,9,16);

                    wasm_v128_store(row_dst, v0);
                    wasm_v128_store(row_dst + 16, v1);
                    wasm_v128_store(row_dst + 32, v2);
                    wasm_v128_store(row_dst + 48, v3);

                    row_src += 48;
                    row_dst += 64;
                }
            }

            // --- Level 2: Single SIMD ---
            const uint8_t* loop_safe_end = bytes + size - 16;
            while (x < width_simd && row_src <= loop_safe_end) {
                v128_t v = wasm_v128_load(row_src);
                v = wasm_i8x16_shuffle(v, alpha_vec, 2,1,0,16, 5,4,3,16, 8,7,6,16, 11,10,9,16); // [FIX]
                wasm_v128_store(row_dst, v);
                row_src += 12;
                row_dst += 16;
                x += 4;
            }

            // --- Level 3: Scalar Tail ---
            for (; x < width; ++x) {
                row_dst[0] = row_src[2];
                row_dst[1] = row_src[1];
                row_dst[2] = row_src[0];
                row_dst[3] = 0xFF;
                row_src += 3;
                row_dst += 4;
            }

            src_ptr += (size_t)width * 3;
            dst_row_start += dst_row_step;
        }
    }

    out_image->data = buffer;
    out_image->width = width;
    out_image->height = height;
    out_image->format = 0;
    out_image->data_size = out_size;

    return true;
}

static bool wcn_wasm_decode_image(
    const uint8_t* image_data,
    size_t data_size,
    WCN_ImageData* out_image
) {
    return wcn_wasm_decode_tga(image_data, data_size, out_image);
}

static WCN_ImageDecoder g_wasm_image_decoder = {
    .decode = wcn_wasm_decode_image,
    .name = "WASM Optimized SIMD Decoder (TGA)"
};

WCN_ImageDecoder* wcn_get_wasm_image_decoder(void) {
    return &g_wasm_image_decoder;
}

WCN_WASM_EXPORT WCN_ImageDecoder* wcn_wasm_get_image_decoder(void) {
    return wcn_get_wasm_image_decoder();
}

#endif // __EMSCRIPTEN__