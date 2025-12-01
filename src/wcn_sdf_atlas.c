#include "wcn_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stddef.h>

#ifdef __EMSCRIPTEN__
#include "wcn_emcc_js.h"
#endif

// ============================================================================
// SDF Atlas 管理
// ============================================================================

// 创建 SDF Atlas
WCN_SDFAtlas* wcn_create_sdf_atlas(WCN_Context* ctx, uint32_t width, uint32_t height) {
    WCN_SDFAtlas* atlas = (WCN_SDFAtlas*)calloc(1, sizeof(WCN_SDFAtlas));
    if (!atlas) return NULL;

    atlas->width = width;
    atlas->height = height;
    atlas->current_x = 0;
    atlas->current_y = 0;
    atlas->row_height = 0;

    // 创建图集纹理 (RGBA8Unorm 格式 for MSDF)
#ifdef __EMSCRIPTEN__
    // Emscripten WebGPU: 必须按照结构体定义的确切顺序初始化
    // 根据 JS 代码的内存偏移量推断的顺序：
    // offset 0: nextInChain
    // offset 4: label (pointer)
    // offset 8: usage
    // offset 12: dimension
    // offset 16: size (WGPUExtent3D)
    // offset 28: format
    // offset 32: mipLevelCount
    // offset 36: sampleCount
    // offset 40: viewFormatCount
    // offset 44: viewFormats
    
    // Emscripten 的 WGPUTextureDescriptor 结构与标准不同
    // 根据 JS 代码的偏移量，正确的布局是：
    // offset 0: nextInChain
    // offset 4: label (const char*, 4 bytes)
    // offset 8: usage
    // offset 12: dimension  
    // offset 16: size (WGPUExtent3D, 12 bytes)
    // offset 28: format
    // offset 32: mipLevelCount
    // offset 36: sampleCount
    // offset 40: viewFormatCount
    // offset 44: viewFormats
    
    WGPUTextureDescriptor tex_desc;
    memset(&tex_desc, 0, sizeof(WGPUTextureDescriptor));
    
    uint8_t* desc_ptr = (uint8_t*)&tex_desc;
    
    *(void**)(desc_ptr + 0) = NULL;  // nextInChain
    *(const char**)(desc_ptr + 4) = "MSDF Atlas Texture";  // label
    *(uint32_t*)(desc_ptr + 8) = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst | WGPUTextureUsage_RenderAttachment;  // usage
    *(uint32_t*)(desc_ptr + 12) = WGPUTextureDimension_2D;  // dimension
    *(uint32_t*)(desc_ptr + 16) = width;  // size.width
    *(uint32_t*)(desc_ptr + 20) = height;  // size.height
    *(uint32_t*)(desc_ptr + 24) = 1;  // size.depthOrArrayLayers
    *(uint32_t*)(desc_ptr + 28) = WGPUTextureFormat_RGBA8Unorm;  // format
    *(uint32_t*)(desc_ptr + 32) = 1;  // mipLevelCount
    *(uint32_t*)(desc_ptr + 36) = 1;  // sampleCount
    *(uint32_t*)(desc_ptr + 40) = 0;  // viewFormatCount
    *(void**)(desc_ptr + 44) = NULL;  // viewFormats
    
    // printf("Texture descriptor values: usage=%u, dim=%u, format=%u, mip=%u, sample=%u\n",
           // *(uint32_t*)(desc_ptr + 8),
           // *(uint32_t*)(desc_ptr + 12),
           // *(uint32_t*)(desc_ptr + 28),
           // *(uint32_t*)(desc_ptr + 32),
           // *(uint32_t*)(desc_ptr + 36));
    
    // 打印结构体字段偏移量以调试内存布局
    // printf("WGPUTextureDescriptor offsets:\n");
    // printf("  nextInChain: %zu\n", offsetof(WGPUTextureDescriptor, nextInChain));
    // printf("  label: %zu\n", offsetof(WGPUTextureDescriptor, label));
    // printf("  usage: %zu\n", offsetof(WGPUTextureDescriptor, usage));
    // printf("  dimension: %zu\n", offsetof(WGPUTextureDescriptor, dimension));
    // printf("  size: %zu\n", offsetof(WGPUTextureDescriptor, size));
    // printf("  format: %zu\n", offsetof(WGPUTextureDescriptor, format));
    // printf("  mipLevelCount: %zu\n", offsetof(WGPUTextureDescriptor, mipLevelCount));
    // printf("  sampleCount: %zu\n", offsetof(WGPUTextureDescriptor, sampleCount));
    // printf("  viewFormatCount: %zu\n", offsetof(WGPUTextureDescriptor, viewFormatCount));
    // printf("  viewFormats: %zu\n", offsetof(WGPUTextureDescriptor, viewFormats));
    // printf("Creating SDF atlas texture (WASM): format=%d, size=%ux%u\n", 
           // (int)tex_desc.format, width, height);
    
    atlas->texture = wgpuDeviceCreateTexture(ctx->device, &tex_desc);
#else
    // 原生 WebGPU (Dawn/wgpu-native): 使用传统的描述符初始化
    WGPUTextureDescriptor tex_desc = {
        .nextInChain = NULL,
        .label = "MSDF Atlas Texture",
        .size = {width, height, 1},
        .mipLevelCount = 1,
        .sampleCount = 1,
        .dimension = WGPUTextureDimension_2D,
        .format = WGPUTextureFormat_RGBA8Unorm,
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst | WGPUTextureUsage_RenderAttachment
    };
    
    atlas->texture = wgpuDeviceCreateTexture(ctx->device, &tex_desc);
#endif

    if (!atlas->texture) {
        free(atlas);
        return NULL;
    }

    // 创建纹理视图
    // 在 Emscripten 中，传 NULL 会使用默认值（所有 mip levels 和 array layers）
    atlas->texture_view = wgpuTextureCreateView(atlas->texture, NULL);

    // 初始化字形缓存
    atlas->glyph_capacity = 256;
    atlas->glyphs = (WCN_AtlasGlyph*)calloc(atlas->glyph_capacity, sizeof(WCN_AtlasGlyph));
    atlas->glyph_count = 0;
    atlas->dirty = false;

    // printf("Created MSDF atlas: %ux%u\n", width, height);
    return atlas;
}

// 销毁 SDF Atlas
void wcn_destroy_sdf_atlas(WCN_SDFAtlas* atlas) {
    if (!atlas) return;

    if (atlas->texture_view) wgpuTextureViewRelease(atlas->texture_view);
    if (atlas->texture) wgpuTextureRelease(atlas->texture);
    free(atlas->glyphs);
    free(atlas);
}

// 刷新 Atlas 到 GPU
void wcn_flush_sdf_atlas(WCN_Context* ctx) {
    if (!ctx || !ctx->sdf_atlas || !ctx->sdf_atlas->dirty) {
        return;
    }

    // In the current implementation, SDF data is uploaded immediately during packing,
    // so the flush operation just needs to clear the dirty flag.
    // In a future optimization, we could batch multiple uploads here.
    ctx->sdf_atlas->dirty = false;
    // printf("Flushed SDF atlas (cleared dirty flag)\n");
}

// 打包字形到 atlas
bool wcn_atlas_pack_glyph(WCN_Context* ctx,
                          unsigned char* sdf_bitmap,
                          int width, int height,
                          float offset_x, float offset_y,
                          float advance,
                          uint32_t codepoint,
                          float font_size,
                          WCN_AtlasGlyph* out_glyph) {
    WCN_SDFAtlas* atlas = ctx->sdf_atlas;
    
    // 添加 padding 避免字形之间的干扰
    // 减少 padding 以节省空间（SDF 本身已经有 padding）
    const int padding = 1;
    int padded_width = width + padding;
    int padded_height = height + padding;
    
    // 简单的行打包算法
    if (atlas->current_x + padded_width > atlas->width) {
        // 换行
        atlas->current_x = 0;
        atlas->current_y += atlas->row_height + padding;
        atlas->row_height = 0;
        
        // printf("Atlas 换行: 新行起始 y=%u\n", atlas->current_y);
    }
    
    if (atlas->current_y + padded_height > atlas->height) {
        // Atlas 已满
        // printf("\n=== SDF Atlas 已满！===\n");
        // printf("当前位置: (%u, %u)\n", atlas->current_x, atlas->current_y);
        // printf("需要空间: %dx%d (原始: %dx%d, padding: %d)\n", 
               // padded_width, padded_height, width, height, padding);
        // printf("Atlas 大小: %ux%u\n", atlas->width, atlas->height);
        // printf("已使用字形数: %zu / %zu\n", atlas->glyph_count, atlas->glyph_capacity);
        // printf("当前字形: U+%04X, 字号: %.1f\n", codepoint, font_size);
        // printf("提示: 考虑增加 Atlas 大小或减少字号\n");
        // printf("========================\n\n");
        return false;
    }
    
    // 记录字形信息
    out_glyph->codepoint = codepoint;
    out_glyph->font_size = font_size;
    out_glyph->x = atlas->current_x;
    out_glyph->y = atlas->current_y;
    out_glyph->width = width;
    out_glyph->height = height;
    out_glyph->offset_x = offset_x;
    out_glyph->offset_y = offset_y;
    out_glyph->advance_width = advance;
    out_glyph->is_valid = true;
    
    // 计算 UV 坐标
    out_glyph->uv_min[0] = (float)out_glyph->x / atlas->width;
    out_glyph->uv_min[1] = (float)out_glyph->y / atlas->height;
    out_glyph->uv_max[0] = (float)(out_glyph->x + width) / atlas->width;
    out_glyph->uv_max[1] = (float)(out_glyph->y + height) / atlas->height;
    
    // 上传到 GPU（使用新的 WebGPU API）
    // MSDF 使用 RGBA 格式，每像素 4 字节
#ifdef __EMSCRIPTEN__
    // WASM: 使用 EM_JS 函数写入纹理数据
    wasm_queue_write_texture(
        ctx->queue,
        atlas->texture,
        out_glyph->x,
        out_glyph->y,
        width,
        height,
        sdf_bitmap,
        width * height * 4
    );
#else
    WGPUTexelCopyTextureInfo dest = {
        .texture = atlas->texture,
        .mipLevel = 0,
        .origin = {out_glyph->x, out_glyph->y, 0},
        .aspect = WGPUTextureAspect_All
    };

    WGPUTexelCopyBufferLayout layout = {
        .offset = 0,
        .bytesPerRow = width * 4,  // RGBA = 4 bytes per pixel
        .rowsPerImage = height
    };

    WGPUExtent3D size = {width, height, 1};

    wgpuQueueWriteTexture(ctx->queue, &dest, sdf_bitmap,
                         width * height * 4, &layout, &size);
#endif

    // 标记 Atlas 为 dirty（虽然 we uploaded immediately, this maintains consistency)
    atlas->dirty = true;

    // 更新打包位置（使用 padded 尺寸）
    atlas->current_x += padded_width;
    atlas->row_height = (height > atlas->row_height) ? height : atlas->row_height;
    
    // 打包成功（调试信息已关闭）
    
    return true;
}

// 在缓存中查找字形
WCN_AtlasGlyph* wcn_find_glyph_in_atlas(WCN_SDFAtlas* atlas,
                                        WCN_FontFace* face,
                                        uint32_t codepoint,
                                        float font_size) {
    if (!atlas || !face) {
        return NULL;
    }

    for (size_t i = 0; i < atlas->glyph_count; i++) {
        WCN_AtlasGlyph* glyph = &atlas->glyphs[i];
        if (glyph->font_face == face &&
            glyph->codepoint == codepoint &&
            fabsf(glyph->font_size - font_size) < 0.1f &&
            glyph->is_valid) {
            return glyph;
        }
    }
    return NULL;
}

// 清理 SDF Atlas 缓存（移除不常用的字形）
void wcn_cleanup_sdf_atlas_cache(WCN_SDFAtlas* atlas, size_t max_glyphs) {
    if (!atlas || atlas->glyph_count <= max_glyphs) {
        return;
    }

    // 简单的清理策略：保留最近使用的字形
    // 在实际应用中，可以实现更复杂的LRU缓存策略
    size_t remove_count = atlas->glyph_count - max_glyphs;

    // 这里简化处理：保留前 max_glyphs 个字形
    // 注意：这会破坏字形顺序，但在当前实现中不影响功能
    for (size_t i = max_glyphs; i < atlas->glyph_count; i++) {
        atlas->glyphs[i].is_valid = false;
    }
    atlas->glyph_count = max_glyphs;

    // 标记 Atlas 需要重新整理（在实际实现中可能需要重新打包）
    atlas->dirty = true;
}

// ???????
WCN_AtlasGlyph* wcn_get_or_create_glyph(WCN_Context* ctx,
                                        WCN_FontFace* face,
                                        uint32_t codepoint,
                                        float font_size) {
    if (!ctx || !ctx->sdf_atlas || !face) {
        return NULL;
    }

    WCN_AtlasGlyph* cached = wcn_find_glyph_in_atlas(ctx->sdf_atlas, face, codepoint, font_size);
    if (cached) {
        return cached;
    }

    if (!ctx->font_decoder || !ctx->font_decoder->get_glyph_sdf) {
        return NULL;
    }

    unsigned char* msdf_bitmap;
    int width, height;
    float offset_x, offset_y, advance;

    if (!ctx->font_decoder->get_glyph_sdf(
            face, codepoint, font_size,
            &msdf_bitmap, &width, &height,
            &offset_x, &offset_y, &advance)) {
        if (codepoint == 0x0020) {
            WCN_Glyph* glyph_info = NULL;
            if (ctx->font_decoder->get_glyph(face, codepoint, &glyph_info)) {
                float scale = font_size / face->units_per_em;
                advance = glyph_info->advance_width * scale;
                ctx->font_decoder->free_glyph(glyph_info);

                if (ctx->sdf_atlas->glyph_count >= ctx->sdf_atlas->glyph_capacity) {
                    size_t new_capacity = ctx->sdf_atlas->glyph_capacity == 0 ?
                                          256 : ctx->sdf_atlas->glyph_capacity * 2;
                    WCN_AtlasGlyph* new_glyphs = realloc(
                        ctx->sdf_atlas->glyphs,
                        new_capacity * sizeof(WCN_AtlasGlyph)
                    );
                    if (!new_glyphs) {
                        return NULL;
                    }
                    ctx->sdf_atlas->glyphs = new_glyphs;
                    ctx->sdf_atlas->glyph_capacity = new_capacity;
                }

                WCN_AtlasGlyph* empty_glyph = &ctx->sdf_atlas->glyphs[ctx->sdf_atlas->glyph_count];
                empty_glyph->codepoint = codepoint;
                empty_glyph->font_face = face;
                empty_glyph->font_size = font_size;
                empty_glyph->x = 0;
                empty_glyph->y = 0;
                empty_glyph->width = 0;
                empty_glyph->height = 0;
                empty_glyph->offset_x = 0;
                empty_glyph->offset_y = 0;
                empty_glyph->advance_width = advance;
                empty_glyph->is_valid = true;
                empty_glyph->uv_min[0] = 0;
                empty_glyph->uv_min[1] = 0;
                empty_glyph->uv_max[0] = 0;
                empty_glyph->uv_max[1] = 0;

                ctx->sdf_atlas->glyph_count++;
                return empty_glyph;
            }
        }
        return NULL;
    }

    const size_t MAX_GLYPH_CACHE_SIZE = 4096;
    if (ctx->sdf_atlas->glyph_count >= MAX_GLYPH_CACHE_SIZE) {
        wcn_cleanup_sdf_atlas_cache(ctx->sdf_atlas, MAX_GLYPH_CACHE_SIZE / 2);
    }

    if (ctx->sdf_atlas->glyph_count >= ctx->sdf_atlas->glyph_capacity) {
        size_t new_capacity = ctx->sdf_atlas->glyph_capacity == 0 ?
                             256 : ctx->sdf_atlas->glyph_capacity * 2;
        if (new_capacity > MAX_GLYPH_CACHE_SIZE) {
            new_capacity = MAX_GLYPH_CACHE_SIZE;
        }

        WCN_AtlasGlyph* new_glyphs = realloc(
            ctx->sdf_atlas->glyphs,
            new_capacity * sizeof(WCN_AtlasGlyph)
        );
        if (!new_glyphs) {
            ctx->font_decoder->free_glyph_sdf(msdf_bitmap);
            return NULL;
        }
        ctx->sdf_atlas->glyphs = new_glyphs;
        ctx->sdf_atlas->glyph_capacity = new_capacity;
    }

    WCN_AtlasGlyph* glyph = &ctx->sdf_atlas->glyphs[ctx->sdf_atlas->glyph_count];

    if (!wcn_atlas_pack_glyph(ctx, msdf_bitmap, width, height,
                              offset_x, offset_y, advance,
                              codepoint, font_size, glyph)) {
        ctx->font_decoder->free_glyph_sdf(msdf_bitmap);
        return NULL;
    }

    ctx->font_decoder->free_glyph_sdf(msdf_bitmap);
    glyph->font_face = face;
    ctx->sdf_atlas->glyph_count++;

    return glyph;
}

