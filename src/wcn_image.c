#include "wcn_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __EMSCRIPTEN__
#include "wcn_emcc_js.h"
#endif

#define WCN_IMAGE_ATLAS_DEFAULT_SIZE 4096u
#define WCN_IMAGE_ATLAS_PADDING 2u

static bool wcn_image_atlas_ensure_capacity(WCN_ImageAtlas* atlas) {
    if (!atlas) {
        return false;
    }
    if (atlas->entry_count < atlas->entry_capacity) {
        return true;
    }

    // 限制图像缓存的最大容量
    const size_t MAX_IMAGE_CACHE_ENTRIES = 1024;

    if (atlas->entry_capacity >= MAX_IMAGE_CACHE_ENTRIES) {
        // 缓存已满，无法添加更多条目
        return false;
    }

    size_t new_capacity = atlas->entry_capacity ? atlas->entry_capacity * 2 : 64;

    // 确保不超过最大限制
    if (new_capacity > MAX_IMAGE_CACHE_ENTRIES) {
        new_capacity = MAX_IMAGE_CACHE_ENTRIES;
    }

    WCN_ImageCacheEntry* entries = (WCN_ImageCacheEntry*)realloc(
        atlas->entries,
        new_capacity * sizeof(WCN_ImageCacheEntry)
    );
    if (!entries) {
        return false;
    }

    atlas->entries = entries;
    memset(atlas->entries + atlas->entry_capacity, 0,
           (new_capacity - atlas->entry_capacity) * sizeof(WCN_ImageCacheEntry));
    atlas->entry_capacity = new_capacity;
    return true;
}

static bool wcn_write_image_to_atlas(WCN_Context* ctx,
                                     WCN_ImageAtlas* atlas,
                                     WCN_ImageData* image,
                                     uint32_t dst_x,
                                     uint32_t dst_y) {
    if (!ctx || !atlas || !image || !image->data) {
        return false;
    }

    const size_t required_size = (size_t)image->width * (size_t)image->height * 4;
    if (image->data_size < required_size) {
        printf("WCN: Image data too small for %ux%u bitmap (got %zu, need %zu)\n",
               image->width, image->height, image->data_size, required_size);
        return false;
    }

#ifdef __EMSCRIPTEN__
    wasm_queue_write_texture(
        ctx->queue,
        atlas->texture,
        dst_x,
        dst_y,
        image->width,
        image->height,
        image->data,
        required_size
    );
#else
    WGPUTexelCopyTextureInfo dest = {
        .texture = atlas->texture,
        .mipLevel = 0,
        .origin = {dst_x, dst_y, 0},
        .aspect = WGPUTextureAspect_All
    };

    WGPUTexelCopyBufferLayout layout = {
        .offset = 0,
        .bytesPerRow = image->width * 4,
        .rowsPerImage = image->height
    };

    WGPUExtent3D size = {image->width, image->height, 1};

    wgpuQueueWriteTexture(ctx->queue, &dest, image->data, required_size, &layout, &size);
#endif

    return true;
}

static bool wcn_pack_image_into_atlas(WCN_Context* ctx,
                                      WCN_ImageAtlas* atlas,
                                      WCN_ImageData* image,
                                      WCN_ImageCacheEntry* entry) {
    if (!atlas || !image || !entry) {
        return false;
    }

    if (image->width == 0 || image->height == 0) {
        return false;
    }

    const uint32_t padded_width = image->width + WCN_IMAGE_ATLAS_PADDING;
    const uint32_t padded_height = image->height + WCN_IMAGE_ATLAS_PADDING;

    if (padded_width > atlas->width || padded_height > atlas->height) {
        printf("WCN: Image %ux%u is too large for the atlas (%ux%u)\n",
               image->width, image->height, atlas->width, atlas->height);
        return false;
    }

    if (atlas->current_x + padded_width > atlas->width) {
        atlas->current_x = 0;
        atlas->current_y += atlas->row_height;
        atlas->row_height = 0;
    }

    if (atlas->current_y + padded_height > atlas->height) {
        printf("WCN: Image atlas is full (requested %ux%u)\n",
               image->width, image->height);
        return false;
    }

    entry->source = image;
    entry->x = atlas->current_x;
    entry->y = atlas->current_y;
    entry->width = image->width;
    entry->height = image->height;
    entry->uv_min[0] = (float)entry->x / (float)atlas->width;
    entry->uv_min[1] = (float)entry->y / (float)atlas->height;
    entry->uv_max[0] = (float)(entry->x + entry->width) / (float)atlas->width;
    entry->uv_max[1] = (float)(entry->y + entry->height) / (float)atlas->height;
    entry->is_valid = true;

    if (!wcn_write_image_to_atlas(ctx, atlas, image, entry->x, entry->y)) {
        entry->is_valid = false;
        return false;
    }

    atlas->current_x += padded_width;
    if (padded_height > atlas->row_height) {
        atlas->row_height = padded_height;
    }

    return true;
}

static WCN_ImageCacheEntry* wcn_find_image_entry(WCN_ImageAtlas* atlas, WCN_ImageData* image) {
    if (!atlas || !image) {
        return NULL;
    }
    for (size_t i = 0; i < atlas->entry_count; i++) {
        WCN_ImageCacheEntry* entry = &atlas->entries[i];
        if (entry->is_valid && entry->source == image) {
            return entry;
        }
    }
    return NULL;
}

WCN_ImageAtlas* wcn_create_image_atlas(WCN_Context* ctx, uint32_t width, uint32_t height) {
    (void)ctx;
    WCN_ImageAtlas* atlas = (WCN_ImageAtlas*)calloc(1, sizeof(WCN_ImageAtlas));
    if (!atlas) {
        return NULL;
    }

    atlas->width = width;
    atlas->height = height;

#ifdef __EMSCRIPTEN__
    WGPUTextureDescriptor tex_desc;
    memset(&tex_desc, 0, sizeof(WGPUTextureDescriptor));

    uint8_t* desc_ptr = (uint8_t*)&tex_desc;
    *(void**)(desc_ptr + 0) = NULL;
    *(const char**)(desc_ptr + 4) = "WCN Image Atlas";
    *(uint32_t*)(desc_ptr + 8) = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    *(uint32_t*)(desc_ptr + 12) = WGPUTextureDimension_2D;
    *(uint32_t*)(desc_ptr + 16) = width;
    *(uint32_t*)(desc_ptr + 20) = height;
    *(uint32_t*)(desc_ptr + 24) = 1;
    *(uint32_t*)(desc_ptr + 28) = WGPUTextureFormat_RGBA8Unorm;
    *(uint32_t*)(desc_ptr + 32) = 1;
    *(uint32_t*)(desc_ptr + 36) = 1;
    *(uint32_t*)(desc_ptr + 40) = 0;
    *(void**)(desc_ptr + 44) = NULL;

    atlas->texture = wgpuDeviceCreateTexture(ctx->device, &tex_desc);
#else
    WGPUTextureDescriptor tex_desc = {
        .nextInChain = NULL,
        .label = "WCN Image Atlas",
        .size = {width, height, 1},
        .mipLevelCount = 1,
        .sampleCount = 1,
        .dimension = WGPUTextureDimension_2D,
        .format = WGPUTextureFormat_RGBA8Unorm,
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
    };

    atlas->texture = wgpuDeviceCreateTexture(ctx->device, &tex_desc);
#endif

    if (!atlas->texture) {
        free(atlas);
        return NULL;
    }

    atlas->texture_view = wgpuTextureCreateView(atlas->texture, NULL);
    if (!atlas->texture_view) {
        wgpuTextureRelease(atlas->texture);
        free(atlas);
        return NULL;
    }

    atlas->entry_capacity = 64;
    atlas->entries = (WCN_ImageCacheEntry*)calloc(atlas->entry_capacity, sizeof(WCN_ImageCacheEntry));
    if (!atlas->entries) {
        wgpuTextureViewRelease(atlas->texture_view);
        wgpuTextureRelease(atlas->texture);
        free(atlas);
        return NULL;
    }

    return atlas;
}

void wcn_destroy_image_atlas(WCN_ImageAtlas* atlas) {
    if (!atlas) {
        return;
    }

    if (atlas->texture_view) {
        wgpuTextureViewRelease(atlas->texture_view);
    }
    if (atlas->texture) {
        wgpuTextureRelease(atlas->texture);
    }
    free(atlas->entries);
    free(atlas);
}

WCN_ImageCacheEntry* wcn_image_atlas_get_entry(WCN_Context* ctx, WCN_ImageData* image) {
    if (!ctx || !ctx->image_atlas || !image) {
        return NULL;
    }

    WCN_ImageAtlas* atlas = ctx->image_atlas;
    WCN_ImageCacheEntry* entry = wcn_find_image_entry(atlas, image);
    if (entry && entry->is_valid) {
        return entry;
    }

    if (!wcn_image_atlas_ensure_capacity(atlas)) {
        return NULL;
    }

    entry = &atlas->entries[atlas->entry_count++];
    memset(entry, 0, sizeof(WCN_ImageCacheEntry));

    if (!wcn_pack_image_into_atlas(ctx, atlas, image, entry)) {
        atlas->entry_count--;
        return NULL;
    }

    return entry;
}

bool wcn_init_image_manager(WCN_Context* ctx) {
    if (!ctx) {
        return false;
    }

    bool created_atlas = false;
    if (!ctx->image_atlas) {
        ctx->image_atlas = wcn_create_image_atlas(
            ctx,
            WCN_IMAGE_ATLAS_DEFAULT_SIZE,
            WCN_IMAGE_ATLAS_DEFAULT_SIZE
        );
        created_atlas = ctx->image_atlas != NULL;
        if (!ctx->image_atlas) {
            printf("WCN: Failed to create image atlas\n");
            return false;
        }
    }

    if (!ctx->image_sampler) {
#ifdef __EMSCRIPTEN__
        ctx->image_sampler = wasm_create_sampler(ctx->device, "WCN Image Sampler");
#else
        const char* label_str = "WCN Image Sampler";
        WGPUSamplerDescriptor sampler_desc = {
            .nextInChain = NULL,
            .label = {
                .data = label_str,
                .length = strlen(label_str)
            },
            .addressModeU = WGPUAddressMode_ClampToEdge,
            .addressModeV = WGPUAddressMode_ClampToEdge,
            .addressModeW = WGPUAddressMode_ClampToEdge,
            .magFilter = WGPUFilterMode_Linear,
            .minFilter = WGPUFilterMode_Linear,
            .mipmapFilter = WGPUMipmapFilterMode_Linear,
            .lodMinClamp = 0.0f,
            .lodMaxClamp = 32.0f,
            .compare = WGPUCompareFunction_Undefined,
            .maxAnisotropy = 1
        };
        ctx->image_sampler = wgpuDeviceCreateSampler(ctx->device, &sampler_desc);
#endif
        if (!ctx->image_sampler) {
            if (created_atlas) {
                wcn_destroy_image_atlas(ctx->image_atlas);
                ctx->image_atlas = NULL;
            }
            printf("WCN: Failed to create image sampler\n");
            return false;
        }
    }

    return true;
}

void wcn_shutdown_image_manager(WCN_Context* ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->image_atlas) {
        wcn_destroy_image_atlas(ctx->image_atlas);
        ctx->image_atlas = NULL;
    }

    if (ctx->image_sampler) {
        wgpuSamplerRelease(ctx->image_sampler);
        ctx->image_sampler = NULL;
    }
}

static uint32_t wcn_image_compute_tint(const WCN_GPUState* state) {
    float alpha = state ? state->global_alpha : 1.0f;
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    uint32_t a = (uint32_t)(alpha * 255.0f + 0.5f);
    if (a > 255u) a = 255u;
    return (a << 24) | 0x00FFFFFFu;
}

static bool wcn_ensure_image_resources(WCN_Context* ctx) {
    return wcn_init_image_manager(ctx);
}

static void wcn_submit_image_draw(WCN_Context* ctx,
                                  WCN_ImageCacheEntry* entry,
                                  float sx, float sy, float sw, float sh,
                                  float dx, float dy, float dw, float dh) {
    if (!ctx || !entry || !ctx->renderer) {
        return;
    }

    if (sw <= 0.0f || sh <= 0.0f || dw == 0.0f || dh == 0.0f) {
        return;
    }

    const float image_width = (float)entry->width;
    const float image_height = (float)entry->height;
    if (image_width <= 0.0f || image_height <= 0.0f) {
        return;
    }

    float src_x0 = sx;
    float src_y0 = sy;
    float src_x1 = sx + sw;
    float src_y1 = sy + sh;

    if (src_x1 <= 0.0f || src_y1 <= 0.0f || src_x0 >= image_width || src_y0 >= image_height) {
        return; // 完全在图像之外
    }

    if (src_x0 < 0.0f) src_x0 = 0.0f;
    if (src_y0 < 0.0f) src_y0 = 0.0f;
    if (src_x1 > image_width) src_x1 = image_width;
    if (src_y1 > image_height) src_y1 = image_height;

    const float full_u_size = entry->uv_max[0] - entry->uv_min[0];
    const float full_v_size = entry->uv_max[1] - entry->uv_min[1];
    const float inv_w = 1.0f / image_width;
    const float inv_h = 1.0f / image_height;

    const float src_u0 = src_x0 * inv_w;
    const float src_v0 = src_y0 * inv_h;
    const float src_u1 = src_x1 * inv_w;
    const float src_v1 = src_y1 * inv_h;

    float uv_min[2];
    float uv_size[2];
    uv_min[0] = entry->uv_min[0] + full_u_size * src_u0;
    uv_min[1] = entry->uv_min[1] + full_v_size * src_v0;
    const float uv_max_x = entry->uv_min[0] + full_u_size * src_u1;
    const float uv_max_y = entry->uv_min[1] + full_v_size * src_v1;
    uv_size[0] = uv_max_x - uv_min[0];
    uv_size[1] = uv_max_y - uv_min[1];

    float dest_x = dx;
    float dest_y = dy;
    float dest_w = dw;
    float dest_h = dh;

    if (dest_w < 0.0f) {
        dest_x += dest_w;
        dest_w = -dest_w;
        uv_min[0] += uv_size[0];
        uv_size[0] = -uv_size[0];
    }

    if (dest_h < 0.0f) {
        dest_y += dest_h;
        dest_h = -dest_h;
        uv_min[1] += uv_size[1];
        uv_size[1] = -uv_size[1];
    }

    if (dest_w <= 0.0f || dest_h <= 0.0f) {
        return;
    }

    WCN_GPUState* state = &ctx->state_stack.states[ctx->state_stack.current_state];
    const uint32_t tint = wcn_image_compute_tint(state);

    // 计算应用变换后的图像位置
    float transformed_x = dest_x * state->transform_matrix[0] + dest_y * state->transform_matrix[4] + state->transform_matrix[12];
    float transformed_y = dest_x * state->transform_matrix[1] + dest_y * state->transform_matrix[5] + state->transform_matrix[13];

    // 提取 2x2 变换矩阵（从 4x4 矩阵的左上角）- 用于实例变换
    const float instance_transform[4] = {
        state->transform_matrix[0], state->transform_matrix[1],
        state->transform_matrix[4], state->transform_matrix[5]
    };

    wcn_renderer_add_image(
        ctx->renderer,
        transformed_x,
        transformed_y,
        dest_w,
        dest_h,
        tint,
        instance_transform,
        uv_min,
        uv_size
    );
}

void wcn_draw_image(WCN_Context* ctx, WCN_ImageData* image, float dx, float dy) {
    if (!image) {
        return;
    }
    float sw = (float)image->width;
    float sh = (float)image->height;
    wcn_draw_image_source(ctx, image, 0.0f, 0.0f, sw, sh, dx, dy, sw, sh);
}

void wcn_draw_image_scaled(WCN_Context* ctx,
                           WCN_ImageData* image,
                           float dx, float dy,
                           float dw, float dh) {
    if (!image) {
        return;
    }
    float sw = (float)image->width;
    float sh = (float)image->height;
    wcn_draw_image_source(ctx, image, 0.0f, 0.0f, sw, sh, dx, dy, dw, dh);
}

void wcn_draw_image_source(WCN_Context* ctx,
                           WCN_ImageData* image,
                           float sx, float sy, float sw, float sh,
                           float dx, float dy, float dw, float dh) {
    if (!ctx || !ctx->in_frame || !ctx->renderer || !image || !image->data) {
        return;
    }

    if (sw <= 0.0f || sh <= 0.0f || image->width == 0 || image->height == 0) {
        return;
    }

    if (!wcn_ensure_image_resources(ctx)) {
        return;
    }

    WCN_ImageCacheEntry* entry = wcn_image_atlas_get_entry(ctx, image);
    if (!entry) {
        printf("WCN: Failed to upload image to atlas\n");
        return;
    }

    wcn_submit_image_draw(ctx, entry, sx, sy, sw, sh, dx, dy, dw, dh);
}

WCN_ImageData* wcn_get_image_data(WCN_Context* ctx,
                                  float x, float y,
                                  float width, float height) {
    (void)ctx;
    (void)x;
    (void)y;
    (void)width;
    (void)height;
    printf("WCN: getImageData is not implemented yet.\n");
    return NULL;
}

void wcn_put_image_data(WCN_Context* ctx, WCN_ImageData* image_data, float x, float y) {
    wcn_draw_image(ctx, image_data, x, y);
}

WCN_ImageData* wcn_decode_image(WCN_Context* ctx, const uint8_t* image_bytes, size_t data_size) {
    if (!ctx || !image_bytes || data_size == 0) {
        return NULL;
    }

    if (!ctx->image_decoder || !ctx->image_decoder->decode) {
        printf("WCN: No image decoder registered\n");
        return NULL;
    }

    WCN_ImageData* image = (WCN_ImageData*)calloc(1, sizeof(WCN_ImageData));
    if (!image) {
        return NULL;
    }

    if (!ctx->image_decoder->decode(image_bytes, data_size, image)) {
        free(image);
        return NULL;
    }

    return image;
}

void wcn_destroy_image_data(WCN_ImageData* image_data) {
    if (!image_data) {
        return;
    }

    if (image_data->data) {
        free(image_data->data);
    }
    free(image_data);
}
