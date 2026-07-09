#include "fullstack_core_private.h"

#include <stdlib.h>
#include <string.h>

bool fs_core_upload_image_rgba8(FS_Core* core, const uint8_t* pixels, uint32_t width, uint32_t height, FS_ImageHandle* out_handle);

bool fs_ensure_canvas_shadow(FS_Core* core) {
    if (!core) {
        return false;
    }
    const size_t needed = (size_t)core->width * (size_t)core->height * 4u;
    if (needed == 0u) {
        free(core->canvas_shadow_rgba);
        core->canvas_shadow_rgba = NULL;
        core->canvas_shadow_size = 0u;
        return true;
    }
    if (needed <= core->canvas_shadow_size && core->canvas_shadow_rgba) {
        return true;
    }
    uint8_t* grown = (uint8_t*)realloc(core->canvas_shadow_rgba, needed);
    if (!grown) {
        return false;
    }
    if (needed > core->canvas_shadow_size) {
        memset(grown + core->canvas_shadow_size, 0, needed - core->canvas_shadow_size);
    }
    core->canvas_shadow_rgba = grown;
    core->canvas_shadow_size = needed;
    return true;
}

bool fs_target_format_is_bgra(WGPUTextureFormat format) {
    return format == WGPUTextureFormat_BGRA8Unorm || format == WGPUTextureFormat_BGRA8UnormSrgb;
}

void fs_canvas_readback_map_callback(
    WGPUMapAsyncStatus status,
    WGPUStringView message,
    void* userdata1,
    void* userdata2
) {
    (void)message;
    (void)userdata2;
    FS_MapReadbackContext* ctx = (FS_MapReadbackContext*)userdata1;
    if (!ctx) {
        return;
    }
    ctx->success = (status == WGPUMapAsyncStatus_Success) ? 1u : 0u;
    ctx->done = 1u;
}

bool fs_refresh_canvas_shadow_from_readback(FS_Core* core) {
    if (!core || !core->canvas_readback_buffer || core->canvas_readback_serial == 0u) {
        return false;
    }
    if (core->canvas_shadow_serial == core->canvas_readback_serial) {
        return true;
    }
    if (!fs_ensure_canvas_shadow(core)) {
        return false;
    }
    if (!core->canvas_shadow_rgba || core->canvas_shadow_size == 0u) {
        return false;
    }

    // Synchronous map: initiate, block until callback fires, read, unmap
    core->canvas_readback_map_ctx.done = 0u;
    core->canvas_readback_map_ctx.success = 0u;
    wgpuBufferMapAsync(
        core->canvas_readback_buffer,
        WGPUMapMode_Read,
        0u,
        core->canvas_readback_buffer_size,
        (WGPUBufferMapCallbackInfo){
            .nextInChain = NULL,
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback = fs_canvas_readback_map_callback,
            .userdata1 = &core->canvas_readback_map_ctx,
            .userdata2 = NULL
        }
    );
    // Block until map completes (single wait poll instead of spin-count loop)
    while (core->canvas_readback_map_ctx.done == 0u) {
        wgpuDevicePoll(core->device, true, NULL);
    }
    if (core->canvas_readback_map_ctx.success == 0u) {
        return false;
    }

    const uint8_t* mapped = (const uint8_t*)wgpuBufferGetConstMappedRange(
        core->canvas_readback_buffer,
        0u,
        core->canvas_readback_buffer_size
    );
    if (!mapped) {
        return false;
    }

    const bool is_bgra = fs_target_format_is_bgra(core->target_format);
    const uint32_t copy_width = core->canvas_readback_width;
    const uint32_t copy_height = core->canvas_readback_height;
    for (uint32_t row = 0u; row < copy_height; ++row) {
        const uint8_t* src = mapped + (size_t)row * (size_t)core->canvas_readback_padded_row_bytes;
        uint8_t* dst = core->canvas_shadow_rgba + (size_t)row * (size_t)core->canvas_readback_row_bytes;
        if (!is_bgra) {
            memcpy(dst, src, (size_t)copy_width * 4u);
        } else {
            // BGRA -> RGBA: swap R/B per pixel using uint32_t word ops
            // TODO: GPU-side swizzle in readback shader would eliminate this CPU pass
            uint32_t* d32 = (uint32_t*)dst;
            const uint32_t* s32 = (const uint32_t*)src;
            for (uint32_t x = 0u; x < copy_width; ++x) {
                uint32_t p = s32[x];
                d32[x] = (p & 0xFF00FF00) | ((p & 0xFF) << 16) | ((p >> 16) & 0xFF);
            }
        }
    }
    wgpuBufferUnmap(core->canvas_readback_buffer);
    core->canvas_shadow_serial = core->canvas_readback_serial;
    return true;
}

bool fs_ensure_canvas_image_data_handle(FS_Core* core, uint32_t width, uint32_t height, FS_ImageHandle* out_handle) {
    if (!core || width == 0u || height == 0u) {
        return false;
    }
    if (!core->canvas_image_data_handle_valid || core->canvas_image_data_handle.width != width ||
        core->canvas_image_data_handle.height != height ||
        core->canvas_image_data_handle.layer >= core->image_atlas_layers ||
        core->canvas_image_data_handle.generation != core->image_atlas_generation[core->canvas_image_data_handle.layer]) {
        FS_ImageHandle handle;
        const size_t zero_size = (size_t)width * (size_t)height * 4u;
        uint8_t* zero = (uint8_t*)calloc(1u, zero_size);
        if (!zero) {
            return false;
        }
        bool ok = fs_core_upload_image_rgba8(core, zero, width, height, &handle);
        free(zero);
        if (!ok) {
            return false;
        }
        core->canvas_image_data_handle = handle;
        core->canvas_image_data_handle_valid = 1u;
    }
    if (out_handle) {
        *out_handle = core->canvas_image_data_handle;
    }
    return true;
}

bool fs_encode_canvas_readback_copy(FS_Core* core, WGPUCommandEncoder encoder, WGPUTexture target_texture) {
    if (!core || !encoder || !target_texture || core->width == 0u || core->height == 0u) {
        return false;
    }
    const uint32_t row_bytes = core->width * 4u;
    const uint32_t padded_row = fs_align_up_u32(row_bytes, 256u);
    const size_t needed = (size_t)padded_row * (size_t)core->height;
    if (needed == 0u) {
        return false;
    }
    if (!core->canvas_readback_buffer || core->canvas_readback_buffer_size < needed ||
        core->canvas_readback_width != core->width || core->canvas_readback_height != core->height) {
        if (core->canvas_readback_buffer) {
            if (core->canvas_readback_mapped) {
                wgpuBufferUnmap(core->canvas_readback_buffer);
                core->canvas_readback_mapped = 0u;
            }
            wgpuBufferRelease(core->canvas_readback_buffer);
        }
        core->canvas_readback_buffer = fs_create_buffer(
            core->device,
            "FS Canvas Readback Buffer",
            WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
            needed
        );
        if (!core->canvas_readback_buffer) {
            core->canvas_readback_buffer_size = 0u;
            return false;
        }
        core->canvas_readback_buffer_size = needed;
        core->canvas_readback_width = core->width;
        core->canvas_readback_height = core->height;
    }
    core->canvas_readback_row_bytes = row_bytes;
    core->canvas_readback_padded_row_bytes = padded_row;
    core->canvas_readback_serial += 1u;
    core->canvas_readback_submission = 0u;
    core->canvas_readback_submission_valid = 0u;

    WGPUTexelCopyTextureInfo src = {
        .texture = target_texture,
        .mipLevel = 0u,
        .origin = {0u, 0u, 0u},
        .aspect = WGPUTextureAspect_All
    };
    WGPUTexelCopyBufferInfo dst = {
        .layout = {
            .offset = 0u,
            .bytesPerRow = padded_row,
            .rowsPerImage = core->height
        },
        .buffer = core->canvas_readback_buffer
    };
    WGPUExtent3D extent = {core->width, core->height, 1u};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);
    return true;
}
