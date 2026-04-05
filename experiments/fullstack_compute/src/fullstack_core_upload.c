#include "fullstack_core_private.h"

#include <stdlib.h>
#include <string.h>

void fs_discard_pending_uploads_for_texture(FS_Core* core, WGPUTexture texture) {
    if (!core || !texture || core->pending_upload_count == 0u || !core->pending_uploads) {
        return;
    }
    if (!core->upload_staging_cpu) {
        core->pending_upload_count = 0u;
        core->upload_staging_used = 0u;
        return;
    }

    FS_PendingTextureUpload* uploads = (FS_PendingTextureUpload*)core->pending_uploads;
    size_t write_idx = 0u;
    size_t new_used = 0u;

    for (size_t i = 0u; i < core->pending_upload_count; ++i) {
        FS_PendingTextureUpload u = uploads[i];
        if (u.texture == texture) {
            continue;
        }
        const size_t upload_size = (size_t)u.padded_row_bytes * (size_t)u.height;
        const size_t new_offset = (size_t)fs_align_up_u32((uint32_t)new_used, 256u);
        if (new_offset != u.src_offset) {
            memmove(core->upload_staging_cpu + new_offset, core->upload_staging_cpu + u.src_offset, upload_size);
        }
        u.src_offset = new_offset;
        uploads[write_idx++] = u;
        new_used = new_offset + upload_size;
    }

    core->pending_upload_count = write_idx;
    core->upload_staging_used = (write_idx > 0u) ? new_used : 0u;
}

bool fs_ensure_upload_staging_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    if (required <= core->upload_staging_cpu_capacity) {
        return true;
    }
    size_t new_capacity = core->upload_staging_cpu_capacity ? core->upload_staging_cpu_capacity : 4096u;
    while (new_capacity < required) {
        if (new_capacity > (SIZE_MAX / 2u)) {
            new_capacity = required;
            break;
        }
        new_capacity *= 2u;
    }
    uint8_t* grown = (uint8_t*)realloc(core->upload_staging_cpu, new_capacity);
    if (!grown) {
        return false;
    }
    core->upload_staging_cpu = grown;
    core->upload_staging_cpu_capacity = new_capacity;
    return true;
}

bool fs_ensure_pending_upload_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    if (required <= core->pending_upload_capacity) {
        return true;
    }
    size_t new_capacity = core->pending_upload_capacity ? core->pending_upload_capacity : 128u;
    while (new_capacity < required) {
        if (new_capacity > (SIZE_MAX / 2u)) {
            new_capacity = required;
            break;
        }
        new_capacity *= 2u;
    }
    FS_PendingTextureUpload* grown =
        (FS_PendingTextureUpload*)realloc(core->pending_uploads, new_capacity * sizeof(FS_PendingTextureUpload));
    if (!grown) {
        return false;
    }
    core->pending_uploads = grown;
    core->pending_upload_capacity = new_capacity;
    return true;
}

bool fs_queue_write_texture_2d(
    FS_Core* core,
    WGPUTexture texture,
    uint32_t layer,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    const uint8_t* pixels,
    uint32_t bytes_per_pixel
) {
    if (!core || !texture || !pixels || width == 0u || height == 0u || bytes_per_pixel == 0u) {
        return false;
    }
    const uint64_t tight_row = (uint64_t)width * (uint64_t)bytes_per_pixel;
    if (tight_row == 0u || tight_row > UINT32_MAX) {
        return false;
    }
    const uint32_t padded_row = fs_align_up_u32((uint32_t)tight_row, 256u);
    const size_t upload_size = (size_t)padded_row * (size_t)height;
    const size_t upload_offset = core->upload_staging_used;
    const size_t required_size = upload_offset + upload_size;
    if (required_size < upload_offset) {
        return false;
    }
    if (!fs_ensure_upload_staging_capacity(core, required_size)) {
        return false;
    }
    if (!fs_ensure_pending_upload_capacity(core, core->pending_upload_count + 1u)) {
        return false;
    }

    uint8_t* dst = core->upload_staging_cpu + upload_offset;
    const size_t src_row = (size_t)width * (size_t)bytes_per_pixel;
    for (uint32_t row = 0u; row < height; ++row) {
        memcpy(dst + (size_t)row * (size_t)padded_row, pixels + (size_t)row * src_row, src_row);
        if (padded_row > src_row) {
            memset(dst + (size_t)row * (size_t)padded_row + src_row, 0, (size_t)padded_row - src_row);
        }
    }

    if (texture == core->image_atlas_texture && bytes_per_pixel == 4u &&
        fs_image_atlas_shadow_bounds_ok(core, layer, x, y, width, height)) {
        const size_t layer_stride = (size_t)core->image_atlas_width * (size_t)core->image_atlas_height * 4u;
        uint8_t* shadow_base = core->image_atlas_shadow_rgba + (size_t)layer * layer_stride;
        const size_t shadow_row = (size_t)core->image_atlas_width * 4u;
        for (uint32_t row = 0u; row < height; ++row) {
            memcpy(
                shadow_base + ((size_t)(y + row) * shadow_row) + (size_t)x * 4u,
                pixels + (size_t)row * src_row,
                src_row
            );
        }
    }

    FS_PendingTextureUpload* uploads = (FS_PendingTextureUpload*)core->pending_uploads;
    uploads[core->pending_upload_count++] = (FS_PendingTextureUpload){
        .texture = texture,
        .mip_level = 0u,
        .layer = layer,
        .x = x,
        .y = y,
        .width = width,
        .height = height,
        .padded_row_bytes = padded_row,
        .src_offset = upload_offset
    };
    core->upload_staging_used = required_size;
    return true;
}

bool fs_flush_pending_texture_uploads(FS_Core* core, WGPUCommandEncoder encoder) {
    if (!core) {
        return false;
    }
    if (core->pending_upload_count == 0u) {
        core->upload_staging_used = 0u;
        return true;
    }
    if (!encoder || !core->queue || !core->device || !core->upload_staging_cpu) {
        return false;
    }
    if (core->upload_staging_used > core->upload_staging_gpu_capacity) {
        size_t new_size = core->upload_staging_gpu_capacity ? core->upload_staging_gpu_capacity : 4096u;
        while (new_size < core->upload_staging_used) {
            if (new_size > (SIZE_MAX / 2u)) {
                new_size = core->upload_staging_used;
                break;
            }
            new_size *= 2u;
        }
        WGPUBuffer new_buf = fs_create_buffer(
            core->device,
            "FS Upload Staging Buffer",
            WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
            new_size
        );
        if (!new_buf) {
            return false;
        }
        if (core->upload_staging_gpu) {
            wgpuBufferRelease(core->upload_staging_gpu);
        }
        core->upload_staging_gpu = new_buf;
        core->upload_staging_gpu_capacity = new_size;
    }
    if (!core->upload_staging_gpu) {
        return false;
    }

    wgpuQueueWriteBuffer(core->queue, core->upload_staging_gpu, 0u, core->upload_staging_cpu, core->upload_staging_used);

    FS_PendingTextureUpload* uploads = (FS_PendingTextureUpload*)core->pending_uploads;
    for (size_t i = 0u; i < core->pending_upload_count; ++i) {
        const FS_PendingTextureUpload* upload = &uploads[i];
        WGPUTexelCopyBufferInfo src = {
            .layout = {
                .offset = upload->src_offset,
                .bytesPerRow = upload->padded_row_bytes,
                .rowsPerImage = upload->height
            },
            .buffer = core->upload_staging_gpu
        };
        WGPUTexelCopyTextureInfo dst = {
            .texture = upload->texture,
            .mipLevel = upload->mip_level,
            .origin = {upload->x, upload->y, upload->layer},
            .aspect = WGPUTextureAspect_All
        };
        WGPUExtent3D extent = {upload->width, upload->height, 1u};
        wgpuCommandEncoderCopyBufferToTexture(encoder, &src, &dst, &extent);
    }

    core->pending_upload_count = 0u;
    core->upload_staging_used = 0u;
    return true;
}

bool fs_image_atlas_shadow_read_rgba(
    const FS_Core* core,
    uint32_t layer,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    uint8_t* out_rgba_pixels
) {
    if (!out_rgba_pixels || !fs_image_atlas_shadow_bounds_ok(core, layer, x, y, width, height)) {
        return false;
    }
    const size_t atlas_row = (size_t)core->image_atlas_width * 4u;
    const size_t out_row = (size_t)width * 4u;
    const size_t layer_stride = (size_t)core->image_atlas_width * (size_t)core->image_atlas_height * 4u;
    const uint8_t* src_base = core->image_atlas_shadow_rgba + (size_t)layer * layer_stride;
    for (uint32_t row = 0u; row < height; ++row) {
        memcpy(
            out_rgba_pixels + (size_t)row * out_row,
            src_base + ((size_t)(y + row) * atlas_row) + (size_t)x * 4u,
            out_row
        );
    }
    return true;
}
