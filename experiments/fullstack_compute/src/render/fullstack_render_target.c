#include "fullstack_render_private.h"

#include <string.h>

static FS_Result fs_render_target_validate(const FS_RenderTargetDesc* desc,
                                           bool imported, FS_Error* error) {
    if (!desc || desc->struct_size < sizeof(*desc) ||
        desc->abi_version != FS_RENDER_ABI_VERSION || !desc->width ||
        !desc->height || desc->format == WGPUTextureFormat_Undefined ||
        !desc->generation) {
        return FS_RESULT_INVALID_ARGUMENT;
    }
    if (imported && (!desc->texture || !desc->view)) {
        FS_ERROR_SET(error, FS_RESULT_INVALID_ARGUMENT, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_render_target_import",
                     "an imported target requires both texture and view");
        return FS_RESULT_INVALID_ARGUMENT;
    }
    return FS_RESULT_OK;
}

static FS_RenderTarget* fs_render_target_allocate(
    const FS_RenderTargetDesc* desc) {
    const FS_Allocator* allocator = desc->allocator ? desc->allocator
                                                    : fs_default_allocator();
    FS_RenderTarget* target = (FS_RenderTarget*)fs_allocator_allocate(
        allocator, sizeof(*target), sizeof(void*));
    if (!target) return NULL;
    memset(target, 0, sizeof(*target));
    target->allocator = allocator;
    target->format = desc->format;
    target->width = desc->width;
    target->height = desc->height;
    target->color_space = desc->color_space;
    target->generation = desc->generation;
    return target;
}

FS_Result FS_CALL fs_render_target_import(const FS_RenderTargetDesc* desc,
                                          FS_RenderTarget** out_target,
                                          FS_Error* error) {
    if (out_target) *out_target = NULL;
    if (!out_target) return FS_RESULT_INVALID_ARGUMENT;
    FS_Result result = fs_render_target_validate(desc, true, error);
    if (result != FS_RESULT_OK) return result;
    FS_RenderTarget* target = fs_render_target_allocate(desc);
    if (!target) return FS_RESULT_OUT_OF_MEMORY;
    target->texture = desc->texture;
    target->view = desc->view;
    target->importable = true;
    wgpuTextureAddRef(target->texture);
    wgpuTextureViewAddRef(target->view);
    *out_target = target;
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_render_target_update_import(
    FS_RenderTarget* target, const FS_RenderTargetDesc* desc, FS_Error* error) {
    if (!target || !target->importable) return FS_RESULT_INVALID_STATE;
    FS_Result result = fs_render_target_validate(desc, true, error);
    if (result != FS_RESULT_OK) return result;
    wgpuTextureAddRef(desc->texture);
    wgpuTextureViewAddRef(desc->view);
    WGPUTexture old_texture = target->texture;
    WGPUTextureView old_view = target->view;
    target->texture = desc->texture;
    target->view = desc->view;
    target->format = desc->format;
    target->width = desc->width;
    target->height = desc->height;
    target->color_space = desc->color_space;
    target->generation = desc->generation;
    if (old_view) wgpuTextureViewRelease(old_view);
    if (old_texture) wgpuTextureRelease(old_texture);
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_render_target_create_texture(
    FS_GpuContext* gpu, const FS_RenderTargetDesc* desc,
    FS_RenderTarget** out_target, FS_Error* error) {
    if (out_target) *out_target = NULL;
    if (!gpu || !out_target) return FS_RESULT_INVALID_ARGUMENT;
    FS_Result result = fs_render_target_validate(desc, false, error);
    if (result != FS_RESULT_OK) return result;
    if (desc->texture || desc->view) return FS_RESULT_INVALID_ARGUMENT;
    FS_RenderTarget* target = fs_render_target_allocate(desc);
    if (!target) return FS_RESULT_OUT_OF_MEMORY;
    WGPUTextureDescriptor texture_desc = {
        .usage = WGPUTextureUsage_RenderAttachment |
                 WGPUTextureUsage_TextureBinding |
                 WGPUTextureUsage_CopySrc,
        .dimension = WGPUTextureDimension_2D,
        .size = {desc->width, desc->height, 1},
        .format = desc->format,
        .mipLevelCount = 1,
        .sampleCount = 1,
        .viewFormatCount = 0,
        .viewFormats = NULL};
    target->texture = wgpuDeviceCreateTexture(fs_gpu_device(gpu), &texture_desc);
    if (target->texture)
        target->view = wgpuTextureCreateView(target->texture, NULL);
    if (!target->texture || !target->view) {
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_render_target_create_texture",
                     "render target texture creation failed");
        fs_render_target_destroy(target);
        return FS_RESULT_INTERNAL_ERROR;
    }
    *out_target = target;
    return FS_RESULT_OK;
}

void FS_CALL fs_render_target_destroy(FS_RenderTarget* target) {
    if (!target) return;
    if (target->view) wgpuTextureViewRelease(target->view);
    if (target->texture) wgpuTextureRelease(target->texture);
    const FS_Allocator* allocator = target->allocator;
    fs_allocator_deallocate(allocator, target, sizeof(*target), sizeof(void*));
}

uint32_t FS_CALL fs_render_target_width(const FS_RenderTarget* target) {
    return target ? target->width : 0;
}
uint32_t FS_CALL fs_render_target_height(const FS_RenderTarget* target) {
    return target ? target->height : 0;
}
uint64_t FS_CALL fs_render_target_generation(const FS_RenderTarget* target) {
    return target ? target->generation : 0;
}
WGPUTexture FS_CALL fs_render_target_texture(const FS_RenderTarget* target) {
    return target ? target->texture : NULL;
}
WGPUTextureView FS_CALL fs_render_target_view(const FS_RenderTarget* target) {
    return target ? target->view : NULL;
}
