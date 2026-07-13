#include "fullstack_present.h"

#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

typedef struct FS_PresenterPipelineCacheEntry {
    const FS_Allocator* allocator;
    WGPUDevice device;
    WGPUTextureFormat format;
    uint32_t references;
    WGPUBindGroupLayout bind_group_layout;
    WGPUPipelineLayout pipeline_layout;
    WGPUShaderModule shader;
    WGPUSampler sampler;
    WGPURenderPipeline pipeline;
    struct FS_PresenterPipelineCacheEntry* next;
} FS_PresenterPipelineCacheEntry;

struct FS_Presenter {
    const FS_Allocator* allocator;
    WGPUDevice device;
    WGPUTextureFormat format;
    FS_ColorSpace source_color_space;
    FS_ColorSpace target_color_space;
    FS_PresenterPipelineCacheEntry* cache;
    WGPUBindGroup source_bind_group;
    WGPUTextureView source_view;
};

static const char* fs_present_shader =
    "@group(0) @binding(0) var tex: texture_2d<f32>;"
    "@group(0) @binding(1) var samp: sampler;"
    "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {"
    " var x = f32(i & 1u) * 4.0 - 1.0;"
    " var y = f32(i >> 1u) * 4.0 - 1.0;"
    " return vec4<f32>(x, y, 0.0, 1.0);"
    "}"
    "@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {"
    " let size = vec2<f32>(textureDimensions(tex));"
    " return textureSample(tex, samp, p.xy / size);"
    "}";

static volatile long fs_present_cache_lock_value;
static FS_PresenterPipelineCacheEntry* fs_present_cache_head;

static void fs_present_cache_lock(void) {
#if defined(_MSC_VER)
    while (_InterlockedExchange(&fs_present_cache_lock_value, 1)) {}
#else
    while (__sync_lock_test_and_set(&fs_present_cache_lock_value, 1)) {}
#endif
}

static void fs_present_cache_unlock(void) {
#if defined(_MSC_VER)
    _InterlockedExchange(&fs_present_cache_lock_value, 0);
#else
    __sync_lock_release(&fs_present_cache_lock_value);
#endif
}

static void fs_present_cache_entry_destroy(
    FS_PresenterPipelineCacheEntry* entry) {
    if (!entry) return;
    if (entry->pipeline) wgpuRenderPipelineRelease(entry->pipeline);
    if (entry->sampler) wgpuSamplerRelease(entry->sampler);
    if (entry->shader) wgpuShaderModuleRelease(entry->shader);
    if (entry->pipeline_layout) wgpuPipelineLayoutRelease(entry->pipeline_layout);
    if (entry->bind_group_layout)
        wgpuBindGroupLayoutRelease(entry->bind_group_layout);
    fs_allocator_deallocate(entry->allocator, entry, sizeof(*entry), sizeof(void*));
}

static FS_PresenterPipelineCacheEntry* fs_present_cache_entry_create(
    const FS_Allocator* allocator, WGPUDevice device, WGPUTextureFormat format) {
    FS_PresenterPipelineCacheEntry* entry =
        (FS_PresenterPipelineCacheEntry*)fs_allocator_allocate(
            allocator, sizeof(*entry), sizeof(void*));
    if (!entry) return NULL;
    memset(entry, 0, sizeof(*entry));
    entry->allocator = allocator;
    entry->device = device;
    entry->format = format;
    entry->references = 1;
    WGPUBindGroupLayoutEntry bindings[2] = {0};
    bindings[0].binding = 0;
    bindings[0].visibility = WGPUShaderStage_Fragment;
    bindings[0].texture.sampleType = WGPUTextureSampleType_Float;
    bindings[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    bindings[1].binding = 1;
    bindings[1].visibility = WGPUShaderStage_Fragment;
    bindings[1].sampler.type = WGPUSamplerBindingType_Filtering;
    entry->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
        device, &(WGPUBindGroupLayoutDescriptor){
            .entryCount = 2, .entries = bindings});
    if (!entry->bind_group_layout) goto fail;
    entry->pipeline_layout = wgpuDeviceCreatePipelineLayout(
        device, &(WGPUPipelineLayoutDescriptor){
            .bindGroupLayoutCount = 1,
            .bindGroupLayouts = &entry->bind_group_layout});
    if (!entry->pipeline_layout) goto fail;
    WGPUShaderSourceWGSL wgsl = {
        .chain = {.sType = WGPUSType_ShaderSourceWGSL},
        .code = {.data = fs_present_shader, .length = WGPU_STRLEN}};
    entry->shader = wgpuDeviceCreateShaderModule(
        device, &(WGPUShaderModuleDescriptor){.nextInChain = &wgsl.chain});
    if (!entry->shader) goto fail;
    entry->sampler = wgpuDeviceCreateSampler(
        device, &(WGPUSamplerDescriptor){
            .addressModeU = WGPUAddressMode_ClampToEdge,
            .addressModeV = WGPUAddressMode_ClampToEdge,
            .addressModeW = WGPUAddressMode_ClampToEdge,
            .magFilter = WGPUFilterMode_Linear,
            .minFilter = WGPUFilterMode_Linear,
            .mipmapFilter = WGPUMipmapFilterMode_Nearest,
            .maxAnisotropy = 1});
    if (!entry->sampler) goto fail;
    WGPUColorTargetState target = {
        .format = format, .writeMask = WGPUColorWriteMask_All};
    WGPUFragmentState fragment = {
        .module = entry->shader,
        .entryPoint = {.data = "fs", .length = 2},
        .targetCount = 1,
        .targets = &target};
    WGPURenderPipelineDescriptor pipeline = {
        .layout = entry->pipeline_layout,
        .vertex = {.module = entry->shader,
                   .entryPoint = {.data = "vs", .length = 2}},
        .primitive = {.topology = WGPUPrimitiveTopology_TriangleList,
                      .stripIndexFormat = WGPUIndexFormat_Undefined,
                      .frontFace = WGPUFrontFace_CCW,
                      .cullMode = WGPUCullMode_None},
        .multisample = {.count = 1, .mask = 0xffffffffu},
        .fragment = &fragment};
    entry->pipeline = wgpuDeviceCreateRenderPipeline(device, &pipeline);
    if (!entry->pipeline) goto fail;
    return entry;
fail:
    fs_present_cache_entry_destroy(entry);
    return NULL;
}

static FS_PresenterPipelineCacheEntry* fs_present_cache_acquire(
    const FS_Allocator* allocator, WGPUDevice device, WGPUTextureFormat format) {
    fs_present_cache_lock();
    for (FS_PresenterPipelineCacheEntry* entry = fs_present_cache_head;
         entry; entry = entry->next) {
        if (entry->device == device && entry->format == format) {
            entry->references++;
            fs_present_cache_unlock();
            return entry;
        }
    }
    fs_present_cache_unlock();
    FS_PresenterPipelineCacheEntry* created = fs_present_cache_entry_create(
        allocator, device, format);
    if (!created) return NULL;
    fs_present_cache_lock();
    for (FS_PresenterPipelineCacheEntry* entry = fs_present_cache_head;
         entry; entry = entry->next) {
        if (entry->device == device && entry->format == format) {
            entry->references++;
            fs_present_cache_unlock();
            fs_present_cache_entry_destroy(created);
            return entry;
        }
    }
    created->next = fs_present_cache_head;
    fs_present_cache_head = created;
    fs_present_cache_unlock();
    return created;
}

static void fs_present_cache_release(FS_PresenterPipelineCacheEntry* entry) {
    if (!entry) return;
    bool destroy = false;
    fs_present_cache_lock();
    if (--entry->references == 0) {
        FS_PresenterPipelineCacheEntry** link = &fs_present_cache_head;
        while (*link && *link != entry) link = &(*link)->next;
        if (*link == entry) *link = entry->next;
        destroy = true;
    }
    fs_present_cache_unlock();
    if (destroy) fs_present_cache_entry_destroy(entry);
}

static bool fs_present_format_is_srgb(WGPUTextureFormat format) {
    return format == WGPUTextureFormat_RGBA8UnormSrgb ||
           format == WGPUTextureFormat_BGRA8UnormSrgb;
}

static bool fs_present_format_is_supported(WGPUTextureFormat format) {
    return format == WGPUTextureFormat_RGBA8Unorm ||
           format == WGPUTextureFormat_RGBA8UnormSrgb ||
           format == WGPUTextureFormat_BGRA8Unorm ||
           format == WGPUTextureFormat_BGRA8UnormSrgb;
}

FS_Result FS_CALL fs_presenter_validate_color_contract(
    WGPUTextureFormat format, FS_ColorSpace source, FS_ColorSpace target,
    FS_Error* error) {
    if (!fs_present_format_is_supported(format)) {
        FS_ERROR_SET(error, FS_RESULT_UNSUPPORTED, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_presenter_validate_color_contract",
                     "unsupported presentation texture format");
        return FS_RESULT_UNSUPPORTED;
    }
    if (source != FS_COLOR_SPACE_LINEAR_SRGB) {
        FS_ERROR_SET(error, FS_RESULT_UNSUPPORTED, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_presenter_validate_color_contract",
                     "only linear-sRGB scene input is currently supported");
        return FS_RESULT_UNSUPPORTED;
    }
    if (target == FS_COLOR_SPACE_DISPLAY_P3 || target == FS_COLOR_SPACE_HDR10) {
        FS_ERROR_SET(error, FS_RESULT_UNSUPPORTED, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_presenter_validate_color_contract",
                     "Display P3 and HDR descriptors are reserved but unsupported");
        return FS_RESULT_UNSUPPORTED;
    }
    if (target != FS_COLOR_SPACE_LINEAR_SRGB && target != FS_COLOR_SPACE_SRGB) {
        return FS_RESULT_INVALID_ARGUMENT;
    }
    if ((target == FS_COLOR_SPACE_SRGB) != fs_present_format_is_srgb(format)) {
        FS_ERROR_SET(error, FS_RESULT_UNSUPPORTED, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_presenter_validate_color_contract",
                     "target color space must match the WebGPU texture format");
        return FS_RESULT_UNSUPPORTED;
    }
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_presenter_create(const FS_PresenterDesc* desc,
                                      FS_Presenter** out_presenter,
                                      FS_Error* error) {
    if (out_presenter) *out_presenter = NULL;
    if (!desc || !out_presenter || desc->struct_size < sizeof(*desc) ||
        desc->abi_version != FS_PRESENT_ABI_VERSION || !desc->device) {
        return FS_RESULT_INVALID_ARGUMENT;
    }
    FS_Result result = fs_presenter_validate_color_contract(
        desc->target_format, desc->source_color_space,
        desc->target_color_space, error);
    if (result != FS_RESULT_OK) return result;

    const FS_Allocator* allocator = desc->allocator ? desc->allocator
                                                    : fs_default_allocator();
    FS_Presenter* presenter = (FS_Presenter*)fs_allocator_allocate(
        allocator, sizeof(*presenter), sizeof(void*));
    if (!presenter) return FS_RESULT_OUT_OF_MEMORY;
    memset(presenter, 0, sizeof(*presenter));
    presenter->allocator = allocator;
    presenter->device = desc->device;
    presenter->format = desc->target_format;
    presenter->source_color_space = desc->source_color_space;
    presenter->target_color_space = desc->target_color_space;

    presenter->cache = fs_present_cache_acquire(
        allocator, presenter->device, presenter->format);
    if (!presenter->cache) goto fail;
    *out_presenter = presenter;
    return FS_RESULT_OK;

fail:
    FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_RENDER, 0,
                 "fs_presenter_create", "presentation resource creation failed");
    fs_presenter_destroy(presenter);
    return FS_RESULT_INTERNAL_ERROR;
}

void FS_CALL fs_presenter_destroy(FS_Presenter* presenter) {
    if (!presenter) return;
    if (presenter->source_bind_group)
        wgpuBindGroupRelease(presenter->source_bind_group);
    fs_present_cache_release(presenter->cache);
    const FS_Allocator* allocator = presenter->allocator;
    fs_allocator_deallocate(allocator, presenter, sizeof(*presenter), sizeof(void*));
}

FS_Result FS_CALL fs_presenter_set_source(FS_Presenter* presenter,
                                          WGPUTextureView view,
                                          FS_Error* error) {
    if (!presenter || !view) return FS_RESULT_INVALID_ARGUMENT;
    if (presenter->source_view == view && presenter->source_bind_group)
        return FS_RESULT_OK;
    if (presenter->source_bind_group) {
        wgpuBindGroupRelease(presenter->source_bind_group);
        presenter->source_bind_group = NULL;
    }
    WGPUBindGroupEntry entries[2] = {0};
    entries[0].binding = 0;
    entries[0].textureView = view;
    entries[1].binding = 1;
    entries[1].sampler = presenter->cache->sampler;
    presenter->source_bind_group = wgpuDeviceCreateBindGroup(
        presenter->device, &(WGPUBindGroupDescriptor){
            .layout = presenter->cache->bind_group_layout,
            .entryCount = 2,
            .entries = entries});
    if (!presenter->source_bind_group) {
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_presenter_set_source", "source bind group creation failed");
        return FS_RESULT_INTERNAL_ERROR;
    }
    presenter->source_view = view;
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_presenter_encode(FS_Presenter* presenter,
                                      WGPUCommandEncoder encoder,
                                      WGPUTextureView target,
                                      uint32_t width, uint32_t height,
                                      FS_Error* error) {
    if (!presenter || !encoder || !target || !presenter->source_bind_group ||
        !width || !height) return FS_RESULT_INVALID_ARGUMENT;
    WGPURenderPassColorAttachment color = {
        .view = target,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        .loadOp = WGPULoadOp_Clear,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = {0, 0, 0, 0}};
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
        encoder, &(WGPURenderPassDescriptor){
            .colorAttachmentCount = 1, .colorAttachments = &color});
    if (!pass) {
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_presenter_encode", "render pass creation failed");
        return FS_RESULT_INTERNAL_ERROR;
    }
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)width, (float)height, 0, 1);
    wgpuRenderPassEncoderSetPipeline(pass, presenter->cache->pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, presenter->source_bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    return FS_RESULT_OK;
}
