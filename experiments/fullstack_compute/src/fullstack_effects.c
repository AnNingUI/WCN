#include "webgpu/webgpu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
// fullstack_effects.h MUST come first: it defines struct FS_EffectResources which
// fullstack_filters.h include guard (set via fullstack_effects.h) would otherwise
// block from being visible before fullstack_core_private.h uses it.
#include "fullstack_effects.h"
#include "fullstack_core_private.h"
#include "fullstack_shaders.h"

static WGPUShaderModule fs_effects_create_shader_module(WGPUDevice device, const char* wgsl, const char* label) {
    WGPUStringView sv = { .data = wgsl, .length = strlen(wgsl) };
    WGPUShaderSourceWGSL source = { .chain = { .next = NULL, .sType = WGPUSType_ShaderSourceWGSL }, .code = sv };
    WGPUShaderModuleDescriptor desc = { .nextInChain = &source.chain, .label = { .data = label, .length = label ? strlen(label) : 0 } };
    return wgpuDeviceCreateShaderModule(device, &desc);
}

static void fs_effects_release_bg(WGPUBindGroup bg) { if (bg) wgpuBindGroupRelease(bg); }
static void fs_effects_release_bgl(WGPUBindGroupLayout bgl) { if (bgl) wgpuBindGroupLayoutRelease(bgl); }
static void fs_effects_release_compute_pipeline(WGPUComputePipeline p) { if (p) wgpuComputePipelineRelease(p); }
static void fs_effects_release_render_pipeline(WGPURenderPipeline p) { if (p) wgpuRenderPipelineRelease(p); }
static void fs_effects_release_pipeline_layout(WGPUPipelineLayout p) { if (p) wgpuPipelineLayoutRelease(p); }

static inline float fs_gaussian_1d(float x, float sigma) { return expf(-(x * x) / (2.0f * sigma * sigma)); }

bool fs_gaussian_kernel_compute(FS_GaussianKernel* out_kernel, float sigma, uint32_t kernel_size) {
    if (!out_kernel || kernel_size < 3 || kernel_size > FS_GAUSSIAN_KERNEL_MAX_SIZE || kernel_size % 2 == 0 || sigma <= 0.0f) return false;
    out_kernel->size = kernel_size; out_kernel->sigma = sigma;
    const uint32_t center = kernel_size / 2; float sum = 0.0f;
    for (uint32_t i = 0; i < kernel_size; i++) { out_kernel->weights[i] = fs_gaussian_1d((float)i - (float)center, sigma); sum += out_kernel->weights[i]; }
    for (uint32_t i = 0; i < kernel_size; i++) out_kernel->weights[i] /= sum;
    return true;
}

bool fs_gaussian_kernel_compute_for_blur(FS_GaussianKernel* out_kernel, float blur_radius, uint32_t* out_kernel_size) {
    if (blur_radius < 0.0f) blur_radius = 0.0f;
    if (blur_radius > FS_GAUSSIAN_BLUR_RADIUS_MAX) blur_radius = FS_GAUSSIAN_BLUR_RADIUS_MAX;
    float sigma = blur_radius / 3.0f;
    uint32_t kernel_size = ((uint32_t)(6.0f * sigma)) | 1u;
    if (kernel_size > FS_GAUSSIAN_KERNEL_MAX_SIZE) kernel_size = FS_GAUSSIAN_KERNEL_MAX_SIZE;
    if (kernel_size < 3) kernel_size = 3;
    if (out_kernel_size) *out_kernel_size = kernel_size;
    if (out_kernel) return fs_gaussian_kernel_compute(out_kernel, sigma, kernel_size);
    return true;
}


void fs_effects_destroy(FS_Core* core) {
    if (!core) return;
    struct FS_EffectResources* res = fs_core_get_effects_resources(core);
    if (!res) return;
    fs_effects_release_bg(res->gaussian_blur_h_bg_a); fs_effects_release_bg(res->gaussian_blur_h_bg_b);
    fs_effects_release_bg(res->gaussian_blur_v_bg_a); fs_effects_release_bg(res->gaussian_blur_v_bg_b);
    fs_effects_release_bg(res->filter_bg_a); fs_effects_release_bg(res->filter_bg_b);
    fs_effects_release_bg(res->filter_copy_bg); fs_effects_release_bg(res->filter_copy_bg_back);
    fs_effects_release_bg(res->drop_shadow_bg); fs_effects_release_bg(res->drop_shadow_temp_bg);
    fs_effects_release_bg(res->shadow_composite_bg);
    fs_effects_release_bg(res->drop_shadow_c_bg);
    fs_effects_release_bgl(res->gaussian_blur_bgl); fs_effects_release_bgl(res->filter_bgl);
    fs_effects_release_bgl(res->filter_copy_bgl); fs_effects_release_bgl(res->drop_shadow_bgl);
    fs_effects_release_bgl(res->shadow_composite_bgl);
    fs_effects_release_compute_pipeline(res->gaussian_blur_h_pipeline);
    fs_effects_release_compute_pipeline(res->gaussian_blur_v_pipeline);
    fs_effects_release_compute_pipeline(res->filter_pipeline);
    fs_effects_release_render_pipeline(res->filter_copy_pipeline);
    fs_effects_release_render_pipeline(res->drop_shadow_pipeline);
    fs_effects_release_render_pipeline(res->shadow_composite_pipeline);
    fs_effects_release_pipeline_layout(res->gaussian_blur_pipeline_layout);
    fs_effects_release_pipeline_layout(res->filter_pipeline_layout);
    fs_effects_release_pipeline_layout(res->filter_copy_pipeline_layout);
    fs_effects_release_pipeline_layout(res->drop_shadow_pipeline_layout);
    fs_effects_release_pipeline_layout(res->shadow_composite_pipeline_layout);
    if (res->gaussian_shader_module) wgpuShaderModuleRelease(res->gaussian_shader_module);
    if (res->filter_shader_module) wgpuShaderModuleRelease(res->filter_shader_module);
    if (res->filter_copy_shader_module) wgpuShaderModuleRelease(res->filter_copy_shader_module);
    if (res->drop_shadow_shader_module) wgpuShaderModuleRelease(res->drop_shadow_shader_module);
    if (res->shadow_composite_shader_module) wgpuShaderModuleRelease(res->shadow_composite_shader_module);
    if (res->vert_shader_module) wgpuShaderModuleRelease(res->vert_shader_module);
    if (res->scene_view) wgpuTextureViewRelease(res->scene_view);
    if (res->scene_texture) wgpuTextureRelease(res->scene_texture);
    if (res->shadow_sampler) wgpuSamplerRelease(res->shadow_sampler);
    if (res->gaussian_uniform_buffer) wgpuBufferRelease(res->gaussian_uniform_buffer);
    if (res->gaussian_kernel_buffer) wgpuBufferRelease(res->gaussian_kernel_buffer);
    if (res->filter_uniform_buffer) wgpuBufferRelease(res->filter_uniform_buffer);
    if (res->shadow_uniform_buffer) wgpuBufferRelease(res->shadow_uniform_buffer);
    if (res->presentation_pipeline) {
        fs_effects_release_bg(res->presentation_scene_bg); res->presentation_scene_bg = NULL;
        fs_effects_release_render_pipeline(res->presentation_pipeline); res->presentation_pipeline = NULL;
    }
    if (res->shadow_composite_view) wgpuTextureViewRelease(res->shadow_composite_view);
    if (res->shadow_composite_texture) wgpuTextureRelease(res->shadow_composite_texture);
    fs_effects_release_compute_pipeline(res->drop_shadow_c_pipeline);
    fs_effects_release_pipeline_layout(res->drop_shadow_c_pipeline_layout);
    fs_effects_release_bgl(res->drop_shadow_c_bgl);
    if (res->drop_shadow_c_shader_module) wgpuShaderModuleRelease(res->drop_shadow_c_shader_module);
    if (res->ping_pong_view_a) wgpuTextureViewRelease(res->ping_pong_view_a);
    if (res->ping_pong_view_b) wgpuTextureViewRelease(res->ping_pong_view_b);
    if (res->ping_pong_texture_a) wgpuTextureRelease(res->ping_pong_texture_a);
    if (res->ping_pong_texture_b) wgpuTextureRelease(res->ping_pong_texture_b);
    free(res); fs_core_set_effects_resources(core, NULL);
}

bool fs_effects_init(FS_Core* core) {
    if (!core || !core->device) return false;
    struct FS_EffectResources* res = (struct FS_EffectResources*)calloc(1, sizeof(struct FS_EffectResources));
    if (!res) return false;
    WGPUDevice device = core->device;
    res->width = core->width; res->height = core->height; res->enabled = false; res->kernel_dirty = true;
    res->filter_textures_ready = false;
    bool ok = true;

    // ============================
    // Phase 1: Always-create resources (no ping-pong dependency)
    // ============================

    // 1. Create scene texture (RGBA8Unorm, RenderAttachment + Storage + CopySrc/Dst)
    if (ok) {
        WGPUTextureDescriptor sceneDesc = {
            .nextInChain = NULL,
            .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst,
            .dimension = WGPUTextureDimension_2D, .size = { core->width, core->height, 1 },
            .format = WGPUTextureFormat_RGBA8Unorm, .mipLevelCount = 1, .sampleCount = 1, .viewFormatCount = 0, .viewFormats = NULL
        };
        sceneDesc.label.data = "FS Scene"; sceneDesc.label.length = 8;
        res->scene_texture = wgpuDeviceCreateTexture(device, &sceneDesc);
        if (!res->scene_texture) ok = false;
    }
    if (ok) {
        WGPUTextureViewDescriptor sceneViewDesc = {
            .nextInChain = NULL, .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D,
            .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All
        };
        if (res->scene_texture) res->scene_view = wgpuTextureCreateView(res->scene_texture, &sceneViewDesc);
        if (!res->scene_view) ok = false;
    }

    // 2. Compile ALL shader modules (needed for BGLs and pipeline layouts)
    if (ok) {
        res->gaussian_shader_module = fs_effects_create_shader_module(device, FS_EFFECTS_GAUSSIAN_BLUR_WGSL, "FS Gaussian Shader");
        res->filter_shader_module = fs_effects_create_shader_module(device, FS_EFFECTS_FILTER_WGSL, "FS Filter Shader");
        res->filter_copy_shader_module = fs_effects_create_shader_module(device, FS_EFFECTS_FILTER_COPY_WGSL, "FS Filter Copy Shader");
        res->vert_shader_module = fs_effects_create_shader_module(device, FS_EFFECTS_VERT_WGSL, "FS Vert Shader");
        res->shadow_composite_shader_module = fs_effects_create_shader_module(device, FS_EFFECTS_SHADOW_COMPOSITE_WGSL, "FS Shadow Composite Shader");
        res->drop_shadow_shader_module = fs_effects_create_shader_module(device, FS_EFFECTS_DROP_SHADOW_WGSL, "FS Drop Shadow Shader");
        res->drop_shadow_c_shader_module = fs_effects_create_shader_module(device, FS_EFFECTS_DROP_SHADOW_C_WGSL, "FS DS Compute Shader");
        if (!res->gaussian_shader_module || !res->filter_shader_module || !res->filter_copy_shader_module ||
            !res->vert_shader_module || !res->shadow_composite_shader_module || !res->drop_shadow_shader_module ||
            !res->drop_shadow_c_shader_module) ok = false;
    }

    // 3. Create Gaussian blur BGL (layout, no texture dependency)
    if (ok) {
        WGPUBindGroupLayoutEntry blurEntries[4]; memset(blurEntries, 0, sizeof(blurEntries));
        blurEntries[0].binding = 0; blurEntries[0].visibility = WGPUShaderStage_Compute;
        blurEntries[0].texture.sampleType = WGPUTextureSampleType_Float; blurEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D; blurEntries[0].texture.multisampled = false;
        blurEntries[1].binding = 1; blurEntries[1].visibility = WGPUShaderStage_Compute;
        blurEntries[1].storageTexture.format = WGPUTextureFormat_RGBA8Unorm; blurEntries[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly; blurEntries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
        blurEntries[2].binding = 2; blurEntries[2].visibility = WGPUShaderStage_Compute; blurEntries[2].buffer.type = WGPUBufferBindingType_Uniform; blurEntries[2].buffer.hasDynamicOffset = false;
        blurEntries[3].binding = 3; blurEntries[3].visibility = WGPUShaderStage_Compute; blurEntries[3].buffer.type = WGPUBufferBindingType_ReadOnlyStorage; blurEntries[3].buffer.hasDynamicOffset = false;
        WGPUBindGroupLayoutDescriptor blurBGLDesc = { .nextInChain = NULL, .label = { .data = "FS Gaussian BGL", .length = 13 }, .entryCount = 4, .entries = blurEntries };
        res->gaussian_blur_bgl = wgpuDeviceCreateBindGroupLayout(device, &blurBGLDesc);
        if (!res->gaussian_blur_bgl) ok = false;
    }

    // 4. Create filter BGL (layout, no texture dependency)
    if (ok) {
        WGPUBindGroupLayoutEntry filterEntries[3]; memset(filterEntries, 0, sizeof(filterEntries));
        filterEntries[0].binding = 0; filterEntries[0].visibility = WGPUShaderStage_Compute;
        filterEntries[0].texture.sampleType = WGPUTextureSampleType_Float; filterEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D; filterEntries[0].texture.multisampled = false;
        filterEntries[1].binding = 1; filterEntries[1].visibility = WGPUShaderStage_Compute;
        filterEntries[1].storageTexture.format = WGPUTextureFormat_RGBA8Unorm; filterEntries[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly; filterEntries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
        filterEntries[2].binding = 2; filterEntries[2].visibility = WGPUShaderStage_Compute; filterEntries[2].buffer.type = WGPUBufferBindingType_Uniform; filterEntries[2].buffer.hasDynamicOffset = false;
        WGPUBindGroupLayoutDescriptor fBGLDesc = { .nextInChain = NULL, .label = { .data = "FS Filter BGL", .length = 12 }, .entryCount = 3, .entries = filterEntries };
        res->filter_bgl = wgpuDeviceCreateBindGroupLayout(device, &fBGLDesc);
        if (!res->filter_bgl) ok = false;
    }

    // 5. Create filter copy BGL (render, no texture dependency)
    if (ok) {
        WGPUBindGroupLayoutEntry fcEntries[2]; memset(fcEntries, 0, sizeof(fcEntries));
        fcEntries[0].binding = 0; fcEntries[0].visibility = WGPUShaderStage_Fragment;
        fcEntries[0].texture.sampleType = WGPUTextureSampleType_Float; fcEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D; fcEntries[0].texture.multisampled = false;
        fcEntries[1].binding = 1; fcEntries[1].visibility = WGPUShaderStage_Fragment; fcEntries[1].sampler.type = WGPUSamplerBindingType_Filtering;
        WGPUBindGroupLayoutDescriptor fcBGLDesc = { .nextInChain = NULL, .label = { .data = "FS Filter Copy BGL", .length = 17 }, .entryCount = 2, .entries = fcEntries };
        res->filter_copy_bgl = wgpuDeviceCreateBindGroupLayout(device, &fcBGLDesc);
        if (!res->filter_copy_bgl) ok = false;
    }

    // 6. Create shadow composite BGL (render, no texture dependency)
    if (ok) {
        WGPUBindGroupLayoutEntry scEntries[3]; memset(scEntries, 0, sizeof(scEntries));
        scEntries[0].binding = 0; scEntries[0].visibility = WGPUShaderStage_Fragment;
        scEntries[0].texture.sampleType = WGPUTextureSampleType_Float; scEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D; scEntries[0].texture.multisampled = false;
        scEntries[1].binding = 1; scEntries[1].visibility = WGPUShaderStage_Fragment; scEntries[1].sampler.type = WGPUSamplerBindingType_Filtering;
        scEntries[2].binding = 2; scEntries[2].visibility = WGPUShaderStage_Fragment; scEntries[2].buffer.type = WGPUBufferBindingType_Uniform; scEntries[2].buffer.hasDynamicOffset = false;
        WGPUBindGroupLayoutDescriptor scBGLDesc = { .nextInChain = NULL, .label = { .data = "FS Shadow Composite BGL", .length = 23 }, .entryCount = 3, .entries = scEntries };
        res->shadow_composite_bgl = wgpuDeviceCreateBindGroupLayout(device, &scBGLDesc);
        if (!res->shadow_composite_bgl) ok = false;
    }

    // 7. Create drop shadow BGL (render, no texture dependency)
    if (ok) {
        WGPUBindGroupLayoutEntry dsEntries[4]; memset(dsEntries, 0, sizeof(dsEntries));
        dsEntries[0].binding = 0; dsEntries[0].visibility = WGPUShaderStage_Fragment;
        dsEntries[0].texture.sampleType = WGPUTextureSampleType_Float; dsEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D; dsEntries[0].texture.multisampled = false;
        dsEntries[1].binding = 1; dsEntries[1].visibility = WGPUShaderStage_Fragment;
        dsEntries[1].texture.sampleType = WGPUTextureSampleType_Float; dsEntries[1].texture.viewDimension = WGPUTextureViewDimension_2D; dsEntries[1].texture.multisampled = false;
        dsEntries[2].binding = 2; dsEntries[2].visibility = WGPUShaderStage_Fragment; dsEntries[2].sampler.type = WGPUSamplerBindingType_Filtering;
        dsEntries[3].binding = 3; dsEntries[3].visibility = WGPUShaderStage_Fragment; dsEntries[3].buffer.type = WGPUBufferBindingType_Uniform; dsEntries[3].buffer.hasDynamicOffset = false;
        WGPUBindGroupLayoutDescriptor dsBGLDesc = { .nextInChain = NULL, .label = { .data = "FS Drop Shadow BGL", .length = 18 }, .entryCount = 4, .entries = dsEntries };
        res->drop_shadow_bgl = wgpuDeviceCreateBindGroupLayout(device, &dsBGLDesc);
        if (!res->drop_shadow_bgl) ok = false;
    }

    // 8. Create drop-shadow compute BGL (compute, no texture dependency)
    if (ok) {
        WGPUBindGroupLayoutEntry dsCEntries[4]; memset(dsCEntries, 0, sizeof(dsCEntries));
        dsCEntries[0].binding = 0; dsCEntries[0].visibility = WGPUShaderStage_Compute;
        dsCEntries[0].texture.sampleType = WGPUTextureSampleType_Float; dsCEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D; dsCEntries[0].texture.multisampled = false;
        dsCEntries[1].binding = 1; dsCEntries[1].visibility = WGPUShaderStage_Compute;
        dsCEntries[1].texture.sampleType = WGPUTextureSampleType_Float; dsCEntries[1].texture.viewDimension = WGPUTextureViewDimension_2D; dsCEntries[1].texture.multisampled = false;
        dsCEntries[2].binding = 2; dsCEntries[2].visibility = WGPUShaderStage_Compute;
        dsCEntries[2].storageTexture.format = WGPUTextureFormat_RGBA8Unorm; dsCEntries[2].storageTexture.access = WGPUStorageTextureAccess_WriteOnly; dsCEntries[2].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
        dsCEntries[3].binding = 3; dsCEntries[3].visibility = WGPUShaderStage_Compute;
        dsCEntries[3].buffer.type = WGPUBufferBindingType_Uniform; dsCEntries[3].buffer.hasDynamicOffset = false;
        WGPUBindGroupLayoutDescriptor dsCBGLDesc = { .nextInChain = NULL, .label = { .data = "FS DS Compute BGL", .length = 17 }, .entryCount = 4, .entries = dsCEntries };
        res->drop_shadow_c_bgl = wgpuDeviceCreateBindGroupLayout(device, &dsCBGLDesc);
        if (!res->drop_shadow_c_bgl) ok = false;
    }

    // 9. Create Gaussian blur pipeline layout
    if (ok) {
        WGPUPipelineLayoutDescriptor blurLayoutDesc = {
            .nextInChain = NULL, .label = { .data = "FS Gaussian Blur Layout", .length = 22 },
            .bindGroupLayoutCount = 1, .bindGroupLayouts = &res->gaussian_blur_bgl
        };
        res->gaussian_blur_pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &blurLayoutDesc);
        if (!res->gaussian_blur_pipeline_layout) ok = false;
    }

    // 10. Create filter pipeline layout
    if (ok) {
        WGPUPipelineLayoutDescriptor fLayoutDesc = {
            .nextInChain = NULL, .label = { .data = "FS Filter Layout", .length = 16 },
            .bindGroupLayoutCount = 1, .bindGroupLayouts = &res->filter_bgl
        };
        res->filter_pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &fLayoutDesc);
        if (!res->filter_pipeline_layout) ok = false;
    }

    // 11. Create filter copy pipeline layout
    if (ok) {
        WGPUPipelineLayoutDescriptor fcLayoutDesc = {
            .nextInChain = NULL, .label = { .data = "FS Filter Copy Layout", .length = 21 },
            .bindGroupLayoutCount = 1, .bindGroupLayouts = &res->filter_copy_bgl
        };
        res->filter_copy_pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &fcLayoutDesc);
        if (!res->filter_copy_pipeline_layout) ok = false;
    }

    // 12. Create shadow composite pipeline layout
    if (ok) {
        WGPUPipelineLayoutDescriptor scLayoutDesc = {
            .nextInChain = NULL, .label = { .data = "FS Shadow Composite Layout", .length = 26 },
            .bindGroupLayoutCount = 1, .bindGroupLayouts = &res->shadow_composite_bgl
        };
        res->shadow_composite_pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &scLayoutDesc);
        if (!res->shadow_composite_pipeline_layout) ok = false;
    }

    // 13. Create drop-shadow pipeline layout
    if (ok) {
        WGPUPipelineLayoutDescriptor dsLayoutDesc = {
            .nextInChain = NULL, .label = { .data = "FS Drop Shadow Layout", .length = 21 },
            .bindGroupLayoutCount = 1, .bindGroupLayouts = &res->drop_shadow_bgl
        };
        res->drop_shadow_pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &dsLayoutDesc);
        if (!res->drop_shadow_pipeline_layout) ok = false;
    }

    // 14. Create drop-shadow compute pipeline layout
    if (ok) {
        WGPUPipelineLayoutDescriptor dsCLayoutDesc = {
            .nextInChain = NULL, .label = { .data = "FS DS Compute Layout", .length = 19 },
            .bindGroupLayoutCount = 1, .bindGroupLayouts = &res->drop_shadow_c_bgl
        };
        res->drop_shadow_c_pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &dsCLayoutDesc);
        if (!res->drop_shadow_c_pipeline_layout) ok = false;
    }

    // 15. Create filter copy pipeline (render — only needs BGL + shaders, no texture dep)
    if (ok) {
        WGPUColorTargetState colorTarget = { .nextInChain = NULL, .format = WGPUTextureFormat_RGBA8Unorm, .blend = NULL, .writeMask = WGPUColorWriteMask_All };
        WGPUFragmentState fragState = { .nextInChain = NULL, .module = res->filter_copy_shader_module, .entryPoint = { .data = "filter_copy_fs_main", .length = 19 }, .constantCount = 0, .constants = NULL, .targetCount = 1, .targets = &colorTarget };
        WGPURenderPipelineDescriptor rpDesc = {
            .nextInChain = NULL, .label = { .data = "FS Filter Copy", .length = 13 }, .layout = res->filter_copy_pipeline_layout,
            .vertex = { .module = res->vert_shader_module, .entryPoint = { .data = "fs_vs_main", .length = 10 }, .constantCount = 0, .constants = NULL, .buffers = NULL, .bufferCount = 0 },
            .primitive = { .topology = WGPUPrimitiveTopology_TriangleList, .stripIndexFormat = WGPUIndexFormat_Undefined, .frontFace = WGPUFrontFace_CCW, .cullMode = WGPUCullMode_None },
            .depthStencil = NULL, .multisample = { .count = 1, .mask = 0xFFFFFFFF, .alphaToCoverageEnabled = false },
            .fragment = &fragState
        };
        res->filter_copy_pipeline = wgpuDeviceCreateRenderPipeline(device, &rpDesc);
        if (!res->filter_copy_pipeline) ok = false;
    }

    // 16. Create shadow sampler (needed by presentation_scene_bg and filter_copy_bg)
    if (ok) {
        WGPUSamplerDescriptor sampDesc = {
            .nextInChain = NULL, .label = { .data = "FS Shadow Sampler", .length = 16 },
            .addressModeU = WGPUAddressMode_ClampToEdge, .addressModeV = WGPUAddressMode_ClampToEdge, .addressModeW = WGPUAddressMode_ClampToEdge,
            .magFilter = WGPUFilterMode_Linear, .minFilter = WGPUFilterMode_Linear, .mipmapFilter = WGPUMipmapFilterMode_Linear,
            .lodMinClamp = 0.0f, .lodMaxClamp = 1.0f, .maxAnisotropy = 1
        };
        res->shadow_sampler = wgpuDeviceCreateSampler(device, &sampDesc);
        if (!res->shadow_sampler) ok = false;
    }

    // 17. Create presentation_scene_bg (samples scene_view -> render to canvas)
    if (ok && res->filter_copy_bgl && res->shadow_sampler && res->scene_view) {
        WGPUTextureViewDescriptor presSceneViewDesc = {
            .nextInChain = NULL, .format = WGPUTextureFormat_RGBA8Unorm,
            .dimension = WGPUTextureViewDimension_2D,
            .baseMipLevel = 0, .mipLevelCount = 1,
            .baseArrayLayer = 0, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All
        };
        WGPUTextureView presSceneTexView = wgpuTextureCreateView(res->scene_texture, &presSceneViewDesc);
        if (presSceneTexView) {
            WGPUBindGroupEntry presEntry[2];
            memset(presEntry, 0, sizeof(presEntry));
            presEntry[0].binding = 0; presEntry[0].textureView = presSceneTexView;
            presEntry[1].binding = 1; presEntry[1].sampler = res->shadow_sampler;
            WGPUBindGroupDescriptor presBGDesc = {
                .nextInChain = NULL, .layout = res->filter_copy_bgl, .entryCount = 2, .entries = presEntry
            };
            res->presentation_scene_bg = wgpuDeviceCreateBindGroup(device, &presBGDesc);
            wgpuTextureViewRelease(presSceneTexView);
            if (!res->presentation_scene_bg) ok = false;
        } else {
            ok = false;
        }
    }

    // 18. filter_copy_bg + filter_copy_bg_back are deferred to Phase 2
    //    (they depend on ping_pong_view_a which doesn't exist yet)

    if (!ok) { fs_effects_destroy(core); return false; }
    res->enabled = true;
    fs_core_set_effects_resources(core, res);
    return true;
}

bool fs_effects_ensure_filter_textures(FS_Core* core) {
    if (!core || !core->device) return false;
    struct FS_EffectResources* res = fs_core_get_effects_resources(core);
    if (!res) return false;
    if (res->filter_textures_ready) return true;

    WGPUDevice device = core->device;
    bool ok = true;

    // ============================
    // Phase 2: Lazy-created filter resources (ping-pong textures, pipelines, bind groups)
    // ============================

    // P1. Create ping-pong textures (RGBA8Unorm for STORAGE_BINDING support)
    if (ok) {
        WGPUTextureDescriptor texDesc = {
            .nextInChain = NULL,
            .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst,
            .dimension = WGPUTextureDimension_2D, .size = { res->width, res->height, 1 },
            .format = WGPUTextureFormat_RGBA8Unorm, .mipLevelCount = 1, .sampleCount = 1, .viewFormatCount = 0, .viewFormats = NULL
        };
        texDesc.label.data = "FS PingPong A"; texDesc.label.length = 12;
        res->ping_pong_texture_a = wgpuDeviceCreateTexture(device, &texDesc);
        texDesc.label.data = "FS PingPong B"; texDesc.label.length = 12;
        res->ping_pong_texture_b = wgpuDeviceCreateTexture(device, &texDesc);
        if (!res->ping_pong_texture_a || !res->ping_pong_texture_b) ok = false;
    }

    // P2. Create ping-pong views (RGBA8Unorm)
    if (ok) {
        WGPUTextureViewDescriptor viewDesc = {
            .nextInChain = NULL, .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D,
            .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All
        };
        if (res->ping_pong_texture_a) res->ping_pong_view_a = wgpuTextureCreateView(res->ping_pong_texture_a, &viewDesc);
        if (res->ping_pong_texture_b) res->ping_pong_view_b = wgpuTextureCreateView(res->ping_pong_texture_b, &viewDesc);
        if (!res->ping_pong_view_a || !res->ping_pong_view_b) ok = false;
    }

    // P3. Create Gaussian kernel buffer
    if (ok) {
        WGPUBufferDescriptor kbDesc = { .nextInChain = NULL, .label = { .data = "FS Gaussian Kernel", .length = 17 }, .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, .size = 252, .mappedAtCreation = false };
        res->gaussian_kernel_buffer = wgpuDeviceCreateBuffer(device, &kbDesc);
        res->gaussian_kernel_buffer_size = 252;
        if (!res->gaussian_kernel_buffer) ok = false;
    }

    // P4. Create Gaussian uniform buffer
    if (ok) {
        WGPUBufferDescriptor ubDesc = { .nextInChain = NULL, .label = { .data = "FS Gaussian Uniform", .length = 18 }, .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, .size = 16, .mappedAtCreation = false };
        res->gaussian_uniform_buffer = wgpuDeviceCreateBuffer(device, &ubDesc);
        if (!res->gaussian_uniform_buffer) ok = false;
    }

    // P5. Create Gaussian H pipeline
    if (ok && res->gaussian_blur_pipeline_layout && res->gaussian_shader_module) {
        WGPUComputePipelineDescriptor cpDesc = {
            .nextInChain = NULL, .label = { .data = "FS Gaussian Blur H", .length = 18 }, .layout = res->gaussian_blur_pipeline_layout,
            .compute = { .module = res->gaussian_shader_module, .entryPoint = { .data = "gaussian_blur_h", .length = 15 }, .constantCount = 0, .constants = NULL }
        };
        res->gaussian_blur_h_pipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);
        if (!res->gaussian_blur_h_pipeline) ok = false;
    }

    // P6. Create Gaussian V pipeline
    if (ok && res->gaussian_blur_pipeline_layout && res->gaussian_shader_module) {
        WGPUComputePipelineDescriptor cpDesc = {
            .nextInChain = NULL, .label = { .data = "FS Gaussian Blur V", .length = 18 }, .layout = res->gaussian_blur_pipeline_layout,
            .compute = { .module = res->gaussian_shader_module, .entryPoint = { .data = "gaussian_blur_v", .length = 15 }, .constantCount = 0, .constants = NULL }
        };
        res->gaussian_blur_v_pipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);
        if (!res->gaussian_blur_v_pipeline) ok = false;
    }

    // P7. Create 4 Gaussian bind groups (depend on ping_pong views + gaussian buffers)
    if (ok && res->gaussian_blur_bgl && res->gaussian_uniform_buffer && res->gaussian_kernel_buffer) {
        WGPUTextureViewDescriptor storageViewDesc = { .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D, .mipLevelCount = 1, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All };
        WGPUTextureView storageViewA = res->ping_pong_texture_a ? wgpuTextureCreateView(res->ping_pong_texture_a, &storageViewDesc) : NULL;
        WGPUTextureView storageViewB = res->ping_pong_texture_b ? wgpuTextureCreateView(res->ping_pong_texture_b, &storageViewDesc) : NULL;
        WGPUTextureViewDescriptor readViewDesc = { .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D, .mipLevelCount = 1, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All };
        WGPUTextureView readViewA = res->ping_pong_texture_a ? wgpuTextureCreateView(res->ping_pong_texture_a, &readViewDesc) : NULL;
        WGPUTextureView readViewB = res->ping_pong_texture_b ? wgpuTextureCreateView(res->ping_pong_texture_b, &readViewDesc) : NULL;
        WGPUBindGroupEntry entries[4]; memset(entries, 0, sizeof(entries));
        entries[2].binding = 2; entries[2].buffer = res->gaussian_uniform_buffer; entries[2].offset = 0; entries[2].size = 16;
        entries[3].binding = 3; entries[3].buffer = res->gaussian_kernel_buffer; entries[3].offset = 0; entries[3].size = res->gaussian_kernel_buffer_size;
        WGPUBindGroupDescriptor bgDesc = { .nextInChain = NULL, .layout = res->gaussian_blur_bgl, .entryCount = 4, .entries = entries };
        entries[0].binding = 0; entries[0].textureView = readViewA; entries[1].binding = 1; entries[1].textureView = storageViewB;
        res->gaussian_blur_h_bg_a = wgpuDeviceCreateBindGroup(device, &bgDesc);
        entries[0].binding = 0; entries[0].textureView = readViewB; entries[1].binding = 1; entries[1].textureView = storageViewA;
        res->gaussian_blur_h_bg_b = wgpuDeviceCreateBindGroup(device, &bgDesc);
        entries[0].binding = 0; entries[0].textureView = readViewB; entries[1].binding = 1; entries[1].textureView = storageViewA;
        res->gaussian_blur_v_bg_a = wgpuDeviceCreateBindGroup(device, &bgDesc);
        entries[0].binding = 0; entries[0].textureView = readViewA; entries[1].binding = 1; entries[1].textureView = storageViewB;
        res->gaussian_blur_v_bg_b = wgpuDeviceCreateBindGroup(device, &bgDesc);
        if (!res->gaussian_blur_h_bg_a || !res->gaussian_blur_h_bg_b || !res->gaussian_blur_v_bg_a || !res->gaussian_blur_v_bg_b) ok = false;
        if (readViewA) wgpuTextureViewRelease(readViewA); if (readViewB) wgpuTextureViewRelease(readViewB);
        if (storageViewA) wgpuTextureViewRelease(storageViewA); if (storageViewB) wgpuTextureViewRelease(storageViewB);
    }

    // P8. Create filter pipeline (compute)
    if (ok && res->filter_pipeline_layout && res->filter_shader_module) {
        WGPUComputePipelineDescriptor fCpDesc = {
            .nextInChain = NULL, .label = { .data = "FS Filter Compute", .length = 15 }, .layout = res->filter_pipeline_layout,
            .compute = { .module = res->filter_shader_module, .entryPoint = { .data = "filter_main", .length = 11 }, .constantCount = 0, .constants = NULL }
        };
        res->filter_pipeline = wgpuDeviceCreateComputePipeline(device, &fCpDesc);
        if (!res->filter_pipeline) ok = false;
    }

    // P9. Create filter uniform buffer
    if (ok) {
        WGPUBufferDescriptor fbDesc = { .nextInChain = NULL, .label = { .data = "FS Filter Uniform", .length = 16 }, .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, .size = sizeof(FS_FilterUniforms), .mappedAtCreation = false };
        res->filter_uniform_buffer = wgpuDeviceCreateBuffer(device, &fbDesc);
        if (!res->filter_uniform_buffer) ok = false;
    }

    // P10. Create filter bind groups (depend on ping_pong views)
    if (ok && res->filter_bgl && res->filter_uniform_buffer) {
        WGPUTextureViewDescriptor rvDesc = { .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D, .mipLevelCount = 1, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All };
        WGPUTextureViewDescriptor svDesc = { .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D, .mipLevelCount = 1, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All };
        WGPUTextureView srcTexA = res->ping_pong_texture_a ? wgpuTextureCreateView(res->ping_pong_texture_a, &rvDesc) : NULL;
        WGPUTextureView srcTexB = res->ping_pong_texture_b ? wgpuTextureCreateView(res->ping_pong_texture_b, &rvDesc) : NULL;
        WGPUTextureView dstTexA = res->ping_pong_texture_a ? wgpuTextureCreateView(res->ping_pong_texture_a, &svDesc) : NULL;
        WGPUTextureView dstTexB = res->ping_pong_texture_b ? wgpuTextureCreateView(res->ping_pong_texture_b, &svDesc) : NULL;
        WGPUBindGroupEntry fEntries[3]; memset(fEntries, 0, sizeof(fEntries));
        fEntries[2].binding = 2; fEntries[2].buffer = res->filter_uniform_buffer; fEntries[2].offset = 0; fEntries[2].size = sizeof(FS_FilterUniforms);
        WGPUBindGroupDescriptor fBGDesc = { .nextInChain = NULL, .layout = res->filter_bgl, .entryCount = 3, .entries = fEntries };
        fEntries[0].binding = 0; fEntries[0].textureView = srcTexA; fEntries[1].binding = 1; fEntries[1].textureView = dstTexB;
        res->filter_bg_a = wgpuDeviceCreateBindGroup(device, &fBGDesc);
        fEntries[0].binding = 0; fEntries[0].textureView = srcTexB; fEntries[1].binding = 1; fEntries[1].textureView = dstTexA;
        res->filter_bg_b = wgpuDeviceCreateBindGroup(device, &fBGDesc);
        if (!res->filter_bg_a || !res->filter_bg_b) ok = false;
        if (srcTexA) wgpuTextureViewRelease(srcTexA); if (srcTexB) wgpuTextureViewRelease(srcTexB);
        if (dstTexA) wgpuTextureViewRelease(dstTexA); if (dstTexB) wgpuTextureViewRelease(dstTexB);
    }

    // P11. Create shadow uniform buffer
    if (ok) {
        WGPUBufferDescriptor suDesc = { .nextInChain = NULL, .label = { .data = "FS Shadow Uniform", .length = 16 }, .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, .size = sizeof(FS_ShadowUniforms), .mappedAtCreation = false };
        res->shadow_uniform_buffer = wgpuDeviceCreateBuffer(device, &suDesc);
        if (!res->shadow_uniform_buffer) ok = false;
    }

    // P12. Create shadow composite pipeline (render)
    if (ok && res->shadow_composite_pipeline_layout && res->shadow_composite_shader_module && res->vert_shader_module) {
        WGPUBlendState scBlend = {
            .color = { .operation = WGPUBlendOperation_Add, .srcFactor = WGPUBlendFactor_SrcAlpha, .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha },
            .alpha = { .operation = WGPUBlendOperation_Add, .srcFactor = WGPUBlendFactor_One, .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha }
        };
        WGPUColorTargetState scColorTarget = { .nextInChain = NULL, .format = WGPUTextureFormat_RGBA8Unorm, .blend = &scBlend, .writeMask = WGPUColorWriteMask_All };
        WGPUFragmentState scFragState = { .nextInChain = NULL, .module = res->shadow_composite_shader_module, .entryPoint = { .data = "shadow_fs_main", .length = 14 }, .constantCount = 0, .constants = NULL, .targetCount = 1, .targets = &scColorTarget };
        WGPURenderPipelineDescriptor scRpDesc = {
            .nextInChain = NULL, .label = { .data = "FS Shadow Composite", .length = 19 }, .layout = res->shadow_composite_pipeline_layout,
            .vertex = { .module = res->vert_shader_module, .entryPoint = { .data = "fs_vs_main", .length = 10 }, .constantCount = 0, .constants = NULL, .buffers = NULL, .bufferCount = 0 },
            .primitive = { .topology = WGPUPrimitiveTopology_TriangleList, .stripIndexFormat = WGPUIndexFormat_Undefined, .frontFace = WGPUFrontFace_CCW, .cullMode = WGPUCullMode_None },
            .depthStencil = NULL, .multisample = { .count = 1, .mask = 0xFFFFFFFF, .alphaToCoverageEnabled = false },
            .fragment = &scFragState
        };
        res->shadow_composite_pipeline = wgpuDeviceCreateRenderPipeline(device, &scRpDesc);
        if (!res->shadow_composite_pipeline) ok = false;
    }

    // P13. Create shadow composite bind group (depends on ping_pong_view_a + shadow_uniform)
    if (ok && res->shadow_composite_bgl && res->shadow_sampler && res->shadow_uniform_buffer && res->ping_pong_view_a) {
        WGPUBindGroupEntry scBGEntries[3]; memset(scBGEntries, 0, sizeof(scBGEntries));
        scBGEntries[0].binding = 0; scBGEntries[0].textureView = res->ping_pong_view_a;
        scBGEntries[1].binding = 1; scBGEntries[1].sampler = res->shadow_sampler;
        scBGEntries[2].binding = 2; scBGEntries[2].buffer = res->shadow_uniform_buffer; scBGEntries[2].offset = 0; scBGEntries[2].size = sizeof(FS_ShadowUniforms);
        WGPUBindGroupDescriptor scBGDesc = { .nextInChain = NULL, .layout = res->shadow_composite_bgl, .entryCount = 3, .entries = scBGEntries };
        res->shadow_composite_bg = wgpuDeviceCreateBindGroup(device, &scBGDesc);
        if (!res->shadow_composite_bg) ok = false;
    }

    // P14. Create drop-shadow pipeline (render)
    if (ok && res->drop_shadow_pipeline_layout && res->drop_shadow_shader_module && res->vert_shader_module) {
        WGPUColorTargetState dsColorTarget = { .nextInChain = NULL, .format = WGPUTextureFormat_RGBA8Unorm, .blend = NULL, .writeMask = WGPUColorWriteMask_All };
        WGPUFragmentState dsFragState = { .nextInChain = NULL, .module = res->drop_shadow_shader_module, .entryPoint = { .data = "drop_shadow_fs_main", .length = 19 }, .constantCount = 0, .constants = NULL, .targetCount = 1, .targets = &dsColorTarget };
        WGPURenderPipelineDescriptor dsRpDesc = {
            .nextInChain = NULL, .label = { .data = "FS Drop Shadow", .length = 14 }, .layout = res->drop_shadow_pipeline_layout,
            .vertex = { .module = res->vert_shader_module, .entryPoint = { .data = "fs_vs_main", .length = 10 }, .constantCount = 0, .constants = NULL, .buffers = NULL, .bufferCount = 0 },
            .primitive = { .topology = WGPUPrimitiveTopology_TriangleList, .stripIndexFormat = WGPUIndexFormat_Undefined, .frontFace = WGPUFrontFace_CCW, .cullMode = WGPUCullMode_None },
            .depthStencil = NULL, .multisample = { .count = 1, .mask = 0xFFFFFFFF, .alphaToCoverageEnabled = false },
            .fragment = &dsFragState
        };
        res->drop_shadow_pipeline = wgpuDeviceCreateRenderPipeline(device, &dsRpDesc);
        if (!res->drop_shadow_pipeline) ok = false;
    }

    // P15. Create drop-shadow bind group (depends on ping_pong views + shadow_uniform)
    if (ok && res->drop_shadow_bgl && res->shadow_sampler && res->shadow_uniform_buffer) {
        WGPUTextureViewDescriptor dsReadDesc = { .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D, .mipLevelCount = 1, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All };
        WGPUTextureView dsViewA = res->ping_pong_texture_a ? wgpuTextureCreateView(res->ping_pong_texture_a, &dsReadDesc) : NULL;
        WGPUTextureView dsViewB = res->ping_pong_texture_b ? wgpuTextureCreateView(res->ping_pong_texture_b, &dsReadDesc) : NULL;
        WGPUBindGroupEntry dsBGEntries[4]; memset(dsBGEntries, 0, sizeof(dsBGEntries));
        dsBGEntries[0].binding = 0; dsBGEntries[0].textureView = dsViewA;
        dsBGEntries[1].binding = 1; dsBGEntries[1].textureView = dsViewB;
        dsBGEntries[2].binding = 2; dsBGEntries[2].sampler = res->shadow_sampler;
        dsBGEntries[3].binding = 3; dsBGEntries[3].buffer = res->shadow_uniform_buffer; dsBGEntries[3].offset = 0; dsBGEntries[3].size = sizeof(FS_ShadowUniforms);
        WGPUBindGroupDescriptor dsBGDesc = { .nextInChain = NULL, .layout = res->drop_shadow_bgl, .entryCount = 4, .entries = dsBGEntries };
        res->drop_shadow_bg = wgpuDeviceCreateBindGroup(device, &dsBGDesc);
        if (!res->drop_shadow_bg) ok = false;
        if (dsViewA) wgpuTextureViewRelease(dsViewA); if (dsViewB) wgpuTextureViewRelease(dsViewB);
    }

    // P16. Create shadow_composite_texture (dedicated output for drop-shadow compute)
    if (ok) {
        WGPUTextureDescriptor scTexDesc = {
            .nextInChain = NULL,
            .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst,
            .dimension = WGPUTextureDimension_2D, .size = { res->width, res->height, 1 },
            .format = WGPUTextureFormat_RGBA8Unorm, .mipLevelCount = 1, .sampleCount = 1, .viewFormatCount = 0, .viewFormats = NULL
        };
        scTexDesc.label.data = "FS Shadow Composite"; scTexDesc.label.length = 16;
        res->shadow_composite_texture = wgpuDeviceCreateTexture(device, &scTexDesc);
        if (res->shadow_composite_texture) {
            WGPUTextureViewDescriptor scViewDesc = {
                .nextInChain = NULL, .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D,
                .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All
            };
            res->shadow_composite_view = wgpuTextureCreateView(res->shadow_composite_texture, &scViewDesc);
            if (!res->shadow_composite_view) ok = false;
        } else {
            ok = false;
        }
    }

    // P17. Create drop-shadow compute pipeline (ALL-COMPUTE)
    if (ok && res->drop_shadow_c_pipeline_layout && res->drop_shadow_c_shader_module) {
        WGPUComputePipelineDescriptor dsCCpDesc = {
            .nextInChain = NULL, .label = { .data = "FS DS Compute", .length = 14 }, .layout = res->drop_shadow_c_pipeline_layout,
            .compute = { .module = res->drop_shadow_c_shader_module, .entryPoint = { .data = "drop_shadow_c_main", .length = 18 }, .constantCount = 0, .constants = NULL }
        };
        res->drop_shadow_c_pipeline = wgpuDeviceCreateComputePipeline(device, &dsCCpDesc);
        if (!res->drop_shadow_c_pipeline) ok = false;
    }

    // P18. Create drop-shadow compute bind group
    if (ok && res->drop_shadow_c_bgl && res->shadow_uniform_buffer) {
        WGPUTextureViewDescriptor dsCReadDesc = { .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D, .mipLevelCount = 1, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All };
        WGPUTextureView dsCViewA = res->ping_pong_texture_a ? wgpuTextureCreateView(res->ping_pong_texture_a, &dsCReadDesc) : NULL;
        WGPUTextureView dsCViewB = res->ping_pong_texture_b ? wgpuTextureCreateView(res->ping_pong_texture_b, &dsCReadDesc) : NULL;
        WGPUTextureViewDescriptor dsCStorageDesc = { .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D, .mipLevelCount = 1, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All };
        WGPUTextureView dsCStorageView = res->shadow_composite_texture ? wgpuTextureCreateView(res->shadow_composite_texture, &dsCStorageDesc) : NULL;
        if (dsCViewA && dsCViewB && dsCStorageView) {
            WGPUBindGroupEntry dsCEntries[4];
            memset(dsCEntries, 0, sizeof(dsCEntries));
            dsCEntries[0].binding = 0; dsCEntries[0].textureView = dsCViewA;
            dsCEntries[1].binding = 1; dsCEntries[1].textureView = dsCViewB;
            dsCEntries[2].binding = 2; dsCEntries[2].textureView = dsCStorageView;
            dsCEntries[3].binding = 3; dsCEntries[3].buffer = res->shadow_uniform_buffer;
            dsCEntries[3].offset = 0; dsCEntries[3].size = sizeof(FS_ShadowUniforms);
            WGPUBindGroupDescriptor dsCBGDesc = {
                .nextInChain = NULL, .layout = res->drop_shadow_c_bgl,
                .entryCount = 4, .entries = dsCEntries
            };
            res->drop_shadow_c_bg = wgpuDeviceCreateBindGroup(device, &dsCBGDesc);
            if (!res->drop_shadow_c_bg) ok = false;
        } else {
            ok = false;
        }
        if (dsCViewA) wgpuTextureViewRelease(dsCViewA);
        if (dsCViewB) wgpuTextureViewRelease(dsCViewB);
        if (dsCStorageView) wgpuTextureViewRelease(dsCStorageView);
    }

    // P19. Recreate filter_copy_bg + filter_copy_bg_back with valid ping_pong_view_a
    if (ok && res->filter_copy_bgl && res->shadow_sampler && res->ping_pong_view_a) {
        // Release Phase-1 bind groups that were created with NULL ping_pong_view_a
        fs_effects_release_bg(res->filter_copy_bg); res->filter_copy_bg = NULL;
        fs_effects_release_bg(res->filter_copy_bg_back); res->filter_copy_bg_back = NULL;
        WGPUBindGroupEntry fcA[2]; memset(fcA, 0, sizeof(fcA));
        fcA[0].binding = 0; fcA[0].textureView = res->ping_pong_view_a;
        fcA[1].binding = 1; fcA[1].sampler = res->shadow_sampler;
        WGPUBindGroupDescriptor fcBGDesc = { .nextInChain = NULL, .layout = res->filter_copy_bgl, .entryCount = 2, .entries = fcA };
        res->filter_copy_bg = wgpuDeviceCreateBindGroup(device, &fcBGDesc);
        res->filter_copy_bg_back = wgpuDeviceCreateBindGroup(device, &fcBGDesc);
        if (!res->filter_copy_bg || !res->filter_copy_bg_back) ok = false;
    }

    if (!ok) {
        // Phase 2 failed — clean up any partially-created resources
        // but keep Phase 1 resources intact (scene_texture, BGLs, layouts, etc.)
        fs_effects_release_bg(res->gaussian_blur_h_bg_a); res->gaussian_blur_h_bg_a = NULL;
        fs_effects_release_bg(res->gaussian_blur_h_bg_b); res->gaussian_blur_h_bg_b = NULL;
        fs_effects_release_bg(res->gaussian_blur_v_bg_a); res->gaussian_blur_v_bg_a = NULL;
        fs_effects_release_bg(res->gaussian_blur_v_bg_b); res->gaussian_blur_v_bg_b = NULL;
        fs_effects_release_bg(res->filter_bg_a); res->filter_bg_a = NULL;
        fs_effects_release_bg(res->filter_bg_b); res->filter_bg_b = NULL;
        fs_effects_release_bg(res->shadow_composite_bg); res->shadow_composite_bg = NULL;
        fs_effects_release_bg(res->drop_shadow_bg); res->drop_shadow_bg = NULL;
        fs_effects_release_bg(res->drop_shadow_c_bg); res->drop_shadow_c_bg = NULL;
        fs_effects_release_compute_pipeline(res->gaussian_blur_h_pipeline); res->gaussian_blur_h_pipeline = NULL;
        fs_effects_release_compute_pipeline(res->gaussian_blur_v_pipeline); res->gaussian_blur_v_pipeline = NULL;
        fs_effects_release_compute_pipeline(res->filter_pipeline); res->filter_pipeline = NULL;
        fs_effects_release_render_pipeline(res->shadow_composite_pipeline); res->shadow_composite_pipeline = NULL;
        fs_effects_release_render_pipeline(res->drop_shadow_pipeline); res->drop_shadow_pipeline = NULL;
        fs_effects_release_compute_pipeline(res->drop_shadow_c_pipeline); res->drop_shadow_c_pipeline = NULL;
        if (res->gaussian_kernel_buffer) { wgpuBufferRelease(res->gaussian_kernel_buffer); res->gaussian_kernel_buffer = NULL; }
        if (res->gaussian_uniform_buffer) { wgpuBufferRelease(res->gaussian_uniform_buffer); res->gaussian_uniform_buffer = NULL; }
        if (res->filter_uniform_buffer) { wgpuBufferRelease(res->filter_uniform_buffer); res->filter_uniform_buffer = NULL; }
        if (res->shadow_uniform_buffer) { wgpuBufferRelease(res->shadow_uniform_buffer); res->shadow_uniform_buffer = NULL; }
        if (res->ping_pong_view_a) { wgpuTextureViewRelease(res->ping_pong_view_a); res->ping_pong_view_a = NULL; }
        if (res->ping_pong_view_b) { wgpuTextureViewRelease(res->ping_pong_view_b); res->ping_pong_view_b = NULL; }
        if (res->ping_pong_texture_a) { wgpuTextureRelease(res->ping_pong_texture_a); res->ping_pong_texture_a = NULL; }
        if (res->ping_pong_texture_b) { wgpuTextureRelease(res->ping_pong_texture_b); res->ping_pong_texture_b = NULL; }
        if (res->shadow_composite_view) { wgpuTextureViewRelease(res->shadow_composite_view); res->shadow_composite_view = NULL; }
        if (res->shadow_composite_texture) { wgpuTextureRelease(res->shadow_composite_texture); res->shadow_composite_texture = NULL; }
        return false;
    }

    res->filter_textures_ready = true;
    return true;
}

bool fs_effects_resize(FS_Core* core, uint32_t width, uint32_t height) {
    if (!core) return false;
    struct FS_EffectResources* res = fs_core_get_effects_resources(core);
    if (!res) return false;
    if (res->width == width && res->height == height && res->enabled) return true;

    // Always release and recreate scene texture/view
    if (res->scene_view) { wgpuTextureViewRelease(res->scene_view); res->scene_view = NULL; }
    if (res->scene_texture) { wgpuTextureRelease(res->scene_texture); res->scene_texture = NULL; }
    if (res->presentation_pipeline) { fs_effects_release_bg(res->presentation_scene_bg); res->presentation_scene_bg = NULL; }

    // If filter textures have been lazy-initialized, release them too
    if (res->filter_textures_ready) {
        if (res->ping_pong_view_a) { wgpuTextureViewRelease(res->ping_pong_view_a); res->ping_pong_view_a = NULL; }
        if (res->ping_pong_view_b) { wgpuTextureViewRelease(res->ping_pong_view_b); res->ping_pong_view_b = NULL; }
        if (res->ping_pong_texture_a) { wgpuTextureRelease(res->ping_pong_texture_a); res->ping_pong_texture_a = NULL; }
        if (res->ping_pong_texture_b) { wgpuTextureRelease(res->ping_pong_texture_b); res->ping_pong_texture_b = NULL; }
        if (res->shadow_composite_view) { wgpuTextureViewRelease(res->shadow_composite_view); res->shadow_composite_view = NULL; }
        if (res->shadow_composite_texture) { wgpuTextureRelease(res->shadow_composite_texture); res->shadow_composite_texture = NULL; }
        fs_effects_release_bg(res->gaussian_blur_h_bg_a); res->gaussian_blur_h_bg_a = NULL;
        fs_effects_release_bg(res->gaussian_blur_h_bg_b); res->gaussian_blur_h_bg_b = NULL;
        fs_effects_release_bg(res->gaussian_blur_v_bg_a); res->gaussian_blur_v_bg_a = NULL;
        fs_effects_release_bg(res->gaussian_blur_v_bg_b); res->gaussian_blur_v_bg_b = NULL;
        fs_effects_release_bg(res->filter_bg_a); res->filter_bg_a = NULL;
        fs_effects_release_bg(res->filter_bg_b); res->filter_bg_b = NULL;
        fs_effects_release_bg(res->filter_copy_bg); res->filter_copy_bg = NULL;
        fs_effects_release_bg(res->filter_copy_bg_back); res->filter_copy_bg_back = NULL;
        fs_effects_release_bg(res->drop_shadow_bg); res->drop_shadow_bg = NULL;
        fs_effects_release_bg(res->shadow_composite_bg); res->shadow_composite_bg = NULL;
        fs_effects_release_bg(res->drop_shadow_c_bg); res->drop_shadow_c_bg = NULL;
    }

    res->width = width; res->height = height;
    WGPUDevice device = core->device;

    // Recreate ping-pong textures only if they've been lazy-initialized
    if (res->filter_textures_ready) {
        WGPUTextureDescriptor texDesc = {
            .nextInChain = NULL,
            .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst,
            .dimension = WGPUTextureDimension_2D, .size = { width, height, 1 }, .format = WGPUTextureFormat_RGBA8Unorm,
            .mipLevelCount = 1, .sampleCount = 1, .viewFormatCount = 0, .viewFormats = NULL
        };
        texDesc.label.data = "FS PingPong A"; texDesc.label.length = 12;
        res->ping_pong_texture_a = wgpuDeviceCreateTexture(device, &texDesc);
        texDesc.label.data = "FS PingPong B"; texDesc.label.length = 12;
        res->ping_pong_texture_b = wgpuDeviceCreateTexture(device, &texDesc);
        if (!res->ping_pong_texture_a || !res->ping_pong_texture_b) return false;
        WGPUTextureViewDescriptor viewDesc = { .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D, .mipLevelCount = 1, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All };
        res->ping_pong_view_a = wgpuTextureCreateView(res->ping_pong_texture_a, &viewDesc);
        res->ping_pong_view_b = wgpuTextureCreateView(res->ping_pong_texture_b, &viewDesc);
        if (!res->ping_pong_view_a || !res->ping_pong_view_b) return false;
    }

    // Always recreate scene texture (RGBA8Unorm, RenderAttachment + Storage + CopySrc/Dst)
    {
        WGPUTextureDescriptor sceneDesc = {
            .nextInChain = NULL,
            .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst,
            .dimension = WGPUTextureDimension_2D, .size = { width, height, 1 },
            .format = WGPUTextureFormat_RGBA8Unorm, .mipLevelCount = 1, .sampleCount = 1, .viewFormatCount = 0, .viewFormats = NULL
        };
        sceneDesc.label.data = "FS Scene"; sceneDesc.label.length = 8;
        res->scene_texture = wgpuDeviceCreateTexture(device, &sceneDesc);
        if (!res->scene_texture) return false;
        WGPUTextureViewDescriptor sceneViewDesc = {
            .nextInChain = NULL, .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D,
            .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All
        };
        res->scene_view = wgpuTextureCreateView(res->scene_texture, &sceneViewDesc);
        if (!res->scene_view) return false;
    }

    // Recreate filter-dependent resources only if they've been lazy-initialized
    if (res->filter_textures_ready) {
        // Recreate shadow_composite_texture
        {
            WGPUTextureDescriptor scTexDesc = {
                .nextInChain = NULL,
                .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst,
                .dimension = WGPUTextureDimension_2D, .size = { width, height, 1 },
                .format = WGPUTextureFormat_RGBA8Unorm, .mipLevelCount = 1, .sampleCount = 1, .viewFormatCount = 0, .viewFormats = NULL
            };
            scTexDesc.label.data = "FS Shadow Composite"; scTexDesc.label.length = 16;
            res->shadow_composite_texture = wgpuDeviceCreateTexture(device, &scTexDesc);
            if (!res->shadow_composite_texture) return false;
            WGPUTextureViewDescriptor scViewDesc = {
                .nextInChain = NULL, .format = WGPUTextureFormat_RGBA8Unorm, .dimension = WGPUTextureViewDimension_2D,
                .baseMipLevel = 0, .mipLevelCount = 1, .baseArrayLayer = 0, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All
            };
            res->shadow_composite_view = wgpuTextureCreateView(res->shadow_composite_texture, &scViewDesc);
            if (!res->shadow_composite_view) return false;
        }

        // Recreate Gaussian bind groups
        if (res->gaussian_blur_bgl) {
            WGPUBindGroupEntry entries[4]; memset(entries, 0, sizeof(entries));
            entries[2].binding = 2; entries[2].buffer = res->gaussian_uniform_buffer; entries[2].offset = 0; entries[2].size = 16;
            entries[3].binding = 3; entries[3].buffer = res->gaussian_kernel_buffer; entries[3].offset = 0; entries[3].size = res->gaussian_kernel_buffer_size;
            WGPUBindGroupDescriptor bgDesc = { .nextInChain = NULL, .layout = res->gaussian_blur_bgl, .entryCount = 4, .entries = entries };
            entries[0].binding = 0; entries[0].textureView = res->ping_pong_view_a; entries[1].binding = 1; entries[1].textureView = res->ping_pong_view_b;
            res->gaussian_blur_h_bg_a = wgpuDeviceCreateBindGroup(device, &bgDesc);
            entries[0].binding = 0; entries[0].textureView = res->ping_pong_view_b; entries[1].binding = 1; entries[1].textureView = res->ping_pong_view_a;
            res->gaussian_blur_h_bg_b = wgpuDeviceCreateBindGroup(device, &bgDesc);
            entries[0].binding = 0; entries[0].textureView = res->ping_pong_view_b; entries[1].binding = 1; entries[1].textureView = res->ping_pong_view_a;
            res->gaussian_blur_v_bg_a = wgpuDeviceCreateBindGroup(device, &bgDesc);
            entries[0].binding = 0; entries[0].textureView = res->ping_pong_view_a; entries[1].binding = 1; entries[1].textureView = res->ping_pong_view_b;
            res->gaussian_blur_v_bg_b = wgpuDeviceCreateBindGroup(device, &bgDesc);
        }

        // Recreate filter bind groups
        if (res->filter_bgl) {
            WGPUBindGroupEntry fEntries[3]; memset(fEntries, 0, sizeof(fEntries));
            fEntries[2].binding = 2; fEntries[2].buffer = res->filter_uniform_buffer; fEntries[2].offset = 0; fEntries[2].size = sizeof(FS_FilterUniforms);
            WGPUBindGroupDescriptor fBGDesc = { .nextInChain = NULL, .layout = res->filter_bgl, .entryCount = 3, .entries = fEntries };
            fEntries[0].binding = 0; fEntries[0].textureView = res->ping_pong_view_a; fEntries[1].binding = 1; fEntries[1].textureView = res->ping_pong_view_b;
            res->filter_bg_a = wgpuDeviceCreateBindGroup(device, &fBGDesc);
            fEntries[0].binding = 0; fEntries[0].textureView = res->ping_pong_view_b; fEntries[1].binding = 1; fEntries[1].textureView = res->ping_pong_view_a;
            res->filter_bg_b = wgpuDeviceCreateBindGroup(device, &fBGDesc);
        }

        // Recreate filter copy bind groups
        if (res->filter_copy_bgl && res->shadow_sampler) {
            WGPUBindGroupEntry fcA[2]; memset(fcA, 0, sizeof(fcA));
            fcA[0].binding = 0; fcA[0].textureView = res->ping_pong_view_a; fcA[1].binding = 1; fcA[1].sampler = res->shadow_sampler;
            WGPUBindGroupDescriptor fcBGDesc = { .nextInChain = NULL, .layout = res->filter_copy_bgl, .entryCount = 2, .entries = fcA };
            res->filter_copy_bg = wgpuDeviceCreateBindGroup(device, &fcBGDesc);
            res->filter_copy_bg_back = wgpuDeviceCreateBindGroup(device, &fcBGDesc);
        }

        // Recreate shadow composite bind group
        if (res->shadow_composite_bgl && res->shadow_sampler && res->shadow_uniform_buffer) {
            WGPUBindGroupEntry scBGEntries[3]; memset(scBGEntries, 0, sizeof(scBGEntries));
            scBGEntries[0].binding = 0; scBGEntries[0].textureView = res->ping_pong_view_a;
            scBGEntries[1].binding = 1; scBGEntries[1].sampler = res->shadow_sampler;
            scBGEntries[2].binding = 2; scBGEntries[2].buffer = res->shadow_uniform_buffer; scBGEntries[2].offset = 0; scBGEntries[2].size = sizeof(FS_ShadowUniforms);
            WGPUBindGroupDescriptor scBGDesc = { .nextInChain = NULL, .layout = res->shadow_composite_bgl, .entryCount = 3, .entries = scBGEntries };
            res->shadow_composite_bg = wgpuDeviceCreateBindGroup(device, &scBGDesc);
        }

        // Recreate drop shadow bind groups
        if (res->drop_shadow_c_bgl && res->shadow_uniform_buffer) {
            if (res->ping_pong_view_a && res->ping_pong_view_b && res->shadow_composite_view) {
                WGPUBindGroupEntry dsCEntries[4];
                memset(dsCEntries, 0, sizeof(dsCEntries));
                dsCEntries[0].binding = 0; dsCEntries[0].textureView = res->ping_pong_view_a;
                dsCEntries[1].binding = 1; dsCEntries[1].textureView = res->ping_pong_view_b;
                dsCEntries[2].binding = 2; dsCEntries[2].textureView = res->shadow_composite_view;
                dsCEntries[3].binding = 3; dsCEntries[3].buffer = res->shadow_uniform_buffer;
                dsCEntries[3].offset = 0; dsCEntries[3].size = sizeof(FS_ShadowUniforms);
                WGPUBindGroupDescriptor dsCBGDesc = {
                    .nextInChain = NULL, .layout = res->drop_shadow_c_bgl,
                    .entryCount = 4, .entries = dsCEntries
                };
                res->drop_shadow_c_bg = wgpuDeviceCreateBindGroup(device, &dsCBGDesc);
            }
        }

        if (res->drop_shadow_bgl && res->shadow_sampler && res->shadow_uniform_buffer) {
            WGPUBindGroupEntry dsBGEntries[4]; memset(dsBGEntries, 0, sizeof(dsBGEntries));
            dsBGEntries[0].binding = 0; dsBGEntries[0].textureView = res->ping_pong_view_a;
            dsBGEntries[1].binding = 1; dsBGEntries[1].textureView = res->ping_pong_view_b;
            dsBGEntries[2].binding = 2; dsBGEntries[2].sampler = res->shadow_sampler;
            dsBGEntries[3].binding = 3; dsBGEntries[3].buffer = res->shadow_uniform_buffer; dsBGEntries[3].offset = 0; dsBGEntries[3].size = sizeof(FS_ShadowUniforms);
            WGPUBindGroupDescriptor dsBGDesc = { .nextInChain = NULL, .layout = res->drop_shadow_bgl, .entryCount = 4, .entries = dsBGEntries };
            res->drop_shadow_bg = wgpuDeviceCreateBindGroup(device, &dsBGDesc);
        }
    }

    // Always recreate presentation_scene_bg (samples scene_view -> render to canvas)
    if (res->filter_copy_bgl && res->shadow_sampler && res->scene_texture) {
        if (res->presentation_scene_bg) { fs_effects_release_bg(res->presentation_scene_bg); res->presentation_scene_bg = NULL; }
        WGPUTextureViewDescriptor presSceneViewDesc = {
            .nextInChain = NULL, .format = WGPUTextureFormat_RGBA8Unorm,
            .dimension = WGPUTextureViewDimension_2D,
            .baseMipLevel = 0, .mipLevelCount = 1,
            .baseArrayLayer = 0, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All
        };
        WGPUTextureView presSceneTexView = wgpuTextureCreateView(res->scene_texture, &presSceneViewDesc);
        if (presSceneTexView) {
            WGPUBindGroupEntry presEntry[2];
            memset(presEntry, 0, sizeof(presEntry));
            presEntry[0].binding = 0; presEntry[0].textureView = presSceneTexView;
            presEntry[1].binding = 1; presEntry[1].sampler = res->shadow_sampler;
            WGPUBindGroupDescriptor presBGDesc = {
                .nextInChain = NULL, .layout = res->filter_copy_bgl, .entryCount = 2, .entries = presEntry
            };
            res->presentation_scene_bg = wgpuDeviceCreateBindGroup(device, &presBGDesc);
            wgpuTextureViewRelease(presSceneTexView);
            if (!res->presentation_scene_bg) return false;
        } else {
            return false;
        }
    }

    res->kernel_dirty = true; return true;
}

bool fs_effects_create_presentation_pipeline(FS_Core* core, WGPUTextureFormat surface_format) {
    if (!core) return false;
    struct FS_EffectResources* fx = fs_core_get_effects_resources(core);
    if (!fx || !fx->filter_copy_shader_module || !fx->vert_shader_module || !fx->filter_copy_bgl) return false;
    if (fx->presentation_pipeline) {
        wgpuRenderPipelineRelease(fx->presentation_pipeline);
        fx->presentation_pipeline = NULL;
    }
    if (fx->presentation_scene_bg) {
        fs_effects_release_bg(fx->presentation_scene_bg);
        fx->presentation_scene_bg = NULL;
    }
    // Create explicit pipeline layout from filter_copy_bgl so bind groups are compatible
    WGPUPipelineLayoutDescriptor plDesc = {
        .nextInChain = NULL,
        .label = { .data = "FS Presentation Layout", .length = 22 },
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &fx->filter_copy_bgl
    };
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(core->device, &plDesc);
    if (!pl) return false;
    WGPUColorTargetState presTarget = {
        .nextInChain = NULL,
        .format = surface_format,
        .blend = NULL,
        .writeMask = WGPUColorWriteMask_All
    };
    WGPUFragmentState presFrag = {
        .nextInChain = NULL,
        .module = fx->filter_copy_shader_module,
        .entryPoint = { .data = "filter_copy_fs_main", .length = 19 },
        .constantCount = 0, .constants = NULL,
        .targetCount = 1, .targets = &presTarget
    };
    WGPURenderPipelineDescriptor rpDesc = {
        .nextInChain = NULL,
        .label = { .data = "FS Presentation", .length = 18 },
        .layout = pl,
        .vertex = {
            .module = fx->vert_shader_module,
            .entryPoint = { .data = "fs_vs_main", .length = 10 },
            .constantCount = 0, .constants = NULL,
            .buffers = NULL, .bufferCount = 0
        },
        .primitive = {
            .topology = WGPUPrimitiveTopology_TriangleList,
            .stripIndexFormat = WGPUIndexFormat_Undefined,
            .frontFace = WGPUFrontFace_CCW,
            .cullMode = WGPUCullMode_None
        },
        .depthStencil = NULL,
        .multisample = { .count = 1, .mask = 0xFFFFFFFF, .alphaToCoverageEnabled = false },
        .fragment = &presFrag
    };
    fx->presentation_pipeline = wgpuDeviceCreateRenderPipeline(core->device, &rpDesc);
    wgpuPipelineLayoutRelease(pl);
    if (!fx->presentation_pipeline) return false;

    // Create presentation_scene_bg: samples scene_view for presentation to canvas
    // View uses scene texture's native format (RGBA8Unorm). Color space conversion
    // (linear ↔ sRGB) is handled by the pipeline's render target format.
    if (fx->scene_texture && fx->shadow_sampler) {
        WGPUTextureViewDescriptor sceneViewDesc = {
            .nextInChain = NULL,
            .format = WGPUTextureFormat_RGBA8Unorm,  // matches scene_texture's format
            .dimension = WGPUTextureViewDimension_2D,
            .baseMipLevel = 0, .mipLevelCount = 1,
            .baseArrayLayer = 0, .arrayLayerCount = 1,
            .aspect = WGPUTextureAspect_All
        };
        WGPUTextureView presSceneView = wgpuTextureCreateView(fx->scene_texture, &sceneViewDesc);
        if (presSceneView) {
            WGPUBindGroupEntry presEntry[2];
            memset(presEntry, 0, sizeof(presEntry));
            presEntry[0].binding = 0; presEntry[0].textureView = presSceneView;
            presEntry[1].binding = 1; presEntry[1].sampler = fx->shadow_sampler;
            WGPUBindGroupDescriptor presBGDesc = {
                .nextInChain = NULL, .layout = fx->filter_copy_bgl, .entryCount = 2, .entries = presEntry
            };
            fx->presentation_scene_bg = wgpuDeviceCreateBindGroup(core->device, &presBGDesc);
            wgpuTextureViewRelease(presSceneView);
        }
    }
    return true;
}

bool fs_style_set_physical_shadow(FS_Core* core, const FS_PhysicalShadowParams* params) {
    if (!core || !params) return false;
    struct FS_EffectResources* fx = fs_core_get_effects_resources(core);
    if (!fx) return false;
    float sigma = params->kernel_sigma; uint32_t kernel_size = params->kernel_size;
    if (kernel_size == 0 || sigma == 0.0f) {
        fs_gaussian_kernel_compute_for_blur(NULL, params->blur_radius, &kernel_size);
        sigma = params->blur_radius / 3.0f;
    }
    uint32_t flags = FS_EFFECT_FLAG_SHADOW_ENABLED;
    if (kernel_size >= 2) flags |= FS_EFFECT_FLAG_SHADOW_SEPARABLE;
    fx->current_shadow_params = *params;
    fx->current_shadow_params.kernel_size = kernel_size; fx->current_shadow_params.kernel_sigma = sigma; fx->current_shadow_params.flags = flags;
    fx->shadow_enabled = true;
    FS_GaussianKernel kernel;
    if (fs_gaussian_kernel_compute(&kernel, sigma, kernel_size)) { fx->current_kernel = kernel; fx->kernel_dirty = true; }
    return true;
}

bool fs_style_get_physical_shadow(FS_Core* core, FS_PhysicalShadowParams* out_params) {
    if (!core || !out_params) return false;
    struct FS_EffectResources* fx = fs_core_get_effects_resources(core);
    if (!fx || !fx->shadow_enabled) return false;
    *out_params = fx->current_shadow_params; return true;
}

bool fs_style_set_gaussian_shadow(FS_Core* core, float blur_radius, uint32_t color_rgba8, float offset_x, float offset_y) {
    FS_PhysicalShadowParams params = { .blur_radius = blur_radius, .color = color_rgba8, .offset_x = offset_x, .offset_y = offset_y, .kernel_size = 0, .kernel_sigma = 0.0f, .flags = 0, .alpha = 1.0f, .spread_px = 0.0f };
    return fs_style_set_physical_shadow(core, &params);
}

bool fs_style_disable_physical_shadow(FS_Core* core) {
    if (!core) return false;
    struct FS_EffectResources* fx = fs_core_get_effects_resources(core);
    if (!fx) return false;
    fx->shadow_enabled = false; memset(&fx->current_shadow_params, 0, sizeof(FS_PhysicalShadowParams)); return true;
}

bool fs_style_is_physical_shadow_enabled(FS_Core* core) {
    if (!core) return false;
    struct FS_EffectResources* fx = fs_core_get_effects_resources(core); return fx && fx->shadow_enabled;
}

bool fs_effects_render_physical_shadow(FS_Core* core, WGPUCommandEncoder encoder,
    const FS_PhysicalShadowParams* params, WGPUTexture source_texture,
    WGPUTextureView source_view, WGPUTextureView dest_view, uint32_t width, uint32_t height) {
    if (!core || !encoder || !params || !source_texture || !source_view || !dest_view || width == 0 || height == 0) return false;
    struct FS_EffectResources* res = fs_core_get_effects_resources(core);
    if (!res) return false;
    (void)source_view;

    // Upload kernel if dirty
    if (res->kernel_dirty) {
        uint32_t kernel_size = res->current_kernel.size;
        size_t upload_size = kernel_size * sizeof(float);
        uint32_t padded = ((kernel_size + 3u) / 4u) * 4u;
        size_t padded_size = padded * sizeof(float);
        float staging[256] = {0};
        memcpy(staging, res->current_kernel.weights, upload_size);
        wgpuQueueWriteBuffer(core->queue, res->gaussian_kernel_buffer, 0, staging, padded_size);
        float uniform_data[4] = { (float)kernel_size, res->current_kernel.sigma, 0.0f, 0.0f };
        wgpuQueueWriteBuffer(core->queue, res->gaussian_uniform_buffer, 0, uniform_data, sizeof(uniform_data));
        res->kernel_dirty = false;
    }

    // --- Step 1: Copy source to ping_pong A ---
    WGPUTexelCopyTextureInfo src_info = {
        .texture = source_texture,
        .mipLevel = 0u,
        .origin = {0u, 0u, 0u},
        .aspect = WGPUTextureAspect_All
    };
    WGPUTexelCopyTextureInfo dst_info = {
        .texture = res->ping_pong_texture_a,
        .mipLevel = 0u,
        .origin = {0u, 0u, 0u},
        .aspect = WGPUTextureAspect_All
    };
    WGPUExtent3D copyExtent = { width, height, 1u };
    wgpuCommandEncoderCopyTextureToTexture(encoder, &src_info, &dst_info, &copyExtent);

    // --- Step 2: Gaussian blur H pass (A -> B) ---
    WGPUComputePassEncoder blurPass = wgpuCommandEncoderBeginComputePass(encoder, &(WGPUComputePassDescriptor){ .nextInChain = NULL, .label = { .data = "Shadow Blur H", .length = 13 } });
    if (blurPass) {
        wgpuComputePassEncoderSetPipeline(blurPass, res->gaussian_blur_h_pipeline);
        wgpuComputePassEncoderSetBindGroup(blurPass, 0, res->gaussian_blur_h_bg_a, 0, NULL);
        uint32_t dispatchW = (width + 255u) / 256u;
        wgpuComputePassEncoderDispatchWorkgroups(blurPass, dispatchW, 1u, 1u);
        wgpuComputePassEncoderEnd(blurPass);
        wgpuComputePassEncoderRelease(blurPass);
    }

    // --- Step 3: Gaussian blur V pass (B -> A) ---
    WGPUComputePassEncoder blurVPass = wgpuCommandEncoderBeginComputePass(encoder, &(WGPUComputePassDescriptor){ .nextInChain = NULL, .label = { .data = "Shadow Blur V", .length = 13 } });
    if (blurVPass) {
        wgpuComputePassEncoderSetPipeline(blurVPass, res->gaussian_blur_v_pipeline);
        wgpuComputePassEncoderSetBindGroup(blurVPass, 0, res->gaussian_blur_v_bg_a, 0, NULL);
        uint32_t dispatchH = (height + 255u) / 256u;
        wgpuComputePassEncoderDispatchWorkgroups(blurVPass, 1u, dispatchH, 1u);
        wgpuComputePassEncoderEnd(blurVPass);
        wgpuComputePassEncoderRelease(blurVPass);
    }

    // --- Step 4: Shadow composite render pass ---
    float shadow_c = (float)((params->color >> 24) & 0xFF) / 255.0f;
    float shadow_b = (float)((params->color >> 16) & 0xFF) / 255.0f;
    float shadow_g = (float)((params->color >> 8) & 0xFF) / 255.0f;
    float shadow_r = (float)((params->color >> 0) & 0xFF) / 255.0f;
    FS_ShadowUniforms su = { .shadow_color = { shadow_r, shadow_g, shadow_b, shadow_c }, .shadow_offset = { params->offset_x, params->offset_y }, ._padding = { 0.0f, 0.0f } };
    wgpuQueueWriteBuffer(core->queue, res->shadow_uniform_buffer, 0, &su, sizeof(su));

    WGPURenderPassColorAttachment rpColorAtt = {
        .view = dest_view,
        .depthSlice = 0xFFFFFFFF,
        .loadOp = WGPULoadOp_Load,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = { 0, 0, 0, 0 }
    };
    WGPURenderPassDescriptor rpDesc = {
        .nextInChain = NULL,
        .label = { .data = "Shadow Composite", .length = 17 },
        .colorAttachmentCount = 1,
        .colorAttachments = &rpColorAtt,
        .depthStencilAttachment = NULL
    };
    WGPURenderPassEncoder shadowPass = wgpuCommandEncoderBeginRenderPass(encoder, &rpDesc);
    if (shadowPass) {
        wgpuRenderPassEncoderSetViewport(shadowPass, 0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f);
        wgpuRenderPassEncoderSetPipeline(shadowPass, res->shadow_composite_pipeline);
        wgpuRenderPassEncoderSetBindGroup(shadowPass, 0, res->shadow_composite_bg, 0, NULL);
        wgpuRenderPassEncoderDraw(shadowPass, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(shadowPass);
        wgpuRenderPassEncoderRelease(shadowPass);
    }
    return true;
}
