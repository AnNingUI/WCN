#include "webgpu/webgpu.h"
#include <stdbool.h>
#include <stdint.h>
#include "fullstack_effects.h"
#include "fullstack_core_debug.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// 私有数据结构
struct FS_EffectResources {
    WGPUTexture ping_pong_texture_a;
    WGPUTextureView ping_pong_view_a;
    WGPUTexture ping_pong_texture_b;
    WGPUTextureView ping_pong_view_b;
    WGPUBuffer gaussian_kernel_buffer;
    size_t gaussian_kernel_buffer_size;
    FS_GaussianKernel current_kernel;
    bool kernel_dirty;
    FS_PhysicalShadowParams current_shadow_params;
    bool shadow_enabled;
    uint32_t width;
    uint32_t height;
    WGPUComputePipeline gaussian_blur_h_pipeline;
    WGPUComputePipeline gaussian_blur_v_pipeline;
    WGPUBindGroupLayout gaussian_blur_bgl;
    WGPUShaderModule gaussian_shader_module;
};

// 数学辅助函数
static inline float fs_gaussian_1d(float x, float sigma) {
    return expf(-(x * x) / (2.0f * sigma * sigma));
}

// 高斯核计算实现
bool fs_gaussian_kernel_compute(FS_GaussianKernel* out_kernel, float sigma, uint32_t kernel_size) {
    if (!out_kernel || kernel_size < 3 || kernel_size > FS_GAUSSIAN_KERNEL_MAX_SIZE || kernel_size % 2 == 0 || sigma <= 0.0f) {
        return false;
    }

    out_kernel->size = kernel_size;
    out_kernel->sigma = sigma;

    const uint32_t center = kernel_size / 2;
    float sum = 0.0f;

    for (uint32_t i = 0; i < kernel_size; i++) {
        const float x = (float)i - (float)center;
        out_kernel->weights[i] = fs_gaussian_1d(x, sigma);
        sum += out_kernel->weights[i];
    }

    for (uint32_t i = 0; i < kernel_size; i++) {
        out_kernel->weights[i] /= sum;
    }

    return true;
}

bool fs_gaussian_kernel_compute_for_blur(FS_GaussianKernel* out_kernel, float blur_radius, uint32_t* out_kernel_size) {
    if (blur_radius < 0.0f) blur_radius = 0.0f;
    if (blur_radius > FS_GAUSSIAN_BLUR_RADIUS_MAX) blur_radius = FS_GAUSSIAN_BLUR_RADIUS_MAX;

    const float sigma = blur_radius / 3.0f;
    uint32_t kernel_size = (uint32_t)(6.0f * sigma) | 1;
    if (kernel_size > FS_GAUSSIAN_KERNEL_MAX_SIZE) kernel_size = FS_GAUSSIAN_KERNEL_MAX_SIZE;
    if (kernel_size < 3) kernel_size = 3;

    if (out_kernel_size) *out_kernel_size = kernel_size;

    if (out_kernel) {
        return fs_gaussian_kernel_compute(out_kernel, sigma, kernel_size);
    }

    return true;
}

// Effects 资源管理实现（简化版，仅包含必要功能）
static bool fs_effects_create_pipelines(FS_EffectResources* res, WGPUDevice device) {
    if (!res || !device) return false;

    // 简化：这里应该创建实际的 compute pipeline
    // 由于完整实现需要大量代码，这里仅做占位
    // 实际实现需要：
    // 1. 编译 WGSL shader
    // 2. 创建 bind group layout
    // 3. 创建 compute pipeline

    return true;
}

static void fs_effects_destroy_pipelines(FS_EffectResources* res) {
    if (!res) return;

    if (res->gaussian_blur_h_pipeline) {
        wgpuComputePipelineRelease(res->gaussian_blur_h_pipeline);
        res->gaussian_blur_h_pipeline = NULL;
    }
    if (res->gaussian_blur_v_pipeline) {
        wgpuComputePipelineRelease(res->gaussian_blur_v_pipeline);
        res->gaussian_blur_v_pipeline = NULL;
    }
    if (res->gaussian_shader_module) {
        wgpuShaderModuleRelease(res->gaussian_shader_module);
        res->gaussian_shader_module = NULL;
    }
}

bool fs_effects_init(FS_Core* core) {
    if (!core) return false;

    FS_EffectResources* res = (FS_EffectResources*)calloc(1, sizeof(FS_EffectResources));
    if (!res) return false;

    // 获取 core 的 device
    // 注意：这里需要 core 提供获取 device 的方法
    // 简化起见，假设可以直接访问

    fs_core_set_effects_resources(core, res);
    return true;
}

void fs_effects_destroy(FS_Core* core) {
    if (!core) return;

    FS_EffectResources* res = fs_core_get_effects_resources(core);
    if (!res) return;

    fs_effects_destroy_pipelines(res);
    // 其他清理...

    free(res);
    fs_core_set_effects_resources(core, NULL);
}

bool fs_effects_resize(FS_Core* core, uint32_t width, uint32_t height) {
    if (!core) return false;

    FS_EffectResources* res = fs_core_get_effects_resources(core);
    if (!res) return false;

    res->width = width;
    res->height = height;

    // 重建 ping-pong 纹理等

    return true;
}

// Shadow API 实现
bool fs_style_set_physical_shadow(FS_Core* core, const FS_PhysicalShadowParams* params) {
    if (!core || !params) return false;

    FS_EffectResources* effects = fs_core_get_effects_resources(core);
    if (!effects) return false;

    float sigma = params->kernel_sigma;
    uint32_t kernel_size = params->kernel_size;

    if (kernel_size == 0 || sigma == 0.0f) {
        kernel_size = 0;
        fs_gaussian_kernel_compute_for_blur(NULL, params->blur_radius, &kernel_size);
        sigma = params->blur_radius / 3.0f;
    }

    uint32_t flags = FS_EFFECT_FLAG_SHADOW_ENABLED;
    if (kernel_size >= 2) {
        flags |= FS_EFFECT_FLAG_SHADOW_SEPARABLE;
    }

    effects->current_shadow_params = *params;
    effects->current_shadow_params.kernel_size = kernel_size;
    effects->current_shadow_params.kernel_sigma = sigma;
    effects->current_shadow_params.flags = flags;
    effects->shadow_enabled = true;

    FS_GaussianKernel kernel;
    if (fs_gaussian_kernel_compute(&kernel, sigma, kernel_size)) {
        effects->current_kernel = kernel;
        effects->kernel_dirty = true;
    }

    return true;
}

bool fs_style_get_physical_shadow(FS_Core* core, FS_PhysicalShadowParams* out_params) {
    if (!core || !out_params) return false;

    FS_EffectResources* effects = fs_core_get_effects_resources(core);
    if (!effects || !effects->shadow_enabled) return false;

    *out_params = effects->current_shadow_params;
    return true;
}

bool fs_style_set_gaussian_shadow(FS_Core* core, float blur_radius, uint32_t color_rgba8, float offset_x, float offset_y) {
    FS_PhysicalShadowParams params = {
        .blur_radius = blur_radius,
        .color = color_rgba8,
        .offset_x = offset_x,
        .offset_y = offset_y,
        .kernel_size = 0,
        .kernel_sigma = 0.0f,
        .flags = 0
    };
    return fs_style_set_physical_shadow(core, &params);
}

bool fs_style_set_gaussian_shadow_simple(FS_Core* core, float blur_radius, uint32_t color_rgba8) {
    return fs_style_set_gaussian_shadow(core, blur_radius, color_rgba8, FS_SHADOW_DEFAULT_OFFSET_X, FS_SHADOW_DEFAULT_OFFSET_Y);
}

bool fs_style_disable_physical_shadow(FS_Core* core) {
    if (!core) return false;

    FS_EffectResources* effects = fs_core_get_effects_resources(core);
    if (!effects) return false;

    effects->shadow_enabled = false;
    memset(&effects->current_shadow_params, 0, sizeof(FS_PhysicalShadowParams));

    return true;
}

bool fs_style_is_physical_shadow_enabled(FS_Core* core) {
    if (!core) return false;

    FS_EffectResources* effects = fs_core_get_effects_resources(core);
    if (!effects) return false;

    return effects->shadow_enabled;
}

// Core 集成辅助函数 - 需要在 fullstack_core.c 中实现
// FS_EffectResources* fs_core_get_effects_resources(FS_Core* core);
// void fs_core_set_effects_resources(FS_Core* core, FS_EffectResources* effects);

// 简化版渲染实现
bool fs_effects_render_physical_shadow(FS_Core* core, WGPUCommandEncoder encoder, const FS_PhysicalShadowParams* params, WGPUTextureView source_view, WGPUTextureView dest_view, uint32_t width, uint32_t height) {
    // 简化实现 - 实际应该执行 compute shader
    (void)core;
    (void)encoder;
    (void)params;
    (void)source_view;
    (void)dest_view;
    (void)width;
    (void)height;
    return true;
}
