#pragma once

#include "webgpu/webgpu.h"
#include <stdbool.h>
#include <stdint.h>

// 高斯核最大大小（奇数，保持 6-bit 编码）
#define FS_GAUSSIAN_KERNEL_MAX_SIZE 63u

// 最大 blur radius（像素）
#define FS_GAUSSIAN_BLUR_RADIUS_MAX 95.0f

// Shadow 渲染标志
#define FS_EFFECT_FLAG_SHADOW_ENABLED     (1u << 0u)
#define FS_EFFECT_FLAG_SHADOW_SEPARABLE   (1u << 1u)

// 默认参数
#define FS_SHADOW_DEFAULT_OFFSET_X        4.0f
#define FS_SHADOW_DEFAULT_OFFSET_Y        4.0f
#define FS_SHADOW_DEFAULT_COLOR           0x00000080u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FS_Core FS_Core;

// 高斯核描述符
typedef struct FS_GaussianKernel {
    float weights[FS_GAUSSIAN_KERNEL_MAX_SIZE];
    uint32_t size;
    float sigma;
} FS_GaussianKernel;

// 物理正确 Shadow 参数
typedef struct FS_PhysicalShadowParams {
    float blur_radius;
    uint32_t color;
    float offset_x;
    float offset_y;
    uint32_t kernel_size;
    float kernel_sigma;
    uint32_t flags;
} FS_PhysicalShadowParams;

// Effect 资源（内部使用）
typedef struct FS_EffectResources FS_EffectResources;

// 初始化/清理 Effect 资源
bool fs_effects_init(FS_Core* core);
void fs_effects_destroy(FS_Core* core);
bool fs_effects_resize(FS_Core* core, uint32_t width, uint32_t height);

// 高斯核计算
bool fs_gaussian_kernel_compute(FS_GaussianKernel* out_kernel, float sigma, uint32_t kernel_size);
bool fs_gaussian_kernel_compute_for_blur(FS_GaussianKernel* out_kernel, float blur_radius, uint32_t* out_kernel_size);

// Shadow API
bool fs_style_set_physical_shadow(FS_Core* core, const FS_PhysicalShadowParams* params);
bool fs_style_get_physical_shadow(FS_Core* core, FS_PhysicalShadowParams* out_params);
bool fs_style_set_gaussian_shadow(FS_Core* core, float blur_radius, uint32_t color_rgba8, float offset_x, float offset_y);
bool fs_style_disable_physical_shadow(FS_Core* core);
bool fs_style_is_physical_shadow_enabled(FS_Core* core);

// 内部渲染 API
bool fs_effects_render_physical_shadow(FS_Core* core, WGPUCommandEncoder encoder, const FS_PhysicalShadowParams* params, WGPUTextureView source_view, WGPUTextureView dest_view, uint32_t width, uint32_t height);

// Core 集成辅助函数
FS_EffectResources* fs_core_get_effects_resources(FS_Core* core);
void fs_core_set_effects_resources(FS_Core* core, FS_EffectResources* effects);

#ifdef __cplusplus
}
#endif
