#ifndef WCN_FULLSTACK_EFFECTS_H
#define WCN_FULLSTACK_EFFECTS_H

#include "webgpu/webgpu.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "fullstack_filters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct FS_Core FS_Core;
typedef struct FS_FilterNode FS_FilterNode;

// FS_GaussianKernel and FS_PhysicalShadowParams are defined in fullstack_filters.h
// (included by fullstack_core_private.h before this header)

// Full effects resources struct (defined here for use by filter pipeline in fullstack_core.c)
struct FS_EffectResources {
    // Ping-pong textures
    WGPUTexture ping_pong_texture_a;
    WGPUTextureView ping_pong_view_a;
    WGPUTexture ping_pong_texture_b;
    WGPUTextureView ping_pong_view_b;

    // Scene texture: internal RGBA8Unorm render target (supports STORAGE_BINDING)
    WGPUTexture scene_texture;
    WGPUTextureView scene_view;

    // Gaussian blur
    WGPUComputePipeline gaussian_blur_h_pipeline;
    WGPUComputePipeline gaussian_blur_v_pipeline;
    WGPUPipelineLayout gaussian_blur_pipeline_layout;
    WGPUBindGroupLayout gaussian_blur_bgl;
    WGPUBuffer gaussian_uniform_buffer;
    WGPUBuffer gaussian_kernel_buffer;
    size_t gaussian_kernel_buffer_size;
    WGPUBindGroup gaussian_blur_h_bg_a;
    WGPUBindGroup gaussian_blur_h_bg_b;
    WGPUBindGroup gaussian_blur_v_bg_a;
    WGPUBindGroup gaussian_blur_v_bg_b;

    // Filter (compute)
    WGPUComputePipeline filter_pipeline;
    WGPUPipelineLayout filter_pipeline_layout;
    WGPUBindGroupLayout filter_bgl;
    WGPUBuffer filter_uniform_buffer;
    WGPUBindGroup filter_bg_a;
    WGPUBindGroup filter_bg_b;

    // Filter copy (render)
    WGPURenderPipeline filter_copy_pipeline;
    WGPUPipelineLayout filter_copy_pipeline_layout;
    WGPUBindGroupLayout filter_copy_bgl;
    WGPUShaderModule filter_copy_shader_module;
    WGPUBindGroup filter_copy_bg;
    WGPUBindGroup filter_copy_bg_back;

    // Drop shadow (render)
    WGPURenderPipeline drop_shadow_pipeline;
    WGPUPipelineLayout drop_shadow_pipeline_layout;
    WGPUBindGroupLayout drop_shadow_bgl;
    WGPUShaderModule drop_shadow_shader_module;
    WGPUBindGroup drop_shadow_bg;
    WGPUBindGroup drop_shadow_temp_bg;  // Per-frame temp BG

    // Shadow composite (render)
    WGPURenderPipeline shadow_composite_pipeline;
    WGPUPipelineLayout shadow_composite_pipeline_layout;
    WGPUBindGroupLayout shadow_composite_bgl;
    WGPUShaderModule shadow_composite_shader_module;
    WGPUBindGroup shadow_composite_bg;
    WGPUBuffer shadow_uniform_buffer;

    // Drop-shadow (ALL-COMPUTE pipeline)
    WGPUComputePipeline drop_shadow_c_pipeline;        // Compute-based drop-shadow composite
    WGPUPipelineLayout drop_shadow_c_pipeline_layout;
    WGPUBindGroupLayout drop_shadow_c_bgl;
    WGPUShaderModule drop_shadow_c_shader_module;
    WGPUBindGroup drop_shadow_c_bg;                // Pre-created compute bind group (blur A + original B + composite output + uniform)
    WGPUTexture shadow_composite_texture;           // Dedicated composite output texture
    WGPUTextureView shadow_composite_view;

    // Shader modules (also used by compute paths)
    WGPUShaderModule gaussian_shader_module;
    WGPUShaderModule filter_shader_module;
    WGPUShaderModule vert_shader_module;        // Shared fullscreen quad vertex shader

    // Sampler
    WGPUSampler shadow_sampler;

    // Presentation (render to canvas) - pipeline with target surface format
    WGPURenderPipeline presentation_pipeline;
    WGPUBindGroup presentation_scene_bg;  // samples scene_view (RGBA8Unorm) for canvas presentation

    // Kernel state
    FS_GaussianKernel current_kernel;
    bool kernel_dirty;

    // Shadow params
    FS_PhysicalShadowParams current_shadow_params;
    bool shadow_enabled;

    // Canvas size
    uint32_t width;
    uint32_t height;

    // Enabled flag
    bool enabled;
};

// Effects subsystem lifecycle
bool fs_effects_init(FS_Core* core);
void fs_effects_destroy(FS_Core* core);
bool fs_effects_resize(FS_Core* core, uint32_t width, uint32_t height);

// Physical shadow rendering
bool fs_effects_render_physical_shadow(
    FS_Core* core,
    WGPUCommandEncoder encoder,
    const FS_PhysicalShadowParams* params,
    WGPUTexture source_texture,
    WGPUTextureView source_view,
    WGPUTextureView dest_view,
    uint32_t width,
    uint32_t height
);

// Gaussian kernel helper
bool fs_gaussian_kernel_compute_for_blur(
    FS_GaussianKernel* out_kernel,
    float blur_radius,
    uint32_t* out_kernel_size
);

// Core integration helpers (defined in fullstack_core.c)
struct FS_EffectResources* fs_core_get_effects_resources(FS_Core* core);
void fs_core_set_effects_resources(FS_Core* core, struct FS_EffectResources* effects);

// Create presentation pipeline with the given target surface format
bool fs_effects_create_presentation_pipeline(FS_Core* core, WGPUTextureFormat surface_format);

#ifdef __cplusplus
}
#endif

#endif // WCN_FULLSTACK_EFFECTS_H
