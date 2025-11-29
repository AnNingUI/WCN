#ifndef WCN_EMCC_JS_H
#define WCN_EMCC_JS_H

#include "wcn_internal.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

// EM_JS function declarations for WASM builds
// These functions work around Emscripten's WebGPU binding issues

// Type definitions
typedef int WGPUTextureViewID;

// Helper function to convert WGPUTextureFormat to string
const char* wgpuTextureFormatToString(WGPUTextureFormat format);

// Render pipeline creation
WGPURenderPipeline wasm_create_render_pipeline(
    WGPUDevice device,
    WGPUPipelineLayout layout,
    WGPUShaderModule shader,
    const char* vs_entry,
    const char* fs_entry,
    uint32_t instance_stride,
    const char* format_str
);

// Bind group creation
WGPUBindGroup wasm_create_bind_group(
    WGPUDevice device,
    WGPUBindGroupLayout layout,
    WGPUBuffer instance_buffer,
    uint64_t instance_buffer_size,
    WGPUBuffer uniform_buffer
);

// Buffer creation
WGPUBuffer wasm_create_buffer(
    WGPUDevice device,
    const char* label,
    uint64_t size,
    uint32_t usage_flags
);

// Bind group layout creation
WGPUBindGroupLayout wasm_create_bind_group_layout(
    WGPUDevice device,
    const char* label,
    uint64_t min_binding_size_0,
    uint64_t min_binding_size_1
);

// Shader module creation
WGPUShaderModule wasm_create_shader_module(
    WGPUDevice device,
    const char* wgsl_code,
    const char* label
);

// SDF bind group layout creation
WGPUBindGroupLayout wasm_create_sdf_bind_group_layout(
    WGPUDevice device,
    const char* label
);

// Pipeline layout creation
WGPUPipelineLayout wasm_create_pipeline_layout(
    WGPUDevice device,
    const char* label,
    WGPUBindGroupLayout layout0,
    WGPUBindGroupLayout layout1
);

// Sampler creation
WGPUSampler wasm_create_sampler(
    WGPUDevice device,
    const char* label
);

// Render pass creation
WGPURenderPassEncoder wasm_begin_render_pass(
    WGPUCommandEncoder encoder,
    WGPUTextureViewID view_id,
    int is_first_pass
);

// SDF bind group creation
WGPUBindGroup wasm_create_sdf_bind_group(
    WGPUDevice device,
    WGPUBindGroupLayout layout,
    WGPUTextureView texture_view,
    WGPUSampler sampler
);

// Texture write function
void wasm_queue_write_texture(
    WGPUQueue queue,
    WGPUTexture texture,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    const unsigned char* data,
    size_t data_size
);

// JavaScript initialization functions
void Init_WCNJS(void);
void Init_WGPUTextureView_Map(void);
void freeWGPUTextureView(int id);

#endif // __EMSCRIPTEN__

#endif // WCN_EMCC_JS_H