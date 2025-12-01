// ============================================================================
// WCN Emscripten JavaScript Interop Functions
// ============================================================================
// This file contains EM_JS function implementations for WASM builds.
// These functions work around Emscripten's WebGPU binding issues.
// ============================================================================

#include "WCN/WCN_WASM.h"
#include "wcn_emcc_js.h"

#ifdef __EMSCRIPTEN__

#include <emscripten.h>

// ============================================================================
// Helper Functions
// ============================================================================

const char* wgpuTextureFormatToString(WGPUTextureFormat format) {
    switch (format) {
        case WGPUTextureFormat_BGRA8Unorm: return "bgra8unorm";
        case WGPUTextureFormat_RGBA8Unorm: return "rgba8unorm";
        case WGPUTextureFormat_RGBA16Float: return "rgba16float";
        default: return "bgra8unorm";  // fallback
    }
}

// ============================================================================
// EM_JS Functions
// ============================================================================

// WASM-specific function to create render pipeline without depthStencil
// This works around Emscripten's issue where NULL pointers become empty objects
EM_JS(WGPURenderPipeline, wasm_create_render_pipeline, (
    WGPUDevice device,
    WGPUPipelineLayout layout,
    WGPUShaderModule shader,
    const char* vs_entry,
    const char* fs_entry,
    uint32_t instance_stride,
    const char* format_str
), {
    const dev = WebGPU.mgrDevice.get(device);
    const pipelineLayout = WebGPU.mgrPipelineLayout.get(layout);
    const shaderModule = WebGPU.mgrShaderModule.get(shader);
    const vsEntry = UTF8ToString(vs_entry);
    const fsEntry = UTF8ToString(fs_entry);
    const formatString = UTF8ToString(format_str);
    
    // Check if required objects are valid
    if (!dev || !pipelineLayout || !shaderModule) {
        console.error('[WASM] Invalid objects for render pipeline creation');
        console.error('[WASM] dev:', !!dev, 'pipelineLayout:', !!pipelineLayout, 'shaderModule:', !!shaderModule);
        return 0; // Return null handle
    }
    
    const vertexBufferLayout = {
        arrayStride: instance_stride,
        stepMode: 'vertex',
        attributes: [
            { shaderLocation: 0, format: 'float32x4', offset: 0 },
            { shaderLocation: 1, format: 'float32x4', offset: 16 },
            { shaderLocation: 2, format: 'float32x2', offset: 32 },
            { shaderLocation: 3, format: 'uint32',   offset: 40 },
            { shaderLocation: 4, format: 'uint32',   offset: 44 },
            { shaderLocation: 5, format: 'float32x2', offset: 48 },
            { shaderLocation: 6, format: 'float32',   offset: 56 },
            { shaderLocation: 7, format: 'float32x2', offset: 64 },
            { shaderLocation: 8, format: 'float32x2', offset: 72 },
            { shaderLocation: 9, format: 'float32x2', offset: 80 },
            { shaderLocation: 10, format: 'float32x2', offset: 88 },
        ]
    };

    const descriptor = {
        label: 'Unified Renderer Pipeline',
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: vsEntry,
            buffers: [vertexBufferLayout]
        },
        primitive: {
            topology: 'triangle-list',
            frontFace: 'ccw',
            cullMode: 'none'
        },
        multisample: {
            count: 1
        },
        fragment: {
            module: shaderModule,
            entryPoint: fsEntry,
            targets: [{
                format: formatString,
                blend: {
                    color: {
                        srcFactor: 'src-alpha',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add'
                    },
                    alpha: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add'
                    }
                },
                writeMask: 0xF
            }]
        }
        // Note: depthStencil is intentionally omitted (undefined)
    };
    
    try {
        // console.log('[WASM] Creating pipeline with descriptor:', descriptor);
        // console.log('[WASM] Pipeline layout:', pipelineLayout);
        // console.log('[WASM] Shader module:', shaderModule);
        // console.log('[WASM] Vertex entry:', vsEntry, 'Fragment entry:', fsEntry);
        const pipeline = dev.createRenderPipeline(descriptor);
        return WebGPU.mgrRenderPipeline.create(pipeline);
    } catch (error) {
        console.error('[WASM] Failed to create render pipeline:', error);
        // Log additional debugging information
        console.error('[WASM] Descriptor used for render pipeline creation:', JSON.stringify(descriptor, null, 2));
        return 0; // Return null handle
    }
});

EM_JS(WGPUComputePipeline, wasm_create_compute_pipeline, (
    WGPUDevice device,
    WGPUPipelineLayout layout,
    WGPUShaderModule shader,
    const char* entry
), {
    const dev = WebGPU.mgrDevice.get(device);
    const pipelineLayout = WebGPU.mgrPipelineLayout.get(layout);
    const shaderModule = WebGPU.mgrShaderModule.get(shader);
    const entryPoint = UTF8ToString(entry);

    // Check if required objects are valid
    if (!dev || !pipelineLayout || !shaderModule) {
        console.error('[WASM] Invalid objects for compute pipeline creation');
        console.error('[WASM] dev:', !!dev, 'pipelineLayout:', !!pipelineLayout, 'shaderModule:', !!shaderModule);
        return 0; // Return null handle
    }

    // Additional debugging - check if pipelineLayout has the expected bind group layouts
    try {
        // This is just for debugging purposes - we can't actually inspect the layout in JS
        console.log('[WASM] Creating compute pipeline with layout and shader module');
    } catch (e) {
        console.error('[WASM] Error during compute pipeline creation debugging:', e);
    }

    const descriptor = {
        label: 'Instance Expander Pipeline',
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: entryPoint
        }
    };

    try {
        const pipeline = dev.createComputePipeline(descriptor);
        return WebGPU.mgrComputePipeline.create(pipeline);
    } catch (error) {
        console.error('[WASM] Failed to create compute pipeline:', error);
        // Log additional debugging information
        console.error('[WASM] Descriptor used for pipeline creation:', JSON.stringify(descriptor, null, 2));
        return 0; // Return null handle
    }
});

// WASM-specific function to create bind group
// Works around Emscripten's bind group entry structure differences
EM_JS(WGPUBindGroup, wasm_create_bind_group, (
    WGPUDevice device,
    WGPUBindGroupLayout layout,
    WGPUBuffer instance_buffer,
    uint64_t instance_buffer_size,
    WGPUBuffer uniform_buffer
), {
    const dev = WebGPU.mgrDevice.get(device);
    const bindGroupLayout = WebGPU.mgrBindGroupLayout.get(layout);
    const instBuffer = WebGPU.mgrBuffer.get(instance_buffer);
    const unifBuffer = WebGPU.mgrBuffer.get(uniform_buffer);
    
    // Check if required objects are valid
    if (!dev || !bindGroupLayout || !instBuffer || !unifBuffer) {
        console.error('[WASM] Invalid objects for bind group creation');
        return 0; // Return null handle
    }
    
    const descriptor = {
        label: 'Unified Renderer Bind Group',
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: instBuffer,
                    offset: 0,
                    size: Number(instance_buffer_size)
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: unifBuffer,
                    offset: 0,
                    size: 16
                }
            }
        ]
    };
    
    try {
        // console.log('[WASM] Creating bind group with descriptor:', descriptor);
        const bindGroup = dev.createBindGroup(descriptor);
        return WebGPU.mgrBindGroup.create(bindGroup);
    } catch (error) {
        console.error('[WASM] Failed to create bind group:', error);
        return 0; // Return null handle
    }
});

EM_JS(WGPUBindGroup, wasm_create_compute_bind_group, (
    WGPUDevice device,
    WGPUBindGroupLayout layout,
    WGPUBuffer instance_buffer,
    uint64_t instance_buffer_size,
    WGPUBuffer vertex_buffer,
    uint64_t vertex_buffer_size,
    WGPUBuffer uniform_buffer,
    uint64_t uniform_buffer_size
), {
    const dev = WebGPU.mgrDevice.get(device);
    const bindGroupLayout = WebGPU.mgrBindGroupLayout.get(layout);
    const instanceBuffer = WebGPU.mgrBuffer.get(instance_buffer);
    const vertexBuffer = WebGPU.mgrBuffer.get(vertex_buffer);
    const uniformBuffer = WebGPU.mgrBuffer.get(uniform_buffer);

    // Check if required objects are valid
    if (!dev || !bindGroupLayout || !instanceBuffer || !vertexBuffer || !uniformBuffer) {
        console.error('[WASM] Invalid objects for compute bind group creation');
        return 0; // Return null handle
    }

    const descriptor = {
        label: 'Instance Expander Compute Bind Group',
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: instanceBuffer, offset: 0, size: Number(instance_buffer_size) }
            },
            {
                binding: 1,
                resource: { buffer: vertexBuffer, offset: 0, size: Number(vertex_buffer_size) }
            },
            {
                binding: 2,
                resource: { buffer: uniformBuffer, offset: 0, size: Number(uniform_buffer_size) }
            }
        ]
    };

    try {
        const bindGroup = dev.createBindGroup(descriptor);
        return WebGPU.mgrBindGroup.create(bindGroup);
    } catch (error) {
        console.error('[WASM] Failed to create compute bind group:', error);
        return 0; // Return null handle
    }
});

// WASM-specific function to create buffers with proper usage flags
EM_JS(WGPUBuffer, wasm_create_buffer, (
    WGPUDevice device,
    const char* label,
    uint64_t size,
    uint32_t usage_flags  // 1=Storage, 2=Uniform, 4=CopyDst, 8=Vertex
), {
    const dev = WebGPU.mgrDevice.get(device);
    const labelStr = UTF8ToString(label);
    
    // Check if required objects are valid
    if (!dev) {
        console.error('[WASM] Invalid device for buffer creation');
        return 0; // Return null handle
    }
    
    let usage = 0;
    if (usage_flags & 1) usage |= GPUBufferUsage.STORAGE;
    if (usage_flags & 2) usage |= GPUBufferUsage.UNIFORM;
    if (usage_flags & 4) usage |= GPUBufferUsage.COPY_DST;
    if (usage_flags & 8) usage |= GPUBufferUsage.VERTEX;
    
    const descriptor = {
        label: labelStr,
        size: Number(size),
        usage: usage,
        mappedAtCreation: false
    };
    
    try {
        // console.log('[WASM] Creating buffer:', labelStr, 'size:', size, 'usage:', usage);
        const buffer = dev.createBuffer(descriptor);
        return WebGPU.mgrBuffer.create(buffer);
    } catch (error) {
        console.error('[WASM] Failed to create buffer:', error);
        return 0; // Return null handle
    }
});

EM_JS(WGPUBindGroupLayout, wasm_create_bind_group_layout, (
    WGPUDevice device,
    const char* label,
    uint64_t min_binding_size_0,
    uint64_t min_binding_size_1
), {
    const dev = WebGPU.mgrDevice.get(device);
    const labelStr = UTF8ToString(label);
    
    // Check if required objects are valid
    if (!dev) {
        console.error('[WASM] Invalid device for bind group layout creation');
        return 0; // Return null handle
    }

    const descriptor = {
        label: labelStr,
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: 'read-only-storage',
                    hasDynamicOffset: false,
                    minBindingSize: Number(min_binding_size_0)
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX,
                buffer: {
                    type: 'uniform',
                    hasDynamicOffset: false,
                    minBindingSize: Number(min_binding_size_1)
                }
            }
        ]
    };
    
    try {
        // console.log('[WASM] Creating bind group layout:', labelStr);
        const layout = dev.createBindGroupLayout(descriptor);
        return WebGPU.mgrBindGroupLayout.create(layout);
    } catch (error) {
        console.error('[WASM] Failed to create bind group layout:', error);
        // Log additional debugging information
        console.error('[WASM] Descriptor used for bind group layout creation:', JSON.stringify(descriptor, null, 2));
        return 0; // Return null handle
    }
});

EM_JS(WGPUBindGroupLayout, wasm_create_compute_bind_group_layout, (
    WGPUDevice device,
    const char* label,
    uint64_t min_binding_size_0,
    uint64_t min_binding_size_1,
    uint64_t min_binding_size_2
), {
    const dev = WebGPU.mgrDevice.get(device);
    const labelStr = UTF8ToString(label);

    // Check if required objects are valid
    if (!dev) {
        console.error('[WASM] Invalid device for compute bind group layout creation');
        return 0; // Return null handle
    }

    const descriptor = {
        label: labelStr,
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'read-only-storage',
                    hasDynamicOffset: false,
                    minBindingSize: Number(min_binding_size_0)
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'storage',
                    hasDynamicOffset: false,
                    minBindingSize: Number(min_binding_size_1)
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'uniform',
                    hasDynamicOffset: false,
                    minBindingSize: Number(min_binding_size_2)
                }
            }
        ]
    };

    try {
        // console.log('[WASM] Creating compute bind group layout:', labelStr);
        const layout = dev.createBindGroupLayout(descriptor);
        return WebGPU.mgrBindGroupLayout.create(layout);
    } catch (error) {
        console.error('[WASM] Failed to create compute bind group layout:', error);
        // Log additional debugging information
        console.error('[WASM] Descriptor used for compute bind group layout creation:', JSON.stringify(descriptor, null, 2));
        return 0; // Return null handle
    }
});

// WASM-specific function to create shader module
EM_JS(WGPUShaderModule, wasm_create_shader_module, (
    WGPUDevice device,
    const char* wgsl_code,
    const char* label
), {
    const dev = WebGPU.mgrDevice.get(device);
    const code = UTF8ToString(wgsl_code);
    const labelStr = label ? UTF8ToString(label) : 'Shader Module';
    
    // Check if required objects are valid
    if (!dev) {
        console.error('[WASM] Invalid device for shader module creation');
        return 0; // Return null handle
    }
    
    const descriptor = {
        label: labelStr,
        code: code
    };
    
    try {
        // console.log('[WASM] Creating shader module:', labelStr, 'code length:', code.length);
        const module = dev.createShaderModule(descriptor);
        return WebGPU.mgrShaderModule.create(module);
    } catch (error) {
        console.error('[WASM] Failed to create shader module:', error);
        // Log additional debugging information
        console.error('[WASM] Descriptor used for shader module creation:', JSON.stringify(descriptor, null, 2));
        // Also log first 500 characters of the shader code for debugging
        console.error('[WASM] Shader code (first 500 chars):', code.substring(0, 500));
        return 0; // Return null handle
    }
});

  // WASM-specific function to create SDF bind group layout (Group 1)
EM_JS(WGPUBindGroupLayout, wasm_create_sdf_bind_group_layout, (
      WGPUDevice device,
      const char* label
  ), {
      const dev = WebGPU.mgrDevice.get(device);
    const labelStr = UTF8ToString(label);
    
    // Check if required objects are valid
    if (!dev) {
        console.error('[WASM] Invalid device for SDF bind group layout creation');
        return 0; // Return null handle
    }
    
      const descriptor = {
          label: labelStr,
          entries: [
              {
                  binding: 0,
                  visibility: GPUShaderStage.FRAGMENT,
                  texture: {
                      sampleType: 'float',
                      viewDimension: '2d',
                      multisampled: false
                  }
              },
              {
                  binding: 1,
                  visibility: GPUShaderStage.FRAGMENT,
                  sampler: {
                      type: 'filtering'
                  }
              },
              {
                  binding: 2,
                  visibility: GPUShaderStage.FRAGMENT,
                  texture: {
                      sampleType: 'float',
                      viewDimension: '2d',
                      multisampled: false
                  }
              },
              {
                  binding: 3,
                  visibility: GPUShaderStage.FRAGMENT,
                  sampler: {
                      type: 'filtering'
                  }
              }
          ]
      };
    
    try {
        // console.log('[WASM] Creating SDF bind group layout:', labelStr);
        const layout = dev.createBindGroupLayout(descriptor);
        return WebGPU.mgrBindGroupLayout.create(layout);
    } catch (error) {
        console.error('[WASM] Failed to create SDF bind group layout:', error);
        // Log additional debugging information
        console.error('[WASM] Descriptor used for SDF bind group layout creation:', JSON.stringify(descriptor, null, 2));
        return 0; // Return null handle
    }
});

// WASM-specific function to create pipeline layout
EM_JS(WGPUPipelineLayout, wasm_create_pipeline_layout, (
    WGPUDevice device,
    const char* label,
    WGPUBindGroupLayout layout0,
    WGPUBindGroupLayout layout1
), {
    const dev = WebGPU.mgrDevice.get(device);
    const labelStr = UTF8ToString(label);
    const bindGroupLayout0 = WebGPU.mgrBindGroupLayout.get(layout0);
    const bindGroupLayout1 = WebGPU.mgrBindGroupLayout.get(layout1);
    
    // Check if required objects are valid
    if (!dev || !bindGroupLayout0) {
        console.error('[WASM] Invalid objects for pipeline layout creation');
        console.error('[WASM] dev:', !!dev, 'bindGroupLayout0:', !!bindGroupLayout0, 'bindGroupLayout1:', !!bindGroupLayout1);
        return 0; // Return null handle
    }

    const descriptor = {
        label: labelStr,
        bindGroupLayouts: [bindGroupLayout0, bindGroupLayout1]
    };
    
    try {
        // console.log('[WASM] Creating pipeline layout:', labelStr, 'with bind group layouts');
        const layout = dev.createPipelineLayout(descriptor);
        return WebGPU.mgrPipelineLayout.create(layout);
    } catch (error) {
        console.error('[WASM] Failed to create pipeline layout:', error);
        // Log additional debugging information
        console.error('[WASM] Descriptor used for pipeline layout creation:', JSON.stringify(descriptor, null, 2));
        return 0; // Return null handle
    }
});

// WASM-specific function to create pipeline layout with single bind group layout
EM_JS(WGPUPipelineLayout, wasm_create_single_bind_group_pipeline_layout, (
    WGPUDevice device,
    const char* label,
    WGPUBindGroupLayout layout
), {
    const dev = WebGPU.mgrDevice.get(device);
    const labelStr = UTF8ToString(label);
    const bindGroupLayout = WebGPU.mgrBindGroupLayout.get(layout);
    
    // Check if required objects are valid
    if (!dev || !bindGroupLayout) {
        console.error('[WASM] Invalid objects for single bind group pipeline layout creation');
        console.error('[WASM] dev:', !!dev, 'bindGroupLayout:', !!bindGroupLayout);
        return 0; // Return null handle
    }

    const descriptor = {
        label: labelStr,
        bindGroupLayouts: [bindGroupLayout]
    };
    
    try {
        // console.log('[WASM] Creating single bind group pipeline layout:', labelStr);
        const pipelineLayout = dev.createPipelineLayout(descriptor);
        return WebGPU.mgrPipelineLayout.create(pipelineLayout);
    } catch (error) {
        console.error('[WASM] Failed to create single bind group pipeline layout:', error);
        // Log additional debugging information
        console.error('[WASM] Descriptor used for pipeline layout creation:', JSON.stringify(descriptor, null, 2));
        return 0; // Return null handle
    }
});

// WASM-specific function to create sampler
EM_JS(WGPUSampler, wasm_create_sampler, (
    WGPUDevice device,
    const char* label
), {
    const dev = WebGPU.mgrDevice.get(device);
    const labelStr = label ? UTF8ToString(label) : 'Sampler';
    
    // Check if required objects are valid
    if (!dev) {
        console.error('[WASM] Invalid device for sampler creation');
        return 0; // Return null handle
    }
    
    const descriptor = {
        label: labelStr,
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
        addressModeW: 'clamp-to-edge',
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
        lodMinClamp: 0.0,
        lodMaxClamp: 32.0,
        maxAnisotropy: 1
    };
    
    try {
        // console.log('[WASM] Creating sampler:', labelStr, descriptor);
        const sampler = dev.createSampler(descriptor);
        return WebGPU.mgrSampler.create(sampler);
    } catch (error) {
        console.error('[WASM] Failed to create sampler:', error);
        return 0; // Return null handle
    }
});

// WASM-specific function to begin render pass without depthStencil issues
EM_JS(WGPURenderPassEncoder, wasm_begin_render_pass, (
    WGPUCommandEncoder encoder,
    WGPUTextureViewID view_id,
    int is_first_pass
), {
    const commandEncoder = WebGPU.mgrCommandEncoder.get(encoder);
    const textureView = window.WCNJS?.getWGPUTextureView(view_id);
    // console.log('[WASM::wasm_begin_render_pass] ', "view_id: ", view_id, "textureView: ", textureView, "commandEncoder: ", commandEncoder);
    
    // Check if required objects are valid
    if (!commandEncoder) {
        console.error('[WASM] Invalid command encoder for render pass creation');
        return 0; // Return null handle
    }
    
    const descriptor = {
        label: 'WCN Render Pass',
        colorAttachments: [{
            view: textureView,
            loadOp: is_first_pass ? 'clear' : 'load',
            storeOp: 'store',
            clearValue: { r: 0.9, g: 0.9, b: 0.9, a: 1.0 }
        }]
        // Note: depthStencilAttachment is intentionally omitted (undefined)
    };
    
    try {
        const renderPass = commandEncoder.beginRenderPass(descriptor);
        return WebGPU.mgrRenderPassEncoder.create(renderPass);
    } catch (error) {
        console.error('[WASM] Failed to begin render pass:', error);
        return 0; // Return null handle
    }
});

// WASM-specific function to create SDF bind group
EM_JS(WGPUBindGroup, wasm_create_sdf_bind_group, (
    WGPUDevice device,
    WGPUBindGroupLayout layout,
    WGPUTextureView sdf_texture_view,
    WGPUSampler sdf_sampler,
    WGPUTextureView image_texture_view,
    WGPUSampler image_sampler
), {
    const dev = WebGPU.mgrDevice.get(device);
    const bindGroupLayout = WebGPU.mgrBindGroupLayout.get(layout);
    const sdfView = WebGPU.mgrTextureView.get(sdf_texture_view);
    const sdfSamplerObj = WebGPU.mgrSampler.get(sdf_sampler);
    const imageView = WebGPU.mgrTextureView.get(image_texture_view);
    const imageSamplerObj = WebGPU.mgrSampler.get(image_sampler);
    
    // Check if required objects are valid
    if (!dev || !bindGroupLayout || !sdfView || !sdfSamplerObj || !imageView || !imageSamplerObj) {
        console.error('[WASM] Invalid objects for SDF bind group creation');
        return 0; // Return null handle
    }
    
    const descriptor = {
        label: 'SDF Bind Group',
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: sdfView
            },
            {
                binding: 1,
                resource: sdfSamplerObj
            },
            {
                binding: 2,
                resource: imageView
            },
            {
                binding: 3,
                resource: imageSamplerObj
            }
        ]
    };
    
    try {
        const bindGroup = dev.createBindGroup(descriptor);
        return WebGPU.mgrBindGroup.create(bindGroup);
    } catch (error) {
        console.error('[WASM] Failed to create SDF bind group:', error);
        return 0; // Return null handle
    }
});

EM_JS(void, Init_WCNJS, (void), {
    if (window.WCNJS) return;
    window.WCNJS = {};
})

EM_JS(void, Init_WGPUTextureView_Map, (void), {
    if (window.WCNJS?.WGPUTextureView_Map_Is_Init) return;
    const WGPUTextureView_Map = new Map();
    const WGPUTextureView_Free_List = [];  // Keep track of freed IDs for reuse
    let WGPUTextureView_next_id = 1;
    
    function WGPUTextureView_store (view) {
        let id;
        if (WGPUTextureView_Free_List.length > 0) {
            // Reuse a freed ID if available
            id = WGPUTextureView_Free_List.pop();
        } else {
            // Otherwise use the next available ID
            id = WGPUTextureView_next_id++;
        }
        WGPUTextureView_Map.set(id, view); 
        return id;
    }

    function WGPUTextureView_get ( id ) { 
        return WGPUTextureView_Map.get(id) || null; 
    }

    function WGPUTextureView_free ( id ) { 
        if (WGPUTextureView_Map.has(id)) {
            WGPUTextureView_Map.delete(id);
            // Add the ID to the free list for reuse
            WGPUTextureView_Free_List.push(id);
        }
    }

    window.WCNJS.WGPUTextureView_Map_Is_Init = true;
    window.WCNJS.storeWGPUTextureView = WGPUTextureView_store;
    window.WCNJS.freeWGPUTextureView  = WGPUTextureView_free;
    window.WCNJS.getWGPUTextureView  = WGPUTextureView_get;
})
WCN_WASM_EXPORT void wcn_init_js() {
    Init_WCNJS();
    Init_WGPUTextureView_Map();
}

EM_JS(void, freeWGPUTextureView, (int id), {
    window.WCNJS?.freeWGPUTextureView?.(id);
})

// WASM-specific function to write texture data
EM_JS(void, wasm_queue_write_texture, (
    WGPUQueue queue,
    WGPUTexture texture,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    const unsigned char* data,
    size_t data_size
), {
    const queueObj = WebGPU.mgrQueue.get(queue);
    const textureObj = WebGPU.mgrTexture.get(texture);
    
    // Create source data from the provided buffer
    const sourceData = new Uint8Array(Module.HEAPU8.buffer, data, data_size);
    
    // Create ImageCopyTexture descriptor
    const destination = {
        texture: textureObj,
        mipLevel: 0,
        origin: { x: x, y: y, z: 0 },
        aspect: 'all'
    };
    
    // Create ImageData layout
    const layout = {
        offset: 0,
        bytesPerRow: width * 4,  // RGBA format
        rowsPerImage: height
    };
    
    // Create size descriptor
    const size = { width: width, height: height, depthOrArrayLayers: 1 };
    
    // console.log('[WASM] Writing texture data:', width, 'x', height, 'bytes:', data_size);
    
    // Write the texture data
    queueObj.writeTexture(destination, sourceData, layout, size);
});

#endif // __EMSCRIPTEN__
