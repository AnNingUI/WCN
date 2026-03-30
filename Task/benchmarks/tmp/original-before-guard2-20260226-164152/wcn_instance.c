#include "wcn_internal.h"
#include "shader/renderer/render_2d.wgsl.h"
#include "shader/compute/instance_expander.wgsl.h"
#include "wcn_emcc_js.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

static bool wcn_is_variation_selector(uint32_t cp) {
    return (cp >= 0xFE00 && cp <= 0xFE0F) || (cp >= 0xE0100 && cp <= 0xE01EF);
}

static bool wcn_is_zero_width_joiner(uint32_t cp) {
    return cp == 0x200D;
}

static bool wcn_renderer_create_render_bind_group(WCN_Renderer* renderer) {
    if (!renderer) return false;

    if (renderer->bind_group) {
        wgpuBindGroupRelease(renderer->bind_group);
        renderer->bind_group = NULL;
    }

#ifdef __EMSCRIPTEN__
    renderer->bind_group = wasm_create_bind_group(
        renderer->device,
        renderer->bind_group_layout,
        renderer->instance_buffer,
        renderer->instance_buffer_size,
        renderer->uniform_buffer
    );
#else
    WGPUBindGroupEntry entries[] = {
        {
            .binding = 0,
            .buffer = renderer->instance_buffer,
            .offset = 0,
            .size = renderer->instance_buffer_size
        },
        {
            .binding = 1,
            .buffer = renderer->uniform_buffer,
            .offset = 0,
            .size = sizeof(WCN_RendererUniforms)
        }
    };

    WGPUBindGroupDescriptor desc = {
        .nextInChain = NULL,
        .label = "Unified Renderer Bind Group",
        .layout = renderer->bind_group_layout,
        .entryCount = 2,
        .entries = entries
    };

    renderer->bind_group = wgpuDeviceCreateBindGroup(renderer->device, &desc);
#endif

    return renderer->bind_group != NULL;
}

static bool wcn_renderer_create_compute_bind_group(WCN_Renderer* renderer) {
    if (!renderer) return false;

    if (renderer->compute_bind_group) {
        wgpuBindGroupRelease(renderer->compute_bind_group);
        renderer->compute_bind_group = NULL;
    }

#ifdef __EMSCRIPTEN__
    renderer->compute_bind_group = wasm_create_compute_bind_group(
        renderer->device,
        renderer->compute_bind_group_layout,
        renderer->instance_buffer,
        renderer->instance_buffer_size,
        renderer->vertex_buffer,
        renderer->vertex_buffer_size,
        renderer->uniform_buffer,
        sizeof(WCN_RendererUniforms)
    );
#else
    WGPUBindGroupEntry entries[] = {
        {
            .binding = 0,
            .buffer = renderer->instance_buffer,
            .offset = 0,
            .size = renderer->instance_buffer_size
        },
        {
            .binding = 1,
            .buffer = renderer->vertex_buffer,
            .offset = 0,
            .size = renderer->vertex_buffer_size
        },
        {
            .binding = 2,
            .buffer = renderer->uniform_buffer,
            .offset = 0,
            .size = sizeof(WCN_RendererUniforms)
        }
    };

    WGPUBindGroupDescriptor desc = {
        .nextInChain = NULL,
        .label = "Instance Expander Compute Bind Group",
        .layout = renderer->compute_bind_group_layout,
        .entryCount = 3,
        .entries = entries
    };

    renderer->compute_bind_group = wgpuDeviceCreateBindGroup(
        renderer->device,
        &desc
    );
#endif

    return renderer->compute_bind_group != NULL;
}

// ============================================================================
// 实例缓冲区管理 (Instance Buffer Management)
// ============================================================================

// 初始化实例缓冲区
bool wcn_instance_buffer_init(WCN_InstanceBuffer* buffer, size_t initial_capacity) {
    if (!buffer) {
        return false;
    }
    
    buffer->instances = (WCN_Instance*)malloc(initial_capacity * sizeof(WCN_Instance));
    if (!buffer->instances) {
        return false;
    }
    
    buffer->count = 0;
    buffer->capacity = initial_capacity;
    
    return true;
}

// 销毁实例缓冲区
void wcn_instance_buffer_destroy(WCN_InstanceBuffer* buffer) {
    if (!buffer) {
        return;
    }
    
    if (buffer->instances) {
        free(buffer->instances);
        buffer->instances = NULL;
    }
    
    buffer->count = 0;
    buffer->capacity = 0;
}

// 清空实例缓冲区
void wcn_instance_buffer_clear(WCN_InstanceBuffer* buffer) {
    if (!buffer) {
        return;
    }
    
    buffer->count = 0;
}

// 扩展实例缓冲区（容量翻倍）
bool wcn_instance_buffer_grow(WCN_InstanceBuffer* buffer) {
    if (!buffer) {
        return false;
    }
    
    size_t new_capacity = buffer->capacity * 2;
    WCN_Instance* new_instances = (WCN_Instance*)realloc(
        buffer->instances,
        new_capacity * sizeof(WCN_Instance)
    );
    
    if (!new_instances) {
        return false;
    }
    
    buffer->instances = new_instances;
    buffer->capacity = new_capacity;
    
    return true;
}

// 添加实例到缓冲区
bool wcn_instance_buffer_add(WCN_InstanceBuffer* buffer, const WCN_Instance* instance) {
    if (!buffer || !instance) {
        return false;
    }
    
    // 如果缓冲区已满，扩展容量
    if (buffer->count >= buffer->capacity) {
        if (!wcn_instance_buffer_grow(buffer)) {
            return false;
        }
    }
    
    // 复制实例数据
    memcpy(&buffer->instances[buffer->count], instance, sizeof(WCN_Instance));
    buffer->count++;
    
    return true;
}

// ============================================================================
// 渲染器创建和销毁 (Renderer Creation and Destruction)
// ============================================================================

// Helper function to create shader module
static WGPUShaderModule wcn_renderer_create_shader_module(
    WGPUDevice device,
    const char* wgsl_code,
    const char* label
) {
    WGPUStringView code_view = {
        .data = wgsl_code,
        .length = strlen(wgsl_code)
    };
    
    WGPUShaderSourceWGSL wgsl_source = {
        .chain = {
            .next = NULL,
            .sType = WGPUSType_ShaderSourceWGSL
        },
        .code = code_view
    };
    
#ifdef __EMSCRIPTEN__
    // Emscripten: label 是 const char*
    WGPUShaderModuleDescriptor shader_desc = {
        .nextInChain = &wgsl_source.chain
    };
    // 修复 label 字段（传 NULL 避免类型问题）
    uint8_t* desc_ptr = (uint8_t*)&shader_desc;
    *(const char**)(desc_ptr + 4) = NULL;  // label 设为 NULL
#else
    // 原生: label 是 WGPUStringView
    WGPUShaderModuleDescriptor shader_desc = {
        .nextInChain = &wgsl_source.chain,
        .label = {
            .data = label,
            .length = strlen(label)
        }
    };
#endif
    
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &shader_desc);
    if (!module) {
        // printf("Failed to create shader module: %s\n", label);
    } else {
        // printf("Shader module created successfully: %s\n", label);
    }
    return module;
}

// 创建统一渲染器
WCN_Renderer* wcn_create_renderer(
    WGPUDevice device,
    WGPUQueue queue,
    WGPUTextureFormat surface_format,
    uint32_t width,
    uint32_t height
) {
    // printf("[wcn_create_renderer] Start: format=%d, size=%ux%u\n", surface_format, width, height);
    
    if (!device || !queue) {
        // printf("[wcn_create_renderer] device or queue is NULL\n");
        return NULL;
    }
    
    // printf("[wcn_create_renderer] Allocating renderer structure\n");
    // Allocate renderer structure
    WCN_Renderer* renderer = (WCN_Renderer*)malloc(sizeof(WCN_Renderer));
    if (!renderer) {
        // printf("[wcn_create_renderer] malloc failed\n");
        return NULL;
    }
    
    memset(renderer, 0, sizeof(WCN_Renderer));
    
    // Store WebGPU resources
    renderer->device = device;
    renderer->queue = queue;
    renderer->width = width;
    renderer->height = height;
    
    // printf("[wcn_create_renderer] Initializing instance buffer\n");
    // Initialize CPU-side instance buffer (initial capacity: 1024 instances)
    if (!wcn_instance_buffer_init(&renderer->cpu_instances, 1024)) {
        // printf("[wcn_create_renderer] Instance buffer init failed\n");
        free(renderer);
        return NULL;
    }
    
    // printf("[wcn_create_renderer] Creating shader module\n");
    // Create shader module from unified shader
#ifdef __EMSCRIPTEN__
    // WASM: Use custom shader module creation
    WGPUShaderModule shader = wasm_create_shader_module(
        device,
        WCN_RENDER_2D_WGSL,
        "Unified Renderer Shader"
    );
#else
    WGPUShaderModule shader = wcn_renderer_create_shader_module(
        device,
        WCN_RENDER_2D_WGSL,
        "Unified Renderer Shader"
    );
#endif

#ifdef __EMSCRIPTEN__
    WGPUShaderModule compute_shader = wasm_create_shader_module(
        device,
        WCN_INSTANCE_EXPANDER_WGSL,
        "Instance Expander Shader"
    );
#else
    WGPUShaderModule compute_shader = wcn_renderer_create_shader_module(
        device,
        WCN_INSTANCE_EXPANDER_WGSL,
        "Instance Expander Shader"
    );
#endif
    
    if (!shader || !compute_shader) {
        if (shader) {
            wgpuShaderModuleRelease(shader);
        }
        if (compute_shader) {
            wgpuShaderModuleRelease(compute_shader);
        }
        // printf("[wcn_create_renderer] Shader module creation failed\n");
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }
    // printf("[wcn_create_renderer] Shader module OK\n");
    
    // Create bind group layout for Group 0
    // Group 0: Instance buffer (storage) + Uniforms (viewport size)
    WGPUBindGroupLayoutEntry bind_group_layout_entries[] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(WCN_Instance)
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Vertex,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(WCN_RendererUniforms)
            }
        }
    };
    
#ifdef __EMSCRIPTEN__
    // WASM: Use custom bind group layout creation
    renderer->bind_group_layout = wasm_create_bind_group_layout(
        device,
        "Unified Renderer Bind Group Layout (Group 0)",
        sizeof(WCN_Instance),
        16
    );
#else
    WGPUBindGroupLayoutDescriptor bind_group_layout_desc = {
        .nextInChain = NULL,
        .label = "Unified Renderer Bind Group Layout (Group 0)",
        .entryCount = 2,
        .entries = bind_group_layout_entries
    };
    
    renderer->bind_group_layout = wgpuDeviceCreateBindGroupLayout(
        device,
        &bind_group_layout_desc
    );
#endif
    
    if (!renderer->bind_group_layout) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuShaderModuleRelease(shader);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }

#ifdef __EMSCRIPTEN__
    renderer->compute_bind_group_layout = wasm_create_compute_bind_group_layout(
        device,
        "Instance Expander Compute Bind Group Layout",
        sizeof(WCN_Instance),
        sizeof(WCN_VertexGPU),
        sizeof(WCN_RendererUniforms)
    );
#else
    WGPUBindGroupLayoutEntry compute_bind_group_layout_entries[] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(WCN_Instance)
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(WCN_VertexGPU)
            }
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .hasDynamicOffset = false,
                .minBindingSize = sizeof(WCN_RendererUniforms)
            }
        }
    };

    WGPUBindGroupLayoutDescriptor compute_bind_group_layout_desc = {
        .nextInChain = NULL,
        .label = "Instance Expander Compute Bind Group Layout",
        .entryCount = 3,
        .entries = compute_bind_group_layout_entries
    };

    renderer->compute_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
        device,
        &compute_bind_group_layout_desc
    );
#endif

    if (!renderer->compute_bind_group_layout) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuShaderModuleRelease(shader);
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }
    
    // Create bind group layout for Group 1 (SDF Atlas)
#ifdef __EMSCRIPTEN__
    // WASM: 使用 EM_JS 函数创建 SDF bind group layout
    renderer->sdf_bind_group_layout = wasm_create_sdf_bind_group_layout(
        device,
        "Unified Renderer Texture Bind Group Layout (Group 1)"
    );
#else
    WGPUBindGroupLayoutEntry sdf_bind_group_layout_entries[] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Fragment,
            .texture = {
                .sampleType = WGPUTextureSampleType_Float,
                .viewDimension = WGPUTextureViewDimension_2D,
                .multisampled = false
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Fragment,
            .sampler = {
                .type = WGPUSamplerBindingType_Filtering
            }
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Fragment,
            .texture = {
                .sampleType = WGPUTextureSampleType_Float,
                .viewDimension = WGPUTextureViewDimension_2D,
                .multisampled = false
            }
        },
        {
            .binding = 3,
            .visibility = WGPUShaderStage_Fragment,
            .sampler = {
                .type = WGPUSamplerBindingType_Filtering
            }
        }
    };

    WGPUBindGroupLayoutDescriptor sdf_bind_group_layout_desc = {
        .nextInChain = NULL,
        .label = "Unified Renderer Texture Bind Group Layout (Group 1)",
        .entryCount = 4,
        .entries = sdf_bind_group_layout_entries
    };

    renderer->sdf_bind_group_layout = wgpuDeviceCreateBindGroupLayout(
        device,
        &sdf_bind_group_layout_desc
    );
#endif
    
    if (!renderer->sdf_bind_group_layout) {
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wgpuShaderModuleRelease(shader);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }
    
    // Create pipeline layout with both bind group layouts
#ifdef __EMSCRIPTEN__
    // WASM: 使用 EM_JS 函数创建 pipeline layout，确保 bind group layouts 正确传递
    WGPUPipelineLayout pipeline_layout = wasm_create_pipeline_layout(
        device,
        "Unified Renderer Pipeline Layout",
        renderer->bind_group_layout,
        renderer->sdf_bind_group_layout
    );
#else
    WGPUBindGroupLayout bind_group_layouts[] = {
        renderer->bind_group_layout,      // Group 0: instances + uniforms
        renderer->sdf_bind_group_layout   // Group 1: SDF atlas texture + sampler
    };
    
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
        .nextInChain = NULL,
        .label = "Unified Renderer Pipeline Layout",
        .bindGroupLayoutCount = 2,
        .bindGroupLayouts = bind_group_layouts
    };
    
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(
        device,
        &pipeline_layout_desc
    );
#endif
    
    if (!pipeline_layout) {
        wgpuBindGroupLayoutRelease(renderer->sdf_bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wgpuShaderModuleRelease(shader);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }
    
    // Create render pipeline with instanced rendering
    WGPUStringView vs_entry = {
        .data = "vs_main",
        .length = 7
    };
    
    // No vertex buffer layout - vertices are generated in shader
    WGPUVertexAttribute vertex_attributes[] = {
        {.shaderLocation = 0, .format = WGPUVertexFormat_Float32x4, .offset = 0},
        {.shaderLocation = 1, .format = WGPUVertexFormat_Float32x4, .offset = 16},
        {.shaderLocation = 2, .format = WGPUVertexFormat_Float32x2, .offset = 32},
        {.shaderLocation = 3, .format = WGPUVertexFormat_Uint32, .offset = 40},
        {.shaderLocation = 4, .format = WGPUVertexFormat_Uint32, .offset = 44},
        {.shaderLocation = 5, .format = WGPUVertexFormat_Float32x2, .offset = 48},
        {.shaderLocation = 6, .format = WGPUVertexFormat_Float32, .offset = 56},
        {.shaderLocation = 7, .format = WGPUVertexFormat_Float32x2, .offset = 64},
        {.shaderLocation = 8, .format = WGPUVertexFormat_Float32x2, .offset = 72},
        {.shaderLocation = 9, .format = WGPUVertexFormat_Float32x2, .offset = 80},
        {.shaderLocation = 10, .format = WGPUVertexFormat_Float32x2, .offset = 88}
    };

    WGPUVertexBufferLayout vertex_buffer_layout = {
        .arrayStride = sizeof(WCN_VertexGPU),
        .stepMode = WGPUVertexStepMode_Vertex,
        .attributeCount = (uint32_t)(sizeof(vertex_attributes) / sizeof(vertex_attributes[0])),
        .attributes = vertex_attributes
    };

    WGPUVertexState vertex_state = {
        .nextInChain = NULL,
        .module = shader,
        .entryPoint = vs_entry,
        .constantCount = 0,
        .constants = NULL,
        .bufferCount = 1,
        .buffers = &vertex_buffer_layout
    };
    
    // Color target with alpha blending
    WGPUBlendState blend_state = {
        .color = {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_SrcAlpha,
            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha
        },
        .alpha = {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_One,
            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha
        }
    };
    
    WGPUColorTargetState color_target = {
        .format = surface_format,
        .blend = &blend_state,
        .writeMask = WGPUColorWriteMask_All
    };
    
    WGPUStringView fs_entry = {
        .data = "fs_main",
        .length = 7
    };
    
    WGPUFragmentState fragment_state = {
        .nextInChain = NULL,
        .module = shader,
        .entryPoint = fs_entry,
        .constantCount = 0,
        .constants = NULL,
        .targetCount = 1,
        .targets = &color_target
    };
    
    WGPURenderPipelineDescriptor pipeline_desc = {
        .nextInChain = NULL,
        .label = "Unified Renderer Pipeline",
        .layout = pipeline_layout,
        .vertex = vertex_state,
        .primitive = {
            .topology = WGPUPrimitiveTopology_TriangleList,
            .stripIndexFormat = WGPUIndexFormat_Undefined,
            .frontFace = WGPUFrontFace_CCW,
            .cullMode = WGPUCullMode_None
        },
        .depthStencil = NULL,
        .multisample = {
            .count = 1,
            .mask = 0xFFFFFFFF,
            .alphaToCoverageEnabled = false
        },
        .fragment = &fragment_state
    };
    
#ifdef __EMSCRIPTEN__
    // WASM workaround: Emscripten converts NULL depthStencil to empty object {}
    // Use custom EM_JS function to create pipeline with depthStencil: undefined
    // printf("[wcn_create_renderer] Creating render pipeline (WASM workaround)\n");
    renderer->pipeline = wasm_create_render_pipeline(
        device,
        pipeline_layout,
        shader,
        vertex_state.entryPoint.data,
        fragment_state.entryPoint.data,
        sizeof(WCN_VertexGPU),
        wgpuTextureFormatToString(surface_format)
    );
#else
    // printf("[wcn_create_renderer] Creating render pipeline, depthStencil=%p\n", (void*)pipeline_desc.depthStencil);
    renderer->pipeline = wgpuDeviceCreateRenderPipeline(device, &pipeline_desc);
#endif
    
    // printf("[wcn_create_renderer] Pipeline created: %p\n", (void*)renderer->pipeline);
    
    // Clean up temporary resources
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuShaderModuleRelease(shader);

    // Check if render pipeline creation failed
    if (!renderer->pipeline) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuBindGroupLayoutRelease(renderer->sdf_bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->compute_bind_group_layout);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }

    // Create pipeline layout for compute pipeline
    WGPUPipelineLayout compute_pipeline_layout;
#ifdef __EMSCRIPTEN__
    // WASM: Use custom function for single bind group layout
    compute_pipeline_layout = wasm_create_single_bind_group_pipeline_layout(
        device,
        "Instance Expander Pipeline Layout",
        renderer->compute_bind_group_layout
    );
#else
    WGPUBindGroupLayout compute_layouts[] = {
        renderer->compute_bind_group_layout
    };
    WGPUPipelineLayoutDescriptor compute_pipeline_layout_desc = {
        .nextInChain = NULL,
        .label = "Instance Expander Pipeline Layout",
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = compute_layouts
    };

    compute_pipeline_layout = wgpuDeviceCreatePipelineLayout(
        device,
        &compute_pipeline_layout_desc
    );
#endif

    // Check if pipeline layout creation failed
    if (!compute_pipeline_layout) {
        wgpuShaderModuleRelease(compute_shader);
        wgpuBindGroupLayoutRelease(renderer->sdf_bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->compute_bind_group_layout);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }

    WGPUStringView compute_entry = {
        .data = "main",
        .length = 4
    };

    WGPUComputePipelineDescriptor compute_pipeline_desc = {
        .nextInChain = NULL,
        .label = "Instance Expander Pipeline",
        .layout = compute_pipeline_layout,
        .compute = {
            .module = compute_shader,
            .entryPoint = compute_entry,
            .constantCount = 0,
            .constants = NULL
        }
    };

#ifdef __EMSCRIPTEN__
    renderer->compute_pipeline = wasm_create_compute_pipeline(
        device,
        compute_pipeline_layout,
        compute_shader,
        compute_entry.data
    );
#else
    renderer->compute_pipeline = wgpuDeviceCreateComputePipeline(
        device,
        &compute_pipeline_desc
    );
#endif

    wgpuPipelineLayoutRelease(compute_pipeline_layout);
    wgpuShaderModuleRelease(compute_shader);
    
    // Check if compute pipeline creation failed
    if (!renderer->pipeline || !renderer->compute_pipeline) {
        wgpuBindGroupLayoutRelease(renderer->sdf_bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->compute_bind_group_layout);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }
    
    // Allocate GPU storage buffer for instances (initial size: 1024 instances)
    renderer->instance_buffer_size = 1024 * sizeof(WCN_Instance);
    
#ifdef __EMSCRIPTEN__
    // WASM: Use custom buffer creation function
    // usage_flags: 1=Storage, 4=CopyDst -> 5
    renderer->instance_buffer = wasm_create_buffer(
        device,
        "Unified Renderer Instance Buffer",
        renderer->instance_buffer_size,
        5  // Storage | CopyDst
    );
#else
    WGPUBufferDescriptor instance_buffer_desc = {
        .nextInChain = NULL,
        .label = "Unified Renderer Instance Buffer",
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = renderer->instance_buffer_size,
        .mappedAtCreation = false
    };
    
    renderer->instance_buffer = wgpuDeviceCreateBuffer(device, &instance_buffer_desc);
#endif
    
    if (!renderer->instance_buffer) {
        wgpuRenderPipelineRelease(renderer->pipeline);
        wgpuComputePipelineRelease(renderer->compute_pipeline);
        wgpuBindGroupLayoutRelease(renderer->sdf_bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->compute_bind_group_layout);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }

    renderer->vertex_batch_instance_capacity = 8192;
    renderer->vertex_buffer_size = renderer->vertex_batch_instance_capacity * 6 * sizeof(WCN_VertexGPU);
#ifdef __EMSCRIPTEN__
    renderer->vertex_buffer = wasm_create_buffer(
        device,
        "Unified Renderer Vertex Buffer",
        renderer->vertex_buffer_size,
        1 | 8  // Storage | Vertex
    );
#else
    WGPUBufferDescriptor vertex_buffer_desc = {
        .nextInChain = NULL,
        .label = "Unified Renderer Vertex Buffer",
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex,
        .size = renderer->vertex_buffer_size,
        .mappedAtCreation = false
    };
    renderer->vertex_buffer = wgpuDeviceCreateBuffer(device, &vertex_buffer_desc);
#endif

    if (!renderer->vertex_buffer) {
        wgpuBufferRelease(renderer->instance_buffer);
        wgpuRenderPipelineRelease(renderer->pipeline);
        wgpuComputePipelineRelease(renderer->compute_pipeline);
        wgpuBindGroupLayoutRelease(renderer->sdf_bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->compute_bind_group_layout);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }

    renderer->vertex_batch_instance_capacity = renderer->vertex_buffer_size / (6 * sizeof(WCN_VertexGPU));
    if (renderer->vertex_batch_instance_capacity == 0) {
        renderer->vertex_batch_instance_capacity = 1;
    }
    
    // Allocate GPU uniform buffer for window size
#ifdef __EMSCRIPTEN__
    // WASM: Use custom buffer creation function
    // usage_flags: 2=Uniform, 4=CopyDst -> 6
    renderer->uniform_buffer = wasm_create_buffer(
        device,
        "Unified Renderer Uniform Buffer",
        sizeof(WCN_RendererUniforms),
        6  // Uniform | CopyDst
    );
#else
    WGPUBufferDescriptor uniform_buffer_desc = {
        .nextInChain = NULL,
        .label = "Unified Renderer Uniform Buffer",
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size = sizeof(WCN_RendererUniforms),
        .mappedAtCreation = false
    };
    
    renderer->uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_buffer_desc);
#endif
    
    if (!renderer->uniform_buffer) {
        wgpuBufferRelease(renderer->instance_buffer);
        wgpuBufferRelease(renderer->vertex_buffer);
        wgpuRenderPipelineRelease(renderer->pipeline);
        wgpuComputePipelineRelease(renderer->compute_pipeline);
        wgpuBindGroupLayoutRelease(renderer->sdf_bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->compute_bind_group_layout);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }
    
    // Create bind group
    // printf("[wcn_create_renderer] Creating bind group: instance_buffer=%p, uniform_buffer=%p\n", 
    //      (void*)renderer->instance_buffer, (void*)renderer->uniform_buffer);
    
    if (!wcn_renderer_create_render_bind_group(renderer)) {
        wgpuBufferRelease(renderer->uniform_buffer);
        wgpuBufferRelease(renderer->instance_buffer);
        wgpuBufferRelease(renderer->vertex_buffer);
        wgpuRenderPipelineRelease(renderer->pipeline);
        wgpuComputePipelineRelease(renderer->compute_pipeline);
        wgpuBindGroupLayoutRelease(renderer->sdf_bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->compute_bind_group_layout);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }

    if (!wcn_renderer_create_compute_bind_group(renderer)) {
        wgpuBindGroupRelease(renderer->bind_group);
        wgpuBufferRelease(renderer->uniform_buffer);
        wgpuBufferRelease(renderer->vertex_buffer);
        wgpuBufferRelease(renderer->instance_buffer);
        wgpuRenderPipelineRelease(renderer->pipeline);
        wgpuComputePipelineRelease(renderer->compute_pipeline);
        wgpuBindGroupLayoutRelease(renderer->sdf_bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        wgpuBindGroupLayoutRelease(renderer->compute_bind_group_layout);
        wcn_instance_buffer_destroy(&renderer->cpu_instances);
        free(renderer);
        return NULL;
    }
    
    // Write initial viewport size to uniform buffer
    WCN_RendererUniforms uniform_data = {
        .viewport_size = {(float)width, (float)height},
        .instance_count = 0,
        .instance_offset = 0
    };
    
    wgpuQueueWriteBuffer(queue, renderer->uniform_buffer, 0, &uniform_data, sizeof(uniform_data));
    
    return renderer;
}

// 销毁统一渲染器
void wcn_destroy_renderer(WCN_Renderer* renderer) {
    if (!renderer) {
        return;
    }
    
    // Release GPU buffers
    if (renderer->uniform_buffer) {
        wgpuBufferRelease(renderer->uniform_buffer);
        renderer->uniform_buffer = NULL;
    }
    
    if (renderer->vertex_buffer) {
        wgpuBufferRelease(renderer->vertex_buffer);
        renderer->vertex_buffer = NULL;
    }
    
    if (renderer->instance_buffer) {
        wgpuBufferRelease(renderer->instance_buffer);
        renderer->instance_buffer = NULL;
    }
    
    // Release bind group and layouts
    if (renderer->bind_group) {
        wgpuBindGroupRelease(renderer->bind_group);
        renderer->bind_group = NULL;
    }

    if (renderer->compute_bind_group) {
        wgpuBindGroupRelease(renderer->compute_bind_group);
        renderer->compute_bind_group = NULL;
    }
    
    if (renderer->bind_group_layout) {
        wgpuBindGroupLayoutRelease(renderer->bind_group_layout);
        renderer->bind_group_layout = NULL;
    }

    if (renderer->compute_bind_group_layout) {
        wgpuBindGroupLayoutRelease(renderer->compute_bind_group_layout);
        renderer->compute_bind_group_layout = NULL;
    }
    
    if (renderer->sdf_bind_group_layout) {
        wgpuBindGroupLayoutRelease(renderer->sdf_bind_group_layout);
        renderer->sdf_bind_group_layout = NULL;
    }
    
    // Release pipeline
    if (renderer->pipeline) {
        wgpuRenderPipelineRelease(renderer->pipeline);
        renderer->pipeline = NULL;
    }

    if (renderer->compute_pipeline) {
        wgpuComputePipelineRelease(renderer->compute_pipeline);
        renderer->compute_pipeline = NULL;
    }
    
    // Free CPU-side instance buffer
    wcn_instance_buffer_destroy(&renderer->cpu_instances);
    
    // Free renderer structure
    free(renderer);
}

// ============================================================================
// 实例添加函数 (Instance Addition Functions)
// ============================================================================

// 添加矩形实例
void wcn_renderer_add_rect(
    WCN_Renderer* renderer,
    float x, float y, float width, float height,
    float radius,
    uint32_t color,
    const float transform[4]
) {
    if (!renderer) {
        return;
    }
    
    // Create RECT instance
    WCN_Instance instance = {0};
    
    // Set position and size
    instance.position[0] = x;
    instance.position[1] = y;
    instance.size[0] = width;
    instance.size[1] = height;
    
    // Set color (already packed as uint32_t)
    instance.color = color;
    
    // Set transform matrix (2x2)
    if (transform) {
        instance.transform[0] = transform[0];
        instance.transform[1] = transform[1];
        instance.transform[2] = transform[2];
        instance.transform[3] = transform[3];
    } else {
        // Identity matrix
        instance.transform[0] = 1.0f;
        instance.transform[1] = 0.0f;
        instance.transform[2] = 0.0f;
        instance.transform[3] = 1.0f;
    }
    
    // Set instance type
    instance.type = WCN_INSTANCE_TYPE_RECT;
    
    // UV coordinates not used for rectangles
    instance.uv[0] = 0.0f;
    instance.uv[1] = 0.0f;
    instance.uvSize[0] = 0.0f;
    instance.uvSize[1] = 0.0f;
    
    // Store corner radius in param0
    instance.flags = 0;
    instance.param0 = radius;
    
    // Add instance to CPU buffer
    wcn_instance_buffer_add(&renderer->cpu_instances, &instance);
}

void wcn_renderer_add_image(
    WCN_Renderer* renderer,
    float x, float y,
    float width, float height,
    uint32_t color,
    const float transform[4],
    const float uv_min[2],
    const float uv_size[2]
) {
    if (!renderer || width <= 0.0f || height <= 0.0f) {
        return;
    }

    WCN_Instance instance = {0};

    instance.position[0] = x;
    instance.position[1] = y;
    instance.size[0] = width;
    instance.size[1] = height;

    instance.color = color;
    instance.type = WCN_INSTANCE_TYPE_IMAGE;

    if (uv_min) {
        instance.uv[0] = uv_min[0];
        instance.uv[1] = uv_min[1];
    }
    if (uv_size) {
        instance.uvSize[0] = uv_size[0];
        instance.uvSize[1] = uv_size[1];
    }

    if (transform) {
        instance.transform[0] = transform[0];
        instance.transform[1] = transform[1];
        instance.transform[2] = transform[2];
        instance.transform[3] = transform[3];
    } else {
        instance.transform[0] = 1.0f;
        instance.transform[3] = 1.0f;
    }

    wcn_instance_buffer_add(&renderer->cpu_instances, &instance);
}

// 添加文本实例
void wcn_renderer_add_text(
    WCN_Renderer* renderer,
    WCN_Context* ctx,
    const char* text,
    float x, float y,
    float font_size,
    uint32_t color,
    const float transform[4]
) {
    if (!renderer || !ctx || !text) {
        return;
    }
    
    // Check if font decoder and SDF support are available
    if (!ctx->font_decoder || !ctx->font_decoder->get_glyph_sdf) {
        return;
    }
    
    WCN_FontFace* primary_face = ctx->current_font_face;
    if (!primary_face && ctx->font_fallback_count > 0) {
        primary_face = ctx->font_fallbacks[0];
    }
    if (!primary_face) {
        return;
    }

    // Iterate through text string (UTF-8 decode)
    float current_x = x;
    const char* ptr = text;
    WCN_FontFace* last_face = primary_face;

    while (*ptr) {
        uint32_t codepoint = wcn_decode_utf8(&ptr);
        if (codepoint == 0) break;

        if (wcn_is_variation_selector(codepoint) || wcn_is_zero_width_joiner(codepoint)) {
            continue;
        }

        WCN_AtlasGlyph* glyph = NULL;
        WCN_FontFace* glyph_face = NULL;

        if (last_face) {
            glyph = wcn_get_or_create_glyph(ctx, last_face, codepoint, font_size);
            if (glyph && glyph->is_valid) {
                glyph_face = last_face;
            }
        }

        if (!glyph_face && primary_face && primary_face != last_face) {
            glyph = wcn_get_or_create_glyph(ctx, primary_face, codepoint, font_size);
            if (glyph && glyph->is_valid) {
                glyph_face = primary_face;
            }
        }

        if (!glyph_face) {
            for (size_t i = 0; i < ctx->font_fallback_count; i++) {
                WCN_FontFace* fallback = ctx->font_fallbacks[i];
                if (!fallback || fallback == last_face || fallback == primary_face) {
                    continue;
                }
                glyph = wcn_get_or_create_glyph(ctx, fallback, codepoint, font_size);
                if (glyph && glyph->is_valid) {
                    glyph_face = fallback;
                    break;
                }
            }
        }

        if (!glyph_face || !glyph || !glyph->is_valid) {
            current_x += 8.0f;  // Default advance
            continue;
        }
        last_face = glyph_face;
        
        // For empty glyphs (like spaces), only advance without rendering
        if (glyph->width == 0 || glyph->height == 0) {
            current_x += glyph->advance_width;
            continue;
        }
        
        // Calculate glyph position
        float glyph_x = current_x + glyph->offset_x;
        float glyph_y = y + glyph->offset_y;
        
        // Create TEXT instance
        WCN_Instance instance = {0};
        
        // Set position and size
        instance.position[0] = glyph_x;
        instance.position[1] = glyph_y;
        instance.size[0] = (float)glyph->width;
        instance.size[1] = (float)glyph->height;
        
        // Set color
        instance.color = color;
        
        // Set UV coordinates
        instance.uv[0] = glyph->uv_min[0];
        instance.uv[1] = glyph->uv_min[1];
        instance.uvSize[0] = glyph->uv_max[0] - glyph->uv_min[0];
        instance.uvSize[1] = glyph->uv_max[1] - glyph->uv_min[1];
        
        // Set transform matrix (2x2)
        if (transform) {
            instance.transform[0] = transform[0];
            instance.transform[1] = transform[1];
            instance.transform[2] = transform[2];
            instance.transform[3] = transform[3];
        } else {
            // Identity matrix
            instance.transform[0] = 1.0f;
            instance.transform[1] = 0.0f;
            instance.transform[2] = 0.0f;
            instance.transform[3] = 1.0f;
        }
        
        // Set instance type
        instance.type = WCN_INSTANCE_TYPE_TEXT;
        
        // Flags and params for SDF rendering
        instance.flags = 0;
        if (glyph->is_color) {
            instance.flags |= 1; // Bit 0: TEXT_FLAG_IS_COLOR
        }
        
        instance.param0 = 0.0f;  // Can be used for SDF width parameter if needed
        
        // Add instance to CPU buffer
        wcn_instance_buffer_add(&renderer->cpu_instances, &instance);
        
        // Advance to next glyph position
        current_x += glyph->advance_width;
    }
}

// 添加三角形实例
void wcn_renderer_add_triangles(
    WCN_Renderer* renderer,
    const WCN_SimpleVertex* vertices, size_t vertex_count,
    const uint16_t* indices, size_t index_count,
    uint32_t color,
    const float transform[4]
) {
    if (!renderer || !vertices || !indices || vertex_count == 0 || index_count == 0) {
        return;
    }
    
    // Process triangles (3 indices per triangle)
    for (size_t i = 0; i + 2 < index_count; i += 3) {
        // Get the three vertices of this triangle
        uint16_t idx0 = indices[i];
        uint16_t idx1 = indices[i + 1];
        uint16_t idx2 = indices[i + 2];
        
        if (idx0 >= vertex_count || idx1 >= vertex_count || idx2 >= vertex_count) {
            continue;  // Skip invalid indices
        }
        
        const WCN_SimpleVertex* v0 = &vertices[idx0];
        const WCN_SimpleVertex* v1 = &vertices[idx1];
        const WCN_SimpleVertex* v2 = &vertices[idx2];
        
        // Apply transform to vertices if provided
        float tv0_x, tv0_y, tv1_x, tv1_y, tv2_x, tv2_y;
        
        if (transform) {
            // Apply 2x2 transform matrix
            // [m00 m01]
            // [m10 m11]
            tv0_x = v0->position[0] * transform[0] + v0->position[1] * transform[1];
            tv0_y = v0->position[0] * transform[2] + v0->position[1] * transform[3];
            
            tv1_x = v1->position[0] * transform[0] + v1->position[1] * transform[1];
            tv1_y = v1->position[0] * transform[2] + v1->position[1] * transform[3];
            
            tv2_x = v2->position[0] * transform[0] + v2->position[1] * transform[1];
            tv2_y = v2->position[0] * transform[2] + v2->position[1] * transform[3];
        } else {
            // No transform - use vertices as-is
            tv0_x = v0->position[0];
            tv0_y = v0->position[1];
            tv1_x = v1->position[0];
            tv1_y = v1->position[1];
            tv2_x = v2->position[0];
            tv2_y = v2->position[1];
        }
        
        // Calculate bounding box for the triangle
        float min_x = tv0_x;
        float max_x = tv0_x;
        float min_y = tv0_y;
        float max_y = tv0_y;
        
        if (tv1_x < min_x) min_x = tv1_x;
        if (tv1_x > max_x) max_x = tv1_x;
        if (tv1_y < min_y) min_y = tv1_y;
        if (tv1_y > max_y) max_y = tv1_y;
        
        if (tv2_x < min_x) min_x = tv2_x;
        if (tv2_x > max_x) max_x = tv2_x;
        if (tv2_y < min_y) min_y = tv2_y;
        if (tv2_y > max_y) max_y = tv2_y;
        
        float width = max_x - min_x;
        float height = max_y - min_y;
        
        // Create PATH instance for this triangle
        WCN_Instance instance = {0};
        
        // Set position (top-left of bounding box) and size
        instance.position[0] = min_x;
        instance.position[1] = min_y;
        instance.size[0] = width;
        instance.size[1] = height;
        
        // Store the three triangle vertices in the available fields:
        // uv: v0.x, v0.y
        // uvSize: v1.x, v1.y
        // param0: v2.x
        // flags: v2.y (bit-cast from float)
        instance.uv[0] = tv0_x;
        instance.uv[1] = tv0_y;
        instance.uvSize[0] = tv1_x;
        instance.uvSize[1] = tv1_y;
        instance.param0 = tv2_x;
        // Bit-cast float to uint32_t for storage in flags
        union { float f; uint32_t u; } converter;
        converter.f = tv2_y;
        instance.flags = converter.u;
        
        // Set color
        instance.color = color;
        
        // Set transform matrix to identity since we've already applied it
        // transform[2] stores edge flags (7 = all edges are outer edges)
        instance.transform[0] = 1.0f;
        instance.transform[1] = 0.0f;
        instance.transform[2] = 7.0f;  // 边缘标记: 所有边都是外边缘
        instance.transform[3] = 1.0f;
        
        // Set instance type
        instance.type = WCN_INSTANCE_TYPE_PATH;
        
        // Add instance to CPU buffer
        wcn_instance_buffer_add(&renderer->cpu_instances, &instance);
    }
}

// 添加线段实例
// cap_flags: bit 0-7 = cap style, bit 8 = render start cap, bit 9 = render end cap
void wcn_renderer_add_line(
    WCN_Renderer* renderer,
    float x1, float y1,
    float x2, float y2,
    float width,
    uint32_t color,
    const float transform[4],
    uint32_t cap_flags  // Line cap style + cap enable flags
) {
    if (!renderer) {
        return;
    }
    
    // Calculate line direction and length
    float dx = x2 - x1;
    float dy = y2 - y1;
    float length = sqrtf(dx * dx + dy * dy);
    
    if (length < 0.001f) {
        return;  // Skip zero-length lines
    }
    
    // Create LINE instance
    WCN_Instance instance = {0};
    
    // Set position (center of line) and size
    instance.position[0] = (x1 + x2) * 0.5f;
    instance.position[1] = (y1 + y2) * 0.5f;
    instance.size[0] = length;
    instance.size[1] = width;
    
    // Set color
    instance.color = color;
    
    // Set transform matrix (2x2)
    if (transform) {
        instance.transform[0] = transform[0];
        instance.transform[1] = transform[1];
        instance.transform[2] = transform[2];
        instance.transform[3] = transform[3];
    } else {
        // Identity matrix
        instance.transform[0] = 1.0f;
        instance.transform[1] = 0.0f;
        instance.transform[2] = 0.0f;
        instance.transform[3] = 1.0f;
    }
    
    // Set instance type
    instance.type = WCN_INSTANCE_TYPE_LINE;
    
    // UV coordinates store line direction for shader
    // Normalized direction vector
    instance.uv[0] = dx / length;
    instance.uv[1] = dy / length;
    instance.uvSize[0] = 0.0f;
    instance.uvSize[1] = 0.0f;
    
    // Store cap flags: bit 0-7 = cap style, bit 8 = start cap, bit 9 = end cap
    instance.flags = cap_flags;
    instance.param0 = width;  // Store line width for shader
    
    // Add instance to CPU buffer
    wcn_instance_buffer_add(&renderer->cpu_instances, &instance);
}

// ============================================================================
// GPU SDF 渲染函数 (Arc, Bezier)
// ============================================================================

// 添加圆弧实例 (GPU SDF 渲染)
void wcn_renderer_add_arc(
    WCN_Renderer* renderer,
    float cx, float cy,
    float radius,
    float start_angle,
    float end_angle,
    float stroke_width,
    uint32_t color,
    const float transform[4],
    uint32_t flags
) {
    if (!renderer || radius <= 0.0f) return;
    
    float half_stroke = stroke_width * 0.5f;
    float bbox_size = (radius + half_stroke) * 2.0f;
    
    WCN_Instance instance = {0};
    
    instance.position[0] = cx;
    instance.position[1] = cy;
    instance.size[0] = radius;
    instance.size[1] = stroke_width;
    instance.uv[0] = start_angle;
    instance.uv[1] = end_angle;
    instance.uvSize[0] = bbox_size;
    instance.uvSize[1] = bbox_size;
    instance.color = color;
    
    if (transform) {
        instance.transform[0] = transform[0];
        instance.transform[1] = transform[1];
        instance.transform[2] = transform[2];
        instance.transform[3] = transform[3];
    } else {
        instance.transform[0] = 1.0f;
        instance.transform[1] = 0.0f;
        instance.transform[2] = 0.0f;
        instance.transform[3] = 1.0f;
    }
    
    instance.type = WCN_INSTANCE_TYPE_ARC;
    instance.flags = flags;
    instance.param0 = 0.0f;
    
    wcn_instance_buffer_add(&renderer->cpu_instances, &instance);
}

// 添加圆形/扇形填充实例 (GPU SDF 渲染)
void wcn_renderer_add_circle_fill(
    WCN_Renderer* renderer,
    float cx, float cy,
    float radius,
    float start_angle,
    float end_angle,
    uint32_t color,
    const float transform[4]
) {
    if (!renderer || radius <= 0.0f) return;
    
    float bbox_size = radius * 2.0f;
    
    WCN_Instance instance = {0};
    
    instance.position[0] = cx;
    instance.position[1] = cy;
    instance.size[0] = radius;
    instance.size[1] = 0.0f;  // 填充不需要 stroke_width
    instance.uv[0] = start_angle;
    instance.uv[1] = end_angle;
    instance.uvSize[0] = bbox_size;
    instance.uvSize[1] = bbox_size;
    instance.color = color;
    
    if (transform) {
        instance.transform[0] = transform[0];
        instance.transform[1] = transform[1];
        instance.transform[2] = transform[2];
        instance.transform[3] = transform[3];
    } else {
        instance.transform[0] = 1.0f;
        instance.transform[1] = 0.0f;
        instance.transform[2] = 0.0f;
        instance.transform[3] = 1.0f;
    }
    
    instance.type = WCN_INSTANCE_TYPE_CIRCLE_FILL;
    instance.flags = 0;
    instance.param0 = 0.0f;
    
    wcn_instance_buffer_add(&renderer->cpu_instances, &instance);
}

// 添加二次贝塞尔曲线实例 (GPU SDF 渲染)
void wcn_renderer_add_quadratic_bezier(
    WCN_Renderer* renderer,
    float x0, float y0,
    float cpx, float cpy,
    float x1, float y1,
    float stroke_width,
    uint32_t color,
    const float transform[4],
    uint32_t flags
) {
    if (!renderer) return;
    
    // 计算边界框
    float min_x = fminf(fminf(x0, cpx), x1);
    float max_x = fmaxf(fmaxf(x0, cpx), x1);
    float min_y = fminf(fminf(y0, cpy), y1);
    float max_y = fmaxf(fmaxf(y0, cpy), y1);
    
    float half_stroke = stroke_width * 0.5f;
    min_x -= half_stroke;
    min_y -= half_stroke;
    max_x += half_stroke;
    max_y += half_stroke;
    
    float width = max_x - min_x;
    float height = max_y - min_y;
    
    if (width < 0.001f || height < 0.001f) return;
    
    WCN_Instance instance = {0};
    
    instance.position[0] = min_x;
    instance.position[1] = min_y;
    instance.size[0] = width;
    instance.size[1] = height;
    
    // 归一化坐标
    instance.uv[0] = (cpx - min_x) / width;
    instance.uv[1] = (cpy - min_y) / height;
    instance.uvSize[0] = (x1 - min_x) / width;
    instance.uvSize[1] = (y1 - min_y) / height;
    
    float norm_x0 = (x0 - min_x) / width;
    float norm_y0 = (y0 - min_y) / height;
    
    if (transform) {
        instance.transform[0] = transform[0];
        instance.transform[1] = transform[1];
    } else {
        instance.transform[0] = 1.0f;
        instance.transform[1] = 0.0f;
    }
    instance.transform[2] = norm_x0;
    instance.transform[3] = norm_y0;
    
    instance.type = WCN_INSTANCE_TYPE_BEZIER;
    instance.flags = flags;
    instance.color = color;
    instance.param0 = stroke_width;
    
    wcn_instance_buffer_add(&renderer->cpu_instances, &instance);
}

// 添加三次贝塞尔曲线 (分解为两个二次贝塞尔)
void wcn_renderer_add_cubic_bezier(
    WCN_Renderer* renderer,
    float x0, float y0,
    float cp1x, float cp1y,
    float cp2x, float cp2y,
    float x1, float y1,
    float stroke_width,
    uint32_t color,
    const float transform[4],
    uint32_t flags
) {
    if (!renderer) return;
    
    // 将三次贝塞尔分解为两个二次贝塞尔
    float mid_x = (x0 + 3.0f * cp1x + 3.0f * cp2x + x1) / 8.0f;
    float mid_y = (y0 + 3.0f * cp1y + 3.0f * cp2y + y1) / 8.0f;
    
    float q0_cp_x = (x0 + cp1x) * 0.5f;
    float q0_cp_y = (y0 + cp1y) * 0.5f;
    float q1_cp_x = (cp2x + x1) * 0.5f;
    float q1_cp_y = (cp2y + y1) * 0.5f;
    
    wcn_renderer_add_quadratic_bezier(renderer, x0, y0, q0_cp_x, q0_cp_y, mid_x, mid_y, stroke_width, color, transform, flags);
    wcn_renderer_add_quadratic_bezier(renderer, mid_x, mid_y, q1_cp_x, q1_cp_y, x1, y1, stroke_width, color, transform, flags);
}

// ============================================================================
// 渲染函数 (Rendering Functions)
// ============================================================================

// 渲染所有实例
// 渲染所有实例
void wcn_renderer_render(
    WCN_Context* ctx,
    WGPUTextureView sdf_atlas_view,
    WGPUTextureView image_atlas_view
) {
    if (!ctx || !ctx->renderer || !ctx->current_command_encoder) {
        return;
    }

    WCN_Renderer* renderer = ctx->renderer;

    if (renderer->cpu_instances.count == 0) {
        return;
    }

    if (!sdf_atlas_view || !ctx->sdf_sampler) {
        printf("Warning: No SDF atlas provided to renderer, skipping draw\n");
        return;
    }

    if (!image_atlas_view || !ctx->image_sampler) {
        printf("Warning: No image atlas provided to renderer, skipping draw\n");
        return;
    }

    // 检查并调整 Instance Buffer 大小
    size_t required_size = renderer->cpu_instances.count * sizeof(WCN_Instance);
    if (required_size > renderer->instance_buffer_size) {
        // 保存旧缓冲区以便释放
        WGPUBuffer old_buffer = renderer->instance_buffer;
        WGPUBindGroup old_bind_group = renderer->bind_group;
        WGPUBindGroup old_compute_bind_group = renderer->compute_bind_group;

        // 计算新的缓冲区大小（2的幂次方，但不超过最大限制）
        size_t new_size = renderer->instance_buffer_size ? renderer->instance_buffer_size : sizeof(WCN_Instance) * 1024;
        const size_t MAX_INSTANCE_BUFFER_SIZE = 64 * 1024 * 1024; // 64MB 最大限制

        while (new_size < required_size && new_size < MAX_INSTANCE_BUFFER_SIZE) {
            new_size *= 2;
        }

        // 如果仍然不够，使用所需大小但不超过最大限制
        if (new_size < required_size) {
            new_size = required_size;
        }
        if (new_size > MAX_INSTANCE_BUFFER_SIZE) {
            new_size = MAX_INSTANCE_BUFFER_SIZE;
            // 如果仍然不够，只渲染部分实例
            if (required_size > MAX_INSTANCE_BUFFER_SIZE) {
                required_size = MAX_INSTANCE_BUFFER_SIZE;
            }
        }

#ifdef __EMSCRIPTEN__
        renderer->instance_buffer = wasm_create_buffer(
            renderer->device,
            "Unified Renderer Instance Buffer (Resized)",
            new_size,
            1 | 4 // Usage: Storage | CopyDst
        );
#else
        WGPUBufferDescriptor instance_buffer_desc = {
            .nextInChain = NULL,
            .label = "Unified Renderer Instance Buffer (Resized)",
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size = new_size,
            .mappedAtCreation = false
        };
        renderer->instance_buffer = wgpuDeviceCreateBuffer(
            renderer->device,
            &instance_buffer_desc
        );
#endif

        // 如果创建新缓冲区失败，恢复旧缓冲区
        if (!renderer->instance_buffer) {
            renderer->instance_buffer = old_buffer;
            return;
        }

        renderer->instance_buffer_size = new_size;

        // 创建新的绑定组
        bool bind_group_ok = wcn_renderer_create_render_bind_group(renderer);
        bool compute_bind_group_ok = wcn_renderer_create_compute_bind_group(renderer);

        if (!bind_group_ok || !compute_bind_group_ok) {
            // 如果绑定组创建失败，回滚到旧缓冲区
            wgpuBufferRelease(renderer->instance_buffer);
            renderer->instance_buffer = old_buffer;
            renderer->instance_buffer_size = old_buffer ? new_size : 0;
            return;
        }

        // 成功创建新缓冲区，释放旧资源
        if (old_buffer) {
            wgpuBufferRelease(old_buffer);
        }
        if (old_bind_group) {
            wgpuBindGroupRelease(old_bind_group);
        }
        if (old_compute_bind_group) {
            wgpuBindGroupRelease(old_compute_bind_group);
        }
    }

    // 上传 Instance 数据
    wgpuQueueWriteBuffer(
        renderer->queue,
        renderer->instance_buffer,
        0,
        renderer->cpu_instances.instances,
        required_size
    );

    size_t total_instances = renderer->cpu_instances.count;
    size_t batch_capacity = renderer->vertex_batch_instance_capacity;
    if (batch_capacity == 0) {
        return;
    }

    size_t instance_offset = 0;

    // --- 开始 Batch 循环 ---
    while (instance_offset < total_instances) {
        size_t batch_instances = total_instances - instance_offset;
        if (batch_instances > batch_capacity) {
            batch_instances = batch_capacity;
        }

        // 更新 Uniforms
        WCN_RendererUniforms uniform_data = {
            .viewport_size = {(float)ctx->width, (float)ctx->height},
            .instance_count = (uint32_t)batch_instances,
            .instance_offset = (uint32_t)instance_offset
        };
        wgpuQueueWriteBuffer(renderer->queue, renderer->uniform_buffer, 0, &uniform_data, sizeof(uniform_data));

        // --- Compute Pass (顶点展开) ---
        WGPUComputePassDescriptor compute_pass_desc = {
            .nextInChain = NULL,
            .label = "Instance Expand Pass"
        };
        WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(
            ctx->current_command_encoder,
            &compute_pass_desc
        );
        if (!compute_pass) {
            return;
        }

        wgpuComputePassEncoderSetPipeline(compute_pass, renderer->compute_pipeline);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, renderer->compute_bind_group, 0, NULL);

        // [Updated] 计算 Dispatch Workgroups
        // 逻辑变更：现在每个线程处理 1 个顶点，而不是 1 个实例。
        // Workgroup Size 从 64 变更为 256 (匹配 Shader 中的 @workgroup_size(256))
        uint32_t total_vertices_in_batch = (uint32_t)batch_instances * 6;
        uint32_t workgroups = (total_vertices_in_batch + 255) / 256;

        // 确保至少分发 1 个组 (虽然 total_vertices_in_batch > 0 时上面公式已保证)
        if (workgroups == 0 && total_vertices_in_batch > 0) {
            workgroups = 1;
        }

        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroups, 1, 1);
        wgpuComputePassEncoderEnd(compute_pass);
        // wgpuComputePassEncoderRelease(compute_pass); // 注意：某些实现可能需要释放 encoder，WebGPU C API 通常由 End 隐式完成或不需要

        // --- Render Pass (绘制) ---
        size_t batch_vertex_bytes = batch_instances * 6 * sizeof(WCN_VertexGPU);

        WGPULoadOp load_op = ctx->render_pass_needs_begin ? ctx->pending_color_load_op : WGPULoadOp_Load;
        ctx->render_pass_needs_begin = false;
        ctx->pending_color_load_op = WGPULoadOp_Load;

#ifdef __EMSCRIPTEN__
        ctx->current_render_pass = wasm_begin_render_pass(
            ctx->current_command_encoder,
            ctx->current_texture_view_id,
            load_op == WGPULoadOp_Clear ? 1 : 0
        );
#else
        WGPURenderPassColorAttachment color_attachment = {
            .view = ctx->current_texture_view,
            .resolveTarget = NULL,
            .loadOp = load_op,
            .storeOp = WGPUStoreOp_Store,
            .clearValue = ctx->pending_clear_color
        };

        WGPURenderPassDescriptor render_pass_desc = {
            .nextInChain = NULL,
            .label = "WCN Render Pass",
            .colorAttachmentCount = 1,
            .colorAttachments = &color_attachment,
            .depthStencilAttachment = NULL,
            .occlusionQuerySet = NULL,
            .timestampWrites = NULL
        };

        ctx->current_render_pass = wgpuCommandEncoderBeginRenderPass(
            ctx->current_command_encoder,
            &render_pass_desc
        );
#endif

        if (!ctx->current_render_pass) {
            return;
        }

        WGPUBindGroup sdf_bind_group = NULL;

        // 创建纹理绑定组，将文本和图像图集一起绑定
#ifdef __EMSCRIPTEN__
        sdf_bind_group = wasm_create_sdf_bind_group(
            renderer->device,
            renderer->sdf_bind_group_layout,
            sdf_atlas_view,
            ctx->sdf_sampler,
            image_atlas_view,
            ctx->image_sampler
        );
#else
        WGPUBindGroupEntry sdf_bind_group_entries[] = {
            { .binding = 0, .textureView = sdf_atlas_view },
            { .binding = 1, .sampler = ctx->sdf_sampler },
            { .binding = 2, .textureView = image_atlas_view },
            { .binding = 3, .sampler = ctx->image_sampler }
        };

        WGPUBindGroupDescriptor sdf_bind_group_desc = {
            .nextInChain = NULL,
            .label = "WCN Texture Bind Group",
            .layout = renderer->sdf_bind_group_layout,
            .entryCount = 4,
            .entries = sdf_bind_group_entries
        };

        sdf_bind_group = wgpuDeviceCreateBindGroup(
            renderer->device,
            &sdf_bind_group_desc
        );
#endif
        if (sdf_bind_group) {
            wgpuRenderPassEncoderSetPipeline(ctx->current_render_pass, renderer->pipeline);
            wgpuRenderPassEncoderSetBindGroup(ctx->current_render_pass, 0, renderer->bind_group, 0, NULL);
            wgpuRenderPassEncoderSetBindGroup(ctx->current_render_pass, 1, sdf_bind_group, 0, NULL);
            wgpuRenderPassEncoderSetVertexBuffer(ctx->current_render_pass, 0, renderer->vertex_buffer, 0, batch_vertex_bytes);

            // Draw 调用保持不变，绘制 batch_instances * 6 个顶点
            wgpuRenderPassEncoderDraw(ctx->current_render_pass, (uint32_t)(batch_instances * 6), 1, 0, 0);

            wgpuBindGroupRelease(sdf_bind_group);
        }

        wgpuRenderPassEncoderEnd(ctx->current_render_pass);
        wgpuRenderPassEncoderRelease(ctx->current_render_pass);
        ctx->current_render_pass = NULL;

        instance_offset += batch_instances;
    }

    // 清理 View 引用
#ifdef __EMSCRIPTEN__
    if (ctx->current_texture_view_id >= 0) {
        freeWGPUTextureView(ctx->current_texture_view_id);
        ctx->current_texture_view_id = -1;
    }
#else
    if (ctx->current_texture_view) {
        wgpuTextureViewRelease(ctx->current_texture_view);
        ctx->current_texture_view = NULL;
    }
#endif
}

// ============================================================================
// 工具函数 (Utility Functions)
// ============================================================================

// 清空渲染器实例缓冲区
void wcn_renderer_clear(WCN_Renderer* renderer) {
    if (!renderer) {
        return;
    }
    
    wcn_instance_buffer_clear(&renderer->cpu_instances);
}

// 调整渲染器视口大小
void wcn_renderer_resize(WCN_Renderer* renderer, uint32_t width, uint32_t height) {
    if (!renderer) {
        return;
    }
    
    // Update viewport dimensions
    renderer->width = width;
    renderer->height = height;
    
    // Update uniform buffer with new window size
    WCN_RendererUniforms uniform_data = {
        .viewport_size = {(float)width, (float)height},
        .instance_count = 0,
        .instance_offset = 0
    };
    
    wgpuQueueWriteBuffer(renderer->queue, renderer->uniform_buffer, 0, &uniform_data, sizeof(uniform_data));
}
