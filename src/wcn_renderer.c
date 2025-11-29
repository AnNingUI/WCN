#include "wcn_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __EMSCRIPTEN__
#include "wcn_emcc_js.h"
#endif

// ============================================================================
// 渲染后端私有函数
// ============================================================================

static WGPUShaderModule wcn_create_shader_module(WGPUDevice device, const char* wgsl_code, const char* label) {
    // 创建WGSL源代码描述符
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
    
    WGPUShaderModuleDescriptor shader_desc = {
        .nextInChain = &wgsl_source.chain,
        .label = label
    };
    
    return wgpuDeviceCreateShaderModule(device, &shader_desc);
}

// ============================================================================
// Old batch rendering pipeline removed
// ============================================================================
// The old wcn_create_render_pipeline function has been removed.
// The unified renderer creates its own pipeline in wcn_create_renderer()
// (see wcn_instance.c)
// ============================================================================

// Old buffer creation function removed - buffers are now managed by unified renderer

// Old bind group creation function removed - bind groups are now managed by unified renderer

// ============================================================================
// 公共渲染后端初始化函数
// ============================================================================

bool wcn_initialize_renderer(WCN_Context* ctx) {
    // printf("[wcn_initialize_renderer] Line 1: Start\n");
    if (!ctx) {
        // printf("[wcn_initialize_renderer] Line 2: ctx is NULL\n");
        return false;
    }
    
    // printf("[wcn_initialize_renderer] Line 3: Creating SDF Atlas\n");
    // 创建 SDF Atlas (4096x4096 for better capacity)
    // 增加到 4096 以支持更多字形和大字号
    ctx->sdf_atlas = wcn_create_sdf_atlas(ctx, 4096, 4096);
    // printf("[wcn_initialize_renderer] Line 4: SDF Atlas created\n");
    if (!ctx->sdf_atlas) {
        // printf("[wcn_initialize_renderer] Line 5: SDF Atlas creation failed\n");
        return false;
    }
    // printf("[wcn_initialize_renderer] Line 6: SDF Atlas OK\n");
    
    // 创建 SDF sampler
    // printf("[wcn_initialize_renderer] Line 7: Creating sampler\n");
#ifdef __EMSCRIPTEN__
    // WASM: 使用 EM_JS 函数创建 sampler，确保 LOD 参数正确
    ctx->sdf_sampler = wasm_create_sampler(ctx->device, "WCN SDF Sampler");
#else
    // 原生: label 是 WGPUStringView
    const char* label_str = "WCN SDF Sampler";
    WGPUSamplerDescriptor sampler_desc = {
        .nextInChain = NULL,
        .label = {
            .data = label_str,
            .length = strlen(label_str)
        },
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .addressModeW = WGPUAddressMode_ClampToEdge,
        .magFilter = WGPUFilterMode_Linear,
        .minFilter = WGPUFilterMode_Linear,
        .mipmapFilter = WGPUMipmapFilterMode_Linear,
        .lodMinClamp = 0.0f,
        .lodMaxClamp = 32.0f,
        .compare = WGPUCompareFunction_Undefined,
        .maxAnisotropy = 1
    };
    ctx->sdf_sampler = wgpuDeviceCreateSampler(ctx->device, &sampler_desc);
#endif
    // printf("[wcn_initialize_renderer] Line 8: Sampler created\n");
    
    if (!ctx->sdf_sampler) {
        // printf("[wcn_initialize_renderer] Line 9: Sampler creation failed\n");
        wcn_destroy_sdf_atlas(ctx->sdf_atlas);
        return false;
    }
    // printf("[wcn_initialize_renderer] Line 10: Sampler OK\n");
    
    // 创建统一渲染器
    // printf("[wcn_initialize_renderer] Line 11: Creating unified renderer\n");
    ctx->renderer = wcn_create_renderer(
        ctx->device,
        ctx->queue,
        ctx->surface_format,
        ctx->width,
        ctx->height
    );
    // printf("[wcn_initialize_renderer] Line 12: Renderer created\n");
    if (!ctx->renderer) {
        // printf("[wcn_initialize_renderer] Line 13: Renderer creation failed\n");
        wgpuSamplerRelease(ctx->sdf_sampler);
        wcn_destroy_sdf_atlas(ctx->sdf_atlas);
        return false;
    }
    // printf("[wcn_initialize_renderer] Line 14: Renderer OK\n");
    
    // printf("[wcn_initialize_renderer] Line 15: Success!\n");
    return true;
}

// ============================================================================
// 渲染命令管理函数
// ============================================================================
#ifdef __EMSCRIPTEN__
bool wcn_begin_render_pass(WCN_Context* ctx, WGPUTextureViewID texture_view) {
#else
bool wcn_begin_render_pass(WCN_Context* ctx, WGPUTextureView texture_view) {
#endif
    if (!ctx || !texture_view) return false;

    if (!ctx->current_command_encoder) {
        WGPUCommandEncoderDescriptor encoder_desc = {
            .nextInChain = NULL,
            .label = "WCN Command Encoder"
        };
        ctx->current_command_encoder = wgpuDeviceCreateCommandEncoder(ctx->device, &encoder_desc);
        if (!ctx->current_command_encoder) return false;
    }

    bool is_first_pass = ctx->current_render_pass == NULL && !ctx->render_pass_needs_begin;
    ctx->pending_color_load_op = is_first_pass ? WGPULoadOp_Clear : WGPULoadOp_Load;
    ctx->pending_clear_color = (WGPUColor){0.9, 0.9, 0.9, 1.0};
    ctx->render_pass_needs_begin = true;

#ifdef __EMSCRIPTEN__
    ctx->current_texture_view_id = texture_view;
#else
    ctx->current_texture_view = texture_view;
#endif
    return true;
}

void wcn_end_render_pass(WCN_Context* ctx) {
    if (!ctx) return;

    if (ctx->current_render_pass) {
        wgpuRenderPassEncoderEnd(ctx->current_render_pass);
        wgpuRenderPassEncoderRelease(ctx->current_render_pass);
        ctx->current_render_pass = NULL;
    }

    ctx->render_pass_needs_begin = false;
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

void wcn_submit_commands(WCN_Context* ctx) {
    if (!ctx || !ctx->current_command_encoder) return;

    // 文字渲染现在使用 SDF Atlas 系统，在 wcn_render_batches() 中处理
    // 不再需要单独的 flush_text_commands 调用

    WGPUCommandBufferDescriptor command_buffer_desc = {
        .nextInChain = NULL,
        .label = "WCN Command Buffer"
    };

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(ctx->current_command_encoder, &command_buffer_desc);
    if (command_buffer) {
        wgpuQueueSubmit(ctx->queue, 1, &command_buffer);
        wgpuCommandBufferRelease(command_buffer);
    }

    wgpuCommandEncoderRelease(ctx->current_command_encoder);
    ctx->current_command_encoder = NULL;
}

// ============================================================================
// 顶点/索引缓冲区管理
// ============================================================================

bool wcn_write_vertex_data(WCN_Context* ctx, const void* data, size_t size, size_t* out_offset) {
    if (!ctx || !data || size == 0) return false;

    // 检查缓冲区是否有足够空间
    if (ctx->vertex_buffer_offset + size > 1024 * 1024) {
        // 缓冲区已满，重置偏移量（简单循环缓冲区）
        ctx->vertex_buffer_offset = 0;
    }

    // 记录当前偏移量
    if (out_offset) {
        *out_offset = ctx->vertex_buffer_offset;
    }

    // 写入顶点数据
    wgpuQueueWriteBuffer(ctx->queue, ctx->vertex_buffer, ctx->vertex_buffer_offset, data, size);

    // 更新偏移量
    ctx->vertex_buffer_offset += size;
    return true;
}

bool wcn_write_index_data(WCN_Context* ctx, const void* data, size_t size, size_t* out_offset) {
    if (!ctx || !data || size == 0) return false;

    // 检查缓冲区是否有足够空间
    if (ctx->index_buffer_offset + size > 1024 * 1024) {
        // 缓冲区已满，重置偏移量（简单循环缓冲区）
        ctx->index_buffer_offset = 0;
    }

    // 记录当前偏移量
    if (out_offset) {
        *out_offset = ctx->index_buffer_offset;
    }

    // 写入索引数据
    wgpuQueueWriteBuffer(ctx->queue, ctx->index_buffer, ctx->index_buffer_offset, data, size);

    // 更新偏移量
    ctx->index_buffer_offset += size;
    return true;
}
