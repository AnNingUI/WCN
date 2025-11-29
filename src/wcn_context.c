#include "wcn_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include "wcn_emcc_js.h"
// 直接声明 Emscripten WebGPU 函数，避免包含 html5_webgpu.h 的类型冲突
extern WGPUDevice emscripten_webgpu_get_device(void);
#endif

// ============================================================================
// 私有辅助函数
// ============================================================================

static bool wcn_create_state_stack(WCN_Context* ctx, uint32_t max_states) {
    WCN_StateStack* stack = &ctx->state_stack;
    
    // 分配 CPU 端状态数组
    stack->states = (WCN_GPUState*)malloc(sizeof(WCN_GPUState) * max_states);
    if (!stack->states) {
        return false;
    }
    
    // 初始化默认状态
    memset(&stack->states[0], 0, sizeof(WCN_GPUState));
    
    // 设置单位矩阵
    stack->states[0].transform_matrix[0] = 1.0f;
    stack->states[0].transform_matrix[5] = 1.0f;
    stack->states[0].transform_matrix[10] = 1.0f;
    stack->states[0].transform_matrix[15] = 1.0f;
    
    // 设置默认颜色为黑色
    stack->states[0].fill_color = 0xFF000000;
    stack->states[0].stroke_color = 0xFF000000;
    stack->states[0].stroke_width = 1.0f;
    stack->states[0].global_alpha = 1.0f;
    
    // 设置默认线帽和连接样式
    stack->states[0].line_cap = WCN_LINE_CAP_BUTT;      // 默认平头
    stack->states[0].line_join = WCN_LINE_JOIN_MITER;   // 默认尖角
    stack->states[0].miter_limit = 10.0f;               // 默认斜接限制
    
    stack->current_state = 0;
    stack->max_states = max_states;
    
    // 创建 GPU 状态缓冲区（改为Uniform，支持动态偏移）
    // 预分配足够的空间用于多个绘制调用
    // WebGPU 要求动态偏移必须是 256 字节的倍数
    #define MAX_DRAW_CALLS_PER_FRAME 1000
    #define GPU_STATE_STRIDE 256  // 每个状态占用 256 字节（WebGPU 动态偏移要求）
    
#ifdef __EMSCRIPTEN__
    // WASM: 使用 EM_JS 函数创建 buffer，确保 usage 正确传递
    stack->state_buffer = wasm_create_buffer(
        ctx->device,
        "WCN State Buffer",
        GPU_STATE_STRIDE * MAX_DRAW_CALLS_PER_FRAME,
        2 | 4  // Uniform | CopyDst
    );
#else
    WGPUBufferDescriptor buffer_desc = {
        .nextInChain = NULL,
        .label = "WCN State Buffer",
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size = GPU_STATE_STRIDE * MAX_DRAW_CALLS_PER_FRAME,
        .mappedAtCreation = false
    };
    
    stack->state_buffer = wgpuDeviceCreateBuffer(ctx->device, &buffer_desc);
#endif
    if (!stack->state_buffer) {
        free(stack->states);
        return false;
    }
    
    return true;
}

static void wcn_destroy_state_stack(WCN_Context* ctx) {
    WCN_StateStack* stack = &ctx->state_stack;
    
    if (stack->states) {
        free(stack->states);
        stack->states = NULL;
    }
    
    if (stack->state_buffer) {
        wgpuBufferDestroy(stack->state_buffer);
        wgpuBufferRelease(stack->state_buffer);
        stack->state_buffer = NULL;
    }
    
    if (stack->bind_group) {
        wgpuBindGroupRelease(stack->bind_group);
        stack->bind_group = NULL;
    }
}

// ============================================================================
// 公共 API 实现
// ============================================================================

WCN_Context* wcn_create_context(WCN_GPUResources* gpu_resources) {
    if (!gpu_resources) {
        return NULL;
    }
    
    // 分配上下文内存
    WCN_Context* ctx = (WCN_Context*)malloc(sizeof(WCN_Context));
    if (!ctx) {
        return NULL;
    }
    
    // 初始化上下文
    memset(ctx, 0, sizeof(WCN_Context));
    
    // 设置 WebGPU 资源
    ctx->instance = gpu_resources->instance;
    ctx->device = gpu_resources->device;
    ctx->queue = gpu_resources->queue;
    ctx->surface = gpu_resources->surface;
    
#ifdef __EMSCRIPTEN__
    // 在 Emscripten 环境中，如果 device 为 0，从 Emscripten 获取预初始化的设备
    if (!ctx->device) {
        // Emscripten 提供了 emscripten_webgpu_get_device() 来获取预初始化的设备
        extern WGPUDevice emscripten_webgpu_get_device(void);
        ctx->device = emscripten_webgpu_get_device();
        
        if (!ctx->device) {
            // printf("ERROR: Failed to get Emscripten WebGPU device\n");
            free(ctx);
            return NULL;
        }
        
        // printf("Emscripten WebGPU mode: obtained device from Emscripten\n");
    }
    
    // 同样，如果 queue 为 0，从 device 获取队列
    if (!ctx->queue && ctx->device) {
        ctx->queue = wgpuDeviceGetQueue(ctx->device);
        if (!ctx->queue) {
            // printf("ERROR: Failed to get queue from device\n");
            free(ctx);
            return NULL;
        }
        // printf("Emscripten WebGPU mode: obtained queue from device\n");
    }
#else
    // 原生构建必须提供有效的 device
    if (!ctx->device) {
        free(ctx);
        return NULL;
    }
#endif
    // 注意：surface_format 应该在 wcn_begin_frame 中设置为实际的表面格式
    ctx->surface_format = WGPUTextureFormat_Undefined;
    
    // 创建状态栈
    // 增加到 512 以支持大量文字渲染（每个字符一个槽位）
    if (!wcn_create_state_stack(ctx, 512)) {
        wcn_destroy_context(ctx);
        return NULL;
    }
    
    // 延迟初始化渲染后端（等到知道surface format时再初始化）
    ctx->renderer_initialized = false;
    
    // 初始化其他状态
    ctx->in_frame = false;
    ctx->frame_width = 0;
    ctx->frame_height = 0;
    ctx->frame_count = 0;
    ctx->font_decoder = NULL;
    ctx->image_decoder = NULL;
    ctx->current_path = NULL;
#ifdef __EMSCRIPTEN__ 
    ctx->current_texture_view_id = -1;
#else
    ctx->current_texture_view = NULL;
#endif    
    ctx->current_command_encoder = NULL;
    ctx->current_render_pass = NULL;
    ctx->render_pass_needs_begin = false;
    ctx->pending_color_load_op = WGPULoadOp_Clear;
    ctx->pending_clear_color = (WGPUColor){0.9, 0.9, 0.9, 1.0};
    ctx->vertex_buffer_offset = 0;
    ctx->index_buffer_offset = 0;
    ctx->user_data = NULL;
    
    // 初始化文本渲染状态
    ctx->current_font_face = NULL;
    ctx->current_font_size = 10.0f;  // 默认字号
    ctx->text_align = WCN_TEXT_ALIGN_LEFT;
    ctx->text_baseline = WCN_TEXT_BASELINE_ALPHABETIC;
    
    // GPU SDF 渲染器将在首次使用时延迟创建
    ctx->gpu_sdf_renderer = NULL;
    
    // 初始化文字渲染命令队列
    ctx->text_commands = NULL;
    ctx->text_command_count = 0;
    ctx->text_command_capacity = 0;
    
    return ctx;
}

void wcn_destroy_context(WCN_Context* ctx) {
    if (!ctx) return;

    // 文字渲染命令队列已被移除，使用 SDF Atlas 系统替代
    // 不再需要清理 text_commands
    
    // 销毁当前路径
    if (ctx->current_path) {
        if (ctx->current_path->points) {
            free(ctx->current_path->points);
        }
        if (ctx->current_path->commands) {
            free(ctx->current_path->commands);
        }
        free(ctx->current_path);
        ctx->current_path = NULL;
    }

    // 销毁状态栈
    wcn_destroy_state_stack(ctx);
    
    // 旧的渲染管线资源已被移除，现在使用统一渲染器
    // (render_pipeline, pipeline_layout, bind_group_layout 已删除)
    
    // 销毁缓冲区
    if (ctx->vertex_buffer) {
        wgpuBufferDestroy(ctx->vertex_buffer);
        wgpuBufferRelease(ctx->vertex_buffer);
        ctx->vertex_buffer = NULL;
    }
    
    if (ctx->index_buffer) {
        wgpuBufferDestroy(ctx->index_buffer);
        wgpuBufferRelease(ctx->index_buffer);
        ctx->index_buffer = NULL;
    }
    
    if (ctx->uniform_buffer) {
        wgpuBufferDestroy(ctx->uniform_buffer);
        wgpuBufferRelease(ctx->uniform_buffer);
        ctx->uniform_buffer = NULL;
    }
    
    // GPU SDF 渲染器已被移除，使用 SDF Atlas 系统替代
    ctx->gpu_sdf_renderer = NULL;

    // 销毁统一渲染器
    if (ctx->renderer) {
        wcn_destroy_renderer(ctx->renderer);
        ctx->renderer = NULL;
    }

    // 销毁 SDF Atlas
    if (ctx->sdf_atlas) {
        wcn_destroy_sdf_atlas(ctx->sdf_atlas);
        ctx->sdf_atlas = NULL;
    }

    // 销毁 SDF 采样器和绑定组
    if (ctx->sdf_sampler) {
        wgpuSamplerRelease(ctx->sdf_sampler);
        ctx->sdf_sampler = NULL;
    }

    if (ctx->sdf_bind_group) {
        wgpuBindGroupRelease(ctx->sdf_bind_group);
        ctx->sdf_bind_group = NULL;
    }

    if (ctx->sdf_bind_group_layout) {
        wgpuBindGroupLayoutRelease(ctx->sdf_bind_group_layout);
        ctx->sdf_bind_group_layout = NULL;
    }
    
    // 释放 WebGPU 资源
    if (ctx->instance) wgpuInstanceRelease(ctx->instance);
    if (ctx->device) wgpuDeviceRelease(ctx->device);
    if (ctx->queue) wgpuQueueRelease(ctx->queue);
    if (ctx->surface) wgpuSurfaceRelease(ctx->surface);
    
    // 释放上下文内存
    free(ctx);
}

void wcn_begin_frame(WCN_Context* ctx, uint32_t width, uint32_t height, WGPUTextureFormat surface_format) {
    if (!ctx || ctx->in_frame) return;

    // 只在第一帧打印
    if (ctx->frame_count == 0) {
        // printf("wcn_begin_frame: width=%d, height=%d, surface_format=%d\n", width, height, surface_format);
    }

    ctx->frame_width = width;
    ctx->frame_height = height;
    ctx->width = width;
    ctx->height = height;
    ctx->surface_format = surface_format;

    // 延迟初始化渲染后端
    if (!ctx->renderer_initialized) {
        // printf("Initializing renderer...\n");
        if (!wcn_initialize_renderer(ctx)) {
            // printf("Failed to initialize renderer with surface format %d\n", surface_format);
            return;
        }
        ctx->renderer_initialized = true;
        // printf("Renderer initialized successfully\n");
    }

    ctx->in_frame = true;
    // 重置缓冲区偏移量以开始新的一帧
    ctx->vertex_buffer_offset = 0;
    ctx->index_buffer_offset = 0;
    ctx->current_draw_call = 0;  // 重置绘制调用计数
    ctx->next_gpu_state_slot = 1;  // 从 1 开始分配，保留槽位 0 作为默认状态
    
    // 重置状态栈到初始状态（关键！）
    ctx->state_stack.current_state = 0;
    
    // 重置状态 0 为单位矩阵
    memset(&ctx->state_stack.states[0], 0, sizeof(WCN_GPUState));
    ctx->state_stack.states[0].transform_matrix[0] = 1.0f;
    ctx->state_stack.states[0].transform_matrix[5] = 1.0f;
    ctx->state_stack.states[0].transform_matrix[10] = 1.0f;
    ctx->state_stack.states[0].transform_matrix[15] = 1.0f;
    ctx->state_stack.states[0].fill_color = 0xFF000000;
    ctx->state_stack.states[0].stroke_color = 0xFF000000;
    ctx->state_stack.states[0].stroke_width = 1.0f;
    ctx->state_stack.states[0].global_alpha = 1.0f;
    ctx->state_stack.states[0].line_cap = WCN_LINE_CAP_BUTT;
    ctx->state_stack.states[0].line_join = WCN_LINE_JOIN_MITER;
    ctx->state_stack.states[0].miter_limit = 10.0f;
}

void wcn_end_frame(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;

    // 使用统一渲染器渲染所有实例
    if (ctx->renderer && (ctx->current_render_pass || ctx->render_pass_needs_begin)) {
        // 刷新 SDF Atlas（如果有更新）
        if (ctx->sdf_atlas && ctx->sdf_atlas->dirty) {
            wcn_flush_sdf_atlas(ctx);
        }
        
        // 渲染所有累积的实例
        wcn_renderer_render(
            ctx,
            ctx->sdf_atlas ? ctx->sdf_atlas->texture_view : NULL
        );
        
        // 清空实例缓冲区
        wcn_renderer_clear(ctx->renderer);
        
        // 结束render pass并提交命令
        wcn_end_render_pass(ctx);
        wcn_submit_commands(ctx);
    }

    ctx->in_frame = false;
    ctx->frame_count++;
}

void wcn_save(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;
    
    WCN_StateStack* stack = &ctx->state_stack;
    if (stack->current_state + 1 >= stack->max_states) {
        // 状态栈溢出
        // printf("ERROR: State stack overflow!\n");
        return;
    }
    
    // 复制当前状态到下一个状态
    memcpy(&stack->states[stack->current_state + 1], 
           &stack->states[stack->current_state], 
           sizeof(WCN_GPUState));
    
    stack->current_state++;
}

void wcn_restore(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;
    
    WCN_StateStack* stack = &ctx->state_stack;
    if (stack->current_state == 0) {
        // 已经在栈底
        return;
    }
    
    stack->current_state--;
}

void wcn_register_font_decoder(WCN_Context* ctx, WCN_FontDecoder* decoder) {
    if (!ctx || !decoder) return;
    ctx->font_decoder = decoder;
}

void wcn_register_image_decoder(WCN_Context* ctx, WCN_ImageDecoder* decoder) {
    if (!ctx || !decoder) return;
    ctx->image_decoder = decoder;
}
// ============================================================================
// 辅助函数
// ============================================================================

WGPUTextureFormat wcn_get_surface_format(WCN_Context* ctx) {
    if (!ctx) return WGPUTextureFormat_Undefined;
    return ctx->surface_format;
}

void wcn_set_surface_format(WCN_Context* ctx, WGPUTextureFormat format) {
    if (!ctx) return;
    ctx->surface_format = format;
}
