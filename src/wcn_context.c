#include "wcn_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include "wcn_emcc_js.h"
// 鐩存帴澹版槑 Emscripten WebGPU 鍑芥暟锛岄伩鍏嶅寘鍚?html5_webgpu.h 鐨勭被鍨嬪啿绐?
extern WGPUDevice emscripten_webgpu_get_device(void);
#endif

// ============================================================================
// 绉佹湁杈呭姪鍑芥暟
// ============================================================================

static bool wcn_create_state_stack(WCN_Context* ctx, uint32_t max_states) {
    WCN_StateStack* stack = &ctx->state_stack;
    
    // 鍒嗛厤 CPU 绔姸鎬佹暟缁?
    stack->states = (WCN_GPUState*)malloc(sizeof(WCN_GPUState) * max_states);
    if (!stack->states) {
        return false;
    }
    
    // 鍒濆鍖栭粯璁ょ姸鎬?
    memset(&stack->states[0], 0, sizeof(WCN_GPUState));
    
    // 璁剧疆鍗曚綅鐭╅樀
    stack->states[0].transform_matrix[0] = 1.0f;
    stack->states[0].transform_matrix[5] = 1.0f;
    stack->states[0].transform_matrix[10] = 1.0f;
    stack->states[0].transform_matrix[15] = 1.0f;
    
    // 璁剧疆榛樿棰滆壊涓洪粦鑹?
    stack->states[0].fill_color = 0xFF000000;
    stack->states[0].stroke_color = 0xFF000000;
    stack->states[0].stroke_width = 1.0f;
    stack->states[0].global_alpha = 1.0f;
    
    // 璁剧疆榛樿绾垮附鍜岃繛鎺ユ牱寮?
    stack->states[0].line_cap = WCN_LINE_CAP_BUTT;      // 榛樿骞冲ご
    stack->states[0].line_join = WCN_LINE_JOIN_MITER;   // 榛樿灏栬
    stack->states[0].miter_limit = 10.0f;               // 榛樿鏂滄帴闄愬埗
    
    stack->current_state = 0;
    stack->max_states = max_states;
    
    // 鍒涘缓 GPU 鐘舵€佺紦鍐插尯锛堟敼涓篣niform锛屾敮鎸佸姩鎬佸亸绉伙級
    // 棰勫垎閰嶈冻澶熺殑绌洪棿鐢ㄤ簬澶氫釜缁樺埗璋冪敤
    // WebGPU 瑕佹眰鍔ㄦ€佸亸绉诲繀椤绘槸 256 瀛楄妭鐨勫€嶆暟
    #define MAX_DRAW_CALLS_PER_FRAME 1000
    #define GPU_STATE_STRIDE 256  // 姣忎釜鐘舵€佸崰鐢?256 瀛楄妭锛圵ebGPU 鍔ㄦ€佸亸绉昏姹傦級
    
#ifdef __EMSCRIPTEN__
    // WASM: 浣跨敤 EM_JS 鍑芥暟鍒涘缓 buffer锛岀‘淇?usage 姝ｇ‘浼犻€?
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
// 鍏叡 API 瀹炵幇
// ============================================================================

WCN_Context* wcn_create_context(WCN_GPUResources* gpu_resources) {
    if (!gpu_resources) {
        return NULL;
    }
    
    // 鍒嗛厤涓婁笅鏂囧唴瀛?
    WCN_Context* ctx = (WCN_Context*)malloc(sizeof(WCN_Context));
    if (!ctx) {
        return NULL;
    }
    
    // 鍒濆鍖栦笂涓嬫枃
    memset(ctx, 0, sizeof(WCN_Context));
    
    // 璁剧疆 WebGPU 璧勬簮
    ctx->instance = gpu_resources->instance;
    ctx->device = gpu_resources->device;
    ctx->queue = gpu_resources->queue;
    ctx->surface = gpu_resources->surface;
    
#ifdef __EMSCRIPTEN__
    // 鍦?Emscripten 鐜涓紝濡傛灉 device 涓?0锛屼粠 Emscripten 鑾峰彇棰勫垵濮嬪寲鐨勮澶?
    if (!ctx->device) {
        // Emscripten 鎻愪緵浜?emscripten_webgpu_get_device() 鏉ヨ幏鍙栭鍒濆鍖栫殑璁惧
        extern WGPUDevice emscripten_webgpu_get_device(void);
        ctx->device = emscripten_webgpu_get_device();
        
        if (!ctx->device) {
            // printf("ERROR: Failed to get Emscripten WebGPU device\n");
            free(ctx);
            return NULL;
        }
        
        // printf("Emscripten WebGPU mode: obtained device from Emscripten\n");
    }
    
    // 鍚屾牱锛屽鏋?queue 涓?0锛屼粠 device 鑾峰彇闃熷垪
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
    // 鍘熺敓鏋勫缓蹇呴』鎻愪緵鏈夋晥鐨?device
    if (!ctx->device) {
        free(ctx);
        return NULL;
    }
#endif
    // 娉ㄦ剰锛歴urface_format 搴旇鍦?wcn_begin_frame 涓缃负瀹為檯鐨勮〃闈㈡牸寮?
    ctx->surface_format = WGPUTextureFormat_Undefined;
    
    // 鍒涘缓鐘舵€佹爤
    // 澧炲姞鍒?512 浠ユ敮鎸佸ぇ閲忔枃瀛楁覆鏌擄紙姣忎釜瀛楃涓€涓Ы浣嶏級
    if (!wcn_create_state_stack(ctx, 512)) {
        wcn_destroy_context(ctx);
        return NULL;
    }
    
    // 寤惰繜鍒濆鍖栨覆鏌撳悗绔紙绛夊埌鐭ラ亾surface format鏃跺啀鍒濆鍖栵級
    ctx->renderer_initialized = false;
    
    // 鍒濆鍖栧叾浠栫姸鎬?
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
    
    // 鍒濆鍖栨枃鏈覆鏌撶姸鎬?
    ctx->current_font_face = NULL;
    ctx->current_font_size = 10.0f;  // 榛樿瀛楀彿
    ctx->text_align = WCN_TEXT_ALIGN_LEFT;
    ctx->text_baseline = WCN_TEXT_BASELINE_ALPHABETIC;
    ctx->font_fallbacks = NULL;
    ctx->font_fallback_count = 0;
    ctx->font_fallback_capacity = 0;

    ctx->text_command_count = 0;
    ctx->text_command_capacity = 0;
    
    return ctx;
}

void wcn_destroy_context(WCN_Context* ctx) {
    if (!ctx) return;

    // 鏂囧瓧娓叉煋鍛戒护闃熷垪宸茶绉婚櫎锛屼娇鐢?SDF Atlas 绯荤粺鏇夸唬
    
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
    
    // 销毁 GPU Native 路径
    if (ctx->gpu_path) {
        if (ctx->gpu_path->commands) {
            free(ctx->gpu_path->commands);
        }
        free(ctx->gpu_path);
        ctx->gpu_path = NULL;
    }

    // 閿€姣佺姸鎬佹爤
    wcn_destroy_state_stack(ctx);
    
    // 鏃х殑娓叉煋绠＄嚎璧勬簮宸茶绉婚櫎锛岀幇鍦ㄤ娇鐢ㄧ粺涓€娓叉煋鍣?
    // (render_pipeline, pipeline_layout, bind_group_layout 宸插垹闄?
    
    // 閿€姣佺紦鍐插尯
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

    // 閿€姣佺粺涓€娓叉煋鍣?
    if (ctx->renderer) {
        wcn_destroy_renderer(ctx->renderer);
        ctx->renderer = NULL;
    }

    // 閿€姣?SDF Atlas
    if (ctx->sdf_atlas) {
        wcn_destroy_sdf_atlas(ctx->sdf_atlas);
        ctx->sdf_atlas = NULL;
    }

    // 閿€姣?SDF 閲囨牱鍣ㄥ拰缁戝畾缁?
    if (ctx->sdf_sampler) {
        wgpuSamplerRelease(ctx->sdf_sampler);
        ctx->sdf_sampler = NULL;
    }

    wcn_shutdown_image_manager(ctx);

    if (ctx->sdf_bind_group) {
        wgpuBindGroupRelease(ctx->sdf_bind_group);
        ctx->sdf_bind_group = NULL;
    }

    if (ctx->sdf_bind_group_layout) {
        wgpuBindGroupLayoutRelease(ctx->sdf_bind_group_layout);
        ctx->sdf_bind_group_layout = NULL;
    }
        if (ctx->font_fallbacks) {
        free(ctx->font_fallbacks);
        ctx->font_fallbacks = NULL;
        ctx->font_fallback_count = 0;
        ctx->font_fallback_capacity = 0;
    }


    // 閲婃斁 WebGPU 璧勬簮
    if (ctx->instance) wgpuInstanceRelease(ctx->instance);
    if (ctx->device) wgpuDeviceRelease(ctx->device);
    if (ctx->queue) wgpuQueueRelease(ctx->queue);
    if (ctx->surface) wgpuSurfaceRelease(ctx->surface);
    
    // 閲婃斁涓婁笅鏂囧唴瀛?
    free(ctx);
}

void wcn_begin_frame(WCN_Context* ctx, uint32_t width, uint32_t height, WGPUTextureFormat surface_format) {
    if (!ctx || ctx->in_frame) return;

    // 鍙湪绗竴甯ф墦鍗?
    if (ctx->frame_count == 0) {
        // printf("wcn_begin_frame: width=%d, height=%d, surface_format=%d\n", width, height, surface_format);
    }

    ctx->frame_width = width;
    ctx->frame_height = height;
    ctx->width = width;
    ctx->height = height;
    ctx->surface_format = surface_format;

    // 寤惰繜鍒濆鍖栨覆鏌撳悗绔?
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
    // 閲嶇疆缂撳啿鍖哄亸绉婚噺浠ュ紑濮嬫柊鐨勪竴甯?
    ctx->vertex_buffer_offset = 0;
    ctx->index_buffer_offset = 0;
    ctx->current_draw_call = 0;  // 閲嶇疆缁樺埗璋冪敤璁℃暟
    ctx->next_gpu_state_slot = 1;  // 浠?1 寮€濮嬪垎閰嶏紝淇濈暀妲戒綅 0 浣滀负榛樿鐘舵€?
    
    // 閲嶇疆鐘舵€佹爤鍒板垵濮嬬姸鎬侊紙鍏抽敭锛侊級
    ctx->state_stack.current_state = 0;
    
    // 閲嶇疆鐘舵€?0 涓哄崟浣嶇煩闃?
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

    // 浣跨敤缁熶竴娓叉煋鍣ㄦ覆鏌撴墍鏈夊疄渚?
    if (ctx->renderer && (ctx->current_render_pass || ctx->render_pass_needs_begin)) {
        // 鍒锋柊 SDF Atlas锛堝鏋滄湁鏇存柊锛?
        if (ctx->sdf_atlas && ctx->sdf_atlas->dirty) {
            wcn_flush_sdf_atlas(ctx);
        }
        
        // 娓叉煋鎵€鏈夌疮绉殑瀹炰緥
        wcn_renderer_render(
            ctx,
            ctx->sdf_atlas ? ctx->sdf_atlas->texture_view : NULL,
            ctx->image_atlas ? ctx->image_atlas->texture_view : NULL
        );
        
        // 娓呯┖瀹炰緥缂撳啿鍖?
        wcn_renderer_clear(ctx->renderer);
        
        // 缁撴潫render pass骞舵彁浜ゅ懡浠?
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
        // 鐘舵€佹爤婧㈠嚭
        // printf("ERROR: State stack overflow!\n");
        return;
    }
    
    // 澶嶅埗褰撳墠鐘舵€佸埌涓嬩竴涓姸鎬?
    memcpy(&stack->states[stack->current_state + 1], 
           &stack->states[stack->current_state], 
           sizeof(WCN_GPUState));
    
    stack->current_state++;
}

void wcn_restore(WCN_Context* ctx) {
    if (!ctx || !ctx->in_frame) return;
    
    WCN_StateStack* stack = &ctx->state_stack;
    if (stack->current_state == 0) {
        // 宸茬粡鍦ㄦ爤搴?
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
// 杈呭姪鍑芥暟
// ============================================================================

WGPUTextureFormat wcn_get_surface_format(WCN_Context* ctx) {
    if (!ctx) return WGPUTextureFormat_Undefined;
    return ctx->surface_format;
}

void wcn_set_surface_format(WCN_Context* ctx, WGPUTextureFormat format) {
    if (!ctx) return;
    ctx->surface_format = format;
}



