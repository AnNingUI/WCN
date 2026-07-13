#ifndef FULLSTACK_RENDER_H
#define FULLSTACK_RENDER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fullstack_core.h"
#include "fullstack_gpu.h"
#include "fullstack_present.h"

#define FS_RENDER_ABI_VERSION FS_ABI_VERSION(1, 0)

typedef struct FS_RenderContext FS_RenderContext;
typedef struct FS_RenderTarget FS_RenderTarget;
typedef struct FS_RenderFrame FS_RenderFrame;

typedef uint32_t FS_RenderFrameState;
#define FS_RENDER_FRAME_IDLE      ((FS_RenderFrameState)0u)
#define FS_RENDER_FRAME_RECORDING ((FS_RenderFrameState)1u)
#define FS_RENDER_FRAME_ENCODED   ((FS_RenderFrameState)2u)
#define FS_RENDER_FRAME_CANCELLED ((FS_RenderFrameState)3u)

typedef uint32_t FS_RenderPassStage;
#define FS_RENDER_PASS_BEFORE_SCENE         ((FS_RenderPassStage)0u)
#define FS_RENDER_PASS_SCENE                ((FS_RenderPassStage)1u)
#define FS_RENDER_PASS_AFTER_SCENE          ((FS_RenderPassStage)2u)
#define FS_RENDER_PASS_BEFORE_EFFECTS       ((FS_RenderPassStage)3u)
#define FS_RENDER_PASS_EFFECTS              ((FS_RenderPassStage)4u)
#define FS_RENDER_PASS_AFTER_EFFECTS        ((FS_RenderPassStage)5u)
#define FS_RENDER_PASS_BEFORE_PRESENT       ((FS_RenderPassStage)6u)
#define FS_RENDER_PASS_PRESENT              ((FS_RenderPassStage)7u)
#define FS_RENDER_PASS_AFTER_PRESENT_ENCODE ((FS_RenderPassStage)8u)

typedef struct FS_RenderTargetDesc {
    uint32_t struct_size;
    uint32_t abi_version;
    const FS_Allocator* allocator;
    WGPUTexture texture;
    WGPUTextureView view;
    WGPUTextureFormat format;
    uint32_t width;
    uint32_t height;
    FS_ColorSpace color_space;
    uint64_t generation;
} FS_RenderTargetDesc;

#define FS_RENDER_TARGET_DESC_INIT { sizeof(FS_RenderTargetDesc), \
    FS_RENDER_ABI_VERSION, NULL, NULL, NULL, WGPUTextureFormat_Undefined, \
    0, 0, FS_COLOR_SPACE_SRGB, 1 }

typedef struct FS_RenderContextDesc {
    uint32_t struct_size;
    uint32_t abi_version;
    const FS_Allocator* allocator;
    FS_DiagnosticSink diagnostics;
    FS_GpuContext* gpu;
    uint32_t width;
    uint32_t height;
    WGPUTextureFormat output_format;
    FS_ColorSpace output_color_space;
    uint32_t max_custom_passes;
} FS_RenderContextDesc;

#define FS_RENDER_CONTEXT_DESC_INIT { sizeof(FS_RenderContextDesc), \
    FS_RENDER_ABI_VERSION, NULL, {0}, NULL, 0, 0, \
    WGPUTextureFormat_BGRA8UnormSrgb, FS_COLOR_SPACE_SRGB, 32 }

typedef struct FS_RenderPassContext {
    uint32_t struct_size;
    FS_RenderPassStage stage;
    WGPUCommandEncoder encoder;
    FS_Core* core;
    const FS_CoreSceneOutput* scene;
    FS_RenderTarget* target;
    uint32_t width;
    uint32_t height;
} FS_RenderPassContext;

typedef FS_Result (FS_CALL *FS_RenderPassCallback)(
    const FS_RenderPassContext* context, void* user_data, FS_Error* error);

typedef struct FS_RenderPassDesc {
    uint32_t struct_size;
    FS_RenderPassStage stage;
    int32_t order;
    FS_RenderPassCallback callback;
    void* user_data;
} FS_RenderPassDesc;

typedef struct FS_CommandBatch {
    uint32_t struct_size;
    WGPUCommandBuffer command_buffer;
    FS_SubmissionToken submission;
    bool submitted;
} FS_CommandBatch;

FS_API FS_Result FS_CALL fs_render_target_import(
    const FS_RenderTargetDesc* desc, FS_RenderTarget** out_target, FS_Error* error);
FS_API FS_Result FS_CALL fs_render_target_create_texture(
    FS_GpuContext* gpu, const FS_RenderTargetDesc* desc,
    FS_RenderTarget** out_target, FS_Error* error);
FS_API FS_Result FS_CALL fs_render_target_update_import(
    FS_RenderTarget* target, const FS_RenderTargetDesc* desc, FS_Error* error);
FS_API void FS_CALL fs_render_target_destroy(FS_RenderTarget* target);
FS_API uint32_t FS_CALL fs_render_target_width(const FS_RenderTarget* target);
FS_API uint32_t FS_CALL fs_render_target_height(const FS_RenderTarget* target);
FS_API uint64_t FS_CALL fs_render_target_generation(const FS_RenderTarget* target);
FS_API WGPUTexture FS_CALL fs_render_target_texture(const FS_RenderTarget* target);
FS_API WGPUTextureView FS_CALL fs_render_target_view(const FS_RenderTarget* target);

FS_API FS_Result FS_CALL fs_render_context_create(
    const FS_RenderContextDesc* desc, FS_RenderContext** out_context,
    FS_Error* error);
FS_API void FS_CALL fs_render_context_destroy(FS_RenderContext* context);
FS_API FS_Core* FS_CALL fs_render_context_core(FS_RenderContext* context);
FS_API FS_Result FS_CALL fs_render_context_try_resize(
    FS_RenderContext* context, uint32_t width, uint32_t height, FS_Error* error);
FS_API FS_Result FS_CALL fs_render_context_recreate_device_resources(
    FS_RenderContext* context, FS_Error* error);

FS_API FS_Result FS_CALL fs_render_context_begin_frame(
    FS_RenderContext* context, FS_RenderTarget* target,
    FS_RenderFrame** out_frame, FS_Error* error);
FS_API FS_Result FS_CALL fs_render_frame_add_pass(
    FS_RenderFrame* frame, const FS_RenderPassDesc* desc, FS_Error* error);
FS_API FS_Result FS_CALL fs_render_frame_encode(
    FS_RenderFrame* frame, float clear_r, float clear_g, float clear_b,
    float clear_a, FS_CommandBatch* out_batch, FS_Error* error);
FS_API FS_Result FS_CALL fs_render_frame_cancel(FS_RenderFrame* frame,
                                                FS_Error* error);
FS_API FS_RenderFrameState FS_CALL fs_render_frame_state(
    const FS_RenderFrame* frame);

FS_API FS_Result FS_CALL fs_command_batch_submit(
    FS_RenderContext* context, FS_CommandBatch* batch, FS_Error* error);
FS_API void FS_CALL fs_command_batch_release(FS_CommandBatch* batch);

#ifdef __cplusplus
}
#endif
#endif
