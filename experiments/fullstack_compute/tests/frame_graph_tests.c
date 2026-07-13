#include "fullstack_render.h"

#include <assert.h>

typedef struct Trace {
    uint32_t values[16];
    uint32_t count;
} Trace;

typedef struct TraceItem {
    Trace* trace;
    uint32_t value;
} TraceItem;

static FS_Result FS_CALL trace_pass(const FS_RenderPassContext* context,
                                    void* data, FS_Error* error) {
    (void)error;
    TraceItem* item = (TraceItem*)data;
    assert(context && context->encoder && context->core && context->target);
    if (context->stage == FS_RENDER_PASS_BEFORE_SCENE)
        assert(context->scene == NULL);
    else
        assert(context->scene && context->scene->view);
    item->trace->values[item->trace->count++] = item->value;
    return FS_RESULT_OK;
}

int main(void) {
    FS_Error error = FS_ERROR_INIT;
    FS_GpuContextDesc gpu_desc = FS_GPU_CONTEXT_DESC_INIT;
    FS_GpuContext* gpu = NULL;
    assert(fs_gpu_context_create(&gpu_desc, &gpu, &error) == FS_RESULT_OK);
    FS_RenderContextDesc context_desc = FS_RENDER_CONTEXT_DESC_INIT;
    context_desc.gpu = gpu;
    context_desc.width = 32;
    context_desc.height = 32;
    context_desc.output_format = WGPUTextureFormat_RGBA8Unorm;
    context_desc.output_color_space = FS_COLOR_SPACE_LINEAR_SRGB;
    FS_RenderContext* context = NULL;
    assert(fs_render_context_create(&context_desc, &context, &error) ==
           FS_RESULT_OK);
    FS_RenderTargetDesc target_desc = FS_RENDER_TARGET_DESC_INIT;
    target_desc.width = 32;
    target_desc.height = 32;
    target_desc.format = WGPUTextureFormat_RGBA8Unorm;
    target_desc.color_space = FS_COLOR_SPACE_LINEAR_SRGB;
    FS_RenderTarget* target = NULL;
    assert(fs_render_target_create_texture(gpu, &target_desc, &target, &error) ==
           FS_RESULT_OK);

    fs_core_begin_commands(fs_render_context_core(context));
    assert(fs_cmd_rect(fs_render_context_core(context), 0, 0, 32, 32, 0,
                       0xffffffffu));
    FS_RenderFrame* frame = NULL;
    assert(fs_render_context_begin_frame(context, target, &frame, &error) ==
           FS_RESULT_OK);
    Trace trace = {0};
    TraceItem items[] = {
        {&trace, 1}, {&trace, 2}, {&trace, 3}, {&trace, 4}, {&trace, 5}};
    FS_RenderPassDesc passes[] = {
        {sizeof(FS_RenderPassDesc), FS_RENDER_PASS_AFTER_EFFECTS, 0,
         trace_pass, &items[3]},
        {sizeof(FS_RenderPassDesc), FS_RENDER_PASS_BEFORE_SCENE, 10,
         trace_pass, &items[1]},
        {sizeof(FS_RenderPassDesc), FS_RENDER_PASS_AFTER_PRESENT_ENCODE, 0,
         trace_pass, &items[4]},
        {sizeof(FS_RenderPassDesc), FS_RENDER_PASS_BEFORE_SCENE, -10,
         trace_pass, &items[0]},
        {sizeof(FS_RenderPassDesc), FS_RENDER_PASS_AFTER_SCENE, 0,
         trace_pass, &items[2]}};
    for (uint32_t i = 0; i < 5; ++i)
        assert(fs_render_frame_add_pass(frame, &passes[i], &error) == FS_RESULT_OK);
    FS_RenderPassDesc reserved = {
        sizeof(FS_RenderPassDesc), FS_RENDER_PASS_PRESENT, 0, trace_pass, &items[0]};
    assert(fs_render_frame_add_pass(frame, &reserved, &error) ==
           FS_RESULT_INVALID_ARGUMENT);

    FS_CommandBatch batch = {0};
    assert(fs_render_frame_encode(frame, 0, 0, 0, 1, &batch, &error) ==
           FS_RESULT_OK);
    assert(trace.count == 5);
    for (uint32_t i = 0; i < 5; ++i) assert(trace.values[i] == i + 1);
    assert(fs_command_batch_submit(context, &batch, &error) == FS_RESULT_OK);
    fs_command_batch_release(&batch);
    fs_render_target_destroy(target);
    fs_render_context_destroy(context);
    fs_gpu_context_destroy(gpu);
    return 0;
}
