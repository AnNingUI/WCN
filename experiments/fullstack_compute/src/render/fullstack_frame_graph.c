#include "fullstack_render_private.h"

bool fs_render_pass_stage_is_custom(FS_RenderPassStage stage) {
    if (stage > FS_RENDER_PASS_AFTER_PRESENT_ENCODE) return false;
    return stage != FS_RENDER_PASS_SCENE &&
           stage != FS_RENDER_PASS_EFFECTS &&
           stage != FS_RENDER_PASS_PRESENT;
}

static bool fs_render_pass_before(const FS_RenderPassSlot* a,
                                  const FS_RenderPassSlot* b) {
    if (a->desc.stage != b->desc.stage)
        return a->desc.stage < b->desc.stage;
    if (a->desc.order != b->desc.order)
        return a->desc.order < b->desc.order;
    return a->insertion_index < b->insertion_index;
}

void fs_render_pass_insert(FS_RenderContext* context, FS_RenderPassSlot slot) {
    uint32_t index = context->frame.pass_count;
    while (index > 0 && fs_render_pass_before(&slot, &context->passes[index - 1])) {
        context->passes[index] = context->passes[index - 1];
        --index;
    }
    context->passes[index] = slot;
    context->frame.pass_count++;
}

FS_Result fs_render_run_stage(FS_RenderFrame* frame, FS_RenderPassStage stage,
                              WGPUCommandEncoder encoder,
                              const FS_CoreSceneOutput* scene,
                              FS_Error* error) {
    FS_RenderContext* context = frame->context;
    FS_RenderPassContext pass_context = {
        sizeof(FS_RenderPassContext), stage, encoder, context->core, scene,
        frame->target, context->width, context->height};
    for (uint32_t i = 0; i < frame->pass_count; ++i) {
        FS_RenderPassSlot* slot = &context->passes[i];
        if (slot->desc.stage != stage) continue;
        FS_Result result = slot->desc.callback(
            &pass_context, slot->desc.user_data, error);
        if (result != FS_RESULT_OK) return result;
    }
    return FS_RESULT_OK;
}
