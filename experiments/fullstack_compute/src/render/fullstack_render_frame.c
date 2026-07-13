#include "fullstack_render_private.h"

#include <string.h>

void fs_render_frame_finish(FS_RenderFrame* frame, FS_RenderFrameState state) {
    frame->state = state;
    frame->target = NULL;
    frame->target_generation = 0;
    frame->pass_count = 0;
}

FS_Result FS_CALL fs_render_frame_add_pass(
    FS_RenderFrame* frame, const FS_RenderPassDesc* desc, FS_Error* error) {
    if (!frame || !desc || desc->struct_size < sizeof(*desc) ||
        !desc->callback) return FS_RESULT_INVALID_ARGUMENT;
    if (frame->state != FS_RENDER_FRAME_RECORDING)
        return FS_RESULT_INVALID_STATE;
    if (!fs_render_pass_stage_is_custom(desc->stage)) {
        FS_ERROR_SET(error, FS_RESULT_INVALID_ARGUMENT, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_render_frame_add_pass",
                     "SCENE, EFFECTS, and PRESENT are runtime-reserved stages");
        return FS_RESULT_INVALID_ARGUMENT;
    }
    FS_RenderContext* context = frame->context;
    if (frame->pass_count == context->max_passes) {
        FS_ERROR_SET(error, FS_RESULT_QUEUE_PRESSURE, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_render_frame_add_pass", "custom pass arena is full");
        return FS_RESULT_QUEUE_PRESSURE;
    }
    FS_RenderPassSlot slot = {*desc, frame->next_insertion_index++};
    fs_render_pass_insert(context, slot);
    return FS_RESULT_OK;
}

static FS_Result fs_render_encode_stage(FS_RenderFrame* frame,
                                        FS_RenderPassStage stage,
                                        WGPUCommandEncoder encoder,
                                        const FS_CoreSceneOutput* scene,
                                        FS_Error* error) {
    return fs_render_run_stage(frame, stage, encoder, scene, error);
}

FS_Result FS_CALL fs_render_frame_encode(
    FS_RenderFrame* frame, float clear_r, float clear_g, float clear_b,
    float clear_a, FS_CommandBatch* out_batch, FS_Error* error) {
    if (!frame || !out_batch || out_batch->command_buffer)
        return FS_RESULT_INVALID_ARGUMENT;
    if (frame->state != FS_RENDER_FRAME_RECORDING)
        return FS_RESULT_INVALID_STATE;
    FS_RenderContext* context = frame->context;
    if (!frame->target || frame->target_generation != frame->target->generation)
        return FS_RESULT_INVALID_STATE;

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        fs_gpu_device(context->gpu), &(WGPUCommandEncoderDescriptor){0});
    if (!encoder) return FS_RESULT_INTERNAL_ERROR;
    FS_Result result = fs_render_encode_stage(
        frame, FS_RENDER_PASS_BEFORE_SCENE, encoder, NULL, error);
    FS_CoreSceneOutput scene = {sizeof(FS_CoreSceneOutput)};
    if (result == FS_RESULT_OK &&
        !fs_core_encode_scene_base(context->core, encoder, clear_r, clear_g,
                                   clear_b, clear_a, &scene)) {
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_render_frame_encode", "Core scene encoding failed");
        result = FS_RESULT_INTERNAL_ERROR;
    }
    if (result == FS_RESULT_OK)
        result = fs_render_encode_stage(
            frame, FS_RENDER_PASS_AFTER_SCENE, encoder, &scene, error);
    if (result == FS_RESULT_OK)
        result = fs_render_encode_stage(
            frame, FS_RENDER_PASS_BEFORE_EFFECTS, encoder, &scene, error);
    if (result == FS_RESULT_OK &&
        !fs_core_encode_scene_effects(context->core, encoder, &scene)) {
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_render_frame_encode", "Core effects encoding failed");
        result = FS_RESULT_INTERNAL_ERROR;
    }
    if (result == FS_RESULT_OK)
        result = fs_render_encode_stage(
            frame, FS_RENDER_PASS_AFTER_EFFECTS, encoder, &scene, error);
    if (result == FS_RESULT_OK)
        result = fs_render_encode_stage(
            frame, FS_RENDER_PASS_BEFORE_PRESENT, encoder, &scene, error);
    if (result == FS_RESULT_OK)
        result = fs_presenter_set_source(context->presenter, scene.view, error);
    if (result == FS_RESULT_OK)
        result = fs_presenter_encode(context->presenter, encoder,
            frame->target->view, context->width, context->height, error);
    if (result == FS_RESULT_OK)
        result = fs_render_encode_stage(frame,
            FS_RENDER_PASS_AFTER_PRESENT_ENCODE, encoder, &scene, error);
    if (result != FS_RESULT_OK) {
        wgpuCommandEncoderRelease(encoder);
        fs_render_frame_finish(frame, FS_RENDER_FRAME_CANCELLED);
        return result;
    }

    WGPUCommandBuffer command = wgpuCommandEncoderFinish(
        encoder, &(WGPUCommandBufferDescriptor){0});
    wgpuCommandEncoderRelease(encoder);
    if (!command) {
        fs_render_frame_finish(frame, FS_RENDER_FRAME_CANCELLED);
        return FS_RESULT_INTERNAL_ERROR;
    }
    memset(out_batch, 0, sizeof(*out_batch));
    out_batch->struct_size = sizeof(*out_batch);
    out_batch->command_buffer = command;
    fs_render_frame_finish(frame, FS_RENDER_FRAME_ENCODED);
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_render_frame_cancel(FS_RenderFrame* frame,
                                         FS_Error* error) {
    (void)error;
    if (!frame) return FS_RESULT_INVALID_ARGUMENT;
    if (frame->state != FS_RENDER_FRAME_RECORDING)
        return FS_RESULT_INVALID_STATE;
    fs_render_frame_finish(frame, FS_RENDER_FRAME_CANCELLED);
    return FS_RESULT_OK;
}

FS_RenderFrameState FS_CALL fs_render_frame_state(const FS_RenderFrame* frame) {
    return frame ? frame->state : FS_RENDER_FRAME_IDLE;
}

FS_Result FS_CALL fs_command_batch_submit(
    FS_RenderContext* context, FS_CommandBatch* batch, FS_Error* error) {
    if (!context || !batch || !batch->command_buffer || batch->submitted)
        return FS_RESULT_INVALID_ARGUMENT;
    FS_Result result = fs_gpu_submit(context->gpu, 1, &batch->command_buffer,
                                     &batch->submission, error);
    if (result == FS_RESULT_OK) {
        batch->submitted = true;
        context->last_submission = batch->submission;
    }
    return result;
}

void FS_CALL fs_command_batch_release(FS_CommandBatch* batch) {
    if (!batch) return;
    if (batch->command_buffer) wgpuCommandBufferRelease(batch->command_buffer);
    memset(batch, 0, sizeof(*batch));
}
