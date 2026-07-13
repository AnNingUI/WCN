#include "fullstack_render_private.h"

#include <string.h>

static void FS_CALL fs_render_retire_core_resize(void* data) {
    fs_core_resize_retirement_destroy((FS_CoreResizeRetirement*)data);
}

typedef struct FS_RenderOwnedRetirement {
    const FS_Allocator* allocator;
    FS_Core* core;
    FS_Presenter* presenter;
} FS_RenderOwnedRetirement;

static void FS_CALL fs_render_retire_owned(void* data) {
    FS_RenderOwnedRetirement* retirement = (FS_RenderOwnedRetirement*)data;
    if (!retirement) return;
    fs_presenter_destroy(retirement->presenter);
    fs_core_destroy(retirement->core);
    fs_allocator_deallocate(retirement->allocator, retirement,
                            sizeof(*retirement), sizeof(void*));
}

static void fs_render_schedule_owned_retirement(
    FS_RenderContext* context, FS_Core* core, FS_Presenter* presenter) {
    if (!core && !presenter) return;
    if (!context->last_submission.serial) {
        fs_presenter_destroy(presenter);
        fs_core_destroy(core);
        return;
    }
    FS_RenderOwnedRetirement* retirement =
        (FS_RenderOwnedRetirement*)fs_allocator_allocate(
            context->allocator, sizeof(*retirement), sizeof(void*));
    if (!retirement) {
        (void)wgpuDevicePoll(fs_gpu_device(context->gpu), true, NULL);
        fs_presenter_destroy(presenter);
        fs_core_destroy(core);
        return;
    }
    *retirement = (FS_RenderOwnedRetirement){
        context->allocator, core, presenter};
    if (fs_gpu_retire_after(context->gpu, context->last_submission,
                            fs_render_retire_owned, retirement, NULL) !=
        FS_RESULT_OK) {
        (void)wgpuDevicePoll(fs_gpu_device(context->gpu), true, NULL);
        fs_render_retire_owned(retirement);
    }
}

static FS_Result fs_render_create_owned_resources(
    FS_RenderContext* context, FS_Core** out_core,
    FS_Presenter** out_presenter, FS_Error* error) {
    *out_core = fs_core_create(fs_gpu_device(context->gpu),
                               fs_gpu_queue(context->gpu),
                               WGPUTextureFormat_RGBA8Unorm,
                               context->width, context->height);
    if (!*out_core) {
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_render_context_create", "Core creation failed");
        return FS_RESULT_INTERNAL_ERROR;
    }
    FS_PresenterDesc presenter_desc = FS_PRESENTER_DESC_INIT;
    presenter_desc.allocator = context->allocator;
    presenter_desc.device = fs_gpu_device(context->gpu);
    presenter_desc.target_format = context->output_format;
    presenter_desc.target_color_space = context->output_color_space;
    FS_Result result = fs_presenter_create(
        &presenter_desc, out_presenter, error);
    if (result != FS_RESULT_OK) {
        fs_core_destroy(*out_core);
        *out_core = NULL;
    }
    return result;
}

FS_Result FS_CALL fs_render_context_create(
    const FS_RenderContextDesc* desc, FS_RenderContext** out_context,
    FS_Error* error) {
    if (out_context) *out_context = NULL;
    if (!desc || !out_context || desc->struct_size < sizeof(*desc) ||
        desc->abi_version != FS_RENDER_ABI_VERSION || !desc->gpu ||
        !desc->width || !desc->height || !desc->max_custom_passes) {
        return FS_RESULT_INVALID_ARGUMENT;
    }
    FS_Result color_result = fs_presenter_validate_color_contract(
        desc->output_format, FS_COLOR_SPACE_LINEAR_SRGB,
        desc->output_color_space, error);
    if (color_result != FS_RESULT_OK) return color_result;

    const FS_Allocator* allocator = desc->allocator ? desc->allocator
                                                    : fs_default_allocator();
    FS_RenderContext* context = (FS_RenderContext*)fs_allocator_allocate(
        allocator, sizeof(*context), sizeof(void*));
    if (!context) return FS_RESULT_OUT_OF_MEMORY;
    memset(context, 0, sizeof(*context));
    context->allocator = allocator;
    context->diagnostics = desc->diagnostics;
    context->gpu = desc->gpu;
    fs_gpu_context_retain(context->gpu);
    context->width = desc->width;
    context->height = desc->height;
    context->output_format = desc->output_format;
    context->output_color_space = desc->output_color_space;
    context->max_passes = desc->max_custom_passes;
    context->passes = (FS_RenderPassSlot*)fs_allocator_allocate(
        allocator, sizeof(*context->passes) * context->max_passes, sizeof(void*));
    if (!context->passes) {
        fs_allocator_deallocate(allocator, context, sizeof(*context), sizeof(void*));
        fs_gpu_context_release(desc->gpu);
        return FS_RESULT_OUT_OF_MEMORY;
    }
    memset(context->passes, 0, sizeof(*context->passes) * context->max_passes);
    context->frame.context = context;
    context->frame.state = FS_RENDER_FRAME_IDLE;
    FS_Result result = fs_render_create_owned_resources(
        context, &context->core, &context->presenter, error);
    if (result != FS_RESULT_OK) {
        fs_render_context_destroy(context);
        return result;
    }
    *out_context = context;
    return FS_RESULT_OK;
}

void FS_CALL fs_render_context_destroy(FS_RenderContext* context) {
    if (!context) return;
    fs_render_schedule_owned_retirement(
        context, context->core, context->presenter);
    if (context->passes)
        fs_allocator_deallocate(context->allocator, context->passes,
            sizeof(*context->passes) * context->max_passes, sizeof(void*));
    const FS_Allocator* allocator = context->allocator;
    FS_GpuContext* gpu = context->gpu;
    fs_allocator_deallocate(allocator, context, sizeof(*context), sizeof(void*));
    fs_gpu_context_release(gpu);
}

FS_Core* FS_CALL fs_render_context_core(FS_RenderContext* context) {
    return context ? context->core : NULL;
}

FS_Result FS_CALL fs_render_context_try_resize(
    FS_RenderContext* context, uint32_t width, uint32_t height,
    FS_Error* error) {
    if (!context || !width || !height) return FS_RESULT_INVALID_ARGUMENT;
    if (context->frame.state == FS_RENDER_FRAME_RECORDING)
        return FS_RESULT_INVALID_STATE;
    if (context->width == width && context->height == height)
        return FS_RESULT_OK;
    FS_CoreResizeRetirement* retirement = NULL;
    if (!fs_core_try_resize_deferred(context->core, width, height, &retirement)) {
        FS_ERROR_SET(error, FS_RESULT_OUT_OF_MEMORY, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_render_context_try_resize",
                     "transactional resize failed; previous resources preserved");
        return FS_RESULT_OUT_OF_MEMORY;
    }
    context->width = width;
    context->height = height;
    if (retirement) {
        if (context->last_submission.serial) {
            FS_Result retire_result = fs_gpu_retire_after(
                context->gpu, context->last_submission,
                fs_render_retire_core_resize, retirement, error);
            if (retire_result != FS_RESULT_OK) {
                (void)wgpuDevicePoll(fs_gpu_device(context->gpu), true, NULL);
                fs_core_resize_retirement_destroy(retirement);
            }
        } else {
            fs_core_resize_retirement_destroy(retirement);
        }
    }
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_render_context_recreate_device_resources(
    FS_RenderContext* context, FS_Error* error) {
    if (!context) return FS_RESULT_INVALID_ARGUMENT;
    if (context->frame.state == FS_RENDER_FRAME_RECORDING)
        return FS_RESULT_INVALID_STATE;
    FS_Core* new_core = NULL;
    FS_Presenter* new_presenter = NULL;
    FS_Result result = fs_render_create_owned_resources(
        context, &new_core, &new_presenter, error);
    if (result != FS_RESULT_OK) return result;
    FS_Core* old_core = context->core;
    FS_Presenter* old_presenter = context->presenter;
    context->core = new_core;
    context->presenter = new_presenter;
    fs_render_schedule_owned_retirement(context, old_core, old_presenter);
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_render_context_begin_frame(
    FS_RenderContext* context, FS_RenderTarget* target,
    FS_RenderFrame** out_frame, FS_Error* error) {
    if (out_frame) *out_frame = NULL;
    if (!context || !target || !out_frame) return FS_RESULT_INVALID_ARGUMENT;
    if (context->frame.state == FS_RENDER_FRAME_RECORDING)
        return FS_RESULT_INVALID_STATE;
    if (target->format != context->output_format ||
        target->color_space != context->output_color_space) {
        FS_ERROR_SET(error, FS_RESULT_INVALID_ARGUMENT, FS_ERROR_DOMAIN_RENDER, 0,
                     "fs_render_context_begin_frame",
                     "render target output contract does not match context");
        return FS_RESULT_INVALID_ARGUMENT;
    }
    FS_Result result = fs_render_context_try_resize(
        context, target->width, target->height, error);
    if (result != FS_RESULT_OK) return result;
    context->frame.target = target;
    context->frame.target_generation = target->generation;
    context->frame.pass_count = 0;
    context->frame.next_insertion_index = 0;
    context->frame.state = FS_RENDER_FRAME_RECORDING;
    *out_frame = &context->frame;
    return FS_RESULT_OK;
}
