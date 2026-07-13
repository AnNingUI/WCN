#include "fullstack_app_private.h"

#include <string.h>

typedef struct FS_AppFrameRetirement {
    const FS_Allocator* allocator;
    WGPUTexture texture;
    WGPUTextureView view;
} FS_AppFrameRetirement;

static void fs_app_frame_release_handles(FS_AppFrame* frame) {
    if (frame->view) wgpuTextureViewRelease(frame->view);
    if (frame->texture) wgpuTextureRelease(frame->texture);
    frame->view = NULL;
    frame->texture = NULL;
}

static void FS_CALL fs_app_frame_retire_handles(void* data) {
    FS_AppFrameRetirement* retirement = (FS_AppFrameRetirement*)data;
    if (!retirement) return;
    if (retirement->view) wgpuTextureViewRelease(retirement->view);
    if (retirement->texture) wgpuTextureRelease(retirement->texture);
    fs_allocator_deallocate(retirement->allocator, retirement,
                            sizeof(*retirement), sizeof(void*));
}

static void fs_app_frame_release_after_submission(FS_AppWindow* window,
                                                   FS_AppFrame* frame) {
    FS_App* app = window->app;
    window->retiring_submission = frame->submission;
    if (!frame->texture && !frame->view) return;
    if (!app->gpu || !frame->submission.serial) {
        fs_app_frame_release_handles(frame);
        return;
    }
    FS_AppFrameRetirement* retirement =
        (FS_AppFrameRetirement*)fs_allocator_allocate(
            app->allocator, sizeof(*retirement), sizeof(void*));
    if (!retirement) {
        (void)wgpuDevicePoll(fs_gpu_device(app->gpu), true, NULL);
        fs_app_frame_release_handles(frame);
        return;
    }
    *retirement = (FS_AppFrameRetirement){
        app->allocator, frame->texture, frame->view};
    frame->texture = NULL;
    frame->view = NULL;
    if (fs_gpu_retire_after(app->gpu, frame->submission,
                            fs_app_frame_retire_handles, retirement, NULL) !=
        FS_RESULT_OK) {
        (void)wgpuDevicePoll(fs_gpu_device(app->gpu), true, NULL);
        fs_app_frame_retire_handles(retirement);
    }
}

static bool fs_app_frame_validate(FS_AppWindow* window, FS_AppFrame* frame,
                                  FS_AppFrameState required,
                                  bool allow_stale, FS_Error* error,
                                  const char* operation) {
    if (!window || !frame || frame->struct_size < sizeof(*frame) ||
        frame->state != required || frame->window_id != window->id ||
        window->active_frame != frame ||
        frame->internal_token[0] != window->active_frame_cookie) {
        FS_ERROR_SET(error, FS_RESULT_INVALID_STATE, FS_ERROR_DOMAIN_APP, 0,
                     operation, "frame token or state is invalid");
        return false;
    }
    if (!allow_stale && frame->surface_generation != window->surface_generation) {
        FS_ERROR_SET(error, FS_RESULT_INVALID_STATE, FS_ERROR_DOMAIN_APP, 0,
                     operation, "frame belongs to a stale Surface generation");
        return false;
    }
    return true;
}

static void fs_app_frame_complete(FS_AppWindow* window, FS_AppFrame* frame,
                                  FS_AppFrameState state) {
    frame->state = state;
    window->active_frame = NULL;
    window->active_frame_cookie = 0;
    window->surface_state = window->pending_surface_flags
        || window->retiring_submission.serial
        ? FS_APP_SURFACE_RECOVERY_PENDING
        : ((window->metrics.framebuffer_width && window->metrics.framebuffer_height)
            ? FS_APP_SURFACE_READY : FS_APP_SURFACE_UNCONFIGURED);
}

bool fs_app_frame_is_active(const FS_AppWindow* window) {
    return window && window->active_frame != NULL;
}

FS_Result FS_CALL fs_app_window_acquire_frame(
    FS_AppWindow* window, FS_AppFrame* frame, FS_Error* error) {
    if (!window || !frame || frame->struct_size < sizeof(*frame))
        return FS_RESULT_INVALID_ARGUMENT;
    FS_App* app = window->app;
    if (!app || app->state != FS_APP_STATE_FRAME_ACTIVE ||
        !app->pumped_this_frame) return FS_RESULT_INVALID_STATE;
    if (frame->state != FS_APP_FRAME_EMPTY || fs_app_frame_is_active(window))
        return FS_RESULT_INVALID_STATE;
    if (!window->metrics.framebuffer_width ||
        !window->metrics.framebuffer_height) {
        frame->acquire_status = FS_APP_FRAME_ACQUIRE_ZERO_SIZE;
        return FS_RESULT_SKIP;
    }
    if (window->pending_surface_flags &
        (FS_SURFACE_PENDING_UNAVAILABLE | FS_SURFACE_PENDING_DEVICE_LOST)) {
        frame->acquire_status = (window->pending_surface_flags &
            FS_SURFACE_PENDING_DEVICE_LOST)
            ? FS_APP_FRAME_ACQUIRE_DEVICE_LOST
            : FS_APP_FRAME_ACQUIRE_SUSPENDED;
        return FS_RESULT_SKIP;
    }
    if (window->surface_state != FS_APP_SURFACE_READY) {
        frame->acquire_status = window->surface_state == FS_APP_SURFACE_DEVICE_LOST
            ? FS_APP_FRAME_ACQUIRE_DEVICE_LOST
            : FS_APP_FRAME_ACQUIRE_SUSPENDED;
        return FS_RESULT_SKIP;
    }
    frame->acquire_status = FS_APP_FRAME_ACQUIRE_AVAILABLE;
    FS_Result result = app->factory->ops->acquire_frame(
        app->backend, window->backend_window, frame, error);
    if (result != FS_RESULT_OK) {
        if (frame->acquire_status == FS_APP_FRAME_ACQUIRE_OUTDATED)
            window->pending_surface_flags |= FS_SURFACE_PENDING_RECONFIGURE;
        else if (frame->acquire_status == FS_APP_FRAME_ACQUIRE_SURFACE_LOST)
            window->pending_surface_flags |= FS_SURFACE_PENDING_AVAILABLE;
        else if (frame->acquire_status == FS_APP_FRAME_ACQUIRE_DEVICE_LOST)
            window->pending_surface_flags |= FS_SURFACE_PENDING_DEVICE_LOST;
        if (window->pending_surface_flags)
            window->surface_state = FS_APP_SURFACE_RECOVERY_PENDING;
        return result;
    }
    if (frame->acquire_status == FS_APP_FRAME_ACQUIRE_SUBOPTIMAL)
        window->pending_surface_flags |= FS_SURFACE_PENDING_RECONFIGURE;
    frame->state = FS_APP_FRAME_ACQUIRED;
    frame->frame_id = app->next_frame_id++;
    frame->window_id = window->id;
    frame->surface_generation = window->surface_generation;
    if (!frame->width) frame->width = window->metrics.framebuffer_width;
    if (!frame->height) frame->height = window->metrics.framebuffer_height;
    uint64_t cookie = app->next_frame_cookie++;
    if (!cookie) cookie = app->next_frame_cookie++;
    frame->internal_token[0] = cookie;
    frame->internal_token[1] = window->id;
    frame->internal_token[2] = app->gpu ? fs_gpu_context_id(app->gpu) : 0;
    window->active_frame = frame;
    window->active_frame_cookie = cookie;
    window->surface_state = FS_APP_SURFACE_FRAME_ACQUIRED;
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_app_frame_mark_submitted(
    FS_AppWindow* window, FS_AppFrame* frame,
    FS_SubmissionToken submission, FS_Error* error) {
    if (!fs_app_frame_validate(window, frame, FS_APP_FRAME_ACQUIRED,
                               false, error, "fs_app_frame_mark_submitted"))
        return FS_RESULT_INVALID_STATE;
    FS_App* app = window->app;
    if (!app->gpu || submission.context_id != fs_gpu_context_id(app->gpu))
        return FS_RESULT_INVALID_ARGUMENT;
    FS_SubmissionStatus status = FS_SUBMISSION_UNKNOWN;
    if (fs_gpu_submission_status(app->gpu, submission, &status, error) !=
        FS_RESULT_OK || (status != FS_SUBMISSION_PENDING &&
                         status != FS_SUBMISSION_SUCCEEDED))
        return FS_RESULT_INVALID_ARGUMENT;
    frame->submission = submission;
    frame->state = FS_APP_FRAME_SUBMITTED;
    window->surface_state = FS_APP_SURFACE_FRAME_SUBMITTED;
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_app_frame_submit(
    FS_AppWindow* window, FS_AppFrame* frame, uint32_t command_count,
    const WGPUCommandBuffer* commands, FS_Error* error) {
    if (!window || !window->app || !window->app->gpu)
        return FS_RESULT_UNSUPPORTED;
    if (!fs_app_frame_validate(window, frame, FS_APP_FRAME_ACQUIRED,
                               false, error, "fs_app_frame_submit"))
        return FS_RESULT_INVALID_STATE;
    FS_SubmissionToken submission = {0};
    FS_Result result = fs_gpu_submit(window->app->gpu, command_count, commands,
                                     &submission, error);
    if (result != FS_RESULT_OK) return result;
    return fs_app_frame_mark_submitted(window, frame, submission, error);
}

FS_Result FS_CALL fs_app_frame_present(
    FS_AppWindow* window, FS_AppFrame* frame, FS_Error* error) {
    if (!fs_app_frame_validate(window, frame, FS_APP_FRAME_SUBMITTED,
                               false, error, "fs_app_frame_present"))
        return FS_RESULT_INVALID_STATE;
    FS_App* app = window->app;
    FS_Result result = app->factory->ops->present_frame(
        app->backend, window->backend_window, frame, error);
    if (result != FS_RESULT_OK) {
        app->factory->ops->cancel_frame(
            app->backend, window->backend_window, frame);
        window->pending_surface_flags |= FS_SURFACE_PENDING_RECONFIGURE;
        fs_app_frame_release_after_submission(window, frame);
        fs_app_frame_complete(window, frame, FS_APP_FRAME_CANCELLED);
        return result;
    }
    fs_app_frame_release_handles(frame);
    fs_app_frame_complete(window, frame, FS_APP_FRAME_PRESENTED);
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_app_frame_cancel(
    FS_AppWindow* window, FS_AppFrame* frame, FS_Error* error) {
    if (!window || !frame ||
        (frame->state != FS_APP_FRAME_ACQUIRED &&
         frame->state != FS_APP_FRAME_SUBMITTED))
        return FS_RESULT_INVALID_STATE;
    if (!fs_app_frame_validate(window, frame, frame->state, true, error,
                               "fs_app_frame_cancel"))
        return FS_RESULT_INVALID_STATE;
    window->app->factory->ops->cancel_frame(
        window->app->backend, window->backend_window, frame);
    if (frame->state == FS_APP_FRAME_SUBMITTED)
        fs_app_frame_release_after_submission(window, frame);
    else
        fs_app_frame_release_handles(frame);
    fs_app_frame_complete(window, frame, FS_APP_FRAME_CANCELLED);
    return FS_RESULT_OK;
}

void fs_app_frame_cancel_for_destroy(FS_AppWindow* window) {
    if (window && window->active_frame)
        (void)fs_app_frame_cancel(window, window->active_frame, NULL);
}

FS_Result FS_CALL fs_app_frame_reset(FS_AppFrame* frame) {
    if (!frame || frame->struct_size < sizeof(*frame))
        return FS_RESULT_INVALID_ARGUMENT;
    if (frame->state != FS_APP_FRAME_EMPTY &&
        frame->state != FS_APP_FRAME_PRESENTED &&
        frame->state != FS_APP_FRAME_CANCELLED)
        return FS_RESULT_INVALID_STATE;
    uint32_t size = frame->struct_size;
    memset(frame, 0, sizeof(*frame));
    frame->struct_size = size;
    frame->state = FS_APP_FRAME_EMPTY;
    frame->acquire_status = FS_APP_FRAME_ACQUIRE_AVAILABLE;
    return FS_RESULT_OK;
}
