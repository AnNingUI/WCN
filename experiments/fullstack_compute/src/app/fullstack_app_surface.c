#include "fullstack_app_private.h"

void fs_app_surface_note_event(FS_AppWindow* window, FS_AppEventType type) {
    if (!window) return;
    switch (type) {
        case FS_APP_EVENT_WINDOW_RESIZED:
        case FS_APP_EVENT_FRAMEBUFFER_RESIZED:
        case FS_APP_EVENT_SCALE_CHANGED:
            window->pending_surface_flags |= FS_SURFACE_PENDING_RECONFIGURE;
            break;
        case FS_APP_EVENT_SUSPEND:
        case FS_APP_EVENT_SURFACE_UNAVAILABLE:
            window->pending_surface_flags |= FS_SURFACE_PENDING_UNAVAILABLE;
            window->pending_surface_flags &= ~FS_SURFACE_PENDING_AVAILABLE;
            break;
        case FS_APP_EVENT_RESUME:
        case FS_APP_EVENT_SURFACE_AVAILABLE:
            if (window->surface_state == FS_APP_SURFACE_DEVICE_LOST) return;
            window->pending_surface_flags |= FS_SURFACE_PENDING_AVAILABLE;
            window->pending_surface_flags &= ~FS_SURFACE_PENDING_UNAVAILABLE;
            break;
        case FS_APP_EVENT_DEVICE_LOST:
            window->pending_surface_flags |= FS_SURFACE_PENDING_DEVICE_LOST;
            break;
        default:
            return;
    }
    if (!fs_app_frame_is_active(window))
        window->surface_state = FS_APP_SURFACE_RECOVERY_PENDING;
}

void fs_app_surface_commit_pending(FS_AppWindow* window) {
    if (!window || fs_app_frame_is_active(window)) return;
    bool retirement_completed = false;
    if (window->retiring_submission.serial) {
        FS_SubmissionStatus status = FS_SUBMISSION_UNKNOWN;
        FS_Result result = window->app->gpu
            ? fs_gpu_submission_status(window->app->gpu,
                window->retiring_submission, &status, NULL)
            : FS_RESULT_SKIP;
        if (result == FS_RESULT_OK && status == FS_SUBMISSION_PENDING) {
            window->surface_state = FS_APP_SURFACE_RECOVERY_PENDING;
            return;
        }
        window->retiring_submission = (FS_SubmissionToken){0};
        retirement_completed = true;
    }
    if (!window->pending_surface_flags) {
        if (retirement_completed) {
            window->surface_state =
                (window->metrics.framebuffer_width &&
                 window->metrics.framebuffer_height)
                ? FS_APP_SURFACE_READY : FS_APP_SURFACE_UNCONFIGURED;
        }
        return;
    }
    uint32_t flags = window->pending_surface_flags;
    window->pending_surface_flags = 0;
    window->surface_generation++;
    if (flags & FS_SURFACE_PENDING_DEVICE_LOST) {
        window->surface_state = FS_APP_SURFACE_DEVICE_LOST;
    } else if (flags & FS_SURFACE_PENDING_UNAVAILABLE) {
        window->surface_state = FS_APP_SURFACE_UNAVAILABLE;
    } else if (!window->metrics.framebuffer_width ||
               !window->metrics.framebuffer_height) {
        window->surface_state = FS_APP_SURFACE_UNCONFIGURED;
    } else {
        window->surface_state = FS_APP_SURFACE_READY;
    }
}

void fs_app_surface_wait_retirement(FS_AppWindow* window) {
    if (!window || !window->retiring_submission.serial || !window->app->gpu)
        return;
    (void)wgpuDevicePoll(fs_gpu_device(window->app->gpu), true, NULL);
    (void)fs_gpu_context_poll(window->app->gpu, NULL);
    window->retiring_submission = (FS_SubmissionToken){0};
}

FS_AppSurfaceState FS_CALL fs_app_window_surface_state(
    const FS_AppWindow* window) {
    return window ? window->surface_state : FS_APP_SURFACE_UNAVAILABLE;
}

uint64_t FS_CALL fs_app_window_surface_generation(
    const FS_AppWindow* window) {
    return window ? window->surface_generation : 0;
}

FS_Result FS_CALL fs_app_window_recover_surface(
    FS_AppWindow* window, FS_Error* error) {
    if (!window || !window->app) return FS_RESULT_INVALID_ARGUMENT;
    if (!fs_app_is_app_thread(window->app)) {
        FS_ERROR_SET(error, FS_RESULT_WRONG_THREAD, FS_ERROR_DOMAIN_APP, 0,
                     "fs_app_window_recover_surface",
                     "surface recovery must run on the App thread");
        return FS_RESULT_WRONG_THREAD;
    }
    if (fs_app_frame_is_active(window)) return FS_RESULT_INVALID_STATE;
    if (window->surface_state != FS_APP_SURFACE_DEVICE_LOST &&
        window->surface_state != FS_APP_SURFACE_UNAVAILABLE &&
        window->surface_state != FS_APP_SURFACE_RECOVERY_PENDING)
        return FS_RESULT_SKIP;
    if (window->app->gpu &&
        fs_gpu_context_state(window->app->gpu) != FS_GPU_STATE_READY)
        return FS_RESULT_INVALID_STATE;
    window->pending_surface_flags &= ~FS_SURFACE_PENDING_DEVICE_LOST;
    window->pending_surface_flags |= FS_SURFACE_PENDING_AVAILABLE;
    window->surface_state = FS_APP_SURFACE_RECOVERY_PENDING;
    return FS_RESULT_OK;
}
