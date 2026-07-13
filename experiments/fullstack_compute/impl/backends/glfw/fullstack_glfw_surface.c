#include "fullstack_glfw_app_backend_private.h"

static bool fs_glfw_surface_has_format(const WGPUSurfaceCapabilities* caps,
                                       WGPUTextureFormat format) {
    for (size_t i = 0; i < caps->formatCount; ++i)
        if (caps->formats[i] == format) return true;
    return false;
}

static WGPUTextureFormat fs_glfw_surface_select_format(
    const WGPUSurfaceCapabilities* caps) {
    const WGPUTextureFormat preferred[] = {
        WGPUTextureFormat_BGRA8UnormSrgb,
        WGPUTextureFormat_RGBA8UnormSrgb,
        WGPUTextureFormat_BGRA8Unorm,
        WGPUTextureFormat_RGBA8Unorm};
    for (uint32_t i = 0; i < 4; ++i)
        if (fs_glfw_surface_has_format(caps, preferred[i])) return preferred[i];
    return caps->formatCount ? caps->formats[0] : WGPUTextureFormat_Undefined;
}

static FS_Result fs_glfw_surface_select_contract(FS_GlfwAppWindow* window,
                                                  FS_Error* error) {
    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    wgpuSurfaceGetCapabilities(window->surface,
        fs_gpu_adapter(window->backend->host->gpu), &caps);
    window->format = fs_glfw_surface_select_format(&caps);
    window->present_mode = WGPUPresentMode_Fifo;
    window->alpha_mode = caps.alphaModeCount
        ? caps.alphaModes[0] : WGPUCompositeAlphaMode_Auto;
    window->usage = WGPUTextureUsage_RenderAttachment;
    bool valid = window->format != WGPUTextureFormat_Undefined;
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    if (!valid) {
        FS_ERROR_SET(error, FS_RESULT_UNSUPPORTED, FS_ERROR_DOMAIN_BACKEND, 0,
                     "fs_glfw_surface_select_contract",
                     "Surface exposes no supported output format");
        return FS_RESULT_UNSUPPORTED;
    }
    return FS_RESULT_OK;
}

static FS_Result fs_glfw_surface_configure(FS_GlfwAppWindow* window,
                                            FS_Error* error) {
    fs_glfw_refresh_metrics(window);
    if (!window->metrics.framebuffer_width ||
        !window->metrics.framebuffer_height) {
        window->configured = false;
        return FS_RESULT_SKIP;
    }
    if (!window->surface) return FS_RESULT_INVALID_STATE;
    WGPUSurfaceConfiguration configuration = {
        .device = fs_gpu_device(window->backend->host->gpu),
        .format = window->format,
        .usage = window->usage,
        .width = window->metrics.framebuffer_width,
        .height = window->metrics.framebuffer_height,
        .presentMode = window->present_mode,
        .alphaMode = window->alpha_mode};
    wgpuSurfaceConfigure(window->surface, &configuration);
    window->configured_width = configuration.width;
    window->configured_height = configuration.height;
    window->configured = true;
    (void)error;
    return FS_RESULT_OK;
}

static FS_Result fs_glfw_surface_recreate(FS_GlfwAppWindow* window,
                                           FS_Error* error) {
    if (window->surface) {
        if (window->configured) wgpuSurfaceUnconfigure(window->surface);
        wgpuSurfaceRelease(window->surface);
    }
    window->surface = fs_glfw_platform_create_surface(window, error);
    window->configured = false;
    if (!window->surface) return FS_RESULT_INTERNAL_ERROR;
    return fs_glfw_surface_select_contract(window, error);
}

FS_Result fs_glfw_surface_initialize(FS_GlfwAppWindow* window, FS_Error* error) {
    window->surface = fs_glfw_platform_create_surface(window, error);
    if (!window->surface) return FS_RESULT_INTERNAL_ERROR;
    FS_Result result = fs_glfw_surface_select_contract(window, error);
    if (result != FS_RESULT_OK) {
        fs_glfw_surface_shutdown(window);
        return result;
    }
    result = fs_glfw_surface_configure(window, error);
    return result == FS_RESULT_SKIP ? FS_RESULT_OK : result;
}

void fs_glfw_surface_shutdown(FS_GlfwAppWindow* window) {
    if (!window) return;
    if (window->surface) {
        if (window->configured) wgpuSurfaceUnconfigure(window->surface);
        wgpuSurfaceRelease(window->surface);
        window->surface = NULL;
    }
    window->configured = false;
    window->acquired = false;
    fs_glfw_platform_release_surface_source(window);
}

static FS_Result fs_glfw_surface_status_result(
    FS_GlfwAppWindow* window, WGPUSurfaceGetCurrentTextureStatus status,
    FS_AppFrame* frame, FS_Error* error) {
    switch (status) {
        case WGPUSurfaceGetCurrentTextureStatus_Timeout:
            frame->acquire_status = FS_APP_FRAME_ACQUIRE_TIMEOUT;
            return FS_RESULT_SKIP;
        case WGPUSurfaceGetCurrentTextureStatus_Occluded:
            frame->acquire_status = FS_APP_FRAME_ACQUIRE_OCCLUDED;
            return FS_RESULT_SKIP;
        case WGPUSurfaceGetCurrentTextureStatus_Outdated:
            frame->acquire_status = FS_APP_FRAME_ACQUIRE_OUTDATED;
            window->configured = false;
            return FS_RESULT_SKIP;
        case WGPUSurfaceGetCurrentTextureStatus_Lost:
            frame->acquire_status = FS_APP_FRAME_ACQUIRE_SURFACE_LOST;
            (void)fs_glfw_surface_recreate(window, error);
            return FS_RESULT_SKIP;
        default:
            frame->acquire_status = FS_APP_FRAME_ACQUIRE_FATAL;
            FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR,
                         FS_ERROR_DOMAIN_BACKEND, (int64_t)status,
                         "fs_glfw_surface_acquire",
                         "wgpuSurfaceGetCurrentTexture failed");
            return FS_RESULT_INTERNAL_ERROR;
    }
}

FS_Result FS_CALL fs_glfw_surface_acquire(
    FS_BackendInstance* instance, FS_BackendWindow* backend_window,
    FS_AppFrame* frame, FS_Error* error) {
    (void)instance;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)backend_window;
    if (!window || !frame || window->acquired) return FS_RESULT_INVALID_STATE;
    if (fs_gpu_context_state(window->backend->host->gpu) != FS_GPU_STATE_READY) {
        frame->acquire_status = FS_APP_FRAME_ACQUIRE_DEVICE_LOST;
        return FS_RESULT_SKIP;
    }
    fs_glfw_refresh_metrics(window);
    if (!window->metrics.framebuffer_width ||
        !window->metrics.framebuffer_height) {
        frame->acquire_status = FS_APP_FRAME_ACQUIRE_ZERO_SIZE;
        return FS_RESULT_SKIP;
    }
    if (!window->configured ||
        window->configured_width != window->metrics.framebuffer_width ||
        window->configured_height != window->metrics.framebuffer_height) {
        FS_Result configure = fs_glfw_surface_configure(window, error);
        if (configure != FS_RESULT_OK) return configure;
    }
    WGPUSurfaceTexture acquired = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(window->surface, &acquired);
    if (acquired.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        acquired.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        if (acquired.texture) wgpuTextureRelease(acquired.texture);
        return fs_glfw_surface_status_result(
            window, acquired.status, frame, error);
    }
    WGPUTextureView view = wgpuTextureCreateView(acquired.texture, NULL);
    if (!view) {
        wgpuTextureRelease(acquired.texture);
        frame->acquire_status = FS_APP_FRAME_ACQUIRE_FATAL;
        return FS_RESULT_INTERNAL_ERROR;
    }
    frame->texture = acquired.texture;
    frame->view = view;
    frame->format = window->format;
    frame->width = window->configured_width;
    frame->height = window->configured_height;
    frame->acquire_status =
        acquired.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal
        ? FS_APP_FRAME_ACQUIRE_SUBOPTIMAL
        : FS_APP_FRAME_ACQUIRE_AVAILABLE;
    window->reconfigure_after_present =
        frame->acquire_status == FS_APP_FRAME_ACQUIRE_SUBOPTIMAL;
    window->acquired = true;
    return FS_RESULT_OK;
}

FS_Result FS_CALL fs_glfw_surface_present(
    FS_BackendInstance* instance, FS_BackendWindow* backend_window,
    FS_AppFrame* frame, FS_Error* error) {
    (void)instance;
    (void)frame;
    (void)error;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)backend_window;
    if (!window || !window->acquired) return FS_RESULT_INVALID_STATE;
    wgpuSurfacePresent(window->surface);
    window->acquired = false;
    if (window->reconfigure_after_present) {
        window->configured = false;
        window->reconfigure_after_present = false;
    }
    return FS_RESULT_OK;
}

void FS_CALL fs_glfw_surface_cancel(
    FS_BackendInstance* instance, FS_BackendWindow* backend_window,
    FS_AppFrame* frame) {
    (void)instance;
    (void)frame;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)backend_window;
    if (window) window->acquired = false;
}
