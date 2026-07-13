#include "fullstack_glfw_app_backend_private.h"

#include <string.h>

void fs_glfw_refresh_metrics(FS_GlfwAppWindow* window) {
    if (!window || !window->window) return;
    int logical_width = 0, logical_height = 0;
    int framebuffer_width = 0, framebuffer_height = 0;
    float scale_x = 1.0f, scale_y = 1.0f;
    glfwGetWindowSize(window->window, &logical_width, &logical_height);
    glfwGetFramebufferSize(window->window, &framebuffer_width, &framebuffer_height);
    glfwGetWindowContentScale(window->window, &scale_x, &scale_y);
    window->metrics.struct_size = sizeof(window->metrics);
    window->metrics.logical_width = logical_width > 0 ? (uint32_t)logical_width : 0;
    window->metrics.logical_height = logical_height > 0 ? (uint32_t)logical_height : 0;
    window->metrics.framebuffer_width = framebuffer_width > 0
        ? (uint32_t)framebuffer_width : 0;
    window->metrics.framebuffer_height = framebuffer_height > 0
        ? (uint32_t)framebuffer_height : 0;
    window->metrics.scale_x = scale_x;
    window->metrics.scale_y = scale_y;
}

FS_Result fs_glfw_window_create(
    FS_BackendInstance* instance, const FS_AppWindowDesc* desc,
    FS_BackendWindow** out_window, FS_Error* error) {
    if (out_window) *out_window = NULL;
    FS_GlfwAppBackend* backend = (FS_GlfwAppBackend*)instance;
    if (!backend || !desc || !out_window || !desc->stable_id)
        return FS_RESULT_INVALID_ARGUMENT;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)fs_allocator_allocate(
        backend->host->allocator, sizeof(*window), sizeof(void*));
    if (!window) return FS_RESULT_OUT_OF_MEMORY;
    memset(window, 0, sizeof(*window));
    window->backend = backend;
    window->stable_id = desc->stable_id;

    if (desc->native_window) {
        if (desc->native_window_ownership != FS_RESOURCE_BORROWED &&
            desc->native_window_ownership != FS_RESOURCE_TRANSFERRED) {
            fs_allocator_deallocate(backend->host->allocator, window,
                                    sizeof(*window), sizeof(void*));
            return FS_RESULT_INVALID_ARGUMENT;
        }
        window->window = (GLFWwindow*)desc->native_window;
        window->owns_window =
            desc->native_window_ownership == FS_RESOURCE_TRANSFERRED;
    } else {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, desc->resizable ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_VISIBLE, desc->visible ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_DECORATED, desc->decorated ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER,
                       desc->transparent ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_SCALE_TO_MONITOR,
                       desc->high_dpi ? GLFW_TRUE : GLFW_FALSE);
        window->window = glfwCreateWindow((int)desc->width, (int)desc->height,
            desc->title ? desc->title : "WCN", NULL, NULL);
        window->owns_window = true;
    }
    if (!window->window) {
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_BACKEND, 0,
                     "fs_glfw_window_create", "glfwCreateWindow failed");
        fs_allocator_deallocate(backend->host->allocator, window,
                                sizeof(*window), sizeof(void*));
        return FS_RESULT_INTERNAL_ERROR;
    }
    fs_glfw_refresh_metrics(window);
    fs_glfw_events_attach(window);
    FS_Result result = fs_glfw_surface_initialize(window, error);
    if (result != FS_RESULT_OK) {
        fs_glfw_events_detach(window);
        if (window->owns_window) glfwDestroyWindow(window->window);
        fs_allocator_deallocate(backend->host->allocator, window,
                                sizeof(*window), sizeof(void*));
        return result;
    }
    backend->window_count++;
    *out_window = (FS_BackendWindow*)window;
    return FS_RESULT_OK;
}

void fs_glfw_window_destroy(FS_BackendInstance* instance,
                            FS_BackendWindow* backend_window) {
    FS_GlfwAppBackend* backend = (FS_GlfwAppBackend*)instance;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)backend_window;
    if (!backend || !window) return;
    fs_glfw_services_cancel_window(backend, window->stable_id);
    fs_glfw_surface_shutdown(window);
    fs_glfw_events_detach(window);
    if (window->owns_window && window->window) glfwDestroyWindow(window->window);
    if (backend->window_count) backend->window_count--;
    fs_allocator_deallocate(backend->host->allocator, window,
                            sizeof(*window), sizeof(void*));
}

FS_Result fs_glfw_window_metrics(FS_BackendInstance* instance,
                                 FS_BackendWindow* backend_window,
                                 FS_AppWindowMetrics* out_metrics,
                                 FS_Error* error) {
    (void)instance;
    (void)error;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)backend_window;
    if (!window || !out_metrics) return FS_RESULT_INVALID_ARGUMENT;
    fs_glfw_refresh_metrics(window);
    *out_metrics = window->metrics;
    return FS_RESULT_OK;
}

bool fs_glfw_window_should_close(FS_BackendInstance* instance,
                                  FS_BackendWindow* backend_window) {
    (void)instance;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)backend_window;
    return !window || !window->window || glfwWindowShouldClose(window->window);
}

GLFWwindow* FS_CALL fs_glfw_backend_native_window(FS_BackendWindow* window) {
    FS_GlfwAppWindow* glfw_window = (FS_GlfwAppWindow*)window;
    return glfw_window ? glfw_window->window : NULL;
}
