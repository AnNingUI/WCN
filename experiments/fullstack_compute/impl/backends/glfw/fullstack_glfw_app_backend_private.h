#ifndef FULLSTACK_GLFW_APP_BACKEND_PRIVATE_H
#define FULLSTACK_GLFW_APP_BACKEND_PRIVATE_H

#include "fullstack_glfw_app_backend.h"
#include "fullstack_app_services.h"

#include <GLFW/glfw3.h>

typedef struct FS_GlfwAppBackend FS_GlfwAppBackend;
typedef struct FS_GlfwAppWindow FS_GlfwAppWindow;

struct FS_GlfwAppBackend {
    FS_AppBackendHost* host;
    uint32_t window_count;
    bool glfw_acquired;
    bool closing;
    uint64_t next_service_request;
    volatile long service_lock;
    void* service_requests;
    GLFWcursor* cursors[6];
};

struct FS_GlfwAppWindow {
    FS_GlfwAppBackend* backend;
    GLFWwindow* window;
    FS_AppWindowId stable_id;
    bool owns_window;
    void* previous_user_pointer;

    GLFWwindowsizefun previous_window_size;
    GLFWframebuffersizefun previous_framebuffer_size;
    GLFWwindowcontentscalefun previous_content_scale;
    GLFWcursorposfun previous_cursor_position;
    GLFWmousebuttonfun previous_mouse_button;
    GLFWscrollfun previous_scroll;
    GLFWkeyfun previous_key;
    GLFWcharfun previous_character;
    GLFWwindowclosefun previous_close;

    FS_AppWindowMetrics metrics;
    WGPUSurface surface;
    WGPUTextureFormat format;
    WGPUPresentMode present_mode;
    WGPUCompositeAlphaMode alpha_mode;
    WGPUTextureUsage usage;
    uint32_t configured_width;
    uint32_t configured_height;
    bool configured;
    bool acquired;
    bool reconfigure_after_present;
    void* platform_layer;
};

void fs_glfw_emit(FS_GlfwAppWindow* window, FS_AppEvent* event);
void fs_glfw_refresh_metrics(FS_GlfwAppWindow* window);
void fs_glfw_events_attach(FS_GlfwAppWindow* window);
void fs_glfw_events_detach(FS_GlfwAppWindow* window);
const FS_CapabilityHeader* fs_glfw_services_query(
    FS_GlfwAppBackend* backend, uint32_t capability, uint32_t version);
void fs_glfw_services_shutdown(FS_GlfwAppBackend* backend);
void fs_glfw_services_cancel_window(FS_GlfwAppBackend* backend,
                                    FS_AppWindowId window_id);

WGPUSurface fs_glfw_platform_create_surface(FS_GlfwAppWindow* window,
                                             FS_Error* error);
void fs_glfw_platform_release_surface_source(FS_GlfwAppWindow* window);
FS_Result fs_glfw_platform_native_handle(FS_GlfwAppWindow* window,
                                         FS_NativeWindowHandle* out_handle,
                                         FS_Error* error);
FS_Result fs_glfw_surface_initialize(FS_GlfwAppWindow* window, FS_Error* error);
void fs_glfw_surface_shutdown(FS_GlfwAppWindow* window);
FS_Result FS_CALL fs_glfw_surface_acquire(FS_BackendInstance* instance,
                                          FS_BackendWindow* window,
                                          FS_AppFrame* frame,
                                          FS_Error* error);
FS_Result FS_CALL fs_glfw_surface_present(FS_BackendInstance* instance,
                                          FS_BackendWindow* window,
                                          FS_AppFrame* frame,
                                          FS_Error* error);
void FS_CALL fs_glfw_surface_cancel(FS_BackendInstance* instance,
                                    FS_BackendWindow* window,
                                    FS_AppFrame* frame);

#endif
