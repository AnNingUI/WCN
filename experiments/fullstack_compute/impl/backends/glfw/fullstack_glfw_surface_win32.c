#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include "fullstack_glfw_app_backend_private.h"
#include <GLFW/glfw3native.h>
#include <windows.h>

WGPUSurface fs_glfw_platform_create_surface(FS_GlfwAppWindow* window,
                                             FS_Error* error) {
    WGPUSurfaceSourceWindowsHWND source = {
        .chain = {.sType = WGPUSType_SurfaceSourceWindowsHWND},
        .hinstance = GetModuleHandleW(NULL),
        .hwnd = glfwGetWin32Window(window->window)};
    WGPUSurface surface = wgpuInstanceCreateSurface(
        fs_gpu_instance(window->backend->host->gpu),
        &(WGPUSurfaceDescriptor){.nextInChain = &source.chain});
    if (!surface)
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_BACKEND, 0,
                     "fs_glfw_platform_create_surface",
                     "Win32 WebGPU Surface creation failed");
    return surface;
}

void fs_glfw_platform_release_surface_source(FS_GlfwAppWindow* window) {
    (void)window;
}

FS_Result fs_glfw_platform_native_handle(FS_GlfwAppWindow* window,
                                         FS_NativeWindowHandle* out_handle,
                                         FS_Error* error) {
    (void)error;
    if (!window || !out_handle) return FS_RESULT_INVALID_ARGUMENT;
    *out_handle = (FS_NativeWindowHandle){
        sizeof(*out_handle), FS_NATIVE_HANDLE_WIN32_HWND,
        (uintptr_t)glfwGetWin32Window(window->window), 0};
    return FS_RESULT_OK;
}
#endif
