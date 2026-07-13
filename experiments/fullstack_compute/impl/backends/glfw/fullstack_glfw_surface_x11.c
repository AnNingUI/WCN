#if defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#include "fullstack_glfw_app_backend_private.h"
#include <GLFW/glfw3native.h>

WGPUSurface fs_glfw_platform_create_surface(FS_GlfwAppWindow* window,
                                             FS_Error* error) {
    WGPUSurfaceSourceXlibWindow source = {
        .chain = {.sType = WGPUSType_SurfaceSourceXlibWindow},
        .display = glfwGetX11Display(),
        .window = glfwGetX11Window(window->window)};
    WGPUSurface surface = wgpuInstanceCreateSurface(
        fs_gpu_instance(window->backend->host->gpu),
        &(WGPUSurfaceDescriptor){.nextInChain = &source.chain});
    if (!surface)
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_BACKEND, 0,
                     "fs_glfw_platform_create_surface",
                     "X11 WebGPU Surface creation failed");
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
        sizeof(*out_handle), FS_NATIVE_HANDLE_X11_WINDOW,
        (uintptr_t)glfwGetX11Window(window->window),
        (uintptr_t)glfwGetX11Display()};
    return FS_RESULT_OK;
}
#endif
