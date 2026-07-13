#if defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#include "fullstack_glfw_app_backend_private.h"
#include <GLFW/glfw3native.h>
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

WGPUSurface fs_glfw_platform_create_surface(FS_GlfwAppWindow* window,
                                             FS_Error* error) {
    NSWindow* native_window = glfwGetCocoaWindow(window->window);
    NSView* content = [native_window contentView];
    [content setWantsLayer:YES];
    CAMetalLayer* layer = [CAMetalLayer layer];
    [layer retain];
    [content setLayer:layer];
    window->platform_layer = layer;
    WGPUSurfaceSourceMetalLayer source = {
        .chain = {.sType = WGPUSType_SurfaceSourceMetalLayer},
        .layer = layer};
    WGPUSurface surface = wgpuInstanceCreateSurface(
        fs_gpu_instance(window->backend->host->gpu),
        &(WGPUSurfaceDescriptor){.nextInChain = &source.chain});
    if (!surface)
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_BACKEND, 0,
                     "fs_glfw_platform_create_surface",
                     "Metal WebGPU Surface creation failed");
    return surface;
}

void fs_glfw_platform_release_surface_source(FS_GlfwAppWindow* window) {
    if (window && window->platform_layer) {
        [(CAMetalLayer*)window->platform_layer release];
        window->platform_layer = NULL;
    }
}
FS_Result fs_glfw_platform_native_handle(FS_GlfwAppWindow* window,
                                         FS_NativeWindowHandle* out_handle,
                                         FS_Error* error) {
    (void)error;
    if (!window || !out_handle) return FS_RESULT_INVALID_ARGUMENT;
    *out_handle = (FS_NativeWindowHandle){
        sizeof(*out_handle), FS_NATIVE_HANDLE_COCOA_WINDOW,
        (uintptr_t)glfwGetCocoaWindow(window->window), 0};
    return FS_RESULT_OK;
}
#endif
