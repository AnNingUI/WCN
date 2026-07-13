#if !defined(_WIN32) && !defined(__linux__) && !defined(__APPLE__)
#include "fullstack_glfw_app_backend_private.h"

WGPUSurface fs_glfw_platform_create_surface(FS_GlfwAppWindow* window,
                                             FS_Error* error) {
    (void)window;
    FS_ERROR_SET(error, FS_RESULT_UNSUPPORTED, FS_ERROR_DOMAIN_BACKEND, 0,
                 "fs_glfw_platform_create_surface",
                 "GLFW WebGPU Surface creation is unsupported on this platform");
    return NULL;
}
void fs_glfw_platform_release_surface_source(FS_GlfwAppWindow* window) {
    (void)window;
}
FS_Result fs_glfw_platform_native_handle(FS_GlfwAppWindow* window,
                                         FS_NativeWindowHandle* out_handle,
                                         FS_Error* error) {
    (void)window;(void)out_handle;(void)error;return FS_RESULT_UNSUPPORTED;
}
#endif
