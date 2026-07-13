#ifndef FULLSTACK_GLFW_APP_BACKEND_H
#define FULLSTACK_GLFW_APP_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fullstack_app.h"

typedef struct GLFWwindow GLFWwindow;

FS_API FS_Result FS_CALL fs_app_register_glfw_backend(FS_Error* error);
FS_API GLFWwindow* FS_CALL fs_glfw_backend_native_window(
    FS_BackendWindow* window);

#ifdef __cplusplus
}
#endif
#endif
