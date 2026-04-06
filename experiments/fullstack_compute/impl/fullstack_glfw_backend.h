#ifndef WCN_FULLSTACK_GLFW_BACKEND_H
#define WCN_FULLSTACK_GLFW_BACKEND_H

#include <stdbool.h>
#include <stdint.h>

#include <GLFW/glfw3.h>
#include <webgpu/wgpu.h>

#include "../include/fullstack_core.h"
#include "../include/fullstack_effects.h"

typedef struct FS_GlfwBackend {
    GLFWwindow* window;
    uint32_t width;
    uint32_t height;
    float scroll_delta_x;
    float scroll_delta_y;

    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    WGPUSurface surface;
    WGPUTextureFormat surface_format;
    WGPUPresentMode present_mode;
    bool supports_surface_copy_src;

    FS_Core* core;
} FS_GlfwBackend;

bool fs_glfw_backend_init(FS_GlfwBackend* backend, uint32_t width, uint32_t height, const char* title);
void fs_glfw_backend_shutdown(FS_GlfwBackend* backend);
bool fs_glfw_backend_should_close(FS_GlfwBackend* backend);
void fs_glfw_backend_poll_events(void);
void fs_glfw_backend_take_scroll_delta(FS_GlfwBackend* backend, float* out_x, float* out_y);
FS_Core* fs_glfw_backend_core(FS_GlfwBackend* backend);
bool fs_glfw_backend_present(FS_GlfwBackend* backend, float clear_r, float clear_g, float clear_b, float clear_a);

#endif
