#ifndef WCN_GLFW_IMPL_H
#define WCN_GLFW_IMPL_H

#include "WCN/WCN.h"
#include <GLFW/glfw3.h>
#include <webgpu/wgpu.h>
#include <stdio.h>
#include <stdlib.h>

// 平台相关的头文件包含
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <windows.h>
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#include <X11/Xlib.h>
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>
#endif

// ============================================================================
// GLFW + WebGPU 集成实现（基于成功的测试程序）
// ============================================================================

// GLFW 窗口包装器
typedef struct {
    GLFWwindow* window;
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    WGPUSurface surface;
    WGPUTextureFormat surface_format;
    
    uint32_t width;
    uint32_t height;
    
    WCN_Context* wcn_ctx;
} WCN_GLFW_Window;

// 错误回调
static void wcn_glfw_error_callback(int error, const char* description) {
    printf("GLFW Error %d: %s\n", error, description);
}

// 设备丢失回调
static void wcn_glfw_device_lost_callback(WGPUDevice const* device,
                                          WGPUDeviceLostReason reason,
                                          WGPUStringView message,
                                          void* userdata1, void* userdata2) {
    printf("Device lost: %.*s\n", (int)message.length, message.data);
}

// 未捕获错误回调
static void wcn_glfw_uncaptured_error_callback(WGPUDevice const* device,
                                               WGPUErrorType type,
                                               WGPUStringView message,
                                               void* userdata1, void* userdata2) {
    printf("Uncaptured device error: %.*s\n", (int)message.length, message.data);
}

// Adapter 请求回调
static void wcn_glfw_adapter_callback(WGPURequestAdapterStatus status,
                                      WGPUAdapter adapter,
                                      WGPUStringView message,
                                      void* userdata1, void* userdata2) {
    if (status == WGPURequestAdapterStatus_Success) {
        *(WGPUAdapter*)userdata1 = adapter;
    } else {
        printf("Failed to get adapter: %.*s\n", (int)message.length, message.data);
    }
}

// Device 请求回调
static void wcn_glfw_device_callback(WGPURequestDeviceStatus status,
                                     WGPUDevice device,
                                     WGPUStringView message,
                                     void* userdata1, void* userdata2) {
    if (status == WGPURequestDeviceStatus_Success) {
        *(WGPUDevice*)userdata1 = device;
    } else {
        printf("Failed to get device: %.*s\n", (int)message.length, message.data);
    }
}

// 创建 Surface（跨平台）
static WGPUSurface wcn_glfw_create_surface(WGPUInstance instance, GLFWwindow* window) {
#if defined(_WIN32)
    HWND hwnd = glfwGetWin32Window(window);
    WGPUSurfaceSourceWindowsHWND surfaceDesc = {
        .chain = {.sType = WGPUSType_SurfaceSourceWindowsHWND},
        .hwnd = hwnd,
        .hinstance = GetModuleHandle(NULL)
    };
    WGPUSurfaceDescriptor desc = {
        .nextInChain = (const WGPUChainedStruct*)&surfaceDesc
    };
    return wgpuInstanceCreateSurface(instance, &desc);
#elif defined(__linux__)
    Display* display = glfwGetX11Display();
    Window x11Window = glfwGetX11Window(window);
    WGPUSurfaceSourceXlibWindow surfaceDesc = {
        .chain = {.sType = WGPUSType_SurfaceSourceXlibWindow},
        .display = display,
        .window = x11Window
    };
    WGPUSurfaceDescriptor desc = {
        .nextInChain = (const WGPUChainedStruct*)&surfaceDesc
    };
    return wgpuInstanceCreateSurface(instance, &desc);
#elif defined(__APPLE__)
    void* metalLayer = glfwGetCocoaWindow(window);
    WGPUSurfaceDescriptorFromMetalLayer surfaceDesc = {
        .chain = {.sType = WGPUSType_SurfaceDescriptorFromMetalLayer},
        .layer = metalLayer
    };
    WGPUSurfaceDescriptor desc = {
        .nextInChain = (const WGPUChainedStruct*)&surfaceDesc
    };
    return wgpuInstanceCreateSurface(instance, &desc);
#else
    return NULL;
#endif
}

// 创建 GLFW 窗口和 WebGPU 上下文（基于成功的测试程序）
static WCN_GLFW_Window* wcn_glfw_create_window(uint32_t width, uint32_t height, const char* title) {
    // 初始化 GLFW
    glfwSetErrorCallback(wcn_glfw_error_callback);
    if (!glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return NULL;
    }
    
    // 创建窗口
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window) {
        printf("Failed to create GLFW window\n");
        glfwTerminate();
        return NULL;
    }
    
    // 等待窗口有效尺寸
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        if (w > 0 && h > 0) {
            width = w;
            height = h;
            break;
        }
    }
    
    // 创建包装器
    WCN_GLFW_Window* wcn_window = malloc(sizeof(WCN_GLFW_Window));
    if (!wcn_window) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return NULL;
    }
    
    wcn_window->window = window;
    wcn_window->width = width;
    wcn_window->height = height;
    
    // 创建 WebGPU 实例
    WGPUInstanceDescriptor instance_desc = {0};
    wcn_window->instance = wgpuCreateInstance(&instance_desc);
    if (!wcn_window->instance) {
        printf("Failed to create WebGPU instance\n");
        free(wcn_window);
        glfwDestroyWindow(window);
        glfwTerminate();
        return NULL;
    }
    
    // 创建 surface
    wcn_window->surface = wcn_glfw_create_surface(wcn_window->instance, window);
    if (!wcn_window->surface) {
        printf("Failed to create surface\n");
        wgpuInstanceRelease(wcn_window->instance);
        free(wcn_window);
        glfwDestroyWindow(window);
        glfwTerminate();
        return NULL;
    }
    
    // 请求 adapter
    wcn_window->adapter = NULL;
    wgpuInstanceRequestAdapter(
        wcn_window->instance,
        &(WGPURequestAdapterOptions){.compatibleSurface = wcn_window->surface},
        (WGPURequestAdapterCallbackInfo){
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = wcn_glfw_adapter_callback,
            .userdata1 = &wcn_window->adapter
        }
    );
    
    if (!wcn_window->adapter) {
        printf("Failed to request adapter\n");
        wgpuSurfaceRelease(wcn_window->surface);
        wgpuInstanceRelease(wcn_window->instance);
        free(wcn_window);
        glfwDestroyWindow(window);
        glfwTerminate();
        return NULL;
    }
    
    // 请求 device
    WGPUFeatureName requiredFeatures[] = {(WGPUFeatureName)0x00030002};
    WGPUDeviceDescriptor device_desc = {
        .requiredFeatureCount = 1,
        .requiredFeatures = requiredFeatures,
        .deviceLostCallbackInfo = (WGPUDeviceLostCallbackInfo){
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = wcn_glfw_device_lost_callback
        },
        .uncapturedErrorCallbackInfo = (WGPUUncapturedErrorCallbackInfo){
            .callback = wcn_glfw_uncaptured_error_callback
        }
    };
    
    wcn_window->device = NULL;
    wgpuAdapterRequestDevice(
        wcn_window->adapter,
        &device_desc,
        (WGPURequestDeviceCallbackInfo){
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = wcn_glfw_device_callback,
            .userdata1 = &wcn_window->device
        }
    );
    
    if (!wcn_window->device) {
        printf("Failed to request device\n");
        wgpuAdapterRelease(wcn_window->adapter);
        wgpuSurfaceRelease(wcn_window->surface);
        wgpuInstanceRelease(wcn_window->instance);
        free(wcn_window);
        glfwDestroyWindow(window);
        glfwTerminate();
        return NULL;
    }
    
    // 获取 surface 能力
    WGPUSurfaceCapabilities surfaceCapabilities;
    wgpuSurfaceGetCapabilities(wcn_window->surface, wcn_window->adapter, &surfaceCapabilities);
    wcn_window->surface_format = surfaceCapabilities.formats[0];
    
    // 配置 surface
    WGPUSurfaceConfiguration config = {
        .device = wcn_window->device,
        .format = wcn_window->surface_format,
        .usage = WGPUTextureUsage_RenderAttachment,
        .presentMode = WGPUPresentMode_Fifo,
        .alphaMode = surfaceCapabilities.alphaModes[0],
        .width = width,
        .height = height
    };
    wgpuSurfaceConfigure(wcn_window->surface, &config);
    
    // 获取队列
    wcn_window->queue = wgpuDeviceGetQueue(wcn_window->device);
    
    // 创建 WCN 上下文
    WCN_GPUResources gpu_resources = {
        .instance = wcn_window->instance,
        .device = wcn_window->device,
        .queue = wcn_window->queue,
        .surface = wcn_window->surface
    };
    
    wcn_window->wcn_ctx = wcn_create_context(&gpu_resources);
    if (!wcn_window->wcn_ctx) {
        printf("Failed to create WCN context\n");
        wgpuQueueRelease(wcn_window->queue);
        wgpuDeviceRelease(wcn_window->device);
        wgpuAdapterRelease(wcn_window->adapter);
        wgpuSurfaceRelease(wcn_window->surface);
        wgpuInstanceRelease(wcn_window->instance);
        free(wcn_window);
        glfwDestroyWindow(window);
        glfwTerminate();
        return NULL;
    }
    
    printf("GLFW 窗口创建成功: %dx%d\n", width, height);
    
    return wcn_window;
}

// 销毁窗口
static void wcn_glfw_destroy_window(WCN_GLFW_Window* wcn_window) {
    if (!wcn_window) return;
    
    if (wcn_window->wcn_ctx) {
        wcn_destroy_context(wcn_window->wcn_ctx);
    }
    
    if (wcn_window->queue) wgpuQueueRelease(wcn_window->queue);
    if (wcn_window->device) wgpuDeviceRelease(wcn_window->device);
    if (wcn_window->adapter) wgpuAdapterRelease(wcn_window->adapter);
    if (wcn_window->surface) wgpuSurfaceRelease(wcn_window->surface);
    if (wcn_window->instance) wgpuInstanceRelease(wcn_window->instance);
    
    if (wcn_window->window) {
        glfwDestroyWindow(wcn_window->window);
    }
    
    free(wcn_window);
    glfwTerminate();
}

// 窗口是否应该关闭
static inline bool wcn_glfw_window_should_close(WCN_GLFW_Window* wcn_window) {
    return glfwWindowShouldClose(wcn_window->window);
}

// 轮询事件
static inline void wcn_glfw_poll_events(void) {
    glfwPollEvents();
}

// 获取当前帧纹理
static WGPUTextureView wcn_glfw_get_current_texture_view(WCN_GLFW_Window* wcn_window) {
    WGPUSurfaceTexture surface_texture;
    wgpuSurfaceGetCurrentTexture(wcn_window->surface, &surface_texture);
    
    if (surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        printf("Failed to get current texture: status=%d\n", surface_texture.status);
        return NULL;
    }
    
    WGPUTextureViewDescriptor view_desc = {
        .format = WGPUTextureFormat_Undefined,
        .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All
    };
    
    return wgpuTextureCreateView(surface_texture.texture, &view_desc);
}

// 呈现
static inline void wcn_glfw_present(WCN_GLFW_Window* wcn_window) {
    wgpuSurfacePresent(wcn_window->surface);
}

// 获取 WCN 上下文
static inline WCN_Context* wcn_glfw_get_context(WCN_GLFW_Window* wcn_window) {
    return wcn_window->wcn_ctx;
}

// 获取窗口尺寸
static inline void wcn_glfw_get_size(WCN_GLFW_Window* wcn_window, uint32_t* width, uint32_t* height) {
    *width = wcn_window->width;
    *height = wcn_window->height;
}

// 处理窗口大小变化
static inline void wcn_glfw_handle_resize(WCN_GLFW_Window* wcn_window, uint32_t new_width, uint32_t new_height) {
    if (!wcn_window || new_width == 0 || new_height == 0) {
        return;
    }

    // 更新窗口尺寸
    wcn_window->width = new_width;
    wcn_window->height = new_height;

    // 获取 surface 能力以获取正确的 alphaMode
    WGPUSurfaceCapabilities surfaceCapabilities;
    wgpuSurfaceGetCapabilities(wcn_window->surface, wcn_window->adapter, &surfaceCapabilities);

    // 重新配置表面
    WGPUSurfaceConfiguration config = {
        .device = wcn_window->device,
        .format = wcn_window->surface_format,
        .usage = WGPUTextureUsage_RenderAttachment,
        .presentMode = WGPUPresentMode_Fifo,
        .alphaMode = surfaceCapabilities.alphaModes[0],
        .width = new_width,
        .height = new_height
    };
    wgpuSurfaceConfigure(wcn_window->surface, &config);

    printf("Surface reconfigured for new size: %dx%d\n", new_width, new_height);
}

// ============================================================================
// 渲染循环辅助函数
// ============================================================================

// 渲染帧上下文（用于管理渲染资源）
typedef struct WCN_GLFW_RenderFrame {
    WGPUSurfaceTexture surface_texture;
    WGPUTextureView texture_view;
    bool is_valid;
} WCN_GLFW_RenderFrame;

// 开始渲染帧
static inline bool wcn_glfw_begin_frame(WCN_GLFW_Window* wcn_window, WCN_GLFW_RenderFrame* frame) {
    // 获取当前表面纹理
    wgpuSurfaceGetCurrentTexture(wcn_window->surface, &frame->surface_texture);
    
    if (frame->surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        frame->surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        printf("Failed to get surface texture: status=%d\n", frame->surface_texture.status);
        frame->is_valid = false;
        return false;
    }
    
    // 创建纹理视图
    WGPUTextureViewDescriptor view_desc = {
        .format = WGPUTextureFormat_Undefined,
        .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All
    };
    frame->texture_view = wgpuTextureCreateView(frame->surface_texture.texture, &view_desc);
    
    if (!frame->texture_view) {
        printf("Failed to create texture view\n");
        // Note: We don't release the surface texture here as it's managed by the surface
        frame->is_valid = false;
        return false;
    }
    
    // 开始 WCN 帧（使用窗口创建时的格式）
    wcn_begin_frame(wcn_window->wcn_ctx, wcn_window->width, wcn_window->height, wcn_window->surface_format);
    
    // 开始渲染通道（使用 WCN 公共 API）
    if (!wcn_begin_render_pass(wcn_window->wcn_ctx, frame->texture_view)) {
        printf("Failed to begin render pass\n");
        wgpuTextureViewRelease(frame->texture_view);
        // Note: We don't release the surface texture here as it's managed by the surface
        frame->is_valid = false;
        return false;
    }
    
    frame->is_valid = true;
    return true;
}

// 结束渲染帧
static inline void wcn_glfw_end_frame(WCN_GLFW_Window* wcn_window, WCN_GLFW_RenderFrame* frame) {
    if (!frame->is_valid) {
        return;
    }
    
    // 结束 WCN 帧（执行批次渲染，必须在 render pass 内部调用）
    wcn_end_frame(wcn_window->wcn_ctx);
    
    // 结束渲染通道并提交命令（使用 WCN 公共 API）
    // Note: wcn_end_render_pass already releases the texture view, so we don't need to do it here
    wcn_end_render_pass(wcn_window->wcn_ctx);
    wcn_submit_commands(wcn_window->wcn_ctx);
    
    // 呈现
    wgpuSurfacePresent(wcn_window->surface);
    
    // 清理资源 - texture_view is already released in wcn_end_render_pass
    // Only release the surface texture, not the view
    wgpuTextureRelease(frame->surface_texture.texture);
    
    frame->is_valid = false;
}

// 简化的渲染循环辅助函数
// 使用示例：
// while (!wcn_glfw_window_should_close(window)) {
//     wcn_glfw_poll_events();
//     
//     WCN_GLFW_RenderFrame frame;
//     if (wcn_glfw_begin_frame(window, &frame)) {
//         WCN_Context* ctx = wcn_glfw_get_context(window);
//         
//         // 你的绘制代码
//         wcn_set_fill_style(ctx, 0xFFFF0000);
//         wcn_fill_rect(ctx, 100, 100, 200, 150);
//         
//         wcn_glfw_end_frame(window, &frame);
//     }
// }

#endif // WCN_GLFW_IMPL_H
