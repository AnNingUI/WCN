#include "WCN/WCN.h"
#include <GLFW/glfw3.h>
#include <webgpu/webgpu.h>
#include <stdio.h>

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

// GLFW 错误回调
void glfw_error_callback(int error, const char *description) {
  printf("GLFW Error %d: %s\n", error, description);
}

// WebGPU 设备丢失回调
static void handle_device_lost(WGPUDevice const *device,
                               WGPUDeviceLostReason reason,
                               WGPUStringView message, void *userdata1,
                               void *userdata2) {
  printf("Device lost: %.*s\n", (int)message.length, message.data);
}

// WebGPU 错误回调
static void handle_uncaptured_error(WGPUDevice const *device,
                                    WGPUErrorType type, WGPUStringView message,
                                    void *userdata1, void *userdata2) {
  printf("Uncaptured device error: %.*s\n", (int)message.length, message.data);
}

// 请求适配器回调
static void request_adapter_callback(WGPURequestAdapterStatus status,
                                     WGPUAdapter adapter,
                                     WGPUStringView message, void *userdata1,
                                     void *userdata2) {
  if (status == WGPURequestAdapterStatus_Success) {
    *(WGPUAdapter *)userdata1 = adapter;
  } else {
    printf("Failed to get adapter: %.*s\n", (int)message.length, message.data);
  }
}

// 请求设备回调
static void request_device_callback(const WGPURequestDeviceStatus status,
                                    WGPUDevice device, WGPUStringView message,
                                    void *userdata1, void *userdata2) {
  if (status == WGPURequestDeviceStatus_Success) {
    *(WGPUDevice *)userdata1 = device;
  } else {
    printf("Failed to get device: %.*s\n", (int)message.length, message.data);
  }
}

// 创建Surface的函数
WGPUSurface CreateSurface(WGPUInstance instance, GLFWwindow *window) {
#ifdef _WIN32
  HWND hwnd = glfwGetWin32Window(window);
  WGPUSurfaceSourceWindowsHWND surfaceDesc = {
      .chain = {.sType = WGPUSType_SurfaceSourceWindowsHWND},
      .hwnd = hwnd,
      .hinstance = GetModuleHandle(NULL)};
  WGPUSurfaceDescriptor desc = {.nextInChain =
                                    (const WGPUChainedStruct *)&surfaceDesc};
  return wgpuInstanceCreateSurface(instance, &desc);
#elif defined(__linux__)
  Display *display = glfwGetX11Display();
  Window x11Window = glfwGetX11Window(window);
  WGPUSurfaceSourceXlibWindow surfaceDesc = {
      .chain = {.sType = WGPUSType_SurfaceSourceXlibWindow},
      .display = display,
      .window = x11Window};
  WGPUSurfaceDescriptor desc = {.nextInChain =
                                    (const WGPUChainedStruct *)&surfaceDesc};
  return wgpuInstanceCreateSurface(instance, &desc);
#elif defined(__APPLE__)
  void *metalLayer = glfwGetCocoaWindow(window);
  WGPUSurfaceDescriptorFromMetalLayer surfaceDesc = {
      .chain = {.sType = WGPUSType_SurfaceDescriptorFromMetalLayer},
      .layer = metalLayer};
  WGPUSurfaceDescriptor desc = {.nextInChain =
                                    (const WGPUChainedStruct *)&surfaceDesc};
  return wgpuInstanceCreateSurface(instance, &desc);
#else
  return NULL;
#endif
}
// 演示所有渲染函数（简化版，不包含帧管理）
void demonstrate_all_rendering_functions_simple(WCN_Context *ctx, int width, int height, bool print_debug) {
    if (print_debug) {
        printf("=== WCN All-in-GPU Canvas2D 完整渲染函数演示 ===\n");
    }

    // 不绘制白色背景，使用默认的浅灰色背景

    // 测试矩形绘制
    wcn_set_fill_style(ctx, 0xFFFF0000); // 红色
    wcn_fill_rect(ctx, 50, 50, 100, 80);

    wcn_set_stroke_style(ctx, 0xFF0000FF); // 蓝色
    wcn_set_line_width(ctx, 3.0f);
    wcn_stroke_rect(ctx, 200, 50, 100, 80);

    // 测试路径操作
    wcn_begin_path(ctx);
    wcn_move_to(ctx, 50, 200);
    wcn_line_to(ctx, 150, 250);
    wcn_line_to(ctx, 50, 300);
    wcn_close_path(ctx);
    wcn_set_fill_style(ctx, 0xFF00FF00); // 绿色
    wcn_fill(ctx);

    // 测试圆弧 - 完整的圆（描边）
    wcn_begin_path(ctx);
    wcn_arc(ctx, 240, 240, 40, 0, 3.14159f * 2.0f, false);
    wcn_set_stroke_style(ctx, 0xFFFF00FF); // 品红色
    wcn_set_line_width(ctx, 2.0f);
    wcn_stroke(ctx);
    
    // 测试圆弧 - 半圆（填充）
    wcn_begin_path(ctx);
    wcn_move_to(ctx, 350, 200);
    wcn_arc(ctx, 350, 200, 30, 0, 3.14159f, false);
    wcn_close_path(ctx);
    wcn_set_fill_style(ctx, 0xFF00FFFF); // 青色
    wcn_fill(ctx);
    
    // 测试圆弧 - 四分之一圆（描边）
    wcn_begin_path(ctx);
    wcn_arc(ctx, 450, 200, 35, 0, 3.14159f * 0.5f, false);
    wcn_set_stroke_style(ctx, 0xFFFF8000); // 橙色
    wcn_set_line_width(ctx, 3.0f);
    wcn_stroke(ctx);

    // 测试变换系统
    wcn_save(ctx);
    wcn_translate(ctx, 400, 150);
    wcn_rotate(ctx, 0.785f); // 45度
    wcn_scale(ctx, 1.2f, 1.2f);
    wcn_set_fill_style(ctx, 0xFFFFFF00); // 黄色
    wcn_fill_rect(ctx, -30, -30, 60, 60);  // 以原点为中心
    wcn_restore(ctx);

    // 测试样式设置
    wcn_save(ctx);
    wcn_set_global_alpha(ctx, 0.7f);
    wcn_set_fill_style(ctx, 0xFF800080); // 紫色
    wcn_fill_rect(ctx, 400, 250, 80, 80);
    wcn_restore(ctx);

    // 测试状态栈
    wcn_save(ctx);
    wcn_set_fill_style(ctx, 0xFFA500FF); // 橙色
    wcn_translate(ctx, 500, 50);
    wcn_fill_rect(ctx, 0, 0, 60, 60);
    wcn_restore(ctx);
    
    // 测试 transform() - 自定义变换矩阵
    wcn_save(ctx);
    wcn_set_fill_style(ctx, 0xFF00FFFF); // 青色
    wcn_translate(ctx, 600, 150);
    wcn_transform(ctx, 1.0f, 0.5f, -0.5f, 1.0f, 0, 0); // 倾斜变换
    wcn_fill_rect(ctx, 0, 0, 50, 50);
    wcn_restore(ctx);
    
    // 测试 set_transform() - 直接设置变换矩阵
    wcn_save(ctx);
    wcn_set_fill_style(ctx, 0xFFFF1493); // 深粉色
    wcn_set_transform(ctx, 0.8f, 0.3f, -0.3f, 0.8f, 700, 250);
    wcn_fill_rect(ctx, 0, 0, 60, 40);
    wcn_restore(ctx);
    
    // 测试 reset_transform() - 重置变换
    wcn_save(ctx);
    wcn_translate(ctx, 100, 100);
    wcn_rotate(ctx, 1.0f);
    wcn_reset_transform(ctx); // 重置为单位矩阵
    wcn_set_fill_style(ctx, 0xFF4169E1); // 皇家蓝
    wcn_fill_rect(ctx, 600, 400, 70, 50);
    wcn_restore(ctx);
}

// 演示所有渲染函数
void demonstrate_all_rendering_functions(WCN_Context *ctx, int width, int height, WGPUTextureFormat surface_format) {
    printf("=== WCN All-in-GPU Canvas2D 完整渲染函数演示 ===\n");

    // 开始帧
    wcn_begin_frame(ctx, width, height, surface_format);

    // 调用简化版的演示函数
    demonstrate_all_rendering_functions_simple(ctx, width, height, true);

    // 结束帧
    wcn_end_frame(ctx);
}

int main() {
    printf("WCN All-in-GPU Canvas2D Complete Test\n");

    // 初始化 GLFW
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return -1;
    }

    // 创建窗口
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(
        1200, 800, "WCN All-in-GPU Canvas2D Complete Test", NULL, NULL);
    if (!window) {
        printf("Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    // 等待窗口大小变为非零
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width > 0 && height > 0) {
            break;
        }
    }

    // 初始化 WebGPU
    const WGPUInstanceDescriptor instanceDesc = {};
    WGPUInstance instance = wgpuCreateInstance(&instanceDesc);
    if (!instance) {
        printf("Failed to create WebGPU instance\n");
        glfwTerminate();
        return -1;
    }

    // 获取窗口大小
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // 创建表面 - 使用新的平台无关函数
    WGPUSurface surface = CreateSurface(instance, window);
    if (!surface) {
        printf("Failed to create surface\n");
        glfwTerminate();
        return -1;
    }

    // 请求适配器
    WGPUAdapter adapter;
    wgpuInstanceRequestAdapter(instance,
                               &(WGPURequestAdapterOptions){
                                   .compatibleSurface = surface,
                               },
                               (WGPURequestAdapterCallbackInfo){
                                   .mode = WGPUCallbackMode_AllowProcessEvents,
                                   .callback = request_adapter_callback,
                                   .userdata1 = &adapter,
                               });

    if (!adapter) {
        printf("Failed to request adapter\n");
        glfwTerminate();
        return -1;
    }

    // 请求设备
    WGPUDevice device;

    // 由于 WCN 需要 TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES 特性（用于读写存储纹理）
    // 必须在设备创建时启用此特性
    // 使用原始值 0x00030002 对应 WGPUNativeFeature_TextureAdapterSpecificFormatFeatures
    WGPUFeatureName requiredFeatures[] = {
        (WGPUFeatureName)0x00030002
    };

    WGPUDeviceDescriptor deviceDesc = {0};
    deviceDesc.requiredFeatureCount = 1;
    deviceDesc.requiredFeatures = requiredFeatures;
    deviceDesc.deviceLostCallbackInfo = (WGPUDeviceLostCallbackInfo){
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = handle_device_lost,
        .userdata1 = NULL,
    };
    deviceDesc.uncapturedErrorCallbackInfo = (WGPUUncapturedErrorCallbackInfo){
        .callback = handle_uncaptured_error,
        .userdata1 = NULL,
    };

    wgpuAdapterRequestDevice(adapter, &deviceDesc,
                           (WGPURequestDeviceCallbackInfo){
                               .mode = WGPUCallbackMode_AllowProcessEvents,
                               .callback = request_device_callback,
                               .userdata1 = &device,
                           });

    if (!device) {
        printf("Failed to request device\n");
        glfwTerminate();
        return -1;
    }

    // 获取表面功能
    WGPUSurfaceCapabilities surfaceCapabilities;
    wgpuSurfaceGetCapabilities(surface, adapter, &surfaceCapabilities);

    // 创建交换链
    WGPUSurfaceConfiguration config = (WGPUSurfaceConfiguration){
        .device = device,
        .format = surfaceCapabilities.formats[0],
        .usage = WGPUTextureUsage_RenderAttachment,
        .presentMode = WGPUPresentMode_Fifo,
        .alphaMode = surfaceCapabilities.alphaModes[0],
        .width = width,
        .height = height,
    };
    wgpuSurfaceConfigure(surface, &config);

    // 创建 WCN 上下文
    WCN_Context *wcn_context = wcn_create_context(&((WCN_GPUResources){
        .device = device,
        .surface = surface,
        .queue = wgpuDeviceGetQueue(device),
        .instance = instance,
    }));
    if (!wcn_context) {
        printf("Failed to create WCN context\n");
        glfwTerminate();
        return -1;
    }

    printf("Starting All-in-GPU Canvas2D test...\n");

    // 主循环
    int frame_count = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 每100帧打印一次调试信息
        if (frame_count % 100 == 0) {
            printf("Frame %d\n", frame_count);
        }
        frame_count++;

        // 检查窗口大小是否发生变化
        int new_width, new_height;
        glfwGetFramebufferSize(window, &new_width, &new_height);

        // 如果窗口大小为 0（最小化或缩放中），跳过这一帧
        if (new_width == 0 || new_height == 0) {
            continue;
        }

        if (new_width != width || new_height != height) {
            printf("Window resized: %dx%d -> %dx%d\n", width, height, new_width, new_height);
            // 窗口大小发生变化，重新配置表面
            width = new_width;
            height = new_height;

            // 重新配置表面
            config.width = width;
            config.height = height;
            wgpuSurfaceConfigure(surface, &config);
        }

        // 获取当前纹理
        WGPUSurfaceTexture surfaceTexture;
        wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);

        if (surfaceTexture.status !=
                WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
            surfaceTexture.status !=
                WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            // 如果纹理获取失败，跳过这一帧
            printf("Failed to acquire surface texture, status: %d\n", surfaceTexture.status);
            continue;
        }

        // 开始帧（这会初始化渲染器）
        wcn_begin_frame(wcn_context, width, height, surfaceCapabilities.formats[0]);

        // 创建纹理视图
        WGPUTextureViewDescriptor texture_view_desc = {
            .nextInChain = NULL,
            .label = "Surface Texture View",
            .format = WGPUTextureFormat_Undefined, // 使用纹理的默认格式
            .dimension = WGPUTextureViewDimension_2D,
            .baseMipLevel = 0,
            .mipLevelCount = 1,
            .baseArrayLayer = 0,
            .arrayLayerCount = 1,
            .aspect = WGPUTextureAspect_All
        };
        WGPUTextureView texture_view = wgpuTextureCreateView(surfaceTexture.texture, &texture_view_desc);
        if (!texture_view) {
            printf("Failed to create texture view\n");
            wcn_end_frame(wcn_context);
            continue;
        }

        // 开始渲染通道
        if (!wcn_begin_render_pass(wcn_context, texture_view)) {
            printf("Failed to begin render pass\n");
            wgpuTextureViewRelease(texture_view);
            wcn_end_frame(wcn_context);
            continue;
        }

        // 演示所有渲染函数（现在只需要调用绘制函数，不需要begin_frame）
        // 只在第一帧打印调试信息
        demonstrate_all_rendering_functions_simple(wcn_context, width, height, frame_count == 0);

        // 结束帧（必须在 render pass 内部调用，因为会执行批次渲染）
        wcn_end_frame(wcn_context);
        
        // 结束渲染通道并提交命令
        wcn_end_render_pass(wcn_context);
        wcn_submit_commands(wcn_context);

        // 释放资源
        wgpuTextureViewRelease(texture_view);

        // 显示帧
        wgpuSurfacePresent(surface);

        // 限制帧率，避免占用过多CPU
        // glfwWaitEventsTimeout(0.016); // 约60 FPS

        // 按ESC键退出
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }
    }

    // 清理资源
    wcn_destroy_context(wcn_context);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuSurfaceRelease(surface);
    wgpuInstanceRelease(instance);
    glfwDestroyWindow(window);
    glfwTerminate();

    printf("All-in-GPU Canvas2D test completed successfully!\n");
    return 0;
}