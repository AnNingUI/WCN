#include "WCN/WCN.h"
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

// 平台相关的头文件包含
#ifdef _WIN32
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
static void request_device_callback(WGPURequestDeviceStatus status,
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

int main() {
  printf("WCN GLFW Test - W3C Canvas API Implementation\n");

  // 初始化 GLFW
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    printf("Failed to initialize GLFW\n");
    return -1;
  }

  // 创建窗口
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow *window = glfwCreateWindow(800, 600, "WCN GLFW Test", NULL, NULL);
  if (!window) {
    printf("Failed to create GLFW window\n");
    glfwTerminate();
    return -1;
  }

  // 初始化 WebGPU
  WGPUInstance instance = wgpuCreateInstance(NULL);
  if (!instance) {
    printf("Failed to create WebGPU instance\n");
    glfwTerminate();
    return -1;
  }

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
  WGPUDeviceDescriptor deviceDesc = {0};
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
  };
  wgpuSurfaceConfigure(surface, &config);

  // 初始化 WCN 上下文
  WCN_Context *wcn_context = wcn_init_context();
  if (!wcn_context) {
    printf("Failed to initialize WCN context\n");
    glfwTerminate();
    return -1;
  }

  printf("Starting render loop...\n");

  // 主循环
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // 获取当前纹理
    WGPUSurfaceTexture surfaceTexture;
    wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);

    if (surfaceTexture.status !=
            WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surfaceTexture.status !=
            WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
      printf("Failed to acquire surface texture\n");
      continue;
    }

    // 创建 Canvas
    WCN_Canvas *canvas =
        wcn_create_canvas(wcn_context, device, config.format, 800, 600);
    if (!canvas) {
      printf("Failed to create WCN canvas\n");
      continue;
    }

    // 创建纹理视图并设置
    WGPUTextureView textureView =
        wgpuTextureCreateView(surfaceTexture.texture, NULL);
    wcn_canvas_set_texture_view(canvas, textureView);

    // 开始渲染
    wcn_begin_render_pass(canvas);

    // === 演示 W3C Canvas API 功能 ===

    // 1. 填充矩形
    WCN_Color red = {1.0f, 0.0f, 0.0f, 1.0f};
    wcn_set_fill_color(canvas, red);
    wcn_fill_rect(canvas, 50, 50, 100, 100);

    // 2. 描边矩形
    WCN_Color blue = {0.0f, 0.0f, 1.0f, 1.0f};
    wcn_set_stroke_color(canvas, blue);
    wcn_set_line_width(canvas, 3.0f);
    wcn_stroke_rect(canvas, 200, 50, 100, 100);

    // 3. 使用路径 API 绘制三角形
    wcn_begin_path(canvas);
    wcn_move_to(canvas, 350, 50);
    wcn_line_to(canvas, 450, 150);
    wcn_line_to(canvas, 350, 150);
    wcn_close_path(canvas);

    // 填充三角形
    WCN_Color green = {0.0f, 1.0f, 0.0f, 1.0f};
    wcn_set_fill_color(canvas, green);
    wcn_fill_path(canvas);

    // 4. 绘制弧线
    wcn_begin_path(canvas);
    wcn_arc(canvas, 550, 100, 50, 0, 3.14159f * 1.5f, false);

    // 描边弧线
    WCN_Color yellow = {1.0f, 1.0f, 0.0f, 1.0f};
    wcn_set_stroke_color(canvas, yellow);
    wcn_set_line_width(canvas, 2.0f);
    wcn_stroke_path(canvas);

    // 5. 使用贝塞尔曲线
    wcn_begin_path(canvas);
    wcn_move_to(canvas, 50, 200);
    wcn_bezier_curve_to(canvas, 150, 150, 250, 250, 300, 200);

    // 描边贝塞尔曲线
    WCN_Color purple = {1.0f, 0.0f, 1.0f, 1.0f};
    wcn_set_stroke_color(canvas, purple);
    wcn_set_line_width(canvas, 4.0f);
    wcn_stroke_path(canvas);

    // 6. 使用二次贝塞尔曲线
    wcn_begin_path(canvas);
    wcn_move_to(canvas, 350, 200);
    wcn_quadratic_curve_to(canvas, 450, 150, 500, 200);

    // 描边二次贝塞尔曲线
    WCN_Color cyan = {0.0f, 1.0f, 1.0f, 1.0f};
    wcn_set_stroke_color(canvas, cyan);
    wcn_set_line_width(canvas, 3.0f);
    wcn_stroke_path(canvas);

    // 7. 演示状态保存和恢复
    wcn_save(canvas);
    wcn_set_fill_color(canvas, (WCN_Color){1.0f, 0.5f, 0.0f, 1.0f}); // 橙色
    wcn_fill_rect(canvas, 550, 200, 50, 50);
    wcn_restore(canvas);
    wcn_fill_rect(canvas, 620, 200, 50, 50); // 应该是默认的红色

    // 结束渲染
    wcn_end_render_pass(canvas);

    // 提交渲染命令
    wcn_submit(canvas);

    // 销毁 Canvas
    wcn_destroy_canvas(canvas);

    // 显示帧
    wgpuSurfacePresent(surface);
  }

  printf("Cleaning up resources...\n");

  // 清理资源
  wcn_destroy_context(wcn_context);
  wgpuDeviceRelease(device);
  wgpuAdapterRelease(adapter);
  wgpuSurfaceRelease(surface);
  wgpuInstanceRelease(instance);
  glfwDestroyWindow(window);
  glfwTerminate();

  printf("Test completed successfully!\n");
  return 0;
}