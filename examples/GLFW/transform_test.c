#include <GLFW/glfw3.h>
#include <math.h>
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

#include "WCN/WCN.h"

// GLFW 错误回调
void glfw_error_callback(int error, const char *description) {
  WCN_DEBUG_PRINT("GLFW Error %d: %s\n", error, description);
}

// WebGPU 设备丢失回调
static void handle_device_lost(WGPUDevice const *device,
                               WGPUDeviceLostReason reason,
                               WGPUStringView message, void *userdata1,
                               void *userdata2) {
  WCN_DEBUG_PRINT("Device lost: %.*s\n", (int)message.length, message.data);
}

// WebGPU 错误回调
static void handle_uncaptured_error(WGPUDevice const *device,
                                    WGPUErrorType type, WGPUStringView message,
                                    void *userdata1, void *userdata2) {
  WCN_DEBUG_PRINT("Uncaptured device error: %.*s\n", (int)message.length, message.data);
}

// 请求适配器回调
static void request_adapter_callback(WGPURequestAdapterStatus status,
                                     WGPUAdapter adapter,
                                     WGPUStringView message, void *userdata1,
                                     void *userdata2) {
  if (status == WGPURequestAdapterStatus_Success) {
    *(WGPUAdapter *)userdata1 = adapter;
  } else {
    WCN_DEBUG_PRINT("Failed to get adapter: %.*s\n", (int)message.length, message.data);
  }
}

// 请求设备回调
static void request_device_callback(WGPURequestDeviceStatus status,
                                    WGPUDevice device, WGPUStringView message,
                                    void *userdata1, void *userdata2) {
  if (status == WGPURequestDeviceStatus_Success) {
    *(WGPUDevice *)userdata1 = device;
  } else {
    WCN_DEBUG_PRINT("Failed to get device: %.*s\n", (int)message.length, message.data);
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

// 演示所有渲染函数
void demonstrate_all_rendering_functions(WCN_Canvas *canvas, int width,
                                         int height) {
  WCN_DEBUG_PRINT("=== WCN 完整渲染函数演示 ===");

  // 首先清空画布为浅灰色背景
  WCN_Color light_gray = {0.9f, 0.9f, 0.9f, 1.0f};
  wcn_clear(canvas, light_gray);
  WCN_DEBUG_PRINT("1. wcn_clear() - 清空画布为浅灰色背景");

  // === 基本形状绘制函数 ===
  WCN_DEBUG_PRINT("--- 基本形状绘制函数 ---");

  // 2. 填充矩形
  WCN_Color red = {1.0f, 0.0f, 0.0f, 1.0f};
  wcn_set_fill_color(canvas, red);
  wcn_fill_rect(canvas, 50, 50, 80, 60);
  WCN_DEBUG_PRINT("2. wcn_fill_rect() - 红色填充矩形 at (50,50) 大小 80x60");

  // 3. 描边矩形
  WCN_Color blue = {0.0f, 0.0f, 1.0f, 1.0f};
  wcn_set_stroke_color(canvas, blue);
  wcn_set_line_width(canvas, 3.0f);
  wcn_stroke_rect(canvas, 150, 50, 80, 60);
  WCN_DEBUG_PRINT("3. wcn_stroke_rect() - 蓝色描边矩形 at (150,50) 大小 80x60, 线宽 3px");

  // 4. 清除矩形区域
  WCN_Color green = {0.0f, 1.0f, 0.0f, 1.0f};
  wcn_set_fill_color(canvas, green);
  wcn_fill_rect(canvas, 250, 50, 80, 60);  // 先绘制绿色矩形
  wcn_clear_rect(canvas, 270, 70, 40, 20); // 清除中心区域
  WCN_DEBUG_PRINT("4. wcn_clear_rect() - 绿色矩形中清除中心区域");

  // 5. 填充路径（在小范围内演示）
  wcn_begin_path(canvas);
  wcn_move_to(canvas, 350, 50);
  wcn_line_to(canvas, 430, 50);
  wcn_line_to(canvas, 430, 110);
  wcn_line_to(canvas, 350, 110);
  wcn_close_path(canvas);
  
  WCN_Color purple = {0.5f, 0.0f, 0.5f, 1.0f};
  wcn_set_fill_color(canvas, purple);
  wcn_fill_path(canvas);
  WCN_DEBUG_PRINT("5. wcn_fill_path() - 紫色填充矩形路径");

  // 6. 描边路径（在小范围内演示）
  wcn_begin_path(canvas);
  wcn_move_to(canvas, 450, 50);
  wcn_line_to(canvas, 530, 50);
  wcn_line_to(canvas, 530, 110);
  wcn_line_to(canvas, 450, 110);
  wcn_close_path(canvas);

  WCN_Color orange = {1.0f, 0.5f, 0.0f, 1.0f};
  wcn_set_stroke_color(canvas, orange);
  wcn_set_line_width(canvas, 2.0f);
  wcn_stroke_path(canvas);
  WCN_DEBUG_PRINT("6. wcn_stroke_path() - 橙色描边矩形路径");

  // === 路径操作函数 ===
  WCN_DEBUG_PRINT("\n--- 路径操作函数 ---");

  // 7. 路径操作：绘制三角形
  wcn_begin_path(canvas);
  WCN_DEBUG_PRINT("7. wcn_begin_path() - 开始新路径");

  wcn_move_to(canvas, 50, 150);
  WCN_DEBUG_PRINT("8. wcn_move_to() - 移动到起始点 (50,150)");

  wcn_line_to(canvas, 120, 150);
  wcn_line_to(canvas, 85, 200);
  WCN_DEBUG_PRINT("9. wcn_line_to() - 绘制线条到各个顶点");

  wcn_close_path(canvas);
  WCN_DEBUG_PRINT("10. wcn_close_path() - 闭合路径");

  WCN_Color cyan = {0.0f, 1.0f, 1.0f, 1.0f};
  wcn_set_fill_color(canvas, cyan);
  wcn_fill_path(canvas);
  WCN_DEBUG_PRINT("11. wcn_fill_path() - 青色填充三角形路径");

  // 8. 描边路径：绘制另一个三角形
  wcn_begin_path(canvas);
  wcn_move_to(canvas, 150, 150);
  wcn_line_to(canvas, 220, 150);
  wcn_line_to(canvas, 185, 200);
  wcn_close_path(canvas);

  WCN_Color magenta = {1.0f, 0.0f, 1.0f, 1.0f};
  wcn_set_stroke_color(canvas, magenta);
  wcn_set_line_width(canvas, 2.0f);
  wcn_stroke_path(canvas);
  WCN_DEBUG_PRINT("12. wcn_stroke_path() - 洋红色描边三角形路径");

  // === 高级路径绘制函数 ===
  WCN_DEBUG_PRINT("\n--- 高级路径绘制函数 ---");

  // 9. 三次贝塞尔曲线
  wcn_begin_path(canvas);
  wcn_move_to(canvas, 250, 150);
  wcn_bezier_curve_to(canvas, 280, 120, 320, 180, 350, 150);

  WCN_Color dark_green = {0.0f, 0.5f, 0.0f, 1.0f};
  wcn_set_stroke_color(canvas, dark_green);
  wcn_set_line_width(canvas, 3.0f);
  wcn_stroke_path(canvas);
  WCN_DEBUG_PRINT("13. wcn_bezier_curve_to() - 深绿色三次贝塞尔曲线");

  // 10. 二次贝塞尔曲线
  wcn_begin_path(canvas);
  wcn_move_to(canvas, 380, 150);
  wcn_quadratic_curve_to(canvas, 415, 120, 450, 150);

  WCN_Color brown = {0.6f, 0.3f, 0.1f, 1.0f};
  wcn_set_stroke_color(canvas, brown);
  wcn_set_line_width(canvas, 3.0f);
  wcn_stroke_path(canvas);
  WCN_DEBUG_PRINT("14. wcn_quadratic_curve_to() - 棕色二次贝塞尔曲线");

  // 11. 弧线绘制
  wcn_begin_path(canvas);
  wcn_arc(canvas, 520, 175, 30, 0, 3.14159f * 1.5f, false);

  WCN_Color yellow = {1.0f, 1.0f, 0.0f, 1.0f};
  wcn_set_stroke_color(canvas, yellow);
  wcn_set_line_width(canvas, 4.0f);
  wcn_stroke_path(canvas);
  WCN_DEBUG_PRINT("15. wcn_arc() - 黄色弧线（270度）");

  // 12. 填充弧线（圆形扇区）
  wcn_begin_path(canvas);
  wcn_move_to(canvas, 580, 175);                     // 移动到圆心
  wcn_arc(canvas, 580, 175, 25, 0, 3.14159f, false); // 半圆弧
  wcn_close_path(canvas);

  WCN_Color pink = {1.0f, 0.7f, 0.8f, 1.0f};
  wcn_set_fill_color(canvas, pink);
  wcn_fill_path(canvas);
  WCN_DEBUG_PRINT("16. wcn_arc() + wcn_fill_path() - 粉色半圆扇形");

  // === 变换函数演示 ===
  WCN_DEBUG_PRINT("\n--- 变换函数演示 ---");

  // 13. 平移变换
  WCN_SAVE_RESTORE(canvas) {
    WCN_Color violet = {0.5f, 0.0f, 1.0f, 1.0f};
    wcn_set_fill_color(canvas, violet);
    wcn_translate(canvas, 100, 50);
    wcn_fill_rect(canvas, 50, 250, 40, 40);
    WCN_DEBUG_PRINT("17. wcn_translate() - 紫色矩形（平移变换）");
  }

  // 14. 旋转变换
  WCN_SAVE_RESTORE(canvas) {
    WCN_Color lime = {0.5f, 1.0f, 0.0f, 1.0f};
    wcn_set_fill_color(canvas, lime);
    wcn_translate(canvas, 200, 270);         // 先移动到旋转中心
    wcn_rotate(canvas, 0.785f);              // 45度旋转
    wcn_fill_rect(canvas, -20, -20, 40, 40); // 在中心绘制
    WCN_DEBUG_PRINT("18. wcn_rotate() - 青柠色矩形（45度旋转）");
  }

  // 15. 缩放变换
  WCN_SAVE_RESTORE(canvas) {
    WCN_Color coral = {1.0f, 0.5f, 0.3f, 1.0f};
    wcn_set_fill_color(canvas, coral);
    wcn_translate(canvas, 300, 270);
    wcn_scale(canvas, 1.5f, 1.5f);
    wcn_fill_rect(canvas, -15, -15, 30, 30);
    WCN_DEBUG_PRINT("19. wcn_scale() - 珊瑚色矩形（1.5倍缩放）");
  }

  // 16. 复合变换
  WCN_SAVE_RESTORE(canvas) {
    WCN_Color gold = {1.0f, 0.8f, 0.0f, 1.0f};
    wcn_set_fill_color(canvas, gold);
    wcn_translate(canvas, 400, 270);
    wcn_rotate(canvas, 0.524f); // 30度
    wcn_scale(canvas, 1.2f, 0.8f);
    wcn_fill_rect(canvas, -20, -15, 40, 30);
    WCN_DEBUG_PRINT("20. 复合变换 - 金色矩形（平移+旋转+缩放）");
  }

  // === 状态管理函数 ===
  WCN_DEBUG_PRINT("\n--- 状态管理函数 ---");

  // 17. 状态保存与恢复演示
  WCN_Color original_color = {0.2f, 0.2f, 0.8f, 1.0f};
  wcn_set_fill_color(canvas, original_color);
  wcn_fill_rect(canvas, 50, 320, 30, 30);
  WCN_DEBUG_PRINT("21. 原始状态 - 深蓝色矩形");

  wcn_save(canvas);
  WCN_DEBUG_PRINT("22. wcn_save() - 保存当前状态");

  WCN_Color temp_color = {0.8f, 0.2f, 0.2f, 1.0f};
  wcn_set_fill_color(canvas, temp_color);
  wcn_set_line_width(canvas, 5.0f);
  wcn_fill_rect(canvas, 100, 320, 30, 30);
  WCN_DEBUG_PRINT("23. 临时状态 - 红色矩形，线宽5px");

  wcn_restore(canvas);
  WCN_DEBUG_PRINT("24. wcn_restore() - 恢复保存的状态");

  wcn_fill_rect(canvas, 150, 320, 30, 30);
  WCN_DEBUG_PRINT("25. 恢复后状态 - 应该是深蓝色矩形");

  // === 裁剪功能演示 ===
  WCN_DEBUG_PRINT("\n--- 裁剪功能演示 ---");

  // 18. 裁剪路径演示
  WCN_SAVE_RESTORE(canvas) {
    // 创建圆形裁剪区域
    wcn_begin_path(canvas);
    wcn_arc(canvas, 300, 345, 40, 0, 2 * 3.14159f, false);
    wcn_clip_path(canvas);
    WCN_DEBUG_PRINT("26. wcn_clip_path() - 设置圆形裁剪区域");

    // 在裁剪区域内绘制矩形
    WCN_Color navy = {0.0f, 0.0f, 0.5f, 1.0f};
    wcn_set_fill_color(canvas, navy);
    wcn_fill_rect(canvas, 260, 305, 80, 80);
    WCN_DEBUG_PRINT("27. 在裁剪区域内绘制 - 海军蓝矩形被裁剪成圆形");
  }

  // === 颜色和样式设置函数 ===
  WCN_DEBUG_PRINT("\n--- 颜色和样式设置函数 ---");

  // 19. 不同线宽演示
  WCN_Color black = {0.0f, 0.0f, 0.0f, 1.0f};
  wcn_set_stroke_color(canvas, black);

  for (int i = 1; i <= 5; i++) {
    wcn_set_line_width(canvas, (float)i);
    wcn_stroke_rect(canvas, 400 + i * 25, 320, 20, 20);
  }
  WCN_DEBUG_PRINT("28. wcn_set_line_width() - 不同线宽演示（1-5px）");

  // 20. 不同颜色演示
  WCN_Color colors[] = {
      {1.0f, 0.0f, 0.0f, 1.0f}, // 红
      {0.0f, 1.0f, 0.0f, 1.0f}, // 绿
      {0.0f, 0.0f, 1.0f, 1.0f}, // 蓝
      {1.0f, 1.0f, 0.0f, 1.0f}, // 黄
      {1.0f, 0.0f, 1.0f, 1.0f}, // 洋红
      {0.0f, 1.0f, 1.0f, 1.0f}  // 青
  };

  for (int i = 0; i < 6; i++) {
    wcn_set_fill_color(canvas, colors[i]);
    wcn_fill_rect(canvas, 50 + i * 35, 380, 30, 30);
  }
  WCN_DEBUG_PRINT("29. wcn_set_fill_color() - 彩虹色矩形演示");
  WCN_DEBUG_PRINT("30. wcn_set_stroke_color() - 描边颜色设置");

  // === 复杂路径演示 ===
  WCN_DEBUG_PRINT("\n--- 复杂路径演示 ---");

  // 21. 复杂路径：心形
  wcn_begin_path(canvas);
  float heart_x = 400, heart_y = 400;
  wcn_move_to(canvas, heart_x, heart_y + 10);
  wcn_bezier_curve_to(canvas, heart_x, heart_y, heart_x - 15, heart_y - 15,
                      heart_x - 30, heart_y);
  wcn_bezier_curve_to(canvas, heart_x - 45, heart_y + 15, heart_x - 45,
                      heart_y + 30, heart_x - 30, heart_y + 30);
  wcn_bezier_curve_to(canvas, heart_x - 15, heart_y + 45, heart_x, heart_y + 60,
                      heart_x, heart_y + 60);
  wcn_bezier_curve_to(canvas, heart_x, heart_y + 60, heart_x + 15, heart_y + 45,
                      heart_x + 30, heart_y + 30);
  wcn_bezier_curve_to(canvas, heart_x + 45, heart_y + 30, heart_x + 45,
                      heart_y + 15, heart_x + 30, heart_y);
  wcn_bezier_curve_to(canvas, heart_x + 15, heart_y - 15, heart_x, heart_y,
                      heart_x, heart_y + 10);
  wcn_close_path(canvas);

  WCN_Color heart_red = {0.8f, 0.1f, 0.1f, 1.0f};
  wcn_set_fill_color(canvas, heart_red);
  wcn_fill_path(canvas);
  WCN_DEBUG_PRINT("31. 复杂路径演示 - 红色心形（多个贝塞尔曲线）");

  // 22. 星形路径
  wcn_begin_path(canvas);
  float star_x = 500, star_y = 430;
  float outer_radius = 25, inner_radius = 12;

  for (int i = 0; i < 10; i++) {
    float angle = i * 3.14159f / 5;
    float radius = (i % 2 == 0) ? outer_radius : inner_radius;
    float x = star_x + radius * cosf(angle - 3.14159f / 2);
    float y = star_y + radius * sinf(angle - 3.14159f / 2);

    if (i == 0) {
      wcn_move_to(canvas, x, y);
    } else {
      wcn_line_to(canvas, x, y);
    }
  }
  wcn_close_path(canvas);

  WCN_Color star_gold = {1.0f, 0.8f, 0.0f, 1.0f};
  wcn_set_fill_color(canvas, star_gold);
  wcn_fill_path(canvas);
  WCN_DEBUG_PRINT("32. 复杂路径演示 - 金色五角星");

  WCN_DEBUG_PRINT("\n=== 所有渲染函数演示完成！===");
  WCN_DEBUG_PRINT("总计演示了 32 个不同的渲染功能和效果\n");
}

int main() {
  WCN_DEBUG_PRINT("WCN Complete Rendering Functions Test");

  // 初始化 GLFW
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    WCN_DEBUG_PRINT("Failed to initialize GLFW");
    return -1;
  }

  // 创建窗口
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow *window = glfwCreateWindow(
      1200, 800, "WCN Complete Rendering Functions Test", NULL, NULL);
  if (!window) {
    WCN_DEBUG_PRINT("Failed to create GLFW window");
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
  WGPUInstance instance = wgpuCreateInstance(NULL);
  if (!instance) {
    WCN_DEBUG_PRINT("Failed to create WebGPU instance");
    glfwTerminate();
    return -1;
  }

  // 获取窗口大小
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);

  // 创建表面 - 使用新的平台无关函数
  WGPUSurface surface = CreateSurface(instance, window);
  if (!surface) {
    WCN_DEBUG_PRINT("Failed to create surface");
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
    WCN_DEBUG_PRINT("Failed to request adapter");
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
    WCN_DEBUG_PRINT("Failed to request device");
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

  // 初始化 WCN 上下文
  WCN_Context *wcn_context = wcn_init_context();
  if (!wcn_context) {
    WCN_DEBUG_PRINT("Failed to initialize WCN context");
    glfwTerminate();
    return -1;
  }

  WCN_DEBUG_PRINT("Starting complete rendering functions test...");

  // 创建 Canvas（在循环外创建一次）
  WCN_Canvas *canvas =
      wcn_create_canvas(wcn_context, device, config.format, width, height);
  if (!canvas) {
    WCN_DEBUG_PRINT("Failed to create WCN canvas");
    glfwTerminate();
    return -1;
  }

  WCN_DEBUG_PRINT("Canvas created successfully with format: %d, size: %dx%d\n",
         config.format, width, height);

  // 主循环
  int frame_count = 0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // 每100帧打印一次调试信息
    if (frame_count % 100 == 0) {
      WCN_DEBUG_PRINT("Frame %d\n", frame_count);
    }
    frame_count++;

    // 检查窗口大小是否发生变化
    int new_width, new_height;
    glfwGetFramebufferSize(window, &new_width, &new_height);
    if (new_width != width || new_height != height) {
      WCN_DEBUG_PRINT("Window resized: %dx%d -> %dx%d\n", width, height, new_width,
             new_height);
      // 窗口大小发生变化，重新配置表面
      width = new_width;
      height = new_height;

      // 重新配置表面
      config.width = width;
      config.height = height;
      wgpuSurfaceConfigure(surface, &config);

      // 更新Canvas尺寸
      wcn_canvas_set_size(canvas, width, height);
    }

    // 获取当前纹理
    WGPUSurfaceTexture surfaceTexture;
    wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);

    if (surfaceTexture.status !=
            WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surfaceTexture.status !=
            WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
      // 如果纹理获取失败，跳过这一帧
      WCN_DEBUG_PRINT("Failed to acquire surface texture, status: %d\n",
             surfaceTexture.status);
      continue;
    }

    // 创建纹理视图并设置
    WGPUTextureView textureView =
        wgpuTextureCreateView(surfaceTexture.texture, NULL);
    wcn_canvas_set_texture_view(canvas, textureView);

    // 开始渲染
    WCN_BEGIN_END_RENDER_PASS(canvas) {
      // 演示所有渲染函数
      demonstrate_all_rendering_functions(canvas, width, height);
    }

    // 提交渲染命令
    wcn_submit(canvas);

    // 释放纹理视图
    wgpuTextureViewRelease(textureView);

    // 显示帧
    wgpuSurfacePresent(surface);

    // 限制帧率，避免占用过多CPU
    // glfwWaitEventsTimeout(0.016); // 约60 FPS

    // 按ESC键退出
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
      break;
    }
  }

  // 销毁 Canvas（在循环外销毁）
  wcn_destroy_canvas(canvas);

  WCN_DEBUG_PRINT("Cleaning up resources...");

  // 清理资源
  wcn_destroy_context(wcn_context);
  wgpuDeviceRelease(device);
  wgpuAdapterRelease(adapter);
  wgpuSurfaceRelease(surface);
  wgpuInstanceRelease(instance);
  glfwDestroyWindow(window);
  glfwTerminate();

  WCN_DEBUG_PRINT("Complete rendering functions test completed successfully!");
  return 0;
}