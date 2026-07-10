// wgpu_minimal_triangle.c — 最小 wgpu + GLFW 三角形渲染
// 用于验证 Dawn 的基准 VRAM 开销，不依赖 WCN/FS 任何代码。

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#endif

#include <webgpu/webgpu.h>

// ── WGSL 着色器 ────────────────────────────────────────────────────────
static const char* SHADER_WGSL =
    "@vertex\n"
    "fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {\n"
    "    let pos = array(vec2( 0.0,  0.5), vec2(-0.5, -0.5), vec2( 0.5, -0.5));\n"
    "    return vec4<f32>(pos[vid], 0.0, 1.0);\n"
    "}\n"
    "@fragment\n"
    "fn fs_main() -> @location(0) vec4<f32> {\n"
    "    return vec4<f32>(1.0, 0.0, 0.0, 1.0);\n"
    "}\n";

// ── GLFW 错误回调 ──────────────────────────────────────────────────────
static void glfw_error_cb(int err, const char* desc) {
    fprintf(stderr, "GLFW error %d: %s\n", err, desc);
}

// ── wgpu 回调（与项目一致的签名） ──────────────────────────────────────
static void adapter_cb(
    WGPURequestAdapterStatus status, WGPUAdapter adapter,
    WGPUStringView message, void* userdata1, void* userdata2)
{
    (void)userdata2; (void)message;
    if (status == WGPURequestAdapterStatus_Success)
        *(WGPUAdapter*)userdata1 = adapter;
}
static void device_cb(
    WGPURequestDeviceStatus status, WGPUDevice device,
    WGPUStringView message, void* userdata1, void* userdata2)
{
    (void)userdata2; (void)message;
    if (status == WGPURequestDeviceStatus_Success)
        *(WGPUDevice*)userdata1 = device;
}

// ── 平台相关 surface ────────────────────────────────────────────────────
#if defined(_WIN32)
#include <windows.h>
static WGPUSurface make_surface(WGPUInstance inst, GLFWwindow* win) {
    WGPUSurfaceSourceWindowsHWND src = {
        .chain = {.sType = WGPUSType_SurfaceSourceWindowsHWND},
        .hinstance = GetModuleHandle(NULL), .hwnd = glfwGetWin32Window(win),
    };
    return wgpuInstanceCreateSurface(inst, &(WGPUSurfaceDescriptor){.nextInChain = &src.chain});
}
#endif

// ── 创建着色器模块 ──────────────────────────────────────────────────────
static WGPUShaderModule make_shader(WGPUDevice dev, const char* wgsl) {
    WGPUStringView sv = {.data = wgsl, .length = strlen(wgsl)};
    WGPUShaderSourceWGSL src = {.chain = {.next = NULL, .sType = WGPUSType_ShaderSourceWGSL}, .code = sv};
    return wgpuDeviceCreateShaderModule(dev, &(WGPUShaderModuleDescriptor){
        .nextInChain = &src.chain, .label = {.data = "triangle", .length = 8}
    });
}

// ── 主程序 ──────────────────────────────────────────────────────────────
int main(void) {
    const uint32_t W = 800, H = 600;

    // 1. GLFW
    glfwSetErrorCallback(glfw_error_cb);
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* win = glfwCreateWindow(W, H, "WGPU Minimal Triangle", NULL, NULL);
    if (!win) { glfwTerminate(); return 1; }

    // 2. WGPU instance
    WGPUInstance inst = wgpuCreateInstance(&(WGPUInstanceDescriptor){0});
    if (!inst) { glfwDestroyWindow(win); glfwTerminate(); return 1; }

    // 3. Surface
    WGPUSurface surface = make_surface(inst, win);
    if (!surface) { wgpuInstanceRelease(inst); glfwDestroyWindow(win); glfwTerminate(); return 1; }

    // 4. Adapter (同步)
    WGPUAdapter adapter = NULL;
    wgpuInstanceRequestAdapter(inst,
        &(WGPURequestAdapterOptions){.compatibleSurface = surface},
        (WGPURequestAdapterCallbackInfo){
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = adapter_cb, .userdata1 = &adapter
        });
    wgpuInstanceProcessEvents(inst);
    if (!adapter) { wgpuSurfaceRelease(surface); wgpuInstanceRelease(inst); glfwDestroyWindow(win); glfwTerminate(); return 1; }

    // 5. Device (同步)
    WGPUDevice device = NULL;
    wgpuAdapterRequestDevice(adapter, &(WGPUDeviceDescriptor){0},
        (WGPURequestDeviceCallbackInfo){
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = device_cb, .userdata1 = &device
        });
    wgpuInstanceProcessEvents(inst);
    if (!device) { wgpuAdapterRelease(adapter); wgpuSurfaceRelease(surface); wgpuInstanceRelease(inst); glfwDestroyWindow(win); glfwTerminate(); return 1; }
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    // 6. Surface 配置
    WGPUSurfaceCapabilities caps = {0};
    wgpuSurfaceGetCapabilities(surface, adapter, &caps);
    WGPUTextureFormat fmt = caps.formats[0];
    wgpuSurfaceCapabilitiesFreeMembers(caps);

    wgpuSurfaceConfigure(surface, &(WGPUSurfaceConfiguration){
        .device = device, .format = fmt, .usage = WGPUTextureUsage_RenderAttachment,
        .presentMode = WGPUPresentMode_Fifo, .alphaMode = WGPUCompositeAlphaMode_Auto,
        .width = W, .height = H
    });

    // 7. Shader
    WGPUShaderModule shader = make_shader(device, SHADER_WGSL);
    if (!shader) { fprintf(stderr, "shader failed\n"); return 1; }

    // 8. Pipeline
    WGPUColorTargetState ct = {.format = fmt, .writeMask = WGPUColorWriteMask_All};
    WGPUFragmentState fs = {.module = shader, .entryPoint = {.data = "fs_main", .length = 7}, .targetCount = 1, .targets = &ct};
    WGPURenderPipeline pipe = wgpuDeviceCreateRenderPipeline(device, &(WGPURenderPipelineDescriptor){
        .label = {.data = "triangle", .length = 8},
        .vertex = {.module = shader, .entryPoint = {.data = "vs_main", .length = 7}},
        .primitive = {.topology = WGPUPrimitiveTopology_TriangleList},
        .multisample = {.count = 1, .mask = 0xFFFFFFFF},
        .fragment = &fs,
    });
    if (!pipe) { fprintf(stderr, "pipeline failed\n"); return 1; }

    printf("=== Minimal wgpu triangle ===\n");
    printf("Check GPU memory now. Close window to exit.\n");

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        WGPUSurfaceTexture st;
        wgpuSurfaceGetCurrentTexture(surface, &st);

        WGPUTextureView view = wgpuTextureCreateView(st.texture, &(WGPUTextureViewDescriptor){
            .format = fmt, .dimension = WGPUTextureViewDimension_2D,
            .mipLevelCount = 1, .arrayLayerCount = 1, .aspect = WGPUTextureAspect_All,
        });

        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &(WGPUCommandEncoderDescriptor){
            .label = {.data = "encoder", .length = 7}
        });

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &(WGPURenderPassDescriptor){
            .colorAttachmentCount = 1,
            .colorAttachments = &(WGPURenderPassColorAttachment){
                .view = view, .loadOp = WGPULoadOp_Clear, .storeOp = WGPUStoreOp_Store,
                .clearValue = {0.0f, 0.0f, 0.1f, 1.0f},
            },
        });
        wgpuRenderPassEncoderSetPipeline(pass, pipe);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);

        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &(WGPUCommandBufferDescriptor){
            .label = {.data = "cmd", .length = 3}
        });
        wgpuCommandEncoderRelease(encoder);

        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuSurfacePresent(surface);

        wgpuTextureViewRelease(view);
        wgpuTextureRelease(st.texture);
    }

    printf("Done. Exiting...\n");
    fflush(stdout);

    // 10. 清理
    wgpuRenderPipelineRelease(pipe);
    wgpuShaderModuleRelease(shader);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuSurfaceRelease(surface);
    wgpuInstanceRelease(inst);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
