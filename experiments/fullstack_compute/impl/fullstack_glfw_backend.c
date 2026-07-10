#include "fullstack_glfw_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void fs_glfw_error_callback(int error, const char* description) {
    printf("GLFW error %d: %s\n", error, description ? description : "(null)");
}

static void fs_glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (!window) {
        return;
    }
    FS_GlfwBackend* backend = (FS_GlfwBackend*)glfwGetWindowUserPointer(window);
    if (!backend) {
        return;
    }
    backend->scroll_delta_x += (float)xoffset;
    backend->scroll_delta_y += (float)yoffset;
}

static void fs_adapter_callback(
    WGPURequestAdapterStatus status,
    WGPUAdapter adapter,
    WGPUStringView message,
    void* userdata1,
    void* userdata2
) {
    (void)userdata2;
    if (status == WGPURequestAdapterStatus_Success) {
        *(WGPUAdapter*)userdata1 = adapter;
    } else {
        printf("RequestAdapter failed: %.*s\n", (int)message.length, message.data);
    }
}

static void fs_device_callback(
    WGPURequestDeviceStatus status,
    WGPUDevice device,
    WGPUStringView message,
    void* userdata1,
    void* userdata2
) {
    (void)userdata2;
    if (status == WGPURequestDeviceStatus_Success) {
        *(WGPUDevice*)userdata1 = device;
    } else {
        printf("RequestDevice failed: %.*s\n", (int)message.length, message.data);
    }
}

static void fs_uncaptured_error_callback(
    WGPUDevice const* device,
    WGPUErrorType type,
    WGPUStringView message,
    void* userdata1,
    void* userdata2
) {
    (void)device;
    (void)userdata1;
    (void)userdata2;
    printf("WebGPU uncaptured error type=%d: %.*s\n", (int)type, (int)message.length, message.data);
}

static void fs_device_lost_callback(
    WGPUDevice const* device,
    WGPUDeviceLostReason reason,
    WGPUStringView message,
    void* userdata1,
    void* userdata2
) {
    (void)device;
    (void)userdata1;
    (void)userdata2;
    printf("WebGPU device lost reason=%d: %.*s\n", (int)reason, (int)message.length, message.data);
}

static const char* fs_surface_status_name(WGPUSurfaceGetCurrentTextureStatus status) {
    switch (status) {
        case WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal: return "SuccessOptimal";
        case WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal: return "SuccessSuboptimal";
        case WGPUSurfaceGetCurrentTextureStatus_Timeout: return "Timeout";
        case WGPUSurfaceGetCurrentTextureStatus_Outdated: return "Outdated";
        case WGPUSurfaceGetCurrentTextureStatus_Lost: return "Lost";
        case WGPUSurfaceGetCurrentTextureStatus_Error: return "Error";
        case WGPUSurfaceGetCurrentTextureStatus_Occluded: return "Occluded";
        default: return "Unknown";
    }
}

static WGPUSurface fs_create_surface(WGPUInstance instance, GLFWwindow* window) {
#if defined(_WIN32)
    WGPUSurfaceSourceWindowsHWND surface_source = {
        .chain = {.sType = WGPUSType_SurfaceSourceWindowsHWND},
        .hinstance = GetModuleHandle(NULL),
        .hwnd = glfwGetWin32Window(window)
    };
    WGPUSurfaceDescriptor desc = {
        .nextInChain = &surface_source.chain
    };
    return wgpuInstanceCreateSurface(instance, &desc);
#elif defined(__linux__)
    WGPUSurfaceSourceXlibWindow surface_source = {
        .chain = {.sType = WGPUSType_SurfaceSourceXlibWindow},
        .display = glfwGetX11Display(),
        .window = glfwGetX11Window(window)
    };
    WGPUSurfaceDescriptor desc = {
        .nextInChain = &surface_source.chain
    };
    return wgpuInstanceCreateSurface(instance, &desc);
#elif defined(__APPLE__)
    WGPUSurfaceSourceMetalLayer surface_source = {
        .chain = {.sType = WGPUSType_SurfaceSourceMetalLayer},
        .layer = glfwGetCocoaWindow(window)
    };
    WGPUSurfaceDescriptor desc = {
        .nextInChain = &surface_source.chain
    };
    return wgpuInstanceCreateSurface(instance, &desc);
#else
    (void)instance;
    (void)window;
    return NULL;
#endif
}

static bool fs_reconfigure_surface(FS_GlfwBackend* backend, uint32_t width, uint32_t height) {
    if (!backend || !backend->surface || !backend->device || width == 0 || height == 0) {
        return false;
    }
    backend->width = width;
    backend->height = height;
    backend->present_mode = WGPUPresentMode_Fifo;
    WGPUTextureUsage usage = WGPUTextureUsage_RenderAttachment;
    backend->supports_surface_copy_src = false;
    if (backend->adapter) {
        WGPUSurfaceCapabilities caps = {0};
        wgpuSurfaceGetCapabilities(backend->surface, backend->adapter, &caps);
        if ((caps.usages & WGPUTextureUsage_CopySrc) != 0u) {
            usage |= WGPUTextureUsage_CopySrc;
            backend->supports_surface_copy_src = true;
        }
        wgpuSurfaceCapabilitiesFreeMembers(caps);
    }

    WGPUSurfaceConfiguration config = {
        .device = backend->device,
        .format = backend->surface_format,
        .usage = usage,
        .presentMode = backend->present_mode,
        .alphaMode = WGPUCompositeAlphaMode_Auto,
        .width = width,
        .height = height
    };
    wgpuSurfaceConfigure(backend->surface, &config);
    if (backend->core) {
        fs_core_resize(backend->core, width, height);
    }
    return true;
}

bool fs_glfw_backend_init(FS_GlfwBackend* backend, uint32_t width, uint32_t height, const char* title) {
    if (!backend) {
        return false;
    }
    memset(backend, 0, sizeof(*backend));

    glfwSetErrorCallback(fs_glfw_error_callback);
    if (!glfwInit()) {
        return false;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    backend->window = glfwCreateWindow((int)width, (int)height, title ? title : "Fullstack Compute", NULL, NULL);
    if (!backend->window) {
        glfwTerminate();
        return false;
    }
    glfwSetWindowUserPointer(backend->window, backend);
    glfwSetScrollCallback(backend->window, fs_glfw_scroll_callback);

    backend->width = width;
    backend->height = height;

    WGPUInstanceDescriptor instance_desc = {0};
    backend->instance = wgpuCreateInstance(&instance_desc);
    if (!backend->instance) {
        fs_glfw_backend_shutdown(backend);
        return false;
    }
    backend->surface = fs_create_surface(backend->instance, backend->window);
    if (!backend->surface) {
        fs_glfw_backend_shutdown(backend);
        return false;
    }

    wgpuInstanceRequestAdapter(
        backend->instance,
        &(WGPURequestAdapterOptions){.compatibleSurface = backend->surface},
        (WGPURequestAdapterCallbackInfo){
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = fs_adapter_callback,
            .userdata1 = &backend->adapter
        }
    );
    if (!backend->adapter) {
        fs_glfw_backend_shutdown(backend);
        return false;
    }

    WGPUDeviceDescriptor device_desc = {
        .requiredFeatureCount = 0,
        .requiredFeatures = NULL,
        .deviceLostCallbackInfo = {
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = fs_device_lost_callback
        },
        .uncapturedErrorCallbackInfo = {
            .callback = fs_uncaptured_error_callback
        }
    };
    wgpuAdapterRequestDevice(
        backend->adapter,
        &device_desc,
        (WGPURequestDeviceCallbackInfo){
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = fs_device_callback,
            .userdata1 = &backend->device
        }
    );
    if (!backend->device) {
        fs_glfw_backend_shutdown(backend);
        return false;
    }
    backend->queue = wgpuDeviceGetQueue(backend->device);
    if (!backend->queue) {
        fs_glfw_backend_shutdown(backend);
        return false;
    }

    WGPUSurfaceCapabilities caps = {0};
    wgpuSurfaceGetCapabilities(backend->surface, backend->adapter, &caps);
    if (caps.formatCount == 0 || !caps.formats) {
        wgpuSurfaceCapabilitiesFreeMembers(caps);
        fs_glfw_backend_shutdown(backend);
        return false;
    }
    backend->surface_format = caps.formats[0];
    /*
     * Temporary color-space compatibility patch.
     *
     * Why this exists:
     * - The current WCN path renders the scene into RGBA8Unorm intermediate textures
     *   and then presents via a fullscreen copy pass.
     * - When the swapchain/surface is configured as *UnormSrgb, the final output becomes
     *   visibly brighter than the intended UI colors.
     * - A minimal validation confirmed that forcing the final surface format to the non-sRGB
     *   variant restores the expected colors.
     *
     * So for now we intentionally downgrade sRGB surface formats to non-sRGB here to keep
     * the current renderer visually correct.
     *
     * TODO(full color-space unification): remove this patch after the whole pipeline is made
     * color-space consistent end-to-end. That future work should include at least:
     * 1. Decide and document whether scene_texture / ping-pong textures store linear or sRGB data.
     * 2. Make render shaders, filter passes, and presentation pass agree on that contract.
     * 3. Audit premultiplied-alpha behavior together with the chosen color space.
     * 4. Revisit presentation shader/pipeline so swapchain *UnormSrgb targets are handled
     *    intentionally instead of by implicit passthrough.
     */
    if (backend->surface_format == WGPUTextureFormat_BGRA8UnormSrgb) {
        backend->surface_format = WGPUTextureFormat_BGRA8Unorm;
    } else if (backend->surface_format == WGPUTextureFormat_RGBA8UnormSrgb) {
        backend->surface_format = WGPUTextureFormat_RGBA8Unorm;
    }
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    if (!fs_reconfigure_surface(backend, width, height)) {
        fs_glfw_backend_shutdown(backend);
        return false;
    }

    backend->core = fs_core_create(
        backend->device,
        backend->queue,
        backend->surface_format,
        width,
        height
    );
    if (!backend->core) {
        fs_glfw_backend_shutdown(backend);
        return false;
    }
    // Create presentation pipeline with canvas surface format (BGRA8UnormSrgb)
    if (!fs_effects_create_presentation_pipeline(backend->core, backend->surface_format)) {
        fs_glfw_backend_shutdown(backend);
        return false;
    }
    return true;
}

void fs_glfw_backend_shutdown(FS_GlfwBackend* backend) {
    if (!backend) {
        return;
    }

    if (backend->core) {
        fs_core_destroy(backend->core);
        backend->core = NULL;
    }

    if (backend->queue) {
        wgpuQueueRelease(backend->queue);
        backend->queue = NULL;
    }
    if (backend->device) {
        wgpuDeviceRelease(backend->device);
        backend->device = NULL;
    }
    if (backend->adapter) {
        wgpuAdapterRelease(backend->adapter);
        backend->adapter = NULL;
    }
    if (backend->surface) {
        wgpuSurfaceRelease(backend->surface);
        backend->surface = NULL;
    }
    if (backend->instance) {
        wgpuInstanceRelease(backend->instance);
        backend->instance = NULL;
    }
    if (backend->window) {
        glfwDestroyWindow(backend->window);
        backend->window = NULL;
    }
    glfwTerminate();
}

bool fs_glfw_backend_should_close(FS_GlfwBackend* backend) {
    if (!backend || !backend->window) {
        return true;
    }
    return glfwWindowShouldClose(backend->window);
}

void fs_glfw_backend_poll_events(void) {
    glfwPollEvents();
}

void fs_glfw_backend_take_scroll_delta(FS_GlfwBackend* backend, float* out_x, float* out_y) {
    if (out_x) {
        *out_x = 0.0f;
    }
    if (out_y) {
        *out_y = 0.0f;
    }
    if (!backend) {
        return;
    }
    if (out_x) {
        *out_x = backend->scroll_delta_x;
    }
    if (out_y) {
        *out_y = backend->scroll_delta_y;
    }
    backend->scroll_delta_x = 0.0f;
    backend->scroll_delta_y = 0.0f;
}

FS_Core* fs_glfw_backend_core(FS_GlfwBackend* backend) {
    if (!backend) {
        return NULL;
    }
    return backend->core;
}

bool fs_glfw_backend_present(FS_GlfwBackend* backend, float clear_r, float clear_g, float clear_b, float clear_a) {
    if (!backend || !backend->window || !backend->device || !backend->surface || !backend->core) {
        return false;
    }

    int fb_w = 0;
    int fb_h = 0;
    glfwGetFramebufferSize(backend->window, &fb_w, &fb_h);
    if (fb_w <= 0 || fb_h <= 0) {
        // Minimized/hidden: no renderable surface this frame.
        return true;
    }
    if (fb_w > 0 && fb_h > 0 &&
        ((uint32_t)fb_w != backend->width || (uint32_t)fb_h != backend->height)) {
        if (!fs_reconfigure_surface(backend, (uint32_t)fb_w, (uint32_t)fb_h)) {
            fprintf(stderr, "Surface reconfigure failed (resize to %dx%d)\n", fb_w, fb_h);
            return false;
        }
        fs_effects_create_presentation_pipeline(backend->core, backend->surface_format);
    }

    WGPUSurfaceTexture surface_texture = {0};
    wgpuSurfaceGetCurrentTexture(backend->surface, &surface_texture);
    if (surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        if (surface_texture.texture) {
            wgpuTextureRelease(surface_texture.texture);
            surface_texture.texture = NULL;
        }
        switch (surface_texture.status) {
            case WGPUSurfaceGetCurrentTextureStatus_Timeout:
            case WGPUSurfaceGetCurrentTextureStatus_Occluded:
                // Transient surface timeout; skip this frame and keep running.
                return true;
            case WGPUSurfaceGetCurrentTextureStatus_Outdated:
            case WGPUSurfaceGetCurrentTextureStatus_Lost:
                if (!fs_reconfigure_surface(backend, (uint32_t)fb_w, (uint32_t)fb_h)) {
                    fprintf(
                        stderr,
                        "Surface reconfigure failed after status=%s\n",
                        fs_surface_status_name(surface_texture.status)
                    );
                    return false;
                }
                fs_effects_create_presentation_pipeline(backend->core, backend->surface_format);
                wgpuSurfaceGetCurrentTexture(backend->surface, &surface_texture);
                if (surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
                    surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
                    if (surface_texture.texture) {
                        wgpuTextureRelease(surface_texture.texture);
                        surface_texture.texture = NULL;
                    }
                    fprintf(
                        stderr,
                        "Surface acquire failed after reconfigure, status=%s\n",
                        fs_surface_status_name(surface_texture.status)
                    );
                    return false;
                }
                break;
            default:
                fprintf(
                    stderr,
                    "Surface acquire failed, status=%s\n",
                    fs_surface_status_name(surface_texture.status)
                );
                return false;
        }
    }

    WGPUTextureViewDescriptor view_desc = {
        .nextInChain = NULL,
        .label = { .data = "FS Swapchain View", .length = 17 },
        .format = WGPUTextureFormat_Undefined,
        .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = 0,
        .mipLevelCount = 1,
        .baseArrayLayer = 0,
        .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All
    };
    WGPUTextureView view = wgpuTextureCreateView(surface_texture.texture, &view_desc);
    if (!view) {
        wgpuTextureRelease(surface_texture.texture);
        return false;
    }

    WGPUCommandEncoderDescriptor encoder_desc = {
        .nextInChain = NULL,
        .label = { .data = "FS Frame Encoder", .length = 16 }
    };
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(backend->device, &encoder_desc);
    if (!encoder) {
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(surface_texture.texture);
        return false;
    }

    bool ok = fs_core_encode(
        backend->core,
        encoder,
        backend->supports_surface_copy_src ? surface_texture.texture : NULL,
        view,
        clear_r,
        clear_g,
        clear_b,
        clear_a
    );
    if (!ok) {
        fprintf(
            stderr,
            "fs_core_encode failed (context_lost=%d)\n",
            fs_core_is_context_lost(backend->core) ? 1 : 0
        );
    }
    WGPUCommandBuffer command_buffer = NULL;
    if (ok) {
        WGPUCommandBufferDescriptor cb_desc = {
            .nextInChain = NULL,
            .label = { .data = "FS Frame Command Buffer", .length = 23 }
        };
        command_buffer = wgpuCommandEncoderFinish(encoder, &cb_desc);
        if (!command_buffer) {
            fprintf(stderr, "wgpuCommandEncoderFinish failed\n");
            ok = false;
        }
    }

    if (command_buffer) {
        WGPUSubmissionIndex submission = wgpuQueueSubmitForIndex(backend->queue, 1, &command_buffer);
        // Process deferred GPU object destruction to prevent unbounded memory growth.
        // Without this poll, wgpu-native internally queues released resources (buffers,
        // textures, bind groups) for deferred cleanup and never reclaims them.
        wgpuDevicePoll(backend->device, false, NULL);
        fs_core_notify_submission(backend->core, submission);
        wgpuCommandBufferRelease(command_buffer);
        wgpuSurfacePresent(backend->surface);
    }

    wgpuCommandEncoderRelease(encoder);
    wgpuTextureViewRelease(view);
    wgpuTextureRelease(surface_texture.texture);
    return ok;
}
