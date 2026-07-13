#include "fullstack_glfw_app_backend.h"
#include "fullstack_app_services.h"
#include "backend_contract_suite.h"

#include <GLFW/glfw3.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

typedef struct DialogState {
    FS_App* app;
    bool complete;
} DialogState;

static void FS_CALL dialog_complete(FS_FileDialogResult* result, void* data) {
    DialogState* state = (DialogState*)data;
    assert(fs_app_is_app_thread(state->app));
    assert(result->result == FS_RESULT_OK);
    assert(result->path_count == 1);
    assert(strcmp((const char*)result->nul_separated_paths_utf8.data,
                  "C:/wcn-test.png") == 0);
    fs_file_dialog_result_release(result);
    state->complete = true;
}

static void render_one(FS_App* app, FS_AppWindow* window, FS_Error* error) {
    FS_AppFrame frame = FS_APP_FRAME_INIT;
    FS_Result acquired = fs_app_window_acquire_frame(window, &frame, error);
    if (acquired == FS_RESULT_SKIP) return;
    assert(acquired == FS_RESULT_OK);
    WGPURenderPassColorAttachment color = {
        .view = frame.view,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
        .loadOp = WGPULoadOp_Clear,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = {0.1, 0.2, 0.3, 1.0}};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        fs_gpu_device(fs_app_gpu(app)), &(WGPUCommandEncoderDescriptor){0});
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
        encoder, &(WGPURenderPassDescriptor){
            .colorAttachmentCount = 1, .colorAttachments = &color});
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    WGPUCommandBuffer command = wgpuCommandEncoderFinish(
        encoder, &(WGPUCommandBufferDescriptor){0});
    assert(fs_app_frame_submit(window, &frame, 1, &command, error) == FS_RESULT_OK);
    wgpuCommandBufferRelease(command);
    wgpuCommandEncoderRelease(encoder);
    assert(fs_app_frame_present(window, &frame, error) == FS_RESULT_OK);
}

int main(void) {
    FS_Error error = FS_ERROR_INIT;
    assert(fs_app_register_glfw_backend(&error) == FS_RESULT_OK);
    fs_backend_contract_basic("glfw", false);
    FS_AppDesc desc = FS_APP_DESC_INIT;
    desc.backend_name = "glfw";
    desc.create_default_window = true;
    desc.default_window.title = "WCN GLFW contract A";
    desc.default_window.width = 64;
    desc.default_window.height = 64;
    desc.default_window.visible = false;
    FS_App* app = NULL;
    assert(fs_app_create(&desc, &app, &error) == FS_RESULT_OK);
    assert(fs_app_backend_name(app) &&
           strcmp(fs_app_backend_name(app), "glfw") == 0);
    FS_AppWindow* first = fs_app_window_at(app, 0);
    FS_AppWindowDesc second_desc = FS_APP_WINDOW_DESC_INIT;
    second_desc.title = "WCN GLFW contract B";
    second_desc.width = 80;
    second_desc.height = 48;
    second_desc.visible = false;
    FS_AppWindow* second = NULL;
    assert(fs_app_create_window(app, &second_desc, &second, &error) == FS_RESULT_OK);
    assert(fs_app_window_count(app) == 2);
    assert(fs_app_clipboard_set_text(app, first, "WCN clipboard", &error) ==
           FS_RESULT_OK);
    FS_AppOwnedBytes clipboard = {0};
    assert(fs_app_clipboard_get_text(app, first, &clipboard, &error) ==
           FS_RESULT_OK);
    assert(strcmp((const char*)clipboard.data, "WCN clipboard") == 0);
    fs_app_owned_bytes_release(&clipboard);
    assert(fs_app_cursor_set_mode(app, first, FS_CURSOR_MODE_NORMAL, &error) ==
           FS_RESULT_OK);
    assert(fs_app_cursor_set_shape(app, first, FS_CURSOR_SHAPE_HAND, &error) ==
           FS_RESULT_OK);
    FS_NativeWindowHandle handle = {0};
    assert(fs_app_window_native_handle(app, first, &handle, &error) ==
           FS_RESULT_OK);
    assert(handle.kind != FS_NATIVE_HANDLE_NONE && handle.value != 0);

#if defined(_WIN32)
    _putenv_s("FS_GLFW_FILE_DIALOG_TEST_RESULT", "C:/wcn-test.png");
#else
    setenv("FS_GLFW_FILE_DIALOG_TEST_RESULT", "C:/wcn-test.png", 1);
#endif
    DialogState dialog = {app, false};
    FS_FileDialogDesc dialog_desc = FS_FILE_DIALOG_DESC_INIT;
    dialog_desc.parent_window = first;
    dialog_desc.callback = dialog_complete;
    dialog_desc.user_data = &dialog;
    FS_AppServiceRequestId request = 0;
    assert(fs_app_file_dialog_request(app, &dialog_desc, &request, &error) ==
           FS_RESULT_OK);
    for (uint32_t i = 0; i < 100 && !dialog.complete; ++i) {
        assert(fs_app_begin_frame(app, &error) == FS_RESULT_OK);
        assert(fs_app_pump_events(app, FS_APP_PUMP_WAIT_TIMEOUT,
                                  1000000, &error) == FS_RESULT_OK);
        assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    }
    assert(dialog.complete);
#if defined(_WIN32)
    _putenv_s("FS_GLFW_FILE_DIALOG_TEST_RESULT", "");
#else
    unsetenv("FS_GLFW_FILE_DIALOG_TEST_RESULT");
#endif

    assert(fs_app_begin_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_pump_events(app, FS_APP_PUMP_POLL, 0, &error) == FS_RESULT_OK);
    render_one(app, first, &error);
    render_one(app, second, &error);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);

    GLFWwindow* native = fs_glfw_backend_native_window(
        fs_app_window_backend_handle(first));
    assert(native);
    glfwShowWindow(native);
    glfwSetWindowSize(native, 96, 72);
    glfwPollEvents();
    glfwHideWindow(native);
    assert(fs_app_begin_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_pump_events(app, FS_APP_PUMP_POLL, 0, &error) == FS_RESULT_OK);
    FS_AppEvent event = FS_APP_EVENT_INIT;
    while (fs_app_poll_event(app, &event, &error) == FS_RESULT_OK)
        fs_app_event_release(&event);
    assert(fs_app_refresh_window_metrics(first, &error) == FS_RESULT_OK);
    render_one(app, first, &error);
    render_one(app, second, &error);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    int actual_width = 0, actual_height = 0;
    glfwGetWindowSize(native, &actual_width, &actual_height);
    assert(actual_width > 0 && actual_height > 0);
    assert(fs_app_window_metrics(first)->logical_width ==
           (uint32_t)actual_width);
    assert(fs_app_window_metrics(first)->logical_height ==
           (uint32_t)actual_height);

    assert(fs_app_begin_destroy(app, &error) == FS_RESULT_OK);
    fs_app_release(app);
    assert(fs_app_backend_unregister("glfw", &error) == FS_RESULT_OK);
    return 0;
}
