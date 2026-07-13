#include "fullstack_mock_app_backend.h"

#include <assert.h>

static void begin_tick(FS_App* app, FS_Error* error) {
    assert(fs_app_begin_frame(app, error) == FS_RESULT_OK);
    assert(fs_app_pump_events(app, FS_APP_PUMP_POLL, 0, error) == FS_RESULT_OK);
}

static void wait_surface_ready(FS_App* app, FS_AppWindow* window,
                               FS_Error* error) {
    (void)wgpuDevicePoll(fs_gpu_device(fs_app_gpu(app)), true, NULL);
    for (uint32_t i = 0; i < 100 &&
         fs_app_window_surface_state(window) != FS_APP_SURFACE_READY; ++i) {
        begin_tick(app, error);
        assert(fs_app_end_frame(app, error) == FS_RESULT_OK);
    }
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_READY);
}

int main(void) {
    FS_Error error = FS_ERROR_INIT;
    assert(fs_app_register_mock_backend(&error) == FS_RESULT_OK);
    FS_GpuContextDesc gpu_desc = FS_GPU_CONTEXT_DESC_INIT;
    FS_GpuContext* gpu = NULL;
    assert(fs_gpu_context_create(&gpu_desc, &gpu, &error) == FS_RESULT_OK);
    FS_AppDesc desc = FS_APP_DESC_INIT;
    desc.backend_name = "mock";
    desc.gpu = gpu;
    desc.create_default_window = true;
    FS_App* app = NULL;
    assert(fs_app_create(&desc, &app, &error) == FS_RESULT_OK);
    FS_AppWindow* window = fs_app_window_at(app, 0);
    assert(window && fs_app_gpu(app) == gpu);

    FS_AppFrame frame = FS_APP_FRAME_INIT;
    begin_tick(app, &error);
    assert(fs_app_window_acquire_frame(window, &frame, &error) == FS_RESULT_OK);
    assert(frame.state == FS_APP_FRAME_ACQUIRED);
    assert(frame.texture && frame.view);
    FS_AppFrame second = FS_APP_FRAME_INIT;
    assert(fs_app_window_acquire_frame(window, &second, &error) ==
           FS_RESULT_INVALID_STATE);
    assert(fs_app_frame_present(window, &frame, &error) ==
           FS_RESULT_INVALID_STATE);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_INVALID_STATE);
    assert(fs_app_frame_submit(window, &frame, 0, NULL, &error) == FS_RESULT_OK);
    assert(frame.state == FS_APP_FRAME_SUBMITTED);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_INVALID_STATE);
    assert(fs_app_frame_submit(window, &frame, 0, NULL, &error) ==
           FS_RESULT_INVALID_STATE);
    assert(fs_app_frame_present(window, &frame, &error) == FS_RESULT_OK);
    assert(frame.state == FS_APP_FRAME_PRESENTED);
    assert(fs_mock_window_present_count(window) == 1);
    assert(fs_app_frame_present(window, &frame, &error) ==
           FS_RESULT_INVALID_STATE);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_frame_reset(&frame) == FS_RESULT_OK);

    begin_tick(app, &error);
    assert(fs_app_window_acquire_frame(window, &frame, &error) == FS_RESULT_OK);
    assert(fs_app_frame_cancel(window, &frame, &error) == FS_RESULT_OK);
    assert(frame.state == FS_APP_FRAME_CANCELLED);
    assert(fs_mock_window_cancel_count(window) == 1);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_frame_reset(&frame) == FS_RESULT_OK);

    begin_tick(app, &error);
    assert(fs_app_window_acquire_frame(window, &frame, &error) == FS_RESULT_OK);
    assert(fs_app_frame_submit(window, &frame, 0, NULL, &error) == FS_RESULT_OK);
    assert(fs_app_frame_cancel(window, &frame, &error) == FS_RESULT_OK);
    assert(fs_mock_window_cancel_count(window) == 2);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_state(window) ==
           FS_APP_SURFACE_RECOVERY_PENDING);
    wait_surface_ready(app, window, &error);

    assert(fs_app_begin_destroy(app, &error) == FS_RESULT_OK);
    fs_app_release(app);
    fs_gpu_context_destroy(gpu);
    assert(fs_app_backend_unregister("mock", &error) == FS_RESULT_OK);
    return 0;
}
