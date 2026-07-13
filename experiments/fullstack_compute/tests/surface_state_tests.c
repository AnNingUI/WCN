#include "fullstack_mock_app_backend.h"

#include <assert.h>

static void begin_tick(FS_App* app, FS_Error* error) {
    assert(fs_app_begin_frame(app, error) == FS_RESULT_OK);
    assert(fs_app_pump_events(app, FS_APP_PUMP_POLL, 0, error) == FS_RESULT_OK);
}

static void poll_one(FS_App* app, FS_AppEventType type, FS_Error* error) {
    FS_AppEvent event = FS_APP_EVENT_INIT;
    assert(fs_app_poll_event(app, &event, error) == FS_RESULT_OK);
    assert(event.type == type);
    fs_app_event_release(&event);
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
    desc.default_window.width = 320;
    desc.default_window.height = 240;
    FS_App* app = NULL;
    assert(fs_app_create(&desc, &app, &error) == FS_RESULT_OK);
    FS_AppWindow* window = fs_app_window_at(app, 0);
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_READY);
    uint64_t generation = fs_app_window_surface_generation(window);

    FS_AppWindowMetrics metrics = *fs_app_window_metrics(window);
    metrics.framebuffer_width = 640;
    assert(fs_mock_window_set_metrics(window, &metrics, &error) == FS_RESULT_OK);
    begin_tick(app, &error);
    poll_one(app, FS_APP_EVENT_FRAMEBUFFER_RESIZED, &error);
    assert(fs_app_window_surface_state(window) ==
           FS_APP_SURFACE_RECOVERY_PENDING);
    FS_AppFrame skipped = FS_APP_FRAME_INIT;
    assert(fs_app_window_acquire_frame(window, &skipped, &error) ==
           FS_RESULT_SKIP);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_generation(window) == generation + 1);
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_READY);
    generation++;

    begin_tick(app, &error);
    FS_AppFrame active = FS_APP_FRAME_INIT;
    assert(fs_app_window_acquire_frame(window, &active, &error) == FS_RESULT_OK);
    metrics.framebuffer_width = 800;
    assert(fs_mock_window_set_metrics(window, &metrics, &error) == FS_RESULT_OK);
    poll_one(app, FS_APP_EVENT_FRAMEBUFFER_RESIZED, &error);
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_FRAME_ACQUIRED);
    assert(active.surface_generation == generation);
    assert(fs_app_frame_cancel(window, &active, &error) == FS_RESULT_OK);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_generation(window) == generation + 1);
    generation++;

    FS_AppEvent suspend = FS_APP_EVENT_INIT;
    suspend.type = FS_APP_EVENT_SUSPEND;
    suspend.window_id = fs_app_window_id(window);
    assert(fs_mock_backend_inject_event(app, &suspend, &error) == FS_RESULT_OK);
    begin_tick(app, &error);
    poll_one(app, FS_APP_EVENT_SUSPEND, &error);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_UNAVAILABLE);
    assert(fs_app_window_surface_generation(window) == generation + 1);
    generation++;

    FS_AppEvent resume = FS_APP_EVENT_INIT;
    resume.type = FS_APP_EVENT_RESUME;
    resume.window_id = fs_app_window_id(window);
    assert(fs_mock_backend_inject_event(app, &resume, &error) == FS_RESULT_OK);
    begin_tick(app, &error);
    poll_one(app, FS_APP_EVENT_RESUME, &error);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_READY);
    assert(fs_app_window_surface_generation(window) == generation + 1);
    generation++;

    assert(fs_mock_window_set_acquire_status(
        window, FS_APP_FRAME_ACQUIRE_OUTDATED) == FS_RESULT_OK);
    begin_tick(app, &error);
    assert(fs_app_frame_reset(&skipped) == FS_RESULT_OK);
    assert(fs_app_window_acquire_frame(window, &skipped, &error) == FS_RESULT_SKIP);
    assert(skipped.acquire_status == FS_APP_FRAME_ACQUIRE_OUTDATED);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_generation(window) == generation + 1);
    assert(fs_mock_window_set_acquire_status(
        window, FS_APP_FRAME_ACQUIRE_AVAILABLE) == FS_RESULT_OK);

    const FS_AppFrameAcquireStatus transient_statuses[] = {
        FS_APP_FRAME_ACQUIRE_TIMEOUT,
        FS_APP_FRAME_ACQUIRE_OCCLUDED,
        FS_APP_FRAME_ACQUIRE_SUSPENDED};
    for (uint32_t i = 0; i < 3; ++i) {
        begin_tick(app, &error);
        assert(fs_app_frame_reset(&skipped) == FS_RESULT_OK);
        assert(fs_mock_window_set_acquire_status(
            window, transient_statuses[i]) == FS_RESULT_OK);
        assert(fs_app_window_acquire_frame(window, &skipped, &error) ==
               FS_RESULT_SKIP);
        assert(skipped.acquire_status == transient_statuses[i]);
        assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    }
    generation = fs_app_window_surface_generation(window);
    assert(fs_mock_window_set_acquire_status(
        window, FS_APP_FRAME_ACQUIRE_SURFACE_LOST) == FS_RESULT_OK);
    begin_tick(app, &error);
    assert(fs_app_frame_reset(&skipped) == FS_RESULT_OK);
    assert(fs_app_window_acquire_frame(window, &skipped, &error) == FS_RESULT_SKIP);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_generation(window) == generation + 1);
    assert(fs_mock_window_set_acquire_status(
        window, FS_APP_FRAME_ACQUIRE_AVAILABLE) == FS_RESULT_OK);

    metrics.framebuffer_width = 0;
    metrics.framebuffer_height = 0;
    assert(fs_mock_window_set_metrics(window, &metrics, &error) == FS_RESULT_OK);
    begin_tick(app, &error);
    poll_one(app, FS_APP_EVENT_FRAMEBUFFER_RESIZED, &error);
    assert(fs_app_frame_reset(&skipped) == FS_RESULT_OK);
    assert(fs_app_window_acquire_frame(window, &skipped, &error) == FS_RESULT_SKIP);
    assert(skipped.acquire_status == FS_APP_FRAME_ACQUIRE_ZERO_SIZE);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_UNCONFIGURED);
    metrics.framebuffer_width = 800;
    metrics.framebuffer_height = 240;
    assert(fs_mock_window_set_metrics(window, &metrics, &error) == FS_RESULT_OK);
    begin_tick(app, &error);
    poll_one(app, FS_APP_EVENT_FRAMEBUFFER_RESIZED, &error);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_READY);

    generation = fs_app_window_surface_generation(window);
    assert(fs_mock_window_set_acquire_status(
        window, FS_APP_FRAME_ACQUIRE_SUBOPTIMAL) == FS_RESULT_OK);
    begin_tick(app, &error);
    assert(fs_app_frame_reset(&skipped) == FS_RESULT_OK);
    assert(fs_app_window_acquire_frame(window, &skipped, &error) == FS_RESULT_OK);
    assert(skipped.acquire_status == FS_APP_FRAME_ACQUIRE_SUBOPTIMAL);
    assert(fs_app_frame_submit(window, &skipped, 0, NULL, &error) == FS_RESULT_OK);
    assert(fs_app_frame_present(window, &skipped, &error) == FS_RESULT_OK);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_generation(window) == generation + 1);
    assert(fs_mock_window_set_acquire_status(
        window, FS_APP_FRAME_ACQUIRE_AVAILABLE) == FS_RESULT_OK);

    generation = fs_app_window_surface_generation(window);
    assert(fs_mock_window_set_present_result(
        window, FS_RESULT_INTERNAL_ERROR) == FS_RESULT_OK);
    begin_tick(app, &error);
    assert(fs_app_frame_reset(&skipped) == FS_RESULT_OK);
    assert(fs_app_window_acquire_frame(window, &skipped, &error) == FS_RESULT_OK);
    assert(fs_app_frame_submit(window, &skipped, 0, NULL, &error) == FS_RESULT_OK);
    assert(fs_app_frame_present(window, &skipped, &error) ==
           FS_RESULT_INTERNAL_ERROR);
    assert(skipped.state == FS_APP_FRAME_CANCELLED);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_state(window) ==
           FS_APP_SURFACE_RECOVERY_PENDING);
    (void)wgpuDevicePoll(fs_gpu_device(fs_app_gpu(app)), true, NULL);
    for (uint32_t i = 0; i < 100 &&
         fs_app_window_surface_generation(window) == generation; ++i) {
        begin_tick(app, &error);
        assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    }
    assert(fs_app_window_surface_generation(window) == generation + 1);
    assert(fs_mock_window_set_present_result(window, FS_RESULT_OK) == FS_RESULT_OK);

    FS_AppEvent lost = FS_APP_EVENT_INIT;
    lost.type = FS_APP_EVENT_DEVICE_LOST;
    lost.window_id = fs_app_window_id(window);
    assert(fs_mock_backend_inject_event(app, &lost, &error) == FS_RESULT_OK);
    begin_tick(app, &error);
    poll_one(app, FS_APP_EVENT_DEVICE_LOST, &error);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_DEVICE_LOST);
    assert(fs_mock_backend_inject_event(app, &resume, &error) == FS_RESULT_OK);
    begin_tick(app, &error);
    poll_one(app, FS_APP_EVENT_RESUME, &error);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_DEVICE_LOST);
    assert(fs_app_window_recover_surface(window, &error) == FS_RESULT_OK);
    begin_tick(app, &error);
    assert(fs_app_end_frame(app, &error) == FS_RESULT_OK);
    assert(fs_app_window_surface_state(window) == FS_APP_SURFACE_READY);

    assert(fs_app_begin_destroy(app, &error) == FS_RESULT_OK);
    fs_app_release(app);
    fs_gpu_context_destroy(gpu);
    assert(fs_app_backend_unregister("mock", &error) == FS_RESULT_OK);
    return 0;
}
