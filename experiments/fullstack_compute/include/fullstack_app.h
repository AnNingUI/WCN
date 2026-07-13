#ifndef FULLSTACK_APP_H
#define FULLSTACK_APP_H
#include "fullstack_app_backend.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t FS_AppState;
#define FS_APP_STATE_READY        ((FS_AppState)1u)
#define FS_APP_STATE_FRAME_ACTIVE ((FS_AppState)2u)
#define FS_APP_STATE_CLOSING      ((FS_AppState)3u)
#define FS_APP_STATE_RELEASED     ((FS_AppState)4u)

typedef uint32_t FS_AppSurfaceState;
#define FS_APP_SURFACE_UNAVAILABLE      ((FS_AppSurfaceState)0u)
#define FS_APP_SURFACE_UNCONFIGURED     ((FS_AppSurfaceState)1u)
#define FS_APP_SURFACE_READY            ((FS_AppSurfaceState)2u)
#define FS_APP_SURFACE_FRAME_ACQUIRED   ((FS_AppSurfaceState)3u)
#define FS_APP_SURFACE_FRAME_SUBMITTED  ((FS_AppSurfaceState)4u)
#define FS_APP_SURFACE_RECOVERY_PENDING ((FS_AppSurfaceState)5u)
#define FS_APP_SURFACE_DEVICE_LOST      ((FS_AppSurfaceState)6u)

typedef struct FS_AppDesc {
    uint32_t struct_size;
    const FS_Allocator* allocator;
    FS_DiagnosticSink diagnostics;
    const char* backend_name;
    FS_GpuContext* gpu;
    FS_EventQueueDesc event_queue;
    bool create_default_window;
    FS_AppWindowDesc default_window;
} FS_AppDesc;
#define FS_APP_DESC_INIT { sizeof(FS_AppDesc), NULL, {0}, NULL, NULL, FS_EVENT_QUEUE_DESC_INIT, false, FS_APP_WINDOW_DESC_INIT }

FS_API FS_Result FS_CALL fs_app_create(const FS_AppDesc* desc, FS_App** out_app, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_begin_destroy(FS_App* app, FS_Error* error);
FS_API void FS_CALL fs_app_release(FS_App* app);
FS_API FS_Result FS_CALL fs_app_begin_frame(FS_App* app, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_pump_events(FS_App* app, FS_AppPumpMode mode, uint64_t timeout_ns, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_poll_event(FS_App* app, FS_AppEvent* out_event, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_end_frame(FS_App* app, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_create_window(FS_App* app, const FS_AppWindowDesc* desc, FS_AppWindow** out_window, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_destroy_window(FS_App* app, FS_AppWindow* window, FS_Error* error);
FS_API uint32_t FS_CALL fs_app_window_count(const FS_App* app);
FS_API FS_AppWindow* FS_CALL fs_app_window_at(FS_App* app, uint32_t index);
FS_API FS_AppWindow* FS_CALL fs_app_find_window(FS_App* app, FS_AppWindowId id);
FS_API FS_AppWindowId FS_CALL fs_app_window_id(const FS_AppWindow* window);
FS_API const FS_AppWindowMetrics* FS_CALL fs_app_window_metrics(const FS_AppWindow* window);
FS_API const FS_InputState* FS_CALL fs_app_window_input(const FS_AppWindow* window);
FS_API FS_AppState FS_CALL fs_app_state(const FS_App* app);
FS_API const FS_AppThreadRoles* FS_CALL fs_app_thread_roles(const FS_App* app);
FS_API bool FS_CALL fs_app_is_app_thread(const FS_App* app);
FS_API void FS_CALL fs_app_request_wake(FS_App* app);
FS_API FS_GpuContext* FS_CALL fs_app_gpu(FS_App* app);
FS_API FS_AppSurfaceState FS_CALL fs_app_window_surface_state(
    const FS_AppWindow* window);
FS_API uint64_t FS_CALL fs_app_window_surface_generation(
    const FS_AppWindow* window);
FS_API FS_Result FS_CALL fs_app_window_recover_surface(
    FS_AppWindow* window, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_window_acquire_frame(
    FS_AppWindow* window, FS_AppFrame* frame, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_frame_submit(
    FS_AppWindow* window, FS_AppFrame* frame, uint32_t command_count,
    const WGPUCommandBuffer* commands, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_frame_mark_submitted(
    FS_AppWindow* window, FS_AppFrame* frame,
    FS_SubmissionToken submission, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_frame_present(
    FS_AppWindow* window, FS_AppFrame* frame, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_frame_cancel(
    FS_AppWindow* window, FS_AppFrame* frame, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_frame_reset(FS_AppFrame* frame);

#ifdef __cplusplus
}
#endif
#endif
