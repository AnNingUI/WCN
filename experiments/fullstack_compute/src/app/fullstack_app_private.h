#ifndef FULLSTACK_APP_PRIVATE_H
#define FULLSTACK_APP_PRIVATE_H
#include "fullstack_app.h"
#include "fullstack_sync_private.h"

#define FS_SURFACE_PENDING_RECONFIGURE (1u << 0)
#define FS_SURFACE_PENDING_UNAVAILABLE (1u << 1)
#define FS_SURFACE_PENDING_AVAILABLE   (1u << 2)
#define FS_SURFACE_PENDING_DEVICE_LOST (1u << 3)

struct FS_AppWindow {
    FS_App* app;
    FS_AppWindowId id;
    FS_BackendWindow* backend_window;
    FS_AppWindowMetrics metrics;
    FS_InputState input;
    FS_AppSurfaceState surface_state;
    uint64_t surface_generation;
    FS_AppFrame* active_frame;
    uint64_t active_frame_cookie;
    FS_SubmissionToken retiring_submission;
    uint32_t pending_surface_flags;
};
struct FS_App {
    const FS_Allocator* allocator;
    FS_DiagnosticSink diagnostics;
    FS_AppState state;
    uint64_t app_thread_id;
    FS_AppThreadRoles thread_roles;
    FS_EventQueue* events;
    FS_AppTaskQueue* tasks;
    FS_Lifetime* lifetime;
    FS_AppBackendHost host;
    FS_GpuContext* gpu;
    const FS_AppBackendFactory* factory;
    FS_BackendInstance* backend;
    FS_AppWindow** windows;
    uint32_t window_count, window_capacity;
    FS_AppWindowId next_window_id;
    uint64_t next_frame_id;
    uint64_t next_frame_cookie;
    bool pumped_this_frame;
};

uint64_t fs_app_current_thread_id(void);
FS_Result fs_app_backend_factory_acquire(const char* name, const FS_AppBackendFactory** out, FS_Error* error);
void fs_app_backend_factory_release(const FS_AppBackendFactory* factory);
void fs_app_surface_note_event(FS_AppWindow* window, FS_AppEventType type);
void fs_app_surface_commit_pending(FS_AppWindow* window);
void fs_app_surface_wait_retirement(FS_AppWindow* window);
bool fs_app_frame_is_active(const FS_AppWindow* window);
void fs_app_frame_cancel_for_destroy(FS_AppWindow* window);
#endif
