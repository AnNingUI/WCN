#ifndef FULLSTACK_APP_PRIVATE_H
#define FULLSTACK_APP_PRIVATE_H
#include "fullstack_app.h"
#include "fullstack_sync_private.h"

struct FS_AppWindow {
    FS_App* app;
    FS_AppWindowId id;
    FS_BackendWindow* backend_window;
    FS_AppWindowMetrics metrics;
    FS_InputState input;
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
    const FS_AppBackendFactory* factory;
    FS_BackendInstance* backend;
    FS_AppWindow** windows;
    uint32_t window_count, window_capacity;
    FS_AppWindowId next_window_id;
    bool pumped_this_frame;
};

uint64_t fs_app_current_thread_id(void);
FS_Result fs_app_backend_factory_acquire(const char* name, const FS_AppBackendFactory** out, FS_Error* error);
void fs_app_backend_factory_release(const FS_AppBackendFactory* factory);
#endif
