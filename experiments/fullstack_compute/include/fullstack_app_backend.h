#ifndef FULLSTACK_APP_BACKEND_H
#define FULLSTACK_APP_BACKEND_H

#include "fullstack_app_event.h"
#include "fullstack_app_runtime.h"
#include "fullstack_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FS_APP_BACKEND_ABI_VERSION FS_ABI_VERSION(1, 1)

typedef struct FS_App FS_App;
typedef struct FS_AppWindow FS_AppWindow;
typedef struct FS_AppFrame FS_AppFrame;
typedef struct FS_AppBackendHost FS_AppBackendHost;
typedef struct FS_BackendInstance FS_BackendInstance;
typedef struct FS_BackendWindow FS_BackendWindow;

typedef uint32_t FS_AppPumpMode;
#define FS_APP_PUMP_POLL         ((FS_AppPumpMode)0u)
#define FS_APP_PUMP_WAIT         ((FS_AppPumpMode)1u)
#define FS_APP_PUMP_WAIT_TIMEOUT ((FS_AppPumpMode)2u)

typedef struct FS_AppThreadRoles {
    uint32_t struct_size;
    uint64_t app_thread;
    uint64_t backend_thread;
    uint64_t event_consumer_thread;
    uint64_t platform_ui_thread;
} FS_AppThreadRoles;
typedef struct FS_AppWindowMetrics {
    uint32_t struct_size;
    uint32_t logical_width, logical_height;
    uint32_t framebuffer_width, framebuffer_height;
    float scale_x, scale_y;
    float safe_left, safe_top, safe_right, safe_bottom;
    uint32_t orientation;
} FS_AppWindowMetrics;

typedef uint32_t FS_AppFrameState;
#define FS_APP_FRAME_EMPTY     ((FS_AppFrameState)0u)
#define FS_APP_FRAME_ACQUIRED  ((FS_AppFrameState)1u)
#define FS_APP_FRAME_SUBMITTED ((FS_AppFrameState)2u)
#define FS_APP_FRAME_PRESENTED ((FS_AppFrameState)3u)
#define FS_APP_FRAME_CANCELLED ((FS_AppFrameState)4u)

typedef uint32_t FS_AppFrameAcquireStatus;
#define FS_APP_FRAME_ACQUIRE_AVAILABLE    ((FS_AppFrameAcquireStatus)0u)
#define FS_APP_FRAME_ACQUIRE_ZERO_SIZE    ((FS_AppFrameAcquireStatus)1u)
#define FS_APP_FRAME_ACQUIRE_TIMEOUT      ((FS_AppFrameAcquireStatus)2u)
#define FS_APP_FRAME_ACQUIRE_OCCLUDED     ((FS_AppFrameAcquireStatus)3u)
#define FS_APP_FRAME_ACQUIRE_SUSPENDED    ((FS_AppFrameAcquireStatus)4u)
#define FS_APP_FRAME_ACQUIRE_OUTDATED     ((FS_AppFrameAcquireStatus)5u)
#define FS_APP_FRAME_ACQUIRE_SURFACE_LOST ((FS_AppFrameAcquireStatus)6u)
#define FS_APP_FRAME_ACQUIRE_DEVICE_LOST  ((FS_AppFrameAcquireStatus)7u)
#define FS_APP_FRAME_ACQUIRE_FATAL        ((FS_AppFrameAcquireStatus)8u)
#define FS_APP_FRAME_ACQUIRE_SUBOPTIMAL   ((FS_AppFrameAcquireStatus)9u)

struct FS_AppFrame {
    uint32_t struct_size;
    FS_AppFrameState state;
    FS_AppFrameAcquireStatus acquire_status;
    uint32_t reserved;
    uint64_t frame_id;
    FS_AppWindowId window_id;
    uint64_t surface_generation;
    uint32_t width;
    uint32_t height;
    WGPUTextureFormat format;
    WGPUTexture texture;
    WGPUTextureView view;
    FS_SubmissionToken submission;
    uint64_t backend_token[4];
    uint64_t internal_token[4];
};
#define FS_APP_FRAME_INIT { sizeof(FS_AppFrame), FS_APP_FRAME_EMPTY, \
    FS_APP_FRAME_ACQUIRE_AVAILABLE, 0, 0, 0, 0, 0, 0, \
    WGPUTextureFormat_Undefined, NULL, NULL, {0,0}, {0,0,0,0}, {0,0,0,0} }

typedef struct FS_AppWindowDesc {
    uint32_t struct_size;
    const char* title;
    uint32_t width, height;
    bool resizable, visible, decorated, transparent, high_dpi;
    void* native_window;
    FS_ResourceOwnership native_window_ownership;
} FS_AppWindowDesc;
#define FS_APP_WINDOW_DESC_INIT { sizeof(FS_AppWindowDesc), "WCN", 1280, 720, true, true, true, false, true, NULL, FS_RESOURCE_NONE }

typedef FS_Result (FS_CALL *FS_BackendHostEnqueueFn)(FS_AppBackendHost*, const FS_AppEvent*, FS_Error*);
typedef FS_Result (FS_CALL *FS_BackendHostPostTaskFn)(FS_AppBackendHost*, FS_AppTaskFn, void*, FS_AppTaskDestroyFn, FS_Error*);
typedef void (FS_CALL *FS_BackendHostWakeFn)(FS_AppBackendHost*);
typedef FS_Result (FS_CALL *FS_BackendHostSetThreadRolesFn)(FS_AppBackendHost*, const FS_AppThreadRoles*, FS_Error*);

struct FS_AppBackendHost {
    uint32_t struct_size;
    const FS_Allocator* allocator;
    FS_DiagnosticSink diagnostics;
    FS_BackendHostEnqueueFn enqueue_event;
    FS_BackendHostPostTaskFn post_task;
    FS_BackendHostWakeFn wake;
    FS_BackendHostSetThreadRolesFn set_thread_roles;
    FS_GpuContext* gpu;
    void* private_data;
};

typedef struct FS_AppBackendDesc {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* backend_name;
} FS_AppBackendDesc;

typedef struct FS_AppBackendOps {
    uint32_t struct_size;
    uint32_t abi_version;
    FS_Result (FS_CALL *create)(FS_AppBackendHost*, const FS_AppBackendDesc*, FS_BackendInstance**, FS_Error*);
    void (FS_CALL *destroy)(FS_BackendInstance*);
    FS_Result (FS_CALL *create_window)(FS_BackendInstance*, const FS_AppWindowDesc*, FS_BackendWindow**, FS_Error*);
    void (FS_CALL *destroy_window)(FS_BackendInstance*, FS_BackendWindow*);
    FS_Result (FS_CALL *pump_events)(FS_BackendInstance*, FS_AppPumpMode, uint64_t, FS_Error*);
    void (FS_CALL *request_wake)(FS_BackendInstance*);
    FS_Result (FS_CALL *get_window_metrics)(FS_BackendInstance*, FS_BackendWindow*, FS_AppWindowMetrics*, FS_Error*);
    bool (FS_CALL *window_should_close)(FS_BackendInstance*, FS_BackendWindow*);
    const FS_CapabilityHeader* (FS_CALL *query_capability)(FS_BackendInstance*, uint32_t, uint32_t);
    FS_Result (FS_CALL *acquire_frame)(FS_BackendInstance*, FS_BackendWindow*, FS_AppFrame*, FS_Error*);
    FS_Result (FS_CALL *present_frame)(FS_BackendInstance*, FS_BackendWindow*, FS_AppFrame*, FS_Error*);
    void (FS_CALL *cancel_frame)(FS_BackendInstance*, FS_BackendWindow*, FS_AppFrame*);
} FS_AppBackendOps;

typedef struct FS_AppBackendFactory {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* name;
    const FS_AppBackendOps* ops;
} FS_AppBackendFactory;

FS_API FS_Result FS_CALL fs_app_backend_register(const FS_AppBackendFactory* factory, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_backend_unregister(const char* name, FS_Error* error);
FS_API const FS_AppBackendFactory* FS_CALL fs_app_backend_find(const char* name);
FS_API uint32_t FS_CALL fs_app_backend_count(void);

#ifdef __cplusplus
}
#endif
#endif
