#ifndef FULLSTACK_APP_SERVICES_H
#define FULLSTACK_APP_SERVICES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fullstack_app.h"

#define FS_APP_CAPABILITY_CLIPBOARD     0x1001u
#define FS_APP_CAPABILITY_CURSOR        0x1002u
#define FS_APP_CAPABILITY_NATIVE_HANDLE 0x1003u
#define FS_APP_CAPABILITY_FILE_DIALOG   0x1004u

typedef uint32_t FS_CursorMode;
#define FS_CURSOR_MODE_NORMAL   ((FS_CursorMode)0u)
#define FS_CURSOR_MODE_HIDDEN   ((FS_CursorMode)1u)
#define FS_CURSOR_MODE_DISABLED ((FS_CursorMode)2u)

typedef uint32_t FS_CursorShape;
#define FS_CURSOR_SHAPE_ARROW   ((FS_CursorShape)0u)
#define FS_CURSOR_SHAPE_IBEAM   ((FS_CursorShape)1u)
#define FS_CURSOR_SHAPE_CROSS   ((FS_CursorShape)2u)
#define FS_CURSOR_SHAPE_HAND    ((FS_CursorShape)3u)
#define FS_CURSOR_SHAPE_HRESIZE ((FS_CursorShape)4u)
#define FS_CURSOR_SHAPE_VRESIZE ((FS_CursorShape)5u)

typedef uint32_t FS_NativeHandleKind;
#define FS_NATIVE_HANDLE_NONE         ((FS_NativeHandleKind)0u)
#define FS_NATIVE_HANDLE_WIN32_HWND   ((FS_NativeHandleKind)1u)
#define FS_NATIVE_HANDLE_X11_WINDOW   ((FS_NativeHandleKind)2u)
#define FS_NATIVE_HANDLE_COCOA_WINDOW ((FS_NativeHandleKind)3u)
#define FS_NATIVE_HANDLE_GLFW_WINDOW  ((FS_NativeHandleKind)4u)

typedef struct FS_NativeWindowHandle {
    uint32_t struct_size;
    FS_NativeHandleKind kind;
    uintptr_t value;
    uintptr_t display;
} FS_NativeWindowHandle;

typedef uint32_t FS_FileDialogMode;
#define FS_FILE_DIALOG_OPEN_FILE   ((FS_FileDialogMode)0u)
#define FS_FILE_DIALOG_OPEN_FILES  ((FS_FileDialogMode)1u)
#define FS_FILE_DIALOG_SAVE_FILE   ((FS_FileDialogMode)2u)
#define FS_FILE_DIALOG_SELECT_DIR  ((FS_FileDialogMode)3u)

typedef uint64_t FS_AppServiceRequestId;

typedef struct FS_FileDialogResult {
    uint32_t struct_size;
    FS_AppServiceRequestId request_id;
    FS_Result result;
    uint32_t path_count;
    FS_AppOwnedBytes nul_separated_paths_utf8;
} FS_FileDialogResult;

typedef void (FS_CALL *FS_FileDialogCallback)(
    FS_FileDialogResult* result, void* user_data);

typedef struct FS_FileDialogDesc {
    uint32_t struct_size;
    FS_FileDialogMode mode;
    FS_AppWindow* parent_window;
    const char* title;
    const char* default_path;
    uint32_t filter_count;
    const char* const* filters;
    const char* filter_description;
    FS_FileDialogCallback callback;
    void* user_data;
} FS_FileDialogDesc;

#define FS_FILE_DIALOG_DESC_INIT { sizeof(FS_FileDialogDesc), \
    FS_FILE_DIALOG_OPEN_FILE, NULL, NULL, NULL, 0, NULL, NULL, NULL, NULL }

typedef struct FS_AppClipboardCapability {
    FS_CapabilityHeader header;
    FS_Result (FS_CALL *set_text)(FS_BackendInstance*, FS_BackendWindow*,
                                  const char*, FS_Error*);
    FS_Result (FS_CALL *get_text)(FS_BackendInstance*, FS_BackendWindow*,
                                  FS_AppOwnedBytes*, FS_Error*);
} FS_AppClipboardCapability;

typedef struct FS_AppCursorCapability {
    FS_CapabilityHeader header;
    FS_Result (FS_CALL *set_mode)(FS_BackendInstance*, FS_BackendWindow*,
                                  FS_CursorMode, FS_Error*);
    FS_Result (FS_CALL *set_shape)(FS_BackendInstance*, FS_BackendWindow*,
                                   FS_CursorShape, FS_Error*);
} FS_AppCursorCapability;

typedef struct FS_AppNativeHandleCapability {
    FS_CapabilityHeader header;
    FS_Result (FS_CALL *get_handle)(FS_BackendInstance*, FS_BackendWindow*,
                                    FS_NativeWindowHandle*, FS_Error*);
} FS_AppNativeHandleCapability;

typedef struct FS_AppFileDialogCapability {
    FS_CapabilityHeader header;
    FS_Result (FS_CALL *request)(FS_BackendInstance*, const FS_FileDialogDesc*,
                                 FS_AppServiceRequestId*, FS_Error*);
    FS_Result (FS_CALL *cancel)(FS_BackendInstance*, FS_AppServiceRequestId,
                                FS_Error*);
} FS_AppFileDialogCapability;

FS_API FS_Result FS_CALL fs_app_clipboard_set_text(
    FS_App* app, FS_AppWindow* window, const char* utf8, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_clipboard_get_text(
    FS_App* app, FS_AppWindow* window, FS_AppOwnedBytes* out_utf8,
    FS_Error* error);
FS_API FS_Result FS_CALL fs_app_cursor_set_mode(
    FS_App* app, FS_AppWindow* window, FS_CursorMode mode, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_cursor_set_shape(
    FS_App* app, FS_AppWindow* window, FS_CursorShape shape, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_window_native_handle(
    FS_App* app, FS_AppWindow* window, FS_NativeWindowHandle* out_handle,
    FS_Error* error);
FS_API FS_Result FS_CALL fs_app_file_dialog_request(
    FS_App* app, const FS_FileDialogDesc* desc,
    FS_AppServiceRequestId* out_request, FS_Error* error);
FS_API FS_Result FS_CALL fs_app_file_dialog_cancel(
    FS_App* app, FS_AppServiceRequestId request, FS_Error* error);
FS_API void FS_CALL fs_file_dialog_result_release(FS_FileDialogResult* result);

#ifdef __cplusplus
}
#endif
#endif
