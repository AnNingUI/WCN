#include "fullstack_app_services.h"
#include "fullstack_app_private.h"

#include <string.h>

static const FS_CapabilityHeader* fs_app_capability(
    FS_App* app, uint32_t id) {
    if (!app || !app->factory || !app->factory->ops->query_capability)
        return NULL;
    return app->factory->ops->query_capability(app->backend, id, 1);
}

static bool fs_app_service_window(FS_App* app, FS_AppWindow* window) {
    return app && window && window->app == app;
}

FS_Result FS_CALL fs_app_clipboard_set_text(
    FS_App* app, FS_AppWindow* window, const char* utf8, FS_Error* error) {
    if (!fs_app_service_window(app, window) || !utf8)
        return FS_RESULT_INVALID_ARGUMENT;
    if (!fs_app_is_app_thread(app)) return FS_RESULT_WRONG_THREAD;
    const FS_AppClipboardCapability* capability =
        (const FS_AppClipboardCapability*)fs_app_capability(
            app, FS_APP_CAPABILITY_CLIPBOARD);
    return capability && capability->set_text
        ? capability->set_text(app->backend, window->backend_window, utf8, error)
        : FS_RESULT_UNSUPPORTED;
}

FS_Result FS_CALL fs_app_clipboard_get_text(
    FS_App* app, FS_AppWindow* window, FS_AppOwnedBytes* out_utf8,
    FS_Error* error) {
    if (out_utf8) memset(out_utf8, 0, sizeof(*out_utf8));
    if (!fs_app_service_window(app, window) || !out_utf8)
        return FS_RESULT_INVALID_ARGUMENT;
    if (!fs_app_is_app_thread(app)) return FS_RESULT_WRONG_THREAD;
    const FS_AppClipboardCapability* capability =
        (const FS_AppClipboardCapability*)fs_app_capability(
            app, FS_APP_CAPABILITY_CLIPBOARD);
    return capability && capability->get_text
        ? capability->get_text(app->backend, window->backend_window,
                               out_utf8, error)
        : FS_RESULT_UNSUPPORTED;
}

FS_Result FS_CALL fs_app_cursor_set_mode(
    FS_App* app, FS_AppWindow* window, FS_CursorMode mode, FS_Error* error) {
    if (!fs_app_service_window(app, window)) return FS_RESULT_INVALID_ARGUMENT;
    if (!fs_app_is_app_thread(app)) return FS_RESULT_WRONG_THREAD;
    const FS_AppCursorCapability* capability =
        (const FS_AppCursorCapability*)fs_app_capability(
            app, FS_APP_CAPABILITY_CURSOR);
    return capability && capability->set_mode
        ? capability->set_mode(app->backend, window->backend_window, mode, error)
        : FS_RESULT_UNSUPPORTED;
}

FS_Result FS_CALL fs_app_cursor_set_shape(
    FS_App* app, FS_AppWindow* window, FS_CursorShape shape, FS_Error* error) {
    if (!fs_app_service_window(app, window)) return FS_RESULT_INVALID_ARGUMENT;
    if (!fs_app_is_app_thread(app)) return FS_RESULT_WRONG_THREAD;
    const FS_AppCursorCapability* capability =
        (const FS_AppCursorCapability*)fs_app_capability(
            app, FS_APP_CAPABILITY_CURSOR);
    return capability && capability->set_shape
        ? capability->set_shape(app->backend, window->backend_window, shape, error)
        : FS_RESULT_UNSUPPORTED;
}

FS_Result FS_CALL fs_app_window_native_handle(
    FS_App* app, FS_AppWindow* window, FS_NativeWindowHandle* out_handle,
    FS_Error* error) {
    if (out_handle) memset(out_handle, 0, sizeof(*out_handle));
    if (!fs_app_service_window(app, window) || !out_handle)
        return FS_RESULT_INVALID_ARGUMENT;
    if (!fs_app_is_app_thread(app)) return FS_RESULT_WRONG_THREAD;
    const FS_AppNativeHandleCapability* capability =
        (const FS_AppNativeHandleCapability*)fs_app_capability(
            app, FS_APP_CAPABILITY_NATIVE_HANDLE);
    return capability && capability->get_handle
        ? capability->get_handle(app->backend, window->backend_window,
                                 out_handle, error)
        : FS_RESULT_UNSUPPORTED;
}

FS_Result FS_CALL fs_app_file_dialog_request(
    FS_App* app, const FS_FileDialogDesc* desc,
    FS_AppServiceRequestId* out_request, FS_Error* error) {
    if (out_request) *out_request = 0;
    if (!app || !desc || desc->struct_size < sizeof(*desc) || !out_request ||
        !desc->callback ||
        (desc->parent_window && desc->parent_window->app != app))
        return FS_RESULT_INVALID_ARGUMENT;
    if (!fs_app_is_app_thread(app)) return FS_RESULT_WRONG_THREAD;
    const FS_AppFileDialogCapability* capability =
        (const FS_AppFileDialogCapability*)fs_app_capability(
            app, FS_APP_CAPABILITY_FILE_DIALOG);
    return capability && capability->request
        ? capability->request(app->backend, desc, out_request, error)
        : FS_RESULT_UNSUPPORTED;
}

FS_Result FS_CALL fs_app_file_dialog_cancel(
    FS_App* app, FS_AppServiceRequestId request, FS_Error* error) {
    if (!app || !request) return FS_RESULT_INVALID_ARGUMENT;
    if (!fs_app_is_app_thread(app)) return FS_RESULT_WRONG_THREAD;
    const FS_AppFileDialogCapability* capability =
        (const FS_AppFileDialogCapability*)fs_app_capability(
            app, FS_APP_CAPABILITY_FILE_DIALOG);
    return capability && capability->cancel
        ? capability->cancel(app->backend, request, error)
        : FS_RESULT_UNSUPPORTED;
}

void FS_CALL fs_file_dialog_result_release(FS_FileDialogResult* result) {
    if (!result) return;
    fs_app_owned_bytes_release(&result->nul_separated_paths_utf8);
    memset(result, 0, sizeof(*result));
}
