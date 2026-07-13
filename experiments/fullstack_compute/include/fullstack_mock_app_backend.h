#ifndef FULLSTACK_MOCK_APP_BACKEND_H
#define FULLSTACK_MOCK_APP_BACKEND_H
#include "fullstack_app.h"
#ifdef __cplusplus
extern "C" {
#endif
FS_API FS_Result FS_CALL fs_app_register_mock_backend(FS_Error* error);
FS_API FS_Result FS_CALL fs_mock_backend_inject_event(FS_App* app, const FS_AppEvent* event, FS_Error* error);
FS_API FS_Result FS_CALL fs_mock_window_set_metrics(FS_AppWindow* window, const FS_AppWindowMetrics* metrics, FS_Error* error);
FS_API FS_Result FS_CALL fs_mock_window_set_acquire_status(
    FS_AppWindow* window, FS_AppFrameAcquireStatus status);
FS_API FS_Result FS_CALL fs_mock_window_set_present_result(
    FS_AppWindow* window, FS_Result result);
FS_API uint32_t FS_CALL fs_mock_window_present_count(const FS_AppWindow* window);
FS_API uint32_t FS_CALL fs_mock_window_cancel_count(const FS_AppWindow* window);
#ifdef __cplusplus
}
#endif
#endif
