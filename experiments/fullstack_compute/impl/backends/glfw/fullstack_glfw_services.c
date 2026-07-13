#include "fullstack_glfw_app_backend_private.h"
#include "tinyfiledialogs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#if defined(_WIN32)
#include <intrin.h>
#include <process.h>
#include <windows.h>
#else
#include <pthread.h>
#endif

typedef struct FS_GlfwDialogRequest {
    FS_GlfwAppBackend* backend;
    const FS_Allocator* allocator;
    struct FS_GlfwDialogRequest* next;
    volatile long references;
    FS_AppServiceRequestId id;
    FS_AppWindowId parent_window_id;
    FS_FileDialogMode mode;
    FS_FileDialogCallback callback;
    void* user_data;
    volatile bool cancelled;
    bool joined;
    bool thread_started;
    char title[256];
    char default_path[1024];
    char filter_description[256];
    char filter_storage[16][128];
    const char* filter_patterns[16];
    uint32_t filter_count;
    FS_FileDialogResult result;
#if defined(_WIN32)
    HANDLE thread;
#else
    pthread_t thread;
#endif
} FS_GlfwDialogRequest;

static void fs_glfw_service_lock(FS_GlfwAppBackend* backend) {
#if defined(_MSC_VER)
    while (_InterlockedExchange(&backend->service_lock, 1)) {}
#else
    while (__sync_lock_test_and_set(&backend->service_lock, 1)) {}
#endif
}
static void fs_glfw_service_unlock(FS_GlfwAppBackend* backend) {
#if defined(_MSC_VER)
    _InterlockedExchange(&backend->service_lock, 0);
#else
    __sync_lock_release(&backend->service_lock);
#endif
}

static void fs_glfw_dialog_retain(FS_GlfwDialogRequest* request) {
#if defined(_MSC_VER)
    _InterlockedIncrement(&request->references);
#else
    __sync_add_and_fetch(&request->references, 1);
#endif
}

static void fs_glfw_dialog_release(FS_GlfwDialogRequest* request) {
    if (!request) return;
#if defined(_MSC_VER)
    long references = _InterlockedDecrement(&request->references);
#else
    long references = __sync_sub_and_fetch(&request->references, 1);
#endif
    if (references) return;
    fs_file_dialog_result_release(&request->result);
    fs_allocator_deallocate(request->allocator, request,
                            sizeof(*request), sizeof(void*));
}

static void fs_glfw_dialog_join(FS_GlfwDialogRequest* request) {
    if (!request || !request->thread_started || request->joined) return;
#if defined(_WIN32)
    WaitForSingleObject(request->thread, INFINITE);
    CloseHandle(request->thread);
#else
    pthread_join(request->thread, NULL);
#endif
    request->joined = true;
}

static void fs_glfw_dialog_unlink(FS_GlfwDialogRequest* request) {
    FS_GlfwAppBackend* backend = request ? request->backend : NULL;
    if (!backend) return;
    fs_glfw_service_lock(backend);
    FS_GlfwDialogRequest** link =
        (FS_GlfwDialogRequest**)&backend->service_requests;
    while (*link && *link != request) link = &(*link)->next;
    if (*link == request) *link = request->next;
    request->backend = NULL;
    fs_glfw_service_unlock(backend);
    fs_glfw_dialog_release(request); /* list reference */
}

static void FS_CALL fs_glfw_dialog_task(void* data) {
    FS_GlfwDialogRequest* request = (FS_GlfwDialogRequest*)data;
    fs_glfw_dialog_join(request);
    bool deliver = request->backend && !request->cancelled;
    fs_glfw_dialog_unlink(request);
    if (deliver && request->callback)
        request->callback(&request->result, request->user_data);
}

static void FS_CALL fs_glfw_dialog_task_destroy(void* data) {
    fs_glfw_dialog_release((FS_GlfwDialogRequest*)data); /* task reference */
}

static void fs_glfw_dialog_set_selection(FS_GlfwDialogRequest* request,
                                         const char* selection) {
    request->result.struct_size = sizeof(request->result);
    request->result.request_id = request->id;
    if (!selection || request->cancelled) {
        request->result.result = FS_RESULT_CANCELLED;
        return;
    }
    size_t length = strlen(selection);
    uint8_t* paths = (uint8_t*)fs_allocator_allocate(
        request->allocator, length + 1u, 1);
    if (!paths) {
        request->result.result = FS_RESULT_OUT_OF_MEMORY;
        return;
    }
    memcpy(paths, selection, length + 1u);
    uint32_t count = 1;
    for (size_t i = 0; i < length; ++i) {
        if (paths[i] == '|') {
            paths[i] = 0;
            count++;
        }
    }
    request->result.result = FS_RESULT_OK;
    request->result.path_count = count;
    request->result.nul_separated_paths_utf8.size = (uint32_t)length;
    request->result.nul_separated_paths_utf8.data = paths;
    request->result.nul_separated_paths_utf8.owner = (void*)request->allocator;
}

static void fs_glfw_dialog_run(FS_GlfwDialogRequest* request) {
    const char* selection = NULL;
    const char* title = request->title[0] ? request->title : NULL;
    const char* default_path = request->default_path[0]
        ? request->default_path : NULL;
    const char* description = request->filter_description[0]
        ? request->filter_description : NULL;
    const char* forced_result = getenv("FS_GLFW_FILE_DIALOG_TEST_RESULT");
    if (forced_result && strcmp(forced_result, "__CANCEL__") != 0) {
        selection = forced_result;
    } else if (!request->cancelled && !forced_result) {
        if (request->mode == FS_FILE_DIALOG_SAVE_FILE) {
            selection = tinyfd_saveFileDialog(title, default_path,
                (int)request->filter_count, request->filter_patterns,
                description);
        } else if (request->mode == FS_FILE_DIALOG_SELECT_DIR) {
            selection = tinyfd_selectFolderDialog(title, default_path);
        } else {
            selection = tinyfd_openFileDialog(title, default_path,
                (int)request->filter_count, request->filter_patterns,
                description,
                request->mode == FS_FILE_DIALOG_OPEN_FILES ? 1 : 0);
        }
    }
    fs_glfw_dialog_set_selection(request, selection);
    FS_GlfwAppBackend* backend = request->backend;
    if (!backend || backend->closing ||
        backend->host->post_task(backend->host, fs_glfw_dialog_task,
            request, fs_glfw_dialog_task_destroy, NULL) != FS_RESULT_OK) {
        fs_glfw_dialog_release(request); /* worker/task reference */
    }
}

#if defined(_WIN32)
static unsigned __stdcall fs_glfw_dialog_thread(void* data) {
    fs_glfw_dialog_run((FS_GlfwDialogRequest*)data);
    return 0;
}
#else
static void* fs_glfw_dialog_thread(void* data) {
    fs_glfw_dialog_run((FS_GlfwDialogRequest*)data);
    return NULL;
}
#endif

static FS_Result fs_glfw_dialog_start(FS_GlfwDialogRequest* request,
                                      FS_Error* error) {
#if defined(_WIN32)
    uintptr_t thread = _beginthreadex(NULL, 0, fs_glfw_dialog_thread,
                                      request, 0, NULL);
    if (!thread) {
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_BACKEND,
                     errno, "fs_glfw_dialog_start", "_beginthreadex failed");
        return FS_RESULT_INTERNAL_ERROR;
    }
    request->thread = (HANDLE)thread;
#else
    int native = pthread_create(&request->thread, NULL,
                                fs_glfw_dialog_thread, request);
    if (native != 0) {
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_BACKEND,
                     native, "fs_glfw_dialog_start", "pthread_create failed");
        return FS_RESULT_INTERNAL_ERROR;
    }
#endif
    request->thread_started = true;
    return FS_RESULT_OK;
}

static FS_Result FS_CALL fs_glfw_clipboard_set(
    FS_BackendInstance* instance, FS_BackendWindow* backend_window,
    const char* utf8, FS_Error* error) {
    (void)instance;
    (void)error;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)backend_window;
    if (!window || !utf8) return FS_RESULT_INVALID_ARGUMENT;
    glfwSetClipboardString(window->window, utf8);
    return FS_RESULT_OK;
}

static FS_Result FS_CALL fs_glfw_clipboard_get(
    FS_BackendInstance* instance, FS_BackendWindow* backend_window,
    FS_AppOwnedBytes* out_utf8, FS_Error* error) {
    FS_GlfwAppBackend* backend = (FS_GlfwAppBackend*)instance;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)backend_window;
    if (!backend || !window || !out_utf8) return FS_RESULT_INVALID_ARGUMENT;
    const char* text = glfwGetClipboardString(window->window);
    if (!text) return FS_RESULT_SKIP;
    size_t length = strlen(text);
    FS_AppEvent temporary = FS_APP_EVENT_INIT;
    temporary.type = FS_APP_EVENT_TEXT_INPUT;
    FS_Result result = fs_app_event_set_utf8(
        &temporary, text, (uint32_t)length, backend->host->allocator, error);
    if (result != FS_RESULT_OK) return result;
    *out_utf8 = temporary.data.text.utf8;
    if (!out_utf8->owner) {
        memcpy(out_utf8->inline_data, temporary.data.text.utf8.inline_data,
               sizeof(out_utf8->inline_data));
        out_utf8->data = out_utf8->inline_data;
    }
    return FS_RESULT_OK;
}

static FS_Result FS_CALL fs_glfw_cursor_mode(
    FS_BackendInstance* instance, FS_BackendWindow* backend_window,
    FS_CursorMode mode, FS_Error* error) {
    (void)instance;
    (void)error;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)backend_window;
    if (!window || mode > FS_CURSOR_MODE_DISABLED)
        return FS_RESULT_INVALID_ARGUMENT;
    const int glfw_modes[] = {GLFW_CURSOR_NORMAL, GLFW_CURSOR_HIDDEN,
                              GLFW_CURSOR_DISABLED};
    glfwSetInputMode(window->window, GLFW_CURSOR, glfw_modes[mode]);
    return FS_RESULT_OK;
}

static int fs_glfw_cursor_shape_id(FS_CursorShape shape) {
    switch (shape) {
        case FS_CURSOR_SHAPE_IBEAM: return GLFW_IBEAM_CURSOR;
        case FS_CURSOR_SHAPE_CROSS: return GLFW_CROSSHAIR_CURSOR;
        case FS_CURSOR_SHAPE_HAND: return GLFW_HAND_CURSOR;
#if defined(GLFW_HRESIZE_CURSOR)
        case FS_CURSOR_SHAPE_HRESIZE: return GLFW_HRESIZE_CURSOR;
        case FS_CURSOR_SHAPE_VRESIZE: return GLFW_VRESIZE_CURSOR;
#endif
        default: return GLFW_ARROW_CURSOR;
    }
}

static FS_Result FS_CALL fs_glfw_cursor_shape(
    FS_BackendInstance* instance, FS_BackendWindow* backend_window,
    FS_CursorShape shape, FS_Error* error) {
    FS_GlfwAppBackend* backend = (FS_GlfwAppBackend*)instance;
    FS_GlfwAppWindow* window = (FS_GlfwAppWindow*)backend_window;
    if (!backend || !window || shape > FS_CURSOR_SHAPE_VRESIZE)
        return FS_RESULT_INVALID_ARGUMENT;
    if (!backend->cursors[shape])
        backend->cursors[shape] = glfwCreateStandardCursor(
            fs_glfw_cursor_shape_id(shape));
    if (!backend->cursors[shape]) {
        FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR, FS_ERROR_DOMAIN_BACKEND, 0,
                     "fs_glfw_cursor_shape", "cursor creation failed");
        return FS_RESULT_INTERNAL_ERROR;
    }
    glfwSetCursor(window->window, backend->cursors[shape]);
    return FS_RESULT_OK;
}

static FS_Result FS_CALL fs_glfw_native_handle(
    FS_BackendInstance* instance, FS_BackendWindow* backend_window,
    FS_NativeWindowHandle* out_handle, FS_Error* error) {
    (void)instance;
    return fs_glfw_platform_native_handle(
        (FS_GlfwAppWindow*)backend_window, out_handle, error);
}

static FS_Result FS_CALL fs_glfw_dialog_request(
    FS_BackendInstance* instance, const FS_FileDialogDesc* desc,
    FS_AppServiceRequestId* out_request, FS_Error* error) {
    FS_GlfwAppBackend* backend = (FS_GlfwAppBackend*)instance;
    if (!backend || !desc || !out_request || desc->filter_count > 16 ||
        desc->mode > FS_FILE_DIALOG_SELECT_DIR) return FS_RESULT_INVALID_ARGUMENT;
    if (backend->closing) return FS_RESULT_INVALID_STATE;
    FS_GlfwDialogRequest* request =
        (FS_GlfwDialogRequest*)fs_allocator_allocate(
            backend->host->allocator, sizeof(*request), sizeof(void*));
    if (!request) return FS_RESULT_OUT_OF_MEMORY;
    memset(request, 0, sizeof(*request));
    request->backend = backend;
    request->allocator = backend->host->allocator;
    request->references = 2; /* backend list + worker/task */
    request->id = backend->next_service_request++;
    request->mode = desc->mode;
    request->callback = desc->callback;
    request->user_data = desc->user_data;
    request->parent_window_id = desc->parent_window
        ? fs_app_window_id(desc->parent_window) : 0;
    if (desc->title) snprintf(request->title, sizeof(request->title), "%s", desc->title);
    if (desc->default_path) snprintf(request->default_path,
        sizeof(request->default_path), "%s", desc->default_path);
    if (desc->filter_description) snprintf(request->filter_description,
        sizeof(request->filter_description), "%s", desc->filter_description);
    request->filter_count = desc->filter_count;
    for (uint32_t i = 0; i < desc->filter_count; ++i) {
        if (!desc->filters || !desc->filters[i]) {
            fs_glfw_dialog_release(request);
            fs_glfw_dialog_release(request);
            return FS_RESULT_INVALID_ARGUMENT;
        }
        snprintf(request->filter_storage[i], sizeof(request->filter_storage[i]),
                 "%s", desc->filters[i]);
        request->filter_patterns[i] = request->filter_storage[i];
    }
    fs_glfw_service_lock(backend);
    request->next = (FS_GlfwDialogRequest*)backend->service_requests;
    backend->service_requests = request;
    fs_glfw_service_unlock(backend);
    FS_Result result = fs_glfw_dialog_start(request, error);
    if (result != FS_RESULT_OK) {
        fs_glfw_dialog_unlink(request);
        fs_glfw_dialog_release(request);
        return result;
    }
    *out_request = request->id;
    return FS_RESULT_OK;
}

static FS_Result FS_CALL fs_glfw_dialog_cancel(
    FS_BackendInstance* instance, FS_AppServiceRequestId id, FS_Error* error) {
    (void)error;
    FS_GlfwAppBackend* backend = (FS_GlfwAppBackend*)instance;
    if (!backend || !id) return FS_RESULT_INVALID_ARGUMENT;
    fs_glfw_service_lock(backend);
    for (FS_GlfwDialogRequest* request =
             (FS_GlfwDialogRequest*)backend->service_requests;
         request; request = request->next) {
        if (request->id == id) {
            request->cancelled = true;
            fs_glfw_service_unlock(backend);
            return FS_RESULT_OK;
        }
    }
    fs_glfw_service_unlock(backend);
    return FS_RESULT_SKIP;
}

static const FS_AppClipboardCapability fs_glfw_clipboard_capability = {
    {sizeof(FS_AppClipboardCapability), FS_APP_CAPABILITY_CLIPBOARD, 1, 0, 0},
    fs_glfw_clipboard_set, fs_glfw_clipboard_get};
static const FS_AppCursorCapability fs_glfw_cursor_capability = {
    {sizeof(FS_AppCursorCapability), FS_APP_CAPABILITY_CURSOR, 1, 0, 0},
    fs_glfw_cursor_mode, fs_glfw_cursor_shape};
static const FS_AppNativeHandleCapability fs_glfw_native_capability = {
    {sizeof(FS_AppNativeHandleCapability), FS_APP_CAPABILITY_NATIVE_HANDLE,
     1, 0, 0}, fs_glfw_native_handle};
static const FS_AppFileDialogCapability fs_glfw_dialog_capability = {
    {sizeof(FS_AppFileDialogCapability), FS_APP_CAPABILITY_FILE_DIALOG,
     1, 0, 0}, fs_glfw_dialog_request, fs_glfw_dialog_cancel};

const FS_CapabilityHeader* fs_glfw_services_query(
    FS_GlfwAppBackend* backend, uint32_t capability, uint32_t version) {
    if (!backend || version > 1) return NULL;
    switch (capability) {
        case FS_APP_CAPABILITY_CLIPBOARD:
            return &fs_glfw_clipboard_capability.header;
        case FS_APP_CAPABILITY_CURSOR:
            return &fs_glfw_cursor_capability.header;
        case FS_APP_CAPABILITY_NATIVE_HANDLE:
            return &fs_glfw_native_capability.header;
        case FS_APP_CAPABILITY_FILE_DIALOG:
            return &fs_glfw_dialog_capability.header;
        default:
            return NULL;
    }
}

void fs_glfw_services_cancel_window(FS_GlfwAppBackend* backend,
                                    FS_AppWindowId window_id) {
    if (!backend || !window_id) return;
    fs_glfw_service_lock(backend);
    for (FS_GlfwDialogRequest* request =
             (FS_GlfwDialogRequest*)backend->service_requests;
         request; request = request->next)
        if (request->parent_window_id == window_id) request->cancelled = true;
    fs_glfw_service_unlock(backend);
}

void fs_glfw_services_shutdown(FS_GlfwAppBackend* backend) {
    if (!backend) return;
    backend->closing = true;
    for (uint32_t i = 0; i < 6; ++i) {
        if (backend->cursors[i]) glfwDestroyCursor(backend->cursors[i]);
        backend->cursors[i] = NULL;
    }
    for (;;) {
        fs_glfw_service_lock(backend);
        FS_GlfwDialogRequest* request =
            (FS_GlfwDialogRequest*)backend->service_requests;
        if (request) request->cancelled = true;
        fs_glfw_service_unlock(backend);
        if (!request) break;
        fs_glfw_dialog_join(request);
        fs_glfw_dialog_unlink(request);
    }
}
