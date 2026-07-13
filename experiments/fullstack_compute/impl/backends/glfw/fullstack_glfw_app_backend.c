#include "fullstack_glfw_app_backend_private.h"

#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

static volatile long fs_glfw_global_lock_value;
static uint32_t fs_glfw_global_references;
static FS_DiagnosticSink fs_glfw_global_diagnostics;

static void fs_glfw_lock(void) {
#if defined(_MSC_VER)
    while (_InterlockedExchange(&fs_glfw_global_lock_value, 1)) {}
#else
    while (__sync_lock_test_and_set(&fs_glfw_global_lock_value, 1)) {}
#endif
}
static void fs_glfw_unlock(void) {
#if defined(_MSC_VER)
    _InterlockedExchange(&fs_glfw_global_lock_value, 0);
#else
    __sync_lock_release(&fs_glfw_global_lock_value);
#endif
}

static void fs_glfw_error_callback(int native_code, const char* message) {
    FS_Diagnostic diagnostic = {
        sizeof(FS_Diagnostic), FS_LOG_ERROR, FS_ERROR_DOMAIN_BACKEND,
        FS_RESULT_INTERNAL_ERROR, 0, 0, "glfw", message ? message : "GLFW error"};
    (void)native_code;
    fs_diagnostic_emit(&fs_glfw_global_diagnostics, &diagnostic);
}

static FS_Result fs_glfw_global_acquire(const FS_DiagnosticSink* diagnostics,
                                        FS_Error* error) {
    fs_glfw_lock();
    if (!fs_glfw_global_references) {
        fs_glfw_global_diagnostics = diagnostics ? *diagnostics
                                                 : (FS_DiagnosticSink){0};
        glfwSetErrorCallback(fs_glfw_error_callback);
        if (!glfwInit()) {
            fs_glfw_unlock();
            FS_ERROR_SET(error, FS_RESULT_INTERNAL_ERROR,
                         FS_ERROR_DOMAIN_BACKEND, 0,
                         "fs_glfw_global_acquire", "glfwInit failed");
            return FS_RESULT_INTERNAL_ERROR;
        }
    }
    fs_glfw_global_references++;
    fs_glfw_unlock();
    return FS_RESULT_OK;
}

static void fs_glfw_global_release(void) {
    fs_glfw_lock();
    if (fs_glfw_global_references && --fs_glfw_global_references == 0) {
        glfwTerminate();
        fs_glfw_global_diagnostics = (FS_DiagnosticSink){0};
    }
    fs_glfw_unlock();
}

static FS_Result FS_CALL fs_glfw_create(
    FS_AppBackendHost* host, const FS_AppBackendDesc* desc,
    FS_BackendInstance** out_instance, FS_Error* error) {
    (void)desc;
    if (out_instance) *out_instance = NULL;
    if (!host || !out_instance || !host->gpu) {
        FS_ERROR_SET(error, FS_RESULT_INVALID_ARGUMENT,
                     FS_ERROR_DOMAIN_BACKEND, 0, "fs_glfw_create",
                     "GLFW backend requires a shared FS_GpuContext");
        return FS_RESULT_INVALID_ARGUMENT;
    }
    FS_Result result = fs_glfw_global_acquire(&host->diagnostics, error);
    if (result != FS_RESULT_OK) return result;
    FS_GlfwAppBackend* backend = (FS_GlfwAppBackend*)fs_allocator_allocate(
        host->allocator, sizeof(*backend), sizeof(void*));
    if (!backend) {
        fs_glfw_global_release();
        return FS_RESULT_OUT_OF_MEMORY;
    }
    memset(backend, 0, sizeof(*backend));
    backend->host = host;
    backend->glfw_acquired = true;
    backend->next_service_request = 1;
    *out_instance = (FS_BackendInstance*)backend;
    return FS_RESULT_OK;
}

static void FS_CALL fs_glfw_destroy(FS_BackendInstance* instance) {
    FS_GlfwAppBackend* backend = (FS_GlfwAppBackend*)instance;
    if (!backend) return;
    const FS_Allocator* allocator = backend->host->allocator;
    fs_glfw_services_shutdown(backend);
    if (backend->glfw_acquired) fs_glfw_global_release();
    fs_allocator_deallocate(allocator, backend, sizeof(*backend), sizeof(void*));
}

static FS_Result FS_CALL fs_glfw_pump(
    FS_BackendInstance* instance, FS_AppPumpMode mode,
    uint64_t timeout_ns, FS_Error* error) {
    (void)instance;
    (void)error;
    if (mode == FS_APP_PUMP_POLL) glfwPollEvents();
    else if (mode == FS_APP_PUMP_WAIT) glfwWaitEvents();
    else if (mode == FS_APP_PUMP_WAIT_TIMEOUT)
        glfwWaitEventsTimeout((double)timeout_ns / 1000000000.0);
    else return FS_RESULT_INVALID_ARGUMENT;
    return FS_RESULT_OK;
}

static void FS_CALL fs_glfw_wake(FS_BackendInstance* instance) {
    (void)instance;
    glfwPostEmptyEvent();
}

static const FS_CapabilityHeader* FS_CALL fs_glfw_capability(
    FS_BackendInstance* instance, uint32_t capability, uint32_t version) {
    return fs_glfw_services_query(
        (FS_GlfwAppBackend*)instance, capability, version);
}

FS_Result fs_glfw_window_create(FS_BackendInstance*, const FS_AppWindowDesc*,
                                FS_BackendWindow**, FS_Error*);
void fs_glfw_window_destroy(FS_BackendInstance*, FS_BackendWindow*);
FS_Result fs_glfw_window_metrics(FS_BackendInstance*, FS_BackendWindow*,
                                 FS_AppWindowMetrics*, FS_Error*);
bool fs_glfw_window_should_close(FS_BackendInstance*, FS_BackendWindow*);

static const FS_AppBackendOps fs_glfw_ops = {
    sizeof(FS_AppBackendOps), FS_APP_BACKEND_ABI_VERSION,
    fs_glfw_create, fs_glfw_destroy,
    fs_glfw_window_create, fs_glfw_window_destroy,
    fs_glfw_pump, fs_glfw_wake,
    fs_glfw_window_metrics, fs_glfw_window_should_close,
    fs_glfw_capability,
    fs_glfw_surface_acquire, fs_glfw_surface_present, fs_glfw_surface_cancel};

static const FS_AppBackendFactory fs_glfw_factory = {
    sizeof(FS_AppBackendFactory), FS_APP_BACKEND_ABI_VERSION,
    "glfw", &fs_glfw_ops};

FS_Result FS_CALL fs_app_register_glfw_backend(FS_Error* error) {
    if (fs_app_backend_find("glfw")) return FS_RESULT_OK;
    return fs_app_backend_register(&fs_glfw_factory, error);
}
