#include "fullstack_gpu_private.h"
#include <string.h>

typedef struct FS_SubmissionRecord {
    uint64_t serial;
    FS_SubmissionStatus status;
} FS_SubmissionRecord;

struct FS_GpuContext {
    const FS_Allocator* allocator;
    FS_DiagnosticSink diagnostics;
    FS_GpuProcs procs;
    uint64_t context_id;
    uint64_t next_serial;
    FS_GpuState state;
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    bool own_instance, own_adapter, own_device, own_queue;
    FS_GpuCapabilities capabilities;
    FS_SubmissionRecord** submissions;
    uint32_t submission_count;
    uint32_t submission_capacity;
};

static uint64_t fs_next_gpu_context_id = 1;
static WGPUInstance FS_CALL fs_real_create_instance(const WGPUInstanceDescriptor* d) { return wgpuCreateInstance(d); }
static void FS_CALL fs_real_instance_release(WGPUInstance h) { wgpuInstanceRelease(h); }
static void FS_CALL fs_real_adapter_release(WGPUAdapter h) { wgpuAdapterRelease(h); }
static void FS_CALL fs_real_device_release(WGPUDevice h) { wgpuDeviceRelease(h); }
static void FS_CALL fs_real_queue_release(WGPUQueue h) { wgpuQueueRelease(h); }
static void FS_CALL fs_real_queue_submit(WGPUQueue q, size_t n, const WGPUCommandBuffer* c) { wgpuQueueSubmit(q, n, c); }
static WGPUFuture FS_CALL fs_real_queue_work_done(WGPUQueue q, WGPUQueueWorkDoneCallbackInfo i) { return wgpuQueueOnSubmittedWorkDone(q, i); }
static const FS_GpuProcs fs_real_procs = {
    fs_real_create_instance, fs_real_instance_release, fs_real_adapter_release,
    fs_real_device_release, fs_real_queue_release, fs_real_queue_submit,
    fs_real_queue_work_done
};


static void fs_gpu_work_done(WGPUQueueWorkDoneStatus status, WGPUStringView message,
                             void* userdata1, void* userdata2) {
    (void)message; (void)userdata2;
    FS_SubmissionRecord* record = (FS_SubmissionRecord*)userdata1;
    if (!record) return;
    record->status = status == WGPUQueueWorkDoneStatus_Success
        ? FS_SUBMISSION_SUCCEEDED : FS_SUBMISSION_FAILED;
}
typedef struct FS_AdapterRequestState {
    bool done;
    WGPURequestAdapterStatus status;
    WGPUAdapter adapter;
    char message[256];
} FS_AdapterRequestState;
typedef struct FS_DeviceRequestState {
    bool done;
    WGPURequestDeviceStatus status;
    WGPUDevice device;
    char message[256];
} FS_DeviceRequestState;
static void fs_copy_wgpu_message(char* dst, size_t capacity, WGPUStringView message) {
    if (!dst || !capacity) return;
    size_t n = message.data ? message.length : 0;
    if (n >= capacity) n = capacity - 1;
    if (n) memcpy(dst, message.data, n);
    dst[n] = '\0';
}
static void fs_adapter_requested(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                 WGPUStringView message, void* userdata1, void* userdata2) {
    (void)userdata2; FS_AdapterRequestState* s = (FS_AdapterRequestState*)userdata1;
    s->status = status; s->adapter = adapter; fs_copy_wgpu_message(s->message, sizeof(s->message), message); s->done = true;
}
static void fs_device_requested(WGPURequestDeviceStatus status, WGPUDevice device,
                                WGPUStringView message, void* userdata1, void* userdata2) {
    (void)userdata2; FS_DeviceRequestState* s = (FS_DeviceRequestState*)userdata1;
    s->status = status; s->device = device; fs_copy_wgpu_message(s->message, sizeof(s->message), message); s->done = true;
}
static void fs_gpu_uncaptured_error(WGPUDevice const* device, WGPUErrorType type,
                                    WGPUStringView message, void* userdata1, void* userdata2) {
    (void)device; (void)userdata2; FS_GpuContext* c = (FS_GpuContext*)userdata1;
    char text[256]; fs_copy_wgpu_message(text, sizeof(text), message);
    FS_Diagnostic d = {sizeof(d), FS_LOG_ERROR, FS_ERROR_DOMAIN_GPU,
        FS_RESULT_INTERNAL_ERROR, 0, c ? c->context_id : 0,
        "wgpu_uncaptured_error", text};
    if (c) fs_diagnostic_emit(&c->diagnostics, &d); (void)type;
}
static void fs_gpu_device_lost(WGPUDevice const* device, WGPUDeviceLostReason reason,
                               WGPUStringView message, void* userdata1, void* userdata2) {
    (void)device; (void)userdata2; FS_GpuContext* c = (FS_GpuContext*)userdata1;
    char text[256]; fs_copy_wgpu_message(text, sizeof(text), message);
    if (c) {
        c->state = FS_GPU_STATE_LOST;
        FS_Diagnostic d = {sizeof(d), FS_LOG_ERROR, FS_ERROR_DOMAIN_GPU,
            FS_RESULT_INTERNAL_ERROR, 0, c->context_id, "wgpu_device_lost", text};
        fs_diagnostic_emit(&c->diagnostics, &d);
    }
    (void)reason;
}
static bool fs_wait_adapter(WGPUInstance instance, FS_AdapterRequestState* state) {
    for (uint32_t i = 0; i < 100000u && !state->done; ++i) wgpuInstanceProcessEvents(instance);
    return state->done && state->status == WGPURequestAdapterStatus_Success && state->adapter;
}
static bool fs_wait_device(WGPUInstance instance, FS_DeviceRequestState* state) {
    for (uint32_t i = 0; i < 100000u && !state->done; ++i) wgpuInstanceProcessEvents(instance);
    return state->done && state->status == WGPURequestDeviceStatus_Success && state->device;
}
static bool fs_valid_ownership(FS_ResourceOwnership o, bool has_handle) {
    if (!has_handle) return o == FS_RESOURCE_NONE;
    return o == FS_RESOURCE_BORROWED || o == FS_RESOURCE_TRANSFERRED;
}
static void fs_gpu_fail(FS_Error* error, FS_Result code, const char* message) {
    FS_ERROR_SET(error, code, FS_ERROR_DOMAIN_GPU, 0, "fs_gpu_context_create", "%s", message);
}
static FS_Result fs_gpu_validate_desc(const FS_GpuContextDesc* d, FS_Error* e) {
    if (!d) return FS_RESULT_INVALID_ARGUMENT;
    FS_Result vr = fs_abi_validate_header((const FS_AbiHeader*)d,
        (uint32_t)(offsetof(FS_GpuContextDesc, allow_offscreen_orphan_device_bundle) + sizeof(bool)),
        FS_GPU_ABI_VERSION);
    if (vr != FS_RESULT_OK) { fs_gpu_fail(e, vr, "invalid GPU context descriptor ABI"); return vr; }
    if (!fs_valid_ownership(d->instance_ownership, d->instance != NULL) ||
        !fs_valid_ownership(d->adapter_ownership, d->adapter != NULL) ||
        !fs_valid_ownership(d->device_ownership, d->device != NULL) ||
        !fs_valid_ownership(d->queue_ownership, d->queue != NULL)) {
        fs_gpu_fail(e, FS_RESULT_INVALID_ARGUMENT, "handle and ownership fields disagree");
        return FS_RESULT_INVALID_ARGUMENT;
    }
    if (d->queue && !d->device) { fs_gpu_fail(e, FS_RESULT_INVALID_ARGUMENT, "Queue requires Device"); return FS_RESULT_INVALID_ARGUMENT; }
    if (d->adapter && !d->instance) { fs_gpu_fail(e, FS_RESULT_INVALID_ARGUMENT, "Adapter requires Instance"); return FS_RESULT_INVALID_ARGUMENT; }
    if (d->device && !d->adapter && (!d->allow_offscreen_orphan_device_bundle || d->compatible_surface)) {
        fs_gpu_fail(e, FS_RESULT_INVALID_ARGUMENT, "orphan Device bundle is offscreen-only and must be explicitly enabled");
        return FS_RESULT_INVALID_ARGUMENT;
    }
    if (d->compatible_surface && (!d->instance || !d->adapter) && d->device) {
        fs_gpu_fail(e, FS_RESULT_INVALID_ARGUMENT, "Surface-compatible injection requires Instance and Adapter lineage");
        return FS_RESULT_INVALID_ARGUMENT;
    }
    return FS_RESULT_OK;
}

FS_Result fs_gpu_context_create_with_procs(const FS_GpuContextDesc* d,
                                           const FS_GpuProcs* p,
                                           FS_GpuContext** out,
                                           FS_Error* error) {
    if (out) *out = NULL;
    if (!out || !p) return FS_RESULT_INVALID_ARGUMENT;
    FS_Result vr = fs_gpu_validate_desc(d, error); if (vr != FS_RESULT_OK) return vr;
    const FS_Allocator* a = d->allocator ? d->allocator : fs_default_allocator();
    FS_GpuContext* c = (FS_GpuContext*)fs_allocator_allocate(a, sizeof(*c), sizeof(void*));
    if (!c) { fs_gpu_fail(error, FS_RESULT_OUT_OF_MEMORY, "GPU context allocation failed"); return FS_RESULT_OUT_OF_MEMORY; }
    memset(c, 0, sizeof(*c)); c->allocator = a; c->diagnostics = d->diagnostics; c->procs = *p;
    c->context_id = fs_next_gpu_context_id++; c->next_serial = 1; c->state = FS_GPU_STATE_INITIALIZING;
    c->instance = d->instance; c->adapter = d->adapter; c->device = d->device; c->queue = d->queue;
    c->own_instance = d->instance_ownership == FS_RESOURCE_TRANSFERRED;
    c->own_adapter = d->adapter_ownership == FS_RESOURCE_TRANSFERRED;
    c->own_device = d->device_ownership == FS_RESOURCE_TRANSFERRED;
    c->own_queue = d->queue_ownership == FS_RESOURCE_TRANSFERRED;
    if (!c->instance) {
        WGPUInstanceDescriptor id = {0}; c->instance = p->create_instance(&id); c->own_instance = c->instance != NULL;
        if (!c->instance) { fs_gpu_context_destroy(c); fs_gpu_fail(error, FS_RESULT_INTERNAL_ERROR, "wgpuCreateInstance failed"); return FS_RESULT_INTERNAL_ERROR; }
    }
    if ((!c->adapter || !c->device) && p != &fs_real_procs) {
        fs_gpu_context_destroy(c);
        fs_gpu_fail(error, FS_RESULT_UNSUPPORTED, "test procs require a complete injected GPU bundle");
        return FS_RESULT_UNSUPPORTED;
    }
    if (!c->adapter) {
        FS_AdapterRequestState request = {0};
        WGPURequestAdapterOptions options = {0};
        options.compatibleSurface = d->compatible_surface;
        options.powerPreference = d->power_preference;
        wgpuInstanceRequestAdapter(c->instance, &options,
            (WGPURequestAdapterCallbackInfo){
                .mode = WGPUCallbackMode_AllowProcessEvents,
                .callback = fs_adapter_requested,
                .userdata1 = &request
            });
        if (!fs_wait_adapter(c->instance, &request)) {
            fs_gpu_context_destroy(c);
            fs_gpu_fail(error, FS_RESULT_INTERNAL_ERROR,
                        request.message[0] ? request.message : "Adapter request failed");
            return FS_RESULT_INTERNAL_ERROR;
        }
        c->adapter = request.adapter; c->own_adapter = true;
    }
    if (!c->device) {
        FS_DeviceRequestState request = {0};
        WGPUDeviceDescriptor device_desc = d->device_descriptor ? *d->device_descriptor : (WGPUDeviceDescriptor){0};
        device_desc.deviceLostCallbackInfo = (WGPUDeviceLostCallbackInfo){
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = fs_gpu_device_lost,
            .userdata1 = c
        };
        device_desc.uncapturedErrorCallbackInfo = (WGPUUncapturedErrorCallbackInfo){
            .callback = fs_gpu_uncaptured_error,
            .userdata1 = c
        };
        wgpuAdapterRequestDevice(c->adapter, &device_desc,
            (WGPURequestDeviceCallbackInfo){
                .mode = WGPUCallbackMode_AllowProcessEvents,
                .callback = fs_device_requested,
                .userdata1 = &request
            });
        if (!fs_wait_device(c->instance, &request)) {
            fs_gpu_context_destroy(c);
            fs_gpu_fail(error, FS_RESULT_INTERNAL_ERROR,
                        request.message[0] ? request.message : "Device request failed");
            return FS_RESULT_INTERNAL_ERROR;
        }
        c->device = request.device; c->own_device = true;
    }
    if (!c->queue) {
        c->queue = wgpuDeviceGetQueue(c->device); c->own_queue = c->queue != NULL;
        if (!c->queue) {
            fs_gpu_context_destroy(c);
            fs_gpu_fail(error, FS_RESULT_INTERNAL_ERROR, "Device Queue acquisition failed");
            return FS_RESULT_INTERNAL_ERROR;
        }
    }
    c->capabilities.struct_size = sizeof(c->capabilities); c->capabilities.lineage = d->lineage;
    c->capabilities.supports_surface_creation = c->instance && c->adapter;
    c->state = FS_GPU_STATE_READY; *out = c; return FS_RESULT_OK;
}
FS_Result FS_CALL fs_gpu_context_create(const FS_GpuContextDesc* d, FS_GpuContext** out, FS_Error* e) {
    return fs_gpu_context_create_with_procs(d, &fs_real_procs, out, e);
}
void FS_CALL fs_gpu_context_destroy(FS_GpuContext* c) {
    if (!c) return; c->state = FS_GPU_STATE_DESTROYING;
    if (c->device && c->procs.queue_work_done) {
        (void)wgpuDevicePoll(c->device, true, NULL);
        if (c->instance) wgpuInstanceProcessEvents(c->instance);
    }
    for (uint32_t i = 0; i < c->submission_count; ++i) {
        fs_allocator_deallocate(c->allocator, c->submissions[i],
                                sizeof(FS_SubmissionRecord), sizeof(void*));
    }
    if (c->submissions) fs_allocator_deallocate(c->allocator, c->submissions,
        c->submission_capacity * sizeof(*c->submissions), sizeof(void*));
    if (c->own_queue && c->queue) c->procs.queue_release(c->queue);
    if (c->own_device && c->device) c->procs.device_release(c->device);
    if (c->own_adapter && c->adapter) c->procs.adapter_release(c->adapter);
    if (c->own_instance && c->instance) c->procs.instance_release(c->instance);
    fs_allocator_deallocate(c->allocator, c, sizeof(*c), sizeof(void*));
}
FS_Result FS_CALL fs_gpu_context_poll(FS_GpuContext* c, FS_Error* e) { (void)e; if (!c) return FS_RESULT_INVALID_ARGUMENT; if (c->instance) wgpuInstanceProcessEvents(c->instance); return FS_RESULT_OK; }
FS_GpuState FS_CALL fs_gpu_context_state(const FS_GpuContext* c) { return c ? c->state : FS_GPU_STATE_LOST; }
uint64_t FS_CALL fs_gpu_context_id(const FS_GpuContext* c) { return c ? c->context_id : 0; }
WGPUInstance FS_CALL fs_gpu_instance(const FS_GpuContext* c) { return c ? c->instance : NULL; }
WGPUAdapter FS_CALL fs_gpu_adapter(const FS_GpuContext* c) { return c ? c->adapter : NULL; }
WGPUDevice FS_CALL fs_gpu_device(const FS_GpuContext* c) { return c ? c->device : NULL; }
WGPUQueue FS_CALL fs_gpu_queue(const FS_GpuContext* c) { return c ? c->queue : NULL; }
const FS_GpuCapabilities* FS_CALL fs_gpu_capabilities(const FS_GpuContext* c) { return c ? &c->capabilities : NULL; }
FS_Result FS_CALL fs_gpu_context_replace_device(
    FS_GpuContext* c,
    WGPUDevice device, FS_ResourceOwnership device_ownership,
    WGPUQueue queue, FS_ResourceOwnership queue_ownership,
    FS_GpuLineage lineage, FS_Error* error) {
    if (!c || !device || !queue ||
        !fs_valid_ownership(device_ownership, true) ||
        !fs_valid_ownership(queue_ownership, true)) {
        return FS_RESULT_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < c->submission_count; ++i) {
        if (c->submissions[i]->status == FS_SUBMISSION_PENDING) {
            FS_ERROR_SET(error, FS_RESULT_PENDING, FS_ERROR_DOMAIN_GPU, 0,
                         "fs_gpu_context_replace_device",
                         "pending submissions must retire before Device replacement");
            return FS_RESULT_PENDING;
        }
    }
    if (c->own_queue && c->queue) c->procs.queue_release(c->queue);
    if (c->own_device && c->device) c->procs.device_release(c->device);
    c->device = device;
    c->queue = queue;
    c->own_device = device_ownership == FS_RESOURCE_TRANSFERRED;
    c->own_queue = queue_ownership == FS_RESOURCE_TRANSFERRED;
    c->capabilities.lineage = lineage;
    c->state = FS_GPU_STATE_READY;
    return FS_RESULT_OK;
}
FS_Result FS_CALL fs_gpu_submit(FS_GpuContext* c, uint32_t count,
                                const WGPUCommandBuffer* commands,
                                FS_SubmissionToken* out, FS_Error* e) {
    if (out) memset(out, 0, sizeof(*out));
    if (!c || !out || (count && !commands)) return FS_RESULT_INVALID_ARGUMENT;
    if (c->state != FS_GPU_STATE_READY) return FS_RESULT_INVALID_STATE;
    if (c->submission_count == c->submission_capacity) {
        uint32_t next = c->submission_capacity ? c->submission_capacity * 2u : 16u;
        size_t old_n = c->submission_capacity * sizeof(*c->submissions);
        size_t new_n = next * sizeof(*c->submissions);
        void* array = fs_allocator_reallocate(c->allocator, c->submissions,
                                               old_n, new_n, sizeof(void*));
        if (!array) {
            FS_ERROR_SET(e, FS_RESULT_OUT_OF_MEMORY, FS_ERROR_DOMAIN_GPU, 0,
                         "fs_gpu_submit", "submission array allocation failed");
            return FS_RESULT_OUT_OF_MEMORY;
        }
        c->submissions = (FS_SubmissionRecord**)array;
        c->submission_capacity = next;
    }
    FS_SubmissionRecord* record = (FS_SubmissionRecord*)fs_allocator_allocate(
        c->allocator, sizeof(*record), sizeof(void*));
    if (!record) {
        FS_ERROR_SET(e, FS_RESULT_OUT_OF_MEMORY, FS_ERROR_DOMAIN_GPU, 0,
                     "fs_gpu_submit", "submission record allocation failed");
        return FS_RESULT_OUT_OF_MEMORY;
    }
    record->serial = c->next_serial++;
    record->status = c->procs.queue_work_done
        ? FS_SUBMISSION_PENDING : FS_SUBMISSION_SUCCEEDED;
    c->submissions[c->submission_count++] = record;
    c->procs.queue_submit(c->queue, count, commands);
    if (c->procs.queue_work_done) {
        (void)c->procs.queue_work_done(c->queue,
            (WGPUQueueWorkDoneCallbackInfo){
                .mode = WGPUCallbackMode_AllowProcessEvents,
                .callback = fs_gpu_work_done,
                .userdata1 = record
            });
    }
    out->context_id = c->context_id;
    out->serial = record->serial;
    return FS_RESULT_OK;
}
FS_Result FS_CALL fs_gpu_submission_status(const FS_GpuContext* c, FS_SubmissionToken t,
                                            FS_SubmissionStatus* out, FS_Error* e) {
    (void)e; if (!c || !out || t.context_id != c->context_id || !t.serial) return FS_RESULT_INVALID_ARGUMENT;
    for (uint32_t i=0;i<c->submission_count;i++) if (c->submissions[i]->serial==t.serial) { *out=c->submissions[i]->status; return FS_RESULT_OK; }
    *out=FS_SUBMISSION_UNKNOWN; return FS_RESULT_SKIP;
}
