#ifndef FULLSTACK_GPU_H
#define FULLSTACK_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fullstack_result.h"
#include <webgpu/wgpu.h>

#define FS_GPU_ABI_VERSION FS_ABI_VERSION(1, 0)

typedef uint32_t FS_ResourceOwnership;
#define FS_RESOURCE_NONE        ((FS_ResourceOwnership)0u)
#define FS_RESOURCE_BORROWED    ((FS_ResourceOwnership)1u)
#define FS_RESOURCE_TRANSFERRED ((FS_ResourceOwnership)2u)

typedef uint32_t FS_GpuState;
#define FS_GPU_STATE_INITIALIZING ((FS_GpuState)0u)
#define FS_GPU_STATE_READY        ((FS_GpuState)1u)
#define FS_GPU_STATE_LOST         ((FS_GpuState)2u)
#define FS_GPU_STATE_DESTROYING   ((FS_GpuState)3u)

typedef uint32_t FS_SubmissionStatus;
#define FS_SUBMISSION_UNKNOWN   ((FS_SubmissionStatus)0u)
#define FS_SUBMISSION_PENDING   ((FS_SubmissionStatus)1u)
#define FS_SUBMISSION_SUCCEEDED ((FS_SubmissionStatus)2u)
#define FS_SUBMISSION_FAILED    ((FS_SubmissionStatus)3u)

typedef struct FS_GpuLineage {
    uint64_t instance_id;
    uint64_t adapter_id;
    uint64_t device_id;
} FS_GpuLineage;

typedef struct FS_SubmissionToken {
    uint64_t context_id;
    uint64_t serial;
} FS_SubmissionToken;

typedef struct FS_GpuContextDesc {
    uint32_t struct_size;
    uint32_t abi_version;
    const FS_Allocator* allocator;
    FS_DiagnosticSink diagnostics;

    WGPUInstance instance;
    FS_ResourceOwnership instance_ownership;
    WGPUAdapter adapter;
    FS_ResourceOwnership adapter_ownership;
    WGPUDevice device;
    FS_ResourceOwnership device_ownership;
    WGPUQueue queue;
    FS_ResourceOwnership queue_ownership;
    FS_GpuLineage lineage;

    WGPUSurface compatible_surface;
    WGPUPowerPreference power_preference;
    const WGPUDeviceDescriptor* device_descriptor;
    bool allow_offscreen_orphan_device_bundle;
} FS_GpuContextDesc;

#define FS_GPU_CONTEXT_DESC_INIT { \
    sizeof(FS_GpuContextDesc), FS_GPU_ABI_VERSION, NULL, {0}, \
    NULL, FS_RESOURCE_NONE, NULL, FS_RESOURCE_NONE, \
    NULL, FS_RESOURCE_NONE, NULL, FS_RESOURCE_NONE, {0,0,0}, \
    NULL, WGPUPowerPreference_Undefined, NULL, false }

typedef struct FS_GpuCapabilities {
    uint32_t struct_size;
    FS_GpuLineage lineage;
    bool supports_surface_creation;
    WGPUBackendType backend_type;
    WGPUAdapterType adapter_type;
    WGPULimits adapter_limits;
    WGPULimits device_limits;
} FS_GpuCapabilities;

typedef struct FS_GpuContext FS_GpuContext;

FS_API FS_Result FS_CALL fs_gpu_context_create(const FS_GpuContextDesc* desc,
                                                FS_GpuContext** out_context,
                                                FS_Error* error);
FS_API void FS_CALL fs_gpu_context_destroy(FS_GpuContext* context);
FS_API FS_Result FS_CALL fs_gpu_context_poll(FS_GpuContext* context,
                                              FS_Error* error);
FS_API FS_GpuState FS_CALL fs_gpu_context_state(const FS_GpuContext* context);
FS_API uint64_t FS_CALL fs_gpu_context_id(const FS_GpuContext* context);
FS_API WGPUInstance FS_CALL fs_gpu_instance(const FS_GpuContext* context);
FS_API WGPUAdapter FS_CALL fs_gpu_adapter(const FS_GpuContext* context);
FS_API WGPUDevice FS_CALL fs_gpu_device(const FS_GpuContext* context);
FS_API WGPUQueue FS_CALL fs_gpu_queue(const FS_GpuContext* context);
FS_API const FS_GpuCapabilities* FS_CALL fs_gpu_capabilities(const FS_GpuContext* context);
FS_API FS_Result FS_CALL fs_gpu_context_replace_device(
    FS_GpuContext* context,
    WGPUDevice device, FS_ResourceOwnership device_ownership,
    WGPUQueue queue, FS_ResourceOwnership queue_ownership,
    FS_GpuLineage lineage, FS_Error* error);
FS_API FS_Result FS_CALL fs_gpu_submit(FS_GpuContext* context,
                                       uint32_t command_count,
                                       const WGPUCommandBuffer* commands,
                                       FS_SubmissionToken* out_token,
                                       FS_Error* error);
FS_API FS_Result FS_CALL fs_gpu_submission_status(const FS_GpuContext* context,
                                                  FS_SubmissionToken token,
                                                  FS_SubmissionStatus* out_status,
                                                  FS_Error* error);

#ifdef __cplusplus
}
#endif

#endif
