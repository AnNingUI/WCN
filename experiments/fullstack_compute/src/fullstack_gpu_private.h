#ifndef FULLSTACK_GPU_PRIVATE_H
#define FULLSTACK_GPU_PRIVATE_H
#include "fullstack_gpu.h"

typedef struct FS_GpuProcs {
    WGPUInstance (FS_CALL *create_instance)(const WGPUInstanceDescriptor*);
    void (FS_CALL *instance_release)(WGPUInstance);
    void (FS_CALL *adapter_release)(WGPUAdapter);
    void (FS_CALL *device_release)(WGPUDevice);
    void (FS_CALL *queue_release)(WGPUQueue);
    void (FS_CALL *queue_submit)(WGPUQueue, size_t, const WGPUCommandBuffer*);
    WGPUFuture (FS_CALL *queue_work_done)(WGPUQueue, WGPUQueueWorkDoneCallbackInfo);
} FS_GpuProcs;

FS_Result fs_gpu_context_create_with_procs(const FS_GpuContextDesc* desc,
                                           const FS_GpuProcs* procs,
                                           FS_GpuContext** out_context,
                                           FS_Error* error);
#endif
