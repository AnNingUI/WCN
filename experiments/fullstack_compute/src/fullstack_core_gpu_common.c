#include "fullstack_core_private.h"

WGPUBuffer fs_create_buffer(WGPUDevice device, const char* label, WGPUBufferUsage usage, size_t size) {
    WGPUBufferDescriptor desc = {
        .nextInChain = NULL,
        .label = label,
        .usage = usage,
        .size = size,
        .mappedAtCreation = false
    };
    return wgpuDeviceCreateBuffer(device, &desc);
}
