#include "fullstack_core_private.h"

#include <stdlib.h>
#include <string.h>

bool fs_recreate_clip_compute_bind_group(FS_Core* core) {
    if (!core || !core->clip_compute_bgl || !core->clip_edge_buffer || !core->clip_job_buffer ||
        !core->clip_mask_view || !core->clip_dispatch_uniform_buffer) {
        return false;
    }
    if (core->clip_compute_bg) {
        wgpuBindGroupRelease(core->clip_compute_bg);
        core->clip_compute_bg = NULL;
    }
    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = core->clip_edge_buffer, .offset = 0, .size = core->clip_edge_buffer_size},
        {.binding = 1, .buffer = core->clip_job_buffer, .offset = 0, .size = core->clip_job_buffer_size},
        {.binding = 2, .textureView = core->clip_mask_view},
        {.binding = 3, .buffer = core->clip_dispatch_uniform_buffer, .offset = 0, .size = sizeof(FS_ClipDispatchUniforms)}
    };
    WGPUBindGroupDescriptor desc = {
        .nextInChain = NULL,
        .label = "FS Clip Compute Bind Group",
        .layout = core->clip_compute_bgl,
        .entryCount = 4,
        .entries = entries
    };
    core->clip_compute_bg = wgpuDeviceCreateBindGroup(core->device, &desc);
    return core->clip_compute_bg != NULL;
}

WGPUBindGroup fs_create_clip_compute_bind_group_range(
    FS_Core* core,
    uint64_t job_offset_bytes,
    uint64_t job_size_bytes
) {
    if (!core || !core->clip_compute_bgl || !core->clip_edge_buffer || !core->clip_job_buffer ||
        !core->clip_mask_view || !core->clip_dispatch_uniform_buffer) {
        return NULL;
    }
    if (job_size_bytes == 0u) {
        return NULL;
    }
    if (job_offset_bytes + job_size_bytes > core->clip_job_buffer_size) {
        return NULL;
    }

    const uint64_t edge_size_bytes = (core->clip_edge_buffer_size > 0u)
                                         ? core->clip_edge_buffer_size
                                         : (uint64_t)sizeof(FS_ClipEdgeGPU);
    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = core->clip_edge_buffer, .offset = 0u, .size = edge_size_bytes},
        {.binding = 1, .buffer = core->clip_job_buffer, .offset = job_offset_bytes, .size = job_size_bytes},
        {.binding = 2, .textureView = core->clip_mask_view},
        {.binding = 3, .buffer = core->clip_dispatch_uniform_buffer, .offset = 0u, .size = sizeof(FS_ClipDispatchUniforms)}
    };
    WGPUBindGroupDescriptor desc = {
        .nextInChain = NULL,
        .label = "FS Clip Compute Bind Group Range",
        .layout = core->clip_compute_bgl,
        .entryCount = 4,
        .entries = entries
    };
    return wgpuDeviceCreateBindGroup(core->device, &desc);
}

WGPUBindGroup fs_create_clip_edge_transform_bind_group_range(
    FS_Core* core,
    uint64_t job_offset_bytes,
    uint64_t job_size_bytes,
    uint64_t xform_offset_bytes,
    uint64_t xform_size_bytes
) {
    if (!core || !core->clip_edge_transform_bgl ||
        !core->clip_edge_local_buffer || !core->clip_job_buffer || !core->clip_job_xform_buffer ||
        !core->clip_edge_buffer) {
        return NULL;
    }
    if (job_size_bytes == 0u || xform_size_bytes == 0u) {
        return NULL;
    }
    if (job_offset_bytes + job_size_bytes > core->clip_job_buffer_size) {
        return NULL;
    }
    if (xform_offset_bytes + xform_size_bytes > core->clip_job_xform_buffer_size) {
        return NULL;
    }
    const uint64_t edge_local_size_bytes = (core->clip_edge_local_buffer_size > 0u)
                                               ? core->clip_edge_local_buffer_size
                                               : (uint64_t)sizeof(FS_ClipEdgeGPU);
    const uint64_t edge_device_size_bytes = (core->clip_edge_buffer_size > 0u)
                                                ? core->clip_edge_buffer_size
                                                : (uint64_t)sizeof(FS_ClipEdgeGPU);
    WGPUBindGroupEntry entries[] = {
        {.binding = 0, .buffer = core->clip_edge_local_buffer, .offset = 0u, .size = edge_local_size_bytes},
        {.binding = 1, .buffer = core->clip_job_buffer, .offset = job_offset_bytes, .size = job_size_bytes},
        {.binding = 2, .buffer = core->clip_job_xform_buffer, .offset = xform_offset_bytes, .size = xform_size_bytes},
        {.binding = 3, .buffer = core->clip_edge_buffer, .offset = 0u, .size = edge_device_size_bytes}
    };
    WGPUBindGroupDescriptor desc = {
        .nextInChain = NULL,
        .label = "FS Clip Edge Transform Bind Group Range",
        .layout = core->clip_edge_transform_bgl,
        .entryCount = 4,
        .entries = entries
    };
    return wgpuDeviceCreateBindGroup(core->device, &desc);
}

bool fs_ensure_clip_edge_gpu_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    const size_t min_count = 256u;
    size_t target = required > min_count ? required : min_count;
    size_t needed = target * sizeof(FS_ClipEdgeGPU);
    if (needed <= core->clip_edge_buffer_size && needed <= core->clip_edge_local_buffer_size &&
        core->clip_edge_buffer && core->clip_edge_local_buffer) {
        return true;
    }
    size_t new_size = core->clip_edge_buffer_size > core->clip_edge_local_buffer_size
                          ? core->clip_edge_buffer_size
                          : core->clip_edge_local_buffer_size;
    if (new_size < sizeof(FS_ClipEdgeGPU) * min_count) {
        new_size = sizeof(FS_ClipEdgeGPU) * min_count;
    }
    while (new_size < needed) {
        if (new_size > (SIZE_MAX / 2u)) {
            new_size = needed;
            break;
        }
        new_size *= 2u;
    }
    WGPUBuffer local_buf = fs_create_buffer(
        core->device,
        "FS Clip Edge Local Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        new_size
    );
    if (!local_buf) {
        return false;
    }
    WGPUBuffer device_buf = fs_create_buffer(
        core->device,
        "FS Clip Edge Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        new_size
    );
    if (!device_buf) {
        wgpuBufferRelease(local_buf);
        return false;
    }
    if (core->clip_edge_local_buffer) {
        wgpuBufferRelease(core->clip_edge_local_buffer);
    }
    if (core->clip_edge_buffer) {
        wgpuBufferRelease(core->clip_edge_buffer);
    }
    core->clip_edge_local_buffer = local_buf;
    core->clip_edge_local_buffer_size = new_size;
    core->clip_edge_buffer = device_buf;
    core->clip_edge_buffer_size = new_size;
    if (core->clip_compute_bgl && core->clip_job_buffer && core->clip_dispatch_uniform_buffer) {
        (void)fs_recreate_clip_compute_bind_group(core);
    }
    return true;
}

bool fs_ensure_clip_job_gpu_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    const size_t min_count = 64u;
    size_t target = required > min_count ? required : min_count;
    size_t needed = target * sizeof(FS_ClipJobGPU);
    if (needed <= core->clip_job_buffer_size && core->clip_job_buffer) {
        return true;
    }
    size_t new_size = core->clip_job_buffer_size ? core->clip_job_buffer_size : sizeof(FS_ClipJobGPU) * min_count;
    while (new_size < needed) {
        if (new_size > (SIZE_MAX / 2u)) {
            new_size = needed;
            break;
        }
        new_size *= 2u;
    }
    WGPUBuffer new_buf = fs_create_buffer(
        core->device,
        "FS Clip Job Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        new_size
    );
    if (!new_buf) {
        return false;
    }
    if (core->clip_job_buffer) {
        wgpuBufferRelease(core->clip_job_buffer);
    }
    core->clip_job_buffer = new_buf;
    core->clip_job_buffer_size = new_size;
    if (core->clip_compute_bgl && core->clip_edge_buffer && core->clip_dispatch_uniform_buffer) {
        (void)fs_recreate_clip_compute_bind_group(core);
    }
    return true;
}

bool fs_ensure_clip_job_transform_gpu_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    const size_t min_count = 64u;
    size_t target = required > min_count ? required : min_count;
    size_t needed = target * sizeof(FS_ClipJobTransformGPU);
    if (needed <= core->clip_job_xform_buffer_size && core->clip_job_xform_buffer) {
        return true;
    }
    size_t new_size = core->clip_job_xform_buffer_size ? core->clip_job_xform_buffer_size : sizeof(FS_ClipJobTransformGPU) * min_count;
    while (new_size < needed) {
        if (new_size > (SIZE_MAX / 2u)) {
            new_size = needed;
            break;
        }
        new_size *= 2u;
    }
    WGPUBuffer new_buf = fs_create_buffer(
        core->device,
        "FS Clip Job Transform Buffer",
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        new_size
    );
    if (!new_buf) {
        return false;
    }
    if (core->clip_job_xform_buffer) {
        wgpuBufferRelease(core->clip_job_xform_buffer);
    }
    core->clip_job_xform_buffer = new_buf;
    core->clip_job_xform_buffer_size = new_size;
    return true;
}

bool fs_ensure_clip_edge_cpu_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    if (required <= core->clip_edge_capacity) {
        return true;
    }
    size_t new_cap = core->clip_edge_capacity ? core->clip_edge_capacity : 256u;
    while (new_cap < required) {
        if (new_cap > (SIZE_MAX / 2u)) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }
    FS_ClipEdgeGPU* grown = (FS_ClipEdgeGPU*)realloc(core->clip_edge_cpu, new_cap * sizeof(FS_ClipEdgeGPU));
    if (!grown) {
        return false;
    }
    core->clip_edge_cpu = grown;
    core->clip_edge_capacity = new_cap;
    return true;
}

bool fs_ensure_clip_job_cpu_capacity(FS_Core* core, size_t required) {
    if (!core) {
        return false;
    }
    if (required <= core->clip_job_capacity && core->clip_job_cpu && core->clip_job_xform_cpu) {
        return true;
    }
    size_t new_cap = core->clip_job_capacity ? core->clip_job_capacity : 64u;
    while (new_cap < required) {
        if (new_cap > (SIZE_MAX / 2u)) {
            new_cap = required;
            break;
        }
        new_cap *= 2u;
    }

    FS_ClipJobGPU* grown_jobs = (FS_ClipJobGPU*)malloc(new_cap * sizeof(FS_ClipJobGPU));
    if (!grown_jobs) {
        return false;
    }
    FS_ClipJobTransformGPU* grown_xforms =
        (FS_ClipJobTransformGPU*)malloc(new_cap * sizeof(FS_ClipJobTransformGPU));
    if (!grown_xforms) {
        free(grown_jobs);
        return false;
    }

    size_t copy_count = core->clip_job_count;
    if (copy_count > new_cap) {
        copy_count = new_cap;
    }
    if (core->clip_job_cpu && copy_count > 0u) {
        memcpy(grown_jobs, core->clip_job_cpu, copy_count * sizeof(FS_ClipJobGPU));
    }
    if (core->clip_job_xform_cpu && copy_count > 0u) {
        memcpy(grown_xforms, core->clip_job_xform_cpu, copy_count * sizeof(FS_ClipJobTransformGPU));
    } else if (copy_count > 0u) {
        const FS_ClipJobTransformGPU identity = {
            .xform0 = {1.0f, 0.0f, 0.0f, 1.0f},
            .xform1 = {0.0f, 0.0f, 0.0f, 0.0f}
        };
        for (size_t i = 0u; i < copy_count; ++i) {
            grown_xforms[i] = identity;
        }
    }

    free(core->clip_job_cpu);
    free(core->clip_job_xform_cpu);
    core->clip_job_cpu = grown_jobs;
    core->clip_job_xform_cpu = grown_xforms;
    core->clip_job_capacity = new_cap;
    return true;
}
