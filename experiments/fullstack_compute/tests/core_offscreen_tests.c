#include "fullstack_gpu.h"
#include "fullstack_core.h"

#include <assert.h>

static FS_CoreSceneOutput encode(FS_GpuContext* gpu, FS_Core* core) {
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        fs_gpu_device(gpu), &(WGPUCommandEncoderDescriptor){0});
    assert(encoder);
    FS_CoreSceneOutput scene = {0};
    scene.struct_size = sizeof(scene);
    assert(fs_core_encode_scene(core, encoder, 0, 0, 0, 1, &scene));
    WGPUCommandBuffer command = wgpuCommandEncoderFinish(
        encoder, &(WGPUCommandBufferDescriptor){0});
    assert(command);
    FS_Error error = FS_ERROR_INIT;
    FS_SubmissionToken token = {0};
    assert(fs_gpu_submit(gpu, 1, &command, &token, &error) == FS_RESULT_OK);
    wgpuCommandBufferRelease(command);
    wgpuCommandEncoderRelease(encoder);
    return scene;
}

int main(void) {
    FS_GpuContextDesc gpu_desc = FS_GPU_CONTEXT_DESC_INIT;
    FS_GpuContext* gpu = NULL;
    FS_Error error = FS_ERROR_INIT;
    assert(fs_gpu_context_create(&gpu_desc, &gpu, &error) == FS_RESULT_OK);

    FS_Core* core = fs_core_create(fs_gpu_device(gpu), fs_gpu_queue(gpu),
                                   WGPUTextureFormat_RGBA8Unorm, 64, 64);
    assert(core);
    fs_core_begin_commands(core);
    assert(fs_cmd_rect(core, 0, 0, 64, 64, 0, 0xff0000ffu));
    FS_CoreSceneOutput first = encode(gpu, core);
    assert(first.texture && first.view && first.width == 64 && first.height == 64);
    assert(first.format == WGPUTextureFormat_RGBA8Unorm);
    assert(first.generation != 0);

    assert(fs_core_try_resize(core, 128, 96));
    fs_core_begin_commands(core);
    assert(fs_cmd_rect(core, 0, 0, 128, 96, 0, 0x00ff00ffu));
    FS_CoreSceneOutput resized = encode(gpu, core);
    assert(resized.width == 128 && resized.height == 96);
    assert(resized.generation > first.generation);

    assert(!fs_core_try_resize(core, 0, 96));
    fs_core_begin_commands(core);
    assert(fs_cmd_rect(core, 0, 0, 128, 96, 0, 0x0000ffffu));
    FS_CoreSceneOutput preserved = encode(gpu, core);
    assert(preserved.width == 128 && preserved.height == 96);
    assert(preserved.generation == resized.generation);

    fs_core_destroy(core);
    fs_gpu_context_destroy(gpu);
    return 0;
}
