#include "fullstack_render.h"

#include <assert.h>
#include <stdlib.h>

typedef struct CountingAllocator {
    FS_Allocator interface;
    uint32_t allocations;
    uint32_t deallocations;
} CountingAllocator;

static void* FS_CALL counting_allocate(void* data, size_t size, size_t alignment) {
    (void)alignment;
    CountingAllocator* allocator = (CountingAllocator*)data;
    allocator->allocations++;
    return malloc(size);
}

static void* FS_CALL counting_reallocate(void* data, void* memory,
                                         size_t old_size, size_t new_size,
                                         size_t alignment) {
    (void)old_size;
    (void)alignment;
    CountingAllocator* allocator = (CountingAllocator*)data;
    if (!memory) allocator->allocations++;
    return realloc(memory, new_size);
}

static void FS_CALL counting_deallocate(void* data, void* memory,
                                        size_t size, size_t alignment) {
    (void)size;
    (void)alignment;
    CountingAllocator* allocator = (CountingAllocator*)data;
    allocator->deallocations++;
    free(memory);
}

static void draw_frame(FS_RenderContext* context, FS_RenderTarget* target) {
    FS_Core* core = fs_render_context_core(context);
    fs_core_begin_commands(core);
    assert(fs_cmd_rect(core, 0, 0,
        (float)fs_render_target_width(target),
        (float)fs_render_target_height(target), 0, 0x336699ffu));
    FS_RenderFrame* frame = NULL;
    FS_Error error = FS_ERROR_INIT;
    assert(fs_render_context_begin_frame(context, target, &frame, &error) ==
           FS_RESULT_OK);
    FS_CommandBatch batch = {0};
    assert(fs_render_frame_encode(frame, 0, 0, 0, 1, &batch, &error) ==
           FS_RESULT_OK);
    assert(fs_command_batch_submit(context, &batch, &error) == FS_RESULT_OK);
    assert(batch.submitted && batch.submission.serial != 0);
    fs_command_batch_release(&batch);
}

int main(void) {
    CountingAllocator counting = {0};
    counting.interface = (FS_Allocator){
        sizeof(FS_Allocator), &counting, counting_allocate,
        counting_reallocate, counting_deallocate};
    FS_GpuContextDesc gpu_desc = FS_GPU_CONTEXT_DESC_INIT;
    gpu_desc.allocator = &counting.interface;
    FS_GpuContext* gpu = NULL;
    FS_Error error = FS_ERROR_INIT;
    assert(fs_gpu_context_create(&gpu_desc, &gpu, &error) == FS_RESULT_OK);

    FS_RenderContextDesc context_desc = FS_RENDER_CONTEXT_DESC_INIT;
    context_desc.allocator = &counting.interface;
    context_desc.gpu = gpu;
    context_desc.width = 64;
    context_desc.height = 64;
    context_desc.output_format = WGPUTextureFormat_RGBA8Unorm;
    context_desc.output_color_space = FS_COLOR_SPACE_LINEAR_SRGB;
    FS_RenderContext* context = NULL;
    assert(fs_render_context_create(&context_desc, &context, &error) ==
           FS_RESULT_OK);
    uint32_t before_shared_context = counting.allocations;
    FS_RenderContext* shared_context = NULL;
    assert(fs_render_context_create(&context_desc, &shared_context, &error) ==
           FS_RESULT_OK);
    assert(counting.allocations == before_shared_context + 3);
    fs_render_context_destroy(shared_context);

    FS_RenderTargetDesc target_desc = FS_RENDER_TARGET_DESC_INIT;
    target_desc.allocator = &counting.interface;
    target_desc.width = 64;
    target_desc.height = 64;
    target_desc.format = WGPUTextureFormat_RGBA8Unorm;
    target_desc.color_space = FS_COLOR_SPACE_LINEAR_SRGB;
    FS_RenderTarget* target = NULL;
    assert(fs_render_target_create_texture(gpu, &target_desc, &target, &error) ==
           FS_RESULT_OK);
    /* RenderContext owns an independent GPU reference. */
    fs_gpu_context_destroy(gpu);

    draw_frame(context, target);
    uint32_t steady_allocations = counting.allocations;
    draw_frame(context, target);
    assert(counting.allocations == steady_allocations);
    assert(fs_render_context_recreate_device_resources(context, &error) ==
           FS_RESULT_OK);
    draw_frame(context, target);

    target_desc.width = 96;
    target_desc.height = 80;
    target_desc.generation = 2;
    FS_RenderTarget* resized_target = NULL;
    assert(fs_render_target_create_texture(
        gpu, &target_desc, &resized_target, &error) == FS_RESULT_OK);
    draw_frame(context, resized_target);

    FS_RenderFrame* active = NULL;
    assert(fs_render_context_begin_frame(
        context, resized_target, &active, &error) == FS_RESULT_OK);
    assert(fs_render_context_try_resize(context, 120, 120, &error) ==
           FS_RESULT_INVALID_STATE);
    assert(fs_render_frame_cancel(active, &error) == FS_RESULT_OK);

    fs_render_target_destroy(resized_target);
    fs_render_target_destroy(target);
    fs_render_context_destroy(context);
    assert(counting.allocations == counting.deallocations);
    return 0;
}
