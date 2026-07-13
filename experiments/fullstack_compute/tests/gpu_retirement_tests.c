#include "fullstack_gpu_private.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct AllocationCounts {
    FS_Allocator allocator;
    uint32_t allocations;
    uint32_t deallocations;
} AllocationCounts;

static void* FS_CALL allocate_counted(void* data, size_t size, size_t alignment) {
    (void)alignment;
    AllocationCounts* counts = (AllocationCounts*)data;
    counts->allocations++;
    return malloc(size);
}
static void* FS_CALL reallocate_counted(void* data, void* memory,
                                        size_t old_size, size_t new_size,
                                        size_t alignment) {
    (void)old_size;
    (void)alignment;
    AllocationCounts* counts = (AllocationCounts*)data;
    if (!memory) counts->allocations++;
    return realloc(memory, new_size);
}
static void FS_CALL deallocate_counted(void* data, void* memory,
                                       size_t size, size_t alignment) {
    (void)size;
    (void)alignment;
    AllocationCounts* counts = (AllocationCounts*)data;
    counts->deallocations++;
    free(memory);
}

static WGPUQueueWorkDoneCallbackInfo pending_callback;
static uint32_t retired;
static WGPUInstance FS_CALL fake_create(const WGPUInstanceDescriptor* desc) {
    (void)desc;
    return (WGPUInstance)(uintptr_t)1;
}
static void FS_CALL ignore_instance(WGPUInstance value) { (void)value; }
static void FS_CALL ignore_adapter(WGPUAdapter value) { (void)value; }
static void FS_CALL ignore_device(WGPUDevice value) { (void)value; }
static void FS_CALL ignore_queue(WGPUQueue value) { (void)value; }
static void FS_CALL fake_submit(WGPUQueue queue, size_t count,
                                const WGPUCommandBuffer* commands) {
    (void)queue;
    (void)count;
    (void)commands;
}
static WGPUFuture FS_CALL fake_work_done(
    WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo callback) {
    (void)queue;
    pending_callback = callback;
    return (WGPUFuture){0};
}
static void FS_CALL fake_process_events(WGPUInstance instance) { (void)instance; }
static WGPUBool FS_CALL fake_poll(WGPUDevice device, WGPUBool wait,
                                  const WGPUSubmissionIndex* index) {
    (void)device;
    (void)wait;
    (void)index;
    return true;
}
static void FS_CALL retire(void* data) {
    uint32_t* value = (uint32_t*)data;
    (*value)++;
}

int main(void) {
    AllocationCounts counts = {0};
    counts.allocator = (FS_Allocator){
        sizeof(FS_Allocator), &counts, allocate_counted,
        reallocate_counted, deallocate_counted};
    FS_GpuContextDesc desc = FS_GPU_CONTEXT_DESC_INIT;
    desc.allocator = &counts.allocator;
    desc.instance = (WGPUInstance)(uintptr_t)11;
    desc.adapter = (WGPUAdapter)(uintptr_t)12;
    desc.device = (WGPUDevice)(uintptr_t)13;
    desc.queue = (WGPUQueue)(uintptr_t)14;
    desc.instance_ownership = FS_RESOURCE_BORROWED;
    desc.adapter_ownership = FS_RESOURCE_BORROWED;
    desc.device_ownership = FS_RESOURCE_BORROWED;
    desc.queue_ownership = FS_RESOURCE_BORROWED;
    desc.max_in_flight_submissions = 2;
    desc.max_deferred_retirements = 2;
    FS_GpuProcs procs = {
        fake_create, ignore_instance, ignore_adapter, ignore_device, ignore_queue,
        fake_submit, fake_work_done, fake_process_events, fake_poll};
    FS_GpuContext* gpu = NULL;
    FS_Error error = FS_ERROR_INIT;
    assert(fs_gpu_context_create_with_procs(&desc, &procs, &gpu, &error) ==
           FS_RESULT_OK);
    uint32_t steady_allocations = counts.allocations;
    FS_SubmissionToken token = {0};
    assert(fs_gpu_submit(gpu, 0, NULL, &token, &error) == FS_RESULT_OK);
    assert(counts.allocations == steady_allocations);
    assert(fs_gpu_retire_after(gpu, token, retire, &retired, &error) ==
           FS_RESULT_OK);
    assert(retired == 0);
    pending_callback.callback(WGPUQueueWorkDoneStatus_Success,
                              (WGPUStringView){0},
                              pending_callback.userdata1,
                              pending_callback.userdata2);
    assert(fs_gpu_collect_retired(gpu) == 1);
    assert(retired == 1);
    fs_gpu_context_destroy(gpu);
    assert(counts.allocations == counts.deallocations);
    return 0;
}
