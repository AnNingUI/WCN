#include "fullstack_gpu.h"
#include <assert.h>
int main(void) {
    FS_GpuContextDesc desc = FS_GPU_CONTEXT_DESC_INIT;
    FS_GpuContext* gpu = NULL;
    FS_Error error = FS_ERROR_INIT;
    assert(fs_gpu_context_create(&desc, &gpu, &error) == FS_RESULT_OK);
    assert(gpu && fs_gpu_context_state(gpu) == FS_GPU_STATE_READY);
    assert(fs_gpu_instance(gpu) && fs_gpu_adapter(gpu));
    assert(fs_gpu_device(gpu) && fs_gpu_queue(gpu));
    FS_SubmissionToken token = {0};
    assert(fs_gpu_submit(gpu, 0, NULL, &token, &error) == FS_RESULT_OK);
    assert(fs_gpu_context_poll(gpu, &error) == FS_RESULT_OK);
    FS_SubmissionStatus status = FS_SUBMISSION_UNKNOWN;
    assert(fs_gpu_submission_status(gpu, token, &status, &error) == FS_RESULT_OK);
    assert(status == FS_SUBMISSION_PENDING || status == FS_SUBMISSION_SUCCEEDED);
    fs_gpu_context_destroy(gpu);
    return 0;
}
