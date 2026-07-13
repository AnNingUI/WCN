#include "fullstack_gpu_private.h"
#include <assert.h>
#include <stdint.h>

static int releases[4]; static int submits;
static WGPUInstance FS_CALL fake_create(const WGPUInstanceDescriptor* d) { (void)d; return (WGPUInstance)(uintptr_t)1; }
static void FS_CALL rel_i(WGPUInstance h){(void)h;releases[0]++;}
static void FS_CALL rel_a(WGPUAdapter h){(void)h;releases[1]++;}
static void FS_CALL rel_d(WGPUDevice h){(void)h;releases[2]++;}
static void FS_CALL rel_q(WGPUQueue h){(void)h;releases[3]++;}
static void FS_CALL submit(WGPUQueue q,size_t n,const WGPUCommandBuffer*c){(void)q;(void)n;(void)c;submits++;}
static FS_GpuProcs procs={fake_create,rel_i,rel_a,rel_d,rel_q,submit,NULL,NULL,NULL};
static FS_GpuContextDesc desc(FS_ResourceOwnership o){
    FS_GpuContextDesc d=FS_GPU_CONTEXT_DESC_INIT;
    d.instance=(WGPUInstance)(uintptr_t)11; d.instance_ownership=o;
    d.adapter=(WGPUAdapter)(uintptr_t)12; d.adapter_ownership=o;
    d.device=(WGPUDevice)(uintptr_t)13; d.device_ownership=o;
    d.queue=(WGPUQueue)(uintptr_t)14; d.queue_ownership=o;
    d.lineage=(FS_GpuLineage){11,12,13}; return d;
}
int main(void){
    FS_Error e=FS_ERROR_INIT; FS_GpuContext*c=NULL;
    FS_GpuContextDesc b=desc(FS_RESOURCE_BORROWED);
    assert(fs_gpu_context_create_with_procs(&b,&procs,&c,&e)==FS_RESULT_OK);
    assert(fs_gpu_context_state(c)==FS_GPU_STATE_READY);
    FS_SubmissionToken t={0}; assert(fs_gpu_submit(c,0,NULL,&t,&e)==FS_RESULT_OK);
    FS_SubmissionStatus st=0; assert(fs_gpu_submission_status(c,t,&st,&e)==FS_RESULT_OK&&st==FS_SUBMISSION_SUCCEEDED);
    assert(submits==1); fs_gpu_context_destroy(c);
    assert(releases[0]+releases[1]+releases[2]+releases[3]==0);
    FS_GpuContextDesc x=desc(FS_RESOURCE_TRANSFERRED);
    assert(fs_gpu_context_create_with_procs(&x,&procs,&c,&e)==FS_RESULT_OK);
    assert(fs_gpu_context_replace_device(c,(WGPUDevice)(uintptr_t)23,FS_RESOURCE_TRANSFERRED,(WGPUQueue)(uintptr_t)24,FS_RESOURCE_TRANSFERRED,(FS_GpuLineage){11,12,23},&e)==FS_RESULT_OK);
    fs_gpu_context_destroy(c);
    assert(releases[0]==1&&releases[1]==1&&releases[2]==2&&releases[3]==2);
    FS_GpuContextDesc bad=FS_GPU_CONTEXT_DESC_INIT; bad.queue=(WGPUQueue)(uintptr_t)3; bad.queue_ownership=FS_RESOURCE_BORROWED;
    assert(fs_gpu_context_create_with_procs(&bad,&procs,&c,&e)==FS_RESULT_INVALID_ARGUMENT);
    return 0;
}
