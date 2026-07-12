#ifndef FULLSTACK_PRESENT_H
#define FULLSTACK_PRESENT_H
#include "fullstack_result.h"
#include <webgpu/wgpu.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct FS_Presenter FS_Presenter;
typedef struct FS_PresenterDesc { uint32_t struct_size; WGPUDevice device; WGPUTextureFormat target_format; } FS_PresenterDesc;
FS_API FS_Result FS_CALL fs_presenter_create(const FS_PresenterDesc*,FS_Presenter**,FS_Error*);
FS_API void FS_CALL fs_presenter_destroy(FS_Presenter*);
FS_API FS_Result FS_CALL fs_presenter_set_source(FS_Presenter*,WGPUTextureView,FS_Error*);
FS_API FS_Result FS_CALL fs_presenter_encode(FS_Presenter*,WGPUCommandEncoder,WGPUTextureView,uint32_t,uint32_t,FS_Error*);
#ifdef __cplusplus
}
#endif
#endif
