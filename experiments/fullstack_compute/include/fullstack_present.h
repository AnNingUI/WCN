#ifndef FULLSTACK_PRESENT_H
#define FULLSTACK_PRESENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "fullstack_result.h"
#include <webgpu/wgpu.h>

#define FS_PRESENT_ABI_VERSION FS_ABI_VERSION(1, 0)

typedef uint32_t FS_ColorSpace;
#define FS_COLOR_SPACE_LINEAR_SRGB ((FS_ColorSpace)0u)
#define FS_COLOR_SPACE_SRGB        ((FS_ColorSpace)1u)
#define FS_COLOR_SPACE_DISPLAY_P3  ((FS_ColorSpace)2u)
#define FS_COLOR_SPACE_HDR10       ((FS_ColorSpace)3u)

typedef struct FS_Presenter FS_Presenter;

typedef struct FS_PresenterDesc {
    uint32_t struct_size;
    uint32_t abi_version;
    const FS_Allocator* allocator;
    WGPUDevice device;
    WGPUTextureFormat target_format;
    FS_ColorSpace source_color_space;
    FS_ColorSpace target_color_space;
} FS_PresenterDesc;

#define FS_PRESENTER_DESC_INIT { sizeof(FS_PresenterDesc), \
    FS_PRESENT_ABI_VERSION, NULL, NULL, WGPUTextureFormat_Undefined, \
    FS_COLOR_SPACE_LINEAR_SRGB, FS_COLOR_SPACE_SRGB }

FS_API FS_Result FS_CALL fs_presenter_create(
    const FS_PresenterDesc* desc, FS_Presenter** out_presenter, FS_Error* error);
FS_API void FS_CALL fs_presenter_destroy(FS_Presenter* presenter);
FS_API FS_Result FS_CALL fs_presenter_set_source(
    FS_Presenter* presenter, WGPUTextureView view, FS_Error* error);
FS_API FS_Result FS_CALL fs_presenter_encode(
    FS_Presenter* presenter, WGPUCommandEncoder encoder,
    WGPUTextureView target, uint32_t width, uint32_t height, FS_Error* error);
FS_API FS_Result FS_CALL fs_presenter_validate_color_contract(
    WGPUTextureFormat target_format, FS_ColorSpace source_color_space,
    FS_ColorSpace target_color_space, FS_Error* error);

#ifdef __cplusplus
}
#endif
#endif
