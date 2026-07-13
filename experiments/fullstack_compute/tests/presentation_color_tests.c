#include "fullstack_gpu.h"
#include "fullstack_present.h"

#include <assert.h>
#include <stdint.h>

typedef struct MapState {
    bool done;
    bool success;
} MapState;

static void map_done(WGPUMapAsyncStatus status, WGPUStringView message,
                     void* data, void* unused) {
    (void)message;
    (void)unused;
    MapState* state = (MapState*)data;
    state->success = status == WGPUMapAsyncStatus_Success;
    state->done = true;
}

static uint8_t render_gray(FS_GpuContext* gpu, WGPUTextureFormat format,
                           FS_ColorSpace color_space) {
    WGPUDevice device = fs_gpu_device(gpu);
    WGPUQueue queue = fs_gpu_queue(gpu);
    WGPUTextureDescriptor source_desc = {
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        .dimension = WGPUTextureDimension_2D,
        .size = {1, 1, 1},
        .format = WGPUTextureFormat_RGBA8Unorm,
        .mipLevelCount = 1,
        .sampleCount = 1};
    WGPUTexture source = wgpuDeviceCreateTexture(device, &source_desc);
    WGPUTextureView source_view = wgpuTextureCreateView(source, NULL);
    const uint8_t pixel[4] = {128, 128, 128, 255};
    WGPUTexelCopyTextureInfo destination = {
        .texture = source, .aspect = WGPUTextureAspect_All};
    WGPUTexelCopyBufferLayout upload_layout = {
        .bytesPerRow = 4, .rowsPerImage = 1};
    WGPUExtent3D one_pixel = {1, 1, 1};
    wgpuQueueWriteTexture(queue, &destination, pixel, sizeof(pixel),
                          &upload_layout, &one_pixel);

    WGPUTextureDescriptor target_desc = {
        .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
        .dimension = WGPUTextureDimension_2D,
        .size = {1, 1, 1},
        .format = format,
        .mipLevelCount = 1,
        .sampleCount = 1};
    WGPUTexture target = wgpuDeviceCreateTexture(device, &target_desc);
    WGPUTextureView target_view = wgpuTextureCreateView(target, NULL);
    WGPUBuffer readback = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor){
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = 256});
    assert(source && source_view && target && target_view && readback);

    FS_PresenterDesc presenter_desc = FS_PRESENTER_DESC_INIT;
    presenter_desc.device = device;
    presenter_desc.target_format = format;
    presenter_desc.target_color_space = color_space;
    FS_Presenter* presenter = NULL;
    FS_Error error = FS_ERROR_INIT;
    assert(fs_presenter_create(&presenter_desc, &presenter, &error) == FS_RESULT_OK);
    assert(fs_presenter_set_source(presenter, source_view, &error) == FS_RESULT_OK);
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        device, &(WGPUCommandEncoderDescriptor){0});
    assert(fs_presenter_encode(presenter, encoder, target_view, 1, 1, &error) ==
           FS_RESULT_OK);
    WGPUTexelCopyTextureInfo copy_source = {
        .texture = target, .aspect = WGPUTextureAspect_All};
    WGPUTexelCopyBufferInfo copy_destination = {
        .layout = {.bytesPerRow = 256, .rowsPerImage = 1},
        .buffer = readback};
    wgpuCommandEncoderCopyTextureToBuffer(
        encoder, &copy_source, &copy_destination, &one_pixel);
    WGPUCommandBuffer command = wgpuCommandEncoderFinish(
        encoder, &(WGPUCommandBufferDescriptor){0});
    FS_SubmissionToken token = {0};
    assert(fs_gpu_submit(gpu, 1, &command, &token, &error) == FS_RESULT_OK);

    MapState map = {0};
    wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, 256,
        (WGPUBufferMapCallbackInfo){
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback = map_done,
            .userdata1 = &map});
    while (!map.done) wgpuDevicePoll(device, true, NULL);
    assert(map.success);
    const uint8_t* bytes = (const uint8_t*)wgpuBufferGetConstMappedRange(
        readback, 0, 256);
    assert(bytes);
    uint8_t result = bytes[0];
    wgpuBufferUnmap(readback);

    wgpuCommandBufferRelease(command);
    wgpuCommandEncoderRelease(encoder);
    fs_presenter_destroy(presenter);
    wgpuBufferRelease(readback);
    wgpuTextureViewRelease(target_view);
    wgpuTextureRelease(target);
    wgpuTextureViewRelease(source_view);
    wgpuTextureRelease(source);
    return result;
}

int main(void) {
    FS_Error error = FS_ERROR_INIT;
    assert(fs_presenter_validate_color_contract(
        WGPUTextureFormat_RGBA8Unorm, FS_COLOR_SPACE_LINEAR_SRGB,
        FS_COLOR_SPACE_LINEAR_SRGB, &error) == FS_RESULT_OK);
    assert(fs_presenter_validate_color_contract(
        WGPUTextureFormat_BGRA8UnormSrgb, FS_COLOR_SPACE_LINEAR_SRGB,
        FS_COLOR_SPACE_SRGB, &error) == FS_RESULT_OK);
    assert(fs_presenter_validate_color_contract(
        WGPUTextureFormat_BGRA8Unorm, FS_COLOR_SPACE_LINEAR_SRGB,
        FS_COLOR_SPACE_SRGB, &error) == FS_RESULT_UNSUPPORTED);
    assert(fs_presenter_validate_color_contract(
        WGPUTextureFormat_BGRA8UnormSrgb, FS_COLOR_SPACE_LINEAR_SRGB,
        FS_COLOR_SPACE_DISPLAY_P3, &error) == FS_RESULT_UNSUPPORTED);
    assert(fs_presenter_validate_color_contract(
        WGPUTextureFormat_RGBA16Float, FS_COLOR_SPACE_LINEAR_SRGB,
        FS_COLOR_SPACE_HDR10, &error) == FS_RESULT_UNSUPPORTED);

    FS_GpuContextDesc gpu_desc = FS_GPU_CONTEXT_DESC_INIT;
    FS_GpuContext* gpu = NULL;
    assert(fs_gpu_context_create(&gpu_desc, &gpu, &error) == FS_RESULT_OK);
    uint8_t linear = render_gray(
        gpu, WGPUTextureFormat_RGBA8Unorm, FS_COLOR_SPACE_LINEAR_SRGB);
    uint8_t srgb = render_gray(
        gpu, WGPUTextureFormat_RGBA8UnormSrgb, FS_COLOR_SPACE_SRGB);
    assert(linear >= 126 && linear <= 130);
    assert(srgb >= 186 && srgb <= 190);
    fs_gpu_context_destroy(gpu);
    return 0;
}
