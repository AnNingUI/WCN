#ifndef FULLSTACK_RENDER_PRIVATE_H
#define FULLSTACK_RENDER_PRIVATE_H

#include "fullstack_render.h"

typedef struct FS_RenderPassSlot {
    FS_RenderPassDesc desc;
    uint32_t insertion_index;
} FS_RenderPassSlot;

struct FS_RenderTarget {
    const FS_Allocator* allocator;
    WGPUTexture texture;
    WGPUTextureView view;
    WGPUTextureFormat format;
    uint32_t width;
    uint32_t height;
    FS_ColorSpace color_space;
    uint64_t generation;
    bool importable;
};

struct FS_RenderFrame {
    FS_RenderContext* context;
    FS_RenderTarget* target;
    FS_RenderFrameState state;
    uint64_t target_generation;
    uint32_t pass_count;
    uint32_t next_insertion_index;
};

struct FS_RenderContext {
    const FS_Allocator* allocator;
    FS_DiagnosticSink diagnostics;
    FS_GpuContext* gpu;
    FS_Core* core;
    FS_Presenter* presenter;
    uint32_t width;
    uint32_t height;
    WGPUTextureFormat output_format;
    FS_ColorSpace output_color_space;
    uint32_t max_passes;
    FS_RenderPassSlot* passes;
    FS_RenderFrame frame;
    FS_SubmissionToken last_submission;
};

bool fs_render_pass_stage_is_custom(FS_RenderPassStage stage);
void fs_render_pass_insert(FS_RenderContext* context, FS_RenderPassSlot slot);
FS_Result fs_render_run_stage(FS_RenderFrame* frame, FS_RenderPassStage stage,
                              WGPUCommandEncoder encoder,
                              const FS_CoreSceneOutput* scene,
                              FS_Error* error);
void fs_render_frame_finish(FS_RenderFrame* frame, FS_RenderFrameState state);

#endif
