#ifndef WCN_FULLSTACK_CORE_DEBUG_H
#define WCN_FULLSTACK_CORE_DEBUG_H

#include "fullstack_core.h"

typedef enum FS_ClipFailureReason {
    FS_CLIP_FAILURE_NONE = 0,
    FS_CLIP_FAILURE_INVALID_INPUT = 1,
    FS_CLIP_FAILURE_LAYER_EXHAUSTED = 2,
    FS_CLIP_FAILURE_EMPTY_PATH = 3,
    FS_CLIP_FAILURE_INVALID_BOUNDS = 4,
    FS_CLIP_FAILURE_EDGE_ALLOC = 5,
    FS_CLIP_FAILURE_JOB_ALLOC = 6
} FS_ClipFailureReason;

typedef struct FS_ClipDiagnostics {
    uint32_t requests_this_frame;
    uint32_t cache_hits_this_frame;
    uint32_t jobs_enqueued_this_frame;
    uint32_t layer_reuses_this_frame;
    uint32_t failures_this_frame;
    uint32_t layers_used_this_frame;
    uint32_t layer_capacity;
    FS_ClipFailureReason last_failure_reason;
    uint32_t last_failure_path_segments;
    uint32_t last_failure_edge_count;
    uint32_t dispatch_batches_this_frame;
    uint32_t dispatch_valid_jobs_this_frame;
    uint64_t dispatch_pixels_ideal_this_frame;
    uint64_t dispatch_pixels_estimated_this_frame;
    uint64_t dispatch_pixels_waste_this_frame;
    uint32_t dispatch_bucket_jobs_this_frame[6];
    uint32_t oriented_quad_commands_this_frame;
    uint32_t oriented_quad_clipped_this_frame;
} FS_ClipDiagnostics;

bool fs_core_get_clip_diagnostics(const FS_Core* core, FS_ClipDiagnostics* out_diagnostics);
void fs_core_set_clip_layer_reuse_reserve(FS_Core* core, uint32_t reserve_layers);
uint32_t fs_core_get_clip_layer_reuse_reserve(const FS_Core* core);
void fs_core_set_clip_cache_enabled(FS_Core* core, bool enabled);
bool fs_core_get_clip_cache_enabled(const FS_Core* core);
void fs_core_set_clip_aa_mode(FS_Core* core, int32_t mode);

#endif
