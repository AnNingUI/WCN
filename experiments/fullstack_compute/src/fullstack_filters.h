#ifndef WCN_FULLSTACK_FILTERS_H
#define WCN_FULLSTACK_FILTERS_H

#include "webgpu/webgpu.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Compile-time feature flag
// ---------------------------------------------------------------------------
#ifndef FS_EFFECTS_ENABLED
#define FS_EFFECTS_ENABLED 1
#endif

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
typedef struct FS_FilterNode FS_FilterNode;
typedef struct FS_Core FS_Core;
typedef struct FS_FilterChain FS_FilterChain;

// ---------------------------------------------------------------------------
// Filter chain type (full definition visible to all consumers)
// ---------------------------------------------------------------------------
struct FS_FilterChain {
    FS_FilterNode* head;
    FS_FilterNode* tail;
    uint32_t count;
};

// ---------------------------------------------------------------------------
// Filter type enumeration
// ---------------------------------------------------------------------------
typedef enum FS_FilterType {
    FS_FILTER_NONE = 0,
    FS_FILTER_BRIGHTNESS = 1,
    FS_FILTER_CONTRAST = 2,
    FS_FILTER_GRAYSCALE = 3,
    FS_FILTER_HUE_ROTATE = 4,
    FS_FILTER_INVERT = 5,
    FS_FILTER_OPACITY = 6,
    FS_FILTER_SATURATE = 7,
    FS_FILTER_SEPIA = 8,
    FS_FILTER_BLUR = 9,
    FS_FILTER_DROP_SHADOW = 10,
} FS_FilterType;

// ---------------------------------------------------------------------------
// Filter parameters union (matches WGSL filter shader params)
// ---------------------------------------------------------------------------
typedef struct FS_FilterParams {
    union {
        float amount;          // brightness, contrast, grayscale, invert, opacity, saturate, sepia
        float degrees;         // hue-rotate
        float radius;          // blur
        struct {
            float offset_x;
            float offset_y;
            float blur_radius;
            uint32_t color;   // RGBA8 packed color
        } drop_shadow;
    };
} FS_FilterParams;

// ---------------------------------------------------------------------------
// Filter node (singly-linked list entry)
// ---------------------------------------------------------------------------
typedef struct FS_FilterNode {
    FS_FilterType type;
    FS_FilterParams params;
    struct FS_FilterNode* next;
} FS_FilterNode;

// Gaussian kernel (used in effects and filter pipeline)
#define FS_GAUSSIAN_KERNEL_MAX_SIZE 63
typedef struct FS_GaussianKernel {
    uint32_t size;
    float sigma;
    float weights[FS_GAUSSIAN_KERNEL_MAX_SIZE];
} FS_GaussianKernel;

// Physical shadow parameters
typedef struct FS_PhysicalShadowParams {
    float blur_radius;
    uint32_t color;
    float offset_x;
    float offset_y;
    uint32_t kernel_size;
    float kernel_sigma;
    uint32_t flags;
    float alpha;
    float spread_px;
} FS_PhysicalShadowParams;

// ---------------------------------------------------------------------------
// GPU shader uniform structs
// ---------------------------------------------------------------------------

// Must match WGSL FilterUniforms (20 bytes, padded to 24)
typedef struct FS_FilterUniforms {
    uint32_t filter_type;
    float param1;
    float param2;
    float param3;
    float param4;
    uint8_t _pad[4];  // explicit padding to reach 24 bytes (WGPU uniform buffer alignment)
} FS_FilterUniforms;

// Must match WGSL ShadowUniforms (32 bytes)
typedef struct FS_ShadowUniforms {
    float shadow_color[4];   // RGBA float
    float shadow_offset[2];
    float _padding[2];
} FS_ShadowUniforms;

// Compile-time assertions using negative-size array idiom (C11 static_assert not available in gnu11 without <assert.h>)
typedef char fs_filter_filter_uniforms_check[(sizeof(FS_FilterUniforms) == 24) ? 1 : -1];
typedef char fs_filter_shadow_uniforms_check[(sizeof(FS_ShadowUniforms) == 32) ? 1 : -1];

// ---------------------------------------------------------------------------
// Filter chain API
// ---------------------------------------------------------------------------

// Create an empty filter chain
FS_FilterChain* fs_filter_chain_create(void);

// Destroy a filter chain (frees all nodes)
void fs_filter_chain_destroy(FS_FilterChain* chain);

// Append a filter node to the chain (takes ownership of node)
bool fs_filter_chain_append(FS_FilterChain* chain, FS_FilterNode* node);

// Parse a CSS filter string into a filter chain
// e.g. "blur(10px) brightness(1.2) sepia(0.5)"
// Returns NULL on parse failure.
FS_FilterChain* fs_filter_chain_parse(const char* filter_string);

// Serialize a filter chain back to a CSS string
// Writes at most buffer_size bytes including the null terminator.
// Returns false if buffer was too small (string is still null-terminated).
bool fs_filter_chain_to_string(const FS_FilterChain* chain, char* buffer, size_t buffer_size);

// Deep-copy a filter chain
FS_FilterChain* fs_filter_chain_clone(const FS_FilterChain* chain);

// Check if a filter string is "none" or empty
bool fs_filter_string_is_none(const char* filter_string);

// Get the head node of a filter chain (for rendering)
const FS_FilterNode* fs_filter_chain_get_head(const FS_FilterChain* chain);

// Get node count
uint32_t fs_filter_chain_get_count(const FS_FilterChain* chain);

// Execute filter chain onto a canvas texture.
// Copies canvas to ping-pong buffer A, applies filters A<->B, copies result back to canvas.
// Returns true on success.
bool fs_filter_chain_execute(
    FS_Core* core,
    WGPUCommandEncoder encoder,
    WGPUTexture target_texture,
    WGPUTextureView target_view,
    const FS_FilterChain* chain
);

#ifdef __cplusplus
}
#endif

#endif // WCN_FULLSTACK_FILTERS_H
