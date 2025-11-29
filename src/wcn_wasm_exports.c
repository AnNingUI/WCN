// ============================================================================
// WCN WebAssembly Exports
// ============================================================================
// This file provides WebAssembly exports for all WCN public API functions.
// When building for WebAssembly with Emscripten, include this file in your
// build to ensure all functions are exported and accessible from JavaScript.
// ============================================================================

#include "WCN/WCN.h"
#include "WCN/WCN_WASM.h"
#include "wcn_internal.h"

#ifdef __EMSCRIPTEN__

// ============================================================================
// Export All WCN Functions
// ============================================================================
// This will mark all WCN functions with EMSCRIPTEN_KEEPALIVE to prevent
// them from being optimized away during linking.
// ============================================================================

WCN_WASM_EXPORT_ALL()

// ============================================================================
// Additional WASM-specific Helper Functions
// ============================================================================

#include <emscripten.h>
#include <stdlib.h>
#include <string.h>

// Helper: Allocate memory in WASM heap (for passing data from JS)
WCN_WASM_EXPORT void* wcn_wasm_malloc(size_t size) {
    return malloc(size);
}

// Helper: Free memory in WASM heap
WCN_WASM_EXPORT void wcn_wasm_free(void* ptr) {
    free(ptr);
}

// Helper: Get WCN version string
WCN_WASM_EXPORT const char* wcn_wasm_get_version(void) {
    return "WCN 0.1.0 (WebAssembly)";
}

// Helper: Create GPU resources structure (for JS interop)
WCN_WASM_EXPORT WCN_GPUResources* wcn_wasm_create_gpu_resources(
    WGPUInstance instance,
    WGPUDevice device,
    WGPUQueue queue,
    WGPUSurface surface
) {
    WCN_GPUResources* resources = (WCN_GPUResources*)malloc(sizeof(WCN_GPUResources));
    if (!resources) {
        return NULL;
    }
    
    resources->instance = instance;
    resources->device = device;
    resources->queue = queue;
    resources->surface = surface;
    
    return resources;
}

// Helper: Create GPU resources using Emscripten's WebGPU device
// This function gets the WebGPU device that Emscripten automatically creates
WCN_WASM_EXPORT WCN_GPUResources* wcn_wasm_create_gpu_resources_auto(void) {
    WCN_GPUResources* resources = (WCN_GPUResources*)malloc(sizeof(WCN_GPUResources));
    if (!resources) {
        return NULL;
    }
    
    // When using Emscripten's WebGPU (-sUSE_WEBGPU=1), 
    // the device is automatically created and managed by Emscripten.
    // We can get it through the preinitializedWebGPUDevice
    // For now, we'll set these to 0 and let the context creation
    // handle getting the device from Emscripten
    resources->instance = 0;
    resources->device = 0;    // Will be obtained from Emscripten
    resources->queue = 0;     // Will be obtained from Emscripten
    resources->surface = 0;
    
    return resources;
}

// Helper: Free GPU resources structure
WCN_WASM_EXPORT void wcn_wasm_free_gpu_resources(WCN_GPUResources* resources) {
    if (resources) {
        free(resources);
    }
}

// Helper: Create ImageData structure (for JS interop)
WCN_WASM_EXPORT WCN_ImageData* wcn_wasm_create_image_data(
    uint32_t width,
    uint32_t height,
    uint32_t format
) {
    WCN_ImageData* image = (WCN_ImageData*)malloc(sizeof(WCN_ImageData));
    if (!image) {
        return NULL;
    }
    
    size_t data_size = width * height * 4; // Assuming RGBA8
    image->data = (uint8_t*)malloc(data_size);
    if (!image->data) {
        free(image);
        return NULL;
    }
    
    image->width = width;
    image->height = height;
    image->format = format;
    image->data_size = data_size;
    
    return image;
}

// Helper: Free ImageData structure
WCN_WASM_EXPORT void wcn_wasm_free_image_data(WCN_ImageData* image) {
    if (image) {
        if (image->data) {
            free(image->data);
        }
        free(image);
    }
}

// Helper: Get ImageData pixel buffer pointer (for JS to write to)
WCN_WASM_EXPORT uint8_t* wcn_wasm_get_image_data_buffer(WCN_ImageData* image) {
    return image ? image->data : NULL;
}

// Helper: Get ImageData buffer size
WCN_WASM_EXPORT size_t wcn_wasm_get_image_data_size(WCN_ImageData* image) {
    return image ? image->data_size : 0;
}

// ============================================================================
// Enum Value Helpers (for JS interop)
// ============================================================================

// Text Align
WCN_WASM_EXPORT int wcn_wasm_text_align_left(void) { return WCN_TEXT_ALIGN_LEFT; }
WCN_WASM_EXPORT int wcn_wasm_text_align_center(void) { return WCN_TEXT_ALIGN_CENTER; }
WCN_WASM_EXPORT int wcn_wasm_text_align_right(void) { return WCN_TEXT_ALIGN_RIGHT; }

// Text Baseline
WCN_WASM_EXPORT int wcn_wasm_text_baseline_top(void) { return WCN_TEXT_BASELINE_TOP; }
WCN_WASM_EXPORT int wcn_wasm_text_baseline_middle(void) { return WCN_TEXT_BASELINE_MIDDLE; }
WCN_WASM_EXPORT int wcn_wasm_text_baseline_bottom(void) { return WCN_TEXT_BASELINE_BOTTOM; }
WCN_WASM_EXPORT int wcn_wasm_text_baseline_alphabetic(void) { return WCN_TEXT_BASELINE_ALPHABETIC; }

// Line Cap
WCN_WASM_EXPORT int wcn_wasm_line_cap_butt(void) { return WCN_LINE_CAP_BUTT; }
WCN_WASM_EXPORT int wcn_wasm_line_cap_round(void) { return WCN_LINE_CAP_ROUND; }
WCN_WASM_EXPORT int wcn_wasm_line_cap_square(void) { return WCN_LINE_CAP_SQUARE; }

// Line Join
WCN_WASM_EXPORT int wcn_wasm_line_join_miter(void) { return WCN_LINE_JOIN_MITER; }
WCN_WASM_EXPORT int wcn_wasm_line_join_round(void) { return WCN_LINE_JOIN_ROUND; }
WCN_WASM_EXPORT int wcn_wasm_line_join_bevel(void) { return WCN_LINE_JOIN_BEVEL; }

// Composite Operation
WCN_WASM_EXPORT int wcn_wasm_composite_source_over(void) { return WCN_COMPOSITE_SOURCE_OVER; }
WCN_WASM_EXPORT int wcn_wasm_composite_source_in(void) { return WCN_COMPOSITE_SOURCE_IN; }
WCN_WASM_EXPORT int wcn_wasm_composite_source_out(void) { return WCN_COMPOSITE_SOURCE_OUT; }
WCN_WASM_EXPORT int wcn_wasm_composite_source_atop(void) { return WCN_COMPOSITE_SOURCE_ATOP; }
WCN_WASM_EXPORT int wcn_wasm_composite_destination_over(void) { return WCN_COMPOSITE_DESTINATION_OVER; }
WCN_WASM_EXPORT int wcn_wasm_composite_destination_in(void) { return WCN_COMPOSITE_DESTINATION_IN; }
WCN_WASM_EXPORT int wcn_wasm_composite_destination_out(void) { return WCN_COMPOSITE_DESTINATION_OUT; }
WCN_WASM_EXPORT int wcn_wasm_composite_destination_atop(void) { return WCN_COMPOSITE_DESTINATION_ATOP; }
WCN_WASM_EXPORT int wcn_wasm_composite_lighter(void) { return WCN_COMPOSITE_LIGHTER; }
WCN_WASM_EXPORT int wcn_wasm_composite_copy(void) { return WCN_COMPOSITE_COPY; }
WCN_WASM_EXPORT int wcn_wasm_composite_xor(void) { return WCN_COMPOSITE_XOR; }

#endif // __EMSCRIPTEN__
