#include "fullstack_filters.h"
#include "fullstack_core_private.h"
#include <math.h>
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define strncasecmp _strnicmp
#else
#include <strings.h>
#endif

#if FS_EFFECTS_ENABLED

// DSUniforms: matches DSUniforms in WGSL drop-shadow shader
typedef struct {
    float color[4];
    float offset[2];
    float _pad[2];
} DSUniforms;

// Internal filter chain representation is in fullstack_filters.h

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static FS_FilterType fs_filter_type_from_name(const char* name, size_t name_len) {
    if (name_len == 0) return FS_FILTER_NONE;
    // Case-insensitive comparison
    if (name_len == 10 && strncasecmp(name, "brightness", 10) == 0) return FS_FILTER_BRIGHTNESS;
    if (name_len == 8 && strncasecmp(name, "contrast", 8) == 0) return FS_FILTER_CONTRAST;
    if (name_len == 9 && strncasecmp(name, "grayscale", 9) == 0) return FS_FILTER_GRAYSCALE;
    if (name_len == 10 && strncasecmp(name, "hue-rotate", 10) == 0) return FS_FILTER_HUE_ROTATE;
    if (name_len == 6 && strncasecmp(name, "invert", 6) == 0) return FS_FILTER_INVERT;
    if (name_len == 7 && strncasecmp(name, "opacity", 7) == 0) return FS_FILTER_OPACITY;
    if (name_len == 8 && strncasecmp(name, "saturate", 8) == 0) return FS_FILTER_SATURATE;
    if (name_len == 5 && strncasecmp(name, "sepia", 5) == 0) return FS_FILTER_SEPIA;
    if (name_len == 4 && strncasecmp(name, "blur", 4) == 0) return FS_FILTER_BLUR;
    if (name_len == 11 && strncasecmp(name, "drop-shadow", 11) == 0) return FS_FILTER_DROP_SHADOW;
    return FS_FILTER_NONE;
}

// Parse a unitless or unit-prefixed float value from a string.
// Advances `ptr` to the end of the consumed number.
// Returns the parsed float value, or NAN on failure.
static float fs_filter_parse_float(const char** ptr) {
    if (!ptr || !*ptr) return NAN;
    const char* p = *ptr;

    // Skip leading whitespace
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;

    // Sign
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    else if (*p == '+') { p++; }

    // Integer part
    float value = 0.0f;
    int has_digits = 0;
    while (*p >= '0' && *p <= '9') {
        value = value * 10.0f + (float)(*p - '0');
        p++;
        has_digits = 1;
    }

    // Fractional part
    if (*p == '.') {
        p++;
        float frac = 0.1f;
        while (*p >= '0' && *p <= '9') {
            value += (float)(*p - '0') * frac;
            frac *= 0.1f;
            p++;
            has_digits = 1;
        }
    }

    // Exponent
    if ((*p == 'e' || *p == 'E') && has_digits) {
        p++;
        int exp_sign = 1;
        if (*p == '-') { exp_sign = -1; p++; }
        else if (*p == '+') { p++; }
        int exp_val = 0;
        while (*p >= '0' && *p <= '9') {
            exp_val = exp_val * 10 + (*p - '0');
            p++;
        }
        value *= powf(10.0f, (float)(exp_sign * exp_val));
    }

    if (!has_digits) return NAN;

    *ptr = p;
    return sign * value;
}

// Parse a positive float (radius/amount value) — rejects negative values.
static float fs_filter_parse_radius(const char** ptr) {
    float v = fs_filter_parse_float(ptr);
    if (v < 0.0f) return 0.0f;
    return v;
}

// Parse a 0..1 float value (percentage or fraction)
static float fs_filter_parse_unit_value(const char** ptr) {
    if (!ptr || !*ptr) return NAN;
    const char* p = *ptr;

    // Skip whitespace
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;

    // Check for trailing %
    float v = fs_filter_parse_float(&p);
    if (isnan(v)) { *ptr = p; return NAN; }

    size_t len = 0;
    const char* check = p;
    while (check[len] == ' ' || check[len] == '\t') len++;
    if (check[len] == '%') {
        v = v / 100.0f;
        p = check + len + 1;
    }

    *ptr = p;
    return v;
}

// Parse a drop-shadow color in hex or rgba format only.
// Returns packed RGBA8, or 0 on failure.
static uint32_t fs_filter_parse_drop_shadow_color(const char** ptr) {
    if (!ptr || !*ptr) return 0;
    const char* p = *ptr;

    // Skip whitespace
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;

    // Hex format: #RRGGBBAA or #RRGGBB
    if (*p == '#') {
        p++;
        size_t hex_len = 0;
        while ((p[hex_len] >= '0' && p[hex_len] <= '9') ||
               (p[hex_len] >= 'A' && p[hex_len] <= 'F') ||
               (p[hex_len] >= 'a' && p[hex_len] <= 'f')) {
            hex_len++;
        }

        if (hex_len != 6 && hex_len != 8) {
            *ptr = p + hex_len;
            return 0;
        }

        unsigned int r = 0, g = 0, b = 0, a = 255;
        sscanf(p, "%2x%2x%2x%2x", &r, &g, &b, &a);
        *ptr = p + hex_len;
        return ((r & 0xFF) << 24) | ((g & 0xFF) << 16) | ((b & 0xFF) << 8) | (a & 0xFF);
    }

    // rgba(r, g, b, a) — limited support per plan requirements
    // Only hex and rgba() formats are accepted; named colors cause parse failure.
    if (strncasecmp(p, "rgba(", 5) == 0) {
        p += 5;
        float rv = fs_filter_parse_float(&p);
        if (isnan(rv) || *p != ',') { *ptr = p; return 0; }
        p++;
        float gv = fs_filter_parse_float(&p);
        if (isnan(gv) || *p != ',') { *ptr = p; return 0; }
        p++;
        float bv = fs_filter_parse_float(&p);
        if (isnan(bv) || *p != ',') { *ptr = p; return 0; }
        p++;
        float av = fs_filter_parse_float(&p);
        if (isnan(av) || *p != ')') { *ptr = p; return 0; }
        p++;
        *ptr = p;
        uint8_t r8 = (uint8_t)(rv < 0 ? 0 : (rv > 255 ? 255 : rv));
        uint8_t g8 = (uint8_t)(gv < 0 ? 0 : (gv > 255 ? 255 : gv));
        uint8_t b8 = (uint8_t)(bv < 0 ? 0 : (bv > 255 ? 255 : bv));
        uint8_t a8 = (uint8_t)(av < 0 ? 0 : (av > 1.0f ? 255 : av * 255.0f));
        return ((uint32_t)r8 << 24) | ((uint32_t)g8 << 16) | ((uint32_t)b8 << 8) | a8;
    }

    // Named colors not supported per plan requirement
    *ptr = p;
    return 0;
}

// Skip to matching closing paren or end of value list
static void fs_filter_skip_to_paren(const char** ptr, int depth) {
    if (!ptr || !*ptr) return;
    const char* p = *ptr;
    while (*p && depth > 0) {
        if (*p == '(') { depth++; p++; }
        else if (*p == ')') { depth--; if (depth > 0) p++; else { p++; break; } }
        else { p++; }
    }
    *ptr = p;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

FS_FilterChain* fs_filter_chain_create(void) {
    FS_FilterChain* chain = (FS_FilterChain*)calloc(1, sizeof(FS_FilterChain));
    return chain;
}

void fs_filter_chain_destroy(FS_FilterChain* chain) {
    if (!chain) return;
    FS_FilterNode* node = chain->head;
    while (node) {
        FS_FilterNode* next = node->next;
        free(node);
        node = next;
    }
    free(chain);
}

bool fs_filter_chain_append(FS_FilterChain* chain, FS_FilterNode* node) {
    if (!chain || !node) return false;
    node->next = NULL;
    if (!chain->head) {
        chain->head = node;
        chain->tail = node;
    } else {
        chain->tail->next = node;
        chain->tail = node;
    }
    chain->count++;
    return true;
}

FS_FilterChain* fs_filter_chain_parse(const char* filter_string) {
    if (!filter_string) return NULL;

    // Skip "none" or empty string
    if (fs_filter_string_is_none(filter_string)) {
        return fs_filter_chain_create();
    }

    FS_FilterChain* chain = fs_filter_chain_create();
    if (!chain) return NULL;

    const char* p = filter_string;
    size_t name_buf_size = 32;
    char* name_buf = (char*)malloc(name_buf_size);
    if (!name_buf) { fs_filter_chain_destroy(chain); return NULL; }

    while (*p) {
        // Skip whitespace and commas between functions
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
        if (!*p) break;

        // Read function name
        const char* name_start = p;
        size_t name_len = 0;
        while ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
               (*p >= '0' && *p <= '9') || *p == '-') {
            if (*p == '-' && name_len == 0) { p++; continue; } // leading hyphen allowed
            name_len++;
            p++;
        }

        // Skip whitespace between name and '('
        while (*p == ' ' || *p == '\t') p++;

        if (*p != '(') {
            // Not a function — skip to next token
            p = name_start + name_len + 1;
            while (*p && *p != ' ' && *p != '\t' && *p != ',') p++;
            continue;
        }
        p++; // skip '('

        FS_FilterType type = fs_filter_type_from_name(name_start, name_len);
        if (type == FS_FILTER_NONE) {
            // Unknown function name — skip to matching ')'
            fs_filter_skip_to_paren(&p, 1);
            continue;
        }

        // Parse parameters based on filter type
        FS_FilterParams params = {{0.0f}};

        switch (type) {
            case FS_FILTER_BRIGHTNESS:
            case FS_FILTER_CONTRAST:
            case FS_FILTER_GRAYSCALE:
            case FS_FILTER_INVERT:
            case FS_FILTER_OPACITY:
            case FS_FILTER_SATURATE:
            case FS_FILTER_SEPIA: {
                float v = fs_filter_parse_unit_value(&p);
                if (isnan(v)) v = 1.0f;
                params.amount = v;
                break;
            }
            case FS_FILTER_HUE_ROTATE: {
                // Degrees (default) or radians
                float v = fs_filter_parse_float(&p);
                if (isnan(v)) v = 0.0f;
                // Check unit suffix
                while (*p == ' ' || *p == '\t') p++;
                if ((p[0] == 'r' || p[0] == 'R') && (p[1] == 'a' || p[1] == 'A')) {
                    // radians — convert to degrees
                    v = v * 180.0f / (float)3.14159265358979;
                    p += 2;
                    while ((p[0] >= 'a' && p[0] <= 'z') || (p[0] >= 'A' && p[0] <= 'Z')) p++;
                } else {
                    // assume degrees (deg suffix allowed)
                    if ((p[0] == 'd' || p[0] == 'D') && (p[1] == 'e' || p[1] == 'E')) {
                        p += 2;
                        while ((p[0] >= 'a' && p[0] <= 'z') || (p[0] >= 'A' && p[0] <= 'Z')) p++;
                    }
                }
                params.degrees = v;
                break;
            }
            case FS_FILTER_BLUR: {
                float v = fs_filter_parse_radius(&p);
                // Consume optional 'px' suffix
                while (*p == ' ' || *p == '\t') p++;
                if (p[0] == 'p' && p[1] == 'x' && (p[2] == ' ' || p[2] == '\t' || p[2] == ',' || p[2] == ')' || p[2] == '\0')) {
                    p += 2;
                }
                params.radius = v;
                break;
            }
            case FS_FILTER_DROP_SHADOW: {
                // drop-shadow(offset-x offset-y [blur-radius] color)
                // offset-x (consume optional 'px' suffix)
                float ox = fs_filter_parse_float(&p);
                if (isnan(ox)) ox = 4.0f;
                if (p[0] == 'p' && p[1] == 'x') p += 2;
                while (*p == ' ' || *p == '\t') p++;
                // offset-y (consume optional 'px' suffix)
                float oy = fs_filter_parse_float(&p);
                if (isnan(oy)) oy = 4.0f;
                if (p[0] == 'p' && p[1] == 'x') p += 2;
                while (*p == ' ' || *p == '\t') p++;
                // Optional blur radius — lookahead parse; if next token
                // starts a color (# or rgba), treat as no blur.
                float br = 0.0f;
                if (*p != '#' && *p != ')' &&
                    !(p[0] == 'r' && p[1] == 'g' && p[2] == 'b')) {
                    const char* save = p;
                    float maybe_br = fs_filter_parse_float(&p);
                    if (!isnan(maybe_br)) {
                        br = maybe_br;
                        if (p[0] == 'p' && p[1] == 'x') p += 2;
                        while (*p == ' ' || *p == '\t') p++;
                    } else {
                        p = save; // rewind — not a number
                    }
                }
                // Color (hex or rgba only)
                uint32_t color = fs_filter_parse_drop_shadow_color(&p);
                if (color == 0) color = 0x00000080u; // default: semi-transparent black
                params.drop_shadow.offset_x = ox;
                params.drop_shadow.offset_y = oy;
                params.drop_shadow.blur_radius = br;
                params.drop_shadow.color = color;
                break;
            }
            default:
                break;
        }

        // Expect ')'
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ')') p++;

        // Create and append node
        FS_FilterNode* node = (FS_FilterNode*)calloc(1, sizeof(FS_FilterNode));
        if (!node) { fs_filter_chain_destroy(chain); free(name_buf); return NULL; }
        node->type = type;
        node->params = params;
        node->next = NULL;
        fs_filter_chain_append(chain, node);
    }

    free(name_buf);
    return chain;
}

bool fs_filter_chain_to_string(const FS_FilterChain* chain, char* buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) return false;
    buffer[0] = '\0';
    if (!chain || !chain->head) return true; // empty chain → empty string

    size_t written = 0;
    const FS_FilterNode* node = chain->head;

    while (node) {
        // Format each filter
        char formatted[256];
        int len = 0;

        switch (node->type) {
            case FS_FILTER_BRIGHTNESS:
                len = snprintf(formatted, sizeof(formatted), "brightness(%.2g)",
                    (double)node->params.amount);
                break;
            case FS_FILTER_CONTRAST:
                len = snprintf(formatted, sizeof(formatted), "contrast(%.2g)",
                    (double)node->params.amount);
                break;
            case FS_FILTER_GRAYSCALE:
                len = snprintf(formatted, sizeof(formatted), "grayscale(%.2g)",
                    (double)node->params.amount);
                break;
            case FS_FILTER_HUE_ROTATE:
                len = snprintf(formatted, sizeof(formatted), "hue-rotate(%.1fdeg)",
                    (double)node->params.degrees);
                break;
            case FS_FILTER_INVERT:
                len = snprintf(formatted, sizeof(formatted), "invert(%.2g)",
                    (double)node->params.amount);
                break;
            case FS_FILTER_OPACITY:
                len = snprintf(formatted, sizeof(formatted), "opacity(%.2g)",
                    (double)node->params.amount);
                break;
            case FS_FILTER_SATURATE:
                len = snprintf(formatted, sizeof(formatted), "saturate(%.2g)",
                    (double)node->params.amount);
                break;
            case FS_FILTER_SEPIA:
                len = snprintf(formatted, sizeof(formatted), "sepia(%.2g)",
                    (double)node->params.amount);
                break;
            case FS_FILTER_BLUR:
                len = snprintf(formatted, sizeof(formatted), "blur(%.2gpx)",
                    (double)node->params.radius);
                break;
            case FS_FILTER_DROP_SHADOW: {
                uint32_t c = node->params.drop_shadow.color;
                len = snprintf(formatted, sizeof(formatted),
                    "drop-shadow(%.2gpx %.2gpx %.2gpx #%02X%02X%02X%02X)",
                    (double)node->params.drop_shadow.offset_x,
                    (double)node->params.drop_shadow.offset_y,
                    (double)node->params.drop_shadow.blur_radius,
                    (c >> 24) & 0xFF, (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF);
                break;
            }
            default:
                len = 0;
                break;
        }

        if (len > 0 && (size_t)len < sizeof(formatted)) {
            if (written + (size_t)len + 1 < buffer_size) {
                if (written > 0) buffer[written++] = ' ';
                memcpy(buffer + written, formatted, (size_t)len);
                written += (size_t)len;
                buffer[written] = '\0';
            } else {
                // Buffer too small
                return false;
            }
        }

        node = node->next;
    }

    return true;
}

FS_FilterChain* fs_filter_chain_clone(const FS_FilterChain* chain) {
    if (!chain) return NULL;
    FS_FilterChain* clone = fs_filter_chain_create();
    if (!clone) return NULL;

    const FS_FilterNode* src = chain->head;
    FS_FilterNode* prev = NULL;

    while (src) {
        FS_FilterNode* copy = (FS_FilterNode*)calloc(1, sizeof(FS_FilterNode));
        if (!copy) {
            fs_filter_chain_destroy(clone);
            return NULL;
        }
        copy->type = src->type;
        copy->params = src->params;
        copy->next = NULL;

        if (!clone->head) {
            clone->head = copy;
            clone->tail = copy;
        } else {
            clone->tail->next = copy;
            clone->tail = copy;
        }
        clone->count++;
        prev = copy;
        src = src->next;
    }

    (void)prev; // unused
    return clone;
}

bool fs_filter_string_is_none(const char* filter_string) {
    if (!filter_string) return true;
    // Skip leading whitespace
    while (*filter_string == ' ' || *filter_string == '\t' ||
           *filter_string == '\n' || *filter_string == '\r') filter_string++;
    if (*filter_string == '\0') return true;
    if (strncasecmp(filter_string, "none", 4) == 0) {
        const char* after = filter_string + 4;
        while (*after == ' ' || *after == '\t') after++;
        if (*after == '\0' || *after == ')') return true;
    }
    return false;
}

const FS_FilterNode* fs_filter_chain_get_head(const FS_FilterChain* chain) {
    return chain ? chain->head : NULL;
}

uint32_t fs_filter_chain_get_count(const FS_FilterChain* chain) {
    return chain ? chain->count : 0;
}

// ---------------------------------------------------------------------------
// Filter chain GPU execution
// ---------------------------------------------------------------------------

bool fs_filter_chain_execute(
    FS_Core* core,
    WGPUCommandEncoder encoder,
    WGPUTexture target_texture,
    WGPUTextureView target_view,
    const FS_FilterChain* chain
) {
    (void)target_view;
    if (!core || !encoder || !target_texture || !chain || !chain->head) {
        return false;
    }

    FS_EffectResources* effects = fs_core_get_effects_resources(core);
    if (!effects || !effects->enabled) {
        return true;  // No effects resources, nothing to do
    }

    // Check required pipelines
    if (!effects->filter_pipeline || !effects->filter_copy_pipeline) {
        return true;  // Filter pipelines not initialized
    }

    // Step 1: Copy scene_texture (RGBA8Unorm) -> ping_pong_texture_a (RGBA8Unorm)
    {
        WGPUTexelCopyTextureInfo src_info = {
            .texture = effects->scene_texture,
            .mipLevel = 0u,
            .origin = {0u, 0u, 0u},
            .aspect = WGPUTextureAspect_All
        };
        WGPUTexelCopyTextureInfo dst_info = {
            .texture = effects->ping_pong_texture_a,
            .mipLevel = 0u,
            .origin = {0u, 0u, 0u},
            .aspect = WGPUTextureAspect_All
        };
        WGPUExtent3D copy_size = {core->width, core->height, 1u};
        wgpuCommandEncoderCopyTextureToTexture(encoder, &src_info, &dst_info, &copy_size);
    }

    // Step 2: Apply filter chain (A <-> B ping-pong)
    const FS_FilterNode* node = chain->head;
    bool write_to_a = false;  // Start: read from A (canvas copy), write to B
    bool last_was_drop_shadow = false;

    while (node) {
        switch (node->type) {
            case FS_FILTER_BLUR: {
                FS_GaussianKernel kernel_blur;
                uint32_t kernel_size_blur;
                fs_gaussian_kernel_compute_for_blur(&kernel_blur, node->params.radius, &kernel_size_blur);

                uint8_t uniform_data_blur[16];
                *(uint32_t*)&uniform_data_blur[0] = kernel_size_blur;
                memset(&uniform_data_blur[4], 0, 12);
                wgpuQueueWriteBuffer(core->queue, effects->gaussian_uniform_buffer, 0, uniform_data_blur, 16);

                uint8_t kernel_data_blur[252];
                memcpy(kernel_data_blur, kernel_blur.weights, kernel_size_blur * sizeof(float));
                wgpuQueueWriteBuffer(core->queue, effects->gaussian_kernel_buffer, 0, kernel_data_blur, kernel_size_blur * sizeof(float));

                // H-pass: source -> opposite
                {
                    WGPUComputePassDescriptor blur_h_desc = {
                        .label = { .data = "FS Filter Blur H", .length = 16 },
                        .timestampWrites = NULL
                    };
                    WGPUComputePassEncoder blur_h_pass = wgpuCommandEncoderBeginComputePass(encoder, &blur_h_desc);
                    wgpuComputePassEncoderSetPipeline(blur_h_pass, effects->gaussian_blur_h_pipeline);
                    wgpuComputePassEncoderSetBindGroup(blur_h_pass, 0, write_to_a ? effects->gaussian_blur_h_bg_b : effects->gaussian_blur_h_bg_a, 0, NULL);
                    const uint32_t blur_dispatch_w = (core->width + 255u) / 256u;
                    wgpuComputePassEncoderDispatchWorkgroups(blur_h_pass, blur_dispatch_w, core->height, 1);
                    wgpuComputePassEncoderEnd(blur_h_pass);
                    wgpuComputePassEncoderRelease(blur_h_pass);
                }

                // V-pass: opposite -> source
                {
                    WGPUComputePassDescriptor blur_v_desc = {
                        .label = { .data = "FS Filter Blur V", .length = 16 },
                        .timestampWrites = NULL
                    };
                    WGPUComputePassEncoder blur_v_pass = wgpuCommandEncoderBeginComputePass(encoder, &blur_v_desc);
                    wgpuComputePassEncoderSetPipeline(blur_v_pass, effects->gaussian_blur_v_pipeline);
                    wgpuComputePassEncoderSetBindGroup(blur_v_pass, 0, write_to_a ? effects->gaussian_blur_v_bg_b : effects->gaussian_blur_v_bg_a, 0, NULL);
                    const uint32_t blur_dispatch_h_v = (core->height + 255u) / 256u;
                    wgpuComputePassEncoderDispatchWorkgroups(blur_v_pass, core->width, blur_dispatch_h_v, 1);
                    wgpuComputePassEncoderEnd(blur_v_pass);
                    wgpuComputePassEncoderRelease(blur_v_pass);
                }
                // After H+V blur, result returns to source buffer: write_to_a=false means result in A.
                // Update the flag so subsequent filters read from A.
                write_to_a = false;
                node = node->next;
                continue;
            }

            case FS_FILTER_DROP_SHADOW: {
                // ALL-COMPUTE drop-shadow:
                // After blur H+V: A = blurred shadow, B = original (scene copy). write_to_a = false.
                // Use drop_shadow_c_pipeline (compute) to read A(blur) + B(original),
                // write to shadow_composite_texture. This avoids WebGPU's COLOR_TARGET vs
                // RESOURCE conflict in render passes.
                FS_ShadowUniforms ds_uni = {
                    .shadow_color = {(float)((node->params.drop_shadow.color >> 24) & 0xFF) / 255.0f,
                                    (float)((node->params.drop_shadow.color >> 16) & 0xFF) / 255.0f,
                                    (float)((node->params.drop_shadow.color >> 8) & 0xFF) / 255.0f,
                                    (float)((node->params.drop_shadow.color >> 0) & 0xFF) / 255.0f},
                    .shadow_offset = {node->params.drop_shadow.offset_x, node->params.drop_shadow.offset_y},
                    ._padding = {0.0f, 0.0f}
                };
                wgpuQueueWriteBuffer(core->queue, effects->shadow_uniform_buffer, 0, &ds_uni, sizeof(ds_uni));

                // Use pre-created compute bind group (binding 0=A blur, 1=B original, 2=composite output, 3=uniform)
                if (effects->drop_shadow_c_bg) {
                    WGPUComputePassDescriptor dsC_cp_desc = {
                        .label = { .data = "FS DS Compute Pass", .length = 17 },
                        .timestampWrites = NULL
                    };
                    WGPUComputePassEncoder dsC_cp_pass = wgpuCommandEncoderBeginComputePass(encoder, &dsC_cp_desc);
                    wgpuComputePassEncoderSetPipeline(dsC_cp_pass, effects->drop_shadow_c_pipeline);
                    wgpuComputePassEncoderSetBindGroup(dsC_cp_pass, 0, effects->drop_shadow_c_bg, 0, NULL);
                    const uint32_t ds_dispatch_w = (core->width + 7u) / 8u;
                    const uint32_t ds_dispatch_h = (core->height + 7u) / 8u;
                    wgpuComputePassEncoderDispatchWorkgroups(dsC_cp_pass, ds_dispatch_w, ds_dispatch_h, 1);
                    wgpuComputePassEncoderEnd(dsC_cp_pass);
                    wgpuComputePassEncoderRelease(dsC_cp_pass);
                }
                // Mark that drop-shadow was the last filter. Result is in shadow_composite_texture.
                // If there are more filters after drop-shadow, copy the result to ping_pong_A
                // so the next filter can read from it (the ping-pong pipeline only reads from A/B).
                if (node->next == NULL) {
                    last_was_drop_shadow = true;
                } else {
                    // Copy shadow_composite_texture -> ping_pong_texture_a for subsequent filters
                    WGPUTexelCopyTextureInfo sc_src = {
                        .texture = effects->shadow_composite_texture,
                        .mipLevel = 0u, .origin = {0u, 0u, 0u}, .aspect = WGPUTextureAspect_All
                    };
                    WGPUTexelCopyTextureInfo sc_dst = {
                        .texture = effects->ping_pong_texture_a,
                        .mipLevel = 0u, .origin = {0u, 0u, 0u}, .aspect = WGPUTextureAspect_All
                    };
                    WGPUExtent3D sc_copy_size = {core->width, core->height, 1u};
                    wgpuCommandEncoderCopyTextureToTexture(encoder, &sc_src, &sc_dst, &sc_copy_size);
                    write_to_a = false; // Result is now in A, matching write_to_a=false
                }
                node = node->next;
                continue;
            }

            default: {
                // Normal filter (brightness, contrast, grayscale, etc.)
                FS_FilterUniforms fu = {
                    .filter_type = node->type,
                    .param1 = 0.0f,
                    .param2 = 0.0f,
                    .param3 = 0.0f,
                    .param4 = 0.0f,
                    ._pad = {0, 0, 0, 0}
                };
                switch (node->type) {
                    case FS_FILTER_BRIGHTNESS: fu.param1 = node->params.amount; break;
                    case FS_FILTER_CONTRAST: fu.param1 = node->params.amount; break;
                    case FS_FILTER_GRAYSCALE: fu.param1 = node->params.amount; break;
                    case FS_FILTER_HUE_ROTATE: fu.param1 = node->params.degrees; break;
                    case FS_FILTER_INVERT: fu.param1 = node->params.amount; break;
                    case FS_FILTER_OPACITY: fu.param1 = node->params.amount; break;
                    case FS_FILTER_SATURATE: fu.param1 = node->params.amount; break;
                    case FS_FILTER_SEPIA: fu.param1 = node->params.amount; break;
                    default: break;
                }
                wgpuQueueWriteBuffer(core->queue, effects->filter_uniform_buffer, 0, &fu, sizeof(fu));

                // Apply filter compute pass (A <-> B ping-pong)
                WGPUComputePassDescriptor filter_desc = {
                    .label = { .data = "FS Filter Pass", .length = 14 },
                    .timestampWrites = NULL
                };
                WGPUComputePassEncoder filter_pass = wgpuCommandEncoderBeginComputePass(encoder, &filter_desc);
                wgpuComputePassEncoderSetPipeline(filter_pass, effects->filter_pipeline);
                wgpuComputePassEncoderSetBindGroup(filter_pass, 0, write_to_a ? effects->filter_bg_b : effects->filter_bg_a, 0, NULL);
                const uint32_t dispatch_w = (core->width + 7u) / 8u;
                const uint32_t dispatch_h = (core->height + 7u) / 8u;
                wgpuComputePassEncoderDispatchWorkgroups(filter_pass, dispatch_w, dispatch_h, 1);
                wgpuComputePassEncoderEnd(filter_pass);
                wgpuComputePassEncoderRelease(filter_pass);
                write_to_a = !write_to_a;
                node = node->next;
                continue;
            }
        }
    }

    // Copy filtered result to target_texture (scene_texture).
    // Result is in ping_pong_A if write_to_a==false, ping_pong_B if write_to_a==true.
    // If last filter was drop-shadow, result is in shadow_composite_texture.
    {
        WGPUTexture result_texture;
        if (last_was_drop_shadow) {
            result_texture = effects->shadow_composite_texture;
        } else if (write_to_a) {
            result_texture = effects->ping_pong_texture_b;
        } else {
            result_texture = effects->ping_pong_texture_a;
        }
        WGPUTexelCopyTextureInfo src_info = {
            .texture = result_texture,
            .mipLevel = 0u,
            .origin = {0u, 0u, 0u},
            .aspect = WGPUTextureAspect_All
        };
        WGPUTexelCopyTextureInfo dst_info = {
            .texture = target_texture,
            .mipLevel = 0u,
            .origin = {0u, 0u, 0u},
            .aspect = WGPUTextureAspect_All
        };
        WGPUExtent3D copy_size = {core->width, core->height, 1u};
        wgpuCommandEncoderCopyTextureToTexture(encoder, &src_info, &dst_info, &copy_size);
    }

    return true;
}

#else // FS_EFFECTS_ENABLED == 0

// ---------------------------------------------------------------------------
// Stub implementations when effects are disabled
// ---------------------------------------------------------------------------

FS_FilterChain* fs_filter_chain_create(void) { return NULL; }
void fs_filter_chain_destroy(FS_FilterChain* chain) { (void)chain; }
bool fs_filter_chain_append(FS_FilterChain* chain, FS_FilterNode* node) {
    (void)chain; (void)node; return false;
}
FS_FilterChain* fs_filter_chain_parse(const char* filter_string) {
    (void)filter_string; return NULL;
}
bool fs_filter_chain_to_string(const FS_FilterChain* chain, char* buffer, size_t buffer_size) {
    (void)chain; (void)buffer; (void)buffer_size; return false;
}
FS_FilterChain* fs_filter_chain_clone(const FS_FilterChain* chain) {
    (void)chain; return NULL;
}
bool fs_filter_string_is_none(const char* filter_string) {
    return filter_string == NULL || *filter_string == '\0';
}
const FS_FilterNode* fs_filter_chain_get_head(const FS_FilterChain* chain) {
    (void)chain; return NULL;
}
uint32_t fs_filter_chain_get_count(const FS_FilterChain* chain) {
    (void)chain; return 0;
}
bool fs_filter_chain_execute(FS_Core* core, WGPUCommandEncoder encoder, WGPUTexture target_texture, WGPUTextureView target_view, const FS_FilterChain* chain) {
    (void)core; (void)encoder; (void)target_texture; (void)target_view; (void)chain; return true;
}

#endif // FS_EFFECTS_ENABLED
