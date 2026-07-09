/**
 * fullstack_clay.h — Clay layout engine → WCN rendering bridge
 *
 * Usage:
 *   1. Call fsclay_init(ctx, measureFn, userData) once at startup
 *   2. Each frame:
 *        Clay_SetLayoutDimensions(...);
 *        // ... declare UI with CLAY() ...
 *        Clay_RenderCommandArray arr = Clay_EndLayout();
 *        fsclay_render_commands(core, &arr);
 *   3. Call fsclay_shutdown() at shutdown
 *
 * Color format: Clay_Color {r,g,b,a} where each channel is 0–255 float.
 * WCN uses ABGR packed uint32: 0xAABBGGRR
 */
#ifndef WCN_FULLSTACK_CLAY_H
#define WCN_FULLSTACK_CLAY_H

/* Define this before including clay.h so the implementation is compiled exactly once.
   All other translation units (fullstack_core.c, etc.) that include fullstack_clay.h
   will only see the declarations. */
#define CLAY_IMPLEMENTATION

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "clay.h"
#include "../include/fullstack_core.h"

/* ─────────────────────────────────────────────────────────────────────────
   Wrapper: traced versions of Clay functions to help debug hangs
   ───────────────────────────────────────────────────────────────────────── */

/* Declared here so the demo can call fsclay_BeginLayout() instead of Clay_BeginLayout() */
extern void Clay_BeginLayout(void); /* actual impl from clay.h */
static void fsclay_BeginLayout(void) {
    /* We can't easily wrap Clay_BeginLayout without modifying the source,
       so instead we rely on the demo to add fprintf around the call.
       These stubs exist for future debugging needs. */
    Clay_BeginLayout();
}

/* ─────────────────────────────────────────────────────────────────────────
   Helpers: Clay_Color → WCN ABGR uint32
   ───────────────────────────────────────────────────────────────────────── */

static inline uint32_t fsclay_color(Clay_Color c) {
    uint8_t r = (uint8_t)(c.r > 255.0f ? 255 : (c.r < 0.0f ? 0 : c.r));
    uint8_t g = (uint8_t)(c.g > 255.0f ? 255 : (c.g < 0.0f ? 0 : c.g));
    uint8_t b = (uint8_t)(c.b > 255.0f ? 255 : (c.b < 0.0f ? 0 : c.b));
    uint8_t a = (uint8_t)(c.a > 255.0f ? 255 : (c.a < 0.0f ? 0 : c.a));
    return ((uint32_t)a << 24u) | ((uint32_t)b << 16u) | ((uint32_t)g << 8u) | (uint32_t)r;
}

static inline uint32_t fsclay_color255(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t)a << 24u) | ((uint32_t)b << 16u) | ((uint32_t)g << 8u) | (uint32_t)r;
}

/* Clamp radius to avoid pathological corner values */
static inline float fsclay_radius(float v) {
    return (v < 0.0f) ? 0.0f : v;
}

/* Max of 4 corner radii → approximate radius for fs_cmd_rect */
static inline float fsclay_corner_max(Clay_CornerRadius cr) {
    float m = cr.topLeft;
    if (cr.topRight > m) m = cr.topRight;
    if (cr.bottomLeft > m) m = cr.bottomLeft;
    if (cr.bottomRight > m) m = cr.bottomRight;
    return fsclay_radius(m);
}

/* ─────────────────────────────────────────────────────────────────────────
   Internal render context
   ───────────────────────────────────────────────────────────────────────── */

typedef struct FSCLay_Context {
    /* Measure text function and userdata passed to Clay_SetMeasureTextFunction */
    Clay_Dimensions (*measureTextFn)(Clay_StringSlice text, Clay_TextElementConfig* config, void* userData);
    void* measureTextUserData;

    /* External clip state */
    int clip_depth;

    /* Cached pointer to the core (set on each render call) */
    FS_Core* core;

    /* Layout-to-screen transform: offset added to all Clay coordinates */
    float offset_x;
    float offset_y;
    float scale;

} FSCLay_Context;

static FSCLay_Context g_fsclay = {
    .measureTextFn = NULL,
    .measureTextUserData = NULL,
    .clip_depth = 0,
    .core = NULL,
    .offset_x = 0.0f,
    .offset_y = 0.0f,
    .scale = 1.0f,
};

/* ─────────────────────────────────────────────────────────────────────────
   Public init / shutdown
   ───────────────────────────────────────────────────────────────────────── */

/**
 * Initialize the Clay+WCN bridge.
 * - measureFn: your text measurement function (required for CLAY_TEXT)
 * - userData: opaque pointer passed to measureFn
 *
 * NOTE: Must also call Clay_SetMaxElementCount() and
 *       Clay_SetMaxRenderCommands() before first use.
 */
static void fsclay_init(
    Clay_Dimensions (*measureFn)(Clay_StringSlice, Clay_TextElementConfig*, void*),
    void* userData
) {
    g_fsclay.measureTextFn = measureFn;
    g_fsclay.measureTextUserData = userData;
    g_fsclay.clip_depth = 0;
    g_fsclay.core = NULL;
    g_fsclay.offset_x = 0.0f;
    g_fsclay.offset_y = 0.0f;
    g_fsclay.scale = 1.0f;

    if (measureFn) {
        Clay_SetMeasureTextFunction(measureFn, userData);
    }
}

static void fsclay_shutdown(void) {
    g_fsclay.measureTextFn = NULL;
    g_fsclay.measureTextUserData = NULL;
    g_fsclay.clip_depth = 0;
}

/**
 * Set the layout-to-screen transform.
 * All Clay layout coordinates (x, y) get translated by (ox, oy) and scaled by s.
 */
static void fsclay_set_transform(float ox, float oy, float s) {
    g_fsclay.offset_x = ox;
    g_fsclay.offset_y = oy;
    g_fsclay.scale = s;
}

/* ─────────────────────────────────────────────────────────────────────────
   Internal helpers
   ───────────────────────────────────────────────────────────────────────── */

static inline float fsclay_x(float clay_x) {
    return g_fsclay.offset_x + clay_x * g_fsclay.scale;
}

static inline float fsclay_y(float clay_y) {
    return g_fsclay.offset_y + clay_y * g_fsclay.scale;
}

static inline float fsclay_w(float clay_w) {
    return clay_w * g_fsclay.scale;
}

static inline float fsclay_h(float clay_h) {
    return clay_h * g_fsclay.scale;
}

static inline float fsclay_s(float v) {
    return v * g_fsclay.scale;
}

/* Make a null-terminated copy of a Clay_StringSlice (up to max_len) */
static char* fsclay_strdup_slice(Clay_StringSlice slice, size_t max_len) {
    size_t len = (size_t)slice.length;
    if (len > max_len) len = max_len;
    char* buf = (char*)malloc(len + 1);
    if (!buf) return NULL;
    memcpy(buf, slice.chars, len);
    buf[len] = '\0';
    return buf;
}

/* ─────────────────────────────────────────────────────────────────────────
   Core render dispatcher
   ───────────────────────────────────────────────────────────────────────── */

/**
 * Render a complete Clay_RenderCommandArray using WCN.
 * Returns the number of commands rendered.
 */
static uint32_t fsclay_render_commands(FS_Core* core, const Clay_RenderCommandArray* arr) {
    if (!core || !arr) return 0;

    g_fsclay.core = core;
    uint32_t count = 0;
    const int32_t length = arr->length;

    for (int32_t i = 0; i < length; i++) {
        /* Clay_RenderCommandArray_Get takes non-const pointer */
        Clay_RenderCommand* cmd = Clay_RenderCommandArray_Get((Clay_RenderCommandArray*)arr, i);
        if (!cmd) continue;

        float bx = fsclay_x(cmd->boundingBox.x);
        float by = fsclay_y(cmd->boundingBox.y);
        float bw = fsclay_w(cmd->boundingBox.width);
        float bh = fsclay_h(cmd->boundingBox.height);

        switch (cmd->commandType) {

            case CLAY_RENDER_COMMAND_TYPE_RECTANGLE: {
                Clay_RectangleRenderData* d = &cmd->renderData.rectangle;
                float radius = fsclay_s(fsclay_corner_max(d->cornerRadius));
                uint32_t color = fsclay_color(d->backgroundColor);
                /* Skip transparent rectangles */
                if ((color >> 24) == 0) break;
                fs_cmd_rect(core, bx, by, bw, bh, radius, color);
                count++;
                break;
            }

            case CLAY_RENDER_COMMAND_TYPE_BORDER: {
                Clay_BorderRenderData* d = &cmd->renderData.border;
                Clay_BorderWidth* w = &d->width;
                float radius = fsclay_s(fsclay_corner_max(d->cornerRadius));
                uint32_t color = fsclay_color(d->color);
                if ((color >> 24) == 0) break;
                /* Draw each border side as a separate stroked rect if width > 0 */
                /* Note: WCN's rect_stroke draws a full rounded rect outline.
                   We approximate by drawing the full border, trusting that overlapping
                   borders on adjacent edges from sibling elements look acceptable. */
                float sw = fsclay_s(fmaxf(fmaxf((float)w->left, (float)w->right),
                                          fmaxf((float)w->top, (float)w->bottom)));
                if (sw > 0.0f) {
                    fs_cmd_rect_stroke(core, bx, by, bw, bh, radius, sw, color);
                }
                count++;
                break;
            }

            case CLAY_RENDER_COMMAND_TYPE_TEXT: {
                Clay_TextRenderData* d = &cmd->renderData.text;
                /* Skip empty strings */
                if (!d->stringContents.chars || d->stringContents.length <= 0) break;
                /* Null-terminate for WCN (we know length) */
                char* text = fsclay_strdup_slice(d->stringContents, 4096);
                if (!text) break;
                uint32_t color = fsclay_color(d->textColor);
                /* textColor.a == 0 means invisible, skip */
                if ((color >> 24) == 0) {
                    free(text);
                    break;
                }
                float font_size = fsclay_s((float)d->fontSize);
                /* WCN text draws from baseline, Clay text uses top-left corner.
                   Adjust baseline: font ascent ≈ 0.8 * fontSize */
                float baseline_y = by + font_size * 0.85f;
                fs_cmd_text_utf8(core, bx, baseline_y, font_size, text, color, bw);
                free(text);
                count++;
                break;
            }

            case CLAY_RENDER_COMMAND_TYPE_IMAGE: {
                Clay_ImageRenderData* d = &cmd->renderData.image;
                float radius = fsclay_s(fsclay_corner_max(d->cornerRadius));
                Clay_Color tint_color = d->backgroundColor;
                if (tint_color.a <= 0.0f) {
                    tint_color = (Clay_Color){255.0f, 255.0f, 255.0f, 255.0f};
                }
                uint32_t tint = fsclay_color(tint_color);
                /* If imageData is an FS_ImageHandle*, draw it */
                if (d->imageData) {
                    fs_cmd_image_handle(core, bx, by, bw, bh,
                                       (const FS_ImageHandle*)d->imageData, tint);
                }
                count++;
                break;
            }

            case CLAY_RENDER_COMMAND_TYPE_SCISSOR_START: {
                /* Save WCN state and set up clip */
                fs_state_save(core);
                g_fsclay.clip_depth++;

                /* Build a clip rect using a closed path */
                fs_path_begin(core);
                fs_path_move_to(core, bx, by);
                fs_path_line_to(core, bx + bw, by);
                fs_path_line_to(core, bx + bw, by + bh);
                fs_path_line_to(core, bx, by + bh);
                fs_path_close(core);
                fs_clip_path(core);
                count++;
                break;
            }

            case CLAY_RENDER_COMMAND_TYPE_SCISSOR_END: {
                if (g_fsclay.clip_depth > 0) {
                    fs_state_restore(core);
                    g_fsclay.clip_depth--;
                }
                count++;
                break;
            }

            case CLAY_RENDER_COMMAND_TYPE_OVERLAY_COLOR_START: {
                /* For overlay, we paint a translucent rect on top of children.
                   Since Clay generates RECTANGLE commands separately for the overlay,
                   we handle the start by saving state and painting the overlay color. */
                Clay_OverlayColorRenderData* d = &cmd->renderData.overlayColor;
                uint32_t color = fsclay_color(d->color);
                if ((color >> 24) > 0) {
                    /* Composite: paint the overlay rect with the given alpha */
                    uint32_t fill_color = color | 0xFF000000u; /* ensure full alpha for fill */
                    /* Use a lighter blend by reducing alpha */
                    uint8_t a = (uint8_t)((color >> 24) & 0xFF);
                    fill_color = (color & 0x00FFFFFFu) | ((uint32_t)(a / 2) << 24);
                    fs_cmd_rect(core, bx, by, bw, bh, 0.0f, fill_color);
                }
                fs_state_save(core);
                g_fsclay.clip_depth++;
                count++;
                break;
            }

            case CLAY_RENDER_COMMAND_TYPE_OVERLAY_COLOR_END: {
                if (g_fsclay.clip_depth > 0) {
                    fs_state_restore(core);
                    g_fsclay.clip_depth--;
                }
                count++;
                break;
            }

            case CLAY_RENDER_COMMAND_TYPE_CUSTOM: {
                /* Custom data is passed through. The user can extend this switch
                   or use userData on elements. We skip for now. */
                count++;
                break;
            }

            case CLAY_RENDER_COMMAND_TYPE_NONE:
            default:
                break;
        }
    }

    /* Restore any dangling clip states (shouldn't happen in well-formed output) */
    while (g_fsclay.clip_depth > 0) {
        fs_state_restore(core);
        g_fsclay.clip_depth--;
    }

    return count;
}

#endif /* WCN_FULLSTACK_CLAY_H */
