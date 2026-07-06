/**
 * @file fs_text_layout_obstacle_wrap_demo.c
 * @brief Pretext-style obstacle-aware text layout demo.
 *
 * Controls:
 *   Drag          — move the rectangle obstacle
 *   Click        — rotate by 30 degrees (future: alpha image)
 *   D            — toggle debug overlay
 *   1            — return to overview demo
 *
 * Demonstrates:
 *   - prepare-once, relayout-every-frame text flow
 *   - scanline slot-based text fragmentation
 *   - multi-slot per-baseline layout via fs_text_layout flow APIs
 *   - FS_LayoutFlowCursor, FS_LayoutSlot, FS_LayoutFragment
 *
 * Stage 1: static rectangle obstacle (no image yet)
 * Stage 2: PNG alpha image obstacle with drag + rotate
 * Stage 3: multi-slot continuation + polish
 */

#define FS_TEXT_LAYOUT_IMPLEMENTATION
#include "fs_text_layout.h"
#include "../impl/fullstack_glfw_backend.h"
#include "../impl/fullstack_freetype2_font_backend.h"
#include "../include/fullstack_core_debug.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ================================================================
   CONFIG
   ================================================================ */

#define ARRAY_COUNT(x) (sizeof(x) / sizeof((x)[0]))
#define PI 3.14159265358979323846f

#define DEMO_DW 1280.0f
#define DEMO_DH 720.0f

enum { MAX_FRAGS = 512 };
enum { MAX_SLOTS = 8 };

/* ================================================================
   COLOR HELPERS — ABGR uint32
   ================================================================ */

static uint32_t rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t)(a) << 24u) | ((uint32_t)(b) << 16u) |
           ((uint32_t)(g) << 8u)  | ((uint32_t)(r));
}

static uint32_t lerp_color(uint32_t a, uint32_t b, float t) {
    float u = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
    uint8_t ar = (uint8_t)(a & 0xFF);
    uint8_t ag = (uint8_t)((a >> 8) & 0xFF);
    uint8_t ab = (uint8_t)((a >> 16) & 0xFF);
    uint8_t aa = (uint8_t)((a >> 24) & 0xFF);
    uint8_t br = (uint8_t)(b & 0xFF);
    uint8_t bg = (uint8_t)((b >> 8) & 0xFF);
    uint8_t bb = (uint8_t)((b >> 16) & 0xFF);
    uint8_t ba = (uint8_t)((b >> 24) & 0xFF);
    return rgba8(
        (uint8_t)(ar + (br - ar) * u),
        (uint8_t)(ag + (bg - ag) * u),
        (uint8_t)(ab + (bb - ab) * u),
        (uint8_t)(aa + (ba - aa) * u)
    );
}

/* ================================================================
   PALETTE
   ================================================================ */

#define RGBA8(r, g, b, a) ((((uint32_t)(a)) << 24u) | (((uint32_t)(b)) << 16u) | (((uint32_t)(g)) << 8u) | ((uint32_t)(r)))

static const uint32_t PAL_BG       = RGBA8(0x0D, 0x11, 0x17, 0xFF);
static const uint32_t PAL_ACCENT1  = RGBA8(0x58, 0xA6, 0xFF, 0xFF);
static const uint32_t PAL_ACCENT2  = RGBA8(0x79, 0xC0, 0xFF, 0xFF);
static const uint32_t PAL_TEXT     = RGBA8(0xC9, 0xD1, 0xD9, 0xFF);
static const uint32_t PAL_TEXT_DIM = RGBA8(0x8B, 0x94, 0x9E, 0xFF);
static const uint32_t PAL_OBSTACLE = RGBA8(0xFF, 0x7B, 0x72, 0xFF);
static const uint32_t PAL_GRID     = RGBA8(0x30, 0x36, 0x3D, 0xFF);
static const uint32_t PAL_DEBUG    = RGBA8(0xFF, 0xA6, 0x57, 0xFF);

/* ================================================================
   BODY TEXT — a meaningful multi-paragraph article
   ================================================================ */

static const char* k_body_text =
    "The web renders text through a pipeline that was designed thirty years ago "
    "for static documents. A browser loads a font, shapes the text into glyphs, "
    "measures their combined width, determines where lines break, and positions "
    "each line vertically. Every step depends on the previous one.\n\n"

    "For a paragraph in a blog post, this pipeline is invisible. The browser "
    "loads, lays out, and paints before the reader's eye has traveled from the "
    "address bar to the first word. But the web is no longer a collection of "
    "static documents. It is a platform for applications, and those applications "
    "need to know about text in ways the original pipeline never anticipated.\n\n"

    "A messaging application needs to know the exact height of every message "
    "bubble before rendering a virtualized list. A masonry layout needs the "
    "height of every card to position them without overlap. An editorial page "
    "needs text to flow around images, advertisements, and interactive elements. "
    "Every one of these operations requires text measurement. And every text "
    "measurement on the web today requires a synchronous layout reflow.\n\n"

    "What if text measurement did not require the DOM at all? What if you could "
    "compute exactly where every line of text would break, exactly how wide each "
    "line would be, and exactly how tall the entire text block would be, using "
    "nothing but arithmetic? This is the core insight of pretext. The browser's "
    "canvas API includes a measureText method that returns the width of any string "
    "in any font without triggering a layout reflow.\n\n"

    "Pretext exploits this asymmetry. When text first appears, pretext measures "
    "every word once via canvas and caches the widths. After this preparation "
    "phase, layout is pure arithmetic: walk the cached widths, track the running "
    "line width, insert line breaks when the width exceeds the maximum, and sum "
    "the line heights. No DOM. No reflow. No layout tree access. The performance "
    "improvement is not incremental — it is categorical. Zero reflows. Zero layout "
    "tree traversals. And the text flows around obstacles in real time.";

/* ================================================================
   DEMO STATE
   ================================================================ */

typedef struct {
    float          dw, dh;
    float          scale;
    float          fbw, fbh;
    float          total_time;
    float          frame_dt;
    uint64_t       frame_count;
    float          fps;
    bool           show_debug;

    /* Obstacle */
    float          obs_x, obs_y;     /* center in page space */
    float          obs_w, obs_h;     /* display size */
    float          obs_angle;        /* current angle (radians) */
    float          obs_target_angle;
    bool           obs_dragging;
    float          obs_drag_off_x, obs_drag_off_y;
    bool           obs_hovered;
    bool           obs_loaded;       /* Stage 2: image loaded */

    /* Body layout */
    FS_TextLayout*     layout_ctx;
    FS_PreparedText*   body_prep;
    FS_LayoutFlowCursor cursor;
    FS_LayoutFragment  frags[MAX_FRAGS];
    uint32_t           frag_count;

    /* Layout params */
    float          body_x, body_y, body_w, body_h;
    float          font_size;
    float          line_height;

    /* Mode */
    bool           single_slot_mode;

    /* Font */
    bool           font_ready;
} DemoState;

static DemoState g_state;

/* ================================================================
   COORDINATE HELPERS
   ================================================================ */

/* Design-space visible extent (accounts for letterboxing under uniform scale) */
static float ds_width(void)  { return (g_state.scale > 0.0f) ? g_state.fbw / g_state.scale : DEMO_DW; }
static float ds_height(void) { return (g_state.scale > 0.0f) ? g_state.fbh / g_state.scale : DEMO_DH; }

static void rotated_rect_corners(float cx, float cy, float hw, float hh, float angle,
                                 float out_x[4], float out_y[4]);
static uint32_t clip_poly_y_min(const float* in_x, const float* in_y, uint32_t in_count, float y_min,
                                float* out_x, float* out_y);
static uint32_t clip_poly_y_max(const float* in_x, const float* in_y, uint32_t in_count, float y_max,
                                float* out_x, float* out_y);

static void cursor_window_to_design(FS_GlfwBackend* backend, double win_x, double win_y,
                                    float* out_x, float* out_y) {
    float design_x = (float)win_x;
    float design_y = (float)win_y;

    if (backend && backend->window) {
        int win_w = 0;
        int win_h = 0;
        glfwGetWindowSize(backend->window, &win_w, &win_h);

        float fb_x = (float)win_x;
        float fb_y = (float)win_y;
        if (win_w > 0 && backend->width > 0) {
            fb_x = (float)win_x * ((float)backend->width / (float)win_w);
        }
        if (win_h > 0 && backend->height > 0) {
            fb_y = (float)win_y * ((float)backend->height / (float)win_h);
        }

        if (g_state.scale > 0.0f) {
            design_x = fb_x / g_state.scale;
            design_y = fb_y / g_state.scale;
        } else {
            design_x = fb_x;
            design_y = fb_y;
        }
    }

    if (out_x) *out_x = design_x;
    if (out_y) *out_y = design_y;
}

/* ================================================================
   STAGE 1/2: RECTANGLE SLOTS — compute 0/1/2 available slots per band
   ================================================================ */

/**
 * Compute the available text slots for one line band.
 *
 * - no intersection: one full-width slot
 * - intersects obstacle: up to two slots (left + right)
 * - fully blocked: zero slots
 */
static uint32_t compute_rect_slots(float band_top, float band_bot,
                                   float body_x, float body_w,
                                   FS_LayoutSlot* out_slots,
                                   uint32_t max_slots) {
    if (!out_slots || max_slots == 0) return 0;

    float body_x0 = body_x;
    float body_x1 = body_x + body_w;

    float cx = g_state.obs_x;
    float cy = g_state.obs_y;
    float hw = g_state.obs_w * 0.5f;
    float hh = g_state.obs_h * 0.5f;

    /* Build rotated rectangle polygon in design space. */
    float poly0_x[8] = {0}, poly0_y[8] = {0};
    float poly1_x[8] = {0}, poly1_y[8] = {0};
    float poly2_x[8] = {0}, poly2_y[8] = {0};
    rotated_rect_corners(cx, cy, hw, hh, g_state.obs_angle, poly0_x, poly0_y);

    /* Clip rectangle polygon against the horizontal band [band_top, band_bot]. */
    uint32_t count1 = clip_poly_y_min(poly0_x, poly0_y, 4u, band_top, poly1_x, poly1_y);
    uint32_t count2 = clip_poly_y_max(poly1_x, poly1_y, count1, band_bot, poly2_x, poly2_y);

    /* No overlap with band => one full slot. */
    if (count2 == 0) {
        out_slots[0].x0 = body_x0;
        out_slots[0].x1 = body_x1;
        return 1;
    }

    /* Find x-extent of obstacle inside this band. */
    float occ_x0 = poly2_x[0];
    float occ_x1 = poly2_x[0];
    for (uint32_t i = 1; i < count2; i++) {
        if (poly2_x[i] < occ_x0) occ_x0 = poly2_x[i];
        if (poly2_x[i] > occ_x1) occ_x1 = poly2_x[i];
    }

    /* Clamp occupied interval to body. */
    if (occ_x0 < body_x0) occ_x0 = body_x0;
    if (occ_x1 > body_x1) occ_x1 = body_x1;

    /* Degenerate overlap => one full slot. */
    if (occ_x1 <= occ_x0) {
        out_slots[0].x0 = body_x0;
        out_slots[0].x1 = body_x1;
        return 1;
    }

    uint32_t count = 0;
    if (occ_x0 > body_x0 && count < max_slots) {
        out_slots[count].x0 = body_x0;
        out_slots[count].x1 = occ_x0;
        if (out_slots[count].x1 > out_slots[count].x0) count++;
    }
    if (occ_x1 < body_x1 && count < max_slots) {
        out_slots[count].x0 = occ_x1;
        out_slots[count].x1 = body_x1;
        if (out_slots[count].x1 > out_slots[count].x0) count++;
    }

    return count;
}

/**
 * Compute final per-line slots with visual clip band, asymmetric gutter,
 * and optional single-slot filtering.  Single source of truth for both
 * demo_relayout() and demo_render_debug().
 */
static uint32_t compute_line_slots(float baseline,
                                   float font_sz,
                                   float body_x,
                                   float body_w,
                                   bool  single_slot_mode,
                                   FS_LayoutSlot* out_slots,
                                   uint32_t max_slots) {
    /* Visual clip band — tighter than full line height */
    float clip_top = baseline - font_sz * 0.8f;
    float clip_bot = baseline + font_sz * 0.2f;

    /* Asymmetric gutters (font-relative) */
    float left_gutter  = font_sz * 0.3f;
    float right_gutter = font_sz * 0.8f;

    /* Raw obstacle intersection against visual band */
    FS_LayoutSlot raw[MAX_SLOTS];
    uint32_t raw_count = compute_rect_slots(clip_top, clip_bot,
                                            body_x, body_w,
                                            raw, MAX_SLOTS);

    /* If no intersection (1 full-width slot) or zero, pass through */
    if (raw_count <= 1) {
        uint32_t n = (raw_count <= max_slots) ? raw_count : max_slots;
        for (uint32_t i = 0; i < n; i++) out_slots[i] = raw[i];
        return n;
    }

    /* Apply asymmetric gutter to the two-slot case:
     *   raw[0] = left slot  [body_x0 .. occ_x0]
     *   raw[1] = right slot [occ_x1 .. body_x1]
     * Shrink left slot from right by left_gutter,
     * shrink right slot from left by right_gutter. */
    FS_LayoutSlot adjusted[MAX_SLOTS];
    uint32_t adj_count = 0;

    for (uint32_t i = 0; i < raw_count && adj_count < MAX_SLOTS; i++) {
        FS_LayoutSlot s = raw[i];
        if (i == 0) {
            /* left slot: trim right edge inward */
            s.x1 -= left_gutter;
        } else {
            /* right slot: trim left edge inward */
            s.x0 += right_gutter;
        }
        /* drop degenerate */
        if (s.x1 > s.x0) {
            adjusted[adj_count++] = s;
        }
    }

    /* Single-slot filtering: keep only the wider slot */
    if (single_slot_mode && adj_count > 1) {
        uint32_t best = 0;
        float best_w = adjusted[0].x1 - adjusted[0].x0;
        for (uint32_t i = 1; i < adj_count; i++) {
            float w = adjusted[i].x1 - adjusted[i].x0;
            if (w > best_w) { best = i; best_w = w; }
        }
        adjusted[0] = adjusted[best];
        adj_count = 1;
    }

    uint32_t n = (adj_count <= max_slots) ? adj_count : max_slots;
    for (uint32_t i = 0; i < n; i++) out_slots[i] = adjusted[i];
    return n;
}

/* ================================================================
   RELAYOUT — rebuild fragment buffer from flow cursor
   ================================================================ */

static void demo_relayout(void) {
    float body_x  = g_state.body_x;
    float body_y  = g_state.body_y;
    float body_w  = g_state.body_w;
    float body_h  = g_state.body_h;
    float font_sz = g_state.font_size;
    float lh      = g_state.line_height;
    float baseline = body_y + font_sz * 0.85f;

    fs_text_layout_flow_cursor_init(&g_state.cursor);
    g_state.frag_count = 0;

    while (g_state.frag_count < MAX_FRAGS) {
        if (g_state.cursor.finished) break;
        if (baseline + font_sz * 0.2f > body_y + body_h) break;

        FS_LayoutSlot slots[MAX_SLOTS];
        uint32_t slot_count = compute_line_slots(
            baseline, font_sz, body_x, body_w,
            g_state.single_slot_mode, slots, MAX_SLOTS);

        if (slot_count > 0) {
            uint32_t emitted = fs_text_layout_layout_line_slots(
                g_state.body_prep,
                &g_state.cursor,
                slots,
                slot_count,
                baseline,
                g_state.frags + g_state.frag_count,
                MAX_FRAGS - g_state.frag_count);

            for (uint32_t i = 0; i < emitted; i++) {
                g_state.frags[g_state.frag_count + i].slot_index = i;
            }
            g_state.frag_count += emitted;
        }

        baseline += lh;
    }
}

/* ================================================================
   FRAGMENT RENDERING
   ================================================================ */

static void draw_fragment_text(FS_Core* core, const FS_LayoutFragment* f, const char* utf8) {
    if (!core || !f || !utf8 || !utf8[0]) return;

    float slot_w = f->x1 - f->x0;
    if (slot_w <= 0.0f) return;

    fs_cmd_text_utf8(core,
                     f->x0,
                     f->baseline_y,
                     g_state.font_size,
                     utf8,
                     PAL_TEXT,
                     slot_w);
}

static void demo_render_frags(FS_Core* core) {
    char tmp[2048];

    for (uint32_t i = 0; i < g_state.frag_count; i++) {
        FS_LayoutFragment* f = &g_state.frags[i];
        if (f->byte_len == 0) continue;

        size_t n = f->byte_len;
        if (n >= sizeof(tmp) - 1) n = sizeof(tmp) - 1;
        memcpy(tmp, f->text_start, n);
        tmp[n] = '\0';

        draw_fragment_text(core, f, tmp);
    }
}

/* ================================================================
   DEBUG OVERLAY — draw slot and fragment boundaries
   ================================================================ */

static void demo_render_debug(FS_Core* core) {
    char tmp[128];

    /* Obstacle AABB */
    float obs_cx = g_state.obs_x;
    float obs_cy = g_state.obs_y;
    float obs_hw = g_state.obs_w * 0.5f;
    float obs_hh = g_state.obs_h * 0.5f;

    fs_cmd_rect_stroke(core,
                       obs_cx - obs_hw, obs_cy - obs_hh,
                       g_state.obs_w, g_state.obs_h,
                       0.0f, 2.0f, PAL_DEBUG);

    /* Visualize line-band slots — using same helper as relayout */
    float font_sz = g_state.font_size;
    float lh = g_state.line_height;
    float baseline = g_state.body_y + font_sz * 0.85f;
    for (uint32_t row = 0; row < 128 && baseline + font_sz * 0.2f <= g_state.body_y + g_state.body_h; row++) {
        FS_LayoutSlot slots[MAX_SLOTS];
        uint32_t slot_count = compute_line_slots(baseline, font_sz,
                                                 g_state.body_x, g_state.body_w,
                                                 g_state.single_slot_mode,
                                                 slots, MAX_SLOTS);
        float band_top = baseline - font_sz * 0.8f;
        for (uint32_t i = 0; i < slot_count; i++) {
            fs_cmd_rect_stroke(core,
                               slots[i].x0, band_top,
                               slots[i].x1 - slots[i].x0,
                               font_sz * 1.0f,
                               0.0f, 1.0f,
                               RGBA8(0x58, 0xA6, 0xFF, 0x44));
        }
        baseline += lh;
    }

    /* Per-fragment bounding boxes */
    for (uint32_t i = 0; i < g_state.frag_count; i++) {
        FS_LayoutFragment* f = &g_state.frags[i];
        float fh = g_state.line_height * 0.9f;

        uint32_t fc = lerp_color(PAL_DEBUG, PAL_ACCENT1,
                                  (float)i / (float)(g_state.frag_count + 1));
        fs_cmd_rect_stroke(core, f->x0, f->baseline_y - g_state.font_size * 0.85f,
                           f->x1 - f->x0, fh, 1.0f, 1.0f, fc);
    }

    snprintf(tmp, sizeof(tmp), "fragments: %u  |  angle: %.0f  |  %s  |  D: debug  S: slot mode",
             g_state.frag_count, g_state.obs_angle * 180.0f / PI,
             g_state.single_slot_mode ? "SINGLE-SLOT" : "MULTI-SLOT");
    fs_cmd_text_utf8(core, 20.0f, ds_height() - 20.0f,
                     10.0f, tmp, PAL_TEXT_DIM, 400.0f);
}

static void draw_rotated_obstacle(FS_Core* core, float cx, float cy, float w, float h, float angle, uint32_t fill, uint32_t stroke) {
    fs_state_save(core);
    fs_translate(core, cx, cy);
    fs_rotate(core, angle);
    fs_cmd_rect(core, -w * 0.5f, -h * 0.5f, w, h, 4.0f, fill);
    fs_cmd_rect_stroke(core, -w * 0.5f, -h * 0.5f, w, h, 4.0f, 2.0f, stroke);
    fs_state_restore(core);
}

/* ================================================================
   TITLE + STATUS RENDERING
   ================================================================ */

static void demo_render_ui(FS_Core* core, bool font_ready) {
    if (font_ready) {
        fs_cmd_text_utf8(core, 20.0f, 20.0f, 18.0f,
                        "obstacle_wrap_demo", PAL_ACCENT1, 400.0f);
        fs_cmd_text_utf8(core, 20.0f, 42.0f, 10.0f,
                        "prepare once | relayout many | scanline slot flow",
                        PAL_TEXT_DIM, 500.0f);
    }

    fs_cmd_rect_stroke(core, g_state.body_x, g_state.body_y,
                       g_state.body_w, g_state.body_h,
                       0.0f, 1.0f, PAL_GRID);

    float obs_cx = g_state.obs_x;
    float obs_cy = g_state.obs_y;
    uint32_t fill = lerp_color(PAL_OBSTACLE, 0x40FFFFFFu,
                               sinf((float)g_state.total_time * 2.0f) * 0.2f + 0.2f);
    draw_rotated_obstacle(core, obs_cx, obs_cy,
                          g_state.obs_w, g_state.obs_h,
                          g_state.obs_angle,
                          fill, PAL_OBSTACLE);

    if (font_ready) {
        fs_cmd_text_utf8(core, obs_cx - 40.0f, obs_cy + 6.0f, 9.0f,
                         "obstacle", PAL_TEXT_DIM, 80.0f);
    }
}

static bool point_in_rotated_rect(float px, float py, float cx, float cy, float w, float h, float angle) {
    float dx = px - cx;
    float dy = py - cy;
    float c = cosf(-angle);
    float s = sinf(-angle);
    float lx = dx * c - dy * s;
    float ly = dx * s + dy * c;
    return fabsf(lx) <= w * 0.5f && fabsf(ly) <= h * 0.5f;
}

static void rotated_rect_corners(float cx, float cy, float hw, float hh, float angle,
                                 float out_x[4], float out_y[4]) {
    const float lx[4] = { -hw,  hw,  hw, -hw };
    const float ly[4] = { -hh, -hh,  hh,  hh };
    float c = cosf(angle);
    float s = sinf(angle);
    for (int i = 0; i < 4; i++) {
        out_x[i] = cx + lx[i] * c - ly[i] * s;
        out_y[i] = cy + lx[i] * s + ly[i] * c;
    }
}

static uint32_t clip_poly_y_min(const float* in_x, const float* in_y, uint32_t in_count, float y_min,
                                float* out_x, float* out_y) {
    if (in_count == 0) return 0;
    uint32_t out_count = 0;
    for (uint32_t i = 0; i < in_count; i++) {
        uint32_t j = (i + in_count - 1u) % in_count;
        float sx = in_x[j], sy = in_y[j];
        float ex = in_x[i], ey = in_y[i];
        bool s_in = sy >= y_min;
        bool e_in = ey >= y_min;

        if (s_in && e_in) {
            out_x[out_count] = ex;
            out_y[out_count] = ey;
            out_count++;
        } else if (s_in && !e_in) {
            float t = (y_min - sy) / (ey - sy);
            out_x[out_count] = sx + (ex - sx) * t;
            out_y[out_count] = y_min;
            out_count++;
        } else if (!s_in && e_in) {
            float t = (y_min - sy) / (ey - sy);
            out_x[out_count] = sx + (ex - sx) * t;
            out_y[out_count] = y_min;
            out_count++;
            out_x[out_count] = ex;
            out_y[out_count] = ey;
            out_count++;
        }
    }
    return out_count;
}

static uint32_t clip_poly_y_max(const float* in_x, const float* in_y, uint32_t in_count, float y_max,
                                float* out_x, float* out_y) {
    if (in_count == 0) return 0;
    uint32_t out_count = 0;
    for (uint32_t i = 0; i < in_count; i++) {
        uint32_t j = (i + in_count - 1u) % in_count;
        float sx = in_x[j], sy = in_y[j];
        float ex = in_x[i], ey = in_y[i];
        bool s_in = sy <= y_max;
        bool e_in = ey <= y_max;

        if (s_in && e_in) {
            out_x[out_count] = ex;
            out_y[out_count] = ey;
            out_count++;
        } else if (s_in && !e_in) {
            float t = (y_max - sy) / (ey - sy);
            out_x[out_count] = sx + (ex - sx) * t;
            out_y[out_count] = y_max;
            out_count++;
        } else if (!s_in && e_in) {
            float t = (y_max - sy) / (ey - sy);
            out_x[out_count] = sx + (ex - sx) * t;
            out_y[out_count] = y_max;
            out_count++;
            out_x[out_count] = ex;
            out_y[out_count] = ey;
            out_count++;
        }
    }
    return out_count;
}
static void demo_update_interaction(FS_GlfwBackend* backend) {
    if (!backend || !backend->window) return;

    double mx = 0, my = 0;
    glfwGetCursorPos(backend->window, &mx, &my);

    float mouse_x = 0.0f;
    float mouse_y = 0.0f;
    cursor_window_to_design(backend, mx, my, &mouse_x, &mouse_y);

    float obs_cx = g_state.obs_x;
    float obs_cy = g_state.obs_y;
    float obs_hw = g_state.obs_w * 0.5f;
    float obs_hh = g_state.obs_h * 0.5f;

    bool hover = point_in_rotated_rect(mouse_x, mouse_y,
                                       obs_cx, obs_cy,
                                       g_state.obs_w, g_state.obs_h,
                                       g_state.obs_angle);
    g_state.obs_hovered = hover;

    int lmb = glfwGetMouseButton(backend->window, GLFW_MOUSE_BUTTON_LEFT);

    static float s_drag_start_x = 0.0f;
    static float s_drag_start_y = 0.0f;
    static float s_drag_obs_x = 0.0f;
    static float s_drag_obs_y = 0.0f;
    static bool  s_was_dragging = false;
    static bool  s_press_started_on_obstacle = false;
    static int   s_prev_lmb = GLFW_RELEASE;

    bool pressed_this_frame = (lmb == GLFW_PRESS && s_prev_lmb != GLFW_PRESS);
    bool released_this_frame = (lmb != GLFW_PRESS && s_prev_lmb == GLFW_PRESS);

    if (pressed_this_frame) {
        s_press_started_on_obstacle = g_state.obs_hovered;
        if (g_state.obs_hovered) {
            g_state.obs_dragging = true;
            s_was_dragging = false;
            s_drag_start_x = mouse_x;
            s_drag_start_y = mouse_y;
            s_drag_obs_x = g_state.obs_x;
            s_drag_obs_y = g_state.obs_y;
        }
    }

    if (lmb == GLFW_PRESS && g_state.obs_dragging) {
        float dx = mouse_x - s_drag_start_x;
        float dy = mouse_y - s_drag_start_y;
        if (fabsf(dx) > 2.0f || fabsf(dy) > 2.0f) s_was_dragging = true;

        float new_x = s_drag_obs_x + dx;
        float new_y = s_drag_obs_y + dy;

        if (new_x - obs_hw < g_state.body_x) new_x = g_state.body_x + obs_hw;
        if (new_x + obs_hw > g_state.body_x + g_state.body_w) new_x = g_state.body_x + g_state.body_w - obs_hw;
        if (new_y - obs_hh < g_state.body_y) new_y = g_state.body_y + obs_hh;
        if (new_y + obs_hh > g_state.body_y + g_state.body_h) new_y = g_state.body_y + g_state.body_h - obs_hh;

        g_state.obs_x = new_x;
        g_state.obs_y = new_y;
    }

    if (released_this_frame) {
        if (s_press_started_on_obstacle && !s_was_dragging) {
            g_state.obs_target_angle += (float)(30.0 * PI / 180.0);
        }
        g_state.obs_dragging = false;
        s_was_dragging = false;
        s_press_started_on_obstacle = false;
    }

    s_prev_lmb = lmb;

    float diff = g_state.obs_target_angle - g_state.obs_angle;
    g_state.obs_angle += diff * 0.12f;
}

/* ================================================================
   PREPARE TEXT
   ================================================================ */

static void prepare_all_texts(void) {
    g_state.layout_ctx = fs_text_layout_create("Arial", g_state.font_size, 1.5f);
    if (g_state.layout_ctx) {
        g_state.body_prep = fs_text_layout_prepare(g_state.layout_ctx, k_body_text);
    }
}

/* ================================================================
   INIT
   ================================================================ */

static void demo_init(float fbw, float fbh) {
    memset(&g_state, 0, sizeof(g_state));
    g_state.dw = DEMO_DW;
    g_state.dh = DEMO_DH;
    g_state.fbw = fbw;
    g_state.fbh = fbh;
    g_state.scale = 1.0f;

    /* All layout stored in DESIGN SPACE (1280x720 logical, unchanged on resize).
       Scale converts to framebuffer pixels only at render time. */
    g_state.body_x = 60.0f;
    g_state.body_y = 65.0f;
    g_state.body_w = 960.0f;
    g_state.body_h = 540.0f;

    g_state.font_size = 13.0f;
    g_state.line_height = g_state.font_size * 1.5f;

    /* Obstacle: centered in body, design space */
    g_state.obs_w = 180.0f;
    g_state.obs_h = 180.0f;
    g_state.obs_x = g_state.body_x + g_state.body_w * 0.5f;  /* = 540 */
    g_state.obs_y = g_state.body_y + g_state.body_h * 0.5f;

    prepare_all_texts();
}

/* ================================================================
   MAIN
   ================================================================ */

int main(void) {
    FS_GlfwBackend backend;
    if (!fs_glfw_backend_init(&backend, (uint32_t)DEMO_DW, (uint32_t)DEMO_DH,
                               "fs_text_layout obstacle_wrap demo")) {
        fprintf(stderr, "fs_glfw_backend_init failed\n");
        return 1;
    }

    FS_Core* core = fs_glfw_backend_core(&backend);
    if (!core) {
        fprintf(stderr, "fs_glfw_backend_core returned NULL\n");
        fs_glfw_backend_shutdown(&backend);
        return 1;
    }

    fs_core_set_font_backend(core, fs_get_freetype2_font_backend());
    const char* font_paths[] = {
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    };
    for (size_t fi = 0; fi < ARRAY_COUNT(font_paths); fi++) {
        if (fs_core_load_font_file(core, font_paths[fi])) {
            g_state.font_ready = true;
        }
    }
    printf("Font loaded: %s\n", g_state.font_ready ? "yes" : "no");

    demo_init((float)backend.width, (float)backend.height);
    demo_relayout();

    double prev_time = glfwGetTime();
    int prev_d_key = GLFW_RELEASE;
    int prev_s_key = GLFW_RELEASE;

    while (!fs_glfw_backend_should_close(&backend)) {
        fs_glfw_backend_poll_events();

        double now = glfwGetTime();
        float dt = (float)(now - prev_time);
        prev_time = now;

        g_state.total_time += dt;
        g_state.frame_dt = dt;
        g_state.frame_count++;
        if (g_state.frame_count % 30 == 0) {
            g_state.fps = (dt > 0.0f) ? (1.0f / dt) : 60.0f;
        }

        /* Scale — converts design space (1280x720) to framebuffer pixels */
        float sx_fb = (backend.width > 0) ? ((float)backend.width / (float)DEMO_DW) : 1.0f;
        float sy_fb = (backend.height > 0) ? ((float)backend.height / (float)DEMO_DH) : 1.0f;
        g_state.scale = (sx_fb < sy_fb) ? sx_fb : sy_fb;
        g_state.fbw = (float)backend.width;
        g_state.fbh = (float)backend.height;

        /* Debug toggle */
        int kd = glfwGetKey(backend.window, GLFW_KEY_D);
        if (kd == GLFW_PRESS && prev_d_key != GLFW_PRESS) {
            g_state.show_debug = !g_state.show_debug;
        }
        prev_d_key = kd;

        /* Single-slot mode toggle */
        int ks = glfwGetKey(backend.window, GLFW_KEY_S);
        if (ks == GLFW_PRESS && prev_s_key != GLFW_PRESS) {
            g_state.single_slot_mode = !g_state.single_slot_mode;
        }
        prev_s_key = ks;

        /* Interaction */
        demo_update_interaction(&backend);

        /* Rebuild layout every frame */
        demo_relayout();

        /* Render */
        fs_core_begin_commands(core);
        fs_context_reset(core);

        /* Background — framebuffer space (before scale) */
        fs_cmd_rect(core, 0, 0, g_state.fbw, g_state.fbh, 0, PAL_BG);

        /* Enter design space: all subsequent draws use design coordinates */
        fs_state_save(core);
        fs_scale(core, g_state.scale, g_state.scale);

        /* UI: title, obstacle, body outline */
        demo_render_ui(core, g_state.font_ready);

        /* Fragments */
        demo_render_frags(core);

        /* Debug overlay */
        if (g_state.show_debug) {
            demo_render_debug(core);
        }

        /* Exit design space */
        fs_state_restore(core);

        /* Present */
        if (!fs_glfw_backend_present(&backend, 0.043f, 0.059f, 0.071f, 1.0f)) {
            fprintf(stderr, "Frame present failed\n");
            break;
        }
    }

    /* Cleanup */
    if (g_state.body_prep) fs_text_layout_prepared_destroy(g_state.body_prep);
    if (g_state.layout_ctx) fs_text_layout_destroy(g_state.layout_ctx);
    fs_glfw_backend_shutdown(&backend);
    return 0;
}
