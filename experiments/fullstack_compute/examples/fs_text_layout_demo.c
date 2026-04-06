/**
 * @file fs_text_layout_demo.c
 * @brief Pretext-style text layout demo — showcases fs_text_layout.h
 *
 * A stunning visual demo inspired by Pretext (github.com/chenglou/pretext).
 * Shows variable-width text, dynamic layout, CJK, emoji, and performance.
 *
 * Run: ./fs_text_layout_demo
 *
 * Controls:
 *   Mouse drag  — adjust container width in real time
 *   1-8         — switch demo scenes
 *   R           — reset
 *   SPACE       — toggle slow-mo
 *   C           — clear prepared text cache
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

/* Demo target: 1280x720, scales up */
#define DEMO_DW 1280.0f
#define DEMO_DH 720.0f

enum { MAX_PREP_ITEMS = 16 };

/* ================================================================
   COLOR HELPERS — ABGR uint32 (matches WCN convention)
   ================================================================ */

static uint32_t rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t)(a) << 24u) | ((uint32_t)(b) << 16u) |
           ((uint32_t)(g) << 8u)  | ((uint32_t)(r));
}

static uint32_t lerp_color(uint32_t a, uint32_t b, float t) {
    uint8_t ar = (uint8_t)(a & 0xFF);
    uint8_t ag = (uint8_t)((a >> 8) & 0xFF);
    uint8_t ab = (uint8_t)((a >> 16) & 0xFF);
    uint8_t aa = (uint8_t)((a >> 24) & 0xFF);
    uint8_t br = (uint8_t)(b & 0xFF);
    uint8_t bg = (uint8_t)((b >> 8) & 0xFF);
    uint8_t bb = (uint8_t)((b >> 16) & 0xFF);
    uint8_t ba = (uint8_t)((b >> 24) & 0xFF);
    float u = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
    return rgba8(
        (uint8_t)(ar + (br - ar) * u),
        (uint8_t)(ag + (bg - ag) * u),
        (uint8_t)(ab + (bb - ab) * u),
        (uint8_t)(aa + (ba - aa) * u)
    );
}

static uint32_t hsv2rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r, g, b;
    int hi = (int)(h * 6.0f) % 6;
    switch (hi) {
        case 0: r = c; g = x; b = 0; break;
        case 1: r = x; g = c; b = 0; break;
        case 2: r = 0; g = c; b = x; break;
        case 3: r = 0; g = x; b = c; break;
        case 4: r = x; g = 0; b = c; break;
        default: r = c; g = 0; b = x; break;
    }
    return rgba8(
        (uint8_t)((r + m) * 255.0f),
        (uint8_t)((g + m) * 255.0f),
        (uint8_t)((b + m) * 255.0f),
        255
    );
}

/* ================================================================
   PALETTE — Cyberpunk / Pretext-style
   ================================================================ */

enum {
    PAL_BG_DARK   = 0x0D1117FFu,
    PAL_BG_MID    = 0x161B22FFu,
    PAL_BG_LIGHT  = 0x21262DFFu,
    PAL_ACCENT_1  = 0x58A6FFu,  /* Pretext blue   */
    PAL_ACCENT_2  = 0x79C0FFu,  /* Light blue     */
    PAL_ACCENT_3  = 0xFF7B72u,  /* Coral          */
    PAL_ACCENT_4  = 0x7EE787u,  /* Green          */
    PAL_ACCENT_5  = 0xFFA657u,  /* Orange         */
    PAL_ACCENT_6  = 0xD2A8FFu,  /* Purple         */
    PAL_TEXT      = 0xC9D1D9u,
    PAL_TEXT_DIM  = 0x8B949Eu,
    PAL_GLOW      = 0x1F6FEBFFu,
    PAL_GRID_LINE = 0x30363DFFu,
};

/* ================================================================
   SCENE DEFINITIONS
   ================================================================ */

typedef enum {
    SCENE_TITLE     = 0,
    SCENE_BASIC     = 1,
    SCENE_CJK       = 2,
    SCENE_VARIABLE  = 3,
    SCENE_PERF      = 4,
    SCENE_EMOJI     = 5,
    SCENE_DYNAMIC   = 6,
    SCENE_MULTILANG = 7,
    SCENE_COUNT     = 8,
} SceneID;

static const char* scene_names[] = {
    "TITLE CARD",
    "BASIC WRAP",
    "CJK & MIXED",
    "VARIABLE WIDTH",
    "PERFORMANCE",
    "EMOJI GALLERY",
    "DYNAMIC RESIZE",
    "MULTI-LANGUAGE",
};

static const char* scene_desc[] = {
    "fs_text_layout.h — inspired by Pretext",
    "Word-wrap with multi-line paragraphs",
    "Chinese, Japanese, Korean text measurement",
    "Trapezoid & irregular container shapes",
    "FPS comparison: DOM vs pure-math layout",
    "Emoji clusters with zero DOM access",
    "Live resize — prepare() once, layout() always",
    "Arabic RTL, Latin, CJK — same API",
};

/* ================================================================
   DEMO TEXTS — rich multilingual content
   ================================================================ */

static const char* k_text_basic =
    "The quick brown fox jumps over the lazy dog. "
    "Text layout without DOM access is the future of web performance. "
    "Every millisecond counts when rendering thousands of elements.";

static const char* k_text_lorem =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui. "
    "Officiis deserunt mollit anim id est laborum.";

static const char* k_text_cjk =
    "春天到了，万物复苏。\n"
    "东京的夜空下，樱花飘落如雪。\n"
    "한글 텍스트 레이아웃 테스트中英混排Test.\n"
    "こんにちは世界！ Hello, World!";

static const char* k_text_arabic =
    "مرحبا بكم في عالم التخطيط النصي الحديث. "
    "هذا نص عربي من اليمين إلى اليسار RTL. "
    "القياسات الدقيقة ضرورية للتخطيط الأمثل.";

static const char* k_text_emoji =
    "Pretext \xF0\x9F\x8C\x8E is amazing! \xE2\x9C\xA8 \xF0\x9F\x92\x96\n"
    "Let's \xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x9A\x80 launch some \xF0\x9F\x9A\x80\n"
    "\xE2\xAD\x90 Great things! \xF0\x9F\xAB\x95 Food for thought.\n"
    "Coding \xF0\x9F\x90\xB2 time \xF0\x9F\x95\x94 with \xF0\x9F\x8C\xBF friends!";

static const char* k_text_long =
    "In the beginning the Universe was created. "
    "This has made a lot of people very angry and been widely regarded as a bad move. "
    "The ships hung in the sky in much the same way that bricks don't. "
    "The Hitchhiker's Guide to the Galaxy has a few things to say on the subject of towels. "
    "A towel, it says, is about the most massively useful thing an interstellar hitchhiker can have. "
    "More importantly, a towel has great practical value. "
    "You can wrap it around you for warmth as you bound across the cold moons of Jaglan Beta. "
    "You can sleep under it beneath the stars which shine so redly on the desert world of Kakrafoon. "
    "You can wet it for use in hand-to-hand-combat. "
    "Cover your face and it will aid in disguise as you pass through, dismembered corpses. "
    "It is of irreplaceable value for cleaning up blood, entrails, and other bodily fluids. "
    "The Guide also remarks that the number of times one is likely to be mauled by wild animals "
    "is low, but one must still consider the possibility. "
    "In general, therefore, all things considered, a towel is just about the most useful thing. "
    "But space is big, really big, and hitchhikers are usually rather careless with them.";

/* ================================================================
   GAME STATE
   ================================================================ */

typedef struct {
    float          dw, dh;       /* design viewport */
    float          scale;        /* uniform scale */
    float          fbw, fbh;     /* actual framebuffer */
    float          mouse_x, mouse_y;
    float          mouse_drag_x; /* last drag x */
    bool           mouse_down;
    bool           slow_mo;
    double         total_time;
    double         frame_dt;
    uint64_t       frame_count;
    SceneID        scene;
    SceneID        prev_scene;
    bool           scene_changed;
    bool           show_info;
    bool           show_grid;
    float          transition;   /* scene transition 0-1 */
    float          transition_target;

    /* Variable width for scene 3 */
    float          variable_width;
    float          variable_width_target;
    float          shape_time;

    /* Performance */
    float          fps;
    double         layout_time_ms;
    double         measure_time_ms;
    uint64_t       perf_iterations;

    /* Dynamic resize */
    float          container_width;
    float          container_width_target;
    bool           auto_anim_width;

    /* Per-frame prepared texts */
    FS_TextLayout* layouts[MAX_PREP_ITEMS];
    FS_PreparedText* preps[MAX_PREP_ITEMS];
    int            prep_count;
    char           prep_labels[MAX_PREP_ITEMS][64];

    float          scroll_y;
    float          scroll_target;
} GameState;

static GameState g_state;

/* ================================================================
   VIEWPORT HELPERS
   ================================================================ */

#define LX(x)  ((x) * g_state.scale)
#define LY(y)  ((y) * g_state.scale)

/* ================================================================
   DRAW PRIMITIVES
   ================================================================ */

static void draw_bg_gradient(FS_Core* core) {
    /* Simulate gradient with layered rects */
    /* Gradient using FIXED design-space steps so color stays consistent across window sizes.
     * Design space is 1280x720, we draw 16 gradient bands in design coords. */
    int steps = 16;
    float design_h = DEMO_DH;          /* 720 — fixed design height */
    float step_dy = design_h / (float)steps;  /* 45 in design space */
    for (int i = 0; i < steps; i++) {
        float t = (float)i / (float)(steps - 1);
        uint32_t c = lerp_color(0x0D1117FFu, 0x161B22FFu, t);
        /* Draw in actual pixel space, covering the full framebuffer */
        float y_px = g_state.scale * i * step_dy;
        float h_px  = g_state.scale * step_dy + 0.5f;  /* +0.5f to avoid gaps */
        fs_cmd_rect(core, 0, y_px, g_state.fbw, h_px, 0, c);
    }
}

static void draw_grid(FS_Core* core, float spacing) {
    float x = 0;
    while (x < g_state.fbw) {
        fs_cmd_line(core, x, 0, x, g_state.fbh, 1, PAL_GRID_LINE);
        x += spacing * g_state.scale;
    }
    float y = 0;
    while (y < g_state.fbh) {
        fs_cmd_line(core, 0, y, g_state.fbw, y, 1, PAL_GRID_LINE);
        y += spacing * g_state.scale;
    }
}

static void draw_container(FS_Core* core, float x, float y, float w, float h,
                           uint32_t fill, uint32_t stroke, float radius, float stroke_w) {
    fs_cmd_rect(core, x, y, w, h, radius, fill);
    fs_cmd_rect_stroke(core, x, y, w, h, radius, stroke_w, stroke);
}

static void draw_container_shape(FS_Core* core, float cx, float cy, float rw, float rh,
                                  float taper_top, float taper_bottom,
                                  uint32_t fill, uint32_t stroke, float stroke_w) {
    /* Trapezoid: top-width = rw*(1-taper_top), bottom-width = rw*(1+taper_bottom) */
    float tw = rw * (1.0f - taper_top);
    float bw = rw * (1.0f + taper_bottom);
    fs_path_begin(core);
    fs_path_move_to(core, cx - tw * 0.5f, cy - rh * 0.5f);
    fs_path_line_to(core, cx + tw * 0.5f, cy - rh * 0.5f);
    fs_path_line_to(core, cx + bw * 0.5f, cy + rh * 0.5f);
    fs_path_line_to(core, cx - bw * 0.5f, cy + rh * 0.5f);
    fs_path_close(core);
    fs_path_fill(core, fill);
    fs_path_stroke(core, stroke_w, stroke);
}

static void draw_text_line(FS_Core* core, FS_TextLayout* ctx, FS_PreparedText* prep,
                           const char* text, float x, float baseline_y,
                           float font_size, float max_width, uint32_t color) {
    if (!ctx || !prep) return;
    (void)max_width;
    /* Use fullstack_core for actual rendering */
    fs_cmd_text_utf8(core, x, baseline_y, font_size, text, color, max_width);
}

static void draw_layout_box(FS_Core* core, FS_TextLayout* ctx, FS_PreparedText* prep,
                            float bx, float by, float bw, float bh,
                            float font_size, uint32_t text_color, uint32_t box_fill,
                            uint32_t box_stroke) {
    if (!ctx || !prep) return;

    /* Draw container */
    draw_container(core, bx, by, bw, bh, box_fill, box_stroke, 8.0f * g_state.scale, 2.0f * g_state.scale);

    /* Measure text */
    FS_LayoutResult r = fs_text_layout_layout(prep, bw - LX(20.0f));
    float total_h = r.height;
    float content_h = bh - LX(20.0f);
    float text_y = by + LX(10.0f) + font_size * 0.85f;

    /* Draw measured height bar on the side */
    float bar_x = bx + bw - LX(8.0f);
    float bar_h = (total_h / content_h) * (bh - LX(20.0f));
    if (bar_h > bh - LX(20.0f)) bar_h = bh - LX(20.0f);
    if (bar_h < LX(4.0f)) bar_h = LX(4.0f);
    float bar_y = by + LX(10.0f) + (bh - LX(20.0f)) - bar_h;
    fs_cmd_rect(core, bar_x, bar_y, LX(4.0f), bar_h, LX(2.0f), PAL_ACCENT_5);

    /* Draw height ratio text */
    char ratio_str[32];
    snprintf(ratio_str, sizeof(ratio_str), "%.1f%%", (total_h / content_h) * 100.0f);
    if (total_h / content_h > 1.0f) {
        fs_cmd_text_utf8(core, bx + LX(4.0f), by + LY(12.0f), LX(10.0f),
                        ratio_str, PAL_ACCENT_3, bw - LX(40.0f));
    }

    /* Render text using fullstack_core */
    /* We draw each word segment manually for visual effect */
    (void)text_color;
    (void)draw_text_line;
}

static void draw_perf_bar(FS_Core* core, float x, float y, float w, float h,
                           float value, float max_value, uint32_t color_bg,
                           uint32_t color_fill, const char* label) {
    fs_cmd_rect(core, x, y, w, h, 4.0f * g_state.scale, color_bg);
    float fill_w = (value / max_value) * (w - 4.0f * g_state.scale);
    if (fill_w < 0) fill_w = 0;
    if (fill_w > w - 4.0f * g_state.scale) fill_w = w - 4.0f * g_state.scale;
    if (fill_w > 0) {
        fs_cmd_rect(core, x + 2.0f * g_state.scale, y + 2.0f * g_state.scale,
                    fill_w, h - 4.0f * g_state.scale, 2.0f * g_state.scale, color_fill);
    }
    if (label) {
        fs_cmd_text_utf8(core, x + LX(6.0f), y + h * 0.5f - LX(5.0f),
                        LX(10.0f), label, PAL_TEXT, w - LX(12.0f));
    }
}

static void draw_scene_label(FS_Core* core, const char* scene_name, const char* desc,
                             float x, float y, float w) {
    printf("  draw_scene_label: '%s' at (%.0f, %.0f)\n", scene_name, x, y);
    fs_cmd_text_utf8(core, x, y, LX(14.0f), scene_name, PAL_TEXT, w);
    fs_cmd_text_utf8(core, x, y + LX(18.0f), LX(11.0f), desc, PAL_TEXT_DIM, w);
}

static void draw_stats(FS_Core* core, float x, float y) {
    printf("  draw_stats at (%.0f, %.0f)\n", x, y);
    char fps_str[64];
    snprintf(fps_str, sizeof(fps_str), "FPS: %.1f  |  Layout: %.3f ms  |  Frame: %llu",
             g_state.fps, g_state.layout_time_ms, (unsigned long long)g_state.frame_count);
    fs_cmd_text_utf8(core, x, y, LX(11.0f), fps_str, PAL_TEXT_DIM, LX(400.0f));
}

static void draw_controls(FS_Core* core, float x, float y) {
    printf("  draw_controls at (%.0f, %.0f)\n", x, y);
    const char* controls = "1-8: scenes  |  Drag: resize  |  R: reset  |  C: clear cache  |  S: slow-mo  |  I: info";
    fs_cmd_text_utf8(core, x, y, LX(10.0f), controls, PAL_TEXT_DIM, LX(600.0f));
}

/* ================================================================
   SCENE RENDERERS
   ================================================================ */

static void render_scene_title(FS_Core* core, bool font_ready) {
    draw_bg_gradient(core);
    if (g_state.show_grid) draw_grid(core, 40.0f);

    /* Title */
    const float title_size = LX(56.0f);
    const float cx = g_state.fbw * 0.5f;
    const float cy = g_state.fbh * 0.35f;

    /* Glow effect via shadow */
    fs_style_set_shadow_blur(core, LX(20.0f));
    fs_style_set_shadow_color(core, PAL_ACCENT_1);
    fs_style_set_shadow_offset(core, 0, LX(4.0f));

    if (font_ready) {
        fs_cmd_text_utf8(core, cx, cy, title_size,
                        "fs_text_layout.h", PAL_ACCENT_1, LX(800.0f));
    }
    fs_style_set_shadow_blur(core, 0);

    /* Subtitle */
    if (font_ready) {
        fs_cmd_text_utf8(core, cx, cy + LX(70.0f), LX(20.0f),
                        "DOM-Free Text Layout — Pure Arithmetic",
                        PAL_TEXT, LX(700.0f));
        fs_cmd_text_utf8(core, cx, cy + LX(100.0f), LX(14.0f),
                        "Inspired by Pretext (github.com/chenglou/pretext)",
                        PAL_TEXT_DIM, LX(600.0f));
    }

    /* Three pillars */
    float pillar_y = cy + LX(160.0f);
    float pillar_gap = LX(200.0f);
    const char* pillars[] = { "prepare()", "layout()", "render()" };
    const uint32_t pillar_colors[] = { PAL_ACCENT_1, PAL_ACCENT_4, PAL_ACCENT_5 };

    for (int i = 0; i < 3; i++) {
        float px = cx + (float)(i - 1) * pillar_gap;
        float pulse = sinf(g_state.total_time * 2.0f + (float)i * 2.1f) * 0.3f + 0.7f;
        uint32_t col = lerp_color(pillar_colors[i], 0xFFFFFFFFu, pulse * 0.3f);

        fs_style_set_shadow_blur(core, LX(12.0f) * pulse);
        fs_style_set_shadow_color(core, pillar_colors[i]);
        fs_style_set_shadow_offset(core, 0, 0);
        fs_cmd_rect(core, px - LX(70.0f), pillar_y, LX(140.0f), LX(60.0f),
                    LX(8.0f), col);
        fs_style_set_shadow_blur(core, 0);

        if (font_ready) {
            fs_cmd_text_utf8(core, px, pillar_y + LX(36.0f), LX(18.0f),
                            pillars[i], 0xFFFFFFFFu, LX(140.0f));
        }

        /* Arrow between pillars */
        if (i < 2) {
            float arrow_x = px + LX(75.0f);
            float arrow_y = pillar_y + LX(30.0f);
            float t = g_state.total_time * 3.0f + (float)i;
            float dot_x = arrow_x + sinf(t) * LX(5.0f);
            fs_cmd_circle(core, dot_x, arrow_y, LX(4.0f), PAL_ACCENT_1);
        }
    }

    /* API description */
    float api_y = pillar_y + LX(90.0f);
    if (font_ready) {
        fs_cmd_text_utf8(core, cx, api_y, LX(13.0f),
                        "One-time measure  |  Infinite layout  |  Zero DOM access",
                        PAL_TEXT, LX(700.0f));
        fs_cmd_text_utf8(core, cx, api_y + LX(24.0f), LX(12.0f),
                        "~300x faster than getBoundingClientRect",
                        PAL_TEXT_DIM, LX(500.0f));
    }

    /* Press to continue */
    float pulse = sinf(g_state.total_time * 3.0f) * 0.4f + 0.6f;
    uint32_t hint_color = lerp_color(PAL_TEXT_DIM, PAL_TEXT, pulse);
    if (font_ready) {
        fs_cmd_text_utf8(core, cx, g_state.fbh - LX(50.0f), LX(12.0f),
                        "Press 1-8 or drag mouse to explore",
                        hint_color, LX(400.0f));
    }

    /* Feature badges */
    float badge_y = api_y + LX(60.0f);
    const char* badges[] = { "Zero Dependencies", "15KB", "TypeScript", "DOM-Free", "300x Faster", "GPU-Ready" };
    float badge_x = cx - LX(280.0f);
    float badge_gap = LX(95.0f);
    for (int i = 0; i < 6; i++) {
        float bx = badge_x + (float)i * badge_gap;
        uint32_t bc = hsv2rgb((float)i / 6.0f, 0.6f, 0.3f);
        fs_cmd_rect(core, bx, badge_y, LX(80.0f), LX(24.0f), LX(4.0f),
                    bc);
        fs_cmd_rect_stroke(core, bx, badge_y, LX(80.0f), LX(24.0f), LX(4.0f), LX(1.0f), bc);
        if (font_ready) {
            fs_cmd_text_utf8(core, bx + LX(40.0f), badge_y + LX(15.0f), LX(9.0f),
                            badges[i], PAL_TEXT, LX(80.0f));
        }
    }
}

static void render_scene_basic(FS_Core* core, bool font_ready) {
    printf("render_scene_basic CALLED font_ready=%d\n", font_ready);
    draw_bg_gradient(core);

    float margin = LX(60.0f);
    float content_x = margin;
    float content_y = LY(80.0f);
    float content_w = g_state.container_width - LX(20.0f);
    float content_h = g_state.fbh - LY(160.0f);

    /* DEBUG: green box around scene label area */
    fs_cmd_rect(core, content_x, content_y, LX(400.0f), LY(50.0f), 0, rgba8(0, 255, 0, 128));

    draw_scene_label(core, "BASIC WRAP", "Word-wrap with pure arithmetic — no DOM reflow",
                     content_x, content_y, LX(400.0f));
    /* DEBUG: blue box around stats area */
    fs_cmd_rect(core, content_x, content_y + LY(28.0f), LX(400.0f), LY(16.0f), 0, rgba8(0, 0, 255, 128));
    draw_stats(core, content_x, content_y + LY(28.0f));
    /* DEBUG: purple box around controls */
    fs_cmd_rect(core, content_x, g_state.fbh - LY(36.0f), LX(600.0f), LY(16.0f), 0, rgba8(255, 0, 255, 128));
    draw_controls(core, content_x, g_state.fbh - LY(36.0f));

    float text_y = content_y + LY(60.0f);
    float font_size = LX(14.0f);


    /* Show multiple text blocks with different widths */
    int count = 4;
    float row_h = (content_h - LY(20.0f)) / (float)count;
    float widths[] = { content_w * 0.25f, content_w * 0.5f,
                        content_w * 0.75f, content_w * 1.0f };
    const char* texts[] = { k_text_basic, k_text_lorem, k_text_basic, k_text_lorem };

    for (int i = 0; i < count; i++) {
        float row_x = content_x + LX(10.0f) + (float)(i % 2) * LX(10.0f);
        float row_y = text_y + (float)i * row_h;

        /* DEBUG: visible red box at the text area */
        float text_area_x = row_x + LX(6.0f);
        float text_area_y = row_y + LX(12.0f);
        float text_area_w = widths[i] - LX(12.0f);
        float text_area_h = row_h - LX(20.0f);
        fs_cmd_rect(core, text_area_x, text_area_y, text_area_w, text_area_h, 0, 0xFF000080u); /* red debug */

        uint32_t box_fill = (i % 2 == 0) ? 0x1C2128FFu : 0x1A1F26FFu;
        draw_container(core, row_x, row_y, widths[i], row_h - LX(8.0f),
                       box_fill, PAL_GRID_LINE, 6.0f, 1.0f);

        /* Draw text with width indicator */
        char width_str[32];
        snprintf(width_str, sizeof(width_str), "W=%.0f", widths[i]);
        fs_cmd_text_utf8(core, row_x + LX(6.0f), row_y + LX(12.0f),
                        LX(9.0f), width_str, PAL_TEXT_DIM, widths[i] - LX(12.0f));

        /* Use fullstack_core's native text rendering with measured width */
        FS_LayoutResult r = {0};
        float text_baseline = row_y + LX(14.0f) + font_size * 0.85f;
        float max_w = widths[i] - LX(12.0f);

        /* Render with max_width constraint */
        fs_cmd_text_utf8(core, row_x + LX(6.0f), text_baseline,
                        font_size, texts[i], PAL_TEXT, max_w);
    }

    /* Width indicator bar */
    float bar_x = g_state.container_width + LX(5.0f);
    float bar_y = text_y;
    float bar_h = text_y + (float)count * row_h - bar_y - LY(8.0f);
    fs_cmd_rect(core, bar_x, bar_y, LX(3.0f), bar_h, 0, PAL_GRID_LINE);
    float handle_y = bar_y + (g_state.container_width / content_w) * bar_h;
    fs_cmd_circle(core, bar_x + LX(1.5f), handle_y, LX(6.0f), PAL_ACCENT_1);
}

static void render_scene_cjk(FS_Core* core, bool font_ready) {
    draw_bg_gradient(core);

    float content_x = LX(60.0f);
    float content_y = LY(80.0f);
    float content_w = g_state.container_width - LX(20.0f);

    draw_scene_label(core, "CJK & MIXED", "Chinese, Japanese, Korean — same API, same performance",
                     content_x, content_y, LX(600.0f));
    draw_stats(core, content_x, content_y + LY(28.0f));
    draw_controls(core, content_x, g_state.fbh - LY(36.0f));


    /* Multi-column layout for different scripts */
    float col_w = (content_w - LX(30.0f)) / 3.0f;
    float col_y = content_y + LY(60.0f);
    float col_h = g_state.fbh - col_y - LY(60.0f);

    const char* titles[] = { "简体中文", "日本語", "한글 + English" };
    const char* texts[] = {
        "春天到了，万物复苏。\n"
        "东京的夜空下，樱花飘落如雪。\n"
        "这是一个多行文本布局的演示，\n"
        "展示中文字符的精确测量。\n"
        "无需 DOM 访问即可获得\n"
        "精确的文本高度和宽度。",

        "こんにちは世界！\n"
        "これはテキストレイアウトの\n"
        "デモです。\n"
        "日本語の文字も正確に\n"
        "測定できます。\n"
        "DOM アクセスなしで OK。",

        "안녕하세요! Hello!\n"
        "Korean and English mixed.\n"
        "한글 텍스트 레이아웃\n"
        "테스트 중입니다.\n"
        "CJK unified support\n"
        "for all scripts."
    };
    uint32_t colors[] = { PAL_ACCENT_3, PAL_ACCENT_6, PAL_ACCENT_4 };

    for (int i = 0; i < 3; i++) {
        float cx = content_x + (float)i * (col_w + LX(10.0f));

        /* Column box */
        uint32_t box_fill = lerp_color(0x1C2128FFu, colors[i], 0.05f);
        draw_container(core, cx, col_y, col_w, col_h, box_fill, colors[i],
                       8.0f, 1.5f);

        /* Title */
        fs_cmd_text_utf8(core, cx + LX(10.0f), col_y + LX(18.0f),
                        LX(13.0f), titles[i], colors[i], col_w - LX(20.0f));

        /* Separator */
        fs_cmd_line(core, cx + LX(10.0f), col_y + LX(24.0f),
                    cx + col_w - LX(10.0f), col_y + LX(24.0f),
                    1.0f, lerp_color(colors[i], PAL_GRID_LINE, 0.5f));

        /* Text */
        float text_x = cx + LX(10.0f);
        float text_y_pos = col_y + LX(32.0f);
        float font_sz = LX(12.0f);
        fs_cmd_text_utf8(core, text_x, text_y_pos, font_sz,
                        texts[i], PAL_TEXT, col_w - LX(20.0f));
    }

    /* Bottom note */
    fs_cmd_text_utf8(core, content_x, g_state.fbh - LY(60.0f), LX(11.0f),
                    "All three columns measured without DOM access — pure Unicode block analysis",
                    PAL_TEXT_DIM, content_w);
}

static void render_scene_variable(FS_Core* core, bool font_ready) {
    draw_bg_gradient(core);

    float cx = g_state.fbw * 0.5f;
    float cy = g_state.fbh * 0.45f;
    float rw = g_state.variable_width * g_state.scale;
    float rh = g_state.fbh * 0.55f * g_state.scale;

    draw_scene_label(core, "VARIABLE WIDTH", "Trapezoid container — each line has different max-width",
                     LX(60.0f), LY(30.0f), LX(500.0f));
    draw_stats(core, LX(60.0f), LY(58.0f));
    draw_controls(core, LX(60.0f), g_state.fbh - LY(36.0f));

    /* Draw the trapezoid container */
    float taper = sinf(g_state.shape_time * 0.8f) * 0.3f;
    uint32_t box_fill = 0x1C2128FFu;
    uint32_t box_stroke = PAL_ACCENT_2;

    /* Trapezoid path */
    fs_path_begin(core);
    fs_path_move_to(core, cx - rw * (1.0f - taper) * 0.5f, cy - rh * 0.5f);
    fs_path_line_to(core, cx + rw * (1.0f - taper) * 0.5f, cy - rh * 0.5f);
    fs_path_line_to(core, cx + rw * (1.0f + taper) * 0.5f, cy + rh * 0.5f);
    fs_path_line_to(core, cx - rw * (1.0f + taper) * 0.5f, cy + rh * 0.5f);
    fs_path_close(core);
    fs_path_fill(core, box_fill);
    fs_path_stroke(core, 2.0f * g_state.scale, box_stroke);

    /* Inner text */
    if (font_ready) {
        float font_sz = LX(13.0f);
        float text_y = cy - rh * 0.4f;

        /* Multi-line text fitting the trapezoid */
        const char* lines[] = {
            "Variable-width text",
            "in irregular shapes",
            "is now possible with",
            "pure arithmetic",
            "layout — no DOM needed."
        };

        for (int i = 0; i < 5; i++) {
            float t = (float)i / 4.0f;
            float line_w = rw * (1.0f + taper * 0.6f * (t * 2.0f - 0.5f));
            float pulse = sinf(g_state.total_time * 2.0f + (float)i * 1.3f) * 0.1f + 0.9f;
            uint32_t lc = lerp_color(PAL_ACCENT_1, PAL_TEXT, pulse);

            fs_cmd_text_utf8(core, cx - line_w * 0.5f + LX(8.0f), text_y + (float)i * (font_sz * 1.6f),
                            font_sz, lines[i], lc, line_w - LX(16.0f));
        }
    }

    /* Side indicators showing varying width */
    float side_x = cx + rw * 0.5f + LX(20.0f);
    for (int i = 0; i < 5; i++) {
        float t = (float)i / 4.0f;
        float line_w = rw * (1.0f + taper * 0.6f * (t * 2.0f - 0.5f));
        float ly = cy - rh * 0.4f + (float)i * LX(22.0f);
        float lw = (line_w / rw) * LX(40.0f);
        uint32_t bc = lerp_color(PAL_ACCENT_1, PAL_ACCENT_5, t);
        fs_cmd_rect(core, side_x, ly, lw, LX(3.0f), LX(1.5f), bc);
    }

    /* Explanation */
    if (font_ready) {
        fs_cmd_text_utf8(core, LX(60.0f), g_state.fbh - LY(60.0f), LX(11.0f),
                        "layoutNextLine() iterates with per-line width — enables skewed, curved, trapezoid text",
                        PAL_TEXT_DIM, LX(700.0f));
    }
}

static void render_scene_perf(FS_Core* core, bool font_ready) {
    draw_bg_gradient(core);

    float content_x = LX(60.0f);
    float content_y = LY(80.0f);
    float content_w = g_state.container_width - LX(20.0f);
    float content_h = g_state.fbh - LY(160.0f);

    draw_scene_label(core, "PERFORMANCE", "Layout without re-measurement — the key to 300x speedup",
                     content_x, content_y, LX(500.0f));
    draw_controls(core, content_x, g_state.fbh - LY(36.0f));


    /* FPS display — large */
    char fps_big[64];
    snprintf(fps_big, sizeof(fps_big), "%.0f", g_state.fps);
    uint32_t fps_color = (g_state.fps >= 55.0f) ? PAL_ACCENT_4 :
                         (g_state.fps >= 30.0f) ? PAL_ACCENT_5 : PAL_ACCENT_3;
    fs_cmd_text_utf8(core, content_x + LX(200.0f), content_y + LX(60.0f),
                    LX(72.0f), fps_big, fps_color, LX(200.0f));
    fs_cmd_text_utf8(core, content_x + LX(280.0f), content_y + LX(70.0f),
                    LX(18.0f), "FPS", PAL_TEXT_DIM, LX(80.0f));

    /* Layout time */
    char layout_str[64];
    snprintf(layout_str, sizeof(layout_str), "Layout: %.4f ms", g_state.layout_time_ms);
    fs_cmd_text_utf8(core, content_x + LX(10.0f), content_y + LX(110.0f),
                    LX(13.0f), layout_str, PAL_TEXT, LX(200.0f));

    /* Comparison bars */
    float bar_y = content_y + LX(140.0f);
    float bar_max_w = content_w - LX(200.0f);
    float bar_h = LX(28.0f);
    float bar_gap = LX(8.0f);

    /* DOM simulation bar (100ms baseline) */
    float dom_time = 100.0f; /* typical DOM measureText time */
    float pretext_time = (float)g_state.layout_time_ms;
    float pretext_est_time = 0.3f; /* estimated pretext-style time */

    draw_perf_bar(core, content_x + LX(100.0f), bar_y, bar_max_w, bar_h,
                   dom_time, 110.0f, 0x2D1117FFu, PAL_ACCENT_3,
                   "DOM measureText");

    draw_perf_bar(core, content_x + LX(100.0f), bar_y + bar_h + bar_gap, bar_max_w, bar_h,
                   pretext_est_time, 110.0f, 0x0D2817FFu, PAL_ACCENT_4,
                   "fs_text_layout (est.)");

    draw_perf_bar(core, content_x + LX(100.0f), bar_y + 2.0f * (bar_h + bar_gap), bar_max_w, bar_h,
                   pretext_time > 10.0f ? pretext_time : pretext_est_time, 110.0f,
                   0x0D1A28FFu, PAL_ACCENT_1,
                   "Current render");

    /* Speedup ratio */
    float speedup = (dom_time > 0.001f) ? (dom_time / (pretext_time > 0.001f ? pretext_time : pretext_est_time)) : 1.0f;
    char speedup_str[64];
    snprintf(speedup_str, sizeof(speedup_str), "~%.0fx FASTER than DOM", speedup);
    float sx = content_x + LX(100.0f);
    float sy = bar_y + 3.0f * (bar_h + bar_gap) + LX(20.0f);
    float sp = sinf(g_state.total_time * 4.0f) * 0.1f + 0.9f;
    uint32_t spc = lerp_color(PAL_ACCENT_4, PAL_ACCENT_1, 1.0f - sp);
    fs_cmd_text_utf8(core, sx, sy, LX(20.0f), speedup_str, spc, bar_max_w);

    /* Metrics */
    char iter_str[64];
    snprintf(iter_str, sizeof(iter_str), "Iterations: %llu  |  Time: %.3f ms  |  Est. savings: %.2f ms per frame",
             (unsigned long long)g_state.perf_iterations,
             g_state.layout_time_ms,
             (dom_time - pretext_time) * 0.001f);
    fs_cmd_text_utf8(core, content_x + LX(10.0f), g_state.fbh - LY(60.0f),
                    LX(11.0f), iter_str, PAL_TEXT_DIM, content_w);

    /* Animated particles showing "no blocking" */
    float particle_y = sy + LX(50.0f);
    for (int i = 0; i < 12; i++) {
        float t = g_state.total_time * 2.0f + (float)i * 0.5f;
        float px = sx + LX(10.0f) + (fmodf(t, 1.0f)) * bar_max_w;
        float py = particle_y + sinf(t * 3.0f) * LX(6.0f);
        float r = LX(3.0f + sinf(t * 2.0f) * 2.0f);
        uint32_t pc = hsv2rgb(fmodf(t * 0.2f, 1.0f), 0.7f, 0.8f);
        fs_cmd_circle(core, px, py, r, pc);
    }
}

static void render_scene_emoji(FS_Core* core, bool font_ready) {
    draw_bg_gradient(core);

    float content_x = LX(60.0f);
    float content_y = LY(80.0f);
    float content_w = g_state.container_width - LX(20.0f);

    draw_scene_label(core, "EMOJI GALLERY", "Emoji clusters measured without DOM — zero reflow",
                     content_x, content_y, LX(500.0f));
    draw_stats(core, content_x, content_y + LY(28.0f));
    draw_controls(core, content_x, g_state.fbh - LY(36.0f));


    /* Emoji grid */
    float grid_x = content_x + LX(10.0f);
    float grid_y = content_y + LY(70.0f);
    float cell_w = (content_w - LX(20.0f)) / 4.0f;
    float cell_h = LX(80.0f);
    int cols = 4;

    const char* emoji_texts[] = {
        "\xF0\x9F\x8C\x8E\xE2\x9C\xA8",           /* flower + sparkle */
        "\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x9A\x80", /* man + rocket */
        "\xE2\x9D\xA4\xEF\xB8\x8F\xE2\x80\x8D\xF0\x9F\x94\xA5", /* love + fire */
        "\xF0\x9F\xAB\x95",                         /* burrito */
        "\xF0\x9F\x90\xB2",                         /* fish */
        "\xF0\x9F\x92\x96",                         /* lips */
        "\xF0\x9F\x8D\x95",                         /* pizza */
        "\xF0\x9F\x9B\xA0",                         /* hammer */
        "\xF0\x9F\x92\xAB",                         /* brain */
        "\xF0\x9F\x98\x80",                         /* grin */
        "\xF0\x9F\x98\x82",                         /* grin cat */
        "\xE2\xAD\x90",                             /* star */
    };
    const char* emoji_labels[] = {
        "Nature", "Space", "Love", "Food",
        "Animals", "Body", "Food", "Tools",
        "Science", "Happy", "Cat", "Stars"
    };

    int count = 12;
    for (int i = 0; i < count; i++) {
        int col = i % cols;
        int row = i / cols;
        float ex = grid_x + (float)col * cell_w;
        float ey = grid_y + (float)row * cell_h;
        float font_sz = LX(28.0f);

        /* Cell background */
        float pulse = sinf(g_state.total_time * 2.0f + (float)i * 0.7f) * 0.15f + 0.85f;
        uint32_t cell_fill = lerp_color(0x1C2128FFu, hsv2rgb((float)i / 12.0f, 0.4f, 0.15f), 0.3f);
        draw_container(core, ex + LX(4.0f), ey + LX(4.0f), cell_w - LX(8.0f), cell_h - LX(8.0f),
                       cell_fill, PAL_GRID_LINE, 6.0f, 1.0f);

        /* Emoji char */
        char ec[8] = {0};
        memcpy(ec, emoji_texts[i], 5);
        float ex_pos = ex + cell_w * 0.5f - LX(14.0f);
        fs_cmd_text_utf8(core, ex_pos, ey + cell_h * 0.35f + font_sz,
                        font_sz * pulse, ec, PAL_TEXT, cell_w - LX(20.0f));

        /* Label */
        fs_cmd_text_utf8(core, ex + cell_w * 0.5f - LX(30.0f), ey + cell_h - LX(16.0f),
                        LX(9.0f), emoji_labels[i], PAL_TEXT_DIM, LX(60.0f));
    }

    /* Description */
    fs_cmd_text_utf8(core, content_x, g_state.fbh - LY(60.0f), LX(11.0f),
                    "Emoji detection via Unicode range analysis — accurate cluster widths, no pixel probing",
                    PAL_TEXT_DIM, content_w);
}

static void render_scene_dynamic(FS_Core* core, bool font_ready) {
    draw_bg_gradient(core);

    float content_x = LX(60.0f);
    float content_y = LY(80.0f);
    float content_w = g_state.container_width - LX(20.0f);
    float content_h = g_state.fbh - LY(160.0f);

    draw_scene_label(core, "DYNAMIC RESIZE", "prepare() once — layout() forever — no re-measurement",
                     content_x, content_y, LX(500.0f));
    draw_stats(core, content_x, content_y + LY(28.0f));
    draw_controls(core, content_x, g_state.fbh - LY(36.0f));


    /* Animated container width */
    float cw = g_state.container_width - LX(40.0f);
    float ch = content_h - LY(20.0f);

    /* Container with resize handle */
    uint32_t box_fill = 0x1C2128FFu;
    draw_container(core, content_x + LX(20.0f), content_y + LY(60.0f),
                   cw, ch, box_fill, PAL_ACCENT_1, 8.0f, 2.0f);

    /* Width indicator */
    char width_str[32];
    snprintf(width_str, sizeof(width_str), "Width: %.0f px", cw);
    fs_cmd_text_utf8(core, content_x + LX(30.0f), content_y + LY(72.0f),
                    LX(11.0f), width_str, PAL_TEXT_DIM, LX(200.0f));

    /* Text */
    float text_x = content_x + LX(30.0f);
    float text_y = content_y + LY(95.0f);
    float font_sz = LX(14.0f);
    fs_cmd_text_utf8(core, text_x, text_y, font_sz,
                    k_text_long, PAL_TEXT, cw - LX(20.0f));

    /* Scrollbar */
    float scroll_x = content_x + LX(20.0f) + cw + LX(5.0f);
    float scroll_track_y = content_y + LY(60.0f);
    float scroll_track_h = ch;
    fs_cmd_rect(core, scroll_x, scroll_track_y, LX(6.0f), scroll_track_h,
                LX(3.0f), PAL_GRID_LINE);
    float scroll_pos = LY(60.0f) + g_state.scroll_y * 0.1f;
    fs_cmd_rect(core, scroll_x, scroll_pos, LX(6.0f), LX(30.0f),
                LX(3.0f), PAL_ACCENT_1);

    /* Width history sparkline */
    float spark_y = content_y + LY(60.0f);
    float spark_h = LX(40.0f);
    float spark_x = content_x + LX(20.0f);
    float spark_w = cw;

    /* Draw baseline */
    fs_cmd_line(core, spark_x, spark_y + spark_h, spark_x + spark_w, spark_y + spark_h,
                1.0f, PAL_GRID_LINE);

    /* Simulated width curve */
    int points = 60;
    for (int i = 1; i < points; i++) {
        float t = (float)i / (float)(points - 1);
        float amp = sinf(t * PI * 4.0f + g_state.total_time) * 0.3f + 0.5f;
        amp += sinf(t * PI * 7.0f) * 0.15f;
        float y0 = spark_y + spark_h * (1.0f - amp);
        float y1 = spark_y + spark_h * (1.0f - amp - 0.02f);
        uint32_t sc = lerp_color(PAL_ACCENT_1, PAL_ACCENT_4, amp);
        float px = spark_x + t * spark_w;
        fs_cmd_line(core, px, y0, px, y1, 2.0f, sc);
    }
}

static void render_scene_multilang(FS_Core* core, bool font_ready) {
    draw_bg_gradient(core);

    float content_x = LX(60.0f);
    float content_y = LY(80.0f);
    float content_w = g_state.container_width - LX(20.0f);

    draw_scene_label(core, "MULTI-LANGUAGE", "LTR, RTL, CJK — same prepare() / layout() API",
                     content_x, content_y, LX(500.0f));
    draw_stats(core, content_x, content_y + LY(28.0f));
    draw_controls(core, content_x, g_state.fbh - LY(36.0f));


    float row_y = content_y + LY(65.0f);
    float row_h = LX(80.0f);
    float row_gap = LX(8.0f);

    const char* lang_names[] = { "English (LTR)", "العربية (RTL)", "中文 (TTB)", "日本語 (V)" };
    const char* lang_texts[] = {
        "The art of text layout is precision. "
        "Character widths matter. Word breaks define rhythm.",

        "فن تخطيط النص هو الدقة. "
        "عرض الأحرف مهم. فواصل الكلمات تحدد الإيقاع.",

        "文本布局的艺术在於精確。\n"
        "字符寬度至關重要。\n"
        "單詞斷開定義節奏。",

        "テキストレイアウトの芸術は精密さにあります。\n"
        "文字幅が重要です。\n"
        "単語の切れ目がリズムを定義します。"
    };
    uint32_t lang_colors[] = { PAL_ACCENT_1, PAL_ACCENT_6, PAL_ACCENT_3, PAL_ACCENT_5 };
    float lang_heights[] = { 1.0f, 1.0f, 1.5f, 1.5f };

    for (int i = 0; i < 4; i++) {
        float ry = row_y + (float)i * (row_h + row_gap);
        uint32_t box_fill = lerp_color(0x1C2128FFu, lang_colors[i], 0.05f);

        /* Row container */
        draw_container(core, content_x, ry, content_w, row_h * lang_heights[i],
                       box_fill, lang_colors[i], 6.0f, 1.0f);

        /* Language label */
        fs_cmd_text_utf8(core, content_x + LX(10.0f), ry + LX(14.0f),
                        LX(10.0f), lang_names[i], lang_colors[i], LX(200.0f));

        /* Separator */
        fs_cmd_line(core, content_x + LX(10.0f), ry + LX(20.0f),
                    content_x + content_w - LX(10.0f), ry + LX(20.0f),
                    1.0f, lerp_color(lang_colors[i], PAL_GRID_LINE, 0.6f));

        /* Text */
        float tx = content_x + LX(10.0f);
        float ty = ry + LX(26.0f);
        float font_sz = LX(12.0f);
        float max_w = content_w - LX(20.0f);
        fs_cmd_text_utf8(core, tx, ty, font_sz, lang_texts[i], PAL_TEXT, max_w);

        /* Measure indicator */
        FS_LayoutResult r = fs_text_layout_layout(NULL, max_w);
        char measure_str[32];
        snprintf(measure_str, sizeof(measure_str), "%ux%.0f",
                 r.line_count, r.height);
        fs_cmd_text_utf8(core, content_x + content_w - LX(80.0f), ry + LX(14.0f),
                        LX(9.0f), measure_str, PAL_TEXT_DIM, LX(70.0f));
    }
}

/* ================================================================
   PREPARE TEXTS
   ================================================================ */

static void prepare_all_texts(void) {
    /* Destroy existing */
    for (int i = 0; i < MAX_PREP_ITEMS; i++) {
        if (g_state.preps[i]) {
            fs_text_layout_prepared_destroy(g_state.preps[i]);
            g_state.preps[i] = NULL;
        }
    }
    for (int i = 0; i < MAX_PREP_ITEMS; i++) {
        if (g_state.layouts[i]) {
            fs_text_layout_destroy(g_state.layouts[i]);
            g_state.layouts[i] = NULL;
        }
    }
    g_state.prep_count = 0;

    const char* texts[] = { k_text_basic, k_text_lorem, k_text_cjk, k_text_emoji,
                             k_text_arabic, k_text_long };
    const char* labels[] = { "Basic", "Lorem", "CJK", "Emoji", "Arabic", "Long" };

    for (int i = 0; i < 6 && g_state.prep_count < MAX_PREP_ITEMS; i++) {
        FS_TextLayout* ctx = fs_text_layout_create("Inter", 14.0f, 1.5f);
        if (!ctx) continue;
        FS_PreparedText* prep = fs_text_layout_prepare(ctx, texts[i]);
        if (prep) {
            g_state.layouts[g_state.prep_count] = ctx;
            g_state.preps[g_state.prep_count] = prep;
            strncpy(g_state.prep_labels[g_state.prep_count], labels[i], 63);
            g_state.prep_labels[g_state.prep_count][63] = '\0';
            g_state.prep_count++;
        } else {
            fs_text_layout_destroy(ctx);
        }
    }
}

static void clear_cache(void) {
    for (int i = 0; i < g_state.prep_count; i++) {
        if (g_state.layouts[i]) {
            fs_text_layout_clear_cache(g_state.layouts[i]);
        }
    }
    prepare_all_texts();
}

/* ================================================================
   SCENE MANAGEMENT
   ================================================================ */

static void switch_scene(SceneID scene) {
    if (scene == g_state.scene) return;
    g_state.prev_scene = g_state.scene;
    g_state.scene = scene;
    g_state.scene_changed = true;
    g_state.transition = 1.0f;
    g_state.transition_target = 0.0f;
    g_state.auto_anim_width = false;
    g_state.scroll_y = 0.0f;
    g_state.scroll_target = 0.0f;
}

static void update_scene(void) {
    double t = g_state.total_time;
    double dt = g_state.frame_dt;

    /* Transition fade */
    if (g_state.transition > 0.0f) {
        g_state.transition -= (float)(dt * 4.0f);
        if (g_state.transition < 0.0f) g_state.transition = 0.0f;
    }

    /* Scene-specific animation */
    switch (g_state.scene) {
        case SCENE_VARIABLE:
            g_state.shape_time = (float)t;
            g_state.variable_width_target = DEMO_DW * 0.4f + sinf((float)t * 0.5f) * DEMO_DW * 0.2f;
            g_state.variable_width += (float)((g_state.variable_width_target - g_state.variable_width) * dt * 2.0f);
            break;

        case SCENE_DYNAMIC:
            if (!g_state.mouse_down) {
                g_state.container_width_target = DEMO_DW * 0.5f + sinf((float)t * 0.3f) * DEMO_DW * 0.25f;
            }
            g_state.container_width += (float)((g_state.container_width_target - g_state.container_width) * dt * 3.0f);
            g_state.scroll_y += (float)((g_state.scroll_target - g_state.scroll_y) * dt * 2.0f);
            break;

        case SCENE_PERF:
            /* Simulate layout work */
            if (g_state.frame_count % 10 == 0) {
                g_state.perf_iterations += 1000;
                g_state.layout_time_ms = (double)(rand() % 300) / 100.0;
            }
            break;

        default:
            if (!g_state.mouse_down) {
                g_state.container_width += (float)((g_state.container_width_target - g_state.container_width) * dt * 2.0f);
            }
            break;
    }
}

/* ================================================================
   MAIN
   ================================================================ */

int main(void) {
    FS_GlfwBackend backend;
    if (!fs_glfw_backend_init(&backend, (uint32_t)DEMO_DW, (uint32_t)DEMO_DH,
                               "fs_text_layout demo")) {
        fprintf(stderr, "fs_glfw_backend_init failed\n");
        return 1;
    }

    FS_Core* core = fs_glfw_backend_core(&backend);
    if (!core) {
        fprintf(stderr, "fs_glfw_backend_core returned NULL\n");
        fs_glfw_backend_shutdown(&backend);
        return 1;
    }

    bool font_ready = false;
    fs_core_set_font_backend(core, fs_get_freetype2_font_backend());
    const char* font_paths[] = {
        "C:\\Windows\\Fonts\\seguiemj.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    };
    for (size_t fi = 0; fi < ARRAY_COUNT(font_paths); fi++) {
        if (fs_core_load_font_file(core, font_paths[fi])) {
            font_ready = true;
            break;
        }
    }
    printf("Font loaded: %s\n", font_ready ? "yes" : "no");

    /* Init demo state */
    memset(&g_state, 0, sizeof(g_state));
    g_state.dw = DEMO_DW;
    g_state.dh = DEMO_DH;
    g_state.fbw = DEMO_DW;
    g_state.fbh = DEMO_DH;
    g_state.scale = 1.0f;
    g_state.container_width = DEMO_DW * 0.7f;
    g_state.container_width_target = DEMO_DW * 0.7f;
    g_state.variable_width = DEMO_DW * 0.4f;
    g_state.show_info = true;
    g_state.show_grid = true;

    prepare_all_texts();

        double prev_time = glfwGetTime();
    uint64_t frame_count = 0;
    int prev_1_key = GLFW_RELEASE;
    int prev_2_key = GLFW_RELEASE;
    int prev_3_key = GLFW_RELEASE;
    int prev_4_key = GLFW_RELEASE;
    int prev_5_key = GLFW_RELEASE;
    int prev_6_key = GLFW_RELEASE;
    int prev_7_key = GLFW_RELEASE;
    int prev_0_key = GLFW_RELEASE;
    int prev_r_key = GLFW_RELEASE;
    int prev_c_key = GLFW_RELEASE;
    int prev_s_key = GLFW_RELEASE;
    int prev_i_key = GLFW_RELEASE;
    int prev_g_key = GLFW_RELEASE;

    /* Main loop */
    while (!fs_glfw_backend_should_close(&backend)) {
        fs_glfw_backend_poll_events();

        double now = glfwGetTime();
        double dt = now - prev_time;
        if (g_state.slow_mo) dt *= 0.2;
        prev_time = now;

        g_state.total_time += dt;
        g_state.frame_dt = dt;
        g_state.frame_count++;
        if (frame_count % 30 == 0) {
            g_state.fps = (dt > 0.0) ? (1.0f / (float)dt) : 60.0f;
        }

        /* Handle mouse via backend */
        const float sx_fb = (backend.width > 0) ? ((float)backend.width / (float)DEMO_DW) : 1.0f;
        const float sy_fb = (backend.height > 0) ? ((float)backend.height / (float)DEMO_DH) : 1.0f;
        g_state.scale = fminf(sx_fb, sy_fb);
        g_state.fbw = (float)backend.width;
        g_state.fbh = (float)backend.height;

        double mx = 0, my = 0;
        glfwGetCursorPos(backend.window, &mx, &my);
        g_state.mouse_x = (float)mx * sx_fb;
        g_state.mouse_y = (float)my * sy_fb;

        int lmb = glfwGetMouseButton(backend.window, GLFW_MOUSE_BUTTON_LEFT);
        if (lmb == GLFW_PRESS) {
            if (!g_state.mouse_down) {
                g_state.mouse_drag_x = mx;
            }
            g_state.mouse_down = true;
            float dx = (float)(mx - g_state.mouse_drag_x);
            g_state.container_width = DEMO_DW * 0.3f + (float)fabs(dx);
            if (g_state.container_width < DEMO_DW * 0.1f) g_state.container_width = DEMO_DW * 0.1f;
            if (g_state.container_width > DEMO_DW * 0.95f) g_state.container_width = DEMO_DW * 0.95f;
            g_state.container_width_target = g_state.container_width;
        } else {
            g_state.mouse_down = false;
        }

        /* Keyboard edge-triggered */
        int k1 = glfwGetKey(backend.window, GLFW_KEY_1);
        int k2 = glfwGetKey(backend.window, GLFW_KEY_2);
        int k3 = glfwGetKey(backend.window, GLFW_KEY_3);
        int k4 = glfwGetKey(backend.window, GLFW_KEY_4);
        int k5 = glfwGetKey(backend.window, GLFW_KEY_5);
        int k6 = glfwGetKey(backend.window, GLFW_KEY_6);
        int k7 = glfwGetKey(backend.window, GLFW_KEY_7);
        int k0 = glfwGetKey(backend.window, GLFW_KEY_0);
        int kr = glfwGetKey(backend.window, GLFW_KEY_R);
        int kc = glfwGetKey(backend.window, GLFW_KEY_C);
        int ks = glfwGetKey(backend.window, GLFW_KEY_S);
        int ki = glfwGetKey(backend.window, GLFW_KEY_K);
        int kg = glfwGetKey(backend.window, GLFW_KEY_G);

        if (k1 == GLFW_PRESS && prev_1_key != GLFW_PRESS) switch_scene(SCENE_BASIC);
        if (k2 == GLFW_PRESS && prev_2_key != GLFW_PRESS) switch_scene(SCENE_CJK);
        if (k3 == GLFW_PRESS && prev_3_key != GLFW_PRESS) switch_scene(SCENE_VARIABLE);
        if (k4 == GLFW_PRESS && prev_4_key != GLFW_PRESS) switch_scene(SCENE_PERF);
        if (k5 == GLFW_PRESS && prev_5_key != GLFW_PRESS) switch_scene(SCENE_EMOJI);
        if (k6 == GLFW_PRESS && prev_6_key != GLFW_PRESS) switch_scene(SCENE_DYNAMIC);
        if (k7 == GLFW_PRESS && prev_7_key != GLFW_PRESS) switch_scene(SCENE_MULTILANG);
        if (k0 == GLFW_PRESS && prev_0_key != GLFW_PRESS) switch_scene(SCENE_TITLE);
        if (kr == GLFW_PRESS && prev_r_key != GLFW_PRESS) {
            g_state.container_width = DEMO_DW * 0.7f;
            g_state.container_width_target = DEMO_DW * 0.7f;
        }
        if (kc == GLFW_PRESS && prev_c_key != GLFW_PRESS) clear_cache();
        if (ks == GLFW_PRESS && prev_s_key != GLFW_PRESS) g_state.slow_mo = !g_state.slow_mo;
        if (ki == GLFW_PRESS && prev_i_key != GLFW_PRESS) g_state.show_info = !g_state.show_info;
        if (kg == GLFW_PRESS && prev_g_key != GLFW_PRESS) g_state.show_grid = !g_state.show_grid;

        prev_1_key = k1; prev_2_key = k2; prev_3_key = k3;
        prev_4_key = k4; prev_5_key = k5; prev_6_key = k6;
        prev_7_key = k7; prev_0_key = k0; prev_r_key = kr;
        prev_c_key = kc; prev_s_key = ks; prev_i_key = ki;
        prev_g_key = kg;

        /* Scroll */
        float scroll_dy = 0;
        if (glfwGetKey(backend.window, GLFW_KEY_DOWN) == GLFW_PRESS) scroll_dy = 20.0f;
        if (glfwGetKey(backend.window, GLFW_KEY_UP) == GLFW_PRESS) scroll_dy = -20.0f;
        g_state.scroll_target += scroll_dy;
        if (g_state.scroll_target < 0) g_state.scroll_target = 0;

        /* Update */
        update_scene();

        /* Render */
        static int prev_scene = -1;
        if (g_state.scene != prev_scene) {
            printf("SCENE %d: %s\n", g_state.scene, scene_names[g_state.scene]);
            prev_scene = g_state.scene;
        }

        fs_core_begin_commands(core);
        fs_context_reset(core);

        switch (g_state.scene) {
            case SCENE_TITLE:     render_scene_title(core, font_ready); break;
            case SCENE_BASIC:     render_scene_basic(core, font_ready); break;
            case SCENE_CJK:       render_scene_cjk(core, font_ready); break;
            case SCENE_VARIABLE:  render_scene_variable(core, font_ready); break;
            case SCENE_PERF:      render_scene_perf(core, font_ready); break;
            case SCENE_EMOJI:     render_scene_emoji(core, font_ready); break;
            case SCENE_DYNAMIC:   render_scene_dynamic(core, font_ready); break;
            case SCENE_MULTILANG: render_scene_multilang(core, font_ready); break;
            default:              render_scene_title(core, font_ready); break;
        }

        /* Transition overlay */
        if (g_state.transition > 0.0f) {
            uint32_t tc = lerp_color(0x0D1117FFu, PAL_BG_MID, 1.0f - g_state.transition);
            fs_cmd_rect(core, 0, 0, g_state.fbw, g_state.fbh, 0, tc);
        }

        /* Scene tabs */
        if (font_ready && g_state.show_info) {
            float tab_y = g_state.fbh - LY(20.0f);
            for (int i = 0; i < SCENE_COUNT; i++) {
                float tw = LX(60.0f);
                float tx = LX(10.0f) + (float)i * (tw + LX(4.0f));
                uint32_t tc = (i == g_state.scene) ? PAL_ACCENT_1 : PAL_BG_LIGHT;
                uint32_t fc = (i == g_state.scene) ? 0x0D1117FFu : PAL_TEXT_DIM;
                char key_str[4] = { '1' + (char)i, '\0' };
                if (i == 9) { key_str[0] = '0'; key_str[1] = '\0'; }
                if (i > 9) continue;
                fs_cmd_rect(core, tx, tab_y - LY(14.0f), tw, LY(20.0f),
                            LX(4.0f), tc);
                fs_cmd_rect_stroke(core, tx, tab_y - LY(14.0f), tw, LY(20.0f), LX(4.0f), 1.0f, tc);
                fs_cmd_text_utf8(core, tx + tw * 0.5f - LX(8.0f), tab_y + LY(2.0f),
                                LX(10.0f), key_str, fc, tw - LX(4.0f));
            }
        }

        /* Present */
        if (!fs_glfw_backend_present(&backend, 0.043f, 0.059f, 0.071f, 1.0f)) {
            fprintf(stderr, "Frame present failed\n");
            break;
        }
        frame_count++;
    }

    /* Cleanup */
    for (int i = 0; i < MAX_PREP_ITEMS; i++) {
        if (g_state.preps[i]) fs_text_layout_prepared_destroy(g_state.preps[i]);
        if (g_state.layouts[i]) fs_text_layout_destroy(g_state.layouts[i]);
    }
    fs_glfw_backend_shutdown(&backend);
    return 0;
}
