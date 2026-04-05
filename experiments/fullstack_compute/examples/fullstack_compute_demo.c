#include "../impl/fullstack_glfw_backend.h"
#include "../impl/fullstack_stb_image_backend.h"
#include "../impl/fullstack_stb_font_backend.h"
#include "../impl/fullstack_freetype2_font_backend.h"
#include "../include/fullstack_core_debug.h"
#include "fullstack_emoji_embedded.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_COUNT(x) (sizeof(x) / sizeof((x)[0]))
#define GATE_C_WINDOW_FRAMES 180u

typedef struct DemoEmojiAsset {
    const char* utf8;
    uint32_t glyph_id;
    const char* png_name;
} DemoEmojiAsset;

static uint32_t rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t)a << 24u) | ((uint32_t)b << 16u) | ((uint32_t)g << 8u) | (uint32_t)r;
}

static const uint8_t k_sample_tga_2x2[] = {
    0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x02, 0x00, 0x18, 0x00,
    0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00,
    0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF
};

static void build_demo_pattern_rgba(uint8_t* out_rgba, uint32_t w, uint32_t h) {
    if (!out_rgba || w == 0u || h == 0u) {
        return;
    }
    for (uint32_t y = 0u; y < h; ++y) {
        for (uint32_t x = 0u; x < w; ++x) {
            const uint32_t idx = (y * w + x) * 4u;
            uint8_t r = 31u;
            uint8_t g = 76u;
            uint8_t b = 123u;
            uint8_t a = 255u;
            if (x < w / 2u && y < h / 2u) {
                r = 94u; g = 199u; b = 255u;
            } else if (x >= w / 2u && y >= h / 2u) {
                r = 255u; g = 180u; b = 93u;
            }
            // Cross guides.
            const int32_t dx = (int32_t)x - (int32_t)(w / 2u);
            const int32_t dy = (int32_t)y - (int32_t)(h / 2u);
            if (abs(dx) <= 1 || abs(dy) <= 1) {
                r = (uint8_t)((r + 245u) / 2u);
                g = (uint8_t)((g + 245u) / 2u);
                b = (uint8_t)((b + 245u) / 2u);
            }
            // Small highlight dot near lower-left.
            const int32_t px = (int32_t)x - (int32_t)(w / 4u);
            const int32_t py = (int32_t)y - (int32_t)((h * 3u) / 4u);
            if ((px * px + py * py) <= (int32_t)((w > h ? h : w) / 8u) * (int32_t)((w > h ? h : w) / 8u)) {
                r = 238u; g = 246u; b = 255u;
            }
            out_rgba[idx + 0u] = r;
            out_rgba[idx + 1u] = g;
            out_rgba[idx + 2u] = b;
            out_rgba[idx + 3u] = a;
        }
    }
}

static const DemoEmojiAsset k_emoji_assets[] = {
    {"\xF0\x9F\x8C\xBF", 0u, "emoji_u1f33f.png"},
    {"\xF0\x9F\x8D\x95", 1u, "emoji_u1f355.png"},
    {"\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x9A\x80", 2u, "emoji_u1f468_200d_1f680.png"},
    {"\xE2\x9C\xA8", 3u, "emoji_u2728.png"},
    {"\xE2\x9D\xA4\xEF\xB8\x8F\xE2\x80\x8D\xF0\x9F\x94\xA5", 4u, "emoji_u2764_200d_1f525.png"},
    {"\xF0\x9F\xAB\x95", 5u, "emoji_u1fad5.png"},
    {"\xE2\xAD\x90", 6u, "emoji_u2b50.png"},
    {"\xF0\x9F\x9B\xA0", 7u, "emoji_u1f6e0.png"},
    {"\xF0\x9F\x90\xB2", 8u, "emoji_u1f432.png"},
    {"\xF0\x9F\x92\xAB", 9u, "emoji_u1f4ab.png"},
};

static const char* clip_view_mode_name(int mode) {
    switch (mode) {
        case 0: return "RAW";
        case 1: return "CLIPPED";
        default: return "COMPARE";
    }
}

static const char* scene_mode_name(int mode) {
    switch (mode) {
        case 0: return "MAIN";
        case 1: return "MDN CLIP EXAMPLES";
        case 2: return "W3C API COVERAGE";
        default: return "MAIN";
    }
}

static const char* clip_failure_reason_name(FS_ClipFailureReason reason) {
    switch (reason) {
        case FS_CLIP_FAILURE_NONE: return "none";
        case FS_CLIP_FAILURE_INVALID_INPUT: return "invalid_input";
        case FS_CLIP_FAILURE_LAYER_EXHAUSTED: return "layer_exhausted";
        case FS_CLIP_FAILURE_EMPTY_PATH: return "empty_path";
        case FS_CLIP_FAILURE_INVALID_BOUNDS: return "invalid_bounds";
        case FS_CLIP_FAILURE_EDGE_ALLOC: return "edge_alloc";
        case FS_CLIP_FAILURE_JOB_ALLOC: return "job_alloc";
        default: return "unknown";
    }
}

static float draw_api_panel(
    FS_Core* core,
    bool font_ready,
    float x,
    float y,
    float w,
    float scale,
    const char* title,
    const char* status,
    uint32_t status_color,
    const char* const* lines,
    size_t line_count
) {
    const float panel_h = (66.0f + (float)line_count * 18.0f) * scale;
    fs_cmd_rect(core, x, y, w, panel_h, 14.0f * scale, rgba8(30, 44, 62, 214));
    fs_cmd_rect_stroke(core, x, y, w, panel_h, 14.0f * scale, 2.0f * scale, rgba8(170, 205, 238, 185));
    if (!font_ready) {
        return panel_h;
    }
    fs_cmd_text_utf8(
        core,
        x + 16.0f * scale,
        y + 24.0f * scale,
        15.0f * scale,
        title,
        rgba8(224, 238, 255, 236),
        w - 32.0f * scale
    );
    if (status && status[0] != '\0') {
        fs_cmd_text_utf8(
            core,
            x + w - 188.0f * scale,
            y + 24.0f * scale,
            11.0f * scale,
            status,
            status_color,
            172.0f * scale
        );
    }
    float line_y = y + 48.0f * scale;
    for (size_t i = 0; i < line_count; ++i) {
        fs_cmd_text_utf8(
            core,
            x + 18.0f * scale,
            line_y,
            11.5f * scale,
            lines[i],
            rgba8(194, 224, 250, 230),
            w - 36.0f * scale
        );
        line_y += 18.0f * scale;
    }
    return panel_h;
}

typedef struct DemoFlowLayout {
    float y;
    float max_bottom;
    float gap_y;
} DemoFlowLayout;

typedef struct DemoColumns {
    float x;
    float total_w;
    float gap_x;
    uint32_t count;
    float cell_w;
} DemoColumns;

static DemoFlowLayout demo_flow_begin(float start_y, float gap_y) {
    DemoFlowLayout flow;
    flow.y = start_y;
    flow.max_bottom = start_y;
    flow.gap_y = gap_y;
    return flow;
}

static float demo_flow_push(DemoFlowLayout* flow, float block_height) {
    if (!flow) {
        return 0.0f;
    }
    const float top = flow->y;
    const float bottom = top + fmaxf(0.0f, block_height);
    if (bottom > flow->max_bottom) {
        flow->max_bottom = bottom;
    }
    flow->y = bottom + flow->gap_y;
    return top;
}

static DemoColumns demo_columns_make(float x, float total_w, uint32_t count, float gap_x) {
    DemoColumns cols;
    cols.x = x;
    cols.total_w = total_w;
    cols.gap_x = gap_x;
    cols.count = count;
    cols.cell_w = total_w;
    if (count > 0u) {
        const float gaps = (float)(count - 1u) * gap_x;
        const float avail = total_w - gaps;
        cols.cell_w = (avail > 0.0f) ? (avail / (float)count) : 0.0f;
    }
    return cols;
}

static float demo_columns_x(const DemoColumns* cols, uint32_t index) {
    if (!cols || cols->count == 0u) {
        return 0.0f;
    }
    if (index >= cols->count) {
        index = cols->count - 1u;
    }
    return cols->x + (cols->cell_w + cols->gap_x) * (float)index;
}

static bool demo_path_circle(FS_Core* core, float cx, float cy, float radius) {
    if (!core || radius <= 0.0f) {
        return false;
    }
    const float k = 0.552284749831f * radius;
    if (!fs_path_move_to(core, cx + radius, cy)) {
        return false;
    }
    if (!fs_path_bezier_curve_to(core, cx + radius, cy + k, cx + k, cy + radius, cx, cy + radius)) {
        return false;
    }
    if (!fs_path_bezier_curve_to(core, cx - k, cy + radius, cx - radius, cy + k, cx - radius, cy)) {
        return false;
    }
    if (!fs_path_bezier_curve_to(core, cx - radius, cy - k, cx - k, cy - radius, cx, cy - radius)) {
        return false;
    }
    if (!fs_path_bezier_curve_to(core, cx + k, cy - radius, cx + radius, cy - k, cx + radius, cy)) {
        return false;
    }
    return fs_path_close(core);
}

static bool demo_path2d_circle(FS_Path2D* path, float cx, float cy, float radius) {
    if (!path || radius <= 0.0f) {
        return false;
    }
    const float k = 0.552284749831f * radius;
    if (!fs_path2d_move_to(path, cx + radius, cy)) {
        return false;
    }
    if (!fs_path2d_bezier_curve_to(path, cx + radius, cy + k, cx + k, cy + radius, cx, cy + radius)) {
        return false;
    }
    if (!fs_path2d_bezier_curve_to(path, cx - k, cy + radius, cx - radius, cy + k, cx - radius, cy)) {
        return false;
    }
    if (!fs_path2d_bezier_curve_to(path, cx - radius, cy - k, cx - k, cy - radius, cx, cy - radius)) {
        return false;
    }
    if (!fs_path2d_bezier_curve_to(path, cx + k, cy - radius, cx + radius, cy - k, cx + radius, cy)) {
        return false;
    }
    return fs_path2d_close(path);
}

static bool load_first_available_font(FS_Core* core, const char* const* candidates, size_t candidate_count) {
    for (size_t i = 0; i < candidate_count; ++i) {
        if (fs_core_load_font_file(core, candidates[i])) {
            printf("Loaded font: %s\n", candidates[i]);
            return true;
        }
    }
    return false;
}

static bool try_load_font(FS_Core* core, bool prefer_color_font) {
    static const char* k_primary_text_candidates[] = {
        "assets/NotoSerif-Medium.ttf",
        "../assets/NotoSerif-Medium.ttf",
        "../../assets/NotoSerif-Medium.ttf",
        "../../../assets/NotoSerif-Medium.ttf",
        ".lookme/use.gpu/public/fonts/Lato-Regular.ttf",
        "../.lookme/use.gpu/public/fonts/Lato-Regular.ttf",
        "../../.lookme/use.gpu/public/fonts/Lato-Regular.ttf",
        "../../../.lookme/use.gpu/public/fonts/Lato-Regular.ttf",
        ".lookme/use.gpu/public/fonts/FiraCode-Regular.otf",
        "../.lookme/use.gpu/public/fonts/FiraCode-Regular.otf",
        "../../.lookme/use.gpu/public/fonts/FiraCode-Regular.otf",
        "../../../.lookme/use.gpu/public/fonts/FiraCode-Regular.otf"
    };
    static const char* k_cjk_candidates[] = {
        "assets/NotoSerifSC-VF.ttf",
        "../assets/NotoSerifSC-VF.ttf",
        "../../assets/NotoSerifSC-VF.ttf",
        "../../../assets/NotoSerifSC-VF.ttf",
        "assets/PingFangSCMedium.ttf",
        "../assets/PingFangSCMedium.ttf",
        "../../assets/PingFangSCMedium.ttf",
        "../../../assets/PingFangSCMedium.ttf"
    };
    static const char* k_color_candidates[] = {
        "assets/NotoColorEmoji.ttf",
        "../assets/NotoColorEmoji.ttf",
        "../../assets/NotoColorEmoji.ttf",
        "../../../assets/NotoColorEmoji.ttf",
        ".lookme/use.gpu/public/fonts/NotoColorEmoji.ttf",
        "../.lookme/use.gpu/public/fonts/NotoColorEmoji.ttf",
        "../../.lookme/use.gpu/public/fonts/NotoColorEmoji.ttf",
        "../../../.lookme/use.gpu/public/fonts/NotoColorEmoji.ttf"
    };
    bool any_loaded = false;
    bool cjk_loaded = false;
    if (prefer_color_font) {
        // Fallback chain: primary Latin text, CJK text, then emoji color font.
        if (load_first_available_font(core, k_primary_text_candidates, ARRAY_COUNT(k_primary_text_candidates))) {
            any_loaded = true;
        }
        if (load_first_available_font(core, k_cjk_candidates, ARRAY_COUNT(k_cjk_candidates))) {
            cjk_loaded = true;
            any_loaded = true;
        }
        if (load_first_available_font(core, k_color_candidates, ARRAY_COUNT(k_color_candidates))) {
            any_loaded = true;
        }
    } else {
        if (load_first_available_font(core, k_primary_text_candidates, ARRAY_COUNT(k_primary_text_candidates))) {
            any_loaded = true;
        }
        if (load_first_available_font(core, k_cjk_candidates, ARRAY_COUNT(k_cjk_candidates))) {
            cjk_loaded = true;
            any_loaded = true;
        }
        // Optional fallback, keeps emoji visible without forcing color-font mode.
        if (load_first_available_font(core, k_color_candidates, ARRAY_COUNT(k_color_candidates))) {
            any_loaded = true;
        }
    }
    if (!cjk_loaded) {
        fprintf(stderr, "Warning: CJK fallback font not loaded; Chinese glyphs may render as tofu boxes\n");
    }
    if (!any_loaded) {
        fprintf(stderr, "Warning: no font loaded, text commands will be skipped\n");
    }
    return any_loaded;
}

static const DemoEmojiAsset* find_emoji_asset(uint32_t glyph_id) {
    for (size_t i = 0; i < ARRAY_COUNT(k_emoji_assets); ++i) {
        if (k_emoji_assets[i].glyph_id == glyph_id) {
            return &k_emoji_assets[i];
        }
    }
    return NULL;
}

static bool load_emoji_png_by_glyph(FS_Core* core, uint32_t image_font_id, uint32_t glyph_id) {
    if (!core || image_font_id == 0u) {
        return false;
    }
    const DemoEmojiAsset* asset = find_emoji_asset(glyph_id);
    if (!asset) {
        return false;
    }
    const FS_EmbeddedEmojiPng* embedded = fs_find_embedded_emoji_png(asset->png_name);
    if (!embedded) {
        fprintf(stderr, "Warning: embedded emoji missing for glyph=%u (%s)\n", glyph_id, asset->png_name);
        return false;
    }
    const bool ok =
        fs_core_load_image_glyph_png_memory(core, image_font_id, glyph_id, embedded->data, embedded->size);
    if (!ok) {
        fprintf(stderr, "Warning: failed uploading embedded emoji glyph=%u (%s)\n", glyph_id, asset->png_name);
    }
    return ok;
}

static void process_missing_emoji_glyphs(FS_Core* core, uint32_t emoji_font_id) {
    const uint32_t missing_count = fs_core_get_missing_image_glyph_count(core);
    if (missing_count == 0u) {
        return;
    }

    FS_MissingImageGlyph* list = (FS_MissingImageGlyph*)calloc((size_t)missing_count, sizeof(FS_MissingImageGlyph));
    if (!list) {
        return;
    }

    uint32_t copied = 0u;
    for (uint32_t i = 0u; i < missing_count; ++i) {
        FS_MissingImageGlyph item = {0};
        if (fs_core_get_missing_image_glyph(core, i, &item)) {
            list[copied++] = item;
        }
    }

    for (uint32_t i = 0u; i < copied; ++i) {
        const FS_MissingImageGlyph* item = &list[i];
        if (item->image_font_id != emoji_font_id) {
            continue;
        }
        (void)load_emoji_png_by_glyph(core, emoji_font_id, item->glyph_id);
    }

    fs_core_clear_missing_image_glyphs(core);
    free(list);
}

int main(void) {
    FS_GlfwBackend backend;
    if (!fs_glfw_backend_init(&backend, 1280, 720, "Fullstack Compute Pipeline Experiment")) {
        fprintf(stderr, "Failed to initialize fullstack compute backend\n");
        return 1;
    }

    FS_Core* core = fs_glfw_backend_core(&backend);
    printf("Rule clip view toggle: press V to cycle RAW / CLIPPED / COMPARE\n");
    printf("Scene toggle: press M to cycle MAIN / MDN CLIP EXAMPLES / W3C API COVERAGE\n");
    printf("Smoothing demo controls (W3C page): press I to toggle enabled, K to cycle quality\n");
    printf("Scroll: use mouse wheel / touchpad to pan page content\n");
    fs_core_set_image_backend(core, fs_get_stb_image_backend());
    const bool prefer_color_font = true;
    bool font_ready = false;
    if (!fs_core_set_font_backend(core, fs_get_freetype2_font_backend()) || !try_load_font(core, prefer_color_font)) {
        fs_core_set_font_backend(core, fs_get_stb_font_backend());
        font_ready = try_load_font(core, prefer_color_font);
    } else {
        font_ready = true;
    }
    printf("Image backend: %s\n", fs_core_get_image_backend_name(core));
    printf("Font backend: %s\n", fs_core_get_font_backend_name(core));

    uint32_t emoji_font_id = 0u;
    FS_ImageFontSequence emoji_sequences[ARRAY_COUNT(k_emoji_assets)];
    for (size_t i = 0; i < ARRAY_COUNT(k_emoji_assets); ++i) {
        emoji_sequences[i].utf8 = k_emoji_assets[i].utf8;
        emoji_sequences[i].glyph_id = k_emoji_assets[i].glyph_id;
    }

    const bool emoji_font_ready =
        fs_core_register_image_font(core, emoji_sequences, (uint32_t)ARRAY_COUNT(emoji_sequences), &emoji_font_id);
    if (!emoji_font_ready) {
        fprintf(stderr, "Warning: emoji image-font register failed\n");
    } else {
        // Preload part of the set so missing queue still demonstrates lazy loading.
        (void)load_emoji_png_by_glyph(core, emoji_font_id, 0u);
        (void)load_emoji_png_by_glyph(core, emoji_font_id, 1u);
        (void)load_emoji_png_by_glyph(core, emoji_font_id, 2u);
        (void)load_emoji_png_by_glyph(core, emoji_font_id, 3u);
    }

    FS_ImageHandle image_handle = {0};
    bool image_ready = fs_core_decode_image_memory(
        core,
        k_sample_tga_2x2,
        sizeof(k_sample_tga_2x2),
        &image_handle
    );
    if (!image_ready) {
        fprintf(stderr, "Warning: sample image decode/upload failed, fallback to default atlas tile\n");
    } else {
        uint8_t patch_rgba[16] = {
            255, 64, 64, 255,   64, 255, 64, 255,
            64, 64, 255, 255,   255, 255, 64, 255
        };
        if (!fs_core_put_image_data_rgba8(core, &image_handle, patch_rgba, sizeof(patch_rgba))) {
            fprintf(stderr, "Warning: putImageData experiment failed\n");
        } else {
            uint8_t readback_rgba[16] = {0};
            if (fs_core_get_image_data_rgba8(core, &image_handle, readback_rgba, sizeof(readback_rgba))) {
                printf(
                    "put/getImageData sample px0=(%u,%u,%u,%u)\n",
                    (unsigned)readback_rgba[0],
                    (unsigned)readback_rgba[1],
                    (unsigned)readback_rgba[2],
                    (unsigned)readback_rgba[3]
                );
            }
        }
    }

    FS_ImageHandle pattern_handle = {0};
    bool pattern_ready = false;
    {
        enum { kPatternW = 48, kPatternH = 48 };
        uint8_t pattern_rgba[kPatternW * kPatternH * 4];
        build_demo_pattern_rgba(pattern_rgba, kPatternW, kPatternH);
        pattern_ready = fs_core_upload_image_rgba8(core, pattern_rgba, kPatternW, kPatternH, &pattern_handle);
        if (!pattern_ready) {
            fprintf(stderr, "Warning: demo pattern upload failed; pattern examples disabled\n");
        }
    }
    {
        uint8_t canvas_data[16];
        if (fs_core_create_image_data_rgba8(2u, 2u, canvas_data, sizeof(canvas_data))) {
            canvas_data[0] = 240; canvas_data[1] = 80;  canvas_data[2] = 80;  canvas_data[3] = 255;
            canvas_data[4] = 80;  canvas_data[5] = 240; canvas_data[6] = 80;  canvas_data[7] = 255;
            canvas_data[8] = 80;  canvas_data[9] = 80;  canvas_data[10] = 240; canvas_data[11] = 255;
            canvas_data[12] = 250; canvas_data[13] = 250; canvas_data[14] = 120; canvas_data[15] = 255;
            (void)fs_core_put_canvas_image_data_rgba8(core, 4, 4, 2u, 2u, canvas_data, sizeof(canvas_data));
            uint8_t canvas_readback[16] = {0};
            if (fs_core_get_canvas_image_data_rgba8(core, 4, 4, 2u, 2u, canvas_readback, sizeof(canvas_readback))) {
                printf(
                    "canvas put/getImageData sample px0=(%u,%u,%u,%u)\n",
                    (unsigned)canvas_readback[0],
                    (unsigned)canvas_readback[1],
                    (unsigned)canvas_readback[2],
                    (unsigned)canvas_readback[3]
                );
            }
        }
    }

    uint32_t frame = 0;
    int rule_clip_view_mode = 2;
    int scene_mode = 0;
    int prev_scene_mode = scene_mode;
    int prev_v_state = GLFW_RELEASE;
    int prev_m_state = GLFW_RELEASE;
    int prev_i_state = GLFW_RELEASE;
    int prev_k_state = GLFW_RELEASE;
    int prev_lmb_state = GLFW_RELEASE;
    bool smoothing_demo_enabled = true;
    FS_ImageSmoothingQuality smoothing_demo_quality = FS_IMAGE_SMOOTHING_QUALITY_HIGH;
    float scroll_design_x = 0.0f;
    float scroll_design_y = 0.0f;
    float scene_content_h_design[3] = {940.0f, 1180.0f, 2400.0f};
    float miter_slider_value = 4.0f;
    bool miter_slider_dragging = false;
    bool clip_stress_enabled = true;
    uint64_t gatec_hist_est[GATE_C_WINDOW_FRAMES] = {0};
    uint64_t gatec_hist_waste[GATE_C_WINDOW_FRAMES] = {0};
    uint32_t gatec_hist_fail[GATE_C_WINDOW_FRAMES] = {0};
    uint32_t gatec_hist_valid_jobs[GATE_C_WINDOW_FRAMES] = {0};
    uint32_t gatec_hist_batches[GATE_C_WINDOW_FRAMES] = {0};
    uint32_t gatec_hist_oq[GATE_C_WINDOW_FRAMES] = {0};
    uint32_t gatec_hist_oq_clip[GATE_C_WINDOW_FRAMES] = {0};
    uint32_t gatec_hist_cursor = 0u;
    uint32_t gatec_hist_count = 0u;
    uint64_t gatec_sum_est = 0u;
    uint64_t gatec_sum_waste = 0u;
    uint32_t gatec_sum_fail = 0u;
    uint32_t gatec_sum_valid_jobs = 0u;
    uint32_t gatec_sum_batches = 0u;
    uint32_t gatec_sum_oq = 0u;
    uint32_t gatec_sum_oq_clip = 0u;
    FS_Path2D* mdn_region_path = fs_path2d_create();
    FS_Path2D* mdn_clip_circle_path = fs_path2d_create();
    FS_Path2D* mdn_clip_square_path = fs_path2d_create();
    FS_Path2D* api_addpath_source = fs_path2d_create();
    FS_Path2D* api_addpath_merged = fs_path2d_create();
    if (!mdn_region_path || !mdn_clip_circle_path || !mdn_clip_square_path || !api_addpath_source || !api_addpath_merged) {
        fprintf(stderr, "Failed to allocate Path2D demo objects\n");
        fs_path2d_destroy(mdn_region_path);
        fs_path2d_destroy(mdn_clip_circle_path);
        fs_path2d_destroy(mdn_clip_square_path);
        fs_path2d_destroy(api_addpath_source);
        fs_path2d_destroy(api_addpath_merged);
        fs_glfw_backend_shutdown(&backend);
        return 1;
    }
    while (!fs_glfw_backend_should_close(&backend)) {
        fs_glfw_backend_poll_events();
        int win_w = 0;
        int win_h = 0;
        glfwGetWindowSize(backend.window, &win_w, &win_h);
        double cursor_x = 0.0;
        double cursor_y = 0.0;
        glfwGetCursorPos(backend.window, &cursor_x, &cursor_y);
        const float sx_fb = (win_w > 0) ? ((float)backend.width / (float)win_w) : 1.0f;
        const float sy_fb = (win_h > 0) ? ((float)backend.height / (float)win_h) : 1.0f;
        const float mouse_fb_x = (float)cursor_x * sx_fb;
        const float mouse_fb_y = (float)cursor_y * sy_fb;
        const int lmb_state = glfwGetMouseButton(backend.window, GLFW_MOUSE_BUTTON_LEFT);
        float wheel_dx = 0.0f;
        float wheel_dy = 0.0f;
        fs_glfw_backend_take_scroll_delta(&backend, &wheel_dx, &wheel_dy);
        int v_state = glfwGetKey(backend.window, GLFW_KEY_V);
        if (v_state == GLFW_PRESS && prev_v_state != GLFW_PRESS) {
            rule_clip_view_mode = (rule_clip_view_mode + 1) % 3;
            printf("Rule clip view mode: %s\n", clip_view_mode_name(rule_clip_view_mode));
        }
        prev_v_state = v_state;
        int m_state = glfwGetKey(backend.window, GLFW_KEY_M);
        if (m_state == GLFW_PRESS && prev_m_state != GLFW_PRESS) {
            scene_mode = (scene_mode + 1) % 3;
            printf("Scene mode: %s\n", scene_mode_name(scene_mode));
        }
        prev_m_state = m_state;
        int i_state = glfwGetKey(backend.window, GLFW_KEY_I);
        if (i_state == GLFW_PRESS && prev_i_state != GLFW_PRESS) {
            smoothing_demo_enabled = !smoothing_demo_enabled;
            printf("Smoothing demo enabled: %s\n", smoothing_demo_enabled ? "ON" : "OFF");
        }
        prev_i_state = i_state;
        int k_state = glfwGetKey(backend.window, GLFW_KEY_K);
        if (k_state == GLFW_PRESS && prev_k_state != GLFW_PRESS) {
            smoothing_demo_quality = (FS_ImageSmoothingQuality)(((int)smoothing_demo_quality + 1) % 3);
            const char* q_name = "low";
            if (smoothing_demo_quality == FS_IMAGE_SMOOTHING_QUALITY_MEDIUM) {
                q_name = "medium";
            } else if (smoothing_demo_quality == FS_IMAGE_SMOOTHING_QUALITY_HIGH) {
                q_name = "high";
            }
            printf("Smoothing demo quality: %s\n", q_name);
        }
        prev_k_state = k_state;
        if (scene_mode != prev_scene_mode) {
            memset(gatec_hist_est, 0, sizeof(gatec_hist_est));
            memset(gatec_hist_waste, 0, sizeof(gatec_hist_waste));
            memset(gatec_hist_fail, 0, sizeof(gatec_hist_fail));
            memset(gatec_hist_valid_jobs, 0, sizeof(gatec_hist_valid_jobs));
            memset(gatec_hist_batches, 0, sizeof(gatec_hist_batches));
            memset(gatec_hist_oq, 0, sizeof(gatec_hist_oq));
            memset(gatec_hist_oq_clip, 0, sizeof(gatec_hist_oq_clip));
            gatec_hist_cursor = 0u;
            gatec_hist_count = 0u;
            gatec_sum_est = 0u;
            gatec_sum_waste = 0u;
            gatec_sum_fail = 0u;
            gatec_sum_valid_jobs = 0u;
            gatec_sum_batches = 0u;
            gatec_sum_oq = 0u;
            gatec_sum_oq_clip = 0u;
            prev_scene_mode = scene_mode;
        }
        fs_core_set_clip_layer_reuse_reserve(core, 0u);
        if (scene_mode != 1) {
            miter_slider_dragging = false;
        }
        if (emoji_font_ready) {
            process_missing_emoji_glyphs(core, emoji_font_id);
        }

        fs_core_begin_commands(core);
        fs_transform_reset(core);
        fs_style_reset(core);
        fs_path_begin(core);

        float t = (float)frame * 0.016f;
        const float design_w = 1280.0f;
        const float design_h = 720.0f;
        const float s = fminf((float)backend.width / design_w, (float)backend.height / design_h);
        const float inv_s = (s > 1e-6f) ? (1.0f / s) : 1.0f;
        const float scroll_step_px = 88.0f;
        scroll_design_x -= wheel_dx * (scroll_step_px * inv_s);
        scroll_design_y -= wheel_dy * (scroll_step_px * inv_s);
        const float design_content_w = 1280.0f;
        const float design_content_h = scene_content_h_design[(scene_mode >= 0 && scene_mode < 3) ? scene_mode : 0];
        const float visible_design_w = (float)backend.width * inv_s;
        const float visible_design_h = (float)backend.height * inv_s;
        const float scroll_margin_screen_px = 20.0f;
        const float scroll_margin_design = scroll_margin_screen_px * inv_s;
        const float max_scroll_x = fmaxf(0.0f, design_content_w - visible_design_w);
        const float max_scroll_y = fmaxf(0.0f, design_content_h - visible_design_h);
        if (scroll_design_x < 0.0f) {
            scroll_design_x = 0.0f;
        } else if (scroll_design_x > max_scroll_x) {
            scroll_design_x = max_scroll_x;
        }
        const float min_scroll_y = -scroll_margin_design;
        const float max_scroll_y_with_margin = max_scroll_y + scroll_margin_design;
        if (scroll_design_y < min_scroll_y) {
            scroll_design_y = min_scroll_y;
        } else if (scroll_design_y > max_scroll_y_with_margin) {
            scroll_design_y = max_scroll_y_with_margin;
        }
        const float ox = ((float)backend.width - design_w * s) * 0.5f - scroll_design_x * s;
        const float oy = ((float)backend.height - design_h * s) * 0.5f - scroll_design_y * s;
#define LX(v) (ox + (v) * s)
#define LY(v) (oy + (v) * s)
#define LS(v) ((v) * s)

        float cx = LX(620.0f + cosf(t * 0.7f) * 170.0f);
        float cy = LY(540.0f + sinf(t * 1.1f) * 85.0f);

        if (scene_mode == 1) {
            const float panel_w = LS(390.0f);
            const float panel_h = LS(250.0f);
            const float panel_y = LY(82.0f);
            const float gap_x = LS(25.0f);
            const float panel_x0 = LX(20.0f);
            const float panel_x1 = panel_x0 + panel_w + gap_x;
            const float panel_x2 = panel_x1 + panel_w + gap_x;
            bool conf_clip_1_ok = true;
            bool conf_clip_2_ok = true;
            bool conf_clip_3_ok = true;
            bool conf_restore_ok = true;

            if (font_ready) {
                fs_cmd_text_utf8(
                    core,
                    LX(24.0f),
                    LY(38.0f),
                    LS(22.0f),
                    "MDN CLIP EXAMPLES (Press M to return)",
                    rgba8(228, 240, 255, 240),
                    LS(1240.0f)
                );
            }

            // Panel backgrounds.
            fs_cmd_rect(core, panel_x0, panel_y, panel_w, panel_h, LS(16.0f), rgba8(32, 46, 62, 210));
            fs_cmd_rect(core, panel_x1, panel_y, panel_w, panel_h, LS(16.0f), rgba8(32, 46, 62, 210));
            fs_cmd_rect(core, panel_x2, panel_y, panel_w, panel_h, LS(16.0f), rgba8(32, 46, 62, 210));
            fs_cmd_rect_stroke(core, panel_x0, panel_y, panel_w, panel_h, LS(16.0f), LS(2.0f), rgba8(170, 205, 238, 185));
            fs_cmd_rect_stroke(core, panel_x1, panel_y, panel_w, panel_h, LS(16.0f), LS(2.0f), rgba8(170, 205, 238, 185));
            fs_cmd_rect_stroke(core, panel_x2, panel_y, panel_w, panel_h, LS(16.0f), LS(2.0f), rgba8(170, 205, 238, 185));

            // Shared canvas rect inside panels.
            const float cv_w = LS(230.0f);
            const float cv_h = LS(150.0f);
            const float cv_off_x = LS(18.0f);
            const float cv_off_y = LS(78.0f);

            // Example 1: beginPath+arc+clip then draw two rects.
            {
                const float ox = panel_x0 + cv_off_x;
                const float oy = panel_y + cv_off_y;
                fs_cmd_rect(core, ox, oy, cv_w, cv_h, LS(6.0f), rgba8(18, 23, 32, 230));
                fs_cmd_rect_stroke(core, ox, oy, cv_w, cv_h, LS(6.0f), LS(1.5f), rgba8(180, 210, 236, 190));
                fs_path_begin(core);
                demo_path_circle(core, ox + LS(100.0f), oy + LS(75.0f), LS(50.0f));
                fs_path_stroke(core, LS(2.0f), rgba8(240, 248, 255, 220));
                fs_state_save(core);
                fs_path_begin(core);
                demo_path_circle(core, ox + LS(100.0f), oy + LS(75.0f), LS(50.0f));
                const bool clip_ok = fs_clip_path(core);
                conf_clip_1_ok = conf_clip_1_ok && clip_ok;
                fs_cmd_rect(core, ox, oy, cv_w, cv_h, 0.0f, rgba8(70, 130, 255, 255));
                fs_cmd_rect(core, ox, oy, LS(100.0f), LS(100.0f), 0.0f, rgba8(255, 172, 64, 255));
                conf_restore_ok = conf_restore_ok && fs_state_restore(core);
                if (font_ready) {
                    fs_cmd_text_utf8(core, panel_x0 + LS(16.0f), panel_y + LS(30.0f), LS(15.0f), "MDN #1 arc() + clip()", rgba8(220, 236, 255, 235), LS(360.0f));
                }
            }

            // Example 2: Path2D rect+rect with evenodd clip.
            {
                const float ox = panel_x1 + cv_off_x;
                const float oy = panel_y + cv_off_y;
                fs_cmd_rect(core, ox, oy, cv_w, cv_h, LS(6.0f), rgba8(18, 23, 32, 230));
                fs_cmd_rect_stroke(core, ox, oy, cv_w, cv_h, LS(6.0f), LS(1.5f), rgba8(180, 210, 236, 190));
                fs_path2d_reset(mdn_region_path);
                fs_path2d_rect(mdn_region_path, ox + LS(80.0f), oy + LS(10.0f), LS(20.0f), LS(130.0f));
                fs_path2d_rect(mdn_region_path, ox + LS(40.0f), oy + LS(50.0f), LS(100.0f), LS(50.0f));
                fs_path_stroke_path2d(core, mdn_region_path, LS(2.0f), rgba8(240, 248, 255, 220));
                fs_state_save(core);
                const bool clip_ok = fs_clip_path2d_with_fill_rule(core, mdn_region_path, FS_FILL_RULE_EVENODD);
                conf_clip_2_ok = conf_clip_2_ok && clip_ok;
                fs_cmd_rect(core, ox, oy, cv_w, cv_h, 0.0f, rgba8(70, 130, 255, 255));
                conf_restore_ok = conf_restore_ok && fs_state_restore(core);
                if (font_ready) {
                    fs_cmd_text_utf8(core, panel_x1 + LS(16.0f), panel_y + LS(30.0f), LS(15.0f), "MDN #2 clip(region, evenodd)", rgba8(220, 236, 255, 235), LS(360.0f));
                }
            }

            // Example 3: sequential clip intersection (circle then square).
            {
                const float ox = panel_x2 + cv_off_x;
                const float oy = panel_y + cv_off_y;
                fs_cmd_rect(core, ox, oy, cv_w, cv_h, LS(6.0f), rgba8(18, 23, 32, 230));
                fs_cmd_rect_stroke(core, ox, oy, cv_w, cv_h, LS(6.0f), LS(1.5f), rgba8(180, 210, 236, 190));
                fs_path2d_reset(mdn_clip_circle_path);
                demo_path2d_circle(mdn_clip_circle_path, ox + LS(150.0f), oy + LS(75.0f), LS(75.0f));
                fs_path2d_reset(mdn_clip_square_path);
                fs_path2d_rect(mdn_clip_square_path, ox + LS(85.0f), oy + LS(10.0f), LS(130.0f), LS(130.0f));
                fs_state_save(core);
                const bool clip_a_ok = fs_clip_path2d(core, mdn_clip_circle_path);
                const bool clip_b_ok = clip_a_ok ? fs_clip_path2d(core, mdn_clip_square_path) : false;
                conf_clip_3_ok = conf_clip_3_ok && clip_a_ok && clip_b_ok;
                fs_cmd_rect(core, ox, oy, cv_w, cv_h, 0.0f, rgba8(70, 130, 255, 255));
                conf_restore_ok = conf_restore_ok && fs_state_restore(core);
                if (font_ready) {
                    fs_cmd_text_utf8(core, panel_x2 + LS(16.0f), panel_y + LS(30.0f), LS(15.0f), "MDN #3 clip(circle); clip(square)", rgba8(220, 236, 255, 235), LS(360.0f));
                }
            }

            if (font_ready) {
                char conformance_line[200];
                snprintf(
                    conformance_line,
                    sizeof(conformance_line),
                    "Conformance: clip#1=%s clip#2(evenodd)=%s clip#3(intersection)=%s save/restore=%s",
                    conf_clip_1_ok ? "PASS" : "FAIL",
                    conf_clip_2_ok ? "PASS" : "FAIL",
                    conf_clip_3_ok ? "PASS" : "FAIL",
                    conf_restore_ok ? "PASS" : "FAIL"
                );
                fs_cmd_text_utf8(
                    core,
                    LX(24.0f),
                    LY(66.0f),
                    LS(12.0f),
                    conformance_line,
                    rgba8(188, 226, 252, 230),
                    LS(1232.0f)
                );
            }

            // Text state panel: textAlign + textBaseline anchor behavior.
            {
                const float tx = panel_x0;
                const float ty = panel_y + panel_h + LS(20.0f);
                const float tw = panel_w * 3.0f + gap_x * 2.0f;
                const float th = LS(250.0f);
                fs_cmd_rect(core, tx, ty, tw, th, LS(14.0f), rgba8(30, 44, 62, 208));
                fs_cmd_rect_stroke(core, tx, ty, tw, th, LS(14.0f), LS(2.0f), rgba8(170, 205, 238, 185));
                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        tx + LS(16.0f),
                        ty + LS(24.0f),
                        LS(15.0f),
                        "TEXT STATE: textAlign + textBaseline anchors",
                        rgba8(220, 236, 255, 235),
                        tw - LS(32.0f)
                    );
                }

                const float a0x = tx + LS(180.0f);
                const float a0y = ty + LS(86.0f);
                const float a1x = tx + tw * 0.5f;
                const float a1y = ty + LS(124.0f);
                const float a2x = tx + tw - LS(180.0f);
                const float a2y = ty + LS(162.0f);
                const float a3x = tx + tw * 0.5f;
                const float a3y = ty + th - LS(36.0f);
                const float mark_len = LS(34.0f);

                fs_cmd_line(core, a0x - mark_len, a0y, a0x + mark_len, a0y, LS(2.0f), rgba8(255, 192, 132, 220));
                fs_cmd_line(core, a0x, a0y - mark_len, a0x, a0y + mark_len, LS(2.0f), rgba8(255, 192, 132, 220));
                fs_cmd_line(core, a1x - mark_len, a1y, a1x + mark_len, a1y, LS(2.0f), rgba8(148, 235, 255, 220));
                fs_cmd_line(core, a1x, a1y - mark_len, a1x, a1y + mark_len, LS(2.0f), rgba8(148, 235, 255, 220));
                fs_cmd_line(core, a2x - mark_len, a2y, a2x + mark_len, a2y, LS(2.0f), rgba8(195, 255, 154, 220));
                fs_cmd_line(core, a2x, a2y - mark_len, a2x, a2y + mark_len, LS(2.0f), rgba8(195, 255, 154, 220));
                fs_cmd_line(core, a3x - mark_len, a3y, a3x + mark_len, a3y, LS(2.0f), rgba8(255, 156, 226, 220));
                fs_cmd_line(core, a3x, a3y - mark_len, a3x, a3y + mark_len, LS(2.0f), rgba8(255, 156, 226, 220));

                fs_style_set_text_align(core, FS_TEXT_ALIGN_START);
                fs_style_set_text_baseline(core, FS_TEXT_BASELINE_ALPHABETIC);
                fs_cmd_text_utf8(
                    core,
                    a0x,
                    a0y,
                    LS(22.0f),
                    "START + ALPHABETIC",
                    rgba8(255, 224, 172, 245),
                    LS(340.0f)
                );

                fs_style_set_text_align(core, FS_TEXT_ALIGN_CENTER);
                fs_style_set_text_baseline(core, FS_TEXT_BASELINE_MIDDLE);
                fs_cmd_text_utf8(
                    core,
                    a1x,
                    a1y,
                    LS(24.0f),
                    "CENTER + MIDDLE",
                    rgba8(186, 242, 255, 245),
                    LS(420.0f)
                );

                fs_style_set_text_align(core, FS_TEXT_ALIGN_END);
                fs_style_set_text_baseline(core, FS_TEXT_BASELINE_TOP);
                fs_cmd_text_utf8(
                    core,
                    a2x,
                    a2y,
                    LS(22.0f),
                    "END + TOP",
                    rgba8(212, 255, 186, 245),
                    LS(320.0f)
                );

                fs_style_set_text_align(core, FS_TEXT_ALIGN_CENTER);
                fs_style_set_text_baseline(core, FS_TEXT_BASELINE_BOTTOM);
                fs_cmd_text_utf8(
                    core,
                    a3x,
                    a3y,
                    LS(22.0f),
                    "CENTER + BOTTOM",
                    rgba8(255, 188, 236, 245),
                    LS(360.0f)
                );

                fs_style_set_text_align(core, FS_TEXT_ALIGN_START);
                fs_style_set_text_baseline(core, FS_TEXT_BASELINE_ALPHABETIC);
            }

            // Join state panel: MDN-style miterLimit polyline.
            {
                const float jx = panel_x0;
                const float jy = panel_y + panel_h + LS(282.0f);
                const float jw = panel_w * 3.0f + gap_x * 2.0f;
                const float jh = LS(248.0f);
                fs_cmd_rect(core, jx, jy, jw, jh, LS(12.0f), rgba8(28, 42, 58, 198));
                fs_cmd_rect_stroke(core, jx, jy, jw, jh, LS(12.0f), LS(2.0f), rgba8(164, 206, 242, 175));
                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        jx + LS(14.0f),
                        jy + LS(20.0f),
                        LS(13.0f),
                        "MDN MITER LIMIT POLYLINE (drag slider to inspect middle sample)",
                        rgba8(212, 234, 255, 235),
                        jw - LS(24.0f)
                    );
                }
                const float sample_margin_x = LS(20.0f);
                const float sample_gap = LS(14.0f);
                const float sample_y = jy + LS(36.0f);
                const float sample_h = LS(138.0f);
                const float sample_w = (jw - sample_margin_x * 2.0f - sample_gap * 2.0f) / 3.0f;

                const float slider_min = 1.0f;
                const float slider_max = 24.0f;
                const float slider_x = jx + LS(220.0f);
                const float slider_y = jy + jh - LS(34.0f);
                const float slider_w = jw - LS(440.0f);
                const float slider_h = LS(7.0f);
                const float thumb_r = LS(9.0f);

                const bool hover_slider =
                    mouse_fb_x >= slider_x - thumb_r &&
                    mouse_fb_x <= slider_x + slider_w + thumb_r &&
                    mouse_fb_y >= slider_y - thumb_r &&
                    mouse_fb_y <= slider_y + slider_h + thumb_r;
                if (lmb_state == GLFW_PRESS && prev_lmb_state != GLFW_PRESS && hover_slider) {
                    miter_slider_dragging = true;
                }
                if (lmb_state != GLFW_PRESS) {
                    miter_slider_dragging = false;
                }
                if (miter_slider_dragging) {
                    float t_slider = (mouse_fb_x - slider_x) / fmaxf(slider_w, 1e-6f);
                    if (t_slider < 0.0f) {
                        t_slider = 0.0f;
                    } else if (t_slider > 1.0f) {
                        t_slider = 1.0f;
                    }
                    miter_slider_value = slider_min + t_slider * (slider_max - slider_min);
                }
                if (miter_slider_value < slider_min) {
                    miter_slider_value = slider_min;
                } else if (miter_slider_value > slider_max) {
                    miter_slider_value = slider_max;
                }
                const float slider_t = (miter_slider_value - slider_min) / (slider_max - slider_min);
                const float thumb_x = slider_x + slider_t * slider_w;
                const float thumb_y = slider_y + slider_h * 0.5f;

                const float limits[3] = {1.6f, miter_slider_value, 20.0f};
                const uint32_t stroke_colors[3] = {
                    rgba8(255, 172, 122, 240),
                    rgba8(147, 230, 255, 240),
                    rgba8(178, 255, 170, 240)
                };
                const uint32_t frame_colors[3] = {
                    rgba8(255, 210, 180, 170),
                    rgba8(184, 236, 255, 170),
                    rgba8(206, 255, 196, 170)
                };
                const float mdn_w = 230.0f;
                const float mdn_h = 150.0f;
                for (int si = 0; si < 3; ++si) {
                    const float sx = jx + sample_margin_x + (float)si * (sample_w + sample_gap);
                    fs_cmd_rect(core, sx, sample_y, sample_w, sample_h, LS(8.0f), rgba8(18, 30, 44, 188));
                    fs_cmd_rect_stroke(core, sx, sample_y, sample_w, sample_h, LS(8.0f), LS(1.5f), frame_colors[si]);
                    const float ms = fminf(sample_w / mdn_w, sample_h / mdn_h);
                    const float mx = sx + (sample_w - mdn_w * ms) * 0.5f;
                    const float my = sample_y + (sample_h - mdn_h * ms) * 0.5f;

                    fs_cmd_rect_stroke(
                        core,
                        mx - 5.0f * ms,
                        my + 50.0f * ms,
                        160.0f * ms,
                        50.0f * ms,
                        0.0f,
                        fmaxf(LS(1.0f), 2.0f * ms),
                        rgba8(0, 153, 255, 220)
                    );

                    fs_style_set_line_cap(core, FS_LINE_CAP_BUTT);
                    fs_style_set_line_join(core, FS_LINE_JOIN_MITER);
                    fs_style_set_line_width(core, fmaxf(LS(1.0f), 10.0f * ms));
                    fs_style_set_miter_limit(core, limits[si]);

                    fs_path_begin(core);
                    fs_path_move_to(core, mx, my + 100.0f * ms);
                    for (int i = 0; i < 24; ++i) {
                        const float dy = (i % 2 == 0) ? 25.0f : -25.0f;
                        const float px = powf((float)i, 1.5f) * 2.0f;
                        fs_path_line_to(core, mx + px * ms, my + (75.0f + dy) * ms);
                    }
                    fs_path_stroke(core, 0.0f, stroke_colors[si]);
                }

                if (font_ready) {
                    char mid_label[64];
                    snprintf(mid_label, sizeof(mid_label), "miterLimit=%.2f", miter_slider_value);
                    const float tx0 = jx + sample_margin_x;
                    const float tx1 = tx0 + sample_w + sample_gap;
                    const float tx2 = tx1 + sample_w + sample_gap;
                    fs_cmd_text_utf8(core, tx0 + LS(8.0f), sample_y + sample_h + LS(16.0f), LS(11.0f), "miterLimit=1.6", rgba8(255, 208, 186, 230), sample_w - LS(16.0f));
                    fs_cmd_text_utf8(core, tx1 + LS(8.0f), sample_y + sample_h + LS(16.0f), LS(11.0f), mid_label, rgba8(188, 236, 255, 230), sample_w - LS(16.0f));
                    fs_cmd_text_utf8(core, tx2 + LS(8.0f), sample_y + sample_h + LS(16.0f), LS(11.0f), "miterLimit=20", rgba8(203, 255, 196, 230), sample_w - LS(16.0f));
                    fs_cmd_text_utf8(
                        core,
                        slider_x - LS(96.0f),
                        slider_y + LS(4.0f),
                        LS(11.0f),
                        "1",
                        rgba8(200, 220, 240, 220),
                        LS(20.0f)
                    );
                    fs_cmd_text_utf8(
                        core,
                        slider_x + slider_w + LS(10.0f),
                        slider_y + LS(4.0f),
                        LS(11.0f),
                        "24",
                        rgba8(200, 220, 240, 220),
                        LS(26.0f)
                    );
                }

                fs_cmd_rect(
                    core,
                    slider_x,
                    slider_y,
                    slider_w,
                    slider_h,
                    slider_h * 0.5f,
                    rgba8(92, 128, 166, 210)
                );
                fs_cmd_rect(
                    core,
                    slider_x,
                    slider_y,
                    slider_t * slider_w,
                    slider_h,
                    slider_h * 0.5f,
                    rgba8(134, 210, 255, 230)
                );
                fs_cmd_circle(core, thumb_x, thumb_y, thumb_r, rgba8(240, 250, 255, 245));
                fs_cmd_circle(
                    core,
                    thumb_x,
                    thumb_y,
                    thumb_r * 0.52f,
                    miter_slider_dragging ? rgba8(96, 198, 255, 255) : rgba8(68, 162, 230, 255)
                );
                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        slider_x + slider_w * 0.5f - LS(168.0f),
                        slider_y - LS(8.0f),
                        LS(11.0f),
                        "MDN polyline: drag slider (LMB) for live miterLimit",
                        rgba8(195, 226, 250, 220),
                        LS(336.0f)
                    );
                }

                fs_style_set_miter_limit(core, 10.0f);
                fs_style_set_line_cap(core, FS_LINE_CAP_ROUND);
                fs_style_set_line_join(core, FS_LINE_JOIN_MITER);
            }

            // Clip stress panel: high clip-path pressure + live diagnostics.
            {
                const float stress_x = panel_x0;
                const float stress_y = panel_y + panel_h + LS(546.0f);
                const float stress_w = panel_w * 3.0f + gap_x * 2.0f;
                const float stress_h = LS(214.0f);
                fs_cmd_rect(core, stress_x, stress_y, stress_w, stress_h, LS(12.0f), rgba8(28, 42, 58, 200));
                fs_cmd_rect_stroke(core, stress_x, stress_y, stress_w, stress_h, LS(12.0f), LS(2.0f), rgba8(164, 206, 242, 175));

                const float stress_canvas_x = stress_x + LS(12.0f);
                const float stress_canvas_y = stress_y + LS(36.0f);
                const float stress_canvas_w = stress_w * 0.56f;
                const float stress_canvas_h = stress_h - LS(48.0f);
                fs_cmd_rect(
                    core,
                    stress_canvas_x,
                    stress_canvas_y,
                    stress_canvas_w,
                    stress_canvas_h,
                    LS(8.0f),
                    rgba8(16, 26, 38, 220)
                );
                fs_cmd_rect_stroke(
                    core,
                    stress_canvas_x,
                    stress_canvas_y,
                    stress_canvas_w,
                    stress_canvas_h,
                    LS(8.0f),
                    LS(1.5f),
                    rgba8(150, 196, 236, 170)
                );

                if (clip_stress_enabled) {
                    const int stress_cols = 8;
                    const int stress_rows = 3;
                    const float gap = LS(4.0f);
                    const float cell_w = (stress_canvas_w - gap * (float)(stress_cols + 1)) / (float)stress_cols;
                    const float cell_h = (stress_canvas_h - gap * (float)(stress_rows + 1)) / (float)stress_rows;
                    FS_ClipDiagnostics pre_clip_diag = {0};
                    (void)fs_core_get_clip_diagnostics(core, &pre_clip_diag);
                    const uint32_t used_layers_pre = pre_clip_diag.layers_used_this_frame;
                    const uint32_t total_layers = pre_clip_diag.layer_capacity;
                    const uint32_t remaining_layers =
                        (total_layers > used_layers_pre) ? (total_layers - used_layers_pre) : 0u;
                    // Each stress cell currently uses two nested path clips.
                    const uint32_t full_clip_cell_budget = remaining_layers / 2u;
                    uint32_t clipped_cells = 0u;
                    for (int r = 0; r < stress_rows; ++r) {
                        for (int c = 0; c < stress_cols; ++c) {
                            const int idx = r * stress_cols + c;
                            const float bx = stress_canvas_x + gap + (float)c * (cell_w + gap);
                            const float by = stress_canvas_y + gap + (float)r * (cell_h + gap);
                            const float phase = t * 0.9f + (float)idx * 0.37f;
                            const float wob = sinf(phase) * cell_w * 0.08f;
                            const float radius = fmaxf(LS(1.0f), cell_h * 0.22f);
                            fs_cmd_rect(core, bx, by, cell_w, cell_h, radius, rgba8(22, 34, 49, 210));

                            const bool allow_full_clip = ((uint32_t)idx < full_clip_cell_budget);
                            if (allow_full_clip) {
                                fs_state_save(core);
                                fs_path_begin(core);
                                fs_path_move_to(core, bx + cell_w * 0.10f + wob, by + cell_h * 0.18f);
                                fs_path_quadratic_curve_to(
                                    core,
                                    bx + cell_w * 0.54f,
                                    by - cell_h * (0.18f + 0.12f * cosf(phase * 1.7f)),
                                    bx + cell_w * 0.90f - wob,
                                    by + cell_h * 0.24f
                                );
                                fs_path_quadratic_curve_to(
                                    core,
                                    bx + cell_w * (1.02f + 0.05f * sinf(phase * 1.3f)),
                                    by + cell_h * 0.56f,
                                    bx + cell_w * 0.76f,
                                    by + cell_h * 0.88f
                                );
                                fs_path_quadratic_curve_to(
                                    core,
                                    bx + cell_w * 0.44f,
                                    by + cell_h * (1.04f + 0.08f * cosf(phase * 0.9f)),
                                    bx + cell_w * 0.14f,
                                    by + cell_h * 0.78f
                                );
                                fs_path_quadratic_curve_to(
                                    core,
                                    bx - cell_w * 0.06f,
                                    by + cell_h * 0.48f,
                                    bx + cell_w * 0.10f + wob,
                                    by + cell_h * 0.18f
                                );
                                fs_path_close(core);
                                const bool outer_ok = fs_clip_path(core);

                                bool inner_ok = false;
                                if (outer_ok) {
                                    fs_path_begin(core);
                                    fs_path_round_rect(
                                        core,
                                        bx + cell_w * 0.20f,
                                        by + cell_h * 0.18f,
                                        cell_w * 0.60f,
                                        cell_h * 0.64f,
                                        fmaxf(LS(1.0f), cell_h * 0.24f)
                                    );
                                    inner_ok = fs_clip_path(core);
                                }

                                if (outer_ok && inner_ok) {
                                    clipped_cells += 1u;
                                    fs_cmd_rect(
                                        core,
                                        bx - cell_w * 0.12f,
                                        by - cell_h * 0.12f,
                                        cell_w * 1.24f,
                                        cell_h * 1.24f,
                                        fmaxf(LS(1.0f), cell_h * 0.18f),
                                        rgba8((uint8_t)(105 + idx * 5), (uint8_t)(150 + idx * 3), 255, 210)
                                    );
                                    fs_cmd_circle(
                                        core,
                                        bx + cell_w * (0.25f + 0.50f * (0.5f + 0.5f * sinf(phase * 1.4f))),
                                        by + cell_h * 0.52f,
                                        fmaxf(LS(1.0f), cell_h * 0.30f),
                                        rgba8(255, 194, 98, 208)
                                    );
                                    fs_cmd_line(
                                        core,
                                        bx + cell_w * 0.08f,
                                        by + cell_h * (0.20f + 0.15f * cosf(phase)),
                                        bx + cell_w * 0.92f,
                                        by + cell_h * (0.80f + 0.15f * sinf(phase * 1.1f)),
                                        fmaxf(LS(1.0f), cell_h * 0.20f),
                                        rgba8(255, 242, 210, 220)
                                    );
                                } else {
                                    fs_cmd_rect(
                                        core,
                                        bx + cell_w * 0.06f,
                                        by + cell_h * 0.06f,
                                        cell_w * 0.88f,
                                        cell_h * 0.88f,
                                        fmaxf(LS(1.0f), cell_h * 0.16f),
                                        rgba8(92, 110, 132, 170)
                                    );
                                }
                                fs_state_restore(core);
                            } else {
                                // Out of clip-layer budget this frame: explicit fallback to avoid stale clip artifacts.
                                fs_cmd_rect(
                                    core,
                                    bx + cell_w * 0.06f,
                                    by + cell_h * 0.06f,
                                    cell_w * 0.88f,
                                    cell_h * 0.88f,
                                    fmaxf(LS(1.0f), cell_h * 0.16f),
                                    rgba8(92, 110, 132, 170)
                                );
                            }
                        }
                    }
                    if (font_ready) {
                        char budget_line[128];
                        snprintf(
                            budget_line,
                            sizeof(budget_line),
                            "clip budget: %u/%u cells (layers left=%u of %u)",
                            (unsigned)clipped_cells,
                            (unsigned)(stress_cols * stress_rows),
                            (unsigned)remaining_layers,
                            (unsigned)total_layers
                        );
                        fs_cmd_text_utf8(
                            core,
                            stress_x + LS(14.0f),
                            stress_y + LS(34.0f),
                            LS(10.0f),
                            budget_line,
                            rgba8(176, 218, 244, 218),
                            stress_w - LS(28.0f)
                        );
                    }
                } else {
                    fs_cmd_rect(
                        core,
                        stress_canvas_x + LS(10.0f),
                        stress_canvas_y + LS(10.0f),
                        stress_canvas_w - LS(20.0f),
                        stress_canvas_h - LS(20.0f),
                        LS(6.0f),
                        rgba8(46, 58, 74, 195)
                    );
                }

                FS_ClipDiagnostics clip_diag = {0};
                (void)fs_core_get_clip_diagnostics(core, &clip_diag);
                const uint32_t reuse_reserve = fs_core_get_clip_layer_reuse_reserve(core);
                if (font_ready) {
                    const float diag_x = stress_canvas_x + stress_canvas_w + LS(14.0f);
                    const float diag_y = stress_y + LS(18.0f);
                    const float diag_w = stress_x + stress_w - diag_x - LS(12.0f);
                    const uint64_t est_px = clip_diag.dispatch_pixels_estimated_this_frame;
                    const uint64_t waste_px = clip_diag.dispatch_pixels_waste_this_frame;
                    const double waste_pct = (est_px > 0u) ? (100.0 * (double)waste_px / (double)est_px) : 0.0;
                    if (gatec_hist_count == GATE_C_WINDOW_FRAMES) {
                        const uint32_t evict = gatec_hist_cursor;
                        gatec_sum_est -= gatec_hist_est[evict];
                        gatec_sum_waste -= gatec_hist_waste[evict];
                        gatec_sum_fail -= gatec_hist_fail[evict];
                        gatec_sum_valid_jobs -= gatec_hist_valid_jobs[evict];
                        gatec_sum_batches -= gatec_hist_batches[evict];
                        gatec_sum_oq -= gatec_hist_oq[evict];
                        gatec_sum_oq_clip -= gatec_hist_oq_clip[evict];
                    } else {
                        gatec_hist_count += 1u;
                    }
                    gatec_hist_est[gatec_hist_cursor] = est_px;
                    gatec_hist_waste[gatec_hist_cursor] = waste_px;
                    gatec_hist_fail[gatec_hist_cursor] = clip_diag.failures_this_frame;
                    gatec_hist_valid_jobs[gatec_hist_cursor] = clip_diag.dispatch_valid_jobs_this_frame;
                    gatec_hist_batches[gatec_hist_cursor] = clip_diag.dispatch_batches_this_frame;
                    gatec_hist_oq[gatec_hist_cursor] = clip_diag.oriented_quad_commands_this_frame;
                    gatec_hist_oq_clip[gatec_hist_cursor] = clip_diag.oriented_quad_clipped_this_frame;
                    gatec_sum_est += est_px;
                    gatec_sum_waste += waste_px;
                    gatec_sum_fail += clip_diag.failures_this_frame;
                    gatec_sum_valid_jobs += clip_diag.dispatch_valid_jobs_this_frame;
                    gatec_sum_batches += clip_diag.dispatch_batches_this_frame;
                    gatec_sum_oq += clip_diag.oriented_quad_commands_this_frame;
                    gatec_sum_oq_clip += clip_diag.oriented_quad_clipped_this_frame;
                    gatec_hist_cursor = (gatec_hist_cursor + 1u) % GATE_C_WINDOW_FRAMES;

                    const double waste_pct_window =
                        (gatec_sum_est > 0u) ? (100.0 * (double)gatec_sum_waste / (double)gatec_sum_est) : 0.0;
                    const double waste_warn_threshold = 55.0;
                    const double waste_fail_threshold = 75.0;
                    const uint32_t oriented_window_guard = 60u;
                    int gate_c_status = 0; // 0=PASS,1=WARN,2=FAIL
                    if (clip_diag.failures_this_frame > 0u || clip_diag.last_failure_reason != FS_CLIP_FAILURE_NONE) {
                        gate_c_status = 2;
                    } else if (clip_diag.dispatch_valid_jobs_this_frame == 0u || clip_diag.dispatch_batches_this_frame == 0u) {
                        gate_c_status = 1;
                    } else if (waste_pct > waste_fail_threshold) {
                        gate_c_status = 2;
                    } else if (waste_pct > waste_warn_threshold) {
                        gate_c_status = 1;
                    }
                    const char* gate_c_name = (gate_c_status == 0) ? "PASS" : (gate_c_status == 1) ? "WARN" : "FAIL";
                    const uint32_t gate_c_color =
                        (gate_c_status == 0) ? rgba8(152, 236, 176, 230)
                        : (gate_c_status == 1) ? rgba8(255, 224, 138, 230)
                        : rgba8(255, 156, 156, 238);
                    int gate_c_window_status = 0; // 0=PASS,1=WARN,2=FAIL
                    if (gatec_sum_fail > 0u) {
                        gate_c_window_status = 2;
                    } else if (gatec_sum_valid_jobs == 0u || gatec_sum_batches == 0u) {
                        gate_c_window_status = 1;
                    } else if (waste_pct_window > waste_fail_threshold) {
                        gate_c_window_status = 2;
                    } else if (waste_pct_window > waste_warn_threshold) {
                        gate_c_window_status = 1;
                    }
                    if (gate_c_window_status < 2 &&
                        gatec_hist_count >= oriented_window_guard &&
                        gatec_sum_oq > 0u &&
                        gatec_sum_oq_clip == 0u) {
                        gate_c_window_status = 1;
                    }
                    const char* gate_c_window_name =
                        (gate_c_window_status == 0) ? "PASS" : (gate_c_window_status == 1) ? "WARN" : "FAIL";
                    const uint32_t gate_c_window_color =
                        (gate_c_window_status == 0) ? rgba8(164, 244, 186, 230)
                        : (gate_c_window_status == 1) ? rgba8(255, 230, 152, 230)
                        : rgba8(255, 166, 166, 236);
                    char line0[168];
                    char line1[168];
                    char line2[168];
                    char line3[168];
                    char line4[196];
                    char line5[196];
                    char line6[220];
                    char line7[220];
                    snprintf(
                        line0,
                        sizeof(line0),
                        "CLIP STRESS (AUTO-MANAGED)"
                    );
                    snprintf(
                        line1,
                        sizeof(line1),
                        "req=%u hit=%u jobs=%u reuse=%u fail=%u",
                        (unsigned)clip_diag.requests_this_frame,
                        (unsigned)clip_diag.cache_hits_this_frame,
                        (unsigned)clip_diag.jobs_enqueued_this_frame,
                        (unsigned)clip_diag.layer_reuses_this_frame,
                        (unsigned)clip_diag.failures_this_frame
                    );
                    snprintf(
                        line2,
                        sizeof(line2),
                        "layers=%u/%u reserve=%u oq=%u oqClip=%u",
                        (unsigned)clip_diag.layers_used_this_frame,
                        (unsigned)clip_diag.layer_capacity,
                        (unsigned)reuse_reserve,
                        (unsigned)clip_diag.oriented_quad_commands_this_frame,
                        (unsigned)clip_diag.oriented_quad_clipped_this_frame
                    );
                    snprintf(
                        line3,
                        sizeof(line3),
                        "last=%s seg=%u edge=%u",
                        clip_failure_reason_name(clip_diag.last_failure_reason),
                        (unsigned)clip_diag.last_failure_path_segments,
                        (unsigned)clip_diag.last_failure_edge_count
                    );
                    snprintf(
                        line4,
                        sizeof(line4),
                        "dispatch=%u validJobs=%u idealPx=%llu estPx=%llu",
                        (unsigned)clip_diag.dispatch_batches_this_frame,
                        (unsigned)clip_diag.dispatch_valid_jobs_this_frame,
                        (unsigned long long)clip_diag.dispatch_pixels_ideal_this_frame,
                        (unsigned long long)clip_diag.dispatch_pixels_estimated_this_frame
                    );
                    snprintf(
                        line5,
                        sizeof(line5),
                        "waste=%llu (%.1f%%) bucket=%u/%u/%u/%u/%u/%u",
                        (unsigned long long)clip_diag.dispatch_pixels_waste_this_frame,
                        waste_pct,
                        (unsigned)clip_diag.dispatch_bucket_jobs_this_frame[0],
                        (unsigned)clip_diag.dispatch_bucket_jobs_this_frame[1],
                        (unsigned)clip_diag.dispatch_bucket_jobs_this_frame[2],
                        (unsigned)clip_diag.dispatch_bucket_jobs_this_frame[3],
                        (unsigned)clip_diag.dispatch_bucket_jobs_this_frame[4],
                        (unsigned)clip_diag.dispatch_bucket_jobs_this_frame[5]
                    );
                    snprintf(
                        line6,
                        sizeof(line6),
                        "GateC=%s (target waste<=%.0f%%, hard-fail>%.0f%%, fail=%u)",
                        gate_c_name,
                        waste_warn_threshold,
                        waste_fail_threshold,
                        (unsigned)clip_diag.failures_this_frame
                    );
                    snprintf(
                        line7,
                        sizeof(line7),
                        "GateC(win %u f)=%s waste=%.1f%% failSum=%u oqClip=%u/%u",
                        (unsigned)gatec_hist_count,
                        gate_c_window_name,
                        waste_pct_window,
                        (unsigned)gatec_sum_fail,
                        (unsigned)gatec_sum_oq_clip,
                        (unsigned)gatec_sum_oq
                    );
                    fs_cmd_text_utf8(core, stress_x + LS(14.0f), stress_y + LS(20.0f), LS(13.0f), line0, rgba8(216, 236, 255, 235), stress_w - LS(28.0f));
                    fs_cmd_text_utf8(core, diag_x, diag_y + LS(16.0f), LS(11.0f), line1, rgba8(208, 232, 255, 230), diag_w);
                    fs_cmd_text_utf8(core, diag_x, diag_y + LS(38.0f), LS(11.0f), line2, rgba8(192, 228, 248, 230), diag_w);
                    fs_cmd_text_utf8(core, diag_x, diag_y + LS(60.0f), LS(11.0f), line3, rgba8(255, 208, 190, 230), diag_w);
                    fs_cmd_text_utf8(core, diag_x, diag_y + LS(82.0f), LS(10.0f), line4, rgba8(174, 220, 246, 225), diag_w);
                    fs_cmd_text_utf8(core, diag_x, diag_y + LS(102.0f), LS(10.0f), line5, rgba8(168, 214, 242, 220), diag_w);
                    fs_cmd_text_utf8(core, diag_x, diag_y + LS(122.0f), LS(9.5f), line6, gate_c_color, diag_w);
                    fs_cmd_text_utf8(core, diag_x, diag_y + LS(140.0f), LS(9.5f), line7, gate_c_window_color, diag_w);
                    fs_cmd_text_utf8(
                        core,
                        diag_x,
                        diag_y + LS(158.0f),
                        LS(9.0f),
                        "reserve is managed internally per scene",
                        rgba8(174, 210, 236, 215),
                        diag_w
                    );
                }
            }

            goto submit_frame;
        }

        if (scene_mode == 2) {
            static const char* k_state_transform_lines[] = {
                "save(), restore()",
                "translate(), rotate(), scale()",
                "transform(a, b, c, d, e, f)",
                "setTransform(...), resetTransform()",
                "state snapshots apply to subsequent draw commands"
            };
            static const char* k_path_lines[] = {
                "beginPath(), closePath()",
                "moveTo(), lineTo()",
                "quadraticCurveTo(), bezierCurveTo()",
                "arcTo(), rect(), roundRect()",
                "arc(), ellipse(), Path2D.addPath()",
                "fill() with nonzero/evenodd, stroke()"
            };
            static const char* k_clip_lines[] = {
                "clip() with current path",
                "clip(path2d), clip(path2d, evenodd)",
                "nested clip intersection + save/restore stack",
                "rect/roundRect analytic fast-path + generic mask fallback",
                "GPU clip-mask pipeline used for irregular path clips"
            };
            static const char* k_stroke_lines[] = {
                "lineWidth",
                "lineCap: butt/round/square",
                "lineJoin: miter/round/bevel",
                "miterLimit",
                "setLineDash(), lineDashOffset",
                "isPointInStroke() / isPointInPath()"
            };
            static const char* k_text_lines[] = {
                "fillText() UTF-8",
                "strokeText() (SDF stroke path)",
                "measureText() subset metrics",
                "textAlign + textBaseline + direction + letterSpacing + wordSpacing",
                "fontKerning subset: auto/normal/none",
                "textRendering subset: auto/optimizeSpeed/optimizeLegibility/geometricPrecision",
                "fontStretch subset: synthetic horizontal metrics scaling",
                "fontVariantCaps subset: ASCII small-caps approximation",
                "color glyph support: native color font + image-font fallback"
            };
            static const char* k_comp_shadow_lines[] = {
                "globalAlpha",
                "globalCompositeOperation common subset",
                "source-over, copy, lighter, destination-over",
                "source-in/out, destination-in/out, xor, source-atop, destination-atop",
                "shadowColor + shadowBlur + shadowOffsetX/Y (experiment subset)"
            };
            static const char* k_image_lines[] = {
                "drawImage() equivalent via fs_cmd_image_handle()",
                "imageSmoothingEnabled (ON/OFF) for image+pattern sampling",
                "imageSmoothingQuality subset: low/medium/high",
                "putImageData()/getImageData() atlas-handle subset",
                "put/get canvas-rect ImageData subset",
                "RGBA CPU read/write for atlas-backed images",
                "image draw participates in clip + alpha + composite pipeline"
            };

            const float page_h_design = fmaxf(1860.0f, scene_content_h_design[2]);
            fs_cmd_rect(core, LX(16.0f), LY(14.0f), LS(1248.0f), LS(page_h_design), LS(20.0f), rgba8(18, 28, 42, 215));
            fs_cmd_rect_stroke(core, LX(16.0f), LY(14.0f), LS(1248.0f), LS(page_h_design), LS(20.0f), LS(2.0f), rgba8(152, 196, 232, 185));
            if (font_ready) {
                fs_cmd_text_utf8(
                    core,
                    LX(28.0f),
                    LY(42.0f),
                    LS(24.0f),
                    "W3C CANVAS SEMANTIC API COVERAGE (Press M to switch scene)",
                    rgba8(228, 240, 255, 242),
                    LS(1224.0f)
                );
                fs_cmd_text_utf8(
                    core,
                    LX(28.0f),
                    LY(70.0f),
                    LS(12.0f),
                    "This page enumerates implemented Canvas 2D semantics in this experiment build. Non-listed APIs are not wired yet.",
                    rgba8(180, 215, 242, 228),
                    LS(1224.0f)
                );
            }

            const DemoColumns info_cols = demo_columns_make(LX(24.0f), LS(1228.0f), 2u, LS(20.0f));
            const float column_w = info_cols.cell_w;
            const float left_x = demo_columns_x(&info_cols, 0u);
            const float right_x = demo_columns_x(&info_cols, 1u);
            float left_y = LY(96.0f);
            float right_y = LY(96.0f);

            left_y += draw_api_panel(
                core,
                font_ready,
                left_x,
                left_y,
                column_w,
                s,
                "State + Transform",
                "IMPLEMENTED",
                rgba8(168, 236, 186, 235),
                k_state_transform_lines,
                ARRAY_COUNT(k_state_transform_lines)
            ) + LS(14.0f);
            left_y += draw_api_panel(
                core,
                font_ready,
                left_x,
                left_y,
                column_w,
                s,
                "Path Construction + Fill/Stroke",
                "IMPLEMENTED",
                rgba8(168, 236, 186, 235),
                k_path_lines,
                ARRAY_COUNT(k_path_lines)
            ) + LS(14.0f);
            left_y += draw_api_panel(
                core,
                font_ready,
                left_x,
                left_y,
                column_w,
                s,
                "Clip Semantics",
                "IMPLEMENTED",
                rgba8(168, 236, 186, 235),
                k_clip_lines,
                ARRAY_COUNT(k_clip_lines)
            ) + LS(14.0f);

            right_y += draw_api_panel(
                core,
                font_ready,
                right_x,
                right_y,
                column_w,
                s,
                "Stroke Styles",
                "IMPLEMENTED",
                rgba8(168, 236, 186, 235),
                k_stroke_lines,
                ARRAY_COUNT(k_stroke_lines)
            ) + LS(14.0f);
            right_y += draw_api_panel(
                core,
                font_ready,
                right_x,
                right_y,
                column_w,
                s,
                "Text Semantics",
                "PARTIAL",
                rgba8(255, 220, 150, 235),
                k_text_lines,
                ARRAY_COUNT(k_text_lines)
            ) + LS(14.0f);
            right_y += draw_api_panel(
                core,
                font_ready,
                right_x,
                right_y,
                column_w,
                s,
                "Compositing + Shadow",
                "PARTIAL",
                rgba8(255, 220, 150, 235),
                k_comp_shadow_lines,
                ARRAY_COUNT(k_comp_shadow_lines)
            ) + LS(14.0f);
            right_y += draw_api_panel(
                core,
                font_ready,
                right_x,
                right_y,
                column_w,
                s,
                "Image Semantics",
                "PARTIAL",
                rgba8(255, 220, 150, 235),
                k_image_lines,
                ARRAY_COUNT(k_image_lines)
            ) + LS(14.0f);

            const float samples_y = fmaxf(left_y, right_y) + LS(24.0f);
            const DemoColumns sample_cols = demo_columns_make(LX(24.0f), LS(1226.0f), 3u, LS(16.0f));
            const float sample_w = sample_cols.cell_w;
            const float sample_h = LS(248.0f);
            const float sample_gap = sample_cols.gap_x;
            const float sx0 = demo_columns_x(&sample_cols, 0u);
            const float sx1 = demo_columns_x(&sample_cols, 1u);
            const float sx2 = demo_columns_x(&sample_cols, 2u);
            DemoFlowLayout sample_flow = demo_flow_begin(samples_y, LS(18.0f));
            const float sy0 = demo_flow_push(&sample_flow, sample_h);
            const float sy1 = demo_flow_push(&sample_flow, sample_h);

            // Sample 1: state + transform.
            fs_cmd_rect(core, sx0, sy0, sample_w, sample_h, LS(12.0f), rgba8(25, 36, 52, 220));
            fs_cmd_rect_stroke(core, sx0, sy0, sample_w, sample_h, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
            if (font_ready) {
                fs_cmd_text_utf8(core, sx0 + LS(14.0f), sy0 + LS(22.0f), LS(13.0f), "Example: save/restore + transform()", rgba8(214, 234, 255, 236), sample_w - LS(24.0f));
            }
            {
                const float vx = sx0 + LS(18.0f);
                const float vy = sy0 + LS(42.0f);
                const float vw = sample_w - LS(36.0f);
                const float vh = sample_h - LS(58.0f);
                const float cxv = vx + vw * 0.5f;
                const float cyv = vy + vh * 0.5f;
                fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));
                fs_cmd_line(core, vx, cyv, vx + vw, cyv, LS(1.5f), rgba8(94, 128, 162, 180));
                fs_cmd_line(core, cxv, vy, cxv, vy + vh, LS(1.5f), rgba8(94, 128, 162, 180));
                fs_state_save(core);
                fs_translate(core, cxv, cyv);
                fs_rotate(core, sinf(t * 0.92f) * 0.75f);
                fs_scale(core, 1.28f, 0.72f);
                fs_cmd_rect(core, LS(-64.0f), LS(-32.0f), LS(128.0f), LS(64.0f), LS(12.0f), rgba8(255, 170, 110, 218));
                fs_cmd_rect_stroke(core, LS(-64.0f), LS(-32.0f), LS(128.0f), LS(64.0f), LS(12.0f), LS(3.0f), rgba8(255, 244, 214, 238));
                fs_state_restore(core);
            }

            // Sample 2: path fill/stroke + evenodd.
            fs_cmd_rect(core, sx1, sy0, sample_w, sample_h, LS(12.0f), rgba8(25, 36, 52, 220));
            fs_cmd_rect_stroke(core, sx1, sy0, sample_w, sample_h, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
            if (font_ready) {
                fs_cmd_text_utf8(core, sx1 + LS(14.0f), sy0 + LS(22.0f), LS(13.0f), "Example: beginPath/fill(evenodd)/stroke", rgba8(214, 234, 255, 236), sample_w - LS(24.0f));
            }
            {
                const float vx = sx1 + LS(18.0f);
                const float vy = sy0 + LS(42.0f);
                const float vw = sample_w - LS(36.0f);
                const float vh = sample_h - LS(58.0f);
                fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));
                fs_style_set_fill_rule(core, FS_FILL_RULE_EVENODD);
                fs_path_begin(core);
                fs_path_round_rect(core, vx + LS(18.0f), vy + LS(16.0f), vw - LS(36.0f), vh - LS(32.0f), LS(22.0f));
                fs_path_rect(core, vx + LS(98.0f), vy + LS(48.0f), LS(112.0f), LS(68.0f));
                (void)fs_path_fill(core, rgba8(255, 126, 160, 190));
                fs_style_set_fill_rule(core, FS_FILL_RULE_NONZERO);
                fs_path_begin(core);
                fs_path_move_to(core, vx + LS(34.0f), vy + LS(130.0f));
                fs_path_bezier_curve_to(core, vx + LS(110.0f), vy + LS(72.0f), vx + LS(208.0f), vy + LS(170.0f), vx + LS(306.0f), vy + LS(98.0f));
                fs_path_stroke(core, LS(5.0f), rgba8(128, 232, 198, 224));
            }

            // Sample 3: nested clip intersection.
            fs_cmd_rect(core, sx2, sy0, sample_w, sample_h, LS(12.0f), rgba8(25, 36, 52, 220));
            fs_cmd_rect_stroke(core, sx2, sy0, sample_w, sample_h, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
            if (font_ready) {
                fs_cmd_text_utf8(core, sx2 + LS(14.0f), sy0 + LS(22.0f), LS(13.0f), "Example: clip() + clip() intersection", rgba8(214, 234, 255, 236), sample_w - LS(24.0f));
            }
            {
                const float vx = sx2 + LS(18.0f);
                const float vy = sy0 + LS(42.0f);
                const float vw = sample_w - LS(36.0f);
                const float vh = sample_h - LS(58.0f);
                fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));
                fs_state_save(core);
                fs_path_begin(core);
                demo_path_circle(core, vx + vw * 0.50f, vy + vh * 0.50f, LS(74.0f));
                (void)fs_clip_path(core);
                fs_path_begin(core);
                fs_path_rect(core, vx + LS(132.0f), vy + LS(18.0f), LS(146.0f), LS(152.0f));
                (void)fs_clip_path(core);
                fs_cmd_rect(core, vx, vy, vw, vh, 0.0f, rgba8(88, 158, 255, 255));
                fs_cmd_rect(core, vx + LS(8.0f), vy + LS(10.0f), LS(136.0f), LS(112.0f), 0.0f, rgba8(255, 180, 82, 248));
                fs_state_restore(core);
                fs_path_begin(core);
                demo_path_circle(core, vx + vw * 0.50f, vy + vh * 0.50f, LS(74.0f));
                fs_path_stroke(core, LS(2.0f), rgba8(236, 246, 255, 210));
                fs_cmd_rect_stroke(core, vx + LS(132.0f), vy + LS(18.0f), LS(146.0f), LS(152.0f), 0.0f, LS(2.0f), rgba8(236, 246, 255, 210));
            }

            // Sample 4: stroke style matrix.
            fs_cmd_rect(core, sx0, sy1, sample_w, sample_h, LS(12.0f), rgba8(25, 36, 52, 220));
            fs_cmd_rect_stroke(core, sx0, sy1, sample_w, sample_h, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
            if (font_ready) {
                fs_cmd_text_utf8(core, sx0 + LS(14.0f), sy1 + LS(22.0f), LS(13.0f), "Example: lineJoin + miterLimit + dash", rgba8(214, 234, 255, 236), sample_w - LS(24.0f));
            }
            {
                const float vx = sx0 + LS(18.0f);
                const float vy = sy1 + LS(42.0f);
                const float vw = sample_w - LS(36.0f);
                fs_cmd_rect(core, vx, vy, vw, sample_h - LS(58.0f), LS(8.0f), rgba8(14, 22, 34, 230));
                fs_style_set_line_width(core, LS(10.0f));
                fs_style_set_line_cap(core, FS_LINE_CAP_BUTT);
                fs_style_set_line_join(core, FS_LINE_JOIN_MITER);
                fs_style_set_miter_limit(core, 8.0f);
                fs_path_begin(core);
                fs_path_move_to(core, vx + LS(20.0f), vy + LS(42.0f));
                fs_path_line_to(core, vx + LS(76.0f), vy + LS(14.0f));
                fs_path_line_to(core, vx + LS(132.0f), vy + LS(42.0f));
                fs_path_stroke(core, 0.0f, rgba8(255, 172, 110, 235));
                fs_style_set_line_join(core, FS_LINE_JOIN_ROUND);
                fs_path_begin(core);
                fs_path_move_to(core, vx + LS(150.0f), vy + LS(42.0f));
                fs_path_line_to(core, vx + LS(206.0f), vy + LS(14.0f));
                fs_path_line_to(core, vx + LS(262.0f), vy + LS(42.0f));
                fs_path_stroke(core, 0.0f, rgba8(128, 226, 255, 235));
                fs_style_set_line_join(core, FS_LINE_JOIN_BEVEL);
                fs_path_begin(core);
                fs_path_move_to(core, vx + LS(278.0f), vy + LS(42.0f));
                fs_path_line_to(core, vx + LS(334.0f), vy + LS(14.0f));
                fs_path_line_to(core, vx + LS(390.0f), vy + LS(42.0f));
                fs_path_stroke(core, 0.0f, rgba8(166, 242, 180, 235));
                {
                    const float dash_pattern[] = {LS(24.0f), LS(10.0f)};
                    fs_style_set_line_width(core, LS(8.0f));
                    fs_style_set_line_join(core, FS_LINE_JOIN_ROUND);
                    fs_style_set_dash(core, dash_pattern, (uint32_t)ARRAY_COUNT(dash_pattern), t * LS(42.0f));
                    fs_cmd_line(core, vx + LS(20.0f), vy + LS(124.0f), vx + vw - LS(20.0f), vy + LS(124.0f), 0.0f, rgba8(255, 236, 172, 235));
                    fs_style_clear_dash(core);
                }
            }

            // Sample 5: text baseline/align + metrics + strokeText.
            fs_cmd_rect(core, sx1, sy1, sample_w, sample_h, LS(12.0f), rgba8(25, 36, 52, 220));
            fs_cmd_rect_stroke(core, sx1, sy1, sample_w, sample_h, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
            if (font_ready) {
                fs_cmd_text_utf8(core, sx1 + LS(14.0f), sy1 + LS(22.0f), LS(13.0f), "Example: text state (direction/kerning/rendering/stretch/variantCaps)", rgba8(214, 234, 255, 236), sample_w - LS(24.0f));
            }
            if (font_ready) {
                const float vx = sx1 + LS(18.0f);
                const float vy = sy1 + LS(42.0f);
                const float vw = sample_w - LS(36.0f);
                const float vh = sample_h - LS(58.0f);
                const float ax = vx + vw * 0.5f;
                const float ay = vy + vh * 0.54f;
                fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));
                fs_cmd_line(core, ax - LS(34.0f), ay, ax + LS(34.0f), ay, LS(2.0f), rgba8(255, 192, 132, 220));
                fs_cmd_line(core, ax, ay - LS(34.0f), ax, ay + LS(34.0f), LS(2.0f), rgba8(255, 192, 132, 220));
                fs_style_set_text_align(core, FS_TEXT_ALIGN_CENTER);
                fs_style_set_text_baseline(core, FS_TEXT_BASELINE_MIDDLE);
                fs_style_set_line_width(core, LS(2.0f));
                fs_cmd_stroke_text_utf8(core, ax, ay, LS(30.0f), "Canvas", rgba8(24, 38, 58, 235), LS(240.0f), LS(2.0f));
                fs_cmd_text_utf8(core, ax, ay, LS(30.0f), "Canvas", rgba8(255, 232, 132, 245), LS(240.0f));
                FS_TextMetrics tm;
                if (fs_measure_text_utf8(core, LS(30.0f), "Canvas", LS(240.0f), &tm)) {
                    const float bx = ax - tm.actual_bounding_box_left;
                    const float by = ay - tm.actual_bounding_box_ascent;
                    const float bw = tm.actual_bounding_box_left + tm.actual_bounding_box_right;
                    const float bh = tm.actual_bounding_box_ascent + tm.actual_bounding_box_descent;
                    fs_cmd_rect_stroke(core, bx, by, bw, bh, LS(4.0f), LS(1.5f), rgba8(206, 232, 255, 210));
                }
                const float text_state_x = vx + LS(10.0f);
                const float text_state_y0 = vy + LS(18.0f);
                const float text_state_y1 = vy + LS(32.0f);
                const float text_state_y2 = vy + LS(46.0f);
                const float text_state_y3 = vy + LS(60.0f);
                const float text_state_y4 = vy + LS(74.0f);
                fs_style_set_text_align(core, FS_TEXT_ALIGN_START);
                fs_style_set_text_baseline(core, FS_TEXT_BASELINE_ALPHABETIC);
                fs_style_set_text_direction(core, FS_TEXT_DIRECTION_LTR);
                fs_style_set_font_kerning(core, FS_FONT_KERNING_AUTO);
                fs_style_set_text_rendering(core, FS_TEXT_RENDERING_AUTO);
                fs_cmd_text_utf8(
                    core,
                    text_state_x,
                    text_state_y0,
                    LS(10.0f),
                    "kerning:auto   AVATAR To WA",
                    rgba8(184, 236, 255, 236),
                    vw - LS(20.0f)
                );
                fs_style_set_font_kerning(core, FS_FONT_KERNING_NONE);
                fs_cmd_text_utf8(
                    core,
                    text_state_x,
                    text_state_y1,
                    LS(10.0f),
                    "kerning:none   AVATAR To WA",
                    rgba8(255, 214, 176, 236),
                    vw - LS(20.0f)
                );
                fs_style_set_font_kerning(core, FS_FONT_KERNING_AUTO);
                fs_style_set_text_rendering(core, FS_TEXT_RENDERING_GEOMETRIC_PRECISION);
                fs_cmd_text_utf8(
                    core,
                    text_state_x + LS(0.35f),
                    text_state_y2,
                    LS(10.0f),
                    "textRendering: geometricPrecision",
                    rgba8(194, 224, 250, 232),
                    vw - LS(20.0f)
                );
                fs_style_set_text_rendering(core, FS_TEXT_RENDERING_AUTO);
                fs_style_set_font_stretch(core, FS_FONT_STRETCH_CONDENSED);
                fs_cmd_text_utf8(
                    core,
                    text_state_x,
                    text_state_y3,
                    LS(10.0f),
                    "stretch: condensed   AVATAR",
                    rgba8(198, 232, 190, 232),
                    vw - LS(20.0f)
                );
                fs_style_set_font_stretch(core, FS_FONT_STRETCH_EXPANDED);
                fs_cmd_text_utf8(
                    core,
                    text_state_x,
                    text_state_y4,
                    LS(10.0f),
                    "stretch: expanded    AVATAR",
                    rgba8(218, 206, 255, 232),
                    vw - LS(20.0f)
                );
                fs_style_set_font_stretch(core, FS_FONT_STRETCH_NORMAL);
                fs_style_set_font_variant_caps(core, FS_FONT_VARIANT_CAPS_SMALL_CAPS);
                fs_cmd_text_utf8(
                    core,
                    text_state_x,
                    text_state_y4 + LS(14.0f),
                    LS(10.0f),
                    "variantCaps: small-caps  Canvas API",
                    rgba8(255, 226, 168, 232),
                    vw - LS(20.0f)
                );
                fs_style_set_font_variant_caps(core, FS_FONT_VARIANT_CAPS_NORMAL);
                const float dir_anchor_x = vx + vw * 0.5f;
                const float dir_y0 = vy + vh - LS(22.0f);
                const float dir_y1 = vy + vh - LS(6.0f);
                fs_cmd_line(
                    core,
                    dir_anchor_x,
                    dir_y0 - LS(14.0f),
                    dir_anchor_x,
                    dir_y1 + LS(2.0f),
                    LS(1.4f),
                    rgba8(186, 214, 238, 196)
                );
                fs_style_set_text_align(core, FS_TEXT_ALIGN_START);
                fs_style_set_text_baseline(core, FS_TEXT_BASELINE_ALPHABETIC);
                fs_style_set_text_direction(core, FS_TEXT_DIRECTION_LTR);
                fs_cmd_text_utf8(
                    core,
                    dir_anchor_x,
                    dir_y0,
                    LS(11.0f),
                    "start / ltr",
                    rgba8(186, 242, 255, 238),
                    LS(150.0f)
                );
                fs_style_set_text_direction(core, FS_TEXT_DIRECTION_RTL);
                fs_cmd_text_utf8(
                    core,
                    dir_anchor_x,
                    dir_y1,
                    LS(11.0f),
                    "start / rtl",
                    rgba8(255, 214, 172, 238),
                    LS(150.0f)
                );
                fs_style_set_text_direction(core, FS_TEXT_DIRECTION_LTR);
                fs_style_set_text_align(core, FS_TEXT_ALIGN_START);
                fs_style_set_text_baseline(core, FS_TEXT_BASELINE_ALPHABETIC);
                fs_style_set_font_kerning(core, FS_FONT_KERNING_AUTO);
                fs_style_set_text_rendering(core, FS_TEXT_RENDERING_AUTO);
                fs_style_set_font_stretch(core, FS_FONT_STRETCH_NORMAL);
                fs_style_set_font_variant_caps(core, FS_FONT_VARIANT_CAPS_NORMAL);
            }

            // Sample 6: composite + shadow + image draw.
            fs_cmd_rect(core, sx2, sy1, sample_w, sample_h, LS(12.0f), rgba8(25, 36, 52, 220));
            fs_cmd_rect_stroke(core, sx2, sy1, sample_w, sample_h, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
            if (font_ready) {
                fs_cmd_text_utf8(core, sx2 + LS(14.0f), sy1 + LS(22.0f), LS(13.0f), "Example: globalAlpha/composite/shadow/image", rgba8(214, 234, 255, 236), sample_w - LS(24.0f));
            }
            {
                const float vx = sx2 + LS(18.0f);
                const float vy = sy1 + LS(42.0f);
                const float vw = sample_w - LS(36.0f);
                const float vh = sample_h - LS(58.0f);
                fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));
                const float gap = LS(10.0f);
                const float cell_w = (vw - gap * 4.0f) / 3.0f;
                const float cell_h = LS(118.0f);
                const float row_y = vy + LS(12.0f);
                const float label_y = row_y + cell_h + LS(16.0f);
                const float x_shadow = vx + gap;
                const float x_comp = x_shadow + cell_w + gap;
                const float x_img = x_comp + cell_w + gap;

                // Shadow block
                fs_cmd_rect(core, x_shadow, row_y, cell_w, cell_h, LS(8.0f), rgba8(22, 34, 50, 240));
                fs_style_set_shadow_color(core, rgba8(0, 0, 0, 214));
                fs_style_set_shadow_blur(core, LS(8.0f));
                fs_style_set_shadow_offset(core, LS(6.0f), LS(5.0f));
                fs_cmd_rect(core, x_shadow + LS(14.0f), row_y + LS(18.0f), cell_w - LS(28.0f), cell_h - LS(36.0f), LS(10.0f), rgba8(255, 120, 170, 228));
                fs_style_set_shadow_offset(core, 0.0f, 0.0f);
                fs_style_set_shadow_blur(core, 0.0f);
                fs_style_set_shadow_color(core, 0u);

                // Composite block
                fs_cmd_rect(core, x_comp, row_y, cell_w, cell_h, LS(8.0f), rgba8(22, 34, 50, 240));
                const float ccx = x_comp + cell_w * 0.50f;
                const float ccy = row_y + cell_h * 0.52f;
                const float cr = fminf(cell_w, cell_h) * 0.28f;
                fs_cmd_circle(core, ccx - cr * 0.38f, ccy, cr, rgba8(255, 126, 136, 228));
                fs_style_set_global_alpha(core, 0.66f);
                fs_style_set_global_composite_operation(core, FS_GLOBAL_COMPOSITE_LIGHTER);
                fs_cmd_circle(core, ccx + cr * 0.38f, ccy, cr, rgba8(116, 210, 255, 236));
                fs_style_set_global_composite_operation(core, FS_GLOBAL_COMPOSITE_SOURCE_OVER);
                fs_style_set_global_alpha(core, 1.0f);

                // Image + putImageData block
                fs_cmd_rect(core, x_img, row_y, cell_w, cell_h, LS(8.0f), rgba8(22, 34, 50, 240));
                const float img_x = x_img + LS(10.0f);
                const float img_y = row_y + LS(10.0f);
                const float img_w = cell_w - LS(20.0f);
                const float img_h = LS(72.0f);
                if (image_ready) {
                    fs_cmd_image_handle(core, img_x, img_y, img_w, img_h, &image_handle, rgba8(255, 255, 255, 242));
                } else {
                    fs_cmd_rect(core, img_x, img_y, img_w, img_h, LS(6.0f), rgba8(108, 158, 212, 180));
                }
                {
                    enum { kIDW = 16, kIDH = 10 };
                    uint8_t canvas_px[kIDW * kIDH * 4];
                    for (uint32_t py = 0u; py < kIDH; ++py) {
                        for (uint32_t px = 0u; px < kIDW; ++px) {
                            const size_t idx = ((size_t)py * kIDW + (size_t)px) * 4u;
                            canvas_px[idx + 0u] = (uint8_t)(80u + px * 10u);
                            canvas_px[idx + 1u] = (uint8_t)(80u + py * 14u);
                            canvas_px[idx + 2u] = (uint8_t)(220u - px * 7u);
                            canvas_px[idx + 3u] = 255u;
                        }
                    }
                    const float id_x = x_img + LS(12.0f);
                    const float id_y = row_y + cell_h - LS(26.0f);
                    (void)fs_core_put_canvas_image_data_rgba8(
                        core,
                        (int32_t)id_x,
                        (int32_t)id_y,
                        kIDW,
                        kIDH,
                        canvas_px,
                        sizeof(canvas_px)
                    );
                    fs_cmd_rect_stroke(core, id_x, id_y, (float)kIDW, (float)kIDH, 0.0f, LS(1.1f), rgba8(235, 245, 255, 220));
                }

                if (font_ready) {
                    fs_cmd_text_utf8(core, x_shadow, label_y, LS(11.0f), "shadowColor + shadowBlur", rgba8(194, 224, 252, 234), cell_w);
                    fs_cmd_text_utf8(core, x_comp, label_y, LS(11.0f), "globalAlpha + lighter", rgba8(194, 224, 252, 234), cell_w);
                    fs_cmd_text_utf8(core, x_img, label_y, LS(11.0f), "drawImage + putImageData", rgba8(194, 224, 252, 234), cell_w);
                }
            }

            // Sample 7: Chinese UTF-8 text demo.
            {
                const float cx = sx0;
                const float cy = demo_flow_push(&sample_flow, LS(246.0f));
                const float cw = sample_w * 3.0f + sample_gap * 2.0f;
                const float ch = LS(246.0f);
                fs_cmd_rect(core, cx, cy, cw, ch, LS(12.0f), rgba8(25, 36, 52, 220));
                fs_cmd_rect_stroke(core, cx, cy, cw, ch, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        cx + LS(14.0f),
                        cy + LS(22.0f),
                        LS(13.0f),
                        "Example: Chinese UTF-8 text + baseline + measureText",
                        rgba8(214, 234, 255, 236),
                        cw - LS(24.0f)
                    );
                }

                if (font_ready) {
                    const float vx = cx + LS(18.0f);
                    const float vy = cy + LS(42.0f);
                    const float vw = cw - LS(36.0f);
                    const float vh = ch - LS(58.0f);
                    const float ax = vx + LS(18.0f);
                    const float ay0 = vy + LS(52.0f);
                    const float ay1 = vy + LS(110.0f);
                    const float ay2 = vy + LS(156.0f);
                    fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));
                    fs_cmd_line(core, vx + LS(8.0f), ay0, vx + vw - LS(8.0f), ay0, LS(1.5f), rgba8(255, 190, 120, 214));
                    fs_cmd_line(core, vx + LS(8.0f), ay1, vx + vw - LS(8.0f), ay1, LS(1.5f), rgba8(120, 220, 255, 214));
                    fs_cmd_line(core, vx + LS(8.0f), ay2, vx + vw - LS(8.0f), ay2, LS(1.5f), rgba8(176, 236, 170, 214));

                    const char* zh_line0 = "\xE4\xB8\xAD\xE6\x96\x87\xE5\x9F\xBA\xE7\xBA\xBF: \xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C" "Canvas 2D\xEF\xBC\x81";
                    const char* zh_line1 = "\xE6\x8F\x8F\xE8\xBE\xB9\xE6\x96\x87\xE6\x9C\xAC: \xE5\xAD\x97\xE5\xBD\xA2\xE8\xBD\xAE\xE5\xBB\x93";
                    const char* zh_line2 = "\xE6\xB7\xB7\xE6\x8E\x92: WCN \xE5\xBC\x95\xE6\x93\x8E abc123 \xF0\x9F\x98\x80\xF0\x9F\x9A\x80";

                    fs_style_set_text_align(core, FS_TEXT_ALIGN_START);
                    fs_style_set_text_baseline(core, FS_TEXT_BASELINE_ALPHABETIC);
                    fs_cmd_text_utf8(core, ax, ay0, LS(30.0f), zh_line0, rgba8(255, 234, 150, 242), vw - LS(28.0f));

                    fs_style_set_line_width(core, LS(2.0f));
                    fs_cmd_stroke_text_utf8(core, ax, ay1, LS(28.0f), zh_line1, rgba8(22, 36, 56, 238), vw - LS(28.0f), LS(2.0f));
                    fs_cmd_text_utf8(core, ax, ay1, LS(28.0f), zh_line1, rgba8(162, 236, 255, 246), vw - LS(28.0f));

                    fs_style_set_text_baseline(core, FS_TEXT_BASELINE_MIDDLE);
                    fs_cmd_text_utf8(core, ax, ay2, LS(22.0f), zh_line2, rgba8(190, 248, 188, 242), vw - LS(28.0f));

                    FS_TextMetrics tm_zh;
                    if (fs_measure_text_utf8(core, LS(22.0f), zh_line2, vw - LS(28.0f), &tm_zh)) {
                        const float mx = ax - tm_zh.actual_bounding_box_left;
                        const float my = ay2 - tm_zh.actual_bounding_box_ascent;
                        const float mw = tm_zh.actual_bounding_box_left + tm_zh.actual_bounding_box_right;
                        const float mh = tm_zh.actual_bounding_box_ascent + tm_zh.actual_bounding_box_descent;
                        fs_cmd_rect_stroke(core, mx, my, mw, mh, LS(4.0f), LS(1.2f), rgba8(210, 236, 255, 196));
                    }

                    fs_style_set_text_align(core, FS_TEXT_ALIGN_START);
                    fs_style_set_text_baseline(core, FS_TEXT_BASELINE_ALPHABETIC);
                }
            }

            // Sample 8: Path2D addPath/addPath(withTransform) + hit-test.
            {
                const float cx = sx0;
                const float cy = demo_flow_push(&sample_flow, LS(222.0f));
                const float cw = sample_w * 3.0f + sample_gap * 2.0f;
                const float ch = LS(222.0f);
                fs_cmd_rect(core, cx, cy, cw, ch, LS(12.0f), rgba8(25, 36, 52, 220));
                fs_cmd_rect_stroke(core, cx, cy, cw, ch, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        cx + LS(14.0f),
                        cy + LS(22.0f),
                        LS(13.0f),
                        "Example: Path2D.addPath() + addPath(path, transform) + hit-test",
                        rgba8(214, 234, 255, 236),
                        cw - LS(24.0f)
                    );
                }

                const float vx = cx + LS(18.0f);
                const float vy = cy + LS(42.0f);
                const float vw = cw - LS(36.0f);
                const float vh = ch - LS(58.0f);
                fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));

                const float src_cx = vx + vw * 0.34f;
                const float src_cy = vy + vh * 0.54f;
                const float dst_cx = vx + vw * 0.70f;
                const float dst_cy = vy + vh * 0.52f;
                const float shape_w = LS(170.0f);
                const float shape_h = LS(108.0f);
                const float r = LS(24.0f);

                fs_path2d_reset(api_addpath_source);
                fs_path2d_reset(api_addpath_merged);
                fs_path2d_round_rect(
                    api_addpath_source,
                    src_cx - shape_w * 0.5f,
                    src_cy - shape_h * 0.5f,
                    shape_w,
                    shape_h,
                    r
                );
                demo_path2d_circle(api_addpath_source, src_cx, src_cy, LS(22.0f));

                fs_path2d_add_path(api_addpath_merged, api_addpath_source);
                {
                    const float a = sinf(t * 0.9f) * 0.62f;
                    const float sc = 0.90f;
                    const float ca = cosf(a) * sc;
                    const float sa = sinf(a) * sc;
                    const float m[6] = {
                        ca,
                        sa,
                        -sa,
                        ca,
                        dst_cx - (ca * src_cx + (-sa) * src_cy),
                        dst_cy - (sa * src_cx + ca * src_cy)
                    };
                    fs_path2d_add_path_with_transform(api_addpath_merged, api_addpath_source, m);
                }

                fs_style_set_fill_rule(core, FS_FILL_RULE_EVENODD);
                fs_path_fill_path2d(core, api_addpath_merged, rgba8(255, 158, 198, 182));
                fs_style_set_line_width(core, LS(5.0f));
                fs_style_set_line_join(core, FS_LINE_JOIN_ROUND);
                fs_style_set_line_cap(core, FS_LINE_CAP_ROUND);
                fs_path_stroke_path2d(core, api_addpath_merged, 0.0f, rgba8(214, 240, 255, 236));

                const bool hit_fill = fs_is_point_in_path2d(core, api_addpath_merged, mouse_fb_x, mouse_fb_y);
                const bool hit_stroke = fs_is_point_in_stroke_path2d(core, api_addpath_merged, mouse_fb_x, mouse_fb_y);
                uint32_t hit_color = rgba8(180, 210, 238, 230);
                if (hit_fill && hit_stroke) {
                    hit_color = rgba8(255, 244, 138, 245);
                } else if (hit_fill) {
                    hit_color = rgba8(148, 250, 186, 242);
                } else if (hit_stroke) {
                    hit_color = rgba8(136, 220, 255, 242);
                }
                fs_cmd_circle(core, mouse_fb_x, mouse_fb_y, LS(4.0f), hit_color);
                fs_cmd_rect_stroke(core, mouse_fb_x - LS(8.0f), mouse_fb_y - LS(8.0f), LS(16.0f), LS(16.0f), 0.0f, LS(1.2f), rgba8(210, 232, 255, 170));

                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        vx + LS(10.0f),
                        vy + vh - LS(8.0f),
                        LS(11.0f),
                        hit_fill ? (hit_stroke ? "hit: fill + stroke" : "hit: fill") : (hit_stroke ? "hit: stroke" : "hit: none"),
                        hit_color,
                        vw - LS(20.0f)
                    );
                }

                fs_style_set_fill_rule(core, FS_FILL_RULE_NONZERO);
                fs_style_set_line_width(core, LS(1.0f));
                fs_style_set_line_join(core, FS_LINE_JOIN_MITER);
                fs_style_set_line_cap(core, FS_LINE_CAP_ROUND);
            }

            // Sample 9: createPattern(image, repeat-mode) dedicated panel.
            {
                fs_style_set_global_alpha(core, 1.0f);
                fs_style_set_global_composite_operation(core, FS_GLOBAL_COMPOSITE_SOURCE_OVER);
                fs_style_set_shadow_color(core, 0u);
                fs_style_set_shadow_blur(core, 0.0f);
                fs_style_set_shadow_offset(core, 0.0f, 0.0f);
                fs_style_set_fill_rule(core, FS_FILL_RULE_NONZERO);
                fs_style_clear_dash(core);
                const float cx = sx0;
                const float cy = demo_flow_push(&sample_flow, LS(224.0f));
                const float cw = sample_w * 3.0f + sample_gap * 2.0f;
                const float ch = LS(224.0f);
                fs_cmd_rect(core, cx, cy, cw, ch, LS(12.0f), rgba8(25, 36, 52, 220));
                fs_cmd_rect_stroke(core, cx, cy, cw, ch, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        cx + LS(14.0f),
                        cy + LS(22.0f),
                        LS(13.0f),
                        "Example: createPattern(image) + setTransform()",
                        rgba8(214, 234, 255, 236),
                        cw - LS(24.0f)
                    );
                }

                const float vx = cx + LS(18.0f);
                const float vy = cy + LS(42.0f);
                const float vw = cw - LS(36.0f);
                const float vh = ch - LS(58.0f);
                fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));

                if (pattern_ready) {
                    const uint32_t prev_fill = fs_style_get_fill_color(core);
                    const uint32_t prev_stroke = fs_style_get_stroke_color(core);
                    const float gap = LS(12.0f);
                    const float cell_w = (vw - gap * 4.0f) / 3.0f;
                    const float cell_h = LS(72.0f);
                    const float row_top = vy + LS(28.0f);
                    const float x0 = vx + gap;
                    const float x1 = x0 + cell_w + gap;
                    const float x2 = x1 + cell_w + gap;
                    const float y0 = row_top;
                    const float y1 = row_top + cell_h + LS(14.0f);

                    FS_Pattern* p_repeat = fs_pattern_create_image(&pattern_handle, FS_PATTERN_REPEAT);
                    FS_Pattern* p_repeat_x = fs_pattern_create_image(&pattern_handle, FS_PATTERN_REPEAT_X);
                    FS_Pattern* p_norepeat = fs_pattern_create_image(&pattern_handle, FS_PATTERN_NO_REPEAT);

                    if (p_repeat) {
                        (void)fs_style_set_fill_pattern(core, p_repeat);
                        (void)fs_style_set_stroke_pattern(core, p_repeat);
                        (void)fs_fill_rect(core, x0, y0, cell_w, cell_h, LS(8.0f));
                        (void)fs_stroke_rect(core, x0, y0, cell_w, cell_h, LS(8.0f), LS(1.4f));
                        fs_pattern_destroy(p_repeat);
                    }
                    if (p_repeat_x) {
                        const float ang = 0.34f;
                        const float sc = 0.86f;
                        const float ca = cosf(ang) * sc;
                        const float sa = sinf(ang) * sc;
                        (void)fs_pattern_set_transform(
                            p_repeat_x,
                            ca,
                            sa,
                            -sa,
                            ca,
                            x1 + cell_w * 0.32f,
                            y0 + cell_h * 0.12f
                        );
                        (void)fs_style_set_fill_pattern(core, p_repeat_x);
                        (void)fs_style_set_stroke_pattern(core, p_repeat_x);
                        (void)fs_fill_rect(core, x1, y0, cell_w, cell_h, LS(8.0f));
                        (void)fs_stroke_rect(core, x1, y0, cell_w, cell_h, LS(8.0f), LS(1.4f));
                        if (font_ready) {
                            (void)fs_fill_text_utf8(core, x1 + LS(8.0f), y1 + LS(14.0f), LS(14.0f), "Pattern Text", cell_w - LS(12.0f));
                        }
                        fs_pattern_destroy(p_repeat_x);
                    }
                    if (p_norepeat) {
                        (void)fs_style_set_fill_pattern(core, p_norepeat);
                        (void)fs_style_set_stroke_pattern(core, p_norepeat);
                        (void)fs_fill_rect(core, x2, y0, cell_w, cell_h, LS(8.0f));
                        (void)fs_stroke_rect(core, x2, y0, cell_w, cell_h, LS(8.0f), LS(1.4f));
                        fs_pattern_destroy(p_norepeat);
                    }

                    (void)fs_style_set_fill_color(core, prev_fill);
                    (void)fs_style_set_stroke_color(core, prev_stroke);

                    if (font_ready) {
                        fs_cmd_text_utf8(core, x0, y1 + LS(14.0f), LS(11.0f), "repeat", rgba8(194, 224, 252, 234), cell_w);
                        fs_cmd_text_utf8(core, x1, y1 + LS(14.0f), LS(11.0f), "repeat-x + transform", rgba8(194, 224, 252, 234), cell_w);
                        fs_cmd_text_utf8(core, x2, y1 + LS(14.0f), LS(11.0f), "no-repeat", rgba8(194, 224, 252, 234), cell_w);
                    }
                } else {
                    fs_cmd_rect(core, vx + LS(12.0f), vy + LS(18.0f), vw - LS(24.0f), vh - LS(30.0f), LS(8.0f), rgba8(98, 142, 188, 145));
                    if (font_ready) {
                        fs_cmd_text_utf8(
                            core,
                            vx + LS(20.0f),
                            vy + LS(54.0f),
                            LS(12.0f),
                            "Pattern sample requires pattern_handle",
                            rgba8(220, 236, 255, 230),
                            vw - LS(40.0f)
                        );
                    }
                }
            }

            // Sample 10: letterSpacing + wordSpacing.
            {
                const float cx = sx0;
                const float cy = demo_flow_push(&sample_flow, LS(214.0f));
                const float cw = sample_w * 3.0f + sample_gap * 2.0f;
                const float ch = LS(214.0f);
                fs_cmd_rect(core, cx, cy, cw, ch, LS(12.0f), rgba8(25, 36, 52, 220));
                fs_cmd_rect_stroke(core, cx, cy, cw, ch, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        cx + LS(14.0f),
                        cy + LS(22.0f),
                        LS(13.0f),
                        "Example: letterSpacing + wordSpacing",
                        rgba8(214, 234, 255, 236),
                        cw - LS(24.0f)
                    );
                }

                if (font_ready) {
                    const float vx = cx + LS(18.0f);
                    const float vy = cy + LS(42.0f);
                    const float vw = cw - LS(36.0f);
                    const float vh = ch - LS(58.0f);
                    const float text_x = vx + LS(16.0f);
                    const float sample_max_w = vw - LS(204.0f);
                    const float y0 = vy + LS(44.0f);
                    const float y1 = y0 + LS(44.0f);
                    const float y2 = y1 + LS(44.0f);
                    const char* sample_text = "A V A W   Canvas word spacing demo 123";
                    fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));
                    fs_cmd_line(core, vx + LS(8.0f), y0, vx + vw - LS(8.0f), y0, LS(1.2f), rgba8(255, 190, 120, 166));
                    fs_cmd_line(core, vx + LS(8.0f), y1, vx + vw - LS(8.0f), y1, LS(1.2f), rgba8(130, 220, 255, 166));
                    fs_cmd_line(core, vx + LS(8.0f), y2, vx + vw - LS(8.0f), y2, LS(1.2f), rgba8(180, 242, 170, 166));

                    fs_style_set_text_align(core, FS_TEXT_ALIGN_START);
                    fs_style_set_text_baseline(core, FS_TEXT_BASELINE_ALPHABETIC);

                    fs_style_set_letter_spacing(core, 0.0f);
                    fs_style_set_word_spacing(core, 0.0f);
                    fs_cmd_text_utf8(core, text_x, y0, LS(22.0f), sample_text, rgba8(255, 232, 152, 242), sample_max_w);
                    FS_TextMetrics tm0 = {0};
                    (void)fs_measure_text_utf8(core, LS(22.0f), sample_text, sample_max_w, &tm0);

                    fs_style_set_letter_spacing(core, LS(2.8f));
                    fs_style_set_word_spacing(core, 0.0f);
                    fs_cmd_text_utf8(core, text_x, y1, LS(22.0f), sample_text, rgba8(154, 232, 255, 242), sample_max_w);
                    FS_TextMetrics tm1 = {0};
                    (void)fs_measure_text_utf8(core, LS(22.0f), sample_text, sample_max_w, &tm1);

                    fs_style_set_letter_spacing(core, 0.0f);
                    fs_style_set_word_spacing(core, LS(11.0f));
                    fs_cmd_text_utf8(core, text_x, y2, LS(22.0f), sample_text, rgba8(184, 246, 176, 242), sample_max_w);
                    FS_TextMetrics tm2 = {0};
                    (void)fs_measure_text_utf8(core, LS(22.0f), sample_text, sample_max_w, &tm2);

                    char wbuf0[64];
                    char wbuf1[64];
                    char wbuf2[64];
                    snprintf(wbuf0, sizeof(wbuf0), "normal  width=%.1f", tm0.width);
                    snprintf(wbuf1, sizeof(wbuf1), "letter  width=%.1f", tm1.width);
                    snprintf(wbuf2, sizeof(wbuf2), "word    width=%.1f", tm2.width);
                    fs_cmd_text_utf8(core, vx + vw - LS(188.0f), y0, LS(12.0f), wbuf0, rgba8(236, 240, 255, 220), LS(176.0f));
                    fs_cmd_text_utf8(core, vx + vw - LS(188.0f), y1, LS(12.0f), wbuf1, rgba8(236, 240, 255, 220), LS(176.0f));
                    fs_cmd_text_utf8(core, vx + vw - LS(188.0f), y2, LS(12.0f), wbuf2, rgba8(236, 240, 255, 220), LS(176.0f));

                    fs_style_set_letter_spacing(core, 0.0f);
                    fs_style_set_word_spacing(core, 0.0f);
                    fs_style_set_text_align(core, FS_TEXT_ALIGN_START);
                    fs_style_set_text_baseline(core, FS_TEXT_BASELINE_ALPHABETIC);
                }
            }

            // Sample 11: pattern stroke under oriented transforms.
            {
                fs_style_set_global_alpha(core, 1.0f);
                fs_style_set_global_composite_operation(core, FS_GLOBAL_COMPOSITE_SOURCE_OVER);
                fs_style_set_shadow_color(core, 0u);
                fs_style_set_shadow_blur(core, 0.0f);
                fs_style_set_shadow_offset(core, 0.0f, 0.0f);
                fs_style_set_fill_rule(core, FS_FILL_RULE_NONZERO);
                fs_style_clear_dash(core);
                const float cx = sx0;
                const float cy = demo_flow_push(&sample_flow, LS(252.0f));
                const float cw = sample_w * 3.0f + sample_gap * 2.0f;
                const float ch = LS(252.0f);
                fs_cmd_rect(core, cx, cy, cw, ch, LS(12.0f), rgba8(25, 36, 52, 220));
                fs_cmd_rect_stroke(core, cx, cy, cw, ch, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        cx + LS(14.0f),
                        cy + LS(22.0f),
                        LS(13.0f),
                        "Example: pattern stroke (plain / rotate / rotate+dash)",
                        rgba8(214, 234, 255, 236),
                        cw - LS(24.0f)
                    );
                }

                const float vx = cx + LS(18.0f);
                const float vy = cy + LS(42.0f);
                const float vw = cw - LS(36.0f);
                const float vh = ch - LS(58.0f);
                fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));

                if (pattern_ready) {
                    const float gap = LS(12.0f);
                    const float cell_w = (vw - gap * 4.0f) / 3.0f;
                    const float cell_h = vh - LS(42.0f);
                    const float x0 = vx + gap;
                    const float x1 = x0 + cell_w + gap;
                    const float x2 = x1 + cell_w + gap;
                    const float y0 = vy + LS(14.0f);

                    FS_Pattern* p_repeat = fs_pattern_create_image(&pattern_handle, FS_PATTERN_REPEAT);
                    if (p_repeat) {
                        fs_state_save(core);
                        (void)fs_style_set_stroke_pattern(core, p_repeat);
                        (void)fs_style_set_line_width(core, LS(11.0f));
                        (void)fs_style_set_line_join(core, FS_LINE_JOIN_ROUND);
                        (void)fs_style_set_line_cap(core, FS_LINE_CAP_ROUND);

                        for (int ci = 0; ci < 3; ++ci) {
                            const float bx = (ci == 0) ? x0 : ((ci == 1) ? x1 : x2);
                            const float by = y0;
                            fs_cmd_rect(core, bx, by, cell_w, cell_h, LS(8.0f), rgba8(20, 31, 46, 222));
                            fs_cmd_rect_stroke(core, bx, by, cell_w, cell_h, LS(8.0f), LS(1.2f), rgba8(188, 216, 242, 170));

                            fs_state_save(core);
                            const float cxv = bx + cell_w * 0.5f;
                            const float cyv = by + cell_h * 0.5f;
                            fs_translate(core, cxv, cyv);
                            if (ci == 1) {
                                fs_rotate(core, 0.52f);
                            } else if (ci == 2) {
                                fs_rotate(core, 0.52f);
                            }

                            if (ci == 2) {
                                const float dash_pattern[] = {LS(22.0f), LS(9.0f)};
                                (void)fs_style_set_dash(core, dash_pattern, (uint32_t)ARRAY_COUNT(dash_pattern), t * LS(42.0f));
                            } else {
                                fs_style_clear_dash(core);
                            }

                            fs_path_begin(core);
                            fs_path_move_to(core, LS(-62.0f), LS(-20.0f));
                            fs_path_line_to(core, LS(-24.0f), LS(22.0f));
                            fs_path_line_to(core, LS(8.0f), LS(-18.0f));
                            fs_path_line_to(core, LS(40.0f), LS(12.0f));
                            fs_path_line_to(core, LS(62.0f), LS(-8.0f));
                            (void)fs_stroke(core, 0.0f);
                            fs_state_restore(core);
                        }

                        fs_style_clear_dash(core);
                        fs_state_restore(core);
                        fs_pattern_destroy(p_repeat);
                    }

                    if (font_ready) {
                        fs_cmd_text_utf8(core, x0, vy + vh - LS(10.0f), LS(11.0f), "plain", rgba8(194, 224, 252, 234), cell_w);
                        fs_cmd_text_utf8(core, x1, vy + vh - LS(10.0f), LS(11.0f), "rotate", rgba8(194, 224, 252, 234), cell_w);
                        fs_cmd_text_utf8(core, x2, vy + vh - LS(10.0f), LS(11.0f), "rotate + dash", rgba8(194, 224, 252, 234), cell_w);
                    }
                } else if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        vx + LS(20.0f),
                        vy + LS(48.0f),
                        LS(12.0f),
                        "Pattern stroke sample requires pattern_handle",
                        rgba8(220, 236, 255, 230),
                        vw - LS(40.0f)
                    );
                }
            }

            // Sample 12: imageSmoothingEnabled + imageSmoothingQuality subset (interactive).
            {
                const float cx = sx0;
                const float cy = demo_flow_push(&sample_flow, LS(236.0f));
                const float cw = sample_w * 3.0f + sample_gap * 2.0f;
                const float ch = LS(236.0f);
                fs_cmd_rect(core, cx, cy, cw, ch, LS(12.0f), rgba8(25, 36, 52, 220));
                fs_cmd_rect_stroke(core, cx, cy, cw, ch, LS(12.0f), LS(2.0f), rgba8(156, 198, 232, 190));
                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        cx + LS(14.0f),
                        cy + LS(22.0f),
                        LS(13.0f),
                        "Example: imageSmoothingEnabled + imageSmoothingQuality  (I/K hotkeys)",
                        rgba8(214, 234, 255, 236),
                        cw - LS(24.0f)
                    );
                }

                const float vx = cx + LS(18.0f);
                const float vy = cy + LS(42.0f);
                const float vw = cw - LS(36.0f);
                const float vh = ch - LS(58.0f);
                fs_cmd_rect(core, vx, vy, vw, vh, LS(8.0f), rgba8(14, 22, 34, 230));

                const float gap = LS(12.0f);
                const float cell_w = (vw - gap * 4.0f) / 3.0f;
                const float cell_h = vh - LS(42.0f);
                const float x0 = vx + gap;
                const float x1 = x0 + cell_w + gap;
                const float x2 = x1 + cell_w + gap;
                const float y0 = vy + LS(14.0f);
                const float label_y = vy + vh - LS(24.0f);
                const float hint_y = vy + vh - LS(9.0f);
                const float img_pad = LS(10.0f);
                const float img_w = cell_w - img_pad * 2.0f;
                const float img_h = cell_h - img_pad * 2.0f;
                const float zoom_s = 1.55f + 0.95f * sinf(t * 1.35f);

                for (int ci = 0; ci < 3; ++ci) {
                    const float bx = (ci == 0) ? x0 : ((ci == 1) ? x1 : x2);
                    const bool smooth = (ci == 0) ? false : smoothing_demo_enabled;
                    const FS_ImageSmoothingQuality quality =
                        (ci == 0) ? FS_IMAGE_SMOOTHING_QUALITY_LOW : smoothing_demo_quality;
                    fs_cmd_rect(core, bx, y0, cell_w, cell_h, LS(8.0f), rgba8(20, 31, 46, 222));
                    fs_cmd_rect_stroke(core, bx, y0, cell_w, cell_h, LS(8.0f), LS(1.2f), rgba8(188, 216, 242, 170));
                    fs_style_set_image_smoothing_enabled(core, smooth);
                    fs_style_set_image_smoothing_quality(core, quality);
                    float draw_w = img_w;
                    float draw_h = img_h;
                    float draw_x = bx + img_pad;
                    float draw_y = y0 + img_pad;
                    if (ci == 2) {
                        draw_w = img_w * zoom_s;
                        draw_h = img_h * zoom_s;
                        draw_x = bx + (cell_w - draw_w) * 0.5f;
                        draw_y = y0 + (cell_h - draw_h) * 0.5f;
                    }
                    if (image_ready) {
                        fs_cmd_image_handle(
                            core,
                            draw_x,
                            draw_y,
                            draw_w,
                            draw_h,
                            &image_handle,
                            rgba8(255, 255, 255, 242)
                        );
                    } else {
                        fs_cmd_rect(
                            core,
                            draw_x,
                            draw_y,
                            draw_w,
                            draw_h,
                            LS(6.0f),
                            rgba8(108, 158, 212, 180)
                        );
                    }
                }
                fs_style_set_image_smoothing_enabled(core, true);
                fs_style_set_image_smoothing_quality(core, FS_IMAGE_SMOOTHING_QUALITY_LOW);

                if (font_ready) {
                    const char* q_name = "LOW";
                    if (smoothing_demo_quality == FS_IMAGE_SMOOTHING_QUALITY_MEDIUM) {
                        q_name = "MEDIUM";
                    } else if (smoothing_demo_quality == FS_IMAGE_SMOOTHING_QUALITY_HIGH) {
                        q_name = "HIGH";
                    }
                    char ctl_label[64];
                    snprintf(
                        ctl_label,
                        sizeof(ctl_label),
                        "CTRL: %s + %s",
                        smoothing_demo_enabled ? "ON" : "OFF",
                        q_name
                    );
                    char zoom_label[64];
                    snprintf(zoom_label, sizeof(zoom_label), "ZOOM x%.2f (minify/magnify)", zoom_s);
                    fs_cmd_text_utf8(core, x0, label_y, LS(11.0f), "BASELINE: OFF", rgba8(194, 224, 252, 234), cell_w);
                    fs_cmd_text_utf8(core, x1, label_y, LS(11.0f), ctl_label, rgba8(194, 224, 252, 234), cell_w);
                    fs_cmd_text_utf8(core, x2, label_y, LS(11.0f), zoom_label, rgba8(194, 224, 252, 234), cell_w);
                    fs_cmd_text_utf8(
                        core,
                        x1,
                        hint_y,
                        LS(10.0f),
                        "Keys: I toggle enabled, K cycle quality",
                        rgba8(162, 204, 236, 224),
                        cell_w * 2.0f + gap
                    );
                }
            }

            if (font_ready) {
                const float note_y = sample_flow.max_bottom + LS(10.0f);
                fs_cmd_text_utf8(
                    core,
                    LX(28.0f),
                    note_y,
                    LS(11.0f),
                    "Status grid + live visual cards are both shown. This is implementation coverage, not full HTML Canvas 2D spec parity.",
                    rgba8(170, 208, 238, 224),
                    LS(1224.0f)
                );
            }
            {
                const float content_bottom_screen = sample_flow.max_bottom + LS(44.0f);
                const float content_bottom_design = (content_bottom_screen - oy) * inv_s;
                scene_content_h_design[2] = fmaxf(720.0f, content_bottom_design);
            }
            goto submit_frame;
        }

        fs_cmd_rect(core, LX(50.0f), LY(60.0f), LS(300.0f), LS(180.0f), LS(28.0f), rgba8(38, 177, 255, 220));
        fs_cmd_rect_stroke(
            core,
            LX(50.0f),
            LY(60.0f),
            LS(300.0f),
            LS(180.0f),
            LS(28.0f),
            LS(8.0f),
            rgba8(230, 248, 255, 235)
        );
        fs_state_save(core);
        fs_clip_rect(core, LX(85.0f), LY(92.0f), LS(220.0f), LS(116.0f));
        fs_cmd_rect(core, LX(20.0f), LY(20.0f), LS(360.0f), LS(220.0f), LS(26.0f), rgba8(255, 128, 98, 130));
        fs_cmd_circle(core, LX(170.0f), LY(150.0f), LS(92.0f), rgba8(255, 239, 173, 180));
        fs_state_restore(core);

        if (image_ready) {
            fs_cmd_image_handle(core, LX(400.0f), LY(60.0f), LS(260.0f), LS(180.0f), &image_handle, rgba8(255, 255, 255, 255));
        } else {
            fs_cmd_image(core, LX(400.0f), LY(60.0f), LS(260.0f), LS(180.0f), 0.0f, 0.0f, 1.0f, 1.0f, rgba8(255, 255, 255, 255));
        }

        fs_cmd_rect(core, LX(390.0f), LY(255.0f), LS(250.0f), LS(210.0f), LS(16.0f), rgba8(36, 54, 78, 130));
        fs_cmd_rect_stroke(
            core,
            LX(390.0f),
            LY(255.0f),
            LS(250.0f),
            LS(210.0f),
            LS(16.0f),
            LS(2.0f),
            rgba8(196, 224, 255, 190)
        );
        if (font_ready) {
            fs_cmd_text_utf8(
                core,
                LX(405.0f),
                LY(282.0f),
                LS(17.0f),
                "COMPLEX CLIP STACK",
                rgba8(216, 236, 255, 240),
                LS(220.0f)
            );
            char clip_mode_buf[64];
            snprintf(clip_mode_buf, sizeof(clip_mode_buf), "RULE CLIP VIEW: %s (V)", clip_view_mode_name(rule_clip_view_mode));
            fs_cmd_text_utf8(
                core,
                LX(405.0f),
                LY(462.0f),
                LS(12.0f),
                clip_mode_buf,
                rgba8(198, 225, 255, 220),
                LS(236.0f)
            );
        }
        fs_state_save(core);
        fs_clip_rect(core, LX(392.0f), LY(292.0f), LS(246.0f), LS(166.0f));
        const float clip_a_x = LX(410.0f);
        const float clip_a_y = LY(295.0f);
        const float clip_a_w = LS(210.0f);
        const float clip_a_h = LS(150.0f);
        const float clip_b_x = LX(448.0f + sinf(t * 0.95f) * 26.0f);
        const float clip_b_y = LY(304.0f);
        const float clip_b_w = LS(134.0f);
        const float clip_b_h = LS(134.0f);
        // Visual guides for rule-clip boundaries.
        fs_cmd_rect_stroke(core, clip_a_x, clip_a_y, clip_a_w, clip_a_h, LS(8.0f), LS(2.0f), rgba8(140, 220, 255, 220));
        fs_cmd_rect_stroke(core, clip_b_x, clip_b_y, clip_b_w, clip_b_h, LS(8.0f), LS(2.0f), rgba8(255, 210, 140, 220));
        if (rule_clip_view_mode == 0 || rule_clip_view_mode == 2) {
            // RAW draw (no clip): deliberately overflowing geometry.
            fs_cmd_rect(core, LX(392.0f), LY(282.0f), LS(250.0f), LS(178.0f), LS(10.0f), rgba8(255, 120, 120, 90));
            fs_cmd_circle(core, LX(520.0f + cosf(t * 1.4f) * 26.0f), LY(366.0f), LS(84.0f), rgba8(255, 210, 110, 95));
            fs_cmd_line(core, LX(410.0f), LY(438.0f), LX(620.0f), LY(305.0f), LS(18.0f), rgba8(255, 245, 200, 110));
            fs_cmd_line(core, LX(360.0f), LY(452.0f), LX(654.0f), LY(276.0f), LS(24.0f), rgba8(255, 156, 230, 105));
            fs_cmd_line(core, LX(362.0f), LY(402.0f), LX(658.0f), LY(252.0f), LS(24.0f), rgba8(118, 210, 255, 100));
        }
        if (rule_clip_view_mode == 1 || rule_clip_view_mode == 2) {
            fs_state_save(core);
            fs_clip_rect(core, clip_a_x, clip_a_y, clip_a_w, clip_a_h);
            fs_clip_rect(core, clip_b_x, clip_b_y, clip_b_w, clip_b_h);
            fs_cmd_rect(core, LX(392.0f), LY(282.0f), LS(250.0f), LS(178.0f), LS(10.0f), rgba8(255, 132, 120, 165));
            fs_cmd_circle(core, LX(520.0f + cosf(t * 1.4f) * 26.0f), LY(366.0f), LS(84.0f), rgba8(125, 255, 211, 190));
            fs_cmd_line(core, LX(410.0f), LY(438.0f), LX(620.0f), LY(305.0f), LS(18.0f), rgba8(255, 248, 182, 230));
            fs_cmd_line(core, LX(360.0f), LY(452.0f), LX(654.0f), LY(276.0f), LS(24.0f), rgba8(255, 156, 230, 155));
            fs_cmd_line(core, LX(362.0f), LY(402.0f), LX(658.0f), LY(252.0f), LS(24.0f), rgba8(118, 210, 255, 150));
            fs_state_restore(core);
        }
        fs_state_restore(core);

        fs_cmd_rect(core, LX(80.0f), LY(280.0f), LS(300.0f), LS(190.0f), LS(14.0f), rgba8(28, 42, 56, 128));
        fs_cmd_rect_stroke(
            core,
            LX(80.0f),
            LY(280.0f),
            LS(300.0f),
            LS(190.0f),
            LS(14.0f),
            LS(2.0f),
            rgba8(180, 215, 255, 190)
        );
        if (font_ready) {
            fs_cmd_text_utf8(
                core,
                LX(96.0f),
                LY(304.0f),
                LS(15.0f),
                "PATH CLIP (IRREGULAR + ARCTO/ROUNDRECT)",
                rgba8(210, 234, 255, 230),
                LS(268.0f)
            );
        }
        fs_state_save(core);
        fs_clip_rect(core, LX(84.0f), LY(312.0f), LS(292.0f), LS(154.0f));

        fs_state_save(core);
        fs_path_begin(core);
        fs_path_move_to(core, LX(120.0f), LY(340.0f));
        fs_path_line_to(core, LX(200.0f), LY(318.0f));
        fs_path_bezier_curve_to(core, LX(246.0f), LY(308.0f), LX(302.0f), LY(354.0f), LX(286.0f), LY(412.0f));
        fs_path_line_to(core, LX(236.0f), LY(448.0f));
        fs_path_quadratic_curve_to(core, LX(186.0f), LY(466.0f), LX(138.0f), LY(440.0f));
        fs_path_line_to(core, LX(112.0f), LY(388.0f));
        fs_path_close(core);
        const bool clip_ok_a = fs_clip_path(core);
        (void)clip_ok_a;
        fs_cmd_rect(core, LX(92.0f), LY(314.0f), LS(248.0f), LS(144.0f), LS(12.0f), rgba8(255, 132, 108, 156));
        fs_cmd_circle(core, LX(190.0f + cosf(t * 1.5f) * 42.0f), LY(386.0f), LS(62.0f), rgba8(126, 255, 211, 190));
        fs_cmd_line(core, LX(104.0f), LY(448.0f), LX(332.0f), LY(322.0f), LS(14.0f), rgba8(255, 245, 176, 225));
        fs_state_restore(core);

        fs_state_save(core);
        // Draw helper clip contour first (unclipped), so cut result has obvious boundary reference.
        fs_path_begin(core);
        fs_path_round_rect(core, LX(252.0f), LY(326.0f), LS(112.0f), LS(128.0f), LS(18.0f));
        fs_path_move_to(core, LX(266.0f), LY(392.0f));
        fs_path_arc_to(core, LX(290.0f), LY(342.0f), LX(334.0f), LY(360.0f), LS(18.0f));
        fs_path_arc_to(core, LX(350.0f), LY(396.0f), LX(326.0f), LY(434.0f), LS(16.0f));
        fs_path_arc_to(core, LX(286.0f), LY(438.0f), LX(266.0f), LY(392.0f), LS(14.0f));
        fs_path_close(core);
        fs_path_stroke(core, LS(2.0f), rgba8(190, 240, 255, 210));

        fs_path_begin(core);
        fs_path_round_rect(core, LX(252.0f), LY(326.0f), LS(112.0f), LS(128.0f), LS(18.0f));
        fs_path_move_to(core, LX(266.0f), LY(392.0f));
        fs_path_arc_to(core, LX(290.0f), LY(342.0f), LX(334.0f), LY(360.0f), LS(18.0f));
        fs_path_arc_to(core, LX(350.0f), LY(396.0f), LX(326.0f), LY(434.0f), LS(16.0f));
        fs_path_arc_to(core, LX(286.0f), LY(438.0f), LX(266.0f), LY(392.0f), LS(14.0f));
        fs_path_close(core);
        const bool clip_ok_b = fs_clip_path(core);
        (void)clip_ok_b;
        fs_cmd_rect(core, LX(228.0f), LY(304.0f), LS(166.0f), LS(164.0f), LS(14.0f), rgba8(116, 210, 255, 154));
        fs_cmd_circle(core, LX(306.0f + cosf(t * 1.2f) * 16.0f), LY(390.0f), LS(34.0f), rgba8(255, 203, 118, 190));
        fs_cmd_line(core, LX(236.0f), LY(462.0f), LX(392.0f), LY(326.0f), LS(12.0f), rgba8(255, 244, 200, 220));
        fs_state_restore(core);
        fs_state_restore(core);

        if (font_ready) {
            FS_TextMetrics title_metrics;
            bool title_metrics_ok =
                fs_measure_text_utf8(core, LS(52.0f), "COMPUTE PIPELINE", LS(520.0f), &title_metrics);
            if (title_metrics_ok) {
                const float box_x = LX(730.0f) - title_metrics.actual_bounding_box_left;
                const float box_y = LY(142.0f) - title_metrics.actual_bounding_box_ascent;
                const float box_w = title_metrics.actual_bounding_box_left + title_metrics.actual_bounding_box_right;
                const float box_h = title_metrics.actual_bounding_box_ascent + title_metrics.actual_bounding_box_descent;
                fs_cmd_rect_stroke(core, box_x, box_y, box_w, box_h, LS(8.0f), LS(2.0f), rgba8(255, 255, 255, 120));
            }
            fs_style_set_line_width(core, 3.0f);
            fs_cmd_stroke_text_utf8(
                core,
                LX(730.0f),
                LY(142.0f),
                LS(52.0f),
                "COMPUTE PIPELINE",
                rgba8(18, 36, 48, 230),
                LS(520.0f),
                3.0f
            );
            fs_cmd_text_utf8(core, LX(730.0f), LY(142.0f), LS(52.0f), "COMPUTE PIPELINE", rgba8(255, 242, 123, 245), LS(520.0f));
            fs_cmd_text_utf8(
                core,
                LX(730.0f),
                LY(196.0f),
                LS(24.0f),
                "EMBED PNG EMOJI: \xF0\x9F\x8C\xBF \xF0\x9F\x8D\x95 \xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x9A\x80 \xE2\x9C\xA8 \xE2\x9D\xA4\xEF\xB8\x8F\xE2\x80\x8D\xF0\x9F\x94\xA5",
                rgba8(214, 235, 255, 235),
                LS(520.0f)
            );
            fs_cmd_text_utf8(
                core,
                LX(730.0f),
                LY(232.0f),
                LS(24.0f),
                "LAZY MISSING QUEUE: \xF0\x9F\xAB\x95 \xE2\xAD\x90 \xF0\x9F\x90\xB2 \xF0\x9F\x92\xAB",
                rgba8(180, 230, 210, 230),
                LS(520.0f)
            );
            fs_cmd_text_utf8(
                core,
                LX(730.0f),
                LY(268.0f),
                LS(24.0f),
                "NATIVE COLOR FONT: \xF0\x9F\x98\x80 \xF0\x9F\x9A\x80 \xF0\x9F\x91\x8D \xF0\x9F\x8E\x89",
                rgba8(255, 255, 255, 255),
                LS(520.0f)
            );
        }

        {
            const float comp_panel_x = LX(700.0f);
            const float comp_panel_y = LY(304.0f);
            const float comp_panel_w = LS(500.0f);
            const float comp_panel_h = LS(176.0f);
            fs_cmd_rect(core, comp_panel_x, comp_panel_y, comp_panel_w, comp_panel_h, LS(14.0f), rgba8(30, 42, 58, 150));
            fs_cmd_rect_stroke(
                core,
                comp_panel_x,
                comp_panel_y,
                comp_panel_w,
                comp_panel_h,
                LS(14.0f),
                LS(2.0f),
                rgba8(182, 216, 250, 180)
            );
            if (font_ready) {
                fs_cmd_text_utf8(
                    core,
                    comp_panel_x + LS(14.0f),
                    comp_panel_y + LS(22.0f),
                    LS(14.0f),
                    "GLOBAL ALPHA + GLOBAL COMPOSITE (extended subset, auto-cycle)",
                    rgba8(222, 238, 255, 238),
                    comp_panel_w - LS(20.0f)
                );
            }
            fs_state_save(core);
            fs_clip_rect(
                core,
                comp_panel_x + LS(6.0f),
                comp_panel_y + LS(30.0f),
                comp_panel_w - LS(12.0f),
                comp_panel_h - LS(24.0f)
            );

            const FS_GlobalCompositeOperation comp_modes_all[] = {
                FS_GLOBAL_COMPOSITE_SOURCE_OVER,
                FS_GLOBAL_COMPOSITE_COPY,
                FS_GLOBAL_COMPOSITE_LIGHTER,
                FS_GLOBAL_COMPOSITE_DESTINATION_OVER,
                FS_GLOBAL_COMPOSITE_SOURCE_IN,
                FS_GLOBAL_COMPOSITE_SOURCE_OUT,
                FS_GLOBAL_COMPOSITE_DESTINATION_IN,
                FS_GLOBAL_COMPOSITE_DESTINATION_OUT,
                FS_GLOBAL_COMPOSITE_XOR,
                FS_GLOBAL_COMPOSITE_SOURCE_ATOP,
                FS_GLOBAL_COMPOSITE_DESTINATION_ATOP
            };
            const char* comp_labels_all[] = {
                "source-over",
                "copy",
                "lighter",
                "destination-over",
                "source-in",
                "source-out",
                "destination-in",
                "destination-out",
                "xor",
                "source-atop",
                "destination-atop"
            };
            const int comp_slots = 4;
            const int comp_mode_count = (int)ARRAY_COUNT(comp_modes_all);
            const int comp_cycle_base = ((int)(t * 0.65f)) % comp_mode_count;
            const float cell_w = LS(112.0f);
            const float cell_h = LS(66.0f);
            const float gap = LS(8.0f);
            const float cell_y = comp_panel_y + LS(34.0f);
            const float alpha_demo = 0.58f;

            for (int i = 0; i < comp_slots; ++i) {
                const int mode_idx = (comp_cycle_base + i) % comp_mode_count;
                const FS_GlobalCompositeOperation comp_mode = comp_modes_all[mode_idx];
                const char* comp_label = comp_labels_all[mode_idx];
                const float cell_x = comp_panel_x + LS(12.0f) + (float)i * (cell_w + gap);
                fs_cmd_rect(core, cell_x, cell_y, cell_w, cell_h, LS(8.0f), rgba8(17, 24, 34, 255));
                fs_cmd_rect(core, cell_x + LS(8.0f), cell_y + LS(10.0f), LS(58.0f), LS(38.0f), LS(6.0f), rgba8(88, 154, 255, 232));
                fs_cmd_circle(core, cell_x + LS(64.0f), cell_y + LS(36.0f), LS(17.0f), rgba8(255, 204, 90, 230));

                fs_style_set_global_alpha(core, alpha_demo);
                fs_style_set_global_composite_operation(core, comp_mode);
                fs_cmd_rect(core, cell_x + LS(36.0f), cell_y + LS(14.0f), LS(62.0f), LS(40.0f), LS(6.0f), rgba8(255, 94, 168, 255));
                fs_style_set_global_alpha(core, 1.0f);
                fs_style_set_global_composite_operation(core, FS_GLOBAL_COMPOSITE_SOURCE_OVER);

                if (font_ready) {
                    fs_cmd_text_utf8(
                        core,
                        cell_x + LS(2.0f),
                        cell_y + cell_h + LS(12.0f),
                        LS(9.5f),
                        comp_label,
                        rgba8(204, 228, 255, 230),
                        cell_w
                    );
                }
            }

            const float shadow_y = comp_panel_y + LS(114.0f);
            fs_cmd_rect(core, comp_panel_x + LS(12.0f), shadow_y, comp_panel_w - LS(24.0f), LS(50.0f), LS(8.0f), rgba8(12, 18, 28, 210));
            fs_cmd_rect(core, comp_panel_x + LS(36.0f), shadow_y + LS(10.0f), LS(88.0f), LS(30.0f), LS(6.0f), rgba8(76, 140, 255, 225));
            fs_style_set_shadow_color(core, rgba8(0, 0, 0, 210));
            fs_style_set_shadow_blur(core, LS(9.0f));
            fs_style_set_shadow_offset(core, LS(8.0f), LS(6.0f));
            fs_cmd_rect(core, comp_panel_x + LS(74.0f), shadow_y + LS(14.0f), LS(88.0f), LS(26.0f), LS(6.0f), rgba8(255, 112, 170, 240));
            fs_style_set_shadow_offset(core, 0.0f, 0.0f);
            fs_style_set_shadow_blur(core, 0.0f);
            fs_style_set_shadow_color(core, 0u);
            if (font_ready) {
                fs_style_set_shadow_color(core, rgba8(0, 0, 0, 220));
                fs_style_set_shadow_blur(core, LS(6.0f));
                fs_style_set_shadow_offset(core, LS(3.0f), LS(3.0f));
                fs_cmd_text_utf8(
                    core,
                    comp_panel_x + LS(188.0f),
                    shadow_y + LS(31.0f),
                    LS(16.0f),
                    "shadowColor + shadowBlur",
                    rgba8(224, 240, 255, 245),
                    comp_panel_w - LS(206.0f)
                );
                fs_style_set_shadow_offset(core, 0.0f, 0.0f);
                fs_style_set_shadow_blur(core, 0.0f);
                fs_style_set_shadow_color(core, 0u);
            }
            fs_state_restore(core);
        }

        const float geo_panel_x = LX(40.0f);
        const float geo_panel_y = LY(476.0f);
        const float geo_panel_w = LS(1200.0f);
        const float geo_panel_h = LS(320.0f);
        fs_cmd_rect(core, geo_panel_x, geo_panel_y, geo_panel_w, geo_panel_h, LS(14.0f), rgba8(28, 40, 56, 152));
        fs_cmd_rect_stroke(core, geo_panel_x, geo_panel_y, geo_panel_w, geo_panel_h, LS(14.0f), LS(2.0f), rgba8(186, 218, 248, 186));
        if (font_ready) {
            fs_cmd_text_utf8(
                core,
                geo_panel_x + LS(14.0f),
                geo_panel_y + LS(20.0f),
                LS(13.0f),
                "GEOMETRY LAB: lines / path fill / bezier / oriented primitives",
                rgba8(220, 236, 255, 236),
                geo_panel_w - LS(24.0f)
            );
        }
        fs_state_save(core);
        fs_clip_rect(core, geo_panel_x + LS(6.0f), geo_panel_y + LS(6.0f), geo_panel_w - LS(12.0f), geo_panel_h - LS(12.0f));
        fs_state_save(core);
        fs_translate(core, 0.0f, LS(62.0f));

        fs_cmd_line(core, LX(70.0f), LY(500.0f), LX(430.0f), LY(645.0f), LS(18.0f), rgba8(255, 110, 110, 235));
        const float dash_pattern[] = {LS(28.0f), LS(14.0f), LS(6.0f), LS(14.0f)};
        fs_style_set_line_width(core, LS(14.0f));
        fs_style_set_line_cap(core, FS_LINE_CAP_ROUND);
        fs_style_set_line_join(core, FS_LINE_JOIN_ROUND);
        fs_style_set_dash(core, dash_pattern, (uint32_t)ARRAY_COUNT(dash_pattern), t * LS(90.0f));
        fs_path_begin(core);
        fs_path_move_to(core, LX(430.0f), LY(645.0f));
        fs_path_line_to(core, LX(680.0f), LY(560.0f));
        fs_path_bezier_curve_to(core, LX(740.0f), LY(520.0f), LX(840.0f), LY(680.0f), LX(900.0f), LY(665.0f));
        fs_path_stroke(core, 0.0f, rgba8(98, 255, 197, 235));
        fs_style_clear_dash(core);

        fs_style_set_fill_rule(core, FS_FILL_RULE_EVENODD);
        const float yb = LS(70.0f);
        fs_path_begin(core);
        fs_path_move_to(core, LX(760.0f), LY(475.0f) + yb);
        fs_path_line_to(core, LX(875.0f), LY(430.0f) + yb);
        fs_path_quadratic_curve_to(core, LX(1020.0f), LY(395.0f) + yb, LX(1130.0f), LY(472.0f) + yb);
        fs_path_line_to(core, LX(1152.0f), LY(560.0f) + yb);
        fs_path_bezier_curve_to(core, LX(1068.0f), LY(640.0f) + yb, LX(900.0f), LY(646.0f) + yb, LX(806.0f), LY(594.0f) + yb);
        fs_path_line_to(core, LX(748.0f), LY(532.0f) + yb);
        fs_path_close(core);
        fs_path_move_to(core, LX(900.0f), LY(520.0f) + yb);
        fs_path_quadratic_curve_to(core, LX(975.0f), LY(490.0f) + yb, LX(1044.0f), LY(524.0f) + yb);
        fs_path_bezier_curve_to(core, LX(1061.0f), LY(558.0f) + yb, LX(1015.0f), LY(595.0f) + yb, LX(955.0f), LY(596.0f) + yb);
        fs_path_quadratic_curve_to(core, LX(915.0f), LY(593.0f) + yb, LX(885.0f), LY(556.0f) + yb);
        fs_path_close(core);
        const bool fill_ok_a = fs_path_fill(core, rgba8(255, 122, 170, 170));
        (void)fill_ok_a;
        fs_path_begin(core);
        fs_path_round_rect(core, LX(1012.0f), LY(436.0f) + yb, LS(150.0f), LS(110.0f), LS(24.0f));
        fs_path_rect(core, LX(1050.0f), LY(468.0f) + yb, LS(74.0f), LS(46.0f));
        const bool fill_ok_b = fs_path_fill(core, rgba8(120, 219, 255, 168));
        (void)fill_ok_b;
        fs_style_set_fill_rule(core, FS_FILL_RULE_NONZERO);
        fs_path_stroke(core, LS(3.0f), rgba8(255, 230, 244, 230));

        fs_state_save(core);
        fs_translate(core, LX(250.0f), LY(595.0f));
        fs_rotate(core, sinf(t * 0.85f) * 0.55f);
        fs_scale(core, 1.15f, 0.85f);
        fs_cmd_rect(core, LS(-78.0f), LS(-48.0f), LS(156.0f), LS(96.0f), LS(18.0f), rgba8(255, 198, 109, 205));
        fs_cmd_rect_stroke(
            core,
            LS(-78.0f),
            LS(-48.0f),
            LS(156.0f),
            LS(96.0f),
            LS(6.0f),
            LS(6.0f),
            rgba8(255, 247, 225, 240)
        );
        if (image_ready) {
            fs_cmd_image_handle(
                core,
                LS(-68.0f),
                LS(-38.0f),
                LS(44.0f),
                LS(44.0f),
                &image_handle,
                rgba8(255, 255, 255, 235)
            );
        }
        if (font_ready) {
            fs_cmd_text_utf8(
                core,
                LS(-12.0f),
                LS(4.0f),
                LS(20.0f),
                "ORIENTED",
                rgba8(245, 250, 255, 240),
                LS(124.0f)
            );
        }
        fs_state_restore(core);

        {
            const float geo_cx = LX(620.0f + cosf(t * 0.7f) * 150.0f);
            const float geo_cy = LY(540.0f + sinf(t * 1.1f) * 45.0f);
            fs_cmd_circle(core, geo_cx, geo_cy, LS(55.0f), rgba8(199, 120, 255, 215));
        }
        fs_cmd_ellipse(core, LX(1080.0f), LY(620.0f), LS(82.0f), LS(44.0f), rgba8(142, 219, 255, 190));
        fs_cmd_arc(
            core,
            LX(968.0f),
            LY(500.0f),
            LS(92.0f),
            LS(22.0f),
            0.25f + t * 0.4f,
            2.8f + t * 0.4f,
            rgba8(255, 191, 105, 255)
        );

        float bx0 = LX(170.0f);
        float by0 = LY(690.0f);
        float bcx = LX(360.0f + sinf(t * 1.3f) * 90.0f);
        float bcy = LY(555.0f + cosf(t * 1.9f) * 65.0f);
        float bx1 = LX(640.0f);
        float by1 = LY(678.0f);
        fs_cmd_bezier_quad(core, bx0, by0, bcx, bcy, bx1, by1, LS(16.0f), rgba8(255, 255, 255, 255));

        fs_style_set_line_width(core, LS(20.0f));
        fs_style_set_line_cap(core, FS_LINE_CAP_SQUARE);
        fs_cmd_line(core, LX(50.0f), LY(650.0f), LX(145.0f), LY(708.0f), 0.0f, rgba8(210, 240, 255, 230));
        fs_style_set_line_cap(core, FS_LINE_CAP_ROUND);

        float cx0 = LX(760.0f);
        float cy0 = LY(690.0f);
        float ccx0 = LX(860.0f + sinf(t * 1.2f) * 56.0f);
        float ccy0 = LY(580.0f + cosf(t * 1.7f) * 34.0f);
        float ccx1 = LX(1050.0f + cosf(t * 1.6f) * 52.0f);
        float ccy1 = LY(704.0f + sinf(t * 1.4f) * 34.0f);
        float cx1 = LX(1160.0f);
        float cy1 = LY(678.0f);
        fs_cmd_bezier_cubic(core, cx0, cy0, ccx0, ccy0, ccx1, ccy1, cx1, cy1, LS(10.0f), rgba8(255, 232, 168, 255));
        fs_state_restore(core);
        fs_state_restore(core);

#undef LX
#undef LY
#undef LS

submit_frame:
        prev_lmb_state = lmb_state;
        if (!fs_glfw_backend_present(&backend, 0.08f, 0.10f, 0.14f, 1.0f)) {
            fprintf(stderr, "Frame encoding/present failed\n");
            break;
        }
        if (frame == 0u) {
            uint8_t frame_px[4] = {0, 0, 0, 0};
            if (fs_core_get_canvas_image_data_rgba8(core, 1, 1, 1u, 1u, frame_px, sizeof(frame_px))) {
                printf(
                    "canvas framebuffer getImageData sample px=(%u,%u,%u,%u)\n",
                    (unsigned)frame_px[0],
                    (unsigned)frame_px[1],
                    (unsigned)frame_px[2],
                    (unsigned)frame_px[3]
                );
            }
        }

        frame++;
    }

    fs_path2d_destroy(mdn_region_path);
    fs_path2d_destroy(mdn_clip_circle_path);
    fs_path2d_destroy(mdn_clip_square_path);
    fs_path2d_destroy(api_addpath_source);
    fs_path2d_destroy(api_addpath_merged);
    fs_glfw_backend_shutdown(&backend);
    return 0;
}
