// SDF Text Rendering Test - 展示新的 SDF 文本渲染系统
#include <WCN/WCN.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define WCN_GLFW_IMPLEMENTATION
#include "../../impl/wcn_glfw_impl.h"

// 包含字体数据（使用 stb_truetype 的示例字体）
#define WCN_STB_TRUETYPE_IMPLEMENTATION
// 包含 stb_truetype 解码器实现
#include "../../impl/wcn_stb_truetype_impl.h"

static const char* PRIMARY_FONT_CANDIDATES[] = {
    "assets/NotoSerifSC-VF.ttf",
    "../../../assets/NotoSerifSC-VF.ttf",
    "../../../../assets/NotoSerifSC-VF.ttf",
    "../../assets/NotoSerifSC-VF.ttf",
    NULL
};

static const char* FALLBACK_FONT_CANDIDATES[] = {
    "assets/font/DejaVuSans.ttf",
    "../../../assets/font/DejaVuSans.ttf",
    "../../../../assets/font/DejaVuSans.ttf",
    "../../assets/font/DejaVuSans.ttf",
    NULL
};

// 辅助函数：从文件加载字体数据
static bool load_font_data_from_file(const char* path, unsigned char** out_data, size_t* out_size) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        printf("无法打开字体文件: %s\n", path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    unsigned char* data = malloc(file_size);
    if (!data) {
        fclose(file);
        return false;
    }

    size_t bytes_read = fread(data, 1, file_size, file);
    fclose(file);

    if (bytes_read != (size_t)file_size) {
        free(data);
        return false;
    }

    *out_data = data;
    *out_size = file_size;
    return true;
}

static const char* wcn_find_existing_font_path(const char* const* candidates) {
    if (!candidates) {
        return NULL;
    }

    for (size_t i = 0; candidates[i] != NULL; ++i) {
        FILE* file = fopen(candidates[i], "rb");
        if (file) {
            fclose(file);
            return candidates[i];
        }
    }
    return NULL;
}

int main(void) {
    printf("=== WCN SDF Text Rendering Test ===\n");

    // 创建窗口
    WCN_GLFW_Window* window = wcn_glfw_create_window(1024, 768, "WCN SDF Text Test");
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        return -1;
    }

    // 获取 WCN 上下文
    WCN_Context* ctx = wcn_glfw_get_context(window);

    // 注册 stb_truetype 字体解码器
    WCN_FontDecoder* stb_decoder = wcn_get_stb_truetype_decoder();
    wcn_register_font_decoder(ctx, stb_decoder);

    // 加载主字体与回退字体
    WCN_FontFace* primary_face = NULL;
    WCN_FontFace* fallback_face = NULL;

    unsigned char* font_data = NULL;
    size_t font_size = 0;

    const char* primary_font_path = wcn_find_existing_font_path(PRIMARY_FONT_CANDIDATES);
    const char* fallback_font_path = wcn_find_existing_font_path(FALLBACK_FONT_CANDIDATES);

    printf("Loading primary font: %s\n", primary_font_path ? primary_font_path : "(not found)");
    if (primary_font_path && load_font_data_from_file(primary_font_path, &font_data, &font_size)) {
        if (stb_decoder->load_font(font_data, font_size, &primary_face)) {
            printf("  -> primary font ready\n");
            wcn_set_font_face(ctx, primary_face, 24.0f);
        } else {
            printf("  -> failed to decode primary font\n");
        }
        free(font_data);
        font_data = NULL;
    } else {
        printf("  -> failed to read primary font file\n");
    }

    printf("Loading fallback font: %s\n", fallback_font_path ? fallback_font_path : "(not found)");
    if (fallback_font_path && load_font_data_from_file(fallback_font_path, &font_data, &font_size)) {
        if (stb_decoder->load_font(font_data, font_size, &fallback_face)) {
            printf("  -> fallback font ready\n");
            if (!wcn_add_font_fallback(ctx, fallback_face)) {
                printf("  -> failed to register fallback font\n");
            }
        } else {
            printf("  -> failed to decode fallback font\n");
        }
        free(font_data);
        font_data = NULL;
    } else {
        printf("  -> failed to read fallback font file\n");
    }

    if (!primary_face) {
        printf("Warning: no primary font loaded, text rendering may be incomplete.\n");
    }

    printf("Starting render loop...\n");

    // 主循环
    int frame_count = 0;
    bool stats_reset = false;
    const int max_frames = 300;
    double last_time = glfwGetTime();
    while (!wcn_glfw_window_should_close(window) && frame_count < max_frames) {
        wcn_glfw_poll_events();

        // 检查窗口大小是否发生变化
        int new_width, new_height;
        glfwGetFramebufferSize(window->window, &new_width, &new_height);

        // 如果窗口大小为 0（最小化或缩放中），跳过这一帧
        if (new_width == 0 || new_height == 0) {
            continue;
        }

        uint32_t current_width, current_height;
        wcn_glfw_get_size(window, &current_width, &current_height);

        if ((uint32_t)new_width != current_width || (uint32_t)new_height != current_height) {
            printf("Window resized: %dx%d -> %dx%d\n", current_width, current_height, new_width, new_height);
            // 窗口大小发生变化，重新配置表面
            wcn_glfw_handle_resize(window, new_width, new_height);
        }

        // 开始渲染帧
        WCN_GLFW_RenderFrame frame;
        if (wcn_glfw_begin_frame(window, &frame)) {
            if (!stats_reset) {
                wcn_reset_render_stats(ctx);
                stats_reset = true;
            }

            uint32_t width, height;
            wcn_glfw_get_size(window, &width, &height);

            // 清屏为深灰色
            wcn_clear_rect(ctx, 0, 0, width, height);

            // === 测试 1: 基本文本渲染 ===
            wcn_set_fill_style(ctx, 0xFFFFFFFF); // 白色文字
            wcn_set_font_face(ctx, primary_face, 32.0f);
            wcn_fill_text(ctx, "WCN SDF Text Rendering Demo", 50, 50);
            wcn_fill_text(ctx, "❤️", 250, 100);
            wcn_fill_text(ctx, "WCN SDF 文字渲染样例", 450, 150);

            // === 测试 2: 不同字体大小 ===
            float font_sizes[] = {12.0f, 16.0f, 24.0f, 32.0f};  // 减少字号数量
            float y_offset = 100;  // 从 100 开始，节省空间
            for (int i = 0; i < 4; i++) {  // 只测试 4 种字号
                char text[64];
                sprintf(text, "Font Size: %.0f", font_sizes[i]);
                wcn_set_font_face(ctx, primary_face, font_sizes[i]);
                wcn_fill_text(ctx, text, 50, y_offset);
                y_offset += font_sizes[i] + 5;  // 减少间距
            }

            // === 测试 3: 文本对齐 ===
            wcn_set_font_face(ctx, primary_face, 24.0f);
            wcn_set_fill_style(ctx, 0xFFFF8000); // 橙色文字

            // 左对齐
            wcn_set_text_align(ctx, WCN_TEXT_ALIGN_LEFT);
            wcn_fill_text(ctx, "Left Aligned Text", 50, y_offset);

            // 居中对齐
            wcn_set_text_align(ctx, WCN_TEXT_ALIGN_CENTER);
            wcn_fill_text(ctx, "Center Aligned Text", width / 2.0f, y_offset + 40);

            // 右对齐
            wcn_set_text_align(ctx, WCN_TEXT_ALIGN_RIGHT);
            wcn_fill_text(ctx, "Right Aligned Text", width - 50, y_offset + 80);

            // 恢复左对齐
            wcn_set_text_align(ctx, WCN_TEXT_ALIGN_LEFT);

            // === 测试 4: 基线对齐 ===
            y_offset += 100;  // 减少间距
            wcn_set_fill_style(ctx, 0xFF00FFFF); // 青色文字

            // 暂时注释掉基线测试，先测试其他功能
            // wcn_set_text_baseline(ctx, WCN_TEXT_BASELINE_TOP);
            // wcn_fill_text(ctx, "Top Baseline", 50, y_offset);

            // wcn_set_text_baseline(ctx, WCN_TEXT_BASELINE_MIDDLE);
            // wcn_fill_text(ctx, "Middle Baseline", 50, y_offset + 30);

            // wcn_set_text_baseline(ctx, WCN_TEXT_BASELINE_BOTTOM);
            // wcn_fill_text(ctx, "Bottom Baseline", 50, y_offset + 60);

            wcn_set_text_baseline(ctx, WCN_TEXT_BASELINE_ALPHABETIC);
            wcn_fill_text(ctx, "Baseline Test (ALPHABETIC)", 50, y_offset);

            // 恢复默认基线
            wcn_set_text_baseline(ctx, WCN_TEXT_BASELINE_ALPHABETIC);

            // === 测试 5: 混合内容渲染（几何 + 文本） ===
            y_offset += 130;  // 减少间距
            wcn_set_fill_style(ctx, 0xFF00FF00); // 绿色矩形
            wcn_fill_rect(ctx, 50, y_offset, 150, 60);  // 缩小矩形

            wcn_set_fill_style(ctx, 0xFFFFFFFF); // 白色文字
            wcn_set_font_face(ctx, primary_face, 16.0f);
            wcn_fill_text(ctx, "Text over Rect", 60, y_offset + 30);

            // 圆形 + 文字
            wcn_begin_path(ctx);
            wcn_arc(ctx, 350, y_offset + 30, 40, 0, 2 * 3.14159f, false);
            wcn_set_fill_style(ctx, 0xFF0000FF); // 蓝色圆形
            wcn_fill(ctx);

            wcn_set_fill_style(ctx, 0xFFFFFFFF); // 白色文字
            wcn_fill_text(ctx, "Circle", 325, y_offset + 35);

            // === 测试 6: 描边文本 ===
            wcn_set_stroke_style(ctx, 0xFFFF0000); // 红色描边
            wcn_set_line_width(ctx, 2.0f);
            wcn_stroke_text(ctx, "Stroked Text", 500, y_offset + 30);

            // === 测试 7: 动态文本（帧计数） ===
            char fps_text[64];
            double current_time = glfwGetTime();
            double delta_time = current_time - last_time;
            if (delta_time > 0) {
                double fps = 1.0 / delta_time;
                sprintf(fps_text, "FPS: %.1f | Frame: %d", fps, frame_count);
                wcn_set_fill_style(ctx, 0xFF00FFFF); // 青色文字
                    wcn_set_font_face(ctx, primary_face, 16.0f);
                wcn_fill_text(ctx, fps_text, width - 300, 30);
            }
            last_time = current_time;

            // === 测试 8: 复杂文本（中文、emoji等） ===
            // 注意：这需要字体支持这些字符
            wcn_set_fill_style(ctx, 0xFFFFFF00); // 黄色文字
            wcn_set_font_face(ctx, primary_face, 20.0f);
            wcn_fill_text(ctx, "Hello 世界! 🎉", 50, height - 50);

            // 结束渲染帧
            wcn_glfw_end_frame(window, &frame);
        }

        frame_count++;

        // 每 100 帧打印一次
        if (frame_count % 100 == 0) {
            printf("Frame %d\n", frame_count);
        }

        // 按ESC键退出
        if (glfwGetKey(window->window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }
    }

    // 清理
    if (primary_face) {
        stb_decoder->free_font(primary_face);
    }
    if (fallback_face) {
        stb_decoder->free_font(fallback_face);
    }
    WCN_RenderStats stats = {0};
    if (wcn_get_render_stats(ctx, &stats)) {
        printf(
            "WCN_STATS queue_write_calls=%llu queue_write_bytes=%llu compute_dispatch_count=%llu draw_call_count=%llu rendered_instance_count=%llu rendered_vertex_count=%llu gpu_compute_ticks=%llu gpu_render_ticks=%llu gpu_timestamp_samples=%llu\n",
            (unsigned long long)stats.queue_write_calls,
            (unsigned long long)stats.queue_write_bytes,
            (unsigned long long)stats.compute_dispatch_count,
            (unsigned long long)stats.draw_call_count,
            (unsigned long long)stats.rendered_instance_count,
            (unsigned long long)stats.rendered_vertex_count,
            (unsigned long long)stats.gpu_compute_ticks,
            (unsigned long long)stats.gpu_render_ticks,
            (unsigned long long)stats.gpu_timestamp_samples
        );
    }

    wcn_glfw_destroy_window(window);

    printf("Test completed. Rendered %d frames\n", frame_count);
    return 0;
}
