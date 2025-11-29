// 基线测试 - 验证文本对齐和基线是否正确
#include <WCN/WCN.h>
#include <stdio.h>

#define WCN_GLFW_IMPLEMENTATION
#include "../../impl/wcn_glfw_impl.h"

#define WCN_STB_TRUETYPE_IMPLEMENTATION
#include "../../impl/wcn_stb_truetype_impl.h"

const char* font_path = "../../assets/NotoSerif-Medium.ttf";

static bool load_font_data_from_file(const char* path, unsigned char** out_data, size_t* out_size) {
    FILE* file = fopen(path, "rb");
    if (!file) return false;

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

int main(void) {
    printf("=== Baseline Test ===\n");

    WCN_GLFW_Window* window = wcn_glfw_create_window(800, 600, "Baseline Test");
    if (!window) return -1;

    WCN_Context* ctx = wcn_glfw_get_context(window);

    WCN_FontDecoder* stb_decoder = wcn_get_stb_truetype_decoder();
    wcn_register_font_decoder(ctx, stb_decoder);

    WCN_FontFace* font_face = NULL;
    unsigned char* font_data = NULL;
    size_t font_size = 0;

    if (load_font_data_from_file(font_path, &font_data, &font_size)) {
        if (stb_decoder->load_font(font_data, font_size, &font_face)) {
            wcn_set_font_face(ctx, font_face, 24.0f);
        }
        free(font_data);
    }

    int frame_count = 0;
    while (!wcn_glfw_window_should_close(window) && frame_count < 5) {
        wcn_glfw_poll_events();

        WCN_GLFW_RenderFrame frame;
        if (wcn_glfw_begin_frame(window, &frame)) {
            uint32_t width, height;
            wcn_glfw_get_size(window, &width, &height);

            wcn_clear_rect(ctx, 0, 0, width, height);

            // 画参考线
            float y = 200;
            wcn_set_stroke_style(ctx, 0xFFFF0000);  // 红色参考线
            wcn_set_line_width(ctx, 1.0f);
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 0, y);
            wcn_line_to(ctx, width, y);
            wcn_stroke(ctx);

            // 测试 ALPHABETIC（默认）
            wcn_set_fill_style(ctx, 0xFFFFFFFF);
            wcn_set_text_baseline(ctx, WCN_TEXT_BASELINE_ALPHABETIC);
            wcn_fill_text(ctx, "ALPHABETIC (baseline on red line)", 50, y);

            // 测试 TOP
            y += 80;
            wcn_set_stroke_style(ctx, 0xFF00FF00);  // 绿色参考线
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 0, y);
            wcn_line_to(ctx, width, y);
            wcn_stroke(ctx);

            wcn_set_fill_style(ctx, 0xFFFFFFFF);
            wcn_set_text_baseline(ctx, WCN_TEXT_BASELINE_TOP);
            wcn_fill_text(ctx, "TOP (top on green line)", 50, y);

            // 测试 MIDDLE
            y += 80;
            wcn_set_stroke_style(ctx, 0xFF0000FF);  // 蓝色参考线
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 0, y);
            wcn_line_to(ctx, width, y);
            wcn_stroke(ctx);

            wcn_set_fill_style(ctx, 0xFFFFFFFF);
            wcn_set_text_baseline(ctx, WCN_TEXT_BASELINE_MIDDLE);
            wcn_fill_text(ctx, "MIDDLE (middle on blue line)", 50, y);

            // 测试 BOTTOM
            y += 80;
            wcn_set_stroke_style(ctx, 0xFFFFFF00);  // 黄色参考线
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 0, y);
            wcn_line_to(ctx, width, y);
            wcn_stroke(ctx);

            wcn_set_fill_style(ctx, 0xFFFFFFFF);
            wcn_set_text_baseline(ctx, WCN_TEXT_BASELINE_BOTTOM);
            wcn_fill_text(ctx, "BOTTOM (bottom on yellow line)", 50, y);

            wcn_glfw_end_frame(window, &frame);
        }

        frame_count++;
        
        if (glfwGetKey(window->window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }
    }

    if (font_face) {
        stb_decoder->free_font(font_face);
    }
    wcn_glfw_destroy_window(window);

    printf("Test completed. Rendered %d frames\n", frame_count);
    return 0;
}
