// 简化的文本渲染测试
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
    printf("=== Simple Text Test ===\n");

    WCN_GLFW_Window* window = wcn_glfw_create_window(800, 600, "Simple Text Test");
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

            // 测试 1: 基本白色文字
            wcn_set_fill_style(ctx, 0xFFFFFFFF);
            wcn_fill_text(ctx, "Line 1: White text", 50, 50);

            // 测试 2: 橙色文字
            wcn_set_fill_style(ctx, 0xFFFF8000);
            wcn_fill_text(ctx, "Line 2: Orange text", 50, 100);

            // 测试 3: 青色文字
            wcn_set_fill_style(ctx, 0xFF00FFFF);
            wcn_fill_text(ctx, "Line 3: Cyan text", 50, 150);

            // 测试 4: 黄色文字
            wcn_set_fill_style(ctx, 0xFFFFFF00);
            wcn_fill_text(ctx, "Line 4: Yellow text", 50, 200);

            // 测试 5: 绿色文字
            wcn_set_fill_style(ctx, 0xFF00FF00);
            wcn_fill_text(ctx, "Line 5: Green text", 50, 250);

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
