#include <WCN/WCN.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#define WCN_GLFW_IMPLEMENTATION
#include "../../impl/wcn_glfw_impl.h"
#include "../../impl/wcn_stb_image_impl.h"

// 2x2 uncompressed TGA (red, green, blue, white)
static const uint8_t WCN_SAMPLE_TGA[] = {
    0x00,0x00,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x02,0x00,0x02,0x00,0x18,0x00,
    0x00,0x00,0xFF, 0x00,0xFF,0x00,
    0xFF,0x00,0x00, 0xFF,0xFF,0xFF
};

int main(void) {
    printf("=== WCN Image Decoder (stb_image) Demo ===\n");

    WCN_GLFW_Window* window = wcn_glfw_create_window(800, 600, "WCN Image Decoder Test");
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        return -1;
    }

    WCN_Context* ctx = wcn_glfw_get_context(window);
    WCN_ImageDecoder* decoder = wcn_get_stb_image_decoder();
    wcn_register_image_decoder(ctx, decoder);

    WCN_ImageData* image = wcn_decode_image(ctx, WCN_SAMPLE_TGA, sizeof(WCN_SAMPLE_TGA));
    if (!image) {
        fprintf(stderr, "Failed to decode embedded image\n");
        wcn_glfw_destroy_window(window);
        return -1;
    }

    printf("Decoded image: %ux%u px\n", image->width, image->height);

    float angle = 0.0f;
    while (!wcn_glfw_window_should_close(window)) {
        wcn_glfw_poll_events();

        WCN_GLFW_RenderFrame frame;
        if (wcn_glfw_begin_frame(window, &frame)) {
            wcn_set_fill_style(ctx, 0xFFF5F5F5);
            wcn_fill_rect(ctx, 0, 0, 800, 600);

            // Draw the raw image
            wcn_draw_image(ctx, image, 100, 100);

            // Draw a scaled version
            wcn_draw_image_scaled(ctx, image, 300, 80, 160, 160);

            // Draw with cropping
            wcn_draw_image_source(ctx, image, 0, 1, 1, 1, 520, 100, 160, 160);

            // Animated tile strip
            float offset = 0.0f;
            for (int i = 0; i < 10; ++i) {
                float size = 48.0f + 8.0f * (float)sin(angle + offset);
                wcn_draw_image_scaled(ctx, image, 120 + i * 60.0f, 320, size, size);
                offset += 0.3f;
            }

            angle += 0.02f;
            wcn_glfw_end_frame(window, &frame);
        }
    }

    wcn_destroy_image_data(image);
    wcn_glfw_destroy_window(window);
    return 0;
}
