// 圆弧测试 - 测试 GPU SDF 圆弧渲染
#include <WCN/WCN.h>
#include <stdio.h>
#include <math.h>

#define WCN_GLFW_IMPLEMENTATION
#include "../../impl/wcn_glfw_impl.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

int main(void) {
    printf("=== WCN Arc Test ===\n");
    
    WCN_GLFW_Window* window = wcn_glfw_create_window(1000, 700, "WCN Arc Test");
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        return -1;
    }
    
    WCN_Context* ctx = wcn_glfw_get_context(window);
    
    printf("Window created successfully\n");
    printf("Starting render loop...\n");
    
    int frame_count = 0;
    while (!wcn_glfw_window_should_close(window)) {
        wcn_glfw_poll_events();
        
        WCN_GLFW_RenderFrame frame;
        if (wcn_glfw_begin_frame(window, &frame)) {
            // 测试 1: 完整圆 (stroke)
            wcn_begin_path(ctx);
            wcn_arc(ctx, 100, 100, 50, 0, M_PI * 2.0f, false);
            wcn_set_stroke_style(ctx, 0xFFFF0000);
            wcn_set_line_width(ctx, 3.0f);
            wcn_stroke(ctx);
            
            // 测试 2: 半圆 (stroke)
            wcn_begin_path(ctx);
            wcn_arc(ctx, 250, 100, 50, 0, M_PI, false);
            wcn_set_stroke_style(ctx, 0xFF00FF00);
            wcn_set_line_width(ctx, 3.0f);
            wcn_stroke(ctx);
            
            // 测试 3: 四分之一圆 (stroke)
            wcn_begin_path(ctx);
            wcn_arc(ctx, 400, 100, 50, 0, M_PI * 0.5f, false);
            wcn_set_stroke_style(ctx, 0xFF0000FF);
            wcn_set_line_width(ctx, 3.0f);
            wcn_stroke(ctx);
            
            // 测试 4: 四分之三圆 (stroke)
            wcn_begin_path(ctx);
            wcn_arc(ctx, 550, 100, 50, 0, M_PI * 1.5f, false);
            wcn_set_stroke_style(ctx, 0xFFFF00FF);
            wcn_set_line_width(ctx, 3.0f);
            wcn_stroke(ctx);
            
            // 测试 5: 逆时针半圆 (stroke)
            wcn_begin_path(ctx);
            wcn_arc(ctx, 700, 100, 50, 0, M_PI, true);
            wcn_set_stroke_style(ctx, 0xFFFFFF00);
            wcn_set_line_width(ctx, 3.0f);
            wcn_stroke(ctx);
            
            // 测试 6: 圆弧连接线 - 从点到圆弧
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 50, 250);
            wcn_line_to(ctx, 100, 250);
            wcn_arc(ctx, 150, 250, 50, M_PI, 0, true);
            wcn_line_to(ctx, 250, 250);
            wcn_set_stroke_style(ctx, 0xFFFF8000);
            wcn_set_line_width(ctx, 4.0f);
            wcn_stroke(ctx);
            
            // 测试 7: 多段圆弧连接
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 300, 300);
            wcn_arc(ctx, 350, 300, 50, M_PI, M_PI * 0.5f, true);
            wcn_arc(ctx, 450, 300, 50, M_PI * 0.5f, 0, true);
            wcn_set_stroke_style(ctx, 0xFF00FFFF);
            wcn_set_line_width(ctx, 3.0f);
            wcn_stroke(ctx);
            
            // 测试 8: 小角度圆弧 (30度)
            wcn_begin_path(ctx);
            wcn_arc(ctx, 100, 400, 50, 0, M_PI / 6.0f, false);
            wcn_set_stroke_style(ctx, 0xFFFF0080);
            wcn_set_line_width(ctx, 5.0f);
            wcn_stroke(ctx);
            
            // 测试 9: 从非零角度开始的圆弧
            wcn_begin_path(ctx);
            wcn_arc(ctx, 250, 400, 50, M_PI * 0.25f, M_PI * 0.75f, false);
            wcn_set_stroke_style(ctx, 0xFF8000FF);
            wcn_set_line_width(ctx, 4.0f);
            wcn_stroke(ctx);
            
            // 测试 10: 完整圆 (fill)
            wcn_begin_path(ctx);
            wcn_arc(ctx, 400, 400, 50, 0, M_PI * 2.0f, false);
            wcn_set_fill_style(ctx, 0x8000FF00);
            wcn_fill(ctx);
            
            // 测试 11: 半圆 (fill)
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 550, 400);
            wcn_arc(ctx, 550, 400, 50, 0, M_PI, false);
            wcn_close_path(ctx);
            wcn_set_fill_style(ctx, 0x800000FF);
            wcn_fill(ctx);
            
            // 标签
            wcn_set_fill_style(ctx, 0xFF000000);
            
            wcn_glfw_end_frame(window, &frame);
        }
        
        frame_count++;
        if (frame_count % 100 == 0) {
            printf("Frame %d\n", frame_count);
        }
    }
    
    wcn_glfw_destroy_window(window);
    
    printf("Test completed. Rendered %d frames\n", frame_count);
    return 0;
}
