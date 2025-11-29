// 线帽和连接样式测试 - 使用 wcn_glfw_impl.h
#include <WCN/WCN.h>
#include <stdio.h>

#define WCN_GLFW_IMPLEMENTATION
#include "../../impl/wcn_glfw_impl.h"

int main(void) {
    printf("=== Line Cap and Join Test ===\n");
    
    // 创建窗口
    WCN_GLFW_Window* window = wcn_glfw_create_window(1000, 600, "Line Cap & Join Test");
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        return -1;
    }
    
    // 获取 WCN 上下文
    WCN_Context* ctx = wcn_glfw_get_context(window);
    
    printf("Window created successfully\n");
    printf("Starting render loop...\n");
    
    // 主循环
    int frame_count = 0;
    while (!wcn_glfw_window_should_close(window) && frame_count < 300) {
        wcn_glfw_poll_events();
        
        // 开始渲染帧
        WCN_GLFW_RenderFrame frame;
        if (wcn_glfw_begin_frame(window, &frame)) {
            uint32_t width, height;
            wcn_glfw_get_size(window, &width, &height);
            
            // 清除背景为白色
            wcn_set_fill_style(ctx, 0xFFFFFFFF);
            wcn_fill_rect(ctx, 0, 0, (float)width, (float)height);
            
            // 设置线条样式
            wcn_set_stroke_style(ctx, 0xFF000000);  // 黑色
            wcn_set_line_width(ctx, 20.0f);
            
            if (frame_count == 0) {
                printf("Drawing lines with different cap styles:\n");
                printf("  - BUTT cap at y=100\n");
                printf("  - ROUND cap at y=200\n");
                printf("  - SQUARE cap at y=300\n\n");
            }
            
            // 测试线帽样式
            // BUTT (默认)
            wcn_set_line_cap(ctx, WCN_LINE_CAP_BUTT);
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 100, 100);
            wcn_line_to(ctx, 300, 100);
            wcn_stroke(ctx);
            
            // ROUND
            wcn_set_line_cap(ctx, WCN_LINE_CAP_ROUND);
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 100, 200);
            wcn_line_to(ctx, 300, 200);
            wcn_stroke(ctx);
            
            // SQUARE
            wcn_set_line_cap(ctx, WCN_LINE_CAP_SQUARE);
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 100, 300);
            wcn_line_to(ctx, 300, 300);
            wcn_stroke(ctx);
            
            if (frame_count == 0) {
                printf("Drawing angles with different join styles:\n");
                printf("  - MITER join at x=500\n");
                printf("  - ROUND join at x=650\n");
                printf("  - BEVEL join at x=800\n\n");
            }
            
            // 测试线连接样式
            // MITER (默认)
            wcn_set_line_join(ctx, WCN_LINE_JOIN_MITER);
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 500, 100);
            wcn_line_to(ctx, 550, 200);
            wcn_line_to(ctx, 600, 100);
            wcn_stroke(ctx);
            
            // ROUND
            wcn_set_line_join(ctx, WCN_LINE_JOIN_ROUND);
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 650, 100);
            wcn_line_to(ctx, 700, 200);
            wcn_line_to(ctx, 750, 100);
            wcn_stroke(ctx);
            
            // BEVEL
            wcn_set_line_join(ctx, WCN_LINE_JOIN_BEVEL);
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 800, 100);
            wcn_line_to(ctx, 850, 200);
            wcn_line_to(ctx, 900, 100);
            wcn_stroke(ctx);
            
            // 结束渲染帧
            wcn_glfw_end_frame(window, &frame);
        }
        
        frame_count++;
        
        // 每 100 帧打印一次
        if (frame_count % 100 == 0) {
            printf("Frame %d\n", frame_count);
        }
    }
    
    // 清理
    wcn_glfw_destroy_window(window);
    
    printf("Test completed. Rendered %d frames\n", frame_count);
    return 0;
}
