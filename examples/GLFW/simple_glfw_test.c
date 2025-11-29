// 简单的 GLFW 测试 - 展示改进后的 wcn_glfw_impl.h API
#include <WCN/WCN.h>
#include <stdio.h>
#include <math.h>

#define WCN_GLFW_IMPLEMENTATION
#include "../../impl/wcn_glfw_impl.h"

int main(void) {
    printf("=== WCN Simple GLFW Test ===\n");
    
    // 创建窗口
    WCN_GLFW_Window* window = wcn_glfw_create_window(800, 600, "WCN Simple Test");
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
    while (!wcn_glfw_window_should_close(window)) {
        wcn_glfw_poll_events();
        
        // 开始渲染帧
        WCN_GLFW_RenderFrame frame;
        if (wcn_glfw_begin_frame(window, &frame)) {
            uint32_t width, height;
            wcn_glfw_get_size(window, &width, &height);
            
            // 绘制一个旋转的矩形（红色）- 固定旋转 45 度
            wcn_save(ctx);
            wcn_translate(ctx, (float)width / 2.0f, (float)height / 2.0f);
            wcn_rotate(ctx, 0.785f); // 45 度
            wcn_set_fill_style(ctx, 0xFFFF0000);
            wcn_fill_rect(ctx, -50, -50, 100, 100);
            wcn_restore(ctx);
            
            // 绘制一个圆（绿色）
            wcn_begin_path(ctx);
            wcn_arc(ctx, (float)width / 2.0f + 150.0f, (float)height / 2.0f, 40, 0, 2 * 3.14159f, false);
            wcn_set_fill_style(ctx, 0xFF00FF00);
            wcn_fill(ctx);
            
            // 绘制一条线（蓝色）
            wcn_begin_path(ctx);
            wcn_move_to(ctx, 100, 100);
            wcn_line_to(ctx, 200, 150);
            wcn_line_to(ctx, 150, 200);
            wcn_set_stroke_style(ctx, 0xFF0000FF);
            wcn_set_line_width(ctx, 5.0f);
            wcn_stroke(ctx);
            
            // 绘制一个静态矩形（黄色）
            wcn_set_fill_style(ctx, 0xFFFFFF00);
            wcn_fill_rect(ctx, 50, 50, 100, 80);
            
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
