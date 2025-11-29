// 最小化测试 - 使用 wcn_glfw_impl.h
#include <WCN/WCN.h>
#include <stdio.h>

#define WCN_GLFW_IMPLEMENTATION
#include "../../impl/wcn_glfw_impl.h"

int main(void) {
    printf("=== Minimal WCN Test ===\n");
    printf("Drawing ONE red square at (100, 100)\n\n");
    
    // 创建窗口
    WCN_GLFW_Window* window = wcn_glfw_create_window(800, 600, "Minimal Test");
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
            // 绘制一个红色矩形
            wcn_set_fill_style(ctx, 0xFFFF0000);
            wcn_fill_rect(ctx, 100, 100, 100, 100);
            
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
