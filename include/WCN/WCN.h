#ifndef WCN_H
#define WCN_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <webgpu/webgpu.h>

static inline void ___prev_clangd___printf() { (void)printf; }

// 调试宏定义
#define WCN_DEBUG_PRINT(fmt, ...)                                              \
  do {                                                                         \
    printf("[WCN DEBUG] " fmt "\n", ##__VA_ARGS__);                            \
  } while (0)

#ifdef __cplusplus
extern "C" {
#endif

// 前向声明
typedef struct WCN_Context WCN_Context;
typedef struct WCN_Canvas WCN_Canvas;
struct WCN_ShaderManager; // 前向声明 ShaderManager

// 颜色结构体
typedef struct {
  float r, g, b, a;
} WCN_Color;

// 矩形结构体
typedef struct {
  float x, y, width, height;
} WCN_Rect;

// 点结构体
typedef struct {
  float x, y;
} WCN_Point;

// 线段结构体
typedef struct {
  WCN_Point start;
  WCN_Point end;
} WCN_Line;

// 圆结构体
typedef struct {
  WCN_Point center;
  float radius;
} WCN_Circle;

// 顶点结构体
typedef struct {
  float position[2]; // x, y
  float color[4];    // r, g, b, a
} WCN_Vertex;

// 初始化上下文
WCN_Context *wcn_init_context(void);

// 销毁上下文
void wcn_destroy_context(WCN_Context *context);

// 创建 Canvas
WCN_Canvas *wcn_create_canvas(WCN_Context *context, WGPUDevice device,
                              WGPUTextureFormat format, uint32_t width,
                              uint32_t height);

// 销毁 Canvas
void wcn_destroy_canvas(WCN_Canvas *canvas);

// 设置纹理视图
void wcn_canvas_set_texture_view(WCN_Canvas *canvas, WGPUTextureView view);

// 设置Canvas尺寸
void wcn_canvas_set_size(WCN_Canvas *canvas, uint32_t width, uint32_t height);

// 开始渲染通道
void wcn_begin_render_pass(WCN_Canvas *canvas);

// 结束渲染通道
void wcn_end_render_pass(WCN_Canvas *canvas);

// 设置填充颜色
void wcn_set_fill_color(WCN_Canvas *canvas, WCN_Color color);

// 设置描边颜色
void wcn_set_stroke_color(WCN_Canvas *canvas, WCN_Color color);

// 设置线条宽度
void wcn_set_line_width(WCN_Canvas *canvas, float width);

// 填充矩形
void wcn_fill_rect(WCN_Canvas *canvas, float x, float y, float width,
                   float height);

// 描边矩形
void wcn_stroke_rect(WCN_Canvas *canvas, float x, float y, float width,
                     float height);

// 清除矩形区域
void wcn_clear_rect(WCN_Canvas *canvas, float x, float y, float width,
                    float height);

// 填充整个 Canvas
void wcn_fill(WCN_Canvas *canvas);

// 描边整个 Canvas
void wcn_stroke(WCN_Canvas *canvas);

// 清除 Canvas
void wcn_clear(WCN_Canvas *canvas, WCN_Color color);

// 开始路径
void wcn_begin_path(WCN_Canvas *canvas);

// 关闭路径
void wcn_close_path(WCN_Canvas *canvas);

// 移动到指定点
void wcn_move_to(WCN_Canvas *canvas, float x, float y);

// 画线到指定点
void wcn_line_to(WCN_Canvas *canvas, float x, float y);

// 画贝塞尔曲线
void wcn_bezier_curve_to(WCN_Canvas *canvas, float cp1x, float cp1y, float cp2x,
                         float cp2y, float x, float y);

// 画二次贝塞尔曲线
void wcn_quadratic_curve_to(WCN_Canvas *canvas, float cpx, float cpy, float x,
                            float y);

// 画弧线
void wcn_arc(WCN_Canvas *canvas, float x, float y, float radius,
             float startAngle, float endAngle, bool anticlockwise);

// 填充路径
void wcn_fill_path(WCN_Canvas *canvas);

// 描边路径
void wcn_stroke_path(WCN_Canvas *canvas);

// 裁剪路径
void wcn_clip_path(WCN_Canvas *canvas);

// 清除矩形区域
void wcn_clear_rect(WCN_Canvas *canvas, float x, float y, float width,
                    float height);

// 保存当前绘图状态
void wcn_save(WCN_Canvas *canvas);

// 恢复之前保存的绘图状态
void wcn_restore(WCN_Canvas *canvas);

// 平移坐标系
void wcn_translate(WCN_Canvas *canvas, float x, float y);

// 旋转坐标系
void wcn_rotate(WCN_Canvas *canvas, float angle);

// 缩放坐标系
void wcn_scale(WCN_Canvas *canvas, float x, float y);

// 获取 WebGPU 渲染通道描述符
WGPURenderPassDescriptor wcn_get_render_pass_descriptor(WCN_Canvas *canvas);

// 提交渲染命令
void wcn_submit(WCN_Canvas *canvas);

// 批渲染刷新函数
void wcn_flush_batch(WCN_Canvas *canvas);

// 设置抗锯齿
// void wcn_set_antialiasing(WCN_Canvas *canvas, bool enabled);

/**
打开一个作用域 WCN_SAVE_RESTORE(canvas) {
  WCN_Color yellow = {1.0f, 1.0f, 0.0f, 1.0f};
  wcn_set_fill_color(canvas, yellow);
  wcn_scale(canvas, 1.5f, 1.5f);
  // 调整位置和大小以适应缩放
  wcn_fill_rect(canvas, center_x - rect_size / 2, center_y + rect_size,
                rect_size, rect_size);
  printf("Drawing yellow rectangle at (%.1f, %.1f) with scaling\n",
         center_x - rect_size / 2, center_y + rect_size);
}

->
wcn_save(canvas);
{
  WCN_Color yellow = {1.0f, 1.0f, 0.0f, 1.0f};
  wcn_set_fill_color(canvas, yellow);
  wcn_scale(canvas, 1.5f, 1.5f);
  // 调整位置和大小以适应缩放
  wcn_fill_rect(canvas, center_x - rect_size / 2, center_y + rect_size,
                rect_size, rect_size);
  printf("Drawing yellow rectangle at (%.1f, %.1f) with scaling\n",
         center_x - rect_size / 2, center_y + rect_size);
}
wcn_restore(canvas);
*/
#define WCN_SAVE_RESTORE(canvas)                                               \
  wcn_save(canvas);                                                            \
  for (int wcn_internal_once = 1; wcn_internal_once--; wcn_restore(canvas))

#define WCN_BEGIN_END_RENDER_PASS(canvas)                                      \
  wcn_begin_render_pass(canvas);                                               \
  for (int wcn_internal_once = 1; wcn_internal_once--;                         \
       wcn_end_render_pass(canvas))

#ifdef __cplusplus
}
#endif

#endif // WCN_H