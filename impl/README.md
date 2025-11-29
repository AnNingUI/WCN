# WCN Implementation Files

这个目录包含 WCN 的各种实现文件，用于与不同的库和平台集成。

## 文件说明

### wcn_glfw_impl.h
GLFW 窗口和 WebGPU 集成实现。

**功能**：
- 跨平台窗口创建（Windows、Linux、macOS）
- WebGPU 初始化和配置
- Surface 创建和管理
- WCN 上下文集成

**使用示例**：
```c
#include "impl/wcn_glfw_impl.h"

// 创建窗口
WCN_GLFW_Window* window = wcn_glfw_create_window(800, 600, "My App");
WCN_Context* ctx = wcn_glfw_get_context(window);

// 主循环
while (!wcn_glfw_window_should_close(window)) {
    wcn_glfw_poll_events();
    
    WGPUTextureView view = wcn_glfw_get_current_texture_view(window);
    wcn_begin_frame(ctx, width, height, window->surface_format);
    wcn_begin_render_pass(ctx, view);
    
    // 渲染代码...
    
    wcn_end_render_pass(ctx);
    wcn_submit_commands(ctx);
    wgpuTextureViewRelease(view);
    wcn_glfw_present(window);
    wcn_end_frame(ctx);
}

// 清理
wcn_glfw_destroy_window(window);
```

### wcn_stb_truetype_impl.h
stb_truetype 字体解码器实现。

**功能**：
- 加载 TrueType/OpenType 字体
- 生成 SDF 位图（使用 `stbtt_GetGlyphSDF`）
- 字形度量和测量
- 零依赖（只需要 stb_truetype.h）

**使用示例**：
```c
#include "impl/wcn_stb_truetype_impl.h"

// 获取解码器
WCN_FontDecoder* decoder = wcn_get_stb_truetype_decoder();
wcn_register_font_decoder(ctx, decoder);

// 加载字体
unsigned char* font_data = load_file("font.ttf", &size);
WCN_FontFace* face;
decoder->load_font(font_data, size, &face);

// 设置字体
wcn_set_font(ctx, face, 24.0f);

// 渲染文本
wcn_fill_text_sdf(ctx, "Hello, World!", 100, 100);
```

### wcn_freetype2_impl.h
FreeType2 字体解码器实现（占位）。

**状态**：未实现

**计划功能**：
- 更高质量的字体渲染
- 支持更多字体格式
- 高级排版功能
- 可选的 msdfgen 集成

## 设计原则

1. **Header-Only**: 所有实现都是 header-only，方便集成
2. **解耦**: 核心 WCN 库不依赖这些实现
3. **可选**: 用户可以选择需要的实现
4. **简单**: 提供简单易用的 API

## 集成指南

### 基本集成

```c
// 1. 包含需要的实现
#include "impl/wcn_glfw_impl.h"
#include "impl/wcn_stb_truetype_impl.h"

// 2. 创建窗口和上下文
WCN_GLFW_Window* window = wcn_glfw_create_window(800, 600, "App");
WCN_Context* ctx = wcn_glfw_get_context(window);

// 3. 注册解码器
wcn_register_font_decoder(ctx, wcn_get_stb_truetype_decoder());

// 4. 使用 WCN API
// ...

// 5. 清理
wcn_glfw_destroy_window(window);
```

### 自定义实现

你可以创建自己的实现文件：

```c
// my_custom_impl.h
#ifndef MY_CUSTOM_IMPL_H
#define MY_CUSTOM_IMPL_H

#include "WCN/WCN.h"

// 实现 WCN_FontDecoder 接口
static bool my_load_font(...) { /* ... */ }
static bool my_get_glyph_sdf(...) { /* ... */ }
// ...

static WCN_FontDecoder my_decoder = {
    .load_font = my_load_font,
    .get_glyph_sdf = my_get_glyph_sdf,
    // ...
    .name = "MyDecoder"
};

static inline WCN_FontDecoder* get_my_decoder(void) {
    return &my_decoder;
}

#endif
```

## 依赖

### wcn_glfw_impl.h
- GLFW 3.x
- WebGPU (wgpu-native)
- 平台相关：
  - Windows: windows.h
  - Linux: X11
  - macOS: Cocoa

### wcn_stb_truetype_impl.h
- stb_truetype.h（包含在 external/stb/include）

### wcn_freetype2_impl.h
- FreeType2（未实现）

## 示例程序

查看 `examples/GLFW/` 目录中的示例：

- `sdf_text_test.c` - SDF 文本渲染测试
- `transform_test.c` - 变换系统测试
- `minimal_test.c` - 最小化示例

## 贡献

欢迎贡献新的实现文件：

1. 创建 `wcn_xxx_impl.h` 文件
2. 实现相应的接口（FontDecoder、ImageDecoder 等）
3. 提供使用示例
4. 更新此 README

## 许可

与 WCN 主项目相同的许可证。
