# WCN (WebGPU Canvas Native) 项目概述

## 项目简介

WCN 是一个与窗口、资源处理无关的基于 webgpu-native 的 W3C 规范的纯 C 语言 Canvas 库，支持 2D 与 3D 渲染。该项目旨在提供一个高性能、跨平台的图形渲染解决方案，符合 C11 标准，并支持 WASM 平台。

## 项目特点

- 纯 C 语言实现，符合 C11 标准
- 基于 WebGPU 的渲染后端，支持高性能图形渲染
- 实现 W3C Canvas API 规范
- 跨平台支持（Windows, Linux, macOS）
- 支持 WASM 平台
- 可扩展的字体/图像解码注册机制
- 函数式编程风格，使用安全宏

## 系统架构

```
WCN (WebGPU Canvas Native)
├── WCN_Context (上下文管理)
├── WCN_Canvas (画布核心)
├── 渲染后端接口
│   └── WebGPU 实现
├── 核心功能模块
│   ├── 2D绘图API
│   ├── 3D渲染API
│   ├── 资源管理
│   ├── 上下文管理
│   └── 变换矩阵支持
├── 扩展机制
│   ├── 字体解码注册器
│   ├── 图像解码注册器
│   └── 第三方解码实现
└── 外部依赖
    └── webgpu-native
```

## 项目结构

```
WCN/
├── examples/           # 使用示例
│   ├── GLFW/          # GLFW 示例
│   ├── Raylib/        # Raylib 示例
│   ├── SDL_gpu/       # SDL_gpu 示例
│   ├── SDL2/          # SDL2 示例
│   ├── SDL3/          # SDL3 示例
│   ├── SFML/          # SFML 示例
│   └── Sokol/         # Sokol 示例
├── external/           # 外部依赖
│   └── wgpu/          # webgpu-native 库
│       ├── include/   # 头文件
│       └── lib/       # 库文件
├── include/            # 公共头文件
│   └── WCN/
│       └── WCN.h      # 主要 API 头文件
├── src/                # 源代码
│   ├── wcn.c          # 核心实现
│   ├── shader_manager.h # 着色器管理器头文件
│   └── shader_manager.c # 着色器管理器实现
├── CMakeLists.txt      # CMake 构建配置
├── make.build.bat      # Windows 构建脚本
├── .clang-format       # 代码格式化配置
├── .clangd             # clangd 配置
└── README.md           # 项目说明
```

## 核心功能

### 1. 上下文管理
- `wcn_init_context()` - 初始化 WCN 上下文
- `wcn_destroy_context()` - 销毁 WCN 上下文

### 2. 画布操作
- `wcn_create_canvas()` - 创建画布
- `wcn_destroy_canvas()` - 销毁画布
- `wcn_canvas_set_size()` - 设置画布尺寸
- `wcn_canvas_set_texture_view()` - 设置纹理视图

### 3. 渲染控制
- `wcn_begin_render_pass()` - 开始渲染通道
- `wcn_end_render_pass()` - 结束渲染通道
- `wcn_submit()` - 提交渲染命令
- `wcn_flush_batch()` - 批渲染刷新

### 4. 基本形状绘制
- `wcn_fill_rect()` - 填充矩形
- `wcn_stroke_rect()` - 描边矩形
- `wcn_clear_rect()` - 清除矩形区域
- `wcn_fill()` - 填充整个画布
- `wcn_stroke()` - 描边整个画布
- `wcn_clear()` - 清除画布

### 5. 路径操作
- `wcn_begin_path()` - 开始路径
- `wcn_close_path()` - 关闭路径
- `wcn_move_to()` - 移动到指定点
- `wcn_line_to()` - 画线到指定点
- `wcn_bezier_curve_to()` - 画贝塞尔曲线
- `wcn_quadratic_curve_to()` - 画二次贝塞尔曲线
- `wcn_arc()` - 画弧线
- `wcn_fill_path()` - 填充路径
- `wcn_stroke_path()` - 描边路径
- `wcn_clip()` - 裁剪路径 (W3C规范)

### 6. 状态管理
- `wcn_save()` - 保存当前绘图状态
- `wcn_restore()` - 恢复之前保存的绘图状态

### 7. 变换操作
- `wcn_translate()` - 平移坐标系
- `wcn_rotate()` - 旋转坐标系
- `wcn_scale()` - 缩放坐标系

### 8. 样式设置
- `wcn_set_fill_color()` - 设置填充颜色
- `wcn_set_stroke_color()` - 设置描边颜色
- `wcn_set_line_width()` - 设置线条宽度

## 构建和运行

### 系统要求
- CMake 3.11 或更高版本
- 支持 C11 标准的编译器
- WebGPU 依赖库（通过 external/download.bat 下载）

### 构建步骤

#### Windows 平台
1. 下载 WebGPU 依赖库：
   ```bash
   cd external
   download.bat
   ```

2. 构建项目：
   ```bash
   make.build.bat
   ```

#### 其他平台
1. 下载 WebGPU 依赖库：
   ```bash
   cd external
   ./download.sh
   ```

2. 创建构建目录并构建：
   ```bash
   mkdir build
   cd build
   cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   make
   ```

### 运行示例

构建完成后，可以在 build/examples 目录下找到各种示例程序。例如，运行 GLFW 示例：

```bash
cd build/examples/GLFW
./transform_test
```

## 开发约定

### 代码风格
- 使用 `.clang-format` 配置文件进行代码格式化
- 基于 LLVM 风格，4 空格缩进
- 指针左对齐：`Type* ptr` 而非 `Type *ptr`
- 每行最大字符数：100

### 调试支持
- 定义 `DEBUG` 宏启用调试输出
- 使用 `WCN_DEBUG_PRINT` 宏输出调试信息

### 安全宏
- `WCN_SAVE_RESTORE` - 自动保存和恢复绘图状态
- `WCN_BEGIN_END_RENDER_PASS` - 自动开始和结束渲染通道

### 内存管理
- 使用 malloc/free 进行内存分配
- 所有资源都有对应的创建和销毁函数
- 使用引用计数管理 WebGPU 资源

## 扩展机制

### 着色器管理
- `WCN_ShaderManager` 负责着色器的加载和管理
- 支持颜色着色器和纹理着色器
- 可扩展更多着色器类型

### 解码器注册
- 支持字体解码器注册
- 支持图像解码器注册
- 可通过第三方实现扩展

## 后续计划

1. WCW (WebGPU Canvas Web) - 一个 WCN Wasm + TS 的 npm 包
2. 完整的 W3C Canvas API 实现
3. 3D 渲染功能支持
4. 更多的扩展解码器实现

## 许可证

[待定]