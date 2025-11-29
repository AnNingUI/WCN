# WCN WebAssembly Build Guide

本文档说明如何将 WCN 编译为 WebAssembly 并在浏览器中使用。

## 概述

WCN 提供了完整的 WebAssembly 导出支持，允许你在浏览器中使用所有 Canvas 2D API。

## 文件说明

### 核心文件

- `include/WCN/WCN_WASM.h` - WASM 导出宏定义
- `src/wcn_wasm_exports.c` - WASM 导出实现和辅助函数

### 导出宏

```c
// 导出所有 WCN 函数
WCN_WASM_EXPORT_ALL()

// 或者分类导出
WCN_WASM_EXPORT_CONTEXT()    // 上下文管理
WCN_WASM_EXPORT_STATE()      // 状态管理
WCN_WASM_EXPORT_RECT()       // 矩形操作
WCN_WASM_EXPORT_PATH()       // 路径操作
WCN_WASM_EXPORT_STYLE()      // 样式设置
WCN_WASM_EXPORT_TRANSFORM()  // 变换操作
WCN_WASM_EXPORT_TEXT()       // 文本渲染
WCN_WASM_EXPORT_IMAGE()      // 图像操作
WCN_WASM_EXPORT_HELPERS()    // 辅助函数
```

## 编译步骤

### 1. 安装 Emscripten

```bash
# 下载 Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk

# 安装最新版本
./emsdk install latest
./emsdk activate latest

# 设置环境变量
source ./emsdk_env.sh
```

### 2. 编译 WCN 为 WASM

```bash
# 创建构建目录
mkdir build-wasm
cd build-wasm

# 使用 Emscripten 配置 CMake
emcmake cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWCN_BUILD_WASM=ON

# 编译
emmake make

# 或者直接使用 emcc 编译
emcc \
  -O3 \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
  -s MODULARIZE=1 \
  -s EXPORT_NAME='createWCNModule' \
  -I../include \
  -I../external/wgpu/include \
  ../src/*.c \
  ../src/wcn_wasm_exports.c \
  -o wcn.js
```

### 3. 编译选项说明

- `-O3`: 最高优化级别
- `-s WASM=1`: 生成 WebAssembly
- `-s ALLOW_MEMORY_GROWTH=1`: 允许内存动态增长
- `-s EXPORTED_RUNTIME_METHODS`: 导出运行时方法
- `-s MODULARIZE=1`: 生成模块化输出
- `-s EXPORT_NAME='createWCNModule'`: 模块名称

## JavaScript 使用示例

### 1. 加载 WCN 模块

```javascript
// 加载 WCN WASM 模块
const WCN = await createWCNModule();

// 获取导出的函数
const {
  // 上下文管理
  _wcn_create_context,
  _wcn_destroy_context,
  _wcn_begin_frame,
  _wcn_end_frame,
  
  // 绘图 API
  _wcn_fill_rect,
  _wcn_stroke_rect,
  _wcn_begin_path,
  _wcn_move_to,
  _wcn_line_to,
  _wcn_fill,
  _wcn_stroke,
  
  // 样式 API
  _wcn_set_fill_style,
  _wcn_set_stroke_style,
  _wcn_set_line_width,
  
  // 变换 API
  _wcn_translate,
  _wcn_rotate,
  _wcn_scale,
  
  // 文本 API
  _wcn_fill_text,
  _wcn_set_font,
  
  // 辅助函数
  _wcn_wasm_malloc,
  _wcn_wasm_free,
  _wcn_wasm_get_version,
} = WCN;
```

### 2. 创建上下文

```javascript
// 获取 WebGPU 设备
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// 创建 GPU 资源结构
const gpuResourcesPtr = _wcn_wasm_create_gpu_resources(
  0, // instance (可以为 0)
  device,
  device.queue,
  0  // surface (在浏览器中可以为 0)
);

// 创建 WCN 上下文
const ctx = _wcn_create_context(gpuResourcesPtr);
```

### 3. 绘制示例

```javascript
// 开始帧
_wcn_begin_frame(ctx, canvas.width, canvas.height, surfaceFormat);

// 设置填充颜色 (RGBA: 0xAABBGGRR)
_wcn_set_fill_style(ctx, 0xFFFF0000); // 红色

// 绘制矩形
_wcn_fill_rect(ctx, 50, 50, 100, 100);

// 绘制路径
_wcn_begin_path(ctx);
_wcn_move_to(ctx, 200, 200);
_wcn_line_to(ctx, 300, 200);
_wcn_line_to(ctx, 250, 300);
_wcn_fill(ctx);

// 应用变换
_wcn_translate(ctx, 400, 200);
_wcn_rotate(ctx, Math.PI / 4);
_wcn_set_fill_style(ctx, 0xFF0000FF); // 蓝色
_wcn_fill_rect(ctx, -50, -50, 100, 100);

// 绘制文本
const textPtr = allocateUTF8("Hello, WCN!");
_wcn_fill_text(ctx, textPtr, 100, 400);
_wcn_wasm_free(textPtr);

// 结束帧
_wcn_end_frame(ctx);
```

### 4. 封装为 Canvas2D API

```javascript
class WCNCanvas {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = null;
    this.WCN = null;
  }
  
  async init() {
    // 加载 WCN 模块
    this.WCN = await createWCNModule();
    
    // 初始化 WebGPU
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    
    // 创建上下文
    const gpuResources = this.WCN._wcn_wasm_create_gpu_resources(
      0, device, device.queue, 0
    );
    this.ctx = this.WCN._wcn_create_context(gpuResources);
  }
  
  beginFrame() {
    this.WCN._wcn_begin_frame(
      this.ctx,
      this.canvas.width,
      this.canvas.height,
      0 // surface format
    );
  }
  
  endFrame() {
    this.WCN._wcn_end_frame(this.ctx);
  }
  
  fillRect(x, y, width, height) {
    this.WCN._wcn_fill_rect(this.ctx, x, y, width, height);
  }
  
  strokeRect(x, y, width, height) {
    this.WCN._wcn_stroke_rect(this.ctx, x, y, width, height);
  }
  
  set fillStyle(color) {
    // 转换 CSS 颜色为 RGBA uint32
    const rgba = this.parseColor(color);
    this.WCN._wcn_set_fill_style(this.ctx, rgba);
  }
  
  set strokeStyle(color) {
    const rgba = this.parseColor(color);
    this.WCN._wcn_set_stroke_style(this.ctx, rgba);
  }
  
  set lineWidth(width) {
    this.WCN._wcn_set_line_width(this.ctx, width);
  }
  
  translate(x, y) {
    this.WCN._wcn_translate(this.ctx, x, y);
  }
  
  rotate(angle) {
    this.WCN._wcn_rotate(this.ctx, angle);
  }
  
  scale(x, y) {
    this.WCN._wcn_scale(this.ctx, x, y);
  }
  
  save() {
    this.WCN._wcn_save(this.ctx);
  }
  
  restore() {
    this.WCN._wcn_restore(this.ctx);
  }
  
  beginPath() {
    this.WCN._wcn_begin_path(this.ctx);
  }
  
  moveTo(x, y) {
    this.WCN._wcn_move_to(this.ctx, x, y);
  }
  
  lineTo(x, y) {
    this.WCN._wcn_line_to(this.ctx, x, y);
  }
  
  fill() {
    this.WCN._wcn_fill(this.ctx);
  }
  
  stroke() {
    this.WCN._wcn_stroke(this.ctx);
  }
  
  fillText(text, x, y) {
    const textPtr = this.allocateString(text);
    this.WCN._wcn_fill_text(this.ctx, textPtr, x, y);
    this.WCN._wcn_wasm_free(textPtr);
  }
  
  // 辅助方法
  parseColor(color) {
    // 简化版本，实际需要完整的 CSS 颜色解析
    if (color.startsWith('#')) {
      const hex = color.slice(1);
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      const a = hex.length === 8 ? parseInt(hex.slice(6, 8), 16) : 255;
      return (a << 24) | (b << 16) | (g << 8) | r;
    }
    return 0xFFFFFFFF; // 默认白色
  }
  
  allocateString(str) {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(str + '\0');
    const ptr = this.WCN._wcn_wasm_malloc(bytes.length);
    this.WCN.HEAPU8.set(bytes, ptr);
    return ptr;
  }
}

// 使用示例
const canvas = document.getElementById('myCanvas');
const wcn = new WCNCanvas(canvas);

await wcn.init();

function render() {
  wcn.beginFrame();
  
  wcn.fillStyle = '#FF0000';
  wcn.fillRect(50, 50, 100, 100);
  
  wcn.save();
  wcn.translate(200, 200);
  wcn.rotate(Math.PI / 4);
  wcn.fillStyle = '#0000FF';
  wcn.fillRect(-50, -50, 100, 100);
  wcn.restore();
  
  wcn.endFrame();
  
  requestAnimationFrame(render);
}

render();
```

## 辅助函数

WCN 提供了一些 WASM 特定的辅助函数：

### 内存管理

```javascript
// 分配内存
const ptr = _wcn_wasm_malloc(size);

// 释放内存
_wcn_wasm_free(ptr);
```

### 枚举值获取

```javascript
// 文本对齐
const LEFT = _wcn_wasm_text_align_left();
const CENTER = _wcn_wasm_text_align_center();
const RIGHT = _wcn_wasm_text_align_right();

// 线帽样式
const BUTT = _wcn_wasm_line_cap_butt();
const ROUND = _wcn_wasm_line_cap_round();
const SQUARE = _wcn_wasm_line_cap_square();

// 线连接样式
const MITER = _wcn_wasm_line_join_miter();
const ROUND_JOIN = _wcn_wasm_line_join_round();
const BEVEL = _wcn_wasm_line_join_bevel();
```

### GPU 资源管理

```javascript
// 创建 GPU 资源
const resources = _wcn_wasm_create_gpu_resources(
  instance, device, queue, surface
);

// 释放 GPU 资源
_wcn_wasm_free_gpu_resources(resources);
```

### 图像数据管理

```javascript
// 创建图像数据
const imageData = _wcn_wasm_create_image_data(width, height, format);

// 获取图像数据缓冲区
const buffer = _wcn_wasm_get_image_data_buffer(imageData);

// 获取缓冲区大小
const size = _wcn_wasm_get_image_data_size(imageData);

// 释放图像数据
_wcn_wasm_free_image_data(imageData);
```

## 性能优化建议

1. **批量绘制**: 尽量减少 JavaScript 和 WASM 之间的调用次数
2. **内存复用**: 复用分配的内存，避免频繁分配和释放
3. **使用 TypedArray**: 直接操作 WASM 内存以提高性能
4. **避免字符串转换**: 缓存字符串指针，避免重复转换

## 调试

### 启用调试信息

```bash
emcc \
  -g \
  -s ASSERTIONS=1 \
  -s SAFE_HEAP=1 \
  -s STACK_OVERFLOW_CHECK=2 \
  ...
```

### 查看导出的函数

```javascript
console.log(Object.keys(WCN).filter(k => k.startsWith('_wcn')));
```

## 已知限制

1. WebGPU 支持需要浏览器支持（Chrome 113+, Edge 113+）
2. 某些高级功能可能需要额外的 polyfill
3. 性能可能不如原生实现

## 下一步

- 创建完整的 TypeScript 类型定义
- 开发 npm 包 (WCW - WebGPU Canvas Web)
- 提供更多示例和文档

## 参考资源

- [Emscripten 文档](https://emscripten.org/docs/)
- [WebGPU 规范](https://www.w3.org/TR/webgpu/)
- [Canvas 2D API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
