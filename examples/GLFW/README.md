# WCN GLFW 示例程序

本目录包含使用 GLFW 的 WCN 测试和示例程序。

## 核心测试程序

### 1. simple_glfw_test
**文件**: `simple_glfw_test.c`

基础功能测试，展示：
- 基本图形绘制（矩形、圆、路径）
- 变换操作（平移、旋转）
- 使用 `wcn_glfw_impl.h` 的简化 API

**用途**: 验证基本渲染功能和 GLFW 集成

### 2. minimal_test
**文件**: `minimal_test.c`

最小化测试程序，只绘制一个红色矩形。

**用途**: 
- 快速验证环境配置
- 作为新项目的起点模板
- 调试基础渲染问题

### 3. transform_test
**文件**: `transform_test.c`

完整的变换系统测试，包括：
- 平移 (translate)
- 旋转 (rotate)
- 缩放 (scale)
- 自定义变换矩阵 (transform)
- 直接设置变换 (set_transform)
- 重置变换 (reset_transform)
- 状态栈 (save/restore)

**用途**: 验证变换系统的正确性

### 4. line_cap_join_test
**文件**: `line_cap_join_test.c`

线条样式测试，展示：
- 线帽样式：BUTT, ROUND, SQUARE
- 线连接样式：MITER, ROUND, BEVEL

**用途**: 验证路径描边的线帽和连接样式

### 5. simple_batch_test
**文件**: `simple_batch_test.c`

批次渲染系统测试。

**用途**: 
- 验证批次收集和优化
- 性能测试
- 调试批次系统

## 构建和运行

### 构建所有测试

```bash
cmake -B build
cmake --build build
```

### 运行特定测试

```bash
# Windows
.\build\examples\GLFW\simple_glfw_test.exe
.\build\examples\GLFW\minimal_test.exe
.\build\examples\GLFW\transform_test.exe
.\build\examples\GLFW\line_cap_join_test.exe
.\build\examples\GLFW\simple_batch_test.exe

# Linux/macOS
./build/examples/GLFW/simple_glfw_test
./build/examples/GLFW/minimal_test
./build/examples/GLFW/transform_test
./build/examples/GLFW/line_cap_join_test
./build/examples/GLFW/simple_batch_test
```

## 使用 wcn_glfw_impl.h

所有测试程序都使用 `wcn_glfw_impl.h` 提供的简化 API：

```c
#define WCN_GLFW_IMPLEMENTATION
#include "../../impl/wcn_glfw_impl.h"

int main(void) {
    // 创建窗口
    WCN_GLFW_Window* window = wcn_glfw_create_window(800, 600, "Title");
    WCN_Context* ctx = wcn_glfw_get_context(window);
    
    // 渲染循环
    while (!wcn_glfw_window_should_close(window)) {
        wcn_glfw_poll_events();
        
        WCN_GLFW_RenderFrame frame;
        if (wcn_glfw_begin_frame(window, &frame)) {
            // 绘制操作
            wcn_fill_rect(ctx, 10, 10, 100, 100);
            
            wcn_glfw_end_frame(window, &frame);
        }
    }
    
    // 清理
    wcn_glfw_destroy_window(window);
    return 0;
}
```

## 批次渲染系统

所有渲染操作自动使用批次系统：
- 绘制调用自动收集到批次
- `wcn_end_frame()` 自动优化和渲染批次
- 状态变化自动管理

详见：[批次系统迁移指南](../../docs/BATCH_SYSTEM_MIGRATION.md)

## 变换系统

变换系统使用状态栈和独立的 GPU 状态槽位：
- 每个批次保存状态快照
- 每帧自动重置状态栈
- 支持嵌套的 save/restore

详见：[变换系统状态槽位修复](../../docs/TRANSFORM_STATE_SLOT_FIX.md)

## 依赖

- GLFW 3.x
- WebGPU (wgpu-native)
- WCN 库

## 相关文档

- [批次渲染实现](../../docs/BATCH_RENDERING_IMPLEMENTATION.md)
- [GLFW 实现改进](../../docs/GLFW_IMPL_IMPROVEMENTS.md)
- [变换系统设计](../../docs/TRANSFORM_SYSTEM.md)
