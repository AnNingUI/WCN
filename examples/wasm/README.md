# WCN WebAssembly Examples

这个目录包含 WCN WebAssembly 版本的示例和测试页面。

## ✅ WebGPU 支持状态

**当前状态：** WASM 构建现在使用 **Emscripten 的 Dawn WebGPU 实现**，并且包含了一个**内置的 HTML5 Canvas 包装器**，使得在浏览器中使用 WCN 变得更加简单和直观。

**技术说明：**
- **原生构建**: 使用 `wgpu-native` (Dawn 的 C 绑定)
- **WASM 构建**: 使用 Emscripten 的 Dawn WebGPU (`-sUSE_WEBGPU=1`)
- **Canvas 包装器**: 提供了类似 HTML5 Canvas2D API 的 JavaScript 接口，简化了 Web 开发

**可用功能：**
- ✅ 模块加载
- ✅ 函数导出（72 个函数已导出）
- ✅ API 验证（test-exports.html）
- ✅ 内存管理函数
- ✅ WebGPU 上下文创建（通过 Emscripten Dawn）
- ✅ 浏览器渲染（通过 Canvas 包装器）
- ✅ HTML5 Canvas2D API 兼容接口
- ✅ 文本渲染支持
- ✅ 变换和样式操作

**测试状态：**
- ✅ **test-exports.html**: 验证所有函数导出正确
- ⚠️ **test.html**: 需要测试实际渲染功能

### 如何使用

由于现在链接了 Emscripten 的 Dawn WebGPU，上下文创建应该能正常工作。请测试 `test.html` 查看实际渲染效果。

## 快速开始

### 1. 构建 WASM 版本

#### Linux/macOS:
```bash
# 从项目根目录运行
./build-wasm.sh
```

#### Windows:
```cmd
REM 从项目根目录运行
build-wasm.bat
```

### 2. 启动本地服务器

由于浏览器的安全限制，你需要通过 HTTP 服务器来访问 WASM 文件。

#### 使用 Python (推荐):
```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```

#### 使用 Node.js:
```bash
npx http-server -p 8000
```

#### 使用 PHP:
```bash
php -S localhost:8000
```

### 3. 在浏览器中打开

打开浏览器访问：
```
http://localhost:8000/examples/wasm/test-exports.html  ✅ 可用
http://localhost:8000/examples/wasm/test.html          ⚠️ 受限
```

## 文件说明

- `test-exports.html` - ✅ 函数导出验证测试（可用）
- `test.html` - ✅ 使用 Canvas 包装器的交互式测试页面
- `transform_test.html` - ✅ 变换操作测试页面（使用 Canvas 包装器）
- `simple_test.html` - ✅ 简单功能测试页面
- `font_decoder_test.html` - ✅ 字体解码器测试页面
- `canvas_wrapper_test.html` - ✅ Canvas 包装器基础功能测试
- `canvas_wrapper_full_test.html` - ✅ Canvas 包装器完整功能测试
- `wcn_canvas.js` - ✅ WCN Canvas 包装器 JavaScript 实现
- `README.md` - 本文件

## 浏览器要求

WCN WebAssembly 版本需要支持以下特性的浏览器：

- **WebGPU**: Chrome 113+, Edge 113+
- **WebAssembly**: 所有现代浏览器
- **ES6 Modules**: 所有现代浏览器

### 启用 WebGPU (如果需要)

某些浏览器可能需要手动启用 WebGPU：

#### Chrome/Edge:
1. 访问 `chrome://flags` 或 `edge://flags`
2. 搜索 "WebGPU"
3. 启用 "Unsafe WebGPU" 标志
4. 重启浏览器

## 测试

### 导出验证（可用）
```bash
# 打开 test-exports.html
# 应显示："23 passed, 0 failed"
# 列出所有 72 个导出的函数
```

### 渲染测试（受限）
```bash
# 打开 test.html
# 显示模块加载成功
# 上下文创建失败（预期行为）
# 演示当前限制
```

## 开发自己的应用

### ✅ 使用 Canvas 包装器（推荐）

现在你可以使用内置的 Canvas 包装器来简化 Web 开发。Canvas 包装器提供了类似 HTML5 Canvas2D API 的接口，让你可以轻松地在浏览器中使用 WCN。

### 基础模板（使用 Canvas 包装器）

```
<!DOCTYPE html>
<html>
<head>
    <title>My WCN App</title>
</head>
<body>
    <canvas id="canvas" width="800" height="600"></canvas>

    <!-- Load WCN WASM module -->
    <script src="../../build-wasm/wcn.js"></script>

    <!-- Load WCN Canvas Wrapper -->
    <script src="./wcn_canvas.js"></script>

    <script>
        (async function () {
            // 创建 WCN 模块
            const WCN = await createWCNModule();

            // 创建 Canvas 包装器
            const wcnCanvas = await createWCNCanvas('canvas', WCN);

            // 开始绘制
            function draw() {
                // Begin frame
                wcnCanvas.beginFrame();

                // Begin render pass
                const renderPassInfo = wcnCanvas.beginRenderPass();
                if (!renderPassInfo) return;

                // 你的绘制代码
                wcnCanvas.setFillStyle('red');
                wcnCanvas.fillRect(50, 50, 100, 100);

                wcnCanvas.setFont('16px Arial');
                wcnCanvas.setFillStyle('black');
                wcnCanvas.fillText('Hello WCN!', 200, 100);

                // End render pass
                wcnCanvas.endRenderPass(renderPassInfo.textureViewId);

                // End frame and submit
                wcnCanvas.endFrame();
                wcnCanvas.submitCommands();

                requestAnimationFrame(draw);
            }

            draw();
        })();
    </script>
</body>
</html>
```

### Canvas 包装器 API

Canvas 包装器提供了以下主要方法：

- **基本绘制**: `fillRect()`, `strokeRect()`, `clearRect()`
- **路径操作**: `beginPath()`, `moveTo()`, `lineTo()`, `arc()`, `closePath()`, `fill()`, `stroke()`
- **样式设置**: `setFillStyle()`, `setStrokeStyle()`, `setLineWidth()`, `setLineCap()`, `setLineJoin()`, `setMiterLimit()`
- **变换操作**: `translate()`, `rotate()`, `scale()`, `transform()`, `setTransform()`, `resetTransform()`, `save()`, `restore()`
- **文本操作**: `setFont()`, `fillText()`, `strokeText()`, `measureText()`
- **帧管理**: `beginFrame()`, `endFrame()`, `beginRenderPass()`, `endRenderPass()`, `submitCommands()`

### ⚠️ 注意

由于上述限制，以下代码模板**目前无法在浏览器中工作**。这些是未来支持的目标 API。

### 基础模板（未来）

```
<!DOCTYPE html>
<html>
<head>
    <title>My WCN App</title>
</head>
<body>
    <canvas id="canvas" width="800" height="600"></canvas>
    
    <script src="../../build-wasm/wcn.js"></script>
    <script>
        async function init() {
            // 加载 WCN 模块
            const WCN = await createWCNModule();
            
            // 初始化 WebGPU
            const adapter = await navigator.gpu.requestAdapter();
            const device = await adapter.requestDevice();
            
            // 创建 WCN 上下文（当前会失败）
            const gpuResources = WCN._wcn_wasm_create_gpu_resources(
                0, device, device.queue, 0
            );
            const ctx = WCN._wcn_create_context(gpuResources);
            
            // 开始绘制
            const canvas = document.getElementById('canvas');
            WCN._wcn_begin_frame(ctx, canvas.width, canvas.height, 0);
            
            // 你的绘制代码
            WCN._wcn_set_fill_style(ctx, 0xFFFF0000);
            WCN._wcn_fill_rect(ctx, 50, 50, 100, 100);
            
            WCN._wcn_end_frame(ctx);
        }
        
        init();
    </script>
</body>
</html>
```

## 调试

### 查看控制台

打开浏览器开发者工具 (F12) 查看：
- 错误信息
- WebGPU 状态
- WASM 加载状态

### 常见问题

1. **"Failed to create WCN context"**
   - 这是预期行为（见上述限制说明）
   - 使用原生 GLFW 示例代替

2. **"WebGPU not supported"**
   - 确保使用支持 WebGPU 的浏览器
   - 检查是否启用了 WebGPU 标志

3. **"Failed to load WASM"**
   - 确保通过 HTTP 服务器访问，不是 file:// 协议
   - 检查 wcn.wasm 文件是否存在

4. **"Module not found"**
   - 检查 wcn.js 的路径是否正确
   - 确保构建成功完成

## 为开发者

如果你想添加完整的浏览器支持：

1. 将 wgpu-native 调用替换为 Emscripten WebGPU
2. 更新 `src/wcn_context.c` 以处理浏览器初始化
3. 修改 `src/wcn_wasm_exports.c` 使用 Emscripten 的 WebGPU 绑定
4. 更新 CMakeLists.txt 链接 Emscripten WebGPU

参见 Emscripten 的 WebGPU 文档：
https://emscripten.org/docs/api_reference/html5.h.html#webgpu

## 更多资源

- [WCN 文档](../../docs/WASM_BUILD.md)
- [WebGPU 规范](https://www.w3.org/TR/webgpu/)
- [Emscripten 文档](https://emscripten.org/docs/)
- [原生 GLFW 示例](../GLFW/) - 完整功能的工作示例
